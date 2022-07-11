import logging
import time
from urllib.error import HTTPError

from felfinder import celery
from felfinder.models import db, File, Folder, User, CommandHistory, CommandFiles
from felfinder.utils import create_service_object, get_rand_id, get_hash, file_ret, folder_ret
from felfinder.utils import get_fobj_from_uid_id
from felfinder.config import ROOT_ID, ROOT_PHASH, ROOT_PATH, ROOT_NAME

def replay_table_entry(qual_param):

    """
    To render the table of commands on the page for replay, we have
    to convert each set of parameters to a sensible entry
    """

    cmd = qual_param["cmd"]

    if cmd == "paste":
        return ["MOVE", qual_param['target_name'],
                "TO", qual_param['dst_path'], qual_param['spec_id'], cmd,
                qual_param['cmd_id']]
    elif cmd == "duplicate":
        return ["COPY", qual_param['target_name'],
                "TO", qual_param['dst_path'], qual_param['spec_id'], cmd,
                qual_param['cmd_id']]
    elif cmd == "rename":
        return ["RENAME", qual_param['old_name'],
                "TO", qual_param['new_name'], qual_param['spec_id'], cmd,
                qual_param['cmd_id']]
    elif cmd == "mkdir":
        return ["MAKE FOLDER", qual_param['new_name'],
                "", "", qual_param['spec_id'], cmd, qual_param['cmd_id']]
    if cmd == "rm":
        return ['MOVE TO TRASH', qual_param['target_name'],
                "", "", qual_param['spec_id'], cmd, qual_param['cmd_id']]

def get_replay_id(cmdh_obj):

    """
    If we created the object in our interface, we assigned it a temporary ID
    that doesn't correspond to any Google ID. To perform replay, we had to
    save the ID that Google eventually gives that folder once we create
    it in the replay. Basically, we return the ID that corresponds to the
    items's Google ID, which may not be the one in its 'id' field
    """

    fobj_id = cmdh_obj.file_id or cmdh_obj.folder_id
    f_obj = get_fobj_from_uid_id(cmdh_obj.user_id, fobj_id)
    gdrive_id = f_obj.assigned_id or f_obj.id

    return gdrive_id

def log_replay_error(cmd, target_id = None, source_id = None):
    cmd_part = "Error on replay with cmd {}".format(cmd.rand_id)
    target_part = ", target {}".format(target_id)
    source_part = ", and source {}".format(source_id)

    out_str = [x if x else "" for x in (cmd_part, target_part, source_part)]

    logging.error("".join(out_str))

def command_list(uid, temp = False, time_before = None):

    commands = CommandHistory.query.filter(CommandHistory.user_id==uid,
                                           CommandHistory.cmd != "find",
                                           CommandHistory.cmd != "open")

    if temp:
        commands = commands.filter(CommandHistory.time_run <= time_before)
    else:
        commands = commands.filter(CommandHistory.replayed==False,
                                   CommandHistory.sel_for_replay==True)

    commands = commands.order_by(CommandHistory.time_run).all()

    return commands



def replay_paste(drive_obj, cmd, duplicate = False):

    """
    We're assuming that the target object already exists in the Google
    Drive because we've ordered by timestamp.

    This is trickier because we have to do the same thing for files
    as we do when we create a directory, because we assign copied
    files a random ID.
    """

    rel_f_objs = CommandFiles.query.filter_by(cmd_id = cmd.rand_id,
                                           user_id = cmd.user_id).\
                                           order_by(CommandFiles.recurse_order,
                                                    CommandFiles.folder_id.desc())
    dst_cmdfs = rel_f_objs.filter_by(target = False).one()
    obj_cmdfs = rel_f_objs.filter_by(target = True).all()

    target_gdrive_id = get_replay_id(dst_cmdfs)

    for obj in obj_cmdfs:

        gdrive_id = get_replay_id(obj)

        try:
            drive_file = drive_obj.files().get(fileId=gdrive_id,
                                               fields='parents').execute()
            prev_parents = ",".join(drive_file.get('parents'))

            if not duplicate:

                up_file = drive_obj.files().update(fileId = gdrive_id,
                                                   addParents = target_gdrive_id,
                                                   removeParents = prev_parents,
                                                   fields = 'id, parents').execute()
            else:

                f_obj = get_fobj_from_uid_id(cmd.user_id,
                                             obj.file_id or obj.folder_id)

                body = {'title': f_obj.name}

                new_file_id = drive_obj.files().copy(fileId = f_obj.id,
                                                  body = body,
                                                  fields = 'id').execute()

                copy_child_fobj = get_fobj_from_uid_id(cmd.user_id, obj.copy_child)
                copy_child_fobj.assigned_id = new_file_id.get('id')

                drive_obj.files().update(fileId = f_obj.id,
                                         addParents = target_gdrive_id,
                                         removeParents = prev_parents).execute()

        except HTTPError:
            log_replay_error(cmd, target_gdrive_id, gdrive_id)
            return False

    return True

def replay_duplicate(drive_obj, cmd):
    replay_paste(drive_obj, cmd, duplicate = True)

def create_trashed_folder(drive_obj, cmd):

    """
    One-off function to create a folder where we trashed all
    their files
    """

    name = "Trashed By Prolific Study"

    f_metadata = {'name': name,
                  'mimeType': 'application/vnd.google-apps.folder'}

    trash_id = drive_obj.files().create(body=f_metadata,
                                        fields='id').execute()
    trash_id = trash_id.get('id')

    curr_time = int(time.time())

    common_attrs = {'rand_id': get_rand_id(),
                    'user_id': cmd.user_id,
                    'id' : trash_id,
                    'name': name,
                    'path': ROOT_PATH + "/" + name,
                    'path_hash': get_hash(trash_id),
                    'parent_id': ROOT_ID,
                    'parent_hash': ROOT_PHASH,
                    'size': 0,
                    'last_modified': curr_time,
                    'created_by_study': True,
                    'created_time': curr_time,
                    'is_shared': False,
                    'is_owner': True,
                    'error': False}

    f = Folder(**common_attrs)
    db.session.add(f)

    return f


def replay_rm(drive_obj, cmd):

    name = "Trashed By Prolific Study"

    trash_fold = Folder.query.filter_by(user_id = cmd.user_id, name = name).one_or_none()

    if trash_fold is None:
        trash_fold = create_trashed_folder(drive_obj, cmd)

    # Because this is a full move and not a duplicate, we only need
    # to look at the items at the first level. All the items below
    # these will keep the same parent ID.

    rel_f_objs = CommandFiles.query.filter_by(cmd_id = cmd.rand_id,
                                              user_id = cmd.user_id,
                                              recurse_order = 0)
    obj_cmdfs = rel_f_objs.filter_by(target = True).all()

    for obj in obj_cmdfs:

        gdrive_id = get_replay_id(obj)

        try:
            drive_file = drive_obj.files().get(fileId=gdrive_id,
                                               fields='parents').execute()
            prev_parents = ",".join(drive_file.get('parents'))

            up_file = drive_obj.files().update(fileId = gdrive_id,
                                               addParents = trash_fold.id,
                                               removeParents = prev_parents,
                                               fields = 'id, parents').execute()

        except HTTPError:
            log_replay_error(cmd, gdrive_id)
            return False

    return True

def replay_rename(drive_obj, cmd):

    to_rename = CommandFiles.query.filter_by(cmd_id = cmd.rand_id,
                                           user_id = cmd.user_id).one()
    rename_id = get_replay_id(to_rename)
    body = {'name': cmd.new_name}

    try:
        up_file = drive_obj.files().update(fileId = rename_id,
                                          body = body,
                                          fields = 'name').execute()

    except HTTPError:
        log_replay_error(cmd, rename_id)
        return False

    return True

def replay_mkdir(drive_obj, cmd):

    added_fold = CommandFiles.query.filter_by(cmd_id = cmd.rand_id,
                                              user_id = cmd.user_id).one()

    fold_obj = Folder.query.filter_by(user_id = cmd.user_id,
                                      id = added_fold.folder_id).one()
    f_metadata = {'name': cmd.new_name,
                  'mimeType': 'application/vnd.google-apps.folder'}
    try:

        updated_id = drive_obj.files().create(body=f_metadata,
                                              fields='id').execute()
        updated_id = updated_id.get('id')

        if fold_obj.parent_id != ROOT_ID:

            drive_file = drive_obj.files().get(fileId=updated_id,
                                               fields='parents').execute()
            prev_parents = ",".join(drive_file.get('parents'))

            f_obj = get_fobj_from_uid_id(cmd.user_id, fold_obj.parent_id)
            parent_gdrive_id = f_obj.assigned_id or f_obj.id

            drive_obj.files().update(fileId = updated_id,
                                     addParents = parent_gdrive_id,
                                     removeParents = prev_parents).execute()

    except HTTPError:
        log_replay_error(cmd, added_fold.folder_id)
        return False

    fold_obj.assigned_id = updated_id
    return True

def replay_single(drive_obj, cmd):

    """
    Replay a single command in the user's actual Google Drive
    to mirror changes, as requested

    Arguments:
      cmd (models.CommandHistory): DB object for command we've executed in
                                   sandbox, but not in participant's drive

    Returns:
      (bool) Indicating success of operation

    """

    replay_dispatch = {'paste': replay_paste,
                       'duplicate': replay_duplicate,
                       'rm': replay_rm,
                       'rename': replay_rename,
                       'mkdir': replay_mkdir}

    return replay_dispatch[cmd.cmd](drive_obj, cmd)


def temp_replay_paste(cmd, duplicate = False):

    if duplicate:
        # Our filter operations in the open function for the connector
        # will take care of this
        return True

    rel_f_objs = CommandFiles.query.filter_by(cmd_id = cmd.rand_id,
                                           user_id = cmd.user_id).\
                                           order_by(CommandFiles.recurse_order)
    dst_cmdfs = rel_f_objs.filter_by(target = False).one()
    obj_cmdfs = rel_f_objs.filter_by(target = True).all()

    dst_obj = Folder.query.filter_by(user_id = cmd.user_id,
                                     id = dst_cmdfs.folder_id).one()

    for obj in obj_cmdfs:

        if obj.file_id is not None:
            o = File.query.filter_by(user_id = cmd.user_id,
                                     id = obj.file_id).one()
        else:
            o = Folder.query.filter_by(user_id = cmd.user_id,
                                       id = obj.folder_id).one()
        o.parent_id = dst_obj.id
        o.parent_hash = dst_obj.path_hash

    return True

def temp_replay_duplicate(cmd):
    temp_replay_paste(cmd, duplicate = True)

def temp_replay_rm(cmd):

    rel_f_objs = CommandFiles.query.filter_by(cmd_id = cmd.rand_id,
                                              user_id = cmd.user_id,
                                              recurse_order = 0)
    obj_cmdfs = rel_f_objs.filter_by(target = True).all()

    for obj in obj_cmdfs:

        if obj.file_id is not None:
            o = File.query.filter_by(user_id = cmd.user_id,
                                     id = obj.file_id).one()
        else:
            o = Folder.query.filter_by(user_id = cmd.user_id,
                                       id = obj.folder_id).one()

        o.trashed = True

    return True

def temp_replay_rename(cmd):

    to_rename = CommandFiles.query.filter_by(cmd_id = cmd.rand_id,
                                           user_id = cmd.user_id).one()

    if to_rename.file_id is not None:
        o = File.query.filter_by(user_id = cmd.user_id,
                                 id = to_rename.file_id).one()
    else:
        o = Folder.query.filter_by(user_id = cmd.user_id,
                                   id = to_rename.folder_id).one()

    o.name = cmd.new_name

    return True

def temp_replay_mkdir(cmd):
    return True


def temp_replay_single(cmd):

    replay_dispatch = {'paste': temp_replay_paste,
                       'duplicate': temp_replay_duplicate,
                       'rm': temp_replay_rm,
                       'rename': temp_replay_rename,
                       'mkdir': temp_replay_mkdir}

    return replay_dispatch[cmd.cmd](cmd)

def temp_replay_all(uid, cmd):

    """
    Rewind file hierarchy state to where it was at the time cmd_id's cmd
    was executed
    """

    #command = CommandHistory.query.filter_by(user_id = uid, rand_id = cmd_id).one()
    time_before = cmd.time_run
    commands = command_list(cmd.user_id, temp = True, time_before = time_before)

    for c in commands:
        temp_replay_single(c)
