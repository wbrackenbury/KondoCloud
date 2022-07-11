import datetime

from felfinder.config import ROOT_ID
from felfinder.models import DriveHistory, DriveFile, NAME_PATCH_LEN
from felfinder.utils import get_rand_id, trunc_str


def ingest_action(user_id, act):

    """
    Returns a set of formatted database objects
    for each action.

    Used documentation from https://developers.google.com/drive/activity/v2/reference/rest/v2/activity/driveactivity#DriveActivity
    Retrieved 9/23/2020

    user_id (str): ID of user we're requesting
    act (dict): return from REST service consisting
                of a single DriveActivity object
    """

    # TODO: IF user_id doesn't match the actor, what then?
    # TODO: IF we're creating a drive itself, pass on it

    hist = DriveHistory(rand_id = get_rand_id(),
                        user_id = user_id)

    action = act['primaryActionDetail']
    act_type = get_sole_key(action)

    hist.cmd = act_type
    hist.cmd_subtype = parse_subtype(act)
    hist.timestamp = parse_time(act)
    hist.old_name, hist.new_name = parse_names(act)
    hist.copy = parse_copy(act)

    parsed_files = parse_files(act, hist)

    return [hist] + parsed_files

def parse_subtype(act):

    """
    Some action types, such as 'create', have subtypes, like
    'upload' or 'new', and we want to capture those
    """

    action = act['primaryActionDetail']
    act_type = get_sole_key(action)

    if act_type == 'create':
        return get_sole_key(action[act_type])

    if act_type == 'comment':
        return get_union_type(action[act_type],
                              ['post', 'assignment', 'suggestion'])

    return None

def parse_time(act):

    """
    Get a distinct timestamp for an activity object.

    If we have a time range, we take the end of the
    time range.
    """

    time_type = get_union_type(act, ['timestamp', 'timeRange'])
    time_item = act[time_type]
    if time_type == 'timeRange':
        time_item = time_item['endTime']

    # Two formats for time data
    try:
        time_run = datetime.datetime.strptime(time_item, '%Y-%m-%dT%H:%M:%S.%fZ')
    except ValueError:
        time_run = datetime.datetime.strptime(time_item, '%Y-%m-%dT%H:%M:%SZ')
    time_run = int(time_run.timestamp())

    return time_run


def parse_names(act):

    """
    Get the old and new names, if applicable, for the action

    """

    action = act['primaryActionDetail']

    act_type = get_sole_key(action)

    if act_type == 'rename':
        old_name = trunc_str(action[act_type]['oldTitle'], NAME_PATCH_LEN)
        new_name = trunc_str(action[act_type]['newTitle'], NAME_PATCH_LEN)
        return old_name, new_name

    # if act_type == 'create':
    #     create_type = get_sole_key(action[act_type])
    #     old_name = None

    #     if create_type == 'new' or create_type == 'upload':
    #         created_item = act['targets'][0]
    #     elif create_type == 'copy':
    #         created_item = action[act_type][create_type]['originalObject']

    #     created_item_type = get_union_type(created_item,
    #                                        ['driveItem', 'drive',
    #                                         'fileComment'])
    #     if created_item_type == 'driveItem':
    #         new_name = created_item[created_item_type]['title']

    #     return old_name, new_name

    return None, None


def parse_copy(act):

    """
    Determine if the action is copying a file or not
    """

    action = act['primaryActionDetail']
    act_type = get_sole_key(action)

    if act_type == 'create':
        create_type = get_sole_key(action[act_type])

        if create_type == 'copy':
            return True
    return False


def parse_files(act, hist):

    """
    Create DriveFile objects

    act (dict): return from REST service consisting
                of a single DriveActivity object
    hist (DriveHistory object): command targeting these files
    """

    action = act['primaryActionDetail']
    act_type = get_sole_key(action)

    act_parses = {'create': parse_files_create,
                  'move': parse_files_move}

    return act_parses.get(act_type, parse_files_from_targets)(act, hist)


def parse_files_create(act, hist):

    """
    Return DriveFile objects to log in case the command
    to log is 'create'
    """

    action = act['primaryActionDetail']
    act_type = get_sole_key(action)

    create_type = get_sole_key(action[act_type])

    orig_file = None

    if create_type == 'copy':
        orig_item = action[act_type][create_type]['originalObject']

        orig_item_type = get_union_type(orig_item,
                                           ['driveItem', 'drive',
                                            'fileComment'])
        if orig_item_type == 'driveItem':
            orig_file = parse_file_drive_item(orig_item[orig_item_type], hist)

    other_files = parse_files_from_targets(act, hist)

    for f in other_files:
        if orig_file:
            orig_file.copy_child = f.file_id or f.folder_id # TODO: Check this

    if orig_file:
        return [orig_file] + other_files
    else:
        return other_files


def parse_files_move(act, hist):

    """
    Return DriveFile objects to log in case the command
    to log is 'move'
    """

    action = act['primaryActionDetail']
    act_type = get_sole_key(action)

    move_cmd = action[act_type]

    old_parent_id = None
    add_parent_id = None

    if 'removedParents' in move_cmd:
        first_del_parent = move_cmd['removedParents'][0]
        is_driveitem_del = get_union_type(first_del_parent, ["driveItem", "drive"])
        if is_driveitem_del == "driveItem":
            old_parent_id = get_drive_item_fold_id(first_del_parent[is_driveitem_del])

    if 'addedParents' in move_cmd:
        first_add_parent = move_cmd['addedParents'][0]
        is_driveitem_add = get_union_type(first_add_parent, ["driveItem", "drive"])
        if is_driveitem_add == "driveItem":
            add_parent_id = get_drive_item_fold_id(first_add_parent[is_driveitem_add])

    new_parent = DriveFile(rand_id = get_rand_id(),
                           user_id = hist.user_id,
                           cmd_id = hist.rand_id,
                           target = False,
                           folder_id = add_parent_id)

    moved_files = parse_files_from_targets(act, hist)
    for f in moved_files:
        f.old_parent_id = old_parent_id

    return [new_parent] + moved_files


def parse_files_from_targets(act, hist):

    """
    If there are no additional file id's to parse from
    the action, we just parse files from the targets field
    """

    files = []
    for item in act['targets']:

        target_type = get_union_type(item, ['driveItem', 'drive', 'fileComment'])
        if target_type == 'driveItem':
            files.append(parse_file_drive_item(item[target_type], hist))

    return files


def parse_file_drive_item(drive_item, hist):

    """
    If item is a DriveItem, return
    a DriveFile object from this

    drive_item (dict): has DriveItem format from https://developers.google.com/drive/activity/v2/reference/rest/v2/activity/driveitem#DriveItem

    """

    drive_item_type = get_union_type(drive_item, ['driveFile', 'driveFolder'])
    created_file_id = get_drive_item_fold_id(drive_item)

    created_file = DriveFile(rand_id = get_rand_id(),
                             user_id = hist.user_id,
                             cmd_id = hist.rand_id,
                             target = True,
                             old_parent_id = None)


    if drive_item_type is None:
        created_file.file_id = created_file_id
        created_file.folder_id = created_file_id
    elif drive_item_type == 'driveFile':
        created_file.file_id = created_file_id
    else:
        created_file.folder_id = created_file_id

    return created_file


def get_drive_item_fold_id(drive_fold):

    """
    Because we might have a different folder ID
    for the root folder, we want to return the
    ROOT_ID in such cases

    TODO: Ensure this aligns with how
    we fixed file paths later
    """

    poss_id = get_id(drive_fold['name'])

    is_fold = get_union_type(drive_fold, ['driveFile', 'driveFolder'])
    if not is_fold == 'driveFolder':
        return poss_id

    fold_type = drive_fold[is_fold]['type']
    if fold_type == 'MY_DRIVE_ROOT':
        return ROOT_ID
    else:
        return poss_id


def get_union_type(container, options):

    for f in options:
        if f in container:
            return f


def get_sole_key(d):
    return next(iter(d.keys()), None)


def get_id(s):
    """ names in API are encoded as COLLECTION_ID/ITEM_ID """
    return s.split("/")[1]
