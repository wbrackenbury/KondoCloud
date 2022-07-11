'''
Comments are taken from https://github.com/Studio-42/elFinder/wiki/Client-Server-API-2.1
on 11/11/2019
'''

import copy
import time
import logging
import pprint

from flask import session
from sqlalchemy import and_, or_, inspect
from sqlalchemy.orm.session import make_transient
from elasticsearch import Elasticsearch
from urllib.parse import urlencode, quote_plus, urlparse, parse_qs

from felfinder.models import db, User, File, Folder, CommandHistory, CommandFiles
from felfinder.models import Recommends, VALID_REC_CONDS, NAME_PATCH_LEN
from felfinder.utils import file_ret, folder_ret, get_hash, get_rand_id
from felfinder.utils import copy_rename, is_file_object, row_conv
from felfinder.utils import elastic_term_query, elastic_wildcard_query
from felfinder.utils import elastic_range_query, get_fobj_from_uid_id
from felfinder.utils import partial_folder_ret, partial_file_ret, trunc_str
from felfinder.workers import has_subdirs, cwd_and_subfolders, recommend_preds
from felfinder.recommend import retrieve_recs
from felfinder.replay import temp_replay_all
from felfinder.config import ROOT_ID, ROOT_PHASH, ROOT_PATH, ROOT_NAME
from felfinder.config import (REC_ACT_DECAY, TEST_SERVER, THRESH,
                              MAX_RATE_CHANGE, MIN_THRESH, MAX_THRESH,
                              DEC_EXPONENT)

# We shouldn't try and perform recommendations for more than this
# number at a time
MAX_REC_TARGETS = 20

class ElFinderConnector():
    _version = 2.1

    def __init__(self, state = None):

        # ID of a point in the command history log to return results from
        self.state = state

        self.state_cmd = None
        if self.state:
            uid = session["user_id"]
            self.state_cmd = CommandHistory.query.filter_by(user_id = uid,
                                                            rand_id = state).one()
            self.time_cutoff = self.state_cmd.time_run

        self.dispatch_table = {'abort': self.abort,
                               'archive': self.archive,
                               'callback': self.callback,
                               'chmod': self.chmod,
                               'dim': self.dim,
                               'duplicate': self.duplicate,
                               'editor': self.editor,
                               'extract': self.extract,
                               'file': self.file,
                               'find': self.find,
                               'get': self.get,
                               'getfile': self.getfile,
                               'info': self.info,
                               'ls': self.ls,
                               'mkdir': self.mkdir,
                               'mkdfile': self.mkdfile,
                               'netmount': self.netmount,
                               'open': self.open,
                               'parents': self.parents,
                               'paste': self.paste,
                               'ping': self.ping,
                               'put': self.put,
                               'rename': self.rename,
                               'resize': self.resize,
                               'rm': self.rm,
                               'search': self.search,
                               'size': self.size,
                               'tmb': self.tmb,
                               'tree': self.tree,
                               'update_recs': self.update_rec_statuses,
                               'upload': self.upload,
                               'url': self.url,
                               'zipdl': self.zipdl}

    def _rename_list_args(self, mut_data):
        """
        Rename arguments with '[]' at the end
        to remove that suffix

        mut_data(dict): Arguments as parsed by parse_qs
        """

        to_del = []
        rename_pairs = []

        for k in mut_data:
            ind = k.find('[')
            is_listarg = ind != -1
            if is_listarg:
                rename = k[:ind]
                rename_pairs.append((rename, mut_data[k]))
                to_del.append(k)

        for k in to_del:
            del mut_data[k]

        for k, v in rename_pairs:
            mut_data[k] = v

        return mut_data

    def _unlist_params(self, mut_data):

        """
        Rename arguments with '[]' at the end
        to remove that suffix

        mut_data(dict): Arguments as parsed by parse_qs, after renaming
                        arguments

        """

        keep_list = ['targets']

        for k,v in mut_data.items():
            if not len(v):
                mut_data[k] = None
            if len(v) == 1 and k not in keep_list:
                mut_data[k] = v[0]

        return mut_data

    def _validate_return(self, f_objs):

        """
        If there is a valid state on the object, any returns on file or
        folder objects should be locked so the user can't make any changes
        """

        if not self.state:
            return f_objs

        ret = [f for f in f_objs if f['created_time'] <= self.time_cutoff]

        for f in ret:
            f['read'] = 1
            f['write'] = 0
            f['locked'] = 1

        return ret

    def _engage_state(self):

        if not self.state:
            return {}

        uid = session["user_id"]

        all_files = File.query.filter_by(user_id = uid).all()
        all_folds = Folder.query.filter_by(user_id = uid).all()

        stored_state = {}

        for f in all_files + all_folds:

            if f.name == 'root':
                continue

            stored_state[f.rand_id] = {'name': f.name,
                                       'parent_id': f.parent_id,
                                       'parent_hash': f.parent_hash,
                                       'trashed': f.trashed}

            f.name = f.original_name
            f.parent_id = f.original_parent_id
            f.parent_hash = f.original_parent_hash
            f.trashed = False

        db.session.commit()

        temp_replay_all(uid, self.state_cmd)

        db.session.commit()

        return stored_state


    def _disengage_state(self, stored_state):

        if not self.state:
            return

        uid = session["user_id"]

        all_files = File.query.filter_by(user_id = uid).\
            with_for_update(of=File).all()
        all_folds = Folder.query.filter_by(user_id = uid).\
            with_for_update(of=Folder).all()

        for f in all_files + all_folds:

            if f.name == 'root':
                continue

            state_dict = stored_state[f.rand_id]
            f.name = state_dict['name']
            f.parent_id = state_dict['parent_id']
            f.parent_hash = state_dict['parent_hash']
            f.trashed = state_dict['trashed']

        db.session.commit()

    # def arg_translation(self, data):

    #     """
    #     The request passes arguments as URL parameters. We need to
    #     translate that to keyword arguments for a connector method.


    #     Arguments:
    #       data (ImmutableMultiDict): URL parameter arguments from a request

    #     Returns:
    #       mut_data (dict): a dictionary of keyword args for a connector method

    #     """

    #     mut_data = {k: d for (k, d) in data.items()}

    #     # Rename list arguments
    #     mut_data = self._rename_list_args(mut_data)
    #     mut_data = self._unlist_params(mut_data)

    #     print(pprint.pformat(mut_data))

    #     cmd = mut_data.get('cmd')
    #     if not cmd:
    #         raise KeyError('No command argument')
    #     del mut_data['cmd']
    #     del mut_data['cmd_state']
    #     del mut_data['_']
    #     if 'reload' in mut_data:
    #         del mut_data['reload']
    #     if 'compare' in mut_data:
    #         del mut_data['compare']
    #     if 'hashes' in mut_data:
    #         del mut_data['hashes']

    #     return cmd, mut_data

    def _fill_list_args(self, data):

        """
        There are several arguments that are passed as individual items
        that should be rolled up to a single list argument. This is a
        helper function to roll those up
        Arguments:
          data (ImmutableMultiDict): URL parameter arguments from a request
        Returns:
          list_args (dict):    arguments with multiple URL parameters rolled up
                               to a single list argument as the value
          largs_to_del (list): arguments that have been rolled up and should
                               be deleted
        """


        list_args = {}
        largs_to_del = []
        for k in data:
            v = data.getlist(k)
            ind = k.find('[')

            if ind != -1:
                larg = k[:ind]

                if larg in list_args:
                    list_args[larg].extend(v)
                else:
                    list_args[larg] = v

                largs_to_del.append(k)
            else:
                list_args[k] = data.get(k)

        return list_args

    def arg_translation(self, data):

        """
        The request passes arguments as URL parameters. We need to
        translate that to keyword arguments for a connector method.
        Arguments:
          data (ImmutableMultiDict): URL parameter arguments from a request
        Returns:
          mut_data (dict): a dictionary of keyword args for a connector method
        """

        mut_data = self._fill_list_args(data)

        if TEST_SERVER:
            print(mut_data)
        else:
            logger = logging.getLogger('gunicorn.error')
            logger.info(mut_data)

        cmd = mut_data.get('cmd')
        if not cmd:
            raise KeyError('No command argument')

        del mut_data['cmd']
        del mut_data['cmd_state']
        #del mut_data['_']
        if 'reload' in mut_data:
            del mut_data['reload']
        if 'compare' in mut_data:
            del mut_data['compare']

        if 'hashes' in mut_data:
            del mut_data['hashes']

        return cmd, mut_data

    def dispatch(self, req):

        stored_state = self._engage_state()

        try:
            #data = parse_qs(urlparse(req.url).query)
            data = req.form
            cmd, mut_data = self.arg_translation(data)
            resp, code = self.dispatch_table[cmd](**mut_data)
        finally:
            self._disengage_state(stored_state)

        #print(pprint.pformat(resp))

        if 'start_org' not in session:
            u = User.query.filter_by(id=session['user_id']).one()
            start_org = u.start_org or 0
            session['start_org'] = start_org
            if u.start_org is None:
                u.start_org = start_org
            db.session.commit()

        mins_taken = int((time.time() - session['start_org']) / 60)

        resp['org_time'] = mins_taken

        return resp, code

    def _sub_items(self, parent):

        """
        Return a list of objects with the specified object as a parent via
        the full database object

        Args:
          parent (sqlalchemy Object): File or Folder object

        Returns:
          (tuple): pair of lists of files and folders with parent as their parent

        """

        uid = session['user_id']
        files = File.query.filter(and_(File.user_id == uid,
                                       File.parent_id == parent.id,
                                       File.trashed == False)).all()
        folds = Folder.query.filter(and_(Folder.user_id == uid,
                                         Folder.parent_id == parent.id,
                                         Folder.trashed == False)).all()

        return files, folds


    def _db_obj_targets(self, targets):

        """
        Return a list of objects corresponding to the specified targets
        """

        uid = session['user_id']
        files = File.query.filter(and_(File.user_id == uid,
                                       File.path_hash.in_(targets),
                                       File.trashed == False)).all()
        folds = Folder.query.filter(and_(Folder.user_id == uid,
                                         Folder.path_hash.in_(targets),
                                         Folder.trashed == False)).all()

        return files, folds

    def _db_obj_ids(self, ids):

        """
        Return a list of objects corresponding to the specified ids
        """

        uid = session['user_id']
        files = File.query.filter(and_(File.user_id == uid,
                                       File.id.in_(ids),
                                       File.trashed == False)).all()
        folds = Folder.query.filter(and_(Folder.user_id == uid,
                                         Folder.id.in_(ids),
                                         Folder.trashed == False)).all()

        return files, folds


    def _log_cmd_hist(self, cmd, copy = False, old_name = None,
                      new_name = None, rec_result_id = None, inv = None):

        if self.state:
            return None, None

        uid = session['user_id']

        cmd_id = get_rand_id()
        time_run = int(time.time())
        cmdh = CommandHistory(rand_id = cmd_id,
                              user_id = uid,
                              time_run = time_run,
                              old_name = old_name,
                              new_name = new_name,
                              cmd = cmd,
                              copy = copy,
                              inv = inv,
                              rec_result_id = rec_result_id)

        db.session.add(cmdh)

        if rec_result_id:
            rec = Recommends.query.filter_by(user_id = uid,
                                             rand_id = rec_result_id).one()

            if inv is not None:

                if (not inv):
                    self._update_user_thresh(uid, move=(rec.action == 'move'),
                                             dec=True)

                rec.accepted = (not inv)
                rec.to_inv = (not inv)
                rec.num_remain = REC_ACT_DECAY

            else: # delete recommendations cannot be undone
                self._update_user_thresh(uid, move=(rec.action == 'move'),
                                         dec=True)
                rec.accepted = True
                rec.to_inv = False

        db.session.commit()

        return cmd_id, time_run

    def _log_cmd_file(self, cmd_id, target = False, file_id = None,
                      folder_id = None, other_kwargs = {}):

        if self.state:
            return

        uid = session['user_id']

        cmd_dst = CommandFiles(rand_id = get_rand_id(),
                               user_id = uid,
                               cmd_id = cmd_id,
                               target = target,
                               file_id = file_id,
                               folder_id = folder_id,
                               **other_kwargs)

        db.session.add(cmd_dst)

        return cmd_dst

    def _log_cmd_f_obj(self, cmd_id, f_obj, target = False, other_kwargs = {}):

        if self.state:
            return

        if not is_file_object(f_obj):
            return self._log_cmd_file(cmd_id, target = target,
                                      folder_id = f_obj.id,
                                      other_kwargs = other_kwargs)
        else:
            return self._log_cmd_file(cmd_id, target = target,
                                      file_id = f_obj.id,
                                      other_kwargs = other_kwargs)

    def _dec_val(self, orig, n=1, exp=DEC_EXPONENT):

        for i in range(n):
            orig = orig**(1 + exp)

        return orig

    def _update_user_thresh(self, uid, move=True, dec=True):

        """
        Update the user's decision threshold, quickly at first, slowly
        later
        """

        user = User.query.filter_by(id=uid).one()

        if move:
            user.move_num_updates += 1
        else:
            user.del_num_updates += 1
        updates = user.move_num_updates if move else user.del_num_updates

        dec_val = self._dec_val(MAX_RATE_CHANGE,
                                n = updates,
                                exp = DEC_EXPONENT)

        if dec:
            if move:
                user.move_thresh -= dec_val
                user.move_thresh = max(user.move_thresh, MIN_THRESH)
            else:
                user.del_thresh -= dec_val
                user.del_thresh = max(user.del_thresh, MIN_THRESH)

        else:
            if move:
                user.move_thresh += dec_val
                user.move_thresh = min(user.move_thresh, MAX_THRESH)
            else:
                user.del_thresh += dec_val
                user.del_thresh = min(user.del_thresh, MAX_THRESH)


        db.session.commit()

    def _update_from_faded(self, uid, num_faded):

        for act, count in num_faded.items():
            for _ in range(count):
                self._update_user_thresh(uid, move=(act == 'move'), dec=False)


    def abort(self, aid):

        """
        Abort the specified request

        Arguments:

          aid : request id to abort

        Response:

          Empty content with HTTP response code "204 No Content"

        """

        return {}, 204

    def archive(self, name, mimetype, target, targets):

        """
        Packs directories / files into an archive.

        Arguments:

          name : file name of the archive to create
          mimetype : mime-type for the archive
          target : hash of the directory that are added to the archive directories / files
          targets[] : an array of hashes of the directories / files to archive

        Response:

          added : (Array) Information about File/Directory of a new archive

        """

        raise NotImplemented()

    def callback(self, node, json, bind, done):

        """
        Output callback result with JavaScript that control elFinder or HTTP redirect to
        callback URL. This is used for communication with the outside, such as OAuth of netmount.

        Arguments:

          node : elFinder node DOM id that accepted /^[a-zA-Z0-9;._-]+$/ only
          json : JSON data for output
          bind : bind command name (optional)
          done : 1 or 0 - 0: output redirect HTTP response, 1: output HTML
                 included JavaScript for IFRAME/Child Window.

        Response:

          HTTP response Location:
            [Redirect URL(Defined in the volume driver)]?node={node}&json={json}&bind={bind}&done=1*

          HTML included JavaScript for IFRAME/Child Window Must
          replace {$node}, {$bind}, {$json} for each value.

        """

        raise NotImplemented()

    def chmod(self, targets, mode):

        """
        chmod target items.

        Arguments:

          targets[] : array of hashed paths of the nodes
          mode : Numeric notation file system permissions

        Response:

          An array of successfully uploaded files if success, an error otherwise.

        changed :
          (Array) of files that were successfully done chmod. Information about File/Directory

        """

        raise NotImplemented()

    def dim(self, target, substitute):

        """
        Returns the dimensions of an image/video

        Arguments:

          target : hash path of the node
          substitute : pixel that requests substitute image (optional) - API >= 2.1030

        Response:

          dim: The dimensions of the media in the format {width}x{height} (e.g. "640x480").
          url: The URL of requested substitute image. (optional)


        """

        raise NotImplemented()

    def _move(self, item, new_parent, time_run, cut):

        """
        Helper method to properly move a new file or folder to a new location

        Arguments:
          item (sqlalchemy obj): File or Folder object in TRANSIENT state
          new_parent (sqlalchemy obj): Folder object that now contains file
          time_run (int): Time that parent command was run
          cut (bool): if True, we're moving, and we shouldn't change the IDs
        """

        fresh_id = get_rand_id()

        if not cut:

            if (new_parent.id == item.parent_id): #If we're copying in the same folder
                item.name = trunc_str(copy_rename(item.name), NAME_PATCH_LEN)

            item.rand_id = get_rand_id()
            item.id = fresh_id
            item.path_hash = get_hash(item.id)
            item.created_by_study = True
            item.created_time = time_run

        item.path = new_parent.path + '/' + item.name if new_parent.path else '/' + item.name
        item.parent_id = new_parent.id
        item.parent_hash = new_parent.path_hash

        if not cut:
            return fresh_id
        else:
            return item.id

    def _recurse_paste(self, recurse_level, item, new_parent, cmd_id, time_run, cut = True):

        """
        Helper method for duplication that actually performs the operation in
        order to avoid reuse between the duplicate and paste methods

        Arguments:
          recurse_level (int): the count of how deep we are in the recursion
                               from the original call. Needed to log the
                               command for replay.
          item (sqlalchemy obj): File or Folder object to recursively paste
          new_parent (sqlalchemy obj): Folder object that now contains file
          cmd_id (str): ID of parent command pasting these files
          time_run (int): Time that parent command was run
          cut (bool): moves if True, copies if False

        Returns:
          (list): List of sqlalchemy objects that we've modified

        """

        uid = session['user_id']

        old_phash = item.path_hash
        old_parent_id = item.parent_id
        pasted, removed = [], [old_phash]

        cmd_f = self._log_cmd_f_obj(cmd_id, item, target = True,
                                    other_kwargs = {'recurse_order': recurse_level,
                                                    'old_parent_id': old_parent_id})

        cmd_log_id = cmd_f.rand_id

        if not cut:
            db.session.expunge(item)
            make_transient(item)
        item_new_id = self._move(item, new_parent, time_run, cut)

        is_file = is_file_object(item)
        if not cut:
            cmd_f.copy_child = item_new_id

            db.session.add(item)
            db.session.commit()
            if is_file:
                item = File.query.filter_by(user_id=uid, id=item_new_id).one()
            else:
                item = Folder.query.filter_by(user_id=uid, id=item_new_id).one()
        else:
            db.session.commit()
            if is_file:
                item = File.query.filter_by(user_id=uid, id=item_new_id).one()
            else:
                item = Folder.query.filter_by(user_id=uid, id=item_new_id).one()

        if not is_file_object(item):

            #We must recurse to copy all the items below
            subfolds = Folder.query.filter_by(user_id = uid,
                                              parent_hash = old_phash,
                                              trashed = False).all()
            subfiles = File.query.filter_by(user_id = uid,
                                            parent_hash = old_phash,
                                            trashed = False).all()

            added_item = partial_folder_ret(item)
            added_item['target_cmdlog'] = cmd_log_id

            pasted.append(added_item)

            for f in subfiles + subfolds:
                sub_pasted, sub_removed = self._recurse_paste(recurse_level + 1, f,
                                                              item, cmd_id,
                                                              time_run, cut)

                pasted += sub_pasted
                removed += sub_removed

        else:
            added_item = partial_file_ret(item)
            added_item['target_cmdlog'] = cmd_log_id

            pasted.append(added_item)

        return pasted, removed

    def duplicate(self, current = None, targets = []):

        """
        Creates a copy of the directory / file. Copy name is generated as follows:
        basedir_name_filecopy+serialnumber.extension (if any)

        Arguments:

          current : hash of the directory in which to create a duplicate
          targets[] : hash of the directory / file that is being duplicated

        Response:

          added (Array) Information about File/Directory of the duplicate.

        """
        if not current:
            current = session['cwd_hash']
        return self.paste(dst = current, targets = targets, cut = '0')

    def editor(self, cmd, name, method, args):

        """


        """

        raise NotImplemented("What even is this?")

    def extract(self, target, makedir):

        """

        Unpacks an archive.

        Arguments:

          cmd : extract
          target : hash of the archive file
          makedir : "1" to extract to new directory

        Response:

          added : (Array) Information about File/Directory of extracted items

        """

        raise NotImplemented()

    def update_rec_statuses(self, rec_ids = [], status = None):

        """
        In order to track the status of recommendations, when a recommendation
        is deleted due to not being accepted quickly enough, we log it here
        as appropriate

        Arguments:
          rec_ids (Array): IDs of recommendations to update
          status (string): the change in status of recommendation

        """

        if self.state:
            return {}, 200

        uid = session['user_id']

        for rec in rec_ids:
            r = Recommends.query.filter_by(user_id = uid, rand_id = rec).one()

            if status == "faded":
                r.faded = True
            elif status == "done_no_acc":
                r.done_no_acc = True
            elif status == "deleted":
                r.deleted = True

            if r.sent:
                self._update_user_thresh(uid, move=(r.action == 'move'), dec=False)

        db.session.commit()

        return {}, 200

    def file(self, target, download, cpath):

        """

        Output file into browser. This command applies to download and preview actions.

        Arguments:

          cmd : file
          target : file's hash,
          download : Send headers to force download file instead of opening it in the browser.
          cpath (API >= 2.1.39) : Cookie path that temporary cookie set up until download starts.
                                  If this parameter is specified, the connector must set the
                                  cookie name as "elfdl" + Request ID (The value is irrelevant,
                                  maybe "1") . This cookie is deleted by the client at the start
                                  of download.


        May need to set Content-Disposition, Content-Location and Content-Transfer-Encoding.
        Content-Disposition should have 'inline' for preview action or 'attachments' for download.

        """

        raise NotImplemented()

    def find(self, target):

        """
        Logs when a user previews a file, and returns recommendations for
        other files they might like to preview

        Arguments:

          target: file's hash

        Returns:

          findrecs: Recommendations for files to view

        """

        uid = session['user_id']
        exp_cond = session['exp_cond']
        rand_recs = (exp_cond == 'rand_recs')

        t_obj = File.query.filter_by(user_id = uid, path_hash = target).one_or_none()
        if t_obj is None:
            return {}, 200

        cmd_id, _ = self._log_cmd_hist('find')
        cmd_f = self._log_cmd_file(cmd_id, target = True, file_id = t_obj.id)

        found = t_obj.path_hash

        recs = []

        db.session.commit()

        if not self.state:
            target_cmdlog = cmd_f.rand_id

            recommend_preds(exp_cond, t_obj.id, uid, 'find',
                            action_spec_opts = {'explain_name': t_obj.name,
                                                'explain_hash': found,
                                                'cmd_id': cmd_id,
                                                'target_cmdlog': target_cmdlog})

            recs, num_faded = retrieve_recs(uid,
                                            action = "find",
                                            dec_recs = False,
                                            find_recs = True,
                                            rand_recs = rand_recs)

            self._update_from_faded(uid, num_faded)



        return {'found': found,
                'findrecs': recs}, 200

    def get(self, current, target, conv):

        """

        Returns the content as String (As UTF-8)

        Arguments:

          current : hash of the directory where the file is stored
          target : hash of the file
          conv : instructions for character encoding conversion of the text file
                   1 : auto detect encoding(Return false as content in response data when failed)
                   0 : auto detect encoding(Return { "doconv" : "unknown" } as response
                       data when failed)

                   Original Character encoding : original character encoding as
                                                 specified by the user

        Response:

          content: file contents (UTF-8 String or Data URI Scheme String) or false
          encoding: (Optional) Detected original character encoding
                               (Require when converting from encoding other than UTF-8)
          doconv: (Optional) "doconv":"unknown" is returned to ask the user for the original
                              encoding if automatic conversion to UTF-8 is not possible
                              when requested with conv = 0.

        """

        #TODO: What to do about this?

        raise NotImplemented()

    def getfile(self, target):

        """
        Returns a file to cache
        """

        user = session['user_id']
        t_obj = File.query.filter_by(user_id = user, path_hash = target).one_or_none()
        return {'files': [file_ret(t_obj)]}, 200

    def info(self, targets):

        """

        Returns information about places nodes

        Arguments:

          targets[] : array of hashed paths of the nodes

        Response:

          files: (Array of data) places directories info data Information about File/Directory

        """

        #TODO: THIS THING

        pass

    def ls(self, target, intersect = []):

        """
        Return a list of item names in the target directory.

        Arguments:

          target : hash of directory,
          intersect[] : An array of the item names for presence check.

        Response:

          list : (Object) item names list with hash as key. Return only duplicate files
                          if the intersect[] is specified.

        """

        #TODO: Handle intersect

        user = session['user_id']

        cwd, folds = cwd_and_subfolders(user, tree = False, target = target)

        files = File.query.filter_by(user_id = user,
                                     parent_hash = cwd['hash'],
                                     trashed = False).all()
        file_resp = {item.path_hash: item.name for item in files}
        for item in folds:
            file_resp[item['hash']] = item['name']


        resp = {'list': file_resp}
        return resp, 200

    def mkdir(self, target, name, dirs = []):

        """
        Create a new directory.

        Arguments:

          target : hash of target directory,
          name : New directory name
          dirs[] : array of new directories path (requests at pre-flight of folder upload)

        Response:

          added : (Array) Array with a single object - a new directory.
          hashes : (Object) Object of the hash value as a key to the given path in the dirs[]

        """

        # TODO: handle dirs

        uid = session['user_id']
        parent_folder = Folder.query.filter_by(user_id = uid,
                                               path_hash = target).one()
        our_id = get_rand_id()
        phash = 'l0_' + get_hash(our_id)

        cmd_id, time_run = self._log_cmd_hist('mkdir', new_name = name)
        self._log_cmd_file(cmd_id, target = True, folder_id = our_id)

        our_path = parent_folder.path + '/' + name if parent_folder.path != '/' else '/' + name

        common_attrs = {'rand_id': get_rand_id(),
                        'user_id': uid,
                        'id' : our_id,
                        'name': name,
                        'original_name': name,
                        'path': our_path,
                        'path_hash': phash,
                        'parent_id': parent_folder.id,
                        'parent_hash': parent_folder.path_hash,
                        'original_parent_id': parent_folder.id,
                        'original_parent_hash': parent_folder.path_hash,
                        'size': 0,
                        'last_modified': time_run,
                        'is_shared': False,
                        'is_owner': True,
                        'created_by_study': True,
                        'created_time': time_run,
                        'error': False}

        f = Folder(**common_attrs)
        db.session.add(f)
        db.session.commit()

        full_f = Folder.query.filter_by(user_id = uid, id = our_id).one()

        ret_item = folder_ret(full_f)

        return {'added': [ret_item]}, 200

    def mkdfile(self, target, name):

        """
        Create a new blank file.

        Arguments:

          target : hash of target directory,
          name : New file name

        Response:

          added : (Array) Array with a single object - a new file. Information about File/Directory

        """

        raise NotImplemented()

    def netmount(self, protocol, host, path, port, user, passw, alias, options):

        """

        Mount network volume during user session.

        Arguments:

          protocol : network protocol. Now only ftp supports. Required.
          host : host name. Required.
          path : root folder path.
          port : port.
          user : user name. Required.
          pass : password. Required.
          alias : mount point name. For future usage. Now not used on client side.
          options : additional options for driver. For future usage. Now not used on client side.


        """

        raise NotImplemented()

    def open(self, target = None, tree = False, init = False, onhistory = '0'):

        """

        Returns information about requested directory and its content, optionally
        can return directory tree as files, and options for the current volume.

        Arguments:

          init : (true|false|not set), optional parameter. If true indicates that this
                 request is an initialization request and its response must include the value api
                 (number or string >= 2) and should include the options object, but will still
                 work without it. Also, this option affects the processing of parameter target
                 hash value. If init == true and target is not set or that directory doesn't
                 exist, then the data connector must return the root directory of the default
                 volume. Otherwise it must return error "File not found".
          target : (string) Hash of directory to open. Required if init == false or init is not set
          tree : (true|false), optional. If true, response must contain the top-level
                 object of other volumes.

        Response:

          api : (Float) The version number of the protocol, must be >= 2.1,
                        ATTENTION - return api ONLY for init request!
          cwd : (Object) Current Working Directory - information about the current directory.
                         Information about File/Directory
          files : (Array) array of objects - files and directories in current directory. If
                          parameter tree == true, then added to the root folder objects of other
                          volumes. The order of files is not important. Information about
                          File/Directory
          netDrivers : (Array) Network protocols list which can be mounted on the fly
                               (using netmount command). Now only ftp supported.


          uplMaxFile : (Optional) (Number) Allowed upload max number of file per request (e.g. 20)
          uplMaxSize : (Optional) (String) Allowed upload max size per request. (e.g. "32M")
          options : (Optional) (Object) Further information about the folder and its volume

        """

        #TODO: what the hell is onhistory?

        if (not ('done_init' in session) and not init) and not target:
            raise ValueError('TODO: Error handling')

        user = session['user_id']

        cwd, folds = cwd_and_subfolders(user, tree, target)
        dst_obj = Folder.query.filter_by(user_id = user, path_hash = cwd['hash']).one()

        cmd_id, _ = self._log_cmd_hist('open')
        self._log_cmd_file(cmd_id, target = True, folder_id = dst_obj.id)

        files = [file_ret(r) for r in File.query.filter_by(user_id=user,
                                                           parent_hash=cwd['hash'],
                                                           trashed=False).all()]

        files += folds
        session['cwd_hash'] = cwd['hash']

        if init:
            session['done_init'] = True

        files = self._validate_return(files)

        resp = {'cwd': cwd,
                'files': files,
                'netDrivers': [],
                'options': {}}

        if init:
            resp['api'] = self._version

        db.session.commit()

        return resp, 200


    def parents(self, target, until = None):

        """

        Returns all parent folders and their's first level (at least) subfolders and own(target)
        stat. This command is invoked when a folder is reloaded or there is no data to display the
        tree of the target folder in the client. Data provided by 'parents' command should enable
        the correct drawing of tree hierarchy directories.

        Arguments:

          target : folder's hash
          until : until to hash, getting parents is enough for that (API >= 2.1024)

        Response:

          tree : (Array) Folders list

        """

        #TODO: This needs building out


        user = session['user_id']


        t_obj = Folder.query.filter_by(user_id = user, path_hash = target).one()
        curr_folder = folder_ret(t_obj)

        par_obj = Folder.query.filter_by(user_id = user,
                                         id = t_obj.parent_id).one()
        cwd, folds = cwd_and_subfolders(user, tree = False,
                                        target = par_obj.path_hash)

        ret_folds = [curr_folder] + [cwd] + folds

        # TODO: Will this return the same directory again?

        # Return all parent objects and their subfolders until we reach the root
        while par_obj.id != ROOT_ID:
            par_obj = Folder.query.filter_by(user_id = user,
                                             id = par_obj.parent_id).one()
            cwd, folds = cwd_and_subfolders(user, tree = False, target = par_obj.path_hash)
            ret_folds += [cwd] + folds

        ret_folds = self._validate_return(ret_folds)

        return {'tree': ret_folds}, 200

    def paste(self, dst, targets, renames = [], suffix = '~',
              cut = None, inv = None, rec_id = None):

        """

        Copies or moves a directory / files

        Arguments:

          dst : hash of the directory to which the files will be copied / moved (the destination)
          targets[] : An array of hashes for the files to be copied / moved
          cut : 1 if the files are moved, missing if the files are copied
          renames[] : Filename list of rename request
          suffix : Suffixes during rename (default is "~")
          inv (SELF ADDED): Whether or not item is an undo
          rec_id (SELF ADDED): If this action is an accepted recommendation,
                               note the id of the recommendation
        Response:

          If the copy / move is successful:

            added : (Array) array of file and directory objects pasted.
                            Information about File/Directory
            removed : (Array) array of file and directory 'hashes' that were successfully deleted

        """

        #TODO: Renames + suffixes deal with
        #TODO: handle similarity rec info under moves / copies

        cut = (cut == '1')
        inv = (inv == '1')
        exp_cond = session['exp_cond']
        rand_recs = (exp_cond == 'rand_recs')

        if type(targets) != type([]):
            targets = [targets]

        uid = session['user_id']
        dst_obj = Folder.query.filter_by(user_id = uid, path_hash = dst).one()
        dst_obj_path_hash = dst_obj.path_hash

        cmd_name = 'paste' if cut else 'duplicate'

        cmd_id, time_run = self._log_cmd_hist(cmd_name,
                                              copy = (not cut),
                                              rec_result_id = rec_id,
                                              inv = inv)
        cmd_dst_obj = self._log_cmd_file(cmd_id,
                                         target = False,
                                         folder_id = dst_obj.id)

        cmd_dst_id = cmd_dst_obj.rand_id

        files_to_move, folds_to_move = self._db_obj_targets(targets)

        added, removed = [], []
        for f in files_to_move + folds_to_move:
            sub_added, sub_removed = self._recurse_paste(0, f, dst_obj, cmd_id, time_run, cut)
            added += sub_added
            removed += sub_removed

        db.session.commit()

        if not inv and cut:
            for i, f_obj in enumerate(added):
                if f_obj['mime'] != 'directory':

                    rel_rec = Recommends.query.filter_by(user_id = uid,
                                                         rec_path_hash = f_obj['hash'],
                                                         is_rand = rand_recs,
                                                         accepted = False,
                                                         **VALID_REC_CONDS).all()

                    for rc in rel_rec:
                        if rc.dst_hash == dst_obj_path_hash and rec_id is None:
                            rc.done_no_acc = True
                        elif rc.dst_hash == dst_obj_path_hash:
                            pass # Shouldn't be an issue, since rec is accepted
                        else:
                            rc.invalidated = True

                    act_spec_opts = {'explain_name': f_obj['name'],
                                     'explain_hash': f_obj['hash'],
                                     'cmd_id': cmd_id,
                                     'dst_cmdlog': cmd_dst_id,
                                     'target_cmdlog': f_obj['target_cmdlog'],
                                     'dst': dst}

                    db.session.commit()

                    if f_obj.get('created_by_study', False):
                        continue

                    if i >= MAX_REC_TARGETS:
                        continue

                    if i > 3:
                        recommend_preds.delay(exp_cond, f_obj['id'], uid, 'move', act_spec_opts)
                    else:
                        recommend_preds(exp_cond, f_obj['id'], uid, 'move', act_spec_opts)

        recs, num_faded = retrieve_recs(uid,
                                        action = "move",
                                        dec_recs = (not inv) or (rec_id is not None),
                                        find_recs = False,
                                        rand_recs = rand_recs)

        self._update_from_faded(uid, num_faded)


        db.session.commit()

        fix_added = []
        for item in added:
            if 'id' in item:
                del item['id']
            fix_added.append(item)

        db.session.commit()

        dict_resp = {'added': fix_added,
                     'removed': removed if cut else [],
                     'recs': recs}

        return dict_resp, 200


    def ping(self):

        """
        Not currently used
        """

        raise NotImplementedError()

    def put(self, target, content, encoding):

        """

        Stores contents data in a file.

        Arguments:

          target : hash of the file
          content : new contents of the file
          encoding : character encoding at the time of saving (Text data will be sent by UTF-8)
                     or "scheme" for URL of contents or Data URI scheme

        content of file data other than text file is sent as string data of Data URI Scheme or
        URL of new contents with param encoding=scheme.

        Response:

          changed : (Array) of files that were successfully uploaded.

        """

        pass

    def rename(self, target, name):

        """

        Renaming a directory/file

        Arguments:

          cmd : rename
          target : hash directory/file renaming
          name : New name of the directory/file

        Response:

          added : (Array) array of file and directory objects renamed.
                          Information about File/Directory
          removed : (Array) array of file and directory 'hashes' that were successfully remoevd

        """

        uid = session['user_id']

        name = trunc_str(name, NAME_PATCH_LEN)

        file_target = File.query.filter_by(user_id = uid, path_hash = target).one_or_none()
        folder_target = Folder.query.filter_by(user_id = uid, path_hash = target).one_or_none()

        f_target = file_target or folder_target

        cmd_id, _ = self._log_cmd_hist('rename', old_name = f_target.name,
                                    new_name = name)
        self._log_cmd_f_obj(cmd_id, f_target, target = True)

        mod_path = f_target.path.split('/')
        prefix_path = '/'.join(mod_path[:-1])
        #parent_path = '/' + prefix_path if prefix_path != '/' else prefix_path
        parent_path = prefix_path
        f_target.path = parent_path + "/" + name
        f_target.name = name

        added = [file_ret(f_target)] if is_file_object(f_target) else [folder_ret(f_target)]
        removed = [f['hash'] for f in added]

        db.session.commit()

        return {'added': added, 'removed': removed}, 200

    def resize(self, mode, target, width, height, x, y, degree, quality):

        """

        Change the size of an image.

        Arguments:

          mode : 'resize' or 'crop' or 'rotate'
          target : hash of the image path
          width : new image width
          height : new image height
          x : x of crop (mode='crop')
          y : y of crop (mode='crop')
          degree : rotate degree (mode='rotate')
          quality: (unknown)

        Response:

          changed : (Array) of files that were successfully resized.

        """

        pass

    def _recurse_rm(self, recurse_level, objs_to_del, cmd_id):

        """
        Helper method to recurse easily on objects to be deleted
        """

        deleted = []

        for f in objs_to_del:
            cmd_f = self._log_cmd_f_obj(cmd_id, f, target = True,
                                        other_kwargs = {'recurse_order': recurse_level})

            cmd_log_id = cmd_f.rand_id

            if is_file_object(f):
                deleted_item = partial_file_ret(f)
            else:
                deleted_item = partial_folder_ret(f)

            deleted_item['target_cmdlog'] = cmd_log_id
            deleted.append(deleted_item)

            if not is_file_object(f):

                sub_files, sub_folds = self._sub_items(f)
                deleted.extend(self._recurse_rm(recurse_level + 1,
                                                sub_files + sub_folds, cmd_id))
            f.trashed = True

            db.session.commit()

        return deleted

    def rm(self, targets, rec_id = None):

        """

        Recursively removes files and directories.

        Arguments:

          targets[] : (Array) array of file and directory hashes to delete
          rec_id (SELF ADDED): If this action is an accepted recommendation,
                               note the id of the recommendation

        Response:

          removed : (Array) array of file and directory 'hashes' that were successfully deleted

        """

        if ROOT_PHASH in targets:
            targets = [t for t in targets if t != ROOT_PHASH]

        uid = session['user_id']
        exp_cond = session['exp_cond']
        rand_recs = (exp_cond == 'rand_recs')
        cmd_id, _ = self._log_cmd_hist('rm', rec_result_id = rec_id)

        files_to_del, folds_to_del = self._db_obj_targets(targets)

        removed = self._recurse_rm(0, files_to_del + folds_to_del, cmd_id)

        rem_hashes = [r['hash'] for r in removed]

        recs = []

        db.session.commit()

        for i, f_obj in enumerate(removed):
            if f_obj['mime'] != 'directory':

                rel_rec = Recommends.query.filter_by(user_id = uid,
                                                     rec_path_hash = f_obj['hash'],
                                                     is_rand = rand_recs,
                                                     accepted = False,
                                                     **VALID_REC_CONDS).all()

                if rel_rec:
                    for rc in rel_rec:
                        if rc.action == 'del' and rec_id is None:
                            rc.done_no_acc = True
                        elif rc.action != 'del':
                            rc.invalidated = True

                act_spec_opts = {'explain_name': f_obj['name'],
                                 'explain_hash': f_obj['hash'],
                                 'cmd_id': cmd_id,
                                 'target_cmdlog': f_obj['target_cmdlog']}

                db.session.commit()

                # If there are too many items, do some recommendations
                # asynchronously. But, return at least a few items
                # immediately

                if f_obj.get('created_by_study', False):
                        continue

                if i >= MAX_REC_TARGETS:
                    continue

                if i > 3:
                    recommend_preds.delay(exp_cond, f_obj['id'], uid, 'del',
                                          action_spec_opts = act_spec_opts)
                else:
                    recommend_preds(exp_cond, f_obj['id'], uid, 'del',
                                    action_spec_opts = act_spec_opts)


            else:
                rel_rec = Recommends.query.filter_by(user_id = uid,
                                                     dst_hash = f_obj['hash'],
                                                     is_rand = rand_recs,
                                                     accepted = False,
                                                     **VALID_REC_CONDS).all()

                for rc in rel_rec:
                    rc.invalidated = True

                db.session.commit()

        recs, num_faded = retrieve_recs(uid,
                                        action = "del",
                                        dec_recs = (rec_id is not None),
                                        find_recs = False,
                                        rand_recs = rand_recs)

        self._update_from_faded(uid, num_faded)

        db.session.commit()

        return {'removed': rem_hashes, 'recs': recs}, 200

    def search(self, q, target = None, mimes = [], owner = None,
               truetype = None, date = None, valusage = None):

        """

        Return a list of files and folders list, that match the search string. arguments:

          q : search string
          target : search target hash (optional)
          mimes : Array of search target MIME-type (optional)
          owner: Owner of file
          truetype: "Text", "Media", or "Other"
          valusage: What the q field means, could be filename, contains words, shared with

        Response:

          files : (Array) array of objects - files and folders list, that match the search string.

        """

        user = session['user_id']

        es = Elasticsearch()

        match_q = []
        if owner:
            owner_q = elastic_term_query("owner", (owner == "Owned by me"))
            match_q.append(owner_q)
        if truetype:
            truetype = truetype.lower()
            truetype_q = elastic_term_query("gentype", truetype)
            match_q.append(truetype_q)
        if date:
            match_q.append(elastic_range_query("timestamp", date))
        if valusage == "Filename":
            match_q.append(elastic_wildcard_query("fname", q))
        elif valusage == "Contains words":
            match_q.append(elastic_term_query("text", q))

        query = {"query": {"bool": {"must": match_q}}}

        res = es.search(index=user, body=query)
        ret_ids = [r["_source"]["id"] for r in res["hits"]["hits"]]
        files, folds = self._db_obj_ids(ret_ids)

        files_ret = [file_ret(r) for r in files]
        folds_ret = [folder_ret(r) for r in folds]

        files_ret = self._validate_return(files_ret)
        folds_ret = self._validate_return(folds_ret)

        return {'files': files_ret + folds_ret}, 200

    def _recurse_size(self, objs_to_size):

        size = 0

        for f in objs_to_size:
            if is_file_object(f):
                size += f.size
            else:
                sub_files, sub_folds = self._sub_items(f)
                size += self._recurse_size(sub_files + sub_folds)

        return size

    def size(self, targets):

        """

        Returns the size of a directory or file.

        Arguments:

          cmd : size
          targets[] : hash paths of the nodes

        Response:

          size: The total size for all the supplied targets.
          fileCnt: (Optional to API >= 2.1025) The total counts of the file for
                                               all the supplied targets.
          dirCnt: (Optional to API >= 2.1025) The total counts of the directory for
                                              all the supplied targets.
          sizes: (Optional to API >= 2.1030) An object of each target size infomation.

        """

        files_to_size, folds_to_size = self._db_obj_targets(targets)
        size = self._recurse_size(files_to_size + folds_to_size)

        return {'size': size}, 200

    def tmb(self, targets):

        """

        Background command. Creates thumbnails for images that do not have them.
        Number of thumbnails created at a time is specified in the Connector_Configuration_RU
        option tmbAtOnce. Default is 5.

        Arguments:

          targets[] : an array of hash path to the directory in which to create thumbnails

        Response:

          {'images': {'(hash_fpath)': '(thumbnail_url)'}}

        """

        raise NotImplemented()

    def tree(self, target):

        """

        Return folder's subfolders.

        Arguments:

          cmd : tree
          target : folder's hash

        Response:

          tree : (Array) Folders list

        """

        user = session['user_id']

        folder_list = [folder_ret(r) for r in Folder.query.filter_by(user_id=user,
                                                                     parent_hash=target,
                                                                     trashed=False)]

        for item in folder_list:
            item['dirs'] = has_subdirs(user, item)

        folder_list = self._validate_return(folder_list)

        resp = {'tree': folder_list}

        return resp, 200

    def upload(self, target, upload, upload_path, mtime, name, renames, suffix, hashes, overwrite):

        """
        Process file upload requests. Client may request the upload of multiple files at once.

        Arguments:

          target : hash of the directory to upload
          upload[] : array of multipart files to upload
          upload_path[] : (optional) array of target directory hash, it has been a
                                     pair with upload[]. (specified at folder upload)
          mtime[] : (optional) array of files UNIX time stamp, it has been a pair with upload[].
          name[] : (optional) array of files name for suggest, it has been a pair with upload[].
          renames[] : (optional) array of rename request filenames
          suffix : (optional) rename suffix
          hashes[hash] : (optional) array of hash: filename pairs
          overwrite : (optional) Flag to overwrite or save another name. If "0" is specified
                                 and the same name file exists, the connector should save the
                                 upload file as a different name.

        Response:

          added : (Array) of files that were successfully uploaded.

        """

        raise NotImplemented()

    def url(self, target, options):

        """

        Returns the url of a file. This method is called if the initial value for the
        file's url is "1".

        Arguments:

          target : hash of file
          options[] : array of options (?)

        Response:

          url: url of the file

        """

        #TODO: Figure out what the hell url is

        pass

    def zipdl(self):

        """
        Not implemented.
        """

        raise NotImplementedError()
