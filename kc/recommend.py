import copy
import time
import pickle
import pprint
import random
import pandas as pd
import numpy as np
import xgboost as xgb

from sqlalchemy import and_, or_, func, desc
from sklearn.metrics.pairwise import euclidean_distances

from felfinder import celery
from felfinder.models import db, File, Simils, Folder, CommandFiles, User
from felfinder.models import CommandHistory, Recommends, VALID_REC_CONDS
from felfinder.utils import get_rand_id
from felfinder.config import (REC_ACT_DECAY, PER_TIME_YIELD_RECS,
                              BINOMIAL_P, MAX_RAND_RECS, THRESH,
                              SWAMPNET_REP_NAME, DIST_THRESH)
from felfinder.pred_utils import (get_clf, pred_matrix_and_inds,
                                  format_recs_for_storage, clf_pred_outputs,
                                  pred_inds)

#MODEL_PATH = 'felfinder/models/xgb_{}_0.pkl'

class FObjRecWrapper:

    """
    Wrapper for translating database results into JSON blobs
    for recommendations

    Current interface assumes a result with the following fields:

      id
      user_id
      path_hash
      name
      parent_hash
      time_run

    """

    def __init__(self, f_obj, action, strength):
        self.f = f_obj
        self.action = action
        self.strength = strength
        self.time_run = self.f.time_run or 0
        self.polite = False
        self.parent_name = None
        self.parent_path = None
        self.dst = None
        self.dst_name = None
        self.explain_name = None
        self.explain_name = None
        self.rec_id = None

    @property
    def id(self):
        return self.f.id

    @property
    def path_hash(self):
        return self.f.path_hash

    @property
    def name(self):
        return self.f.name

    @property
    def parent_hash(self):
        return self.f.parent_hash

    @property
    def user_id(self):
        return self.f.user_id

    def is_polite(self):
        return self.polite

    def set_explain(self, name, hsh = None):
        self.explain_name = name
        self.explain_hash = hsh

    def set_rand(self, rand):
        self.rand = rand

    def ready_dst_attrs(self, dst_obj):

        self.dst_ready = True

        self.dst = dst_obj.path_hash
        self.dst_name = dst_obj.path

    def ready_parent_attrs(self, uid):

        self.parent_ready = True

        parent_obj = Folder.query.filter_by(user_id = uid,
                                            path_hash = self.parent_hash).one()
        self.parent_name = parent_obj.name
        self.parent_path = parent_obj.path
        self.parent_id = parent_obj.id

    def ready_since_touched(self, curr_time):

        self.time_ready = True
        self.since_touched = curr_time - self.time_run
        self.timestamp = curr_time

    def finalize_polite(self, uid):

        if not self.action == 'move':
            self.dst_ready = True

        prepared = [self.parent_ready, self.time_ready, self.dst_ready]
        assert all(prepared)

        two_mins = 2 * 60
        is_time_polite = self.since_touched > two_mins
        if self.action == 'move':
            is_not_done = self.parent_hash != self.dst
        else:
            is_not_done = True

        rel_rec = Recommends.query.filter_by(user_id = uid,
                                             is_rand = self.rand,
                                             rec_path_hash = self.path_hash,
                                             accepted = False,
                                             **VALID_REC_CONDS).all()
        is_only_for_id = rel_rec == []

        self.is_only_for_id = is_only_for_id
        self.is_time_polite = is_time_polite
        self.is_not_done = is_not_done

        polite_conditions = [is_time_polite, is_not_done, is_only_for_id]
        #polite_conditions = [is_time_polite, is_not_done]

        self.polite = all(polite_conditions)

    def db_rec_obj(self, uid, action, action_spec_opts):

        self.rec_id = get_rand_id()

        rec = Recommends(rand_id = self.rec_id,
                         user_id = uid,
                         cmd_id = action_spec_opts['cmd_id'],
                         rec_path_hash = self.path_hash,
                         rec_name = self.name,
                         dst_hash = getattr(self, "dst"),
                         dst_name = getattr(self, "dst_name"),
                         explain_name = self.explain_name,
                         explain_hash = getattr(self, "explain_hash"),
                         explain_target = action_spec_opts.get('target_cmdlog'),
                         explain_dst = action_spec_opts.get('dst_cmdlog'),
                         parent_hash = self.parent_hash,
                         parent_path = self.parent_path,
                         parent_id = getattr(self, "parent_id"),
                         timestamp = int(time.time()),
                         is_only_for_id = self.is_only_for_id,
                         time_polite = self.is_time_polite,
                         not_done = self.is_not_done,
                         action = action,
                         strength = self.strength,
                         num_remain = REC_ACT_DECAY,
                         is_rand = self.rand)

        return rec

class RecWrapper:

    def __init__(self, r):
        self.rec = r

    def move_ret(self):

        blob = {'rec_id': self.rec.rand_id,
                'phash': self.rec.rec_path_hash,
                'action': 'move',
                'new': (self.rec.num_remain == REC_ACT_DECAY) and not (self.rec.sent),
                'name': self.rec.rec_name,
                'explain_name': self.rec.explain_name,
                'explain_hash': self.rec.explain_hash,
                'dst': self.rec.dst_hash,
                'dstname': self.rec.dst_name,
                'ppath': self.rec.parent_path,
                'pid': self.rec.parent_id,
                'parhash': self.rec.parent_hash,
                'strength': self.rec.strength,
                'inv': self.rec.to_inv}

        return blob

    def find_ret(self):

        blob = {'rec_id': self.rec.rand_id,
                'phash': self.rec.rec_path_hash,
                'action': 'find',
                'new': (self.rec.num_remain == REC_ACT_DECAY) and not (self.rec.sent),
                'name': self.rec.rec_name,
                'explain_name': self.rec.explain_name,
                'explain_hash': self.rec.explain_hash,
                'ppath': self.rec.parent_path,
                'parhash': self.rec.parent_hash,
                'strength': self.rec.strength,
                'inv': self.rec.to_inv}

        return blob

    def del_ret(self):

        blob = {'rec_id': self.rec.rand_id,
                'phash': self.rec.rec_path_hash,
                'action': 'del',
                'new': (self.rec.num_remain == REC_ACT_DECAY) and not (self.rec.sent),
                'name': self.rec.rec_name,
                'explain_name': self.rec.explain_name,
                'explain_hash': self.rec.explain_hash,
                'ppath': self.rec.parent_path,
                'pid': self.rec.parent_id,
                'parhash': self.rec.parent_hash,
                'strength': self.rec.strength,
                'inv': self.rec.to_inv}

        return blob

    def json_ret(self):

        formats = {'move': self.move_ret,
                   'find': self.find_ret,
                   'del': self.del_ret}
        return formats[self.rec.action]()

def multi_rec_check(find_recs, render_recs):

    """
    Because recommendations can be created simultaneously, we need
    to do another check at retrieval time whether there are multiple
    Recommendations active for a single path hash
    """

    if find_recs:
        return render_recs

    seen_hash = set()
    recs_to_use = []
    for r in render_recs:
        if r.rec_path_hash in seen_hash:
            r.is_only_for_id = False
        else:
            recs_to_use.append(r)
            seen_hash.add(r.rec_path_hash)

    return recs_to_use

def retrieve_recs(uid, action, dec_recs, find_recs, rand_recs):

    render_recs = Recommends.query.filter_by(user_id = uid,
                                             is_rand = rand_recs,
                                             **VALID_REC_CONDS)

    if find_recs:
        render_recs = render_recs.filter(Recommends.action == "find")
    else:
        render_recs = render_recs.filter(Recommends.action != "find")

    render_recs = render_recs.order_by(desc(Recommends.strength * Recommends.num_remain)).all()

    render_recs = multi_rec_check(find_recs, render_recs)
    ret_json = [RecWrapper(r).json_ret() for r in render_recs
                if not (r.action == 'del' and r.accepted)]

    num_faded = {'move': 0, 'del': 0}

    if dec_recs:
        for r in render_recs:
            if not r.to_inv and r.sent:
                dec_val = 1 if r.action == action else 2
                r.num_remain -= dec_val
                if r.num_remain <= 0:
                    r.faded = True
                    num_faded[r.action] += 1

    for r in render_recs:
        r.sent = True

    db.session.commit()

    return ret_json, num_faded

def rand_recommend(file_id, uid, action, action_spec_opts):

    do_rec_items = np.random.binomial(1, PER_TIME_YIELD_RECS)
    if do_rec_items == 0:
        return

    num_recs = np.random.binomial(MAX_RAND_RECS, BINOMIAL_P)

    user_files = File.query.filter_by(user_id = uid,
                                      trashed = False).all()

    chosen_files = random.sample(user_files, num_recs)
    pred_file_ids = [f.id for f in chosen_files]
    preds = [1 for _ in pred_file_ids]
    certain_items = 0.5 * np.random.random(len(preds))

    res = pd.DataFrame(pd.np.column_stack([preds, certain_items]),
                       index=pred_file_ids, columns=["pred_class", "max_pred"])

    res = res.sort_values(by=["pred_class", "max_pred"], ascending=False)

    action_spec_opts['rand'] = True

    store_positive_recs(uid, res, action, action_spec_opts)

def get_simil_rows(file_id, uid, action):

    """
    Get rows from similarity table to predict on
    """

    user = User.query.filter_by(id=uid).one()
    pairs = Simils.query.filter(and_(Simils.user_id == uid,
                                     or_(Simils.file_id_A == file_id,
                                         Simils.file_id_B == file_id)))
    if action == 'find':
        pairs = pairs.filter(Simils.find_true_cert >= THRESH)
    elif action == 'move':
        pairs = pairs.filter(Simils.move_true_cert >= user.move_thresh)
    elif action == 'del':
        pairs = pairs.filter(Simils.del_true_cert >= user.del_thresh)

    pred_mat = pd.read_sql(pairs.statement, con=db.engine)

    return pred_mat


def simil_recommend(file_id, uid, action, action_spec_opts):


    simil_rows = get_simil_rows(file_id, uid, action)
    if len(simil_rows) == 0:
        return

    #clf = get_clf(action)
    #pred_mat, pred_file_ids = pred_matrix_and_inds(simil_rows, file_id)
    pred_file_ids = pred_inds(simil_rows, file_id)

    pred_col, cert_col = action + "_pred", action + "_cert"
    preds, certainty = simil_rows[pred_col], simil_rows[cert_col]

    # _, preds, certainty, _ = clf_pred_outputs(clf, pred_mat)
    res = format_recs_for_storage(pred_file_ids, preds, certainty)

    action_spec_opts['rand'] = False

    store_positive_recs(uid, res, action, action_spec_opts)


def get_swampnet_mat(uid):

    with open(SWAMPNET_REP_NAME.format(uid), 'rb') as of:
        base_ids, swamp_mat = pickle.load(of)

    return base_ids, swamp_mat

def swampnet_recommend(file_id, uid, action, action_spec_opts):

    base_ids, swamp_mat = get_swampnet_mat(uid)

    mat_index = base_ids.index(file_id)
    vec_rep = swamp_mat[mat_index, :]

    dists = euclidean_distances(vec_rep.reshape((1, -1)), swamp_mat).reshape((-1,))

    # Remove our base file
    dists = np.delete(dists, mat_index)
    base_ids.remove(file_id)

    preds = (dists < DIST_THRESH).astype('int32')
    certainty = np.maximum(0, 1 - dists)

    res = format_recs_for_storage(base_ids, preds, certainty)
    action_spec_opts['rand'] = False

    store_positive_recs(uid, res, action, action_spec_opts)


def merge_result_set(f_objs):

    last_touched = {}
    for f in f_objs:
        curr_f = last_touched.get(f.id, None)
        if curr_f is None or f.time_run >= curr_f.time_run:
            last_touched[f.id] = f

    return list(last_touched.values())


def rec_ids_to_rec_wrappers(uid, recs, action):

    """
    Translates IDs of files to recommend into wrappers
    that contain all the needed information for each file
    for recommendations.

    Specifically, we want to find the time of the most
    recent touch on a file. We do this by selecting
    the database File objects corresponding to the IDs in recs,
    matching these to the times they appeared in command logs
    with an outer join (bc files might not have been touched
    and LEFT JOIN is apparently not available), and then matching
    this to the CommandHistory table to find the time the commands
    corresponding to these logs were run (again using an outer
    join because some files have not been touched). We then
    project onto the columns we need.

    Assumptions of the current query output:

      1. IDs are not unique in result set
      2. time_run may be None
      3. No order assumed
      4. All file ids from rec_ids are in the output, but do not
         necessarily have all attributes from tables besides File

    We make IDs unique and make time_run non-null in merge_result_set

    """

    rec_ids = list(recs.keys())

    relevant_files = File.query.filter(File.user_id == uid,
                                       File.id.in_(rec_ids),
                                       File.trashed == False)

    logs_rel_files = relevant_files.outerjoin(CommandFiles,
                                              and_(File.user_id == CommandFiles.user_id,
                                                   File.id == CommandFiles.file_id))

    # Necessary to filter bc File <--> CommandFiles is many to many
    del_unrel_flogs = logs_rel_files.filter(File.id != None)

    # Not necessary to filter bc File / CommandFiles <--> CommandHistory is many to one
    cmd_times = del_unrel_flogs.outerjoin(CommandHistory,
                                          and_(File.user_id == CommandHistory.user_id,
                                               CommandFiles.cmd_id == CommandHistory.rand_id))
    rec_files = cmd_times.add_columns(File.id,
                                      File.user_id,
                                      File.path_hash,
                                      File.name,
                                      File.parent_hash,
                                      CommandHistory.time_run).all()

    f_obj_set = [FObjRecWrapper(r, action, recs[r.id]['max_pred']) for r in rec_files]
    f_objs = merge_result_set(f_obj_set)

    return f_objs

def augment_fobj_action(uid, f_objs, action, action_spec_opts):

    """
    Augments each FObjRecWrapper with attributes necessary
    to determine whether it's a recommendation
    we should be providing

    Arguments:
      uid (str): ID of user for recommendations
      f_objs (list): FObjRecWrapper list with unique IDs, last time run
      action (str): the action to recommend
      action_spec_opts (dict): Options specific to particular actions

    Returns:
      (void) : mutates state of each f_obj

    """

    now = int(time.time())

    for f in f_objs:

        f.ready_parent_attrs(uid)
        f.ready_since_touched(now)
        f.set_explain(action_spec_opts['explain_name'],
                      action_spec_opts.get('explain_hash'))
        f.set_rand(action_spec_opts['rand'])

        if action == 'move':
            dst = action_spec_opts['dst']
            dst_obj = Folder.query.filter_by(user_id = uid, path_hash = dst).one()
            f.ready_dst_attrs(dst_obj)

        f.finalize_polite(uid)


def store_recs(uid, rec_files, action, action_spec_opts):

    """
    We need to record what recommendations we have provided to
    the user in the database
    """

    for f in rec_files:
        db.session.add(f.db_rec_obj(uid, action, action_spec_opts))

def store_positive_recs(uid, recs, action, action_spec_opts):

    """
    The recommendations are in a Pandas DataFrame--let's fix that.

    Arguments:
      uid (str): ID of user for recommendations
      recs (pd.DataFrame): predicted classes and their probability of prediction
      action (str): the action to recommend
      action_spec_opts (dict): Options specific to particular actions

    """

    pos_recs = recs[recs['pred_class'] > 0][['max_pred']].to_dict('index')
    rec_files = rec_ids_to_rec_wrappers(uid, pos_recs, action)
    augment_fobj_action(uid, rec_files, action, action_spec_opts)
    store_recs(uid, rec_files, action, action_spec_opts)

    db.session.commit()
