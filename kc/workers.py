import datetime
import copy
import json
import random
import time
import logging
import queue
import sys
import pickle
import os
import numpy as np
import pprint

import cProfile
import pstats

#import pandas as pd

from guppy import hpy
from sqlalchemy import and_, or_, func, desc, cast, literal, select, union_all
from klepto.archives import dir_archive, file_archive
from pprint import pformat
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from elasticsearch.exceptions import AuthorizationException as ElasticAuthExc
from itertools import combinations, repeat, chain
from urllib.error import HTTPError
from oauth2client.client import GoogleCredentials, AccessTokenCredentialsError
from oauth2client.client import AccessTokenCredentials
from google.protobuf.json_format import MessageToJson, MessageToDict
from google.cloud import vision
from google.cloud.vision import types
from googleapiclient.errors import HttpError as GoogHttpError
from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
from datasketch import MinHash
from celery import chain as cel_chain, chord, subtask
from cached_property import threaded_cached_property

from felfinder import celery as cel_app
from felfinder.activity_parse import ingest_action
from felfinder.pred_utils import get_simils_from_tuples, precompute_preds
from felfinder.models import (db, User, File, Folder, Simils, ImageFeats,
                              Schemas, FileErrors, SharedUsers, ActionSample,
                              CommandHistory, CommandFiles, GroupSample,
                              NAME_PATCH_LEN)
from felfinder.utils import (create_service_object, get_rand_id, get_hash,
                             file_ret, folder_ret, last_mod_func,
                             schema_jaccard, google_perm_jaccard, path_dist,
                             bigram_sim, get_file_type, refresh_google_token,
                             get_fobj_from_uid_id, rand_prol_id, grouper,
                             simple_cosine_sim, load_if_exists, jaccard,
                             even_partition, trunc_str, token_simils, med_sel)
from felfinder.recommend import (simil_recommend, rand_recommend,
                                 swampnet_recommend)
from felfinder.replay import command_list, replay_single
from felfinder.text_process import ProcessTextThread, tfidf_analyze, word_to_vec, edit_dist
from felfinder.text_process import finalize_text_rep, text_proc_celery, get_metadata
from felfinder.img_process import get_image_links_google, get_image_labels, goog_image_simil_feat
from felfinder.img_process import finalize_im_rep, local_image_feats
from felfinder.table_process import schema_extract
from felfinder.config import (TEST_SERVER, TOP_TEXT_SIZE, MAX_SELECT,
                              SWAMPNET_RECS, ROOT_ID, ROOT_PHASH, ROOT_PATH,
                              ROOT_NAME, LOCAL_IMG_FEATS, DO_BERT_REPR,
                              SWAMPNET_REP_NAME)
from felfinder.sampling import (save_sample_objs, sample_actions, sample_explans,
                                save_explan_samps, TOTAL_NUM_EXPLAN_QS,
                                action_samp_from_recs, action_samp_from_cmds)

from felfinder.swampnet.dl_model import proc_full_reps
from felfinder.swampnet.utils import (encode_paths, gen_token_universe,
                                      shares_from_db_obj, calc_share_reps)

WORKSPACE = "workspace/"
MEM_SIZE = 1000
NUM_REFRESH = 10
GROUP_VALS = 2500
WRAP_GROUP_VALS = 50
DL_LIMIT = int(1e8)

pared_reason_dict = {"cmd_cmg_move": 4,
                     "cmd_cmg_del": 2,
                     "hist_cmg_move": 2,
                     "false_pos_move": 4,
                     "false_pos_del": 2,}

full_reason_dict = {"accepted_move_same": 2,
                    "accepted_move_diff": 2,
                    "accepted_del_same": 1,
                    "accepted_del_diff": 1,
                    "untaken_move_same": 3,
                    "untaken_move_diff": 3,
                    "untaken_del_same": 2,
                    "untaken_del_diff": 1}

explan_reason_dict = {'small': 2, 'med': 2, 'large': 2,
                      'complex': 2, 'not_complex': 2,
                      'discrim': 2, 'not_discrim': 2}

@cel_app.task(bind=True)
def populate_google(self, uid):
    """
    Given our credentials, requests the metadata for all the users' files and folders

    Arguments:
        uid (String): the id of the user as specified in the User object in the database

    Returns:
       (void)

    """

    log = logging.getLogger(__name__)
    log.setLevel(10)

    proc_state = GoogProcState(uid)
    file_collection = {}

    page_token = None
    page_token = batch_update_file_set(page_token, file_collection, proc_state)
    while page_token is not None:
        page_token = batch_update_file_set(page_token, file_collection, proc_state)

    # TODO: If collection has less than 100 REAL items, BOP EM
    # TODO: Other eligibility criteria

    root_folder = Folder(rand_id=get_rand_id(),
                         user_id=uid,
                         id=ROOT_ID,
                         name=ROOT_NAME,
                         path=ROOT_PATH,
                         path_hash=ROOT_PHASH,
                         created_by_study = False,
                         created_time = int(time.time()))
    file_collection[ROOT_ID] = root_folder

    file_collection = fix_file_paths(file_collection)

    for item in file_collection.values():
        if item:
            db.session.add(item)

    db.session.commit()

    return uid

@cel_app.task(bind=True)
def activity_logs(self, uid):

    """
    Get the activity logs for each file
    """

    proc_state = GoogProcState(uid, 'driveactivity', 'v2')
    page_token = None
    activities = []
    page_token = batch_update_activities(page_token, activities, proc_state)
    while page_token is not None:
        page_token = batch_update_activities(page_token, activities, proc_state)

    db.session.bulk_save_objects(activities)

    db.session.commit()

    return uid

@cel_app.task(bind=True)
def recommend_preds(self, rand_cond, file_id, uid, action, action_spec_opts):

    """
    Wrapper function for recommendations

    Arguments:
      file_id (str): ID of file in database that had the initial action applied
      uid (str): ID of user who applied the action
      action (str): Name of the action taken
      action_spec_opts (dict): Options specific to particular actions

    Returns:
      (list): List of file IDs (names?) to recommend the same action for

    """

    if rand_cond == "rand_recs":
        rand_recommend(file_id, uid, action, action_spec_opts)
    if SWAMPNET_RECS:
        swampnet_recommend(file_id, uid, action, action_spec_opts)
    else:
        simil_recommend(file_id, uid, action, action_spec_opts)

@cel_app.task(bind=True)
def replay_all(self, uid):

    user = User.query.filter_by(id = uid).one()
    if user.acc_changes:
        print("User {} already accepted changes".format(user))

    commands = command_list(uid)
    drive_obj = create_service_object(uid, scope_level = "replay")

    for c in commands:
        replay_single(drive_obj, c)
        c.replayed = True

    user.acc_changes = True

    db.session.commit()

def save_fwrap_res(proc_state, fwrap):
    with open(WORKSPACE + proc_state.uid + "_" + fwrap.id, 'wb') as of:
        pickle.dump(fwrap, of)


def save_fwrap_dir(proc_state, fwrap):

    all_wrap = load_all_wrap(proc_state, noload = True)
    all_wrap[fwrap.id] = fwrap
    all_wrap.dump()


def load_fwrap_res(proc_state, fid):

    fname = WORKSPACE + proc_state.uid + "_" + fid
    with open(fname, 'rb') as of:
        res = pickle.load(of)
        shred_file(fname)
    return res

def all_wrap_fname(proc_state):
    return WORKSPACE + 'all_wrap_' + proc_state.uid + '/'

def load_all_wrap(proc_state, cached = False, readonly=False, noload = False, keys=[]):

    if readonly:
        all_wrapper_recon = dir_archive(all_wrap_fname(proc_state), {},
                                         serialized = True,
                                         memmode = 'r',
                                         memsize = MEM_SIZE)
    else:
        all_wrapper_recon = dir_archive(all_wrap_fname(proc_state), {},
                                         serialized = True,
                                         memsize = MEM_SIZE)

    if not noload:
        if keys:
            fkeys = [k for k in keys if k is not None]
            all_wrapper_recon.load(*fkeys)
        else:
            all_wrapper_recon.load()
    return all_wrapper_recon

def shred_file(fname):
    os.system("shred -u " + fname)

def shred_dir(dirpath):
    os.system("srm -r " + dirpath)

@cel_app.task(bind=True)
def goog_celery_proc(self, i, all_wrap_len, uid, fwrap):

    fwrap = GoogFileWrapper.from_json(fwrap)

    res = {}

    try:
        proc_state = GoogProcState(uid)
        logging.warning("{} out of {}: {}".format(i, all_wrap_len, fwrap.id))
        goog_process_fwrap(proc_state, fwrap)
        #save_fwrap_res(proc_state, fwrap)
        save_fwrap_dir(proc_state, fwrap)
        res['fid'] = fwrap.id

    except Exception as e:
        goog_process_error(uid, fwrap, e)

    logging.warning("Finished processing id {0}".format(fwrap.id))

    if i == 0:
        res['uid'] = proc_state.uid

    return res

@cel_app.task(bind=True)
def precompute_recs(self, clutter, uid):

    logging.warning("Similarities computed for user: {}".format(uid))
    precompute_preds(uid)
    logging.warning("Recommendations precomputed for user: {}".format(uid))

    return uid

@cel_app.task(bind=True)
def set_user_complete(self, clutter, uid):

    proc_state = GoogProcState(uid)
    proc_state.set_complete()
    logging.warning("User's info marked as loaded for user: {}".format(uid))

    #shred_file(all_wrap_fname(proc_state))
    shred_dir(all_wrap_fname(proc_state))


    return uid

@cel_app.task(bind=True)
def index_and_finalize_text(self, uid):

    """
    Finalize text values and index in Elasticsearch
    """

    proc_state = GoogProcState(uid)

    files_to_process = [f.id for f in
                        File.query.filter_by(user_id=proc_state.uid,
                                             error = False).all()]

    for fid_keys in grouper(files_to_process, WRAP_GROUP_VALS, fillvalue=None):

        all_wrappers = load_all_wrap(proc_state, keys=fid_keys)

        logging.warning("Finalizing text and saving image objects" + \
                        " for user: {}".format(proc_state.uid))

        for fwrap in all_wrappers.values():
            fwrap.finalize_usable_text()

        im_objs = finalize_im_rep(proc_state.uid, all_wrappers.values())
        if LOCAL_IMG_FEATS:
            db.session.bulk_save_objects(im_objs)

        logging.warning("Skip representation for user: {}".format(proc_state.uid))

        # We no longer need this for similarity, and we can save on the load time
        # for the archive by removing this data
        for f in all_wrappers.values():
            f.fb = None

        db.session.commit()
        all_wrappers.dump()

    all_wrappers = load_all_wrap(proc_state)

    return uid

@cel_app.task(bind=True)
def collect_results(self, ps_fwrap_dicts):
    """
    In order to pass proc_state through, we need to collect it from the first
    item

    Arguments:
      ps_fwrap_tuples (list): [{'fid': fwrap_id,
                                'uid': user_id} for _ in range all_wrappers]

                              'fid' doesn't exist if the file errored
                              'uid' only exists for one element

    """

    uid = [ft['uid'] for ft in ps_fwrap_dicts if 'uid' in ft][0]
    proc_state = GoogProcState(uid)

    logging.warning("Finished downloading for user: {}... ".format(proc_state.uid) + \
                    "Wait until text processing is complete")

    fids_to_proc = [ft['fid'] for ft in ps_fwrap_dicts if 'fid' in ft]

    # all_wrapper_recon = load_all_wrap(proc_state)

    # for i, fid in enumerate(fids_to_proc):

    #     full_fwrap = load_fwrap_res(proc_state, fid)
    #     all_wrapper_recon[fid] = full_fwrap

    # all_wrapper_recon.dump()

    return uid

@cel_app.task(bind=True)
def post_proc_wrapper(self, uid, task, post_proc):

    """

    ref: https://github.com/celery/celery/issues/5490#issuecomment-511537591
    ref: https://stackoverflow.com/questions/60907707/chain-a-celery-task-that-returns-a-list-into-a-group-in-middle-of-chain?noredirect=1

    Alright, so, Celery is horrific. Basically, primitives (chain, group, chord)
    play poorly with each other. Our workflow looks like:

    A)            group process
                       |
    B)           break up group
                       |
            --<------<--->------>--
            | ........ | ........ |
            v ........ v ........ v
    C)   process    process    process
            | ........ | ........ |
            v ........ v ........ v
            ---->---->---<----<----
                       |
    D)        reconstitute group
                       |
    E)       perform post processing

    Steps A-B are easy in celery by just using a chain. Step C complicates things.
    For step C, we need a group or a chord. However, the trick is that the result
    of step B is the iterator for the group, i.e.:

       group(proc_item.s(a) for a in PREVIOUS_RESULT)

    Ergo, we need a specialty way of passing this, which uses this sub-task.
    By providing this signature for step C, our arguments look like:

    Arguments:
      it (iterator): Iterator of arguments to be applied
      task (dict): dictionary of the signature for the step C task to be applied
                   on individual items
      post_proc (dict): dictionary of the chain for the steps D-E tasks

    Now, because we need to perform a callback after our group, we need to
    use a chord. At the time I write this (5/19/20), I don't understand the
    precise syntax of a chord, other than that it's roughly,

      chord(group_task, callback)   OR   chord(group_task)(callback)

    I'm not sure of the differences between the two. But either way,
    we'll use the subtask function and clone method to create Signatures out of
    our task and post_proc arguments, and then return the application of
    a chord

    """

    proc_state = GoogProcState(uid)
    files_to_process = File.query.filter_by(user_id=proc_state.uid,
                                            error = False).all()
    #all_wrappers = load_all_wrap(proc_state)
    all_wrappers = {f.id: GoogFileWrapper(f, proc_state.uid) for f in files_to_process}

    it = FwrapIter(all_wrappers, proc_state)

    cb = subtask(task)
    post_cb = subtask(post_proc)
    task_l = [cb.clone(arg) for arg in it]

    return chord(task_l)(post_cb.clone())

@cel_app.task(bind=True)
def simil_wrapper(self, uid, task, post_proc):

    proc_state = GoogProcState(uid)
    all_wrappers = load_all_wrap(proc_state)
    logging.warning("About to compute similarity for user {}".format(proc_state.uid))

    init_simils(uid)
    make_text_mats(proc_state, all_wrappers)

    full_sims = Simils.query.filter_by(user_id = uid).all()
    full_sim_tups = [(s.file_id_A, s.file_id_B) for s in full_sims]
    full_sim_tups = sorted(full_sim_tups, key=lambda x: x[0])

    it = grouper(full_sim_tups,
                 GROUP_VALS, fillvalue = (None, None))

    cb = subtask(task)
    post_cb = subtask(post_proc)
    task_l = [cb.clone([arg]) for arg in it]

    return chord(task_l)(post_cb.clone())

@cel_app.task(bind=True)
def swampnet_rep(self, uid, task, post_proc):

    proc_state = GoogProcState(uid)
    all_wrappers = load_all_wrap(proc_state)
    logging.warning("Creating SwampNet representation for user {}".format(proc_state.uid))
    logging.warning("Making text matrices for {}".format(proc_state.uid))

    make_text_mats(proc_state, all_wrappers)

    base_ids = sorted(list(all_wrappers.keys()))

    paths = [all_wrappers[k].path for k in base_ids]
    path_codes = encode_paths(paths)

    toks = gen_token_universe()

    share_use = SharedUsers.query.filter_by(user_id = uid).all()
    shared_maps = calc_share_reps(shares_from_db_obj(base_ids, share_use))

    tfidf_mat, w2v_mat = load_tfidf_w2v(proc_state)

    if tfidf_mat is not None:
        id_list = tfidf_mat['ids']
        id_match = {fid: i for i, fid in enumerate(id_list)}
        for k in all_wrappers.keys():
            if k in id_match:
                all_wrappers[k].w2v_rep = w2v_mat[id_match[k]]


    rep_dict = proc_full_reps(base_ids, all_wrappers, path_codes, toks, shared_maps)

    for k in rep_dict:
        all_wrappers[k].swampnet_rep = rep_dict[k]

    all_wrappers.dump()

    rep_mat = np.vstack([all_wrappers[k].swampnet_rep for k in base_ids])

    with open(SWAMPNET_REP_NAME.format(uid), 'wb') as of:
        pickle.dump((base_ids, rep_mat), of)

    return uid


def elastic_bulk_wrapper(elastic, to_index):
    for success, info in streaming_bulk(elastic, to_index,
                                        chunk_size = 50, request_timeout = 60):
        if not success:
            print("A document failed to index: {}".format(info))

def save_vector_reps(proc_state, all_wrappers):
    """
    Have to save the vector representations for all files
    """

    rep_obj = None
    rep_path = "vector_reps/{}".format(proc_state.uid)

    text_reps = {'ids': [f.id for f in all_wrappers.values() if f.text_rep is not None],
                 'reps': np.array([f.text_rep for f in all_wrappers.values()
                                   if f.text_rep is not None])}
    im_reps = {'ids': [f.id for f in all_wrappers.values() if f.im_rep is not None],
               'reps': np.array([f.im_rep for f in all_wrappers.values()
                                 if f.im_rep is not None])}

    reps_obj = {'user': proc_state.uid,
                'text': text_reps,
                'imgs': im_reps}

    with open(rep_path, 'wb') as of:
        pickle.dump(reps_obj, of)


@cel_app.task(bind=True)
def simil_proc(self, sim_tuples, uid):

    """
    Process similarities in small chunks:

    Arguments:
      uid (str): ID of user to compute similarities for
      sim_tuples (list): list of (a, b) tuples of file IDs to compute
                         similarities for

    """

    proc_state = GoogProcState(uid)
    all_wrappers = load_all_wrap(proc_state, readonly=True)
    simils = get_simils_from_tuples(proc_state.uid, sim_tuples)
    simils, image_pairs = general_simil_feat(proc_state, all_wrappers, simils)
    simils = goog_image_simil_feat(proc_state.uid, simils, image_pairs)
    simils = text_simil_feat(proc_state, simils)
    simils = perm_simil_feat(proc_state, simils)
    db.session.commit()

    return uid

@cel_app.task(bind=True)
def google_simils(self, uid):

    """
    And now that we've downloaded all the content, we need to actually
    create the similarity pairs. This function could later be
    replaced or augmented by other processing functions

    Arguments:
      all_wrappers (dict): Maps file ID to FileWrapper objects

    Returns:
      (None)

    """

    proc_state = GoogProcState(uid)
    all_wrappers = load_all_wrap(proc_state)

    simils = init_simils(proc_state, all_wrappers)
    simils, image_pairs = general_simil_feat(proc_state, all_wrappers, simils)
    simils = goog_image_simil_feat(proc_state.uid, simils, image_pairs)
    simils = text_simil_feat(proc_state, all_wrappers, simils)

    db.session.bulk_save_objects(simils)
    db.session.commit()

    return uid


def set_up_routing(uid, reason_nums, reserve_reasons, samp_type = 'action'):

    samp_meth = sample_actions if samp_type == 'action' else sample_explans

    desired_total = sum(reason_nums.values())

    print("Desired total: {}".format(desired_total))

    chosen_samps = []

    for r, v in reason_nums.items():
        chosen_samps.extend(samp_meth(uid, r, v, chosen_samps))

    remaining = desired_total - len(chosen_samps)
    new_sample_nums = {r: v for r, v in
                       zip(reserve_reasons,
                           even_partition(remaining,
                                          len(reserve_reasons)))}

    print("remaining: {}".format(remaining))
    print("New sample num: {}".format(pprint.pformat(new_sample_nums)))

    for r, v in new_sample_nums.items():
        chosen_samps.extend(samp_meth(uid, r, v, chosen_samps))

    # Fill out the remainder of samples, unless there are aren't
    # enough samples available
    remaining = desired_total - len(chosen_samps)
    last_remaining = 0
    while (len(chosen_samps) != desired_total) and (last_remaining != remaining):
        new_sample_nums = {r: v for r, v in
                           zip(reserve_reasons,
                               even_partition(remaining,
                                              len(reserve_reasons)))}

        print("remaining: {}".format(remaining))
        print("New sample num: {}".format(pprint.pformat(new_sample_nums)))

        for r, v in new_sample_nums.items():
            chosen_samps.extend(samp_meth(uid, r, v, chosen_samps))

        last_remaining = remaining
        remaining = desired_total - len(chosen_samps)

    random.shuffle(chosen_samps)

    # A final check for sensibility

    if len(chosen_samps) < 1 and samp_type == 'action':

        dummy = ActionSample(rand_id = get_rand_id(),
                             user_id = uid,
                             sample_reason = "dummy")
        db.session.add(dummy)

        chosen_samps = [dummy]

    for i, c in enumerate(chosen_samps):
        if samp_type == 'action':
            c.ask_order = i
            c.sel_for_qs = True
        else:
            c.sel_for_qs_ind = i

    db.session.commit()


def set_up_explan_samps(uid):

    """
    Routing for the explanation study
    """

    samps = GroupSample.query.filter_by(user_id = uid).all()

    if len(samps) < 1:
        return
    elif len(samps) <= TOTAL_NUM_EXPLAN_QS:
        for i in range(len(samps)):
            samps[i].sel_for_qs_ind = i

    samps = sorted(samps, key=lambda x: x.size)
    rel_inds = med_sel(samps, samps_per_ind = [3, 3, 3], qs = [0.25, 0.5, 0.75])
    for i, j in enumerate(rel_inds):
        samps[j].sel_for_qs_ind = i

    db.session.commit()


@cel_app.task()
def set_up_qual_routing_full(uid):

    rec_samps = action_samp_from_recs(uid)
    cmds_samps = action_samp_from_cmds(uid)

    db.session.bulk_save_objects(rec_samps + cmds_samps)
    db.session.commit()

    reserve_reasons = ["untaken_move_same", "untaken_move_diff",
                       "untaken_del_same", "untaken_del_diff"]
    set_up_routing(uid, full_reason_dict, reserve_reasons)


@cel_app.task()
def set_up_qual_routing_pared(uid):

    """
    Set up the qualitative question routing for
    the pared-down study.
    """

    save_sample_objs(uid)
    reserve_reasons = ["false_pos_move", "false_pos_del"]
    set_up_routing(uid, pared_reason_dict, reserve_reasons)


@cel_app.task()
def set_up_group_routing(uid):

    """
    Set up the qualitative question routing for
    the pared-down study.
    """

    save_sample_objs(uid)
    reserve_reasons = ["false_pos_move", "false_pos_del"]
    set_up_routing(uid, pared_reason_dict, reserve_reasons)

@cel_app.task()
def set_up_explan_routing(clutter, uid):

    """
    Set up the question routing for the explanation study
    """

    proc_state = GoogProcState(uid)
    all_wrappers = load_all_wrap(proc_state)

    save_explan_samps(uid, all_wrappers)

    reserve_reasons = ['small', 'med', 'large']
    set_up_routing(uid, explan_reason_dict, reserve_reasons, samp_type = 'explan')
    #set_up_explan_samps(uid)


def init_simils(uid):

    proc_state = GoogProcState(uid)
    files_to_sel = File.query.filter_by(user_id = uid, error = False).\
        order_by(func.random()).limit(MAX_SELECT).all()
    for f in files_to_sel:
        f.sel_for_simils = True
    sel_vals = [f.id for f in files_to_sel]

    for fa, fb in combinations(sel_vals, 2):
        s = Simils(user_id=proc_state.uid,
                     file_id_A = fa,
                     file_id_B = fb)
        db.session.add(s)

    db.session.commit()

def general_simil_feat(proc_state, all_wrappers, simils):

    image_pairs = []
    for s in simils:
        wrapA = all_wrappers[s.file_id_A]
        wrapB = all_wrappers[s.file_id_B]
        image_pairs.append(wrapA.is_image and wrapB.is_image)
        s.filename_A = wrapA.fname
        s.filename_B = wrapB.fname
        s.edit_dist = edit_dist(wrapA.fname, wrapB.fname)
        s.bin_simil = wrapA.minhash_content.jaccard(wrapB.minhash_content)
        s.tree_dist = path_dist(wrapA.path, wrapB.path)
        s.size_dist = abs(wrapA.size - wrapB.size)
        s.last_mod_simil = last_mod_func(wrapA.last_modified, wrapB.last_modified, True)
        s.schema_sim = schema_jaccard(proc_state, wrapA, wrapB)
        s.bigram_simil = bigram_sim(wrapA.fname.lower(), wrapB.fname.lower())
        s.token_simil = token_simils(wrapA.fname.lower(), wrapB.fname.lower())

    return simils, image_pairs

def perm_simil_feat(proc_state, simils):

    """
    Broke this out into a separate function, as previously it was
    folded into the general_simil_feat function. However,
    repeated DB calls were more expensive than needed when
    we can likely load all this info into memory
    """

    id_tups = [(s.file_id_A, s.file_id_B) for s in simils]
    full_ids = set(chain.from_iterable(id_tups))

    full_perms = SharedUsers.query.filter(SharedUsers.user_id==proc_state.uid,
                                          SharedUsers.id.in_(full_ids)).all()

    perms_in_mem = {}
    for p in full_perms:
        if p.id in perms_in_mem:
            perms_in_mem[p.id].append(p.shared_user)
        else:
            perms_in_mem[p.id] = [p.shared_user]

    for s in simils:
        base = perms_in_mem.get(s.file_id_A, [])
        if len(base) <= 1:
            s.perm_simil = 0.0
            continue
        other = perms_in_mem.get(s.file_id_B, [])
        if len(other) <= 1:
            s.perm_simil = 0.0
            continue
        s.perm_simil = jaccard(base, other)

    return simils

def sim_from_mat(id_match, idA, idB, mat):

    vec_A = mat[id_match[idA]]
    vec_B = mat[id_match[idB]]
    return simple_cosine_sim(vec_A, vec_B)

def text_simil_feat(proc_state, simils):

    tfidf_mat, w2v_mat = load_tfidf_w2v(proc_state)

    if tfidf_mat is not None:
        id_list = tfidf_mat['ids']
        id_match = {fid: i for i, fid in enumerate(id_list)}
        tfidf_full_mat = tfidf_mat['tfidf']

    for s in simils:
        if tfidf_mat is not None:
            if s.file_id_A in id_match and s.file_id_B in id_match:
                s.tfidf_sim = sim_from_mat(id_match, s.file_id_A,
                                           s.file_id_B, tfidf_full_mat)
                if w2v_mat is not None:
                    s.word_vec_sim = sim_from_mat(id_match, s.file_id_A,
                                                  s.file_id_B, w2v_mat)

    return simils

def make_text_mats(proc_state, all_wrappers):

    id_corpus_match = [(i, f.text) for i, f in all_wrappers.items()
                       if f.usable_text and f.sel_for_simils]
    id_list = [pair[0] for pair in id_corpus_match]
    corpus = [pair[1] for pair in id_corpus_match]

    word_vec_matrix = None
    tfidf_matrix = None

    try:

        tfidf_matrix, vocab = tfidf_analyze(corpus)
        word_vec_matrix = word_to_vec(corpus)

    except (MemoryError, ValueError) as e:
        logging.error(e)

    if tfidf_matrix is not None:
        save_tfidf_matrix(proc_state, id_list, tfidf_matrix, vocab)

    if word_vec_matrix is not None:
        save_w2v_matrix(proc_state, word_vec_matrix)

def get_tfidf_path(proc_state):
    return "vector_reps/tfidf_{}".format(proc_state.uid)

def get_w2v_path(proc_state):
    return "vector_reps/word2vec_{}".format(proc_state.uid)

def load_tfidf_w2v(proc_state):

    tfidf_path = get_tfidf_path(proc_state)
    w2v_path = get_w2v_path(proc_state)

    tfidf_mat = None
    if os.path.exists(tfidf_path):
        tfidf_mat = file_archive(tfidf_path, {},
                                 serialized = True,
                                 cached = True,
                                 memmode='r',
                                 memsize = MEM_SIZE / 4)
        tfidf_mat.load()

    w2v_mat = None
    if os.path.exists(w2v_path):
        w2v = file_archive(w2v_path,
                           serialized = True,
                           cached = True,
                           memmode='r',
                           memsize = MEM_SIZE / 4)
        w2v.load()
        w2v_mat = w2v['w2v']

    return tfidf_mat, w2v_mat

def save_tfidf_matrix(proc_state, id_list, tfidf_matrix, vocab):
    tf_idf_path = get_tfidf_path(proc_state)

    to_save_dict = {'user': proc_state.uid,
                    'tfidf': tfidf_matrix,
                    'vocab': vocab,
                    'ids': id_list}

    tfidf = file_archive(tf_idf_path, to_save_dict,
                         serialized = True,
                         cached = True,
                         memsize = MEM_SIZE / 4)

    tfidf.dump()

def save_w2v_matrix(proc_state, word_to_vec_mat):
    word_vec_path = get_w2v_path(proc_state)
    w2v = file_archive(word_vec_path,
                       cached = False)
    w2v['w2v'] = word_to_vec_mat
    w2v.dump()

def clean_fileset(uid, added_files):

    """
    Because of how Google Drive is set up, it's possible for files to be
    given to us multiple times. This filters out files we've already seen
    or were trashed and should be ignored
    """

    seen_files = File.query.filter_by(user_id=uid).all()
    seen_ids = [f.id for f in seen_files]

    clean_files = [f for f in added_files if (not f in seen_ids and not f['trashed'])]
    return clean_files

def add_file_google(uid, cur_file):

    """
    Takes information about the Google files and converts them to our needed database
    objects.

    Arguments:
      uid (string): The id of the User object in the database
      cur_file (dict): a dictionary of the API response from the Google Drive API

    Returns:
      (File) OR (Folder): database object extracted from API response object

      OR

      (void) if the item has been deleted or is already in the database

    """

    view_link = cur_file['webViewLink']
    last_mod = cur_file.get('modifiedTime')
    if last_mod:
        last_mod = datetime.datetime.strptime(last_mod, '%Y-%m-%dT%H:%M:%S.%fZ')
        last_mod = last_mod.timestamp()

    parent_id = cur_file['parents'][0] if 'parents' in cur_file else 'root'
    parent_hash = "l0_" + get_hash(parent_id) if 'parents' in cur_file else 'l0_'

    # if 'imageMediaMetadata' in cur_file:
    #     media_info = get_google_media(cur_file)

    # Need to make this small enough to fit in the current database architecture

    common_attrs = {'rand_id': get_rand_id(),
                    'user_id': uid,
                    'id' : cur_file['id'],
                    'name': trunc_str(cur_file['name'], NAME_PATCH_LEN),
                    'original_name': trunc_str(cur_file['name'], NAME_PATCH_LEN),
                    'path': '', # This gets fixed after the whole collection is downloaded
                    'path_hash': ROOT_PHASH + get_hash(cur_file['id']),
                    'parent_id': parent_id,
                    'parent_hash': parent_hash,
                    'original_parent_id': parent_id,
                    'original_parent_hash': parent_hash,
                    'size': cur_file.get('size'),
                    'last_modified': last_mod,
                    'is_shared': cur_file.get('shared'),
                    'is_owner': cur_file.get('ownedByMe'),
                    'created_by_study': False,
                    'created_time': int(time.time()),
                    'view_link': cur_file['webViewLink'],
                    'error': False} # TODO: error handling

    if not cur_file['mimeType'] == 'application/vnd.google-apps.folder':
        ext = cur_file.get('fullFileExtension')
        f = File(file_extension = ext.lower() if ext else ext,
                 goog_mime_type = cur_file.get('mimeType'),
                 elf_mime_type = cur_file.get('mimeType'), # TODO: add the conversion here
                 media_info = json.dumps(cur_file.get('imageMediaMetadata')),
                 is_gdoc = 'size' not in cur_file,
                 **common_attrs)
    else:
        f = Folder(**common_attrs)

    return f


def retrieve_files(proc_state, page_token):
    if page_token:
        all_files = proc_state.drive_obj.files().\
            list(pageSize=100,fields="*", pageToken=page_token).execute()
    else:
        all_files = proc_state.drive_obj.files().list(pageSize=100,fields="*").execute()

    return all_files

def act_request(proc_state, page_token):

    if page_token:
        all_acts = proc_state.drive_obj.activity().\
            query(body={'pageSize': 100,
                        'pageToken': page_token}).execute()
    else:
        all_acts = proc_state.drive_obj.activity().\
            query(body={'pageSize': 100}).execute()

    return all_acts

def retrieve_activities(proc_state, page_token):

    all_acts = None
    curr_tries = 0
    RETRIES = 5

    for _ in range(RETRIES):

        try:
            all_acts = act_request(proc_state, page_token)
        except GoogHttpError as e:
            if int(e.resp.status) == 429 or int(e.resp.status) == 403: #Rate limit error
                all_acts = None
                time.sleep(60)
            else:
                raise e

    return all_acts


def save_perms(uid, to_process):
    """
    Save the shared users for each file to the SharedUsers table

    """

    file_set = []

    for cur_file in to_process:

        perms = [x.get('emailAddress', '')
                 for x in cur_file.get('permissions', [])
                 if 'emailAddress' in x and x.get('role', 'owner') != 'owner']

        perms = set(list(perms)) #Unique permissions

        filz = [SharedUsers(rand_id = get_rand_id(),
                            user_id=uid,
                            id=cur_file['id'],
                            shared_user=shared) for shared in perms]

        file_set.extend(filz)

    db.session.bulk_save_objects(file_set)

    return file_set

def save_image_feats(uid, to_process):

    thumbnails_links = get_image_links_google(uid, to_process)
    db.session.bulk_save_objects(get_image_labels(uid, thumbnails_links))


def batch_update_file_set(page_token, file_collection, proc_state):

    """
    We receive files from the Google API in batches, and need to update the
    set of returned files, and the plaintext names in memory

    This updates state.

    Arguments:
      page_token (str): nextPageToken attribute from previous
      file_collection (dict): holds the File and Folder objects we extracted
      proc_state (ProcState): process state such as current user, client object

    Returns:
      (string): the token needed to accept the next page of files from the Google Drive
                API response

      OR

      (None)

    """

    uid = proc_state.uid
    start_time = time.time()

    # Failed requests due to expiration of access tokens are automatically retried
    # after tokens automatically refresh
    all_files = retrieve_files(proc_state, page_token)

    curr_time = (time.time() - start_time)
    num_files_ret = len(all_files['files'])
    print("*"*35, num_files_ret,"files returned by the Google API in", curr_time,"seconds","#"*35)

    start_time = time.time()

    to_process = clean_fileset(uid, all_files['files'])

    added_files = {f['id']: add_file_google(uid, f) for f in to_process}
    file_collection.update(added_files)

    save_perms(uid, to_process)
    if not LOCAL_IMG_FEATS:
        save_image_feats(uid, to_process)

    curr_time = (time.time() - start_time)
    files_proc = len(added_files)
    print("$"*35, files_proc,"files processed by comanager in", curr_time,"seconds","#"*35)

    if 'nextPageToken' in all_files:
        return all_files['nextPageToken']
    else:
        return None


def batch_update_activities(page_token, activities, proc_state):

    """
    Works the same as batch_update_file_set, but on activities
    """

    uid = proc_state.uid
    start_time = time.time()

    # Failed requests due to expiration of access tokens are automatically retried
    # after tokens automatically refresh
    all_acts = retrieve_activities(proc_state, page_token)
    if all_acts is not None:
        page_token = all_acts.get('nextPageToken', None)
    else:
        return None

    curr_time = (time.time() - start_time)
    num_acts_ret = len(all_acts.get('activities', []))
    print("*"*35, num_acts_ret,"activities returned by the Google API in", curr_time,"seconds","#"*35)
    start_time = time.time()

    temp_acts = []
    for a in all_acts.get('activities', []):
        try:
            ingested_act = ingest_action(proc_state.uid, a)
            temp_acts.append(ingested_act)
        except Exception as e:
            cl, msg, tb = sys.exc_info()

            print("Unable to ingest action for user {}: \n{} = {} on {}".\
                  format(proc_state.uid, str(cl), e, tb.tb_lineno))

    all_acts = list(chain.from_iterable(temp_acts))

    activities.extend(all_acts)

    return page_token

class ProcState:

    """
    Contains assorted variables dealing with the ongoing extraction from the
    cloud account
    """

    def __init__(self, uid, service='drive', v = 'v3'):

        self.uid = uid
        self.user = None
        self.v = v
        self.service = service
        self.init_service()

    def init_service(self):
        raise NotImplementedError

class GoogProcState(ProcState):

    def init_service(self):

        """
        Initialize the API client objects and other Google-specific state
        """

        self.user = User.query.filter_by(id=self.uid).first()
        self.drive_obj = create_service_object(self.uid, "minimal",
                                               self.service, self.v)
        self.is_dbx = False

    def set_complete(self):
        self.user = User.query.filter_by(id=self.uid).first()
        self.user.info_loaded = True
        db.session.commit()

    @property
    def access_token(self):
        return self.user.access_token

    @property
    def refresh_token(self):
        return self.user.refresh_token

    def refresh_drive(self):
        self.drive_obj = refresh_google_token(self)

class TextState:

    def __init__(self, uid, num_files, num_thr = 1):

        self.num_thr = num_thr

        self.processQ = queue.Queue(self.num_thr + num_files)
        self.thrd_list = [ProcessTextThread(uid, self.processQ) for i in range(self.num_thr)]
        for thr in self.thrd_list:
            thr.start()

    def finalize(self):
        for j in range(self.num_thr):
            self.processQ.put(None)
        self.processQ.join()

    def put(self, to_process):
        self.processQ.put(to_process)



class FileWrapper:

    """
    Carries extra characteristics of an individual DB File object that we don't
    want to persist.
    """

    def __init__(self, base_file, uid):

        text_types = set(['txt', 'doc', 'docx', 'rtf', 'dotx', 'dot', 'odt',
                          'pages', 'tex', 'pdf', 'ps', 'eps', 'prn'])
        image_types = set(['jpg', 'jpeg', 'png', 'tiff', 'tif', 'gif', 'bmp'])
        spread_types =  set(['tsv', 'csv', 'xls', 'xlsx', 'xltx', 'xlt', 'ods',
                             'xlsb', 'xlsm', 'xltm'])

        ACCEPTABLE_SIZE = 1e8

        self.has_downloadable_content = True
        self.usable_text = False

        self.is_gdoc = getattr(base_file, "is_gdoc", False)
        self.id = getattr(base_file, "id", None)
        self.uid = uid

        self.ext = getattr(base_file, "file_extension", "")
        self.elf_mime_type = getattr(base_file, "elf_mime_type", "")
        self.fname = base_file.name
        self.size = base_file.size
        self.last_modified = base_file.last_modified
        self.is_owner = base_file.is_owner
        self.parent_id = base_file.parent_id
        self.path = base_file.path

        self.is_text = self.ext in text_types
        self.is_image = self.ext in image_types
        self.is_spread = self.ext in spread_types

        self.gen_type = "text" if self.is_text else ("media" if self.is_image else "other")

        self.ok_size = base_file.size <= ACCEPTABLE_SIZE

        self.fb = None
        self.tried_to_download = False

        self.text = ""
        self.minhash_content = None
        self.web_labels = []
        self.susers = []

        self.im_rep = None
        self.text_rep = None
        self.w2v_rep = None


    def dump_vec_feats(self):

        users = SharedUsers.query.filter(SharedUsers.user_id==self.uid,
                                         SharedUsers.id == self.id).all()

        img_feat = ImageFeats.query.\
            filter_by(user_id=self.uid).\
            filter_by(id=self.id).one_or_none()

        avg_color = img_feat.avg_color if img_feat is not None else None
        web_labels = img_feat.web_labels if img_feat is not None else None

        return {'id': self.id,
                'last_modified': self.last_modified,
                'size': self.size,
                'color': avg_color,
                'fname': self.fname,
                'path': self.path,
                'text_rep': self.text_rep,
                'tfidf_rep': getattr(self, 'tfidf_rep'),
                'im_rep': self.im_rep,
                'text': self.text,
                'shares': [u.shared_user for u in users],
                'web_labels': web_labels,
                'minhash_content': self.minhash_content}

    def get_base_file(self):

        f = File.query.filter_by(user_id = self.uid,
                                 id = self.id).one()

        return f

    @threaded_cached_property
    def errored(self):
        f = self.get_base_file()
        return f.error

    # @threaded_cached_property
    # def path(self):
    #     f = self.get_base_file()
    #     return f.path

    @threaded_cached_property
    def name(self):
        f = self.get_base_file()
        return f.name

    @threaded_cached_property
    def file_type(self):
        f = self.get_base_file()
        return f.file_type

    @threaded_cached_property
    def mime_type(self):
        f = self.get_base_file()
        return f.goog_mime_type

    @threaded_cached_property
    def sel_for_simils(self):
        f = self.get_base_file()
        return f.sel_for_simils

    def get_image_feat(self, proc_state):
        raise NotImplementedError

    def get_shared_users(self, proc_state):
        susers = SharedUsers.query.filter_by(user_id = proc_state.uid,
                                             id = self.id).all()
        self.susers = [f.shared_user for f in susers]


    def get_doc_meta(self):

        assert self.tried_to_download

        if self.has_downloadable_content:
            f = self.get_base_file()
            f.doc_metadata = get_metadata(self.fb, self.ext)
            db.session.commit()


    def get_text(self, proc_state):

        #Load the file into the queue to process its text
        #Bow out if the file is too large

        assert self.tried_to_download

        if self.is_image:

            image_feat = self.get_image_feat(proc_state)

            if image_feat is None:
                return

            if (image_feat.text != "None") and (image_feat.text is not None):
                self.text = copy.deepcopy(image_feat.text)

        elif self.has_downloadable_content and self.ok_size:
            #text_state.put(self)
            text_proc_celery(self)

    def local_img_feats(self, proc_state):

        """
        Dead code
        """

        assert self.tried_to_download
        if self.has_downloadable_content and self.is_image:
            im_feat = local_image_feats(self.id, proc_state.uid, self.fb)
            db.session.add(im_feat)

    def finalize_usable_text(self):

        f = self.get_base_file()

        is_usable = self.text != ""
        self.usable_text = is_usable
        f.usable_text = is_usable
        #https://www.elastic.co/guide/en/elasticsearch/reference/current/general-recommendations.html
        if self.usable_text:
            self.text = self.text[:TOP_TEXT_SIZE]

        db.session.commit()

    def pull_schema(self, proc_state):
        raise NotImplementedError

    def init_minhash(self, proc_state):

        NUM_PERM = 128
        self.minhash_content = MinHash(num_perm=NUM_PERM)

        CHUNK_SZ = 64

        if self.has_downloadable_content and self.ok_size:
            for chunk_range in range(0, len(self.fb), CHUNK_SZ):
                self.minhash_content.update(self.fb[chunk_range:chunk_range + CHUNK_SZ])

    def json_rep(self):

        items_to_rep = ['has_downloadable_content',
                        'usable_text',
                        'is_gdoc',
                        'id',
                        'uid',
                        'ext',
                        'elf_mime_type',
                        'fname',
                        'is_owner',
                        'is_text',
                        'is_image',
                        'is_spread',
                        'gen_type',
                        'ok_size',
                        'tried_to_download']

        return {k: getattr(self, k, None) for k in items_to_rep}

    @classmethod
    def from_json(cls, jsn):

        uid = jsn['uid']
        base_file_id = jsn['id']

        base_file = File.query.filter_by(user_id=uid,
                                         id=base_file_id).one()
        obj = cls(base_file, uid)

        for k, v in jsn.items():
            setattr(obj, k, v)

        return obj
    def elastic_rep(self, uid):

        elastic_text = self.text

        body =  {"id": self.id,
                 "timestamp": self.last_modified,
                 "text": elastic_text,
                 "mime": self.elf_mime_type,
                 "fname": self.fname,
                 "owner": self.is_owner,
                 "gentype": self.gen_type
        }


        return {'index': uid,
                'id': self.id,
                'body': body}

    def __repr__(self):
        return "Id: {}\n".format(self.id) + \
            "Name: {}\n".format(self.fname) + \
            "Has downloadable content: {}\n".format(self.has_downloadable_content) + \
            "Has usable text: {}\n".format(self.usable_text) + \
            "Is gdoc?: {}\n".format(self.is_gdoc) + \
            "Extension: {}\n".format(self.ext) + \
            "Is image?: {}\n".format(self.is_image) + \
            "Is spreadsheet?: {}\n".format(self.is_spread) + \
            "Is ok size?: {}\n".format(self.ok_size) + \
            "Tried to download?: {}\n".format(self.tried_to_download) + \
            "Text: {}\n\n".format("") + \
            "Text representation: {}\n\n".format(self.text_rep) + \
            "Image representation: {}\n\n".format(self.im_rep)

class GoogFileWrapper(FileWrapper):

    def pull_schema(self, proc_state):

        if self.has_downloadable_content and self.ok_size and self.is_spread:
            schema = schema_extract(self.id, self.fb, self.ext)

            to_add = [Schemas(rand_id = get_rand_id(),
                              file_id = self.id,
                              user_id = proc_state.uid,
                              feat = head_item)
                      for head_item in schema]

            db.session.bulk_save_objects(to_add)
            db.session.commit()

    def get_image_feat(self, proc_state):
        return ImageFeats.query.\
            filter_by(user_id = proc_state.uid, id = self.id).one_or_none()

class FwrapIter:

    """
    In order to pass file wrapper processing via Celery queues,
    we need an iterator to provide several ugly arguments.
    We abstract the complexity of that through this
    """

    def __init__(self, all_wrappers, proc_state):
        self.all_wrappers = all_wrappers
        self.uid = proc_state.uid
        self.curr_iter = enumerate(self.all_wrappers.values())

    def __iter__(self):
        return self

    def __next__(self):
        i, fwrap = next(self.curr_iter)
        return (i, len(self.all_wrappers), self.uid, fwrap.json_rep())

        # if i < 3:
        #     return (i, len(self.all_wrappers), self.uid, fwrap.json_rep())
        # else:
        #     return next(iter([]))

#TODO: Handle formats not in format dict
def media_download(client, fwrap):
    """
    Selects the proper download function for a file depending on whether it is a gdoc
    Arguments:
      client (WrappedClient): Wrapped Google Drive client object to call API functions
      fwrap (FileWrapper): wrapper for various file content features that are not persisted

    Returns:
      (BytesArray?): Bytes associated with file
    """

    gdoc_mapping = {"application/vnd.google-apps.document": 'text/plain',
                    "application/vnd.google-apps.photo": 'image/jpeg',
                    "application/vnd.google-apps.drawing": 'image/jpeg',
                    "application/vnd.google-apps.presentation":
                    'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                    "application/vnd.google-apps.script": 'application/vnd.google-apps.script+json',
                    "application/vnd.google-apps.spreadsheet": 'text/csv'}

    if fwrap.is_gdoc:

        dl_mime_type = gdoc_mapping.get(fwrap.mime_type, "application/pdf")

        return client.files().\
            export_media(fileId=fwrap.id, mimeType = dl_mime_type).execute()
    else:
        request = client.files().get_media(fileId=fwrap.id)
        request.headers["Range"] = "bytes={}-{}".format(0, 0 + DL_LIMIT)
        return request.execute()


def goog_download_fb(proc_state, fwrap):

    """
    Download file content

    Arguments:
      proc_state (ProcState): associated process state for individual user
      fwrap (FileWrapper): wrapper for various file content features that are not persisted

    Returns:
      (None): mutates state of fwrap
    """

    fwrap.tried_to_download = True

    if not fwrap.errored:

        try:
            fwrap.fb = media_download(proc_state.drive_obj, fwrap)
        except HTTPError as e:
            logging.error("HTTPerror for Google: {} = {}, \n{}".format(e.code, e.reason, e.headers))
            fwrap.has_downloadable_content = False
    else:
        fwrap.has_downloadable_content = False

def goog_process_fwrap(proc_state, fwrap):

    goog_download_fb(proc_state, fwrap)
    general_process_fwrap(proc_state, fwrap, is_dbx = False)

def general_process_fwrap(proc_state, fwrap, is_dbx):

    """
    Associate additional information with file wrappers

    Arguments:
      proc_state (ProcState): associated process state for individual user
      fwrap (FileWrapper): wrapper for various file content features that are not persisted
      is_dbx (Boolean): True if the file is taken from a Dropbox repo

    Returns:
      (None): mutates state of fwrap


    """

    fwrap.get_text(proc_state)
    fwrap.pull_schema(proc_state)
    fwrap.init_minhash(proc_state)
    fwrap.get_doc_meta()

def goog_process_error(uid, fwrap, e):

    cl, msg, tb = sys.exc_info()
    db.session.merge(FileErrors(user_id = uid, file_id = fwrap.id,
                                error = str(cl), lineno = tb.tb_lineno))
    f = fwrap.get_base_file()
    f.error = True
    db.session.commit()
    logging.error("Error for user {} with file {}. \nError: {} = {}, on {}".\
                  format(uid, fwrap.id, str(cl), e, tb.tb_lineno))

def has_subdirs(uid, f_obj):

    """
    When returning information about subdirectories, we need to return whether they themselves
    have subdirectories, which is a database query for each subfolder.

    Arguments:
      uid (string): the ID of the user that the folder in question belongs to
      f_obj (dict): a dictionary object corresponding to a Folder object from the database

    Returns:
      (bool): whether or not the folder in question has subdirectories

    """

    folds = [r for r in Folder.query.filter_by(user_id=uid,
                                               parent_hash=f_obj['hash'],
                                               trashed=False)]
    return 1 if len(folds) > 0 else 0

def cwd_and_subfolders(uid, tree=False, target=''):

    """

    We need to do some processing to make the folders and the current working directory
    returns sensible (i.e. finding subdirectories), and in order to avoid database
    processing in the utils file, we include the function here. This is used primarily
    for the open function in connector

    """

    to_find_cwd = target if target else 'l0_'

    cwd = folder_ret(Folder.query.filter_by(user_id=uid, path_hash=to_find_cwd).one())
    cwd['dirs'] = has_subdirs(uid, cwd)

    folds = [folder_ret(r) for r in Folder.query.filter_by(user_id=uid,
                                                           parent_hash=cwd['hash'],
                                                           trashed=False).all()]
    for item in folds:
        item['dirs'] = has_subdirs(uid, item)

    if tree:
        root_fold = folder_ret(Folder.query.filter_by(user_id=uid, name=ROOT_NAME).one())
        root_fold['dirs'] = has_subdirs(uid, root_fold)

        folds.append(root_fold)

    return cwd, folds


def fix_file_paths(f_objs):

    """

    Before loading files and folders into the database, we need to accomplish two things:

      1. Create their full file paths. We can't do this until we've downloaded the whole collection
      2. Link orphaned folders to the root. This happens in Drive, where items have a parent
         folder that never gets downloaded (like "My Drive") and we need to link that to
         our browser

    The algorithm works by selecting objects from the set to be added to the database,
    and attempting to resolve its path via checking if its parent's path is resolved.
    Three things may occur:
      1. The parent's path has been resolved. We then resolve our current object's
         path
      2. The object's parent does not exist. This is possible because some Google
         Drive objects live in odd places like an Applications folder. We are
         not able to download these Applications folders, and therefore have
         no information about them. We therefore place these "orphans" under
         the root directory, resolving their path.
      3. The parent exists, but its path has not been resolved. We therefore
         add the current child's ID to a "daisy chain" called parent_chain that
         tracks items to be resolved next after the current item is resolved, and
         change our current object to the parent object WITHOUT RESOLVING THE PATH

    In cases 1 and 2, we can remove the ID of the current object from the
    to_resolve variable. If there are no subsequent IDs to resolve from the
    parent_chain list, we randomly select one from the remaining ones in to_resolve.
    However, if there is an object in the parent chain that isn't our current
    object, we pop that off and resolve that next. This continues until
    there are no more paths to resolve.

    Arguments:
      f_objs (dict): mapping of id -> File / Folder object to be inserted into DB

    Returns:
      (dict): The same dictionary with mutated state, such that all items have full file paths

    """

    if not f_objs:
        return f_objs

    # Make this deterministic
    random.seed(0)

    f_objs = {k:v for k,v in f_objs.items() if v is not None}

    to_resolve = set(f_objs.keys())
    to_resolve.remove(ROOT_ID)
    resolved = {ROOT_ID: f_objs[ROOT_ID]} # Root path is already resolved

    parent_chain = []  # "Daisy chain" of next items to resolve

    # Initialize by selecting random object
    curr_id = random.sample(to_resolve, 1)[0]
    curr_obj = f_objs[curr_id]

    while to_resolve:

        pid = curr_obj.parent_id

        parent_is_resolved = pid in resolved
        parent_couldnt_dl = pid not in f_objs

        if parent_is_resolved:
            curr_obj.path = resolved[pid].path + '/' + curr_obj.name
            resolved[curr_id] = curr_obj
            to_resolve.remove(curr_id)
        elif parent_couldnt_dl:
            curr_obj.parent_id = ROOT_ID
            curr_obj.parent_hash = ROOT_PHASH
            curr_obj.path = ROOT_PATH + "/" + curr_obj.name
            curr_obj.original_parent_id = ROOT_ID
            curr_obj.original_parent_hash = ROOT_PHASH
            resolved[curr_id] = curr_obj
            to_resolve.remove(curr_id)
        else: # Have to first resolve parent before resolving this
            parent_chain.append(curr_id)

        first_resolve_parent = parent_chain and parent_chain[-1] == curr_id
        now_resolve_children = parent_chain and parent_chain[-1] != curr_id

        if first_resolve_parent:
            curr_id = pid
        elif now_resolve_children:
            curr_id = parent_chain.pop(-1)
        elif to_resolve:
            curr_id = random.sample(to_resolve, 1)[0]

        curr_obj = f_objs[curr_id]

    return resolved
