import pickle
import pprint
import time
import unittest
import flask_testing
import requests
import argparse
import multiprocessing
import random
import socketserver
import socket
import cProfile
import pstats
import graphviz
import subprocess as sp

from collections import Counter
from unittest.mock import patch
from flask import url_for, jsonify, session
from itertools import combinations, product, chain, repeat
from flask_testing import TestCase, LiveServerTestCase
from urllib.parse import urlencode, quote_plus
from sqlalchemy import and_, or_, func

from felfinder import app
from felfinder.config import SQLALCHEMY_DB_TEST_URI
from felfinder.models import (db, User, File, Folder, Simils, CommandFiles,
                              Schemas, HeaderRow, Recommends, ActionSample,
                              GroupExplan)
from felfinder.models import CommandHistory, ActionSample
from felfinder.connector import ElFinderConnector
from felfinder.utils import file_ret, folder_ret, row_conv, get_rand_id
from felfinder.workers import populate_google, fix_file_paths, GoogProcState
from felfinder.workers import swampnet_rep
from felfinder.workers import GoogFileWrapper, google_simils, load_tfidf_w2v
from felfinder.pred_utils import (precompute_preds, get_simils_from_tuples_both)
from felfinder.recommend import swampnet_recommend, simil_recommend
from felfinder.routes import (backend_process, precompute_recs, start_content,
                              generate_tbl_text, generate_explan_html)
from felfinder.sampling import (save_sample_objs, GeneralQualParams,
                                FalsePosParams, MoveQualParams)
from felfinder.workers import GoogProcState, GoogFileWrapper, FwrapIter, init_simils
from felfinder.workers import simil_proc, load_all_wrap, goog_celery_proc, simil_wrapper
from felfinder.workers import collect_results, google_simils
from felfinder.workers import index_and_finalize_text, set_user_complete
from felfinder.workers import (make_text_mats, set_up_qual_routing_pared,
                               set_up_qual_routing_full, set_up_explan_routing)
from felfinder.utils import path_dist, jaccard, bigram_sim, grouper

from utils_testing.faker import ascii_string, rand_word_content, file_gen
from utils_testing.faker import real_gen_file_tree
from utils_testing.fixtures import EMPTY_FUNC, ERROR_FUNC, http_service_mock
from utils_testing.fixtures import REPLACE_PARTIAL_F, HTTP_ERROR_FUNC
from utils_testing.objs import FakeService
from utils_testing.utils import blend_file, blend_folder, blend_user

UID = 'abc'
# SAMP_UID = '1761c5c4a2fc083d9fc8079210382006'

# POSTGRES_USER = "new_swamp"
# POSTGRES_PASS = "62f8bpcazbbuix4d2q7pkdgqu8ngvx5fkakikdtgrwwou7e53pizivhjy4ff8et326a3szvnysjauti"
# POSTGRES_DB = "cloudclippy_full"

# SQLALCHEMY_DATABASE_URI = "postgresql://{}:{}@localhost:5432/{}".\
#     format(POSTGRES_USER, POSTGRES_PASS, POSTGRES_DB)


def can_ping_server(port):

    """
    Determine if the server is live
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(("localhost", port))
    except socket.error as e:
        success = False
    else:
        success = True
    finally:
        sock.close()

    return success


def spawn_live_server(app, port_value):

    """
    Create a separate process with a live Flask application
    for our sqlalchemy database to attach to
    """

    process = None
    options = {'ssl_context': ('fake-server.crt', 'fake-server.key'),
               'threaded': True}

    def worker(app, port):
        # AMC: use the lower one for local dev
        app.run(port=port, use_reloader=False, **options)
        # app.run(port=port, use_reloader=True)

    process = multiprocessing.Process(
        target=worker, args=(app, port_value)
    )

    process.start()

    timeout = app.config.get('LIVESERVER_TIMEOUT', 5)
    start_time = time.time()

    while True:
        elapsed_time = (time.time() - start_time)
        if elapsed_time > timeout:
            raise RuntimeError(
                "Failed to start the server after %d seconds. " % timeout
            )

        if can_ping_server(port_value):
            break

    return process

@patch('felfinder.workers.get_image_labels')
@patch('felfinder.workers.create_service_object')
def mod_backend_process(uid, root_dir,
                        mock_create_service_object, mock_im_labels):

    """
    This function was originally created to try and test the
    memory footprint of the main flow of our application
    """

    start = time.time()

    pr = cProfile.Profile()
    pr.enable()


    mock_create_service_object.return_value = FakeService(
        full_files = real_gen_file_tree(root_dir),
        real_gen = True)
    mock_im_labels.return_value = []

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('tottime')
    ps.print_stats(100)


    first = time.time()

    print("Generated file tree in {0:0.3f}s".format(first - start))
    print()
    print()

    pr = cProfile.Profile()
    pr.enable()


    populate = populate_google(uid)

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('tottime')
    ps.print_stats(100)


    second = time.time()
    print("Populated Gdrive info in {0:0.3f}s".format(second - first))
    print()
    print()

    pr = cProfile.Profile()
    pr.enable()


    proc_state = GoogProcState(uid)
    files_to_process = File.query.filter_by(user_id=uid).all()
    all_wrappers = {f.id: GoogFileWrapper(f, uid) for f in files_to_process}

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('tottime')
    ps.print_stats(100)


    third = time.time()
    print("Generated wrappers in {0:0.3f}s".format(third - second))
    print()
    print()

    pr = cProfile.Profile()
    pr.enable()


    it = FwrapIter(all_wrappers, proc_state)

    task_l = [goog_celery_proc(*arg) for arg in it]

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('tottime')
    ps.print_stats(100)


    fourth = time.time()
    print("Proc'd all wrappers in {0:0.3f}s".format(fourth - third))
    print()
    print()

    pr = cProfile.Profile()
    pr.enable()


    more_uid = collect_results(task_l)

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('tottime')
    ps.print_stats(100)


    fifth = time.time()
    print("Collected results in {0:0.3f}s".format(fifth - fourth))
    print()
    print()

    pr = cProfile.Profile()
    pr.enable()


    more_uid = index_and_finalize_text(uid)

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('tottime')
    ps.print_stats(100)


    sixth = time.time()
    print("Indexed and finalized text (no Elasticsearch) in {0:0.3f}s".format(sixth - fifth))
    print()
    print()

    pr = cProfile.Profile()
    pr.enable()


    proc_state = GoogProcState(uid)
    all_wrappers = load_all_wrap(proc_state)
    init_simils(uid)

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('tottime')
    ps.print_stats(100)


    seventh = time.time()
    print("Initialized simils in {0:0.3f}s".format(seventh - sixth))
    print()
    print()

    pr = cProfile.Profile()
    pr.enable()


    make_text_mats(proc_state, all_wrappers)

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('tottime')
    ps.print_stats(100)


    eighth = time.time()
    print("Made text matrices in {0:0.3f}s".format(eighth - seventh))
    print()
    print()

    full_sims = Simils.query.filter_by(user_id = uid).all()
    full_sim_tups = [(s.file_id_A, s.file_id_B) for s in full_sims]
    full_sim_tups = sorted(full_sim_tups, key=lambda x: x[0])

    GROUPS = 3000
    it = grouper(full_sim_tups,
                 GROUPS, fillvalue = (None, None))

    pr = cProfile.Profile()
    pr.enable()


    for sim_tups in it:
        simil_proc(sim_tups, uid)

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('tottime')
    ps.print_stats(100)

    ninth = time.time()
    print("Computed similarity in {0:0.3f}s".format(ninth - eighth))
    print()
    print()

    precompute_recs([], uid)

    tenth = time.time()
    print("Precomputed recommendations in {0:0.3f}s".format(tenth - ninth))
    print()
    print()


    set_up_explan_routing([], uid)

    set_user_complete([], uid)



@patch('felfinder.workers.get_image_labels')
@patch('felfinder.workers.create_service_object')
def swampnet_backend_process(uid, root_dir,
                        mock_create_service_object, mock_im_labels):

    """
    This function was originally created to try and test the
    memory footprint of the main flow of our application
    """

    start = time.time()

    pr = cProfile.Profile()
    pr.enable()

    mock_create_service_object.return_value = FakeService(
        full_files = real_gen_file_tree(root_dir),
        real_gen = True)
    mock_im_labels.return_value = []

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('tottime')
    #ps.print_stats(100)


    first = time.time()

    print("Generated file tree in {0:0.3f}s".format(first - start))
    print()
    print()

    pr = cProfile.Profile()
    pr.enable()


    populate = populate_google(uid)

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('tottime')
    #ps.print_stats(100)


    second = time.time()
    print("Populated Gdrive info in {0:0.3f}s".format(second - first))
    print()
    print()

    pr = cProfile.Profile()
    pr.enable()


    proc_state = GoogProcState(uid)
    files_to_process = File.query.filter_by(user_id=uid).all()
    all_wrappers = {f.id: GoogFileWrapper(f, uid) for f in files_to_process}

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('tottime')
    #ps.print_stats(100)


    third = time.time()
    print("Generated wrappers in {0:0.3f}s".format(third - second))
    print()
    print()

    pr = cProfile.Profile()
    pr.enable()


    it = FwrapIter(all_wrappers, proc_state)

    task_l = [goog_celery_proc(*arg) for arg in it]

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('tottime')
    #ps.print_stats(100)


    fourth = time.time()
    print("Proc'd all wrappers in {0:0.3f}s".format(fourth - third))
    print()
    print()

    pr = cProfile.Profile()
    pr.enable()


    more_uid = collect_results(task_l)

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('tottime')
    #ps.print_stats(100)


    fifth = time.time()
    print("Collected results in {0:0.3f}s".format(fifth - fourth))
    print()
    print()

    pr = cProfile.Profile()
    pr.enable()


    more_uid = index_and_finalize_text(uid)

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('tottime')
    #ps.print_stats(100)


    sixth = time.time()
    print("Indexed and finalized text (no Elasticsearch) in {0:0.3f}s".format(sixth - fifth))
    print()
    print()

    pr = cProfile.Profile()
    pr.enable()

    swampnet_rep(uid, None, None)

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('tottime')
    #ps.print_stats(100)


    ninth = time.time()
    print("Made swampnet reps {0:0.3f}s".format(ninth- sixth))
    print()
    print()

    set_user_complete([], uid)



@patch('felfinder.workers.get_image_labels')
@patch('felfinder.workers.create_service_object')
def pre_mod_backend_process(uid, root_dir,
                        mock_create_service_object, mock_im_labels):

    """
    Dump the file wrappers for use in training scalable models
    """

    mock_create_service_object.return_value = FakeService(
        full_files = real_gen_file_tree(root_dir),
        real_gen = True)
    mock_im_labels.return_value = []

    populate = populate_google(uid)

    proc_state = GoogProcState(uid)
    files_to_process = File.query.filter_by(user_id=uid).all()
    all_wrappers = {f.id: GoogFileWrapper(f, uid) for f in files_to_process}

    it = FwrapIter(all_wrappers, proc_state)

    task_l = [goog_celery_proc(*arg) for arg in it]

    more_uid = collect_results(task_l)
    more_uid = index_and_finalize_text(uid)
    proc_state = GoogProcState(uid)
    all_wrappers = load_all_wrap(proc_state)

    tfidf_mat, w2v_mat = load_tfidf_w2v(proc_state)

    if tfidf_mat is not None:
        id_list = tfidf_mat['ids']
        id_match = {fid: i for i, fid in enumerate(id_list)}

        for k in all_wrappers.keys():
            all_wrappers[k].tfidf_rep = None
            all_wrappers[k].text_rep = None
            if k in id_match:
                all_wrappers[k].tfidf_rep = tfidf_mat[id_match[k]]
                all_wrappers[k].text_rep = w2v_mat[id_match[k]]

    dump_wrappers = {k: v.dump_vec_feats() for k, v in all_wrappers.items()}

    with open('workspace/full_all_wrap', 'wb') as of:
        pickle.dump(dict(dump_wrappers), of)



def qual_route(uid):

    set_up_qual_routing_full(uid)


def rec_timing_tests(swampnet=False):


    uid = UID
    file_ids = [f.id for f in File.query.all()]
    dst_hashes = [f.path_hash for f in Folder.query.all()]
    net_text = 'Basic' if not swampnet else 'Swampnet'
    rec_func = swampnet_recommend if swampnet else simil_recommend

    for i in [10, 100, 1000]:

        f_samp = random.sample(file_ids, i)
        act = ['move' if random.randint(0, 1) == 1 else 'del' for _ in range(i)]
        dst = ['_' if a != 'move' else random.choice(dst_hashes) for a in act]


        # pr = cProfile.Profile()
        # pr.enable()

        start = time.time()

        for f, a, d  in zip(f_samp, act, dst_hashes):

            rec_func(f, uid, a, {'cmd_id': '_',
                                 'explain_name': '_',
                                 'explain_hash': '_',
                                 'dst': d})

        # pr.disable()
        # ps = pstats.Stats(pr).sort_stats('tottime')
        # ps.print_stats(100)

        end = time.time()

        print("{} took {}s on {} recommendations".format(net_text, end - start, i))


def param_test():

    acts = ActionSample.query.filter(ActionSample.ask_order > -1).all()

    for next_act in acts:

        qual_type = GeneralQualParams
        if "false_pos" in next_act.sample_reason or "untaken" in next_act.sample_reason:
            qual_type = FalsePosParams
        elif "move" in next_act.sample_reason:
            qual_type = MoveQualParams

        qual_obj = qual_type(next_act)

        query_params = qual_obj.to_json()

        print(pprint.pformat(query_params))


def adhoc_gen_tbl_text():

    pass


def adhoc_db_query():

    pass

def main():
    parser = argparse.ArgumentParser(description='Manual testing for elfinder-flask')

    parser.add_argument('--file-tree', default='small_hier_test',
                        dest='file_tree',
                        help='Path to local file tree to load into the DB')
    parser.add_argument('--exp-cond', default='full_recs',
                        dest='exp_cond',
                        help='Path to local file tree to load into the DB')
    parser.add_argument('--uid', default=None,
                        dest='uid',
                        help='User id to load in with')
    parser.add_argument('--load-to-db', default=False,
                        dest='load_to_db', action="store_true",
                        help='If True, load file tree into the DB')
    parser.add_argument('--dump-wrappers', default=False,
                        dest='dump_wrappers', action="store_true",
                        help='If True, dump file wrappers to copy')
    parser.add_argument('--swampnet-load', default=False,
                        dest='swampnet_load', action="store_true",
                        help='If True, loads files to DB in swampnet fashion')
    parser.add_argument('--create-db', default=False,
                        dest='create_db', action="store_true",
                        help='If True, create the DB')
    parser.add_argument('--del-db', default=False,
                        dest='del_db', action="store_true",
                        help='If True, remove the DB')
    parser.add_argument('--launch-window', default=False,
                        dest='launch_window', action="store_true",
                        help='If True, launch test window')
    parser.add_argument('--db-query', default=False,
                        dest='db_query', action="store_true",
                        help='If True, perform short DB query')
    parser.add_argument('--gen-tbl', default=False,
                        dest='gen_tbl', action="store_true",
                        help='If True, perform short DB query')
    parser.add_argument('--qual-route', default=False,
                        dest='qual_route', action="store_true",
                        help='If True, set up full routing')
    parser.add_argument('--route-route', default=False,
                        dest='route_route', action="store_true",
                        help='Test qual_qs route')
    parser.add_argument('--param-test', default=False,
                        dest='param_test', action="store_true",
                        help='Test output of parameters')
    parser.add_argument('--time-sample', default=False,
                        dest='time_sample', action="store_true",
                        help='If True, test the timing on sample objects')
    parser.add_argument('--test-sim-tups', default=False,
                        dest='sim_tups', action="store_true",
                        help='If True, test sampling under high load')
    parser.add_argument('--swampnet-recs', default=False,
                        dest='swampnet_recs', action="store_true",
                        help='If True, tests swampnet recommendations')
    parser.add_argument('--rec-timing', default=False,
                        dest='rec_timing', action="store_true",
                        help='If True, tests timing of recommendations')
    argv = parser.parse_args()

    app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DB_TEST_URI
    #app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI

    db.init_app(app)

    @app.before_request
    def set_session():
        session['user_id'] = argv.uid or UID
        session['exp_cond'] = argv.exp_cond
        session['prolific_id'] = "THEGUY"

    # Get the app
    configured_port = 5000
    port_value = 5000
    #port_value = multiprocessing.Value('i', configured_port)

    ctx = app.test_request_context()
    ctx.push()

    process = None
    try:
        process = spawn_live_server(app, port_value)

        if argv.db_query:
            adhoc_db_query()
            return

        if argv.gen_tbl:
            adhoc_gen_tbl_text()
            return


        if argv.qual_route:
            qual_route(argv.uid or UID)

        if argv.param_test:
            param_test()

        if argv.time_sample:
            timing_test_sampling()
            return

        if argv.rec_timing:
            rec_timing_tests(argv.swampnet_recs)
            return

        if argv.sim_tups:
            test_simil_tuples()
            return

        if argv.del_db:
            db.session.remove()
            db.drop_all()
        if argv.create_db:
            db.create_all()

        u = User.query.filter_by(id=argv.uid or UID).one_or_none()
        if u is None:
            db.session.add(User(id=argv.uid or UID))
            db.session.commit()

        if argv.load_to_db:
            mod_backend_process(argv.uid or UID, argv.file_tree)

        if argv.swampnet_load:
            swampnet_backend_process(argv.uid or UID, argv.file_tree)

        if argv.dump_wrappers:
            pre_mod_backend_process(argv.uid or UID, argv.file_tree)

        if argv.launch_window:
            # AMC: use upper stuff for local dev
            # while True:
            #     iiiii = 0
            _ = sp.run(["chromium-browser", "https://localhost:5000/cm"])

        if argv.route_route:

            resp = sp.run(["curl", "-k", "https://localhost:5000/qual_qs"])

            print(resp)

            # with requests.Session() as sess:
            #     resp = sess.get("https://localhost:5000/qual_qs")
            #     print(resp)
            #     assert resp.status_code == 200



    finally:
        if process:
            process.terminate()
        if ctx:
            ctx.pop()

if __name__ == "__main__":
    main()
