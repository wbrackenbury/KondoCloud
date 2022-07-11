import sys
import os
import bcrypt
import logging
import re
import requests
import random
import time
import pprint
import warnings
warnings.filterwarnings("error", message="Scope has changed from")

from celery import chain as cel_chain, chord, group
from datetime import datetime

from itertools import chain, product
from flask import request, session, redirect, jsonify, send_from_directory, abort, url_for
from flask_cors import cross_origin
from flask_dance.consumer import oauth_authorized
from urllib.parse import urlencode, quote_plus, urlparse, parse_qs
from pprint import pformat

import google.oauth2.credentials
import google_auth_oauthlib.flow

from oauthlib.oauth2.rfc6749.errors import MissingCodeError
from sqlalchemy.orm.exc import NoResultFound

from felfinder import app
from felfinder.connector import ElFinderConnector
from felfinder.models import (db, User, CommandHistory,
                              CommandFiles, File, Folder,
                              DriveHistory, DriveFile,
                              ActionSample, HeaderRow,
                              GroupSample, GroupFiles, GroupExplan)
from felfinder.sampling import (SHORT_NAME_LEN, GeneralQualParams,
                                FalsePosParams, MoveQualParams, DummyQualParams,
                                RecMoveQualParams, TOTAL_NUM_ACTION_QS,
                                TOTAL_NUM_EXPLAN_QS)
from felfinder.utils import (generate_csrf_token, templated, get_hash,
                             scope_flow, rand_prol_id, get_rand_id,
                             even_partition, trunc_str)
from felfinder.workers import (populate_google, replay_all,
                               index_and_finalize_text, activity_logs,
                               collect_results, simil_wrapper,
                               google_simils, set_user_complete,
                               goog_celery_proc, post_proc_wrapper,
                               simil_proc, precompute_recs, swampnet_rep,
                               set_up_qual_routing_pared,
                               set_up_qual_routing_full,
                               set_up_explan_routing)
from felfinder.replay import replay_table_entry
from felfinder.explain_utils import (json_to_exp)
from felfinder.explain import html_explain, size_text
from felfinder.config import (QUALTRICS_FULL_ACTION, QUALTRICS_SUS_LINK,
                              QUALTRICS_DEMO_LINK, QUALTRICS_CONSENT_LINK,
                              PROLIFIC_PART_TWO_END, QUALTRICS_PARED_ACTION,
                              QUALTRICS_EXPLAN_LINK, TEST_SERVER, PARED_PROTOCOL)


NAME_LEN = 25
PATH_LEN = 45

TBL_ROW_PERCS = "28% 43% 18% 10%"

@app.route("/", methods=['GET'])
@templated("home.html")
def index():

    prol_id = request.args.get("ProlificID") or rand_prol_id()
    session['prolific_id'] = prol_id
    session["part_two"] = False
    session["failed"] = False

    current_token = generate_csrf_token()

    if 'csrf_token' not in session:
        session['csrf_token'] = current_token

    return dict(csrf_token = session['csrf_token'])


def scope_route(scope_level):

    flow = scope_flow(redirect_uri = url_for("oauth2callback",
                                             _external = True),
                      scope_level = scope_level,
                      state = None)

    session["scope_level"] = scope_level


    if not 'explan_part_two' in session:
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent')
    else:
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true')

    session['orig_login_state'] = state

    return authorization_url

@app.route("/home", methods=['GET'])
def true_launch():

    if 'prolific_id' not in session:
        session['prolific_id'] = request.args.get("ProlificID") or rand_prol_id()

    session["part_two"] = False
    session["failed"] = False

    current_token = generate_csrf_token()

    if 'csrf_token' not in session:
        session['csrf_token'] = current_token

    return redirect(scope_route("minimal"))

def replay_params_convert(uid, cmd):

    """
    Convert CommandHistory objects to items that we can
    parse in a table such that a participant can
    determine what actions we would be replaying.
    """

    if cmd.cmd == 'paste' or cmd.cmd == 'duplicate':

        cfs = CommandFiles.query.filter_by(user_id=uid,
                                           cmd_id=cmd.rand_id,
                                           recurse_order=0)

        dst = cfs.filter_by(target=False).one()
        orig_dst = Folder.query.filter_by(user_id=uid, id=dst.folder_id).one()

        files = cfs.filter_by(target=True).all()

        rets = []
        for f in files:

            if f.file_id is not None:
                orig_f = File.query.filter_by(user_id=uid,
                                              id=f.file_id).one()
            else:
                orig_f = Folder.query.filter_by(user_id=uid,
                                                id=f.folder_id).one()

            c = {'cmd': cmd.cmd,
                 'cmd_id': cmd.rand_id,
                 'target_name': trunc_str(orig_f.name, SHORT_NAME_LEN),
                 'dst_path': orig_dst.path,
                 'spec_id': get_rand_id()}

            rets.append(c)

        return rets

    elif cmd.cmd == 'rename':
        c = {'cmd': cmd.cmd,
             'cmd_id': cmd.rand_id,
             'old_name': trunc_str(cmd.old_name, SHORT_NAME_LEN),
             'new_name': trunc_str(cmd.new_name, SHORT_NAME_LEN),
             'spec_id': get_rand_id()}
        return [c]

    elif cmd.cmd == 'mkdir':

        c = {'cmd': cmd.cmd,
             'cmd_id': cmd.rand_id,
             'new_name': trunc_str(cmd.new_name, SHORT_NAME_LEN),
             'spec_id': get_rand_id()}
        return [c]

    if cmd.cmd == 'rm':

        files = CommandFiles.query.filter_by(user_id=uid,
                                             cmd_id=cmd.rand_id).all()

        rets = []
        for f in files:

            if f.file_id is not None:
                orig_f = File.query.filter_by(user_id=uid,
                                              id=f.file_id).one()
            else:
                orig_f = Folder.query.filter_by(user_id=uid,
                                                id=f.folder_id).one()

            c = {'cmd': cmd.cmd,
                 'cmd_id': cmd.rand_id,
                 'target_name': trunc_str(orig_f.name, SHORT_NAME_LEN),
                 'spec_id': get_rand_id()}

            rets.append(c)

        return rets

    return []

@app.route("/group_q")
@templated("grouping.html")
def group():

    if "user_id" not in session:
        logging.error("user_id not in session for replay")
        return redirect(url_for("part_two"))

    uid = session["user_id"]

    cmd_list = CommandHistory.query.filter(CommandHistory.user_id == uid,
                                           CommandHistory.cmd != "open",
                                           CommandHistory.cmd != "find").\
                                           order_by(CommandHistory.time_run).all()

    qual_param_list = [replay_params_convert(uid, c) for c in cmd_list]
    qual_param_list = list(chain.from_iterable(qual_param_list))
    t_entries = [replay_table_entry(qp) for qp in qual_param_list]

    child_to_key, key_to_child = {}, {}

    return {"commands": t_entries,
            'child_to_key': child_to_key,
            'key_to_child': key_to_child}

@app.route("/group_post", methods=['POST'])
def group_post():

    if "user_id" not in session:
        logging.error("user_id not in session for route replay_post")
        redirect(url_for("part_two"))

    uid = session["user_id"]

    group_rows = request.form

    process_group_post(uid, group_rows)

    return 'OK'


def recreate_group_rows(group_rows):


    """
    Key values in the form submission look like: cmd_ids[0][row_title]

    Need to turn this into its original arrays of dicts
    """

    rec_d = {}

    for k, v in group_rows.items():

        re_group = re.match("cmd_ids\[([0-9]*?)\]\[([A-Za-z\_]*?)\]", k)
        ind = re_group.group(1)
        attr = re_group.group(2)
        if ind not in rec_d:
            rec_d[ind] = {}
        rec_d[ind][attr] = v

    full_l = []
    list_len = max([int(i) for i in rec_d]) + 1
    for i in range(list_len):
        full_l.append(rec_d[str(i)])

    return full_l


def process_group_post(uid, group_rows):

    """
    The grouping route returns a list of items and their associated header
    rows in a peculiar format that we have to turn into something sensible
    to understand which commands were grouped into the same high-level
    organizational tasks
    """

    group_rows = recreate_group_rows(group_rows)

    assoc_header_id = None

    for row in group_rows:

        if row['row_type'] == 'header':
            assoc_header_id = get_rand_id()

            header_obj = HeaderRow(rand_id = assoc_header_id,
                                   user_id = uid,
                                   header_title = row['row_title'],
                                   header_explain = row['explain'])

            db.session.add(header_obj)

        if row['row_type'] == 'data' and assoc_header_id is not None:

            ch_obj = CommandHistory.query.filter_by(user_id=uid,
                                                    rand_id=row['command']).one()
            ch_obj.assoc_header = assoc_header_id

    db.session.commit()



@app.route("/start_content")
@templated("loading.html")
def start_content():

    if 'user_id' not in session:
        logging.error("user_id not in session for route start_content")
        return redirect(url_for('index'))

    uid = session["user_id"]
    user = User.query.filter_by(id=uid).one()
    if user.end_org is None:
        user.end_org = int(time.time())
    num_action_qs = user.num_action_qs
    exp_cond = user.exp_condition

    # Redirect if survey is completed
    if num_action_qs >= TOTAL_NUM_ACTION_QS:
        params = {'ProlificID': session['prolific_id']}
        return redirect(PROLIFIC_PART_TWO_END)
        #return redirect(QUALTRICS_SUS_LINK + "?" + urlencode(params))

    # Select the next action to ask about or end

    user_acts = ActionSample.query.filter_by(user_id=uid).all()

    if num_action_qs == 0 and len(user_acts) < 1:
        if TEST_SERVER:
            if PARED_PROTOCOL or exp_cond == 'reg_int':
                set_up_qual_routing_pared(uid)
            else:
                set_up_qual_routing_full(uid)
        else:
            if PARED_PROTOCOL or exp_cond == 'reg_int':
                set_up_qual_routing_pared.delay(uid)
            else:
                set_up_qual_routing_full.delay(uid)

    return {"forward_value": False}


@app.route("/get_active")
def get_active():

    """
    Route called as a temporary stopgap while we load
    the files to be sampled
    """

    uid = session["user_id"]
    user = User.query.filter_by(id=uid).one()
    if user.end_org is None:
        user.end_org = int(time.time())
    num_action_qs = user.num_action_qs

    if num_action_qs > 0:
        return jsonify({'forward_value': True})

    next_act = ActionSample.query.\
        filter_by(user_id = uid,
                  ask_order = num_action_qs).one_or_none()

    return jsonify({'forward_value': next_act is not None})


@app.route("/qual_qs", methods=['GET'])
def qual_qs_routing():

    if 'user_id' not in session:
        logging.error("user_id not in session for route qual_qs")
        return redirect(url_for('index'))

    if 'on_questions' not in session:
        session['on_questions'] = True

    uid = session["user_id"]
    user = User.query.filter_by(id=uid).one()
    if user.end_org is None:
        user.end_org = int(time.time())
    num_action_qs = user.num_action_qs
    exp_cond = user.exp_condition

    # Redirect if survey is completed

    params = {'ProlificID': session['prolific_id'],
              'exp_cond': exp_cond}
    if num_action_qs >= TOTAL_NUM_ACTION_QS:
        db.session.commit()
        return redirect(QUALTRICS_SUS_LINK + "?" + urlencode(params))

    next_act = ActionSample.query.\
        filter_by(user_id = uid,
                  ask_order = num_action_qs).one_or_none()
    if next_act is None:
        db.session.commit()
        return redirect(QUALTRICS_SUS_LINK + "?" + urlencode(params))

    # Select method to select query parameters
    qual_type = GeneralQualParams
    if "false_pos" in next_act.sample_reason or "untaken" in next_act.sample_reason:
        qual_type = FalsePosParams
    elif "move" in next_act.sample_reason:
        qual_type = MoveQualParams
    elif next_act.sample_reason == "dummy":
        qual_type = DummyQualParams

    qual_obj = qual_type(next_act)

    query_params = qual_obj.to_json()
    base_link = QUALTRICS_FULL_ACTION if exp_cond == 'full_recs' else QUALTRICS_PARED_ACTION
    link = base_link + "?" + urlencode(query_params)
    user.num_action_qs += 1

    db.session.commit()

    return redirect(link)


#@app.route("/temp_front_explan_qs", methods=['GET'])


@app.route("/explan_qs", methods=['GET'])
def explan_qs_routing():

    if 'user_id' not in session:
        logging.error("user_id not in session for route explan_qs")
        return redirect(url_for('explan_part_two'))

    # if 'on_questions' not in session:
    #     session['on_questions'] = True

    uid = session["user_id"]
    user = User.query.filter_by(id=uid).one()
    # if user.end_org is None:
    #     user.end_org = int(time.time())
    num_action_qs = user.num_action_qs

    # Redirect if survey is completed
    if num_action_qs >= TOTAL_NUM_EXPLAN_QS:
        db.session.commit()
        return redirect(PROLIFIC_PART_TWO_END)

    next_act = GroupExplan.query.\
        filter_by(user_id = uid,
                  sel_for_qs_ind = num_action_qs).one_or_none()

    if next_act is None:
        db.session.commit()
        return redirect(PROLIFIC_PART_TWO_END)

    query_params = {"ind": num_action_qs, "exptype": next_act.exp_type,
                    'uid': uid, 'qnum': num_action_qs + 1}

    link = QUALTRICS_EXPLAN_LINK + "?" + urlencode(query_params)
    user.num_action_qs += 1

    db.session.commit()

    return redirect(link)



@app.route("/replay_post", methods=['POST'])
def replay_post():

    if "user_id" not in session:
        logging.error("user_id not in session for route replay_post")
        redirect(url_for("part_two"))

    uid = session["user_id"]

    cmd_ids = request.form.getlist("cmd_ids[]")

    cmds = CommandHistory.query.filter(CommandHistory.user_id == uid,
                                      CommandHistory.rand_id.in_(cmd_ids)).all()

    for c in cmds:
        c.sel_for_replay = True

    db.session.commit()

    return 'OK'

@app.route("/replay_login", methods=['GET'])
def replay_login():

    session["replay"] = True

    return redirect(scope_route("replay"))

@app.route("/oauth2callback", methods=['GET'])
def oauth2callback():

    if 'orig_login_state' not in session or 'scope_level' not in session:
        logging.error("orig_login_state or scope_level not in session for oauth2callback")
        return redirect(url_for('index'))

    state = session['orig_login_state']
    scope_level = session["scope_level"]

    flow = scope_flow(redirect_uri = url_for('oauth2callback',
                                             _external=True),
                      scope_level = scope_level,
                      state = state)

    authorization_response = request.url
    try:
        flow.fetch_token(authorization_response=authorization_response)
    except Warning:
        return redirect(url_for('login_fail'))
    except MissingCodeError:
        return redirect(url_for('login_fail'))

    credentials = flow.credentials
    google_logged_in(credentials, scope_level)

    return redirect(url_for('login_routing'))

@app.route("/login_fail", methods=['GET'])
@templated("login_fail.html")
def login_fail():

    """
    If a user fails out on logging in to part 1 because
    they didn't enable all the permissions
    """

    return dict()


@app.route("/login_routing", methods=['GET'])
def login_routing():

    if 'orig_login_state' not in session or 'scope_level' not in session:
        logging.error("orig_login_state or scope_level not in session for login_routing")
        return redirect(url_for('index'))

    if "replay" in session:
        replay_all.delay(session["user_id"])
        return redirect(PROLIFIC_PART_TWO_END)

    if session["failed"]:
        return redirect(url_for("find_fail"))

    if session["info_loaded"] and 'part_two' in session and session['part_two']:
        if not session.get('on_questions', False):
            return redirect(url_for("cm"))
        else:
            return redirect(url_for("qual_qs_routing"))
    elif session['info_loaded'] and 'explan_part_two' in session:
        return redirect(url_for("explan_qs_routing"))
    else:
        params = {'ProlificID': session['prolific_id']}
        return redirect(QUALTRICS_DEMO_LINK + "?" + urlencode(params))

@app.route("/replay_end", methods=['GET'])
def replay_end():

    """
    Want to add basic validation to stop someone from just finding the
    route and then accepting the Prolific reward
    """

    if "user_id" in session:
        return redirect(PROLIFIC_PART_TWO_END)
    else:
        return redirect(url_for("part_two"))


@app.route("/full_consent", methods=['GET'])
def full_consent():

    if 'prolific_id' not in session:
        logging.error("prolific_id not in session for full_consent")
        return redirect(url_for('index'))

    params = {'ProlificID': session['prolific_id']}
    return redirect(QUALTRICS_CONSENT_LINK + "?" + urlencode(params))

@app.route("/temp_end", methods=['GET'])
@templated("thanks.html")
def thanks():
    return dict()

@app.route("/find_fail", methods=['GET'])
@templated("find_fail.html")
def find_fail():
    return dict()

@app.route("/policy", methods=['GET'])
@templated("policy.html")
def policy():
    return dict()

@app.route("/instructions", methods=['GET'])
@templated("instruct.html")
def instruct():

    condition = session.get('condition', 'no_recs')
    ordering = session.get('ordering', 'del_first')
    if condition == 'no_recs':
        post_text_one = ""
    else:
        post_text_one = ", but with some enhancements that will be explained when you navigate to the interface."

    act_text_one = "deleting files"
    act_text_two = "moving files, creating folders, and renaming items"

    if ordering != "del_first":
        act_text_one, act_text_two = act_text_two, act_text_one

    return {'post_text_one': post_text_one,
            'act_text_one': act_text_one,
            'act_text_two': act_text_two}


@app.route("/google90610295932759ab.html", methods=['GET'])
@templated("google90610295932759ab.html")
def google_verify():
    return dict()

@app.route("/consent", methods=['GET'])
@templated("explan-consent-full.html")
def consent():
    return dict()


def row_html(row):

    (name, path, link, lastmod, size) = row

    row_head = '<span style="margin-left: 3%; height: 20px; display: grid; grid-template-columns: {}; grid-gap: 2px">'.format(TBL_ROW_PERCS)

    first_data = '<a href="{}" target="_blank">{}</a>'.\
        format(link, trunc_str(name, NAME_LEN))
    first = '<span style="grid-column: 1/4; grid-row: 1/1;" title="{}">{}</span>'.format(name, first_data)

    mid = '<span style="grid-column: 2/4; grid-row: 1/1;" title="{}">{}</span>'.\
        format(path, trunc_str(path, PATH_LEN))

    second_data = datetime.utcfromtimestamp(lastmod).strftime('%Y-%m-%d, %H:%M')
    second = '<span style="grid-column: 3/4; grid-row: 1/1;">{}</span>'.format(second_data)
    third = '<span style="grid-column: 4/4; grid-row: 1/1; text-align: right;">{}</span>'.format(size_text(size))

    row_end = '</span>'

    return '\n'.join([row_head, first, mid, second, third, row_end])

def generate_tbl_text(uid, ge):

    gid = ge.group_id
    gfs = GroupFiles.query.filter_by(user_id = uid, group_id = gid,
                                     explan_id = ge.explan_id)


    # if ge is not None:
    #     gfs = GroupFiles.query.filter_by(user_id = uid, group_id = gid,
    #                                      explan_id = ge.explan_id)
    # else:
    #     gfs = GroupFiles.query.filter_by(user_id = uid, group_id = gid,
    #                                      explan_id = "no_exp")

    fids = [f.file_id for f in gfs]
    rel_files = File.query.filter_by(user_id = uid).filter(File.id.in_(fids)).all()
    row_data = [(f.name, f.path, f.view_link, f.last_modified, f.size) for f in rel_files]

    top = '<div style="margin-left: auto; margin-right: auto; margin-top: 10px; margin-bottom: 10px; border: solid;">'
    scroll = '<div style="height: 300px; overflow-y: scroll;">'

    label_head = '<span style="margin-left: 1%; margin-bottom: 2px; border-bottom: 2px solid black; height: 30px; display: grid; grid-template-columns: {}; grid-gap: 2px">'.format(TBL_ROW_PERCS)
    label_first = '<span style="grid-column: 1/4; grid-row: 1/1;"><b>Filename</b></span>'
    label_second = '<span style="grid-column: 2/4; grid-row: 1/1;"><b>Filepath</b></span>'
    label_third = '<span style="grid-column: 3/4; grid-row: 1/1;"><b>Last modified</b></span>'
    label_fourth = '<span style="grid-column: 4/4; grid-row: 1/1;"><b>Size</b></span>'
    label_end = '</span>'

    row_text = '\n'.join([row_html(r) for r in row_data])

    bottom = "</div></div>"
    #return '<div style="color: blue;">WOOLEY</div>'

    return '\n'.join([top, scroll, label_head, label_first, label_second,
                      label_third, label_fourth, label_end, row_text, bottom])


def tbl_text(uid, ind, exptype):

    ge = GroupExplan.query.filter_by(user_id = uid,
                                     sel_for_qs_ind = ind,
                                     exp_type = exptype).one()

    return generate_tbl_text(uid, ge)

@app.route("/explan_file_tbl", methods=['GET'])
@cross_origin(origins="*")
def qualtrics_ex():

    uid = request.args.get("uid")
    ind = request.args.get("ind")
    exptype = request.args.get("exptype")

    return tbl_text(uid, ind, exptype)

def generate_explan_html(uid, group_exp, exptype):

    group_base = GroupSample.query.filter_by(user_id = uid, group_id = group_exp.group_id).one()

    if exptype == "no_exp":
        base_f = File.query.filter_by(user_id = uid, id= group_base.base_file_id).one()
        return "<span>Because you moved, shared, or deleted <span title='{}'><u>{}</u></span> (<span title='{}'>{}</span>).</span>".format(base_f.name, trunc_str(base_f.name, NAME_LEN), base_f.path, trunc_str(base_f.path, PATH_LEN))

    elif exptype == "rules":
        return html_explain(json_to_exp(group_exp.exp_text, exptype))

    elif exptype == 'dt' or exptype == 'rulesdt':

        return group_exp.exp_blob

        # image_tag = "<img src='https://madison.cs.uchicago.edu/dtimg/{}_{}.png' height='400'>\n".format(uid, group_base.base_file_id)
        # wrapper = "<div style='display: flex; justify-content: center;'>\n"
        # return wrapper + image_tag + "</div>"


def get_explan_html(uid, ind, exptype):

    group_exp = GroupExplan.query.\
        filter_by(user_id = uid, sel_for_qs_ind = ind).one()

    return generate_explan_html(uid, group_exp, exptype)

@app.route("/explan_html", methods=['GET'])
@cross_origin(origins="*")
def qualtrics_explan_html():

    uid = request.args.get("uid")
    ind = request.args.get("ind")
    exptype = request.args.get("exptype")

    return get_explan_html(uid, ind, exptype)



@app.route("/explan_base_exp", methods=['GET'])
@cross_origin(origins="*")
def qualtrics_explan_base():

    ALTER_NAME_LEN = 45

    uid = request.args.get("uid")
    ind = request.args.get("ind")

    group_exp = GroupExplan.query.\
        filter_by(user_id = uid, sel_for_qs_ind = ind).one()

    group_base = GroupSample.query.filter_by(user_id = uid, group_id = group_exp.group_id).one()

    base_f = File.query.filter_by(user_id = uid, id= group_base.base_file_id).one()
    return '<span>Suppose that you shared, moved, or deleted <span title="{}"><a href="{}" target="_blank">{}</a></span> (<span title="{}">{}</span>).</span>'.format(base_f.name, base_f.view_link, trunc_str(base_f.name, ALTER_NAME_LEN), base_f.path, trunc_str(base_f.path, PATH_LEN))


@app.route("/explan_dt_extra", methods=['GET'])
@cross_origin(origins="*")
def qualtrics_dt_extra():

    uid = request.args.get("uid")
    ind = request.args.get("ind")

    group_exp = GroupExplan.query.\
        filter_by(user_id = uid, sel_for_qs_ind = ind).one()

    assert group_exp.exp_type == 'dt' or group_exp.exp_type == 'rulesdt'

    gfs = GroupFiles.query.filter_by(user_id = uid,
                                     explan_id = group_exp.explan_id).all()

    fid_nodes = {f.file_id: f.dt_exp_node for f in gfs}
    fs = File.query.filter_by(user_id = uid).\
        filter(File.id.in_(list(fid_nodes.keys()))).all()

    ret_dict = {}
    for f in fs:
        n = fid_nodes[f.id]
        ret_dict[str(n)] = ret_dict.get(str(n), []) + [trunc_str(f.name, NAME_LEN)]

    return jsonify(ret_dict)


@app.route("/rerun", methods=['GET'])
def rerun():

    user_id = request.args.get("user")
    if not user_id:
        return 'OK'

    explan_backend(user_id).delay()

    return 'OK'

@app.route("/replay", methods=['GET'])
#@templated("replay.html")
def replay():

    return redirect(PROLIFIC_PART_TWO_END)

    if "user_id" not in session:
        logging.error("user_id not in session for replay")
        return redirect(url_for("part_two"))

    uid = session["user_id"]

    cmd_list = CommandHistory.query.filter(CommandHistory.user_id == uid,
                                           CommandHistory.cmd != "open",
                                           CommandHistory.cmd != "find").\
                                           order_by(CommandHistory.time_run).all()

    qual_param_list = [replay_params_convert(uid, c) for c in cmd_list]
    qual_param_list = list(chain.from_iterable(qual_param_list))
    t_entries = [replay_table_entry(qp) for qp in qual_param_list]

    #child_to_key, key_to_child = get_command_graph(uid, qual_param_list)
    child_to_key, key_to_child = {}, {}

    return {"commands": t_entries,
            'child_to_key': child_to_key,
            'key_to_child': key_to_child}

@app.route("/tc", methods=['GET'])
@templated("tc.html")
def tc():
    return dict()

@app.route("/cm", methods=['GET', 'POST'])
@templated("elfinder.src.html")
def cm():

    if "user_id" not in session:
        logging.error("user_id not in session for cm")
        return redirect(url_for("part_two"))

    uid = session['user_id']

    if "exp_cond" not in session:
        user = User.query.filter_by(id = uid).one()
        if user.exp_condition is None:
            logging.error("User with loaded info has no experimental condition")
            return redirect(url_for("part_two"))
        session["exp_cond"] = user.exp_condition
        exp_cond = user.exp_condition
    else:
        exp_cond = session["exp_cond"]

    do_open = folder_phash = file_phash = cmd_state = "0"
    if request.args.get("folderphash", None) is not None:
        do_open = "1"
        folder_phash = request.args.get("folderphash")
        file_phash = request.args.get("filephash", "0")
        cmd_state = request.args.get("cmd_state", "0")
    else:
        user = User.query.filter_by(id = uid).one()
        if user.start_org is None:
            user.start_org = int(time.time())
            session['start_org'] = user.start_org
            db.session.commit()

    return {"do_open": do_open,
            "folder_phash": folder_phash,
            "file_phash": file_phash,
            "cmd_state": cmd_state,
            "exp_cond": exp_cond}

@app.route("/part_two", methods=['GET'])
@templated("p2_instruct.html")
def part_two():

    os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'

    log = logging.getLogger()
    log.setLevel(10)

    session["part_two"] = True
    session["failed"] = False

    # session['condition'] = request.args.get("condition")
    # session['ordering'] = request.args.get("ordering")

    session['condition'] = 'our_recs'
    session['ordering'] = 'del_first'

    return {'p2_route': scope_route('minimal')}


@app.route("/explan_part_two", methods=['GET'])
def explan_part_two():

    session["explan_part_two"] = True
    session["failed"] = False

    return redirect(scope_route('minimal'))


@app.route("/css/<path:filename>", methods=['GET', 'POST'])
def css(filename):
    rootdir = os.path.dirname(os.getcwd())
    return send_from_directory(os.path.join(rootdir, 'elfinder-flask', 'felfinder', 'css'),
                               filename)

@app.route("/jquery/<path:filename>", methods=['GET', 'POST'])
def jquery(filename):
    rootdir = os.path.dirname(os.getcwd())
    return send_from_directory(os.path.join(rootdir, 'elfinder-flask', 'felfinder', 'jquery'),
                               filename)

@app.route("/js/<path:filename>", methods=['GET', 'POST'])
def js(filename):
    rootdir = os.path.dirname(os.getcwd())
    return send_from_directory(os.path.join(rootdir, 'elfinder-flask', 'felfinder', 'js'),
                               filename)

@app.route("/img/<path:filename>", methods=['GET', 'POST'])
def img(filename):
    rootdir = os.path.dirname(os.getcwd())
    return send_from_directory(os.path.join(rootdir, 'elfinder-flask', 'felfinder', 'img'),
                               filename)


@app.route("/dtimg/<path:filename>", methods=['GET', 'POST'])
def dtimg(filename):
    rootdir = os.path.dirname(os.getcwd())
    return send_from_directory(os.path.join(rootdir, 'elfinder-flask', 'felfinder', 'dtimg'),
                               filename)



@app.route("/connector", methods=['GET', 'POST'])
def connector():

    data = request.form
    cmd = data.get('cmd')

    if not cmd:
        return 'OK'

    state = data['cmd_state'] if data['cmd_state'] != "0" else None


    # data = parse_qs(urlparse(request.url).query)
    # cmd = data.get('cmd')

    # print(request.form)

    # if not cmd:
    #     return jsonify(""), 200

    # state = data['cmd_state'][0] if data['cmd_state'] != ["0"] else None

    elf = ElFinderConnector(state)
    resp, code = elf.dispatch(request)

    if not code:
        code = 200

    return jsonify(resp), code

def google_logged_in(creds, scope_level):

    """
    TODO: Add the documentation here
    """

    random.seed(time.time())

    if not creds:
        session["failed"] = True
        return

    session["salt"]= bcrypt.gensalt().decode("utf-8")
    session["target_service"] = "Google Drive"

    userinfo_pay = {"alt": "json",
                    "access_token": creds.token}

    user_info_link = "https://www.googleapis.com/oauth2/v2/userinfo?" + urlencode(userinfo_pay)
    user_info_resp = requests.get(user_info_link)

    if user_info_resp.status_code in [400, 401, 404, 500]:
        session["failed"] = True
        return

    user_info = user_info_resp.json()
    user_id = get_hash(user_info['id'])
    session['user_id'] = user_id
    access_token = creds.token

    user_query = User.query.filter_by(id=user_id)

    try:

        user = user_query.one()
        user.access_token = access_token
        #user.refresh_token = creds.refresh_token
        #session['info_loaded'] = user.info_loaded
        session['info_loaded'] = True
        session['prolific_id'] = user.prolific_id
        session['exp_cond'] = user.exp_condition

        db.session.commit()

    except NoResultFound:

        if not session["part_two"]:

            EXP_CONDS = ["reg_int", "full_recs"]
            exp_cond = random.choice(EXP_CONDS)
            #exp_cond = "reg_int" if PARED_PROTOCOL else "full_recs" #For piloting
            #exp_cond = "full_recs"

            refresh_token = creds.refresh_token
            prol_id = session['prolific_id']
            session['info_loaded'] = False
            session['exp_cond'] = exp_cond

            user = User(id=user_id,
                        access_token=access_token,
                        refresh_token=refresh_token,
                        token_uri=creds.token_uri,
                        prolific_id=prol_id,
                        exp_condition = exp_cond)

            db.session.merge(user)
            db.session.commit()

            explan_backend(user_id).delay()

        else:
            session["failed"] = True


def swampnet_backend(user_id):

    populate = populate_google.s(user_id)
    full_chain = cel_chain(populate | activity_logs.s())

    populate = populate_google.s(user_id)
    after_proc = cel_chain(collect_results.s() | index_and_finalize_text.s() | \
                           swampnet_rep.s() | set_user_complete.s(user_id))

    end_steps = post_proc_wrapper.s(goog_celery_proc.s(), after_proc)
    full_chain = cel_chain(populate | activity_logs.s() | end_steps)

    return full_chain


def pared_backend(user_id):
    populate = populate_google.s(user_id)
    full_chain = cel_chain(populate | activity_logs.s())

    populate = populate_google.s(user_id)
    post_simil = (precompute_recs.s(user_id) | set_user_complete.s(user_id))
    post_bulk_feats = simil_wrapper.s(simil_proc.s(user_id), post_simil)
    after_proc = cel_chain(collect_results.s() | index_and_finalize_text.s() | \
                           post_bulk_feats)

    end_steps = post_proc_wrapper.s(goog_celery_proc.s(), after_proc)
    full_chain = cel_chain(populate | activity_logs.s() | end_steps)

    return full_chain

def explan_backend(user_id):
    populate = populate_google.s(user_id)

    populate = populate_google.s(user_id)
    post_simil = (precompute_recs.s(user_id) | set_up_explan_routing.s(user_id) | \
                  set_user_complete.s(user_id))
    post_bulk_feats = simil_wrapper.s(simil_proc.s(user_id), post_simil)
    after_proc = cel_chain(collect_results.s() | index_and_finalize_text.s() | \
                           post_bulk_feats)

    end_steps = post_proc_wrapper.s(goog_celery_proc.s(), after_proc)
    full_chain = cel_chain(populate | end_steps)

    return full_chain


def backend_process(user_id):
    populate = populate_google.s(user_id)
    post_bulk_feats = simil_wrapper.s(simil_proc.s(user_id),
                                      set_user_complete.s(user_id))
    after_proc = cel_chain(collect_results.s() | index_and_finalize_text.s() | \
                           post_bulk_feats)

    end_steps = post_proc_wrapper.s(goog_celery_proc.s(), after_proc)
    full_chain = cel_chain(populate | end_steps)

    return full_chain
