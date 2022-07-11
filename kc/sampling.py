import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import re
import time
import random
import pickle
import pprint
import graphviz
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

from collections import Counter
from datetime import datetime
from flask import session
from itertools import chain, product
from sqlalchemy import or_, and_
from sqlalchemy.orm import aliased
from sklearn import tree

from felfinder.config import GROUP_SAMP_SIZE
from felfinder.comanaged import (get_hist_cmg_info, bulk_pairs_to_samples,
                                 get_cmd_cmg_move_pairs, new_folders,
                                 bulk_folders_to_samples, get_cmd_del_cmg,
                                 get_false_positives,
                                 bulk_folder_pairs_to_samples, new_folder_pairs)
from felfinder.models import (db, User, CommandHistory, Recommends,
                              CommandFiles, File, Folder, Simils,
                              DriveHistory, DriveFile, ActionSample,
                              GroupSample, GroupFiles, GroupExplan)
from felfinder.pred_utils import simil_rows_to_action, simil_rows_for_precomp
from felfinder.explain import (BaselineExplainer, Explainer, LocalDTExplainer,
                               graph_exp, FULL_FEATURES, FULL_ATTRS)
from felfinder.explain_utils import (attr_convert, group_dicts, exp_to_json,
                                     is_complex_exp)
from felfinder.utils import get_hash, get_rand_id, even_partition, trunc_str

NUM_UNI_ITEMS = 2
TOTAL_NUM_ACTION_QS = 14
TOTAL_NUM_EXPLAN_QS = 99
SHORT_NAME_LEN = 30
SAMPLE_MAX = 1000 # We aren't able to convert as many pairs to samples as we
                  # would like in the time participants are waiting, so we limit
                  # how many we examine

def trunc_pair(id_pairs, cmg_pairs):

    """
    We want to limit the number of pairs to consider in the first place
    """

    pair_consider = [(i, c) for i, c in zip(id_pairs, cmg_pairs)]
    samp = trunc_sample(pair_consider)
    if len(samp) > 0:
        id_pairs, cmg_pairs = zip(*samp)
    else:
        id_pairs, cmg_pairs = [], []

    return id_pairs, cmg_pairs


def trunc_sample(to_samp):
    return random.sample(to_samp, min(SAMPLE_MAX, len(to_samp)))





def fold_counts_tbl(uid):

    """
    Get a pandas table with the number of files in each folder
    """

    q = "SELECT parent_id, COUNT(1) FROM file WHERE user_id=%s GROUP BY parent_id"
    df = pd.read_sql(q, con=db.engine, params=[uid])
    mapping = df.set_index('parent_id').iloc[:, 0].to_dict()

    return mapping


def discrim_score(exp_items, fold_counts):

    """
    Given a set of items explained over, and the counts of
    items in various folders, calculate the min and max
    discrim_scores for the explanation
    """

    per_f = Counter([f['parent_id'] for f in exp_items])
    perc_scores = [f_count / fold_counts[pid] for pid, f_count in per_f.items()]

    return max(perc_scores), min(perc_scores)


def counts_tbl_from_simils(df):

    """
    Given a subset of the Simils table, want to get
    a count of the positive predictions for all file
    id's
    """

    true_df = df[df['del_pred'] == True]

    user_file_ids = pd.concat((true_df['file_id_A'],
                               true_df['file_id_B'])).unique()

    a_count = true_df['file_id_A'].value_counts().reindex(user_file_ids,
                                                          fill_value = 0)
    b_count = true_df['file_id_B'].value_counts().reindex(user_file_ids,
                                                          fill_value = 0)

    return a_count + b_count


def fixup_graphviz(gviz, exp_dict):

    """
    We want the graphviz text to look like not-garbage,
    so this does some stuff post-hoc to correct for that.

    Example input:

    digraph Tree {
    node [shape=box, style="filled", color="black"] ;
    0 [label="bigram_simil <= 0.67\ngini = 0.482\nsamples = 99\nvalue = [40, 59]", fillcolor="#bfdff7"] ;
    1 [label="size_dist <= 0.391\ngini = 0.484\nsamples = 68\nvalue = [40, 28]", fillcolor="#f7d9c4"] ;
    0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;
    2 [label="gini = 0.395\nsamples = 48\nvalue = [35, 13]", fillcolor="#efb083"] ;
    1 -> 2 ;
    3 [label="gini = 0.375\nsamples = 20\nvalue = [5, 15]", fillcolor="#7bbeee"] ;
    1 -> 3 ;
    4 [label="gini = 0.0\nsamples = 31\nvalue = [0, 31]", fillcolor="#399de5"] ;
    0 -> 4 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;
    }


    """

    # with open('utils_testing/gviz.txt', 'wb') as of:
    #     pickle.dump((gviz, exp_dict), of)
    # exit(0)


    def repl_func(pos_node, leaf_node):

        # THIS IS CURRENTLY WRONG
        # Need to potentially take the out values as well, if those are
        # with the in-crowd
        def value_func(m):

            ret = m.group(0)
            v_str = "value = "
            ret = ret[len(v_str):-1] #Cut off quote mark

            if ']' not in ret:
                return "# of files: " + str(int(float(ret))) + '"'

            split_vals = [int(i) for i in ret.strip('][').split(",")]

            if pos_node:
                return "# of files: " + str(sum(split_vals)) + '"'
            elif not leaf_node:
                return '"'
            else:
                return 'None"'

        return value_func

    out_str = ""
    for row in gviz.split('\n'):

        if row.startswith("node") or row.startswith("digraph") or row.startswith("}"):
            out_str += row + '\n'
            continue

        if "->" in row:
            if '[' in row:

                mod_row = row.replace("<=", ">")
                if "True" in row:
                    mod_row = row.replace("True", "False")
                elif "False" in row:
                    mod_row = row.replace("False", "True")

                out_str += mod_row + '\n'
                continue

            arrow_lab = '[labeldistance=2.5, labelangle={}, headlabel="{}"]'
            to_node = int(row[:-1].split("->")[1])
            if to_node in exp_dict['left_children']:
                out_str += row[:-1] + arrow_lab.format("45", "False") + '\n'
            else:
                out_str += row[:-1] + arrow_lab.format("-45", "True") + '\n'

            continue

        row_num = int(str(row[0]))
        pos_node = (row_num in exp_dict['in_nodes']) or (row_num in exp_dict['out_nodes'])
        leaf_node = not any([k in row for k in FULL_FEATURES])

        val_rep = repl_func(pos_node, leaf_node)

        mod_row = re.sub("gini = [0-9]*\.[0-9]*\\\\n", "", row)
        mod_row = re.sub("samples = [0-9]*\.*[0-9]*\\\\n", "", mod_row)

        for f, repname in FULL_FEATURES.items():
            mod_row = re.sub(f, repname, mod_row)

        mod_row = mod_row.replace("<=", ">")

        mod_row = re.sub('value = .*?"', val_rep, mod_row)

        if not leaf_node:
            mod_row = re.sub('fillcolor="#[A-Za-z0-9]{6}"', 'fillcolor="#ffffff"', mod_row)
        elif pos_node:
            mod_row = re.sub('fillcolor="#[A-Za-z0-9]{6}"', 'fillcolor="#01c221"', mod_row)
        else:
            mod_row = re.sub('fillcolor="#[A-Za-z0-9]{6}"', 'fillcolor="#ff5555"', mod_row)

        out_str += mod_row + '\n'


    return out_str
    #return gviz.replace("\\n", "")
    #return rem_label_newlines(gviz)


def variant_spec(uid, fid, exp_dict, exp_type):

    """
    Perform any processing necessary for specific variants of explanation
    conditions
    """

    DT_IMG_PATH = "felfinder/dtimg/{}_{}.png"

    if exp_type == 'dt':


        dot_data = tree.export_graphviz(exp_dict['variant_clf'], out_file=None,
                            feature_names=exp_dict['variant_feats'],
                            filled=True)

        # graph = graphviz.Source(dot_data, format="png")
        # png_bytes = graph.pipe(format = 'png')
        # with open(DT_IMG_PATH.format(uid, fid), 'wb') as of:
        #     of.write(png_bytes)



def explans_and_files(uid, fid, rel_ids, gid, mod_all_wraps,
                      fold_counts, exp_obj, exp_type):

    """
    Given a grouping generated by the classifier, produce an explanation
    and the relevant files that attend it
    """
    start = time.time()
    exp_dict = exp_obj.explain(fid, rel_ids,
                               meth = 'full_approx',
                               score_weight = 'standard_mod')
    end = time.time()

    # If you don't explain any of the in-group, what are you even doing?
    if len(exp_dict['in']) < 1:
        return [], []

    exp_items = [mod_all_wraps[f] for f in exp_dict['in'] + exp_dict['out']]
    max_score, min_score = discrim_score(exp_items, fold_counts)

    exp_blob = ''
    if exp_type == 'dt':

        dot_data = tree.export_graphviz(exp_dict['variant_clf'], out_file=None,
                            feature_names=exp_dict['variant_feats'],
                            filled=True)

        #exp_blob = dot_data
        exp_blob = fixup_graphviz(dot_data, exp_dict)


    elif exp_type == 'rulesdt':
        exp_blob = graph_exp(exp_dict['exp_text'], len(exp_dict['in'] + exp_dict['out']))


    exp_id = get_rand_id()
    ge = GroupExplan(user_id = uid,
                     group_id = gid,
                     explan_id = exp_id,
                     exp_type = exp_type,
                     exp_text = exp_to_json(exp_dict['exp_text'], exp_type),
                     exp_blob = exp_blob,
                     rec = exp_dict['rec'],
                     prec = exp_dict['prec'],
                     is_complex = is_complex_exp(exp_dict['exp_text'], exp_type),
                     min_discrim_score = min_score,
                     max_discrim_score = max_score,
                     full_score = exp_dict['full_score'],
                     size = len(exp_dict['in'] + exp_dict['out']),
                     time_taken = end - start)

    variant_spec(uid, fid, exp_dict, exp_type)

    nodes = [-1 for n in (exp_dict['in'] + exp_dict['out'])]
    if exp_type == 'dt' or exp_type == 'rulesdt':
        nodes = exp_dict['in_nodes'] + exp_dict['out_nodes']

    exp_gfs = [GroupFiles(rand_id = get_rand_id(),
                          user_id = uid,
                          group_id = gid,
                          explan_id = exp_id,
                          dt_exp_node = n,
                          file_id = f,
                          out_group_sample = False)
               for f, n in zip((exp_dict['in'] + exp_dict['out']), nodes)]


    return [ge], exp_gfs


def save_explan_samps(uid, all_wraps):

    """
    Produce groups to later sample from for the explanation study
    """

    #q = """SELECT "file_id_A", "file_id_B", del_pred, del_true_cert FROM simils WHERE user_id=%s;"""
    #q = """SELECT * FROM simils WHERE user_id=%s;"""
    #df = pd.read_sql(q, con=db.engine, params=[uid])

    df = simil_rows_for_precomp(uid)

    if len(df) < 1:
        return

    mod_all_wraps = {k: attr_convert(f) for k, f in all_wraps.items()}

    #print(pprint.pformat(mod_all_wraps))

    # This is just a pre-filter, as we will be enforcing some
    # disjointness criteria on the file id's that will eliminate
    # even more subsequently
    true_preds = counts_tbl_from_simils(df)
    fold_counts = fold_counts_tbl(uid)

    examine = true_preds[true_preds >= GROUP_SAMP_SIZE].index.tolist()

    group_samps = []
    group_files = []
    group_exps = []

    seen_files = set()

    group_dict, out_groups = group_dicts(df, examine, use_proba = True)

    our_exp = Explainer(list(mod_all_wraps.values()),
                    group_dict, out_groups,
                    attrs_to_explain = FULL_ATTRS,
                    num_uni_items = NUM_UNI_ITEMS)

    dt_exp = LocalDTExplainer(df, df[['file_id_A', 'file_id_B']].values.tolist())

    base_exp = BaselineExplainer()

    for fid, mod_rel_ids in group_dict.items():

        if fid in seen_files:
            continue

        # If we have no out_group, wild as that might be.
        if len(mod_rel_ids) >= len(mod_all_wraps) - 1:
            continue

        rel_ids = [x[0] for x in mod_rel_ids]
        for r in rel_ids:
            seen_files.add(r)

        gid = get_rand_id()
        g = GroupSample(user_id=uid,
                        group_id = gid,
                        base_file_id = fid,
                        action = 'del',
                        size = len(rel_ids))

        group_samps.append(g)

        # exp_objs = [base_exp, dt_exp, our_exp]
        # exp_types = ['no_exp', 'dt', 'rules']

        exp_objs = [base_exp, dt_exp, our_exp, our_exp]
        exp_types = ['no_exp', 'dt', 'rulesdt', 'rules']


        for exp_obj, exp_type in zip(exp_objs, exp_types):
            ges, gfs = explans_and_files(uid, fid, rel_ids, gid, mod_all_wraps,
                                         fold_counts, exp_obj, exp_type)
            group_exps.extend(ges)
            group_files.extend(gfs)

    db.session.bulk_save_objects(group_samps)
    db.session.bulk_save_objects(group_exps)
    db.session.bulk_save_objects(group_files)

    db.session.commit()



def save_sample_objs(uid):

    """
    Produce the items we want to later sample from
    """

    # Get history actions co-managed
    orig_hist_cmg_id_pairs, cmg_pairs = get_hist_cmg_info(uid)
    hist_cmg_id_pairs, cmg_pairs = trunc_pair(orig_hist_cmg_id_pairs, cmg_pairs)
    hist_acts = bulk_pairs_to_samples(uid, hist_cmg_id_pairs,
                                      cmg_pairs, "move",
                                      hist = True,
                                      sample_reason = "hist_cmg_move")
    if hist_acts:
        db.session.bulk_save_objects(hist_acts)

    # Get current actions, move
    orig_cmd_cmg_id_pairs, cmg_pairs = get_cmd_cmg_move_pairs(uid)
    cmd_cmg_id_pairs, cmg_pairs = trunc_pair(orig_cmd_cmg_id_pairs, cmg_pairs)
    cmd_acts = bulk_pairs_to_samples(uid, cmd_cmg_id_pairs,
                                      cmg_pairs, "move",
                                      hist = False,
                                      sample_reason = "cmd_cmg_move")

    if cmd_acts:
        db.session.bulk_save_objects(cmd_acts)

    # Get cmd deletes
    orig_cmd_cmg_del_id_pairs, cmg_pairs = get_cmd_del_cmg(uid)
    cmd_cmg_del_id_pairs, cmg_pairs = trunc_pair(orig_cmd_cmg_del_id_pairs, cmg_pairs)
    cmd_dels = bulk_pairs_to_samples(uid, cmd_cmg_del_id_pairs,
                                     cmg_pairs, "del",
                                     hist = False,
                                     sample_reason = "cmd_cmg_del")
    if cmd_dels:
        db.session.bulk_save_objects(cmd_dels)

    # Get move false positives
    full_id_pairs = chain(orig_hist_cmg_id_pairs,
                          orig_cmd_cmg_id_pairs)
    unique_id_pairs = set([str(tuple(sorted(p))) for p in full_id_pairs])
    move_false_positives = get_false_positives(uid, unique_id_pairs,
                                               "move", "false_pos_move")
    if move_false_positives:
        db.session.bulk_save_objects(move_false_positives)

    # Get delete false positives
    unique_id_pairs = set([str(tuple(sorted(p)))
                           for p in orig_cmd_cmg_del_id_pairs])
    del_false_positives = get_false_positives(uid, unique_id_pairs,
                                          "del", "false_pos_del")
    if del_false_positives:
        db.session.bulk_save_objects(del_false_positives)

    # Get history folders create
    # ids, cmds = new_folder_pairs(uid, hist = True)
    # ids, cmds = trunc_pair(ids, cmds)
    # hist_folds = bulk_folder_pairs_to_samples(uid, ids, cmds, hist = True)
    # db.session.bulk_save_objects(hist_folds)

    # Get cmd folders create
    # ids, cmds = new_folder_pairs(uid, hist = False)
    # ids, cmds = trunc_pair(ids, cmds)
    # hist_folds = bulk_folder_pairs_to_samples(uid, ids, cmds, hist = False)
    # db.session.bulk_save_objects(hist_folds)

    db.session.commit()

def filter_samps(samp_pop, already_chosen, att = 'rand_id'):

    """
    In order to ensure we don't sample the same item
    more than once, we have to filter out the already_chosen
    ActionSample objects from the potential samp_pop

    """

    chosen_ids = [getattr(c, att) for c in already_chosen]
    samp_pop = [c for c in samp_pop if getattr(c, att) not in chosen_ids]

    return samp_pop


def sample_actions(uid, reason, num, curr_samps = []):

    """
    Sample a single subset of actions with
    a given sample reason and a desired number
    of actions
    """

    # NOTE: Some of the logic of this function is partially
    # dependent on the fact that each ActionSample object
    # is distinct

    print("reason: {}".format(reason))

    # to_split = ["cmd_cmg_move", "cmd_cmg_del",
    #             "false_pos_move", "false_pos_del"]

    to_split = ["cmd_cmg_move", "cmd_cmg_del",
                "false_pos_move", "false_pos_del",
                "hist_cmg_move"]

    samps = ActionSample.query.filter_by(user_id=uid, sample_reason=reason)

    samp_subsets = []
    samp_avail = []

    chosen_samps = []

    if reason in to_split:
        pos_samps = samps.filter_by(pred=True).all()
        # neg_samps = samps.filter_by(pred=False,
        #                             near_miss=True).all()
        neg_samps = samps.filter_by(pred=False).all()

        pos_samps = filter_samps(pos_samps, curr_samps)
        neg_samps = filter_samps(neg_samps, curr_samps)

        samp_subsets.append(pos_samps)
        samp_avail.append(len(pos_samps))

        samp_subsets.append(neg_samps)
        samp_avail.append(len(neg_samps))
    else:
        full_samps = samps.all()
        full_samps = filter_samps(full_samps, curr_samps)
        samp_subsets.append(full_samps)
        samp_avail.append(len(full_samps))

    # TODO: This will have to change eventually, but it's OK for now
    desired_choices = even_partition(num, len(samp_subsets), samp_avail)

    for s, desired, avail in zip(samp_subsets, desired_choices, samp_avail):
        print("Desired, avail: {}, {}".format(desired, avail))

        chosen = random.sample(s, min(desired, avail))


        # TODO: The below might now be unnecessary
        # Ensure we haven't already selected these actions to sample
        already_chosen_ids = [c.rand_id for c in curr_samps]
        while not all([c.rand_id not in already_chosen_ids for c in chosen]):
            chosen = random.sample(s, min(desired, avail))


        chosen_samps.extend(chosen)

    return chosen_samps


def sample_explans(uid, reason, num, curr_samps = []):

    """
    Sample a single subset of explanations with
    a given sample reason and a desired number
    of actions
    """

    # NOTE: Some of the logic of this function is partially
    # dependent on the fact that each ActionSample object
    # is distinct

    if num == 0:
        return []

    print("reason: {}".format(reason))

    if reason == 'complex':
        samps = GroupExplan.query.filter_by(user_id = uid, is_complex = True).all()
    elif reason == 'not_complex':
        samps = GroupExplan.query.filter_by(user_id = uid, is_complex = False).all()
    elif reason == 'not_discrim':
        samps = GroupExplan.query.filter_by(user_id = uid).all()
        samps = sorted(samps, key=lambda x: -x.min_discrim_score)[:int(len(samps) / 2)]
    elif reason == 'discrim':
        samps = GroupExplan.query.filter_by(user_id = uid).all()
        samps = sorted(samps, key=lambda x: x.max_discrim_score)[:int(len(samps) / 2)]
    elif reason == 'small' or reason == 'med' or reason == 'large':
        samps = GroupExplan.query.filter_by(user_id = uid).all()
        samps = sorted(samps, key=lambda x: x.size)
        small_ind = int(0.25 * len(samps))
        med_ind = int(0.75 * len(samps))
        if reason == 'small':
            samps = samps[0:small_ind + 1]
        elif reason == 'med':
            samps = samps[small_ind:med_ind + 1]
        else:
            samps = samps[med_ind:]

    samp_subsets = []
    samp_avail = []

    chosen_samps = []

    full_samps = filter_samps(samps, curr_samps, 'group_id')
    chosen = random.sample(full_samps, min(num, len(full_samps)))

    for c in chosen:
        c.sample_reason = reason

    db.session.commit()

    return chosen



def OLD_set_up_qual_routing(uid):

    """
    If this is the first time a participant hits this route, we have to
    select file management actions we want to ask them about.

    Current strategy: select as many as 5 commands that were accepted
    recommendations, and select the remainder as random file management
    actions.

    """

    done_commands = CommandHistory.query.filter(CommandHistory.user_id==uid,
                                                CommandHistory.cmd != "open",
                                                CommandHistory.cmd != "find")

    rec_cmds = done_commands.filter(CommandHistory.rec_result_id != None).all()
    non_rec_cmds = done_commands.filter_by(rec_result_id = None).all()

    rec_samp = min(len(rec_cmds), max(TOTAL_NUM_ACTION_QS // 2,
                                      TOTAL_NUM_ACTION_QS - len(non_rec_cmds)))
    non_rec_samp = min(len(non_rec_cmds), TOTAL_NUM_ACTION_QS - rec_samp)

    sel_rec_cmds = random.sample(rec_cmds, rec_samp)
    sel_non_rec_cmds = random.sample(non_rec_cmds, non_rec_samp)

    sel_commands = sel_rec_cmds + sel_non_rec_cmds
    random.shuffle(sel_commands)

    for i, c in enumerate(sel_commands):
        c.sel_for_qs_ind = i
        c.ask_order = i

    db.session.commit()



def action_samp_from_rec_objs(uid, rec_objs, accepted=False):

    """
    Given a set of database objects that are either plain
    Recommends objects, or temporary objects joined to the
    CommandHistory table, we will be producing ActionSample
    objects with the correct sampling reasons
    """


    act_samps = []

    for u in rec_objs:

        # Get the files that were to be acted on, and the original file
        # action that led to the recommendation
        f_a = File.query.filter_by(user_id=uid,
                                   path_hash=u.rec_path_hash).one()
        f_b = File.query.filter_by(user_id=uid,
                                   path_hash=u.explain_hash).one()

        diff_par = f_a.original_parent_id != f_b.original_parent_id

        # If it's a move action, we want to get the relevant destination folder
        fold_id = None
        if u.dst_hash is not None:
            fold = Folder.query.filter_by(user_id=uid, path_hash=u.dst_hash).one()
            fold_id = fold.id

        # Extract similarity information used for the recommendation
        s_row = Simils.query.filter(Simils.user_id==uid,
                                    or_(and_(Simils.file_id_A==f_a.id,
                                             Simils.file_id_B==f_b.id),
                                        and_(Simils.file_id_A==f_b.id,
                                             Simils.file_id_B==f_a.id)))
        s_rows = pd.read_sql(s_row.statement, con=db.engine)

        _, _, feat_a, feat_b, _, near_miss = simil_rows_to_action(s_rows,
                                                                  u.action)[0]

        # Build the sample
        if not accepted:
            samp_reason = "untaken_move" if u.action == "move" else "untaken_del"
        else:
            samp_reason = "accepted_move" if u.action == "move" else "accepted_del"

        samp_reason += "_diff" if diff_par else "_same"


        a = ActionSample(rand_id = get_rand_id(), user_id = uid,
                         file_id_a = f_a.id, file_id_b = f_b.id,
                         pred = True, top_feat = feat_a, folder_id = fold_id,
                         cmd_id_a = getattr(u, "cmd_id_a", None),
                         cmd_id_b = getattr(u, "cmd_id_b", None),
                         second_feat = feat_b, near_miss = near_miss,
                         corr_pred = False, hist = False,
                         sample_reason = samp_reason)

        act_samps.append(a)

    return act_samps


def action_samp_from_recs(uid):

    """
    Produces ActionSample objects from untaken recommendations
    """

    untaken_recs = Recommends.query.filter(Recommends.user_id==uid,
                                           Recommends.sent==True,
                                           Recommends.accepted==False,
                                           or_(Recommends.deleted==True,
                                               Recommends.done_no_acc==True,
                                               Recommends.faded==True)).all()


    # Because of how recommendations are structured,
    # it's possible for there to be multiple recommendations
    # for a single file that are totally glossed over.
    # We don't want to accept any of these "phantom" recommendations
    acc_recs = Recommends.query.filter(Recommends.user_id==uid,
                                           Recommends.sent==True,
                                           Recommends.accepted==True).all()

    acc_path_hashes = set([p.rec_path_hash for p in acc_recs])
    untaken_recs = [r for r in untaken_recs if r.rec_path_hash not in acc_path_hashes]

    return action_samp_from_rec_objs(uid, untaken_recs, accepted=False)

def action_samp_from_cmds(uid):

    """
    Produces ActionSample objects from CommandHistory objects
    """

    done_commands = db.session.query(CommandHistory.rand_id.label('cmd_id_a'),
                                     CommandHistory.user_id,
                                     CommandHistory.cmd,
                                     CommandHistory.rec_result_id,
                                     Recommends.rec_path_hash,
                                     Recommends.action,
                                     Recommends.dst_hash,
                                     Recommends.explain_hash,
                                     Recommends.cmd_id.label('cmd_id_b')).\
                                     filter(CommandHistory.user_id==uid,
                                            or_(CommandHistory.cmd == "paste",
                                                CommandHistory.cmd == "rm"),
                                            CommandHistory.rec_result_id==Recommends.rand_id,
                                            CommandHistory.user_id==Recommends.user_id).all()

    return action_samp_from_rec_objs(uid, done_commands, accepted=True)


def set_up_qual_routing(uid):

    """
    If this is the first time a participant hits this route, we have to
    select file management actions we want to ask them about.

    """

    rec_samps = action_samp_from_recs(uid)
    cmds_samps = action_samp_from_cmds(uid)

    db.session.bulk_save_objects(rec_samps + cmds_samps)
    db.session.commit()






    rec_cmds = done_commands.filter(CommandHistory.rec_result_id != None).all()
    non_rec_cmds = done_commands.filter_by(rec_result_id = None).all()

    rec_samp = min(len(rec_cmds), max(TOTAL_NUM_ACTION_QS // 2,
                                      TOTAL_NUM_ACTION_QS - len(non_rec_cmds)))
    non_rec_samp = min(len(non_rec_cmds), TOTAL_NUM_ACTION_QS - rec_samp)

    sel_rec_cmds = random.sample(rec_cmds, rec_samp)
    sel_non_rec_cmds = random.sample(non_rec_cmds, non_rec_samp)

    sel_commands = sel_rec_cmds + sel_non_rec_cmds
    random.shuffle(sel_commands)

    for i, c in enumerate(sel_commands):
        c.sel_for_qs_ind = i
        c.ask_order = i

    db.session.commit()


class DummyQualParams:

    """
    If participants have nothing we want to ask about,
    we still want to provide a dummy action_sample
    object to allow them to continue.
    """

    def __init__(self, action_sample):
        pass


    def to_json(self):

        params = {'ProlificID': session['prolific_id'],
                  'samp_reason': 'dummy'}

        return params

class QualParams:

    """
    Ingests an action sample, and upon request, turns it into
    the query parameters that should be sent back with the url.
    Because both commands in an ActionSample are of the same type,
    we can then build subclasses for each action type
    """

    def __init__(self, action_sample):

        self.suffixes = ['_a', '_b']
        self.param_list = ['cmd', 'cmd_id',
                           'dst_path', 'target_name',
                           'target_phash', 'target_parent_hash',
                           'view_link', 'dst_phash',
                           'old_name', 'new_name',
                           'date_str', 'time_str']

        # Assumes we should only sample actions for suffixes for which we have
        # commands
        self.no_cmds = False
        self.act_samp = action_sample

        self.base_cmd_tbl = CommandHistory if not self.act_samp.hist \
            else DriveHistory
        self.base_file_tbl = CommandFiles if not self.act_samp.hist \
            else DriveFile

        self.cmds = {}

        self._convert()

    def _get_query_params(self, suffix):
        raise NotImplementedError

    def _get_cmds(self):

        for s in self.suffixes:
            cmd_rand_id = getattr(self.act_samp, 'cmd_id' + s, None)
            if cmd_rand_id:

                cmd = self.base_cmd_tbl.query.\
                    filter_by(user_id=self.act_samp.user_id,
                              rand_id=cmd_rand_id).one()
                self.cmds[s] = cmd

    def _convert(self):

        self._get_cmds()
        for s in self.suffixes:
            if s in self.cmds:
                self._get_query_params(s)

    def to_json(self):

        all_params = {'ProlificID': session['prolific_id'],
                      'corr_pred': self.act_samp.corr_pred,
                      'samp_reason': self.act_samp.sample_reason,
                      'number': self.act_samp.ask_order + 1,
                      'hist': self.act_samp.hist,
                      'pred': self.act_samp.pred,
                      'param_one': self.act_samp.top_feat,
                      'param_two': self.act_samp.second_feat}

        for p, s in product(self.param_list, self.suffixes):
            if s not in self.cmds and not self.no_cmds:
                continue
            all_params['spec_id' + s] = get_rand_id()
            all_params[p + s] = getattr(self, p + s, None)

        return {k: v for k, v in all_params.items() if v is not None}

class MoveQualParams(QualParams):

    def _get_query_params(self, suffix):

        cmd = self.cmds[suffix]

        setattr(self, 'cmd' + suffix, cmd.cmd)
        setattr(self, 'cmd_id' + suffix, cmd.rand_id)

        dst = self.base_file_tbl.query.\
            filter_by(user_id = self.act_samp.user_id,
                      cmd_id = cmd.rand_id, target = False).one()

        dst_fobj = Folder.query.filter_by(user_id = self.act_samp.user_id,
                                          id = dst.folder_id).one()

        setattr(self, 'dst_path' + suffix, dst_fobj.path)
        setattr(self, 'dst_phash' + suffix, dst_fobj.path_hash)

        # Assume it's moved files and not folders

        fid = getattr(self.act_samp, 'file_id' + suffix)
        target_fobj = File.query.filter_by(user_id = self.act_samp.user_id,
                                           id = fid).one()


        short_name = trunc_str(target_fobj.name, SHORT_NAME_LEN)
        setattr(self, 'target_name' + suffix, short_name)
        setattr(self, 'target_phash' + suffix, target_fobj.path_hash)
        setattr(self, 'target_parent_hash' + suffix, target_fobj.parent_hash)
        setattr(self, 'view_link' + suffix, target_fobj.view_link)

        if self.act_samp.hist:
            timestamp = cmd.timestamp
        else:
            timestamp = cmd.time_run

        dt = datetime.fromtimestamp(timestamp)
        date_format = "%Y-%m-%d"
        time_format = "%H:%M"
        setattr(self, 'date_str' + suffix, dt.strftime(date_format))
        setattr(self, 'time_str' + suffix, dt.strftime(time_format))


class RecMoveQualParams(QualParams):

    """
    QualParams for the full study for an untaken recommendation
    """

    def _get_query_params(self, suffix):

        dst_fobj = Folder.query.filter_by(user_id = self.act_samp.user_id,
                                          id = self.act_samp.folder_id).one()

        setattr(self, 'dst_path' + suffix, dst_fobj.path)
        setattr(self, 'dst_phash' + suffix, dst_fobj.path_hash)

        # Assume it's moved files and not folders

        fid = getattr(self.act_samp, 'file_id' + suffix)
        target_fobj = File.query.filter_by(user_id = self.act_samp.user_id,
                                           id = fid).one()


        short_name = trunc_str(target_fobj.name, SHORT_NAME_LEN)
        setattr(self, 'target_name' + suffix, short_name)
        setattr(self, 'target_phash' + suffix, target_fobj.path_hash)
        setattr(self, 'target_parent_hash' + suffix, target_fobj.parent_hash)
        setattr(self, 'view_link' + suffix, target_fobj.view_link)


class GeneralQualParams(QualParams):

    """
    Works for Mkdir, Rename, or Delete

    TODO: Double check this works for delete
    """

    def _get_query_params(self, suffix):

        cmd = self.cmds[suffix]
        setattr(self, 'cmd' + suffix, cmd.cmd)
        setattr(self, 'cmd_id' + suffix, cmd.rand_id)

        # Will be only in a mkdir or rename
        target = self.base_file_tbl.query.\
            filter_by(user_id = self.act_samp.user_id,
                      cmd_id = cmd.rand_id).first()

        short_old_name = trunc_str(getattr(cmd, "old_name", ""), SHORT_NAME_LEN)
        short_new_name = trunc_str(getattr(cmd, "new_name", ""), SHORT_NAME_LEN)
        setattr(self, 'old_name' + suffix, short_old_name)
        setattr(self, 'new_name' + suffix, short_new_name)

        ftbl = File if target.file_id else Folder
        target_fobj = ftbl.query.\
            filter_by(user_id = self.act_samp.user_id,
                      id = target.file_id or target.folder_id).one()

        short_name = trunc_str(target_fobj.name, SHORT_NAME_LEN)
        setattr(self, 'target_name' + suffix, short_name)
        setattr(self, 'target_phash' + suffix, target_fobj.path_hash)
        setattr(self, 'target_parent_hash' + suffix, target_fobj.parent_hash)
        setattr(self, 'view_link' + suffix, target_fobj.view_link)

        if self.act_samp.hist:
            timestamp = cmd.timestamp
        else:
            timestamp = cmd.time_run

        dt = datetime.fromtimestamp(timestamp)
        date_format = "%Y-%m-%d"
        time_format = "%H:%M"
        setattr(self, 'date_str' + suffix, dt.strftime(date_format))
        setattr(self, 'time_str' + suffix, dt.strftime(time_format))


class FalsePosParams(QualParams):

    """
    Gets query parameters for samples due to a
    false positive. This means they have no commands
    attached to them, which changes a lot of how they work
    """


    def __init__(self, action_sample):

        self.cmds = {}
        self.act_samp = action_sample
        self.param_list = ["target_name", "target_phash",
                           "view_link", "dst_path"]
        self.suffixes = ["_a", "_b"]
        self.no_cmds = True

        target_fobj_a = File.query.\
            filter_by(user_id = self.act_samp.user_id,
                      id = self.act_samp.file_id_a).one()

        target_fobj_b = File.query.\
            filter_by(user_id = self.act_samp.user_id,
                      id = self.act_samp.file_id_b).one()

        dst_path_a = None
        if self.act_samp.folder_id is not None:
            dst_obj_a = Folder.query.filter_by(user_id = self.act_samp.user_id,
                                               id = self.act_samp.folder_id).one()
            dst_path_a = dst_obj_a.path

        short_a_name = trunc_str(target_fobj_a.name, SHORT_NAME_LEN)
        short_b_name = trunc_str(target_fobj_b.name, SHORT_NAME_LEN)
        setattr(self, "target_name_a", short_a_name)
        setattr(self, "target_name_b", short_b_name)
        setattr(self, "target_phash_a", target_fobj_a.path_hash)
        setattr(self, "target_phash_b", target_fobj_b.path_hash)
        if dst_path_a is not None:
            setattr(self, "dst_path_a", dst_path_a)
        setattr(self, "view_link_a", target_fobj_a.view_link)
        setattr(self, "view_link_b", target_fobj_b.view_link)
        setattr(self, "corr_pred", self.act_samp.corr_pred)
        setattr(self, "pred", self.act_samp.pred)
        setattr(self, "samp_reason", self.act_samp.sample_reason)
        setattr(self, "number", self.act_samp.ask_order + 1)
        setattr(self, "hist", self.act_samp.hist)
        setattr(self, "param_a", self.act_samp.top_feat)
        setattr(self, "param_b", self.act_samp.second_feat)
