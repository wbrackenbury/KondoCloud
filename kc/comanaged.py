import random
import numpy as np
import pandas as pd

from itertools import combinations
from sqlalchemy import and_, or_, func, desc, cast, literal, select, union_all

from felfinder.config import MAX_ACT_SAMPS_NEEDED, MAX_SELECT
from felfinder.models import (db, DriveHistory, DriveFile,
                              CommandHistory, CommandFiles, Simils, ActionSample)
from felfinder.pred_utils import (get_simils_from_tuples_both,
                                  pred_matrix_and_inds, format_for_preds,
                                  get_cmds_from_pairs, simil_rows_to_action)
from felfinder.recommend import (get_clf, pred_matrix_and_inds, clf_pred_outputs)
from felfinder.utils import get_rand_id

SECS_IN_MIN = 60
THIRTY_MINS = SECS_IN_MIN * 30
TWO_MINS = SECS_IN_MIN * 2
FIVE_MINS = SECS_IN_MIN * 5

def sessions(df, time_arr, interval, hist=False):

    """
    Returns a pandas column that numbers each
    time by what session it fell into, where
    each session consists of times separated
    by no more than (interval).

    NOTE: REQUIRES VALUES ARE SORTED BY TIME

    """

    time_col = 'timestamp' if hist else 'time_run'

    diffs = np.diff(time_arr)

    (sess_splits,) = np.where(diffs > interval)
    sess_splits += 1

    # At this point, sess_splits contains the indexes
    # of the immediate successors to the
    # last action in a session.
    # We therefore need to add a zeroth index
    # point in order to create the left
    # edge of a bin, and also create another
    # bin edge just beyond the max value.
    # We do this by taking the argmax of the
    # series, taking the values
    # of the series at these indices, and
    # adding 1 to the maximum.
    # This will turn this into a series of left-aligned bins

    sess_splits = np.hstack((np.array([0]),
                             sess_splits,
                             np.array([np.argmax(time_arr)])))

    binvals = df.iloc[sess_splits][time_col]

    z = np.zeros(len(binvals), dtype=np.uint8)
    z[-1] += 1
    binvals += z

    return pd.cut(df[time_col], bins=binvals,
                  right=False, labels=False)


def comanaged_moves(df, history=True):

    """
    Get file pairs from the participant's Google
    Drive command or activity history that were moved
    to the same location during a session
    """

    # Subset actions of interest
    # NOTE: by doing this before creating the sessions variable,
    # the session refers specifically to move actions
    # instead of any action
    if history:
        local_df = df[(df['cmd'] == 'move') & (df['copy'] == False)]
        time_col = 'timestamp'
    else:
        local_df = df[(df['cmd'] == 'paste')]
        time_col = 'time_run'

    if len(local_df) < 1:
        return [], []

    # Break actions into sessions
    local_df = local_df.sort_values(by=time_col)
    time_arr = local_df[time_col].to_numpy()
    if history:
        local_df['session'] = sessions(local_df, time_arr, THIRTY_MINS, history)
    else:
        local_df['session'] = sessions(local_df, time_arr, TWO_MINS, history)

    # Find target folder for each file that was moved
    files_df = local_df[(~(local_df['file_id'].isnull()))].\
        drop('folder_id', axis=1)
    folds_df = local_df[(~(local_df['folder_id'].isnull())) & \
                        (local_df['target'] == False)].\
        drop('file_id', axis=1)

    alt_df = files_df.join(folds_df.set_index('cmd_id'),
                           on='cmd_id', rsuffix='_y').\
                           rename(columns={'folder_id': 'target_folder_id'})

    alt_df = alt_df.reset_index()

    sess_vals = set(alt_df['session'].values.tolist())

    if len(sess_vals) == 0:
        return [], []

    LIMIT_PER = max(int(MAX_SELECT / len(sess_vals)), 1)

    for s in sess_vals:
        sess_df = alt_df[alt_df['session'] == s]
        to_samp = min(LIMIT_PER, len(sess_df))
        sampled = sess_df.sample(n=to_samp, random_state=0)
        to_drop = sess_df[~sess_df.index.isin(sampled.index)]
        alt_df = alt_df[~(alt_df.index.isin(to_drop.index))]

    # Take actions in the same session that moved to the same folder
    final_df = alt_df.join(alt_df.set_index(['session', 'target_folder_id']),
                           on=['session', 'target_folder_id'],
                           lsuffix='_a', rsuffix='_b')

    # Subset actions that were not part of the same command
    #final_df = final_df[final_df['cmd_id_a'] != final_df['cmd_id_b']]

    final_df = final_df[final_df['file_id_a'] != final_df['file_id_b']]

    # Drop duplicate pairs where file_a, file_b == file_b, file_a
    sort_vals = final_df['file_id_a'] < final_df['file_id_b']
    final_df.loc[sort_vals, 'file_id_tup'] = final_df.loc[sort_vals, 'file_id_a'].\
        str.cat(final_df.loc[sort_vals, 'file_id_b'], sep=',')
    final_df.loc[~sort_vals, 'file_id_tup'] = final_df.loc[~sort_vals, 'file_id_b'].\
        str.cat(final_df.loc[~sort_vals, 'file_id_a'], sep=',')
    final_df = final_df.drop_duplicates(subset='file_id_tup')

    ret_vals = final_df[['file_id_a', 'file_id_b', 'cmd_id_a', 'cmd_id_b']].values.tolist()

    id_pairs = [(a, b) for a, b, _, _ in ret_vals]
    cmd_pairs = [(a, b) for _, _, a, b in ret_vals]

    return id_pairs, cmd_pairs


def comanaged_deletes(df):

    """
    Get file pairs from the participant's
    command history that were deleted
    during the same session, and the commands
    that deleted them
    """

    # Subset actions of interest
    alt_df = df[(df['cmd'] == 'rm')]

    if len(alt_df) < 1:
        return [], []

    # Break actions into sessions
    alt_df = alt_df.sort_values(by='time_run')
    time_arr = alt_df['time_run'].to_numpy()
    alt_df['session'] = sessions(alt_df, time_arr, TWO_MINS)

    cmg_del_pairs = []

    sess_vals = set(alt_df['session'].values.tolist())
    if len(sess_vals) == 0:
        return [], []

    # Prevent taking too many pairs to sample
    LIMIT_PER = max(int(MAX_SELECT / len(sess_vals)), 1)

    for s in sess_vals:
        sub_df = alt_df[alt_df['session'] == s]
        del_files_in_sess = [tuple(x) for x in sub_df[['file_id', 'cmd_id']].\
                                 to_records(index=False)]
        random.shuffle(del_files_in_sess)
        del_files_in_sess = set(del_files_in_sess[:LIMIT_PER])

        del_pairs_in_sess = [(a, b) for (a, b) in
                             combinations(del_files_in_sess, 2)
                             if a != b]
        cmg_del_pairs.extend(del_pairs_in_sess)

    id_pairs = [(a, b) for (a, _), (b, _) in cmg_del_pairs]
    cmd_pairs = [(a, b) for (_, a), (_, b) in cmg_del_pairs]

    return id_pairs, cmd_pairs

def new_folders_move():

    """
    We want to examine new folders that participants
    created and then moved files to
    """

    q = "SELECT * FROM drive_history,drive_file WHERE drive_history.rand_id=drive_file.cmd_id;"
    df = pd.read_sql(q, con=db.engine)

    new_folds_df = df[(df['cmd'] == 'create') & (~df['folder_id'].isnull()) \
                      & (df['file_id'].isnull())]

    new_folds = new_folds_df['folder_id'].values.tolist()

    files_df = df[(~(df['file_id'].isnull()))].drop('folder_id', axis=1)
    folds_df = df[(~(df['folder_id'].isnull()))].drop('file_id', axis=1)

    alt_df = files_df.join(folds_df.set_index('cmd_id'),
                           on='cmd_id', rsuffix='_y').\
                           rename(columns={'folder_id': 'target_folder_id'})

    to_new_folder = alt_df[alt_df['target_folder_id'].isin(new_folds)]

    return set(to_new_folder['file_id'].values.tolist())


def new_folders_query(uid, hist = False):

    if not hist:

        folds = db.session.query(CommandHistory, CommandFiles).\
            filter(and_(CommandHistory.user_id==uid, CommandHistory.cmd=='mkdir')).\
            join(CommandFiles, and_(CommandFiles.user_id==CommandHistory.user_id,
                                    CommandFiles.cmd_id==CommandHistory.rand_id)).all()

    else:

        folds = db.session.query(DriveHistory, DriveFile).\
            filter(and_(DriveHistory.user_id==uid,
                        DriveHistory.cmd=='create',
                        DriveHistory.cmd_subtype=='new')).\
            join(DriveFile, and_(DriveFile.user_id==DriveHistory.user_id,
                                 DriveFile.cmd_id==DriveHistory.rand_id)).\
                                 filter(DriveFile.folder_id != None).all()

    return folds


def new_folders(uid, hist = False):

    """
    Look at newly created folders for sampling
    """

    folds = new_folders_query(uid, hist)

    ids = [df.folder_id for dh, df in folds]
    cmds = [dh.rand_id for dh, df in folds]

    return ids, cmds

def new_folder_pairs(uid, hist = False):

    """
    Look at newly created folders for sampling
    """

    folds = new_folders_query(uid, hist)

    id_pairs = [(dfa.folder_id, dfb.folder_id) for (_, dfa), (_, dfb) in
                combinations(folds, 2)
                if dfa.folder_id != dfb.folder_id]

    cmd_pairs = [(dha.rand_id, dhb.rand_id) for (dha, _), (dhb, _) in
                 combinations(folds, 2)
                 if dha.rand_id != dhb.rand_id]

    return id_pairs, cmd_pairs


def get_hist_cmg_info(uid):

    q = """SELECT * FROM drive_history, drive_file
           WHERE drive_history.user_id=drive_file.user_id AND
           drive_history.rand_id=drive_file.cmd_id AND
           (drive_file.folder_id IN
                (SELECT id FROM Folder) OR
            drive_file.file_id IN
                (SELECT id FROM File)) AND drive_file.user_id=%(uid)s;"""

    df = pd.read_sql(q, con=db.engine, params=[{'uid': uid}])

    ret_vals = comanaged_moves(df, history=True)
    return ret_vals


def get_cmd_del_cmg(uid):

    q = db.session.query(CommandHistory, CommandFiles).\
        filter(CommandHistory.user_id==uid).\
        join(CommandFiles, and_(CommandFiles.user_id==CommandHistory.user_id,
                                CommandFiles.cmd_id==CommandHistory.rand_id,
                                CommandFiles.file_id != None))

    df = pd.read_sql(q.statement, con=db.engine)

    return comanaged_deletes(df)


def get_cmd_cmg_move_pairs(uid):

    q = db.session.query(CommandHistory, CommandFiles).\
        filter(CommandHistory.user_id==uid).\
        join(CommandFiles, and_(CommandFiles.user_id==CommandHistory.user_id,
                                CommandFiles.cmd_id==CommandHistory.rand_id))

    df = pd.read_sql(q.statement, con=db.engine)

    return comanaged_moves(df, False)

def get_false_positives(uid, cmgd_pairs, action, sample_reason = ""):

    """
    We want to find file pairs that we said should be co-managed,
    but weren't co-managed in practice. We want to ask participants
    about whether they would have co-managed them if given the chance,
    or if it's an incorrect prediction
    """

    q = Simils.query.filter_by(user_id = uid)
    simil_rows = pd.read_sql(q.statement, con=db.engine)

    if len(simil_rows) < 1:
        return []

    sort_fid_a = simil_rows[['file_id_A', 'file_id_B']].min(axis=1)
    sort_fid_b = simil_rows[['file_id_A', 'file_id_B']].max(axis=1)

    simil_rows['unique_pair'] = "(" + sort_fid_a + ", " + sort_fid_b + ")"
    simil_rows = simil_rows[~(simil_rows['unique_pair'].isin(cmgd_pairs))]

    if len(simil_rows) < 1:
        return []

    simil_rows = simil_rows.sample(n=min(MAX_ACT_SAMPS_NEEDED,
                                         len(simil_rows)),
                                   axis=0)

    fill_vals = simil_rows_to_action(simil_rows, action)


    print(fill_vals)

    # Return sample objects with commands and prediction info
    full_samples = [ActionSample(rand_id = get_rand_id(), user_id= uid,
                                 file_id_a = fa, file_id_b = fb,
                                 pred = pred, top_feat = feat_a,
                                 second_feat = feat_b,
                                 near_miss = near_miss,
                                 corr_pred = pred < 1 if pred is not None else False,
                                 hist = False,
                                 sample_reason = sample_reason)
                    for fa, fb, feat_a, feat_b, pred, near_miss,
                    in fill_vals]

    return full_samples


def bulk_folders_to_samples(uid, ids, cmds, hist):

    sample_reason = "cmd_new_folder" if not hist else "hist_new_folder"

    full_samples = [ActionSample(rand_id = get_rand_id(),
                                 user_id= uid,
                                 cmd_id_a = cmd,
                                 folder_id = fid,
                                 hist = hist,
                                 sample_reason = sample_reason)
                    for cmd, fid in zip(cmds, ids)]

    return full_samples

def bulk_folder_pairs_to_samples(uid, ids, cmds, hist):

    sample_reason = "cmd_new_folder" if not hist else "hist_new_folder"

    # I don't think we need an action_sample folder_id

    full_samples = [ActionSample(rand_id = get_rand_id(),
                                 user_id= uid,
                                 cmd_id_a = cmd_a,
                                 cmd_id_b = cmd_b,
                                 hist = hist,
                                 sample_reason = sample_reason)
                    for (cmd_a, cmd_b), (_, _) in zip(cmds, ids)]

    return full_samples


def bulk_pairs_to_samples(uid, id_pairs, cmd_pairs, action,
                          hist = False, sample_reason = ""):

    """
    Take a set of file id pairs and associated commands that
    act on them, and turn these into samples to later
    sample and ask participants about
    """

    # Perform bulk predictions
    simil_rows = get_simils_from_tuples_both(uid, id_pairs)

    if len(simil_rows) < 1:
        return

    fill_vals = simil_rows_to_action(simil_rows, action)

    # Return sample objects with commands and prediction info
    full_samples = [ActionSample(rand_id = get_rand_id(), user_id= uid,
                                 cmd_id_a = cmd_a, cmd_id_b = cmd_b,
                                 file_id_a = fa, file_id_b = fb,
                                 pred = pred, top_feat = feat_a,
                                 second_feat = feat_b, near_miss = near_miss,
                                 corr_pred = pred, hist = hist,
                                 sample_reason = sample_reason)
                    for (cmd_a, cmd_b), \
                    (fa, fb, feat_a, feat_b, pred, near_miss)
                    in zip(cmd_pairs, fill_vals)]

    return full_samples
