import logging
import copy
import pickle
import pandas as pd
import numpy as np

from itertools import chain, product
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import (and_, or_, func, desc,
                        cast, literal, select, union_all)
from sqlalchemy.sql.expression import alias
from sqlalchemy.orm.util import aliased

from felfinder.config import CLF_MAX_VALS, EXT_TO_TYPE, THRESH
from felfinder.models import db, Simils, CommandHistory, DriveHistory, File
from felfinder.utils import rand_prol_id

OLD_MODEL_PATH = 'felfinder/models/full_LogisticRegression_{}.pkl'
MODEL_PATH = 'felfinder/models/predict_{}.pkl'
CLOSE_THRESH = 0.5
TOL = 0.1
DT_EXPLAIN_MAX_DEPTH = 2

FULL_FEATURES = ["tfidf_sim", "word_vec_sim",
                 "color_sim", "obj_sim", 'token_simil',
                 "bin_simil", "perm_simil", "tree_dist",
                 "size_dist", "bigram_simil", "last_mod_simil"]

META_FEATURES = ["bin_simil", "perm_simil", "tree_dist",
                 "size_dist", "bigram_simil", "last_mod_simil"]

IMG_FEATURES = ["color_sim", "obj_sim",
                "bin_simil", "perm_simil", "tree_dist",
                "size_dist", "bigram_simil", "last_mod_simil"]

TEXT_ONLY = ["tfidf_sim", "word_vec_sim",
             "bin_simil", "perm_simil", "tree_dist",
             "size_dist", "bigram_simil", "last_mod_simil"]

FEAT_MAP = {'all_other': META_FEATURES,
            'image': IMG_FEATURES,
            'text': TEXT_ONLY}


def get_simils_from_tuples(uid, sim_tuples):

    """
    Select the entries from the Simils table that correspond
    to the listed tuples.

    NOTE: this specifically only looks at the tuples
    in a particular order, as the sim_tuples are required
    to be fed in the same order as what's in the database
    """

    sim_tbl_name = 'simq' + rand_prol_id()
    stmts = [select([cast(literal(a), db.String(256)).label("A"),
                     cast(literal(b), db.String(256)).label("B")])
             for a, b in sim_tuples if a is not None]

    subq = union_all(*stmts)
    subq = subq.cte(name=sim_tbl_name)
    simils = Simils.query.filter_by(user_id = uid).\
        join(subq, and_(subq.c.A == Simils.file_id_A,
                        subq.c.B == Simils.file_id_B)).all()

    return simils

def get_simils_from_tuples_both(uid, sim_tuples, as_pandas=True):

    """
    Same as the above, but looks at tuples in
    either order in the database
    """

    if not sim_tuples:
        return []


    sim_states = [Simils.query.\
                  filter_by(user_id = uid).\
                  filter(or_(and_(Simils.file_id_A==a,
                                  Simils.file_id_B==b),
                             and_(Simils.file_id_B==a,
                                  Simils.file_id_A==b)))
                  for a, b in sim_tuples]

    if as_pandas:
        simils = pd.concat([pd.read_sql(q.statement, con=db.engine)
                             for q in sim_states])
    else:
        simils = [s.one() for s in sim_states]

    return simils


def get_cmds_from_pairs(uid, pairs, history=False):

    """
    We're given pairs of IDs corresponding to activity in either
    the CommandHistory or DriveHistory table, and we want to
    return the associated objects
    """

    unique_ids = set(chain.from_iterable(pairs))
    base_tbl = CommandHistory if not history else DriveHistory

    commands = base_tbl.query.filter(base_tbl.user_id==uid,
                                     base_tbl.rand_id.in_(unique_ids)).all()

    cmd_look = {c.rand_id: c for c in commands}

    return [(cmd_look[a], cmd_look[b]) for a, b in pairs]


def OLD_get_clf(action):

    """ Pull relevant classifier """

    action_trans = {'move': 'Mov',
                    'find': 'Rel',
                    'del': 'Del'}

    with open(MODEL_PATH.format(action_trans[action]), 'rb') as of:
        clf = pickle.load(of)

    return clf

def get_clf(clf_type):

    """ Pull relevant classifier """

    with open(MODEL_PATH.format(clf_type), 'rb') as of:
        clf = pickle.load(of)

    return clf


def pred_matrix_and_inds(pred_mat, file_id):

    """ Normalize data, select features, and identify
        file ids to predict """

    pred_file_ids = peel_off_fileids(pred_mat, file_id)
    pred_mat = format_for_preds(pred_mat)

    return pred_mat, pred_file_ids

def pred_inds(pred_mat, file_id):

    """ Return the file ids of interest"""

    pred_file_ids = peel_off_fileids(pred_mat, file_id)
    return pred_file_ids


def simil_rows_to_action(simil_rows, action):

    """
    Takes an output from the Simil relation in a pandas
    table and converts it to an list of lists
    with items needed for ActionSamples
    """

    if len(simil_rows) < 1:
        return []

    simil_rows.loc[simil_rows[action + "_cert"].isnull(), action + "_cert"] = 0
    simil_rows.loc[:, action + "_cert"] = simil_rows.loc[:, action + "_cert"].astype('float64')

    simil_rows.loc[:, 'near_miss'] = np.isclose(simil_rows[action + "_cert"],
                                                CLOSE_THRESH, atol=TOL)

    cols = ['file_id_A', 'file_id_B', action + "_top_feat",
            action + "_second_feat", action + "_pred", "near_miss"]
    fill_vals = simil_rows[cols].values.tolist()

    return fill_vals

def simil_rows_for_precomp(uid):

    """
    Return the rows from the Simils table with the file type
    attached to the dataframe, with the dataframe appropriately sorted
    """

    fa_alias = aliased(File, name='fa')
    fb_alias = aliased(File, name='fb')
    q = db.session.query(Simils, fa_alias.file_extension.label('file_id_A_ext'),
                         fb_alias.file_extension.label('file_id_B_ext')).join(fa_alias,
                          and_(fa_alias.user_id==Simils.user_id,
                               fa_alias.id==Simils.file_id_A)).\
                    join(fb_alias,
                         and_(fb_alias.user_id==Simils.user_id,
                              fb_alias.id==Simils.file_id_B)).\
                              filter(Simils.user_id==uid)

    simil_rows = pd.read_sql(q.statement, con=db.engine)

    if len(simil_rows) < 1:
        return []

    simil_rows.loc[:, 'file_type_A'] = simil_rows['file_id_A_ext'].map(EXT_TO_TYPE)
    simil_rows.loc[:, 'file_type_B'] = simil_rows['file_id_B_ext'].map(EXT_TO_TYPE)
    simil_rows.loc[:, 'clf_type'] = 'all_other'
    simil_rows.loc[(simil_rows['file_type_A'] == 'image') & \
                   (simil_rows['file_type_B'] == 'image'), 'clf_type'] = 'image'
    simil_rows.loc[(simil_rows['file_type_A'] == 'text') & \
                   (simil_rows['file_type_B'] == 'text'), 'clf_type'] = 'text'

    simil_rows = simil_rows.sort_values(by='clf_type')

    return simil_rows


def preds_from_simil_rows(uid, simil_rows):

    add_df = None

    actions = ['find', 'move', 'del']
    cols = ['_pred', '_top_feat', '_second_feat', '_cert', '_true_cert']
    pred_cols = list([''.join(x) for x in product(actions, cols)])

    for clf_type in ['all_other', 'image', 'text']:

        rel_rows = simil_rows[simil_rows['clf_type'] == clf_type]
        if len(rel_rows) < 1:
            continue

        full_set = clf_type == 'all_other'
        pred_mat = format_for_preds(rel_rows, FEAT_MAP[clf_type], full_set)

        if len(pred_mat) < 1:
            continue

        clf = get_clf(clf_type)
        _, preds, certainty, true_class_cert, top_feats = clf_pred_outputs(clf, pred_mat,
                                                                           FEAT_MAP[clf_type])

        data = [(bool(p), f1, f2, c, tc) * len(actions)
                for p, (f1, f2), c, tc in zip(preds, top_feats, certainty, true_class_cert)]

        frame_df = pd.DataFrame(data, columns=pred_cols, index=pred_mat.index)

        if add_df is None:
            add_df = frame_df
        else:
            add_df = pd.concat((add_df, frame_df), axis=0)

    return add_df, pred_cols


def dt_preds_from_simil_rows(simil_rows, clf_type = 'all_other'):

    """
    Utility function for DecisionTree classifiers in particular
    """

    add_df = None

    if len(simil_rows) < 1:
        return None, simil_rows

    full_set = clf_type == 'all_other'
    pred_mat = format_for_preds(simil_rows, FEAT_MAP[clf_type], full_set)

    if len(pred_mat) < len(simil_rows):
        clf_type = "all_other"
        pred_mat = format_for_preds(simil_rows, FEAT_MAP[clf_type], full_set, dt_preds = True)

    null_rows = simil_rows.loc[:, FEAT_MAP[clf_type]].isna().any(axis=1)

    clf = DecisionTreeClassifier(max_depth = DT_EXPLAIN_MAX_DEPTH, random_state = 0)
    clf.fit(pred_mat, simil_rows['temp_preds'])

    preds = clf.predict(pred_mat)
    nodes = clf.apply(pred_mat)

    simil_rows['dt_preds'] = preds
    simil_rows['dt_nodes'] = nodes

    return clf, simil_rows, clf_type



def precompute_preds(uid):

    """
    This function takes all similarities computed
    for a participant, and predicts whether the items
    should be found, moved, or deleted together.
    """

    simil_rows = simil_rows_for_precomp(uid)

    if len(simil_rows) < 1:
        return

    add_df, pred_cols = preds_from_simil_rows(uid, simil_rows)

    if add_df is None:
        return

    simil_rows = simil_rows.drop(columns=pred_cols)
    simil_rows = simil_rows.join(add_df)


    simil_rows.to_sql('precomp_preds_' + uid, con=db.engine)

    try:
        q = """UPDATE simils
               SET find_pred=jt.find_pred,
                   find_top_feat=jt.find_top_feat,
                   find_second_feat=jt.find_second_feat,
                   find_cert=jt.find_cert,
                   find_true_cert=jt.find_true_cert,
                   move_pred=jt.move_pred,
                   move_top_feat=jt.move_top_feat,
                   move_second_feat=jt.move_second_feat,
                   move_cert=jt.move_cert,
                   move_true_cert=jt.move_true_cert,
                   del_pred=jt.del_pred,
                   del_top_feat=jt.del_top_feat,
                   del_second_feat=jt.del_second_feat,
                   del_cert=jt.del_cert,
                   del_true_cert=jt.del_true_cert
               FROM precomp_preds_{} AS jt
               WHERE jt.user_id=simils.user_id AND
                     jt."file_id_A"=simils."file_id_A" AND
                     jt."file_id_B"=simils."file_id_B";""".format(uid)

        db.engine.execute(q)
    except Exception as e:
        logging.error("Unable to precompute predictions due to: {}".format(e))
    finally:
        db.engine.execute("DROP TABLE precomp_preds_" + uid + ";")

def clf_pred_outputs(clf, pred_mat, feats):

    """
    Performs predictions on data frame and
    returns several relevant metrics:

      pred_probs: probabilities of each class in (num_samp, num_class) shape
      preds: predicted class for each example, (num_samp,) shape
      certainty: probability of the predicted class, (num_samp,) shape
    """

    pred_probs = clf.predict_proba(pred_mat)
    preds = preds_thresh_skl_proba(pred_probs, THRESH)
    certainty = np.array(pred_probs).max(axis=1).tolist()
    true_class_cert = np.array(pred_probs)[:, 1].tolist()
    top_feats = top_params(clf, pred_mat, feats).tolist()

    return pred_probs, preds, certainty, true_class_cert, top_feats


def top_params(clf, pred_mat, feats, n_params=2):

    """
    Assuming clf is a LogisticRegression classifier,
    we're interested in selecting the n_params
    for each example that are the most explanatory
    of the outcome.
    """

    feat_map = {'tfidf_sim': 'Text Topic',
                'word_vec_sim': "Text Contents",
                'color_sim': 'Image Colors',
                'obj_sim': "Objects in Image",
                'token_simil': "Words in Filename",
                'bin_simil': "File Contents",
                'perm_simil': "Persons File is Shared With",
                'tree_dist': "File Location",
                'size_dist': "File Size",
                'bigram_simil': "Letters of Filename",
                "last_mod_simil": "Last Accessed Date"}

    feature_names = [feat_map[f] for f in feats]
    feature_names = np.array(feature_names)

    coef = clf.coef_
    mult_mat = coef * pred_mat.to_numpy()

    top_params = np.argsort(-abs(mult_mat), axis=1)[:, :n_params]

    return feature_names[top_params]


def format_recs_for_storage(pred_file_ids, preds, certainty):

    res = pd.DataFrame(pd.np.column_stack([preds, certainty]),
                       index=pred_file_ids, columns=["pred_class", "max_pred"])

    res = res.sort_values(by=["pred_class", "max_pred"], ascending=False)

    return res


def custom_rescale(tbl, feats, max_train_vals = CLF_MAX_VALS):

    """
    The MinMaxScaler is generally useful, but unfortunately,
    some of our values have unbounded range. Correspondingly,
    the table maximum might differ from participant to participant,
    which is not acceptable. We therefore took the maximum values
    from our training data as our effective maximums: every value
    in a prediction table that is outside that range is coerced
    to that range before min-max scaling occurs.
    """

    for f in feats:
        train_max = max_train_vals[f]
        train_min = 0.0

        tbl.loc[(tbl[f] > train_max) & ~(tbl[f].isnull()), f] = train_max
        tbl.loc[(tbl[f] < train_min) & ~(tbl[f].isnull()), f] = train_min

        tbl.loc[:, f] = (tbl[f] - train_min) / (train_max - train_min)

    # mmxs = MinMaxScaler()
    # tbl[feats] = mmxs.fit_transform(tbl[feats])

    return tbl



def rescale(pred_mat, features):

    """
    Replace nulls with minimum or maximum values along that dimension
    """

    dist_metrs = ['edit_dist', 'tree_dist', 'size_dist',
                  'last_mod_simil', 'color_sim']

    for f in features:

        col = pred_mat.loc[:, f]
        if f in dist_metrs:
            pred_mat.loc[:, f][pred_mat[f].isnull()] = 1.0
            pred_mat.loc[:, f][pred_mat[f] == np.nan] = 1.0
            pred_mat.loc[:, f] = abs(1 - pred_mat.loc[:, f])
        else:
            pred_mat.loc[:, f][pred_mat[f].isnull()] = 0.0
            pred_mat.loc[:, f][pred_mat[f] == np.nan] = 0.0

    return pred_mat


def drop(pred_mat, features):
    inds = pred_mat.isna().any(axis=1)
    pred_mat = pred_mat[~inds]

    dist_metrs = ['edit_dist', 'tree_dist', 'size_dist',
                  'last_mod_simil', 'color_sim']

    for f in features:
        if f in dist_metrs:
            pred_mat.loc[:, f] = abs(1 - pred_mat[f])

    return pred_mat.loc[~inds, :]



def zero_replace(pred_mat, features):
    # inds = pred_mat.isna().any(axis=1)
    # pred_mat = pred_mat[~inds]

    dist_metrs = ['edit_dist', 'tree_dist', 'size_dist',
                  'last_mod_simil', 'color_sim']

    overlap_d_metrs = [d for d in dist_metrs if d in features]

    pred_mat.loc[:, overlap_d_metrs] = pred_mat.loc[:, overlap_d_metrs].fillna(1.0)
    pred_mat.loc[:, pred_mat.columns.difference(overlap_d_metrs)] = \
        pred_mat.loc[:, pred_mat.columns.difference(overlap_d_metrs)].fillna(0.0)

    for f in features:
        if f in dist_metrs:
            pred_mat.loc[:, f] = abs(1 - pred_mat[f])

    return pred_mat



def format_for_preds(pred_mat, features, full_set = False, dt_preds = False):

    """
    Transforming data features for predictions. If full_set,
    then we rescale null values, whereas if they are sub-frames,
    where we do not expect nulls, then we drop them
    """

    feature_order = features

    pred_mat = pred_mat.loc[:, features]

    for feat_col in features:
        pred_mat.loc[:, feat_col] = pred_mat.loc[:, feat_col].astype(float)
        if feat_col == "size_dist" or feat_col == "last_mod_simil":
            pred_mat.loc[:, feat_col] = np.log(1 + pred_mat.loc[:, feat_col])

    pred_mat = custom_rescale(pred_mat, features)

    if not dt_preds:
        pred_mat = drop(pred_mat, features)
    else:
        pred_mat = zero_replace(pred_mat, features)
    #pred_mat = rescale(pred_mat, features)

    # if full_set:
    #     pred_mat = rescale(pred_mat, features)
    # else:
    #     pred_mat = drop(pred_mat, features)

    return pred_mat[feature_order]


def peel_off_fileids(pred_mat, main_fileid):

    """
    For similarity pairs, we're filtering for one file pair that just had
    the action applied, and now we want to return a list of the other
    file ID in the pair to apply it to

    Arguments:
      pred_mat (pd.DataFrame): Dataframe of the Simils table with relevant items
      main_fileid (str): ID of file that had action just applied

    Returns:
      (list): List of same length as DF that has the ID of the file in the pair
              that isn't main_fileid

    """

    idsA = pred_mat['file_id_A']
    idsB = pred_mat['file_id_B']

    full_ids = idsA
    full_ids.loc[full_ids == main_fileid] = idsB

    return full_ids

def preds_thresh_skl_proba(proba, thresh):

    alt_proba = copy.deepcopy(proba)
    alt_proba[:, 0] = thresh

    return np.argmax(alt_proba, axis=1)
