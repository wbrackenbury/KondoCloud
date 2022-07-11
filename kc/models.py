from flask_sqlalchemy import SQLAlchemy
import time
import pprint

from felfinder.config import THRESH

NUM_ATTEMPTS = 15
NAME_PATCH_LEN = 240

for i in range(NUM_ATTEMPTS):
    try:
        db = SQLAlchemy()
    except:
        time.sleep(5)

db = SQLAlchemy()

class User(db.Model):

    id = db.Column(db.String(256), unique=True, index=True, primary_key=True)

    # OAuth2 token items
    access_token = db.Column(db.String(256))
    refresh_token = db.Column(db.String(256))
    #id_token = db.Column(db.String(256))
    token_uri = db.Column(db.String(256))

    prolific_id = db.Column(db.String(256))

    cwd = db.Column(db.String(256), default = 'root')
    init = db.Column(db.Boolean, default = False)

    info_loaded = db.Column(db.Boolean, default = False)
    acc_changes = db.Column(db.Boolean, default = False)

    start_org = db.Column(db.BigInteger)
    end_org = db.Column(db.BigInteger)

    exp_condition = db.Column(db.String(64), default = 'reg_int')

    num_action_qs = db.Column(db.Integer(), default = 0)

    move_num_updates = db.Column(db.Integer(), default = 0)
    move_thresh = db.Column(db.Float(), default = THRESH)

    del_num_updates = db.Column(db.Integer(), default = 0)
    del_thresh = db.Column(db.Float(), default = THRESH)


class File(db.Model):

    rand_id = db.Column(db.UnicodeText, primary_key=True)
    user_id = db.Column(db.UnicodeText, db.ForeignKey(
        User.id), primary_key=True)
    id = db.Column(db.UnicodeText, primary_key=True, index=True) # ID from API

    # For files that we copy, in order to do replay, we need to save
    # the ID that Google eventually assigns to the file so we can
    # use that to later put files into that folder
    assigned_id = db.Column(db.UnicodeText)

    name = db.Column(db.UnicodeText)
    original_name = db.Column(db.UnicodeText)
    path = db.Column(db.UnicodeText) #of the form ROOT_PATH + /name
    path_hash = db.Column(db.UnicodeText)
    parent_id = db.Column(db.UnicodeText)
    parent_hash = db.Column(db.UnicodeText)
    original_parent_id = db.Column(db.UnicodeText)
    original_parent_hash = db.Column(db.UnicodeText)

    file_extension = db.Column(db.String(128))
    size = db.Column(db.BigInteger, default=0)
    last_modified = db.Column(db.BigInteger) # Might want to modify to the Unix time
    goog_mime_type = db.Column(db.String(128))
    elf_mime_type = db.Column(db.String(128))

    is_shared = db.Column(db.Boolean, default=False)
    is_gdoc = db.Column(db.Boolean, default=False)
    is_owner = db.Column(db.Boolean, default=True) # owner, editor, etc.
    trashed = db.Column(db.Boolean, default=False)

    created_by_study = db.Column(db.Boolean, default=False)
    created_time = db.Column(db.BigInteger)

    media_info = db.Column(db.UnicodeText) # For images / video on Drive
    doc_metadata = db.Column(db.UnicodeText) # For pdf / docx files
    sel_for_simils = db.Column(db.Boolean, default=False)

    view_link = db.Column(db.UnicodeText)

    error=db.Column(db.Boolean, default=False)

class Folder(db.Model):

    rand_id = db.Column(db.UnicodeText, primary_key=True)
    user_id = db.Column(db.UnicodeText, db.ForeignKey(
        User.id), primary_key=True)
    id = db.Column(db.UnicodeText, primary_key=True, index=True) # ID from API

    # For folders that we create, in order to do replay, we need to save
    # the ID that Google eventually assigns to the folder so we can
    # use that to later put files into that folder
    assigned_id = db.Column(db.UnicodeText)

    name = db.Column(db.UnicodeText)
    original_name = db.Column(db.UnicodeText)
    path = db.Column(db.UnicodeText)
    path_hash = db.Column(db.UnicodeText)
    parent_id = db.Column(db.UnicodeText)
    parent_hash = db.Column(db.UnicodeText)
    original_parent_id = db.Column(db.UnicodeText)
    original_parent_hash = db.Column(db.UnicodeText)

    size = db.Column(db.BigInteger, default=0)
    last_modified = db.Column(db.BigInteger) # Might want to modify to the Unix time

    is_shared = db.Column(db.Boolean, default=False)
    is_owner = db.Column(db.Boolean, default=True) # owner, editor, etc.
    trashed = db.Column(db.Boolean, default=False)

    created_by_study = db.Column(db.Boolean, default=False)
    created_time = db.Column(db.BigInteger)

    view_link = db.Column(db.UnicodeText)

    error=db.Column(db.Boolean, default=False)

# This table logs the following commands:
#   paste
#   duplicate
#   rm
#   open
#   rename
#   mkdir

# We use regular logs for others

class CommandHistory(db.Model):

    rand_id = db.Column(db.UnicodeText, primary_key=True)
    user_id = db.Column(db.UnicodeText,
                        db.ForeignKey(User.id),
                        primary_key=True)

    time_run = db.Column(db.BigInteger, default=0)

    cmd = db.Column(db.String(32))

    replayed = db.Column(db.Boolean, default = False)
    sel_for_replay = db.Column(db.Boolean, default = False)
    sel_for_qs_ind = db.Column(db.Integer(), default = -1)

    # Parameter for rename and mkdir commands
    old_name = db.Column(db.String(256))
    new_name = db.Column(db.String(256))
    copy = db.Column(db.Boolean, default = False)

    # Whether command is an inversion of previous recommendation
    inv = db.Column(db.Boolean, default = False)

    # If the command is an accepted recommendation, we note that here
    rec_result_id = db.Column(db.UnicodeText)

    # Can be nullable, but associated with a HeaderRow object
    assoc_header = db.Column(db.UnicodeText)

    def __repr__(self):


        json = {"rand_id": self.rand_id,
                "user_id": self.user_id,
                "time_run": self.time_run,
                "cmd": self.cmd,
                "replayed": self.replayed,
                "sel_for_replay": self.sel_for_replay,
                "sel_for_qs_ind": self.sel_for_qs_ind,
                "old_name": self.old_name,
                "new_name": self.new_name,
                "copy": self.copy,
                "inv": self.inv,
                "rec_result_id": self.rec_result_id,
                "assoc_header": self.assoc_header}

        return pprint.pformat(json)


class CommandFiles(db.Model):

    """
    IDs of files and folders associated with a given command in the
    CommandHistory table, either as a target or destination
    """

    rand_id = db.Column(db.UnicodeText,
                        primary_key=True)
    user_id = db.Column(db.UnicodeText,
                        db.ForeignKey(User.id),
                        primary_key=True)
    cmd_id = db.Column(db.UnicodeText,
                       primary_key=True)

    # True if target, False if destination
    target = db.Column(db.Boolean, default = True)

    # The ID of the item that was created from this file
    # via a duplicate operation
    copy_child = db.Column(db.UnicodeText)

    # Controls the order in which we perform replay
    # on this file. 0 is earliest, larger is later
    recurse_order = db.Column(db.Integer(), default = 0)

    # If from a move or duplicate, indicates the
    # ID of the previous parent
    old_parent_id = db.Column(db.UnicodeText)

    file_id = db.Column(db.UnicodeText)
    folder_id = db.Column(db.UnicodeText)


# For recommendations, we store essentially all the
# recommendations that we generate, but we only
# want to display ones to the user / participant
# that obey some properties. Therefore we have many
# of the same filter conditions applied to the
# Recommends table. We store these here.

VALID_REC_CONDS = {"deleted": False,
                   "done_no_acc": False,
                   "faded": False,
                   "invalidated": False,
                   "time_polite": True,
                   "not_done": True,
                   "is_only_for_id": True}


class Recommends(db.Model):

    """
    Storage for recommendations provided to a user
    """
    rand_id = db.Column(db.UnicodeText,
                        primary_key=True)

    user_id = db.Column(db.String(256),
                        primary_key=True)
    cmd_id = db.Column(db.UnicodeText,
                       primary_key=True)

    rec_path_hash = db.Column(db.UnicodeText)
    rec_name = db.Column(db.UnicodeText)
    dst_hash = db.Column(db.UnicodeText) #hash of dst folder
    dst_name = db.Column(db.UnicodeText) #Name of dst folder

    explain_name = db.Column(db.UnicodeText)
    explain_hash = db.Column(db.UnicodeText)

    explain_target = db.Column(db.UnicodeText)
    explain_dst = db.Column(db.UnicodeText)

    parent_hash = db.Column(db.UnicodeText)
    parent_path = db.Column(db.UnicodeText)
    parent_id = db.Column(db.UnicodeText)

    timestamp = db.Column(db.BigInteger)

    action = db.Column(db.String(32))
    strength = db.Column(db.Float())
    num_remain = db.Column(db.Integer())

    sent = db.Column(db.Boolean, default = False)
    time_polite = db.Column(db.Boolean, default = False)
    is_only_for_id = db.Column(db.Boolean, default = False)
    not_done = db.Column(db.Boolean, default = False)

    accepted = db.Column(db.Boolean, default = False)
    deleted = db.Column(db.Boolean, default = False)
    done_no_acc = db.Column(db.Boolean, default = False)
    faded = db.Column(db.Boolean, default = False)
    # If a file is removed, we shall do nothing more on its recommendation
    # Or if the file is moved somewhere else, the rec is no longer valid
    invalidated = db.Column(db.Boolean, default = False)

    to_inv = db.Column(db.Boolean, default = False)

    is_rand = db.Column(db.Boolean, default = False)

    def __repr__(self):

        json = { 'rand_id': self.rand_id,
                 'user_id': self.user_id,
                 'cmd_id': self.cmd_id,
                 'rec_path_hash': self.rec_path_hash,
                 'rec_name': self.rec_name,
                 'dst_hash': self.dst_hash,
                 'dst_name': self.dst_name,
                 'explain_name': self.explain_name,
                 'explain_hash': self.explain_hash,
                 'explain_target': self.explain_target,
                 'explain_dst': self.explain_dst,
                 'parent_hash': self.parent_hash,
                 'parent_path': self.parent_path,
                 'parent_id': self.parent_id,
                 'timestamp': self.timestamp,
                 'action': self.action,
                 'strength': self.strength,
                 'num_remain': self.num_remain,
                 'sent': self.sent,
                 'time_polite': self.time_polite,
                 'is_only_for_id': self.is_only_for_id,
                 'not_done': self.not_done,
                 'accepted': self.accepted,
                 'deleted': self.deleted,
                 'done_no_acc': self.done_no_acc,
                 'faded': self.faded,
                 'invalidated': self.invalidated,
                 'to_inv': self.to_inv,
                 'is_rand': self.is_rand
        }

        return pprint.pformat(json)


class Simils(db.Model):
    #User who owns both files
    user_id = db.Column(db.String(256),
                        db.ForeignKey(User.id),
                        primary_key=True)
    #Filename of first file in comparison
    filename_A = db.Column(db.UnicodeText)
    #ID of of first file in comparison
    file_id_A = db.Column(db.String(256),
                           primary_key=True)
    #Filename of second file in comparison
    filename_B = db.Column(db.UnicodeText)
    #ID of second file in comparison
    file_id_B = db.Column(db.String(256),
                           primary_key=True)
    #Edit distance between the two strings
    edit_dist = db.Column(db.Integer())
    #Jaccard similarity of binary content
    bin_simil = db.Column(db.Float())
    #Modified Jaccard similarity of permissions
    perm_simil = db.Column(db.Float())
    #Tree distance between files
    tree_dist = db.Column(db.Integer())
    #difference in file size (absolute)
    size_dist = db.Column(db.Float())
    #similarity in last time modified score
    last_mod_simil = db.Column(db.Float())
    #similarity in schema (for spreadsheets)
    schema_sim = db.Column(db.Float())
    #Filename bigram similarity
    bigram_simil = db.Column(db.Float())
    #Similarity of the colors in the image, average of total diff
    color_sim = db.Column(db.Float())
    #Similarity of predicted objects
    obj_sim = db.Column(db.Float())
    #Similarity of the keywords
    tfidf_sim = db.Column(db.Float())
    #word2vec similarity
    word_vec_sim = db.Column(db.Float())
    #token similarity
    token_simil = db.Column(db.Float())


    # Precompute recommendations for this similarity pair
    # per each participant
    find_pred = db.Column(db.Boolean, default=False)
    find_top_feat = db.Column(db.UnicodeText)
    find_second_feat = db.Column(db.UnicodeText)
    find_cert = db.Column(db.Float())
    find_true_cert = db.Column(db.Float())
    move_pred = db.Column(db.Boolean, default=False)
    move_top_feat = db.Column(db.UnicodeText)
    move_second_feat = db.Column(db.UnicodeText)
    move_cert = db.Column(db.Float())
    move_true_cert = db.Column(db.Float())
    del_pred = db.Column(db.Boolean, default=False)
    del_top_feat = db.Column(db.UnicodeText)
    del_second_feat = db.Column(db.UnicodeText)
    del_cert = db.Column(db.Float())
    del_true_cert = db.Column(db.Float())


    def __repr__(self):

        json = {
            'user_id': self.user_id,
            'filename_A': self.filename_A,
            'file_id_A': self.file_id_A,
            'filename_B': self.filename_B,
            'file_id_B': self.file_id_B,
            'edit_dist': self.edit_dist,
            'bin_simil': self.bin_simil,
            'perm_simil': self.perm_simil,
            'tree_dist': self.tree_dist,
            'size_dist': self.size_dist,
            'last_mod_simil': self.last_mod_simil,
            'schema_sim': self.schema_sim,
            'bigram_simil': self.bigram_simil,
            'color_sim': self.color_sim,
            'obj_sim': self.obj_sim,
            'tfidf_sim': self.tfidf_sim,
            'word_vec_sim': self.word_vec_sim,
            'find_pred': self.find_pred,
            'find_top_feat': self.find_top_feat,
            'find_second_feat': self.find_second_feat,
            'find_cert': self.find_cert,
            'move_pred': self.move_pred,
            'move_top_feat': self.move_top_feat,
            'move_second_feat': self.move_second_feat,
            'move_cert': self.move_cert,
            'del_pred': self.del_pred,
            'del_top_feat': self.del_top_feat,
            'del_second_feat': self.del_second_feat,
            'del_cert': self.del_cert
        }

        return pprint.pformat(json)


db.Index('simils_composite', Simils.file_id_A, Simils.file_id_B)

class ImageFeats(db.Model):

    user_id = db.Column(db.String(256), db.ForeignKey(
        User.id), primary_key=True)
    # file id as per Google API
    id = db.Column(db.String(256), primary_key=True, index=True)
    # labels score and topicality in tuples in the order: (label, score, tpoicality)
    labels = db.Column(db.UnicodeText, default=None)
    colors = db.Column(db.UnicodeText, default=None)
    # web based entity labels in tuples in the order (label, score). Some entities don't have code
    web_labels = db.Column(db.UnicodeText, default=None)
    # best guess labels list of tuples in the order (label, language_code). Language code optional
    best_guess_labels = db.Column(db.UnicodeText, default=None)
    # utf-8 based string of any OCR based text in the image
    text = db.Column(db.UnicodeText, default=None)
    #Elementwise average of RGB pixes
    avg_color = db.Column(db.String(32), default=None)
    # Error variable
    error=db.Column(db.Boolean, default=False)

class Schemas(db.Model):
    rand_id = db.Column(db.UnicodeText,
                        primary_key=True)
    #ID of file feature corresponds to
    file_id = db.Column(db.String(256),
                        primary_key=True)
    #user that owns file
    user_id = db.Column(db.String(256),
                        db.ForeignKey(User.id),
                        primary_key=True)
    #Name of extracted schema header
    feat = db.Column(db.UnicodeText, primary_key=True)

class SharedUsers(db.Model):
    rand_id = db.Column(db.UnicodeText,
                        primary_key=True)
    user_id = db.Column(db.String(256),
                        db.ForeignKey(User.id),
                        primary_key=True)
    # ID of file that user has permissions for
    id = db.Column(db.String(256),
                   primary_key=True,
                   index=True)
    #User that has permissions
    shared_user = db.Column(db.String(256),
                            primary_key=True)

class FileErrors(db.Model):
    user_id = db.Column(db.String(256), db.ForeignKey(
        User.id), primary_key=True)
    file_id = db.Column(db.String(256), primary_key=True)
    error = db.Column(db.String(128), default=False)
    lineno = db.Column(db.Integer)

class DriveHistory(db.Model):

    # We're going to filter out any items that aren't
    # done by the original user

    rand_id = db.Column(db.UnicodeText,
                        primary_key=True,
                        unique=True)
    user_id = db.Column(db.String(256), db.ForeignKey(
        User.id), primary_key=True)

    cmd = db.Column(db.String(64))
    cmd_subtype = db.Column(db.String(64))

    old_name = db.Column(db.String(256))
    new_name = db.Column(db.String(256))
    copy = db.Column(db.Boolean, default = False)

    timestamp = db.Column(db.BigInteger)

class DriveFile(db.Model):

    rand_id = db.Column(db.UnicodeText,
                        primary_key=True,
                        unique=True)
    user_id = db.Column(db.UnicodeText,
                        db.ForeignKey(User.id),
                        primary_key=True)
    cmd_id = db.Column(db.UnicodeText,
                       primary_key=True)

    # True if target, False if destination
    target = db.Column(db.Boolean, default = True)

    # The ID of the item that was created from this file
    # via a duplicate operation
    copy_child = db.Column(db.UnicodeText)

    # If from a move or duplicate, indicates the
    # ID of the previous parent
    old_parent_id = db.Column(db.UnicodeText)

    # If both are present, this is because
    # it was not possible to determine whether
    # it was a file or folder
    file_id = db.Column(db.UnicodeText)
    folder_id = db.Column(db.UnicodeText)

class ActionSample(db.Model):

    """
    Table for sampled actions and action pairs
    to ask participants about in the part 2
    action section. Because we're only
    sampling co-management, if there
    are 2 commands, both will be of
    the same type.
    """

    rand_id = db.Column(db.UnicodeText,
                        primary_key=True,
                        unique=True)
    user_id = db.Column(db.UnicodeText,
                        db.ForeignKey(User.id),
                        primary_key=True)

    cmd_id_a = db.Column(db.UnicodeText)
    cmd_id_b = db.Column(db.UnicodeText)
    file_id_a = db.Column(db.UnicodeText)
    file_id_b = db.Column(db.UnicodeText)
    folder_id = db.Column(db.UnicodeText)
    pred = db.Column(db.Boolean, default=False)
    corr_pred = db.Column(db.Boolean, default=False)
    near_miss = db.Column(db.Boolean, default=False)

    top_feat = db.Column(db.UnicodeText)
    second_feat = db.Column(db.UnicodeText)

    # Whether the action asks about prior activity
    # history or actions made during the study
    hist = db.Column(db.Boolean, default=False)

    sample_reason = db.Column(db.String(64))

    sel_for_qs = db.Column(db.Boolean, default=False)
    ask_order = db.Column(db.Integer(), default = -1)

    def __repr__(self):


        json = {'rand_id': self.rand_id,
                'user_id': self.user_id,
                "cmd_id_a": self.cmd_id_a,
                "cmd_id_b": self.cmd_id_b,
                "file_id_a": self.file_id_a,
                "file_id_b": self.file_id_b,
                "folder_id": self.folder_id,
                "pred": self.pred,
                "corr_pred": self.corr_pred,
                "near_miss": self.near_miss,
                "top_feat": self.top_feat,
                "second_feat": self.second_feat,
                "hist": self.hist,
                "sample_reason": self.sample_reason,
                "ask_order": self.ask_order}

        return pprint.pformat(json)


class GroupSample(db.Model):

    user_id = db.Column(db.String(256),
                        db.ForeignKey(User.id),
                        primary_key=True)
    group_id = db.Column(db.String(256),
                         primary_key=True,
                         index=True)

    base_file_id = db.Column(db.String(256))

    action = db.Column(db.String(128))

    # Not dernormalized, but cheaper for later
    # sampling
    size = db.Column(db.Integer(), default = 0)

    def __repr__(self):

        return pprint.pformat({'user_id': self.user_id,
                               'group_id': self.group_id,
                               'base_file_id': self.base_file_id,
                               'size': self.size})

class GroupExplan(db.Model):

    user_id = db.Column(db.String(256),
                        db.ForeignKey(User.id),
                        primary_key=True)
    group_id = db.Column(db.String(256),
                         primary_key=True,
                         index=True)
    explan_id = db.Column(db.String(256),
                         primary_key=True,
                         index=True)

    exp_type = db.Column(db.String(64))
    exp_blob = db.Column(db.UnicodeText)
    exp_text = db.Column(db.UnicodeText)
    size = db.Column(db.Integer())

    rec = db.Column(db.Float())
    prec = db.Column(db.Float())
    full_score = db.Column(db.Float())

    is_complex = db.Column(db.Boolean, default=False)
    min_discrim_score = db.Column(db.Float())
    max_discrim_score = db.Column(db.Float())

    sel_for_qs_ind = db.Column(db.Integer(), default = -1)
    sample_reason = db.Column(db.UnicodeText)

    time_taken = db.Column(db.Integer())

    def __repr__(self):

        return pprint.pformat({'user_id': self.user_id,
                               'group_id': self.group_id,
                               'explan_id': self.explan_id,
                               'exp_type': self.exp_type,
                               'exp_blob': self.exp_blob,
                               'exp_text': self.exp_text,
                               'rec': self.rec,
                               'prec': self.prec,
                               'score': self.full_score,
                               'size': self.size,
                               'is_complex': self.is_complex,
                               'min_discrim_score': self.min_discrim_score,
                               'max_discrim_score': self.max_discrim_score,
                               'sel_for_qs_ind': self.sel_for_qs_ind,
                               'sample_reason': self.sample_reason,
                               'time_taken': self.time_taken})


class GroupFiles(db.Model):

    rand_id = db.Column(db.UnicodeText,
                        primary_key=True)
    user_id = db.Column(db.String(256),
                        db.ForeignKey(User.id),
                        primary_key=True)
    group_id = db.Column(db.String(256),
                         primary_key=True,
                         index=True)
    explan_id = db.Column(db.String(256))


    dt_exp_node = db.Column(db.Integer(), default=-1)

    # ID of file included in group
    file_id = db.Column(db.String(256))

    out_group_sample = db.Column(db.Boolean, default=False)


    def __repr__(self):

        return pprint.pformat({'user_id': self.user_id,
                               'group_id': self.group_id,
                               'file_id': self.file_id,
                               'out_group_sample': self.out_group_sample})


class HeaderRow(db.Model):

    rand_id = db.Column(db.UnicodeText,
                        primary_key=True,
                        unique=True)
    user_id = db.Column(db.UnicodeText,
                        db.ForeignKey(User.id),
                        primary_key=True)

    header_title = db.Column(db.UnicodeText)
    header_explain = db.Column(db.UnicodeText)

    def __repr__(self):

        return pprint.pformat({'header_title': self.header_title,
                               'header_explain': self.header_explain})
