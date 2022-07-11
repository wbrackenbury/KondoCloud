import uuid
import hashlib
import random
import time
import datetime
import os
import pickle
import math
import re

import google.oauth2.credentials
import google_auth_oauthlib.flow

from functools import wraps
from flask import render_template, session
from oauth2client.client import GoogleCredentials, AccessTokenCredentialsError
from oauth2client.client import AccessTokenCredentials
from googleapiclient.discovery import build
from datasketch import MinHash
from itertools import tee, zip_longest
from sklearn.metrics.pairwise import cosine_similarity

from felfinder import app
from felfinder.models import SharedUsers, Schemas, User, File, Folder
from felfinder.config import (BASIC_SCOPES, MINIMAL_SCOPES, REPLAY_SCOPES,
                              TEST_SERVER, ACTIVITY_SCOPES,
                              GOOGLE_OAUTH_CLIENT_ID,
                              GOOGLE_OAUTH_CLIENT_SECRET)

def templated(template=None):
    def decorator(func):
        @wraps(func)
        def decorated_function(*args, **kwargs):
            template_name = template
            if template_name is None:
                template_name = request.endpoint \
                        .replace('.', '/') + '.html'
            ctx = func(*args, **kwargs)
            if ctx is None:
                ctx = {}
            elif not isinstance(ctx, dict):
                return ctx
            return render_template(template_name, **ctx)
        return decorated_function
    return decorator

def generate_csrf_token():
    if 'csrf_token' not in session:
        session['csrf_token'] = random.getrandbits(128)
    return session['csrf_token']

def trunc_str(s, length):

    if not s:
        return s

    if len(s) >= length:
        return s[:length] + '...'
    else:
        return s

def get_hash(dbid, uid):
    """
    TODO: This might be totally broken
    """

    salt = app.config['SALT']
    return hashlib.sha256(str(dbid+salt).encode('utf-8')).hexdigest()

def get_scopes(scope_level):

    scopes = BASIC_SCOPES
    if scope_level == "replay":
        scopes = REPLAY_SCOPES
    elif scope_level == "minimal":
        scopes = MINIMAL_SCOPES
    elif scope_level == 'activity':
        scopes = ACTIVITY_SCOPES

    return scopes

def scope_flow(redirect_uri, scope_level, state):

    if not TEST_SERVER:
        client_secret_file = 'cloudassist_client_secret.json'
    else:
        client_secret_file = 'quickstart_client_secret.json'

    state_kwargs = {"state": state} if state else {}

    print("Scope flow scopes: {}".format(get_scopes(scope_level)))

    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        client_secret_file, get_scopes(scope_level), **state_kwargs)
    flow.redirect_uri = redirect_uri

    return flow


def create_service_object(uid, scope_level, api='drive', v='v3'):
    user = User.query.filter_by(id = uid).one()
    creds = google.oauth2.credentials.Credentials(
        user.access_token,
        refresh_token = user.refresh_token,
        token_uri = user.token_uri,
        client_id = GOOGLE_OAUTH_CLIENT_ID,
        client_secret = GOOGLE_OAUTH_CLIENT_SECRET,
        scopes = get_scopes(scope_level))

    return build(api, v, credentials=creds, cache_discovery=False)

def refresh_google_token(proc_state):

    """
    Google OAuth tokens expire after an hour. We need to refresh it when our
    requests begin failing
    """

    #TODO: This likely still doesn't work.

    cur_user = User.query.filter_by(id=proc_state.uid).one()

    try:
        drive_obj = create_service_object(proc_state.access_token)
        this_user = drive_obj.about().get(fields="user").execute()

    except Exception as e:
        refresh_token = proc_state.refresh_token
        cred = GoogleCredentials(proc_state.access_token, GOOGLE_OAUTH_CLIENT_ID,
        GOOGLE_OAUTH_CLIENT_SECRET,refresh_token,time.now(),
        "https://accounts.google.com/o/oauth2/token",'my-user-agent/1.0')

        http = cred.authorize(httplib2.Http())
        cred.refresh(http)
        new_token = cred.get_access_token()[0]
        new_expires = time.time()+cred.get_access_token()[1]

        #update the new token in the database
        cur_user.access_token= new_token
        db.session.commit()

    return create_service_object(proc_state.access_token)


# https://stackoverflow.com/questions/1958219/convert-sqlalchemy-row-object-to-python-dict
def row_conv(sql_row):

    """
    We're supposed to return dictionary objects, but we're pulling items out of a database
    in an ORM. This function converts a SQLAlchemy object to the appropriate dictionary

    Arguments:
      sql_row (SQLAlchemy Row object): a row from a SQLAlchemy database table to convert

    Response:
      (Dict): dictionary object corresponding to row attributes

    """

    dictret = dict(sql_row.__dict__)
    dictret.pop('_sa_instance_state', None)

    return dictret


def inner_file_ret(f_obj, attrs_to_del, attrs_to_conv):

    """
    The file object we pull out of the database has parameters that aren't expected
    in the JSON we're to return per the client-server API. This converts the file object
    to a dictionary and formats parameters as appropriate.

    Arguments:
      sql_row (SQLAlchemy Row object): a row from a SQLAlchemy database table to convert

    Response:
      (Dict): dictionary file object properly formatted for our JSON response to client

    """

    f_obj = row_conv(f_obj)

    for attr in attrs_to_del:
        del f_obj[attr]

    # Convert to proper response object names

    for attr, conv in attrs_to_conv.items():
        f_obj[attr] = f_obj[conv]
        del f_obj[conv]

    # Add needed items

    can_write = f_obj['is_owner']
    isowner = f_obj['is_owner']

    f_obj['read'] = 1
    f_obj['write'] = 1 if can_write else 0
    f_obj['locked'] = 0
    f_obj['isowner'] = isowner
    f_obj['options'] = {'disabled': []}

    return f_obj

def file_ret(f_obj):

    attrs_to_del = ['rand_id', 'user_id', 'id', 'path',
                    'parent_id', 'file_extension',
                    'goog_mime_type', 'media_info',
                    'is_gdoc', 'error']

    attrs_to_conv = {'ts': 'last_modified',
                     'hash': 'path_hash',
                     'phash': 'parent_hash',
                     'mime': 'elf_mime_type'}


    return inner_file_ret(f_obj, attrs_to_del, attrs_to_conv)

def partial_file_ret(f_obj):

    """
    See explanation for partial_folder_ret
    """

    attrs_to_del = ['rand_id', 'user_id', 'path',
                    'parent_id', 'file_extension',
                    'goog_mime_type', 'media_info',
                    'is_gdoc', 'error']

    attrs_to_conv = {'ts': 'last_modified',
                     'hash': 'path_hash',
                     'phash': 'parent_hash',
                     'mime': 'elf_mime_type'}


    return inner_file_ret(f_obj, attrs_to_del, attrs_to_conv)


def inner_folder_ret(f_obj, attrs_to_del, attrs_to_conv):

    """
    Same as function above, but for folders
    """


    f_obj = row_conv(f_obj)

    if(f_obj['name'] == 'root'):
        # If the object is the root, it doesn't need a parent hash
        attrs_to_del.append('parent_hash')
        del attrs_to_conv['phash']

    for attr in attrs_to_del:
        del f_obj[attr]

    # Convert to proper response object names

    for attr, conv in attrs_to_conv.items():
        f_obj[attr] = f_obj[conv]
        del f_obj[conv]

    # Add needed items

    can_write = f_obj['is_owner']
    isowner = f_obj['is_owner']

    f_obj['read'] = 1
    f_obj['write'] = 1
    f_obj['locked'] = 0
    f_obj['isowner'] = isowner
    f_obj['options'] = {}
    f_obj['volumeid'] = 'l0_'
    f_obj['mime'] = 'directory'

    return f_obj

def remove_part_slack(partition, avail):

    """
    There's the possibility that the mismatch of
    desirability and availability is skewed. This means
    we want to shift around the weight such that
    we properly ask for items that are available
    """

    #print("Before slack: {}".format(l))

    slack = 0
    for i in range(len(partition)):

        is_last = i == len(partition) - 1
        avail_diff = partition[i] - avail[i]
        not_enough = avail_diff > 0
        can_take_more = avail_diff < 0

        if not_enough and not is_last:
            partition[i] = avail[i]
            slack += avail_diff
        elif (can_take_more or is_last) and slack > 0:
            can_accept = min(avail_diff, slack)
            if is_last: # Dump the remainder of slack in here if we can't satisfy
                can_accept = slack
            partition[i] += can_accept
            slack -= can_accept

    return partition

def even_partition(total, parts, avail = []):

    """
    Takes a total sum, and the number
    of parts to split into, then try
    and match as closely as possible
    to a perfectly even split
    """

    l = [math.floor(total / parts) for p in range(parts)]
    remainder = total % parts
    for i in range(remainder):
        l[i] += 1

    if avail != []:
        l = remove_part_slack(l, avail)

    assert sum(l) == total
    return l

def folder_ret(f_obj):

    attrs_to_del = ['rand_id', 'user_id', 'id', 'path',
                    'parent_id', 'view_link', 'error']

    attrs_to_conv = {'ts': 'last_modified',
                     'hash': 'path_hash',
                     'phash': 'parent_hash'}


    return inner_folder_ret(f_obj, attrs_to_del, attrs_to_conv)

def partial_folder_ret(f_obj):

    """
    Very similar to the above, but instead we keep ID in the
    object and delete later. This is because in recursive
    copies, the actual database object flushes, but we need
    to retain the ID's to generate recommendations. We
    manually delete them later.
    """

    attrs_to_del = ['rand_id', 'user_id', 'path',
                    'parent_id', 'view_link', 'error']

    attrs_to_conv = {'ts': 'last_modified',
                     'hash': 'path_hash',
                     'phash': 'parent_hash'}


    return inner_folder_ret(f_obj, attrs_to_del, attrs_to_conv)

def get_rand_id():
    return str(uuid.uuid4())

def rand_prol_id():
    ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    NUMS = "0123456789"

    ALPHANUM = ALPHA + ALPHA.lower() + NUMS
    LENGTH = 32

    return "".join([random.choice(ALPHANUM) for _ in range(LENGTH)])

def get_hash(item):
    return hashlib.sha256(str(item).encode('utf-8')).hexdigest()[:32]

def jaccard(A, B):

    sA = set(A)
    sB = set(B)

    inter = sA.intersection(sB)
    un = sA.union(sB)

    if len(un) == 0:
        return 0

    return abs(len(inter)) / abs(len(un))

def last_mod_func(mod1, mod2, isGoogle):

    # ORIG = datetime.datetime(1970, 1, 1)

    # d_mod1 = mod1
    # d_mod2 = mod2

    # if isGoogle:

    #     d_mod1 = mod1[:-5]
    #     d_mod2 = mod2[:-5]
    # else:
    #     d_mod1 = mod1[:-1]
    #     d_mod2 = mod2[:-1]


    # d1 = datetime.datetime.strptime(d_mod1, "%Y-%m-%dT%H:%M:%S")
    # d2 = datetime.datetime.strptime(d_mod2, "%Y-%m-%dT%H:%M:%S")

    diff = mod1 - mod2
    diff = abs(diff)

    return diff

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def pairwise(t):
    a, b = tee(t)
    next(b, None)
    return zip(a, b)

def bigrams(s):
    for a, b in pairwise(s):
        yield a + b

# def bigram_sim(s1, s2):

#     if s1 is None or s2 is None:
#         return 0.0

#     NUM_PERM = 128
#     ENCODING = 'utf-8'

#     m1 = MinHash(num_perm=NUM_PERM)
#     m2 = MinHash(num_perm=NUM_PERM)

#     bigrams_s1 = [bi for bi in bigrams(s1)]
#     bigrams_s2 = [bi for bi in bigrams(s2)]

#     for bi in bigrams_s1:
#         m1.update(bi.encode(ENCODING))
#     for bi in bigrams_s2:
#         m2.update(bi.encode(ENCODING))

#     return m1.jaccard(m2)



def token_simils(fnamea, fnameb):

    SPLIT_TOKENS = "\.|_|-|~|\+| "

    """
    We want to evaluate similarity between filenames
    that have similar patterns (e.g., IMG001, IMG002)
    without getting caught up in the minutiae
    """

    a_tok = re.split(SPLIT_TOKENS, fnamea)
    b_tok = re.split(SPLIT_TOKENS, fnameb)

    a_tok = [c.lower() for c in a_tok]
    b_tok = [c.lower() for c in b_tok]

    j = jaccard(a_tok, b_tok)

    return j


def bigram_sim(s1, s2):

    if s1 is None or s2 is None:
        return 0.0

    ENCODING = 'utf-8'


    bigrams_s1 = [bi for bi in bigrams(s1)]
    bigrams_s2 = [bi for bi in bigrams(s2)]

    return jaccard(bigrams_s1, bigrams_s2)

def schema_jaccard(proc_state, fwrap, prev_fwrap):

    if fwrap.is_spread and prev_fwrap.is_spread:

        cur_file_schema_items = Schemas.query.filter_by(user_id=proc_state.uid).\
                             filter_by(file_id = fwrap.id).all()
        prior_file_schema_items = Schemas.query.filter_by(user_id=proc_state.uid).\
                             filter_by(file_id = prev_fwrap.id).all()

        schema_cur = [item.feat for item in cur_file_schema_items]
        schema_prior = [item.feat for item in prior_file_schema_items]

        schema_sim = jaccard(schema_cur, schema_prior)

        return schema_sim

    else:
        return None

def google_perm_jaccard(uid, idA, idB):

    perm_objsA = SharedUsers.query.filter_by(user_id=uid, id=idA).all()
    perm_objsB = SharedUsers.query.filter_by(user_id=uid, id=idB).all()

    permsA = [obj.shared_user for obj in perm_objsA]
    permsB = [obj.shared_user for obj in perm_objsB]

    if (len(permsA) <= 1 or len(permsB) <= 1):
        return 0
    else:
        return jaccard(permsA, permsB)


def get_file_type(file_path):

    file_path = file_path.lower()
    image = set(['jpg', 'jpeg', 'png', 'tiff', 'tif', 'gif', 'bmp'])
    video = set(['3gp', '3g2', 'avi', 'f4v', 'flv', 'm4v', 'asf', 'wmv', 'mpeg', 'mp4', 'qt'])
    document = set(['txt', 'doc', 'docx', 'rtf', 'dotx', 'dot', 'odt', 'pages', 'tex', 'pdf', 'ps', 'eps', 'prn'])
    web = set(['html', 'xhtml', 'php', 'js', 'xml', 'war', 'ear' 'dhtml', 'mhtml'])
    spreathseet =  set(['xls', 'xlsx', 'csv', 'tsv', 'xltx', 'xlt', 'ods', 'xlsb', 'xlsm', 'xltm'])
    presentation = set(['ppt', 'pptx', 'pot', 'potx', 'odp', 'ppsx', 'pps', 'pptm', 'potm', 'ppsm', 'key'])

    if "~" in file_path.split("/")[-1] or "$" in file_path.split("/")[-1]:
        return ("other", None)

    if "." in file_path and file_path[0] != '.':
        extension = file_path.split(".")

        if len(extension) > 14:
            return ("other", None)

        if extension[-1] in image:
            return("image", extension[-1])
        if extension[-1] in video:
            return("video", extension[-1])
        if extension[-1] in document:
            return("document", extension[-1])
        if extension[-1] in web:
            return("web", extension[-1])
        if extension[-1] in spreathseet:
            return("spreadsheet", extension[-1])
        if extension[-1] in presentation:
            return("presentation", extension[-1])
        return ("other", extension[-1])

    return ("other", None)

def copy_rename(fname):

    """
    Rename a file when copied into the same folder
    """

    if '.' in fname:
        fsplit = fname.split(".")
        ext = fsplit[-1]
        upto = '.'.join(fsplit[:-1])
        return upto + '_copy.' + ext
    else:
        return fname + '_copy'

def path_dist(p1, p2):

    p1_folds = p1.split("/")[:-1]
    p2_folds = p2.split("/")[:-1]

    for i, (fa, fb) in enumerate(zip_longest(p1_folds, p2_folds)):
        if fa != fb:
            p1_dist_to_end = max((len(p1_folds) - i), 0)
            p2_dist_to_end = max((len(p2_folds) - i), 0)
            return p1_dist_to_end + p2_dist_to_end

    return 0

def is_file_object(obj):
    return hasattr(obj, 'file_extension')

def elastic_term_query(field, term):

    """
    Helper for elastic search syntax
    """

    return {"term": {field: term}}

def elastic_wildcard_query(field, term):
    return {"wildcard": {field: {"value": "*" + term + "*"}}}

def elastic_range_query(field, age_range):

    curr = time.time()

    seconds_in_day = 60 * 60 * 24

    rng = {'Last 7 days': seconds_in_day * 7,
           'Last 30 days': seconds_in_day * 30,
           'Last 90 days': seconds_in_day * 90}


    return {"range": {field: {"gte": curr - rng[age_range]}}}

def get_fobj_from_uid_id(user_id, fobj_id):

    file_obj = File.query.filter_by(user_id = user_id, id = fobj_id).one_or_none()
    folder_obj = Folder.query.filter_by(user_id = user_id, id = fobj_id).one_or_none()

    return file_obj or folder_obj

def load_if_exists(path):
    if os.path.exists(path):
        with open(path, 'rb') as of:
            return pickle.load(of)
        return None

def simple_cosine_sim(a, b):
    return float(cosine_similarity(a.reshape(1, -1),
                                   b.reshape(1, -1))[0][0])

def remove_dups(l):
    n = []
    for item in l:
        if item not in n:
            n.append(item)
    return n


def med_sel(l, samps_per_ind = [3, 3, 3], qs = [0.25, 0.5, 0.75]):

    """
    Given a list ordered by size, we want to select some number of items
    around the indices that are at the percentiles indicated by qs
    """

    if len(l) <= sum(samps_per_ind):
        return list(range(len(l)))

    max_seen = -1
    rel_inds = []
    sel_att = [int(q * len(l)) - 1 for q in qs]

    print(sel_att)

    # Try and select the number of samps right before the ind indicated
    # by q, push them to the items just beyond as needed
    for i, num_samp in zip(sel_att, samps_per_ind):

        ind_to_find = max(i, max_seen + 1) # Ind to search around
        range_around = ind_to_find - num_samp + 1 # Ind of lowest item to get
        num_to_move = max(0, max_seen + 1 - range_around) # How many to look beyond ind_to_find

        bot_range = range_around + num_to_move
        top_range = ind_to_find + num_to_move + 1

        rel_inds.extend(list(range(bot_range, top_range)))
        max_seen = top_range - 1

        print(max_seen)
        print(rel_inds)
        print()

    return rel_inds
