import numpy as np

from collections import Counter
from itertools import chain, permutations, tee, accumulate

MAX_SHARES = 1000

MAX_DEPTH = 20
MAX_FAN = 20
MAX_DIST = 2 * MAX_DEPTH
ROOT_CODE = (0, 0, 0)

def shares_from_db_obj(base_ids, shared_users):

    """
    Takes the SharedUsers db objects and puts
    them in the form needed for calc_share_reps
    """

    shares = {}
    for su_obj in shared_users:
        l = shares.get(su_obj.id, [])
        l.append(su_obj.shared_user)
        shares[su_obj.id] = l

    full_shares = [shares.get(k, []) for k in base_ids]

    return full_shares

def calc_share_reps(uids):

    """
    Calculate one-hot vector representation of list of participant
    IDs. Note that uids must be a list, or we have to tee up an iterable again
    """

    curr = 0
    uid_map = {}

    uid_count = Counter(chain.from_iterable(uids))

    for u, _ in uid_count.most_common():
        if u not in uid_map:
            uid_map[u] = str(curr)
            if curr < MAX_SHARES - 1: # Drop all users > 1000 into last bucket
                curr += 1


    uid_vec_reps = np.zeros((len(uids), MAX_SHARES), dtype='float32')
    for i, uid_set in enumerate(uids):
        uid_inds = [int(uid_map[u]) for u in uid_set]
        uid_vec_reps[i, uid_inds] = 1

    return uid_vec_reps


def encode_paths(paths):

    """
    Take in a set of file paths and return the set
    of triplet codes corresponding to their positions

    NB for the moment, we're assuming that folder names
    are unique
    """

    path_codes = {'/': ROOT_CODE}
    groups_seed = set([ROOT_CODE])

    for p in paths:
        path_comps = [pc + '/' for pc in p.split("/")]

        par_code = ROOT_CODE
        for depth, pc in enumerate(accumulate(path_comps)):

            if depth >= MAX_DEPTH:
                raise ValueError("Exceeded maximum depth")

            if pc in path_codes:
                par_code = path_codes[pc]
                continue

            group = MAX_FAN * par_code[2] + par_code[1]
            pos = 0
            while (depth, pos, group) in groups_seed:
                if pos >= MAX_FAN:
                    break
                    #raise ValueError("Exceeded maximum fan")
                pos += 1

            code = (depth, pos, group)
            path_codes[pc] = code
            groups_seed.add(code)
            par_code = code

    return path_codes


def triplet_to_vec(triplet):

    """
    Take a triplet and encode it as a 128-bit binary vector
    """

    a, b, c = triplet
    bin_rep = "{0:05b}{1:05b}{2:0118b}".format(a,b,c)
    if len(bin_rep) > 128:
        print(a, b, c)

    return np.array(list(bin_rep), dtype='int32')


def up_one_level(depth, pos, group):
    return (depth - 1, group % MAX_FAN, group // MAX_FAN)


def tree_dist_codes(a, b):

    """
    Given two codes of the (depth, pos, group) variety,
    find the tree distance between the two
    """

    if a == b:
        return 0

    dual_raise = False
    left_branch_len = 0
    right_branch_len = 0

    while a[0] > b[0]:
        a = up_one_level(*a)
        left_branch_len += 1
    while b[0] > a[0]:
        b = up_one_level(*b)
        right_branch_len += 1

    while a != b:
        dual_raise = True
        a = up_one_level(*a)
        left_branch_len += 1
        b = up_one_level(*b)
        right_branch_len += 1

    # Because of how we write file paths here, /a and /b,
    # for example, are in the same folder, but we need
    # to reach their parent in the algorithm, which adds
    # 2 to the correct distance
    adjustment = (2 * int(dual_raise))

    return left_branch_len + right_branch_len - adjustment


def pairwise(t):
    a, b = tee(t)
    next(b, None)
    return zip(a, b)


def bigrams(s):
    for a, b in pairwise(s):
        yield a + b


def gen_token_universe():

    alpha = "abcdefghijklmnopqrstuvwxyz_-()"
    return [''.join(p) for p in permutations(alpha, 2)] + ['*']


def tokenize_fname(fname, toks):

    """
    Generate a canonical 1-hot encoding of the bigrams of a
    filename
    """

    bgs = set(bigrams(fname))

    vec = [1 if t in bgs else 0 for t in toks]
    if any([b not in toks for b in bgs]):
        vec[-1] = 1

    return np.array(vec, dtype='int32')
