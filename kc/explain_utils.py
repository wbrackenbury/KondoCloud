import time
import json
import re

from copy import deepcopy
from itertools import combinations

MAX_LEN = 1000
MAX_TOKS = 1000
SPLITTERS = "\r|\n|-|_|\+|\%|\^|\*|\.|\?|\!|\&|\@|\#|\$|'|\"| "

class ExplainCand:

    def __init__(self, inds, inter):
        self.inds = set(inds)
        self.inter = inter

    def sat(self):
        return len(self.inter)

    def num_inds(self):
        return len(self.inds)

    def get_set(self):
        return self.inter

    def get_inds(self):
        return self.inds

    def check_sat(self, new_set):
        return len(self.inter.intersection(new_set))

    def check_used(self, ind):
        return (ind in self.inds)

    def add_ind(self, ind, new_set):

        if ind in self.inds:
            raise ValueError("This explainer is already part of the set")

        self.inds.add(ind)
        self.inter = self.inter.intersection(new_set)

    def __repr__(self):
        return "Inds: {}\nSet: {}".format(self.inds, self.inter)

def get_ext_set(w):

    if '.' in w:
        ext_split = w.split(".")
        exts = [ext_split[-1]]
    else:
        exts = []

    return exts


def text_toks(s):
    return [w[:MAX_LEN] for w in re.split(SPLITTERS, s)[:MAX_TOKS]]

def shared_toks(s):
    return s[:MAX_TOKS]


def string_preproc(w, include_prefix = True):

    PREFIX_START = 2
    PREFIX_END = 7

    ext = ""
    if '.' in w:
        ext_split = w.split(".")
        ext = ext_split[-1]

    split_toks = [tok.lower()[:MAX_LEN] for tok in re.split(SPLITTERS, w)
                  if (tok != '') and (tok != ext)]
    if include_prefix:
        prefix_toks = [w[:i].lower() for i in range(PREFIX_START, PREFIX_END)]
    else:
        prefix_toks = []

    return set(split_toks + prefix_toks)

def get_starting(w):

    PREFIX_START = 2
    PREFIX_END = 1000

    prefix_toks = list(reversed([w[:i].lower() for i in range(PREFIX_START, PREFIX_END)]))

    return prefix_toks


def best_group_explain(tok_sets):

    """
    Algorithm to find best group, assuming size of
    explainer set wants to be greater than 1
    """

    cand_sets = []
    single_fallback = None

    for (i, set_a), (j, set_b) in combinations(enumerate(tok_sets), 2):
        cand_set = set_a.intersection(set_b)
        cand = ExplainCand(inds=[i, j], inter=cand_set)

        if cand.sat() > 1:
            cand_sets.append(cand)
        if cand.sat() > 0 and len(cand_sets) < 1:
            single_fallback = cand

    if len(cand_sets) < 1:
        if single_fallback is None:
            return []
        return single_fallback.get_set()

    while len(cand_sets) > 0:

        cand = cand_sets.pop()

        check_inds = range(max(cand.get_inds()) + 1, len(tok_sets))

        for i in check_inds:
            new_set = tok_sets[i]

            if cand.check_sat(new_set) > 1:
                new_cand = deepcopy(cand)
                new_cand.add_ind(i, new_set)
                cand_sets.append(new_cand)

    if cand is None:
        print(tok_sets)

    return cand.get_set()


def overlap_explain(l, strategy='total'):

    """
    Take sets of tokens and find if there are any sets
    of words that allow for maximum overlap
    """

    if not l:
        return []

    if strategy == 'total':

        main_set = l[0]

        for item in l[1:]:
            main_set = main_set.intersection(item)

    if strategy == 'large_group':
        main_set = l[0]

        for item in l[1:]:
            part_set = main_set.intersection(item)
            if len(part_set) > 0:
                main_set = part_set

    if strategy == 'best_group':
        main_set = best_group_explain(l)

    # Remove substrings
    # main_set = [m for m in main_set
    #             if not any([m in a for a in main_set.difference(set([m]))])]
    main_set = [m for m in main_set
                if not any([a.startswith(m) for a in main_set.difference(set([m]))])]

    return list(main_set)


def word_overlap_explain(wl, strategy='total'):

    tok_list = [string_preproc(w) for w in wl]
    return overlap_explain(tok_list, strategy=strategy)


def base_overlap_explain(tl, strategy='total'):
    return overlap_explain([set(t) for t in tl], strategy=strategy)

def path_explain(p):
    #Exclude the blank root, and the filename itself
    return p.split("/")[1:-1]

def attr_convert(a):

    return {'id': a.id,
            'date': a.last_modified,
            'size': a.size,
            'fname': a.fname,
            'parent_id': a.parent_id,
            'path': path_explain(a.path),
            'ext': get_ext_set(a.fname),
            'shared': shared_toks(a.susers),
            'starttoks': get_starting(a.fname),
            'filetoks': string_preproc(a.fname),
            'modfiletoks': string_preproc(a.fname, include_prefix = False),
            'texttoks': text_toks(a.text),
            'imgobjs': a.web_labels}


def group_dicts(df, examine, use_proba = True, thresh = 0.65):

    """
    Given the scores on pairs of files, we generate
    the relevant groups to explain
    """

    rel_groups = {}
    out_groups = {}

    for fid in examine:

        #cond = (df['del_true_cert'] >= thresh) if use_proba else (df['del_pred'] == True)

        sub_df = df[((df['file_id_A'] == fid) | (df['file_id_B'] == fid))]

        alt_ids = sub_df['file_id_A']
        alt_ids[alt_ids == fid] = sub_df['file_id_B']

        probs = sub_df['del_true_cert'].values.tolist() if use_proba \
            else sub_df['del_pred'].values.tolist()
        alt_ids = alt_ids.tolist()

        for add_id, prob in zip(alt_ids, probs):

            cond = prob > thresh if use_proba else prob == True

            if cond:
                rel_groups[fid] = rel_groups.get(fid, []) + [(add_id, prob)]
            else:
                out_groups[fid] = out_groups.get(fid, []) + [(add_id, prob)]

    return rel_groups, out_groups


def exp_to_json(e, exp_type):
    base = {}
    if exp_type == 'rules' or exp_type == 'rulesdt':
        base = {'exp': [{a: v} for a, v in e]}
    elif exp_type == 'dt':
        base = {'exp': [{'branch':
                         [{'attr': x[0], 'sign': x[1], 'val': x[2]}
                          for x in branch]}
                        for branch in e]}
    elif exp_type == 'no_exp':
        base = {'exp': e}
    return json.dumps(base)


def json_to_exp(e, exp_type):
    e = json.loads(e)
    exp = []

    if exp_type == 'rules':
        for d in e['exp']:
            a = list(d.keys())[0]
            b = list(d.values())[0]
            exp.append((a, b))
    elif exp_type == 'dt':
        #[[('bigram_simil', '<=', 0.148542158305645)], [('bigram_simil', '>', 0.148542158305645)]]

        for d in e['exp']:
            v = [(x['attr'], x['sign'], x['val']) for x in d['branch']]
            exp.append(v)

    return exp


def is_complex_exp(e, exp_type):

    if exp_type == 'rules' or exp_type == 'dt' or exp_type == 'rulesdt':
        if len(e) > 1:
            return True
        elif len(e) > 0 and 'rules' in exp_type:
            attr, exp_comp = e[0]

            if attr != 'date' and attr != 'size':
                if len(exp_comp) > 1 and exp_comp[1] != []:
                    return True
        elif len(e) > 0 and exp_type == 'dt':
            return len(e[0]) > 1

    return False
