import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pprint
import math
import pickle
import datetime
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm
from textwrap import wrap
from itertools import (combinations, combinations_with_replacement,
                       product, chain)
from pprint import pformat
from copy import deepcopy

from felfinder.pred_utils import (format_for_preds, clf_pred_outputs, get_clf,
                                  dt_preds_from_simil_rows,
                                  FULL_FEATURES, META_FEATURES,
                                  IMG_FEATURES, TEXT_ONLY, FEAT_MAP)

MODELS_BASE = "/home/will/Documents/Research_2021/kc-extend/kc-scaling/src/explain/models/"
#FULL_ATTRS = ["date", "size", "ext", "starttoks", "modfiletoks",
#              "filetoks", "texttoks", "imgobjs"]
FULL_ATTRS = ["date", "size", "ext", "starttoks", "modfiletoks",
              "shared", "texttoks", "path", "imgobjs"]

RANGE_ATTRS = ["date", "size"]

WEIGHTS = [0.25, 0.75]
LINE_LEN = 40

FULL_FEATURES = {"tfidf_sim": "Text topic similarity",
                 "word_vec_sim": "Text contents similarity",
                 "color_sim": "Image color similarity",
                 "obj_sim": "Image contents similarity",
                 "bin_simil": "File contents similarity",
                 "perm_simil": "Similarity of users shared-with",
                 "tree_dist": "Similarity in folder structure",
                 "size_dist": "File size similarity",
                 "bigram_simil": "Filename similarity",
                 "last_mod_simil": "Last modified date similarity"}


# ATTR_MAP = {"date": RangeItems,
#             "size": RangeItems,
#             "filetoks": SetItems,
#             "texttoks": SetItems,
#             "imgobjs": SetItems}


class SetCollectionFactory:


    def __init__(self, set_elements):

        self.lookup = {v: i for i, v in enumerate(set_elements)}

    def new_set(self, els = []):
        return SetCollection(self.lookup, els)


class SetCollection:


    def __init__(self, lookup = {}, els = [], universe = None):

        self.lookup = lookup
        num_els = (len(lookup) // 32) + 1
        self.s = [0 for _ in range(num_els)]
        self.curr_len = 0
        self._multi_set(els)

    def new_set(self, els):

        ns = deepcopy(self)
        ns.s = [0 for _ in range(len(self.s))]
        ns.curr_len = 0
        ns._multi_set(els)

        return ns

    def _create_lookup(self, universe):
        return {v: i for i, v in enumerate(set_elements)}

    def add(self, el):
        if el not in self:
            self.curr_len += 1
        self._set_el(el)


    def remove(self, el):
        if el in self:
            self.curr_len -= 1

        self._unset_el(el)

    def intersection(self, other):

        new_set = deepcopy(self)

        new_len = 0
        for i in range(len(self.s)):
            new_set.s[i] &= other.s[i]
            new_len += new_set._count_pos_bits(new_set.s[i])

        new_set.curr_len = new_len

        return new_set

    def __contains__(self, el):

        li, si = self._sub_ind(self.lookup[el])
        mask = 1 << si
        return (self.s[li] & mask) > 0

    def __len__(self):
        return self.curr_len

    def _count_pos_bits(self, l):
        return bin(l).count("1")

    def _multi_set(self, els):

        ind_list = [self._sub_ind(self.lookup[e]) for e in els]
        ind_list = sorted(ind_list)

        curr_ind = 0
        for li, si in ind_list:
            mask = 1 << si
            if not ((mask & self.s[li]) > 0):
                self.curr_len += 1

            self.s[li] |= mask

    def _set_el(self, el):
        full_ind = self.lookup[el]
        li, si = self._sub_ind(full_ind)

        mask = 1 << si
        self.s[li] |= mask

    def _unset_el(self, el):
        full_ind = self.lookup[el]
        li, si = self._sub_ind(full_ind)

        mask = 0 << si
        self.s[li] &= mask

    def _sub_ind(self, full_ind):

        list_ind = full_ind // 32
        sub_list_ind = full_ind % 32

        return list_ind, sub_list_ind


class SetExplain:

    """
    An explanation for a set-based explainer
    """

    def __init__(self, attr, dec_set, out_set, num_uni_sets = 1):

        self.attr = attr
        self.num_uni_sets = num_uni_sets

        self.toks = [[] for _ in range(num_uni_sets)]
        # self.dec_sets = [deepcopy(dec_set) for _ in range(num_uni_sets)]
        # self.out_sets = [deepcopy(out_set) for _ in range(num_uni_sets)]
        self.dec_sets = [set() for _ in range(num_uni_sets)]
        self.out_sets = [set() for _ in range(num_uni_sets)]

        self.set_init = [False for _ in range(num_uni_sets)]

    def __len__(self):
        return len(self.toks)

    def __getitem__(self, i):
        return (self.toks[i], self.dec_sets[i], self.out_sets[i])

    def test_tok_in_set(self, base_change, out_change, i):

        if self.set_init[i]:
            dsets = base_change.intersection(self.dec_sets[i])
            osets = out_change.intersection(self.out_sets[i])
        else:
            dsets = base_change
            osets = out_change

        for j, (d, o, init) in enumerate(zip(self.dec_sets, self.out_sets, self.set_init)):

            if j == i or not init:
                continue

            dsets = dsets.union(d)
            osets = osets.union(o)

        return dsets, osets

        # dec_sets = deepcopy(self.dec_sets)
        # out_sets = deepcopy(self.out_sets)

        # dec_sets[i] = base_change.intersection(dec_sets[i])
        # out_sets[i] = out_change.intersection(out_sets[i])

        # return union_all(dec_sets), union_all(out_sets)

    def sets(self):

        dec = union_all([d for d, t in zip(self.dec_sets, self.set_init) if t])
        out = union_all([d for d, t in zip(self.out_sets, self.set_init) if t])

        return dec, out

    def add_tok(self, tok, i):
        self.toks[i] += [tok]
        self.set_init[i] = True

    def update_sets(self, base, out, i):
        self.dec_sets[i] = base
        self.out_sets[i] = out

    def explain(self):
        return self.toks


class ScoreManager:

    """
    Manage an array of scores for set explanations,
    in which we track the scores for whether we add
    a token to one set or another
    """

    def __init__(self, num_uni_sets, num_toks):
        self.x_len = num_uni_sets
        self.y_len = num_toks
        self.scores = 100 * np.ones((num_uni_sets, num_toks))

        self.order = [np.arange(num_toks) for _ in range(self.x_len)]

    def set_val(self, i, j, value):
        self.scores[i, j] = value

    def set_order(self):
        self.order = [np.argsort(-self.scores[i]) for i in range(self.x_len)]

    def get_order(self, i):
        return self.order[i]

    def has_pos_score(self):
        return any([self.scores[i, self.order[i]][0] > 0 for i in range(self.x_len)])

    def best_ind(self):
        return np.argmax([self.scores[i, self.order[i]][0] > 0 for i in range(self.x_len)])


class RangeItems:

    """
    Identify whether values are in a range or not
    """

    def __init__(self, attr, dec_ids, out_ids, dec_vals, out_vals, num_uni_items = None):

        #num_uni_items is a garbage parameter for compatibility

        self.attr = attr

        order = np.argsort(dec_vals)
        self.dec_ids = np.array(dec_ids)[order]
        self.dec_vals = np.array(dec_vals)[order]

        order = np.argsort(out_vals)
        self.out_ids = np.array(out_ids)[order]
        self.out_vals = np.array(out_vals)[order]


    def exclude_copy(self, ids):
        """
        Return a deep copy of this object that excludes
        the relevant ids
        """

        new_obj = deepcopy(self)

        rel_ids = np.isin(self.ids, ids)
        new_obj.ids = np.delete(new_obj.ids, rel_ids)
        new_obj.vals = np.delete(new_obj.vals, rel_ids)

        return new_obj

    def _binsearch(self, val, low, high, low_end = False, dec = True):

        """
        We want it to behave 2 different ways depending on whether
        it's searching for the low end or the top end of the range:

        if low_end == True, should find lowest index <= val, or
        min index, and include_val should note the value's in the range

        if low_end == False, should find highest index >= val, or
        top index, and include_val should note the value's in the range
        """

        val_arr = self.dec_vals if dec else self.out_vals

        curr_ind = low + high // 2
        curr_val = val_arr[curr_ind]

        while low < high:

            #print("Low: {}, High: {}".format(low, high))
            #print("Curr_Ind: {}, Curr_val: {} ({})".format(curr_ind, curr_val, val))

            if curr_val > val or (low_end and curr_val == val):
                high = curr_ind - int(low_end)
            elif curr_val < val or (not low_end and curr_val == val):
                low = curr_ind + int(not low_end)

            curr_ind = (low + high) // 2 if not low_end else math.ceil((low + high) / 2)
            curr_val = val_arr[curr_ind]

            #print("Low: {}, High: {}".format(low, high))
            #print("Curr_Ind: {}, Curr_val: {} ({})".format(curr_ind, curr_val, val))
            #print()

        include_val = (curr_val == val) or \
            (low_end and val < curr_val) or \
            (not low_end and val > curr_val)

        #print("Curr_Ind: {}, Curr_val: {} ({})".format(curr_ind, curr_val, val))
        #print("Include val: {}".format(include_val))

        return curr_ind, include_val


    def range_q(self, l, h):

        """
        Return % of total items within the given range
        """

        low_ind, low_eq = self._binsearch(l, 0, len(self.dec_vals) - 1,
                                          low_end = True)
        high_ind, high_eq = self._binsearch(h, 0, len(self.dec_vals) - 1,
                                            low_end = False)

        dec_low = low_ind if low_eq else low_ind + 1
        dec_high = high_ind + 1 if high_eq else high_ind
        dec_inds = set(self.dec_ids[dec_low:dec_high].tolist())

        #ret_val = high_ind - low_ind - 1 + low_eq + high_eq

        low_ind, low_eq = self._binsearch(l, 0, len(self.out_vals) - 1,
                                          low_end = True, dec = False)

        #print("Final low: {}".format(low_ind))

        high_ind, high_eq = self._binsearch(h, 0, len(self.out_vals) - 1,
                                            low_end = False, dec = False)

        out_low = low_ind if low_eq else low_ind + 1
        out_high = high_ind + 1 if high_eq else high_ind
        out_inds = set(self.out_ids[out_low:out_high].tolist())


        return dec_inds, out_inds

    # def perc_range(self, l, h):
    #     return self.range_q(l, h) / len(self.vals)



    def __repr__(self):
        dec = pformat(self.dec_ids) + '\n' + pformat(self.dec_vals)
        out = pformat(self.out_ids) + '\n' + pformat(self.out_vals)
        return self.attr + '\n\n' + dec + '\n\n' + out

class SetItems:

    """
    Takes in ids and sets of values, and can perform
    membership queries
    """

    def __init__(self, attr, dec_ids, out_ids, dec_vals, out_vals, num_uni_items = 1):
        self.attr = attr

        self.dec_ids = dec_ids
        self.dec_index = self._construct_inv_index(dec_ids, dec_vals)

        self.toks = set(self.dec_index.keys())

        self.out_ids = out_ids
        self.out_index = self._construct_inv_index(out_ids, out_vals)

        self.num_uni_sets = num_uni_items

    # def exclude_copy(self, ids):
    #     """
    #     Return a deep copy of this object that excludes
    #     the relevant ids
    #     """

    #     new_obj = deepcopy(self)

    #     for i in ids:
    #         self.ids.remove(i)

    #     for key in self.index:
    #         for i in ids:
    #             self.index[key].remove(i)

    #     return new_obj


    def _construct_inv_index(self, ids, vals, toks = None):

        """
        Construct the inverse index showing ids for values.
        Since we only have to construct the index over
        the token universe for the decision set, we use toks
        to avoid building the index over tokens that are not
        in the decision set
        """

        index = {}

        universe = set()
        for i, val_set in zip(ids, vals):
            universe.add(i)

        #scf = SetCollectionFactory(universe)

        for i, val_set in zip(ids, vals):
            for v in val_set:

                if toks is not None and v not in toks:
                    continue

                curr_inds = index.get(v, set())
                curr_inds.add(i)
                index[v] = curr_inds

        return index


    def _find_exp_vals(self, index, exp_set):

        rev_sets = []
        for item in exp_set:
            rev_sets.append(index.get(item, set()))

        base_set = intersect_all(rev_sets)

        return base_set


    def explain_count(self, exp_set):

        """
        Return count of ids that match the explainer set
        """

        decs = [self._find_exp_vals(self.dec_index, e_set) for e_set in exp_set]
        outs = [self._find_exp_vals(self.out_index, e_set) for e_set in exp_set]

        return union_all(decs), union_all(outs)


    def _calc_scores(self, curr_toks, exp_obj,
                     dec_scores, out_scores, score_weight):

        scores = ScoreManager(self.num_uni_sets, len(curr_toks))

        dec_sub, out_sub = exp_obj.sets()

        curr_score, _, _ = full_eval(dec_sub, len(self.dec_ids),
                                     out_sub, len(self.out_ids),
                                     dec_scores, out_scores,
                                     WEIGHTS, score_weight)

        for i, t in enumerate(curr_toks):
            for j in range(self.num_uni_sets):

                base, out = exp_obj.test_tok_in_set(self.dec_index[t],
                                                    self.out_index.get(t, set()), j)

                changed_score, _, _ = full_eval(base, len(self.dec_ids),
                                          out, len(self.out_ids),
                                          dec_scores, out_scores,
                                          WEIGHTS, score_weight)


                # print(self.out_index.get(t, set()))
                # print(dec_sub, out_sub)
                # print(base, out)

                # print(t, curr_score, changed_score)
                # print()
                # s = full_pot_eval(dec_sub, base, self.dec_ids,
                #                   out_sub, out, self.out_ids,
                #                   dec_scores, out_scores,
                #                   WEIGHTS, score_weight)

                scores.set_val(j, i, changed_score - curr_score)

        return scores

    def set_greedy(self, base_set, out_set,
                   dec_scores, out_scores, score_weight, weights = WEIGHTS):

        """
        Greedily add items to the explanation set using
        the sub-setted inverse indices
        """

        exp_obj = SetExplain(self.attr, base_set, out_set, self.num_uni_sets)

        if len(self.toks) < 1:
            return (set(), set()), []

        curr_toks = np.array(list(self.toks))

        scores = self._calc_scores(curr_toks, exp_obj,
                                   dec_scores, out_scores, score_weight)

        scores.set_order()

        no_exp = True # Must run loop at least once

        while no_exp or scores.has_pos_score():

            ind = scores.best_ind()
            score_order = scores.get_order(ind)

            #print(scores.scores)

            curr_toks = np.array(curr_toks)[score_order]

            #print(curr_toks[:5])

            exp_obj.add_tok(curr_toks[0], ind)

            new_base, new_out = self.explain_count([exp_obj.explain()[ind]])
            exp_obj.update_sets(new_base, new_out, ind)

            # print(ind)
            # print(exp_obj.explain())
            # print(len(new_base), len(new_out))
            # print([len(s) for s in exp_obj.dec_sets], [len(s) for s in exp_obj.out_sets])
            # print()
            # print()
            #exit(0)

            scores = self._calc_scores(curr_toks, exp_obj,
                                       dec_scores, out_scores, score_weight)
            scores.set_order()

            #print(scores.scores)


            no_exp = False

        (base, out), exp = self.explain_count(exp_obj.explain()), exp_obj.explain()

        return (base, out), exp

    # def explain_perc(self, exp_set):
    #     return len(self.explain_count(exp_set)) / len(self.ids)

    def __repr__(self):
        dec = pformat(self.dec_ids) + '\n' + pformat(self.dec_index)
        out = pformat(self.out_ids) + '\n' + pformat(self.out_index)
        return self.attr + '\n\n' + dec + '\n\n' + out


class Explainer:

    """
    Takes in a universe of items that might require
    explanation, then when given the items that are
    part of the decision set, can synthesize an optimal
    explanation for the set according to a set of parameters
    for the loss function
    """

    def __init__(self, full_set,
                 group_scores, out_scores,
                 attrs_to_explain = FULL_ATTRS,
                 max_depth = 2, num_uni_items = 1):

        """
        Assume full_set = {id: {attrs}}
        """

        self.full_set = full_set
        self.dec_scores = group_scores
        self.out_scores = out_scores
        self.attrs = sorted(attrs_to_explain, key = lambda x: x not in RANGE_ATTRS)
        self.max_depth = max_depth
        self.num_uni_items = num_uni_items

        self.curr_dec_scores = {}
        self.curr_out_scores = {}

    def _enum_explain(self, base_id, dec_ids, score_weight):

        """
        Given a set of ids as the original decision set,
        find the optimal explanation for the grouping
        """

        ATT_MAP = {a: (RangeItems if a in RANGE_ATTRS else SetItems)
                   for a in self.attrs}

        dec_set = [d for d in self.full_set if d['id'] in dec_ids]
        out_set = [d for d in self.full_set if
                   d['id'] not in dec_ids and d['id'] != base_id]
        out_ids = [d['id'] for d in out_set]

        evals = {att: ATT_MAP[att](att, dec_ids, out_ids,
                                   dec_vals = [d[att] for d in dec_set],
                                   out_vals = [d[att] for d in out_set],
                                   num_uni_items = self.num_uni_items)
                 for att in self.attrs}

        exps = []

        for i in range(1, self.max_depth + 1):
            for att_list in combinations(self.attrs, i):

                cand_cats = []


                for a in att_list:

                    att_cands = [(a, cand) for cand in prod_cands(a, dec_set)]
                    cand_cats.append(att_cands)

                exps.extend(list(product(*cand_cats)))
                # Prune candidate list?

        opt_cand, score, other_metrs = best_cand(dec_ids, out_ids,
                                                 exps, evals,
                                                 self.curr_dec_scores,
                                                 self.curr_out_scores,
                                                 score_weight)

        return opt_cand, score, other_metrs


    def _part_enum_explain(self, base_id, dec_ids, score_weight):

        """
        We only enumerate the full set of range predicates,
        but we instead compute the optimal set-based explainer
        for each restricted subset
        """

        ATT_MAP = {a: (RangeItems if a in RANGE_ATTRS else SetItems)
                   for a in self.attrs}

        dec_set = [d for d in self.full_set if d['id'] in dec_ids]
        out_set = [d for d in self.full_set if
                   d['id'] not in dec_ids and d['id'] != base_id]
        out_ids = [d['id'] for d in out_set]

        evals = {att: ATT_MAP[att](att, dec_ids, out_ids,
                                   dec_vals = [d[att] for d in dec_set],
                                   out_vals = [d[att] for d in out_set],
                                   num_uni_items = self.num_uni_items)
                 for att in self.attrs}

        exps = []

        for i in range(1, self.max_depth + 1):
            for att_list in combinations(self.attrs, i):

                cand_cats = []

                for a in att_list:

                    if a in RANGE_ATTRS:
                        att_cands = [(a, cand) for cand in prod_cands(a, dec_set)]
                        cand_cats.append(att_cands)
                    else:
                        cand_cats.append([(a, None)])

                exps.extend(list(product(*cand_cats)))
                # Prune candidate list?

        opt_cand, score, other_metrs = part_cands(dec_ids, out_ids,
                                                  exps, evals,
                                                  self.curr_dec_scores,
                                                  self.curr_out_scores,
                                                  score_weight)

        return opt_cand, score, other_metrs



    def _full_approx_explain(self, base_id, dec_ids, score_weight):

        """
        We compute the optimal set for each attribute, then chain them together
        in the style of the set_greedy algorithm
        """

        ATT_MAP = {a: (RangeItems if a in RANGE_ATTRS else SetItems)
                   for a in self.attrs}

        dec_set = [d for d in self.full_set if d['id'] in dec_ids]
        dec_ids = [d['id'] for d in dec_set]
        out_set = [d for d in self.full_set if
                   d['id'] not in dec_ids and d['id'] != base_id]
        out_ids = [d['id'] for d in out_set]

        evals = {att: ATT_MAP[att](att, dec_ids, out_ids,
                                   dec_vals = [d[att] for d in dec_set],
                                   out_vals = [d[att] for d in out_set],
                                   num_uni_items = self.num_uni_items)
                 for att in self.attrs}

        exp = []
        used_attrs = []

        curr_base = set(dec_ids)
        curr_out = set(out_ids)

        base_len = len(curr_base)
        out_len = len(curr_out)

        full_exp = []

        while len(used_attrs) < len(self.attrs):

            curr_exps = []
            scores = []
            bases = []
            outs = []

            mod_attrs = [a for a in self.attrs if a not in used_attrs]

            for att in mod_attrs:

                ev = evals[att]

                if att in RANGE_ATTRS:
                    att_cands = [[(att, cand)] for cand in prod_cands(att, dec_set)]
                    exp, _, _ = best_cand(dec_set, out_set, att_cands, evals,
                                          self.curr_dec_scores,
                                          self.curr_out_scores,
                                          score_weight)

                    exp = exp[0][1]

                    a, b = exp
                    base, out = ev.range_q(a, b)

                else:
                    (base, out), exp = ev.set_greedy(curr_base, curr_out,
                                                     self.curr_dec_scores,
                                                     self.curr_out_scores,
                                                     score_weight)

                curr_exps.append(exp)
                bases.append(base)
                outs.append(out)

                # s = full_pot_eval(curr_base, base, dec_set,
                #                   curr_out, out, out_set,
                #                   self.curr_dec_scores,
                #                   self.curr_out_scores,
                #                   WEIGHTS, score_weight)


                curr_score, _, _ = full_eval(curr_base, len(dec_set),
                                             curr_out, len(out_set),
                                             self.curr_dec_scores,
                                             self.curr_out_scores,
                                             WEIGHTS, score_weight)



                changed_score, _, _ = full_eval(base, len(dec_set),
                                                out, len(out_set),
                                                self.curr_dec_scores,
                                                self.curr_out_scores,
                                                WEIGHTS, score_weight)

                scores.append(changed_score - curr_score)

            # print(curr_base)
            # print(bases)
            # print(mod_attrs)
            # print(curr_exps)
            # print(scores)
            # print()

            score_sort = np.argsort([-1 * s for s in scores])

            score_ind = score_sort[0]
            best_score = scores[score_ind]
            chosen_attr = mod_attrs[score_ind]
            if (best_score < 0) and len(used_attrs) > 0:
                break
            full_exp.append((chosen_attr, curr_exps[score_ind]))
            curr_base = curr_base.intersection(bases[score_ind])
            curr_out = curr_out.intersection(outs[score_ind])
            used_attrs.append(chosen_attr)

        final_score, rec, prec = full_eval(curr_base, base_len, curr_out, out_len,
                                           self.curr_dec_scores, self.curr_out_scores,
                                           WEIGHTS, score_weight)


        ret = {'exp_text': full_exp,
               'full_score': final_score,
               'rec': rec, 'prec': prec,
               'in': list(curr_base), 'out': list(curr_out),
               'in_nodes': [len(full_exp) for _ in curr_base],
               'out_nodes': [len(full_exp) for _ in curr_out]}


        #return full_exp, final_score, (rec, prec)
        return ret


    def _calc_scores(self, score_list):

        """
        Given a list of items [(id, score)], we produce
        a dictionary mapping the item to its normalized score
        """

        total = sum([s for _, s in score_list])
        return {i: s / total for i, s in score_list}

    def explain(self, base_id, dec_ids,
                meth = 'full_approx',
                score_weight = 'clf'):

        dispatch = {'full_approx': self._full_approx_explain,
                    'part_enum': self._part_enum_explain,
                    'full_enum': self._enum_explain}

        if score_weight == 'clf':
            self.curr_dec_scores = self._calc_scores(self.dec_scores[base_id])
            self.curr_out_scores = self._calc_scores(self.out_scores[base_id])

        return dispatch[meth](base_id, dec_ids, score_weight)


class DTBase:

    """
    Base class for both global and local DT explainers
    """

    def __init__(self, pred_mat, fid_pairs):

        self.pred_mat = pred_mat
        self.fid_pairs = fid_pairs

        self.rows = {}

        for i, (a, b) in enumerate(fid_pairs):
            self.rows[a] = self.rows.get(a, []) + [i]
            self.rows[b] = self.rows.get(b, []) + [i]

        for a in self.rows:
            self.rows[a] = set(self.rows[a])



    def _vote_clf_type(self, pos_pred_mat):

        """
        Depending on what we're explaining over, we might be able to
        use a text or image classifier instead of all_other, if every
        row has the same alternate clf type
        """

        types = pos_pred_mat['clf_type'].to_numpy()

        if (types == 'text').all():
            return 'text'
        elif (types == 'image').all():
            return 'image'
        else:
            return 'all_other'

    # def _active_nodes(self, tree, row, clf_type):

    #     act_nodes = [0]

    #     feat = tree.feature[0]
    #     thresh = tree.threshold[0]

    #     if row[FEAT_MAP[clf_type][feat]] <= thresh:
    #         arr = tree.children_left
    #     else:
    #         arr = tree.children_right

    #     next_ind = arr[0]

    #     while next_ind != -1:

    #         act_nodes.append(next_ind)

    #         feat = tree.feature[next_ind]
    #         thresh = tree.threshold[next_ind]

    #         if row[FEAT_MAP[clf_type][feat]] <= thresh:
    #             arr = tree.children_left
    #         else:
    #             arr = tree.children_right

    #         next_ind = arr[next_ind]

    #     return act_nodes


    def _active_nodes(self, tree, leaf_nodes):

        lnodes = list(set(leaf_nodes))

        left_children = tree.children_left.tolist()
        right_children = tree.children_right.tolist()

        touched_nodes = set(lnodes)
        for l in lnodes:

            curr = l
            while curr != 0:
                try:
                    parent = left_children.index(curr)
                except ValueError:
                    parent = right_children.index(curr)

                touched_nodes.add(parent)
                curr = parent

        return touched_nodes

    def _avail_nodes(self, tree):
        return set([i for i, f in enumerate(tree.feature) if f != -2])


    def _form_exp(self, tree, touched_nodes, clf_type):

        """
        Will take the form of a list of lists
        """

        exp = []

        feat = tree.feature[0]
        thresh = tree.threshold[0]
        feat_name = FEAT_MAP[clf_type][feat]

        left_child = tree.children_left[0]
        right_child = tree.children_right[0]

        if left_child in touched_nodes:

            l = tree.children_left[left_child] in touched_nodes
            r = tree.children_right[left_child] in touched_nodes
            leaf_node = (tree.children_left[left_child] == -1) and \
                (tree.children_right[left_child] == -1)

            top_level = (feat_name, '<=', thresh)

            if (l and r) or leaf_node:
                exp.append([top_level])

            elif l:
                next_level = (FEAT_MAP[clf_type][tree.feature[left_child]], '<=', tree.threshold[left_child])
                exp.append([top_level, next_level])

            elif r:
                next_level = (FEAT_MAP[clf_type][tree.feature[left_child]], '>', tree.threshold[left_child])
                exp.append([top_level, next_level])

        if right_child in touched_nodes:

            l = tree.children_left[right_child] in touched_nodes
            r = tree.children_right[right_child] in touched_nodes
            leaf_node = (tree.children_left[right_child] == -1) and \
                (tree.children_right[right_child] == -1)

            top_level = (feat_name, '>', thresh)

            if (l and r) or leaf_node:
                exp.append([top_level])

            elif tree.children_left[right_child] in touched_nodes:
                next_level = (FEAT_MAP[clf_type][tree.feature[right_child]], '<=', tree.threshold[right_child])
                exp.append([top_level, next_level])

            elif tree.children_right[right_child] in touched_nodes:
                next_level = (FEAT_MAP[clf_type][tree.feature[right_child]], '>', tree.threshold[right_child])
                exp.append([top_level, next_level])


        return exp



class DTExplainer(DTBase):

    """
    Similarly offers explanations of particular decision sets,
    but does so by fitting a short (max depth = 2) decision tree
    to the relevant rows
    """

    def __init__(self, pred_mat, fid_pairs):

        super().__init__(pred_mat, fid_pairs)

        self.clfs = {}
        self._load_dts()


    def _load_dt(self, c):

        base = MODELS_BASE + "dt_{}.pkl"
        with open(base.format(c), 'rb') as of:
            return pickle.load(of)


    def _load_dts(self):

        """
        Load pretrained dt classifiers
        """

        clf_types = ["all_other", "image", "text"]

        for c in clf_types:
            self.clfs[c] = self._load_dt(c)



    def explain(self, base_id, dec_ids, meth = None, score_weight = None):

        sub_pred_mat = self.pred_mat.iloc[list(self.rows[base_id]), :]
        pos_pred_mat = sub_pred_mat[sub_pred_mat['preds'] == 1]

        clf_type_vote = self._vote_clf_type(pos_pred_mat)
        exp_clf = self._load_dt(clf_type_vote)
        tree = exp_clf.tree_

        all_nodes = self._avail_nodes(tree)
        touched_nodes = set()

        for _, row in pos_pred_mat.iterrows():
            act_nodes = self._active_nodes(tree, row, clf_type_vote)
            touched_nodes = touched_nodes.union(set(act_nodes))
            if all_nodes == touched_nodes:
                break

        #print(touched_nodes)

        return self._form_exp(tree, touched_nodes, clf_type_vote)

        # Set up inverse index, pick out specific rows to make
        # temporary dataframes depending on base id, label the
        # ones as positive given a threshold score for the logistic
        #


        #This actually isn't what we want. What we want is, indeed,
        # a pretrained classifier, that we then have to select
        # paths from that explain its outputs



class LocalDTExplainer(DTBase):

    """
    Trains a Decision Tree of a set max depth on each
    explanation
    """

    def _get_ids_nodes(self, dec_pred_mat, base_id):

        in_ids = dec_pred_mat['file_id_A']
        in_ids[in_ids == base_id] = dec_pred_mat['file_id_B']
        in_ids = in_ids[dec_pred_mat['dt_preds'] > 0]
        in_nodes = dec_pred_mat.loc[dec_pred_mat['dt_preds'] > 0, 'dt_nodes']

        return in_ids.values.tolist(), in_nodes.values.tolist()

    def explain(self, base_id, dec_ids, meth = None, score_weight = None):

        sub_pred_mat = self.pred_mat.iloc[list(self.rows[base_id]), :]

        dec_rows = (sub_pred_mat['file_id_A'].isin(dec_ids)) | \
            (sub_pred_mat['file_id_B'].isin(dec_ids))

        sub_pred_mat.loc[:, 'temp_preds'] = (dec_rows).astype('int32')
        clf_type_vote = self._vote_clf_type(sub_pred_mat[dec_rows])

        exp_clf, sub_pred_mat, clf_type_vote = dt_preds_from_simil_rows(sub_pred_mat,
                                                                        clf_type = clf_type_vote)

        dec_pred_mat = sub_pred_mat[dec_rows]
        out_pred_mat = sub_pred_mat[~dec_rows]

        tree = exp_clf.tree_

        in_ids, in_nodes = self._get_ids_nodes(dec_pred_mat, base_id)
        out_ids, out_nodes = self._get_ids_nodes(out_pred_mat, base_id)

        rec = recall_score(sub_pred_mat['temp_preds'], sub_pred_mat['dt_preds'])
        prec = precision_score(sub_pred_mat['temp_preds'], sub_pred_mat['dt_preds'])
        full_score = eval_metrs(rec, prec, weights = WEIGHTS)

        touched_nodes = self._active_nodes(tree, in_nodes + out_nodes)

        ret_dict = {'exp_text': self._form_exp(tree, touched_nodes, clf_type_vote),
                    'rec': rec,
                    'prec': prec,
                    'full_score': full_score,
                    'in': in_ids,
                    'out': out_ids,
                    'in_nodes': in_nodes,
                    'out_nodes': out_nodes,
                    'total_nodes': tree.node_count,
                    'left_children': [i for i in tree.children_left if i != -1],
                    'right_children': [i for i in tree.children_right if i != -1],
                    'variant_clf': exp_clf,
                    'variant_feats': FEAT_MAP[clf_type_vote]}


        return ret_dict


class BaselineExplainer:

    def __init__(self):
        pass

    def explain(self, base_id, dec_ids, meth = None, score_weight = None):

        ret_dict = {'exp_text': base_id,
                    'rec': 0.0,
                    'prec': 0.0,
                    'full_score': 0.0,
                    'in': dec_ids, 'out': []}

        return ret_dict



def intersect_all(s_list):

    if not s_list:
        return set()

    base_set = s_list[0]
    for s in s_list[1:]:
        base_set = base_set.intersection(s)

    return base_set


def union_all(s_list):

    if not s_list:
        return set()

    base_set = s_list[0]
    for s in s_list[1:]:
        base_set = base_set.union(s)

    return base_set


def prod_cands(attr, dec_set):

    """
    The only viable candidates for optimal
    programs are based on the values of the
    decision set.
    """

    dec_vals = [d[attr] for d in dec_set]

    if attr in RANGE_ATTRS:
        return list([sorted(c) for c in combinations_with_replacement(dec_vals, 2)])
    else:

        dec_vals = [set(v) for v in dec_vals]

        full_combos = []
        for i in range(1, len(dec_vals)):
            combos = list(combinations(dec_vals, i))
            full_combos.extend([intersect_all(c) for c in combos])

        return full_combos


def best_cand(dec_set, out_set, exps, evals,
              dec_scores, out_scores, score_weight):

    """
    Use evals to identify the best explanation of the group
    and return
    """

    scores = []
    rec_precs = []
    for exp in exps:

        base_sets = []
        out_sets = []

        for attr_d in exp:

            att, expl = attr_d
            ev = evals[att]
            if att in RANGE_ATTRS:
                a, b = expl
                base, out = ev.range_q(a, b)
            else:
                base, out = ev.explain_count(expl)

            base_sets.append(base)
            out_sets.append(out)

        final_base = intersect_all(base_sets)
        final_out = intersect_all(out_sets)

        s, rec, prec = full_eval(final_base, len(dec_set), final_out, len(out_set),
                                 dec_scores, out_scores,
                                 WEIGHTS, score_weight)

        scores.append(s)
        rec_precs.append((rec, prec))

    best_score = np.argmax(scores)

    return exps[best_score], max(scores), rec_precs[best_score]


def eval_cand(att, expl, base_sets, out_sets, evals, cache_results,
              dec_scores, out_scores, score_weight):

    """
    Return values in decision set and outside decision
    set covered by explanation
    """

    ev = evals[att]
    if att in RANGE_ATTRS:
        ret_exp = expl
        a, b = expl
        lookup = str(a) + '-' + str(b)
        if att in cache_results and lookup in cache_results[att]:
            base, out = cache_results[att][lookup]
        else:
            base, out = ev.range_q(a, b)
            if att not in cache_results:
                cache_results[att] = {}
            cache_results[att][lookup] = (base, out)
    else:
        (base, out), ret_exp = ev.set_greedy(base_sets, out_sets,
                                             dec_scores, out_scores, score_weight)

    return (base, out), ret_exp

def part_cands(dec_set, out_set, exps, evals,
               dec_scores, out_scores, score_weight):

    """
    We first build out the range queries, and then
    based on the subsets those induce, we perform
    a greedy optimization in the style of the approximation
    algorithm for knapsack to produce an optimal candidate for
    that subset
    """

    scores = []
    rec_precs = []
    temp_exps = []

    cache_results = {}

    dec_len = len(dec_set)
    out_len = len(out_set)

    #print("Number of candidates: {}".format(len(exps)))

    orig_dec_sets = set(dec_set)
    orig_out_sets = set(out_set)

    for exp in exps:

        base_sets = orig_dec_sets
        out_sets = orig_out_sets

        temp_exp = []

        for att, expl in exp:

            (base, out), attr_exp = eval_cand(att, expl, base_sets,
                                              out_sets, evals, cache_results,
                                              dec_scores, out_scores,
                                              score_weight)
            base_sets = base_sets.intersection(base)
            out_sets = out_sets.intersection(out)
            temp_exp.append((att, attr_exp))

        final_score, rec, prec = full_eval(base_sets, dec_len, out_sets, out_len,
                                           dec_scores, out_scores,
                                           WEIGHTS, score_weight)

        scores.append(final_score)
        rec_precs.append((rec, prec))
        temp_exps.append(temp_exp)

    best_score = np.argmax(scores)

    #return exps[best_score], scores[best_score]
    return temp_exps[best_score], max(scores), rec_precs[best_score]




def pot_eval(base_take, base_len, out_take, out_len, weights = WEIGHTS):

    """
    Evaluate the change in score based on how many items
    would be removed from each set
    """

    w1, w2 = weights
    base_diff = -w1 * base_take / base_len
    out_diff = w2 * out_take / out_len

    rec = -base_take / base_len
    prec = out_take / out_len

    return (base_diff + out_diff)



def eval_metrs(faux_rec, faux_prec, weights = [0.5, 0.5]):
    w1, w2 = weights
    if np.isclose(faux_rec, 0.0) or np.isclose(faux_prec, 0.0):
        return 0.0
    else:
        return (1 / ((w1 / faux_rec) + (w2 / faux_prec)))
    #return (w1 * faux_rec + w2 * faux_prec)


def clf_pot_eval(base_change, base_scores, out_change, out_scores, weights = [0.5, 0.5]):

    w1, w2 = weights
    rec = -sum([base_scores[b] for b in base_change])
    prec = sum([out_scores[b] for b in out_change])

    if np.isclose(rec, 0.0) or np.isclose(prec, 0.0):
        return 0.0
    else:
        return (1 / ((w1 / rec) + (w2 / prec)))


    #return w1 * rec + w2 * prec

def clf_score_eval(base_set, out_set, base_scores, out_scores, weights = [0.5, 0.5]):

    w1, w2 = weights
    rec = sum([base_scores[b] for b in base_set])
    prec = (1 - sum([out_scores[b] for b in out_set]))

    if np.isclose(rec, 0.0) or np.isclose(prec, 0.0):
        return 0.0, rec, prec
    else:
        return (1 / ((w1 / rec) + (w2 / prec))), rec, prec
    #return w1 * rec + w2 * prec, rec, prec

def full_pot_eval(curr_base, base, full_base,
                  curr_out, out, full_out,
                  dec_scores, out_scores, weights, score_weight):

    if score_weight == 'standard':
        base_take = len(curr_base) - len(curr_base.intersection(base))
        out_take = len(curr_out) - len(curr_out.intersection(out))
        s = pot_eval(base_take, len(full_base), out_take, len(full_out), weights = WEIGHTS)
    elif score_weight == 'clf':
        base_change = curr_base - curr_base.intersection(base)
        out_change = curr_out - curr_out.intersection(out)
        s = clf_pot_eval(base_change, dec_scores,
                         out_change, out_scores,
                         weights = WEIGHTS)

    return s

def full_eval(curr_base, base_len, curr_out, out_len,
              base_scores, out_scores, weights, score_weight):

    """
    Evaluate score of an explanation depending on strategy
    """


    if score_weight == 'standard':

        rec = len(curr_base) / base_len
        prec = 1 - (len(curr_out) / out_len)
        final_score = eval_metrs(rec, prec, weights = WEIGHTS)

    elif score_weight == 'standard_mod':

        rec = len(curr_base) / base_len if base_len > 0 else 0.0
        prec = len(curr_base) / (base_len + len(curr_out)) if base_len > 0 else 0.0
        final_score = eval_metrs(rec, prec, weights = WEIGHTS)

    elif score_weight == 'clf':
        final_score, rec, prec = clf_score_eval(curr_base, curr_out,
                                     base_scores, out_scores,
                                     weights = WEIGHTS)

    # Ran into float instability issues. Hopefully this fixes it.

    DECIMALS = 5
    final_score = np.around(final_score, decimals = DECIMALS)
    rec = np.around(rec, decimals = DECIMALS)
    prec = np.around(prec, decimals = DECIMALS)

    return final_score, rec, prec

def size_text(val, gdoc = True):

    SMALL_SIZE = int(1e6)
    MED_SIZE = int(1e9)


    if gdoc and np.isclose(val, 0.0):
        return "---"

    if val <= SMALL_SIZE:
        return "{0:,.1f} Kb".format(val / 1e3)
    elif val <= MED_SIZE:
        return "{0:,.1f} Mb".format(val / 1e6)
    else:
        return "{0:,.1f} Gb".format(val / 1e9)


def plaintext_explain(exp):

    """
    Given an explanation over several attributes, provide a plaintext /
    readable version of the explanation
    """

    full_exp = []

    for att, e in exp:
        p = plaintext_attr_exp(att, e)
        full_exp.append(p)

    #full_exp[0] = full_exp[0].capitalize()

    return "The system is recommending all file(s) that" + " AND ".join(full_exp)


def plaintext_attr_exp(att, e):

    if att == 'date':
        a, b = e
        a = datetime.datetime.utcfromtimestamp(a).strftime("%Y-%m-%d, %H:%M")
        b = datetime.datetime.utcfromtimestamp(b).strftime("%Y-%m-%d, %H:%M")

        p = "the file(s) were last modified between {} and {}.".format(a, b)

    elif att == 'size':
        a, b = e

        p = "the file(s) have size from {} to {}.".\
            format(size_text(a, gdoc = False), size_text(b, gdoc = False))
        #p = "the file has size between {0:0.3f}Mb and {1:0.3f}Mb".format(a / 1e6, b / 1e6)
    elif att == 'ext':
        if len(e) > 1  and len(e[1]) > 0:
            a, b = e
            p = "the file(s) have one of the extensions [" + str(a[0]) + "," +  str(b[0]) + "]."
        else:
            p = "the file(s) have the extension '" + e[0][0] + "'."
    elif att == 'starttoks':

        if len(e) > 1  and len(e[1]) > 0:
            a, b = e
            p = "the filename(s) start with '" + str(a[0]) + "' OR '" +  str(b[0]) + "'."
        else:
            p = "the filename(s) start with '" + e[0][0] + "'."

    elif att == 'filetoks' or att == 'modfiletoks':
        if len(e) > 1 and len(e[1]) > 0:
            a, b, = e
            p = "the filename(s) contain either sub-part(s) {} OR sub-part(s) {}.".format(a, b)
        else:
            p = "the filename(s) contain sub-part(s) {}.".format(e[0])
    elif att == 'texttoks':
        if len(e) > 1  and len(e[1]) > 0:
            a, b = e
            p = "the file data contains the word(s) {} OR the word(s) {}.".format(a, b)
        else:
            p = "the file data contains the word(s) {}.".format(e[0])
    elif att == 'shared':
        if len(e) > 1  and len(e[1]) > 0:
            a, b = e
            p = "the file(s) are shared with {} OR the word(s) {}.".format(a, b)
        else:
            p = "the file(s) are shared with {}.".format(e[0])
    elif att == 'imgobjs':
        if len(e) > 1  and len(e[1]) > 0:
            a, b = e
            p = "the system thought it saw the object(s) {} OR the object(s) {} in the image(s).".format(a, b)
        else:
            p = "the system thought it saw the object(s) {} in the image(s).".format(e[0])
    elif att == 'path':
        if len(e) > 1  and len(e[1]) > 0:
            a, b = e
            p = "the folder(s) {} OR the folder(s) {} appear in the filepath.".format(a, b)
        else:
            p = "the folder(s) {} appear in the filepath.".format(e[0])



    return p



def html_attr_exp(att, e):

    if att == 'date':
        a, b = e
        a = datetime.datetime.utcfromtimestamp(a).strftime("%Y-%m-%d, %H:%M")
        b = datetime.datetime.utcfromtimestamp(b).strftime("%Y-%m-%d, %H:%M")

        p = "The file(s) were <b>last modified</b> between <u>{}</u> and <u>{}</u>".format(a, b)

    elif att == 'size':
        a, b = e

        p = "The file(s) have <b>size</b> from <u>{}</u> to <u>{}</u>".\
            format(size_text(a, gdoc = False), size_text(b, gdoc = False))
        #p = "the file has size between {0:0.3f}Mb and {1:0.3f}Mb".format(a / 1e6, b / 1e6)
    elif att == 'ext':
        if len(e) > 1  and len(e[1]) > 0:
            a, b = e
            p = "The file(s) have one of the <b>extensions</b> <u>[" + str(a[0]) + "," +  str(b[0]) + "]</u>"
        else:
            p = "The file(s) have the <b>extension</b> <u>'" + e[0][0] + "'</u>"
    elif att == 'starttoks':

        if len(e) > 1  and len(e[1]) > 0:
            a, b = e
            p = "The <b>filename(s) start with</b> <u>'" + str(a[0]) + "'</u> <span style='color: purple'><b>OR</b></span> <u>'" +  str(b[0]) + "'</u>"
        else:
            p = "The <b>filename(s) start with</b> <u>'" + e[0][0] + "'</u>"

    elif att == 'filetoks' or att == 'modfiletoks':
        if len(e) > 1 and len(e[1]) > 0:
            a, b, = e
            p = "The <b>filename(s) contain</b> either sub-part(s) <u>{}</u> <span style='color: purple'><b>OR</b></span> sub-part(s) <u>{}</u>".format(a, b)
        else:
            p = "The <b>filename(s) contain</b> sub-part(s) <u>{}</u>".format(e[0])
    elif att == 'texttoks':
        if len(e) > 1  and len(e[1]) > 0:
            a, b = e
            p = "The <b>file data contains the word(s)</b> <u>{}</u> <span style='color: purple'><b>OR</b></span> the word(s) <u>{}</u>".format(a, b)
        else:
            p = "The file data contains the word(s) <u>{}</u>".format(e[0])
    elif att == 'shared':
        if len(e) > 1  and len(e[1]) > 0:
            a, b = e
            p = "The <b>file(s) are shared with</b> <u>{}</u> <span style='color: purple'><b>OR</b></span> the word(s) <u>{}</u>".format(a, b)
        else:
            p = "The file(s) are shared with <u>{}</u>".format(e[0])
    elif att == 'imgobjs':
        if len(e) > 1  and len(e[1]) > 0:
            a, b = e
            p = "The system thought it <b>saw the object(s)</b> <u>{}</u> <span style='color: purple'><b>OR</b></span> the object(s) <u>{}</u> in the image(s)".format(a, b)
        else:
            p = "The system thought it <b>saw the object(s)</b> <u>{}</u> in the image(s)".format(e[0])

    elif att == 'path':
        if len(e) > 1  and len(e[1]) > 0:
            a, b = e
            p = "The folder(s) <u>{}</u> <span style='color: purple'><b>OR</b></span> the folder(s) <u>{}</u> <b>appear in the filepath</b>".format(a, b)
        else:
            p = "The folder(s) <u>{}</u> <b>appear in the filepath</b>".format(e[0])

    return p


def html_explain(exp):

    """
    Given an explanation over several attributes, provide a plaintext /
    readable version of the explanation
    """

    full_exp = []

    for att, e in exp:
        p = html_attr_exp(att, e)

        full_exp.append("<div>" + p + "</div>")

    #full_exp[0] = full_exp[0].capitalize()

    outer_div = "<div style='display: flex; flex-direction: column; justify-content: space-evenly; column-gap: 5px; align-items: center;'>"
    mid_div = "<div>The system is recommending every file that matches the following criteria: </div>"

    return outer_div + mid_div + " <div style='color: blue; width=100%;'><b>AND</b></div> ".join(full_exp) + "</div>"


def escape(s):

    """
    Escape the relevant characters to ensure a string can go
    in quotes for graphviz syntax without erroring out
    """

    s = s.replace("'", "\\\'")
    s = s.replace('"', '\\\"')
    s = s.replace('\n', '\\n')

    return s


def graph_exp(exp, num_samps):

    """
    Produce a graphviz visualization of a decision tree for our rule-based
    explanations
    """

    start_text = "digraph Tree {\n"
    node_text = 'node [shape=box, style="filled", color="black"] ;\n'
    end_text = "}"

    full_text = start_text + node_text

    node = 0
    for att, e in exp:
        p = plaintext_attr_exp(att, e).capitalize()
        p = escape(p)
        node_row = '{} [label="{}", fillcolor="#ffffff"] ;\n'.format(node, "\\n".join(wrap(p, LINE_LEN)))

        if node != 0:
            arrow_row = '{} -> {} [labeldistance=2.5, labelangle=45, headlabel="True"] ;\n'.format(node - 1, node)
            full_text += arrow_row

        full_text += node_row
        node += 1


    node_row = '{} [label="# of files: {}", fillcolor="#01c221"] ;\n'.format(node, num_samps)
    arrow_row = '{} -> {} [labeldistance=2.5, labelangle=45, headlabel="True"] ;\n'.format(node - 1, node)

    full_text += node_row
    full_text += arrow_row

    node += 1

    for i in range(node - 2, -1, -1):
        node_row = '{} [label="None", fillcolor="#ff5555"] ;\n'.format(node, num_samps)
        arrow_row = '{} -> {} [labeldistance=2.5, labelangle=-45, headlabel="False"] ;\n'.format(i, node)

        full_text += node_row
        full_text += arrow_row

        node += 1


    return full_text + end_text




def create_clause(clause, embel = False):

    """
    Take clause of the form (feature, > / <=, thresh) and turn
    it into plaintext

    """

    if not embel:
        return FULL_FEATURES[clause[0]] + " is " + clause[1] + " {0:.2f}".format(clause[2])
    else:
        return "<b>" + FULL_FEATURES[clause[0]] + "</b> is <u>" + clause[1] + " {0:.2f}</u>".format(clause[2])



def dt_plaintext(exp):

    """
    Turn DT explanations into plaintext
    """

    if len(exp) > 1:
        joint_a, joint_b = exp

        if len(joint_a) > 1:
            a, b = joint_a
            clause_a = create_clause(a) + " AND " + create_clause(b)
        else:
            clause_a = create_clause(joint_a[0])

        if len(joint_b) > 1:
            a, b = joint_b
            clause_b = create_clause(a) + " AND " + create_clause(b)
        else:
            clause_b = create_clause(joint_b[0])

        return "(" + clause_a + ")" + " OR " + "(" + clause_b + ")"

    else:
        joint_a = exp[0]
        if len(joint_a) > 1:
            a, b = joint_a
            return create_clause(a) + " AND " + create_clause(b)
        else:
            return create_clause(joint_a[0])
