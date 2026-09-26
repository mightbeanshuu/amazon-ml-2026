"""Turn pair probabilities into per-S1 match sets that maximise macro F0.5.

Rule A (EUM, thresholds tuned directly on out-of-fold data):
  keep the top-1 if p >= t1; keep each extra candidate if p >= t2 and p >= p_top - delta.
  Optional exclusivity: an S2/S3 record may only go to the S1 where it has its highest p,
  because S1 is deduplicated.
  Theory (see the playbook): t1 ~ 0.5 and t2 ~ 0.73-0.76 for calibrated p.
Rule B (decision-theoretic): isotonic-calibrate p, then expected-F0.5 set selection per S1.
The macro F0.5 is computed vectorised: per S1, F = 1.25*tp / (0.25*n_gold + n_pred).
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .metrics import best_set_fbeta


class Evaluator:
    """Fast macro-F0.5 over ALL S1 entities (including those with no candidates)."""

    def __init__(self, s1_codes, y, n_gold, n_s1):
        self.s1 = s1_codes            # int code of the S1 per candidate row
        self.y = y.astype(bool)
        self.n_gold = n_gold          # true match count per S1 code (length n_s1)
        self.n_s1 = n_s1

    def score(self, sel, per_entity=False):
        tp = np.bincount(self.s1, weights=(sel & self.y), minlength=self.n_s1)
        npred = np.bincount(self.s1, weights=sel, minlength=self.n_s1)
        g = self.n_gold
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(g == 0, (npred == 0).astype(float), np.where(tp > 0, 1.25 * tp / (0.25 * g + npred), 0.0))
        return f if per_entity else float(f.mean())


def exclusive_mask(r_codes, p, slack=0.0):
    """True where p is the maximum (within slack) among all S1 candidates of this S2/S3 record."""
    mx = pd.Series(p).groupby(r_codes).transform("max").values
    return p >= mx - slack


def _prep_order(s1_codes, q):
    order = np.lexsort((-q, s1_codes))
    s_sorted, q_sorted = s1_codes[order], q[order]
    first = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
    top_idx = np.maximum.accumulate(np.where(first, np.arange(len(q_sorted)), 0))
    return order, q_sorted, first, q_sorted[top_idx]


def rule_select(s1_codes, p, t1, t2, delta, excl_mask=None, t_shift=None, pre=None):
    """Vectorised rule A. t_shift (per row) raises thresholds, e.g. for unseen country labels.
    pre: cached _prep_order output for this (s1_codes, p, excl_mask), reused during grid search."""
    q = p if excl_mask is None else np.where(excl_mask, p, -1.0)
    order, q_sorted, first, p_top = pre if pre is not None else _prep_order(s1_codes, q)
    shift = 0.0 if t_shift is None else t_shift[order]
    top_ok = p_top >= t1 + shift
    sel_sorted = np.where(first, q_sorted >= t1 + shift,
                          top_ok & (q_sorted >= t2 + shift) & (q_sorted >= p_top - delta))
    sel = np.zeros(len(p), bool)
    sel[order] = sel_sorted
    return sel


def tune_rule(ev, s1_codes, r_codes, p, t_shift=None, grid=None):
    grid = grid or dict(t1=np.round(np.arange(0.25, 0.86, 0.05), 2), t2=np.round(np.arange(0.5, 0.97, 0.05), 2),
                        delta=[1.0, 0.4, 0.2], excl=[True, False])
    masks = {True: exclusive_mask(r_codes, p), False: None}
    pres = {ex: _prep_order(s1_codes, p if m is None else np.where(m, p, -1.0)) for ex, m in masks.items()}
    best = (-1, None)
    for t1, t2, d, ex in itertools.product(grid["t1"], grid["t2"], grid["delta"], grid["excl"]):
        if t2 < t1:
            continue
        s = ev.score(rule_select(s1_codes, p, t1, t2, d, masks[ex], t_shift, pre=pres[ex]))
        if s > best[0]:
            best = (s, dict(t1=float(t1), t2=float(t2), delta=float(d), excl=bool(ex)))
    return best


def fit_calibrator(p, y):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    return iso


def gfm_select(s1_codes, r_codes, p_cal, excl=True):
    """Rule B: per-S1 expected-F0.5-optimal set (independence approximation)."""
    q = np.where(exclusive_mask(r_codes, p_cal), p_cal, 0.0) if excl else p_cal
    sel = np.zeros(len(q), bool)
    order = np.argsort(s1_codes, kind="stable")
    bounds = np.flatnonzero(np.r_[True, s1_codes[order][1:] != s1_codes[order][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        pick, _ = best_set_fbeta(q[idx])
        sel[idx[pick]] = True
    return sel
