"""Decision-rule lab on out-of-fold probabilities: which way of turning pair p into match sets scores best?

    python -m tools.rule_lab --run runs/real_v2

Recomputes the stage-A OOF p exactly as train_phase does (same folds, params, seed) and caches it as oof_pA.npy.
It then compares the following, scored with 2-fold cross-fitting over S1 (fit on one half, score the other) so
tuned rules are not graded on their own tuning data:
  A_global    the current rule (t1, t2, delta, exclusivity) tuned on all countries
  A_country   the same rule tuned per country
  A_rank      separate thresholds for the top-1, the 2nd and the 3rd+ candidate
  GFM         isotonic-calibrated p, then the expected-F0.5-optimal set per S1
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import decide, model  # noqa: E402
from src.metrics import best_set_fbeta  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rank_select(s1c, q, t1, t2, t3):
    """Top-1 if q>=t1, 2nd if also q>=t2, 3rd and later if also q>=t3 (q already carries the exclusivity mask)."""
    order = np.lexsort((-q, s1c))
    s, v = s1c[order], q[order]
    first = np.r_[True, s[1:] != s[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(s)), 0))
    rk = np.arange(len(s)) - start
    top_ok = v[start] >= t1
    ok = np.where(rk == 0, v >= t1, top_ok & np.where(rk == 1, v >= t2, v >= t3))
    sel = np.zeros(len(q), bool)
    sel[order] = ok
    return sel


def gfm(s1c, q, max_m=10, floor=0.02):
    """Expected-F0.5-optimal set per S1 from calibrated q (rows with q < floor are never picked)."""
    sel = np.zeros(len(q), bool)
    idx = np.flatnonzero(q >= floor)
    order = idx[np.lexsort((-q[idx], s1c[idx]))]
    s = s1c[order]
    b = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
    for a, z in zip(b[:-1], b[1:]):
        rows = order[a:z]
        pick, _ = best_set_fbeta(q[rows], max_m=max_m)
        sel[rows[pick]] = True
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    meta = json.load(open(f"{a.run}/models/meta.json"))
    cfg, feat_cols = meta["cfg"], meta["feat_cols"]
    T = pd.read_parquet(f"{a.run}/train_features.parquet")
    y = T.y.values.astype(bool)
    cache = f"{a.run}/oof_pA.npy"
    if os.path.exists(cache):
        pA = np.load(cache)
    else:
        log("recomputing OOF pA")
        pA, _ = model.oof_predict(T[feat_cols], y.astype(int), T.s1_i.values, cfg["n_splits"], params=cfg["lgb"],
                                  seeds=tuple(cfg["seeds"]))
        np.save(cache, pA)
    s1_id, r_i, country = T.s1_id.values, T.r_i.values, T.country.values
    del T
    gold_sizes = json.load(open(f"{a.run}/train_sample_gold_sizes.json"))
    ids = [e for c in gold_sizes for e in gold_sizes[c]]
    code = {e: i for i, e in enumerate(ids)}
    ng = np.array([gold_sizes[c][e] for c in gold_sizes for e in gold_sizes[c]])
    ctry_of = np.array([c for c in gold_sizes for _ in gold_sizes[c]])
    s1c = np.array([code[e] for e in s1_id])
    ev = decide.Evaluator(s1c, y, ng, len(ids))
    excl = decide.exclusive_mask(r_i, pA)
    q = np.where(excl, pA, -1.0)

    # 2-fold cross-fit over S1 codes
    half = np.random.RandomState(1).rand(len(ids)) < 0.5
    rows_half = half[s1c]

    def cv(fit, apply):
        """fit(row_mask, s1_mask) -> params; apply(params) -> sel over all rows. Returns per-country and overall."""
        fe = np.zeros(len(ids))
        for h in (True, False):
            prm = fit(rows_half == h, half == h)
            f = ev.score(apply(prm), per_entity=True)
            fe[half != h] = f[half != h]
        out = {c: round(float(fe[ctry_of == c].mean()), 5) for c in gold_sizes}
        out["all"] = round(float(fe.mean()), 5)
        return out

    res = {}
    # Tuning on one half: rows of the other half are dropped and its S1 get n_gold 0, so they score a constant 1.0.
    def fit_A(rmask, smask, cmask=None):
        m = rmask if cmask is None else rmask & cmask
        e = decide.Evaluator(s1c[m], y[m], np.where(smask, ng, 0), len(ids))
        return decide.tune_rule(e, s1c[m], r_i[m], pA[m])[1]

    def apply_A(prm):
        return decide.rule_select(s1c, pA, prm["t1"], prm["t2"], prm["delta"], excl if prm["excl"] else None)

    log("A_global")
    res["A_global"] = cv(fit_A, apply_A)
    log(res["A_global"])

    log("A_country")
    ctry_rows = country

    def fit_Ac(rmask, smask):
        return {c: fit_A(rmask, smask & (ctry_of == c), ctry_rows == c) for c in gold_sizes}

    def apply_Ac(prm):
        sel = np.zeros(len(pA), bool)
        for c, p in prm.items():
            m = ctry_rows == c
            sel[m] = decide.rule_select(s1c[m], pA[m], p["t1"], p["t2"], p["delta"], excl[m] if p["excl"] else None)
        return sel

    res["A_country"] = cv(fit_Ac, apply_Ac)
    log(res["A_country"])

    log("A_rank")
    grid = np.round(np.arange(0.3, 0.96, 0.05), 2)

    def fit_R(rmask, smask):
        e = decide.Evaluator(s1c[rmask], y[rmask], np.where(smask, ng, 0), len(ids))
        best = (-1, None)
        for t1, t2, t3 in itertools.product(grid, grid, grid):
            if t2 < t1 or t3 < t1:
                continue
            s = e.score(rank_select(s1c[rmask], q[rmask], t1, t2, t3))
            if s > best[0]:
                best = (s, (t1, t2, t3))
        return best[1]

    res["A_rank"] = cv(fit_R, lambda prm: rank_select(s1c, q, *prm))
    log(res["A_rank"])

    log("GFM")

    def fit_G(rmask, smask):
        return decide.fit_calibrator(pA[rmask & excl], y[rmask & excl])

    def apply_G(iso):
        qc = np.where(excl, iso.predict(pA), 0.0)
        return gfm(s1c, qc)

    res["GFM"] = cv(fit_G, apply_G)
    log(res["GFM"])
    json.dump(res, open(f"{a.run}/rule_lab.json", "w"), indent=1)
    log("saved", f"{a.run}/rule_lab.json")


if __name__ == "__main__":
    main()
