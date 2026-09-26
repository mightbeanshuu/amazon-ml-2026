"""How small can the candidate set get with a deterministic blocking rule (no scoring model, so the pruned set is
legitimately the blocking output)? Measures, on out-of-fold train data, mean candidates per S1, pair completeness
and the final macro F0.5 of the stage-A OOF p restricted to the survivors.

    python -m tools.cascade_lab --run runs/real_v2
Needs oof_pA.npy (written by tools.rule_lab).

Rule: fused blocking score s = key + w * (name-only + address-only); keep a candidate if it is among its S1's
top-k by s AND s >= rho * (best s of that S1).
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
from src import decide  # noqa: E402
from src.block_big import group_rank  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fused(T, w):
    return (np.nan_to_num(T.b_key.values) + w * (np.nan_to_num(T.b_nkey.values) + np.nan_to_num(T.b_akey.values)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    meta = json.load(open(f"{a.run}/models/meta.json"))
    rule = meta["rule"]
    T = pd.read_parquet(f"{a.run}/train_features.parquet",
                        columns=["b_key", "b_nkey", "b_akey", "y", "r_i", "s1_id", "country"])
    y = T.y.values.astype(bool)
    pA = np.load(f"{a.run}/oof_pA.npy")
    gold_sizes = json.load(open(f"{a.run}/train_sample_gold_sizes.json"))
    ids = [e for c in gold_sizes for e in gold_sizes[c]]
    code = {e: i for i, e in enumerate(ids)}
    ng = np.array([gold_sizes[c][e] for c in gold_sizes for e in gold_sizes[c]])
    s1c = np.array([code[e] for e in T.s1_id.values])
    r_i = T.r_i.values
    n_s1, n_gold = len(ids), int(ng.sum())
    ev = decide.Evaluator(s1c, y, ng, n_s1)

    def final_f(keep):
        idx = np.flatnonzero(keep)
        excl = np.zeros(len(pA), bool)
        excl[idx] = decide.exclusive_mask(r_i[idx], pA[idx])
        q = np.where(keep, pA, 0.0)
        sel = decide.rule_select(s1c, q, rule["t1"], rule["t2"], rule["delta"], excl if rule["excl"] else None)
        return ev.score(sel & keep)

    res = {"base": dict(cands=round(len(T) / n_s1, 2), PC=round(y.sum() / n_gold, 4),
                        F=round(final_f(np.ones(len(T), bool)), 5))}
    log("base", res["base"])
    for w in (0.5, 1.0):
        s = fused(T, w).astype(np.float32)
        rk = group_rank(s1c, s)
        smax = pd.Series(s).groupby(s1c).transform("max").values
        rel = np.where(smax > 0, s / np.maximum(smax, 1e-9), 0)
        for k, rho in itertools.product((5, 6, 8, 10, 12, 15), (0.0, 0.3, 0.4, 0.5, 0.6)):
            keep = (rk <= k) & (rel >= rho)
            key = f"w{w}_k{k}_rho{rho}"
            res[key] = dict(cands=round(keep.sum() / n_s1, 2), PC=round((y & keep).sum() / n_gold, 4),
                            F=round(final_f(keep), 5))
            log(key, res[key])
    json.dump(res, open(f"{a.run}/cascade_lab_rule.json", "w"), indent=1)
    log("saved")


if __name__ == "__main__":
    main()
