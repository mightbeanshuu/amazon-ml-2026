"""Where does blocking lose true pairs? Runs the big-pipeline blocking for one train country on an S1 sample,
before any pruning, and reports pair completeness (PC) per field, for the union, and after top-K pruning for
several K and two ranking scores. Dumps the never-proposed true pairs with their normalised text.

    python -m tools.block_diag --prep prep --data student_resource/dataset --country india --n 30000 \
        --out runs/block_diag_india
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.block_big import KEY_FIELDS, block_country, group_rank  # noqa: E402
from src.metrics import parse_ids  # noqa: E402
from src.run_big import BLOCK_COLS, load, load_r  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--country", required=True)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--kscale", type=float, default=1.0)
    ap.add_argument("--max_df_keys", type=int, default=3000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    c = a.country
    s1 = load(a.prep, "train", "s1", c, BLOCK_COLS).reset_index(drop=True)
    r = load_r(a.prep, "train", c, BLOCK_COLS)
    samp = np.sort(np.random.RandomState(0).choice(len(s1), min(a.n, len(s1)), replace=False))
    keep = np.zeros(len(s1), bool)
    keep[samp] = True
    log(f"[{c}] s1 {len(s1)} r {len(r)} sample {len(samp)}")
    pairs = block_country(s1, r, fields=KEY_FIELDS, kscale=a.kscale, max_df_keys=a.max_df_keys, log=log, keep_s1=keep)
    gt = pd.read_csv(f"{a.data}/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False,
                     quoting=csv.QUOTE_NONE)
    sid = set(s1.entity_id.values[samp])
    gt = gt[gt.source1_entity_id.isin(sid)]
    rpos = pd.Series(np.arange(len(r)), index=r.entity_id.values)
    s1pos = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
    g_s1, g_r = [], []
    for e, m in zip(gt.source1_entity_id, gt.matched_entity_ids):
        for x in parse_ids(m):
            if x in rpos.index:
                g_s1.append(s1pos[e])
                g_r.append(rpos[x])
    n_r = len(r)
    gkey = np.array(g_s1, np.int64) * n_r + np.array(g_r, np.int64)
    pkey = pairs.s1_i.values.astype(np.int64) * n_r + pairs.r_i.values
    hit = np.isin(pkey, gkey)
    pairs["y"] = hit
    n_gold = len(gkey)
    rep = {"n_gold_pairs": n_gold, "n_pairs": len(pairs), "cands_per_s1": round(len(pairs) / len(samp), 2)}
    for f in KEY_FIELDS:
        rep[f"PC_{f}"] = round(float(pairs.loc[pairs[f].notna(), "y"].sum()) / n_gold, 4)
    rep["PC_union"] = round(float(hit.sum()) / n_gold, 4)
    alt = np.fmax(pairs["b_nkey"].values, pairs["b_akey"].values)
    scores = {
        "cur": np.where(np.isnan(pairs["b_key"].values), np.nan_to_num(alt) * 0.5, pairs["b_key"].values),
        "best": pairs["b_best"].values,
        "sum": np.nan_to_num(pairs["b_key"].values) + 0.5 * np.nan_to_num(pairs["b_nkey"].values)
        + 0.5 * np.nan_to_num(pairs["b_akey"].values),
    }
    for nm, sc in scores.items():
        rk = group_rank(pairs.s1_i.values, sc.astype(np.float32))
        for K in (30, 40, 50, 60, 80):
            rep[f"PC_{nm}_top{K}"] = round(float(hit[rk <= K].sum()) / n_gold, 4)
    log(json.dumps(rep, indent=1))
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
    # never-proposed true pairs, with text
    miss = np.setdiff1d(gkey, pkey[hit])
    ms1, mr = miss // n_r, miss % n_r
    full_s1 = load(a.prep, "train", "s1", c, ["entity_id", "name_clean", "addr_norm"]).reset_index(drop=True)
    full_r = load_r(a.prep, "train", c, ["entity_id", "name_clean", "addr_norm"])
    d = pd.DataFrame({"s1_id": full_s1.entity_id.values[ms1], "r_id": full_r.entity_id.values[mr],
                      "s1_name": full_s1.name_clean.values[ms1], "r_name": full_r.name_clean.values[mr],
                      "s1_addr": full_s1.addr_norm.values[ms1], "r_addr": full_r.addr_norm.values[mr]})
    d.to_csv(f"{a.out}/never_proposed.tsv", sep="\t", index=False)
    pairs.to_parquet(f"{a.out}/pairs.parquet", compression="zstd", index=False)
    log(f"never proposed: {len(d)}; saved to {a.out}")


if __name__ == "__main__":
    main()
