#!/usr/bin/env python3
"""Stage 1 — candidate generation + recall report.

Measurement run (fast, unbiased full-scale recall estimate: a random fraction of S2/S3
records queries the FULL S1 index):
    python run_stage1.py --data-dir DATASET_DIR --split train --query-frac 0.05

Full run (writes candidate edges for all S2/S3 records):
    python run_stage1.py --data-dir DATASET_DIR --split train --query-frac 1.0
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er.data import load_split, truth_index, log
from er.candidates import PASSES, run_pass, union

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder containing train/ and test/")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--work-dir", default="/kaggle/working/er_work")
    ap.add_argument("--query-frac", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=10, help="neighbours kept per pass")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.work_dir, exist_ok=True)
    T0 = time.time()

    s1, s23, gt = load_split(a.data_dir, a.split, a.work_dir, n_proc=a.threads)
    log(f"S1={len(s1):,}  S2+S3={len(s23):,}  countries S1={s1.country.value_counts().to_dict()}")
    rng = np.random.default_rng(a.seed)
    q_idx = np.sort(rng.choice(len(s23), size=int(len(s23) * a.query_frac), replace=False)) if a.query_frac < 1 \
        else np.arange(len(s23))
    log(f"querying {len(q_idx):,} S2/S3 records ({a.query_frac:.0%}), k={a.k}, threads={a.threads}")

    edges, times = [], {}
    for spec in PASSES:
        t = time.time()
        e = run_pass(s1, s23, q_idx, spec, a.k, a.threads)
        times[spec[0]] = time.time() - t
        edges.append(e)
    E = pd.concat(edges, ignore_index=True)
    tag = f"{a.split}_q{a.query_frac:g}_k{a.k}"
    E.to_parquet(os.path.join(a.work_dir, f"edges_{tag}.parquet"), index=False)
    log(f"saved edges_{tag}.parquet ({len(E):,} rows)")

    print("\n================ STAGE 1 REPORT ================")
    print(f"split={a.split} query_frac={a.query_frac} k={a.k} total_time={(time.time()-T0)/60:.1f} min")
    for p, s in times.items():
        proj = s / a.query_frac if a.query_frac < 1 else s
        print(f"  pass {p:6}: {s/60:6.1f} min  (projected for 100% queries: {proj/60:6.1f} min)")
    if gt is not None:
        owner = truth_index(gt, s1, s23)
        qo = owner[q_idx]; n_true = int((qo >= 0).sum())
        print(f"\n  queried records with a true S1: {n_true:,} of {len(q_idx):,} ({n_true/len(q_idx):.1%})")
        print(f"  {'pass':8}" + "".join(f"  recall@{k:<3}" for k in (1, 3, 5, 10) if k <= a.k))
        for p in [s[0] for s in PASSES] + ["UNION"]:
            sub = E if p == "UNION" else E[E["pass"] == p]
            line = f"  {p:8}"
            for k in (1, 3, 5, 10):
                if k > a.k: continue
                u = union(sub, k)
                hit = (owner[u.q.values] == u.s1.values).sum()
                line += f"  {hit/n_true:.4f}    "
            print(line)
        for k in (3, 5, 10):
            if k > a.k: continue
            u = union(E, k)
            print(f"  UNION top-{k:<2}: {len(u)/len(q_idx):5.1f} candidates per S2/S3 record "
                  f"-> ~{len(u)/len(q_idx)*len(s23)/1e6:,.0f}M pairs at full scale")
        # misses by country for the chosen k=5
        u = union(E, min(5, a.k)); found = set(u.q.values[owner[u.q.values] == u.s1.values])
        miss = np.array([q for q in q_idx if owner[q] >= 0 and q not in found])
        print(f"\n  misses @5 by country: {pd.Series(s23.country.values[miss]).value_counts().to_dict()}")
        print(f"  misses @5 with empty address: {(s23.ad.values[miss] == '').mean():.1%}")
    print("================================================")

if __name__ == "__main__":
    main()
