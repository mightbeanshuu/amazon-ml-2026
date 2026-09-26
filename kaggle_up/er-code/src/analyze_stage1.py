#!/usr/bin/env python3
"""Analyse saved Stage 1 edges: what each pass adds, cheaper pass combinations, and a sample
of missed pairs for inspection. No new search — reads edges_*.parquet and the cached data.

    python analyze_stage1.py --data-dir DATASET_DIR --edges /kaggle/working/er_work/edges_train_q0.05_k10.parquet
Writes misses_sample.tsv next to the edges file (upload that file back).
"""
import argparse, itertools, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er.data import load_split, truth_index, log

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--work-dir", default="/kaggle/working/er_work")
    ap.add_argument("--n-miss", type=int, default=3000)
    a = ap.parse_args()

    s1, s23, gt = load_split(a.data_dir, "train", a.work_dir)
    owner = truth_index(gt, s1, s23)
    E = pd.read_parquet(a.edges)
    E["hit"] = owner[E.q.values] == E.s1.values
    q_all = np.unique(E.q.values)                        # queried records (with >=1 edge)
    n_true = int((owner[q_all] >= 0).sum())
    passes = sorted(E["pass"].unique())
    log(f"edges={len(E):,} queried={len(q_all):,} with-true-S1={n_true:,}")

    # per-pass hit sets at each k (set of q that found its true S1)
    hits = {(p, k): set(E.q.values[(E["pass"] == p) & (E["rank"] < k) & E.hit]) for p in passes for k in (1, 3, 5, 10)}
    npairs = {(p, k): E[(E["pass"] == p) & (E["rank"] < k)][["q", "s1"]] for p in passes for k in (1, 3, 5, 10)}

    def evaluate(combo):   # combo: list of (pass, k)
        h = set().union(*[hits[c] for c in combo])
        n = len(pd.concat([npairs[c] for c in combo]).drop_duplicates())
        return len(h) / n_true, n / len(q_all)

    print("\n================ STAGE 1 ANALYSIS ================")
    print("Marginal recall: what is lost if ONE pass is removed (all others at the same k)")
    for k in (3, 5, 10):
        full, _ = evaluate([(p, k) for p in passes])
        line = f"  k={k:<2} all={full:.4f} |"
        for p in passes:
            r, _ = evaluate([(q, k) for q in passes if q != p])
            line += f" -{p}: {full - r:+.4f}"
        print(line)

    print("\nCombinations (pass@k) -> recall, candidates per S2/S3 record")
    combos = []
    base = ["na_w", "ad_w"]
    extras = [p for p in passes if p not in base]
    for kb in (5, 10):
        for ke in (0, 1, 3, 5):
            for r in range(0, len(extras) + 1):
                for ex in itertools.combinations(extras, r):
                    if ke == 0 and ex: continue
                    if ke > 0 and not ex: continue
                    combos.append([(b, kb) for b in base] + [(e, ke) for e in ex])
    rows = []
    for c in combos:
        r, per = evaluate(c)
        rows.append((" + ".join(f"{p}@{k}" for p, k in c), r, per))
    R = pd.DataFrame(rows, columns=["combo", "recall", "cands"]).sort_values("cands")
    # keep the Pareto frontier: best recall for its cost
    R = R[R.recall.cummax() == R.recall].drop_duplicates("recall")
    for x in R.itertuples():
        print(f"  {x.recall:.4f}  {x.cands:5.1f}  {x.combo}")

    # rank of the true S1 inside na_w when it is found by na_w
    t = E[(E["pass"] == "na_w") & E.hit]
    print(f"\n  na_w true-S1 rank distribution: {t['rank'].value_counts().sort_index().to_dict()}")

    # duplicate query strings (cheap speed-up potential)
    for f in ("na", "ad", "nm"):
        u = s23[["country", f]].drop_duplicates().shape[0]
        print(f"  unique (country,{f}) strings among S2+S3: {u/len(s23):.1%}")

    # sample of misses (not found by any pass at k=10)
    found = set().union(*[hits[(p, 10)] for p in passes])
    miss = np.array([q for q in q_all if owner[q] >= 0 and q not in found])
    rng = np.random.default_rng(0)
    ms = rng.choice(miss, size=min(a.n_miss, len(miss)), replace=False)
    top = E[(E["pass"] == "na_w") & (E["rank"] == 0)].set_index("q")
    out = pd.DataFrame({
        "q_name": s23.business_name.values[ms], "q_addr": s23.business_address.values[ms], "country": s23.country.values[ms],
        "true_name": s1.business_name.values[owner[ms]], "true_addr": s1.business_address.values[owner[ms]],
    })
    tq = top.reindex(ms)
    ok = tq.s1.notna().values
    out["na_top1_name"] = ""; out["na_top1_addr"] = ""; out["na_top1_score"] = np.nan
    out.loc[ok, "na_top1_name"] = s1.business_name.values[tq.s1.values[ok].astype(int)]
    out.loc[ok, "na_top1_addr"] = s1.business_address.values[tq.s1.values[ok].astype(int)]
    out.loc[ok, "na_top1_score"] = tq.score.values[ok]
    p = os.path.join(os.path.dirname(a.edges), "misses_sample.tsv")
    out.to_csv(p, sep="\t", index=False)
    print(f"\n  misses at k=10: {len(miss):,} ({len(miss)/n_true:.2%}); wrote {len(out)} examples to {p}")
    print("==================================================")

if __name__ == "__main__":
    main()
