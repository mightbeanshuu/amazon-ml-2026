"""Balanced name/address key blocking experiments on a train S1 sample vs full country R."""
import csv, sys, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from src.block_big import hashed_tfidf, name_keys, addr_keys, _topk_chunks
from src.metrics import blocking_report, parse_ids
prep, data, country, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
s1 = pd.read_parquet(f"{prep}/train/s1_{country}").sample(n, random_state=0).reset_index(drop=True)
r = pd.concat([pd.read_parquet(f"{prep}/train/s2_{country}"), pd.read_parquet(f"{prep}/train/s3_{country}")], ignore_index=True)
gt = pd.read_csv(f"{data}/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
gt = gt[gt.source1_entity_id.isin(set(s1.entity_id))]
gold = {e: parse_ids(v) for e, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
for mdf in [int(x) for x in sys.argv[5].split(",")]:
    t = time.time()
    An, Bn, _ = hashed_tfidf(s1, r, name_keys, mdf)
    Aa, Ba, _ = hashed_tfidf(s1, r, addr_keys, mdf)
    for wn in [0.5, 0.6]:
        A = (An * np.sqrt(wn) + Aa * np.sqrt(1 - wn)).tocsr()
        B = (Bn * np.sqrt(wn) + Ba * np.sqrt(1 - wn)).tocsr()
        i, j, v = _topk_chunks(A, B, 40)
        df = pd.DataFrame({"s1_i": i, "r_i": j, "v": v})
        df["rk"] = df.groupby("s1_i")["v"].rank(ascending=False, method="first")
        for K in (12, 16, 24, 32, 40):
            sub = df[df.rk <= K]
            c = {}
            for x, y in zip(s1.entity_id.values[sub.s1_i.values], r.entity_id.values[sub.r_i.values]):
                c.setdefault(x, set()).add(y)
            rep = blocking_report(c, gold, len(r))
            print(f"max_df {mdf} w_name {wn} top{K}: PC {rep['PC']} oracle {rep['oracle_macroF05']}", flush=True)
    print(f"  ({time.time()-t:.0f}s)", flush=True)
