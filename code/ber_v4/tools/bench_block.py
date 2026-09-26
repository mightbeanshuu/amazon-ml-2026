"""Benchmark scalable blocking on a train S1 sample against the FULL S2/S3 of one country.
Usage: python tools/bench_block.py <prep_dir> <dataset_dir> <country> <n_s1_sample> [max_df] [kscale]"""
import csv
import sys
import time

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src.block_big import block_country, KEY_FIELDS
from src.metrics import blocking_report, parse_ids

prep, data, country, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
max_df = float(sys.argv[5]) if len(sys.argv) > 5 else 0.02
kscale = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
t = time.time()
s1 = pd.read_parquet(f"{prep}/train/s1_{country}").sample(n, random_state=0).reset_index(drop=True)
r = pd.concat([pd.read_parquet(f"{prep}/train/s2_{country}"), pd.read_parquet(f"{prep}/train/s3_{country}")],
              ignore_index=True)
gt = pd.read_csv(f"{data}/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False,
                 quoting=csv.QUOTE_NONE)
gt = gt[gt.source1_entity_id.isin(set(s1.entity_id))]
gold = {e: parse_ids(v) for e, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
print(f"loaded s1 {len(s1)} r {len(r)} in {time.time() - t:.0f}s", flush=True)
t = time.time()
pairs = block_country(s1, r, fields=KEY_FIELDS, kscale=kscale, max_df_keys=max_df, reverse=False)
print(f"blocking {time.time() - t:.0f}s, pairs {len(pairs)}", flush=True)
a, b = s1.entity_id.values[pairs.s1_i.values], r.entity_id.values[pairs.r_i.values]
pairs["y"] = [y in gold[x] for x, y in zip(a, b)]
cand = {}
for x, y in zip(a, b):
    cand.setdefault(x, set()).add(y)
print("UNION", blocking_report(cand, gold, len(r)))
for f in ["b_nkey", "b_akey", "b_key"]:
    sub = pairs[pairs[f].notna()]
    c = {}
    for x, y in zip(s1.entity_id.values[sub.s1_i.values], r.entity_id.values[sub.r_i.values]):
        c.setdefault(x, set()).add(y)
    print(f, blocking_report(c, gold, len(r)))
# recall if we keep only the top-K per S1 by the best field cosine (what a cheap pruner would do at minimum)
pairs["rk"] = pairs.groupby("s1_i")["b_best"].rank(ascending=False, method="first")
for K in (8, 12, 16, 24):
    sub = pairs[pairs.rk <= K]
    c = {}
    for x, y in zip(s1.entity_id.values[sub.s1_i.values], r.entity_id.values[sub.r_i.values]):
        c.setdefault(x, set()).add(y)
    print(f"top{K} by b_best", blocking_report(c, gold, len(r)))
pairs.drop(columns=["y"]).assign(y=pairs.y).to_parquet(f"/Users/mac/amazon-ml-2026/eda/bench_pairs_{country}.parquet")
