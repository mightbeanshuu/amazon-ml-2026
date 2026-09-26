"""Dump OOF error pairs (missed matches and false merges) with their normalised text, one country at a time.

    python -m tools.oof_examples --run runs/real_v2 --prep prep --out runs/real_v2/oof_examples.tsv
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.run_big import countries, load, load_r  # noqa: E402

COLS = ["entity_id", "name_clean", "addr_norm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    e = pd.read_parquet(f"{a.run}/oof_errors.parquet")
    parts, r_off = [], 0
    for c in countries(a.prep, "train"):
        s1 = load(a.prep, "train", "s1", c, COLS).set_index("entity_id")
        r = load_r(a.prep, "train", c, COLS)
        ec = e[e.country == c].copy()
        loc = ec.r_i.values - r_off
        ec["r_id"] = r.entity_id.values[loc]
        ec["r_name"] = r.name_clean.values[loc]
        ec["r_addr"] = r.addr_norm.values[loc]
        ec["s1_name"] = s1.name_clean.reindex(ec.s1_id).values
        ec["s1_addr"] = s1.addr_norm.reindex(ec.s1_id).values
        r_off += len(r)
        parts.append(ec)
        del s1, r
    out = pd.concat(parts, ignore_index=True)
    out["kind"] = np.where(out.y, "MISS", "FALSE")
    out = out[["kind", "country", "p", "s1_id", "s1_name", "r_name", "s1_addr", "r_addr", "r_id"]]
    out.to_csv(a.out, sep="\t", index=False)
    print(out.groupby(["kind", "country"]).size())


if __name__ == "__main__":
    main()
