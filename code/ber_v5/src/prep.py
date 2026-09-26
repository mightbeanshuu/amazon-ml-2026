"""One-time preprocessing: stream every source TSV in chunks, normalise in parallel, write Parquet per country.

    python -m src.prep --data <dataset dir> --out <prep dir> [--workers 8]

Output: <prep>/<split>/s{1,2,3}_<country>.parquet with the normalised columns used by blocking and features.
Only the provided files are read. Memory is bounded by chunk size x workers.
"""
import argparse
import csv
import os
import time
from multiprocessing import Pool

import pandas as pd

from .normalize import normalize_frame

KEEP = ["entity_id", "country_norm", "name_clean", "name_core", "name_legal", "name_trade", "name_phon", "name_ns",
        "addr_norm", "addr_core", "addr_landmark", "addr_alpha", "addr_nums", "postcode", "house_no"]


def _norm(chunk):
    return normalize_frame(chunk)[KEEP]


def run_file(path, out_dir, tag, workers, chunksize=100_000):
    t = time.time()
    reader = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
                         chunksize=chunksize)
    n, countries = 0, set()
    with Pool(workers) as pool:
        for k, df in enumerate(pool.imap(_norm, reader)):     # each chunk is written out immediately
            n += len(df)
            for c, g in df.groupby("country_norm"):
                d = os.path.join(out_dir, f"{tag}_{c or 'none'}")
                os.makedirs(d, exist_ok=True)
                g.to_parquet(os.path.join(d, f"part-{k:04d}.parquet"), compression="zstd", index=False)
                countries.add(c)
    print(f"{path}: {n} rows, countries {sorted(countries)} in {time.time() - t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--splits", default="train,test")
    a = ap.parse_args()
    for split in a.splits.split(","):
        d = os.path.join(a.out, split)
        os.makedirs(d, exist_ok=True)
        for i in (1, 2, 3):
            run_file(os.path.join(a.data, split, f"{split}_source{i}.tsv"), d, f"s{i}", a.workers)


if __name__ == "__main__":
    main()
