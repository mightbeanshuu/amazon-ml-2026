"""Normalise every India train record exactly as prep.py + run_big.load() do (normalize_frame, then the lexicon
renormalisation, then name_phon / name_ns recomputed), preserving prep order: S1 in file order; R = S2 rows in file
order then S3 rows in file order. Writes BLOCK_COLS parquet parts. 3 workers, 100k-row chunks (memory-light)."""
import csv, json, os, sys, time
from multiprocessing import Pool
import pandas as pd
sys.path.insert(0, "/Users/mac/amazon-ml-2026/code/ber_v5")
from src.normalize import normalize_frame, renormalize_names, phon_line, clean

REAL = "/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad/real"
OUT = "/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad/autopsy/norm"
LEX = json.load(open("/Users/mac/amazon-ml-2026/code/ber_v5/src/lexicon.json"))
COLS = ["entity_id", "country_norm", "name_clean", "name_core", "name_trade", "name_phon", "name_ns",
        "addr_alpha", "addr_nums", "addr_norm", "addr_core"]
_cc = {}


def work(df):
    cn = df["country"].map(lambda c: _cc.setdefault(c, clean(c) if isinstance(c, str) else ""))
    df = df[(cn == "india").values].copy()
    if not len(df):
        return None
    df = normalize_frame(df)[COLS]
    renormalize_names(df, LEX)
    df["name_phon"] = df["name_core"].map(phon_line)
    df["name_ns"] = df["name_core"].str.replace(" ", "", regex=False)
    return df


def run(fname, tag):
    os.makedirs(f"{OUT}/{tag}", exist_ok=True)
    rd = pd.read_csv(f"{REAL}/{fname}", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
                     chunksize=100_000)
    n, t = 0, time.time()
    with Pool(3) as pool:
        for k, df in enumerate(pool.imap(work, rd)):
            if df is None:
                continue
            df.to_parquet(f"{OUT}/{tag}/part-{k:04d}.parquet", compression="zstd", index=False)
            n += len(df)
    print(f"{fname}: {n} india rows in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    run("train_source1.tsv", "s1")
    run("train_source2.tsv", "s2")
    run("train_source3.tsv", "s3")
