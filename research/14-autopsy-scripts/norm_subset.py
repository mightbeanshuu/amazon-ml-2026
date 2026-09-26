"""Normalise ONLY the India rows the at-risk simulation needs (suitor S1s by visible index, at-risk records by R
index), exactly as norm_india.py does, keeping name_core / name_phon / addr_norm. Writes sub_s1.parquet
(s1_vis_i) and sub_r.parquet (r_i)."""
import csv, json, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/mac/amazon-ml-2026/code/ber_v5")
from src.normalize import normalize_frame, renormalize_names, phon_line, clean

AU = "/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad/autopsy"
REAL = AU + "/../real"
LEX = json.load(open("/Users/mac/amazon-ml-2026/code/ber_v5/src/lexicon.json"))
C3 = ["name_core", "name_phon", "addr_norm"]
_cc = {}


def india(df):
    cn = df["country"].map(lambda c: _cc.setdefault(c, clean(c) if isinstance(c, str) else ""))
    return df[(cn == "india").values]


def norm(df):
    df = normalize_frame(df)
    renormalize_names(df, LEX)
    df["name_phon"] = df["name_core"].map(phon_line)
    return df[C3 + ["_g"]]


def run(files, need):
    """need: sorted array of India-row indices (in file order across `files`) to keep. Streams: each chunk is
    filtered and normalised at once, only the 3 output columns are kept (single process, memory-light)."""
    off, chunks = 0, []
    for fn in files:
        for ch in pd.read_csv(f"{REAL}/{fn}", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
                              chunksize=50_000):
            ch = india(ch)
            gi = np.arange(off, off + len(ch))
            off += len(ch)
            m = np.isin(gi, need)
            if m.any():
                x = ch[m].copy()
                x["_g"] = gi[m]
                chunks.append(norm(x))
            del ch
    return pd.concat(chunks, ignore_index=True), off


if __name__ == "__main__":
    t = time.time()
    S = pd.concat([pd.read_parquet(f"{AU}/atrisk_suitors_{f}.parquet", columns=["r_i", "s1_vis_i"])
                   for f in ["b_key", "b_nkey", "b_akey", "b_xkey", "b_cgram", "b_pkey"]])
    AR = pd.read_parquet(f"{AU}/atrisk.parquet")
    rs = np.random.RandomState(0)
    vis = np.flatnonzero(rs.rand(883188) >= 0.19)
    s_need_vis = np.unique(np.concatenate([S.s1_vis_i.values, AR.s1_vis_i.values]))
    s1, n1 = run(["train_source1.tsv"], np.sort(vis[s_need_vis]))
    assert n1 == 883188, n1
    s1["s1_vis_i"] = np.searchsorted(vis, s1._g.values)
    assert (vis[s1.s1_vis_i.values] == s1._g.values).all()
    s1.drop(columns="_g").to_parquet(f"{AU}/sub_s1.parquet", index=False)
    r, nr = run(["train_source2.tsv", "train_source3.tsv"], np.unique(AR.r_i.values))
    assert nr == 4133346, nr
    r.rename(columns={"_g": "r_i"}).to_parquet(f"{AU}/sub_r.parquet", index=False)
    print(f"s1 {len(s1)} r {len(r)} in {time.time() - t:.0f}s", flush=True)
