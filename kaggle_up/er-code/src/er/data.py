"""Loading + cached normalization of the three sources."""
import csv, os, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from .normalize import clean_name, clean_address, skeleton

READ_KW = dict(sep="\t", dtype=str, keep_default_na=False, na_filter=False,
               quoting=csv.QUOTE_NONE, engine="c")

def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)

def read_tsv(path):
    return pd.read_csv(path, **READ_KW)

def _norm_chunk(args):
    names, addrs = args
    nm = [clean_name(x) for x in names]
    sk = [skeleton(x) for x in nm]
    ad = [clean_address(x) for x in addrs]
    return nm, sk, ad

def normalize_frame(df, n_proc=4, chunk=50_000):
    parts = [(df.business_name.values[i:i + chunk], df.business_address.values[i:i + chunk])
             for i in range(0, len(df), chunk)]
    with Pool(n_proc) as p:
        out = p.map(_norm_chunk, parts)
    df = df.copy()
    df["nm"] = [x for o in out for x in o[0]]
    df["sk"] = [x for o in out for x in o[1]]
    df["ad"] = [x for o in out for x in o[2]]
    df["na"] = df["nm"] + " " + df["ad"]
    return df

def load_split(data_dir, split, work_dir, n_proc=4):
    """Returns (s1, s23, gt_or_None). s23 = S2 and S3 concatenated. Normalized columns cached as parquet."""
    cache = os.path.join(work_dir, f"norm_{split}.parquet")
    if os.path.exists(cache):
        log(f"loading cached normalized data {cache}")
        allrec = pd.read_parquet(cache)
    else:
        frames = []
        for s in (1, 2, 3):
            p = os.path.join(data_dir, split, f"{split}_source{s}.tsv")
            t = time.time(); d = read_tsv(p); d["src"] = np.int8(s)
            log(f"read {p}: {len(d):,} rows ({time.time()-t:.0f}s)")
            frames.append(d)
        allrec = pd.concat(frames, ignore_index=True)
        t = time.time()
        allrec = normalize_frame(allrec, n_proc=n_proc)
        log(f"normalized {len(allrec):,} rows ({time.time()-t:.0f}s)")
        allrec.to_parquet(cache, index=False)
    s1 = allrec[allrec.src == 1].reset_index(drop=True)
    s23 = allrec[allrec.src != 1].reset_index(drop=True)
    gt = None
    gp = os.path.join(data_dir, split, f"{split}_ground_truth.tsv")
    if os.path.exists(gp):
        gt = read_tsv(gp)
    return s1, s23, gt

def truth_index(gt, s1, s23):
    """For each s23 row -> index of its true S1 row (or -1)."""
    s1_pos = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
    m = gt.matched_entity_ids.str.split(",")
    pairs = pd.DataFrame({"s1": gt.source1_entity_id.repeat(m.str.len()), "m": [x for l in m for x in l]})
    pairs = pairs[pairs.m != ""]
    owner = pd.Series(s1_pos.reindex(pairs.s1.values).values, index=pairs.m.values)
    return owner.reindex(s23.entity_id.values).fillna(-1).astype(np.int64).values
