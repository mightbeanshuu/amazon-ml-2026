"""Multi-pass sparse TF-IDF candidate generation.

Direction: every S2/S3 record queries the S1 records of the same country and keeps its
top-k S1 neighbours per pass. (Each S2/S3 record belongs to at most one S1, so this is
the natural direction; the per-S1 candidate lists are the inverse of these edges.)
Common n-grams (document frequency > max_df) are dropped: they carry little TF-IDF weight
but dominate compute.
Licenses: scikit-learn (BSD-3), sparse_dot_topn (Apache-2.0), scipy/numpy (BSD).
"""
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from .data import log

# name, field, analyzer, ngram_range, max_df
PASSES = [
    ("na_w",  "na", "word",    (1, 1), 0.02),
    ("nm_w",  "nm", "word",    (1, 1), 0.02),
    ("sk_w",  "sk", "word",    (1, 1), 0.02),
    ("ad_w",  "ad", "word",    (1, 2), 0.01),
    ("sk_c3", "sk", "char_wb", (3, 3), 0.02),
]

def run_pass(s1, s23, q_idx, spec, k, n_threads, chunk=200_000):
    """Returns DataFrame(q, s1, score, rank) with global row indices into s23 / s1."""
    name, field, analyzer, ngram, max_df = spec
    out = []
    for c in sorted(set(s1.country.unique()) & set(s23.country.unique())):
        a_idx = np.flatnonzero(s1.country.values == c)
        b_all = np.flatnonzero(s23.country.values == c)
        b_idx = np.intersect1d(b_all, q_idx, assume_unique=True)
        if len(a_idx) == 0 or len(b_idx) == 0:
            continue
        t = time.time()
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, max_df=max_df, min_df=1,
                              sublinear_tf=True, dtype=np.float32, lowercase=False)
        vec.fit(pd.concat([s1[field].iloc[a_idx], s23[field].iloc[b_all]], ignore_index=True))
        A_T = vec.transform(s1[field].iloc[a_idx]).T.tocsr()
        t_fit = time.time() - t; t = time.time()
        for i in range(0, len(b_idx), chunk):
            bi = b_idx[i:i + chunk]
            B = vec.transform(s23[field].iloc[bi])
            C = sp_matmul_topn(B, A_T, top_n=k, threshold=0.0, sort=True, n_threads=n_threads)
            rows = np.repeat(np.arange(C.shape[0]), np.diff(C.indptr))
            rank = np.arange(C.nnz) - C.indptr[rows]
            out.append(pd.DataFrame({"q": bi[rows].astype(np.int32), "s1": a_idx[C.indices].astype(np.int32),
                                     "score": C.data.astype(np.float32), "rank": rank.astype(np.int8)}))
        log(f"  pass {name:6} country={c:8} S1={len(a_idx):>9,} queries={len(b_idx):>9,} "
            f"vocab={len(vec.vocabulary_):>9,} fit {t_fit:5.0f}s  search {time.time()-t:5.0f}s")
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["q", "s1", "score", "rank"])
    df["pass"] = name
    return df

def union(edges, k):
    """Union of per-pass top-k edges -> unique (q, s1) pairs."""
    e = edges[edges["rank"] < k]
    return e[["q", "s1"]].drop_duplicates()
