"""Candidate generation (blocking).

A union of top-k searches over char-3-gram TF-IDF:
  B1 name core, B2 full address, B3 name+address, B4 reverse B3 (each S2/S3 record's top S1).
Top-k keeps recall at a fixed output size, which a threshold cannot do (Sparkly, PVLDB'23).
IDF is fitted on the provided records of the split being processed, so no external data is used.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


def _vec(texts_fit, analyzer="char_wb", ngram=(3, 3)):
    v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, sublinear_tf=True, min_df=1, dtype=np.float32)
    v.fit(texts_fit)
    return v


def _topk(A, B, k, n_threads=8):
    """Top-k cosine of each row of A against the rows of B (both L2-normalised). Returns COO triples."""
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.array([], int), np.array([], int), np.array([], np.float32)
    C = sp_matmul_topn(A, B.T.tocsr(), top_n=min(k, B.shape[0]), n_threads=n_threads).tocoo()
    return C.row, C.col, C.data


def rowwise_cos(A, B, ia, ib):
    """Cosine for aligned pairs (A[ia], B[ib]). Both matrices are L2-normalised."""
    out = np.zeros(len(ia), np.float32)
    step = 200_000
    for s in range(0, len(ia), step):
        a, b = A[ia[s:s + step]], B[ib[s:s + step]]
        out[s:s + step] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


FIELDS = {"cos_name": "name_core", "cos_addr": "addr_norm", "cos_comb": "name_addr", "cos_loc": "name_loc",
          "cos_phon": "name_phon"}


def build_matrices(s1, r):
    mats = {}
    for key, col in FIELDS.items():
        v = _vec(pd.concat([s1[col], r[col]]).values)
        mats[key] = (v.transform(s1[col].values), v.transform(r[col].values))
    return mats


def generate(s1, r, k_name=15, k_addr=10, k_comb=20, k_rev=5, partition_by_country=True, mats=None, k_loc=15):
    """s1 and r are normalised frames with a fresh RangeIndex; r is S2 and S3 concatenated.
    Returns a DataFrame with s1_i and r_i (row positions), the TF-IDF cosines and blocking ranks."""
    mats = mats or build_matrices(s1, r)
    pieces = []
    if partition_by_country:
        groups = [(np.flatnonzero(s1["country_norm"].values == c), np.flatnonzero(r["country_norm"].values == c))
                  for c in s1["country_norm"].unique()]
    else:
        groups = [(np.arange(len(s1)), np.arange(len(r)))]
    for i1, i2 in groups:
        if len(i1) == 0 or len(i2) == 0:
            continue
        for key, k in (("cos_name", k_name), ("cos_addr", k_addr), ("cos_comb", k_comb), ("cos_loc", k_loc)):
            if k <= 0:
                continue
            A, B = mats[key]
            ro, co, _ = _topk(A[i1], B[i2], k)
            pieces.append(np.stack([i1[ro], i2[co]], 1))
        for key in ("cos_comb", "cos_loc"):                # reverse direction: each S2/S3 record's best S1s
            A, B = mats[key]
            ro, co, _ = _topk(B[i2], A[i1], k_rev)
            pieces.append(np.stack([i1[co], i2[ro]], 1))
    if not pieces:
        return pd.DataFrame(columns=["s1_i", "r_i"])
    pairs = np.unique(np.concatenate(pieces), axis=0)
    df = pd.DataFrame({"s1_i": pairs[:, 0], "r_i": pairs[:, 1]})
    for key, (A, B) in mats.items():
        df[key] = rowwise_cos(A, B, df.s1_i.values, df.r_i.values)
    add_rank_features(df, RANK_COLS)
    return df


RANK_COLS = ["cos_name", "cos_addr", "cos_comb", "cos_loc"]


def add_rank_features(df, cols):
    """Rank of each candidate within its S1 list and within its S2/S3 record's S1 list,
    plus the margin to the best candidate. These are relative, so they are robust to
    shifts in overall similarity level (e.g. an unseen country)."""
    for c in cols:
        g1 = df.groupby("s1_i")[c]
        df[f"{c}_rk1"] = g1.rank(ascending=False, method="min").astype(np.float32)
        df[f"{c}_gap1"] = (g1.transform("max") - df[c]).astype(np.float32)
        g2 = df.groupby("r_i")[c]
        df[f"{c}_rk2"] = g2.rank(ascending=False, method="min").astype(np.float32)
        df[f"{c}_gap2"] = (g2.transform("max") - df[c]).astype(np.float32)
    df["n_cand_s1"] = df.groupby("s1_i")["r_i"].transform("size").astype(np.float32)
    df["n_cand_r"] = df.groupby("r_i")["s1_i"].transform("size").astype(np.float32)
    return df
