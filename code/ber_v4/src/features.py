"""Pair features for the matcher. All are language-agnostic string similarities or relative/rank signals.
There is no country one-hot, so unseen country labels (France) are handled by the same code path."""
import math
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein


CP_WORKERS = -1      # set to 1 inside worker processes (they are already parallel)


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=CP_WORKERS, dtype=np.float32)


def idf_table(*series):
    """Token IDF over all provided records of this split (S1+S2+S3): provided data only. Tokens seen in only one
    record are left out of the dict: lookups fall back to the default weight, which is effectively the same value,
    and this keeps the table (copied into every feature worker) small."""
    df = Counter()
    n = 0
    for s in series:
        for text in s:
            n += 1
            df.update(set(text.split()))
    default = math.log((n + 1) / 2) + 1.0
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items() if c > 1}, default


def _abbr_match(a, b):
    """a matches b if equal, a is a prefix of b (len >= 2), or a is an in-order subsequence of b
    that shares its first letter (ltd~limited, blvd~boulevard). No per-country lists needed."""
    if a == b:
        return True
    if len(a) >= 2 and len(a) < len(b) and a[0] == b[0]:
        if b.startswith(a):
            return True
        it = iter(b)
        return all(ch in it for ch in a)
    return False


def _pair_token_feats(t1, t2, idf, idf_max):
    s1, s2 = set(t1), set(t2)
    if not s1 or not s2:
        return (np.nan,) * 6
    inter = s1 & s2
    w = lambda s: sum(idf.get(t, idf_max) for t in s)
    wj = w(inter) / max(w(s1 | s2), 1e-6)
    wcov1 = w(inter) / max(w(s1), 1e-6)
    wcov2 = w(inter) / max(w(s2), 1e-6)
    # abbreviation-aware coverage of the shorter side by the longer
    short, long_ = (t1, t2) if len(t1) <= len(t2) else (t2, t1)
    abbr = sum(any(_abbr_match(a, b) or _abbr_match(b, a) for b in long_) for a in short) / len(short)
    max_idf_shared = max((idf.get(t, idf_max) for t in inter), default=0.0)
    rarest_missing = max((idf.get(t, idf_max) for t in (s1 ^ s2)), default=0.0)
    return wj, wcov1, wcov2, abbr, max_idf_shared, rarest_missing


def _acronym(t1, t2):
    if len(t1) >= 2 and len(t2) == 1:
        return float("".join(x[0] for x in t1) == t2[0])
    if len(t2) >= 2 and len(t1) == 1:
        return float("".join(x[0] for x in t2) == t1[0])
    return 0.0


def _tri(a, b):
    """1 = equal, 0 = conflict, NaN = at least one side missing. Missing is not a mismatch."""
    return np.where((a == "") | (b == ""), np.nan, (a == b).astype(np.float32)).astype(np.float32)


def build(pairs: pd.DataFrame, s1: pd.DataFrame, r: pd.DataFrame, idf_name, idf_addr, aligned=False):
    """pairs has s1_i, r_i and the blocking features. Returns a float32 feature frame aligned to pairs.
    aligned=True: s1 and r are already the per-pair rows (row k of s1/r belongs to pair k)."""
    if aligned:
        A, B = s1.reset_index(drop=True), r.reset_index(drop=True)
    else:
        A = s1.iloc[pairs.s1_i.values].reset_index(drop=True)
        B = r.iloc[pairs.r_i.values].reset_index(drop=True)
    F = {}
    na, nb = A.name_core.tolist(), B.name_core.tolist()
    F["n_ratio"] = _cp(na, nb, fuzz.ratio)
    F["n_partial"] = _cp(na, nb, fuzz.partial_ratio)
    F["n_tsort"] = _cp(na, nb, fuzz.token_sort_ratio)
    F["n_tset"] = _cp(na, nb, fuzz.token_set_ratio)
    F["n_jw"] = _cp(na, nb, JaroWinkler.normalized_similarity)
    F["n_lev"] = _cp(na, nb, Levenshtein.normalized_similarity)
    F["n_full_tset"] = _cp(A.name_clean.tolist(), B.name_clean.tolist(), fuzz.token_set_ratio)
    F["n_phon_tset"] = _cp(A.name_phon.tolist(), B.name_phon.tolist(), fuzz.token_set_ratio)
    F["n_phon_ratio"] = _cp(A.name_phon.tolist(), B.name_phon.tolist(), fuzz.ratio)
    F["n_ns_ratio"] = _cp(A.name_ns.tolist(), B.name_ns.tolist(), fuzz.ratio)          # '#pioneerfashion'
    F["n_ns_partial"] = _cp(A.name_ns.tolist(), B.name_ns.tolist(), fuzz.partial_ratio)
    # trade/DBA names: best of core-vs-trade combinations when either side has a trade part
    ta, tb = A.name_trade.tolist(), B.name_trade.tolist()
    alt = np.maximum.reduce([_cp([x or "\x00" for x in ta], nb, fuzz.token_set_ratio),
                             _cp(na, [x or "\x00" for x in tb], fuzz.token_set_ratio),
                             _cp([x or "\x00" for x in ta], [x or "\x01" for x in tb], fuzz.token_set_ratio)])
    F["n_trade_best"] = np.maximum(alt, F["n_tset"])
    F["has_trade"] = ((A.name_trade != "") | (B.name_trade != "")).values.astype(np.float32)
    aa, ab = A.addr_norm.tolist(), B.addr_norm.tolist()
    F["a_ratio"] = _cp(aa, ab, fuzz.ratio)
    F["a_tset"] = _cp(aa, ab, fuzz.token_set_ratio)
    F["a_partial"] = _cp(aa, ab, fuzz.partial_ratio)
    F["a_alpha_tset"] = _cp(A.addr_alpha.tolist(), B.addr_alpha.tolist(), fuzz.token_set_ratio)
    F["a_core_tset"] = _cp(A.addr_core.tolist(), B.addr_core.tolist(), fuzz.token_set_ratio)
    la, lb = A.addr_landmark.values, B.addr_landmark.values
    F["lm_a"] = (la != "").astype(np.float32)
    F["lm_b"] = (lb != "").astype(np.float32)
    F["lm_tset"] = np.where((la == "") | (lb == ""), np.nan,
                            _cp([x or "\x00" for x in la], [x or "\x01" for x in lb], fuzz.token_set_ratio))
    # landmark text of one side found anywhere in the other side's address
    F["lm_in_other"] = np.maximum(_cp([x or "\x00" for x in la], ab, fuzz.partial_ratio),
                                  _cp([x or "\x00" for x in lb], aa, fuzz.partial_ratio))
    F["addr_short_a"] = A.addr_core.str.split().str.len().values.astype(np.float32)
    F["addr_short_b"] = B.addr_core.str.split().str.len().values.astype(np.float32)
    pa, pb = A.postcode.values, B.postcode.values
    F["pc_eq"] = _tri(pa, pb)
    F["pc_pre3"] = _tri(np.array([x[:3] for x in pa]), np.array([x[:3] for x in pb]))
    F["pc_pre2"] = _tri(np.array([x[:2] for x in pa]), np.array([x[:2] for x in pb]))
    F["house_eq"] = _tri(A.house_no.values, B.house_no.values)
    # truncated house numbers are a noise pattern (707 -> 7, 759 -> 75): one side a strict prefix of the other
    F["house_prefix"] = np.array([np.nan if (not x or not y) else float(x != y and (x.startswith(y) or y.startswith(x)))
                                  for x, y in zip(A.house_no.values, B.house_no.values)], np.float32)
    # number-set Jaccard
    nums_a = A.addr_nums.str.split().tolist()
    nums_b = B.addr_nums.str.split().tolist()
    F["num_jacc"] = np.array([len(set(x) & set(y)) / len(set(x) | set(y)) if (x and y) else np.nan
                              for x, y in zip(nums_a, nums_b)], np.float32)
    # IDF-weighted token features
    idf_n, idf_n_max = idf_name
    idf_a, idf_a_max = idf_addr
    tn = [_pair_token_feats(x.split(), y.split(), idf_n, idf_n_max) for x, y in zip(na, nb)]
    tn = np.array(tn, np.float32)
    for j, k in enumerate(["n_wjacc", "n_wcov1", "n_wcov2", "n_abbr", "n_maxidf_shared", "n_rarest_diff"]):
        F[k] = tn[:, j]
    ta_ = [_pair_token_feats(x.split(), y.split(), idf_a, idf_a_max) for x, y in zip(aa, ab)]
    ta_ = np.array(ta_, np.float32)
    for j, k in enumerate(["a_wjacc", "a_wcov1", "a_wcov2", "a_abbr", "a_maxidf_shared", "a_rarest_diff"]):
        F[k] = ta_[:, j]
    F["n_first_eq"] = np.array([float(x.split()[:1] == y.split()[:1]) if x and y else np.nan
                                for x, y in zip(na, nb)], np.float32)
    F["n_acronym"] = np.array([_acronym(x.split(), y.split()) for x, y in zip(na, nb)], np.float32)
    F["n_len_a"] = np.array([len(x) for x in na], np.float32)
    F["n_len_b"] = np.array([len(x) for x in nb], np.float32)
    F["n_len_ratio"] = np.minimum(F["n_len_a"], F["n_len_b"]) / np.maximum(np.maximum(F["n_len_a"], F["n_len_b"]), 1)
    F["n_exact"] = (A.name_core.values == B.name_core.values).astype(np.float32)
    F["n_rarity_a"] = np.array([np.mean([idf_n.get(t, idf_n_max) for t in x.split()]) if x else np.nan
                                for x in na], np.float32)
    lga, lgb = A.name_legal.values, B.name_legal.values
    F["legal_tri"] = np.array([np.nan if (not x or not y) else float(bool(set(x.split("|")) & set(y.split("|"))))
                               for x, y in zip(lga, lgb)], np.float32)
    F["is_s3"] = B.entity_id.str.startswith("S3").values.astype(np.float32)
    feats = pd.DataFrame(F)
    block_cols = [c for c in pairs.columns if c not in ("s1_i", "r_i")]
    feats = pd.concat([feats, pairs[block_cols].reset_index(drop=True)], axis=1).astype(np.float32)
    # S1-level competition signals (all candidates of an S1 are in the same chunk)
    key1 = pairs.s1_i.values
    tmp = pd.DataFrame({"k1": key1, "n": F["n_tset"], "a": F["a_tset"]})
    feats["n_same_addr_for_s1"] = tmp.assign(x=tmp.a >= 90).groupby("k1")["x"].transform("sum").values
    feats["n_tset_gap1"] = (tmp.groupby("k1")["n"].transform("max") - tmp.n).values
    feats["a_tset_gap1"] = (tmp.groupby("k1")["a"].transform("max") - tmp.a).values
    return feats.astype(np.float32)


def add_r_level(df):
    """R-level competition signals, computed GLOBALLY over all pairs (an S2/S3 record's S1 candidates can
    span chunks). df needs s1_i, r_i, n_tset, a_tset."""
    g = df.groupby("r_i")
    df["n_similar_names_for_r"] = df.assign(_x=df.n_tset >= 90).groupby("r_i")["_x"].transform("sum").astype(np.float32)
    df["n_tset_gap2"] = (g["n_tset"].transform("max") - df.n_tset).astype(np.float32)
    df["a_tset_gap2"] = (g["a_tset"].transform("max") - df.a_tset).astype(np.float32)
    df["n_s1_for_r"] = g["s1_i"].transform("size").astype(np.float32)
    return df


# ---- parallel builder: workers receive only their chunk's aligned rows (no copy of the big tables) --------
NEEDED = ["entity_id", "name_clean", "name_core", "name_legal", "name_trade", "name_phon", "addr_norm",
          "addr_core", "addr_landmark", "addr_nums", "postcode", "house_no"]


def _derive(d):
    """Columns that are cheap to rebuild in the worker instead of holding them for millions of rows."""
    d = d.reset_index(drop=True)
    d["name_ns"] = d["name_core"].str.replace(" ", "", regex=False)
    d["addr_alpha"] = [" ".join(t for t in x.split() if not t.isdigit()) for x in d["addr_core"].values]
    return d
_IDF = {}


def _init(idf_n, idf_a):
    global CP_WORKERS
    CP_WORKERS = 1
    _IDF["n"], _IDF["a"] = idf_n, idf_a


def _work(task):
    P, A, B = task
    return build(P, _derive(A), _derive(B), _IDF["n"], _IDF["a"], aligned=True)


class FeaturePool:
    """A worker pool created once (IDF tables sent once per worker) and reused for many chunks."""

    def __init__(self, idf_n, idf_a, workers=6):
        from multiprocessing import get_context
        self.workers = workers
        self.pool = get_context("spawn").Pool(workers, initializer=_init, initargs=(idf_n, idf_a),
                                              maxtasksperchild=8)     # recycle workers: bounded memory

    def build(self, pairs, s1, r, chunk=120_000):
        """pairs must be sorted by s1_i so each chunk holds complete S1 groups. Dispatch in bounded waves."""
        pairs = pairs.reset_index(drop=True)
        s1v = pairs.s1_i.values
        cuts = [0]
        while cuts[-1] < len(pairs):
            nxt = min(cuts[-1] + chunk, len(pairs))
            while nxt < len(pairs) and s1v[nxt] == s1v[nxt - 1]:
                nxt += 1
            cuts.append(nxt)
        bounds = list(zip(cuts[:-1], cuts[1:]))
        s1n, rn = s1[NEEDED], r[NEEDED]
        out = []
        for w in range(0, len(bounds), self.workers):
            tasks = []
            for lo, hi in bounds[w:w + self.workers]:
                P = pairs.iloc[lo:hi]
                tasks.append((P, s1n.iloc[P.s1_i.values], rn.iloc[P.r_i.values]))
            out.extend(self.pool.map(_work, tasks))
            del tasks
        return pd.concat(out, ignore_index=True)

    def close(self):
        self.pool.close()
        self.pool.join()


def build_parallel(pairs, s1, r, idf_n, idf_a, workers=6, chunk=120_000):
    fp = FeaturePool(idf_n, idf_a, workers)
    try:
        return fp.build(pairs, s1, r, chunk)
    finally:
        fp.close()
