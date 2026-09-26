"""Scalable blocking for millions of records, one country at a time and one field at a time.

For each field (no-space name, address, name+locality) we fit char-3-gram TF-IDF on the split's records,
drop n-grams that occur in more than `max_df` of documents (they are uninformative and dominate the
sparse-product cost), and run top-k cosine for S1 chunks against all S2/S3 records (sparse_dot_topn,
multi-threaded). A reverse pass (each S2/S3 record's best S1s) covers S1 rows whose own top-k is crowded
by namesakes. Output: one pair table with the per-field cosine (NaN when that field did not propose the pair)
and within-S1 rank.
"""
import gc
import time

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

_NS_TAIL = ("com", "org", "net", "in", "co", "private", "pvt", "limited", "ltd", "llp", "llc", "inc", "corp")


def _ns_variants(ns):
    """'gdtradingcliniccom' -> also 'gdtradingclinic'; 'rewardprivatecom' -> 'rewardprivate' -> 'reward'."""
    out, t = [], ns[3:] if ns.startswith("www") else ns
    for _ in range(3):
        for tail in _NS_TAIL:
            if t.endswith(tail) and len(t) - len(tail) >= 4:
                t = t[:-len(tail)]
                out.append(t)
                break
        else:
            break
    return out


def _name_keys(core, trade, phon, ns):
    nt = core.split() + trade.split()
    k = ["n:" + t for t in nt]
    k += ["nb:" + a + "_" + b for a, b in zip(nt, nt[1:])]
    k += ["p:" + t for t in phon.split()]
    if len(ns) >= 4:
        k += ["ns:" + ns, "ns5:" + ns[:5], "nse:" + ns[-5:]]
        k += ["ns:" + v for v in _ns_variants(ns)]
    return k


def _cross_keys(phon, nums):
    """Phonetic name token x address number: rare even when the name is transliterated ('suprim sistams') and
    the address is only generic words plus a house number ('flat 901 raigad maharastra')."""
    pt = [t for t in phon.split() if len(t) >= 2][:6]
    nn = [x for x in nums.split()][:4]
    return ["x:" + t + "_" + x for t in pt for x in nn]


def _addr_keys(alpha, nums):
    at = alpha.split()
    k = ["a:" + t for t in at if len(t) >= 3]
    k += ["ab:" + a + "_" + b for a, b in zip(at, at[1:])]
    k += ["d:" + x for x in nums.split() if len(x) >= 2]
    return k


def _cols(d):
    return zip(d["name_core"].values, d["name_trade"].values, d["name_phon"].values, d["name_ns"].values,
               d["addr_alpha"].values, d["addr_nums"].values)


def name_keys(d):
    return pd.Series([" ".join(_name_keys(c, t, p, n)) for c, t, p, n, _, _ in _cols(d)], index=d.index)


def addr_keys(d):
    return pd.Series([" ".join(_addr_keys(a, x)) for _, _, _, _, a, x in _cols(d)], index=d.index)


def cross_keys(d):
    return pd.Series([" ".join(_cross_keys(p, x)) for _, _, p, _, _, x in _cols(d)], index=d.index)


def record_keys(d):
    """Name keys + address keys, space-joined. Word-level and rare, so posting lists stay short."""
    return pd.Series([" ".join(_name_keys(c, t, p, n) + _addr_keys(a, x)) for c, t, p, n, a, x in _cols(d)],
                     index=d.index)


KEY_FIELDS = {"b_key": (record_keys, "hashed_keys", 24, 4),      # combined name+address keys (main)
              "b_nkey": (name_keys, "hashed_keys", 8, 2),        # name-only: degraded/empty addresses
              "b_akey": (addr_keys, "hashed_keys", 8, 2),        # address-only: garbled/renamed names
              "b_xkey": (cross_keys, "hashed_keys", 8, 2)}       # phonetic name token x address number

_HV = HashingVectorizer(analyzer=str.split, n_features=2 ** 25, alternate_sign=False, norm=None, binary=True,
                        dtype=np.float32)


def _hashed(frame, build, chunk=400_000):
    """Binary hashed key matrix, built chunk by chunk so the key strings never all live in memory."""
    mats = []
    for s in range(0, len(frame), chunk):
        mats.append(_HV.transform(build(frame.iloc[s:s + chunk]).values))
    return sp.vstack(mats).tocsr()


def hashed_tfidf(s1, r, build, max_df_abs):
    """IDF-weighted, L2-normalised hashed keys. Keys in more than max_df_abs records, or in only one record,
    are dropped: they either cost too much or cannot link anything."""
    A, B = _hashed(s1, build), _hashed(r, build)
    df = np.bincount(A.indices, minlength=A.shape[1]) + np.bincount(B.indices, minlength=B.shape[1])
    n = A.shape[0] + B.shape[0]
    w = (np.log((n + 1) / (df + 1)) + 1).astype(np.float32)
    w[(df > max_df_abs) | (df < 2)] = 0
    out = []
    for M in (A, B):
        M.data = w[M.indices]
        M.eliminate_zeros()
        out.append(normalize(M, norm="l2", copy=False))
    return out[0], out[1], int((w > 0).sum())

FIELDS = {
    # name: (column builder, analyzer, k forward, k reverse)
    "b_name": (lambda d: d["name_ns"], "char", 20, 6),
    "b_addr": (lambda d: d["addr_norm"], "char_wb", 12, 4),
    "b_loc": (lambda d: d["name_core"] + " " + d["addr_alpha"], "char_wb", 20, 6),
}


def _topk_chunks(A, B, k, chunk=20000, threads=8):
    rows, cols, vals = [], [], []
    if A.nnz == 0 or B.nnz == 0:              # sparse_dot_topn can segfault on an all-empty operand
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.float32)
    Bt = B.T.tocsr()
    for s in range(0, A.shape[0], chunk):
        if A[s:s + chunk].nnz == 0:
            continue
        C = sp_matmul_topn(A[s:s + chunk], Bt, top_n=k, n_threads=threads).tocoo()
        rows.append(C.row + s)
        cols.append(C.col)
        vals.append(C.data.astype(np.float32))
    if not rows:
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def _hash_chunks(frame, build, chunk=400_000):
    for s in range(0, len(frame), chunk):
        yield s, _HV.transform(build(frame.iloc[s:s + chunk]).values)


def _key_weights(s1, r, build, max_df_abs):
    """Pass 1: document frequency of every hashed key, without keeping any matrix."""
    df = np.zeros(_HV.n_features, np.int64)
    n = 0
    for frame in (s1, r):
        for _, M in _hash_chunks(frame, build):
            df += np.bincount(M.indices, minlength=_HV.n_features)
            n += M.shape[0]
    w = (np.log((n + 1) / (df + 1)) + 1).astype(np.float32)
    w[(df > max_df_abs) | (df < 2)] = 0
    return w


def _weighted(frame, build, w):
    """Pass 2: IDF-weighted, L2-normalised hashed matrix built chunk by chunk (zero-weight keys dropped early)."""
    mats = []
    for _, M in _hash_chunks(frame, build):
        M.data = w[M.indices]
        M.eliminate_zeros()
        mats.append(normalize(M, norm="l2", copy=False))
    return sp.vstack(mats).tocsr() if mats else sp.csr_matrix((0, _HV.n_features), dtype=np.float32)


def block_country(s1, r, max_df=0.02, log=print, fields=None, kscale=1.0, max_df_keys=1000, reverse=True,
                  keep_s1=None, r_parts=3, s1_parts=4, post=None):
    """s1, r: normalised frames of ONE country (fresh RangeIndex). Hashed rare-key fields only.
    The S2/S3 side is processed in `r_parts` slices: forward top-k is taken per slice and merged into an exact
    global top-k per S1; the reverse search (each S2/S3 record's best S1s) is naturally per slice. Peak memory is
    about 1/r_parts of building the whole S2/S3 matrix. keep_s1: optional mask; pairs of other S1 rows are
    discarded as soon as they are found. Pairs are held as int64 keys s1_i * n_r + r_i."""
    fields = fields or KEY_FIELDS
    n_r = len(r)
    keys, vals, rstats = {}, {}, {}
    bounds = np.linspace(0, n_r, r_parts + 1).astype(int)
    for name, (build, _, kf, kr) in fields.items():
        t = time.time()
        w = _key_weights(s1, r, build, max_df_keys)
        A = _weighted(s1, build, w)
        kf_, kr_ = max(1, int(kf * kscale)), max(1, int(kr * kscale))
        fk, fv, rk_, rv = [], [], [], []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            B = _weighted(r.iloc[lo:hi], build, w)
            i, j, v = _topk_chunks(A, B, kf_)
            fk.append(i.astype(np.int64) * n_r + (j + lo))
            fv.append(v)
            if reverse:
                j2, i2, v2 = _topk_chunks(B, A, kr_)                  # rows of B are S2/S3 records
                rk_.append(i2.astype(np.int64) * n_r + (j2 + lo))
                rv.append(v2)
            del B
            gc.collect()
        fk, fv = np.concatenate(fk), np.concatenate(fv)
        if r_parts > 1:                                           # exact global top-k from per-slice top-k
            rank = group_rank((fk // n_r).astype(np.int64), fv)
            fk, fv = fk[rank <= kf_], fv[rank <= kf_]
        n_fwd = len(fk)
        k_ = np.concatenate([fk] + rk_) if rk_ else fk
        v_ = np.concatenate([fv] + rv) if rv else fv
        rstats[name] = _r_best_two(k_, v_, n_r)                      # over ALL S1, before any sampling
        if keep_s1 is not None:
            m = keep_s1[k_ // n_r]
            k_, v_ = k_[m], v_[m]
        keys[name], vals[name] = k_, v_
        log(f"  {name}: fwd {n_fwd}, rev {sum(len(x) for x in rk_)}, kept {len(k_)}, {time.time() - t:.0f}s")
        del A, w, fk, fv, rk_, rv
        gc.collect()
    # merge per S1 range: every rank is within-S1, so partitioning by S1 is exact and cuts peak memory
    n_s1 = len(s1)
    cuts = np.linspace(0, n_s1, s1_parts + 1).astype(np.int64)
    out = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        kl, kh = lo * n_r, hi * n_r
        sub = {nm: (keys[nm] >= kl) & (keys[nm] < kh) for nm in fields}
        uk = np.unique(np.concatenate([keys[nm][sub[nm]] for nm in fields]))
        df = pd.DataFrame({"s1_i": (uk // n_r).astype(np.int32), "r_i": (uk % n_r).astype(np.int32)})
        for nm in fields:
            col = np.full(len(uk), -1.0, np.float32)
            np.maximum.at(col, np.searchsorted(uk, keys[nm][sub[nm]]), vals[nm][sub[nm]])
            col[col < 0] = np.nan
            df[nm] = col
        del uk, sub
        g = df["s1_i"].values
        for nm in fields:
            df[f"{nm}_rk"] = group_rank(g, df[nm].values)
        best = df[list(fields)[0]].values.copy()
        for nm in list(fields)[1:]:
            best = np.fmax(best, df[nm].values)
        df["b_best"] = best
        ri = df["r_i"].values
        for nm in fields:                                         # R-side competition margins
            rmax, r2 = rstats[nm]
            v = df[nm].values
            other = np.where(v >= rmax[ri], r2[ri], rmax[ri])     # best score of any OTHER S1 for this record
            df[f"{nm}_rmarg"] = np.where(np.isnan(v), np.nan, v - np.where(other < 0, 0.0, other)).astype(np.float32)
        out.append(post(df) if post is not None else df)
        del df
        gc.collect()
    del keys, vals
    return pd.concat(out, ignore_index=True)


def _r_best_two(k, v, n_r):
    """Per S2/S3 record: the best and second-best blocking score over ALL S1 that proposed it (forward) or that it
    proposed (reverse). Two S1 tied at the top make the second-best equal to the best. Used for the margin
    feature "does another S1 fit this record better?", which a sampled-S1 training set could not see otherwise."""
    rr = (k % n_r).astype(np.int64)
    rmax = np.full(n_r, -1.0, np.float32)
    np.maximum.at(rmax, rr, v)
    top = v >= rmax[rr]
    r2 = np.full(n_r, -1.0, np.float32)
    np.maximum.at(r2, rr[~top], v[~top])
    n_top = np.bincount(np.unique(k[top]) % n_r, minlength=n_r)   # distinct S1 at the top (dedup fwd/rev copies)
    r2 = np.where(n_top > 1, rmax, r2)
    return rmax, r2


def group_rank(groups, vals):
    """Descending rank (1 = best, ties by position) of vals within each group; NaN stays NaN. Pure numpy, so
    far lighter than pandas groupby().rank() on tens of millions of rows."""
    v = np.where(np.isnan(vals), -np.inf, vals)
    order = np.lexsort((-v, groups))
    gs = groups[order]
    idx = np.arange(len(gs))
    start = np.maximum.accumulate(np.where(np.r_[True, gs[1:] != gs[:-1]], idx, 0))
    rk = np.empty(len(gs), np.float32)
    rk[order] = idx - start + 1
    rk[np.isnan(vals)] = np.nan
    return rk
