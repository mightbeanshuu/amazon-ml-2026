"""Siblings: the 3-4 corrupted copies of one business inside S2/S3.

The Kaggle-track error analysis showed 96% of records the S1 search misses have a sibling that found the right
S1. Two uses here:
  1) candidate EXPANSION at blocking time: a record with only weak S1 hits inherits its strongest sibling's top
     S1 candidates (tagged with the sibling cosine).
  2) VOUCH features at scoring time: for a pair (s1, r), the best sibling of r that also has s1 as a candidate
     contributes sib_sim * p_sibling -> "my sibling points at this S1 too".

The sibling graph is a hashed rare-key top-k self-search of R against R (same machinery as S1 blocking), with
self-hits removed.
"""
import gc
import time

import numpy as np
import pandas as pd

from .block_big import KEY_FIELDS, _key_weights, _topk_chunks, _weighted, group_rank


def sibling_edges(r, k=3, max_df_keys=3000, log=print, chunk_rows=1_500_000):
    """Top-k sibling edges within one country's S2/S3 records, on the combined name+address keys.
    Returns (q_i, s_i, sim) with q_i != s_i, both indices into r."""
    build = KEY_FIELDS["b_key"][0]
    t = time.time()
    w = _key_weights(r, r.iloc[0:0], build, max_df_keys)          # df over R only (passing r twice would double df)
    n_r = len(r)
    kk = k + 1                                                    # a record's best hit is itself
    parts = max(1, (n_r + 1_200_000 - 1) // 1_200_000)            # index side sliced: bounded transpose memory
    bounds = np.linspace(0, n_r, parts + 1).astype(int)
    qs, ss, vs = [], [], []
    for lo in range(0, n_r, chunk_rows):
        A = _weighted(r.iloc[lo:lo + chunk_rows], build, w)
        nq = A.shape[0]
        # running top-kk per query row, merged after every index slice (memory stays ~nq * 2kk floats)
        bv = np.full((nq, kk), -1.0, np.float32)
        bi = np.full((nq, kk), -1, np.int64)
        for blo, bhi in zip(bounds[:-1], bounds[1:]):
            B = _weighted(r.iloc[blo:bhi], build, w)
            i, j, v = _topk_chunks(A, B, kk)
            del B
            nv = np.full((nq, kk), -1.0, np.float32)
            ni = np.full((nq, kk), -1, np.int64)
            order = np.lexsort((-v, i))
            i_s, j_s, v_s = i[order], j[order], v[order]
            first = np.r_[True, i_s[1:] != i_s[:-1]]
            start = np.maximum.accumulate(np.where(first, np.arange(len(i_s)), 0))
            col = np.arange(len(i_s)) - start
            keep = col < kk
            nv[i_s[keep], col[keep]] = v_s[keep]
            ni[i_s[keep], col[keep]] = j_s[keep] + blo
            both_v = np.hstack([bv, nv])
            both_i = np.hstack([bi, ni])
            sel = np.argsort(-both_v, axis=1)[:, :kk]
            bv = np.take_along_axis(both_v, sel, axis=1)
            bi = np.take_along_axis(both_i, sel, axis=1)
            del nv, ni, both_v, both_i, i, j, v
            gc.collect()
        rows = np.repeat(np.arange(nq) + lo, kk)
        flat_i, flat_v = bi.ravel(), bv.ravel()
        keep = (flat_i >= 0) & (flat_i != rows)
        qs.append(rows[keep])
        ss.append(flat_i[keep])
        vs.append(flat_v[keep])
        del A, bv, bi
        gc.collect()
    del w
    q, s2, v = np.concatenate(qs), np.concatenate(ss), np.concatenate(vs)
    rk = group_rank(q, v)
    m = rk <= k
    log(f"  siblings: {m.sum()} edges for {len(r)} records ({m.sum() / len(r):.2f} each), {time.time() - t:.0f}s")
    return q[m].astype(np.int32), s2[m].astype(np.int32), v[m].astype(np.float32)


def two_hop(sib, n_r, min_sim=0.45, cap=3):
    """Compose the sibling graph with itself: q->s and s->t gives q->t with sim = sim1*sim2. A record two hops
    from a well-linked sibling still descends from the same S1 (a peer measured candidate ceiling 0.978->0.993
    from exactly this). Keeps the top `cap` composed edges per record, direct edges preferred."""
    q, s2, v = sib
    e = pd.DataFrame({"q": q, "s": s2, "v": v})
    j = e.merge(e.rename(columns={"q": "s", "s": "t", "v": "v2"}), on="s")
    j = j[j.q != j.t]
    j["sim"] = (j.v * j.v2).astype(np.float32)
    j = j[j.sim >= min_sim]
    j = j.sort_values("sim", ascending=False).drop_duplicates(["q", "t"])
    both = pd.concat([e.rename(columns={"s": "t", "v": "sim"})[["q", "t", "sim"]], j[["q", "t", "sim"]]],
                     ignore_index=True).sort_values("sim", ascending=False).drop_duplicates(["q", "t"])
    rk = group_rank(both.q.values, both.sim.values.astype(np.float32))
    both = both[rk <= cap]
    return both.q.values.astype(np.int32), both.t.values.astype(np.int32), both.sim.values.astype(np.float32)


def expand_pairs(pairs, sib, n_r, top_s1=5, min_sim=0.55, weak_q=0.30):
    """Candidate expansion. A record is WEAK when its best b_key cosine over all its pairs is under the weak_q
    quantile (or it has no pair at all). For each weak record with a sibling of sim >= min_sim, add the sibling's
    top-s1 candidates (by b_key) as new pairs carrying sib_sim. Returns the extra pairs frame."""
    q, s2, v = sib
    best = np.full(n_r, -1.0, np.float32)
    np.maximum.at(best, pairs.r_i.values, np.nan_to_num(pairs.b_key.values, nan=-1.0))
    thr = np.quantile(best[best >= 0], weak_q)
    weak = (best < thr)                                           # includes never-proposed records (best -1)
    em = weak[q] & (v >= min_sim) & ~weak[s2]                     # weak record, strong sibling
    if not em.any():
        return None
    # sibling's top S1 candidates
    good = pairs[group_rank(pairs.r_i.values, np.nan_to_num(pairs.b_key.values, nan=-1.0)) <= top_s1]
    don = pd.DataFrame({"s_i": good.r_i.values, "s1_i": good.s1_i.values})
    ext = pd.DataFrame({"r_i": q[em], "s_i": s2[em], "sib_sim": v[em]}).merge(don, on="s_i")
    ext = ext.sort_values("sib_sim", ascending=False).drop_duplicates(["r_i", "s1_i"])
    return ext[["s1_i", "r_i", "sib_sim"]].reset_index(drop=True)


def vouch_features(s1_i, r_i, p, sib, n_r, chunk=4_000_000):
    """For every pair (s1, r): the strongest 'my sibling also points at this S1' signal.
    vouch = max over siblings s of r that share the pair (s1, s): sib_sim(r, s) * p(s1, s).
    Plus the number of siblings with p > 0.5 on this S1 and the record's best sibling sim overall.
    Memory-light: sibling p is found with searchsorted on the sorted pair key, chunk by chunk."""
    q, s2, v = sib
    key = s1_i.astype(np.int64) * n_r + r_i
    order = np.argsort(key, kind="stable")
    key_sorted, p_sorted = key[order], p[order]

    # up to 3 sibling slots per record, aligned arrays
    rk = group_rank(q, v)
    slots = []
    for slot in (1, 2, 3):
        m = rk == slot
        sid = np.full(n_r, -1, np.int64)
        ssim = np.zeros(n_r, np.float32)
        sid[q[m]] = s2[m]
        ssim[q[m]] = v[m]
        slots.append((sid, ssim))
    vouch = np.zeros(len(key), np.float32)
    n_vouch = np.zeros(len(key), np.float32)
    for lo in range(0, len(key), chunk):
        s1c, rc = s1_i[lo:lo + chunk].astype(np.int64), r_i[lo:lo + chunk]
        for sid, ssim in slots:
            sib_id = sid[rc]
            ok = sib_id >= 0
            kk = s1c * n_r + np.where(ok, sib_id, 0)
            pos = np.searchsorted(key_sorted, kk)
            hit = ok & (pos < len(key_sorted)) & (key_sorted[np.minimum(pos, len(key_sorted) - 1)] == kk)
            ps = np.where(hit, p_sorted[np.minimum(pos, len(key_sorted) - 1)], 0.0).astype(np.float32)
            np.maximum(vouch[lo:lo + chunk], ssim[rc] * ps, out=vouch[lo:lo + chunk])
            n_vouch[lo:lo + chunk] += (ps > 0.5)
    sib_best = np.zeros(n_r, np.float32)
    np.maximum.at(sib_best, q, v)
    return vouch, n_vouch, sib_best[r_i].astype(np.float32)
