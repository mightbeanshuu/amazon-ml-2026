"""Per-family mechanical diagnosis of v10 India blocking misses at FULL distractor density.

For every missed pair (s, r) and every KEY_FIELDS family f (exact v10 functions, weights, max_df):
  cos      = cosine of s and r in family f (0 = no surviving shared key)
  fwd_rank = 1 + #R records r' with cos_f(s, r') > cos (over all 4.13M India S2/S3)      -> retrieved if <= kf
  rev_rank = 1 + #visible S1 s' with cos_f(s', r) > cos (over all 715k visible India S1) -> retrieved if <= kr
  shared-key accounting: shared raw keys, shared keys alive, shared keys killed by max_df (df > cap)
Also run on 3000 RETRIEVED positives as a control (must come out retrieved; cos must equal the box's b_* value).
Streams parquet parts; never holds the corpus in memory.  usage: fam_diag.py [family ...]"""
import gc, glob, json, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, "/Users/mac/amazon-ml-2026/code/ber_v5")
from src.block_big import KEY_FIELDS, _HV, group_rank
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

AU = "/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad/autopsy"
COLS = ["name_core", "name_trade", "name_phon", "name_ns", "addr_alpha", "addr_nums"]
T = 160          # per-chunk top-n kept when counting better competitors (saturation flagged)
NT = 4
parts = lambda tag: sorted(glob.glob(f"{AU}/norm/{tag}/part-*.parquet"))
S1P, RP = parts("s1"), parts("s2") + parts("s3")
n_s1 = sum(pd.read_parquet(p, columns=["name_core"]).shape[0] for p in S1P)
rs = np.random.RandomState(0)
vis_mask = rs.rand(n_s1) >= 0.19
visible = np.flatnonzero(vis_mask)


def _grouped(gen, rows=300_000):
    buf, ib, n = [], [], 0
    for d, gi in gen:
        buf.append(d); ib.append(gi); n += len(d)
        if n >= rows:
            yield pd.concat(buf, ignore_index=True), np.concatenate(ib)
            buf, ib, n = [], [], 0
    if buf:
        yield pd.concat(buf, ignore_index=True), np.concatenate(ib)


def _s1_parts():
    off = 0
    for p in S1P:
        d = pd.read_parquet(p, columns=COLS)
        m = vis_mask[off:off + len(d)]
        vi = np.searchsorted(visible, np.arange(off, off + len(d))[m])
        off += len(d)
        yield d[m].reset_index(drop=True), vi


def _r_parts():
    off = 0
    for p in RP:
        d = pd.read_parquet(p, columns=COLS)
        yield d, np.arange(off, off + len(d))
        off += len(d)


def iter_s1():
    """Yields (frame of VISIBLE S1 rows, their visible indices), ~300k rows at a time."""
    return _grouped(_s1_parts())


def iter_r():
    return _grouped(_r_parts())


def raw(frame, build):
    return _HV.transform(build(frame).values).tocsr()


def weigh(M, w):
    M = M.copy()
    M.data = w[M.indices]
    M.eliminate_zeros()
    return normalize(M, norm="l2", copy=False)


def gather(idx_sorted_unique, it):
    """Rows (by global index) of the S1-visible or R corpus, as a frame in the order of idx_sorted_unique."""
    want = pd.Index(idx_sorted_unique)
    out = []
    for d, gi in it():
        m = want.get_indexer(gi) >= 0
        if m.any():
            x = d[m].copy()
            x["_g"] = gi[m]
            out.append(x)
    f = pd.concat(out, ignore_index=True).set_index("_g").loc[idx_sorted_unique].reset_index(drop=True)
    return f


def main():
    fams = sys.argv[1:] or list(KEY_FIELDS)
    M = pd.read_parquet(f"{AU}/missed.parquet")
    C = pd.read_parquet(f"{AU}/control.parquet")
    Q = pd.concat([M[["s1_vis_i", "r_i"]].assign(ctl=0), C[["s1_vis_i", "r_i"]].assign(ctl=1)], ignore_index=True)
    us, ps = np.unique(Q.s1_vis_i.values, return_inverse=True)
    ur, pr = np.unique(Q.r_i.values, return_inverse=True)
    t0 = time.time()
    # at-risk positives: kept in train ONLY thanks to a record-side protection computed among SAMPLED S1s
    PP = pd.read_parquet(f"{AU}/prune_pos.parquet", columns=["s1_i", "r_i", "b_rank", "rk", "keep"])
    rs2 = np.random.RandomState(0)
    vis2 = np.flatnonzero(rs2.rand(n_s1) >= 0.19)
    samp = np.sort(rs2.choice(len(vis2), 120000, replace=False))
    PP["s1_vis_i"] = samp[PP.s1_i.values]
    AR = PP[(PP.b_rank == 40) | (PP.keep & (PP.rk > 28))].reset_index(drop=True)
    AR["kprot"] = (AR.b_rank == 40).values
    AR["pprot"] = (AR.keep & (AR.rk > 28)).values
    AR.to_parquet(f"{AU}/atrisk.parquet", index=False)
    uar, par = np.unique(AR.r_i.values, return_inverse=True)
    uas, pas = np.unique(AR.s1_vis_i.values, return_inverse=True)
    FA, FAS = gather(uar, iter_r), gather(uas, iter_s1)
    print(f"at-risk positives {len(AR)} (records {len(uar)})", flush=True)
    FS = gather(us, iter_s1)
    FR = gather(ur, iter_r)
    print(f"queries {len(Q)} (S1 {len(us)}, R {len(ur)}) gathered {time.time() - t0:.0f}s", flush=True)
    for f in fams:
        build, _, kf, kr, mdf = KEY_FIELDS[f]
        cap = mdf or 3000
        t = time.time()
        df = np.zeros(_HV.n_features, np.int32)
        n = 0
        for it in (iter_s1, iter_r):
            for d, _ in it():
                X = raw(d, build)
                u, cnt = np.unique(X.indices, return_counts=True)
                df[u] += cnt.astype(np.int32)
                n += X.shape[0]
                del X, u, cnt
        w = np.log(np.float32(n + 1) / (df.astype(np.float32) + 1)) + np.float32(1)
        w[(df > cap) | (df < 2)] = 0
        print(f"[{f}] df pass n={n} alive keys {(w > 0).sum()} {time.time() - t:.0f}s", flush=True)
        RSr, RRr = raw(FS, build), raw(FR, build)
        QS, QR = weigh(RSr, w), weigh(RRr, w)
        cos = np.asarray(QS[ps].multiply(QR[pr]).sum(1)).ravel().astype(np.float32)
        sh = RSr[ps].multiply(RRr[pr]).tocsr()                      # shared raw keys per pair
        shi = sh.indices
        row = np.repeat(np.arange(sh.shape[0]), np.diff(sh.indptr))
        n_sh = np.bincount(row, minlength=len(Q))
        n_alive = np.bincount(row, weights=(w[shi] > 0), minlength=len(Q)).astype(int)
        n_maxdf = np.bincount(row, weights=(df[shi] > cap), minlength=len(Q)).astype(int)
        mind = np.full(len(Q), -1, np.int64)
        if len(shi):
            o = np.lexsort((df[shi], row))
            first = np.r_[True, row[o][1:] != row[o][:-1]]
            mind[row[o][first]] = df[shi][o][first]
        s_nnz, r_nnz = np.diff(QS.indptr)[ps], np.diff(QR.indptr)[pr]
        QA, QAS = weigh(raw(FA, build), w), weigh(raw(FAS, build), w)
        cosA = np.asarray(QAS[pas].multiply(QA[par]).sum(1)).ravel().astype(np.float32)
        pd.DataFrame({"r_i": AR.r_i.values, "s1_vis_i": AR.s1_vis_i.values, "cos": cosA}).to_parquet(
            f"{AU}/atrisk_true_{f}.parquet", index=False)
        qcols = np.union1d(np.union1d(QS.indices, QR.indices), QA.indices)

        def compact(X):
            """Keep only the query-side columns (dot products unchanged; X is already L2-normalised)."""
            pos = np.searchsorted(qcols, X.indices)
            pos[pos >= len(qcols)] = 0
            ok = qcols[pos] == X.indices
            rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
            ip = np.r_[0, np.cumsum(np.bincount(rows[ok], minlength=X.shape[0]))]
            return sp.csr_matrix((X.data[ok], pos[ok].astype(np.int32), ip), shape=(X.shape[0], len(qcols)))
        QSc, QRc, QAc = compact(QS), compact(QR), compact(QA)
        a_r, a_s, a_v = [], [], []
        s_raw, r_raw = np.diff(RSr.indptr)[ps], np.diff(RRr.indptr)[pr]
        # competitor counts
        fg, fge, fsat = np.zeros(len(Q), int), np.zeros(len(Q), int), np.zeros(len(Q), bool)
        rg, rge, rsat = np.zeros(len(Q), int), np.zeros(len(Q), int), np.zeros(len(Q), bool)
        rows_of_s = pd.Series(np.arange(len(Q))).groupby(ps).apply(np.array).to_dict()
        rows_of_r = pd.Series(np.arange(len(Q))).groupby(pr).apply(np.array).to_dict()

        def count(Cm, rows_of, g, ge, sat):
            Cm = Cm.tocsr()
            for a, pp in rows_of.items():
                seg = Cm.data[Cm.indptr[a]:Cm.indptr[a + 1]]
                if not len(seg):
                    continue
                for p in pp:
                    c = cos[p]
                    g[p] += int((seg > c + 1e-6).sum())
                    ge[p] += int((seg >= c - 1e-6).sum())
                    if len(seg) >= T and seg.min() > c + 1e-6:
                        sat[p] = True
        for d, gi in iter_r():
            B = compact(weigh(raw(d, build), w))
            if B.nnz:
                count(sp_matmul_topn(QSc, B.T.tocsr(), top_n=T, n_threads=NT), rows_of_s, fg, fge, fsat)
            del B
        for d, gi in iter_s1():
            A = compact(weigh(raw(d, build), w))
            if A.nnz:
                At = A.T.tocsr()
                count(sp_matmul_topn(QRc, At, top_n=T, n_threads=NT), rows_of_r, rg, rge, rsat)
                CA = sp_matmul_topn(QAc, At, top_n=30, n_threads=NT).tocoo()
                a_r.append(CA.row.astype(np.int32)); a_s.append(gi[CA.col].astype(np.int32)); a_v.append(CA.data.astype(np.float32))
                del At, CA
            del A
        gc.collect()
        if a_r:
            ar_, as_, av_ = np.concatenate(a_r), np.concatenate(a_s), np.concatenate(a_v)
            rkA = group_rank(ar_, av_)
            m_ = rkA <= 30
            pd.DataFrame({"r_i": uar[ar_[m_]], "s1_vis_i": as_[m_], "cos": av_[m_], "rrk": rkA[m_]}).to_parquet(
                f"{AU}/atrisk_suitors_{f}.parquet", index=False)
        live = cos > 0
        out = pd.DataFrame({"cos": cos, "fwd_rank": np.where(live, fg + 1, 10 ** 6),
                            "fwd_rank_ge": np.where(live, fge, 10 ** 6), "fwd_sat": fsat,
                            "rev_rank": np.where(live, rg + 1, 10 ** 6), "rev_rank_ge": np.where(live, rge, 10 ** 6),
                            "rev_sat": rsat, "n_sh": n_sh, "n_alive": n_alive, "n_maxdf": n_maxdf, "min_df": mind,
                            "s_nnz": s_nnz, "r_nnz": r_nnz, "s_raw": s_raw, "r_raw": r_raw, "ctl": Q.ctl.values})
        out["in_f"] = live & ((out.fwd_rank <= kf) | (out.rev_rank <= kr))
        out["in_f_ge"] = live & ((out.fwd_rank_ge <= kf) | (out.rev_rank_ge <= kr))
        out.to_parquet(f"{AU}/diag_{f}.parquet", index=False)
        c = out[out.ctl == 1]
        box = C[f].values
        agree = np.nanmax(np.abs(np.nan_to_num(box, nan=0) - c.cos.values)) if len(c) else -1
        bp = np.isfinite(box)
        print(f"[{f}] control crosstab box-proposed x my in_f: "
              f"{pd.crosstab(bp, c.in_f.values).to_dict()} | in_f_ge {pd.crosstab(bp, c.in_f_ge.values).to_dict()}", flush=True)
        print(f"[{f}] done {time.time() - t:.0f}s | miss: cos>0 {live[Q.ctl.values == 0].mean():.3f} "
              f"in_f {out.in_f[out.ctl == 0].mean():.4f} | control: in_f {c.in_f.mean():.4f} "
              f"(box has pair {np.isfinite(box).mean():.4f}); |cos-box| max over box-present "
              f"{np.nanmax(np.abs(box - c.cos.values)):.2e}", flush=True)
        del df, w, RSr, RRr, QS, QR, sh
        gc.collect()


if __name__ == "__main__":
    main()
