"""B-IN on the EXACT v10 miss set: Ayan's reverse combo view (record -> top-K visible S1 by
0.5*cos(char_wb-3 name TF-IDF) + 0.5*cos(word address TF-IDF), max_df .05, sublinear) — same construction as
SCR/bin_recall.py, on the pipeline-normalised columns. Reports rank of the true S1 for missed + control pairs."""
import glob, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

AU = "/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad/autopsy"
parts = lambda tag: sorted(glob.glob(f"{AU}/norm/{tag}/part-*.parquet"))
t0 = time.time()
C2 = ["name_core", "addr_norm"]
S = pd.concat([pd.read_parquet(p, columns=C2) for p in parts("s1")], ignore_index=True)
rs = np.random.RandomState(0)
visible = np.sort(np.flatnonzero(rs.rand(len(S)) >= 0.19))
S = S.iloc[visible].reset_index(drop=True)
rng = np.random.RandomState(2)
fit, Rq = [], []
M = pd.read_parquet(f"{AU}/missed.parquet")
C = pd.read_parquet(f"{AU}/control.parquet")
need = pd.Index(np.unique(np.r_[M.r_i.values, C.r_i.values]))
off = 0
for p in parts("s2") + parts("s3"):
    d = pd.read_parquet(p, columns=C2)
    fit.append(d[rng.rand(len(d)) < 0.15])
    gi = np.arange(off, off + len(d))
    m = need.get_indexer(gi) >= 0
    x = d[m].copy(); x["_g"] = gi[m]; Rq.append(x)
    off += len(d)
F = pd.concat(fit, ignore_index=True)
Rq = pd.concat(Rq, ignore_index=True).set_index("_g")
print(f"visible S1 {len(S)} fitpool {len(F)} queries {len(Rq)} {time.time() - t0:.0f}s", flush=True)
vn = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=0.05, sublinear_tf=True, dtype=np.float32)
va = TfidfVectorizer(analyzer="word", token_pattern=r"[a-z0-9]+", min_df=2, max_df=0.05, sublinear_tf=True,
                     dtype=np.float32)
vn.fit(pd.concat([S.name_core, F.name_core])); va.fit(pd.concat([S.addr_norm, F.addr_norm]))
del F
w = np.float32(np.sqrt(0.5))
combo = lambda d: sp.hstack([vn.transform(d.name_core) * w, va.transform(d.addr_norm) * w]).tocsr()
St = combo(S).T.tocsr()
del S
out = {}
for name, T in (("missed", M), ("control", C)):
    Q = combo(Rq.loc[T.r_i.values])
    K = 100
    hits = np.full(len(T), 10 ** 6)
    own = T.s1_vis_i.values
    for i in range(0, len(T), 4000):
        Mx = sp_matmul_topn(Q[i:i + 4000], St, top_n=K, n_threads=4).tocsr()
        for j in range(Mx.shape[0]):
            a, b = Mx.indptr[j], Mx.indptr[j + 1]
            cols, vals = Mx.indices[a:b], Mx.data[a:b]
            o = np.argsort(-vals, kind="stable")
            r = np.flatnonzero(cols[o] == own[i + j])
            if len(r):
                hits[i + j] = r[0] + 1
    out[name] = hits
    print(name, {k: round(float(np.mean(hits <= k)), 4) for k in (1, 3, 8, 20, 40, 100)}, f"{time.time() - t0:.0f}s",
          flush=True)
M["ayan_rank"] = out["missed"]
M[["s1_id", "r_id", "ayan_rank"]].to_parquet(f"{AU}/ayan_rank_missed.parquet", index=False)
