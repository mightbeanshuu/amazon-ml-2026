"""pprot positives (prune rank > 28, kept by sampled rev<=2): how many S1-side pairs ranked above them are
K-protected (b_rank==40 & K-rank>40, i.e. likely cut at test before the prune)? Adjusted rank at test."""
import glob, sys
import numpy as np, pandas as pd, pyarrow as pa
sys.path.insert(0, "/Users/mac/amazon-ml-2026/code/ber_v5")
from src.block_big import group_rank
from rapidfuzz import fuzz, process
C3 = ["name_core", "name_phon", "addr_norm"]
parts = lambda tag: sorted(glob.glob(f"norm/{tag}/part-*.parquet"))
FAMS = ["b_key", "b_nkey", "b_akey", "b_xkey", "b_cgram", "b_pkey"]
PP = pd.read_parquet("prune_pos.parquet", columns=["s1_i", "r_i", "rk", "keep", "score"])
pp = PP[PP.keep & (PP.rk > 28)]
S1set = sorted(set(pp.s1_i))
P = pd.read_parquet("box/train_block_india.parquet", columns=["s1_i", "r_i", "y", "b_rank"] + FAMS, filters=[("s1_i", "in", S1set)])
P = P.sort_values(["s1_i", "b_rank"]).reset_index(drop=True)
alt = np.fmax.reduce([P[c].values for c in FAMS[1:]])
fam = P[FAMS].notna().any(axis=1).values
ks = np.where(np.isnan(P.b_key.values), np.nan_to_num(alt) * 0.5, P.b_key.values).astype(np.float32)
# K-rank within these S1s' retained family pairs (S1-side lists are complete in the table up to rank 40)
krk = group_rank(P.s1_i.values, np.where(fam, ks, np.nan).astype(np.float32))
P["kprot_pair"] = fam & (krk > 40)
rs = np.random.RandomState(0)
n_s1 = sum(pd.read_parquet(p, columns=["name_core"]).shape[0] for p in parts("s1"))
vis = np.flatnonzero(rs.rand(n_s1) >= 0.19); samp = np.sort(rs.choice(len(vis), 120000, replace=False))
S1 = pd.concat([pd.read_parquet(p, columns=C3) for p in parts("s1")], ignore_index=True).iloc[vis[samp]].reset_index(drop=True)
need = pd.Index(np.unique(P.r_i.values)); Rs = []; off = 0
for p in parts("s2") + parts("s3"):
    d = pd.read_parquet(p, columns=C3); gi = np.arange(off, off + len(d)); off += len(d)
    m = need.get_indexer(gi) >= 0
    if m.any():
        x = d[m].copy(); x["_g"] = gi[m]; Rs.append(x)
Rs = pd.concat(Rs).set_index("_g")
A = {c: S1[c].values[P.s1_i.values].tolist() for c in C3}
B = {c: Rs[c].reindex(P.r_i.values).tolist() for c in C3}
cp = lambda x, y, sc: process.cpdist(x, y, scorer=sc, workers=4, dtype=np.float32)
nm = np.maximum.reduce([cp(A["name_core"], B["name_core"], fuzz.token_set_ratio),
                        cp([a.replace(" ", "") for a in A["name_core"]], [b.replace(" ", "") for b in B["name_core"]], fuzz.ratio),
                        cp(A["name_phon"], B["name_phon"], fuzz.token_set_ratio)])
P["score"] = nm + cp(A["addr_norm"], B["addr_norm"], fuzz.token_set_ratio) + 50 * np.nan_to_num(P.b_key.values)
X = pp.merge(P[["s1_i", "score", "kprot_pair"]].rename(columns={"score": "sc2"}), on="s1_i")
X = X[X.sc2 > X.score + 1e-4]
ab = X.groupby(["s1_i", "r_i"]).agg(n_above=("sc2", "size"), n_above_kprot=("kprot_pair", "sum")).reset_index()
pp = pp.merge(ab, on=["s1_i", "r_i"], how="left").fillna({"n_above": 0, "n_above_kprot": 0})
print("pprot positives", len(pp), "| check rk-1 == n_above:", round(float(np.mean(pp.rk - 1 == pp.n_above)), 3))
for cut in (0.4, 0.6, 0.8, 1.0):
    adj = pp.rk - cut * pp.n_above_kprot
    print(f"if a share {cut} of K-protected pairs is cut at test: pprot positives back within rank 28: {int((adj <= 28).sum())} ({(adj <= 28).mean():.3f})")
pp.to_parquet("pprot_refine.parquet", index=False)
