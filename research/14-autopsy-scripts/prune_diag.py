"""Exact replay of run_big.prune_by_similarity (prune_k=28, rev_rank<=2) on the box's real v10 India candidate
pairs (9.83M, 120k S1), then knob sweeps: what each alternative keeps (positives) and costs (pairs/S1)."""
import glob, sys, time, json
import numpy as np, pandas as pd, pyarrow as pa
sys.path.insert(0, "/Users/mac/amazon-ml-2026/code/ber_v5")
from src.block_big import group_rank
from rapidfuzz import fuzz, process

AU = "/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad/autopsy"
C3 = ["name_core", "name_phon", "addr_norm"]
parts = lambda tag: sorted(glob.glob(f"{AU}/norm/{tag}/part-*.parquet"))
t0 = time.time()
S1 = pd.concat([pd.read_parquet(p, columns=C3 + ["entity_id"]) for p in parts("s1")], ignore_index=True)
rs = np.random.RandomState(0)
visible = np.sort(np.flatnonzero(rs.rand(len(S1)) >= 0.19))
samp = np.sort(rs.choice(len(visible), 120000, replace=False))
S1 = S1.iloc[visible[samp]].reset_index(drop=True)
R = {c: pa.chunked_array([pa.array(pd.read_parquet(p, columns=[c])[c].values, pa.large_string())
                          for p in parts("s2") + parts("s3")]).combine_chunks() for c in C3}
print(f"loaded S1 {len(S1)} R {len(R['name_core'])} {time.time() - t0:.0f}s", flush=True)
cols = ["s1_i", "r_i", "y", "b_key", "b_rank", "b_key_rk", "b_nkey_rk", "b_akey_rk", "b_xkey_rk", "b_cgram_rk",
        "b_pkey_rk", "sib_sim", "b_cgram", "b_pkey", "b_nkey", "b_akey", "b_xkey"]
P = pd.read_parquet(f"{AU}/box/train_block_india.parquet", columns=cols)
P = P.sort_values(["s1_i", "b_rank"]).reset_index(drop=True)
si, ri = P.s1_i.values, P.r_i.values
n = len(P)
nm = np.empty(n, np.float32); ad = np.empty(n, np.float32)
cp = lambda x, y, sc: process.cpdist(x, y, scorer=sc, workers=4, dtype=np.float32)
CH = 1_000_000
for lo in range(0, n, CH):
    a, b = si[lo:lo + CH], ri[lo:lo + CH]
    A = {c: S1[c].values[a].tolist() for c in C3}
    tb = pa.array(b)
    B = {c: R[c].take(tb).to_pylist() for c in C3}
    ns_a = [x.replace(" ", "") for x in A["name_core"]]
    ns_b = [x.replace(" ", "") for x in B["name_core"]]
    nm[lo:lo + CH] = np.maximum.reduce([cp(A["name_core"], B["name_core"], fuzz.token_set_ratio),
                                        cp(ns_a, ns_b, fuzz.ratio),
                                        cp(A["name_phon"], B["name_phon"], fuzz.token_set_ratio)])
    ad[lo:lo + CH] = cp(A["addr_norm"], B["addr_norm"], fuzz.token_set_ratio)
    print(f"  scored {lo + len(a)} {time.time() - t0:.0f}s", flush=True)
score = nm + ad + 50 * np.nan_to_num(P.b_key.values)
rk = group_rank(si, score)
rev = group_rank(ri, score)
keep = (rk <= 28) | (rev <= 2)
y = P.y.values
gold = pd.read_parquet(f"{AU}/gold_sizes.parquet")
json_gold = json.load(open(f"{AU}/box/train_block_india.json"))["gold_sizes"]
tot = sum(json_gold.values())
print(f"pairs {n} -> kept {keep.sum()} ({keep.sum() / 120000:.2f}/S1); positives {y.sum()} -> {(y & keep).sum()} "
      f"(kept frac {(y & keep).sum() / y.sum():.4f}); PC pre {y.sum() / tot:.4f} post {(y & keep).sum() / tot:.4f}", flush=True)
samp_ids = S1.entity_id.values
del R, S1
gn = np.array([json_gold[e] for e in samp_ids], np.int32)


def oracle(mask):
    hit = np.bincount(si[mask & y], minlength=120000)
    f = np.ones(120000)
    nz = gn > 0
    r = hit[nz] / gn[nz]
    f[nz] = np.where(r == 0, 0.0, 1.25 * r / (0.25 + r))
    return f.mean()


allm = np.ones(n, bool)
print(f"oracle pre-prune {oracle(allm):.4f}  post-prune {oracle(keep):.4f}", flush=True)
rows = []
def rep(tag, m):
    rows.append((tag, round(m.sum() / 120000, 2), int((m & y).sum()), round((m & y).sum() / tot, 4),
                 round((m & y).sum() / y.sum(), 4), round(oracle(m), 4)))
for k in (28, 32, 36, 40, 48, 56, 64, 80, 100):
    rep(f"prune_k={k}, rev<=2", (rk <= k) | (rev <= 2))
for rv in (1, 3, 4, 6):
    rep(f"prune_k=28, rev<={rv}", (rk <= 28) | (rev <= rv))
famrk = {c: np.nan_to_num(P[c].values, nan=1e9) for c in ["b_key_rk", "b_nkey_rk", "b_akey_rk", "b_xkey_rk", "b_cgram_rk", "b_pkey_rk"]}
mn = np.minimum.reduce(list(famrk.values()))
for m_ in (1, 2, 3, 5):
    rep(f"k=28 + any-family-rank<={m_}", keep | (mn <= m_))
for c, v in famrk.items():
    rep(f"k=28 + {c}<=3", keep | (v <= 3))
rep("k=28 + all sibling-expanded", keep | (P.sib_sim.values > 0))
rep("k=40 + rev<=3 + any-family-rank<=2", (rk <= 40) | (rev <= 3) | (mn <= 2))
print(pd.DataFrame(rows, columns=["rule", "pairs/S1", "pos_kept", "PC", "kept_frac_of_blocked", "oracle"]).to_string(index=False), flush=True)
# per-S1 28th-best score (threshold) for margin analysis
o = np.lexsort((-score, si))
first = np.r_[0, np.flatnonzero(np.diff(si[o])) + 1]
cnt = np.diff(np.r_[first, n])
thr = np.full(120000, -1.0, np.float32)
has = cnt >= 28
thr[si[o][first[has]]] = score[o][first[has] + 27]
P["score"], P["nm"], P["ad"], P["rk"], P["rev"], P["keep"] = score, nm, ad, rk, rev, keep
P["thr28"] = thr[si]
P["n_cand"] = cnt[np.searchsorted(si[o][first], si)]
P["gold_n"] = gn[si]
P[P.y].to_parquet(f"{AU}/prune_pos.parquet", index=False)
print("saved", time.time() - t0)
