"""Memory-light (batched) version of atrisk_sim.py: test-time survival of train positives kept only by a
record-side protection, at the K stage AND the similarity-prune stage, against all visible S1 suitors."""
import glob, sys
import numpy as np, pandas as pd, pyarrow as pa
sys.path.insert(0, "/Users/mac/amazon-ml-2026/code/ber_v5")
from src.block_big import KEY_FIELDS
from rapidfuzz import fuzz, process

AU = "/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad/autopsy"
FAMS = list(KEY_FIELDS)
C3 = ["name_core", "name_phon", "addr_norm"]
parts = lambda tag: sorted(glob.glob(f"{AU}/norm/{tag}/part-*.parquet"))
TOT, KEPT = 415630, 402141
AR = pd.read_parquet(f"{AU}/atrisk.parquet")
PP = pd.read_parquet(f"{AU}/prune_pos.parquet")
alt = np.fmax.reduce([PP[c].values for c in FAMS[1:]])
PP["kscore"] = np.where(np.isnan(PP.b_key.values), np.nan_to_num(alt) * 0.5, PP.b_key.values)
ar = AR[AR.keep].merge(PP[["s1_i", "r_i", "kscore", "score"]], on=["s1_i", "r_i"]).reset_index(drop=True)
print("at-risk kept-in-train", len(ar), "kprot", int(ar.kprot.sum()), "pprot", int(ar.pprot.sum()), flush=True)

# visible S1 strings as arrow (compact), records' strings
rs = np.random.RandomState(0)
n_s1 = sum(pd.read_parquet(p, columns=["name_core"]).shape[0] for p in parts("s1"))
vis_mask = rs.rand(n_s1) >= 0.19
S1s = {c: [] for c in C3}
off = 0
for p in parts("s1"):
    d = pd.read_parquet(p, columns=C3)
    m = vis_mask[off:off + len(d)]; off += len(d)
    for c in C3:
        S1s[c].append(pa.array(d[c].values[m], pa.large_string()))
S1s = {c: pa.chunked_array(v).combine_chunks() for c, v in S1s.items()}
need = pd.Index(np.unique(ar.r_i.values))
Rs, off = [], 0
for p in parts("s2") + parts("s3"):
    d = pd.read_parquet(p, columns=C3)
    gi = np.arange(off, off + len(d)); off += len(d)
    m = need.get_indexer(gi) >= 0
    if m.any():
        x = d[m].copy(); x["_g"] = gi[m]; Rs.append(x)
Rs = pd.concat(Rs).set_index("_g")
print("strings loaded", flush=True)

S = []
for f in FAMS:
    x = pd.read_parquet(f"{AU}/atrisk_suitors_{f}.parquet")
    x["fam"] = f
    x["strict"] = x.rrk <= KEY_FIELDS[f][3]
    S.append(x)
S = pd.concat(S, ignore_index=True)
cp = lambda x, y, sc: process.cpdist(x, y, scorer=sc, workers=4, dtype=np.float32)
recs = ar.r_i.values
cnt = {}
B = 1500
for lo in range(0, len(recs), B):
    rb = set(recs[lo:lo + B])
    Sb = S[S.r_i.isin(rb)]
    W = Sb.pivot_table(index=["r_i", "s1_vis_i"], columns="fam", values="cos", aggfunc="max").reindex(columns=FAMS)
    st = Sb.groupby(["r_i", "s1_vis_i"]).strict.any().reindex(W.index).values
    bst = Sb[(Sb.fam == "b_key") & Sb.strict].set_index(["r_i", "s1_vis_i"]).index
    W = W.reset_index()
    bk = W.b_key.values
    al = np.fmax.reduce([W[c].values for c in FAMS[1:]])
    idx = pa.array(W.s1_vis_i.values.astype(np.int64))
    A = {c: S1s[c].take(idx).to_pylist() for c in C3}
    Rb = {c: Rs[c].reindex(W.r_i.values).tolist() for c in C3}
    nm = np.maximum.reduce([cp(A["name_core"], Rb["name_core"], fuzz.token_set_ratio),
                            cp([a.replace(" ", "") for a in A["name_core"]], [b.replace(" ", "") for b in Rb["name_core"]], fuzz.ratio),
                            cp(A["name_phon"], Rb["name_phon"], fuzz.token_set_ratio)])
    ad = cp(A["addr_norm"], Rb["addr_norm"], fuzz.token_set_ratio)
    arb = ar[ar.r_i.isin(rb)]
    for variant in ("strict", "loose"):
        if variant == "strict":
            pb = pd.MultiIndex.from_frame(W[["r_i", "s1_vis_i"]]).isin(bst); mem = st
        else:
            pb = np.isfinite(bk); mem = np.ones(len(W), bool)
        ks = np.where(pb, np.nan_to_num(bk), np.nan_to_num(al) * 0.5)
        ps = nm + ad + 50 * np.where(pb, np.nan_to_num(bk), 0.0)
        T = pd.DataFrame({"r_i": W.r_i.values, "comp": W.s1_vis_i.values, "ks": ks, "ps": ps})[mem]
        for G in (1, 6):
            J = arb[["r_i", "s1_vis_i", "kscore", "score"]].merge(T, on="r_i")
            J = J[J.comp != J.s1_vis_i]
            if G > 1:
                J = J[(J.comp % G) == (J.s1_vis_i % G)]
            kb = (J.ks > J.kscore + 1e-6).groupby(J.r_i).sum()
            pbt = (J.ps > J.score + 1e-4).groupby(J.r_i).sum()
            key = (variant, G)
            c = cnt.setdefault(key, [pd.Series(dtype=float), pd.Series(dtype=float)])
            c[0] = pd.concat([c[0], kb]); c[1] = pd.concat([c[1], pbt])
    print(f"  batch {lo} pairs {len(W)}", flush=True)

rows = []
for (variant, G), (kb, pbt) in cnt.items():
    k_better = ar.r_i.map(kb).fillna(0).values
    p_better = ar.r_i.map(pbt).fillna(0).values
    k_ok = ~ar.kprot.values | (k_better <= 1)
    p_ok = (ar.rk.values <= 28) | (p_better <= 1)
    lost = int((~(k_ok & p_ok)).sum())
    ar[f"surv_{variant}_{G}"] = k_ok & p_ok
    rows.append((variant, "global (test as-is)" if G == 1 else f"grouped G={G}", int((~k_ok).sum()),
                 int((k_ok & ~p_ok).sum()), lost, round((KEPT - lost) / TOT, 4)))
out = pd.DataFrame(rows, columns=["suitors", "protection", "lost@K", "lost@prune(after K ok)", "lost total",
                                  "PC after prune (at-risk only)"])
print(out.to_string(index=False))
out.to_csv(f"{AU}/atrisk_sim.tsv", sep="\t", index=False)

# oracle macro-F0.5 on the 120k sample: gold = positives in table + missed; hits = train-kept minus scenario losses
rs = np.random.RandomState(0)
vis = np.flatnonzero(rs.rand(883188) >= 0.19); samp = np.sort(rs.choice(len(vis), 120000, replace=False))
Mi = pd.read_parquet(f"{AU}/missed.parquet")
gold = np.bincount(PP.s1_i.values, minlength=120000) + np.bincount(np.searchsorted(samp, Mi.s1_vis_i.values), minlength=120000)
hit_train = np.bincount(PP.s1_i.values[PP.keep.values], minlength=120000)


def oracle(hit):
    f = np.ones(120000); nz = gold > 0
    r = hit[nz] / gold[nz]
    f[nz] = np.where(r == 0, 0, 1.25 * r / (0.25 + r))
    return f.mean()


print("oracle train post-prune", round(oracle(hit_train), 4), "| pre-prune", round(oracle(np.bincount(PP.s1_i.values, minlength=120000)), 4))
for col in [c for c in ar.columns if c.startswith("surv_")]:
    lost = np.bincount(ar.s1_i.values[~ar[col].values], minlength=120000)
    print(col, "PC", round((KEPT - lost.sum()) / TOT, 4), "oracle", round(oracle(hit_train - lost), 4))
