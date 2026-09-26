"""atrisk_sim2 with the ACTUAL v10 test semantics (run_big.py:500 + block_big.py:249-280, run_big.py:522):
  K stage   : prune_topk runs per S1 RANGE (s1_parts=4 contiguous index ranges), so the record-top-2 protection
              ranks a record's suitors among the S1s of the true S1's range only.
  prune stage: prune_by_similarity runs on the whole country table, so rev_rank<=2 ranks among ALL S1s whose pair
              with the record survived the K stage.
Competitor survival of the K stage (needed for the prune-stage pool) is bracketed:
  pool=all   : every suitor is a prune-stage competitor (upper bound on competition -> pessimistic)
  pool=rtop2 : only suitors that are record-top-2 by K-score within THEIR OWN range (lower bound -> optimistic;
               ignores suitors kept by their own S1-side K-rank <= 40)
parts = 4 (inclusion 0.25) and 3 (0.33, stands in for the 13% denser test S1 table: 810k vs 715k visible)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/mac/amazon-ml-2026/code/ber_v5")
from src.block_big import KEY_FIELDS, group_rank
from rapidfuzz import fuzz, process

AU = "/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad/autopsy"
FAMS = list(KEY_FIELDS)
TOT, KEPT, NVIS = 415630, 402141, 715441
AR = pd.read_parquet(f"{AU}/atrisk.parquet")
PP = pd.read_parquet(f"{AU}/prune_pos.parquet")
alt = np.fmax.reduce([PP[c].values for c in FAMS[1:]])
PP["kscore"] = np.where(np.isnan(PP.b_key.values), np.nan_to_num(alt) * 0.5, PP.b_key.values)
ar = AR[AR.keep].merge(PP[["s1_i", "r_i", "kscore", "score"]], on=["s1_i", "r_i"]).reset_index(drop=True)
S1s = pd.read_parquet(f"{AU}/sub_s1.parquet").set_index("s1_vis_i")
Rs = pd.read_parquet(f"{AU}/sub_r.parquet").set_index("r_i")
S = []
for f in FAMS:
    x = pd.read_parquet(f"{AU}/atrisk_suitors_{f}.parquet")
    x["fam"] = f
    x["strict"] = x.rrk <= KEY_FIELDS[f][3]
    S.append(x)
S = pd.concat(S, ignore_index=True)
W = S.pivot_table(index=["r_i", "s1_vis_i"], columns="fam", values="cos", aggfunc="max").reindex(columns=FAMS)
st = S.groupby(["r_i", "s1_vis_i"]).strict.any().reindex(W.index).values
bst = S[(S.fam == "b_key") & S.strict].set_index(["r_i", "s1_vis_i"]).index
W = W.reset_index()
bk = W.b_key.values
al = np.fmax.reduce([W[c].values for c in FAMS[1:]])
cp = lambda x, y, sc: process.cpdist(x, y, scorer=sc, workers=4, dtype=np.float32)
A = {c: S1s[c].reindex(W.s1_vis_i.values).tolist() for c in ["name_core", "name_phon", "addr_norm"]}
B = {c: Rs[c].reindex(W.r_i.values).tolist() for c in ["name_core", "name_phon", "addr_norm"]}
nm = np.maximum.reduce([cp(A["name_core"], B["name_core"], fuzz.token_set_ratio),
                        cp([a.replace(" ", "") for a in A["name_core"]], [b.replace(" ", "") for b in B["name_core"]], fuzz.ratio),
                        cp(A["name_phon"], B["name_phon"], fuzz.token_set_ratio)])
ad = cp(A["addr_norm"], B["addr_norm"], fuzz.token_set_ratio)
del A, B
print("suitor pairs", len(W), flush=True)

# harness check: recompute the true pair's prune score from the subset strings; must equal the table's score
T0 = ar[["r_i", "s1_vis_i", "score"]].merge(pd.DataFrame({"r_i": W.r_i.values, "s1_vis_i": W.s1_vis_i.values,
                                                          "nm": nm, "ad": ad, "bk": bk}), on=["r_i", "s1_vis_i"])
d = np.abs(T0.nm + T0.ad + 50 * np.nan_to_num(T0.bk) - T0.score)
print(f"harness: true pair found among suitors {len(T0)}/{len(ar)}; |prune score diff| median {d.median():.4f} "
      f"share<0.01 {(d < 0.01).mean():.3f}", flush=True)


def rng(idx, parts):
    cuts = np.linspace(0, NVIS, parts + 1).astype(np.int64)
    return np.searchsorted(cuts, idx, side="right") - 1


rows = []
surv = {}
for variant in ("strict", "loose"):
    if variant == "strict":
        pb = pd.MultiIndex.from_frame(W[["r_i", "s1_vis_i"]]).isin(bst); mem = st
    else:
        pb = np.isfinite(bk); mem = np.ones(len(W), bool)
    ks = np.where(pb, np.nan_to_num(bk), np.nan_to_num(al) * 0.5).astype(np.float32)
    ps = nm + ad + 50 * np.where(pb, np.nan_to_num(bk), 0.0)
    T = pd.DataFrame({"r_i": W.r_i.values, "comp": W.s1_vis_i.values, "ks": ks, "ps": ps})[mem].reset_index(drop=True)
    for parts in (1, 3, 4):
        T["q"] = rng(T.comp.values, parts)
        # competitor's own record-top-2 within its range (for the optimistic prune pool)
        grp = T.r_i.values.astype(np.int64) * 8 + T.q.values
        T["rtop2"] = group_rank(grp, T.ks.values) <= 2
        J = ar[["r_i", "s1_vis_i", "kscore", "score"]].merge(T, on="r_i")
        J = J[J.comp != J.s1_vis_i]
        Jq = J[J.q == rng(J.s1_vis_i.values, parts)]
        kb = (Jq.ks > Jq.kscore + 1e-6).groupby(Jq.r_i).sum()
        k_better = ar.r_i.map(kb).fillna(0).values
        k_ok = ~ar.kprot.values | (k_better <= 1)
        for pool in ("all", "rtop2"):
            Jp = J if pool == "all" else J[J.rtop2]
            pbt = (Jp.ps > Jp.score + 1e-4).groupby(Jp.r_i).sum()
            p_better = ar.r_i.map(pbt).fillna(0).values
            p_ok = (ar.rk.values <= 28) | (p_better <= 1)
            s = k_ok & p_ok
            surv[(variant, parts, pool)] = s
            rows.append((variant, parts, pool, int((~k_ok).sum()), int((k_ok & ~p_ok).sum()), int((~s).sum()),
                         round((KEPT - (~s).sum()) / TOT, 4)))
        print(f"  {variant} parts={parts}: share of competitor suitors in the true S1's range "
              f"{len(Jq) / max(len(J), 1):.3f}", flush=True)
out = pd.DataFrame(rows, columns=["suitors", "K-stage parts", "prune pool", "lost@K", "lost@prune(after K ok)",
                                  "lost total", "PC after prune"])
pd.set_option("display.width", 200)
print(out.to_string(index=False))
out.to_csv(f"{AU}/atrisk_sim3.tsv", sep="\t", index=False)

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


print("oracle train post-prune", round(oracle(hit_train), 4))
for k, s in surv.items():
    lost = np.bincount(ar.s1_i.values[~s], minlength=120000)
    print(k, "oracle", round(oracle(hit_train - lost), 4))

# ---- expected values (strict suitors, which pass the IPW calibration check in kstage_ipw.py) ----
# K stage: at-risk positive i survives w.p. P(Bin(nb_i, q) <= 1); positives the cap CUT in train are rescued in
# expectation by the IPW inflow (kstage_ipw.py). Prune stage: evaluated directly with the full-table pool.
from scipy.stats import binom
Q_TR = 119999 / 715440
pb = pd.MultiIndex.from_frame(W[["r_i", "s1_vis_i"]]).isin(bst)
ks = np.where(pb, np.nan_to_num(bk), np.nan_to_num(al) * 0.5).astype(np.float32)
ps = nm + ad + 50 * np.where(pb, np.nan_to_num(bk), 0.0)
T = pd.DataFrame({"r_i": W.r_i.values, "comp": W.s1_vis_i.values, "ks": ks, "ps": ps})[st].reset_index(drop=True)
J = ar[["r_i", "s1_vis_i", "kscore", "score"]].merge(T, on="r_i")
J = J[J.comp != J.s1_vis_i]
nb = ar.r_i.map((J.ks > J.kscore + 1e-6).groupby(J.r_i).sum()).fillna(0).values
for q, lab in ((0.25, "quarters q=.25"), (0.283, "quarters at test density q=.283")):
    pk = np.where(ar.kprot.values, binom.cdf(1, nb, q), 1.0)
    w = np.where(ar.kprot.values, 1 / binom.cdf(1, nb, Q_TR), 0.0)
    inflow = float((binom.cdf(1, nb, q) * (w - 1) * ar.kprot.values).sum())   # train-cut positives rescued at K
    for pool in ("all", "rtop2"):
        T["q"] = rng(T.comp.values, 4)
        T["rtop2"] = group_rank(T.r_i.values.astype(np.int64) * 8 + T.q.values, T.ks.values) <= 2
        Jp = J.merge(T[["r_i", "comp", "rtop2"]], on=["r_i", "comp"])
        if pool == "rtop2":
            Jp = Jp[Jp.rtop2]
        pbt = ar.r_i.map((Jp.ps > Jp.score + 1e-4).groupby(Jp.r_i).sum()).fillna(0).values
        p_ok = (ar.rk.values <= 28) | (pbt <= 1)
        exp_kept = (pk * p_ok).sum()
        lost = len(ar) - exp_kept - 0.86 * inflow      # 0.86 = test-like prune survival of A misses (section 3)
        print(f"[expected] {lab:32s} prune pool={pool:5s}: K loss {(1 - pk).sum():6.0f}, prune loss {(pk * ~p_ok).sum():6.0f}, "
              f"K inflow x0.86 {0.86 * inflow:5.0f} -> net lost {lost:6.0f} -> India PC after prune {(KEPT - lost) / TOT:.4f}")
