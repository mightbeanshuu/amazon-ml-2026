"""Sibling route for the v10 misses: does the missed record have a sibling edge (the box's own train_india_sib.npz,
top-3 b_key self-search within R) to a TRUE co-record of the same S1 that blocking DID retrieve? And which of the
expansion gates (weak_q=0.30, min_sim=0.55, sibling's top_s1=5 by b_key, two-hop) stopped it."""
import csv, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/mac/amazon-ml-2026/code/ber_v5")
from src.block_big import group_rank
from src.normalize import clean

AU = "/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad/autopsy"
REAL = AU + "/../real"
TOT = 415630
M = pd.read_parquet(f"{AU}/missed_full.parquet", columns=["s1_id", "r_id", "r_i", "s1_i", "cause", "r_addr_empty",
                                                         "r_oov", "s1_dup_name"])
gtid = {}
want = set(M.s1_id)
for ch in pd.read_csv(f"{REAL}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, chunksize=500000):
    ch = ch[ch.source1_entity_id.isin(want)]
    for s_, v_ in zip(ch.source1_entity_id, ch.matched_entity_ids):
        gtid[s_] = [x for x in v_.split(",") if x]
need = {x for v_ in gtid.values() for x in v_}
pos, off, cc = {}, 0, {}
for fn in ("train_source2.tsv", "train_source3.tsv"):
    for ch in pd.read_csv(f"{REAL}/{fn}", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
                          chunksize=500000, usecols=["entity_id", "country"]):
        m = ch.country.map(lambda c: cc.setdefault(c, clean(c) == "india")).values
        e = ch.entity_id.values[m]
        for j in np.flatnonzero(pd.Series(e).isin(need).values):
            pos[e[j]] = off + j
        off += len(e)
assert off == 4133346, off
assert all(pos[r] == i for r, i in zip(M.r_id.values, M.r_i.values))
gt = {s_: np.array([pos[x] for x in v_]) for s_, v_ in gtid.items()}
P = pd.read_parquet(f"{AU}/box/train_block_india.parquet", columns=["s1_i", "r_i", "y", "b_key"])
n_r = 4133346
best = np.full(n_r, -1.0, np.float32)
np.maximum.at(best, P.r_i.values, np.nan_to_num(P.b_key.values, nan=-1.0))
thr = np.quantile(best[best >= 0], 0.30)
weak = best < thr
rk5 = group_rank(P.r_i.values, np.nan_to_num(P.b_key.values, nan=-1.0))
found = P[P.y]
top5 = set(zip(found.s1_i.values[rk5[P.y.values] <= 5], found.r_i.values[rk5[P.y.values] <= 5]))
found_set = set(zip(found.s1_i.values, found.r_i.values))
del P
z = np.load(f"{AU}/box/train_india_sib.npz")
q, s2, v = z["q"], z["s"], z["v"]
E = pd.DataFrame({"q": q, "s": s2, "v": v})
Eq = E[E.q.isin(set(M.r_i))]
# two-hop edges from missed records (same composition rule as siblings.two_hop, min_sim .45)
J = Eq.merge(E.rename(columns={"q": "s", "s": "t", "v": "v2"}), on="s")
J = J[J.q != J.t]
J["v"] = J.v * J.v2
J = J[J.v >= 0.45][["q", "t", "v"]].rename(columns={"t": "s"})
H = pd.concat([Eq.assign(hop=1), J.assign(hop=2)], ignore_index=True)
H = H.sort_values("v", ascending=False).drop_duplicates(["q", "s"])
H["hrk"] = group_rank(H.q.values, H.v.values.astype(np.float32))
rows = []
for s1, r, s1i in zip(M.s1_id.values, M.r_i.values, M.s1_i.values):
    co = set(gt[s1].tolist()) - {r}
    e = H[H.q == r] if False else None
    rows.append((s1i, r, co))
Hg = H.groupby("q")
res = []
for s1i, r, co in rows:
    if r not in Hg.groups:
        res.append((0, 0, 0, 0, 0, 0.0))
        continue
    e = Hg.get_group(r)
    tr = e[e.s.isin(co)]                        # sibling edges that point at a TRUE co-record
    f1 = tr[np.array([(s1i, x) in found_set for x in tr.s.values], bool)]
    f1d = f1[f1.hop == 1]
    ok_sim = f1[(f1.v >= 0.55) & (f1.hrk <= 3)]
    ok_top5 = ok_sim[np.array([(s1i, x) in top5 for x in ok_sim.s.values], bool)]
    ok_weak_sib = ok_top5[np.array([not weak[x] for x in ok_top5.s.values], bool)]
    res.append((len(tr) > 0, len(f1d) > 0, len(f1) > 0, len(ok_top5) > 0, len(ok_weak_sib) > 0,
                float(tr.v.max()) if len(tr) else 0.0))
R = pd.DataFrame(res, columns=["sib_is_true_corec", "direct_sib_found", "any_sib_found", "gate_sim_top5",
                               "gate_sim_top5_sibstrong", "best_true_sib_v"])
M = pd.concat([M.reset_index(drop=True), R], axis=1)
M["r_weak"] = weak[M.r_i.values]
print("misses", len(M), "| S1 with some found co-record:", round(np.mean([len(set(gt[s]) - {r}) > 0 for s, r in zip(M.s1_id, M.r_i)]), 3))
for c in ["sib_is_true_corec", "direct_sib_found", "any_sib_found", "gate_sim_top5", "gate_sim_top5_sibstrong"]:
    print(f"{c:28s} {int(M[c].sum()):6d} {M[c].mean():.3f}  (+{M[c].sum() / TOT * 100:.2f} PC pts)")
print("of gate_sim_top5_sibstrong: record weak (should be ~0, else expansion would have added it):",
      int((M.gate_sim_top5_sibstrong & M.r_weak).sum()))
print("by cause (any_sib_found):", M.groupby("cause").any_sib_found.mean().round(3).to_dict())
print("by cause (gate_sim_top5_sibstrong):", M.groupby("cause").gate_sim_top5_sibstrong.mean().round(3).to_dict())
print("weak threshold", thr, "| missed records weak:", round(M.r_weak.mean(), 3))
M.to_parquet(f"{AU}/missed_sib.parquet", index=False)
