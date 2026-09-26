"""Exact v10 India miss set from the box's own train_block_india.parquet (after family top-k, K=40 cap, sibling
expansion). Reconstructs R order (S2 india rows then S3, file order = prep order) and the visible/sample split
(run_big.train_split, seed 0, hide 0.19, 120k). Writes missed.parquet + control.parquet (retrieved positives)."""
import csv, glob, json, numpy as np, pandas as pd

AU = "/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad/autopsy"
REAL = AU + "/../real"
rd = lambda tag: pd.concat([pd.read_parquet(p, columns=["entity_id"]) for p in sorted(glob.glob(f"{AU}/norm/{tag}/part-*.parquet"))],
                           ignore_index=True).entity_id.values
s1_ids = rd("s1")
r_ids = np.concatenate([rd("s2"), rd("s3")])
meta = json.load(open(f"{AU}/box/train_block_india.json"))
print("n_s1", len(s1_ids), "n_r", len(r_ids), "box n_r", meta["n_r"])
assert len(r_ids) == meta["n_r"]
rs = np.random.RandomState(0)
visible = np.sort(np.flatnonzero(rs.rand(len(s1_ids)) >= 0.19))
samp = np.sort(rs.choice(len(visible), 120000, replace=False))
samp_ids = s1_ids[visible[samp]]
print("visible", len(visible), "sample ids match json:", set(samp_ids) == set(meta["gold_sizes"]))
gold = {}
for ch in pd.read_csv(f"{REAL}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, chunksize=500000):
    ch = ch[ch.source1_entity_id.isin(set(samp_ids))]
    for s, v in zip(ch.source1_entity_id.values, ch.matched_entity_ids.values):
        gold[s] = [x for x in v.split(",") if x]
P = pd.read_parquet(f"{AU}/box/train_block_india.parquet",
                    columns=["s1_i", "r_i", "y", "s1_id", "b_key", "b_nkey", "b_akey", "b_xkey", "b_cgram", "b_pkey",
                             "sib_sim", "b_rank", "n_cand_s1"])
print("pairs", len(P), "positives in cands", int(P.y.sum()), "gold total", sum(len(v) for v in gold.values()))
assert (samp_ids[P.s1_i.values] == P.s1_id.values).all()
P["r_id"] = r_ids[P.r_i.values]
ret = set(zip(P.s1_id.values[P.y.values], P.r_id.values[P.y.values]))
rix = pd.Index(r_ids)
vix = pd.Index(s1_ids[visible])
rows = []
for s, g in gold.items():
    for r in g:
        if (s, r) not in ret:
            rows.append((s, r))
M = pd.DataFrame(rows, columns=["s1_id", "r_id"])
M["r_i"] = rix.get_indexer(M.r_id.values)
M["s1_vis_i"] = vix.get_indexer(M.s1_id.values)
M["gold_n"] = M.s1_id.map({k: len(v) for k, v in gold.items()})
nc = P.groupby("s1_id").size()
M["n_cand"] = M.s1_id.map(nc).fillna(0).astype(int)
M["n_found"] = M.s1_id.map(P[P.y].groupby("s1_id").size()).fillna(0).astype(int)
print("missed pairs", len(M), "unresolved r_i", int((M.r_i < 0).sum()), "PC", 1 - len(M) / sum(len(v) for v in gold.values()))
print("missed S1 unique", M.s1_id.nunique(), "S1 with ALL gold missed", int((M.n_found == 0).sum()))
M.to_parquet(f"{AU}/missed.parquet", index=False)
# control: retrieved positives proposed by at least one family (not sibling-expansion-only)
fam = ["b_key", "b_nkey", "b_akey", "b_xkey", "b_cgram", "b_pkey"]
C = P[P.y & P[fam].notna().any(axis=1)].sample(3000, random_state=0).copy()
C["s1_vis_i"] = vix.get_indexer(C.s1_id.values)
C[["s1_id", "r_id", "r_i", "s1_vis_i"] + fam].to_parquet(f"{AU}/control.parquet", index=False)
print("sibling-expansion-only positives:", int((P.y & P[fam].isna().all(axis=1)).sum()))
# per-S1 gold size table for oracle calcs
pd.DataFrame({"s1_id": list(gold), "gold_n": [len(v) for v in gold.values()]}).to_parquet(f"{AU}/gold_sizes.parquet", index=False)
