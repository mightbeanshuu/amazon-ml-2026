# Research pass 8 — EXACT code specs from the top public repos (26 Sep 2026, ~18:00 IST)

Deep code extraction for v10. Everything below is read from the actual source files (fetched raw from GitHub 26 Sep ~17:50 IST), not from READMEs. Verbatim snippets are marked. Local copies of every file cited: `/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad/{ayan,tony,priy,disa,skull}` (session-temporary — re-fetch from GitHub if gone).

Our pipeline for the DELTA notes: hashed word-key + phonetic + cgram record-side blocking with prune-28; 89 features incl rapidfuzz sims/IDF/TF-evidence/rev_rank/sibling vouch; stage B vouching; enrichment x2; LOCO-gated selftrain; per-country isotonic + GFM decision; NaN masking done.

---

## 1. AyanAhmedKhan/amazon-ml-challenge — local stacked 0.98703 ± 0.00016

Repo layout: `src/ber/*.py` (library) + `modal_app/stage_*.py` (pipeline stages on Modal). Runtime: every stage runs `@app.function(cpu=32, memory=131072..229376, timeout≤20h)` — i.e. **32 vCPU / 128–224 GB RAM Modal containers**. Blocking on full train took 7,790 s wall (India 1,927 s, US 5,587 s) at 32 cpu (from `experiments/reports/block_train_v1.json`).

### 1a. Blocking / retrieval — `src/ber/blocking.py` + `modal_app/stage_block.py`

**Vectorizers (verbatim, `ber/blocking.py`):**
```python
def _vectorizer(kind: str, max_df: float, min_df: int = 2):
    from sklearn.feature_extraction.text import TfidfVectorizer
    if kind == "char_wb":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=min_df, max_df=max_df,
                               sublinear_tf=True, dtype=np.float32, lowercase=False)
    if kind == "char":
        return TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=min_df, max_df=max_df,
                               sublinear_tf=True, dtype=np.float32, lowercase=False)
    if kind == "word":
        return TfidfVectorizer(analyzer="word", token_pattern=r"[a-z0-9]+", min_df=min_df, max_df=max_df,
                               sublinear_tf=True, dtype=np.float32, lowercase=False)
```
- Fit on a **3M random sample of the union of S1+targets per country partition** (`tfidf_pair`, `fit_sample=3_000_000`), transform both sides in parallel joblib chunks of 200k.
- `max_df=0.05` for all three views, `min_df=2`, `sublinear_tf=True`, **no lowercase** (text already normalized upstream).
- Hybrid view (verbatim): `hstack_views` — "Weighted concat of L2-normalised views -> cosine = sum_w w * cos_view":
```python
mats = [m.multiply(np.float32(np.sqrt(w))).tocsr() for m, w in views]  # w_name=0.5
return sp.hstack(mats).tocsr()
```
- Top-k engine: `sparse_dot_topn.sp_matmul_topn(Q[chunk], T.T, top_n=k, threshold=0.0, sort=True, n_threads=32)`, Q chunked 250k rows.

**Views and K (verbatim `stage_block.py` DEFAULT_CFG + the flow):**
```python
DEFAULT_CFG = {"K": {"name": 40, "cat": 20, "addr": 40, "combo": 40}, "K_rev": 8,
               "max_df": {"name": 0.05, "cat": 0.05, "addr": 0.05}, "w_name": 0.5, "tag": "v1"}
```
Per country partition (S1 = src==1 queries, targets = src!=1):
- `name`: char_wb 3-grams of normalized name `nm`, top-40
- `cat` : char 3-grams of `nm` with spaces removed (`s.replace(" ", "")`), top-20
- `addr`: word unigrams `[a-z0-9]+` of normalized address `ad`, top-40
- `combo`: `[sqrt(0.5)*name | sqrt(0.5)*addr]` hstack, top-40
- `rev`: **the SAME combo matrices with Q and T swapped** — `topk(Tx, Qx, cfg["K_rev"])` — i.e. each target retrieves its top-8 S1 records. One line of code, no separate index.

Union: the 5 per-view frames are outer-joined on (q,t); per-view score `s_<view>` fill_null(0.0), per-view rank `r_<view>` fill_null(999) as Int16 — so retrieval provenance (rank in every view) survives into features.

**Measured recall (verbatim from `experiments/reports/block_train_v1.json`, full train pool):**
- name R@40 = 0.6773; cat R@20 = 0.6561; addr R@40 = 0.8863; combo R@10 = 0.9600, R@40 = 0.9751
- **rev R@1 = 0.9717, R@3 = 0.9825, R@5 = 0.9855, R@8 = 0.98746**
- union pair recall = **0.99051** at 112.3 cands/S1; uniform-K grid: K=5 → 0.98823 @ 41.9/S1, K=10 → 0.98899 @ 51.6/S1
- entity_full_recall (all true matches of an S1 in) = 0.9670; **oracle macro F0.5 = 0.99715** (US 0.99741, India 0.99678)
- Vocab sizes at max_df 0.05: name ~20k, cat ~23-30k, addr 173-299k n-grams/words.

**Final prune (verbatim `configs/final.json`):** `"prune": {"rev": 8, "combo": 10, "name": 5, "cat": 5, "addr": 5}` — keep a pair if ANY view ranks it under its cap (`stage_features.py`: `m = m | (pl.col(f"r_{v}") < k)`). ≈43–51 cands/S1 at recall 0.989, oracle 0.9965.

DELTA vs ours: we have rev_rank as a *feature* but our blocking is record-side hashed keys, not a target→S1 top-8 dense-scored TF-IDF view unioned with 4 forward views with per-view rank caps. Their rev view ALONE beats a full forward union (R@1 0.972 vs combo R@40 0.975). Our prune-28 budget is smaller than their ~45.

### 1b. Features — `stage_features.py` + `ber/features.py` (~89 cols)

Groups: (i) exact TF-IDF cosines recomputed per country for all candidate pairs: `c_name` (char_wb3), `c_cat` (concat char3), `c_nword` (word, max_df 0.5), `c_addr` (word), `c_combo = 0.5*c_name + 0.5*c_addr`; (ii) optional potion-multilingual-128M static embedding cosines `e_name/e_addr/e_full`; (iii) 13 RapidFuzz sims via `cpdist(workers=-1)`: n_ratio/tsort/tset/partial/WRatio/JaroWinkler + concat ratio/partial/Levenshtein + addr ratio/tsort/tset/partial; (iv) 30 token features `TOK_NAMES` (IDF-weighted name Jaccard/coverage, rarest-shared-token IDF, core-token (non-generic) Jaccard/equality, initials-match, phonetic Jaccard, alias-part Jaccard, truncation-prefix, address number Jaccard/first-eq/shared/missing/big(≥3-digit)-num Jaccard, addr IDF overlap, **house-number geometry: min |a−b|, min relative diff, min digit edit distance over first 4×4 numeric tokens**, all −1.0 sentinel when absent); (v) structural: `t_src`, script codes, has_domain/alias, **name-frequency ambiguity counts `q_nm_freq_s1`, `t_nm_freq_s1`, `t_nm_freq_t`** (exact normalized-name counts per country per side); (vi) context/competition (below).

Generic tokens are **data-driven per country**: document frequency > 1% of that country's names (no hand list).

**Competition features (verbatim `ber/features.py::context_feats`)** — for each score col in `["c_combo","c_name","c_addr","e_full"]`:
```python
pl.col(c).rank("ordinal", descending=True).over("q").alias(f"{c}_qrank"),
(pl.col(c).max().over("q") - pl.col(c)).alias(f"{c}_qgap"),
pl.col(c).rank(...).over("t").alias(f"{c}_trank"),
(pl.col(c).max().over("t") - pl.col(c)).alias(f"{c}_tgap"),
# plus q_ncand, t_nq, and the many-to-one margin:
g = df.group_by("t").agg(pl.col(c).max().alias("_b1"), pl.col(c).top_k(2).min().alias("_b2"), pl.len().alias("_n"))
pl.when(pl.col(c) < pl.col("_b1")).then(pl.col(c) - pl.col("_b1"))
 .when(pl.col("_n") >= 2).then(pl.col(c) - pl.col("_b2")).otherwise(pl.col(c)).alias(f"{c}_tmargin")
```
`c_combo_tmargin` — "margin vs the best OTHER S1 competing for the same target; > 0 only when this q is the unique best S1 for t" — is their **top gain feature by far** (FINAL_RECOMMENDATION: "c_combo_tmargin ≫ others; exclusivity adds ≈0 on top").

### 1c. Stage-1 / stage-2 models — `stage_train.py`, `stage_collective.py`

**LightGBM params (verbatim, both stages):**
```python
DEFAULT_PARAMS = {"objective": "binary", "learning_rate": 0.08, "num_leaves": 255,
    "min_data_in_leaf": 400, "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1,
    "lambda_l2": 1.0, "max_bin": 127, "num_threads": 32, "verbose": -1, "seed": 42,
    "deterministic": True, "force_row_wise": True}
```
Protocol: fold = numeric S1 id % 10. `train_cv`: cv folds 1,2,3, eval fold 0. Round count picked once by early stopping (patience 50) on the last cv fold, then **rounds = int(best_iteration * 1.1)** and each fold model trains WITHOUT early stopping on the other two folds. OOF predictions on cv folds + fold-averaged predictions on the val fold are both saved (`_oof.parquet`, `_valpred.parquet`) — these are exactly what stage 2 consumes (no leakage).

**The "16 collective features" (verbatim `ber/features.py::collective_batch`, `COLL_NAMES`):** computed per S1 group over its candidates, given stage-1 `p1`:
```python
COLL_NAMES = ["col_p_anchor", "col_a_tset", "col_n_tset", "col_a_eq", "col_n_eq", "col_same_src",
              "col_num_jacc", "col_n_sib_addr90", "col_n_sib_name90", "col_n_conf",
              "col_wsupport_addr", "col_wsupport_name", "col_num_agree_conf", "col_best_sib_same_src",
              "col_best_sib_other_src", "col_isolated"]
```
Exact formulas: order candidates by p1 desc; `conf = {p1>=0.5}`; `support_pool = top-15 of {p1>=0.05}`; **anchor = the S1's best OTHER candidate by p1**. Then per candidate i (target ti):
- 0 `col_p_anchor` = p1[anchor]; 1 `col_a_tset` = token_set_ratio(ad[ti], ad[anchor]) (−1 if either empty); 2 `col_n_tset` = same on names; 3/4 `col_a_eq`/`col_n_eq` = exact normalized addr/name equality with anchor; 5 `col_same_src` = src[ti]==src[anchor]; 6 `col_num_jacc` = Jaccard of address-number sets vs anchor (−1 if both empty). All −1 when no anchor.
- 7/8 `col_n_sib_addr90`/`_name90` = # confident co-candidates (p1≥0.5, j≠i) with token_set_ratio ≥ 90 on addr / name; 9 `col_n_conf` = #conf excluding self.
- 10/11 `col_wsupport_addr`/`_name` = Σ_j p1[j]·tsr(ti,tj)/100 ÷ Σ_j p1[j] over support_pool (p1-weighted mean sibling similarity).
- 12 `col_num_agree_conf` = fraction of confident siblings sharing ≥1 address number; 13/14 best sibling addr-tsr split by same/other source; 15 `col_isolated` = 1 if conf nonempty but NO sibling hits ≥90 on either field.
Rationale (their docstring): "Copies of one entity share a corrupted base record, and sibling distractor entities form their own coherent copy cluster."

**Stage-2 input** = all ~89 stage-1 features + `p1` + the 16 above + p1-context (`stage_collective.py`):
```python
pl.col("p1").rank("ordinal", descending=True).over("q").alias("p1_qrank"),
(pl.col("p1").max().over("q") - pl.col("p1")).alias("p1_qgap"),
pl.col("p1").sum().over("q").alias("p1_qsum"),
```
plus optional `p1_tmargin` (per-target competition on p1 — requires features for ALL 10 folds; **final.json ships `p1_competition: false`**, so their 0.98703 does NOT use it). Stage-2 is the same `train_cv` (same params, fresh 3-fold cross-fit). Measured: stage-1 0.9846 → cross-fit ens 0.98492 → **stacking +0.0021 → 0.98703**.

### 1d. Decision rule — `ber/decide.py`

- `exclusive_mask` (verbatim semantics): "True for pairs whose q is the (first) argmax-p S1 for their target t" — sort by (t, p desc, q asc), keep the first row per target. Pure many-to-one exclusivity, computed **over the whole predicted graph** (OOF+val+other folds pooled, so val sees the same competitor set as test will).
- Final rule = `p >= tau` AND exclusive, **τ picked on stage-2 OOF** (`tau_oof`, grid 0.40–0.95 step 0.05), never on the eval fold. τ = 0.7 at train density, **0.8 under P-dense** (their test-density protocol). `configs/final.json: "tau": null` → chosen on OOF at run time.
- They implemented and rejected two fancier rules, both fully coded: `expected_f_select` (per-S1 Bayes-optimal top-k under E[F0.5] with Poisson-binomial DP, `extra_miss` = measured blocking-miss mean added as expected FN) and `pi_select` (hurdle: S1-level match-exists LightGBM `pi` on aggregates p_max/p_2nd/p_3rd/p_sum/n_p50/n_p20/n_p05/n_cand + feature-at-best; select non-empty iff `pi * E[F|nonempty] > 1 - pi`). VERDICT (FINAL_RECOMMENDATION): "expected-F and π-hurdle did not beat the tuned threshold. Singleton accuracy ≈ 0.98."

### 1e. Density-matched training (P-dense) — `stage_features.py --drop`

Verbatim `configs/final.json`: `"dense_drop": {"folds": [4,5,6,7,8,9], "frac": 0.311}`. Code:
```python
s1f = ... (pl.col("entity_id").str.slice(3).cast(pl.Int64) % 10).alias("fold"),
        ((pl.col("entity_id").str.slice(3).cast(pl.Int64) // 10) % 1000).alias("h")
dropped = s1f.filter(pl.col("fold").is_in(dcfg["folds"]) & (pl.col("h") < int(1000 * dcfg["frac"])))["rid"]
C = C.filter(~pl.col("q").is_in(dropped))
# re-rank r_rev among surviving S1s (the rev view had ranked the dropped owner too):
pl.when(pl.col("r_rev") < 999).then(pl.col("s_rev").rank("ordinal", descending=True)
   .over(pl.col("t"), pl.col("r_rev") < 999).cast(pl.Int16) - 1).otherwise(pl.col("r_rev"))
```
i.e. **delete 31.1% of the S1s of folds 4–9 (hash-deterministic) but keep their true targets in the pool as orphan distractors**, then rebuild r_rev and the S1-name-frequency feature (`q_nm_freq_s1` counts only surviving S1s). Train folds 0–3 features are built under this world (`train_folds_features: "0,1,2,3"`, `train_feat_tag: v1_dense`). Effect measured: 0.98343 → **0.98479** (train the matcher at test density beats retuning only). This is the reason τ moves 0.7→0.8.

### 1f. Normalization (short) — `ber/records.py`

Fields per record: `nm, nm_a, nm_b (alias parts of "X dba Y"), nm_dom (domain/@handle core), nm_scr, ad, ad_num (numeric tokens, leading zeros stripped: n.lstrip("0") or "0"), ad_scr`. Leetspeak repair only on mostly-letter tokens (`0→o,1→l,3→e,5→s,4→a,7→t`); NFKC + zero-width strip; alias split regex covers dba/aka/fka/trading as/t/a; native→Latin token dictionary LEARNED from training pairs (covers 96.4% of test native tokens); adjacent-duplicate token dedup.

### 1g. Submission status
Their 0.98703 is local CV. `sub_v1` (stage-1, τ 0.7+excl) validated; "Local estimate 0.983–0.985". The dense stacked sub_v2 was stopped by a Modal spend limit — it may never have hit the public LB.

---

## 2. Tony-AJ/business_entity_resolution — public LB 0.961 (measured)

### 2a. Decision rule semantics (verbatim `src/entity_resolution/decision.py`)

```python
@dataclass(frozen=True)
class DecisionRule:
    tau_abs: float = 0.5      # absolute floor: keep a pair only if prob >= tau_abs
    tau_rel: float = 0.0      # relative floor: prob >= tau_rel * p_max of its S1 entity
    tau_single: float = 0.5   # singleton floor: nothing is kept unless p_max >= tau_single
    max_matches: int = 11     # cap per S1 entity (train maximum is 11 matches)
    one_to_one: bool = True   # a pool record is kept only for its highest-prob S1 entity
```
Order of application (module docstring): **1) pool-side 1-to-1 first** (each pool record stays only with the S1 scoring it highest; ties: prob desc, then S1 id, then pool id, stable); **2) per-S1 thresholds** `prob >= tau_abs AND prob >= tau_rel * p_max AND p_max >= tau_single`; **3) cap** `rank < max_matches`. So in the tuned submission-04 triple (LEADERBOARD.md line 64): **tau_abs 0.725** = per-pair floor; **tau_rel 0.7** = drop any pair scoring < 0.7 × the S1's own best surviving prob (kills weak tails behind a strong best); **tau_single 0.775** = the S1 emits NOTHING unless its best surviving prob ≥ 0.775 (explicit singleton switch, `single_delta = tau_single − tau_abs = 0.05`); max_matches 11; one_to_one True. Tuning: full grid 31·7·6·3 = 3,906 rules × macro-F0.5 from count arrays, then tau_abs refined ±0.03 in 0.005 steps; ties go to the most conservative rule.

### 2b. Mock world (verbatim `src/entity_resolution/mock.py`)

Why: fixed val fold (441k S1 vs 2.06M pool) sits at train density; test is 1.73M S1 vs 10.0M pool → "a test entity meets 3x (US) to 6x (India) more same-name records… ~40% of the test pool has no S1 owner at all (26% in train). The first two uploads scored 0.031 below val."
`build_mock` per country: (1) keep S1-clusters and unowned pool records by id-hash with probability `frac = min(1, test_pool_c / train_pool_c)`; (2) drop kept S1 entities (matcher-trained ones first, then fit-side by hash) until `pool_per_s1` matches the test ratio — **their true records stay in the pool as unowned decoys**; (3) every surviving S1 is scored and competes in the pool-side 1-to-1. Roles: rules tuned on `tune` entities, scored on `val`; val/tune entities never dropped. Seeds 5151/5152, order-independent.
**Calibrated public estimator (verbatim constants):**
```python
# public = 1 - L_FN - FP_WEIGHT * L_FP - PUBLIC_OFFSET; fits 0.955/0.961 vs mock 0.9677/0.9704
FP_WEIGHT = 1.45      # a false merge costs 1.45x more on test
PUBLIC_OFFSET = 0.0072
```
Their measured mock→public offset shrank 0.0127 → 0.0094 after retuning the τ triple on the mock (TRACKER).

### 2c. Stage-2 record-side features (verbatim `src/entity_resolution/stacking.py`)

```python
STACK_COLUMNS = ["p1", "s1_rank", "s1_best_other", "s1_gap", "s1_p1_sum", "s1_n_likely",
                 "pool_rank", "pool_best_other", "pool_gap", "pool_p1_sum", "pool_n_likely", "pool_degree"]
...
rank, best_other = group_stats(pool, p, len(pool_ids))   # group = the pool record, across S1 entities
cols["pool_rank"], cols["pool_best_other"] = rank[rows], best_other[rows]
cols["pool_gap"] = cols["p1"] - cols["pool_best_other"]  # > 0 only for the record's best entity
cols["pool_p1_sum"] = np.bincount(pool, weights=p, ...)[pk]   # "sum well above 1 means ambiguity"
cols["pool_n_likely"] = np.bincount(pool, weights=(p>=0.5), ...)[pk]
cols["pool_degree"] = np.bincount(pool, ...)[pk]         # S1 entities holding this record as candidate
```
**`pool_gap` = p1 − (best p1 of the SAME pool record under any OTHER S1)** — the record-side analogue of Ayan's `c_combo_tmargin` but on stage-1 probability. `group_stats` is fully vectorised (lexsort + first/second flags; the group's best row sees the runner-up as its rival). Computed over the WHOLE partition ("a subset would miss rivals").
Also coded: `anchor_features` (anc_p1/anc_addr_ts/anc_addr_ratio/anc_name_ts/anc_nums_eq vs the S1's best other candidate, NaN sentinels), `cohesion_features` (p1-weighted mean sibling similarity + support count over ALL within-entity pairs), `rival_features` (the pool record compared against its best rival S1's own name/address: riv_addr_ts, riv_name_ts, riv_addr_gap = own_addr_ts − riv_addr_ts).

DELTA vs ours: we have sibling vouch and rev_rank; we do NOT have `pool_p1_sum`/`pool_n_likely`/`pool_degree` (record-side ambiguity mass) or `rival_features` (compare the record against the rival S1's OWN fields, not just its p). And our GFM decision has no tau_rel/tau_single equivalents.

---

## 3. priyanshiiitr/amazon-ml-challenge — the +0.011 error-analysis fixes (exact code)

### 3a. Empty-field NaN (verbatim `src/pairfeat.py`)
```python
# An empty address is MISSING data, not disagreement. token_set_ratio("x","") returns 0 ...
addr_missing = (la == "") | (ra == "")
for tag in ("tsr", "tso", "rat"):
    v = _cpdist(la, ra, _SCORERS[tag])
    v[addr_missing] = np.nan
    f[f"addr_{tag}"] = v
...
f["addr_empty"] = ((la == "") | (ra == "")).astype(np.int8)   # plus an explicit flag
```
Also NaN'd: addr_tok_inter/jacc, addr_num_inter/jacc/conflict/conflict_rate. (We have this — parity.)

### 3b. Mid-token unit numbers (verbatim)
```python
def numset(s):
    # ANY token containing a digit, not just tokens starting with one.
    # ... "room no 6 404 d1" vs "404 d6" scored 0.999 because neither token was extracted.
    return {t for t in s.split() if any(c.isdigit() for c in t)} if s else set()
...
num_c[i] = len(sa ^ sb)   # tokens on ONE side only: positive evidence AGAINST a match
f["addr_num_inter"], f["addr_num_jacc"], f["addr_num_conflict"], f["addr_num_conflict_rate"]
```
Two parts: digit-ANYWHERE token extraction (catches d1/c2/e50 unit designators), and a **symmetric-difference conflict count** — disagreement as its own signal, not just low overlap.

### 3c. Token-IDF name-rarity (verbatim `src/tokidf.py` + pairfeat)
DF built once per split over that split's **Source 2/3** `s_name` tokens (train and test each use their own stats). Per pair, on the S1 (left) normalized name:
```python
vals = [log(1.0 + n_docs / max(df.get(t, 1), 1)) for t in set(s.split())]
f["name_idf_min"], f["name_idf_mean"], f["name_idf_max"]  # NaN when unknown/empty
```
Note this is **record-level rarity of the name itself** ("is this name generic?"), distinct from overlap-IDF features — it tells the model an exact match on "new delhi india" is weak evidence. `name_idf_min` (the most generic token) is the flag.

### 3d. rev_rank (verbatim `src/reverse.py`)
Reverse retrieval S2/S3 → S1 index (top-3 default, per country, IDF-weighted key channels). Per pool-record shortlist sorted by score:
```python
order = np.lexsort((-sc, qi)); ...; rank = (arange - group_start)   # 0 = best
parts.append(pd.DataFrame({"source1_entity_id": s1_ids[si], "cand_id": ...,
                           "rev_score": sc, "rev_rank": rank}))
```
Feature semantics (pairfeat comment): "rev_rank==0 means this entity is the record's single best owner among all 2.2M Source 1 entities", passed with `rev_score, fwd_hit, rev_hit, both_hit`. HANDOFF: top feature by 5×, singleton 0.871→0.904. (We have rev_rank — parity; check we also carry rev_score + fwd/rev/both_hit provenance flags.)

Their `pairfeat.py` also has our-style anchor/transitivity (`skel_vs_anchor`, `addr_vs_anchor`, `anchor_gap_*`) and a TOPM=16 within-entity **cluster-support graph**: pairwise tsr over the top-16 candidates, edge if max(name,addr) ≥ 85 → `clus_deg`, `clus_mass` (sum of sim_block of connected siblings), `clus_deg_rel`. Basis stat: co-matched pairs max(name,addr) sim averages 97.2 vs 46.7 random.

---

## 4. Other ≥0.98-local repos

### Disastrio/BuisnessEntityResolutionML (local 0.9818 single / 0.9820 dual)
`src/config.py` verbatim knobs: `USE_HARD_NEGATIVES=True, HARD_NEGATIVE_WEIGHT=3.0` (hard negatives = plausible-on-one-field non-matches, upweighted 3× via sample_weight); `USE_CONSERVATIVE_RULES=True, RULE_THRESHOLD_BOOST=0.10` (risky pairs — near-identical name but zero address/postal confirmation — need τ+0.10, soft reject not hard); per-source S2/S3 thresholds and dual models coded but shipped OFF; Soundex(4) + rare-trigram blocking tiers; LightGBM lr 0.05 + auto `scale_pos_weight=n_neg/n_pos`; t*≈0.72. Nothing here beyond Ayan/Tony except the **hard-negative 3× upweighting** and the **+0.10 risky-pair boost**, both trivially portable.

### Skullybutcher/ml_challenge_2026 (OOF 0.9844–0.9847)
`kaggle_run.ipynb` is a thin driver (`--n 5000 --sample-s1 20000`, actual src in a git submodule). The 0.984x is a **5k-sample OOF** — sample-optimistic by construction (fewer same-name distractors). Ignore for v10.

---

## IMPLEMENT IN V10 — ranked gaps

1. **Density-matched TRAINING world (Ayan 1e) — +0.0014 measured, and it moves τ.** Port the `dense_drop` snippet verbatim: hash-drop ~31% of non-eval-fold S1s, keep their targets as orphan distractors, re-rank r_rev among survivors, recompute S1-name-frequency over survivors, train (not just tune) there. Our LOCO-gated selftrain and per-country isotonic tune at TRAIN density today; every strong repo found decisions move (0.7→0.8, Tony's whole mock apparatus) — and Tony measured a 0.031 val→public gap before density matching, 0.0094 after.
2. **True stage-2 stacking on OOF p1 with the 16 collective features + p1 context (Ayan 1c) — +0.0021, the largest modelled gain any public team measured.** Our "stage B vouching" vouches; theirs feeds `col_*` + `p1_qrank/p1_qgap/p1_qsum` + Tony's record-side `pool_gap/pool_p1_sum/pool_n_likely/pool_degree` + `rival_features` (riv_addr_ts/riv_name_ts/riv_addr_gap) into a second cross-fit LightGBM with the exact params above. Snippets: `collective_batch` + `context_feats` (Ayan), `competition_features` + `rival_features` (Tony) — all verbatim above.
3. **Reverse target→S1 TF-IDF combo view as a BLOCKING view, K_rev=8, plus per-view rank-cap prune {rev 8, combo 10, name 5, cat 5, addr 5} (Ayan 1a).** rev R@8 = 0.98746 ALONE; union 0.9905 → pruned 0.989 at ~45/S1. If our prune-28 record-side recall is below ~0.985, this is the ceiling-raiser: `hstack_views` + `topk(Tx, Qx, 8)` — two functions, verbatim above. Measure our union recall first; oracle F0.5 at their recall is 0.9965–0.9972.
4. **Decision-rule upgrades (Tony 2a + Ayan 1d):** add `tau_rel` (≥0.7×p_max) and `tau_single` (p_max floor, +0.05 over tau_abs) axes to our GFM/threshold tuning, apply pool-side 1-to-1 BEFORE thresholds, cap at 11. Warning from Ayan's measurements: expected-F/π-hurdle (≈our GFM) did NOT beat a plain OOF-tuned τ + exclusivity — benchmark our GFM head-to-head against `thr_τ_excl` under the density-matched world before trusting it.
5. **Cheap feature ports (priy 3b/3c, Disastrio):** digit-ANYWHERE numset + `addr_num_conflict` (symmetric difference); record-level `name_idf_min/mean/max` of the S1 name from S2/S3 DF (rarity of the name itself, not overlap); house-number geometry (min absdiff / reldiff / digit-edit, Ayan TOK 25–29); `t_nm_freq_t`/`q_nm_freq_s1` exact-name ambiguity counts; hard-negative 3× sample weight; +0.10 τ boost for name-high/addr-absent pairs. Each is ≤20 lines, all verbatim above.
