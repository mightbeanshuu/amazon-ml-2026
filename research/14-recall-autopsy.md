# 14 — Recall autopsy: why v10 India blocking misses 2.9% of true pairs (26–27 Sep 2026)

Status: DONE (27 Sep ~00:15 IST). All numbers [RV] on real train data unless marked. Scripts: `research/14-autopsy-scripts/`.

## Headline
1. **The biggest India recall problem is not a blocking miss. It is a train/test mismatch.**
   - In train, the two record-side protections (K-stage record-top-2, prune rev ≤ 2) rank each record's suitors among
     the 120k SAMPLED S1s only. At test they rank among all S1s.
   - **CORRECTED 27 Sep ~02:00 (§4c).** The first version of §4 applied test's K-stage record-top-2 over ALL S1s.
     v10 test applies it per S1 quarter (`run_big.py:500` → `block_big.py:249–280`). With quarters, **v10's
     test-time India post-prune PC ≈ 0.958–0.959 (oracle ≈ 0.985)**, not the train-measured 0.9675 (oracle 0.9886).
     The retracted figure was 0.948.
   - research/15 measures 0.9618 with the same semantics. §4c attributes most of the 0.003 gap to 15's harness
     keeping only the top-3 reverse competitors per family, plus visible-world quarter density.
   - T1a (grouped protection at test) restores the train value in expectation: **≈ +0.8–1.0 India PC pts, ≈ +0.003
     India oracle**, no retrain.
2. **The 11,877 pre-prune misses (2.86 pts):**
   - 46% were FOUND by a family and then cut by the K=40 cap: A 28.6% certain, A2 17.5% namesake ties.
   - 27% are far-rank namesake crowding (F).
   - 22% are near/mid rank (D+E).
   - 5% are max_df-killed random names (C).
   - Common profile: the S1's exact name is shared by a median **42** visible S1s (retrieved positives: 2). Records
     are over-represented for empty address (4.2×), native script (2.6×) and generator-random names (3.9×) (§2–3).
3. **Knobs, by post-prune India PC recovered:**
   - T1a ≈ +0.8–1.0 (corrected; 15 measures +0.6 to +1.5 for its T1a variants).
   - Ayan reverse-combo top-8 with protection +1.38, of which only +0.51 is new beyond fixing the K cap.
   - Exempting each family's rank-1/2 pairs from the K cap +0.71/+0.83. Sampling-invariant and cheap.
   - Family k ×2/×3 +0.17/+0.27.
   - Sibling-gate loosening ≤ +0.26.
   - max_df ≤ +0.04.
   - Residual Bayes-hard: 0.77 pts (§5, §7).
4. **The v10 similarity prune costs only 0.39 pts in train (v4's 2% is obsolete).** But 96% of what it drops is
   exact-name + empty-address namesake records, and its rev ≤ 2 rescue is the skewed protection of point 1 (§1).

## 0. Setup and ground truth of the miss set

- Source of truth for the miss set: the REAL v10 artifact `~/out/train_block_india.parquet` + `.json` copied off Box 1
  (217 MB; no compute on the box). It holds the 9,834,279 candidate pairs (after per-family top-k, K=40 cap and
  sibling expansion) for the 120,000 sampled visible India S1s, with `y` labels. Its JSON: 415,630 GT positives,
  PC 0.9714, oracle 0.9898.
- Box code == local `code/ber_v5/src` (md5 identical for every .py and lexicon.json), so local re-computation
  uses the exact v10 functions.
- Raw text: `SCR/real/train_source{1,2,3}.tsv` + GT (already on the Mac; nothing re-downloaded).
- Scripts: `SCR/autopsy/` (SCR = the session scratchpad).
- **Miss set reproduced exactly** (`SCR/autopsy/miss_set.py`): R order rebuilt from raw TSVs (S2 india then S3
  india, file order) = 4,133,346 records = box `n_r`; visible/sample split rebuilt with `train_split` (seed 0) and the
  120,000 sampled S1 ids equal the box JSON keys. GT positives 415,630; in candidates 403,753;
  **missed 11,877 pairs → PC 0.97142 (box: 0.9714)**. The misses sit on 9,705 S1s; **541 S1s lost ALL their
  records** (F = 0 each under a perfect matcher). 964 retrieved positives came only from sibling expansion.
  This is the full population of v10 India pre-prune misses on the train sample, not a sub-sample; binomial error
  on a fraction p of the 11,877 is ±1.96·sqrt(p(1−p)/11877) ≤ ±0.9 pp.

## 1. The similarity prune on v10 (exact replay, `SCR/autopsy/prune_diag.py`)

Replayed `prune_by_similarity` (prune_k 28, rev_rank ≤ 2) on all 9,834,279 real v10 India pairs [RV]:

| rule | pairs/S1 | PC | share of blocked positives kept | oracle macro-F0.5 |
|---|---|---|---|---|
| none (blocking only) | 81.95 | 0.9714 | 1.0000 | 0.9898 |
| **v10: k=28, rev≤2** | 67.18 | **0.9675** | 0.9960 | **0.9886** |
| k=40, rev≤2 | 71.90 | 0.9693 | 0.9979 | 0.9892 |
| k=56, rev≤2 | 75.86 | 0.9706 | 0.9991 | 0.9896 |
| k=28, rev≤4 | 76.42 | 0.9693 | 0.9978 | 0.9892 |
| k=28 + any family's in-table rank ≤ 5 | 69.68 | 0.9687 | 0.9972 | 0.9890 |
| k=40, rev≤3, any-family rank ≤ 2 | 76.47 | 0.9701 | 0.9987 | 0.9894 |

- The v10 prune loses **1,612 positives (0.40% of blocked; −0.0039 PC, −0.0012 oracle)**, not v4's 2%.
- **Who it loses: 96% have address token-set similarity 0** (in practice an empty address on one side; 3.5% of all
  positives have similarity 0), with a
  name score of 100 (median). Exact-name, empty-address records of an S1 that has ≥ 28 namesake candidates whose
  address overlaps a little. Their within-S1 prune rank: median 42, p75 54, p90 74. Their rev rank among the sampled
  S1s: 3 (410), 4 (315), 5 (214), 6 (181)…
- **CAVEAT — the train measurement is optimistic for test [RV, mechanism].** Both record-side protections are
  computed on a pair table that holds only the 120k SAMPLED S1s (16.8% of the 715k visible S1s):
  - `prune_by_similarity`'s `rev_rank ≤ 2`: 6,134 positives (1.5% of positives) survive ONLY through it
    (their within-S1 rank is > 28). With no rev protection at all, PC after the prune would be **0.9528**.
  - `prune_topk`'s record-top-2 (K=40 cap): 9,667 family-proposed positives (2.4%) have a K-score rank > 40 and
    are in the table only through record-top-2 among sampled S1s.
  - At test the pair table holds every S1, so a record has ~6× more suitors and both protections bite much less.
    Removing both protections entirely (keep a positive only if K-score rank < 40 AND prune rank ≤ 28) gives
    PC **0.931**. True test-time India PC after the prune therefore lies between **0.931 and 0.9675** (and the
    pre-prune PC is below the train 0.9714). §4 estimates where in that band it falls.
    The same bias inflates train mean_cands (81.95 vs v4's 49) and trains the matcher on a candidate distribution
    richer in protected empty-address pairs than test will have.

## 2. What the missed records look like (text noise modes; all 11,877 misses vs 3,000 retrieved positives)

Flags come from raw text plus the pipeline's own normalised fields (`SCR/autopsy/classify.py text`).
Control = 3,000 random retrieved positives proposed by a family. Binomial 95% half-width: ≤ ±0.9 pp (missed),
≤ ±1.8 pp (control).

| record property | missed | retrieved | over-representation |
|---|---|---|---|
| record address empty | 16.7% | 4.0% | 4.2× |
| name in a native Indian script | 39.9% | 15.4% | 2.6× |
| name is a generator-random string (≥ 99% of tokens unseen in S1 names) | 14.5% | 3.7% | 3.9× |
| name is a #handle / domain | 8.8% | 5.2% | 1.7× |
| **true S1's exact name shared by ≥ 40 visible S1s** | **55.9%** | 24.0% | 2.3× |
| median # visible S1s sharing the true S1's exact name | **42** | 2 | — |
| median # address word tokens in the record | 4 | 8 | — |
| leet digits in name | 1.6% | 2.7% | 0.6× (leet is solved) |
| letter-scramble of the S1 name | 1.2% | 1.1% | 1.1× |

Name relation (record name vs S1 name): exact 10.4% (retrieved 62.3%), typo/near 35.3% (22.4%), partial 22.6%
(6.4%), **different / renamed / random 20.1% (4.2%)**, phonetic/transliteration 9.0% (2.9%), shuffle 1.0% (1.0%).
Address relation: token-set ≥ 80 in 57.8% (retrieved 90.5%) — mostly because the record's address is a short
SUBSET of the S1's (city/state/number), which token-set scores as 100; empty 16.7%.

Reading: the typical miss is **a degraded copy (native script, random or partial name, or a 4-token address) of an
S1 whose name is shared by dozens of namesakes.** Leet, shuffle and scramble are already handled.

## 3. Why blocking missed them: mechanical causes at FULL distractor density (`SCR/autopsy/fam_diag.py`)

For each of the 11,877 missed pairs and each of the 6 v10 families, with the exact v10 key builders, IDF weights and
per-family max_df over all 715,441 visible S1 + 4,133,346 India records: the pair's cosine, its forward rank (#records
beating it for the S1) and reverse rank (#visible S1s beating it for the record), and shared-key accounting.
**Harness validation on 3,000 retrieved positives:** b_key membership matches the box table 3000/3000 and every
family's cosine matches the box value to ≤ 3e-7. Name-only families (b_nkey, b_pkey) have massive exact ties
(identical namesake key sets); a "strict" rank (ties broken in our favour) over-claims, a "ge" rank (all ties ahead)
never falsely claimed a retrieval on the controls. Categories below use the certain "ge" rank first.

| cause (mechanical) | n | share of misses | PC points | record addr empty | native-script name | random name | median namesakes of S1 | Ayan view R@8 | would survive prune (rank ≤ 28) if retrieved |
|---|---|---|---|---|---|---|---|---|---|
| **A: retrieved by a family, then CUT by the K=40 cap** (`prune_topk`) | 3,396 | **28.6%** | 0.82 | 7% | 43% | 4% | 43 | 92% | 86% |
| A2: true S1 sits in a cosine TIE that straddles a family's k (namesake tie) — or A | 2,077 | 17.5% | 0.50 | **33%** | 32% | 6% | 46 | 52% | 59% |
| D: near miss, best family rank ≤ 2× its k | 1,213 | 10.2% | 0.29 | 20% | 35% | 13% | 34 | 66% | 63% |
| E: mid miss, 2–5× k | 1,356 | 11.4% | 0.33 | 16% | 46% | 13% | 40 | 62% | 66% |
| **F: far miss, > 5× k in every family (namesake crowding)** | 3,260 | **27.4%** | 0.78 | 17% | 43% | 24% | 40 | 36% | 56% |
| C: shares keys, but every shared key is killed by max_df | 572 | 4.8% | 0.14 | 6% | 28% | **59%** | 9.5 | 9% | 30% |
| B: no shared key in any family | 3 | 0.03% | 0.00 | — | — | — | — | — | — |

(PC points = pairs / 415,630 × 100. "Would survive prune" = its prune score ≥ its S1's 28th-best score, i.e. without
record-side rev protection — the test-like condition; 4–5% per row have no S1 threshold available and are excluded.)

Families that certainly retrieved the A pairs: b_key 1,958, b_cgram 1,921, b_xkey 1,877, b_akey 441, b_nkey 134,
b_pkey 113 (overlapping). Best family for D/E/F: b_cgram 2,915, b_xkey 1,514, b_key 871.

## 4. Where test-time PC really sits — the sampled-S1 protection skew (priority item for lever T1)

> **Correction:** §4a and §4b below apply the test K-stage record-top-2 over ALL visible S1s. That is wrong for
> v10, so their "as-is" rows (0.948, oracle 0.981) are **retracted**. The corrected estimate is in §4c. The mechanism,
> the prune-stage numbers and the T1a-by-construction argument still stand.

**Mechanism [RV].** In train, `block_country(keep_s1=…)` drops every pair of a non-sampled S1 BEFORE `prune_topk`
and `prune_by_similarity` run, so both record-side protections (K-stage record-top-2, prune-stage rev_rank ≤ 2) rank a
record's suitors among the 120k sampled S1s only (a random 16.8% of the 715k visible). At test every S1 is present.

**At-risk set:** 15,234 train positives survive only through a protection (10,288 with `b_rank`==40 at the K stage,
6,134 with prune rank > 28 kept by rev ≤ 2; overlapping). For each, `fam_diag.py` collected the record's top-30 S1s
per family by exact reverse search over ALL 715k visible S1s. Two suitor sets bracket the truth:
- strict = S1s inside the record's reverse top-kr of some family (an exact subset of the real suitors);
- loose = S1s inside its reverse top-30 of some family (over-covers them).

**K stage (record-top-2 by K-score among all suitors), K-protected positives kept in train (n = 10,262)** [RV-sim,
`kstage_quick.py`]:

| suitors | protection | lost at K stage | share |
|---|---|---|---|
| strict | global (v10 test as-is) | 6,282 | 61% |
| loose | global (v10 test as-is) | 6,088 | 59% |
| strict | grouped, G=6 (T1a) | 1,728 | 17% |
| loose | grouped, G=6 (T1a) | 3,682 | 36% |

- As-is, **~6,100–6,300 train-kept positives (1.47–1.51 PC points) would be cut at the K stage at test**, before the
  similarity prune. Test pre-prune India PC ≈ 0.9714 − 0.015 ≈ **0.956**, not 0.9714.
- Grouped rows are NOT the T1a gain. The at-risk set is conditioned on having survived THIS train sample's
  realisation, and a fresh grouping re-draws it (some survive, and train-lost positives can survive instead).
  Because grouping by `s1_i % G` with ~120k S1s per group gives each competitor the same 1/G ≈ 0.168 inclusion
  probability as train sampling, **T1a's expected PC equals the train-measured PC (0.9714 pre / 0.9675 post-prune) by
  construction.** T1a's gain = train-measured PC − test-as-is PC.
- The prune-stage estimate (text scores against all suitors) is in the next block.

**Both stages, all 15,208 at-risk positives kept in train** [RV-sim, `atrisk_sim2.py`, 11 s, peak RSS 0.75 GB]. The
prune score is recomputed as the pipeline does (max name sim + address token-set + 50·b_key) against every suitor:

| suitors | protection | lost at K | lost at prune (after K ok) | lost total | India PC after prune |
|---|---|---|---|---|---|
| strict | **global = v10 test as-is** | 6,282 | 1,643 | 7,925 | **0.9485** |
| loose | **global = v10 test as-is** | 6,088 | 2,026 | 8,114 | **0.9480** |
| (train, sampled-S1 protection) | — | 0 | 0 | 0 | 0.9675 |

- **Estimate: v10's test-time India post-prune PC ≈ 0.948** (the two suitor brackets agree to 0.0005), against 0.9675
  measured in train. **Oracle macro-F0.5 as-is ≈ 0.9812–0.9813 vs 0.9886 in train (−0.0074 India ≈ −0.0035 of
  global ceiling).** T1a (grouped protection) restores the train value in expectation: **≈ +0.019 India PC, +0.0074
  India oracle** (the realised matcher gain is smaller than the oracle gain).
- Direction of residual error:
  - Test India has 13% more S1 than the train-visible world (809,986 vs 715,441), so there is slightly more
    competition: as-is could be ~0.946.
  - At test, fewer K-protected pairs sit above a positive in its S1's post-K list, so a few prune ranks improve: as-is
    could be slightly better.
  - These two partly cancel.
- **T1a flooding check.** In train, protection-only pairs are the bulk of the table:
  - K stage: 4,210,667 of 8,963,753 family pairs (47%, 35/S1) have K-rank > 40.
  - Prune stage: 67.2 pairs/S1 kept against ≤ 28 by rank.
  - T1a at test reproduces those volumes: ~75–82 candidates/S1 after K and ~67/S1 after the prune (train numbers).
    As-is test keeps at most 2 suitors per record, i.e. roughly 28–40/S1.
  - **So T1a roughly doubles India test candidates and India feature time (≈ ×1.7–2.4). It does not flood beyond
    what the matcher was trained on**: the candidate distribution becomes the training distribution again.

### 4c. Corrected: v10's test K stage is per S1 QUARTER (flagged by the integrator from research/15)

**Code (confirms the flag):**
- Train (`run_big.py:269`, `:276`):
  - `block(s1_full, r, cfg, keep_s1=keep)` runs with no `post`.
  - `prune_topk(pairs, K)` then runs ONCE over the whole sampled table.
  - So the record-top-2 pool is all 120k sampled S1s (16.8% of visible).
- Test (`run_big.py:500`):
  - `block(s1, r, cfg, post=lambda d: prune_topk(d, K))`.
  - `block_country` calls `post` once per S1 index range (`block_big.py:249–280`, `s1_parts=4`).
  - So the test record-top-2 pool is the true S1's quarter: 202k of 810k S1s. Relative to the train sample that is
    ≈ 1.5× (visible world) to 1.7× (test density), not 6×.
- The prune stage is global at test: `run_big.py:522` calls `prune_by_similarity` on the whole country table, so
  `rev_rank ≤ 2` ranks among every S1 whose pair survived K.

**Is the S1 order random with respect to namesakes?** Yes. 24.4–24.9% of a record's competitor suitors fall in the
true S1's quarter (25% expected).

**Re-simulation** (`atrisk_sim3.py`, `kstage_ipw.py`, `kstage_fix.py`; strings re-normalised for only the 441k needed
S1s + 15k records by `norm_subset.py`, and every true-pair prune score matches the table exactly, 15,208/15,208):
- **K-stage inflow.** A quarter is a different random subset from the train sample, not a superset. So positives the
  cap cut in train (class A) can survive at test.
  - Estimated by inverse-probability weighting on nb = number of competitors with a higher K-score:
    P(kept) = P(Bin(nb, q) ≤ 1), q_train = 0.168, q_test = 0.25 (0.283 at test density).
  - Calibration: the kept set implies 2,863 positives cut by the cap in train, against ≥ 3,396 observed (A).
    So nb is right or slightly low, and the estimate is if anything optimistic.
- **Competitor K-score** is built exactly as research/15's `testmode.py` does: only families listing the competitor
  within their reverse top-kr; b_key cosine, else 0.5 × best such cosine.

| stage (strict suitors = reverse top-kr per family) | q = 0.25 (quarters, visible world) | q = 0.283 (test density) |
|---|---|---|
| K stage: at-risk positives lost | 2,472 | 2,810 |
| K stage: train-cut positives rescued (inflow) | +940 | +765 |
| **K stage net** | **−1,532 (−0.37 pts)** | **−2,045 (−0.49 pts)** |
| prune stage (rev ≤ 2 over all K survivors; bracket on competitor K survival) | −1,600 to −1,800 (−0.39 to −0.43) | −1,590 to −1,790 |
| **India PC after prune (inflow × 0.86 prune survival)** | **0.9588–0.9593** | **0.9576–0.9581** |
| oracle macro-F0.5 (realised quarter draw, no inflow) | 0.9847–0.9849 | — |

**Estimate: v10 test India post-prune PC ≈ 0.958–0.959, oracle ≈ 0.985** (train 0.9675 / 0.9886).

**Against research/15 (0.9618, oracle 0.9867):** the prune-stage losses agree (≈ 0.4 pts). The K stage differs:
15 loses 0.13 pts, this estimate 0.37–0.49. Two concrete causes:
1. **15's reverse competitor lists are truncated at top-3 per family.**
   - `pricing/blk.py:167–197` stores `rtop_i` with width 3 (`m4 = rk <= 3`); `testmode.py` reads `min(stored, kr)`.
   - v10 kr is 4 (b_key) and 8 (b_cgram), so reverse-rank-4 b_key and rank-4–8 cgram competitors are unseen.
   - The same cap in this harness moves the K stage from −0.37 to −0.26 pts, and it fails the calibration check
     (implied cap-cut 1,456 vs ≥ 3,396 observed).
2. **Quarter density.**
   - 15 uses visible-world quarters: 179k of 715k.
   - Test India quarters hold 202k of 810k S1s, ≈ 13% more competitors: another −0.12 pts.
   - Both move 15's 0.9618 toward ≈ 0.9595. About 0.1 pt stays unexplained (IPW model vs 15's direct count).

**Consequences:**
- The v10 train→test India gap is ≈ 0.8–1.0 PC pts and ≈ −0.003 to −0.004 India oracle (≈ −0.0015 to −0.0018
  global), against 15's −0.0057 / −0.0019.
- T1 is larger than a "moderate" item but smaller than this doc first said.
- 15's T1a variants were measured with the same truncated lists. Their ranking should hold, but the gain of the
  K-side half (`s1_parts=24`) is probably understated by ~0.1–0.2 pts, because at ~30k ranges almost everything
  passes either way.
- Widening `rtop` to 8 (`blk.py:167`, `m4 = rk <= 8`) would settle it; the cgram re-run is ~44 min.

## 5. Knobs vs the 11,877 pre-prune misses (retrieval-level; overlaps resolved) [RV]

These counts are "pair would be proposed". Newly proposed pairs must still pass the K=40 cap and the prune.
The prune pass rates are those in §3, so every "PC pts" here is an UPPER bound unless the knob also protects its pairs.

| knob | misses recovered | share | + India PC pts (pre-prune) |
|---|---|---|---|
| Ayan reverse combo view, record→S1 top-8 (B-IN) | 7,086 | 59.7% | +1.70 |
| … of which pairs no v10 family proposed at all | 2,864 | 24.1% | +0.69 |
| fix the K=40 cap, certain (A) | 3,396 | 28.6% | +0.82 |
| fix the K=40 cap, incl. tie-straddlers (A+A2) | 5,473 | 46.1% | +1.32 |
| every family's k ×2 (certain rank), new pairs only | 1,858 | 15.6% | +0.45 |
| b_cgram rev k 8→16 / 24 / 40 | 669 / 1,164 / 1,851 | — | +0.16 / +0.28 / +0.45 |
| b_xkey rev k 2→16 / 24 / 40 | 1,390 / 1,675 / 1,995 | — | +0.33 / +0.40 / +0.48 |
| b_key rev k 4→16 / 24 / 40 | 783 / 1,222 / 2,017 | — | +0.19 / +0.29 / +0.49 |
| K-cap fix (A+A2) + Ayan@8 | 8,337 | 70.2% | +2.01 |
| K-cap fix (A+A2) + Ayan@40 | 9,270 | 78.1% | +2.23 |
| K-cap fix (A+A2) + Ayan@8 + families ×2 | 8,665 | 73.0% | +2.08 |

- **Residual after every knob above: 3,212 pairs (0.77 PC pts).** By cause: F 2,086, E 522, C 518. The records:
  38% renamed/random name, 30% partial name, 23% empty address, 28% generator-random name, median 39 namesakes.
- That residual is Bayes-hard for any name/address retrieval: the record's name carries no evidence and the S1's
  name is shared by ~40 others. Only the sibling route can reach it (§6).
- **Ayan@8 is mostly re-finding what v10 already found and then capped.** 92% of A misses are in Ayan's top-8,
  against 36% of F. B-IN's net new retrieval is +0.69 pts; the rest of its value is equivalent to fixing the K cap.

### 5a. The K=40 cap in detail (`kcap.py`, on the real table re-copied from Box 1)

- The 4,623 S1s holding A/A2 misses have **113 retained pairs each** (the cap binds hard; 47% of retained family pairs
  table-wide are record-top-2 protected).
- The median missed pair has 46 retained pairs of its own S1 scoring above it under the v10 K-score.
- 2,044 of the 5,473 A/A2 pairs were proposed by b_key. The rest came from cgram, xkey or akey, and the
  **v10 K-score halves every non-b_key cosine** (`0.5·max(alt)`).

The recovered counts below are upper bounds, because the table only holds the survivors beyond rank 40:

| K-stage change | certain A (3,396) | A+A2 (5,473) |
|---|---|---|
| K 40 → 60 (same score) | ≤ 1,865 | ≤ 3,679 |
| K 40 → 80 | ≤ 2,258 | ≤ 4,186 |
| K 40 → 120 | ≤ 2,675 | ≤ 4,671 |
| K=40, score = max(b_key, other families) (no halving) | ≤ 1,881 | ≤ 3,510 |
| **exempt every family's fwd/rev rank-1 pair from the cap** | **2,051** (exact) | 3,872 |
| exempt family rank ≤ 2 | 2,583 | 4,511 |
| exempt family rank ≤ 3 | 2,826 | 4,799 |

**"Exempt family rev rank-1" is a record-side protection that is NOT skewed by S1 sampling.** A family's reverse
rank is computed by the reverse search over ALL visible S1s, in train and at test alike. So it recovers what the
sampled record-top-2 was meant to recover, identically at train and test.
- Cost upper bound: 6 families × one S1 per record ≈ 35 pairs/S1. Most overlap with the existing top-40, so the true
  cost is lower (not measurable from the post-cap table).
- It needs `block_country` to record each pair's own reverse rank. A code change: retrain-window only.

### 4b. Refinement: shorter S1 lists at test (`prune_refine.py`)
At test, many K-protected pairs vanish before the prune, so some prune-rank-> 28 positives move up. 6,134 positives
were kept in train by rev ≤ 2 (pprot); the share back within rank 28 depends on how many K-protected pairs above
them are cut at test:

| share of K-protected pairs cut | 0.4 | 0.6 (≈ sim's 59–61%) | 0.8 | 1.0 |
|---|---|---|---|---|
| pprot positives back within rank 28 | 331 (5.4%) | **564 (9.2%)** | 837 | 1,355 |

At most ~560 of the 1,643–2,026 prune-stage losses are saved (+≤ 0.0013 PC). The 13% higher S1 density at test
pushes the other way.
~~Final estimate ≈ 0.948–0.950~~ **RETRACTED, see §4c: ≈ 0.958–0.959 (oracle ≈ 0.985).**

## 6. The sibling route (`sib.py`, on the box's own `train_india_sib.npz`)
- 98.2% of misses belong to an S1 for which blocking DID find another true record. The obvious rescue is "a sibling
  of mine found my S1".
- But the v10 sibling graph (b_key top-3 within R, plus 2-hop ≥ 0.45, cap 3) links **only 24.0%** of missed records
  to ANY true co-record. **Only 9.0% (1,069 pairs, 0.26 PC pts)** link to a co-record that was actually found for
  the S1.
- The expansion gates cut that to 3.7% (sim ≥ 0.55 + sibling's record-top-5), then to 1.5% (sibling not weak).
  76.5% of missed records are "weak", so the weak gate is not the binding one.
- Consistency check: only 1 pair passes every gate (it should be ~0, since passing would mean expansion adds it).
- **Verdict: loosening expansion gates is worth ≤ 0.26 PC pts.** The sibling graph itself misses the link. The
  sibling search is b_key-only among 4.13M records, and namesake records crowd a degraded copy's top-3.
- The same sampled-S1 skew applies at test: the "top_s1=5 by b_key" donor set is ranked among all S1s there.

## 7. Deliverable: miss cause × frequency × knob × expected PC recovery (India, v10, real train)

Populations: 11,877 pre-prune misses (2.86 PC pts, the full set, not sampled), plus 1,612 prune losses (0.39 pts),
plus the test-only protection skew. The "post-prune" column counts the pair only if it would also rank ≤ 28 by the
v10 prune score in its S1 (test-like, no rev protection), and assumes the knob exempts it from the K cap.

| # | cause | n (share) | PC pts lost | knob that fixes it | recovery: retrieval → post-prune (India PC pts) | grade |
|---|---|---|---|---|---|---|
| T | **test-only: sampled-S1 record protections are weaker at test** (K record-top-2 per S1 quarter ≈ 1.5–1.7× the train pool; prune rev≤2 over all S1 ≈ 6×) | ~3.4–4.1k net positives | **≈ 0.8–1.0 (test PC ≈ 0.958–0.959 vs 0.9675; §4c)** | **T1a: compute both protections within S1 groups of ~120k at test** (no retrain) | restores ≈ +0.8–1.0 pts post-prune, oracle ≈ +0.003 India (in expectation, by construction) | RV-sim |
| A | family retrieved it, **K=40 cap cut it** (`prune_topk`; non-b_key cosines halved) | 3,396 (28.6%) | 0.82 | exempt each family's fwd/rev rank-1 (≤ 2) pair from the cap | A+A2: +0.93 → **+0.71** (rank ≤ 2: +1.09 → +0.83) | RV |
| A2 | namesake cosine TIE straddles a family's k (or capped) | 2,077 (17.5%) | 0.50 | same as A, plus tie-break inside equal-cosine groups by a secondary field (address cosine) | included in A row | RV |
| F | far rank, > 5× k in every family: **namesake crowding** (median 40 namesakes, 43% native script, 24% random name) | 3,260 (27.4%) | 0.78 | no k fix; different view: Ayan reverse combo (R@8 36%, R@40 51% of F) | Ayan@8 over ALL misses: +1.70 → **+1.38**; of which new vs A/A2: +0.69 → +0.51 | RV |
| E | mid rank, 2–5× k | 1,356 (11.4%) | 0.33 | family k ×3–5 (best family: cgram rev, xkey rev) | k ×3, new beyond A/A2: +0.40 → +0.27 | RV |
| D | near rank, ≤ 2× k | 1,213 (10.2%) | 0.29 | family k ×2 (cgram rev 8→16, xkey rev 2→4–8, b_key rev 4→8) | k ×2, new beyond A/A2: +0.26 → +0.17 | RV |
| C | shared keys all killed by max_df (59% random names, shared token is only a common word) | 572 (4.8%) | 0.14 | raise max_df — cost-prohibitive; Ayan@8 gets only 9% | ≤ +0.14 → ≤ +0.04 | RV |
| B | no shared key at all | 3 | 0.00 | — | 0 | RV |
| P | prune: 96% exact-name + EMPTY-address records of namesake-crowded S1s, prune rank median 42 | 1,612 | 0.39 (train) | prune_k 28→40 (+0.18), rev ≤ 4 (+0.18), k=40+rev≤3+family-rank≤2 (+0.26); at test these interact with T | train-measured | RV |
| S | sibling expansion gates | — | — | loosen weak_q/min_sim/top_s1 | ≤ +0.26 (the graph misses 76% of links) | RV |

**Combined blocking knobs (overlaps resolved):** exempt family rank-1 + Ayan@8 = +1.93 → **+1.46 pts post-prune**.
Adding rank ≤ 2 and k ×2 = +2.03 → +1.51. The residual 3,212 pairs (0.77 pts) are Bayes-hard renamed/random names
of 40-namesake S1s.

**Ranking for the retrain window, by India PC per unit of work:**
1. **T1a**: test-side only, no retrain, ≈ +0.8–1.0 pts post-prune (corrected from +1.9, §4c). It fixes a
   train/test mismatch, not just a recall gap.
2. **Ayan reverse combo top-8 with protection through the K cap and the prune (B-IN)**: +1.38 post-prune. 60% of its
   value duplicates fixing the K cap.
3. **K-cap exemption of family rank-1/2 pairs**: +0.71–0.83. Cheaper than B-IN (no new view; `block_country` must
   keep each pair's own family rank), and it is sampling-invariant, so it also shrinks the T gap.
4. Family k increases: ≤ +0.27, and cgram rev k is the expensive family (44 min on the box at k=8).

**Arithmetic sanity:** train-measured post-prune PC 0.9675 → as-is test ≈ 0.958 (corrected) → with T1a ≈ 0.9675 → with 2+3
≈ 0.982 (upper; pairs still have to be matched). The India oracle ceiling rises accordingly, but these are
recall ceilings, not F0.5 gains. Most recovered pairs are namesake/empty-address cases, where §1.3 of
`10-frontier-0995.md` shows the matcher is weakest.

## 8. Reproducing
- Scripts and the per-miss table (`missed_full.parquet`, 11,877 rows: text flags, per-family cos/ranks, cause,
  Ayan rank, prune check) are in `research/14-autopsy-scripts/`. They read from `SCR/autopsy/`.
- To free Mac disk, the large intermediates were deleted after use:
  - `norm/`, 380 MB: regenerate with `norm_india.py`, ~12 min on 3 workers.
  - the box's `train_block_india.parquet` (217 MB) and `train_india_sib.npz` (85 MB): re-copy from Box 1 `~/out/`.
- Run order: `norm_india` → `miss_set` → `prune_diag` → `fam_diag` (6 families ≈ 8 min, peak ~1.3 GB, under
  `ramlock.sh`) → `classify text` / `classify diag` → `bview` → `knobs` → `kcap` → `kstage_quick` / `atrisk_sim2`
  → `prune_refine` → `sib`.
