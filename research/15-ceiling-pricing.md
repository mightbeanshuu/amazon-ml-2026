# 15 — Ceiling pricing: measured PC-gain vs CPU cost of cheap blocking raisers (India, real train)

Status: **IN PROGRESS** (started 23:05 IST, 26 Sep). Written incrementally; every number below comes from a run executed in this session on the local Mac. Nothing is estimated unless marked **[est]**.

Scratch: `SCR/pricing/` (SCR = `/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad`).

## 0. Method (fixed before any run)

- **World**: India train, exactly as `run_big.block_train` builds it. S1 in TSV order, `train_split` seed 0, hide_frac 0.19 → 715,441 visible S1. All 4,133,346 India S2+S3 records (hidden S1s' records stay in as distractors). Normalisation is the v10 code path (`normalize_frame` → `renormalize_names(lexicon)` → `phon_line`), run locally from the raw TSVs (no box compute, no download).
- **Sample S**: Box 1's own 120,000-S1 training sample (same `train_split` RandomState). Numbers sit directly next to Box 1's `PC 0.9714`.
- **Full distractor density, exact**:
  - forward = S rows vs ALL 4.13M records (exact global top-k);
  - reverse = ALL 4.13M records vs ALL 715k visible S1 (exact per-record top-k). Pairs landing on S are kept, with their within-record rank.
  - Every field is run once at a raised k; smaller k are prefixes of the same top-k lists, so each k curve comes from one run.
- **Stages scored** (per variant, same S):
  1. `PC_union`: the 6-family union before any prune (exact).
  2. `PC_trainK`: after `prune_topk(K=40)` with train semantics (record-side rank among S's pairs only; this is what Box 1's 0.9714 measures).
  3. `PC_test`: test-time semantics. `prune_topk` record top-2 is taken within the S1 quarter (`s1_parts=4`), and competitors come from each record's full reverse lists. Then `prune_by_similarity(28)`, with `rev_rank ≤ 2` among the record's competing S1s by the real prune score.
     - Approximation, stated: forward-only proposers from S1s outside S are not seen as record-side competitors.
- **CI**: variants are paired on the same positives. ΔPC 95% CI = ±1.96·√(gained+lost)/N_pos (McNemar). The oracle macro-F0.5 uses `metrics.blocking_report`'s definition.
- **Cost**:
  - Blocking time is dominated by the sparse product, which does not depend on k. It is timed here per field, and field times are calibrated to Box 1's v10 India log (`b_key` 202 s … `b_cgram` 2612 s).
  - k bumps cost mostly *volume*. Volume is converted to seconds with per-pair rates taken from logs.

- **Priority addition (coordinator): T1a**, the test-side grouped protections from `10` §3, priced on the same positives. Variants:
  - `T1a_K6_G6`: `prune_topk` record-top-2 within ~119k-S1 ranges (`s1_parts=6`), plus `prune_by_similarity` rev_rank within `s1 % 6` groups.
  - `T1a_K24_G6`: the same with ~30k-S1 ranges (`s1_parts=24`).
  - `T1a_Kmod6_G6`: modulo groups on both prunes.
  - Each half alone: `T1a_simonly_G6` and `T1a_Konly_K24`.
  - The alternative fix: `rev12_global` (rev_rank ≤ 12 among all S1) and `rev12_rec12_global`.
  - G = 715,441 / 120k ≈ 6 here; at test India it is 809,986 / 120k ≈ 7.
- **Volume of protected extras**: the protections keep *false* pairs too. Their volume is measured exactly for the K-prune. For the similarity prune it is measured on a fixed 5% S1 subsample (6,000 S1, all their pairs), which gives `rev_extras_per_s1`.

## 1. Replication checks (before any variant is trusted)

- **World matches Box 1:** 715,441 visible S1, 4,133,346 records, and 415,630 GT positives in the 120k sample. All three equal Box 1's `train_block_india.json`.
- **Code parity:** md5 of `block_big.py`, `run_big.py`, `normalize.py`, `lexicon.json`, `anchor.py` and `siblings.py` is identical on Box 1 and in the local `code/ber_v5`.
- **Sample size:** the sample is Box 1's full 120k training sample, not 60k.
  - The extra forward queries cost 8% of one family's forward pass.
  - It makes train-mode record-side competition identical to Box 1's, so PC is directly comparable to 0.9714.
  - Peak RSS is 1.1 GB per family run.
- **b_xkey:** my full reverse at k=2 produces **6,797,547** pairs. Box 1's log says 6,797,547 (exact).
  - Pairs kept for the sample: 1,983,735 against Box 1's 1,986,477 (−0.14%).
  - The gap comes from top-k tie-breaking: a k=16 list truncated to 8 does not order ties the same way as a native k=8 list, and x-keys are binary, so ties are frequent.

- **Box 1 expansion set**: the autopsy agent deleted its copy of Box 1's `train_block_india.parquet`. I re-copied it transiently and reduced it to candidate keys (`SCR/pricing/box_keys.npz`, 33 MB), then deleted the parquet.
  - 9,834,279 pairs, of which 870,526 are sibling-expansion pairs.
  - 403,753 positives are found, and 964 of them only through expansion. This matches `14`.

- **Train-mode replication: PASS.** My v10 union → `prune_topk(40)` → Box 1's expansion gives PC **0.97136** / oracle **0.98979**. Box 1 has 0.9714 / 0.9898, and the exact miss-set count in `14` gives 0.97142 (a 25-positive difference in 415,630).
  - Pair-level overlap is 98.0%: 8.78M common, 180k ours-only, 183k box-only.
  - These swaps sit on equal-cosine ties at the k boundaries: sp_matmul_topn at k_max vs native k.
- **Where v10 train-mode loses its 2.9%** (same sample, measured):

  | stage | PC | oracle | pairs/S1 |
  |---|---|---|---|
  | 6-family union | 0.97859 | 0.99273 | 102.3 |
  | after `prune_topk(K=40)` | 0.96905 | 0.98917 | 74.7 |
  | + Box 1 sibling expansion | 0.97136 | 0.98979 | 81.9 |

  - **The K=40 cap alone throws away 0.0095 PC (0.0036 oracle)** that blocking already found.

## 2. Results

### 2.0 FIRST: the `prune_topk` K cap is the cheapest ceiling knob (measured, 00:50 IST)
Same 415,630 positives. Deltas are paired against v10 and show ΔPC×1e4 ± 95% CI. testSim is the candidate set the matcher actually scores at test time: K-prune, then expansion, then similarity prune to 28.

| variant | trainK PC | trainSim PC | testK PC / pairs·S1⁻¹ into sim-prune | **testSim PC (Δ)** | testSim pairs/S1 (final) | testSim oracle |
|---|---|---|---|---|---|---|
| v10 (K=40) | 0.9714 | 0.9675 | 0.9701 / 79.1 | **0.9618** | 31.25 | 0.9867 |
| K=60 | 0.9749 | 0.9709 | 0.9739 / 85.6 | **0.9653 (+34.4 ± 1.9)** | 31.4 | 0.9881 |
| K=80 | 0.9762 | 0.9720 | 0.9754 / 89.5 | **0.9665 (+47.2 ± 2.2)** | 31.6 | 0.9886 |
| K=120 | — | 0.9729 | 0.9766 / 94.2 | **0.9676 (+58.2 ± 2.4)** | 31.8 | 0.9891 |
| K=∞ (no cap) | — | 0.9752 | 0.9798 / 108.9 | **0.9704 (+86.1 ± 2.9)** | 32.7 | 0.9902 |

- **89% of the pairs the cap releases survive the similarity prune.** At K=60, testK gains +38.5 and testSim +34.4. These are pairs v10 had already proposed.
- **The similarity prune absorbs the volume.** The final per-S1 volume moves only +0.15 (K=60) / +0.35 (K=80) pairs/S1, so features and scoring barely change: +0.5% / +1.1%.
- **The cost is rapidfuzz in the similarity prune:** +6.5 (K=60) / +10.4 (K=80) pairs/S1 enter it.
  - At test India (810k S1) that is +5.3M / +8.4M pairs, ≈ +20 s / +35 s per India test pass at the ~250k pairs/s measured for this function (v3 box log: 30.7M in 117 s).
  - Blocking time is unchanged: same union, and K only filters it.
- **Removing the cap entirely (K=∞) is the largest K-side gain: +0.0086.**
  - Cost: +29.9 pairs/S1 into rapidfuzz (+38%, ≈ +90 s per India test pass) and +1.45 final pairs/S1 (+4.6% features).
  - Memory: the test pair table grows from 79 to 109 pairs/S1 (≈ 64M → 88M rows for India test), before the similarity prune cuts it back.
  - p95 union size is 218/S1 and the max is 5,830.
- **Oracle macro-F0.5 at test India: +0.0014 (K=60) / +0.0019 (K=80). About +0.0007 / +0.0009 global before matcher conversion** (India is 46.75% of test).

### 2.0a Cap exemptions: which pairs to exempt from the K cap (measured 01:30 IST, on K=40, test semantics)
- `ex_fam r`: exempt a pair when it is rank ≤ r for its **S1** in any family. This is block_country's `{fam}_rk`.
- `ex_rev r`: exempt a pair when the S1 is rank ≤ r in the **record's own reverse list** of any family. That rank is taken over ALL S1, in train and test alike, because the reverse search already computes it.

| variant | trainSim PC (Δ) | testK PC / into sim-prune | **testSim PC (ΔPC×1e4 ± CI)** | final pairs/S1 | testSim oracle |
|---|---|---|---|---|---|
| v10 | 0.9675 | 0.9701 / 79.1 | 0.9618 | 31.25 | 0.9867 |
| K=80 | 0.9721 (+45.5) | 0.9754 / 89.5 | 0.9665 (+47.2 ± 2.2) | 31.55 | 0.9886 |
| ex_fam1 | 0.9682 (+7.0) | 0.9709 / 79.1 | 0.9627 (+8.6 ± 0.9) | 31.25 | 0.9875 |
| ex_fam2 | 0.9690 (+15.0) | 0.9720 / 79.2 | 0.9637 (+18.7 ± 1.3) | 31.26 | 0.9880 |
| **ex_rev1** | 0.9726 (+50.8) | 0.9765 / **82.3** | **0.9680 (+61.9 ± 2.4)** | 31.47 | 0.9892 |
| **ex_rev2** | 0.9737 (+62.3) | 0.9780 / 89.5 | **0.9691 (+73.0 ± 2.6)** | 31.89 | 0.9897 |
| ex_fam2 + ex_rev2 | 0.9738 (+63.1) | 0.9780 / 89.5 | 0.9692 (+73.7 ± 2.7) | 31.90 | 0.9898 |
| `T1a_Konly_K24` (for reference) | = v10 | 0.9786 / 106.0 | 0.9693 (+74.7 ± 2.7) | 32.56 | 0.9898 |

- **Answer to the bundle's top question: exempt record-side family rank ≤ 2 (`ex_rev2`).**
  - It gives +0.0073 test PC, against +0.0047 for K=80, at the *same* similarity-prune volume (89.5/S1). Final volume is +0.64/S1 (+2%).
  - `ex_rev1` gets +0.0062 with only +3.2 pairs/S1 into rapidfuzz.
  - S1-side family ranks are nearly worthless, because the S1's top pairs are already inside K.
- **Why record-side:** the lost positives are records whose true S1 is their own best or second-best suitor in some family, but that S1 has more than K stronger-looking candidates, often namesakes.
- **It is train/test-consistent by construction.** The rank comes from the reverse top-k over ALL S1 in both modes, unlike the pair-table record-top-2, so it does not reintroduce the T1 skew.
- **Implementation (~10 lines):**
  - `block_country`: in the reverse loop, `rk = group_rank(j2, v2)` per slice. Keep `min` over families per pair as `rev_fam_rk`, NaN for forward-only pairs.
  - `prune_topk`: `keep |= rev_fam_rk <= 2`.
  - Protection by S1 partitions stays as is.
- Combined with K=80 (the shipped value), see §2.0c.

### 2.0c Exemptions on top of the shipped K=80 (measured 01:20 IST, test semantics, paired vs K=80)

| variant | testK PC / into sim-prune | **testSim PC (ΔPC×1e4 ± CI vs K=80)** | final pairs/S1 | testSim oracle | cost vs K=80 (India test pass) |
|---|---|---|---|---|---|
| K=80 | 0.9754 / 89.5 | 0.9665 | 31.55 | 0.9886 | — |
| K=80 + ex_fam2 | 0.9764 / 89.6 | 0.9675 (+9.9 ± 1.0) | 31.56 | 0.9893 | ≈ 0 |
| **K=80 + ex_rev1** | 0.9784 / 91.8 | **0.9694 (+28.8 ± 1.6)** | 31.73 | 0.9898 | +7 s prune, +6 s features |
| **K=80 + ex_rev2** | 0.9791 / 96.4 | **0.9699 (+33.8 ± 1.8)** | 32.06 | **0.9900** | +21 s prune, +18 s features |
| (ref) K=∞, from the K=40 table | 0.9798 / 108.9 | 0.9704 | 32.70 | 0.9902 | +60 s prune, +40 s features |

- **K=80 + ex_rev2 recovers 94% of the no-cap gain** (+0.0081 of +0.0086 over v10) with 12.5 fewer pairs/S1 in the pair table.
- **For the bundle: K=80 + `ex_rev2`.** It costs ≈ +40 s per India test pass (≈ +1.5 min for all countries) and gives +0.0034 India test PC over K=80 (+0.0014 India oracle).
  - If memory allows, K=∞ is the simplest code change: one config value.

### 2.0b T1a: test-side grouped protections (coordinator priority, measured 01:05 IST)
- These are test-only changes, so the train columns equal v10: trainK 0.9714 / trainSim 0.9675.
- **"final pairs/S1"** is what features, scoring, stage B/E and the anchor pass all scale with. v10 is 31.25; the ratio is the feature/scoring cost multiplier.
- **"into sim-prune"** is the rapidfuzz volume.

| variant (test semantics) | testK PC / into sim-prune pairs·S1⁻¹ | **testSim PC (ΔPC×1e4 ± CI)** | final pairs/S1 (×v10) | testSim oracle |
|---|---|---|---|---|
| v10: quarters + global rev ≤ 2 | 0.9701 / 79.1 | 0.9618 | 31.25 (1.00×) | 0.9867 |
| K=80 (shipped) | 0.9754 / 89.5 | 0.9665 (+47.2 ± 2.2) | 31.55 (1.01×) | 0.9886 |
| `T1a_Konly_K24`: `s1_parts=24` (~30k-S1 ranges) only | 0.9786 / 106.0 | **0.9693 (+74.7 ± 2.7)** | 32.56 (1.04×) | 0.9898 |
| `T1a_simonly_G6`: sim-prune rev within `s1 % 6` only | 0.9701 / 79.1 | 0.9684 (+65.8 ± 2.5) | 64.52 (2.06×) | 0.9887 |
| `T1a_K6_G6`: `s1_parts=6` + rev in `s1 % 6` | 0.9733 / 89.5 | 0.9715 (+97.3 ± 3.1) | 71.52 (2.29×) | 0.9900 |
| `T1a_K24_G6`: `s1_parts=24` + rev in `s1 % 6` | 0.9786 / 106.0 | **0.9766 (+148.0 ± 3.7)** | 81.95 (2.62×) | **0.9920** |
| `rev12_global`: sim-prune rev_rank ≤ 12 over all S1 (the alternative fix) | 0.9701 / 79.1 | 0.9697 (+79.3 ± 2.7) | 74.68 (2.39×) | 0.9892 |

What this shows:
1. **The K-side half of T1a (`s1_parts=24`) is a strictly better K-cap fix than K=80.**
   - It gives +0.0075 PC against +0.0047 at the same final volume (+4%).
   - At ~30k-S1 ranges the record-top-2 protection passes almost every union pair, so testK = 0.9786, the union ceiling.
   - The cost is +27 pairs/S1 into rapidfuzz (+34% similarity-prune time, ≈ +1.5 min per India test pass). Blocking and features are unchanged.
2. **`rev12_global` against grouped rev (sim-prune side):** the two are about equally efficient, ≈ +0.0002 PC per extra final pair/S1.
   - grouped: +0.0066 for +33 pairs/S1;
   - rev ≤ 12 global: +0.0079 for +43 pairs/S1.
   - rev12 also changes train, where trainSim goes to 81 pairs/S1. The K-side fix is ~25× more volume-efficient than either.
3. **The similarity-prune half (rev within `s1 % 6`) buys the rest (+0.0073 more on top) but roughly doubles the final volume.**
   - v10 test keeps 31/S1 while train keeps 67/S1. Grouped rev makes test ≈ train (82 vs 67), which is the sampling-consistency point of T1.
   - It costs ~2.6× the features, scoring, stage B/E and anchor time of the test phase.
4. **Test-time India PC after the prune (v10) = 0.9618.** This is inside `14`'s 0.931–0.9675 band, near the top.
   - The quarter-partitioned `prune_topk` record-top-2 still protects 32 pairs/S1 at test. That is why testK is 0.9701 and not ~0.95.

### 2.2 (a) cgram k: **a loser at test time** (measured 01:40 IST, on the K=80 base, paired vs K=80)

| variant | union PC (Δ) / pairs·S1⁻¹ | trainSim PC (Δ) | testK into sim-prune | **testSim ΔPC×1e4 ± CI** (gained/lost) | final pairs/S1 |
|---|---|---|---|---|---|
| K=80 (cgram fwd 6 / rev 8) | 0.9786 / 102.3 | 0.9721 | 89.5 | — | 31.55 |
| cgram rev 12 | 0.9795 (+9) / 122.8 | 0.9718 (−2.8) | 99.3 | **−0.9 ± 1.1** | 31.81 |
| cgram rev 16 | 0.9803 (+17) / 144.0 | 0.9715 (−6.0) | 108.0 | **−1.9 ± 1.4** | 32.01 |
| cgram rev 24 | 0.9816 (+30) / 187.4 | 0.9710 (−10.4) | 122.3 | **−5.3 ± 1.8** (656/877) | 32.28 |
| cgram fwd 12 | 0.9790 (+4) / 104.5 | 0.9723 (+2.1) | 91.3 | +1.4 ± 0.7 (126/67) | 31.57 |

- The deeper reverse cgram hits are real: union PC rises +0.0030 at rev 24. But they carry only 0.5 × cgram in the K score, and a low text score in the similarity prune, so almost none survive.
- They also **displace** true pairs from the per-S1 top-K and top-28 (877 lost at rev 24), so the net test PC goes *down*.
- Cost: +20 / +42 / +85 union pairs/S1 (merge memory and time) and +10 / +18 / +33 pairs/S1 into rapidfuzz.
- **Do not raise cgram reverse k.** `fwd 12` is a statistically real but tiny +0.00014 for +1.8 pairs/S1 into the prune (≈ +6 s): optional.

### 2.3 (c) Similarity-prune cap and record-top-3 (measured 01:45 IST, K=80 base, paired vs K=80)
Cost is per India test pass, using §2.0d: 3.1 s per pair/S1 into the prune, 35 s per final pair/S1.

| variant | trainSim PC (Δ) / pairs·S1⁻¹ | testK into sim-prune | **testSim ΔPC×1e4 ± CI** | final pairs/S1 (Δ) | India test Δ s | ΔPC per extra final pair/S1 |
|---|---|---|---|---|---|---|
| K=80 | 0.9721 / 67.7 | 89.5 | — | 31.55 | — | — |
| prune_k 32 | 0.9724 (+3.4) / 69.4 | 89.5 | +8.2 ± 0.9 | 35.20 (+3.65) | +128 | 2.2e-4 |
| **prune_k 40** | 0.9735 (+14.7) / 72.9 | 89.5 | **+26.8 ± 1.6** | 42.45 (+10.9) | +380 | 2.5e-4 |
| rec-top-3 in `prune_topk` | 0.9735 (+14.7) / 68.0 | 98.4 | +18.9 ± 1.4 | 32.13 (+0.58) | +48 | 3.3e-3 |
| rev_rank ≤ 3 in sim-prune | 0.9732 (+11.2) / 78.2 | 89.5 | +15.6 ± 1.2 | 35.53 (+4.0) | +140 | 3.9e-4 |
| both record-top-3 | 0.9748 (+27.1) / 84.0 | 98.4 | +35.0 ± 1.8 | 36.69 (+5.1) | +206 | 6.8e-4 |
| (§2.0c) ex_rev2 | 0.9747 (+26.6) / 68.0 | 96.4 | +33.8 ± 1.8 | 32.06 (+0.51) | +39 | 6.6e-3 |

- **prune_k 28 → 40 is real but expensive.** It adds +0.0027 test PC and costs +35% on every per-pair stage of the test phase (≈ +6 min India, ≈ +13.5 min all countries per test pass).
- The record-top-3 variants are dominated by `ex_rev2`, which gets the same PC for a tenth of the volume.

### 2.4 (d) Anchor-retrieval k 4 → 8: **a winner, with a model-conversion caveat** (measured 01:50 IST)
Method: `SCR/pricing/anc.py` replays `anchor.anchor_retrieve` on the full India world.
- Index: all 715,441 visible S1 as enriched entity docs.
- Queries: every missed positive's record.
- df runs over all unresolved records plus entity docs, with max_df 50k, as in the code.
- The missed set is **K=80's test-mode candidate set after the similarity prune**: 13,910 missed positives.
- **Proxy for the model's confident set**, since there is no local model: a true pair with prune text score ≥ τ that survived the pipeline. Two τ values bracket it. Enrichment therefore uses only true tokens, so this is optimistic about FP-free anchors.

| τ (anchor proxy) | unresolved share of records | recovered @k=1 / 2 / **4 (v10)** / 6 / **8** / 12 / 16 | ΔPC k=8 vs k=4 | ΔPC k=16 vs k=4 |
|---|---|---|---|---|
| 150 | 45.5% | 3,479 / 4,201 / **5,011** / 5,515 / **5,920** / 6,466 / 6,875 | **+0.0022** | +0.0045 |
| 170 | 49.6% | 3,264 / 3,999 / **4,800** / 5,340 / **5,710** / 6,259 / 6,674 | **+0.0022** | +0.0045 |

- **v10's existing k=4 anchor pass already adds ≈ +0.012 PC** (5.0k of 13.9k misses). The effective test candidate PC after the anchor pass is ≈ 0.978 for K=80, not 0.9665.
- **k=8 adds another +0.0022 ± 0.00014 PC**, and it is insensitive to τ.
- **Cost.** At most +k new pairs per unresolved record: India test ≈ 4.72M × 0.46 × 4 ≈ **+8.7M stage-A pairs (upper bound; pairs already in candidates are dropped)**.
  - That is ≈ +185 s of features per India test pass, ≈ +6.6 min all countries. The retrieval product itself is k-independent (§2.1).
- **Caveat.** Anchor pairs enter with NaN blocking columns and a model never trained on them, so the PC→F conversion is unmeasured here. The v10 test log's "anchor pass added N pairs (max p …)" is the check.

### 2.0d Cost model: pair volume → box seconds
Every rate below is read from a Box 1 / v3 log on the 8-core box.

| stage | rate on 8-core box | source |
|---|---|---|
| per-family blocking, India train (715k S1 × 4.13M rec) | b_key 202 s, nkey 111, akey 129, xkey 85, **cgram 2,612**, pkey 87; siblings 409 | Box 1 v10 log |
| per-family blocking, US train (1.07M × 6.19M) | b_key 273, nkey 180, akey 154, xkey 116, **cgram 6,742**, pkey 140; siblings 736 | Box 1 v10 log |
| `prune_by_similarity` (rapidfuzz, 4 scorers) | **≈ 262k pairs/s** | v3 test India: 30.7M pairs in 117 s |
| pair features (89 cols, 7 workers) | **≈ 47k pairs/s** (India), 58k (US) | Box 1 v10: IN 8.06M in 173 s, US 9.36M in 162 s |

- The test phase builds features for every final pair at least twice: scoring, then the self-training rescore.
- So one final pair/S1 on India test (810k S1) ≈ 810k × 2 / 47k ≈ **35 s**.
- One pair/S1 into the similarity prune ≈ 810k / 262k ≈ **3.1 s**.
- US test (663k S1) and France (259k) scale the same way: all-country test ≈ 2.14× India.
- The train side (240k sampled S1 for IN+US) is ≈ 0.3× the India test cost per pair/S1.

### 2.1 Blocking-time cost of a k bump is zero (measured)
`bench.py`, same operands, 3 interleaved repeats. Query set: 60,000 records drawn from one 1.4M-record slice, run against the full 715,441 visible-S1 cgram index (`sp_matmul_topn`, 6 threads).

| cgram reverse top-n | 8 | 12 | 16 | 24 |
|---|---|---|---|---|
| median wall (s) | 16.10 | 15.99 | 16.05 | 15.92 |
| ratio vs 8 | 1.000 | 0.993 | 0.997 | 0.989 |

- The sparse product dominates, and top-n selection is noise.
- **Every k bump's CPU cost is therefore pure downstream pair volume**: the merge, `prune_topk`, rapidfuzz in the similarity prune and, if the pairs survive it, features and scoring.
- The full cgram family on the Mac took 401 s forward (120k S1 × 4.13M records) and 1,099 s reverse (4.13M × 715k).
  - Scaled to all 715k S1 forward, that is ~3,600 s, against Box 1's 2,612 s for the same work. The Mac runs ~1.4× slower than the box.

