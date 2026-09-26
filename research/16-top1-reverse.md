# 16 — Reverse-engineering the top-1 (0.9906): what stack plausibly reaches it, and what we can build before the ~09:00 retrain window (27 Sep)

Written incrementally from ~23:10 IST, 26 Sep 2026. Tags:
- **[ARITH]** hard arithmetic from the metric and our measured priors (reproducible script in §A).
- **[RV]** measured on real train data in this pass or in `10-frontier-0995.md`.
- **[CITED]** third-party evidence with URL.
- **[SPEC]** speculation, labelled as such.

Status: **COMPLETE (~00:05 IST, 27 Sep).** §2.E, the train-vs-test shift measurement, is appended last.

## 0. Headline — read this first

**1. The arithmetic bounds what top-1 must have [ARITH, §1].**
- 0.9906 leaves a total loss budget of 0.0094. By the identity LB = 1 − B − M − 0.15·g:
  - **Matcher loss ≥ 0.0095 makes 0.9906 impossible even with perfect blocking and France at parity.** The best public GBDT matcher is Ayan's, at M ≈ 0.0095 (train density) to 0.012 (test density). Ours was 0.016–0.018 in v4.
  - In every plausible cell, top-1 has **blocking PC ≥ 0.985–0.99**. Our v10 India PC 0.9714 costs 0.0102 on its own, more than the whole budget.
  - It has **in-candidate M ≤ ~0.005**, i.e. per-record recall ≥ 0.99 and ≤ 0.01 false records per S1.
  - It has **France ≥ 0.966–0.977**. Two independent teams' France-blanking probes put mid-pack France at 0.944; our v3 implied France was 0.895.
- Our ~0.0156 gap to top-1 splits roughly evenly across blocking (~0.006), matcher (~0.007) and France (~0.0035–0.0055). **No single component closes it.**

**2. The top of the board is a cluster, not an outlier [CITED, §2.A].**
- The top 3 are 0.9906 / 0.9894 / 0.9888, and the top 5 sit at 0.989–0.991. The leader climbed 0.980 → 0.987 → 0.9884 (09:05) → 0.9906 during the day.
- A mechanism shared by ≥ 5 teams exists. It is not a leak: test/train S1 exact overlap is 0.0000 in every country (§1.1).

**3. Most parsimonious reconstruction of that mechanism [CITED pieces, SPEC composition, §2.A-2]:**
- **(a) Ayan-class retrieval**, public: a reverse record→S1 TF-IDF view unioned with forward views, PC ≈ 0.99, B ≈ 0.003.
- **(b) Generator-aware LightGBM**, public: house-number edit types, word-difference odds, competition features.
- **(c) A fine-tuned multilingual cross-encoder on the uncertain band, stacked as a feature.**
  - pkd-prashant, on this data: `xlm-roberta-base` on the 1.74M train pairs with 0.02 < p < 0.98, a V100 run, AUC 0.985 vs LightGBM 0.931.
  - OOF 0.9773 → 0.9828, **M 0.0117 → 0.0062 (−47%)**.
- **(d) France near parity** through test-side vocabulary adaptation / self-training.
- Plugging in: B 0.003 + M 0.006 + 0.15·g (g 0.005–0.02) → **LB 0.988–0.990**, the observed top-5 band.
- **Everything except (c) is the 0.985 wall.** This matches the pattern of every documented AMC winner (a GPU-fine-tuned pretrained model, §2.D) and Foursquare's leak-free winners (cross-encoder −27% error, §2.B).

**4. Measured negatives [RV, this pass].**
- Per-noise-mode canonicalisation to exact joins covers **0.1% of residual FNs** and 0.5% of FPs. It is worth ≈ 0 (§1.7).
- The residual is relational (per-source perturbed numbers), not lexical.

**5. ★ New risk for tomorrow [CITED, §2.A-4].**
- Four teams' sibling/group ("vouching") features gained on validation and **lost on the LB**; pkd went 0.956 → 0.921.
- Per pkd, **a 19% S1-hide density simulation does not reproduce the failure**. That is our `hide_frac` validation.
- v10's stage B and R4 are group features. **Use the label-free detector:** a variant's change in test links/S1 against its change in OOF links/S1 (pkd saw +10% vs +1%).
- **§2.E confirms the shift on our own data and normaliser.** Per unique-name S1, same-name/different-house pool records rise from 0.549 to 0.802 in US (+46%; ≥ 2 such records +82%) and from 0.601 to 0.717 in India (+19%). Own-copy counts are flat (+1%).
- So OOF under-states test FPs in exactly the neighbour-house class.

**6. What to do in the ~09:00 window (§3.4).**
- **Box retrain bundle:** R4 in stage B, the reverse-view protection with prune 40, the test-side reverse view for IN and US (all 3 countries on 32-core), and 400k/country if the learning curve allows. Together ≈ **+0.005–0.0097**, i.e. ≈ 0.981–0.987 from 0.975.
- **The parallel GPU cross-encoder track (F8)** is the single highest-EV item (**+0.0025–0.0045**) and the only thing that plausibly clears the wall. It is a **conditional GO**: it needs its own owner and Kaggle/Colab GPU, the export at ~04:15, and a hard kill at 13:00.
- **NO-GO:** training on all 2.2M S1s in one window, dense dual retrieval, scaled-up self-training without LOCO, canonicalisation, Hungarian, TSV merging, and CPU cross-encoders.
- **0.9906 itself is out of reach tomorrow.**

**7. Measurement caution.**
- harsh's per-country blanking probes are inconsistent with full-test weights: they imply F_US = 1.08. **The public subset is not country-proportional.**
- `10` §5.3's implied-France formula is therefore ±0.02. A France-blank probe is exact if a slot can be spared.

## 1. Forensic decomposition of 0.9906

### 1.1 Inputs [ARITH / RV]
- **Test S1 shares (counted from `test_source1.tsv`):** India 809,986 (0.4675), US 663,106 (0.3827), France 259,452 (0.1498). Total 1,732,544.
- **Size prior (real train GT, `10-frontier` §1.1):** 5.58% singletons, mean 3.46 true records per S1, identical in IN and US.
- **Per-S1 score:** F0.5 = 1.25k / (1.25k + 0.25(n−k) + f), where k = true records predicted, f = false ones, n = truth size.
  - A singleton scores 1 only if f = 0.
  - A non-singleton predicted empty scores 0.
- **Public-LB sampling noise is negligible.** Even with only ~30% of 1.73M S1s public, SD ≈ √(0.005/520k) ≈ 0.0001. So 0.9906 is not luck.
- **No train→test text leak [RV, this pass]:**
  - Exact (name, address) overlap of test S1 with train S1 is **0.0000** in all three countries.
  - Names alone overlap 44% (IN) and 35% (US) because the generator reuses name vocabulary; France overlaps 0.0001.
  - With 15 uploads in total, LB probing cannot recover labels.
  - So 0.9906 is a modelling result on unseen entities.

### 1.2 The budget identity [ARITH]
LB = 1 − B − M − 0.1498·g, where:
- B = seen-country blocking loss (1 − oracle).
- M = seen-country in-candidate matcher loss.
- g = France gap (F_seen − F_FR).

**The total loss budget for 0.9906 is 0.0094.** Every component must fit inside it:

| France gap g | F_seen needed | F_FR implied | seen budget B+M |
|---|---|---|---|
| 0 | 0.9906 | 0.9906 | 0.0094 |
| 0.005 | 0.9913 | 0.9863 | 0.0087 |
| 0.010 | 0.9921 | 0.9821 | 0.0079 |
| 0.020 | 0.9936 | 0.9736 | 0.0064 |
| 0.030 | 0.9951 | 0.9651 | 0.0049 |
| 0.047 (our v4 LOCO gap) | 0.9976 | 0.9506 | 0.0024 |

**Consequence 1:**
- A realistic matcher loss of **M ≥ 0.0095 makes 0.9906 arithmetically impossible, even with perfect blocking and France at parity.**
- The best public matcher is Ayan's: local 0.98703 against a pruned oracle of 0.9965, so M ≈ 0.0095 at train density and ≈ 0.012 at test density (P-dense 0.98479).
- Our v4 was M = 0.0157 (IN) and 0.0180 (US).
- **Top-1 is therefore not "the public recipe with better blocking". Its matcher makes at most about half the errors of the best public matcher.**

### 1.3 Blocking PC that 0.9906 forces [ARITH]
Blocking loss is mapped to PC in two ways:
- **Independent misses:** loss ≈ 0.295·(1 − PC). This matches Ayan: PC 0.9905 → oracle 0.99715 against a model value of 0.99720.
- **Our clustered misses:** v10 IN PC 0.9714 → oracle 0.9898, a ratio of 0.357. Our misses cluster on empty-address namesakes, where whole S1s are lost.

Feasible region (the minimum PC column uses our 0.357 ratio; the random-miss value is in brackets):

| M (matcher) | g = 0 | g = 0.01 | g = 0.02 |
|---|---|---|---|
| 0.003 | PC ≥ 0.982 (0.978) | ≥ 0.986 (0.983) | ≥ 0.991 (0.989) |
| 0.004 | ≥ 0.985 (0.982) | ≥ 0.989 (0.987) | ≥ 0.993 (0.992) |
| 0.005 | ≥ 0.988 (0.985) | ≥ 0.992 (0.990) | ≥ 0.996 (0.995) |
| 0.006 | ≥ 0.991 (0.989) | ≥ 0.995 (0.994) | ≥ 0.999 |
| 0.007 | ≥ 0.993 (0.992) | ≥ 0.998 (0.997) | impossible |
| 0.008 | ≥ 0.996 (0.995) | impossible | impossible |

**Consequence 2:**
- In every plausible cell (M 0.004–0.006, g 0–0.02), **top-1's blocking PC is ≥ 0.985 and most likely ≥ 0.99**. That is at least Ayan's 0.9905 union.
- **Our v10 India PC 0.9714 (B = 0.0102) alone exceeds the entire 0.0094 budget.** Even a perfect matcher on v10 India candidates cannot reach 0.9906.

### 1.4 What M ≤ 0.005 means in error rates [ARITH]
M = 1 − macro(r_m, λ) at perfect blocking, where r_m = in-candidate recall of true records and λ = false records per S1:

| r_m \ λ | 0.005 | 0.010 | 0.020 | 0.030 |
|---|---|---|---|---|
| 0.973 | 0.0094 | 0.0106 | 0.0131 | **0.0155 ← v4 IN** |
| 0.985 | 0.0057 | 0.0069 | 0.0094 | 0.0119 |
| 0.990 | 0.0042 | **0.0054** | 0.0079 | 0.0104 |
| 0.995 | 0.0027 | 0.0039 | 0.0064 | 0.0089 |

- Top-1 needs roughly **r_m ≥ 0.99 and ≤ 0.01 false records per S1**. That is pair precision ≈ 0.997, including **singleton FP ≤ ~1.5%**; each singleton FP costs a full 1.0, 0.0558 × rate.
- For comparison:
  - v4 had r_m ≈ 0.973 and λ ≈ 0.03, with singleton FP 3.8–4.7%.
  - Ayan reports "singleton accuracy ≈ 0.98".
- So top-1 has **~3× fewer FPs and ~2.5× fewer FNs than v4**, and about half the errors of the best public stack.

### 1.5 Where the gap between us and top-1 sits [ARITH, using our measured components]
Expected v10 decomposition (≈ 0.975):
- B̄ ≈ 0.009: IN 0.0102 pre-prune, US ~0.007 by v4 ratio.
- M̄ ≈ 0.012: v4 0.016 minus v5–v10 gains.
- 0.15·g ≈ 0.005: g ≈ 0.03–0.047.
- Total ≈ 0.026.

Top-1's feasible point, e.g. B 0.003 / M 0.005 / 0.15g 0.0015, totals 0.0094. **The 0.0156 gap splits about evenly:**

| Arm | Gap | Top-1 needs |
|---|---|---|
| Blocking | ~0.006 | PC ≈ 0.99 |
| Matcher | ~0.007 | M ≈ 0.005 |
| France | ~0.0035–0.0055 (g 0.03–0.047 → 0.01) | g ≈ 0.01 |

- **No single component closes the gap; all three arms are required.**
- **The 0.985–0.986 wall is exactly the Ayan-class point:** B ≈ 0.003, M ≈ 0.009–0.012, g ≈ 0.01–0.02 → LB ≈ 0.982–0.9865.
- Top-1's +0.005 over the wall must come mostly from the **matcher arm**, since blocking at PC 0.99 is already in the public recipe. Matcher loss has to fall from ~0.010 to ~0.005 together with a small France gap.

### 1.6 LB-minus-local offset [ARITH + CITED]
- Public teams show LB − local ≈ −0.01 to −0.02:
  - priyanshiiitr: "LB−local offset stable at −0.02" (`07` §3).
  - Tony: 0.955/0.961 against mock 0.9677/0.9704, −0.009 to −0.013 (`08` §2b).
  - Our v3: 0.947 against seen OOF ≈ 0.955, −0.008.
- A local score ≤ 1 caps top-1's offset at ≥ −0.009, and a realistic local ≤ 0.995 caps it at ≥ −0.004.
- **Top-1's local validation must be density-matched and France-robust.** They lose ≤ 0.004 to the test world, where everyone else loses 0.008–0.02. That is the France arm restated.

### 1.7 Canonicalisation cannot be the missing lever [RV, this pass]
Test: on v4's 6,931 in-candidate error rows (5,453 FN / 1,478 FP, `real/v4_err_s2.parquet`), apply an aggressive canonicaliser to name and address. It does NFKD, leet inversion, legal-suffix and stop-token drop, street-abbreviation expansion and zero-stripped numbers (`top1/canon_err.py`).

| | FN rows | FP rows |
|---|---|---|
| canon name equal | 45.8% | 49.8% |
| number set equal | **9.0%** | **18.2%** |
| addr token Jaccard ≥ 0.8 | 44.7% | 46.8% |
| **canonical exact join** (all three) | **0.1%** | **0.5%** |

- The residual errors are **not** invertible noise. Only 0.1% of FNs would become an exact join, and FPs are 5× *more* likely to exact-join than FNs.
- True-but-missed records have *fewer* equal numbers than decoys (9% vs 18%). This is the per-source perturbed-base signature from `10` §1.5: the generator perturbs a true copy's number and plants decoys at near-offset numbers.
- **Per-noise-mode canonicalisation to exact joins is worth ≈ 0 on the residual.** Whatever top-1 does to halve M, it is *relational* (cluster- and source-level evidence), not string normalisation.

### 1.7b France floor for top-1 [ARITH]
F_FR = (0.9906 − 0.8502 · F_seen) / 0.1498:

| Seen-country F assumed | Plausibility | France must be ≥ |
|---|---|---|
| 0.9972 | Ayan's *oracle*, i.e. a perfect matcher: impossible | **0.954** (absolute floor) |
| 0.995 | B 0.0015, M 0.0035: best conceivable | **0.966** |
| 0.993 | B 0.003, M 0.004: plausible top-1 | **0.977** |

- Our measured France is far below that:
  - v3 implied France **0.895** (`10` §1.3);
  - v4 LOCO 0.915.
- **Top-1's France is ≥ 0.05–0.08 above ours.** At the 0.1498 weight that is worth **0.0075–0.012 LB**, as much as the whole matcher arm.
- v10's France is unmeasured until Slot 1's implied-France read (P4). It is the single most informative number tomorrow:
  - **implied France ≥ 0.95:** our gap to top-1 is mostly seen-country B + M;
  - **implied France ≤ 0.93:** France is our largest single deficit, worth more than any seen-country item in §3.

### 1.8 The "dark matter" requirement on the matcher [ARITH on RV error shares]
- Empty-address records are 18% of v4's in-candidate errors (`10` §1.3). Their namesake ambiguity is symmetric, since name discrepancies are per-copy, not per-base (`10` §6). Treat them as a floor, ≈ 0.18 × M, which top-1 shares.
- **OOF inflates M relative to test.** 20% of OOF FPs are visible-other-owner, an artefact of scoring only 17% of competing S1s. On test, exclusivity removes them. So v4's in-test M ≈ 0.0157 − 0.2 × 0.0075 ≈ **0.014 (IN)**.
- **Top-1's M ≤ 0.005 needs a 70–80% cut in our addressable matcher error:**

  | Model | Floor | Addressable part | Addressable target at M = 0.005 | Cut needed |
  |---|---|---|---|---|
  | v4 | 0.0025 | 0.0115 | 0.0025 | −78% |
  | v10 (M ≈ 0.012 on OOF) | 0.0022 | 0.0098 | 0.0028 | −71% |

- **Nothing measured or published delivers a cut that size:**

  | Lever | Share of M it removes |
  |---|---|
  | R4 (+0.0014–0.0024) | ≈ 15–20% |
  | Ayan's collective stage 2 (+0.0021) | ≈ 18% |
  | P-dense (+0.0014) | ≈ 12% |

  - Even stacked at full value these reach ≈ 45–50%.
- **The remaining ~25–30 points of error reduction was, before §2.A, unexplained by any public artefact.**
- **§2.A-2 now supplies it.** A fine-tuned `xlm-roberta-base` cross-encoder on the uncertain band cut pkd-prashant's in-candidate M from 0.0117 to 0.0062 (−47%) on this exact data.
  - Together with Ayan-class blocking, that lands on the observed 0.988–0.990 top-5 band.
  - Secondary contributors, [SPEC], are more training data (F2) and deeper generator-structure features.

## 2. Evidence sweep

### 2.A Amazon ML Challenge 2026: live evidence (sweep 22:45–23:30 IST, 26 Sep) [CITED]
The sub-agent's notes are at `SCR/top1/notes.md`. Scale of the sweep:
- 1,242 GitHub repos pushed since 20 Sep, 547 READMEs and ~4k md/log/json files grepped.
- Reddit read via pullpush, plus Exa, Firecrawl and web search.
- **No public artefact claims a public LB ≥ 0.98.** The top-10 teams' methods are unpublished.

**1. The top of the board is a cluster, not an outlier.**
- "Top of the leaderboard on 2026-09-26: **0.9906, 0.9894, 0.9888**" ([harshgitty58 PROGRESS.md](https://github.com/harshgitty58/Amazon_ML_Challenge/blob/HEAD/business_entity_resolution/PROGRESS.md), pushed 20:54 IST).
- "The public-leaderboard top 5 are at 0.989–0.991" ([YatharthJangid REPORT_v3](https://github.com/YatharthJangid/amazon_ml/blob/HEAD/REPORT_v3.md), 20:30 IST).
- The leader's score over the day:

  | When | Leader | Source |
  |---|---|---|
  | earlier | 0.980473 | Skullybutcher, `07` |
  | earlier | 0.9869 | priyanshiiitr, `07` |
  | 09:05 IST | **0.9884** | [HarpalKalankar](https://github.com/HarpalKalankar/business-entity-resolution) |
  | evening | 0.9906 | harshgitty58 |

- **So ≥ 5 teams sit at 0.989+.** The winning mechanism is shared by several independent teams, not one private trick.
- A self-described member of the #1 team (MTech student with a full-time job) declined to share the approach ([Reddit r/Btechtards 1wqp2n5](https://www.reddit.com/r/Btechtards/comments/1wqp2n5/amazon_ml_challenge_2026/), 20:21 IST).
- The board is dense: 0.9779 was "rank 410", 0.967 "rank is still 700" (21:33 IST), and several teams are "stuck at 0.985" / "stuck on 0.9847" (same thread).

**2. ★ The strongest single mechanism datum: a cross-encoder halves matcher loss on this exact data.**
- Source: [pkd-prashant/amazon-mlss Documentation_template.md](https://github.com/pkd-prashant/amazon-mlss/blob/HEAD/Documentation_template.md), public LB 0.9779.
- Their own version ladder (OOF macro F0.5 on train):

  | Version | OOF | Note |
  |---|---|---|
  | v8 | 0.9773 | 55-feature LightGBM incl. house-number edit types + learned word-difference odds |
  | **v9 = v8 + `xlm-roberta-base` cross-encoder as a stacked feature** | **0.9828** | US 0.987 / India 0.977 |

- Their candidate-set oracle is **0.989**, so **M went 0.0117 → 0.0062, a −47% cut from the cross-encoder alone.**
- Setup, verbatim:
  - "fine-tuned on a V100 as a pair classifier over `"<name> | <address>" [SEP] "<name> | <address>"`".
  - "trained on the 1.74 M train pairs the LightGBM model is uncertain about (probability 0.02–0.98), using the same 2 entity-grouped folds, so the stacked feature is out-of-fold".
  - "On these hard pairs it reaches **AUC 0.985**, against 0.931 for LightGBM. Test pairs in the same band (2.0 M) are scored by averaging the two fold models."
- **This is the §1.8 dark matter, measured.** With Ayan-class blocking (B ≈ 0.003) and their M ≈ 0.006, LB = 1 − 0.003 − 0.006 − 0.15g:

  | France gap g | LB |
  |---|---|
  | 0.02 | **0.988** |
  | 0.01 | **0.9895** |
  | 0.005 | **0.990** |

  That is the observed top-5 band.
- **[SPEC but tightly constrained] The top-1 stack is:**
  - Ayan-class multi-view + reverse retrieval;
  - LightGBM with generator-aware features;
  - a **fine-tuned multilingual cross-encoder on the uncertain band**;
  - France near parity (vocabulary adaptation / self-training).
- pkd themselves are at 0.9779 only because of weak blocking (pair recall 96.7% at 9.3 cands/S1, B = 0.011) and a France gap.

**3. France measured by two teams through blanking probes (two independent sources agree).**
- Method: blank France (its singletons still score 1).
  - [harshgitty58](https://github.com/harshgitty58/Amazon_ML_Challenge/blob/HEAD/business_entity_resolution/PROGRESS.md): 0.969 → **0.836**.
  - [rithishbarathn VERSIONS.md](https://github.com/rithishbarathn/amazon-ml-challenge-2026/blob/HEAD/submissions/VERSIONS.md): 0.97509 → **0.842**.
- [ARITH] F_FR = ΔLB / 0.1498 + 0.0558:
  - both give **F_FR = 0.944**;
  - seen countries: harsh 0.9735, rithish 0.9805 (rithish states "US + India ~ 0.981 and France ~ 0.944").
  - **Mid-pack France gap g ≈ 0.03–0.037.**
- By §1.7b, top-1's France must be ≥ ~0.975 when seen ≤ 0.993. **Top-1's France is ≥ 0.03 above mid-pack.**
- **Caution for our P4 read-out.** harsh's India-blank (0.580) and US-blank (0.578) probes *cannot* be fitted with full-test weights: they imply F_US = 1.08.
  - The public subset is therefore **not country-proportional**. If F_IN ≈ F_US, then w_IN ≈ 0.424 and w_US ≈ 0.426 on the public board, against 0.4675 / 0.3827 on the full test.
  - `10` §5.3's implied-France formula uses full weights. **Treat its France read as ±0.02 until reconciled.**
  - A direct France-blank probe on our best file would pin it (one slot, exact).
  - Single source; the same file also claims "uploads are unlimited", which contradicts the 5/day rule. Weight accordingly.

**4. ★ Warning: sibling/group ("vouching") features gain on validation and LOSE on the LB (4 independent teams).**

| Team | Validation effect | LB effect | Source |
|---|---|---|---|
| pkd-prashant | OOF 0.9656 → 0.9743 | **0.956 → 0.921** | pkd doc (above) |
| Sid-techweb E10 | — | 0.933, withdrawn | [REPORT](https://github.com/Sid-techweb/AmazonML-New/blob/HEAD/experiments/v2/REPORT.md) |
| rishabhiitj25 stacked | — | 0.965 vs 0.969 | [repo](https://github.com/rishabhiitj25/AmazonML26) |
| harshgitty58 | +0.010 | +0.002 | PROGRESS.md |

- pkd's mechanism, verbatim: "A stage-2 model with self-supported group features raised CV but let 'sibling' clusters vouch for themselves on test". Their stage 2 added **+10% test assignments against +1% on train**. The extra pairs were mostly neighbouring-house-number siblings.
- **"Simulating the density shift by dropping 19% of S1s did not reproduce the failure"**. This is exactly our `hide_frac = 0.19` validation.
- pkd also reports that test top candidates with a disagreeing house number are **"34% more frequent per entity in test"** (same name) and "67% more frequent" (different name), with 5.75 vs 4.68 records/S1.
- We re-measure this ourselves in §2.E.
- **Direct relevance:** v10's stage B vouching, sibling expansion and R4 are all group features. Their gains are OOF-measured, the class of evidence that failed on the LB four times.
- **Mitigation:** compare the predicted-links-per-S1 of test vs OOF for every group-feature variant. pkd's +10% vs +1% is a *label-free* detector: if a variant raises test links/S1 by several times its OOF rise, it is vouching for sibling clusters.

### 2.E Our own check of the test shift: more neighbour-house namesakes in test [RV, this pass, `shift.py`]
**Method:**
- Same normaliser primitives as prep (`name_core`, `house_no`).
- Restricted to S1s whose `name_core` is **unique among the country's S1s**. There, every same-name pool record is either the S1's own copy or an unowned or hidden namesake, so namesake-S1 competition does not confound it.
- Train is the full world (no hide), US 4.67 and India 4.68 recs/S1. Test is the real test pool: US 5.76, India 5.82, France 5.53 recs/S1.

| per unique-name S1 | train US | test US | Δ | train IN | test IN | Δ | test FR |
|---|---|---|---|---|---|---|---|
| same-name & **same** house (≈ own copies) | 1.491 | 1.508 | +1% | 1.410 | 1.431 | +1% | 2.141 |
| same-name & **different** house | 0.549 | **0.802** | **+46%** | 0.601 | **0.717** | **+19%** | 0.564 |
| P(≥ 1 different-house namesake) | 0.385 | 0.530 | +38% | 0.358 | 0.420 | +17% | 0.393 |
| P(≥ 2 different-house namesakes) | 0.093 | **0.169** | **+82%** | 0.145 | 0.172 | +19% | 0.093 |
| unique-name S1 share | 0.482 | 0.554 | | 0.420 | 0.426 | | 0.505 |

**Findings:**
- **Own-copy structure is unchanged** (+1%): same generator, same copies.
- The **neighbour-house namesake load per S1 is much higher in test**: US +46%, and +82% at ≥ 2.
- This happens although test US has **half** train US's S1 count (663k vs 1.32M). With a shared name vocabulary, fewer entities should mean *fewer* namesake collisions.
- This independently confirms pkd's "34% more frequent per entity in test" (§2.A-4) on our own data and normaliser.
- **Mechanism [SPEC]:** test's ~1.1 extra unowned records per S1 are not random. They are preferentially **same-name different-address siblings** (chain branches or hidden namesake entities). These are exactly the records that sibling/vouch features wrongly pull in and that R4's cross-source signature flags.
- **France looks train-like** on this statistic (0.564; P≥2 0.093), with *more* same-house copies (2.14), so its decoy load is not the issue. Its gap is vocabulary/ranking (`10` §1.3), consistent with pkd's 51% word-odds coverage.
- **Implication:** our FP rate on test is under-estimated by OOF in exactly the neighbour-house class, and more so in US.
  - Decision thresholds and group features tuned on OOF will over-merge on test.
  - The label-free links/S1 detector in §2.A-4 is the guard.
  - [SPEC] Raising the decision threshold for pairs with a *different* house number and a *shared* name, in a US-first parity probe, is the cheapest test.
- **Confound (not resolved tonight):** the train column is the full world, with no 19% hide.
  - Hiding S1s leaves pool records unchanged. But it can turn a visible S1 whose namesakes were all hidden into a "unique-name" S1 that now counts those namesakes' records, which pushes the train value *up*.
  - The density-matched mock `research/16-scripts/shift2.py` does this: entity subsample to test's S1 count, then 19% hide. It was written but cancelled, because it was queued behind another job on the RAM lock.
  - pkd reports that their 19%-hide simulation did *not* reproduce the test failure, which suggests the excess is real rather than a hide artefact. Treat the +46%/+19% as an upper bound on the excess until `shift2.py` runs.
  - It takes ~8 min and ~1–3 GB under `ramlock.sh`, needs no box, and can run whenever the lock is free.

### 2.B Strongest analogue: Foursquare Location Matching 2022, plus Shopee and SIGMOD [CITED]
The sub-agent's notes are at `SCR/top1/wincomp_notes.md`. Every number below is quoted from the linked writeup.

**The Foursquare top 3 won on a train/test leak. Their "recipe" is not evidence for us.**
- About 67% of test rows were exact copies of train rows. Leak-only with no model scored 0.884 (74th) ([disc/335799](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335799)).
- Kaggle's clean re-score flipped the order ([disc/338035](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/338035)):
  - 10th-place Ri became **1st at 0.9328**, with no writeup, so the method is UNVERIFIED.
  - re:waiwai fell from 1st to 4th at 0.9206.
- Leak contributions:

  | Team | Without leak | With leak |
  |---|---|---|
  | re:waiwai | 0.900 | 0.971 |
  | 2:30 | 0.949 | 0.971 |
  | 4th | 0.939 | 0.957 |

- **Our data has no such leak.** §1.1 measured 0.0000 exact test↔train S1 overlap. So a leak-type explanation of 0.9906 is ruled out, and 0.9906 has to be a modelling result.

**Transferable, leak-free lessons (their numbers):**

| Lesson | Evidence | Relative error reduction | URL |
|---|---|---|---|
| **Transformer cross-encoder on top of GBDT** | re:waiwai CV 0.875–0.878 (LightGBM/CatBoost) → mdeberta-v3-base 0.907 → ensemble with xlm-roberta-large **0.911**. Trained on 1 A100 + a 64-vCPU box. 2:30 used xlm-roberta-base as well. 13th (213tubo, **clean 2nd**) used xlm-r + mdeberta cross-encoders. | error 0.122 → 0.089 = **−27%** | [336055](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336055), [336072](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336072), [13th writeup](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/team-merge-master-13th-place-solution-gnn) |
| **Graph-level post-processing** | 213tubo: a GNN over each id's 2-hop neighbourhood (max IoU 0.993) trained on an IoU loss, 0.924 → **0.946**; clean rank 2. Shopee 1st: union of matches + neighbourhood blending .743 → .776 → .793 LB; "individual model improvement helped less". | **−29%** (Foursquare), −19% (Shopee) | [13th](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/team-merge-master-13th-place-solution-gnn), [Shopee 1st](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238136) |
| **Candidate recall ≥ 0.98 is table stakes, not the differentiator** | Max IoU: re:waiwai 0.979; 2:30 0.9895 at 128 cands (0.986 after transformer blocking); 9th 0.994; 213tubo 0.971 → 0.993 with 2-hop | — | as above |
| **More data** | 8th: 0.908 → 0.948 as ids grew 550k → 1.1M. 9th: +0.010 from training on all data (`12` §1.B). 4th trees-only: 1.1M rows, 200+ features, 20 folds → 0.939 without the leak. | 8th: −43% | [4th, 335810](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335810) |
| **Synthetic, rule-built data** | SIGMOD 2021 winner: regex + alias tables + per-brand keys, no ML, F1 .965. SIGMOD 2020: "all top-5 teams reached a 0.99 F-measure". SIGMOD 2022 data was script-generated (word shuffles, deletions, case changes), and **no finalist documented inverting it**. | — | [SIGMOD21 poster](https://dbgroup.ing.unimo.it/sigmod21contest/posters/SUSTech_DBGroup.pdf), [contest paper](https://dl.acm.org/doi/pdf/10.1145/3615952.3615965) |
| **Generator inversion where it *was* done** | Instant Gratification: inferring sklearn `make_classification` gave QDA 0.970 → GMM 0.975, against a perfect-classifier bound of 0.97560. Santander: synthetic-row detection .914 → .921. Quora: a pair-sampling graph feature took logloss 0.20 → 0.157. | — | [IG](https://www.kaggle.com/competitions/instant-gratification/writeups/chris-deotte-how-to-score-lb-0-975), [Santander 1st](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/writeups/wizardry-1-solution), [Quora](https://www.kaggle.com/competitions/quora-question-pairs/discussion/33287) |

**What this implies for the §1.8 "dark matter" (70–80% error cut) [ARITH on CITED relative reductions; composition is SPEC].** This was written before §2.A-2 landed. The in-domain measurement (−47% from a cross-encoder on this data) supersedes the −27% Foursquare figure below. The conclusion only gets stronger.
- Leak-free top teams got their edge from two levers:
  - **a cross-encoder stage, −27% error**;
  - **graph/collective post-processing, −20–29%**.
- Compose the relative cuts multiplicatively:
  - cross-encoder −27% × collective/graph −20% × P-dense −12% × R4 −15% × 4–8× more training data −15%
  - → remaining error 0.73 × 0.80 × 0.88 × 0.85 × 0.85 ≈ 0.37, a **−63% cut**.
- That is within reach of the −71% that top-1 needs (§1.8).
- **Without the cross-encoder** the product is 0.51, a −49% cut, and M stays ≈ 0.006–0.007. That lands at LB ≈ 0.986–0.988, i.e. **the wall**.
- **This is the most parsimonious reconstruction of top-1:**
  - public Ayan-class blocking and collective stacking;
  - many more training pairs;
  - **a fine-tuned GPU transformer matcher** (the official Q&A allows it; see §2.C);
  - near-parity France.
- Everything *except* the transformer is the 0.985 wall.

### 2.C What the rules allow a top team to run [CITED: `research/official_QA.tsv`, organiser answers]
- **GPUs:** "Using AWS/SageMaker is encouraged but not mandatory. You may develop on other platforms of your choice" (24 Sep 23:10; the question named Colab and Kaggle "for GPU-based components").
- **Models:** "Pretrained open-weight models are allowed if they are MIT/Apache-2.0 licensed, up to 8B parameters, run offline … fine-tuned only on the provided data. The limit is per model" (25 Sep 00:04, repeated about 15 times).
  - The 03:14 question named **Qwen3-Embedding, Qwen3-Reranker and DeBERTa** explicitly and got the same answer.
- **Test-side learning:** "Computing unsupervised statistics (TF-IDF, token frequencies, a blocking index) on the provided test files … is allowed. **Self-training and generating synthetic pairs** from the provided records are also fine" (25 Sep 02:52).
- **Dictionaries:** "small hand-written normalization dictionaries" are allowed; gazetteers and libpostal are not (25 Sep 01:34).
- **Consequence:** a fine-tuned cross-encoder on Kaggle or Colab GPUs, test-time self-training on France, and French hand dictionaries are all legal. Every component of the §2.B reconstruction is permitted.

### 2.D Past Amazon ML Challenges: what the winners did that mid-pack did not [CITED, from `01` §4.2–4.3 plus this pass]
- **2024 winner (NeuralNinjas)** fine-tuned Qwen2-VL-7B on 20K noisy samples, then **hand-curated 1,600 clean labels** and fine-tuned again. F1 went 0.617 → 0.679 → **0.865** ([KhadgaA/Amazon-ML-Challenge](https://github.com/KhadgaA/Amazon-ML-Challenge)).
- **2025 2nd runner-up (00_Team_Rocket):** DeBERTa-v3-large plus engineered features, fused with cross-attention ([repo](https://github.com/parthrastogicoder/Amazon-ML-Challenge-2025-3rd)).
- **2025 SPAM_LLMs** (3rd public → 5th private): frozen Qwen3-4B / SigLIP2 / DINOv3 embeddings with MLP heads ([repo](https://github.com/RudrakshSJoshi/amlc-multimodal-mlp)).
- **2023 #2:** fine-tuned BERT/RoBERTa ([greenfish8090/AmazonML](https://github.com/greenfish8090/AmazonML)).
- **Leaderboard #1 won the finale in both 2023 and 2024** ([Reddit](https://www.reddit.com/r/Btechtards/comments/1ntmvrr/amazon_ml_challange_is_back_for_2025_i_won_it_2/)).
- **The pattern:** every documented AMC winner or top-3 ran a **GPU-fine-tuned pretrained transformer**, and the 2024 winner's decisive step was **label quality, not model size**.
  - Caveat: those were text/vision regression and extraction tasks, where transformers were also the mid-pack baseline.
  - In 2026, by contrast, **every public artefact, from 0.961 LB down, is CPU GBDT on similarity features** (`07`, `08`).
  - [SPEC] A team that brought the AMC-winner habit (a GPU-fine-tuned cross-encoder) to a field of GBDT pipelines is exactly the profile that sits +0.005 above a GBDT wall. This is consistent with §2.B, where a cross-encoder is worth a −27% error cut.

## 3. Feasibility map for tomorrow

### 3.1 How the Δ column is computed
- Δ is global LB, read through the §1.2 identity: ΔLB = w·ΔB + w·ΔM + 0.1498·Δg.
- **"Ceiling"** means perfect conversion. A realised Δ applies a conversion factor. For blocking, 0.5–0.8 is realistic: recovered records are the hard ones, so the matcher accepts fewer of them.
- **Wall-clock sources:**
  - 8-core figures come from `10` §5.2, Box 1's v10 log: IN train blocking 62 min (cgram 44), features ~30k pairs/s, LightGBM ≈ 1 min per M rows at 8 threads.
  - 32-core figures assume `PARAMS["num_threads"]` and `_topk_chunks(threads=…)` are raised from the hard-coded 8. Without that edit, the gain is only 1.3–1.5×.

### 3.2 The map

| # | Component (what top-1 plausibly has) | Arm | Evidence for size | Expected Δ for us | 8-core wall | 32-core wall | GO / NO-GO for the ~09:00 window |
|---|---|---|---|---|---|---|---|
| F1 | **Ayan-class retrieval: reverse record→S1 TF-IDF combo top-8 as a first-class view**, unioned with forward views, per-view rank caps, ~45–50 cands/S1 | B | [CITED] Ayan union PC 0.9905 / oracle 0.9972 ([block_train_v1](https://github.com/AyanAhmedKhan/amazon-ml-challenge)). [RV] our IN: the view alone R@8 0.978, R@40 0.989, above v10's 6-family union at 0.9714. [ARITH] §1.3: PC ≥ 0.985–0.99 is *necessary* for 0.9906. | Ceiling: IN 0.0067×0.4675 = 0.0031; + US and FR ≈ 0.0025 → **0.0056 all countries**. Realised ×0.6 → **+0.002–0.0035** if all 3 countries; +0.001–0.002 IN-only. | ~750 reverse queries/s on 8 threads (`10` §6: 30k in 40 s). Records to query: test ≈ 9.5M + train ≈ 10M → **~7 h for all splits**. IN-test-only ≈ 1.5 h. | ~2 h all splits, ~25 min IN test | **GO on 32-core** as the headline change of the retrain. **On 8-core: GO only for the test side of IN + US** (≈ 3 h), with train-side reverse queries restricted to the train *sample*'s neighbourhood. Otherwise the model never sees `r_rev` distributions from the new view: features shift, and it is a NO-GO without a train-side twin. |
| F2 | **Train on many more labelled S1s** (all ~1.79M visible, or 400k/country) | M | [CITED] Foursquare 8th: LB 0.908 → 0.928 → 0.948 at 550k/700k/1.1M ids. 9th: +0.010 from all data (`12` §1.B). Counter [CITED]: WDC Products, Magellan-RF flat. [SPEC] At loss ∝ N^−α with α 0.1–0.2 (typical tabular GBDT), each doubling cuts M by 7–13% ≈ **0.0008–0.0016 per doubling** at M ≈ 0.012. | 120k→400k (1.7 doublings): **+0.0014–0.0027**. → all S1s (~2.9 doublings): +0.0024–0.0046. **Gated on `learning_curve.py`.** | `ber_v5/src/run_big.py:262–275` already blocks **all visible S1** and keeps only the sample's pairs (`keep_s1`), so the marginal cost is pair features plus LightGBM, linear in the sample. 400k: ≈ 8 h total incl. B-IN and the test phase (`10` §5.2). **All S1s:** ~72M train pairs at prune 40, ≈ 3.4 h features + training by `10`'s 32M ≈ 1.5 h scaling, and a ≈ 29 GB float32 feature matrix before copies → **≈ 11–12 h and RAM-risky on 64 GB. NO.** | 400k: 2.5–3 h. All: ~4–5 h (RAM still the risk) | **GO for 400k** if the learning-curve slope is ≥ 0.0008/doubling (read at ~05:00). **NO-GO for "all S1s"** in one window on either box. It is also the lever most likely to *explain* top-1's M ≈ 0.005, so it is the single most important curve to read. |
| F3 | **Collective/cluster stacking incl. within/cross-source shared perturbation (R4)** | M | [RV-lab] +0.0014–0.0024 over stage A on real India (`10` §3 R4). [CITED] Ayan collective stage 2: +0.0021. §1.7 shows the residual is relational, not lexical. | **+0.001–0.0025** on top of v10's stage B | Folded into stage B: +~15 min features | same | **GO, with a new LB-transfer guard.** Already specced (`10` §5.2 b).<br>**Warning (§2.A-4):** four teams' group/sibling features gained on validation and *lost* on the LB. pkd lost 0.956 → 0.921, and **`hide_frac`-style density simulation did not catch it**.<br>R4's cross-source and "shared with nothing" terms push *against* vouching, which is the right direction for test's excess neighbour-house siblings. The within-source term (`sh_maxp`) *is* a vouching signal.<br>**Guard:** compare predicted links/S1 on test against OOF for v10 and for v10 + R4. If R4 raises test links/S1 by more than 2× its OOF rise, ship it only as a parity-half probe. |
| F4 | **Density-matched training world** (train, not just tune, at test's 5.76 recs/S1) | M | [CITED] Ayan P-dense +0.0014 and τ 0.7→0.8. Tony val→public gap 0.031 → 0.0094 after density matching (`08` §IMPLEMENT-1). | +0.000–0.0014 if not already in v10 (`hide_frac=0.19` covers it if the *training* pairs come from the hidden world, which `run_big.py:77` does) | 0 (config) | 0 | **Already in** (`hide_frac` sampling at `run_big.py:77`). Verify only. |
| F5 | **France arm: test-time self-training at scale** (France pseudo-labels at ≥ 99% precision, retrain with them) | g | [CITED] Ayan: +0.0022 (IN→US) / **−0.0017** (US→IN), sign flips (`05` §2). [ARITH] each 0.01 of g is worth 0.0015 LB; top-1 needs g ≤ ~0.01–0.02. | 0 ± 0.002 (sign unknown) | one extra test pass ≈ 3.5 h | ≈ 1.2 h | **NO-GO as a new build.** v10 already has a LOCO-gated selftrain, so keep that gate. Scaling it up without a LOCO read is a coin flip with the downside the same size as the upside. |
| F6 | **France arm: country-agnostic features + test-fitted IDF + French hand dictionaries** (rue/r, bd, av, sarl/sas/sa/eurl, accents) | g | [CITED] official Q&A allows test-fitted IDF and "small hand-written normalization dictionaries" (`official_QA.tsv`, 25 Sep 02:52 and 01:34). [RV] v4 LOCO gap 0.047 = 0.007 LB; `10` §1.3: 93% is ranking loss. | +0.000–0.003 (D1 in `12`) | adversarial-validation ranking ≈ 1 h + retrain | same | **Bundle only**, with LOCO proof (`10` §5.2 rule). v5 France normalisation is already in. |
| F7 | **TF-IDF + dense-embedding dual retrieval** (e.g. model2vec potion-multilingual static, MIT, CPU-fast) | B / g | [CITED] EnsembleLink: dense ∪ sparse union (`12` §1.C); Foursquare 8th: SBERT fine-tuned candidate IoU 0.97 → 0.986 (GPU). [RV] our residual misses are empty-address namesakes (R@8 0.74, **Bayes-hard for any retriever**) and native-script names (R@8 0.973). | +0.000–0.001 (only the native-script slice is addressable) | new ANN index over ~10M vectors: 1–2 h to build and integrate, untested | same | **NO-GO**: new infrastructure, small addressable slice. |
| F8 | **★ GPU cross-encoder on the uncertain band**, stacked as one feature. `xlm-roberta-base` (MIT, 278M) as pkd used, or multilingual MiniLM-L12 (MIT) if on T4, on Kaggle / Colab GPU | M, g | **[CITED, this data]** pkd-prashant v8 → v9: OOF 0.9773 → **0.9828**; M 0.0117 → 0.0062 (**−47%**). AUC on the 0.02–0.98 band 0.985 vs LightGBM 0.931. Their full OOF ladder transferred to LB ≥ 1:1 (v5a → v10r: OOF +0.0141, LB +0.0169), but the cross-encoder step itself was not LB-isolated. [CITED] Foursquare re:waiwai: cross-encoder −27% error. The official Q&A allows it. Counter: our stack already has Ayan-class features, so the relative cut should be smaller than on pkd's v8. | [ARITH on CITED] our M ≈ 0.012 × (−25% … −45%) → **+0.003–0.005 seen**, ×0.85 plus France → **+0.0025–0.0045 LB**. The largest single item on this map. | CPU infeasible for training. Inference on a 5M band with MiniLM-L6 would be ~1.5–3 h, but the boxes are busy. | Not a box job. **Kaggle T4×2 [SPEC throughput: xlm-r-base training ~450 pairs/s/T4 at seq 64 fp16; inference ~2k pairs/s/T4]:**<br>• fine-tune 2 S1-grouped folds, ~0.9M pairs each, in parallel: ≈ 35–45 min;<br>• OOF over the train band (~1.7M): ≈ 10 min;<br>• test band (our ~40 cands/S1 gives an est. 5–8M pairs × 2 fold models): ≈ 1–2 h on 2 T4s;<br>• export, upload and download ≈ 1 h;<br>• **≈ 3–4 h wall** after the export, plus ≈ 2 h to write the script. | **CONDITIONAL GO, as a parallel GPU track** with its own owner. It is *not* part of the box retrain. It is the highest-EV item here and the one mechanism that §1.8 / §2.A say separates 0.985 from 0.990.<br>**Prerequisites:** a phone-verified Kaggle account (or Colab) with GPU quota; Box 1's v10 `out/` with OOF pA, the pair lists and test p, expected ~04:00.<br>**Steps:**<br>1. At ~04:15, export band pairs (0.02 < p < 0.98) as `"<name> \| <address>" [SEP] "<name> \| <address>"` raw text, with the S1-grouped fold id.<br>2. Fine-tune 2 folds, 1 epoch, lr 2e-5, max_len 64.<br>3. Score the OOF band and the test band.<br>4. Join the logit as one extra column into `stage_c.py` via a small `--extra` join on (s1, rid).<br>**Gates:** stage-C OOF ≥ +0.001 **and** LOCO IN→US not negative **and** test links/S1 change ≤ 2× the OOF change (the §2.A-4 detector).<br>**Ship** as a Slot 3 (14:00) parity-half probe or full file. **Hard kill** at 13:00 if the test band is not scored. |
| F9 | Per-noise-mode canonicalisation → exact joins | M | [RV, §1.7] only **0.1% of v4 FNs** and 0.5% of FPs are canonical-exact joins | **≈ 0** | — | — | **NO-GO** (measured). |
| F10 | Hungarian / global assignment; TSV merging; CPU cross-encoders | — | `10` §2, `12` §2: null / negative / infeasible | 0 or < 0 | — | — | **NO-GO** (known). |
| F11 | Learned light-GBDT prefilter replacing prune-28 (keeps recovered F1 records) | B | [CITED] Foursquare 9th: 420M → 3M pairs for −0.007 max-IoU. v4 prune dropped ~2% of positives (`10` §1.3). | +0.001–0.003, **only if paired with F1** (a larger union needs a smarter cut) | ≈ 2 h code + rerun | same | **Conditional.** A cheaper substitute is to protect the reverse view's top-8 in `prune_by_similarity` and raise prune_k to 40 (`10` B-IN). **GO for the substitute.** |

### 3.3 What the retrain bundle can reach, set against top-1
Additive upper bounds, with overlaps making them smaller:
- **F1** (all countries on 32-core; IN+US test side only on 8-core): +0.002–0.0035.
- **F2** at 400k: +0.0014–0.0027.
- **F3**: +0.001–0.0025.
- **F11 substitute**: +0.0005–0.001.
- **Sum: +0.005–0.0097 in the bundle**, plus Slot-1/2 post-hoc items (`10` §4.1: +0.001–0.002).
- From v10 ≈ 0.975 that is **≈ 0.981–0.987**: the 0.985 wall is reachable only at the optimistic edge.
- **The F8 cross-encoder track adds a further +0.0025–0.0045 if it lands**, which makes it the only route that plausibly puts us *above* the wall, at **≈ 0.985–0.99**.

**0.9906 is out of reach.** It needs all three arms at top-1 level at once:
- B ≈ 0.003;
- M ≈ 0.005, which needs F8;
- g ≤ 0.01, with France ≥ 0.975.

We have none of the three measured at that level, and there is one retrain window left.

### 3.4 Ordered GO list for the ~09:00 window
**Box retrain bundle** (one run, one judgement):
1. **F3 (R4 in stage B).** Code specced; synth- and real-lab-verified. Apply the links/S1 LB-transfer guard.
2. **F11 substitute** (protect the reverse top-8, prune_k 40). It is a one-line config plus a small code path.
3. **F1 test-side reverse view for IN and US**, with the train side from the same code path over the train sample. **All 3 countries if the 32-core quota lands.**
4. **F2 at 400k/country**, only if the learning-curve slope is ≥ 0.0008/doubling.

**Parallel GPU track:**
5. **F8 cross-encoder on the uncertain band** (Kaggle/Colab), with its own owner, the export at ~04:15, and a hard kill at 13:00. It stacks onto either v10 or the retrain through stage C.

**Not tomorrow:**
6. F5, F6, F7, F9 and F10 are NO-GO.

## A. Scripts and raw numbers
Scripts are copied to **`research/16-scripts/`** (originals in `SCR/top1/`, where `SCR` = `/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad`). Run them with `code/business_entity_resolution/.venv/bin/python`. The folder also holds the sub-agent notes (`notes.md`, `wincomp_notes.md`) and pkd-prashant's full doc (`pkd_doc.md`).

| Script | What it does |
|---|---|
| `arith.py` | Exact E[macro-F0.5] under the real size prior for (per-record recall r, FP-per-S1 λ); the oracle(PC) mapping; the g → F_seen table |
| `arith2.py` | Required B / PC grid for 0.9906; matcher-loss grid; singleton-FP cost |
| `overlap.py` | Test S1 vs train S1 exact (name, address) / name / address overlap per country (§1.1) |
| `canon_err.py` | Canonical exact-join rates on v4's 6,931 in-candidate error rows (§1.7) |
| `shift.py` | Train vs test per-S1 same-name / same-house / different-house pool counts, with the prep normaliser (§2.E). Run it under `SCR/ramlock.sh`: it peaks at ~1–3 GB. |

`wincomp_notes.md` holds the sub-agent's notes on the Foursquare / Shopee / SIGMOD winners (§2.B).

Raw oracle(PC) mapping, independent-miss model:

| PC | oracle |
|---|---|
| 0.95 | 0.98463 |
| 0.956 | 0.98656 |
| 0.9714 | 0.99140 (measured 0.9898 → our misses are clustered, ratio 0.357 vs 0.301) |
| 0.98 | 0.99404 |
| 0.9905 | 0.99720 (Ayan measured 0.99715 ✓) |
| 0.995 | 0.99853 |
