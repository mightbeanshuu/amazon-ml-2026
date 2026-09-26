# 10 — Frontier plan: v10 (~0.975 expected) → 0.99 (26 Sep 2026, rewritten incrementally from ~21:30 IST)

Status: **LOOP 3, integrator mode (01:00 IST, 27 Sep).** `14`/`15`/`16` are reconciled into §3–§5. §5 is the single runbook, and §6 logs real numbers as they land.

Tags:
- **[RV]** verified on REAL train data (GT + raw text, local scan of `train_source{1,2,3}.tsv` + GT).
- **[RV-box]** read from the running real v10 job's own log.
- **[SV]** synth-verified across 8 seed replicates (with SD).
- **[L]** literature or peer claim, with URL (see `12-journal-sweep-99.md` and `scratch/frontier_web.md`).
- **[S]** speculative.

Deltas are global macro-F0.5 unless marked. Scripts live in `SCR = /private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad/`. Every script is reproduced verbatim in **Appendix A**, since scratch is session-temporary.

## 0. Headline — read this first

1. **0.99 is not reachable with v10's candidate generator. The limit is now measured, not argued.** This is *our* ceiling, not the problem's: top-1 0.9906 proves a better generator exists. Raising it is the retrain bundle's job (B-IN, plus the ceiling-raisers priced in `14`/`15`, integrated in §5.2).
   - Real v10 train blocking for India, read from Box 1's log at 22:26 IST, gave PC 0.9714 and **oracle macro-F0.5 0.9898**.
   - That oracle is taken *before* the similarity prune to 28/S1. The v10 prune costs a further 0.4% of positives (`14` §1); v4's cost 2%.
   - India is 46.8% of the test. A *perfect* matcher on v10's India candidates tops out at ≈0.985–0.99, and France's transfer gap sits on top (§1.3).
   - The top-1 0.9906 therefore needs India blocking at Ayan-class recall (oracle 0.9968) *and* a near-perfect matcher. We have neither in one retrain window.
   - **Correction from `14` §1 (exact replay on Box 1's real `train_block_india.parquet`):**
     - After the v10 prune, train PC is 0.9675 and the oracle 0.9886 (the prune costs −0.0039 PC).
     - The train numbers are **optimistic for test**. Record-side protections (prune_topk record-top-2, `rev_rank ≤ 2`) are computed among only the 120k *sampled* S1s (16.8% of the visible set). At test every S1 competes, so a record has ~6× more suitors.
     - 14 bounded test India post-prune PC at 0.931–0.9675. `15` §2.0's simulation, with the correct per-quarter K-stage semantics, puts it at **≈ 0.962** (oracle 0.9867).
     - This is a train/test skew in the candidate distribution, and the retrain bundle must fix it (§5.2).
2. **Best evidence-backed stack: v10 → +0.0045 conservative / ≈ +0.013 optimistic** (§4.1).
   - The optimistic figure includes T1 (a test-side skew fix) and F8 (GPU cross-encoder, a separate track).
   - v10's *test* India candidate ceiling is below its train measurement: 0.9867 vs 0.9886 per `15` §2.0, corrected from `14`'s 0.948 PC.
   - **Expect X ≈ 0.971–0.974.** The adjudicated test India PC is 0.958–0.962 (`14` §4c IPW 0.958–0.959, `15` 0.962), India oracle ≈ 0.985–0.987 vs 0.9886 train.
   - **T1a** (a test re-run with the same models, on Box3/Box4) is priced by `15` §2.0b: India test oracle 0.9867 → **0.9920**. It recovers the skew and more, because ~30k-S1 ranges let almost every union pair through the K stage.
   - The biggest levers are:
     - **F8, a GPU cross-encoder column** (`16`: on this data a pkd-prashant cross-encoder cut in-candidate matcher loss by 47%). This is the one mechanism that separates the 0.985 wall from 0.99. It runs as a parallel GPU track with a 13:00 hard kill (§5.3).
     - **R4 (new)**: house-number geometry plus *within/cross-source shared-perturbation* features. Real mini-labs measured **+0.0014–0.0024 (India) and +0.0047–0.0057 (US)** over a v10-feature stage A. It runs post-hoc (stage C) with no pipeline re-run, or folded into stage B in the retrain.
     - **B-IN**: India blocking recall, in the retrain window.
3. **Two claims in `09` did not replicate** across 8 synth seeds (§1.4):
   - "rule beats GFM" (+0.0059) → −0.00001 ± 0.0021.
   - "France wants −0.20" → negative.
   - Synth seed SD is 0.002. Treat any single-seed synth delta below 0.004 as noise.
4. **France is a ranking problem, not a threshold problem.** On real v4, threshold retuning closed only 7% of the LOCO gap. Slot 2 as a France-threshold probe has a ceiling of about ±0.0005.

## 1. Deep-scan findings

### 1.1 Match-set sizes: a country-independent prior [RV]
From `train_ground_truth.tsv` (2,206,821 S1):
- Size distribution: 0: 5.58% · 1: 5.40% · 2: 17.0% · 3: 24.1% · 4: 21.9% · 5: 14.6% · 6: 7.5% · 7: 2.9% · 8: 0.85% · 9: 0.19% · 10+: 0.03%. Mean **3.4613**.
- It is **identical across countries**: US 5.58% singletons / mean 3.459; India 5.59% / 3.465.
- Singleton status is independent of address-sharing (5.61% vs 5.58%) and of name-sharing (5.56% vs 5.60%). Singletons are a uniform-random 5.6% with **no records at all**.
- Given a match exists, n2 and n3 are independent (observed/expected 1.02–1.07 everywhere; only the (0,0) singleton cell is ×3.55). Caps: n2 ≤ 5, n3 ≤ 6.
- **France almost surely shares the prior: mean 3.46, 5.6% empty.** This gives two label-free France checks:
  - **Excess-empty** = predicted-empty fraction − 0.0558. It is a lower bound on the share of non-singletons that score 0.
  - **Mean links per S1** vs 3.46 × recall / precision.
  - Real v4 France had 5.72% empty and 3.13 links/S1. That is about the seen countries' v3 levels (IN 7.0% / 3.10, US 6.1% / 3.24), so France is **not** failing at the "empty S1" level.

### 1.2 No leakage [RV]
- The within-entity S2 row span (median 2.04M) is no tighter than random (1.68M).
- S1 vs S2 row-order correlation is 0.003, and id correlation is −0.002.

### 1.3 Where the loss sits: real v4 OOF errors mapped to raw text [RV]
The mapping was validated: 1.000 label consistency on FN rows.
- **v4 blocking loss:** 0.017 IN / 0.011 US. **v4 in-candidate matcher loss:** 0.0157 IN / 0.0180 US, split as:

  | Component | IN | US |
  |---|---|---|
  | FP-only | 0.0075 | 0.0074 |
  | FN-only | 0.0082 | 0.0106 |
  | Singleton FP | 0.0022 | 0.0026 |

  The singleton FPs come from 3.8% (IN) and 4.7% (US) of singletons.
- **FP composition:**
  - **65% generator decoys**: GT-unowned records.
  - 15% hidden-owner: the owner S1 was hidden by hide_frac, so it acts like test's extra unowned pool.
  - 20% visible-other-owner. This is mostly an OOF-sampling artefact: only 17% of competing S1s are scored in OOF, so exclusivity cannot act.
- **Decoys are singleton records, not copy clusters.** 2.9% have a near-identical unowned S2 sibling, the same as a random unowned record (3.1%). Owned records have one 28.5% of the time.
- **Aggregate similarity and cohesion features are exhausted.** FP and FN rows have identical record→S1 similarity (median 83.6 vs 81.9) and record→best-true-sibling similarity (80.3 vs 82.6).
- **Error hot-spots:**
  - **Empty-address records are 18% of all errors** (19.1% of FP, 17.7% of FN) against a 3.36% base rate, 5.5× over-represented. With the address gone, namesake S1s cannot be told apart (35–44% of S1 names are shared).
  - Generator-syllable random names ("Brixjaxvantage", "Halopyrabrix") are 0.33% of S2 and **95% owned** (vs 73%). They are 3.5% of FNs.
- **House numbers** (primary-number test with zero-strip and prefix/suffix truncation):

  | Relation | True IN | True US | Decoys |
  |---|---|---|---|
  | keep | 72% | 76% | 27% |
  | keep+extra number | 14% | 7.5% | 36% |
  | **replaced** | **0.9%** | **3.1%** | **27%** |
  | record has none | 4% | 13% | 2% |

  - "Replaced" carries a likelihood ratio of about 13.
  - Decoys are *near-offset* neighbour houses (108/106, 914/911): |Δ| ≤ 10 for 61% of decoys against 30% of true pairs.
  - The generator *also* perturbs true copies' numbers (7805→7804), so a number mismatch alone is never decisive. §1.5 is what resolves it.
- **France from LB history:**
  - v3 (hide_frac 0.19, so seen ≈ OOF): implied France = (0.947 − 0.4675·0.9488 − 0.3827·0.9627)/0.1498 = **0.895**. v3 LOCO-tuned was **0.896**, so LOCO ≈ France.
  - v4: LOCO 0.9146 against OOF 0.9618, a 0.047 gap in France units ≈ **0.007 of LB**.
  - Threshold retuning on LOCO recovers only +0.003 of that gap (7%). **France is ranking loss, not threshold loss.**
  - v5 added France-gated normalisation, so v10's France is unmeasured until Slot 1 (P4).

### 1.4 Synth instrument resolution; two `09` claims retracted [SV, 8 seeds]
Replicates of synth_v10 were run with seeds 1–7 (`SCR/synth_seeds/`, ~75 s each, identical candidates).
- **Seed SD of the synth test score is 0.0018–0.0020.**
  - Seed 0 (synth_v10) was the unluckiest: 0.95546 against an 8-seed mean of 0.95866.
- **"Rule beats GFM" was noise.** OOF-tuned rule minus stored GFM, per seed: +.0028, −.0001, −.0010, −.0020, −.0019, +.0021. Mean −0.00001.
- **France shift is ≈0.** Mean over 6 seeds in France units: −0.10 → +0.0013 (SD .0053), −0.05 → +0.0011, +0.05 → −0.0011. **`09`'s −0.20 → −0.0009.**
  - Global effect ≤ ±0.0002.
  - Real v4 LOCO: +0.1 gives +0.002 in LOCO units, ≈ +0.0004 global.
- **On synth LOCO, the pipeline's unseen-country penalty (calibrated p − 0.025) costs −0.0021** against no penalty (0.95105 vs 0.95318). The real LOCO check is in `decision_lab_real.py`.

### 1.5 NEW — within-source shared perturbation is the decoy killer [RV] ★ strongest structural finding
The generator builds each source's copies of an entity from **one per-source perturbed base record**. Decoys are one-off singletons.
- **The perturbation is shared within a source.** A true record whose house-type number the S1 cannot explain carries **the same number in another copy of the same source** 82.3% of the time for S2 (n = 113k) and 79.8% for S3 (n = 116k). Across sources it is only **4.9–5.0%**.
  - Example: S1 "9236 Meadowmont View Dr, Charlotte". Every S2 copy reads "9238 … CHRALOTTE", with the same number and the same typo.
- **Decoys share nothing.** For a decoy FP with an unexplained number, that number appears in a true same-source copy of the S1 only **1.7%** of the time (n = 470). For in-candidate **FNs** the rate is **69.0%** (n = 2,132).
  - The likelihood ratio is ≈40 on the tested subset. That subset covers ~41% of FNs and ~54% of decoy FPs.
  - At the token level (any discrepancy token shared) the signal is weaker: 57.6% vs 22.6%, LR 2.5. **Numbers are the sharp channel.**
- **v10 does not capture it.**
  - The sibling "vouch" features score record-record similarity, and a decoy is as similar to the true copies as they are to each other.
  - Enrichment donates tokens from *confident* matches, but the perturbed-base copies are exactly the ones that are not confident.
- **Test-time caveat, measured on real v4 France candidates:** "shared with *any* other candidate" fires coincidentally for 37% of low-p pairs, because neighbour entities' own copies share their number.
  - The usable features are therefore the *conditional* ones: `sh_maxp` (the best p among the sharing candidates) together with the name channel already inside p.
  - These are exactly what a stacked model learns. Hence R4 is built as a GBDT/LR stage with a real-OOF gate, not as a hand rule.
- **Cross-source agreement means ANOTHER entity (real GT; small n = 50, direction clear).**
  - Some FPs are records owned by a *different* S1 (hidden or neighbour). When such an FP has an unexplained number, the owner's copies in the **other** source carry that same number **62%** of the time. For true copies the figure is **5%** (n > 230k).
  - The reason: a neighbour or hidden entity's copies agree on *its* real number across sources, while a true copy's deviation is a per-source artefact.
  - Resulting three-way signature on an unexplained number:

    | Sharing pattern | Meaning |
    |---|---|
    | shared **within** the source only | our S1's perturbed base → **match** |
    | shared **across** sources | another entity → **non-match** |
    | shared with nothing | decoy singleton → **non-match** |

  - These are added as `shx_n` and `shx_maxp`.
- **Label-free look at France (real v4 test, 5.19M candidate pairs):**
  - Among uncertain pairs (0.3 < p < 0.99) that carry an unexplained number, ~47% show the cross-source "other entity" signature, ~30% within-only, and ~22–26% neither.
  - v4 *selected* 70–100% of the 0.75–0.99 band, which is 10.5k such pairs over 259k France S1s.
  - If the train-GT mechanism holds in France (same generator: identical size prior), about half of those selections are other-entity FPs. That is roughly **0.002–0.004 France F**, and it is exactly the class R4 targets.
  - Example: S1 "amicale du neige | 182 r fernand audeguil". v4 selected the S3 record "amicale du neige sa | **191** r fernand…" at p = 0.891. An S2 record "amicale neige holding | **191** r ferannd…" also exists, so 191 agrees across sources, which points to a different entity at 191.
- Script: `SCR/shared_feature.py`, vectorised: 5.19M pairs in 16.5 s.
- The test-time "near-identical-name bucket" check was dropped for memory reasons. The real-OOF gate in `stage_c.py fit` supersedes it.

## 2. What the literature adds (web agent + journal sweep `12`, reconciled)
- **No CPU-feasible single technique worth +0.015.** Both independent sweeps agree.
- **Assignment is solved.** With record capacity 1, the per-record argmax *is* the max-weight b-matching. Hungarian/LP gives 0 ([Papadakis EDBT'22](https://arxiv.org/html/2112.14030v3): greedy UMC F1 .618 vs max-weight BAH .408).
- **Calibrate after blending** ([Wu & Gales 2021](https://arxiv.org/abs/2101.05397)): member-calibrated ECE 3.24 vs post-combination 2.19. This is now implemented in `blend_runs.py --refit`.
  - Our synth blend lab already fitted GFM calibrators on the blended OOF. Logit-average vs prob-average made no difference there: k=2 GFM +0.00198 vs +0.00208.
- **EFP / size prior** ([Dembczynski ICML'13](http://proceedings.mlr.press/v28/dembczynski13.html)): cardinality-aware expected-F beats independence on 5 of 6 datasets (−0.9 … +4.2 F1).
  - Our `metrics.best_set_fbeta` ignores unretrieved matches, so it over-values "empty".
  - The fix is implemented as `best_set_miss` (Poisson(λ_c) extra misses; λ = 0 reproduces the shipped GFM exactly).
  - Counter-evidence on this data: Ayan's `extra_miss` expected-F **did not beat** a tuned τ (`08` §1d). Expected value is small.
- **More training S1s:**
  - Foursquare 8th place: LB 0.908 → 0.928 → 0.948 at 550k → 700k → 1.1M ids.
  - Foursquare 9th place: +0.010 from training on all data.
  - Counter-evidence: WDC Products Magellan-RF is flat.
  - **Gate on our own learning curve** (`learning_curve.py`).
- **Not feasible:** CPU cross-encoders. MiniLM-L6 runs at 478–980 pairs/s, so 60M pairs take ~17–35 h. EnsembleLink is top-1-only and zero-shot.

## 3. Candidate improvements, quantified (loop 2)

| id | item | mechanism | est. Δ global | evidence | needs | tool / file | when |
|---|---|---|---|---|---|---|---|
| **R4** | **Stage C: within/cross-source shared-perturbation + house-number geometry** (`u_any, sh_n, sh_maxp, shx_n, shx_maxp, s1_kept, num_dmin`) stacked on the final p; OOF-gated at +0.0005 | within-source sharing rescues perturbed-base true copies (FN side is ~28% of FN rows); cross-source sharing flags other-entity copies; no sharing flags decoy singletons | **+0.0015–0.003** | **[RV-lab, 2 countries]** **US lab** (same protocol, pair recall 0.975): A 0.96092 / 0.95975 → C[lp] 0.96016 → C[lp+geometry] 0.96417 / 0.96349 → **C[all] 0.96558 / 0.96448** → **A with R4 columns 0.96651 / 0.96545 (+0.0056 / +0.0057 vs A)**. In US, geometry carries most of it (US true pairs have 3.1% replaced and 13% no-number records). Global ≈ 0.468·0.002 + 0.383·0.005 + 0.15·0.002 ≈ +0.003 on a lab-strength base, discounted ×0.5–1 for v10's stronger base. Real India mini-lab, same session (`SCR/real_r4_lab.py`: 15k visible S1 at full 4.13M-record distractor density, 600k pairs, v10 pair features), OOF rule / GFM:<br>• A 0.89404 / 0.89416<br>• C[lp] 0.89301 / 0.89312<br>• C[lp+geometry] 0.89430 / 0.89454<br>• **C[lp+geometry+sharing] 0.89596 / 0.89558**<br>• **A with R4 columns 0.89648 / 0.89559**<br>So **+0.0014–0.0024 over A**, and the sharing columns alone add +0.0010–0.0017 on top of geometry. Positive rate given an unexplained number: within-only 3.8% vs cross 0.5% vs neither 0.9%. Synth end-to-end also runs (synth has no such signal). | train_features + oof + sib npz + prep; **no re-run** | `SCR/stage_c.py fit` then `apply --runs BOX2` | 04:00–05:00 → Slot 2 |
| **T1** | **Sampling-consistent test candidates and features** (train/test skew fix, from `14` §1). At test, compute the record-side **protection** ranks (prune_topk record-top-2 and `prune_by_similarity` rev_rank ≤ 2) **within random S1 groups the size of the train sample (~16.8% of S1s)**: G ≈ n_s1_test / 120k groups by `s1_i % G`.
- **T1a** (recommended): protection only. Features stay global. Global rank makes false pairs look *more* false than in train, which may help precision, so do not remove that.
- **T1b**: also partition the `rev_rank` / `b_key_revrk` / stage-B `p_rk2` / `p_gap2` features. Riskier; LB-probe it only if T1a wins.
- In both, exclusivity stays *global* (the real constraint). | The model was trained on protections and rank features measured among 120k sampled S1s. At test they are measured among all S1s (~6× more suitors), which shifts both the candidate set and the features. | **+0.001–0.003.** **Priced by `15` §2.0b (01:05 IST, test semantics, real India v10 table):**
- v10 test: PC 0.9618, oracle 0.9867.
- **Configured launch cfg** (`test_range_s1` 30000 → ~27 ranges, `test_group_s1` 120000 → G = 7 ≈ `T1a_K24_G6`): **PC 0.9766 (+0.0148), oracle 0.9920 (+0.0053 India ≈ +0.0025 global ceiling)**, at 2.62× final pairs.
- The K-side half alone (`s1_parts`≈24) gives +0.0075 PC and oracle 0.9898 at +4% volume. It is strictly better than K=80, and **K=80 is redundant on top of it** (testK already hits the union ceiling).
- Realised ≈ 40–60% of the oracle gain (namesake-heavy). US shares the skew.
- `14` and `15` differ on v10's *as-is* test PC (0.949 vs 0.962; `15` models the per-quarter K stage correctly). Both agree on the T1a *target*. | mechanism [RV-box, exact replay in `14`]; delta unmeasured, since test labels are needed. Proxy: the LB (full-file swap). | **test-phase re-run only, no retrain:** block + features + score ≈ 3.5–4.5 h on 8-core with the **same v10 models** | `block_big.block_country(s1_parts=…)` via a new `cfg["test_s1_parts"]` in `run_big.test_block_phase`; `run_big.prune_by_similarity` gains an `s1_groups` arg for `group_rank(ri*G + s1_i%G, …)`; the same grouping in `stage_b_frame` at test | Box 2 as soon as v10 finishes (~04:15) → Slot 2 candidate (~08:30) |
| **F8** | **GPU cross-encoder column** (xlm-roberta-base, uncertain band 0.02 < p < 0.98, 2 S1-grouped folds) stacked into stage C via `--extra-train/--extra-test` | pair-level text model catches what similarity features miss | +0.0025–0.0045 if delivered (`16` F8) | **[CITED on this data]** pkd-prashant: M 0.0117 → 0.0062 (−47%), band AUC 0.985 vs LightGBM 0.931. Not LB-isolated. | GPU (Kaggle T4×2 or the RTX 5060 laptop); **not** a box job | `SCR/export_band.py` (Box 1, 04:30) → `SCR/ce_train.py` (GPU) → `stage_c.py --extra-*`. The stage-C join path is smoke-tested on synth with a fake logit column. | track C → Slot 3 |
| P1 | Box1 + Box2 blend, **calibrate after blending** | variance reduction; isotonic refit on blended OOF | +0.0003–0.001 | [SV] k=2 GFM +0.0021 / rule +0.0009; k=4 +0.0014 / +0.0022 (synth has high model variance, so expect less on real) | both out dirs on one box | `SCR/blend_runs.py --refit --prep PREP --runs B1 B2` | Slot 1 |
| P1-x | ~~TSV extras/strict merge~~ | — | **−0.0002 / −0.0003** | [SV] 7 of 8 pairs negative | — | — | never |
| P2 | Decision rule: rule A vs calibrated GFM | — | **rule ≥ GFM on real (+0.0006–0.0009), honestly measured** | [SV] null over 6 seeds. **[RV-lab] US honest 2-fold: rule 0.96063 vs GFM 0.95971; with R4 columns 0.96606 vs 0.96542.** The pipeline's `use_d` gate compares *in-sample* calibrated GFM, which flatters GFM. | box OOF | `decision_lab_real.py` prints `honest_2fold`. **Ship the rule unless GFM wins honestly by ≥ 0.0005**; `blend_runs.py --decision rule` | Slot 1 |
| B-IN | **India blocking recall**: add Ayan's exact reverse *combo* view (sklearn TF-IDF char_wb-3 name ⊕ word address, max_df .05, record→S1 top-8), protect its top-8 in `prune_by_similarity`, prune_k 28 → 40 | raises IN oracle 0.9898 (pre-prune) toward Ayan's 0.9968 | +0.001–0.002 (ceiling +0.0033 = 0.007 IN × 0.468) | **[RV]** that view ALONE on real India (hide 0.19 world, 30k owned records, our normaliser): **R@1 0.953, R@8 0.978, R@40 0.989**, above v10's whole 6-family union PC 0.9714. Misses concentrate in empty-address records (R@8 0.741, n = 1,168) and native-script names (0.973); other records reach 0.992. Cost: ~45–90 min per split on 8 threads (30k queries took ~40 s), which does not fit an 8-core window. **32-core only**, `sp_matmul_topn(n_threads=30)`. | full re-run + ~1.5 h coding, synth-tested | `block_big.py` new `KEY_FIELDS` entry (dense view, not hashed) | retrain window |
| R3 | train_per_country 120k → 400k | more rare generator ops seen | +0.001–0.004 if the curve slope ≥ 0.0008 per doubling; else 0 | [L] Foursquare; gate = our curve | full re-run (the train-block cache is per-sample) | cfg `{"train_per_country": 400000}`; `SCR/learning_curve.py` first | retrain window |
| A4 | ~~GFM with Poisson(λ_c) unretrieved-match correction~~, plus GT caps n2≤5 / n3≤6 | fixes best_set_fbeta's over-valued "empty" | **≤ 0: DROP λ** (caps ≈ 0) | **[RV-lab] negative.** US lab, honest 2-fold (calibrate/tune on half the S1s, score the other half): GFM 0.95971 → GFM(λ = 0.088) 0.95933 → GFM(λ/2) 0.95951. Same sign with R4 columns (0.96542 → 0.96527). Consistent with Ayan. | real OOF | `SCR/decision_lab_real.py` | 04:30, fold into Slot 2 if > +0.0005 |
| **P6** | **Global precision shift for the test decoy surplus** (all thresholds +δ, δ ∈ {0.03, 0.05}) on a **parity half**, stacked with another probe per `09` §1.5 | `16` §2.E [RV]: per unique-name S1, same-name/different-house pool records rise train→test by **+46% (US)** and **+19% (India)**, with own-copy counts flat. OOF therefore under-states test FPs in the neighbour-house class, and `hide_frac` does not reproduce it (pkd). | ±0.001 (sign unknown until measured). R4 attacks the same class from the feature side. | [RV mechanism] / delta unmeasured | none (re-decide from saved p) | `blend_runs.py --shift δ` (implemented; synth run OK); build the parity-half file with `probe_kit.py mix` | Slot 2/3 parity half |
| P3 | France decision choice via LOCO (GFM-with-other-calibrator vs penalty −0.025 vs rule ± shift) | calibration under shift | ≤ ±0.0005 | [SV] ≈0; synth LOCO says the −0.025 penalty costs 0.002 in LOCO units (≈0.0003 global) | real LOCO | `decision_lab_real.py` (loco block) | fold into Slot 2 France rows |
| P4 | Implied-France read-out | France_impl = (S_slot1 − 0.4675·OOF_IN − 0.3827·OOF_US)/0.1498 | instrument | [RV] (v3 check) | Slot 1 score + run_log `oof_by_country` | arithmetic | Slot 1 |
| R1 | Ayan's 16 collective + Tony's pool/rival features in stage B | cohesion / competition | +0.0005–0.0015 over our stage B | [L] | retrain | `run_big.stage_b_frame` | only if the retrain happens anyway |
| D1 | France feature-invariance (adversarial validation) | drop or quantile-normalise the top-shift features | +0.000–0.003 | [L] DADER; speculative magnitude | LOCO + retrain | new | only with LOCO proof |
| P5 | S1-side rescue search for predicted-empty S1s | excess-empty over 5.58% | ≤ +0.0008 | [S] | 2–3 h build | — | skip unless IN excess-empty > 1% |
| B2 | Metric-aware sample weights | align logloss with per-S1 F0.5 | +0.000–0.002 | [S], no ablation anywhere | retrain | `model._train` weights | bundle only |
| E1 | CSLS hubness score/features | hub correction at the prune | +0.000–0.002 | [L] | retrain | `prune_by_similarity` | bundle only |

## 4. The stack and why it stops short of 0.99

### 4.1 Stacked estimate (v10 = X, the Slot-1 truth)

| step | Δ conservative | Δ optimistic | evidence grade | gate |
|---|---|---|---|---|
| P1 blend + calibrate-after-blend (track A) | +0.0003 | +0.001 | SV | blended OOF ≥ best single OOF |
| P2 honest decision choice (rule vs GFM) + P3 France | 0 | +0.001 | RV-lab (rule +0.0006–0.0009 honest in US) | honest_2fold Δ ≥ +0.0005 |
| R4 + R4b stage C (track A) | +0.0015 | +0.003 | **RV-lab**: IN +0.0014–0.0024, US +0.0047–0.0057, R4b +0.0005 (lab-strength base) | OOF Δ ≥ +0.0005 **and** pkd detector ok |
| T1a test re-run (track B; Box3/Box4; cfg ≈ `T1a_K24_G6`) | +0.001 | +0.003 | **RV-sim (`15` §2.0b):** India test PC 0.9618 → 0.9766, oracle 0.9867 → 0.9920 (+0.0025 global ceiling), 2.6× volume | LB full swap / India splice |
| F8 cross-encoder column (track C) | 0 (may not deliver) | +0.0045 | CITED on this data (pkd M −47%, `16` §2.A-2) | CE AUC > p AUC; stage-C Δ ≥ +0.001; LOCO ≥ 0; detector ok |
| Retrain bundle (track D: protection fix + R4 in stage B + 400k? + B-IN on 32-core) | +0.0005 | +0.004 | RV-box / RV-lab / L; overlaps with R4 and T1 | OOF + 0.001, LOCO ≥, IN oracle ≥ |
| **Total** (overlaps shave the optimistic sum) | **≈ +0.0045** | **≈ +0.013** | | |

- **X is below the OOF-implied value.** The adjudicated test India ceiling sits 0.0008–0.0017 (global) below train (`14` §4c vs `15` §2.0), so **X ≈ 0.971–0.974** pending the CV-based pre-registration in §5.5.
- Adding the stack: **≈ 0.977–0.980 conservative / ≈ 0.985–0.988 optimistic**.
- **Caution:** `14`'s skew finding means X itself may land *below* 0.975.
- **0.99 would need X ≥ 0.977 plus every optimistic gate, including the GPU cross-encoder delivered by 13:00.** That is possible in principle but is not the plan's central case.

### 4.2 Exactly why 0.99 is out of reach, and what would change the verdict
1. **The India ceiling (measured).** The v10 India oracle is 0.9898 *before* the prune and 0.9886 after it, on train. It is lower at test because of the sampled-S1 protection skew (`14` §1: test PC 0.931–0.9675).
   - Loss from India blocking alone ≥ 0.0102 × 0.468 = **0.0048 global**, even with a perfect matcher.
   - The peer best (Ayan) reached 0.9968 India with 112 cands/S1 on 32-vCPU Modal boxes. Matching that is the retrain's B-IN item, and it is at best half-achievable in one window.
2. **The matcher is near its feature Bayes-wall.** v4's residual FPs and FNs are indistinguishable on every aggregate similarity (§1.3). The only sharp new signal found is R4, worth ≈ +0.0015–0.003.
3. **France.** v3 and v4 show France ≈ LOCO ≈ 0.05 below the seen countries in France units, ≈ 0.007 global. 93% of it is ranking loss that post-hoc methods cannot touch.
4. **Arithmetic.** Seen countries need ≥ 0.992 with France ≥ 0.98 for a global 0.99. The best *local* seen-country number in public (Ayan, 0.987, no France) is already below that.

**Measurements that would change the verdict:**
- (a) Slot 1 ≥ 0.981 (v10 is better than every estimate).
- (b) Implied France ≥ OOF − 0.01 (the France gap has closed).
- (c) The R4 real-OOF gate returns > +0.004.
- (d) The v10 US oracle ≥ 0.997 **and** the prune loss is < 0.5%. **Measured: US 0.9965 (train, pre-prune) and IN prune loss 0.4% (`14`). This is essentially met, but it does not reopen 0.99 alone.**

Any two of these re-open 0.99. All are read by 05:00 on 27 Sep from the tools in §5.

## 5. Tomorrow's runbook — SINGLE SOURCE OF TRUTH (integrates `14`/`15`/`16`; IST, 27 Sep)

**⚠ Timing update (00:20 IST):** US train cgram is taking ~100+ min on both boxes (India took 44). Projected v10 finish is **~05:30–06:30 IST**, not 03:45. Shift the Track A/B/C start times below by about +2 h. The Slot times stay; T1 moves to Slot 3 unless it is spliced India-only. v10 `models/` exist at the end of the train phase (~02:00), so a spare CPU could start T1 then.

**Tracks (run in parallel, each with its own owner):**

| Track | Machine | Content |
|---|---|---|
| **A** | Box 1, 03:45–05:30 | instruments + post-hoc files |
| **B** | Box 2, 04:15–08:30 | T1 test-phase re-run with the same v10 models |
| **C** | GPU, 05:00–13:00 hard kill | F8 cross-encoder |
| **D** | Box 1 after A, or the 32-core box | the retrain bundle |

- Heavy local jobs on the Mac go through `SCR/ramlock.sh`.
- All `SCR` tools are reproduced verbatim in Appendix A. Copy them to `~/lab/` on the boxes and run them from `~/er`.

### 5.1 Track A — Box 1 instrument pass (03:45–~06:00), in PRIORITY order
Precondition: copy Box 2's `out/` into `~/out2`. The run dirs stay read-only; outputs go to `~/lab/`, and the stage-B OOF rebuild is cached in `~/lab/cache`.

**Real-scale sizes (measured 00:56):** v10 train has 8.06M (IN) + 9.36M (US) = **17.4M OOF rows**. Durations below are estimated for 8 cores. Track A takes ≈ 2.5 h sequential from the moment v10 finishes: the Slot-1 file is ready ≈ +35 min, stage C ≈ +1 h 45.

Run steps 1 → 2 → 3 sequentially, because they share the cached stage-B OOF. Run step 6 in parallel from the start (it is I/O-bound).
1. **Slot-1 file** (~25–35 min, most of it two stage-B OOF rebuilds, cached for later steps):

   ```
   python ~/lab/blend_runs.py --refit --how logit --decision rule --prep ~/prep --runs ~/out ~/out2 --s1 ~/a/test/test_source1.tsv --out ~/lab/blend
   ```

   - Record `corr(p0,p1)` and both OOF scores.
   - Switch to `--decision gfm` only if step 3's `honest_2fold` says GFM wins by ≥ 0.0005.
2. `python ~/lab/stage_c.py fit --prep ~/prep --out ~/out --work ~/lab/stc` (~45–60 min at 17.4M OOF rows). This is R4 + R4b.
   - Gate: min(in-sample, honest) Δ ≥ +0.0005.
   - It prints the OOF side of the pkd detector.
   - If it passes, run `stage_c.py apply --prep ~/prep --out ~/out --runs ~/out2 --work ~/lab/stc` (~15 min), then `blend_runs.py --runs ~/lab/stc/run_C --decision auto --s1 … --out ~/lab/blendC`. **This is the Slot-2 R4 candidate.**
   - Read the detector line (`run_C/detector.json`). `red_flag` → **parity-half probe only**. Four teams' group features gained on validation and lost on the LB (`16` §2.A-4; pkd 0.956 → 0.921).
3. `python ~/lab/decision_lab_real.py --out ~/out --prep ~/prep` (~30–40 min).
   - **`honest_2fold` decides rule vs GFM**: rule unless GFM wins by ≥ 0.0005; the US lab had the rule ahead by +0.0006–0.0009.
   - The LOCO block decides the France family: change France only at ≥ +0.003 in LOCO units.
   - A4 is dropped (negative on the real lab).
   - **Log the v10 reference detector ratio here**: test mean_links/S1 (`test_<c>_stats.json`) vs OOF-decision links/S1.
4. `python ~/lab/learning_curve.py ~/out --threads 8` (~25–35 min). This gates R3 (400k) in track D.
   - If track D must launch before this finishes, launch at 120k and treat R3 as off.
5. Hand track D its launch decision (§5.4) by ~06:00.
6. (parallel, from 03:45) **F8 export**: `python ~/lab/export_band.py --prep ~/prep --data ~/a --out ~/out --dst ~/ce` (~10–20 min). Push the output to the GPU owner (private Kaggle dataset or scp) by ~05:00.

### 5.2 Track B — T1 test re-run (architecture as of 00:45 IST)
**Where it runs:**
- **Box3** (r6a.xlarge, 2 physical cores): India, then France.
- **Box4** (r7a.xlarge, 4 cores): US.
- Both autopilot from Box1's `~/out/models` (~02:00). Box1 and Box2 stay free for their own v10 runs, tracks A and C, and the retrain.

**Timing risk:**
- India test blocking alone (cgram) took ~60+ min on 8 threads for v10. `_topk_chunks(threads=8)` is hard-coded, so it oversubscribes 4 vCPUs. T1 also doubles India feature volume.
- **India on Box3 may take 5–10 h (landing ~07:00–12:00).**
- **Re-point rule:** if Box3 has not finished India's `test_block` phase by the time Box1's v10 completes (~05:30–06:30), start India on Box1 as well. The first to finish wins.

(Original single-box plan below, for reference: India test candidates ×1.7–2.4 per `14` §4.)
- Same v10 models: copy `~/out/models` into a fresh run dir with `train_*.json`. Do not copy `test_*.parquet`, or the phase would skip.
- Apply the tested patch `SCR/t1a.patch` (Appendix A.11; `patch ~/t1code/src/run_big.py < t1a.patch` on a COPY of `~/er`; +40 lines to `run_big.py`, default behaviour unchanged). It adds three things:
  - `cfg["test_range_s1"]`: prune_topk's record-top-2 ranges sized like train's ~30k sampled S1s per range.
  - `cfg["test_group_s1"]`: `prune_by_similarity` protection `rev_rank ≤ test_prot` ranked within `s1_i % G` groups, with G = ceil(n_s1/120000). The `rev_rank` feature stays global (T1a).
  - `cfg["keep_test_ckpt"]`: keep `test_<c>_pairs.parquet`. It also always writes a `pA` column.
- **Regression check on synth:** default cfg reproduces v10 exactly (0.95546, identical candidate counts). With grouping forced (G=3) the code path runs: 0.956103, +0.7% candidates.
- Run command:

  ```
  python -m src.run_big --prep ~/prep --data ~/a --out ~/t1 --phase all --cfg '{"test_group_s1":120000,"test_range_s1":30000,"keep_test_ckpt":1,"feat_workers":7}'
  ```

- Output is a full test file from the same models, judged by an **LB full swap**; there is no OOF read, because candidates change.
- **Prioritise India, then US, then France.** The run is per country (resumable), so if time runs short, splice T1 countries into the v10 file by S1. Splicing is exact because macro-F is per S1.
- T1 needs only a finished `models/` dir plus `train_*.json`. **Start it on whichever box finishes its v10 run first**, using that box's own models. Its blend partner is the other box's v10 output, through `blend_runs.py` on the common pairs.
- Expect test links/S1 to *rise* moderately, since protected positives are restored. Log per-country pairs/S1 against v10's.

### 5.3 Track C — F8 GPU cross-encoder (owner: GPU laptop or Kaggle T4×2; hard kill 13:00)
- **Smoke test first** (~2 min): `python ce_train.py --band ~/ce --out ~/ce_smoke --max-train 2000`, killed after fold 0 starts inferring. `ce_train.py` is only syntax-checked here (no torch on the Mac); `export_band.py` is smoke-tested on synth.
- `python ce_train.py --band ~/ce --out ~/ce_out --model xlm-roberta-base --max-len 64 --bs 64 --epochs 1 --lr 2e-5 --kill-at 13:00`
  - Fallback on an 8 GB GPU: `--model microsoft/Multilingual-MiniLM-L12-H384 --bs 128`.
  - It trains 2 S1-grouped folds, then writes `ce_oof.parquet` (train band, out-of-fold) and `ce_test.parquet` (test band, mean of the 2 fold models).
  - It prints the OOF AUC against stage-A p on the same rows. **Go signal: CE AUC > p AUC** (pkd: 0.985 vs 0.931).
- Join the results on Box 1: `stage_c.py fit … --extra-train ce_oof.parquet`, then `apply … --extra-test ce_test.parquet`.
- **Gates** (all three, `16` F8):
  - stage-C OOF Δ ≥ +0.001 over stage C without CE;
  - LOCO not negative;
  - pkd detector ok.
- Deliverable by ~13:00 → **Slot 3**.

### 5.4 Track D — retrain bundle (ONE run, ONE judgement)
- **Box:** Box 1 after track A (~05:30), or the 32-core box if the quota lands.
- **Launch:** by 09:00 at the latest. On 8-core this finishes ~13:30 if launched 05:30, and ~17:00 if launched 09:00.
- **Config:**

  ```
  {"train_per_country": 400000 (only if the R3 slope ≥ 0.0008/doubling, else 120000),
   "K": 80 (K-cap fix: `14` §5 recovers +0.82–1.32 IN PC pts; pending `15`'s CPU pricing),
   "prune_k": 40, "feat_workers": 7, "seeds": [0, 1],
   "test_group_s1": 120000, "test_range_s1": 30000, "keep_test_ckpt": 1}   # T1a at test (patch A.11)
  ```

- **Code changes:**
  - (a) **K-cap exemption `ex_rev2`** (`15` §2.0a, ~10 lines in `block_country` + `prune_topk`): keep any pair whose S1 is rank ≤ 2 in the **record's own reverse list** of any family.
    - That rank is over ALL S1 in train and test alike, so it is **train/test-consistent by construction**.
    - Test PC +0.0073, India oracle 0.9867 → 0.9897, at +2% final volume. It beats K=80 (+0.0047).
  - (a2) **T1a grouping at test** (patch A.11 cfg) for the remaining sampled-S1 protections (prune `rev_rank ≤ 2`, per-range record-top-2).
    - With ex_rev2 in place, use `test_range_s1` 30000 but consider `test_group_s1` = 0 (off) if volume matters. The prune-side grouping doubles final volume (`15` §2.0b: 31 → 64–82 pairs/S1, ≈ 35 s per pair/S1 on India test).
  - (b) R4 + R4b folded into `stage_b_frame`, via `shared_feature.compute` / `compute_tok` / `num_dmin` with pA. The same code runs at test.
  - (c) **32-core only**: B-IN dense reverse combo view (sklearn TF-IDF char_wb-3 name ⊕ word address, max_df .05, top-8), with its top-8 protected in the prune. Raise `PARAMS["num_threads"]` and `_topk_chunks(threads=)` to 30.
  - (d) Keep `test_<c>_pairs.parquet` and write the pA column.
  - **Nothing else** (no D1/B2/E1/R1). An uninterpretable judgement is worse than a missing +0.001.
- **Judgement** (from `run_log_train.json`, before its test phase ends), all three required:
  - OOF ≥ v10 OOF + 0.001;
  - A_loco_tuned ≥ v10's;
  - IN blocking oracle ≥ v10's (0.9898 pre-prune).
  - If any fails, it is an ensemble member only (prob-blend via `blend_runs.py`).
- **Wall-clock:**
  - 8-core: train blocking ≈ 2.6 h + features and training (400k×40×2 ≈ 32M pairs) ≈ 1.5 h + test ≈ 3.5–4 h → **≈ 8 h**. At 120k per country it is ≈ 6.5 h.
  - 32-core (threads raised to 30, B-IN included): ≈ 3–3.5 h.

### 5.5 Slots (LB ranking = max over uploads; the final ZIP must reproduce the best file byte-identically)
- **Slot 1 (08:30): track-A blend (P1).**
  - Record S1 and check the 6-decimal display.
  - **Pre-register the prediction** before the upload, from Box 1's `run_log_train.json`:
    - S1_pred = 0.4675·OOF_IN + 0.3827·OOF_US + 0.1498·A_loco_tuned − skew allowance [0, 0.005].
    - LOCO proxies France (v3: LOCO 0.896 ≈ implied France 0.895). The skew allowance covers `14`'s train/test protection gap.
    - Skew allowance = **0.0008–0.0017**, from the `14`/`15` adjudication. The band from OOF alone is recorded in §6 once the CV lands.
    - A miss below the band by more than 0.005 means a bug or a larger skew. In that case, prioritise track B (T1) over everything else.
    - **Not commissioned:** `15`'s rtop-8 cgram re-run (44 min) to close the last ~0.1 PC point between `14` and `15`. The Slot-1 LB read measures X directly, and the T1a config is unaffected.
  - **Implied France is ±0.02 only** (`16` §7: blanking probes show the public subset is *not* country-proportional; they imply F_US = 1.08). Use it as a coarse flag only.
- **Slot 2 (11:00): the better of** T1 full file (track B) and R4 stage C (track A step 5), **as a full file** — unless the pkd detector is red, in which case run a parity-half probe.
  - If both T1 and R4 are ready and green: a stacked probe with T1 on one parity half and R4 on the other (`09` §1.5 algebra, exact against the Slot-1 anchor) measures both at once.
- **Slot 3 (14:00): best validated combination** (T1 candidates + R4 stage C [+ CE column if track C passed]), full file.
  - Optional instead: a **France-blank probe** for an exact France read (`16` §7). Spend it only if France decisions are still open.
- **Slot 4 (17:00): retrain bundle** (track D) full, or its prob-blend with Slot 3's config. Predict the score first from the Slot 1–3 reads.
- **Slot 5 (19:30): safety.** Resubmit best-known-good if anything misfired. Hard stop 20:30.
- **Kill rules:**
  - No TSV-level merges.
  - No France threshold change without LOCO ≥ +0.003 (LOCO units).
  - No group-feature variant as a full swap with a red pkd detector.
  - Nothing unmeasured in a full-test slot, except T1, which can only be measured on the LB and so is itself the measurement.

## 6. Loop 3 log (appended as real numbers land)
- **00:56 IST, Box 1 train prune + features:**

  | | pairs | per S1 | positives kept |
  |---|---|---|---|
  | India | 9.83M → 8.06M | 67.18 | 0.996 |
  | US | 11.03M → 9.36M | 77.97 | 0.9977 |

  - India features: 8.06M × 89 in 173 s (≈ 47k pairs/s at 7 workers).
  - **T ≈ 17.4M OOF rows**, so the track-A tools will run toward the upper end of their time estimates.
- **00:49 IST, Box 1 `[train us] blocking after top-40`: PC 0.9883, oracle 0.9965**, mean_cands 91.95 (sibling expansion +1,261,701), p95 218, max 8,199.
  - US is already at Ayan-class (Ayan US 0.99741), so **the ceiling problem is India-specific.**
  - US cgram took 6,742 s (112 min) and siblings 736 s, so the US train-blocking pass took 2 h 21 min.
  - Train-side global ceiling, pre-prune, with France assumed ≈ 0.99: ≈ 0.4675·0.9898 + 0.3827·0.9965 + 0.1498·0.99 ≈ **0.992**.
  - After the prune and the test-protection skew it is ≈ 0.990. **0.99 needs a near-perfect matcher on top of that.**
- 22:26 IST, Box 1 `[train india] blocking after top-40`: **PC 0.9714, oracle 0.9898**, mean_cands 81.95 (sibling expansion +870,526 pairs), p95 218, max 5,830. v4 was PC 0.9561 / oracle 0.9833. The India ceiling rose by +0.0065 but is still < 0.99.
- Timings (Box 1, 8 cores): keys ~9 min, **cgram 2,612 s (44 min)**, pkey 87 s, siblings 409 s. That is ~62 min per IN train-blocking pass.
- 23:45 IST, **R4b rare address-token sharing** (`shared_feature.compute_tok`).
  - GT mechanism: a true record's rare address-discrepancy token (typos like "chralotte") recurs in a same-source copy 55.3% of the time, cross-source 4.8%, decoys 10%. LR ≈ 5.5, and it covers ~15% of true records.
  - US lab: C[all+tok] 0.96618 / 0.96498 vs C[all] 0.96558 / 0.96448, so **+0.0005–0.0006 in stage C**.
  - Folded into stage A it is mixed (rule +0.0003, GFM −0.0004).
  - Verdict: optional small add-on; include it in stage C, not decisive.
- 23:20 IST, **R4 US mini-lab** (`real_r4_lab.py 15000 US`): A 0.96092 → **A+R4 0.96651 (+0.0056)**. Honest 2-fold decision test on the same artefacts (`SCR/lab_decisions.py US`):
  - Rule beats calibrated GFM by +0.0006–0.0009.
  - The Poisson(λ) unretrieved-match correction *hurts* GFM by −0.0002–0.0004, so A4 is dropped.
- 22:50 IST, **R4 real mini-lab** (`SCR/real_r4_lab.py 15000`, 12 min). Candidate pair recall was only 0.84 (forward top-40), so absolute scores are low, but the *increments* are what count:
  - Stacking a GBDT on logit(p) alone costs −0.001 at 600k rows. This is stacking noise, and why the gate exists.
  - Number geometry recovers +0.0013.
  - The sharing columns add +0.0010–0.0017.
  - Folding everything into stage A: **+0.0024 (rule) / +0.0014 (GFM) vs A**.
  - Verdict: R4 is real on real data. Prefer folding it into stage B/A when a retrain happens; the post-hoc stage C has to clear its own stacking noise, which the gate enforces.
- 22:35 IST, local real-India measurement (`SCR/bin_recall.py India 30000`, 6.5 min): Ayan's reverse combo view gives R@1/3/8/20/40 = 0.953/0.970/**0.978**/0.985/0.989. A single view beats v10's whole 6-family union (0.9714). The union of the two is therefore ≥ 0.978; Ayan's 5-view union reached 0.9905. Remaining misses are **empty-address namesakes** (R@8 0.74). Those are the same class as 18% of matcher errors: Bayes-hard by name alone.
- Negative result [RV]: name discrepancies are **per-copy, not per-base**. A rare name token that is absent from the S1 is shared with a same-source sibling only 3.0% of the time (cross-source 2.9%, n ≈ 17–20k empty-address true records). The shared-perturbation channel is **address/number-only**, so R4 is kept to numbers.

## Appendix A — tool sources (verbatim copies of SCR/*.py; scratch is session-temporary)
Recreate each file with its name below, place it next to the pipeline `src/` dir (or anywhere; they add the cwd to sys.path), and run from the code dir. `stage_c.py` and `decision_lab_real.py` import `shared_feature.py` / `blend_runs.py` from their own directory.

### A.1 `shared_feature.py`
```python
"""Within-source shared-perturbation features (item R4 in research/10).

Real-data finding (26 Sep, train GT + S1/S2/S3 text): copies of one entity inside ONE source share a perturbed base
record. When a true record carries a house-type number that the S1 cannot explain (not equal, not a prefix/suffix
truncation), the SAME number appears in another same-source copy 80-82% of the time (cross-source only 5%).
Generator decoys are singleton records: their unexplained number is shared with a true copy only 1.7% of the time.
So "my unexplained number also appears in another same-source candidate of this S1" separates a true copy with a
perturbed base (LR ~40 vs decoys) from a neighbour-house decoy.

compute(pairs, s1_nums, r_nums, r_src, p=None) -> DataFrame aligned to pairs with
  u_any       1 if the record has a <=4-digit number the S1 cannot explain, else 0 (NaN if the record has no numbers)
  sh_n        max over its unexplained numbers of #OTHER same-source candidates of this S1 carrying that number
  sh_maxp     max p of those other candidates (0 when none; needs p)
  shx_n/shx_maxp  same, but over the OTHER source's candidates (cross-source agreement = another entity)
  s1_kept     1 if some S1 number is present in the record (equal or truncation), 0 if the record has numbers but
              none explains an S1 number ('replaced'), NaN if either side has no numbers
All vectorised with pandas merges; ~1-2 min per 10M pairs.
"""
import numpy as np
import pandas as pd


def _explode_nums(ids, nums, maxlen=4):
    """ids: array of row keys; nums: array of space-joined number strings -> DataFrame(key, num)."""
    s = pd.Series(nums, index=ids).str.split()
    e = s.explode().dropna()
    e = e[(e.str.len() > 0) & (e.str.len() <= maxlen)]
    return pd.DataFrame({"key": e.index.values, "num": e.values})


def _explainers(e):
    """For each S1 number, the set of strings it explains: itself + all proper prefixes + all proper suffixes."""
    out = [e]
    L = e.num.str.len()
    for k in range(1, 4):
        m = L > k
        out.append(pd.DataFrame({"key": e.key.values[m], "num": e.num.values[m].astype(str)}).assign(
            num=lambda d: d.num.str[:-k]))                       # prefix (drop last k)
        out.append(pd.DataFrame({"key": e.key.values[m], "num": e.num.values[m].astype(str)}).assign(
            num=lambda d: d.num.str[k:]))                        # suffix (drop first k)
    x = pd.concat(out, ignore_index=True)
    return x[x.num.str.len() > 0].drop_duplicates()


def compute(pairs, s1_nums, r_nums, r_src, p=None):
    si = np.asarray(pairs["s1_i"].values); ri = np.asarray(pairs["r_i"].values)
    n = len(si)
    pid = np.arange(n)
    pp = np.zeros(n, np.float32) if p is None else np.asarray(p, np.float32)
    # record numbers per pair
    rn = _explode_nums(pid, np.asarray(r_nums, dtype=object)[ri])
    rn["s1"] = si[rn.key.values]
    rn["src"] = np.asarray(r_src)[ri][rn.key.values]
    rn["p"] = pp[rn.key.values]
    # S1 explainers
    us1 = np.unique(si)
    s1e = _explainers(_explode_nums(us1, np.asarray(s1_nums, dtype=object)[us1]))
    s1e = s1e.rename(columns={"key": "s1"})
    s1e["expl"] = True
    rn = rn.merge(s1e, on=["s1", "num"], how="left")
    rn["expl"] = rn.expl.fillna(False).astype(bool)
    # S1 has any numbers at all?
    s1_has = np.zeros(int(si.max()) + 1, bool)
    s1_has[s1e.s1.unique()] = True
    # sharing: per (s1, src, num): distinct pairs + top-2 p
    g = rn.drop_duplicates(["key", "num"])
    cnt = g.groupby(["s1", "src", "num"]).key.transform("size").values
    g = g.assign(cnt=cnt)
    g = g.sort_values(["s1", "src", "num", "p"], ascending=[True, True, True, False])
    grp = g.groupby(["s1", "src", "num"])
    g["p1"] = grp.p.transform("first").values
    g["p2"] = _second(g)
    g["other_maxp"] = np.where(g.p.values >= g.p1.values, g.p2.values, g.p1.values)
    # cross-source sharing: the same unexplained number in the OTHER source's candidates of this S1.
    # True copies share a per-source perturbed base (cross-source only ~5%); a hidden/neighbour entity's own copies
    # agree across sources (~62%, real train GT) -> cross-source sharing is evidence of ANOTHER entity.
    gm = g.groupby(["s1", "src", "num"]).agg(xcnt=("key", "size"), xmaxp=("p", "max")).reset_index()
    gm["src"] = 5 - gm["src"]                                   # look up the other source (2 <-> 3)
    g = g.merge(gm, on=["s1", "src", "num"], how="left")
    g["xcnt"] = g.xcnt.fillna(0); g["xmaxp"] = g.xmaxp.fillna(0)
    un = g[~g.expl & s1_has[g.s1.values]]
    agg = un.groupby("key").agg(sh_n=("cnt", "max"), sh_maxp=("other_maxp", "max"), shx_n=("xcnt", "max"),
                                shx_maxp=("xmaxp", "max"))
    agg["sh_n"] = agg.sh_n - 1
    out = pd.DataFrame(index=pid)
    has_num = np.zeros(n, bool); has_num[rn.key.unique()] = True
    u_any = np.zeros(n, np.float32); u_any[agg.index.values] = 1.0
    u_any[~has_num | ~s1_has[si]] = np.nan
    out["u_any"] = u_any
    out["sh_n"] = np.float32(0); out.loc[agg.index, "sh_n"] = agg.sh_n.values.astype(np.float32)
    out["sh_maxp"] = np.float32(0); out.loc[agg.index, "sh_maxp"] = agg.sh_maxp.values.astype(np.float32)
    out["shx_n"] = np.float32(0); out.loc[agg.index, "shx_n"] = agg.shx_n.values.astype(np.float32)
    out["shx_maxp"] = np.float32(0); out.loc[agg.index, "shx_maxp"] = agg.shx_maxp.values.astype(np.float32)
    kept = g[g.expl].key.unique()
    s1k = np.zeros(n, np.float32); s1k[kept] = 1.0
    s1k[~has_num | ~s1_has[si]] = np.nan
    out["s1_kept"] = s1k
    return out.reset_index(drop=True)


def _second(g):
    """Second-highest p within (s1, src, num) groups for a frame sorted by group then p desc (vectorised)."""
    key = pd.factorize(pd.MultiIndex.from_frame(g[["s1", "src", "num"]]))[0]
    first = np.r_[True, key[1:] != key[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(key)), 0))
    nxt = np.minimum(start + 1, len(key) - 1)
    ok = (start + 1 < len(key)) & (key[nxt] == key)
    return np.where(ok, g.p.values[nxt], 0.0)


_TOK = None


def rare_tokens(addrs, max_df=20, min_len=3):
    """Per record: space-joined address tokens (alpha, len>=min_len) whose document frequency over ALL given
    addresses is <= max_df (typos like 'chralotte', rare street words). Computed once per country table."""
    import re
    from collections import Counter
    tok = re.compile(r"[a-z]{%d,}" % min_len)
    toks = [set(tok.findall(a)) for a in addrs]
    df = Counter(t for s in toks for t in s)
    return np.array([" ".join(t for t in s if df[t] <= max_df) for s in toks], dtype=object)


def compute_tok(pairs, s1_addr, r_rare, r_src, p=None):
    """R4b: the same within/cross-source sharing test on RARE address tokens absent from the S1 address.
    Real train GT: a true record's rare discrepancy token recurs in a same-source copy 55% of the time (cross-source
    4.8%); decoys 10%. Returns st_any, st_n, st_maxp, stx_n, stx_maxp aligned to pairs."""
    si = np.asarray(pairs["s1_i"].values); ri = np.asarray(pairs["r_i"].values)
    n = len(si); pid = np.arange(n)
    pp = np.zeros(n, np.float32) if p is None else np.asarray(p, np.float32)
    rt = pd.Series(np.asarray(r_rare, dtype=object)[ri], index=pid).str.split().explode().dropna()
    rt = rt[rt.str.len() > 0]
    g = pd.DataFrame({"key": rt.index.values, "tok": rt.values})
    g["s1"] = si[g.key.values]; g["src"] = np.asarray(r_src)[ri][g.key.values]; g["p"] = pp[g.key.values]
    us1 = np.unique(si)
    s1t = pd.Series(np.asarray(s1_addr, dtype=object)[us1], index=us1).str.split().explode().dropna()
    s1t = pd.DataFrame({"s1": s1t.index.values, "tok": s1t.values, "in_s1": True}).drop_duplicates()
    g = g.merge(s1t, on=["s1", "tok"], how="left")
    g = g[g.in_s1.isna()].drop(columns="in_s1").drop_duplicates(["key", "tok"])
    out = pd.DataFrame({"st_any": np.zeros(n, np.float32), "st_n": np.float32(0), "st_maxp": np.float32(0),
                        "stx_n": np.float32(0), "stx_maxp": np.float32(0)})
    if not len(g):
        return out
    grp = g.groupby(["s1", "src", "tok"])
    g["cnt"] = grp.key.transform("size").values
    g = g.sort_values(["s1", "src", "tok", "p"], ascending=[True, True, True, False])
    g["p1"] = g.groupby(["s1", "src", "tok"]).p.transform("first").values
    g["p2"] = _second(g.rename(columns={"tok": "num"}))
    g["other_maxp"] = np.where(g.p.values >= g.p1.values, g.p2.values, g.p1.values)
    gm = g.groupby(["s1", "src", "tok"]).agg(xcnt=("key", "size"), xmaxp=("p", "max")).reset_index()
    gm["src"] = 5 - gm["src"]
    g = g.merge(gm, on=["s1", "src", "tok"], how="left")
    g["xcnt"] = g.xcnt.fillna(0); g["xmaxp"] = g.xmaxp.fillna(0)
    agg = g.groupby("key").agg(st_n=("cnt", "max"), st_maxp=("other_maxp", "max"), stx_n=("xcnt", "max"),
                               stx_maxp=("xmaxp", "max"))
    out.loc[agg.index, "st_any"] = 1.0
    out.loc[agg.index, "st_n"] = (agg.st_n.values - 1).astype(np.float32)
    out.loc[agg.index, "st_maxp"] = agg.st_maxp.values.astype(np.float32)
    out.loc[agg.index, "stx_n"] = agg.stx_n.values.astype(np.float32)
    out.loc[agg.index, "stx_maxp"] = agg.stx_maxp.values.astype(np.float32)
    return out
```

### A.2 `stage_c.py`
```python
"""Stage C (item R4, research/10): within-source shared-perturbation + house-number features stacked on the FINAL
pair probability, fitted on real OOF and applied post-hoc to saved test parquets. No pipeline re-run.

  fit:   cd <code dir> && python stage_c.py fit   --prep PREP --out RUN_OUT --work WORKDIR [--threads 8]
  apply: cd <code dir> && python stage_c.py apply --prep PREP --out RUN_OUT --work WORKDIR [--runs RUN_OUT2 ...]

fit   rebuilds stage-B OOF (pB) exactly like run_big.train_phase (only if meta.use_b; else uses pA), computes the new
      columns [u_any, sh_n, sh_maxp, s1_kept, num_dmin], and trains stage C = LightGBM on [p, new cols] with
      GroupKFold(3) OOF. It prints OOF macro-F0.5 of the base (p) and of stage C under the same decision family
      (rule A tuned on OOF, and per-country calibrated GFM), and GATES: stage C is saved only if it wins by >= 0.0005.
      Saves WORKDIR/{stageC.txt, stageC_meta.json, calibC.pkl}. Reads RUN_OUT, never writes there.
apply reads WORKDIR/stageC*; for each RUN_OUT's test_<c>.parquet (p = the run's final p) computes the new columns on
      the test candidate lists, predicts p_C, decides with stage C's own OOF-tuned decision, and writes
      WORKDIR/run_<i>/test_<c>.parquet (+ a copy of the run's models/meta.json with use_d/rule of stage C) so
      blend_runs.py / final TSV writing can consume it. With several --runs it averages their p first (prob-avg).
Unseen country (france): stage C applies unchanged (its inputs are country-agnostic); GFM uses the mean calibrator.
Refuses to run when meta.use_e is true (stored test p would then be post-enrichment, not what stage C was fit on).
"""
import argparse, json, os, pickle, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import decide, model  # noqa: E402
from src.run_big import load, load_r, train_split, countries, stage_b_frame, load_sib  # noqa: E402
from shared_feature import compute, rare_tokens, compute_tok  # noqa: E402

NEW = ["u_any", "sh_n", "sh_maxp", "shx_n", "shx_maxp", "s1_kept", "num_dmin"]
TOK = ["st_any", "st_n", "st_maxp", "stx_n", "stx_maxp"]          # R4b rare address-token sharing (US lab +0.0005)
PRM = {"learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100, "feature_fraction": 1.0, "max_bin": 255}


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)


def num_dmin(s1_nums, r_nums, si, ri):
    A = [[int(x) for x in s.split() if len(x) <= 4] for s in s1_nums]
    B = [[int(x) for x in s.split() if len(x) <= 4] for s in r_nums]
    out = np.full(len(si), np.nan, np.float32)
    for k, (a, b) in enumerate(zip(si, ri)):
        x, y = A[a], B[b]
        if x and y:
            out[k] = min(abs(u - v) for u in x for v in y)
    return out


def design(XC):
    """Indicator design for the LR variant: logit(p) + number-evidence indicators (NaN-safe)."""
    u = XC.u_any.fillna(-1).values; sh = XC.sh_n.fillna(0).values; k = XC.s1_kept.fillna(-1).values
    dm = XC.num_dmin.values
    sx = XC.shx_n.fillna(0).values
    return np.column_stack([XC.lp.values, (u == 1) & (sh > 0) & (sx == 0), (u == 1) & (sh == 0) & (sx == 0),
                            (u == 1) & (sx > 0), (u == 1) & (XC.sh_maxp.values > 0.5), (u == 1) & (XC.shx_maxp.values > 0.5),
                            k == 0, k == 1, (u == 1) & (dm <= 10), (u == -1)]).astype(np.float32)


class LRStage:
    def __init__(self, C=1.0):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(C=C, max_iter=500)

    def fit(self, XC, y):
        self.m.fit(design(XC), y); return self

    def predict(self, XC):
        return self.m.predict_proba(design(XC))[:, 1]


def lr_oof(XC, y, groups):
    from sklearn.model_selection import GroupKFold
    out = np.zeros(len(y))
    for tr, va in GroupKFold(3).split(XC, y, groups):
        out[va] = LRStage().fit(XC.iloc[tr], y[tr]).predict(XC.iloc[va])
    return out


def feats(pairs, s1_nums, r_nums, src, p, s1_addr=None, r_rare=None):
    F = compute(pairs, s1_nums, r_nums, src, p)
    F["num_dmin"] = num_dmin(s1_nums, r_nums, pairs.s1_i.values, pairs.r_i.values)
    if r_rare is not None:
        F = pd.concat([F.reset_index(drop=True), compute_tok(pairs, s1_addr, r_rare, src, p).reset_index(drop=True)], axis=1)
    return F


def gfm_score(ev, s1c, r, p, y, country, calib=None):
    calib = calib or {}
    pc = p.copy()
    for c in np.unique(country):
        m = country == c
        if c not in calib:
            calib[c] = decide.fit_calibrator(p[m], y[m])
        pc[m] = calib[c].predict(p[m])
    return ev.score(decide.gfm_select(s1c, r, pc.astype(np.float32), excl=True)), calib


def fit(a):
    t0 = time.time()
    meta = json.load(open(f"{a.out}/models/meta.json"))
    if meta.get("use_e"):
        sys.exit("meta.use_e is true: stored test p is post-enrichment; stage C would be fit on a different p. Abort.")
    cfg = meta["cfg"]
    T = pd.read_parquet(f"{a.out}/train_features.parquet", columns=["y", "s1_i", "r_i", "s1_id", "country"])
    pA = np.load(f"{a.out}/oof_pA.npy")
    blk = json.load(open(f"{a.out}/train_blocking.json"))
    gs = json.load(open(f"{a.out}/train_sample_gold_sizes.json"))
    y = T.y.values.astype(int)
    XB, NEWF, order, RID = [], [], [], []
    for c in gs:
        m = np.flatnonzero((T.country == c).values)
        s1_all = load(a.prep, "train", "s1", c, ["entity_id", "addr_nums", "addr_norm"]).reset_index(drop=True)
        vis, smp = train_split(len(s1_all), cfg)
        s1t = s1_all.iloc[vis].reset_index(drop=True).iloc[smp].reset_index(drop=True)
        n2 = len(load(a.prep, "train", "s2", c, ["entity_id"]))
        r = load_r(a.prep, "train", c, ["entity_id", "addr_nums", "addr_norm"])
        r_rare = rare_tokens(r.addr_norm.values) if a.tok else None
        src = np.r_[np.full(n2, 2), np.full(len(r) - n2, 3)].astype(np.int8)
        code = pd.Series(np.arange(len(s1t)), index=s1t.entity_id.values)
        pairs = pd.DataFrame({"s1_i": code.loc[T.s1_id.values[m]].values, "r_i": T.r_i.values[m] - blk[c]["r_off"]})
        NEWF.append((pairs, s1t.addr_nums.values, r.addr_nums.values, src, s1t.addr_norm.values, r_rare))
        RID.append(r.entity_id.values[pairs.r_i.values])
        order.append(m)
        print(f"[{c}] loaded {len(m)} pairs, {time.time()-t0:.0f}s", flush=True)
    if meta.get("use_b"):
        from blend_runs import oof_for_run                        # shared, cached stage-B OOF rebuild
        _, p0 = oof_for_run(a.out, a.prep)
        print(f"stage-B OOF ready (cached in ~/lab/cache), {time.time()-t0:.0f}s", flush=True)
    else:
        p0 = pA
    cols = NEW + (TOK if a.tok else [])
    F = pd.DataFrame(index=np.arange(len(T)), columns=cols, dtype=np.float32)
    for m, (pairs, sn, rn, src, sa, rr) in zip(order, NEWF):
        F.iloc[m] = feats(pairs, sn, rn, src, p0[m], sa, rr)[cols].values
    XC = pd.concat([pd.DataFrame({"lp": lg(p0)}), F.astype(np.float32)], axis=1)
    extra_cols = []
    if a.extra_train:                                  # e.g. F8 cross-encoder OOF logits (ce_oof.parquet)
        rid = np.empty(len(T), object)
        for m, rr in zip(order, RID):
            rid[m] = rr
        E = pd.read_parquet(a.extra_train)
        extra_cols = [c for c in E.columns if c not in ("s1_id", "r_id")]
        key = pd.DataFrame({"s1_id": T.s1_id.values, "r_id": rid})
        XE = key.merge(E, on=["s1_id", "r_id"], how="left")[extra_cols].astype(np.float32)
        print(f"extra columns {extra_cols}: coverage {XE.notna().mean().round(4).to_dict()}", flush=True)
        XC = pd.concat([XC, XE.reset_index(drop=True)], axis=1)
    if a.kind == "lr":
        pC, itC = lr_oof(XC, y, T.s1_i.values), 0
    else:
        pC, itC = model.oof_predict(XC, y, T.s1_i.values, 3, params=PRM)
    ids = [e for c in gs for e in gs[c]]; cmap = {e: i for i, e in enumerate(ids)}
    ng = np.array([gs[c][e] for c in gs for e in gs[c]])
    s1c = np.array([cmap[e] for e in T.s1_id.values])
    ev = decide.Evaluator(s1c, T.y.values, ng, len(ids))
    country = T.country.values
    res = {"base_rule": decide.tune_rule(ev, s1c, T.r_i.values, p0), "C_rule": decide.tune_rule(ev, s1c, T.r_i.values, pC)}
    res["base_gfm"], _ = gfm_score(ev, s1c, T.r_i.values, p0, y, country)
    res["C_gfm"], calibC = gfm_score(ev, s1c, T.r_i.values, pC, y, country)
    base_best = max(res["base_rule"][0], res["base_gfm"]); c_best = max(res["C_rule"][0], res["C_gfm"])
    res["delta"] = c_best - base_best
    from decision_lab_real import honest_2fold               # rule vs GFM with no in-sample tuning
    res["C_honest_2fold"] = honest_2fold(s1c, T.r_i.values, T.y.values.astype(bool), pC, ng, country)
    res["base_honest_2fold"] = honest_2fold(s1c, T.r_i.values, T.y.values.astype(bool), p0, ng, country)
    res["use_gfm"] = res["C_honest_2fold"]["gfm"] > res["C_honest_2fold"]["rule"] + 0.0005

    def links(p_, rl):
        sel = decide.rule_select(s1c, p_, rl["t1"], rl["t2"], rl["delta"],
                                 decide.exclusive_mask(T.r_i.values, p_) if rl["excl"] else None)
        return float(sel.sum()) / len(ids)
    res["oof_links_base"] = links(p0, res["base_rule"][1]); res["oof_links_C"] = links(pC, res["C_rule"][1])
    res["oof_links_rel_change"] = res["oof_links_C"] / res["oof_links_base"] - 1
    print(f"DETECTOR (OOF side): links/S1 base {res['oof_links_base']:.4f} -> C {res['oof_links_C']:.4f} "
          f"({100*res['oof_links_rel_change']:+.2f}%). Compare with the test side printed by `apply`.")
    res["delta_honest"] = max(res["C_honest_2fold"].values()) - max(res["base_honest_2fold"].values())
    print(json.dumps(res, default=str, indent=1), flush=True)
    os.makedirs(a.work, exist_ok=True)
    json.dump(res, open(f"{a.work}/stageC_oof.json", "w"), default=str, indent=1)
    print(f"honest (2-fold) delta {res['delta_honest']:+.5f} | in-sample delta {res['delta']:+.5f}")
    if min(res["delta"], res["delta_honest"]) < a.min_gain:
        print(f"GATE: stage C delta {res['delta']:+.5f} < +0.0005 -> NOT saved (do not deploy)")
        return
    if a.kind == "lr":
        pickle.dump(LRStage().fit(XC, y), open(f"{a.work}/stageC_lr.pkl", "wb"))
    else:
        model.fit_full(XC, y, itC, params=PRM)[0].save_model(f"{a.work}/stageC.txt")
    pickle.dump(calibC, open(f"{a.work}/calibC.pkl", "wb"))
    json.dump({"kind": a.kind, "tok": bool(a.tok), "cols": cols, "extra_cols": extra_cols, "rule": res["C_rule"][1], "use_gfm": bool(res["use_gfm"]), "itC": int(itC), "seen": sorted(set(country)),
               "unseen_shift": cfg.get("unseen_shift", 0.05)}, open(f"{a.work}/stageC_meta.json", "w"), indent=1)
    print(f"GATE passed ({res['delta']:+.5f}); saved stage C to {a.work}, {time.time()-t0:.0f}s")


def apply(a):
    import lightgbm as lgb
    cm = json.load(open(f"{a.work}/stageC_meta.json"))
    mC = pickle.load(open(f"{a.work}/stageC_lr.pkl", "rb")) if cm.get("kind") == "lr" else \
        lgb.Booster(model_file=f"{a.work}/stageC.txt")
    calibC = pickle.load(open(f"{a.work}/calibC.pkl", "rb"))
    runs = [a.out] + (a.runs or [])
    od = f"{a.work}/run_C"
    os.makedirs(f"{od}/models", exist_ok=True)
    det = {}
    for c in countries(a.prep, "test"):
        ds = [pd.read_parquet(f"{r}/test_{c}.parquet", columns=["s1_id", "r_id", "p"]) for r in runs]
        d = ds[0]
        for i, x in enumerate(ds[1:], 1):
            d = d.merge(x.rename(columns={"p": f"p{i}"}), on=["s1_id", "r_id"], how="outer")
        d["p"] = np.nanmean(d[["p"] + [f"p{i}" for i in range(1, len(ds))]].values, 1)
        d = d[["s1_id", "r_id", "p"]]
        s1 = load(a.prep, "test", "s1", c, ["entity_id", "addr_nums", "addr_norm"]).reset_index(drop=True)
        n2 = len(load(a.prep, "test", "s2", c, ["entity_id"]))
        r = load_r(a.prep, "test", c, ["entity_id", "addr_nums", "addr_norm"])
        r_rare = rare_tokens(r.addr_norm.values) if cm.get("tok") else None
        src = np.r_[np.full(n2, 2), np.full(len(r) - n2, 3)].astype(np.int8)
        si = pd.Series(np.arange(len(s1)), index=s1.entity_id.values).loc[d.s1_id.values].values
        ri = pd.Series(np.arange(len(r)), index=r.entity_id.values).loc[d.r_id.values].values
        pairs = pd.DataFrame({"s1_i": si, "r_i": ri})
        F = feats(pairs, s1.addr_nums.values, r.addr_nums.values, src, d.p.values, s1.addr_norm.values, r_rare)
        XC = pd.concat([pd.DataFrame({"lp": lg(d.p.values)}), F[cm.get("cols", NEW)].astype(np.float32).reset_index(drop=True)], axis=1)
        if cm.get("extra_cols"):
            if not a.extra_test:
                sys.exit("stage C was fit with extra columns: pass --extra-test")
            XE = d[["s1_id", "r_id"]].merge(pd.read_parquet(a.extra_test), on=["s1_id", "r_id"], how="left")[cm["extra_cols"]]
            XC = pd.concat([XC, XE.astype(np.float32).reset_index(drop=True)], axis=1)
        pc = mC.predict(XC).astype(np.float32)
        order = np.argsort(si, kind="stable")
        si, ri, pc, d = si[order], ri[order], pc[order], d.iloc[order].reset_index(drop=True)
        if cm["use_gfm"]:
            q = calibC[c].predict(pc) if c in calibC else np.clip(
                np.mean([k.predict(pc) for k in calibC.values()], 0) - cm["unseen_shift"] * 0.5, 0, 1)
            sel = decide.gfm_select(si, ri, q.astype(np.float32), excl=True)
        else:
            rl = cm["rule"]; sh = 0.0 if c in cm["seen"] else cm["unseen_shift"]
            sel = decide.rule_select(si, pc, rl["t1"], rl["t2"], rl["delta"],
                                     decide.exclusive_mask(ri, pc) if rl["excl"] else None,
                                     t_shift=np.full(len(pc), sh, np.float32))
        out = pd.DataFrame({"s1_id": d.s1_id.values, "r_id": d.r_id.values, "p": pc, "sel": sel})
        out.to_parquet(f"{od}/test_{c}.parquet", index=False)
        nl = np.bincount(si[sel], minlength=len(s1))
        base_sel = pd.read_parquet(f"{runs[0]}/test_{c}.parquet", columns=["sel"]).sel.values
        tl_base, tl_c = base_sel.sum() / len(s1), sel.sum() / len(s1)
        det.setdefault("base", 0.0); det.setdefault("C", 0.0); det["base"] += base_sel.sum(); det["C"] += sel.sum()
        print(f"[{c}] DETECTOR test links/S1: base (run[0] stored sel) {tl_base:.4f} -> C {tl_c:.4f} ({100*(tl_c/tl_base-1):+.2f}%)")
        print(f"[{c}] stage C applied: pairs {len(out)} links {int(sel.sum())} mean_links {nl.mean():.3f} "
              f"frac_empty {(nl == 0).mean():.4f}", flush=True)
    meta = json.load(open(f"{a.out}/models/meta.json"))
    meta["use_d"] = bool(cm["use_gfm"]); meta["rule"] = cm["rule"]
    json.dump(meta, open(f"{od}/models/meta.json", "w"), indent=1)
    pickle.dump(calibC, open(f"{od}/models/calib.pkl", "wb"))
    oof = json.load(open(f"{a.work}/stageC_oof.json"))
    t_rel = det["C"] / det["base"] - 1; o_rel = oof.get("oof_links_rel_change", 0.0)
    flag = (t_rel > 0.01) and (t_rel > 2 * max(o_rel, 0.0) + 0.005)
    print(f"DETECTOR (research/16 §2.A-4, pkd): test links/S1 change {100*t_rel:+.2f}% vs OOF change {100*o_rel:+.2f}% -> "
          f"{'RED FLAG: group-feature vouching on test; ship only as a parity-half probe' if flag else 'ok'}")
    json.dump({"test_rel": t_rel, "oof_rel": o_rel, "red_flag": bool(flag)}, open(f"{od}/detector.json", "w"))
    print("wrote", od, "(feed to blend_runs.py --runs", od, "--decision auto)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "apply"])
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--kind", default="gbdt", choices=["gbdt", "lr"])
    ap.add_argument("--min-gain", type=float, default=0.0005)
    ap.add_argument("--tok", type=int, default=1, help="1 = add R4b rare address-token sharing columns")
    ap.add_argument("--extra-train", help="parquet (s1_id, r_id, cols...) of OOF extra features, e.g. ce_oof.parquet")
    ap.add_argument("--extra-test", help="parquet (s1_id, r_id, cols...) of test extra features, e.g. ce_test.parquet")
    a = ap.parse_args()
    fit(a) if a.mode == "fit" else apply(a)
```

### A.3 `blend_runs.py`
```python
"""Blend several v10 run directories at the PROBABILITY level and decide once (tomorrow's ensemble tool).

    python blend_runs.py --runs RUN_A RUN_B [RUN_C ...] --s1 test_source1.tsv --out OUT_DIR [--gt hidden.tsv]
                         [--decision auto|gfm|rule] [--fr-shift 0.0] [--how prob|logit]

- Reads each run's test_<country>.parquet (s1_id, r_id, p, sel). Pairs are aligned on (s1_id, r_id); a pair present
  in only some runs (anchor-pass extras) is averaged over the runs that scored it.
- Decision = the FIRST run's decision config: its models/calib.pkl + calibrated GFM when meta.use_d is true (unseen
  country: mean of the seen calibrators minus unseen_shift/2, exactly as run_big.test_phase), else its meta.rule.
- Writes OUT_DIR/matching_results.tsv (row order = test_source1.tsv) and OUT_DIR/candidate_pairs.tsv (union of the
  scored pairs = every pair the blended matcher scored). With --gt, prints the macro F0.5 (synth only).
Synth evidence (6 seeds, 26 Sep): prob-avg + GFM k=2 +0.0021, k=4 +0.0009, k=6 +0.0003 vs mean single seed;
TSV-level extras/strict agreement -0.0003/-0.0004 (3/3 pairs negative) -> never merge at the TSV level.
"""
import argparse, csv, json, os, pickle, sys
import numpy as np, pandas as pd

sys.path.insert(0, "/Users/mac/amazon-ml-2026/code/ber_v5")
sys.path.insert(0, os.getcwd())                     # on the boxes: run from the pipeline code dir (~/er)
from src import decide  # noqa: E402


def load(run):
    out = []
    for f in sorted(os.listdir(run)):
        if f.startswith("test_") and f.endswith(".parquet") and "_sib" not in f and "_pairs" not in f:
            c = f[5:-8]
            d = pd.read_parquet(os.path.join(run, f), columns=["s1_id", "r_id", "p"])
            d["c"] = c
            out.append(d)
    return pd.concat(out, ignore_index=True)


def _blend(P, how):
    if how == "prob":
        return np.nanmean(P, 1)
    q = np.clip(P, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-np.nanmean(np.log(q / (1 - q)), 1)))


def oof_for_run(run, prep):
    """OOF p consistent with the run's stored test p: stage-B OOF rebuilt exactly as run_big.train_phase (use_b),
    else stage-A OOF. Returns (T, p_oof)."""
    sys.path.insert(0, os.getcwd())
    from src import model
    from src.run_big import stage_b_frame, load_sib
    meta = json.load(open(f"{run}/models/meta.json"))
    if meta.get("use_e"):
        sys.exit(f"{run}: use_e=True -> stored test p is post-enrichment; OOF refit not supported")
    T = pd.read_parquet(f"{run}/train_features.parquet", columns=["y", "s1_i", "r_i", "s1_id", "country"])
    pA = np.load(f"{run}/oof_pA.npy")
    if not meta.get("use_b"):
        return T, pA
    cache = os.path.join(os.path.expanduser("~/lab/cache"), os.path.abspath(run).strip("/").replace("/", "_") + "_pB_oof.npy")
    if os.path.exists(cache):                                   # stage-B OOF rebuild is ~10 min at real scale: reuse
        return T, np.load(cache)
    blk = json.load(open(f"{run}/train_blocking.json"))
    parts, order = [], []
    for c in sorted(set(T.country)):
        m = np.flatnonzero((T.country == c).values)
        sib = load_sib(f"{run}/train_{c}_sib.npz")
        parts.append(stage_b_frame(T.s1_i.values[m], T.r_i.values[m] - blk[c]["r_off"], pA[m], sib, blk[c]["n_r"]))
        order.append(m)
    XB = pd.concat(parts, ignore_index=True).iloc[np.argsort(np.concatenate(order))].reset_index(drop=True)
    pB, _ = model.oof_predict(XB, T.y.values.astype(int), T.s1_i.values, meta["cfg"].get("n_splits", 3),
                              params={"learning_rate": 0.08, "num_leaves": 255, "min_data_in_leaf": 400})
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, pB)
    return T, pB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--s1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt")
    ap.add_argument("--decision", default="auto", choices=["auto", "gfm", "rule"])
    ap.add_argument("--fr-shift", type=float, default=0.0, help="extra threshold shift (rule) / calib offset (gfm) for france")
    ap.add_argument("--shift", type=float, default=0.0, help="P6: global threshold shift (rule) / calib offset (gfm), all countries")
    ap.add_argument("--how", default="prob", choices=["prob", "logit"])
    ap.add_argument("--refit", action="store_true",
                    help="blend each run's OOF p the same way and refit per-country isotonic + rule on the BLENDED OOF "
                         "(calibrate-after-blend, Wu & Gales 2021). Needs --prep and cwd = pipeline code dir.")
    ap.add_argument("--prep")
    a = ap.parse_args()

    frames = [load(r).rename(columns={"p": f"p{i}"}) for i, r in enumerate(a.runs)]
    m = frames[0]
    for f in frames[1:]:
        m = m.merge(f, on=["s1_id", "r_id", "c"], how="outer")
    P = m[[f"p{i}" for i in range(len(a.runs))]].values.astype(float)
    m["p"] = _blend(P, a.how).astype(np.float32)
    print(f"runs {len(a.runs)} | pairs {len(m)} | pairs scored by all runs {np.isfinite(P).all(1).mean():.4f} | "
          f"corr(p0,p1) {np.corrcoef(np.nan_to_num(P[:,0]), np.nan_to_num(P[:,1]))[0,1] if P.shape[1]>1 else 1:.4f}")

    meta = json.load(open(os.path.join(a.runs[0], "models", "meta.json")))
    cfg = meta.get("cfg", {})
    use_gfm = meta.get("use_d") if a.decision == "auto" else a.decision == "gfm"
    calib = pickle.load(open(os.path.join(a.runs[0], "models", "calib.pkl"), "rb")) if (use_gfm and not a.refit) else None
    rule = meta["rule"]
    if a.refit:
        Ts, Os = zip(*[oof_for_run(r, a.prep) for r in a.runs])
        for T_ in Ts[1:]:
            assert len(T_) == len(Ts[0]) and (T_.y.values == Ts[0].y.values).all(), "runs have different train pairs"
        T = Ts[0]
        po = _blend(np.column_stack(Os), a.how)
        gs = json.load(open(os.path.join(a.runs[0], "train_sample_gold_sizes.json")))
        ids = [e for c in gs for e in gs[c]]; cmap = {e: i for i, e in enumerate(ids)}
        ng = np.array([gs[c][e] for c in gs for e in gs[c]])
        s1c = np.array([cmap[e] for e in T.s1_id.values])
        ev = decide.Evaluator(s1c, T.y.values, ng, len(ids))
        sc_rule, rule = decide.tune_rule(ev, s1c, T.r_i.values, po)
        calib, pc_oof = {}, po.copy()
        for c in sorted(set(T.country)):
            mm = (T.country == c).values
            calib[c] = decide.fit_calibrator(po[mm], T.y.values[mm])
            pc_oof[mm] = calib[c].predict(po[mm])
        sc_gfm = ev.score(decide.gfm_select(s1c, T.r_i.values, pc_oof.astype(np.float32), excl=True))
        singles = [decide.tune_rule(ev, s1c, T.r_i.values, o)[0] for o in Os]
        print(f"blended OOF: rule {sc_rule:.5f} {rule} | calibrated GFM {sc_gfm:.5f} | single-run OOF (rule) {np.round(singles, 5)}")
        if a.decision == "auto":
            use_gfm = sc_gfm > sc_rule + 1e-4
    unseen = float(cfg.get("unseen_shift", 0.05))
    sel = np.zeros(len(m), bool)
    for c, g in m.groupby("c").groups.items():
        idx = np.asarray(g)
        d = m.iloc[idx].sort_values("s1_id", kind="stable")
        idx = d.index.values
        s1c, _ = pd.factorize(d.s1_id.values)
        rc, _ = pd.factorize(d.r_id.values)
        pp = d.p.values
        if use_gfm:
            if c in calib:
                pc = np.clip(calib[c].predict(pp) - a.shift, 0, 1)
            else:
                pc = np.clip(np.mean([k.predict(pp) for k in calib.values()], 0) - unseen * 0.5 - (a.fr_shift if c == "france" else 0) - a.shift, 0, 1)
            s = decide.gfm_select(s1c, rc, pc.astype(np.float32), excl=True)
        else:
            sh = (0.0 if c in meta["seen"] else unseen) + (a.fr_shift if c == "france" else 0.0) + a.shift
            ex = decide.exclusive_mask(rc, pp) if rule["excl"] else None
            s = decide.rule_select(s1c, pp, rule["t1"], rule["t2"], rule["delta"], ex, t_shift=np.full(len(pp), sh, np.float32))
        sel[idx] = s
    m["sel"] = sel
    print(f"decision: {'calibrated GFM' if use_gfm else 'rule ' + json.dumps(rule)} | links {int(sel.sum())}")

    order = []
    with open(a.s1, encoding="utf-8") as fh:
        r = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE); h = next(r); ei = h.index("entity_id")
        order = [row[ei] for row in r]
    m = m.sort_values(["s1_id", "p"], ascending=[True, False], kind="stable")
    match = m[m.sel].groupby("s1_id").r_id.apply(list).to_dict()
    cand = m.groupby("s1_id").r_id.apply(list).to_dict()
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/matching_results.tsv", "w", encoding="utf-8", newline="") as fm, \
            open(f"{a.out}/candidate_pairs.tsv", "w", encoding="utf-8", newline="") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for e in order:
            fm.write(f"{e}\t{','.join(match.get(e, []))}\n")
            fc.write(f"{e}\t{','.join(cand.get(e, []))}\n")
    print("wrote", a.out)
    if a.gt:
        sys.path.insert(0, "/Users/mac/amazon-ml-2026/code/business_entity_resolution")
        from src.metrics import macro_fbeta, parse_ids
        with open(a.gt) as fh:
            r = csv.reader(fh, delimiter="\t"); next(r); gt = {e: parse_ids(v) for e, v in r}
        pred = {e: set(match.get(e, [])) for e in order}
        print(f"macro F0.5 = {macro_fbeta(pred, gt):.6f}")


if __name__ == "__main__":
    main()
```

### A.4 `decision_lab_real.py`
```python
"""Real-OOF decision lab (items P2, A4 size-prior GFM, caps, France-by-LOCO) — run after a v10 train phase.

    cd <code dir> && python decision_lab_real.py --out RUN_OUT [--prep PREP]   (reads only; ~5-15 min)

Compares on the run's OOF (stage-A OOF, or rebuilt stage-B OOF when meta.use_b and --prep is given):
  rule A (tuned on OOF) | calibrated GFM (lambda=0, = shipped) | GFM with Poisson(lambda_c) extra misses
  (lambda_c = measured unretrieved true matches per S1 of country c: the fix for best_set_fbeta's empty bias)
  | each + GT caps (<=5 S2, <=6 S3 per S1).
France proxy: LOCO p (loco_pA) decided with (a) OOF-tuned rule +{0,.05,.1}, (b) GFM with the OTHER country's
calibrator (what France gets), (c) (b) + lambda. The winner family on LOCO is the France decision (journal A1).
Prints a table and writes /tmp/decision_lab_real.json. Every number is OOF-in-sample for isotonic (millions of pts).
"""
import argparse, json, math, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import decide  # noqa: E402


def _pb(ps):
    f = np.zeros(len(ps) + 1); f[0] = 1.0
    for q in ps:
        f[1:] = f[1:] * (1 - q) + f[:-1] * q; f[0] *= (1 - q)
    return f


def best_set_miss(p, lam, beta=0.5, max_m=12, kmax_miss=8):
    """best_set_fbeta with n_gold = (in-candidate positives) + M, M ~ Poisson(lam) independent (unretrieved matches).
    lam=0 reproduces metrics.best_set_fbeta exactly."""
    p = np.asarray(p, float); order = np.argsort(-p)[:max_m]; ps = p[order]; m = len(ps); b2 = beta * beta
    pm = np.array([math.exp(-lam) * lam ** j / math.factorial(j) for j in range(kmax_miss + 1)]) if lam > 0 else np.array([1.0])
    best_val, best_k = float(np.prod(1 - ps)) * pm[0], 0
    for k in range(1, m + 1):
        tp, fn = _pb(ps[:k]), np.convolve(_pb(ps[k:]), pm)
        t = np.arange(k + 1)[:, None]; f = np.arange(len(fn))[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            val = np.where(t > 0, (1 + b2) * t / (b2 * (t + f) + k), 0.0)
        val = float((val * tp[:, None] * fn[None, :]).sum())
        if val > best_val:
            best_val, best_k = val, k
    return order[:best_k]


def gfm_miss(s1c, r, pc, lam_rows, excl=True):
    q = np.where(decide.exclusive_mask(r, pc), pc, 0.0) if excl else pc
    sel = np.zeros(len(q), bool)
    order = np.argsort(s1c, kind="stable")
    b = np.flatnonzero(np.r_[True, s1c[order][1:] != s1c[order][:-1], True])
    for lo, hi in zip(b[:-1], b[1:]):
        idx = order[lo:hi]
        sel[idx[best_set_miss(q[idx], float(lam_rows[idx[0]]))]] = True
    return sel


def caps(sel, s1c, is_s3, p, c2=5, c3=6):
    sel = sel.copy()
    for flag, cap in ((False, c2), (True, c3)):
        m = np.flatnonzero(sel & (is_s3 == flag))
        if not len(m):
            continue
        d = pd.DataFrame({"i": m, "s": s1c[m], "p": p[m]}).sort_values(["s", "p"], ascending=[True, False])
        rk = d.groupby("s").cumcount().values
        sel[d.i.values[rk >= cap]] = False
    return sel


def honest_2fold(s1c, r, y, p, ng, country=None):
    """Rule vs calibrated GFM with NO in-sample tuning: rule/calibrator fitted on even S1 codes, scored on odd, and
    vice versa (real US mini-lab 26 Sep: rule 0.96063 vs GFM 0.95971 -> rule wins by ~0.0006-0.0009 honestly)."""
    n1 = len(ng); half = (np.arange(n1) % 2 == 0); out = {"rule": [], "gfm": []}
    for fh in (True, False):
        fr = half[s1c] == fh; m = ~fr; nsc = (half != fh).sum()
        evf = decide.Evaluator(s1c[fr], y[fr], np.where(half == fh, ng, 0), n1)
        evs = decide.Evaluator(s1c[m], y[m], np.where(half != fh, ng, 0), n1)
        _, rl = decide.tune_rule(evf, s1c[fr], r[fr], p[fr])
        sel = decide.rule_select(s1c[m], p[m], rl["t1"], rl["t2"], rl["delta"],
                                 decide.exclusive_mask(r[m], p[m]) if rl["excl"] else None)
        out["rule"].append((evs.score(sel) * n1 - (n1 - nsc)) / nsc)
        pc = p[m].copy()
        for c in (np.unique(country) if country is not None else [None]):
            cm = np.ones(len(y), bool) if c is None else (country == c)
            cal = decide.fit_calibrator(p[fr & cm], y[fr & cm])
            pc[(cm[m])] = cal.predict(p[m][cm[m]])
        out["gfm"].append((evs.score(decide.gfm_select(s1c[m], r[m], pc.astype(np.float32), excl=True)) * n1 - (n1 - nsc)) / nsc)
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--prep")
    a = ap.parse_args()
    meta = json.load(open(f"{a.out}/models/meta.json"))
    T = pd.read_parquet(f"{a.out}/train_features.parquet", columns=["y", "s1_i", "r_i", "s1_id", "country", "is_s3"])
    if meta.get("use_b") and a.prep:
        from blend_runs import oof_for_run
        _, p = oof_for_run(a.out, a.prep); tag = "stage-B OOF (rebuilt)"
    else:
        p = np.load(f"{a.out}/oof_pA.npy"); tag = "stage-A OOF"
    loco = np.load(f"{a.out}/loco_pA.npy")
    gs = json.load(open(f"{a.out}/train_sample_gold_sizes.json"))
    ids = [e for c in gs for e in gs[c]]; cmap = {e: i for i, e in enumerate(ids)}
    ng = np.array([gs[c][e] for c in gs for e in gs[c]])
    s1c = np.array([cmap[e] for e in T.s1_id.values]); r = T.r_i.values; y = T.y.values; ctry = T.country.values
    ev = decide.Evaluator(s1c, y, ng, len(ids)); is3 = T.is_s3.values > 0.5
    lam = {}
    for c in gs:
        m = ctry == c
        lam[c] = float((sum(gs[c].values()) - y[m].sum()) / len(gs[c]))
    lam_rows = np.array([lam[c] for c in ctry])
    calib = {c: decide.fit_calibrator(p[ctry == c], y[ctry == c]) for c in gs}
    pc = p.copy()
    for c in gs:
        pc[ctry == c] = calib[c].predict(p[ctry == c])
    res = {"source": tag, "lambda": lam}
    ra = decide.tune_rule(ev, s1c, r, p); res["rule_A"] = ra
    selA = decide.rule_select(s1c, p, ra[1]["t1"], ra[1]["t2"], ra[1]["delta"], decide.exclusive_mask(r, p) if ra[1]["excl"] else None)
    sel0 = decide.gfm_select(s1c, r, pc.astype(np.float32), excl=True)
    selL = gfm_miss(s1c, r, pc, lam_rows)
    for nm, sel in (("rule_A", selA), ("gfm_l0", sel0), ("gfm_lambda", selL)):
        res[nm + "_score"] = ev.score(sel); res[nm + "_caps"] = ev.score(caps(sel, s1c, is3, p))
    fe = {nm: ev.score(sel, per_entity=True) for nm, sel in (("rule_A", selA), ("gfm_l0", sel0), ("gfm_lambda", selL))}
    res["honest_2fold"] = honest_2fold(s1c, r, y.astype(bool), p, ng, ctry)
    res["by_country"] = {nm: {c: round(float(v[[cmap[e] for e in gs[c]]].mean()), 5) for c in gs} for nm, v in fe.items()}
    # France proxy on LOCO p
    lc = {}
    for sh in (0.0, 0.05, 0.1):
        lc[f"rule_shift{sh}"] = ev.score(decide.rule_select(s1c, loco, ra[1]["t1"] + sh, ra[1]["t2"] + sh, ra[1]["delta"],
                                                           decide.exclusive_mask(r, loco) if ra[1]["excl"] else None))
    pl = loco.copy()
    for c in gs:                                     # each held-out country gets the OTHER country's calibrator
        other = [k for k in gs if k != c]
        pl[ctry == c] = np.mean([calib[k].predict(loco[ctry == c]) for k in other], 0)
    lc["gfm_other_calib"] = ev.score(decide.gfm_select(s1c, r, pl.astype(np.float32), excl=True))
    lc["gfm_other_calib_minus.025"] = ev.score(decide.gfm_select(s1c, r, np.clip(pl - 0.025, 0, 1).astype(np.float32), excl=True))
    lc["gfm_other_calib_lambda"] = ev.score(gfm_miss(s1c, r, pl, lam_rows))
    lc["rule_tuned_on_loco(oracle)"] = decide.tune_rule(ev, s1c, r, loco)[0]
    res["loco_france_proxy"] = lc
    print(json.dumps(res, indent=1, default=str))
    json.dump(res, open("/tmp/decision_lab_real.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
```

### A.5 `learning_curve.py`
```python
"""Stage-A learning curve on a finished run's train_features.parquet (gates R3: 'train on more S1').

    python learning_curve.py RUN_DIR [--fracs 0.25,0.5,1.0] [--threads 8]

For each GroupKFold(3) fold, the matcher is trained on a random frac of the TRAINING S1s (same validation fold for
every frac), with the run's own LightGBM params and its itA round count (no early stopping, so sizes are comparable).
Prints OOF macro-F0.5 (rule tuned per size on its own OOF, plus a fixed rule) per frac and the per-doubling slope.
Read: slope per doubling s -> training on 2x more S1 is worth ~s (upper bound on the macro scale, stage A only).
Cost on the real box: ~3 fracs x 3 folds LightGBM fits on <= 2/3 of the pairs.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/ubuntu/er") if os.path.isdir("/home/ubuntu/er/src") else None
sys.path.insert(0, "/Users/mac/amazon-ml-2026/code/ber_v5")
sys.path.insert(0, os.getcwd())
from src import decide, model  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--fracs", default="0.25,0.5,1.0")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    meta = json.load(open(f"{a.run}/models/meta.json"))
    feat = meta["feat_cols"]; cfg = meta["cfg"]; itA = int(meta.get("itA", 300))
    T = pd.read_parquet(f"{a.run}/train_features.parquet", columns=feat + ["y", "s1_i", "r_i", "s1_id", "country"])
    gs = json.load(open(f"{a.run}/train_sample_gold_sizes.json"))
    ids = [e for c in gs for e in gs[c]]; code = {e: i for i, e in enumerate(ids)}
    ng = np.array([gs[c][e] for c in gs for e in gs[c]])
    s1c = np.array([code[e] for e in T.s1_id.values])
    ev = decide.Evaluator(s1c, T.y.values, ng, len(ids))
    y = T.y.values.astype(int); grp = T.s1_i.values
    params = dict(cfg.get("lgb", {}), num_threads=a.threads)
    fracs = [float(x) for x in a.fracs.split(",")]
    rng = np.random.RandomState(7)
    oofs = {f: np.zeros(len(T)) for f in fracs}
    for k, (tr, va) in enumerate(GroupKFold(n_splits=3).split(T, y, grp)):
        u = np.unique(grp[tr]); rng.shuffle(u)
        for f in fracs:
            keep = np.isin(grp[tr], u[: max(1, int(len(u) * f))])
            m = model._train(T[feat].iloc[tr[keep]], y[tr[keep]], rounds=itA, params=params)
            oofs[f][va] = m.predict(T[feat].iloc[va])
            print(f"fold {k} frac {f}: trained on {keep.sum()} pairs", flush=True)
    res = {}
    for f in fracs:
        tuned = decide.tune_rule(ev, s1c, T.r_i.values, oofs[f])
        fixed = ev.score(decide.rule_select(s1c, oofs[f], 0.5, 0.75, 1.0, decide.exclusive_mask(T.r_i.values, oofs[f])))
        res[f] = (tuned[0], fixed)
        print(f"frac {f}: OOF tuned {tuned[0]:.5f} {tuned[1]} | fixed(.5/.75/excl) {fixed:.5f}", flush=True)
    fs = sorted(res)
    for lo, hi in zip(fs[:-1], fs[1:]):
        d = np.log2(hi / lo)
        print(f"slope per doubling {lo}->{hi}: tuned {(res[hi][0]-res[lo][0])/d:+.5f} fixed {(res[hi][1]-res[lo][1])/d:+.5f}")
    json.dump({str(k): v for k, v in res.items()}, open(os.path.join("/tmp", "learning_curve.json"), "w"))


if __name__ == "__main__":
    main()
```

### A.6 `bin_recall.py`
```python
"""B-IN measurement on REAL India train data: recall of Ayan's reverse combo view (record -> top-K S1 by
0.5*cos(char_wb3 name) + 0.5*cos(word address), sklearn TF-IDF max_df .05) in the hide_frac=0.19 world, for a sample
of owned records. Uses ber_v5 normalize (transliteration + lexicon) so names match what the pipeline sees."""
import sys, time, json, numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, "/Users/mac/amazon-ml-2026/code/ber_v5")
from src.normalize import normalize_frame, renormalize_names
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

R = "/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad/real"
C = sys.argv[1] if len(sys.argv) > 1 else "India"
NQ = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
t0 = time.time()
LEX = json.load(open("/Users/mac/amazon-ml-2026/code/ber_v5/src/lexicon.json"))
rd = lambda f: pd.read_csv(f"{R}/{f}", sep="\t", dtype=str, keep_default_na=False, quoting=3)
s1 = rd("train_source1.tsv"); s1 = s1[s1.country == C].reset_index(drop=True)
rs = np.random.RandomState(0); vis = rs.rand(len(s1)) >= 0.19            # run_big.train_split, seed 0
s1v = s1[vis].reset_index(drop=True); del s1
vis_ids = set(s1v.entity_id)
# sample query pairs straight from GT (memory-light): owned records of VISIBLE S1s
pairs = []
for ch in pd.read_csv(f"{R}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, chunksize=500000):
    ch = ch[ch.source1_entity_id.isin(vis_ids) & (ch.matched_entity_ids != "")]
    for s, v in zip(ch.source1_entity_id.values, ch.matched_entity_ids.values):
        for r in v.split(","):
            pairs.append((r, s))
rng = np.random.RandomState(1)
sel = rng.choice(len(pairs), NQ, replace=False)
own = {pairs[i][0]: pairs[i][1] for i in sel}
del pairs
qs, fit = [], []
for f in ("train_source2.tsv", "train_source3.tsv"):
    for ch in pd.read_csv(f"{R}/{f}", sep="\t", dtype=str, keep_default_na=False, quoting=3, chunksize=500000):
        ch = ch[ch.country == C]
        qs.append(ch[ch.entity_id.isin(own)])
        fit.append(ch.sample(frac=0.15, random_state=2))
q = pd.concat(qs, ignore_index=True); q["owner"] = q.entity_id.map(own)
fitpool = pd.concat(fit, ignore_index=True)
print(f"[{C}] visible S1 {len(s1v)} | queries {len(q)} | {time.time()-t0:.0f}s", flush=True)


def norm(df):
    df = normalize_frame(df.copy())
    renormalize_names(df, LEX)
    return df


def normp(df, chunk=100000):
    return pd.concat([norm(df.iloc[i:i + chunk]) for i in range(0, len(df), chunk)], ignore_index=True)


S = normp(s1v); Q = normp(q); Fp = normp(fitpool)
print(f"normalized {time.time()-t0:.0f}s", flush=True)
vn = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=0.05, sublinear_tf=True, dtype=np.float32)
va = TfidfVectorizer(analyzer="word", token_pattern=r"[a-z0-9]+", min_df=2, max_df=0.05, sublinear_tf=True, dtype=np.float32)
vn.fit(pd.concat([S.name_core, Fp.name_core])); va.fit(pd.concat([S.addr_norm, Fp.addr_norm]))
w = np.float32(np.sqrt(0.5))
combo = lambda d: sp.hstack([vn.transform(d.name_core) * w, va.transform(d.addr_norm) * w]).tocsr()
Sx, Qx = combo(S), combo(Q)
print(f"vectorized {time.time()-t0:.0f}s nnz S {Sx.nnz}", flush=True)
St = Sx.T.tocsr()
K = 40
hits = np.full(len(Q), 99)
owner_idx = pd.Series(np.arange(len(S)), index=S.entity_id).loc[q.owner.values].values
for i in range(0, len(Q), 5000):
    M = sp_matmul_topn(Qx[i:i + 5000], St, top_n=K, n_threads=8).tocsr()
    for j in range(M.shape[0]):
        a, b = M.indptr[j], M.indptr[j + 1]
        cols, vals = M.indices[a:b], M.data[a:b]
        o = np.argsort(-vals)
        r = np.flatnonzero(cols[o] == owner_idx[i + j])
        if len(r): hits[i + j] = r[0]
res = {k: float(np.mean(hits < k)) for k in (1, 3, 8, 20, 40)}
nonascii = q.business_name.map(lambda s: any(ord(ch) > 127 for ch in s)).values
emp = q.business_address.str.strip().isin(["", "null", "NULL"]).values
print(f"[{C}] rev-combo recall@K: {res} | {time.time()-t0:.0f}s")
print(f"   R@8 by type: native-script names {np.mean(hits[nonascii] < 8):.4f} (n {nonascii.sum()}), "
      f"empty addr {np.mean(hits[emp] < 8):.4f} (n {emp.sum()}), other {np.mean(hits[~nonascii & ~emp] < 8):.4f}")
miss = q[hits >= 8].copy()
miss["owner_name"] = S.set_index("entity_id").loc[miss.owner, "name_core"].values
miss["owner_addr"] = S.set_index("entity_id").loc[miss.owner, "addr_norm"].values
miss["q_name"] = Q.name_core.values[hits >= 8]; miss["q_addr"] = Q.addr_norm.values[hits >= 8]
miss.head(400).to_csv(f"{R}/bin_miss_{C}.tsv", sep="\t", index=False)
```

### A.7 `real_r4_lab.py`
```python
"""Real-data mini-lab for R4 (shared-perturbation stage C) BEFORE the boxes finish.

India, hide_frac 0.19 world. 15k sampled visible S1s; candidates = forward top-40 records by
0.5*cos(char_wb3 name)+0.5*cos(word addr) over ALL India S2+S3 records (full distractor density).
Stage A = LightGBM on v10 pair features (features.build; blocking columns absent), 3-fold OOF by S1.
Stage C variants on top of OOF pA: [lp] | [lp + number geometry (u_any, s1_kept, num_dmin)] | [lp + all R4 cols].
Reports OOF macro-F0.5 (rule tuned per variant, calibrated GFM) -> the sharing features' marginal delta.
"""
import sys, time, json, gc, numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, "/Users/mac/amazon-ml-2026/code/ber_v5")
sys.path.insert(0, "/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad")
from src import decide, model, features
from src.normalize import normalize_frame, renormalize_names
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from shared_feature import compute, rare_tokens, compute_tok
from stage_c import num_dmin, lg, lr_oof, NEW, PRM

R = "/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad/real"
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
C = sys.argv[2] if len(sys.argv) > 2 else "India"
K = 40
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
LEX = json.load(open("/Users/mac/amazon-ml-2026/code/ber_v5/src/lexicon.json"))
s1 = pd.read_csv(f"{R}/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
s1 = s1[s1.country == C].reset_index(drop=True)
vis = np.random.RandomState(0).rand(len(s1)) >= 0.19
s1 = s1[vis].reset_index(drop=True)
samp = s1.sample(NS, random_state=11).reset_index(drop=True)
del s1
parts = []
for f in ("train_source2.tsv", "train_source3.tsv"):
    for ch in pd.read_csv(f"{R}/{f}", sep="\t", dtype=str, keep_default_na=False, quoting=3, chunksize=500000):
        parts.append(ch[ch.country == C])
rec = pd.concat(parts, ignore_index=True); del parts
log("records", len(rec))
gold = {}
sid = set(samp.entity_id)
for ch in pd.read_csv(f"{R}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, chunksize=500000):
    ch = ch[ch.source1_entity_id.isin(sid)]
    for s, v in zip(ch.source1_entity_id.values, ch.matched_entity_ids.values):
        gold[s] = set(x for x in v.split(",") if x)
light = lambda s: s.str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
vn = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=0.05, sublinear_tf=True, dtype=np.float32)
va = TfidfVectorizer(analyzer="word", token_pattern=r"[a-z0-9]+", min_df=2, max_df=0.05, sublinear_tf=True, dtype=np.float32)
fitp = rec.sample(600000, random_state=3)
vn.fit(light(fitp.business_name)); va.fit(light(fitp.business_address)); del fitp
w = np.float32(np.sqrt(0.5))
combo = lambda d: sp.hstack([vn.transform(light(d.business_name)) * w, va.transform(light(d.business_address)) * w]).tocsr()
Xr = combo(rec); Xs = combo(samp)
log("vectorized nnz", Xr.nnz)
XrT = Xr.T.tocsr(); del Xr; gc.collect()
M = sp_matmul_topn(Xs, XrT, top_n=K, n_threads=8).tocoo()
del XrT; gc.collect()
pairs = pd.DataFrame({"s1_i": M.row.astype(np.int32), "rg": M.col.astype(np.int64)})
log("pairs", len(pairs))
ur = np.unique(pairs.rg.values)
rsub = rec.iloc[ur].reset_index(drop=True)
pairs["r_i"] = np.searchsorted(ur, pairs.rg.values)
del rec; gc.collect()


def norm(df):
    out = []
    for i in range(0, len(df), 100000):
        d = normalize_frame(df.iloc[i:i + 100000].copy()); renormalize_names(d, LEX); out.append(d)
    return pd.concat(out, ignore_index=True)


S = features._derive(norm(samp)); Rn = features._derive(norm(rsub))
log("normalized", len(S), len(Rn))
pairs["y"] = [Rn.entity_id.iat[r] in gold.get(S.entity_id.iat[s], ()) for s, r in zip(pairs.s1_i.values, pairs.r_i.values)]
pairs = pairs.sort_values("s1_i", kind="stable").reset_index(drop=True)
retr = pairs.groupby("s1_i").y.sum()
log("positives", int(pairs.y.sum()), "of gold", sum(len(v) for v in gold.values()),
    "| pair recall", round(pairs.y.sum() / max(1, sum(len(gold.get(e, ())) for e in S.entity_id)), 4))
idf_n, idf_a = features.idf_table(S.name_core, Rn.name_core), features.idf_table(S.addr_norm, Rn.addr_norm)
X = pd.concat([features.build(pairs.iloc[i:i + 100000][["s1_i", "r_i"]], S, Rn, idf_n, idf_a)
               for i in range(0, len(pairs), 100000)], ignore_index=True)
log("features", X.shape)
y = pairs.y.values.astype(int); grp = pairs.s1_i.values
pA, it = model.oof_predict(X, y, grp, 3, params={"learning_rate": 0.1})
ng = np.array([len(gold.get(e, ())) for e in S.entity_id])
ev = decide.Evaluator(grp, pairs.y.values, ng, len(S))
log("stage A OOF rule", decide.tune_rule(ev, grp, pairs.r_i.values, pA)[0], "iter", it)
src = np.where(Rn.entity_id.str.startswith("S2").values, 2, 3).astype(np.int8)
F = compute(pairs[["s1_i", "r_i"]], S.addr_nums.values, Rn.addr_nums.values, src, pA)
F["num_dmin"] = num_dmin(S.addr_nums.values, Rn.addr_nums.values, pairs.s1_i.values, pairs.r_i.values)
F = F[NEW].astype(np.float32)
Rn_rare = rare_tokens(Rn.addr_norm.values)      # df over the candidate records (lab approximation of the country table)
FT = compute_tok(pairs[["s1_i", "r_i"]], S.addr_norm.values, Rn_rare, src, pA)
TOKC = list(FT.columns)
F = pd.concat([F, FT.astype(np.float32)], axis=1)
lp = pd.DataFrame({"lp": lg(pA)})
base_cols, geo_cols = [], ["u_any", "s1_kept", "num_dmin"]
res = {}


def score(p):
    r = decide.tune_rule(ev, grp, pairs.r_i.values, p)[0]
    pc = decide.fit_calibrator(p, y).predict(p)
    g = ev.score(decide.gfm_select(grp, pairs.r_i.values, pc.astype(np.float32), excl=True))
    return round(r, 5), round(g, 5)


res["A"] = score(pA)
for name, cols in (("C_lp", []), ("C_geo", geo_cols), ("C_all", NEW), ("C_all_tok", NEW + TOKC)):
    XC = pd.concat([lp, F[cols]], axis=1)
    pC, _ = model.oof_predict(XC, y, grp, 3, params=PRM)
    res[name] = score(pC)
    log(name, res[name])
XA = pd.concat([X, F], axis=1)                                   # R4 folded into stage A (retrain variant)
pA2, _ = model.oof_predict(XA, y, grp, 3, params={"learning_rate": 0.1})
res["A_plus_R4feats"] = score(pA2)
log("A+R4 features in stage A", res["A_plus_R4feats"])
m = (F.u_any.values == 1)
log("pos-rate | u_any&within-only", y[m & (F.sh_n.values > 0) & (F.shx_n.values == 0)].mean().round(3),
    "| u_any&cross", y[m & (F.shx_n.values > 0)].mean().round(3), "| u_any&neither",
    y[m & (F.sh_n.values == 0) & (F.shx_n.values == 0)].mean().round(3))
print(json.dumps(res))
out = pairs[["s1_i", "r_i", "y"]].copy(); out["pA"] = pA; out["pA2"] = pA2
out = pd.concat([out, F.reset_index(drop=True)], axis=1)
out.to_parquet(f"{R}/lab_{C}_pairs.parquet", index=False)
np.save(f"{R}/lab_{C}_ng.npy", ng)
```

### A.8 `lab_decisions.py`
```python
"""Decision-layer tests on the real mini-lab artifacts (lab_<C>_pairs.parquet): rule vs calibrated GFM vs
GFM with Poisson(lambda) unretrieved-match correction, +/- caps. 2-fold honest protocol: calibrator / rule tuned on
half A of the S1s, scored on half B, and vice versa (no in-sample tuning)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/Users/mac/amazon-ml-2026/code/ber_v5")
sys.path.insert(0, "/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad")
from src import decide
from decision_lab_real import gfm_miss

R = "/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad/real"
C = sys.argv[1]
col = sys.argv[2] if len(sys.argv) > 2 else "pA"
d = pd.read_parquet(f"{R}/lab_{C}_pairs.parquet"); ng = np.load(f"{R}/lab_{C}_ng.npy")
s1, r, y, p = d.s1_i.values, d.r_i.values, d.y.values, d[col].values
n1 = len(ng)
half = (np.arange(n1) % 2 == 0)
lam = float((ng.sum() - y.sum()) / n1)
print(f"{C}: S1 {n1}, pairs {len(d)}, lambda (unretrieved true per S1) = {lam:.3f}")
tot = {k: [] for k in ("rule", "gfm", "gfm_lam", "gfm_lam_half")}
for fit_half in (True, False):
    fit_rows = half[s1] == fit_half
    ev_fit = decide.Evaluator(s1[fit_rows], y[fit_rows], np.where(half == fit_half, ng, 0), n1)
    ev_sc = decide.Evaluator(s1[~fit_rows], y[~fit_rows], np.where(half != fit_half, ng, 0), n1)
    _, rule = decide.tune_rule(ev_fit, s1[fit_rows], r[fit_rows], p[fit_rows])
    cal = decide.fit_calibrator(p[fit_rows], y[fit_rows])
    m = ~fit_rows
    sel = decide.rule_select(s1[m], p[m], rule["t1"], rule["t2"], rule["delta"],
                             decide.exclusive_mask(r[m], p[m]) if rule["excl"] else None)
    nsc = (half != fit_half).sum()
    tot["rule"].append(ev_sc.score(sel) * n1 / nsc - (n1 - nsc) / nsc)     # rescale: non-scored S1 count as F=1 (gold 0, no pred)
    pc = cal.predict(p[m]).astype(np.float32)
    tot["gfm"].append(ev_sc.score(decide.gfm_select(s1[m], r[m], pc, excl=True)) * n1 / nsc - (n1 - nsc) / nsc)
    tot["gfm_lam"].append(ev_sc.score(gfm_miss(s1[m], r[m], pc, np.full(m.sum(), lam))) * n1 / nsc - (n1 - nsc) / nsc)
    tot["gfm_lam_half"].append(ev_sc.score(gfm_miss(s1[m], r[m], pc, np.full(m.sum(), lam / 2))) * n1 / nsc - (n1 - nsc) / nsc)
for k, v in tot.items():
    print(f"  {k:14s} {np.mean(v):.5f}  (halves {np.round(v, 5)})")
```

### A.9 `export_band.py`
```python
"""F8 export (research/16 F8, research/10 §5): uncertain-band pairs with RAW text for a GPU cross-encoder.

    cd ~/er && python export_band.py --prep ~/prep --data ~/a --out ~/out --dst ~/ce [--lo 0.02 --hi 0.98]

Reads only. Writes DST/train_band.parquet and DST/test_band.parquet, zstd, columns:
  train: s1_id, r_id, fold (hash(s1_id) % 2, S1-grouped), y (0/1), p (OOF stage-A p), a, b
  test : s1_id, r_id, country, p (stored final test p), a, b
where a = "<S1 name> | <S1 address>", b = "<record name> | <record address>" (raw TSV text, whitespace-collapsed).
Band = lo < p < hi. Train p = oof_pA.npy (stage A OOF), a superset-ish proxy of the final-p band.
"""
import argparse, csv, glob, hashlib, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from src.run_big import load_r, countries  # noqa: E402


def raw_text(data, split, ids):
    """entity_id -> 'name | address' for the requested ids, streaming the raw TSVs."""
    ids = set(ids); out = {}
    for i in (1, 2, 3):
        f = f"{data}/{split}/{split}_source{i}.tsv"
        for ch in pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE, chunksize=500000,
                              usecols=["entity_id", "business_name", "business_address"]):
            ch = ch[ch.entity_id.isin(ids)]
            for e, n, a in zip(ch.entity_id.values, ch.business_name.values, ch.business_address.values):
                out[e] = " ".join(f"{n} | {a}".split())
    return out


def fold_of(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--lo", type=float, default=0.02); ap.add_argument("--hi", type=float, default=0.98)
    a = ap.parse_args(); os.makedirs(a.dst, exist_ok=True)
    # ---- train band
    T = pd.read_parquet(f"{a.out}/train_features.parquet", columns=["y", "r_i", "s1_id", "country"])
    p = np.load(f"{a.out}/oof_pA.npy")
    blk = json.load(open(f"{a.out}/train_blocking.json"))
    m = (p > a.lo) & (p < a.hi)
    B = T[m].copy(); B["p"] = p[m].astype(np.float32)
    rid = np.empty(len(B), object)
    for c in sorted(set(B.country)):
        mm = (B.country == c).values
        r = load_r(a.prep, "train", c, ["entity_id"])
        rid[mm] = r.entity_id.values[B.r_i.values[mm] - blk[c]["r_off"]]
    B["r_id"] = rid
    txt = raw_text(a.data, "train", set(B.s1_id) | set(B.r_id))
    B["a"] = B.s1_id.map(txt); B["b"] = B.r_id.map(txt)
    B["fold"] = B.s1_id.map(fold_of).astype(np.int8)
    B["y"] = B.y.astype(np.int8)
    B[["s1_id", "r_id", "fold", "y", "p", "a", "b"]].to_parquet(f"{a.dst}/train_band.parquet", compression="zstd", index=False)
    print(f"train band: {len(B)} pairs ({m.mean():.3%} of OOF pairs), positives {B.y.mean():.3f}, "
          f"missing text {B.a.isna().sum() + B.b.isna().sum()}", flush=True)
    # ---- test band
    parts = []
    for c in countries(a.prep, "test"):
        d = pd.read_parquet(f"{a.out}/test_{c}.parquet", columns=["s1_id", "r_id", "p"])
        d = d[(d.p > a.lo) & (d.p < a.hi)].copy(); d["country"] = c; parts.append(d)
    D = pd.concat(parts, ignore_index=True)
    txt = raw_text(a.data, "test", set(D.s1_id) | set(D.r_id))
    D["a"] = D.s1_id.map(txt); D["b"] = D.r_id.map(txt)
    D[["s1_id", "r_id", "country", "p", "a", "b"]].to_parquet(f"{a.dst}/test_band.parquet", compression="zstd", index=False)
    print(f"test band: {len(D)} pairs by country {D.country.value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
```

### A.10 `ce_train.py`
```python
"""F8 GPU cross-encoder on the uncertain band (research/16 F8; pkd-prashant measured M -47% on this data).

    pip install torch transformers pandas pyarrow            # (Kaggle GPU images already have them)
    python ce_train.py --band DIR_WITH_PARQUETS --out DIR [--model xlm-roberta-base] [--max-len 64] [--bs 64]
                       [--epochs 1] [--lr 2e-5] [--max-train 900000] [--kill-at 13:00]

Model: xlm-roberta-base (MIT, 278M) — multilingual (Hindi/Telugu/French text is raw). Fallback on small GPUs:
microsoft/Multilingual-MiniLM-L12-H384 (MIT). Two S1-grouped folds (column `fold`): train on fold!=k, predict
OOF on fold k, and predict the whole test band with each fold model (test logit = mean of the two).
Outputs (feed to stage_c.py --extra-train / --extra-test):
  OUT/ce_oof.parquet  : s1_id, r_id, ce_logit      (train band, out-of-fold)
  OUT/ce_test.parquet : s1_id, r_id, ce_logit      (test band)
Kill switch: exits cleanly (writing what exists) when local time passes --kill-at, so the owner never overruns.
Throughput guide [SPEC]: T4 fp16, seq 64: train ~400-500 pairs/s, infer ~2k pairs/s; RTX 5060 8GB similar or
faster with bf16. Checkpoints nothing on purpose (1 epoch, minutes per fold).
"""
import argparse, os, time, datetime
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup


class PairDS(Dataset):
    def __init__(self, a, b, y=None):
        self.a, self.b, self.y = list(a), list(b), (None if y is None else np.asarray(y, np.float32))
    def __len__(self): return len(self.a)
    def __getitem__(self, i): return i


def killed(t):
    if not t: return False
    h, m = map(int, t.split(":")); now = datetime.datetime.now()
    return now.hour * 60 + now.minute >= h * 60 + m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="xlm-roberta-base"); ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--max-train", type=int, default=900000)
    ap.add_argument("--kill-at", default="13:00")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.bfloat16 if (dev == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    tok = AutoTokenizer.from_pretrained(a.model)
    tr = pd.read_parquet(f"{a.band}/train_band.parquet"); te = pd.read_parquet(f"{a.band}/test_band.parquet")
    for d in (tr, te):
        d["a"] = d.a.fillna(""); d["b"] = d.b.fillna("")

    def batches(df, bs, shuffle, rng=None):
        idx = np.arange(len(df)); (rng.shuffle(idx) if shuffle else None)
        for s in range(0, len(idx), bs):
            j = idx[s:s + bs]
            enc = tok(list(df.a.values[j]), list(df.b.values[j]), truncation=True, max_length=a.max_len,
                      padding=True, return_tensors="pt")
            yield j, {k: v.to(dev) for k, v in enc.items()}

    @torch.no_grad()
    def predict(model, df):
        model.eval(); out = np.zeros(len(df), np.float32)
        for j, enc in batches(df, a.bs * 4, False):
            with torch.autocast(dev, dtype=amp, enabled=dev == "cuda"):
                out[j] = model(**enc).logits.float().squeeze(-1).cpu().numpy()
            if killed(a.kill_at): raise SystemExit("kill-at reached during inference")
        return out

    oof = np.full(len(tr), np.nan, np.float32); te_logit = np.zeros(len(te), np.float32); n_models = 0
    rng = np.random.RandomState(0)
    for k in (0, 1):
        fit = tr[tr.fold != k]
        if len(fit) > a.max_train: fit = fit.sample(a.max_train, random_state=k)
        model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        steps = a.epochs * ((len(fit) + a.bs - 1) // a.bs)
        sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
        scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda" and amp == torch.float16))
        lossf = torch.nn.BCEWithLogitsLoss(); t0 = time.time(); step = 0
        for ep in range(a.epochs):
            model.train()
            for j, enc in batches(fit, a.bs, True, rng):
                y = torch.tensor(fit.y.values[j], dtype=torch.float32, device=dev)
                with torch.autocast(dev, dtype=amp, enabled=dev == "cuda"):
                    loss = lossf(model(**enc).logits.squeeze(-1).float(), y)
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step(); step += 1
                if step % 500 == 0:
                    print(f"fold {k} step {step}/{steps} loss {loss.item():.4f} {(step*a.bs)/(time.time()-t0):.0f} pairs/s", flush=True)
                if killed(a.kill_at): raise SystemExit("kill-at reached during training")
        vm = (tr.fold == k).values
        oof[vm] = predict(model, tr[vm]); te_logit += predict(model, te); n_models += 1
        pd.DataFrame({"s1_id": tr.s1_id.values, "r_id": tr.r_id.values, "ce_logit": oof}).to_parquet(f"{a.out}/ce_oof.parquet", index=False)
        pd.DataFrame({"s1_id": te.s1_id.values, "r_id": te.r_id.values, "ce_logit": te_logit / n_models}).to_parquet(f"{a.out}/ce_test.parquet", index=False)
        from sklearn.metrics import roc_auc_score
        print(f"fold {k} done in {time.time()-t0:.0f}s | OOF AUC on fold {k}: {roc_auc_score(tr.y.values[vm], oof[vm]):.4f} "
              f"| stage-A p AUC on same rows: {roc_auc_score(tr.y.values[vm], tr.p.values[vm]):.4f}", flush=True)
        del model, opt; torch.cuda.empty_cache() if dev == "cuda" else None


if __name__ == "__main__":
    main()
```

### A.11 `splice_country.py`
```python
"""Splice per-country decisions from a run dir into a base submission (exact: macro F0.5 is a mean over S1s).

    python splice_country.py --base-dir BASE_OUTPUT_DIR --run RUN_DIR --countries india [us ...] \
                             --s1 test_source1.tsv --out OUT_DIR [--gt hidden.tsv]

BASE_OUTPUT_DIR holds matching_results.tsv + candidate_pairs.tsv (e.g. the Slot-1 blend). For every S1 of the listed
countries, both rows are replaced from RUN_DIR/test_<country>.parquet (sel -> matches, all rows -> candidates, ordered
by p desc); every other row is copied byte-identically. Row order = BASE file order.
"""
import argparse, csv, os, sys
import pandas as pd


def read_tsv(p):
    with open(p, encoding="utf-8") as fh:
        r = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        head = next(r)
        return head, [(row[0], row[1] if len(row) > 1 else "") for row in r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--countries", nargs="+", required=True); ap.add_argument("--s1", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--gt")
    a = ap.parse_args()
    cty = {}
    with open(a.s1, encoding="utf-8") as fh:
        r = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE); h = next(r)
        ei, ci = h.index("entity_id"), h.index("country")
        for row in r:
            cty[row[ei]] = row[ci].strip().lower()
    match, cand = {}, {}
    for c in a.countries:
        d = pd.read_parquet(f"{a.run}/test_{c}.parquet", columns=["s1_id", "r_id", "p", "sel"])
        d = d.sort_values(["s1_id", "p"], ascending=[True, False], kind="stable")
        for s, g in d.groupby("s1_id", sort=False):
            cand[s] = ",".join(dict.fromkeys(g.r_id.values))
            match[s] = ",".join(dict.fromkeys(g.r_id.values[g.sel.values]))
    os.makedirs(a.out, exist_ok=True)
    n_rep = 0
    for fname, src in (("matching_results.tsv", match), ("candidate_pairs.tsv", cand)):
        head, rows = read_tsv(os.path.join(a.base_dir, fname))
        with open(os.path.join(a.out, fname), "w", encoding="utf-8", newline="") as fh:
            fh.write("\t".join(head) + "\n")
            for e, v in rows:
                if cty.get(e) in a.countries:
                    v = src.get(e, "")                     # S1 with no candidates in the run -> empty
                    n_rep += fname == "matching_results.tsv"
                fh.write(f"{e}\t{v}\n")
    print(f"spliced {n_rep} S1 rows for {a.countries}; other rows byte-identical to {a.base_dir}")
    if a.gt:
        sys.path.insert(0, "/Users/mac/amazon-ml-2026/code/business_entity_resolution")
        from src.metrics import macro_fbeta, parse_ids
        rd = lambda p: {e: parse_ids(v) for e, v in read_tsv(p)[1]}
        print(f"macro F0.5 = {macro_fbeta(rd(os.path.join(a.out, 'matching_results.tsv')), rd(a.gt)):.6f}")


if __name__ == "__main__":
    main()
```

### A.12 `t1a.patch` (unified diff against code/ber_v5/src/run_big.py; apply with `patch <copy>/src/run_big.py < t1a.patch`)
```diff
--- /Users/mac/amazon-ml-2026/code/ber_v5/src/run_big.py	2026-09-26 17:56:56
+++ src/run_big.py	2026-09-26 23:50:36
@@ -90,7 +90,7 @@
              "n_cand_s1"] + EXTRA_PAIR + ["prune_score", "rev_rank", "b_key_revrk"]
 
 
-def prune_by_similarity(pairs, s1, r, k, chunk=2_000_000):
+def prune_by_similarity(pairs, s1, r, k, chunk=2_000_000, groups=1, prot=2):
     """The LAST candidate-generation step, a fixed rule (no learned model): score each blocked pair with
     max(name token-set, no-space name ratio, phonetic token-set) + address token-set + 50 x combined-key cosine,
     and keep each S1's top-k. The survivors are exactly what the matcher scores, i.e. candidate_pairs.tsv.
@@ -110,7 +110,10 @@
         score[lo:lo + chunk] = name + cp(A["addr_norm"], B["addr_norm"], fuzz.token_set_ratio) \
             + 50 * np.nan_to_num(pairs.b_key.values[lo:lo + chunk])
     rev_rank = group_rank(ri, score)
-    keep = (group_rank(si, score) <= k) | (rev_rank <= 2)
+    # T1a: at test, rank the record-side PROTECTION among S1 groups the size of the train sample (train saw only the
+    # sampled S1s as a record's suitors), so test keeps the protected positives the model was trained on.
+    prot_rank = rev_rank if groups <= 1 else group_rank(ri.astype(np.int64) * groups + (si % groups), score)
+    keep = (group_rank(si, score) <= k) | (prot_rank <= prot)
     out = pairs[keep].reset_index(drop=True)
     out["prune_score"] = score[keep]
     out["rev_rank"] = rev_rank[keep]                              # this S1's standing among the record's suitors
@@ -222,9 +225,9 @@
     return out.sort_values(["s1_i", "b_rank"]).reset_index(drop=True)
 
 
-def block(s1, r, cfg, keep_s1=None, post=None):
+def block(s1, r, cfg, keep_s1=None, post=None, s1_parts=4):
     return block_country(s1, r, fields=KEY_FIELDS, kscale=cfg["kscale"], max_df_keys=cfg["max_df_keys"], log=log,
-                         keep_s1=keep_s1, post=post)
+                         keep_s1=keep_s1, post=post, s1_parts=s1_parts)
 
 
 def idf_for(s1, r):
@@ -497,7 +500,8 @@
     s1 = load(a.prep, "test", "s1", c, BLOCK_COLS).reset_index(drop=True)
     r = load_r(a.prep, "test", c, BLOCK_COLS)
     log(f"[test {c}] s1 {len(s1)} r {len(r)}")
-    pairs = block(s1, r, cfg, post=lambda d: prune_topk(d, cfg["K"]))      # merged + pruned per S1 range
+    parts = int(np.ceil(len(s1) / cfg["test_range_s1"])) if cfg.get("test_range_s1") else 4   # T1a
+    pairs = block(s1, r, cfg, post=lambda d: prune_topk(d, cfg["K"]), s1_parts=max(4, parts))  # merged + pruned per S1 range
     pairs = add_dup_counts(pairs, s1, r)
     sib = sibling_edges(r, k=cfg["sib_k"], max_df_keys=cfg["max_df_keys"], log=log)
     np.savez_compressed(f"{a.out}/test_{c}_sib.npz", q=sib[0], s=sib[1], v=sib[2])
@@ -519,7 +523,9 @@
     s1 = load(a.prep, "test", "s1", c, features.NEEDED).reset_index(drop=True)
     r = load_r(a.prep, "test", c, features.NEEDED)
     n0 = len(pairs)
-    pairs = prune_by_similarity(pairs, s1, r, meta["cfg"]["prune_k"])[PAIR_COLS]
+    G = int(np.ceil(len(s1) / cfg["test_group_s1"])) if cfg.get("test_group_s1") else 1           # T1a
+    pairs = prune_by_similarity(pairs, s1, r, meta["cfg"]["prune_k"], groups=G,
+                                prot=cfg.get("test_prot", 2))[PAIR_COLS]
     log(f"[test {c}] similarity prune keeps {len(pairs)} of {n0} ({len(pairs) / len(s1):.2f} per S1)")
     log(f"[test {c}] scoring {len(pairs)} candidates")
     idf_n, idf_a = idf_for(s1, r)
@@ -582,6 +588,7 @@
             p = p[order]
             log(f"[test {c}] anchor pass added {len(e_i)} pairs (max p {pa.max():.3f})")
     fpool.close()
+    pA_keep = np.asarray(p, np.float32).copy()                     # stage-A p for post-hoc stage-B variants
     if meta.get("use_b"):
         mB = [lgb.Booster(model_file=f"{a.out}/models/B_{i}.txt") for i in range(meta["n_models_b"])]
         sib = load_sib(f"{a.out}/test_{c}_sib.npz")
@@ -612,14 +619,14 @@
                                  decide.exclusive_mask(pairs.r_i.values, p) if rule["excl"] else None,
                                  t_shift=np.full(len(p), shift, np.float32))
     out = pd.DataFrame({"s1_id": s1.entity_id.values[pairs.s1_i.values],
-                        "r_id": r.entity_id.values[pairs.r_i.values], "p": p, "sel": sel})
+                        "r_id": r.entity_id.values[pairs.r_i.values], "p": p, "sel": sel, "pA": pA_keep})
     atomic_parquet(out, f"{a.out}/test_{c}.parquet")
     nl = np.bincount(pairs.s1_i.values[sel], minlength=len(s1))
     stats = {"n_s1": len(s1), "n_cand": len(pairs), "links": int(sel.sum()), "mean_links": round(float(nl.mean()), 3),
              "frac_empty": round(float((nl == 0).mean()), 4), "shift": shift}
     json.dump(stats, open(f"{a.out}/test_{c}_stats.json", "w"))
     for f in (ppath, spath):                                     # checkpoints no longer needed
-        if os.path.exists(f):
+        if os.path.exists(f) and not cfg.get("keep_test_ckpt"):
             os.remove(f)
     log(f"[test {c}] done", stats)
```
