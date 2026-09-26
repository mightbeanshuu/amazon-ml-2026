# 06 — Classifier / objective / calibration frontier (research pass 6)

Written 26 Sep 2026 (~36 h before close). Web research + two local reads only:
`student_resource/README.md` (metric: **macro F0.5 per S1 entity, singletons score 1.0 for a correct empty list, 0.0 for any false merge**) and
`runs/real_v4/run_log_train.json` (**OOF A = 0.9607 @ t1=0.65/t2=0.70; OOF B = 0.9618 @ t1=0.50/t2=0.75, excl=True; blocking oracle 0.9833 India / 0.9889 US; France LOCO 0.9146 @ t=0.85; seeds=[0]; train_per_country=60k**).
Scope: model/objective/calibration only. Blocking, features, pseudo-labelling, SIGMOD/collective ER are in 03/04/05 and are NOT repeated.

**Headroom framing [E].** Matcher loss = oracle − OOF ≈ 0.021 (India) / 0.025 (US). Everything below fights over that ~0.02, so treat any single-model claim of "+0.01 from a better booster" as fantasy; the measured literature deltas at this stage are 0.0005–0.003 each, and they stack.

Tags: **[V]** = read in the cited source; **[U]** = another team's claim; **[E]** = our estimate.

---

## Ranked: do these, in this order

| # | Change | Expected LB delta [E] | CPU cost (our scale) | Risk |
|---|---|---|---|---|
| 1 | **Calibrate stage-B per country (isotonic on OOF), then replace the t1/t2 grid with the plug-in expected-F0.5 subset decision per S1** | +0.001–0.003 | minutes | low |
| 2 | **Seed bagging ×5 (LGB, seeds 0–4, average raw scores before thresholding)** | +0.0005–0.002 | ~5× one train (~1–2 h wall on Kaggle 4 vCPU at our 6.1M-pair, 60k/ctry scale; run overnight) | ~zero |
| 3 | **Add CatBoost (ordered boosting) + XGBoost `hist` on the same features; rank-average with LGB** | +0.001–0.003 | CatBoost ≈ 3–10× LGB time; XGB hist ≈ 1.5–3× | low |
| 4 | **LightGBM DART for the blend (not as the solo model)** | +0.000–0.002 | 2–4× LGB, no early stopping | med |
| 5 | **Monotone constraints on the ~10 cleanest similarity features — France/LOCO insurance, not an i.i.d. win** | +0.000–0.002 (France only) | free | low |
| 6 | Focal / weighted CE objectives | ~0 after threshold retuning | cheap to test | low |
| 7 | LambdaRank per-S1 groups | **skip** (breaks calibration; no measured win for this decision shape) | — | high |
| 8 | Deep tabular (TabM/FT-T/SAINT/TabNet/NODE/GRANDE), EBM, linear_tree | **skip** (verdict below) | hours–days | high |

---

## 1. GBDT frontier — what actually moves the needle

### 1.1 CatBoost / XGBoost vs LightGBM: measured deltas are small and dataset-dependent
- Florek & Zagdański, *Benchmarking state-of-the-art gradient boosting algorithms for classification* (arXiv:2305.17094) [V]: across their suite, **baseline (untuned) XGBoost and CatBoost have the best AUC**, while **LightGBM is 15–37× faster**; after tuning, the three converge to within noise on most datasets. <https://arxiv.org/pdf/2305.17094>
- Riskified's four fraud datasets (imbalanced, engineered features — the closest published shape to ours) [V]: no single winner; deltas between tuned LGB/XGB/CatBoost are fractions of a percent of AUC. <https://www.riskified.com/resources/article/boosting-comparison/>
- **Implication:** CatBoost's ordered boosting is not a solo upgrade over our tuned LGB (expect ±0.000–0.002 OOF F0.5 solo [E]). Its value is **decorrelated errors for the blend** (see §4). Licences: LightGBM MIT, XGBoost Apache-2.0, CatBoost Apache-2.0 — all compliant.

**Exact config to try (CatBoost, CPU):** `CatBoostClassifier(iterations=3000, learning_rate=0.08, depth=8, l2_leaf_reg=6, bootstrap_type='Bernoulli', subsample=0.8, eval_metric='Logloss', od_wait=200, thread_count=4)` on the identical feature matrix (no cat features → ordered target stats don't apply, so also try `boosting_type='Ordered'` vs `'Plain'`; on pure-numeric data `Plain` is usually equal and 2–3× faster).
**10-min OOF test:** train on 1 fold's train split subsampled to 500k pairs, score that fold's val: compare fold-F0.5 (after re-running the existing threshold tuner on its scores) vs LGB same fold. If within −0.001, keep for blend; if ≥ +0.001 solo, extend to full data.

### 1.2 DART
- AMEX Default Prediction 2022: the top of the LB was dominated by **LightGBM DART** models; e.g. 9th place explicitly mixes `dart-LGBM` + `goss` + gbdt "to generate diversity" [V] <https://www.kaggle.com/competitions/amex-default-prediction/writeups/george-reus-9th-place-solution-xgboost-lgbm-nn>; 21st place likewise gbdt+dart LGB [V] <https://www.kaggle.com/competitions/amex-default-prediction/writeups/angus-ryota-joe-21st-solution-and-code-sharing>. Community-reported dart-vs-gbdt gaps there were ~+0.001–0.002 of the (F-like) AMEX metric [U].
- kazanova (Kaggle #1 GM) recommended `boosting=dart` as the one thing to try, from Instacart (per-group F1 metric!) [U] <https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56429>.
- **Costs:** no early stopping (fix `num_iterations`), 2–4× train time.
**Config:** `boosting='dart', drop_rate=0.1, skip_drop=0.5, num_iterations=<1.3× the gbdt best_iter>, learning_rate=0.08`, rest unchanged.
**10-min OOF test:** one fold, 500k-pair subsample, fixed 800 iters: fold-F0.5 (thresholds retuned) vs gbdt same budget. Keep only if ≥ +0.0005; else use it purely as a 4th blend member.

### 1.3 GOSS, deeper trees, row subsampling
- GOSS is a **speed** technique: the LightGBM paper reports up to 20× speedup at "almost the same accuracy" — i.e. accuracy-neutral at best [V] <https://proceedings.neurips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf>. Skip for accuracy; use only if we need faster retrains, or as a diversity member (AMEX 9th used one goss model [V]).
- Deeper trees + subsampling: no published measured win specific to similarity-pair data; standard tuning territory. If our `num_leaves` < 127, one cheap probe: `num_leaves=255, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, learning_rate=0.05` — interaction depth is what lets the tree express "name sim high AND address number differs", our decoy failure mode [E]. 10-min test: same one-fold protocol.

### 1.4 Monotone constraints on similarity features
- Honest evidence status: **no strong published measured i.i.d. gain**; LightGBM supports `monotone_constraints` with `monotone_constraints_method='advanced'` (Auguste et al. method) at negligible cost. The argument for us is **distribution shift**: constraints stop the model exploiting country-specific score quirks that reverse on France — and our France LOCO threshold (0.85 vs 0.50 OOF) says the model IS over-confident off-distribution [E, from run_log].
**Config:** +1 constraints on the ~10 monotone-by-construction features (name jaro, token-set ratio, char-3gram cos, address cos, …), 0 elsewhere; `monotone_constraints_method='advanced'`.
**10-min OOF test:** one fold + our existing LOCO(India→US) harness: accept if LOCO-F0.5 improves ≥ +0.001 with OOF drop ≤ 0.0005.

### 1.5 linear_tree
- `linear_tree=true` fits linear models in leaves; helps on smooth monotone targets, costs ~2–3× RAM/time, and has only anecdotal Kaggle evidence (TPS threads) — **skip** at 36 h unless everything else is done [E].

---

## 2. Objectives

### 2.1 Focal loss / class-balanced CE: real in papers, mostly absorbed by our threshold tuner
- *Improving GBDT Performance on Imbalanced Datasets* (arXiv:2407.14381) [V]: WCE/focal/ASL/ACE/AWE over LGB/XGB/CatBoost/SketchBoost, 15 binary datasets: improvements on 13/15, **+0.38% to +28.91% F1**, weighted CE the most consistent; but "the optimal class-balanced technique varies by dataset". <https://arxiv.org/html/2407.14381v1>
- The classic LGB-focal-loss write-up reports **~+2% F1** on two imbalanced datasets [V] <https://medium.com/data-science/lightgbm-with-the-focal-loss-for-imbalanced-datasets-9836a9ae00ca> (implementation recipe: <https://maxhalford.github.io/blog/lightgbm-focal-loss/>).
- **The catch for us:** those baselines use the default 0.5 cut. Re-weighting mostly *shifts the score distribution*; a tuned per-country threshold (which we already have) recovers most of it, and *any* weighting change forces recalibration (§3). At 16% positives we are not even severely imbalanced. Expected residual gain ≈ 0 to +0.001 [E].
- Downsampling vs weighting: for a precision-heavy target, keep **all negatives** (they define the precision boundary); never downsample negatives. If anything, try `scale_pos_weight < 1` (e.g. 0.5) to sharpen precision, then retune thresholds — no published number for this direction; it's a free experiment. (Background on scale_pos_weight semantics: <https://stats.stackexchange.com/questions/243207/what-is-the-proper-usage-of-scale-pos-weight-in-xgboost-for-imbalanced-datasets>.)
**10-min OOF test:** one fold, 500k pairs, three runs {baseline, focal γ=2 α=0.25, spw=0.5}; compare fold-F0.5 after threshold retune. Drop unless ≥ +0.001.

### 2.2 LambdaRank / per-group ranking: SKIP — the metric wants calibrated subsets, not orderings
- Our decision is "choose a **subset** (possibly empty) per S1", with singleton reward. LambdaRank (LGB `objective='lambdarank'`, group = S1) optimises NDCG-style *ordering inside the group*: it produces **uncalibrated scores that cannot cross groups or feed a threshold/expected-F rule**, and it literally cannot express "predict nothing" (docs: <https://xgboost.readthedocs.io/en/latest/tutorials/learning_to_rank.html>, mechanics: <https://ffineis.github.io/blog/2021/05/01/lambdarank-lightgbm.html>).
- Decision-theoretically, the Bayes-optimal F-measure decision comes from **calibrated pointwise probabilities + a plug-in subset rule** — Waegeman et al., JMLR 2014 [V] <https://www.jmlr.org/papers/v15/waegeman14a.html>; that's §3.2, not a new objective.
- Empirically, the entity-matching Kaggle closest to us — **Foursquare Location Matching 2022** (POI matching, per-group IoU) — was won with **binary pair classifiers + per-group post-processing**, not rankers: 9th place = binary LGB over 40 features, tuned prob thresholds, CV 0.905 [U] <https://www.kaggle.com/competitions/foursquare-location-matching/writeups/taksai-9th-place-solution>; 16th = shallow LGBs on 10-NN groups [U] <https://www.kaggle.com/competitions/foursquare-location-matching/writeups/silogram-a-few-notes-on-16-solution>. No top write-up we found reports lambdarank beating binary on this shape.
- **What to keep from the ranking idea:** per-group *features* (rank within S1, margin to the group's best, record-side margin) inside the binary model — already recommended in 04 §3 and partly built (stage B siblings). That captures the competition signal without losing calibration.

---

## 3. Calibration → decision layer (the highest risk-adjusted item here)

### 3.1 Which calibrator
- Niculescu-Mizil & Caruana, ICML 2005 [V]: boosted trees are systematically **under-confident (sigmoid-distorted)**; both Platt and isotonic fix them, and **with ≥ ~1000 calibration points isotonic ≥ Platt always** — we have millions of OOF points, so isotonic is safe from its small-sample overfitting failure mode. <https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf>
- Large-scale post-hoc comparison (arXiv:2601.19944, 2026) [V]: expected log-loss reduction **Venn-Abers −14.2%, beta −13.7%, Platt −9.8%**; beta calibration improves the metric most frequently (57.5% of instances), Venn-Abers 54.2%, isotonic 52.5%. <https://arxiv.org/html/2601.19944v1>
- Beta calibration (Kull et al., AISTATS 2017) [V]: 3-parameter map, fixes exactly the skewed-score regime, "fitting is as easy as fitting a logistic curve" <https://proceedings.mlr.press/v54/kull17a.html>, code <https://betacal.github.io/>.
- Venn-Abers: `pip install venn-abers` (ip200) [V] <https://github.com/ip200/venn-abers> — gives calibrated p plus an uncertainty interval (p0,p1); the interval width is itself a useful "abstain → singleton" signal. Cost: an isotonic fit, seconds.
- Caveat: raw AUC/ranking is untouched by any monotone calibrator — **the win only materialises through a probability-consuming decision rule**, which is §3.2, and through cross-country comparability (fit **per country**; our France threshold 0.85-vs-0.50 gap is a calibration gap wearing a threshold costume [E]).

### 3.2 Replace the t1/t2 grid with the plug-in expected-F0.5 subset decision
- For each S1: sort calibrated candidate probs p1≥p2≥…; for k=0..K compute expected F0.5 of taking the top-k (k=0 has expected utility Π(1−pi), the singleton reward — the metric's empty-set rule drops out of the algebra naturally); pick the argmax. Under independence this is the Bayes-optimal decision; exact O(n²)–O(n³) algorithms: Jansche/Ye et al. ICML 2012 [V] <https://icml.cc/2012/papers/175.pdf>, exact general maximizer: Dembczyński et al. NIPS 2011 [V] <https://proceedings.neurips.cc/paper/2011/hash/71ad16ad2c4d81f348082ff6c4b20768-Abstract.html>, consistency theory: Waegeman et al. JMLR 2014 [V] <https://www.jmlr.org/papers/v15/waegeman14a.html>.
- Measured, on THIS competition [U]: Arshbir1 got **+0.0017** macro-F0.5 from expected-F0.5 decisions on *calibrated stage-2* scores and **−0.0013** on raw stage-1 scores (04 §5's evidence — calibrate first or it backfires) <https://github.com/Arshbir1/Amazon_ML_Challenge/blob/main/student_resource/business_entity_resolution/experiments/results.md>; Epic021's soft-exclusivity beat hard argmax by +0.0008 [U].
- Keep the hard caps (≤5 S2 / ≤6 S3 per S1) as a mask on top.
**Exact plan:** fit per-country isotonic + beta on stage-B OOF (pick by which yields higher F0.5 under the expected-F rule, not by log-loss); apply expected-F0.5 subset selection with the existing exclusivity logic.
**10-min OOF test:** pure post-processing on the saved `oof_pA.npy`/stage-B OOF — no retraining: score {raw+t1/t2 (current), isotonic+expected-F, beta+expected-F} per country. Ship whichever wins; risk ≈ 0 because the current rule stays available.

---

## 4. Ensembling: the boring, measured +0.001–0.003

- **Seed bagging**: NVIDIA Kaggle-Grandmaster playbook (Deotte/Titericz/Onodera) [V]: 100-seed XGB ensemble ≈ **0.379 vs 0.376 MAP@3** single-seed (~+0.8% relative), monotone improvement, plateauing early — most of the gain arrives by ~5 seeds. <https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/> Our `run_log` shows `seeds=[0]` — this is unharvested. Average **raw margins** across seeds before calibration/thresholds.
- **Family blending / stacking depth**: Kaggle Playground S6E4 3rd-place stacking log [U]: LB 0.98110 (XGB+CB+LGB+MLP) → 0.98121 (+2 NN families) → 0.98145 (+HGB/RF) → 0.98201 (+LogReg/SVC) — i.e. **+0.0009 total from widening a stack that already had the big three**; diminishing but real, and a single-level stacker (LGB/LogReg on OOF) captured it — depth ≥ 2 added nothing. <https://www.kaggle.com/competitions/playground-series-s6e4/writeups/error-diversity-matters-200-model-stacking-soluti>
- For us [E]: 5-seed LGB bag (+0.0005–0.001), + CatBoost + XGB rank-average or feed all three OOFs into the existing stacking stage (+0.0005–0.002 more). Time-boxed: seeds tonight, families tomorrow morning.
**10-min OOF test:** we already have LGB OOF; train 1 extra seed on one fold, check corr(seed0, seed1) of scores — if < 0.995, the bag will pay; blend the two and confirm fold-F0.5 moves ≥ +0.0003.

---

## 5. Deep tabular reality check: don't

- Grinsztajn, Oyallon, Varoquaux, NeurIPS 2022 D&B [V]: tuned **tree ensembles beat FT-Transformer/ResNet/SAINT-class models** on 45 medium tabular datasets, even before their speed advantage; NNs specifically lose on *irregular target functions, uninformative features, non-rotation-invariant data* — engineered similarity features are the textbook case. <https://arxiv.org/abs/2207.08815>
- McElfresh et al., NeurIPS 2023 (176 datasets, 19 algos) [V]: "the NN-vs-GBDT debate is overemphasized — **light HPO on a GBDT beats choosing between NNs and GBDTs** on most datasets"; GBDTs win precisely on **skewed/heavy-tailed features** (our ratios and cosines pile up at 0 and 100). <https://arxiv.org/abs/2305.02997>
- TabM (ICLR 2025, Apache-2.0) is the one credible modern exception — rank 1.7 avg across 46 datasets, ahead of tuned GBDT [V] <https://arxiv.org/abs/2410.24210>, <https://github.com/yandex-research/tabm> — but it's a GPU-era MLP ensemble: at 6.1M pairs × 75 features on 4 CPU threads a proper tuned run is **hours per fold**, and its reported edge is ~rank-level, not the +0.005 we'd need to justify that. **No** at 36 h; note it for future comps.
- GRANDE (ICLR 2024) [V] <https://openreview.net/forum?id=XEFWBxi075>: wins mostly on *small/medium* datasets, TensorFlow/GPU, doesn't claim large-N CPU viability — no.
- TabPFN — licence fails (already ruled out in 03); NODE/TabNet — consistently below tuned GBDT in both benchmarks above; EBMs (interpretml, MIT) — interpretability tool, accuracy ≤ tuned LGB on these benchmarks — no.
- Cross-encoders — ruled out in 03; not revisited.

---

## 6. What this changes in the pipeline (summary)

Tonight: (1) §3 calibrate + expected-F0.5 decision — pure post-processing on existing OOF, the only item with a *this-competition* measured number (+0.0017 [U]); (2) launch 5-seed LGB bag + one CatBoost + one XGB on Kaggle kernels. Tomorrow: (3) blend/stack the families, retune thresholds per country on blended calibrated scores; (4) if slack remains, the DART member and monotone-constraint LOCO test. Skip lambdarank, focal loss (test-then-drop), and all deep tabular. Combined honest expectation: **+0.002–0.006 macro-F0.5** from this axis [E] — real, but smaller than the blocking-recall lever in 04; do not let it displace 04 items 2–4.
