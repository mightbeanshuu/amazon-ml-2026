# 12 — Journal + competition sweep for 0.975 → ≥0.99 (26 Sep 2026, written incrementally from ~22:00 IST)

Scope: literature/competition techniques that could move **macro F0.5 per S1 entity** on our clean-clean, record-side-exclusive, synthetic-noise ER task. Sources: Consensus API (peer-reviewed), arXiv full text, Kaggle/competition writeups. Every "reported gain" is quoted from the source's OWN experiments; if I could not trace a number to the primary source it says so. Tags: **[LIT]** literature-backed transfer argument, **[SPEC]** speculative.

Status: **COMPLETE** (loop 1 sweep + loop 2 gap re-queries, ~23:55 IST). Consensus API hard cap hit mid-sweep; the remainder via OpenAlex/arXiv/Kaggle.

## 0. Ranked shortlist

**Bottom line:** the literature offers **no CPU-feasible single technique worth +0.015**.
- Every source-backed lever below lands in the +0.0005–0.004 band.
- Stacked optimistically (they overlap, so this is not additive), they are worth **~+0.004–0.008**. That puts v10's ~0.975 at **~0.979–0.983**, consistent with `10-frontier` §0.
- The two items with the largest *absolute* upside are both matcher/recall scale-ups, not new models:
  - **B1**: v10 trains on only ~5–14% of the labelled S1s.
  - **B6**: a learned prefilter replacing the fixed-rule top-28 prune.
- Both need a retrain window. Everything else is a decision-layer or calibration fix that runs in minutes.

Ranking = (mid expected Δ × confidence) ÷ cost in hours, shown ×10⁴. "Conf" is my probability the sign is positive and the magnitude is in range. Δ is in global macro-F0.5.

| # | Technique (section) | Expected Δ | Conf | Cost | Score | Evidence (their own numbers) |
|---|---|---|---|---|---|---|
| 1 | **Calibrate after blending** tonight's two seed families: average raw logits, then refit per-country isotonic on the blended OOF, not on members (F1). This confirms the `10-frontier` P1 design; just make sure the isotonic refit happens *after* the blend | +0.000–0.001 | 0.8 | 0.25 h | 16 | Wu & Gales 2021: CIFAR-100 ECE 11.55 → members-calibrated 3.24 → **post-combination 2.19**. "Ensemble predictions will be under-confident." |
| 2 | **Per-country decision choice**: GFM vs LOCO-tuned rule picked *per country by LOCO*, not globally by OOF (A1) | +0.000–0.002 | 0.5 | 0.5 h | 10 | Ye et al. ICML 2012: DTA 39.9 vs tuned threshold 36.6 when calibrated. EUM "more robust against model misspecification". Our synth: GFM wins OOF, loses test. |
| 3 | **France label-free anchoring**: choose FR shift so predicted links/S1 and empty-rate hit prior-implied targets (B7) | +0.000–0.002 | 0.3 | 0.5 h | 6 | Shopee 3rd place: target mean cluster size, 0.781 → 0.79 (**+0.009**, uncalibrated-embedding regime). Our synth says ≤ ±0.0005. |
| 4 | **Size-cap sanity**: ≤5 S2 and ≤6 S3 per S1 (GT caps); drop the lowest-p extras (A2) | +0.0000–0.0003 | 0.9 | 0.25 h | 5.4 | GT caps (`10-frontier` §1.1). A2/A3: nothing fancier than greedy/argmax is needed. |
| 5 | **Count-prior-aware E[F0.5]**: tilt the per-S1 Poisson-binomial toward the OOF within-candidate size prior. This fixes `best_set_fbeta`'s over-valuation of "predict empty" = Π(1−p) (A4) | +0.000–0.002 | 0.4 | 1 h | 4 | Dembczynski ICML 2013: EFP > LFP on 5/6 datasets (−0.9 … **+4.2** F1). Shopee 3rd (+0.009). |
| 6 | **CSLS hubness correction** (2cos − r_q − r_t) as the prefilter score plus 3 features (E1) | +0.000–0.002 | 0.4 | 1.5 h (+ retrain for features) | 2.7 | CSLS P@1 en→fr 74.9 → **81.1**; en→it sentence retrieval 42.6 → **66.1**. Kiez: median **+3.99% hits@50**. Shopee 2nd: top-K sim mean/std features "to handle the difference between train and test". |
| 7 | **Learned light-LGBM prefilter** replacing `prune_by_similarity` (a fixed rule, top-28 ∪ rev≤2), with a recall-targeted cut per S1 and per record (B6) | **+0.001–0.003** (if v4's ~2% positive loss at the prune still holds) | 0.5 | ~5 h incl. stage-2 rerun | 2 | Foursquare 9th: max-IoU 0.994 @ 420M → **0.987 @ 3M** candidates. 1st/2nd places used the same stage. GSM PVLDB 2022. |
| 8 | **More training S1s**: 60k → ~250k per country, full refit at 1.1× best_iter, gated on a 60k/120k/240k learning curve (B1) | **+0.001–0.004** | 0.5 | 6–10 h | 1.6 (largest absolute) | Foursquare 8th: LB **0.908 → 0.928 → 0.948** at 550k/700k/1.1M ids. 9th: **+0.010** from training on all data. Counter: WDC Products Magellan RF flat (31.4 → 35.8 → 35.4). |
| 9 | **Adversarial-validation feature invariance** for France: rank features by IN∪US-vs-FR AUC, neutralise or quantile-normalise the top shifters; LOCO-validated (D1) | +0.000–0.003 | 0.3 | 3 h incl. retrain | 1.5 | DADER SIGMOD 2022: DA gain tracks the source-only gap (**+0.0** when NoDA 97.2; **+6.8 … +14.5** F1 when NoDA 57–78). Ours: LOCO 0.915 vs in-domain 0.962. |
| 10 | **Metric-aware pair weights** (ΔF0.5 of a FP/FN given S1 size; singleton-FP = 1.0; FN at n=1 = 1.0) (B2) | +0.000–0.002 | 0.25 | 0.5 h if bundled into #8, else a full retrain | 5 bundled / 0.4 alone | Foursquare 7th (IoU-loss weights; pos 0.8 / neg 1.0) and Shopee 2nd (1/size^0.4). **Neither published an ablation.** |

- Below the cut: E2 canonical views (measure the recoverable fraction first), E3 BM25 reverse probe, B3 bridge-edge pruning, D2 IW calibration. The discard list is in §2.
- **Bundle rule:** #6 (CSLS features), #8, #9 and #10 (and `10-frontier` R2 house-number geometry) all need the same retrain. If one retrain window exists, do them together and gate the whole bundle on one LOCO + OOF comparison.


## 1. Surviving techniques (detail)

Instrument note: the Consensus API hit its **hard monthly cap (30 searches, resets 1 Oct)** after ~26 of my queries (raw JSON in scratchpad `sweep12/raw_q*.json`). The rest of the sweep uses OpenAlex, arXiv full text (pdftotext), and Kaggle writeups. Every number below was read from the primary source's own table or text.

### 1.A Decision layer (set selection for macro F0.5)

**A1. Expected-F plug-in (DTA) vs thresholding (EUM): which wins depends on calibration quality.** [LIT]
- Source: Ye, Chai, Lee, Chieu, *Optimizing F-measures: A Tale of Two Approaches*, ICML 2012, arXiv:1206.4625 (full text read).
- Their own numbers:
  - Reuters-21578 macro-F1, LR, topics with ≥1 positive: threshold-0.5 **35.8**, tuned threshold (MLδ) **36.6**, EUM-trained Fδ **37.3**, DTA (MLE) **39.9**. At ≥50 positives the ranking flips: 75.2 / 75.7 / **76.6** / 75.6. DTA helps most where positives are rare.
  - Multilabel, macro-F1 DTA vs tuned threshold: yeast 48.16 vs 47.14; medical 51.48 vs 48.88; enron 21.61 vs 19.70 (C=1). Sometimes it loses: scene 68.57 vs 68.80; medical C=50: 90.12 vs Fδ 91.56.
  - Conclusion, verbatim: "EUM seems more robust against model misspecification, while given a good model, DTA seems better for handling rare classes." Under a synthetic P(X) shift the pattern reversed: Truthδ/TruthE/MLδ/MLE = 21/38/11/36 %, so DTA was more robust there.
- Transfer to us:
  - This is the textbook explanation of our own synth result in `10-frontier` §1.4. GFM (a DTA) won on OOF but lost to the tuned rule on synth test. Our match sets are small, 1–11 per S1, which is exactly the "rare/small" regime where DTA gains *if* the probabilities are right.
  - **Seen countries:** isotonic calibration is fitted in-domain, so the model is good and DTA should win.
  - **France:** the score→probability map is borrowed from other countries. That is *misspecification* of P(y|score), which is where EUM is the robust choice, so a threshold rule tuned on LOCO is the safer default.
  - Caveat from the same paper: under *pure* covariate shift with a correct model, DTA was the robust one (the 21/38/11/36 result). So this is an empirical call: **pick FR's rule by LOCO**, and do not assume either direction.
- Implementation: a per-country decision switch, GFM/expected-F for IN/US and the LOCO-tuned rule for FR. Choose each by LOCO on the held-out country, not by OOF. Cost: minutes.
- Expected delta: +0.000–0.002 LB, concentrated in France. Literature-backed direction; magnitude speculative. Our own evidence is mixed: synth measured +0.0039±0.0020 for "rule beats GFM" on one test (`10-frontier` §1.4), but ≈0 (−0.00001 ± 0.0021) over 6 seeds (`10-frontier` P2). That is why the choice must be made per country on LOCO rather than by assumption.

**A2. Expected-F-optimal one-to-one linkage via k-constrained assignment.** [LIT, low transfer]
- Source: Bai, Binette, Reiter, *Optimal F-score Clustering/Matching for Bipartite Record Linkage*, Statistics & Computing 2025, doi:10.1007/s11222-025-10701-y, arXiv:2311.13923.
- Method: for each k, solve an LSAP on an augmented matrix whose dummy rows have weight M = k(1+β²)/β². This forces exactly k links. Then pick the k maximising plug-in E[F_β]. It supports β≠1 and pairwise probabilities directly.
- Their numbers (F vs Sadinle's Bayes estimator, BRL):
  - sim low-error 0.91 vs 0.89; sim moderate error 0.45 vs **0.0**; RLdata500 0.98 vs 0.98; Union Army 0.57 vs 0.23.
  - The big wins are over a Bayes estimator that abstains to empty. They are **not** over a tuned threshold, and there is no Hungarian or threshold baseline.
- Transfer:
  - It optimises one global (micro) F, but we score macro-F0.5 per S1.
  - Our exclusivity is many-to-one (record → ≤1 S1), so the per-record argmax already solves the assignment exactly when S1 capacity is unbounded.
  - The only new ingredient would be S1-side capacity: GT caps are n2≤5 and n3≤6 (`10-frontier` §1.1).
- Expected delta: ≈0 (+0.0000–0.0003). **Skip**, except as a cheap sanity cap: if an S1 is predicted >5 S2 or >6 S3 records, drop the lowest-p extras.

**A3. Which bipartite clustering wins for clean-clean ER.** [LIT, confirms current design]
- Source: Papadakis, Efthymiou, Thanos, Hassanzadeh, EDBT 2022, arXiv:2112.14030 (full text read). The extended version is VLDB J 2023, doi:10.1007/s00778-023-00791-3.
- Their numbers: macro-average over 739 similarity graphs (Table 4), F1:

  | Algorithm | F1 |
  |---|---|
  | KRC | 0.619 |
  | UMC | 0.618 |
  | EXC | 0.591 |
  | BMC | 0.586 |
  | RCA | 0.518 |
  | RSR | 0.499 |
  | CNC | 0.490 |
  | BAH | 0.408 |

  - UMC (greedy: sort edges and accept if both ends are free) is the most balanced (P−R gap 0.017). It gives the best F1/runtime trade-off (F1 0.781 at 4 ms on schema-based inputs).
  - UMC over plain TF-IDF cosine reached F1 0.99 on D4, equal to DITTO's 0.99. It was 2% below DITTO on D5 and 21% below on D3 (Table 7).
- Transfer: our record-side argmax plus threshold *is* UMC with the S1 side uncapacitated. The literature says nothing fancier (Hungarian-style max-weight, i.e. RCA/BAH) beats it, and max-weight methods do worse (0.518/0.408). **Keep what we have; do not spend time on global assignment solvers.**

**A4. Cardinality-aware exact expected-F (EFP) vs the independence plug-in (LFP).** [LIT]
- Source: Dembczynski, Jachnik, Kotłowski, Waegeman, Hüllermeier, *Optimizing the F-Measure in Multi-Label Classification: Plug-in Rule Approach versus Structured Loss Minimization*, ICML 2013, PMLR v28 (full text read). Theory: Dembczynski et al., *An Exact Algorithm for F-Measure Maximization* (GFM), NeurIPS 2011.
- Method:
  - GFM is exact given the m² quantities P(y_i=1, |y|=s) plus P(y=0).
  - **EFP** estimates them directly with one multinomial model per label over the count s, so it knows the set-size distribution.
  - **LFP** assumes label independence (Poisson-binomial). This is what a "plug-in expected-F from per-pair probabilities" decision does.
- Their numbers (F1 %, EFP vs LFP vs BR@0.5):

  | Dataset | EFP | LFP | BR@0.5 |
  |---|---|---|---|
  | Image | **59.77** | 58.86 | 43.63 |
  | Scene | 74.44 | 74.38 | 55.73 |
  | Yeast | 65.47 | 65.02 | 60.59 |
  | Medical | 80.39 | **81.27** | 70.19 |
  | Enron | **61.04** | 56.86 | 55.49 |
  | Mediamill | 55.16 | 55.15 | 51.21 |

  - EFP wins 5/6. The margin over independence is −0.9 … +4.2 F1 points and is largest where labels co-occur (Enron).
- Transfer:
  - Our per-S1 sets are exactly this object: 1–11 labels, a strongly structured count, and a **known, country-independent size prior** (0: 5.58%, 1: 5.40%, 2: 17.0%, 3: 24.1%, 4: 21.9%, 5: 14.6%, 6: 7.5% …; `10-frontier` §1.1).
  - Candidates of one S1 are *not* independent: near-dup copies of one record co-move, and sibling-S1 competition too.
  - Cheapest EFP-flavoured fix: tilt the Poisson-binomial count distribution toward the empirical **within-candidate** size prior, measured on OOF because blocking truncates sets. Use P(y) ∝ Π Bern(p_i) · π(|y|)/PB(|y|), then evaluate E[F0.5] exactly for each top-k prefix by DP over counts. n ≤ ~15 per S1, so this is trivial on CPU.
  - Because π is country-independent, **it transfers to France**, where the p_i are least trustworthy.
- **Code check:** `metrics.best_set_fbeta` is the exact E[F0.5] under **independence** (Poisson-binomial over in-candidate labels, brute-force-checked), i.e. LFP.
  - It treats "all candidates negative" as "truth empty": the empty-set value is Π(1−p). It ignores the S1's true records that blocking never retrieved, so it **over-values predicting empty** whenever Σp is small but the S1 is actually a non-singleton.
  - The count prior (only 5.58% of S1s are truly empty) is precisely the correction for that. Each wrong empty-vs-nonempty call costs a full 1.0 of that S1's F.
- Cost: ~1 h code, seconds to run.
- Expected delta: +0.000–0.002 (seen countries), with possibly more on France if its p_i are systematically over- or under-confident, since the count prior acts as a label-free anchor. Literature-backed direction; magnitude speculative. **Test on OOF + LOCO before shipping. A1 warns that DTA-style rules lose when p is misspecified.**

### 1.B Competition evidence: Kaggle Foursquare Location Matching (2022)

This is the closest public analog. The metric there was mean per-id **IoU of the predicted match set** (a per-entity set metric like ours), with name/address noise and a retrieve → GBDT → post-process pipeline. All numbers are from the writeups (kaggle.com/competitions/foursquare-location-matching/writeups/...).

**B1. Train the final model on all labelled data, more rounds, 5-model blend.** [LIT, strong]
- 9th place (taksai) LB ladder:

  | Step | LB |
  |---|---|
  | CatBoost | 0.925 |
  | + post-process | 0.930 (+0.005) |
  | + LightGBM blend | 0.933 (+0.003) |
  | **+ train on all data in 1 fold** | **0.943 (+0.010)** |
  | + 5-model ensemble | 0.945 (+0.002) |
  | + more iterations ("overfit") | 0.948 (+0.003) |

- 8th place: public LB **0.908 → 0.928 → 0.948** as training ids went 550k → 700k → 1.1M ("the larger the number of ids … the larger the stopping iteration … the better the score").
- Transfer:
  - If any v10 stage-2 model is trained on a subsample (OOF sampling was 120k S1s in the v4 audit), or on k−1 folds without a full refit, the refit is the largest single cheap lever in that competition.
  - Seed/fold ensembling was worth only +0.002 there, which is consistent with our "two seed families tonight" expecting a small gain.
- **Code check (26 Sep, read-only):** `code/ber_v5/src/run_big.py` `CFG["train_per_country"]=60_000`. Run logs show 60k (20 hits) and 100k (2 hits). Train GT has 2,206,821 S1s, so ~1.79M are visible after hide_frac 0.19, i.e. several hundred thousand per country. **The matcher therefore sees only ~5–14% of the available labelled S1s.** The final fit is `model.fit_full` on that sample. This is the lever the Foursquare ladder says matters most (it is item R3 in `10-frontier` §3, which had no magnitude).
- Counter-evidence: WDC Products (Peeters, Der, Bizer, arXiv:2301.09521, Table 3) shows the **Magellan random forest on similarity features is nearly flat** in training size (50% corner cases, seen: 31.38 → 35.83 → 35.41 F1 for small/medium/large). Only the deep matchers gain strongly. The authors add that more data mostly improved *precision* on "negative corner-cases". Our FP-heavy residual (65% decoys, `10-frontier` §1.3) is exactly that class.
- Net: the GBDT-with-rich-features case (Foursquare 8th/9th) is closer to ours than Magellan on product titles. Still, **gate it on a 60k→120k→240k learning curve** on the same held-out S1s before paying for 400k+.
- Cost: one full refit, rounds = 1.1 × best_iter (the Ayan recipe in `08`). Features for 4× the S1s are the expensive part: roughly 4× the train-phase feature time.
- Expected delta: **+0.001–0.004** at 60k → ~250k per country. Literature-backed direction (two Foursquare teams, +0.010 and +0.040 in a lower-score regime); magnitude speculative. The WDC flat curve is the downside case (≈0).

**B2. Metric-aware sample weights.** [SPEC]
- 7th place (in-tokyo) weighted each pair by the IoU loss its misprediction would cause, given all other pairs of that POI are right. The weights averaged 0.8 for positives and 1.0 for negatives. **No ablation number was published.**
- Transfer to F0.5 per S1 (everything else correct):
  - One extra FP on an S1 with n true records costs ΔF = 1 − 1.25n/(1.25n+1). That is 0.444 at n=1, 0.286 at n=2 and 0.138 at n=5.
  - One missed record costs ΔF = 1 − 1.25(n−1)/(1.25n−1). That is **1.0 at n=1** (the prediction becomes empty), 0.167 at n=2 and 0.048 at n=5.
  - A FP on a *singleton* S1 costs the full 1.0 (F goes from 1 to 0).
  - So the pair weights should depend strongly on the S1's expected size and on whether the pair is the S1's *only* candidate. Weighting pairs by these marginal costs aligns the logloss with the metric.
- Cost: one retrain.
- Expected delta: +0.000–0.002, speculative.

**B3. Graph post-processing.** [LIT, mostly already in v10]
- 15th place: a group-average-linkage merge gave ≈**+0.025** IoU over no post-processing, and using raw logits rather than probabilities in the merge gave +0.002.
- 7th place: candidate-set max-IoU was 0.9778 from retrieval alone and **0.9935** after graph post-processing (union-find over edges above threshold, drop edges by betweenness centrality, keep vertices within 2 hops).
- 9th place: post-processing was +0.005 LB.
- Transfer: Foursquare is symmetric dedupe, so transitivity is free signal there. Ours is S1-anchored, and v10's sibling-graph expansion plus anchor re-retrieval already harvest the "record ↔ record" transitivity.
- The one untried variant: **betweenness/bridge-edge pruning** to cut decoy records that attach to a true cluster through one weak edge. Expected +0.000–0.001, speculative.

**B4. Multi-view asymmetric candidate union.** [LIT, already in v10]
- 7th place used 5 retrieval views with per-view k (4/12/4/8/4), reaching max-IoU 0.9778 at 32 candidates.
- 9th place used 400 candidates per id to reach max-IoU 0.994, then a light CatBoost prefilter to 3M pairs at max-IoU 0.987.
- The Ayan repo (`08` §1a) is the same idea and reports union recall 0.9905 at ≈45/S1.
- Nothing new for us beyond the per-view rank caps already specced in `08`.

**B5. Two-round hard-negative fine-tuned embedding for retrieval.** [LIT, GPU-bound]
- 8th place: SBERT with contrastive loss. Round 2 re-mined neighbours with the round-1 model, and ideal IoU (candidate recall) rose **0.97 → 0.986**.
- Transfer: this is a GPU technique. On CPU the analogue is a learned re-weighting of char-n-gram TF-IDF dimensions, i.e. a supervised blocker (see 1.C).
- Expected delta for us: speculative, and blocked by time.

**B6. Learned light-GBDT prefilter instead of a similarity-rank prune.** [LIT]
- Foursquare 9th place:
  - 400 candidates/id (kNN latlon 350 + name-TF-IDF 50) gave **max-IoU 0.994** on 420M candidates.
  - A light CatBoost with 40 cheap TF-IDF/diff features, cut at p>0.005, kept **3M candidates at max-IoU 0.987**: 140× fewer pairs for −0.007 oracle.
  - 1st place (2:30) and re:waiwai both used the same "LightGBM with limited features to reduce candidates" stage; re:waiwai kept top-40 per id for stage 2.
- Peer-reviewed analog: *Generalized Supervised Meta-blocking*, Gagliardelli, Papadakis et al., PVLDB 2022, doi:10.14778/3538598.3538611 (arXiv:2204.08801, read).
  - A probabilistic classifier over cheap per-pair co-occurrence features, followed by top-weighted pruning.
  - BLAST vs CNP: "recall is lower by −4.1%, while its precision and F1 are higher by 34.9% and by 33.6%". On large datasets it "reduces recall by 3.5%, but consistently maintains it above 0.93, while increasing precision and F1 by a whole order of magnitude".
- Transfer:
  - `10-frontier` §1.3 measured that the v4 similarity prune dropped **~2% of positives** after blocking. A prefilter scored by a 1st-stage LGBM with a *recall-targeted* cut (top-M by p1 per S1 **and** per record) usually loses far less than a single-score rank prune at the same budget.
  - It needs only features that already exist at blocking time: per-view cosines, per-view ranks, rev-rank and CSLS (E1).
  - **Code check:** v10's last candidate step is `run_big.prune_by_similarity`, documented as "a fixed rule (no learned model)". It scores max(name token-set, no-space ratio, phonetic token-set) + address token-set + 50×key-cosine, then keeps top-28 per S1 ∪ rev_rank≤2. So B6 applies as-is. (The older `run.py` path *had* a learned stage-1 pruner; v10 does not use it.)
- Cost: one small LGBM (~20 features) plus re-running stage-2 on the new candidate set. ~2–4 h.
- Expected delta, back-of-envelope:
  - Recovering ¼–½ of a 2% positive loss returns 0.5–1% of true records, i.e. ~0.017–0.035 records per S1.
  - The matcher accepts perhaps 75% of those.
  - Each accepted record is worth ~0.10 F on average: an FN costs 1.0 at n=1, 0.17 at n=2 and ~0.07 at n=3–4 (B2).
  - Net: **+0.001–0.003**.
  - Literature-backed mechanism; magnitude speculative; **conditional on measuring v10's actual loss at `prune_by_similarity`**, which takes one OOF query.

**B7. Kaggle Shopee Price Match Guarantee (2021): mean per-item F1 of match sets, the same metric family.** [LIT]
- **3rd place (btbpanda):** a CatBoost pair model, then agglomerative clustering over GBM probabilities, merging until the **average cluster size hits a target** (2.8 on train; the best LB used 2.6). "This algorithm boost me to 0.79" from 0.781 (**+0.009**). A size-prior-anchored decision gave the largest single post-processing gain.
- **2nd place (lyakaap/tkm):**
  - Features "Avg and std of top-K cosine similarities of each item, K=5,10,15,30 … normalized … to handle the difference between train and test" are local-density/hubness features (supports E1).
  - Query expansion +0.001; recursive highest-betweenness edge removal "about 0.001"; sample weights 1/(group size)^0.4 (no isolated number).
  - Per-model threshold-subtracted score sum across the ensemble.
  - The ensemble itself was the big step: 0.767 / 0.765 → 0.777 after team merge.
- Transfer:
  - The 3rd-place trick is the competition-grade version of A4 and of our own France "mean links per S1" instrument (`10-frontier` §1.1).
  - For **France** (no labels), pick its threshold/shift so that predicted links per S1 and the predicted-empty rate hit their label-free targets: mean ≈ 3.46 × (France blocking recall proxy), and empty ≈ 5.58% + excess. Do not tune it on LOCO F alone.
  - Our synth says France shifts are worth ≤ ±0.0005 (`10-frontier` P3). Shopee's +0.009 came from an uncalibrated embedding-threshold regime, so expect much less here.
- Expected delta: +0.000–0.002 on France. Speculative.

### 1.C EnsembleLink (2026): pulled in full, **discard for the main pipeline**
- Source: Dasanaike, arXiv:2601.21138 v2 (Mar 2026), full text read.
- What it actually is: **not** a model ensemble. It is a *retrieval union*: Qwen3-Embedding-0.6B dense top-30 ∪ char 2–4-gram TF-IDF top-30, giving ≈52 candidates. Then a zero-shot Jina Reranker v2 Multilingual (278M) cross-encoder, then **top-1**. There is no set decision: every query gets exactly one match.
- Numbers (top-1 accuracy, 40% test split):
  - EnsembleLink vs fastLink vs fuzzylink: city 0.901 / 0.713 / 0.717; candidate-voter 0.990 / 0.230 / 0.965; organization 0.961 / 0.282 / 0.922; multilingual parties 0.843 (0.926 with fuzzy country blocking).
  - DBLP-Scholar pair-F1 at the best τ=0.8 is 89.0, against Ditto 94.3 and DeepMatcher 94.7, which are supervised.
  - Fine-tuning with 25–300 labels gave "no consistent improvement".
  - **CPU throughput: Jina v2 = 25 pairs/s; MiniLM-L6 = 478 pairs/s.**
- Transfer:
  - Our ~40–100M candidate pairs would take ~20–50 days on CPU with Jina, and ~1–2.5 days with MiniLM (which is English-only).
  - It is zero-shot and supervised pipelines beat it; our in-domain LGBM is far stronger on synthetic leet/typo noise that no pretrained reranker was trained on.
  - The only conceivable use is a France-only feature on a thin uncertain band (≤200k pairs ≈ 2.2 h with Jina on one core-pool), and expected value is ≈0.
- The "dense ∪ sparse union" retrieval idea is already captured by our multi-view union.



### 1.D Unseen-country transfer (France) — domain adaptation and calibration under shift

**D1. Domain adaptation for ER: gains are proportional to the source-only gap, and ≈0 when source-only is already good.** [LIT]
- Source: Tu, Fan, Tang, Li et al., *Domain Adaptation for Deep Entity Resolution* (DADER), SIGMOD 2022, doi:10.1145/3514221.3517870 (full text read).
- Their numbers, F1 on target (NoDA → best DA):

  | Source → Target | NoDA | Best DA | Gain |
  |---|---|---|---|
  | DBLP-Scholar → DBLP-ACM | 97.2 | 97.2 | **+0.0** |
  | DBLP-ACM → DBLP-Scholar | 77.8 | 92.3 | +14.5 |
  | Walmart-Amazon → Abt-Buy | 65.8 | 72.6 | +6.8 |
  | Zomato-Yelp → Fodors-Zagats | 85.4 | 94.5 | +9.1 |

  - WDC same-site categories: −1.5 … +8.3.
  - Discrepancy-based (MMD) is the stable winner. Adversarial InvGAN often *hurts* (e.g. 29.7 vs NoDA 47.6).
  - Verbatim, "the improvement of DA is not always significant … the models originally trained on source already predict target well."
- Transfer:
  - DADER aligns *deep* feature extractors. Our matcher is GBDT on similarity features, which are already largely domain-invariant: cosines, fuzz ratios and IDF computed per-country.
  - The analog for GBDT is **feature-level invariance**: drop or neutralise features whose *distribution* shifts most between IN/US and FR (domain-classifier AUC per feature, i.e. adversarial validation), then retrain and check LOCO.
  - Our LOCO (0.915) vs in-domain (0.962) gap is 0.047, the "NoDA is mediocre" regime where the paper saw +7–15 F1. But their shifts were *schema/attribute* shifts, and ours is a *language/vocabulary* shift inside the same schema. That is closer to WDC categories (gain −1.5…+8.3).
- Implementation: per feature, train a 1-feature classifier IN∪US-pairs vs FR-pairs on candidate pairs (no labels needed) and rank by AUC. For the top-shift features, either remove them or replace them with per-country rank/quantile-normalised versions. Validate by LOCO(IN→US, US→IN). Cost: ~1–2 h.
- Expected delta: +0.000–0.003 LB (France share 15%). Speculative magnitude; direction literature-backed.

**D2. Importance-weighted calibration under covariate shift is unstable. Do not rely on it alone.** [LIT, negative]
- Source: Park, Bastani, Weimer, Lee, *Calibrated Prediction with Covariate Shift via Unsupervised Domain Adaptation*, AISTATS 2020, arXiv:2003.00343 (full text read).
- Their numbers, target ECE, Temp-scaling vs **IW+Temp**:

  | Shift | Temp | IW+Temp | Result |
  |---|---|---|---|
  | U→M | 36.89% | 7.68% | big win |
  | M→U | 12.16% | 15.92% | worse |
  | M→S | 10.28% | 25.47% ± 8.98 | much worse |
  | A→W | 26.69% | 21.82% | better |

  - Their best, feature-learning + IW (FL+IW+Temp), won only some shifts, with std up to ±6.5 pts.
- Transfer: re-fitting France's isotonic map on source pairs importance-weighted by a domain classifier p(FR|x)/p(src|x) is cheap. The literature says it is **as likely to hurt as help** without a label-free check.
- Only do it if LOCO (IN as pseudo-target) confirms. Expected delta: 0 ± 0.002. **Low priority.**

### 1.E Retrieval / blocking

**E1. Hubness reduction (CSLS / NICDM) on the candidate lists.** [LIT]
- Sources:
  - Conneau, Lample, Ranzato, Denoyer, Jégou, *Word Translation Without Parallel Data*, ICLR 2018, arXiv:1710.04087 (read).
  - Obraczka & Rahm, *Fast Hubness-Reduced Nearest Neighbor Search for Entity Alignment in Knowledge Graphs*, SN Computer Science 2022, doi:10.1007/s42979-022-01417-1, with library `kiez`.
- Their numbers:
  - CSLS vs plain NN, P@1, Procrustes: en→es 77.4 → **81.4**, en→fr 74.9 → **81.1**, fr→en 76.1 → **82.4**. Sentence retrieval P@1 42.6 → **66.1**.
  - Entity alignment: NICDM/CSLS/DSL "significantly outperform using no hubness reduction", with a median **+3.99% hits@50** (exact kNN) at "practically no cost". Mutual Proximity was no better than none.
- Transfer:
  - Formula: CSLS(q,t) = 2·cos(q,t) − r_q − r_t, where r_x is the mean cosine of x to its top-K neighbours on the other side.
  - Our top-k lists are asymmetric by construction: generic short names (e.g. "royal enterprises", "sri sai …") are hubs that occupy many S1s' top-k, and rare S1s get crowded out. Reverse top-k is the partial fix already in v10.
  - CSLS is the cheap principled fix. Our forward (S1→record) and reverse (record→S1) top-k lists already contain everything needed for r_q and r_t, so no extra kNN pass is required.
  - Use it (a) as the ranking score for the prune/prefilter (B6), and (b) as a feature: `csls_combo`, and `r_q`, `r_t` themselves.
  - Radovanović, Nanopoulos, Ivanović, *Hubs in Space*, JMLR 2010, shows hubness is a general high-(intrinsic)-dimensionality effect, not an embedding artefact. Still, our dense-embedding-free TF-IDF space should be *less* hub-prone than KG embeddings. The Ayan-style `qgap`/`tgap`/`tmargin` competition features already let the matcher see hubness, but they cannot rescue positives already dropped at prune.
- Cost: ~30 min on existing candidate frames.
- Expected delta: +0.000–0.002, via prune recall. Speculative magnitude; mechanism literature-backed.

**E2. Normalisation/canonicalisation: add it as *extra views and features*, never as a replacement.** [LIT, cautionary]
- Source: Randall, Ferrante, Boyd, Semmens, *The effect of data cleaning on record linkage quality*, BMC Med Inform Decis Mak 2013, doi:10.1186/1472-6947-13-64 (read).
- Verbatim: "Data cleaning made little difference to the overall linkage quality, with heavy cleaning leading to a decrease in quality … although correct records were now more likely to match, incorrect records were also more likely to match, and these incorrect matches outweighed the correct matches."
- Transfer:
  - Our noise is rule-generated (leet 0/1/5/8, legal-suffix add/drop, Indic transliteration, #handles/domains). Inverting it (`0→o, 1→i|l, 5→s, 8→b`; strip legal suffixes; phonetic skeleton; split handles/domains) raises agreement for true pairs.
  - The Randall result says it *also* raises collisions, and our **near-duplicate sibling S1s** and **neighbour-house decoys** are exactly the collision class that loses.
  - So: blocking view = union(raw, canonical), and features = both `sim_raw` and `sim_canon` plus `canon_equal`. Let the GBDT learn when canonical agreement is trustworthy. Do not overwrite the raw text.
- Literature gives no magnitude for leet/transliteration inversion on business names, and I found none; everything found was spam/hate-speech deobfuscation with classification metrics, not linkage.
- Measure it directly: on train GT, the fraction of true (record, S1) pairs *not in the current candidate set* whose canonical names are equal. That fraction × the blocking-loss share is the ceiling. Speculative until measured.

**E3. Sparkly (PVLDB 2023): BM25 over 3-grams, probing from the larger table.** [LIT, mostly already done]
- Source: Paulsen, Govind, Doan, *Sparkly: A Simple yet Surprisingly Strong TF/IDF Blocker for Entity Matching*, PVLDB 16(6) 2023, doi:10.14778/3583140.3583163 (full text read).
- Their findings:
  - Top-k BM25 over character 3-grams "outperforms 8 state-of-the-art blockers", with recall 86.57%–99.96% across datasets.
  - Verbatim: "probing from the larger table rather than the smaller one tends to produce higher recall, given the same k value". Probing *both* directions and taking the union "can significantly increase runtime yet improve recall only minimally".
  - Dropping IDF hurts a lot ("TFIDF-cosine greatly outperforms TFIDF-cosine-no-idf"); dropping TF does not.
  - kNN-cosine over 3-grams ≈ 5-grams.
- Transfer: our larger side is the 10M records, so the literature's preferred direction is **record → S1 top-k**. That is the "reverse top-k" already in v10 (Ayan measured rev R@1 0.972 vs forward combo R@40 0.975, `08` §1a).
- The only untested item is **BM25 instead of sublinear-TF cosine** for that reverse probe. BM25's length normalisation (b≈0.75) matters for short names with appended legal suffixes and #handles.
- Expected delta: 0 to +0.001 blocking recall. Speculative. Cost: ~1 h via a sparse BM25 weighting of the existing char-3gram matrices.

### 1.F Ensembling tonight's two seed families

**F1. Calibrate *after* combining, not before.** [LIT, directly actionable tonight]
- Source: Wu & Gales, *Should Ensemble Members Be Calibrated?*, arXiv:2101.05397 (2021), full text read.
- Verbatim conclusion: "calibrating members of the ensemble is not sufficient to ensure that the ensemble prediction is itself calibrated. The resulting ensemble predictions will be under-confident, requiring calibration functions to be optimised for the ensemble prediction."
- Their numbers, CIFAR-100 ECE ×10⁻²:

  | Ensemble | Uncalibrated | Pre-combination (members calibrated) | Post-combination |
  |---|---|---|---|
  | LEN | 11.55 | 3.24 | **2.19** |
  | RSN | 3.22 | 1.88 | 1.82 |
  | DSN100 | 2.54 | 2.08 | 1.83 |

- Transfer:
  - v10's decision is a plug-in expected-F0.5 driven by *calibrated* p. Averaging two per-country-isotonic outputs gives a flatter, under-confident p. The E[F]-optimal set then shrinks toward high-precision/low-recall, and the singleton/empty decision shifts.
  - Do instead: average the **raw margins/logits** of the two families (or rank-average), then refit per-country isotonic **on the averaged OOF**.
  - For France (no labels), fit the LOCO-derived map on the averaged LOCO scores, not on members.
- Evidence the ensemble itself is worth little: Foursquare 9th-place 5-model ensemble +0.002 LB (B1); 15th place: raw logits instead of probabilities in post-processing +0.002.
- Cost: minutes.
- Expected delta: +0.000–0.001. It mainly *prevents* losing part of the ensemble gain.

## 2. Discarded (and why)

| Technique | Source (read) | Why discarded for *this* problem |
|---|---|---|
| EnsembleLink zero-shot retrieve + cross-encoder | arXiv:2601.21138 | Top-1 only (no set decision). CPU 25 pairs/s (Jina v2), so our tens of millions of pairs take days. Zero-shot pretrained rerankers never saw leet/rule noise, and supervised in-domain GBDT beats them. |
| Ditto / DeepMatcher / HierGAT / R-SupCon deep matchers | Li et al. PVLDB 2020; WDC Products arXiv:2301.09521 | They do beat symbolic matchers on textual product data by 5–16 F1 (WDC). But fine-tuning plus inference over ~10⁷–10⁸ pairs is GPU-bound. Our frontier doc already measured MiniLM-L6 CPU at 600–980 pairs/s (~17 h for 60M pairs, before fine-tuning). |
| Global max-weight bipartite assignment (Hungarian, RCA, BAH) | Papadakis et al. EDBT 2022 | Lost to greedy UMC/KRC (F1 0.518/0.408 vs 0.618/0.619). Our many-to-one exclusivity is already solved exactly by record-side argmax. |
| Bai–Binette–Reiter F-optimal LSAP estimator | Stat. Comput. 2025 | Global micro-F, one-to-one. Its wins are over an abstaining Bayes estimator, not over tuned thresholds. |
| Deep domain adaptation (MMD/GRL/InvGAN feature alignment) | DADER, SIGMOD 2022 | Needs a deep feature extractor. InvGAN often hurts. Only the *idea* survives, as feature-invariance selection (D1). |
| Importance-weighted calibration as a standalone France fix | Park et al. AISTATS 2020 | ECE worsened on 2 of 4 shifts (M→S 10.28 → 25.47%). Only with a LOCO check (D2). |
| Supervised meta-blocking as block restructuring (BLAST/RCNP/GSM) | PVLDB 2022, Inf. Syst. 2023 | Built for schema-agnostic token *blocks*. We do top-k retrieval. Its transferable core, a learned pair-level prefilter, is kept as B6. |
| Heavy data cleaning / replacing raw text with canonical text | Randall et al. BMC MIDM 2013 | "Heavy cleaning leading to a decrease in quality". Kept only as additional views/features (E2). |
| GNN node-classification post-processing (Foursquare) | Foursquare blog 2022 | 0.907 → 0.946 is a whole-framework number with no isolated GNN gain. Our S1-anchored graph is shallow, and the sibling/enrichment passes cover it. |
| Leetspeak/algospeak deobfuscation with neural models | IJIMAI 2023, WOAH 2024 (Consensus hits) | Spam/toxicity classification metrics, not linkage. Our leet map is a fixed 4-digit substitution: deterministic inversion suffices, and neural deobfuscation adds nothing. |
| LLM matchers (GPT-4, Qwen3-8B reranker) | Peeters et al. 2024 via EnsembleLink Table 6 | Zero-shot DBLP-Scholar F1 89.8 < supervised 94.7. No API/GPU budget at this scale. |
| Fellegi–Sunter / fastLink EM as France calibrator | EnsembleLink Table 1 (fastLink 0.23–0.71 top-1) | Far weaker than our transferred GBDT. Our label-free France instruments (mean links, excess-empty) are already better anchored (B7/A4). |

## 3. Query log (for audit)

- **Consensus API** (26 queries before the hard cap, "You've used all 30 searches this month; resets on October 1st"):
  - blocking recall/char n-gram; Sparkly; supervised meta-blocking; Ditto/PLM matching; GBDT vs deep ER; GFM/plug-in F-measure; one-to-one/UMC; EnsembleLink.
  - leetspeak normalisation; Indian-name transliteration; legal-suffix company names; address/house numbers; self-training ER; ER domain adaptation; cross-lingual zero-shot names; collective ER.
  - record-linkage calibration; expected-F set prediction; LightGBM seed stacking; POI conflation; Foursquare matching; hard negatives; string similarity for names; GeCo corruption generators.
  - Raw JSON: `scratchpad/sweep12/raw_q*.json`.
- **OpenAlex** (fallback): hubness/CSLS; data cleaning and linkage quality; legal-form matching; *Hubs in Space*.
- **Full text read (pdftotext/fetch):**
  - arXiv:2601.21138 (EnsembleLink); arXiv:2112.14030 (bipartite matching); arXiv:2311.13923 (F-optimal linkage); arXiv:1206.4625 (EUM vs DTA); PMLR v28 Dembczynski 2013 (EFP/LFP).
  - DADER SIGMOD 2022; arXiv:2003.00343 (IW calibration); arXiv:1710.04087 (CSLS); Obraczka & Rahm 2022 (kiez).
  - arXiv:2204.08801 (GSM); arXiv:2207.04802 (PromptEM); arXiv:2101.05397 (ensemble calibration); Sparkly PVLDB 2023; Randall 2013 (PMC3688507); WDC Products arXiv:2301.09521.
- **Competition writeups:**
  - Foursquare Location Matching: 7th, 8th, 9th, 15th places and the official winners blog. The 1st-place Kaggle writeup URL 404s.
  - Shopee Price Match: 2nd and 3rd places.
- **Loop 2 (gap re-queries):**
  - Training-size magnitude → WDC Products learning curves (flat for RF) vs Foursquare (steep).
  - Count-prior magnitude → Shopee 3rd place (+0.009 from a target mean cluster size).
  - Hubness in non-embedding spaces → Radovanović JMLR 2010.
  - Canonicalisation magnitude → none found. Measurement recipe given (E2).
  - Metric-aware weights → no isolated ablation anywhere (Foursquare 7th and Shopee 2nd both unquantified).

