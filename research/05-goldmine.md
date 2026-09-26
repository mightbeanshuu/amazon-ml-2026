# Amazon ML Challenge 2026: goldmine hunt (research pass 5)

Written 26 Sep 2026, ~17:00 IST (~31 h to the 27 Sep 23:59 IST close). Web research only; local reads: `research/03`, `research/04`, `student_resource/README.md`, `READ-NOTES.md`, and 100+ rows of `runs/real_v2/oof_examples.tsv` (24,747 error rows: 19,787 MISS / 4,960 FALSE). Builds on 03/04, does not repeat them.

Tags: **[V]** read in the cited source · **[U]** third-party claim, not reproduced · **[E]** our estimate.

---

## The 3 bets for v7, ranked

### Bet 1 — Full test-density "mock world" + decision-rule retune + record-side competition stage 2
**The freshest measured LB gain by any team.** Tony-AJ, **26 Sep 10:50 IST: public LB 0.955 → 0.961 (+0.006)** purely from retuning the decision rule on a test-shaped mock validation ([LEADERBOARD.md](https://github.com/Tony-AJ/business_entity_resolution/blob/main/LEADERBOARD.md), [TRACKER.md](https://github.com/Tony-AJ/business_entity_resolution/blob/main/TRACKER.md)) [U].

Their exact recipe [U]:
- **Mock construction**: per country, sample train to match the test's *pool size and pool-per-S1 ratio* ("US 3.82M pool, 5.76/S1; India 4.13M, 5.82/S1"), keep whole clusters together by id hash, then **drop query S1s by hash so their records become unowned decoys** → "40.2% of the pool unowned". This is our `hide_frac=0.19` idea taken to full scale with per-country ratios.
- **Rule retune on mock**: thresholds moved to **τ 0.725 / rel 0.7 / single 0.775** (a triple: absolute, relative-to-best, and singleton-specific threshold). Mock +0.0027 → **public +0.006** (the mock understates the LB gain).
- **Stage-2 XGBoost cross-fitted on the mock's fit entities: +0.0040 mock.** Ablation: "competition features +0.0018 … **pool_gap carries 72% of the gain**" — pool_gap = margin over the record's best other S1, i.e. the same `tmargin` that is Ayan's top-gain feature (04 §1.2).
- They steer uploads with a fitted LB model `public ≈ 1 − L_FN − 1.45·L_FP − 0.0072` — FP weighted 1.45×, matching F0.5.

Supporting: Ayan's P-dense — a model *trained* under test density gains +0.0014 and the best threshold shifts 0.7 → 0.8 ([RESEARCH_LOG](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/RESEARCH_LOG.md)) [U]; ICE's prior-corrected threshold cost only 0.0002 under ×2 decoys ([REPORT](https://github.com/jeetsidhu/AMAZON-ML-Challenege/blob/main/docs/REPORT.md)) [U].

**Spec (pseudo-code level):**
1. Build the mock per country: target pool/S1 = test ratio (5.76 US / 5.82 India / measure France); drop S1 by `hash(id) % k` until ratio met; keep their S2/S3 records as decoys.
2. Retrain LightGBM on pairs *from the mock world* (not only re-tune thresholds).
3. Grid the triple (τ_abs, τ_rel = p/p_best_other_S1_of_record, τ_single for 1-candidate S1s) on mock F0.5; expect the optimum near 0.72–0.78.
4. Ensure record-side competition features exist: `pool_gap` = p_best − p_second among S1s competing for the same record, rank, and margin (add to stage B if missing).

**Expected gain:** +0.004–0.008 LB (Tony-AJ measured +0.006 public; stage-2 competition +0.004 mock) [U]. **Effort:** 2–4 h. **Risk:** low — it is threshold tuning plus one retrain; upload A/B confirms.

### Bet 2 — Anchor-merge expansion + one round of test self-training (France-weighted)
Two independent lines of evidence say "use your own confident predictions to create new candidates and labels":
- **Kaggle Foursquare 1st place (same problem shape: name+address matching), stage 4**: "Compare the matches of two ids and **merge the matches if the common id exceeds 50%**, then **predict the newly created pairs** with the model" → **CV 0.911 → 0.9166 (+0.0056)** at threshold 0.02 for new pairs ([writeup](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution)) [V]. Their stage-1 max-IoU was 0.979 — same regime as our PC.
- **Foursquare 7th place**: graph post-processing (predict pairs within graph distance ≤ 2 of confident matches, prune low-betweenness edges) took retrieval ceiling **0.9778 → 0.9935 maxIoU** ([writeup](https://future-architect.github.io/articles/20220720a/)) [V].
- **Self-training on the unseen country, measured on THIS data**: Ayan got **+0.0022 (India→US) / −0.0017 (US→India)** — "it helps only when pseudo-label precision is ≥ 99%" ([RESEARCH_LOG](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/RESEARCH_LOG.md)) [U]. PromptEM's uncertainty-selection (keep least-uncertain, not most-confident) is the principled filter ([arXiv 2207.04802](https://arxiv.org/abs/2207.04802)) [V]. Organisers explicitly allow self-training and synthetic pairs [Q&A].
- Lineage: this is SERF/Swoosh "merge-then-match" — merging matched records "may generate additional record matches" (Whang et al., [SIGMOD'09 iterative blocking](https://dl.acm.org/doi/10.1145/1559845.1559870); [SERF project](http://infolab.stanford.edu/serf/)) [V, no modern measured gains beyond the Kaggle numbers above].

**Spec:**
1. Round-1 predict on test. Anchors = matches with p ≥ 0.97, fold-std ≤ 0.02, record's top-S1 by margin ≥ 0.5, house-number ∈ {equal, both_missing}.
2. **Enrich each S1 with its anchors' tokens** (union of normalised name tokens + address tokens; keep counts). Re-run the cheap blocker for *below-median-confidence S1s only*, querying with the enriched representation → new candidate pairs (cap: +2/S1). Score them with the existing model; accept only above the normal τ. (This inverts "record dropped the rare token the S1 needs": an anchor sibling often carries it — 04 §2.2 truncated-name 15.6%.)
3. One self-training round: pseudo-positives = anchors, pseudo-negatives = other top-5 S1s of anchored records (exclusivity-guaranteed hard negatives) + synthetic French positives from our own noise ops (03 §4.2 catalogue). French pairs weight 0.5. **Validate the whole recipe first on the LOCO simulation (train US → pseudo-label India); adopt only if India F0.5 rises** — Ayan's sign flip (US→India −0.0017) is the warning.
4. Candidate-file compliance: new pairs from step 2 go INTO `candidate_pairs.tsv` (they are fed to the model — the rule in 03 §read-first is satisfied as long as the file is the union actually scored).

**Expected gain:** +0.004–0.010 (Foursquare +0.0056 CV for expansion alone; self-training +0.002 if the LOCO gate passes; France is where the headroom is, ≈0.91 vs 0.94+) [E from U/V]. **Effort:** 4–6 h. **Risk:** medium — confirmation bias; mitigated by the ≥99%-precision gate and the LOCO dry run.

### Bet 3 — Operator-inverse feature pack, derived from our own OOF errors
Read from `oof_examples.tsv` (real_v2): the FN band and the p>0.99 FP band are dominated by five inversible operators (examples verbatim):
1. **Indic phonetic round-trip renders**: "dijital kanasaltin praibhet limited" (= digital consultant private ltd), "lotas prodakts elaelapi" (= lotus products **LLP**), "menej ment el el pi", "arc vodyaolaay" (vidyalaya), "prilate/privet" (private).
2. **Leet inside words**: "8uilder" (builder), "6mr" (gmr).
3. **Random-name true pairs at an identical address**: "mirahalo", "beloiri", "verakor", "arianexfaye" — model stuck at p≈0.70, missed.
4. **State endonyms/abbreviations**: "pascimavang" (W Bengal), "dilli", "panjab", mh/gj/ka/hr/wb.
5. **Near-copy decoys** at the same address = the FP wall: "satara marketing pvt ltd" vs "satara pvt ltd center" (p 0.9997, FALSE) — one rare token dropped + one generic token added.

Published inverses with numbers:
- **Phonetic key blocking/features**: Arshbir1 measured on THIS data: metaphone(name)+address blocker **+0.0095 recall for +1.1 cands/S1** at 3.04M encodings/s ([results.md](https://github.com/Arshbir1/Amazon_ML_Challenge/blob/main/student_resource/business_entity_resolution/experiments/results.md)) [U]. Double Metaphone generalises Soundex with primary+secondary codes ([overview](https://www.babelstreet.com/blog/fuzzy-name-matching-techniques)); Indian-language Soundex variants exist but add little over metaphone-on-romanised [V, no strong numbers]. Add: metaphone-key blocker leg + per-token metaphone-equality feature ("praibhet"→PRBT ≈ "private"→PRFT is imperfect; ALSO do the learned map below).
- **Learned substitution channel from aligned train pairs**: Bilenko & Mooney's learnable edit distance (EM-estimated per-character costs, affine gaps) "improves duplicate detection accuracy over traditional techniques" on name/address fields ([KDD'03](https://www.cs.utexas.edu/~ml/papers/marlin-kdd-03.pdf)) [V, no single headline number]; Brill–Moore substring channel is the classic ([ACL'00](https://doi.org/10.3115/1075218.1075255)). **Cheap version for us [E]:** align true-pair token pairs (same position, ED ≤ 40% of length), mine frequent substring rewrites (ai→i, bh→v, ks→x, ee→i, digit-leet, "el el pi"→llp), apply as a canonicaliser before char-3-gram features — the same machinery as the Indic lexicon, but for *Latin* phonetic renders. Build per fold.
- **Splink-style TF-adjusted agreement (exact formulation)**: the TF adjustment is an additive match-weight term **log2(u_col / tf(value))** — an exact agreement on a rare value carries more evidence; for fuzzy agreements "use the greater of the two term frequencies"; floor `tf_minimum_u_value = 0.001`; damp with `tf_adjustment_weight` ([Splink TF docs](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html)) [V — the docs give no accuracy benchmark; fastLink likewise publishes no headline gain]. As LightGBM features: `−log df(rarest agreeing name token)` (max df of the two spellings when fuzzy), `−log df(full normalised name)` within country, `−log df(street+number key)`. This directly attacks operator 5: the dropped token "marketing" is rare → its absence should hurt; "center" added is common → cheap.
- **Added/dropped-token log-odds** (Epic021 G3, label-free, works on test France): France table learned "participations −1.99, holding −1.99, développement −1.35" without labels (04 §1.2) [U]; Siva's difference/margin features + dictionary moved **0.9707 → 0.9783** ([doc §5](https://github.com/Siva402-ai/amazon-ml-challenge-2026/blob/main/Documentation_template.md)) [U].
- **Random-name rescue rule** (operator 3): flag = all name tokens have zero df in the country's S1 vocabulary AND min-JW to candidate tokens is high (a generator word, not a typo); then let the address decide: exact street+number+PIN equality with `n_s1_same_street_number == 1` → feature that lets LightGBM push past 0.7. Karthikatstuffmama: random name × address collision is only ~0.11% of pairs — ceiling ~0.999 ([FINDINGS](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026/blob/main/FINDINGS.md)) [U].
- **State-endonym map**: learn `token → state` the same way as the Indic lexicon (aligned trailing admin tokens in true pairs): pascimavang→wb, dilli→dl, panjab→pb. Feeds the state-conflict filter in 03 §1.2 without a banned gazetteer.

**Expected gain:** +0.004–0.010 (metaphone leg +0.0095 recall ≈ +0.004 F0.5 at the peers' 0.4× conversion; TF + log-odds features attack both FN and the p>0.99 FP wall) [E]. **Effort:** 3–5 h total; the metaphone leg and TF features are ~1 h each. **Risk:** low–medium (each piece is independently measurable on OOF).

**Ordering under 31 h [E]:** Bet 1 first (cheapest, highest certainty, fixes the decision layer that Bets 2–3 feed into) → Bet 3's metaphone + TF + log-odds in one retrain → Bet 2's expansion + self-training last (needs a full round-1 test run to exist). Freeze ≥3 h before close (03's upload-hang warnings).

---

## 1. SIGMOD Programming Contests 2020–2022 (what the winners actually did)

- **Organisers' retrospective** ([SIGMOD Record 52(1), PDF](https://iris.unimore.it/retrieve/handle/11380/1310126/574065/09_Reports_DeAngelis.pdf)) [V]:
  - "Differently from many solutions in literature, **the best approaches were all optimized for the provided datasets**": pre-processing on few useful attributes, regex + hand-rule feature extraction (brand/model), alias dictionaries. Matching mostly **rule-based**; in 2021 "binary random forest classifiers … or **combining XGBoost with a rule-based matcher to solve the uncertainties of the latter**".
  - 2020 (cameras): **all top-5 teams reached F 0.99**; runtime broke the tie.
  - 2022 (blocking): finalists used sorted neighborhood, similarity joins, BERT sentence encoding, a distilled-transformer architecture, and supervised contrastive learning with extra WDC data — "followed by a **pair/block cleaning and ranking step based on intra-pair similarity**".
  - Takeaway for us [E]: the contest evidence backs *dataset-specific operator inversion + light ML* (our GBDT+rules) over generic neural matchers — consistent with Papadakis ICDE'24 (04 §2.1).
- **2020 winner** (Blacher/Klaus/Mitterreiter, Jena): "**mock labels** semi-automatically generated" + sorted integer sets; F 0.99 in 0.61 s on 30k records ([DI2KG'20 paper page](https://lacuna.tiptreesystems.com/work/fast-entity-resolution-with-mock-labels-and-sorted-integer-sets/wrk_2970dee00be37cd6f02c6b5995913bd0); [contest site](http://www.inf.uniroma3.it/db/sigmod2020contest/index.html)) [V title/abstract level]. Mock labels = labels derived from data regularities — the ancestor of our exclusivity-derived pseudo-negatives (Bet 2).
- **2021 winner**: SUSTech_DBGroup (Fu/Yin/Lu); finalist posters/code linked from the [leaderboard](https://dbgroup.ing.unimo.it/sigmod21contest/leaders.shtml) (bit.ly links; not fetched) [V page]. Second places: UKN (Tsinghua), panda (Georgia Tech, weak supervision lineage).
- **2022 winner WBSG** (Brinkmann & Peeters, Mannheim): embeddings from a transformer pre-trained with **supervised contrastive learning** on extra CommonCrawl/WDC product data, FAISS kNN, then **re-rank retrieved pairs with a symbolic similarity** ([announcement](https://www.uni-mannheim.de/en/news/wbsg-wins-sigmod-programming-contest-2022/); [contest](https://dbgroup.ing.unimore.it/sigmod22contest/)) [V]. GPU + external data → not adoptable here (external data is banned anyway); the transferable part is *dense-retrieve → symbolic re-rank*, which our pipeline already embodies. SABER's fine-tuned arctic-xs (04 §1.2) is the same idea and needed a GPU.
- **Verdict:** no unexploited goldmine in the SIGMOD contests for THIS task; they confirm the current architecture and Bet 3's philosophy. The 2022 task (recall-only blocking) rewarded exactly the hybrid char-3-gram + reverse-kNN retrieval already planned in 04 item 2.

## 2. Self-training / pseudo-labelling for ER under domain shift

- **Measured on this challenge** (strongest evidence): Ayan — self-training the unseen country gave **+0.0022 (India→US), −0.0017 (US→India)**; works only at pseudo-label precision ≥ 99% [U]. Tony-AJ's mock/stage-2 uses **cross-fitting** (predictions for feature-building come from a model not trained on that fold) — adopt for any stage that consumes round-1 probabilities [U].
- **PromptEM** ([arXiv 2207.04802](https://arxiv.org/abs/2207.04802)) [V]: select pseudo-labels by *lowest uncertainty* (MC-dropout std; our analogue: std across 5 fold models), not top confidence; keep a small fraction per round; one round suffices.
- **DADER** ([PVLDB 15(12)](https://www.vldb.org/pvldb/vol15/p3666-fan.pdf); [SIGMOD'22 paper](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/entity-sigmod-2022.pdf)) [V abstract]: feature alignment (MMD/adversarial) between source and target for PLM matchers — consistent big F1 gains on unlabeled targets, but needs a neural matcher; **not adoptable in 31 h on CPU**. Same for Sudowoodo/Rotom: their gains concentrate in the low-label regime (Rotom "+20 F1 with 3× fewer labels" vs weak baselines; Sudowoodo matches Rotom with 1/3 labels) ([Rotom](https://github.com/megagonlabs/rotom), [Sudowoodo arXiv 2207.04122](https://arxiv.org/pdf/2207.04122)) [V] — we have millions of labels, so the transfer is the *procedure* (augment + pseudo-label under shift), not the model.
- **Kaggle under train/test overlap** (Foursquare): the 1st place "merge train" is leakage-specific, but its structure — anchor test records to known entities, then **propagate labels through anchors and delete contradicted pairs** — is exactly our anchor/exclusivity propagation; LB 0.900 → 0.943 → 0.971 as each propagation rule was added [V]. No unseen-country Kaggle ER analogue with published self-training numbers was found beyond this.

## 3. Term-frequency / Fellegi–Sunter weighting

- **Exact Splink formulation** ([TF docs](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html)) [V]: TF adjustment is an additive match-weight term, independent of the EM-estimated m/u — conceptually `Δweight = log2(u_col / tf(value))`; fuzzy levels use **max(tf_left, tf_right)**; `tf_minimum_u_value=0.001` floors rare values; `tf_adjustment_weight` damps. The docs claim accuracy benefit but publish **no quantitative benchmark**; fastLink likewise. Mark: formulation [V], gain [unverified].
- Peer evidence on this data: Tony-AJ's v101 "name frequency" features (pool records sharing exact name/address) were worth only **+0.0004** [U] — the *entity-frequency* form is weak; the *token-rarity-of-agreement* form (Bet 3) is the one the near-copy-decoy FPs need, and our OOF FP examples show the model currently ignores rarity (drops "marketing", keeps p=0.9997).
- SoftTFIDF (Cohen et al. 2003) remains the best hybrid name similarity ([paper](https://pubs.dbs.uni-leipzig.de/dc/files/Cohen2003Acomparisonofstringdistance.pdf)) [V] — already in 03 item 7.

## 4. Merge/enrichment-based collective ER

- **Foursquare 1st place stage 4** (merge candidate sets >50% overlap → score only the new pairs): **+0.0056 CV** [V] — the only modern, measured "merge-then-match" number found. 7th place's 2-hop graph expansion: ceiling 0.9778 → 0.9935 [V].
- **SERF/Swoosh/D-Swoosh & iterative blocking** ([SIGMOD'09](https://dl.acm.org/doi/10.1145/1559845.1559870), [SERF](http://infolab.stanford.edu/serf/)) [V]: establishes the mechanism (merged records surface new matches) and the stopping rule (fixpoint / one round); papers report accuracy "may" improve + runtime effects, **no headline F gains** on modern data.
- Adaptation to our 1-to-many-with-owner topology (Bet 2): enrichment flows S1←anchors only; never record↔record transitive closure ("worst precision in every comparative study" — Ayan graph note [U]). Precision safeguards: only add candidates for low-confidence S1s, score with the unchanged matcher, normal τ.

## 5. Decision layer for macro-F0.5 one-to-many with exclusivity

- Nothing beyond what 03/04 already hold. The literature's exact per-entity expected-F machinery (GFM [NIPS'11](https://proceedings.neurips.cc/paper_files/paper/2011/file/71ad16ad2c4d81f348082ff6c4b20768-Paper.pdf); plug-in consistency [Waegeman JMLR'14](https://www.jmlr.org/papers/v15/waegeman14a.html); sparse-estimate variant [Jasinska ICML'16](http://proceedings.mlr.press/v48/jasinska16.pdf)) [V] assumes conditionally independent labels — copies violate it, and every team that measured expected-F on stage-1 scores lost to a plain threshold (Epic021 −0.0005, Ayan −0.0004, Arshbir1 −0.0013); it wins only on calibrated stage-2 (+0.0017, Arshbir1) [U].
- **Assignment solvers (LP/Hungarian/min-cost-flow): dead end confirmed.** The CCER bipartite study ([arXiv 2112.14030](https://arxiv.org/abs/2112.14030)) assumes one-to-one; on this data only 10 of 1.78M candidates were contested above threshold (Arshbir1 E8) [U]. No published macro-F-beta-optimal one-to-many-with-exclusivity method was found beyond greedy-argmax + per-entity set choice — the search came back empty on anything with measured gains.
- The actionable decision-layer move IS Bet 1's τ-triple retune (+0.006 public, measured 26 Sep) plus soft exclusivity p′=o/(1+Σo) (+0.0008, Epic021) [U].

## 6. Fresh chatter (26 Sep)

- **Tony-AJ is the live signal: public LB 0.954 (25 Sep 18:59) → 0.955 → 0.961 (26 Sep 10:50)**, local 0.9858, blocking recall 0.9906@~35; two submissions left; v107 (two-stage + expected-F on stage-2, tight rule) estimates public 0.9659 [U]. Their LB-model constant (−1.45·L_FP) is a useful upload-planning tool.
- **Epic021's repo returned 404 on 26 Sep** (was public on 25 Sep — 03/04 quote it). Teams are going dark before the deadline; 04's extracts are the last public snapshot. AyanAhmedKhan's repo now shows only 2 squashed commits (25 Sep) [V api].
- GitHub repo sweep (search API, pushed 25–26 Sep): ~18 new/updated repos, all early-stage or undocumented (PrateekTechie XGBoost pipeline, SiddeshKhandekar, Pratheesh-555 baseline-only README) — **no public claims above 0.961** in the last 24 h [V]. Reddit/X/Kaggle: dataset mirrors and prep guides only, no LB numbers [V searches].
- Implication [E]: known public evidence tops out at LB 0.961 vs the rumoured 0.979–0.987 top-50 band (04 §0) — the top teams are not publishing. Bets must come from the mechanism evidence above, not from copying a visible leader.

---

## Source index (new in this pass)
- [Tony-AJ LEADERBOARD.md](https://github.com/Tony-AJ/business_entity_resolution/blob/main/LEADERBOARD.md) · [TRACKER.md](https://github.com/Tony-AJ/business_entity_resolution/blob/main/TRACKER.md) — LB 0.961, mock protocol, τ triple, XGB stage 2 [U]
- [Foursquare 1st place](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution) · [2nd place (colum2131 part)](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/2-30-2nd-place-solution-colum2131-part) · [7th place](https://future-architect.github.io/articles/20220720a/) [V]
- [SIGMOD Record retrospective PDF](https://iris.unimore.it/retrieve/handle/11380/1310126/574065/09_Reports_DeAngelis.pdf) · [2021 leaderboard](https://dbgroup.ing.unimo.it/sigmod21contest/leaders.shtml) · [2022 contest](https://dbgroup.ing.unimore.it/sigmod22contest/) · [WBSG announcement](https://www.uni-mannheim.de/en/news/wbsg-wins-sigmod-programming-contest-2022/) · [2020 winner paper page](https://lacuna.tiptreesystems.com/work/fast-entity-resolution-with-mock-labels-and-sorted-integer-sets/wrk_2970dee00be37cd6f02c6b5995913bd0) [V]
- [Splink TF docs](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html) [V] · [Bilenko & Mooney KDD'03](https://www.cs.utexas.edu/~ml/papers/marlin-kdd-03.pdf) [V] · [Whang iterative blocking SIGMOD'09](https://dl.acm.org/doi/10.1145/1559845.1559870) · [SERF](http://infolab.stanford.edu/serf/) [V]
- [DADER PVLDB demo](https://www.vldb.org/pvldb/vol15/p3666-fan.pdf) · [DADER SIGMOD'22](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/entity-sigmod-2022.pdf) · [Sudowoodo](https://arxiv.org/pdf/2207.04122) · [Rotom](https://github.com/megagonlabs/rotom) [V abstracts]
- [Jasinska ICML'16 sparse F-measure](http://proceedings.mlr.press/v48/jasinska16.pdf) [V]
- Local: `runs/real_v2/oof_examples.tsv` (operator taxonomy in Bet 3), `student_resource/README.md` (metric: per-entity F0.5, singleton 1.0/0.0)
