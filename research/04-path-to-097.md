# Amazon ML Challenge 2026: path from 0.941 to 0.97+ (research pass 4)

Written 26 Sep 2026, about 04:30–05:30 IST (about 42 h before the 27 Sep 23:59 IST close). **Web research only**: nothing was run on the data. The only local reads were `research/03-refinement-research.md`, `research/official_QA.tsv`, and a skim of `code/ber_v4/src/{block_big,run_big}.py` and `runs/*.log`, so the advice fits what the pipeline already does.

**Tags**
- **[V]** read in the cited source.
- **[U]** another team's claim, read in their repo but not reproduced by us.
- **[E]** our own estimate or arithmetic.

This file builds on `03-refinement-research.md` and does not repeat it. Items that 03 recommended and the code already has are noted as done: `hide_frac=0.19` density-matched validation, the Indic lexicon, sibling vouching (stage B), and the France normaliser.

---

## 0. Headline

1. **Where the leaderboard is.**
   - At about 00:00 IST on 26 Sep, **#1 was 0.987 and #50 was 0.979** ([Epic021 STRATEGY §R1](https://github.com/Epic021/amazon-ml-challenge-2026/blob/main/STRATEGY.md)) [U].
   - Another team puts the **top-10 cut at 0.986** ([Karthikatstuffmama README](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026/blob/main/README.md)) [U].
   - A third says the top team was **at 0.98 on day 1** ([VanshGupta18 SKILL.md](https://github.com/VanshGupta18/amazon-ml-template/blob/main/.claude/skills/amazon-er-challenge/SKILL.md)) [U].
   - **So 0.97 is probably outside the top 50, and the top 50 get the PPIs.** Plan for about 0.98.
2. **Public LB scores we found.** All [U], each from the linked repo:

   | Team | Public LB | Holdout / validation behind it | Source |
   |---|---|---|---|
   | Epic021 | 0.952 | holdout 0.9662 | [STRATEGY §R4](https://github.com/Epic021/amazon-ml-challenge-2026/blob/main/STRATEGY.md) |
   | Team ICE | ~0.95 | OOF 0.9725 | [REPORT §5](https://github.com/jeetsidhu/AMAZON-ML-Challenege/blob/main/docs/REPORT.md) |
   | Karthikatstuffmama | 0.945193 | dev slice 0.984, inflated | [EXPERIMENTS](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026/blob/main/EXPERIMENTS.md) |
   | **Us** | **0.941** | CV 0.946 | — |

   **Every team at or below ~0.95 relied on word or key blocking with capped document frequencies.**

3. **Teams with realistic full-pool validation at 0.978–0.985 share four things** (§1.1):
   - **Record-side retrieval.** Every S2/S3 record retrieves its top 3–10 S1, using a *character 3-gram* TF-IDF on the space-less name plus a word TF-IDF on the address. Measured pair recall is **0.987–0.99**, against our **India 0.933 / US 0.965**.
   - **A learned Indic→Latin token lexicon.** We already have this.
   - **Record-side competition features**, such as the margin over the record's best other S1.
   - **"What differs" features** aimed at the generator's *neighbour decoys*: an added business or geographic word plus a changed house number.

4. **Rough loss split for us [E].**
   - The test mix is India 46.8% / US 38.3% / France 15% (from [Epic021 §2](https://github.com/Epic021/amazon-ml-challenge-2026/blob/main/STRATEGY.md) counts: 810k / 663k / 259k S1).
   - Solving 0.468·0.938 + 0.383·0.955 + 0.15·F = 0.941 gives **France ≈ 0.91** if our CV transfers. Epic021 derived France ≈ 0.88 the same way.
   - About 0.029 of LB points sit in US and India (to reach 0.975 / 0.985), and about 0.005–0.01 in France.
   - **Blocking recall is the largest single lever.** Peers convert +1 point of pair recall into about **+0.4 points of macro F0.5**:
     - Mounika-Reddy: recall 0.9398 → 0.9589 gave F0.5 0.9574 → 0.9655 ([log](https://github.com/Mounika-Reddy-0802/Business-Entity-Resolution/blob/main/benchmarks/experiments.md)) [U].
     - Karthikatstuffmama: union recall 97.42% → 98.18% raised the oracle ceiling by 0.0030 ([LITERATURE.md](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026/blob/main/LITERATURE.md)) [U].

---

## Do these next (ranked)

Gains are private-LB macro-F0.5 estimates [E] unless a source is given. The CPU column refers to this 8 GB, 8-core Mac while the pipeline is also running.

| # | Change | Expected gain | Effort | CPU-feasible? | Risk | Evidence |
|---|---|---|---|---|---|---|
| 1 | **Loss decomposition from the existing OOF** (details below the table). Decides between items 2, 3 and 4. | 0 (it steers the other items) | 0.5 h | yes, seconds | none | Karthikatstuffmama ([FINDINGS](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026/blob/main/FINDINGS.md)): blocking 0.0093 + matcher 0.0069 [U]. Epic021: ceiling 0.9768, model reaches 98.9% of it [U]. Ayan ([FINAL_RECOMMENDATION](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/FINAL_RECOMMENDATION.md)): oracle 0.9972 vs LGB 0.9846 [U] |
| 2 | **Record-side char-3-gram hybrid retriever**, unioned into blocking, plus **record-rank protection in `prune_by_similarity`** (details below the table). | **PC India +3–5 pts, US +1.5–2.5 pts → +0.008–0.015 LB** | 3–4 h code + ~1 h compute (train + test) | **yes.** `sparse_dot_topn` is already in `block_big.py`. The index side is S1 (the small side), so cost is O(records × nnz × S1 postings) | med: RAM, and +2–4 candidates per S1; needs a retrain | See the evidence list below the table |
| 3 | **Label-free "word-role" and number features for the neighbour decoys** (details below the table). | +0.003–0.01 (larger in France) | 2–3 h | yes | low–med | Epic021 top gain features: `gap_c, rank_q, is_c_best, add_lo_sum, add_lo_min, num_rel` ([STRATEGY](https://github.com/Epic021/amazon-ml-challenge-2026/blob/main/STRATEGY.md), [features.py](https://github.com/Epic021/amazon-ml-challenge-2026/blob/main/src/features.py)) [U]. Siva: difference and margin features plus the Indic dictionary gave **0.9707 → 0.9783** ([doc §5](https://github.com/Siva402-ai/amazon-ml-challenge-2026/blob/main/Documentation_template.md)) [U] |
| 4 | **France pass** (details below the table). | **+0.004–0.012 LB** | 2 h + 2 uploads | yes | med: the threshold direction is disputed | Epic021: "France ≈ 0.88 … worth ~+1.2 LB points"; French decoys add "Développement / Participations / France" plus a new number [U]. ICE LOCO: under-confident, best thr 0.12. Ayan LOCO: over-confident, best thr 0.8–0.9 [U] |
| 5 | **Decision freebies** (details below the table). | +0.000–0.002 | 0.5–1 h | yes | low | Karthikatstuffmama [FINDINGS](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026/blob/main/FINDINGS.md) (caps, 0 violations) [U]. Epic021 B1 (soft beats hard by +0.0008) [U]. Arshbir1 [E9](https://github.com/Arshbir1/Amazon_ML_Challenge/blob/main/student_resource/business_entity_resolution/experiments/results.md) (+0.0017 expected-F on stage 2, −0.0013 on raw stage 1) [U] |
| 6 | **Lexicon QA** (details below the table). | +0–0.005 (India) | 0.5–1 h | yes | low | Ayan: 1,347 tokens, 96.4% test-name coverage, name token-set p10 **8 → 100** ([RESEARCH_LOG](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/RESEARCH_LOG.md)) [U]. Siva: Indic recall@10 **91.3% → 99.4%** [U] |
| 7 | **Stage-B additions**, only if they are missing (details below the table). | +0.001–0.003 | 1–2 h | yes | med under country shift | Ayan stage 2 **+0.0021 (13 SE)** [U]. Arshbir1 **+0.0056** [U]. ICE: +0.0005 i.i.d. but **−0.014** under country shift ([REPORT §5](https://github.com/jeetsidhu/AMAZON-ML-Challenege/blob/main/docs/REPORT.md)) [U] |

**Details for each row**

1. **Loss decomposition.** From the existing OOF, compute:
   - the per-country **oracle macro-F0.5 on our candidate set**, so that blocking loss = 1 − oracle and matcher loss = oracle − CV;
   - **PC before vs after `prune_by_similarity`**;
   - a **blocking-miss taxonomy**: Indic script / domain or handle / empty target address / renamed (name share < 50) / typo in the only rare token / "shares tokens but ranked out".

   **What to do with the result:**
   - If our matcher loss is ≫ 0.012 (the peers are at 0.007–0.017), weight items 3 and 7 up.
   - If most of the loss is blocking, item 2 comes first.

2. **Record-side retriever.**
   - For every S2/S3 record, take the top-3 S1 by 0.5·cos(char-3-gram TF-IDF of the *space-less core name*) + 0.5·cos(word TF-IDF of the address), with sublinear TF and IDF fitted on S1.
   - Protect in `prune_by_similarity`: keep any pair that is **the record's rank ≤ 2** S1, even if it is outside the S1's top-20.
   - Code is in §A.

   **Evidence:**
   - Siva: recall@10 **98.7%**, rank-1 alone **97.1%**, on the full 10.3M pool ([doc §3.2](https://github.com/Siva402-ai/amazon-ml-challenge-2026/blob/main/Documentation_template.md)) [U].
   - Ayan: reverse **R@1 0.972, R@8 0.9875** on the full pool ([RESEARCH_LOG](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/RESEARCH_LOG.md)) [U].
   - Epic021: adding name-char and address retrievers took recall **0.941 → 0.968** and the ceiling **0.9768 → 0.9867** [U].
   - SABER: reverse trigram top-5 alone gets **0.986 India / 0.984 US** ([BLOCKING §4.4](https://github.com/yogeshwars-cys/Amazon-ML-2026-SABER/blob/main/docs/BLOCKING.md)) [U].

3. **Word-role and number features.**
   - **(a) Per-country log-odds of each *added* or *dropped* name token.** Learn them from rank-1 candidate pairs whose address token-set is ≥ 90, labelled "same house number" = 1 and "different number" = 0. That labelling is label-free and works on test France.
   - **(b) House-number |Δ| buckets** {0, 1–5, 6–20, 21–50, >50, missing} plus digit-edit class.
   - **(c) Leftover-token features**: counts of tokens present on one side only, and a fuzzy ratio between the two leftover sets.
   - Details in §1.3 and §B.
   - Supporting evidence: Ayan found |Δ|∈[1,5] in 24% of FP pairs vs 1.9% of true pairs ([synthetic_noise §0](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/research/open_scope/synthetic_noise.md)) [U].

4. **France pass.**
   - (a) Mine **département↔région** and **street-type aliases** from the test France records by city co-occurrence (label-free, and allowed: "any algorithm using only the provided records" [Q&A]).
   - (b) Item 3's word roles, computed on France.
   - (c) Two France-only LB uploads that differ only in the France threshold.

5. **Decision freebies.**
   - Hard per-S1 caps: **≤5 S2 and ≤6 S3** (they hold exactly on 7.64M train records).
   - Try **soft exclusivity** p′ = o/(1+Σo) against the current hard argmax.
   - Use expected-F0.5 only on *calibrated stage-B* output.

6. **Lexicon QA.**
   - Measure the share of Indic-script test tokens that the lexicon covers. Peers get 96.4%.
   - Add address-side native-script state and city tokens.
   - Check the 200 most frequent Indic tokens by eye.
   - An old log line `teknolojis -> classic` shows that alignment can go wrong. The current `lexicon.json` maps it correctly.

7. **Stage-B additions.** Add these only if missing:
   - the S1's **house-number consensus** among its confident records;
   - **S2∧S3 agreement**;
   - an isolation flag.

   Fall back to stage A for France if LOCO says stage B hurts.

**Suggested order for the next ~40 h** [E]:
1. Item 1 (30 min).
2. Items 2, 5 and 6 together, in one retrain → **upload A**.
3. Item 3 → **upload B**.
4. Item 4's France threshold A/B → **uploads C and D**.
5. Freeze by 27 Sep ~18:00 IST and make the final upload by ~21:00 IST (03 §upload plan: 100 MB uploads have hung).

**Don't do (measured dead ends or not feasible here)**
- **Cross-encoder on this Mac.** No-go; see §4.
- **Fine-tuning a bi-encoder on CPU.** SABER needed an RTX 3050 for 20 min, plus 70 min of GPU encoding ([BLOCKING §3](https://github.com/yogeshwars-cys/Amazon-ML-2026-SABER/blob/main/docs/BLOCKING.md)) [U].
- **BM25 as a *replacement* for blocking.** It got 94.29% recall at k=40 vs the incumbent's 97.42% at 21 ([LITERATURE.md](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026/blob/main/LITERATURE.md)) [U]. As a *union* it helps.
- **Per-country percentile standardisation of features.** US→India fell from 0.94475 to 0.93369 ([RESEARCH_LOG](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/RESEARCH_LOG.md)) [U].
- **Hungarian or global assignment.** It is a no-op: of 1,781,822 candidates, only **10** were contested above the threshold ([Arshbir1 E8](https://github.com/Arshbir1/Amazon_ML_Challenge/blob/main/student_resource/business_entity_resolution/experiments/results.md)) [U]. Ayan's exclusivity added **+0.00002** [U].
- **More threshold tuning.** Karthikatstuffmama measured the best global τ at −0.000274 and per-country τ at −0.000083 against their shipped rule [U]. ICE's curve is flat within 0.0005 over 0.6–0.9 [U].

---

## 1. What other 2026 teams did

Repos were found with a GitHub search sweep: **724 repos** pushed since 24 Sep that match "amazon ml 2026 / business entity resolution / AMLC". **286** READMEs mention the task, and we mined their ~650 non-README docs. Social media (Reddit r/Btechtards, LinkedIn, X, Kaggle) gave nothing quantitative: only upload-issue threads. Every number below is the team's own claim [U] unless it is marked otherwise.

### 1.1 Comparison table (realistic, full-pool validations first)

| Team | Validation (protocol) | LB | Pair recall @ cands/S1 | Blocking | Model / decision | Source |
|---|---|---|---|---|---|---|
| **AyanAhmedKhan** | **0.98456** stage 1 → **0.98703** stage 2 (fold 0 = 220k S1, full pool) | not uploaded (as of 25 Sep) | union 0.9905 @ 112; budget ≈ 0.989 @ 43–51; **reverse R@1 0.972, R@8 0.9875** | 5 views: char_wb-3g name, char-3g space-less name, word address, hybrid, **target→S1 reverse top-8** | LGB ~87 feats; top gain `c_combo_tmargin`; thr 0.7 + exclusivity; stage-2 collective | [RESEARCH_LOG](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/RESEARCH_LOG.md), [FINAL_RECOMMENDATION](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/FINAL_RECOMMENDATION.md) |
| **Tony-AJ** | **0.9844** (tune side 441k S1) | uploaded, score not logged | 0.9906 @ ~34–37 (10k-S1 val: 0.987 India / 0.994 US at ~30) | exact name keys + **name+address word uni+bigram TF-IDF top-25 (max_df 0.01)** + name char-3g for short-address records; cap 60 | LGB 47 feats, 6.72M pairs from 200k S1; τ 0.47, one-to-one | [LEADERBOARD.md](https://github.com/Tony-AJ/business_entity_resolution/blob/main/LEADERBOARD.md), [TRACKER.md](https://github.com/Tony-AJ/business_entity_resolution/blob/main/TRACKER.md) |
| **Siva402-ai** | **0.9783** (20% S1 held out vs **all 10.3M** records; 2× decoys → 0.9759) | committed test file; LB not stated | **98.7% @ 10 per record** (~57 per test S1); India 98.3 / US 98.9 | **record → S1**: char-3g of space-less core name (sublinear, max_df 0.05) + address words (max_df 0.10), score 0.5/0.5 | LGB 64 feats, 892 trees, 8.2M pairs; thr 0.7 per record; expected-F < +0.001 | [Documentation_template](https://github.com/Siva402-ai/amazon-ml-challenge-2026/blob/main/Documentation_template.md) |
| **Epic021** | 0.9662 (India 0.9546, US 0.9739) | **0.952** | 0.937 pruned (ceiling 0.9768) → with char retrievers **0.968** (ceiling 0.9867) | word + skeleton TF-IDF both directions → + name-char + address retrievers | LGB 60 feats, isotonic, **soft exclusivity**, τ 0.55 | [STRATEGY](https://github.com/Epic021/amazon-ml-challenge-2026/blob/main/STRATEGY.md) |
| **Team ICE** (jeetsidhu) | OOF 0.9725 (leaky folds); 5% subset 0.991 | **~0.95** | subset R@3 0.988 | record → S1 top-3; blocks: name tokens, **order-free token pairs**, **8-char compact prefix**, addr tokens + bigrams, **house-number × key token**, **name × place cross** | 2-stage LGB, Platt, thr 0.79 | [REPORT](https://github.com/jeetsidhu/AMAZON-ML-Challenege/blob/main/docs/REPORT.md), [blocking.py](https://github.com/jeetsidhu/AMAZON-ML-Challenege/blob/main/src/blocking.py) |
| **Karthikatstuffmama** | dev slice 0.9838 (3% world; inflated by 0.039) | **0.945193** | 97.42% @ 20.9 (dev) | keys with `max_df` caps (the diagnosed failure) + planned BM25 union | 2-stage LGB, expected-F | [README](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026/blob/main/README.md), [EXPERIMENTS](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026/blob/main/EXPERIMENTS.md) |
| **Arshbir1** | 0.938 → **0.9436** with stage 2 (40k S1, full haystack) | — | 0.9689 @ ~37 | S1-side, 8 blockers; **metaphone name + address = best ratio (+0.0095 recall for +1.1 cands)** | LGB 111 feats + **stage-2 reranker +0.0056** | [README](https://github.com/Arshbir1/Amazon_ML_Challenge/blob/main/README.md), [results.md](https://github.com/Arshbir1/Amazon_ML_Challenge/blob/main/student_resource/business_entity_resolution/experiments/results.md) |
| **Mounika-Reddy** | 0.9574 → **0.9655** (address-pair key) | — | 0.9398 → 0.9589 @ 19.8 | name+location keys + "same address, any name" key; cap-20 ranker | LGB + rule tuning | [experiments.md](https://github.com/Mounika-Reddy-0802/Business-Entity-Resolution/blob/main/benchmarks/experiments.md) |
| **SABER** (yogeshwars-cys) | blocking only | — | 2.5% world: **US 0.9978 @ 24.8, India 0.9903 @ 29.1** | fine-tuned arctic-embed-xs dense leg (India R@20 0.835 → **0.992**) + char_wb-3g trigram + key legs, **both directions** | selector planned | [BLOCKING.md](https://github.com/yogeshwars-cys/Amazon-ML-2026-SABER/blob/main/docs/BLOCKING.md) |
| adbhargav | synthetic only (0.976) | — | "≥99% union of 5 paths, pruned to ≤20" | 5 paths + GBDT pruner | expected-F (MC) + one-to-one | [STRATEGY](https://github.com/adbhargav/amazon-ml-challenge/blob/claude/nifty-noether-ab17py/docs/STRATEGY.md), [decide.py](https://github.com/adbhargav/amazon-ml-challenge/blob/claude/nifty-noether-ab17py/code/business_entity_resolution/src/ber/decide.py) |
| **Us** | CV 0.946 (US 0.955 / India 0.938) | **0.941** | India 0.933 / US 0.965 @ 20 | hashed word keys (`max_df_keys` 3000), fwd 24/8/8/8 + rev 4/2/2/2, S1-top-20 similarity prune | LGB ~75 feats + stage B; thr 0.7 + exclusivity | local |

**Reading the table.**
- The ~0.98 validation group (Ayan, Tony-AJ, Siva) all reach **pair recall ≥ 0.987**.
- The ~0.94–0.95 group (Arshbir1, Mounika, Epic021-B1, us) sits at **0.94–0.97**.
- The exceptions are the teams whose number is inflated by a small pool: Karthikatstuffmama's dev slice and ICE's subset.
- **Recall at the blocking stage is what separates the groups.** The matcher is second.

### 1.2 Per-team extraction (techniques, thresholds, numbers)

**AyanAhmedKhan** ([RESEARCH_LOG](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/RESEARCH_LOG.md), [FEATURE_CATALOG](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/FEATURE_CATALOG.md), [ERROR_ANALYSIS](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/ERROR_ANALYSIS.md)) [U]

*Blocking*
- Benchmark on 10% of India: name-only top-40 = 0.826, address top-40 = 0.909, hybrid top-40 = 0.984, **reverse target→S1 top-8 = 0.996**.
- On the full pool, **the name view collapses to R@40 = 0.677** because same-name records flood it. The reverse view is the strongest single blocker.
- Final budget: rev8 ∪ combo10 ∪ name/cat/addr top-5 ≈ 45 candidates per S1 at ≈ 0.989.

*Features*
- **Competition features over the full candidate graph**, for each c ∈ {combo, name, addr}:
  - `qrank`, `qgap`: rank and gap within the S1's list;
  - `trank`, `tgap`: rank and gap among the S1s competing for the same record;
  - **`tmargin`**: margin over the best *other* S1 for the record. `c_combo_tmargin` is the top gain feature by far.
- "Core" tokens = name tokens with within-country DF ≤ 1%. So generic tokens are learned, not listed.

*Model and results*
- LightGBM: lr 0.08, 255 leaves, min_data 400, ff/bf 0.7. Early stop at 777 rounds.
- **Stage-1 score: 0.98456.**
- Deterministic rule baselines on the same candidates: 0.422 / 0.714 / 0.767.
- The exclusivity rule adds +0.00002. Expected-F (0.98415) and a π-hurdle "match-exists" model (0.98466) did not beat the threshold.
- **Stage 2 (+16 collective features + p1 competition): 0.98703 (+0.0021 ≈ 13 SE).**
  - Features: p1-weighted support, same- vs cross-source sibling confirmation, house-number agreement with the confident cluster, isolation flag.
  - Pair precision went 0.9959 → 0.9973.
- **Density-matched ("P-dense")**, with 410k S1 hidden so that train reaches 5.74 targets per S1:
  - A normally trained model falls to 0.98343 and its best threshold rises 0.7 → 0.8.
  - A model *trained* under P-dense gets **0.98479**.

*Error analysis* (FN ≈ 5.4× FP, like ours)
- **FN:** 49% have an empty target address, 59% a shared S1 name, 46% are not best-for-target. Mostly undecidable, so leave them.
- **FP:** planted siblings, e.g. "Amber Inc 340 vs 345 Olympic Park Dr". But true copies also get small offsets, e.g. "Downtown Deli 104 vs 103".

*Leave-one-country-out (LOCO) and self-training*
- US→India 0.94475 (best thr 0.9), India→US 0.97390 (thr 0.8).
- **In their runs, OOD models were over-confident**, which is the opposite of ICE's finding.
- Self-training on the unseen country: +0.0022 India→US, −0.0017 US→India. It helps only when pseudo-label precision is ≥ 99%.

*Bugs worth checking in our code*
- ZWNJ (U+200C) was left inside lexicon keys, so frequent Telugu/Kannada tokens never matched.
- Abbreviation maps were not closed under composition (kl→kerala→keralam).
- Gurmukhi tippi/addak and Malayalam chillu were dropped.

**Tony-AJ** ([LEADERBOARD.md](https://github.com/Tony-AJ/business_entity_resolution/blob/main/LEADERBOARD.md), [TRACKER.md](https://github.com/Tony-AJ/business_entity_resolution/blob/main/TRACKER.md)) [U]
- v001 = val 0.9844, cand recall 0.9906 at 34–37 per test S1. It uses a 539-token learned Indic map.
- Blocking: exact core-name, sorted-name and squashed-name keys (pool groups ≤ 50), plus name+address word uni+bigram TF-IDF top-25 with `max_df 0.01` and `max_df_abs 10k`, plus name char-3-gram top-10 for pool records with short addresses. Cap 60.
- **Note:** their word TF-IDF keeps common tokens in the norm; only the df > 1% terms are dropped from the product. We zero any key with df > 3,000 **entirely**.
- Decision: τ_abs 0.47, τ_single 0.52, max 11 matches, one-to-one. The model hit its 2,000-round cap, so it was under-trained.

**Siva402-ai** ([Documentation_template](https://github.com/Siva402-ai/amazon-ml-challenge-2026/blob/main/Documentation_template.md), [retrieval.py](https://github.com/Siva402-ai/amazon-ml-challenge-2026/blob/main/business_entity_resolution/src/retrieval.py), [indic_dictionary.py](https://github.com/Siva402-ai/amazon-ml-challenge-2026/blob/main/business_entity_resolution/src/indic_dictionary.py)) [U]

*Retrieval*
- Hybrid views: name-only recall@10 was 61% on India and address-only 86%; **combined 97%**.
- Exact settings:
  - name view: `TfidfVectorizer(analyzer="char", ngram_range=(3,3), min_df=2, max_df=0.05, sublinear_tf=True)` on `"<"+core_name_without_spaces+">"`;
  - address view: `analyzer="word", max_df=0.10, sublinear_tf=True`.
- GPU CountSketch 1024-d approximate top-40, then an exact rescore, keep 10.

*Indic dictionary*
- Align tokens by position when the token counts are equal. Keep a mapping if it occurs ≥ 3 times and has ≥ 60% share. The result is 526 maps, learned only on training-fold entities.

*Features*
- 64 features, including "what differs": `nm_q_only`, `nm_s_only`, `nm_diff_ratio` (fuzzy ratio of the leftover tokens: a typo scores high, a substituted word low), `ad_qonly_in_s`, `num_q_only`, `num_s_only`, `num_rel_diff`, `num_set_eq`.
- Margins vs runner-up for cos_name, cos_addr, name token-set, ratio, address token-set and number coverage.
- Also: "number of S1 in country with same core name", and "number of records that retrieved this S1 at rank 1".
- Top gains: gap to the best *other* candidate, rank, combined score, count of query-only numbers.

*Threshold sweep* (flat top)
- 0.6 → 0.9782, **0.7 → 0.9783**, 0.8 → 0.9774. Doubling the decoys leaves the best threshold at 0.7.

*Test profile*
- 58–63% of records are matched per country (France 63.3%).

**Epic021** ([STRATEGY](https://github.com/Epic021/amazon-ml-challenge-2026/blob/main/STRATEGY.md), [bottle_necks](https://github.com/Epic021/amazon-ml-challenge-2026/blob/main/bottle_necks.md), [research_sota](https://github.com/Epic021/amazon-ml-challenge-2026/blob/main/research_sota.md), [features.py](https://github.com/Epic021/amazon-ml-challenge-2026/blob/main/src/features.py)) [U]

*Blocking*
- Miss causes, over 449k misses (5.9% of true pairs):

  | Cause | Share |
  |---|---|
  | Indian script | 25% |
  | Domains, handles, number typos | 24% |
  | Similar but ranked out | 21% |
  | Empty candidate address | 18% |
  | Renamed | 12% |

- Char-3-gram retrievers (name without spaces; name only; address only) target 75–80% of these. They added only **+2.7 pts recall** because the word blocker already found most pairs. Found only by one retriever: word 3.1%, namechar 0.8%, address 1.1%.

*Generator facts*
- Positives' house numbers: equal 65%, candidate drops the number 12%, differs 10% (31→30, 1600→160), candidate adds a number 10%.
- 44% of unmatched records have a near-identical address to some S1. **For those, the number differs in 88.2% and is equal in 2.5%.**
- **Even at name/address similarity ≥ 92, 14% of best candidates are negatives.**
- Decoys **add** words such as holdings, group, enterprises, industries, exports, public, overseas, ventures, infratech, north/south/east/west, uptown/downtown/midtown, metro, central, valley, harbor, summit, eastgate/westgate. French examples: "Développement", "Participations", "France".

*Label-free word roles (their G3)*
- Take rank-1 candidate pairs with address token-set ≥ 90.
- Label y=1 if the house number is equal and y=0 if it differs.
- Per country, compute smoothed log-odds of each added and dropped token. Features: `add_lo_sum/max/min`, `drop_lo_sum/max/min`.
- What it learned on the full run:
  - US: exactly the decoy vocabulary.
  - France: participations −1.99, holding −1.99, international −2.01, développement −1.35, groupe −1.31, france −1.13.
  - Caveat: France's scale is shifted (median −0.69).

*Decisions*
- **Soft exclusivity beats hard by +0.0008.**
- An exact expected-F DP (0.9657) lost to the plain threshold (0.9662).

*Label shift*
- The mean Σp per S1 is similar on test (France 3.34, India 3.14, US 3.42) and on the holdout (3.25). **So the extra test records are mostly decoys.** Ayan and Siva agree.

*France structure*
- About 15 cities in 3 régions (Lille, Roubaix, Tourcoing, Dunkerque, Calais, Bordeaux, Pessac, Mérignac, La Teste-de-Buch, Lège-Cap-Ferret, Nantes, Saint-Nazaire, Saint-Herblain, Pornic, La Baule).
- This makes streets very dense, so house-number features matter more.
- Région↔département swap: Nord / Pas-de-Calais ↔ Hauts-de-France, Gironde ↔ Nouvelle-Aquitaine, Loire-Atlantique ↔ Pays de la Loire.
- Name vocabulary repeats (sarl, sas, club, ecole, amicale, comite).

**Team ICE** ([REPORT](https://github.com/jeetsidhu/AMAZON-ML-Challenege/blob/main/docs/REPORT.md), read in full; [blocking.py](https://github.com/jeetsidhu/AMAZON-ML-Challenege/blob/main/src/blocking.py)) [U]

*Blocking*
- Retrieval direction is record → S1, top-3.
- Recall@k on a 5% name-group subset (optimistic, because the index is 20× smaller): **@1 0.9792, @2 0.9857, @3 0.9882, @5 0.9906, @10 0.9931**.
- Blocking features, one TF-IDF per block, IDF from S1, `df_cap 1000` for the index but **all features in the norms**:
  - `n:` token, `p:` order-free token pair (first 6 tokens), `k:` 8-char compact-name prefix;
  - `a:` address token, `b:` address bigram, `h:` house-number × key address token;
  - `c:` core-name token × key address token.
  - Score = cos_name + cos_addr + cos_cross.

*Model*
- Ablation (ΔF0.5 when the group is removed): house numbers −0.00101, legal form −0.00070, stage 2 −0.00046, name strings −0.00039. Removing retrieval features costs nothing, because they are redundant with the string features.
- Strongest single-feature AUCs: `gap_second` 0.9926, score 0.9797, cos_addr 0.9595.
- Platt calibration: a = 0.805, b = −0.065. ECE 0.00086 → 0.00032.
- Threshold 0.79 (calibrated). The curve is flat within 0.0005 over 0.6–0.9.

*Robustness*
- 2× decoys: regret 0. **Turning 6% of entities into singletons costs −0.055**, all in precision.
- LOCO: India held out 0.9047 at the global thr (**best thr 0.12**); US held out 0.9853 (0.83). Stage 2 hurts under shift (0.9667 → 0.9530).

*Errors*
- Recall misses: retrieval miss 1.18% (72% with no address, 81% in namesake chains), out-ranked 0.49%, below threshold 0.77%.
- False positives: decoys linked at mean p 0.935. Near-copy decoys ("Hashmi … MD, DDS PC" vs "… MMD, DDS") set a precision floor.

*Text rules*
- Keep country tokens inside names ("Air India"). Strip ambiguous legal forms (co, pc, pa, sa, ag, lp, ei) only in legal position.
- Collapse long forms to short forms, never the reverse.
- French street rules only for France: ungated, "R K Puram" became "rue k puram".

**Karthikatstuffmama** ([FINDINGS](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026/blob/main/FINDINGS.md), [LITERATURE](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026/blob/main/LITERATURE.md), [PLAN](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026/blob/main/PLAN.md)) [U]

*Generator operator frequencies* (139,271 true pairs; see §2.2)

*Structural facts on the full train* (0 violations each)
- **One owner per record.**
- **n_S2 ≤ 5 and n_S3 ≤ 6 per S1.**
- Every S1 is unique by normalised name + address.

*Error mix* (dev slice)

| Category | Share |
|---|---|
| Blocking miss | 63.1% |
| Missed in candidates | 29.1% |
| FP from decoys | 6.1% |
| FP stolen from another S1 | 1.0% |
| FP on singletons | 0.7% |

- **99.92% of missed pairs share at least one token with their S1.** The `max_df` caps and top-K discarded them.
- Miss lift: non-Latin 37.1% (6.1×), empty address 23.0% (6.1×), no shared name token 72.1% (5.5×).
- `anyascii` renders पावर→"pavara", which never token-matches "power".

*Decision bounds*
- `oracle_topk` 0.990587 ≈ `oracle_subset` 0.990727, so **ranking is solved**. The remaining ~0.0065 is per-entity k selection: predicted 3.344 matches per S1 vs 3.462 true.
- Idea, not built: a group-size head that predicts n_S2 and n_S3, clipped to the caps.

*BM25 as a union*
- k=5 gives union recall 98.18% (+0.76 pts) for +1.24 candidates per S1, and **oracle +0.0030**.
- k=10: +0.0039 oracle for +4.9 candidates.

**Arshbir1** ([results.md](https://github.com/Arshbir1/Amazon_ML_Challenge/blob/main/student_resource/business_entity_resolution/experiments/results.md)) [U]

*Normalisation*
- A "repeat-squeeze" view (`raam maarketting` → `ram marketing`) raised cross-script positives with token-set ≥ 70 from 42.3% to 66.1%.
- **Metaphone name + address blocker:** +0.0095 recall for +1.1 candidates (India S3 0.9341 → ~0.949). Metaphone runs at 3.04M encodings/s.

*Stage-2 reranker*
- 27 features over the stage-1 probability field; best iteration ~109 of 600.
- F0.5 **0.93803 → 0.94363**. Singleton accuracy 0.877 → 0.901.
- **Expected-F loses on raw stage-1 scores (−0.0013) but wins on stage-2 scores (+0.0017).**

**SABER / yogeshwars-cys** ([BLOCKING.md](https://github.com/yogeshwars-cys/Amazon-ML-2026-SABER/blob/main/docs/BLOCKING.md)) [U]

*Zero-shot encoder screen* (R@20, US / India, 2.5% world)

| Model | Licence | Params | US | India |
|---|---|---|---|---|
| arctic-embed-xs | Apache-2.0 | 22.6M | 0.990 | 0.929 |
| multilingual-e5-small | MIT | 118M | 0.988 | **0.946** |
| paraphrase-multilingual-MiniLM-L12 | Apache-2.0 | — | 0.870 | 0.730 |

- Base-size models are no better and about 4× slower.
- **Contrastive fine-tune of arctic-xs** (symmetric InfoNCE, batches of name-sorted hard negatives, 3k steps, batch 256, 20 min on an RTX 3050): **India R@20 0.992**, US 0.996. The trigram leg for comparison: India 0.975, US 0.996.

*Trigram `max_df` sweep* (R@20)

| max_df | US | India |
|---|---|---|
| 1.0 | 0.9958 | 0.9746 |
| 0.05 | 0.9944 | 0.9731 |
| 0.02 | 0.9882 | 0.9602 |
| 0.01 | 0.9709 | 0.9426 |

- **Pruning below max_df 0.05 costs recall quickly.**
- The reverse direction is "very strong and cheap": reverse-trigram top-5 alone gets 0.986 India / 0.984 US.

**Mounika-Reddy** ([experiments.md](https://github.com/Mounika-Reddy-0802/Business-Entity-Resolution/blob/main/benchmarks/experiments.md)) [U]
- An address-pair key ("same address, any name") took recall **0.9398 → 0.9595** and val F0.5 **0.9574 → 0.9655**.
- On synthetic data:
  - caps 4/3: −0.0001; one-to-one: +0.0004; stage 2: +0.003;
  - MiniLM cosine features: not kept;
  - LightGBM lr 0.05 → 0.02: +0.0003 (noise).

**Other score claims** (small or synthetic samples; not comparable) [U]
- Aamod007: 0.98006 on 1,498 held-out S1, "global consistency +0.00078".
- Akash-bardia: 0.9761.
- Graybeep: 0.9648 grouped 5-fold on 330k S1.
- Nikhil-iitg27: 0.9672, recall 0.9817 at 65.6.
- DevrajBijpuria: 0.9709.

### 1.3 What a 0.98 pipeline looks like (common denominator)
1. **Normalise.**
   - Fold accents.
   - Apply the learned Indic token map (≥ 526 entries, ≥ 95% test coverage).
   - Split aliases (dba, aka, fka) and domains or handles.
   - Repair leet.
   - Handle legal forms by position.
2. **Block.**
   - Per country. **Record → S1 top 3–10** on a char-3-gram (space-less name) + word-address hybrid.
   - Union with an S1-side top-k.
   - 30–50 candidates per S1, **pair recall ≥ 0.987**.
3. **Features.**
   - Rapidfuzz, IDF overlap, house-number relation and |Δ|.
   - Legal-form agreement.
   - **Record-side margin over the best other S1** (top feature).
   - Within-S1 rank and gap.
   - Leftover-token ("what differs") and added-word log-odds.
   - Name-frequency ambiguity counts.
4. **LightGBM** trained under test-like decoy density, then optionally a stage-2 collective LGB (+0.002–0.006).
5. **Decide.**
   - Per-record argmax (hard or soft).
   - Threshold ≈ 0.7–0.8.
   - Expected-F only on calibrated stage-2 output.

---

## 2. Exploiting the synthetic generator (research question 2)

### 2.1 What the literature says about generator-corrupted and "dirty" ER
- **DeepMatcher (SIGMOD'18)** ([PDF](https://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf)) [V]:
  - Its "dirty" benchmarks are *synthetically* corrupted: each attribute value is moved into `title` with 50% probability.
  - "DL … significantly outperform[s] [Magellan] on textual and dirty EM", improving F1 by **6.2–32.6%**.
  - DL is only competitive on structured EM.
  - The Magellan baseline there uses *automatic generic* features, not operator-specific hand features like ours.
- **Ditto (VLDB'21)** ([arXiv 2004.00584](https://arxiv.org/abs/2004.00584), Table 5) [V]:
  - F1 on the dirty sets: DBLP-ACM 99.03, DBLP-Scholar 95.75, iTunes-Amazon 95.65, Walmart-Amazon 85.69. Against DM+, the gains are +0.93 / +1.95 / +16.25 / +31.89.
  - **"The average improvement on the 7 smallest datasets is 15.6% vs. 1.48% on average on the rest."**
  - So PLM matchers help most when labels are scarce. We have millions of labelled pairs, so a well-featured GBDT is close [E].
- **Papadakis et al., ICDE'24** ([arXiv 2307.01231](https://arxiv.org/abs/2307.01231)) [V]:
  - "Most of the popular datasets pose rather easy classification tasks." The gap between the best non-linear and linear matchers is small.
  - Generator-based data behaves the same way. Once every operator is inverted in the features, the residue is **structural ambiguity**, not string noise.
- **Generator anatomy.**
  - GeCo ([Christen & Vatsalan, CIKM'13](https://doi.org/10.1145/2505515.2507815)) and FEBRL ([Christen, IDEAL'05](https://doi.org/10.1007/11508069_15)) have the same template:
    1. clean originals;
    2. k duplicates per original from a count distribution with a cap;
    3. per-attribute corruption probability;
    4. a weighted choice of corruptor (edit, keyboard, OCR, phonetic, missing, swap);
    5. edit positions biased away from the string start;
    6. FEBRL-style *household or family* records: copy an original, change a field, keep the address.
  - Ayan's analysis maps our data onto this template ([synthetic_noise §1–2](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/research/open_scope/synthetic_noise.md)) [U]:
    - planted siblings = the household generator;
    - S2 ≤ 5 / S3 ≤ 6 = `max_num_dup_per_rec`;
    - copies within one source share a corrupted base;
    - house-number noise is an **arithmetic offset or digit drop**, not a keyboard typo (hypothesis tested).
- **Inverting typo channels.**
  - The Brill–Moore noisy-channel model ([ACL 2000](https://doi.org/10.3115/1075218.1075255)) learns substring substitution probabilities from aligned pairs.
  - Its analogue here is a **learned substitution cost from aligned train pairs** (leet 5↔s, 0↔o, 1↔l, and transliteration vowel shifts). Use it as a weighted edit-distance feature [E].
  - This is a P2 idea. The char n-grams plus the lexicon already cover most of it.

### 2.2 Operator frequencies measured on the real ground truth, and the inverse each needs

Frequencies come from Karthikatstuffmama over 139,271 true pairs ([FINDINGS, Check 1](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026/blob/main/FINDINGS.md)) [U]. The "inverse / feature" column is our recommendation [E].

| Operator (share of true pairs) | Inverse / feature | Likely status in `ber_v4` |
|---|---|---|
| Street type abbreviated (29.1%) | Symmetric canonical map. Learn it from train pairs, and from test France anchors for France | have (canonicalisation) |
| Partial component reorder (28.0%), exact reorder (4.1%, S3-skewed) | Order-free token-set / IDF overlap | have |
| Address UPPERCASED (S2 53.35%, S3 0.01%) | Nothing: the source is known from the ID. Don't let case reach features | have (lowercase) |
| **Number perturbed (15.7%)** / fully changed (7.9%) | **HN relation with \|Δ\| buckets.** Digit drop and truncation (1515→151, 707→7) and leading zeros (00412) mean *match*. A small offset 1–5 at the same street means *decoy* (24% FP vs 1.9% TP, Ayan) | partly (tri-state) → **item 3** |
| Name truncated (15.6%) | Coverage asymmetry (the IDF share of the S1 name covered by the record) and a token-prefix feature | check |
| Diacritics injected (13.6%) | Fold on both sides | have |
| Legal suffix dropped (13.5%) | Strip legal forms by position; keep a legal-agreement feature | have |
| Address component dropped (11.75%, S2 18.8%) | Coverage features, not symmetric Jaccard | check |
| **Token appended (11.65%)**, of which descriptor words are only 0.94% | **Added-token log-odds / word role** (decoys add holdings, group, …; positives add dba, com, www, center, services, lp, shri) | **missing → item 3** |
| Non-Latin name (6.7%) | Lexicon (have), char-3-grams on the lexicon output, metaphone on transliteration | have lexicon; char-3-gram retriever → **item 2** |
| Domain or handle (5.7%) | Space-less name, char-3-grams, strip TLD and legal tails (`_ns_variants`), word-break against the S1 vocabulary | partly (keys) → **item 2** covers it |
| Word transposition (5.6%) | Token-set, order-free | have |
| Typo in name (5.5%) | JW, char-3-gram cosine. **Word keys miss a typo in the only rare token** | → **item 2** |
| Address empty (4.5%) | Name-only retrieval; name-frequency ambiguity; abstain when the name is shared | have |
| Random name (2.08%) | Address-only retrieval plus uniqueness of the address in S1 | have (b_akey) |

**Ceiling.**
- Karthikatstuffmama estimates about 0.11% of pairs are genuinely ambiguous: random name × address collision, giving **a ceiling of about 0.999** [U].
- ICE shows near-copy decoys at the same address set a small precision floor [U].
- The top LB score of 0.987 fits this. **The remaining gap to the top is engineering, not a noise floor** [E].

---

## 3. Set-level and collective decisions for one-to-many with ≤ 1 owner (research question 3)

| Idea | Measured on this challenge | Verdict |
|---|---|---|
| Hard exclusivity (record → best S1) | +0.00002 (Ayan); +0.00003, with only 10 of 1.78M candidates contested above thr (Arshbir1); +0.0004 (Mounika, synthetic); +0.00078 (Aamod, 1.5k S1) [U] | Keep (free), but it is not a lever: record-side competition features already encode it |
| **Soft exclusivity** p′ = o/(1+Σ_k o_kj), then decode per S1 | **+0.0008 over hard** (Epic021 B1) [U]. Exact under "independent Bernoullis conditioned on at most one S1" ([research_sota §4](https://github.com/Epic021/amazon-ml-challenge-2026/blob/main/research_sota.md)) | Try it: one line, OOF-testable |
| Hungarian / min-cost-flow / bipartite matching | The CCER algorithms ([Papadakis et al. arXiv 2112.14030](https://arxiv.org/abs/2112.14030)) [V] assume *one-to-one* between two clean sources. Ours is 1-to-many (mean 3.46) | Don't use |
| Per-S1 caps (S2 ≤ 5, S3 ≤ 6) | Hold exactly on 7.64M records; one team had 348 violating entities in its output (Karthikatstuffmama) [U]; "caps 4/3" (the wrong caps) −0.0001 (Mounika) [U] | Enforce the true caps: free precision |
| Expected-F0.5 set per S1 (GFM / DP) | Plain threshold wins on raw or stage-1 scores (Epic021 −0.0005, Ayan −0.0004, Arshbir1 −0.0013, Siva < 0.001). **Wins on calibrated stage-2 scores (+0.0017, Arshbir1)** [U] | Only on calibrated stage-B output |
| Group-size head (predict n_S2, n_S3, clip to caps, take top-n) | Proposed, not measured: Karthikatstuffmama puts the remaining headroom at about 0.0065 on the dev slice [U] | Optional. The expected-F layer on calibrated p approximates it |
| Collective / stacked features (stage 2) | +0.0021 (Ayan, 13 SE), +0.0056 (Arshbir1), +0.0005 (ICE), +0.003 (Mounika, synthetic), **ours +0.001** [U] | Have. Add HN-consensus and S2∧S3 agreement if missing. Watch LOCO |
| Multi-source clean/dirty (MSCD) clustering: at most one clean-source record per cluster (Lerm, Saeedi & Rahm, BTW'21; Splink `cluster_using_single_best_links`) | Not measured here. Ayan's graph note recommends anchored clustering only as a repair or veto step ([graph.md §0](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/research/notes/graph.md)) [U; Splink API name not verified] | Skip. Never use transitive closure: "worst precision" in every comparative study cited there |

**Decision theory** ([fmeasure.md §0](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/research/notes/fmeasure.md)) [U]:
- For large sets, include an item iff its calibrated p ≥ **F\*/(1+β²) = 0.8·F\***, following Lipton et al. [arXiv 1402.1892](https://arxiv.org/abs/1402.1892). With F\* ≈ 0.9 the cut-off is about 0.72, which matches everyone's 0.7–0.8.
- **Never use Π(1−p) as P(∅)**: the labels of copies are strongly dependent.

---

## 4. Small cross-encoder reranker on CPU (research question 4): **NO-GO**

**Licences**
- `cross-encoder/ms-marco-MiniLM-L6-v2`: 22.7M params, English. Named as the benchmark model on [sbert.net](https://sbert.net/docs/cross_encoder/usage/efficiency.html) [V]. Apache-2.0 per 03.
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`: Apache-2.0, 117.6M, multilingual (03 §licence).
- `Snowflake/snowflake-arctic-embed-xs`: Apache-2.0, 22.6M ([SABER table](https://github.com/yogeshwars-cys/Amazon-ML-2026-SABER/blob/main/docs/BLOCKING.md)) [U].
- All are within the 8B, MIT/Apache rule [Q&A].

**Throughput on this Mac** [E, derived]:
- The MiniLM-L6 *bi-encoder* runs at 137.8 texts/s on 2 CPU threads, and e5-small at ~280 texts/s on an 8-core Ryzen (03 §2.3, sources there).
- A cross-encoder sees name+address for *both* records, about 2× the tokens.

| Setting on 8 free M-series cores | Pairs/s |
|---|---|
| MiniLM-L6 CE, fp32 | ~250–400 |
| MiniLM-L6 CE, ONNX int8 | ~600–1,000. sbert.net says int8 dynamic quantisation gives CPU speedups but publishes the ratios only as images [V] |
| Multilingual L12 (the one needed for Indic and French) | about half of the above |

**The 3M-pair budget.**
- 3M uncertain pairs in < 3 h needs ≥ 280 pairs/s *sustained*.
- That is feasible only for int8 MiniLM-L6 with the CPU otherwise idle. **It is not feasible for the multilingual L12 while the LightGBM pipeline shares 8 GB and 8 cores.**
- Fine-tuning on ~200k pairs on CPU: forward + backward ≈ 3× inference, so about 20–40 min per epoch for L6 and about 1–1.5 h for L12. MPS may be 2–5× faster but competes for the same 8 GB.

**Expected gain.**
- No team reports a *measured* cross-encoder gain on this data.
- The peers that measured say ranking is essentially solved: stage-2 AUC 0.99996, `oracle_topk` ≈ `oracle_subset` (Karthikatstuffmama [U]).
- Mounika found that MiniLM cosine features did not help (synthetic) [U].
- Ditto's big wins come from small training sets (§2.1) [V].

**Verdict.** No-go within 43 h on this machine.
- The same hours are worth 3–10× more in items 2 and 3.
- The only neural option worth reconsidering is a *static* Model2Vec feature (03 item 9). Ayan used `potion-multilingual-128M` on 32 vCPU at about 36 min for 37.5M texts [U]. That is about 45 min on 8 cores for the test split with one text per record, or about 2 h with three texts per record [E]. Its gain here is still unmeasured.

---

## 5. Other levers with evidence of ≥ +0.01
1. **Blocking recall** (item 2): the evidence is in §0 and §1.1.
   - Epic021's ceiling rose 0.9768 → 0.9867.
   - Mounika's recall +1.9 pts gave F0.5 +0.008.
   - Siva's v1 → v2 was +0.0076.
2. **France** (item 4):
   - Epic021 derives France ≈ 0.88 from the LB against 0.955–0.974 for US/India, i.e. "worth about +1.2 LB points" [U]. Our own estimate is ≈ 0.91 [E], so about +0.5–0.8 LB points are available.
   - Specific mechanisms:
     - (a) French decoy words are unknown to English-trained added-word features, so use **label-free per-country word roles** (§B).
     - (b) The région↔département swap: mine it by city co-occurrence on test France. Ayan synthetic_noise §4 A3; Epic021's miner "learned France's region ↔ department pairs without labels" [U].
     - (c) France addresses have **no** null tokens, `#` junk, landmarks or PO boxes, and they use US forms ("AVE", "BLVD") on French street types ([synthetic_noise §2.5](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/research/open_scope/synthetic_noise.md)) [U]. So the US-learned abbreviation map partly transfers.
     - (d) Name-noise rates are identical across countries (all-upper S2 0.208 in France vs 0.204 in the US; domain 0.035 vs 0.036), so name features transfer [U].
3. **Density-matched training.** We already have `hide_frac=0.19`. Ayan measured +0.0014 from *training* under density, not just validating [U]. Make sure the LightGBM training pairs come from the hidden-S1 world too, not only the threshold tuning.

---

## A. Code sketch for item 2 (record-side char-3-gram hybrid, plus prune protection)

```python
# Reuses the machinery already in block_big.py (sp_matmul_topn, group_rank). Per country; s1/r are the normalised frames.
import numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

def c3_reverse(s1, r, k=3, w=0.5, max_df_n=0.05, max_df_a=0.10, rows=200_000, chunk=20_000, threads=8):
    """Each S2/S3 record's top-k S1 by w*cos(char3 of space-less core name) + (1-w)*cos(address words).
    IDF fitted on S1 only (as the peers do). Returns (s1_i, r_i, score). Deterministic -> it is blocking."""
    ns = lambda d: ("<" + d.name_core.str.replace(" ", "", regex=False) + ">").tolist()
    vn = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=2, max_df=max_df_n, sublinear_tf=True, dtype=np.float32)
    va = TfidfVectorizer(analyzer=str.split, min_df=2, max_df=max_df_a, sublinear_tf=True, dtype=np.float32)
    a, b = np.float32(np.sqrt(w)), np.float32(np.sqrt(1 - w))
    St = sp.hstack([vn.fit_transform(ns(s1)) * a, va.fit_transform(s1.addr_norm.tolist()) * b]).T.tocsr()
    out_s, out_r, out_v = [], [], []
    for lo in range(0, len(r), rows):                      # bounded RAM: ~rows x ~35 nnz at a time
        d = r.iloc[lo:lo + rows]
        Q = sp.hstack([vn.transform(ns(d)) * a, va.transform(d.addr_norm.tolist()) * b]).tocsr()
        for c0 in range(0, Q.shape[0], chunk):
            C = sp_matmul_topn(Q[c0:c0 + chunk], St, top_n=k, n_threads=threads).tocoo()
            out_s.append(C.col.astype(np.int32)); out_r.append((C.row + lo + c0).astype(np.int32)); out_v.append(C.data)
    return np.concatenate(out_s), np.concatenate(out_r), np.concatenate(out_v)
```

**Integration steps**
1. **Pilot.** Time 100k India records and extrapolate.
   - Cost scales with Σ over each query's trigrams of their S1 posting lengths.
   - With max_df 0.05 we estimate about 5–20 min per country on 8 threads [E]. SABER's "hours" figure was the forward direction, with S2/S3 as the index.
   - If it is too slow, lower max_df to 0.03, but not below 0.02 (see SABER's table).
2. **Merge.** Merge the `(s1_i, r_i)` pairs into `pairs` as a new column `b_c3` (NaN when absent).
3. **Features.** Add its per-record rank and margin, like `_rmarg`.
4. **Protect the pairs in the prune** (change to `prune_by_similarity`):
   ```python
   keep = (group_rank(si, score) <= k) | (group_rank(ri, score) <= 2)   # record's top-2 S1 always survive
   # group_rank expects arrays; if it assumes sorted groups, sort by r_i first or use a pandas groupby-rank
   ```
5. **Run on train and test** (same `train_split` world), then retrain.
6. **Report** PC, oracle, cands/S1 per country for the methodology doc. The candidate-set size is graded [Q&A].

## B. Code sketch for item 3 (label-free word roles, from Epic021 G3; works on test France)

```python
# 1) Take each record's rank-1 S1 pair within the country, and keep those whose address token_set >= 90 (rapidfuzz cpdist).
# 2) y = 1 if the house numbers are equal, y = 0 if both are present and differ (decoys: 88% "differ"; positives: 65% "equal").
# 3) For each name token t: added = t in record name but not in the S1 name; dropped = the reverse.
#    lo_add[t] = log((a1+α)/(n1+2α)) - log((a0+α)/(n0+2α)), with min support 20. Same for dropped.
# 4) Pair features: add_lo_sum / add_lo_min / add_lo_max and drop_lo_* over the pair's added/dropped tokens (0 if none).
# Recompute the table on each split (train → train, test → test per country). No labels are used, so France gets its own table.
# Also: hn_absdiff bucket {0, 1-5, 6-20, 21-50, >50, missing}, hn_class {equal, prefix/truncated, suffix/dropped-leading,
#       leading-zero, 1-digit-sub, transposed, other}, nm_q_only, nm_s_only, nm_diff_ratio = fuzz.ratio(" ".join(q_only), " ".join(s_only)).
```

---

## Source index
**2026 teams** (all [U]):
- [Epic021 STRATEGY](https://github.com/Epic021/amazon-ml-challenge-2026/blob/main/STRATEGY.md) · [bottle_necks](https://github.com/Epic021/amazon-ml-challenge-2026/blob/main/bottle_necks.md) · [research_sota](https://github.com/Epic021/amazon-ml-challenge-2026/blob/main/research_sota.md) · [features.py](https://github.com/Epic021/amazon-ml-challenge-2026/blob/main/src/features.py)
- [AyanAhmedKhan RESEARCH_LOG](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/RESEARCH_LOG.md) · [FINAL_RECOMMENDATION](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/FINAL_RECOMMENDATION.md) · [FEATURE_CATALOG](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/FEATURE_CATALOG.md) · [ERROR_ANALYSIS](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/ERROR_ANALYSIS.md) · [synthetic_noise](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/research/open_scope/synthetic_noise.md) · [graph](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/research/notes/graph.md) · [fmeasure](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/research/notes/fmeasure.md) · [neural_em](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/research/notes/neural_em.md)
- [Siva402-ai methodology](https://github.com/Siva402-ai/amazon-ml-challenge-2026/blob/main/Documentation_template.md) · [README](https://github.com/Siva402-ai/amazon-ml-challenge-2026/blob/main/README.md)
- [Tony-AJ LEADERBOARD](https://github.com/Tony-AJ/business_entity_resolution/blob/main/LEADERBOARD.md) · [TRACKER](https://github.com/Tony-AJ/business_entity_resolution/blob/main/TRACKER.md)
- [Karthikatstuffmama README](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026/blob/main/README.md) · [EXPERIMENTS](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026/blob/main/EXPERIMENTS.md) · [FINDINGS](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026/blob/main/FINDINGS.md) · [LITERATURE](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026/blob/main/LITERATURE.md) · [PLAN](https://github.com/Karthikatstuffmama/amazon-ml-challenge-2026/blob/main/PLAN.md)
- [Team ICE REPORT](https://github.com/jeetsidhu/AMAZON-ML-Challenege/blob/main/docs/REPORT.md) · [blocking.py](https://github.com/jeetsidhu/AMAZON-ML-Challenege/blob/main/src/blocking.py)
- [adbhargav STRATEGY](https://github.com/adbhargav/amazon-ml-challenge/blob/claude/nifty-noether-ab17py/docs/STRATEGY.md)
- [SABER BLOCKING](https://github.com/yogeshwars-cys/Amazon-ML-2026-SABER/blob/main/docs/BLOCKING.md)
- [Arshbir1 results](https://github.com/Arshbir1/Amazon_ML_Challenge/blob/main/student_resource/business_entity_resolution/experiments/results.md)
- [Mounika-Reddy experiments](https://github.com/Mounika-Reddy-0802/Business-Entity-Resolution/blob/main/benchmarks/experiments.md)
- [VanshGupta18 SKILL](https://github.com/VanshGupta18/amazon-ml-template/blob/main/.claude/skills/amazon-er-challenge/SKILL.md)

**Literature** ([V] where read):
- [DeepMatcher SIGMOD'18](https://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf)
- [Ditto arXiv 2004.00584](https://arxiv.org/abs/2004.00584)
- [Papadakis et al. ICDE'24, arXiv 2307.01231](https://arxiv.org/abs/2307.01231)
- [Bipartite matching for CCER, arXiv 2112.14030](https://arxiv.org/abs/2112.14030)
- [GeCo CIKM'13](https://doi.org/10.1145/2505515.2507815)
- [FEBRL generator IDEAL'05](https://doi.org/10.1007/11508069_15)
- [Brill & Moore ACL'00](https://doi.org/10.3115/1075218.1075255)
- [Lipton et al. arXiv 1402.1892](https://arxiv.org/abs/1402.1892)
- [sbert cross-encoder efficiency](https://sbert.net/docs/cross_encoder/usage/efficiency.html)
- [sparse_dot_topn](https://github.com/ing-bank/sparse_dot_topn): "up to 6× faster … on Apple M2 Pro … 8 cores" [V]

**Official**: `research/official_QA.tsv`. It allows self-training and synthetic pairs, "any algorithm using only the provided records", and "small hand-written normalization dictionaries". It bans gazetteers, libpostal and hosted LLMs. Models must be MIT/Apache and ≤ 8B each.
