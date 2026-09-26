# Amazon ML Challenge 2026: refinement research for real_v2 → v3

Written 26 Sep 2026, about 03:50 IST (roughly 44 h before the 27 Sep 23:59 IST close). This was web research only: nothing was run on the data.

**Tags**
- **[V]** read in the cited source.
- **[E]** our estimate or inference.
- **[U]** a third party's claim that we did not verify.
- **[Q&A]** from the organisers' answers in `research/official_QA.tsv` (local copy of the Unstop Q&A).

Companion files: `01-challenge-intel.md` covers the rules and past editions, and `02-technical-playbook.md` covers the baseline design. This file does not repeat them.

---

## ⚠️ Read first: the candidate-file rule probably rules out the planned v3 design

The organisers answered this twice, on 25 Sep at 03:14 and 06:35 IST [Q&A]:

> "candidate_pairs.tsv should contain the final candidate set actually fed to your matching model for inference **(in a cascade, the input to the first scoring model)**. Every ID in matching_results.tsv must also appear in candidate_pairs.tsv."

The problem statement says the same thing: "the final candidate list *just before* the ML model scores them … whatever your model actually runs inference over" (`source/problem_statement.txt` l.126–131) [V].

**What this means for v3 [E]:**
- If the pruner is a *stage-1 LightGBM* (a scoring model), then `candidate_pairs.tsv` has to be that LightGBM's **input**, the ~35 per S1. It cannot be the ~5–10 it outputs.
- Reviewers will check the package ("top teams' packages are reviewed in detail"). A learned-model pruner reported as the candidate set is therefore a compliance risk.
- **So shrink the candidate set inside blocking, using deterministic rules** (rank and score rules, i.e. meta-blocking; see §1). LightGBM then stays the *first and only* scoring model.
- A grid search over a few rule thresholds on train is ordinary blocking tuning. A trained classifier is not.

Two questions are still unanswered in the Q&A: how the candidate-set size is weighed in the final ranking, and whether a pruned second stage is acceptable. Keep the F0.5 cost of shrinking the candidate set to about 0.002 or less.

---

## Do these next (ranked)

Impact estimates are for the private LB macro-F0.5 and for mean candidates per S1. Effort is wall-clock hours for one person. Test facts from the task brief:
- Test has **(4.9M + 5.1M) / 1.73M = 5.76 S2/S3 records per S1**, against **4.67 in train**.
- If the matches per S1 stay at 3.46, the **unmatched "decoy" records per S1 rise from about 1.21 to about 2.30 (×1.9)** [E, arithmetic].
- Team ICE measured the same ratio as ×2.02 ([REPORT §3, §7](https://github.com/jeetsidhu/AMAZON-ML-Challenege/blob/main/docs/REPORT.md)) [U].

| # | Change | Expected impact | Effort | Risk | Support |
|---|---|---|---|---|---|
| 1 | **Rule-based reciprocal meta-blocking inside blocking**, replacing the stage-1 GBDT pruner; add a state/département consistency filter (details below the table). | **35 → ~6–10 cands/S1**; oracle F0.5 −0.000…−0.003; it also removes the compliance risk. | 3–4 h | Low–med | GSM ([PVLDB 15(9)](https://www.vldb.org/pvldb/vol15/p1902-gagliardelli.pdf)); ICDE'23 filtering benchmark ([arXiv 2202.12521](https://arxiv.org/pdf/2202.12521)); Team ICE recall@k; Q&A |
| 2 | **Test-like validation**: hide **~19%** of the train S1 so that their S2/S3 records become decoys (details below the table). Re-tune every threshold (matching t1/t2, blocking floor) on that set. | +0.002–0.006 LB relative to thresholds tuned on the i.i.d. OOF (precision at ×1.9 decoys). | 1–2 h | Low | [Team ICE REPORT §3/§7/§9](https://github.com/jeetsidhu/AMAZON-ML-Challenege/blob/main/docs/REPORT.md); [adbhargav STRATEGY](https://github.com/adbhargav/amazon-ml-challenge/blob/claude/nifty-noether-ab17py/docs/STRATEGY.md) |
| 3 | **India script diagnostic, then a learned Indic→Latin token lexicon** (details below the table). | If Indic-script records lag: India PC +2–4 pts and **+0.003–0.01 overall** [E]. | 15 min diagnostic + 2–3 h | Med (leakage if not per-fold) | Team ICE: lexicon + leakage audit; adbhargav: "22% of Indian S2 and 12% of S3 names are Indic… token-aligned renderings… lexicon covers ~96% of test tokens" [U] |
| 4 | **Country-gated French normaliser**, using the block in §6 (details below the table). | France recall; +0.002–0.008 overall [E] (France is ~15% of test S1). | 1.5–2 h | Low if gated | [UPU France](https://www.upu.int/UPU/media/upu/PostalEntitiesFiles/addressingUnit/fraEn.pdf), [La Poste SP8855](https://lastation.laposte.fr/sites/p8_u1/files/2023-03/SP8855-Volume%202_V1.11%20-%20Adressage%20des%20plis.pdf); ICE §3 (ungated French rules broke India: "R K Puram → rue k puram") |
| 5 | **Fix the unseen-country threshold direction** (details below the table). | ±0.003–0.008. **The sign matters**, so measure it before submitting. | 1–2 h + 2 uploads | Med | ICE LOCO: held-out India best threshold 0.12 vs global 0.79; "stage 2 hurts under country shift (0.967 → 0.953)" [U, their data] |
| 6 | **France self-training** (details below the table). Validate the recipe on simulated LOCO first (US → India). | +0.003–0.01 overall [E] | 4–6 h | Med (confirmation bias) | Organisers explicitly allow self-training and synthetic pairs [Q&A]; PromptEM uncertainty-aware pseudo-labels ([arXiv 2207.04802](https://arxiv.org/abs/2207.04802)); co-training ([Blum & Mitchell 1998](https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf)) |
| 7 | **Features for the p 0.3–0.7 FN band**, i.e. random, handle and heavy-typo names (details below the table). | +0.003–0.01 [E] (FN 19.8k vs FP 5k per 120k S1 means recall headroom is where the points are) | 3–5 h | Low–med | ICE ablation: house-number and legal-form groups are "the two groups the model cannot do without"; [Cohen et al. 2003](https://pubs.dbs.uni-leipzig.de/dc/files/Cohen2003Acomparisonofstringdistance.pdf) (SoftTFIDF best); [Splink TF adjustments](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html) |
| 8 | **Calibrate (isotonic on OOF), then use an exact expected-F0.5 set per S1** after the record-exclusivity pre-pass. Compare with the current fixed 0.7 rule on the item-2 validation. | +0.001–0.003 [E] (ICE: the F0.5 curve is flat within 0.0005 for thresholds 0.6–0.9) | 1–2 h | Low | [GFM NIPS'11](https://proceedings.neurips.cc/paper_files/paper/2011/file/71ad16ad2c4d81f348082ff6c4b20768-Paper.pdf); [Waegeman et al. JMLR'14](https://www.jmlr.org/papers/v15/waegeman14a.html); code in `02-technical-playbook.md` §6.2 |
| 9 | *(Only if time is left)* **Model2Vec static multilingual embedding cosine** as one extra feature (details below the table). | +0.000–0.003 [E] | 1–2 h | Low | [model2vec](https://github.com/MinishLab/model2vec) ("up to 500 times faster on CPU"); [potion card](https://huggingface.co/minishlab/potion-multilingual-128M) |

**Details for each row**

1. **Reciprocal meta-blocking.**
   - Fuse the blockers into one score with reciprocal-rank fusion.
   - Keep a pair only if all three hold:
     - the record ranks it ≤ r (1–2)
     - the S1 ranks it ≤ k (8–15)
     - its score is ≥ ρ·(the record's best score)
   - Always keep each record's best S1 if it clears a floor.
   - Add the filter: drop a pair when both states or départements parse and disagree.
   - Grid-search (r, k, ρ, floor) on train. Pseudo-code is in §1.
2. **Test-like validation.** Hiding ~19% of train S1 turns their S2/S3 records into decoys, which reproduces 5.76 records per S1. The derivation: 4.67/(1−h) = 5.76 gives h ≈ 0.19.
3. **India lexicon.**
   - First split India PC and OOF-F0.5 by the script of the S2/S3 name (Latin vs Devanagari, Bengali, …).
   - If Indic-script rows lag, learn a token lexicon (Indic token → Latin S1 token) from aligned train ground-truth pairs.
   - Build it **per fold** for OOF, and on all of train for test.
4. **French normaliser.**
   - Street types, legal forms, elisions, Saint/St, bis/ter, CEDEX/BP stripping.
   - Numbers inside street names ("rue du 8 Mai 1945") must not become house numbers.
   - Mine the département ↔ name mapping from the test records themselves.
5. **Unseen-country threshold.**
   - Check on our LOCO whether the held-out country's optimal threshold is **lower** or higher than the global one. Team ICE found an unseen country **under-confident**: a *lower* best threshold.
   - Then set France's threshold by prior matching: France's predicted links per S1 should be about the expected France matched-record count per S1, estimated from test record counts per country (§4).
   - Stage-2 context features: turn them off for France.
   - Use **France-only LB A/B**: only French rows change between the two uploads.
6. **France self-training.**
   - Synthetic French positives: apply our own noise ops to French S1 records.
   - Pseudo-labels: positives at p ≥ 0.97 that are the record's top S1 with a 0.5 margin and low disagreement between fold models. Negatives come free from exclusivity: the other S1 candidates of a confidently matched record.
   - Retrain once with French pairs at weight 0.5.
7. **Features for the FN band.**
   - A name-randomness flag: share of out-of-vocabulary tokens against the same country's S1-name vocabulary, and the distance to the nearest S1 token.
   - Handle/domain word-break using S1-derived unigram frequencies.
   - char-3gram TF-IDF cosine on the no-space name and on the address.
   - An 8-way house-number relation.
   - Address uniqueness: how many S1 share the street and number.
   - A SoftTFIDF-style fuzzy IDF coverage.
8. No further detail beyond the table row.
9. **Model2Vec.** Use `minishlab/potion-multilingual-128M` (MIT, 128M). Do **not** run e5-small over all 11.7M test records on this Mac: at ~280 texts/s that is ~11 h.

**Don't do**
- A Hungarian or one-to-one assignment. The task is one-to-many, and per-record argmax exclusivity is the right constraint.
- A hosted LLM or a 7B judge on CPU.
- A stage-2 stacker for France.
- Any gazetteer: the Q&A explicitly bans "large static country/state/city tables".

**Upload plan (5 per day) [E]**
- **Today:**
  1. The pruned-candidate pipeline, with the matcher otherwise unchanged. The LB should move by no more than about 0.001; this is the compliance check.
  2. The item-2 thresholds.
  3. The item-5 France threshold variant (France-only A/B).
  4. The item-3 India lexicon.
  5. Keep one upload in reserve.
- **Tomorrow:** items 6 and 7, then the final submission by **about 21:00 IST**. Uploads of about 100 MB have hung on "Please Wait" for other teams [Q&A log], and in 2025 an 11:58 pm upload never reached the board ([01-challenge-intel §3](01-challenge-intel.md)).

---

## 1. Minimising the candidate set without losing recall

### 1.1 Findings
**Meta-blocking vocabulary** ([GSM, PVLDB 15(9) 2022](https://www.vldb.org/pvldb/vol15/p1902-gagliardelli.pdf); [TR arXiv 2204.08801](https://arxiv.org/abs/2204.08801)) [V]:
- Weight-based pruning:
  - WEP: a global average.
  - WNP: a per-node average.
  - RWNP: the pair must beat *both* nodes' averages.
  - BLAST: keeps a pair if p > r·(the sum of the two nodes' max probabilities), with r = 0.35.
- Cardinality-based pruning:
  - CEP: a global top-K.
  - CNP: top-k per node, where either endpoint suffices.
  - **RCNP**: the pair must be in the top-k of **both** endpoints.
- **BLAST is the best weight-based algorithm** (recall-oriented).
  - Its best features are {CF-IBF, RACCB, RS, NRS}.
  - It averages recall 0.882 and precision 0.193.
- **RCNP is the best cardinality-based algorithm** (precision-oriented). Against CNP it gives **−4.1% recall, +34.9% precision, +33.6% F1**.
  - Its best features are {CF-IBF, RACCB, JS, LCP, WJS}. LCP is the node's candidate count.
- **50 labelled pairs (25 per class) suffice** for their probabilistic classifier. More labels raise recall slightly but cut precision.
- Their block collections start at PC 0.95–0.99. After supervised meta-blocking, recall is about 0.85–0.88. Generic meta-blocking with tiny training sets therefore loses a lot of recall.
- **Our advantage** [E]: millions of labels and much richer per-pair signals. Only the *pruning shapes* (node-local, reciprocal, relative-to-max) transfer; their absolute recall numbers do not.

**Cardinality (local) thresholds beat similarity (global) thresholds** ([Papadakis et al., ICDE 2023](https://doi.org/10.1109/icde55515.2023.00389), [arXiv 2202.12521](https://arxiv.org/pdf/2202.12521)) [V]:
- kNN-Join ranks well above ε-Join (average position 3.2 vs 5.7) "because the latter apply a global condition, unlike the former, which operate locally, selecting the best candidates per query entity."
- The reverse direction ("RVS": which side is indexed) is a key parameter.
- At PC ≥ 0.9, the tuned cardinality threshold "does not exceed 26".
- Code and results: [ContinuousFilteringBenchmark](https://github.com/gpapadis/continuousfilteringbenchmark/).

**Other blockers.**
- Sparkly (TF-IDF/BM25 top-k) beats 8 state-of-the-art blockers, and IDF is essential ([PVLDB'23](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf)); already summarised in `02` §3.
- SC-Block: "smaller candidate sets" at 99.5% PC, and pipelines 1.5–8× faster ([arXiv 2303.03132](https://arxiv.org/abs/2303.03132)) [V abstract]. It needs a GPU-trained contrastive encoder, so it is not worth it now [E].

**Same-data evidence (other 2026 teams)**
- **Team ICE** retrieves the top S1 per S2/S3 record (the record side).
  - Recall@1, @3, @5, @10 = **0.979 / 0.988 / 0.991 / 0.993** on a 5% name-group sample.
  - They warn that the sample's S1 index is 20× smaller, so these numbers are optimistic.
  - 72% of retrieval misses have **no address** and 81% belong to **namesake chains** (base rates 4% and 40%) [U] ([REPORT §4, §8](https://github.com/jeetsidhu/AMAZON-ML-Challenege/blob/main/docs/REPORT.md)).
- adbhargav prunes a union of 5 paths to "≤20 / S1" with a cheap GBDT and writes that as `candidate_pairs.tsv`. That is the same compliance risk as our v3 plan [U] ([STRATEGY](https://github.com/adbhargav/amazon-ml-challenge/blob/claude/nifty-noether-ab17py/docs/STRATEGY.md)).
- Bishal-NITS writes "Top-35/S1", with 99.59% recall on a small split [U] ([repo](https://github.com/Bishal-NITS-2003/amazon-ml-challeneg-2026)).

### 1.2 Why record-side pruning is almost free here [E]
- Each S2/S3 record belongs to at most one S1, and the decision step already keeps only a record's best S1. So a pair (x, r) where x is not among r's top-2 S1 is nearly never predicted. Pruning it costs almost no *final* F0.5, even where it costs some *oracle* F0.5.
- Record-side top-r gives ≤ r × 5.76 candidates per S1 **on average**, decoys included. Examples: r=1 gives ≤5.8/S1 and r=2 gives ≤11.5/S1. Adding the S1-side cap k (RCNP) and a relative floor ρ cuts this further, most of all for decoys, whose best score is low.
- This is adaptive k for free:
  - S1s in chains get more candidates only when the scores are close.
  - Clean S1s get about their true count (3.46 on average).
- **State or département consistency.** adbhargav reports that the "US state agrees in 99.8% of true pairs when both parse" [U].
  - Rule: if both sides parse to different states (US/India) or different postcode départements (France: first 2 digits; first 3 for 97x), drop the pair. Keep it if either side is missing.
  - This targets exactly the namesake-chain candidates that inflate |C| and cause false positives. Verify the 99.8% on our train before applying it.

### 1.3 Drop-in rule (deterministic, so it is "blocking")
```python
import numpy as np, pandas as pd

def fuse(c, blockers=("r_comb", "r_name", "r_addr"), k0=60):
    """Reciprocal-rank fusion of the blockers' per-S1 ranks (1 = best; NaN = not retrieved by that blocker).
    Parameter-free (Cormack et al., SIGIR'09 RRF: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)."""
    return sum((1.0 / (k0 + c[b])).fillna(0.0) for b in blockers)

def prune(c, r_max=2, k_max=12, rho=0.7, floor=0.0):
    """c: one row per (s1, rec) from the union, with fused score s (higher = better).
    RCNP-style: keep if in the record's top-r AND the S1's top-k AND within rho of the record's best;
    always keep each record's best S1 if it clears the floor (exclusivity-safe)."""
    c = c.copy()
    c["rk_rec"] = c.groupby("rec")["s"].rank(ascending=False, method="first")
    c["rk_s1"] = c.groupby("s1")["s"].rank(ascending=False, method="first")
    best_rec = c.groupby("rec")["s"].transform("max")
    keep = (c.rk_rec <= r_max) & (c.rk_s1 <= k_max) & (c.s >= rho * best_rec) & (c.s >= floor)
    keep |= (c.rk_rec == 1) & (c.s >= floor)
    keep &= ~c["state_conflict"]          # both parsed & different -> drop (verify 99.8% first)
    return c[keep]
```

**Grid search.** Search r ∈ {1,2,3}, k ∈ {5,8,12,20}, ρ ∈ {0, .5, .7, .8}, and floor over score quantiles. Run it on the **item-2 hidden-S1 validation**, where decoy density matches test. For every point, report:
- mean, median and p95 candidates per S1
- the per-country share of S1 with an empty candidate list
- PC, PQ and RR
- the **oracle macro-F0.5** (formula in `02` §3.3)
- the **final macro-F0.5** after retraining the matcher on the pruned distribution. The within-S1 rank and gap features change, so retraining is required.

Pick the knee where final F0.5 loses ≤ 0.002. Put the PC-vs-candidates curve in the methodology doc [E].

**Typical achievable numbers.**
- Literature: fine-tuned cardinality methods reach PC ≥ 0.9 with k ≤ 26 ([ICDE'23](https://arxiv.org/pdf/2202.12521)) [V]. RCNP trades about 4% recall for about 35% precision against CNP [V].
- Our data [E]: from the current PC (India 0.918, US 0.959 at ~35/S1), expect ~6–10/S1 at a PC drop of ≤0.5 pt. Most of the lost pairs are outranked pairs that exclusivity would drop anyway.

---

## 2. Matching improvements for noisy multilingual names and addresses on CPU

### 2.1 What the evidence says adds most
**House numbers and legal forms dominate the ablation.** Team ICE's LightGBM ablation, with ΔF0.5 when the group is removed [U, their data]:

| Group removed | ΔF0.5 |
|---|---|
| House numbers | −0.00101 |
| Legal form | −0.00070 |
| Stage 2 | −0.00046 |
| Name strings | −0.00039 |

"Both act on recall, i.e. they rescue true pairs that the string similarities alone leave below the threshold" ([REPORT §11](https://github.com/jeetsidhu/AMAZON-ML-Challenege/blob/main/docs/REPORT.md)).
- Their strongest single features by AUC are `gap_second` 0.9926, the blocking score 0.9797, `cos_addr` 0.9595 and address Jaccard 0.959.
- Relative features (the gap to the second-best S1) beat absolute similarities [U].

**An 8-way house-number relation** instead of our tri-state. Team FaizJamal06 checks these classes in this order ([notes/HANDOFF_R3.md](https://github.com/FaizJamal06/Amazon-ML-26)) [U]:
1. both_missing
2. one_missing
3. equal
4. same_base_diff_suffix
5. transposed
6. digit_dropped
7. small_shift (≤50)
8. different

Add `hn_abs_diff_log`.
- adbhargav's measured residuals: "same street, nearby number (7751 vs 7755); true pairs with number noise (00412, 1515→151, ±1); ~18% of true pairs share no number because the candidate dropped it" [U].
- Encode *missing* separately from *different* (NaN vs 0) [U].

**Term-frequency-adjusted agreement (Splink/Fellegi–Sunter).**
- An exact agreement on a rare value carries more evidence. Splink adds a TF adjustment and, for fuzzy matches, uses "the greater of the two term frequencies" with a floor `tf_minimum_u_value` of 0.001 ([Splink docs](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html)) [V].
- In LightGBM terms [E], add:
  - `log(df)` of the rarest agreeing name token, taking the *max* df of the two spellings for fuzzy agreements
  - `log(freq)` of the full normalised name within the country (the chain signal)
  - `log(freq)` of the (street, number) key (the multi-tenant signal)

**SoftTFIDF and Monge-Elkan.**
- Cohen, Ravikumar & Fienberg found the hybrid **SoftTFIDF (TF-IDF + Jaro-Winkler token match)** best across name-matching tasks ([paper](https://pubs.dbs.uni-leipzig.de/dc/files/Cohen2003Acomparisonofstringdistance.pdf)) [V].
- Our IDF-weighted Jaccard or coverage is the *exact-token* version. Add the soft version:
  - For each S2 token, take the best JW match among the S1 tokens.
  - Count it if JW ≥ 0.9, weighted by its IDF.
  - Normalise by the total IDF.
- This covers typos and word shuffles together. `rapidfuzz.process.cdist` with `JaroWinkler` makes it vectorised [E].

**char-3gram TF-IDF cosine as a pair feature.** Our blocking keys are word-level and hashed. Compute these per pair:
- `analyzer='char'` cosines on the **no-space** name
- the address with punctuation stripped
- name + address

These are sparse row-dot products, a few µs per pair [E]. They repair handle and domain names ("UNIFIEDSECURITYWORKS.COM", "#unifiedsecurity") and glued or split tokens.

### 2.2 Random, handle and heavy-typo names (the FN band at p 0.3–0.7) [E]
**Name-randomness flag.** Compute, per record:
- the share of name tokens with zero document frequency in the **same country's S1 names**
- the minimum JW distance of those tokens to any token in the candidate S1 name

Typos stay close to an S1 token; generator words like "Lumhalo" or "belobrixlum" (example in [Bishal repo](https://github.com/Bishal-NITS-2003/amazon-ml-challeneg-2026) [U]) do not. With the flag, LightGBM can learn to lean on the address when the name is random.

**Address-only safety.** Compute `n_s1_same_street_number` (how many S1 share the normalised street and number) and the address-cosine margin to the second-best S1. A random name with a **unique** address and an equal house number and postcode is a safe link. A shared address (multi-tenant) is not. Team ICE also warns of near-copy decoys at the *same* address labelled unrelated (for example "Hashmi, Clotilda W., MD, DDS PC" vs "… MMD, DDS"). These cap precision [U].

**Handle and domain word-break.**
1. Strip `# @ .com www`.
2. Undo l33t (0→o, 1→i/l, 3→e, 4→a, 5→s, 7→t; adbhargav reports l33t digits in 5.6% of domain names [U]).
3. Viterbi word-break with **unigram frequencies from the country's S1 names**. This keeps it data-derived: `wordninja` is MIT but ships its own English word list.
4. Compare with the S1 core name: `partial_ratio`, and the share of the S1 no-space name's characters covered.

**Legal-form handling** (Team ICE §3 [U]):
- Strip "strict" forms anywhere (inc, ltd, llc, pvt, gmbh, sarl, sas, eurl).
- Strip "ambiguous" forms (co, pc, pa, sa, ag, lp, ei) only in legal position: trailing, after "&", or next to another legal token.
- Keep the stripped legal form as an agreement feature.
- Never delete country tokens inside names ("Air India", "Air France").
- **Never apply French rules outside France.**

### 2.3 Is a small multilingual embedding model worth it on CPU?
**Throughput** [V]:
- e5-small via sentence-transformers fp32 runs at "~280 texts/s" batch on an 8-core Ryzen 5800X3D ([nanoE5.c README](https://github.com/cnmoro/nanoe5.c/blob/main/README.md)).
- Another harness, pinned to 2 CPUs: MiniLM-L6 at 137.8 texts/s, **multilingual-e5-small at 67.6/s**, multilingual-e5-base at 21.3/s ([benchmark](https://www.tiespecialistas.com.br/en/minilm-vs-multilingual-e5-embedding-benchmark/)).
- Elastic's optimised export is about 54% faster for inputs under 50 characters ([card](https://huggingface.co/elastic/multilingual-e5-small-optimized/blob/main/README.md)).

**What that means here** [E]. Embeddings are per *record*, not per pair; 60M pair cosines cost nothing once the records are embedded. But ~11.7M test records plus the train sample at ~300–1,000 texts/s is **4–11 h** on this Mac, while the pipeline also needs its 8 GB. That is not worth it for synthetic character-level noise, where char n-grams already do the work.

**The cheap option.** **Model2Vec** static embeddings:
- MIT library; its "only major dependency is `numpy`"; "up to 500 times faster on CPU" ([repo](https://github.com/MinishLab/model2vec)) [V]
- `potion-multilingual-128M` is MIT, 128M params, distilled from bge-m3, 101 languages ([card](https://huggingface.co/minishlab/potion-multilingual-128M)) [V]

Expected value is small (+0–0.003). Use it only as a feature for the Indic-script and French residue after items 3 and 4.

---

## 3. Decision and set selection for macro F0.5 per entity

**Theory** (details and verified code in `02` §6).
- The exact expected-F maximiser needs only m²+1 parameters (GFM) and allows F(∅,∅)=1, which is our singleton rule ([NIPS'11](https://proceedings.neurips.cc/paper_files/paper/2011/file/71ad16ad2c4d81f348082ff6c4b20768-Paper.pdf)).
- Surrogates such as Hamming loss or thresholding marginals have "high worst-case regret" in general. The GFM-style algorithm is "Bayes-optimal, regardless of the underlying distribution" ([Waegeman et al. JMLR 2014](https://www.jmlr.org/papers/v15/waegeman14a.html)) [V].
- A plug-in rule is consistent, whereas structured-SVM approaches are not ([Dembczyński et al. ICML'13](https://proceedings.mlr.press/v28/dembczynski13.html)) [V].
- Under independence the optimum is the empty set or the top-k by marginal. So with calibrated p, the sweep over k in `02` §6.2 is exact.

**How much it adds here.** Probably little [E]:
- Team ICE's calibrated sweep is flat: 0.9904 at 0.6, 0.99088 at 0.79, 0.9902 at 0.9.
- Fold-optimal thresholds span 0.70–0.79 and gain ≤ 0.00009 over the global one [U].
- Expect +0.001–0.003 from calibration plus the expected-F set. Most of it comes from S1s whose top-1 is 0.5–0.8, where "empty vs top-1" matters on singletons.

**Calibration.** Team ICE: Platt cut ECE from 0.00086 to 0.00032, isotonic to 0.00025. Raw LightGBM is over-confident (Platt a = 0.805) [U].

**Density shift.**
- At ×2 and ×3 decoys, ICE's training threshold had regret ≤ 0.0007, and the prior-corrected threshold (0.835) cost 0.0002 [U].
- So the item-2 re-tune is insurance. The larger risk is **France**, not decoys.

**Exclusivity.**
- The task is one-to-many, with each record going to at most one S1. The exact projection is a per-record argmax before the per-S1 set choice.
- The Hungarian or one-to-one algorithms from the bipartite-matching study ([Papadakis et al., arXiv 2112.14030](https://arxiv.org/abs/2112.14030)) assume one-to-one per source, which does not hold here (mean 3.46 matches per S1).
- Order that works [E]:
  1. calibrate
  2. per-record argmax (keep the runner-up margin as a feature)
  3. per-S1 expected-F0.5 set

**Collective ER (S2↔S3 transitivity).**
- Our stacker gave +0.001.
- ICE: stage 2 gives +0.0003 i.i.d. but **−0.014 under country shift** (0.9667 → 0.9530) [U].
- Keep collective features out of the French path.

---

## 4. Domain shift to an unseen France

### 4.1 What to expect
**Team ICE LOCO** ([REPORT §5](https://github.com/jeetsidhu/AMAZON-ML-Challenege/blob/main/docs/REPORT.md)) [U]:
- Held-out India (trained on US): 0.9047 at the global threshold, **0.9299 at its own best threshold 0.12**.
- Held-out US (trained on India): 0.9853 at the global threshold, 0.9861 at its own 0.83.
- "The threshold does not transfer across countries… a model that has never seen a country is under-confident on it."
- **Their public LB was ~0.95 against an i.i.d. OOF of 0.9725.** They read the gap as a French deficit.

**Implication for us.**
- Our "+0.05 threshold for unseen countries" may push France the wrong way. Check the direction on our own LOCO (US → India and India → US) before keeping it.
- France is Latin script, like the US, so US → India overstates the gap [E].

**Measure the France priors from the test files** (no labels needed) [E]:
- France records per S1: `n(S2∪S3, country=France) / n(S1, France)`. Compare with US and India in test.
- Expected French matched records per S1 ≈ the US/India value (3.46) scaled by that ratio's excess over the decoy level.
- **Prior matching:** choose France's t1/t2 so that predicted links per French S1 fall between 0.9× and 1.0× that expectation, erring low for F0.5. Confirm with one France-only LB A/B.

### 4.2 Self-training and synthetic pairs (explicitly allowed [Q&A])
- The organisers answered: "Self-training and generating synthetic pairs from the provided records are also fine." They also allow unsupervised statistics on test (TF-IDF, frequencies).
- **Pseudo-label selection.** PromptEM ranks unlabelled pairs by uncertainty (the standard deviation of 10 MC-dropout passes) and keeps the least uncertain fraction, rather than the most confident ([arXiv 2207.04802](https://arxiv.org/abs/2207.04802), §4.2) [V].
  - GBDT analogue [E]: use the standard deviation across our 5 fold models.
  - Positives: p̄ ≥ 0.97, std ≤ 0.02, the record's top S1 with a margin ≥ 0.5 over the runner-up, and house-number relation ∈ {equal, both_missing}.
  - Negatives: every *other* S1 candidate of a record that has a confident positive. These are guaranteed by the at-most-one rule and are hard negatives. Add pairs with p ≤ 0.01 inside the top-5.
- **Co-training variant** [E] ([Blum & Mitchell 1998](https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf)): a name-view model labels pairs for the address-view model, and vice versa. It targets France's random-name and French-address failure modes.
- **Synthetic French positives** [E]:
  - Apply our own noise catalogue to French S1 records:
    - typos, word shuffle, brackets
    - accent add or drop
    - legal-form add or drop (SARL, SAS…)
    - street-type abbreviation (§6)
    - Saint ↔ St
    - bis/ter ↔ letter
    - CEDEX add or drop
    - leading zeros
    - handle or domain versions
    - random-word name (keeping the address)
  - Negatives come from the French blocking neighbourhood.
  - RobEM warns that augmentation alone does not fix shift, so pair it with pseudo-labels ([CIKM'22](https://ehsk.github.io/assets/pdf/CIKM22_RobustEM.pdf), cited in `02` §7).
- **Validate the recipe offline.** Run the identical procedure on "US-trained → India-unlabelled" and keep it only if India F0.5 rises. Then apply it to France. One round, French weight 0.5.
- **DA literature.** DAME (a mixture of experts over source domains) shows zero-shot transfer and fine-tuning gains for PLM matchers ([arXiv 2204.09244](https://arxiv.org/abs/2204.09244)). DADER ([PVLDB demo](https://www.vldb.org/pvldb/vol15/p3666-fan.pdf)) aligns features, which helps when the source does poorly on the target [V abstracts]. Both need neural matchers, so do not adopt them now [E].

### 4.3 French address format (for the normaliser and the house-number parser) [V]
Official sources: [UPU France](https://www.upu.int/UPU/media/upu/PostalEntitiesFiles/addressingUnit/fraEn.pdf); [UPU thoroughfare types](https://www.upu.int/UPU/media/upu/documents/PostCode/Thorougfare-Types-and-Abbreviations.pdf); [La Poste SP8855](https://lastation.laposte.fr/sites/p8_u1/files/2023-03/SP8855-Volume%202_V1.11%20-%20Adressage%20des%20plis.pdf).

- **Line order** runs from name to locality, in at most 6 lines of 38 characters:
  1. company name
  2. recipient or service
  3. building, résidence or ZI (inside and outside the building)
  4. **number + street type + street name** ("25 RUE DE L EGLISE")
  5. BP, lieu-dit or commune
  6. **5-digit postcode + locality** ("33380 MIOS"), or "33506 LIBOURNE CEDEX"
- There is no punctuation from line 4 onward, so there are no commas after the number.
- **Postcode.** 5 digits to the left of the locality; the first two digits are the **département** (the UPU coding diagram labels 33 as the department). Overseas postcodes start 97x or 98x, so use 3 digits there [E].
- **CEDEX** (business mail) may follow the town, with an optional 1–2-digit number ("34092 MONTPELLIER CEDEX 5"). Strip it for matching.
- **BP nnnnn** (boîte postale) is a PO box. Strip it too, or keep it as a separate token.
- **SAINT and SAINTE become ST and STE** in the locality lines ("SAINT AGNANT → ST AGNANT").
- Official street-type abbreviations: ALL, AV, BD, CTRE, CCAL, IMM, IMP, LD, LOT, PAS, PL, RES, RPT, RTE, SQ, VLGE, ZA, ZAC, ZAD, ZI; plus CHEM (chemin) and SENT (sentier) in the UPU list.
- Other official abbreviations: BAT, BP, CP, APP, ETG, ENT, ETS, CIE, SOC, ST/STE/STS/STES, GD/GDE, PT/PTE, ND (Notre Dame), HT, VX, UNIV.
- Common informal ones (not in the official list [E]): R (rue), CH (chemin), CRS (cours), FG/FBG (faubourg), QU (quai), BLD/BVD (boulevard).
- **Numbers inside street names are not house numbers**: "Rue du 4 Septembre", "Avenue du 8 Mai 1945", "Place du 11 Novembre" [E]. Take the house number as the digit run immediately **before** a street-type token, or at the start of the address line.
- **House-number suffixes:** bis, ter and quater (and B, T, Q) become one canonical suffix. "12 bis" = "12B" = "12 B" [E].
- **Paris, Lyon and Marseille arrondissements** appear as 75001–75020, 69001–69009 and 13001–13016, or as "Paris 2e / 2ème / IIe". Normalise ordinals (1er, 2e, 2eme, 2ème) to digits [E].
- **Départements and regions.** adbhargav reports France records carry département *names* ("Nord", "Gironde") rather than regions [U]. A 101-row département table is close to the Q&A's banned "large static … state … tables".
  - **Mine the mapping from the test records instead.** For each trailing admin token in French addresses, count co-occurrence with the postcode's 2-digit prefix. Keep a token → département mapping when one prefix holds ≥ 90% of its occurrences. That uses only provided records, which is allowed [E].

---

## 5. Past Amazon ML Challenges and 2026 chatter

**Score levels found for 2026** (as of 26 Sep, about 03:30 IST).
- **Team ICE: public LB ≈ 0.95** for a full-data version with i.i.d. OOF 0.9725 [U] ([REPORT §5](https://github.com/jeetsidhu/AMAZON-ML-Challenege/blob/main/docs/REPORT.md)). This is the only public LB number we found.
- Every other 2026 figure is validation on small samples, so it is **not comparable**:

  | Source | Claimed score | Notes |
  |---|---|---|
  | [Aamod007](https://github.com/Aamod007/Amazon-ML-Challenge-2026-Business-Entity-Resolution) | 0.980 | On 5,258 ground-truth pairs [U] |
  | [Bishal-NITS](https://github.com/Bishal-NITS-2003/amazon-ml-challeneg-2026) | 0.9687 | Falls to 0.9602 on a "full distractor test" [U] |
  | [guruprasath-s-b](https://github.com/guruprasath-s-b/Business-Entity-Resolution) | 0.9189 | On 5k S1 [U] |
  | adbhargav | 0.976 | Synthetic data only [U] |

- Our OOF of 0.946, if ICE's OOF-to-LB gap holds (−0.02), suggests a first LB around 0.92–0.93 [E]. Treat that as a calibration point for our first upload, not a target.

**Useful peer practices** (all [U]):
- Namesake-grouped folds: 38% of S1 share an exact name with another S1 in the same country, so random folds leak.
- A per-fold Indic lexicon (it leaked when built on all labels).
- A shuffled-label canary (OOF AUC ≈ 0.5).
- A "hidden 15% S1" validation to mimic test decoys.
- France-gated rules only.

Sources: ICE REPORT §2–3; adbhargav STRATEGY.

**Platform lessons.**
- Several 2026 teams report ~100 MB `matching_results.tsv` uploads stuck at "Please Wait… 0" for over 30 min (Q&A log, 25 Sep 10:47 and 14:46).
- In 2025, uploads at 11:58 pm missed the board, and final ranks moved slightly after the official evaluation (for example 22 → 17, [LinkedIn](https://www.linkedin.com/posts/krishnamohan07_amazonmlchallenge-machinelearning-hackathon-activity-7384545950347988993-D_FL); 29 → 26, [LinkedIn](https://www.linkedin.com/posts/akshit-manocha-503150285_from-rank-29-26-on-the-final-leaderboard-activity-7387328296482635776-nElB)).
- In 2023 and 2024 the leaderboard #1 team won ([01-challenge-intel §4.3](01-challenge-intel.md)).
- France is in both the public and private splits [Q&A]. The LB ranks each team's max score, and ties go to the earlier upload (`01` §0).
- Past winners' approaches (2024 VLM extraction, 2025 price regression) don't transfer to this task. The transferable lessons are to ensemble near the end and to submit early (`01` §4.2).

---

## Licence check (every model or library mentioned)

Licences come from the HF model API tags and PyPI metadata, checked 26 Sep 2026. Params are the safetensors totals.

| Name | Type | Licence | Params | OK? | Source |
|---|---|---|---|---|---|
| LightGBM | GBDT library | MIT | n/a | ✓ | [GitHub](https://github.com/microsoft/LightGBM) |
| intfloat/multilingual-e5-small | embedder | MIT | 117.7M | ✓ (slow on this CPU) | [HF](https://huggingface.co/intfloat/multilingual-e5-small) |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | embedder | Apache-2.0 | 117.7M | ✓ | [HF](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) |
| minishlab/potion-multilingual-128M | static embedder (Model2Vec, distilled from bge-m3) | MIT | 128.1M | ✓ **cheapest** | [HF](https://huggingface.co/minishlab/potion-multilingual-128M) |
| minishlab/potion-base-8M | static embedder, English | MIT | 7.6M | ✓ | [HF](https://huggingface.co/minishlab/potion-base-8M) |
| model2vec | library | MIT | n/a | ✓ | [GitHub](https://github.com/MinishLab/model2vec) |
| sentence-transformers/LaBSE | embedder | Apache-2.0 | 470.9M | ✓ (too slow here) | [HF](https://huggingface.co/sentence-transformers/LaBSE) |
| BAAI/bge-m3 | embedder | MIT | ~568M | ✓ (too slow here) | [HF](https://huggingface.co/BAAI/bge-m3) |
| cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | cross-encoder | Apache-2.0 | 117.6M | ✓ (GPU-only in practice) | [HF](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) |
| almanach/camembert-base | French encoder | MIT | 111.2M | ✓ (not recommended now) | [HF](https://huggingface.co/almanach/camembert-base) |
| indic-transliteration 2.3.82 | library | MIT | n/a | ✓ | [PyPI](https://pypi.org/project/indic-transliteration/) |
| anyascii 0.3.3 | library (any script → ASCII) | ISC (permissive) | n/a | ✓ as a pure algorithm plus a transliteration table [E]; Bishal-NITS uses it | [PyPI](https://pypi.org/project/anyascii/) |
| rapidfuzz 3.14.6 | library | MIT | n/a | ✓ | [PyPI](https://pypi.org/project/rapidfuzz/) |
| sparse-dot-topn 1.2.0 | library | Apache-2.0 | n/a | ✓ | [PyPI](https://pypi.org/project/sparse-dot-topn/) |
| wordninja 2.0.0 | library | MIT per GitHub; PyPI metadata is empty | n/a | ⚠️ bundles an English word list; prefer our S1-derived vocabulary | [GitHub](https://github.com/keredson/wordninja) |
| libpostal | address parser | MIT | n/a | ❌ **banned by name** in the Q&A (bundles postal data) | [Q&A] |

---

## 6. French normalisation dictionary (paste-ready)

- This is generic linguistic knowledge only: no city, postcode or département lists.
- **Gate every rule on `country == France`**, or on the open-set label you map to France. Ungated rules broke Indian and US records for Team ICE ("R K Puram → rue k puram"; `DE` and `LA` state codes deleted) [U].
- In the US, "St" means Street, so the `st → saint` rule must stay France-only.

```python
import re, unicodedata

# --- canonical street types (key) <- variants seen in data (lowercased, accents stripped, dots removed) ---
# [V] = La Poste / UPU official abbreviation; others are common informal variants [E]
FR_STREET_TYPES = {
    "rue":        ["rue", "r"],
    "avenue":     ["avenue", "av", "ave", "avn", "aven"],          # AV [V]
    "boulevard":  ["boulevard", "bd", "bld", "blvd", "bvd", "boul"],# BD [V]
    "place":      ["place", "pl"],                                  # PL [V]
    "chemin":     ["chemin", "chem", "ch", "che", "chm"],           # CHEM [V]
    "impasse":    ["impasse", "imp"],                               # IMP [V]
    "allee":      ["allee", "allees", "all", "al"],                 # ALL [V]
    "route":      ["route", "rte", "rt"],                           # RTE [V]
    "quai":       ["quai", "qu", "qai"],
    "cours":      ["cours", "crs"],
    "faubourg":   ["faubourg", "fg", "fbg", "faub"],
    "square":     ["square", "sq"],                                 # SQ [V]
    "passage":    ["passage", "pas", "pass", "psg"],                # PAS [V]
    "residence":  ["residence", "res", "resid"],                    # RES [V]
    "rond point": ["rond point", "rondpoint", "rpt", "rd pt"],      # RPT [V]
    "lotissement":["lotissement", "lot", "lotiss"],                 # LOT [V]
    "lieu dit":   ["lieu dit", "lieudit", "ld"],                    # LD [V]
    "hameau":     ["hameau", "ham"],
    "sentier":    ["sentier", "sent", "sen"],                       # SENT [V]
    "promenade":  ["promenade", "prom"],
    "esplanade":  ["esplanade", "espl"],
    "village":    ["village", "vlge", "vge"],                       # VLGE [V]
    "villa":      ["villa", "vla"],
    "traverse":   ["traverse", "trav", "tra"],
    "montee":     ["montee", "mte"],
    "chaussee":   ["chaussee", "chs", "chau"],
    "carrefour":  ["carrefour", "carr", "car"],
    "galerie":    ["galerie", "gal"],
    "parvis":     ["parvis", "parv"],
    "cite":       ["cite"],
    "clos":       ["clos"],
    "domaine":    ["domaine", "dom"],
    "port":       ["port"],
    "voie":       ["voie"],
    "centre commercial": ["centre commercial", "ccal", "c cial", "cc"],   # CCAL [V]
    "centre":     ["centre", "ctre"],                               # CTRE [V]
    "immeuble":   ["immeuble", "imm"],                              # IMM [V]
    "zone industrielle": ["zone industrielle", "zi", "z i"],        # ZI [V]
    "zone d activite":   ["zone d activite", "zone d activites", "za", "z a"],  # ZA [V]
    "zone artisanale":   ["zone artisanale"],                       # also abbreviated ZA -> keep 'za' token
    "zac":        ["zone d amenagement concerte", "zac"],           # ZAC [V]
    "zad":        ["zone d amenagement differe", "zad"],            # ZAD [V]
}
# Map every variant -> short canonical token (use the SHORT form so both sides collapse to the same token)
FR_STREET_CANON = {"rue":"rue","avenue":"av","boulevard":"bd","place":"pl","chemin":"ch","impasse":"imp",
    "allee":"all","route":"rte","quai":"quai","cours":"crs","faubourg":"fg","square":"sq","passage":"pas",
    "residence":"res","rond point":"rpt","lotissement":"lot","lieu dit":"ld","hameau":"ham","sentier":"sent",
    "promenade":"prom","esplanade":"espl","village":"vlge","villa":"vla","traverse":"trav","montee":"mte",
    "chaussee":"chs","carrefour":"carr","galerie":"gal","parvis":"parv","cite":"cite","clos":"clos",
    "domaine":"dom","port":"port","voie":"voie","centre commercial":"ccal","centre":"ctre","immeuble":"imm",
    "zone industrielle":"zi","zone d activite":"za","zone artisanale":"za","zac":"zac","zad":"zad"}

# --- other address words: building / delivery / honorific words in street names ---
FR_ADDR_WORDS = {   # variant -> canonical
    "batiment":"bat", "bat":"bat", "bt":"bat", "escalier":"esc", "esc":"esc", "etage":"etg", "etg":"etg",
    "appartement":"app", "appt":"app", "apt":"app", "app":"app", "entree":"ent", "ent":"ent",
    "boite postale":"bp", "bp":"bp", "case postale":"cp", "cedex":"cedex",
    "saint":"st", "st":"st", "sainte":"ste", "ste":"ste", "saints":"sts", "sts":"sts", "saintes":"stes", "stes":"stes",
    "notre dame":"nd", "nd":"nd", "grand":"gd", "gd":"gd", "grande":"gde", "gde":"gde",
    "petit":"pt", "pt":"pt", "petite":"pte", "pte":"pte", "haut":"ht", "vieux":"vx",
    "general":"gal", "gen":"gal", "gal":"gal", "marechal":"mal", "mal":"mal", "president":"pdt", "pdt":"pdt",
    "docteur":"dr", "dr":"dr", "professeur":"pr", "prof":"pr", "commandant":"cdt", "cdt":"cdt",
    "colonel":"col", "capitaine":"cne", "cne":"cne", "lieutenant":"lt", "monseigneur":"mgr", "mgr":"mgr",
    "universite":"univ", "univ":"univ",
}
FR_STOP = {"le","la","les","l","de","du","des","d","au","aux","a","en","et","sur","sous","lez","les"}  # drop only for token-set features, keep in raw strings

# --- legal forms (strip from core name; keep as a 'legal_form' feature) ---
FR_LEGAL_STRICT = {  # safe to strip anywhere in the name
    "sarl","eurl","sas","sasu","selarl","selas","selafa","selca","sel","snc","scs","sca","sci","scp","scm",
    "scop","scic","scea","gaec","earl","gie","eirl","saem","sem","spfpl",
    "societe a responsabilite limitee","entreprise unipersonnelle a responsabilite limitee",
    "societe par actions simplifiee","societe par actions simplifiee unipersonnelle","societe anonyme",
    "societe en nom collectif","societe civile immobiliere","societe civile professionnelle",
    "societe d exercice liberal","groupement d interet economique","micro entreprise","auto entrepreneur",
    "entreprise individuelle",
}
FR_LEGAL_AMBIGUOUS = {"sa","ei","se"}   # strip only in legal position (trailing / next to another legal token)
FR_COMPANY_WORDS = {  # generic words the generator may add/drop; canonicalise, keep for similarity
    "societe":"ste", "ste":"ste", "etablissements":"ets", "etablissement":"ets", "ets":"ets",
    "compagnie":"cie", "cie":"cie", "et cie":"cie", "groupe":"groupe", "freres":"freres", "fr":"freres",
    "fils":"fils", "et fils":"fils", "holding":"holding", "entreprise":"entr", "entr":"entr",
    "international":"intl", "internationale":"intl", "services":"svcs", "service":"svcs",
}

# --- regexes ---
RE_ELISION   = re.compile(r"\b([ldjmnstc]|qu)['’`´]\s*", re.I)            # l'eglise -> eglise, d'orsay -> orsay
RE_CEDEX     = re.compile(r"\bcedex(?:\s*\d{1,2})?\b", re.I)
RE_BP        = re.compile(r"\b(?:bp|b\.p\.|boite postale|cs|tsa)\s*\d{1,6}\b", re.I)
RE_FR_PC     = re.compile(r"\b(\d{5})\b")                                   # dept = pc[:2], or pc[:3] if pc starts with 97/98
RE_NUM_SUFFIX= re.compile(r"\b(\d{1,5})\s*(bis|ter|quater|b|t|q)\b", re.I)  # 12 bis / 12b / 12 B -> 12b
RE_ORDINAL   = re.compile(r"\b(\d{1,2})\s*(?:er|re|e|eme|ème|ieme|ième)\b", re.I)  # 1er, 2e, 2eme -> 1, 2
RE_LEAD0     = re.compile(r"\b0+(\d)")

SUFFIX_CANON = {"bis":"b","b":"b","ter":"t","t":"t","quater":"q","q":"q"}

def strip_accents(s: str) -> str:
    s = s.replace("œ", "oe").replace("Œ", "oe").replace("æ", "ae").replace("Æ", "ae")
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def fr_norm_address(a: str) -> str:
    """France-only. Order matters: accents -> elisions -> cedex/bp -> punctuation -> suffixes -> abbreviations."""
    a = strip_accents(a).lower()
    a = RE_ELISION.sub("", a)
    a = RE_CEDEX.sub(" ", a); a = RE_BP.sub(" ", a)
    a = re.sub(r"[-_/,.;:()\[\]#@]+", " ", a)
    a = RE_NUM_SUFFIX.sub(lambda m: m.group(1) + SUFFIX_CANON[m.group(2).lower()], a)
    a = RE_ORDINAL.sub(r"\1", a); a = RE_LEAD0.sub(r"\1", a)
    toks = a.split(); out = []; i = 0
    # greedy longest-match over multi-word variants (max 4 words)
    variant2canon = {v: FR_STREET_CANON[k] for k, vs in FR_STREET_TYPES.items() for v in vs}
    variant2canon.update(FR_ADDR_WORDS)
    while i < len(toks):
        for n in (4, 3, 2, 1):
            key = " ".join(toks[i:i+n])
            if key in variant2canon:
                out.append(variant2canon[key]); i += n; break
        else:
            out.append(toks[i]); i += 1
    return " ".join(out)

def fr_house_number(a_norm: str):
    """House number = digit run (with optional b/t/q suffix) immediately BEFORE a street-type token, else at line start.
    Avoids 'rue du 8 mai 1945' -> 8. Returns None if not found."""
    street = set(FR_STREET_CANON.values())
    toks = a_norm.split()
    for j in range(len(toks) - 1):
        if re.fullmatch(r"\d{1,5}[btq]?", toks[j]) and toks[j+1] in street:
            return toks[j]
    return toks[0] if toks and re.fullmatch(r"\d{1,5}[btq]?", toks[0]) else None
```

---

## Source index
- **Official, local:**
  - `research/official_QA.tsv` answers of 25 Sep 03:14 and 06:35 IST: the cascade clause, self-training and synthetic pairs allowed, per-model 8B limit, the libpostal/gazetteer ban, "small hand-written normalization dictionaries" allowed.
  - `source/problem_statement.txt` l.126–131.
- **Meta-blocking and filtering:**
  - [GSM PVLDB 15(9)](https://www.vldb.org/pvldb/vol15/p1902-gagliardelli.pdf); [GSM TR](https://arxiv.org/abs/2204.08801)
  - [ICDE'23 filtering benchmark](https://doi.org/10.1109/icde55515.2023.00389), [arXiv 2202.12521](https://arxiv.org/pdf/2202.12521), [code](https://github.com/gpapadis/continuousfilteringbenchmark/)
  - [Sparkly PVLDB'23](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf)
  - [SC-Block](https://arxiv.org/abs/2303.03132)
  - [RRF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
  - [bipartite matching for CCER](https://arxiv.org/abs/2112.14030)
- **Matching:**
  - [Cohen et al. 2003](https://pubs.dbs.uni-leipzig.de/dc/files/Cohen2003Acomparisonofstringdistance.pdf)
  - [Splink TF adjustments](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html)
  - [nanoE5.c throughput](https://github.com/cnmoro/nanoe5.c/blob/main/README.md)
  - [MiniLM vs e5 CPU benchmark](https://www.tiespecialistas.com.br/en/minilm-vs-multilingual-e5-embedding-benchmark/)
  - [Elastic e5-small optimised](https://huggingface.co/elastic/multilingual-e5-small-optimized/blob/main/README.md)
  - [Model2Vec](https://github.com/MinishLab/model2vec); [potion-multilingual-128M](https://huggingface.co/minishlab/potion-multilingual-128M)
- **Decision:**
  - [GFM NIPS'11](https://proceedings.neurips.cc/paper_files/paper/2011/file/71ad16ad2c4d81f348082ff6c4b20768-Paper.pdf)
  - [Waegeman JMLR'14](https://www.jmlr.org/papers/v15/waegeman14a.html)
  - [Dembczyński ICML'13](https://proceedings.mlr.press/v28/dembczynski13.html)
  - Lipton et al. [arXiv 1402.1892](https://arxiv.org/abs/1402.1892)
- **Domain shift:**
  - [PromptEM](https://arxiv.org/abs/2207.04802)
  - [DAME](https://arxiv.org/abs/2204.09244)
  - [DADER demo](https://www.vldb.org/pvldb/vol15/p3666-fan.pdf)
  - [co-training](https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf)
  - [RobEM](https://ehsk.github.io/assets/pdf/CIKM22_RobustEM.pdf)
- **France:**
  - [UPU France addressing](https://www.upu.int/UPU/media/upu/PostalEntitiesFiles/addressingUnit/fraEn.pdf)
  - [UPU thoroughfare types](https://www.upu.int/UPU/media/upu/documents/PostCode/Thorougfare-Types-and-Abbreviations.pdf)
  - [La Poste SP8855](https://lastation.laposte.fr/sites/p8_u1/files/2023-03/SP8855-Volume%202_V1.11%20-%20Adressage%20des%20plis.pdf)
  - [legal forms overview (Stripe)](https://stripe.com/resources/more/business-legal-types-france)
  - [legal forms table](https://www.lecoindesentrepreneurs.fr/tableau-comparatif-formes-juridiques-ei-eirl-eurl-sasu-sarl-sas/) (EIRL has been closed to new businesses since Feb 2022)
- **2026 peers** (all [U]):
  - [Team ICE REPORT](https://github.com/jeetsidhu/AMAZON-ML-Challenege/blob/main/docs/REPORT.md)
  - [adbhargav STRATEGY](https://github.com/adbhargav/amazon-ml-challenge/blob/claude/nifty-noether-ab17py/docs/STRATEGY.md)
  - [FaizJamal06](https://github.com/FaizJamal06/Amazon-ML-26)
  - [Bishal-NITS](https://github.com/Bishal-NITS-2003/amazon-ml-challeneg-2026)
  - [Aamod007](https://github.com/Aamod007/Amazon-ML-Challenge-2026-Business-Entity-Resolution)
  - [HF mirror of the problem statement](https://huggingface.co/datasets/logicalguy/amazon-ml-challenge-2026)
