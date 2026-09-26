# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** pass the J
**Team Members:** Anshu Aman (Team Leader), Rishav Tiwari
**Submission Date:** [to fill at final submission]

---

## 1. Executive Summary
We resolve Source-2/3 records against the deduplicated Source-1 reference with a three-part pipeline:
1. **Rare-key hashed blocking.** A forward search (each S1's best records) and a reverse search (each S2/S3 record's best S1) run over IDF-weighted name, address and combined keys.
2. **Two-stage LightGBM matcher.** Language-agnostic string-similarity features feed the first stage. A collective second stage compares each candidate with its competitors on both the S1 side and the S2/S3 side.
3. **Macro-F0.5-optimal decision rule.** It enforces the exclusivity that the data guarantees: every S2/S3 record belongs to at most one S1.

The pipeline is CPU-only, uses no external data and no neural models, and every threshold is tuned on out-of-fold and leave-one-country-out validation.

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from EDA on the full training set (2.2M S1, 10.3M S2+S3 records, 7.64M true links):

**Link structure**
- The singleton rate is only **5.6%**.
- An S1 has **3.46 true matches on average**, with a maximum of 11. Several noisy copies of the same business sit inside S2 and inside S3.
- **No S2/S3 record is linked to more than one S1** (0 of 7.64M), so a strict exclusivity constraint is valid.
- 73–75% of S2/S3 records are matched. The rest are distractors.
- There are **no cross-country links**, so blocking runs per country label. The label is treated as an open set, so France uses the same code path.

**Name noise**
- Legal forms moved to the start or middle of the name ("Pvt EFS Print Ventures Ltd").
- Bracketed forms, duplicated words, typos and leetspeak ("5ons").
- Fake accents ("Léarning"), social-media handles ("#pioneerfashion"), and "aka"/"DBA"/"formerly" aliases.
- Whole names rewritten in Devanagari, Tamil, Kannada, Bengali or Gujarati script.

**Address noise**
- St→"Saint", and state codes swapped with full names or native-script names (VA↔Virginia, दिल्ली).
- Leading zeros, NULL and N/A tokens, number ranges.
- Reordered components, dropped components, landmark-only addresses, and empty addresses (~3.4% of S2/S3).

**Hard negatives**
- Namesakes: the same business name at different addresses.
- Different businesses at the same address.
- Near-copies whose house number differs.

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage gradient-boosted classifier + F0.5-optimal set selection.
**Core Innovation:** (1) Hashed rare-key blocking that combines a forward and a reverse search, so degraded S2/S3 records still find their owner. (2) A collective second stage together with the exclusivity-aware macro-F0.5 decision rule.

---

## 3. Candidate Generation (Blocking)

**Normalisation** (hand-written rules, applied identically to every country label):
- Unicode NFKD accent folding (stdlib).
- Transliteration of 9 Indian scripts to Latin with `indic_transliteration` (MIT, offline), plus Hindi schwa-deletion rules.
- Legal-form removal anywhere in the name. The form is kept as a separate feature.
- DBA/aka split, leetspeak repair, duplicate-word removal.
- Symmetric abbreviation canonicalisation (street/saint→st, road→rd, …).
- A transliteration-tolerant phonetic key.
- Landmark segment extraction, postcode-like and house-number extraction, and NULL/N/A removal.

**Blocking keys used:** each record becomes a set of word-level keys:
- **Name keys:** tokens, token bigrams, phonetic tokens, and the no-space name with its 5-character prefix and suffix.
- **Address keys:** alphabetic tokens, bigrams, and number tokens.

Keys are feature-hashed (2^25 buckets) and IDF-weighted. Keys that occur in more than 3,000 records, or in only one, are dropped. Each record vector is L2-normalised.

Top-k cosine is computed with `sparse_dot_topn` over three key sets:
- **combined:** forward k=24, reverse k=4
- **name-only:** forward 8, reverse 2
- **address-only:** forward 8, reverse 2

The reverse search gives every S2/S3 record its best S1s. This recovers degraded records (empty address, garbled name) that lose to namesakes in the forward ranking. The union is capped at K=40 per S1.

- **Candidate pairs generated:** [fill from run_log: test n_cand_pairs]
- **How we ensured true matches were not lost:**
  - Measured on the full India train split: pair completeness **0.918** and oracle macro-F0.5 ceiling **0.966**, at a reduction ratio of 0.99999.
  - The reverse search raised completeness from 0.885 (forward-only).
  - [US numbers and final numbers to fill]

---

## 4. Matching Model

**Features used (~75, all language-agnostic, float32):**
- **Name features:**
  - RapidFuzz ratio, partial, token-sort and token-set, Jaro-Winkler and normalised Levenshtein on the legal-stripped core name.
  - Token-set on the full name.
  - Ratio and token-set on the phonetic key; ratio and partial on the no-space name.
  - The best DBA/trade-name combination.
  - IDF-weighted Jaccard and coverage, the rarest shared and rarest unshared token, and abbreviation-aware token coverage.
  - Acronym match, first-token equality, and legal-form agreement.
- **Address features:**
  - RapidFuzz scores on the full, alphabetic-only and landmark-free address.
  - Landmark presence and similarity, and landmark-in-other-address.
  - Postcode equality plus 3-digit and 2-digit prefix equality, house-number equality (each equal/conflict/missing), and number-set Jaccard.
  - IDF-weighted address token overlap.
- **Blocking and competition features:**
  - Per-field cosine and within-S1 rank.
  - Candidate counts; name and address gaps to the best candidate of the S1 and of the S2/S3 record.
  - The number of S1s claiming the same record, and same-address counts.

**Model type:**
- **Stage A:** LightGBM (MIT) on the pair features.
- **Stage B:** LightGBM on out-of-fold stage-A probabilities plus collective statistics: probability rank and gap within the S1 and within the S2/S3 record, the second-best probability, and counts above 0.5.

**Threshold selection method:** macro F0.5 is computed exactly and vectorised (per S1: 1.25·TP/(0.25·|gold|+|pred|), with singletons scoring 1 only when the prediction is empty). A grid is tuned on out-of-fold predictions over:
- the top-1 threshold t1,
- the extra-link threshold t2,
- the margin Δ to the top candidate,
- exclusivity (each S2/S3 record may go only to its highest-probability S1).

Theory for calibrated probabilities predicts t1≈0.5 and t2≈0.73. Country labels unseen in training get stricter thresholds (+0.05), chosen from leave-one-country-out validation.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [OOF and LOCO from run_log]
- **Common false positives (wrong merges):** [fill from oof_errors]
- **Common false negatives (missed matches):** [fill]

---

## 6. Conclusion
[fill]

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
- `src/prep.py`: parallel normalisation to per-country Parquet.
- `src/normalize.py`: all normalisation rules.
- `src/block_big.py`: hashed rare-key blocking.
- `src/features.py`: pair features.
- `src/model.py`: LightGBM OOF, LOCO and collective features.
- `src/decide.py`: F0.5 decision rules.
- `src/run_big.py`: end-to-end entry point.
- `src/io_utils.py`: TSV I/O and validator.

Reproduce:
```
python -m src.prep --data <dataset> --out prep
python -m src.run_big --prep prep --data <dataset> --out run
```
Both TSVs are written to `run/output/`.

### B. Compliance
- **No external data, APIs or geocoding.** All statistics (IDF, key frequencies) come from the provided files.
- **libpostal is deliberately not used** (its parser is trained on OpenStreetMap). **Unidecode (GPL) is not used.**
- **Final model:** LightGBM (MIT), with zero neural parameters.
- **Libraries:**
  - pandas, numpy, scipy: BSD
  - scikit-learn: BSD-3
  - rapidfuzz: MIT
  - sparse_dot_topn: Apache-2.0
  - indic_transliteration: MIT
  - pyarrow: Apache-2.0
