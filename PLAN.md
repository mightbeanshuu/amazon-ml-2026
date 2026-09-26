# Amazon ML Challenge 2026: refined execution plan

Sources: READ-NOTES.md (PDFs + video), research/01-challenge-intel.md, research/02-technical-playbook.md.
Code: code/business_entity_resolution/ (built and tested on synthetic data on 25 Sep).

## Hard constraints (the checklist before every upload)
- [ ] `matching_results.tsv` + `candidate_pairs.tsv` pass `utils/validate_submission.py` (the official one) AND our `io_utils.validate`.
- [ ] Final model is MIT/Apache-2.0 and ≤8B params **summed over every neural model**. The current pipeline has **zero neural params** (LightGBM + TF-IDF only).
- [ ] No external data or API at runtime. No libpostal parser, no Unidecode (GPL), no geocoders, and no LLM API in the pipeline. The 2024 rule discarded submissions that used LLM APIs.
- [ ] The final zip is **≤ 50 MB** and ships no weights. Its `matching_results.tsv` must be **byte-identical** to the leaderboard upload of the best submission.
- [ ] Every upload is archived in `submissions/<n>_<time>/`: both TSVs, `run_log.json`, the git hash, the cfg and the public LB score.

## Machine reality
- MacBook: **8 GB RAM**, 8 cores, no CUDA. No GPUs are provided by the challenge (only AWS Free Tier CPU hours).
- So a 7B LLM judge is impractical: at 4-bit it needs about 4.5 GB and runs at a few pairs per second. It is dropped from the plan unless a free GPU appears.
- A small multilingual cross-encoder (118M) or e5-small embedder is feasible on CPU/MPS for about 100k pairs. Measure first.

## Pipeline (implemented)
1. **Normalise** (`src/normalize.py`)
   - accent fold with stdlib; collapse dotted acronyms; strip French elisions; `&`→`and`
   - multi-country legal-form strip, keeping the legal family as a feature; DBA/trade split
   - symmetric token canonicalisation (st/street/saint → `st`)
   - phonetic transliteration key
   - address: comma-segment landmark split, postcode-like (5/6 digits), house number
2. **Block** (`src/blocking.py`): char-3-gram TF-IDF top-k within each country label.
   - Name k=15, address k=10, name+address k=20, reverse k=5; unioned.
   - Rank and gap features for every search.
3. **Stage 1 prune:** LightGBM on blocking features only, then **keep the top-K=12 per S1. That set is `candidate_pairs.tsv`**, the exact set the later stages score.
4. **Stage 2 matcher:** LightGBM on about 75 language-agnostic features (rapidfuzz family, IDF-weighted token overlap, abbreviation-aware match, phonetic, postcode/house tri-state, landmark, legal-form agreement, competition counts).
5. **Stage 3 collective:** LightGBM on stage-2 features plus probability rank and gap within the S1 and within the S2/S3 record (out-of-fold stacking).
6. **Decide:**
   - Top-1 if p ≥ t1; extras if p ≥ t2 and within Δ of the top; exclusivity (an S2/S3 record only goes to its best S1).
   - Tuned on OOF: synthetic run → t1=0.50, t2=0.75, exclusive (matches the theory).
   - GFM expected-F0.5 set selection is computed as an alternative.
   - Unseen country labels (France) get +0.05 on both thresholds.

## Validation protocol (don't overfit the public leaderboard)
- GroupKFold(5) by S1 gives the OOF score. **Leave-one-country-out** (train US → test India, and back) is the proxy for France.
- Choose between variants by the **mean of OOF and LOCO**. Never by the public LB alone: 2025 had a 1/3 public subset and ranks moved.
- For the blocking report, look at the oracle macro-F0.5 ceiling, PC, RR and mean candidates, per country. Put these tables in the doc.

## When the real data lands (run order)
1. `python -m src.run --data <student_resource>/dataset --out runs/real_eda --eda-only`, then read:
   - singleton rate; max S2/S3 matches per S1 (≤1 → one-to-one assignment per source)
   - whether any S2/S3 record is matched to several S1 (tests the exclusivity assumption)
   - cross-country GT pairs (decides `partition_by_country`); non-ASCII script; label noise
2. Eyeball 50 matched pairs and 50 hard negatives, per country. Extend the normaliser lists with what the data shows. Mining token synonyms from GT pairs is allowed.
3. Run the full pipeline and check that the blocking oracle is ≥ 0.97. If it isn't, raise k or add number-key and phonetic blocks.
4. **Submission #1 by the end of day 1**, since submitting early wins ties. Keep #2 and #3 for threshold variants.
5. **Optional probe:** an all-empty submission scores exactly the public singleton rate. Spend one of the 15 submissions only if the singleton prior looks off.

## Day 2–3 upgrades (each kept only if OOF + LOCO improve)
- Dense blocking + feature: `intfloat/multilingual-e5-small` (MIT, 118M) with faiss-cpu (MIT).
- Cross-encoder feature: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache, 118M), fine-tuned on hard negatives, CPU/MPS.
- Seed ensemble (3 seeds) + CatBoost (Apache).
- Transitivity across S2↔S3.
- Synthetic French-style noise augmentation of train pairs, checked on LOCO.
- Pseudo-labelling is a grey zone ("use only the provided data"). Skip it unless it is clearly documented and clearly helps.

## Deliverables
- `code/business_entity_resolution/`: `src/`, `README.md` (exact commands), `requirements.txt` (pinned).
- `Documentation_template.md`, filled in: method, blocking tables, feature list, licence table, LOCO results, the "no external data" interpretation (libpostal/Unidecode excluded and why).
- 1–2 page approach doc (guidelines PDF), including experiments and conclusion.
- Zip ≤ 50 MB. Build it well before 27 Sep 23:59 IST: 2025 teams saw uploads fail in the last minutes.
