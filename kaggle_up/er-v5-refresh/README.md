# Business Entity Resolution: Amazon ML Challenge 2026

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` end to end from the provided data.
It is CPU-only, uses no external data or network calls, and has no neural models: TF-IDF + LightGBM (MIT).

## Setup (Python 3.11)
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run at full scale (the leaderboard submissions)
The real data (2.2M train S1, 1.7M test S1, ~20M S2/S3 records) runs through the memory-bounded, restartable
runner on an 8 GB / 8-core laptop. Preprocess once, then run every phase:
```bash
python -m src.prep --data <student_resource>/dataset --out ../../prep                      # ~25 min, Parquet per country
python -X faulthandler -u -m src.run_big --prep ../../prep --data <student_resource>/dataset \
    --out ../../runs/real_v2 --phase all                                                    # train, test per country, final
```
`--phase all` trains (sample of 60k S1 per train country, 3-fold GroupKFold OOF plus leave-one-country-out), then
blocks and scores each test country in its own process, then streams `output/matching_results.tsv` and
`output/candidate_pairs.tsv` and runs the official validator. Every phase checkpoints to disk, so re-running the
same command resumes after a crash. Blocking for the full-scale runner is `src/block_big.py` (hashed rare word keys:
combined name+address, name-only and address-only, forward and reverse top-k, capped at K=40 candidates per S1).

## Run on a small dataset
`--data` points at the folder that holds `train/` and `test/` (i.e. `student_resource/dataset`).
```bash
python -m src.run --data ../../dataset --out ../../run_final        # full pipeline, about 7 min for 30k S1 on a laptop
python -m src.run --data ../../dataset --out ../../run_eda --eda-only
```
Outputs: `<out>/output/matching_results.tsv`, `<out>/output/candidate_pairs.tsv`, and `<out>/run_log.json`
(EDA, blocking recall/reduction per country, CV and leave-one-country-out scores, tuned thresholds, validation).

Then check the files with the official validator, run from `student_resource/`:
```bash
python3 utils/validate_submission.py --matching <out>/output/matching_results.tsv \
  --candidate <out>/output/candidate_pairs.tsv --test-dir dataset/test
```

## Pipeline
| Stage | File | What it does |
|---|---|---|
| Normalise | `src/normalize.py` | Accent folding (stdlib), legal-form stripping (multi-country), DBA split, symmetric abbreviation canon, phonetic key, landmark/postcode/house-number extraction |
| Block | `src/blocking.py` | Top-k char-3-gram TF-IDF cosine within each country label: name, address, name+address, name+locality, plus reverse searches |
| Stage 1 | `src/run.py`, `src/model.py` | LightGBM on blocking scores keeps the top-K per S1. **That set is `candidate_pairs.tsv`** |
| Stage 2 | `src/features.py` | ~75 language-agnostic pair features, LightGBM, GroupKFold by S1 |
| Stage 3 | `src/model.py` | Collective re-scoring from the stage-2 probabilities of competing candidates |
| Decide | `src/decide.py` | Threshold rule tuned for macro F0.5 on out-of-fold predictions, with S2/S3 exclusivity. Stricter thresholds for country labels unseen in training |
| Metric | `src/metrics.py` | The official macro F0.5 (singletons included), blocking PC/RR/oracle ceiling |

`tools/make_synthetic.py` builds a synthetic dataset in the official format, used only to smoke-test the code.
`tools/block_sweep.py` and `tools/block_misses.py` are blocking diagnostics.
