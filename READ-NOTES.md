# Amazon ML Challenge 2026: what the sources say (read 25 Sep 2026)

Sources in ./source: problem statement PDF (8pp, text only), guidelines PDF (2pp), video (6 min, 6 slides; transcript.txt + blk.png).

## Task
Business Entity Resolution. 3 TSV sources. S1 = clean deduplicated reference. For each S1 entity, output ALL matching S2/S3 ids (0, 1 or many).
Columns: entity_id (S1-/S2-/S3- prefix), business_name, business_address, country. Only name + address, no phone/email.
Train countries: US, India. **Test adds France (unseen).** Country is an open set: no hard-coding, and every test S1 row must be output.

## Metric
Macro F0.5, computed per S1 entity and averaged. Singletons: empty prediction = 1.0, any prediction = 0.0.
Precision counts 2x. "When in doubt, don't merge."

## Outputs (tab-separated, no quoting, comma-separated id lists)
- output/matching_results.tsv: `source1_entity_id	matched_entity_ids`. Only file scored.
- output/candidate_pairs.tsv: `source1_entity_id	candidate_entity_ids`. This is the FINAL set the model scores, not an early blocking pass. Audited for recall ceiling and reduction ratio. Matches must be a subset.
- Rules: exactly one row per S1, no duplicate rows, no duplicate ids within a list, only S2/S3 ids that exist in test.
- Validate: `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` (run from student_resource/)

## Final zip: <team>_submission.zip
output/{matching_results,candidate_pairs}.tsv; code/business_entity_resolution/{src/, README.md, requirements.txt pinned}; Documentation_template.md filled in.
Guidelines also ask for a 1-2 page doc (approach, models, experiments, conclusion) plus commented source code.

## Hard constraints
- Final model: MIT/Apache-2.0 license, <= 8B params.
- NO external lookup: no ER APIs, govt registries, geocoding, or internet augmentation. "Use only the provided data." Code is audited; a violation means disqualification.
- 5 leaderboard submissions per day, 3 days. Keep a version history of every submission.
- Window: 25 Sep 00:00 IST to 27 Sep 23:59 IST. Public LB = subset; final ranking = private LB (the guidelines say both are considered).
- One login per participant, desktop only.

## Noise patterns
Names: Corp/Corporation, Pvt/Private, Ltd/Limited, legal suffixes, DBA/trade names, & vs "and", word reordering, typos, transliteration.
Addresses: Rd/Road, St/Street, transliteration, missing PIN/state, landmarks ("Near SBI ATM", "Nr City Hall"), municipal numbering, reordered components.

## Video example (blk.png)
S1 "Acme Robotics Inc, 500 Market St, San Jose". Its block pulled in S2-118820, S2-540221, S3-905477, S3-063118.
Model kept S2-118820 (Acme Robotics Inc, 500 Market Street) and S3-905477 (Acme Robotics, **Nr City Hall**, San Jose: a landmark-only address still matches).
Model rejected S2-540221 (Acme Robotix, 12 Elm Rd: similar name, different address) and S3-063118 (Acme Bakery, same address, different business).
=> Blocking must use name OR address keys. The matcher must need both to agree, and must tolerate landmark addresses.
Tips: blocking sets the recall ceiling; country-specific address patterns; Jaccard/Levenshtein/TF-IDF features.

## Full-frame pass (second read)
- The video has 10,676 frames at 30fps. mpdecimate keeps 261 visually distinct ones, and all 261 were reviewed. Every change happens during a transition (0, 29-30, 64-65, 88-90, 108-110, 163-165, 179-180, 228, 279, 345s). The screen is static in between.
- The first pass had missed nothing. There is no hidden text, extra slide or data sample.
- Small details seen in the animations:
  - Form caption: "Phone and email exist in reality. This challenge uses name and address only."
  - Matching-model animation: the rejected S2-540221 and S3-063118 drop out with a red ✗. The output row is `S1-732914 → S2-118820, S3-905477`, labelled "one row per Source 1 entity".
  - Graph slide: the look-alike is "Acme Bakery LLC" (Source 2). The Source 3 match is shown as "Acme Robotics, City Hall".
- Transcript note: "Its ID may tee to" = "its ID maps to" (a Whisper mishearing).
