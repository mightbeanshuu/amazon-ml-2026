# Research pass 7 — hunt for the "viral 0.985 solution" (26 Sep 2026, ~19:30 IST)

Task: the public LB exploded today — dozens of teams tied at 0.985–0.986 (submissions 10:00–15:00 IST), top 0.9906. Find the public solution behind it.

## VERDICT: NOT FOUND as a public artifact — and it almost certainly is not one

After sweeping GitHub (repo search, code search for `0.985`/`0.986`/`0.9853`/`0.9856`/`0.9906`/"LB 0.98"/"public leaderboard 0.98", gists, issues, fork counts), Kaggle (all 12 notebooks on the mirror dataset), Reddit (r/hackathon, r/Btechtards, r/learnmachinelearning via search + pullpush), X/LinkedIn (exa), YouTube (qdr:d — zero uploads today), HuggingFace (dataset mirrors + models), and Telegram/t.me indexes:

- **No public repo, notebook, gist, or video claims a public LB above 0.961** (Tony-AJ, unchanged since 10:50 IST). The highest local claim public anywhere is **0.98703 (AyanAhmedKhan, cross-validated local, pushed 25 Sep)** — below the 0.985–0.986 *public* wall.
- **No fork/copy wave**: the most-forked "amazon ml challenge 2026" repo has 1 fork; the top Kaggle notebook (mobeenfatimah, 45 votes, rule-based EDA) reports no score; sudharsans22 (52 votes) is a sampled-comparison notebook, no LB claim.
- **The tie pattern has a documented precedent in this exact competition**: after AMC 2025, Chirag Saha publicly wrote "multiple teams from the same college had identical scores, strongly suggesting duplicated submissions… Kindly request Amazon/Unstop to look into this" ([LinkedIn](https://www.linkedin.com/posts/chiragsaha3_mlcompetition-amazon-machinelearningchallenge-activity-7384164179844313088-Aox2)). Identical scores across many colleges = **the same `matching_results.tsv` circulating in WhatsApp/college groups**, not a public method. Unstop's tie-break is "earlier submission wins" ([rules recap](https://www.linkedin.com/feed/update/urn:li:activity:7503777274245292032)) — which explains today's 10:00–15:00 IST submission rush on the same file.
- Corroboration that 0.98+ existed at the top *before* today: priyanshiiitr's team HANDOFF (public) records "Top of leaderboard **0.9869**"; a Reddit comment 7h ago mentions a team "pushed till **0.981**" after 15 submissions ([thread](https://www.reddit.com/r/hackathon/comments/1wqhjjr/amazon_ml_challenge_submission_limit/)). So 0.985–0.986 is a copy of a genuinely strong team's output (or a very confident team's file leaked by a member), two versions of which would neatly give the 0.985 vs 0.986 split.

**Implication for us:** there is nothing to copy, and copying a TSV would be plagiarism the organisers say they screen for (2025 pattern was called out publicly; top-50 packages are reviewed in detail). The route to 0.985+ has to be mechanism, and the best public blueprint follows.

---

## The 3 strongest public artifacts with score claims (last 24–36 h), ranked

### 1. AyanAhmedKhan — local stacked **0.98703 ± 0.00016** (the real "0.985-class" public blueprint)
Repo: <https://github.com/AyanAhmedKhan/amazon-ml-challenge> (pushed 25 Sep 10:15 UTC, 2 commits; goldmine 05 quoted its self-training numbers but NOT the full stack).
Key files: [FINAL_RECOMMENDATION.md](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/FINAL_RECOMMENDATION.md), [RESEARCH_LOG.md](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/RESEARCH_LOG.md).

The complete measured recipe (their words, verbatim where quoted):

**Blocking — this is the piece that clears the 0.97-recall wall:**
> "rev (target→S1 hybrid) top-8 ∪ hybrid name+address top-10 ∪ name char3 top-5 ∪ name-concat char3 top-5 ∪ address words top-5"
- "**Reverse target→S1 retrieval R@1 = 0.972**, R@8 = 0.9875. On the full pool the name view collapses to R@40 = 0.677 (same-name flooding)."
- "Train, full pool: union pair recall **0.9905**, oracle macro F0.5 **0.9972**, 112 candidates/S1" → pruned budget "≈43–51 per S1 at recall 0.989, oracle 0.9965".

**Model ladder (fold-0/CV local F0.5):**
- LightGBM stage-1: "best rule 0.767 → LightGBM **0.9846**" (thr 0.7 + exclusivity; 777 rounds, 5 min).
- cross-fit 3-fold ensemble: "+0.0004 (**0.98492**)".
- "**stacking + collective features: +0.0021 → 0.98703 ± 0.00016** (13 SE)" — stage 2 LightGBM adds "p1 + 16 collective features + p1 competition" where collective = sibling-support features from out-of-fold stage-1 probabilities.
- "density-matched training: under test-like density, 0.98343 → **0.98479**" (train the matcher at test distractor density, not just retune).
- LightGBM params: "lr 0.08, 255 leaves, min_data 400, ff/bf 0.7, max_bin 127. 3-fold cross-fit at both stages."
- Decision: "Threshold + many-to-one exclusivity, τ 0.7 at train density, **0.8 under P-dense**… expected-F and π-hurdle did not beat the tuned threshold. Singleton accuracy ≈ 0.98."
- Normalization: "learned, country-agnostic… native→Latin dictionary learned from training pairs; covers 96.4% of test native tokens."
- Submission status: "sub_v1… Local estimate 0.983–0.985. sub_v2 (dense stacked) pending compute" (Modal spend limit stopped it — they may never have uploaded the stacked model).
- License/compliance: uses only train-derived dictionaries, LightGBM; consistent with rules.

### 2. Tony-AJ — public LB **0.961** measured (26 Sep 10:50 IST), unchanged since
Repo: <https://github.com/Tony-AJ/business_entity_resolution> ([LEADERBOARD.md](https://github.com/Tony-AJ/business_entity_resolution/blob/main/LEADERBOARD.md), [TRACKER.md](https://github.com/Tony-AJ/business_entity_resolution/blob/main/TRACKER.md)). Already in 05-goldmine; NEW today: Submission 04 (v107, two-stage + expected-F on stage-2, mock est. public ~0.9659) is queued with "Public F0.5: (fill in after upload)" — still blank at fetch time. Their measured mock→public offset shrank 0.0127 → 0.0094 after the τ-triple retune; local 0.9858 / cand recall 0.9906. Nothing in their repo reaches 0.985 public.

### 3. priyanshiiitr team HANDOFF — public LB 0.883, but the best public *diagnosis of what 0.985 requires*
Repo: <https://github.com/priyanshiiitr/amazon-ml-challenge/blob/main/HANDOFF.md>. Verbatim:
> "Oracle ceiling on those candidates **0.96585**… Reaching 0.985 additionally needs **blocking recall above 0.97**, and ours is 0.9501 even at K=1000."
> "**The single biggest feature:** `rev_rank` from reverse retrieval — 'is this entity this record's single best owner among all 2.2M?' It is the top feature by **5×** and lifted the singleton score 0.871 → 0.904."
> "Error analysis produced more than every architectural idea combined" — 3 bugs worth **+0.011**: (1) empty addresses scored as disagreement — emit **NaN**, not 0, for `token_set_ratio(x,"")` (LightGBM handles missing natively); (2) leading zeros broke numeric matching (`b 011 c 2` vs `b 11 c 2`); (3) no token-IDF name-rarity features. Round 2: mid-token unit numbers (`404 d1` vs `404 d6` scored 0.999 — extraction only caught digit-*initial* tokens).
> "Co-matched records resemble each other more than their own S1 record: max(name,address) sim averages **97.2** over co-matched pairs vs 46.7 random" — the basis for sibling-support stacking.
> Top of leaderboard: **0.9869** (as recorded there). LB−local offset stable at −0.02 for them.

### Honourable mentions (score-claimed, last 36 h)
- **Disastrio/BuisnessEntityResolutionML** — local F0.5 **0.9818** (single model + hard negatives) / **0.9820** (dual models + per-source S2/S3 thresholds); 7-tier blocking incl. rare-token IDF index and Soundex; 28 features; t*≈0.72. <https://github.com/Disastrio/BuisnessEntityResolutionML>
- **Siva402-ai** — validation **0.9783** end-to-end vs all 10.3M train targets (Indic dictionary + difference/margin features; already in goldmine as 0.9707→0.9783). <https://github.com/Siva402-ai/amazon-ml-challenge-2026>
- **Mitanshp5/ML-Devs** [F05_098_IMPLEMENTATION_PLAN.md](https://github.com/Mitanshp5/ML-Devs/blob/main/F05_098_IMPLEMENTATION_PLAN.md) — an audited plan toward >0.98; useful independent confirmations: blocking oracle at 94.1% recall is only 0.9793 ("even perfect classification cannot exceed 0.98 on that sample"); train/test exact ID overlap is **0** in all three sources (kills the naive train→test ID-leak theory for the 0.985 wall).
- **Skullybutcher/ml_challenge_2026** [RUN_STATUS.md](https://github.com/Skullybutcher/ml_challenge_2026/blob/main/RUN_STATUS.md) — records "Board top: 0.980473" at an earlier snapshot; OOF 0.9844–0.9847 on a 5k sample (sample-optimistic).

---

## What makes 0.985+ (synthesis for v7, beyond goldmine 05)

The public evidence triangulates one story. 0.985 public ≈ 0.99+ local, which needs BOTH:
1. **Blocking recall > 0.97 at test density** — the only public route shown is **reverse (target→S1) retrieval** as a first-class blocking view (Ayan: R@8 0.9875 alone, union 0.9905 at ~45/S1; priyanshiiitr independently: forward K=1000 saturates at 0.950, so more K is NOT the route).
2. **Near-oracle matching via stacking on collective evidence** — OOF stage-1 probabilities → 16 sibling-support features → stage-2 LightGBM (+0.0021, the largest single modelled gain any public team measured), on top of `rev_rank` / competition features, trained at test-matched density (+0.0014), decided by τ≈0.8 + exclusivity.
3. Plus the cheap error-analysis fixes if we lack them: empty-field NaN (not 0), digit-normalised numeric tokens including mid-token units, token-IDF rarity features (+0.011 for priyanshiiitr's team).

And the 0.985–0.986 wall itself is, on all available evidence, **one strong team's submission file passed around**, not a method — do not chase it as a method, and expect the organisers' package review + the 2025 plagiarism precedent to purge it from the final board.

## Search log (for reproducibility)
- GitHub repo search `amazon ml challenge pushed:>2026-09-25` → 263 repos (flood, all low-signal); `entity resolution created:>2026-09-24` → 331; sort=stars/forks → max 3 stars, 1 fork (no copy wave).
- GitHub code search: `"amazon ml challenge" 0.985` (23 hits, all triaged above), `"LB 0.98"`, `"public leaderboard" "0.98" entity`, `0.9853`, `0.9856`, `"amazon ml" 0.9906`, `matching_results 0.98` — no public-LB ≥0.97 claim anywhere.
- Kaggle: dataset mirror [satwiksps/amazon-ml-challenge-2026](https://www.kaggle.com/datasets/satwiksps/amazon-ml-challenge-2026) (+ [code tab](https://www.kaggle.com/datasets/satwiksps/amazon-ml-challenge-2026/code), 12 notebooks): [mobeenfatimah](https://www.kaggle.com/code/mobeenfatimah/amazon-ml-challenge-2026-entity-resolution) (45▲, rule-based, no score), [sudharsans22](https://www.kaggle.com/code/sudharsans22/amazonchallenge) (52▲, sampled comparison), rest are EDA/no-score. HF mirror: [akshatbakshi/amazon-ml-challenge-2026](https://huggingface.co/datasets/akshatbakshi/amazon-ml-challenge-2026) (data only, no predictions).
- Reddit last 24 h: [submission-limit thread](https://www.reddit.com/r/hackathon/comments/1wqhjjr/amazon_ml_challenge_submission_limit/) ("pushed till 0.981"), [leaderboard-rank thread](https://www.reddit.com/r/hackathon/comments/1wqfqg7/leaderboard_rank_amazon_challenge/), upload-button threads — no solution links.
- YouTube uploads today: none (qdr:d empty). Telegram/t.me, X: nothing indexed. Unstop discussion page: JS-walled, no public discussion content served ([ML round page](https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604/coding-challenge/290344)).
