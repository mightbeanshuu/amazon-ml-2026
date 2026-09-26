# Amazon ML Challenge 2026: challenge intel (research pass, 25 Sep 2026, ~01:20 IST)

This file covers only what is NOT already in `../READ-NOTES.md`. Every claim has a source URL.
Tags: **[OFFICIAL]** = Unstop listing/API, Amazon, or AWS staff. **[3P]** = participant or third-party repost. **[UNVERIFIED]** = could not confirm. **[INFERENCE]** = my reading; nobody states it.

The raw Unstop API dump is saved next to this file as `unstop-competition-1743604.json`. It was fetched 25 Sep 2026 01:14 IST from `https://unstop.com/api/public/competition/1743604`, the JSON the listing page renders from. Round IDs, dates and rule text below come from it unless another source is named.

Timing caveat: this pass ran about 80 minutes after the window opened. Almost nothing from inside the event has been posted publicly yet.

---

## 0. Decision-relevant findings

1. **The final ZIP has a 50 MB limit.** A hidden Unstop round called "Code Submission" takes a single `.zip` with `max_file_size: 50`. That ZIP has to hold `candidate_pairs.tsv`, so keep the candidate set lean and do not ship model weights in it. [OFFICIAL, API `rounds[round_order=4].submission_types`]
2. **Tie-break on the live leaderboard:** "ranked on the maximum score of the submission and submission time… if two… have the same score, then the… team that finished earlier will be ranked higher." Your best score counts, and submitting early wins ties. [OFFICIAL, https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604/coding-challenge/290344]
3. **Private leaderboard = the complete test set**, revealed after the hackathon, "based on your final solution submission." The split % is not published. In 2025 the public leaderboard was 25K of 75K (one third). [OFFICIAL 2026 round text; 2025 split: https://github.com/AdiSinghCodes/Amazon-ML-Challenge-2025/blob/main/README.md]
4. **No GPUs are provided.** The "$200 credits" are the standard AWS Free Tier for new accounts ($100 at sign-up plus up to $100 for activities). The SageMaker free tier is CPU only (ml.t3.medium notebooks, ml.m5.xlarge training). The **top 500 teams at the 48-hour mark (about 27 Sep 00:00 IST)** get an extra $100 credit code, so have a decent leaderboard score by then. [OFFICIAL: listing + guide PDF https://d8it4huxumps7.cloudfront.net/uploads/attachements/files/4cd78da9-38ed-4832-a95e-e074a69251c7.pdf]
5. **Top-50 results come on 2 Oct.** The top 50 get PPIs (in practice an online assessment plus an interview) for Applied Scientist Intern. The top 10 present at the Grand Finale on 7 Oct, 10:00 to 15:00 IST. The finale reorders the top 10, so the approach doc matters. [OFFICIAL listing; OA detail is 3P]
6. **2024 precedent:** submissions that used LLM APIs (OpenAI, Anthropic, Google, etc.) were discarded. No 2026 statement either way was found. Keep every model local and open-weight (MIT/Apache, ≤8B). [OFFICIAL 2024: `https://unstop.com/api/public/competition/1100713`]

---

## 1. Official Unstop listing

**URLs**
- Listing: https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604 (AMP copy: https://unstop.com/hackathons/amazon-ml-challenge-2026-amazon-1743604/amp)
- ML round (ID 290344): https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604/coding-challenge/290344
- Unstop classifies it as `type: hackathons`, `subtype: hiring_challenge`. [OFFICIAL, API]

### 1.1 Eligibility and team
- Open to full-time PhD / M.E. / M.Tech / M.S. / MS by Research / B.E. / B.Tech students at engineering campuses in India who graduate in **2027 or 2028**. [OFFICIAL, listing]
- **Team size is 2 to 4** (`min_team_size: 2, max_team_size: 4`). Each team needs a leader. Cross-college teams are allowed. A student can be on only one team. [OFFICIAL, listing + API `regnRequirements`]
  - Several reposts say "3–4 members", including the Builder Center post https://builder.aws.com/post/3JBxLF0Pm7NZGZZaf24EnivwNcf_p/amazon-ml-challenge-2026-is-live-on-unstop and https://internshala.com/competitions/amazon-ml-challenge-2026-win-%E2%82%B9225000/. That is the 2024/2025 rule and is out of date. [3P, stale]
- Stage 0: every member needs a valid **AWS Builder Center Profile ID alias**. An invalid alias leaves the registration incomplete. [OFFICIAL, listing]
- As of 25 Sep 01:14 IST there were **89,407 registered** individuals and `players_count` 26,624. In 2025 the same field was 19,556 against "20,000+ teams" quoted by participants, so it is probably the team count [INFERENCE]. [OFFICIAL, API]

### 1.2 Prizes [OFFICIAL, API `prizes` + listing]
| Who | Reward |
|---|---|
| Winner | INR 1,00,000 + certificates + goodies |
| 1st runner-up | INR 75,000 + certificates + goodies |
| 2nd runner-up | INR 50,000 + certificates + goodies |
| Top 50 teams | PPIs for **Applied Scientist Intern** at Amazon (`pre_placement_internship: 1`) |
| Top 10 teams | Certificates + exclusive Amazon swag |
| Top 10 women-only teams | Certificates + exclusive Amazon swag |
| All registrants | "$200 worth of credits" (in practice the standard AWS Free Tier; see §2) |
| Top 500 teams at the 48-hour mark | Extra $100 AWS credit code |

- The prize table appears on the listing and in the AWS blog, which also gives "79,000+ students registered" as of 21 Sep: https://builder.aws.com/content/3HiM6zDmFrF98fRzOUETnGFDoqz/amazon-ml-challenge-2026-your-complete-prep-guide-with-live-demo
- What a PPI looks like in practice: "The top 50 teams get an Online Assessment (OA) and an interview opportunity for the Applied Scientist internship position" (Poojan, 2023 winner and 2024 runner-up, now an Amazon Applied Scientist): https://www.reddit.com/r/Btechtards/comments/1ntmvrr/amazon_ml_challange_is_back_for_2025_i_won_it_2/ [3P]

### 1.3 Timeline (exact times from the API; all IST) [OFFICIAL]
| Round (order) | Start | End | Notes |
|---|---|---|---|
| 1 Registration | 07 Sep 20:00 | 20 Sep 23:59 (listing text) | Registration requirements say `end_regn_dt` 22 Sep 23:59:45 (extended). The header shows "Registration Deadline 22 Sep 26, 11:59 PM IST". |
| 2 Best Practices Virtual Session | 21 Sep 17:00 | 21 Sep 18:00 | Twitch VOD: https://www.twitch.tv/videos/2880101957 |
| 3 **ML Challenge, 72-Hour Hackathon** (ID 290344) | **25 Sep 00:00:50** | **27 Sep 23:59:49** | Duration "2 days 23:59:00", status LIVE |
| 4 "Code Submission" (**hidden**) | 08 Oct 12:46 | 09 Oct 12:46 | ZIP upload; see §1.5. These dates fall after the finale and are probably placeholders [UNVERIFIED]. |
| 5 Machine Learning Round Results | 02 Oct 10:00 | 02 Oct 23:59 | "Check the results for the Top 50 teams here." |
| 6 Grand Finale | 07 Oct 10:00 | 07 Oct 15:00 | Top 10, live and virtual, presenting to Amazon scientists |
| 7 "Email for ID Card" (hidden) | 12 Oct | 25 Oct | Offline round; the same round existed in 2024 and 2025 |

- Internshala lists the ML window as "25 Sep 9:00 AM – 27 Sep 9:00 PM IST". That conflicts with the official 00:00 to 23:59, so ignore it. [3P, wrong]

### 1.4 ML round mechanics [OFFICIAL]
- Round text: "Teams can track their performance through the leaderboard… There will be a private leaderboard that will be revealed after the hackathon, based on your final solution submission and validation on the complete test dataset." (listing)
- Assessment-guidelines page (https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604/coding-challenge/290344):
  - "Every participant/team will be ranked on the **maximum score** of the submission and submission time… if two participants/teams have the same score, then the participant/team that **finished earlier** will be ranked higher."
  - "You won't be able to modify your answers during the assessment. Once you have given your answer, it is stored and cannot be changed." (generic Unstop text; a submission cannot be withdrawn [INFERENCE])
  - "Any participant resorting to unfair practices will be directly disqualified." Eligibility and final judgement rest with Unstop and the organiser. No negative marking.
  - "attempt the assessment in one sitting as once you start… the timer won't stop"; the window ends 27 Sep 23:59 IST.
- API fields on round 290344: `contest_type: ml`, `leaderboard_style: lbml`, `live_leaderboard: 1`, `show_round_score: 1`, `eliminator_round: 1`, `result_count: 100`, `other_data: "5"`, `no_of_submissions_file_allowed: 1`, `proctoring: {"maxFullscreenViolations": 5}`.
  - `other_data: "5"`: in 2025 this field was also "5" and the 2025 rule said "You can make only 5 submissions per day", which matches the 5/day already known. (2025 API: `https://unstop.com/api/public/competition/1560375`)
  - `maxFullscreenViolations: 5` suggests the Unstop assessment UI may count exits from fullscreen. Whether it is enforced on an ML round, and what happens at 5, is not stated. **[UNVERIFIED]** Do not repeatedly leave fullscreen in the Unstop assessment tab.
  - `result_count: 100` might mean the results display 100 entries. Its meaning is **[UNVERIFIED]**.

### 1.5 The final ZIP (hidden "Code Submission" round, text created 24 Sep 16:47 IST) [OFFICIAL, API]
- "Every team must submit a single ZIP archive containing your code and outputs for the **best submission**. We use it to reproduce your results, audit your blocking approach, and verify compliance with the fair-play and model-license rules. The top teams' packages will be reviewed in detail before the final rankings are confirmed."
- The structure matches READ-NOTES. New details:
  - `matching_results.tsv` must be "**identical to the file uploaded to the leaderboard**".
  - The pipeline "should be capable of regenerating both output files from the provided training/test data using only the contents of this folder".
  - `Documentation_template.md` is enough on its own; a PDF export is also accepted; "no need to rename the file".
- Upload field: type `file`, accepts `zip`, **`max_file_size: 50` (MB)**, mandatory.
- Round description: "Top Teams from 72-Hour Hackathon will be required to submit documentation regarding the approach behind their best submission. The approach & code will be evaluated by the Amazon team to finalize the top teams of ML Challenge 2026."
- [INFERENCE] The full ZIP may only be collected from top teams after the hackathon. Still, have it built, under 50 MB, before 27 Sep 23:59 in case the leaderboard upload also asks for it. In 2025 teams uploaded a ZIP with every CSV (see §4.3).

### 1.6 FAQ / Discussions
- The listing has a "Frequently Asked Questions/Discussions" section ("Updated On: 24 Sep 26, 11:38 PM IST"). Its entries do **not** render publicly: headless scrapes returned only the heading, and the AMP page has no FAQ. They are probably login-gated. The Chrome extension was not connected, so a logged-in view could not be read. **Action for the team: read it while logged in.**

### 1.7 Dataset size and split
- Neither the listing nor the API gives dataset sizes or the public/private split %. **[Not found]**

---

## 2. The "blog for ML Challenge on best practices and live demo"

- The Unstop banner ("Missed the Best Practices Virtual Session?", active 24 Sep 12:58 to 28 Sep 12:58 IST) links to **"Amazon ML Challenge 2026: Your Complete Prep Guide with Live Demo"** by Jatin Mehrotra, AWS Developer Advocate, published 21 Sep 2026: https://builder.aws.com/content/3HiM6zDmFrF98fRzOUETnGFDoqz/amazon-ml-challenge-2026-your-complete-prep-guide-with-live-demo [OFFICIAL]. Recording: https://www.twitch.tv/videos/2880101957
- **It contains nothing specific to entity resolution.** The demo is XGBoost churn prediction on a synthetic telecom dataset. Concrete advice:
  - For the challenge, use "Notebook Instance + local training + local prediction. **You submit a CSV, not a running API.**" Endpoints cost about $0.12/hour even when idle; skip them.
  - Use an `ml.t3.medium` Notebook Instance, not Studio, because new accounts may hit Studio domain or quota delays. Use **us-east-1**, set billing alerts, and stop notebooks when idle (stop rather than delete, or files are lost).
  - Check the label distribution first: "in the ML Challenge, your data might be heavily skewed to one side. That changes how you build and evaluate your model."
  - "the ML Challenge data won't be this clean! You'll need to handle missing values and noisy data."
  - "Feature engineering often matters more than your choice of algorithm. Spend 70% of your time here." Drop unique-ID columns as features.
  - Use train/validation/test splits and "never touch test data until you're done experimenting."
  - The author wrote "For the ML Challenge dataset, this is all you need" about local XGBoost. This was written before the problem was released and is generic; it is not evidence of dataset size [INFERENCE].
- Free resources listed in the blog [OFFICIAL]:
  - SageMaker free tier: 250 h ml.t3.medium notebooks, 50 h ml.m5.xlarge training, 125 h ml.m5.xlarge inference, for 2 months. All CPU.
  - Student Rewards: 7 badges = $10 credits, 14 = $20, 21 = a $100 certification voucher.
  - "$200 in AWS credits" = $100 on sign-up plus $100 for five starter activities.
- Official setup-guide PDF attached to the listing: https://d8it4huxumps7.cloudfront.net/uploads/attachements/files/4cd78da9-38ed-4832-a95e-e074a69251c7.pdf [OFFICIAL]
  - "Each team member gets their own AWS Free Tier account. You can pool credits by sharing trained model artifacts across accounts via S3", using cross-account bucket policies or presigned URLs.
  - The Free plan auto-closes after 6 months or once the $200 is used up.
  - The top-500 extra credits are redeemed as promo codes under Billing > Credits.

---

## 3. Official clarifications during the event

- As of about 01:20 IST on 25 Sep, the only active organiser message on Unstop is the blog banner above (API `current_messages`, id 79892). [OFFICIAL]
- No public organiser answers were found on LLM use, GPUs, or LLM APIs during the event. Searches covered Reddit, LinkedIn, Builder Center, and web/Telegram/Discord queries. [Not found]
- Relevant precedents:
  - **LLM APIs (2024):** "we request that you refrain from using publicly or commercially available large language model (LLM) APIs such as those provided by OpenAI, Anthropic, Microsoft, Facebook, Google or other AI companies. The submissions using any of the LLM APIs will be discarded." Source: the 2024 ML round rules in `https://unstop.com/api/public/competition/1100713`; mirrored at https://www.scribd.com/document/863057657/Amazon-ML-Challenge-2024-1. [OFFICIAL 2024]
  - **Model license and size (2025):** the 2025 problem statement already said "Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters". It banned external lookup ("Any evidence of external price lookup or data augmentation from internet sources will result in immediate disqualification") and said "All submitted approaches, methodologies, and code pipelines will be thoroughly reviewed". Mirrors: https://huggingface.co/datasets/Rupesh2/Amazon-ML-Challenge and https://github.com/AdiSinghCodes/Amazon-ML-Challenge-2025/blob/main/README.md. **So pretrained open-weight encoders and LLMs of 8B or less under MIT/Apache were allowed in 2025 and used by top teams** (Qwen3-4B, SigLIP2, DINOv3, DeBERTa-v3; see §4). [3P mirrors of the official PDF]
  - **GPUs:** none provided in any edition. 2026 gives only AWS credits (§2). Poojan's advice: "40-80 GB of GPU memory will make things comfortable"; his team won 2023 on Kaggle GPUs. [3P]
- Whether an LLM API may be used **during development**, for example to label pairs or help write code, is not addressed anywhere public. **[UNVERIFIED]** Given the 2024 wording and the 2026 "use only the provided data" rule with code audits, the safe reading is that nothing in the submitted pipeline may call an API or the internet [INFERENCE].
- Participant-reported mechanics from 2025, relevant because Unstop runs the same platform:
  - "in the mail they said 5 valid submissions per day" (only valid, scored submissions counted, per that comment): https://www.reddit.com/r/learnmachinelearning/comments/1o4hezg/amazon_ml_challenge_2025/ [3P]
  - Upload failures ("failed due to 'tuple indices must be integers or slices, not str'… issue with the zip file"): https://www.reddit.com/r/Btechtards/comments/1ntmvrr/amazon_ml_challange_is_back_for_2025_i_won_it_2/ [3P]
  - "I uploaded code at 11:58pm, the leaderboard didn't update as time got over": https://www.reddit.com/r/learnmachinelearning/comments/1o4hezg/amazon_ml_challenge_2025/ [3P]. **Do not rely on last-minute uploads.**

---

## 4. Past editions

### 4.1 Summary table
| | 2023 | 2024 | 2025 |
|---|---|---|---|
| Platform | HackerEarth | Unstop (1100713) | Unstop (1560375) |
| Task | Product length from catalog text | Entity value extraction from product images | "Smart Product Pricing" (text + image to price) |
| Metric | `max(0, 100*(1-MAPE))` | Custom F1 (a wrong non-empty prediction counts as FP) | SMAPE (lower is better) |
| Data | 2.2M products; submission 734,736 rows | about 230K train images; 131,187 test rows | 75K train, 75K test |
| Public/private | n/a | Not published | **Public = 25K subset; final = full 75K + documentation** |
| Submission cap | No cap (one top team made ~200) | **15 total** | **5 per day** |
| Registrations / teams | ~26K participants, ~7K teams | 74,823 regs, 17,512 teams (API). Teams that actually submitted: "~1000" per one post, "2500+" per the winner | 82,787 regs, 19,556 teams (API) |
| Winning score | [not found] | F1 0.865 (winner's post) | ~39–40 SMAPE at the top of the public leaderboard |
| Winner | ART in Artificial Intelligence (leaderboard #1, then winner) | NeuralNinjas (leaderboard #1, then winner) | [not found]; 2nd runner-up 00_Team_Rocket (IIIT Delhi) |

Sources for the table:
- 2023 metric and data: https://github.com/ananyakapoor19/Amazon-ML-2023, https://github.com/VectorNd/Amazon-ML-Challenge-2023. Field size and 2nd place: https://github.com/greenfish8090/AmazonML (mirror https://github.laiyagushi.com/greenfish8090/AmazonML). Submission count: https://www.youtube.com/watch?v=C2z9kwx15MY [3P]
- 2024 rules and registration numbers: Unstop API 1100713 [OFFICIAL]. Test size: https://medium.com/@manoj632004s/my-amazon-ml-challenge-2024-experience-4ae6417261d9. "~1000 teams" and the score distribution: https://medium.com/@aman_prakash/how-we-secured-357th-spot-among-74-850-in-amazon-ml-challenge-2024-aef31f4e31b0. Winner: https://www.linkedin.com/posts/agrawal-kushal_amazonmlchallenge2024-machinelearning-activity-7244412615639080960-PmGD and https://github.com/KhadgaA/Amazon-ML-Challenge [3P]
- 2025 rules and registration numbers: Unstop API 1560375 [OFFICIAL]. Split: https://github.com/AdiSinghCodes/Amazon-ML-Challenge-2025/blob/main/README.md. Train/test sizes: https://www.linkedin.com/posts/rudr-pratap-singh_rank-11-in-amazon-ml-challenge-2025-just-activity-7386578786881822720-give [3P]

### 4.2 What top teams did
- **2023 #2 (greenfish8090):**
  - Fine-tuned BERT and RoBERTa with a learned product-type embedding, on a log-transformed and clipped target.
  - Snapped predictions to the nearest value seen in training, then took the minimum of the two models.
  - https://github.com/greenfish8090/AmazonML [3P]
- **2023 top-10 KNN approach:** "for every test product we identified the K nearest products from the training data and then on those nearest neighbors we trained the meta model." This retrieve-then-rerank pattern is structurally close to blocking plus a matcher: https://www.youtube.com/watch?v=C2z9kwx15MY [3P, speaker's team rank not independently verified]
- **2024 winner NeuralNinjas:**
  - Fine-tuned Qwen2-VL-7B-Instruct (QLoRA 8-bit, LLaMA-Factory) on 20K noisy samples.
  - Hand-curated **1,600 clean labels** and fine-tuned again on those.
  - F1 0.617 zero-shot, then 0.679 after SFT, then 0.865 after the curated second stage.
  - https://github.com/KhadgaA/Amazon-ML-Challenge [3P]
  - **Lesson: a small, carefully relabelled set beat more noisy data.**
- **2024 #6 DBkaScam:** an ensemble of MiniCPM-2.6 and Qwen2-VL-7B with zero-shot and few-shot prompts plus post-processing; best 0.718 F1 on their eval. https://github.com/arnav10goel/Amazon-ML-Challenge-24 [3P]
- **2025 2nd runner-up 00_Team_Rocket:**
  - DeBERTa-v3-large plus engineered features fused with cross-attention, trained with Smooth L1. **Text only, images dropped as noise.**
  - Validation 39.83 against public leaderboard 40.329.
  - "Occam's Razor did us well." They were briefly #1.
  - Finale jury: Ajay Srinivasamurthy, Arunita Das, Deepak Gupta.
  - https://github.com/parthrastogicoder/Amazon-ML-Challenge-2025-3rd and https://www.linkedin.com/posts/parth-rastogi-151444258_amazonmlchallenge-machinelearning-iiitdelhi-activity-7385315749113991168-Z3Ze [3P]
- **2025 SPAM_LLMs (IIT ISM):**
  - **Frozen** pretrained embeddings (Qwen3-4B text, SigLIP2 ~2B, DINOv3 ~0.8B) with modality-specific MLPs.
  - "3rd in Public LB, 5th in Private LB", 6th after the finale.
  - "Try out bigger models before finetuning smaller models."
  - https://github.com/RudrakshSJoshi/amlc-multimodal-mlp and https://www.linkedin.com/in/loki-silvres [3P]
- **2025 score ladder** (all public-leaderboard SMAPE unless noted):
  - rank 8: 40.777 (https://github.com/VishalTheHuman/Amazon-ML-Challenge-2025)
  - rank 17: 41.10 (https://github.com/siddeshrizwani/AmazonML-Price-Prediction-Transformer)
  - rank ~47: 42.254 (https://github.com/adityabagrii/Price-Prediction-via-Fine-tuning-DeBERTa-Large)
  - rank 112: 44.167 (https://www.linkedin.com/posts/sabari-ganesan-0970312a9_amazonmlchallenge-machinelearning-nlp-activity-7387032018460721152-eDjB)
  - rank 183: 44.8 (https://github.com/BhavyaGoyal777/AMAZON_ML_SOLUTION)
  - rank ~1600: ~54 (https://www.linkedin.com/posts/meghrajpatil_machinelearning-datascience-amazonmlchallenge-activity-7388583488939175936-uOqm)
  - The top 50 were packed within about 2 SMAPE points. [3P]
- **2024 score ladder:** "top 3 teams had scores around 80%, but by rank 50… 60%. By rank 250… 40%": https://medium.com/@aman_prakash/how-we-secured-357th-spot-among-74-850-in-amazon-ml-challenge-2024-aef31f4e31b0. This conflicts with the winner's 0.865 and may reflect a different leaderboard snapshot [3P].

### 4.3 Shortlisting, finale and lessons
- **Leaderboard #1 won in both 2023 and 2024:** "the trend for both years was the same: the team ranked #1 on the leaderboard was also the final winner". Poojan's team was leaderboard #5 in 2024 and finished **1st runner-up** after presentations. https://www.reddit.com/r/Btechtards/comments/1ntmvrr/amazon_ml_challange_is_back_for_2025_i_won_it_2/ [3P]
- **Public vs private leaderboard movement in 2025:**
  - SPAM_LLMs went from 3rd public to 5th private [3P, repo above].
  - One team: "my last solution wasn't evaluated due to which we ended up at 51 rank but in the final rankings we jumped to top 25."
  - Another: "I was 13th on public leaderboard, still didn't receive link."
  - https://www.reddit.com/r/learnmachinelearning/comments/1oc8ci3/amazon_ml_challenge_update/ [3P]
  - Final rankings are recomputed on the full test set, and code or doc review can drop teams.
- **Results were slow in 2025.** Results were scheduled for 14 Oct, but people were still asking on 17 and 18 Oct, and top-50 OA links arrived around 21 Oct. Sources: https://www.reddit.com/r/learnmachinelearning/comments/1o4hezg/amazon_ml_challenge_2025/ and https://www.reddit.com/r/learnmachinelearning/comments/1oc8ci3/amazon_ml_challenge_update/ [3P]
- **Plagiarism scrutiny in 2025.** Adjacent teams from the same college with near-identical scores (for example 42.106 / 42.106 / 42.107) were called out publicly: https://www.reddit.com/r/Btechtards/comments/1o77kk6/amazon_ml_challenge_2025/ [3P]. Together with the 2026 "audit your blocking approach" wording, this means an original, reproducible pipeline and a clear doc matter.
- **Poojan's playbook** [3P, same Reddit post]:
  - Day 1: every member tries a different approach against a holdout set.
  - Day 2: refine what works and "Don't try anything new."
  - Day 3: ensemble and protect your leaderboard rank.
  - Use LoRA and data sampling instead of training big models.
  - "It's relatively easy to get into the top 50 since the problem itself is difficult."
  - "ChatGPT and other AI tools suck for core ML tasks."
  - Finale: present the end-to-end solution with metrics for what worked and what didn't, cover limitations, scaling and latency, and be able to defend every component in depth.
- **How many submissions to save:**
  - 2026 allows 5 per day (15 total), and the live leaderboard ranks your **max** score.
  - The ZIP must match "the best submission", and its `matching_results.tsv` must be identical to the uploaded file.
  - So keep a versioned copy (outputs + commit hash + config) of every uploaded file, and be ready to rebuild whichever one scores best.
  - Nobody has published whether the private leaderboard scores the best-public, the last, or a team-selected submission. **[UNVERIFIED]**

---

## 5. Public dataset facts for 2026

- **None found.** Nobody has published row counts, singleton rates, S2/S3 sizes or country distribution for the 2026 data. The event was about 80 minutes old at search time.
- `https://www.kaggle.com/datasets/lashfire/amazon-ml-challenge-2026` is titled "2026" but predates the event: views start 25 Aug 2026 and it shows 6 files, 12 columns, 146.77 MB. It is **not** the 2026 entity-resolution data and was **not downloaded**. `https://www.kaggle.com/datasets/murtuza26/amazon-ml` says it uses the 2025 dataset. [3P]
- No leaked labels or submission files were sought or opened.

---

## Not found / unverified

| Item | What was tried | Status |
|---|---|---|
| 2026 public/private split % | Listing, API, AMP page, guidelines page, web search | Not published; 2025 was 33% public |
| 2026 dataset sizes, singleton rate, S2/S3 sizes, country mix | Web, Reddit, Kaggle, LinkedIn, X/Telegram queries | Nothing public yet |
| Unstop FAQ/Discussions entries | Headless fetch, AMP page, API guesses (`/api/public/competition/discussions/1743604` returns an empty 200) | Login-gated; Chrome extension not connected. **Check while logged in.** |
| Which submission the private leaderboard uses (best, last, or selected) | Round text, guidelines, 2025 anecdotes | Ambiguous ("final solution submission" vs "best submission" in the ZIP spec) |
| Whether the fullscreen-violation limit (5) applies to the ML round | API field only | Unverified |
| Whether LLM APIs may be used during development | 2026 pages; 2024 precedent found | No 2026 statement |
| Whether GPUs are provided | Listing, blog, guide PDF | No evidence of GPUs; only Free Tier credits (CPU SageMaker hours) |
| "Code Submission" round dates (8 to 9 Oct) | API | Conflict with the 7 Oct finale; probably placeholders |
| Meaning of `result_count: 100` | API | Unknown |
| 2025 1st-place and 1st runner-up teams | Exa, Firecrawl, LinkedIn searches | Not found |
| 2023 winning score | Repos and posts | Not found (one unverified repo claims top "63.1") |
| Contents of the Twitch VOD of the 21 Sep session | Only the companion blog was read | Not reviewed |
| Official Discord/Telegram channel | Web search | None found; 2024 had an in-round "link to ask any queries" |
