# top1 sweep notes (26 Sep ~23:15 IST)
- harshgitty58/Amazon_ML_Challenge PROGRESS.md: LB 0.967/0.969; top3 on 26 Sep 0.9906, 0.9894, 0.9888; probes France-blank 0.836, India-blank 0.580, US-blank 0.578; test 5.75 S2/S3 per S1 vs 4.67 train
- HarpalKalankar: LB 0.967; leader 0.9884 at writing
- Karthikatstuffmama: LB 0.945193 baseline; top-10 cut 0.986
- rishabhiitj25/AmazonML26: LB v1 0.969, stacked v2 0.965; OOF 0.9872
- Sid-techweb/AmazonML-New: LB 0.970441 (local 0.98321); sibling-feature run E10 LB 0.933 withdrawn
- pkd-prashant/amazon-mlss: v3 LB 0.921 "stage-2 group features fooled by sibling businesses in test"

# SHOPEE private LB https://www.kaggle.com/competitions/shopee-product-matching/leaderboard
1 Upstage 0.780 | 2 lyaka&tkm 0.779 | 3 Btbpanda 0.779 | 4 Watercooled (Psi,Dieter,Pascal) 0.777 | 5 0.767 | 6 0.764 | 10 shimacos 0.760 | 14 Chris Deotte 0.759 | 20 0.754

# SIGMOD 2022 blocking (synthetic) 
- Organizers paper https://dl.acm.org/doi/pdf/10.1145/3615952.3615965 : 2022 datasets SYNTHETIC ~1M records each, script: (i) new tuple by picking first word from random tuple, 2nd from another...; (ii) matches by randomly shuffling words, deleting some words, changing letter case. 2020: top-5 all reached 0.99 F-measure (tie broken by runtime). Best 2020/21 approaches "all optimized for the provided datasets", regex brand/model extraction, rule-based matching; 2021 some RF / XGBoost + rules.
- Winner WBSG (Mannheim) poster https://dbgroup.ing.unimo.it/sigmod22contest/posters/WBSG.pdf : avg recall 0.529 (D1 0.713, D2 0.345), runtime 1914s; 16 CPU/32GB no GPU, 35 min; xtremedistil-l6-h256 + supcon + extra CommonCrawl schema.org data; FAISS IVF+PQ; rerank avg(Jaccard, cosine). "characteristics of products in D2 remain unknown"
- 2nd? April poster: avg recall 0.520 (D1 0.743, D2 0.297) similarity joins; "Hindsight: Could use the training data ground-truth to boost the recall"
- SUSTech poster: X1 0.692, X2 0.323, 1671s, regex feats + BERT triplet + HNSW
- 11th: avg 0.469 (D1 0.71, D2 0.227) (theodoratrz github)

# SIGMOD 2022 LB https://dbgroup.ing.unimore.it/sigmod22contest/leaders.shtml
1 WBSG 0.529 (1914s) | 2 April 0.520 | 3 QaisHousien 0.514 | 4 SUSTech 0.507 | 5 RUPikachu 0.507 | 6 HyTUM 0.504 | 9 0.501 | 10 0.487 | 20 0.375 | baseline 0.191
# SIGMOD 2021 LB (avg F over Notebook, NotebookLarge, Altosight) https://dbgroup.ing.unimore.it/sigmod21contest/leaders.shtml
1 SUSTech 0.965 | 2 UKN 0.957 | 3 panda 0.957 | 4 CyberPunk 0.946 | 5 BoomBoomChicken 0.935 | 6 0.888 | 10 0.838
- SIGMOD 2021 winner SUSTech poster https://dbgroup.ing.unimo.it/sigmod21contest/posters/SUSTech_DBGroup.pdf : regex feature extraction (brand, capacity), alias tables, per-brand primary-key features; 3-stage blocking (complete/recycle/residual) = matching (all pairs in block). X2 P .995 R .971 F .983; X3 .991/.980/.986; X4 .980/.880/.927. NO ML.
- SIGMOD 2022 3rd QaisHousien: sorted neighbourhood + Jaccard over spaCy noun tokens; recall .726/.301 avg .514

# GENERATOR INVERSION (non-ER)
- Instant Gratification: data from sklearn make_classification; n_clusters_per_class=3 inferred by EDA. QDA+PL LB 0.970 -> GMM 6 ellipses LB 0.975 (Chris Deotte https://www.kaggle.com/competitions/instant-gratification/writeups/chris-deotte-how-to-score-lb-0-975). "perfect classifier" private 0.97560. 1st place CV .9754 pub .9744 priv .97598 (https://www.kaggle.com/competitions/instant-gratification/writeups/whoami-1st-place-solution-sharing). QDA alone ~0.966 (SemenovAlex github). hakubishin3 github: inferred params n_clusters_per_class=3, hypercube=True, flip_y=0.05, class_sep=1
- Santander: synthetic fake test rows (YaG320 kernel). jfpuget: freq on all data 0.901 pub; remove fake rows when computing freq 0.913; NB 200 lgb 0.922; FE+ensemble 0.925 (https://www.kaggle.com/competitions/santander-customer-transaction-prediction/writeups/gryffindor-public-lb-0-922-magic-notebook). 1st: NN .92687 pub/.92546 priv; train-only uniqueness .910->.914 LB; with real-test uniqueness .921 (https://www.kaggle.com/competitions/santander-customer-transaction-prediction/writeups/wizardry-1-solution)
- Quora: graph intersection-count of common neighbours (graph built from how pairs were sampled) "improvement from 0.20 to 0.157" logloss https://www.kaggle.com/competitions/quora-question-pairs/discussion/33287 ; k-core ~0.005 LB https://www.kaggle.com/c/quora-question-pairs/discussion/33371 ; 3rd place: best single xgb 0.185 CV, stack 0.157 CV; freq-based odds adjustment factor 4.7 https://www.kaggle.com/competitions/quora-question-pairs/writeups/jared-turkewitz-sjv-overview-of-3rd-place-solution

# FSQ 4th (Vincent, Youri, Theo) https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335810  -- ALL GBDT, no transformer
- cands: near neighbours, name sim, same words, grouped-category equality, cleaned phone equality, same address, tfidf ("less than 3% of true matches added")
- LGBM1 5-fold few feats prefilter thr <0.007; LGBM2 20-fold 200+ feats; trained on full 1.1M rows off-Kaggle
- PP key idea: adapt threshold to sizes of groups being merged
- leak TP-only boost 0.939 (15th) -> 0.957 (4th); key cleaned_name+round(lat,5)+round(lon,5); did not use leak to delete FPs
- XGB/CatBoost stacking gain "very poor"
- pkd-prashant/amazon-mlss Documentation_template.md (Team Beyond Baseline): PUBLIC LB 0.9779 "rank 410 at the time of submission"; v10r = v9 (55-feat LGBM + xlm-roberta-base cross-encoder stacked on uncertain pairs, LaBSE for Indic) + transductive France vocab adaptation from test pseudo-labels (p>0.97/p<0.03) + exact-identity restore; perfect-matcher ceiling 0.989; test 5.75 vs 4.68 recs/S1; sibling house-number mismatch 34%/67% more frequent in test; stage2 group feats 0.956->0.921
- rithishbarathn/amazon-ml-challenge-2026 submissions/VERSIONS.md: v6 LB 0.97509 (rank 249), v3 LB 0.964; France-empty probe 0.842; France 14.98% test S1 -> US+India ~0.981, France ~0.944
- YatharthJangid/amazon_ml REPORT_v3.md (26 Sep 20:30 IST): "public-leaderboard top 5 are at 0.989-0.991"; "48 h / Top-500 credit cut-off is 27 Sep 00:00 IST"; small candidate set per S1 "ranked higher in final evaluation"
- Karthikatstuffmama: LB 0.945193; "Top-10 cut 0.986"; 92.2% errors FN; blocking_miss 63.1%
- Tony-AJ: public now 0.966 (v107)
- Unstop API: /api/public/competition/1743604 works (registerCount 89399; round 1593683 = CodeContests 290344 "ML Challenge | 72-Hour Hackathon", live_leaderboard=1, players_count 1648, ends 2026-09-27T23:59:49+05:30; round "Top 500 Teams" 27 Sep 00:00-23:59; results 2 Oct; finale 7 Oct). No leaderboard endpoint found (15+ guesses 404).
- Reddit r/Btechtards 1wqp2n5 (post 26 Sep 17:46 IST, via api.pullpush.io comments up to 21:54 IST): Responsible-Mix9555 20:21 IST "I'm on the team currently ranked #1" (MTech student w/ full-time job; won't reveal approach); normiehubro 20:41 "stuck at 0.985"; Simple_Letterhead296 21:23 "stuck on 0.9847"; Correct_Scene143 21:33 "rank is still 700 at 0.967"; Puzzled-Ad8231 18:56 "On telegram people are selling top 50 for 10k"; Toad__Sage__ 21:08 union embeddings+mapping candidates
- Unstop round page 290344 (firecrawl): "ranked on the maximum score of the submission and submission time" (ties -> earlier wins)
- 2023 winner notebook (pj-mathematician): 5 sentence-transformers -> top-10 KNN neighbour labels x4 + top-100 HNSW ANN labels = 140 features -> LightGBM custom log-error objective, extra_trees, 5000 trees, num_leaves 2^15, GPU
- 2025: winning SMAPE 39.7 (NeelDevenShah README); rank 8 CTRL+ALT+DEV CLIP+DistilBERT fusion 40.777; rank 11 Qwen-3 1.7B + Liger; rank 20 EmbeddingGemma+SigLIP2 cross-attn; IIT Patna rank 5 & 24; winner name NOT FOUND
