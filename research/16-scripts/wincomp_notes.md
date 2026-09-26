# Winning-solution sweep (FSQ / Shopee / SIGMOD / generator inversion) -- web-research agent, 26 Sep 2026
(notes.md in this dir is shared with another agent and was overwritten mid-run; this file is the full copy)

## FSQ 1st re:waiwai https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336055 ; code https://github.com/TakoiHirokazu/Kaggle-Foursquare-Location-Matching
- S1: 100 cands/id latlon-euclid + 100 by name-emb cos (bert-base-multilingual-uncased, cuml knn); LGBM few feats -> top20 each (~40/id). Max IOU 0.979
- S2: ~120 feats LGBM CV 0.875, thr 0.01 -> ~10% of cands remain; max iou 0.974 after S2 (Takoi comment)
- S3: CatBoost CV .878; xlm-roberta-large (+~70 num feats, 3ep); mdeberta-v3-base (FGM+EMA, 4ep) CV .907; ensemble CV .911 thr .5
- S4: PP merge id-match-sets if common ids >50%; XLM-R-large rescoring new pairs thr .02; CV .9166
- LEAK (stage-2 sub LB): no merge .900; (1) add train-train TP .943; (1)+(2) .957; (1)+(2)+(3) .971
- Private/Public: overfit no-merge .946/.947; merge .976/.976; non-overfit v1 no-merge .941/.941, merge .977/.978
- HW: n1-standard-64 CPU 240GB + 1x A100
## FSQ 2nd 2:30: LB 0.949 no leak / 0.971 leak https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336090 ; 336062 ; 336072
- 128 cands/id (latlon NN + name-tfidf NN, country-specific ratios) Max IoU 0.9895
- Transformer candidate blocking: single LB 0.918; thr 0.005 + union-find -> avg 4.1 cands/id, Max IoU 0.986
- XGB + LGBM (edit dists x11 methods x8 fields, jaccard/dice/simpson, tfidf word+char cos) + xlm-roberta-base cross-encoder (dropout 0, 5 ep); CV purposely leaky GroupKFold(5, src_id)
## FSQ 3rd Psi https://www.kaggle.com/competitions/foursquare-location-matching/writeups/psi-3rd-place-solution
- ArcFace xlm-roberta-large (class=POI), latlon as text; all-pairs cos on GPU + threshold; recall "somewhere in range of 0.97 recall maybe?"
- stage2 pair-concat transformer ("bi-encoder"), blend with stage1; "minor QE post processing, but not much was helping"; 90 min per 600k records
## Kaggle no-overlap rescore https://www.kaggle.com/competitions/foursquare-location-matching/discussion/338035
Ri 1(new)/10(old) .931957/.932782/.948152 ; merge master 2/13 .922713/.922937/.946495 ; Ethan&qyxs&hyd 3/14 .922516/.922847/.945548 ; re:waiwai 4/1 .920238/.920594/.977683 ; Psi 5/3 .919703/.919698/.967847
- leak ~67% test rows; self-match-only 0.65; leak-only 0.884 (74th) https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335799
## FSQ 13th / no-overlap 2nd (213tubo GNN) https://www.kaggle.com/competitions/foursquare-location-matching/writeups/team-merge-master-13th-place-solution-gnn
- 60 cands (30 tfidf text + 30 haversine); LGBM -> 3.7M pairs Max IoU .971; xlmr-base + mdeberta Ditto-style; GNN 2-hop subgraph (max iou .993), IoU loss; GNN "+0.02x"; public .907 (20 cands) -> .924 (60 cands) -> .946 (GNN)
## FSQ 4th (GBDT only) https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335810
- cands: NN, name sim, same words, category group, phone, address, tfidf (<3% of TPs); LGBM prefilter thr<0.007 then 200+ feat 20-fold LGBM on 1.1M rows; PP = group-size-adaptive threshold; 0.939 -> 0.957 with leak TPs only
## FSQ LB https://www.kaggle.com/competitions/foursquare-location-matching/leaderboard : 1 .97768, 2 .97260, 3 .96847, 4 .95717, 10 .94815, 13 .94649, 14 .94554, 20 .92772
## Shopee 1st https://www.kaggle.com/competitions/shopee-product-matching/discussion/238136 ; LB https://www.kaggle.com/competitions/shopee-product-matching/leaderboard (1 .780, 2 .779, 3 .779, 4 .777, 5 .767, 10 .760, 20 .754)
- pub LB: .7 img/.64 txt; concat .724; min2 .743; normalize-then-concat .753; full data .757; union comb/img/txt .776; INB+diverse txt .784; joint thr .793; INB CV .906->.925; iteration CV .9225->.9256 (+~.002 LB); no 2nd-stage model
## SIGMOD 2021 https://dbgroup.ing.unimore.it/sigmod21contest/leaders.shtml (1 .965, 3 .957, 5 .935, 6 .888, 10 .838); winner regex+alias+rule blocking, no ML: https://dbgroup.ing.unimo.it/sigmod21contest/posters/SUSTech_DBGroup.pdf
## SIGMOD 2022 https://dbgroup.ing.unimore.it/sigmod22contest/leaders.shtml (1 .529, 2 .520, 3 .514, 9 .501, 10 .487, 20 .375, baseline .191); generator disclosed: https://dl.acm.org/doi/pdf/10.1145/3615952.3615965
## Generator inversion: Instant Gratification (QDA .966/.970 -> GMM .975; 1st .97598); Santander (.901 -> .913 removing fake rows -> .922/.925); Quora (graph intersection .20 -> .157)
