# Amazon ML Challenge 2026, Business Entity Resolution: technical playbook

Research pass on 25 Sep 2026. The problem facts come from `../READ-NOTES.md` and `../source/problem_statement.txt`. Every external claim has a URL next to it.

**Tags used below**
- **[V]**: verified from a primary source (paper, official doc, model card, or LICENSE file) during this pass.
- **[V-API]**: licence and parameter count read from the Hugging Face model API (`cardData.license`, `safetensors.total`) on 25 Sep 2026.
- **[D]**: derived here from the metric definition (arithmetic, no citation needed).
- **[E]**: engineering judgement or estimate. Treat it as a hypothesis to test on validation.
- **[U]**: unverified.

---

## 0. Recommended stack

| Stage | Choice | Licence | Why |
|---|---|---|---|
| Normalise | Your own Python: stdlib `unicodedata` NFKD accent-fold; legal-form and abbreviation lists written as code constants (seeded from cleanco, the libpostal dictionaries and the Indian MCA Rule 8 list); a landmark splitter; a number extractor | stdlib (PSF); cleanco [MIT](https://github.com/psolin/cleanco/blob/master/LICENSE.txt); libpostal [MIT](https://github.com/openvenues/libpostal/blob/master/LICENSE) | No external data at runtime. Works for any country label (§4, §5). |
| Blocking (produces `candidate_pairs.tsv`) | Union of: **(a)** char-3-gram TF-IDF on the name, top-k in both directions via `sparse_dot_topn`; **(b)** the same on the address; **(c)** rare-token and number keys (postcode-like digits, house number); **(d)** from Day 2, dense top-k from `intfloat/multilingual-e5-small` or `-base` in `faiss-cpu`. Then a cheap stage-1 LightGBM keeps the top-K per S1. | sparse_dot_topn [Apache-2.0](https://github.com/ing-bank/sparse_dot_topn/blob/master/LICENSE); faiss [MIT](https://github.com/facebookresearch/faiss/blob/main/LICENSE); e5 MIT [V-API] | Top-k TF/IDF is a very strong blocker ([Sparkly, PVLDB'23](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf)). A union of lexical and dense blockers adds recall at little extra size ([DeepBlocker, PVLDB'21](https://www.vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf)). The video shows a match whose address is only a landmark, so names and addresses each need their own block. |
| Matcher | **LightGBM** on about 60 to 100 language-agnostic pair features, including "competition" features (rank and margin against the other candidates). From Day 2, add the probability from a fine-tuned multilingual cross-encoder as one more feature. | LightGBM [MIT](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE); cross-encoder base `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0, 118M) or `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M) [V-API] | On business names and addresses, a random forest over many string comparators beats any single comparator by about 6% F1 ([US Census MAMBA](https://ngoldschlag.github.io/papers/squeeze.pdf)). PLM matchers are the accuracy state of the art ([Ditto](https://arxiv.org/abs/2004.00584)) but are brittle under distribution shift ([RobEM, CIKM'22](https://ehsk.github.io/assets/pdf/CIKM22_RobustEM.pdf)), so they go in as a feature, not as the only decider. |
| Decision | Isotonic calibration, then **per-S1 expected-F0.5 set selection** (GFM-style, §6), plus a **per-S2/S3-record exclusivity** rule (each S2/S3 record goes to at most its best S1), plus an entity-level "has any match" model. Tune everything on **leave-one-country-out** validation. | sklearn (BSD-3) or your own code | The metric is macro, per entity, with singletons. The rule for the first link differs from the rule for extra links (§6.1). |
| Optional LLM judge (Day 3) | `Qwen/Qwen2.5-7B-Instruct` (7.62B) or `Qwen/Qwen3-4B-Instruct-2507` (4.02B), both Apache-2.0 [V-API]. Run it only on the uncertain probability band, via `mlx-lm` on the Mac. | mlx-lm [MIT](https://github.com/ml-explore/mlx-lm/blob/main/LICENSE) | LLM matchers are more robust to unseen entities ([Peeters et al., EDBT'25](https://arxiv.org/abs/2310.11244)), which is the France problem. **Keep the summed parameter count of all neural models at or below 8B** (§5.5). |

**What "done" means on Day 1:** a pipeline that validates, a blocking recall ceiling and reduction ratio reported on the validation split, and a LightGBM matcher with thresholds tuned for macro F0.5.

---

## 1. How the metric shapes every choice [D]

Per S1 entity, F0.5 = 1.25·P·R / (0.25·P + R), averaged over all S1 entities, with singletons included. An empty prediction on a singleton scores 1, and any prediction on a singleton scores 0 ([problem statement](../source/problem_statement.txt)). A non-empty truth with an empty prediction is assumed to score 0 (P is undefined).

Worked values, where T is the truth set and H is the prediction:

| T (truth) | H (predicted) | F0.5 |
|---|---|---|
| {a} | {a} | 1.000 |
| {a,b} | {a} (missed one) | **0.833** |
| {a,b,c} | {a} (missed two) | 0.714 |
| {a} | {a,b} (one false merge) | **0.556** |
| {a,b} | {a,b,c} | 0.714 (the official example) |
| {a,b} | {a,c} | 0.500 |
| ∅ | anything | 0 |
| non-empty | ∅ | 0 |

What follows from this:
1. **Recall loss is cheap and false merges are expensive.** Missing half of an entity's matches still scores 0.83. One wrong extra link on a 1-match entity drops it to 0.56.
2. **The first link and the extra links need different bars.** In the simplest two-hypothesis model, predict the top candidate `a` over ∅ iff P(a∈T)·E[F | a∈T] > P(T=∅). When `a` is the only plausible match, that is **p > 0.5**. When `a` is already confirmed, adding a second candidate `b` pays off only if **q > 0.727**, and a third only if **r > 0.759**. These are derived from the table above. For comparison, the global-threshold result for calibrated scores is threshold = F*/(1+β²) = 0.8·F*; for F1 it is F*/2 ([Lipton et al. 2014](https://arxiv.org/abs/1402.1892)), and the F0.5 version follows from the same argument.
3. **An entity-level "does this S1 have any match?" estimate is worth building.** The empty-vs-top-1 decision compares against P(T=∅), not against 1−p(top).
4. **Cheap leaderboard probe (optional).** An all-empty submission scores exactly the singleton fraction of the public-LB subset. That reveals the test prior, including France's effect on it, for one of the 5 daily submissions.
5. **Structural constraint:** S1 is deduplicated, so each S2/S3 record belongs to at most one S1 entity. S1 may still own several S2/S3 records (one-to-many). §6.3 covers how to exploit this.

---

## 2. SOTA entity-resolution approaches: what is practical in 3 days

| Approach | What it is | Code licence | 3-day verdict for this task |
|---|---|---|---|
| **Ditto** ([arXiv 2004.00584](https://arxiv.org/abs/2004.00584), PVLDB 14(1)) | Fine-tuned BERT/RoBERTa cross-encoder over serialised pairs, with domain-knowledge injection, summarisation and augmentation. Up to +29% F1 over the previous SOTA. **On two company datasets (789K × 412K records) it reached 96.5% F1.** | [Apache-2.0](https://github.com/megagonlabs/ditto/blob/master/LICENSE) | Use the *idea*: serialise `COL name VAL … COL address VAL …` and fine-tune a multilingual cross-encoder. Don't port the repo (old deps). |
| **HierGAT** ([SIGMOD'22](https://doi.org/10.1145/3514221.3517872)) | Hierarchical graph attention plus a Transformer. Up to +8.7% F1 over Ditto. HierGAT+ decides collectively over candidates. | [MIT](https://github.com/CGCL-codes/HierGAT/blob/main/LICENSE) | Research-grade and too slow to adapt. Borrow the *collective* idea through competition features (§6.3). |
| **Sudowoodo** ([arXiv 2207.04122](https://arxiv.org/abs/2207.04122)) | Contrastive self-supervised representations for matching and blocking, useful with few labels. | [BSD-3](https://github.com/megagonlabs/sudowoodo/blob/main/license.md) | Not needed, since train labels exist. The contrastive idea could adapt the blocker to France without labels, but that is out of scope for 3 days. |
| **AnyMatch** ([arXiv 2409.04073](https://arxiv.org/abs/2409.04073)) | Small LM (GPT-2 or T5-base per the [repo](https://github.com/Jantory/anymatch)) fine-tuned for *zero-shot* EM. Second-highest F1 across 9 benchmarks, within 4.4% of GPT-4 MatchGPT, at 3,899× lower cost. | **Repo has no LICENSE**; HF model `utokyo-dbgroup/AnyMatch` returns 401 [U] | The idea is relevant to France (transfer to unseen data). Reimplement it with an MIT/Apache base if at all. Low priority. |
| **Unicorn** ([SIGMOD Record 2024 summary](https://sigmodrecord.org/publications/sigmodRecord/2403/pdfs/12_unicorn-fan.pdf)) | Unified multi-task matcher: PLM encoder, then mixture-of-experts, then a binary classifier. Supports zero-shot. | **[Repo](https://github.com/ruc-datalab/Unicorn) has no LICENSE** | Skip: unclear licence and heavy. |
| **JedAI / pyJedAI** | Blocking, filtering, matching and clustering toolkit, including Unique-Mapping Clustering. | [Apache-2.0](https://github.com/AI-team-UoA/pyJedAI/blob/main/LICENSE) | Optional. Its clustering algorithms are easy to reimplement (§6.3). |
| **Magellan / py_entitymatching** ([PVLDB 2016](http://www.vldb.org/pvldb/vol9/p1197-pkonda.pdf)) | How-to-guided EM in the PyData stack: blockers, auto-generated similarity features, sklearn matchers. | [BSD-3](https://github.com/anhaidgroup/py_entitymatching/blob/master/LICENSE) | The *recipe* (similarity features + GBDT/RF) is exactly Day 1. Writing it directly with rapidfuzz is faster than the package. |
| **Splink** (Fellegi-Sunter) | Unsupervised EM over m/u probabilities, with term-frequency adjustments. "Links a million records on a laptop in around a minute" ([README](https://github.com/moj-analytical-services/splink)). | [MIT](https://github.com/moj-analytical-services/splink/blob/master/LICENSE) | Skip as the matcher, since supervised GBDT wins with labels. Steal **TF adjustment**: a match on a *common* token ("Enterprises", "Traders", "Pharmacy") is weaker evidence ([docs](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html), [FS theory](https://moj-analytical-services.github.io/splink/topic_guides/theory/fellegi_sunter.html)). |
| **dedupe** | ML fuzzy matching trained from human-labelled pairs ([README](https://github.com/dedupeio/dedupe)) | [MIT](https://github.com/dedupeio/dedupe/blob/main/LICENSE) | Skip (built for interactive labelling). |
| **recordlinkage** | Indexing (blocking, sorted neighbourhood), comparators, classifiers ([README](https://github.com/J535D165/recordlinkage)) | [BSD-3](https://github.com/J535D165/recordlinkage/blob/master/LICENSE) | Usable for sorted neighbourhood, but not needed. |
| **ZeroER** ([arXiv 1908.06049](https://arxiv.org/abs/1908.06049)) | GMM over similarity vectors with zero labels. Comparable to supervised on 5 benchmarks. Uses transitivity. | [Apache-2.0](https://github.com/chu-data-lab/zeroer/blob/main/LICENSE) | Idea for France: fit an unsupervised match/unmatch mixture on French candidate features to sanity-check the supervised model's score distribution. |
| **LLM EM**: [Peeters, Steiner & Bizer, EDBT'25](https://arxiv.org/abs/2310.11244) | The best LLMs need no or few examples to match PLMs fine-tuned on thousands, and are **more robust to unseen entities**. The prompt must be tuned per model and dataset. | n/a | Main reason to keep an LLM judge as a Day-3 option for France. |
| MatchGPT / [Using ChatGPT for EM](https://arxiv.org/abs/2305.03423) | ChatGPT zero-shot 82.35% F1, where RoBERTa needed 2,000 examples to reach similar performance. Demonstrations add up to +7.85%. | [Repo](https://github.com/wbsg-uni-mannheim/MatchGPT) has no LICENSE | Prompt ideas only. A hosted LLM is banned (APIs). |
| [Fine-tuning LLMs for EM](https://arxiv.org/abs/2409.08185) | Fine-tuning helps small LLMs and in-domain transfer, **but hurts cross-domain transfer**. | n/a | Warning: a LoRA fine-tuned on US+India could transfer *worse* to France than zero or few-shot. Validate it with leave-one-country-out. |
| **ComEM** ([arXiv 2405.16884](https://arxiv.org/abs/2405.16884)) | Compares match, compare and select strategies. **"Select"** (show the LLM the anchor plus all its candidates and ask which match) wins because it uses record interactions. Compound: a cheap matcher filters, an LLM selects. | n/a | Right LLM prompt shape for this task: one prompt per S1 with its top candidates. |
| **BatchER** ([arXiv 2312.03987](https://arxiv.org/abs/2312.03987)) | Batch prompting with covering-based demonstration selection, which cuts cost. | n/a | Only if the LLM stage becomes the bottleneck. |

### What works on company name + address specifically
- **US Census MAMBA** ([paper](https://ngoldschlag.github.io/papers/squeeze.pdf), [CES-WP-18-46](https://www.census.gov/library/working-papers/2018/adrm/ces-wp-18-46.html)) [V] is the closest published analogue: business names and addresses. Pipeline:
  1. Standardise abbreviations and suffixes.
  2. Block geographically (5-digit ZIP, city, 3-digit ZIP, state), then a Soundex word block.
  3. Run a random forest over many string comparators. It gains about 6% F1, 6.5% recall and 3% precision over single comparators. The best single comparator was 3-character LCS.
- **IBM "Fast Record Linkage for Company Entities"** ([arXiv 1907.08667](https://arxiv.org/abs/1907.08667)) [V]: MinHash blocking plus ML *short-name extraction* ("Cisco Systems, Inc." → "Cisco"). Recall 91% vs 73% for the baseline.
- **Ditto on company data**: 96.5% F1 on 789K × 412K company records ([arXiv 2004.00584](https://arxiv.org/abs/2004.00584)) [V].
- **Legal form handling matters**: a hybrid rule-plus-ML legal-form extractor worked best for company matching ([Kruse et al. 2021](https://www.tib-op.org/ojs/index.php/bis/article/view/44)). Legal-form classification over 1.1M LEI names in 30 jurisdictions uses the ISO 20275 ELF codes ([arXiv 2310.12766](https://doi.org/10.48550/arxiv.2310.12766)) [V].
- **Ranking for these 3 days [E]:** similarity features + GBDT on Day 1, then a cross-encoder probability stacked into the GBDT, then an LLM only on the ambiguous band.

---

## 3. Blocking and candidate generation

### 3.1 Evidence [V]
- **Top-k beats thresholding.** In Sparkly, true-match similarity scores spread over the whole [0,1] range on noisy data, so a threshold either loses recall or explodes the output. Top-k keeps recall at a fixed output size. IDF is essential (removing it hurts badly). BM25 over 3-grams beat 8 SOTA blockers on 15 datasets. Probing from the larger table gives higher recall, and adding the other direction gave minimal recall gain in their tests ([PVLDB'23](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf), [TR](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-tr22.pdf)).
- **Deep blockers:** all 8 DeepBlocker variants reach >90% recall with small candidate sets. **Unioning** a DL blocker with a rule blocker raised recall by 0.3–6.7 points on structured, 8.5–14.5 on textual and 0.2–9.9 on dirty data, for a size increase of up to 49.9% / 57.6% / 12.3% ([PVLDB'21](https://www.vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf)).
- **Pre-trained embeddings:** SentenceBERT-family models gave the best blocking recall of 12 LMs over 17 datasets. S-GTR-T5 at k=10 beat DeepBlocker's recall by about 15% on 8 datasets ([Zeakis et al., PVLDB 16(9)](https://www.vldb.org/pvldb/vol16/p2225-skoutas.pdf)).
- **Classic techniques** (token or standard blocking, sorted neighbourhood, q-gram, canopy, suffix array): [Christen, TKDE 2012](https://users.cecs.anu.edu.au/~christen/publications/christen2011indexing.pdf) and the [Papadakis et al. survey](https://arxiv.org/abs/1905.06167). Christen's advice is to use several blocking keys over different fields, so each true match shares at least one.
- **MinHash LSH** (Jaccard threshold) is available in [datasketch](https://github.com/ekzhu/datasketch) (MIT) and used at IBM for company names ([arXiv 1907.08667](https://arxiv.org/abs/1907.08667)). For a top-k style it is weaker than TF-IDF top-k [E].

### 3.2 Recommended design [E]
1. **Partition by the normalised `country` string**, never by a hard-coded list; treat it as an open set. If the label can be missing or disagree across sources, add a no-partition fallback. Check the train data first.
2. **Name block (B1):** `TfidfVectorizer(analyzer='char_wb', ngram_range=(3,3), sublinear_tf=True)` on the *normalised core name*, with the legal form stripped and the DBA part split out. Use `sparse_dot_topn.sp_matmul_topn(A, B.T, top_n=k)` in **both directions**: each S1 gets its top-k from S2∪S3, and each S2/S3 record gets its top-k′ S1. The reverse direction also feeds the exclusivity features. On an Apple M2 Pro, `sparse_dot_topn` is "up to 6 times faster" on 20k×193k TF-IDF matrices keeping the top 10 with 8 cores, and ships macOS ARM wheels ([README](https://github.com/ing-bank/sparse_dot_topn)) [V].
3. **Address block (B2):** the same, on the normalised address with landmark clauses kept, top-k. This is what catches "Acme Robotix" vs "Acme Robotics" when names differ in the S1 index.
4. **Key blocks (B3):** exact shared (postcode-like digit run) plus (first letter of core name); (house number) plus (a rare street token); any shared name token with IDF above the 90th percentile. Cap each block's size to drop chains and very common tokens.
5. **Dense block (B4, Day 2):** `intfloat/multilingual-e5-small` (MIT, 118M) or `-base` (MIT, 278M) on `"query: {name} | {address}"`, cosine top-k via `faiss.IndexFlatIP`. `faiss-cpu` has a macOS arm64 wheel. `hnswlib` 0.8.0 has **no macOS arm64 wheel on PyPI** and needs a compiler, checked 25 Sep 2026 via the PyPI JSON API ([pypi hnswlib](https://pypi.org/project/hnswlib/)).
6. **Union, then prune:** a stage-1 LightGBM on cheap features (the B1–B4 scores and ranks) keeps the top-K per S1, K≈20–50. **This pruned set is `candidate_pairs.tsv`**, because the rules say the file must be the exact set the model scores ([problem statement](../source/problem_statement.txt)).

### 3.3 Reporting the recall ceiling and reduction ratio
Definitions (pair completeness, reduction ratio: [Christen 2012](https://users.cecs.anu.edu.au/~christen/publications/christen2011indexing.pdf); [Papadakis survey](https://arxiv.org/abs/1905.06167)):
- N_all = |S1| × (|S2| + |S3|)
- **RR** = 1 − |C| / N_all. Example [D]: K=30 per S1 with |S2|+|S3| = 100k gives RR = 0.9997.
- **PC (recall ceiling)** = |C ∩ M| / |M|
- **PQ** = |C ∩ M| / |C|
- **Oracle macro-F0.5 ceiling [D]** (the number that matters). For each S1 entity *e*, where C_e is its candidate set and T_e its true matches:
  - If T_e = ∅: score 1 (a perfect matcher predicts empty).
  - If C_e ∩ T_e = ∅: score 0.
  - Otherwise: with R_e = |C_e ∩ T_e| / |T_e|, score 1.25·R_e / (0.25 + R_e).
  - The ceiling is the mean over all S1 entities.
- Also report mean, p95 and max |C_e|, and the % of S1 with an empty C_e, split **by country**. Put a curve of PC vs K per blocker, plus the union, in the methodology doc.

```python
def blocking_report(cands: dict, gold: dict, n_right: int):
    # cands, gold: S1 id -> set of S2/S3 ids; gold has an entry (possibly empty) for every S1
    tot_c = sum(len(v) for v in cands.values()); tot_m = sum(len(v) for v in gold.values())
    hit = sum(len(cands.get(e, set()) & g) for e, g in gold.items())
    oracle = []
    for e, g in gold.items():
        if not g: oracle.append(1.0); continue
        r = len(cands.get(e, set()) & g) / len(g)
        oracle.append(0.0 if r == 0 else 1.25 * r / (0.25 + r))
    return dict(PC=hit / max(tot_m, 1), RR=1 - tot_c / (len(gold) * n_right),
                PQ=hit / max(tot_c, 1), oracle_macroF05=sum(oracle) / len(oracle))
```

---

## 4. Name normalisation

### 4.1 Legal-suffix lists (seed lists, applied regardless of country)
Put these in code as constants. Don't gate them on the country label: the rules forbid hard-coding the pipeline to {US, India}, and applying all lists everywhere is the safest open-set behaviour.

- **US** [V]: `inc, inc., incorporated, corp, corp., corporation, co, co., company, & co, and company, llc, l.l.c., llp, l.l.p., lp, ltd, limited, pllc`. Source: the US list in cleanco [termdata.py](https://github.com/psolin/cleanco/blob/master/cleanco/termdata.py). Also add `pc, p.c., plc` [E].
- **India** [V]: the MCA name-comparison rule, *Companies (Incorporation) Rules 2014, Rule 8(2)(a)*, says to **disregard** "Private, Pvt, Pvt., (P), OPC Pvt. Ltd., IFSC Limited, IFSC Pvt. Limited, Producer Limited, Limited, Unlimited, Ltd, Ltd., LLP, Limited Liability Partnership, company, and company, & co, & co., co., co, corporation, corp, corpn, corp or group" ([Indian Kanoon](https://indiankanoon.org/doc/198488503/)). Names must end in "Limited" (public) or "Private Limited" (private) under Companies Act 2013 s.4(1)(a) ([text](https://corporatelawreporter.com/companies_act/section-4-of-companies-act-2013-memorandum/)). OPC names carry "(OPC) Private Limited" (secondary source: [advocategandhi.com](https://advocategandhi.com/section-4-memorandum-of-association-the-foundational-charter-of-a-company-under-the-companies-act-2013/)). Also strip the Indian prefix `M/s`, `Messrs` [E], and treat `HUF` / "hindu undivided family" as a legal form (it is in libpostal's [en/company_types.txt](https://github.com/openvenues/libpostal/blob/master/resources/dictionaries/en/company_types.txt)) [V].
- **France** [V]: EI, EIRL, micro-entrepreneur, SARL, EURL, EARL, SELARL, SA, SAS, SASU, SCA, SNC, SCS, SCOP, SCM, SCP, SCI, SCPI, SCCV ([INPI list](https://www.inpi.fr/ressources/formalites-dentreprises/differents-statuts-et-formes-juridiques-de-lentreprise); [service-public](https://entreprendre.service-public.gouv.fr/vosdroits/F23844)). Include the long forms and dotted or spaced variants ("société à responsabilité limitée", "s.a.r.l.", "s a r l"), which libpostal's [fr/company_types.txt](https://github.com/openvenues/libpostal/blob/master/resources/dictionaries/fr/company_types.txt) already enumerates (also GIE, SEM, SICAV, "groupe"). Add the French prefixes `société/sté`, `ets/établissements`, `cie/compagnie`, `& cie` [E].
- **Gaps in cleanco to know about** [V]: its India list is only `ltd., pse, psu, pvt. ltd.`; its France list has `sarl, sas, sasu, eurl, sci, snc, …` but no bare `sa`. So merge cleanco with the lists above rather than calling `cleanco.basename()` alone.

**Implementation [E].** Normalise first: casefold, NFKD accent-fold, `&`→`and`, strip punctuation, collapse dotted acronyms (`s.a.r.l.`→`sarl`). Then remove legal-form tokens **at the end or start of the name only**, and keep the removed form as a categorical feature. "ABC SAS" vs "ABC SARL" may be different entities; make legal-form conflict a feature, not a hard rule.

**Data-driven suffixes (the France safety net) [E].** On the *test* S1∪S2∪S3 names for each country label, compute document frequency for tokens in the last-two and first-one positions. Tokens in the top-N by DF are legal or generic words; down-weight them via IDF. This uses only provided data and finds `sarl/sas/ets` even if the seed list misses one.

### 4.2 DBA and trade names [E]
- Split on `dba`, `d/b/a`, `doing business as` (all in libpostal [en/company_types](https://github.com/openvenues/libpostal/blob/master/resources/dictionaries/en/company_types.txt) [V]), plus `t/a`, `trading as`, `aka`, `a/k/a`, `formerly`, `(…)`, ` - `.
- For France, also split on `enseigne` and `nom commercial`.
- Compute name similarity as the **max over all part pairs** (legal × legal, legal × trade, trade × trade), and add a flag for "matched via trade part".

### 4.3 Indian transliteration and phonetic variants
- MCA Rule 8 treats as the "same" name [V]:
  - different phonetic spellings ("Chemtech = Kemtek")
  - singular/plural and tense changes
  - word order
  - articles
  - www/.com
  - spacing and punctuation
  - **Hindi↔English transliteration or translation** ("National Electricity Corporation" = "Rashtriya Vidyut Nigam")

  Source: [Rule 8](https://indiankanoon.org/doc/198488503/). This list is a ready-made spec for name features. Translation equivalents cannot be caught by string features; multilingual embeddings may help partly [E].
- Real Indian e-commerce data shows 18 spellings of "koramangala" and 61 of "Bannerghatta Road" in about 10k Bangalore addresses ([Babu & Kakkar, SIGIR eCom](https://sigir-ecom.weebly.com/uploads/1/0/2/9/102947274/paper_21.pdf)) [V]. Edit-distance plus phonetic preprocessing is the standard remedy ([Mangalgi et al., arXiv 2007.03020](https://arxiv.org/pdf/2007.03020)) [V].
- **Latin-script folding key [E]**, used as an extra feature and never as a replacement:
  - `aa→a, ee→i, oo→u, ph→f, w→v, ksh→x, z→j`
  - drop `h` after consonants
  - collapse doubled letters
  - `shri|shree|sri→sri`
  - `and|&→and`
- **Devanagari in the data?** Use `indic_transliteration` ([MIT](https://github.com/indic-transliteration/indic_transliteration_py/blob/master/LICENSE.txt)) to romanise, then apply the key. AI4Bharat IndicXlit is also [MIT](https://github.com/AI4Bharat/IndicXlit/blob/master/LICENSE). Check the data for non-Latin script before building either.

### 4.4 Accent stripping and the licence trap [V]
- **`Unidecode` is GPL-2.0-or-later** ([PyPI](https://pypi.org/project/Unidecode/), [LICENSE](https://github.com/avian2/unidecode/blob/master/LICENSE)). **Do not use it**, because the solution must be MIT/Apache-clean.
- `text-unidecode` is Artistic *or* GPL ([LICENSE](https://github.com/kmike/text-unidecode/blob/master/LICENSE)). Avoid it too.
- `anyascii` is **ISC** ([LICENSE](https://github.com/anyascii/anyascii/blob/master/LICENSE)). ISC is permissive, but it is neither MIT nor Apache, so it is a small audit risk.
- **Use stdlib**: `unicodedata.normalize('NFKD', s)`, then drop combining marks ([docs](https://docs.python.org/3/library/unicodedata.html)). Hand-map `œ→oe, æ→ae, ß→ss, ø→o, đ→d`. For French elisions (`l'`, `d'`, `qu'`), strip them or split on the apostrophe; see libpostal's [fr/elisions.txt](https://github.com/openvenues/libpostal/blob/master/resources/dictionaries/fr/elisions.txt).

### 4.5 Abbreviation-aware token match (language-agnostic) [E]
Build one feature that treats token *a* as matching token *b* when either:
- they are equal, or
- *a* is a prefix of *b* (`av`→`avenue`, `corp`→`corporation`, `pvt`→`private`), or
- *a* is an in-order subsequence of *b* that shares its first letter (`bd`/`blvd`→`boulevard`, `ltd`→`limited`, `st`→`street`, `rd`→`road`, `ste`→`societe`).

This covers most US, Indian and French abbreviations **without per-country lists**, so it transfers to France. Also mine a synonym table from the train ground truth: align tokens between matched pairs and count (tok_a, tok_b) co-occurrences. That uses only provided data.

---

## 5. Address normalisation with no external API

### 5.1 libpostal: capability, size, and the "external data" risk [V]
- **Code**: [MIT](https://github.com/openvenues/libpostal/blob/master/LICENSE). Python binding `postal` (MIT). Homebrew has a **bottled** formula, libpostal 1.1.4 ([formula](https://formulae.brew.sh/formula/libpostal)). A source build on Apple Silicon needs `--disable-sse2` ([README](https://github.com/openvenues/libpostal)).
- **Size**: the data files are downloaded from S3 at build time. The default model is about **1.8 GB** and the Senzing model about **2.2 GB**. The README's REST server note says it "need[s] ~4Gb memory". It runs **offline** once the data is on disk ([README](https://github.com/openvenues/libpostal)).
- **Provenance**: the parser is a CRF "trained on over 1 billion addresses" from **OpenStreetMap (ODbL)** and **OpenAddresses (mostly CC-BY)**. It also uses the OpenCage templates, an OSM-trained language classifier, and toponym and place data ([README](https://github.com/openvenues/libpostal)).
- **Risk assessment [E]:** the rules ban "external data augmentation from internet sources" and geocoding, and the final model must be MIT/Apache ([problem statement](../source/problem_statement.txt)). libpostal's *weights* are derived from ODbL/CC-BY geodata and embed gazetteer knowledge of real place names. A strict auditor could call that external geographic knowledge, and it is not an "MIT/Apache model" in the sense the rules mean.
- **Safe interpretation:**
  1. Don't use the libpostal parser or data in the final pipeline.
  2. You may copy abbreviation lists from `resources/dictionaries/*` (hand-curated text in an MIT repo, such as [fr/street_types.txt](https://github.com/openvenues/libpostal/blob/master/resources/dictionaries/fr/street_types.txt), [hi/street_types.txt](https://github.com/openvenues/libpostal/blob/master/resources/dictionaries/hi/street_types.txt) and [en/near.txt](https://github.com/openvenues/libpostal/tree/master/resources/dictionaries/en)) into your own code, with the MIT notice.
  3. **Don't** copy `toponyms.txt` or `place_names.txt`, which are gazetteers.
  4. Say all of this explicitly in the methodology doc.
- **usaddress** ([MIT](https://github.com/datamade/usaddress/blob/main/LICENSE)): a CRF trained on labelled US addresses, US-only. Low value for India and France. Skip it.

### 5.2 Patterns to encode (regex and features, not a parser)
**Common to all countries [E]:**
- extract all digit runs, then derive:
  - `postcode_like`: 5 digits (US ZIP or FR code postal), 5+4 (ZIP+4) or 6 digits (IN PIN)
  - `house_no`: the first number, allowing `12bis`, `12-B`, `12/3`, `#12`
  - `all_numbers`: the set
- For each feature, three states: **equal / conflict / missing**. Missing ≠ mismatch: LightGBM routes NaN to its own branch by default ([docs, "Missing Value Handle"](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html)) [V].

**India:**
- PIN is 6 digits. The first digit is the region, the first two the postal circle, the first three the sorting district, the last three the delivery post office ([India Post](https://uat.indiapost.gov.in/MBE/Pages/Content/Pincode.aspx); [Wikipedia](https://en.wikipedia.org/wiki/Postal_Index_Number)) [V]. **Partial-PIN agreement** (first 3 digits) is a useful graded feature [E].
- Real addresses are landmark-heavy, e.g. "OPPOSITE SHANI MANDIR", "Near ITER College", "behind LVK Garments, Opp SRS Water tank", "1st stage, 4th cross" ([SIGIR eCom paper](https://sigir-ecom.weebly.com/uploads/1/0/2/9/102947274/paper_21.pdf)) [V].
- Landmark triggers include near, opposite, behind, beside. Locality suffixes include Nagar, Colony, Society, Sector ([indiaddr README](https://pypi.org/project/indiaddr/), for the patterns only) [V]. libpostal's Hindi street types include `bazaar|bazar; marg; flyover` [V].
- Add these tokens [E]:
  - landmark triggers: `nr, opp, bh, behind, beside, next to, adj, above, below`
  - unit prefixes: `plot no, h no, door no, flat, shop no, gala, khasra, sy no, survey no`
  - locality words: `stage, phase, block, sector, cross, main, layout, chowk, gali, mohalla`

**France** [V]:
- AFNOR NF Z10-011 layout: at most 6 lines of ≤38 characters. Line 4 is the **number, then the street type, then the street name**. Line 6 is the **postcode and town**, or a **CEDEX** code with its label. Example: "35 IMPASSE DES GABARRES / BP 18 ARVEYRES / 33506 LIBOURNE CEDEX" ([La Poste spec](https://lastation.laposte.fr/sites/p8_u1/files/2023-03/SP8855-Volume%202_V1.11%20-%20Adressage%20des%20plis.pdf)).
- The postcode is 5 digits, and the first two are the département. A postcode can cover several communes ([CNIG address standard](https://cnig.gouv.fr/IMG/pdf/prj_standard_adresse_pour_revue.pdf)).
- Street-type abbreviations are enumerated in libpostal's [fr/street_types.txt](https://github.com/openvenues/libpostal/blob/master/resources/dictionaries/fr/street_types.txt): `avenue|av|ave`, `boulevard|bd|blvd`, `allée|all`, `chemin|ch|che`, `chaussée|chs`, …
- Tokens to handle [E]:
  - `bis, ter, quater` (number suffixes)
  - `BP` (boîte postale) and `CS`, which are PO-box-like: drop them when comparing house numbers
  - `CEDEX nn`: the CEDEX code replaces the postcode for business mail, so a CEDEX code vs a normal postcode is *not* a conflict
  - `ZI, ZA, ZAC` (industrial or activity zones)
  - `rue, r, place/pl, quai, impasse/imp, route/rte, faubourg/fg`
- **Ambiguity traps [E]:**
  - `st` = Street (US) vs Saint (FR)
  - `ste` = Suite (US) vs Sainte (FR)
  - `dr` = Drive vs Docteur (as in "rue du Dr …")

  Never expand these one-sidedly. Apply the same canonicaliser to both sides, or rely on the abbreviation-aware match (§4.5).

**US** [V]: USPS Publication 28 lists the street suffixes and secondary unit designators (Ste, Apt, Fl, #) ([Pub 28 C2](https://pe.usps.com/text/pub28/28apc_002.htm)). Handle ZIP vs ZIP+4 and two-letter state codes [E].

### 5.3 Landmark handling (from the video example) [E]
The official video's S3 match has *only* "Nr City Hall, San Jose" as its address ([READ-NOTES](../READ-NOTES.md)).
- Split each address into a `core` part and a `landmark` part (trigger word plus up to about 4 following tokens).
- Features:
  - `landmark_only` flag (per side)
  - core-core similarity
  - landmark-vs-full containment
  - city-token overlap (the last non-numeric tokens)
- The GBDT then learns "name strong + city agrees + one side landmark-only → match" without penalising the missing street number.

---

## 5.5 Licence-compliant models (MIT/Apache-2.0, ≤8B)

Licence and parameter counts come from the HF model API [V-API] (`https://huggingface.co/api/models/<id>`, 25 Sep 2026). "Params" is the safetensors total where published; otherwise it is estimated from checkpoint size (marked ~).

### Sentence embedders (blocking and features)
| HF id | Licence | Params | Multilingual (FR) | Note |
|---|---|---|---|---|
| `intfloat/multilingual-e5-small` | MIT | 118M | yes | **Default dense blocker.** Fast on a Mac. |
| `intfloat/multilingual-e5-base` | MIT | 278M | yes | Better recall, still cheap. |
| `intfloat/multilingual-e5-large` / `-large-instruct` | MIT | 560M | yes | |
| `intfloat/e5-small-v2`, `e5-base-v2`, `e5-large-v2` | MIT | 33M / 109M / 335M | English | |
| `BAAI/bge-small-en-v1.5`, `bge-base-en-v1.5`, `bge-large-en-v1.5` | MIT | 33M / 109M / 335M | English | |
| `BAAI/bge-m3` | MIT | ~568M (2.27 GB fp32) | yes | Dense, sparse and multi-vector. |
| `thenlper/gte-small`, `gte-base`, `gte-large` | **MIT** | 33M / 109M / 335M | English | |
| `Alibaba-NLP/gte-base-en-v1.5`, `gte-large-en-v1.5` | **Apache-2.0** | 137M / 434M | English | Different licence from thenlper gte. Custom code [U]. |
| `Alibaba-NLP/gte-multilingual-base` | Apache-2.0 | 305M | yes | Custom code (`trust_remote_code`) [U]. |
| `Alibaba-NLP/gte-modernbert-base` | Apache-2.0 | 149M | English | |
| `Alibaba-NLP/gte-Qwen2-1.5B-instruct` / `-7B-instruct` | Apache-2.0 | 1.78B / 7.61B | yes | Too large for blocking. |
| `sentence-transformers/all-MiniLM-L6-v2` / `-L12-v2` | Apache-2.0 | 23M / 33M | English | |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Apache-2.0 | 118M | yes | |
| `sentence-transformers/LaBSE` | Apache-2.0 | 471M | yes | Strong for cross-script names. |
| `Snowflake/snowflake-arctic-embed-m-v2.0` / `-l-v2.0` | Apache-2.0 | 305M / 568M | yes | |
| `nomic-ai/nomic-embed-text-v1.5` | Apache-2.0 | 137M | English | |
| `mixedbread-ai/mxbai-embed-large-v1` | Apache-2.0 | 335M | English | |
| `ibm-granite/granite-embedding-107m-multilingual` / `-278m-` | Apache-2.0 | 107M / 278M | yes | |
| `Qwen/Qwen3-Embedding-0.6B` / `-4B` / `-8B` | Apache-2.0 | 0.60B / 4.02B / **7.57B** | yes (100+) | The "8B" is actually 7.57B. |
| ❌ `jinaai/jina-embeddings-v3` | **CC-BY-NC-4.0** | 572M | | Not allowed. |
| ❌ `BAAI/bge-multilingual-gemma2` | **Gemma** | 9.24B | | Not allowed. |

### Cross-encoders and rerankers (fine-tune as the pair classifier)
| HF id | Licence | Params | FR | Note |
|---|---|---|---|---|
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Apache-2.0 | 118M | yes | **Default: fine-tunes fast on MPS or a T4.** |
| `BAAI/bge-reranker-v2-m3` | Apache-2.0 | 568M | yes | Stronger, slower. |
| `BAAI/bge-reranker-base` / `-large` | **MIT** | 278M / 560M | EN+ZH [E] | Older bge rerankers are MIT; v2 is Apache. |
| `Alibaba-NLP/gte-multilingual-reranker-base` | Apache-2.0 | 306M | yes | Custom code [U]. |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | Apache-2.0 | 23M | English only | Fine as a US/IN-only baseline. Weak for FR [E]. |
| `mixedbread-ai/mxbai-rerank-base-v1` / `-v2` | Apache-2.0 | 184M / 494M | | |
| `Qwen/Qwen3-Reranker-0.6B` / `-4B` | Apache-2.0 | 0.60B / 4.02B | yes | LLM-style reranker. |
| `BAAI/bge-reranker-v2-gemma` | Apache-2.0 (card) | 2.51B | | Card says Apache, but it is built on Gemma; audit risk [E]. Avoid. |
| ❌ `jinaai/jina-reranker-v2-base-multilingual` | **CC-BY-NC-4.0** | 278M | | Not allowed. |

### Encoders to fine-tune Ditto-style
- `FacebookAI/xlm-roberta-base` / `-large`: **MIT**, 279M / 561M, multilingual.
- `microsoft/mdeberta-v3-base`: MIT, ~0.3B (1.33 GB checkpoint), multilingual.
- `google-bert/bert-base-multilingual-cased`: Apache-2.0, 179M.
- `distilbert/distilbert-base-multilingual-cased`: Apache-2.0, 135M.
- `almanach/camembert-base`: MIT, 111M, French only.
- `answerdotai/ModernBERT-base`: Apache-2.0, 150M, English.
- `jhu-clsp/mmBERT-base`: MIT, ~0.3B, multilingual.
- `EuroBERT/EuroBERT-210m`: Apache-2.0, 310M.
- **Character-level:** `google/canine-s` (Apache-2.0, 132M) and `google/byt5-small` (Apache-2.0, ~0.3B).

### Small LLMs (optional judge)
| HF id | Licence | Params | ≤8B? | Verdict |
|---|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct`, `-1.5B-Instruct` | Apache-2.0 | 0.49B / 1.54B | ✓ | OK |
| ❌ **`Qwen/Qwen2.5-3B-Instruct`** | **`qwen-research`: non-commercial, "FOR NON-COMMERCIAL PURPOSES ONLY"** ([LICENSE](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE)) | 3.09B | | **Not allowed.** |
| `Qwen/Qwen2.5-7B-Instruct` | Apache-2.0 | **7.62B** | ✓ | **Best allowed 7B-class judge.** Multilingual incl. French per its card. |
| ❌ `Qwen/Qwen2.5-14B-Instruct` | Apache-2.0 | 14.8B | ✗ | Too big. |
| `Qwen/Qwen3-0.6B`, `-1.7B`, `-4B`, `-4B-Instruct-2507` | Apache-2.0 | 0.75B / 2.03B / 4.02B / 4.02B | ✓ | Qwen3-4B is the sweet spot on a Mac. |
| ❌ **`Qwen/Qwen3-8B`** | Apache-2.0 | **8.19B** (card: "8.2B") | **✗** | **Over 8B despite the name.** |
| `mistralai/Mistral-7B-Instruct-v0.3` | Apache-2.0 | 7.25B | ✓ | OK |
| ❌ `mistralai/Ministral-8B-Instruct-2410` | **`mrl`** (Mistral Research Licence) | 8.02B | ✗ | Not allowed. |
| `mistralai/Ministral-3-3B-Instruct-2512` | Apache-2.0 | 3.85B (incl. 0.4B vision encoder) | ✓ | OK |
| ❌ `mistralai/Ministral-3-8B-Instruct-2512` | Apache-2.0 | **8.92B** incl. vision | ✗ | Over 8B. |
| `microsoft/Phi-3-mini-4k-instruct`, `Phi-3.5-mini-instruct`, `Phi-4-mini-instruct` | **MIT** | 3.82B / 3.82B / 3.84B | ✓ | OK. The Phi-4-mini card says "broad multilingual" use. French quality [U]. |
| ❌ `microsoft/phi-4` | MIT | 14.7B | ✗ | Too big. |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct`, `SmolLM3-3B` | Apache-2.0 | 1.71B / 3.08B | ✓ | OK |
| `ibm-granite/granite-3.3-2b-instruct` | Apache-2.0 | 2.53B | ✓ | OK |
| ❌ `ibm-granite/granite-3.3-8b-instruct` | Apache-2.0 | **8.17B** | ✗ | Over 8B. |
| ❌ `google/gemma-2-2b-it`, `gemma-3-4b-it` | **Gemma terms** | | | Not allowed. |
| ❌ `meta-llama/Llama-3.2-3B-Instruct`, `Llama-3.1-8B-Instruct` | **llama3.2 / llama3.1** | | | Not allowed. |
| ❌ `NECOUDBFM/Jellyfish-7B` / `-8B` / `-13B` (ER-tuned LLMs) | **CC-BY-NC-4.0** | | | Not allowed, even though they are built for EM. |
| `openai-community/gpt2` (AnyMatch base) | MIT | 137M | ✓ | OK |

**Parameter budget [E].** The rule says "Final model … up to 8 Billion parameters" ([problem statement](../source/problem_statement.txt)). It does not say whether several models are summed, so assume they are.
- ✅ Qwen2.5-7B (7.62B) + e5-small (0.12B) + mMiniLM cross-encoder (0.12B) = 7.86B.
- ✅ Qwen3-4B (4.02B) + bge-m3 (0.57B) + bge-reranker-v2-m3 (0.57B) = 5.2B.
- ❌ Qwen2.5-7B + bge-m3 = 8.19B.

**Mac practicality [E].** A 7B model at 4-bit via `mlx-lm` fits in 16 GB. Prefill-bound yes/no scoring runs at roughly a few pairs per second, so keep the LLM to at most about 10–30k uncertain pairs, or use Qwen3-4B. LoRA on MLX is possible, but beware the cross-domain warning in [arXiv 2409.08185](https://arxiv.org/abs/2409.08185). The Unstop listing mentions $200 of credits per participant (see `unstop-competition-1743604.json`), which could rent a GPU for cross-encoder fine-tuning; check the terms. **Size limit:** the code ZIP cap is 50 MB (`01-challenge-intel.md`), so ship a script that downloads the base weights from HF and retrains or fine-tunes. Don't ship the weights.

---

## 6. Optimising macro F0.5

### 6.1 Theory [V]
- **Two ways to optimise F** ([Ye, Chai, Lee & Chieu, ICML'12](https://arxiv.org/abs/1206.4625); "Nan et al." is Nan Ye):
  - **EUM**: tune a threshold directly on validation. It is more robust to model misspecification.
  - **Decision-theoretic**: calibrated probabilities, then pick the set with the highest *expected* F. It is better **for rare classes and for domain adaptation**.

  France is a domain-adaptation case, so run both and compare on leave-one-country-out.
- **Jansche (ACL'07)**: the maximum-expected-utility framework for F under label independence ([P07-1093](https://aclanthology.org/P07-1093/)). Its exact version is O(m⁴) ([GFM poster](https://www.weiweicheng.com/research/slidesposters/cheng-nips11poster.pdf)).
- **Dembczyński et al. (NIPS'11), GFM**: exact expected-F maximiser from m²+1 parameters in O(m³), with **no independence assumption**, and the convention F(∅,∅)=1 (0/0 = 1). That convention is exactly this competition's singleton rule. Under independence, the optimum is always the top-k by marginal probability, or the empty set ([paper](https://proceedings.neurips.cc/paper_files/paper/2011/file/71ad16ad2c4d81f348082ff6c4b20768-Paper.pdf), [poster](https://www.weiweicheng.com/research/slidesposters/cheng-nips11poster.pdf)).
- **Lipton et al.**: with calibrated scores, the F1-optimal threshold is half the optimal F1 ([arXiv 1402.1892](https://arxiv.org/abs/1402.1892)). For F0.5 the same argument gives F*/1.25 [D].

### 6.2 Per-entity set selection (drop-in) [D]
Under independence, E[F_β(H)] for H = the top-k candidates needs only the Poisson-binomial distributions of true positives inside H and outside H. With β = 0.5, (1+β²) = 1.25 and β² = 0.25.

```python
import numpy as np

def pb_pmf(ps):
    """Poisson-binomial pmf: P(sum of independent Bernoulli(ps) == j)."""
    d = np.zeros(len(ps) + 1); d[0] = 1.0
    for q in ps:
        d[1:] = d[1:] * (1 - q) + d[:-1] * q   # RHS is evaluated before assignment
        d[0] *= (1 - q)
    return d

def best_set_fbeta(p, beta=0.5):
    """p: calibrated match probs of one S1's candidates. Returns (indices to predict, expected F)."""
    p = np.asarray(p, float); order = np.argsort(-p); ps = p[order]; m = len(ps); b2 = beta * beta
    best_val, best_k = float(np.prod(1 - ps)), 0            # predict empty: F=1 iff truth empty
    for k in range(1, m + 1):
        tp, fn = pb_pmf(ps[:k]), pb_pmf(ps[k:])
        t = np.arange(k + 1)[:, None]; f = np.arange(m - k + 1)[None, :]
        val = float(((1 + b2) * t / (b2 * (t + f) + k) * tp[:, None] * fn[None, :]).sum())
        if val > best_val: best_val, best_k = val, k
    return order[:best_k], best_val
```

- **Checked:** the function matched brute-force expected-F0.5 enumeration over all 2^m subsets on 200 random cases with m ≤ 5 (25 Sep 2026). The exact thresholds from §1 are q* = 8/11 = 0.727 and r* = 22/29 = 0.759.
- **Calibrate first.** Use out-of-fold LightGBM scores with isotonic regression or Platt scaling ([sklearn calibration](https://scikit-learn.org/stable/modules/calibration.html)).
- The independence assumption is violated: duplicates in S2/S3 of the same business are positively correlated. So also try the **EUM rule** and keep whichever wins on validation. The EUM rule has three knobs, grid-searched on out-of-fold data:
  - t1: the minimum probability for the top-1 candidate
  - t2: the minimum for each extra link
  - Δ: the maximum gap from the top-1 score

  §1 predicts t1 ≈ 0.5 and t2 ≈ 0.73–0.76 before calibration error [D].
- **Entity-level model:** a second LightGBM predicts P(S1 has ≥1 match) from S1-level aggregates: max, second-max and count of pair probabilities above thresholds, name IDF rarity, and the candidate count. Blend it with ∏(1−p_i) when deciding "empty vs top-1".

### 6.3 Exclusivity, transitivity and assignment
- **Constraint:** S1 is deduplicated, so each S2/S3 record matches **at most one** S1 [D, from the problem statement].
- **This is one-to-many, not one-to-one.** The exact optimum under this constraint is a per-record argmax: for each S2/S3 record, keep only its best S1, if that S1 passes the threshold. **The Hungarian algorithm is the wrong tool** unless the train data shows that each S1 has at most one match *per source*.
- **Check on Day 1:** the distribution of the number of S2 matches per S1 and the number of S3 matches per S1. If both are ≤1, run a one-to-one assignment per source (S1↔S2 and S1↔S3 separately).
- **Algorithms for that case:** Unique Mapping Clustering (a greedy sort over edges that skips already-matched nodes) and the Hungarian or best-assignment heuristics are all compared in [Papadakis et al., VLDB J. 2023](https://link.springer.com/article/10.1007/s00778-023-00791-3) [V]. UMC is the simple strong default [E].
- **Competition features [E]**, computed for every pair (x, r):
  - rank of x among r's S1 candidates, and rank of r among x's candidates
  - `p(x,r) − max_{x'≠x} p(x',r)` (reverse margin)
  - number of S1 candidates with name similarity >0.9 (a chain-store signal)
  - number of S1 candidates sharing r's exact address (a multi-tenant signal: "Acme Bakery, same address, different business" from the video)

  These carry the collective-ER benefit of HierGAT+ ([SIGMOD'22](https://doi.org/10.1145/3514221.3517872)) and of ComEM's "select" strategy ([arXiv 2405.16884](https://arxiv.org/abs/2405.16884)) into a GBDT.
- **Transitivity across S2 and S3 [E]:** if S2-a and S3-b are near-duplicates and S2-a is confidently linked to S1-x, raise p(x, S3-b). Implement it as a stage-2 feature: `max_{r'} min(p(x,r'), sim(r,r'))` over the other source. ZeroER also exploits transitivity ([arXiv 1908.06049](https://arxiv.org/abs/1908.06049)). This is a Day-3 item.

---

## 7. Domain shift to an unseen France

**Evidence [V]:**
- Fine-tuned PLM matchers learn spurious correlations and degrade under domain and structural shift. Class imbalance is a key cause, and **augmentation alone doesn't fix it** ([RobEM, CIKM'22](https://ehsk.github.io/assets/pdf/CIKM22_RobustEM.pdf)).
- LLMs are more robust to unseen entities ([arXiv 2310.11244](https://arxiv.org/abs/2310.11244)). LLM fine-tuning hurts cross-domain transfer ([arXiv 2409.08185](https://arxiv.org/abs/2409.08185)).
- Domain adaptation (feature alignment via MMD or adversarial training) helps when the source model does poorly on the target, and a "close" source domain helps more ([DADER](https://github.com/ruc-datalab/DADER), [PVLDB demo](https://www.vldb.org/pvldb/vol15/p3666-fan.pdf)).
- Data augmentation for EM ([Rotom, SIGMOD'21](https://www.miaozhengjie.com/assets/pdf/rotom-sigmod21.pdf)) gives up to +6% F1 in low-resource settings.

**Playbook [E]:**
1. **Validate the way you'll be tested.** Use **leave-one-country-out** (train on US, validate on India, and the reverse), plus a group-k-fold by S1. The LOCO macro-F0.5 is your proxy for France. Choose features, models and thresholds by the *worse* of the two LOCO scores.
2. **Language-agnostic features only.** Use char n-gram TF-IDF, edit-distance family metrics (rapidfuzz `ratio`, `token_set_ratio`, `partial_ratio`, Jaro-Winkler), number agreement, IDF-weighted token overlap, the abbreviation-aware match (§4.5), and embedding cosine from a *multilingual* model. Avoid word lists as features unless they are multi-country (§4.1).
3. **Per-country statistics, computed on the test data itself.** Compute IDF, token document frequency and "common suffix" sets per country label from the unlabelled test sources. This is provided data, not external data.
4. **Relative features over absolute ones.** A candidate's score minus the best score in its list, its rank, and a z-score within the S1's candidate list. These are invariant to a global shift in similarity levels when French strings are longer or more accented.
5. **Synthetic "French-style" noise on train pairs.** Apply to US/India matched pairs, with probability p each:
   - drop accents / add apostrophes
   - swap `Street↔St`-style full/abbrev forms using *generic* rules (truncation, vowel removal)
   - move the house number before or after the street
   - append or drop a postcode-like token, or append "CEDEX nn"
   - add or strip a legal form token drawn from the multi-country list
   - reorder name tokens
   - insert a landmark clause

   This teaches invariances. Per RobEM it will not by itself give robustness, so pair it with LOCO validation.
6. **Char-level vs word-level.** Char n-grams are the most transferable. Subword multilingual encoders (XLM-R, mDeBERTa, multilingual-e5) handle accents and French morphology. Word-level TF-IDF breaks on elisions and accents without normalisation. Pure char models (CANINE, ByT5) are an option but slower to fine-tune; keep them as a stretch goal.
7. **Guard rail for unseen countries (open-set rule, no France hard-coding).** If a record's country label never appeared in train, apply a slightly stricter t2 (extra-link threshold). Missing France recall is cheap; false merges on French entities are expensive (§1). Pick the margin from LOCO: how much did thresholds tuned on one country over-link on the other?
8. **Pseudo-labelling on test France.** Take high-confidence French pairs and re-fit. This is transductive use of the provided test data and is probably allowed, but it is a grey zone ("use only the provided data"). If used, document it, and never touch external data.

---

## 8. Three-day plan

The window runs from 25 Sep 00:00 to 27 Sep 23:59 IST, with 5 LB submissions per day ([guidelines](../source/guidelines.txt)). Log every submission: timestamp, git hash, LB score and offline LOCO score.

### Day 1 (25 Sep): baseline end to end
1. **EDA (1h):**
   - sizes per source and country
   - % singletons, and the matches-per-S1 histogram *per source* (this decides §6.3)
   - label noise
   - non-Latin script
   - country label consistency
2. **Normaliser v1 (2h):** §4.1–4.5 and §5.2.
3. **Blocking v1 (2h):** B1 + B2 + B3 with `sparse_dot_topn`. Report PC, RR and the oracle macro-F0.5 per blocker and for the union, by country (§3.3). Iterate until the oracle ceiling is ≥0.97 on validation at the smallest K [E].
4. **Features v1 (2h):** rapidfuzz, jellyfish (Jaro-Winkler, metaphone), TF-IDF cosines, number tri-states, landmark flags, legal-form agreement, rank and margin features.
5. **LightGBM matcher (1h):** group-k-fold by S1, plus LOCO.
6. **Decision v1 (1h):** the EUM grid over (t1, t2, Δ) plus exclusivity.
7. **Validate and submit:**
   - Run `utils/validate_submission.py`, then submit #1 (baseline).
   - Optional: submit the all-empty probe to learn the public singleton rate.
   - Submit #2 and #3 as threshold variants.

### Day 2 (26 Sep): recall and matcher upgrades
1. **Dense blocking (B4)** with multilingual-e5-small or base plus faiss. Measure the union gain and the size cost (§3.1 suggests single-digit recall points).
2. **Stage-1 pruner** (a cheap LightGBM) to fix K per S1. Freeze what goes into `candidate_pairs.tsv`.
3. **Cross-encoder:** fine-tune `mmarco-mMiniLMv2-L12-H384-v1` (or `bge-reranker-v2-m3` on a GPU) on Ditto-serialised pairs, with hard negatives from blocking, for 1–2 epochs. Feed its out-of-fold probability into LightGBM.
4. **Calibration + GFM-style set selection (§6.2)** vs EUM, compared on LOCO.
5. **Synthetic French-noise augmentation (§7.5)**, only if LOCO improves.
6. Submit #6–#10, one change at a time.

### Day 3 (27 Sep): robustness, ensembling, freeze by 20:00 IST
1. **Ensemble:** 3–5 LightGBM seeds, plus CatBoost (Apache), plus the cross-encoder, via rank-average or a stacker, then recalibrate.
2. **Entity-level no-match model;** transitivity feature (§6.3).
3. **Optional LLM judge** (Qwen2.5-7B-Instruct or Qwen3-4B via mlx-lm) on the uncertain band. Use a ComEM-style "select" prompt: S1 plus up to 5 candidates, "which refer to the same business? answer ids or NONE". Keep the change only if LOCO improves, and check the parameter budget (§5.5).
4. **Final thresholds** from LOCO, not from the public LB (it is a subset; don't overfit to it).
5. **Package:**
   - regenerate both TSVs from `src/` in a clean env
   - pin `requirements.txt`
   - ZIP ≤50 MB with no weights
   - Documentation_template.md: blocking PC/RR/oracle table, feature list, model and licence table, LOCO results, a note on the "no external data" interpretation (§5.1)

### Libraries (all checked on GitHub or PyPI, 25 Sep 2026)
| Lib | Licence | Mac arm64 | Use |
|---|---|---|---|
| rapidfuzz | [MIT](https://github.com/rapidfuzz/RapidFuzz/blob/main/LICENSE) | wheels | string similarities |
| jellyfish | [MIT](https://github.com/jamesturk/jellyfish/blob/main/LICENSE) | wheels | Jaro-Winkler, metaphone, NYSIIS |
| lightgbm | [MIT](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE) | wheel | matcher |
| xgboost / catboost | [Apache-2.0](https://github.com/dmlc/xgboost/blob/master/LICENSE) / [Apache-2.0](https://github.com/catboost/catboost/blob/master/LICENSE) | wheels | ensemble |
| sparse_dot_topn | [Apache-2.0](https://github.com/ing-bank/sparse_dot_topn/blob/master/LICENSE) | wheels | TF-IDF top-k |
| string_grouper | [MIT](https://github.com/Bergvca/string_grouper/blob/master/LICENSE) | pure | alternative. "663 000 names in less than 18 seconds" ([README](https://github.com/Bergvca/string_grouper)) |
| faiss-cpu | [MIT](https://github.com/facebookresearch/faiss/blob/main/LICENSE) | wheel | dense top-k |
| hnswlib | [Apache-2.0](https://github.com/nmslib/hnswlib/blob/master/LICENSE) | **no wheel** | skip on Mac |
| datasketch | [MIT](https://github.com/ekzhu/datasketch/blob/master/LICENSE) | pure | MinHash LSH (optional) |
| sentence-transformers | [Apache-2.0](https://github.com/huggingface/sentence-transformers/blob/main/LICENSE) | pure | embedders, cross-encoder fine-tune |
| transformers / peft | [Apache-2.0](https://github.com/huggingface/transformers/blob/main/LICENSE) / [Apache-2.0](https://github.com/huggingface/peft/blob/main/LICENSE) | pure | |
| mlx-lm | [MIT](https://github.com/ml-explore/mlx-lm/blob/main/LICENSE) | pure | LLM on Mac |
| cleanco | [MIT](https://github.com/psolin/cleanco/blob/master/LICENSE.txt) | pure | seed legal-form list |
| indic_transliteration | [MIT](https://github.com/indic-transliteration/indic_transliteration_py/blob/master/LICENSE.txt) | pure | Devanagari→Latin |
| polars / duckdb | [MIT](https://github.com/pola-rs/polars/blob/main/LICENSE) / [MIT](https://github.com/duckdb/duckdb/blob/main/LICENSE) | | fast joins |
| scikit-learn | [BSD-3](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | | TF-IDF, isotonic. Permissive; the MIT/Apache rule is about the *model* [E] |
| ⚠️ Unidecode | **GPL-2.0+** | | **avoid** |
| ⚠️ libpostal / postal | MIT code, ODbL/CC-BY-derived data | brew bottle | **avoid in final pipeline** (§5.1) |

---

## 9. Top risks

| # | Risk | Mitigation |
|---|---|---|
| 1 | **France shift leads to over-linking.** Unseen legal forms, `St`=Saint, accents, CEDEX, number-first streets. Thresholds tuned on US/IN are too loose. | LOCO validation, language-agnostic and relative features, per-country IDF from test, a stricter extra-link threshold for unseen country labels (§7). |
| 2 | **Licence or parameter violations.** Qwen2.5-3B is non-commercial. Qwen3-8B (8.19B), granite-3.3-8b (8.17B) and Ministral-3-8B (8.92B) exceed 8B. Jina and Jellyfish are CC-BY-NC; Gemma and Llama are not MIT/Apache. The summed-parameter interpretation is unresolved. Unidecode is GPL. | Use only the ✓ rows in §5.5, keep the sum ≤8B, and put the licence table in the methodology doc. |
| 3 | **"External data" audit.** libpostal's OSM-trained parser and gazetteers, and ISO/GLEIF legal-form lists fetched at runtime, could be flagged. | Hand-written rules in the repo; pretrained MIT/Apache models only; no runtime downloads other than HF weights; document the interpretation (§5.1). |
| 4 | **Candidate-file audit.** `candidate_pairs.tsv` must be the exact set scored, and matches must be a subset. Too small a K caps recall; too large a K hurts RR. Code ZIP ≤50 MB. | Freeze the pruned set as the single source for both files; report PC, RR and the oracle ceiling; validator before every submission. |
| 5 | **Metric traps.** Singletons (any link = 0). Chain stores (same name, different address). Multi-tenant buildings (same address, different name). Probability miscalibration. Overfitting a public-LB subset with 25 submissions total. | Entity-level no-match model, competition features, isotonic calibration, GFM vs EUM chosen offline, one change per submission. |

---

## 10. Source index
- Problem: `../source/problem_statement.txt`, `../source/guidelines.txt`, `../READ-NOTES.md`, `./01-challenge-intel.md`
- ER systems:
  - [Ditto](https://arxiv.org/abs/2004.00584)
  - [HierGAT](https://doi.org/10.1145/3514221.3517872)
  - [Sudowoodo](https://arxiv.org/abs/2207.04122)
  - [AnyMatch](https://arxiv.org/abs/2409.04073)
  - [Unicorn](https://sigmodrecord.org/publications/sigmodRecord/2403/pdfs/12_unicorn-fan.pdf)
  - [Magellan](http://www.vldb.org/pvldb/vol9/p1197-pkonda.pdf)
  - [ZeroER](https://arxiv.org/abs/1908.06049)
  - [Splink](https://github.com/moj-analytical-services/splink)
  - [dedupe](https://github.com/dedupeio/dedupe)
  - [recordlinkage](https://github.com/J535D165/recordlinkage)
  - [pyJedAI](https://github.com/AI-team-UoA/pyJedAI)
- LLM EM:
  - [Peeters et al. EDBT'25](https://arxiv.org/abs/2310.11244)
  - [ChatGPT for EM](https://arxiv.org/abs/2305.03423)
  - [Fine-tuning LLMs for EM](https://arxiv.org/abs/2409.08185)
  - [ComEM](https://arxiv.org/abs/2405.16884)
  - [BatchER](https://arxiv.org/abs/2312.03987)
- Blocking:
  - [Sparkly](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf)
  - [DeepBlocker](https://www.vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf)
  - [Zeakis et al.](https://www.vldb.org/pvldb/vol16/p2225-skoutas.pdf)
  - [Christen TKDE](https://users.cecs.anu.edu.au/~christen/publications/christen2011indexing.pdf)
  - [Papadakis survey](https://arxiv.org/abs/1905.06167)
- Company matching:
  - [Census MAMBA](https://ngoldschlag.github.io/papers/squeeze.pdf)
  - [IBM company RL](https://arxiv.org/abs/1907.08667)
  - [Kruse et al.](https://www.tib-op.org/ojs/index.php/bis/article/view/44)
  - [ELF classification](https://doi.org/10.48550/arxiv.2310.12766)
- F-measure:
  - [Ye et al.](https://arxiv.org/abs/1206.4625)
  - [Jansche](https://aclanthology.org/P07-1093/)
  - [Dembczyński GFM](https://proceedings.neurips.cc/paper_files/paper/2011/file/71ad16ad2c4d81f348082ff6c4b20768-Paper.pdf)
  - [Lipton et al.](https://arxiv.org/abs/1402.1892)
- Clustering: [Papadakis 1-1 matching](https://link.springer.com/article/10.1007/s00778-023-00791-3)
- Shift:
  - [RobEM](https://ehsk.github.io/assets/pdf/CIKM22_RobustEM.pdf)
  - [DADER](https://www.vldb.org/pvldb/vol15/p3666-fan.pdf)
  - [Rotom](https://www.miaozhengjie.com/assets/pdf/rotom-sigmod21.pdf)
- Addresses and names:
  - [libpostal](https://github.com/openvenues/libpostal)
  - [cleanco termdata](https://github.com/psolin/cleanco/blob/master/cleanco/termdata.py)
  - [MCA Rule 8](https://indiankanoon.org/doc/198488503/)
  - [INPI forms](https://www.inpi.fr/ressources/formalites-dentreprises/differents-statuts-et-formes-juridiques-de-lentreprise)
  - [La Poste spec](https://lastation.laposte.fr/sites/p8_u1/files/2023-03/SP8855-Volume%202_V1.11%20-%20Adressage%20des%20plis.pdf)
  - [CNIG](https://cnig.gouv.fr/IMG/pdf/prj_standard_adresse_pour_revue.pdf)
  - [India Post PIN](https://uat.indiapost.gov.in/MBE/Pages/Content/Pincode.aspx)
  - [USPS Pub 28](https://pe.usps.com/text/pub28/28apc_002.htm)
  - [Indian address noise](https://sigir-ecom.weebly.com/uploads/1/0/2/9/102947274/paper_21.pdf)
  - [Mangalgi et al.](https://arxiv.org/pdf/2007.03020)
- Model licences: `https://huggingface.co/api/models/<id>` for each id in §5.5 (queried 25 Sep 2026). [Qwen2.5-3B LICENSE](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE), [Qwen3-8B card](https://huggingface.co/Qwen/Qwen3-8B).
