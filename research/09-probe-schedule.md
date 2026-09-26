# 09 — Probe schedule for the final day (27 Sep), with estimator math

Written 26 Sep 2026. Inputs: `01-challenge-intel.md` (ranking rules), `03-refinement-research.md` §3–4 (decision + France variants), `05-goldmine.md` Bet 1 (τ-triple retune). LB history: v2 = 0.941, v3 = 0.947 (full-test uploads, both already on the board). Tonight's round6 full upload (~0.975 expected) becomes baseline **B** and, crucially, the **anchor** S_B_pub.

Constants used throughout:

| symbol | value | source |
|---|---|---|
| N (test S1) | 1,732,544 | task brief |
| France S1 | 259,452 → w_F = 0.14976 | task brief |
| public fraction f | **assumed 1/3** (2025 was 25K/75K) | `01` §3; unpublished for 2026 |
| M = f·N (public S1) | ≈ 577,515 | derived |
| LB precision | 6 decimals (0.947442 observed) | our screenshots |
| ranking | **max score over all uploads**; ties → earlier upload wins | `01` §15, §68 |
| private | full test, "based on your final solution submission"; best-submission ZIP audited | `01` §16, §78 |

---

## 1. Estimator math

### 1.1 The difference estimator (use this, not the raw decomposition)

Public score of any submission = mean per-S1 F0.5 over the **fixed unknown** public set Q, |Q| = M. Scoring is deterministic: given a submission and Q, there is no noise. All "error" is (a) how well P∩Q represents P, and (b) how well the public read generalises to the private (full-test) mean.

Mixed submission: variant A on a known subset P (|P| = wN), baseline B bit-identical elsewhere. Let w_Q = |P∩Q|/M (realised public weight of P). Then

    S_mixed = w_Q · mean_{P∩Q}(F_A) + (1−w_Q) · mean_{Q\P}(F_B)

The raw decomposition S_A_on_P = (S_mixed − (1−w)·S_B_on_comp)/w needs an estimate of the complement's B-score. **Don't estimate it — cancel it.** Because tonight's round6 upload scores the identical B predictions on the same fixed Q:

    D := S_mixed − S_B_pub = w_Q · mean_{P∩Q}(δ),   δ_i := F0.5_A(i) − F0.5_B(i)

The complement cancels **exactly** (bit-identical predictions ⇒ identical per-S1 scores on Q\P). The estimator of the deployed-on-P effect is

    Δ̂_P = D / w        (per-S1-of-P units; global deployed effect if P = everything the variant touches)
    global contribution = w · Δ_P   (so D itself IS the mixed submission's global-score delta)

Requirements: (i) the complement predictions are byte-identical to the anchor upload's; (ii) both scores read from the same LB (Q is fixed for the whole event, so anchors persist across days).

### 1.2 Error source 1 — public-subset proportionality (the w_Q ≠ w risk)

Our split (parity of a hash of the S1 id) is random with respect to Q, whatever Q is. |P∩Q| is Hypergeometric(N, wN, M); with the finite-population factor (N−M)/(N−1) ≈ 2/3:

    SD(w_Q) = sqrt( w(1−w)/M · (N−M)/(N−1) )

| split | w | SD(w_Q) | relative (SD/w) |
|---|---|---|---|
| parity half | 0.500 | 5.4×10⁻⁴ | 0.11% |
| France-only | 0.14976 | 3.8×10⁻⁴ | **0.26%** |
| parity half of non-France | 0.4251 | 5.3×10⁻⁴ | 0.12% |

This enters Δ̂ multiplicatively (factor w_Q/w), so it is a ≤0.3% **relative** error on Δ — e.g. ±8×10⁻⁶ on a Δ_F of 0.003. Negligible against source 2.

Caveat for France: France is an *attribute*, not a random split. If the organisers stratified Q by country at a different rate than the test (nothing suggests they did; France is confirmed present in both splits, `01` Q&A), w_Q,F could be off *systematically*. Even then D = w_Q·mean(δ) keeps the **sign** and approximate scale; only the multiplier is wrong. Treat |Δ̂_F| as ±20% scale-uncertain, sign-certain.

### 1.3 Error source 2 — generalisation from P∩Q to all of P (dominant)

δ_i ≠ 0 only on "affected" S1s (decision flips). Let q = affected fraction of P, r² = E[δ²|affected]. Typical flip magnitudes: adding/dropping one record from a ~3.46-match set changes per-S1 F0.5 by ~0.1–0.3; empty-set/singleton flips by up to 1.0. So r ≈ 0.3–0.5. Then σ_δ = sqrt(q·r²) (mean-squared term second-order).

Private scores all of P; the public read covers P∩Q. Error of Δ̂_P as an estimate of the full-P mean:

    SE(Δ̂_P) ≈ σ_δ · sqrt( (1−f) / (f · wN) )   = σ_δ · sqrt(2/(wN)) at f = 1/3

| split | wN | q=1%, r=0.3 | q=2%, r=0.5 | q=5%, r=0.5 |
|---|---|---|---|---|
| parity (wN=866k) | 866,272 | SE = 4.6×10⁻⁵ | 1.07×10⁻⁴ | 1.70×10⁻⁴ |
| France (wN=259k) | 259,452 | 8.3×10⁻⁵ | 1.96×10⁻⁴ | 3.10×10⁻⁴ |

Affected-S1 counts (sanity): q = 1–5% of 1.73M = **17k–87k flips full-test**, of which ~5.8k–29k land in public; France q = 1–5% = 2.6k–13k flips, ~870–4,300 in public. Ample for the SEs above.

### 1.4 Detectability verdict

3σ criterion, in the units of the thing measured:

- **Parity probe, deployed-global Δ:** resolvable down to ~1.4×10⁻⁴ (q=1%) to 5×10⁻⁴ (q=5%). **Δ = 0.001 always comfortably resolvable; Δ = 0.0005 resolvable** except in the worst noisy case (q=5%, r=0.5 → 2.9σ, still directionally trustworthy).
- **France probe, Δ_F in France units:** resolvable down to ~2.5×10⁻⁴–9×10⁻⁴. **Δ_F = 0.001 resolvable (≈3–12σ); Δ_F = 0.0005 marginal (1.6–6σ)**. In global units the France probe resolves contributions of ~4×10⁻⁵–1.4×10⁻⁴ — i.e. any France change worth deploying (`03` expects +0.002–0.008 global from France work) is measured at ≥5σ.
- **LB rounding:** 6 decimals ⇒ ±5×10⁻⁷. Irrelevant. If the LB ever displays 3 decimals for us, every mixed-probe read below ~0.002·w dies — re-check the display before trusting a read.
- **If f = 1/5 instead of 1/3:** every SE scales by sqrt((1−f)/f · 3/2)... concretely ×1.4. Verdicts unchanged for 0.001; 0.0005 parity becomes 2σ.

### 1.5 Stacked (two-in-one) probes

Disjoint subsets separate algebraically. With P₁ = France (variant A) and P₂ = parity-half of non-France (variant C) in ONE upload:

    S_stack − S_B_pub = w_F·Δ_F + w₂·Δ_C,   w₂ = 0.4251

If Δ_F is already known from an earlier probe, Δ_C = (S_stack − S_B_pub − w_F·Δ̂_F)/w₂, at the cost of adding Δ_F's SE (scaled by w_F/w₂ ≈ 0.35) to Δ_C's. Usable; costs ~+6% SE on Δ_C. Never stack two *unknowns* — one equation, two unknowns.

---

## 2. Tomorrow's 5-upload schedule (27 Sep, all times IST)

### 2.0 Principles

1. **Every upload is a full, defensible submission.** Ranking is max-over-uploads, so a probe never hurts rank; but the private score is computed from the *best public* submission's predictions, and its ZIP is audited — so every upload must be reproducible (freeze config + git hash + seed per upload) and built on the current best base. Worst case a probe becomes the final max: acceptable, because it is best-base everywhere except a subset where the variant genuinely won.
2. **Probes measure only what offline mock validation cannot.** The Bet-1 mock world (test-density, per-country ratios) validates the global decision rule (GFM vs τ-triple), singleton threshold, and ensemble mode offline. It CANNOT validate France (zero French labels). LB slots therefore go to: (a) full model swap, (b) France, (c) deploy, (d) safety.
3. **Hard timing:** last upload in the queue by **20:30** (platform hangs at ~100MB, `03` §5); the deploy upload (#4) by 17:30 so there is time to verify it landed and react.
4. **Tie-break favours earlier uploads** at equal score — never delay a config you believe is the best.

### 2.1 Prerequisite — tonight (26 Sep, today's quota)

- **round6 full upload.** This is B and the anchor S_B_pub. Without it every probe tomorrow is uninterpretable. Verify the score renders at 6 decimals and record it in `PLAN.md`.
- Overnight: finish the mock-world retune (Bet 1 τ-triple), pick the offline winner between calibrated-GFM and τ-triple, pick singleton threshold, pre-build tomorrow's prediction files so each upload is a file-swap, not a pipeline run.

### 2.2 The five slots

| # | time (queue by) | content | what it answers |
|---|---|---|---|
| 1 | **08:30** | **v9/v10 full swap** | Is the new model > round6? Direct read, no estimator. Decides the base for the day. |
| 2 | **11:00** | **France-threshold probe**: best base everywhere; France S1s get the shifted threshold (prior-matched t, `03` §4.1) | Δ̂_F = (S₂ − S_base_pub)/0.14976 |
| 3 | **14:00** | **Stacked probe**: France variant kept as in #2 (whatever its sign — anchor algebra handles it) + the offline-winning decision-rule change (GFM-vs-τ or singleton shift, whichever the mock left uncertain) on parity-half of non-France | Δ̂_dec = (S₃ − S_base_pub − w_F·Δ̂_F)/0.4251 — the LB confirmation of the mock's verdict |
| 4 | **17:00** | **BEST full deploy**: base model + every variant that measured/validated positive, each applied to its full scope (France change on all France; decision rule globally) | The submission intended to be final. Predict its score first: S_base + w_F·Δ̂_F·1[deploy_F] + Δ̂_dec·1[deploy_dec]; a large miss (>4σ) means a bug — investigate before #5. |
| 5 | **19:30** (hard 20:30) | **Safety/ensemble**: if #4 landed as predicted → ensemble intersect-extras (or the second-best decision variant) built ON #4's config, only if mock said ≥ 0; if #4 misfired or hung → clean resubmission of the best-known-good config | Last chance; must not be a measurement we depend on. |

### 2.3 Decision tree

```
Tonight: round6 full  → S_B_pub  (anchor)
│
#1 08:30 v9 full → S_v9
├─ S_v9 > S_B + 0.001      → base := v9, anchor := S_v9. Probes #2/#3 rebuilt on v9 preds (pre-build both variants for BOTH bases tonight).
├─ |S_v9 − S_B| ≤ 0.001    → base := round6 (already anchored + earlier tie-break). v9 kept only as ensemble input.
└─ S_v9 < S_B − 0.001      → discard v9. Base := round6.
│
#2 11:00 France probe → Δ̂_F
├─ Δ̂_F ≥ +0.001 (≥3σ)      → deploy France change in #4. (Global gain = 0.150·Δ̂_F.)
├─ |Δ̂_F| < 0.001            → treat as null; do NOT deploy (F0.5 asymmetry punishes precision loss; null + risk = no).
└─ Δ̂_F ≤ −0.001             → deploy the OPPOSITE direction only if `03` §4.1's prior-matching independently predicted it; else drop France work.
│
#3 14:00 stacked probe → Δ̂_dec
├─ Δ̂_dec ≥ +0.0005 AND mock agreed      → deploy globally in #4.
├─ mock said + but LB says − (>3σ)      → trust the LB (it samples the real France/decoy mix); don't deploy.
└─ |Δ̂_dec| below resolution              → deploy only if mock gain ≥ +0.001 (mock is then the better instrument).
│
#4 17:00 deploy best config → compare to predicted score
├─ within 4σ of prediction → proceed to #5 ensemble (if mock ≥ 0) or stop.
└─ miss > 4σ OR upload hangs → #5 becomes: resubmit best-known-good (or retry #4 file). No new experiments.
│
#5 19:30 → as decided above. Nothing queued after 20:30.
```

### 2.4 What is deliberately NOT probed on the LB

- **Calibrated-GFM vs τ-triple as a *standalone* slot** — the mock world measures it to ±0.0003 with labels; peers found expected-F loses to a plain threshold on stage-1 scores (Epic021 −0.0005, Ayan −0.0004; wins only on calibrated stage-2, `05` §5). One stacked half-slot (#3) is confirmation, not discovery.
- **Singleton aggressiveness alone** — same reason; fold it into #3's variant if the mock flags it as the bigger unknown.
- **Ensemble intersect-extras measurement** — precision-oriented by construction; mock validates it; it rides #5 as a bonus, never as a dependency.

---

## 3. Risk table

| # | risk | size | mitigation |
|---|---|---|---|
| 1 | Public subset not proportional to our split (w_Q ≠ w) | SD(w_Q): 5.4×10⁻⁴ parity / 3.8×10⁻⁴ France → ≤0.3% *relative* error on Δ̂ | Negligible for parity (hash-random vs fixed Q). France: sign-safe, scale ±20% worst case if Q was country-stratified oddly; decisions use sign + threshold, not the exact value. |
| 2 | Generalisation P∩Q → P (dominant noise) | SE(Δ̂): ~0.5–1.7×10⁻⁴ parity, ~0.8–3.1×10⁻⁴ France (§1.3) | 3σ decision thresholds (0.001 France-units, 0.0005 parity); don't act on smaller reads. |
| 3 | Public fraction f unknown (maybe not 1/3) | SEs scale ×1.4 if f=1/5 | Verdicts on 0.001 unchanged; nothing in the plan depends on resolving <0.0005. |
| 4 | LB rounding | 6 decimals observed → ±5×10⁻⁷ | Confirm 6-decimal display on tonight's round6 read before trusting any probe. |
| 5 | Anchor mismatch (complement not bit-identical) | Silently breaks the exact cancellation → arbitrary bias | Build every mixed file by row-substitution into the anchor's TSV; `diff` the complement rows = 0 before upload. |
| 6 | Base swap mid-day (#1 wins) invalidates pre-built probes | Wasted slot if probe files reference the wrong base | Pre-build France + decision variants against BOTH round6 and v9 tonight. |
| 7 | Upload hangs / "Please Wait" (100MB files, `03` §5; 2025's 23:58 upload never landed) | Lose a slot, or lose the deploy | #4 by 17:00, nothing after 20:30, verify each score renders before building the next dependent step. |
| 8 | Max-score ranking + ZIP audit: a mixed probe ends up as the max | Private is scored from the best submission; ZIP must reproduce it | Every probe = best base + variant on subset ⇒ always defensible; keep per-upload config/hash/seed manifest so any of the 5 can be the ZIP. |
| 9 | Tie-break timing | Equal scores rank by earlier upload | Submit best-believed config as early as it exists; never hold back a ready improvement. |
| 10 | Probe drags the mixed score below the board's current best | None for rank (max-over-uploads keeps the previous best) | Worst realistic case: parity probe of a −0.005 variant costs the *mixed upload* −0.0025 vs base — the board keeps the base's score. Zero rank cost; only a slot is spent. |
| 11 | F0.5's precision asymmetry: a "null" France read hides a precision loss offset by recall | Δ̂ near 0 could mask internal trade | Also log predicted links/S1 for France in each variant; deploy only when Δ̂ > 0 AND links/S1 stays ≤ the prior-matched expectation (`03` §4.1). |

---

## 4. One-line summary of the instrument

With tonight's round6 anchor on the fixed public set, a mixed upload measures a variant's effect **exactly on the public intersection** — the only real noise is generalising ~86k–290k public S1s of P to all of P (SE ≈ 1–3×10⁻⁴), so **0.001 is comfortably measurable on both split types, 0.0005 is measurable on the parity split**, and the 6-decimal LB contributes nothing. Spend reads on the model swap and France; let the mock world settle everything that has labels; slots #4–#5 deploy, never discover.

## Addendum (26 Sep, 19:05) — redecide.py synth evidence pins Slot 2's variant
`tools/redecide.py` (validated vs synth hidden GT) re-decided synth_v10's saved pair probabilities:
stored `sel` = 0.955460; best rethreshold = **0.961406** at t1=0.5, t2=0.55, δ=0.35, **France shift −0.20**.
Every negative France shift beat 0; every positive one lost. On synth, v10's unseen-country precision
bias points the WRONG WAY: the unseen country wants LOWER thresholds (recall-starved, not precision-starved).
Magnitude is inflated (tuned on 2k entities) — sign is the finding.
**Slot 2 therefore probes the NEGATIVE France shift** (t−0.15 in calibrated units), not the precision-raised
variant from `03` §4.1. If Δ̂_F > 0 at ≥3σ, deploy in Slot 4's full run via redecide on the artifact parquets.
Slot 3's parity-half decision variant = the global (t2↓, δ↑) rethreshold from the same scan, minted by
`redecide.py --emit` from the artifact run's test_*.parquet (no model re-run needed).

## CORRECTION to the 26-Sep addendum (frontier agent, 8-seed replication)
The +0.0059 redecide gain and the negative-France-shift finding were largely SEED-0 LUCK plus tuning
on test GT: across 8 synth seed replicates (SD 0.0018-0.0020 per seed), rule-vs-GFM mean = -0.00001
and the France shift means are within noise (±0.001, SD 0.005). Real v4 LOCO points mildly POSITIVE
(+0.002-0.003 LOCO units); a France threshold move is worth at most ±0.0005 globally either way.
Slot 2 remains a France probe but with a ±0.0005 ceiling — deprioritize if a stronger variant exists.
VALIDATED instead: probability-level blending of independent seed runs (+0.0021 GFM k=2 on synth;
expect +0.0003-0.001 real) via scratch/blend_runs.py (bit-exact vs pipeline decisions).
TSV-level merges (tsv_ensemble extras/strict) are NEGATIVE on 7 of 8 seed pairs — do NOT use.
New v11 feature leads from real-data scan: house-number replace LR≈13 (missing |Δ|/keep-replace
features), empty-address records 5x error rate, 65% of OOF FPs are unowned generator decoys.
