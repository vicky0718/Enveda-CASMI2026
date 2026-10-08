# Autopilot progress log — own pipeline (no third-party code, weights or pickles)

## Submission notebook versions (`vigneshnehru/casmi26-own-submit`, private)

| version | content | runtime (400 mol) | public LB |
|---|---|---|---|
| V5 | library + analog propagation + MetFrag-lite, LightGBM ranker (NP panels A+C) | 16 min | **0.324** |
| V6 | + class-3 analog-edit generator (top-20 analogs, 23 edits) | 12 min | not submitted |
| V7 | generator with top-100 analogs, 37 edits (homologs, CH2↔C=O, methylenedioxy) | 17 min | **0.307** |
| V8 (`-nogen` v2) | V5 + own-spectrum features + frag_disc + popularity tie-break + 600-hit fix | 13 min | **0.342** |
| V10 (`-nogen` v4) | + edge-case features, fixed FP model, ranker trained without generator | 12 min | 0.335 |
| V11 (`-pc`) | V10 features (no FP) + top-10 PubChem candidates, ungated | 22 min | 0.333 |
| V12 (`-moe`) | multi-model system's all-evidence expert (FP incl.) trained with the analog-thinned regime C2H + timsTOF mass window (±5 ppm, centre −0.8) | 20 min | 0.342 (= V8; predicted +0.01 is below the ±0.016 resolution) |
| V13 (`-moe-pc`) | V12 + 100 most popular PubChem structures per mass window, ranked with f·z (full expert trained on C1/C2/C2H/C2P, 3 seeds); PubChem window follows the timsTOF window | 24 min | **0.372** |
| V14 (`-moe-pc` v2) | V13 + NP-likeness feature | 25 min | 0.353 (−0.019 vs V13: validation gain did not transfer — like curated-DB flags, NP-likeness favours well-documented validation truths; feature dropped) |
| probe (`-probe-class1`) | V12 lists restricted to candidates with a direct library match (= f1 × MRR1) | 19 min | **0.149** = f1 × MRR1 ≈ 0.16 × 0.93: class 1 is at the library-search ceiling; V13's classes 2+3 = 0.223 |
| V8 (V9 of own-submit) | same with generator | 13 min | not to be submitted (generator hurts LB) |
| reference: `casmi26-fusion-base` (third-party fork) | | | 0.380 |

**LB calibration.** The generator *costs* 0.017 on the leaderboard (V7 0.307 vs V5 0.324) although
validation credited it +0.10–0.20: real class-3 molecules have no close library analogs (validation C3
truths do), so generated structures only dilute class-2 lists. The pool-only ranker's panel-C estimate
(0.29–0.38) brackets V5's 0.324, so **panel C without the generator is the calibrated proxy**; panel A and
any C3 credit are not.

## Validation (molecules held out across every library; folds held out by molecule)

Panels: **A** = 250 np-examples (timsTOF, the host's test-pipeline examples); **C** = 909 public-library
natural products (NP-likeness > 1, other instruments); **B** = 975 NP-like enveda-180 structures
(synthetic-like, reported but not used to train the ranker — it teaches provenance shortcuts).
Regimes: **C1** library spectra of the structure available (own library removed); **C2** structure
in the pool but no spectra anywhere; **C3** structure in no database (removed from the pool too).

MRR@25, out of fold:

| configuration | A C1 | A C2 | A C3 | C C1 | C C2 | C C3 |
|---|---|---|---|---|---|---|
| heuristic (direct match else analog) | 0.932 | 0.541 | 0 | 0.653 | 0.475 | 0 |
| ranker, pool only (V5) | 0.904 | 0.755 | 0 | 0.691 | 0.590 | 0 |
| + generator, 20 sources (V6) | 0.894 | 0.734 | 0.354 | 0.673 | 0.562 | 0.159 |
| + 100 sources, 37 edits (V7) | 0.891 | 0.726 | **0.437** | 0.664 | 0.552 | **0.231** |

Scoring fix (casmi.rank): a generated structure whose standard InChIKey14 equals the truth's but
differs from its metric key was previously not counted; with label-based MRR the same V7 model reads
A C3 **0.459**, C C3 **0.247**.

Class-share weighted estimate (f1 .16, f2 .30–.45, f3 .54–.39): V7 ≈ 0.61–0.65 on A, 0.40–0.45 on C
(panel A is analog-rich, so it is the optimistic end; the leaderboard is the arbiter).

Oracle coverage of the generator (truth among products, C3): one-step edits from 100 analogs —
A 58 %, C 43 %; two-step — A 63 %, C 49 % at ~6–10× more products. **Rejected after ranking**:
two-step (top-30 analogs) gives A C3 0.423 / C C3 0.217 (vs 0.437 / 0.231) and 13× the runtime —
the extra coverage is outweighed by dilution.

## Running

* Own FP transformer (spectrum → 6,919 fingerprint bits; BCE + same-window decoy softmax):
  `casmi26-fpnet-train` (A/B held out) and `casmi26-fpnet-train2` (A/B/C held out), T4, ~7.5 h each.
  Our egress policy blocks Kaggle's output-file host, so the f·z feature is computed and the ranker
  retrained **on Kaggle** (`kaggle/ranker_fp/ranker_fp.py`, kernel `casmi26-ranker-fp`): it attaches
  the FP-training output + our datasets `casmi26-artifacts` / `casmi26-eval`, prints the out-of-fold
  table (with / without f·z / f·z alone) to the log, and saves `fpnet.pt` + rankers; the submission
  notebook attaches that kernel output. Dry run passed locally with a random tiny model.

## FP model, run 1 (BCE + 0.01 × decoy softmax; A/B held out, C seen in training)

13.6 epochs on 2×T4 (7.7 h); monitor slice of *training* structures: top-1 among ~28 same-window
decoys 0.89, but per-bit BCE only 0.267 → 0.205. On held-out panels f·z alone is weak
(A C2 0.215 vs random ≈ 0.12; B C2 0.387) and adding it to the ranker does not help
(weighted A 0.604 vs 0.612 without) — the contrastive term memorised training structures instead of
learning transferable substructure bits. Run 3 (pure BCE) launched; V7 (no FP) stays the submission.

## FP model, run 2 (same loss; A/B/C all held out) — honest on every panel

f·z alone: A C2 0.211, C C2 0.122 (random ≈ 0.12 / 0.05); ranker with f·z vs without:
weighted A 0.611 vs 0.612, C 0.402 vs 0.404 — neutral. Ranker capacity (31/63/15 leaves) within ±0.004.
Decision: V7 stays the submission; run 3 (pure BCE, all panels held out) decides whether the FP
channel is worth shipping.


## Round: multi-model system + analog-thinned stress regime (C2H)

LB V8/V10/V11 sit inside ±0.009 — resubmission noise is ±0.007, so +0.01 changes are invisible; only
≥ 0.02 changes can be judged. Hypothesis: validation molecules have measured relatives (best analog Tanimoto
≥ 0.7 for 24 % / 30 % of panels A / C), the ranker leans on analog propagation, real test molecules have
fewer close relatives. Stress regime **C2H** additionally removes every library spectrum of the truth's
close analogs (ECFP4 Tanimoto ≥ 0.7).

| model (out of fold) | A C2 | A C2H | C C1 | C C2 | C C2H |
|---|---|---|---|---|---|
| current ranker (C1+C2, no FP) | 0.812 | 0.704 | 0.721 | 0.640 | 0.527 |
| full expert with FP, trained with C2H | 0.839 | **0.777** | 0.715 | 0.646 | **0.595** |
| stacked multi-model (library / analog / spectral / full + router) | 0.809 | 0.756 | 0.718 | 0.642 | 0.593 |
| RRF of experts | 0.734 | 0.686 | 0.697 | 0.581 | 0.545 |

Stacking/routing ties the all-evidence expert once FP and C2H are in training; RRF is worse. V12 ships the
all-evidence expert (config `moe.CONFIG["inference"] = "full"`; meta selectable).

## Round: what the test's classes look like (Oct 8)

* **Panel membership.** Panel A (host np-examples, the most test-like) is 99.2 % in COCONUT — already in our
  pool; panel C 62 %; panel B 0.4 %. In PubChem (±5 ppm window, InChIKey14): A 99 %, B 97 %, C 76 %; the
  truth's popularity rank in its PubChem window: A median 0 (top-10 for 87 %), C median 30, B ~6,500.
* **V13 = PubChem top-100 (by popularity) in every list + f·z for every row** (`kaggle/moe_fp_pc`, out of
  fold, 3-seed full expert):

  | | A C1 | A C2 | A C2H | A C2P | C C1 | C C2 | C C2H | C C2P |
  |---|---|---|---|---|---|---|---|---|
  | without PubChem rows | 0.927 | 0.837 | 0.787 | 0 | 0.717 | 0.650 | 0.596 | 0 |
  | with PubChem rows (V13) | 0.931 | 0.841 | 0.781 | **0.815** | 0.685 | 0.607 | 0.561 | **0.378** |

  Panel A pays nothing for the extra rows; on panel C V13 breaks even at ~5.5 % PubChem-only test molecules.
* **Our FP model alone** (f·z, pool window): top-1 A 0.224 / B 0.492 / C 0.291 (MRR 0.33 / 0.63 / 0.43);
  the forum quotes 0.46–0.49 top-1 for the public FPNet on np-examples (candidate sets may differ) → FP run 4
  (`kaggle/fp_train4`: pairwise m/z-difference attention bias) is training.
* **Real novelty.** Panel N = 3,015 NP-like training structures absent from both PubChem and COCONUT
  (`scripts/build/make_novel_panel.py`). Generator reach in C3 (`scripts/eval/novel_eval.py`): N 35 %
  (top-1 8 %, top-5 17 %) vs simulated panel C 32 % (top-1 14 %, top-5 23 %) — real novel molecules are
  rebuilt about as often (their best analog similarity is high, median 0.88). V7's loss came from mixing
  generated rows into class-1/2 lists, not from reach → gated merge prototype (`scripts/eval/merge_c3.py`).

## Round: class-1 probe, NP-likeness (Oct 8, afternoon)

* **Class-1 probe = 0.149** (V12 lists restricted to candidates with a direct library match): f1 × MRR1 ≈
  0.16 × 0.93 — class 1 is at the library-search ceiling. V13's 0.372 = 0.149 (class 1) + 0.223 (classes
  2+3, i.e. ~0.50 MRR on class 2 if class 3 ≈ 0). Every further point is class 2/3.
* **NP-likeness** (Ertl, RDKit Contrib NP_Score; structure-intrinsic, no membership flag): truths median
  1.08 (panel A 1.47) vs all candidates −0.58 — PubChem windows are mostly synthetic. Out of fold on the
  V13 harness (no FP): A C1 0.932→0.946, C2H 0.772→0.798, C2P 0.815→0.840; C C1 0.682→0.703,
  C2H 0.543→0.577, C2P 0.369→0.405 (panel C was selected on NP-likeness, so C overstates; A is not).
  → V14 = V13 + NP-likeness.
* **V14 validation with f·z** (`casmi26-moe-fp-pc` v2, CPU): with PubChem rows, NP-likeness on vs off —
  A C1 0.930→0.946, C2 0.827→0.834, C2H 0.791→0.789, C2P 0.812→0.827; C C1 0.690→0.704, C2 0.615→0.636,
  C2H 0.569→0.587, C2P 0.383→0.412; panel-C estimate (15 % PubChem-only) 0.345→0.358.
* **V14 LB 0.353 < V13 0.372** → NP-likeness off (kept in code, not in the shipped feature set). Lesson, as the forum warned for membership flags: priors that separate known library NPs from distractors overstate on our panels.
* **PubChem budget** (`casmi26-moe-fp-pc` v3, one top-300 harness, top-n subsets on the same folds, no NP):
  panel-C estimate (15 % PubChem-only) top-50 0.341, top-100 0.346, top-200 0.346, top-300 0.344 — flat;
  each extra reachable truth brings matching distractors. V13's top-100 stays; not worth a submission.
  The decisive lever is isomer ranking (FP model runs 4/5, then formula-annotated peaks).
