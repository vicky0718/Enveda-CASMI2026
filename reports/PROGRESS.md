# Autopilot progress log — own pipeline (no third-party code, weights or pickles)

## Submission notebook versions (`vigneshnehru/casmi26-own-submit`, private)

| version | content | runtime (400 mol) | public LB |
|---|---|---|---|
| V5 | library + analog propagation + MetFrag-lite, LightGBM ranker (NP panels A+C) | 16 min | _to fill_ |
| V6 | + class-3 analog-edit generator (top-20 analogs, 23 edits) | 12 min | _to fill_ |
| **V7** | generator with top-100 analogs, 37 edits (homologs, CH2↔C=O, methylenedioxy) | 17 min | _to fill_ |

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
