# Edge cases of spectrum→structure scoring and the workarounds built for them

Measured on the calibrated proxy (panel C = 909 public-library natural products, regimes C1/C2, no
generator; this configuration predicted V5's leaderboard 0.324). Ranker out-of-fold, molecules held out.

| # | edge case (measured on our data) | workaround (code) | effect |
|---|---|---|---|
| 0 | **FP model inference bug**: checkpoints saved in fp16 rounded the sinusoidal m/z tables → 119 rad phase error at m/z 500; even *training* structures scored barely above the bit-frequency prior (BCE 0.22 vs 0.25) | load / save `SinEmb.inv` in float32 (`pipeline.load_fp_models`, `fpmodel.load`, `train_fp.py`) | held-out within-formula MRR of f·z: C 0.22 → 0.42, B 0.20 → 0.60; seen structures 0.35 → 0.91 |
| 1 | merged-spectrum view is out of distribution for a model trained on single spectra | per-spectrum logits only, reliability-weighted (`pipeline.fp_logits`) | C 0.31 → 0.42 (within-formula, f·z alone) |
| 2 | FP generic-structure bias (f·z grows with bits the model calls likely a priori) | prior-normalised f·(z − z_prior) as `fp_norm` (`edge.fp_prior_logits`, `fp_prior.npy`) | within-formula C 0.42 → 0.45, B 0.60 → 0.66 |
| 3 | 33–34 % of class-2 truths have a same-formula isomer with ECFP Tanimoto ≥ 0.9 (positional swaps) | atom-pair (topological-distance) similarity in analog propagation: `analog_ap`, `analog_ap_w`, `ap_tmax` (`edge.ap_matrix`) | with 2–4: C C2 0.603 → 0.613 |
| 4 | minor adducts (Na/K/NH4/Cl: ≤ 1.8 % of test, 7.5 k–178 k training spectra) and sparse spectra weigh as much as rich [M+H]+ ones | per-spectrum reliability = adduct prior × min(1, peaks/8) for analog fusion (`analog_w`) and FP fusion (`edge.spectrum_weights`) | (included in row 3) |
| 5 | low-information molecules (few peaks, low entropy, no good library hit) | per-molecule descriptors broadcast to candidates (`q_npeaks`, `q_entropy`, `q_best_hit`, `q_best_direct`, `q_reliable`, …) so the ranker learns when spectra are trustworthy | (included in row 3) |
| 6 | the submitted no-generator notebook used a ranker trained *with* generated candidates and C3 | ranker trained on the no-generator configuration (`train_ranker.py --nogen`, ranker kernel) | C C2 0.567 → 0.603 |
| 7 | isomers with their own non-matching library spectra; analog self-support | `own_n`, `own_sim`, `own_neg`, `analog_noself` (earlier round) | C C2 +0.010 |
| 8 | fragments every isomer explains carry no isomer information | `frag_disc` (earlier round) | C weighted +0.008 |
| 9 | popularity is leaky as a feature (famous host examples) | tie-breaker λ = 0.1 tuned on B/C only | C weighted +0.007, B −0.005 |

Combined (V10): no generator + rows 0–9 + fixed FP model (run 3, pure BCE) with f·z and fp_norm in the
ranker: **C C1 0.707, C C2 0.626** (V5: 0.691 / 0.590; V8-nogen ranker: 0.695 / 0.603); weighted C
0.301–0.395.

Not yet addressed: class-3 truths absent from every list (41–60 %), library precursor/adduct mislabels
(neutral mass from precursor rather than structure), NP-weighted FP training (6.7 % of training
structures are NP-like), and a forward (structure→spectrum) model for fingerprint-blind isomers.
