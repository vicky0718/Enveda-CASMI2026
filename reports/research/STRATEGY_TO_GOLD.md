# Strategy to gold — Enveda CASMI 2026

*Synthesis of our EDA (`reports/eda/EDA_REPORT.md`, passes 1–5), the forum/notebook intelligence
(`KAGGLE_FORUM_INSIGHTS.md`) and the literature (`LITERATURE_REVIEW.md`). Written 6 Oct 2026;
deadlines: team merge/entry 7 Dec, final submission 14 Dec.*

---

## 1. Where the points are

The leaderboard behaves like

$$\text{MRR} \;\approx\; f_1 c_1 \;+\; f_2\,\rho\,c_2 \;+\; f_3 c_3$$

| class | share f (forum algebra) | today's best c | today's contribution | headroom |
|---|---|---|---|---|
| 1 — has public spectra | ≈ 0.16 | ≈ 0.93 (saturated) | ≈ 0.15 | ~0 |
| 2 — in PubChem/COCONUT, no spectra | 0.27–0.45 (ρ ≈ 1 for train∪COCONUT) | ≈ 0.6–0.7 | ≈ 0.18–0.27 | **isomer ranking**: same-formula separation is 95 % of the remaining retrieval gap |
| 3 — in no database | 0.39–0.55 | **≈ 0 for every public pipeline** | ≈ 0 | **largest unclaimed block** |

What a realistic improvement is worth overall:

| lever | plausible gain in c | Δ MRR |
|---|---|---|
| class-2 isomer ranking +0.05 (e.g. timsTOF forward model) | c₂ 0.65 → 0.70 | +0.015–0.02 |
| class-3 de novo at 5 % top-1-equivalent | c₃ 0 → 0.05 | +0.02–0.03 |
| class-3 at 10 % (FRIGID-level on benchmarks: 15–25 % top-1) | c₃ 0 → 0.10 | **+0.04–0.055** |

The public stack (≈ 0.40–0.42 public, ~0.41 without the leak) plus either lever lands in today's gold
range (≈ 0.43–0.44 public). Both together is the safest route. **Everyone forks the same retrieval
stack; class 3 is where we can separate.**

## 2. Non-negotiable engineering rules (from EDA pass 5 and forum post-mortems)

1. **Metric fidelity**: rdkit **2026.3.3**, keys from `casmi.metric.candidate_key` (stereo stripped →
   tautomer canonical → InChIKey14). Dedup by metric key; never pad with invalid SMILES.
2. **Data hygiene**: drop non-trivial conflicting-duplicate rows (1,735 groups, mostly pluskal);
   ignore the pluskal formate `precursor_error_ppm`; re-assign or drop offset-mislabelled library
   adducts; drop dimers and non-test adducts from references; −0.4 mDa on Enveda positive m/z.
3. **Tolerances**: fragments ≤ 3 mDa high-res (two-tier with 0.5 Da for ion trap / QqQ references);
   precursor window ±10 ppm on formula mass.
4. **Inputs**: timsTOF peaks carry signal to ~0.03 %; use soft weighting and two views
   (per-spectrum + merged) rather than a hard 1 % cut.
5. **Molecule-level fusion**: 52 % of test-like molecules have both polarities, 64 % multiple adducts.
6. **Runtime**: everything keyed to the hidden test is built *inside* the scored run; nothing derived
   from visible-test masses; a per-molecule fallback so one failure never zeroes the file.
7. **Licences**: prize-eligible only — no NIST/METLIN-trained weights except the host-approved
   CFM-ID 4; MassSpecGym-trained ICEBERG/GLACIER/MIST/FRIGID, DreaMS, COCONUT, PubChem, ChEBI,
   LIPID MAPS are approved. Do **not** use the visible molecule_id → SMILES leak.

## 3. Validation harness (build first; everything else is measured on it)

* **Splits by metric key across all libraries** (InChIKey14-wide masking), three regimes:
  * *C1*: query's own source library removed, other libraries' spectra of the structure kept;
  * *C2*: every spectrum of the structure removed from every library, structure stays in the pool;
  * *C3*: additionally removed from the candidate pool (and from PubChem/COCONUT gates).
* **Query panels**: (a) the 250 np-examples (timsTOF, test pipeline) and (b) ~1,000 public NP
  structures (NP-likeness > 1, COCONUT-covered), each resampled to **test-like spectra counts** (mean 3)
  with ≥ 5 draws.
* **Leak checks** that must read zero: library self-matches under C2/C3, pool-provenance features,
  ranker folds held out by molecule.
* **Combine** as predicted LB = f₁·c₁ + f₂·c₂ + f₃·c₃ with a prior over f (16/30/54 and 16/45/39
  bracket) and report both ends.
* **LB discipline**: measure our noise band (two identical submissions); single-variable A/B only;
  ignore deltas < 0.015; re-fit f/ρ after each scored submission.

## 4. Plan

### Phase 0 — foundation (now → ~12 Oct, CPU here)
1. Cleaned, metric-keyed spectral library + pool (train ∪ COCONUT ∪ ChEBI/LIPID MAPS) as artifacts.
2. Validation harness (§3) with the honest baselines: entropy library search (C1), analog
   propagation (C2), and their numbers on both panels.
3. Reproduce the public ~0.40 fusion stack *from our own artifacts* (FPNet-style `f·z` model trained by
   us, analog engine, ranker) so every component is ours, licence-clean, and measurable.

### Phase 1 — isomer separation for class 2 (~12 Oct → 2 Nov, GPU: Kaggle / private cloud)
4. **timsTOF forward model**: initialise GLACIER/ICEBERG from MassSpecGym weights, fine-tune on
   enveda-180 (1.15M spectra, exact 20/40/60 eV) + np-examples, collision-energy conditioned; score
   candidates by entropy similarity of predicted vs observed spectra per CE, fused over a molecule's
   spectra. Success criterion: within-formula MRR on the C2 panel above the analog channel (0.54).
5. **Retrieval-augmented forward prediction** (MARASON-style): transfer intensities from the measured
   spectra of close analogs (we have them for 74 % of test-like molecules at Tanimoto ≥ 0.7).
6. **Formula-annotated peak view** (MIST-style, 2–3 mDa, offset-corrected) as an extra encoder view.
7. Re-sweep ranker, re-fit the class prior; one LB A/B per change.

### Phase 2 — class 3 (~26 Oct → 23 Nov)
8. **Generators**: (a) FRIGID with MassSpecGym weights (formula + MIST fingerprint conditioned,
   forward-model-guided refinement); (b) **analog-edit generator**: ±CH₂, ±O, ±hexose, ±H₂, ±C₂H₄,
   ±C₂H₂O on the top analogs (covers 56 % of test-like molecules in principle, §11.8); (c) FOAM-style
   formula-constrained GA seeded by (a) and (b). All scored by the Phase-1 forward model.
9. **Calibrated merge**: model P(answer ∈ pool) per molecule (features: best analog similarity,
   pool `f·z` margin, forward-model fit of the best pool candidate, generator agreement) and
   P(correct) per candidate; sort by P (expected-MRR optimal). Accept only if the C1/C2 panels
   lose < 0.005 while C3 gains.

### Phase 3 — hardening (23 Nov → 14 Dec)
10. Seed bagging, runtime budget on a T4 (≤ 9 h with margin), fallbacks, licence audit, and two final
    selections: **conservative** (retrieval + forward model) and **aggressive** (+ gated generation).

## 5. Immediate next tasks (in order)
1. Build `casmi/clean.py` + library/pool artifacts with the §2 rules.
2. Build `casmi/validate.py` (C1/C2/C3 masks, count-matched resampling, predicted-LB report).
3. Entropy library search + analog propagation baselines on the harness.
4. Ranker over the channel features with molecule-held-out folds.
5. Kaggle submission notebook skeleton (offline wheels incl. rdkit 2026.3.3, artifact loading,
   per-molecule fallback) and two identical submissions to measure our noise band.
