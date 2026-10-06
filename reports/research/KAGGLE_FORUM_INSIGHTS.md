# Kaggle forum & public-notebook intelligence — Enveda CASMI 2026

*Snapshot: 6 Oct 2026, three weeks into a 13-week competition. Scraped with
`scripts/research/scrape_kaggle_forum.py` (raw posts stay in git-ignored `data/external/`).
Everything below is other teams' **self-reported** measurement unless marked "verified by us".
Thread numbers are Kaggle discussion IDs: `kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/<id>`.*

---

## 0. What was scraped, and an honest caveat about "top-100 rankers"

| item | count |
|---|---|
| discussion topics | 81 (70 substantive after removing team-recruitment spam) |
| comments + replies | 207 |
| distinct authors | 105 |
| public leaderboard teams | 2,618 |
| public notebooks inspected (source pulled) | 12 of the most-voted + the official metric |

**The top-100 leaderboard teams are almost silent.** Only **4 of the top-100 users** have posted
(6 posts total: M Sato #41, Patrick Chan #43, Mike.X #81, Farah Mohamed Ahmed #131 — all short
questions or "+1"s). The real intelligence comes from (a) the highest-engagement threads, written
mostly by teams ranked ~230–650, and (b) the public notebooks, which most of the leaderboard forks.
That is normal three weeks in; top teams usually publish only after the deadline.

### Leaderboard shape (public LB, 33% of the hidden test ≈ 130 molecules)

| rank | 1 | 5 | 10 | 20 | 50 | 100 | 200 | 300 |
|---|---|---|---|---|---|---|---|---|
| score | 0.473 | 0.443 | 0.435 | 0.431 | 0.425 | 0.422 | 0.417 | 0.415 |

* **169 teams sit in a 0.415–0.420 band** — forks of public fusion notebooks plus the
  visible-ID leak (see §7). The public LB is therefore heavily compressed and partly inflated.
* Gold on Kaggle for ~2,600 teams ≈ top 10 + 0.2 % ≈ **top ~15**; today that is ≈ 0.43–0.44 public.
* The public LB has ~130 molecules: **one molecule = 0.0077 public / 0.0025 overall MRR**;
  same-code resubmission noise is **±0.006–0.007** (#743254, prvsiyan notebook). Differences
  < ~0.015 between two single submissions are indistinguishable from noise.

---

## 1. The most-engaged threads, ranked (votes + comments)

| # | thread | votes | comments | one-line takeaway |
|---|---|---|---|---|
| 741359 | Welcome (host) | 24 | 9 | licence rules: no NIST, no non-commercial data in prize-eligible solutions |
| 744470 | Chasing medals with prize-ineligible sources | 13 | 14 | NIST-trained anything = disqualification; enforcement worry for non-prize medals |
| 745715 | Visible test molecule_ids leak class-1 answers | 18 | 8 | public notebook hard-codes visible id→SMILES; ~two dozen forks at 0.416–0.418 |
| 741404 | Complete statistical tour (EDA) | 23 | 2 | library search ≈ 0.907 on class-1; whole score differential is classes 2–3 |
| 743974 | All you need is a good ranker | 21 | 3 | recall is high; the ranker is the bottleneck |
| 741745 | [0.339] 4-channel analog propagation + neural Bayes rerank | 17 | 0 | the 4-channel architecture most of the LB descends from |
| 745029 | Understanding the actual task (hengck23) | 15 | 3 | read Enveda's own papers (MS2Mol, EnvedaDark, PRISM); NP databases COCONUT/LOTUS/NPAtlas |
| 744343 | trick: train a SMILES→MS/MS predictor too | 14 | 3 | forward model for re-ranking + synthetic data |
| 745070 | Claude CoT gives a clue | 12 | 7 | peaks → fragment formulas → assemble (SIRIUS-style), iterative refinement |
| 743254 | Lessons from ~30 submissions | 11 | 4 | **the best methodological thread** — see §3 |
| 744893 | DreaMS retrieval demo | 10 | 3 | DreaMS as a fast coverage probe; enveda-180 reference gives ~0 on hidden test |
| 742055 | Six variants, one plateau at 0.33–0.34 | 1 | 6 | **failure taxonomy: ranking beats recall 17:1** — see §2 |
| 741597 | np-examples as CV doesn't correlate with LB | 5 | 4 | holdout must be by InChIKey14 across *all* libraries |
| 742088 | Dissecting the plateau with probe submissions | 0 | 1 | top-25 ceiling ≈0.40 of a pipeline; filler SMILES scored 0.000 |

---

## 2. Where the score actually is (the most important decoded insight)

### 2.1 Test composition — inferred from leaderboard algebra, not given
The hosts refuse to disclose class shares (#745758). Two public derivations:

* **Class 1 ≈ 16 %**: library-only submission = 0.151 LB, class-1 MRR on honest simulation ≈ 0.93
  → f₁ ≈ 0.162 (prvsiyan; haideptry in #741597).
* **Class 2 ≥ 27 %**: hard lower bound f₂ ≥ f₂·ρ = 0.271 (ρ = pool recall ≤ 1). After a costly
  PubChem-expansion submission (0.335 → 0.205) prvsiyan concludes ρ ≈ 1 for the COCONUT+train pool,
  so **f₂ ≈ 0.27–0.30 and class 3 ≈ 0.55**. Others quote 16/45/39 — the 45/39 split rests on a
  weaker assumption. **Treat class 3 as 40–55 % of the test.**
* Consequence: **a retrieval-only system is capped at ≈ f₁ + f₂ ≈ 0.43–0.61**, and the realistic
  retrieval ceiling with today's rankers is ≈ 0.43 (prvsiyan). The current top (0.473 public) is
  above that, i.e. either a better-than-assumed class-2 ranker, some class-3 success, the leak, or noise.

### 2.2 Within retrieval, ranking ≫ recall
* Failure taxonomy on 250 held-out NPs under an honest class-2 condition (#742055, DancingLumberjack,
  replicated independently by alex chilton): truth absent from pool 1.6 %, truth ranked 2–25 **40 %**,
  ranked >25 2.4 % → **ranking headroom : recall headroom ≈ 17 : 1**.
* Oracle: removing same-formula "blockers" lifts MRR 0.73 → **0.987**; perfect same-formula separation
  is **95 % of the remaining gap**. In rank-2 failures 40/42 blockers have the truth's exact formula,
  69 % a *different* Murcko scaffold, median Tanimoto(truth, blocker) 0.48.
* ~66 of ~83 candidates in a 10 ppm window share the truth's formula (#742055); 98 % of wrong winners
  share the formula (#743254). **Formula/mass cannot separate them by construction.**
* Corollary measured by several teams: bigger pools (PubChem, NPAtlas, LOTUS, derivative enumeration)
  add ~nothing or hurt — unless paired with a ranker that separates same-formula isomers
  (huseyinemreaksoy's *gated* PubChem join was the exception: +0.011, 0.409 → 0.420).

### 2.3 Which signals separate same-formula isomers (unconfounded measurement, prvsiyan)
MRR inside the truth's exact-mass isomer group (random = 0.169, SE ≈ 0.025, n = 241):

| channel | MRR | vs random |
|---|---|---|
| mass-shifted analog propagation | 0.541 | +0.37 |
| spectrum→fingerprint model (`f·z`) | 0.490 | +0.32 |
| MetFrag-lite fragmentation | 0.278 | +0.11 |
| library similarity | 0.118 | **−0.05** (anti-informative on class 2) |

Methodology warning from the same work: evaluating a channel *on cases the ranker already failed*
makes every channel look sub-random (selection effect). Select evaluation sets by chemistry (formula),
not by the system under test.

---

## 3. Validation methodology the community converged on (and paid for)

1. **Hold out by InChIKey14 across every library**, never by `ingest_lib` or rows. The 250
   np-examples structures have 55,599 spectra in other libraries; a library-name holdout is class-1,
   not class-2. Sanity check: library search on the holdout must return **zero** self-matches.
2. **Class-1 simulation** must drop the query's own source library too, or it retrieves the identical
   spectrum and reports 1.000.
3. **Pool-provenance leak**: features like "is a training structure" or NaN-for-train-only columns
   scored 0.94 on a task that honestly scores 0.55.
4. **Match spectra-per-molecule to the test** (test placeholder 3.03; CV sets 4.3–5.0). Aggregation
   features measured +0.019 at CV counts but +0.0065 ± 0.0033 count-matched (#742055). Average ≥ 5
   resampling draws; a single draw is worth ±0.011.
5. **Tune ranker with folds held out by query (molecule)**; in-sample sweeps ranked configurations
   almost exactly backwards (prvsiyan).
6. **Hyperparameters tuned on the evaluation panel inflate effects ~2.8×** (+0.065 → +0.023 on
   held-out panels, #743254).
7. **Panel → LB conversion**: multiply a same-formula-panel gain by the share of isomer-limited
   molecules (~⅓) and discount for overlap; LB-visible threshold ≈ 0.03 offline (#743254).
8. **Pin and bag ranker seeds** (`HistGradientBoosting` default `random_state=None` → ±0.006 LB).
9. **Submit the unchanged notebook twice** to measure your own noise band before believing any delta.
10. **Never build artifacts keyed to the visible test's masses/formulas** — the hidden file is
    different; the feature silently returns the baseline (#744172).
11. **Validation does not rank working configurations reliably** on this competition (domain gap
    between held-out library NPs and hidden timsTOF NPs). Use local numbers as *filters* (leaks,
    recall, retention), spend submissions as *decisions* (prvsiyan).
12. Two proxies can disagree (public-library class-2 holdout vs timsTOF NP panel); for NP-targeted
    changes the timsTOF panel predicted the LB better (#743254).

---

## 4. What measurably worked (public LB deltas, single submissions unless noted)

| change | Δ public LB | source |
|---|---|---|
| spectral library search (entropy similarity) | 0.151 baseline | prvsiyan |
| + COCONUT candidates ranked by mass-shifted analog propagation | +0.08 | prvsiyan, starkhushi |
| + in-silico fragmentation (MetFrag-lite) channel | +0.02–0.03 | prvsiyan, starkhushi |
| + spectrum→fingerprint transformer (FPNet, 6,930 bits, softmax over 63 same-window decoys) | +0.03 | prvsiyan |
| **two input views (per-spectrum model + merged-spectra model), averaged** | **+0.019 (largest isolated effect)** | prvsiyan |
| entropy similarity instead of cosine (class 1 0.893 → 0.93) | — | prvsiyan |
| analog representatives restricted to the test's exact instrument (timsTOF) | analog c₂ 0.521 → 0.551 | prvsiyan |
| index library on formula mass instead of precursor+adduct | small, mechanical | prvsiyan / dariushafshar |
| PubChem/PubMed **popularity prior** inside same-formula groups | +0.005–0.010 | huseyinemreaksoy, haideptry |
| ICEBERG + GLACIER forward-model re-rank within same-formula groups | ≈ +0.01; GLACIER on [M+H]+ only +0.006–0.009 | fusion notebooks |
| gated PubChem join (only when lib_max < 0.7) | +0.011 | huseyinemreaksoy |
| fusion of two engines by weighted reciprocal-rank fusion | to ~0.40–0.42 | fusion notebooks |
| ChEBI + LIPID MAPS expansion | unresolved (inside noise) | prvsiyan |

Architecture of today's ~0.42 public stack: v4n engine (FPNet + LightGBM ranker over a 711k
train∪COCONUT pool, with "generate=True" derivative candidates and an XScorer cross-encoder) +
prvsiyan's analog engine, fused by RRF; ICEBERG/GLACIER re-rank same-formula groups; popularity
prior; PubChem-only proposals gated in by library-similarity and confidence ("Skomuro promotion").

## 5. Measured dead ends (save the submissions)

* Blind PubChem expansion: 0.335 → **0.205** (catastrophic dilution); top-K PubChem isomers equally bad.
* Candidate cap of 80 "by library similarity × 100 − |Δmass|": silently deletes 24 % of class-2
  answers (library sim is 0 for class 2 by definition) → −0.053.
* Wider mass windows (±20–30 ppm): worse; timsTOF precursors are within ~5 ppm (99 %), +1.4 ppm bias.
* NP-likeness as a prior: worse than random. Membership flags in curated DBs: great on holdouts, hurt LB.
* Targeted derivative enumeration (±O, ±CH₂, ±hexose, +acetyl): 0.337 → 0.335.
* Third model in the same input view: −0.006. Cross-encoder over (z, f) without seeing peaks: ≤ 0.
* DreaMS embeddings as analog similarity: worse than entropy on mass-shifted peaks (no shift invariance);
  on same-formula blocked pairs DreaMS is below chance (20/42).
* Fine-tuning FPNet on timsTOF (= enveda-180 chemistry): offline +0.03, LB −0.009.
* Fragmentation scorer tweaks: hand-written bond-propensity rules ≈ 0; **parsimony helps**
  (down-weight 2-bond cleavages ×0.6, linear intensity, 0.005 Da tolerance, H-shift −2..+3).
* Formula/sub-formula gate, CE-depth, base-neutral-loss, ring-cut infeasibility: real separation on
  selected cohorts, ~0 marginal over the ranker.
* Isotope M+1 in MS2: unbiased carbon-count estimate (±25 %), usable for ~41 % of molecules, +0.011
  over precursor mass alone → dropped. M+2 suppressed ~9×.
* Adduct re-hypothesis for the test: 0/250 np-examples miss under their labelled adduct — Enveda's
  own adducts are clean; only library training data needs cleaning.
* Generated (de novo) candidates added to the list: no LB gain *so far* for anyone who reported.

---

## 6. Data quirks surfaced by the community (we verify the important ones in EDA pass 5)

1. **Positive-mode m/z +0.55 mDa electron-mass offset** in Enveda processing (Na calibrant used the
   neutral-atom mass). Fragments read ~0.49 mDa high in [M+H]+ vs [M−H]−. Host is checking which test
   spectra are affected (#743395). Matters for 0.005 Da fragment tolerances.
2. **Tautomer key ≠ shipped `inchikey14`** for ~4.4 % of NP structures; ~1 in 22 have a metric-key
   twin (#742042). Deduplicate and match on the metric key, never the shipped column.
3. **Answers are stereo-stripped before canonicalisation** (host, #744556): strip stereo from candidates.
4. **Invalid guesses consume a rank**; padding rows with filler was reported to score 0.000 (#742088).
5. **22 pairs of byte-identical spectra with conflicting labels** (msdial vs riken, #744446).
6. **pluskal formate** `precursor_error_ppm` artefact; GNPS/MassBank adduct mislabels (≈12k formate rescues).
7. Hidden test has **no ion-mobility/CCS** (host); same columns as test.parquet; **test and
   sample_submission are both replaced** at scoring.
8. Visible test: 97/400 molecules have both polarities; multi-adduct neutral masses agree to median
   1.3 ppm; only 24 % have the full 20/40/60+merged positive ladder.
9. Enveda-180 Zenodo release has 816 structures absent from train (filtered in prep; host: fine to use).

## 7. Rules & host rulings that constrain the design (prize-eligible)

| allowed | not allowed |
|---|---|
| PubChem structures for retrieval; COCONUT (own licence suffices); ChEBI, LIPID MAPS | NIST (any model trained on it ⇒ disqualification) |
| MassSpecGym and models trained on it (ICEBERG, GLACIER, MIST/FRIGID weights) | METLIN-trained / vendor-library-trained weights (CFM-ID 4 stock models explicitly *allowed*) |
| DreaMS weights; ChemBERTa; CFM-ID 4 (+ LGPL runtime deps) | non-commercial inputs → removal from **leaderboard**, not just prizes (host, #745841) |
| newer GNPS / MassBank / MoNA releases incl. CC BY-SA; Enveda-180 Zenodo release | — |
| models trained on train.parquet; training on private cloud compute | — |

**The visible-ID leak (#745715).** A public notebook hard-codes the 400 visible molecule_id → SMILES
pairs; it scores because the hidden rerun apparently reuses some ids. Several teams asked the hosts to
remap ids before the final rerun. **We will not use it**: it is not identification, the private LB is
likely remapped, and it inflates the public LB we would be measuring against. Practical consequence:
**discount public-LB scores of forks in the 0.415–0.420 band**.

---

## 8. What nobody has cracked yet — our opportunities

1. **Class 3 is the largest unclaimed block (40–55 % of molecules) and every public pipeline scores ~0
   there.** Even 10 % top-1 on class 3 is worth ≈ +0.04–0.05 MRR — larger than any retrieval tweak
   on the forum. Recent literature (FRIGID, FOAM; see LITERATURE_REVIEW.md) reports 15–25 % top-1
   de novo on NPLIB1/MassSpecGym with forward-model-guided refinement, and its weights are
   MassSpecGym-trained (eligible). Nobody on the forum has deployed it.
2. **MRR-optimal merging of retrieval and generation.** "Generation can hurt before it helps"
   (#741659) is only true for naive merging. Ordering all candidates by a *calibrated* P(correct) is
   expected-MRR optimal (rearrangement inequality), so the problem reduces to calibrating
   P(answer ∈ pool) and P(generated = truth) per molecule. The forum's best "not in pool" gate had
   AUC 0.63 — this is a modelling target, not a dead end.
3. **Instrument-matched forward model.** Public ICEBERG/GLACIER are MassSpecGym-trained (mostly
   Orbitrap/QTOF). enveda-180 gives **1.15M timsTOF spectra at 20/40/60 eV**: fine-tuning a forward
   model on it teaches timsTOF fragmentation physics (instrument transfers even if chemistry does not —
   the reverse of the FPNet fine-tune that failed). Retrieval-augmented forward prediction (MARASON:
   27 % vs 19 % top-1) is the matching upgrade for isomer separation.
4. **More input views** — the largest isolated effect (+0.019) was a second *view*. Untried views:
   CE-conditioned encoder, per-polarity encoders fused at molecule level (97/400 molecules have both
   polarities), neutral-loss view, fragment-formula-annotated view.
5. **Fragment-formula annotation as a first-class representation** (SIRIUS/MIST-style), with the
   0.55 mDa positive-mode correction, enabling 0.005 Da tolerances.
6. **Molecule-level multi-adduct / multi-polarity consensus** (neutral masses agree to 1.3 ppm) as a
   formula anchor and as extra evidence for the forward model.
