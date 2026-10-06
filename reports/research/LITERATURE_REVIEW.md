# Literature review — identifying natural products from MS/MS (CASMI 2026)

*Scope: the chemistry/physics that determines what an MS/MS spectrum can tell us, and the ML methods
for each sub-problem. Numbers marked **[v]** were re-checked against the paper's abstract/summary via
web search during this review (arXiv/PMC full texts are blocked from our environment); unmarked
numbers are from prior knowledge of the papers and should be re-verified before being relied on.
Eligibility = whether weights/data can be used in a prize-eligible solution per host rulings
(see KAGGLE_FORUM_INSIGHTS.md §7).*

---

## Part A — Theory: what the measurement physically encodes

### A1. Ionisation and adducts
* ESI produces **even-electron** ions: [M+H]⁺, [M+Na]⁺, [M+NH₄]⁺, [M−H]⁻, [M+HCOO]⁻ (= [M+CH₂O₂−H]⁻),
  [M+Cl]⁻; in-source water losses ([M−H₂O+H]⁺, [M−2H₂O+H]⁺) are typical of polyols, glycosides and
  terpenoid alcohols — themselves a chemistry hint.
* **Na⁺/K⁺ adducts fragment poorly and differently** (charge fixed on the metal, fewer
  charge-directed cleavages, sugar losses dominate). Models should condition on adduct.
* **Formate adducts in negative mode** typically lose HCOOH (46.005) first, giving [M−H]⁻ — visible as
  the second-most-frequent neutral loss in our np-examples analysis (fig. 12).

### A2. Fragmentation chemistry of even-electron ions (CID / HCD)
* **Even-electron rule**: even-electron precursors preferentially lose *neutral molecules*, giving
  even-electron fragments; radical losses (•CH₃, •OH) occur but are diagnostic exceptions (e.g. methoxy
  groups of flavonoids lose •CH₃ in negative mode).
* **Charge-directed** (heterolytic, near the protonation site) vs **charge-remote** (far from the
  charge, e.g. along fatty-acyl chains) mechanisms. Proton mobility decides which sites cleave: the
  proton migrates to the most basic site at low energy and becomes mobile at higher energy.
* **Characteristic neutral losses** (our fig. 12 confirms their enrichment in NPs): H₂O 18.011,
  NH₃ 17.027, CO 27.995, CO₂ 43.990, HCOOH 46.005, CH₃OH 32.026, C₂H₂O 42.011 (acetyl),
  C₅H₈ 68.063 (prenyl), sugars: pentose 132.042, deoxyhexose 146.058, hexose 162.053,
  glucuronic acid 176.032, malonyl-hexose 248.053; SO₃ 79.957 (sulfates), HPO₃ 79.966 (phosphates).
* **Glycosides**: glycosidic bond cleavage gives Y-type aglycone ions (Domon–Costello nomenclature);
  position of the sugar changes Y₀ abundance (e.g. flavonoid 5-O- vs 7-O-glycosides behave
  differently in negative mode **[v]**), and radical aglycone ions [Y₀−H]⁻• are diagnostic of
  O- vs C-glycosides. C-glycosides show cross-ring cleavages (⁰,²X, ⁰,³X: −120, −90 Da) instead of
  whole-sugar loss.
* **Retro-Diels–Alder (RDA)** in flavonoids and other C-ring heterocycles: ¹,³A/¹,³B ions report the
  substitution of the A- and B-rings separately **[v]** — exactly the positional information that
  separates same-formula isomers.
* **Collision energy**: survival yield of the precursor falls sigmoidally with energy; the energy at
  50 % survival scales with molecular size/degrees of freedom. Low energy reveals large fragments
  (substituents, sugars), high energy reveals cores (rings, small aromatic ions). Our data: precursor
  present in 98 %/54 %/5 % of enveda-180 [M+H]⁺ spectra at 20/40/60 eV.
* **Beam-type CID (timsTOF, QTOF) vs Orbitrap HCD vs ion-trap CID**: HCD and QTOF-CID spectra are
  broadly comparable at adjusted energies (QTOF 20–50 eV ≈ Orbitrap 30–60 NCE for optimal matching
  **[v]**); ion traps lack low-m/z fragments (1/3 cut-off) and multi-generation fragments — consistent
  with our near-zero timsTOF↔ion-trap similarity.

### A3. Mass, formula and constraints
* High-resolution precursor mass + **Seven Golden Rules** (Kind & Fiehn 2007: element ratios,
  heuristic H/C, N/C, O/C limits, LEWIS/SENIOR valence rules, isotope ratios) prune formulas.
* **RDBE** (rings + double bonds) = C − H/2 + N/2 + 1 (+ halogen/P corrections): fragments of an
  even-electron ion have half-integer RDBE; every fragment formula must be a sub-formula of the ion.
* **Mass defect**: H-rich (lipids, terpenoids) → positive defect; O/halogen-rich → smaller/negative;
  fragment mass defect constrains elemental composition independently of exact formula.
* **Isotopes**: M+1/M ≈ 1.1 % × nC; in our MS2 lists precursor isotopes up to +2 Da survive
  (forum: unbiased carbon-count estimate, ±25 %, usable for ~41 % of molecules).
* **Electron mass** (0.000549 Da): forum found Enveda's positive-mode calibration ~0.55 mDa high;
  irrelevant at 10 ppm, relevant for 0.005 Da fragment tolerances.

### A4. Natural-product chemistry as a prior
* Biosynthetic origin determines building blocks: **terpenoids** (C₅H₈ isoprene units, C₁₀/C₁₅/C₂₀/C₃₀
  skeletons, many stereocentres, few N), **polyketides** (C₂H₂O acetate units, phenolics),
  **phenylpropanoids/flavonoids** (C₆–C₃ / C₆–C₃–C₆ from shikimate), **alkaloids** (amino-acid N),
  **glycosylation**, methylation (+CH₂ = +14.016), hydroxylation (+O = +15.995), acylation.
* Hence the **"biosynthetic delta"** structure of NP space: congeners differ by +CH₂, +O, +H₂, +C₂H₂O,
  +hexose… — exactly the deltas that mass-shifted analog propagation exploits and that derivative
  enumeration (MINE, BioTransformer, MetNC) generates.
* NP-likeness (Ertl 2008) distinguishes NP from synthetic space well globally, but is useless for
  ranking within a same-formula NP isomer set (forum: worse than random).
* Classifiers: NPClassifier (pathway/superclass/class), CANOPUS (ClassyFire from fingerprints) —
  class priors per spectrum are a cheap structured feature.

---

## Part B — ML/algorithmic methods by sub-problem

### B1. Spectrum ↔ spectrum similarity (class 1 and analog search)
| method | idea | notes |
|---|---|---|
| cosine / **modified cosine** | matched peaks; modified allows a precursor-mass shift | basis of GNPS molecular networking and analog search |
| **spectral entropy similarity** (Li et al., *Nat Methods* 2021) | entropy of merged spectrum vs individual; weighted for low-entropy spectra | best classical library metric; forum class-1 0.893 → 0.93 vs cosine |
| Flash entropy search (2023) | indexed entropy search, ms-scale over millions of spectra | engineering for big libraries |
| Spec2Vec (2021) | word2vec over peaks/losses | captures structural similarity beyond shared peaks |
| MS2DeepScore (2021; 2.0 2024) | Siamese net predicting Tanimoto from spectra pairs | cross-instrument robust; good analog ranker |
| **MS2Query** (de Jonge *Nat Commun* 2023) **[v]** | random forest over Spec2Vec+MS2DeepScore+precursor features to rank analogs/exact matches | analog Tanimoto 0.85, recall 63 % for >600 Da when no exact match |
| **DreaMS** (Bushuiev *Nat Biotech* 2025) **[v]** | transformer pretrained self-supervised on GeMS (millions of unlabelled GNPS spectra; masked peaks + RT order) | SOTA on fingerprints/similarity after fine-tuning; **eligible**. Forum: lacks mass-shift invariance → worse for analog propagation |
| molecular networking / **NAP**, ConCISE **[v]** | propagate annotations over spectral networks; re-rank in-silico candidates by network consensus | the principled version of "analog propagation" |

### B2. Molecular formula
* **SIRIUS** (isotope pattern + fragmentation trees, Böcker) — gold standard; needs MS1 isotopes we do
  not have (only MS2 precursor-isotope remnants).
* **BUDDY** (*Nat Methods* 2023): bottom-up MS/MS interrogation — fragment/loss formula pairs.
* **MIST-CF** (Goldman 2023): transformer formula ranking from MS2 alone.
* Forum: formula is ~95 % solvable already; it barely narrows candidates (72 % of window candidates
  share the true formula). Formula matters mainly as a **constraint for de novo generation**.

### B3. Spectrum → structure retrieval (classes 1–2)
| method | idea | notes |
|---|---|---|
| **CSI:FingerID** (Dührkop *PNAS* 2015) + COSMIC confidence (2021) | fragmentation tree → kernel SVM fingerprint prediction → Bayes score vs candidates | the forum's `f·z` dot product is the Bayes log-likelihood shortcut |
| **MIST** (Goldman *Nat Mach Intell* 2023) | set transformer over **formula-annotated peaks** → fingerprint | MassSpecGym/CANOPUS weights released with FRIGID (**eligible**: `mist_msg`) **[v]** |
| **JESTR** (Kalia 2024) **[v]** | contrastive joint embedding of spectra and molecules | outperformed SIRIUS/CFM-ID on MassSpecGym subsets |
| **GLMR** (AAAI 2026) **[v]** | contrastive pre-retrieval → generative model conditioned on retrieved candidates → re-rank by similarity to generated structures | Recall@1 +46 % / +56 % over JESTR (mass / formula retrieval) |
| CFM-ID 4, MetFrag, MAGMa+ (rule/fragment based) | in-silico fragmentation scoring | CFM-ID 4 stock models **allowed** by host |

### B4. Molecule → spectrum (forward models) — the isomer separators
| method | idea | reported retrieval / notes |
|---|---|---|
| CFM-ID 4 (2021) | probabilistic fragmentation Markov model | classic; METLIN-trained but allowed |
| SCARF (Goldman, NeurIPS 2023) | prefix-tree formula decoding | fast, formula-level intensities |
| **ICEBERG** (Goldman 2024; Anal. Chem / bioRxiv 2025) **[v]** | autoregressive fragmentation DAG (Generate) + intensity/H-shift scoring (Score); CE- and polarity-aware | top-1 retrieval 29 % vs 20 % next best on a natural-product set **[v]** |
| **GLACIER** (Wang, Wang, Coley, Jun 2026) **[v]** | **single-stage** transformer fragment detection (DETR-like) | MassSpecGym retrieval top-1 **70.0 %** (prev. 64.0 %), NIST20 52.5 %; ~8× faster than two-stage **[v]**; MassSpecGym weights **eligible** |
| **MARASON** (ICML 2025) **[v]** | retrieval-augmented ICEBERG: retrieve reference spectra of similar molecules, **neural graph matching** of fragments to transfer intensities | top-1 identification 27 % vs 19 % non-retrieval **[v]** — directly addresses same-scaffold isomers |
| MassFormer, GrAFF-MS (Enveda, ICML 2023), FraGNNet (2024) | graph transformers / fragment-graph models | GrAFF-MS NIST-trained (ineligible weights; architecture usable) |

**Implication**: forward models are the strongest *training-free-at-test-time* isomer discriminators,
and they are trained on MassSpecGym (Orbitrap/QTOF heavy). Fine-tuning on enveda-180 timsTOF
(1.15M spectra at known 20/40/60 eV) should transfer **instrument physics** independent of chemistry.

### B5. De novo generation (class 3)
| method | idea | reported result |
|---|---|---|
| MSNovelist (2022), Spec2Mol, MassGenie | fingerprint/spectrum → SMILES LSTM/transformer | low exact-match rates |
| **MS2Mol** (Enveda, 2023) **[v]** | seq2seq transformer; **EnvedaDark** benchmark (226 dark NPs) | 21 % close-match, 62 % meaningful similarity; confidence scorer → 63 % close-match in top-10 % confident **[v]** — the hosts' own prior work |
| **MADGEN** (ICLR 2025) **[v]** | retrieve scaffold contrastively, then spectrum-attended generation from scaffold | strong with oracle scaffold; scaffold retrieval is the bottleneck |
| **DiffMS** (ICML 2025) **[v]** | formula-constrained discrete graph diffusion; decoder pretrained on fingerprint→structure pairs (unlimited data) | MassSpecGym top-1 2.3 % (strongest at the time) |
| **MS-BART** (NeurIPS 2025) **[v]** | shared vocabulary for spectra & molecules, multi-task pretraining on fingerprint–molecule pairs, chemical feedback | SOTA on 5/12 metrics; 10× faster than diffusion |
| test-time-tuned LMs (2025) **[v]** | adapt an LM per spectrum at inference | MassSpecGym top-10 5.4 %, NPLIB1 12.9 % |
| **FRIGID** (Coley, Apr 2026) **[v]** | masked diffusion LM over SAFE strings conditioned on **MIST fingerprint + formula**; **ICEBERG-guided inference-time refinement** (simulate, find hallucinated fragments, remask, re-denoise); log-linear scaling with rounds | **NPLIB1 top-1 25.0 %**, MassSpecGym >15–18 % top-1; 6.6 s/spectrum (20× faster than DiffMS) **[v]**; MassSpecGym weights released |
| **FOAM** (Coley, Feb 2026) **[v]** | formula-constrained **graph genetic algorithm** scored by ICEBERG similarity; standalone or seeded by inverse models | encounters the true molecule 68 % of the time on NIST20 **[v]** |
| MS-GPT (Jul 2026) **[v]** | spectrum-induced posterior querying of a molecule LM pretrained on molecule-only corpus with oracle fingerprint+formula; adapters on HF | candidate-pool scaling |
| FlowMS, many-body / line-graph diffusion (2026) | flow matching / improved graph diffusion | incremental |

**Benchmark honesty**: "MassSpecGym in the Wild" (ICML 2026) **[v]** found evaluation problems in
**17 of 26** papers (leakage, shortcut learning, metric bugs) → MassSpecGym v1.5. Treat headline
de novo numbers as upper bounds; validate on our own InChIKey14-disjoint NP holdout.

### B6. Candidate expansion for unseen structures
* **MINE** databases, **BioTransformer**, **MetNC** (NP-specific biotransformation; best coverage vs
  GLORYx/BioTransformer) **[v]**; rule-based derivatives of a spectrally similar known compound given
  the mass difference **[v]**.
* COCONUT 2.0 (~0.7M NPs), LOTUS, NPAtlas (microbial), PubChemLite; forum: NPAtlas/LOTUS add ~0 over
  COCONUT; PubChem only helps behind a strong gate.

### B7. Ranking, fusion and calibration
* Learning-to-rank (LambdaMART/LightGBM ranker, HistGBM classifier) over channel features; the forum's
  best stacks use reciprocal-rank fusion of engines.
* **Expected-MRR-optimal ordering**: for independent calibrated probabilities p_i that candidate i is
  the answer, E[RR] = Σ_r p_(r)/r is maximised by sorting candidates by p (rearrangement inequality).
  So merging retrieval and de novo candidates is *a calibration problem*: estimate P(answer ∈ pool)
  per molecule and P(candidate correct | source, scores), then sort. COSMIC (Hoffmann 2021) is the
  published precedent for confidence on CSI:FingerID hits; MS2Mol shows de novo confidence is learnable.

---

## Part C — Synthesis: ideas ranked for this competition

| # | idea | targets | evidence | eligibility | expected value |
|---|---|---|---|---|---|
| 1 | **De novo generation for class 3** with FRIGID-style formula+fingerprint conditioned generation and forward-model (ICEBERG/GLACIER) refinement; FOAM-style GA as a second generator seeded from analogs | class 3 (40–55 %) | 15–25 % top-1 on benchmarks; nobody on LB uses it | MassSpecGym weights OK | **highest ceiling**; even 5–10 % top-1 on class 3 ≈ +0.02–0.05 |
| 2 | **Calibrated merge** of pool + generated candidates (P(in pool) gate) | all | rearrangement inequality; forum gate AUC 0.63 only | — | prerequisite for #1 to not hurt |
| 3 | **timsTOF-fine-tuned forward model** (GLACIER/ICEBERG on enveda-180 + np-examples) for same-formula re-ranking | class 2 (+ class 3 refinement) | forward re-rank already +0.01 untuned; GLACIER 70 % MSG top-1 | own training OK | high; isomer separation is 95 % of retrieval gap |
| 4 | **Retrieval-augmented forward prediction** (MARASON) using analogs' measured timsTOF/library spectra | class 2 | 27 % vs 19 % top-1 | MIT | high, same lever as #3 |
| 5 | More **input views** (CE-conditioned; per-polarity fused per molecule; formula-annotated peaks à la MIST) | class 2 | +0.019 from second view | own | medium–high |
| 6 | MS2Query/MS2DeepScore-style learned **analog** scorer *with* mass-shift features | class 2 | analog propagation is the top isomer separator | own | medium |
| 7 | Neutral-loss / RDA / glycoside **diagnostic features** (positional isomer evidence) | class 2–3 | chemistry theory; fig. 12 | — | medium, cheap |
| 8 | Popularity prior + gated PubChem join (as public stack) | class 2 | +0.01 each | PubChem OK | known, table stakes |

**Not worth it** (consistent across literature and forum): bigger candidate databases without a
better isomer ranker; NIST-trained anything (disqualifying); NP-likeness priors; generic LLM reasoning.

---

## Sources (via web search; full texts not reachable from this environment)
* MassSpecGym — [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/c6c31413d5c53b7d1c343c1498734b0f-Paper-Datasets_and_Benchmarks_Track.pdf)
* MassSpecGym in the Wild — [arXiv 2606.19624](https://arxiv.org/pdf/2606.19624)
* FRIGID — [arXiv 2604.16648](https://arxiv.org/html/2604.16648v1), [code](https://github.com/coleygroup/FRIGID)
* FOAM — [arXiv 2602.07709](https://arxiv.org/abs/2602.07709), [code](https://github.com/coleygroup/foam)
* GLACIER — [arXiv 2606.29161](https://arxiv.org/abs/2606.29161); ms-pred [code](https://github.com/coleygroup/ms-pred)
* ICEBERG — [bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.05.28.656653.full.pdf)
* MARASON — [ICML 2025](https://proceedings.mlr.press/v267/wang25dg.html)
* DiffMS — [arXiv 2502.09571](https://arxiv.org/html/2502.09571v1)
* MADGEN — [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/369b98cf81d9ee544f64f1784049e44f-Abstract-Conference.html)
* MS-BART — [arXiv 2510.20615](https://arxiv.org/pdf/2510.20615), [code](https://github.com/OpenDFM/MS-BART)
* GLMR — [AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37132)
* MS-GPT — [arXiv 2607.23607](https://arxiv.org/pdf/2607.23607)
* Test-time tuned LMs — [OpenReview](https://openreview.net/pdf/435aacca8721f135ad31e0056a7e8d463a1db1f6.pdf)
* MS2Mol / EnvedaDark — [ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/6492507524989702c2b082fc)
* MS2Query — [Nat Commun 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10060387/)
* JESTR — [PMC11601792](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11601792/)
* DreaMS — [docs](https://dreams-docs.readthedocs.io/en/latest/), [ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/6566e510cf8b3c3cd76098b2)
* NAP — [PMC5927460](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5927460/)
* MINE/BioTransformer/MetNC — [Front. Chem. 2022](https://www.frontiersin.org/journals/chemistry/articles/10.3389/fchem.2022.881975/pdf), [MINEs](https://figshare.com/collections/MINEs_open_access_databases_of_computationally_predicted_enzyme_promiscuity_products_for_untargeted_metabolomics/3697504)
* Flavonoid RDA / glycoside fragmentation — [Molecules 2022](https://dx.doi.org/10.3390/molecules27031032), [Fabre 2001](https://www.farm.ucl.ac.be/Abstracts-FARM/Fabre-2001-1.htm)
* CID even-electron mechanisms — [CFM paper](https://arxiv.org/pdf/1312.0264), [RSC review](https://pubs.rsc.org/en/content/getauthorversionpdf/c5np00073d)
* QTOF vs Orbitrap CE matching — [PMC6359582](https://pmc.ncbi.nlm.nih.gov/articles/PMC6359582)
* CASMI 2022 results — [Fiehn lab](https://fiehnlab.ucdavis.edu/casmi/casmi-2022-results)
