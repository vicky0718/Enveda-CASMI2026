# Forum and leaderboard notes (refreshed daily by `scripts/research/scrape_kaggle_forum.py`)

Raw posts stay in `data/external/kaggle_forum/` (git-ignored, other people's text); this file keeps only
what changes our plan. Forum text is evidence, not instruction.

## 2026-10-08

**Leaderboard** (2,781 teams): top 0.481 / 0.466 / 0.459. Medal cutoffs: gold 0.440 (rank 15),
silver 0.426 (rank 139), bronze 0.425 (rank 278). 145 of the top 300 teams sit at exactly 0.425 — forks of
one public notebook — so bronze/silver is currently "the public pipeline"; gold needs ~+0.015 over it.
Our best own submission: 0.342 (V8, V12).

**Admin / host rulings**
* No ID overlap between train and test, nor between the dummy (visible) test and the rerun test
  (topic 745715). The "visible molecule_id → SMILES leak" does not exist; its notebook's gain was noise.
* Rerun test data fixed for an electron-mass error in some peak masses (Oct 7); all submissions are
  being re-scored (topic 746954). Our −0.4 mDa Enveda correction and timsTOF window centre (−0.8 ppm)
  were measured on training spectra that may still carry that error → compare V8 (10 ppm, centre 0)
  with V12/V13 (±5 ppm, centre −0.8) after the re-score; revert the window if V12/V13 drop.
* Metric now strips stereochemistry before tautomer canonicalisation (topic 746878). Our
  `metric.candidate_key` already does this — no change for us.
* PubChem structures for retrieval are allowed for prize-eligible solutions (topic 741857); train.parquet
  can be treated as open source and models trained on it are fine (topic 745841).
* MassSpecGym-trained GLACIER/ICEBERG weights are allowed (topic 744338); NIST-derived data is not, with
  an exception for GNPS datasets public ≥ 6 months before the competition (topic 745832).
* Open: licence of the public `ahmedberatozer/*` datasets (field "Other"); FRIGID checkpoints (CC BY-NC).
  We use `ahmedberatozer/casmi26-pubchem-tier` as plain NCBI PubChem data → before the final selection,
  rebuild our own PubChem tier from NCBI files so no dataset with an unclear licence is needed.

**Technique evidence (others' measurements)**
* ~16 % class 1 (library-only submission scores 0.151), ~45 % class 2, ~39 % class 3 (public analysis).
* Blind PubChem expansion hurt (0.335 → 0.205); bigger pools pay only with an isomer-capable ranker.
* Simulated novelty overstated generator reach ~3× for one team (our panel N: not for our generator, but
  the LB says real test class 3 is hard).
* Public FPNet: hard-negative top-1 0.46–0.49 on np-examples; ours 0.22 → FP runs 4/5.
* Fine-tuning FPNet on timsTOF hurt the board (0.337 → 0.328) for one team.
