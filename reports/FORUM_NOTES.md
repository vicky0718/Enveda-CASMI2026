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

## 2026-10-09

**Leaderboard** (2,849 teams): top 0.484 / 0.481 / 0.459. Cutoffs: gold 0.442 (rank 15), silver 0.432 (rank 142),
bronze 0.425 (rank 284) — the 0.433 cluster grew (66 teams), i.e. a newer public pipeline. Ours: V13 0.372 best;
V14 0.353, V15 0.359; V16a/b (C2X-trained ranker) pending.

**Forum (7 new posts):** nothing that changes data, metric or rules. Open licensing questions on the
`ahmedberatozer/*` datasets (incl. `casmi26-pubchem-tier`, which we use as plain PubChem data) are still
unanswered → task: rebuild our own PubChem tier from NCBI before the final selection. The current leader
(rank 1) estimates a realistic maximum of ~0.55.

## 2026-10-10

**Leaderboard** (2,950 teams): top 0.484 (MarvinTMB) / 0.481 / 0.461. Cutoffs: gold 0.443 (rank 15); silver and bronze
both at 0.433 — 153 teams sit at exactly 0.433 (the current public pipeline), so any medal needs > 0.433. Ours: 0.387
(V21a, popularity weight 0.3), rank ~1,049.

**Host rulings (David Healey)**
* train.parquet is open source (the non-commercial clause covers the test data); datasets that are non-commercial
  only by inheriting from train.parquet are fine; PubChem / COCONUT and "most external datasets prominently published
  as open source" are fine. The concern is proprietary libraries (NIST, instrument vendors). Non-compliant solutions
  can be disqualified; compliance-check scope to be announced.
* **GNPS-NIST14-MATCHES is allowed** (library matching and training) because it has been in the open GNPS repository
  for years; newly created NIST-matched derivatives are not.

**Other:** the leader (0.484) says no NIST or extra data is needed for a better score; molecule IDs of the visible test
may be remapped before the end (no effect on us — we use no ID information).

## 2026-10-11

**Leaderboard** (3,020 teams): top 0.484 / 0.465 (rank 3) / 0.455 (rank 10). Medal cutoffs: gold ≥ 0.446 (16 teams),
silver ≥ 0.438, bronze ≥ 0.434 (rank ~300); 0.425 is rank ~500 and 0.401 rank ~1,000. Ours: V21a 0.387, rank ~1,110.
Oct 11 submissions so far: V22 0.385, V24 0.372; V23 / V25 / V26 (GLACIER notebooks) still scoring.

**Forum (5 new posts)**
* **Filler guesses may lower the score (topic 748029, open).** A team reports that the same rank-1 guesses score
  0.426 with ranks 2–25 empty or as originally ranked, but 0.388–0.397 with three other sets of valid ranks 2–25
  (RDKit-parseable, neutral, single-fragment). Under the published metric (v14, which our `metric.py` replicates;
  re-checked today, identical code) guesses after rank 1 can only add. Related: thread 742088 (padded submission
  scored 0.000); the hosts say the live scorer equals the published one and are "investigating". **Action:** LB
  probe `casmi26-own-submit-top1` (V21a pipeline, rank 1 only). Published metric ⇒ it must score *below* 0.387
  (ranks 2–25 add ~0.05–0.08 MRR in our validation). If it scores ≥ 0.387, our ranks 2–25 are poisoning rows and the
  candidate filler needs filtering (find which guesses trigger it).
* **Public 0.43+ lineage is not CV-validated (topic 748068).** Components were accepted on a ~130-molecule public LB
  (one molecule ≈ 0.008). ICEBERG was "+0.013" in one version and dropped in a later one ("0.402 with vs 0.417
  without"). This matches our policy: a component must win on held-out validation (panels A / C and the
  not-in-MassSpecGym slice), and the LB is a noisy check (SE ≈ 0.016). GLACIER's and the forward model's LB effects
  will be noisy too, so judge them on paired comparisons and validation together.
* **Eligibility questions, unanswered:**
  * Third-party weights without published training code; non-commercial labels; compliance-check scope
    (topic 748065).
  * **Whether PubMed-link counts may be used as a ranking feature (topic 741857 follow-up).** Our popularity prior
    is z(log1p substances + log1p PubMed). If the hosts rule PubMed counts out, we need a substances-only prior.
    Keep that variant ready and measure it on validation.
  * Our own stack is unaffected otherwise: our own models, official MassSpecGym GLACIER weights (host-approved),
    and an NCBI PubChem tier.

**Approach updates from today's scrape**
1. Filler-poisoning probe (above): one of tomorrow's slots, ready today.
2. Popularity-prior hedge: a validation run with a substances-only popularity term (no PubMed), so a ruling against
   PubMed counts costs no time.
3. Keep validation first: V26/V27 (forward model) selection follows the validation table; the LB confirms or vetoes
   only beyond ~2 SE.
