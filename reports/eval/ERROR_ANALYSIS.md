# Error analysis — what outranks the truth (validation panels, out-of-fold ranker)

Script: `scripts/eval/error_analysis.py` (per-case table `data/artifacts/eval/errors.parquet`,
worst cases with SMILES `reports/eval/error_cases.csv`). Panels: A = 250 np-examples (timsTOF),
C = 909 public-library natural products. Regimes: C1 library spectra available, C2 structure in pool
without spectra, C3 structure in no database.

## 1. Confusion-style table: what is ranked first (share of molecules)

| panel / regime | truth | same-formula isomer **with** library spectra | same-formula COCONUT isomer | generated structure | other formula | truth not in list |
|---|---|---|---|---|---|---|
| A C1 | 0.864 | 0.068 | 0.048 | 0.016 | 0.004 | 0 |
| A C2 | 0.624 | 0.156 | 0.152 | 0.060 | 0.008 | 0 |
| A C3 | 0.400 | — | — | 0.116 | 0.484 | 0.412 |
| C C1 | 0.586 | 0.097 | 0.146 | 0.057 | 0.114 | 0.103 |
| C C2 | 0.449 | 0.129 | 0.224 | 0.079 | 0.119 | 0.103 |
| C C3 | 0.197 | — | — | 0.141 | 0.662 | 0.601 |

**The dominant error is same-formula isomer confusion (31–35 % of C2 molecules).** Generator dilution is
secondary on validation (6–8 %). In C3, 41–60 % of truths are not generated at all.

## 2. Why the isomer wins (truth in list but beaten; mean feature, truth vs winner)

| feature | A truth | A winner | P(winner > truth) | C truth | C winner | P(winner > truth) |
|---|---|---|---|---|---|---|
| analog (sim³·Tanimoto) | 0.246 | 0.304 | **0.81** | 0.257 | 0.309 | **0.76** |
| tmax (best Tanimoto to a hit) | 0.691 | 0.797 | 0.70 | 0.632 | 0.757 | 0.69 |
| direct (own spectrum) | 0.129 | 0.297 | 0.37 | 0.049 | 0.154 | 0.18 |
| frag (MetFrag-lite) | 0.358 | 0.367 | **0.41** | 0.288 | 0.311 | **0.42** |

Analog propagation rewards "looks like the library neighbours": an isomer that sits between the
neighbours, or that *is* a neighbour, outscores the truth. Fragmentation is the only signal on the truth's
side, and it is weak.

## 3. Manual review (panel A, C2, truth ranked 2nd)

* hesperetin (truth) vs homoeriodictyol (winner): OMe/OH swap on the B ring; the winner has its own
  library spectra (Tanimoto 1.00 to a hit) that match the query only moderately.
* tetrazole N1 vs N2 isomer of a macrolide; 3′,5′- vs 2′,3′-cyclic AMP; glycerol 1- vs 2-monoacylglyceride;
  regio-isomeric protoberberines; Cl-phenyl amino-acid positional isomers.
* Pattern: many winners are **library compounds whose own spectrum matches only moderately**, yet that
  spectrum feeds their own analog score — "has spectra that do not match" was indistinguishable from
  "has no spectra" (both direct = 0).

## 4. Features engineered from this

| feature | idea | effect (out of fold) |
|---|---|---|
| `own_n`, `own_sim`, `own_neg` | count of the candidate's own library spectra for the query's adducts and exact similarity to them (not limited to the top-600 hits): spectra that exist but do not match are evidence *against* | C C2 0.550 → 0.560; isomer-with-spectra wins 115 → 92 (C), 39 → 34 (A) |
| `analog_noself` | analog evidence from other structures only | (included above) |
| popularity prior (PubChem substances + PubMed + patents; public `pool_popularity.csv`) | well-documented compounds are likelier answers | as a ranker feature it is **leaky** (A C2 0.73 → 0.94 because the host's examples are famous; B C2 0.76 → 0.68); used instead as a tie-breaker λ = 0.1 tuned on B/C only: C weighted +0.007, B −0.005 |

Remaining dominant error: COCONUT isomers with higher analog similarity than the truth (C C2: 215 cases).
Next candidates: isomer separation by fragment localisation (which fragments the mass-shifted analog peaks
map to), and a working spectrum→fingerprint model (pure-BCE run 3 pending).
