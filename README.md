# Enveda CASMI 2026 — Molecule ID From Mass Spectra

Kaggle competition: <https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra>

Predict the 2D structure (SMILES) of small molecules from their MS/MS spectra.
For each `molecule_id`, submit up to 25 ranked candidates. Scoring is MRR@25 on
the InChIKey first block, so stereochemistry and tautomers don't matter.
Submissions are made through Kaggle notebooks (≤9 h, no internet).

## Setup

```bash
pip install -r requirements.txt
# Credentials: KAGGLE_USERNAME / KAGGLE_KEY env vars, or ~/.kaggle/kaggle.json
# Accept the competition rules on kaggle.com first, then:
python scripts/download_data.py
```

The data lands in `data/raw/`, which is git-ignored. The competition rules
forbid redistributing it.

## EDA

A plain-language PDF guide (competition goal + findings) is
[`reports/CASMI2026_Competition_and_EDA_Guide.pdf`](reports/CASMI2026_Competition_and_EDA_Guide.pdf)
(rebuild: `PYTHONPATH=src python scripts/make_guide_pdf.py`). The technical report is
[`reports/eda/EDA_REPORT.md`](reports/eda/EDA_REPORT.md). Regenerate it with:

```bash
PYTHONPATH=src python scripts/eda/01_metadata.py        # ~30 s
PYTHONPATH=src python scripts/eda/02_peaks.py           # ~15 min first run (3 workers, ~10 GB RAM), cached after
PYTHONPATH=src python scripts/eda/03_chemistry.py       # ~10 min (RDKit, 4 workers), descriptors cached
PYTHONPATH=src python scripts/eda/04_library_search.py  # ~5 min, needs the 03 descriptor cache
PYTHONPATH=src python scripts/eda/05a_metric_keys.py    # ~15 min, metric keys for all structures
PYTHONPATH=src python scripts/eda/05b_deep_dive.py      # ~10 min, needs 05a
PYTHONPATH=src python scripts/eda/05c_candidate_pool.py # needs COCONUT CSV in data/external/coconut/
```

Scoring uses `rdkit==2026.3.3` exactly; `src/casmi/metric.py` replicates the official scorer.

Intermediate caches go to `data/interim/` (git-ignored).

## Layout

| Path | Contents |
|---|---|
| `scripts/download_data.py` | Downloads the competition files via the Kaggle REST API |
| `scripts/eda/` | The four EDA passes (metadata, peaks, chemistry, library-search baseline) |
| `src/casmi/` | Shared code: streaming parquet IO, peak features, spectrum cleaning and cosine, plot style |
| `reports/eda/` | EDA report, figures and stats tables (aggregates only, no raw data) |
| `reports/research/` | Kaggle forum & notebook intelligence, literature review, strategy to gold |
| `scripts/research/` | Read-only scraper for the competition forum and leaderboard |
