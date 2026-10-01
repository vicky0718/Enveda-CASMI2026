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

## Layout

| Path | Contents |
|---|---|
| `scripts/download_data.py` | Downloads the competition files via the Kaggle REST API |
| `notebooks/` | Exploratory notebooks |
| `reports/` | Generated EDA reports and figures (aggregates only, no raw data) |
