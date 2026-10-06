from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
TRAIN = RAW / "train.parquet"
TEST = RAW / "test.parquet"
SAMPLE_SUB = RAW / "sample_submission.csv"

REPORT = ROOT / "reports" / "eda"
FIG = REPORT / "figures"
STATS = REPORT / "stats"

for d in (INTERIM, FIG, STATS):
    d.mkdir(parents=True, exist_ok=True)
