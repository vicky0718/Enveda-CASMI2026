"""Private Kaggle dataset with the harness features (with SMILES) and queries, for the ranker+FP kernel.

    PYTHONPATH=src python scripts/build/make_eval_ds.py [--upload | --version "notes"]
"""

import shutil
import sys

from casmi.paths import ROOT

EVAL = ROOT / "data" / "artifacts" / "eval"
OUT = ROOT / "data" / "artifacts" / "eval_ds"
SLUG = "casmi26-eval"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("features.parquet", "queries.parquet"):
        shutil.copy(EVAL / f, OUT / f)
    (OUT / "casmi26_eval.txt").write_text("marker for the ranker+FP kernel\n")
    if "--upload" in sys.argv or "--version" in sys.argv:
        sys.path.insert(0, str(ROOT / "scripts"))
        import kaggle_api as K
        if "--upload" in sys.argv:
            print(K.dataset_create(OUT, SLUG, SLUG))
        else:
            print(K.dataset_version(OUT, SLUG, sys.argv[sys.argv.index("--version") + 1]))


if __name__ == "__main__":
    main()
