"""Private Kaggle dataset with the harness features (with SMILES) and queries, for the ranker+FP kernel.

    PYTHONPATH=src python scripts/build/make_eval_ds.py [--features=features_hardpc.parquet]
        [--slug=casmi26-eval-pc] [--upload | --version "notes"]
(the chosen features file is shipped as features.parquet; each slug gets its own folder)
"""

import shutil
import sys

from casmi.paths import ROOT

EVAL = ROOT / "data" / "artifacts" / "eval"
SLUG = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--slug=")), "casmi26-eval")
FEATURES = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--features=")), "features.parquet")
OUT = ROOT / "data" / "artifacts" / ("eval_ds" if SLUG == "casmi26-eval" else f"eval_ds_{SLUG}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(EVAL / FEATURES, OUT / "features.parquet")
    shutil.copy(EVAL / "queries.parquet", OUT / "queries.parquet")
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
