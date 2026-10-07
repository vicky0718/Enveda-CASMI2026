"""Assemble the private Kaggle dataset used by the submission notebook (our code + our artifacts).

    PYTHONPATH=src python scripts/build/make_submit_ds.py [--upload | --version "notes"]
-> data/artifacts/submit_ds/  (flat: our upload client sends top-level files only; code modules are
   stored as code__<module>.py and re-assembled into a package by the notebook)
"""

import shutil
import sys

import pandas as pd

from casmi.paths import ROOT

ART = ROOT / "data" / "artifacts"
OUT = ART / "submit_ds"
MODULES = ["__init__", "simkernels", "spectra", "library", "formula", "fpmodel", "search", "pipeline", "metric",
           "frag", "edits", "fp", "rank", "edge"]
SLUG = "casmi26-artifacts"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for m in MODULES:
        shutil.copy(ROOT / "src" / "casmi" / f"{m}.py", OUT / f"code__{m}.py")
    pool = pd.read_parquet(ART / "pool" / "pool.parquet")
    pool[["key", "smiles", "exact_mass", "src"]].to_parquet(OUT / "pool.parquet")
    for f in ("fp_full.npy", "fp_bits.npy", "pop.npy", "fp_prior.npy"):
        dst = OUT / f
        if not dst.exists() or dst.stat().st_mtime < (ART / "pool" / f).stat().st_mtime:
            shutil.copy(ART / "pool" / f, dst)
    shutil.copy(ART / "library" / "library.npz", OUT / "library.npz")
    tr = pool[pool.key_is_metric]
    pd.DataFrame({"smiles": tr.smiles.values, "metric_key": tr.key.values}).to_parquet(OUT / "keycache.parquet")
    for p in (ART / "submit").glob("*"):
        shutil.copy(p, OUT / p.name)
    (OUT / "casmi26_artifacts.txt").write_text("marker for the submission notebook\n")
    print(sorted(p.name for p in OUT.iterdir() if p.is_file()))
    if "--upload" in sys.argv or "--version" in sys.argv:
        sys.path.insert(0, str(ROOT / "scripts"))
        import kaggle_api as K
        if "--upload" in sys.argv:
            print(K.dataset_create(OUT, SLUG, SLUG))
        else:
            print(K.dataset_version(OUT, SLUG, sys.argv[sys.argv.index("--version") + 1]))


if __name__ == "__main__":
    main()
