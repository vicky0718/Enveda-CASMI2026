"""Generate the offline submission notebook (our code only) -> kaggle/submit/casmi26_submit.ipynb

Inputs on Kaggle: the competition data (incl. the host's rdkit 2026.3.3 wheel dataset attached to the
competition) and our private dataset vigneshnehru/casmi26-artifacts (code/, pool/, library.npz,
fpnet*.pt, ranker*.txt).
"""

import json
from pathlib import Path

CELLS = [
    r'''# CASMI 2026 — own pipeline: library + analog propagation + FP model + ranker (offline)
import glob, os, shutil, subprocess, sys, time
T0 = time.time()
whl = glob.glob("/kaggle/input/**/rdkit*.whl", recursive=True)
if whl:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", *whl], check=False)
import rdkit; print("rdkit", rdkit.__version__)
ART = os.path.dirname(glob.glob("/kaggle/input/**/casmi26_artifacts.txt", recursive=True)[0])
os.makedirs("/kaggle/working/code/casmi", exist_ok=True)  # writable copy: numba caches next to the code
for f in glob.glob(f"{ART}/code__*.py"):
    shutil.copy(f, "/kaggle/working/code/casmi/" + os.path.basename(f)[len("code__"):])
sys.path.insert(0, "/kaggle/working/code")
COMP = os.path.dirname(glob.glob("/kaggle/input/**/test.parquet", recursive=True)[0])
print("artifacts", ART, "competition", COMP)''',
    r'''import numpy as np, pandas as pd
from casmi import library as L
from casmi import pipeline as P
from casmi.search import Library, Pool
pool = Pool(ART)
lib = Library(L.load(f"{ART}/library.npz"), pool)
test = pd.read_parquet(f"{COMP}/test.parquet")
print(len(test), "spectra", test.molecule_id.nunique(), "molecules", f"{time.time() - T0:.0f}s")''',
    r'''ranker = P.load_ranker(ART)
fpm = P.load_fp_models(ART)
rows = P.run(test, pool, lib, ranker=ranker, fp_models=fpm)
print(f"ranked {len(rows)} molecules, {time.time() - T0:.0f}s")''',
    r'''from casmi.metric import candidate_key
sub_ids = pd.read_csv(f"{COMP}/sample_submission.csv").molecule_id
keycache = P.load_keycache(ART)
out = {}
for mid, smiles in rows:
    picked, seen = [], set()
    for s in smiles:
        k = keycache.get(s) or candidate_key(s)
        if k is None or k in seen:
            continue
        seen.add(k); picked.append(s)
        if len(picked) == 25:
            break
    out[mid] = picked
sub = pd.DataFrame({"molecule_id": sub_ids,
                    "smiles": [";".join(out.get(m) or ["C"]) for m in sub_ids]})
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(sub.shape, f"{time.time() - T0:.0f}s"); sub.head()''',
]


def main():
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": c.strip("\n").splitlines(keepends=True)} for c in CELLS],
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    p = Path(__file__).with_name("casmi26_submit.ipynb")
    p.write_text(json.dumps(nb, indent=1))
    print(p)


if __name__ == "__main__":
    main()
