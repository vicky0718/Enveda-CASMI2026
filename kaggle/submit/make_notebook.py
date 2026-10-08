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
tag = f"cp{sys.version_info.major}{sys.version_info.minor}"  # one wheel per Python version is shipped
whl = [w for w in glob.glob("/kaggle/input/**/rdkit*.whl", recursive=True) if f"-{tag}-" in w][:1]
print("wheels", whl)
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
    r'''# models: a dedicated model dataset (casmi26_models.txt) if attached, else our ranker+FP kernel output
# (fpnet.pt + rankers trained with f·z), else the artifacts dataset's own rankers
mdl = glob.glob("/kaggle/input/**/casmi26_models.txt", recursive=True)
fpn = glob.glob("/kaggle/input/**/fpnet.pt", recursive=True)
MODEL = os.path.dirname(mdl[0]) if mdl else (os.path.dirname(fpn[0]) if fpn else ART)
print("model dir", MODEL)
from casmi import moe
ranker = moe.load(MODEL) or P.load_ranker(MODEL)  # multi-model ranker if present
fpm = P.load_fp_models(MODEL)
if fpm is not None:
    fpm["bits"] = np.load(f"{ART}/fp_bits.npy")
    fpm["prior"] = np.load(f"{ART}/fp_prior.npy")  # for the prior-normalised f·z feature
print("artifact files:", sorted(os.listdir(ART)))
from casmi.edge import np_like
print("NP-likeness check (flavone):", np_like("O=C1C=C(c2ccc(O)cc2)Oc2cc(O)cc(O)c21"), "| pool scores:", pool.np_like is not None)
print("gen K:", getattr(P, "GEN_K_REFS", None), "| ranker:", "loaded" if ranker is not None else "NONE (heuristic)",
      "| fp models:", 0 if fpm is None else len(fpm["nets"]))
USE_GEN = __USE_GEN__
print("generator:", USE_GEN)
# PubChem candidate channel (NCBI PubChem tier + row-aligned popularity), if both datasets are attached
pcm = glob.glob("/kaggle/input/**/pc_mass.npy", recursive=True)
pcp = glob.glob("/kaggle/input/**/pc_lsid.npy", recursive=True)
PC_N = __PC_N__
tier = None
if pcm and pcp and PC_N > 0:
    from casmi.pubchem import PubChemTier
    tier = PubChemTier(os.path.dirname(pcm[0]), os.path.dirname(pcp[0]))
print("pubchem:", None if tier is None else f"{tier.n:,} structures, top {PC_N}")
MASS_WINDOW = moe.CONFIG["mass_window"]["timsTOF"] if __TIMS_WINDOW__ else None  # test is 100 % timsTOF
print("mass window:", MASS_WINDOW)
DIRECT_ONLY = __DIRECT_ONLY__  # diagnostic probe: only candidates with a library spectrum (class 1 alone)
print("direct only:", DIRECT_ONLY)
rows = P.run(test, pool, lib, ranker=ranker, fp_models=fpm, use_gen=USE_GEN, pubchem=tier, pc_top_n=PC_N,
             mass_window=MASS_WINDOW, direct_only=DIRECT_ONLY)
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
n = sub.smiles.str.split(";").str.len()
print(sub.shape, "candidates/molecule min/median", n.min(), n.median(), f"{time.time() - T0:.0f}s"); sub.head()''',
]


def main():
    import sys
    use_gen = "--nogen" not in sys.argv
    pc_n = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--pc=")), 0)
    tims = "--tims-window" in sys.argv
    direct = "--direct-only" in sys.argv
    cells = [c.replace("__USE_GEN__", str(use_gen)).replace("__PC_N__", str(pc_n)).replace("__TIMS_WINDOW__", str(tims))
             .replace("__DIRECT_ONLY__", str(direct)) for c in CELLS]
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": c.strip("\n").splitlines(keepends=True)} for c in cells],
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    name = "casmi26_submit.ipynb" if use_gen else "casmi26_submit_nogen.ipynb"
    if pc_n:
        name = f"casmi26_submit_pc{pc_n}.ipynb"
    if "--moe" in sys.argv:
        name = f"casmi26_submit_moe_pc{pc_n}.ipynb" if pc_n else "casmi26_submit_moe.ipynb"
    if direct:
        name = name.replace(".ipynb", "_direct.ipynb")
    p = Path(__file__).with_name(name)
    p.write_text(json.dumps(nb, indent=1))
    print(p)


if __name__ == "__main__":
    main()
