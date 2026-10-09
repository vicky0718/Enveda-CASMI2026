"""Compare our NCBI-built PubChem tier with the third-party one on the test queries: for each molecule, the
top-100 PubChem candidates (by popularity, not in our pool, timsTOF mass window) from both tiers, as InChIKey
first blocks — Jaccard of the sets, overlap of the top-10, same first candidate, and window sizes."""

import glob
import os
import shutil
import subprocess
import sys


whl = [w for w in glob.glob("/kaggle/input/**/rdkit*.whl", recursive=True)
       if f"-cp{sys.version_info.major}{sys.version_info.minor}-" in w][:1]
if whl:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", *whl], check=False)
ART = os.path.dirname(glob.glob("/kaggle/input/**/casmi26_artifacts.txt", recursive=True)[0])
os.makedirs("/kaggle/working/code/casmi", exist_ok=True)
for f in glob.glob(f"{ART}/code__*.py"):
    shutil.copy(f, "/kaggle/working/code/casmi/" + os.path.basename(f)[len("code__"):])
sys.path.insert(0, "/kaggle/working/code")

import pandas as pd  # noqa: E402

from casmi import pipeline as P  # noqa: E402
from casmi.pubchem import PubChemTier, pubchem_candidates  # noqa: E402
from casmi.search import Pool  # noqa: E402

mass_dirs = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/pc_mass.npy", recursive=True)]
pop_dirs = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/pc_lsid.npy", recursive=True)]
new_dir = next(d for d in mass_dirs if "casmi26-pubchem-tier-ncbi" in d)
old_dir = next(d for d in mass_dirs if d != new_dir)
old_pop = next(d for d in pop_dirs if d != new_dir)
print("new", new_dir, "| old", old_dir, old_pop, flush=True)
new, old = PubChemTier(new_dir), PubChemTier(old_dir, old_pop)
print(f"structures new {new.n:,} old {old.n:,}", flush=True)

pool = Pool(ART)
COMP = os.path.dirname(glob.glob("/kaggle/input/**/test.parquet", recursive=True)[0])
qs = P.queries_from_test(pd.read_parquet(f"{COMP}/test.parquet"))
rows = []
for i, q in enumerate(qs):
    cand = pool.window(q.neutral_mass, ppm=5.0, center_ppm=-0.8)
    keys = set(pool.key[cand])
    a = pubchem_candidates(q, new, keys, top_n=100, ppm=5.0, center_ppm=-0.8)
    b = pubchem_candidates(q, old, keys, top_n=100, ppm=5.0, center_ppm=-0.8)
    ka, kb = list(a.key), list(b.key)
    sa, sb = set(ka), set(kb)
    rows.append({"jaccard": len(sa & sb) / max(len(sa | sb), 1), "top10": len(set(ka[:10]) & set(kb[:10])) / 10,
                 "same_first": bool(ka and kb and ka[0] == kb[0]), "n_new": len(ka), "n_old": len(kb),
                 "win_new": len(new.window(q.neutral_mass, 5.0, center_ppm=-0.8)),
                 "win_old": len(old.window(q.neutral_mass, 5.0, center_ppm=-0.8))})
    if i % 100 == 0:
        print(i, flush=True)
r = pd.DataFrame(rows)
print(r.describe().round(3).to_string())
print("same first candidate:", round(r.same_first.mean(), 3), "| median Jaccard:", round(r.jaccard.median(), 3))
