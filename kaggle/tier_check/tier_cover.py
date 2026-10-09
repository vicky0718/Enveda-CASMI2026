"""Which PubChem popularity ordering ranks the truth higher? Panels A / C (truths taken out of nothing — this
only looks at the PubChem window): rank of the truth's InChIKey first block in its ±5 ppm window ordered by
  old   third-party tier + popularity arrays
  new   our NCBI tier, per-CID popularity (log1p SIDs + log1p PMIDs)
  agg   our tier, popularity summed over all CIDs with the same InChIKey first block (stereo variants)
Reported: share of truths ranked within the top 10 / 100 / 300."""

import glob
import os
import shutil
import subprocess
import sys

import numpy as np

whl = [w for w in glob.glob("/kaggle/input/**/rdkit*.whl", recursive=True)
       if f"-cp{sys.version_info.major}{sys.version_info.minor}-" in w][:1]
if whl:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", *whl], check=False)
ART = os.path.dirname(glob.glob("/kaggle/input/**/casmi26_artifacts.txt", recursive=True)[0])
EV = os.path.dirname(glob.glob("/kaggle/input/**/casmi26_eval.txt", recursive=True)[0])
os.makedirs("/kaggle/working/code/casmi", exist_ok=True)
for f in glob.glob(f"{ART}/code__*.py"):
    shutil.copy(f, "/kaggle/working/code/casmi/" + os.path.basename(f)[len("code__"):])
sys.path.insert(0, "/kaggle/working/code")

import pandas as pd  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

from casmi.pubchem import PubChemTier  # noqa: E402

RDLogger.DisableLog("rdApp.*")
mass_dirs = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/pc_mass.npy", recursive=True)]
pop_dirs = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/pc_lsid.npy", recursive=True)]
new_dir = next(d for d in mass_dirs if "casmi26-pubchem-tier-ncbi" in d)
old_dir = next(d for d in mass_dirs if d != new_dir)
old_pop = next(d for d in pop_dirs if d != new_dir)
NEW, OLD = PubChemTier(new_dir), PubChemTier(old_dir, old_pop)


def ik(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToInchiKey(m)[:14] if m is not None else None


def rank_of(T, mass, truth, agg=False, limit=1000):
    rows = T.window(mass, 5.0)
    pop = T.pop(rows).astype(np.float64)
    if agg:  # sum exp(log1p counts) - 1 over rows sharing the InChIKey first block, then log1p again
        keys = np.array([ik(T.smiles(int(r))) for r in rows], dtype=object)
        s = pd.Series(np.expm1(pop)).groupby(keys).transform("sum").values
        pop = np.log1p(s)
    order = np.argsort(-pop, kind="stable")
    seen = []
    for j in order[:limit * 3]:
        k = ik(T.smiles(int(rows[j]))) if not agg else None
        k = k if k is not None else (keys[j] if agg else None)
        if k is None or k in seen:
            continue
        seen.append(k)
        if k == truth:
            return len(seen)
        if len(seen) >= limit:
            break
    return 10 ** 6


f = pd.read_parquet(f"{EV}/features.parquet", columns=["qkey", "panel"]).drop_duplicates("qkey")
pool = pd.read_parquet(f"{ART}/pool.parquet", columns=["key", "smiles", "exact_mass"]).drop_duplicates("key")
m = f.merge(pool, left_on="qkey", right_on="key")
m = m[m.panel.isin(["A", "C"])].sample(frac=1, random_state=0).groupby("panel").head(300)
rows = []
for i, r in enumerate(m.itertuples()):
    t = ik(r.smiles)
    rows.append({"panel": r.panel, "old": rank_of(OLD, r.exact_mass, t), "new": rank_of(NEW, r.exact_mass, t),
                 "agg": rank_of(NEW, r.exact_mass, t, agg=True)})
    if i % 100 == 0:
        print(i, flush=True)
d = pd.DataFrame(rows)
for p, g in d.groupby("panel"):
    print(p, {v: {k: round((g[v] <= k).mean(), 3) for k in (10, 100, 300)} for v in ("old", "new", "agg")}, flush=True)
