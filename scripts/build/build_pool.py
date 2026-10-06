"""Build our own candidate pool: training structures ∪ COCONUT 2.0 (public CSV), keyed by the metric key.

Outputs (data/artifacts/pool/):
  pool.parquet  — key (metric key, stereo stripped), smiles, formula, exact_mass, src, coconut_id,
                  np_pathway; sorted by exact_mass
  fp_full.npy   — packed full fingerprints (uint8, 10,407 bits), row-aligned with pool.parquet
  fp_bits.npy   — informative bit indices selected on training structures

    PYTHONPATH=src python scripts/build/build_pool.py
"""

import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from casmi import fp as F
from casmi.metric import metric_key
from casmi.paths import INTERIM, ROOT

RDLogger.DisableLog("rdApp.*")
OUT = ROOT / "data" / "artifacts" / "pool"
OUT.mkdir(parents=True, exist_ok=True)
COCONUT = ROOT / "data" / "external" / "coconut" / "coconut_csv_lite-09-2026.csv"


def _coconut_row(smi):
    """Largest fragment, neutral, 100–1300 Da, stereo stripped -> (key, smiles, formula, mass) or None."""
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    if Chem.GetFormalCharge(mol) != 0:
        return None
    Chem.RemoveStereochemistry(mol)
    s = Chem.MolToSmiles(mol)
    m = ExactMolWt(mol)
    if not (100 <= m <= 1300) or not (5 <= mol.GetNumHeavyAtoms() <= 120):
        return None
    k = metric_key(s)
    return None if k is None else (k, s, CalcMolFormula(mol), m)


def _train_row(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return CalcMolFormula(mol), ExactMolWt(mol)


def main():
    t0 = time.time()
    keys = pd.read_parquet(INTERIM / "metric_keys.parquet")  # normalized_smiles -> metric_key (pass 5a)
    tr = keys.dropna().drop_duplicates("metric_key").rename(columns={"normalized_smiles": "smiles", "metric_key": "key"})
    with Pool(4) as p:
        fm = p.map(_train_row, tr.smiles.tolist(), chunksize=500)
    tr["formula"] = [x[0] if x else None for x in fm]
    tr["exact_mass"] = [x[1] if x else np.nan for x in fm]
    tr = tr.dropna(subset=["exact_mass"])[["key", "smiles", "formula", "exact_mass"]].assign(src="train", coconut_id=None)
    print(f"train structures: {len(tr):,} ({time.time() - t0:.0f}s)", flush=True)

    co = pd.read_csv(COCONUT, usecols=["identifier", "canonical_smiles", "np_classifier_pathway"], low_memory=False)
    with Pool(4) as p:
        res = p.map(_coconut_row, co.canonical_smiles.tolist(), chunksize=500)
    ok = [i for i, r in enumerate(res) if r is not None]
    cc = pd.DataFrame([res[i] for i in ok], columns=["key", "smiles", "formula", "exact_mass"])
    cc["coconut_id"] = co.identifier.values[ok]
    cc["np_pathway"] = co.np_classifier_pathway.values[ok]
    cc = cc.drop_duplicates("key")
    in_co = dict(zip(cc.key, cc.coconut_id))
    path_of = dict(zip(cc.key, cc.np_pathway))
    print(f"COCONUT usable: {len(cc):,} of {len(co):,} ({time.time() - t0:.0f}s)", flush=True)

    tr["coconut_id"] = tr.key.map(in_co)
    tr["np_pathway"] = tr.key.map(path_of)
    pool = pd.concat([tr, cc[~cc.key.isin(set(tr.key))].assign(src="coconut")], ignore_index=True)
    pool = pool.sort_values("exact_mass", kind="mergesort").reset_index(drop=True)

    with Pool(4) as p:
        fps = p.map(F.full_fp, pool.smiles.tolist(), chunksize=500)
    bad = [i for i, f in enumerate(fps) if f is None]
    if bad:
        pool = pool.drop(index=bad).reset_index(drop=True)
        fps = [f for f in fps if f is not None]
    fp_full = np.stack(fps)
    bits = F.select_bits(fp_full[(pool.src == "train").values])
    np.save(OUT / "fp_full.npy", fp_full)
    np.save(OUT / "fp_bits.npy", bits)
    pool.to_parquet(OUT / "pool.parquet")
    print(f"pool: {len(pool):,} structures ({(pool.src == 'train').sum():,} train, "
          f"{(pool.src == 'coconut').sum():,} COCONUT-only); {len(bits)} informative bits; "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
