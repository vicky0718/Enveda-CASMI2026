"""Build our own candidate pool: training structures ∪ COCONUT 2.0 (public CSV), keyed by the metric key.

Outputs (data/artifacts/pool/):
  pool.parquet  — key (metric key for train structures; standard InChIKey14 for COCONUT-only rows,
                  see key_is_metric), smiles, formula, exact_mass, src, coconut_id, np_pathway,
                  coconut_np_likeness; sorted by exact_mass
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
from casmi.paths import INTERIM, ROOT

RDLogger.DisableLog("rdApp.*")
OUT = ROOT / "data" / "artifacts" / "pool"
OUT.mkdir(parents=True, exist_ok=True)
COCONUT = ROOT / "data" / "external" / "coconut" / "coconut_csv_lite-09-2026.csv"


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

    # COCONUT: the CSV already carries InChIKey, formula and exact mass. Computing the metric key
    # (tautomer canonicalisation) costs ~60 ms per natural product, so COCONUT-only rows are keyed by
    # the standard InChIKey14 (key_is_metric=False); metric keys are computed lazily for the few
    # candidates that reach a submitted top-25 (dedup at output).
    co = pd.read_csv(COCONUT, usecols=["identifier", "canonical_smiles", "standard_inchi_key", "molecular_formula",
                                       "exact_molecular_weight", "formal_charge", "heavy_atom_count",
                                       "np_likeness", "np_classifier_pathway"], low_memory=False)
    co = co[co.standard_inchi_key.notna() & co.canonical_smiles.notna()
            & ~co.canonical_smiles.str.contains(".", regex=False) & (co.formal_charge == 0)
            & co.exact_molecular_weight.between(100, 1300) & co.heavy_atom_count.between(5, 120)]
    cc = pd.DataFrame({"key": co.standard_inchi_key.str[:14].values, "smiles": co.canonical_smiles.values,
                       "formula": co.molecular_formula.values, "exact_mass": co.exact_molecular_weight.values,
                       "coconut_id": co.identifier.values, "np_pathway": co.np_classifier_pathway.values,
                       "coconut_np_likeness": co.np_likeness.values}).drop_duplicates("key")
    in_co = dict(zip(cc.key, cc.coconut_id))
    path_of = dict(zip(cc.key, cc.np_pathway))
    print(f"COCONUT usable: {len(cc):,} of {len(co):,} ({time.time() - t0:.0f}s)", flush=True)

    # a train structure is "in COCONUT" if either its metric key or its shipped InChIKey14 matches
    ik_of = keys.dropna().drop_duplicates("metric_key").set_index("metric_key").inchikey14
    tr_ik = tr.key.map(ik_of).fillna(tr.key)
    tr["coconut_id"] = tr.key.map(in_co).fillna(tr_ik.map(in_co))
    tr["np_pathway"] = tr.key.map(path_of).fillna(tr_ik.map(path_of))
    tr["key_is_metric"] = True
    seen = set(tr.key) | set(keys.inchikey14.dropna())
    cc = cc[~cc.key.isin(seen)].assign(src="coconut", key_is_metric=False)
    pool = pd.concat([tr, cc], ignore_index=True)
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
