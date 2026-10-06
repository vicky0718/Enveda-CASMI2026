"""Assemble the private Kaggle dataset for FP-model training (our code + derived index arrays).

    PYTHONPATH=src python scripts/build/make_fp_dataset.py [--upload]
-> data/artifacts/fp_train_ds/{fpmodel.py, row_struct.npy, fp_targets.npy, nbits.txt, decoys.npy}

fp_targets: packed informative bits for every pool row; row_struct: pool row per train row
(-1 excluded, -2 held out); decoys: per pool row, up to 31 other pool rows in its ±10 ppm window
(held-out structures never used), -1 padded — for the contrastive (f·z softmax) loss.
"""

import shutil
import sys

import numpy as np
import pandas as pd

from casmi import fp as F
from casmi import library as L
from casmi.io import load_meta
from casmi.paths import INTERIM, ROOT

POOL = ROOT / "data" / "artifacts" / "pool"
OUT = ROOT / "data" / "artifacts" / "fp_train_ds"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pool = pd.read_parquet(POOL / "pool.parquet")
    fp_full = np.load(POOL / "fp_full.npy", mmap_mode="r")
    bits = np.load(POOL / "fp_bits.npy")
    targets = np.concatenate([np.packbits(F.subset(np.asarray(fp_full[a:a + 100000]), bits), axis=1)
                              for a in range(0, len(pool), 100000)])
    hold = set(pd.read_parquet(ROOT / "data" / "artifacts" / "holdout.parquet").key)
    mass = pool.exact_mass.values
    ok = ~pool.key.isin(hold).values
    rng = np.random.default_rng(0)
    decoys = np.full((len(pool), 31), -1, np.int32)
    lo = np.searchsorted(mass, mass - mass * 1e-5)
    hi = np.searchsorted(mass, mass + mass * 1e-5)
    for i in np.flatnonzero((pool.src == "train").values):
        c = np.arange(lo[i], hi[i])
        c = c[(c != i) & ok[c]]
        if len(c) > 31:
            c = rng.choice(c, 31, replace=False)
        decoys[i, :len(c)] = c

    meta = load_meta()
    keys = pd.read_parquet(INTERIM / "metric_keys.parquet").drop_duplicates("normalized_smiles")
    row_key = meta[["normalized_smiles"]].merge(keys, on="normalized_smiles", how="left").metric_key.values
    hashes = pd.read_parquet(INTERIM / "spectrum_hashes.parquet").hash.values
    conflict = L.conflicting_rows(hashes, row_key)
    idx = pd.Series(np.arange(len(pool)), index=pool.key.values)
    rs = idx.reindex(row_key).values
    rs = np.where(np.isnan(rs), -1, rs).astype(np.int32)
    rs[pd.Series(row_key).isin(hold).values] = -2
    rs[conflict] = -1

    np.save(OUT / "row_struct.npy", rs)
    np.save(OUT / "fp_targets.npy", targets)
    np.save(OUT / "decoys.npy", decoys)
    (OUT / "nbits.txt").write_text(str(len(bits)))
    shutil.copy(ROOT / "src" / "casmi" / "fpmodel.py", OUT / "fpmodel.py")
    print(f"pool rows {len(pool):,}, mean decoys {(decoys[(pool.src == 'train').values] >= 0).sum(1).mean():.1f}, nbits {len(bits)}, rows train {(rs >= 0).sum():,}, "
          f"holdout {(rs == -2).sum():,}, excluded {(rs == -1).sum():,}")
    if "--upload" in sys.argv:
        sys.path.insert(0, str(ROOT / "scripts"))
        import kaggle_api as K
        print(K.dataset_create(OUT, "casmi26-fp-train", "casmi26-fp-train"))


if __name__ == "__main__":
    main()
