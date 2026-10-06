"""Build the cleaned reference library -> data/artifacts/library/library.npz

    PYTHONPATH=src python scripts/build/build_library.py
"""

import time

import numpy as np
import pandas as pd

from casmi import library as L
from casmi.io import load_meta
from casmi.paths import INTERIM, ROOT, TRAIN

OUT = ROOT / "data" / "artifacts" / "library"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    meta = load_meta()
    keys = pd.read_parquet(INTERIM / "metric_keys.parquet").drop_duplicates("normalized_smiles")
    row_key = meta[["normalized_smiles"]].merge(keys, on="normalized_smiles", how="left").metric_key.values
    hashes = pd.read_parquet(INTERIM / "spectrum_hashes.parquet").hash.values
    drop = L.conflicting_rows(hashes, row_key)
    print(f"meta {len(meta):,}; no key {pd.isna(row_key).sum():,}; conflicting-duplicate rows {drop.sum():,}",
          flush=True)
    lib = L.build(TRAIN, meta, row_key, drop, log=lambda s: print(s, flush=True))
    np.savez(OUT / "library.npz", **lib)
    n = len(lib["row"])
    print(f"library: {n:,} spectra, {len(lib['mz']):,} peaks, {len(np.unique(lib['key_code'])):,} structures; "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
