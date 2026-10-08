"""Add NP-likeness (np_like) to a harness features file: pool rows from pool/np.npy, PubChem / generated rows
scored from their SMILES. Relative columns (np_like_gap / _rk) are computed by the training code.

    PYTHONPATH=src python scripts/eval/add_np.py data/artifacts/eval/features_hardpc300.parquet
"""

import sys
from multiprocessing import Pool as MP

import numpy as np
import pandas as pd

from casmi.edge import np_like
from casmi.paths import ROOT


def main():
    path = sys.argv[1]
    f = pd.read_parquet(path)
    pool_np = np.load(ROOT / "data" / "artifacts" / "pool" / "np.npy")
    v = np.full(len(f), np.nan, np.float32)
    pr = f.pool_row.values
    v[pr >= 0] = pool_np[pr[pr >= 0]]
    new = pd.unique(f.smiles.values[pr < 0])
    with MP(4) as mp:
        sc = dict(zip(new, mp.map(np_like, list(new), chunksize=1000)))
    v[pr < 0] = [sc[s] for s in f.smiles.values[pr < 0]]
    f["np_like"] = v
    f.to_parquet(path)
    lab = f[f.label == 1]
    print(f"{len(f):,} rows, {len(new):,} new structures scored; np_like median all {np.nanmedian(v):.2f}, "
          f"truths {np.nanmedian(lab.np_like):.2f}; by panel (truths):",
          lab.groupby("panel").np_like.median().round(2).to_dict())


if __name__ == "__main__":
    main()
