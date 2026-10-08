"""NP-likeness of every pool structure, row-aligned with pool.parquet -> data/artifacts/pool/np.npy (float32).

    PYTHONPATH=src python scripts/build/build_np.py
"""

from multiprocessing import Pool as MP

import numpy as np
import pandas as pd

from casmi.edge import np_like
from casmi.paths import ROOT

POOL = ROOT / "data" / "artifacts" / "pool"


def main():
    smi = pd.read_parquet(POOL / "pool.parquet", columns=["smiles"]).smiles.tolist()
    with MP(4) as mp:
        v = np.array(mp.map(np_like, smi, chunksize=2000), np.float32)
    np.save(POOL / "np.npy", v)
    print(f"{len(v):,} structures; NaN {np.isnan(v).mean():.4f}; median {np.nanmedian(v):.2f}")


if __name__ == "__main__":
    main()
