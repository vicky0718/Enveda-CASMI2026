"""Metric keys (stereo strip -> tautomer canonical -> InChIKey14) for COCONUT-only pool rows.

Slow (~60 ms/molecule), so it runs niced in the background in resumable chunks:
    PYTHONPATH=src nice -n 19 python scripts/build/coconut_metric_keys.py
-> data/artifacts/pool/coconut_keys/chunk_XXXX.parquet (pool row, metric_key)
"""

import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from casmi.metric import candidate_key
from casmi.paths import ROOT

POOL = ROOT / "data" / "artifacts" / "pool"
OUT = POOL / "coconut_keys"
CHUNK = 20000


def _key(s):
    try:
        return candidate_key(s)
    except Exception:
        return None


def main():
    OUT.mkdir(exist_ok=True)
    pool = pd.read_parquet(POOL / "pool.parquet", columns=["smiles", "key_is_metric"])
    rows = np.flatnonzero(~pool.key_is_metric.values)
    t0 = time.time()
    with Pool(4) as p:
        for c in range(0, len(rows), CHUNK):
            f = OUT / f"chunk_{c // CHUNK:04d}.parquet"
            if f.exists():
                continue
            r = rows[c:c + CHUNK]
            keys = p.map(_key, pool.smiles.values[r].tolist(), chunksize=50)
            pd.DataFrame({"row": r, "metric_key": keys}).to_parquet(f)
            print(f"{c + len(r):,}/{len(rows):,} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
