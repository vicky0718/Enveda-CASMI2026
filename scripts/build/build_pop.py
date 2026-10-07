"""Pool-aligned popularity prior from the public PubChem popularity table (dmitriigluzdov/
casmi26-pubchem-popularity-prior, pool_popularity.csv: PubChem substance / PubMed / patent counts,
NCBI free reuse). Membership flags (COCONUT / LOTUS / NPAtlas) are deliberately not used.

    PYTHONPATH=src python scripts/build/build_pop.py   -> data/artifacts/pool/pop.npy (n_pool, 3) float32
columns: pop (= log1p substances + log1p PubMed), log1p_patents, log1p_pubmed; NaN where unknown.
"""

import numpy as np
import pandas as pd

from casmi.paths import ROOT

POOL = ROOT / "data" / "artifacts" / "pool"


def main():
    pp = pd.read_csv(ROOT / "data" / "external" / "pop_prior" / "pool_popularity.csv",
                     usecols=["inchikey14", "pop", "log1p_patents", "log1p_pubmed"]).drop_duplicates("inchikey14")
    pool = pd.read_parquet(POOL / "pool.parquet", columns=["key"])
    m = pool[["key"]].merge(pp, left_on="key", right_on="inchikey14", how="left")
    arr = m[["pop", "log1p_patents", "log1p_pubmed"]].to_numpy(np.float32)
    np.save(POOL / "pop.npy", arr)
    print("pop coverage", float(np.isfinite(arr[:, 0]).mean()))


if __name__ == "__main__":
    main()
