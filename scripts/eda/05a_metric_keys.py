"""EDA pass 5a — metric keys for every training structure (cached).

Computes the competition metric's key (RDKit 2026.03.3 tautomer-canonical InChIKey14) for every
unique normalized_smiles and compares it with the shipped `inchikey14` column.

    PYTHONPATH=src python scripts/eda/05a_metric_keys.py
"""

import json
import time
from multiprocessing import Pool

import pandas as pd

from casmi.io import load_meta
from casmi.metric import metric_key
from casmi.paths import INTERIM, STATS

OUT = INTERIM / "metric_keys.parquet"


def main():
    m = load_meta()
    u = m.drop_duplicates("normalized_smiles")[["normalized_smiles", "inchikey14"]].reset_index(drop=True)
    if not OUT.exists():
        t0 = time.time()
        with Pool(4) as pool:
            keys = pool.map(metric_key, u.normalized_smiles.tolist(), chunksize=200)
        u["metric_key"] = keys
        u.to_parquet(OUT)
        print(f"computed {len(u):,} keys in {time.time() - t0:.0f}s")
    u = pd.read_parquet(OUT)
    st = {
        "unique_smiles": len(u),
        "unparseable": int(u.metric_key.isna().sum()),
        "shipped_ik14_distinct": int(u.inchikey14.nunique()),
        "metric_key_distinct": int(u.metric_key.nunique()),
        "share_smiles_key_differs_from_shipped": float((u.metric_key != u.inchikey14).mean()),
    }
    # Shipped-distinct structures that the metric treats as one ("tautomer twins").
    g = u.dropna().groupby("metric_key").inchikey14.nunique()
    st["metric_keys_covering_>1_shipped_ik14"] = int((g > 1).sum())
    st["shipped_ik14_in_twin_groups"] = int(g[g > 1].sum())
    # Per library: share of structures whose key changes.
    mm = m[["ingest_lib", "normalized_smiles", "inchikey14"]].drop_duplicates(["ingest_lib", "inchikey14"])
    mm = mm.merge(u[["normalized_smiles", "metric_key"]], on="normalized_smiles", how="left")
    st["share_key_changes_by_library"] = mm.groupby("ingest_lib").apply(
        lambda d: float((d.metric_key != d.inchikey14).mean())).round(4).to_dict()
    (STATS / "05a_metric_keys.json").write_text(json.dumps(st, indent=2))
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
