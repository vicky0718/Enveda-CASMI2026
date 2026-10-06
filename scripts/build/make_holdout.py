"""Fixed validation panels (metric keys), held out from every model we train.

  A: every enveda-np-examples structure (the host's test-pipeline examples)
  B: 1,000 enveda-180 structures with NP-likeness > 0 (timsTOF; only ~1.9k of 183k enveda-180
     structures are NP-like, median NP-likeness -1.5 vs +1.5 for np-examples), seed 0
  C: 1,000 natural products (NP-likeness > 1) with spectra in public libraries only (no Enveda
     spectra), seed 0 — the class-2 proxy for NP chemistry (A is small, B is synthetic-like)

    PYTHONPATH=src python scripts/build/make_holdout.py   -> data/artifacts/holdout.parquet
"""

import pandas as pd

from casmi.io import load_meta
from casmi.paths import INTERIM, ROOT


def main():
    meta = load_meta()[["ingest_lib", "normalized_smiles"]]
    keys = pd.read_parquet(INTERIM / "metric_keys.parquet").drop_duplicates("normalized_smiles")
    desc = pd.read_parquet(INTERIM / "structure_descriptors.parquet")[["normalized_smiles", "np_likeness"]]
    m = meta.merge(keys[["normalized_smiles", "metric_key"]], on="normalized_smiles", how="left")
    a = set(m.loc[m.ingest_lib == "enveda-np-examples", "metric_key"].dropna())
    e = (m[m.ingest_lib == "enveda-180"].drop_duplicates("metric_key")
         .merge(desc.drop_duplicates("normalized_smiles"), on="normalized_smiles", how="left"))
    e = e[(e.np_likeness > 0) & ~e.metric_key.isin(a)].sort_values("metric_key")
    b = set(e.metric_key.sample(1000, random_state=0))
    libs_of = m.groupby("metric_key").ingest_lib.agg(lambda x: set(x.astype(str)))
    c = (m.drop_duplicates("metric_key").merge(desc.drop_duplicates("normalized_smiles"), on="normalized_smiles",
                                                 how="left"))
    c = c[(c.np_likeness > 1) & ~c.metric_key.isin(a | b)]
    c = c[c.metric_key.map(lambda k: not (libs_of[k] & {"enveda-180", "enveda-np-examples"}))]
    c = set(c.sort_values("metric_key").metric_key.sample(1000, random_state=0))
    out = pd.DataFrame({"key": sorted(a) + sorted(b) + sorted(c),
                        "panel": ["A"] * len(a) + ["B"] * len(b) + ["C"] * len(c)})
    path = ROOT / "data" / "artifacts" / "holdout.parquet"
    out.to_parquet(path)
    print(out.panel.value_counts().to_dict(), "->", path)


if __name__ == "__main__":
    main()
