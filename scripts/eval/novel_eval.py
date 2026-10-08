"""Class-3 reach on genuinely novel molecules (panel N: absent from PubChem and COCONUT) vs simulated
novelty (panel C structures deleted from the pool). Regime C3 for both: every spectrum of the truth is
removed from the library and the truth from the pool. For each molecule the generator (pipeline.generate)
runs on the analog hits; we report whether the truth is generated (reach), how many structures are
generated, and the truth's rank among them by generator similarity.

    PYTHONPATH=src python scripts/eval/novel_eval.py [n_per_panel]
-> data/artifacts/eval_novel/reach.parquet
"""

import sys
import time
from multiprocessing import Pool as MP

import numpy as np
import pandas as pd

from casmi import pipeline as P
from casmi.paths import ROOT

sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from harness import load, query_of  # noqa: E402

ART = ROOT / "data" / "artifacts"
OUT = ART / "eval_novel"
_G = {}


def queries(lib, keys, rng):
    """Up to 3 positive-mode spectra per structure from its best-covered library (as panel C)."""
    sel = np.flatnonzero(np.isin(lib.key, list(keys)) & (lib.L["mode"] > 0))
    df = pd.DataFrame({"lrow": sel, "key": lib.key[sel], "src_lib": lib.L["lib_code"][sel]})
    best = df.groupby(["key", "src_lib"]).size().reset_index(name="n").sort_values(["key", "n"], ascending=[True, False])
    df = df.merge(best.drop_duplicates("key")[["key", "src_lib"]], on=["key", "src_lib"])
    return df.sample(frac=1.0, random_state=int(rng.integers(1 << 30))).groupby("key").head(3)


def job(args):
    from rdkit import Chem
    key, rows, panel = args
    pool, lib, ik = _G["pool"], _G["lib"], _G["ik"]
    truth = {key, ik.get(key, key)}
    q = query_of(lib, rows, key)
    h = P.analog_hits(q, lib, top=600)
    h = h[~h.key.isin(truth)]
    cand = pool.window(q.neutral_mass)
    keys = pool.key[cand]
    keys = keys[~np.isin(keys, list(truth))]
    t = time.time()
    gen = P.generate(q, pool, h, cand_keys=keys)
    dt = time.time() - t
    smi = _G["smiles"][key]
    tik = Chem.MolToInchiKey(Chem.MolFromSmiles(smi))[:14]
    hit = gen.key.isin(truth | {tik}).values if len(gen) else np.zeros(0, bool)
    rank = int(np.flatnonzero(hit[np.argsort(-gen.gen_sim.values, kind="stable")])[0]) + 1 if hit.any() else -1
    best_sim = float(h.sim.max()) if len(h) else 0.0
    return {"key": key, "panel": panel, "n_gen": len(gen), "reach": bool(hit.any()), "gen_rank": rank,
            "best_hit": best_sim, "n_pool": len(keys), "secs": dt}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    OUT.mkdir(parents=True, exist_ok=True)
    pool, lib = load()
    from casmi.paths import INTERIM
    km = pd.read_parquet(INTERIM / "metric_keys.parquet").dropna().drop_duplicates("metric_key")
    nov = pd.read_parquet(ART / "holdout_novel.parquet")
    nov = nov[~nov.in_pubchem]
    ho = pd.read_parquet(ART / "holdout.parquet")
    rng = np.random.default_rng(0)
    smiles = dict(zip(pool.df.key.values, pool.df.smiles.values))
    panels = {"N": list(nov.key.sample(min(n, len(nov)), random_state=0)),
              "C": list(ho[ho.panel == "C"].key.sample(n, random_state=0))}
    jobs = []
    for p, keys in panels.items():
        qs = queries(lib, set(keys), rng)
        for k, g in qs.groupby("key", sort=False):
            jobs.append((k, g.lrow.values, p))
    _G.update(pool=pool, lib=lib, ik=dict(zip(km.metric_key, km.inchikey14)), smiles=smiles)
    print("molecules", pd.Series([j[2] for j in jobs]).value_counts().to_dict(), flush=True)
    t0, res = time.time(), []
    with MP(4) as mp:
        for i, r in enumerate(mp.imap_unordered(job, jobs, chunksize=2)):
            res.append(r)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    r = pd.DataFrame(res)
    r.to_parquet(OUT / "reach.parquet")
    r["rr"] = np.where(r.gen_rank > 0, 1.0 / r.gen_rank.clip(lower=1), 0.0)
    r.loc[r.gen_rank > 25, "rr"] = 0.0
    print(r.groupby("panel").agg(n=("key", "size"), reach=("reach", "mean"), mrr_gen_only=("rr", "mean"),
                                 n_gen=("n_gen", "median"), best_hit=("best_hit", "median"),
                                 secs=("secs", "median")).round(3).to_string())


if __name__ == "__main__":
    main()
