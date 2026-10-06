"""Validation harness on the held-out panels (data/artifacts/holdout.parquet).

Queries are library rows of held-out structures (panel A: all enveda-np-examples spectra of a
structure; panel B: <= 3 random enveda-180 spectra per structure, test-like counts). Regimes:
  C1  only the query's own library is removed for its structure (other libraries keep it)
  C2  every spectrum of the structure is removed from the library; structure stays in the pool
Step 1 (cached): per query spectrum, the top reference spectra by direct+shifted entropy similarity.
Step 2: channel scores per candidate and MRR@25.

    PYTHONPATH=src python scripts/eval/harness.py hits      # step 1 (slow, cached)
    PYTHONPATH=src python scripts/eval/harness.py frag      # step 1b: fragmentation feature (cached)
    PYTHONPATH=src python scripts/eval/harness.py score     # step 2
"""

import sys
import time

import numpy as np
import pandas as pd

from casmi import library as L
from casmi.paths import ROOT
from casmi import pipeline as P
from casmi.search import Library, Pool, Query, analog_hits

ART = ROOT / "data" / "artifacts"
EVAL = ART / "eval"
EVAL.mkdir(parents=True, exist_ok=True)


def load():
    pool = Pool(ART / "pool")
    lib = Library(L.load(ART / "library" / "library.npz"), pool)
    return pool, lib


def build_queries(lib: Library, seed=0):
    hold = pd.read_parquet(ART / "holdout.parquet")
    libs = list(lib.L["libs"])
    lc = lib.L["lib_code"]
    rng = np.random.default_rng(seed)
    out = []
    for panel, src in (("A", "enveda-np-examples"), ("B", "enveda-180")):
        keys = set(hold.key[hold.panel == panel])
        rows = np.flatnonzero((lc == libs.index(src)) & np.isin(lib.key, list(keys)))
        df = pd.DataFrame({"lrow": rows, "key": lib.key[rows]})
        if panel == "B":
            df = df.sample(frac=1.0, random_state=int(rng.integers(1 << 30))).groupby("key").head(3)
        df["panel"] = panel
        df["src_lib"] = libs.index(src)
        out.append(df)
    return pd.concat(out, ignore_index=True)


def query_of(lib: Library, rows, key):
    Lb = lib.L
    from casmi.library import ADDUCTS
    q = Query(key, [], [], [], [], [], [])
    for r in rows:
        a, b = Lb["off"][r], Lb["off"][r + 1]
        q.mz.append(Lb["mz"][a:b])
        q.p.append(Lb["p"][a:b])
        q.adduct.append(ADDUCTS[Lb["adduct_code"][r]])
        q.mode.append(int(Lb["mode"][r]))
        q.prec.append(float(Lb["prec_mz"][r]))
        q.ce.append(float(Lb["ce"][r]))
    return q


def hits():
    pool, lib = load()
    qs = build_queries(lib)
    qs.to_parquet(EVAL / "queries.parquet")
    print(qs.groupby("panel").agg(spectra=("lrow", "size"), molecules=("key", "nunique")), flush=True)
    t0 = time.time()
    parts = []
    for i, (key, g) in enumerate(qs.groupby("key", sort=False)):
        q = query_of(lib, g.lrow.values, key)
        h = analog_hits(q, lib, top=600)
        h["qkey"] = key
        h["spec_row"] = g.lrow.values[h.spec.values]
        parts.append(h)
        if i % 100 == 0:
            print(f"  {i} molecules, {time.time() - t0:.0f}s", flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(EVAL / "hits.parquet")


def _frag_job(args):
    key, rows, cand = args
    from casmi.frag import frag_scores
    q = query_of(_G["lib"], rows, key)
    return pd.DataFrame({"qkey": key, "pool_row": cand,
                         "frag": frag_scores(list(_G["pool"].df.smiles.values[cand]), q)})


_G = {}


def frag():
    """Fragmentation-explanation feature per (molecule, candidate); regime-independent, cached."""
    from multiprocessing import Pool as MP
    pool, lib = load()
    _G.update(pool=pool, lib=lib)
    qs = pd.read_parquet(EVAL / "queries.parquet")
    jobs = []
    for key, g in qs.groupby("key", sort=False):
        q = query_of(lib, g.lrow.values, key)
        jobs.append((key, g.lrow.values, pool.window(q.neutral_mass)))
    t0 = time.time()
    with MP(4) as mp:  # fork: workers share the loaded pool/library
        parts = []
        for i, r in enumerate(mp.imap_unordered(_frag_job, jobs, chunksize=4)):
            parts.append(r)
            if i % 100 == 0:
                print(f"  frag {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(EVAL / "frag.parquet")


def mrr(ranked_keys, truth):
    for i, k in enumerate(ranked_keys[:25]):
        if k == truth:
            return 1.0 / (i + 1)
    return 0.0


def score():
    pool, lib = load()
    qs = pd.read_parquet(EVAL / "queries.parquet")
    H = pd.read_parquet(EVAL / "hits.parquet")
    H["ref_lib"] = lib.L["lib_code"][H.ref.values]
    src_of = qs.drop_duplicates("key").set_index("key").src_lib
    FR = pd.read_parquet(EVAL / "frag.parquet") if (EVAL / "frag.parquet").exists() else None
    fr_of = {k: g.set_index("pool_row").frag for k, g in FR.groupby("qkey")} if FR is not None else {}
    res, feats = [], []
    for key, h in H.groupby("qkey", sort=False):
        g = qs[qs.key == key]
        q = query_of(lib, g.lrow.values, key)
        cand = pool.window(q.neutral_mass)
        for regime in ("C1", "C2"):
            same = h.key.values == key
            excl = same if regime == "C2" else same & (h.ref_lib.values == src_of[key])
            fr = fr_of[key].reindex(cand).fillna(0).values if key in fr_of else None
            f = P.channel_scores(q, pool, lib, h[~excl], cand, frag=fr)
            f["label"] = (f.key.values == key).astype(np.int8)
            f["qkey"], f["regime"], f["panel"] = key, regime, g.panel.iloc[0]
            feats.append(f)
            ranked = list(dict.fromkeys(f.key.values[P.heuristic_rank(f)]))
            ranked_a = list(dict.fromkeys(f.key.values[np.argsort(-f.analog.values, kind="stable")]))
            res.append({"key": key, "panel": g.panel.iloc[0], "regime": regime, "n_cand": f.key.nunique(),
                        "in_pool": bool(f.label.any()), "mrr": mrr(ranked, key), "mrr_analog": mrr(ranked_a, key)})
    r = pd.DataFrame(res)
    r.to_parquet(EVAL / "scores.parquet")
    pd.concat(feats, ignore_index=True).to_parquet(EVAL / "features.parquet")
    print(r.groupby(["panel", "regime"])[["mrr", "mrr_analog", "in_pool", "n_cand"]].mean().round(4))


if __name__ == "__main__":
    {"hits": hits, "frag": frag, "score": score}[sys.argv[1]]()
