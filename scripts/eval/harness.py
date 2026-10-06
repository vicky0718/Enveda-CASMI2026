"""Validation harness on the held-out panels (data/artifacts/holdout.parquet).

Queries are library rows of held-out structures (panel A: all enveda-np-examples spectra of a
structure; panel B: <= 3 random enveda-180 spectra per structure, test-like counts). Regimes:
  C1  only the query's own library is removed for its structure (other libraries keep it)
  C2  every spectrum of the structure is removed from the library; structure stays in the pool
Step 1 (cached): per query spectrum, the top reference spectra by direct+shifted entropy similarity.
Step 2: channel scores per candidate and MRR@25.

    PYTHONPATH=src python scripts/eval/harness.py hits      # step 1 (slow, cached)
    PYTHONPATH=src python scripts/eval/harness.py score     # step 2
"""

import sys
import time

import numpy as np
import pandas as pd

from casmi import library as L
from casmi.paths import ROOT
from casmi.search import Library, Pool, Query, analog_hits, tanimoto

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
            df = df.groupby("key", group_keys=False).apply(
                lambda g: g.sample(min(3, len(g)), random_state=int(rng.integers(1 << 30))))
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


def mrr(ranked_keys, truth):
    for i, k in enumerate(ranked_keys[:25]):
        if k == truth:
            return 1.0 / (i + 1)
    return 0.0


def score(p_exp=3.0, q_exp=1.0):
    pool, lib = load()
    qs = pd.read_parquet(EVAL / "queries.parquet")
    H = pd.read_parquet(EVAL / "hits.parquet")
    H["ref_lib"] = lib.L["lib_code"][H.ref.values]
    src_of = qs.drop_duplicates("key").set_index("key").src_lib
    res = []
    for key, h in H.groupby("qkey", sort=False):
        g = qs[qs.key == key]
        M = query_of(lib, g.lrow.values, key).neutral_mass
        cand = pool.window(M)
        ckeys = pool.key[cand]
        for regime in ("C1", "C2"):
            same = h.key.values == key
            excl = same if regime == "C2" else same & (h.ref_lib.values == src_of[key])
            hh = h[~excl & (h.pool_row.values >= 0)]
            direct = hh[np.abs(hh.delta.values) < 0.01]
            dscore = direct.groupby("key").sim.max()
            # analog propagation: max over hits of sim^p * T^q, then mean over the molecule's spectra
            top = hh.sort_values("sim", ascending=False).groupby("spec").head(50)
            T = tanimoto(pool.ecfp4[cand], pool.ecfp4[top.pool_row.values])
            contrib = (top.sim.values[None, :] ** p_exp) * (T ** q_exp)
            spec = top.spec.values
            per_spec = np.stack([contrib[:, spec == s].max(1) for s in np.unique(spec)], 1) \
                if len(top) else np.zeros((len(cand), 1))
            ascore = per_spec.mean(1)
            d = dscore.reindex(ckeys).fillna(0).values
            # direct library match dominates when strong; analog score otherwise
            final = np.where(d > 0.5, 1.0 + d, ascore)
            order = np.argsort(-final, kind="stable")
            ranked = list(dict.fromkeys(ckeys[order]))
            res.append({"key": key, "panel": g.panel.iloc[0], "regime": regime, "n_cand": len(set(ckeys)),
                        "in_pool": key in set(ckeys),
                        "mrr": mrr(ranked, key),
                        "mrr_analog": mrr(list(dict.fromkeys(ckeys[np.argsort(-ascore, kind="stable")])), key)})
    r = pd.DataFrame(res)
    r.to_parquet(EVAL / "scores.parquet")
    print(r.groupby(["panel", "regime"])[["mrr", "mrr_analog", "in_pool", "n_cand"]].mean().round(4))


if __name__ == "__main__":
    {"hits": hits, "score": score}[sys.argv[1]]()
