"""Validation harness on the held-out panels (data/artifacts/holdout.parquet).

Queries are library rows of held-out structures (panel A: all enveda-np-examples spectra of a
structure; panel B: <= 3 random enveda-180 spectra per structure, test-like counts). Regimes:
  C1  only the query's own library is removed for its structure (other libraries keep it)
  C2  every spectrum of the structure is removed from the library; structure stays in the pool
Step 1 (cached): per query spectrum, the top reference spectra by direct+shifted entropy similarity.
Step 2: channel scores per candidate and MRR@25.

    PYTHONPATH=src python scripts/eval/harness.py hits      # step 1 (slow, cached)
    PYTHONPATH=src python scripts/eval/harness.py frag      # step 1b: fragmentation feature (cached)
    PYTHONPATH=src python scripts/eval/harness.py fp <dir>  # step 1c: FP-model logits (our fpnet*.pt)
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
    # panel C: public-library NPs; one source library per structure (the one with most high-res spectra)
    keys = set(hold.key[hold.panel == "C"])
    rows = np.flatnonzero(np.isin(lib.key, list(keys)) & np.isin(lib.L["instr"], [1, 2]))
    df = pd.DataFrame({"lrow": rows, "key": lib.key[rows], "src_lib": lc[rows]})
    best = df.groupby(["key", "src_lib"]).size().reset_index(name="n").sort_values(["key", "n"], ascending=[True, False])
    best = best.drop_duplicates("key")[["key", "src_lib"]]
    df = df.merge(best, on=["key", "src_lib"])
    df = df.sample(frac=1.0, random_state=int(rng.integers(1 << 30))).groupby("key").head(3)
    df["panel"] = "C"
    out.append(df[["lrow", "key", "panel", "src_lib"]])
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
    old = pd.read_parquet(EVAL / "queries.parquet") if (EVAL / "queries.parquet").exists() else None
    if old is not None:  # keep earlier panels' sampled spectra (the cached hits refer to them)
        qs = pd.concat([old, qs[~qs.key.isin(set(old.key))]], ignore_index=True)
    qs.to_parquet(EVAL / "queries.parquet")
    print(qs.groupby("panel").agg(spectra=("lrow", "size"), molecules=("key", "nunique")), flush=True)
    t0 = time.time()
    parts = [pd.read_parquet(EVAL / "hits.parquet")] if (EVAL / "hits.parquet").exists() else []
    done = set(parts[0].qkey) if parts else set()
    for i, (key, g) in enumerate(qs[~qs.key.isin(done)].groupby("key", sort=False)):
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


def fp(model_dir):
    """FP-model logits per validation molecule (two views, as in the pipeline) -> eval/fpz.npz.
    model_dir holds our fpnet*.pt; it must be a run that held out every panel scored here."""
    from casmi.io import read_rows
    pool, lib = load()
    qs = pd.read_parquet(EVAL / "queries.parquet")
    fpm = P.load_fp_models(model_dir, device="cpu")
    fpm["bits"] = np.load(ART / "pool" / "fp_bits.npy")
    trows = lib.L["row"][qs.lrow.values]
    raw = read_rows(trows)
    libs = list(lib.L["libs"])
    out = {}
    t0 = time.time()
    for i, (key, g) in enumerate(qs.groupby("key", sort=False)):
        q = query_of(lib, g.lrow.values, key)
        for lr in g.lrow.values:
            mz, it = raw[int(lib.L["row"][lr])]
            enveda = libs[lib.L["lib_code"][lr]].startswith("enveda") and lib.L["mode"][lr] > 0
            q.raw.append((mz + (L.ENVEDA_POS_SHIFT if enveda else 0.0), it))
        out[key] = P.fp_logits(q, fpm).astype(np.float32)
        if i % 200 == 0:
            print(f"  fp {i} {time.time() - t0:.0f}s", flush=True)
    np.savez(EVAL / "fpz.npz", keys=np.array(list(out)), z=np.stack(list(out.values())))


def mrr(ranked_keys, truth):
    for i, k in enumerate(ranked_keys[:25]):
        if k == truth:
            return 1.0 / (i + 1)
    return 0.0


def _score_job(key):
    from casmi.frag import frag_scores
    pool, lib, qs, h = _G["pool"], _G["lib"], _G["qs"], _G["H"].get(key)
    g = qs[qs.key == key]
    q = query_of(lib, g.lrow.values, key)
    cand_all = pool.window(q.neutral_mass)
    truth = {key, _G["ik_of"].get(key, key)}
    fr_all = _G["fr_of"][key].reindex(cand_all).fillna(0).values if key in _G["fr_of"] else np.zeros(len(cand_all))
    res, feats, gens = [], [], {}
    for regime in ("C1", "C2", "C3"):
        same = h.key.values == key
        excl = same & (h.ref_lib.values == _G["src_of"][key]) if regime == "C1" else same
        hh = h[~excl]
        keep = ~np.isin(pool.key[cand_all], list(truth)) if regime == "C3" else np.ones(len(cand_all), bool)
        cand = cand_all[keep]
        gkey = "C1" if regime == "C1" else "C23"
        if gkey not in gens:
            gens[gkey] = P.generate(q, pool, hh, cand_keys=pool.key[cand_all])
        gen = gens[gkey]
        if regime == "C3":  # the truth is not in the pool: generated copies of it must stay
            gen = P.generate(q, pool, hh, cand_keys=pool.key[cand])
        fr = np.r_[fr_all[keep], frag_scores(list(gen.smiles), q) if len(gen) else []]
        f = P.channel_scores(q, pool, lib, hh, cand, frag=fr, z=_G["z_of"].get(key), bits=_G["bits"], gen=gen)
        f["label"] = np.isin(f.key.values, list(truth)).astype(np.int8)
        f["qkey"], f["regime"], f["panel"] = key, regime, g.panel.iloc[0]
        feats.append(f)
        ranked = list(dict.fromkeys(f.key.values[P.heuristic_rank(f)]))
        res.append({"key": key, "panel": g.panel.iloc[0], "regime": regime, "n_cand": f.key.nunique(),
                    "n_gen": int(f.is_gen.sum()), "in_list": bool(f.label.any()),
                    "mrr": max(mrr(ranked, t) for t in truth)})
    return res, pd.concat(feats, ignore_index=True)


def score():
    from multiprocessing import Pool as MP
    from casmi.paths import INTERIM
    pool, lib = load()
    qs = pd.read_parquet(EVAL / "queries.parquet")
    H = pd.read_parquet(EVAL / "hits.parquet")
    H["ref_lib"] = lib.L["lib_code"][H.ref.values]
    FR = pd.read_parquet(EVAL / "frag.parquet") if (EVAL / "frag.parquet").exists() else None
    Z = np.load(EVAL / "fpz.npz") if (EVAL / "fpz.npz").exists() else None
    keys = pd.read_parquet(INTERIM / "metric_keys.parquet").dropna().drop_duplicates("metric_key")
    _G.update(pool=pool, lib=lib, qs=qs, H={k: g for k, g in H.groupby("qkey", sort=False)},
              src_of=qs.drop_duplicates("key").set_index("key").src_lib,
              fr_of={k: g.set_index("pool_row").frag for k, g in FR.groupby("qkey")} if FR is not None else {},
              z_of=dict(zip(Z["keys"], Z["z"])) if Z is not None else {},
              bits=np.load(ART / "pool" / "fp_bits.npy"), ik_of=dict(zip(keys.metric_key, keys.inchikey14)))
    t0 = time.time()
    res, feats = [], []
    with MP(4) as mp:  # fork: workers share the loaded pool / library
        for i, (r, f) in enumerate(mp.imap_unordered(_score_job, list(_G["H"]), chunksize=4)):
            res += r
            feats.append(f)
            if i % 300 == 0:
                print(f"  score {i}/{len(_G['H'])} {time.time() - t0:.0f}s", flush=True)
    r = pd.DataFrame(res)
    r.to_parquet(EVAL / "scores.parquet")
    pd.concat(feats, ignore_index=True).to_parquet(EVAL / "features.parquet")
    print(r.groupby(["panel", "regime"])[["mrr", "in_list", "n_cand", "n_gen"]].mean().round(4))


if __name__ == "__main__":
    {"hits": hits, "frag": frag, "score": score, "fp": fp}[sys.argv[1]](*sys.argv[2:])
