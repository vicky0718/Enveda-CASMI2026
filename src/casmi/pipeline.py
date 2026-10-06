"""End-to-end inference on a test parquet: molecule queries -> candidates -> channel scores -> ranked
SMILES (25 per molecule, deduplicated by metric key). Used by the harness and the Kaggle notebook.
"""

import os

import numpy as np
import pandas as pd

from .frag import frag_scores
from .search import Library, Pool, Query, analog_hits, make_query, tanimoto

P_EXP, Q_EXP, TOP_PER_SPEC = 3.0, 1.0, 50
FULL_BITS = 4096 + 4096 + 2048 + 167


def load_fp_models(art, device=None):
    """Our trained FP nets (fpnet*.pt) + the informative-bit index. Returns None if absent."""
    import glob

    import torch

    from . import fpmodel as M
    paths = sorted(glob.glob(f"{art}/fpnet*.pt"))
    if not paths:
        return None
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    nets = []
    for pth in paths:
        ck = torch.load(pth, map_location="cpu", weights_only=True)
        net = M.FPNet(int(ck["nbits"]), d=int(ck["d"]), layers=int(ck["layers"]))
        net.load_state_dict({k: v.float() for k, v in ck["model"].items()})
        nets.append(net.to(device).eval())
    bits = f"{art}/fp_bits.npy" if os.path.exists(f"{art}/fp_bits.npy") else f"{art}/pool/fp_bits.npy"
    return {"nets": nets, "bits": np.load(bits), "device": device}


def fp_logits(q: Query, fpm):
    """Mean logits over a molecule's spectra, plus the merged-spectrum view (two input views)."""
    from . import fpmodel as M
    if fpm is None or not q.raw:
        return None
    peaks = [M.prep_peaks(m, i, pm) for (m, i), pm in zip(q.raw, q.prec)]
    merged = M.prep_peaks(*M.merge_peaks(q.raw), float(np.median(q.prec)))
    ins = ["timsTOF"] * (len(peaks) + 1)
    ces = [0.0 if np.isnan(c) else c for c in q.ce]
    ces = ces + [float(np.mean(ces))]
    modes = [1.0 if m > 0 else 0.0 for m in q.mode]
    modes = modes + [modes[0]]
    adducts = list(q.adduct) + [q.adduct[0]]
    precs = list(q.prec) + [float(np.median(q.prec))]
    z = M.logits(fpm["nets"], peaks + [merged], precs, adducts, ins, ces, modes, device=fpm["device"])
    return 0.5 * z[:-1].mean(0) + 0.5 * z[-1]


def fp_scores(pool: Pool, cand, z, bits):
    """f·z for each candidate (Bayes log-likelihood up to a constant)."""
    y = np.unpackbits(np.asarray(pool.fp[cand]), axis=1, count=FULL_BITS)[:, bits].astype(np.float32)
    return y @ z.astype(np.float32)


def queries_from_test(test: pd.DataFrame):
    out = []
    for mid, g in test.groupby("molecule_id", sort=False):
        specs = [dict(mz=r.ms2_mzs, it=r.ms2_normalized_intensities, adduct=r.adduct, mode=r.ionization_mode,
                      prec=float(r.precursor_mz),
                      ce=float(np.mean(r.collision_energy_ev)) if r.collision_energy_ev is not None
                      and len(r.collision_energy_ev) else np.nan)
                 for r in g.itertuples()]
        out.append(make_query(mid, specs, enveda=True))
    return out


def channel_scores(q: Query, pool: Pool, lib: Library, hits: pd.DataFrame, cand: np.ndarray,
                   p_exp=P_EXP, q_exp=Q_EXP, top_per_spec=TOP_PER_SPEC, z=None, bits=None,
                   frag=None):
    """Per-candidate features for one molecule. `hits` must already exclude any references the
    evaluation regime forbids. Returns DataFrame indexed like `cand`."""
    ckeys = pool.key[cand]
    hh = hits[hits.pool_row.values >= 0]
    direct = hh[np.abs(hh.delta.values) < 0.01]
    d = direct.groupby("key").sim.max().reindex(ckeys).fillna(0).values
    dn = direct[direct.sim > 0.3].groupby("key").spec.nunique().reindex(ckeys).fillna(0).values
    top = hh.sort_values("sim", ascending=False).groupby("spec").head(top_per_spec)
    if len(top):
        T = tanimoto(pool.ecfp4[cand], pool.ecfp4[top.pool_row.values])
        contrib = (top.sim.values[None, :] ** p_exp) * (T ** q_exp)
        spec = top.spec.values
        per = np.stack([contrib[:, spec == s].max(1) for s in np.unique(spec)], 1)
        a_mean, a_max = per.mean(1), per.max(1)
        tmax = T.max(1)
        # same instrument as the test (timsTOF) references only; and the best few analogs' consensus
        tims = lib.L["instr"][top.ref.values] == 0
        a_tims = np.stack([np.where(tims[spec == s], contrib[:, spec == s], 0).max(1) for s in np.unique(spec)],
                          1).mean(1)
        k = min(5, contrib.shape[1])
        a_top5 = np.sort(contrib, 1)[:, -k:].mean(1)
        sw = top.sim.values ** p_exp
        t_wmean = (T * sw[None, :]).sum(1) / max(sw.sum(), 1e-9)
    else:
        a_mean = a_max = tmax = a_tims = a_top5 = t_wmean = np.zeros(len(cand))
    f = pd.DataFrame({"pool_row": cand, "key": ckeys, "direct": d, "direct_n": dn, "analog": a_mean,
                      "analog_max": a_max, "tmax": tmax, "analog_tims": a_tims, "analog_top5": a_top5,
                      "t_wmean": t_wmean,
                      "mass_err_ppm": (pool.mass[cand] - q.neutral_mass) / q.neutral_mass * 1e6})
    if z is not None:
        fz = fp_scores(pool, cand, z, bits)
        f["fp"] = fz - fz.max()
        f["fp_rank"] = pd.Series(-fz).rank(method="min").values
    if frag is not None:
        f["frag"] = frag
    return add_relative(f)


REL_COLS = ["direct", "analog", "analog_max", "tmax", "frag", "analog_tims", "analog_top5", "t_wmean"]


def add_relative(f: pd.DataFrame) -> pd.DataFrame:
    """Within-molecule relative features (gap to the best candidate, rank), shared by training and
    inference."""
    for c in [c for c in REL_COLS if c in f.columns]:
        f[c + "_gap"] = f[c] - f[c].max()
        f[c + "_rk"] = f[c].rank(ascending=False, method="min")
    f["n_cand"] = len(f)
    return f


def heuristic_rank(f: pd.DataFrame) -> np.ndarray:
    score = np.where(f.direct.values > 0.5, 1.0 + f.direct.values, f.analog.values)
    return np.argsort(-score, kind="stable")


def load_ranker(art):
    """LightGBM ranker(s) trained by scripts/eval/train_ranker.py, or None (heuristic ranking)."""
    import glob
    import json
    paths = sorted(glob.glob(f"{art}/ranker*.txt"))
    if not paths:
        return None
    import lightgbm as lgb
    boosters = [lgb.Booster(model_file=p) for p in paths]
    cols = json.load(open(f"{art}/ranker_features.json"))

    def rank(f):
        x = f.reindex(columns=cols).astype(np.float32).values
        return np.argsort(-np.mean([b.predict(x) for b in boosters], 0), kind="stable")
    return rank


def load_keycache(art):
    """smiles -> metric key for pool rows whose key is known (train structures, keyed COCONUT rows)."""
    p = f"{art}/keycache.parquet"
    if not os.path.exists(p):
        return {}
    k = pd.read_parquet(p)
    return dict(zip(k.smiles, k.metric_key))


def run(test: pd.DataFrame, pool: Pool, lib: Library, ranker=None, fp_models=None, log=print):
    rows = []
    qs = queries_from_test(test)
    for i, q in enumerate(qs):
        try:
            cand = pool.window(q.neutral_mass) if q.mz else np.zeros(0, np.int64)
            if len(cand) == 0:
                rows.append((q.mid, []))
                continue
            h = analog_hits(q, lib, top=300)
            z = fp_logits(q, fp_models)
            fr = frag_scores(list(pool.df.smiles.values[cand]), q)
            f = channel_scores(q, pool, lib, h, cand, z=z, bits=None if fp_models is None else fp_models["bits"],
                               frag=fr)
            order = ranker(f) if ranker is not None else heuristic_rank(f)
            rows.append((q.mid, list(pool.df.smiles.values[cand[order]])))
        except Exception as e:  # one bad molecule must never sink the file
            log(f"  {q.mid}: {type(e).__name__}: {e}")
            rows.append((q.mid, []))
        if i % 50 == 0:
            log(f"  {i}/{len(qs)} molecules")
    return rows
