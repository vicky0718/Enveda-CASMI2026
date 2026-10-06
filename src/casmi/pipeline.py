"""End-to-end inference on a test parquet: molecule queries -> candidates -> channel scores -> ranked
SMILES (25 per molecule, deduplicated by metric key). Used by the harness and the Kaggle notebook.
"""

import numpy as np
import pandas as pd

from .search import Library, Pool, Query, analog_hits, make_query, tanimoto

P_EXP, Q_EXP, TOP_PER_SPEC = 3.0, 1.0, 50


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
                   p_exp=P_EXP, q_exp=Q_EXP, top_per_spec=TOP_PER_SPEC):
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
    else:
        a_mean = a_max = tmax = np.zeros(len(cand))
    return pd.DataFrame({"pool_row": cand, "key": ckeys, "direct": d, "direct_n": dn, "analog": a_mean,
                         "analog_max": a_max, "tmax": tmax,
                         "mass_err_ppm": (pool.mass[cand] - q.neutral_mass) / q.neutral_mass * 1e6})


def heuristic_rank(f: pd.DataFrame) -> np.ndarray:
    score = np.where(f.direct.values > 0.5, 1.0 + f.direct.values, f.analog.values)
    return np.argsort(-score, kind="stable")


def run(test: pd.DataFrame, pool: Pool, lib: Library, ranker=None, log=print):
    rows = []
    qs = queries_from_test(test)
    for i, q in enumerate(qs):
        cand = pool.window(q.neutral_mass) if q.mz else np.zeros(0, np.int64)
        if len(cand) == 0:
            rows.append((q.mid, []))
            continue
        h = analog_hits(q, lib, top=300)
        f = channel_scores(q, pool, lib, h, cand)
        order = ranker(f) if ranker is not None else heuristic_rank(f)
        rows.append((q.mid, list(pool.df.smiles.values[cand[order]])))
        if i % 50 == 0:
            log(f"  {i}/{len(qs)} molecules")
    return rows
