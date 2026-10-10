"""LightGBM lambdarank over per-candidate features — shared by the local harness and the Kaggle
ranker kernel. Folds are held out by molecule; models are fit only on the NP panels by default.
"""

import json

import numpy as np
import pandas as pd

BASE_FEATS = ["direct", "direct_n", "analog", "analog_max", "tmax", "mass_err_ppm", "frag", "analog_tims",
              "analog_top5", "t_wmean", "is_gen", "gen_sim", "gen_nsrc", "gen_steps", "gen_rule", "gen_absdelta",
              "analog_noself", "own_n", "own_sim", "own_neg", "frag_disc",
              "analog_w", "analog_ap", "analog_ap_w", "ap_tmax",
              "q_nspec", "q_npeaks", "q_entropy", "q_best_hit", "q_best_direct", "q_pos", "q_reliable", "q_mass",
              "is_pc", "np_like"]
# Popularity is NOT a ranker feature: validation truths are library compounds, far better documented than
# real class-2/3 answers (panel A: truth beats same-formula library isomers on popularity 96 % of the
# time; panel B loses 0.085 when the ranker learns it). It is applied as a small tie-breaker instead,
# tuned on panels B/C only: final = z(ranker score) + POP_LAMBDA * z(pop), z within the molecule's list.
POP_LAMBDA = 0.1


def blend_pop(score, pop):
    """z(score) + POP_LAMBDA * z(pop) within one molecule's candidate list; unknown pop -> list median."""
    score = np.asarray(score, float)
    pop = np.asarray(pop, float)
    if POP_LAMBDA == 0 or len(score) < 2 or np.all(np.isnan(pop)):
        return score
    pop = np.where(np.isnan(pop), np.nanmedian(pop), pop)
    z = lambda x: (x - x.mean()) / (x.std() or 1.0)  # noqa: E731
    return z(score) + POP_LAMBDA * z(pop)
PARAMS = dict(objective="lambdarank", metric="map", eval_at=[25], learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=50, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
              lambdarank_truncation_level=25, verbose=-1, seed=0, deterministic=True, num_threads=4)
ROUNDS = 300


def feature_cols(f: pd.DataFrame):
    feats = [c for c in BASE_FEATS + ["fp", "fp_rank", "fp_norm", "fwd"] if c in f.columns]
    rel = [c for c in f.columns if (c.endswith("_gap") or c.endswith("_rk")) and not c.startswith("pop")]
    return feats + rel + ["n_cand"]


def mrr_of(f: pd.DataFrame, score) -> pd.DataFrame:
    """MRR@25 per (molecule, regime): rank of the first labelled row after key de-duplication."""
    out = []
    for (k, r), g in f.assign(s=score).groupby(["qkey", "regime"], sort=False):
        o = np.argsort(-g.s.values, kind="stable")
        keys, labs = g.key.values[o], g.label.values[o]
        seen, rr = set(), 0.0
        for key, lab in zip(keys, labs):
            if key in seen:
                continue
            seen.add(key)
            if lab:
                rr = 1.0 / len(seen)
                break
            if len(seen) == 25:
                break
        out.append({"qkey": k, "regime": r, "panel": g.panel.iloc[0], "mrr": rr})
    return pd.DataFrame(out)


def _train(tr, feats, seed=0):
    import lightgbm as lgb
    tr = tr[tr.groupby("grp").label.transform("max") > 0]  # queries whose answer is in the list
    ds = lgb.Dataset(tr[feats].astype(np.float32), tr.label, group=tr.groupby("grp", sort=False).size().values)
    return lgb.train({**PARAMS, "seed": seed}, ds, ROUNDS)


def prepare(f: pd.DataFrame, regimes=("C1", "C2", "C3")):
    f = f[f.regime.isin(regimes)].reset_index(drop=True)
    f["grp"] = f.qkey + "|" + f.regime
    mols = f.qkey.unique()
    rng = np.random.default_rng(0)
    f["fold"] = f.qkey.map(dict(zip(mols, rng.integers(0, 5, len(mols)))))
    return f


def cv(f: pd.DataFrame, feats, panels=("A", "C")):
    oof = np.zeros(len(f))
    for k in range(5):
        b = _train(f[(f.fold != k) & f.panel.isin(panels)], feats)
        va = f[f.fold == k]
        oof[va.index] = b.predict(va[feats].astype(np.float32))
    return oof


def report(rep: pd.DataFrame, cols=("mrr",)):
    tab = rep.groupby(["panel", "regime"])[list(cols)].mean()
    lines = [tab.round(4).to_string()]
    for f1, f2, f3 in ((.16, .30, .54), (.16, .45, .39)):
        est = {p: round(sum(w * tab.loc[(p, r), "mrr"] for w, r in ((f1, "C1"), (f2, "C2"), (f3, "C3"))
                            if (p, r) in tab.index), 4) for p in ("A", "C")}
        lines.append(f"weighted (f1={f1}, f2={f2}, f3={f3}): {est}")
    return "\n".join(lines)


def fit_save(f: pd.DataFrame, feats, out_dir, panels=("A", "C"), seeds=3):
    tr = f[f.panel.isin(panels)]
    for seed in range(seeds):
        _train(tr, feats, seed).save_model(f"{out_dir}/ranker{seed}.txt")
    with open(f"{out_dir}/ranker_features.json", "w") as fh:
        json.dump(feats, fh)
