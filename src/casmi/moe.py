"""Multi-model ranking: evidence-family experts + a router-conditioned stacked meta-ranker.

Why: a single ranker over all features learns the dominant validation signal (analog propagation, strong
because held-out library compounds have measured relatives) and plateaus on the leaderboard. Here each
expert sees one family of evidence, so it stays useful where the others are silent, and the meta-ranker
decides per molecule — from router descriptors (library-match strength, analog strength, spectral
information, polarity, adduct reliability, list size) — how much to trust each expert.

    experts   library  : the candidate's own library spectra (class-1 specialist)
              analog   : propagation from spectrally similar library structures (ECFP + atom-pair)
              spectral : spectrum-intrinsic evidence — fragmentation, fingerprint model f·z (no neighbours)
              full     : everything
    level 2   per expert: z-score and rank within the molecule's list, margin to the list's best,
              + router features -> LightGBM lambdarank; popularity tie-break applied after.

CONFIG also holds the problem-specific candidate filters (mass windows per instrument) and source rules.
"""

import json

import numpy as np
import pandas as pd

from . import rank as R

CONFIG = {
    "experts": {
        "library": ["direct", "direct_n", "own_n", "own_sim", "own_neg"],
        "analog": ["analog", "analog_max", "analog_top5", "analog_tims", "analog_noself", "analog_w", "analog_ap",
                   "analog_ap_w", "tmax", "ap_tmax", "t_wmean"],
        "spectral": ["frag", "frag_disc", "fp", "fp_rank", "fp_norm"],
        "full": None,  # None = every ranker feature
    },
    "always": ["mass_err_ppm"],
    "router": ["q_nspec", "q_npeaks", "q_entropy", "q_best_hit", "q_best_direct", "q_pos", "q_reliable", "q_mass",
               "n_cand"],
    # precursor-mass windows (ppm half-width, centre offset of candidate − query in ppm): timsTOF truths sit at
    # −0.6 to −1.0 ppm with 95–99 % inside ±3 ppm; other instruments are wider and unbiased
    "mass_window": {"timsTOF": {"ppm": 5.0, "center_ppm": -0.8}, "default": {"ppm": 10.0, "center_ppm": 0.0}},
    # which model ranks at inference: "full" (the all-evidence expert) or "meta" (stacked). With the FP expert
    # and the analog-thinned regime C2H in training they tie on validation (C 0.300 / 0.300; A 0.391 vs 0.383)
    "inference": "full",
    "level1_rounds": 300,
    "level2_rounds": 200,
}


def expert_features(f: pd.DataFrame, name: str):
    base = CONFIG["experts"][name]
    if base is None:
        return R.feature_cols(f)
    cols = [c for c in base + CONFIG["always"] if c in f.columns]
    rel = [f"{c}_{s}" for c in base for s in ("gap", "rk") if f"{c}_{s}" in f.columns]
    return cols + rel


def level2_frame(f: pd.DataFrame, scores: dict) -> pd.DataFrame:
    """Per-expert within-list z-score, rank and margin to the list's best, plus router features."""
    out = {}
    g = f.groupby("grp", sort=False)
    for name, s in scores.items():
        s = pd.Series(s, index=f.index)
        gs = s.groupby(f.grp, sort=False)
        mu, sd = gs.transform("mean"), gs.transform("std").fillna(0).replace(0, 1.0)
        out[f"e_{name}_z"] = (s - mu) / sd
        out[f"e_{name}_rk"] = gs.rank(ascending=False, method="min")
        out[f"e_{name}_gap"] = s - gs.transform("max")
    for c in CONFIG["router"]:
        if c in f.columns:
            out[c] = f[c].values
    out["n_cand"] = g["key"].transform("size").values
    return pd.DataFrame(out, index=f.index)


def _lgb_train(X, y, groups, rounds, seed=0):
    import lightgbm as lgb
    params = {**R.PARAMS, "seed": seed}
    return lgb.train(params, lgb.Dataset(X.astype(np.float32), y, group=groups), rounds)


def _fit(f, cols, rounds, seed=0):
    tr = f[f.groupby("grp").label.transform("max") > 0]
    return _lgb_train(tr[cols], tr.label, tr.groupby("grp", sort=False).size().values, rounds, seed)


def cv(f: pd.DataFrame, panels=("A", "C")):
    """Out-of-fold level-1 expert scores and level-2 (meta) scores; folds held out by molecule."""
    names = list(CONFIG["experts"])
    l1 = {n: np.zeros(len(f)) for n in names}
    for k in range(5):
        tr, va = f[(f.fold != k) & f.panel.isin(panels)], f[f.fold == k]
        for n in names:
            cols = expert_features(f, n)
            l1[n][va.index] = _fit(tr, cols, CONFIG["level1_rounds"]).predict(va[cols].astype(np.float32))
    X2 = level2_frame(f, l1)
    f2 = f[["grp", "label", "fold", "panel"]].join(X2)
    meta = np.zeros(len(f))
    cols2 = list(X2.columns)
    for k in range(5):
        tr, va = f2[(f2.fold != k) & f2.panel.isin(panels)], f2[f2.fold == k]
        meta[va.index] = _fit(tr, cols2, CONFIG["level2_rounds"]).predict(va[cols2].astype(np.float32))
    return l1, meta


def fit_save(f: pd.DataFrame, out_dir, panels=("A", "C")):
    """Final models: experts fit on all data; meta-ranker fit on out-of-fold expert scores."""
    names = list(CONFIG["experts"])
    tr = f[f.panel.isin(panels)]
    spec = {"experts": {}, "config": CONFIG}
    for n in names:
        cols = expert_features(f, n)
        _fit(tr, cols, CONFIG["level1_rounds"]).save_model(f"{out_dir}/moe_{n}.txt")
        spec["experts"][n] = cols
    l1, _ = cv(f, panels)
    X2 = level2_frame(f, l1)
    f2 = f[["grp", "label", "fold", "panel"]].join(X2)
    cols2 = list(X2.columns)
    _fit(f2[f2.panel.isin(panels)], cols2, CONFIG["level2_rounds"]).save_model(f"{out_dir}/moe_meta.txt")
    spec["meta"] = cols2
    with open(f"{out_dir}/moe.json", "w") as fh:
        json.dump(spec, fh)


def load(model_dir):
    """Inference closure over one molecule's candidate frame -> ranking order (or None if absent)."""
    import os

    import lightgbm as lgb
    if not os.path.exists(f"{model_dir}/moe.json"):
        return None
    spec = json.load(open(f"{model_dir}/moe.json"))
    experts = {n: (lgb.Booster(model_file=f"{model_dir}/moe_{n}.txt"), cols) for n, cols in spec["experts"].items()}
    meta = lgb.Booster(model_file=f"{model_dir}/moe_meta.txt")

    use = CONFIG.get("inference", "meta")

    def rank(f):
        f = f.assign(grp="q")
        if use == "full":
            b, cols = experts["full"]
            s = b.predict(f.reindex(columns=cols).astype(np.float32).values)
        else:
            scores = {n: b.predict(f.reindex(columns=cols).astype(np.float32).values)
                      for n, (b, cols) in experts.items()}
            X2 = level2_frame(f, scores).reindex(columns=spec["meta"])
            s = meta.predict(X2.astype(np.float32).values)
        if "pop" in f.columns:
            s = R.blend_pop(s, f["pop"].values)
        return np.argsort(-s, kind="stable")
    return rank
