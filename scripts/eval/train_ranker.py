"""LightGBM lambdarank over harness features (data/artifacts/eval/features.parquet).

Folds are held out by molecule (query key), never by row. Reports out-of-fold MRR@25 per
panel × regime vs the heuristic, then fits on everything and writes the submission rankers.

    PYTHONPATH=src python scripts/eval/train_ranker.py [--regimes=C1,C2] [--train-panels=A,C] [--save]

Out-of-fold scores are reported for every panel, but models are fit only on --train-panels
(B, the synthetic-like enveda-180 panel, teaches library-provenance shortcuts; see harness notes).
"""

import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from casmi import pipeline as P
from casmi.paths import ROOT

ART = ROOT / "data" / "artifacts"
EVAL = ART / "eval"
BASE_FEATS = ["direct", "direct_n", "analog", "analog_max", "tmax", "mass_err_ppm", "frag", "analog_tims",
              "analog_top5", "t_wmean", "is_gen", "gen_sim", "gen_nsrc"]
PARAMS = dict(objective="lambdarank", metric="map", eval_at=[25], learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=50, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
              lambdarank_truncation_level=25, verbose=-1, seed=0, deterministic=True, num_threads=4)
ROUNDS = 300


def mrr_of(f, score):
    out = []
    for (k, r), g in f.assign(s=score).groupby(["qkey", "regime"], sort=False):
        o = np.argsort(-g.s.values, kind="stable")
        ranked = list(dict.fromkeys(g.key.values[o]))[:25]
        out.append({"qkey": k, "regime": r, "panel": g.panel.iloc[0],
                    "mrr": 1.0 / (ranked.index(k) + 1) if k in ranked else 0.0})
    return pd.DataFrame(out)


def main():
    regimes = next((a.split("=")[1].split(",") for a in sys.argv if a.startswith("--regimes=")), ["C1", "C2", "C3"])
    f = pd.read_parquet(EVAL / "features.parquet")
    panels = next((a.split("=")[1].split(",") for a in sys.argv if a.startswith("--train-panels=")), ["A", "C"])
    f = f[f.regime.isin(regimes)].reset_index(drop=True)
    feats = [c for c in BASE_FEATS + ["fp", "fp_rank"] if c in f.columns]
    feats += [c for c in f.columns if c.endswith("_gap") or c.endswith("_rk")] + ["n_cand"]
    f["grp"] = f.qkey + "|" + f.regime
    mols = f.qkey.unique()
    rng = np.random.default_rng(0)
    fold_of = dict(zip(mols, rng.integers(0, 5, len(mols))))
    f["fold"] = f.qkey.map(fold_of)
    oof = np.zeros(len(f))
    for k in range(5):
        tr, va = f[(f.fold != k) & f.panel.isin(panels)], f[f.fold == k]
        tr = tr[tr.groupby("grp").label.transform("max") > 0]  # queries whose answer is in the pool
        ds = lgb.Dataset(tr[feats].astype(np.float32), tr.label, group=tr.groupby("grp", sort=False).size().values)
        b = lgb.train(PARAMS, ds, ROUNDS)
        oof[va.index] = b.predict(va[feats].astype(np.float32))
    heur = np.zeros(len(f))
    for _, g in f.groupby("grp", sort=False):
        o = P.heuristic_rank(g)
        heur[g.index[o]] = -np.arange(len(g))
    rep = mrr_of(f, oof).merge(mrr_of(f, heur).rename(columns={"mrr": "mrr_heur"}), on=["qkey", "regime", "panel"])
    print("train panels", panels)
    tab = rep.groupby(["panel", "regime"])[["mrr", "mrr_heur"]].mean()
    print(tab.round(4))
    # class-share weighted estimate (forum algebra: f1 ~ .16, f2 ~ .27-.45, f3 ~ .39-.55)
    for f1, f2, f3 in ((.16, .30, .54), (.16, .45, .39)):
        est = {p: round(sum(w * tab.loc[(p, r), "mrr"] for w, r in ((f1, "C1"), (f2, "C2"), (f3, "C3"))
                            if (p, r) in tab.index), 4) for p in ("A", "C")}
        print(f"weighted (f1={f1}, f2={f2}, f3={f3}):", est)
    rep.to_parquet(EVAL / "ranker_oof.parquet")
    if "--save" in sys.argv:
        out = ART / "submit"
        out.mkdir(exist_ok=True)
        tr = f[(f.groupby("grp").label.transform("max") > 0) & f.panel.isin(panels)]
        for seed in range(3):
            ds = lgb.Dataset(tr[feats].astype(np.float32), tr.label, group=tr.groupby("grp", sort=False).size().values)
            lgb.train({**PARAMS, "seed": seed}, ds, ROUNDS).save_model(str(out / f"ranker{seed}.txt"))
        (out / "ranker_features.json").write_text(json.dumps(feats))
        print("saved", out)


if __name__ == "__main__":
    main()
