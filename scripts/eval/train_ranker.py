"""LightGBM lambdarank over harness features (data/artifacts/eval/features.parquet); see casmi.rank.

    PYTHONPATH=src python scripts/eval/train_ranker.py [--regimes=C1,C2,C3] [--train-panels=A,C] [--nogen] [--save]

Out-of-fold scores are reported for every panel, but models are fit only on --train-panels
(B, the synthetic-like enveda-180 panel, teaches library-provenance shortcuts).
"""

import sys

import numpy as np
import pandas as pd

from casmi import pipeline as P
from casmi import rank as R
from casmi.paths import ROOT

ART = ROOT / "data" / "artifacts"
EVAL = ART / "eval"


def arg(name, default):
    return next((a.split("=")[1].split(",") for a in sys.argv if a.startswith(f"--{name}=")), default)


def main():
    panels = arg("train-panels", ["A", "C"])
    raw = pd.read_parquet(EVAL / "features.parquet")
    if "--nogen" in sys.argv:  # ranker for the generator-free submission (LB: generator costs 0.017)
        raw = raw[raw.is_gen == 0]
    f = R.prepare(raw, arg("regimes", ["C1", "C2"] if "--nogen" in sys.argv else ["C1", "C2", "C3"]))
    feats = R.feature_cols(f)
    oof = R.cv(f, feats, panels)
    heur = np.zeros(len(f))
    for _, g in f.groupby("grp", sort=False):
        heur[g.index[P.heuristic_rank(g)]] = -np.arange(len(g))
    rep = R.mrr_of(f, oof).merge(R.mrr_of(f, heur).rename(columns={"mrr": "mrr_heur"}),
                                 on=["qkey", "regime", "panel"])
    print("train panels", panels)
    print(R.report(rep, ("mrr", "mrr_heur")))
    rep.to_parquet(EVAL / "ranker_oof.parquet")
    if "--save" in sys.argv:
        out = ART / "submit"
        out.mkdir(exist_ok=True)
        R.fit_save(f, feats, out, panels)
        print("saved", out)


if __name__ == "__main__":
    main()
