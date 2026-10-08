"""Train / evaluate the multi-model ranker (casmi.moe) on harness features, no generator, C1/C2.

Reports, out of fold (molecules held out): each expert alone, the single full ranker, the stacked
meta-ranker, and a tuned rank-fusion (RRF) baseline.

    PYTHONPATH=src python scripts/eval/train_moe.py [--features=PATH] [--save]
"""

import sys

import numpy as np
import pandas as pd

from casmi import moe
from casmi import rank as R
from casmi.paths import ROOT

ART = ROOT / "data" / "artifacts"


def blended(f, s):
    out = np.empty(len(f))
    for _, g in f.groupby("grp", sort=False):
        out[g.index] = R.blend_pop(s[g.index], g["pop"].values) if "pop" in g else s[g.index]
    return out


def main():
    path = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--features=")),
                str(ART / "eval" / "features.parquet"))
    raw = pd.read_parquet(path)
    raw = raw[raw.is_gen == 0]
    regimes = next((a.split("=")[1].split(",") for a in sys.argv if a.startswith("--regimes=")), ["C1", "C2"])
    f = R.prepare(raw, tuple(regimes))
    l1, meta = moe.cv(f)
    rows = {}
    for n, s in l1.items():
        rows[f"expert:{n}"] = R.mrr_of(f, blended(f, s))
    rows["meta (stacked)"] = R.mrr_of(f, blended(f, meta))
    if "C2H" in regimes:  # today's ranker: one model, trained without the analog-thinned regime
        feats = R.feature_cols(f)
        cur = np.zeros(len(f))
        for k in range(5):
            tr = f[(f.fold != k) & f.panel.isin(["A", "C"]) & f.regime.isin(["C1", "C2"])]
            va = f[f.fold == k]
            cur[va.index] = R._train(tr, feats).predict(va[feats].astype(np.float32))
        rows["full, trained C1+C2 only (current)"] = R.mrr_of(f, blended(f, cur))
    # rank-fusion baseline: reciprocal-rank fusion of the specialist experts
    ranks = {n: pd.Series(s, index=f.index).groupby(f.grp).rank(ascending=False) for n, s in l1.items()}
    for k in (10, 30):
        rrf = sum(1.0 / (k + ranks[n]) for n in ("library", "analog", "spectral", "full"))
        rows[f"rrf k={k}"] = R.mrr_of(f, blended(f, rrf.values))
    tab = pd.DataFrame({name: r.groupby(["panel", "regime"]).mrr.mean() for name, r in rows.items()}).T
    pd.set_option("display.width", 200)
    print(tab.round(4).to_string())
    w = {name: {p: round(0.16 * tab.loc[name, (p, "C1")] + 0.30 * tab.loc[name, (p, "C2")], 4) for p in "AC"}
         for name in tab.index}
    print("weighted (0.16·C1 + 0.30·C2):", w)
    if "C2H" in regimes:  # class 2 split between molecules with and without close library relatives
        wh = {name: {p: round(0.16 * tab.loc[name, (p, "C1")] + 0.15 * tab.loc[name, (p, "C2")]
                              + 0.15 * tab.loc[name, (p, "C2H")], 4) for p in "AC"} for name in tab.index}
        print("weighted (0.16·C1 + 0.15·C2 + 0.15·C2H):", wh)
    if "--save" in sys.argv:
        out = ART / "models_moe"
        out.mkdir(exist_ok=True)
        moe.fit_save(f, out)  # trained on the same regimes as evaluated
        (out / "casmi26_models.txt").write_text("multi-model ranker (casmi.moe)\n")
        print("saved", out)


if __name__ == "__main__":
    main()
