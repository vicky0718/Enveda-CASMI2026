"""PubChem channel analysis on features_pc.parquet (regimes C1, C2, C2P; no generator).

One out-of-fold ranker is trained on all PubChem rows; the candidate budget N and the gate τ (admit PubChem
rows only when the molecule's best library-hit similarity is below τ) are then swept by masking rows.
Popularity tie-break (POP_LAMBDA) is applied as in inference.

    PYTHONPATH=src python scripts/eval/pc_analysis.py [--save]
"""

import sys

import numpy as np
import pandas as pd

from casmi import rank as R
from casmi.paths import ROOT

ART = ROOT / "data" / "artifacts"
EVAL = ART / "eval"


def blended(f, s):
    out = np.empty(len(f))
    for _, g in f.groupby("grp", sort=False):
        out[g.index] = R.blend_pop(s[g.index], g["pop"].values)
    return out


def main():
    raw = pd.read_parquet(EVAL / "features_pc.parquet")
    f = R.prepare(raw, ("C1", "C2", "C2P"))
    feats = R.feature_cols(f)
    s = blended(f, R.cv(f, feats))
    f["s"] = s
    truth_pc = f[(f.regime == "C2P") & (f.label == 1)]
    print("C2P truth reachable among top-N PubChem rows:")
    for n in (5, 10, 20, 30):
        cov = truth_pc[truth_pc.pc_rank < n].groupby("panel").qkey.nunique() / \
            f[f.regime == "C2P"].groupby("panel").qkey.nunique()
        print(f"  N={n}:", cov.round(3).to_dict())
    rows = []
    for n in (0, 5, 10, 20, 30):
        for tau in (None, 0.9, 0.8, 0.7, 0.6, 0.5):
            drop = (f.is_pc.values == 1) & (f.pc_rank.values >= n)
            if tau is not None:
                drop |= (f.is_pc.values == 1) & (f.q_best_hit.values >= tau)
            sc = np.where(drop, -1e9, f.s.values)
            r = R.mrr_of(f, sc).groupby(["panel", "regime"]).mrr.mean()
            rows.append({"N": n, "tau": tau, **{f"{p}_{g}": round(v, 4) for (p, g), v in r.items() if p in "AC"}})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(t.to_string(index=False))
    if "--save" in sys.argv:
        out = ART / "submit_pc"
        out.mkdir(exist_ok=True)
        R.fit_save(f, feats, out)
        print("saved", out)


if __name__ == "__main__":
    main()
