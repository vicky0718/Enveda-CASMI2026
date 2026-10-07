"""Error analysis of the ranker on the validation panels: what outranks the truth, and why.

For every (molecule, regime) the out-of-fold ranker scores (molecules held out) are used to find:
  * error bucket — truth not in the list / rank > 25 / 6–25 / 2–5 / 1
  * winner type — what was ranked first when it was not the truth:
      gen            a generated (edited-analog) structure
      iso_lib        same molecular formula as the truth, structure has library spectra (train)
      iso_coconut    same formula, COCONUT-only structure (no spectra anywhere)
      other_formula  different formula inside the mass window
  * which features favour the winner over the truth (sign and size of the gap)
Outputs: printed confusion-style tables, data/artifacts/eval/errors.parquet (one row per case) and
reports/eval/error_cases.csv (worst cases with SMILES for manual review).

    PYTHONPATH=src python scripts/eval/error_analysis.py
"""

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from casmi import rank as R
from casmi.paths import ROOT

RDLogger.DisableLog("rdApp.*")
ART = ROOT / "data" / "artifacts"
EVAL = ART / "eval"
REP = ROOT / "reports" / "eval"
REP.mkdir(parents=True, exist_ok=True)
FEATS_SHOW = ["direct", "analog", "analog_max", "analog_top5", "analog_tims", "tmax", "t_wmean", "frag",
              "mass_err_ppm", "gen_sim"]


def formula(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return CalcMolFormula(m) if m is not None else None


def main():
    f = R.prepare(pd.read_parquet(EVAL / "features.parquet"))
    f["s"] = R.cv(f, R.feature_cols(f))
    pool = pd.read_parquet(ART / "pool" / "pool.parquet", columns=["formula", "src"])
    f["formula"] = np.where(f.pool_row >= 0, pool.formula.values[f.pool_row.clip(lower=0)], None)
    f["src"] = np.where(f.pool_row >= 0, pool.src.values[f.pool_row.clip(lower=0)], "gen")

    cases = []
    for (key, regime), g in f.groupby(["qkey", "regime"], sort=False):
        g = g.sort_values("s", ascending=False).drop_duplicates("key")
        g = g.reset_index(drop=True)
        hit = np.flatnonzero(g.label.values == 1)
        rank = int(hit[0]) + 1 if len(hit) else None
        truth = g.iloc[hit[0]] if len(hit) else None
        top = g.iloc[0]
        tf = truth.formula if truth is not None else None
        if truth is not None and tf is None:
            tf = formula(truth.smiles)
        wf = top.formula if top.formula is not None else formula(top.smiles)
        if rank == 1:
            wtype = "truth"
        elif top.is_gen:
            wtype = "gen"
        elif tf is not None and wf == tf:
            wtype = "iso_lib" if top.src == "train" else "iso_coconut"
        else:
            wtype = "other_formula"
        bucket = ("not_in_list" if rank is None else "1" if rank == 1 else "2-5" if rank <= 5
                  else "6-25" if rank <= 25 else ">25")
        row = {"qkey": key, "regime": regime, "panel": g.panel.iloc[0], "rank": rank, "bucket": bucket,
               "winner": wtype, "n_cand": len(g), "n_gen": int(g.is_gen.sum()),
               "n_same_formula": int((g.formula == tf).sum()) if tf else 0,
               "truth_smiles": None if truth is None else truth.smiles, "winner_smiles": top.smiles,
               "truth_formula": tf, "winner_formula": wf}
        for c in FEATS_SHOW:
            if c in g.columns:
                row[f"t_{c}"] = None if truth is None else float(truth[c])
                row[f"w_{c}"] = float(top[c])
        cases.append(row)
    E = pd.DataFrame(cases)
    E.to_parquet(EVAL / "errors.parquet")

    pd.set_option("display.width", 200)
    for panel in ("A", "C"):
        e = E[E.panel == panel]
        print(f"\n=== panel {panel}: error bucket (share of molecules) ===")
        print(pd.crosstab(e.regime, e.bucket, normalize="index").round(3)
              .reindex(columns=["1", "2-5", "6-25", ">25", "not_in_list"], fill_value=0))
        print(f"--- panel {panel}: what is ranked first (share of molecules) ===")
        print(pd.crosstab(e.regime, e.winner, normalize="index").round(3))
        w = e[(e.winner != "truth") & e["rank"].notna()]
        print(f"--- panel {panel}: truth in list but beaten — mean feature value, truth vs winner ===")
        gap = pd.DataFrame({c: [w[f"t_{c}"].mean(), w[f"w_{c}"].mean(), (w[f"w_{c}"] > w[f"t_{c}"]).mean()]
                            for c in FEATS_SHOW if f"t_{c}" in w.columns},
                           index=["truth", "winner", "P(winner>truth)"]).T.round(3)
        print(gap)
        print(f"--- panel {panel}: beaten cases by winner type × regime (counts) ===")
        print(pd.crosstab(w.winner, w.regime))
    worst = E[(E.winner != "truth")].sort_values(["panel", "regime", "rank"], na_position="first")
    worst.to_csv(REP / "error_cases.csv", index=False)
    print("\nwrote", REP / "error_cases.csv", len(worst), "cases")


if __name__ == "__main__":
    main()
