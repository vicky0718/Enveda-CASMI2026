"""EDA pass 5c — the realistic candidate pool (training structures ∪ COCONUT 2.0).

Coverage of the test-like np-examples by COCONUT, same-formula isomer density inside the mass
window (the ranking difficulty), and the biosynthetic-pathway mix (NPClassifier) of test-like
molecules versus the pool.

Requires data/external/coconut/coconut_csv_lite-09-2026.csv (Kaggle dataset
pri2si17/coconut-natural-products-2026-09, CC0; COCONUT itself is CC BY 4.0).

    PYTHONPATH=src python scripts/eda/05c_candidate_pool.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from casmi.io import load_meta
from casmi.paths import INTERIM, ROOT, STATS
from casmi.plot import KEY, OTHER, SERIES, save

COCONUT = ROOT / "data" / "external" / "coconut" / "coconut_csv_lite-09-2026.csv"
POOL_CACHE = INTERIM / "pool_train_coconut.parquet"


def load_pool(m: pd.DataFrame) -> pd.DataFrame:
    if POOL_CACHE.exists():
        return pd.read_parquet(POOL_CACHE)
    cols = ["identifier", "canonical_smiles", "standard_inchi_key", "exact_molecular_weight", "molecular_formula",
            "formal_charge", "np_classifier_pathway", "np_classifier_superclass", "contains_sugar", "murcko_framework"]
    co = pd.read_csv(COCONUT, usecols=cols, low_memory=False)
    co = co[co.standard_inchi_key.notna()]
    co["ik14"] = co.standard_inchi_key.str[:14]
    co = co.drop_duplicates("ik14")
    co["src"] = "coconut"
    desc = pd.read_parquet(INTERIM / "structure_descriptors.parquet", columns=["normalized_smiles", "molecular_formula",
                                                                                 "exact_mass", "scaffold"])
    tr = desc.reset_index().rename(columns={"inchikey14": "ik14", "exact_mass": "exact_molecular_weight",
                                            "normalized_smiles": "canonical_smiles", "scaffold": "murcko_framework"})
    tr["src"] = "train"
    pool = pd.concat([tr, co[~co.ik14.isin(set(tr.ik14))]], ignore_index=True)
    pool["in_coconut"] = pool.ik14.isin(set(co.ik14))
    keep = ["ik14", "canonical_smiles", "molecular_formula", "exact_molecular_weight", "src", "in_coconut",
            "formal_charge", "np_classifier_pathway", "np_classifier_superclass", "contains_sugar", "murcko_framework"]
    pool = pool[keep]
    # carry COCONUT annotations onto training structures that are also in COCONUT
    ann = co.set_index("ik14")[["np_classifier_pathway", "np_classifier_superclass", "contains_sugar", "formal_charge"]]
    tr_mask = pool.src == "train"
    for col in ann.columns:
        pool.loc[tr_mask, col] = pool.loc[tr_mask, "ik14"].map(ann[col])
    pool.to_parquet(POOL_CACHE)
    return pool


def main():
    m = load_meta()
    pool = load_pool(m)
    stats = {"pool_size": int(len(pool)), "pool_from_train": int((pool.src == "train").sum()),
             "pool_coconut_only": int((pool.src == "coconut").sum())}

    # ---- coverage of each library's structures by COCONUT
    cov = {}
    in_co = set(pool.ik14[pool.in_coconut])
    for lib, s in m.groupby("ingest_lib").inchikey14:
        u = set(s)
        cov[lib] = round(len(u & in_co) / len(u), 4)
    stats["coconut_coverage_by_library"] = cov

    # ---- isomer density in the ±10 ppm window for test-like molecules
    pool = pool[pool.exact_molecular_weight.notna()].sort_values("exact_molecular_weight").reset_index(drop=True)
    mass = pool.exact_molecular_weight.values
    npk = m[m.ingest_lib == "enveda-np-examples"].drop_duplicates("inchikey14").set_index("inchikey14")
    p_idx = pool.set_index("ik14")
    rows = []
    for ik in npk.index:
        if ik not in p_idx.index:
            continue
        me = p_idx.loc[ik]
        mm, f = float(me.exact_molecular_weight), me.molecular_formula
        lo, hi = np.searchsorted(mass, mm * (1 - 10e-6)), np.searchsorted(mass, mm * (1 + 10e-6), "right")
        win = pool.iloc[lo:hi]
        same = win[win.molecular_formula == f]
        rows.append({"ik14": ik, "window": len(win), "same_formula": len(same),
                     "same_formula_scaffolds": same.murcko_framework.nunique(),
                     "pathway": me.np_classifier_pathway})
    iso = pd.DataFrame(rows)
    iso.to_csv(STATS / "np_examples_isomer_density.csv", index=False)
    stats["np_examples_window_10ppm"] = iso.window.describe(percentiles=[.25, .5, .75, .9]).round(1).to_dict()
    stats["np_examples_same_formula"] = iso.same_formula.describe(percentiles=[.25, .5, .75, .9]).round(1).to_dict()
    stats["share_window_candidates_same_formula"] = float(iso.same_formula.sum() / iso.window.sum())
    stats["random_rank_mrr_within_same_formula"] = float(np.mean([sum(1 / r for r in range(1, min(n, 25) + 1)) / n
                                                                 for n in iso.same_formula if n > 0]))

    # ---- pathway mix
    def pathway_share(s):
        s = s.fillna("unclassified").str.split("|").str[0]
        return s.value_counts(normalize=True)
    pw = pd.DataFrame({
        "np-examples (test-like)": pathway_share(iso.pathway),
        "COCONUT in test mass range 157–1159": pathway_share(pool[(pool.src == "coconut")
                                                                 & pool.exact_molecular_weight.between(157, 1159)].np_classifier_pathway),
    }).fillna(0).sort_values("np-examples (test-like)", ascending=False)
    pw.to_csv(STATS / "pathway_mix.csv")
    stats["pathway_mix"] = pw.round(4).to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"wspace": 0.55, "width_ratios": [1.1, 1]})
    top = pw.head(9)
    y = np.arange(len(top))[::-1]
    axes[0].barh(y + 0.19, top.iloc[:, 0] * 100, height=0.36, color=KEY["enveda-np-examples"], label=top.columns[0])
    axes[0].barh(y - 0.19, top.iloc[:, 1] * 100, height=0.36, color=OTHER, label=top.columns[1])
    axes[0].set_yticks(y, top.index)
    axes[0].grid(axis="y", visible=False)
    axes[0].set_xlabel("% of structures (NPClassifier pathway)")
    axes[0].set_title("Biosynthetic pathway mix")
    axes[0].legend(fontsize=8, loc="lower right")
    bins = np.unique(np.logspace(0, np.log10(max(iso.window.max(), 2) + 1), 30).astype(int))
    axes[1].hist(iso.window, bins=bins, histtype="step", linewidth=2, color=SERIES[0], label="all candidates in ±10 ppm")
    axes[1].hist(iso.same_formula, bins=bins, histtype="step", linewidth=2, color=SERIES[1], label="same formula as truth")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Candidates per test-like molecule (train ∪ COCONUT pool, log)")
    axes[1].set_ylabel("Molecules")
    axes[1].set_title("The ranking problem: same-formula isomers in the window")
    axes[1].legend(fontsize=8)
    save(fig, "26_candidate_pool_landscape")

    (STATS / "05c_candidate_pool.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
