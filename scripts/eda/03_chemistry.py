"""EDA pass 3 — chemistry of the labelled structures.

On the ~276k unique 2D structures (InChIKey14): RDKit descriptors, NP-likeness,
element composition, chemical-space comparison between enveda-180 (synthetic),
public libraries and enveda-np-examples (test-like natural products), formula /
mass ambiguity (how many candidates a perfect formula or mass leaves), and
nearest-analog Tanimoto similarity (what a "novel structure" looks like
relative to the training set).

    PYTHONPATH=src python scripts/eda/03_chemistry.py
"""

import json

from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Contrib.NP_Score import npscorer

from casmi.io import load_meta
from casmi.paths import INTERIM, STATS
from casmi.plot import INK2, KEY, MUTED, OTHER, save

RDLogger.DisableLog("rdApp.*")
DESC_CACHE = INTERIM / "structure_descriptors.parquet"
GROUPS = {"enveda-180": KEY["enveda-180"], "public libraries": OTHER, "enveda-np-examples": KEY["enveda-np-examples"]}
_NP_MODEL = None


def _init():
    global _NP_MODEL
    RDLogger.DisableLog("rdApp.*")
    _NP_MODEL = npscorer.readNPModel()


def _describe(smi: str) -> dict:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {"valid": False}
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    cnt = pd.Series(atoms).value_counts().to_dict() if atoms else {}
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaf = ""
    return {
        "valid": True,
        "exact_mass": Descriptors.ExactMolWt(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "n_C": cnt.get("C", 0), "n_N": cnt.get("N", 0), "n_O": cnt.get("O", 0), "n_S": cnt.get("S", 0),
        "n_P": cnt.get("P", 0), "n_halogen": sum(cnt.get(x, 0) for x in ("F", "Cl", "Br", "I")),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "fsp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "stereocentres": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)),
        "formal_charge": Chem.GetFormalCharge(mol),
        "n_fragments": len(Chem.GetMolFrags(mol)),
        "np_likeness": npscorer.scoreMol(mol, _NP_MODEL),
        "scaffold": scaf,
    }


def compute_descriptors(structs: pd.DataFrame) -> pd.DataFrame:
    if DESC_CACHE.exists():
        return pd.read_parquet(DESC_CACHE)
    with Pool(4, initializer=_init) as pool:
        rows = pool.map(_describe, structs.normalized_smiles.tolist(), chunksize=500)
    d = pd.DataFrame(rows, index=structs.index)
    d = structs.join(d)
    d.to_parquet(DESC_CACHE)
    return d


def morgan(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


def main():
    m = load_meta()
    stats = {}
    lib_sets = m.groupby("inchikey14").ingest_lib.agg(lambda s: sorted(set(s)))
    structs = (m.drop_duplicates("inchikey14").set_index("inchikey14")[["normalized_smiles", "molecular_formula"]]
               .join(lib_sets.rename("libs")))
    structs["libs"] = structs.libs.map(lambda l: ",".join(l))
    d = compute_descriptors(structs)
    libs_list = d.libs.str.split(",")
    d["in_e180"] = libs_list.map(lambda l: "enveda-180" in l)
    d["in_np"] = libs_list.map(lambda l: "enveda-np-examples" in l)
    d["in_public"] = libs_list.map(lambda l: any(not x.startswith("enveda") for x in l))
    stats["invalid_smiles"] = int((~d.valid).sum())
    stats["multi_fragment_structures"] = int((d.n_fragments > 1).sum())
    stats["charged_structures"] = int((d.formal_charge != 0).sum())

    grp = {"enveda-180": d[d.in_e180], "public libraries": d[d.in_public], "enveda-np-examples": d[d.in_np]}
    desc_cols = ["exact_mass", "heavy_atoms", "np_likeness", "fsp3", "logp", "tpsa", "aromatic_rings",
                 "rings", "stereocentres", "hbd", "n_N", "n_O"]
    summary = pd.DataFrame({g: x[desc_cols].median() for g, x in grp.items()}).T
    summary["structures"] = [len(x) for x in grp.values()]
    summary["share_with_N"] = [float((x.n_N > 0).mean()) for x in grp.values()]
    summary["share_with_S"] = [float((x.n_S > 0).mean()) for x in grp.values()]
    summary["share_with_halogen"] = [float((x.n_halogen > 0).mean()) for x in grp.values()]
    summary["share_with_P"] = [float((x.n_P > 0).mean()) for x in grp.values()]
    summary["share_CHO_only"] = [float(((x.n_N + x.n_S + x.n_P + x.n_halogen) == 0).mean()) for x in grp.values()]
    summary["unique_scaffolds"] = [x.scaffold.nunique() for x in grp.values()]
    summary.to_csv(STATS / "chemistry_by_group.csv")
    stats["chemistry_by_group"] = summary.round(3).to_dict(orient="index")

    # ---------- F14: descriptor distributions (small multiples) ----------
    panels = [("exact_mass", "Monoisotopic mass (Da)", (100, 1300)), ("np_likeness", "NP-likeness score", (-4, 4)),
              ("fsp3", "Fraction sp3 carbon", (0, 1)), ("logp", "Crippen logP", (-6, 10)),
              ("aromatic_rings", "Aromatic rings", (0, 7)), ("stereocentres", "Stereocentres (potential)", (0, 20))]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), gridspec_kw={"hspace": 0.45, "wspace": 0.28})
    for ax, (col, lab, (lo, hi)) in zip(axes.flat, panels):
        integer = col in ("aromatic_rings", "stereocentres")
        bins = np.arange(lo, hi + 1) - 0.5 if integer else np.linspace(lo, hi, 50)
        for g, x in grp.items():
            ax.hist(x[col].clip(lo, hi), bins=bins, density=True, histtype="step", linewidth=2, color=GROUPS[g], label=g)
        ax.set_title(lab)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Chemical space: test-like natural products differ sharply from the timsTOF-matched enveda-180",
                 x=0.125, ha="left", fontsize=12, fontweight="bold")
    save(fig, "14_descriptor_distributions")

    # ---------- F15: element composition ----------
    comp_cols = ["share_CHO_only", "share_with_N", "share_with_S", "share_with_halogen", "share_with_P"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(comp_cols))
    w = 0.26
    for i, g in enumerate(grp):
        ax.bar(x + (i - 1) * w, summary.loc[g, comp_cols] * 100, width=w, color=GROUPS[g], label=g, edgecolor="#fcfcfb")
        for xi, v in zip(x, summary.loc[g, comp_cols] * 100):
            ax.text(xi + (i - 1) * w, v + 1, f"{v:.0f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, ["C/H/O only", "contains N", "contains S", "contains halogen", "contains P"])
    ax.set_ylabel("% of structures")
    ax.grid(axis="x", visible=False)
    ax.legend()
    ax.set_title("Element composition: natural products are mostly C/H/O; screening compounds are N-rich and halogenated")
    save(fig, "15_element_composition")

    # ---------- formula / mass ambiguity ----------
    f_count = d.groupby("molecular_formula").size()
    np_ = d[d.in_np]
    amb = pd.DataFrame({
        "enveda-np-examples": np_.molecular_formula.map(f_count).describe(percentiles=[.25, .5, .75, .9]),
        "public libraries": d[d.in_public].molecular_formula.map(f_count).describe(percentiles=[.25, .5, .75, .9]),
    })
    amb.to_csv(STATS / "formula_isomer_counts.csv")
    masses = np.sort(d.exact_mass.dropna().values)

    def n_within(mass, ppm):
        tol = mass * ppm / 1e6
        return np.searchsorted(masses, mass + tol, "right") - np.searchsorted(masses, mass - tol, "left")

    for ppm in (2, 5, 10):
        stats[f"np_examples_median_train_structures_within_{ppm}ppm"] = float(np.median([n_within(v, ppm) for v in np_.exact_mass]))
    stats["np_examples_median_train_isomers_same_formula"] = float(np_.molecular_formula.map(f_count).median())
    stats["share_structures_unique_formula_in_train"] = float((d.molecular_formula.map(f_count) == 1).mean())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"wspace": 0.3})
    bins = np.unique(np.logspace(0, 3.3, 30).astype(int))
    axes[0].hist(d[d.in_public].molecular_formula.map(f_count), bins=bins, density=True, histtype="step", color=OTHER, linewidth=2, label="public libraries")
    axes[0].hist(np_.molecular_formula.map(f_count), bins=bins, density=True, histtype="step", color=KEY["enveda-np-examples"], linewidth=2, label="enveda-np-examples")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Training structures sharing the exact molecular formula (log)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Isomers in train: a correct formula still leaves many candidates")
    axes[0].legend()
    ppms = [1, 2, 3, 5, 10, 20]
    for g, x, c in [("public libraries", d[d.in_public].sample(5000, random_state=0), OTHER),
                    ("enveda-np-examples", np_, KEY["enveda-np-examples"])]:
        med = [np.median([n_within(v, p) for v in x.exact_mass]) for p in ppms]
        axes[1].plot(ppms, med, marker="o", markersize=7, color=c, label=g)
    axes[1].set_xscale("log")
    axes[1].set_xticks(ppms, [str(p) for p in ppms])
    axes[1].set_xlabel("Mass tolerance (ppm)")
    axes[1].set_ylabel("Median training structures in window")
    axes[1].set_title("Candidates by neutral mass alone (train only; PubChem is far larger)")
    axes[1].legend()
    save(fig, "16_formula_mass_ambiguity")

    # ---------- nearest-analog similarity ----------
    rng = np.random.default_rng(0)
    e180_idx = d.index[d.in_e180]
    pub_only = d.index[d.in_public]
    fp_all = dict(zip(d.index, morgan(d.normalized_smiles)))
    fps_e180 = [fp_all[i] for i in e180_idx]
    fps_pub = [fp_all[i] for i in pub_only]
    pub_pos = {k: i for i, k in enumerate(pub_only)}

    def nn(query_keys, pool_fps, exclude_pos=None):
        out = []
        for k in query_keys:
            sims = np.array(DataStructs.BulkTanimotoSimilarity(fp_all[k], pool_fps))
            if exclude_pos is not None and k in exclude_pos:
                sims[exclude_pos[k]] = -1
            out.append(sims.max())
        return np.array(out)

    q_np = list(np_.index)
    q_pub = list(rng.choice(pub_only, 2000, replace=False))
    sim = {
        "np-examples → nearest enveda-180": nn(q_np, fps_e180),
        "np-examples → nearest other public structure": nn(q_np, fps_pub, pub_pos),
        "public (random 2k) → nearest other public": nn(q_pub, fps_pub, pub_pos),
    }
    sim_df = pd.DataFrame({k: pd.Series(v).describe(percentiles=[.1, .25, .5, .75, .9]) for k, v in sim.items()})
    sim_df.to_csv(STATS / "nearest_analog_tanimoto.csv")
    stats["nearest_analog_tanimoto_median"] = {k: float(np.median(v)) for k, v in sim.items()}
    stats["share_np_with_public_analog_ge_0.7"] = float((sim["np-examples → nearest other public structure"] >= 0.7).mean())

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    bins = np.linspace(0, 1, 41)
    for (k, v), c in zip(sim.items(), [KEY["enveda-180"], KEY["enveda-np-examples"], MUTED]):
        ax.hist(v, bins=bins, density=True, histtype="step", linewidth=2, color=c, label=f"{k} (median {np.median(v):.2f})")
    ax.set_xlabel("Max Tanimoto similarity to the nearest *different* training structure (Morgan r=2, 2048 bits)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Novel natural products have close analogs in public libraries, not in enveda-180")
    save(fig, "17_nearest_analog_similarity")

    # ---------- chemical-space map (SVD of fingerprints) ----------
    from sklearn.decomposition import TruncatedSVD

    sample = (list(rng.choice(e180_idx, 6000, replace=False)) + list(rng.choice(pub_only, 6000, replace=False)) + q_np)
    X = np.zeros((len(sample), 2048), dtype=np.float32)
    for i, k in enumerate(sample):
        DataStructs.ConvertToNumpyArray(fp_all[k], X[i])
    Z = TruncatedSVD(2, random_state=0).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sl = [slice(0, 6000), slice(6000, 12000), slice(12000, None)]
    for (g, c), s, size in zip(GROUPS.items(), sl, [6, 6, 22]):
        ax.scatter(Z[s, 0], Z[s, 1], s=size, color=c, alpha=0.45 if size < 10 else 0.95, label=g, linewidth=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("SVD 1 of Morgan fingerprints"); ax.set_ylabel("SVD 2")
    ax.legend(markerscale=2)
    ax.set_title("Chemical-space map (6k + 6k sampled, all 250 NP examples)")
    save(fig, "18_chemical_space_map")

    # Top scaffolds among public natural-product-like structures.
    top_scaf = d[d.in_public & (d.np_likeness > 1)].scaffold.replace("", "<acyclic>").value_counts().head(20)
    top_scaf.to_csv(STATS / "top_scaffolds_public_np_like.csv")

    (STATS / "03_chemistry.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
