"""EDA pass 1 — spectrum-level metadata (no peak arrays).

Covers library composition and overlap, adducts, ionisation mode, instruments,
collision energy, precursor-mass error (label quality), peak counts, and the
provenance of the placeholder test file. Writes figures and stats CSV/JSON
into reports/eda/.

    PYTHONPATH=src python scripts/eda/01_metadata.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from casmi.io import load_meta, load_test
from casmi.meta import instrument_family
from casmi.paths import STATS
from casmi.plot import INK2, KEY, MUTED, OTHER, SEQ, SERIES, hbar, save
from matplotlib.colors import LinearSegmentedColormap

TEST_ADDUCTS = ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
                "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", ["#fcfcfb"] + SEQ)


def main():
    m = load_meta()
    t = load_test()
    t["num_peaks"] = t.ms2_mzs.map(len)
    stats: dict = {}
    libs = m.ingest_lib.value_counts().index.tolist()

    # ---------- 1. library composition ----------
    comp = m.groupby("ingest_lib").agg(
        spectra=("inchikey14", "size"), structures=("inchikey14", "nunique"),
        formulas=("molecular_formula", "nunique"),
    ).loc[libs]
    comp["spectra_per_structure"] = comp.spectra / comp.structures
    comp.to_csv(STATS / "library_composition.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"wspace": 0.55})
    hbar(axes[0], libs, comp.spectra.values, highlight=KEY)
    axes[0].set_title("Spectra per library")
    order = comp.structures.sort_values(ascending=False).index.tolist()
    hbar(axes[1], order, comp.structures.loc[order].values, highlight=KEY)
    axes[1].set_title("Unique 2D structures (InChIKey14) per library")
    save(fig, "01_library_composition")
    stats["n_spectra"] = int(len(m))
    stats["n_structures"] = int(m.inchikey14.nunique())

    # ---------- 2. structure overlap between libraries ----------
    S = {l: set(m.inchikey14[m.ingest_lib == l]) for l in libs}
    ov = pd.DataFrame([[len(S[a] & S[b]) / len(S[a]) for b in libs] for a in libs], index=libs, columns=libs)
    ov.to_csv(STATS / "library_overlap.csv")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(ov.values, cmap=SEQ_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(len(libs)), libs, rotation=45, ha="right")
    ax.set_yticks(range(len(libs)), libs)
    ax.grid(False)
    for i in range(len(libs)):
        for j in range(len(libs)):
            v = ov.values[i, j]
            ax.text(j, i, f"{v:.2f}" if v >= 0.005 else "·", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.55 else INK2)
    ax.set_title("Share of the row library's structures also present in the column library")
    fig.colorbar(im, ax=ax, shrink=0.75)
    save(fig, "02_library_overlap")
    nlib = m.groupby("inchikey14").ingest_lib.nunique()
    exclusive = {l: len(S[l] - set().union(*[S[o] for o in libs if o != l])) / len(S[l]) for l in libs}
    pd.Series(exclusive, name="exclusive_share").to_csv(STATS / "library_exclusive_share.csv")
    stats["structures_in_one_library_share"] = float((nlib == 1).mean())

    # ---------- 3. spectra per structure ----------
    spp = m.groupby("inchikey14").size()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    vc = nlib.value_counts().sort_index()
    axes[0].bar(vc.index, vc.values, color=SERIES[0], width=0.72, edgecolor="#fcfcfb")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Number of libraries containing the structure")
    axes[0].set_ylabel("Structures (log)")
    axes[0].set_title("Most structures come from a single library")
    bins = np.unique(np.logspace(0, np.log10(spp.max() + 1), 40).astype(int))
    axes[1].hist(spp, bins=bins, color=SERIES[0], edgecolor="#fcfcfb", linewidth=0.5)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Spectra per structure (log)")
    axes[1].set_ylabel("Structures (log)")
    axes[1].set_title("Spectra per structure (median %d, p99 %d)" % (spp.median(), spp.quantile(0.99)))
    save(fig, "03_spectra_per_structure")
    stats["spectra_per_structure"] = spp.describe(percentiles=[.5, .9, .99]).round(2).to_dict()

    # Per-"molecule" groups in train (same library + structure) mirror test molecule_id grouping.
    grp = m.groupby(["ingest_lib", "inchikey14"]).size()
    tg = t.groupby("molecule_id").size()
    per_mol = pd.DataFrame({
        "median_spectra_per_molecule": grp.groupby(level=0).median(),
        "max_spectra_per_molecule": grp.groupby(level=0).max(),
    }).loc[libs]
    per_mol.loc["test (placeholder)"] = [tg.median(), tg.max()]
    per_mol.to_csv(STATS / "spectra_per_molecule_group.csv")

    # ---------- 4. adducts & ionisation ----------
    m["adduct_grp"] = m.adduct.where(m.adduct.isin(TEST_ADDUCTS), "other (not in test)")
    ad = pd.crosstab(m.ingest_lib, m.adduct_grp, normalize="index").loc[libs]
    cols = [a for a in TEST_ADDUCTS if a in ad.columns] + ["other (not in test)"]
    ad = ad[cols]
    tad = t.adduct.value_counts(normalize=True).reindex(cols, fill_value=0)
    ad.loc["test (placeholder)"] = tad.values
    ad.to_csv(STATS / "adduct_share_by_library.csv")
    fig, ax = plt.subplots(figsize=(10.5, 6))
    im = ax.imshow(ad.values, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right")
    ax.set_yticks(range(len(ad)), ad.index)
    ax.grid(False)
    for i in range(ad.shape[0]):
        for j in range(ad.shape[1]):
            v = ad.values[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7.5, color="white" if v > 0.55 else INK2)
    ax.set_title("Adduct mix per library (share of spectra) — columns are the 10 test adducts")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, "04_adducts_by_library")
    stats["n_adducts_train"] = int(m.adduct.nunique())
    stats["share_spectra_in_test_adducts"] = float(m.adduct.isin(TEST_ADDUCTS).mean())
    stats["top_adducts_train"] = m.adduct.value_counts().head(15).to_dict()
    stats["test_placeholder_adducts"] = t.adduct.value_counts().to_dict()

    ion = pd.crosstab(m.ingest_lib, m.ionization_mode, normalize="index").loc[libs]
    ion.loc["test (placeholder)"] = t.ionization_mode.value_counts(normalize=True).reindex(ion.columns).values
    ion.to_csv(STATS / "ionization_by_library.csv")

    # ---------- 5. instruments ----------
    m["instrument_family"] = m.instrument_type.astype(object).map(instrument_family)
    inst = pd.crosstab(m.ingest_lib, m.instrument_family, normalize="index").loc[libs]
    fam_order = ["timsTOF", "Orbitrap / FT", "Q-TOF / TOF", "Triple quad", "Ion trap", "other", "unknown"]
    inst = inst.reindex(columns=[f for f in fam_order if f in inst.columns], fill_value=0)
    inst.to_csv(STATS / "instrument_family_by_library.csv")
    (m.instrument_type.astype(object).fillna("<null>").value_counts().rename_axis("instrument_type")
     .to_frame("spectra").assign(family=lambda d: d.index.map(instrument_family))
     .to_csv(STATS / "instrument_type_raw_values.csv"))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    left = np.zeros(len(inst))
    fam_colors = {"timsTOF": SERIES[0], "Orbitrap / FT": SERIES[1], "Q-TOF / TOF": SERIES[2],
                  "Triple quad": SERIES[3], "Ion trap": SERIES[4], "other": "#898781", "unknown": OTHER}
    y = np.arange(len(inst))[::-1]
    for f in inst.columns:
        ax.barh(y, inst[f].values, left=left, color=fam_colors[f], label=f, height=0.72, edgecolor="#fcfcfb", linewidth=1.5)
        left += inst[f].values
    ax.set_yticks(y, inst.index)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=len(inst.columns), loc="upper left", bbox_to_anchor=(0, -0.08))
    ax.set_title(f"Instrument family per library ({m.instrument_type.nunique()} raw free-text values mapped)")
    save(fig, "05_instrument_family")
    stats["n_instrument_strings"] = int(m.instrument_type.nunique())

    # ---------- 6. collision energy ----------
    ceu = pd.crosstab(m.ingest_lib, m.collision_energy_orig_units, normalize="index").loc[libs]
    ceu["has_ce_ev"] = m.groupby("ingest_lib").ce_mean.apply(lambda s: s.notna().mean()).loc[libs]
    ceu["multi_energy_merged"] = m.groupby("ingest_lib").ce_n.apply(lambda s: (s > 1).mean()).loc[libs]
    ceu.to_csv(STATS / "collision_energy_units_by_library.csv")
    t_ce = t.collision_energy_ev.map(lambda a: np.mean(np.abs(a)) if len(a) else np.nan)
    t_multi = t.collision_energy_ev.map(len) > 1
    stats["test_ce_values"] = t.collision_energy_orig.value_counts().to_dict()
    stats["test_multi_energy_share"] = float(t_multi.mean())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.45})
    hbar(axes[0], libs, (ceu.has_ce_ev * 100).values, fmt="{:.0f}%", highlight=KEY)
    axes[0].set_title("Spectra with a usable collision_energy_ev (%)")
    axes[0].set_xlim(0, 118)
    bins = np.arange(0, 125, 2.5)
    for lib, c in [("pluskal_ms2", OTHER), ("riken", "#898781")]:
        axes[1].hist(m.ce_mean[m.ingest_lib == lib].dropna().clip(upper=122), bins=bins, density=True,
                     histtype="step", color=c, linewidth=1.5, label=lib)
    for lib in ["enveda-180", "enveda-np-examples"]:
        axes[1].hist(m.ce_mean[m.ingest_lib == lib].dropna(), bins=bins, density=True,
                     histtype="step", color=KEY[lib], linewidth=2, label=lib)
    axes[1].hist(t_ce.dropna(), bins=bins, density=True, histtype="step", color=KEY["test"], linewidth=2, label="test (placeholder)")
    axes[1].set_xlabel("Mean collision energy per spectrum (eV; |value| for test, which signs negative mode)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Collision energy distribution")
    axes[1].legend()
    save(fig, "06_collision_energy")

    # ---------- 7. precursor error / label quality ----------
    m["err_da"] = m.precursor_error_ppm * m.precursor_mz / 1e6
    q = m.groupby("ingest_lib").precursor_error_ppm.agg(
        median_abs_ppm=lambda s: s.abs().median(),
        p99_abs_ppm=lambda s: s.abs().quantile(.99),
        share_gt_10ppm=lambda s: (s.abs() > 10).mean(),
        share_gt_50ppm=lambda s: (s.abs() > 50).mean(),
        share_null=lambda s: s.isna().mean(),
    ).loc[libs]
    q.to_csv(STATS / "precursor_error_by_library.csv")
    big = m[m.precursor_error_ppm.abs() > 50]
    offsets = (big.assign(offset_da=big.err_da.round(2)).groupby(["ingest_lib", "offset_da"]).size()
               .rename("spectra").reset_index().sort_values("spectra", ascending=False))
    offsets.head(60).to_csv(STATS / "precursor_error_offsets_gt50ppm.csv", index=False)
    pk = m[(m.ingest_lib == "pluskal_ms2") & (m.adduct == "[M+CH2O2-H]-")]
    stats["pluskal_formate_share_offset_1p007"] = float((pk.err_da.round(2) == 1.01).mean())
    stats["pluskal_formate_spectra"] = int(len(pk))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"wspace": 0.4})
    hbar(axes[0], libs, (q.share_gt_10ppm * 100).values, fmt="{:.1f}%", highlight=KEY)
    axes[0].set_title("Spectra with |precursor error| > 10 ppm (%)")
    bins = np.arange(-3, 25, 0.05)
    e = big.err_da[(big.err_da > -3) & (big.err_da < 25)]
    axes[1].hist(e, bins=bins, color=SERIES[0])
    axes[1].set_yscale("log")
    for x, lab in [(1.007, "+H (1.007)"), (17.027, "NH3 (17.03)"), (18.011, "H2O (18.01)"), (21.982, "Na−H (21.98)")]:
        axes[1].axvline(x, color=MUTED, linewidth=0.8, linestyle=":")
        axes[1].text(x + 0.2, axes[1].get_ylim()[1] * 0.4, lab, fontsize=7.5, color=INK2, rotation=90, va="top")
    axes[1].text(0.15, axes[1].get_ylim()[1] * 0.4, "nominal-mass\nprecursors", fontsize=7.5, color=INK2, va="top")
    axes[1].set_xlabel("Precursor error in Da (spectra with |error| > 50 ppm)")
    axes[1].set_ylabel("Spectra (log)")
    axes[1].set_title("Large errors are discrete mass offsets — adduct mislabels")
    save(fig, "07_precursor_error")

    # ---------- 8. precursor m/z & peak counts ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"wspace": 0.45})
    bins = np.arange(50, 1800, 20)
    axes[0].hist(m.precursor_mz.clip(upper=1790), bins=bins, density=True, histtype="stepfilled", color=OTHER, alpha=0.6, label="all train")
    for lib in ["enveda-180", "enveda-np-examples"]:
        axes[0].hist(m.precursor_mz[m.ingest_lib == lib], bins=bins, density=True, histtype="step", color=KEY[lib], linewidth=2, label=lib)
    axes[0].hist(t.precursor_mz, bins=bins, density=True, histtype="step", color=KEY["test"], linewidth=2, label="test (placeholder)")
    axes[0].axvspan(157, 1159, color=OTHER, alpha=0.15, linewidth=0)
    axes[0].text(165, axes[0].get_ylim()[1] * 0.95, "hidden-test neutral mass range 157–1159 Da", fontsize=7.5, color=INK2, va="top")
    axes[0].set_xlabel("Precursor m/z")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Precursor m/z")
    axes[0].legend(loc="upper right")
    data = [m.num_peaks[m.ingest_lib == l].values for l in libs] + [t.num_peaks.values]
    labels = libs + ["test (placeholder)"]
    bp = axes[1].boxplot(data, orientation="horizontal", whis=(5, 95), showfliers=False, patch_artist=True, widths=0.6,
                         medianprops={"color": "#0b0b0b"})
    for patch, lab in zip(bp["boxes"], labels):
        patch.set_facecolor(KEY.get(lab.split(" ")[0], OTHER)); patch.set_edgecolor("#52514e")
    axes[1].set_yticks(range(1, len(labels) + 1), labels)
    axes[1].invert_yaxis()
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Peaks per spectrum (log; box = IQR, whiskers = 5–95%)")
    axes[1].set_title("Peak counts differ by two orders of magnitude")
    save(fig, "08_precursor_mz_and_peaks")
    m.groupby("ingest_lib").num_peaks.describe(percentiles=[.05, .5, .95]).loc[libs].to_csv(STATS / "num_peaks_by_library.csv")

    # ---------- 9. test placeholder provenance ----------
    k = m[m.base_peak_intensity.notna()][["ingest_lib", "precursor_mz", "base_peak_intensity", "num_peaks", "inchikey14"]]
    j = t.merge(k, on=["precursor_mz", "base_peak_intensity", "num_peaks"], how="left")
    stats["test_placeholder_match"] = {
        "spectra": int(len(t)), "molecules": int(t.molecule_id.nunique()),
        "matched_spectra": int(j.drop_duplicates("spectrum_id").ingest_lib.notna().sum()),
        "source_libraries": j.drop_duplicates("spectrum_id").ingest_lib.value_counts().to_dict(),
    }
    stats["test_spectra_per_molecule"] = tg.describe().round(2).to_dict()
    stats["test_base_peak_intensity"] = t.base_peak_intensity.describe().round(0).to_dict()
    stats["test_num_peaks"] = t.num_peaks.describe(percentiles=[.05, .5, .95]).round(0).to_dict()

    (STATS / "01_metadata.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
