"""EDA pass 2 — peak-level statistics.

Streams train.parquet one row group at a time (3 worker processes, sliced), computing
per-spectrum peak features (cached to data/interim/, which is git-ignored) and
per-library histograms, then renders figures and summary tables.

    PYTHONPATH=src python scripts/eda/02_peaks.py
"""

import json
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi.io import PeakBatch, load_meta, load_test
from casmi.paths import INTERIM, STATS, TRAIN
from casmi.peaks import Histograms, spectrum_features
from casmi.plot import KEY, OTHER, SERIES, hbar, save

FEAT_CACHE = INTERIM / "train_peak_features.parquet"
HIST_CACHE = INTERIM / "peak_histograms.pkl"
TEST_FEAT_CACHE = INTERIM / "test_peak_features.parquet"
TEST_GROUP = "test (placeholder)"

NEUTRAL_LOSSES = {  # common small-molecule / natural-product neutral losses (monoisotopic, Da)
    "NH3": 17.0265, "H2O": 18.0106, "CO": 27.9949, "C2H4": 28.0313, "CH2O": 30.0106, "CH3OH": 32.0262,
    "2H2O": 36.0211, "C2H2O (ketene)": 42.0106, "CO2": 43.9898, "HCOOH": 46.0055,
    "C3H6 (propene)": 42.0470, "C4H8 (isobutene)": 56.0626, "C5H8 (isoprene)": 68.0626,
    "pentose": 132.0423, "deoxyhexose": 146.0579, "hexose": 162.0528, "glucuronide": 176.0321,
    "malonylhexose": 248.0532, "rutinoside": 308.1107, "SO3": 79.9568, "HPO3": 79.9663,
}


RG_DIR = INTERIM / "peak_rg"
SLICE = 32768  # spectra per slice; bounds peak-array temporaries to ~8M peaks


def _worker(rg: int) -> int:
    """Features + histograms for one row group, written to RG_DIR (restartable)."""
    import pyarrow.compute as pc

    out = RG_DIR / f"rg{rg:02d}.pkl"
    if out.exists():
        return rg
    f = pq.ParquetFile(TRAIN)
    tbl = f.read_row_group(rg, columns=["ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "ingest_lib"])
    start = sum(f.metadata.row_group(i).num_rows for i in range(rg))
    feats, H = [], Histograms()
    for s0 in range(0, tbl.num_rows, SLICE):
        part = tbl.slice(s0, SLICE)
        mzs, ints = part["ms2_mzs"].combine_chunks(), part["ms2_normalized_intensities"].combine_chunks()
        lengths = pc.list_value_length(mzs).to_numpy()
        b = PeakBatch(start + s0, np.concatenate([[0], np.cumsum(lengths)]),
                      pc.list_flatten(mzs).to_numpy(zero_copy_only=False),
                      pc.list_flatten(ints).to_numpy(zero_copy_only=False),
                      part.select(["precursor_mz", "ingest_lib"]).to_pandas())
        feats.append(spectrum_features(b))
        H.update(b, b.extra["ingest_lib"].to_numpy())
        del b, mzs, ints
    pd.to_pickle((pd.concat(feats), H.h), out)
    print(f"  row group {rg} done ({tbl.num_rows:,} spectra)", flush=True)
    return rg


def test_batch() -> PeakBatch:
    t = load_test()
    lengths = t.ms2_mzs.map(len).to_numpy()
    return PeakBatch(0, np.concatenate([[0], np.cumsum(lengths)]),
                     np.concatenate(t.ms2_mzs.to_numpy()), np.concatenate(t.ms2_normalized_intensities.to_numpy()),
                     t[["precursor_mz"]].assign(ingest_lib=TEST_GROUP))


def compute():
    """Returns (train features indexed by train row, test features, histogram frame)."""
    if FEAT_CACHE.exists() and TEST_FEAT_CACHE.exists() and HIST_CACHE.exists():
        return pd.read_parquet(FEAT_CACHE), pd.read_parquet(TEST_FEAT_CACHE), pd.read_pickle(HIST_CACHE)
    RG_DIR.mkdir(exist_ok=True)
    n_rg = pq.ParquetFile(TRAIN).num_row_groups
    with Pool(3, maxtasksperchild=1) as pool:
        list(pool.imap_unordered(_worker, range(n_rg)))
    feats, H = [], Histograms()
    for rg in range(n_rg):
        fr, h = pd.read_pickle(RG_DIR / f"rg{rg:02d}.pkl")
        feats.append(fr)
        for k, v in h.items():
            H.h[k] = H.h.get(k, 0) + v
    feats = pd.concat(feats)
    tb = test_batch()
    tf = spectrum_features(tb)
    H.update(tb, tb.extra["ingest_lib"].to_numpy())
    feats.to_parquet(FEAT_CACHE)
    tf.to_parquet(TEST_FEAT_CACHE)
    hist = H.to_frame()
    hist.to_pickle(HIST_CACHE)
    return feats, tf, hist


def main():
    tr, te, hist = compute()
    m = load_meta()
    tr = tr.join(m[["ingest_lib", "adduct", "ionization_mode", "precursor_mz", "ce_mean"]])
    te = te.assign(ingest_lib=TEST_GROUP)
    allf = pd.concat([tr, te])
    libs = m.ingest_lib.value_counts().index.tolist() + [TEST_GROUP]
    stats = {}

    # ---------- summary table ----------
    g = allf.groupby("ingest_lib")
    summ = pd.DataFrame({
        "median_peaks": g.n_peaks.median(),
        "median_peaks_ge_0.1pct": g.n_ge_0p1pct.median(),
        "median_peaks_ge_1pct": g.n_ge_1pct.median(),
        "share_peaks_lt_0.1pct": 1 - g.n_ge_0p1pct.sum() / g.n_peaks.sum(),
        "share_spectra_lt6_peaks_ge_1pct": g.n_ge_1pct.apply(lambda s: (s < 6).mean()),
        "share_spectra_with_peaks_above_prec2": g.n_above_prec2.apply(lambda s: (s > 0).mean()),
        "share_precursor_peak_present": g.prec_present.mean(),
        "median_precursor_rel_int_if_present": g.prec_peak_rel_int.apply(lambda s: s[s > 0].median()),
        "median_entropy": g.entropy.median(),
        "share_spectra_mz_rounded_2dec": g.share_mz_le2dec.apply(lambda s: (s > 0.9).mean()),
        "median_share_c13_companion": g.share_with_c13_companion.median(),
        "median_base_peak_rel_mz": g.base_peak_rel_mz.median(),
        "share_unsorted_spectra": g.unsorted.mean(),
    }).loc[libs]
    summ.to_csv(STATS / "peak_summary_by_library.csv")
    stats["peak_summary"] = summ.round(4).to_dict(orient="index")
    stats["n_bad_values_total"] = int(allf.n_bad_values.sum())

    # ---------- F9: peak counts at intensity floors ----------
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(libs))[::-1]
    for col, c, lab in [("n_peaks", OTHER, "all peaks"), ("n_ge_0p1pct", SERIES[0], "≥ 0.1% of base peak"),
                        ("n_ge_1pct", SERIES[1], "≥ 1%")]:
        med = g[col].median().loc[libs].values
        ax.scatter(med, y, s=64, color=c, label=lab, zorder=3, edgecolor="#fcfcfb", linewidth=1.5)
    for yi, lib in zip(y, libs):
        lo, hi = g.n_ge_1pct.median()[lib], g.n_peaks.median()[lib]
        ax.plot([lo, hi], [yi, yi], color=OTHER, linewidth=1, zorder=1)
    ax.set_xscale("log")
    ax.set_yticks(y, libs)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Median peaks per spectrum (log)")
    ax.legend(loc="lower right")
    ax.set_title("Most peaks in timsTOF spectra are low-intensity: median peak count at relative-intensity floors")
    save(fig, "09_peaks_vs_intensity_floor")

    # ---------- F10: precursor peak / peaks above precursor / rounding ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"wspace": 0.75})
    hl = {**KEY, TEST_GROUP: KEY["test"]}
    hbar(axes[0], libs, (summ.share_precursor_peak_present * 100).values, fmt="{:.0f}%", highlight=hl)
    axes[0].set_title("Precursor peak present (±0.01 Da), %")
    hbar(axes[1], libs, (summ.share_spectra_with_peaks_above_prec2 * 100).values, fmt="{:.1f}%", highlight=hl)
    axes[1].set_title("Spectra with peaks > precursor + 2 Da, %")
    hbar(axes[2], libs, (summ.share_spectra_mz_rounded_2dec * 100).values, fmt="{:.0f}%", highlight=hl)
    axes[2].set_title("Spectra with m/z rounded to ≤ 2 decimals, %")
    for a in axes[1:]:
        a.set_yticklabels([])
    save(fig, "10_precursor_and_precision")

    # ---------- histograms ----------
    def H(group, name):
        r = hist[(hist.group == group) & (hist.hist == name)]
        return r.counts.iloc[0] if len(r) else None

    def ctr(bins):
        return (bins[:-1] + bins[1:]) / 2

    focus = ["enveda-180", "enveda-np-examples", TEST_GROUP]
    context = ["gnps", "pluskal_ms2", "riken"]
    col = {"enveda-180": KEY["enveda-180"], "enveda-np-examples": KEY["enveda-np-examples"], TEST_GROUP: KEY["test"]}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), gridspec_kw={"wspace": 0.3})
    for grp in context + focus:
        c = col.get(grp, OTHER)
        lw = 2 if grp in focus else 1.2
        h = H(grp, "log_int")
        axes[0].plot(ctr(Histograms.LOGI_BINS), h / h.sum(), color=c, linewidth=lw, label=grp,
                     linestyle="-" if grp in focus else "--")
        h = H(grp, "rel_mz_int")
        axes[1].plot(ctr(Histograms.REL_BINS), h / h.sum(), color=c, linewidth=lw, label=grp, linestyle="-" if grp in focus else "--")
    axes[0].set_xlabel("log10 relative intensity (base peak = 0)")
    axes[0].set_ylabel("Share of peaks")
    axes[0].set_title("Peak intensity distribution")
    axes[0].legend(fontsize=7.5)
    axes[1].set_xlabel("Fragment m/z ÷ precursor m/z")
    axes[1].set_ylabel("Share of total intensity")
    axes[1].set_title("Where the intensity sits relative to the precursor")
    for grp in focus:
        h = H(grp, "mass_defect")
        axes[2].plot(ctr(Histograms.MD_BINS), h / h.sum(), color=col[grp], linewidth=2, label=grp)
    axes[2].set_xlabel("Fragment mass defect (m/z − nearest integer)")
    axes[2].set_ylabel("Share of peaks")
    axes[2].set_title("Fragment mass defect")
    save(fig, "11_intensity_relmz_massdefect")

    # ---------- F12: neutral losses ----------
    nl_ctr = ctr(Histograms.NL_BINS)
    nl_rows = []
    for grp in [l for l in libs]:
        h = H(grp, "neutral_loss")
        n_spec = H(grp, "_n_spectra")
        if h is None:
            continue
        for name, mass in NEUTRAL_LOSSES.items():
            w = np.abs(nl_ctr - mass) <= 0.0075
            nl_rows.append({"group": grp, "loss": name, "mass": mass, "per_spectrum": h[w].sum() / n_spec.sum()})
    nl = pd.DataFrame(nl_rows).pivot(index="loss", columns="group", values="per_spectrum")
    nl = nl[[l for l in libs if l in nl.columns]]
    nl.to_csv(STATS / "neutral_loss_rates.csv")
    order = nl[["enveda-np-examples", "enveda-180"]].max(axis=1).sort_values(ascending=False).index[:16]
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(order))[::-1]
    w = 0.38
    ax.barh(y + w / 2, nl.loc[order, "enveda-np-examples"], height=w, color=KEY["enveda-np-examples"], label="enveda-np-examples (natural products, timsTOF)")
    ax.barh(y - w / 2, nl.loc[order, "enveda-180"], height=w, color=KEY["enveda-180"], label="enveda-180 (synthetic screening, timsTOF)")
    ax.set_yticks(y, [f"{o}  ({NEUTRAL_LOSSES[o]:.3f})" for o in order])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Matching fragment peaks (≥1% intensity) per spectrum")
    ax.legend(loc="lower right")
    ax.set_title("Neutral losses from the precursor: glycoside and water losses mark natural products")
    save(fig, "12_neutral_losses")

    # Top unannotated neutral-loss peaks for natural products (data-driven).
    h = H("enveda-np-examples", "neutral_loss")
    top = np.argsort(h)[::-1][:25]
    pd.DataFrame({"neutral_loss_da": nl_ctr[top].round(3), "count": h[top]}).to_csv(
        STATS / "top_neutral_losses_np_examples.csv", index=False)

    # ---------- F13: entropy & quality vs instrument ----------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"wspace": 0.3})
    bins = np.linspace(0, 6, 61)
    for grp in context + focus:
        s = allf.entropy[allf.ingest_lib == grp]
        axes[0].hist(s, bins=bins, density=True, histtype="step", color=col.get(grp, OTHER),
                     linewidth=2 if grp in focus else 1.2, linestyle="-" if grp in focus else "--", label=grp)
    axes[0].set_xlabel("Spectral entropy (nats)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Spectral entropy")
    axes[0].legend(fontsize=7.5)
    e180 = tr[tr.ingest_lib == "enveda-180"]
    ce = e180.ce_mean.round()
    for i, (cev, c) in enumerate(zip(sorted(ce.dropna().unique()), [SERIES[0], SERIES[1], SERIES[2]])):
        s = e180.base_peak_rel_mz[ce == cev]
        axes[1].hist(s, bins=np.linspace(0, 1.05, 64), density=True, histtype="step", color=c, linewidth=2, label=f"{cev:.0f} eV")
    axes[1].set_xlabel("Base peak m/z ÷ precursor m/z")
    axes[1].set_ylabel("Density")
    axes[1].set_title("enveda-180: higher collision energy moves the base peak to smaller fragments")
    axes[1].legend()
    save(fig, "13_entropy_and_ce_effect")
    stats["enveda180_base_peak_rel_mz_by_ce"] = e180.groupby(ce).base_peak_rel_mz.median().round(3).to_dict()
    stats["enveda180_prec_present_by_ce"] = e180.groupby(ce).prec_present.mean().round(3).to_dict()

    (STATS / "02_peaks.json").write_text(json.dumps(stats, indent=2, default=str))
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
