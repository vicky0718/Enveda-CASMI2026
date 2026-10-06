"""EDA pass 5b — closing the loose ends that matter for isomer discrimination.

1. Fragment sub-formula annotation (with a decoy control): fragment mass accuracy per library and
   polarity (tests the reported +0.55 mDa positive-mode calibration offset), and the share of peaks
   that are *chemically explainable* as a function of relative intensity (an objective noise floor).
2. Exact-duplicate spectra carrying different structure labels, across all of train.
3. Per-molecule coverage (polarities, energy ladders, adducts) of the test-like np-examples.
4. "Biosynthetic deltas": formula difference between a natural product and its nearest *different*
   training structure — what candidate generation must bridge for unseen molecules.
5. Same-compound cross-instrument similarity with tolerance-based entropy similarity at 0.01 Da and
   0.5 Da (separates "low precision" from "different fragmentation").

    PYTHONPATH=src python scripts/eda/05b_deep_dive.py
"""

import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from casmi.formula import ADDUCT_ATOMS, ion_formula, match_peaks, parse, subformula_masses
from casmi.io import load_meta, read_rows
from casmi.meta import instrument_family
from casmi.paths import INTERIM, STATS, TRAIN
from casmi.plot import KEY, MUTED, OTHER, SERIES, hbar, save

RDLogger.DisableLog("rdApp.*")
RNG = np.random.default_rng(0)
TOL_DA = 0.02          # wide matching window, so the error distribution is visible
DECOY_SHIFT = 0.05     # Da; same mass-defect region, far beyond instrument error
SAMPLE = {"enveda-np-examples": None, "enveda-180": 2500, "gnps": 1000, "pluskal_ms2": 1000,
          "riken": 1000, "mona": 1000, "massbank": 1000, "msdial": 1000}


# ---------------------------------------------------------------- 1. annotation
def annotate(m: pd.DataFrame) -> pd.DataFrame:
    cache = INTERIM / "fragment_annotation.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    pool = m[m.adduct.isin(ADDUCT_ATOMS) & (m.precursor_error_ppm.abs() < 10)]
    rows = []
    for lib, n in SAMPLE.items():
        d = pool[pool.ingest_lib == lib]
        rows.append(d if n is None or len(d) <= n else d.sample(n, random_state=0))
    q = pd.concat(rows)
    peaks = read_rows(q.index.values)
    recs = []
    sub_cache = {}
    for i, r in zip(q.index, q.itertuples()):
        key = (r.molecular_formula, r.adduct)
        if key not in sub_cache:
            try:
                ion = ion_formula(r.molecular_formula, r.adduct)
                sub_cache[key] = subformula_masses(*ion) if ion else None
            except ValueError:
                sub_cache[key] = None
        sub = sub_cache[key]
        if sub is None:
            continue
        sm = sub[0]
        mz, it = peaks[i]
        keep = (mz <= r.precursor_mz + 0.5) & (it > 0)
        mz, it = mz[keep], it[keep] / it[keep].max()
        err, _ = match_peaks(mz, sm, TOL_DA)
        e_plus, _ = match_peaks(mz + DECOY_SHIFT, sm, TOL_DA)
        e_minus, _ = match_peaks(mz - DECOY_SHIFT, sm, TOL_DA)
        recs.append(pd.DataFrame({
            "row": i, "lib": r.ingest_lib, "mode": r.ionization_mode, "adduct": r.adduct,
            "ce": r.ce_mean, "mz": mz, "rel_int": it, "err_da": err, "dec_p": e_plus, "dec_m": e_minus,
            "frac_mz": mz / r.precursor_mz,
        }))
    a = pd.concat(recs, ignore_index=True)
    a.to_parquet(cache)
    return a


def mass_accuracy(a: pd.DataFrame, stats: dict):
    strong = a[(a.rel_int >= 0.05) & a.err_da.notna() & (a.frac_mz < 0.98)]  # exclude precursor itself
    g = strong.groupby(["lib", "mode"]).err_da
    acc = pd.DataFrame({
        "n_peaks": g.size(),
        "median_err_mDa": g.median() * 1e3,
        "robust_sd_mDa": g.apply(lambda s: 1.4826 * np.median(np.abs(s - s.median()))) * 1e3,
        "median_err_ppm": strong.assign(p=strong.err_da / strong.mz * 1e6).groupby(["lib", "mode"]).p.median(),
    }).round(3)
    acc.to_csv(STATS / "fragment_mass_accuracy.csv")
    stats["fragment_mass_accuracy"] = {f"{k[0]}|{k[1]}": v for k, v in acc.to_dict(orient="index").items()}
    for lib in ("enveda-np-examples", "enveda-180"):
        try:
            stats[f"{lib}_pos_minus_neg_median_mDa"] = float(acc.loc[(lib, "positive"), "median_err_mDa"]
                                                            - acc.loc[(lib, "negative"), "median_err_mDa"])
        except KeyError:
            pass

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"wspace": 0.25})
    bins = np.linspace(-10, 10, 81)
    for ax, lib in zip(axes, ("enveda-np-examples", "enveda-180")):
        for mode, c in (("positive", SERIES[0]), ("negative", SERIES[1])):
            s = strong[(strong.lib == lib) & (strong["mode"] == mode)].err_da * 1e3
            ax.hist(s.clip(-10, 10), bins=bins, histtype="step", linewidth=2, color=c, density=True,
                    label=f"{mode} (median {s.median():+.2f} mDa, n={len(s):,})")
        ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_xlabel("Fragment m/z − nearest sub-formula m/z (mDa), peaks ≥ 5 %")
        ax.set_ylabel("Density")
        ax.set_title(f"{lib}: fragment mass accuracy")
        ax.legend(fontsize=8)
    save(fig, "21_fragment_mass_accuracy")
    return acc


def noise_floor(a: pd.DataFrame, acc: pd.DataFrame, stats: dict):
    """Chance-corrected share of peaks explained by a sub-formula, per relative-intensity bin."""
    bins = np.array([1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0001])
    labels = ["<0.01%", "0.01–0.03%", "0.03–0.1%", "0.1–0.3%", "0.3–1%", "1–3%", "3–10%", "≥10%"]
    out = []
    for lib in SAMPLE:
        d = a[(a.lib == lib) & (a.frac_mz < 0.98)]
        if d.empty:
            continue
        for mode in ("positive", "negative"):
            dm = d[d["mode"] == mode]
            if dm.empty or (lib, mode) not in acc.index:
                continue
            med = acc.loc[(lib, mode), "median_err_mDa"] / 1e3
            w = max(3 * acc.loc[(lib, mode), "robust_sd_mDa"] / 1e3, 0.002)  # instrument-specific window
            hit = (dm.err_da - med).abs() <= w
            dec = (((dm.dec_p - med).abs() <= w).astype(float) + ((dm.dec_m - med).abs() <= w).astype(float)) / 2
            cut = pd.cut(dm.rel_int, bins, labels=labels, right=False)
            tab = pd.DataFrame({"hit": hit, "dec": dec, "bin": cut}).groupby("bin", observed=False).agg(
                n=("hit", "size"), hit=("hit", "mean"), decoy=("dec", "mean"))
            tab["chance_corrected"] = (tab.hit - tab.decoy) / (1 - tab.decoy)
            tab = tab.assign(lib=lib, mode=mode, window_mDa=w * 1e3).reset_index()
            out.append(tab)
    nf = pd.concat(out)
    nf.to_csv(STATS / "noise_floor_explained_by_intensity.csv", index=False)
    stats["noise_floor"] = (nf[nf["mode"] == "positive"].pivot(index="bin", columns="lib", values="chance_corrected")
                            .round(3).to_dict())

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(labels))
    for lib in SAMPLE:
        s = nf[(nf.lib == lib) & (nf["mode"] == "positive")].set_index("bin").reindex(labels)
        if s.n.fillna(0).sum() == 0:
            continue
        s = s[s.n >= 30]
        c = KEY.get(lib, OTHER)
        lw = 2.2 if lib in KEY else 1.2
        ax.plot([labels.index(b) for b in s.index], s.chance_corrected, marker="o", markersize=5 if lib in KEY else 3,
                color=c, linewidth=lw, label=lib, linestyle="-" if lib in KEY else "--")
    ax.set_xticks(x, labels)
    ax.set_ylim(-0.05, 1.0)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("Peak relative intensity (base peak = 100 %)")
    ax.set_ylabel("Chance-corrected share explained\nby a sub-formula of the ion")
    ax.set_title("Where real fragments end and noise begins (positive mode; decoy-corrected)")
    ax.legend(fontsize=7.5, ncol=2)
    save(fig, "22_noise_floor_explained_peaks")
    return nf


# ---------------------------------------------------------------- 2. duplicates
def duplicate_conflicts(m: pd.DataFrame, keys: pd.DataFrame, stats: dict):
    cache = INTERIM / "spectrum_hashes.parquet"
    if not cache.exists():
        f = pq.ParquetFile(TRAIN)
        hs = []
        for rg in range(f.num_row_groups):
            tbl = f.read_row_group(rg, columns=["ms2_mzs", "ms2_normalized_intensities"])
            mz = tbl["ms2_mzs"].combine_chunks()
            it = tbl["ms2_normalized_intensities"].combine_chunks()
            off = mz.offsets.to_numpy()
            fm = pc.list_flatten(mz).to_numpy(zero_copy_only=False)
            fi = pc.list_flatten(it).to_numpy(zero_copy_only=False)
            for a, b in zip(off[:-1], off[1:]):
                h = hashlib.blake2b(fm[a:b].tobytes(), digest_size=12)
                h.update(fi[a:b].tobytes())
                hs.append(h.hexdigest())
        pd.DataFrame({"hash": hs}).to_parquet(cache)
    h = pd.read_parquet(cache).hash.values
    d = m[["ingest_lib", "normalized_smiles", "inchikey14"]].copy()
    d["hash"] = h
    d = d.merge(keys[["normalized_smiles", "metric_key"]], on="normalized_smiles", how="left")
    dup = d[d.duplicated("hash", keep=False)]
    g = dup.groupby("hash")
    conflict = g.metric_key.nunique()
    conflict = conflict[conflict > 1]
    cd = dup[dup.hash.isin(conflict.index)]
    stats["duplicate_spectra_rows"] = int(len(dup))
    stats["duplicate_groups"] = int(g.ngroups)
    stats["duplicate_groups_conflicting_metric_key"] = int(len(conflict))
    stats["rows_in_conflicting_groups"] = int(len(cd))
    stats["conflict_library_pairs"] = (cd.groupby("hash").ingest_lib.agg(lambda s: "+".join(sorted(set(s))))
                                       .value_counts().head(10).to_dict())
    cd.groupby("hash").agg(libs=("ingest_lib", lambda s: "+".join(sorted(set(s)))),
                           keys=("metric_key", lambda s: "|".join(sorted(set(s.dropna()))))).to_csv(
        STATS / "duplicate_spectra_conflicting_labels.csv")


# ---------------------------------------------------------------- 3. coverage
def molecule_coverage(m: pd.DataFrame, stats: dict):
    npx = m[m.ingest_lib == "enveda-np-examples"]
    g = npx.groupby("inchikey14")
    cov = pd.DataFrame({
        "n_spectra": g.size(),
        "polarities": g.ionization_mode.agg(lambda s: "+".join(sorted(set(s)))),
        "n_adducts": g.adduct.nunique(),
        "has_merged": g.ce_n.agg(lambda s: bool((s > 1).any())),
        "n_single_energies": g.apply(lambda d: d[d.ce_n == 1].ce_mean.nunique()),
    })
    stats["np_examples_coverage"] = {
        "molecules": int(len(cov)),
        "spectra_per_molecule": cov.n_spectra.describe().round(2).to_dict(),
        "polarity_mix": cov.polarities.value_counts().to_dict(),
        "share_multi_adduct": float((cov.n_adducts > 1).mean()),
        "share_with_merged_spectrum": float(cov.has_merged.mean()),
        "single_energy_count_dist": cov.n_single_energies.value_counts().sort_index().to_dict(),
    }
    cov.to_csv(STATS / "np_examples_molecule_coverage.csv")


# ---------------------------------------------------------------- 4. biosynthetic deltas
def formula_delta(fa: str, fb: str) -> str:
    """fb − fa as a signed formula string, e.g. '+CH2', '+C6H10O5', '-O+S'."""
    try:
        a, b = parse(fa), parse(fb)
    except ValueError:
        return "?"
    els = sorted(set(a) | set(b), key=lambda e: ("CHNOPS".find(e) if e in "CHNOPS" else 9, e))
    pos = "".join(f"{e}{'' if (b[e] - a[e]) == 1 else b[e] - a[e]}" for e in els if b[e] - a[e] > 0)
    neg = "".join(f"{e}{'' if (a[e] - b[e]) == 1 else a[e] - b[e]}" for e in els if a[e] - b[e] > 0)
    if not pos and not neg:
        return "isomer (0)"
    return (f"+{pos}" if pos else "") + (f"-{neg}" if neg else "")


def biosynthetic_deltas(m: pd.DataFrame, stats: dict):
    desc = pd.read_parquet(INTERIM / "structure_descriptors.parquet",
                           columns=["normalized_smiles", "molecular_formula", "libs", "np_likeness", "exact_mass"])
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in desc.normalized_smiles]
    keys = desc.index.values
    np_keys = set(m.inchikey14[m.ingest_lib == "enveda-np-examples"])
    q_np = [i for i, k in enumerate(keys) if k in np_keys]
    pub_np = np.flatnonzero(desc.np_likeness.values > 1)
    q_pub = list(RNG.choice(pub_np, 2000, replace=False))
    rows = []
    for name, qs in (("enveda-np-examples", q_np), ("public NP-like (NP-likeness > 1)", q_pub)):
        for i in qs:
            sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps))
            sims[i] = -1
            j = int(sims.argmax())
            rows.append({"set": name, "query": keys[i], "nn": keys[j], "tanimoto": sims[j],
                         "delta": formula_delta(desc.molecular_formula.iloc[j], desc.molecular_formula.iloc[i]),
                         "dmass": desc.exact_mass.iloc[i] - desc.exact_mass.iloc[j]})
    bd = pd.DataFrame(rows)
    bd.to_csv(STATS / "nearest_analog_formula_deltas.csv", index=False)
    common = {"isomer (0)", "+CH2", "-CH2", "+O", "-O", "+H2", "-H2", "+C2H2O", "-C2H2O", "+C6H10O5", "-C6H10O5",
              "+C2H4", "-C2H4", "+CH2O", "-CH2O", "+C5H8", "-C5H8", "+C6H10O4", "-C6H10O4", "+C5H8O4", "-C5H8O4",
              "+H2O", "-H2O", "+O2", "-O2", "+C2H4O", "-C2H4O", "+CO", "-CO", "+CO2", "-CO2"}
    summary = {}
    for name, d in bd.groupby("set"):
        close = d[d.tanimoto >= 0.7]
        summary[name] = {
            "n": int(len(d)), "median_tanimoto": float(d.tanimoto.median()),
            "share_nn_tanimoto_ge_0.7": float((d.tanimoto >= 0.7).mean()),
            "share_delta_in_common_set": float(d.delta.isin(common).mean()),
            "share_close_and_common": float((d.tanimoto >= 0.7).mul(d.delta.isin(common)).mean()),
            "top_deltas_close_analogs": close.delta.value_counts().head(15).to_dict(),
        }
    stats["biosynthetic_deltas"] = summary

    fig, ax = plt.subplots(figsize=(10, 5.4))
    d = bd[(bd["set"] == "enveda-np-examples") & (bd.tanimoto >= 0.7)]
    top = d.delta.value_counts().head(14)
    hbar(ax, top.index.tolist(), (top / len(bd[bd["set"] == "enveda-np-examples"]) * 100).values,
         color=KEY["enveda-np-examples"], fmt="{:.1f}%")
    ax.set_xlabel("% of the 250 test-like natural products")
    ax.set_title("Formula change from each natural product's nearest training analog (Tanimoto ≥ 0.7)")
    save(fig, "24_biosynthetic_deltas")


# ---------------------------------------------------------------- 5. entropy similarity
def _clean(mz, it, prec):
    keep = (mz <= prec - 1.5) & (it >= 0.01 * it.max())  # drop precursor region & <1 % noise
    mz, it = mz[keep], it[keep]
    return mz, it / it.sum() if len(it) else it


def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _weight(p):
    s = _entropy(p)
    if s < 3:
        p = p ** (0.25 + 0.25 * s)
        p = p / p.sum()
    return p


def entropy_similarity(a, b, tol):
    """Weighted spectral entropy similarity (Li et al. 2021) with greedy peak matching."""
    (ma, pa), (mb, pb) = a, b
    if len(pa) == 0 or len(pb) == 0:
        return 0.0
    pa, pb = _weight(pa), _weight(pb)
    pa, pb = pa / 2, pb / 2
    # pairwise candidates within tolerance, greedily matched by combined intensity
    i, j = np.nonzero(np.abs(ma[:, None] - mb[None, :]) <= tol)
    order = np.argsort(-(pa[i] + pb[j]))
    used_a, used_b, merged = set(), set(), []
    for k in order:
        if i[k] in used_a or j[k] in used_b:
            continue
        used_a.add(i[k]); used_b.add(j[k])
        merged.append(pa[i[k]] + pb[j[k]])
    rest = [pa[x] for x in range(len(pa)) if x not in used_a] + [pb[x] for x in range(len(pb)) if x not in used_b]
    s_ab = _entropy(np.array(merged + rest))
    return float(1 - (2 * s_ab - _entropy(pa * 2) - _entropy(pb * 2)) / np.log(4))


def cross_instrument(m: pd.DataFrame, stats: dict):
    m = m.assign(fam=m.instrument_type.astype(object).map(instrument_family))
    q = m[(m.ingest_lib == "enveda-np-examples") & m.adduct.isin(ADDUCT_ATOMS)]
    ref = m[(m.ingest_lib != "enveda-np-examples") & m.inchikey14.isin(set(q.inchikey14)) & m.adduct.isin(ADDUCT_ATOMS)]
    pairs = q.reset_index().merge(ref.reset_index(), on=["inchikey14", "adduct"], suffixes=("_q", "_r"))
    if len(pairs) > 60000:
        pairs = pairs.sample(60000, random_state=0)
    peaks = read_rows(np.concatenate([pairs.index_q.unique(), pairs.index_r.unique()]))
    cq = {i: _clean(*peaks[i], p) for i, p in zip(q.index, q.precursor_mz)}
    cr = {i: _clean(*peaks[i], p) for i, p in zip(pairs.index_r, pairs.precursor_mz_r)}
    out = []
    for r in pairs.itertuples():
        a, b = cq[r.index_q], cr[r.index_r]
        out.append((r.inchikey14, r.fam_r, r.ingest_lib_r, entropy_similarity(a, b, 0.01), entropy_similarity(a, b, 0.5)))
    s = pd.DataFrame(out, columns=["ik", "family", "lib", "ent_0p01", "ent_0p5"])
    best = s.groupby(["ik", "family"])[["ent_0p01", "ent_0p5"]].max().reset_index()
    summ = best.groupby("family")[["ent_0p01", "ent_0p5"]].median().round(3)
    summ["molecules"] = best.groupby("family").size()
    summ.to_csv(STATS / "same_compound_entropy_by_family.csv")
    stats["same_compound_best_entropy_by_family"] = summ.to_dict(orient="index")

    fig, ax = plt.subplots(figsize=(10, 4.4))
    fams = summ.sort_values("ent_0p01", ascending=False).index.tolist()
    y = np.arange(len(fams))[::-1]
    ax.barh(y + 0.19, summ.loc[fams, "ent_0p01"], height=0.36, color=SERIES[0], label="0.01 Da tolerance")
    ax.barh(y - 0.19, summ.loc[fams, "ent_0p5"], height=0.36, color=SERIES[1], label="0.5 Da tolerance")
    ax.set_yticks(y, [f"{f} (n={summ.loc[f, 'molecules']})" for f in fams])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Median best entropy similarity, timsTOF query vs same molecule + adduct")
    ax.set_title("Is low cross-instrument similarity precision or chemistry? (tight vs loose tolerance)")
    ax.legend(loc="lower right")
    save(fig, "25_cross_instrument_entropy")


def main():
    m = load_meta()
    keys = pd.read_parquet(INTERIM / "metric_keys.parquet")
    stats = {}
    a = annotate(m)
    acc = mass_accuracy(a, stats)
    noise_floor(a, acc, stats)
    molecule_coverage(m, stats)
    duplicate_conflicts(m, keys, stats)
    cross_instrument(m, stats)
    biosynthetic_deltas(m, stats)
    (STATS / "05b_deep_dive.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
