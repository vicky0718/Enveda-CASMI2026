"""EDA pass 4 — how far does plain spectral library search get?

Simulates novelty class 1 (structure has public reference spectra) with the
250 enveda-np-examples molecules: timsTOF natural products processed like the
test set. Every one of them also has spectra in the public libraries.

For each query molecule (all its spectra, grouped like a test molecule_id):
  1. Neutral mass from precursor m/z + adduct.
  2. Candidates = all training structures within 10 ppm of that mass.
  3. Score candidate = max binned cosine (0.01 Da bins, sqrt intensities, >=1%,
     top 64 peaks) between any query spectrum and any of the candidate's
     reference spectra in the same ionisation mode, excluding enveda-np-examples.
  4. MRR@25 at molecule level, vs random ranking inside the mass window.

Also measures same-compound cross-instrument cosine (domain shift).

    PYTHONPATH=src python scripts/eda/04_library_search.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi.io import load_meta
from casmi.meta import instrument_family
from casmi.paths import INTERIM, STATS, TRAIN
from casmi.plot import INK2, KEY, OTHER, SERIES, save
from casmi.spectra import ADDUCT_SHIFT, binned_matrix, clean, neutral_mass

PPM = 10.0
QUERY_LIB = "enveda-np-examples"
DESC_CACHE = INTERIM / "structure_descriptors.parquet"


def read_rows(rows: np.ndarray) -> dict:
    """Return {row_index: (mz, intensity)} for the requested train rows."""
    rows = np.sort(np.unique(rows))
    f = pq.ParquetFile(TRAIN)
    out, start = {}, 0
    for rg in range(f.num_row_groups):
        n = f.metadata.row_group(rg).num_rows
        sel = rows[(rows >= start) & (rows < start + n)] - start
        if len(sel):
            tbl = f.read_row_group(rg, columns=["ms2_mzs", "ms2_normalized_intensities"]).take(sel)
            for local, mz, it in zip(sel, tbl["ms2_mzs"].to_pylist(), tbl["ms2_normalized_intensities"].to_pylist()):
                out[start + int(local)] = (np.asarray(mz), np.asarray(it))
        start += n
    return out


def main():
    m = load_meta()
    desc = pd.read_parquet(DESC_CACHE, columns=["exact_mass"])  # index = inchikey14
    stats = {}
    m["instrument_family"] = m.instrument_type.astype(object).map(instrument_family)

    q = m[m.ingest_lib == QUERY_LIB].copy()
    q = q[q.adduct.isin(ADDUCT_SHIFT)]
    q["M"] = [neutral_mass(p, a) for p, a in zip(q.precursor_mz, q.adduct)]
    ref = m[(m.ingest_lib != QUERY_LIB)]

    # Candidate structures by neutral mass.
    masses = desc.exact_mass.dropna().sort_values()
    mv, mk = masses.values, masses.index.values
    mol_M = q.groupby("inchikey14").M.median()
    cands = {}
    for ik, M in mol_M.items():
        tol = M * PPM / 1e6
        lo, hi = np.searchsorted(mv, M - tol), np.searchsorted(mv, M + tol, "right")
        cands[ik] = set(mk[lo:hi])
    n_cand = pd.Series({k: len(v) for k, v in cands.items()})
    stats["candidates_per_molecule_10ppm"] = n_cand.describe().round(1).to_dict()
    stats["true_structure_in_candidate_set"] = float(np.mean([k in v for k, v in cands.items()]))

    # Reference spectra for all candidate structures (same ion mode as the query molecule).
    all_cands = set().union(*cands.values())
    ref_c = ref[ref.inchikey14.isin(all_cands)]
    print(f"queries: {len(q)} spectra / {q.inchikey14.nunique()} molecules; reference spectra: {len(ref_c):,}")
    peaks = read_rows(np.concatenate([q.index.values, ref_c.index.values]))

    def cleaned(df):
        return [clean(*peaks[i], p) for i, p in zip(df.index, df.precursor_mz)]

    Q = binned_matrix(cleaned(q))
    R = binned_matrix(cleaned(ref_c))
    S = (Q @ R.T).tocsr()  # cosine, n_query x n_ref (sparse; zero where no shared bins)

    q_pos = {i: k for k, i in enumerate(q.index)}
    ref_ik = ref_c.inchikey14.values
    ref_mode = ref_c.ionization_mode.astype(str).values
    ref_lib = ref_c.ingest_lib.astype(str).values
    ref_fam = ref_c.instrument_family.values
    ref_adduct = ref_c.adduct.astype(str).values

    results, same_cpd = [], []
    for ik, grp in q.groupby("inchikey14"):
        cset = cands[ik]
        mode = grp.ionization_mode.astype(str).iloc[0]
        best = {c: 0.0 for c in cset}
        for i in grp.index:
            row = S.getrow(q_pos[i])
            cols, vals = row.indices, row.data
            for c, v in zip(cols, vals):
                if ref_mode[c] == mode and ref_ik[c] in best and v > best[ref_ik[c]]:
                    best[ref_ik[c]] = v
            # same-compound similarities for the domain-shift analysis
            same = np.flatnonzero((ref_ik == ik) & (ref_adduct == str(grp.adduct[i])))
            dense = row.toarray().ravel()
            for c in same:
                same_cpd.append({"inchikey14": ik, "query_ce": grp.ce_mean[i], "ref_lib": ref_lib[c],
                                 "ref_family": ref_fam[c], "cosine": dense[c]})
        scores = pd.Series(best).sort_values(ascending=False, kind="stable")
        has_ref = ik in set(ref_ik[ref_mode == mode])
        rank = (scores.index.get_loc(ik) + 1) if ik in scores.index else np.inf
        # Ties at the true score: count as average position among ties (fair to the baseline).
        if np.isfinite(rank):
            s_true = scores[ik]
            rank = (scores > s_true).sum() + ((scores == s_true).sum() + 1) / 2
        results.append({"inchikey14": ik, "n_spectra": len(grp), "n_candidates": len(cset), "rank": rank,
                        "true_score": best.get(ik, np.nan), "has_ref_same_mode": has_ref,
                        # E[1/rank] for a uniformly random order of the window, rank <= 25 only
                        "random_rr": (ik in cset) * sum(1 / r for r in range(1, min(len(cset), 25) + 1)) / max(len(cset), 1)})
    res = pd.DataFrame(results)
    res["rr"] = np.where(res["rank"] <= 25, 1 / res["rank"], 0.0)
    res.to_csv(STATS / "library_search_baseline_per_molecule.csv", index=False)
    stats["mrr25_library_search"] = float(res.rr.mean())
    stats["mrr25_random_in_mass_window"] = float(res.random_rr.mean())
    stats["top1"] = float((res["rank"] <= 1).mean())
    stats["top5"] = float((res["rank"] <= 5).mean())
    stats["top25"] = float((res["rank"] <= 25).mean())
    stats["share_with_reference_in_same_mode"] = float(res.has_ref_same_mode.mean())

    sc = pd.DataFrame(same_cpd)
    sc_best = sc.groupby(["inchikey14", "ref_family"]).cosine.max().reset_index()
    fam_summary = sc.groupby("ref_family").cosine.describe(percentiles=[.25, .5, .75]).round(3)
    fam_summary.to_csv(STATS / "same_compound_cosine_by_ref_instrument.csv")
    stats["same_compound_cosine_median_by_family"] = sc.groupby("ref_family").cosine.median().round(3).to_dict()
    stats["same_compound_best_cosine_median_by_family"] = sc_best.groupby("ref_family").cosine.median().round(3).to_dict()

    # ---------- figures ----------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"wspace": 0.45})
    ranks = res["rank"].replace(np.inf, 999)
    buckets = pd.cut(ranks, [0, 1, 2, 3, 5, 10, 25, 1000], labels=["1", "2", "3", "4–5", "6–10", "11–25", ">25 / miss"])
    vc = buckets.value_counts().reindex(buckets.cat.categories)
    axes[0].bar(range(len(vc)), vc.values, color=[KEY["enveda-np-examples"]] * 6 + [OTHER], width=0.72, edgecolor="#fcfcfb")
    axes[0].set_xticks(range(len(vc)), vc.index)
    for x, v in enumerate(vc.values):
        axes[0].text(x, v + 1, str(v), ha="center", fontsize=8, color=INK2)
    axes[0].grid(axis="x", visible=False)
    axes[0].set_xlabel("Rank of the true structure")
    axes[0].set_ylabel("Molecules")
    axes[0].set_title(f"Mass filter + cosine: MRR@25 {stats['mrr25_library_search']:.3f}\n"
                      f"(random order in the same window: {stats['mrr25_random_in_mass_window']:.3f})")
    order = sc_best.groupby("ref_family").cosine.median().sort_values(ascending=False).index.tolist()
    data = [sc_best.cosine[sc_best.ref_family == f].values for f in order]
    bp = axes[1].boxplot(data, orientation="horizontal", whis=(5, 95), showfliers=False, patch_artist=True, widths=0.6,
                         medianprops={"color": "#0b0b0b"})
    for p in bp["boxes"]:
        p.set_facecolor(SERIES[0]); p.set_alpha(0.75); p.set_edgecolor("#52514e")
    axes[1].set_yticks(range(1, len(order) + 1), [f"{f} (n={len(d)})" for f, d in zip(order, data)])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Best cosine per molecule (same adduct), by reference instrument family")
    axes[1].set_title("Same molecule, other instruments:\nbest cosine to the timsTOF query")
    save(fig, "19_library_search_baseline")

    fig, ax = plt.subplots(figsize=(8, 4))
    all_true = res.true_score.dropna()
    ax.hist(all_true, bins=np.linspace(0, 1, 41), color=KEY["enveda-np-examples"], alpha=0.9, label="score of the true structure")
    ax.set_xlabel("Best cosine of the true structure's reference spectra")
    ax.set_ylabel("Molecules")
    ax.set_title("How similar is the best matching reference spectrum of the right answer?")
    save(fig, "20_true_structure_scores")

    (STATS / "04_library_search.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
