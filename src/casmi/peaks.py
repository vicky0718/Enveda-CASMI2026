"""Vectorised per-spectrum peak features over CSR peak batches."""

import numpy as np
import pandas as pd

from .io import PeakBatch

C13 = 1.003355
PREC_TOL_DA = 0.01


def _seg_sum(x: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Per-spectrum sum of a per-peak array; empty spectra give 0."""
    cs = np.concatenate([[0], np.cumsum(x, dtype=np.float64)])
    return cs[offsets[1:]] - cs[offsets[:-1]]


def spectrum_features(b: PeakBatch) -> pd.DataFrame:
    n = np.diff(b.offsets)
    sid = b.spec_idx
    prec = b.extra["precursor_mz"].to_numpy()[sid]
    mz, it = b.mz, b.inten
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)

    f = pd.DataFrame(index=np.arange(b.n) + b.row_offset)
    f["n_peaks"] = n
    f["n_bad_values"] = _seg_sum(~ok, b.offsets).astype(int)
    for thr, name in [(0.001, "0p1"), (0.01, "1"), (0.02, "2")]:
        f[f"n_ge_{name}pct"] = _seg_sum(it >= thr, b.offsets).astype(int)

    above = mz > prec + 2.0
    f["n_above_prec2"] = _seg_sum(above, b.offsets).astype(int)
    tot = _seg_sum(np.where(ok, it, 0), b.offsets)
    f["int_share_above_prec2"] = np.divide(_seg_sum(np.where(above & ok, it, 0), b.offsets), tot,
                                           out=np.zeros(b.n), where=tot > 0)

    near_prec = np.abs(mz - prec) <= PREC_TOL_DA
    f["prec_peak_rel_int"] = np.maximum.reduceat(np.where(near_prec, it, 0), np.minimum(b.offsets[:-1], len(mz) - 1)) if len(mz) else 0
    f.loc[n == 0, "prec_peak_rel_int"] = 0
    f["prec_present"] = f["prec_peak_rel_int"] > 0

    # Spectral entropy of the intensity distribution (natural log).
    p = np.where(ok, it, 0) / np.where(tot > 0, tot, 1)[sid]
    plogp = np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)
    f["entropy"] = -_seg_sum(plogp, b.offsets)
    f["norm_entropy"] = np.divide(f["entropy"], np.log(np.maximum(n, 2)))

    # m/z precision: share of peaks whose m/z has <= 2 / <= 4 decimals.
    f["share_mz_le2dec"] = np.divide(_seg_sum(np.abs(mz * 100 - np.round(mz * 100)) < 1e-6, b.offsets), n,
                                     out=np.zeros(b.n), where=n > 0)
    f["share_mz_le4dec"] = np.divide(_seg_sum(np.abs(mz * 1e4 - np.round(mz * 1e4)) < 1e-5, b.offsets), n,
                                     out=np.zeros(b.n), where=n > 0)

    # Sortedness & 13C isotope companions (peak at +1.00336 Da, +/-5 mDa, lower intensity).
    d = np.diff(mz)
    unsorted_pair = np.concatenate([d < 0, [False]]) & (sid == np.concatenate([sid[1:], [-1]]))
    f["unsorted"] = _seg_sum(unsorted_pair, b.offsets) > 0
    key = sid * 1e5 + mz
    order = np.argsort(key, kind="stable")
    ks = key[order]
    lo = np.searchsorted(ks, key + C13 - 0.005)
    hi = np.searchsorted(ks, key + C13 + 0.005)
    has_iso = hi > lo
    if has_iso.any():
        cand = order[np.minimum(lo, len(ks) - 1)]
        has_iso &= it[cand] < it  # first candidate in window, lighter companion
    f["share_with_c13_companion"] = np.divide(_seg_sum(has_iso, b.offsets), n, out=np.zeros(b.n), where=n > 0)

    # Base peak position relative to precursor.
    if len(mz):
        starts = np.minimum(b.offsets[:-1], len(mz) - 1)
        bp_int = np.maximum.reduceat(it, starts)
        is_bp = it == bp_int[sid]
        bp_mz = np.full(b.n, np.nan)
        idx = np.flatnonzero(is_bp)
        bp_mz[sid[idx][::-1]] = mz[idx][::-1]  # first occurrence wins
        f["base_peak_rel_mz"] = bp_mz / b.extra["precursor_mz"].to_numpy()
        f["max_int"] = bp_int
    f.loc[n == 0, ["base_peak_rel_mz", "max_int"]] = np.nan
    f["min_mz"] = np.where(n > 0, np.minimum.reduceat(mz, np.minimum(b.offsets[:-1], max(len(mz) - 1, 0))) if len(mz) else np.nan, np.nan)
    f["max_mz"] = np.where(n > 0, np.maximum.reduceat(mz, np.minimum(b.offsets[:-1], max(len(mz) - 1, 0))) if len(mz) else np.nan, np.nan)
    return f


class Histograms:
    """Accumulates per-group peak histograms across batches."""

    REL_BINS = np.linspace(0, 1.2, 121)            # fragment m/z / precursor m/z
    MZ_BINS = np.arange(0, 2001, 5.0)              # fragment m/z
    LOGI_BINS = np.linspace(-7, 0, 71)             # log10 relative intensity
    NL_BINS = np.arange(0, 300.0005, 0.005)        # neutral loss (Da), peaks >= 1%
    MD_BINS = np.linspace(-0.5, 0.5, 101)          # fragment mass defect

    def __init__(self):
        self.h: dict = {}

    def _add(self, group, name, values, bins, weights=None):
        c, _ = np.histogram(values, bins=bins, weights=weights)
        key = (group, name)
        self.h[key] = self.h.get(key, 0) + c

    def update(self, b: PeakBatch, groups: np.ndarray):
        sid = b.spec_idx
        prec = b.extra["precursor_mz"].to_numpy()[sid]
        g_peak = groups[sid]
        for g in np.unique(groups):
            sel = g_peak == g
            mz, it, pr = b.mz[sel], b.inten[sel], prec[sel]
            ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
            mz, it, pr = mz[ok], it[ok], pr[ok]
            self._add(g, "rel_mz_count", mz / pr, self.REL_BINS)
            self._add(g, "rel_mz_int", mz / pr, self.REL_BINS, it)
            self._add(g, "mz_count", mz, self.MZ_BINS)
            self._add(g, "mz_int", mz, self.MZ_BINS, it)
            self._add(g, "log_int", np.log10(it), self.LOGI_BINS)
            strong = it >= 0.01
            self._add(g, "neutral_loss", pr[strong] - mz[strong], self.NL_BINS)
            md = mz - np.round(mz)
            self._add(g, "mass_defect", md, self.MD_BINS)
            self._add(g, "_n_spectra", [0.5], [0, 1], [len(np.unique(sid[sel]))])

    def to_frame(self) -> pd.DataFrame:
        rows = [{"group": g, "hist": name, "counts": np.asarray(c, dtype=float)} for (g, name), c in self.h.items()]
        return pd.DataFrame(rows)
