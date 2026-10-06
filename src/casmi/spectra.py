"""Spectrum cleaning and binned-cosine similarity."""

import numpy as np
import scipy.sparse as sp

PROTON = 1.007276
# Neutral monoisotopic mass M = (precursor_mz - shift) for the ten test adducts.
ADDUCT_SHIFT = {
    "[M+H]+": PROTON,
    "[M+NH4]+": 18.033823,
    "[M-H2O+H]+": PROTON - 18.010565,
    "[M-2H2O+H]+": PROTON - 2 * 18.010565,
    "[M+Na]+": 22.989218,
    "[M+K]+": 38.963158,
    "[M-H]-": -PROTON,
    "[M-H2O-H]-": -PROTON - 18.010565,
    "[M+CH2O2-H]-": 44.998201,
    "[M+Cl]-": 34.969402,
}


def neutral_mass(precursor_mz: float, adduct: str) -> float:
    return precursor_mz - ADDUCT_SHIFT[adduct]


def clean(mz: np.ndarray, it: np.ndarray, precursor_mz: float, min_rel: float = 0.01, top_n: int = 64):
    """Common curation: drop peaks > precursor+2 Da and < min_rel of base, keep top-N, sqrt-scale."""
    keep = (mz <= precursor_mz + 2.0) & (it > 0)
    mz, it = mz[keep], it[keep]
    if len(it) == 0:
        return mz, it
    it = it / it.max()
    keep = it >= min_rel
    mz, it = mz[keep], it[keep]
    if len(it) > top_n:
        idx = np.argsort(it)[-top_n:]
        mz, it = mz[idx], it[idx]
    return mz, np.sqrt(it)


def binned_matrix(spectra, bin_width: float = 0.01, max_mz: float = 2000.0) -> sp.csr_matrix:
    """L2-normalised sparse matrix (n_spectra x n_bins) from cleaned (mz, intensity) pairs."""
    n_bins = int(max_mz / bin_width) + 1
    rows, cols, vals = [], [], []
    for i, (mz, it) in enumerate(spectra):
        if len(mz) == 0:
            continue
        b = np.clip((mz / bin_width).round().astype(np.int64), 0, n_bins - 1)
        rows.append(np.full(len(b), i))
        cols.append(b)
        vals.append(it)
    if not rows:
        return sp.csr_matrix((len(spectra), n_bins))
    M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(spectra), n_bins))
    M.sum_duplicates()
    norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
    norms[norms == 0] = 1
    return sp.diags(1 / norms) @ M
