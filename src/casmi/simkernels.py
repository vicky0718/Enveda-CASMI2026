"""Numba kernels: weighted spectral entropy similarity (Li et al., Nat Methods 2021), direct and
mass-shifted (for analog propagation), between one query and many CSR-stored reference spectra.

With halved intensities a = p_q/2, b = p_r/2, the entropy similarity
    1 - (2 S_AB - S_A - S_B) / ln 4
reduces exactly to  G / ln 2,  G = sum over matched pairs [(a+b)ln(a+b) - a ln a - b ln b].
Spectra must be m/z-sorted with intensities summing to 1 (after weighting). Matching is greedy
two-pointer, nearest peak within tolerance, each peak used at most once.
"""

import numpy as np
from numba import njit, prange

LN2 = np.log(2.0)


def prepare(mz, it, prec_mz, floor=0.002, top_k=256, power=1.0, ent_weight=True, drop_above=1.5):
    """Clean one spectrum: drop > precursor+drop_above, relative floor, top-k, entropy weighting."""
    mz = np.asarray(mz, np.float64)
    it = np.asarray(it, np.float64)
    keep = (mz <= prec_mz + drop_above) & (it > 0)
    mz, it = mz[keep], it[keep]
    if len(it) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    it = it / it.max()
    keep = it >= floor
    mz, it = mz[keep], it[keep]
    if len(it) > top_k:
        sel = np.argsort(-it)[:top_k]
        mz, it = mz[sel], it[sel]
    o = np.argsort(mz)
    mz, it = mz[o], it[o] ** power
    p = it / it.sum()
    if ent_weight:
        s = -(p * np.log(p)).sum()
        if s < 3.0:
            p = p ** (0.25 + 0.25 * s)
            p = p / p.sum()
    return mz.astype(np.float32), p.astype(np.float32)


@njit(cache=True)
def _xlogx(x):
    return x * np.log(x) if x > 0 else 0.0


@njit(cache=True)
def _match_pass(qm, qp, rm, rp, tol, shift, used_q, used_r):
    """Match unused query peaks to unused reference peaks at r.mz + shift; return entropy gain."""
    n, m = len(qm), len(rm)
    gain = 0.0
    j = 0
    for i in range(n):
        if used_q[i]:
            continue
        target = qm[i] - shift
        while j < m and rm[j] < target - tol:
            j += 1
        best = -1
        bd = tol + 1.0
        k = j
        while k < m and rm[k] <= target + tol:
            if not used_r[k]:
                d = abs(rm[k] - target)
                if d < bd:
                    bd = d
                    best = k
            k += 1
        if best >= 0:
            used_q[i] = True
            used_r[best] = True
            a = qp[i] * 0.5
            b = rp[best] * 0.5
            gain += _xlogx(a + b) - _xlogx(a) - _xlogx(b)
    return gain


@njit(cache=True)
def _entropy_sim(qm, qp, rm, rp, tol, shift):
    if len(qm) == 0 or len(rm) == 0:
        return 0.0
    used_q = np.zeros(len(qm), np.bool_)
    used_r = np.zeros(len(rm), np.bool_)
    g = _match_pass(qm, qp, rm, rp, tol, 0.0, used_q, used_r)
    if shift != 0.0:
        g += _match_pass(qm, qp, rm, rp, tol, shift, used_q, used_r)
    s = g / LN2
    return 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)


@njit(parallel=True, cache=True)
def sim_many(qm, qp, off, allmz, allp, idx, tol, shifts):
    """Similarity of one query against reference rows `idx`; shifts[t] = neutral-mass shift for
    reference t (0 = direct matching only)."""
    out = np.zeros(len(idx), np.float32)
    for t in prange(len(idx)):
        r = idx[t]
        a, b = off[r], off[r + 1]
        out[t] = _entropy_sim(qm, qp, allmz[a:b], allp[a:b], tol, shifts[t])
    return out


def entropy_similarity(q, r, tol=0.01, shift=0.0):
    return float(_entropy_sim(q[0], q[1], r[0], r[1], tol, shift))


if __name__ == "__main__":
    a = prepare([100.0, 150.0, 200.0], [0.2, 1.0, 0.5], 300.0)
    b = prepare([300.0, 350.0], [1.0, 1.0], 400.0)
    c = prepare([114.0157, 164.0157, 214.0157], [0.2, 1.0, 0.5], 314.0157)  # a shifted by +CH2
    assert abs(entropy_similarity(a, a) - 1.0) < 1e-6
    assert entropy_similarity(a, b) == 0.0
    assert entropy_similarity(c, a) == 0.0
    assert abs(entropy_similarity(c, a, shift=14.0157) - 1.0) < 1e-6
    print("kernels OK")
