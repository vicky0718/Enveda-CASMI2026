"""MetFrag-lite: combinatorial bond-cleavage fragments of a candidate, matched to observed peaks.

Fragments: one cut of any acyclic bond, two cuts of bonds in the same ring (ring opening), and two
acyclic cuts (middle pieces). Each fragment's neutral mass is the sum of its heavy atoms with their
hydrogens; ions are frag + h·H ± proton for h in H_SHIFTS (hydrogen rearrangements on cleavage).
Score = intensity-weighted fraction of peaks explained, one-cut fragments weighted 1, two-cut
fragments 0.6 (parsimony), within `tol` Da.
"""

from collections import deque

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
H = 1.00782503207
PROTON = 1.00727646688
H_SHIFTS = np.array([-2, -1, 0, 1, 2, 3])
_MASS = {}


def _atom_mass(a):
    sym = a.GetSymbol()
    if sym not in _MASS:
        _MASS[sym] = Chem.GetPeriodicTable().GetMostCommonIsotopeMass(sym)
    return _MASS[sym] + a.GetTotalNumHs() * H


def _component(adj, start, cut, n):
    seen = np.zeros(n, bool)
    seen[start] = True
    dq = deque([start])
    while dq:
        u = dq.popleft()
        for v, b in adj[u]:
            if b in cut or seen[v]:
                continue
            seen[v] = True
            dq.append(v)
    return seen


def fragments(smiles, max_pairs=1500):
    """-> (masses, weights): unique neutral fragment masses with parsimony weights."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return np.zeros(0), np.zeros(0)
    n = mol.GetNumAtoms()
    am = np.array([_atom_mass(a) for a in mol.GetAtoms()])
    total = am.sum()
    adj = [[] for _ in range(n)]
    bonds = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i].append((j, b.GetIdx()))
        adj[j].append((i, b.GetIdx()))
        bonds.append((i, j, b.IsInRing()))
    out = {}

    def add(m, w):
        k = round(m, 4)
        if out.get(k, 0) < w:
            out[k] = w

    bridges = [bi for bi, (_, _, r) in enumerate(bonds) if not r]
    side = {}
    for bi in bridges:
        i, j, _ = bonds[bi]
        s = _component(adj, i, {bi}, n)
        side[bi] = s
        m = am[s].sum()
        add(m, 1.0)
        add(total - m, 1.0)
    ri = mol.GetRingInfo()
    pairs = 0
    for ring in ri.BondRings():
        ring = list(ring)
        for a in range(len(ring)):
            for b in range(a + 1, len(ring)):
                if pairs >= max_pairs:
                    break
                i = bonds[ring[a]][0]
                s = _component(adj, i, {ring[a], ring[b]}, n)
                if s.all():
                    continue  # fused ring: still connected
                m = am[s].sum()
                add(m, 0.6)
                add(total - m, 0.6)
                pairs += 1
    for a in range(len(bridges)):
        for b in range(a + 1, len(bridges)):
            if pairs >= 2 * max_pairs:
                break
            sa, sb = side[bridges[a]], side[bridges[b]]
            # the piece between the two cuts: atoms on the "other" side of both, picked consistently
            for x in (sa, ~sa):
                for y in (sb, ~sb):
                    mid = x & y
                    if mid.any() and not (mid == x).all() and not (mid == y).all():
                        add(am[mid].sum(), 0.6)
            pairs += 1
    if not out:
        return np.zeros(0), np.zeros(0)
    k = np.array(list(out.keys()))
    return k, np.array([out[x] for x in k])


def ion_table(masses, weights, mode):
    """Fragment ion m/z (sorted) and weights for all H shifts."""
    if len(masses) == 0:
        return np.zeros(0), np.zeros(0)
    sign = PROTON if mode > 0 else -PROTON
    mz = (masses[:, None] + H_SHIFTS[None, :] * H + sign).ravel()
    w = np.repeat(weights, len(H_SHIFTS))
    o = np.argsort(mz)
    return mz[o], w[o]


def explain(peak_mz, peak_p, ion_mz, ion_w, tol=0.005):
    """Intensity-weighted explained fraction of peaks."""
    if len(ion_mz) == 0 or len(peak_mz) == 0:
        return 0.0
    idx = np.searchsorted(ion_mz, peak_mz)
    best = np.zeros(len(peak_mz))
    for off in (-1, 0):
        j = np.clip(idx + off, 0, len(ion_mz) - 1)
        ok = np.abs(ion_mz[j] - peak_mz) <= tol
        best = np.maximum(best, np.where(ok, ion_w[j], 0.0))
    # also check idx+1 for exact ties
    j = np.clip(idx + 1, 0, len(ion_mz) - 1)
    best = np.maximum(best, np.where(np.abs(ion_mz[j] - peak_mz) <= tol, ion_w[j], 0.0))
    return float((best * peak_p).sum() / max(peak_p.sum(), 1e-12))


def frag_scores(smiles_list, q, tol=0.005):
    """Per candidate: mean over the molecule's spectra of the explained fraction (precursor-region
    peaks excluded)."""
    out = np.zeros(len(smiles_list))
    for c, s in enumerate(smiles_list):
        m, w = fragments(s)
        tabs = {}
        sc = []
        for mz, p, mode, pm in zip(q.mz, q.p, q.mode, q.prec):
            keep = mz < pm - 1.5
            if keep.sum() == 0:
                continue
            if mode not in tabs:
                tabs[mode] = ion_table(m, w, mode)
            sc.append(explain(mz[keep].astype(np.float64), p[keep].astype(np.float64), *tabs[mode], tol=tol))
        out[c] = np.mean(sc) if sc else 0.0
    return out


def match_weights(peak_mz, ion_mz, ion_w, tol=0.005):
    """Per peak: parsimony weight of the best fragment ion within `tol` (0 if unexplained)."""
    best = np.zeros(len(peak_mz))
    if len(ion_mz) == 0 or len(peak_mz) == 0:
        return best
    idx = np.searchsorted(ion_mz, peak_mz)
    for off in (-1, 0, 1):
        j = np.clip(idx + off, 0, len(ion_mz) - 1)
        best = np.maximum(best, np.where(np.abs(ion_mz[j] - peak_mz) <= tol, ion_w[j], 0.0))
    return best


def frag_matrix(smiles_list, q, tol=0.005):
    """(M, pw): M[c, k] = how candidate c explains fragment peak k (concatenated over the molecule's
    spectra, precursor region excluded); pw[k] = peak weight, each spectrum summing to 1 / n_spectra.
    frag score = M @ pw (identical to frag_scores)."""
    peaks, modes, pws = [], [], []
    for mz, p, mode, pm in zip(q.mz, q.p, q.mode, q.prec):
        keep = mz < pm - 1.5
        if keep.sum() == 0:
            continue
        pk = p[keep].astype(np.float64)
        peaks.append(mz[keep].astype(np.float64))
        modes.append(mode)
        pws.append(pk / max(pk.sum(), 1e-12))
    if not peaks:
        return np.zeros((len(smiles_list), 0), np.float32), np.zeros(0)
    pw = np.concatenate(pws) / len(peaks)
    M = np.zeros((len(smiles_list), len(pw)), np.float32)
    for c, s in enumerate(smiles_list):
        m, w = fragments(s)
        tabs = {}
        cols = []
        for mz, mode in zip(peaks, modes):
            if mode not in tabs:
                tabs[mode] = ion_table(m, w, mode)
            cols.append(match_weights(mz, *tabs[mode], tol=tol))
        M[c] = np.concatenate(cols)
    return M, pw


def frag_disc(M, pw, masses, tol=0.002):
    """Isomer-discriminative fragment evidence: explained peak intensity weighted by the share of
    same-formula (same exact mass) candidates that do NOT explain that peak. Peaks every isomer explains
    carry no information about which isomer it is."""
    out = np.zeros(len(M))
    if M.shape[1] == 0:
        return out
    order = np.argsort(masses)
    ms = np.asarray(masses)[order]
    start = 0
    for i in range(1, len(ms) + 1):
        if i == len(ms) or ms[i] - ms[start] > tol:
            grp = order[start:i]
            if len(grp) > 1:
                e = (M[grp] > 0).astype(np.float32)
                share_explaining = e.mean(0)
                out[grp] = (M[grp] * (1.0 - share_explaining)[None, :]) @ pw
            start = i
    return out
