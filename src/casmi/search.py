"""Retrieval engine: candidate generation from our pool + library / analog / fingerprint channels.

All structures are identified by the metric key (casmi.metric). The reference library is the CSR
array set from casmi.library; the pool is data/artifacts/pool (exact-mass sorted).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .simkernels import prepare, sim_many
from .spectra import ADDUCT_SHIFT
from .library import ADDUCTS, ENVEDA_POS_SHIFT

POP8 = np.array([bin(i).count("1") for i in range(256)], np.uint8)
ECFP4_BYTES = 4096 // 8


@dataclass
class Query:
    """One unknown molecule: its spectra, already m/z-corrected."""

    mid: str
    mz: list
    p: list
    adduct: list
    mode: list           # +1 / -1
    prec: list
    ce: list = field(default_factory=list)
    raw: list = field(default_factory=list)   # (mz, intensity) as acquired (m/z-corrected), for the FP model

    @property
    def neutral_mass(self) -> float:
        return float(np.median([pm - ADDUCT_SHIFT[a] for pm, a in zip(self.prec, self.adduct)]))


def make_query(mid, spectra, enveda=True, floor=0.002, top_k=64):
    """spectra: iterable of dicts with mz, it, adduct, mode ('positive'/'negative'), prec, ce."""
    q = Query(mid, [], [], [], [], [], [])
    for s in spectra:
        if s["adduct"] not in ADDUCT_SHIFT:
            continue
        pos = s["mode"] == "positive"
        sh = ENVEDA_POS_SHIFT if (enveda and pos) else 0.0
        m, p = prepare(np.asarray(s["mz"]) + sh, s["it"], s["prec"] + sh, floor=floor, top_k=top_k)
        if len(m) == 0:
            continue
        q.mz.append(m)
        q.p.append(p)
        q.adduct.append(s["adduct"])
        q.mode.append(1 if pos else -1)
        q.prec.append(s["prec"] + sh)
        q.ce.append(s.get("ce", np.nan))
        q.raw.append((np.asarray(s["mz"], np.float64) + sh, np.asarray(s["it"], np.float64)))
    return q


class Pool:
    def __init__(self, pool_dir):
        self.df = pd.read_parquet(f"{pool_dir}/pool.parquet")
        self.mass = self.df.exact_mass.values
        self.key = self.df.key.values
        self.row_of_key = pd.Series(np.arange(len(self.df)), index=self.key)
        self.fp = np.load(f"{pool_dir}/fp_full.npy", mmap_mode="r")
        self._e4 = None
        import os
        self.pop = np.load(f"{pool_dir}/pop.npy") if os.path.exists(f"{pool_dir}/pop.npy") else None
        self.np_like = np.load(f"{pool_dir}/np.npy") if os.path.exists(f"{pool_dir}/np.npy") else None

    @property
    def ecfp4(self):
        if self._e4 is None:
            self._e4 = np.ascontiguousarray(self.fp[:, :ECFP4_BYTES])
        return self._e4

    def window(self, m, ppm=10.0, min_da=0.002, center_ppm=0.0):
        """Pool rows with exact mass within ±ppm of m·(1 + center_ppm·1e-6) (centre = known bias of the
        instrument: timsTOF truths sit ~0.8 ppm below the measured neutral mass)."""
        c = m * (1 + center_ppm * 1e-6)
        tol = max(m * ppm * 1e-6, min_da)
        a, b = np.searchsorted(self.mass, [c - tol, c + tol])
        return np.arange(a, b)


def tanimoto(a, b):
    """Packed fingerprints a (n, B), b (m, B) -> (n, m) Tanimoto."""
    inter = POP8[a[:, None, :] & b[None, :, :]].sum(-1, dtype=np.int32)
    na = POP8[a].sum(-1, dtype=np.int32)
    nb = POP8[b].sum(-1, dtype=np.int32)
    return inter / np.maximum(na[:, None] + nb[None, :] - inter, 1)


class Library:
    def __init__(self, lib: dict, pool: Pool):
        self.L = lib
        self.n = len(lib["row"])
        add = np.array(ADDUCTS)[lib["adduct_code"]]
        self.neutral = lib["prec_mz"] - np.array([ADDUCT_SHIFT[a] for a in add])
        self.key = lib["keys"][lib["key_code"]]
        prow = pool.row_of_key.reindex(lib["keys"]).values
        self.pool_row = np.where(np.isnan(prow), -1, prow).astype(np.int64)[lib["key_code"]]
        self.by_adduct = {}
        for a in np.unique(lib["adduct_code"]):
            idx = np.flatnonzero(lib["adduct_code"] == a)
            self.by_adduct[ADDUCTS[a]] = idx[np.argsort(self.neutral[idx])]

    def rows_of_keys(self, keys):
        """Library rows of each structure key -> (rows, owner index into `keys`). Uses the integer key
        codes (CSR over library rows sorted by code), not string search over 2 M rows."""
        if not hasattr(self, "_code_off"):
            kc = self.L["key_code"]
            self._code_rows = np.argsort(kc, kind="stable")
            self._code_off = np.concatenate([[0], np.cumsum(np.bincount(kc, minlength=len(self.L["keys"])))])
        uk = self.L["keys"]
        keys = np.asarray(keys).astype(uk.dtype)
        pos = np.clip(np.searchsorted(uk, keys), 0, len(uk) - 1)
        hit = uk[pos] == keys
        lo = np.where(hit, self._code_off[pos], 0)
        hi = np.where(hit, self._code_off[pos + 1], 0)
        n = hi - lo
        if n.sum() == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        owner = np.repeat(np.arange(len(keys)), n)
        rows = self._code_rows[np.concatenate([np.arange(a, b) for a, b in zip(lo, hi) if b > a])]
        return rows.astype(np.int64), owner

    def candidates_for(self, adduct, m, max_shift=None):
        idx = self.by_adduct.get(adduct)
        if idx is None:
            return np.zeros(0, np.int64)
        if max_shift is None:
            return idx
        nm = self.neutral[idx]
        a, b = np.searchsorted(nm, [m - max_shift, m + max_shift])
        return idx[a:b]

    def sims(self, qm, qp, qneutral, refs, tol=0.01, shifted=True):
        shifts = (qneutral - self.neutral[refs]).astype(np.float64) if shifted else np.zeros(len(refs))
        return sim_many(qm, qp, self.L["off"], self.L["mz"], self.L["p"], refs.astype(np.int64), tol, shifts)


def analog_hits(q: Query, lib: Library, top=300, max_shift=300.0, tol=0.01):
    """Per query spectrum: top reference spectra by (direct + mass-shifted) entropy similarity.
    Returns DataFrame(spec, ref, sim, key, pool_row, delta)."""
    out = []
    M = q.neutral_mass
    for s, (m, p, a) in enumerate(zip(q.mz, q.p, q.adduct)):
        refs = lib.candidates_for(a, M, max_shift)
        if len(refs) == 0:
            continue
        sim = lib.sims(m, p, M, refs, tol=tol)
        k = min(top, len(refs))
        sel = np.argpartition(-sim, k - 1)[:k]
        r = refs[sel]
        out.append(pd.DataFrame({"spec": s, "ref": r, "sim": sim[sel], "key": lib.key[r],
                                 "pool_row": lib.pool_row[r], "delta": M - lib.neutral[r]}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["spec", "ref", "sim", "key", "pool_row", "delta"])
