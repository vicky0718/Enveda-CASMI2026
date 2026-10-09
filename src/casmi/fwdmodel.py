"""Forward model (structure -> spectrum), NEIMS-style (Wei et al. 2019), trained by us on the competition data.

A candidate's fingerprint bits plus the measurement conditions (adduct, instrument family, collision energy,
polarity, precursor m/z) predict two 1-Da-binned spectra: fragment m/z ("forward") and loss from the
precursor ("reverse", indexed by precursor − m/z, which carries the neutral losses). A candidate is scored by
the cosine between its prediction and the query spectrum binned the same way — the opposite direction to the
fingerprint model (spectrum -> bits), so the two err differently.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fpmodel import ADDUCT_IX, ADDUCT_LIST, INSTR_LIST, instr_family

NBINS = 1000  # 0 .. 999 Da (fragments above 1000 Da or losses above 1000 Da are dropped)


def bin_spectrum(mz, it, prec, nbins=NBINS):
    """(forward, reverse) 1-Da binned, sqrt-intensity, L2-normalised vectors (numpy)."""
    mz = np.asarray(mz, float)
    it = np.sqrt(np.clip(np.asarray(it, float), 0, None))
    keep = (mz <= prec - 0.5) & (it > 0)
    mz, it = mz[keep], it[keep]
    f = np.zeros(nbins, np.float32)
    r = np.zeros(nbins, np.float32)
    fi = np.floor(mz).astype(int)
    ri = np.floor(prec - mz).astype(int)
    ok = (fi >= 0) & (fi < nbins)
    np.add.at(f, fi[ok], it[ok])
    ok = (ri >= 0) & (ri < nbins)
    np.add.at(r, ri[ok], it[ok])
    return f / max(np.linalg.norm(f), 1e-9), r / max(np.linalg.norm(r), 1e-9)


class Res(nn.Module):
    def __init__(self, d, drop):
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.f = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d))

    def forward(self, x):
        return x + self.f(self.n(x))


class FwdNet(nn.Module):
    def __init__(self, nbits, d=2048, blocks=3, drop=0.1, nbins=NBINS):
        super().__init__()
        self.inp = nn.Linear(nbits, d)
        self.ad = nn.Embedding(len(ADDUCT_LIST), d)
        self.ins = nn.Embedding(len(INSTR_LIST), d)
        self.cond = nn.Linear(3, d)  # collision energy / 100, polarity, precursor / 1000
        self.blocks = nn.ModuleList([Res(d, drop) for _ in range(blocks)])
        self.norm = nn.LayerNorm(d)
        self.fwd = nn.Linear(d, nbins)
        self.rev = nn.Linear(d, nbins)

    def forward(self, bits, ad, ins, ce, mode, prec):
        h = self.inp(bits) + self.ad(ad) + self.ins(ins) + self.cond(
            torch.stack([ce / 100.0, mode, prec / 1000.0], -1))
        h = F.gelu(h)
        for b in self.blocks:
            h = b(h)
        h = self.norm(h)
        return F.softplus(self.fwd(h)), F.softplus(self.rev(h))


def load(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = FwdNet(int(ck["nbits"]), d=int(ck["d"]), blocks=int(ck["blocks"]))
    net.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    return net.to(device).eval()


@torch.no_grad()
def scores(net, cand_bits, q, weights=None, device="cpu", batch=4096):
    """Per candidate: (weighted) mean over the query's spectra of 0.5·(cos_forward + cos_reverse).
    cand_bits: (n, nbits) 0/1 matrix in the model's bit order; q: casmi.search.Query (raw peaks used)."""
    n = len(cand_bits)
    if n == 0 or not q.raw:
        return np.zeros(n)
    X = torch.as_tensor(np.asarray(cand_bits, np.float32), device=device)
    w = np.ones(len(q.raw)) / len(q.raw) if weights is None else np.asarray(weights, float)
    out = np.zeros(n)
    instr = getattr(q, "instr", None) or ["timsTOF"] * len(q.raw)
    for s, ((mz, it), pm, ad, ce, mode, ins) in enumerate(zip(q.raw, q.prec, q.adduct, q.ce, q.mode, instr)):
        fq, rq = bin_spectrum(mz, it, pm)
        if not fq.any():
            continue
        fq, rq = torch.as_tensor(fq, device=device), torch.as_tensor(rq, device=device)
        sim = np.zeros(n)
        for a in range(0, n, batch):
            xb = X[a:a + batch]
            m = len(xb)
            T = lambda v, dt=torch.float32: torch.full((m,), v, dtype=dt, device=device)  # noqa: E731
            pf, pr = net(xb, T(ADDUCT_IX.get(ad, ADDUCT_IX["<unk>"]), torch.long), T(instr_family(ins), torch.long),
                         T(0.0 if np.isnan(ce) else float(ce)), T(1.0 if mode > 0 else 0.0), T(float(pm)))
            cf = F.cosine_similarity(pf, fq[None, :], dim=1)
            cr = F.cosine_similarity(pr, rq[None, :], dim=1)
            sim[a:a + m] = (0.5 * (cf + cr)).cpu().numpy()
        out += w[s] * sim
    return out
