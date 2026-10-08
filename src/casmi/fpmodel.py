"""Spectrum -> fingerprint transformer (architecture as described in the public, CC0 prvsiyan
"Analog Propagation" notebook; trained by us from scratch on the competition data, see
kaggle/fp_train/).

Candidates are ranked by f·z (the Bayes log-likelihood of the fingerprint bits up to a constant).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_PEAKS = 128
ADDUCT_LIST = ["[M+H]+", "[M+NH4]+", "[M+Na]+", "[M+K]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M]+",
               "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+C2H4O2-H]-", "[M+Cl]-", "[M]-",
               "[M+2H]2+", "[M-2H]-", "[2M+H]+", "[2M+Na]+", "[2M+NH4]+", "[2M-H]-", "[2M+K]+",
               "[2M+CH2O2-H]-", "[2M+C2H4O2-H]-", "[2M+Na-2H]-", "[M+Na-2H]-", "[M-H2O]+", "<unk>"]
ADDUCT_IX = {a: i for i, a in enumerate(ADDUCT_LIST)}
INSTR_LIST = ["timsTOF", "Orbitrap", "QTOF", "IT", "other"]


def instr_family(s) -> int:
    if s is None:
        return 4
    t = str(s).lower()
    if "timstof" in t:
        return 0
    if any(k in t for k in ("orbitrap", "qft", "ftms", "hybrid ft", "itft", "exactive")):
        return 1
    if "tof" in t:
        return 2
    if "trap" in t or "qq" in t:
        return 3
    return 4


def prep_peaks(mz, inten, prec_mz, max_peaks=MAX_PEAKS, floor=1e-3, win=50.0, per_win=8):
    """Drop peaks above precursor+1.5, floor, window-diversified top-N, sort by m/z, sqrt intensity."""
    mz = np.asarray(mz, np.float64)
    it = np.asarray(inten, np.float64)
    keep = mz <= prec_mz + 1.5
    mz, it = mz[keep], it[keep]
    if len(mz) == 0 or it.max() <= 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = it >= floor * it.max()
    mz, it = mz[keep], it[keep]
    if len(mz) > 8 * max_peaks:  # bound the per-peak Python loop below on very dense spectra
        sel = np.argpartition(-it, 8 * max_peaks)[:8 * max_peaks]
        mz, it = mz[sel], it[sel]
    if len(mz) > max_peaks:
        order = np.argsort(-it)
        bucket = (mz // win).astype(np.int64)
        cnt, sel = {}, []
        for i in order:
            c = cnt.get(bucket[i], 0)
            if c < per_win:
                cnt[bucket[i]] = c + 1
                sel.append(i)
        sel = np.array(sel)
        if len(sel) > max_peaks:
            sel = sel[np.argsort(-it[sel])[:max_peaks]]
        elif len(sel) < max_peaks:
            chosen = set(sel.tolist())
            rest = np.array([i for i in order if i not in chosen])
            if len(rest):
                sel = np.concatenate([sel, rest[:max_peaks - len(sel)]])
        mz, it = mz[sel], it[sel]
    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    return mz.astype(np.float32), np.sqrt(it / it.max()).astype(np.float32)


class SinEmb(nn.Module):
    def __init__(self, dim, lo=-2.0, hi=3.2, power=1.0):
        super().__init__()
        n = dim // 2
        wav = torch.pow(10.0, (hi - lo) * torch.pow(torch.linspace(0, 1, n), power) + lo)
        self.register_buffer("inv", (2 * math.pi) / wav)

    def forward(self, x):
        a = x.unsqueeze(-1) * self.inv
        return torch.cat([torch.sin(a), torch.cos(a)], -1)


class Block(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.h = h
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, pad, bias=None):
        B, N, D = x.shape
        y = self.n1(x)
        q, k, v = self.qkv(y).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        mask = (~pad)[:, None, None, :]
        if bias is not None:  # additive per-head bias from pairwise m/z differences; padded keys masked
            mask = bias.to(q.dtype).masked_fill(pad[:, None, None, :], float("-inf"))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D)))
        return x + self.drop(self.ff(self.n2(x)))


class PairBias(nn.Module):
    """Per-layer, per-head attention bias from the m/z difference of every token pair (the global token
    stands at the precursor m/z, so its row carries the neutral losses). Fragment-to-fragment losses
    (sugars 162.053 / 146.058 / 132.042, CO vs C2H4, CH2, H2O, CO2 ...) become directly visible to
    attention instead of having to be inferred from two absolute positions."""

    def __init__(self, layers, heads, dim=32, hidden=32):
        super().__init__()
        self.layers, self.heads = layers, heads
        self.emb = SinEmb(dim, lo=-2.0, hi=2.7)  # wavelengths 0.01 .. 500 Da
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, layers * heads))

    def forward(self, tok_mz):
        d = (tok_mz[:, :, None] - tok_mz[:, None, :]).abs()  # (B, N, N)
        b = self.mlp(self.emb(d))  # (B, N, N, L*H)
        B, N = tok_mz.shape
        return b.view(B, N, N, self.layers, self.heads).permute(3, 0, 4, 1, 2)  # (L, B, H, N, N)


# typical element counts, to scale formula vectors (order = formula.ANNOT_ELS: C H N O P S F Cl Br I Na K)
FORMULA_SCALE = (40.0, 80.0, 8.0, 20.0, 2.0, 3.0, 6.0, 3.0, 2.0, 2.0, 1.0, 1.0)
N_ELS = len(FORMULA_SCALE)


class FPNet(nn.Module):
    def __init__(self, nbits, d=512, layers=6, heads=8, drop=0.1, rel=False, formula=False):
        super().__init__()
        self.d = d
        self.pair = PairBias(layers, heads) if rel else None
        if formula:  # per peak: fragment sub-formula, loss formula, annotated flag; precursor formula on the global token
            self.register_buffer("fscale", torch.tensor(FORMULA_SCALE))
            self.fpeak = nn.Linear(2 * N_ELS + 1, d)
            self.fprec = nn.Linear(N_ELS, d)
        self.formula = formula
        self.mz_emb = SinEmb(d)
        self.nl_emb = SinEmb(d)
        self.pk = nn.Linear(2 * d + 1, d)
        self.prec_emb = SinEmb(d)
        self.ad = nn.Embedding(len(ADDUCT_LIST), d)
        self.ins = nn.Embedding(len(INSTR_LIST), d)
        self.gl = nn.Linear(d + 3, d)
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, 2048), nn.GELU(), nn.Dropout(drop), nn.Linear(2048, nbits))

    def forward(self, mz, it, pad, prec, ad, ins, ce, mode, frag=None, pform=None):
        """frag: (B, N, N_ELS) sub-formula counts per peak (zeros = unexplained); pform: (B, N_ELS) precursor
        ion formula — both required when the model was built with formula=True."""
        B, N = mz.shape
        nl = (prec[:, None] - mz).clamp(min=0)
        p = self.pk(torch.cat([self.mz_emb(mz), self.nl_emb(nl), it.unsqueeze(-1)], -1))
        g = self.gl(torch.cat([self.prec_emb(prec), (ce / 100.0).unsqueeze(-1), mode.unsqueeze(-1),
                               torch.log1p(prec).unsqueeze(-1) / 10.0], -1)) + self.ad(ad) + self.ins(ins)
        if self.formula:
            fr, pf = frag.float(), pform.float()
            ann = (fr.sum(-1, keepdim=True) > 0).float()
            loss = (pf[:, None, :] - fr) * ann  # neutral loss formula of annotated peaks
            p = p + self.fpeak(torch.cat([fr / self.fscale, loss / self.fscale, ann], -1))
            g = g + self.fprec(pf / self.fscale)
        x = torch.cat([g.unsqueeze(1), p], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], 1)
        bias = self.pair(torch.cat([prec[:, None], mz], 1).float()) if self.pair is not None else None
        for i, b in enumerate(self.blocks):
            x = b(x, pad, None if bias is None else bias[i])
        x = self.norm(x)
        msk = (~pad[:, 1:]).float().unsqueeze(-1)
        mean = (x[:, 1:] * msk).sum(1) / msk.sum(1).clamp(min=1)
        return self.head(torch.cat([x[:, 0], mean], -1))


def load(path, device="cpu"):
    """Load one of our own checkpoints ({model, nbits, d, layers}) with the tensor-only loader."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = FPNet(int(ck["nbits"]), d=int(ck["d"]), layers=int(ck["layers"]), rel=bool(ck.get("rel", False)),
                formula=bool(ck.get("formula", False)))
    net = net.to(device).eval()
    # keep the float32 sinusoidal tables built by the constructor (fp16 checkpoints destroy them)
    net.load_state_dict({k: v.float() for k, v in ck["model"].items() if not k.endswith(".inv")}, strict=False)
    return net


def merge_peaks(spectra):
    """Collapse a molecule's spectra into one peak list (near-duplicate m/z merged, stronger kept)."""
    mz = np.concatenate([np.asarray(m, float) for m, _ in spectra])
    it = np.concatenate([np.asarray(i, float) / max(float(np.max(i)), 1e-9) for _, i in spectra])
    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    keep = np.ones(len(mz), bool)
    for j in range(1, len(mz)):
        if mz[j] - mz[j - 1] < 0.005:
            if it[j] >= it[j - 1]:
                keep[j - 1] = False
            else:
                keep[j] = False
    return mz[keep], it[keep]


@torch.no_grad()
def logits(nets, peak_lists, precs, adducts, instruments, ces, modes, device="cpu", frags=None, pforms=None):
    """Mean logits over nets for a batch of (already prepared) peak lists. For formula models, frags is a list
    of (n_peaks, N_ELS) annotation arrays aligned with the peak lists and pforms the precursor formulas."""
    B = len(peak_lists)
    N = max(1, max(len(a) for a, _ in peak_lists))
    mz = np.zeros((B, N), np.float32)
    it = np.zeros((B, N), np.float32)
    pad = np.ones((B, N), bool)
    for r, (a, b) in enumerate(peak_lists):
        mz[r, :len(a)], it[r, :len(a)], pad[r, :len(a)] = a, b, False
    pad[pad.all(1), 0] = False
    T = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=device)  # noqa: E731
    inp = (T(mz), T(it), T(pad, torch.bool), T(precs), T([ADDUCT_IX.get(a, ADDUCT_IX["<unk>"]) for a in adducts],
                                                        torch.long),
           T([instr_family(s) for s in instruments], torch.long), T(ces), T(modes))
    extra = ()
    if frags is not None:
        fr = np.zeros((B, N, N_ELS), np.float32)
        for r, a in enumerate(frags):
            fr[r, :len(a)] = a
        extra = (T(fr), T(np.asarray(pforms, np.float32)))
    return torch.stack([n(*inp, *(extra if n.formula else ())).float() for n in nets]).mean(0).cpu().numpy()
