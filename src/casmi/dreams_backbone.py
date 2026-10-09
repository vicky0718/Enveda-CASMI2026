"""DreaMS backbone (Bushuiev et al., Nat. Biotechnol. 2025) as a plain torch module, for fine-tuning on our data.

The layer code below is adapted from the official release, github.com/pluskal-lab/DreaMS @ dbec3a0b
(dreams/models/dreams/{dreams,layers}.py, dreams/models/layers/{fourier_features,feed_forward}.py),
MIT License, Copyright (c) the DreaMS developers. Parameter names match the official modules, so the
self-supervised GeMS checkpoint (ssl_model.ckpt, Zenodo 10997887 / huggingface.co/roman-bushuiev/DreaMS) loads
with a strict key check. Only the forward pass of the encoder is kept (no Lightning, no SSL heads, no data code).

Input: (B, 1 + n, 2) float — token 0 is the precursor (m/z, 1.1), then up to n peaks (m/z in Da, intensity
relative to the base peak, ≤ 1); padding rows are all zero (m/z == 0 marks padding).
Output: (B, 1 + n, d_model); the precursor token [:, 0] is the spectrum embedding.
"""

import pickle
import types
from argparse import Namespace
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

PREC_INTENS = 1.1  # official SpectrumPreprocessor precursor-token intensity


class FourierFeatures(nn.Module):
    def __init__(self, strategy, x_min, x_max, trainable=True, funcs="both", sigma=10, num_freqs=512):
        super().__init__()
        assert strategy in {"random", "voronov_et_al", "lin_float_int"} and x_min < 1
        self.funcs = funcs
        if strategy == "random":
            b = torch.randn(num_freqs) * sigma
        elif strategy == "voronov_et_al":
            b = torch.tensor([1 / (x_min * (x_max / x_min) ** (2 * i / (num_freqs - 2))) for i in range(1, num_freqs)])
        else:  # lin_float_int
            b = torch.tensor([1 / (x_min * i) for i in range(2, ceil(1 / x_min), 2)] +
                             [1 / (1 * i) for i in range(2, ceil(x_max), 1)])
        self.b = nn.Parameter(b.unsqueeze(0), requires_grad=trainable)
        self.register_parameter("Fourier frequencies", self.b)

    def forward(self, x):
        x = 2 * torch.pi * x @ self.b
        if self.funcs == "both":
            return torch.cat((torch.cos(x), torch.sin(x)), dim=-1)
        return torch.cos(x) if self.funcs == "cos" else torch.sin(x)

    def num_features(self):
        return self.b.shape[1] if self.funcs != "both" else 2 * self.b.shape[1]


def _interpolate(a, b, n):
    return [int(round(v)) for v in np.linspace(a, b, n + 2)[1:-1]]


class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, depth=None, act_last=True, act=nn.ReLU, bias=True, dropout=0):
        super().__init__()
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * depth
        elif hidden_dim == "interpolated":
            hidden_dim = _interpolate(in_dim, out_dim, depth - 1)
        else:
            depth = len(hidden_dim)
        layers = []
        for i in range(depth):
            d1 = hidden_dim[i - 1] if i != 0 else in_dim
            d2 = hidden_dim[i] if i != depth - 1 else out_dim
            layers.append(nn.Linear(d1, d2, bias=bias))
            if i != depth - 1:
                layers.append(nn.Dropout(p=dropout))
            if i != depth - 1 or act_last:
                layers.append(act())
        self.ff = nn.Sequential(*layers)

    def forward(self, x):
        return self.ff(x)


class MultiheadAttention(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.d_model, self.n_heads, self.dropout = a.d_model, a.n_heads, a.att_dropout
        self.use_bias = not a.no_transformer_bias
        self.attn_mech = a.attn_mech
        self.d_graphormer_params = a.d_graphormer_params
        self.head_dim = self.d_model // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.weights = Parameter(torch.empty(4 * self.d_model, self.d_model))
        nn.init.normal_(self.weights, 0, (2 / (5 * self.d_model)) ** 0.5)
        if self.use_bias:
            self.biases = Parameter(torch.zeros(4 * self.d_model))
        if self.d_graphormer_params:
            self.lin_graphormer = nn.Linear(self.d_graphormer_params, self.n_heads, bias=False)
        if self.attn_mech == "additive_v":
            self.additive_v = Parameter(torch.empty(self.n_heads, self.head_dim))
            nn.init.normal_(self.additive_v, 0, (2 / (5 * self.d_model)) ** 0.5)

    def _proj(self, x, start=0, end=None):
        return F.linear(x, self.weights[start:end], None if not self.use_bias else self.biases[start:end])

    def forward(self, x, mask, four=None):
        """`four`: (bs, n, d_fourier) Fourier token features. The official bias lin(f_i - f_j) over the
        (bs, n, n, d_fourier) difference tensor equals p_i - p_j with p = lin(f) (linear, no bias): computed
        per token, without the pairwise tensor."""
        bs, n, _ = x.size()
        q, k, v = self._proj(x, end=3 * self.d_model).chunk(3, dim=-1)
        sp = lambda t: t.reshape(bs, n, self.n_heads, self.head_dim).transpose(1, 2)  # noqa: E731
        q, k, v = sp(q), sp(k), sp(v)
        if self.attn_mech == "dot-product":
            att = torch.einsum("bhnd,bhdm->bhnm", q, k.transpose(-2, -1))
        else:
            att = q.unsqueeze(-2) - k.unsqueeze(-3)
            if self.attn_mech == "additive_v":
                att = att * self.additive_v.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            att = att.sum(dim=-1)
        att = att * self.scale
        if four is not None:
            if self.d_graphormer_params:
                p = self.lin_graphormer(four)  # (bs, n, heads)
                att = att + (p.unsqueeze(2) - p.unsqueeze(1)).permute(0, 3, 1, 2)
            else:
                p = four.sum(dim=-1)  # (bs, n)
                att = att + (p.unsqueeze(2) - p.unsqueeze(1)).unsqueeze(1)
        if mask is not None:  # official code masks query rows of padding tokens (keys stay visible)
            att = att.masked_fill(mask.unsqueeze(1).unsqueeze(-1), -1e9)
        att = F.dropout(F.softmax(att.float(), dim=-1).to(v.dtype), p=self.dropout, training=self.training)
        out = (att @ v).transpose(1, 2).reshape(bs, n, -1)
        return self._proj(out, start=3 * self.d_model)


class TFeedForward(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.dropout = a.ff_dropout
        bias = not a.no_transformer_bias
        self.in_proj = nn.Linear(a.d_model, 4 * a.d_model, bias=bias)
        self.out_proj = nn.Linear(4 * a.d_model, a.d_model, bias=bias)

    def forward(self, x):
        return self.out_proj(F.dropout(F.relu(self.in_proj(x)), p=self.dropout, training=self.training))


class ScaleNorm(nn.Module):
    def __init__(self, scale, eps=1e-5):
        super().__init__()
        self.scale = Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        return x * (self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps))


class TransformerEncoder(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.residual_dropout, self.n_layers, self.pre_norm = a.residual_dropout, a.n_layers, a.pre_norm
        self.atts = nn.ModuleList([MultiheadAttention(a) for _ in range(a.n_layers)])
        self.ffs = nn.ModuleList([TFeedForward(a) for _ in range(a.n_layers)])
        n_sc = a.n_layers * 2 + 1 if a.pre_norm else a.n_layers * 2
        self.scales = nn.ModuleList([ScaleNorm(a.d_model ** 0.5) if a.scnorm else nn.LayerNorm(a.d_model)
                                     for _ in range(n_sc)])
        self.checkpointing = False

    def _layer(self, i, x, mask, gd):
        att_scale, ff_scale = self.scales[2 * i], self.scales[2 * i + 1]
        r = x
        x = att_scale(x) if self.pre_norm else x
        x = r + F.dropout(self.atts[i](x, mask, gd), p=self.residual_dropout, training=self.training)
        x = x if self.pre_norm else att_scale(x)
        r = x
        x = ff_scale(x) if self.pre_norm else x
        x = r + F.dropout(self.ffs[i](x), p=self.residual_dropout, training=self.training)
        return x if self.pre_norm else ff_scale(x)

    def forward(self, x, mask, gd=None):
        x = F.dropout(x, p=self.residual_dropout, training=self.training)
        for i in range(self.n_layers):
            if self.checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(self._layer, i, x, mask, gd, use_reentrant=False)
            else:
                x = self._layer(i, x, mask, gd)
        return self.scales[-1](x) if self.pre_norm else x


class DreaMSBackbone(nn.Module):
    """Encoder part of the official `DreaMS` LightningModule (same parameter names)."""

    def __init__(self, args: Namespace, max_mz: float, max_tbxic_stdev: float):
        super().__init__()
        a = Namespace(**vars(args))
        a.d_model = sum(d for d in [a.d_fourier, a.d_peak, getattr(a, "d_mz_token", 0)] if d)
        a.d_graphormer_params = (a.d_fourier if a.d_fourier else 1) \
            if (a.graphormer_mz_diffs and a.graphormer_parametrized) else 0
        assert a.d_fourier and not a.vanilla_transformer, "only the published configuration is supported"
        self.a, self.max_mz, self.d_model = a, float(max_mz), a.d_model
        self.max_tbxic_stdev = float(max_tbxic_stdev)
        self.charge_feature = bool(a.charge_feature)
        self.fourier_enc = FourierFeatures(a.fourier_strategy, num_freqs=a.fourier_num_freqs,
                                           x_min=a.fourier_min_freq or max_tbxic_stdev, x_max=max_mz,
                                           trainable=a.fourier_trainable)
        self.ff_fourier = FeedForward(self.fourier_enc.num_features(), a.d_fourier, a.ff_fourier_d,
                                      depth=a.ff_fourier_depth, dropout=a.dropout, bias=not a.no_ffs_bias)
        self.ff_peak = FeedForward(3 if a.charge_feature else 2, a.d_peak, a.d_peak, depth=a.ff_peak_depth,
                                   dropout=a.dropout, bias=not a.no_ffs_bias)
        self.transformer_encoder = TransformerEncoder(a)

    def forward(self, spec, charge=None):
        pad = spec[:, :, 0] == 0
        if self.charge_feature:
            c = torch.ones(spec.shape[0], device=spec.device, dtype=spec.dtype) if charge is None else charge
            spec = torch.cat([spec, (~pad * c.unsqueeze(-1)).unsqueeze(-1).to(spec.dtype)], dim=-1)
        norm = torch.ones(spec.shape[-1], device=spec.device, dtype=spec.dtype)
        norm[0] = self.max_mz
        peak = self.ff_peak(spec / norm)
        # Fourier features of m/z up to 5,000 cycles/Da: phases must be computed in float32 (never fp16)
        with torch.autocast(device_type=spec.device.type, enabled=False):
            ff = self.fourier_enc(spec[..., [0]].float())
        four = self.ff_fourier(ff)
        x = torch.cat([peak.to(four.dtype), four], dim=-1)
        return self.transformer_encoder(x, pad, four if self.a.graphormer_mz_diffs else None)


# --- checkpoint loading without the dreams / msml packages ----------------------------------------------------

class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"_state": state})


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:  # noqa: BLE001 — classes of packages we do not install (dreams, msml)
            return type(name, (_Stub,), {"__module__": module})


_PICKLE = types.SimpleNamespace(Unpickler=_Unpickler, load=pickle.load, __name__="pickle")


def read_checkpoint(path):
    """(hparams args Namespace, dformat attrs dict, state_dict) of an official Lightning checkpoint."""
    ck = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_PICKLE)
    hp = ck["hyper_parameters"]
    args = hp["args"]
    df = getattr(args, "dformat", None)
    # data-format settings are class attributes of dreams.utils.dformats (not pickled): by class name
    known = {"DataFormatA": dict(max_mz=1000.0, max_tbxic_stdev=1e-4, max_prec_mz=1000.0, max_peaks_n=128),
             "DataFormatB": dict(max_mz=1500.0, max_tbxic_stdev=1e-3, max_prec_mz=1500.0, max_peaks_n=128),
             "DataFormatC": dict(max_mz=1500.0, max_tbxic_stdev=1e-3, max_prec_mz=1500.0, max_peaks_n=128)}
    name = type(df).__name__ if df is not None else "DataFormatA"
    if name not in known:
        raise RuntimeError(f"unknown DreaMS data format {name}")
    dformat = dict(known[name], name=name)
    return args, dformat, ck["state_dict"], hp


def load_backbone(path, prefix=None):
    """DreaMSBackbone with the checkpoint's weights (strict on every backbone parameter)."""
    args, dformat, sd, _ = read_checkpoint(path)
    max_mz = dformat.get("max_mz", 1000.0)
    tb = dformat.get("max_tbxic_stdev", 1e-4)
    net = DreaMSBackbone(args, max_mz, tb)
    if prefix is None:  # ssl_model: no prefix; fine-tuning heads: "backbone."
        prefix = "backbone." if any(k.startswith("backbone.") for k in sd) else ""
    own = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    want = net.state_dict()
    missing = [k for k in want if k not in own]
    shape = [k for k in want if k in own and tuple(own[k].shape) != tuple(want[k].shape)]
    if missing or shape:
        raise RuntimeError(f"DreaMS checkpoint mismatch: missing {missing[:8]}, shape {shape[:8]}")
    net.load_state_dict({k: own[k].float() for k in want})
    return net, args, dformat


def prep_spectrum(mz, it, prec_mz, n_peaks=100):
    """Official preprocessing: n highest peaks, intensities relative to the base peak, precursor token first,
    zero-padded to 1 + n_peaks rows. Returns float32 (1 + n_peaks, 2)."""
    mz = np.asarray(mz, np.float64)
    it = np.asarray(it, np.float64)
    ok = (it > 0) & (mz > 0)
    mz, it = mz[ok], it[ok]
    out = np.zeros((1 + n_peaks, 2), np.float32)
    out[0] = (prec_mz, PREC_INTENS)
    if len(mz):
        sel = np.argsort(it)[-n_peaks:]
        sel = sel[np.argsort(mz[sel])]
        out[1:1 + len(sel), 0] = mz[sel]
        out[1:1 + len(sel), 1] = it[sel] / it.max()
    return out


# --- our fingerprint model on top of the backbone -------------------------------------------------------------

class DreamsFP(nn.Module):
    """Spectrum -> fingerprint-bit logits: DreaMS backbone, then [precursor token, mean of peak tokens, condition
    embedding (adduct, instrument family, collision energy, polarity, precursor m/z)] -> MLP -> nbits.
    Vocabulary sizes are passed in (casmi.fpmodel.ADDUCT_LIST / INSTR_LIST) to keep this module dependency-free."""

    def __init__(self, backbone: DreaMSBackbone, nbits: int, n_adducts: int, n_instr: int, d_cond=64, hidden=2048,
                 drop=0.1):
        super().__init__()
        self.backbone = backbone
        d = backbone.d_model
        self.ad = nn.Embedding(n_adducts, d_cond)
        self.ins = nn.Embedding(n_instr, d_cond)
        self.cond = nn.Linear(3, d_cond)
        self.head = nn.Sequential(nn.LayerNorm(2 * d + d_cond), nn.Linear(2 * d + d_cond, hidden), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(hidden, nbits))
        self.nbits, self.d_cond, self.hidden = nbits, d_cond, hidden

    def forward(self, spec, ad, ins, ce, mode):
        h = self.backbone(spec)
        real = (spec[:, 1:, 0] != 0).to(h.dtype)
        mean = (h[:, 1:] * real.unsqueeze(-1)).sum(1) / real.sum(1, keepdim=True).clamp(min=1)
        c = self.ad(ad) + self.ins(ins) + self.cond(torch.stack([ce / 100.0, mode, spec[:, 0, 0] / 1000.0], -1)
                                                    .to(h.dtype))
        return self.head(torch.cat([h[:, 0], mean, c.to(h.dtype)], -1))


_PRIMS = (int, float, str, bool, type(None))


def save_fp(net: DreamsFP, path, n_peaks: int, extra=None):
    """Tensors + primitives only (loads with weights_only=True). fp16 weights except the Fourier frequencies,
    which must stay float32 (up to 5,000 cycles/Da)."""
    sd = {k: (v.detach().float().cpu() if k.startswith("backbone.fourier_enc.") else v.detach().half().cpu())
          for k, v in net.state_dict().items()}
    bb = net.backbone
    meta = {"kind": "dreams_fp", "nbits": net.nbits, "n_adducts": net.ad.num_embeddings,
            "n_instr": net.ins.num_embeddings, "d_cond": net.d_cond, "hidden": net.hidden, "n_peaks": int(n_peaks),
            "max_mz": bb.max_mz, "max_tbxic_stdev": bb.max_tbxic_stdev,
            "args": {k: v for k, v in vars(bb.a).items() if isinstance(v, _PRIMS)}}
    meta.update(extra or {})
    torch.save({"model": sd, **meta}, path)


def load_fp(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    args = Namespace(**ck["args"])
    bb = DreaMSBackbone(args, ck["max_mz"], ck["max_tbxic_stdev"])
    net = DreamsFP(bb, int(ck["nbits"]), int(ck["n_adducts"]), int(ck["n_instr"]), int(ck["d_cond"]),
                   int(ck["hidden"]))
    net.load_state_dict({k: v.float() for k, v in ck["model"].items()})  # strict
    net.n_peaks = int(ck["n_peaks"])
    net.kind = "dreams_fp"
    return net.to(device).eval()


@torch.no_grad()
def fp_logits(net: DreamsFP, raw, precs, ad_idx, ins_idx, ces, modes, batch=256):
    """(n_spectra, nbits) logits. raw: list of (mz, intensity) arrays (as measured), the rest per spectrum."""
    dev = next(net.parameters()).device
    X = np.stack([prep_spectrum(mz, it, pm, net.n_peaks) for (mz, it), pm in zip(raw, precs)])
    out = []
    for a in range(0, len(X), batch):
        sl = slice(a, a + batch)
        T = lambda v, dt=torch.float32: torch.as_tensor(np.asarray(v[sl]), dtype=dt, device=dev)  # noqa: E731
        out.append(net(T(X), T(ad_idx, torch.long), T(ins_idx, torch.long), T(ces), T(modes)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, net.nbits), np.float32)
