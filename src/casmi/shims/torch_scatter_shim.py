"""torch_scatter stand-in on torch.Tensor.scatter_reduce (see README.md)."""

import sys
import types

import torch


def _prep(src, index, dim, out, dim_size, fill):
    dim = dim if dim >= 0 else src.dim() + dim
    if index.dim() != src.dim():  # broadcast a 1-D index along `dim`
        shape = [1] * src.dim()
        shape[dim] = -1
        index = index.view(shape).expand_as(src)
    if out is None:
        size = list(src.shape)
        size[dim] = int(dim_size) if dim_size is not None else int(index.max()) + 1 if index.numel() else 0
        out = torch.full(size, fill, dtype=src.dtype, device=src.device)
    return index, dim, out


def scatter_add(src, index, dim=-1, out=None, dim_size=None):
    index, dim, out = _prep(src, index, dim, out, dim_size, 0)
    return out.scatter_add_(dim, index, src)


scatter_sum = scatter_add


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    index, dim, out = _prep(src, index, dim, out, dim_size, 0)
    s = out.scatter_add(dim, index, src)
    c = torch.zeros_like(s).scatter_add_(dim, index, torch.ones_like(src))
    return s / c.clamp(min=1)


def _extreme(src, index, dim, out, dim_size, reduce):
    index, dim, out = _prep(src, index, dim, out, dim_size, 0)
    res = out.scatter_reduce(dim, index, src, reduce=reduce, include_self=False)
    return res, None  # arg positions are not provided (GLACIER inference discards them)


def scatter_max(src, index, dim=-1, out=None, dim_size=None):
    return _extreme(src, index, dim, out, dim_size, "amax")


def scatter_min(src, index, dim=-1, out=None, dim_size=None):
    return _extreme(src, index, dim, out, dim_size, "amin")


def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
    if reduce in ("sum", "add"):
        return scatter_add(src, index, dim, out, dim_size)
    if reduce == "mean":
        return scatter_mean(src, index, dim, out, dim_size)
    return (scatter_max if reduce == "max" else scatter_min)(src, index, dim, out, dim_size)[0]


def install():
    m = types.ModuleType("torch_scatter")
    for k in ("scatter_add", "scatter_sum", "scatter_mean", "scatter_max", "scatter_min", "scatter"):
        setattr(m, k, globals()[k])
    sys.modules["torch_scatter"] = m
    return m
