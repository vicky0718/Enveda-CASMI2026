"""Import-only stand-in for DGL (see README.md). Any attribute resolves to a placeholder. Calls made while modules are
imported (e.g. `fn.copy_u(u="h", out="m")` message-function constants) return placeholders; the calls that build or
batch graphs raise, so an unexpected runtime use of DGL fails loudly instead of silently."""

import sys
import types


_GRAPH_OPS = {"batch", "unbatch", "heterograph", "update_all", "apply_edges", "add_self_loop",
              "to_bidirected", "khop_graph", "laplacian_pe"}


class _Missing:
    def __init__(self, name):
        self._name = name

    def __call__(self, *a, **k):
        if self._name.split(".")[-1] in _GRAPH_OPS:
            raise NotImplementedError(f"DGL API used at runtime: {self._name} (not provided by the stand-in)")
        return _Missing(f"{self._name}()")

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        return _Missing(f"{self._name}.{item}")

    def __mro_entries__(self, bases):  # allows `class X(dgl.nn.SomeModule)` at import time
        return (object,)


class DGLGraph:
    """Minimal homogeneous graph container (edge list + node / edge feature dicts) — enough for the per-molecule
    random-walk positional encoding GLACIER computes while featurising (dataset.get_pe_for_tensor)."""

    def __init__(self, src=(), dst=(), num_nodes=None):
        import torch
        self._src = torch.as_tensor(list(src), dtype=torch.long)
        self._dst = torch.as_tensor(list(dst), dtype=torch.long)
        n = num_nodes if num_nodes is not None else (int(max(self._src.max(), self._dst.max())) + 1
                                                   if len(self._src) else 0)
        self._n = int(n)
        self.ndata, self.edata = {}, {}

    def num_nodes(self):
        return self._n

    number_of_nodes = num_nodes

    def num_edges(self):
        return int(len(self._src))

    number_of_edges = num_edges

    def edges(self, order=None, form="uv"):
        return self._src, self._dst

    def out_degrees(self, v=None):
        import torch
        return torch.bincount(self._src, minlength=self._n)

    def in_degrees(self, v=None):
        import torch
        return torch.bincount(self._dst, minlength=self._n)

    @property
    def device(self):
        return self._src.device

    def to(self, device):
        return self


def graph(data, num_nodes=None, **kwargs):
    src, dst = data
    return DGLGraph(src, dst, num_nodes)


def random_walk_pe(g, k, eweight_name=None):
    """DGL's random_walk_pe: diagonal of the 1..k-step powers of the (edge-weighted) row-normalised transition matrix
    -> (num_nodes, k) float32."""
    import torch
    n = g.num_nodes()
    row, col = g.edges()
    if eweight_name is None:
        w = torch.ones(len(row), dtype=torch.float32)
    else:
        w = g.edata[eweight_name].reshape(-1).to(torch.float32)
    adj = torch.zeros((n, n), dtype=torch.float32)
    if len(row):
        adj.index_put_((row, col), w, accumulate=True)
    rw = adj / (adj.sum(1, keepdim=True) + 1e-30)
    out, pe = rw, [torch.diagonal(rw)]
    for _ in range(k - 1):
        out = out @ rw
        pe.append(torch.diagonal(out))
    return torch.stack(pe, dim=-1)


class _Module(types.ModuleType):
    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        sub = f"{self.__name__}.{item}"
        if sub in sys.modules:
            return sys.modules[sub]
        return _Missing(sub)


def install():
    """Register `dgl` and its common sub-packages in sys.modules."""
    root = _Module("dgl")
    root.DGLGraph = DGLGraph
    root.graph = graph
    root.random_walk_pe = random_walk_pe
    root.__path__ = []
    sys.modules["dgl"] = root
    for sub in ("nn", "nn.pytorch", "function", "ops", "data", "backend", "utils", "readout"):
        m = _Module(f"dgl.{sub}")
        m.__path__ = []
        sys.modules[f"dgl.{sub}"] = m
        parent, _, leaf = f"dgl.{sub}".rpartition(".")
        setattr(sys.modules[parent], leaf, m)
    return root
