"""Import-only stand-in for DGL (see README.md). Any attribute resolves to a placeholder. Calls made while modules are
imported (e.g. `fn.copy_u(u="h", out="m")` message-function constants) return placeholders; the calls that build or
batch graphs raise, so an unexpected runtime use of DGL fails loudly instead of silently."""

import sys
import types


_GRAPH_OPS = {"graph", "batch", "unbatch", "heterograph", "update_all", "apply_edges", "add_self_loop",
              "to_bidirected", "khop_graph", "random_walk_pe", "laplacian_pe"}


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


class DGLGraph:  # isinstance checks only
    pass


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
    root.__path__ = []
    sys.modules["dgl"] = root
    for sub in ("nn", "nn.pytorch", "function", "ops", "data", "backend", "utils", "readout"):
        m = _Module(f"dgl.{sub}")
        m.__path__ = []
        sys.modules[f"dgl.{sub}"] = m
        parent, _, leaf = f"dgl.{sub}".rpartition(".")
        setattr(sys.modules[parent], leaf, m)
    return root
