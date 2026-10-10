"""Print what we need to re-implement GLACIER inference without DGL: the glacier package sources, every ms_pred module
they import (recursively), and the official MassSpecGym checkpoint's hyper-parameters and parameter shapes."""

import glob
import os
import pickle
import re
import tarfile
import types

import torch

SRC = glob.glob("/kaggle/input/**/mspred_src.tar.gz", recursive=True)[0]
with tarfile.open(SRC) as t:
    t.extractall("/tmp/src")
ROOT = "/tmp/src/ms-pred/src"


def mod_path(name):
    p = os.path.join(ROOT, *name.split("."))
    for c in (p + ".py", os.path.join(p, "__init__.py")):
        if os.path.exists(c):
            return c
    return None


seen, queue = set(), [f[len(ROOT) + 1:-3].replace("/", ".") for f in sorted(glob.glob(f"{ROOT}/ms_pred/glacier/*.py"))]
while queue:
    m = queue.pop(0)
    if m in seen:
        continue
    seen.add(m)
    p = mod_path(m)
    if p is None:
        continue
    txt = open(p).read()
    for a, b in re.findall(r"^\s*(?:from\s+(ms_pred[\w.]*)\s+import|import\s+(ms_pred[\w.]*))", txt, re.M):
        dep = a or b
        queue.append(dep)
        for sub in re.findall(rf"from\s+{re.escape(dep)}\s+import\s+\(?([\w ,\n]+)\)?", txt):
            for s in [x.strip() for x in sub.split(",") if x.strip()]:
                if mod_path(f"{dep}.{s}"):
                    queue.append(f"{dep}.{s}")
print("modules:", sorted(seen), flush=True)
for m in sorted(seen):
    p = mod_path(m)
    if p is None or m.endswith(("train_joint", "train_contr_joint")):
        continue
    txt = open(p).read()
    print(f"\n===== {m} ({len(txt)} chars) =====\n{txt}\n===== end {m} =====", flush=True)


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        self.__dict__.update(s if isinstance(s, dict) else {"_s": s})


class _U(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:  # noqa: BLE001
            return type(name, (_Stub,), {})


PM = types.SimpleNamespace(Unpickler=_U, load=pickle.load, __name__="pickle")
for ck in sorted(glob.glob("/kaggle/input/**/mspred_weights/**/best.ckpt", recursive=True)):
    print("\n#####", ck, os.path.getsize(ck), flush=True)
    try:
        c = torch.load(ck, map_location="cpu", weights_only=False, pickle_module=PM)
        print("keys", list(c.keys()))
        for k, v in (c.get("hyper_parameters") or {}).items():
            print("  hp", k, "=", repr(v)[:300])
        sd = c.get("state_dict", {})
        print("  n_params", sum(v.numel() for v in sd.values()))
        for k, v in sd.items():
            print("  sd", k, tuple(v.shape))
    except Exception as e:  # noqa: BLE001
        print("load failed", type(e).__name__, str(e)[:300])
for y in sorted(glob.glob("/kaggle/input/**/mspred_weights/**/hparams.yaml", recursive=True)):
    print(f"\n===== {y} =====\n{open(y).read()[:3000]}", flush=True)
print("done", flush=True)
