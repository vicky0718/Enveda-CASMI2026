"""Print the DreaMS sources we need (architecture, preprocessing, API) and inspect the checkpoints saved by
casmi26-dreams-probe. Kernel output files cannot be downloaded into our development sandbox, kernel logs can."""

import glob
import os
import subprocess
import sys
import tarfile
import time
import zipfile

T0 = time.time()
SRC = glob.glob("/kaggle/input/**/dreams_src.tar.gz", recursive=True)[0]
WD = os.path.dirname(SRC)
print("probe output", WD, sorted(os.listdir(WD)), flush=True)
with tarfile.open(SRC) as t:
    t.extractall("/tmp/src")
ROOT = "/tmp/src/DreaMS"
for rel in ("dreams/models/dreams/dreams.py", "dreams/models/dreams/layers.py", "dreams/models/layers/fourier_features.py",
            "dreams/models/layers/feed_forward.py", "dreams/api.py", "dreams/utils/spectra.py", "dreams/utils/dformats.py",
            "dreams/models/heads/heads.py", "dreams/definitions.py", "README.md"):
    p = os.path.join(ROOT, rel)
    if os.path.exists(p):
        txt = open(p).read()
        print(f"\n===== {rel} ({len(txt)} chars) =====\n{txt}\n===== end {rel} =====", flush=True)

# checkpoints: size, magic, zip integrity, then a full load (Lightning checkpoints pickle their hparams)
for p in sorted(glob.glob(f"{WD}/**/*.ckpt", recursive=True)):
    sz = os.path.getsize(p)
    with open(p, "rb") as fh:
        head = fh.read(8)
        fh.seek(max(0, sz - 22))
        tail = fh.read(22)
    print(f"\n{p} size {sz} head {head!r} tail {tail!r} zip {zipfile.is_zipfile(p)}", flush=True)
subprocess.run("pip install -q pytorch-lightning==2.0.8 torchmetrics 2>&1 | tail -2", shell=True)
import torch  # noqa: E402

sys.path.insert(0, ROOT)
for p in sorted(glob.glob(f"{WD}/**/*.ckpt", recursive=True)):
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        print("load failed", p, type(e).__name__, str(e)[:300], flush=True)
        continue
    print("\n#####", os.path.basename(p), "keys", list(ck.keys()) if isinstance(ck, dict) else type(ck), flush=True)
    hp = ck.get("hyper_parameters", {}) if isinstance(ck, dict) else {}
    for k, v in dict(hp).items():
        print("  hp", k, "=", repr(v)[:200])
    sd = ck.get("state_dict", {}) if isinstance(ck, dict) else {}
    print("  n_params", sum(v.numel() for v in sd.values() if hasattr(v, "numel")))
    for k, v in sd.items():
        print("  sd", k, tuple(v.shape), v.dtype)
print("done", time.time() - T0, flush=True)
