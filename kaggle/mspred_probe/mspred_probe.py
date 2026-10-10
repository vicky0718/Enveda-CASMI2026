"""Probe v2 of the official ms-pred release (Coley group, MIT; GLACIER / ICEBERG, MassSpecGym-trained weights —
host-approved): downloads the official MassSpecGym checkpoints (Dropbox folders linked from the README), lists them,
prints the GLACIER / ICEBERG model sources and their third-party imports, and checks which dependencies install on
the Kaggle image. Output (kept for later kernels to mount): mspred_src.tar.gz, mspred_weights/<model>/...
Every step is guarded so the kernel completes (an errored kernel cannot be mounted)."""

import glob
import os
import re
import subprocess
import sys
import time
import urllib.request
import zipfile

W = "/kaggle/working"
T0 = time.time()
WEIGHTS = {  # README "Pretrained ... MassSpecGym" links (Dropbox shared folders; dl=1 returns a zip)
    "glacier_msg": "https://www.dropbox.com/scl/fo/ta99j0mp1w7qzek5zvy3w/AFCLQNO8W2EP7ZDMOj9nxWI?rlkey=563zxtyvvhwfu1uyz6n10n2ui&dl=1",
    "iceberg_msg_all": "https://www.dropbox.com/scl/fo/kwm35ih8tlfnshfrcq8ot/AOeS4M0_v9MhqeEZys9sCRQ?rlkey=f1n6pbzx94g1k2el2wcbmee61&dl=1",
    "iceberg_msg_simulation": "https://www.dropbox.com/scl/fo/mcj0ngdvuj2xkhr983frb/ABGLrS8ZCeyYUWhzRfxPtRg?rlkey=rccvo52cehh75ma1yy8jji46t&dl=1",
}


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def sh(cmd, tail=8000):
    log("$", cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[-tail:], r.stderr[-3000:], flush=True)
    return r.stdout


try:
    sh("git clone --depth 1 https://github.com/coleygroup/ms-pred.git /tmp/ms-pred && cd /tmp/ms-pred && git log -1 --format='%H %cd'")
    sh(f"cd /tmp && tar --exclude=.git -czf {W}/mspred_src.tar.gz ms-pred && ls -la {W}/mspred_src.tar.gz")
except Exception as e:  # noqa: BLE001
    log("clone failed", e)

# 1. official checkpoints
for name, url in WEIGHTS.items():
    try:
        dst = f"/tmp/{name}.zip"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=1200) as r, open(dst, "wb") as f:
            while True:
                b = r.read(1 << 24)
                if not b:
                    break
                f.write(b)
        log(name, "zip", os.path.getsize(dst))
        out = f"{W}/mspred_weights/{name}"
        os.makedirs(out, exist_ok=True)
        with zipfile.ZipFile(dst) as z:
            for i in z.infolist():
                print("   ", i.filename, i.file_size, flush=True)
            z.extractall(out)
        os.remove(dst)
    except Exception as e:  # noqa: BLE001
        log("weights failed", name, type(e).__name__, str(e)[:300])
sh(f"find {W}/mspred_weights -type f | head -60; du -sh {W}/mspred_weights/* 2>/dev/null")

# 2. model sources and their imports
try:
    for d in ("glacier", "iceberg"):
        files = sorted(glob.glob(f"/tmp/ms-pred/src/ms_pred/{d}/*.py"))
        print(f"\n##### src/ms_pred/{d}: {[os.path.basename(f) for f in files]}", flush=True)
        imports = set()
        for f in files:
            for m in re.findall(r"^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", open(f).read(), re.M):
                imports.add(m.split(".")[0])
        print("third-party imports:", sorted(imports), flush=True)
    for rel in ("src/ms_pred/glacier/glacier_model.py", "src/ms_pred/glacier/predict.py",
                "run_scripts/glacier/02_predict_inten.py", "src/ms_pred/glacier/glacier_data.py"):
        p = f"/tmp/ms-pred/{rel}"
        if os.path.exists(p):
            txt = open(p).read()
            print(f"\n===== {rel} ({len(txt)} chars) =====\n{txt[:20000]}\n===== end {rel} =====", flush=True)
    sh("cd /tmp/ms-pred && ls src/ms_pred src/ms_pred/glacier configs/glacier run_scripts/glacier && "
       "cat configs/glacier/*.yaml | head -150")
except Exception as e:  # noqa: BLE001
    log("source dump failed", e)

# 3. environment: can the dependencies be installed here?
sh(f"{sys.executable} -c 'import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda)'")
for pkg in ("dgl", "torch-scatter", "torch-geometric", "pygmtools", "linsatnet", "msbuddy", "omegaconf", "einops"):
    sh(f"pip install -q {pkg} 2>&1 | tail -2; {sys.executable} -c 'import importlib; "
       f"m = importlib.import_module(\"{pkg.replace('-', '_')}\"); print(\"{pkg} OK\", getattr(m, \"__version__\", \"\"))'")
log("done")
