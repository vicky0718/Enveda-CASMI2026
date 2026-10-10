"""Probe the official ms-pred release (Coley group; GLACIER / ICEBERG spectrum simulators, MassSpecGym-trained —
host-approved for this competition) from a Kaggle kernel with internet: our sandbox cannot reach GitHub / Zenodo /
figshare. Prints the README, licence, quickstart / download scripts and the model code layout into the log, then
tries the download links those scripts name for MassSpecGym / GLACIER / ICEBERG checkpoints (kept in this kernel's
output for later kernels to mount). Nothing is executed from the repository except reading files."""

import glob
import os
import re
import subprocess
import time
import urllib.request

W = "/kaggle/working"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def sh(cmd, tail=6000):
    log("$", cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[-tail:], r.stderr[-2000:], flush=True)
    return r.stdout


sh("git clone --depth 1 https://github.com/coleygroup/ms-pred.git /tmp/ms-pred")
sh("cd /tmp/ms-pred && git log -1 --format='%H %cd' && ls -la && head -30 LICENSE* && cat setup.py pyproject.toml "
   "environment.yml requirements*.txt 2>/dev/null | head -150")
readme = open("/tmp/ms-pred/README.md").read() if os.path.exists("/tmp/ms-pred/README.md") else ""
print("===== README.md =====\n" + readme[:30000] + "\n===== end README =====", flush=True)
sh("cd /tmp/ms-pred && find . -iname '*glacier*' -o -iname '*iceberg*' -o -iname '*marason*' | grep -v '.git/' | head -80")
sh("cd /tmp/ms-pred && find . -path ./.git -prune -o -type f \\( -name '*.sh' -o -name '*.md' \\) -print | head -150")
urls = set()
for p in glob.glob("/tmp/ms-pred/**/*", recursive=True):
    if "/.git/" in p or not os.path.isfile(p) or os.path.getsize(p) > 500_000:
        continue
    if p.endswith((".sh", ".md", ".py", ".yaml", ".yml", ".txt")):
        txt = open(p, errors="ignore").read()
        if p.endswith(".sh") and re.search(r"wget|curl|download", txt) and ("quickstart" in p or "download" in p):
            print(f"\n===== {os.path.relpath(p, '/tmp/ms-pred')} =====\n{txt[:4000]}\n===== end =====", flush=True)
        for u in re.findall(r"https?://[^\s\"')>]+", txt):
            if any(k in u.lower() for k in ("zenodo", "figshare", "dropbox", "drive.google", "huggingface", "release")):
                urls.add(u.rstrip(".,;"))
print("\n===== candidate download URLs =====", flush=True)
for u in sorted(urls):
    print(u)
# only fetch checkpoints that look like MassSpecGym / GLACIER / ICEBERG models (small listing first)
os.makedirs(f"{W}/mspred_weights", exist_ok=True)
for u in sorted(urls):
    if not any(k in u.lower() for k in ("glacier", "iceberg", "msg", "massspecgym", "pretrained", "model")):
        continue
    dst = f"{W}/mspred_weights/" + re.sub(r"[^A-Za-z0-9._-]", "_", u.split("/")[-1].split("?")[0])[:120]
    try:
        with urllib.request.urlopen(u, timeout=600) as r, open(dst, "wb") as f:
            n = 0
            while True:
                b = r.read(1 << 24)
                if not b:
                    break
                f.write(b)
                n += len(b)
                if n > 6e9:
                    break
        log("downloaded", u, os.path.getsize(dst))
    except Exception as e:  # noqa: BLE001
        log("failed", u, type(e).__name__, str(e)[:200])
sh(f"ls -la {W}/mspred_weights; cd {W}/mspred_weights && for f in *; do file \"$f\"; done 2>/dev/null | head -40")
with open(f"{W}/mspred_src.tar.gz", "wb"):
    pass
sh(f"cd /tmp && tar --exclude=.git -czf {W}/mspred_src.tar.gz ms-pred && ls -la {W}/mspred_src.tar.gz")
log("done")
