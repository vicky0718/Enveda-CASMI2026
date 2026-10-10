"""Offline wheels for running GLACIER (ms-pred) in a submission notebook (no internet there): installs the
pure-Python dependencies the Kaggle image lacks, records exactly which distributions were added (pip freeze diff),
and builds wheels for those only (torch / numpy / rdkit etc. come from the image or the competition's RDKit wheel).
Output: /kaggle/working/wheels/*.whl + requirements_offline.txt"""

import subprocess
import sys

PKGS = ["linsatnet", "pygmtools", "pytorch-lightning", "omegaconf", "einops", "h5py"]


def freeze():
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    return {l.strip() for l in out.splitlines() if "==" in l}


before = freeze()
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PKGS], capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
added = sorted(freeze() - before)
print("added distributions:", added, flush=True)
open("/kaggle/working/requirements_offline.txt", "w").write("\n".join(added) + "\n")
r = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", "/kaggle/working/wheels", *added],
                   capture_output=True, text=True)
print(r.stdout[-4000:], r.stderr[-3000:], flush=True)
subprocess.run("ls -la /kaggle/working/wheels", shell=True)
# check: imports work
for mod in ("LinSATNet", "pygmtools", "pytorch_lightning", "omegaconf", "einops", "h5py"):
    try:
        __import__(mod)
        print("import ok", mod, flush=True)
    except Exception as e:  # noqa: BLE001
        print("import FAILED", mod, e, flush=True)
