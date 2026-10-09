"""Probe the official DreaMS release (Bushuiev et al., Nat. Biotechnol. 2025; code MIT, github.com/pluskal-lab/DreaMS;
weights Zenodo 10997887) from a Kaggle kernel with internet: the sandbox we develop in cannot reach GitHub / Zenodo /
Hugging Face. Outputs (/kaggle/working):
    dreams_src.tar.gz            python sources + configs of the repo (to read the architecture and preprocessing)
    dreams_weights/*.ckpt        pretrained checkpoints (GeMS self-supervised; contrastive embedding model)
    probe.json                   checkpoint hyper-parameters, parameter shapes, embedding smoke test
The training kernel mounts this kernel's output, so it needs no internet itself.
"""

import glob
import inspect
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request

W = "/kaggle/working"
T0 = time.time()
REPORT = {}


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def sh(cmd, tail=4000):
    log("$", cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[-tail:], r.stderr[-tail:], flush=True)
    return r.returncode, r.stdout


# 1. code
sh("git clone --depth 1 https://github.com/pluskal-lab/DreaMS.git /tmp/DreaMS")
_, head = sh("cd /tmp/DreaMS && git log -1 --format='%H %cd'")
REPORT["commit"] = head.strip()
sh("cd /tmp/DreaMS && ls -la && head -5 LICENSE; cat setup.py pyproject.toml 2>/dev/null | head -120; "
   "cat requirements*.txt 2>/dev/null | head -60; find . -name '*.py' -not -path './.git/*' | head -80")
with tarfile.open(f"{W}/dreams_src.tar.gz", "w:gz") as t:
    for p in glob.glob("/tmp/DreaMS/**/*", recursive=True):
        if ("/.git/" not in p and os.path.isfile(p) and os.path.getsize(p) < 2_000_000
                and p.endswith((".py", ".md", ".sh", ".yaml", ".yml", ".toml", ".cfg", ".txt", ".json", "LICENSE"))):
            t.add(p, arcname=os.path.relpath(p, "/tmp"))
log("sources archived", os.path.getsize(f"{W}/dreams_src.tar.gz"))
for rel in ("dreams/models/dreams/dreams.py", "dreams/models/dreams/layers.py", "dreams/models/layers/fourier_features.py",
            "dreams/models/layers/feed_forward.py", "dreams/api.py", "dreams/utils/spectra.py", "dreams/utils/dformats.py",
            "dreams/models/heads/heads.py", "dreams/definitions.py"):
    p = os.path.join("/tmp/DreaMS", rel)
    if os.path.exists(p):
        txt = open(p).read()
        print(f"\n===== {rel} ({len(txt)} chars) =====\n{txt}\n===== end {rel} =====", flush=True)

# 2. weights (Hugging Face first, Zenodo as fallback; skip what exists)
os.makedirs(f"{W}/dreams_weights", exist_ok=True)
for name in ("ssl_model.ckpt", "embedding_model.ckpt"):
    dst = f"{W}/dreams_weights/{name}"
    if os.path.exists(dst):
        continue
    for url in (f"https://huggingface.co/roman-bushuiev/DreaMS/resolve/main/{name}",
                f"https://huggingface.co/roman-bushuiev/DreaMS/resolve/main/weights/{name}",
                f"https://zenodo.org/records/10997887/files/{name}?download=1"):
        try:
            with urllib.request.urlopen(url, timeout=900) as r, open(dst + ".part", "wb") as f:
                while True:
                    b = r.read(1 << 24)
                    if not b:
                        break
                    f.write(b)
            os.rename(dst + ".part", dst)
            log(name, "from", url, os.path.getsize(dst))
            REPORT[f"url_{name}"] = url
            break
        except Exception as e:  # noqa: BLE001
            log("failed", url, type(e).__name__, e)
sh(f"ls -la {W}/dreams_weights")

# 3. import the package (no dependency changes to torch); install missing light dependencies on demand
sh("pip install --no-deps -e /tmp/DreaMS")
sys.path.insert(0, "/tmp/DreaMS")
for _ in range(15):
    try:
        import dreams  # noqa: F401
        from dreams import api  # noqa: F401
        break
    except ModuleNotFoundError as e:
        mod = e.name.split(".")[0]
        pkg = {"sklearn": "scikit-learn", "yaml": "pyyaml", "rdkit": "rdkit", "pytorch_lightning": "pytorch-lightning",
               "lightning": "lightning", "pyteomics": "pyteomics", "matchms": "matchms", "h5py": "h5py"}.get(mod, mod)
        log("missing module", mod, "-> pip", pkg)
        sh(f"pip install -q {pkg}")
    except Exception as e:  # noqa: BLE001
        log("import error", type(e).__name__, e)
        break

# 4. checkpoints (after the package import: Lightning checkpoints may pickle dreams objects): hyper-parameters and parameter shapes
import torch  # noqa: E402

import zipfile  # noqa: E402

for p in sorted(glob.glob(f"{W}/dreams_weights/*.ckpt")):
    sz = os.path.getsize(p)
    with open(p, "rb") as fh:
        head = fh.read(8)
        fh.seek(max(0, sz - 22))
        tail = fh.read(22)
    log(p, "size", sz, "head", head, "tail", tail, "zip", zipfile.is_zipfile(p))
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        log("load failed", p, type(e).__name__, str(e)[:300])
        REPORT[os.path.basename(p)] = {"load_error": str(e)[:300], "size": sz, "head": repr(head)}
        continue
    info = {"top_keys": sorted(map(str, ck.keys())) if isinstance(ck, dict) else str(type(ck))}
    if isinstance(ck, dict):
        hp = ck.get("hyper_parameters") or ck.get("hparams") or {}
        info["hparams"] = {str(k): repr(v)[:300] for k, v in dict(hp).items()}
        sd = ck.get("state_dict") or ck.get("model") or {}
        info["n_params"] = int(sum(v.numel() for v in sd.values() if hasattr(v, "numel")))
        info["shapes"] = {k: list(v.shape) for k, v in sd.items() if hasattr(v, "shape")}
    REPORT[os.path.basename(p)] = info
    log(os.path.basename(p), "params", info.get("n_params"), "hparams", json.dumps(info.get("hparams"))[:3000])
    for k, v in list(info.get("shapes", {}).items())[:80]:
        print("   ", k, v)

try:
    from dreams import api

    REPORT["api_members"] = [n for n in dir(api) if not n.startswith("_")]
    for n in ("dreams_embeddings", "dreams_predictions", "PreTrainedModel", "compute_dreams_predictions"):
        if hasattr(api, n):
            obj = getattr(api, n)
            try:
                print(f"----- api.{n} -----\n" + inspect.getsource(obj)[:6000], flush=True)
            except Exception:  # noqa: BLE001
                pass
except Exception as e:  # noqa: BLE001
    log("api unavailable", type(e).__name__, e)

# 5. smoke test: embeddings of a few training spectra (positive and negative mode) through the official API
try:
    import pyarrow.parquet as pq

    train = glob.glob("/kaggle/input/**/train.parquet", recursive=True)[0]
    t = pq.ParquetFile(train).read_row_group(0, columns=["ms2_mzs", "ms2_normalized_intensities", "precursor_mz",
                                                         "adduct", "ionization_mode", "inchikey14"]).slice(0, 64)
    rows = t.to_pylist()
    with open("/tmp/probe.mgf", "w") as fh:
        for i, r in enumerate(rows):
            fh.write(f"BEGIN IONS\nTITLE=s{i}\nPEPMASS={r['precursor_mz']}\nCHARGE=1+\n")
            for mz, it in zip(r["ms2_mzs"], r["ms2_normalized_intensities"]):
                fh.write(f"{mz} {it}\n")
            fh.write("END IONS\n")
    from dreams.api import dreams_embeddings

    for ck in sorted(glob.glob(f"{W}/dreams_weights/*.ckpt")):
        t1 = time.time()
        try:
            e = dreams_embeddings("/tmp/probe.mgf", model_ckpt=ck)
        except TypeError:
            e = dreams_embeddings("/tmp/probe.mgf")
        log("embeddings", os.path.basename(ck), getattr(e, "shape", None), f"{time.time() - t1:.1f}s for 64 spectra")
        REPORT[f"emb_{os.path.basename(ck)}"] = {"shape": list(getattr(e, "shape", [])),
                                                 "sec_per_64": time.time() - t1}
except Exception as e:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    REPORT["smoke_error"] = f"{type(e).__name__}: {e}"

with open(f"{W}/probe.json", "w") as fh:
    json.dump(REPORT, fh, indent=1, default=str)
log("done")
