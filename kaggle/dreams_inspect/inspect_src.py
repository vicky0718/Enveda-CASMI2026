"""CPU check of casmi.dreams_backbone on the real DreaMS checkpoints (from casmi26-dreams-probe's output):
hyper-parameters, strict weight loading, and a functional test — the pretrained masked-peak head (ff_out) must
recover masked peak m/z through our backbone far above chance. Also prints the official masking / preprocessing
code (dreams/utils/data.py) so the fine-tuning input matches pretraining."""

import glob
import os
import re
import tarfile
import time

import numpy as np
import pyarrow.parquet as pq
import torch

import dreams_backbone as DB

T0 = time.time()
WD = os.path.dirname(glob.glob("/kaggle/input/**/dreams_src.tar.gz", recursive=True)[0])
with tarfile.open(f"{WD}/dreams_src.tar.gz") as t:
    t.extractall("/tmp/src")
data_py = open("/tmp/src/DreaMS/dreams/utils/data.py").read()
for name, n in (("    def __call__(self, spec: np.array, prec_mz=None", 3500), ("    def __getitem__", 6000)):
    for m in re.finditer(re.escape(name), data_py):
        print(f"----- data.py at '{name.strip()}' -----\n{data_py[m.start():m.start() + n]}\n", flush=True)

for ck in sorted(glob.glob(f"{WD}/**/ssl_model.ckpt", recursive=True)):
    print("\n#####", ck, os.path.getsize(ck), flush=True)
    args, dformat, sd, hp = DB.read_checkpoint(ck)
    print("hparams keys", list(hp.keys()))
    print("args", {k: v for k, v in sorted(vars(args).items()) if not k.startswith("_")})
    print("dformat", dformat)
    sp = hp.get("spec_preproc")
    if sp is not None:
        print("spec_preproc", {k: (v if isinstance(v, (int, float, str, bool, type(None))) else type(v).__name__)
                               for k, v in vars(sp).items()})
    print("state_dict", len(sd), "tensors;", sum(v.numel() for v in sd.values()), "values")
    print("non-encoder keys", [k for k in sd if "transformer_encoder" not in k][:40])
    net, a, df = DB.load_backbone(ck)
    print("backbone loaded strictly:", sum(p.numel() for p in net.parameters()), "params; d_model", net.d_model,
          f"{time.time() - T0:.0f}s", flush=True)

    # functional test with the SSL masked-m/z head (ssl_model only)
    head_keys = sorted(k for k in sd if k.startswith("ff_out."))
    if not head_keys:
        continue
    print("ff_out", [(k, tuple(sd[k].shape)) for k in head_keys])
    lin = [k for k in head_keys if k.endswith("weight")]
    layers = []
    for i, k in enumerate(lin):
        w = sd[k].float()
        b = sd.get(k.replace("weight", "bias"))
        l = torch.nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
        l.weight.data = w
        if b is not None:
            l.bias.data = b.float()
        layers.append(l)
        if i < len(lin) - 1:
            layers.append(torch.nn.ReLU())
    head = torch.nn.Sequential(*layers).eval()
    ncls = layers[-1].out_features
    bin_size = float(getattr(a, "hot_mz_bin_size", 1000.0 / ncls))
    mask_val = float(getattr(a, "mask_val", -1.0))
    print("classes", ncls, "bin", bin_size, "mask_val", mask_val, "objective", getattr(a, "train_objective", None))
    train = glob.glob("/kaggle/input/**/train.parquet", recursive=True)[0]
    tb = pq.ParquetFile(train).read_row_group(3, columns=["ms2_mzs", "ms2_normalized_intensities", "precursor_mz",
                                                         "ionization_mode", "instrument_type"]).to_pylist()
    rng = np.random.default_rng(0)
    rows = [r for r in tb if r["ionization_mode"] == "positive" and r["precursor_mz"] < 1000 and len(r["ms2_mzs"]) >= 8]
    rows = [rows[i] for i in rng.choice(len(rows), min(256, len(rows)), replace=False)]
    for npk in (60, 100):
        X = np.stack([DB.prep_spectrum(r["ms2_mzs"], r["ms2_normalized_intensities"], r["precursor_mz"], npk)
                      for r in rows])
        n_real = (X[:, 1:, 0] > 0).sum(1)
        res = {}
        for variant in ("mz_only", "mz_and_intensity"):
            Xm = X.copy()
            js = 1 + np.array([rng.integers(k) for k in n_real])  # one masked real peak per spectrum
            true = X[np.arange(len(X)), js, 0].copy()
            Xm[np.arange(len(X)), js, 0] = mask_val
            if variant == "mz_and_intensity":
                Xm[np.arange(len(X)), js, 1] = mask_val
            with torch.no_grad():
                h = net(torch.as_tensor(Xm))
                logits = head(h[torch.arange(len(X)), torch.as_tensor(js)])
            pred = (logits.argmax(1).numpy() + 0.5) * bin_size
            res[variant] = (float(np.mean(np.abs(pred - true) <= bin_size)), float(np.median(np.abs(pred - true))))
        chance = float(np.mean([np.mean(np.abs(X[b, 1:1 + n_real[b], 0] - X[b, 1 + rng.integers(n_real[b]), 0])
                                        <= bin_size) for b in range(len(X))]))
        print(f"masked-m/z head through our backbone, {npk} peaks: within one bin {res} "
              f"(within-spectrum chance ~{chance:.3f})", flush=True)
print("done", f"{time.time() - T0:.0f}s", flush=True)
