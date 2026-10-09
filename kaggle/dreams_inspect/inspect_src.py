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
for name, n in (("    def get_spec(self", 5000),):
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
    rows = [rows[i] for i in rng.choice(len(rows), min(512, len(rows)), replace=False)]
    import copy

    ctrl = copy.deepcopy(net)
    torch.manual_seed(0)
    for n_, p_ in ctrl.named_parameters():  # control: same architecture, re-initialised encoder weights
        if "fourier_enc" not in n_:
            p_.data = torch.randn_like(p_) * p_.std().clamp(min=1e-3) if p_.dim() > 1 else p_.data
    for inst in ("all", "orbitrap"):
        sub = [r for r in rows if inst == "all" or "orbitrap" in str(r["instrument_type"]).lower()]
        if len(sub) < 32:
            continue
        X = np.stack([DB.prep_spectrum(r["ms2_mzs"], r["ms2_normalized_intensities"], r["precursor_mz"], 60)
                      for r in sub])
        # pretraining-style masks: 30 % of the peaks with relative intensity >= 0.1, drawn in proportion to intensity
        Xm, M_ = X.copy(), np.zeros(X.shape[:2], bool)
        for b in range(len(X)):
            cand = np.flatnonzero((X[b, :, 1] >= 0.1) & (np.arange(X.shape[1]) > 0) & (X[b, :, 0] > 0))
            if len(cand) == 0:
                continue
            k = max(1, min(len(cand), int(round(0.3 * (X[b, 1:, 0] > 0).sum()))))
            pr = X[b, cand, 1] / X[b, cand, 1].sum()
            sel = rng.choice(cand, k, replace=False, p=pr)
            M_[b, sel] = True
        Xm[M_, 0] = mask_val
        for name, model in (("pretrained", net), ("control", ctrl)):
            with torch.no_grad():
                h = model(torch.as_tensor(Xm))
                logits = head(h[torch.as_tensor(M_)])
            pred = (logits.argmax(1).numpy() + 0.5) * bin_size
            true = X[M_, 0]
            err = np.abs(pred - true)
            print(f"[{inst}, {len(sub)} spectra, {M_.sum()} masked peaks] {name}: within 1 bin {np.mean(err <= bin_size):.3f}"
                  f", within 0.5 Da {np.mean(err <= 0.5):.3f}, median error {np.median(err):.2f} Da", flush=True)
print("done", f"{time.time() - T0:.0f}s", flush=True)
