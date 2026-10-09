"""Fine-tune the DreaMS backbone (self-supervised on GeMS, the public MassIVE/GNPS spectra; ssl_model.ckpt) to predict
our fingerprint bits from the competition train file — our own model on a pretrained encoder.

Same targets / exclusions / loss as FP run 3 (pure BCE over the informative bits; the held-out validation
structures are excluded by row_struct == -2), DreaMS preprocessing (n highest peaks, intensities relative to the base
peak, precursor token first). The model is casmi.dreams_backbone.DreamsFP (bundled into this script by
kaggle/bundle.py); the weights come from the output of our casmi26-dreams-probe kernel (official release).

Kaggle GPU script. Inputs: competition data; vigneshnehru/casmi26-fp-train (row_struct.npy, fp_targets.npy,
decoys.npy, nbits.txt); vigneshnehru/casmi26-artifacts (code__fpmodel.py: adduct / instrument vocabularies);
vigneshnehru/casmi26-dreams-probe (dreams_weights/ssl_model.ckpt).
Output (/kaggle/working): dreams_fp.pt (casmi.dreams_backbone.load_fp), train_log.csv
"""

import glob
import math
import os
import shutil
import sys
import time
from multiprocessing import Pool

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

import dreams_backbone as DB

T_START = time.time()
BUDGET_S = float(os.environ.get("FP_BUDGET_S", 10.6 * 3600))  # Kaggle GPU sessions stop at 12 h
SMOKE = os.environ.get("FP_SMOKE") == "1"  # local CPU check: random tiny backbone, one row group, few steps
DS = os.environ.get("FP_DS") or os.path.dirname(glob.glob("/kaggle/input/**/fp_targets.npy", recursive=True)[0])
TRAIN = os.environ.get("FP_TRAIN") or glob.glob("/kaggle/input/**/train.parquet", recursive=True)[0]
_CODE = glob.glob("/kaggle/input/**/code__fpmodel.py", recursive=True)
if _CODE:
    os.makedirs("/kaggle/working/fpcode", exist_ok=True)
    shutil.copy(_CODE[0], "/kaggle/working/fpcode/fpmodel.py")
    sys.path.insert(0, "/kaggle/working/fpcode")
else:
    sys.path.insert(0, os.environ.get("FP_CODE", DS))
import fpmodel as M  # noqa: E402

N_PEAKS = int(os.environ.get("FP_NPEAKS", 60))
BS = 16 if SMOKE else int(os.environ.get("FP_BS", 128))
LR_BB, LR_HEAD, WD, WARM = 5e-5, 5e-4, 0.01, (5 if SMOKE else 1500)
ENVEDA_POS_SHIFT = -0.0004
OUT = os.environ.get("FP_OUT", "/kaggle/working")
print("dataset", DS, "train", TRAIN, "gpus", torch.cuda.device_count(), "peaks", N_PEAKS, "batch", BS, flush=True)


def _rg(rg):
    f = pq.ParquetFile(TRAIN)
    rs = np.load(os.path.join(DS, "row_struct.npy"), mmap_mode="r")
    start = sum(f.metadata.row_group(i).num_rows for i in range(rg))
    n = f.metadata.row_group(rg).num_rows
    lab = np.asarray(rs[start:start + n])
    sel = np.flatnonzero(lab != -1)
    tbl = f.read_row_group(rg, columns=["ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "adduct",
                                        "instrument_type", "ionization_mode", "collision_energy_ev",
                                        "ingest_lib"]).take(sel)
    mzs, its = tbl["ms2_mzs"].combine_chunks(), tbl["ms2_normalized_intensities"].combine_chunks()
    off = mzs.offsets.to_numpy()
    fm = pc.list_flatten(mzs).to_numpy(zero_copy_only=False)
    fi = pc.list_flatten(its).to_numpy(zero_copy_only=False)
    prec = tbl["precursor_mz"].to_numpy(zero_copy_only=False).astype(np.float64)
    pos = np.asarray(tbl["ionization_mode"].to_pylist()) == "positive"
    lib = np.asarray(tbl["ingest_lib"].to_pylist())
    shift = np.where(pos & np.isin(lib, ["enveda-180", "enveda-np-examples"]), ENVEDA_POS_SHIFT, 0.0)
    prec = prec + shift
    ce = np.array([np.mean(c) if c else 0.0 for c in tbl["collision_energy_ev"].to_pylist()], np.float32)
    k = len(sel)
    MZ = np.zeros((k, 1 + N_PEAKS), np.float32)
    IT = np.zeros((k, 1 + N_PEAKS), np.float16)
    for j in range(k):
        a, b = off[j], off[j + 1]
        x = DB.prep_spectrum(fm[a:b] + shift[j], fi[a:b], prec[j], N_PEAKS)
        MZ[j], IT[j] = x[:, 0], x[:, 1]
    ad = np.array([M.ADDUCT_IX.get(a, M.ADDUCT_IX["<unk>"]) for a in tbl["adduct"].to_pylist()], np.int16)
    ins = np.array([M.instr_family(s) for s in tbl["instrument_type"].to_pylist()], np.int8)
    return MZ, IT, ad, ins, ce, pos.astype(np.float32), lab[sel]


def load_data():
    nrg = 1 if SMOKE else pq.ParquetFile(TRAIN).num_row_groups
    with Pool(4) as p:
        parts = p.map(_rg, range(nrg), chunksize=1)
    cat = [np.concatenate([q[i] for q in parts]) for i in range(7)]
    keep = (cat[0][:, 1] > 0)  # at least one peak
    print(f"prepared {keep.sum():,} spectra in {time.time() - T_START:.0f}s", flush=True)
    return [c[keep] for c in cat]


def backbone():
    if SMOKE:
        from argparse import Namespace
        a = Namespace(d_fourier=40, d_peak=24, d_mz_token=None, n_layers=2, n_heads=4, att_dropout=0.1, ff_dropout=0.1,
                      residual_dropout=0.1, dropout=0.1, no_transformer_bias=False, attn_mech="dot-product",
                      pre_norm=False, scnorm=False, graphormer_mz_diffs=True, graphormer_parametrized=True,
                      fourier_strategy="lin_float_int", fourier_num_freqs=None, fourier_min_freq=None,
                      fourier_trainable=False, ff_fourier_d=32, ff_fourier_depth=2, ff_peak_depth=2, no_ffs_bias=False,
                      charge_feature=False, vanilla_transformer=False)
        return DB.DreaMSBackbone(a, 1000.0, 1e-2)
    ck = glob.glob("/kaggle/input/**/ssl_model.ckpt", recursive=True)[0]
    net, args, dformat = DB.load_backbone(ck)
    print("DreaMS backbone", ck, "params", sum(p.numel() for p in net.parameters()), "d_model", net.d_model,
          "dformat", dformat, flush=True)
    return net


def main():
    MZ, IT, AD, INS, CE, MODE, LAB = load_data()
    fps = np.load(os.path.join(DS, "fp_targets.npy"))
    nbits = int(open(os.path.join(DS, "nbits.txt")).read())
    dev = torch.device("cpu" if SMOKE else "cuda")
    tr = np.flatnonzero(LAB >= 0)
    rng = np.random.default_rng(0)
    mon = rng.choice(tr, min(20000, len(tr) // 10), replace=False)
    tr = np.setdiff1d(tr, mon)
    print(f"train {len(tr):,}  monitor {len(mon):,}  nbits {nbits}", flush=True)

    g = lambda a, dt=None: torch.as_tensor(a if dt is None else a.astype(dt)).to(dev)  # noqa: E731
    gMZ, gIT = g(MZ), g(IT)
    gAD, gINS, gCE, gMODE = g(AD, np.int64), g(INS, np.int64), g(CE), g(MODE)
    gLAB, gFP = g(np.maximum(LAB, 0), np.int64), g(fps)
    gDEC = g(np.load(os.path.join(DS, "decoys.npy")), np.int64)
    del MZ, IT
    shifts = torch.arange(7, -1, -1, device=dev, dtype=torch.uint8)

    def batch(idx, aug):
        idx = torch.as_tensor(idx, device=dev)
        mz, it = gMZ[idx], gIT[idx].float()
        if aug:  # drop 10 % of the peaks (never the precursor token nor all peaks), jitter intensities
            real = mz[:, 1:] > 0
            drop = (torch.rand(real.shape, device=dev) < 0.1) & real
            drop &= drop.sum(1, keepdim=True) < real.sum(1, keepdim=True)
            mz = mz.clone()
            mz[:, 1:] = mz[:, 1:].masked_fill(drop, 0.0)
            it = it.clone()
            it[:, 1:] = (it[:, 1:] * torch.exp(0.1 * torch.randn_like(it[:, 1:]))).clamp(0, 1).masked_fill(
                mz[:, 1:] == 0, 0.0)
        spec = torch.stack([mz, it], -1)
        lab = gLAB[idx]
        cand = torch.cat([lab[:, None], gDEC[lab]], 1)
        yc = ((gFP[cand.clamp(min=0)][..., None] >> shifts) & 1).reshape(len(idx), cand.shape[1], -1)[..., :nbits]
        return (spec, gAD[idx], gINS[idx], gCE[idx], gMODE[idx]), yc[:, 0].float(), (yc, cand >= 0)

    def contrastive(z, yc, valid):
        s = torch.einsum("bkn,bn->bk", yc.float(), z.float()).masked_fill(~valid, float("-inf"))
        return (s.argmax(1) == 0).float().mean(), valid.sum(1).float().mean()

    net = DB.DreamsFP(backbone(), nbits, len(M.ADDUCT_LIST), len(M.INSTR_LIST)).to(dev)
    model = nn.DataParallel(net) if torch.cuda.device_count() > 1 else net
    bb_params = list(net.backbone.parameters())
    bb_ids = {id(p) for p in bb_params}
    head_params = [p for p in net.parameters() if id(p) not in bb_ids]
    opt = torch.optim.AdamW([{"params": [p for p in bb_params if p.requires_grad], "lr": LR_BB, "base": LR_BB},
                             {"params": head_params, "lr": LR_HEAD, "base": LR_HEAD}],
                            weight_decay=WD, betas=(0.9, 0.98))
    scaler = torch.amp.GradScaler(enabled=not SMOKE)
    steps_per_epoch = 20 if SMOKE else len(tr) // BS
    total, step, ep = None, 0, 0
    t_train = time.time()
    last_save = time.time()
    log = open(os.path.join(OUT, "train_log.csv"), "w")
    log.write("epoch,step,train_bce,mon_bce,mon_top1_in_window,mon_cands,lr_bb,elapsed_s\n")

    def factor(s):
        if s < WARM:
            return (s + 1) / WARM
        if total is None:
            return 1.0
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - WARM) / max(1, total - WARM))))

    def save():
        DB.save_fp(net, os.path.join(OUT, "dreams_fp.pt"), N_PEAKS, extra={"steps": step, "epochs": ep})

    def evaluate():
        model.eval()
        ml = acc = ncand = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=not SMOKE):
            for b in range(0, len(mon), 512):
                x, y, (yc, valid) = batch(mon[b:b + 512], False)
                z = model(*x).float()
                ml += F.binary_cross_entropy_with_logits(z, y, reduction="sum").item()
                a, nc = contrastive(z, yc, valid)
                acc += a.item() * len(y)
                ncand += nc.item() * len(y)
        model.train()
        return ml / (len(mon) * nbits), acc / len(mon), ncand / len(mon)

    done = False
    while not done:
        perm = rng.permutation(tr)
        run, nrun = 0.0, 0
        model.train()
        for b in range(steps_per_epoch):
            x, y, _ = batch(perm[b * BS:(b + 1) * BS], True)
            f = factor(step)
            for gp in opt.param_groups:
                gp["lr"] = gp["base"] * f
            with torch.autocast("cuda", dtype=torch.float16, enabled=not SMOKE):
                z = model(*x)
            loss = F.binary_cross_entropy_with_logits(z.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if b % 50 == 0:
                run += loss.item()
                nrun += 1
            step += 1
            if total is None and step == (10 if SMOKE else 300):
                rate = step / (time.time() - t_train)
                remain = BUDGET_S - (time.time() - T_START) - 900
                total = int(min((2 if SMOKE else 30) * steps_per_epoch, step + remain * rate))
                print(f"{rate:.2f} steps/s ({rate * BS:.0f} spectra/s); total steps {total:,} "
                      f"(~{total / steps_per_epoch:.2f} epochs)", flush=True)
            if step % (5 if SMOKE else 2000) == 0:
                print(f"  step {step:,} loss {loss.item():.5f} lr_bb {opt.param_groups[0]['lr']:.2e} "
                      f"{time.time() - T_START:.0f}s", flush=True)
            if time.time() - last_save > 2700:  # checkpoint every 45 min: a timeout must not lose the run
                save()
                last_save = time.time()
            if total is not None and step >= total:
                done = True
                break
        ep += 1
        ml, acc, nc = evaluate()
        msg = (f"{ep},{step},{run / max(nrun, 1):.5f},{ml:.5f},{acc:.4f},{nc:.1f},{opt.param_groups[0]['lr']:.2e},"
               f"{time.time() - T_START:.0f}")
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
        save()
        last_save = time.time()
    print("done", time.time() - T_START, flush=True)


if __name__ == "__main__":
    main()
