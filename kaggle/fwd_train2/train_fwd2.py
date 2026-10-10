# Generated from kaggle/fwd_train/train_fwd.py (edit that file): forward model "fwd2" settings.
import os
os.environ.update(FWD_LAMBDA="1.0", FWD_K="15", FWD_TAU="0.05", FWD_NCON="256", FWD_EPOCHS="30",
                  FWD_MON_STRUCT="0.02")
"""Train our forward model (structure bits + conditions -> binned spectrum; casmi.fwdmodel.FwdNet, NEIMS-style)
on the competition train file, with the FP trainer's data pipeline (same spectra, same held-out panels).

Loss = (1 - cos) on the forward (fragment m/z) and reverse (loss from the precursor) 1-Da binned spectra, plus
(FWD_LAMBDA > 0, "fwd2") a contrastive term: for FWD_NCON spectra per batch, cross-entropy of softmax(sim / FWD_TAU)
over {truth} ∪ FWD_K same-window decoys (sim = cos_forward + cos_reverse, as the ranker feature uses it), so the
model learns to separate same-mass candidates, not only to reproduce spectra.
Monitor: cosine and top-1 among same-window decoys, on a training slice (structures seen in training) and, with
FWD_MON_STRUCT > 0, on spectra of a held-out fraction of the training structures (generalisation).
Output (/kaggle/working): fwdnet.pt ({model, nbits, d, blocks}; tensors only), train_log.csv

--- data notes from the FP trainer this reuses: ---

Kaggle GPU script. Inputs:
  * competition data (train.parquet)
  * our private dataset vigneshnehru/casmi26-fp-train:
      fpmodel.py       — model + peak preparation (our code)
      row_struct.npy   — int32 per train row: pool row of its structure (>= 0), -1 = excluded
                         (conflicting duplicate), -2 = held-out validation structure
      fp_targets.npy   — uint8 packed fingerprint bits per pool row (informative-bit subset)
      decoys.npy       — int32 (pool, 31): same ±10 ppm window pool rows, -1 padded
      nbits.txt
Loss = mean-bit BCE + LAMBDA * cross-entropy of softmax(f·z) over {truth} ∪ decoys (f·z is the
Bayes log-likelihood of a candidate fingerprint up to a candidate-independent constant).
Output (/kaggle/working): fpnet.pt ({model, nbits, d, layers}; tensors only), train_log.csv
"""

import glob
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

T_START = time.time()
BUDGET_S = float(os.environ.get("FP_BUDGET_S", 7.6 * 3600))
SMOKE = os.environ.get("FP_SMOKE") == "1"  # local CPU check: one row group, tiny model, few steps
DS = os.environ.get("FP_DS") or os.path.dirname(glob.glob("/kaggle/input/**/fp_targets.npy", recursive=True)[0])
TRAIN = os.environ.get("FP_TRAIN") or glob.glob("/kaggle/input/**/train.parquet", recursive=True)[0]
_CODE = glob.glob("/kaggle/input/**/code__fwdmodel.py", recursive=True)
if _CODE:  # our model code from casmi26-artifacts (package casmi: fwdmodel imports fpmodel)
    import shutil
    os.makedirs("/kaggle/working/fwcode/casmi", exist_ok=True)
    for f in glob.glob(os.path.join(os.path.dirname(_CODE[0]), "code__*.py")):
        shutil.copy(f, "/kaggle/working/fwcode/casmi/" + os.path.basename(f)[len("code__"):])
    sys.path.insert(0, "/kaggle/working/fwcode")
else:
    sys.path.insert(0, os.environ.get("FW_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src")))
from casmi import fpmodel as M  # noqa: E402
from casmi import fwdmodel as W  # noqa: E402

D, BLOCKS, NP = (128, 1, M.MAX_PEAKS) if SMOKE else (2048, 3, M.MAX_PEAKS)
BS, LR, WD, WARM = (64, 3e-4, 0.01, 10) if SMOKE else (1024, 3e-4, 0.01, 1000)
LAMBDA = float(os.environ.get("FWD_LAMBDA", 0.0))  # contrastive weight (0 = the original NEIMS-style model)
K_DEC = int(os.environ.get("FWD_K", 15))  # decoys per contrastive spectrum
TAU = float(os.environ.get("FWD_TAU", 0.05))
NCON = min(BS, int(os.environ.get("FWD_NCON", 16 if SMOKE else 256)))  # contrastive spectra per batch
EPOCHS = int(os.environ.get("FWD_EPOCHS", 30))
MON_STRUCT = float(os.environ.get("FWD_MON_STRUCT", 0.0))  # fraction of training structures held out (monitor)
ENVEDA_POS_SHIFT = -0.0004
OUT = os.environ.get("FP_OUT", "/kaggle/working")
print("dataset", DS, "train", TRAIN, "gpus", torch.cuda.device_count(), flush=True)


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
    MZ = np.zeros((k, NP), np.float32)
    IT = np.zeros((k, NP), np.float16)
    NPK = np.zeros(k, np.int16)
    for j in range(k):
        a, b = off[j], off[j + 1]
        m, i = M.prep_peaks(fm[a:b] + shift[j], fi[a:b], prec[j])
        MZ[j, :len(m)], IT[j, :len(m)], NPK[j] = m, i, len(m)
    ad = np.array([M.ADDUCT_IX.get(a, M.ADDUCT_IX["<unk>"]) for a in tbl["adduct"].to_pylist()], np.int16)
    ins = np.array([M.instr_family(s) for s in tbl["instrument_type"].to_pylist()], np.int8)
    return MZ, IT, NPK, prec.astype(np.float32), ad, ins, ce, pos.astype(np.float32), lab[sel]


def load_data():
    nrg = 1 if SMOKE else pq.ParquetFile(TRAIN).num_row_groups
    with Pool(4) as p:
        parts = p.map(_rg, range(nrg), chunksize=1)
    cat = [np.concatenate([q[i] for q in parts]) for i in range(9)]
    keep = cat[2] > 0
    print(f"prepared {keep.sum():,} spectra in {time.time() - T_START:.0f}s", flush=True)
    return [c[keep] for c in cat]


def main():
    MZ, IT, NPK, PREC, AD, INS, CE, MODE, LAB = load_data()
    fps = np.load(os.path.join(DS, "fp_targets.npy"))
    nbits = int(open(os.path.join(DS, "nbits.txt")).read())
    dev = torch.device("cpu" if SMOKE else "cuda")
    tr = np.flatnonzero(LAB >= 0)
    print(f"train rows {len(tr):,}  nbits {nbits}", flush=True)
    rng = np.random.default_rng(0)
    mon = rng.choice(tr, min(20000, len(tr) // 10), replace=False)
    tr = np.setdiff1d(tr, mon)
    mon_new = np.zeros(0, np.int64)
    if MON_STRUCT > 0:  # structure-disjoint monitor: all spectra of a random fraction of the training structures
        st = np.unique(LAB[tr])
        held = rng.choice(st, int(len(st) * MON_STRUCT), replace=False)
        mon_new = tr[np.isin(LAB[tr], held)]
        mon_new = rng.choice(mon_new, min(20000, len(mon_new)), replace=False)
        tr = tr[~np.isin(LAB[tr], held)]
        mon = mon[~np.isin(LAB[mon], held)]
    print(f"train {len(tr):,}  monitor seen {len(mon):,}  monitor held-out structures {len(mon_new):,}  "
          f"lambda {LAMBDA} K {K_DEC} tau {TAU} ncon {NCON}", flush=True)
    g = lambda a, dt=None: torch.as_tensor(a if dt is None else a.astype(dt)).to(dev)  # noqa: E731
    gMZ, gIT, gNPK, gPREC = g(MZ), g(IT), g(NPK, np.int64), g(PREC)
    gAD, gINS, gCE, gMODE = g(AD, np.int64), g(INS, np.int64), g(CE), g(MODE)
    gLAB, gFP = g(np.maximum(LAB, 0), np.int64), g(fps)
    gDEC = g(np.load(os.path.join(DS, "decoys.npy")), np.int64)
    del MZ, IT
    shifts = torch.arange(7, -1, -1, device=dev, dtype=torch.uint8)
    NB = W.NBINS

    def bits_of(rows):
        return ((gFP[rows][..., None] >> shifts) & 1).reshape(*rows.shape, -1)[..., :nbits].float()

    def targets(idx):
        mz, it, n = gMZ[idx], gIT[idx].float(), gNPK[idx]
        pad = torch.arange(NP, device=dev)[None, :] >= n[:, None]
        prec = gPREC[idx][:, None]
        it = it.masked_fill(pad | (mz > prec - 0.5), 0)
        fi = mz.floor().long().clamp(0, NB - 1)
        ri = (prec - mz).floor().long().clamp(0, NB - 1)
        f = torch.zeros(len(idx), NB, device=dev).scatter_add_(1, fi, it)
        r = torch.zeros(len(idx), NB, device=dev).scatter_add_(1, ri, it)
        return F.normalize(f, dim=1), F.normalize(r, dim=1)

    def cond(idx):
        return gAD[idx], gINS[idx], gCE[idx], gMODE[idx], gPREC[idx]

    net = W.FwdNet(nbits, d=D, blocks=BLOCKS).to(dev)
    model = nn.DataParallel(net) if torch.cuda.device_count() > 1 else net
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    steps_per_epoch = 30 if SMOKE else len(tr) // BS

    @torch.no_grad()
    def monitor(rows):
        """Mean forward / reverse cosine of the truth's prediction and top-1 among the truth + all same-window decoys."""
        if len(rows) == 0:
            return float("nan"), float("nan"), float("nan")
        cf = cr = top1 = 0.0
        for b in range(0, len(rows), 512):
            idx = torch.as_tensor(rows[b:b + 512], device=dev)
            tf, trv = targets(idx)
            lab = gLAB[idx]
            cand = torch.cat([lab[:, None], gDEC[lab]], 1)  # truth first, then same-window decoys
            valid = cand >= 0
            k = cand.shape[1]
            c = [x.repeat_interleave(k) for x in cond(idx)]
            pf, pr = model(bits_of(cand.clamp(min=0)).reshape(-1, nbits), *c)
            sim = (F.cosine_similarity(pf.float(), tf.repeat_interleave(k, 0), dim=1)
                   + F.cosine_similarity(pr.float(), trv.repeat_interleave(k, 0), dim=1)).reshape(-1, k)
            sim = sim.masked_fill(~valid, -9)
            cf += F.cosine_similarity(pf.float().reshape(-1, k, NB)[:, 0], tf, dim=1).sum().item()
            cr += F.cosine_similarity(pr.float().reshape(-1, k, NB)[:, 0], trv, dim=1).sum().item()
            top1 += (sim.argmax(1) == 0).float().sum().item()
        return cf / len(rows), cr / len(rows), top1 / len(rows)

    def contrastive(idx, pf, pr, tf, trv):
        """Cross-entropy of the truth among itself + K sampled same-window decoys (first NCON rows of the batch)."""
        idx, pf, pr, tf, trv = idx[:NCON], pf[:NCON], pr[:NCON], tf[:NCON], trv[:NCON]
        dec = gDEC[gLAB[idx]]
        pick = torch.argsort(torch.rand(dec.shape, device=dev), 1)[:, :K_DEC]
        d = dec.gather(1, pick)
        valid = d >= 0
        c = [x.repeat_interleave(K_DEC) for x in cond(idx)]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not SMOKE):
            df, dr = model(bits_of(d.clamp(min=0)).reshape(-1, nbits), *c)
        sd = (F.cosine_similarity(df.float(), tf.repeat_interleave(K_DEC, 0), dim=1)
              + F.cosine_similarity(dr.float(), trv.repeat_interleave(K_DEC, 0), dim=1)).reshape(-1, K_DEC)
        st = F.cosine_similarity(pf.float(), tf, dim=1) + F.cosine_similarity(pr.float(), trv, dim=1)
        logits = torch.cat([st[:, None], sd.masked_fill(~valid, float("-inf"))], 1) / TAU
        return F.cross_entropy(logits, torch.zeros(len(idx), dtype=torch.long, device=dev))
    total, step, ep = None, 0, 0
    t_train = time.time()
    log = open(os.path.join(OUT, "train_log.csv"), "w")
    log.write("epoch,step,train_loss,mon_cos_fwd,mon_cos_rev,mon_top1_in_window,new_cos_fwd,new_cos_rev,"
              "new_top1_in_window,elapsed_s\n")

    def lr_at(s):
        if s < WARM:
            return LR * (s + 1) / WARM
        if total is None:
            return LR
        return LR * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - WARM) / max(1, total - WARM))))

    def save():
        torch.save({"model": {k: v.detach().half().cpu() for k, v in net.state_dict().items()}, "nbits": nbits,
                    "d": D, "blocks": BLOCKS}, os.path.join(OUT, "fwdnet.pt"))

    while True:
        perm = rng.permutation(tr)
        run, nrun = 0.0, 0
        model.train()
        for b in range(steps_per_epoch):
            idx = torch.as_tensor(perm[b * BS:(b + 1) * BS], device=dev)
            tf, trv = targets(idx)
            for gp in opt.param_groups:
                gp["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not SMOKE):
                pf, pr = model(bits_of(gLAB[idx]), *cond(idx))
            loss = (2 - F.cosine_similarity(pf.float(), tf, dim=1) - F.cosine_similarity(pr.float(), trv, dim=1)).mean()
            if LAMBDA > 0:
                loss = loss + LAMBDA * contrastive(idx, pf, pr, tf, trv)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if b % 50 == 0:
                run += loss.item()
                nrun += 1
            step += 1
            if total is None and step == (20 if SMOKE else 300):
                rate = step / (time.time() - t_train)
                remain = BUDGET_S - (time.time() - T_START) - 600
                total = int(min((2 if SMOKE else EPOCHS) * steps_per_epoch, step + remain * rate))
                print(f"{rate:.2f} steps/s; total steps {total:,} (~{total / steps_per_epoch:.1f} epochs)", flush=True)
            if total is not None and step >= total:
                break
        ep += 1
        model.eval()
        cf, cr, top1 = monitor(mon)
        nf, nr, ntop1 = monitor(mon_new)
        msg = (f"{ep},{step},{run / max(nrun, 1):.4f},{cf:.4f},{cr:.4f},{top1:.4f},{nf:.4f},{nr:.4f},{ntop1:.4f},"
               f"{time.time() - T_START:.0f}")
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
        save()
        if total is not None and step >= total:
            break
    print("done", time.time() - T_START, flush=True)


if __name__ == "__main__":
    main()
