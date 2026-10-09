"""Train our spectrum -> fingerprint transformer (casmi.fpmodel.FPNet) on the competition train file.

Run 6 = run 3's recipe (pure BCE; no PairBias, jitter, bit-prior init or merge augmentation) + formula-annotated
peaks (MIST-style): every peak carries the sub-formula of the precursor ion that explains it (formula.annotate,
10 ppm / 2 mDa; unexplained = zeros) and the loss formula; the global token carries the precursor ion formula.
At inference the formula is a candidate's: the model runs once per distinct candidate formula.
The model code comes from our artifacts dataset (code__fpmodel.py), the targets from casmi26-fp-train.

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
_CODE = glob.glob("/kaggle/input/**/code__fpmodel.py", recursive=True)
if _CODE:  # the current model code (with PairBias) from casmi26-artifacts
    os.makedirs("/kaggle/working/fpcode", exist_ok=True)
    import shutil
    shutil.copy(_CODE[0], "/kaggle/working/fpcode/fpmodel.py")
    shutil.copy(_CODE[0].replace("code__fpmodel.py", "code__formula.py"), "/kaggle/working/fpcode/formula.py")
    sys.path.insert(0, "/kaggle/working/fpcode")
else:
    sys.path.insert(0, DS)
import formula as Fm  # noqa: E402
import fpmodel as M  # noqa: E402
REL = False  # run 4 (PairBias) tied run 3 on held-out panels at 25 % lower speed: spend the budget on epochs
FORMULA = True
assert "formula" in M.FPNet.__init__.__code__.co_varnames, "fpmodel without formula inputs"

# run 5's extras (PairBias, m/z jitter, bit-prior init, merge augmentation) scored below run 3 on the LB twice
# (V15 vs V13, V16b vs V16a): run 6 = run 3's recipe + formula annotations only, so the LB isolates formulas
MERGE_P = float(os.environ.get("FP_MERGE_P", 0.0))
JITTER = False
PRIOR_INIT = False
assert "rel" in M.FPNet.__init__.__code__.co_varnames, "fpmodel without PairBias"

D, LAYERS, NP = (64, 1, M.MAX_PEAKS) if SMOKE else (512, 6, M.MAX_PEAKS)
BS, LR, WD, WARM = (64, 4e-4, 0.01, 10) if SMOKE else (512, 4e-4, 0.01, 2000)
LAMBDA = float(os.environ.get("FP_LAMBDA", 0.0))  # run 3: pure BCE (contrastive term dominated run 1)
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
                                        "ingest_lib", "molecular_formula"]).take(sel)
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
    FR = np.zeros((k, NP, M.N_ELS), np.uint8)
    PF = np.zeros((k, M.N_ELS), np.int16)
    forms, adds = tbl["molecular_formula"].to_pylist(), tbl["adduct"].to_pylist()
    # spectra grouped by (formula, adduct): one sub-formula table in memory at a time (they reach ~1e6 rows)
    order = sorted(range(k), key=lambda j: (str(forms[j]), str(adds[j])))
    cache, ckey = {}, None
    for j in order:
        a, b = off[j], off[j + 1]
        m, i = M.prep_peaks(fm[a:b] + shift[j], fi[a:b], prec[j])
        MZ[j, :len(m)], IT[j, :len(m)], NPK[j] = m, i, len(m)
        if (forms[j], adds[j]) != ckey:
            cache, ckey = {}, (forms[j], adds[j])
        an = Fm.annotate(m, forms[j], adds[j], cache=cache) if len(m) else None
        if an is not None:
            FR[j, :len(m)], PF[j] = an[0], np.minimum(an[1], 32767)
    ad = np.array([M.ADDUCT_IX.get(a, M.ADDUCT_IX["<unk>"]) for a in tbl["adduct"].to_pylist()], np.int16)
    ins = np.array([M.instr_family(s) for s in tbl["instrument_type"].to_pylist()], np.int8)
    return MZ, IT, NPK, prec.astype(np.float32), ad, ins, ce, pos.astype(np.float32), lab[sel], FR, PF


def load_data():
    nrg = 1 if SMOKE else pq.ParquetFile(TRAIN).num_row_groups
    with Pool(4) as p:
        parts = p.map(_rg, range(nrg), chunksize=1)
    cat = [np.concatenate([q[i] for q in parts]) for i in range(11)]
    keep = cat[2] > 0
    print(f"prepared {keep.sum():,} spectra in {time.time() - T_START:.0f}s", flush=True)
    return [c[keep] for c in cat]


def main():
    MZ, IT, NPK, PREC, AD, INS, CE, MODE, LAB, FR, PF = load_data()
    has = PF.sum(1) > 0
    print(f"formula-annotated spectra {has.mean():.3f}; peaks explained "
          f"{((FR.sum(2) > 0).sum(1)[has] / np.maximum(NPK[has], 1)).mean():.3f}", flush=True)
    fps = np.load(os.path.join(DS, "fp_targets.npy"))
    nbits = int(open(os.path.join(DS, "nbits.txt")).read())
    dev = torch.device("cpu" if SMOKE else "cuda")
    tr = np.flatnonzero(LAB >= 0)
    va = np.flatnonzero(LAB == -2)
    print(f"train {len(tr):,}  val rows {len(va):,}  nbits {nbits}", flush=True)
    # The validation structures have no targets here (held out entirely); monitor a train slice instead.
    rng = np.random.default_rng(0)
    mon = rng.choice(tr, min(20000, len(tr) // 10), replace=False)
    tr = np.setdiff1d(tr, mon)

    g = lambda a, dt=None: torch.as_tensor(a if dt is None else a.astype(dt)).to(dev)  # noqa: E731
    gMZ, gIT, gNPK, gPREC = g(MZ), g(IT), g(NPK, np.int64), g(PREC)
    gAD, gINS, gCE, gMODE = g(AD, np.int64), g(INS, np.int64), g(CE), g(MODE)
    gLAB, gFP = g(np.maximum(LAB, 0), np.int64), g(fps)
    gFR, gPF = g(FR), g(PF, np.float32)
    del FR
    # merge partners: spectra sharing structure and adduct (CSR over (label, adduct) groups)
    gk = LAB.astype(np.int64) * 64 + AD.astype(np.int64)
    order = np.argsort(gk, kind="stable")
    _, start, cnt = np.unique(gk[order], return_index=True, return_counts=True)
    grp_of = np.empty(len(gk), np.int64)
    grp_of[order] = np.repeat(np.arange(len(start)), cnt)
    gORD, gSTART, gCNT, gGRP = g(order, np.int64), g(start, np.int64), g(cnt, np.int64), g(grp_of, np.int64)
    print(f"merge groups {len(start):,}, spectra with a partner {(cnt[grp_of] > 1).mean():.2f}", flush=True)
    gDEC = g(np.load(os.path.join(DS, "decoys.npy")), np.int64)
    del MZ, IT
    shifts = torch.arange(7, -1, -1, device=dev, dtype=torch.uint8)

    def batch(idx, aug):
        idx = torch.as_tensor(idx, device=dev)
        mz, it, n = gMZ[idx], gIT[idx].float(), gNPK[idx]
        fr, pf = gFR[idx], gPF[idx]
        pad = torch.arange(NP, device=dev)[None, :] >= n[:, None]
        if aug and MERGE_P > 0:  # merge with a random same-structure, same-adduct spectrum
            gi = gGRP[idx]
            pick = gORD[gSTART[gi] + (torch.rand(len(idx), device=dev) * gCNT[gi]).long().clamp(max=gCNT[gi] - 1)]
            m = (torch.rand(len(idx), device=dev) < MERGE_P) & (gCNT[gi] > 1) & (pick != idx)
            if m.any():
                i2 = pick[m]
                mz2, it2, n2 = gMZ[i2], gIT[i2].float(), gNPK[i2]
                pad2 = torch.arange(NP, device=dev)[None, :] >= n2[:, None]
                cm = torch.cat([mz[m], mz2], 1)
                ci = torch.cat([it[m].masked_fill(pad[m], -1), it2.masked_fill(pad2, -1)], 1)
                cf = torch.cat([fr[m], gFR[i2]], 1)
                top = ci.topk(NP, dim=1).indices
                tm, ti = cm.gather(1, top), ci.gather(1, top)
                tf = cf.gather(1, top[..., None].expand(-1, -1, cf.shape[-1]))
                tm = torch.where(ti < 0, torch.full_like(tm, 1e4), tm)
                o = tm.argsort(1)
                tm, ti = tm.gather(1, o), ti.gather(1, o)
                tf = tf.gather(1, o[..., None].expand(-1, -1, tf.shape[-1]))
                newpad = ti < 0
                mz, it, pad, fr = mz.clone(), it.clone(), pad.clone(), fr.clone()
                mz[m], it[m], pad[m] = tm.masked_fill(newpad, 0), ti.clamp(min=0), newpad
                fr[m] = tf.masked_fill(newpad[..., None], 0)
        if aug:
            drop = (torch.rand(pad.shape, device=dev) < 0.1) & ~pad
            drop &= drop.sum(1, keepdim=True) < n[:, None]
            pad = pad | drop
            it = (it * torch.exp(0.1 * torch.randn_like(it))).clamp(0, 1.5)
            if JITTER:
                mz = mz * (1 + 3e-6 * torch.randn_like(mz))  # instrument-level m/z error
        it = it.masked_fill(pad, 0)
        lab = gLAB[idx]
        cand = torch.cat([lab[:, None], gDEC[lab]], 1)  # (B, 32): truth first
        yc = ((gFP[cand.clamp(min=0)][..., None] >> shifts) & 1).reshape(len(idx), cand.shape[1], -1)[..., :nbits]
        return (mz, it, pad, gPREC[idx], gAD[idx], gINS[idx], gCE[idx], gMODE[idx], fr, pf), yc[:, 0].float(), \
            (yc, cand >= 0)

    def contrastive(z, yc, valid):
        s = torch.einsum("bkn,bn->bk", yc.float(), z.float()).masked_fill(~valid, float("-inf"))
        return F.cross_entropy(s, torch.zeros(len(s), dtype=torch.long, device=s.device)), \
            (s.argmax(1) == 0).float().mean(), valid.sum(1).float().mean()

    net = M.FPNet(nbits, d=D, layers=LAYERS, rel=REL, formula=FORMULA).to(dev)
    if PRIOR_INIT:  # bit-prior initialisation (off for run 6): start from each bit's training frequency
        with torch.no_grad():
            smp = torch.as_tensor(rng.choice(tr, min(50000, len(tr)), replace=False), device=dev)
            yb = ((gFP[gLAB[smp]][..., None] >> shifts) & 1).reshape(len(smp), -1)[:, :nbits].float().mean(0)
            net.head[-1].bias.copy_(torch.logit(yb.clamp(1e-4, 1 - 1e-4)))
            print("bit prior init: mean freq", float(yb.mean()), flush=True)
    model = nn.DataParallel(net) if torch.cuda.device_count() > 1 else net
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.98))
    scaler = torch.amp.GradScaler(enabled=not SMOKE)
    steps_per_epoch = 30 if SMOKE else len(tr) // BS
    total, step, ep = None, 0, 0
    t_train = time.time()
    log = open(os.path.join(OUT, "train_log.csv"), "w")
    log.write("epoch,step,train_bce,train_ce,mon_bce,mon_top1_in_window,mon_cands,lr,elapsed_s\n")

    def lr_at(s):
        if s < WARM:
            return LR * (s + 1) / WARM
        if total is None:
            return LR
        return LR * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - WARM) / max(1, total - WARM))))

    def save():
        # fp16 halves the file, but the sinusoidal tables (*.inv) must stay float32
        sd = {k: (v.detach().cpu() if k.endswith(".inv") else v.detach().half().cpu())
              for k, v in net.state_dict().items()}
        torch.save({"model": sd, "nbits": nbits, "d": D, "layers": LAYERS, "rel": REL,
                    "formula": FORMULA}, os.path.join(OUT, "fpnet.pt"))

    while True:
        perm = rng.permutation(tr)
        run = run_ce = 0.0
        model.train()
        for b in range(steps_per_epoch):
            x, y, (yc, valid) = batch(perm[b * BS:(b + 1) * BS], True)
            for gp in opt.param_groups:
                gp["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=torch.float16, enabled=not SMOKE):
                z = model(*x)
            bce = F.binary_cross_entropy_with_logits(z.float(), y)
            ce, _, _ = contrastive(z, yc, valid)
            loss = bce + LAMBDA * ce
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if b % 50 == 0:
                run += bce.item()
                run_ce += ce.item()
            step += 1
            if total is None and step == (20 if SMOKE else 400):
                rate = step / (time.time() - t_train)
                remain = BUDGET_S - (time.time() - T_START) - 600
                total = int(min((2 if SMOKE else 40) * steps_per_epoch, step + remain * rate))
                print(f"{rate:.2f} steps/s; total steps {total:,} (~{total / steps_per_epoch:.1f} epochs)",
                      flush=True)
            if total is not None and step >= total:
                break
        ep += 1
        model.eval()
        ml = acc = ncand = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=not SMOKE):
            for b in range(0, len(mon), 1024):
                x, y, (yc, valid) = batch(mon[b:b + 1024], False)
                z = model(*x).float()
                ml += F.binary_cross_entropy_with_logits(z, y, reduction="sum").item()
                _, a, nc = contrastive(z, yc, valid)
                acc += a.item() * len(y)
                ncand += nc.item() * len(y)
        ml /= len(mon) * nbits
        nb = max(1, steps_per_epoch // 50)
        msg = (f"{ep},{step},{run / nb:.5f},{run_ce / nb:.4f},{ml:.5f},{acc / len(mon):.4f},{ncand / len(mon):.1f},"
               f"{lr_at(step):.2e},{time.time() - T_START:.0f}")
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
        save()
        if total is not None and step >= total:
            break
    print("done", time.time() - T_START, flush=True)


if __name__ == "__main__":
    main()
