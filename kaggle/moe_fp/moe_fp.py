"""Add our FP model's f·z feature to the harness candidates and train the ranker (Kaggle GPU).

Inputs: competition data (train.parquet for raw query peaks); host rdkit wheel; our private datasets
casmi26-artifacts (code, pool fingerprints, library) and casmi26-eval (harness features with SMILES,
queries); our FP-training kernel output (fpnet.pt). The FP model must have held out every panel.
Output (/kaggle/working): ranker{0,1,2}.txt, ranker_features.json, fpnet.pt — attached by the
submission notebook. The validation table is printed to the log.
"""

import glob
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
INP = os.environ.get("KAGGLE_INPUT", "/kaggle/input")  # overridable for a local dry run
OUT = os.environ.get("RANKER_OUT", "/kaggle/working")
tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
whl = [w for w in glob.glob(f"{INP}/**/rdkit*.whl", recursive=True) if f"-{tag}-" in w][:1]
if whl:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", *whl], check=False)
ART = os.path.dirname(glob.glob(f"{INP}/**/casmi26_artifacts.txt", recursive=True)[0])
EV = os.path.dirname(glob.glob(f"{INP}/**/casmi26_eval.txt", recursive=True)[0])
FPN = glob.glob(f"{INP}/**/fpnet.pt", recursive=True)
TRAIN = glob.glob(f"{INP}/**/train.parquet", recursive=True)[0]
os.makedirs(f"{OUT}/code/casmi", exist_ok=True)
for f in glob.glob(f"{ART}/code__*.py"):
    shutil.copy(f, f"{OUT}/code/casmi/" + os.path.basename(f)[len("code__"):])
sys.path.insert(0, f"{OUT}/code")
print("artifacts", ART, "eval", EV, "fpnet", FPN, flush=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from casmi import library as L  # noqa: E402
from casmi import pipeline as P  # noqa: E402
from casmi import rank as R  # noqa: E402
from casmi.fp import full_fp  # noqa: E402
from casmi import fpmodel as M  # noqa: E402
from casmi.search import Query  # noqa: E402


def raw_peaks(rows):
    rows = np.sort(np.unique(rows))
    f = pq.ParquetFile(TRAIN)
    out, start = {}, 0
    for rg in range(f.num_row_groups):
        n = f.metadata.row_group(rg).num_rows
        sel = rows[(rows >= start) & (rows < start + n)] - start
        if len(sel):
            t = f.read_row_group(rg, columns=["ms2_mzs", "ms2_normalized_intensities"]).take(sel)
            for loc, mz, it in zip(sel, t["ms2_mzs"].to_pylist(), t["ms2_normalized_intensities"].to_pylist()):
                out[start + int(loc)] = (np.asarray(mz, float), np.asarray(it, float))
        start += n
    return out


def main():
    lib = L.load(f"{ART}/library.npz")
    qs = pd.read_parquet(f"{EV}/queries.parquet")
    feats_df = pd.read_parquet(f"{EV}/features.parquet")
    # the submission runs without the generator (LB: it costs 0.017), so train / validate likewise
    feats_df = feats_df[(feats_df.is_gen == 0) & feats_df.regime.isin(["C1", "C2"])].reset_index(drop=True)
    fp_full = np.load(f"{ART}/fp_full.npy", mmap_mode="r")
    bits = np.load(f"{ART}/fp_bits.npy")
    fpm = P.load_fp_models(os.path.dirname(FPN[0]))
    fpm["bits"] = bits
    print("fp nets", len(fpm["nets"]), "device", fpm["device"], f"{time.time() - T0:.0f}s", flush=True)

    raw = raw_peaks(lib["row"][qs.lrow.values])
    libs = list(lib["libs"])
    z_of = {}
    for key, g in qs.groupby("key", sort=False):
        q = Query(key, [], [], [], [], [], [])
        for lr in g.lrow.values:
            a, b = lib["off"][lr], lib["off"][lr + 1]
            q.mz.append(lib["mz"][a:b])
            q.p.append(lib["p"][a:b])
            q.adduct.append(L.ADDUCTS[lib["adduct_code"][lr]])
            q.mode.append(int(lib["mode"][lr]))
            q.prec.append(float(lib["prec_mz"][lr]))
            q.ce.append(float(lib["ce"][lr]))
            mz, it = raw[int(lib["row"][lr])]
            env = libs[lib["lib_code"][lr]].startswith("enveda") and lib["mode"][lr] > 0
            q.raw.append((mz + (L.ENVEDA_POS_SHIFT if env else 0.0), it))
        q.instr = [M.INSTR_LIST[int(lib["instr"][lr])] for lr in g.lrow.values]  # true instrument
        z_of[key] = P.fp_logits(q, fpm)
    print("logits", len(z_of), f"{time.time() - T0:.0f}s", flush=True)

    # candidate fingerprints: pool rows from the packed matrix, generated rows recomputed
    gen = feats_df.pool_row.values < 0
    gsmi = pd.unique(feats_df.smiles.values[gen])
    gfp = {s: full_fp(s) for s in gsmi}
    print("generated fingerprints", len(gfp), f"{time.time() - T0:.0f}s", flush=True)
    fz = np.zeros(len(feats_df), np.float32)
    zp = np.load(f"{ART}/fp_prior.npy") if os.path.exists(f"{ART}/fp_prior.npy") else None
    fzn = np.zeros(len(feats_df), np.float32) if zp is not None else None
    for (key, _), g in feats_df.groupby(["qkey", "regime"], sort=False):
        z = z_of[key].astype(np.float32)
        pr = g.pool_row.values
        fps = np.stack([np.asarray(fp_full[r]) if r >= 0 else gfp[s] for r, s in zip(pr, g.smiles.values)])
        y = np.unpackbits(fps, axis=1, count=P.FULL_BITS)[:, bits].astype(np.float32)
        fz[g.index.values] = y @ z
        if zp is not None:
            fzn[g.index.values] = y @ (z - zp)
    feats_df["fz"] = fz
    grp = feats_df.groupby(["qkey", "regime"], sort=False).fz
    feats_df["fp"] = feats_df.fz - grp.transform("max")
    feats_df["fp_rank"] = grp.rank(ascending=False, method="min")
    if fzn is not None:  # f·(z - z_prior): what the spectrum adds over the bit-frequency prior
        feats_df["fzn"] = fzn
        gn = feats_df.groupby(["qkey", "regime"], sort=False).fzn
        feats_df["fp_norm"] = feats_df.fzn - gn.transform("max")
        feats_df = feats_df.drop(columns=["fzn"])
    print("f·z done", f"{time.time() - T0:.0f}s", flush=True)

    f = R.prepare(feats_df.drop(columns=["fz"]), regimes=("C1", "C2"))
    from casmi import moe
    l1, meta = moe.cv(f)

    def blended(s):
        out = np.empty(len(f))
        for _, g in f.groupby("grp", sort=False):
            out[g.index] = R.blend_pop(s[g.index], g["pop"].values) if "pop" in g else s[g.index]
        return out

    rows = {f"expert:{n}": R.mrr_of(f, blended(s)) for n, s in l1.items()}
    rows["meta (stacked)"] = R.mrr_of(f, blended(meta))
    tab = pd.DataFrame({k: r.groupby(["panel", "regime"]).mrr.mean() for k, r in rows.items()}).T
    pd.set_option("display.width", 200)
    print("=== MULTI-MODEL VALIDATION (out of fold, molecules held out, with FP) ===")
    print(tab.round(4).to_string())
    print("weighted (0.16 C1 + 0.30 C2):", {k: {p: round(0.16 * tab.loc[k, (p, "C1")] + 0.30 * tab.loc[k, (p, "C2")], 4)
                                              for p in "AC"} for k in tab.index}, flush=True)
    moe.fit_save(f, OUT)
    open(f"{OUT}/casmi26_models.txt", "w").write("multi-model ranker with FP expert\n")
    shutil.copy(FPN[0], f"{OUT}/fpnet.pt")
    print("saved", sorted(os.listdir(OUT)), f"{time.time() - T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
