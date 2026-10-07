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


def bce(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    lib = L.load(f"{ART}/library.npz")
    qs = pd.read_parquet(f"{EV}/queries.parquet")
    fp_full = np.load(f"{ART}/fp_full.npy", mmap_mode="r")
    bits = np.load(f"{ART}/fp_bits.npy")
    zp = np.load(f"{ART}/fp_prior.npy")
    pool = pd.read_parquet(f"{ART}/pool.parquet", columns=["key", "exact_mass", "src"])
    row_of = pd.Series(np.arange(len(pool)), index=pool.key.values)
    fpm = P.load_fp_models(os.path.dirname(FPN[0]))
    fpm["bits"] = bits
    from casmi import fpmodel as M
    # held-out panel molecules: single-spectrum view only (exactly the training input) and pipeline view
    rng = np.random.default_rng(0)
    qs = qs[qs.key.isin(rng.choice(qs.key.unique(), 600, replace=False))]
    # control: 300 *training* structures (seen by the model), one library spectrum each
    lkeys = lib["keys"][lib["key_code"]]
    held = set(pd.read_parquet(f"{EV}/queries.parquet").key)
    cand_rows = rng.choice(len(lkeys), 20000, replace=False)
    cand_rows = [r for r in cand_rows if lkeys[r] not in held][:300]
    qs = pd.concat([qs, pd.DataFrame({"lrow": cand_rows, "key": lkeys[cand_rows], "panel": "TRAIN(seen)"})],
                   ignore_index=True)
    raw = raw_peaks(lib["row"][qs.lrow.values])
    libs = list(lib["libs"])
    res = []
    for key, g in qs.groupby("key", sort=False):
        if key not in row_of.index:
            continue
        pr = int(row_of[key])
        y = np.unpackbits(np.asarray(fp_full[pr]), count=P.FULL_BITS)[bits].astype(np.float32)
        q = Query(key, [], [], [], [], [], [])
        for lr in g.lrow.values:
            a, b = lib["off"][lr], lib["off"][lr + 1]
            q.mz.append(lib["mz"][a:b]); q.p.append(lib["p"][a:b])
            q.adduct.append(L.ADDUCTS[lib["adduct_code"][lr]]); q.mode.append(int(lib["mode"][lr]))
            q.prec.append(float(lib["prec_mz"][lr])); q.ce.append(float(lib["ce"][lr]))
            mz, it = raw[int(lib["row"][lr])]
            env = libs[lib["lib_code"][lr]].startswith("enveda") and lib["mode"][lr] > 0
            q.raw.append((mz + (L.ENVEDA_POS_SHIFT if env else 0.0), it))
        lr0 = g.lrow.values[0]
        single = M.logits(fpm["nets"], [M.prep_peaks(*q.raw[0], q.prec[0])], [q.prec[0]], [q.adduct[0]],
                          [str(libs[lib["lib_code"][lr0]])], [0.0 if np.isnan(q.ce[0]) else q.ce[0]],
                          [1.0 if q.mode[0] > 0 else 0.0], device=fpm["device"])[0]
        zpipe = P.fp_logits(q, fpm)
        # same-formula (same exact mass) candidates in the pool window
        m = pool.exact_mass.values[pr]
        lo, hi = np.searchsorted(pool.exact_mass.values, [m - 1e-4, m + 1e-4])
        cand = np.arange(lo, hi)
        Y = np.unpackbits(np.asarray(fp_full[cand]), axis=1, count=P.FULL_BITS)[:, bits].astype(np.float32)
        out = {"key": key, "panel": g.panel.iloc[0], "n_iso": len(cand)}
        for name, z in (("single", single), ("pipeline", zpipe)):
            s = Y @ z
            rank = 1 + int((s > s[cand == pr][0]).sum())
            out[f"rr_{name}"] = 1.0 / rank
            out[f"bce_{name}"] = bce(1 / (1 + np.exp(-z)), y)
            sn = Y @ (z - zp)
            out[f"rr_{name}_norm"] = 1.0 / (1 + int((sn > sn[cand == pr][0]).sum()))
        out["bce_prior"] = bce(1 / (1 + np.exp(-zp)), y)
        out["rr_random"] = float(np.mean(1.0 / np.arange(1, len(cand) + 1)))
        res.append(out)
    r = pd.DataFrame(res)
    r = r[r.n_iso > 1]
    pd.set_option("display.width", 200)
    print("=== FP DIAGNOSTIC: held-out molecules, truth vs its same-formula pool isomers ===")
    print(r.groupby("panel")[[c for c in r.columns if c.startswith(("rr_", "bce_"))]].mean().round(4))
    print("n", r.groupby("panel").size().to_dict())


if __name__ == "__main__":
    main()
