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
        z_of[key] = P.fp_logits(q, fpm)
    print("logits", len(z_of), f"{time.time() - T0:.0f}s", flush=True)

    # candidate fingerprints: pool rows from the packed matrix, generated rows recomputed
    gen = feats_df.pool_row.values < 0
    gsmi = pd.unique(feats_df.smiles.values[gen])
    gfp = {s: full_fp(s) for s in gsmi}
    print("generated fingerprints", len(gfp), f"{time.time() - T0:.0f}s", flush=True)
    fz = np.zeros(len(feats_df), np.float32)
    for (key, _), g in feats_df.groupby(["qkey", "regime"], sort=False):
        z = z_of[key].astype(np.float32)
        pr = g.pool_row.values
        fps = np.stack([np.asarray(fp_full[r]) if r >= 0 else gfp[s] for r, s in zip(pr, g.smiles.values)])
        y = np.unpackbits(fps, axis=1, count=P.FULL_BITS)[:, bits].astype(np.float32)
        fz[g.index.values] = y @ z
    feats_df["fz"] = fz
    grp = feats_df.groupby(["qkey", "regime"], sort=False).fz
    feats_df["fp"] = feats_df.fz - grp.transform("max")
    feats_df["fp_rank"] = grp.rank(ascending=False, method="min")
    print("f·z done", f"{time.time() - T0:.0f}s", flush=True)

    f = R.prepare(feats_df.drop(columns=["fz"]))
    feats = R.feature_cols(f)
    base = [c for c in feats if not c.startswith("fp")]
    rep_fp = R.mrr_of(f, R.cv(f, feats))
    rep_base = R.mrr_of(f, R.cv(f, base))
    fz_only = R.mrr_of(f, f.fp.values + 1e-6 * np.random.default_rng(0).random(len(f)))
    rep = rep_fp.merge(rep_base.rename(columns={"mrr": "mrr_nofp"}), on=["qkey", "regime", "panel"]) \
                .merge(fz_only.rename(columns={"mrr": "mrr_fz_only"}), on=["qkey", "regime", "panel"])
    print("=== VALIDATION (out of fold, molecules held out) ===")
    print(R.report(rep, ("mrr", "mrr_nofp", "mrr_fz_only")), flush=True)
    R.fit_save(f, feats, OUT)
    shutil.copy(FPN[0], f"{OUT}/fpnet.pt")
    print("saved", sorted(os.listdir(OUT)), f"{time.time() - T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
