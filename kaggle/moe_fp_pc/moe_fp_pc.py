"""V13 ranker: PubChem candidates scored with our FP model (Kaggle GPU).

The harness features (casmi26-eval-pc) carry the 100 most popular PubChem structures of each molecule's
mass window in every regime (C1, C2, C2H) plus C2P, where the truth is only in PubChem. Here f·z is
added for every row (PubChem rows' fingerprints recomputed from SMILES) and the all-evidence ranker is
validated out of fold, molecules held out, twice on the same folds:
    without PubChem rows (today's lists)  vs  with them (V13)
so the cost on molecules already in the pool (C1/C2/C2H) is read against the gain on C2P.
Output (/kaggle/working): moe_full.txt + moe.json (full expert only), fpnet.pt, casmi26_models.txt.
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


def relative(f):
    """Within-list relative features (pipeline.add_relative), vectorised over groups."""
    g = f.groupby("grp", sort=False)
    for c in [c for c in P.REL_COLS if c in f.columns]:
        f[c + "_gap"] = f[c] - g[c].transform("max")
        f[c + "_rk"] = g[c].rank(ascending=False, method="min")
    f["n_cand"] = g["key"].transform("size")
    f["fp"] = f.fz - g.fz.transform("max")
    f["fp_rank"] = g.fz.rank(ascending=False, method="min")
    if "fzn" in f.columns:
        f["fp_norm"] = f.fzn - g.fzn.transform("max")
    if "pop" in f.columns:
        f["pop_gap"] = f["pop"] - g["pop"].transform("max")
    return f


def main():
    lib = L.load(f"{ART}/library.npz")
    qs = pd.read_parquet(f"{EV}/queries.parquet")
    feats_df = pd.read_parquet(f"{EV}/features.parquet")
    feats_df = feats_df[feats_df.is_gen == 0].reset_index(drop=True)  # PubChem rows: is_gen 0, is_pc 1
    print("rows", len(feats_df), "PubChem rows", int(feats_df.is_pc.sum()),
          feats_df.groupby("regime").qkey.nunique().to_dict(), flush=True)
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

    # candidate fingerprints: pool rows from the packed matrix, PubChem rows recomputed from SMILES
    gen = feats_df.pool_row.values < 0
    gsmi = pd.unique(feats_df.smiles.values[gen])
    gfp = {s: full_fp(s) for s in gsmi}
    print("PubChem fingerprints", len(gfp), f"{time.time() - T0:.0f}s", flush=True)
    zp = np.load(f"{ART}/fp_prior.npy") if os.path.exists(f"{ART}/fp_prior.npy") else None
    fz = np.zeros(len(feats_df), np.float32)
    fzn = np.zeros(len(feats_df), np.float32)
    for (key, _), g in feats_df.groupby(["qkey", "regime"], sort=False):
        z = z_of[key].astype(np.float32)
        pr = g.pool_row.values
        fps = np.stack([np.asarray(fp_full[r]) if r >= 0 else gfp[s] for r, s in zip(pr, g.smiles.values)])
        y = np.unpackbits(fps, axis=1, count=P.FULL_BITS)[:, bits].astype(np.float32)
        fz[g.index.values] = y @ z
        if zp is not None:
            fzn[g.index.values] = y @ (z - zp)
    feats_df["fz"] = fz
    if zp is not None:
        feats_df["fzn"] = fzn
    print("f·z done", f"{time.time() - T0:.0f}s", flush=True)

    def blended(f, s):
        out = np.empty(len(f))
        for _, g in f.groupby("grp", sort=False):
            out[g.index] = R.blend_pop(s[g.index], g["pop"].values) if "pop" in g else s[g.index]
        return out

    from casmi import moe
    regimes = ("C1", "C2", "C2H", "C2P")
    f_pc = relative(R.prepare(feats_df, regimes=regimes))
    folds = f_pc.drop_duplicates("qkey").set_index("qkey").fold  # same molecule folds for both variants
    f_no = feats_df[feats_df.is_pc == 0].copy()
    f_no = relative(R.prepare(f_no, regimes=regimes))
    f_no["fold"] = f_no.qkey.map(folds).values
    cols = moe.expert_features(f_pc, "full")
    print("features", len(cols), flush=True)
    tabs = {}
    for name, f in (("FP alone (f·z), no PubChem", f_no), ("FP alone (f·z), with PubChem", f_pc)):
        rep = R.mrr_of(f, f.fz.values)
        tabs[name] = rep.groupby(["panel", "regime"]).mrr.mean()
        # top-1 among the window's candidates (forum's FPNet: 0.46-0.49 on np-examples)
        tabs[name + " top1"] = (rep.mrr == 1).groupby([rep.panel, rep.regime]).mean()
    for name, f in (("without PubChem rows", f_no), ("with PubChem rows (V13)", f_pc)):
        oof = np.zeros(len(f))
        for k in range(5):
            tr = f[(f.fold != k) & f.panel.isin(["A", "C"])]
            va = f[f.fold == k]
            oof[va.index] = moe._fit(tr, cols, moe.CONFIG["level1_rounds"]).predict(va[cols].astype(np.float32))
        rep = R.mrr_of(f, blended(f, oof))
        tabs[name] = rep.groupby(["panel", "regime"]).mrr.mean()
        # rows: where the truth's best PubChem / pool competitor sits; is_pc share of first places
        top = f.assign(s=blended(f, oof)).sort_values(["grp", "s"], ascending=[True, False]).groupby("grp").head(1)
        print(name, "| share of lists topped by a PubChem row:",
              top.groupby(["panel", "regime"]).is_pc.mean().round(3).to_dict(), flush=True)
        print(name, f"{time.time() - T0:.0f}s", flush=True)
    tab = pd.DataFrame(tabs).T
    pd.set_option("display.width", 220)
    print("=== V13 VALIDATION (out of fold, molecules held out, FP incl., MRR@25) ===")
    print(tab.round(4).to_string())
    # C2P counts as 0 without PubChem rows (the truth is unreachable); the class-2 split between
    # in-pool and PubChem-only molecules is unknown, so report a range of PubChem-only shares
    for s2p in (0.1, 0.2, 0.3):
        w = {n: {p: round(0.16 * tab.loc[n].get((p, "C1"), 0) + (0.45 - s2p) / 2 * (tab.loc[n].get((p, "C2"), 0)
                                                                                   + tab.loc[n].get((p, "C2H"), 0))
                          + s2p * tab.loc[n].get((p, "C2P"), 0), 4) for p in "AC"} for n in tab.index}
        print(f"weighted, PubChem-only share of test {s2p}:", w, flush=True)
    moe.fit_save(f_pc, OUT, names=["full"], stack=False, seeds=3)
    open(f"{OUT}/casmi26_models.txt", "w").write("V13: full ranker with FP, trained with PubChem rows\n")
    shutil.copy(FPN[0], f"{OUT}/fpnet.pt")
    print("saved", sorted(os.listdir(OUT)), f"{time.time() - T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
