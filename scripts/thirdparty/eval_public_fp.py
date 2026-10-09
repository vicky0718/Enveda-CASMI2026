"""FP alone (f·z over the C2-regime pool window) on panels A / C for the public FPNet checkpoints
(data/external/kaggle_datasets/casmi26-fp-models-v2: fp_single_s2.pt, fp_merged_m1.pt — same architecture as our
casmi.fpmodel.FPNet, 6,930-bit index from coconut-casmi26-candidates/fp_bits.npy over the same 10,407-bit
fingerprint). Checkpoints are loaded tensors-only (torch weights_only=True through casmi.pipeline.load_fp_models).

Caveat: the public checkpoints were probably trained on all training structures, i.e. also on our validation
panels, so these numbers can be inflated; the leaderboard is the honest test.

    python3 scripts/thirdparty/eval_public_fp.py
Compare: our run 3 — A MRR 0.332 / top-1 0.224, C 0.427 / 0.291 (same windows, casmi26-moe-fp-pc logs).
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from casmi import fpmodel as M  # noqa: E402
from casmi import library as L  # noqa: E402
from casmi import pipeline as P  # noqa: E402
from casmi import rank as R  # noqa: E402
from casmi.search import Query  # noqa: E402

EXT = ROOT / "data" / "external" / "kaggle_datasets"


def raw_peaks(rows):
    pf = pq.ParquetFile(ROOT / "data" / "raw" / "train.parquet")
    rows = np.sort(np.unique(rows))
    out, start = {}, 0
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        sel = rows[(rows >= start) & (rows < start + n)] - start
        if len(sel):
            t = pf.read_row_group(rg, columns=["ms2_mzs", "ms2_normalized_intensities"]).take(sel)
            for loc, mz, it in zip(sel, t["ms2_mzs"].to_pylist(), t["ms2_normalized_intensities"].to_pylist()):
                out[start + int(loc)] = (np.asarray(mz, float), np.asarray(it, float))
        start += n
    return out


def main():
    lib = L.load(ROOT / "data" / "artifacts" / "library" / "library.npz")
    qs = pd.read_parquet(ROOT / "data" / "artifacts" / "eval" / "queries.parquet")
    f = pd.read_parquet(ROOT / "data" / "artifacts" / "eval" / "features_hardpc_x.parquet",
                        columns=["qkey", "regime", "panel", "pool_row", "label", "key"])
    f = f[(f.regime == "C2") & (f.pool_row >= 0) & f.panel.isin(["A", "C"])].reset_index(drop=True)
    qs = qs[qs.key.isin(set(f.qkey))]
    raw = raw_peaks(lib["row"][qs.lrow.values])
    libs = list(lib["libs"])
    fp_full = np.load(ROOT / "data" / "artifacts" / "pool" / "fp_full.npy", mmap_mode="r")
    bits = np.load(EXT / "coconut-casmi26-candidates" / "fp_bits.npy", allow_pickle=False)
    tmp = Path(tempfile.mkdtemp())
    for name, fn in (("single", "fp_single_s2.pt"), ("merged", "fp_merged_m1.pt")):
        (tmp / name).mkdir()
        os.symlink(EXT / "casmi26-fp-models-v2" / fn, tmp / name / "fpnet.pt")

    def query(key, g):
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
        q.instr = [M.INSTR_LIST[int(lib["instr"][lr])] for lr in g.lrow.values]
        return q

    for name in ("single", "merged"):
        fpm = P.load_fp_models(str(tmp / name))
        print(name, "nets", len(fpm["nets"]), "nbits", fpm["nets"][0].head[-1].out_features, flush=True)
        fz = np.zeros(len(f), np.float32)
        for key, g in qs.groupby("key", sort=False):
            q = query(key, g)
            if name == "merged":  # merged model: one merged peak list per molecule
                mz, it = M.merge_peaks(q.raw)
                pm = max(q.prec)
                ce = 0.0 if np.isnan(q.ce[0]) else q.ce[0]
                z = M.logits(fpm["nets"], [M.prep_peaks(mz, it, pm)], [pm], [q.adduct[0]], [q.instr[0]], [ce],
                             [1.0 if q.mode[0] > 0 else 0.0])[0]
            else:
                z = P.fp_logits(q, fpm)
            idx = np.flatnonzero(f.qkey.values == key)
            y = np.unpackbits(np.asarray(fp_full[f.pool_row.values[idx]]), axis=1, count=P.FULL_BITS)[:, bits]
            fz[idx] = y.astype(np.float32) @ z.astype(np.float32)
        r = R.mrr_of(f.assign(grp=f.qkey), fz)
        print(name, "MRR", r.groupby("panel").mrr.mean().round(4).to_dict(),
              "top-1", (r.mrr == 1).groupby(r.panel).mean().round(4).to_dict(), flush=True)


if __name__ == "__main__":
    main()
