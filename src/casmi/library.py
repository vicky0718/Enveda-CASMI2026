"""Cleaned, metric-keyed reference spectral library in CSR form.

Built from the competition train file only (so the same code can run inside the scored notebook):
  * test-type adducts only (formula.ADDUCT_ATOMS), no dimers / exotic adducts;
  * rows whose exact peak list is shared by spectra labelled with different metric keys are dropped
    (EDA pass 5: 1,735 such groups, mostly pluskal_ms2);
  * Enveda positive-mode m/z corrected by -0.4 mDa (EDA pass 5 calibration);
  * peaks prepared with simkernels.prepare (precursor cut, relative floor, top-k, entropy weight).

Arrays (np.savez): off (n+1), mz, p, key_code (into `keys`), adduct_code (into ADDUCTS), mode (+1/-1),
instr (fpmodel.INSTR_LIST code), lib_code (into `libs`), ce (mean eV, nan if unknown), prec_mz, row.
"""

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .formula import ADDUCT_ATOMS
from .fpmodel import instr_family
from .simkernels import prepare

ADDUCTS = list(ADDUCT_ATOMS)
ADDUCT_IX = {a: i for i, a in enumerate(ADDUCTS)}
ENVEDA_LIBS = ("enveda-180", "enveda-np-examples")
ENVEDA_POS_SHIFT = -0.0004  # Da, applied to peaks and precursor of Enveda positive-mode spectra


def conflicting_rows(hashes: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Boolean mask of rows whose identical peak list carries more than one metric key."""
    d = pd.DataFrame({"h": hashes, "k": keys})
    n = d.dropna().groupby("h").k.nunique()
    bad = set(n[n > 1].index)
    return d.h.isin(bad).values


def build(train_path, meta: pd.DataFrame, row_key: np.ndarray, drop: np.ndarray,
          floor=0.002, top_k=64, log=print):
    """meta: train metadata (row-aligned with train_path); row_key: metric key per row (None = skip);
    drop: rows to exclude."""
    use = meta.adduct.isin(ADDUCTS).values & pd.notna(row_key) & ~drop
    keys, key_code = np.unique(np.asarray(row_key[use], dtype=object).astype(str), return_inverse=True)
    libs = sorted(meta.ingest_lib.astype(str).unique())
    lib_ix = {l: i for i, l in enumerate(libs)}
    code_of_row = np.full(len(meta), -1, np.int64)
    code_of_row[np.flatnonzero(use)] = key_code
    lib_all = meta.ingest_lib.astype(str).values
    pos_all = (meta.ionization_mode.astype(str).values == "positive")
    shift_all = np.where(np.isin(lib_all, ENVEDA_LIBS) & pos_all, ENVEDA_POS_SHIFT, 0.0)
    prec_all = meta.precursor_mz.values.astype(np.float64) + shift_all

    mz_parts, p_parts, lens, rows = [], [], [], []
    f = pq.ParquetFile(train_path)
    start = 0
    for rg in range(f.num_row_groups):
        n = f.metadata.row_group(rg).num_rows
        sel = np.flatnonzero(use[start:start + n])
        if len(sel):
            tbl = f.read_row_group(rg, columns=["ms2_mzs", "ms2_normalized_intensities"])
            mzs = tbl["ms2_mzs"].combine_chunks()
            its = tbl["ms2_normalized_intensities"].combine_chunks()
            off = mzs.offsets.to_numpy()
            fm = pc.list_flatten(mzs).to_numpy(zero_copy_only=False)
            fi = pc.list_flatten(its).to_numpy(zero_copy_only=False)
            for loc in sel:
                r = start + loc
                a, b = off[loc], off[loc + 1]
                m, p = prepare(fm[a:b] + shift_all[r], fi[a:b], prec_all[r], floor=floor, top_k=top_k)
                if len(m) == 0:
                    continue
                mz_parts.append(m)
                p_parts.append(p)
                lens.append(len(m))
                rows.append(r)
            del tbl, mzs, its, fm, fi
        start += n
        log(f"  row group {rg + 1}/{f.num_row_groups}: {len(rows):,} spectra kept")
    rows = np.asarray(rows, np.int64)
    sub = meta.iloc[rows]
    ce = sub["ce_mean"].values.astype(np.float32) if "ce_mean" in sub else np.full(len(rows), np.nan, np.float32)
    return dict(
        off=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
        mz=np.concatenate(mz_parts).astype(np.float32),
        p=np.concatenate(p_parts).astype(np.float32),
        key_code=code_of_row[rows].astype(np.int32),
        keys=keys.astype("U14"),
        adduct_code=np.array([ADDUCT_IX[a] for a in sub.adduct.astype(str)], np.int8),
        mode=np.where(pos_all[rows], 1, -1).astype(np.int8),
        instr=np.array([instr_family(s) for s in sub.instrument_type.astype(object).where(
            pd.notna(sub.instrument_type), None)], np.int8),
        lib_code=np.array([lib_ix[l] for l in lib_all[rows]], np.int8),
        libs=np.array(libs, dtype="U32"),
        ce=ce,
        prec_mz=prec_all[rows],
        row=rows,
    )


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}
