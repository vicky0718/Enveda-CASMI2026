"""Memory-conscious readers for the competition parquet files.

The train peak arrays are ~11 GB uncompressed, so peak-level work streams one
row group at a time and returns flat numpy arrays plus CSR-style offsets.
"""

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .paths import INTERIM, TEST, TRAIN

PEAK_COLS = ["ms2_mzs", "ms2_normalized_intensities"]
META_CACHE = INTERIM / "train_meta.parquet"


def load_meta(path=TRAIN, cache: bool = True) -> pd.DataFrame:
    """All non-peak columns, with collision_energy_ev summarised to scalars."""
    if cache and path == TRAIN and META_CACHE.exists():
        return pd.read_parquet(META_CACHE)
    schema = pq.read_schema(path)
    cols = [c for c in schema.names if c not in PEAK_COLS]
    tbl = pq.read_table(path, columns=cols)
    ce = tbl["collision_energy_ev"]
    df = tbl.drop(["collision_energy_ev"]).to_pandas()
    df["ce_n"] = pc.list_value_length(ce).fill_null(0).to_numpy()
    df["ce_min"] = _ce_reduce(ce, "min")
    df["ce_max"] = _ce_reduce(ce, "max")
    df["ce_mean"] = _ce_reduce(ce, "mean")
    for c in df.columns:
        if df[c].dtype == object and df[c].nunique(dropna=False) < 5000:
            df[c] = df[c].astype("category")
    if "num_peaks" not in df.columns:  # test file lacks it
        df["num_peaks"] = pc.list_value_length(pq.read_table(path, columns=["ms2_mzs"])["ms2_mzs"]).to_numpy()
    if cache and path == TRAIN:
        df.to_parquet(META_CACHE)
    return df


def _ce_reduce(ce: pa.ChunkedArray, how: str) -> np.ndarray:
    out = []
    for chunk in ce.chunks:
        n = len(chunk)
        lengths = pc.list_value_length(chunk).fill_null(0).to_numpy()
        vals = pc.list_flatten(chunk).to_numpy(zero_copy_only=False).astype(float)
        res = np.full(n, np.nan)
        nz = lengths > 0
        if nz.any():
            starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])[nz]
            if how == "min":
                res[nz] = np.minimum.reduceat(vals, starts)
            elif how == "max":
                res[nz] = np.maximum.reduceat(vals, starts)
            else:
                res[nz] = np.add.reduceat(vals, starts) / lengths[nz]
        out.append(res)
    return np.concatenate(out) if out else np.array([])


@dataclass
class PeakBatch:
    """Peaks of a block of spectra in CSR layout."""

    row_offset: int          # index of the first spectrum in the full file
    offsets: np.ndarray      # len n_spectra + 1
    mz: np.ndarray
    inten: np.ndarray
    extra: pd.DataFrame      # requested per-spectrum columns

    @property
    def n(self) -> int:
        return len(self.offsets) - 1

    @property
    def spec_idx(self) -> np.ndarray:
        """Spectrum index (local) of every peak."""
        return np.repeat(np.arange(self.n), np.diff(self.offsets))


def iter_peaks(path=TRAIN, columns=("precursor_mz",)) -> Iterator[PeakBatch]:
    f = pq.ParquetFile(path)
    start = 0
    for rg in range(f.num_row_groups):
        tbl = f.read_row_group(rg, columns=PEAK_COLS + list(columns))
        mzs = tbl["ms2_mzs"].combine_chunks()
        ints = tbl["ms2_normalized_intensities"].combine_chunks()
        mzs = mzs.fill_null([]) if mzs.null_count else mzs
        ints = ints.fill_null([]) if ints.null_count else ints
        lengths = pc.list_value_length(mzs).to_numpy()
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        batch = PeakBatch(
            row_offset=start,
            offsets=offsets,
            mz=pc.list_flatten(mzs).to_numpy(zero_copy_only=False),
            inten=pc.list_flatten(ints).to_numpy(zero_copy_only=False),
            extra=tbl.select(list(columns)).to_pandas(),
        )
        start += batch.n
        yield batch


def load_test() -> pd.DataFrame:
    return pd.read_parquet(TEST)
