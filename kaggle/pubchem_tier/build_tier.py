"""Build our own mass-sorted PubChem candidate tier from NCBI PubChem FTP files (Kaggle CPU, internet on).

Replaces the third-party Kaggle datasets we used as plain data (licence field "Other"), so the final
solution depends only on NCBI's public files. Output (/kaggle/working), the format casmi.pubchem reads:
    pc_mass.npy    float64 (n,)   monoisotopic mass, ascending
    pc_off.npy     int64 (n+1,)   offsets into pc_smiles
    pc_smiles.npy  uint8          concatenated stereo-stripped SMILES (ASCII)
    pc_lsid.npy    float16 (n,)   log1p(number of substances, CID-SID)
    pc_lpmid.npy   float16 (n,)   log1p(number of PubMed references, CID-PMID)
Filters: elements C H N O P S F Cl Br I only, 150 <= mass <= 1250 Da (as the tier it replaces). Stereo is
stripped textually (@, /, \\); one row per CID like the tier it replaces (stereo variants that collapse to the
same structure are de-duplicated by InChIKey when candidates are drawn, casmi.pubchem.pubchem_candidates).
"""

import gzip
import os
import tempfile
import time
import urllib.request

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv

T0 = time.time()
BASE = os.environ.get("TIER_BASE", "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/")
TMP = os.path.join(tempfile.gettempdir(), "pctier")
OUT = os.environ.get("TIER_OUT", "/kaggle/working")
os.makedirs(TMP, exist_ok=True)
ALLOWED = r"^(?:(?:C|H|N|O|P|S|F|Cl|Br|I)\d*)+$"


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def download(name, tries=4):
    dst = f"{TMP}/{name}"
    for k in range(tries):
        if os.path.exists(dst):
            break
        try:
            with urllib.request.urlopen(BASE + name, timeout=600) as r, open(dst + ".part", "wb") as f:
                while True:
                    b = r.read(1 << 24)
                    if not b:
                        break
                    f.write(b)
            os.rename(dst + ".part", dst)
        except Exception as e:  # noqa: BLE001
            log("download retry", name, k, type(e).__name__, e)
            time.sleep(30 * (k + 1))
    log(name, f"{os.path.getsize(dst) / 1e9:.2f} GB")
    return dst


def batches(path, ncols, types, include):
    """Stream a gzipped tab-separated file (no header) as record batches; columns by position f0, f1, ... The
    column count is taken from the file itself (CID-SID / CID-PMID carry a third column), `ncols` is only the
    minimum we need."""
    opts = pcsv.ReadOptions(autogenerate_column_names=True, block_size=1 << 26)
    parse = pcsv.ParseOptions(delimiter="\t", quote_char=False)
    conv = pcsv.ConvertOptions(column_types=types, include_columns=include)
    with gzip.open(path, "rb") as fh:
        for b in pcsv.open_csv(fh, read_options=opts, parse_options=parse, convert_options=conv):
            yield b


def counts(name, maxcid):
    """Rows per CID in a CID-X file (CID-SID / CID-PMID)."""
    c = np.zeros(maxcid + 1, np.int32)
    try:
        path = download(name)
    except Exception as e:  # noqa: BLE001 — popularity is optional
        log("skip", name, type(e).__name__, e)
        return c
    for b in batches(path, 2, {"f0": pa.int64()}, ["f0"]):
        cid = b.column("f0").to_numpy()
        cid = cid[(cid >= 0) & (cid <= maxcid)]
        c += np.bincount(cid, minlength=maxcid + 1).astype(np.int32)
    os.remove(path)
    log(name, "rows counted", int(c.sum()))
    return c


def connectivity_groups(name, maxcid):
    """Per CID: integer id of its InChIKey first block (-1 if unknown). CID-InChI-Key: CID, InChI, InChIKey.
    Keys are encoded as base-26 integers of their first 13 letters (vectorised; a Python dict of ~1e8 strings
    would not fit in memory) and grouped with np.unique."""
    try:
        path = download(name)
    except Exception as e:  # noqa: BLE001
        log("skip", name, type(e).__name__, e)
        return None
    pw = (26 ** np.arange(12, -1, -1)).astype(np.int64)
    cids, codes = [], []
    for b in batches(path, 3, {"f0": pa.int64(), "f2": pa.string()}, ["f0", "f2"]):
        cid = b.column("f0").to_numpy()
        key = b.column("f2")
        ok = (cid <= maxcid) & (pc.utf8_length(key).to_numpy(zero_copy_only=False) == 27)
        k13 = pc.cast(pc.utf8_slice_codeunits(pc.filter(key, pa.array(ok)), 0, 13), pa.binary())
        buf = np.frombuffer(k13.buffers()[2], np.uint8)
        off = np.frombuffer(k13.buffers()[1], np.int32)[k13.offset:k13.offset + len(k13) + 1]
        mat = buf[off[0]:off[-1]].reshape(-1, 13).astype(np.int64) - 65
        cids.append(cid[ok])
        codes.append(mat @ pw)
    os.remove(path)
    cids, codes = np.concatenate(cids), np.concatenate(codes)
    _, inv = np.unique(codes, return_inverse=True)
    grp = np.full(maxcid + 1, -1, np.int64)
    grp[cids] = inv
    log(name, "groups", int(inv.max()) + 1)
    return grp


def main():
    # 1. masses and formulas -> the CIDs we keep (CID-Mass: CID, formula, monoisotopic mass, exact mass)
    mass_path = download("CID-Mass.gz")
    cids, masses = [], []
    for b in batches(mass_path, 4, {"f0": pa.int64(), "f1": pa.string(), "f2": pa.float64()}, ["f0", "f1", "f2"]):
        m = b.column("f2").to_numpy(zero_copy_only=False)
        ok = pc.match_substring_regex(b.column("f1"), ALLOWED).fill_null(False).to_numpy(zero_copy_only=False)
        ok &= (m >= 150.0) & (m <= 1250.0)
        cids.append(b.column("f0").to_numpy()[ok])
        masses.append(m[ok])
    os.remove(mass_path)
    cids, masses = np.concatenate(cids), np.concatenate(masses)
    maxcid = int(cids.max())
    mass_of = np.full(maxcid + 1, np.nan)
    mass_of[cids] = masses
    del cids, masses
    log("kept by mass/elements:", int((~np.isnan(mass_of)).sum()))

    # 2. SMILES of the kept CIDs, stereo stripped
    smi_path = download("CID-SMILES.gz")
    out_cid, out_smi = [], []
    for b in batches(smi_path, 2, {"f0": pa.int64(), "f1": pa.large_string()}, ["f0", "f1"]):
        cid = b.column("f0").to_numpy()
        keep = cid <= maxcid
        keep[keep] = ~np.isnan(mass_of[cid[keep]])
        if not keep.any():
            continue
        s = pc.filter(b.column("f1"), pa.array(keep))
        out_cid.append(cid[keep])
        out_smi.append(pc.replace_substring_regex(s, pattern=r"[@/\\]", replacement=""))
    os.remove(smi_path)
    cid = np.concatenate(out_cid)
    smi = pa.chunked_array(out_smi).combine_chunks()
    del out_smi
    log("SMILES rows:", len(cid))

    # 3. popularity (log1p counts of substances and PubMed references), summed over all CIDs sharing the InChIKey
    #    first block (stereo variants / tautomer records of one structure): on the validation panels this recovers
    #    the truth coverage of the third-party arrays (panel C top-100 0.653 vs 0.657; per-CID counts 0.613)
    sid_c = counts("CID-SID.gz", maxcid).astype(np.int64)
    pmid_c = counts("CID-PMID.gz", maxcid).astype(np.int64)
    grp = connectivity_groups("CID-InChI-Key.gz", maxcid)
    if grp is not None:
        ok = grp >= 0
        ng = int(grp.max()) + 1
        gs = np.bincount(grp[ok], weights=sid_c[ok], minlength=ng)
        gp = np.bincount(grp[ok], weights=pmid_c[ok], minlength=ng)
        sid_c = np.where(ok, gs[np.maximum(grp, 0)], sid_c)
        pmid_c = np.where(ok, gp[np.maximum(grp, 0)], pmid_c)
        log("popularity aggregated over", ng, "InChIKey first blocks")

    # 4. mass order and the arrays
    mass = mass_of[cid]
    order = np.argsort(mass, kind="stable")
    smi = pc.cast(smi.take(pa.array(order)), pa.large_string())
    cid, mass = cid[order], mass[order]
    offs = np.frombuffer(smi.buffers()[1], np.int64)[smi.offset:smi.offset + len(smi) + 1]
    data = np.frombuffer(smi.buffers()[2], np.uint8)[offs[0]:offs[-1]]
    np.save(f"{OUT}/pc_mass.npy", mass.astype(np.float64))
    np.save(f"{OUT}/pc_off.npy", (offs - offs[0]).astype(np.int64))
    np.save(f"{OUT}/pc_smiles.npy", data)
    np.save(f"{OUT}/pc_lsid.npy", np.log1p(sid_c[cid]).astype(np.float16))
    np.save(f"{OUT}/pc_lpmid.npy", np.log1p(pmid_c[cid]).astype(np.float16))
    np.save(f"{OUT}/pc_cid.npy", cid.astype(np.int64))  # provenance: PubChem CID of every row
    with open(f"{OUT}/casmi26_pubchem_tier.txt", "w") as fh:
        fh.write(f"NCBI PubChem tier: {len(cid)} structures, built {time.strftime('%Y-%m-%d')} from {BASE}\n")
    log("written", sorted(os.listdir(OUT)), f"{data.nbytes / 1e9:.2f} GB SMILES")


if __name__ == "__main__":
    main()
