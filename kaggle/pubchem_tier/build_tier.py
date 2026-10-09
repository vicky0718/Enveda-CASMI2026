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
    """Stream a gzipped tab-separated file (no header; columns by position f0, f1, ...) as record batches."""
    opts = pcsv.ReadOptions(column_names=[f"f{i}" for i in range(ncols)], block_size=1 << 26)
    parse = pcsv.ParseOptions(delimiter="\t", quote_char=False, invalid_row_handler=lambda row: "skip")
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

    # 3. popularity (log1p counts of substances and PubMed references)
    sid_c = counts("CID-SID.gz", maxcid)
    pmid_c = counts("CID-PMID.gz", maxcid)

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
