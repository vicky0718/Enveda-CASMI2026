"""Download the mass-sorted PubChem tier (NCBI PubChem public data; ahmedberatozer/casmi26-pubchem-tier) and the
row-aligned popularity arrays (dmitriigluzdov/casmi26-pubchem-popularity-prior) -> data/external/pubchem/.
Plain NumPy arrays only (loaded with allow_pickle=False)."""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kaggle_api as K  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "data" / "external" / "pubchem"
FILES = [("ahmedberatozer/casmi26-pubchem-tier", f) for f in ("pc_mass.npy", "pc_off.npy", "pc_smiles.npy")] + \
        [("dmitriigluzdov/casmi26-pubchem-popularity-prior", f) for f in ("pc_lsid.npy", "pc_lpmid.npy")]


def fetch(ref, name):
    dst = OUT / name
    if dst.exists() and dst.stat().st_size > 0:
        return
    req = urllib.request.Request(f"{K.API}/datasets/download/{ref}/{name}", headers=K.AUTH)
    with urllib.request.urlopen(req, context=K.CTX, timeout=7200) as r, open(str(dst) + ".part", "wb") as f:
        while True:
            b = r.read(1 << 24)
            if not b:
                break
            f.write(b)
    Path(str(dst) + ".part").rename(dst)
    print(name, dst.stat().st_size, flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for ref, name in FILES:
        fetch(ref, name)
