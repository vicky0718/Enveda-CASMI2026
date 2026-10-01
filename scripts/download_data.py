"""Download the Enveda CASMI 2026 competition files into data/raw/.

Credentials are read from KAGGLE_USERNAME / KAGGLE_KEY, falling back to
~/.kaggle/kaggle.json. Uses the www.kaggle.com/api/v1 endpoints rather than the
`kaggle` CLI, because the CLI (v2+) talks to api.kaggle.com, which some
sandboxed environments block.

You must accept the competition rules on kaggle.com before downloads work.

    python scripts/download_data.py                 # all files
    python scripts/download_data.py test.parquet    # specific files
"""

import base64
import json
import os
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
API = "https://www.kaggle.com/api/v1"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _auth_header() -> dict:
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (user and key):
        cfg = Path.home() / ".kaggle" / "kaggle.json"
        if not cfg.exists():
            sys.exit("No Kaggle credentials: set KAGGLE_USERNAME/KAGGLE_KEY or create ~/.kaggle/kaggle.json")
        creds = json.loads(cfg.read_text())
        user, key = creds["username"], creds["key"]
    token = base64.b64encode(f"{user}:{key}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _ssl_context() -> ssl.SSLContext:
    bundle = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    return ssl.create_default_context(cafile=bundle) if bundle else ssl.create_default_context()


def _open(url: str, headers: dict, ctx: ssl.SSLContext):
    try:
        return urllib.request.urlopen(urllib.request.Request(url, headers=headers), context=ctx)
    except urllib.error.HTTPError as e:
        body = e.read()[:300].decode(errors="replace")
        if e.code == 403 and "accept" in body.lower():
            sys.exit(f"403: accept the rules first at https://www.kaggle.com/competitions/{COMPETITION}/rules")
        raise


def list_files(headers: dict, ctx: ssl.SSLContext) -> dict:
    with _open(f"{API}/competitions/data/list/{COMPETITION}", headers, ctx) as r:
        files = json.load(r)
    files = files.get("files", files) if isinstance(files, dict) else files
    return {f["name"]: int(f.get("totalBytes") or 0) for f in files}


def download(name: str, expected: int, headers: dict, ctx: ssl.SSLContext, retries: int = 4) -> None:
    dest = RAW_DIR / name
    if dest.exists() and (not expected or dest.stat().st_size == expected):
        print(f"[skip] {name} already present ({dest.stat().st_size:,} bytes)")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries + 1):
        try:
            with _open(f"{API}/competitions/data/download/{COMPETITION}/{name}", headers, ctx) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done, t0, last = 0, time.time(), 0.0
                while chunk := r.read(8 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if time.time() - last > 10:
                        last = time.time()
                        rate = done / max(last - t0, 1e-6) / 1e6
                        pct = f"{100 * done / total:5.1f}%" if total else ""
                        print(f"  {name}: {done / 1e6:,.0f} MB {pct} @ {rate:.1f} MB/s", flush=True)
            break
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            if attempt == retries:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  retry {attempt + 1}/{retries} in {wait}s: {e}")
            time.sleep(wait)

    # Large files are sometimes served zipped; unwrap transparently.
    if zipfile.is_zipfile(tmp):
        with zipfile.ZipFile(tmp) as z:
            member = next(m for m in z.namelist() if m.endswith(name) or len(z.namelist()) == 1)
            with z.open(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
        tmp.unlink()
    else:
        tmp.replace(dest)
    print(f"[done] {name} -> {dest} ({dest.stat().st_size:,} bytes)")


def main(argv: list[str]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    headers, ctx = _auth_header(), _ssl_context()
    available = list_files(headers, ctx)
    wanted = argv or sorted(available, key=available.get)  # smallest first
    for name in wanted:
        if name not in available:
            sys.exit(f"Unknown file {name!r}; available: {sorted(available)}")
        download(name, available[name], headers, ctx)


if __name__ == "__main__":
    main(sys.argv[1:])
