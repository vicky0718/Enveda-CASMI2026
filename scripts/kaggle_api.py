"""Minimal Kaggle REST client over www.kaggle.com/api/v1 (api.kaggle.com is blocked here).

    python scripts/kaggle_api.py status <user/slug>
    python scripts/kaggle_api.py output <user/slug> <out_dir>
    python scripts/kaggle_api.py push <kernel-metadata.json>        # private by default
    python scripts/kaggle_api.py dataset-create <dir> <slug> <title> # private dataset from files in <dir>
    python scripts/kaggle_api.py dataset-version <dir> <slug> <notes>
"""

import base64
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://www.kaggle.com/api/v1"


def _auth():
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (user and key):
        c = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
        user, key = c["username"], c["key"]
    return user, {"Authorization": "Basic " + base64.b64encode(f"{user}:{key}".encode()).decode()}


USER, AUTH = _auth()
CTX = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or "/root/.ccr/ca-bundle.crt")


def call(path, body=None, method=None, raw=False, headers=None, timeout=300):
    h = dict(AUTH)
    if body is not None and not raw:
        h["Content-Type"] = "application/json"
        body = json.dumps(body).encode()
    h.update(headers or {})
    req = urllib.request.Request(path if path.startswith("http") else API + path, data=body, headers=h, method=method)
    with urllib.request.urlopen(req, context=CTX, timeout=timeout) as r:
        data = r.read()
        return json.loads(data) if data[:1] in (b"{", b"[") else data


def status(ref):
    u, s = ref.split("/")
    return call(f"/kernels/status?userName={u}&kernelSlug={s}")


def output(ref, out_dir):
    u, s = ref.split("/")
    d = call(f"/kernels/output?userName={u}&kernelSlug={s}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "log.txt").write_text(d.get("log") or "")
    skip = os.environ.get("KAGGLE_OUT_SKIP")  # regex of output files to leave on Kaggle (e.g. large weights)
    for f in d.get("files", []):
        url = f.get("url")
        if skip and re.search(skip, f["fileName"]):
            print("skipped", f["fileName"], file=sys.stderr)
            continue
        if url:
            with urllib.request.urlopen(urllib.request.Request(url), context=CTX, timeout=600) as r:
                (out / f["fileName"]).write_bytes(r.read())
    return d


def push(meta_path):
    meta = json.loads(Path(meta_path).read_text())
    code = (Path(meta_path).parent / meta["code_file"]).read_text()
    body = {"slug": meta["id"], "newTitle": meta["title"], "text": code, "language": meta.get("language", "python"),
            "kernelType": meta.get("kernel_type", "script"), "isPrivate": meta.get("is_private", True),
            "enableGpu": meta.get("enable_gpu", False), "enableInternet": meta.get("enable_internet", False),
            "datasetDataSources": meta.get("dataset_sources", []),
            "competitionDataSources": meta.get("competition_sources", []),
            "kernelDataSources": meta.get("kernel_sources", []), "modelDataSources": meta.get("model_sources", []),
            "categoryIds": []}
    return call("/kernels/push", body)


def _upload_file(path: Path):
    """Two-step blob upload; returns the token to reference the file in dataset create/version."""
    size = path.stat().st_size
    meta = call("/blobs/upload", {"type": "dataset", "name": path.name, "contentLength": size,
                                  "lastModifiedEpochSeconds": int(path.stat().st_mtime)})
    url, token = meta["createUrl"], meta["token"]
    with open(path, "rb") as f:
        req = urllib.request.Request(url, data=f, method="PUT",
                                     headers={"Content-Type": "application/octet-stream", "Content-Length": str(size)})
        with urllib.request.urlopen(req, context=CTX, timeout=3600) as r:
            r.read()
    return token


def dataset_create(folder, slug, title):
    files = [{"token": _upload_file(p)} for p in sorted(Path(folder).iterdir()) if p.is_file()]
    body = {"ownerSlug": USER, "slug": slug, "title": title, "licenseName": "CC0-1.0", "isPrivate": True,
            "files": files, "subtitle": "", "description": "", "categoryIds": []}
    return call("/datasets/create/new", body)


def dataset_version(folder, slug, notes):
    files = [{"token": _upload_file(p)} for p in sorted(Path(folder).iterdir()) if p.is_file()]
    body = {"versionNotes": notes, "files": files, "deleteOldVersions": False, "categoryIds": []}
    return call(f"/datasets/create/version/{USER}/{slug}", body)


def wait(ref, every=60, limit=6 * 3600):
    t0 = time.time()
    while time.time() - t0 < limit:
        s = status(ref)
        if s.get("status") not in ("queued", "running", "QUEUED", "RUNNING"):
            return s
        time.sleep(every)
    return status(ref)


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    fn = {"status": status, "output": output, "push": push, "dataset-create": dataset_create,
          "dataset-version": dataset_version, "wait": wait}[cmd]
    res = fn(*args)
    print(json.dumps(res, indent=1)[:3000] if not isinstance(res, bytes) else res[:500])
