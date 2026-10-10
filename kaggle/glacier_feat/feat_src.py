"""GLACIER scores for the evaluation harness (one shard of the validation molecules; CPU).

For every (validation molecule, candidate SMILES) of panels A and C in casmi26-eval-pc's features: GLACIER (official
ms-pred code, MassSpecGym checkpoint; DGL / torch_scatter replaced by our stand-ins) predicts the candidate's spectrum
once per supported adduct of the molecule's query spectra (at that adduct's median collision energy; instrument
family from the library), and the score is the mean cosine with those measured spectra (sqrt intensity, ±0.02 Da).
Also flags whether the molecule's structure is in MassSpecGym (GLACIER's training data) so its value can be judged
on unseen structures — the hidden test's compounds have no public spectra.
Output (/kaggle/working): glacier_shard{SHARD}.parquet (qkey, smiles, gl_cos, gl_n), msg_keys{SHARD}.parquet.
"""

import glob
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

import dgl_shim
import torch_scatter_shim

T0 = time.time()
SHARD, NSHARD = int(os.environ.get("GL_SHARD", 0)), int(os.environ.get("GL_NSHARD", 4))
PANELS = tuple(os.environ.get("GL_PANELS", "A,C").split(","))
INP = "/kaggle/input"


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
whl = [w for w in glob.glob(f"{INP}/**/rdkit*.whl", recursive=True) if f"-{tag}-" in w][:1]
if whl:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", *whl], check=False)
for pkg, mod in (("linsatnet", "LinSATNet"), ("pygmtools", "pygmtools"), ("pytorch-lightning", "pytorch_lightning"),
                 ("omegaconf", "omegaconf"), ("einops", "einops"), ("h5py", "h5py")):
    try:
        __import__(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
dgl_shim.install()
try:
    import torch_scatter  # noqa: F401
except ImportError:
    torch_scatter_shim.install()
with tarfile.open(glob.glob(f"{INP}/**/mspred_src.tar.gz", recursive=True)[0]) as t:
    t.extractall("/tmp/src")
sys.path.insert(0, "/tmp/src/ms-pred/src")
ART = os.path.dirname(glob.glob(f"{INP}/**/casmi26_artifacts.txt", recursive=True)[0])
EV = os.path.dirname(glob.glob(f"{INP}/**/casmi26_eval.txt", recursive=True)[0])
os.makedirs("/kaggle/working/code/casmi", exist_ok=True)
for f in glob.glob(f"{ART}/code__*.py"):
    shutil.copy(f, "/kaggle/working/code/casmi/" + os.path.basename(f)[len("code__"):])
sys.path.insert(0, "/kaggle/working/code")
from casmi import fpmodel as M  # noqa: E402
from casmi import library as L  # noqa: E402
from ms_pred.glacier import dataset, joint_model  # noqa: E402

torch.set_num_threads(os.cpu_count() or 4)
model = joint_model.JointModel.load_from_checkpoint(
    glob.glob(f"{INP}/**/glacier_msg/**/best.ckpt", recursive=True)[0], map_location="cpu").eval()
NB, UL = int(model.num_bins), float(model.upper_limit)
log("GLACIER loaded; shard", SHARD, "of", NSHARD, "panels", PANELS)

# molecules of this shard and their candidates
feats = pd.read_parquet(glob.glob(f"{EV}/features*.parquet")[0], columns=["qkey", "panel", "smiles", "label"])
feats = feats[feats.panel.isin(PANELS)]
keys = pd.unique(feats.qkey)
keys = [k for k in keys if zlib.crc32(str(k).encode()) % NSHARD == SHARD]
feats = feats[feats.qkey.isin(set(keys))]
pairs = feats.drop_duplicates(["qkey", "smiles"])
log("molecules", len(keys), "candidate pairs", len(pairs))

# MassSpecGym membership of the true structures (InChIKey first block)
try:
    tsv = (glob.glob(f"{INP}/**/MassSpecGym*.tsv", recursive=True) or ["/tmp/MassSpecGym1.5.tsv"])[0]
    if not os.path.exists(tsv):  # membership flags only (diagnostics); never used for training or inference
        import urllib.request
        urllib.request.urlretrieve("https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/main/data/"
                                   "MassSpecGym1.5.tsv", tsv)
    msg = pd.read_csv(tsv, sep="\t", usecols=lambda c: c.lower() in ("inchikey", "inchikey14"))
    msg_keys = set(msg.iloc[:, 0].astype(str).str[:14])
    log("MassSpecGym structures", len(msg_keys))
except Exception as e:  # noqa: BLE001
    msg_keys = None
    log("MassSpecGym table unavailable", e)

# query spectra (library rows) of these molecules
lib = L.load(f"{ART}/library.npz")
qs = pd.read_parquet(f"{EV}/queries.parquet")
qs = qs[qs.key.isin(set(keys))]
rows = np.sort(np.unique(lib["row"][qs.lrow.values]))
train = glob.glob(f"{INP}/**/train.parquet", recursive=True)[0]
pf = pq.ParquetFile(train)
raw, start = {}, 0
for rg in range(pf.num_row_groups):
    n = pf.metadata.row_group(rg).num_rows
    sel = rows[(rows >= start) & (rows < start + n)] - start
    if len(sel):
        t = pf.read_row_group(rg, columns=["ms2_mzs", "ms2_normalized_intensities", "inchikey14"]).take(sel)
        for loc, mz, it, ik in zip(sel, t["ms2_mzs"].to_pylist(), t["ms2_normalized_intensities"].to_pylist(),
                                   t["inchikey14"].to_pylist()):
            raw[start + int(loc)] = (np.asarray(mz, float), np.asarray(it, float), ik)
    start += n
libs = list(lib["libs"])
GL_ADDUCTS = {"[M+H]+": "[M+H]+", "[M+Na]+": "[M+Na]+", "[M+K]+": "[M+K]+", "[M-H2O+H]+": "[M-H2O+H]+",
              "[M+NH4]+": "[M+H3N+H]+"}
GL_INSTR = {"timsTOF": "QTOF", "QTOF": "QTOF", "Orbitrap": "Orbitrap", "IT": "IT-FT"}


def vec(mz, it):
    v = np.zeros(NB, np.float32)
    it = np.sqrt(np.clip(it, 0, None))
    ok = (mz > 0) & (mz < UL)
    np.maximum.at(v, (mz[ok] / UL * NB).astype(int).clip(0, NB - 1), it[ok].astype(np.float32))
    return F.max_pool1d(torch.as_tensor(v)[None, None], 5, 1, 2)[0, 0]


spec_of, truth_in_msg = {}, {}
for key, g in qs.groupby("key", sort=False):
    groups = {}
    for lr in g.lrow.values:
        ad = GL_ADDUCTS.get(L.ADDUCTS[lib["adduct_code"][lr]])
        if ad is None or lib["mode"][lr] <= 0:
            continue
        mz, it, ik = raw[int(lib["row"][lr])]
        env = libs[lib["lib_code"][lr]].startswith("enveda")
        ins = GL_INSTR.get(M.INSTR_LIST[int(lib["instr"][lr])], "Unknown")
        groups.setdefault((ad, ins), []).append((vec(mz + (L.ENVEDA_POS_SHIFT if env else 0.0), it),
                                                 float(lib["ce"][lr])))
        if msg_keys is not None:
            truth_in_msg[key] = ik in msg_keys
    spec_of[key] = groups
log("spectra prepared", sum(len(v) for v in spec_of.values()), "adduct/instrument groups")

# predictions: one per (candidate, adduct, instrument group) at the group's median collision energy
recs, meta = [], []
for (key, smi) in zip(pairs.qkey.values, pairs.smiles.values):
    for gi, ((ad, ins), specs) in enumerate(spec_of.get(key, {}).items()):
        ces = [c for _, c in specs if np.isfinite(c)]
        ce = float(np.median(ces)) if ces else 30.0
        name = f"p{len(recs)}"
        recs.append({"spec": name, "smiles": smi, "ionization": ad, "collision_energies": [ce], "instrument": ins})
        meta.append((key, smi, ad, ins))
log("predictions to make", len(recs))
df = pd.DataFrame(recs)
tp = dataset.TreeProcessor(pe_embed_k=model.pe_embed_k, root_encode="graphormer",
                           embed_elem_group=model.embed_elem_group, multi_hop_max_dist=model.multi_hop_max_dist)
score = {}
CH = 4000
for a in range(0, len(df), CH):
    part = df.iloc[a:a + CH]
    try:
        ds = dataset.IntenPredDataset(part, root_encode="graphormer", embed_elem_group=model.embed_elem_group,
                                      tree_processor=tp, num_workers=0)
        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, collate_fn=ds.get_collate_fn())
    except Exception as e:  # noqa: BLE001
        log("chunk featurisation failed", a, type(e).__name__, e)
        continue
    with torch.inference_mode():
        for batch in loader:
            try:
                out = model.predict_inten(batch["graphormer_input"], batch["num_atoms"], batch["adducts"],
                                          batch["collision_engs"], batch["root_form_vecs"], batch["masses"],
                                          batch["adduct_mass_shifts"], batch["atom_form_vecs"], batch["adj_matrices"],
                                          batch["atom_hs"], batch["total_hs"],
                                          batch["instruments"] if model.embed_instrument else None, binned_out=True)
            except Exception as e:  # noqa: BLE001
                log("batch failed", type(e).__name__, str(e)[:200])
                continue
            spec = out["spec"]
            spec = spec.to_dense() if getattr(spec, "is_sparse", False) else spec
            p = F.max_pool1d(torch.sqrt(spec.float().clamp(min=0))[:, None], 5, 1, 2)[:, 0]
            p = p / (p.norm(dim=1, keepdim=True) + 1e-9)
            for name, pv in zip(batch["names"], p):
                i = int(name.split("_collision")[0][1:])
                key, smi, ad, ins = meta[i]
                cs = [float(pv @ v) / float(v.norm() + 1e-9) for v, _ in spec_of[key][(ad, ins)]]
                score.setdefault((key, smi), []).extend(cs)
    log(f"chunk {a // CH + 1}/{(len(df) + CH - 1) // CH} done, scored pairs {len(score)}")

out = pd.DataFrame([(k, s, float(np.mean(v)), len(v)) for (k, s), v in score.items()],
                   columns=["qkey", "smiles", "gl_cos", "gl_n"])
out.to_parquet(f"/kaggle/working/glacier_shard{SHARD}.parquet")
pd.DataFrame({"qkey": list(truth_in_msg), "truth_in_msg": list(truth_in_msg.values())}).to_parquet(
    f"/kaggle/working/msg_keys{SHARD}.parquet")
# GLACIER alone on the C2 lists, by panel and MassSpecGym membership of the truth
f = feats.merge(out, on=["qkey", "smiles"], how="left")
res = []
for key, g in f.groupby("qkey", sort=False):
    if g.label.max() <= 0 or g.gl_cos.isna().all():
        continue
    s = g.gl_cos.fillna(-1).values
    t = s[g.label.values == 1].max()
    res.append((g.panel.iloc[0], truth_in_msg.get(key), 1.0 / (1 + int((s > t).sum())), len(g)))
r = pd.DataFrame(res, columns=["panel", "in_msg", "rr", "n"])
log("GLACIER alone, MRR over each molecule's candidate union:\n", r.groupby(["panel", "in_msg"]).agg(
    mrr=("rr", "mean"), top1=("rr", lambda x: (x == 1).mean()), n=("rr", "size"), cands=("n", "mean")).round(3))
log("done")
