"""GLACIER spectrum-simulation scores for candidate lists (isomer separation).

GLACIER (Wang, Wang & Coley 2026; official ms-pred code, MIT, github.com/coleygroup/ms-pred) with the official
MassSpecGym checkpoint (host-approved) predicts each candidate's MS/MS spectrum (0.01-Da bins); a candidate is scored
by the mean cosine of its prediction with the query's measured spectra (sqrt intensity, ±0.02 Da tolerance by
max-pooling), one prediction per supported (adduct, instrument family) group at that group's median collision energy.
DGL is not installable on Kaggle's Python image: `casmi.dgl_shim` / `casmi.torch_scatter_shim` stand in (GLACIER's
Graphormer path builds no DGL graphs; see reports/GLACIER_SHIMS.md). Validated in kaggle/glacier_feat (same logic).

    scorer = load(src_tar, ckpt, wheels_dir=None)    # once
    s = scorer(q, smiles_list)                       # (n,) float, NaN where GLACIER cannot score
"""

import glob
import os
import subprocess
import sys
import tarfile

import numpy as np

ADDUCTS = {"[M+H]+": "[M+H]+", "[M+Na]+": "[M+Na]+", "[M+K]+": "[M+K]+", "[M-H2O+H]+": "[M-H2O+H]+",
           "[M+NH4]+": "[M+H3N+H]+"}
INSTR = {"timsTOF": "QTOF", "QTOF": "QTOF", "Orbitrap": "Orbitrap", "IT": "IT-FT"}


def _install(wheels_dir):
    """Offline install of the pure-Python dependencies (wheels built by kaggle/glacier_env) if missing."""
    need = []
    for mod in ("LinSATNet", "pygmtools", "pytorch_lightning", "omegaconf", "einops", "h5py"):
        try:
            __import__(mod)
        except ImportError:
            need.append(mod)
    if need and wheels_dir and os.path.isdir(wheels_dir):
        whl = sorted(glob.glob(f"{wheels_dir}/*.whl"))
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", *whl], check=False)
    return need


def load(src_tar, ckpt, wheels_dir=None, work="/tmp/mspred", device=None, batch_size=32):
    import torch
    import torch.nn.functional as F

    from . import dgl_shim, torch_scatter_shim
    _install(wheels_dir)
    dgl_shim.install()
    try:
        import torch_scatter  # noqa: F401
    except ImportError:
        torch_scatter_shim.install()
    if not os.path.isdir(f"{work}/ms-pred/src"):
        os.makedirs(work, exist_ok=True)
        with tarfile.open(src_tar) as t:
            t.extractall(work)
    sys.path.insert(0, f"{work}/ms-pred/src")
    import ms_pred.common as common
    from ms_pred.glacier import dataset, joint_model
    from rdkit import Chem

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = joint_model.JointModel.load_from_checkpoint(ckpt, map_location="cpu").eval().to(device)
    nb, ul = int(model.num_bins), float(model.upper_limit)
    tp = dataset.TreeProcessor(pe_embed_k=model.pe_embed_k, root_encode="graphormer",
                               embed_elem_group=model.embed_elem_group, multi_hop_max_dist=model.multi_hop_max_dist)
    ok_cache = {}

    def supported(smi):
        if smi not in ok_cache:
            m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and "." not in smi else None
            ok_cache[smi] = m is not None and all(a.GetSymbol() in common.ELEMENT_TO_MASS for a in m.GetAtoms())
        return ok_cache[smi]

    def vec(mz, it):
        v = np.zeros(nb, np.float32)
        mz, it = np.asarray(mz, float), np.sqrt(np.clip(np.asarray(it, float), 0, None))
        ok = (mz > 0) & (mz < ul)
        np.maximum.at(v, (mz[ok] / ul * nb).astype(int).clip(0, nb - 1), it[ok].astype(np.float32))
        t = F.max_pool1d(torch.as_tensor(v)[None, None], 5, 1, 2)[0, 0]
        return t / (t.norm() + 1e-9)

    def score(q, smiles):
        """Mean cosine per candidate over the query's spectra GLACIER supports; NaN if none / unsupported."""
        import pandas as pd
        out = np.full(len(smiles), np.nan)
        groups = {}
        instr = getattr(q, "instr", None) or ["timsTOF"] * len(q.raw)
        for (mz, it), ad, mode, ce, ins in zip(q.raw, q.adduct, q.mode, q.ce, instr):
            a = ADDUCTS.get(ad)
            if a is None or mode <= 0:
                continue
            groups.setdefault((a, INSTR.get(ins, "Unknown")), []).append((vec(mz, it), ce))
        if not groups:
            return out
        idx = [i for i, s in enumerate(smiles) if supported(s)]
        if not idx:
            return out
        recs, meta = [], []
        for (a, ins), specs in groups.items():
            ces = [c for _, c in specs if c is not None and np.isfinite(c)]
            ce = float(np.median(ces)) if ces else 30.0
            for i in idx:
                recs.append({"spec": f"p{len(recs)}", "smiles": smiles[i], "ionization": a,
                             "collision_energies": [ce], "instrument": ins})
                meta.append((i, (a, ins)))
        acc = {}
        try:
            ds = dataset.IntenPredDataset(pd.DataFrame(recs), root_encode="graphormer",
                                          embed_elem_group=model.embed_elem_group, tree_processor=tp, num_workers=0)
            loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                                 collate_fn=ds.get_collate_fn())
            with torch.inference_mode():
                for batch in loader:
                    for k in batch:
                        if isinstance(batch[k], torch.Tensor):
                            batch[k] = batch[k].to(device)
                        if k == "graphormer_input" and batch[k] is not None:
                            for gk in batch[k]:
                                batch[k][gk] = batch[k][gk].to(device)
                    pred = model.predict_inten(batch["graphormer_input"], batch["num_atoms"], batch["adducts"],
                                               batch["collision_engs"], batch["root_form_vecs"], batch["masses"],
                                               batch["adduct_mass_shifts"], batch["atom_form_vecs"],
                                               batch["adj_matrices"], batch["atom_hs"], batch["total_hs"],
                                               batch["instruments"] if model.embed_instrument else None,
                                               binned_out=True)["spec"]
                    pred = pred.to_dense() if getattr(pred, "is_sparse", False) else pred
                    p = F.max_pool1d(torch.sqrt(pred.float().clamp(min=0))[:, None], 5, 1, 2)[:, 0].cpu()
                    p = p / (p.norm(dim=1, keepdim=True) + 1e-9)
                    for name, pv in zip(batch["names"], p):
                        i, g = meta[int(name.split("_collision")[0][1:])]
                        acc.setdefault(i, []).extend(float(pv @ v) for v, _ in groups[g])
        except Exception as e:  # noqa: BLE001 — a failed molecule keeps NaN scores (feature missing)
            print(f"  GLACIER failed for {getattr(q, 'mid', '?')}: {type(e).__name__}: {e}", flush=True)
        for i, v in acc.items():
            out[i] = float(np.mean(v))
        return out

    return score
