"""PubChem candidate channel: class-2 answers that are in PubChem but not in our pool (train ∪ COCONUT).

The tier (NCBI PubChem public data, mass-sorted, stereo stripped, CHNOPS + halogens, 150–1250 Da; 106 M
structures) has ~800 structures in a ±10 ppm window, so blind expansion dilutes every list (forum:
0.335 → 0.205). We take only the N most popular structures of the window (popularity = log1p substances +
log1p PubMed, the same prior as the pool) that are not already pool members, and only for molecules the
gate admits; the ranker sees them with `is_pc` and their popularity.
"""

import os

import numpy as np
import pandas as pd


class PubChemTier:
    def __init__(self, d, pop_dir=None):
        """d: the tier arrays (pc_mass / pc_off / pc_smiles); pop_dir: row-aligned pc_lsid / pc_lpmid
        (defaults to d)."""
        pop_dir = pop_dir or d
        self.mass = np.load(f"{d}/pc_mass.npy", mmap_mode="r")
        self.off = np.load(f"{d}/pc_off.npy", mmap_mode="r")
        self.smi = np.load(f"{d}/pc_smiles.npy", mmap_mode="r")
        self.lsid = np.load(f"{pop_dir}/pc_lsid.npy", mmap_mode="r") if os.path.exists(f"{pop_dir}/pc_lsid.npy") \
            else None
        self.lpmid = np.load(f"{pop_dir}/pc_lpmid.npy", mmap_mode="r") if os.path.exists(f"{pop_dir}/pc_lpmid.npy") \
            else None
        self.n = len(self.mass)

    def smiles(self, i):
        a = int(self.off[i])
        b = int(self.off[i + 1]) if i + 1 < len(self.off) else len(self.smi)
        return bytes(self.smi[a:b]).decode("ascii", "ignore").strip("\x00 \n")

    def pop(self, rows):
        if self.lsid is None:
            return np.zeros(len(rows))
        p = np.asarray(self.lsid[rows], np.float32)
        if self.lpmid is not None:
            p = p + np.asarray(self.lpmid[rows], np.float32)
        return p

    def window(self, m, ppm=10.0, min_da=0.002):
        tol = max(m * ppm * 1e-6, min_da)
        a, b = np.searchsorted(self.mass, [m - tol, m + tol])
        return np.arange(a, b)


def pubchem_candidates(q, tier: PubChemTier, pool_keys, top_n=10, ppm=10.0, min_pop=0.0):
    """Top-N most popular PubChem structures in the mass window that are not pool members.
    Returns DataFrame(smiles, key, mass, fp, pc_pop) in the generator's layout (+ is_pc = 1)."""
    from rdkit import Chem
    from rdkit.Chem.Descriptors import ExactMolWt

    from .fp import full_fp
    rows = tier.window(q.neutral_mass, ppm)
    cols = ["smiles", "key", "mass", "fp", "gen_sim", "gen_nsrc", "gen_steps", "gen_rule", "gen_absdelta", "pc_pop",
            "is_pc"]
    if len(rows) == 0:
        return pd.DataFrame(columns=cols)
    pop = tier.pop(rows)
    order = np.argsort(-pop, kind="stable")
    seen = set(pool_keys)
    out = []
    for j in order:
        if len(out) >= top_n or pop[j] < min_pop:
            break
        smi = tier.smiles(int(rows[j]))
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        key = Chem.MolToInchiKey(mol)[:14]
        if not key or key in seen:
            continue
        f = full_fp(smi)
        if f is None:
            continue
        seen.add(key)
        out.append((smi, key, ExactMolWt(mol), f, 0.0, 0, 0, -1, 0.0, float(pop[j]), 1))
    return pd.DataFrame(out, columns=cols)
