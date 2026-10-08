"""Panel N: training natural products that are absent from PubChem and from COCONUT — the closest thing in
our data to the test's class 3 ("novel natural products"). Deleting a known structure from the pool
(regime C3) overstates how often a generator can rebuild it (forum: ~3x), because known structures are
typical; genuinely unpublished ones are not. Class-3 ideas are judged on this panel.

Membership: InChIKey first block of the structure vs every exact-formula isomer in the PubChem tier (tier
masses equal ours to 1e-9). Tautomer copies under another key are missed (~8 % per the forum), so a few
panel members may in fact be in PubChem.

    PYTHONPATH=src python scripts/build/make_novel_panel.py [n_sample]
-> data/artifacts/holdout_novel.parquet (key, smiles, exact_mass, np_likeness, in_pubchem, n_isomers)
"""

import sys
from multiprocessing import Pool as MP

import numpy as np
import pandas as pd

from casmi.paths import INTERIM, ROOT
from casmi.pubchem import PubChemTier

ART = ROOT / "data" / "artifacts"
_T = {}


def _ik14(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(m)[:14] if m is not None else None


def _job(args):
    smi, mass = args
    T = _T["t"]
    a, b = np.searchsorted(T.mass, [mass - 1e-6, mass + 1e-6])
    ik = _ik14(smi)
    for j in range(a, b):
        if _ik14(T.smiles(j)) == ik:
            return True, b - a
    return False, b - a


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    pool = pd.read_parquet(ART / "pool" / "pool.parquet", columns=["key", "smiles", "exact_mass", "src", "coconut_id"])
    tr = pool[(pool.src == "train") & pool.coconut_id.isna()].drop_duplicates("key")
    keys = pd.read_parquet(INTERIM / "metric_keys.parquet").dropna().drop_duplicates("metric_key")
    desc = pd.read_parquet(INTERIM / "structure_descriptors.parquet", columns=["normalized_smiles", "np_likeness"])
    npl = keys.merge(desc.drop_duplicates("normalized_smiles"), on="normalized_smiles")
    npl = npl.drop_duplicates("metric_key").set_index("metric_key").np_likeness
    tr = tr.assign(np_likeness=tr.key.map(npl).values)
    hold = set(pd.read_parquet(ART / "holdout.parquet").key)
    cand = tr[(tr.np_likeness > 1) & ~tr.key.isin(hold)]
    print("NP-like train structures outside COCONUT and the panels:", len(cand), flush=True)
    cand = cand.sample(min(n, len(cand)), random_state=0).reset_index(drop=True)
    _T["t"] = PubChemTier(str(ROOT / "data" / "external" / "pubchem"))
    with MP(4) as mp:
        res = mp.map(_job, list(zip(cand.smiles, cand.exact_mass)), chunksize=16)
    cand["in_pubchem"] = [r[0] for r in res]
    cand["n_isomers"] = [r[1] for r in res]
    print("in PubChem:", round(cand.in_pubchem.mean(), 3), "| median isomers in PubChem:",
          int(cand.n_isomers.median()), flush=True)
    out = cand[["key", "smiles", "exact_mass", "np_likeness", "in_pubchem", "n_isomers"]]
    out.to_parquet(ART / "holdout_novel.parquet")
    print("absent from PubChem and COCONUT:", int((~out.in_pubchem).sum()), "->", ART / "holdout_novel.parquet")


if __name__ == "__main__":
    main()
