"""Exact local replica of the competition metric (metric/casmi-mean-reciprocal-rank v13).

Key = InChIKey first block of the RDKit 2026.03.3 tautomer-canonical molecule. Requires
rdkit==2026.3.3: TautomerEnumerator output changes between releases.

Two facts from the official code and host answers that the pipeline must respect:
* invalid / unparseable guesses still consume a rank position (they never match);
* answers are stereo-stripped before canonicalisation, so candidates must be too
  (`candidate_key` strips stereo; a stereo-bearing SMILES can get a different key).
"""

import warnings

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
EXPECTED_RDKIT = "2026.03.3"
if rdkit.__version__ != EXPECTED_RDKIT:
    warnings.warn(f"RDKit {rdkit.__version__} != metric's {EXPECTED_RDKIT}; keys may differ")

_TAUT = rdMolStandardize.TautomerEnumerator()


def metric_key(smiles: str):
    """The metric's InChIKey14 for a SMILES exactly as the official scorer computes it."""
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToInchiKey(_TAUT.Canonicalize(mol))[:14]
    except (ValueError, RuntimeError, TypeError):
        return None


def strip_stereo(smiles: str):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def candidate_key(smiles: str):
    """Key for a *candidate*: stereo stripped first, mirroring how answers are prepared."""
    s = strip_stereo(smiles)
    return metric_key(s) if s else None


def mrr_at_k(truth_keys: dict, ranked_keys: dict, k: int = 25) -> float:
    """truth_keys: id -> set of acceptable keys; ranked_keys: id -> list of keys (None allowed,
    still occupies a rank). Missing ids score 0."""
    scores = []
    for mid, acc in truth_keys.items():
        rr = 0.0
        for r, key in enumerate(ranked_keys.get(mid, [])[:k], 1):
            if key is not None and key in acc:
                rr = 1.0 / r
                break
        scores.append(rr)
    return float(np.mean(scores)) if scores else 0.0


def dedup_by_key(smiles_list, keys=None, k: int = 25):
    """Keep the first occurrence of each metric key (a repeated key only wastes a slot)."""
    keys = keys if keys is not None else [candidate_key(s) for s in smiles_list]
    seen, out = set(), []
    for s, key in zip(smiles_list, keys):
        if key is None or key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) == k:
            break
    return out


if __name__ == "__main__":  # reproduce the official doctests
    sol = pd.DataFrame({"molecule_id": ["m_1", "m_2"], "smiles": ["CCO", "c1ccccc1"]})
    sub = {"m_1": ["CCO", "CC(=O)O"], "m_2": ["CC(=O)O", "c1ccccc1"]}
    truth = {r.molecule_id: {metric_key(r.smiles)} for r in sol.itertuples()}
    got = mrr_at_k(truth, {m: [metric_key(s) for s in v] for m, v in sub.items()})
    assert abs(got - 0.75) < 1e-12, got
    assert metric_key("C[C@H](N)C(=O)O") == metric_key("CC(N)C(=O)O")
    print("metric replica OK:", got)
