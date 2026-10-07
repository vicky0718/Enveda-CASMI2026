"""Workarounds for the edge cases where spectrum→structure scoring fails (see reports/eval/EDGE_CASES.md).

1. Fingerprint-blind isomers (a third of class-2 truths have a same-formula isomer with ECFP Tanimoto
   >= 0.9): atom-pair fingerprints encode the topological distance between every atom pair, so moving a
   methoxy or a sugar to another position changes them even when ECFP environments barely change.
   -> analog propagation and best-analog similarity recomputed with atom-pair Tanimoto.
2. Minor adducts / sparse spectra: a 3-peak [M+Na]+ spectrum should not weigh as much as a rich [M+H]+
   one. -> per-spectrum reliability weights (adduct x peak count) for molecule-level fusion.
3. Low-information molecules: per-molecule descriptors (peaks, spectral entropy, best library match)
   that the ranker can interact with — e.g. trust priors more when the spectra say little.
4. FP-model generic-structure bias: f·z grows with the number of set bits that the model calls likely
   a priori; f·(z − z_prior) scores only what the spectrum adds over the bit-frequency prior.
"""

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
AP_BITS = 2048
ADDUCT_RELIABILITY = {"[M+H]+": 1.0, "[M-H]-": 1.0, "[M+CH2O2-H]-": 0.8, "[M+NH4]+": 0.6, "[M+Cl]-": 0.6,
                      "[M+Na]+": 0.4, "[M+K]+": 0.4, "[M-H2O+H]+": 0.6, "[M-2H2O+H]+": 0.5,
                      "[M-H2O-H]-": 0.6}
_AP = {}
_GEN = []


def _ap_gen():
    if not _GEN:
        _GEN.append(rdFingerprintGenerator.GetAtomPairGenerator(fpSize=AP_BITS))
    return _GEN[0]


def ap_fp(smiles):
    """Packed atom-pair fingerprint (memoised; zeros for unparsable SMILES)."""
    v = _AP.get(smiles)
    if v is None:
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        a = np.zeros(AP_BITS, np.uint8)
        if mol is not None:
            DataStructs.ConvertToNumpyArray(_ap_gen().GetFingerprint(mol), a)
        v = np.packbits(a)
        if len(_AP) > 300_000:
            _AP.clear()
        _AP[smiles] = v
    return v


def ap_matrix(smiles_list):
    return np.stack([ap_fp(s) for s in smiles_list]) if len(smiles_list) else np.zeros((0, AP_BITS // 8), np.uint8)


def spectrum_weights(q):
    """Reliability of each query spectrum for molecule-level fusion: adduct prior x peak richness."""
    w = []
    for m, a in zip(q.mz, q.adduct):
        w.append(ADDUCT_RELIABILITY.get(a, 0.5) * min(1.0, len(m) / 8.0))
    w = np.asarray(w, float)
    return w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / max(len(w), 1))


def query_info(q, hits):
    """Per-molecule descriptors broadcast to every candidate row."""
    ent = []
    for p in q.p:
        p = np.asarray(p, float)
        p = p[p > 0]
        ent.append(float(-(p * np.log(p)).sum()) if len(p) else 0.0)
    best = float(hits.sim.max()) if len(hits) else 0.0
    best_direct = float(hits.sim[np.abs(hits.delta.values) < 0.01].max()) if len(hits) and \
        (np.abs(hits.delta.values) < 0.01).any() else 0.0
    return {"q_nspec": len(q.mz), "q_npeaks": float(np.mean([len(m) for m in q.mz])) if q.mz else 0.0,
            "q_entropy": float(np.mean(ent)) if ent else 0.0, "q_best_hit": best, "q_best_direct": best_direct,
            "q_pos": float(np.mean([m > 0 for m in q.mode])) if q.mode else 0.0,
            "q_reliable": float(any(a in ("[M+H]+", "[M-H]-") for a in q.adduct)),
            "q_mass": float(q.neutral_mass)}


def fp_prior_logits(bit_freq):
    p = np.clip(np.asarray(bit_freq, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).astype(np.float32)
