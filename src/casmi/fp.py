"""Molecular fingerprints for candidate ranking.

Full vector = ECFP4 (radius 2, 4096) ‖ ECFP6 (radius 3, 4096) ‖ RDKit path FP (2048) ‖ MACCS (167)
= 10,407 bits, as described in the public analog-propagation notebook (prvsiyan). A subset of
informative bits (frequency in [0.5 %, 99.5 %] over training structures) is selected by `select_bits`.
"""

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
FULL_BITS = 4096 + 4096 + 2048 + 167
_GEN = {}


def _gens():
    if not _GEN:
        _GEN["e4"] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
        _GEN["e6"] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
        _GEN["rd"] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)
    return _GEN


def full_fp(smiles: str):
    """Packed full fingerprint (uint8, FULL_BITS bits) or None."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    g = _gens()
    parts = []
    for fp in (g["e4"].GetFingerprint(mol), g["e6"].GetFingerprint(mol), g["rd"].GetFingerprint(mol),
               MACCSkeys.GenMACCSKeys(mol)):
        a = np.zeros(fp.GetNumBits(), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, a)
        parts.append(a)
    return np.packbits(np.concatenate(parts))


def select_bits(packed: np.ndarray, lo: float = 0.005, hi: float = 0.995) -> np.ndarray:
    freq = np.unpackbits(packed, axis=1, count=FULL_BITS).mean(axis=0)
    return np.flatnonzero((freq >= lo) & (freq <= hi))


def subset(packed: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """Unpacked 0/1 matrix restricted to `bits`."""
    return np.unpackbits(packed, axis=1, count=FULL_BITS)[:, bits]
