"""Molecular-formula utilities: parsing, adduct -> ion formula, fragment sub-formula enumeration."""

import re
from collections import Counter

import numpy as np

MASS = {  # monoisotopic masses (IUPAC/NIST)
    "H": 1.00782503207, "C": 12.0, "N": 14.0030740048, "O": 15.99491461956, "P": 30.97376163,
    "S": 31.97207100, "F": 18.99840322, "Cl": 34.96885268, "Br": 78.9183371, "I": 126.904473,
    "Na": 22.9897692809, "K": 38.96370668, "Si": 27.9769265325, "B": 11.0093054, "Se": 79.9165213,
}
VALENCE = {"H": 1, "F": 1, "Cl": 1, "Br": 1, "I": 1, "Na": 1, "K": 1, "C": 4, "Si": 4, "N": 3, "P": 3, "B": 3,
           "O": 2, "S": 2, "Se": 2}
ELECTRON = 0.00054857990946

# Adduct -> (atoms added (negative = removed), charge)
ADDUCT_ATOMS = {
    "[M+H]+": ({"H": 1}, 1), "[M+Na]+": ({"Na": 1}, 1), "[M+K]+": ({"K": 1}, 1),
    "[M+NH4]+": ({"N": 1, "H": 4}, 1), "[M-H2O+H]+": ({"H": -1, "O": -1}, 1),
    "[M-2H2O+H]+": ({"H": -3, "O": -2}, 1), "[M-H]-": ({"H": -1}, -1),
    "[M-H2O-H]-": ({"H": -3, "O": -1}, -1), "[M+CH2O2-H]-": ({"C": 1, "H": 1, "O": 2}, -1),
    "[M+Cl]-": ({"Cl": 1}, -1),
}

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse(formula: str) -> Counter:
    if not isinstance(formula, str) or re.search(r"[^A-Za-z0-9]", formula):
        raise ValueError(f"unsupported formula {formula!r}")
    c = Counter()
    for el, n in _TOKEN.findall(formula):
        c[el] += int(n) if n else 1
    if "".join(f"{e}{n}" for e, n in _TOKEN.findall(formula)) != formula:
        raise ValueError(f"unparsed characters in {formula!r}")
    return c


def mass(counts) -> float:
    return float(sum(MASS[e] * n for e, n in counts.items()))


def ion_formula(formula: str, adduct: str):
    """Return (ion element counts, charge) or None if adduct unsupported/impossible."""
    if adduct not in ADDUCT_ATOMS:
        return None
    c = parse(formula)
    add, z = ADDUCT_ATOMS[adduct]
    for e, n in add.items():
        c[e] += n
    if any(v < 0 for v in c.values()):
        return None
    return +c, z


def ion_mz(counts, z: int) -> float:
    return mass(counts) - z * ELECTRON


def subformula_masses(ion_counts, z: int, max_combos: int = 2_000_000):
    """All sub-formulas of an ion formula with RDBE >= -0.5, as (sorted ion m/z, element matrix, elements).

    Fragment ions of a singly-charged ion carry the same charge; H count is free (covers H shifts).
    """
    els = [e for e in ion_counts if e in MASS]
    if len(els) != len(ion_counts):
        return None
    heavy = [e for e in els if e != "H"]
    ranges = [np.arange(ion_counts[e] + 1) for e in heavy]
    n_heavy = int(np.prod([len(r) for r in ranges])) if ranges else 1
    nH = ion_counts.get("H", 0)
    if n_heavy * (nH + 1) > max_combos:
        return None
    grids = np.meshgrid(*ranges, indexing="ij") if ranges else []
    H = np.stack([g.ravel() for g in grids], axis=1) if ranges else np.zeros((1, 0), dtype=int)
    H = H[H.sum(axis=1) > 0]  # at least one heavy atom
    hm = H @ np.array([MASS[e] for e in heavy]) if heavy else np.zeros(len(H))
    # 2 * RDBE = 2 + sum_e n_e (valence_e - 2)  (H and other monovalents contribute -1 each)
    two_rdbe_heavy = 2 + H @ np.array([VALENCE[e] - 2 for e in heavy])
    hs = np.arange(nH + 1)
    two_rdbe = two_rdbe_heavy[:, None] - hs[None, :]          # (n_heavy, nH+1)
    ok = two_rdbe >= -1                                         # RDBE >= -0.5
    hi, hj = np.nonzero(ok)
    m = hm[hi] + hs[hj] * MASS["H"] - z * ELECTRON
    order = np.argsort(m)
    comp = np.concatenate([H[hi], hs[hj][:, None]], axis=1)[order]
    return m[order], comp, heavy + ["H"]


def match_peaks(peak_mz: np.ndarray, sub_mz: np.ndarray, tol_da: float):
    """Nearest sub-formula for each peak: returns (signed error in Da, index) with NaN if none within tol."""
    idx = np.clip(np.searchsorted(sub_mz, peak_mz), 1, len(sub_mz) - 1)
    left, right = sub_mz[idx - 1], sub_mz[idx]
    use_left = np.abs(peak_mz - left) <= np.abs(peak_mz - right)
    best = np.where(use_left, left, right)
    best_i = np.where(use_left, idx - 1, idx)
    err = peak_mz - best
    err = np.where(np.abs(err) <= tol_da, err, np.nan)
    return err, best_i


# --- peak annotation (sub-formula of the precursor ion for every peak) -----------------------------------
ANNOT_ELS = ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I", "Na", "K")
DIMERS = {"[2M+H]+": "[M+H]+", "[2M+Na]+": "[M+Na]+", "[2M+NH4]+": "[M+NH4]+", "[2M+K]+": "[M+K]+",
          "[2M-H]-": "[M-H]-", "[2M+CH2O2-H]-": "[M+CH2O2-H]-"}


def ion_formula_any(formula: str, adduct: str):
    """ion_formula with dimer adducts ([2M+X]) handled by doubling the molecule and intrinsically charged
    molecules ("C12H19N2O2+", adduct [M]+ / [M]-) taken as the ion itself."""
    charge = formula[-1] if formula and formula[-1] in "+-" else ""
    formula = formula.rstrip("+-")
    if adduct in ("[M]+", "[M]-") or charge:
        if adduct not in ("[M]+", "[M]-"):
            return None
        c = parse(formula)
        return +c, (1 if adduct == "[M]+" else -1)
    if adduct in DIMERS:
        c = parse(formula)
        return ion_formula("".join(f"{e}{2 * n}" for e, n in c.items()), DIMERS[adduct])
    return ion_formula(formula, adduct)


def annotate(peak_mz, formula: str, adduct: str, ppm: float = 10.0, min_da: float = 0.002, cache=None):
    """Sub-formula of the precursor ion explaining each peak (nearest within max(ppm, min_da)).
    Returns (frag counts (n, len(ANNOT_ELS)) uint8 — all-zero row = unexplained, precursor ion counts
    (len(ANNOT_ELS),) int32) or None when the formula / adduct cannot be enumerated."""
    key = (formula, adduct)
    if cache is not None and key in cache:
        sub = cache[key]
    else:
        try:
            io = ion_formula_any(formula, adduct) if isinstance(formula, str) else None
        except (ValueError, KeyError):
            io = None
        sub = None
        if io is not None and all(e in ANNOT_ELS for e in io[0]):
            s = subformula_masses(*io)
            if s is not None:
                cols = [ANNOT_ELS.index(e) for e in s[2]]
                comp = np.zeros((len(s[1]), len(ANNOT_ELS)), np.int32)
                comp[:, cols] = s[1]
                pf = np.zeros(len(ANNOT_ELS), np.int32)
                for e, n in io[0].items():
                    pf[ANNOT_ELS.index(e)] = n
                sub = (s[0], comp, pf)
        if cache is not None:
            cache[key] = sub
    if sub is None:
        return None
    mz = np.asarray(peak_mz, float)
    out = np.zeros((len(mz), len(ANNOT_ELS)), np.uint8)
    if len(mz) and len(sub[0]) > 1:
        err, idx = match_peaks(mz, sub[0], 0.05)
        hit = np.abs(err) <= np.maximum(mz * ppm * 1e-6, min_da)
        out[hit] = np.clip(sub[1][idx[hit]], 0, 255)
    return out, sub[2]


if __name__ == "__main__":
    # Hand-checked: caffeine C8H10N4O2, [M+H]+ = 195.08765 ; glucose [M-H]- = 179.05611
    c, z = ion_formula("C8H10N4O2", "[M+H]+")
    assert abs(ion_mz(c, z) - 195.087652) < 2e-6, ion_mz(c, z)
    c, z = ion_formula("C6H12O6", "[M-H]-")
    assert abs(ion_mz(c, z) - 179.056113) < 2e-6, ion_mz(c, z)
    c, z = ion_formula("C15H10O6", "[M+CH2O2-H]-")
    assert abs(ion_mz(c, z) - 331.045941) < 2e-6, ion_mz(c, z)  # luteolin + HCOO-
    sm, comp, els = subformula_masses(*ion_formula("C8H10N4O2", "[M+H]+"))
    # caffeine's classic fragment C6H8N3O+ at m/z 138.0662 must be enumerable
    err, _ = match_peaks(np.array([138.06619]), sm, 0.002)
    assert np.isfinite(err[0]), "missing C6H8N3O+"
    print("formula utils OK;", len(sm), "caffeine sub-formulas")
