"""Biosynthetic one-step edits of a reference structure (class-3 candidate generation).

Given an analog reference structure whose neutral mass differs from the unknown by a known
transformation delta, enumerate every product of that transformation at every applicable site.
Forward and reverse reactions are both listed; products are stereo-free canonical SMILES.
"""

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# name: (delta mass added to the reference, [reaction SMARTS])
EDITS = {
    "+CH2": (14.015650, ["[OX2H1:1]>>[O:1]C", "[NX3;H1,H2;!$(NC=O):1]>>[N:1]C", "[c;H1:1]>>[c:1]C",
                         "[C;H3:1]>>[C:1]C"]),
    "-CH2": (-14.015650, ["[O:1][CH3]>>[OH1:1]", "[N:1][CH3]>>[N:1]", "[c:1][CH3]>>[cH1:1]"]),
    "+O": (15.994915, ["[c;H1:1]>>[c:1]O", "[C;X4;!H0:1]>>[C:1]O", "[N;X3;H0;!$(N=*);!$(N-[!#6]):1]>>[N+:1][O-]"]),
    "-O": (-15.994915, ["[c:1][OH1]>>[cH1:1]", "[C;X4:1][OH1]>>[C:1]"]),
    "+H2": (2.015650, ["[C:1]=[C:2]>>[C:1][C:2]", "[C:1]=[O:2]>>[C:1][O:2]"]),
    "-H2": (-2.015650, ["[C;!H0:1]-[C;!H0:2]>>[C:1]=[C:2]", "[C;!H0:1][OH1:2]>>[C:1]=[O:2]"]),
    "+hexose": (162.052824, ["[OX2H1:1]>>[O:1]C1OC(CO)C(O)C(O)C1O"]),
    "-hexose": (-162.052824, ["[O:1]C1OC(CO)C(O)C(O)C1O>>[OH1:1]"]),
    "+deoxyhexose": (146.057909, ["[OX2H1:1]>>[O:1]C1OC(C)C(O)C(O)C1O"]),
    "-deoxyhexose": (-146.057909, ["[O:1]C1OC(C)C(O)C(O)C1O>>[OH1:1]"]),
    "+pentose": (132.042259, ["[OX2H1:1]>>[O:1]C1OCC(O)C(O)C1O"]),
    "+glucuronide": (176.032088, ["[OX2H1:1]>>[O:1]C1OC(C(=O)O)C(O)C(O)C1O"]),
    "+acetyl": (42.010565, ["[OX2H1:1]>>[O:1]C(C)=O", "[NX3;H1,H2;!$(NC=O):1]>>[N:1]C(C)=O"]),
    "-acetyl": (-42.010565, ["[O:1]C(=O)[CH3]>>[OH1:1]"]),
    "+H2O": (18.010565, ["[C:1]=[C:2]>>[C:1][C:2]O"]),
    "-H2O": (-18.010565, ["[C;!H0:1][C:2][OH1]>>[C:1]=[C:2]"]),
    "+CO2": (43.989829, ["[c;H1:1]>>[c:1]C(=O)O"]),
    "-CO2": (-43.989829, ["[c:1]C(=O)[OH1]>>[cH1:1]"]),
    "+SO3": (79.956815, ["[OX2H1:1]>>[O:1]S(=O)(=O)O"]),
    "+prenyl": (68.062600, ["[c;H1:1]>>[c:1]CC=C(C)C", "[OX2H1:1]>>[O:1]CC=C(C)C"]),
    "+malonyl": (86.000394, ["[OX2H1:1]>>[O:1]C(=O)CC(=O)O"]),
    "+OCH2": (30.010565, ["[c;H1:1]>>[c:1]OC"]),           # methoxylation (+O +CH2)
    "-OCH2": (-30.010565, ["[c:1]O[CH3]>>[cH1:1]"]),
}
_RXN = {}


def _rxns(name):
    if name not in _RXN:
        _RXN[name] = [AllChem.ReactionFromSmarts(s) for s in EDITS[name][1]]
    return _RXN[name]


def edits_for_delta(delta, tol=0.005):
    return [n for n, (m, _) in EDITS.items() if abs(m - delta) <= tol]


def apply_edit(smiles, name, max_products=200):
    """All single-site products of edit `name` on `smiles` (canonical, stereo-free, unique)."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return []
    Chem.RemoveStereochemistry(mol)
    out = set()
    for rxn in _rxns(name):
        try:
            prods = rxn.RunReactants((mol,), maxProducts=max_products)
        except Exception:
            continue
        for ps in prods:
            p = ps[0]
            try:
                Chem.SanitizeMol(p)
                out.add(Chem.MolToSmiles(p))
            except Exception:
                continue
    return sorted(out)
