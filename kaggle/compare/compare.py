"""Compare two submission files (ours vs the fusion fork) molecule by molecule. Reads only their
outputs (submission.csv), no third-party code. Prints agreement / coverage statistics."""

import glob
import os
import subprocess
import sys

tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
whl = [w for w in glob.glob("/kaggle/input/**/rdkit*.whl", recursive=True) if f"-{tag}-" in w][:1]
if whl:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", *whl], check=False)
import pandas as pd  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem.MolStandardize import rdMolStandardize  # noqa: E402

RDLogger.DisableLog("rdApp.*")
TAUT = rdMolStandardize.TautomerEnumerator()
subs = glob.glob("/kaggle/input/**/submission.csv", recursive=True)
print("submissions", subs)
ours = [s for s in subs if "own-submit" in s][0]
theirs = [s for s in subs if "fusion" in s][0]
ART = os.path.dirname(glob.glob("/kaggle/input/**/casmi26_artifacts.txt", recursive=True)[0])
pool_keys = set(pd.read_parquet(f"{ART}/pool.parquet", columns=["key"]).key)
cache = {}


def key(s):
    if s not in cache:
        m = Chem.MolFromSmiles(s)
        if m is None:
            cache[s] = None
        else:
            Chem.RemoveStereochemistry(m)
            m2 = Chem.MolFromSmiles(Chem.MolToSmiles(m))
            cache[s] = (Chem.MolToInchiKey(TAUT.Canonicalize(m2))[:14] if m2 is not None else None,
                        Chem.MolToInchiKey(m2)[:14] if m2 is not None else None)
    return cache[s]


A = pd.read_csv(ours).set_index("molecule_id").smiles.str.split(";")
B = pd.read_csv(theirs).set_index("molecule_id").smiles.str.split(";")
rows = []
for mid in A.index.intersection(B.index):
    a = [key(s) for s in A[mid]]
    b = [key(s) for s in B[mid]]
    ak = [x[0] for x in a if x]
    bk = [x[0] for x in b if x]
    if not bk:
        continue
    b1 = b[0]
    rows.append({
        "top1_agree": bool(ak) and ak[0] == bk[0],
        "their_top1_in_our25": bk[0] in ak,
        "their_top1_rank_in_ours": (ak.index(bk[0]) + 1) if bk[0] in ak else None,
        "their_top1_in_our_pool": bool(b1) and (b1[0] in pool_keys or b1[1] in pool_keys),
        "our_top1_in_their25": bool(ak) and ak[0] in bk,
        "overlap25": len(set(ak) & set(bk)) / max(1, len(set(bk))),
    })
r = pd.DataFrame(rows)
print("=== COMPARE (molecules:", len(r), ") ===")
print(r.drop(columns=["their_top1_rank_in_ours"]).mean().round(3).to_string())
print("their top-1 rank inside our list (when present):",
      r.their_top1_rank_in_ours.dropna().describe().round(2).to_dict())
print("their top-1 NOT in our pool:", int((~r.their_top1_in_our_pool).sum()))
