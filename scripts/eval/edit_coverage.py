"""Class-3 oracle coverage of one-step biosynthetic edits from analog hits (regime C3: the truth is
neither in the library nor in the pool). For each validation molecule: take the top-K reference
structures by (shifted) entropy similarity, keep those whose mass differs from the unknown by a
known edit, enumerate products, and check whether the truth's InChIKey14 is among them.

    PYTHONPATH=src python scripts/eval/edit_coverage.py [K]
"""

import sys
import time
from multiprocessing import Pool as MP

import pandas as pd
from rdkit import Chem, RDLogger

from casmi.edits import apply_edit, edits_for_delta
from casmi.paths import INTERIM, ROOT

RDLogger.DisableLog("rdApp.*")
EVAL = ROOT / "data" / "artifacts" / "eval"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 20
TWO = "--two" in sys.argv
SMALL = ["+CH2", "-CH2", "+O", "-O", "+H2", "-H2", "+O-H2", "-O+H2", "+H2O", "-H2O"]


def ik14(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToInchiKey(m)[:14] if m is not None else None


def job(args):
    key, truth_ik, M, refs = args
    prods = set()
    used = 0
    from casmi.edits import EDITS
    for smi, mass in refs:
        names = edits_for_delta(M - mass)
        if names:
            used += 1
            for n in names:
                prods.update(apply_edit(smi, n))
        elif TWO:
            for a in SMALL:  # first a small edit, then any edit that closes the remaining gap
                rest = edits_for_delta(M - mass - EDITS[a][0])
                if not rest:
                    continue
                used += 1
                mids = apply_edit(smi, a)[:30]
                for mid in mids:
                    for n in rest:
                        prods.update(apply_edit(mid, n)[:30])
    iks = {ik14(p) for p in prods}
    return {"key": key, "refs_with_edit": used, "n_products": len(prods), "covered": truth_ik in iks}


def main():
    pool = pd.read_parquet(ROOT / "data" / "artifacts" / "pool" / "pool.parquet", columns=["key", "smiles", "exact_mass"])
    qs = pd.read_parquet(EVAL / "queries.parquet")
    H = pd.read_parquet(EVAL / "hits.parquet")
    keys = pd.read_parquet(INTERIM / "metric_keys.parquet").dropna().drop_duplicates("metric_key").set_index("metric_key")
    panel = qs.drop_duplicates("key").set_index("key").panel
    mass_of = pool.set_index("key").exact_mass
    jobs = []
    for key, h in H.groupby("qkey", sort=False):
        if panel[key] == "B":
            continue
        h = h[(h.key != key) & (h.pool_row >= 0)].sort_values("sim", ascending=False).drop_duplicates("key").head(K)
        refs = list(zip(pool.smiles.values[h.pool_row.values], pool.exact_mass.values[h.pool_row.values]))
        jobs.append((key, keys.inchikey14.get(key, key), float(mass_of[key]), refs))
    t0 = time.time()
    with MP(4) as mp:
        res = pd.DataFrame(mp.map(job, jobs, chunksize=8))
    res["panel"] = res.key.map(panel)
    print(f"K={K}, {time.time() - t0:.0f}s")
    print(res.groupby("panel").agg(covered=("covered", "mean"), any_edit_ref=("refs_with_edit", lambda x: (x > 0).mean()),
                                   products_median=("n_products", "median"), products_p90=("n_products", lambda x: x.quantile(.9))).round(3))
    res.to_parquet(EVAL / f"edit_coverage_K{K}.parquet")


if __name__ == "__main__":
    main()
