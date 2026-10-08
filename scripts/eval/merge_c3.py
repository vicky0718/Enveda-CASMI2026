"""Class-3 merge prototype: let generated structures in only when the answer is probably not in our lists.

Three models, all out of fold (molecules held out), on the generator-era harness features (C1/C2/C3):
  pool ranker  — pool rows, trained on C1/C2 lists (the truth is in the pool)
  gen ranker   — generated rows, trained on C3 lists (the truth may have been generated)
  gate         — list-level P(truth not in the pool) from the two rankers' score profiles + query info
Merge:  pool row  log(1-g) + log softmax(s_pool / T1)      gen row  log g + log softmax(s_gen / T2)
compared with the pool-only ranker (class 3 = 0) and the V7 single mixed ranker.

    PYTHONPATH=src python scripts/eval/merge_c3.py
"""

import numpy as np
import pandas as pd

from casmi import rank as R
from casmi.paths import ROOT

EVAL = ROOT / "data" / "artifacts" / "eval"
MIX = {"C1": 0.16, "C2": 0.45, "C3": 0.39}
Q = ["q_nspec", "q_npeaks", "q_entropy", "q_best_hit", "q_best_direct", "q_pos", "q_reliable", "q_mass"]


def oof(f, mask_rows, mask_train, feats, rounds=300):
    out = np.full(len(f), np.nan)
    for k in range(5):
        tr = f[mask_rows & mask_train & (f.fold != k) & f.panel.isin(["A", "C"])]
        va = f[mask_rows & (f.fold == k)]
        b = R._train(tr, feats) if rounds == 300 else None
        out[va.index] = b.predict(va[feats].astype(np.float32))
    return out


def lse(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


def main():
    f = R.prepare(pd.read_parquet(EVAL / "features.parquet"), ("C1", "C2", "C3"))
    feats = R.feature_cols(f)
    pool, gen = (f.is_gen == 0).values, (f.is_gen == 1).values
    s_pool = oof(f, pool, f.regime.isin(["C1", "C2"]).values, feats)
    s_gen = oof(f, gen, (f.regime == "C3").values, feats)
    s_mix = oof(f, np.ones(len(f), bool), np.ones(len(f), bool), feats)  # V7-style single ranker
    f["s_pool"], f["s_gen"] = s_pool, s_gen

    # list-level gate features
    rows = []
    for g_id, g in f.groupby("grp", sort=False):
        p, q = g[g.is_gen == 0], g[g.is_gen == 1]
        sp = np.sort(p.s_pool.values)[::-1] if len(p) else np.zeros(1)
        sg = np.sort(q.s_gen.values)[::-1] if len(q) else np.full(1, -10.0)
        r = {"grp": g_id, "fold": g.fold.iloc[0], "panel": g.panel.iloc[0], "regime": g.regime.iloc[0],
             "y": int(g.regime.iloc[0] == "C3"), "p1": sp[0], "p_marg": sp[0] - (sp[1] if len(sp) > 1 else sp[0] - 1),
             "p_std": sp.std(), "n_pool": len(p), "g1": sg[0], "n_gen": len(q),
             "g_marg": sg[0] - (sg[1] if len(sg) > 1 else sg[0] - 1),
             "dmax": p.direct.max() if len(p) else 0, "amax": p.analog.max() if len(p) else 0,
             "gsim": q.gen_sim.max() if len(q) else 0}
        r.update({c: g[c].iloc[0] for c in Q if c in g})
        rows.append(r)
    L = pd.DataFrame(rows)
    gcols = [c for c in L.columns if c not in ("grp", "fold", "panel", "regime", "y")]
    import lightgbm as lgb
    gate = np.zeros(len(L))
    for k in range(5):
        tr = L[(L.fold != k) & L.panel.isin(["A", "C"])]
        w = tr.regime.map({"C1": MIX["C1"] / 1, "C2": MIX["C2"], "C3": MIX["C3"]}).values
        b = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 30,
                       "verbose": -1, "seed": 0}, lgb.Dataset(tr[gcols].astype(np.float32), tr.y, weight=w), 300)
        va = L.fold == k
        gate[va.values] = b.predict(L.loc[va, gcols].astype(np.float32))
    L["gate"] = gate
    from sklearn.metrics import roc_auc_score
    for p in "ABC":
        m = L.panel == p
        print(f"gate AUC panel {p}: {roc_auc_score(L.y[m], L.gate[m]):.3f}")
    f = f.merge(L[["grp", "gate"]], on="grp", how="left")

    def merged(T1, T2, a=1.0):
        out = np.empty(len(f))
        for _, g in f.groupby("grp", sort=False):
            gi = np.clip(g.gate.values[0] ** a, 1e-6, 1 - 1e-6)
            ip, ig = g.index[g.is_gen.values == 0], g.index[g.is_gen.values == 1]
            if len(ip):
                x = f.s_pool.values[ip] / T1
                out[ip] = np.log(1 - gi) + x - lse(x)
            if len(ig):
                x = f.s_gen.values[ig] / T2
                out[ig] = np.log(gi) + x - lse(x)
        return out

    def score_tab(s):
        r = R.mrr_of(f, s)
        t = r.groupby(["panel", "regime"]).mrr.mean()
        w = {p: round(sum(MIX[k] * t.get((p, k), 0) for k in MIX), 4) for p in "AC"}
        return t, w

    res = {}
    only_pool = np.where(pool, f.s_pool.values, -1e9)
    res["pool only (no generator)"] = score_tab(only_pool)
    res["V7 single mixed ranker"] = score_tab(s_mix)
    best = None
    for T1 in (0.5, 1.0, 2.0):
        for T2 in (0.5, 1.0, 2.0):
            for a in (0.5, 1.0, 2.0):
                t, w = score_tab(merged(T1, T2, a))
                if best is None or w["C"] > best[0]["C"]:
                    best = (w, (T1, T2, a), t)
    res[f"gated merge T1,T2,a={best[1]}"] = (best[2], best[0])
    pd.set_option("display.width", 220)
    print(pd.DataFrame({k: v[0] for k, v in res.items()}).T.round(4).to_string())
    for k, v in res.items():
        print(f"{k:45s} weighted (0.16/0.45/0.39): {v[1]}")


if __name__ == "__main__":
    main()
