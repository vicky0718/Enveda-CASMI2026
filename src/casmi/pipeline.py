"""End-to-end inference on a test parquet: molecule queries -> candidates -> channel scores -> ranked
SMILES (25 per molecule, deduplicated by metric key). Used by the harness and the Kaggle notebook.
"""

import os

import numpy as np
import pandas as pd

from .edge import ap_matrix, query_info, spectrum_weights
from .frag import frag_disc as frag_disc_score
from .frag import frag_matrix
from .search import ECFP4_BYTES, Library, Pool, Query, analog_hits, make_query, tanimoto

P_EXP, Q_EXP, TOP_PER_SPEC = 3.0, 1.0, 50
FULL_BITS = 4096 + 4096 + 2048 + 167
GEN_K_REFS = int(os.environ.get("CASMI_GEN_K", 100))  # analog structures used as edit sources
GEN_TWO_K = int(os.environ.get("CASMI_GEN_TWO_K", 0))  # top analogs also tried with two-step edits (0 = off)
SMALL_EDITS = ["+CH2", "-CH2", "+O", "-O", "+H2", "-H2", "+O-H2", "-O+H2", "+H2O", "-H2O"]


def load_fp_models(art, device=None):
    """Our trained FP nets (fpnet*.pt) + the informative-bit index. Returns None if absent."""
    import glob

    import torch

    from . import fpmodel as M
    paths = sorted(glob.glob(f"{art}/fpnet*.pt"))
    if not paths:
        return None
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    nets = []
    for pth in paths:
        ck = torch.load(pth, map_location="cpu", weights_only=True)
        net = M.FPNet(int(ck["nbits"]), d=int(ck["d"]), layers=int(ck["layers"]))
        net.load_state_dict({k: v.float() for k, v in ck["model"].items()})
        nets.append(net.to(device).eval())
    bits = next((b for b in (f"{art}/fp_bits.npy", f"{art}/pool/fp_bits.npy") if os.path.exists(b)), None)
    return {"nets": nets, "bits": None if bits is None else np.load(bits), "device": device}


def fp_logits(q: Query, fpm):
    """Mean logits over a molecule's spectra, plus the merged-spectrum view (two input views)."""
    from . import fpmodel as M
    if fpm is None or not q.raw:
        return None
    peaks = [M.prep_peaks(m, i, pm) for (m, i), pm in zip(q.raw, q.prec)]
    merged = M.prep_peaks(*M.merge_peaks(q.raw), float(np.median(q.prec)))
    ins = ["timsTOF"] * (len(peaks) + 1)
    ces = [0.0 if np.isnan(c) else c for c in q.ce]
    ces = ces + [float(np.mean(ces))]
    modes = [1.0 if m > 0 else 0.0 for m in q.mode]
    modes = modes + [modes[0]]
    adducts = list(q.adduct) + [q.adduct[0]]
    precs = list(q.prec) + [float(np.median(q.prec))]
    z = M.logits(fpm["nets"], peaks + [merged], precs, adducts, ins, ces, modes, device=fpm["device"])
    w = spectrum_weights(q)  # sparse / minor-adduct spectra count less
    return 0.5 * (w[:, None] * z[:-1]).sum(0) + 0.5 * z[-1]


def fp_scores(pool: Pool, cand, z, bits):
    """f·z for each candidate (Bayes log-likelihood up to a constant)."""
    y = np.unpackbits(np.asarray(pool.fp[cand]), axis=1, count=FULL_BITS)[:, bits].astype(np.float32)
    return y @ z.astype(np.float32)


def queries_from_test(test: pd.DataFrame):
    out = []
    for mid, g in test.groupby("molecule_id", sort=False):
        specs = [dict(mz=r.ms2_mzs, it=r.ms2_normalized_intensities, adduct=r.adduct, mode=r.ionization_mode,
                      prec=float(r.precursor_mz),
                      ce=float(np.mean(r.collision_energy_ev)) if r.collision_energy_ev is not None
                      and len(r.collision_energy_ev) else np.nan)
                 for r in g.itertuples()]
        out.append(make_query(mid, specs, enveda=True))
    return out


def generate(q: Query, pool: Pool, hits: pd.DataFrame, cand_keys=(), k_refs=None, ppm=10.0):
    """Class-3 candidates: one-step biosynthetic edits of the top-k analog reference structures whose
    mass differs from the unknown by a known transformation. Returns DataFrame(smiles, key, mass, fp,
    gen_sim, gen_nsrc) without structures already among the pool candidates."""
    from rdkit import Chem
    from rdkit.Chem.Descriptors import ExactMolWt

    from .edits import EDITS, apply_edit, apply_edit_rules, edits_for_delta
    from .fp import full_fp
    M = q.neutral_mass
    k_refs = k_refs or GEN_K_REFS
    h = hits[hits.pool_row.values >= 0].sort_values("sim", ascending=False).drop_duplicates("key").head(k_refs)
    prods = {}
    tol = max(0.005, M * ppm * 1e-6)

    def put(p, sim, steps, rule=-1, delta=0.0):
        g = prods.setdefault(p, [0.0, 0, steps, rule, delta])
        if sim > g[0]:  # rule / delta of the best-matching source
            g[0], g[3], g[4] = float(sim), rule, delta
        g[1] += 1
        g[2] = min(g[2], steps)

    for i, (r, sim) in enumerate(zip(h.pool_row.values, h.sim.values)):
        names = edits_for_delta(M - pool.mass[r], tol=tol)
        for n in names:
            for p, rule in apply_edit_rules(pool.df.smiles.values[r], n).items():
                put(p, sim, 1, rule, abs(EDITS[n][0]))
        if not names and i < GEN_TWO_K:  # a small edit, then any edit closing the remaining gap
            for a in SMALL_EDITS:
                rest = edits_for_delta(M - pool.mass[r] - EDITS[a][0], tol=tol)
                if not rest:
                    continue
                for mid in apply_edit(pool.df.smiles.values[r], a)[:30]:
                    for n in rest:
                        for p in apply_edit(mid, n)[:30]:
                            put(p, sim, 2, -1, abs(EDITS[a][0] + EDITS[n][0]))
    rows, seen = [], set(cand_keys)
    for smi, (sim, nsrc, steps, rule, delta) in prods.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        key = Chem.MolToInchiKey(mol)[:14]
        mass = ExactMolWt(mol)
        if not key or key in seen or abs(mass - M) > max(M * ppm * 1e-6, 0.002):
            continue
        f = full_fp(smi)
        if f is None:
            continue
        seen.add(key)
        rows.append((smi, key, mass, f, sim, nsrc, steps, rule, delta))
    return pd.DataFrame(rows, columns=["smiles", "key", "mass", "fp", "gen_sim", "gen_nsrc", "gen_steps",
                                       "gen_rule", "gen_absdelta"])


def channel_scores(q: Query, pool: Pool, lib: Library, hits: pd.DataFrame, cand: np.ndarray,
                   p_exp=P_EXP, q_exp=Q_EXP, top_per_spec=TOP_PER_SPEC, z=None, bits=None,
                   frag=None, gen=None, excl_rows=None, frag_disc=None, z_prior=None):
    """Per-candidate features for one molecule: pool candidates `cand` followed by generated
    candidates `gen` (from `generate`). `hits` must already exclude any references the evaluation
    regime forbids, and `excl_rows` the same library rows (for the own-spectrum features).
    `frag` covers pool + generated rows in that order."""
    ng = 0 if gen is None else len(gen)
    ckeys = np.concatenate([pool.key[cand], gen.key.values]) if ng else pool.key[cand]
    cfp = np.concatenate([np.asarray(pool.fp[cand]), np.stack(gen.fp.values)]) if ng else np.asarray(pool.fp[cand])
    cmass = np.concatenate([pool.mass[cand], gen.mass.values]) if ng else pool.mass[cand]
    n = len(ckeys)
    hh = hits[hits.pool_row.values >= 0]
    direct = hh[np.abs(hh.delta.values) < 0.01]
    d = direct.groupby("key").sim.max().reindex(ckeys).fillna(0).values
    dn = direct[direct.sim > 0.3].groupby("key").spec.nunique().reindex(ckeys).fillna(0).values
    top = hh.sort_values("sim", ascending=False).groupby("spec").head(top_per_spec)
    if len(top) and n:
        T = tanimoto(np.ascontiguousarray(cfp[:, :ECFP4_BYTES]), pool.ecfp4[top.pool_row.values])
        contrib = (top.sim.values[None, :] ** p_exp) * (T ** q_exp)
        spec = top.spec.values
        per = np.stack([contrib[:, spec == s].max(1) for s in np.unique(spec)], 1)
        a_mean, a_max = per.mean(1), per.max(1)
        tmax = T.max(1)
        # same instrument as the test (timsTOF) references only; and the best few analogs' consensus
        tims = lib.L["instr"][top.ref.values] == 0
        a_tims = np.stack([np.where(tims[spec == s], contrib[:, spec == s], 0).max(1) for s in np.unique(spec)],
                          1).mean(1)
        k = min(5, contrib.shape[1])
        a_top5 = np.sort(contrib, 1)[:, -k:].mean(1)
        # analog evidence from *other* structures only: a candidate's own library spectrum must not
        # vouch for it (isomers with spectra otherwise outrank truths that have none)
        selfm = np.asarray(top.key.values, dtype=object)[None, :] == np.asarray(ckeys, dtype=object)[:, None]
        cns = np.where(selfm, 0.0, contrib)
        a_noself = np.stack([cns[:, spec == s].max(1) for s in np.unique(spec)], 1).mean(1)
        sw = top.sim.values ** p_exp
        t_wmean = (T * sw[None, :]).sum(1) / max(sw.sum(), 1e-9)
        # edge cases: reliability-weighted fusion over spectra; distance-aware (atom-pair) similarity
        specs = np.unique(spec)
        wspec = spectrum_weights(q)[specs.astype(int)] if len(specs) else np.zeros(0)
        wspec = wspec / wspec.sum() if wspec.sum() > 0 else np.full(len(specs), 1.0 / len(specs))
        a_w = per @ wspec
        csmi = np.concatenate([pool.df.smiles.values[cand], gen.smiles.values]) if ng else pool.df.smiles.values[cand]
        Tap = tanimoto(ap_matrix(list(csmi)), ap_matrix(list(pool.df.smiles.values[top.pool_row.values])))
        cap = (top.sim.values[None, :] ** p_exp) * (Tap ** q_exp)
        per_ap = np.stack([cap[:, spec == s].max(1) for s in specs], 1)
        a_ap, a_ap_w, ap_tmax = per_ap.mean(1), per_ap @ wspec, Tap.max(1)
    else:
        a_mean = a_max = tmax = a_tims = a_top5 = t_wmean = a_noself = np.zeros(n)
        a_w = a_ap = a_ap_w = ap_tmax = np.zeros(n)
    own_n, own_sim = own_spectrum_features(q, lib, ckeys, excl_rows)
    f = pd.DataFrame({"pool_row": np.concatenate([cand, np.full(ng, -1)]), "key": ckeys, "direct": d,
                      "direct_n": dn, "analog": a_mean,
                      "analog_max": a_max, "tmax": tmax, "analog_tims": a_tims, "analog_top5": a_top5,
                      "t_wmean": t_wmean, "analog_noself": a_noself, "own_n": own_n, "own_sim": own_sim,
                      "own_neg": (own_n > 0) * (1.0 - own_sim), "mass_err_ppm": (cmass - q.neutral_mass) / q.neutral_mass * 1e6,
                      "is_gen": np.r_[np.zeros(len(cand)), np.ones(ng)],
                      "gen_sim": np.r_[np.zeros(len(cand)), gen.gen_sim.values if ng else []],
                      "gen_nsrc": np.r_[np.zeros(len(cand)), gen.gen_nsrc.values if ng else []],
                      "gen_steps": np.r_[np.zeros(len(cand)), gen.gen_steps.values if ng else []],
                      "gen_rule": np.r_[np.full(len(cand), -1), gen.gen_rule.values if ng else []],
                      "gen_absdelta": np.r_[np.zeros(len(cand)), gen.gen_absdelta.values if ng else []],
                      "analog_w": a_w, "analog_ap": a_ap, "analog_ap_w": a_ap_w, "ap_tmax": ap_tmax})
    for k, v in query_info(q, hh).items():
        f[k] = v
    f["smiles"] = np.concatenate([pool.df.smiles.values[cand], gen.smiles.values]) if ng \
        else pool.df.smiles.values[cand]
    if z is not None and n:
        y = np.unpackbits(cfp, axis=1, count=FULL_BITS)[:, bits].astype(np.float32)
        fz = y @ z.astype(np.float32)
        f["fp"] = fz - fz.max()
        f["fp_rank"] = pd.Series(-fz).rank(method="min").values
        if z_prior is not None:  # what the spectrum adds over the bit-frequency prior
            fn = y @ (z.astype(np.float32) - z_prior)
            f["fp_norm"] = fn - fn.max()
    if frag is not None:
        f["frag"] = frag
    if frag_disc is not None:
        f["frag_disc"] = frag_disc
    if pool.pop is not None:
        add_pop(f, pool.pop)
    return add_relative(f)


def add_pop(f: pd.DataFrame, pop: np.ndarray, group=None):
    """Popularity prior (PubChem substances + PubMed, patents); generated / unknown structures NaN.
    Relative features are taken within the molecule's list (or within `group` columns)."""
    pr = f.pool_row.values
    vals = np.where((pr >= 0)[:, None], pop[np.clip(pr, 0, None)], np.nan)
    f["pop"], f["pop_patents"], f["pop_pubmed"] = vals[:, 0], vals[:, 1], vals[:, 2]
    g = f.groupby(group, sort=False)["pop"] if group else f["pop"]
    mx = g.transform("max") if group else g.max()
    f["pop_gap"] = f["pop"] - mx
    f["pop_rk"] = g.rank(ascending=False, method="min")
    return f


def own_spectrum_features(q: Query, lib: Library, ckeys, excl_rows=None, tol=0.01):
    """For each candidate: how many library spectra it has for the query's adducts, and the best
    unshifted entropy similarity of the query to them. A candidate *with* spectra that do not match
    is evidence against it; one without spectra is merely unknown."""
    from .library import ADDUCT_IX
    n = len(ckeys)
    own_n = np.zeros(n)
    own_sim = np.zeros(n)
    rows, owner = lib.rows_of_keys(ckeys)
    if len(rows) and excl_rows is not None and len(excl_rows):
        keep = ~np.isin(rows, excl_rows)
        rows, owner = rows[keep], owner[keep]
    if not len(rows):
        return own_n, own_sim
    ad = lib.L["adduct_code"][rows]
    qad = {ADDUCT_IX[a] for a in q.adduct if a in ADDUCT_IX}
    use = np.isin(ad, list(qad))
    np.add.at(own_n, owner[use], 1)
    for m, p, a in zip(q.mz, q.p, q.adduct):
        sel = ad == ADDUCT_IX.get(a, -1)
        if not sel.any():
            continue
        sims = lib.sims(m, p, 0.0, rows[sel], tol=tol, shifted=False)
        np.maximum.at(own_sim, owner[sel], sims)
    return own_n, own_sim


REL_COLS = ["direct", "analog", "analog_max", "tmax", "frag", "analog_tims", "analog_top5", "t_wmean",
            "analog_noself", "own_sim", "frag_disc", "analog_w", "analog_ap", "analog_ap_w", "ap_tmax"]


def add_relative(f: pd.DataFrame) -> pd.DataFrame:
    """Within-molecule relative features (gap to the best candidate, rank), shared by training and
    inference."""
    for c in [c for c in REL_COLS if c in f.columns]:
        f[c + "_gap"] = f[c] - f[c].max()
        f[c + "_rk"] = f[c].rank(ascending=False, method="min")
    f["n_cand"] = len(f)
    return f


def heuristic_rank(f: pd.DataFrame) -> np.ndarray:
    score = np.where(f.direct.values > 0.5, 1.0 + f.direct.values, f.analog.values)
    return np.argsort(-score, kind="stable")


def load_ranker(art):
    """LightGBM ranker(s) trained by scripts/eval/train_ranker.py, or None (heuristic ranking)."""
    import glob
    import json
    paths = sorted(glob.glob(f"{art}/ranker*.txt"))
    if not paths:
        return None
    import lightgbm as lgb
    boosters = [lgb.Booster(model_file=p) for p in paths]
    cols = json.load(open(f"{art}/ranker_features.json"))

    from .rank import blend_pop

    def rank(f):
        x = f.reindex(columns=cols).astype(np.float32).values
        s = np.mean([b.predict(x) for b in boosters], 0)
        if "pop" in f.columns:
            s = blend_pop(s, f["pop"].values)
        return np.argsort(-s, kind="stable")
    return rank


def load_keycache(art):
    """smiles -> metric key for pool rows whose key is known (train structures, keyed COCONUT rows)."""
    p = f"{art}/keycache.parquet"
    if not os.path.exists(p):
        return {}
    k = pd.read_parquet(p)
    return dict(zip(k.smiles, k.metric_key))


def run(test: pd.DataFrame, pool: Pool, lib: Library, ranker=None, fp_models=None, use_gen=False, log=print):
    rows = []
    qs = queries_from_test(test)
    for i, q in enumerate(qs):
        try:
            cand = pool.window(q.neutral_mass) if q.mz else np.zeros(0, np.int64)
            if len(cand) == 0:
                rows.append((q.mid, []))
                continue
            h = analog_hits(q, lib, top=600)  # must match the harness the ranker was trained on
            z = fp_logits(q, fp_models)
            gen = generate(q, pool, h, cand_keys=pool.key[cand]) if use_gen else None
            smiles = list(pool.df.smiles.values[cand]) + ([] if gen is None else list(gen.smiles))
            masses = np.r_[pool.mass[cand], [] if gen is None else gen.mass.values]
            FM, pw = frag_matrix(smiles, q)
            f = channel_scores(q, pool, lib, h, cand, z=z, bits=None if fp_models is None else fp_models["bits"],
                               frag=FM @ pw if FM.shape[1] else np.zeros(len(smiles)), gen=gen,
                               frag_disc=frag_disc_score(FM, pw, masses))
            order = ranker(f) if ranker is not None else heuristic_rank(f)
            rows.append((q.mid, list(f.smiles.values[order])))
        except Exception as e:  # one bad molecule must never sink the file
            log(f"  {q.mid}: {type(e).__name__}: {e}")
            rows.append((q.mid, []))
        if i % 50 == 0:
            log(f"  {i}/{len(qs)} molecules")
    return rows
