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
        # architecture options travel with the checkpoint (run 4+: PairBias, run 6+: formula inputs)
        net = M.FPNet(int(ck["nbits"]), d=int(ck["d"]), layers=int(ck["layers"]), rel=bool(ck.get("rel", False)),
                      formula=bool(ck.get("formula", False)))
        # the sinusoidal m/z tables (SinEmb.inv) are deterministic and must stay float32: checkpoints are
        # saved in fp16, whose 3 significant digits turn a 628 rad/Da frequency into >100 rad of phase
        # error at m/z 500 — scrambling every high-resolution feature the model learned
        sd = {k: v.float() for k, v in ck["model"].items() if not k.endswith(".inv")}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        bad = [k for k in missing if not k.endswith(".inv") and not k.endswith("fscale")] + list(unexpected)
        if bad:  # only the deterministic tables may be absent; anything else is an architecture mismatch
            raise RuntimeError(f"{pth}: checkpoint / architecture mismatch {bad[:5]}")
        nets.append(net.to(device).eval())
    bits = next((b for b in (f"{art}/fp_bits.npy", f"{art}/pool/fp_bits.npy") if os.path.exists(b)), None)
    out = {"nets": nets, "bits": None if bits is None else np.load(bits), "device": device}
    if os.path.exists(f"{art}/fp_calib.npz"):  # per-bit logistic calibration z' = a·z + c (fit on held-out truths)
        c = np.load(f"{art}/fp_calib.npz")
        out["calib"] = (c["a"].astype(np.float32), c["c"].astype(np.float32))
    return out


def calibrate_logits(z, fpm):
    if z is None or fpm is None or "calib" not in fpm:
        return z
    a, c = fpm["calib"]
    if isinstance(z, dict):
        return {k: a * v + c for k, v in z.items()}
    return a * z + c


def formula_nets(fpm) -> bool:
    return fpm is not None and any(getattr(n, "formula", False) for n in fpm["nets"])


def fp_logits(q: Query, fpm, formulas=None):
    """Reliability-weighted mean of per-spectrum logits. The merged-spectrum view is NOT used: the model
    was trained on single spectra and a merged peak list is out of distribution (held-out within-formula
    MRR on public NPs 0.42 single vs 0.31 with the merged view averaged in).

    Formula models (run 6+) read each peak's sub-formula of a candidate formula: with `formulas` given the
    result is {formula: logits} (one forward pass per formula and spectrum, batched); without, only the
    formula-free nets are used."""
    from . import fpmodel as M
    from .formula import annotate
    if fpm is None or not q.raw:
        return None
    peaks = [M.prep_peaks(m, i, pm) for (m, i), pm in zip(q.raw, q.prec)]
    ces = [0.0 if np.isnan(c) else c for c in q.ce]
    modes = [1.0 if m > 0 else 0.0 for m in q.mode]
    instr = getattr(q, "instr", None) or ["timsTOF"] * len(peaks)  # test: all timsTOF; validation sets it
    w = spectrum_weights(q)  # sparse / minor-adduct spectra count less
    if not formula_nets(fpm) or formulas is None:
        nets = [n for n in fpm["nets"] if not getattr(n, "formula", False)]
        if not nets:
            return None
        z = M.logits(nets, peaks, list(q.prec), list(q.adduct), instr, ces, modes, device=fpm["device"])
        return calibrate_logits((w[:, None] * z).sum(0), fpm)
    formulas = list(dict.fromkeys(formulas))
    S = len(peaks)
    frags, pforms = [], []
    for fo in formulas:
        cache = {}  # sub-formula tables can be large: one formula at a time
        for (mz, _), ad in zip(peaks, q.adduct):
            an = annotate(mz, fo, ad, cache=cache) if len(mz) else None
            frags.append(an[0] if an is not None else np.zeros((len(mz), M.N_ELS), np.uint8))
            pforms.append(an[1] if an is not None else np.zeros(M.N_ELS, np.int32))
    out = {}
    B = 256
    rep = lambda x: [v for _ in formulas for v in x]  # noqa: E731
    allz = []
    P_, PR, AD, IN, CE, MO = rep(peaks), rep(list(q.prec)), rep(list(q.adduct)), rep(instr), rep(ces), rep(modes)
    for a in range(0, len(P_), B):
        allz.append(M.logits(fpm["nets"], P_[a:a + B], PR[a:a + B], AD[a:a + B], IN[a:a + B], CE[a:a + B],
                             MO[a:a + B], device=fpm["device"], frags=frags[a:a + B], pforms=pforms[a:a + B]))
    allz = np.concatenate(allz)
    for i, fo in enumerate(formulas):
        out[fo] = (w[:, None] * allz[i * S:(i + 1) * S]).sum(0)
    return calibrate_logits(out, fpm)


def candidate_formulas(pool: Pool, cand, gen=None):
    """Molecular formula of every candidate row (pool rows from the pool table, others from SMILES)."""
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    f = list(pool.df.formula.values[cand]) if "formula" in pool.df else [None] * len(cand)
    if gen is not None and len(gen):
        for smi in gen.smiles.values:
            m = Chem.MolFromSmiles(smi)
            f.append(CalcMolFormula(m) if m is not None else None)
    return f


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
    is_pc = np.r_[np.zeros(len(cand)), gen.is_pc.values.astype(float) if ng and "is_pc" in gen.columns
                  else np.zeros(ng)]
    f["is_pc"] = is_pc
    f["is_gen"] = np.r_[np.zeros(len(cand)), np.ones(ng)] * (1 - is_pc)
    for k, v in query_info(q, hh).items():
        f[k] = v
    f["smiles"] = np.concatenate([pool.df.smiles.values[cand], gen.smiles.values]) if ng \
        else pool.df.smiles.values[cand]
    if isinstance(z, dict) and n:  # formula model: every candidate scored under its own formula
        forms = candidate_formulas(pool, cand, gen if ng else None)
        y = np.unpackbits(cfp, axis=1, count=FULL_BITS)[:, bits].astype(np.float32)
        zl = list(z.values())
        Z = np.stack([z.get(fo, zl[0]) for fo in forms]).astype(np.float32)
        fz = (y * Z).sum(1)
        f["fp"] = fz - fz.max()
        f["fp_rank"] = pd.Series(-fz).rank(method="min").values
        if z_prior is not None:
            fn = (y * (Z - z_prior)).sum(1)
            f["fp_norm"] = fn - fn.max()
    elif z is not None and n:
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
        if ng and "pc_pop" in gen.columns:  # PubChem candidates carry their own popularity
            f.loc[f.is_pc.values == 1, "pop"] = gen.loc[gen.is_pc == 1, "pc_pop"].values
            f["pop_gap"] = f["pop"] - f["pop"].max()
            f["pop_rk"] = f["pop"].rank(ascending=False, method="min")
    if getattr(pool, "np_like", None) is not None:  # NP-likeness: pool rows precomputed, new structures scored here
        from .edge import np_like
        f["np_like"] = np.r_[pool.np_like[cand], [np_like(s) for s in gen.smiles] if ng else []]
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
            "analog_noself", "own_sim", "frag_disc", "analog_w", "analog_ap", "analog_ap_w", "ap_tmax", "np_like"]


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


def pc_gate(h, threshold):
    """Admit PubChem candidates only when the library evidence is weak (best hit similarity below τ)."""
    return threshold is None or (h.sim.max() if len(h) else 0.0) < threshold


def run(test: pd.DataFrame, pool: Pool, lib: Library, ranker=None, fp_models=None, use_gen=False, log=print,
        pubchem=None, pc_top_n=10, pc_gate_tau=None, mass_window=None, direct_only=False):
    rows = []
    qs = queries_from_test(test)
    for i, q in enumerate(qs):
        try:
            mw = mass_window or {"ppm": 10.0, "center_ppm": 0.0}  # test queries are timsTOF (moe.CONFIG)
            cand = pool.window(q.neutral_mass, ppm=mw["ppm"], center_ppm=mw["center_ppm"]) if q.mz \
                else np.zeros(0, np.int64)
            if len(cand) == 0:
                rows.append((q.mid, []))
                continue
            h = analog_hits(q, lib, top=600)  # must match the harness the ranker was trained on
            z = fp_logits(q, fp_models)
            gen = generate(q, pool, h, cand_keys=pool.key[cand]) if use_gen else None
            if pubchem is not None and pc_gate(h, pc_gate_tau):
                from .pubchem import pubchem_candidates
                pcs = pubchem_candidates(q, pubchem, set(pool.key[cand]) | set([] if gen is None else gen.key),
                                         top_n=pc_top_n, ppm=mw["ppm"], center_ppm=mw["center_ppm"])
                if len(pcs):
                    gen = pcs if gen is None else pd.concat([gen.assign(is_pc=0, pc_pop=np.nan), pcs],
                                                            ignore_index=True)
            if formula_nets(fp_models):  # formula model: one prediction per candidate formula
                forms = [fo for fo in candidate_formulas(pool, cand, gen) if fo]
                z = fp_logits(q, fp_models, formulas=forms) if forms else z
            smiles = list(pool.df.smiles.values[cand]) + ([] if gen is None else list(gen.smiles))
            masses = np.r_[pool.mass[cand], [] if gen is None else gen.mass.values]
            FM, pw = frag_matrix(smiles, q)
            f = channel_scores(q, pool, lib, h, cand, z=z, bits=None if fp_models is None else fp_models["bits"],
                               frag=FM @ pw if FM.shape[1] else np.zeros(len(smiles)), gen=gen,
                               frag_disc=frag_disc_score(FM, pw, masses),
                               z_prior=None if fp_models is None else fp_models.get("prior"))
            order = ranker(f) if ranker is not None else heuristic_rank(f)
            if direct_only:  # diagnostic: keep candidates with a library spectrum (scores class 1 alone)
                order = order[f.direct.values[order] > 0]
            rows.append((q.mid, list(f.smiles.values[order])))
        except Exception as e:  # one bad molecule must never sink the file
            log(f"  {q.mid}: {type(e).__name__}: {e}")
            rows.append((q.mid, []))
        if i % 50 == 0:
            log(f"  {i}/{len(qs)} molecules")
    return rows
