"""V13 ranker: PubChem candidates scored with our FP model (Kaggle GPU).

The harness features (casmi26-eval-pc) carry the 100 most popular PubChem structures of each molecule's
mass window in every regime (C1, C2, C2H) plus C2P, where the truth is only in PubChem. Here f·z is
added for every row (PubChem rows' fingerprints recomputed from SMILES) and the all-evidence ranker is
validated out of fold, molecules held out, twice on the same folds:
    without PubChem rows (today's lists)  vs  with them (V13)
so the cost on molecules already in the pool (C1/C2/C2H) is read against the gain on C2P.
Output (/kaggle/working): moe_full.txt + moe.json (full expert only), fpnet.pt, casmi26_models.txt.
"""

import glob
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
INP = os.environ.get("KAGGLE_INPUT", "/kaggle/input")  # overridable for a local dry run
OUT = os.environ.get("RANKER_OUT", "/kaggle/working")
tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
whl = [w for w in glob.glob(f"{INP}/**/rdkit*.whl", recursive=True) if f"-{tag}-" in w][:1]
if whl:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", *whl], check=False)
ART = os.path.dirname(glob.glob(f"{INP}/**/casmi26_artifacts.txt", recursive=True)[0])
EV = os.path.dirname(glob.glob(f"{INP}/**/casmi26_eval.txt", recursive=True)[0])
FPN = glob.glob(f"{INP}/**/fpnet.pt", recursive=True)
DFP = glob.glob(f"{INP}/**/dreams_fp.pt", recursive=True)  # DreaMS-backbone FP models (casmi26-dreams-ft)
FWD = glob.glob(f"{INP}/**/fwdnet.pt", recursive=True)  # forward model (casmi26-fwdnet-train)
GLF = glob.glob(f"{INP}/**/glacier_shard*.parquet", recursive=True)  # GLACIER cosines (casmi26-glacier-feat-*)
MSGF = glob.glob(f"{INP}/**/msg_keys*.parquet", recursive=True)  # truth in MassSpecGym (GLACIER's training data)
PUBLIC = False
_pub = glob.glob(f"{INP}/**/fp_single_s2.pt", recursive=True)
if not FPN and _pub:  # public FPNet checkpoint (same architecture, own 6,930-bit index) as the FP model
    _bits = glob.glob(f"{INP}/**/coconut-casmi26-candidates/**/fp_bits.npy", recursive=True) or \
        [b for b in glob.glob(f"{INP}/**/fp_bits.npy", recursive=True) if "casmi26-artifacts" not in b]
    os.makedirs(f"{OUT}/fppub", exist_ok=True)
    shutil.copy(_pub[0], f"{OUT}/fppub/fpnet.pt")
    shutil.copy(_bits[0], f"{OUT}/fppub/fp_bits.npy")
    FPN, PUBLIC = [f"{OUT}/fppub/fpnet.pt"], True
TRAIN = glob.glob(f"{INP}/**/train.parquet", recursive=True)[0]
os.makedirs(f"{OUT}/code/casmi", exist_ok=True)
for f in glob.glob(f"{ART}/code__*.py"):
    shutil.copy(f, f"{OUT}/code/casmi/" + os.path.basename(f)[len("code__"):])
sys.path.insert(0, f"{OUT}/code")
print("artifacts", ART, "eval", EV, "fpnet", FPN, "dreams", DFP, "fwd", FWD, flush=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from casmi import library as L  # noqa: E402
from casmi import pipeline as P  # noqa: E402
from casmi import rank as R  # noqa: E402
from casmi.fp import full_fp  # noqa: E402
from casmi import fpmodel as M  # noqa: E402
from casmi.search import Query  # noqa: E402


def raw_peaks(rows):
    rows = np.sort(np.unique(rows))
    f = pq.ParquetFile(TRAIN)
    out, start = {}, 0
    for rg in range(f.num_row_groups):
        n = f.metadata.row_group(rg).num_rows
        sel = rows[(rows >= start) & (rows < start + n)] - start
        if len(sel):
            t = f.read_row_group(rg, columns=["ms2_mzs", "ms2_normalized_intensities"]).take(sel)
            for loc, mz, it in zip(sel, t["ms2_mzs"].to_pylist(), t["ms2_normalized_intensities"].to_pylist()):
                out[start + int(loc)] = (np.asarray(mz, float), np.asarray(it, float))
        start += n
    return out


USE_NP = False
# the saved ranker: V13 budget (PubChem top-100), trained on every regime present (incl. C2X when there);
# validation alone (est) has preferred variants the LB then rejected, so the choice is explicit
FORCE = os.environ.get("CASMI_FORCE") or "V13 budget, trained without C2PX"  # = V16a's regimes (C1 C2 C2H C2X C2P)
CALIBRATE = True


def fit_calib(Z, Y, l2=1.0, iters=30):
    """Per-bit logistic regression y ~ sigmoid(a·z + c), all bits at once (Newton steps on the 2x2 systems),
    L2 pull towards a = 1, c = 0. Z, Y: (molecules, bits)."""
    a = np.ones(Z.shape[1])
    c = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(a * Z + c, -30, 30)))
        g_a = ((p - Y) * Z).sum(0) + l2 * (a - 1)
        g_c = (p - Y).sum(0) + l2 * c
        w = p * (1 - p)
        h_aa = (w * Z * Z).sum(0) + l2
        h_ac = (w * Z).sum(0)
        h_cc = w.sum(0) + l2
        det = h_aa * h_cc - h_ac ** 2
        det = np.where(np.abs(det) < 1e-9, 1e-9, det)
        # clipped Newton steps: from saturated starting points full steps overshoot
        a -= np.clip((h_cc * g_a - h_ac * g_c) / det, -0.25, 0.25)
        c -= np.clip((-h_ac * g_a + h_aa * g_c) / det, -1.0, 1.0)
        a = np.clip(a, 0.0, 3.0)
    return a, c


def calibrate(z_of, feats_df, fp_full, bits):
    """Cross-fitted per-bit calibration of the molecules' FP logits on their own truths (5 molecule folds:
    a molecule's logits are calibrated with parameters fitted without it); the all-molecule fit is saved for
    inference (fp_calib.npz)."""
    tr = feats_df[(feats_df.label == 1) & (feats_df.pool_row >= 0)].drop_duplicates("qkey")
    keys = [k for k in tr.qkey if k in z_of and z_of[k] is not None]
    rows = tr.set_index("qkey").pool_row.reindex(keys).values.astype(int)
    Y = np.unpackbits(np.asarray(fp_full[np.sort(rows)])[np.argsort(np.argsort(rows))], axis=1,
                      count=P.FULL_BITS)[:, bits].astype(np.float64)
    Z = np.stack([z_of[k] for k in keys]).astype(np.float64)
    fold = np.random.default_rng(1).integers(0, 5, len(keys))
    out = dict(z_of)
    for k in range(5):
        a, c = fit_calib(Z[fold != k], Y[fold != k])
        for i in np.flatnonzero(fold == k):
            out[keys[i]] = (a * Z[i] + c).astype(np.float32)
    a, c = fit_calib(Z, Y)
    np.savez(f"{OUT}/fp_calib.npz", a=a.astype(np.float32), c=c.astype(np.float32))
    ll = lambda zz: float(np.mean(Y * zz - np.logaddexp(0, zz)))  # noqa: E731
    print(f"per-bit calibration on {len(keys)} molecules: mean slope {a.mean():.3f}, "
          f"mean log-lik raw {ll(Z):.4f} -> cross-fitted {ll(np.stack([out[k] for k in keys])):.4f}", flush=True)
    return out


def train_rows(f, name):
    """Training rows of a variant: regime exclusions / sub-sampling by variant name."""
    import zlib
    if "without C2X" in name:
        f = f[~f.regime.isin(["C2X", "C2PX"])]
    if "without C2PX" in name or name.startswith("V16a regimes"):
        f = f[f.regime != "C2PX"]
    if "C2P halved" in name:  # C2P lists of half the molecules (stable hash of the molecule key)
        half = f.qkey.map(lambda k: zlib.crc32(str(k).encode()) % 2 == 0)
        f = f[~((f.regime == "C2P") & half)]
    if "no C2P" in name:
        f = f[f.regime != "C2P"]
    if "low-pop x3" in name and "pop" in f.columns:
        # the hidden test's class 2/3 compounds are far less documented than validation truths (popularity alone:
        # A 0.78 / C 0.21 MRR): lists whose truth is not among the 3 most popular candidates count three times
        pr = f["pop"].fillna(-1.0).groupby(f.grp, sort=False).rank(ascending=False, method="min")
        tr_rank = pr.where(f.label.values == 1).groupby(f.grp, sort=False).transform("min")
        low = f[(tr_rank > 3).values]
        f = pd.concat([f] + [low.assign(grp=low.grp.astype(str) + f"#dup{i}") for i in (1, 2)])
        f = f.reset_index(drop=True)
    return f


def copy_fp(out):
    """The FP model(s) next to the ranker for the submission notebook (fpnet*.pt are all loaded there; the
    calibration fp_calib.npz written by calibrate() sits in the same directory)."""
    for i, pth in enumerate(sorted(FPN)):
        shutil.copy(pth, f"{out}/fpnet.pt" if len(FPN) == 1 else f"{out}/fpnet_{i}.pt")
    for i, pth in enumerate(sorted(DFP)):
        shutil.copy(pth, f"{out}/dreams_fp.pt" if len(DFP) == 1 else f"{out}/dreams_fp_{i}.pt")
    if FWD:
        shutil.copy(FWD[0], f"{out}/fwdnet.pt")
    b = f"{os.path.dirname(FPN[0])}/fp_bits.npy" if FPN else ""
    if b and os.path.exists(b):  # a model with its own bit index ships it next to the weights
        shutil.copy(b, f"{out}/fp_bits.npy")


def _formula_of(smi):
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    m = Chem.MolFromSmiles(smi)
    return CalcMolFormula(m) if m is not None else None


def relative(f):
    """Within-list relative features (pipeline.add_relative), vectorised over groups."""
    g = f.groupby("grp", sort=False)
    for c in [c for c in P.REL_COLS if c in f.columns]:
        f[c + "_gap"] = f[c] - g[c].transform("max")
        f[c + "_rk"] = g[c].rank(ascending=False, method="min")
    f["n_cand"] = g["key"].transform("size")
    f["fp"] = f.fz - g.fz.transform("max")
    f["fp_rank"] = g.fz.rank(ascending=False, method="min")
    if "fzn" in f.columns:
        f["fp_norm"] = f.fzn - g.fzn.transform("max")
    if "pop" in f.columns:
        f["pop_gap"] = f["pop"] - g["pop"].transform("max")
    return f


def main():
    lib = L.load(f"{ART}/library.npz")
    qs = pd.read_parquet(f"{EV}/queries.parquet")
    feats_df = pd.read_parquet(f"{EV}/features.parquet")
    feats_df = feats_df[feats_df.is_gen == 0].reset_index(drop=True)  # PubChem rows: is_gen 0, is_pc 1
    if not USE_NP:  # NP-likeness lost 0.019 on the LB (V14): not a ranker feature
        feats_df = feats_df.drop(columns=[c for c in feats_df.columns if c.startswith("np_like")])
    if GLF:  # GLACIER (fragment-level spectrum simulator): mean cosine with the molecule's measured spectra
        gl = pd.concat([pd.read_parquet(p_) for p_ in GLF]).drop_duplicates(["qkey", "smiles"])
        feats_df = feats_df.merge(gl[["qkey", "smiles", "gl_cos"]].rename(columns={"gl_cos": "glacier"}),
                                  on=["qkey", "smiles"], how="left")
        print("GLACIER scores", len(gl), "| rows covered", round(float(feats_df.glacier.notna().mean()), 3),
              feats_df.groupby("panel").glacier.apply(lambda x: round(float(x.notna().mean()), 3)).to_dict(), flush=True)
    if MSGF:
        m_ = pd.concat([pd.read_parquet(p_) for p_ in MSGF]).drop_duplicates("qkey").set_index("qkey").truth_in_msg
        feats_df["in_msg"] = feats_df.qkey.map(m_).astype("float32")
        print("molecules with the truth in MassSpecGym:", feats_df.drop_duplicates("qkey").groupby("panel").in_msg.mean()
              .round(3).to_dict(), flush=True)
    if os.environ.get("DRY_N"):  # local dry run: a molecule subsample, random f·z (no FP model)
        keep = feats_df.qkey.drop_duplicates().sample(int(os.environ["DRY_N"]), random_state=0)
        feats_df = feats_df[feats_df.qkey.isin(keep)].reset_index(drop=True)
        feats_df["fz"] = np.random.default_rng(0).normal(size=len(feats_df)).astype(np.float32)
        feats_df["fzn"] = feats_df.fz
        return evaluate_and_save(feats_df)
    print("rows", len(feats_df), "PubChem rows", int(feats_df.is_pc.sum()),
          feats_df.groupby("regime").qkey.nunique().to_dict(), flush=True)
    fp_full = np.load(f"{ART}/fp_full.npy", mmap_mode="r")
    own_bits = f"{os.path.dirname(FPN[0])}/fp_bits.npy" if FPN else ""
    bits = np.load(own_bits if os.path.exists(own_bits) else f"{ART}/fp_bits.npy", allow_pickle=False)
    print("FP bit index:", len(bits), "public checkpoint" if PUBLIC else "", flush=True)
    if len(FPN) + len(DFP) > 1 or DFP:  # several FP models (FPNet / DreaMS): logits averaged over nets
        os.makedirs(f"{OUT}/fpbank", exist_ok=True)
        for i, pth in enumerate(sorted(FPN)):
            shutil.copy(pth, f"{OUT}/fpbank/fpnet_{i}.pt")
        for i, pth in enumerate(sorted(DFP)):
            shutil.copy(pth, f"{OUT}/fpbank/dreams_fp_{i}.pt")
        fpm = P.load_fp_models(f"{OUT}/fpbank")
    else:
        fpm = P.load_fp_models(os.path.dirname(FPN[0]))
    fpm["bits"] = bits
    print("fp nets", len(fpm["nets"]), "device", fpm["device"], f"{time.time() - T0:.0f}s", flush=True)

    raw = raw_peaks(lib["row"][qs.lrow.values])
    libs = list(lib["libs"])
    # formula models (run 6+): every candidate is scored under its own molecular formula
    use_form = P.formula_nets(fpm)
    if use_form:
        from multiprocessing import Pool as MP
        pool_form = pd.read_parquet(f"{ART}/pool.parquet", columns=["formula"]).formula.values
        new = pd.unique(feats_df.smiles.values[feats_df.pool_row.values < 0])
        with MP(os.cpu_count() or 4) as mp:
            fmap = dict(zip(new, mp.map(_formula_of, list(new), chunksize=512)))
        pr = feats_df.pool_row.values
        feats_df["formula"] = [pool_form[r] if r >= 0 else fmap.get(sm) for r, sm in zip(pr, feats_df.smiles.values)]
        forms_of = feats_df.groupby("qkey").formula.agg(lambda x: [v for v in pd.unique(x) if v])
        print("candidate formulas per molecule", forms_of.str.len().describe().round(1).to_dict(), flush=True)
    z_of, q_of = {}, {}
    for key, g in qs.groupby("key", sort=False):
        q = Query(key, [], [], [], [], [], [])
        for lr in g.lrow.values:
            a, b = lib["off"][lr], lib["off"][lr + 1]
            q.mz.append(lib["mz"][a:b])
            q.p.append(lib["p"][a:b])
            q.adduct.append(L.ADDUCTS[lib["adduct_code"][lr]])
            q.mode.append(int(lib["mode"][lr]))
            q.prec.append(float(lib["prec_mz"][lr]))
            q.ce.append(float(lib["ce"][lr]))
            mz, it = raw[int(lib["row"][lr])]
            env = libs[lib["lib_code"][lr]].startswith("enveda") and lib["mode"][lr] > 0
            q.raw.append((mz + (L.ENVEDA_POS_SHIFT if env else 0.0), it))
        q.instr = [M.INSTR_LIST[int(lib["instr"][lr])] for lr in g.lrow.values]  # true instrument
        z_of[key] = P.fp_logits(q, fpm, formulas=forms_of.get(key, [])) if use_form else P.fp_logits(q, fpm)
        q_of[key] = q
    print("logits", len(z_of), f"{time.time() - T0:.0f}s", flush=True)
    if CALIBRATE and not use_form and not PUBLIC:  # a public model has seen the panels: no calibration on them
        z_of = calibrate(z_of, feats_df, fp_full, bits)

    # candidate fingerprints: pool rows from the packed matrix, PubChem rows recomputed from SMILES
    gen = feats_df.pool_row.values < 0
    gsmi = pd.unique(feats_df.smiles.values[gen])
    from multiprocessing import Pool as MP
    with MP(os.cpu_count() or 4) as mp:
        gfp = dict(zip(gsmi, mp.map(full_fp, list(gsmi), chunksize=256)))
    print("PubChem fingerprints", len(gfp), f"{time.time() - T0:.0f}s", flush=True)
    zp = np.load(f"{ART}/fp_prior.npy") if os.path.exists(f"{ART}/fp_prior.npy") else None
    if zp is not None and len(zp) != len(bits):  # the prior belongs to our bit index
        zp = None
    fz = np.zeros(len(feats_df), np.float32)
    fzn = np.zeros(len(feats_df), np.float32)
    for (key, _), g in feats_df.groupby(["qkey", "regime"], sort=False):
        pr = g.pool_row.values
        fps = np.stack([np.asarray(fp_full[r]) if r >= 0 else gfp[s] for r, s in zip(pr, g.smiles.values)])
        y = np.unpackbits(fps, axis=1, count=P.FULL_BITS)[:, bits].astype(np.float32)
        zk = z_of[key]
        if isinstance(zk, dict):  # per-row logits by the candidate's formula
            first = next(iter(zk.values()))
            Z = np.stack([zk.get(fo, first) for fo in g.formula.values]).astype(np.float32)
            fz[g.index.values] = (y * Z).sum(1)
            if zp is not None:
                fzn[g.index.values] = (y * (Z - zp)).sum(1)
        else:
            z = zk.astype(np.float32)
            fz[g.index.values] = y @ z
            if zp is not None:
                fzn[g.index.values] = y @ (z - zp)
    feats_df["fz"] = fz
    if zp is not None:
        feats_df["fzn"] = fzn
    print("f·z done", f"{time.time() - T0:.0f}s", flush=True)
    if FWD:  # forward model: cosine of the predicted spectrum (from the candidate's bits) with the measured ones
        from casmi import fwdmodel as W
        from casmi.edge import spectrum_weights
        net = W.load(FWD[0])
        fwd = np.zeros(len(feats_df), np.float32)
        for key, g in feats_df.groupby("qkey", sort=False):  # each candidate once per molecule (all regimes)
            u = g.drop_duplicates("smiles")
            fps = np.stack([np.asarray(fp_full[r]) if r >= 0 else gfp[s_] for r, s_ in zip(u.pool_row.values,
                                                                                          u.smiles.values)])
            y = np.unpackbits(fps, axis=1, count=P.FULL_BITS)[:, bits].astype(np.float32)
            sc = dict(zip(u.smiles.values, W.scores(net, y, q_of[key], weights=spectrum_weights(q_of[key]))))
            fwd[g.index.values] = [sc[s_] for s_ in g.smiles.values]
        feats_df["fwd"] = fwd
        rep = R.mrr_of(feats_df.assign(grp=feats_df.qkey + "|" + feats_df.regime), feats_df.fwd.values)
        print("forward model alone (cosine) MRR", rep.groupby(["panel", "regime"]).mrr.mean().round(3).to_dict(),
              f"{time.time() - T0:.0f}s", flush=True)
    evaluate_and_save(feats_df)


def evaluate_and_save(feats_df):
    def blended(f, s):
        out = np.empty(len(f))
        for _, g in f.groupby("grp", sort=False):
            out[g.index] = R.blend_pop(s[g.index], g["pop"].values) if "pop" in g else s[g.index]
        return out

    from casmi import moe
    regimes = tuple(r for r in ("C1", "C2", "C2H", "C2X", "C2P", "C2PX") if r in set(feats_df.regime))
    f_pc = relative(R.prepare(feats_df, regimes=regimes))
    folds = f_pc.drop_duplicates("qkey").set_index("qkey").fold  # same molecule folds for both variants
    f_no = feats_df[feats_df.is_pc == 0].copy()
    f_no = relative(R.prepare(f_no, regimes=regimes))
    f_no["fold"] = f_no.qkey.map(folds).values
    cols = moe.expert_features(f_pc, "full")
    print("features", len(cols), flush=True)
    # PubChem budgets: with pc_rank (popularity rank in the window) the top-n subsets of one harness run
    # are exactly what a top-n channel would produce, so several budgets are compared on the same folds
    variants = [("without PubChem rows", f_no)]
    np_cols = [c for c in ("np_like", "np_like_gap", "np_like_rk") if c in f_pc.columns]
    budgets = [None]
    if "pc_rank" in feats_df.columns and feats_df.pc_rank.max() >= 150:
        budgets = [n for n in (50, 100, 200, 300) if n <= feats_df.pc_rank.max() + 1]
    for n in budgets:
        if n is None:
            variants.append(("with PubChem rows (V13)", f_pc))
            continue
        fb = feats_df[(feats_df.is_pc == 0) | (feats_df.pc_rank < n)]
        fb = relative(R.prepare(fb, regimes=regimes))
        fb["fold"] = fb.qkey.map(folds).values
        variants.append((f"with PubChem top-{n}", fb))
    tabs = {}

    def lowpop_keys(f):
        """(qkey, regime) lists whose truth is not among the list's 3 most popular candidates."""
        pr = f.assign(p=f["pop"].fillna(-1.0)).groupby("grp", sort=False).p.rank(ascending=False, method="min")
        t = f.assign(pr=pr.values)[f.label.values == 1].groupby(["qkey", "regime"]).pr.min()
        return set(t[t > 3].index)

    def add_tab(name, f, rep):
        tabs[name] = rep.groupby(["panel", "regime"]).mrr.mean()
        if "pop" in f.columns:
            lp = lowpop_keys(f)
            m = [(k, r) in lp for k, r in zip(rep.qkey, rep.regime)]
            tabs[name + " | low-pop truths"] = rep[m].groupby(["panel", "regime"]).mrr.mean()
        if "in_msg" in f.columns:  # structures GLACIER was not trained on (the hidden test's situation)
            unseen = set(f.qkey[f.in_msg == 0])
            tabs[name + " | truth not in MassSpecGym"] = rep[rep.qkey.isin(unseen)].groupby(["panel", "regime"]).mrr.mean()
    if "glacier" in f_pc.columns:
        add_tab("GLACIER alone, with PubChem", f_pc, R.mrr_of(f_pc, f_pc.glacier.fillna(-1).values))
    for name, f in (("FP alone (f·z), no PubChem", f_no), ("FP alone (f·z), with PubChem", f_pc)):
        rep = R.mrr_of(f, f.fz.values)
        add_tab(name, f, rep)
        # top-1 among the window's candidates (forum's FPNet: 0.46-0.49 on np-examples)
        tabs[name + " top1"] = (rep.mrr == 1).groupby([rep.panel, rep.regime]).mean()
    if np_cols:  # NP-likeness on/off at the V13 budget, same folds
        f100 = dict(variants).get("with PubChem top-100", f_pc)
        variants.append(("with PubChem top-100, no NP-likeness", f100))
    # ranker trained on all three panels (B = enveda-180 isomer lists) instead of A + C, V13 budget
    base = dict(variants).get("with PubChem top-100", dict(variants).get("with PubChem rows (V13)"))
    if base is not None:
        variants.append(("V13 budget, ranker trained on A+B+C", base))
        if "C2X" in regimes:  # does training on the analog-starved regime change the ranker?
            variants.append(("V13 budget, trained without C2X", base))
        if "C2PX" in regimes:
            variants.append(("V13 budget, trained without C2PX", base))
            # PubChem-promotion dose (LB: 33 % PubChem-only training lists 0.351, 25 % 0.372, 20 % 0.379)
            variants.append(("V16a regimes, C2P halved", base))
            variants.append(("V16a regimes, no C2P", base))
            variants.append(("V16a regimes, low-pop x3", base))
            if "glacier" in f_pc.columns:
                variants.append(("V16a regimes, no GLACIER", base))
                variants.append(("V16a regimes, low-pop x3, no GLACIER", base))
    frames = dict(variants)
    for name, f in variants:
        train_panels = ["A", "B", "C"] if "A+B+C" in name else ["A", "C"]
        cols = [c for c in moe.expert_features(f_pc, "full") if not (name.endswith("no NP-likeness") and c in np_cols)]
        if "no GLACIER" in name:
            cols = [c for c in cols if not c.startswith("glacier")]
        oof = np.zeros(len(f))
        for k in range(5):
            tr = train_rows(f[(f.fold != k) & f.panel.isin(train_panels)], name)
            va = f[f.fold == k]
            oof[va.index] = moe._fit(tr, cols, moe.CONFIG["level1_rounds"]).predict(va[cols].astype(np.float32))
        rep = R.mrr_of(f, blended(f, oof))
        add_tab(name, f, rep)
        # rows: where the truth's best PubChem / pool competitor sits; is_pc share of first places
        top = f.assign(s=blended(f, oof)).sort_values(["grp", "s"], ascending=[True, False]).groupby("grp").head(1)
        print(name, "| share of lists topped by a PubChem row:",
              top.groupby(["panel", "regime"]).is_pc.mean().round(3).to_dict(), flush=True)
        print(name, f"{time.time() - T0:.0f}s", flush=True)
    tab = pd.DataFrame(tabs).T
    pd.set_option("display.width", 220)
    print("=== V13 VALIDATION (out of fold, molecules held out, FP incl., MRR@25) ===")
    print(tab.round(4).to_string())
    # C2P counts as 0 without PubChem rows (the truth is unreachable); the class-2 split between
    # in-pool and PubChem-only molecules is unknown, so report a range of PubChem-only shares
    if ("C", "C2PX") in tab.columns:  # analog-poor proxy: class 2 = C2X (in pool) + C2PX (PubChem only)
        for s2p in (0.1, 0.2):
            print(f"analog-poor estimate (C2X/C2PX), PubChem-only share {s2p}:",
                  {n: {p: round(0.16 * tab.loc[n].get((p, "C1"), 0) + (0.45 - s2p) * tab.loc[n].get((p, "C2X"), 0)
                                + s2p * tab.loc[n].get((p, "C2PX"), 0), 4) for p in "AC"} for n in tab.index},
                  flush=True)
    for s2p in (0.1, 0.2, 0.3):
        w = {n: {p: round(0.16 * tab.loc[n].get((p, "C1"), 0) + (0.45 - s2p) / 2 * (tab.loc[n].get((p, "C2"), 0)
                                                                                   + tab.loc[n].get((p, "C2H"), 0))
                          + s2p * tab.loc[n].get((p, "C2P"), 0), 4) for p in "AC"} for n in tab.index}
        print(f"weighted, PubChem-only share of test {s2p}:", w, flush=True)
    # keep the variant with the best panel-C estimate at a 15 % PubChem-only share (V13's LB gain implies
    # roughly 10-15 % under the panel-C proxy)
    def est(n, s2p=0.15):
        """Panel-C estimate; with C2X present the in-pool class 2 is (C2H + C2X) / 2 — the LB's class-2 MRR
        (~0.50) sits below C2H, so the analog-starved regimes are the closer proxy."""
        t = tab.loc[n]
        c2a, c2b = ("C2H", "C2X") if ("C", "C2X") in t.index else ("C2", "C2H")
        return (0.16 * t.get(("C", "C1"), 0) + (0.45 - s2p) / 2 * (t.get(("C", c2a), 0) + t.get(("C", c2b), 0))
                + s2p * t.get(("C", "C2P"), 0))
    cand = [n for n, _ in variants if n != "without PubChem rows"]
    best = FORCE if FORCE in dict(variants) else max(cand, key=est)
    print("chosen:", best, {n: round(est(n), 4) for n in cand}, flush=True)
    if "A+B+C" in best:  # the final fit follows the chosen training panels
        moe.fit_save(frames[best], OUT, panels=("A", "B", "C"), names=["full"], stack=False, seeds=3)
        open(f"{OUT}/casmi26_models.txt", "w").write(best + "\n")
        if FPN or DFP or FWD:
            copy_fp(OUT)
        print("saved", sorted(os.listdir(OUT)), flush=True)
        return
    if best.endswith("no NP-likeness"):
        frames[best] = frames[best].drop(columns=np_cols)
    # the final fit sees the same training rows as the variant's cross-validation (regime exclusions included)
    moe.fit_save(train_rows(frames[best], best).reset_index(drop=True), OUT, names=["full"], stack=False, seeds=3)
    open(f"{OUT}/pc_budget.txt", "w").write(best + "\n")
    open(f"{OUT}/casmi26_models.txt", "w").write("V13: full ranker with FP, trained with PubChem rows\n")
    if FPN or DFP or FWD:
        copy_fp(OUT)
    print("saved", sorted(os.listdir(OUT)), f"{time.time() - T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
