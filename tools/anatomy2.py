"""Anatomy 2 -- channel drift of the chain against EXACT pre-activation labels.

    python tools/anatomy2.py <candidate.py> --panel fit --n 20 \
        --out C:/Users/samsu/Documents/arc-whest/experiments/2026-09-25_anatomy2/fit20.json
    python tools/anatomy2.py <candidate.py> --verify --out <...>/verify.json
    python tools/anatomy2.py --report <...>/fit20.json          markdown tables

HANDOFF 2026-09-25 Step 1 / PLAN rev 13 §3 Step 1. Labels: keenanpepper's
`arc-whestbench-p2-full1000-N1e9` (raw moments to 4th order, dense pairwise, N=1e9;
`tools/labels/cumulants.py` turns them into cumulant slices) plus `gate_p` from
`whest-p2-bakev2-bench-supp`. Both exist only for the `full` split, so this reads
FIT nets (full rows 0..99); DEV (`mini`) has no dense pre-activation labels.

THE OBJECTS, and what the chain calls them (all at the pre-activation z_l):

    mu    mu_pre            E[z_i]
    var   var               kappa_ii
    C11   cov_pre * off     kappa_ij, i != j
    D3    D3                kappa_iii
    D21   D21               kappa_iij  (first index squared, as the labels)
    dG    dG = D4 / m       kappa_iiii / m,  m = metric = 2
    D4    D4 (after star)   kappa_iiii -- v82+ adds the star D4_star AFTER dG, so a dG
                            substitution alone counts the star twice; every arm that
                            substitutes dG also substitutes D4 (the identity on a
                            star-free chain such as v59, so Anatomy 2 is comparable)
    D22   D22               kappa_iijj   -- the chain's form is (m/6)(dG_i + dG_j)
    Q31n  nb["Q31"]         kappa_iiij / (s_i^3 s_j) -- chain's form (m/2) lam C_ij

WHAT IS MEASURED, per net and layer l = 1..15 (0-based; "layer 2..16" 1-based):

1. Object errors of the unmodified chain against the labels (relative Frobenius,
   MSE; covariance error split into diagonal / trace-part / traceless shares), for
   v59 and for the unshared T0 chain (+ T0 with exact P2 births) so the old-tier
   compression share of the D21 error is separable. Also cov_out (post-ReLU).
2. kappa4 model sufficient statistics (normalised slices; Gram matrices so any
   train/eval split of nets can be fitted afterwards): kappa_iijj against
   {B = chain form, rho, rho^2, g-bar rho, alpha_i alpha_j}; kappa_iiij against
   {chain form C/(s_i^3 s_j), rho, rho^2, g_i rho, alpha_i alpha_j}.
3. Truth substitution.
   ONE-STEP: from the chain's own state after layer l-1, step layer l with mu
   labelled, and additionally var / D3 / dG / all four labelled; mean MSE at l.
   The layer-l mean reads ONLY these marginals, so the off-diagonal objects are
   tested by a TWO-STEP branch: from the state after l-2, step l-1 with mu and X
   labelled (X in C11, D21, D22, Q31n, all of them, cov = var + C11), then step l
   with mu labelled, and read the mean at l.
   ROLLED: the whole chain with X substituted at every layer 1..P, free-running
   after P; mean MSE at every layer, in particular the final one.

MSE WITHOUT A NOISE FLOOR. aicrowd's `gt_mean` and keenanpepper's `mean` are two
independent N = 1e9 estimates, so mean((p - gt) * (p - kp)) is an unbiased
estimate of the noise-free MSE. Both it and the plain MSE vs gt are stored.

The candidate is patched with ONE inert call per object, `X = _a2("X", X)`, which
returns its argument unchanged while `_A2 is None`. `--verify` requires the
patched chain driven by this file's own layer loop to equal the unpatched
`predict` BIT FOR BIT, and a branch-laden pass to leave the main chain unchanged.

Accuracy backend only (cupy shims); no FLOP is ever quoted from here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL_SIM = ROOT / "local_sim"
GPU_SHIM = ROOT / "tools" / "gpu_shim"

BACKEND = os.environ.get("WHEST_BACKEND", "cupy").lower()
# Same deliberate insert as tools/probe.py and tools/anatomy.py: accuracy only,
# never a FLOP, so the shims in front are the point.
sys.path.insert(0, str(LOCAL_SIM))
if BACKEND == "cupy":
    sys.path.insert(0, str(GPU_SHIM))
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "tools"))
sys.path.append(str(ROOT / "tools" / "labels"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np  # noqa: E402

BUDGET = 2 ** 41
METRIC = 2.0
BENCH_DIR = ROOT / "cache" / "hf" / "whest-p2-bakev2-bench-supp" / "s00"

# ---------------------------------------------------------------- the patch

_GLOBALS = '''_CORR_HOOK = None

# --- tools/anatomy2.py instrumentation. `_A2` is None in every shipped file and
# this copy is never submitted. `_a2` returns its argument unchanged when unset.
_A2 = None


def _a2(name, value, *aux):
    if _A2 is None:
        return value
    return _A2(_CUR_LAYER, name, value, *aux)
'''

PATCHES = [
    ("_CORR_HOOK = None\n", _GLOBALS),
    ("    mu_pre = WT @ mu\n",
     "    mu_pre = WT @ mu\n    mu_pre = _a2('mu', mu_pre)\n"),
    ("    metric = float(init_scale)\n",
     "    var = _a2('var', var)\n    metric = float(init_scale)\n"),
    ("    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n",
     "    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n"
     "    D3 = _a2('D3', D3)\n"),
    ("    dG = fnp.where(fnp.isfinite(dG), dG, dtype(0.0))\n",
     "    dG = fnp.where(fnp.isfinite(dG), dG, dtype(0.0))\n    dG = _a2('dG', dG)\n"),
    ("    if last_layer:\n        S = {(1,): mu_pre, (3,): D3, (4,): D4}\n",
     "    D4 = _a2('D4', D4)\n"
     "    if last_layer:\n        S = {(1,): mu_pre, (3,): D3, (4,): D4}\n"),
    ("        C11 = cov_pre * off_mask\n",
     "        C11 = cov_pre * off_mask\n        C11 = _a2('C11', C11)\n"),
    ("        D21 = (_dslice21(adv, off_mask) if adv is not None\n"
     "               else fnp.zeros((n, n), dtype=dtype))\n",
     "        D21 = (_dslice21(adv, off_mask) if adv is not None\n"
     "               else fnp.zeros((n, n), dtype=dtype))\n"
     "        D21 = _a2('D21', D21)\n"),
    ("        D22 = ((metric / 6.0) * (dG[:, None] + dG[None, :])) * off_mask\n",
     "        D22 = ((metric / 6.0) * (dG[:, None] + dG[None, :])) * off_mask\n"
     "        D22 = _a2('D22', D22)\n"),
    ('        nb["Q13"] = nb["Q31"].T\n',
     '        nb["Q31"] = _a2("Q31n", nb["Q31"], inv, inv2)\n'
     '        nb["Q13"] = nb["Q31"].T\n'),
]


def patch_source(src: str) -> str:
    for anchor, repl in PATCHES:
        c = src.count(anchor)
        if c != 1:
            raise SystemExit(f"anatomy2: anchor occurs {c} times, expected 1:\n{anchor}")
        src = src.replace(anchor, repl, 1)
    return src


# ------------------------------------------------------------------ harness


class _MLP:
    __slots__ = ("weights", "width", "depth", "seed")

    def __init__(self, weights, width, depth, seed):
        self.weights, self.width, self.depth, self.seed = weights, width, depth, seed


class _Ctx:
    def __init__(self, width, depth, budget, seed):
        self.width, self.depth = width, depth
        self.flop_budget, self.seed = budget, seed
        self.api_version, self.scratch_dir, self.submission_dir = "1.0", None, None


def load_module(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class GenCap(Exception):
    pass


class Chain:
    """The candidate's `_tower` loop, driven from here so it can branch."""

    def __init__(self, mod, est, w32, width, depth, budget=BUDGET):
        self.mod, self.est = mod, est
        self.w, self.width, self.depth = w32, width, depth
        self.dtype = mod.fnp.float32
        self.table = mod._tables()
        self.gens = mod._fit_generations(width, depth, budget)
        if self.gens is not None:
            raise GenCap(f"generation cap {self.gens} -- not the exact tower")
        self.const = est._constants(width, self.dtype)

    def init_state(self):
        fnp, dt, n = self.mod.fnp, self.dtype, self.width
        return (fnp.zeros(n, dtype=dt), fnp.eye(n, dtype=dt), None, 0.0,
                fnp.zeros(n, dtype=dt))

    def step(self, l, state):
        m = self.mod
        s = m._layer(self.w[l], state, self.table, self.dtype, self.const, METRIC,
                     last_layer=(l == self.depth - 1), gens=self.gens,
                     layer_index=l, depth=self.depth)
        mu = m.fnp.where(m.fnp.isfinite(s[0]), s[0], self.dtype(0.0))
        return (mu, s[1], s[2], s[3], s[4])


class Ctl:
    """`_A2`: records the chain's objects and substitutes labelled ones."""

    def __init__(self, xp):
        self.xp = xp
        self.truth = None          # {l: {name: array in chain dtype}}
        self.sub = frozenset()     # names substituted in the CURRENT step
        self.rec = None            # dict to record into, or None

    def __call__(self, l, name, value, *aux):
        if name == "Q31n":
            inv, inv2 = aux
            norm = (inv2 * inv)[:, None] * inv[None, :]
            if self.rec is not None:
                self.rec["Q31"] = value / norm
            if name in self.sub:
                return (self.truth[l]["Q31"] * norm).astype(value.dtype)
            return value
        if self.rec is not None:
            self.rec[name] = value
        if name in self.sub:
            return self.truth[l][name].astype(value.dtype)
        return value


# ------------------------------------------------------------------- labels


def load_truth(gidx: int, xp, dtype):
    """{l: {name: xp array (chain dtype)}}, float64 host marginals, post truths."""
    from cumulants import NetLabels
    L = NetLabels(gidx, xp)
    mg = L.pre_marginals()
    gate = np.load(BENCH_DIR / f"net_{gidx:05d}" / "marg.npz")
    assert str(gate["name"]) == L.name, "bench-supp / full1000 name mismatch"
    gate_p = np.asarray(gate["gate_p"], dtype=np.float64)
    depth = mg["mu"].shape[0]
    truth = {}
    post_cov = {}
    for l in range(depth):
        d = L.layer(l)
        t = {"mu": mg["mu"][l], "var": mg["var"][l], "D3": mg["k3"][l],
             "dG": mg["k4"][l] / METRIC, "D4": mg["k4"][l],
             "C11": d["C"], "D21": d["k3_iij"], "D22": d["k4_iijj"], "Q31": d["k4_iiij"]}
        eye = xp.eye(d["C"].shape[0], dtype=bool)
        t["C11"] = t["C11"].copy()
        t["C11"][eye] = 0.0
        truth[l] = {k: xp.asarray(v).astype(dtype) for k, v in t.items()}
        truth[l]["_Cfull"] = d["C"].astype(dtype)
        M11 = xp.asarray(L.dense("M11")[l], dtype=xp.float64)
        a = xp.asarray(L.post_mean()[l])
        post_cov[l] = (M11 - a[:, None] * a[None, :]).astype(dtype)
        del d, M11
    return L, mg, gate_p, truth, post_cov


# ------------------------------------------------------------------ metrics


def _f(x, xp):
    return x.astype(xp.float64)


def mse_pair(pred_l, gt_l, kp_l):
    """(plain MSE vs gt, cross-estimate of noise-free MSE). Host float64 inputs."""
    a, b = pred_l - gt_l, pred_l - kp_l
    return float((a * a).mean()), float((a * b).mean())


def object_errors(rec, t, post_cov_l, state_cov, xp):
    """Relative Frobenius error and MSE of each carried object at one layer."""
    out = {}

    def rel(x, y):
        x, y = _f(x, xp), _f(y, xp)
        e = x - y
        ee = float((e * e).sum())
        return {"rel": float(np.sqrt(ee / max(float((y * y).sum()), 1e-300))),
                "mse": ee / e.size}

    for k in ("mu", "var", "D3", "dG", "D4", "C11", "D21", "D22", "Q31"):
        if k in rec:
            out[k] = rel(rec[k], t[k])
    if "C11" in rec:
        # the full covariance error, split three ways
        n = rec["var"].shape[0]
        E = _f(rec["C11"], xp) + xp.diag(_f(rec["var"], xp)) - _f(t["_Cfull"], xp)
        tot = float((E * E).sum())
        dE = xp.diagonal(E)
        tr = float(dE.mean())
        out["cov_split"] = {"diag_share": float((dE * dE).sum()) / tot,
                            "trace_share": n * tr * tr / tot,
                            "traceless_share": 1.0 - n * tr * tr / tot,
                            "rel": float(np.sqrt(tot / float((_f(t["_Cfull"], xp) ** 2).sum())))}
    if state_cov is not None:
        out["cov_out"] = rel(state_cov, post_cov_l)
    return out


def k4_stats(l, t, mg, gate_p, k4post_prev, W, xp):
    """Gram sufficient statistics for the kappa4 slice models at one layer.

    The last column of each design is the TRANSPORTED DIAGONAL: the part of the
    pre-activation kappa4 slice that the previous layer's per-neuron post-ReLU
    kappa4 alone produces, z = a W  =>  kappa(z_i,z_i,z_j,z_j) contains
    sum_a W_ai^2 W_aj^2 kappa4(a_a), and kappa(z_i,z_i,z_i,z_j) contains
    sum_a W_ai^3 W_aj kappa4(a_a). Built from the TRUE post kappa4 of layer l-1.
    """
    mu = xp.asarray(mg["mu"][l])
    s = xp.sqrt(xp.asarray(mg["var"][l]))
    k4d = xp.asarray(mg["k4"][l])
    al = mu / s
    g = xp.asarray(gate_p[l])
    C = _f(t["_Cfull"], xp)
    rho = C / (s[:, None] * s[None, :])
    n = rho.shape[0]
    off = ~xp.eye(n, dtype=bool)
    si, sj = s[:, None], s[None, :]
    res = {}
    # kappa_iijj, normalised by s_i^2 s_j^2
    y = (_f(t["D22"], xp) / (si ** 2 * sj ** 2))[off]
    B = ((k4d[:, None] + k4d[None, :]) / 6.0 / (si ** 2 * sj ** 2))[off]
    r = rho[off]
    Wd = W.astype(xp.float64)
    gp = xp.asarray(k4post_prev)[:, None]
    W2 = Wd * Wd
    T22 = (W2.T @ (gp * W2)) / (si ** 2 * sj ** 2)
    T31 = ((W2 * Wd).T @ (gp * Wd)) / (si ** 3 * sj)
    X = xp.stack([B, r, r * r, (((g[:, None] + g[None, :]) / 2.0) * rho)[off],
                  (al[:, None] * al[None, :])[off], T22[off]], axis=1)
    res["k22"] = {"cols": ["B", "rho", "rho2", "grho", "aa", "Tdiag"],
                  "XtX": xp.asnumpy(X.T @ X).tolist(), "Xty": xp.asnumpy(X.T @ y).tolist(),
                  "sx": xp.asnumpy(X.sum(axis=0)).tolist(),
                  "yty": float(y @ y), "sy": float(y.sum()), "n": int(y.size)}
    # kappa_iiij, normalised by s_i^3 s_j
    y = (_f(t["Q31"], xp) / (si ** 3 * sj))[off]
    chain = (C / (si ** 3 * sj))[off]
    X = xp.stack([chain, r, r * r, (g[:, None] * rho)[off],
                  (al[:, None] * al[None, :])[off], T31[off]], axis=1)
    res["k31"] = {"cols": ["Cform", "rho", "rho2", "girho", "aa", "Tdiag"],
                  "XtX": xp.asnumpy(X.T @ X).tolist(), "Xty": xp.asnumpy(X.T @ y).tolist(),
                  "sx": xp.asnumpy(X.sum(axis=0)).tolist(),
                  "yty": float(y @ y), "sy": float(y.sum()), "n": int(y.size)}
    return res


# ------------------------------------------------------------------- arms

ONESTEP = {                      # substituted at layer l, mean read at l
    "base": ("mu",),
    "var": ("mu", "var"),
    "D3": ("mu", "D3"),
    "D4": ("mu", "dG", "D4"),
    "marg": ("mu", "var", "D3", "dG", "D4"),
}
TWOSTEP = {                      # substituted at l-1 (mu at l-1 and l), mean read at l
    "base2": ("mu",),
    "var": ("mu", "var"),
    "C11": ("mu", "C11"),
    "cov": ("mu", "var", "C11"),
    "D3": ("mu", "D3"),
    "D21": ("mu", "D21"),
    "D4": ("mu", "dG", "D4"),
    "D22": ("mu", "D22"),
    "Q31": ("mu", "Q31n"),
    "offdiag": ("mu", "C11", "D21", "D22", "Q31n"),
    "all": ("mu", "var", "C11", "D3", "D21", "dG", "D4", "D22", "Q31n"),
}
_NOMU = ("var", "C11", "D3", "D21", "dG", "D4", "D22", "Q31n")
ROLLED = {
    "mu": ("mu",), "var": ("var",), "C11": ("C11",), "cov": ("var", "C11"),
    "D3": ("D3",), "D21": ("D21",), "k3": ("D3", "D21"),
    "D4": ("dG", "D4"), "D22": ("D22",), "Q31": ("Q31n",), "k4": ("dG", "D4", "D22", "Q31n"),
    "nomu": _NOMU, "all": ("mu",) + _NOMU,
}
ROLLED_PREFIX_OBJS = ("cov", "D21", "k3", "k4", "nomu")
ROLLED_PREFIXES = (4, 8, 12)     # substitute at layers 1..P (0-based), free after
# M6.1 (HANDOFF 2026-10-11 Lane M6): substitute ONLY inside a band, free before and
# after. 0-based (lo, hi) inclusive = 1-based layers 2-6, 7-11, 12-15.
ROLLED_BAND_OBJS = ("var", "C11", "D3", "D21", "k4")
ROLLED_BANDS = ((1, 5), (6, 10), (11, 14))
ARCH_ARMS = {"T0": {"_SHARED_ON": False},
             "T0x": {"_SHARED_ON": False, "_P2_RANK": 0}}


def run_net(mod, est, ctl, wf, width, depth, truth, post_cov, mg, gate_p, gt, kp, xp,
            k4post=None, quick=False):
    ch = Chain(mod, est, wf, width, depth)
    # `predict` clamps at 0 (post-ReLU means are non-negative); so do we.
    host = lambda a: np.maximum(
        np.asarray(xp.asnumpy(a) if xp is not None else a, dtype=np.float64), 0.0)
    ctl.truth = truth
    res = {"objects": {}, "onestep": {k: {} for k in ONESTEP},
           "twostep": {k: {} for k in TWOSTEP}, "rolled": {}, "k4": {}, "arch": {}}

    def m2(l, mu):
        return mse_pair(host(mu), gt[l], kp[l])

    # ---- the main pass: objects, one-step and two-step branches ------------
    states = [ch.init_state()]      # states[l] = state after layer l-1
    mus = []
    for l in range(depth):
        if l >= 1 and not quick:
            for arm, names in ONESTEP.items():
                ctl.sub, ctl.rec = frozenset(names), None
                res["onestep"][arm][l] = m2(l, ch.step(l, states[l])[0])
            if l >= 2:
                for arm, names in TWOSTEP.items():
                    ctl.sub, ctl.rec = frozenset(names), None
                    s1 = ch.step(l - 1, states[l - 1])
                    ctl.sub = frozenset(("mu",))
                    res["twostep"][arm][l] = m2(l, ch.step(l, s1)[0])
                    s1 = None
        ctl.sub, ctl.rec = frozenset(), {}
        s = ch.step(l, states[l])
        rec = ctl.rec
        ctl.rec = None
        if l >= 1:
            res["objects"][l] = object_errors(rec, truth[l], post_cov[l],
                                              s[1] if l < depth - 1 else None, xp)
            if not quick and l < depth - 1:
                res["k4"][l] = k4_stats(l, truth[l], mg, gate_p, k4post[l - 1], wf[l], xp)
                # the (m/2) lam the chain's Q31 used at this layer (LAM of l-1)
                res["k4"][l]["lam_chain"] = 0.5 * METRIC * float(mod._own_lambda(l - 1, depth))
        rec = None
        mus.append(s[0])
        states.append(s)
        if len(states) > 3:
            states[-4] = None       # keep only what the branches read
    res["chain"] = {l: m2(l, mus[l]) for l in range(1, depth)}
    pred_chain = [host(m) for m in mus]
    states = mus = None

    # ---- rolled substitution -----------------------------------------------
    def rolled(names, P, lo=1):
        st = ch.init_state()
        per = {}
        for l in range(depth):
            ctl.sub = frozenset(names) if lo <= l <= P else frozenset()
            st = ch.step(l, st)
            if l >= 1:
                per[l] = m2(l, st[0])
        ctl.sub = frozenset()
        return per

    if not quick:
        for obj, names in ROLLED.items():
            res["rolled"][f"{obj}@{depth - 1}"] = rolled(names, depth - 1)
        for obj in ROLLED_PREFIX_OBJS:
            for P in ROLLED_PREFIXES:
                res["rolled"][f"{obj}@{P}"] = rolled(ROLLED[obj], P)
        for obj in ROLLED_BAND_OBJS:
            for lo, hi in ROLLED_BANDS:
                res["rolled"][f"{obj}@{lo}-{hi}"] = rolled(ROLLED[obj], hi, lo)

    # ---- architecture arms: object errors only -----------------------------
    for arm, over in ({} if quick else ARCH_ARMS).items():
        saved = {k: getattr(mod, k) for k in over}
        try:
            for k, v in over.items():
                setattr(mod, k, v)
            # diagnostic arms: FLOPs are irrelevant, so never let the budget
            # guard trim the tower (T0x would otherwise be capped at 10 gens)
            cha = Chain(mod, est, wf, width, depth, budget=BUDGET * 64)
            st = cha.init_state()
            ob, fin = {}, {}
            for l in range(depth):
                ctl.sub, ctl.rec = frozenset(), {}
                st = cha.step(l, st)
                if l >= 1:
                    ob[l] = object_errors(ctl.rec, truth[l], post_cov[l],
                                          st[1] if l < depth - 1 else None, xp)
                    fin[l] = m2(l, st[0])
                ctl.rec = None
            res["arch"][arm] = {"objects": ob, "chain": fin}
        except GenCap as e:
            res["arch"][arm] = {"skipped": str(e)}
        finally:
            for k, v in saved.items():
                setattr(mod, k, v)
            ctl.rec = None
    return res, pred_chain


# --------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("candidate", nargs="?")
    ap.add_argument("--panel", default="fit")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--quick", action="store_true", help="main pass only (timing)")
    ap.add_argument("--report", default="")
    a = ap.parse_args(argv)
    if a.report:
        from anatomy2_report import report
        print(report(Path(a.report)))
        return 0

    import panels
    import real_data as rd

    xp = None
    if BACKEND == "cupy":
        import cupy as xp
        xp.get_default_memory_pool().set_limit(
            size=int(float(os.environ.get("PROBE_GPU_LIMIT_GIB", "24")) * 2 ** 30))

    cand = Path(a.candidate)
    if not cand.is_absolute():
        cand = ROOT / cand
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    patched = out.parent / f"_patched_{cand.stem}.py"
    patched.write_text(patch_source(cand.read_text(encoding="utf-8")),
                       encoding="utf-8", newline="\n")
    mod = load_module(patched, "_anatomy2_candidate")
    ctl = Ctl(xp)
    mod._A2 = ctl
    est = mod.Estimator()

    rows = panels.nets(a.panel)[a.lo:a.lo + a.n]

    def load(row):
        split, idx, seed, _nm = row
        w, tr, _ = rd.get(idx, split, dtype=np.float32)
        depth, width = w.shape[0], w.shape[1]
        wf = []
        for x in w:
            arr = np.asarray(x, dtype=np.float64)
            arr.setflags(write=False)
            wf.append(xp.asarray(arr) if xp is not None else arr)
        return idx, wf, width, depth, int(seed), np.asarray(tr, dtype=np.float64)

    if a.verify:
        base = load_module(cand, "_anatomy2_base")
        best = base.Estimator()
        idx, wf, width, depth, seed, tr = load(rows[0])
        best.setup(_Ctx(width, depth, BUDGET, 0))
        est.setup(_Ctx(width, depth, BUDGET, 0))
        ref = best.predict(_MLP(wf, width, depth, seed), BUDGET)
        ref = np.asarray(xp.asnumpy(ref) if xp is not None else ref)
        # (a) this file's loop on the patched module, nothing substituted
        w32 = [x.astype(mod.fnp.float32) for x in wf]
        ch = Chain(mod, est, w32, width, depth)
        ctl.sub, ctl.rec = frozenset(), None
        st, mine = ch.init_state(), []
        for l in range(depth):
            st = ch.step(l, st)
            mine.append(np.asarray(xp.asnumpy(st[0]) if xp is not None else st[0]))
        a_ok = all(np.array_equal(np.maximum(mine[l].astype(np.float64), 0.0), ref[l])
                   for l in range(1, depth))
        # (b) recording on, still nothing substituted
        ctl.rec = {}
        st = ch.init_state()
        for l in range(depth):
            st = ch.step(l, st)
        ctl.rec = None
        fin_b = np.asarray(xp.asnumpy(st[0]) if xp is not None else st[0])
        b_ok = np.array_equal(np.maximum(fin_b.astype(np.float64), 0.0), ref[depth - 1])
        # (c) substituting every object with the chain's OWN recorded values
        #     must also be the identity -- proves the substitution plumbing
        own = {}
        st = ch.init_state()
        for l in range(depth):
            ctl.rec = {}
            st = ch.step(l, st)
            own[l] = dict(ctl.rec)
        ctl.rec = None
        ctl.truth = own
        st = ch.init_state()
        for l in range(depth):
            ctl.sub = frozenset(k if k != "Q31" else "Q31n" for k in own[l])
            st = ch.step(l, st)
        ctl.sub = frozenset()
        fin_c = np.asarray(xp.asnumpy(st[0]) if xp is not None else st[0])
        c_err = float(np.max(np.abs(np.maximum(fin_c.astype(np.float64), 0.0) - ref[depth - 1])))
        rep = {"net": idx, "loop_bit_identical": bool(a_ok),
               "recording_bit_identical": bool(b_ok),
               "self_substitution_max_abs": c_err,
               "ref_final_mse_vs_split_truth": float(((ref[depth - 1] - tr[depth - 1]) ** 2).mean())}
        print(json.dumps(rep, indent=1))
        out.write_text(json.dumps(rep, indent=1))
        return 0 if (a_ok and b_ok and c_err < 1e-5) else 1

    res = {"candidate": cand.name, "panel": a.panel, "backend": BACKEND, "nets": {}}
    if out.exists():
        res = json.loads(out.read_text(encoding="utf-8"))
    setup_done = False
    progress = os.environ.get("WORKERQ_PROGRESS")
    t0 = time.time()
    for i, row in enumerate(rows):
        key = str(row[1])
        if key in res["nets"]:
            continue
        idx, wf, width, depth, seed, tr = load(row)
        if not setup_done:
            est.setup(_Ctx(width, depth, BUDGET, 0))
            setup_done = True
        w32 = [x.astype(mod.fnp.float32) for x in wf]
        L, mg, gate_p, truth, post_cov = load_truth(idx, xp, mod.fnp.float32)
        assert np.allclose(L.gt_mean, tr, atol=1e-6), "label gt_mean != split truth"
        kp = L.post_mean()
        t1 = time.time()
        r, _pred = run_net(mod, est, ctl, w32, width, depth, truth, post_cov, mg, gate_p,
                           tr, kp, xp, k4post=L.post_k4(), quick=a.quick)
        r["name"] = L.name
        r["seconds"] = round(time.time() - t1, 1)
        res["nets"][key] = r
        out.write_text(json.dumps(res), encoding="utf-8")
        print(f"[anatomy2] net {idx} {L.name}: final chain MSE "
              f"{r['chain'][depth - 1][0]:.4e} (x-est {r['chain'][depth - 1][1]:.4e}) "
              f"in {r['seconds']} s, total {time.time() - t0:.0f} s", flush=True)
        truth = post_cov = L = None
        if xp is not None:
            xp.get_default_memory_pool().free_all_blocks()
        if progress:
            try:
                Path(progress).write_text(str((i + 1) / len(rows)))
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
