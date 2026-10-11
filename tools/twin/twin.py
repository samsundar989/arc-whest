"""The differentiable twin: the SHIPPED chain source, run on torch, trained end to end.

    .venv-gpu/Scripts/python.exe tools/twin/twin.py parity   [--cand F] [--nets 0,1,2]
    .venv-gpu/Scripts/python.exe tools/twin/twin.py eval     --panel dev|hold100|train [--params P]
    .venv-gpu/Scripts/python.exe tools/twin/twin.py train    --stage S1 --out DIR [--epochs 2]

HANDOFF 2026-09-25 Step 2 / PLAN rev 13 §3 Step 2.

WHY NOT A RE-IMPLEMENTATION. The handoff asks for a PyTorch re-implementation of
v59's forward math and accepts it at 1e-4 relative on per-net MSE. Rather than
write the Wick program, closure tables and tower a second time -- which is how a
twin silently stops being a twin -- this runs the estimator file ITSELF with
`flopscope` replaced by a torch shim (`tools/twin/shim`). Every array is a
`torch.Tensor`, so autograd flows through exactly the arithmetic that ships:
Strassen products, `_advance`, `_dslice3/21`, `_shared_join_lean`, hubs, P2
compress (`torch.linalg.qr`), kappa4 regeneration, the corrector pages. Parity is
then a float32 round-off question, checked by `parity` against the cupy probe.

Memory: one `torch.utils.checkpoint` per layer (non-reentrant), one net per
forward, gradients accumulated over `--accum` nets per optimiser step.

PARAMETERS are a flat dict of leaf tensors (`Params`) so any subset can be frozen:
  S1  corrector pages W1,b1,W2,b2,W3,b3 (the standardisers fmu/fisd stay frozen)
  (S2+ -- lambda/theta/feed profiles, term gains, multi-mode kappa4, sigma page --
  are added by patching the module's scalar hooks; see `Params.install`.)

Objective: mean final-layer MSE over TRAIN nets, + `--w-all` x mean MSE of rows
1..depth-2 (row 0 is an exact closed form in `predict`; the last row is the
objective). Outputs are clamped at 0 as `predict` does.

Panels (`data/panels/twin_train_split.json`): TRAIN = full rows 0..899, HOLDOUT-100
= rows 900..999, DEV = mini 0..99 (never trained on, read for selection).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHIM = ROOT / "tools" / "twin" / "shim"
sys.path.insert(0, str(SHIM))                     # flopscope -> torch (this process only)
sys.path.append(str(ROOT / "local_sim"))          # whestbench shim (MLP, BaseEstimator)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

import flopscope.numpy as fnp  # noqa: E402  (the torch shim)

assert "twin" in fnp.__file__, f"wrong flopscope on the path: {fnp.__file__}"
torch.backends.cuda.matmul.allow_tf32 = False     # float32 means float32
torch.backends.cudnn.allow_tf32 = False

BUDGET = 2 ** 41
DEV = torch.device(os.environ.get("TWIN_DEVICE") or "cuda")   # TWIN_DEVICE=cpu: CPU-only jobs
CACHE = Path(os.environ.get("TWIN_CACHE") or ROOT / "cache" / "twin")   # worktrees: point at the main checkout
#: keenanpepper/arc-whestbench-p2-d8b-corpus-14k shards 00..15 (nets 0..2047, MC N = 1e8,
#: noise ~9e-10 at the final layer): per-layer post-ReLU means, (2048, 16, 1024) float32,
#: row = net idx. Built by `twin.py d8bprep`; weights regenerate from seed 770000 + idx
#: (checked 2026-09-26 against the exact layer-0 closed form: MSE 7e-9 = the N = 1e8 floor,
#: vs 3e-4 for the wrong net, on idx 0, 1, 2, 777, 2047).
D8B_MEAN = CACHE.parent / "d8b" / "d8b_mean.npy"
DEFAULT_CAND = ROOT / "estimator_k3_v59_corr400.py"
#: Stage -> trainable leaves, as "group/name" over the chain's constant groups.
TRAINABLE = {
    "S1": tuple(f"corr/{k}" for k in ("W1", "b1", "W2", "b2", "W3", "b3")),
    "S2": tuple(f"corr/{k}" for k in ("W1", "b1", "W2", "b2", "W3", "b3"))
    + ("gain/lam", "gain/theta", "gain/feed"),
    "S2t": tuple(f"corr/{k}" for k in ("W1", "b1", "W2", "b2", "W3", "b3"))
    + ("gain/lam", "gain/theta", "gain/feed", "gain/tt", "gain/t1", "gain/cu", "gain/cb"),
    "S3": tuple(f"corr/{k}" for k in ("W1", "b1", "W2", "b2", "W3", "b3"))
    + ("gain/lam", "gain/theta", "gain/feed", "gain/tt", "gain/t1", "gain/cu", "gain/cb",
       "gain/q31"),
    "S4": tuple(f"sig/{k}" for k in ("S1", "c1", "S2", "c2")),
    # Step 3.3 (HANDOFF 09-27): the S3 leaves + the calibrated-closure nets (tools/build_clos.py)
    "S5": tuple(f"corr/{k}" for k in ("W1", "b1", "W2", "b2", "W3", "b3"))
    + ("gain/lam", "gain/theta", "gain/feed", "gain/tt", "gain/t1", "gain/cu", "gain/cb",
       "gain/q31")
    + tuple(f"clos/{k}" for k in ("uW1", "ub1", "uW2", "ub2", "uW3", "ub3",
                                  "bW1", "bb1", "bW2", "bb2")),
    "S5c": tuple(f"clos/{k}" for k in ("uW1", "ub1", "uW2", "ub2", "uW3", "ub3",
                                       "bW1", "bb1", "bW2", "bb2")),
    "S14": tuple(f"corr/{k}" for k in ("W1", "b1", "W2", "b2", "W3", "b3"))
    + tuple(f"sig/{k}" for k in ("S1", "c1", "S2", "c2")),
}
# 2026-09-27 gates (tools/build_gate.py, experiments/2026-09-27_gates/DESIGN.md): the
# state-dependent factor gates live in the candidate's `const["gate"]` group (identity at
# W2 = b2 = 0). Birth (G1b*) and transport (G1t*) arms share keys; the candidate file
# decides the arm. G1b/G1t = the S3 leaves + gates (compare with the S3b continued-training
# control); G1bg/G1tg = gates only (the S3 leaves frozen at --init). Additive: no existing
# stage reads the "gate" group, and a candidate without it has no `const["gate"]`.
_GATE_LEAVES = tuple(f"gate/{k}" for k in ("W1", "b1", "W2", "b2"))
TRAINABLE.update({"G1b": TRAINABLE["S3"] + _GATE_LEAVES, "G1t": TRAINABLE["S3"] + _GATE_LEAVES,
                  "G1bg": _GATE_LEAVES, "G1tg": _GATE_LEAVES})
# 2026-10-07 Lane P2: the gentle refit stage (S2t) plus the gain-mode coefficients.
TRAINABLE["S2tgb"] = TRAINABLE["S2t"] + ("gain/gmb",)
TRAINABLE["S2tgg"] = TRAINABLE["S2t"] + ("gain/gmg",)
# 2026-10-06 Lane M4: the neighbor head (`--arm nbr`) alone; everything else frozen at --init.
TRAINABLE["N1"] = tuple(f"nbr/{k}" for k in ("W1", "b1", "W2", "b2"))


# --------------------------------------------------------------------- data

class Data:
    def __init__(self):
        self.w, self.truth, self.meta = {}, {}, {}
        for split, n in (("full", 1000), ("mini", 100)):
            p = CACHE / f"{split}_w.f32"
            if p.exists():
                # a path, not a memmap: mapped pages would count against this
                # process's working set across 900 nets (see export_weights.py)
                self.w[split] = p
                self.truth[split] = np.load(CACHE / f"{split}_truth.npy")
                self.meta[split] = json.loads((CACHE / f"{split}_meta.json").read_text())
        if D8B_MEAN.exists():
            self.truth["d8b"] = np.load(D8B_MEAN)

    def net(self, split, i):
        if split == "d8b":                          # weights regenerate from the card's seed rule
            g = torch.Generator(device="cpu").manual_seed(770000 + i)
            a = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * (2.0 / 1024) ** 0.5
            return [a[l].to(DEV) for l in range(16)], self.truth["d8b"][i]
        if not self.meta[split]["done"][i]:
            raise RuntimeError(f"{split} row {i} not exported yet")
        per_net = 16 * 1024 * 1024                  # float32 elements in one net
        a = np.fromfile(self.w[split], dtype=np.float32, count=per_net,
                        offset=i * per_net * 4).reshape(16, 1024, 1024)
        w = [torch.from_numpy(a[l].copy()).to(DEV) for l in range(16)]
        return w, self.truth[split][i]


def panel(name):
    s = json.loads((ROOT / "data" / "panels" / "twin_train_split.json").read_text())
    if name == "train":
        return [("full", i) for i in s["train"]["indices"]]
    if name == "hold100":
        return [("full", i) for i in s["holdout_100"]["indices"]]
    if name == "dev":
        return [("mini", i) for i in range(100)]
    if name == "d8b":
        return [("d8b", i) for i in range(2048)]
    if name == "train+d8b":
        return panel("train") + panel("d8b")
    if name == "fit80":                              # Step 2 refit TRAIN (labels exist)
        return [("full", i) for i in range(80)]
    if name == "fit20h":                             # Step 2 refit HOLDOUT-20
        return [("full", i) for i in range(80, 100)]
    if name.startswith("fit"):
        return [("full", i) for i in range(int(name[3:] or 100))]
    raise SystemExit(f"unknown panel {name}")


# -------------------------------------------------------------------- model

class _Ctx:
    def __init__(self, width, depth, budget, seed):
        self.width, self.depth = width, depth
        self.flop_budget, self.seed = budget, seed
        self.api_version, self.scratch_dir, self.submission_dir = "1.0", None, None


#: S2 profile hooks. The shipped file turns lambda into a Python float at two
#: sites, which would cut the gradient; the twin's COPY of the source replaces
#: them (and routes theta / feed through per-layer gain tensors). The exported
#: estimator keeps plain constants -- `tools/build_twin.py` bakes the gains in.
TWIN_PATCHES = [
    ('"lam": float(_lam_prev), "metric": metric})',
     '"lam": _lam_prev, "metric": metric})'),
    ("_lam_h = float(_own_lambda(_li_h, depth))",
     "_lam_h = _own_lambda(_li_h, depth)"),
    ('            _sf = _OWN_S_FEED * _prof_at("feed", layer_index, depth)\n'
     '            if _sf != 1.0:\n',
     '            _sf = _OWN_S_FEED * _prof_at("feed", layer_index, depth)\n'
     '            if _TWG is not None:\n'
     '                _sf = _sf * _TWG["feed"][layer_index]\n'
     '            if _TWG is not None or _sf != 1.0:\n'),
    ('def _own_extra(x):\n    th = _OWN_THETA_FAM * _prof("theta")\n',
     '_TWG = None\n\n\ndef _own_extra(x):\n    th = _OWN_THETA_FAM * _prof("theta")\n'
     '    if _TWG is not None:\n        return x * (th * _TWG["theta"][_CUR_LAYER])\n'),
    # --- S2t: per-TERM gains (HANDOFF 09-25 Step 2 (iii)). Each is live only
    # when its key is installed, so S1/S2 runs are unaffected.
    # (a) the bivariate closure-table terms, by row index in table["terms"]
    ('    for row in table["terms"]:\n',
     '    for _ridx, row in enumerate(table["terms"]):\n'),
    ("        pK[ip] = term if ip not in pK else pK[ip] + term\n",
     '        if _TWG is not None and "tt" in _TWG:\n'
     '            term = term * _TWG["tt"][layer_index, _ridx]\n'
     "        pK[ip] = term if ip not in pK else pK[ip] + term\n"),
    # (b) the 1-D fast path: kappa3 / kappa4 contributions per output order p
    ('        W4 = fnp.stack([w[kp] for kp in prog["w4_keys"]])\n',
     '        W4 = fnp.stack([w[kp] for kp in prog["w4_keys"]])\n'
     '        if _TWG is not None and "t1" in _TWG:\n'
     '            W3 = W3 * _TWG["t1"][layer_index, 0][:, None]\n'
     '            W4 = W4 * _TWG["t1"][layer_index, 1][:, None]\n'),
    # (c) closure univariate extras, per (term, p)
    ("        cvec = fnp.array([c for fs, k, c in sel], dtype=dtype)\n",
     "        cvec = fnp.array([c for fs, k, c in sel], dtype=dtype)\n"
     '        if _TWG is not None and "cu" in _TWG:\n'
     '            cvec = cvec * _TWG["cu"][_CUR_LAYER, [_u for _u, (_f, _k, _c) in enumerate(prog["uni"]) if str(p) in _c], p - 1]\n'),
    # (e) S3: multi-mode kappa4_iiij. Anatomy 2 §2: the chain's (m/2) lam C form
    # explains R^2 ~0.48 of the true slice; {rho, rho^2, Phi(alpha_i) rho,
    # alpha_i alpha_j, 1} lifts it to ~0.58. Added to the NORMALISED Q31
    # (kappa_iiij / s_i^3 s_j), which is exactly what the closure consumes.
    ('        nb["Q31"] = Cn * inv2[:, None] * (0.5 * blocks["metric"] * blocks["lam"])\n',
     '        nb["Q31"] = Cn * inv2[:, None] * (0.5 * blocks["metric"] * blocks["lam"])\n'
     '        if _TWG is not None and "q31" in _TWG:\n'
     '            _q = _TWG["q31"][_CUR_LAYER]\n'
     '            _gi = flops.stats.norm.cdf(alpha)[:, None]\n'
     '            nb["Q31"] = (nb["Q31"] + Cn * _q[0] + (Cn * Cn) * _q[1] + (_gi * Cn) * _q[2]\n'
     '                         + (alpha[:, None] * alpha[None, :]) * _q[3] + _q[4])\n'),
    # (d) closure bivariate extras, per term
    ("        Rj = fnp.stack([H[row[(kj, out[1])]] * c for i, ki, kj, c in sel])\n",
     "        Rj = fnp.stack([H[row[(kj, out[1])]] * c for i, ki, kj, c in sel])\n"
     '        if _TWG is not None and "cb" in _TWG:\n'
     '            Rj = Rj * _TWG["cb"][_CUR_LAYER, [i for i, *_ in sel]][:, None]\n'),
]


#: HANDOFF 09-27 Step 2 (refit ceilings): label-substitution hooks, the SAME anchors and
#: object names as `tools/anatomy2.py` PATCHES (copied, not imported: importing anatomy2
#: would put its numpy/cupy shims in front of this process's torch shim). Inert while
#: `_A2` is None, so every non-substituting run is unchanged.
SUB_PATCHES = [
    ("_CORR_HOOK = None\n",
     "_CORR_HOOK = None\n\n_A2 = None\n\n\ndef _a2(name, value, *aux):\n"
     "    if _A2 is None:\n        return value\n"
     "    return _A2(_CUR_LAYER, name, value, *aux)\n"),
    ("    metric = float(init_scale)\n",
     "    var = _a2('var', var)\n    metric = float(init_scale)\n"),
    ("    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n",
     "    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n"
     "    D3 = _a2('D3', D3)\n"),
    ("    dG = fnp.where(fnp.isfinite(dG), dG, dtype(0.0))\n",
     "    dG = fnp.where(fnp.isfinite(dG), dG, dtype(0.0))\n    dG = _a2('dG', dG)\n"),
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
    # Step 3 closure bench (tools/closure_bench.py): mu, and a read-only tap on K
    ("    mu_pre = WT @ mu\n", "    mu_pre = WT @ mu\n    mu_pre = _a2('mu', mu_pre)\n"),
    ("    mu_out = K[(1,)]\n", "    _a2('K', K)\n    mu_out = K[(1,)]\n"),
]
#: Step 2 channels. "all" is every object but mu: substituting mu_pre at every layer
#: makes the final mean nearly exact by construction, which leaves nothing to refit.
SUBST = {"cov": ("var", "C11"), "k3": ("D3", "D21"), "k4": ("dG", "D22", "Q31n"),
         "all": ("var", "C11", "D3", "D21", "dG", "D22", "Q31n")}
LABELS = CACHE.parent / "twin_labels"


class Subst:
    """`_A2` for one net: returns the exact label of every object in `names`, layers >= 1."""

    def __init__(self, gidx, names):
        z = np.load(LABELS / f"fit_{gidx:03d}.npz")
        keys = {("Q31" if k == "Q31n" else k) for k in names}
        self.t = {k: torch.from_numpy(np.asarray(z[k])).to(DEV) for k in keys}
        self.names = frozenset(names)

    def __call__(self, l, name, value, *aux):
        if name not in self.names or l < 1:
            return value
        if name == "Q31n":                          # anatomy2 Ctl: raw kappa_iiij, normalised
            inv, inv2 = aux                         # by the chain's own sigma
            return self.t["Q31"][l] * ((inv2 * inv)[:, None] * inv[None, :])
        return self.t[name][l].to(value.dtype)


#: Optional ARMS: source changes tested through the twin before any estimator
#: file is built (PLAN rev 13 §3 Step 3: "test first").
ARMS = {
    # cov_out from the CORRECTED mean: K2 = pK2 - pK1^2 with pK1 -> pK1 + delta,
    # i.e. diag(cov_out) -= 2 delta pK1 + delta^2. K11 has no pK1 term.
    "pk1cov": [
        ('    if _CORR_ON and const.get("corr") is not None:\n',
         '    _mu_unc = mu_out\n    if _CORR_ON and const.get("corr") is not None:\n'),
        ("    cov_out = fnp.diag(K[(2,)]) + K[(1, 1)]\n",
         "    cov_out = fnp.diag(K[(2,)]) + K[(1, 1)]\n"
         "    _dl = mu_out - _mu_unc\n"
         "    cov_out = cov_out - fnp.diag(_dl * _mu_unc * 2.0 + _dl * _dl)\n"),
    ],
    # HANDOFF 2026-10-06 Lane P, P1: decided-neuron masks on the v73 kappa3 path,
    # accuracy only (nothing is compacted, so no FLOP is saved here). Inert while
    # `_MASK` is None. alpha_l = mu_pre / sigma at the layer the mask is read.
    #   R: rows i with alpha_l,i < -tau_R of every carried leg, after the Wick scale
    #      (inherited: before the join, so Q is masked too) and of the newborn legs.
    #      The newborn M = I is implicit and left alone (its transport is free).
    #   H: newborn two-leg + hub columns j with |alpha_b,j| > tau_H (A_L[:, j], w2b_j,
    #      cA_j, cB_j; e_b_j rides on A_L so its separable S21 row moves into the P2
    #      residual); slice columns (sb_j, residual row j) only where alpha_b,j < -tau_H.
    #   D: rows and columns i of D21 with alpha_l,i < -tau_D (no D21 at the last layer).
    # mu, C and D3 are never masked.
    "mask": [
        ("def _layer(W, state, table, dtype, const, init_scale, last_layer, gens,\n",
         "_MASK = None\n\n\n"
         "def _mask_keep(alpha, tau, mode, dtype):\n"
         "    if mode == 'dead':\n"
         "        return fnp.where(alpha < -tau, dtype(0.0), dtype(1.0))\n"
         "    return fnp.where(alpha * alpha > tau * tau, dtype(0.0), dtype(1.0))\n\n\n"
         "def _mask_rows(tw, k):\n"
         "    c = k[:, None]\n"
         "    tw.A_L = tw.A_L * c\n"
         "    for _a in ('M', 'Z', 'Y', 'Q', 'F'):\n"
         "        if getattr(tw, _a) is not None:\n"
         "            setattr(tw, _a, getattr(tw, _a) * c)\n"
         "    tw._legs = tw._p2 = tw._hub = tw._p3 = None\n"
         "    return tw\n\n\n"
         "def _mask_rows_new(tw, k, n):\n"
         "    c = k[:, None]\n"
         "    tw.A_L = fnp.concatenate([tw.A_L[:, :-n], tw.A_L[:, -n:] * c], axis=1)\n"
         "    r = tw.L.shape[2]\n"
         "    tw.Z = fnp.concatenate([tw.Z[:, :-r], tw.Z[:, -r:] * c], axis=1)\n"
         "    if tw.Y is not None:\n"
         "        tw.Y = fnp.concatenate([tw.Y[:, :-2], tw.Y[:, -2:] * c], axis=1)\n"
         "    tw._legs = tw._p2 = tw._hub = tw._p3 = None\n"
         "    return tw\n\n\n"
         "def _layer(W, state, table, dtype, const, init_scale, last_layer, gens,\n"),
        ('    w = _wick(mu_pre, sigma, alpha, table, dtype, const["wick"])\n',
         '    w = _wick(mu_pre, sigma, alpha, table, dtype, const["wick"])\n'
         "    _mR = _mH = _mS = _mD = None\n"
         "    _kz = 1.0\n"
         "    if _MASK is not None and not last_layer:\n"
         "        if _MASK.get('R'):\n"
         "            _mR = _mask_keep(alpha, _MASK['R'], 'dead', dtype)\n"
         "        if _MASK.get('H'):\n"
         "            _mH = _mask_keep(alpha, _MASK['H'], 'abs', dtype)\n"
         "            _mS = _mask_keep(alpha, _MASK['H'], 'dead', dtype)\n"
         "        if _MASK.get('D'):\n"
         "            _mD = _mask_keep(alpha, _MASK['D'], 'dead', dtype)\n"
         "    if _MASK is not None and _MASK.get('Ktab') and not last_layer:\n"
         "        _rk = fnp.argsort(fnp.argsort(alpha))\n"
         "        for _role in ('R', 'H', 'D'):\n"
         "            _kt = _MASK['Ktab'].get(_role)\n"
         "            if _kt is None or not _MASK.get(_role):\n"
         "                continue\n"
         "            _k = int(_kt[layer_index]) if layer_index < len(_kt) else n\n"
         "            _m = fnp.where(_rk >= n - _k, dtype(1.0), dtype(0.0))\n"
         "            if _role == 'H' and _MASK.get('Hrank'):\n"
         "                # P6.1: keep the K_H SMALLEST scores (abs: alpha^2; asym a,b: max(-alpha/a, alpha/b)),\n"
         "                # as the estimator's `_kh = argsort(score)[:K_H]` -- drops always-on columns too\n"
         "                _hr = _MASK['Hrank']\n"
         "                _sc = (alpha * alpha if _hr[0] == 'abs' else\n"
         "                       fnp.maximum(alpha * dtype(-1.0 / _hr[1]), alpha * dtype(1.0 / _hr[2])))\n"
         "                _m = fnp.where(fnp.argsort(fnp.argsort(_sc)) < _k, dtype(1.0), dtype(0.0))\n"
         "            if _role == 'R':\n"
         "                _mR = _m\n"
         "            elif _role == 'H':\n"
         "                _mH = _mS = _m\n"
         "            else:\n"
         "                _mD = _m\n"
         "    if _MASK is not None and not last_layer:\n"
         "        _MASK['_prevR'] = _mR\n"
         "        _MASK['_prevD'] = _mD\n"
         "    _mon = _mDd = None\n"
         "    if _MASK is not None and _MASK.get('Da') and not last_layer:\n"
         "        _mon = fnp.where(alpha > _MASK['Da'], dtype(0.0), dtype(1.0))\n"
         "        _mDd = _mD\n"
         "        _mD = _mon if _mD is None else _mD * _mon\n"
         "    _mA = _mW = _mC = _mSl = None\n"
         "    if _mH is not None:\n"
         "        _hm = _MASK.get('Hmode', '')\n"
         "        _mA = {'k': _mS, 'kh': _mS, 'd': _mS, 'w': None, 's': None}.get(_hm, _mH)\n"
         "        _mW = {'d': _mS, 's': None}.get(_hm, _mH)\n"
         "        _mC = {'kh': _mS, 'd': _mS, 'w': None, 's': None}.get(_hm, _mH)\n"
         "        _mSl = {'w': None}.get(_hm, _mS)\n"
         "        if _MASK.get('Hfull'):\n"
         "            _mSl = _mH       # P6.1: residual rows + S3 dropped on every dropped column (as shipped)\n"
         "    if _MASK is not None and _MASK.get('stats') is not None:\n"
         "        _st = {'l': layer_index}\n"
         "        for _t in (2.0, 2.5, 3.0, 3.5):\n"
         "            _st['dead%.1f' % _t] = float((alpha < -_t).sum()) / n\n"
         "            _st['dec%.1f' % _t] = float((alpha * alpha > _t * _t).sum()) / n\n"
         "        for _t in (1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0):\n"
         "            _st['dead%.2f' % _t] = float((alpha < -_t).sum()) / n\n"
         "        _MASK['stats'].append(_st)\n"),
        # P5.2 (ii)+(iii): `Dleg` -- the transported legs keep only D-set rows, so D3, the
        # star-D4 part and both sides of D21 vanish on dead rows (the compact chain never
        # computes them). Applied before the readouts; the R mask (tau 2.0, inside D) follows.
        ("    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n",
         "    if _mD is not None and _MASK.get('Dleg') and adv is not None:\n"
         "        adv = _mask_rows(adv, _mD)\n"
         "    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n"),
        ("        D22 = ((metric / 6.0) * (dG[:, None] + dG[None, :])) * off_mask\n",
         "        _D21c = _D21f = D21\n"
         "        if _mD is not None and _MASK.get('stats') is not None and _MASK['stats']:\n"
         "            # P5.2 rho: d21e energy over all vs kept columns, on kept rows\n"
         "            _iw = _mD * (1.0 / (var * var))\n"
         "            _rs = D21 * D21\n"
         "            _MASK['stats'][-1]['e_all'] = float((_rs.sum(axis=1) * _iw).sum())\n"
         "            _MASK['stats'][-1]['e_kept'] = float(((_rs * _mD[None, :]).sum(axis=1) * _iw).sum())\n"
         "        if _mD is not None and _MASK.get('Drows'):\n"
         "            D21 = D21 * _mD[:, None]\n"
         "        elif _mD is not None:\n"
         "            D21 = D21 * (_mD[:, None] * _mD[None, :])\n"
         "        if _mD is not None and not _MASK.get('Dfull'):\n"
         "            _D21c = D21\n"
         "        if _mD is not None and _MASK.get('rho') and not _MASK.get('Dfull'):\n"
         "            _rho = _MASK['rho']\n"
         "            _D21c = _D21c * (float(_rho[layer_index]) ** 0.5 if layer_index < len(_rho) else 1.0)\n"
         "        if _mD is not None and _MASK.get('Dzero'):\n"
         "            _kz = _mD\n"
         "        if _mon is not None and _MASK.get('Daz', 'z') != 'z':\n"
         "            # always-on rows keep their corrector delta; its d21e feature reads the\n"
         "            # full row ('f', diagnostic) or the row's own COLUMN ('c', servable)\n"
         "            _kz = 1.0 if _mDd is None else _mDd\n"
         "            _off = 1.0 - _mon\n"
         "            if _MASK['Daz'] == 'f':\n"
         "                _D21c = D21 + _D21f * _off[:, None]\n"
         "            else:\n"
         "                _D21c = D21 + D21.T * _off[:, None]\n"
         "        D22 = ((metric / 6.0) * (dG[:, None] + dG[None, :])) * off_mask\n"),
        # Dz: the corrector's delta is zeroed on the D-dropped rows
        ("        mu_out = mu_out + _corr_delta(\n",
         "        mu_out = mu_out + _kz * _corr_delta(\n"),
        # the corrector's d21e feature reads _D21c: masked D21 unless `Dc` (diagnostic)
        ('            layer_index, const["corr"], mu_pre, sigma, D3, D4, W,\n'
         "            None if last_layer else D21, mu, mu_out, n, dtype)\n",
         '            layer_index, const["corr"], mu_pre, sigma, D3, D4, W,\n'
         "            None if last_layer else _D21c, mu, mu_out, n, dtype)\n"),
        ("        fac1 = w1e[:, None] * C11\n",
         "        fac1 = w1e[:, None] * C11\n"
         "        if _mA is not None:\n"
         "            fac1 = fac1 * _mA[None, :]\n"
         "        if _mW is not None:\n"
         "            w2m = w2m * _mW\n"),
        ("        S_sep = (e_b[:, None] * C11) * w1e[None, :]\n",
         "        S_sep = (e_b[:, None] * C11) * w1e[None, :]\n"
         "        if _mA is not None:\n"
         "            S_sep = S_sep * _mA[:, None]\n"),
        ("            resid = S21 - S_sep\n",
         "            if _mC is not None:\n"
         "                cA = cA * _mC\n"
         "                cB = cB * _mC\n"
         "            resid = S21 - S_sep\n"),
        ("            if _PROTECT_FEED:\n",
         "            if _mA is not None:\n"
         "                e_b = e_b * _mA\n"
         "            if _mSl is not None:\n"
         "                resid = resid * _mSl[:, None]\n"
         "                S3 = S3 * _mSl\n"
         "            if _PROTECT_FEED:\n"),
        ("        adv = _wick_tower(w1t[:, None], adv)\n",
         "        adv = _wick_tower(w1t[:, None], adv)\n"
         "        if _mR is not None and _MASK.get('Rpart', '') != 'n':\n"
         "            adv = _mask_rows(adv, _mR)\n"),
        # Rpart '' both, 'i' inherited legs only (post-Wick), 'n' newborn A_L/Z/Y rows only
        # P5.1 item 1 (key J): no old-tier join at the last J non-final layers
        ("            adv = _shared_join_lean(adv)\n",
         "            if _MASK is None or not _MASK.get('J') or layer_index < depth - 1 - int(_MASK['J']):\n"
         "                adv = _shared_join_lean(adv)\n"),
        ("    out_tw = _truncate(out_tw, gens, n)\n",
         "    out_tw = _truncate(out_tw, gens, n)\n"
         "    if _mR is not None and _MASK.get('Rpart', '') == '':\n"
         "        out_tw = _mask_rows(out_tw, _mR)\n"
         "    elif _mR is not None and _MASK.get('Rpart', '') == 'n':\n"
         "        out_tw = _mask_rows_new(out_tw, _mR, n)\n"),
        # P2.4.2 (key C): the covariance sandwich reads C on the previous layer's R rows only
        # (C=1; C=2 keeps the dropped variances). P5.1 item 6: C=3 / C=4 the same on the
        # previous layer's D set.
        ("    mu_pre = WT @ mu\n",
         "    mu_pre = WT @ mu\n"
         "    _covS = cov\n"
         "    _ck = None if _MASK is None or not _MASK.get('C') else ('_prevD' if _MASK['C'] >= 3.0 else '_prevR')\n"
         "    if _ck is not None and _MASK.get(_ck) is not None:\n"
         "        _pc = _MASK[_ck]\n"
         "        _covS = cov * (_pc[:, None] * _pc[None, :])\n"
         "        if _MASK['C'] in (2.0, 4.0):\n"
         "            # keep the dropped rows' VARIANCES; drop only their cross-covariances\n"
         "            _covS = _covS + fnp.diag(fnp.diagonal(cov) * (1.0 - _pc))\n"),
        ("        _X4 = _mm(WT, cov)\n", "        _X4 = _mm(WT, _covS)\n"),
        ("        cov_pre = (_p4_sym_sandwich(WT, cov, W) if _P4_BLOCKSYM\n"
         "                   else _mm(_mm(WT, cov), W))\n",
         "        cov_pre = (_p4_sym_sandwich(WT, _covS, W) if _P4_BLOCKSYM\n"
         "                   else _mm(_mm(WT, _covS), W))\n"),
    ],
    # HANDOFF 2026-10-06 Lane V+M, M4: neighbor-feature corrector screen (twin only).
    # A second additive head on mu_out, beside the shipped corrector, reading each
    # neuron's strongest-correlated partners (Puffi-style). Kernel weights are the
    # SIMILARITY exp(-beta (1 - rho^2)) (the handoff's exp(-beta rho^2) would weight the
    # uncorrelated bulk most); C+ and R+ are ridge solves (1e-3 relative) in float32.
    # Its output page W2/b2 is
    # zero-initialised, so the arm starts as an exact identity on the chain. Features
    # are detached inputs (only `nbr/*` trains) and RMS-normalised per net, so no
    # standardiser file is needed. At the last layer (no cov_pre) the neighbor block
    # is zero and only the neuron's own features remain.
    "nbr": [
        ("def _layer(W, state, table, dtype, const, init_scale, last_layer, gens,\n",
         "_NBR_K = 8\n"
         "_NBR_OUT = 1e-3\n\n\n"
         "def _nbr_feats(alpha, sigma, D3, cov_pre, last_layer):\n"
         "    import torch as _t\n"
         "    with _t.no_grad():\n"
         "        n = alpha.shape[0]\n"
         "        Phi = _t.special.ndtr(alpha)\n"
         "        phi = _t.exp(-0.5 * alpha * alpha) * 0.3989422804014327\n"
         "        g3 = D3 / (sigma * sigma * sigma)\n"
         "        own = _t.stack([alpha, Phi, phi, g3], 1)\n"
         "        if last_layer or cov_pre is None:\n"
         "            nb = _t.zeros((n, _NBR_K * 5 + 12), dtype=alpha.dtype, device=alpha.device)\n"
         "        else:\n"
         "            rho = cov_pre / (sigma[:, None] * sigma[None, :])\n"
         "            off = rho - _t.diag(_t.diagonal(rho))\n"
         "            top = _t.topk(off.abs(), _NBR_K, dim=1).indices\n"
         "            r_top = _t.gather(off, 1, top)\n"
         "            nbk = _t.cat([r_top[:, :, None], own[top]], 2).reshape(n, -1)\n"
         "            eye = _t.eye(n, device=alpha.device, dtype=alpha.dtype)\n"
         "            kern = []\n"
         "            for beta in (8.0, 32.0):\n"
         "                Kb = _t.exp(-beta * (1.0 - off * off)) * (1.0 - eye)\n"
         "                m = Kb.sum(1, keepdim=True)\n"
         "                kern += [(Kb @ own) / (m + 1e-30), _t.log(m + 1e-30)]\n"
         "            rC = 1e-3 * _t.diagonal(cov_pre).mean()\n"
         "            Cp_a = _t.linalg.solve(cov_pre + rC * eye, alpha[:, None])[:, 0]\n"
         "            Rp_1 = _t.linalg.solve(rho + 1e-3 * eye, _t.ones_like(alpha)[:, None])[:, 0]\n"
         "            nb = _t.cat([nbk] + kern + [Cp_a[:, None], Rp_1[:, None]], 1)\n"
         "        X = _t.cat([own, nb], 1)\n"
         "        X = X / (X.pow(2).mean(0, keepdim=True).sqrt() + 1e-12)\n"
         "        X = _t.where(_t.isfinite(X), X, _t.zeros_like(X))\n"
         "    return X\n\n\n"
         "def _nbr_delta(li, p, alpha, sigma, D3, cov_pre, last_layer, dtype):\n"
         "    import torch as _t\n"
         "    if li >= p['W1'].shape[0]:\n"
         "        return 0.0\n"
         "    X = _nbr_feats(alpha, sigma, D3, cov_pre, last_layer)\n"
         "    h = _t.tanh(X @ p['W1'][li] + p['b1'][li])\n"
         "    return ((h @ p['W2'][li])[:, 0] + p['b2'][li]) * _NBR_OUT\n\n\n"
         "def _layer(W, state, table, dtype, const, init_scale, last_layer, gens,\n"),
        ("    if last_layer:\n        return (mu_out, None, None, 0.0, dG)\n",
         "    if const.get('nbr') is not None:\n"
         "        mu_out = mu_out + _nbr_delta(layer_index, const['nbr'], alpha, sigma, D3,\n"
         "                                     cov_pre, last_layer, dtype)\n"
         "    if last_layer:\n        return (mu_out, None, None, 0.0, dG)\n"),
    ],
    # HANDOFF 2026-10-07 Lane P2, P2.1: DRESSED LEGS (R1 §5.1). AUGMENT's heavy-edge kappa3
    # birth families fold into the newborn's light leg:
    #   X~[a, j] = w1e_a C_aj + H[a, j],
    #   H[a, j]  = 1/2 w2_a D21[a, j]
    #            + (w3_j / w2_j) (1/2 w1_a D21[j, a] + 1/4 w2_a D22[a, j] + 1/2 w2_a C_aj^2)
    # with w3/w2 = -alpha/sigma, D21[a, j] = kappa3_aaj and D22 the chain's (2,2) slice, all
    # off-diagonal. The stored A_L (fac1) IS the light leg, so every repeated-slice term that
    # reads it is recomputed from X~ -- p111_21's two-leg slice (through fac1), S_sep =
    # (X~ d(e))^T and the hub compensation d(y) X~ d(c) -- which keeps the newborn's repeated
    # slices at the table's values and changes only its distinct-index content. No matmul.
    # `_DRESS` None = inert (exact identity). Keys: c2, d21, d22 (family scales, 1 = on),
    # gate ('light': the birth gate multiplies w1 C only; 'both': it multiplies X~).
    "dress": [
        ("def _layer(W, state, table, dtype, const, init_scale, last_layer, gens,\n",
         "_DRESS = None\n\n\n"
         "def _dress_H(w, alpha, sigma, C11, D21, D22, dtype):\n"
         "    w1, w2 = w[(1, 1)], w[(2, 1)]\n"
         "    r = -(alpha / sigma)\n"
         "    H = None\n"
         "    _d = _DRESS\n"
         "    if _d.get('d21', 0.0):\n"
         "        H = (w2[:, None] * D21 + (w1[:, None] * D21.T) * r[None, :]) * (0.5 * _d['d21'])\n"
         "    if _d.get('d22', 0.0):\n"
         "        _t = ((w2[:, None] * D22) * r[None, :]) * (0.25 * _d['d22'])\n"
         "        H = _t if H is None else H + _t\n"
         "    if _d.get('c2', 0.0):\n"
         "        _t = ((w2[:, None] * (C11 * C11)) * r[None, :]) * (0.5 * _d['c2'])\n"
         "        H = _t if H is None else H + _t\n"
         "    return H\n\n\n"
         "def _layer(W, state, table, dtype, const, init_scale, last_layer, gens,\n"),
        ("        fac1 = w1e[:, None] * C11\n",
         "        fac1 = w1e[:, None] * C11\n"
         "        if _DRESS is not None:\n"
         "            _Hd = _dress_H(w, alpha, sigma, C11, D21, D22, dtype)\n"
         "            if _Hd is not None:\n"
         "                if _DRESS.get('gate', 'light') == 'both':\n"
         "                    _Hd = _Hd * (w1e / fnp.where(w1 == 0.0, dtype(1.0), w1))[:, None]\n"
         "                fac1 = fac1 + _Hd\n"
         "        _Xd = fac1\n"),
        ("        S_sep = (e_b[:, None] * C11) * w1e[None, :]\n",
         "        S_sep = e_b[:, None] * _Xd.T\n"),
        ("                resid = (resid - ((yA * w1e)[:, None] * C11) * (cA * _t)[None, :]\n"
         "                         - ((yB * w1e)[:, None] * C11) * (cB * _t)[None, :])\n",
         "                resid = (resid - (yA[:, None] * _Xd) * (cA * _t)[None, :]\n"
         "                         - (yB[:, None] * _Xd) * (cB * _t)[None, :])\n"),
    ],
    # HANDOFF 2026-10-07 Lane P2, P2.2: GAIN-MODE kappa4 (R2 §1.5). z = g zhat with Var g =
    # c^2, c^2 = m^T C m / |m|^4 from the chain's own post-ReLU state (one matvec). Templates
    # (pre-activation): D22_ij = 4 b c^2 (s_i^2 s_j^2 + 2 C_ij^2), Q31_ij = 12 b c^2 s_i^2 C_ij
    # (normalised by s_i^3 s_j: 12 b c^2 Cn_ij). D4 stays the chain's dG form.
    # `_GM` None = inert. Keys: mode 'a' (D22 template), 'b' (D22 + Q31 templates), 'c'
    # (blend: existing + gamma * template, on D22, and on Q31 if q31b=1); beta (default 0.7)
    # / gamma (default 0), overridden per layer by gain/gmb, gain/gmg when learn=1 (training);
    # q31 = 'keep' | 'drop' (the 5 learned q31 modes).
    "gm": [
        ("def _layer(W, state, table, dtype, const, init_scale, last_layer, gens,\n",
         "_GM = None\n_GMS = {}\n\n\n"
         "def _gm_coef(key, default, li):\n"
         "    tk = {'beta': 'gmb', 'gamma': 'gmg'}[key]\n"
         "    if _GM.get('learn') and _TWG is not None and tk in _TWG:\n"
         "        return _TWG[tk][li]\n"
         "    return _GM.get(key, default)\n\n\n"
         "def _layer(W, state, table, dtype, const, init_scale, last_layer, gens,\n"),
        ("        D22 = ((metric / 6.0) * (dG[:, None] + dG[None, :])) * off_mask\n",
         "        D22 = ((metric / 6.0) * (dG[:, None] + dG[None, :])) * off_mask\n"
         "        _GMS.clear()\n"
         "        if _GM is not None:\n"
         "            _mm2 = (mu * mu).sum()\n"
         "            _c2 = (mu * (cov @ mu)).sum() / fnp.maximum(_mm2 * _mm2, dtype(1e-30))\n"
         "            _tpl = ((var[:, None] * var[None, :]) + (C11 * C11) * 2.0) * off_mask\n"
         "            if _GM['mode'] in ('a', 'b'):\n"
         "                _bc = _gm_coef('beta', 0.7, layer_index) * _c2\n"
         "                D22 = _tpl * (_bc * 4.0)\n"
         "            else:\n"
         "                _bc = _gm_coef('gamma', 0.0, layer_index) * _c2\n"
         "                D22 = D22 + _tpl * (_bc * 4.0)\n"
         "            if _GM['mode'] == 'b' or (_GM['mode'] == 'c' and _GM.get('q31b')):\n"
         "                _GMS['q31'] = _bc * 12.0\n"),
        ('        nb["Q31"] = Cn * inv2[:, None] * (0.5 * blocks["metric"] * blocks["lam"])\n',
         '        nb["Q31"] = Cn * inv2[:, None] * (0.5 * blocks["metric"] * blocks["lam"])\n'
         "        if 'q31' in _GMS:\n"
         "            nb['Q31'] = (Cn * _GMS['q31'] if _GM['mode'] == 'b'\n"
         "                         else nb['Q31'] + Cn * _GMS['q31'])\n"),
        ('        if _TWG is not None and "q31" in _TWG:\n',
         '        if (_TWG is not None and "q31" in _TWG\n'
         "                and not (_GM is not None and _GM.get('q31') == 'drop')):\n"),
    ],
}
# HANDOFF 2026-10-08 Lane P3, P3.2: the kappa4-core loop (tools/twin/k4l_arm.py).
import k4l_arm  # noqa: E402
ARMS["k4l"] = k4l_arm.PATCHES + [
    ("            out_tw = _append(adv, None, fac1, w2b_vec, n, Zb, Lb, _sb, e_b,\n",
     "            if _k4l_offpost is not None and _TWG is not None and 'k4g' in _TWG:\n"
     "                _k4l_offpost = _k4l_offpost * _TWG['k4g'][layer_index]\n"
     "            out_tw = _append(adv, None, fac1, w2b_vec, n, Zb, Lb, _sb, e_b,\n"),
]
TRAINABLE["S2tk"] = TRAINABLE["S2t"] + ("gain/k4g",)

# HANDOFF 2026-10-08 Lane P3, P3.1b/c ("bank"): `_BANK` None = inert.
#   vg   the variance CONSUMED at layer l times (1 + gain/vg[l]) (init 0): a per-layer level gain
#   star D4 += gain/sg[l] * D4_star (init 0), D4_star[a] = 4 sum_j w3b_j A_L[a,j]^3 M[a,j] over the
#        RECENT lineages (R3 3.1), w3b = the birth layer's w(3,1); carried in the c4 slot
ARMS["bank"] = [
    ("    mu, cov, tw, c4, gk4 = state\n",
     "    mu, cov, tw, c4, gk4 = state\n"
     "    _bkst = c4 if isinstance(c4, dict) else {}\n"),
    ("    metric = float(init_scale)\n",
     "    if _BANK is not None and _BANK.get('vg') and _TWG is not None:\n"
     "        var = var * (1.0 + _TWG['vg'][layer_index])\n"
     "    metric = float(init_scale)\n"),
    ("    D4 = metric * dG\n",
     "    D4 = metric * dG\n"
     "    if (_BANK is not None and _BANK.get('star') and _TWG is not None and adv is not None\n"
     "            and _bkst.get('w3b')):\n"
     "        _AL, _MM = (_full_legs_lean(adv) if _BANK.get('starall') else (adv.A_L, adv.M))\n"
     "        _w3c = fnp.concatenate(_bkst['w3b'][-(_AL.shape[1] // n):])\n"
     "        _d4s = (((_AL * _AL * _AL) * _MM) * _w3c[None, :]).sum(axis=1) * 4.0\n"
     "        D4 = D4 + _d4s * _TWG['sg'][layer_index]\n"
     "    if (_BANK is not None and _BANK.get('path') and _TWG is not None and adv is not None\n"
     "            and _bkst.get('cb')):\n"
     "        # R3 3.2 path: D4 = 12 rowsum((A.J) . V), V = (A.J) C~_birth, recent lineages\n"
     "        _kr0 = adv.k - adv.ko\n"
     "        _cbs = _bkst['cb'][-_kr0:]\n"
     "        _w2r = adv.w2b[adv.w2b.shape[0] - _kr0 * n:]\n"
     "        _d4p = None\n"
     "        for _i, _Cb in enumerate(_cbs):\n"
     "            _AJ = adv.A_L[:, _i * n:(_i + 1) * n] * (adv.M[:, _i * n:(_i + 1) * n]\n"
     "                                                    * _w2r[_i * n:(_i + 1) * n][None, :])\n"
     "            _t = ((_AJ @ _Cb) * _AJ).sum(axis=1)\n"
     "            _d4p = _t if _d4p is None else _d4p + _t\n"
     "        D4 = D4 + _d4p * (12.0 * _TWG['pg'][layer_index])\n"),
    ("    return (mu_out, cov_out, out_tw, c4_out, g_out)\n",
     "    if _BANK is not None and _BANK.get('star'):\n"
     "        _kr = out_tw.k if _BANK.get('starall') else out_tw.k - out_tw.ko\n"
     "        _l3 = (list(_bkst.get('w3b') or []) + [w[(3, 1)]])[-_kr:]\n"
     "        c4_out = {'c4': c4_out, 'w3b': _l3}\n"
     "        if _BANK.get('path'):\n"
     "            c4_out['cb'] = (list(_bkst.get('cb') or []) + [C11])[-(out_tw.k - out_tw.ko):]\n"
     "    return (mu_out, cov_out, out_tw, c4_out, g_out)\n"),
    ("def _layer(W, state, table, dtype, const, init_scale, last_layer, gens,\n",
     "_BANK = None\n\n\ndef _layer(W, state, table, dtype, const, init_scale, last_layer, gens,\n"),
]
# HANDOFF 2026-10-11 Lane M6, M6.3: alpha-dependent covariance gains (bank keys, zero-init =
# exact identity). `acov`: the D21 / D22 S-blocks the bivariate (covariance) terms read get the
# row factor (1 + gain/ab[l] alpha_a^2) (D22 symmetrised as the row/column mean); the tower
# births keep reading the raw D21. `avar`: the variance consumed at layer l times
# (1 + gain/va[l] alpha^2), alpha^2 = mu_pre^2 / var (M6.1: var in band 12-15 is 19 % of error).
ARMS["bank"] += [
    ("    metric = float(init_scale)\n",
     "    if _BANK is not None and _BANK.get('avar') and _TWG is not None:\n"
     "        var = var * (1.0 + _TWG['va'][layer_index] * ((mu_pre * mu_pre) / var))\n"
     "    metric = float(init_scale)\n"),
    ("             (3,): D3, (4,): D4, (2, 2): D22}\n",
     "             (3,): D3, (4,): D4, (2, 2): D22}\n"
     "        if _BANK is not None and _BANK.get('acov') and _TWG is not None:\n"
     "            _fa = 1.0 + _TWG['ab'][layer_index] * (alpha * alpha)\n"
     "            S[(2, 1)] = D21 * _fa[:, None]\n"
     "            S[(1, 2)] = S[(2, 1)].T\n"
     "            S[(2, 2)] = D22 * ((_fa[:, None] + _fa[None, :]) * 0.5)\n"),
]
TRAINABLE["S2tv"] = TRAINABLE["S2t"] + ("gain/vg",)
TRAINABLE["S2ts"] = TRAINABLE["S2t"] + ("gain/sg",)
TRAINABLE["S2tsp"] = TRAINABLE["S2t"] + ("gain/sg", "gain/pg")

# HANDOFF 2026-10-10 Lane M5, M5.1 / M5.2 (tools/twin/k4x_arm.py): the masked star-Q31 fold
# (gain/fg, init = gain/sg) and the radial Sym(mu_pre x kappa3) kappa4 terms (gain/rq, r22, r4,
# init at the quenched-gain-law v'). Both need `bank` (star, starall) before them in --arm.
import k4x_arm  # noqa: E402
ARMS["fold"] = list(k4x_arm.PATCHES[:6])
ARMS["rad"] = [k4x_arm.PATCHES[0]] + list(k4x_arm.PATCHES[6:])
TRAINABLE["S2tsf"] = TRAINABLE["S2ts"] + ("gain/fg",)
TRAINABLE["S2tsfa"] = TRAINABLE["S2tsf"] + ("gain/ab", "gain/va")   # M6.3
TRAINABLE["S2tsr"] = TRAINABLE["S2ts"] + ("gain/rq", "gain/r22", "gain/r4")
TRAINABLE["S2tsfr"] = TRAINABLE["S2tsf"] + ("gain/rq", "gain/r22", "gain/r4")
ARMS["qx"] = list(k4x_arm.QX_PATCHES)
TRAINABLE["S2tsfx"] = TRAINABLE["S2tsf"] + ("gain/qx",)

#: M4 neighbor head: 4 own + 5*K partners + 2 x (4 kernel means + log mass) + C+alpha + R+1.
NBR_F, NBR_H = 4 + 5 * 8 + 12, 32


def parse_mask(spec):
    """'R=2.0,H=3.0,D=3.0' -> {'R': 2.0, 'H': 3.0, 'D': 3.0}; '' -> {}.

    `Hk=tau` is H with the always-on columns' A_L kept as the carrier of e_b
    (only w2b and the hubs are zeroed there); plain `H` drops A_L[:, j] for every
    decided column and folds e_b's separable S21 row into the rank-4 P2 residual."""
    out = {}
    for part in filter(None, spec.split(",")):
        k, v = part.split("=")
        if k == "K":             # P2 variant A: top-K per role/layer from a table (JSON path)
            out["Ktab"] = json.loads(Path(v).read_text())["K"]
            continue
        # H parts: A = A_L column (carries e_b), W = two-leg w2b, C = hubs, Sl = slice
        # (sb, residual row). Default H: A W C on |alpha| > tau, Sl on alpha < -tau.
        #   Hk  A on dead only            Hkh  A, C on dead only     Hd  everything dead-only
        #   Hw  W only (diagnostic)       Hs   Sl only (diagnostic)
        hmodes = {"Hk": "k", "Hkh": "kh", "Hd": "d", "Hw": "w", "Hs": "s"}
        rparts = {"Ri": "i", "Rn": "n"}     # R on inherited legs only / newborn rows only
        if k == "Da":            # P2.4.1: ALSO drop D21 rows with alpha > +tau (always-on)
            out["Da"] = float(v)
            continue
        if k == "Dleg":          # P5.2 (ii)+(iii): legs' output rows on the D set only
            out["Dleg"] = bool(float(v))
            continue
        if k == "rho":           # P5.2: per-layer d21e energy ratio (JSON list or {"rho": [...]})
            _r = json.loads(Path(v).read_text())
            out["rho"] = _r["rho"] if isinstance(_r, dict) else _r
            continue
        if k == "J":             # P5.1 item 1: skip the old-tier join at the last J non-final layers
            out["J"] = int(float(v))
            continue
        if k == "Hrank":         # P6.1: K-table H ranking 'abs' (alpha^2) or 'asym:a:b' (max(-alpha/a, alpha/b))
            out["Hrank"] = ("abs",) if v == "abs" else ("asym", *map(float, v.split(":")[1:]))
            if not (v == "abs" or (v.startswith("asym:") and len(out["Hrank"]) == 3)):
                raise SystemExit(f"--mask: bad Hrank {v}")
            continue
        if k == "Hfull":         # P6.1: _mSl = _mH (residual rows and S3 follow the full H drop)
            out["Hfull"] = bool(float(v))
            continue
        if k == "Daz":           # always-on rows' corrector: z zero (Dz), f full row, c column
            out["Daz"] = {"0": "z", "1": "f", "2": "c"}.get(v, v)
            continue
        if k not in ("R", "H", "D", "C", "Dc", "Dr", "Dz", "D2", "D2c", *hmodes, *rparts):
            raise SystemExit(f"--mask: unknown role {k}")
        if k in rparts:
            k, out["Rpart"] = "R", rparts[k]
        if k in hmodes:
            k, out["Hmode"] = "H", hmodes[k]
        if k == "Dc":            # diagnostic: the corrector's d21e reads the UNMASKED D21
            k, out["Dfull"] = "D", True
        if k == "Dr":            # dead ROWS of D21 only: every live row's d21e stays exact
            k, out["Drows"] = "D", True
        if k == "Dz":            # Dr + no corrector delta on the dropped rows (d21e unknown)
            k, out["Drows"], out["Dzero"] = "D", True, True
        if k == "D2":            # P5.2 two-sided D (rows AND columns) + no delta on dropped rows
            k, out["Dzero"] = "D", True
        if k == "D2c":           # diagnostic: D2 with the corrector reading the UNMASKED d21e
            k, out["Dzero"], out["Dfull"] = "D", True, True
        out[k] = float(v)
    return out


def parse_dress(spec):
    """'c2=1,d21=1,d22=1,gate=light' -> dict; 'all' = every family at 1; '' -> None (inert)."""
    if not spec:
        return None
    if spec == "all":
        spec = "c2=1,d21=1,d22=1"
    out = {"gate": "light"}
    for part in filter(None, spec.split(",")):
        k, v = part.split("=")
        if k == "gate":
            assert v in ("light", "both"), v
            out[k] = v
        elif k in ("c2", "d21", "d22"):
            out[k] = float(v)
        else:
            raise SystemExit(f"--dress: unknown key {k}")
    return out


def parse_gm(spec):
    """'mode=a,beta=0.7,q31=keep' -> dict; '' -> None (inert)."""
    if not spec:
        return None
    out = {"mode": "a", "q31": "keep"}
    for part in filter(None, spec.split(",")):
        k, v = part.split("=")
        if k == "mode":
            assert v in ("a", "b", "c"), v
            out[k] = v
        elif k == "q31":
            assert v in ("keep", "drop"), v
            out[k] = v
        elif k in ("beta", "gamma", "q31b", "learn"):
            out[k] = float(v)
        else:
            raise SystemExit(f"--gm: unknown key {k}")
    return out


def load_module(path: Path, name="_twin_candidate", arm="", raw=False):
    """Import the candidate from a PATCHED copy (see TWIN_PATCHES), kept for audit.

    `raw=True` skips TWIN_PATCHES (the gain hooks), for EVALUATING a file whose
    gains are already baked in by tools/build_s2.py -- its rewritten lines no
    longer match the hooks' anchors. `install` refuses gain parameters then."""
    import importlib.util
    src = path.read_text(encoding="utf-8")
    arm_patches = [pt for x in arm.split("+") if x for pt in ARMS[x]]   # "mask+dress"
    for a, b in ([] if raw else TWIN_PATCHES) + arm_patches + SUB_PATCHES:
        if src.count(a) != 1:
            raise SystemExit(f"twin patch anchor occurs {src.count(a)}x:\n{a}")
        src = src.replace(a, b, 1)
    pp = ROOT / "cache" / "twin" / f"_twin_{path.stem}{'_' + arm.replace('+', '_') if arm else ''}.py"
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(src, encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location(name, pp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if raw:                                      # no gain hooks, no lambda wrapper
        mod._TWG = None
        return mod
    _orig_lam = mod._own_lambda

    def _own_lambda(i, depth):                   # lambda gain, per SOURCE layer i
        v = _orig_lam(i, depth)
        g = mod._TWG
        if g is None or not (0 <= i < g["lam"].shape[0]):
            return v
        return g["lam"][i] * v

    mod._own_lambda = _own_lambda
    return mod


class Twin:
    def __init__(self, cand: Path, width=1024, depth=16, arm="", raw=False):
        self.raw = raw
        self.mod = load_module(cand, arm=arm, raw=raw)
        self.est = self.mod.Estimator()
        self.est.setup(_Ctx(width, depth, BUDGET, 0))
        self.width, self.depth = width, depth
        self.dtype = fnp.float32
        self.table = self.mod._tables()
        self.gens = self.mod._fit_generations(width, depth, BUDGET)
        assert self.gens is None, "generation cap -- not the exact tower"
        self.const = self.est._constants(width, self.dtype)
        #: the frozen base values of every learnable group the file carries
        self.base = {g: {k: v.detach().clone() for k, v in self.const[g].items()}
                     for g in ("corr", "sig", "clos", "gate") if self.const.get(g) is not None}
        one = torch.ones(depth, dtype=torch.float32, device=DEV)
        self.base["gain"] = {"lam": one.clone(), "theta": one.clone(), "feed": one.clone()}
        # S2t per-term gains: table terms, 1-D kappa3/kappa4 per p, closure extras
        nterm = len(self.table["terms"])
        cprog = self.const["closure"]["prog"] if self.const.get("closure") else {"uni": [], "biv": []}
        nps = len(self.const["wick"]["ps"])
        mk = lambda *shape: torch.ones(shape, dtype=torch.float32, device=DEV)
        self.base["gain"].update({"tt": mk(depth, nterm), "t1": mk(depth, 2, nps),
                                  "cu": mk(depth, len(cprog["uni"]), 4),
                                  "cb": mk(depth, len(cprog["biv"]))})
        # S3: additive multi-mode correction to the normalised kappa4_iiij slice,
        # zero-initialised (an exact identity) -- see TWIN_PATCHES "q31"
        self.base["gain"]["q31"] = torch.zeros((depth, 5), dtype=torch.float32, device=DEV)
        # P2.2 gain-mode kappa4: per-layer beta (template scale, init 0.7) and gamma (blend
        # weight, init 0). Read only by the `gm` arm, and only once installed as params.
        self.base["gain"]["gmb"] = torch.full((depth,), 0.7, dtype=torch.float32, device=DEV)
        self.base["gain"]["gmg"] = torch.zeros((depth,), dtype=torch.float32, device=DEV)
        # P3.2 kappa4-core loop: per-layer gain on the off-diagonal core births (k4l arm)
        self.base["gain"]["k4g"] = torch.ones((depth,), dtype=torch.float32, device=DEV)
        self.base["gain"]["vg"] = torch.zeros((depth,), dtype=torch.float32, device=DEV)
        self.base["gain"]["sg"] = torch.zeros((depth,), dtype=torch.float32, device=DEV)
        self.base["gain"]["pg"] = torch.zeros((depth,), dtype=torch.float32, device=DEV)
        self.base["gain"]["fg"] = torch.zeros((depth,), dtype=torch.float32, device=DEV)
        self.base["gain"]["ab"] = torch.zeros((depth,), dtype=torch.float32, device=DEV)
        self.base["gain"]["va"] = torch.zeros((depth,), dtype=torch.float32, device=DEV)
        _vp = torch.tensor(k4x_arm.vprime(depth), dtype=torch.float32, device=DEV)
        for _k in ("rq", "r22", "r4"):
            self.base["gain"][_k] = _vp.clone()
        self.base["gain"]["qx"] = torch.zeros((depth, 3), dtype=torch.float32, device=DEV)
        if "nbr" in arm.split("+"):
            # M4 neighbor head; W2/b2 zero -> exact identity at init. W1 deterministic.
            g = torch.Generator(device="cpu").manual_seed(20261006)
            self.base["nbr"] = {
                "W1": (torch.randn((depth, NBR_F, NBR_H), generator=g) / math.sqrt(NBR_F)).to(DEV),
                "b1": torch.zeros((depth, NBR_H), device=DEV),
                "W2": torch.zeros((depth, NBR_H, 1), device=DEV),
                "b2": torch.zeros((depth,), device=DEV)}
            self.const["nbr"] = self.base["nbr"]

    def leaf(self, key):
        g, k = key.split("/")
        return self.base[g][k]

    def install(self, params: dict):
        """Point each group at `params` ("group/name" -> tensor) + its frozen rest.

        `gain/*` is live only when some gain is actually in `params`: with no
        gains installed `_TWG` stays None and the twin runs the file's own
        Python-float profiles, bit for bit."""
        unknown = [k for k in params if k.split("/")[0] not in self.base
                   or k.split("/")[1] not in self.base[k.split("/")[0]]]
        if self.raw and any(k.startswith("gain/") for k in params):
            raise SystemExit("install: gain parameters need the twin's source hooks (not --raw)")
        if unknown:        # a silently ignored key made one DEV read a no-op once
            raise SystemExit(f"install: unknown parameter keys {unknown}")
        for g, vals in self.base.items():
            d = dict(vals)
            d.update({k.split("/")[1]: v for k, v in params.items() if k.startswith(g + "/")})
            if g == "gain":
                self.mod._TWG = d if any(k.startswith("gain/") for k in params) else None
            else:
                self.const[g] = d

    def _step(self, l, W, state):
        s = self.mod._layer(W, state, self.table, self.dtype, self.const, 2.0,
                            last_layer=(l == self.depth - 1), gens=self.gens,
                            layer_index=l, depth=self.depth)
        mu = torch.where(torch.isfinite(s[0]), s[0], torch.zeros_like(s[0]))
        return (mu, s[1], s[2], s[3], s[4])

    def forward(self, w, ckpt=False):
        """Rows 1..depth-1 of the prediction (clamped at 0), as a list of tensors."""
        n = self.width
        state = (fnp.zeros(n), fnp.eye(n), None, 0.0, fnp.zeros(n))
        mus = []
        for l in range(self.depth):
            if ckpt:
                state = checkpoint(self._step, l, w[l], state, use_reentrant=False)
            else:
                state = self._step(l, w[l], state)
            mus.append(torch.clamp_min(state[0], 0.0))
        return mus


def mse_rows(mus, truth):
    """Per-row MSE (float64) for rows 1..depth-1."""
    t = torch.from_numpy(np.asarray(truth)).to(DEV)
    return [((m.double() - t[l]) ** 2).mean() for l, m in enumerate(mus) if l >= 1]


def set_subst(tw, subst, split, i):
    """Step 2: point the module's `_A2` at net i's exact labels (or clear it)."""
    if not subst:
        tw.mod._A2 = None
        return
    assert split == "full" and i < 100, "exact labels exist for FIT rows 0..99 only"
    tw.mod._A2 = Subst(i, SUBST[subst])


def load_params(path):
    """npz -> {"group/name": tensor}. Bare names (old S1 files) mean group corr."""
    out = {}
    for k, v in np.load(path).items():
        k = k.replace("__", "/")
        out[k if "/" in k else f"corr/{k}"] = torch.from_numpy(v).to(DEV)
    return out


def save_params(path, params):
    np.savez(path, **{k.replace("/", "__"): v.detach().cpu().numpy() for k, v in params.items()})


# ------------------------------------------------------------------ commands

def _set_k4x(tw, a, params):
    """M5.1 / M5.2 specs onto the module (`--fold`, `--rad`); learn=1 when gain/fg is installed."""
    if getattr(a, "fold", ""):
        assert hasattr(tw.mod, "_FOLD"), "--fold needs --arm ...+bank+fold"
        f = k4x_arm.parse_fold(a.fold)
        if params is not None and "gain/fg" in params:
            f["learn"] = 1.0
        tw.mod._FOLD = f
    if getattr(a, "rad", ""):
        assert hasattr(tw.mod, "_RAD"), "--rad needs --arm ...+rad"
        tw.mod._RAD = k4x_arm.parse_rad(a.rad)


def cmd_parity(a):
    """Twin forward vs the cupy probe (anatomy2's plain chain MSE on the same nets)."""
    tw = Twin(Path(a.cand))
    data = Data()
    ref = json.loads(Path(a.ref).read_text())["nets"] if a.ref else {}
    out = []
    for i in [int(x) for x in a.nets.split(",")]:
        w, truth = data.net("full", i)
        t0 = time.time()
        with torch.no_grad():
            mus = tw.forward(w)
        torch.cuda.synchronize()
        dt = time.time() - t0
        m = [float(x) for x in mse_rows(mus, truth)]
        r = ref.get(str(i), {}).get("chain", {}).get(str(tw.depth - 1))
        rel = (m[-1] / r[0] - 1) if r else float("nan")
        out.append({"net": i, "twin_final_mse": m[-1], "cupy_final_mse": r[0] if r else None,
                    "rel": rel, "seconds": round(dt, 2)})
        print(json.dumps(out[-1]), flush=True)
    return out


def cmd_eval(a, tw=None, params=None, quiet=False):
    tw = tw or Twin(Path(a.cand), arm=getattr(a, "arm", ""), raw=getattr(a, "raw", False))
    if params is None and a.params:
        params = load_params(a.params)
    if params is not None:
        tw.install(params)
    data = Data()
    rows = panel(a.panel)[: a.n or None]
    fin, fstats = [], []
    mask = parse_mask(getattr(a, "mask", ""))
    stats = {} if getattr(a, "mask_stats", False) else None
    if getattr(a, "dress", ""):
        assert hasattr(tw.mod, "_DRESS"), "--dress needs --arm dress (or mask+dress)"
        tw.mod._DRESS = parse_dress(a.dress)
    if getattr(a, "gm", ""):
        assert hasattr(tw.mod, "_GM"), "--gm needs --arm gm (or mask+gm)"
        tw.mod._GM = parse_gm(a.gm)
    if getattr(a, "k4l", ""):
        assert hasattr(tw.mod, "_K4L"), "--k4l needs --arm ...+k4l"
        tw.mod._K4L = k4l_arm.parse(a.k4l)
    if getattr(a, "bank", ""):
        assert hasattr(tw.mod, "_BANK"), "--bank needs --arm ...+bank"
        tw.mod._BANK = {k: 1 for k in a.bank.split(",")}
    if getattr(a, "shared_rank", 0):
        tw.mod._SHARED_RANK = a.shared_rank
    _set_k4x(tw, a, params)
    if mask or stats is not None:
        assert hasattr(tw.mod, "_MASK"), "--mask needs --arm mask"
    with torch.no_grad():
        for split, i in rows:
            w, truth = data.net(split, i)
            set_subst(tw, getattr(a, "subst", ""), split, i)
            if mask or stats is not None:
                tw.mod._MASK = dict(mask, stats=[] if stats is not None else None)
            if getattr(tw.mod, "_FOLDS", None) is not None:
                tw.mod._FOLDS["stats"] = []
            fin.append(float(mse_rows(tw.forward(w), truth)[-1]))
            if getattr(tw.mod, "_FOLD", None) and tw.mod._FOLD.get("stats"):
                fstats.append(tw.mod._FOLDS["stats"])
            if stats is not None:
                stats[f"{split}:{i}"] = tw.mod._MASK["stats"]
    r = {"panel": a.panel, "arm": getattr(a, "arm", ""), "n": len(fin),
         "raw_mse": float(np.mean(fin)), "per_net": fin,
         "dress": getattr(tw.mod, "_DRESS", None), "gm": getattr(tw.mod, "_GM", None),
         "k4l": getattr(tw.mod, "_K4L", None), "shared_rank": getattr(tw.mod, "_SHARED_RANK", None)}
    r["fold"] = getattr(tw.mod, "_FOLD", None)
    r["rad"] = getattr(tw.mod, "_RAD", None)
    if fstats:
        r["fold_stats"] = fstats
    if mask or stats is not None:
        r["mask"] = mask
        r["nets"] = [f"{s}:{i}" for s, i in rows]
    if stats is not None:
        r["stats"] = stats
    if getattr(a, "json", ""):
        Path(a.json).write_text(json.dumps(r))
    if not quiet:
        print(json.dumps({k: v for k, v in r.items() if k not in ("per_net", "stats")}),
              flush=True)
    return r


def cmd_sigstats(a):
    """Per-layer mean / (1/std) of the sigma page's 27 expanded features on TRAIN.

    Collected at the point `var` is formed, from the chain as it stands (sigma page
    inert), over `--n` TRAIN nets. Written as sig__smu / sig__sisd, the frozen
    standardisers `tools/build_v62.py --params` bakes into the file.
    """
    tw = Twin(Path(a.cand))
    m = tw.mod
    assert hasattr(m, "_sig_raw"), "candidate has no sigma corrector (build_v62 first)"
    acc = {}
    orig = m._sig_factor

    def spy(li, sig, mu_pre, var, D3, W, dtype):
        X = m._sig_raw(mu_pre, var, D3, W, dtype)
        X = torch.cat([X, X * X, X[:, m._SIG_PI] * X[:, m._SIG_PJ]], dim=1).double()
        s1, s2, c = acc.get(li, (0.0, 0.0, 0))
        acc[li] = (s1 + X.sum(0), s2 + (X * X).sum(0), c + X.shape[0])
        return orig(li, sig, mu_pre, var, D3, W, dtype)

    m._sig_factor = spy
    data = Data()
    with torch.no_grad():
        for split, i in panel("train")[: a.n or 20]:
            w, _ = data.net(split, i)
            tw.forward(w)
    m._sig_factor = orig
    L = len(acc)
    smu = np.zeros((L, acc[0][0].numel()), np.float32)
    sisd = np.ones_like(smu)
    for li, (s1, s2, c) in acc.items():
        mu = (s1 / c).cpu().numpy()
        sd = np.sqrt(np.maximum((s2 / c).cpu().numpy() - mu * mu, 1e-30))
        smu[li], sisd[li] = mu, 1.0 / np.maximum(sd, 1e-12)
    np.savez(a.out, smu=smu, sisd=sisd)
    print(f"wrote {a.out}: {L} layers x {smu.shape[1]} features")


def cmd_d8bprep(a):
    """Gather the d8b corpus shards' per-layer means (moments[:, 0]) into D8B_MEAN."""
    out = np.zeros((2048, 16, 1024), np.float32)
    seen = np.zeros(2048, bool)
    for s in range(16):
        z = np.load(D8B_MEAN.parent / f"shard{s:02d}.npz")
        assert int(z["n_done"]) == len(z["idx"]) and int(z["n_samples"]) == 10 ** 8
        out[z["idx"]] = z["moments"][:, 0]
        seen[z["idx"]] = True
    assert seen.all(), "d8b shards do not cover idx 0..2047"
    np.save(D8B_MEAN, out)
    print(f"wrote {D8B_MEAN}: {out.shape}")


def cmd_train(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tw = Twin(Path(a.cand), arm=a.arm)
    if a.mask:                    # P1 refit on the masked chain: set once, read every forward
        assert hasattr(tw.mod, "_MASK"), "--mask needs --arm mask"
        tw.mod._MASK = parse_mask(a.mask)
        print(f"  mask {tw.mod._MASK}", flush=True)
    if a.dress:
        assert hasattr(tw.mod, "_DRESS"), "--dress needs --arm dress (or mask+dress)"
        tw.mod._DRESS = parse_dress(a.dress)
        print(f"  dress {tw.mod._DRESS}", flush=True)
    if a.gm:
        assert hasattr(tw.mod, "_GM"), "--gm needs --arm gm (or mask+gm)"
        tw.mod._GM = parse_gm(a.gm)
        print(f"  gm {tw.mod._GM}", flush=True)
    if a.k4l:
        assert hasattr(tw.mod, "_K4L"), "--k4l needs --arm ...+k4l"
        tw.mod._K4L = k4l_arm.parse(a.k4l)
        print(f"  k4l {tw.mod._K4L}", flush=True)
    if a.shared_rank:
        tw.mod._SHARED_RANK = a.shared_rank
        print(f"  shared rank {a.shared_rank}", flush=True)
    if a.bank:
        assert hasattr(tw.mod, "_BANK"), "--bank needs --arm ...+bank"
        tw.mod._BANK = {k: 1 for k in a.bank.split(",")}
        print(f"  bank {tw.mod._BANK}", flush=True)
    names = TRAINABLE[a.stage]
    params = {k: tw.leaf(k).clone().requires_grad_(True) for k in names}
    _init = load_params(a.init) if a.init else {}
    if a.init:
        for k, v in _init.items():
            if k in params:
                params[k] = v.clone().requires_grad_(True)
            else:                                   # frozen, but not the file's value
                params[k] = v
    if "gain/fg" in names and "gain/fg" not in _init and "gain/sg" in params:
        # M5.1: the fold gain starts at the star gain, so D4 and D31 start consistent
        params["gain/fg"] = params["gain/sg"].detach().clone().requires_grad_(True)
    _set_k4x(tw, a, params)
    for _k in ("_FOLD", "_RAD"):
        if getattr(tw.mod, _k, None) is not None:
            print(f"  {_k[1:].lower()} {getattr(tw.mod, _k)}", flush=True)
    tw.install(params)
    # PER-TENSOR RELATIVE STEP SIZES. Adam moves every weight by ~lr whatever its
    # gradient, and the corrector's output page W3/b3 carries the target scale
    # (~1e-4): an absolute 1e-3 step sent the train MSE 2.4e-8 -> 3.5e-3 in ONE
    # step (smoke run, 2026-09-25). So lr_k = lr x RMS(tensor_k): every step is a
    # relative change of ~lr. A zero-initialised tensor (sigma page S2/c2) takes
    # the RMS of its group's non-zero siblings times --zero-scale.
    groups = []
    for k, v in params.items():
        if not v.requires_grad:
            continue
        rms = float(v.detach().pow(2).mean().sqrt())
        if rms == 0.0:
            rms = a.zero_scale
        mult = dict((kv.split("=")[0], float(kv.split("=")[1]))
                    for kv in filter(None, getattr(a, "lr_mult", "").split(",")))
        rms = rms * mult.get(k, 1.0)
        groups.append({"params": [v], "lr": a.lr * rms, "name": k})
        print(f"  {k}: rms {rms:.3e} -> lr {a.lr * rms:.3e}", flush=True)
    opt = torch.optim.Adam(groups)
    data = Data()
    train = panel(a.train_panel)[: a.train_n or None]
    rng = np.random.default_rng(20260925)
    steps_per_epoch = len(train) // a.accum
    total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    log = open(out / "train.log", "a", encoding="utf-8")

    def evaluate(tag):
        ev = argparse.Namespace(panel=a.hold_panel, n=a.hold_n, cand=a.cand, params=None,
                                subst=a.subst, mask=a.mask, k4l=a.k4l,
                                shared_rank=a.shared_rank, bank=a.bank, fold=a.fold, rad=a.rad)
        r = cmd_eval(ev, tw=tw, params=params, quiet=True)
        msg = f"[{tag}] HOLDOUT-100 raw {r['raw_mse']:.6e} (n={r['n']})"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
        return r["raw_mse"]

    # RESUME (worker-q preemption): `ckpt.pt` is written at every evaluation; a restarted
    # job reloads it and skips the steps already taken. The data order is a pure function
    # of the seed, so the permutations are regenerated, not stored.
    ck = out / "ckpt.pt"
    start = 0
    if ck.exists():
        st = torch.load(ck, map_location=DEV, weights_only=False)
        with torch.no_grad():
            for k, v in st["params"].items():
                params[k].copy_(v)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        best, bad, start = st["best"], st["bad"], st["step"]
        msg = f"[resume] step {start}, best {best:.6e}, bad {bad}"
        print(msg, flush=True)
        log.write(msg + "\n")
    else:
        best = evaluate("init")
        save_params(out / "params_best.npz", params)
        bad = 0

    def checkpoint_state(step):
        torch.save({"params": {k: v.detach() for k, v in params.items()},
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "best": best, "bad": bad, "step": step}, str(ck) + ".tmp")
        os.replace(str(ck) + ".tmp", ck)

    step = 0
    progress = os.environ.get("WORKERQ_PROGRESS")
    t0 = time.time()
    for ep in range(a.epochs):
        order = rng.permutation(len(train))
        for s in range(steps_per_epoch):
            if step < start:
                step += 1
                continue
            opt.zero_grad(set_to_none=True)
            lsum = 0.0
            for j in order[s * a.accum:(s + 1) * a.accum]:
                split, i = train[j]
                w, truth = data.net(split, i)
                # set per net and left set: checkpoint recompute in backward re-reads it
                set_subst(tw, a.subst, split, i)
                mus = tw.forward(w, ckpt=True)
                rows = mse_rows(mus, truth)
                loss = rows[-1] + a.w_all * torch.stack(rows[:-1]).mean()
                (loss * (1e8 / a.accum)).backward()
                lsum += float(rows[-1].detach())
                del mus, rows, loss, w
            opt.step()
            sched.step()
            step += 1
            msg = (f"ep {ep} step {step}/{total} train final {lsum / a.accum:.4e} "
                   f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
            log.write(msg + "\n")
            if step % 10 == 0:
                print(msg, flush=True)
                log.flush()
            if progress:
                try:
                    Path(progress).write_text(str(step / total))
                except OSError:
                    pass
            if a.max_steps and step >= a.max_steps:
                print(f"smoke: {step} steps, peak VRAM "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
                      f"{(time.time() - t0) / (step * a.accum):.2f} s/net", flush=True)
                return
            if step % a.eval_every == 0 or step == total:
                h = evaluate(f"step {step}")
                if h < best:
                    best, bad = h, 0
                    save_params(out / "params_best.npz", params)
                else:
                    bad += 1
                    if bad >= a.patience:
                        log.write(f"early stop at step {step}\n")
                        print(f"early stop at step {step}", flush=True)
                        checkpoint_state(total)
                        return
                checkpoint_state(step)
    log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["parity", "eval", "train", "sigstats", "d8bprep"])
    ap.add_argument("--cand", default=str(DEFAULT_CAND))
    ap.add_argument("--nets", default="0,1,2")
    ap.add_argument("--ref", default=str(ROOT / "experiments" / "2026-09-25_anatomy2" / "fit20.json"))
    ap.add_argument("--panel", default="dev")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--params", default="")
    ap.add_argument("--stage", default="S1")
    ap.add_argument("--init", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-all", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=28)
    ap.add_argument("--hold-n", type=int, default=100)
    ap.add_argument("--hold-panel", default="hold100")
    ap.add_argument("--train-n", type=int, default=0, help="first N TRAIN rows only")
    ap.add_argument("--train-panel", default="train", help="train | train+d8b | fit80")
    ap.add_argument("--subst", default="", choices=["", *SUBST],
                    help="Step 2: substitute this channel exact at layers >= 1 (FIT rows only)")
    ap.add_argument("--arm", default="", help="optional source arm (see ARMS)")
    ap.add_argument("--raw", action="store_true",
                    help="eval a file with baked-in gains: skip the twin's gain hooks")
    ap.add_argument("--json", default="", help="eval: write per-net results here")
    ap.add_argument("--mask", default="", help="eval with --arm mask: 'R=2.0,H=3.0,D=3.0'")
    ap.add_argument("--dress", default="",
                    help="with --arm dress / mask+dress: 'all' or 'c2=1,d21=1,d22=1,gate=light'")
    ap.add_argument("--gm", default="",
                    help="with --arm gm / mask+gm: 'mode=a|b|c,beta=0.7,gamma=0,q31=keep|drop,q31b=1'")
    ap.add_argument("--k4l", default="",
                    help="with --arm ...+k4l: 'core=1,k211=exact|trunc|none,rad=hub|exact|fold|none,e21b=0|1,old=keep|drop'")
    ap.add_argument("--shared-rank", type=int, default=0, help="override _SHARED_RANK (old tier)")
    ap.add_argument("--bank", default="", help="with --arm ...+bank: 'vg', 'star' or 'vg,star'")
    ap.add_argument("--fold", default="",
                    help="with --arm ...+bank+fold: 'tau=2,mode=ns|st|rec,scale=1,stats=1' (M5.1)")
    ap.add_argument("--rad", default="",
                    help="with --arm ...+rad: 'all' or 'q=1,d22=1,d4=1,scale=1' (M5.2)")
    ap.add_argument("--mask-stats", action="store_true",
                    help="eval with --arm mask: log per-layer alpha kept fractions per net")
    ap.add_argument("--lr-mult", default="",
                    help="train: per-leaf step multipliers, 'gain/rq=300,gain/r4=300'")
    ap.add_argument("--zero-scale", type=float, default=1e-2,
                    help="RMS assumed for a zero-initialised tensor's step size")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (smoke)")
    ap.add_argument("--patience", type=int, default=3)
    a = ap.parse_args()
    {"parity": cmd_parity, "eval": cmd_eval, "train": cmd_train,
     "sigstats": cmd_sigstats, "d8bprep": cmd_d8bprep}[a.cmd](a)


if __name__ == "__main__":
    main()
