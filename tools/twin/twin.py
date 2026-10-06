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
CACHE = ROOT / "cache" / "twin"
#: keenanpepper/arc-whestbench-p2-d8b-corpus-14k shards 00..15 (nets 0..2047, MC N = 1e8,
#: noise ~9e-10 at the final layer): per-layer post-ReLU means, (2048, 16, 1024) float32,
#: row = net idx. Built by `twin.py d8bprep`; weights regenerate from seed 770000 + idx
#: (checked 2026-09-26 against the exact layer-0 closed form: MSE 7e-9 = the N = 1e8 floor,
#: vs 3e-4 for the wrong net, on idx 0, 1, 2, 777, 2047).
D8B_MEAN = ROOT / "cache" / "d8b" / "d8b_mean.npy"
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
LABELS = ROOT / "cache" / "twin_labels"


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
         "            if _role == 'R':\n"
         "                _mR = _m\n"
         "            elif _role == 'H':\n"
         "                _mH = _mS = _m\n"
         "            else:\n"
         "                _mD = _m\n"
         "    _mA = _mW = _mC = _mSl = None\n"
         "    if _mH is not None:\n"
         "        _hm = _MASK.get('Hmode', '')\n"
         "        _mA = {'k': _mS, 'kh': _mS, 'd': _mS, 'w': None, 's': None}.get(_hm, _mH)\n"
         "        _mW = {'d': _mS, 's': None}.get(_hm, _mH)\n"
         "        _mC = {'kh': _mS, 'd': _mS, 'w': None, 's': None}.get(_hm, _mH)\n"
         "        _mSl = {'w': None}.get(_hm, _mS)\n"
         "    if _MASK is not None and _MASK.get('stats') is not None:\n"
         "        _st = {'l': layer_index}\n"
         "        for _t in (2.0, 2.5, 3.0, 3.5):\n"
         "            _st['dead%.1f' % _t] = float((alpha < -_t).sum()) / n\n"
         "            _st['dec%.1f' % _t] = float((alpha * alpha > _t * _t).sum()) / n\n"
         "        _MASK['stats'].append(_st)\n"),
        ("        D22 = ((metric / 6.0) * (dG[:, None] + dG[None, :])) * off_mask\n",
         "        _D21c = D21\n"
         "        if _mD is not None and _MASK.get('Drows'):\n"
         "            D21 = D21 * _mD[:, None]\n"
         "        elif _mD is not None:\n"
         "            D21 = D21 * (_mD[:, None] * _mD[None, :])\n"
         "        if _mD is not None and not _MASK.get('Dfull'):\n"
         "            _D21c = D21\n"
         "        if _mD is not None and _MASK.get('Dzero'):\n"
         "            _kz = _mD\n"
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
        ("    out_tw = _truncate(out_tw, gens, n)\n",
         "    out_tw = _truncate(out_tw, gens, n)\n"
         "    if _mR is not None and _MASK.get('Rpart', '') == '':\n"
         "        out_tw = _mask_rows(out_tw, _mR)\n"
         "    elif _mR is not None and _MASK.get('Rpart', '') == 'n':\n"
         "        out_tw = _mask_rows_new(out_tw, _mR, n)\n"),
    ],
}


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
        if k not in ("R", "H", "D", "Dc", "Dr", "Dz", *hmodes, *rparts):
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
        out[k] = float(v)
    return out


def load_module(path: Path, name="_twin_candidate", arm="", raw=False):
    """Import the candidate from a PATCHED copy (see TWIN_PATCHES), kept for audit.

    `raw=True` skips TWIN_PATCHES (the gain hooks), for EVALUATING a file whose
    gains are already baked in by tools/build_s2.py -- its rewritten lines no
    longer match the hooks' anchors. `install` refuses gain parameters then."""
    import importlib.util
    src = path.read_text(encoding="utf-8")
    for a, b in ([] if raw else TWIN_PATCHES) + (ARMS[arm] if arm else []) + SUB_PATCHES:
        if src.count(a) != 1:
            raise SystemExit(f"twin patch anchor occurs {src.count(a)}x:\n{a}")
        src = src.replace(a, b, 1)
    pp = ROOT / "cache" / "twin" / f"_twin_{path.stem}{'_' + arm if arm else ''}.py"
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
    fin = []
    mask = parse_mask(getattr(a, "mask", ""))
    stats = {} if getattr(a, "mask_stats", False) else None
    if mask or stats is not None:
        assert hasattr(tw.mod, "_MASK"), "--mask needs --arm mask"
    with torch.no_grad():
        for split, i in rows:
            w, truth = data.net(split, i)
            set_subst(tw, getattr(a, "subst", ""), split, i)
            if mask or stats is not None:
                tw.mod._MASK = dict(mask, stats=[] if stats is not None else None)
            fin.append(float(mse_rows(tw.forward(w), truth)[-1]))
            if stats is not None:
                stats[f"{split}:{i}"] = tw.mod._MASK["stats"]
    r = {"panel": a.panel, "arm": getattr(a, "arm", ""), "n": len(fin),
         "raw_mse": float(np.mean(fin)), "per_net": fin}
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
    names = TRAINABLE[a.stage]
    params = {k: tw.leaf(k).clone().requires_grad_(True) for k in names}
    if a.init:
        for k, v in load_params(a.init).items():
            if k in params:
                params[k] = v.clone().requires_grad_(True)
            else:                                   # frozen, but not the file's value
                params[k] = v
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
                                subst=a.subst)
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
    ap.add_argument("--mask-stats", action="store_true",
                    help="eval with --arm mask: log per-layer alpha kept fractions per net")
    ap.add_argument("--zero-scale", type=float, default=1e-2,
                    help="RMS assumed for a zero-initialised tensor's step size")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (smoke)")
    ap.add_argument("--patience", type=int, default=3)
    a = ap.parse_args()
    {"parity": cmd_parity, "eval": cmd_eval, "train": cmd_train,
     "sigstats": cmd_sigstats, "d8bprep": cmd_d8bprep}[a.cmd](a)


if __name__ == "__main__":
    main()
