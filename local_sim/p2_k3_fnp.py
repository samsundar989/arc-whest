"""WP14 / C0-ii: the `fnp` port of the regenerating K3 flow. THE CHASSIS.

Contract: `experiments/LOG.md`, 2026-09-02, "CONTRACT (Rule 1): WP14", HEAD
`d915f83`. **No gate in that contract scores this module.** It is on the critical
path whatever `C1`-`C4` decide, because every branch this project has left ends in
a file the grader can run, and the grader sandbox has no torch.

**THIS PORT IS INCOMPLETE AND THIS DOCSTRING IS THE HONEST STATUS.** What is
finished is stated below; what is not is listed, by name, at the bottom. Nothing
here is imported by `estimator.py` and nothing here has been submitted.

WHY THIS FILE IS THE CRITICAL PATH, AND WHY IT IS NOW WORTH BUILDING
---------------------------------------------------------------------
`C0-iii` (2026-09-02) measured that a regenerating K3 flow at **float32** costs
`99,116,660,736` counted flopscope FLOPs, and that composed with the f32 K2
backbone this repository already measured and reviewed on 2026-08-26
(`68,802,510,336`, accuracy-neutral on 96/96 networks) the total is
`167,919,171,072` = **`0.07636 B`** -- the `0.1` multiplier floor with `0.02364 B`
to spare, with no Strassen and no rank cap.

That composition is a **projection**: the two halves are measured in different
spellings and have never been executed as one program. **This module is what turns
it into a measurement.** Until it runs end to end under `whest validate`, through
`certify_candidate`, and through a brokered residual soak, the `0.07636 B` is
arithmetic and not a candidate.

THE PORT IS BOUNDED, AND THAT IS THE FIRST RESULT HERE
-------------------------------------------------------
Upstream's nonlinear step sums over a term list produced by
`kprop_harmonic.get_all_terms_iso(k_max=3, d_max=4, augment=False)`, filtered by
`factored_keeps_term`. That list depends on **nothing but `(k_max, d_max,
augment)`** -- not on `n`, not on the weights, not on the data. So it can be
enumerated once, frozen, and checked, which converts an open-ended
"re-derive a research codebase in a restricted dialect" into a bounded
"implement 25 named contractions".

Enumerated 2026-09-02 by `local_sim/p2_k3_terms.py` against upstream rev
`93d091a4`:

    35 terms enumerated, 25 kept in SIMPLE mode

    int_part (1,)     3 vec_parts        int_part (1, 1)   4 vec_parts
    int_part (2,)     3 vec_parts        int_part (2, 1)   5 vec_parts
    int_part (3,)     3 vec_parts        int_part (2, 2)   4 vec_parts
    int_part (4,)     3 vec_parts

Artifact `experiments/2026-09-02_wp14_fnp/k3_terms_simple.json`,
sha256 `824358c01b891ce15d3d45755746d5459181abd303ebf9e9ad56af807c75b2a3`.
`--check-terms` re-enumerates against the live upstream checkout and refuses if
the table has drifted, so the frozen copy cannot rot silently.

The `(1,1,1)` int_part is NOT in that list: upstream factors it by hand, and in
SIMPLE mode it is exactly two things -- the wick-scaled incoming `kappa_3`, and
ONE two-leg path block, `legs = [(1, 1, WK_11, 1)]` with `rights` the same single
entry (`factor_k3.py:378-403`). That is the whole of the third-order state update,
and it is why a regenerating flow is three matmuls a layer.

THE ONE PLACE A PORT LIKE THIS GOES WRONG
------------------------------------------
`local_sim/p2_slice_cost.py::_third_order_layer` already writes this layer in
`fnp` -- but it is a COST probe, and its own docstring says "values are
arbitrary". It uses `w1` for both wick factors where upstream uses `w(1)` on one
leg and `w(2) * w(1)` on the other (`factor_k3.py:401-403`). Its FLOP count is
therefore right and its numbers are wrong. **Do not promote that spelling by
copying it.** This module re-derives from upstream and checks against upstream.

TWO VENVS, AND THEY ARE NOT INTERCHANGEABLE
--------------------------------------------
`--check-terms` re-enumerates against the upstream checkout, which imports torch,
so it runs in **`.venv-gpu`** -- which has no flopscope. `--k2-parity` and
`--price` meter real flopscope operations, so they run in **`.venv`** -- which has
no torch. That split is the whole reason this port exists: the grader sandbox is
the `.venv` side, and nothing that needs torch can ever ship.

Run:
    .venv-gpu\Scripts\python.exe local_sim/p2_k3_fnp.py --check-terms
    uv run python local_sim/p2_k3_fnp.py --k2-parity --width 128 --depth 4
    uv run python local_sim/p2_k3_fnp.py --layer-k2    --width 256
    uv run python local_sim/p2_k3_fnp.py --layer-price --width 1024
    uv run python local_sim/p2_k3_fnp.py --price
    uv run python local_sim/p2_k3_fnp.py --todo
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# APPEND, never insert: `local_sim/` holds numpy shims named `flopscope` and
# `whestbench`, and prepending this directory turns every counted FLOP below into
# fiction.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

WIDTH, DEPTH = 1024, 16
BUDGET = 2 ** 41

#: `experiments/2026-09-02_wp14_fnp/k3_terms_simple.json`, frozen 2026-09-02.
TERMS_ARTIFACT = os.path.join(_ROOT, "experiments", "2026-09-02_wp14_fnp",
                              "k3_terms_simple.json")
TERMS_SHA256 = "824358c01b891ce15d3d45755746d5459181abd303ebf9e9ad56af807c75b2a3"

#: C0-iii's measured halves. Both are MEASURED; their SUM is a projection.
K2_ANCHOR_F64 = 137_600_326_656
K2_ANCHOR_F32 = 68_802_510_336
K3_INCREMENT_F64 = 198_233_321_472
K3_INCREMENT_F32 = 99_116_660_736

_INV_SQRT_2PI = 0.3989422804014327


def _require_real_flopscope(flops) -> None:
    path = os.path.abspath(getattr(flops, "__file__", "") or "")
    if path.startswith(os.path.abspath(_HERE) + os.sep):
        raise RuntimeError(
            f"REFUSED: `flopscope` resolved to the local shim at {path}. Every "
            "counted FLOP below would be fiction.")
    if not hasattr(flops, "BudgetContext"):
        raise RuntimeError(f"flopscope at {path} has no BudgetContext.")


# ==========================================================================
# the K2 backbone, in fnp, at a declared dtype
# ==========================================================================

def k2_closure(fnp, w, width: int, depth: int, dtype):
    """The shipped `estimator.py::_closure`, parameterised by dtype.

    Reproduced here rather than imported because `estimator.py` is the shipped
    file and is not touched by research code -- and because the dtype is the
    whole point: `docs/CURRENT_STATE.md` (2026-08-26, independently reviewed)
    records that this closure in float32 costs `0.50002x` its float64 self and
    is accuracy-neutral on 96/96 networks, with the `var_pre` clamp firing zero
    times and zero non-finite outputs.

    **The float32 trap this must avoid**, and the reason the two `.astype` calls
    below are load-bearing: flopscope 0.12 promotes `stats.norm.cdf` / `.pdf`
    inputs to float64 "to match scipy.stats" and emits a warning. An arm that
    does not cast the results back is float32 in name only -- `Phi` and `phi`
    become float64 and so does the covariance accumulator, where nearly all the
    cost lives. That is exactly how `local_sim/p2_arms/arm_k2p1_f32.py` measured
    a 3.1% saving instead of 50%.

    Returns `(mus, phis)`: the per-layer post-ReLU means and the per-layer
    `Phi = Prob(pre-activation > 0)` the K3 step's wick coefficients need.
    """
    import flopscope as flops

    mu = fnp.zeros(width, dtype=dtype)
    cov = fnp.eye(width, dtype=dtype)
    off_mask = fnp.ones((width, width), dtype=dtype) - fnp.eye(width, dtype=dtype)
    mus, phis, sigs, alphas = [], [], [], []
    for wl in w:
        mu_pre = wl.T @ mu
        cov_pre = wl.T @ cov @ wl
        var_pre = fnp.maximum(fnp.diagonal(cov_pre), dtype(1e-12))
        sig = fnp.sqrt(var_pre)
        alpha = mu_pre / sig
        Phi = flops.stats.norm.cdf(alpha).astype(dtype)
        phi = flops.stats.norm.pdf(alpha).astype(dtype)
        mu = sig * (alpha * Phi + phi)
        ez2 = (var_pre + mu_pre * mu_pre) * Phi + mu_pre * sig * phi
        var = fnp.maximum(ez2 - mu * mu, dtype(0.0))
        cov = fnp.outer(Phi, Phi) * cov_pre * off_mask + fnp.diag(var)
        mus.append(mu)
        phis.append(Phi)
        sigs.append(sig)
        alphas.append(alpha)
    return {"mus": mus, "phis": phis, "sigs": sigs, "alphas": alphas}


# ==========================================================================
# ONE K3 LAYER, IN fnp, METERED -- SKEPTIC WP14's named follow-up
# ==========================================================================
#
# `SKEPTIC WP14` rated CLAIM A (the 0.07636 B floor) PROVISIONAL and named the
# cheapest discriminating follow-up:
#
#   "execute ONE layer of the regenerating K3 step in fnp at n = 1024 -- the
#    (1,1,1) block, all 25 kept pK terms and the harmonic projection at d = 3, 4
#    -- inside a real BudgetContext at float32, with the .astype(float32)
#    cast-back after every flops.stats.* call, and read the meter."
#
# That is what `cmd_layer_price` does. Everything the layer needs that is pure
# combinatorics is FROZEN in `k3_coefs_simple.json` (generated from upstream by
# `local_sim/p2_k3_coefs.py`) rather than re-derived here: the 25-term program
# with its scalars and per-axis Wick indices, the pK -> K vector partitions with
# their folded coefficients, the d = 4 Laplacian coefficients, and the
# `_relu_wick_poly` polynomials. `proj_coef(n, 4, 2)` is NOT frozen -- it is a
# closed form in `n` and is evaluated, so a width change cannot invalidate it.
#
# WHAT IS DELIBERATELY NOT UPSTREAM'S SPELLING, AND WHY THAT IS LEGITIMATE
# -----------------------------------------------------------------------
# This is not a replay of upstream's op sequence. The claim under test is
# precisely that a LEAN spelling of the same algorithm is affordable, and
# upstream's own cost (the review's torch dispatch census: 19.4 n^3-matmul-
# equivalents/layer at 512 x 16) is dominated by work a shipped estimator need
# not do. Three lean choices are made here, each exact rather than approximate:
#
#   1. `contract_W` on the two IDENTITY legs of `(M, I, I)` returns `W.T`; no
#      matmul is issued for them.
#   2. The `(2,1)` slice's three block permutations collapse to two distinct
#      matmuls because `A2 == A3`.
#   3. The `d = 4` slices `(3,1)` and `(2,1,1)` are not formed. They are exactly
#      zero in the `r_out = 2` projection -- `_lap_m_dslice` admits no multigraph
#      meeting their arity condition -- and `(2,1,1)` is an `n^3` object upstream
#      materialises for nothing.
#
# Everything else is executed at full width, including objects a cleverer
# spelling could fold to a scalar (`K[4]`'s constant slices), so the meter reads
# HIGH rather than low wherever this port had a choice.
#
# WHAT THIS MEASURES AND WHAT IT DOES NOT
# ---------------------------------------
# Counted FLOPs on this path are value-independent (`p2_slice_cost.py
# --value-invariance`), so the meter reading is a fact about the op sequence, not
# about the numbers flowing through it. **The numbers are NOT yet checked against
# upstream** -- that is step 6 of `NOT_DONE` and no accuracy claim may rest on
# this module until it passes. What is settled here is the price.

_COEFS_ARTIFACT = os.path.join(_ROOT, "experiments", "2026-09-02_wp14_fnp",
                               "k3_coefs_simple.json")
#: sha256 of the frozen coefficient table. `SKEPTIC WP14-FOLLOWUP` found this
#: table load-bearing for BOTH the op sequence and the values, with no hash and
#: no re-enumeration check -- unlike the term table beside it, which has both.
#: A frozen table nobody checks is a liability the moment anything writes to it,
#: and `p2_k3_coefs.py` writes straight over this path.
COEFS_SHA256 = "894185e450533765f633312b36345ab7a29bfb4219a9ab5ee0c36a12d5988fa7"


def load_coefs(*, verify: bool = True) -> dict:
    """The frozen combinatorial tables, REFUSED if they have moved on disk."""
    import hashlib

    raw = open(_COEFS_ARTIFACT, "rb").read()
    if verify:
        got = hashlib.sha256(raw).hexdigest()
        if got != COEFS_SHA256:
            raise RuntimeError(
                f"REFUSED: {os.path.relpath(_COEFS_ARTIFACT, _ROOT)} has changed "
                f"on disk (sha256 {got[:12]} != {COEFS_SHA256[:12]}). Every "
                "coefficient below would be unverified. Re-enumerate with "
                "`.venv-gpu\\Scripts\\python.exe local_sim/p2_k3_coefs.py` and "
                "update COEFS_SHA256 in the same commit as the new table.")
    return json.loads(raw.decode("utf-8"))


class DtypeAudit:
    """Every array the layer produces, with its dtype, and a refusal on drift.

    The trap this exists for is named in `SKEPTIC WP14`: flopscope 0.12 promotes
    `stats.norm.cdf` / `.pdf` inputs to float64 to match scipy, and ONE
    un-cast result re-promotes the graph downstream -- at which point every
    counted FLOP is billed at the float64 rate `2.0` and an "f32 measurement" is
    a float64 measurement wearing its name. It already cost this repository once
    (`local_sim/p2_arms/arm_k2p1_f32.py`, a 3.1% saving where 50% was available).
    """

    def __init__(self, dtype_name: str):
        self.want = dtype_name
        self.seen = {}
        self.violations = []
        self.stats_calls = []
        self.warnings = []

    def check(self, name, arr):
        import numpy as np
        got = str(np.dtype(getattr(arr, "dtype", np.dtype(np.float64))))
        self.seen[name] = got
        if got != self.want:
            self.violations.append({"where": name, "dtype": got, "want": self.want})
        return arr

    def stats(self, name, arr_raw, arr_cast):
        """Record a `flops.stats.*` call: what it RETURNED and what we cast to."""
        import numpy as np
        self.stats_calls.append({
            "call": name,
            "returned": str(np.dtype(arr_raw.dtype)),
            "after_astype": str(np.dtype(arr_cast.dtype)),
            "promoted": str(np.dtype(arr_raw.dtype)) != self.want,
        })
        return self.check(name, arr_cast)

    def summary(self) -> dict:
        return {"requested_dtype": self.want,
                "arrays_checked": len(self.seen),
                "violations": self.violations,
                "stats_calls": self.stats_calls,
                "flopscope_warnings": self.warnings,
                "clean": not self.violations}


def _evalpoly(coefs, x):
    """Horner, with PYTHON floats as the coefficients.

    `np.float64(c)` would be a CONCRETE scalar under NEP-50 and would promote an
    f32 array on contact; a Python float stays weak. This is the same trap
    `p2_slice_cost.py` records in a comment, in a second place.
    """
    y = x * 0.0 + float(coefs[-1])
    for c in reversed(coefs[:-1]):
        y = y * x + float(c)
    return y


def wick_coefs(fnp, flops, mean, sigma, alpha, table, dtype, audit):
    """`wick.relu_wick_coef` for every `(k, p)` the layer needs. O(n) each.

    The two `flops.stats.*` calls below are the only place a float64 can enter,
    and both are cast back immediately and recorded by the audit.
    """
    Phi_raw = flops.stats.norm.cdf(alpha)
    Phi = audit.stats("stats.norm.cdf(alpha)", Phi_raw, Phi_raw.astype(dtype))
    phi_raw = flops.stats.norm.pdf(alpha)
    phi = audit.stats("stats.norm.pdf(alpha)", phi_raw, phi_raw.astype(dtype))

    he = {0: alpha * 0.0 + 1.0, 1: alpha, 2: alpha * alpha - 1.0}

    def base(k):
        if k == 0:
            return sigma * phi + mean * Phi
        if k == 1:
            return Phi
        den = sigma
        for _ in range(k - 2):
            den = den * sigma
        return ((-1.0) ** (k - 2)) * he[k - 2] * phi / den

    out = {}
    for _key, spec in table["wick"].items():
        k, p = int(spec["k"]), int(spec["p"])
        if spec["branch"] == "base":
            v = base(k)
        elif spec["branch"] == "reduce":
            v = float(spec["factorial_p"]) * base(int(spec["k_reduced"]))
        else:
            s_pow = alpha * 0.0 + 1.0
            for _ in range(p - k):
                s_pow = s_pow * sigma
            v = s_pow * (_evalpoly(spec["P1"], alpha) * phi
                         + _evalpoly(spec["P2"], alpha) * Phi)
        out[(k, p)] = audit.check("wick(%d,%d)" % (k, p), v)
    return out


def _proj_coef_n_d4_r2(n: int) -> float:
    """`harmonic.proj_coef(n, 4, 2)[2]`, evaluated rather than frozen.

    c = d - r + n/2 - 1;  coef = (-1)^r (c - r) / (4^j r! (j - r)! c (1-c)^(rising j))
    at r = j = 2, d = 4. Evaluated so a width change cannot silently invalidate it.
    """
    d, r, j = 4, 2, 2
    c = d - r + n / 2.0 - 1
    rising = 1.0
    for m in range(j):
        rising *= (1 - c + m)
    return ((-1.0) ** r * (c - r) / 4.0 ** j / math.factorial(r)
            / math.factorial(j - r) / c / rising)


def _parse_int_part(key: str) -> tuple:
    return tuple(int(x) for x in key.strip("()").split(",") if x.strip())


def _get_pk_block(pK, b, d):
    """`cumulants.DS_part_sum.get_block`: the pK slice a vector-partition block reads.

    A block is a vector like `(1, 0)` or `(1, 2)` over the output slice's `d`
    axes. Zero entries mean the block does not touch that axis, so the lower-order
    slice is broadcast back in; a block whose nonzero entries are not sorted
    descending reads the same stored slice transposed, because `DSTensor` keys its
    slices by the SORTED int_part and `get_slice` permutes on the way out.
    """
    nz = tuple(i for i, x in enumerate(b) if x > 0)
    key = tuple(b[i] for i in nz)
    f = pK[tuple(sorted(key, reverse=True))]
    if len(key) == 2 and key[0] < key[1]:
        f = f.T
    if len(nz) != d:
        f = f[:, None] if nz == (0,) else f[None, :]
    return f


def k3_layer(fnp, flops, W, state, table, dtype, off_mask, audit,
             init_scale=2.0, last_layer=False):
    """ONE full layer -- linear step then nonlinear step -- of the regenerating
    k_max=3 SIMPLE flow, in `fnp`, at the declared dtype.

    State carried between layers, and it is the WHOLE state:
        mu   (n,)     K[1] core, the post-ReLU mean
        cov  (n, n)   K[2] core
        M    (n, n)   K[3] as FactoredTensor factors (M, I, I) -- the
                      REGENERATING form: `from_dstensor` rebuilds it each layer
                      from its own two read-slices, so the rank never grows
        c4   scalar   K[4] as HTensor(core=scalar, r=2). r_x(4, 3, SIMPLE) = 2
                      and 4 - 2*2 = 0, so the core really is order zero.

    The `n^3`-class operations, and there are exactly five:
        cov_pre = W.T @ cov @ W        two   (the K2 backbone)
        P       = W.T @ M              one   (contract_W, leg 1 only)
        (P.Q) @ Q.T  and  (Q.Q) @ P.T  two   (the (2,1) slice's block perms)
    Everything else below is O(n^2) or less, and the meter is what says so.
    """
    n = W.shape[0]
    mu, cov, M, c4 = state["mu"], state["cov"], state["M"], state["c4"]

    # ---- linear step -------------------------------------------------------
    mu_pre = audit.check("mu_pre", W.T @ mu)
    cov_pre = audit.check("cov_pre", W.T @ cov @ W)          # 2 matmuls
    P = audit.check("P", W.T @ M)                            # 1 matmul
    Q = W.T                                                  # the identity legs
    # `set_metric = mlp.init_scale[l]`: the average-case metric E[W W^T].
    # 2.0 AT EVERY LAYER IS CORRECT FOR OUR NETWORKS AND NOT FOR UPSTREAM'S.
    # Upstream's `init_kind='he'` sets `w_var[0] = 1.0` and `w_var[1:] = 2.0`;
    # the organisers' Phase-2 MLPs are scale 2.0 at EVERY layer, which is why
    # `p2_kprop_bridge.py` (convention 3, verified against upstream source) runs
    # the reference with `init_kind='manual', w_var=2.0`. For W with entries
    # ~ N(0, 2/n), E[W W^T] = 2 I. Raised by `SKEPTIC WP14-FOLLOWUP` as a
    # possible defect; it is not one, and it is recorded here so it is not
    # raised a third time. A Python float, so it stays weak under NEP-50.
    metric = float(init_scale)

    # ---- the nonlinearity's Gaussian anchor --------------------------------
    var = audit.check("var", fnp.maximum(fnp.diagonal(cov_pre), dtype(1e-12)))
    sigma = audit.check("sigma", fnp.sqrt(var))
    alpha = audit.check("alpha", mu_pre / sigma)
    w = wick_coefs(fnp, flops, mu_pre, sigma, alpha, table, dtype, audit)

    # ---- diagonal slices of the incoming tower -----------------------------
    # `zero_repeated` is a diagonal kill; `off_mask` is that kill, built once.
    C11 = audit.check("C11", cov_pre * off_mask)
    PQ = P * Q
    D3 = audit.check("D3", (PQ * Q).sum(axis=1))
    D21 = (2.0 * (PQ @ Q.T) + ((Q * Q) @ P.T)) / 3.0         # 2 matmuls
    D21 = audit.check("D21", D21 * off_mask)
    # K[4] = Sym(c4 (x) metric^(x)2) with a DIAGONAL metric: the (4,) slice is
    # c4 * m^2 and the (2,2) slice is c4 * m^2 / 3 off the diagonal. Both are
    # constant here, and both are materialised at full width anyway so the meter
    # charges what a spelling that did NOT know they were constant would pay.
    ones_n = fnp.ones(n, dtype=dtype)
    D4 = audit.check("D4", (c4 * metric * metric) * ones_n)
    D22 = audit.check("D22", (c4 * metric * metric / 3.0)
                      * fnp.outer(ones_n, ones_n) * off_mask)

    S = {(1,): mu_pre, (2,): var, (1, 1): C11, (2, 1): D21, (1, 2): D21.T,
         (3,): D3, (4,): D4, (2, 2): D22}

    # ---- the 25 kept pK terms ---------------------------------------------
    pK = {}
    for row in table["terms"]:
        ip = tuple(row["int_part"])
        d = len(ip)
        term = None
        for blk in row["blocks"]:
            f = S[tuple(blk["vec"])]
            term = f if term is None else term * f
        if term is None:                  # the empty vec_part: the all-ones tensor
            term = ones_n if d == 1 else fnp.outer(ones_n, ones_n)
        term = term * float(row["scalar"])
        for axis, kp in enumerate(row["wick"]):
            wk = w[(int(kp[0]), int(kp[1]))]
            term = term * (wk if d == 1 else (wk[:, None] if axis == 0 else wk[None, :]))
        pK[ip] = term if ip not in pK else pK[ip] + term
    # symmetrise within each Wick-label class of the int_part
    for ip in list(pK):
        if len(ip) == 2 and ip[0] == ip[1]:
            pK[ip] = (pK[ip] + pK[ip].T) * 0.5
        audit.check("pK%s" % (ip,), pK[ip])

    # ---- the (1, 1, 1) block, and its repeated slices -----------------------
    # SIMPLE mode is exactly two things: the wick-scaled incoming kappa_3, and
    # ONE two-leg path block with `legs = [(1, 1, WK_11, 1)]` and `rights` the
    # same single entry (factor_k3.py:378-403). NOTE the asymmetry between w(1)
    # and w(2) * w(1) -- p2_slice_cost.py uses w1 for both and is numerically
    # wrong while being cost-correct.
    w1, w2 = w[(1, 1)], w[(2, 1)]
    fac1 = audit.check("fac1", w1[:, None] * C11)
    fac3 = audit.check("fac3", (w2[:, None] * w1[None, :] * C11).T)
    # Block 1 -- factors (w1.P, w1.Q, w1.Q). The Wick scaling is a rank-1
    # rescale of the UNSCALED slices already computed, so no new matmul is
    # issued: (3,) -> w1^3 . D3 ; (2,1) -> w1_i^2 w1_j . D21.
    w1_sq = w1 * w1
    p111_3 = audit.check("p111_(3,)", w1_sq * w1 * D3)
    p111_21 = audit.check("p111_(2,1)", (w1_sq[:, None] * w1[None, :]) * D21)
    # Block 2 -- factors (fac1, 3 I, fac3^T). The middle leg is a multiple of the
    # identity, so every block permutation degenerates to a row scaling or dies
    # on C11's zero diagonal. Its (3,) slice is identically zero; its (2,1)
    # slice is fac1 . fac3 with the diagonal killed.
    p111_21 = audit.check("p111_(2,1)+blk2", p111_21 + (fac1 * fac3) * off_mask)

    # ---- pK -> K, per diagonal slice (the frozen vector-partition table) ----
    K = {}
    for ip_key, rows in table["pK_to_K"].items():
        ip = _parse_int_part(ip_key)
        d = len(ip)
        acc = None
        for r in rows:
            prod = None
            for b in r["blocks"]:
                f = _get_pk_block(pK, tuple(int(x) for x in b), d)
                prod = f if prod is None else prod * f
            term = prod * float(r["coef"])
            acc = term if acc is None else acc + term
        K[ip] = audit.check("K%s" % (ip,), acc)

    # ---- the outputs -------------------------------------------------------
    mu_out = audit.check("mu_out", K[(1,)])
    cov_out = audit.check("cov_out", fnp.diag(K[(2,)]) + K[(1, 1)])
    # K[3]: subtract the (1,1,1) block's own repeated part, then REGENERATE --
    # `from_dstensor` returns factors (diag(S3) + 3 S21^T, I, I).
    S3 = audit.check("S3", K[(3,)] - p111_3)
    S21 = audit.check("S21", K[(2, 1)] - p111_21)
    M_out = None if last_layer else audit.check(
        "M_out", fnp.diag(S3) + 3.0 * S21.T)
    # K[4]: DS_harmonic_proj(., r_out=2). Both surviving Laplacian multigraphs
    # cap every leg, so the core really is a scalar.
    lap = table["lap2_d4"]
    c4_out = _proj_coef_n_d4_r2(n) * (
        float(lap["(4,)"]["terms"][0]["coef"])
        * float(lap["(4,)"]["int_partition_coef"]) * K[(4,)].sum()
        + float(lap["(2, 2)"]["terms"][0]["coef"])
        * float(lap["(2, 2)"]["int_partition_coef"]) * K[(2, 2)].sum())

    return {"mu": mu_out, "cov": cov_out, "M": M_out, "c4": c4_out}


def _relu_cumulants_quad(mu, sig, deg=300):
    """kappa_1..kappa_3 of ReLU(Z), Z ~ N(mu, sig^2), by GAUSS-LEGENDRE quadrature.

    Deliberately computed a DIFFERENT way from the layer: numerical integration
    rather than the Wick/Hermite closed forms the port uses, so agreement is
    evidence about the port and not a restatement of it.

    **Gauss-HERMITE is the wrong rule here and was tried first.** It integrates
    against `exp(-t^2)` over the whole line, and `max(z, 0)` has a kink at the
    origin, so its convergence is algebraic, not spectral: at `deg = 160` it
    disagreed with the exact `sigma phi + mu Phi` by `1.1e-3` -- an error in the
    REFERENCE that would have been read as an error in the port. Splitting the
    integral at the kink and using Gauss-Legendre on `[0, max(mu, 0) + 14 sigma]`
    leaves an analytic integrand, and the discarded tail is below `1e-42` of the
    mass.
    """
    import numpy as np

    x, wq = np.polynomial.legendre.leggauss(deg)
    if not (np.isfinite(x).all() and np.isfinite(wq).all()):
        raise RuntimeError("REFUSED: leggauss(%d) returned non-finite nodes." % deg)
    mu = np.asarray(mu, dtype=np.float64)
    sig = np.asarray(sig, dtype=np.float64)
    upper = np.maximum(mu, 0.0) + 14.0 * sig                      # (n,)
    z = 0.5 * upper[:, None] * (x[None, :] + 1.0)                 # (n, deg)
    jac = 0.5 * upper[:, None] * wq[None, :]
    dens = (np.exp(-0.5 * ((z - mu[:, None]) / sig[:, None]) ** 2)
            / (sig[:, None] * np.sqrt(2.0 * np.pi)))
    m1 = (z * dens * jac).sum(1)
    m2 = (z ** 2 * dens * jac).sum(1)
    m3 = (z ** 3 * dens * jac).sum(1)
    return m1, m2 - m1 ** 2, m3 - 3 * m1 * m2 + 2 * m1 ** 3


#: The mutations `SKEPTIC WP14-FOLLOWUP` ran against `--layer-k2`, and the
#: coverage number they produce. **Seven targeted value-destroying edits all
#: passed all three checks identically, one of them 20% cheaper** -- so the
#: sentence "the price is a price OF SOMETHING" was true of the checks that
#: exist and false of the program that is priced.
#:
#: The reason is structural and worth stating: at `M = 0` we have
#: `P = W.T @ M = 0`, so `D3` and `D21` are identically zero, the two
#: `(2,1)`-slice matmuls execute on zeros, and `S21`'s diagonal is zero -- so
#: `diag(M_out) = S3` and the whole off-diagonal of `M_out` is never read.
#: **The `kappa_3 = kappa_4 = 0` limit exercises the cheap half and prices the
#: expensive half.**
#:
#: `--mutation-coverage` runs them so the gap is a MEASURED NUMBER rather than a
#: sentence, and so it cannot quietly close or quietly widen. It is not a fix:
#: closing it needs parity against upstream at non-zero `kappa_3`
#: (`NOT_DONE` step 6), which needs torch and therefore the other venv.
MUTATIONS = [
    ("d21_coef_3_to_2", "the (2,1) slice's 1/3 coefficient",
     "/ 3.0)", "/ 2.0)"),
    ("d21_drop_block_perm", "one of the two surviving block permutations "
                            "-- ONE MATMUL CHEAPER",
     "D21 = (2.0 * (PQ @ Q.T) + ((Q * Q) @ P.T)) / 3.0",
     "D21 = (3.0 * (PQ @ Q.T)) / 3.0"),
    ("fac3_w2_to_w1", "the w(2)/w(1) asymmetry -- the exact p2_slice_cost.py "
                      "defect this module's docstring warns about",
     'audit.check("fac3", (w2[:, None] * w1[None, :] * C11).T)',
     'audit.check("fac3", (w1[:, None] * w1[None, :] * C11).T)'),
    ("rebuild_3_to_2", "from_dstensor's `diag(S3) + 3 S21^T` rebuild",
     '"M_out", fnp.diag(S3) + 3.0 * S21.T)',
     '"M_out", fnp.diag(S3) + 2.0 * S21.T)'),
    ("drop_block2", "the (1,1,1) two-leg block -- the mechanism WP15 rests on",
     'p111_21 + (fac1 * fac3) * off_mask)',
     'p111_21 + (fac1 * fac3) * off_mask * 0.0)'),
    ("s21_sign", "the sign of the repeated-part subtraction",
     'audit.check("S21", K[(2, 1)] - p111_21)',
     'audit.check("S21", K[(2, 1)] + p111_21)'),
    ("d3_half", "the (3,) slice",
     'audit.check("D3", (PQ * Q).sum(axis=1))',
     'audit.check("D3", 0.5 * (PQ * Q).sum(axis=1))'),
]


def cmd_mutation_coverage(a) -> int:
    """How much of the layer do the `--layer-k2` checks actually pin? Measured.

    Each mutation is applied to a SCRATCH COPY of this module, which is imported
    under its own name; the repository file is never written. A mutation that
    still passes is a hole, named.
    """
    import importlib.util
    import shutil
    import tempfile

    with open(os.path.abspath(__file__), encoding="utf-8") as fh:
        src = fh.read()
    # Each anchor appears TWICE in this file -- once in `k3_layer`, where it is
    # code, and once in `MUTATIONS` above, where it is data. Mutate only the
    # region before the table, or every anchor is ambiguous and nothing runs.
    split = src.index("MUTATIONS = [")
    head, tail = src[:split], src[split:]

    rows = []
    tmp = tempfile.mkdtemp(prefix="k3mut_")
    try:
        for name, what, old, new in MUTATIONS:
            if head.count(old) != 1:
                rows.append({"mutation": name, "target": what,
                             "applied": False, "detected": None,
                             "note": f"anchor matched {head.count(old)} times "
                                     "in the code region"})
                continue
            path = os.path.join(tmp, f"mut_{name}.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(head.replace(old, new, 1) + tail)
            spec = importlib.util.spec_from_file_location(f"mut_{name}", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            # The mutant lives in a temp directory, so its own `_HERE` / `_ROOT`
            # point there and it would look for the frozen tables under %TEMP%.
            # Repoint it at the real repository: we are mutating the MATH, not
            # the file layout.
            mod._HERE, mod._ROOT = _HERE, _ROOT
            mod._COEFS_ARTIFACT = _COEFS_ARTIFACT
            mod.TERMS_ARTIFACT = TERMS_ARTIFACT
            args = argparse.Namespace(width=a.width, seed=a.seed, out="",
                                      depth=DEPTH)
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = mod.cmd_layer_k2_degeneracy(args)
            detected = (rc != 0)
            rows.append({"mutation": name, "target": what, "applied": True,
                         "detected": detected})
            print(f"  {name:<22} {'DETECTED' if detected else 'SURVIVES '}  {what}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    live = [r for r in rows if r.get("applied")]
    killed = [r for r in live if r["detected"]]
    print()
    print(f"  mutation coverage of --layer-k2:  {len(killed)} / {len(live)} killed")
    if len(killed) < len(live):
        print("  The surviving mutations are the measure of what the")
        print("  kappa_3 = kappa_4 = 0 limit CANNOT see. Closing it needs parity")
        print("  against upstream at non-zero kappa_3 -- NOT_DONE step 6.")
    if a.out:
        _dump(a.out, {
            "check": "mutation coverage of the --layer-k2 closed-form checks",
            "origin": "SKEPTIC WP14-FOLLOWUP; seven mutations, all surviving",
            "width": a.width, "seed": a.seed, "rows": rows,
            "killed": len(killed), "live": len(live),
            "interpretation": "a SURVIVING mutation is a hole in CLAIM B, not a "
                              "pass; the number is reported so it cannot quietly "
                              "close or quietly widen",
        })
    # This command REPORTS coverage; it does not fail on a low number, because
    # the number is known to be low and the fix is elsewhere.
    return 0


def cmd_layer_k2_degeneracy(a) -> int:
    """Three independent checks on the layer with `kappa_3 = 0` and `kappa_4 = 0`.

    A price measured on an op sequence that computes nothing is worthless, and a
    full parity check against upstream needs torch (`NOT_DONE` step 6). These are
    the checks that do not need it. Setting `M = 0` and `c4 = 0` kills every term
    carrying a `(3,)`, `(4,)`, `(2,1)` or `(2,2)` block and leaves a flow whose
    answer is known in closed form:

    **A. the mean and the variance are the K2 closure, exactly.**
    `pK(1,) = w(0,1)` and `K(2,) = w(0,2) - w(0,1)^2` are `E[ReLU]` and
    `Var[ReLU]` term for term.

    **B. the off-diagonal covariance is the K2 closure PLUS one k_max=3 term.**
    `pK(1,1)` keeps two live blocks, not one:
        `Phi_i Phi_j C_ij`  +  `0.5 w(2)_i w(2)_j C_ij^2`.
    The second is the second-order Edgeworth correction K2 does not have, so a
    layer that reproduced K2 exactly here would be WRONG. Its size is checked too.

    **C. the third-cumulant diagonal against Gauss-Hermite quadrature.**
    `S3 = 2 pK1^3 - 3 pK1 pK2 + pK3` must be the third cumulant of `ReLU(Z_i)`.
    That is an identity about the ReLU and the Gaussian, computed here by
    numerical integration -- an independent route through the `p = 1, 2, 3` Wick
    polynomial branches and the frozen `pK -> K` coefficients `(+2, -3, +1)`.
    """
    import numpy as np

    import flopscope as flops
    import flopscope.numpy as fnp

    _require_real_flopscope(flops)
    table = load_coefs()
    n = a.width
    rng = np.random.default_rng(a.seed)
    rows = []
    A = rng.standard_normal((n, n)) / n ** 0.5
    cov_np = np.eye(n) + 0.3 * (A @ A.T)
    W_np = rng.standard_normal((n, n)) * (2.0 / n) ** 0.5
    mu_np = np.abs(rng.standard_normal(n))

    for dt_name, dt, npdt in (("f64", fnp.float64, np.float64),
                              ("f32", fnp.float32, np.float32)):
        audit = DtypeAudit(str(np.dtype(npdt)))

        def mk(x, _npdt=npdt):
            return fnp.array(np.asarray(x, dtype=_npdt))

        W, mu, cov = mk(W_np), mk(mu_np), mk(cov_np)
        with flops.BudgetContext(flop_budget=10 * BUDGET, quiet=True):
            off_mask = fnp.ones((n, n), dtype=dt) - fnp.eye(n, dtype=dt)
            got = k3_layer(fnp, flops, W, {"mu": mu, "cov": cov,
                                           "M": fnp.zeros((n, n), dtype=dt),
                                           "c4": 0.0},
                           table, dt, off_mask, audit)
            # the K2 reference, spelled exactly as the shipped closure does
            mu_pre = W.T @ mu
            cov_pre = W.T @ cov @ W
            var_pre = fnp.maximum(fnp.diagonal(cov_pre), dt(1e-12))
            sig = fnp.sqrt(var_pre)
            alpha = mu_pre / sig
            Phi = flops.stats.norm.cdf(alpha).astype(dt)
            phi = flops.stats.norm.pdf(alpha).astype(dt)
            ref_mu = sig * (alpha * Phi + phi)
            ez2 = (var_pre + mu_pre * mu_pre) * Phi + mu_pre * sig * phi
            ref_var = ez2 - ref_mu * ref_mu
            # w(k=2, p=1) = -He_0(alpha) phi(alpha) / sigma ... at k=2 the sign
            # is (-1)^0 = +1, so w(2) = phi / sigma.
            w2 = phi / sig
            C11 = cov_pre * off_mask
            k2_cov = fnp.outer(Phi, Phi) * C11 + fnp.diag(ref_var)
            edge = 0.5 * fnp.outer(w2, w2) * (C11 * C11)
            ref_cov_k3 = k2_cov + edge

        def relrms(g, r):
            g = np.asarray(g, dtype=np.float64)
            r = np.asarray(r, dtype=np.float64)
            return float(np.sqrt(((g - r) ** 2).mean()
                                 / max((r ** 2).mean(), 1e-300)))

        got_cov = np.asarray(got["cov"], dtype=np.float64)
        m_out = np.asarray(got["M"], dtype=np.float64)
        # S21 has a zero diagonal (every pK(2,1) block carries C11), so the
        # diagonal of `M = diag(S3) + 3 S21^T` is S3 itself.
        s3 = np.diag(m_out)
        q1, q2, q3 = _relu_cumulants_quad(np.asarray(mu_pre, dtype=np.float64),
                                          np.asarray(sig, dtype=np.float64))
        row = {
            "dtype": dt_name,
            "A_rel_rms_mu_vs_k2": relrms(got["mu"], ref_mu),
            "A_rel_rms_var_vs_k2": relrms(np.diag(got_cov), ref_var),
            "B_rel_rms_cov_vs_k2_plus_edge": relrms(got_cov, ref_cov_k3),
            "B_rel_rms_cov_vs_bare_k2": relrms(got_cov, np.asarray(k2_cov,
                                                                   dtype=np.float64)),
            "B_edge_term_relative_size": relrms(np.asarray(edge, dtype=np.float64) * 0,
                                                np.asarray(edge, dtype=np.float64)) * 0
            + float(np.abs(np.asarray(edge, dtype=np.float64)).max()
                    / max(np.abs(np.asarray(k2_cov, dtype=np.float64)).max(), 1e-300)),
            "C_rel_rms_S3_vs_quadrature": relrms(s3, q3),
            "C_rel_rms_mu_vs_quadrature": relrms(got["mu"], q1),
            "C_rel_rms_var_vs_quadrature": relrms(np.diag(got_cov), q2),
            "c4_out": float(got["c4"]),
            "audit": audit.summary(),
        }
        rows.append(row)
        print("  %s  A mu %.3e  var %.3e  |  B cov(+edge) %.3e  cov(bare K2) %.3e"
              % (dt_name, row["A_rel_rms_mu_vs_k2"], row["A_rel_rms_var_vs_k2"],
                 row["B_rel_rms_cov_vs_k2_plus_edge"], row["B_rel_rms_cov_vs_bare_k2"]))
        print("        C vs Gauss-Legendre quadrature:  kappa1 %.3e  kappa2 %.3e"
              "  kappa3 %.3e"
              % (row["C_rel_rms_mu_vs_quadrature"], row["C_rel_rms_var_vs_quadrature"],
                 row["C_rel_rms_S3_vs_quadrature"]))
        print("        edge term is %.2e of the K2 covariance; c4_out %.3e"
              % (row["B_edge_term_relative_size"], row["c4_out"]))

    # Two tolerance families, because they are limited by different things: the
    # closed-form checks by the arithmetic, the quadrature checks by the rule.
    tol = {"f64": 1e-11, "f32": 2e-5}
    tol_quad = {"f64": 1e-9, "f32": 2e-5}
    keys = ["A_rel_rms_mu_vs_k2", "A_rel_rms_var_vs_k2",
            "B_rel_rms_cov_vs_k2_plus_edge"]
    keys_quad = ["C_rel_rms_S3_vs_quadrature", "C_rel_rms_mu_vs_quadrature",
                 "C_rel_rms_var_vs_quadrature"]
    import math as _math
    bad = [(r["dtype"], k) for r in rows for k in keys
           if not _math.isfinite(r[k]) or r[k] > tol[r["dtype"]]]
    bad += [(r["dtype"], k) for r in rows for k in keys_quad
            if not _math.isfinite(r[k]) or r[k] > tol_quad[r["dtype"]]]
    # The edge term must be REAL: a layer agreeing with bare K2 would mean the
    # k_max=3 covariance correction was silently dropped.
    for r in rows:
        if r["B_rel_rms_cov_vs_bare_k2"] < 1e-9:
            bad.append((r["dtype"], "edge term missing"))
    print("  tolerances: closed-form f64 1e-11 / f32 2e-5; quadrature f64 1e-9 /"
          " f32 2e-5; the k_max=3 edge term must be non-zero")
    for d, k in bad:
        print("  FAIL  %s  %s" % (d, k))
    print("  VERDICT  %s" % ("PASS" if not bad else "FAIL"))
    if a.out:
        _dump(a.out, {"check": "fnp K3 layer with kappa_3 = kappa_4 = 0: K2 collapse, "
                               "the k_max=3 edge term, and Gauss-Hermite cumulants",
                      "width": n, "seed": a.seed, "rows": rows,
                      "tolerances": tol, "tolerances_quadrature": tol_quad,
                      "failures": [list(b) for b in bad], "pass": not bad})
    return 0 if not bad else 1


def k3_flow(fnp, flops, weights, table, dtype, audit, depth=None):
    """The WHOLE regenerating flow: `depth` layers, one program, one context.

    Input cumulants for `x ~ N(0, I)`: `K[1] = 0`, `K[2] = I`, `K[3] = 0`,
    `K[4] = 0`. Returns the `(depth, width)` array of post-ReLU means, which is
    exactly what the grader asks for.

    This exists because `SKEPTIC WP14-FOLLOWUP` refused to take `16 x` one layer
    on trust and built it. Three of the things that extrapolation asserted are
    wrong or backwards, and the flow is what says so:

    * **layer 0 is not cheaper, it is dearer.** Nothing branches on `cov = I` or
      `M = 0`, so it pays the same matmuls plus the `off_mask` build.
    * **the last layer saves almost nothing** -- `last_layer=True` skips the `M`
      rebuild (`2,098,176` FLOPs, `0.001` matmul) and NOT the two `(2,1)`-slice
      matmuls, which the `(2,1)` pK terms need at every layer.
    * the `off_mask` build is amortised over `depth`, which is the one place the
      extrapolation really was conservative.

    Net, the flow comes in BELOW `16 x` the steady-state layer -- so `0.07886 B`
    was an upper bound, but for one reason rather than three.
    """
    n = weights[0].shape[0]
    depth = len(weights) if depth is None else depth
    off_mask = fnp.ones((n, n), dtype=dtype) - fnp.eye(n, dtype=dtype)
    state = {"mu": fnp.zeros(n, dtype=dtype),
             "cov": fnp.eye(n, dtype=dtype),
             "M": fnp.zeros((n, n), dtype=dtype),
             "c4": 0.0}
    mus = []
    for l in range(depth):
        state = k3_layer(fnp, flops, weights[l], state, table, dtype, off_mask,
                         audit, last_layer=(l == depth - 1))
        mus.append(state["mu"])
    return fnp.stack(mus), state


def cmd_flow_price(a) -> int:
    """The whole flow, metered -- FLOPs, operation count and residual.

    **The operation count and the residual are the point, not the FLOPs.**
    Phase 2 sets `lambda = 0`, so FLOPs below the floor are free and the `0.4 s`
    per-MLP residual cliff is the only cost that can still fail an MLP -- and it
    tracks the NUMBER of operations and their Python glue, not FLOPs
    (`docs/GOTCHAS.md`). `SKEPTIC WP14-FOLLOWUP` measured this flow at ~5,300
    operations against the shipped closure's 453, with a local in-process
    residual that crossed `0.4 s` on 2 of 6 readings, and that is the finding
    this command exists to keep in front of anyone reading a FLOP number.

    **The absolute second is NOT quotable** (`AGENTS.md`: authority is
    `--runner subprocess`; the shim and this host are not the grader). What
    transfers is the RATIO to a matched control measured in the same process on
    the same host in the same run, so the K2 closure is run beside it every
    time and both are reported.
    """
    import numpy as np

    import flopscope as flops
    import flopscope.numpy as fnp

    _require_real_flopscope(flops)
    table = load_coefs()
    n, d = a.width, a.depth
    from mint_provenance import canonical_mint
    w_np, _e, _s = canonical_mint(n, d, a.seed)

    rows = []
    for rep in range(a.reps):
        for arm in ("k2", "k3"):
            audit = DtypeAudit(str(np.dtype(np.float32)))
            w = [fnp.array(np.asarray(x, dtype=np.float32)) for x in w_np]
            with flops.BudgetContext(flop_budget=10 * BUDGET, quiet=True) as ctx:
                if arm == "k2":
                    out = k2_closure(fnp, w, n, d, fnp.float32)
                    final = out["mus"][-1]
                else:
                    stacked, st = k3_flow(fnp, flops, w, table, fnp.float32,
                                          audit)
                    final = st["mu"]
            used = float(ctx.flops_used)
            ops = len(getattr(ctx, "op_log", []) or [])
            resid = float(getattr(ctx, "residual_wall_time_s", float("nan")))
            finite = bool(np.isfinite(np.asarray(final, dtype=np.float64)).all())
            rows.append({"rep": rep, "arm": arm, "flops": used, "ops": ops,
                         "residual_wall_time_s": resid, "finite_output": finite,
                         "dtype_violations": len(audit.summary()["violations"])})

    def agg(arm, key, fn):
        return fn([r[key] for r in rows if r["arm"] == arm])

    k3f = agg("k3", "flops", max)
    k2f = agg("k2", "flops", max)
    k3ops, k2ops = agg("k3", "ops", max), agg("k2", "ops", max)
    k3r, k2r = agg("k3", "residual_wall_time_s", max), agg("k2", "residual_wall_time_s", max)
    matmul = 2.0 * n ** 3
    print(f"  {a.reps} reps, {n} x {d}, float32, real canonical weights\n")
    print(f"  {'':<26}{'K2 closure':>18}{'K3 flow':>18}{'ratio':>10}")
    print(f"  {'counted FLOPs':<26}{k2f:>18,.0f}{k3f:>18,.0f}{k3f / k2f:>10.2f}")
    print(f"  {'flopscope operations':<26}{k2ops:>18,}{k3ops:>18,}{k3ops / max(k2ops, 1):>10.2f}")
    print(f"  {'residual (worst, LOCAL)':<26}{k2r:>18.4f}{k3r:>18.4f}{k3r / max(k2r, 1e-9):>10.2f}")
    print(f"  {'headroom vs the 0.4 s cliff':<26}{0.4 / max(k2r, 1e-9):>17.1f}x"
          f"{0.4 / max(k3r, 1e-9):>17.1f}x")
    print()
    print(f"  K3 flow x budget          {k3f / BUDGET:.6f}     multiplier "
          f"{max(0.1, k3f / BUDGET):.5f}")
    print(f"  K3 flow / layer           {k3f / d:>18,.0f} = {k3f / d / matmul:.3f}"
          f" dense f32 {n}-matmuls")
    print(f"  vs 16 x the steady-state layer ({DEPTH} x 10,837,935,618 = "
          f"173,406,969,888): {k3f - 173_406_969_888:+,.0f}")
    print()
    print("  THE ABSOLUTE RESIDUAL SECOND IS NOT QUOTABLE. Authority is")
    print("  `whest run --runner subprocess`. What transfers is the RATIO to the")
    print("  K2 control above, measured in the same process, same host, same run.")
    if a.out:
        _dump(a.out, {
            "check": "the whole regenerating K3 flow: FLOPs, OPERATION COUNT and "
                     "local residual, against a matched K2 control",
            "origin": "SKEPTIC WP14-FOLLOWUP: the op count and the 0.4 s cliff, "
                      "not the FLOPs, are what can still fail an MLP at lambda=0",
            "width": n, "depth": d, "seed": a.seed, "reps": a.reps,
            "rows": rows,
            "k3_flops": k3f, "k2_flops": k2f,
            "k3_ops": k3ops, "k2_ops": k2ops,
            "k3_residual_worst_local_s": k3r, "k2_residual_worst_local_s": k2r,
            "residual_ratio_k3_over_k2": k3r / max(k2r, 1e-9),
            "k3_x_budget": k3f / BUDGET,
            "extrapolation_16x_one_layer": 173_406_969_888,
            "caveat": "LOCAL in-process residual. Not quotable as an absolute; "
                      "the authority is --runner subprocess. The RATIO to the "
                      "matched K2 control is what transfers.",
        })
    return 0


def cmd_layer_price(a) -> int:
    """SKEPTIC WP14's follow-up: one metered layer, at both dtypes, audited."""
    import warnings

    import numpy as np

    import flopscope as flops
    import flopscope.numpy as fnp

    _require_real_flopscope(flops)
    table = load_coefs()
    n = a.width
    rng = np.random.default_rng(a.seed)
    rows = []
    for dt_name, dt, npdt in (("f64", fnp.float64, np.float64),
                              ("f32", fnp.float32, np.float32)):
        audit = DtypeAudit(str(np.dtype(npdt)))

        def mk(x, _npdt=npdt):
            return fnp.array(np.asarray(x, dtype=_npdt))

        # Built OUTSIDE the BudgetContext, exactly as the grader builds weights.
        W = mk(rng.standard_normal((n, n)) * (2.0 / n) ** 0.5)
        state = {"mu": mk(np.abs(rng.standard_normal(n))),
                 "cov": mk(np.eye(n) + 1e-3 * rng.standard_normal((n, n))),
                 "M": mk(rng.standard_normal((n, n)) * 1e-3),
                 "c4": float(1e-4)}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with flops.BudgetContext(flop_budget=10 * BUDGET, quiet=True) as ctx:
                off_mask = fnp.ones((n, n), dtype=dt) - fnp.eye(n, dtype=dt)
                out = k3_layer(fnp, flops, W, state, table, dt, off_mask, audit)
            used = float(ctx.flops_used)
        audit.warnings = [str(x.message)[:160] for x in caught]
        for key in ("mu", "cov", "M"):
            audit.check("out.%s" % key, out[key])
        matmul = 2.0 * n ** 3 * (2.0 if dt_name == "f64" else 1.0)
        st = audit.summary()
        rows.append({"dtype": dt_name, "width": n, "flops_one_layer": used,
                     "n3_matmul_equivalents": used / matmul,
                     "over_n3": used / n ** 3, "audit": st})
        print("  %s  one layer counted %18s   = %6.3f dense %d-matmuls"
              "   = %6.3f n^3"
              % (dt_name, format(used, ",.0f"), used / matmul, n, used / n ** 3))
        print("        dtype audit: %d arrays checked, %d violations, clean=%s"
              % (st["arrays_checked"], len(st["violations"]), st["clean"]))
        for c in st["stats_calls"]:
            print("        %-26s returned %-8s -> cast %-8s promoted=%s"
                  % (c["call"], c["returned"], c["after_astype"], c["promoted"]))
        for v in st["violations"][:8]:
            print("        VIOLATION %s is %s" % (v["where"], v["dtype"]))
        if st["flopscope_warnings"]:
            print("        flopscope warnings: %d" % len(st["flopscope_warnings"]))
            for wmsg in st["flopscope_warnings"][:3]:
                print("          %s" % wmsg)

    f32 = [r for r in rows if r["dtype"] == "f32"][0]
    f64 = [r for r in rows if r["dtype"] == "f64"][0]
    # The projection is a 1024 x 16 figure. Comparing a narrower probe against it
    # would be an apples-to-oranges ratio, so it is only formed at the live width.
    at_live_width = (n == WIDTH)
    per_layer_projection = (K2_ANCHOR_F32 + K3_INCREMENT_F32) / DEPTH
    est_total = f32["flops_one_layer"] * DEPTH
    print()
    print("  f32 / f64 counted ratio  %.5f   (0.5 is the flopscope dtype-rate"
          " ratio; above it means an op promoted)"
          % (f32["flops_one_layer"] / f64["flops_one_layer"]))
    if at_live_width:
        print("  projection's implied per-layer f32 cost   %18s"
              % format(per_layer_projection, ",.0f"))
        print("  measured here, one layer, f32             %18s"
              % format(f32["flops_one_layer"], ",.0f"))
        print("  ratio measured / projected                %18.4f"
              % (f32["flops_one_layer"] / per_layer_projection))
        print("  x %d layers (NO last-layer saving)         %18s   = %.5f B"
              "   multiplier %.5f"
              % (DEPTH, format(est_total, ",.0f"), est_total / BUDGET,
                 max(0.1, est_total / BUDGET)))
        head = int(0.1 * BUDGET) - est_total
        print("  headroom under the 0.1 floor              %18s   = %.5f B"
              "   = %.2f f32 %d-matmuls/layer"
              % (format(head, ",.0f"), head / BUDGET,
                 head / (2.0 * n ** 3) / DEPTH, n))
    else:
        print("  width %d is NOT the live shape: the 1024 x 16 projection is not"
              " comparable and is not printed." % n)
    print()
    print("  This is ONE layer of a real program, not a whole estimator: it says")
    print("  nothing about whest validate, the 4 GiB single-array rule, the 8 GiB")
    print("  cap or the 0.4 s residual cliff, and its NUMBERS are not yet checked")
    print("  against upstream (NOT_DONE step 6). What is settled is the price.")
    if a.out:
        _dump(a.out, {
            "check": "SKEPTIC WP14 follow-up: one metered fnp K3 layer",
            "contract": "LOG 2026-09-02 CONTRACT WP14, HEAD d915f83; "
                        "SKEPTIC WP14 'cheapest discriminating follow-up'",
            "width": n, "seed": a.seed, "depth_for_projection": DEPTH,
            "flopscope_version": str(getattr(flops, "__version__", "unknown")),
            "dtype_rates": _dtype_rates(),
            "rows": rows,
            "k2_anchor_f32": K2_ANCHOR_F32,
            "k3_increment_f32": K3_INCREMENT_F32,
            "projection_per_layer_f32": per_layer_projection,
            "measured_x_depth": est_total,
            "measured_x_budget": est_total / BUDGET,
            "at_live_width": at_live_width,
            "caveat": "cost only; the layer's VALUES are not yet checked against "
                      "upstream (NOT_DONE step 6)",
        })
    return 0


def _dtype_rates() -> dict:
    """flopscope's ACTIVE per-dtype billing rates, read from the package."""
    try:
        from flopscope._weights import _ACTIVE_DTYPE_RATES
        return {k: float(v) for k, v in sorted(_ACTIVE_DTYPE_RATES.items())}
    except Exception as exc:      # a private name; record the failure, never guess
        return {"UNREADABLE": str(exc)}


# ==========================================================================
# the frozen term table, and the check that it has not rotted
# ==========================================================================

def load_terms() -> dict:
    with open(TERMS_ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


def cmd_check_terms(a) -> int:
    """Re-enumerate against the LIVE upstream checkout and refuse on drift.

    A frozen table is a liability the moment nobody checks it. This is the check,
    and it is why the table may be trusted as a literal in the port below.
    """
    import hashlib

    doc = load_terms()
    sha = hashlib.sha256(open(TERMS_ARTIFACT, "rb").read()).hexdigest()
    print(f"  artifact   {os.path.relpath(TERMS_ARTIFACT, _ROOT)}")
    print(f"  sha256     {sha}")
    print(f"  matches    {sha == TERMS_SHA256}")
    if sha != TERMS_SHA256:
        print("  REFUSED: the frozen term table has changed on disk.")
        return 1

    import p2_kprop_bridge as bridge

    src = bridge._src()
    v = bridge.verify_upstream(src)
    if not v["pass"]:
        print("  REFUSED: upstream is not rev 93d091a4 plus the recorded patch.")
        return 1
    from mlp_kprop.kprop_harmonic import factored_keeps_term, get_all_terms_iso

    live = get_all_terms_iso(k_max=3, d_max=4, augment=False)
    rows = []
    for int_part, vpd in live.items():
        for vec_part, count in vpd.items():
            keep = (len(int_part) <= 3
                    and int_part not in [(3, 1), (2, 1, 1)]
                    and int_part != (1, 1, 1)
                    and (int_part, set(vec_part)) != ((2, 1, 1), {(1, 1, 1)})
                    and factored_keeps_term(3, int_part, vec_part))
            rows.append((tuple(int_part), tuple(map(tuple, vec_part)), count, keep))
    frozen = {(tuple(r["int_part"]), tuple(map(tuple, r["vec_part"])), r["count"],
               r["kept_by_factored_simple"]) for r in doc["terms"]}
    same = frozen == set(rows)
    n_kept = sum(1 for r in rows if r[3])
    print(f"  upstream   {src}  rev 93d091a4 VERIFIED")
    print(f"  enumerated {len(rows)} terms, {n_kept} kept in SIMPLE mode")
    print(f"  identical to the frozen table: {same}")
    return 0 if same else 1


# ==========================================================================
# parity and pricing
# ==========================================================================

def cmd_k2_parity(a) -> int:
    """The fnp K2 backbone against `p2_cal_lab`'s NumPy mirror, at both dtypes.

    This is the half of the chassis that is FINISHED. It is checked against the
    mirror that is itself verified to `4e-30` relative mean square against the
    instrumented arm this project ships, so agreement here means the wiring is
    right -- not that K3 is.
    """
    import numpy as np

    import flopscope as flops
    import flopscope.numpy as fnp

    _require_real_flopscope(flops)
    from mint_provenance import canonical_mint
    import p2_cal_lab as cal

    w_np, _e, _s = canonical_mint(a.width, a.depth, a.seed)
    # `p2_cal_lab.closure` returns (depth, width) post-ReLU means, layer 0 first,
    # and is itself verified to 4e-30 relative mean square against the
    # instrumented arm this project ships. It is the reference, not a rewrite.
    ref_mu = cal.closure(w_np)[-1]

    rows = []
    for dt_name, dt in (("f64", fnp.float64), ("f32", fnp.float32)):
        w = [fnp.array(np.asarray(x, dtype=np.float64)).astype(dt) for x in w_np]
        with flops.BudgetContext(flop_budget=10 * BUDGET, quiet=True) as ctx:
            out = k2_closure(fnp, w, a.width, a.depth, dt)
        used = float(ctx.flops_used)
        mu = np.asarray(out["mus"][-1], dtype=np.float64)
        rel = float(np.sqrt(((mu - ref_mu) ** 2).mean()
                            / max((ref_mu ** 2).mean(), 1e-300)))
        rows.append({"dtype": dt_name, "flops": used, "x_budget": used / BUDGET,
                     "rel_rms_vs_mirror": rel})
        print(f"  {dt_name}   counted {used:>18,.0f}   = {used / BUDGET:.6f} B"
              f"   rel-rms vs the NumPy mirror {rel:.3e}")
    if len(rows) == 2:
        print(f"  f32 / f64 cost ratio  {rows[1]['flops'] / rows[0]['flops']:.5f}"
              "   (0.50002 is the reviewed grader-spelling figure)")
    if a.out:
        _dump(a.out, {"check": "WP14 C0-ii: fnp K2 backbone parity and price",
                      "contract": "LOG 2026-09-02 CONTRACT WP14, HEAD d915f83",
                      "width": a.width, "depth": a.depth, "seed": a.seed,
                      "rows": rows,
                      "reference": "local_sim/p2_cal_lab.closure (NumPy mirror)"})
    return 0


def cmd_price(a) -> int:
    """Where the composed projection stands, printed with its own caveat."""
    total_f32 = K2_ANCHOR_F32 + K3_INCREMENT_F32
    total_f64 = K2_ANCHOR_F64 + K3_INCREMENT_F64
    floor = int(0.1 * BUDGET)
    print(f"  budget 2**41                              {BUDGET:>18,}")
    print(f"  the 0.1 multiplier floor                  {floor:>18,}")
    print()
    for nm, k2, k3 in (("all f64 (WP13's record)", K2_ANCHOR_F64, K3_INCREMENT_F64),
                       ("both f32 (C0-iii)", K2_ANCHOR_F32, K3_INCREMENT_F32)):
        t = k2 + k3
        print(f"  {nm:<26} K2 {k2:>15,}  K3 {k3:>15,}")
        print(f"  {'':<26} total {t:>13,}  = {t / BUDGET:.5f} B"
              f"   multiplier {max(0.1, t / BUDGET):.5f}")
    head = floor - total_f32
    print()
    print(f"  headroom under the floor at f32           {head:>18,}  "
          f"= {head / BUDGET:.5f} B")
    # 16 layers, not 15. SKEPTIC WP14 caught this dividing by 15 and reporting
    # 1.61 matmuls/layer of headroom where the figure is 1.51.
    print(f"  in f32 dense 1024-matmuls                 {head / 2_146_435_072:>18.2f}"
          f"   ({head / 2_146_435_072 / 16:.2f} per layer, over 16 layers)")
    print()
    print("  CAVEAT, and it is the reason this module exists: the two halves are")
    print("  measured in DIFFERENT spellings and their sum has never been executed")
    print("  as one program. It implies nothing about whest validate, the 4 GiB")
    print(f"  single-array rule, the 8 GiB cap or the 0.4 s residual cliff.")
    return 0


def _dump(path, doc):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print(f"  wrote {os.path.relpath(path, _ROOT)}")


# ==========================================================================
# WHAT IS NOT DONE -- by name, so the next session does not have to guess
# ==========================================================================

NOT_DONE = """
DONE 2026-09-02 (SKEPTIC WP14's named follow-up): items 1, 2, 3, 4 and 5 below
are implemented in `k3_layer` and priced by `--layer-price`; `--layer-k2` checks
the layer in the `kappa_3 = kappa_4 = 0` limit against the K2 closure and against
Gauss-Legendre quadrature. What remains is item 6 (full parity against upstream
at nonzero `kappa_3`, which needs torch) and item 7 (the shipped file).

1. `pK_slices` for the 25 kept terms. Needs `eval_part`, `multiply_wicks` and
   `check_vec_partition` in fnp. The int_parts are (1,), (2,), (3,), (4,),
   (1,1), (2,1), (2,2) -- all of them slices of degree <= 2 in the carried
   state, so none is n^3.
2. The `(1,1,1)` block, which IS the n^3 work and IS the regenerating flow:
   `pK_111 = wick(1) * WK[3]` plus ONE two-leg block with
   `fac1 = w(1)[:,None] * WK_11`, `fac2 = 3 I`,
   `fac3 = (w(2)[:,None] * w(1)[None,:] * WK_11).T`  (factor_k3.py:378-403).
   NOTE the asymmetry between `w(1)` and `w(2)*w(1)` -- p2_slice_cost.py uses
   `w1` for both and is therefore numerically wrong while being cost-correct.
3. `DS_pK_to_K` and the harmonic projection `proj_geq_r` at d = 3, 4. These
   carry n-dependent coefficients (`2/(2n+8)` and friends) that must be
   transcribed from `harmonic.py`, not re-derived.
4. `from_dstensor`'s regeneration in fnp: `M = diag(S3) + 3 S21^T`, plus the
   diagonal-zeroing convention. Already written correctly in
   `p2_slice_cost._rebuild_M`.
5. The wick coefficients `WICK_COEF_D['relu']` as closed forms in `alpha`, from
   `wick.py`. `k2_closure` above already returns `alphas` and `sigs` for this.
6. Parity: the whole flow against `p2_range_lab.means_of(hook_nl=regen)` on the
   same networks, to f32 rounding, at 128x4 and then at 1024x16.
7. Only then: `whest validate`, `certify_candidate`, a brokered residual soak,
   and the array-op count against the 0.4 s cliff.

Steps 1-5 are transcription against a frozen, checked table. Step 6 is the one
that can fail for an interesting reason.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-terms", action="store_true")
    ap.add_argument("--k2-parity", action="store_true")
    ap.add_argument("--price", action="store_true")
    ap.add_argument("--layer-price", action="store_true")
    ap.add_argument("--layer-k2", action="store_true")
    ap.add_argument("--mutation-coverage", action="store_true")
    ap.add_argument("--flow-price", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--todo", action="store_true")
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--seed", type=int, default=1_000_921)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    print("=" * 78)
    print("WP14 / C0-ii  --  the fnp port of the regenerating K3 flow (INCOMPLETE)")
    print("=" * 78)
    rc = 0
    if a.check_terms:
        print("\n-- the frozen term table -------------------------------------")
        rc |= cmd_check_terms(a)
    if a.k2_parity:
        print(f"\n-- K2 backbone parity and price, {a.width} x {a.depth} ------------")
        rc |= cmd_k2_parity(a)
    if a.mutation_coverage:
        print("\n-- what do the closed-form checks actually PIN? ---------------")
        rc |= cmd_mutation_coverage(a)
    if a.layer_k2:
        print("\n-- the K3 layer must collapse to the K2 closure --------------")
        rc |= cmd_layer_k2_degeneracy(a)
    if a.flow_price:
        print("\n-- the WHOLE flow: FLOPs, OPERATIONS and residual -------------")
        rc |= cmd_flow_price(a)
    if a.layer_price:
        print("\n-- ONE metered K3 layer, both dtypes -------------------------")
        rc |= cmd_layer_price(a)
    if a.price:
        print("\n-- where the composed projection stands ----------------------")
        rc |= cmd_price(a)
    if a.todo or not any((a.check_terms, a.k2_parity, a.price,
                          a.layer_price, a.layer_k2,
                          a.mutation_coverage, a.flow_price)):
        print("\n-- NOT DONE, by name -----------------------------------------")
        print(NOT_DONE)
    print("=" * 78)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
