"""Generate the Stage 2 experiment-1 arms: a coherent fourth-order core.

Contract: `experiments/LOG.md`, 2026-09-10, *CONTRACT (Rule 1): Stage 2
experiment 1 — a coherent fourth-order core*, committed at `08f787b` BEFORE
any data existed.

WHY A GENERATOR AND NOT FOUR HAND-EDITED FILES
-----------------------------------------------
`local_sim/p2_build_arms.py` set the precedent and the reason is the same: four
copies of a 1,100-line chain drift, and a drifted control is worse than no
control. The math lives here once; the arms are emitted from
`estimator_k3_full.py` plus a header that sets three constants.

WHAT THE INTERVENTION ACTUALLY IS
----------------------------------
`estimator_k3_full.py` ALREADY carries a fourth-order state. It is `c4`, and it
is **one float for all 1,024 neurons**:

    D4  = (c4 * metric^2) * ones_n          -> S[(4,)]     a CONSTANT vector
    D22 = (c4 * metric^2 / 3) * off_mask    -> S[(2,2)]    a CONSTANT matrix
    c4_out = proj * (a * K[(4,)].sum() + b * K[(2,2)].sum())

So the chain projects the whole fourth cumulant onto one isotropic number, and
the off-diagonal is pinned at exactly one third of the diagonal. The
intervention is to carry the **per-neuron deviation** from that number as well:

    gdev            an (n,) vector with mean zero, carried between layers
    D4   += theta * gdev
    D22  += theta * lam * (gdev[:,None] + gdev[None,:]) * 0.5 * off_mask
    gdev_out = raw4 - mean(raw4),   scaled exactly as c4_out is

**`theta = 0` is BIT-IDENTICAL to the shipped chain.** That is the whole design:
the control is nested exactly rather than approximately, so a difference cannot
come from a re-spelling, and `theta` is a single scalar whose zero is the null
hypothesis. It also gives a gate-strength positive control for free -- the
repair the T2 Skeptic demanded on 2026-09-10.

TRANSPORT IS THE STAGE-1 IDENTITY, AND IT IS FREE HERE
-------------------------------------------------------
The carried deviation is a fourth-order core diagonal, so it transports as

    g_pre = (W^T . W^T) g + lam [ diag(W^T C W) - (W^T . W^T) diag(C) ]

which is `local_sim/p2_compact_reference.py`'s P3, verified exact at `3.3e-16`.
And `diag(W^T C W)` is `fnp.diagonal(cov_pre)`, which `_layer` computes on the
line above as `var`. So the whole channel is one Hadamard and two matrix-vector
products against an `O(n^3)` chain -- Stage 1 projected `0.000038 B`, and this
arm is where that projection becomes a metered measurement.

THE OFF-DIAGONAL FORM, AND WHAT IS NOT TESTED
----------------------------------------------
`(gdev_i + gdev_j) / 2` is used for the core's off-diagonal because it is
sign-safe (a fourth cumulant is not non-negative, so a geometric mean is not
available) and because at constant `gdev` it reduces to `lam * gdev`, matching
the shipped `1/3` at `lam = 1/3`. The **correlation-coupled** form
`lam * gbar_ij * rho_ij` is a DIFFERENT core and is not tested here. Saying so
is a scope statement, not a result about it.

THE ARMS
--------
    arm_k3_exact      theta = 0            the control, bit-identical
    arm_k3_k4coh      theta = THETA        consumed AND carried forward
    arm_k3_k4nobirth  theta = THETA, no carry: the deviation is consumed at
                      the layer it is born and never propagated. Separates
                      "per-neuron K4 here" from "per-neuron K4 HISTORY", which
                      is the coherence claim's load-bearing half.

Run:
    uv run python local_sim/p2_build_k4_arms.py --emit
    uv run python local_sim/p2_build_k4_arms.py --parity     # theta=0 == control
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
ARMS_DIR = os.path.join(_HERE, "p2_arms")
SOURCE = os.path.join(_ROOT, "estimator_k3_full.py")

#: The intervention strength and the off-diagonal coupling. FITTED ON P07 ONLY
#: (DESIGN, reusable), frozen before P01 is read. These are placeholders until
#: the fit runs; `--fit` writes them back and records the artifact it fitted on.
DEFAULT_THETA = 1.0
DEFAULT_LAM = 1.0 / 3.0


# --------------------------------------------------------------------------- #
# The patch. Each entry is (exact source text, replacement, why).
# A missing anchor is a hard failure: a generator that silently skips a patch
# emits a control wearing the intervention's name.
# --------------------------------------------------------------------------- #

def _patches(theta: float, lam: float, carry: bool):
    """Each entry is `(anchor, replacement, why, expected_count)`.

    `expected_count` is asserted, not merely used: `_tower` and `_regen` hold
    the SAME initial-state text, so a patch that quietly hit one of them would
    leave the regenerating fallback threading a four-tuple into a `_layer` that
    returns five. That fallback only fires at shapes this file was not designed
    for, which is exactly where nobody would notice.
    """
    P = []

    # 1. the state tuple gains the carried deviation
    P.append((
        """                 None, 0.0)
        mus = []""",
        """                 None, 0.0,
                 # THE CARRIED PER-NEURON FOURTH-ORDER DEVIATION. Zero at the
                 # input, where the state is exactly Gaussian, so the first
                 # layer is unaffected by construction.
                 fnp.zeros(width, dtype=dtype))
        mus = []""",
        "the initial state carries gdev, in BOTH towers", 2))

    P.append((
        """            mu = fnp.where(fnp.isfinite(mu), mu, dtype(0.0))
            state = (mu, state[1], state[2], state[3])""",
        """            mu = fnp.where(fnp.isfinite(mu), mu, dtype(0.0))
            state = (mu, state[1], state[2], state[3], state[4])""",
        "the invariant ladder threads gdev, in BOTH towers", 2))

    # 2. unpack it
    P.append((
        """    mu, cov, tw, c4 = state""",
        """    mu, cov, tw, c4, gdev = state""",
        "_layer unpacks the fifth element", 1))

    # 3. transport it -- Stage 1's P3, with `var` already in hand
    P.append((
        """    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)
    D4 = (c4 * metric * metric) * ones_n""",
        """    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)

    # ---- THE FOURTH-ORDER DEVIATION, TRANSPORTED --------------------------
    # Stage 1's P3, exact for the core `diag(g) + lam C_off`:
    #
    #   diag(W^T G W) = (W^T . W^T) g + lam [ diag(W^T C W) - (W^T.W^T) diag(C) ]
    #
    # `diag(W^T C W)` IS `var`, computed four lines above as the diagonal of
    # `cov_pre`. That is why this channel is O(n^2) beside an O(n^3) chain:
    # the only term that would have cost a matmul is already on the table.
    #
    # The result is re-centred to mean zero every layer, because the isotropic
    # part is `c4`'s job and carrying it twice would double-count it.
    # `_K4_THETA` is a MODULE CONSTANT, so this branch is resolved once per
    # arm and is not a data-dependent control flow. It matters: without it the
    # theta=0 control PAYS for a channel whose result it multiplies by zero,
    # the two arms come out at identical cost, and the score comparison --
    # which is mse x multiplier -- silently stops seeing the intervention's
    # price. A control that is charged for the intervention is not a control.
    if _K4_THETA:
        WT2 = WT * WT
        gdev_pre = WT2 @ gdev + _K4_LAM * (var - WT2 @ fnp.diagonal(cov))
        gdev_pre = gdev_pre - fnp.mean(gdev_pre)
        gdev_pre = fnp.where(fnp.isfinite(gdev_pre), gdev_pre, dtype(0.0))
        D4 = (c4 * metric * metric) * ones_n + _K4_THETA * gdev_pre
    else:
        gdev_pre = gdev
        D4 = (c4 * metric * metric) * ones_n""",
        "P3 transport, re-centred, NaN-guarded", 1))

    # 4. the off-diagonal consumer
    P.append((
        """        D22 = (c4 * metric * metric / 3.0) * off_mask""",
        """        # The core's off-diagonal. `(g_i + g_j)/2` is sign-safe -- a fourth
        # cumulant is not non-negative, so a geometric mean is unavailable --
        # and at constant `gdev` it reduces to `lam * gdev`, which is the
        # shipped `1/3` at `lam = 1/3`. The correlation-coupled form is a
        # DIFFERENT core and is not tested by this arm.
        if _K4_THETA:
            D22 = ((c4 * metric * metric / 3.0)
                   + _K4_THETA * _K4_LAM * 0.5
                   * (gdev_pre[:, None] + gdev_pre[None, :])) * off_mask
        else:
            D22 = (c4 * metric * metric / 3.0) * off_mask""",
        "the off-diagonal consumer", 1))

    # 5. the last-layer S dict has no D22; it still needs the state threaded
    P.append((
        """    if last_layer:
        return (mu_out, None, None, 0.0)""",
        """    if last_layer:
        return (mu_out, None, None, 0.0, gdev_pre)""",
        "the last layer returns a five-tuple too", 1))

    # 6. the BIRTH: emit the per-neuron deviation instead of discarding it
    carry_expr = ("_gd_out" if carry else
                  "fnp.zeros(n, dtype=dtype)  # NO-BIRTH ARM: consumed here, "
                  "never carried")
    P.append((
        """    lap = table["lap2_d4"]
    c4_out = _proj_coef_d4_r2(n) * (
        float(lap["(4,)"]["terms"][0]["coef"])
        * float(lap["(4,)"]["int_partition_coef"]) * K[(4,)].sum()
        + float(lap["(2, 2)"]["terms"][0]["coef"])
        * float(lap["(2, 2)"]["int_partition_coef"]) * K[(2, 2)].sum())
    return (mu_out, cov_out, out_tw, c4_out)""",
        """    lap = table["lap2_d4"]
    _a = (float(lap["(4,)"]["terms"][0]["coef"])
          * float(lap["(4,)"]["int_partition_coef"]))
    _b = (float(lap["(2, 2)"]["terms"][0]["coef"])
          * float(lap["(2, 2)"]["int_partition_coef"]))
    _proj = _proj_coef_d4_r2(n)
    # UNCHANGED: the isotropic scalar the shipped chain carries. Its two `.sum()`
    # calls are the same reductions, in the same order, on the same arrays.
    c4_out = _proj * (_a * K[(4,)].sum() + _b * K[(2, 2)].sum())
    # NEW: the per-neuron deviation the shipped chain throws away. `K[(2,2)]`
    # is summed along its second axis so both contributions are (n,) vectors in
    # the same units the scalar carries, and the mean is removed so `c4_out`
    # and `_gd_out` partition the same quantity rather than overlapping.
    if _K4_THETA:
        _raw4 = _proj * (_a * K[(4,)] + _b * K[(2, 2)].sum(axis=1))
        _gd_out = _raw4 - fnp.mean(_raw4)
    else:
        _gd_out = gdev_pre
    return (mu_out, cov_out, out_tw, c4_out, %s)""" % carry_expr,
        "the birth: per-neuron deviation, mean removed", 1))

    return P


_HEADER = '''"""Stage 2 experiment 1 arm: {name}.

{blurb}

GENERATED by `local_sim/p2_build_k4_arms.py` from `estimator_k3_full.py`
sha256 `{src_sha}`. **Do not edit this file** -- edit the generator, which holds
the patch once so the arms cannot drift apart.

Contract: `experiments/LOG.md`, 2026-09-10, *CONTRACT (Rule 1): Stage 2
experiment 1*, committed at `08f787b` before any data existed.

theta = {theta!r}   lam = {lam!r}   carry = {carry!r}

`theta = 0` reproduces `estimator_k3_full.py` BIT-IDENTICALLY, which is what
makes this a nested control rather than a re-spelling.

Grader-legal by inheritance: imports only flopscope, flopscope.numpy and
whestbench; never mutates a grader array; shape-generic; deterministic given
the network. **This is a research arm, not a submission candidate.**
"""

'''


def build(name: str, theta: float, lam: float, carry: bool, blurb: str,
          src: str, src_sha: str) -> str:
    body = src
    for old, new, why, count in _patches(theta, lam, carry):
        found = body.count(old)
        if found != count:
            raise SystemExit(
                f"PATCH ANCHOR COUNT WRONG for arm {name!r} ({why}): expected "
                f"{count}, found {found}.\n"
                "`estimator_k3_full.py` has moved under this generator. A "
                "generator that silently skips or over-applies a patch emits a "
                "CONTROL wearing the INTERVENTION's name, so this is fatal "
                "rather than a warning.\nAnchor:\n" + old[:300])
        body = body.replace(old, new, count)

    # Strip the source's own module docstring and prepend ours.
    i = body.find('"""')
    j = body.find('"""', i + 3)
    body = body[j + 3:].lstrip("\n")
    head = _HEADER.format(name=name, blurb=blurb, src_sha=src_sha,
                          theta=theta, lam=lam, carry=carry)
    consts = (f"_K4_THETA = {theta!r}\n"
              f"_K4_LAM = {lam!r}\n"
              f"_K4_CARRY = {carry!r}\n\n")
    # The constants must land AFTER the imports, or the patched `_layer` cannot
    # see them. The source's imports end at its first blank line following the
    # last `from`/`import` at column zero.
    lines = body.splitlines(keepends=True)
    last_import = 0
    for k, ln in enumerate(lines):
        if ln.startswith(("import ", "from ")):
            last_import = k
    lines.insert(last_import + 1, "\n" + consts)
    return head + "".join(lines)


ARMS = [
    ("k3_exact", 0.0, DEFAULT_LAM, True,
     "THE CONTROL. theta = 0, so this is `estimator_k3_full.py` bit for bit, "
     "reached through the generator so the control and the intervention cannot "
     "drift apart."),
    ("k3_k4coh", None, None, True,
     "THE INTERVENTION. The per-neuron fourth-order deviation is consumed by "
     "both `S[(4,)]` and `S[(2,2)]` AND carried forward, so its consumers and "
     "the third-order births it induces come from one state."),
    ("k3_k4nobirth", None, None, False,
     "HALF-ARM CONTROL. The same deviation is consumed at the layer it is born "
     "and never carried. If this matches the intervention, the coherence claim "
     "is about a single layer and not about history."),
]


def _src():
    with open(SOURCE, "rb") as fh:
        raw = fh.read()
    return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()


def emit(theta: float, lam: float) -> int:
    src, sha = _src()
    os.makedirs(ARMS_DIR, exist_ok=True)
    print(f"source  estimator_k3_full.py  sha256 {sha[:16]}...")
    for name, th, lm, carry, blurb in ARMS:
        th = theta if th is None else th
        lm = lam if lm is None else lm
        text = build(name, th, lm, carry, blurb, src, sha)
        path = os.path.join(ARMS_DIR, f"arm_{name}.py")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print(f"  wrote arm_{name}.py   theta={th!r} lam={lm!r} carry={carry!r}"
              f"   {len(text):,} bytes")
    return 0


def parity() -> int:
    """`theta = 0` must be the shipped chain, not merely close to it."""
    import numpy as np

    sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
    for p in (_HERE, _ROOT):
        if p not in sys.path:
            sys.path.append(p)
    import flopscope as flops  # noqa: E402
    import harness  # noqa: E402
    import lifecycle_eval as le  # noqa: E402
    from mint_provenance import canonical_mint  # noqa: E402

    flops.configure(symmetry_warnings=False)
    width, depth, budget = 96, 5, 2 ** 41
    ok = True
    ref_mod, _ = le.load_candidate(SOURCE)
    arm_mod, _ = le.load_candidate(os.path.join(ARMS_DIR, "arm_k3_exact.py"))
    coh_mod, _ = le.load_candidate(os.path.join(ARMS_DIR, "arm_k3_k4coh.py"))
    for est in (ref_mod, arm_mod, coh_mod):
        est.Estimator().setup(le.SetupContext(
            le.LifecycleSpec.for_round("v2-phase2", seed=0)))

    print(f"parity at {width}x{depth}")
    for seed in (90000000301, 90000000302, 90000000303):
        w, es, _ = canonical_mint(width, depth, seed)
        outs, costs = [], []
        for mod in (ref_mod, arm_mod, coh_mod):
            mlp = harness.MLP(weights=[np.asarray(x) for x in w], width=width,
                              depth=depth, seed=es)
            ctx = flops.BudgetContext(flop_budget=budget, quiet=True)
            with ctx:
                p = mod.Estimator().predict(mlp, budget)
            outs.append(np.asarray(p, dtype=np.float64))
            costs.append(float(ctx.flops_used))
        same = np.array_equal(outs[0], outs[1])
        cost_same = costs[0] == costs[1]
        d = float(np.max(np.abs(outs[0] - outs[2])))
        s = max(float(np.max(np.abs(outs[0]))), 1e-300)
        print(f"  seed {seed}: theta=0 bit-identical: {same}   "
              f"same counted FLOPs: {cost_same} ({costs[0]:,.0f} vs "
              f"{costs[1]:,.0f})")
        print(f"               intervention differs by {d / s:.3e} relative"
              f"   cost {costs[2]:,.0f} ({costs[2] / costs[0]:.6f}x)")
        if not same:
            ok = False
            print("               *** theta=0 IS NOT THE CONTROL ***")
        if d / s < 1e-12:
            ok = False
            print("               *** the intervention changed NOTHING -- a "
                  "no-op arm cannot test a hypothesis ***")
    print("\nPARITY OK" if ok else "\nPARITY FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--theta", type=float, default=DEFAULT_THETA)
    ap.add_argument("--lam", type=float, default=DEFAULT_LAM)
    a = ap.parse_args(argv)
    if a.parity:
        return parity()
    if a.emit:
        return emit(a.theta, a.lam)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
