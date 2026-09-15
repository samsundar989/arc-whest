"""Generate `estimator_k3_v30.py` from F21 — P1, the per-neuron κ4 core.

Slice 6a step 2, `docs/CLAUDE_CODE_HANDOFF_2026-09-14.md` §2.2. Our own code,
F21 lineage, one mechanism per commit.

WHY A GENERATOR. F21 is 1,127 lines. A hand-edited copy drifts from its
control, and a drifted control is worse than none. The mechanism lives here as
a patch set with asserted anchor counts; a missing or over-applied anchor is
fatal, because a generator that silently skips one emits a control wearing the
candidate's name.

THE MECHANISM, AND WHY IT IS A GENERALISATION RATHER THAN AN ADDITION
---------------------------------------------------------------------
F21 already carries fourth-order state. It is `c4`, and it is **one float for
all 1,024 neurons**:

    D4  = (c4 * metric^2) * ones_n            -> S[(4,)]
    D22 = (c4 * metric^2 / 3) * off_mask      -> S[(2,2)]
    c4_out = proj * (a * K[(4,)].sum() + b * K[(2,2)].sum())

The handoff's per-neuron forms are

    q4  = m * dG
    q22 = (m / 6) * (dG_i + dG_j)

and at a CONSTANT `dG = c4 * metric` those are `c4 * metric^2` and
`(m/6)(2 c4 m) = c4 * metric^2 / 3` — **exactly F21's two lines**. So the
per-neuron core is not a new term bolted on; it is the same term with the
isotropic projection removed. That is what makes an exact nested control
possible, and it is why `_K4_PERNEURON = False` reproduces F21 bit for bit.

TRANSPORT. The handoff's law, which is the identity verified exact to 3.3e-16
in `local_sim/p2_compact_reference.py::t_k4_identity`:

    dG = (W^T . W^T) @ (g_prev - lam * diag C_out) + lam * diag C_pre

`diag C_pre` is `var`, which `_layer` computes two lines above, so the whole
channel is one Hadamard and two matrix-vector products against an O(n^3) chain.

THE BIRTH PROJECTION, AND THE ORACLE THAT SETTLED IT
----------------------------------------------------
The per-neuron core is the **r = 1 matrix-core** harmonic projection of the
degree-4 tensor:

    g = (K4 + rowsum K22) * 6/(n+4) - (sum K4 + sum rowsum K22) * 3/((n+2)(n+4))

F21's `_proj_coef_d4_r2(n)` is the **r = 2** projection -- a scalar, the fully
contracted isotropic part. They are DIFFERENT PROJECTIONS of the same tensor,
and the first three versions of this generator tried to obtain one by rescaling
the other. That produced three normalisations, three residuals and a design
screen 9.2x worse than the control. The trace subtraction is what makes the
projection harmonic, and omitting it is what let the core compound with depth
(measured: ratio to F21's scalar 1.000 at layer 0, 6.4x by layer 4).

Settled against the 504aldo parity oracle, `external/504aldo` commit
`18c17e2d`, `estimators/estimator_v17.py:679`, whose own comment names it the
"r=1 matrix-core harmonic projection". The oracle also confirms the consumers
this file already had -- `D4 = metric * dG` and `D22 = (metric/6)(dG_i + dG_j)`
with `METRIC_C = 2.0` -- and the transport operand orientation.

LAMBDA IS A FREE PARAMETER, NOT A FITTED CONSTANT
--------------------------------------------------
`_K4_LAM` is a per-layer table. `experiments/2026-09-14_slice6a_lambda/`
measured its SHAPE and **retracted** any claim about its value at width 1024:
the leave-one-out window spread is 8.89x in the median, so the extrapolated
value is set by the fit window rather than by the data. What survives is that
the mechanism is real — the coefficient explains 62-84% of the off-diagonal
core's variance at every width we can reach, rising with depth.

So the table here is an **initialiser**, to be fitted end to end against score
on fresh design networks. It is never quoted as a measured constant, and
504aldo's published table is never imported: it was fitted on eight public
MLPs, and our policy fits on fresh mints only.

WHAT THIS COMMIT DOES NOT DO
-----------------------------
The κ4→κ3 feed (`X3 = A·d(lam w1) + P·d(w1^2 dG)`) and the `q31` consumer are
**not** here. `q31` needs a `(3,1)` entry the frozen Wick table does not carry,
which is a change to the table and belongs in its own commit with its own
parity evidence. One mechanism per commit, as the contract says.

Run:
    uv run python local_sim/p2_build_v30.py --emit
    uv run python local_sim/p2_build_v30.py --parity
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
SOURCE = os.path.join(_ROOT, "estimator_k3_strassen_L4_f21.py")
TARGET = os.path.join(_ROOT, "estimator_k3_v30.py")
CONTROL = os.path.join(_ROOT, "estimator_k3_v30_control.py")
LAM_NPZ = os.path.join(_ROOT, "experiments", "2026-09-14_slice6a_lambda",
                       "lam_own.npz")


def lam_table(depth=16):
    """The per-layer λ initialiser, as a plain Python list.

    Read from OUR measurement, never from 504aldo's published table. Falls back
    to a flat 0.0 — which disables the channel — rather than to a guess.
    """
    try:
        import numpy as np
        z = np.load(LAM_NPZ)
        v = [float(x) for x in z["lam_target"]]
    except Exception:
        return [0.0] * depth
    # `lam_target` has depth-1 entries (λ_l is read at pre{l+1}); pad the last
    # layer, which never consumes one, with a zero rather than a repeat.
    return v + [0.0] * max(0, depth - len(v))


def _patches(per_neuron: bool):
    """`(anchor, replacement, why, expected_count)`; count is ASSERTED."""
    P = []

    # 1. the carried state gains the per-neuron core diagonal
    P.append((
        """                 None, 0.0)
        mus = []""",
        """                 None, 0.0,
                 # THE PER-NEURON FOURTH-ORDER CORE DIAGONAL. Zero at the
                 # input, where the state is exactly Gaussian, so layer 0 is
                 # unaffected by construction and the control is nested there
                 # too.
                 fnp.zeros(width, dtype=dtype))
        mus = []""",
        "the initial state carries g, in BOTH towers", 2))

    P.append((
        """            mu = fnp.where(fnp.isfinite(mu), mu, dtype(0.0))
            state = (mu, state[1], state[2], state[3])""",
        """            mu = fnp.where(fnp.isfinite(mu), mu, dtype(0.0))
            state = (mu, state[1], state[2], state[3], state[4])""",
        "the invariant ladder threads g, in BOTH towers", 2))

    P.append((
        """    mu, cov, tw, c4 = state""",
        """    mu, cov, tw, c4, gk4 = state""",
        "_layer unpacks the fifth element", 1))

    # 2. `_layer` needs to know WHICH layer it is, to index the λ table
    P.append((
        """def _layer(W, state, table, dtype, const, init_scale, last_layer, gens):""",
        """def _layer(W, state, table, dtype, const, init_scale, last_layer, gens,
           layer_index=0):""",
        "_layer takes its own index, for the per-layer lambda", 1))
    P.append((
        """            state = _layer(w[l], state, table, dtype, const, 2.0,
                           last_layer=(l == depth - 1), gens=gens)""",
        """            state = _layer(w[l], state, table, dtype, const, 2.0,
                           last_layer=(l == depth - 1), gens=gens,
                           layer_index=l)""",
        "the exact tower passes its index", 1))
    P.append((
        """            state = _layer(w[l], state, table, dtype, const, 2.0,
                           last_layer=(l == depth - 1), gens=1)""",
        """            state = _layer(w[l], state, table, dtype, const, 2.0,
                           last_layer=(l == depth - 1), gens=1,
                           layer_index=l)""",
        "the regenerating fallback passes its index", 1))

    # 3. TRANSPORT, and the two consumers
    if per_neuron:
        d4 = '''    # ---- THE PER-NEURON FOURTH-ORDER CORE ---------------------------------
    # Transport, exactly the identity verified in p2_compact_reference:
    #
    #   dG = (W^T . W^T) @ (g_prev - lam * diag C_out) + lam * diag C_pre
    #
    # `diag C_pre` IS `var`, computed above, so the only term that would have
    # cost a matmul is already on the table. `lam` is indexed by the PREVIOUS
    # layer: 504aldo's table models G_pre_off(l+1) with LAM[l], so the value
    # consumed here at layer `l` is LAM[l-1].
    #
    # A per-layer Python scalar and a module-level table make this branch
    # shape-generic and data-independent: no control flow depends on a value.
    # The array goes on the LEFT of every scalar product. With the scalar
    # first, numpy's __mul__ wins the dispatch and flopscope auto-routes
    # with a warning -- meaning the operation was nearly billed by the
    # wrong meter. Explicit is the whole point of this file.
    _lam_prev = dtype(_K4_LAM[layer_index - 1] if layer_index > 0 else 0.0)
    _WT2 = WT * WT
    dG = _WT2 @ (gk4 - fnp.diagonal(cov) * _lam_prev) + var * _lam_prev
    dG = fnp.where(fnp.isfinite(dG), dG, dtype(0.0))

    # The consumers. At a CONSTANT dG these are F21's own two lines exactly --
    # q4 = m*dG is c4*metric^2, and q22 = (m/6)(dG_i+dG_j) is c4*metric^2/3 --
    # which is why this is a generalisation and not an addition.
    D4 = metric * dG'''
        d22 = '''        D22 = ((metric / 6.0) * (dG[:, None] + dG[None, :])) * off_mask'''
    else:
        d4 = '''    # CONTROL ARM: the isotropic scalar path, untouched. `dG` is carried only
    # so the state tuple has a fifth slot; nothing reads it.
    dG = gk4
    D4 = (c4 * metric * metric) * ones_n'''
        d22 = '''        D22 = (c4 * metric * metric / 3.0) * off_mask'''

    P.append((
        """    D4 = (c4 * metric * metric) * ones_n""",
        d4, "transport and the q4 consumer", 1))
    P.append((
        """        D22 = (c4 * metric * metric / 3.0) * off_mask""",
        d22, "the q22 consumer", 1))

    # 4. the last layer returns a five-tuple too
    P.append((
        """    if last_layer:
        return (mu_out, None, None, 0.0)""",
        """    if last_layer:
        return (mu_out, None, None, 0.0, dG)""",
        "the last layer returns a five-tuple", 1))

    # 5. the BIRTH
    if per_neuron:
        birth = '''    lap = table["lap2_d4"]
    _a4 = (float(lap["(4,)"]["terms"][0]["coef"])
           * float(lap["(4,)"]["int_partition_coef"]))
    _b22 = (float(lap["(2, 2)"]["terms"][0]["coef"])
            * float(lap["(2, 2)"]["int_partition_coef"]))
    _proj = _proj_coef_d4_r2(n)
    # UNCHANGED: the isotropic scalar F21 carries, same reductions in the same
    # order, so the control's arithmetic is bit-identical when it is read.
    c4_out = _proj * (_a4 * K[(4,)].sum() + _b22 * K[(2, 2)].sum())
    # NEW: the per-neuron core F21 projects away. `6/(n+4) (K4 + rowsum K22)`
    # is `_proj` applied to the same two objects WITHOUT the trailing sum.
    # THE BIRTH IS A DIFFERENT HARMONIC PROJECTION, not a rescaling of F21's.
    #
    # This is what the first three attempts got wrong, and the error was
    # conceptual rather than arithmetic. `_proj_coef_d4_r2(n)` is
    # `harmonic.proj_coef(n, 4, 2)` -- the **r = 2** projection, which is a
    # SCALAR: the fully contracted isotropic part of the degree-4 tensor. The
    # per-neuron core is the **r = 1** matrix-core projection, a different
    # object with its own coefficients, and no amount of rescaling the r=2
    # scalar produces it. Trying to derive one from the other gave three
    # normalisations and three residuals (5e-3, 1.2e-3, 3e-2) and a design
    # screen 9.2x worse than the control.
    #
    #     g = (K4 + rowsum K22) * 6/((n+4))
    #         - (sum K4 + sum rowsum K22) * 3/((n+2)(n+4))
    #
    # The second term is the TRACE SUBTRACTION that makes it a harmonic
    # (traceless) projection, and omitting it is what let the core grow with
    # depth. Confirmed against the 504aldo parity oracle
    # (`external/504aldo`, commit 18c17e2d, `estimators/estimator_v17.py:679`),
    # whose own comment names it "r=1 matrix-core harmonic projection".
    #
    # F21's isotropic scalar `c4_out` is UNCHANGED above and still carried:
    # the two projections coexist, exactly as they do in the oracle.
    _cA = 6.0 / (n + 4.0)
    _cI = -3.0 / ((n + 2.0) * (n + 4.0))
    _k22row = K[(2, 2)].sum(axis=1)
    _k4 = K[(4,)]
    g_out = (_k4 + _k22row) * _cA + (_k4.sum() + _k22row.sum()) * _cI
    g_out = fnp.where(fnp.isfinite(g_out), g_out, dtype(0.0))
    return (mu_out, cov_out, out_tw, c4_out, g_out)'''
    else:
        birth = '''    lap = table["lap2_d4"]
    c4_out = _proj_coef_d4_r2(n) * (
        float(lap["(4,)"]["terms"][0]["coef"])
        * float(lap["(4,)"]["int_partition_coef"]) * K[(4,)].sum()
        + float(lap["(2, 2)"]["terms"][0]["coef"])
        * float(lap["(2, 2)"]["int_partition_coef"]) * K[(2, 2)].sum())
    return (mu_out, cov_out, out_tw, c4_out, gk4)'''
    P.append((
        """    lap = table["lap2_d4"]
    c4_out = _proj_coef_d4_r2(n) * (
        float(lap["(4,)"]["terms"][0]["coef"])
        * float(lap["(4,)"]["int_partition_coef"]) * K[(4,)].sum()
        + float(lap["(2, 2)"]["terms"][0]["coef"])
        * float(lap["(2, 2)"]["int_partition_coef"]) * K[(2, 2)].sum())
    return (mu_out, cov_out, out_tw, c4_out)""",
        birth, "the birth: per-neuron core, or the scalar control", 1))
    return P


_HEADER = '''"""%(title)s

%(blurb)s

GENERATED by `local_sim/p2_build_v30.py` from `estimator_k3_strassen_L4_f21.py`
sha256 `%(sha)s`.
**Do not edit this file** -- edit the generator, which holds the mechanism once
so the candidate and its control cannot drift apart.

Contract: `docs/CLAUDE_CODE_HANDOFF_2026-09-14.md` Slice 6a step 2 (P1).
Provenance and attribution: `docs/PROVENANCE_504aldo.md`.

`_K4_LAM` is an INITIALISER measured on our own fresh design networks, not a
fitted constant and never 504aldo's published table. Its value at width 1024 is
explicitly undetermined -- see `experiments/2026-09-14_slice6a_lambda/`.

Grader-legal by inheritance: imports only flopscope, flopscope.numpy and
whestbench; never mutates a grader array; shape-generic; deterministic given the
network. **Not a submission candidate.** Samuel approves every submission
individually and designates alone.
"""

'''


def build(per_neuron, title, blurb, src, sha, lam):
    body = src
    for old, new, why, count in _patches(per_neuron):
        found = body.count(old)
        if found != count:
            raise SystemExit(
                "PATCH ANCHOR COUNT WRONG (%s): expected %d, found %d.\n"
                "F21 has moved under this generator. A generator that silently "
                "skips or over-applies a patch emits a CONTROL wearing the "
                "CANDIDATE's name, so this is fatal.\nAnchor:\n%s"
                % (why, count, found, old[:280]))
        body = body.replace(old, new, count)

    i = body.find('"""')
    j = body.find('"""', i + 3)
    body = body[j + 3:].lstrip("\n")
    head = _HEADER % {"title": title, "blurb": blurb, "sha": sha}
    consts = ("_K4_PERNEURON = %r\n"
              "#: Per-layer lambda INITIALISER. Fitted end to end against\n"
              "#: score; never quoted as a measured constant.\n"
              "_K4_LAM = %r\n\n" % (per_neuron, lam))
    lines = body.splitlines(keepends=True)
    last_import = 0
    for k, ln in enumerate(lines):
        if ln.startswith(("import ", "from ")):
            last_import = k
    lines.insert(last_import + 1, "\n" + consts)
    return head + "".join(lines)


def emit(lam=None):
    with open(SOURCE, "rb") as fh:
        raw = fh.read()
    src, sha = raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()
    lam = lam if lam is not None else lam_table()
    print("source  estimator_k3_strassen_L4_f21.py  sha256 %s..." % sha[:16])
    print("lambda  %s" % ", ".join("%.4g" % x for x in lam[:6]) + ", ...")
    for path, pn, title, blurb in (
            (TARGET, True, "v30: F21 plus a PER-NEURON fourth-order core (P1).",
             "F21 carries one kappa-4 float for all n neurons. This carries the\n"
             "core's diagonal per neuron, transported by the exact identity and\n"
             "consumed by the same two terms F21 already has -- which at a\n"
             "constant core reduce to F21's own lines exactly."),
            (CONTROL, False, "v30 CONTROL: F21, reached through the generator.",
             "_K4_PERNEURON = False. Bit-identical to F21 in output and in\n"
             "counted FLOPs; it exists so the candidate and its control cannot\n"
             "drift apart through separate editing."),
    ):
        text = build(pn, title, blurb, src, sha, lam)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print("  wrote %-34s per_neuron=%-5s %,d bytes".replace(",d", "d")
              % (os.path.basename(path), pn, len(text)))
    return 0


def parity():
    """The control must be F21 bit for bit; the candidate must differ."""
    import numpy as np
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]
    import flopscope as flops
    import harness
    import lifecycle_eval as le
    from mint_provenance import canonical_mint

    flops.configure(symmetry_warnings=False)
    width, depth, budget = 96, 6, 2 ** 41
    spec = le.LifecycleSpec.for_round("v2-phase2", seed=0)
    mods = {}
    for name, path in (("F21", SOURCE), ("control", CONTROL),
                       ("v30", TARGET)):
        m, _ = le.load_candidate(path)
        e = m.Estimator()
        e.setup(le.SetupContext(spec))
        mods[name] = m

    print("parity at %dx%d" % (width, depth))
    ok = True
    for seed in (90000000401, 90000000402, 90000000403):
        w, es, _ = canonical_mint(width, depth, seed)
        out, cost = {}, {}
        for name, m in mods.items():
            mlp = harness.MLP(weights=[np.asarray(x) for x in w], width=width,
                              depth=depth, seed=es)
            ctx = flops.BudgetContext(flop_budget=budget, quiet=True)
            with ctx:
                p = m.Estimator().predict(mlp, budget)
            out[name] = np.asarray(p, dtype=np.float64)
            cost[name] = float(ctx.flops_used)
        same = np.array_equal(out["F21"], out["control"])
        same_cost = cost["F21"] == cost["control"]
        s = max(float(np.max(np.abs(out["F21"]))), 1e-300)
        d = float(np.max(np.abs(out["F21"] - out["v30"]))) / s
        dc = cost["v30"] / cost["F21"]
        print("  seed %d" % seed)
        print("    control == F21 bitwise: %-5s   same counted FLOPs: %-5s "
              "(%,d)".replace(",d", "d") % (same, same_cost, cost["F21"]))
        print("    v30 differs by %.3e relative   cost %.6fx" % (d, dc))
        if not same or not same_cost:
            ok = False
            print("    *** THE CONTROL IS NOT F21 ***")
        if d < 1e-12:
            ok = False
            print("    *** v30 CHANGED NOTHING -- a no-op cannot test P1 ***")
    print("\nPARITY OK" if ok else "\nPARITY FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--lam-scale", type=float, default=1.0,
                    help="scale the whole lambda initialiser; the end-to-end "
                         "fit's single knob until a per-layer fit exists")
    a = ap.parse_args(argv)
    if a.parity:
        return parity()
    if a.emit:
        return emit([x * a.lam_scale for x in lam_table()])
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
