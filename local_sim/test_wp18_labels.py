"""Does the NEW endpoint check actually fail on a broken labeller?

The old one could not: `explained_k0` was `1 - x/x`, and its companion compared a
function against itself. This mutates the regenerating path in four ways a real
defect could plausibly take, and reports whether the cross-path comparison fires.
A check that survives any of them is not a check for that defect.

One mutation was DROPPED after it was written, and the reason is worth keeping:
"make the single-layer restart regenerate" is a no-op, because `_run_layers` with
`read_pre=p` returns immediately after the linear step and never reaches the
regeneration branch. It changed nothing, so a check cannot be blamed for not
catching it -- a mutation that does not mutate tests the mutant, not the check.
"""
import os
import sys

sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.abspath("local_sim")]
for _p in ("local_sim", "."):
    if _p not in sys.path:
        sys.path.append(_p)

import torch  # noqa: E402
import p2_wp18_labels as lab  # noqa: E402
import p2_wp15_depth as wp15  # noqa: E402
import p2_range_lab as rl  # noqa: E402

W, D, P = 128, 6, 5
SEED = 90_000_000_802          # seed-ledger: no block named
TOL = 1e-10

lab._upstream()
mlp, K_in = wp15._build(SEED, W, D, "cuda", torch.float64)
GOOD_SNAP = wp15.regen_snapshots(mlp, K_in, D)
state = {"snap": GOOD_SNAP}


def crosspath():
    """The check's own comparison, isolated: (3,) and (2,1), by two routes."""
    s3, s21, _mu, _Sig = lab.regen_state(mlp, state["snap"], P)
    store = {}
    with torch.no_grad():
        rl.run_flow(mlp, K_in, hook_lin=rl._capture_hook(store, {P}),
                    hook_nl=rl._regen_hook_nl(),
                    up_to_layer="act%d" % (D - 1), output_d_max=1)
    g3, g21 = store[P][0].double(), store[P][1].double()
    return (float((s3.double() - g3).norm() / max(float(g3.norm()), 1e-300)),
            float((s21.double() - g21).norm() / max(float(g21.norm()), 1e-300)))


def report(name, r):
    hit = max(r) > TOL
    print("  %-38s rel3 %.3e  rel21 %.3e   %s"
          % (name, r[0], r[1], "CAUGHT" if hit else "MISSED"))
    return hit


print("  baseline (unmutated)                   rel3 %.3e  rel21 %.3e" % crosspath())
fired = []
real_regen_state = lab.regen_state

# M1: the "regenerating" flow never regenerates. This is the defect that would
#     make E3/E21 identically zero and every R^2 meaningless.
def _no_regen_snapshots(mlp_, K_in_, depth):
    import torch as _t
    from mlp_kprop.kprop_harmonic import (SIMPLE, clone_tower, coerce_input,
                                          linear_kprop, nonlin_kprop)
    from mlp_kprop.wick import WICK_COEF_D
    nw = [WICK_COEF_D[nm] for nm in mlp_.nonlin_names]
    K = coerce_input(K_in_, k_max=3, kind=SIMPLE)
    snap = {-1: clone_tower(K)}
    with _t.no_grad():
        for l in range(depth):
            K = linear_kprop(K, mlp_.Ws[l].weight, k_max=3,
                             set_metric=mlp_.init_scale[l], bias=mlp_.Ws[l].bias)
            K = nonlin_kprop(K, nonlin_wick_coef=nw[l], k_max=3, kind=SIMPLE,
                             use_pK=True, factor=True)
            snap[l] = clone_tower(K)          # <-- the regeneration is GONE
    return snap

state["snap"] = _no_regen_snapshots(mlp, K_in, D)
fired.append(report("M1 flow never regenerates", crosspath()))
state["snap"] = GOOD_SNAP

# M2: off by one layer -- restart from the wrong snapshot.
lab.regen_state = lambda m, s, p: real_regen_state(m, s, p - 1)
fired.append(report("M2 restarts one layer early", crosspath()))
lab.regen_state = real_regen_state

# M3: the (3,) slice taken from the diagonal of the (2,1) slice -- a
#     transcription defect leaving both objects present and both norms plausible.
def _m3(m, s, p):
    a, b, mu, Sig = real_regen_state(m, s, p)
    return b.diagonal().clone(), b, mu, Sig

lab.regen_state = _m3
fired.append(report("M3 (3,) taken from diag((2,1))", crosspath()))
lab.regen_state = real_regen_state

# M4: a 1e-6 RELATIVE perturbation. Tests that the tolerance is tight enough to
#     see a small numerical defect, not only a structural one -- WP16's whole
#     lesson was a 0.12% output difference hiding a 3.3x score defect.
def _m4(m, s, p):
    a, b, mu, Sig = real_regen_state(m, s, p)
    return a * (1.0 + 1e-6), b * (1.0 + 1e-6), mu, Sig

lab.regen_state = _m4
fired.append(report("M4 1e-6 relative perturbation", crosspath()))
lab.regen_state = real_regen_state

print()
print("  -> %d of %d mutations caught at tol %.0e   %s"
      % (sum(fired), len(fired), TOL,
         "the check is real" if all(fired) else "THE CHECK IS NOT REAL"))
raise SystemExit(0 if all(fired) else 1)
