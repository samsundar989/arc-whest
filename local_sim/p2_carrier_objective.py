"""WP2 — a TRUTH-FREE objective that ranks point sets, and the frozen panel.

Contract: `experiments/LOG.md`, *CONTRACT WP2* (2026-08-31), committed at
`6059c2c` **before this file existed**. `GAME_PLAN.md` § WP2.

THE OBJECTIVE
-------------
For a zonal kernel on the sphere, the worst-case squared integration error over
the RKHS unit ball is the maximum mean discrepancy. Against the *uniform*
measure it collapses to one Gram sum, because `E_v[k(u,v)]` is the same for
every unit `u` by rotational symmetry:

    MMD^2(U) = (1/R^2) sum_{i,j} kappa^L(<u_i, u_j>)  -  c_L(n)
    c_L(n)   = E[kappa^L(rho)],  rho = <two independent uniform unit vectors>

`kappa` is the arc-cosine (NNGP) kernel of a bias-free He-init ReLU layer on
unit vectors,

    kappa(rho) = ( sqrt(1 - rho^2) + rho * (pi - arccos rho) ) / pi

with `kappa(1) = 1`, `kappa(0) = 1/pi`, `kappa(-1) = 0`. It maps [-1,1] to
[0,1] and fixes 1, so depth composes as `kappa^L` and the unit sphere is
preserved. `MMD^2` is a squared RKHS norm, hence **non-negative**; expanding
`kappa^L` in Gegenbauer polynomials makes it a non-negative sum over harmonic
degrees, and a point set that integrates degree-`t` harmonics exactly zeroes
those terms.

Nothing here reads a label. That is the point: it ranks carriers *before* a
DESIGN block is spent, so the basis count is frozen rather than fitted.

TWO PATHS, AND THEY MUST AGREE
------------------------------
The structured carrier's Gram spectrum is exactly `{+1, -1, 0, +1/sqrt(N),
-1/sqrt(N)}` with a census fixed by `(k, N)`, so `MMD^2` has a **closed form**
costing O(1):

    +1              : R                      (self)
    -1              : R                      (antipode)
    0               : 4*k*N*(N-1)            (same basis, different column)
    +/-1/sqrt(N)    : 4*N^2*k*(k-1)          (different bases, split evenly)

An arbitrary point set needs the **chunked general** O(R^2) path. Both are
implemented and `test_p2_carrier_objective.py` requires them to agree — which
is what makes the cheap one trustworthy.

`kappa^L(-1)` IS NOT ZERO FOR L >= 2. It is `0` only at `L = 1`; at `L >= 2` it
is `kappa(0) = 1/pi`. Assuming otherwise produced NEGATIVE "squared
discrepancies" during design, which is how the error announced itself.

WHAT THIS CANNOT DO, AND IT IS NOT A DETAIL
--------------------------------------------
**An infinite-width NNGP kernel cannot certify finite-width, network-specific
ranking.** `MMD^2` measures prior-averaged integration error, and a realised
`1024 x 16` network departs from its infinite-width kernel by an amount this
objective cannot see. A good `MMD^2` is therefore **necessary, not sufficient**.
`GAME_PLAN.md` § 7 fixes what happens if it fails to predict DESIGN ranking:
*discard the proxy; do not optimize it harder.* **This module promotes nothing.**
It freezes a panel and a rule; a DESIGN block decides.

Run:  uv run python local_sim/p2_carrier_objective.py
      uv run python local_sim/p2_carrier_objective.py --panel --emit <path.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# local_sim ships shims NAMED flopscope/whestbench: APPEND, never insert.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

#: This is a RESEARCH objective, computed offline. It never runs inside
#: `predict`, so NumPy is legal here exactly as it is in `kerdock_mub.py`.
#: Nothing in this file may be inlined into a candidate.

DEFAULT_DEPTH = 16
#: Gauss-Legendre nodes for `c_L`. 6000 is far past convergence at every width
#: used here; `test_p2_carrier_objective.py` pins the convergence rather than
#: trusting the number.
CL_DEGREE = 6000


# --------------------------------------------------------------------------- #
# 1. the kernel
# --------------------------------------------------------------------------- #

def kappa(rho):
    """Arc-cosine (NNGP) kernel of one bias-free He-init ReLU layer, unit inputs.

    `kappa(1) = 1` exactly, so the recursion is norm-preserving on the sphere.
    """
    r = np.clip(np.asarray(rho, dtype=np.float64), -1.0, 1.0)
    return (np.sqrt(np.maximum(1.0 - r * r, 0.0))
            + r * (np.pi - np.arccos(r))) / np.pi


def kappa_L(rho, depth):
    """`kappa` composed `depth` times — the deep-ReLU kernel at that depth."""
    if depth < 0:
        raise ValueError("depth must be >= 0; got %r" % (depth,))
    r = np.asarray(rho, dtype=np.float64)
    for _ in range(depth):
        r = kappa(r)
    return r


def c_L(n, depth, degree=CL_DEGREE):
    """`E[kappa^L(rho)]` for `rho` the inner product of two uniform unit vectors.

    The density is `(1 - rho^2)^((n-3)/2)`, normalised. Quadrature rather than a
    closed form because `kappa^L` has no elementary antiderivative; the weight
    is exact and the integrand is smooth, so Gauss-Legendre converges fast.
    """
    if n < 2:
        raise ValueError("n must be >= 2; got %r" % (n,))
    x, w = np.polynomial.legendre.leggauss(int(degree))
    dens = (1.0 - x * x) ** ((n - 3) / 2.0)
    w = w * dens
    w = w / w.sum()
    return float((w * kappa_L(x, depth)).sum())


# --------------------------------------------------------------------------- #
# 2. the general path — any point set, chunked
# --------------------------------------------------------------------------- #

def mmd2(points, depth=DEFAULT_DEPTH, chunk=2048, cl=None, degree=CL_DEGREE):
    """`MMD^2` of a UNIT-NORM point set against the uniform sphere measure.

    `points` is `(R, n)`. Chunked over rows: the full Gram at `R = 16,384` is
    268M float64 entries (2 GiB), and this repository has already watched a
    neighbour's job die of commit pressure (`docs/GOTCHAS.md`).

    The rows must be unit norm — the kernel is only a sphere kernel there — and
    that is CHECKED, not assumed, because a mis-scaled carrier would otherwise
    return a plausible number.
    """
    U = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    R, n = U.shape
    norms = np.sqrt((U * U).sum(axis=1))
    worst = float(np.abs(norms - 1.0).max())
    if worst > 1e-6:
        raise ValueError(
            "MMD^2 needs UNIT-NORM rows; worst |‖u‖ - 1| = %.3e. A mis-scaled "
            "point set would otherwise return a plausible number." % worst)

    total = 0.0
    for s in range(0, R, chunk):
        block = U[s:s + chunk] @ U.T                     # (chunk, R)
        total += float(kappa_L(block, depth).sum())
    if cl is None:
        cl = c_L(n, depth, degree)
    return total / float(R) ** 2 - cl


def gram_power_sum(points, power, chunk=2048):
    """`sum_ij <u_i,u_j>^power`, chunked. For the exact algebraic nulls."""
    U = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    R = U.shape[0]
    total = 0.0
    for s in range(0, R, chunk):
        total += float(np.power(U[s:s + chunk] @ U.T, power).sum())
    return total


# --------------------------------------------------------------------------- #
# 3. the closed form — the structured carrier only
# --------------------------------------------------------------------------- #

def carrier_gram_census(k, n, antipodes=True):
    """The EXACT multiset of `k*N*(1 or 2)` squared-Gram entries, as counts.

    Verified against a direct Gram in `test_p2_carrier_objective.py`. Returns
    `{value: ordered-pair count}`; the counts sum to `R^2`.
    """
    if not antipodes:
        raise NotImplementedError(
            "the census is derived for the ANTIPODAL carrier, which is the "
            "only form the candidate ships")
    R = 2 * k * n
    c = 1.0 / math.sqrt(n)
    census = {
        1.0: R,                                  # self
        -1.0: R,                                 # antipode
        0.0: 4 * k * n * (n - 1),                # same basis, different column
        c: 2 * n * n * k * (k - 1),              # cross-basis, positive half
        -c: 2 * n * n * k * (k - 1),             # cross-basis, negative half
    }
    if sum(census.values()) != R * R:
        raise AssertionError("census does not close: %d vs %d"
                             % (sum(census.values()), R * R))
    return census


def mmd2_carrier_closed(k, n, depth=DEFAULT_DEPTH, cl=None, degree=CL_DEGREE):
    """`MMD^2` of the antipodal Kerdock/MUB carrier, in O(1).

    The whole Gram spectrum is `{+1, -1, 0, +1/sqrt(N), -1/sqrt(N)}` with a
    census fixed by `(k, N)`, so no `R^2` work is needed. At the live width this
    is the difference between microseconds and a 2 GiB allocation.
    """
    R = 2 * k * n
    census = carrier_gram_census(k, n)
    total = sum(cnt * float(kappa_L(val, depth)) for val, cnt in census.items())
    if cl is None:
        cl = c_L(n, depth, degree)
    return total / float(R) ** 2 - cl


# --------------------------------------------------------------------------- #
# 4. the controls, at matched row count
# --------------------------------------------------------------------------- #

def control_iid_antithetic(rows, n, seed):
    """Equal-row IID Gaussian directions, antipodally paired.

    Antithetic because the carrier is: comparing an antipodal set to a
    non-antipodal one would confound the pairing with the structure, and
    `PHASE2_FRONTIER_AUDIT` records antipodes as an independently useful trick.
    """
    if rows % 2:
        raise ValueError("antithetic control needs an even row count")
    rng = np.random.default_rng(seed)
    half = rng.standard_normal((rows // 2, n))
    half /= np.sqrt((half * half).sum(axis=1, keepdims=True))
    out = np.empty((rows, n), dtype=np.float64)
    out[0::2], out[1::2] = half, -half
    return out


def control_random_orthogonal(rows, n, seed):
    """Random orthonormal blocks, antipodally paired — the structured control.

    This is the fair comparison for a union of MUBs: same block structure, same
    row count, no mutual-unbiasedness.
    """
    if rows % (2 * n):
        raise ValueError("need a whole number of orthonormal blocks")
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(rows // (2 * n)):
        q, r = np.linalg.qr(rng.standard_normal((n, n)))
        blocks.append(q * np.sign(np.diag(r)))           # a Haar sample
    B = np.concatenate(blocks, axis=0)
    out = np.empty((rows, n), dtype=np.float64)
    out[0::2], out[1::2] = B, -B
    return out


def control_scrambled_lowdisc(rows, n, seed):
    """Scrambled low-discrepancy directions, antipodally paired.

    A randomised Hadamard/Rademacher scramble of the Walsh basis: cheap at any
    power-of-two width and genuinely low-discrepancy, without importing a Sobol
    generator this module has no other use for.
    """
    if rows % (2 * n):
        raise ValueError("need a whole number of blocks")
    import kerdock_mub as REF
    rng = np.random.default_rng(seed)
    W = REF.walsh_matrix(int(math.log2(n))).astype(np.float64) / math.sqrt(n)
    blocks = []
    for _ in range(rows // (2 * n)):
        s = rng.integers(0, 2, size=n) * 2 - 1           # row scramble
        p = rng.permutation(n)                            # column scramble
        blocks.append((W * s[:, None])[:, p])
    B = np.concatenate(blocks, axis=0)
    out = np.empty((rows, n), dtype=np.float64)
    out[0::2], out[1::2] = B, -B
    return out


def control_randomized_carrier(rows, n, k, seed):
    """The carrier itself under a random global rotation.

    `MMD^2` is rotation-invariant, so this must score IDENTICALLY to the
    carrier. It is a NULL CONTROL: if it does not, the objective is not
    rotation-invariant and every ranking it produces is suspect.
    """
    import kerdock_mub as REF
    rng = np.random.default_rng(seed)
    U = REF.carrier_float32(int(math.log2(n)) - 1, k, antipodes=True
                            ).astype(np.float64)
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return U @ (q * np.sign(np.diag(r)))


CONTROLS = {
    "iid_antithetic": control_iid_antithetic,
    "random_orthogonal": control_random_orthogonal,
    "scrambled_lowdisc": control_scrambled_lowdisc,
}


# --------------------------------------------------------------------------- #
# 5. THE FROZEN PANEL, and the rule that reads it
# --------------------------------------------------------------------------- #

#: Nested basis counts. 4/6/8/10/12 is the panel `GAME_PLAN.md` § 3B names;
#: 1, 2 and 16 are endpoints, present so monotonicity is visible and so 8 is
#: never the extreme of the range it is compared in. **Nested** means policy
#: `k` reuses the exact prefix of policy `k' > k`: added compute is added
#: bases, never a different object.
PANEL_K = (1, 2, 4, 6, 8, 10, 12, 16)

#: The counted-FLOP ceiling a policy must respect. Propagation is
#: `2 * R * n^2 * L` with `R = 2*k*n`, so `k` maps to cost linearly.
BUDGET = 2 ** 41

#: THE DETERMINISTIC PROXY-TO-POLICY RULE, written before any panel number
#: exists. No discretion at selection time.
POLICY_RULE = (
    "Among panel policies whose counted propagation cost is <= "
    "`cost_ceiling`, choose the one with the smallest MMD^2. Ties (relative "
    "difference < 1e-12) break toward the SMALLER k, because a smaller k is "
    "cheaper and the tie says the objective cannot separate them. If no policy "
    "fits the ceiling, the rule REFUSES rather than returning the cheapest."
)

#: What a later DESIGN block must show for the proxy to be believed at all.
DESIGN_FALSIFIER = (
    "The proxy is DISCARDED, not re-tuned, if the DESIGN ranking of the panel "
    "by paired adjusted score does not agree with the MMD^2 ranking in its "
    "top-3 ordering (Spearman over the panel <= 0). GAME_PLAN section 7: "
    "'carrier proxy does not predict DESIGN ranking -> discard the proxy; do "
    "not optimize it harder.'"
)

#: The score gate a later STRATEGIC contract would propose. NOT a gate yet:
#: recorded here so the number is fixed before any data, and it must be
#: restated and accepted in that contract before it binds.
PROPOSED_SCORE_GATE = (
    "A carrier policy is promotable only if the paired ARITHMETIC adjusted-"
    "score difference against exact v2.7, with a studentised-t or basic-"
    "bootstrap 95% interval excluding zero, shows >= 2x improvement on a fresh "
    "DESIGN block, replicated on a disjoint DESIGN block. Percentile bootstrap "
    "and BCa are refused. Every MSE carries its ground-truth noise floor."
)


def propagation_flops(k, n, depth):
    """`2 * R * n^2 * L`, `R = 2*k*n`. The carrier branch's whole counted cost."""
    return 2 * (2 * k * n) * n * n * depth


def build_panel(n=1024, depth=DEFAULT_DEPTH, ks=PANEL_K, cost_ceiling=None):
    """The frozen panel: every policy, its cost, and its MMD^2.

    Uses the closed form, which `test_p2_carrier_objective.py` pins against the
    chunked general path.
    """
    cl = c_L(n, depth)
    rows = []
    for k in ks:
        rows.append({
            "k": k,
            "rows": 2 * k * n,
            "distinct_lines": k * n,
            "propagation_flops": propagation_flops(k, n, depth),
            "multiplier": max(0.1, propagation_flops(k, n, depth) / BUDGET),
            "mmd2": mmd2_carrier_closed(k, n, depth, cl=cl),
        })
    return {
        "artifact": "p2_carrier_panel",
        "artifact_version": 1,
        "width": n,
        "depth": depth,
        "kernel": "arccos degree-1 (NNGP), composed L times, unit sphere",
        "c_L": cl,
        "budget": BUDGET,
        "cost_ceiling": cost_ceiling,
        "policy_rule": POLICY_RULE,
        "design_falsifier": DESIGN_FALSIFIER,
        "proposed_score_gate": PROPOSED_SCORE_GATE,
        "nested": True,
        "policies": rows,
        "limitation": (
            "An infinite-width NNGP kernel cannot certify finite-width, "
            "network-specific ranking. A good MMD^2 is NECESSARY, NOT "
            "SUFFICIENT. This panel selects nothing; a DESIGN block does."
        ),
    }


def select_policy(panel, cost_ceiling):
    """Apply `POLICY_RULE`. Deterministic, and it can REFUSE."""
    fits = [p for p in panel["policies"]
            if p["propagation_flops"] <= cost_ceiling]
    if not fits:
        raise ValueError(
            "no panel policy fits a %s-FLOP ceiling; the rule REFUSES rather "
            "than returning the cheapest" % format(cost_ceiling, ","))
    best = min(fits, key=lambda p: p["mmd2"])
    close = [p for p in fits
             if abs(p["mmd2"] - best["mmd2"]) / abs(best["mmd2"]) < 1e-12]
    return min(close, key=lambda p: p["k"])


def panel_digest(panel):
    """sha256 over a canonical serialisation, so the panel is content-addressed.

    Excludes nothing: if a policy, a cost, an MMD^2, the rule, the falsifier or
    the proposed gate moves, the digest moves. That is what "frozen" has to mean
    to be checkable.
    """
    return hashlib.sha256(
        json.dumps(panel, sort_keys=True, separators=(",", ":"),
                   default=str).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    ap.add_argument("--panel", action="store_true")
    ap.add_argument("--emit", default="")
    a = ap.parse_args()

    print("WP2 - the truth-free carrier objective")
    print("  kernel   arccos degree-1 (NNGP), composed %d times" % a.depth)
    print("  kappa(1) = %.10f   kappa(0) = %.10f   kappa(-1) = %.10f"
          % (kappa(1.0), kappa(0.0), kappa(-1.0)))
    print("  c_L(%d, %d) = %.12f" % (a.width, a.depth, c_L(a.width, a.depth)))
    print()

    panel = build_panel(a.width, a.depth)
    print("  the FROZEN panel at %d x %d" % (a.width, a.depth))
    print("    %-4s %-8s %-18s %-9s %s" % ("k", "rows", "prop FLOPs", "mult", "MMD^2"))
    prev = None
    for p in panel["policies"]:
        arrow = ""
        if prev is not None:
            arrow = "  v %.3fx" % (prev / p["mmd2"]) if p["mmd2"] < prev else "  ^ WORSE"
        prev = p["mmd2"]
        print("    %-4d %-8d %-18s %-9.4f %.6e%s"
              % (p["k"], p["rows"], format(p["propagation_flops"], ","),
                 p["multiplier"], p["mmd2"], arrow))
    print()
    print("  digest  %s" % panel_digest(panel))
    print()
    print("  RULE      %s" % POLICY_RULE)
    print()
    print("  NOTE: this panel SELECTS NOTHING. An infinite-width kernel cannot")
    print("  certify finite-width ranking; a DESIGN block decides, and")
    print("  GAME_PLAN section 7 discards the proxy if it does not predict that")
    print("  ranking.")

    if a.emit:
        d = os.path.dirname(os.path.abspath(a.emit))
        if d:
            os.makedirs(d, exist_ok=True)
        out = dict(panel, digest=panel_digest(panel))
        with open(a.emit, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(out, indent=2) + "\n")
        print("\n  emitted -> %s" % a.emit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
