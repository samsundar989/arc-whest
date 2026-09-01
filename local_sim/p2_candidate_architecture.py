"""WP5 — the one shape-generic research candidate API every branch plugs into.

`GAME_PLAN.md` § 3A and § WP5. This module owns **configuration, per-MLP seed
derivation, chunking, result assembly, cost metadata and fallbacks**, and owns
no branch mathematics at all. The carrier's math lives in
`local_sim/p2_carrier_fnp.py`; the analytic control's lives in `_k2_diag` below
and nowhere else. Duplicating a branch here is the failure this file exists to
prevent — `local_sim/competition_score.py` was written for the same reason,
after twelve probes each carried a partial copy of the grader's identity.

THE CONFIGURATIONS
------------------
| name | branch | state |
|---|---|---|
| `k2_diag` | analytic control only | implemented |
| `carrier` | structured angular carrier only | implemented (WP1) |
| `carrier_k2_zero` | carrier + K2 at weight **exactly 0** | an IDENTITY test, not a design |
| `carrier_k2_frozen` | the hybrid placeholder | **coefficient UNSET; it REFUSES** |
| `augment` / `augment_k3` | WP3 / WP4 | **not implemented; they REFUSE** |

`carrier_k2_frozen` deliberately raises. `GAME_PLAN.md` § 3D requires a frozen
hybrid's interaction to be **preregistered**, and no DESIGN contract exists. A
placeholder that silently picked 0.5 would be a fitted constant with no
provenance, which is exactly what the three-tier protocol forbids. The refusal
is the feature; `test_p2_candidate_architecture.py` asserts it.

FALLBACK POLICY, AND WHY IT IS LOUD
------------------------------------
`GAME_PLAN.md` § 3D: *"Operational failure must not silently select a different
mathematical estimator."* A width with no Kerdock construction (not a power of
two, or `m` outside the recorded primitives) cannot carry the carrier. The
candidate therefore **falls back to `k2_diag` and records
`fallback_reason`** — it never pretends the carrier ran. `strict=True` turns the
fallback into a refusal, which is what a deployment gate should use.

CONVERTIBLE TO ONE FILE
-----------------------
Nothing here imports anything this repository owns except `p2_carrier_fnp`, and
that module is itself stdlib + `fnp`. `single_file_sources()` returns the exact
module list a promotion would concatenate, so the conversion is mechanical
rather than a rewrite. **No frozen `estimator_v*.py` archive is created here** —
that is a STRATEGIC promotion step with its own contract.

WHAT THIS MODULE DOES NOT CLAIM
-------------------------------
No accuracy, no score, no selection. `k2_diag` is a **reference control
implemented here**, NOT the shipped v2.6.1/v2.7 path; any comparison against the
baseline must run the shipped file itself. Nothing here reads truth.

Run: uv run python local_sim/p2_candidate_architecture.py
"""

from __future__ import annotations

import math

import p2_carrier_fnp as carrier_mod

#: Configurations, in the order a report should list them.
CONFIGS = ("k2_diag", "carrier", "carrier_k2_zero", "carrier_k2_frozen",
           "augment", "augment_k3")

IMPLEMENTED = ("k2_diag", "carrier", "carrier_k2_zero")

_SQRT_2PI = math.sqrt(2.0 * math.pi)


class CandidateRefusal(RuntimeError):
    """A configuration that exists but must not silently produce a number."""


# --------------------------------------------------------------------------- #
# branch 1 — the analytic control
# --------------------------------------------------------------------------- #

def _k2_diag(fnp, flops, weights, width, depth, dtype):
    """Diagonal second-moment (K2) propagation. The CONTROL, not the baseline.

    `h_0 = x ~ N(0, I)`, `z_l = h_{l-1} @ W_l`, `h_l = max(0, z_l)`, weights
    stored `(input, output)`.

    Layer 1 is EXACT and needs no Gaussian map: `z_j ~ N(0, ||W[:,j]||^2)`, so
    `E[ReLU(z_j)] = ||W[:,j]|| / sqrt(2 pi)`. Later layers use the exact ReLU
    Gaussian moments under a per-neuron normal assumption:

        a          = mu / sd
        E[ReLU]    = mu * Phi(a) + sd * phi(a)
        E[ReLU^2]  = (mu^2 + var) * Phi(a) + mu * sd * phi(a)

    and propagate `var_z = var_h @ (W * W)`, which DROPS the off-diagonal
    covariance. That omission is the whole reason this is a control: 94.61% of
    K2's error is accumulated mean/variance-state drift
    (`docs/CURRENT_STATE.md`, P2-CP3R).

    `flops.stats.norm.cdf/pdf` are used for `Phi`/`phi`. Flopscope 0.12
    **promotes their float32 input to float64** and says so with a
    `FlopscopeWarning`; `docs/CURRENT_STATE.md` records that this is exactly how
    an "f32" arm stayed float64 in name only. The dtype of the returned state is
    therefore reported by the telemetry rather than assumed.
    """
    zero = fnp.array(0.0, dtype=dtype)
    tiny = fnp.array(1e-30, dtype=dtype)

    w0 = weights[0]
    var_z = fnp.sum(w0 * w0, axis=0)                     # (width,)
    sd = fnp.sqrt(fnp.maximum(var_z, tiny))
    m = sd / _SQRT_2PI                                   # exact, layer 1
    e2 = 0.5 * var_z                                     # E[ReLU(z)^2] = var/2
    v = fnp.maximum(e2 - m * m, zero)
    # The REPORTED row 0 comes from the shared float64 closed form, not from
    # this branch's working-precision copy of it. In float32 the same identity
    # lands ~2.3e-7 from exact -- rounding, not method error, but still three
    # orders above what a closed-form row should cost. Reusing
    # `exact_layer1_means` also keeps ONE implementation of the identity: the
    # state below still needs `sd` in the working dtype, and that is the only
    # reason the two lines above exist at all.
    rows = [carrier_mod.exact_layer1_means(fnp, w0, dtype=dtype)]

    for li in range(1, depth):
        wl = weights[li]
        mu = m @ wl
        var = v @ (wl * wl)
        sd = fnp.sqrt(fnp.maximum(var, tiny))
        a = mu / sd
        Phi = flops.stats.norm.cdf(a)
        phi = flops.stats.norm.pdf(a)
        m = mu * Phi + sd * phi
        e2 = (mu * mu + var) * Phi + mu * sd * phi
        v = fnp.maximum(e2 - m * m, zero)
        rows.append(m)
    return fnp.stack(rows)


# --------------------------------------------------------------------------- #
# the candidate
# --------------------------------------------------------------------------- #

class ResearchCandidate:
    """One configurable, shape-generic Phase-2 candidate.

    `setup()` builds whatever depends only on the WIDTH — for the carrier
    branch, the whole point set — because the grader calls it once, off-budget.
    `predict()` is metered and must materialise no scalar.
    """

    def __init__(self, config="carrier", k=8, antipodes=True, chunk_rows=4096,
                 dtype="float32", blend=None, strict=False,
                 randomize=False, exact_layer1=True):
        if config not in CONFIGS:
            raise ValueError("unknown config %r; known: %s"
                             % (config, ", ".join(CONFIGS)))
        self.config = config
        self.k = k
        self.antipodes = antipodes
        self.chunk_rows = chunk_rows
        self.dtype = dtype
        self.strict = strict
        self.randomize = randomize
        #: Row 0 has a closed form; the carrier must not estimate it. See
        #: `p2_carrier_fnp.exact_layer1_means`. Off only for research
        #: comparisons of the pure quadrature.
        self.exact_layer1 = exact_layer1
        #: The carrier/analytic mixing weight. `None` means UNSET, and an unset
        #: coefficient must never be guessed.
        self.blend = blend
        self.plan = None
        self.fallback_reason = ""
        self.telemetry = {}

    # -- lifecycle --------------------------------------------------------- #
    def setup(self, context):
        fnp = _fnp()
        width = int(context.width)
        self.width = width
        self.depth = int(getattr(context, "depth", 0) or 0)

        if self.config in ("augment", "augment_k3"):
            raise CandidateRefusal(
                "config %r is declared but NOT IMPLEMENTED (GAME_PLAN WP3/WP4). "
                "It refuses rather than silently returning another branch's "
                "answer." % self.config)
        # THE REFUSAL IS ABOUT THE PROPERTY, NOT THE CONFIG NAME.
        #
        # It used to be scoped to `carrier_k2_frozen` alone, so
        # `carrier_k2_zero` with `blend=0.7` produced a hybrid carrying an
        # unregistered coefficient with no refusal and no `fallback_reason` --
        # an independent Skeptic constructed it. `carrier_k2_zero` exists to
        # test the two ENDPOINT identities (`blend=0` is exactly the carrier,
        # `blend=1` exactly the analytic branch); an interior coefficient there
        # is a design choice wearing a test's name.
        if (self.config == "carrier_k2_zero" and self.blend is not None
                and self.blend not in (0.0, 1.0)):
            raise CandidateRefusal(
                "config 'carrier_k2_zero' takes blend 0.0 or 1.0 only -- it "
                "exists to test the endpoint identities. blend=%r is an "
                "interior mixing coefficient, and GAME_PLAN section 3D requires "
                "a frozen hybrid's interaction to be preregistered. Use "
                "'carrier_k2_frozen' with an explicit contract." % (self.blend,))
        if self.config == "carrier_k2_frozen" and self.blend is None:
            raise CandidateRefusal(
                "config 'carrier_k2_frozen' has NO coefficient. GAME_PLAN "
                "section 3D requires a frozen hybrid's interaction to be "
                "preregistered, and no DESIGN contract exists. Picking a "
                "number here would fit a constant with no provenance.")

        if self._wants_carrier():
            try:
                self.plan = carrier_mod.CarrierPlan(
                    fnp, width, k=self.k, antipodes=self.antipodes,
                    chunk_rows=self.chunk_rows, dtype=self.dtype,
                    exact_layer1=self.exact_layer1)
            except ValueError as exc:
                # A width with no Kerdock construction. Say so, loudly.
                self.fallback_reason = (
                    "no Kerdock construction at width %d (%s); fell back to "
                    "k2_diag" % (width, exc))
                if self.strict:
                    raise CandidateRefusal(
                        "strict=True and " + self.fallback_reason) from exc
                self.plan = None
        self._setup_done = True
        return self

    #: Set by `setup()`. `predict()` before `setup()` used to fall through to
    #: the analytic branch and return the `k2_diag` answer bit-identically --
    #: "operational failure silently selects a different mathematical
    #: estimator", which this module's own docstring forbids and an independent
    #: Skeptic demonstrated. The grader always calls `setup()`, so it never bit
    #: in deployment; it bit every research caller who forgot.
    _setup_done = False

    def predict(self, mlp, flop_budget):
        if not self._setup_done:
            raise CandidateRefusal(
                "predict() called before setup(). The grader calls setup() once, "
                "off-budget, and the carrier is built there -- without it this "
                "would silently return the ANALYTIC branch's answer under a "
                "carrier config.")
        fnp, flops = _fnp(), _flops()
        weights = mlp.weights
        depth = len(weights)
        width = int(mlp.width)

        analytic = None
        carrier = None
        if self._use_carrier(width):
            car = self.plan.carrier
            if self.randomize:
                car = self._randomize(fnp, car, mlp)
            carrier = carrier_mod.propagate_means(
                fnp, car, weights, chunk_rows=self.chunk_rows,
                radial=self.plan.radial, dtype=self.dtype,
                exact_layer1=self.exact_layer1)
        if self._use_analytic(width):
            analytic = _k2_diag(fnp, flops, weights, width, depth, self.dtype)

        if carrier is None and analytic is None:
            raise CandidateRefusal(
                "config %r produced no branch at width %d" % (self.config, width))
        if carrier is None:
            return analytic
        if analytic is None:
            return carrier
        w = 0.0 if self.blend is None else float(self.blend)
        return carrier * (1.0 - w) + analytic * w

    # -- policy ------------------------------------------------------------ #
    def _wants_carrier(self):
        return self.config in ("carrier", "carrier_k2_zero", "carrier_k2_frozen")

    def _use_carrier(self, width):
        return self.plan is not None and int(self.plan.width) == width

    def _use_analytic(self, width):
        if self.config == "k2_diag":
            return True
        if self.config in ("carrier_k2_zero", "carrier_k2_frozen"):
            return True
        return not self._use_carrier(width)      # the fallback

    def _randomize(self, fnp, car, mlp):
        """A measure-preserving sign flip derived ONLY from `mlp.seed`.

        The carrier is closed under negation, so flipping the sign of any subset
        of COLUMNS maps the point set to another valid one with the same Gram
        structure -- it is an isometry of the sphere that permutes nothing about
        the design's strength. `GAME_PLAN.md` § 3B permits exactly this.

        `fnp.random.default_rng(mlp.seed)` is the only randomness, per hard rule
        2, and the flip is applied as an array multiply, so no scalar is
        materialised. Default OFF: whether randomising helps is a DESIGN
        question with no contract yet.
        """
        rng = fnp.random.default_rng(mlp.seed)
        bits = rng.integers(0, 2, size=int(car.shape[1]))
        signs = fnp.astype(1 - 2 * bits, car.dtype)
        return car * signs

    # -- metadata ---------------------------------------------------------- #
    def cost_model(self, depth):
        """Counted FLOPs this configuration expects, as an explicit formula.

        Reported so a measurement can DISAGREE with it. The carrier term is
        exact: `2 * R * n^2 * L` for the propagation. The analytic term is an
        order estimate only -- `flops.stats.norm` pricing is not modelled here
        and must be measured.
        """
        out = {"config": self.config, "depth": depth, "carrier_flops": 0,
               "analytic_flops_order": 0, "exact": False}
        if self.plan is not None:
            out["carrier_flops"] = self.plan.forward_flops(depth)
            out["carrier_rows"] = self.plan.n_rows
            out["exact"] = True
        if self._use_analytic(getattr(self, "width", 0)):
            n = getattr(self, "width", 0)
            out["analytic_flops_order"] = 4 * n * n * max(depth - 1, 0)
            out["exact"] = False
        out["multiplier_floor_headroom"] = (
            None if not out["carrier_flops"]
            else 0.1 * (2 ** 41) - out["carrier_flops"])
        return out

    def describe(self):
        return {
            "config": self.config,
            "implemented": self.config in IMPLEMENTED,
            "k": self.k, "antipodes": self.antipodes,
            "chunk_rows": self.chunk_rows, "dtype": self.dtype,
            "blend": self.blend, "strict": self.strict,
            "randomize": self.randomize, "exact_layer1": self.exact_layer1,
            "carrier_rows": None if self.plan is None else self.plan.n_rows,
            "carrier_lines": None if self.plan is None else self.plan.distinct_lines,
            "radial": None if self.plan is None else self.plan.radial,
            "fallback_reason": self.fallback_reason,
        }

    def __repr__(self):
        return "ResearchCandidate(%s)" % ", ".join(
            "%s=%r" % kv for kv in sorted(self.describe().items()))


def single_file_sources():
    """The modules a promotion would concatenate into ONE estimator file.

    Order matters: a later module may reference an earlier one's names once the
    `import` lines are stripped. Returned as paths rather than text so nothing
    here can be mistaken for having ALREADY produced a shippable artifact.
    """
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    return [os.path.join(here, "p2_carrier_fnp.py"),
            os.path.join(here, "p2_candidate_architecture.py")]


def _fnp():
    import flopscope.numpy as fnp
    return fnp


def _flops():
    import flopscope
    return flopscope


def main():
    import importlib
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p or ".") != here] + [root, here]
    flops = importlib.import_module("flopscope")
    try:
        flops.configure(symmetry_warnings=False)
    except Exception:
        pass

    class Ctx:
        width, depth, flop_budget, seed, n_mlps = 1024, 16, 2 ** 41, 1, 1

    print("research candidate configurations at 1024x16\n")
    for cfg in CONFIGS:
        try:
            c = ResearchCandidate(config=cfg).setup(Ctx())
        except CandidateRefusal as exc:
            print("  %-18s REFUSES -- %s" % (cfg, str(exc)[:88]))
            continue
        cm = c.cost_model(16)
        print("  %-18s rows=%-7s carrier FLOPs %-16s mult %s"
              % (cfg, c.describe()["carrier_rows"],
                 format(cm["carrier_flops"], ","),
                 "%.4f" % max(0.1, cm["carrier_flops"] / 2 ** 41)
                 if cm["carrier_flops"] else "n/a (analytic only)"))
    print()
    print("  single-file sources:")
    for p in single_file_sources():
        print("    %s" % os.path.relpath(p, root).replace("\\", "/"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
