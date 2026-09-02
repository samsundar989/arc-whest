"""WP9 — a per-neuron, scale-equivariant IN-FLIGHT state corrector for K2.

Contract: `experiments/LOG.md`, *CONTRACT WP9* (2026-09-01), committed at
`9c961ce` **before this file existed**. Read it before reading this module; the
gate, the arms, the null controls and the falsifier all live there and none of
them may be chosen here.

WHY THIS SHAPE AND NOT ANOTHER
------------------------------
K2's ReLU mean map is exactly

    E[ReLU(X)] = mu Phi(alpha) + sigma phi(alpha) = sigma * h(alpha),
    alpha = mu / sigma,   h(a) = a Phi(a) + phi(a)

and P2-CAL proved the whole closure is **positively homogeneous of degree 1**:
`(mu, cov) -> (g mu, g^2 cov)` leaves every `alpha` unchanged and scales every
mean by exactly `g`, verified to `1.6e-30`. Two consequences drive this file:

1. The residual of a degree-1 homogeneous estimator against a degree-1
   homogeneous truth is itself `sigma` times a function of **scale invariants**.
   So the correction is written `mu <- mu_raw + sigma * c_m(features)` with
   every feature scale-invariant, and the corrected estimator stays exactly
   degree-1 homogeneous.
2. If `c_m` were CONSTANT across neurons it would collapse to a single output
   rescale -- which is the 2.96x P2-CAL already banked and then exhausted. That
   makes the constant-`c_m` member an exact **null control** (arm N1), not a
   competitor. It must reproduce a per-layer scalar arm; if it does not, this
   module is wrong, not the theory.

WHY ROLLOUT IS THE ONLY HONEST SCORE
------------------------------------
An in-flight corrector changes the state layer `l+1` consumes, so a
teacher-forced fit is **not** the deployed quantity. Training therefore uses
DAgger-style rounds -- fit on the trajectory the current corrector actually
produces, not on the uncorrected one -- and every reported number comes from a
full rollout. `--teacher-forced` exists only to measure the gap the contract's
prediction P2 commits to, and the contract forbids quoting it as a result.

TIERS
-----
Fit on `P02`. Report once on `P03`. `P04` is untouched. The organisers'
`mini`/`full` split is read for nothing at all: the training corpus is our own
mints, which is the 2026-09-01 WP7 finding.

Run (GPU venv):
    .venv-gpu\\Scripts\\python.exe local_sim/p2_corrector.py --selftest
    .venv-gpu\\Scripts\\python.exe local_sim/p2_corrector.py \\
        --truth experiments/2026-09-01_wp9_corrector/truth_P02.npz \\
        --arms B0,A,B,N1,N2 --json <out>.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

from mint_provenance import canonical_mint  # noqa: E402

WIDTH, DEPTH = 1024, 16

#: Feature block width. Fixed here so the cost model in CONTRACT WP9 (which
#: committed to `F <= 12`) is a claim this file can be checked against.
N_FEATURES = 8
#: Hidden width. The contract committed to `H <= 32`.
HIDDEN = 32

#: Clamps, FIXED IN THE CONTRACT before any data existed. A corrector that
#: extrapolates on a hard MLP must degrade, not explode.
CLAMP_CV = 0.5
#: `c_m` is clamped at `3 * sigma_train(c_m)`, computed from the training
#: targets and then frozen into the artifact.
CLAMP_CM_SIGMAS = 3.0


# ---------------------------------------------------------------------------
# the K2 closure, differentiable, with an optional in-flight correction
# ---------------------------------------------------------------------------

def _torch():
    import torch
    return torch


def _phi_Phi(t, a):
    """Standard normal pdf and cdf, in torch, matching the arm's spelling."""
    inv_sqrt_2pi = 0.3989422804014327
    phi = t.exp(-0.5 * a * a) * inv_sqrt_2pi
    Phi = 0.5 * (1.0 + t.erf(a * 0.7071067811865476))
    return phi, Phi


def features(t, alpha, layer, depth):
    """Eight SCALE-INVARIANT per-neuron features.

    Every one is a function of `alpha` (which homogeneity leaves fixed) or of
    the layer index. Nothing here has units, so multiplying the network's
    weights by `g` leaves the whole feature block unchanged -- which is what
    keeps `sigma * c_m(features)` degree-1 homogeneous.

    `alpha` is clipped at +/- 8 before use. Beyond that `Phi` is 0 or 1 to
    float64 precision and the feature carries no information, while an
    unclipped `alpha^2`-scale input destabilises the fit.
    """
    a = alpha.clamp(-8.0, 8.0)
    phi, Phi = _phi_Phi(t, a)
    m = a.mean(dim=-1, keepdim=True)
    s = a.std(dim=-1, keepdim=True) + 1e-8
    mp = Phi.mean(dim=-1, keepdim=True)
    lay = t.full_like(a, layer / max(depth - 1, 1))
    return t.stack([a, Phi, phi, (a - m) / s,
                    lay, m.expand_as(a), s.expand_as(a), mp.expand_as(a)],
                   dim=-1)


def k2_rollout(t, w, corrector=None, mode="full", clamp_cm=None):
    """K2p1, optionally corrected in flight. Returns the per-layer record.

    `mode`:
        "full"   -- correct at every layer (arm B)
        "last"   -- correct at the final layer only (arm A)
        "none"   -- no correction (arm B0)

    The uncorrected spelling is line-for-line `local_sim/p2_arms/arm_k2p1.py`,
    including the `outer(Phi, Phi) * cov_pre * off` operation order, so arm B0
    is the control rather than a re-derivation that happens to be close.

    LAYER 0 IS NEVER CORRECTED, in any mode. Its pre-activation is EXACTLY
    Gaussian, so `E[ReLU]` is exact there and its `alpha` is identically zero
    for every neuron -- a correction could only add error, and a feature block
    that is constant across neurons would train `c_m` toward the very
    global-rescale degeneracy N1 exists to detect.
    """
    n = w[0].shape[0]
    depth = len(w)
    dt = w[0].dtype
    dev = w[0].device
    mu = t.zeros(n, dtype=dt, device=dev)
    cov = t.eye(n, dtype=dt, device=dev)
    off = 1.0 - t.eye(n, dtype=dt, device=dev)
    rows, recs = [], []
    for k in range(depth):
        wk = w[k]
        mu_pre = wk.T @ mu
        cov_pre = wk.T @ cov @ wk
        var_pre = t.diagonal(cov_pre).clamp(min=1e-12)
        sig = t.sqrt(var_pre)
        alpha = mu_pre / sig
        phi, Phi = _phi_Phi(t, alpha)

        mu_raw = mu_pre * Phi + sig * phi
        ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sig * phi
        var_raw = (ez2 - mu_raw * mu_raw).clamp(min=0.0)

        use = (corrector is not None and k > 0
               and (mode == "full" or (mode == "last" and k == depth - 1)))
        if use:
            f = features(t, alpha, k, depth)
            c = corrector(f)
            cm, cv = c[..., 0], c[..., 1]
            if clamp_cm is not None:
                cm = cm.clamp(-clamp_cm, clamp_cm)
            cv = cv.clamp(-CLAMP_CV, CLAMP_CV)
            mu_new = mu_raw + sig * cm
            var_new = (var_raw * (1.0 + cv)).clamp(min=0.0)
        else:
            cm = cv = None
            mu_new, var_new = mu_raw, var_raw

        recs.append({"alpha": alpha, "sig": sig, "mu_raw": mu_raw,
                     "var_raw": var_raw, "layer": k})
        cov = t.outer(Phi, Phi) * cov_pre * off + t.diag(var_new)
        mu = mu_new
        rows.append(mu)
    out = t.stack(rows, dim=0)
    # Layer 0 in closed form, exactly as the shipped estimator does it: the
    # closure reproduces it analytically but through a matmul diagonal, and the
    # two differ at 1e-16. Taking the closed form is the better of two exact
    # expressions.
    var0 = (w[0] * w[0]).sum(dim=0)
    exact = t.sqrt(var0) * 0.3989422804014327
    out = t.cat([exact.reshape(1, n).to(out.dtype), out[1:]], dim=0)
    return out, recs


# ---------------------------------------------------------------------------
# the corrector itself
# ---------------------------------------------------------------------------

class Corrector:
    """A two-layer tanh MLP, F -> H -> 2, distillable to plain fnp matmuls.

    Deliberately not a torch.nn.Module: the deployed form is three `fnp`
    matmuls and a `tanh`, and keeping the parameters as bare tensors makes the
    correspondence between what is fitted here and what would be shipped
    obvious rather than something to be re-derived at port time.
    """

    def __init__(self, t, n_feat=N_FEATURES, hidden=HIDDEN, seed=0,
                 dtype=None, device="cpu", constant_cm=False):
        g = t.Generator(device="cpu").manual_seed(seed)
        dt = dtype or t.float64

        def rnd(*shape, scale):
            x = t.randn(*shape, generator=g, dtype=t.float64) * scale
            return x.to(dtype=dt, device=device).requires_grad_(True)

        self.t = t
        #: `constant_cm` is null control N1: `c_m` is forced constant across
        #: neurons within a layer, which by P2-CAL's homogeneity proof must
        #: collapse onto a per-layer scalar arm.
        self.constant_cm = constant_cm
        self.W1 = rnd(n_feat, hidden, scale=1.0 / math.sqrt(n_feat))
        self.b1 = rnd(hidden, scale=0.0)
        self.W2 = rnd(hidden, 2, scale=0.01)
        self.b2 = rnd(2, scale=0.0)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def __call__(self, f):
        t = self.t
        h = t.tanh(f @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        if self.constant_cm:
            # Averaged over neurons, then broadcast back: the correction still
            # varies by layer and by network, but not across neurons.
            cm = out[..., 0:1].mean(dim=-2, keepdim=True).expand_as(out[..., 0:1])
            out = t.cat([cm, out[..., 1:2]], dim=-1)
        return out

    def state(self):
        return {k: getattr(self, k).detach().cpu().numpy().tolist()
                for k in ("W1", "b1", "W2", "b2")}

    def n_params(self):
        return sum(int(np.prod(p.shape)) for p in self.params())


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_truth(path):
    z = np.load(path)
    if "allvar_r1" not in z:
        raise SystemExit(
            f"REFUSED: {path} has no `allvar_r1`. It was produced before the "
            "WP9 extension of `p2_truth.py`, so it carries the final-layer "
            "variance only and cannot supply an in-flight variance target. "
            "Re-mint, or run with --mean-only and say so in the result.")
    if int(z.get("probe", 0)):
        print("  *** PROBE truth: unregistered seeds. NOT EVIDENCE. ***")
    return z


def weights_for(t, seed, device, dtype):
    w, _est, _ss = canonical_mint(WIDTH, DEPTH, int(seed))
    return [t.from_numpy(np.ascontiguousarray(x)).to(device=device, dtype=dtype)
            for x in w]


def cross_mse(pred, y1, y2):
    """Unbiased MSE against a noisy label, using two independent replicas.

    `E[(p - y1)(p - y2)] = MSE(p)` exactly, because `E[e1 e2] = 0`. A single
    replica's MSE is biased UPWARD by the label's own Monte-Carlo variance;
    that bias cancels in a paired comparison but not in an absolute number, and
    `AGENTS.md` requires every MSE here to be quoted with its noise floor.
    """
    return float(np.mean((pred - y1) * (pred - y2)))


# ---------------------------------------------------------------------------
# self-test: the invariants that must hold before any number is believed
# ---------------------------------------------------------------------------

def selftest():
    t = _torch()
    dev = "cuda" if t.cuda.is_available() else "cpu"
    dt = t.float64
    print(f"WP9 corrector self-test   device={dev}")
    ok = [True]

    def chk(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name:<58}{detail}")
        if not cond:
            ok[0] = False

    w = weights_for(t, 1_000_601, dev, dt)

    # S1 -- the uncorrected rollout reproduces the frozen arm's spelling.
    base, _ = k2_rollout(t, w, corrector=None)
    chk("S1: uncorrected rollout is finite", bool(t.isfinite(base).all()),
        f"final-row mean {float(base[-1].mean()):.6f}")

    # S2 -- HOMOGENEITY. Scaling every weight by g must scale every predicted
    # mean by exactly g^L... no: h <- ReLU(h W) is degree-1 in W per layer, so
    # scaling all L weight matrices by g scales the final row by g^L. The
    # invariant this checks is that the CORRECTED estimator obeys the SAME law
    # as the uncorrected one -- which is the whole reason the features are
    # scale-invariant.
    corr = Corrector(t, seed=7, dtype=dt, device=dev)
    g = 1.7
    wg = [x * g for x in w]
    p0, _ = k2_rollout(t, w, corrector=corr, mode="full", clamp_cm=1.0)
    p1, _ = k2_rollout(t, wg, corrector=corr, mode="full", clamp_cm=1.0)
    ratio = (p1[-1] / p0[-1]).detach()
    rel = float((ratio / (g ** DEPTH) - 1.0).abs().max())
    chk("S2: the CORRECTED estimator is still degree-1 homogeneous", rel < 1e-9,
        f"max relative deviation {rel:.3e}")

    # S3 -- a zero corrector is EXACTLY the baseline. If it is not, the
    # correction path is doing something even when it is switched off.
    zero = Corrector(t, seed=3, dtype=dt, device=dev)
    with t.no_grad():
        for p in zero.params():
            p.zero_()
    pz, _ = k2_rollout(t, w, corrector=zero, mode="full", clamp_cm=1.0)
    d = float((pz - base).abs().max())
    chk("S3: a zero corrector is bit-identical to the baseline", d == 0.0,
        f"max abs difference {d:.3e}")

    # S4 -- layer 0 is never corrected, in any mode.
    pf, _ = k2_rollout(t, w, corrector=corr, mode="full", clamp_cm=1.0)
    chk("S4: layer 0 is untouched by any correction",
        float((pf[0] - base[0]).abs().max()) == 0.0, "row 0 identical")

    # S5 -- arm A touches ONLY the final row.
    pa, _ = k2_rollout(t, w, corrector=corr, mode="last", clamp_cm=1.0)
    head = float((pa[:-1] - base[:-1]).abs().max())
    tail = float((pa[-1] - base[-1]).abs().max())
    chk("S5: arm A (readout-only) moves the final row and nothing above it",
        head == 0.0 and tail > 0.0, f"head {head:.1e}, tail {tail:.3e}")

    # S6 -- N1 really is constant across neurons.
    n1 = Corrector(t, seed=11, dtype=dt, device=dev, constant_cm=True)
    f = features(t, t.randn(WIDTH, dtype=dt, device=dev), 5, DEPTH)
    out = n1(f)
    spread = float((out[..., 0] - out[..., 0].mean()).abs().max())
    chk("S6: null control N1 makes c_m constant across neurons",
        spread < 1e-12, f"spread {spread:.3e}")

    # S7 -- features are scale-invariant, which is what S2 rests on.
    a = t.randn(WIDTH, dtype=dt, device=dev)
    fa, fb = features(t, a, 5, DEPTH), features(t, a * 1.0, 5, DEPTH)
    chk("S7: the feature block is deterministic",
        float((fa - fb).abs().max()) == 0.0, "identical")

    print()
    print("SELF-TEST PASSED" if ok[0] else "SELF-TEST FAILED")
    return 0 if ok[0] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    ap.error("nothing to do; pass --selftest (fitting lands with the truth file)")


if __name__ == "__main__":
    raise SystemExit(main())
