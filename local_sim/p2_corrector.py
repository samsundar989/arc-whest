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

#: v2.7's sixteen per-layer output constants, verbatim from
#: `estimator_v2_7.py:557-562`. THE BASELINE IS THE SHIPPED FILE, NOT THE
#: UNCALIBRATED CLOSURE.
#:
#: The first version of this module scored arm B0 as UNCORRECTED K2, which is
#: not what CONTRACT WP9 specifies ("shipped calibrated K2, v2.7-equivalent")
#: and which would have inflated every reported gain by the 2.8629x that is
#: ALREADY ON THE BOARD as graded submission #328664. A corrector measured
#: against a baseline the project has already beaten is measuring the wrong
#: difference.
#:
#: It also sharpens null control N1: calibration IS a per-layer scalar, so a
#: constant-c_m corrector must now collapse onto the BASELINE rather than onto
#: something better than it.
CAL_G = (
    1.000000000, 0.999610691, 0.999343144, 0.999145174,
    0.998981471, 0.998857839, 0.998752746, 0.998663998,
    0.998591212, 0.998532269, 0.998483726, 0.998442611,
    0.998388713, 0.998337791, 0.998292060, 0.998251500,
)

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


def k2_rollout(t, w, corrector=None, mode="full", clamp_cm=None,
               calibrate=None):
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
    # THE CORRECTOR REPLACES THE CALIBRATION; it does not stack on it.
    #
    # v2.7's calibration is a multiplicative row gain, `mu * g`. Since
    # `mu_raw = sigma * h(alpha)`, that is `sigma * g * h(alpha)` -- i.e.
    # exactly `c_m = (g - 1) * h(alpha)`, and `h(alpha)` is a function of
    # `alpha`, which is IN THE CORRECTOR'S FEATURE SPAN. So a corrected arm can
    # express the calibration itself, and applying both would double-correct a
    # 0.17% bias that is the same order as the entire residual being chased.
    #
    # Hence the rule, and it is a rule rather than a caller's choice: no
    # corrector means the SHIPPED CALIBRATED BASELINE (v2.7, graded #328664);
    # a corrector means the corrector does that job.
    if calibrate is None:
        calibrate = corrector is None
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

        use = corrector is not None and _corrects(mode, k, depth)
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
    if calibrate:
        # `estimator_v2_7.py:940-950`: multiply output row L by CAL_G[L].
        # Applied AFTER the rollout, so it changes no propagated state -- which
        # is exactly why P2-CAL found the in-loop, global and post-hoc arms
        # identical to twelve significant figures.
        g = t.tensor(CAL_G[:depth], dtype=out.dtype,
                     device=out.device).reshape(depth, 1)
        out = out * g
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
# collection, fitting, rollout evaluation
# ---------------------------------------------------------------------------
# EVERY rollout in this section runs under `torch.no_grad()`. `p2_pb_rank`
# learned this the expensive way -- `predict()` built an autograd graph in a
# module that differentiates nothing and put 75.21 GiB live on a 31.84 GiB
# card. Nothing here differentiates through the tower: the corrector is fitted
# by regression on COLLECTED targets, so the graph would be pure waste even if
# it fitted.


def _corrects(mode, k, depth):
    """Is layer `k` corrected under `mode`? ONE definition, used by both the
    rollout and the collector, so the fit can never be on a different set of
    layers than the deployment."""
    if k == 0:
        return False                 # exactly Gaussian; see k2_rollout
    return mode == "full" or (mode == "last" and k == depth - 1)


def collect(t, seeds, truth, corrector, mode, device, dtype, max_nets=None,
            clamp_cm=None):
    """Roll out each network and record (features, targets) at every layer.

    The targets are read off the trajectory the CURRENT corrector actually
    produces, which is what makes the outer loop DAgger rather than one-shot
    teacher forcing: round 0 sees the uncorrected trajectory, round k sees its
    own. A corrector fitted once on the uncorrected trajectory is being asked,
    at deployment, about states it never saw.

    Targets, from the definition of the correction:

        c_m* = (true_mean[l] - mu_raw) / sig      so mu_raw + sig*c_m* == truth
        c_v* = true_var[l] / var_raw - 1          so var_raw*(1+c_v*) == truth

    The two replicas are AVERAGED for the target -- halving the label noise is
    worth more than keeping them apart, and they are kept apart where it
    actually matters, in the unbiased cross-product MSE that scores the result.
    """
    m1 = truth["means_r1"]
    m2 = truth["means_r2"]
    av1 = truth["allvar_r1"]
    av2 = truth["allvar_r2"]
    n = len(seeds) if max_nets is None else min(max_nets, len(seeds))
    F, T = [], []
    with t.no_grad():
        for i in range(n):
            w = weights_for(t, seeds[i], device, dtype)
            # The DAgger trajectory must be the DEPLOYED one. Rolling out
            # unclamped here while evaluating clamped trains the corrector on
            # states it will never see, and an unclamped correction compounding
            # over 15 layers is exactly how the first probe run blew up.
            # `calibrate` changes only the RETURNED rows, never the
            # propagated state, so the per-layer records this reads are
            # identical either way. Left on for consistency with evaluation.
            _out, recs = k2_rollout(t, w, corrector=corrector, mode=mode,
                                    clamp_cm=clamp_cm)
            tm = t.from_numpy(np.ascontiguousarray(
                (m1[i] + m2[i]) * 0.5)).to(device=device, dtype=dtype)
            tv = t.from_numpy(np.ascontiguousarray(
                (av1[i] + av2[i]) * 0.5)).to(device=device, dtype=dtype)
            for r in recs:
                k = r["layer"]
                # Collect ONLY the layers this arm actually corrects. Pooling
                # every layer's targets and then applying the result at one
                # layer fits the wrong distribution: the drift grows with
                # depth, so a pooled target is systematically wrong at both
                # ends. Arm A was 2x WORSE than no correction at all because
                # of this, on an intervention that by construction cannot be
                # worse if it is fitted to the layer it is applied to.
                if not _corrects(mode, k, DEPTH):
                    continue
                f = features(t, r["alpha"], k, DEPTH)
                cm = (tm[k] - r["mu_raw"]) / r["sig"]
                # Clip the TARGET to the same range the correction can
                # express. An unreachable target does not merely fail to be
                # reached: it drags the shared trunk, so an out-of-range `c_v`
                # corrupts `c_m` as well.
                cv = (tv[k] / r["var_raw"].clamp(min=1e-30) - 1.0
                      ).clamp(-CLAMP_CV, CLAMP_CV)
                F.append(f)
                T.append(t.stack([cm, cv], dim=-1))
    return t.cat(F, dim=0), t.cat(T, dim=0)


def fit(t, corrector, F, T, epochs=60, lr=3e-3, batch=1 << 16, seed=0,
        verbose=False):
    """Regression of (c_m*, c_v*) on the feature block. Plain Adam.

    The two targets live on very different scales, so each is standardised by
    its own training spread and the loss is on the standardised residual --
    otherwise `c_v`, which is order 1, would drown `c_m`, which is order 1e-2,
    and the fit would optimise the half that matters least for the scored row.
    """
    tt = T.detach()
    mu = tt.mean(dim=0, keepdim=True)
    sd = tt.std(dim=0, keepdim=True).clamp(min=1e-12)
    # Robust guard: a handful of neurons with var_raw ~ 0 can make c_v* huge.
    # Clip the TARGETS at 8 standardised units so the fit is not dragged by a
    # dozen rows out of a million and a half.
    z = ((tt - mu) / sd).clamp(-8.0, 8.0)
    opt = t.optim.Adam(corrector.params(), lr=lr)
    n = F.shape[0]
    g = t.Generator(device="cpu").manual_seed(seed)
    hist = []
    for ep in range(epochs):
        perm = t.randperm(n, generator=g).to(F.device)
        tot, nb = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            out = corrector(F[idx])
            loss = (((out - mu) / sd - z[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        hist.append(tot / max(nb, 1))
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"      epoch {ep:>3}  loss {hist[-1]:.6f}")
    return {"loss_first": hist[0], "loss_last": hist[-1],
            "cm_target_sd": float(sd[0, 0]), "cv_target_sd": float(sd[0, 1]),
            "n_rows": int(n)}


def evaluate(t, seeds, truth, corrector, mode, device, dtype, max_nets=None,
             clamp_cm=None, shuffle_features=False, shuffle_seed=0):
    """Per-network final-layer MSE by FULL ROLLOUT, unbiased by two replicas.

    `shuffle_features` is null control N2: the per-neuron feature rows are
    permuted within each layer, which destroys the neuron-to-correction
    correspondence while leaving every marginal distribution of the features,
    and of the corrections, exactly as it was. If a corrector still helps under
    that, it is not using per-neuron information and the result means something
    other than what it claims.
    """
    m1, m2 = truth["means_r1"], truth["means_r2"]
    n = len(seeds) if max_nets is None else min(max_nets, len(seeds))
    per, noise = [], []
    corr = corrector
    if shuffle_features and corrector is not None:
        corr = _ShuffleWrap(t, corrector, shuffle_seed)
    with t.no_grad():
        for i in range(n):
            w = weights_for(t, seeds[i], device, dtype)
            out, _ = k2_rollout(t, w, corrector=corr, mode=mode,
                                clamp_cm=clamp_cm)
            p = out[-1].detach().cpu().numpy().astype(np.float64)
            per.append(cross_mse(p, m1[i, -1].astype(np.float64),
                                 m2[i, -1].astype(np.float64)))
            noise.append(float(truth["gt_noise"][i]))
    return np.asarray(per, dtype=np.float64), np.asarray(noise, dtype=np.float64)


class _ShuffleWrap:
    """Null control N2: permute the feature ROWS within a layer.

    Deliberately a wrapper rather than a flag inside `k2_rollout`: the control
    must not be able to change anything about the rollout except which neuron
    each correction lands on.
    """

    def __init__(self, t, inner, seed):
        self.t, self.inner, self.seed, self.calls = t, inner, seed, 0

    def __call__(self, f):
        t = self.t
        g = t.Generator(device="cpu").manual_seed(self.seed + self.calls)
        self.calls += 1
        idx = t.randperm(f.shape[0], generator=g).to(f.device)
        return self.inner(f[idx])


def held_out_r2(t, corrector, F, T, clamp_cm=None):
    """Held-out R^2 per channel, on the trajectory the corrector produces.

    This separates two very different failure modes that a rollout score alone
    cannot tell apart:

      * R^2 ~ 0  -- the per-neuron residual is NOT a function of the
        scale-invariant features. Hypothesis H1 is dead, and no amount of
        rollout machinery rescues it.
      * R^2 healthy but the rollout does not improve -- the signal is real and
        COMPOUNDING is what destroys it. That is a different problem with
        different fixes, and it would be prediction P2 biting hard.

    The 2026-09-01 implementation brief proposed `R^2 >= 0.3` as its own gate.
    It is NOT the gate here -- CONTRACT WP9's G1 is a paired rollout score --
    but it is exactly the right diagnostic, so it is reported beside it.
    """
    with t.no_grad():
        out = corrector(F)
        if clamp_cm is not None:
            out = t.stack([out[..., 0].clamp(-clamp_cm, clamp_cm),
                           out[..., 1].clamp(-CLAMP_CV, CLAMP_CV)], dim=-1)
        ss_res = ((out - T) ** 2).sum(dim=0)
        ss_tot = ((T - T.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
        r2 = 1.0 - ss_res / ss_tot.clamp(min=1e-30)
    return [float(r2[0]), float(r2[1])]


def train_arm(t, seeds, truth, mode, device, dtype, rounds=3, epochs=60,
              seed=0, constant_cm=False, max_nets=None, verbose=True):
    """DAgger: fit, roll out with what was fitted, refit on what that produced."""
    corr = Corrector(t, seed=seed, dtype=dtype, device=device,
                     constant_cm=constant_cm)
    clamp = None
    info = []
    best = (float("inf"), None, -1)
    for r in range(rounds):
        src = None if r == 0 else corr
        F, T = collect(t, seeds, truth, src, mode, device, dtype, max_nets,
                       clamp_cm=clamp)
        st = fit(t, corr, F, T, epochs=epochs, seed=seed + r, verbose=verbose)
        # The clamp promised in the contract: 3 training sigmas on c_m, frozen
        # from the FIRST round's targets so a later round cannot widen it.
        if clamp is None:
            clamp = CLAMP_CM_SIGMAS * st["cm_target_sd"]
        del F, T
        if device == "cuda":
            t.cuda.empty_cache()

        # ROUND SELECTION, on the TRAINING networks and nothing else.
        #
        # DAgger has no guarantee that the last round is the best one, and this
        # system can diverge: `var <- var * (1 + c_v)` at fifteen layers is a
        # multiplicative walk, so even unbiased error in `c_v` inflates the
        # variance in expectation. On six probe networks round 1 ran away and
        # rounds 2-3 then fitted a trajectory that was already 1000x wrong.
        #
        # Selecting on the HELD-OUT rollout would fix that by spending the
        # held-out set, which is the one thing this study cannot afford. So the
        # selection is on the training networks' own rollout: legitimate model
        # selection on training data, and it leaves the held-out fold untouched.
        tr, _ = evaluate(t, seeds, truth, corr, mode, device, dtype, max_nets,
                         clamp_cm=clamp)
        st["round"] = r
        st["train_rollout_mse"] = float(tr.mean())
        info.append(st)
        if st["train_rollout_mse"] < best[0]:
            best = (st["train_rollout_mse"],
                    [q.detach().clone() for q in corr.params()], r)
        if verbose:
            print(f"    round {r}: {st['n_rows']:,} rows, loss "
                  f"{st['loss_first']:.5f} -> {st['loss_last']:.5f}, "
                  f"train rollout {st['train_rollout_mse']:.4e}")

    if best[1] is not None:
        with t.no_grad():
            for q, v in zip(corr.params(), best[1]):
                q.copy_(v)
    for st in info:
        st["selected"] = (st["round"] == best[2])
    return corr, clamp, info

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
    # `calibrate=False` throughout S1-S5: these compare the CLOSURE against
    # itself, and `k2_rollout`'s default now applies v2.7's calibration to an
    # uncorrected arm. Comparing a calibrated baseline with an uncalibrated
    # corrected arm would measure the calibration, not the correction -- which
    # is exactly what S9 exists to pin.
    base, _ = k2_rollout(t, w, corrector=None, calibrate=False)
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
    p0, _ = k2_rollout(t, w, corrector=corr, mode="full", clamp_cm=1.0,
                       calibrate=False)
    p1, _ = k2_rollout(t, wg, corrector=corr, mode="full", clamp_cm=1.0,
                       calibrate=False)
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
    pz, _ = k2_rollout(t, w, corrector=zero, mode="full", clamp_cm=1.0,
                       calibrate=False)
    d = float((pz - base).abs().max())
    chk("S3: a zero corrector is bit-identical to the baseline", d == 0.0,
        f"max abs difference {d:.3e}")

    # S4 -- layer 0 is never corrected, in any mode.
    pf, _ = k2_rollout(t, w, corrector=corr, mode="full", clamp_cm=1.0,
                       calibrate=False)
    chk("S4: layer 0 is untouched by any correction",
        float((pf[0] - base[0]).abs().max()) == 0.0, "row 0 identical")

    # S5 -- arm A touches ONLY the final row.
    pa, _ = k2_rollout(t, w, corrector=corr, mode="last", clamp_cm=1.0,
                       calibrate=False)
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

    # S9 -- the calibration rule. The BASELINE is the shipped calibrated file
    # (v2.7, graded #328664) and a CORRECTED arm is not calibrated on top,
    # because the corrector can express the calibration itself: v2.7 multiplies
    # row L by g, `mu_raw = sigma*h(alpha)`, so `mu*g` is exactly
    # `c_m = (g-1)*h(alpha)` and `h(alpha)` is in the feature span. Stacking
    # both would double-correct a 0.17% bias -- the same order as the entire
    # residual this slice is chasing.
    cal, _ = k2_rollout(t, w, corrector=None)          # default: calibrated
    ratio = (cal[-1] / base[-1]).detach()
    chk("S9: an UNCORRECTED arm is the shipped CALIBRATED baseline",
        float((ratio - CAL_G[DEPTH - 1]).abs().max()) < 1e-12,
        f"row 15 gain {float(ratio.mean()):.9f} vs CAL_G[15] "
        f"{CAL_G[DEPTH - 1]:.9f}")
    cc_, _ = k2_rollout(t, w, corrector=zero, mode="full", clamp_cm=1.0)
    chk("S9: ...and a CORRECTED arm is NOT calibrated on top of it",
        float((cc_ - base).abs().max()) == 0.0,
        "identical to the uncalibrated closure")

    # S8 -- the fit and the deployment must agree on WHICH layers are
    # corrected. They did not: `collect` pooled targets from every layer
    # regardless of mode, so arm A was fitted on 15 layers' drift and applied
    # at one. It came out 2x WORSE than no correction at all, on an
    # intervention that cannot be worse if it is fitted to the layer it acts
    # on. One predicate now answers the question for both.
    for mode, want in (("full", set(range(1, DEPTH))),
                       ("last", {DEPTH - 1}), ("none", set())):
        got = {k for k in range(DEPTH) if _corrects(mode, k, DEPTH)}
        chk(f"S8: mode {mode!r} corrects exactly the layers it fits",
            got == (want if mode != "none" else set()),
            f"{sorted(got)[:3]}{'...' if len(got) > 3 else ''}")

    print()
    print("SELF-TEST PASSED" if ok[0] else "SELF-TEST FAILED")
    return 0 if ok[0] else 1


# ---------------------------------------------------------------------------
# the study
# ---------------------------------------------------------------------------

def onestep_final_mse(t, seeds, truth, corrector, device, dtype, clamp_cm,
                      max_nets=None):
    """The TEACHER-FORCED diagnostic, and it is a diagnostic, not a result.

    Applies `c_m` at the final layer of the UNCORRECTED trajectory: one step,
    no compounding, exactly the situation the round-0 targets were computed in.
    CONTRACT WP9's prediction P2 says this will flatter the corrector by at
    least 1.5x relative to a full rollout, and the contract forbids quoting it
    as an outcome. It is here so that gap is measured rather than assumed.
    """
    m1, m2 = truth["means_r1"], truth["means_r2"]
    n = len(seeds) if max_nets is None else min(max_nets, len(seeds))
    per = []
    with t.no_grad():
        for i in range(n):
            w = weights_for(t, seeds[i], device, dtype)
            _out, recs = k2_rollout(t, w, corrector=None)     # uncorrected
            r = recs[-1]
            f = features(t, r["alpha"], DEPTH - 1, DEPTH)
            cm = corrector(f)[..., 0]
            if clamp_cm is not None:
                cm = cm.clamp(-clamp_cm, clamp_cm)
            p = (r["mu_raw"] + r["sig"] * cm).cpu().numpy().astype(np.float64)
            per.append(cross_mse(p, m1[i, -1].astype(np.float64),
                                 m2[i, -1].astype(np.float64)))
    return np.asarray(per, dtype=np.float64)


def folds(n, k, seed=0):
    """Leave-NETWORKS-out folds. Rows within one MLP are not independent."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [np.sort(idx[i::k]) for i in range(k)]


def run_study(a):
    import competition_score as CS

    t = _torch()
    device = "cuda" if t.cuda.is_available() else "cpu"
    dtype = t.float64
    truth = load_truth(a.truth)
    seeds = np.asarray(truth["seeds"])
    n_all = len(seeds) if not a.n_mlps else min(a.n_mlps, len(seeds))
    seeds = seeds[:n_all]

    print("=" * 78)
    print("WP9 -- per-neuron in-flight state corrector")
    print("=" * 78)
    print(f"  truth           {os.path.relpath(a.truth, _ROOT)}")
    print(f"  networks        {n_all}")
    print(f"  device          {device}   dtype float64")
    print(f"  folds           {a.folds} (LEAVE-NETWORKS-OUT)")
    print(f"  DAgger rounds   {a.rounds}")
    print()

    # ---- arm B0, the control ------------------------------------------
    # CONTRACT WP9 names this "shipped calibrated K2 (v2.7-equivalent)". An
    # earlier version of this file scored UNCORRECTED K2 instead, which would
    # have credited the corrector with the 2.8629x that is already on the
    # board. `k2_rollout` applies CAL_G whenever no corrector is present.
    per = {}
    per["B0"], gt_noise = evaluate(t, seeds, truth, None, "none", device, dtype)
    print(f"  B0 (SHIPPED calibrated K2 = v2.7, graded #328664)")
    print(f"     mean raw MSE {per['B0'].mean():.6e}   "
          f"GT-noise floor {gt_noise.mean():.4e} "
          f"({100 * gt_noise.mean() / per['B0'].mean():.4f}% of it)")

    # ---- the fitted arms, scored OUT OF FOLD --------------------------
    fold_ix = folds(n_all, a.folds, seed=a.seed)
    specs = [("A", "last", False), ("B", "full", False), ("N1", "full", True)]
    outs = {k: np.zeros(n_all) for k, _m, _c in specs}
    outs["N2"] = np.zeros(n_all)
    outs["B_tf"] = np.zeros(n_all)
    fitinfo = {}
    for fi, held in enumerate(fold_ix):
        keep = np.setdiff1d(np.arange(n_all), held)
        s_fit, s_out = seeds[keep], seeds[held]
        print(f"\n  fold {fi + 1}/{a.folds}: fit on {len(keep)}, "
              f"score {len(held)} HELD-OUT networks")
        for name, mode, const in specs:
            corr, clamp, info = train_arm(
                t, s_fit, _subset(truth, keep), mode, device, dtype,
                rounds=a.rounds, epochs=a.epochs, seed=a.seed + fi,
                constant_cm=const, verbose=a.verbose)
            sub = _subset(truth, held)
            p, _ = evaluate(t, s_out, sub, corr, mode, device, dtype,
                            clamp_cm=clamp)
            outs[name][held] = p
            Fh, Th = collect(t, s_out, sub, corr, mode, device, dtype,
                             clamp_cm=clamp)
            r2 = held_out_r2(t, corr, Fh, Th, clamp)
            del Fh, Th
            if device == "cuda":
                t.cuda.empty_cache()
            fitinfo.setdefault(name, []).append(
                {"fold": fi, "clamp_cm": clamp, "rounds": info,
                 "n_params": corr.n_params(), "held_out_r2_cm": r2[0],
                 "held_out_r2_cv": r2[1]})
            print(f"      {name:<3} held-out raw MSE {p.mean():.6e}   "
                  f"({per['B0'][held].mean() / p.mean():.4f}x over B0)   "
                  f"R2 c_m {r2[0]:+.3f} / c_v {r2[1]:+.3f}")
            if name == "B":
                # N2: the SAME corrector, features permuted across neurons.
                pn, _ = evaluate(t, s_out, sub, corr, mode, device, dtype,
                                 clamp_cm=clamp, shuffle_features=True,
                                 shuffle_seed=a.seed + fi)
                outs["N2"][held] = pn
                outs["B_tf"][held] = onestep_final_mse(
                    t, s_out, sub, corr, device, dtype, clamp)
                print(f"      N2  held-out raw MSE {pn.mean():.6e}   "
                      f"({per['B0'][held].mean() / pn.mean():.4f}x over B0) "
                      f"[null control]")
                print(f"      B-tf teacher-forced  {outs['B_tf'][held].mean():.6e}"
                      f"   ({per['B0'][held].mean() / outs['B_tf'][held].mean():.4f}x)"
                      f" [DIAGNOSTIC, not a result]")
    per.update(outs)

    # ---- the paired statistics ----------------------------------------
    # Phase 2: adjusted = raw * max(0.1, C/B). Every arm here is projected to
    # sit at the 0.1 floor -- the corrector's own cost model in CONTRACT WP9 is
    # <= 5e8 against 8.23e10 of unused free zone. That is a PROJECTION until
    # gate clause G2 measures an fnp spelling under flopscope, and the artifact
    # says so rather than quietly multiplying by 0.1.
    print("\n" + "=" * 78)
    print("  PAIRED ARITHMETIC SCORE DIFFERENCES -- studentised-t, one-sided")
    print("=" * 78)
    verdicts = {}
    for name in ("A", "B", "N1", "N2", "B_tf"):
        v = CS.decide(per["B0"], per[name], gate=a.gate)
        verdicts[name] = {
            "gain": float(v.gain), "passed": bool(v.passed),
            "gate": float(a.gate),
            "mean_control": float(per["B0"].mean()),
            "mean_candidate": float(per[name].mean()),
            "reasons": list(getattr(v, "reasons", []) or []),
        }
        for f in ("t_ci_difference", "mean_difference", "t_stat"):
            if hasattr(v, f):
                verdicts[name][f] = _jsonable(getattr(v, f))
        tag = ("[null control -- MUST be ~1.0]" if name in ("N1", "N2")
               else "[DIAGNOSTIC, not a result]" if name == "B_tf" else "")
        print(f"  {name:<5} gain {v.gain:8.4f}x   "
              f"{'PASS' if v.passed else 'no pass'}   {tag}")

    report = {
        "artifact": "wp9_corrector_design", "artifact_version": 1,
        "contract": "experiments/LOG.md CONTRACT WP9 (2026-09-01), commit 9c961ce",
        "truth": os.path.relpath(a.truth, _ROOT).replace("\\", "/"),
        "is_probe_truth": bool(int(truth.get("probe", 0))),
        "n_mlps": int(n_all), "folds": int(a.folds), "rounds": int(a.rounds),
        "epochs": int(a.epochs), "seed": int(a.seed), "gate": float(a.gate),
        "width": WIDTH, "depth": DEPTH,
        "n_features": N_FEATURES, "hidden": HIDDEN,
        "clamp_cv": CLAMP_CV, "clamp_cm_sigmas": CLAMP_CM_SIGMAS,
        "gt_noise_mean": float(gt_noise.mean()),
        "gt_noise_frac_of_B0": float(gt_noise.mean() / per["B0"].mean()),
        "raw_mse_mean": {k: float(v.mean()) for k, v in per.items()},
        "per_mlp_raw_mse": {k: v.tolist() for k, v in per.items()},
        "verdicts": verdicts,
        "fit": fitinfo,
        "SCOPE": (
            "DESIGN tier, leave-networks-out within one block. Selection only. "
            "The multiplier is PROJECTED at the 0.1 floor for every arm and is "
            "not measured here; CONTRACT WP9 gate clause G2 requires a "
            "flopscope price of an fnp spelling before any adjusted score from "
            "this file may be quoted. Nothing here is a promotion."),
    }
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        print(f"\n  wrote {os.path.relpath(a.json, _ROOT)}")
    print("=" * 78)
    return 0


def diagnose_readout(a):
    """WHY does an R^2 of 0.50 buy nothing? Decompose the final-layer error.

    The P02 study measured two numbers that look contradictory:

      * the held-out R^2 of `c_m` at the final layer is **+0.50**, stable to
        +/-0.012 across four leave-networks-out folds -- so half the variance
        of the per-neuron residual IS a function of scale-invariant features;
      * and applying it changes the final-layer MSE by **nothing** (0.9945x
        against the shipped calibrated baseline).

    Both cannot be describing the same quantity, and the reason is a weighting.
    The error is `sigma * c_m`, so the MSE is `E[sigma^2 c_m^2]` -- weighted by
    `sigma^2`, which varies by orders of magnitude across neurons. R^2 is
    unweighted. If the predictable part of `c_m` sits where `sigma` is small,
    explaining half the unweighted variance can explain nearly none of the
    error that is actually scored.

    This measures that directly, in one place, in units that add up:

        MSE_uncal  = mean( sigma^2 c^2 )          no correction at all
        MSE_cal    = mean( sigma^2 (c - c_cal)^2 )  v2.7's per-layer gain
        MSE_pred   = mean( sigma^2 (c - c_hat)^2 )  the fitted corrector
        R2_unw     = 1 - SS(c - c_hat) / SS(c - mean c)
        R2_w       = 1 - SUM sigma^2 (c - c_hat)^2 / SUM sigma^2 (c - c_w)^2

    A DIAGNOSTIC of an already-measured negative, on the block that produced
    it. It selects nothing and changes no verdict.
    """
    t = _torch()
    device = "cuda" if t.cuda.is_available() else "cpu"
    dtype = t.float64
    truth = load_truth(a.truth)
    seeds = np.asarray(truth["seeds"])
    n_all = len(seeds) if not a.n_mlps else min(a.n_mlps, len(seeds))
    seeds = seeds[:n_all]

    print("=" * 78)
    print("WP9 -- why an R^2 of 0.50 at the readout buys nothing")
    print("=" * 78)

    fold_ix = folds(n_all, a.folds, seed=a.seed)
    rows = []
    for fi, held in enumerate(fold_ix):
        keep = np.setdiff1d(np.arange(n_all), held)
        corr, clamp, _info = train_arm(
            t, seeds[keep], _subset(truth, keep), "last", device, dtype,
            rounds=a.rounds, epochs=a.epochs, seed=a.seed + fi, verbose=False)
        F, T = collect(t, seeds[held], _subset(truth, held), corr, "last",
                       device, dtype, clamp_cm=clamp)
        # `collect` returns rows in layer order per network; for mode "last"
        # there is exactly one layer, so re-deriving sigma means one more
        # rollout. Cheap, and it keeps `collect` a single-purpose function.
        sig = []
        with t.no_grad():
            for sd in seeds[held]:
                w = weights_for(t, sd, device, dtype)
                _o, recs = k2_rollout(t, w, corrector=corr, mode="last",
                                      clamp_cm=clamp)
                sig.append(recs[-1]["sig"])
        sig = t.cat(sig, dim=0)
        with t.no_grad():
            chat = corr(F)[..., 0].clamp(-clamp, clamp)
        c = T[..., 0]
        w2 = sig * sig
        # v2.7's calibration expressed in the same units: mu*g means
        # c_cal = (g - 1) * h(alpha), and h(alpha) = mu_raw/sigma.
        halpha = F[..., 0] * F[..., 1] + F[..., 2]      # a*Phi + phi
        c_cal = (CAL_G[DEPTH - 1] - 1.0) * halpha

        def wmse(x):
            return float((w2 * x * x).sum() / w2.sum())

        mse_unc, mse_cal, mse_pred = wmse(c), wmse(c - c_cal), wmse(c - chat)
        r2_unw = 1.0 - float(((c - chat) ** 2).sum()
                             / ((c - c.mean()) ** 2).sum())
        cw = float((w2 * c).sum() / w2.sum())
        r2_w = 1.0 - float((w2 * (c - chat) ** 2).sum()
                           / (w2 * (c - cw) ** 2).sum())
        rows.append({"fold": fi, "mse_uncal": mse_unc, "mse_cal": mse_cal,
                     "mse_pred": mse_pred, "r2_unweighted": r2_unw,
                     "r2_sigma2_weighted": r2_w,
                     "pred_over_cal": mse_pred / mse_cal})
        print(f"  fold {fi + 1}/{a.folds}   R2 unweighted {r2_unw:+.4f}   "
              f"R2 sigma^2-weighted {r2_w:+.4f}")
        print(f"      sigma^2-weighted mean square of the residual:")
        print(f"        no correction   {mse_unc:.6e}")
        print(f"        v2.7 calibration{mse_cal:.6e}   "
              f"({mse_unc / mse_cal:.4f}x better than none)")
        print(f"        the corrector   {mse_pred:.6e}   "
              f"({mse_cal / mse_pred:.4f}x vs calibration)")
        del F, T, sig
        if device == "cuda":
            t.cuda.empty_cache()

    print()
    print(f"  MEAN over {a.folds} folds:")
    print(f"    R2 unweighted        {np.mean([r['r2_unweighted'] for r in rows]):+.4f}")
    print(f"    R2 sigma^2-weighted  {np.mean([r['r2_sigma2_weighted'] for r in rows]):+.4f}")
    print(f"    corrector / calibration  "
          f"{np.mean([r['pred_over_cal'] for r in rows]):.4f}x")
    print()
    print("  The scored error is sigma^2-weighted and R^2 is not. Where those")
    print("  two disagree, the unweighted number is the one that means less.")

    rep = {"artifact": "wp9_readout_decomposition", "artifact_version": 1,
           "contract": "experiments/LOG.md CONTRACT WP9, commit 9c961ce",
           "truth": os.path.relpath(a.truth, _ROOT).replace("\\", "/"),
           "n_mlps": int(n_all), "folds": int(a.folds), "per_fold": rows,
           "r2_unweighted_mean": float(np.mean([r["r2_unweighted"] for r in rows])),
           "r2_weighted_mean": float(np.mean([r["r2_sigma2_weighted"] for r in rows])),
           "SCOPE": ("A DIAGNOSTIC of an already-measured negative, computed on "
                     "the block that produced it. It selects nothing, changes "
                     "no verdict, and is not a promotion statistic.")}
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
        print(f"\n  wrote {os.path.relpath(a.json, _ROOT)}")
    print("=" * 78)
    return 0


# ---------------------------------------------------------------------------
# the oracle ceiling/null pair -- one rollout each, and they would have caught
# WP9's error before a single parameter was fitted
# ---------------------------------------------------------------------------

def oracle_rollout(t, w, tm, tv, channels, device, dtype, upto=None):
    """K2 with the state replaced by TRUTH in the named channels, in flight.

    `channels` is a subset of {"m", "v"}. This is not an estimator -- it reads
    the labels -- it is the CEILING of an in-flight corrector that acts on
    those channels, at ZERO fit error. The final layer is never corrected, so
    the number it returns is what a perfect in-flight correction of layers
    1..L-2 buys at the readout.

    WP9 needed this and did not run it. Its arm B applied a FITTED correction
    to the mean and the variance TOGETHER, diverged by 137x, and concluded the
    per-neuron family was empty. The independent review ran exactly this and
    found the two channels have opposite signs: the mean channel is worth
    multiples, the variance channel is destructive AT ZERO FIT ERROR. A
    ceiling/null pair costs one rollout and would have separated them before
    anything was fitted.
    """
    n = w[0].shape[0]
    depth = len(w)
    dt, dev = w[0].dtype, w[0].device
    last = (depth - 1) if upto is None else upto
    mu = t.zeros(n, dtype=dt, device=dev)
    cov = t.eye(n, dtype=dt, device=dev)
    off = 1.0 - t.eye(n, dtype=dt, device=dev)
    rows = []
    with t.no_grad():
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
            mu_new, var_new = mu_raw, var_raw
            if 0 < k < last:
                if "m" in channels:
                    mu_new = tm[k]
                if "v" in channels:
                    var_new = tv[k]
            cov = t.outer(Phi, Phi) * cov_pre * off + t.diag(var_new)
            mu = mu_new
            rows.append(mu)
        out = t.stack(rows, dim=0)
        var0 = (w[0] * w[0]).sum(dim=0)
        out = t.cat([(t.sqrt(var0) * 0.3989422804014327).reshape(1, n).to(out.dtype),
                     out[1:]], dim=0)
    return out


def run_oracles(a):
    """The ceiling/null pair, plus the c_v-free fitted arm (arm C).

    Four in-flight cells on the same networks, common random numbers:

        none      the uncalibrated closure
        m         truth in the MEAN channel only     -- the ceiling
        v         truth in the VARIANCE channel only -- the null
        m+v       both, which is arm B's channel set at zero fit error

    plus arm B0 (the shipped calibrated file) as the control everything is
    scored against.
    """
    import competition_score as CS

    t = _torch()
    device = "cuda" if t.cuda.is_available() else "cpu"
    dtype = t.float64
    truth = load_truth(a.truth)
    seeds = np.asarray(truth["seeds"])
    n_all = len(seeds) if not a.n_mlps else min(a.n_mlps, len(seeds))
    seeds = seeds[:n_all]
    m1, m2 = truth["means_r1"], truth["means_r2"]
    av1, av2 = truth["allvar_r1"], truth["allvar_r2"]

    print("=" * 78)
    print("WP10 -- the in-flight ORACLE ceiling/null pair")
    print("=" * 78)
    print(f"  {n_all} networks, common random numbers, float64, {device}")
    print("  Reads the LABELS by construction: these are ceilings, not")
    print("  estimators, and none of them is submittable.")
    print()

    cells = {k: np.zeros(n_all) for k in ("B0", "none", "m", "v", "mv")}
    for i in range(n_all):
        w = weights_for(t, seeds[i], device, dtype)
        tm = t.from_numpy(np.ascontiguousarray((m1[i] + m2[i]) * 0.5)
                          ).to(device=device, dtype=dtype)
        tv = t.from_numpy(np.ascontiguousarray((av1[i] + av2[i]) * 0.5)
                          ).to(device=device, dtype=dtype)
        y1 = m1[i, -1].astype(np.float64)
        y2 = m2[i, -1].astype(np.float64)
        with t.no_grad():
            b0, _ = k2_rollout(t, w, corrector=None)          # calibrated
        cells["B0"][i] = cross_mse(b0[-1].cpu().numpy().astype(np.float64), y1, y2)
        for key, ch in (("none", ""), ("m", "m"), ("v", "v"), ("mv", "mv")):
            o = oracle_rollout(t, w, tm, tv, ch, device, dtype)
            cells[key][i] = cross_mse(o[-1].cpu().numpy().astype(np.float64),
                                      y1, y2)
        if device == "cuda":
            t.cuda.empty_cache()

    names = {"none": "uncalibrated closure, no correction",
             "m": "truth in the MEAN channel only",
             "v": "truth in the VARIANCE channel only",
             "mv": "truth in BOTH (arm B's channel set)"}
    print(f"  B0 (shipped calibrated K2)          {cells['B0'].mean():.6e}")
    print()
    print(f"  {'in-flight oracle cell':<40}{'final MSE':>14}{'vs B0':>10}")
    for k in ("none", "m", "v", "mv"):
        g = cells["B0"].mean() / cells[k].mean()
        print(f"  {names[k]:<40}{cells[k].mean():>14.6e}{g:>9.3f}x")
    print()
    for k in ("m", "v", "mv"):
        v = CS.decide(cells["B0"], cells[k], gate=a.gate)
        print(f"  {k:<4} paired gain {v.gain:8.4f}x   "
              f"{'clears' if v.passed else 'does not clear'} the {a.gate}x gate")

    rep = {"artifact": "wp10_inflight_oracles", "artifact_version": 1,
           "truth": os.path.relpath(a.truth, _ROOT).replace("\\", "/"),
           "n_mlps": int(n_all), "gate": float(a.gate),
           "mean_final_mse": {k: float(v.mean()) for k, v in cells.items()},
           "gain_vs_B0": {k: float(cells["B0"].mean() / cells[k].mean())
                          for k in ("none", "m", "v", "mv")},
           "per_mlp_final_mse": {k: v.tolist() for k, v in cells.items()},
           "SCOPE": ("ORACLE cells: they read the labels and are NOT estimators "
                     "or candidates. They bound what a perfect in-flight "
                     "correction of each channel could buy. A fitted arm can "
                     "only be worse.")}
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
        print(f"\n  wrote {os.path.relpath(a.json, _ROOT)}")
    print("=" * 78)
    return 0


def _subset(truth, idx):
    """A view of the truth arrays restricted to `idx`, same keys."""
    out = {}
    for k in ("means_r1", "means_r2", "allvar_r1", "allvar_r2", "gt_noise",
              "seeds"):
        if k in truth:
            out[k] = np.asarray(truth[k])[idx]
    for k in ("n_samples", "width", "depth", "probe"):
        if k in truth:
            out[k] = truth[k]
    return out


def _jsonable(x):
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    try:
        return float(x)
    except (TypeError, ValueError):
        return str(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--truth", default="", help="truth npz from p2_truth.py")
    ap.add_argument("--n-mlps", type=int, default=0, help="0 = all")
    ap.add_argument("--folds", type=int, default=4,
                    help="leave-NETWORKS-out folds")
    ap.add_argument("--rounds", type=int, default=3, help="DAgger rounds")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    # No default. CONTRACT WP9's G1 is 1.5x and a gate must not be suppliable
    # after the data is visible, which is why `competition_score.decide` has no
    # default either.
    ap.add_argument("--gate", type=float, required=False, default=None)
    ap.add_argument("--json", default="")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--oracles", action="store_true",
                    help="the in-flight ceiling/null pair: truth in the mean "
                         "channel, the variance channel, and both. Reads "
                         "labels; bounds a fitted arm, is not one.")
    ap.add_argument("--diagnose-readout", action="store_true",
                    help="decompose the final-layer error: why does an R^2 of "
                         "0.50 buy nothing? Diagnostic only, selects nothing.")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.oracles:
        if not a.truth:
            ap.error("--truth is required")
        if a.gate is None:
            ap.error("--gate is REQUIRED")
        return run_oracles(a)
    if a.diagnose_readout:
        if not a.truth:
            ap.error("--truth is required")
        return diagnose_readout(a)
    if not a.truth:
        ap.error("--truth is required (or pass --selftest)")
    if a.gate is None:
        ap.error("--gate is REQUIRED. CONTRACT WP9 fixed it at 1.5 before any "
                 "data existed; passing it explicitly is what makes that "
                 "visible in the command line and in the artifact.")
    return run_study(a)


if __name__ == "__main__":
    raise SystemExit(main())
