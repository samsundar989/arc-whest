"""D2 — the O4 oracle: how much is a PERFECT κ₄, carried IN FLIGHT, worth?

Contract: `experiments/LOG.md`, *CONTRACT (Rule 1): SLICE 3 STEP 1* (2026-09-08),
committed at `432f7fd` **before this file existed**. Handoff:
`docs/STRATEGIST_RESPONSE_2026-09-08.md` §5 step 1, arm D2.

## The question, and why the existing oracle cannot answer it

`local_sim/p2_cumulant_oracle.py` (2026-08-21) measured `O3 + TRUE κ₄ = 0.46%`.
That number is quoted in `CURRENT_STATE.md` as "the ceiling on SSC's readout" and
it is correct **for what it measured**: arms O0–O3 are *readout* corrections —
Edgeworth terms applied to the RETURNED array — on a **K2** closure's state.
Nothing in that module propagates κ₄ and nothing in it runs a K3 chain.

The `94.61%` in the same table is the propagated-STATE error, and that is where
the money is. This module asks the question that bounds it: **if the chain's κ₄
were exactly right at every layer, how much of that state error goes away?**

## Where truth is injected, and why THAT point

At the **output of the nonlinear step**, i.e. the post-ReLU activation's κ₄,
replacing `K_out[4]` before the next `linear_kprop` consumes it.

That point is not a convenience — it is the only frame in which upstream's own
projection convention applies verbatim:

    kprop_harmonic.linear_kprop:308
        assert K_d.has_identity_metric(), "linear_kprop expects identity metric
                                           on input HTensors"

So between the nonlinearity and the next linear step the metric is **exactly the
identity**, which is the frame `harmonic.proj_geq_r` builds its cores in. Inject
one step later — at the *input* to the nonlinearity — and the metric is
`W g W^T`, the core lives in a different frame, and a silently mis-framed core
would still run, still produce finite numbers, and still look like a measurement.
This repository has already paid for one wrong-convention flow: the weight
orientation error that read `7,123x` worse at layer 1 (MOONSHOT G0).

## The identity that makes this computable at n = 1024

The r=1 harmonic core of a symmetric 4-tensor κ is, by `proj_geq_r`,

    core = a1 * L(κ)  +  a2 * R(L²(κ))          a1, a2 from upstream's proj_coef
    L(κ)_ij = 12 * T_ij         (harmonic.lap: d(d-1) = 12 at d = 4)
    L²(κ)   = 24 * tr(T)        (harmonic.lap: d(d-1) = 2  at d = 2)
    R(s)    = s * I             (harmonic.rad of a scalar, identity metric)

    ==>  core = 12*a1 * T  +  24*a2 * tr(T) * I

where `T_ij = Σ_k κ₄(i,j,k,k)` is the **trace matrix**, the only part of κ₄ the
r=1 representation can hold. And T needs no `n⁴` tensor — for centred `y`:

    T = E[y_i y_j |y|²]  -  C_ij tr(C)  -  2 (C²)_ij

three `(n, n)` objects. That identity is what makes D2 possible at all at
`n = 1024`, and `--validate-small` checks it against upstream's own
`proj_geq_r` on a densely materialised κ₄ at small `n` rather than trusting the
derivation above.

## The ladder, and what it actually brackets

All four arms run in `kind=AUGMENT`, so they differ **only** in how much of the
true T is used. Running variant (a) as `kind=SIMPLE` instead would have confounded
the representation with the information, which is the mistake this ladder exists
to avoid.

    control  the chain's own T                          (== D1's AUGMENT arm)
    (a)      T_chain, retraced to tr(T_true)            only the SCALAR is true
    (b)      T_chain with diag(T) set true              the trace matrix's DIAGONAL
    (c)      T_true                                     the whole trace matrix

**Rung (b) is NOT "per-neuron κ₄".** `diag(T)_i = Σ_k κ₄(i,i,k,k)` is a
per-neuron *trace*, not `κ₄(i,i,i,i)`. Measured against a dense κ₄ at small n,
the per-neuron `κ₄(i,i,i,i)` actually represented after rung (b) is `0.243 /
0.132 / 0.060` of truth at `n = 16 / 32 / 64` — a `~1/n` law that reaches
**≈0.4% at n = 1024**. The strategist's D2(b) / §3b "per-neuron diagonal κ₄ in
flight" is therefore **not measured here and cannot be**: an `r=1` HTensor has no
per-neuron κ₄ degree of freedom to set. That carrier requires leaving this
representation, which is Step 3. (Skeptic, 2026-09-08; `experiments/LOG.md`
ERRATUM of that date.)

Rung (c) sets the degree-4 state to exactly `proj_geq_r(κ₄_true, n, 1)`. Note
what that is **not**: it replaces the *core* only. The `(2,2)` diagslice that
`factor_k3.py:396` reads is then *derived* from `Sym(core ⊗ I)` rather than set
to the true `κ₄(i,i,j,j)`, and no metric is replaced. (c) is still the ceiling of
the representation; an earlier draft of the contract described it as replacing
each slice independently, which is not what the code does.

`T_chain` is recovered by inverting the map above, which is exact:

    tr(T)  = tr(core) / (12*a1 + 24*a2*n)
    T      = (core - 24*a2*tr(T)*I) / (12*a1)

so the ladder is strictly nested, and the **no-op arm** — inject `T_chain`
itself — must reproduce the control to float precision. That round trip is the
seam's own proof and it runs on every network (`--noop-tol`).

## WHAT THIS DOES NOT BRACKET — read before quoting it

The handoff says D2 "brackets every κ₄-in-flight scheme, including SSC". **It does
not, and this module refuses to claim it.** An `r=1` HTensor is `Sym(core ⊗ I)`;
it can carry κ₄'s trace matrix and nothing else. κ₄'s **traceless** part — which
is precisely what Puffi's SSC response modes and a per-neuron-diagonal carrier
are built to hold — is not representable in this state at all. So:

    D2(c) is the ceiling of the AUGMENT REPRESENTATION, not of κ₄ in general.

A `D2(c)/K3 < 1.5x` result therefore kills *the trace-matrix family* — AUGMENT
and anything equivalent to it. It does **not** kill a traceless carrier, and
using it that way would close a lane on a measurement of a different method,
which is the exact error the Skeptic retracted on 2026-08-21.

## The independent-stream rule, which is load-bearing

The oracle's κ₄ and the evaluation labels must share no state. The asset's halves
are `seed_streams(s)[1].spawn(2)`; this module draws `spawn(3)[2]`, a third child
of the same substream, and `--verify-streams` asserts that children 0 and 1 still
reproduce `p2_truth._replica_seeds` exactly — so the third is provably disjoint
rather than assumed to be. An earlier oracle that shared a stream scored
`3.6e-10` against a `7.2e-8` noise floor; the impossibility was the only tell,
and `_impossibility_check` reproduces that guard here.

Research only. Never imported by `estimator.py`.
`# seed-ledger: no block named` — reads the existing P01 DESIGN asset.

Run (GPU venv, via workerq):
    .venv-gpu\\Scripts\\python.exe -u local_sim/p2_d2_k4_oracle.py \\
        --out experiments/2026-09-08_slice3/d2_o4.json
    .venv-gpu\\Scripts\\python.exe -u local_sim/p2_d2_k4_oracle.py --validate-small
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

# Allocator hint, set BEFORE torch is imported anywhere in this process.
#
# INOPERATIVE ON THIS HOST. Torch on Windows prints "expandable_segments not
# supported on this platform" in every run here, so this is NOT why an AUGMENT
# chain fits: jobs 399/402 OOMed while the desktop held ~7 GiB of VRAM, and
# 409/411 succeeded after a reboot with the card 88% free. The difference was
# CAPACITY, not fragmentation. Corrected by the 2026-09-09 ERRATUM.
# Kept because it is correct on Linux, the deployment authority -- but it must
# never be cited as the reason a run fits.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402

# Pre-registered in the contract at 432f7fd, before any datum. Do not edit.
D2_LANE_GATE = 1.5        # D2(c)/K3 < this  -> the trace-matrix family is dead
D2_CARRIER_GATE = 1.5     # D2(c)/D2(a) >= this -> build the traceless carrier
D2_PRED_C_OVER_A = 3.0    # prediction 3


# --------------------------------------------------------------------------
# streams
# --------------------------------------------------------------------------

def oracle_seed(input_seed: int, verify: bool = True, child: int = 2) -> int:
    """A third child of the evaluator's sampling substream, disjoint from A and B.

    `SeedSequence.spawn` is deterministic and append-only, so `spawn(3)[:2]` are
    the same two children `p2_truth._replica_seeds` produces with `spawn(2)`.
    `verify` asserts exactly that, which is what makes child 2's independence a
    checked fact instead of a plausible story.
    """
    from p2_truth import seed_streams, _replica_seeds
    _w, sample_ss, _e = seed_streams(input_seed)
    assert child >= 2, "children 0 and 1 ARE the label halves; never draw them"
    kids = sample_ss.spawn(child + 1)
    to_int = lambda ss: int.from_bytes(ss.generate_state(2).tobytes(), "little")
    ids = [to_int(k) for k in kids]
    a, b, c = ids[0], ids[1], ids[child]
    if verify:
        ra, rb = _replica_seeds(input_seed)
        assert (a, b) == (ra, rb), (
            f"REFUSED: spawn(3)[:2] = {(a, b)} does not reproduce the asset's "
            f"halves {(ra, rb)}. The oracle stream's independence from the "
            f"labels is unproven, and a shared stream is how an oracle scores "
            f"below the label's own noise floor.")
    return c


# --------------------------------------------------------------------------
# truth: the kappa_4 trace matrix of every post-ReLU activation
# --------------------------------------------------------------------------

def true_trace_matrices(weights, seed64, n_samples, device, chunk, dtype=None,
                        split_halves=False, matmul_f64=False):
    """T_ij = sum_k kappa_4(i,j,k,k) for the post-ReLU activation of every layer.

    Two passes over the SAME seed, deliberately:

      pass 1  the mean mu and the covariance C
      pass 2  S_ij = E[y_i y_j |y|^2]   with y = a - mu, mu now known

    A single-pass raw-moment accumulation would need the 4th-order raw-to-central
    conversion at (n, n) level, which is a long expression whose errors are
    silent. Two passes cost one extra forward and are obviously right.

        T = S - C * tr(C) - 2 * C @ C

    `split_halves=True` additionally returns T computed on each half of the
    sample stream, so the oracle's OWN noise can be reported beside its result --
    the same discipline `gt_noise` gets.
    """
    import torch

    dtype = dtype or torch.float32
    depth = len(weights)
    n = weights[0].shape[0]
    Ws = [torch.as_tensor(np.ascontiguousarray(W), device=device, dtype=dtype)
          for W in weights]

    def _forward_chunks(seed, total):
        """Yields (layer_index, preact, postact) chunk by chunk, deterministically."""
        g = torch.Generator(device=device)
        g.manual_seed(int(seed) & ((1 << 63) - 1))
        done = 0
        while done < total:
            m = min(chunk, total - done)
            x = torch.randn(m, n, generator=g, device=device, dtype=dtype)
            acts = []
            h = x
            for W in Ws:
                z = h @ W
                h = torch.relu(z)
                acts.append(h)
            done += m
            yield m, acts

    # ---- pass 1: mean and covariance ---------------------------------------
    # MATMUL in f32, ACCUMULATE in f64. A consumer card runs f64 matmul at 1/64
    # rate, so an all-f64 accumulation would cost hours per network; and f64
    # accumulation of f32 products is where the precision actually matters,
    # because T is an O(n) cancellation (S and C*tr(C) are both O(n*sigma^4)
    # while T is O(sigma^4)). TF32 is disabled for the same reason -- its 10-bit
    # mantissa would leave nothing after a 1024-fold cancellation.
    _tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        s1 = [torch.zeros(n, device=device, dtype=torch.float64) for _ in range(depth)]
        s2 = [torch.zeros(n, n, device=device, dtype=torch.float64) for _ in range(depth)]
        N = 0
        for m, acts in _forward_chunks(seed64, n_samples):
            N += m
            for l, a in enumerate(acts):
                s1[l] += a.sum(dim=0, dtype=torch.float64)
                aa = a.to(torch.float64) if matmul_f64 else a
                s2[l] += (aa.T @ aa).to(torch.float64)
        mu = [s / N for s in s1]
        C = [s2[l] / N - torch.outer(mu[l], mu[l]) for l in range(depth)]

        # ---- pass 2: S_ij = E[y_i y_j |y|^2], optionally per half ----------
        n_halves = 2 if split_halves else 1
        S_h = [[torch.zeros(n, n, device=device, dtype=torch.float64)
                for _ in range(depth)] for _ in range(n_halves)]
        N_h = [0] * n_halves
        mu32 = [m_.to(torch.float64 if matmul_f64 else dtype) for m_ in mu]
        seen = 0
        for m, acts in _forward_chunks(seed64, n_samples):
            h_idx = 0 if (not split_halves or seen < n_samples // 2) else 1
            seen += m
            N_h[h_idx] += m
            for l, a in enumerate(acts):
                y = (a.to(torch.float64) if matmul_f64 else a) - mu32[l]
                r2 = (y * y).sum(dim=1)             # |y|^2 per sample
                S_h[h_idx][l] += ((y * r2[:, None]).T @ y).to(torch.float64)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = _tf32

    def _T(S_list, Nn):
        """T is symmetric by construction; f32 matmuls make it only nearly so.

        Symmetrising here rather than letting upstream's HTensor do it keeps its
        "not symmetric, symmetrizing" warning meaningful -- if it ever fires
        again it means a real asymmetry, not f32 noise.
        """
        out = []
        for l in range(depth):
            S = S_list[l] / Nn
            Cl = C[l]
            T = S - Cl * torch.trace(Cl) - 2.0 * (Cl @ Cl)
            out.append(0.5 * (T + T.T))
        return out

    if split_halves:
        T_a = _T(S_h[0], N_h[0])
        T_b = _T(S_h[1], N_h[1])
        T_full = [(T_a[l] * N_h[0] + T_b[l] * N_h[1]) / N for l in range(depth)]
        return T_full, (T_a, T_b)
    return _T(S_h[0], N_h[0]), None


# --------------------------------------------------------------------------
# the core <-> T map, in upstream's own coefficients
# --------------------------------------------------------------------------

def _proj_constants(n, device, dtype):
    """(c1, c2) with  core = c1 * T + c2 * tr(T) * I, read from upstream.

    Derived from `harmonic.proj_geq_r(A, n, r_out=1)` at d = 4:
        core = a1 * L(A) + a2 * R(L^2(A)),  L(A) = 12 T,  L^2(A) = 24 tr(T),
        R(s) = s * I.
    Nothing here re-derives proj_coef; it is called.
    """
    import torch
    from mlp_kprop.harmonic import proj_coef
    P = sum(proj_coef(n, 4, r) for r in range(1, 3))   # r_out = 1, geq
    a1, a2 = float(P[1]), float(P[2])
    c1, c2 = 12.0 * a1, 24.0 * a2
    assert abs(c1) > 0, f"REFUSED: c1 = 0, the core<->T map is not invertible (n={n})"
    return c1, c2


def core_from_T(T, c1, c2):
    import torch
    n = T.shape[0]
    I = torch.eye(n, device=T.device, dtype=T.dtype)
    return c1 * T + c2 * float(torch.trace(T).detach()) * I


def T_from_core(core, c1, c2):
    """Exact inverse of `core_from_T`. tr(core) = tr(T)(c1 + c2*n)."""
    import torch
    n = core.shape[0]
    denom = c1 + c2 * n
    assert abs(denom) > 0, "REFUSED: core->T map is singular"
    trT = float(torch.trace(core).detach()) / denom
    I = torch.eye(n, device=core.device, dtype=core.dtype)
    return (core - c2 * trT * I) / c1


def mix_T(T_chain, T_true, variant):
    """The nested ladder. Each rung uses strictly more of T_true than the last.

    Rung "b" sets the DIAGONAL OF THE TRACE MATRIX, `diag(T)_i = sum_k
    kappa_4(i,i,k,k)`. It is NOT `kappa_4(i,i,i,i)` and must not be described as
    per-neuron kappa_4 -- see the module docstring.
    """
    import torch
    n = T_chain.shape[0]
    if variant == "scalar":
        # ARM A5 (2026-09-09): keep kappa_4 PRESENT and SELF-CONSISTENT but strip
        # the (n,n) structure -- carry only its pure-radial part, which is exactly
        # what kind=SIMPLE carries. Uses the CHAIN's own T, so unlike variant
        # "zero" it substitutes no wrong value; it removes structure only.
        return (float(torch.trace(T_chain).detach()) / n) * torch.eye(
            n, device=T_chain.device, dtype=T_chain.dtype)
    if variant == "zero":
        # ARM A3 (2026-09-09 contract): remove every kappa_4 INFORMATION surface
        # while keeping the raised diagram budget, to attribute D1's 2.094x.
        # T_true is not consulted and may be None.
        return torch.zeros_like(T_chain)
    if variant == "noop":
        return T_chain
    if variant == "c":
        return T_true
    if variant == "a":
        # keep the chain's traceless part; make only the SCALAR true
        d = (float(torch.trace(T_true).detach()) - float(torch.trace(T_chain).detach())) / n
        return T_chain + d * torch.eye(n, device=T_chain.device, dtype=T_chain.dtype)
    if variant == "b":
        out = T_chain.clone()
        out.diagonal().copy_(T_true.diagonal())
        return out
    raise ValueError(variant)


# --------------------------------------------------------------------------
# the injection seam
# --------------------------------------------------------------------------

def kprop_means_injected(weights, T_true_by_layer, variant, *, device, dtype,
                         record=None):
    """AUGMENT K3 with degree-4 state overridden at every nonlinearity's OUTPUT.

    Upstream is NOT modified. `kprop_harmonic.nonlin_kprop` imports its factored
    target inside the function body, so rebinding the module attribute is picked
    up at call time and restored in a `finally`.
    """
    import torch
    import p2_kprop_bridge as bridge
    from mlp_kprop import factor_k3 as F
    from mlp_kprop.kprop_harmonic import AUGMENT, mlp_kprop
    from mlp_kprop.harmonic import HTensor

    real = F.factored_nonlin_kprop_k3
    state = {"layer": 0}

    def patched(*args, **kw):
        out = real(*args, **kw)
        l = state["layer"]
        state["layer"] += 1
        if 4 not in out:
            return out
        if T_true_by_layer is not None and l >= len(T_true_by_layer):
            return out
        h = out[4]
        assert isinstance(h, HTensor) and h.r == 1, (
            f"REFUSED: expected an r=1 degree-4 HTensor at the nonlinearity's "
            f"output, got r={getattr(h, 'r', None)}. The injection frame "
            f"assumption does not hold and the core would be mis-framed.")
        assert h.has_identity_metric(), (
            "REFUSED: degree-4 state at the nonlinearity's output does not have "
            "an identity metric. proj_geq_r's core convention does not apply and "
            "the injected core would be in the wrong frame.")
        n = h.n
        c1, c2 = _proj_constants(n, h.core.device, h.core.dtype)
        T_chain = T_from_core(h.core.to(torch.float64), c1, c2)
        T_l = (None if T_true_by_layer is None
               else T_true_by_layer[l].to(torch.float64))
        T_new = mix_T(T_chain, T_l, variant)
        if record is not None and T_l is not None:
            record.append({
                "layer": l,
                "tr_T_chain": float(torch.trace(T_chain).detach()),
                "tr_T_true": float(torch.trace(T_l).detach()),
                "rel_fro_err_T": float(
                    torch.linalg.norm(T_chain - T_l) / torch.linalg.norm(T_l)),
            })
        out[4] = HTensor(core_from_T(T_new, c1, c2).to(h.core.dtype), r=1, n=n)
        return out

    F.factored_nonlin_kprop_k3 = patched
    try:
        with torch.no_grad():
            depth, width = len(weights), weights[0].shape[0]
            mlp = bridge.build_mlp(weights, device=device, dtype=dtype)
            K_in = {1: torch.zeros(width, device=device, dtype=dtype),
                    2: torch.eye(width, device=device, dtype=dtype)}
            out = mlp_kprop(mlp, K_in, k_max=3, output_all=True, kind=AUGMENT,
                            use_avg_metric=True, factor=True,
                            up_to_layer=f"act{depth - 1}", output_d_max=1)
            return np.stack([bridge._degree1(out[f"act{l}"][1]) for l in range(depth)],
                            axis=0)
    finally:
        F.factored_nonlin_kprop_k3 = real


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def xmse(pred, A, B):
    return float(np.mean((pred - A) * (pred - B)))


def studentised_t(d, alpha=0.05):
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    try:
        from scipy import stats
        crit = float(stats.t.ppf(1 - alpha / 2, n - 1))
    except Exception:
        crit = 1.96 if n > 60 else 2.064
    return m, se, (m - crit * se, m + crit * se)


def _impossibility_check(rows, gt_noise_per_half):
    """An arm cannot beat the label's own noise. If one does, a stream is shared."""
    bad = []
    for r in rows:
        for k, v in r.items():
            if k.startswith("xmse_") and v is not None and v < 0.25 * gt_noise_per_half:
                bad.append((r["seed"], k, v))
    return bad


# --------------------------------------------------------------------------
# small-n validation of the core <-> T identity, against upstream itself
# --------------------------------------------------------------------------

def validate_small(n=24, depth=2, n_samples=1 << 19, device="cuda", seed=12345):
    """Materialise kappa_4 densely at small n and check the T shortcut EXACTLY.

    This is the check that makes the (n, n) shortcut usable at n = 1024: it
    compares `core_from_T(T)` against upstream's own
    `proj_geq_r(dense_kappa4, n, r_out=1).core`, with no shared code between the
    two paths except `proj_coef`.
    """
    import torch
    from mlp_kprop.harmonic import proj_geq_r

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    W = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32)
         for _ in range(depth)]

    g = torch.Generator(device=device); g.manual_seed(seed)
    Ws = [torch.as_tensor(w, device=device, dtype=torch.float64) for w in W]
    x = torch.randn(n_samples, n, generator=g, device=device, dtype=torch.float64)
    h = x
    acts = []
    for Wt in Ws:
        h = torch.relu(h @ Wt)
        acts.append(h)

    out = []
    for l, a in enumerate(acts):
        y = a - a.mean(dim=0)
        C = (y.T @ y) / y.shape[0]
        # dense kappa_4 (n^4) -- only affordable because n is small, and
        # accumulated in sample chunks so the (s, n, n, n) intermediate that a
        # single einsum would build never exists.
        M4 = torch.zeros(n, n, n, n, device=y.device, dtype=y.dtype)
        for lo in range(0, y.shape[0], 4096):
            yc = y[lo:lo + 4096]
            M4 += torch.einsum("si,sj,sk,sl->ijkl", yc, yc, yc, yc)
        M4 /= y.shape[0]
        K4 = (M4
              - torch.einsum("ij,kl->ijkl", C, C)
              - torch.einsum("ik,jl->ijkl", C, C)
              - torch.einsum("il,jk->ijkl", C, C))
        T_dense = torch.einsum("ijkk->ij", K4)
        r2 = (y * y).sum(dim=1)
        T_fast = ((y * r2[:, None]).T @ y) / y.shape[0] - C * torch.trace(C) - 2 * (C @ C)

        c1, c2 = _proj_constants(n, C.device, C.dtype)
        core_up = proj_geq_r(K4, n=n, r_out=1).core
        core_ours = core_from_T(T_dense, c1, c2)
        T_round = T_from_core(core_ours, c1, c2)

        def rel(a_, b_):
            return float(torch.linalg.norm(a_ - b_) / torch.linalg.norm(b_))

        out.append({
            "layer": l,
            "rel_T_fast_vs_dense": rel(T_fast, T_dense),
            "rel_core_ours_vs_upstream": rel(core_ours, core_up),
            "rel_roundtrip_T": rel(T_round, T_dense),
        })
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

# Pre-registered by the 2026-09-08 ERRATUM, before any 24-net datum.
TWO_STREAM_REFUSAL = 0.2   # spread must be <= this FRACTION OF THE (a)->(c) GAP


def two_stream_check(seeds, man, a, torch, bridge, canonical_mint):
    """REFUSAL gate: is the oracle quiet enough for the (a)->(c) gap to be real?

    The oracle's own noise adds `E||delta||^2` to `xmse_c` -- a POSITIVE bias, so
    it pushes toward the pre-registered kill. Recording `oracle_rel_noise_T` and
    never gating on it (the earlier draft) means a kill could be an artefact of
    too few samples.

    The bar is deliberately the `(a) -> (c)` gap and NOT the larger
    (c)-vs-control gap: `(a) -> (c)` is the contrast the whole slice turns on and
    it is the smaller of the two, so gating on the larger one would pass an
    oracle too noisy for the number actually being quoted.
    """
    from mlp_kprop.kprop_harmonic import AUGMENT
    s0 = seeds[0]
    w, _e, _ss = canonical_mint(man["width"], man["depth"], s0)
    out = []
    for log2n in (a.two_stream_log2 or [20, 22, 24]):
        ns = 1 << log2n
        preds, noises = {}, {}
        for child in (2, 3):
            T, halves = true_trace_matrices(
                w, oracle_seed(s0, verify=(child == 2), child=child), ns,
                a.device, a.chunk, dtype=torch.float32, split_halves=True)
            Ta, Tb = halves
            noises[child] = float(np.mean([
                float(torch.linalg.norm(Ta[l] - Tb[l]) / 2 / torch.linalg.norm(T[l]))
                for l in range(len(T))]))
            preds[child] = {v: kprop_means_injected(w, T, v, device=a.device,
                                                    dtype=torch.float32)[-1]
                            for v in ("a", "c")}
        gap = float(np.linalg.norm(preds[2]["c"] - preds[2]["a"]))
        spread = float(np.linalg.norm(preds[2]["c"] - preds[3]["c"]))
        rec = {"log2_samples": log2n, "samples": ns,
               "oracle_rel_noise_T_stream2": noises[2],
               "oracle_rel_noise_T_stream3": noises[3],
               "a_to_c_gap": gap, "two_stream_spread_on_c": spread,
               "spread_over_gap": (spread / gap) if gap else float("inf"),
               "passes": bool(gap and spread / gap <= TWO_STREAM_REFUSAL)}
        out.append(rec)
        print(f"    2^{log2n:<2} ({ns:>10,})  rel_noise_T {noises[2]:.3f}   "
              f"(a)->(c) gap {gap:.3e}   two-stream spread {spread:.3e}   "
              f"ratio {rec['spread_over_gap']:.3f}   "
              f"{'PASS' if rec['passes'] else 'REFUSE'}")
    return out


def _progress(frac):
    p = os.environ.get("WORKERQ_PROGRESS")
    if not p:
        return
    try:
        with open(p, "w") as fh:
            fh.write(str(min(max(frac, 0.0), 1.0)))
    except Exception:
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="experiments/2026-08-26_p2_gt")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log2-oracle-samples", type=int, default=22)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--noop-tol", type=float, default=1e-9,
                    help="max relative |noop arm - control| before HALT")
    ap.add_argument("--d1-xmse", type=float, default=None,
                    help="D1's measured AUGMENT xmse. REQUIRED to evaluate the "
                         "lane gate, which is conjunctive: (c) failing 1.5x over "
                         "SIMPLE *and* D1 >= 3.5e-8. Without it the gate reports "
                         "None rather than half of itself.")
    ap.add_argument("--validate-small", action="store_true")
    ap.add_argument("--precision-check", action="store_true",
                    help="f32-matmul vs f64-matmul T at the REAL width, where the "
                         "O(n) cancellation actually bites")
    ap.add_argument("--precision-samples", type=int, default=22)
    ap.add_argument("--two-stream-check", action="store_true",
                    help="the pre-registered REFUSAL: two independent oracle "
                         "streams on one net must agree to <=0.2 of the (a)->(c) gap")
    ap.add_argument("--two-stream-log2", type=int, nargs="*", default=None)
    ap.add_argument("--small-n", type=int, default=24)
    a = ap.parse_args(argv)

    import torch
    import p2_kprop_bridge as bridge
    from mint_provenance import canonical_mint

    v = bridge.verify_upstream(bridge._src())
    if not v["pass"]:
        raise SystemExit(f"REFUSED: upstream is not the pinned rev + patch: {v}")

    if a.validate_small:
        print("=" * 78)
        print("D2 instrument check -- the core <-> T identity vs upstream's own proj_geq_r")
        print("=" * 78)
        rep = validate_small(n=a.small_n, device=a.device)
        worst = 0.0
        for r in rep:
            print(f"  layer {r['layer']}   T_fast vs dense {r['rel_T_fast_vs_dense']:.3e}   "
                  f"core ours vs upstream {r['rel_core_ours_vs_upstream']:.3e}   "
                  f"roundtrip {r['rel_roundtrip_T']:.3e}")
            worst = max(worst, r["rel_core_ours_vs_upstream"], r["rel_roundtrip_T"])
        ok = worst < 1e-9
        print(f"\n  worst (excluding MC noise in T_fast): {worst:.3e} -> "
              f"{'PASS' if ok else 'FAIL -- the shortcut is NOT upstream-equivalent'}")
        print("  note: all three are ALGEBRAIC identities on the same samples, not")
        print("        noise comparisons -- T_fast is an exact rearrangement of the")
        print("        same moments the dense kappa_4 is built from, so float64")
        print("        roundoff is the only expected residue.")
        return 0 if ok else 1

    if a.two_stream_check:
        from mint_provenance import canonical_mint as _cm2
        gt0 = os.path.join(_ROOT, a.gt_dir)
        man0 = json.load(open(os.path.join(gt0, "gt_manifest.json"), encoding="utf-8"))
        sds = [int(x) for x in np.load(os.path.join(gt0, "gt_halfA.npz"))["seeds"]]
        print("=" * 78)
        print("D2 PRE-REGISTERED REFUSAL -- two independent oracle streams, one net")
        print("=" * 78)
        print(f"  bar: two-stream spread on (c) must be <= {TWO_STREAM_REFUSAL} "
              f"of the (a)->(c) gap")
        print("  (the (a)->(c) gap, not the larger (c)-vs-control gap: (a)->(c) is")
        print("   the contrast the slice turns on and is the smaller of the two)")
        print()
        recs = two_stream_check(sds, man0, a, torch, bridge, _cm2)
        ok = [r for r in recs if r["passes"]]
        print()
        if ok:
            print(f"  -> PASS at 2^{min(r['log2_samples'] for r in ok)} samples; "
                  f"use that (or more) for the 24-net run")
        else:
            print("  -> REFUSED at every sample count tried. The 24-net D2 run must")
            print("     NOT be submitted: the oracle's own noise is comparable to the")
            print("     (a)->(c) gap, and it biases xmse_c UPWARD, i.e. toward the kill.")
        if a.out:
            os.makedirs(os.path.dirname(os.path.join(_ROOT, a.out)), exist_ok=True)
            with open(os.path.join(_ROOT, a.out), "w", encoding="utf-8") as fh:
                json.dump({"check": "D2 two-stream oracle-noise refusal",
                           "refusal_bar_fraction_of_a_to_c_gap": TWO_STREAM_REFUSAL,
                           "contract": "experiments/LOG.md ERRATUM 2026-09-08",
                           "rows": recs}, fh, indent=2)
        return 0 if ok else 1

    if a.precision_check:
        from mint_provenance import canonical_mint as _cm
        print("=" * 78)
        print("D2 precision check -- f32-matmul vs f64-matmul T at width 1024")
        print("=" * 78)
        print("  T = S - C tr(C) - 2 C^2 is an O(n) cancellation, so the question")
        print("  is not 'is f32 accurate' but 'does f32 survive a 1024-fold one'.")
        w, _e, _ss = _cm(1024, 16, 40821001144)
        ns = 1 << a.precision_samples
        osd = oracle_seed(40821001144, verify=True)
        t32, h32 = true_trace_matrices(w, osd, ns, a.device, a.chunk,
                                       dtype=torch.float32, matmul_f64=False,
                                       split_halves=True)
        t64, _ = true_trace_matrices(w, osd, ns, a.device, a.chunk,
                                     dtype=torch.float32, matmul_f64=True)
        # The RIGHT bar is this oracle's OWN measured sampling noise, not
        # 1/sqrt(N). T = S - C tr(C) - 2C^2 is an O(n) cancellation and the noise
        # in S does not cancel, so the true scale is ~1.7*n/sqrt(N) -- at n=1024
        # that is a factor of ~1700 looser than 1/sqrt(N), and comparing against
        # 1/sqrt(N) would have failed a perfectly adequate f32 path and pushed
        # the whole oracle to f64 for nothing. (Skeptic, 2026-09-08.)
        Ta, Tb = h32
        worst = worst_mc = 0.0
        for l in range(len(t32)):
            rel = float(torch.linalg.norm(t32[l] - t64[l]) / torch.linalg.norm(t64[l]))
            mcl = float(torch.linalg.norm(Ta[l] - Tb[l]) / 2 / torch.linalg.norm(t32[l]))
            worst, worst_mc = max(worst, rel), max(worst_mc, mcl)
            print(f"    layer {l:>2}  |T_f32-T_f64|/|T_f64| = {rel:.3e}   "
                  f"measured MC noise = {mcl:.3e}   ratio {rel / mcl if mcl else float('nan'):.3e}")
        print("")
        print(f"  worst f32-vs-f64 {worst:.3e}   worst MEASURED oracle noise {worst_mc:.3e}"
              f"   ({ns:,} samples)")
        print(f"  (1/sqrt(N) = {1.0 / np.sqrt(ns):.3e} is NOT the bar -- see the comment above)")
        ok = worst < 0.1 * worst_mc
        print(f"  -> f32 matmul error is {'well below' if ok else 'NOT clearly below'} "
              f"a tenth of the sampling noise it sits beside, so it is "
              f"{'not the limiting error' if ok else 'worth re-examining'}")
        return 0 if ok else 1

    from mlp_kprop.kprop_harmonic import SIMPLE, AUGMENT

    gt = os.path.join(_ROOT, a.gt_dir)
    man = json.load(open(os.path.join(gt, "gt_manifest.json"), encoding="utf-8"))
    if not man.get("all_pass"):
        raise SystemExit("REFUSED: the P2-GT asset did not pass its own fitness checks.")
    zA = np.load(os.path.join(gt, "gt_halfA.npz"))
    zB = np.load(os.path.join(gt, "gt_halfB.npz"))
    seeds = [int(s) for s in zA["seeds"]]
    A_all, B_all = zA["means"], zB["means"]
    if a.limit:
        seeds, A_all, B_all = seeds[:a.limit], A_all[:a.limit], B_all[:a.limit]

    n_samp = 1 << a.log2_oracle_samples
    gt_noise = man["gt_noise_per_half_mean"]

    print("=" * 78)
    print("D2  --  O4 oracle: TRUE kappa_4 carried IN FLIGHT on the K3 chain")
    print("=" * 78)
    print(f"  asset             {a.gt_dir}  ({len(seeds)} networks)")
    print(f"  gt_noise per half {gt_noise:.4e}")
    print(f"  oracle stream     spawn(3)[2], {n_samp:,} samples, verified disjoint from A/B")
    print(f"  upstream          {v['rev'][:12]} + recorded patch, verified")
    print(f"  injection point   OUTPUT of every nonlinearity (identity-metric frame)")
    print(f"  arms              control(AUGMENT) | noop | (a) trace | (b) diagonal | (c) full T")
    print(f"  PRE-REGISTERED    lane gate D2(c)/K3 < {D2_LANE_GATE}   "
          f"carrier gate D2(c)/D2(a) >= {D2_CARRIER_GATE}   pred (c)/(a) >= {D2_PRED_C_OVER_A}")
    print()

    rows, t0 = [], time.perf_counter()
    for i, s in enumerate(seeds):
        w, _e, _ss = canonical_mint(man["width"], man["depth"], s)
        Af, Bf = A_all[i, -1, :], B_all[i, -1, :]
        row = {"seed": s}

        osd = oracle_seed(s, verify=(i == 0))
        tt = time.perf_counter()
        T_true, (T_h1, T_h2) = true_trace_matrices(
            w, osd, n_samp, a.device, a.chunk, dtype=torch.float32, split_halves=True)
        row["oracle_seconds"] = time.perf_counter() - tt
        # the oracle's OWN noise, reported beside its result
        row["oracle_rel_noise_T"] = float(np.mean([
            float(torch.linalg.norm(T_h1[l] - T_h2[l]) / 2
                  / torch.linalg.norm(T_true[l])) for l in range(len(T_true))]))

        with torch.no_grad():
            row["xmse_simple"] = xmse(
                bridge.kprop_means(w, k_max=3, factor=True, kind=SIMPLE,
                                   device=a.device, dtype=torch.float32)[-1], Af, Bf)
            row["xmse_control"] = xmse(
                bridge.kprop_means(w, k_max=3, factor=True, kind=AUGMENT,
                                   device=a.device, dtype=torch.float32)[-1], Af, Bf)

        rec = []
        for variant in ("noop", "a", "b", "c"):
            m = kprop_means_injected(w, T_true, variant, device=a.device,
                                     dtype=torch.float32,
                                     record=rec if variant == "c" else None)
            row[f"xmse_{variant}"] = xmse(m[-1], Af, Bf)
        row["chain_vs_true_T"] = rec

        # THE SEAM'S OWN PROOF: injecting the chain's own T must be a no-op.
        den = abs(row["xmse_control"]) or 1.0
        row["noop_rel_dev"] = abs(row["xmse_noop"] - row["xmse_control"]) / den
        rows.append(row)

        el = time.perf_counter() - t0
        print(f"    {i + 1:>3}/{len(seeds)}  seed {s}   "
              f"SIMPLE {row['xmse_simple']:.3e}  AUG {row['xmse_control']:.3e}  "
              f"a {row['xmse_a']:.3e}  b {row['xmse_b']:.3e}  c {row['xmse_c']:.3e}   "
              f"[noop dev {row['noop_rel_dev']:.1e}, oracle noise {row['oracle_rel_noise_T']:.1e}, "
              f"eta {el / (i + 1) * (len(seeds) - i - 1) / 60:.0f}m]")
        _progress((i + 1) / len(seeds))

        if row["noop_rel_dev"] > a.noop_tol:
            _dump(a, rows, man, v, seeds,
                  halted=f"the no-op injection changed the result by "
                         f"{row['noop_rel_dev']:.3e} (tol {a.noop_tol:.1e}) on seed {s}. "
                         f"The core<->T round trip is not exact in this frame, so "
                         f"every oracle number would be measuring the seam rather "
                         f"than kappa_4.")
            raise SystemExit(1)
        if a.out:
            _dump(a, rows, man, v, seeds, partial=True)

    return _report(a, rows, man, v, seeds, gt_noise)


def _summary(rows, gt_noise, d1_xmse=None):
    if not rows:
        return {}
    g = lambda k: np.array([r[k] for r in rows])
    out = {"n": len(rows), "gt_noise_per_half": gt_noise}
    for k in ("simple", "control", "noop", "a", "b", "c"):
        m, se, ci = studentised_t(g(f"xmse_{k}"))
        out[k] = {"mean_xmse": m, "se": se, "ci95": list(ci)}
    for lo, hi, name in (("control", "a", "control_minus_a"),
                         ("control", "b", "control_minus_b"),
                         ("control", "c", "control_minus_c"),
                         ("simple", "c", "simple_minus_c"),
                         ("a", "c", "a_minus_c")):
        m, se, ci = studentised_t(g(f"xmse_{lo}") - g(f"xmse_{hi}"))
        out[name] = {"paired_diff": m, "se": se, "ci95": list(ci)}
    k3 = out["simple"]["mean_xmse"]
    # Named for what they HOLD. An earlier draft called these c_over_a / c_over_b
    # while holding a/c and b/c; the printed labels were right and the artifact
    # keys were inverted, which is the class of convention error this repository
    # has been bitten by twice.
    out["ratios_diagnostic"] = {
        "K3_over_c_CONFOUNDED": k3 / out["c"]["mean_xmse"] if out["c"]["mean_xmse"] else None,
        "a_over_c": out["a"]["mean_xmse"] / out["c"]["mean_xmse"] if out["c"]["mean_xmse"] else None,
        "b_over_c": out["b"]["mean_xmse"] / out["c"]["mean_xmse"] if out["c"]["mean_xmse"] else None,
        "control_over_c": out["control"]["mean_xmse"] / out["c"]["mean_xmse"] if out["c"]["mean_xmse"] else None,
        "_note": ("K3_over_c is CONFOUNDED: K3 is SIMPLE and (c) is AUGMENT+truth, "
                  "so it contains the representation change AND the truth injection. "
                  "a_over_c is the only ratio here that isolates truth."),
    }
    r = out["ratios_diagnostic"]
    # ---- the gates, through decide(), on PAIRED DIFFERENCES ------------------
    # docs/VALIDATION.md 2b-i: decide() is the only function in this repository
    # that returns a promotion boolean. An earlier draft read its verdicts
    # straight out of ratios_diagnostic -- a dict this module itself labels
    # diagnostic only -- while computing and then ignoring the intervals.
    from competition_score import decide

    def _verdict(control_key, cand_key, gate):
        v = decide(g(f"xmse_{control_key}"), g(f"xmse_{cand_key}"), gate=gate)
        return {"passed": bool(v.passed), "gain": float(v.gain), "gate": float(v.gate),
                "ci_paired_difference": list(v.ci),
                "ci_excludes_zero": bool(v.ci_excludes_zero),
                "geometric_gain_diagnostic": float(v.geometric_gain),
                "n": int(v.n), "n_candidate_better": int(v.n_candidate_better),
                "reasons": list(v.reasons)}

    # THE headline: (a) -> (c) is the only contrast that isolates TRUTH from the
    # SIMPLE -> AUGMENT representation change.
    out["verdict_a_to_c"] = _verdict("a", "c", D2_CARRIER_GATE)
    out["verdict_simple_to_c_CONFOUNDED"] = _verdict("simple", "c", D2_LANE_GATE)
    out["verdict_control_to_c"] = _verdict("control", "c", D2_LANE_GATE)

    out["gates"] = {
        # Conjunctive, exactly as pre-registered: BOTH, not either. d1_xmse must
        # be supplied; without D1's number this gate cannot be evaluated at all.
        "lane_dead": (None if d1_xmse is None else bool(
            (not out["verdict_simple_to_c_CONFOUNDED"]["passed"]) and d1_xmse >= 3.5e-8)),
        "lane_dead_inputs": {"d1_xmse": d1_xmse,
                             "simple_to_c_passed_1.5x": out["verdict_simple_to_c_CONFOUNDED"]["passed"],
                             "requires": "BOTH (c) failing 1.5x over SIMPLE AND D1 >= 3.5e-8"},
        "prediction_3_a_over_c_ge_3": bool(
            r["a_over_c"] is not None and r["a_over_c"] >= D2_PRED_C_OVER_A),
        # DEMOTED from a gate to a diagnostic by the 2026-09-08 ERRATUM: it is a
        # ratio measured entirely INSIDE the trace-matrix subspace, and it was
        # being used to decide whether to build a carrier for that subspace's
        # ORTHOGONAL COMPLEMENT. Neither direction of that inference follows.
        "carrier_DIAGNOSTIC_ONLY_not_a_gate": out["verdict_a_to_c"]["passed"],
    }
    out["worst_noop_rel_dev"] = float(max(r_["noop_rel_dev"] for r_ in rows))
    out["mean_oracle_rel_noise_T"] = float(np.mean([r_["oracle_rel_noise_T"] for r_ in rows]))
    out["impossible_arms"] = _impossibility_check(rows, gt_noise)
    return out


def _dump(a, rows, man, v, seeds, partial=False, halted=""):
    if not a.out:
        return
    out = os.path.join(_ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rec = {
        "slice": "D2 -- O4 oracle, true kappa_4 in flight on the K3 chain, 1024x16",
        "contract": "experiments/LOG.md, CONTRACT (Rule 1) Slice 3 step 1, commit 432f7fd",
        "asset": a.gt_dir, "n_mlps_requested": len(seeds),
        "oracle_samples": 1 << a.log2_oracle_samples,
        "oracle_stream": "seed_streams(s)[1].spawn(3)[2]; children 0,1 verified "
                         "to reproduce p2_truth._replica_seeds, so disjointness "
                         "from both label halves is checked, not assumed",
        "injection_point": "OUTPUT of every nonlinearity, where linear_kprop:308 "
                           "guarantees an identity metric and proj_geq_r's core "
                           "convention applies verbatim",
        "upstream": v, "partial": partial, "halted": halted,
        "lane_gate": D2_LANE_GATE, "carrier_gate": D2_CARRIER_GATE,
        "tier": ("DESIGN (P01), re-read. Orients priority; promotes NOTHING."),
        "scope_caveat": (
            "D2(c) is the ceiling of the AUGMENT REPRESENTATION -- an r=1 HTensor "
            "Sym(core (x) I) can carry kappa_4's TRACE MATRIX and nothing else. "
            "kappa_4's traceless part, which is exactly what an SSC-style response-mode "
            "carrier holds, is NOT representable in this state. So a small (c)/K3 "
            "kills the trace-matrix family (AUGMENT and equivalents) and does NOT "
            "kill a traceless carrier. The Sep 8 handoff's claim that D2 'brackets "
            "every kappa_4-in-flight scheme, including SSC' is NOT supported and is "
            "not made here."),
        "cost_note": "No cost claim; torch meter, not flopscope.",
        "summary": _summary(rows, man["gt_noise_per_half_mean"], a.d1_xmse),
        "rows": rows,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)
    return rec


def _report(a, rows, man, v, seeds, gt_noise) -> int:
    s = _summary(rows, gt_noise, a.d1_xmse)
    print()
    print(f"  gt_noise per half          {gt_noise:.4e}")
    for k, label in (("simple", "K3 SIMPLE (control)"), ("control", "AUGMENT (own k4)"),
                     ("noop", "noop injection"), ("a", "(a) trace only"),
                     ("b", "(b) per-neuron diag"), ("c", "(c) full trace matrix")):
        e = s[k]
        print(f"  {label:<24s}  {e['mean_xmse']:.4e}   se {e['se']:.2e}   "
              f"95% [{e['ci95'][0]:.4e}, {e['ci95'][1]:.4e}]")
    print()
    for name in ("control_minus_a", "control_minus_b", "control_minus_c",
                 "simple_minus_c", "a_minus_c"):
        e = s[name]
        print(f"  paired {name:<20s} {e['paired_diff']:+.4e}   "
              f"95% [{e['ci95'][0]:+.4e}, {e['ci95'][1]:+.4e}]")
    r = s["ratios_diagnostic"]
    print()
    print(f"  ratios (DIAGNOSTIC only)   (a)/(c) {r['a_over_c']:.3f}x   "
          f"(b)/(c) {r['b_over_c']:.3f}x   control/(c) {r['control_over_c']:.3f}x")
    print(f"                             K3/(c) {r['K3_over_c_CONFOUNDED']:.3f}x  "
          f"<- CONFOUNDED: contains the SIMPLE->AUGMENT change AND the truth")
    print(f"  seam    worst no-op deviation {s['worst_noop_rel_dev']:.2e}")
    print(f"  oracle  mean relative noise in T {s['mean_oracle_rel_noise_T']:.2e}")
    if s["impossible_arms"]:
        print(f"  *** IMPOSSIBLE: {len(s['impossible_arms'])} arm(s) scored below the "
              f"label noise floor. A sample stream is shared. Results VOID. ***")
    print()
    g = s["gates"]
    va = s["verdict_a_to_c"]
    print(f"  HEADLINE  (a) -> (c), the ONLY contrast isolating truth:")
    print(f"            gain {va['gain']:.3f}x vs gate {va['gate']:.2f}x   "
          f"paired 95% [{va['ci_paired_difference'][0]:+.3e}, {va['ci_paired_difference'][1]:+.3e}]   "
          f"decide() -> {'PASS' if va['passed'] else 'FAIL'}")
    for why in va["reasons"]:
        print(f"              - {why}")
    print()
    if g["lane_dead"] is None:
        print(f"  GATE  lane_dead: NOT EVALUABLE -- pass --d1-xmse. The gate is "
              f"CONJUNCTIVE ((c) failing {D2_LANE_GATE}x over SIMPLE AND D1 >= 3.5e-8);")
        print(f"        reporting half of it is how a printed verdict gets quoted downstream.")
    else:
        print(f"  GATE  lane_dead -> {'FIRED: the TRACE-MATRIX family is dead' if g['lane_dead'] else 'not fired'}"
              f"   inputs {g['lane_dead_inputs']}")
    print(f"  PRED 3  (a)/(c) >= {D2_PRED_C_OVER_A} -> "
          f"{'HELD' if g['prediction_3_a_over_c_ge_3'] else 'MISSED'}")
    print(f"  DIAGNOSTIC (NOT a gate)  carrier {D2_CARRIER_GATE}x -> "
          f"{'passed' if g['carrier_DIAGNOSTIC_ONLY_not_a_gate'] else 'not passed'}")
    print()
    print("  SCOPE, and it binds BOTH gates, not just the lane one:")
    print("    (c) is the ceiling of the AUGMENT REPRESENTATION, not of kappa_4. An r=1")
    print("    HTensor carries the TRACE MATRIX only -- measured, that subspace holds")
    print("    ~0.2% of kappa_4's Frobenius mass at n=1024. So a small (a)->(c) does NOT")
    print("    kill an SSC-style traceless carrier, and a large one is not evidence FOR")
    print("    one either: both are measured inside the subspace, and the carrier lives")
    print("    in its orthogonal complement. Step 3 needs its own oracle.")
    _dump(a, rows, man, v, seeds)
    if a.out:
        print(f"\n  artifact  {a.out}")
    _progress(1.0)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
