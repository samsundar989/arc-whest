"""END-TO-END proof that the dense `n³` `(2,1,1)` slice enters the AUGMENT chain
ONLY through its axis-0 sum.

Run:  .venv-gpu\\Scripts\\python.exe local_sim/test_p2_fused_reduction.py

## Why a perturbation test and not a reimplementation

Three separate results were established on 2026-09-09:

1. `K_ds[(2,1,1)] -> r=1 core` depends only on the slice's axis-0 sum
   (`L²` of a `(2,1,1)` slice is identically zero; `L¹` is `12 x sym(axis-0 sum)`).
2. 15 of the 17 vector partitions reaching `pK[(2,1,1)]` have every block
   spanning at most two axes.
3. The two that span three are rank-1 separable: `(1/6)·g[i]·A[j,k]` and
   `(1/6)·A[i,j]·g[k]`.

Each was verified on its own. **None of that proves they COMPOSE.** A
reimplementation of the fused algebra would not settle it either — it could be
self-consistently wrong, and it would be a new thing to trust rather than a test
of the existing one.

So this file asks the question directly, of the REAL chain:

> Perturb the `n³` slice in a way that **exactly preserves its axis-0 sum**.
> If the reduction claim is true, the chain's final output must not move **at
> all**.

That is a single falsifiable statement about the whole composition — `eval_part`
-> `DS_pK_to_K` -> `DS_harmonic_proj` -> the next `linear_kprop` -> fifteen more
layers -> the readout. It cannot pass by accident: the perturbation is large
(comparable to the slice itself) and touches the exact object the port intends to
never build.

## The perturbation, and why it is exactly sum-preserving

Diagonal slices are stored **zero-repeated** — entries with any two indices equal
are held at zero — so a naive `X - X.mean(0)` does not preserve the sum on the
support, which is what made the first attempt at this test on 2026-09-09 report a
false refutation. Here the perturbation is built on the support and then
explicitly re-projected:

    R      random, restricted to the zero-repeated support
    S_R    its axis-0 sum
    P      R minus, for each (j,k), S_R[j,k] spread uniformly over the VALID i

so `sum_i P[i,j,k] = 0` exactly, on the support, by construction — and the test
asserts that before it trusts the arm.

Upstream is never modified: `factor_k3.DS_pK_to_K` is rebound and restored.
CPU, seconds. `# seed-ledger: no block named`.
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import torch  # noqa: E402
import p2_kprop_bridge as bridge  # noqa: E402

bridge._src()

from mlp_kprop.kprop_harmonic import AUGMENT  # noqa: E402
import mlp_kprop.factor_k3 as F  # noqa: E402

RESULTS: list[bool] = []
PART = (2, 1, 1)


def _support_mask(n, device, dtype):
    """1 where all three indices are distinct, 0 where any two coincide."""
    idx = torch.arange(n, device=device)
    i = idx[:, None, None]
    j = idx[None, :, None]
    k = idx[None, None, :]
    return ((i != j) & (i != k) & (j != k)).to(dtype)


def sum_preserving_perturbation(X, gen):
    """A LARGE perturbation of X with an exactly zero axis-0 sum, on the support."""
    n = X.shape[0]
    m = _support_mask(n, X.device, X.dtype)
    R = torch.randn(X.shape, generator=gen, device=X.device, dtype=X.dtype) * m
    # symmetric in the two singleton axes, as a (2,1,1) slice is
    R = 0.5 * (R + R.transpose(1, 2))
    R = R * m
    cnt = m.sum(dim=0)                      # valid i per (j,k)
    corr = R.sum(dim=0) / torch.where(cnt > 0, cnt, torch.ones_like(cnt))
    P = (R - corr[None, :, :]) * m
    return P


def check(name):
    def deco(fn):
        def run(*a, **kw):
            try:
                fn(*a, **kw)
            except Exception as e:
                RESULTS.append(False)
                print(f"  FAIL  {name}\n        {type(e).__name__}: {e}")
                if not isinstance(e, AssertionError):
                    traceback.print_exc()
                return False
            RESULTS.append(True)
            print(f"  ok    {name}")
            return True
        return run
    return deco


def run_chain(weights, perturb_gen=None, record=None):
    """The AUGMENT chain, optionally perturbing pK[(2,1,1)] sum-preservingly."""
    real = F.DS_pK_to_K

    def patched(pK, *a, **kw):
        if perturb_gen is not None and 4 in pK and PART in pK[4].slices:
            X = pK[4].slices[PART]
            if X.ndim == 3:
                P = sum_preserving_perturbation(X, perturb_gen)
                if record is not None:
                    record.append({
                        "rel_size": float(P.abs().max() / max(float(X.abs().max()), 1e-300)),
                        "axis0_sum_resid": float(P.sum(dim=0).abs().max()),
                    })
                pK[4].slices[PART] = X + P
        return real(pK, *a, **kw)

    F.DS_pK_to_K = patched
    try:
        with torch.no_grad():
            return bridge.kprop_means(weights, k_max=3, factor=True, kind=AUGMENT,
                                      device="cpu", dtype=torch.float64)
    finally:
        F.DS_pK_to_K = real


def _net(n, d, seed):
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float64)
            for _ in range(d)]


@check("the perturbation is LARGE and its axis-0 sum is EXACTLY zero")
def t_perturbation_is_valid(recs):
    assert recs, "the perturbation hook never fired -- the (2,1,1) slice was never built"
    worst_sum = max(r["axis0_sum_resid"] for r in recs)
    biggest = max(r["rel_size"] for r in recs)
    print(f"          fired on {len(recs)} layers; max |axis-0 sum of P| = {worst_sum:.3e}; "
          f"largest |P|/|X| = {biggest:.2f}")
    assert worst_sum < 1e-12, f"perturbation does not preserve the axis-0 sum ({worst_sum:.3e})"
    assert biggest > 0.1, f"perturbation is too small to be a test ({biggest:.3e})"


@check("a sum-preserving change to the n^3 slice moves the chain output by ZERO")
def t_reduction_holds(base, pert):
    den = float(np.linalg.norm(base)) or 1.0
    rel = float(np.linalg.norm(pert - base) / den)
    print(f"          relative change in the final-layer means = {rel:.3e}")
    assert rel < 1e-12, (
        f"the chain moved by {rel:.3e}. The n^3 slice carries information beyond "
        f"its axis-0 sum, and the fused port as specified would be WRONG.")


@check("the test BITES: a sum-BREAKING change of the same size does move it")
def t_test_bites(weights, base, gen):
    """Without this, a test that always passes proves nothing."""
    real = F.DS_pK_to_K

    def patched(pK, *a, **kw):
        if 4 in pK and PART in pK[4].slices:
            X = pK[4].slices[PART]
            if X.ndim == 3:
                m = _support_mask(X.shape[0], X.device, X.dtype)
                R = torch.randn(X.shape, generator=gen, device=X.device, dtype=X.dtype)
                R = 0.5 * (R + R.transpose(1, 2)) * m
                pK[4].slices[PART] = X + R      # NOT sum-preserving
        return real(pK, *a, **kw)

    F.DS_pK_to_K = patched
    try:
        with torch.no_grad():
            out = bridge.kprop_means(weights, k_max=3, factor=True, kind=AUGMENT,
                                     device="cpu", dtype=torch.float64)
    finally:
        F.DS_pK_to_K = real
    rel = float(np.linalg.norm(out - base) / (float(np.linalg.norm(base)) or 1.0))
    print(f"          relative change from a sum-BREAKING perturbation = {rel:.3e}")
    assert rel > 1e-8, (
        f"a sum-breaking perturbation changed the output by only {rel:.3e} -- the "
        f"hook is not reaching the arithmetic, so the passing test above is vacuous")


def main() -> int:
    print("=" * 78)
    print("FUSED REDUCTION -- end-to-end, on the REAL AUGMENT chain")
    print("=" * 78)
    v = bridge.verify_upstream(bridge._src())
    print(f"  upstream {v['rev'][:12]} + recorded patch, pass={v['pass']}")
    if not v["pass"]:
        print("  REFUSED: upstream is not the pinned rev + patch.")
        return 1

    for n, d in ((24, 4), (32, 3)):
        print(f"\n  --- width {n}, depth {d} ---")
        w = _net(n, d, seed=11)
        base = run_chain(w)
        recs = []
        gen = torch.Generator().manual_seed(5)
        pert = run_chain(w, perturb_gen=gen, record=recs)
        t_perturbation_is_valid(recs)
        t_reduction_holds(base, pert)
        t_test_bites(w, base, torch.Generator().manual_seed(9))

    n_ok = sum(1 for r in RESULTS if r)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
