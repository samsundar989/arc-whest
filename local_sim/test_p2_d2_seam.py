"""Regression test for the D2 injection seam. CPU-only, seconds, no GPU, no asset.

## Why this file exists

The seam evidence for `p2_d2_k4_oracle.py` originally lived in a commit message.
The independent Skeptic (2026-09-08) called that a continuity defect and it was
right: `AGENTS.md` § Continuity says no agent may depend on another agent's chat
history, and a commit message is not runnable. So the evidence is here.

## What the no-op check can and cannot prove

Injecting the chain's OWN recovered `T` must reproduce the unmodified chain
exactly. It does — `0.000e+00`, not merely small. But that check is an **exact
algebraic bijection**: `T_from_core` and `core_from_T` are inverses, so the
constants `c1, c2` CANCEL. The no-op therefore cannot detect

  * a wrong `c1` or `c2`,
  * a wrong `T_true`,
  * a layer misalignment between the truth list and the injection counter.

Calling it "the seam's own proof" (as the module docstring once did) overstated
it. The mutations below are what prove the seam is LIVE: if the injected value
did not reach the chain's arithmetic, `3 * T_true` and `T = 0` would both be
no-ops too, and they are not.

## What each check pins

| check | pins |
|---|---|
| no-op is bit-identical | the round trip and the rebind/restore are faithful |
| ladder nested and monotone | (a) sub (b) sub (c) actually differ, in order |
| seam is live | the injected value reaches the arithmetic (Skeptic's mutations) |
| core<->T round trip | `T_from_core . core_from_T = id` |
| rung (b) sets the trace diagonal | the ERRATUM's correction, so the mislabel cannot return silently |
| patch restored on exception | a raising chain does not leave upstream monkeypatched |

No pytest: this repository's tests are plain scripts returning 0/1, invoked the
way `scripts/preflight.ps1` invokes `test_competition_score.py`. Needs torch, so
it runs in the GPU venv even though it never touches a GPU.

Run:  .venv-gpu\\Scripts\\python.exe local_sim/test_p2_d2_seam.py
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

bridge._src()  # registers the pinned upstream checkout on sys.path

import p2_d2_k4_oracle as D2  # noqa: E402
from mlp_kprop.kprop_harmonic import AUGMENT  # noqa: E402

N, DEPTH, SEED, NSAMP = 64, 5, 7, 1 << 15
DEV, DT = "cpu", torch.float32
RESULTS: list[bool] = []


def check(name):
    """Decorator: run, report, never let one failure hide the others."""
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
        run.__name__ = fn.__name__
        return run
    return deco


def _net():
    rng = np.random.default_rng(SEED)
    return [(rng.standard_normal((N, N)) * np.sqrt(2.0 / N)).astype(np.float32)
            for _ in range(DEPTH)]


def _control(w):
    with torch.no_grad():
        return bridge.kprop_means(w, k_max=3, factor=True, kind=AUGMENT,
                                  device=DEV, dtype=DT)


def _rel(m, ctrl):
    return float(np.linalg.norm(m - ctrl) / np.linalg.norm(ctrl))


@check("no-op injection is BIT-IDENTICAL to the unmodified chain")
def t_noop(w, ctrl, T):
    m = D2.kprop_means_injected(w, T, "noop", device=DEV, dtype=DT)
    r = _rel(m, ctrl)
    assert r == 0.0, (
        f"relative deviation {r:.3e}, not 0. The core<->T round trip is not exact "
        f"in this frame; every oracle number would be measuring the seam rather "
        f"than kappa_4.")


@check("the ladder is nested and monotone: 0 < (a) < (b) < (c)")
def t_ladder(w, ctrl, T):
    d = {v: _rel(D2.kprop_means_injected(w, T, v, device=DEV, dtype=DT), ctrl)
         for v in ("a", "b", "c")}
    assert 0 < d["a"] < d["b"] < d["c"], f"not nested/monotone: {d}"


@check("the seam is LIVE (Skeptic's mutations; the no-op cannot show this)")
def t_live(w, ctrl, T):
    base = _rel(D2.kprop_means_injected(w, T, "c", device=DEV, dtype=DT), ctrl)
    r3 = _rel(D2.kprop_means_injected(w, [3.0 * t for t in T], "c",
                                      device=DEV, dtype=DT), ctrl)
    r0 = _rel(D2.kprop_means_injected(w, [torch.zeros_like(t) for t in T], "c",
                                      device=DEV, dtype=DT), ctrl)
    assert r3 > 3 * base, f"3*T_true barely moved the chain ({r3:.3e} vs {base:.3e})"
    assert r0 > 0.25 * base, f"T=0 barely moved the chain ({r0:.3e} vs {base:.3e})"


@check("core<->T round trip is exact to f64 roundoff")
def t_roundtrip():
    g = torch.Generator().manual_seed(3)
    A = torch.randn(N, N, generator=g, dtype=torch.float64)
    T = 0.5 * (A + A.T)
    c1, c2 = D2._proj_constants(N, T.device, T.dtype)
    back = D2.T_from_core(D2.core_from_T(T, c1, c2), c1, c2)
    r = float(torch.linalg.norm(back - T) / torch.linalg.norm(T))
    assert r < 1e-12, f"round trip relative error {r:.3e}"


@check("rung (b) sets the TRACE MATRIX's diagonal, not per-neuron kappa_4")
def t_rung_b():
    """Pins the 2026-09-08 ERRATUM so the mislabel cannot come back quietly.

    diag(T)_i = sum_k kappa_4(i,i,k,k) is a per-neuron TRACE. Rung (b) sets
    exactly that: the off-diagonal must stay the chain's own.
    """
    g = torch.Generator().manual_seed(11)
    a1 = torch.randn(N, N, generator=g, dtype=torch.float64)
    a2 = torch.randn(N, N, generator=g, dtype=torch.float64)
    T_chain, T_true = 0.5 * (a1 + a1.T), 0.5 * (a2 + a2.T)
    out = D2.mix_T(T_chain, T_true, "b")
    assert torch.allclose(out.diagonal(), T_true.diagonal()), "diagonal not set true"
    off = ~torch.eye(N, dtype=torch.bool)
    assert torch.allclose(out[off], T_chain[off]), (
        "rung (b) must touch the DIAGONAL of the trace matrix ONLY")


@check("upstream is restored even when the chain raises")
def t_restore():
    from mlp_kprop import factor_k3 as F
    before = F.factored_nonlin_kprop_k3
    try:
        D2.kprop_means_injected([np.zeros((N, N), dtype=np.float32)],
                                ["not a tensor"], "c", device=DEV, dtype=DT)
    except Exception:
        pass
    assert F.factored_nonlin_kprop_k3 is before, (
        "upstream was left patched; the next arm would silently inject too")


def main() -> int:
    print("=" * 78)
    print(f"D2 seam regression -- {N}x{DEPTH} on CPU, upstream pinned")
    print("=" * 78)
    v = bridge.verify_upstream(bridge._src())
    print(f"  upstream {v['rev'][:12]} + recorded patch, pass={v['pass']}")
    if not v["pass"]:
        print("  REFUSED: upstream is not the pinned rev + patch.")
        return 1
    w = _net()
    ctrl = _control(w)
    T, _ = D2.true_trace_matrices(w, 424242, NSAMP, DEV, 4096, dtype=DT)
    print()
    t_noop(w, ctrl, T)
    t_ladder(w, ctrl, T)
    t_live(w, ctrl, T)
    t_roundtrip()
    t_rung_b()
    t_restore()
    n_ok = sum(1 for r in RESULTS if r)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
