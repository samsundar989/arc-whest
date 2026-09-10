"""Symmetric-packed spelling: FEASIBILITY, priced on the installed meter.

Contract: `experiments/LOG.md`, *CONTRACT (Rule 1): the SYMMETRIC-PACKED
spelling* (2026-09-10), committed **before this file existed**. Handoff
`docs/STRATEGIST_RESPONSE_2026-09-10.md` §2.

WHY THIS IS A DAY AND NOT A WEEK
--------------------------------
The lever's upper bound was never computed. The strategist's framing was "does
any of our 797 matmuls have a symmetric operand at all"; this module answers a
sharper question, because the first thing it measures is that **a symmetric
operand is not what this meter discounts**.

TWO SEPARATE THINGS ARE MEASURED, AND NEITHER IS READ OFF THE SOURCE
--------------------------------------------------------------------
1. **The meter's discount law.** Six spellings of the same arithmetic, billed by
   the real installed flopscope. This is what says which redundancy is worth
   money.
2. **The chain's matmul inventory.** `_mm` is wrapped and the shipped chain is
   run, so every issued matmul is recorded with its operands' *object identity*,
   their numerical symmetry and the result's. Nothing is inferred from reading
   `_dslice21`.

WIDTH
-----
The run is at a reduced width **only** to enumerate matmuls: which products the
tower issues is fixed by `depth` and the lineage count, not by `n`. Every FLOP
figure printed is priced at the LIVE `1024 x 16` from the shape law
`m n (2k - 1)`, and the enumeration is verified against `_matmuls(depth, None)`
before any pricing happens. `docs/VALIDATION.md`'s standing lesson — a *ratio*
measured at a small width does not transfer — is respected by never transferring
one: the small run contributes a structure, not a number.

READ-ONLY
---------
`estimator_k3_strassen.py` is imported, never edited: its sha256 `3ad05d75...`
is submission `330327`'s bytes.

ROUTINE by classification, but it carries a pre-registered gate because it closes
or opens a lane. `# seed-ledger: no block named` — no truth, no accuracy, no
promotion.

Run:
    uv run python local_sim/p2_packed_feasibility.py \
        --out experiments/2026-09-10_slice5/packed_feasibility.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# `local_sim` ships numpy shims NAMED flopscope/whestbench. APPEND, never insert:
# an insert here would bill every "counted FLOP" below in a shim and make the
# whole artifact fiction.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

import flopscope as flops  # noqa: E402  (the REAL one)
import flopscope.numpy as fnp  # noqa: E402

LIVE_WIDTH, LIVE_DEPTH = 1024, 16
BUDGET = 2 ** 41
#: The gate, from the contract. Chosen before any datum.
GATE = 1.5
#: Symmetry tolerance for the numerical test on an f32 buffer.
SYM_RTOL = 1e-5


# --------------------------------------------------------------------------- #
# 1. the meter's discount law                                                  #
# --------------------------------------------------------------------------- #

def _bill(fn):
    """Counted FLOPs and operations for one spelling, in the real flopscope."""
    with flops.budget(BUDGET, quiet=True) as b:
        fn()
        return int(b.flops_used), len(b.op_log or [])


def discount_law(n=256, k=3, seed=0):
    """What does this meter actually pay for?

    Six spellings of products the chain issues or could issue. The dense
    reference is the shape law `m n (2k - 1)` that `_projected_flops` uses.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)).astype(np.float32)
    B = rng.standard_normal((n, n)).astype(np.float32)
    S = (A + A.T).astype(np.float32)
    S2 = (B + B.T).astype(np.float32)
    X = rng.standard_normal((n, k * n)).astype(np.float32)
    Y = rng.standard_normal((n, k * n)).astype(np.float32)

    sq = n * n * (2 * n - 1)
    rect = n * n * (2 * k * n - 1)

    rows = []

    def row(label, base, fn, note):
        try:
            f, o = _bill(fn)
            rows.append({"spelling": label, "flops": f, "ops": o,
                         "dense_ref": base, "ratio": f / base, "note": note})
        except Exception as exc:                       # pragma: no cover
            rows.append({"spelling": label, "error": f"{type(exc).__name__}: {exc}",
                         "dense_ref": base, "note": note})

    row("A @ B", sq, lambda: fnp.array(A) @ fnp.array(B),
        "the dense reference")
    row("as_symmetric(S) @ B", sq,
        lambda: flops.as_symmetric(fnp.array(S), symmetry=(0, 1)) @ fnp.array(B),
        "SYMMETRIC OPERAND, left")
    row("A @ as_symmetric(S)", sq,
        lambda: fnp.array(A) @ flops.as_symmetric(fnp.array(S), symmetry=(0, 1)),
        "SYMMETRIC OPERAND, right")
    row("as_symmetric(S1) @ as_symmetric(S2)", sq,
        lambda: (flops.as_symmetric(fnp.array(S), symmetry=(0, 1))
                 @ flops.as_symmetric(fnp.array(S2), symmetry=(0, 1))),
        "SYMMETRIC OPERANDS, both")
    row("A @ A.T  (symmetric result, `@`)", sq,
        lambda: fnp.array(A) @ fnp.array(A).T,
        "symmetric RESULT, written with @")

    def _alias_sq():
        a = fnp.array(A)
        return fnp.einsum("ij,kj->ik", a, a)
    row("einsum ij,kj->ik  (ALIASED operand)", sq, _alias_sq,
        "symmetric RESULT, one object passed twice")

    row("X @ Y.T  (rect, distinct)", rect,
        lambda: fnp.array(X) @ fnp.array(Y).T,
        "the `_dslice21` shape, distinct blocks")

    def _alias_rect():
        x = fnp.array(X)
        return fnp.einsum("ij,kj->ik", x, x)
    row("einsum ij,kj->ik  X,X (rect, ALIASED)", rect, _alias_rect,
        "the discount, at the lineage shape")

    def _alias_view():
        x = fnp.array(X)
        return fnp.einsum("ij,kj->ik", x, x[:, :])
    row("einsum X, X[:, :]  (a VIEW, not the object)", rect, _alias_view,
        "does a view still alias? this decides whether a spelling can fake it")

    def _half_alias():
        x = fnp.array(X)
        y = fnp.array(Y)
        return fnp.einsum("ij,kj->ik", x * y, x)
    row("einsum (X.Y), X  -- exactly `_dslice21`", rect, _half_alias,
        "one operand shares a FACTOR with the other. Does that count?")

    def _congruence():
        W = fnp.array(A)
        C = fnp.array(S)
        return (W.T @ C) @ W
    row("(W.T @ C) @ W  (the covariance step)", sq, _congruence,
        "two matmuls; ratio is against ONE, so 2.0 is the dense cost")

    def _congruence_chol():
        W = fnp.array(A)
        C = fnp.array(S + n * np.eye(n, dtype=np.float32))
        L = fnp.linalg.cholesky(C)
        Z = fnp.einsum("ji,jk->ik", W, L)
        return fnp.einsum("ij,kj->ik", Z, Z)
    row("cholesky congruence: Z=(W^T L); Z Z^T aliased", sq, _congruence_chol,
        "the ONE symmetry-aware spelling of the covariance step")

    return rows


# --------------------------------------------------------------------------- #
# 2. the chain's matmul inventory                                              #
# --------------------------------------------------------------------------- #

def _load(path):
    spec = importlib.util.spec_from_file_location("cand_packed", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cand_packed"] = mod
    spec.loader.exec_module(mod)
    return mod


def _base_id(a):
    """Identity of the underlying buffer, so a view is not mistaken for a copy."""
    raw = getattr(a, "_data", None)
    if raw is None:
        raw = getattr(a, "data", None)
    arr = raw if isinstance(raw, np.ndarray) else None
    if arr is None:
        try:
            arr = np.asarray(a)
        except Exception:                              # pragma: no cover
            return id(a)
    base = arr
    while getattr(base, "base", None) is not None:
        base = base.base
    return id(base)


def _np(a):
    try:
        return np.asarray(a, dtype=np.float64)
    except Exception:                                  # pragma: no cover
        return None


def _is_sym(a):
    v = _np(a)
    if v is None or v.ndim != 2 or v.shape[0] != v.shape[1]:
        return False
    scale = float(np.max(np.abs(v))) or 1.0
    return bool(np.max(np.abs(v - v.T)) <= SYM_RTOL * scale)


class _Recorder:
    """Wraps `_mm`, records every issued matmul, and changes no arithmetic."""

    def __init__(self, mod):
        self.mod = mod
        self.inner = mod._mm
        self.rows = []

    def __call__(self, A, B):
        out = self.inner(A, B)
        # `_mm(A, B)` is issued as `A @ B`; the caller has already transposed.
        m, kk = A.shape[0], A.shape[1]
        nn = B.shape[1]
        # The call SITE, so a claim about which product carries a symmetry can be
        # checked against the source line rather than inferred from a shape.
        frame = sys._getframe(1)
        self.rows.append({
            "site": f"{os.path.basename(frame.f_code.co_filename)}"
                    f":{frame.f_lineno} in {frame.f_code.co_name}",
            "m": m, "k": kk, "n": nn,
            "aliased": _base_id(A) == _base_id(B),
            "a_symmetric": _is_sym(A),
            "b_symmetric": _is_sym(B),
            "result_symmetric": _is_sym(out),
            "square_result": m == nn,
        })
        return out

    def __enter__(self):
        self.mod._mm = self
        return self

    def __exit__(self, *exc):
        self.mod._mm = self.inner
        return False


def _mlp(width, depth, seed):
    """A He-init net in the grader's dtype. No labels; nothing here reads truth."""
    rng = np.random.default_rng(seed)
    scale = np.float32(np.sqrt(2.0 / width))
    weights = [(rng.standard_normal((width, width)).astype(np.float32) * scale)
               for _ in range(depth)]

    class _M:
        pass

    m = _M()
    m.width = width
    m.depth = depth
    m.weights = weights
    m.seed = int(seed)
    return m


def inventory(candidate, width, depth, seed):
    mod = _load(candidate)
    # Strassen is a spelling of each matmul, not a different set of matmuls, and
    # the identity check below is against the DENSE shape law. Forcing depth 0 in
    # the loaded MODULE (never in the file) makes the two directly comparable and
    # removes `_depth_for`'s width dependence from the enumeration entirely.
    mod._STRASSEN_DEPTH = 0
    est = mod.Estimator()
    if hasattr(est, "setup"):
        class _Ctx:
            pass
        ctx = _Ctx()
        ctx.seed = int(seed)
        ctx.width = width
        ctx.depth = depth
        try:
            est.setup(ctx)
        except Exception:
            pass
    mlp = _mlp(width, depth, seed)
    with _Recorder(mod) as rec:
        with flops.budget(BUDGET, quiet=True):
            est.predict(mlp, BUDGET)
    return mod, rec.rows


# --------------------------------------------------------------------------- #
# 3. price the best case at the LIVE shape                                     #
# --------------------------------------------------------------------------- #

def price(rows, probe_width, law):
    """Dense bill vs the best symmetry-aware bill, both at 1024 x 16.

    Every recorded matmul's shape is a multiple of the probe width; scaling each
    dimension by `LIVE_WIDTH / probe_width` gives the live-shape product the same
    call site issues. That is a SHAPE identity, not a transferred ratio.
    """
    s = LIVE_WIDTH // probe_width
    assert probe_width * s == LIVE_WIDTH, "probe width must divide 1024"

    by = {r["spelling"]: r for r in law}
    alias_ratio = by["einsum ij,kj->ik  X,X (rect, ALIASED)"]["ratio"]
    chol_ratio = by["cholesky congruence: Z=(W^T L); Z Z^T aliased"]["ratio"]
    congr_ratio = by["(W.T @ C) @ W  (the covariance step)"]["ratio"]

    dense = 0
    best = 0
    n_alias = 0
    n_symres = 0
    n_symop = 0
    congruence_pairs = 0
    for r in rows:
        m, k, n = r["m"] * s, r["k"] * s, r["n"] * s
        f = m * n * (2 * k - 1)
        dense += f
        if r["aliased"]:
            n_alias += 1
            best += int(f * alias_ratio)
        else:
            best += f
        if r["result_symmetric"]:
            n_symres += 1
        if r["a_symmetric"] or r["b_symmetric"]:
            n_symop += 1
        # The covariance congruence is the ONE symmetric-result pair a spelling
        # can reach, and it is reached as a PAIR, so it is priced as one.
        if r["result_symmetric"] and r["square_result"] and r["b_symmetric"] is False \
                and r["a_symmetric"] is False:
            congruence_pairs += 1

    # Apply the Cholesky route to each congruence pair: it replaces TWO dense
    # matmuls (`congr_ratio` of one) by `chol_ratio` of one.
    one = LIVE_WIDTH * LIVE_WIDTH * (2 * LIVE_WIDTH - 1)
    congr_saving = congruence_pairs * (congr_ratio - chol_ratio) * one
    best_with_congr = best - congr_saving

    one_live = LIVE_WIDTH * LIVE_WIDTH * (2 * LIVE_WIDTH - 1)
    return {
        "mm_calls_recorded": len(rows),
        "matmul_equivalents_live": dense / one_live,
        "calls_with_aliased_operands": n_alias,
        "calls_with_a_symmetric_operand": n_symop,
        "calls_with_a_symmetric_result": n_symres,
        "congruence_pairs": congruence_pairs,
        "dense_flops_live": int(dense),
        "best_flops_live": int(best_with_congr),
        "saving_ratio": dense / best_with_congr if best_with_congr else float("nan"),
        "alias_ratio_measured": alias_ratio,
        "cholesky_congruence_ratio_measured": chol_ratio,
        "dense_congruence_ratio_measured": congr_ratio,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", default="estimator_k3_strassen.py")
    ap.add_argument("--probe-width", type=int, default=128,
                    help="enumeration only; every FLOP is priced at 1024")
    ap.add_argument("--depth", type=int, default=LIVE_DEPTH)
    ap.add_argument("--seed", type=int, default=1_000_003)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    warnings.simplefilter("ignore")
    cand = os.path.join(_ROOT, a.candidate) if not os.path.isabs(a.candidate) \
        else a.candidate
    sha = hashlib.sha256(open(cand, "rb").read()).hexdigest()

    print("=" * 78)
    print("SYMMETRIC-PACKED SPELLING -- feasibility on the installed meter")
    print("=" * 78)
    print(f"  candidate    {os.path.relpath(cand, _ROOT)}  sha256 {sha[:16]}...")
    print(f"  flopscope    {getattr(flops, '__version__', '?')}   (the REAL package)")
    print(f"  probe shape  {a.probe_width} x {a.depth}  -- ENUMERATION ONLY")
    print(f"  priced at    {LIVE_WIDTH} x {LIVE_DEPTH}")
    print()

    print("-- 1. what does this meter discount? ".ljust(78, "-"))
    law = discount_law()
    for r in law:
        if "error" in r:
            print(f"   {r['spelling']:44s}  {r['error'][:28]}")
        else:
            print(f"   {r['spelling']:44s}  {r['ratio']:7.4f} x dense"
                  f"   ops {r['ops']:>3}")
    print()
    for r in law:
        if "error" not in r:
            print(f"     {r['spelling'][:40]:42s} {r['note']}")
    print()

    print("-- 2. the chain's matmul inventory ".ljust(78, "-"))
    mod, rows = inventory(cand, a.probe_width, a.depth, a.seed)
    # THE IDENTITY CHECK. `_mm` is issued on BLOCKED operands -- one call on an
    # `(n, k n)` block is `k` of the cost model's `(n,n)@(n,n)` matmuls -- so
    # counting CALLS against `_matmuls` compares two different things, and the
    # first version of this module did exactly that and refused itself. The
    # honest identity is on FLOPs at the probe width.
    #
    # There is one more term, and it is exact rather than a tolerance. A blocked
    # call `(n, Kn) @ (Kn, n)` bills `n^2 (2Kn - 1)`, while `K` separate ones bill
    # `K n^2 (2n - 1)`. The blocked form is DEARER by exactly `n^2 (K - 1)` -- one
    # `-1` per matmul folded into one `-1` per call -- so the reconciliation is
    #     equivalents  =  _matmuls  +  sum_calls (K - 1) / (2w - 1)
    # and it vanishes as `w` grows (it is `0.178` at the live width, not `1.43`).
    w = a.probe_width
    one = w * w * (2 * w - 1)
    issued = sum(r["m"] * r["n"] * (2 * r["k"] - 1) for r in rows)
    expected = mod._matmuls(a.depth, None)
    equiv = issued / one
    blocking = sum(r["k"] // w - 1 for r in rows) / (2 * w - 1)
    resid = equiv - blocking - expected
    print(f"   `_mm` CALLS issued                 {len(rows)}")
    print(f"   matmul-equivalents from shapes     {equiv:.4f}")
    print(f"   blocking term  sum(K-1)/(2w-1)     {blocking:.4f}")
    print(f"   `_matmuls({a.depth}, None)` says          {expected}"
          f"   residual {resid:+.4f}"
          f"   {'MATCH' if abs(resid) < 0.5 else '*** MISMATCH ***'}")
    if abs(resid) >= 0.5:
        print("   The cost model and the chain disagree about how much matmul")
        print("   exists. Nothing below is priced until that is resolved.")
        return 2
    print()

    print("-- 3. the best case at the live shape ".ljust(78, "-"))
    p = price(rows, a.probe_width, law)
    for key in ("mm_calls_recorded", "matmul_equivalents_live",
                "calls_with_aliased_operands",
                "calls_with_a_symmetric_operand",
                "calls_with_a_symmetric_result", "congruence_pairs"):
        v = p[key]
        print(f"   {key:38s} {v:.4f}" if isinstance(v, float)
              else f"   {key:38s} {v}")
    print()
    print(f"   dense counted FLOPs at 1024x16     {p['dense_flops_live']:,}")
    print(f"   best symmetry-aware bill           {p['best_flops_live']:,}")
    print(f"   SAVING                             {p['saving_ratio']:.4f} x")
    print(f"   GATE                               {GATE:.2f} x")
    verdict = "CONTINUE" if p["saving_ratio"] >= GATE else "CLOSE THE LANE"
    print(f"   VERDICT                            {verdict}")
    print()

    rec = {
        "artifact_kind": "packed_spelling_feasibility",
        "contract": "experiments/LOG.md 2026-09-10 CONTRACT symmetric-packed",
        "candidate": os.path.relpath(cand, _ROOT),
        "candidate_sha256": sha,
        "flopscope_version": getattr(flops, "__version__", None),
        "probe": {"width": a.probe_width, "depth": a.depth, "seed": a.seed},
        "priced_at": {"width": LIVE_WIDTH, "depth": LIVE_DEPTH},
        "discount_law": law,
        "matmuls": rows,
        "price": p,
        "gate": GATE,
        "verdict": verdict,
        "seed_ledger": "no block named",
    }
    if a.out:
        out = a.out if os.path.isabs(a.out) else os.path.join(_ROOT, a.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2)
        print(f"  artifact  {os.path.relpath(out, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
