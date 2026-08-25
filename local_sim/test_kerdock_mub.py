"""Falsify the width-1024 real-MUB carrier. Phase 1A verifier, V1-V7.

Specification: `docs/KERDOCK_MUB_SPEC.md` §7. Prediction K1-K11:
`experiments/LOG.md`, `PHASE 1A ... PREDICTION`.

WHY THIS IS A SEPARATE FILE WITH DIFFERENT LOGIC
------------------------------------------------
`kerdock_mub.py` emits bases from the Galois ring `GR(4, m)` through the Gray
map. If this file checked the same object the same way, a shared misunderstanding
would make both agree and neither would notice. So every substantive check here
reaches the answer by a route the generator does not use:

* **algebraic normal form + rank over F2.** From the sign vectors alone, recover
  the Boolean functions, Möbius-transform them to their ANF, assert degree ≤ 2,
  extract the alternating matrices, and check that every pairwise **difference is
  nonsingular**. That is the defining property of a Kerdock set, established by
  linear algebra, with no ring, no trace and no Gray map in sight.
* **a naive O(N²) Walsh transform** cross-checked against a fast butterfly one.
* **direct integer inner products**, exhaustive over every ordered pair of rows,
  at the reduced dimensions where that is affordable.

K6 IS CORRECTED, NOT DEFENDED
-----------------------------
The Phase-1A prediction K6 said generator and verifier are "two genuinely
different code paths (direct emission vs. an independent reference)". **That
overclaims and is withdrawn.** This file imports the generator and checks
*properties of its output*. That is a strong independent PROPERTY CHECKER — ANF
degree, rank over F2, a second transform implementation, exhaustive integer Gram
matrices — and it is not an independently generated reference object.

The honest scope: **nothing here would catch a construction that is wrong in a
way that still satisfies every property checked.** What rules that out is that
the properties checked are the *defining* ones — flat `±2^(n/2)` cross spectrum,
ANF degree 2, and nonsingular pairwise differences over all C(512,2) pairs — and
together they are the definition of a Kerdock set, not a sample of its
consequences.

Reconstructing CCKS Example 9.2 as a true second generator was attempted and
abandoned: the finite-field part is additive in `λ` but the radical-correcting
term goes as `λ^(−1/3)` and is not, so pairwise nonsingularity is not automatic
and the correction cannot be pinned down from memory. Building a "reference" and
then adjusting it until it agreed with the generator would be worse than having
none. Recorded as a limitation; the gate is narrowed to match.

THE ONE IDENTITY THIS SUITE LEANS ON, AND HOW IT IS EARNED
----------------------------------------------------------
`⟨B_λ[:,y], B_μ[:,y']⟩` depends only on `y ⊕ y'`, and equals the Walsh coefficient
of `s_λ ⊙ s_μ` there. So one length-`N` spectrum decides all `N²` cross pairs.

That is an identity, not a shortcut — but this file does not assume it. It is
**verified exhaustively against direct inner products at N = 16, 64 and 256**
(V7), and only then used at `N = 1024`, where an exhaustive `16,384²` integer
Gram matrix would be ~1 GB. At 1024 the spectrum result is additionally spot-
checked against direct inner products on a fixed, deterministic sample.

Run:  uv run python local_sim/test_kerdock_mub.py
"""

from __future__ import annotations

import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

from kerdock_mub import (GaloisRing, carrier_float32, carrier_int8,  # noqa: E402
                         sign_vectors, walsh_matrix)

_PASS = 0
_FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global _PASS, _FAIL
    ok = bool(ok)
    if ok:
        _PASS += 1
    else:
        _FAIL += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<66} {detail}")
    return ok


def head(t: str) -> None:
    print(f"\n{t}\n{'-' * len(t)}")


# --------------------------------------------------------------------------- #
# independent machinery -- none of this is used by the generator
# --------------------------------------------------------------------------- #

def wht_fast(v: np.ndarray) -> np.ndarray:
    """In-place-style fast Walsh-Hadamard transform, int64."""
    a = v.astype(np.int64).copy()
    n = a.size
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            x = a[i:i + h].copy()
            y = a[i + h:i + 2 * h].copy()
            a[i:i + h] = x + y
            a[i + h:i + 2 * h] = x - y
        h *= 2
    return a


def wht_naive(v: np.ndarray, W: np.ndarray) -> np.ndarray:
    """The same transform as an explicit matrix product. Different code path."""
    return W.astype(np.int64) @ v.astype(np.int64)


def anf(truth: np.ndarray, n: int) -> np.ndarray:
    """Möbius transform: truth table -> algebraic normal form, over F2."""
    a = truth.astype(np.uint8).copy()
    for i in range(n):
        bit = 1 << i
        idx = np.arange(a.size)
        hi = idx[(idx & bit) != 0]
        a[hi] ^= a[hi ^ bit]
    return a


def quadratic_matrix(anf_coeffs: np.ndarray, n: int):
    """`(A, max_degree)` — the alternating matrix of a degree-≤2 Boolean function.

    `A[i][j] = 1` iff the monomial `x_i x_j` is present. Diagonal is zero: linear
    terms are not part of the quadratic form and do not affect bentness.
    """
    A = np.zeros((n, n), dtype=np.uint8)
    maxdeg = 0
    for mask in np.nonzero(anf_coeffs)[0]:
        bits = [i for i in range(n) if mask >> i & 1]
        maxdeg = max(maxdeg, len(bits))
        if len(bits) == 2:
            i, j = bits
            A[i, j] = A[j, i] = 1
    return A, maxdeg


def rank_gf2(M: np.ndarray) -> int:
    A = M.astype(np.uint8).copy()
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if A[i, c]:
                piv = i
                break
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        r += 1
    return r


# --------------------------------------------------------------------------- #

def exhaustive_gram(m: int, k: int) -> tuple:
    """Every ordered pair of rows, by DIRECT integer inner product.

    Returns `(within_max_abs, cross_values)` — the exact sets the spec predicts:
    `0` off-diagonal within a basis, `±2^((m+1)/2)` across.
    """
    N = 1 << (m + 1)
    rows = carrier_int8(m, k, antipodes=False).astype(np.int64)
    G = rows @ rows.T
    within_off, cross = [], []
    for a in range(k):
        for b in range(k):
            blk = G[a * N:(a + 1) * N, b * N:(b + 1) * N]
            if a == b:
                off = blk[~np.eye(N, dtype=bool)]
                within_off.append(np.abs(off).max() if off.size else 0)
            else:
                cross.append(np.unique(np.abs(blk)))
    return (max(within_off) if within_off else 0,
            np.unique(np.concatenate(cross)) if cross else np.array([]))


def main() -> int:
    print("=" * 100)
    print("KERDOCK / REAL-MUB CARRIER -- V1-V7 against docs/KERDOCK_MUB_SPEC.md")
    print("=" * 100)

    # ------------------------------------------------------------------ #
    head("V7 (first, because it licenses everything else) -- EXHAUSTIVE at reduced N")
    # The spec's identity is used at N=1024 only after being confirmed against
    # brute force where brute force is affordable. m = 3, 5, 7 -- all odd.
    for m in (3, 5, 7):
        N = 1 << (m + 1)
        k = min(8, 1 << m)
        want = int(np.sqrt(N))
        within, cross = exhaustive_gram(m, k)
        check(f"m={m} N={N:>4} k={k}: within-basis off-diagonal is EXACTLY 0",
              within == 0, f"max |.| = {within}")
        check(f"m={m} N={N:>4} k={k}: cross-basis |.| is EXACTLY {want}, "
              f"never 0, never other",
              cross.size == 1 and cross[0] == want,
              f"distinct |values| = {cross.tolist()}")

    # ------------------------------------------------------------------ #
    head("V7b -- the spectrum identity agrees with brute force, so it may be used at 1024")
    for m in (3, 5):
        N = 1 << (m + 1)
        k = min(6, 1 << m)
        s = sign_vectors(m, k)
        W = walsh_matrix(m + 1)
        rows = carrier_int8(m, k, antipodes=False).astype(np.int64)
        agree = True
        for a in range(k):
            for b in range(a + 1, k):
                spec = wht_fast(s[a].astype(np.int64) * s[b].astype(np.int64))
                blk = rows[a * N:(a + 1) * N] @ rows[b * N:(b + 1) * N].T
                yy = np.arange(N)
                pred = spec[yy[:, None] ^ yy[None, :]]
                if not np.array_equal(blk, pred):
                    agree = False
        check(f"m={m}: <col y, col y'> == WHT(s_a*s_b)[y XOR y'] for every pair",
              agree, "identity holds elementwise on the full Gram block")
        naive_ok = all(np.array_equal(wht_fast(s[i].astype(np.int64)),
                                      wht_naive(s[i], W)) for i in range(k))
        check(f"m={m}: fast WHT == naive O(N^2) WHT", naive_ok,
              "two independent transform implementations")

    # ------------------------------------------------------------------ #
    head("V1-V5 at the LIVE shape -- m=9, N=1024, 16 bases")
    m, k = 9, 16
    R = GaloisRing(m)
    N, n = R.N, R.n
    want = int(np.sqrt(N))
    s = sign_vectors(m, k)

    check("V1: sign vectors are exactly +-1",
          set(np.unique(s).tolist()) == {-1, 1}, f"values {np.unique(s).tolist()}")

    # V2 -- weak by construction, and labelled as such
    W = walsh_matrix(n)
    b0 = (s[0][:, None] * W).astype(np.int64)
    g0 = b0.T @ b0
    check("V2: within-basis Gram is exactly N*I  (WEAK -- holds for any signs)",
          np.array_equal(g0, N * np.eye(N, dtype=np.int64)),
          f"off-diagonal max |.| = {np.abs(g0 - N * np.eye(N, dtype=np.int64)).max()}")

    # V3 -- THE discriminator, via the identity licensed above
    worst, allflat = None, True
    for a in range(k):
        for b in range(a + 1, k):
            spec = wht_fast(s[a].astype(np.int64) * s[b].astype(np.int64))
            u = np.unique(np.abs(spec))
            if u.size != 1 or u[0] != want:
                allflat = False
                worst = (a, b, u.tolist())
    check(f"V3: cross-basis |correlation| is EXACTLY {want} for all "
          f"{k * (k - 1) // 2} basis pairs x {N}^2 vector pairs",
          allflat, "flat Walsh spectrum on every pair" if allflat else f"{worst}")

    # ...and spot-check the identity against direct inner products at 1024
    rng = np.random.default_rng(0)
    rows0 = (s[0][:, None] * W).astype(np.int64)
    rows1 = (s[1][:, None] * W).astype(np.int64)
    spec01 = wht_fast(s[0].astype(np.int64) * s[1].astype(np.int64))
    ys = rng.choice(N, size=64, replace=False)
    ok_spot = all(int(rows0[:, y] @ rows1[:, y2]) == int(spec01[y ^ y2])
                  for y in ys for y2 in ys[:8])
    check("V3b: direct inner products agree with the spectrum at N=1024 too",
          ok_spot, "512 sampled pairs, exact integers")

    rows = carrier_int8(m, k, antipodes=True)
    check(f"V4: row count is {2 * k * N:,} and dimension {N}",
          rows.shape == (2 * k * N, N), f"{rows.shape}")
    check("V4b: the set is exactly closed under negation",
          np.array_equal(rows[0::2], -rows[1::2]), "every +v is paired with -v")
    uniq = np.unique(rows, axis=0).shape[0]
    check(f"V5: all {2 * k * N:,} rows are distinct", uniq == 2 * k * N,
          f"{uniq:,} distinct")
    lines = np.unique(np.where((rows[:, :1] != 0) & (rows[:, :1] < 0), -rows, rows),
                      axis=0).shape[0]
    check(f"V4c: they span exactly {k * N:,} distinct LINES", lines == k * N,
          f"{lines:,} lines")

    # ------------------------------------------------------------------ #
    head("V7c -- INDEPENDENT characterisation: ANF degree and rank over F2")
    # No ring, no trace, no Gray map: recover the Boolean functions from the
    # signs and test the Kerdock property by linear algebra.
    g = ((1 - s) // 2).astype(np.uint8)
    mats, degs = [], []
    for i in range(k):
        A, d = quadratic_matrix(anf(g[i], n), n)
        mats.append(A)
        degs.append(d)
    check("every basis function has ANF degree <= 2 (it is a quadratic form)",
          max(degs) <= 2, f"max degree {max(degs)}")
    ranks = []
    for a in range(k):
        for b in range(a + 1, k):
            ranks.append(rank_gf2(mats[a] ^ mats[b]))
    check(f"every pairwise matrix DIFFERENCE is nonsingular (rank {n})",
          all(r == n for r in ranks),
          f"{len(ranks)} pairs, ranks {sorted(set(ranks))} -- the Kerdock property")

    # ------------------------------------------------------------------ #
    head("V7d -- the FULL Kerdock set, not just the 16 we chose")
    # Stronger than V7c and independent of the basis subset: build every one of
    # the 512 phase functions at m=9 and check all C(512,2) = 130,816 pairwise
    # differences. If the construction only worked for our chosen prefix, this
    # is where that would show.
    s_all = sign_vectors(9, 512)
    g_all = ((1 - s_all) // 2).astype(np.uint8)
    A_all, deg_all = [], 0
    for i in range(512):
        A, d = quadratic_matrix(anf(g_all[i], 10), 10)
        A_all.append(A)
        deg_all = max(deg_all, d)
    check("all 512 phase functions have ANF degree <= 2", deg_all <= 2,
          f"max degree {deg_all}")
    Aflat = np.array([A.reshape(-1) for A in A_all], dtype=np.uint8)
    bad = 0
    for i in range(512):
        diffs = Aflat[i + 1:] ^ Aflat[i]
        for row in diffs:
            if rank_gf2(row.reshape(10, 10)) != 10:
                bad += 1
    check("all C(512,2) = 130,816 pairwise differences are nonsingular",
          bad == 0, f"{bad} singular -- the WHOLE Kerdock set, not a lucky subset")

    # ------------------------------------------------------------------ #
    head("NEGATIVE CONTROL -- the rejected coordinate mapping, still failing")
    # A rejection nobody can re-run is an anecdote. This re-executes the mapping
    # the spec first specified and records what it actually produces.
    from kerdock_mub import bad_mapping_sign_vectors
    sb = bad_mapping_sign_vectors(9, 16)
    gb = ((1 - sb) // 2).astype(np.uint8)
    degs_b = [quadratic_matrix(anf(gb[i], 10), 10)[1] for i in range(16)]
    spec_b = set()
    ranks_b = set()
    mats_b = [quadratic_matrix(anf(gb[i], 10), 10)[0] for i in range(16)]
    for a in range(16):
        for b in range(a + 1, 16):
            spec_b |= set(np.unique(np.abs(
                wht_fast(sb[a].astype(np.int64) * sb[b].astype(np.int64)))).tolist())
            ranks_b.add(rank_gf2(mats_b[a] ^ mats_b[b]))
    check("the rejected mapping does NOT give a flat 32 spectrum",
          spec_b != {32},
          f"{len(spec_b)} distinct |values|, min {min(spec_b)} max {max(spec_b)}")
    check("...and does NOT give degree-2 phase functions",
          max(degs_b) > 2, f"max ANF degree {max(degs_b)}")
    check("...and does NOT give nonsingular differences",
          ranks_b != {10}, f"ranks {sorted(ranks_b)}")
    # the check that would have MISSED it -- the more useful half of the lesson
    Wb = walsh_matrix(10)
    bb = (sb[0][:, None] * Wb).astype(np.int64)
    gb0 = bb.T @ bb
    check("...while within-basis orthogonality passes for it PERFECTLY",
          np.array_equal(gb0, 1024 * np.eye(1024, dtype=np.int64)),
          "V2 could not have caught this, and did not")
    art = os.path.join(_ROOT, "experiments", "2026-08-25_p1a",
                       "rejected_mapping_negative_control.txt")
    os.makedirs(os.path.dirname(art), exist_ok=True)
    with open(art, "w", encoding="utf-8") as fh:
        fh.write("REJECTED coordinate mapping: index = 2*rank(x) + t\n")
        fh.write("(rank = position in the MULTIPLICATIVE enumeration; not F2-linear)\n\n")
        fh.write(f"cross-basis |spectrum| values : {sorted(spec_b)}\n")
        fh.write(f"  -> expected for a valid set : [32]\n")
        fh.write(f"ANF degrees                   : {sorted(set(degs_b))}\n")
        fh.write(f"  -> expected                 : [2]\n")
        fh.write(f"pairwise difference ranks     : {sorted(ranks_b)}\n")
        fh.write(f"  -> expected                 : [10]\n\n")
        fh.write("within-basis Gram == N*I      : "
                 f"{np.array_equal(gb0, 1024 * np.eye(1024, dtype=np.int64))}\n")
        fh.write("  -> the weak check passes for the WRONG construction, which is\n")
        fh.write("     why the spec calls it weak and why V3/V7c exist.\n")
    print(f"      wrote {os.path.relpath(art, _ROOT)}")

    # ------------------------------------------------------------------ #
    head("LIVE SUBSET -- k = 1/2/4/8/16, configurable bases, float32")
    prev = None
    nested = True
    for kk in (1, 2, 4, 8, 16):
        c = carrier_int8(9, kk)
        if c.shape != (2 * kk * 1024, 1024):
            check(f"k={kk} shape", False, f"{c.shape}")
        if prev is not None and not np.array_equal(c[:prev.shape[0]], prev):
            nested = False
        prev = c
    check("k = 1/2/4/8/16 give 2*k*1024 rows and are NESTED prefixes", nested,
          "k=8 output is exactly the first half of k=16")
    s16 = sign_vectors(9, 16)
    s_pick = sign_vectors(9, 3, basis_indices=[5, 2, 9])
    check("basis_indices selects and ORDERS the requested bases",
          np.array_equal(s_pick[0], s16[5]) and np.array_equal(s_pick[1], s16[2]),
          "explicit indices, in the order given")
    cf = carrier_float32(9, 16)
    norms = np.linalg.norm(cf.astype(np.float64), axis=1)
    check("float32 carrier has dtype float32 and shape (32768, 1024)",
          cf.dtype == np.float32 and cf.shape == (32768, 1024),
          f"{cf.dtype}, {cf.shape}, {cf.nbytes / 2**20:.0f} MiB")
    check("...and every row has unit norm to float32 precision",
          np.abs(norms - 1.0).max() < 1e-6,
          f"max |‖v‖ - 1| = {np.abs(norms - 1.0).max():.3e}")
    check("...entries are exactly +-1/32",
          set(np.unique(cf).tolist()) == {-1 / 32, 1 / 32},
          f"{np.unique(cf).tolist()}")

    # ------------------------------------------------------------------ #
    head("V6 -- determinism")
    a1 = carrier_int8(9, 4).tobytes()
    a2 = carrier_int8(9, 4).tobytes()
    check("two calls in this process are byte-identical", a1 == a2,
          f"{len(a1):,} bytes")
    code = ("import sys,os;sys.path=[p for p in sys.path if os.path.abspath(p or '.')"
            f"!={_HERE!r}];sys.path.append({_HERE!r});"
            "import hashlib;from kerdock_mub import carrier_int8;"
            "print(hashlib.sha256(carrier_int8(9,4).tobytes()).hexdigest())")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=_ROOT)
    import hashlib
    mine = hashlib.sha256(a1).hexdigest()
    check("a FRESH PROCESS produces the same bytes",
          p.returncode == 0 and p.stdout.strip() == mine,
          f"{mine[:16]}... vs {p.stdout.strip()[:16]}...")

    # ------------------------------------------------------------------ #
    head("K9 -- the counted-FLOP fingerprint, arithmetic only")
    rowsn, depth = 2 * k * N, 16
    dense = 2 * rowsn * N * N * depth
    leader = 1_099_802_385_000
    check("dense 32,768-direction propagation costs 1,099,511,627,776 FLOPs",
          dense == 1_099_511_627_776, f"{dense:,}")
    print(f"      leader published   {leader:,}")
    print(f"      overhead           {leader - dense:,}  (+{100 * (leader / dense - 1):.4f}%)")
    print("      NOTE: that 16 bases is INFERRED from this arithmetic, not disclosed.")

    print("\n" + "=" * 100)
    print(f"{_PASS}/{_PASS + _FAIL} checks passed")
    print(f"KERDOCK-MUB-RESULT: {_PASS}/{_PASS + _FAIL} checks passed, "
          f"exit {0 if _FAIL == 0 else 1}")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
