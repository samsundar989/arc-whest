"""WP5 — the integration layer: branch isolation, refusals, fallback, identities.

Contract: `experiments/LOG.md`, *CONTRACT WP1+WP5*.

DIVISION OF LABOUR, SO NOTHING IS TESTED TWICE OR ZERO TIMES
-------------------------------------------------------------
| property | where it is tested |
|---|---|
| carrier byte parity, Walsh/ring/Gram, chunk edges, radial factor | `test_p2_carrier_fnp.py` |
| `whest validate`, fresh-worker determinism, immutability, FLOPs, multiplier, residual/wall, scalar sync, memory, subprocess smoke | `certify_candidate.py`, run on the BUILT single file |
| **branch isolation, blend identities, refusals, fallback, config matrix, cost model, shape genericity** | **here** |

This suite deliberately does NOT re-implement the certifier. A second copy of
those checks would drift from the first, which is the defect
`local_sim/competition_score.py` exists to prevent.

`1024 x 16` IS EXERCISED FIRST, because a suite that only ever runs small
shapes has never met the live path -- and v2.6 shipped an odd-width defect that
produced finite, plausible, wrong numbers on 7 of 17 shapes.

Run: uv run python local_sim/test_p2_candidate_architecture.py
     uv run python local_sim/test_p2_candidate_architecture.py --quick  # skip 1024
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402  -- the ORACLE's language, not the candidate's
import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402

import p2_candidate_architecture as A  # noqa: E402
import p2_carrier_fnp as C  # noqa: E402
from mint_provenance import canonical_mint  # noqa: E402

try:
    flops.configure(symmetry_warnings=False)
except Exception:
    pass

_FAILED = []
_TOTAL = [0]

#: Outside every registered block (`local_sim/seed_blocks.py`). This suite reads
#: no truth and claims no block.
SEEDS = (90000000501, 90000000502, 90000000503)


def check(name, ok, detail=""):
    _TOTAL[0] += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<66}{detail}")
    if not ok:
        _FAILED.append(name)


def raises(name, fn, exc=A.CandidateRefusal, detail=""):
    try:
        fn()
    except exc as e:
        check(name, True, detail or str(e)[:56])
        return
    except Exception as e:
        check(name, False, f"raised {type(e).__name__}, expected {exc.__name__}")
        return
    check(name, False, "NO REFUSAL -- it returned a number")


class Ctx:
    def __init__(self, width, depth):
        self.width, self.depth = width, depth
        self.flop_budget, self.seed, self.n_mlps = 2 ** 41, 1, 1


def mlp_of(width, depth, seed):
    w_np, est_seed, _ = canonical_mint(width, depth, seed)
    from whestbench import MLP
    return MLP(weights=[fnp.array(x) for x in w_np], width=width, depth=depth,
               seed=est_seed), w_np


def run(cfg, width, depth, seed, **kw):
    c = A.ResearchCandidate(config=cfg, **kw).setup(Ctx(width, depth))
    m, w_np = mlp_of(width, depth, seed)
    return np.asarray(c.predict(m, 2 ** 41), dtype=np.float64), c, w_np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help="skip the 1024x16 sections")
    a = ap.parse_args()
    shapes = ([] if a.quick else [(1024, 16)]) + [(256, 8), (64, 4)]

    print("WP5 - the integration layer\n")

    # ------------------------------------------------------------------ S --
    print("S  the LIVE shape first, then smaller shape-generic cases")
    for width, depth in shapes:
        for cfg in ("carrier", "k2_diag"):
            out, c, w_np = run(cfg, width, depth, SEEDS[0])
            # A post-ReLU mean cannot be negative as a QUANTITY, but a float
            # sum landing on a near-zero neuron can round below it. Measured:
            # `k2_diag` at 1024x16 puts 5 of 16,384 entries at `-8.1e-17`
            # against row means near `0.5` -- one ULP, `1.6e-17` relative. The
            # shipped v2.6.1 does the same thing (7 of 65,536 at `-9.1e-17`).
            # `>= 0` exactly would fail a correct estimator; the bar is scaled
            # to the row, four orders above the noise and many below a sign
            # error, and matches `certify_candidate.py` section 5.
            bar = -1e-12 * max(float(np.abs(out).max()), 1.0)
            ok = (out.shape == (depth, width) and np.all(np.isfinite(out))
                  and float(out.min()) >= bar)
            check(f"S: {cfg} at {width}x{depth} returns a finite non-negative "
                  f"({depth}, {width})", ok, f"{out.shape}, "
                  f"final-row mean {out[-1].mean():.6f}, min {out.min():.2e}")
        # the exact layer-1 identity, on BOTH branches
        w0 = w_np[0]
        exact = np.sqrt((np.asarray(w0, dtype=np.float64) ** 2).sum(axis=0)) \
            / np.sqrt(2 * np.pi)
        for cfg in ("carrier", "k2_diag"):
            out, _, _ = run(cfg, width, depth, SEEDS[0])
            rel = float(np.abs(out[0] - exact).max() / np.abs(exact).max())
            check(f"S: {cfg} at {width}x{depth} layer 1 is the CLOSED FORM",
                  rel < 1e-12, f"max rel {rel:.3e}")

    # ------------------------------------------------------------------ E --
    print("\nE  hidden-layer permutation equivariance")
    # Relabelling layer k's neurons leaves the network's function unchanged, so
    # row k must permute and every other row must not move. This is the check
    # that catches a transposed index or an axis swapped in a reduction --
    # mistakes that produce perfectly finite, perfectly plausible, wrong numbers.
    rng = np.random.default_rng(90000000511)
    for width, depth, kperm in ((256, 8, 3), (64, 4, 1)):
        base, cand, w_np = run("carrier", width, depth, SEEDS[1])
        P = rng.permutation(width)
        w2 = [np.array(x) for x in w_np]
        w2[kperm] = w2[kperm][:, P]
        w2[kperm + 1] = w2[kperm + 1][P, :]
        from whestbench import MLP
        m2 = MLP(weights=[fnp.array(x) for x in w2], width=width, depth=depth,
                 seed=1)
        perm = np.asarray(cand.predict(m2, 2 ** 41), dtype=np.float64)
        scale = float(np.abs(base[-1]).mean())
        d_k = float(np.abs(perm[kperm] - base[kperm][P]).max())
        others = [l for l in range(depth) if l != kperm]
        d_o = float(np.abs(perm[others] - base[others]).max())
        check(f"E: {width}x{depth} permuting layer {kperm} PERMUTES row "
              f"{kperm}", d_k / max(scale, 1e-30) < 1e-5, f"{d_k:.3e}")
        check(f"E: {width}x{depth} ...and moves the other rows only by float "
              "association", d_o / max(scale, 1e-30) < 1e-5, f"{d_o:.3e}")

    # ------------------------------------------------------------------ B --
    print("\nB  branch ISOLATION and the zero-weight blend IDENTITY")
    width, depth = 256, 8
    car, _, _ = run("carrier", width, depth, SEEDS[0])
    k2, _, _ = run("k2_diag", width, depth, SEEDS[0])
    zero, _, _ = run("carrier_k2_zero", width, depth, SEEDS[0], blend=0.0)
    check("B: the two branches produce DIFFERENT answers",
          not np.allclose(car, k2, rtol=1e-3),
          f"max rel {np.abs(car - k2).max() / np.abs(car).max():.3e} -- "
          "otherwise isolation is untestable")
    check("B: blend=0.0 is EXACTLY the carrier branch",
          np.array_equal(zero, car), "an identity, bitwise")
    one, _, _ = run("carrier_k2_zero", width, depth, SEEDS[0], blend=1.0)
    check("B: blend=1.0 is EXACTLY the analytic branch",
          np.array_equal(one, k2), "the other end of the same identity")
    half, _, _ = run("carrier_k2_zero", width, depth, SEEDS[0], blend=0.25)
    check("B: an intermediate blend is the stated convex combination",
          np.allclose(half, car * 0.75 + k2 * 0.25, rtol=1e-12, atol=0),
          "carrier*(1-w) + analytic*w")

    # ------------------------------------------------------------------ R --
    print("\nR  the refusals, which are the point of the placeholders")
    raises("R: 'carrier_k2_frozen' with NO coefficient REFUSES",
           lambda: A.ResearchCandidate(config="carrier_k2_frozen")
           .setup(Ctx(256, 8)))
    raises("R: 'augment' REFUSES (WP3 not implemented)",
           lambda: A.ResearchCandidate(config="augment").setup(Ctx(256, 8)))
    raises("R: 'augment_k3' REFUSES (WP4 not implemented)",
           lambda: A.ResearchCandidate(config="augment_k3").setup(Ctx(256, 8)))
    ok_frozen = A.ResearchCandidate(config="carrier_k2_frozen", blend=0.3)
    ok_frozen.setup(Ctx(256, 8))
    check("R: ...but WITH an explicit coefficient it proceeds",
          ok_frozen.plan is not None,
          "so the refusal is about provenance, not a broken branch")
    raises("R: an unknown config is rejected at construction",
           lambda: A.ResearchCandidate(config="nope"), exc=ValueError)

    # ------------------------------------------------------------------ F --
    print("\nF  the fallback is LOUD, and strict mode refuses it")
    c = A.ResearchCandidate(config="carrier").setup(Ctx(300, 4))   # not 2^n
    check("F: an unsupported width falls back and RECORDS why",
          c.plan is None and "no Kerdock construction at width 300"
          in c.fallback_reason, c.fallback_reason[:52])
    m, _ = mlp_of(300, 4, SEEDS[0]) if False else (None, None)
    raises("F: strict=True turns the same fallback into a REFUSAL",
           lambda: A.ResearchCandidate(config="carrier", strict=True)
           .setup(Ctx(300, 4)))
    c128 = A.ResearchCandidate(config="carrier").setup(Ctx(128, 4))
    check("F: width 128 (m=6, EVEN) also falls back -- odd m is required",
          c128.plan is None and "m = 6" in c128.fallback_reason,
          c128.fallback_reason[:52])
    out128, _, _ = run("carrier", 128, 4, SEEDS[0])
    check("F: ...and the fallback still returns a valid answer",
          out128.shape == (4, 128) and np.all(np.isfinite(out128)),
          "a loud fallback, not a crash")
    # a plan built for one width must not be used at another
    cross = A.ResearchCandidate(config="carrier").setup(Ctx(256, 8))
    o64 = np.asarray(cross.predict(mlp_of(64, 4, SEEDS[0])[0], 2 ** 41),
                     dtype=np.float64)
    check("F: a 256-carrier is NOT applied to a width-64 network",
          o64.shape == (4, 64) and np.all(np.isfinite(o64)),
          "the width guard routes it to the analytic branch")

    # ------------------------------------------------------------------ D --
    print("\nD  determinism and first-call behaviour")
    c = A.ResearchCandidate(config="carrier").setup(Ctx(256, 8))
    m, _ = mlp_of(256, 8, SEEDS[2])
    hs = [hashlib.sha256(np.ascontiguousarray(
        np.asarray(c.predict(m, 2 ** 41), dtype=np.float64)).tobytes()
    ).hexdigest() for _ in range(3)]
    check("D: three calls on ONE instance are bitwise identical",
          len(set(hs)) == 1, hs[0][:16])
    c2 = A.ResearchCandidate(config="carrier").setup(Ctx(256, 8))
    h2 = hashlib.sha256(np.ascontiguousarray(
        np.asarray(c2.predict(m, 2 ** 41), dtype=np.float64)).tobytes()).hexdigest()
    check("D: a FRESH instance agrees -- no setup state leaks between MLPs",
          h2 == hs[0], h2[:16])
    check("D: the FIRST call equals the third (no warm-up transient in the "
          "VALUES)", hs[0] == hs[2], "a warm-up would change timing, not bytes")

    # ------------------------------------------------------------------ P --
    print("\nP  pathological and degenerate networks")
    from whestbench import MLP
    width, depth = 256, 4
    cases = {
        "all-zero layer 2": lambda w: [w[0], np.zeros_like(w[1]), w[2], w[3]],
        "one layer scaled 1e-6": lambda w: [w[0], w[1] * np.float32(1e-6),
                                            w[2], w[3]],
        "one layer scaled 1e3": lambda w: [w[0], w[1] * np.float32(1e3),
                                           w[2], w[3]],
        "a rank-1 layer": lambda w: [
            w[0], np.outer(w[1][:, 0], w[1][0, :]).astype(np.float32), w[2], w[3]],
    }
    cand = A.ResearchCandidate(config="carrier").setup(Ctx(width, depth))
    _, w_np, = None, canonical_mint(width, depth, SEEDS[0])[0]
    for name, f in cases.items():
        ww = f([np.array(x) for x in w_np])
        mm = MLP(weights=[fnp.array(x) for x in ww], width=width, depth=depth,
                 seed=1)
        try:
            o = np.asarray(cand.predict(mm, 2 ** 41), dtype=np.float64)
            ok = (o.shape == (depth, width) and np.all(np.isfinite(o))
                  and np.all(o >= -1e-9))
            detail = f"final mean {o[-1].mean():.3e}"
        except Exception as exc:
            ok, detail = False, f"RAISED {type(exc).__name__}: {exc}"
        check(f"P: {name} -> finite, non-negative, right shape", ok, detail)

    # ------------------------------------------------------------------ C --
    print("\nC  the cost model, and it must be able to disagree with reality")
    for width, depth, k in ((1024, 16, 8), (1024, 16, 4), (256, 8, 8)):
        if a.quick and width == 1024:
            continue
        c = A.ResearchCandidate(config="carrier", k=k).setup(Ctx(width, depth))
        cm = c.cost_model(depth)
        rows = k * width * 2
        expect = 2 * rows * width * width * depth
        check(f"C: {width}x{depth} k={k} carrier FLOPs = 2*R*n^2*L",
              cm["carrier_flops"] == expect,
              f"{cm['carrier_flops']:,} (R={rows})")
        # MEASURED, so the model can be wrong
        m, _ = mlp_of(width, depth, SEEDS[0])
        ctx = flops.BudgetContext(flop_budget=2 ** 41, quiet=True)
        with ctx:
            c.predict(m, 2 ** 41)
        got = float(ctx.flops_used)
        over = got / expect - 1.0
        # the per-layer row reduction is R*n per layer and is NOT in the model
        red = rows * width * depth
        check(f"C: {width}x{depth} k={k} the measured excess is the row "
              "reduction",
              abs(got - expect - red) / got < 0.02,
              f"measured {got:,.0f}, model {expect:,}, +{over * 100:.3f}%; "
              f"R*n*L = {red:,}")
    check("C: the model DECLARES itself inexact when the analytic branch runs",
          A.ResearchCandidate(config="k2_diag").setup(Ctx(256, 8))
          .cost_model(8)["exact"] is False,
          "an order estimate is labelled as one")

    # ------------------------------------------------------------------ X --
    print("\nX  the single-file conversion is real, not asserted")
    srcs = A.single_file_sources()
    check("X: both sources exist", all(os.path.exists(p) for p in srcs),
          ", ".join(os.path.basename(p) for p in srcs))
    import p2_build_candidate as BUILD
    text = BUILD.build(config="carrier", k=8)
    # IMPORT LINES, not raw text. The generated header legitimately NAMES its
    # sources ("built from local_sim/p2_carrier_fnp.py"), and a substring search
    # flagged that provenance line as a dependency. Provenance in a docstring is
    # the opposite of a dependency.
    imports = [l.strip() for l in text.splitlines()
               if l.strip().startswith(("import ", "from "))]
    allowed = {"import math", "import flopscope as flops",
               "import flopscope.numpy as fnp",
               "from whestbench import BaseEstimator",
               "from __future__ import annotations"}
    extra = [i for i in imports if i not in allowed]
    check("X: the built file's ONLY imports are the four legal ones",
          not extra, "; ".join(extra) if extra
          else f"{len(imports)} import lines, {len(text.encode('utf-8')):,} bytes")
    check("X: and none of them names local_sim or a research module",
          not any(("local_sim" in i or "p2_carrier_fnp" in i
                   or "p2_candidate_architecture" in i or "numpy as np" in i)
                  for i in imports),
          "the property the single-file conversion exists to demonstrate")
    check("X: it subclasses whestbench.BaseEstimator",
          "class Estimator(BaseEstimator)" in text,
          "whest validate refuses anything else")
    check("X: it compiles", _compiles(text), "syntactically valid Python")
    check("X: it carries the sha256 of every source it was built from",
          all(hashlib.sha256(open(p, "rb").read()).hexdigest()[:12] in text
              for p in srcs),
          "so a built artifact names its inputs")

    print()
    if _FAILED:
        print(f"FAILED {len(_FAILED)}: " + "; ".join(_FAILED[:6]))
        raise SystemExit(1)
    print(f"{_TOTAL[0]}/{_TOTAL[0]} checks passed")


def _compiles(text):
    try:
        compile(text, "<built>", "exec")
        return True
    except SyntaxError:
        return False


if __name__ == "__main__":
    main()
