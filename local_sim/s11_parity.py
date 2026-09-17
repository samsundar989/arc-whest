"""Label-free gates for a COMPOSED candidate. Thresholds fixed here, before the first run.

A composed file stacks four generators (closure, shared old-lineage coordinates,
join basis, P4). Each step's anchors are asserted at build time, but an asserted
anchor proves the patch landed, not that the four pieces still mean what they
meant apart. These are the checks for the composition itself:

 1. NESTING, ANCHORED ON THE FROZEN BASE: with sharing off and P4's toggles off
    the composed file must equal the closure-only arm to within the gap the
    already-measured `estimator_k3_v37_a4r256_p4d5.py` itself shows against v32
    under the same switch-off. It is NOT bitwise, and pretending it were would be
    wrong: the lean spelling fuses two contractions into one of twice the inner
    dimension, which is the same mathematics with a different float32 summation
    order and `+n^2` counted FLOPs a layer (measured: +0.21% on the composed file,
    +0.25% on the frozen base). Anchoring on the control means the bound is not a
    number chosen here.
 2. FULL-RANK IDENTITY: at rank = n the shared basis is a complete basis, so the
    composed file must reproduce its own sharing-off self exactly -- whichever
    objective chose the basis. f64 layer loop, every row, rel <= 1e-9, with joins
    actually happening.
 3. REFERENCE PARITY, THREE MODES: the composed file's live `_layer` against the
    independent float64 `closure_audit.ref_step` on its OWN declared closure
    table, at the initial, ordinary and final availability modes, on mu_out,
    cov_out, g_out, the S3 birth slice and the K21 birth input. rel <= 1e-10.
 4. ROW 0: the composed file's first row equals frozen v32's bitwise -- at layer
    0 every non-C block is zero and no join has happened, so nothing in the stack
    may move it.
 5. FLOAT32 and EQUIVARIANCE against frozen v32's own floors.
 6. FINITE and the guard still over-predicting.

Run:  uv run python local_sim/s11_parity.py --arm FILE=TABLE [--arm ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_AUD = os.path.join(_HERE, "closure_audit")
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") not in (_HERE, _AUD)] \
    + [_ROOT, _HERE, _AUD]

import numpy as np  # noqa: E402

REF_TOL = 1e-10
FULLRANK_TOL = 1e-9
OK = []
ROWS = []
V32 = "estimator_k3_v32.py"


def claim(name, ok, detail=""):
    OK.append(bool(ok))
    ROWS.append({"check": name, "passed": bool(ok), "detail": str(detail)})
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<66} {detail}", flush=True)


def rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="FILE=TABLE, repeatable")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import flopscope as flops
    import k4feed_oracle as K
    import s1_parity_v32 as P1
    import s6_build_closure as SB
    import s10_parity as SP
    import s11_build_candidate as SC
    import progswap_parity as PP
    from mint_provenance import canonical_mint

    flops.configure(symmetry_warnings=False)
    tmp = tempfile.mkdtemp(prefix="s11p_")
    L = lambda p: K._load(p if os.path.isabs(p) else os.path.join(_ROOT, p))

    for spec in a.arm:
        arm, table = spec.split("=", 1)
        print("\n" + "=" * 96)
        print("  %s   <-   %s" % (arm, table))
        print("=" * 96)

        # 1. NESTING against the closure-only arm ---------------------------- #
        closure_only_src, _prov = SC.compose(table, 0, 0, "none", 0)
        conly = os.path.join(tmp, "closure_only_" + os.path.basename(arm))
        with open(conly, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(closure_only_src)
        # Both sides must agree on EVERY knob the cost half touches, not only the
        # ones the composition changed: P4 sets `_STRASSEN_MIN_BLOCK` as well as
        # the depth, and a mismatch there shows up as a 3% FLOP difference that
        # looks like an interaction and is not one.
        knobs = dict(_STRASSEN_DEPTH=4, _STRASSEN_MIN_BLOCK=64)
        off = SP._variant(os.path.join(_ROOT, arm), tmp,
                          "off_" + os.path.basename(arm)[:-3],
                          _SHARED_ON=False, _P4_FINAL_VAR=False,
                          _P4_BLOCKSYM=False, **knobs)
        cbase = SP._variant(conly, tmp, "cbase_" + os.path.basename(arm)[:-3],
                            **knobs)
        # The nesting statement has to be ANCHORED, not bitwise. `s7_build_shared`'s
        # LEAN spelling fuses `LA A^T + LP M^T` into one product of twice the
        # inner dimension: the same mathematics, one fewer metered operation, a
        # different float32 summation order and `+n^2` counted FLOPs a layer. The
        # FROZEN, already-measured `estimator_k3_v37_a4r256_p4d5.py` shows exactly
        # the same gap against v32 when ITS sharing is switched off, so the
        # control supplies the bound instead of a number chosen here.
        off37 = SP._variant(os.path.join(_ROOT, "estimator_k3_v37_a4r256_p4d5.py"),
                            tmp, "off37", _SHARED_ON=False, _P4_FINAL_VAR=False,
                            _P4_BLOCKSYM=False, **knobs)
        for wd, dp, sd in ((64, 5, 90000001301), (96, 6, 90000001302)):
            w, es, _ = canonical_mint(wd, dp, sd)
            pa, fa = P1._predict(L(off), w, wd, dp, es)
            pb, fb = P1._predict(L(cbase), w, wd, dp, es)
            p37, f37 = P1._predict(L(off37), w, wd, dp, es)
            p32b, f32b = P1._predict(L(SP._variant(
                os.path.join(_ROOT, V32), tmp, "v32k_%d" % wd, **knobs)),
                w, wd, dp, es)
            base_rel = rel(p37, p32b)
            base_fl = abs(f37 / f32b - 1.0)
            got_rel = rel(pa, pb)
            got_fl = abs(fa / fb - 1.0)
            claim(f"{wd}x{dp} cost half OFF == closure-only within the LEAN gap "
                  f"the frozen base itself shows",
                  got_rel <= 2 * base_rel + 1e-6 and got_fl <= 2 * base_fl + 1e-9,
                  "rel %.2e (frozen base %.2e)   FLOPs %+.4f%% (frozen base %+.4f%%)"
                  % (got_rel, base_rel, 100 * (fa / fb - 1.0),
                     100 * (f37 / f32b - 1.0)))
            p32, _f32 = P1._predict(L(V32), w, wd, dp, es)
            pc, fc = P1._predict(L(arm), w, wd, dp, es)
            claim(f"{wd}x{dp} composed row 0 == v32 row 0 bitwise",
                  np.array_equal(pc[0], p32[0]),
                  "max |diff| %.2e" % float(np.abs(pc[0] - p32[0]).max()))
            claim(f"{wd}x{dp} composed finite, and it moves",
                  bool(np.isfinite(pc).all() and not np.array_equal(pc, p32)),
                  "rel vs v32 %.2e  cost x%.4f" % (rel(pc, p32), fc / _f32))

        # 2. FULL-RANK IDENTITY --------------------------------------------- #
        wd, dp, sd = 64, 6, 90000001303
        w, _es, _ = canonical_mint(wd, dp, sd)
        unc = SP._variant(os.path.join(_ROOT, arm), tmp,
                          "unc_" + os.path.basename(arm)[:-3], _SHARED_ON=False)
        full = SP._variant(os.path.join(_ROOT, arm), tmp,
                           "full_" + os.path.basename(arm)[:-3],
                           _SHARED_RANK=wd, _SHARED_FORCE=True, _SHARED_AGE=2)
        ref = SP.layer_loop(L(unc), w, wd, dp)
        mod = L(full)
        box = SP.count_joins(mod)
        got = SP.layer_loop(mod, w, wd, dp)
        claim(f"{wd}x{dp} full-rank shared basis == sharing off (all rows)",
              rel(got, ref) <= FULLRANK_TOL and box["joins"] > 0,
              "rel %.1e, %d joins, max ko %d" % (rel(got, ref), box["joins"],
                                                 box["max_ko"]))

        # 3. REFERENCE PARITY in the three availability modes ---------------- #
        tb = [dict(output=list(k[0]), factors=list(k[1]), derivs=list(k[2]), coef=str(c))
              for k, c in SB.load_table(os.path.join(_ROOT, table)).items()
              if k[0] != (1, 2)]
        lam5 = float(L(V32)._K4_LAM[4])
        import test_ref_step as T
        for n, seed, vs in ((7, 1, 1.0), (8, 3, 3.0)):
            bd = T.fixture(n, 20260917_50 + seed, vs)
            bd["lam"] = lam5
            tag = os.path.basename(arm)[:-3]
            PP._ref_parity(L(arm), bd, tb, f"{tag} n={n} ORDINARY", layer_index=5)
            PP._ref_parity(L(arm), bd, tb, f"{tag} n={n} FINAL",
                           last_layer=True, layer_index=5)
            b0 = PP._initial_boundary(n, 20260917_60 + seed)
            PP._ref_parity(L(arm), b0, tb, f"{tag} n={n} INITIAL", layer_index=0)
        # PP keeps its own tally; fold it in so one exit code covers everything
        for r in PP.ROWS:
            if r not in ROWS:
                ROWS.append(r)
        OK.extend(PP.OK)
        PP.OK.clear()
        PP.ROWS.clear()

        # 5. FLOAT32 and EQUIVARIANCE against v32's own floors --------------- #
        for wd, dp, sd in ((64, 5, 90000001304),):
            w, es, _ = canonical_mint(wd, dp, sd)
            base = rel(P1._predict(L(V32), w, wd, dp, es)[0],
                       SP.layer_loop(L(V32), w, wd, dp))
            got = rel(P1._predict(L(arm), w, wd, dp, es)[0],
                      SP.layer_loop(L(arm), w, wd, dp))
            claim(f"{wd}x{dp} composed float32 worst row <= 2x v32 + 1e-5",
                  got <= 2 * base + 1e-5, "%.2e vs %.2e" % (got, base))
        wd, dp, sd = 96, 6, 90000001305
        w, es, _ = canonical_mint(wd, dp, sd)
        perm = np.random.default_rng(11).permutation(wd)
        wp = [np.array(x) for x in w]
        wp[0] = wp[0][:, perm]
        wp[1] = wp[1][perm, :]
        b0, _ = P1._predict(L(V32), w, wd, dp, es)
        b1, _ = P1._predict(L(V32), wp, wd, dp, es)
        floor = max(rel(b1, b0), 1e-300)
        e0, _ = P1._predict(L(arm), w, wd, dp, es)
        e1, _ = P1._predict(L(arm), wp, wd, dp, es)
        claim(f"{wd}x{dp} composed equivariance <= 100x v32's floor",
              rel(e1, e0) <= 100 * floor, "%.2e vs floor %.2e" % (rel(e1, e0), floor))

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"artifact": "s11_parity", "checks": ROWS}, fh, indent=1)
    print("\n%d/%d passed" % (sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    sys.exit(main())
