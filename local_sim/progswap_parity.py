"""Label-free gates for the reciprocal complete-program chains, BEFORE any truth read.

Handoff 2026-09-17b section 1. Thresholds are fixed here, before the first run.
Nothing in this file reads truth, scores anything, or promotes anything.

WHAT THE HANDOFF DEMANDS, AND WHICH CHECK PAYS IT
--------------------------------------------------
"Verify all same-program endpoint outputs match the unchanged originals at tiny
and live shapes before consuming truth."
    -> 1. v38_off == frozen v32 bitwise, output AND counted FLOPs.
    -> 2. _diag/estimator_v17_identity.py == native v17 bitwise. This is the
          rewrite machinery's own no-op control: chain 4 is produced by editing
          v17's TERM_SPECS literal, so an editor that perturbs anything would
          make chain 4's loss a transplant artefact rather than a program result.

"Additionally, verify EACH swapped live one-step program against the independent
reference on identical native boundaries: initial availability mode, ordinary
recurrence, and final-layer trimming. Compare pK and resulting covariance/core/
birth inputs, not only means."
    -> 3. chain 3 (`v38_v17prog`) live `_layer` vs `closure_audit.ref_step` on
          the COMPLETE v17 table, in all three modes, comparing mu_out, cov_out,
          g_out, S3 (the birth's third-order slice) and K21 (its bivariate
          birth input) -- not only the mean.
    -> 4. chain 4's compiled availability programs (`_term_prog(0/1/2)`) are
          canonicalised and compared to the correspondingly filtered v32 table,
          which is the same statement for the file whose runtime we cannot
          inject a boundary into.
    -> 5. the INITIAL mode has an exact consequence each transplant must satisfy
          and which no amount of program change can excuse: at layer 0 the input
          is N(0, I), so D3 = D21 = dG = 0 and every extra term carries a zero
          factor. Each transplant must therefore emit a first row IDENTICAL TO
          ITS OWN PARENT'S, bitwise.

          As first written this said "all five chains share row 0", and that is
          WRONG: v32 and v17 are different implementations of the same layer-0
          content and their linear steps accumulate in different orders, so they
          differ by float32 rounding (measured 1.74e-7 on a scale of 0.61 at
          1024x16) without either transplant being at fault. The corrected
          statement -- each transplant against its own parent -- holds EXACTLY,
          at tiny shapes and at 1024x16. The gap between the two natives is
          reported, never gated.

          Similarly, NON-NEGATIVITY of the predictions is reported and not
          gated. A truncated moment expansion of a post-ReLU mean can land
          slightly below zero, and the UNMODIFIED public reference does so at
          1024x16 (79 entries, worst -8.6e-9) before any transplant touches it.
          A gate the control itself fails is not a gate.

"Do not increase numerical tolerances after seeing errors."
    f64 reference parity: 1e-10 relative, the same figure `s6_parity` fixed on
    2026-09-17 for the earlier closure arms. float32 live agreement: chain 3's
    worst row error must be within 2x frozen v32's own float32 floor + 1e-5.

Run:  uv run python local_sim/progswap_parity.py              (tiny shapes, ~1 min)
      uv run python local_sim/progswap_parity.py --live --out <json>   (1024x16)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_AUD = os.path.join(_HERE, "closure_audit")
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") not in (_HERE, _AUD)] \
    + [_ROOT, _HERE, _AUD]

import numpy as np  # noqa: E402

REF_TOL = 1e-10
OK = []
ROWS = []

CHAIN3 = "estimator_k3_v38_v17prog.py"
CHAIN3_LAST = "estimator_k3_v38_v17prog_last.py"
CHAIN3_OFF = "estimator_k3_v38_off.py"
CHAIN4 = "_diag/estimator_v17_v32prog.py"
CHAIN4_ID = "_diag/estimator_v17_identity.py"
V17 = "external/504aldo/estimators/estimator_v17.py"
V29 = "external/504aldo/estimators/estimator_v29.py"
V32 = "estimator_k3_v32.py"


def claim(name, ok, detail=""):
    OK.append(bool(ok))
    ROWS.append({"check": name, "passed": bool(ok), "detail": str(detail)})
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<62} {detail}", flush=True)


def rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


# --------------------------------------------------------------------------- #

def _tables():
    import canon
    import s6_build_closure as B
    v32_terms, _ = canon.canon_ours(os.path.join(_ROOT, V32))
    v17_terms, _ = canon.canon_public(os.path.join(_ROOT, V17))
    t17 = [dict(output=list(k[0]), factors=list(k[1]), derivs=list(k[2]), coef=str(c))
           for k, c in v17_terms.items() if k[0] != (1, 2)]
    t32 = [dict(output=list(k[0]), factors=list(k[1]), derivs=list(k[2]), coef=str(c))
           for k, c in v32_terms.items() if k[0] != (1, 2)]
    del B
    return v32_terms, v17_terms, t32, t17


def _initial_boundary(n, seed):
    """The layer-0 boundary EXACTLY as the live chain sees it: x ~ N(0, I).

    mu = 0 and cov = I before the linear step, so mu_pre = 0 and cov_pre = W^T W;
    the carried core is zero and lambda is off the front of the table, so
    dG = 0 -- hence D3 = D21 = D4 = D22 = Q31 = 0. This is the INITIAL
    AVAILABILITY MODE, and it is a boundary, not a fixture.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n, n)) * np.sqrt(2.0 / n)
    return {"mu_pre": np.zeros(n), "cov_pre": W.T @ W,
            "D3": np.zeros(n), "D21": np.zeros((n, n)),
            "dG": np.zeros(n), "metric": 2.0, "lam": 0.0}


def _ref_parity(mod, bd, table, tag, *, last_layer=False, layer_index=0):
    """One live `_layer` against the float64 reference on the SAME boundary."""
    import ref_step
    import test_ref_step as T
    ref = ref_step.step(bd, table, want_families=True)
    live = T.run_live(mod, bd, dtype=None, layer_index=layer_index,
                      last_layer=last_layer)
    n = bd["mu_pre"].shape[0]
    off = 1.0 - np.eye(n)
    claim(f"{tag} mu_out", rel(live["mu_out"], ref["mu_out"]) <= REF_TOL,
          "%.1e" % rel(live["mu_out"], ref["mu_out"]))
    if last_layer:
        # The final layer forms nothing else: the grader reads post-ReLU means.
        return
    mu, sigma = bd["mu_pre"], np.sqrt(np.maximum(np.diag(bd["cov_pre"]), 1e-12))
    w1, w2 = ref_step.wick(1, 1, mu, sigma), ref_step.wick(2, 1, mu, sigma)
    C11 = bd["cov_pre"] * off
    fac1 = w1[:, None] * C11
    p111 = ((w1[:, None] ** 2 * w1[None, :]) * bd["D21"]
            + ((fac1 * fac1) * w2[None, :]) * off)
    K21 = live["R"] + (live["eb"][:, None] * C11) * w1[None, :] + p111
    for key, a, b in (("cov_out", live["cov_out"], ref["cov_out"]),
                      ("g_out", live["g_out"], ref["g_out"]),
                      ("S3_birth", live["sb"], ref["K"][(3,)] - w1 ** 3 * bd["D3"]),
                      ("K21_birth", K21, ref["K"][(2, 1)])):
        claim(f"{tag} {key}", rel(a, b) <= REF_TOL, "%.1e" % rel(a, b))


# --------------------------------------------------------------------------- #

def tiny():
    import flopscope as flops
    import canon
    import k4feed_oracle as K
    import s1_parity_v32 as P1
    from mint_provenance import canonical_mint

    flops.configure(symmetry_warnings=False)
    L = lambda name: K._load(os.path.join(_ROOT, name))
    v32_terms, v17_terms, t32, t17 = _tables()

    print("\n1. ENDPOINT PARITY -- the unchanged originals (tiny shapes)")
    m32, moff = L(V32), L(CHAIN3_OFF)
    m17, mid = L(V17), L(CHAIN4_ID)
    for wd, dp, sd in ((64, 5, 90000000801), (96, 6, 90000000802)):
        w, es, _ = canonical_mint(wd, dp, sd)
        pb, fb = P1._predict(m32, w, wd, dp, es)
        po, fo = P1._predict(moff, w, wd, dp, es)
        claim(f"{wd}x{dp} v38_off == v32 bitwise + FLOPs",
              np.array_equal(pb, po) and fb == fo, f"flops {fo:.0f} vs {fb:.0f}")
        p17, f17 = P1._predict(m17, w, wd, dp, es)
        pid, fid = P1._predict(mid, w, wd, dp, es)
        claim(f"{wd}x{dp} v17_identity == v17 bitwise + FLOPs",
              np.array_equal(p17, pid) and f17 == fid, f"flops {fid:.0f} vs {f17:.0f}")
        m4 = L(CHAIN4)
        p4, f4 = P1._predict(m4, w, wd, dp, es)
        claim(f"{wd}x{dp} chain4 differs from v17 (not a null edit)",
              not np.array_equal(p4, p17), "rel %.2e" % rel(p4, p17))
        claim(f"{wd}x{dp} chain4 row 0 == v17 row 0 (initial mode identical)",
              np.array_equal(p4[0], p17[0]))
        m3 = L(CHAIN3)
        p3, f3 = P1._predict(m3, w, wd, dp, es)
        claim(f"{wd}x{dp} chain3 differs from v32 (not a null edit)",
              not np.array_equal(p3, pb), "rel %.2e" % rel(p3, pb))
        claim(f"{wd}x{dp} chain3 row 0 == v32 row 0 (initial mode identical)",
              np.array_equal(p3[0], pb[0]))
        claim(f"{wd}x{dp} chain3 finite and non-negative",
              bool(np.isfinite(p3).all() and (p3 >= 0).all()))

    print("\n2. CHAIN 3 vs the independent f64 reference, THREE availability modes")
    import test_ref_step as T
    # `_layer` does NOT read a lambda off the boundary: it takes
    # `_K4_LAM[layer_index - 1]`, and that value feeds kappa_iiij = (m/2) lam C,
    # which the COMPLETE v17 program consumes and the BUDGET tables never did.
    # A fixture whose `lam` disagrees with its layer index therefore asks the
    # reference a different question from the one the live path answers. Fixed
    # HERE, before the first run, as a harness-consistency requirement -- the
    # tolerance stays 1e-10 and the fixtures agree to ~1e-15 once the two
    # lambdas are the same number.
    lam5 = float(L(CHAIN3)._K4_LAM[4])
    for n, seed, vs in ((7, 1, 1.0), (9, 2, 0.15), (8, 3, 3.0)):
        mod = L(CHAIN3)
        bd = T.fixture(n, 20260917_10 + seed, vs)
        bd["lam"] = lam5
        _ref_parity(mod, bd, t17, f"n={n} ORDINARY", layer_index=5)
        mod = L(CHAIN3)
        _ref_parity(mod, bd, t17, f"n={n} FINAL", last_layer=True, layer_index=5)
        mod = L(CHAIN3)
        b0 = _initial_boundary(n, 20260917_20 + seed)
        _ref_parity(mod, b0, t17, f"n={n} INITIAL", layer_index=0)
        # and, at the initial boundary, the COMPLETE program must agree with
        # v32's OWN program: every extra term carries a zero factor there.
        mod = L(CHAIN3)
        _ref_parity(mod, b0, t32, f"n={n} INITIAL(vs v32 table)", layer_index=0)

    print("\n3. CHAIN 4's compiled availability programs, canonicalised")
    m4 = L(CHAIN4)
    for mode, drop in ((0, canon_mode0_drop()), (1, set()), (2, {"Q31", "Q13"})):
        got = canon_from_prog(m4, mode)
        want = {k: c for k, c in v32_terms.items()
                if k[0] != (1, 2) and not (set(k[1]) & drop)}
        got = {k: c for k, c in got.items() if k[0] != (1, 2)}
        claim(f"chain4 _term_prog({mode}) == v32 program filtered for that mode",
              got == want, f"{len(got)} terms vs {len(want)}")
    m17b = L(V17)
    for mode in (0, 1, 2):
        got = {k: c for k, c in canon_from_prog(m17b, mode).items() if k[0] != (1, 2)}
        n_extra = len([k for k in got if k not in v32_terms])
        claim(f"native v17 _term_prog({mode}) still carries its extras",
              n_extra > 0 or mode == 0, f"{len(got)} terms, {n_extra} beyond v32")

    print("\n4. FLOAT32 behaviour of chain 3 against v32's own floor")
    import s6_parity as SP
    for wd, dp, sd in ((64, 5, 90000000803),):
        w, es, _ = canonical_mint(wd, dp, sd)
        base = SP.rel(P1._predict(L(V32), w, wd, dp, es)[0],
                      SP.layer_loop(L(V32), w, wd, dp))
        got = SP.rel(P1._predict(L(CHAIN3), w, wd, dp, es)[0],
                     SP.layer_loop(L(CHAIN3), w, wd, dp))
        claim(f"{wd}x{dp} chain3 float32 worst row <= 2x v32 + 1e-5",
              got <= 2 * base + 1e-5, "%.2e vs %.2e" % (got, base))


def declared(pairs):
    """Three-mode f64 reference parity for ANY (arm file, declared table) pair.

    Used for our OWN derived closures as well as the transplanted one: the gate
    is identical, so an arm we derived and an arm we transplanted are held to
    the same standard.
    """
    import flopscope as flops
    import k4feed_oracle as K
    import s6_build_closure as SB
    import test_ref_step as T
    from mint_provenance import canonical_mint
    import s1_parity_v32 as P1

    flops.configure(symmetry_warnings=False)
    L = lambda name: K._load(os.path.join(_ROOT, name))
    m32 = L(V32)
    lam5 = float(L(V32)._K4_LAM[4])
    for arm, table in pairs:
        tb = [dict(output=list(k[0]), factors=list(k[1]), derivs=list(k[2]), coef=str(c))
              for k, c in SB.load_table(os.path.join(_ROOT, table)).items()
              if k[0] != (1, 2)]
        print("  %s  <-  %s  (%d oriented terms)" % (arm, table, len(tb)))
        for n, seed, vs in ((7, 1, 1.0), (8, 3, 3.0)):
            bd = T.fixture(n, 20260917_30 + seed, vs)
            bd["lam"] = lam5
            _ref_parity(L(arm), bd, tb, f"{os.path.basename(arm)[:-3]} n={n} ORDINARY",
                        layer_index=5)
            _ref_parity(L(arm), bd, tb, f"{os.path.basename(arm)[:-3]} n={n} FINAL",
                        last_layer=True, layer_index=5)
            b0 = _initial_boundary(n, 20260917_40 + seed)
            _ref_parity(L(arm), b0, tb, f"{os.path.basename(arm)[:-3]} n={n} INITIAL",
                        layer_index=0)
        for wd, dp, sd in ((64, 5, 90000000811),):
            w, es, _ = canonical_mint(wd, dp, sd)
            pb, _fb = P1._predict(m32, w, wd, dp, es)
            pa, _fa = P1._predict(L(arm), w, wd, dp, es)
            claim(f"{os.path.basename(arm)[:-3]} {wd}x{dp} row 0 == v32 row 0 (initial mode)",
                  np.array_equal(pa[0], pb[0]))
            claim(f"{os.path.basename(arm)[:-3]} {wd}x{dp} finite, non-negative, moves",
                  bool(np.isfinite(pa).all() and (pa >= 0).all()
                       and not np.array_equal(pa, pb)),
                  "rel vs v32 %.2e" % rel(pa, pb))


def canon_mode0_drop():
    """The blocks v17's mode 0 does not have. v32's layer 0 has them all zero."""
    return {"D21", "D12", "D22", "D3", "D3i", "D3j", "D4", "D4i", "D4j",
            "Q31", "Q13"}


def canon_from_prog(mod, mode):
    """Canonicalise a v17-shaped module's COMPILED program for one mode.

    Reads `_term_prog(mode)`'s own `AB2`/`SL2`/`SR2`/`IND2` (and the univariate
    tables) rather than the source literal, so it reports what the file will
    actually execute.
    """
    import canon
    from fractions import Fraction
    prog = mod._term_prog(mode)
    pairs = mod.WICK_PAIRS
    acc = {}
    for t, (a, b) in enumerate(prog["AB2"]):
        g = prog["IND2"][t].index(1.0)
        ip = mod.D2_IPS[g]
        wl = [i for i, v in enumerate(prog["SL2"][t]) if v != 0.0][0]
        wr = [i for i, v in enumerate(prog["SR2"][t]) if v != 0.0][0]
        coef = canon.frac(prog["SL2"][t][wl])
        (kl, _), (kr, _) = pairs[wl], pairs[wr]
        factors = [canon._V17_BLOCK[x] for x in (a, b)
                   if canon._V17_BLOCK[x] is not None]
        if "wk431" in (a, b):
            factors = ["Q31" if kl == 3 else "Q13" if f == "Q13" else f
                       for f in factors]
        canon.add_term(acc, ip, factors, [kl, kr], coef)
    for t, (a, b) in enumerate(prog["B1"]):
        g = prog["IND1"][t].index(1.0)
        ip = mod.D1_IPS[g]
        wl = [i for i, v in enumerate(prog["SL1"][t]) if v != 0.0][0]
        coef = canon.frac(prog["SL1"][t][wl])
        kl, _ = pairs[wl]
        factors = [canon._V17_BLOCK[x] for x in (a, b)
                   if canon._V17_BLOCK[x] is not None]
        canon.add_term(acc, ip, factors, [kl], coef)
    del Fraction
    return canon.finish(acc, {(1, 1), (2, 2)})


# --------------------------------------------------------------------------- #

def live(out_path):
    """The same statements at the LIVE 1024x16 shape. Heavy: run under workerq."""
    import flopscope as flops
    import k4feed_oracle as K
    import same_state as SS
    import ref_step
    import test_ref_step as T
    from mint_provenance import canonical_mint

    flops.configure(symmetry_warnings=False)
    L = lambda name: K._load(os.path.join(_ROOT, name))
    v32_terms, v17_terms, t32, t17 = _tables()
    n, depth, seed = 1024, 16, SS.SEED
    w, es, _ = canonical_mint(n, depth, seed)

    print("\nA. ENDPOINT PARITY at 1024x16 (one canonical network)")
    import s1_parity_v32 as P1
    pb, fb = P1._predict(L(V32), w, n, depth, es)
    po, fo = P1._predict(L(CHAIN3_OFF), w, n, depth, es)
    claim("1024x16 v38_off == v32 bitwise + FLOPs",
          np.array_equal(pb, po) and fb == fo, f"flops {fo:.0f} vs {fb:.0f}")
    p17, f17 = P1._predict(L(V17), w, n, depth, es)
    pid, fid = P1._predict(L(CHAIN4_ID), w, n, depth, es)
    claim("1024x16 v17_identity == v17 bitwise + FLOPs",
          np.array_equal(p17, pid) and f17 == fid, f"flops {fid:.0f} vs {f17:.0f}")
    p3, f3 = P1._predict(L(CHAIN3), w, n, depth, es)
    p4, f4 = P1._predict(L(CHAIN4), w, n, depth, es)
    # FINITENESS is a gate. NON-NEGATIVITY IS NOT, and the first version of this
    # module was wrong to make it one: a truncated moment expansion of a
    # post-ReLU mean can land slightly below zero, and the UNMODIFIED public
    # reference does so at 1024x16 (79 entries, worst -8.6e-9) before any
    # transplant touches it. A gate the control fails is not a gate. The
    # negativity is REPORTED per arm instead, because how far a chain goes
    # negative is an accuracy diagnostic worth having, and the grader scores the
    # MSE either way.
    claim("1024x16 chain3 finite", bool(np.isfinite(p3).all()))
    claim("1024x16 chain4 finite", bool(np.isfinite(p4).all()))
    negativity = {}
    for nm, arr in (("v32", pb), ("v17", p17), ("chain3", p3), ("chain4", p4)):
        negativity[nm] = {"n_negative": int((arr < 0).sum()),
                          "worst": float(arr.min()),
                          "worst_over_max": float(abs(arr.min()) / max(float(arr.max()), 1e-300))}
        print("   negativity %-7s %4d entries, worst %.3e (%.1e of max)"
              % (nm, negativity[nm]["n_negative"], negativity[nm]["worst"],
                 negativity[nm]["worst_over_max"]), flush=True)

    # ROW 0, stated as the claim that actually means something: each transplant
    # must match ITS OWN PARENT bitwise, because at layer 0 the input is
    # N(0, I), every non-C block is zero, and the programs coincide. Lumping all
    # five together was wrong: v32 and v17 are DIFFERENT IMPLEMENTATIONS of the
    # same layer-0 content and differ by float32 rounding in their own linear
    # steps, which says nothing about either transplant.
    claim("1024x16 chain3 row 0 == v32 row 0 BITWISE",
          np.array_equal(p3[0], pb[0]),
          "max |diff| %.3e" % float(np.abs(p3[0] - pb[0]).max()))
    claim("1024x16 chain4 row 0 == v17 row 0 BITWISE",
          np.array_equal(p4[0], p17[0]),
          "max |diff| %.3e" % float(np.abs(p4[0] - p17[0]).max()))
    print("   the two NATIVE chains' row 0 differ by %.3e on a scale of %.3e "
          "(float32 rounding between two implementations, reported not gated)"
          % (float(np.abs(pb[0] - p17[0]).max()), float(np.abs(pb[0]).max())),
          flush=True)
    costs = {"v32": fb, "chain3": f3, "chain4": f4, "v17": f17}
    print("  counted FLOPs:", {k: "%.4g" % v for k, v in costs.items()})

    print("\nB. CHAIN 3 vs the f64 reference on CAPTURED LIVE boundaries")
    os.environ["SS_LAYERS"] = "0,3,8,15"
    import importlib
    importlib.reload(SS)
    caps, _p32 = SS.capture_v32(w)
    live_rows = []
    for l in sorted(caps):
        bd = dict(caps[l])
        last = (l == depth - 1)
        mod = L(CHAIN3)
        tag = ("INITIAL" if l == 0 else "FINAL" if last else "ORDINARY")
        _ref_parity(mod, bd, t17, f"L{l} {tag} f64", last_layer=last, layer_index=l)
        # float32, the dtype the live chain runs in: report, do not gate on the
        # reference tolerance -- the f32 floor is v32's own and was measured on
        # 2026-09-17 (mu 1.8-2.5e-6, cov 3.4-3.8e-5, g 0.85-1.3e-4).
        mod = L(CHAIN3)
        import flopscope.numpy as fnp
        lv = T.run_live(mod, bd, dtype=fnp.float32, layer_index=l, last_layer=last)
        rf = ref_step.step(bd, t17, want_families=False)
        live_rows.append({"layer": l, "mode": tag, "dtype": "f32",
                          "mu_out": rel(lv["mu_out"], rf["mu_out"]),
                          "cov_out": (None if last else
                                      rel(lv["cov_out"], rf["cov_out"])),
                          "g_out": (None if last else
                                    rel(lv["g_out"], rf["g_out"]))})
        print("   f32 L%d %s" % (l, live_rows[-1]), flush=True)

    print("\nC. CHAIN 4 on its OWN native boundaries vs the f64 v32-program reference")
    import tempfile
    caps4, pred4, diag_sha, _v17sha = SS.capture_v17(
        w, tempfile.mkdtemp(prefix="progswap_"), source=os.path.join(_ROOT, CHAIN4))
    c4_rows = []
    for l in sorted(caps4):
        bd = caps4[l]
        rf = ref_step.step(bd, t32, want_families=False)
        r = rel(rf["mu_out"], pred4[l])
        c4_rows.append({"layer": l, "mu_out_vs_own_row": r})
        claim(f"chain4 L{l} own row == f64 reference on the v32 program",
              r <= 3e-4, "%.1e (float32 chain vs float64 reference)" % r)
    caps17, pred17, _s, _s2 = SS.capture_v17(
        w, tempfile.mkdtemp(prefix="progswap17_"), source=os.path.join(_ROOT, V17))
    for l in sorted(caps17):
        rf = ref_step.step(caps17[l], t17, want_families=False)
        r = rel(rf["mu_out"], pred17[l])
        claim(f"native v17 L{l} own row == f64 reference on ITS program",
              r <= 3e-4, "%.1e" % r)

    payload = {"artifact": "progswap_parity.live", "label_free": True,
               "negativity": negativity,
               "native_row0_float32_gap": float(np.abs(pb[0] - p17[0]).max()),
               "network": {"width": n, "depth": depth, "seed": seed},
               "counted_flops": {k: float(v) for k, v in costs.items()},
               "checks": ROWS, "f32_rows": live_rows, "chain4_rows": c4_rows,
               "chain4_debug_copy_sha256": diag_sha,
               "sources": {p: _sha(os.path.join(_ROOT, p))
                           for p in (V32, V17, CHAIN3, CHAIN3_OFF, CHAIN4,
                                     CHAIN4_ID)}}
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=1)
        print("wrote", out_path)


def _sha(path):
    import hashlib
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--declared", action="append", default=[],
                    help="ARM=TABLE, repeatable: three-mode reference parity for "
                         "an arm against its own declared closure table")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    print("=" * 96)
    print("RECIPROCAL PROGRAM-SWAP PARITY -- label-free, decides nothing, gates the truth read")
    print("=" * 96)
    if a.declared:
        declared([tuple(x.split("=", 1)) for x in a.declared])
    elif a.live:
        live(a.out)
    else:
        tiny()
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
                json.dump({"artifact": "progswap_parity.tiny", "checks": ROWS},
                          fh, indent=1)
    print("\n%d/%d passed" % (sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    sys.exit(main())
