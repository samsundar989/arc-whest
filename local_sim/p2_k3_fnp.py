"""WP14 / C0-ii: the `fnp` port of the regenerating K3 flow. THE CHASSIS.

Contract: `experiments/LOG.md`, 2026-09-02, "CONTRACT (Rule 1): WP14", HEAD
`d915f83`. **No gate in that contract scores this module.** It is on the critical
path whatever `C1`-`C4` decide, because every branch this project has left ends in
a file the grader can run, and the grader sandbox has no torch.

**THIS PORT IS INCOMPLETE AND THIS DOCSTRING IS THE HONEST STATUS.** What is
finished is stated below; what is not is listed, by name, at the bottom. Nothing
here is imported by `estimator.py` and nothing here has been submitted.

WHY THIS FILE IS THE CRITICAL PATH, AND WHY IT IS NOW WORTH BUILDING
---------------------------------------------------------------------
`C0-iii` (2026-09-02) measured that a regenerating K3 flow at **float32** costs
`99,116,660,736` counted flopscope FLOPs, and that composed with the f32 K2
backbone this repository already measured and reviewed on 2026-08-26
(`68,802,510,336`, accuracy-neutral on 96/96 networks) the total is
`167,919,171,072` = **`0.07636 B`** -- the `0.1` multiplier floor with `0.02364 B`
to spare, with no Strassen and no rank cap.

That composition is a **projection**: the two halves are measured in different
spellings and have never been executed as one program. **This module is what turns
it into a measurement.** Until it runs end to end under `whest validate`, through
`certify_candidate`, and through a brokered residual soak, the `0.07636 B` is
arithmetic and not a candidate.

THE PORT IS BOUNDED, AND THAT IS THE FIRST RESULT HERE
-------------------------------------------------------
Upstream's nonlinear step sums over a term list produced by
`kprop_harmonic.get_all_terms_iso(k_max=3, d_max=4, augment=False)`, filtered by
`factored_keeps_term`. That list depends on **nothing but `(k_max, d_max,
augment)`** -- not on `n`, not on the weights, not on the data. So it can be
enumerated once, frozen, and checked, which converts an open-ended
"re-derive a research codebase in a restricted dialect" into a bounded
"implement 25 named contractions".

Enumerated 2026-09-02 by `local_sim/p2_k3_terms.py` against upstream rev
`93d091a4`:

    35 terms enumerated, 25 kept in SIMPLE mode

    int_part (1,)     3 vec_parts        int_part (1, 1)   4 vec_parts
    int_part (2,)     3 vec_parts        int_part (2, 1)   5 vec_parts
    int_part (3,)     3 vec_parts        int_part (2, 2)   4 vec_parts
    int_part (4,)     3 vec_parts

Artifact `experiments/2026-09-02_wp14_fnp/k3_terms_simple.json`,
sha256 `824358c01b891ce15d3d45755746d5459181abd303ebf9e9ad56af807c75b2a3`.
`--check-terms` re-enumerates against the live upstream checkout and refuses if
the table has drifted, so the frozen copy cannot rot silently.

The `(1,1,1)` int_part is NOT in that list: upstream factors it by hand, and in
SIMPLE mode it is exactly two things -- the wick-scaled incoming `kappa_3`, and
ONE two-leg path block, `legs = [(1, 1, WK_11, 1)]` with `rights` the same single
entry (`factor_k3.py:378-403`). That is the whole of the third-order state update,
and it is why a regenerating flow is three matmuls a layer.

THE ONE PLACE A PORT LIKE THIS GOES WRONG
------------------------------------------
`local_sim/p2_slice_cost.py::_third_order_layer` already writes this layer in
`fnp` -- but it is a COST probe, and its own docstring says "values are
arbitrary". It uses `w1` for both wick factors where upstream uses `w(1)` on one
leg and `w(2) * w(1)` on the other (`factor_k3.py:401-403`). Its FLOP count is
therefore right and its numbers are wrong. **Do not promote that spelling by
copying it.** This module re-derives from upstream and checks against upstream.

TWO VENVS, AND THEY ARE NOT INTERCHANGEABLE
--------------------------------------------
`--check-terms` re-enumerates against the upstream checkout, which imports torch,
so it runs in **`.venv-gpu`** -- which has no flopscope. `--k2-parity` and
`--price` meter real flopscope operations, so they run in **`.venv`** -- which has
no torch. That split is the whole reason this port exists: the grader sandbox is
the `.venv` side, and nothing that needs torch can ever ship.

Run:
    .venv-gpu\Scripts\python.exe local_sim/p2_k3_fnp.py --check-terms
    uv run python local_sim/p2_k3_fnp.py --k2-parity --width 128 --depth 4
    uv run python local_sim/p2_k3_fnp.py --price
    uv run python local_sim/p2_k3_fnp.py --todo
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# APPEND, never insert: `local_sim/` holds numpy shims named `flopscope` and
# `whestbench`, and prepending this directory turns every counted FLOP below into
# fiction.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

WIDTH, DEPTH = 1024, 16
BUDGET = 2 ** 41

#: `experiments/2026-09-02_wp14_fnp/k3_terms_simple.json`, frozen 2026-09-02.
TERMS_ARTIFACT = os.path.join(_ROOT, "experiments", "2026-09-02_wp14_fnp",
                              "k3_terms_simple.json")
TERMS_SHA256 = "824358c01b891ce15d3d45755746d5459181abd303ebf9e9ad56af807c75b2a3"

#: C0-iii's measured halves. Both are MEASURED; their SUM is a projection.
K2_ANCHOR_F64 = 137_600_326_656
K2_ANCHOR_F32 = 68_802_510_336
K3_INCREMENT_F64 = 198_233_321_472
K3_INCREMENT_F32 = 99_116_660_736

_INV_SQRT_2PI = 0.3989422804014327


def _require_real_flopscope(flops) -> None:
    path = os.path.abspath(getattr(flops, "__file__", "") or "")
    if path.startswith(os.path.abspath(_HERE) + os.sep):
        raise RuntimeError(
            f"REFUSED: `flopscope` resolved to the local shim at {path}. Every "
            "counted FLOP below would be fiction.")
    if not hasattr(flops, "BudgetContext"):
        raise RuntimeError(f"flopscope at {path} has no BudgetContext.")


# ==========================================================================
# the K2 backbone, in fnp, at a declared dtype
# ==========================================================================

def k2_closure(fnp, w, width: int, depth: int, dtype):
    """The shipped `estimator.py::_closure`, parameterised by dtype.

    Reproduced here rather than imported because `estimator.py` is the shipped
    file and is not touched by research code -- and because the dtype is the
    whole point: `docs/CURRENT_STATE.md` (2026-08-26, independently reviewed)
    records that this closure in float32 costs `0.50002x` its float64 self and
    is accuracy-neutral on 96/96 networks, with the `var_pre` clamp firing zero
    times and zero non-finite outputs.

    **The float32 trap this must avoid**, and the reason the two `.astype` calls
    below are load-bearing: flopscope 0.12 promotes `stats.norm.cdf` / `.pdf`
    inputs to float64 "to match scipy.stats" and emits a warning. An arm that
    does not cast the results back is float32 in name only -- `Phi` and `phi`
    become float64 and so does the covariance accumulator, where nearly all the
    cost lives. That is exactly how `local_sim/p2_arms/arm_k2p1_f32.py` measured
    a 3.1% saving instead of 50%.

    Returns `(mus, phis)`: the per-layer post-ReLU means and the per-layer
    `Phi = Prob(pre-activation > 0)` the K3 step's wick coefficients need.
    """
    import flopscope as flops

    mu = fnp.zeros(width, dtype=dtype)
    cov = fnp.eye(width, dtype=dtype)
    off_mask = fnp.ones((width, width), dtype=dtype) - fnp.eye(width, dtype=dtype)
    mus, phis, sigs, alphas = [], [], [], []
    for wl in w:
        mu_pre = wl.T @ mu
        cov_pre = wl.T @ cov @ wl
        var_pre = fnp.maximum(fnp.diagonal(cov_pre), dtype(1e-12))
        sig = fnp.sqrt(var_pre)
        alpha = mu_pre / sig
        Phi = flops.stats.norm.cdf(alpha).astype(dtype)
        phi = flops.stats.norm.pdf(alpha).astype(dtype)
        mu = sig * (alpha * Phi + phi)
        ez2 = (var_pre + mu_pre * mu_pre) * Phi + mu_pre * sig * phi
        var = fnp.maximum(ez2 - mu * mu, dtype(0.0))
        cov = fnp.outer(Phi, Phi) * cov_pre * off_mask + fnp.diag(var)
        mus.append(mu)
        phis.append(Phi)
        sigs.append(sig)
        alphas.append(alpha)
    return {"mus": mus, "phis": phis, "sigs": sigs, "alphas": alphas}


# ==========================================================================
# the frozen term table, and the check that it has not rotted
# ==========================================================================

def load_terms() -> dict:
    with open(TERMS_ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


def cmd_check_terms(a) -> int:
    """Re-enumerate against the LIVE upstream checkout and refuse on drift.

    A frozen table is a liability the moment nobody checks it. This is the check,
    and it is why the table may be trusted as a literal in the port below.
    """
    import hashlib

    doc = load_terms()
    sha = hashlib.sha256(open(TERMS_ARTIFACT, "rb").read()).hexdigest()
    print(f"  artifact   {os.path.relpath(TERMS_ARTIFACT, _ROOT)}")
    print(f"  sha256     {sha}")
    print(f"  matches    {sha == TERMS_SHA256}")
    if sha != TERMS_SHA256:
        print("  REFUSED: the frozen term table has changed on disk.")
        return 1

    import p2_kprop_bridge as bridge

    src = bridge._src()
    v = bridge.verify_upstream(src)
    if not v["pass"]:
        print("  REFUSED: upstream is not rev 93d091a4 plus the recorded patch.")
        return 1
    from mlp_kprop.kprop_harmonic import factored_keeps_term, get_all_terms_iso

    live = get_all_terms_iso(k_max=3, d_max=4, augment=False)
    rows = []
    for int_part, vpd in live.items():
        for vec_part, count in vpd.items():
            keep = (len(int_part) <= 3
                    and int_part not in [(3, 1), (2, 1, 1)]
                    and int_part != (1, 1, 1)
                    and (int_part, set(vec_part)) != ((2, 1, 1), {(1, 1, 1)})
                    and factored_keeps_term(3, int_part, vec_part))
            rows.append((tuple(int_part), tuple(map(tuple, vec_part)), count, keep))
    frozen = {(tuple(r["int_part"]), tuple(map(tuple, r["vec_part"])), r["count"],
               r["kept_by_factored_simple"]) for r in doc["terms"]}
    same = frozen == set(rows)
    n_kept = sum(1 for r in rows if r[3])
    print(f"  upstream   {src}  rev 93d091a4 VERIFIED")
    print(f"  enumerated {len(rows)} terms, {n_kept} kept in SIMPLE mode")
    print(f"  identical to the frozen table: {same}")
    return 0 if same else 1


# ==========================================================================
# parity and pricing
# ==========================================================================

def cmd_k2_parity(a) -> int:
    """The fnp K2 backbone against `p2_cal_lab`'s NumPy mirror, at both dtypes.

    This is the half of the chassis that is FINISHED. It is checked against the
    mirror that is itself verified to `4e-30` relative mean square against the
    instrumented arm this project ships, so agreement here means the wiring is
    right -- not that K3 is.
    """
    import numpy as np

    import flopscope as flops
    import flopscope.numpy as fnp

    _require_real_flopscope(flops)
    from mint_provenance import canonical_mint
    import p2_cal_lab as cal

    w_np, _e, _s = canonical_mint(a.width, a.depth, a.seed)
    # `p2_cal_lab.closure` returns (depth, width) post-ReLU means, layer 0 first,
    # and is itself verified to 4e-30 relative mean square against the
    # instrumented arm this project ships. It is the reference, not a rewrite.
    ref_mu = cal.closure(w_np)[-1]

    rows = []
    for dt_name, dt in (("f64", fnp.float64), ("f32", fnp.float32)):
        w = [fnp.array(np.asarray(x, dtype=np.float64)).astype(dt) for x in w_np]
        with flops.BudgetContext(flop_budget=10 * BUDGET, quiet=True) as ctx:
            out = k2_closure(fnp, w, a.width, a.depth, dt)
        used = float(ctx.flops_used)
        mu = np.asarray(out["mus"][-1], dtype=np.float64)
        rel = float(np.sqrt(((mu - ref_mu) ** 2).mean()
                            / max((ref_mu ** 2).mean(), 1e-300)))
        rows.append({"dtype": dt_name, "flops": used, "x_budget": used / BUDGET,
                     "rel_rms_vs_mirror": rel})
        print(f"  {dt_name}   counted {used:>18,.0f}   = {used / BUDGET:.6f} B"
              f"   rel-rms vs the NumPy mirror {rel:.3e}")
    if len(rows) == 2:
        print(f"  f32 / f64 cost ratio  {rows[1]['flops'] / rows[0]['flops']:.5f}"
              "   (0.50002 is the reviewed grader-spelling figure)")
    if a.out:
        _dump(a.out, {"check": "WP14 C0-ii: fnp K2 backbone parity and price",
                      "contract": "LOG 2026-09-02 CONTRACT WP14, HEAD d915f83",
                      "width": a.width, "depth": a.depth, "seed": a.seed,
                      "rows": rows,
                      "reference": "local_sim/p2_cal_lab.closure (NumPy mirror)"})
    return 0


def cmd_price(a) -> int:
    """Where the composed projection stands, printed with its own caveat."""
    total_f32 = K2_ANCHOR_F32 + K3_INCREMENT_F32
    total_f64 = K2_ANCHOR_F64 + K3_INCREMENT_F64
    floor = int(0.1 * BUDGET)
    print(f"  budget 2**41                              {BUDGET:>18,}")
    print(f"  the 0.1 multiplier floor                  {floor:>18,}")
    print()
    for nm, k2, k3 in (("all f64 (WP13's record)", K2_ANCHOR_F64, K3_INCREMENT_F64),
                       ("both f32 (C0-iii)", K2_ANCHOR_F32, K3_INCREMENT_F32)):
        t = k2 + k3
        print(f"  {nm:<26} K2 {k2:>15,}  K3 {k3:>15,}")
        print(f"  {'':<26} total {t:>13,}  = {t / BUDGET:.5f} B"
              f"   multiplier {max(0.1, t / BUDGET):.5f}")
    head = floor - total_f32
    print()
    print(f"  headroom under the floor at f32           {head:>18,}  "
          f"= {head / BUDGET:.5f} B")
    # 16 layers, not 15. SKEPTIC WP14 caught this dividing by 15 and reporting
    # 1.61 matmuls/layer of headroom where the figure is 1.51.
    print(f"  in f32 dense 1024-matmuls                 {head / 2_146_435_072:>18.2f}"
          f"   ({head / 2_146_435_072 / 16:.2f} per layer, over 16 layers)")
    print()
    print("  CAVEAT, and it is the reason this module exists: the two halves are")
    print("  measured in DIFFERENT spellings and their sum has never been executed")
    print("  as one program. It implies nothing about whest validate, the 4 GiB")
    print(f"  single-array rule, the 8 GiB cap or the 0.4 s residual cliff.")
    return 0


def _dump(path, doc):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print(f"  wrote {os.path.relpath(path, _ROOT)}")


# ==========================================================================
# WHAT IS NOT DONE -- by name, so the next session does not have to guess
# ==========================================================================

NOT_DONE = """
1. `pK_slices` for the 25 kept terms. Needs `eval_part`, `multiply_wicks` and
   `check_vec_partition` in fnp. The int_parts are (1,), (2,), (3,), (4,),
   (1,1), (2,1), (2,2) -- all of them slices of degree <= 2 in the carried
   state, so none is n^3.
2. The `(1,1,1)` block, which IS the n^3 work and IS the regenerating flow:
   `pK_111 = wick(1) * WK[3]` plus ONE two-leg block with
   `fac1 = w(1)[:,None] * WK_11`, `fac2 = 3 I`,
   `fac3 = (w(2)[:,None] * w(1)[None,:] * WK_11).T`  (factor_k3.py:378-403).
   NOTE the asymmetry between `w(1)` and `w(2)*w(1)` -- p2_slice_cost.py uses
   `w1` for both and is therefore numerically wrong while being cost-correct.
3. `DS_pK_to_K` and the harmonic projection `proj_geq_r` at d = 3, 4. These
   carry n-dependent coefficients (`2/(2n+8)` and friends) that must be
   transcribed from `harmonic.py`, not re-derived.
4. `from_dstensor`'s regeneration in fnp: `M = diag(S3) + 3 S21^T`, plus the
   diagonal-zeroing convention. Already written correctly in
   `p2_slice_cost._rebuild_M`.
5. The wick coefficients `WICK_COEF_D['relu']` as closed forms in `alpha`, from
   `wick.py`. `k2_closure` above already returns `alphas` and `sigs` for this.
6. Parity: the whole flow against `p2_range_lab.means_of(hook_nl=regen)` on the
   same networks, to f32 rounding, at 128x4 and then at 1024x16.
7. Only then: `whest validate`, `certify_candidate`, a brokered residual soak,
   and the array-op count against the 0.4 s cliff.

Steps 1-5 are transcription against a frozen, checked table. Step 6 is the one
that can fail for an interesting reason.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-terms", action="store_true")
    ap.add_argument("--k2-parity", action="store_true")
    ap.add_argument("--price", action="store_true")
    ap.add_argument("--todo", action="store_true")
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--seed", type=int, default=1_000_921)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    print("=" * 78)
    print("WP14 / C0-ii  --  the fnp port of the regenerating K3 flow (INCOMPLETE)")
    print("=" * 78)
    rc = 0
    if a.check_terms:
        print("\n-- the frozen term table -------------------------------------")
        rc |= cmd_check_terms(a)
    if a.k2_parity:
        print(f"\n-- K2 backbone parity and price, {a.width} x {a.depth} ------------")
        rc |= cmd_k2_parity(a)
    if a.price:
        print("\n-- where the composed projection stands ----------------------")
        rc |= cmd_price(a)
    if a.todo or not any((a.check_terms, a.k2_parity, a.price)):
        print("\n-- NOT DONE, by name -----------------------------------------")
        print(NOT_DONE)
    print("=" * 78)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
