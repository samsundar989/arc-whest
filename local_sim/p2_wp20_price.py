"""WP20 deployment: what would it cost to SHIP the `3.330x` cell?

**Strategist ruling, 2026-09-04, carve-out to Ruling 1.** The `carry3_hybrid:192`
cell measured at `3.330x` end-to-end drift reduction (`experiments/2026-09-03_wp20/
c2_carry3_hybrid_P05.json`) may be shipped **only** under a promotion-class
deployment contract, and **only if its ops and FLOPs are priced first and it
stays at the multiplier floor with the grading server's `+2**24`**. If it does
not fit, it is dropped **without a further contract**. This module is that price.

It is a PRICE, never an accuracy claim. Values are arbitrary; counted flopscope
FLOPs on this dense path are a function of shapes and dtypes only
(`p2_slice_cost.py --value-invariance` established that, and this module inherits
it rather than re-deriving it).

`# seed-ledger: no block named` -- no seed block is drawn, no truth is read,
nothing is selected. Weights are `default_rng` noise at the live shape.

WHAT IS BEING PRICED, AND WHY IT IS NOT THE CELL THAT WAS MEASURED
------------------------------------------------------------------
`carry3_hybrid:192` is a CEILING. It reads `D(p) = window_3(p) - window_0(p)`
from two offline flows, truncates `D21` to rank `192` AFTER an exact three-layer
propagation, keeps `D3` whole, and injects. A shipped estimator has no offline
flow: it must CARRY the last three generations of the all-distinct block through
its own forward pass. That carry is the cost, and it is what this module meters.

The carried object is upstream's own. `factor_k3.py`, read 2026-09-04:

    contract_W(W)            ->  `W @ factor` for EACH of the three legs
    _factored_get_dslice     ->  one `(A_i * A_j) @ A_k^T` per block permutation
    from_dstensor(ds)        ->  factors `(diag(S3) + 3 S21^T, I, I)`
    the generated block      ->  `fac2 = torch.eye(n) * 3`  (line 400)

**Correction, `SKEPTIC` 2026-09-04.** An earlier revision of this docstring said
`_factored_get_dslice` costs *"three matmuls when the legs differ, two when two
coincide"*. That is a true statement about the ARITHMETIC and a false one about
upstream: `block_perms` is built from INDEX permutations (`factor_k3.py:42-44`)
and is always three for `d = 3, part = (2,1)`, whatever the leg values. Two of
the three products coincide when `A2 == A3`, and upstream does not realise the
saving -- this module's `regen` spelling realises it by hand. The claim is kept
here in corrected form rather than deleted, because it was quoted as a fact about
upstream and it was not one.

The regenerating flow is cheap for exactly one reason: two of its three legs are
the identity, so `contract_W` on them is free and the `(2,1)` slice's three block
permutations collapse to two distinct products. **A carried generation destroys
that structure once it has been propagated twice.** The block layer `l` generates
is `(w1 . WK_11, 3 I, (w2 (x) w1 . WK_11)^T)` (`p2_wp15_depth.fresh_block`); its
FIRST `contract_W` is two products and a column scaling, because the middle leg
is still a scaled identity, and every hop after that is three real `(n, R)`
matrices at the generic price.

THE SPELLINGS, CHEAPEST FIRST
-----------------------------
`regen`         the CONTROL: WP13's structured regenerating third-order layer at
                f32. It reproduces a published per-layer figure, so a meter or a
                dtype that has drifted shows up against a known number before it
                reaches a verdict.

`floor_r`       **a LOWER BOUND that is not achievable, and it FITS.** The
                carried window's `contract_W` ONLY -- no slice read, no
                fresh-block construction, no truncation, no injection -- with
                every leg at `(n, r)` and the age-1 middle leg charged as the
                scaling it is. **The window WARMS UP**: at layer `l` the live
                generations are those made at `l-1, l-2, l-3`, so the counts are
                `1, 2, 3, 3, ...` = 42 block propagations over 15 rounds, not 45.
                `p2_wp15_depth.slices_window` is the authority -- *"`k >= p` is
                the uncapped flow, because there was never an older generation to
                discard"*.

                **`SKEPTIC` 2026-09-04, defect 1, and it flipped this row.** The
                first revision propagated three generations in every round and
                billed the age-1 middle leg as a matmul, giving `54,331,637,760`
                = `1.060x` the headroom, on which this module claimed *"the gap
                is arithmetic, not implementation"*. Corrected it is
                `44,675,629,056` = `0.872x` -- **AT the floor**. That claim is
                WITHDRAWN. The lower bound bounds nothing useful; the verdict
                rests on `carry3_full` instead, where the margin is a factor of
                ten rather than six percent.

`floor_full`    the same lower bound with honest `(n, n)` legs.

`carry3_r`      three carried blocks at rank `r`: `contract_W` + the `(2,1)`
                slice + the fresh block + the rank-`r` re-truncation + the
                injection.

                **It prices an object that CANNOT EXIST** (`SKEPTIC` 2026-09-04,
                defect 3). A `FactoredTensor`'s rank index is SHARED across all
                three legs (`factor_k3.py:80`, `add_factors_` concatenates along
                `dim=1`), so replacing each leg by an orthonormal basis of its
                range and discarding `B_i = Q_i^T A_i` yields a Tucker form whose
                `(r,r,r)` core is thrown away -- not a rank-`r` CP carry. Reading
                the `(2,1)` slice of the honest object needs `Q_i B_i`
                re-materialised at `(n, n)`, which is `floor_full`'s price.
                It therefore **UNDER-prices**, which is conservative for the
                verdict, and **the rank sweep below describes a fiction**: no
                conclusion of the form "the largest carried rank that fits is X"
                may be drawn from it.

`carry3_full`   **the price of the object that measured `3.330x`.** The same,
                with the carried legs at their honest rank. A generated block's
                middle leg is `3 I`, whose rank is `n`; there is no rank-`r`
                representation of it, so an EXACT three-generation carry -- which
                is what `D(p)` is -- carries `(n, n)` legs.

`svd_full`      one exact `linalg.svd` of an `(n, n)` matrix, for scale: the cell
                spells its truncation that way and a shipped file could not.

THE BAR
-------
`--anchor` is the canary's own metered total (`experiments/2026-09-03_wp18_canary/
certify_guarded.json`: `sections/metered_runs[*]/flops_used`, identical across
four runs, and confirmed by `sections/subprocess_smoke/flops_per_mlp`) =
`168,636,407,398`. The multiplier is `max(0.1, (anchor + added + 2**24) / 2**41)`,
and "stays at the floor" means it is still exactly `0.1`. The headroom that
leaves is printed as `budget_headroom` and must equal `CONTRACT WP20`'s own
`51,249,140,941` -- asserted below, because a headroom that silently disagreed
with the contract would decide this on the wrong number.

Operations are the SECOND bar, not a footnote: under `lambda = 0` the cost that
can kill a submission is an OPERATION COUNT, because that is what the `0.4 s`
residual cliff is paid in (write-up 6.6). The canary bills `2,674`.

**But an op count is not portable between spellings, and this module's first
revision added one that was not** (`SKEPTIC` 2026-09-04, defect 2). `regen`
bills `469` operations where the canary's OWN third-order increment is `2,255`
(`flow_price_wp17b.json`, `2,708 - 453`) -- the same arithmetic in `4.81x` fewer
array calls, and the auditor measured `3x` spelling-dependence inside identical
arithmetic besides. A lab op increment may therefore NEVER be added to a
shipped-file op total. `added_ops_shipped_estimate` translates through that ratio
first; `ops_x_canary_lab_spelling_DO_NOT_USE` is retained under that name so the
old number is traceable rather than deleted.

Run:
    uv run python local_sim/p2_wp20_price.py
    uv run python local_sim/p2_wp20_price.py --rank 192 --out experiments/<slice>/wp20_price.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# APPEND, never insert: `local_sim/` holds numpy shims named `flopscope` and
# `whestbench`. Prepending this directory turns every counted FLOP into fiction.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

WIDTH, DEPTH, RANK = 1024, 16, 192
BUDGET = 2 ** 41

#: The canary's own metered per-MLP total, from `certify_guarded.json`.
CANARY_FLOPS = 168_636_407_398
#: `sections/metered_runs[*]/n_ops`, identical across four runs.
CANARY_OPS = 2_674
#: The grading server bills an unexplained `+2**24` over the local official
#: runner. Included because this decision is decided AT a multiplier boundary.
SERVER_OFFSET_FLOPS = 2 ** 24
#: `CONTRACT WP20`'s own headroom figure. Asserted, not trusted.
CONTRACT_HEADROOM = 51_249_140_941

#: `experiments/2026-09-03_wp16_parity/flow_price_wp17b.json` -- the canary's own
#: flow, metered on the SHIPPED instrument. `SKEPTIC` 2026-09-04 defect 2: this
#: module's `regen` is an excellent FLOP control (`-0.78%` against the increment
#: below) and a POOR operation control, because a lab transcription spells the
#: same arithmetic in far fewer array calls. Operation counts are `3x`
#: spelling-dependent inside identical arithmetic, so a lab op increment may NOT
#: be added to a shipped-file op total. These constants are what makes the
#: `n_ops` column translatable instead of misleading.
SHIPPED_K3_OPS = 2708      # the canary's whole flow
SHIPPED_K2_OPS = 453       # its second-order backbone
SHIPPED_K3_FLOPS = 168_676_988_572
SHIPPED_K2_FLOPS = 68_782_549_504


def _require_real_flopscope(flops) -> None:
    path = os.path.abspath(getattr(flops, "__file__", "") or "")
    if path.startswith(os.path.abspath(_HERE) + os.sep):
        raise RuntimeError(
            f"REFUSED: `flopscope` resolved to the local shim at {path}. Every "
            "counted FLOP below would be fiction."
        )
    if not hasattr(flops, "BudgetContext"):
        raise RuntimeError(f"flopscope at {path} has no BudgetContext.")


def _assert_f32(*arrays) -> None:
    """A single CONCRETE numpy scalar anywhere promotes the arm to f64 and
    doubles every bill. `docs/GOTCHAS.md`. Asserted, never assumed."""
    for a in arrays:
        dt = str(getattr(a, "dtype", None))
        if dt != "float32":
            raise RuntimeError(f"REFUSED: an operand is {dt}, not float32.")


# --------------------------------------------------------------------------
# the spellings. Each body below is ONE layer; the driver runs `depth` of them.
# --------------------------------------------------------------------------

def _regen_layer(fnp, W, M, w1, Sig_off, *, last_layer: bool):
    """WP13's structured regenerating third-order layer, transcribed.

    Kept here as the CONTROL. `p2_slice_cost.py` published `11.539 n^3` per layer
    at f64; f32 is billed at half the rate, so this must land at `~5.77 n^3`.
    """
    P = W.T @ M
    Q = W.T
    S3_z = (P * Q * Q).sum(axis=1)
    if last_layer:
        # nothing downstream reads the (2,1) slice at the final layer.
        return S3_z, None
    Pw = P * w1[:, None]
    Qw = Q * w1[:, None]
    S21 = (2.0 * ((Pw * Qw) @ Qw.T) + ((Qw * Qw) @ Pw.T)) / 3.0
    fac1 = Sig_off * w1[:, None]
    fac3 = Sig_off * w1[None, :]
    # Python scalars, NOT `np.float64(3.0)`: under NEP-50 a numpy scalar is
    # CONCRETE and promotes an f32 array back to f64.
    S21 = S21 + (fac1 * 3.0) + (fac1 * fac3) * 3.0 + fac3 * 3.0
    S3_z = S3_z + (fac1 * fac3).sum(axis=1)
    S21 = S21 - fnp.diag(fnp.diag(S21))
    return S3_z, S21


def _carry_propagate(W, blocks, w2=None):
    """`contract_W` on every carried generation: `W @ factor`, three legs each.

    Upstream's `FactoredTensor.contract_W` verbatim in shape terms. This is the
    irreducible cost of carrying: a block that is not propagated is not carried,
    and none of its three legs is the identity after the first hop.

    **`SKEPTIC` 2026-09-04, defect 1(b), and it is a real saving I was billing
    for.** A block's FIRST `contract_W` is not three matmuls. Upstream builds the
    generated block with `fac2 = torch.eye(n) * 3` (`factor_k3.py:400`), so at
    age 1 the middle leg is a scaled identity and `W @ (3 I_w)` is a COLUMN
    SCALING of `W`, not a product. Exactly one carried block is at age 1 in every
    round. `blocks` therefore carries `(legs, age)` and the age-1 middle leg is
    billed as the scaling it is.
    """
    out = []
    for legs, age in blocks:
        A1, A2, A3 = legs
        if age == 0 and w2 is not None:
            # The middle leg is still `3 I` scaled by the wick vector, so this
            # hop is a scaling and not a product. Billed on the leg's OWN shape,
            # which is the cheapest defensible charge -- appropriate because the
            # only spellings that take this branch are lower bounds. (The
            # constant 3 folds into `w2` and does not change what is billed.)
            mid = A2 * w2[:, None]
        else:
            mid = W.T @ A2
        out.append(((W.T @ A1, mid, W.T @ A3), age + 1))
    return out


def _carry_slice21(blocks):
    """The `(2,1)` slice contributed by the carried blocks.

    `_factored_get_dslice` sums one `(A_i * A_j) @ A_k^T` per distinct block
    permutation; three distinct legs give three. The sum over independent
    generation blocks is separable -- concatenating the legs and slicing once
    gives the identical arithmetic -- so summing per block is not an
    approximation and does not understate the price.
    """
    out = None
    for (A1, A2, A3), _age in blocks:
        s = (A1 * A2) @ A3.T + (A1 * A3) @ A2.T + (A2 * A3) @ A1.T
        out = s if out is None else out + s
    return out


def _fresh_block(fnp, Sig_off, w1, w2):
    """The block layer `l` GENERATES: `(w1 . WK_11, 3 I, (w2 (x) w1 . WK_11)^T)`.

    `p2_wp15_depth.fresh_block`, in shape terms. `O(n^2)` to build; its middle
    leg is a scaled identity, and that is the structure the first `contract_W`
    destroys.
    """
    A1 = Sig_off * w1[:, None]
    A2 = fnp.diag(w2) * 3.0
    A3 = (Sig_off * w2[:, None] * w1[None, :]).T
    return (A1, A2, A3)


def _truncate_block(fnp, blk, r):
    """Re-express a carried block at rank `r`, by a randomized range finder.

    An exact `linalg.svd` of an `(n, n)` matrix bills `26 n^3` -- more than half
    the entire budget headroom for ONE call, measured by the `svd_full` spelling
    below -- so no shipped file could spell it the way the lab cell does. This is
    the cheapest honest alternative: one sketch, one QR, one projection, per leg.

    It is also the step that has no correct answer: the middle leg of a generated
    block is `3 I`, whose rank is `n`. Truncating it to `r` changes the object,
    which is why `carry3_full` and not `carry3_r` is the price of the thing that
    measured `3.330x`.
    """
    out = []
    for A in blk:
        Om = A[:, :r]          # the sketch matrix; values are arbitrary here
        Y = A @ Om             # (n, n) @ (n, r)
        Qm, _ = fnp.linalg.qr(Y)
        _B = Qm.T @ A          # (r, n) @ (n, n) -- the projection, priced
        out.append(Qm)         # the rank-`r` leg the carry would keep
    return tuple(out)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def measure(spelling: str, *, width: int, depth: int, rank: int,
            seed: int = 20_260_904) -> dict:
    import flopscope as flops
    import flopscope.numpy as fnp

    _require_real_flopscope(flops)
    rng = np.random.default_rng(seed)
    n, r = width, rank

    def mk(*shape):
        return fnp.array(rng.standard_normal(shape).astype(np.float32) * 1e-3)

    # Built OUTSIDE the BudgetContext, exactly as the grader builds weights
    # (`MLP.from_row` runs during dataset load, not during `predict`).
    Ws = [mk(n, n) for _ in range(depth)]
    M0 = mk(n, n)
    Sig = mk(n, n)
    w1 = fnp.array(np.abs(rng.standard_normal(n)).astype(np.float32))
    w2 = fnp.array(np.abs(rng.standard_normal(n)).astype(np.float32))
    blocks_r = [tuple(mk(n, r) for _ in range(3)) for _ in range(3)]
    blocks_n = [tuple(mk(n, n) for _ in range(3)) for _ in range(3)]
    _assert_f32(M0, Sig, w1, w2, Ws[0], blocks_r[0][0], blocks_n[0][0])

    with flops.BudgetContext(flop_budget=1000 * BUDGET, quiet=True) as ctx:
        if spelling == "regen":
            M = M0
            for l in range(depth):
                S3, S21 = _regen_layer(fnp, Ws[l], M, w1, Sig,
                                       last_layer=(l == depth - 1))
                if S21 is not None:
                    M = fnp.diag(S3) + 3.0 * S21.T

        elif spelling in ("floor_r", "floor_full"):
            src = blocks_r if spelling == "floor_r" else blocks_n
            # The final layer reads only the (3,) slice, so the carry is
            # propagated `depth - 1` times, not `depth`.
            #
            # **SKEPTIC 2026-09-04, defect 1(a).** The FIRST version of this
            # loop propagated three generations in every one of the fifteen
            # rounds. `p2_wp15_depth.slices_window` forbids that -- "`k >= p` is
            # the uncapped flow, because there was never an older generation to
            # discard" -- so at layer `l` the live generations are those made at
            # `l-1, l-2, l-3` and the counts are `1, 2, 3, 3, ...` = 42 block
            # propagations, not 45. It was a 6.7% over-count on the row the
            # verdict rested on, and it flipped that row.
            blocks = []
            for l in range(depth - 1):
                # drop anything older than three generations, admit the block
                # this layer generates (age 0, middle leg still `3 I`), then
                # propagate every live block. Counts: 1, 2, 3, 3, ... = 42.
                blocks = [b for b in blocks if b[1] < 3]
                blocks = blocks + [(src[l % 3], 0)]
                blocks = _carry_propagate(Ws[l], blocks, w2)

        elif spelling in ("carry3_r", "carry3_full"):
            wide = (spelling == "carry3_full")
            blocks = []
            M = M0
            for l in range(depth):
                last = (l == depth - 1)
                # the regenerating backbone the carry rides on
                S3, S21 = _regen_layer(fnp, Ws[l], M, w1, Sig, last_layer=last)
                if last:
                    break
                # roll the window: drop anything older than three generations,
                # admit the block this layer generates. SKEPTIC 2026-09-04
                # defect 1(a) -- the window WARMS UP, it is not three deep at
                # layer 0.
                blocks = [b for b in blocks if b[1] < 3]
                fresh = _fresh_block(fnp, Sig, w1, w2)
                if not wide:
                    fresh = _truncate_block(fnp, fresh, r)
                blocks = blocks + [(fresh, 0)]
                # propagate every carried generation
                blocks = _carry_propagate(Ws[l], blocks, None if not wide else w2)
                # what they contribute to the discarded (2,1) and (3,) slices
                D21 = _carry_slice21(blocks)
                D21 = D21 - fnp.diag(fnp.diag(D21))
                b0 = blocks[0][0]
                D3 = (b0[0] * b0[1] * b0[2]).sum(axis=1)
                # inject the correction the way `add_factors` takes it:
                # `M_corr = diag(D3) + 3 D21^T`, summed into the regenerated M.
                M = fnp.diag(S3) + 3.0 * S21.T + (fnp.diag(D3) + 3.0 * D21.T)

        elif spelling == "svd_full":
            fnp.linalg.svd(M0, full_matrices=False)

        else:
            raise SystemExit(f"unknown spelling {spelling!r}")

    return {"spelling": spelling, "width": width, "depth": depth, "rank": rank,
            "flops": int(ctx.flops_used), "n_ops": len(ctx.op_log),
            "flops_per_n3": ctx.flops_used / float(width ** 3)}


#: how many shipped-instrument operations one lab-transcription operation of the
#: third-order flow stands for. `SKEPTIC` 2026-09-04 defect 2.
def _op_scale(regen_ops: int) -> float:
    return (SHIPPED_K3_OPS - SHIPPED_K2_OPS) / float(regen_ops)


def _verdict(added_flops: int, added_ops: int, anchor: int,
             regen_ops: int | None = None) -> dict:
    total = anchor + added_flops + SERVER_OFFSET_FLOPS
    mult = max(0.1, total / BUDGET)
    headroom = int(0.1 * BUDGET) - anchor - SERVER_OFFSET_FLOPS
    out = {"added_flops": int(added_flops),
           "added_ops_lab_spelling": int(added_ops),
           "total_with_server_offset": int(total),
           "x_budget": total / BUDGET, "multiplier": mult,
           "at_floor": bool(mult <= 0.1 + 1e-15),
           "budget_headroom": int(headroom),
           "x_headroom": added_flops / float(headroom)}
    if regen_ops:
        # NEVER add a lab op increment to a shipped op total. Translate first.
        scale = _op_scale(regen_ops)
        est = CANARY_OPS + added_ops * scale
        out.update({"op_instrument_scale": scale,
                    "added_ops_shipped_estimate": est - CANARY_OPS,
                    "ops_total_shipped_estimate": est,
                    "ops_x_canary_shipped_estimate": est / float(CANARY_OPS),
                    "ops_x_canary_lab_spelling_DO_NOT_USE":
                        (CANARY_OPS + added_ops) / float(CANARY_OPS)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--anchor", type=int, default=CANARY_FLOPS)
    ap.add_argument("--sweep", default="16,32,48,64,96,128,192",
                    help="carried ranks to price, so the verdict is a CROSSING "
                         "and not a single point")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    hd = int(0.1 * BUDGET) - a.anchor - SERVER_OFFSET_FLOPS
    if a.anchor == CANARY_FLOPS and hd != CONTRACT_HEADROOM:
        raise SystemExit(
            f"REFUSED: headroom {hd:,} disagrees with CONTRACT WP20's "
            f"{CONTRACT_HEADROOM:,}. The bar is wrong before any measurement.")

    print("=" * 78)
    print("WP20 DEPLOYMENT PRICE  --  what would it cost to SHIP the 3.330x cell?")
    print("=" * 78)
    print(f"  shape            {a.width} x {a.depth}, float32")
    print(f"  carried rank     {a.rank}")
    print(f"  anchor (canary)  {a.anchor:,} FLOPs, {CANARY_OPS:,} operations")
    print(f"  headroom to the 0.1 floor, with +2**24    {hd:,} FLOPs")
    print()

    order = ["regen", "floor_r", "carry3_r", "floor_full", "carry3_full", "svd_full"]
    rows = {}
    base = None
    for sp in order:
        m = measure(sp, width=a.width, depth=a.depth, rank=a.rank)
        if sp == "regen":
            base = m
        # `regen` is already inside the anchor, so the carry spellings are
        # charged NET of it. The floor spellings are propagation only and are
        # already net of everything.
        sub = base["flops"] if sp.startswith("carry3") else 0
        sub_ops = base["n_ops"] if sp.startswith("carry3") else 0
        m["added"] = _verdict(max(m["flops"] - sub, 0),
                              max(m["n_ops"] - sub_ops, 0), a.anchor,
                              regen_ops=base["n_ops"] if sp.startswith("carry3") else None)
        rows[sp] = m

    print(f"  {'spelling':<14}{'counted FLOPs':>20}{'ops':>7}{'/n^3':>10}"
          f"{'x headroom':>12}{'multiplier':>12}  verdict")
    print("  " + "-" * 76)
    for sp in order:
        m, v = rows[sp], rows[sp]["added"]
        tag = "" if sp in ("regen", "svd_full") else (
            "AT FLOOR" if v["at_floor"] else "OVER THE FLOOR")
        print(f"  {sp:<14}{m['flops']:>20,}{m['n_ops']:>7}"
              f"{m['flops_per_n3']:>10.3f}{v['x_headroom']:>12.3f}"
              f"{v['multiplier']:>12.6f}  {tag}")
    print()
    fits = rows["carry3_full"]["added"]["at_floor"]
    lb_fits = rows["floor_r"]["added"]["at_floor"]
    cf = rows["carry3_full"]["added"]
    print(f"  VERDICT   the 3.330x cell as built "
          f"{'FITS' if fits else 'DOES NOT FIT'} at the 0.1 multiplier floor: "
          f"{cf['x_headroom']:.2f}x the headroom on FLOPs, and "
          f"{cf['ops_x_canary_shipped_estimate']:.2f}x the canary's operations "
          f"on the shipped-instrument estimate (the 0.4 s cliff sits at about "
          f"{0.4/0.1797:.2f}x).")
    print(f"            The lower bound `floor_r` "
          f"{'FITS' if lb_fits else 'does not fit'} "
          f"({rows['floor_r']['added']['x_headroom']:.2f}x the headroom), so it "
          f"bounds nothing useful and the verdict rests on the as-built row, "
          f"where the margin is a factor of ten rather than six percent.")

    # The rank sweep. `carry3_r` is already LESS accurate than the 3.330x cell
    # at every rank, so this cannot rescue the lane -- it says how far from
    # fitting the lane is, and at what rank the arithmetic first allows a
    # three-generation carry at all.
    sweep = []
    ranks = [int(x) for x in a.sweep.split(",") if x.strip()]
    print(f"\n  rank sweep -- `carry3_r`, net of the regenerating backbone\n")
    print(f"  {'rank':>6}{'added FLOPs':>18}{'x headroom':>12}"
          f"{'added ops (lab)':>17}{'ops x canary*':>15}  FLOPs fit?")
    print("  " + "-" * 66)
    biggest_fit = None
    for rk in ranks:
        m = measure("carry3_r", width=a.width, depth=a.depth, rank=rk)
        v = _verdict(max(m["flops"] - base["flops"], 0),
                     max(m["n_ops"] - base["n_ops"], 0), a.anchor,
                     regen_ops=base["n_ops"])
        m["added"] = v
        sweep.append(m)
        if v["at_floor"]:
            biggest_fit = rk
        print(f"  {rk:>6}{v['added_flops']:>18,}{v['x_headroom']:>12.3f}"
              f"{v['added_ops_lab_spelling']:>17,}"
              f"{v['ops_x_canary_shipped_estimate']:>15.2f}"
              f"  {'yes' if v['at_floor'] else 'NO'}")
    print("\n  * operations translated to the shipped instrument; the raw lab "
          "count is not portable.")
    print("    The FLOPs column is a LOWER BOUND ON A FICTION -- `carry3_r` "
          "prices a Tucker")
    print("    truncation, not a valid rank-`r` CP carry (SKEPTIC 2026-09-04, "
          "defect 3), so NO")
    print("    conclusion of the form 'the largest rank that fits is X' may be "
          "drawn from it.")
    print(f"    (For traceability only, the withdrawn figure was: "
          f"{biggest_fit if biggest_fit is not None else 'none'}.)")

    out = {"check": "WP20 deployment price -- ops and FLOPs of shipping carry3_hybrid",
           "ruling": "strategist 2026-09-04, carve-out to Ruling 1",
           "width": a.width, "depth": a.depth, "rank": a.rank,
           "budget": BUDGET, "anchor_flops": a.anchor, "anchor_ops": CANARY_OPS,
           "server_offset_flops": SERVER_OFFSET_FLOPS,
           "budget_headroom_flops": hd,
           "spellings": rows,
           "rank_sweep": sweep,
           "rank_sweep_caveat": "carry3_r prices a Tucker truncation, not a "
                                "valid rank-r CP carry (SKEPTIC 2026-09-04 "
                                "defect 3). It UNDER-prices. No 'largest rank "
                                "that fits' conclusion may be drawn.",
           "largest_rank_at_floor_WITHDRAWN": biggest_fit,
           "as_built_fits_at_floor": bool(fits),
           "lower_bound_fits_at_floor": bool(lb_fits)}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
