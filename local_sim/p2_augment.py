"""WP3 — the FROZEN unfactored AUGMENT target, and its price.

Contract: `experiments/LOG.md`, *CONTRACT WP3* (2026-08-31), committed at
`35b8477` **before this file existed**. `GAME_PLAN.md` § WP3.

WHY `factor=False` IS THE ONLY ADMISSIBLE TARGET
-------------------------------------------------
Upstream says it itself (`kprop_harmonic.py:538`):

    WARNING: factor=True with kind=AUGMENT drops the augmented diagrams that
    do not factor, so it is NOT equivalent to factor=False

So a parity target built from `factor=True` would be parity against a *different
estimator*. `GAME_PLAN.md` § WP3 says the same thing and this module obeys it:
every frozen target here is unfactored.

WHAT "AUGMENTED K2" ACTUALLY RETAINS
------------------------------------
`get_d_max(2, AUGMENT) = 4`, `get_r_x = {1:0, 2:0, 3:1, 4:2}`. An HTensor of
degree `d` at radial order `r` has `d - 2r` free indices, so the state is:

    degree 1  r=0  (n,)     the mean
    degree 2  r=0  (n, n)   the covariance
    degree 3  r=1  (n,)     a VECTOR, not an n^3 tensor
    degree 4  r=2  ()       a SCALAR

**AUGMENT adds `n + 1` numbers to K2's `n + n^2`** — at width 1024 that is
1,025 on 1,049,600, a `0.098%` state increase. Memory cannot be what kills this
lane, which is why the contract makes the slice about PRICE.

THE METER CAUTION
-----------------
Upstream's `NamedFlopCounter` is **torch's** counter, not flopscope's.
`docs/CURRENT_STATE.md` records a `5.49x` "affordability target" that turned out
to compare two different meters and *"must not be used as a projection"*. So
everything here is a **RATIO measured inside one meter**, with both arms run
identically. Any absolute flopscope number derived from it is a **projection**
and is labelled one.

READ-ONLY
---------
The upstream checkout is a separate project and is never modified. Small shapes
run on CPU; nothing here needs a GPU, so nothing goes through `gpuq`.

Run:  .venv-gpu\\Scripts\\python.exe local_sim/p2_augment.py
      .venv-gpu\\Scripts\\python.exe local_sim/p2_augment.py --price --width 256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

import p2_kprop_bridge as BRIDGE  # noqa: E402

#: The shipped K2p1's flopscope price at 1024x16, reconciled op by op against
#: the official runner (`p2_flop_anchor.py --gate`, delta zero).
K2P1_FLOPSCOPE_ANCHOR = 137_600_326_656
BUDGET = 2 ** 41
FLOOR_FLOPS = int(0.1 * BUDGET)          # 219,902,325,555 -- the free zone
#: The contract's screen: the ratio over K2p1 that still leaves the multiplier
#: pinned at the 0.1 floor.
AFFORDABLE_RATIO = FLOOR_FLOPS / K2P1_FLOPSCOPE_ANCHOR


def _torch():
    import torch
    return torch


def _kinds():
    from mlp_kprop.kprop_harmonic import AUGMENT, BASE, SIMPLE
    return {"BASE": BASE, "SIMPLE": SIMPLE, "AUGMENT": AUGMENT}


def _ensure_upstream():
    """Load upstream and REFUSE unless it is the pinned rev plus the recorded patch."""
    BRIDGE.ensure_upstream() if hasattr(BRIDGE, "ensure_upstream") else None
    src = BRIDGE.DEFAULT_SRC
    if src not in sys.path:
        sys.path.insert(0, src)
    v = BRIDGE.verify_upstream(src)
    if not v["pass"]:
        raise RuntimeError(
            "upstream is NOT the pinned object: rev %s (want %s), diff md5 %s "
            "(want %s). A parity target built against an unverified reference "
            "is parity against nothing." % (v["rev"][:12], v["expected_rev"][:12],
                                            v["changed_md5"][:8], v["recorded_md5"][:8]))
    return v


# --------------------------------------------------------------------------- #
# 1. the frozen target
# --------------------------------------------------------------------------- #

def run_augment(weights, k_max=2, kind="AUGMENT", factor=False, dtype=None,
                device="cpu", use_avg_metric=True):
    """Upstream's flow, unmodified, returning the FULL retained state per layer."""
    torch = _torch()
    from mlp_kprop.kprop_harmonic import mlp_kprop
    dtype = dtype or torch.float64
    depth, width = len(weights), weights[0].shape[0]
    mlp = BRIDGE.build_mlp(weights, device=device, dtype=dtype)
    K_in = {1: torch.zeros(width, device=device, dtype=dtype),
            2: torch.eye(width, device=device, dtype=dtype)}
    return mlp_kprop(mlp, K_in, k_max=k_max, output_all=True,
                     kind=_kinds()[kind], use_avg_metric=use_avg_metric,
                     factor=factor, up_to_layer="act%d" % (depth - 1))


def state_inventory(out, layer_key):
    """`{degree: {"r": r, "shape": (...), "numel": N}}` for one layer's state."""
    inv = {}
    for d, obj in sorted(out[layer_key].items()):
        core = getattr(obj, "core", obj)
        inv[int(d)] = {"r": int(getattr(obj, "r", 0)),
                       "shape": tuple(int(s) for s in core.shape),
                       "numel": int(core.numel())}
    return inv


def degree1_means(out, depth):
    """`(depth, width)` post-ReLU means — the scored quantity."""
    return np.stack([BRIDGE._degree1(out["act%d" % l][1]) for l in range(depth)],
                    axis=0)


def freeze_target(weights, k_max=2, dtype=None, device="cpu"):
    """The content-addressed unfactored AUGMENT target for one network."""
    torch = _torch()
    dtype = dtype or torch.float64
    depth, width = len(weights), weights[0].shape[0]
    out = run_augment(weights, k_max=k_max, kind="AUGMENT", factor=False,
                      dtype=dtype, device=device)
    layers = {}
    for l in range(depth):
        key = "act%d" % l
        comp = {}
        for d, obj in sorted(out[key].items()):
            core = getattr(obj, "core", obj)
            arr = np.ascontiguousarray(core.detach().cpu().numpy().astype(np.float64))
            comp[str(int(d))] = {
                "r": int(getattr(obj, "r", 0)),
                "shape": list(arr.shape),
                "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                # a few summary statistics so a divergence is READABLE, not just
                # a hash mismatch
                "sum": float(arr.sum()), "absmax": float(np.abs(arr).max()),
                "rms": float(np.sqrt(np.mean(arr * arr))),
            }
        layers[key] = comp
    means = degree1_means(out, depth)
    return {
        "artifact": "p2_augment_unfactored_target",
        "artifact_version": 1,
        "upstream": _ensure_upstream(),
        "k_max": k_max, "kind": "AUGMENT", "factor": False,
        "width": width, "depth": depth, "dtype": str(dtype),
        "state_inventory": state_inventory(out, "act%d" % (depth - 1)),
        "layers": layers,
        "means_sha256": hashlib.sha256(
            np.ascontiguousarray(means).tobytes()).hexdigest(),
        "means_final_row_rms": float(np.sqrt(np.mean(means[-1] ** 2))),
        "why_unfactored": (
            "kprop_harmonic.py:538 -- factor=True with kind=AUGMENT drops the "
            "augmented diagrams that do not factor, so it is NOT equivalent. A "
            "target built from it would be parity against a different estimator."
        ),
    }


def target_digest(target):
    return hashlib.sha256(json.dumps(target, sort_keys=True,
                                     separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# 2. the price, as a RATIO inside one meter
# --------------------------------------------------------------------------- #

def measure_flops(weights, kind, k_max=2, factor=False, dtype=None,
                  device="cpu"):
    """Upstream's OWN torch-side counter. NOT flopscope. Never mix the two."""
    torch = _torch()
    from mlp_kprop.flop_utils import NamedFlopCounter
    dtype = dtype or torch.float64
    with torch.no_grad(), NamedFlopCounter() as ctr:
        run_augment(weights, k_max=k_max, kind=kind, factor=factor,
                    dtype=dtype, device=device)
    # `NamedFlopCounter` accumulates into `_accum` and delegates the raw total to
    # its `mode`. Reading a `total_flops` attribute that does not exist returned
    # a silent 0, and a 0 denominator then produced `ratio = nan` -- which the
    # screen dutifully classified as KILL. A measurement that did not happen
    # must not reach a verdict, so this REFUSES instead of returning a number.
    total = 0
    for attr in ("_accum", "_raw_accum"):
        d = getattr(ctr, attr, None)
        if isinstance(d, dict) and d:
            total = sum(int(v) for v in d.values())
            break
    if not total:
        mode = getattr(ctr, "mode", None)
        if mode is not None and hasattr(mode, "get_total_flops"):
            total = int(mode.get_total_flops())
    if total <= 0:
        raise RuntimeError(
            "upstream's flop counter reported %r for kind=%s. A zero or absent "
            "count is a BROKEN MEASUREMENT, not a cheap arm, and must not reach "
            "the affordability screen." % (total, kind))
    return int(total)


def price_ratio(weights, k_max=2, dtype=None, device="cpu"):
    """`AUGMENT / BASE` in ONE meter, both arms run identically."""
    base = measure_flops(weights, "BASE", k_max=k_max, dtype=dtype, device=device)
    aug = measure_flops(weights, "AUGMENT", k_max=k_max, dtype=dtype, device=device)
    return {"base_upstream_flops": base, "augment_upstream_flops": aug,
            "ratio": (aug / base) if base else float("nan"),
            "meter": "upstream torch NamedFlopCounter -- NOT flopscope"}


def screen(ratio):
    """The contract's affordability screen, applied. Chosen before any number.

    REFUSES on a non-finite or non-positive ratio. The first version returned
    "KILL" for `ratio = nan`, produced by a flop counter that silently reported
    zero -- a broken instrument scoring a verdict against the lane. A screen
    that cannot tell "expensive" from "unmeasured" is worse than no screen.
    """
    if not (isinstance(ratio, float) and ratio == ratio and ratio > 0
            and ratio != float("inf")):
        raise ValueError(
            "refusing to screen a ratio of %r -- that is an unmeasured arm, "
            "not an expensive one" % (ratio,))
    projected = K2P1_FLOPSCOPE_ANCHOR * ratio
    mult = max(0.1, projected / BUDGET)
    if ratio <= AFFORDABLE_RATIO:
        verdict, why = "PORT", ("the projected price keeps the multiplier "
                                "pinned at the 0.1 floor")
    elif ratio <= 3.0:
        verdict, why = "CONDITIONAL", ("affordable only if a DESIGN result "
                                       "shows the accuracy is worth leaving "
                                       "the floor; DO NOT port yet")
    else:
        verdict, why = "KILL", "priced out, as W-EFF was"
    return {"ratio": ratio, "affordable_ratio_bar": AFFORDABLE_RATIO,
            "projected_flopscope_flops": projected,
            "projected_multiplier": mult, "verdict": verdict, "why": why,
            "PROJECTION_CAVEAT": (
                "projected_flopscope_flops is the K2p1 flopscope anchor times a "
                "ratio measured in UPSTREAM'S meter. It is a PROJECTION, not a "
                "price. CURRENT_STATE records a 5.49x figure that compared two "
                "meters and had to be retracted. Only an fnp spelling billed by "
                "flopscope is a price.")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--seed", type=int, default=90000000701)
    ap.add_argument("--k-max", type=int, default=2)
    ap.add_argument("--price", action="store_true")
    ap.add_argument("--widths", default="", help="comma-separated, for --price")
    ap.add_argument("--emit", default="")
    a = ap.parse_args()

    v = _ensure_upstream()
    from mint_provenance import canonical_mint
    print("WP3 - the unfactored AUGMENT target")
    print("  upstream  rev %s  patch md5 %s  VERIFIED"
          % (v["rev"][:12], v["changed_md5"][:8]))

    w, _, _ = canonical_mint(a.width, a.depth, a.seed)
    tgt = freeze_target(w, k_max=a.k_max)
    print("  shape     %dx%d   k_max=%d  kind=AUGMENT  factor=False"
          % (a.width, a.depth, a.k_max))
    print("  state inventory at the last layer:")
    tot = 0
    for d, rec in sorted(tgt["state_inventory"].items()):
        tot += rec["numel"]
        print("    degree %d  r=%d  shape %-12s numel %d"
              % (d, rec["r"], str(rec["shape"]), rec["numel"]))
    print("    total state numbers: %d" % tot)
    print("  digest    %s" % target_digest(tgt))

    if a.price:
        widths = ([int(x) for x in a.widths.split(",")] if a.widths
                  else [a.width])
        print()
        print("  PRICE -- ratio inside ONE meter (upstream's torch counter)")
        rows = []
        for n in widths:
            ww, _, _ = canonical_mint(n, a.depth, a.seed)
            pr = price_ratio(ww, k_max=a.k_max)
            sc = screen(pr["ratio"])
            rows.append(dict(pr, width=n, **sc))
            print("    n=%-5d BASE %-14s AUGMENT %-14s ratio %.4f  -> %s"
                  % (n, format(pr["base_upstream_flops"], ","),
                     format(pr["augment_upstream_flops"], ","),
                     pr["ratio"], sc["verdict"]))
        print()
        print("    screen bar: ratio <= %.4f keeps the multiplier at the 0.1 floor"
              % AFFORDABLE_RATIO)
        print("    CAVEAT: upstream's meter, NOT flopscope. Any absolute figure")
        print("    derived from this is a PROJECTION until an fnp spelling is billed.")
        tgt["price"] = rows

    if a.emit:
        d = os.path.dirname(os.path.abspath(a.emit))
        if d:
            os.makedirs(d, exist_ok=True)
        out = dict(tgt, digest=target_digest(tgt))
        with open(a.emit, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(out, indent=2, default=str) + "\n")
        print("\n  emitted -> %s" % a.emit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
