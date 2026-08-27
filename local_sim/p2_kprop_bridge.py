"""P2-KPROP bridge: run ARC's cumulant propagation on OUR canonically-minted 1024x16 networks.

Research only. **Never imported by `estimator.py`** -- upstream needs torch, and
the grader sandbox has no torch. This module exists to answer P-A / P-B / P-C
(`experiments/LOG.md`, the P2-GT contract) with the reference implementation
rather than a re-derivation of it, so that a disappointing number is a fact about
the METHOD and not about our transcription of it.

## The upstream pin, and why the diff was checked rather than trusted

`C:\\Users\\samsu\\Documents\\mlp_cumulant_propagation`, rev **`93d091a4`**, with
`local_sim/mlp_kprop_py310.patch` applied. Verified 2026-08-26 against a pristine
clone of the same rev: the working tree differs by **exactly** the 39 lines of
that patch across 7 files, md5 of the changed-line set identical. A third-party
reference that has been quietly edited would corrupt every P-A number while
looking completely normal, so this is checked, not assumed
(`--verify-upstream`).

Set `MLP_KPROP_SRC` to the checkout's `src` directory. Nothing is installed into
the participant environment.

## The four conventions, each verified against upstream source, not from memory

1. **Weight orientation.** Our forward is `x = relu(x @ W)` with `W` stored
   `(in, out)`. Upstream layers are `nn.Linear`, whose `weight` is `(out, in)`
   and which computes `x @ weight.T`. Therefore `weight = W.T`. Getting this
   backwards is not a small error: the repository measured the wrong convention
   at 7,123x worse at layer 1 and 130,000x by layer 2 (MOONSHOT G0, 2026-08-07).
2. **Depth.** `MLP(num_layers=17)` builds `len(nonlins) == 16`, so `act0..act15`
   are exactly our 16 post-ReLU layers and `Ws[16]` is an output layer we never
   reach. `up_to_layer='act15'` stops there.
3. **Init scale.** Upstream's `init_kind='he'` sets `w_var[0] = 1.0` and
   `w_var[1:] = 2.0`. **Our networks are scale 2.0 at EVERY layer**, so we build
   with `init_kind='manual', w_var=2.0`. `mlp.init_scale` is not decoration --
   `mlp_kprop` reads it as the average-case metric `E[WW^T] = init_scale * I`
   when `use_avg_metric=True`, so a wrong value silently changes the flow.
4. **No biases.** `b_var=0, b_mean=0` makes `has_bias_i` false, so upstream
   builds `nn.Linear(..., bias=False)` and `linear_kprop` receives `bias=None`.

The weights themselves are OURS -- `mint_provenance.canonical_mint`, proven
bit-identical to the organisers' published `v2-phase2` weights. Upstream's own
initialiser is never used for anything but shape.

## What `--verify` proves, and what it does not

`--verify` runs upstream at `k_max=2` against `p2_cal_lab.closure`, the NumPy
mirror that is itself verified to `4e-30` relative mean square against the
instrumented `fnp` arm we actually ship. Agreement there means the **wiring** is
right: orientation, depth, metric, input cumulants. It says nothing about
whether `k_max=3` is accurate or affordable -- that is P-A and P-B, and each has
its own contract.

Run (GPU venv):

    .venv-gpu\\Scripts\\python.exe local_sim/p2_kprop_bridge.py --verify-upstream
    .venv-gpu\\Scripts\\python.exe local_sim/p2_kprop_bridge.py --verify
    .venv-gpu\\Scripts\\python.exe local_sim/p2_kprop_bridge.py --flops --k-max 3 --factor
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

WIDTH, DEPTH = 1024, 16
NUM_LAYERS = DEPTH + 1          # convention 2
INIT_SCALE = 2.0                # convention 3
UPSTREAM_REV = "93d091a4c26c042bfffa28f2e76a81bc0aba94bb"
DEFAULT_SRC = r"C:\Users\samsu\Documents\mlp_cumulant_propagation\src"


def _src() -> str:
    src = os.environ.get("MLP_KPROP_SRC") or DEFAULT_SRC
    if not os.path.isdir(src):
        raise SystemExit(
            f"REFUSED: MLP_KPROP_SRC={src!r} is not a directory. Point it at the "
            "checkout's src/ directory."
        )
    if src not in sys.path:
        sys.path.insert(0, src)
    return src


def verify_upstream(src: str) -> dict:
    """The checkout is rev 93d091a4 plus exactly the recorded patch, and nothing else."""
    repo = os.path.dirname(src)
    rev = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(["git", "-C", repo, "diff", "--", "src"],
                          capture_output=True, text=True).stdout
    changed = sorted(l for l in diff.splitlines()
                     if l[:1] in "+-" and not l.startswith(("+++", "---")))
    patch_path = os.path.join(_HERE, "mlp_kprop_py310.patch")
    with open(patch_path, "r", encoding="utf-8") as fh:
        recorded = sorted(l for l in fh.read().splitlines()
                          if l[:1] in "+-" and not l.startswith(("+++", "---")))
    got = hashlib.md5("\n".join(changed).encode()).hexdigest()
    want = hashlib.md5("\n".join(recorded).encode()).hexdigest()
    ok = (rev == UPSTREAM_REV) and (got == want)
    return {"rev": rev, "expected_rev": UPSTREAM_REV,
            "changed_lines": len(changed), "recorded_lines": len(recorded),
            "changed_md5": got, "recorded_md5": want, "pass": ok}


def build_mlp(weights, device="cuda", dtype=None):
    """A kprop MLP that is a CONTAINER for our weights, never a generator of them."""
    import torch
    from mlp_kprop.mlp import MLP

    dtype = dtype or torch.float64
    depth = len(weights)
    width = weights[0].shape[0]
    mlp = MLP(input_dim=width, hidden_dim=width, output_dim=width,
              num_layers=depth + 1, nonlin="relu",
              init_kind="manual", w_var=INIT_SCALE, b_var=0.0, b_mean=0.0)
    assert len(mlp.nonlins) == depth, (
        f"convention 2 broken: {len(mlp.nonlins)} nonlins for depth {depth}")
    assert not mlp.has_bias(), "convention 4 broken: upstream built biases"
    assert all(s == INIT_SCALE for s in mlp.init_scale), (
        f"convention 3 broken: init_scale={mlp.init_scale[:3]}...")
    with torch.no_grad():
        for l, W in enumerate(weights):
            # convention 1: nn.Linear.weight is (out, in) and computes x @ weight.T
            mlp.Ws[l].weight.copy_(torch.as_tensor(np.ascontiguousarray(W.T), dtype=dtype))
    return mlp.to(device=device, dtype=dtype)


def kprop_means(weights, *, k_max=3, factor=True, kind=None, device="cuda",
                dtype=None, use_avg_metric=True):
    """(depth, width) predicted post-ReLU means from ARC's flow, unmodified."""
    import torch
    from mlp_kprop.kprop_harmonic import SIMPLE, mlp_kprop

    dtype = dtype or torch.float64
    depth, width = len(weights), weights[0].shape[0]
    mlp = build_mlp(weights, device=device, dtype=dtype)
    K_in = {1: torch.zeros(width, device=device, dtype=dtype),
            2: torch.eye(width, device=device, dtype=dtype)}
    out = mlp_kprop(mlp, K_in, k_max=k_max, output_all=True,
                    kind=kind or SIMPLE, use_avg_metric=use_avg_metric,
                    factor=factor, up_to_layer=f"act{depth - 1}",
                    output_d_max=1)
    return np.stack([_degree1(out[f"act{l}"][1]) for l in range(depth)], axis=0)


def _degree1(obj):
    """Upstream returns degree-1 cumulants as an HTensor, not a bare tensor.

    An HTensor is `Sym(core (x) metric^(x)r)` with total degree `core.ndim + 2r`,
    so degree 1 forces `r == 0` and the core IS the mean vector. The assert is
    here rather than a silent `getattr` because an HTensor with `r > 0` reaching
    this line would mean the flow returned a different degree than asked for,
    and quietly reading `.core` would hand back a wrong-shaped array that still
    looks like a prediction.
    """
    import torch

    core = obj
    if hasattr(obj, "core"):
        assert getattr(obj, "r", 0) == 0, f"degree-1 HTensor with r={obj.r}"
        core = obj.core
    core = torch.as_tensor(core)
    assert core.ndim == 1, f"expected a (width,) mean vector, got {tuple(core.shape)}"
    return core.detach().to(torch.float64).cpu().numpy()


def _mint(seed):
    from mint_provenance import canonical_mint
    w, _e, _s = canonical_mint(WIDTH, DEPTH, seed)
    return w


def cmd_verify(a) -> int:
    """k_max=2 upstream vs our own verified K2 mirror. This checks WIRING."""
    import p2_cal_lab as lab

    print("  k_max=2 upstream  vs  p2_cal_lab.closure (mirror of the shipped arm)")
    print("  a mismatch here is a convention error, not a method result.\n")
    worst = 0.0
    rows = []
    for i, seed in enumerate(a.seeds):
        w = _mint(seed)
        t0 = time.perf_counter()
        up = kprop_means(w, k_max=2, factor=False, device=a.device)
        el = time.perf_counter() - t0
        ours = lab.closure(w)
        num = float(((up - ours) ** 2).mean())
        den = float((ours ** 2).mean())
        rel = num / den
        worst = max(worst, rel)
        # per-layer, because a wiring error that only bites after layer 1 is the
        # one that looks like a rounding difference at layer 0.
        per = [float(((up[k] - ours[k]) ** 2).mean() / max((ours[k] ** 2).mean(), 1e-300))
               for k in range(DEPTH)]
        rows.append({"seed": seed, "rel_mse": rel, "per_layer_rel_mse": per,
                     "seconds": el})
        print(f"    seed {seed}   rel mean-square {rel:.3e}   worst layer "
              f"{max(per):.3e} (layer {int(np.argmax(per))})   {el:.2f}s")
    ok = worst < a.tol
    print(f"\n  worst over {len(a.seeds)} networks: {worst:.3e}   tol {a.tol:.1e}   "
          f"{'PASS' if ok else 'FAIL'}")
    if a.out:
        _dump(a.out, {"check": "kprop k_max=2 vs p2_cal_lab.closure",
                      "width": WIDTH, "depth": DEPTH, "tol": a.tol,
                      "worst_rel_mse": worst, "pass": ok, "rows": rows})
    return 0 if ok else 1


def cmd_flops(a) -> int:
    """Upstream's OWN counted-FLOP model. Not flopscope pricing -- see the caveat."""
    from mlp_kprop.flop_utils import fit_flop_polys

    # CONVENTION TRAP, and it cost a wrong number before it was caught.
    # Upstream's polynomial variable `l` is the ACTIVATION INDEX, not a layer
    # count: `_fit_flop_polys` builds `MLP(num_layers=l + 2)` and runs
    # `up_to_layer=f'act{l}'`, and `get_flops`'s own docstring says "l=0 means
    # up_to_layer='act0'". Our 16 post-ReLU layers are act0..act15, so l = 15.
    # Passing l = 16 prices a SEVENTEEN-layer network and reads ~13% high.
    # `num_layers` in mlp_kwargs is stripped by upstream with a warning, so it
    # is not passed here at all rather than being silently ignored.
    layer_index = DEPTH - 1
    kw = {"nonlin": "relu"}
    kprop_kw = {"k_max": a.k_max, "factor": a.factor}
    print(f"  fit_flop_polys(mlp={kw}, kprop={kprop_kw}) evaluated at "
          f"n={WIDTH}, layer='act{layer_index}'")
    polys = fit_flop_polys(kw, kprop_kw)
    budget = 2 ** 41
    out = {}
    for key, poly in sorted(polys.items()):
        v = float(poly(WIDTH, layer_index))
        out[key] = v
        if abs(v) > 1e6:
            print(f"    {key:<44} {v:>14.4e}   {v / budget:>8.4f} B")
    # Upstream's own accounting: `get_flops(..., name=None)` is total minus
    # 'mlp setup', because 'mlp setup' is weight INITIALISATION. The grader hands
    # us the weights, so we never pay it and it must not be in a cost claim.
    total = out.get("total", float("nan")) - out.get("mlp setup", 0.0)
    print(f"\n  TOTAL (upstream accounting, 'mlp setup' excluded) {total:.4e} FLOPs "
          f"= {total / budget:.4f} x budget (2**41)")
    print(f"    'mlp setup' excluded: {out.get('mlp setup', 0.0):.4e} "
          "(weight init -- the grader gives us the weights)")
    print("  CAVEAT: this is UPSTREAM's own FLOP model, in upstream's units. It is")
    print("  NOT flopscope 0.12 pricing and it is NOT a counted cost for this")
    print("  competition. docs/GOTCHAS.md records flopscope counterexamples in BOTH")
    print("  directions. Any operating point must be priced by flopscope before it")
    print("  is believed.")
    if a.out:
        _dump(a.out, {"check": "upstream fit_flop_polys", "width": WIDTH,
                      "depth": DEPTH, "k_max": a.k_max, "factor": a.factor,
                      "budget": budget, "total": total,
                      "x_budget": total / budget, "terms": out,
                      "caveat": "upstream FLOP model, not flopscope 0.12 pricing"})
    return 0


def _dump(path, doc):
    path = os.path.abspath(os.path.join(_ROOT, path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    print(f"  wrote {os.path.relpath(path, _ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-upstream", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--flops", action="store_true")
    ap.add_argument("--k-max", type=int, default=3)
    ap.add_argument("--factor", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="UNREGISTERED probe seeds. This command measures no "
                         "truth and makes no decision, so it needs arbitrary "
                         "deterministic inputs, not experimental provenance.")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.seeds is None:
        a.seeds = [1_000_101, 1_000_102]

    src = _src()
    print("=" * 78)
    print("P2-KPROP BRIDGE  --  ARC cumulant propagation on our minted 1024 x 16")
    print("=" * 78)
    print(f"  MLP_KPROP_SRC   {src}")

    v = verify_upstream(src)
    print(f"  upstream rev    {v['rev'][:12]}  ({'as pinned' if v['rev'] == UPSTREAM_REV else 'MISMATCH'})")
    print(f"  local patch     {v['changed_lines']} changed lines vs recorded "
          f"{v['recorded_lines']}  -> {'IDENTICAL' if v['changed_md5'] == v['recorded_md5'] else 'DIFFERENT'}")
    if not v["pass"]:
        raise SystemExit(
            "REFUSED: the upstream checkout is not rev 93d091a4 plus exactly the "
            "recorded patch. Every number from it would be unattributable."
        )
    print()
    if a.verify_upstream and not (a.verify or a.flops):
        print("  upstream integrity PASS. Nothing else run.")
        print("=" * 78)
        return 0

    rc = 0
    if a.flops:
        rc |= cmd_flops(a)
    if a.verify:
        rc |= cmd_verify(a)
    if not (a.flops or a.verify):
        print("  nothing to do; pass --verify and/or --flops")
    print("=" * 78)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
