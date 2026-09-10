"""CAPPED-GENERATION AUGMENT — the one measurement that can reopen the port.

Contract: `experiments/LOG.md`, 2026-09-09, *"CONTRACT (Rule 1): capped-generation
AUGMENT"*, committed before this file was run. Nominated by the Rule-2 Skeptic as
the cheapest discriminating follow-up to the AUGMENT port price.

WHY THIS AND NOTHING ELSE
--------------------------
`local_sim/p2_augment_port_price.py` established that `kind=AUGMENT` on our chain
is `2.32x`–`10.8x` over the **hard per-MLP FLOP budget** at every Strassen depth.
That is not a "bad score" — `whestbench/scoring.py:739` returns zeros on
`BudgetExhaustedError`.

But a real `estimator_k3_aug.py` would never reach that state, because
`estimator_k3_strassen._fit_generations` already **drops carried generations until
the projected cost clears `0.95 B`**. An AUGMENT candidate would shrink itself the
same way and land near **two carried generations out of fifteen**.

**Nothing on disk measures AUGMENT there.** That number, and only that number,
decides whether Slice 4 item 2 is rewritten or dropped. Re-measuring the `7.45x`
is worthless: the decision is insensitive from `2.32x` to `10.8x`, wider than any
honest error bar on the transfer.

THE SEAM IS IN OUR CODE, NOT IN UPSTREAM
------------------------------------------
Capping requires intervening between `linear_kprop` and `nonlin_kprop`, which
`mlp_kprop` does not expose. Rather than patch the pinned checkout — which would
break `verify_upstream` and the whole provenance chain — this module **re-walks
upstream's own loop**, calling `linear_kprop` and `nonlin_kprop` unmodified and
applying the cap between layers.

**The control that makes that legitimate:** with `gens=None` this walk must
reproduce `mlp_kprop` **bit for bit**. `--verify-walk` asserts exactly that at two
shapes and both kinds, and `run()` refuses to produce a single datum until it
passes. A re-implementation that drifts from upstream is not a cheaper measurement,
it is a different estimator, and this repository has already lost a lane to
measuring one method and reporting another.

WHY TRUNCATING `_factors` IS SAFE, MEASURED RATHER THAN ASSUMED
-----------------------------------------------------------------
`FactoredTensor.add_factors_` concatenates on the RIGHT, so the youngest
generation is the last block of columns and `[:, -keep:]` is the youngest-`g`
slice — the same convention `estimator_k3_strassen._truncate` uses.

`FactoredTensor` also carries a `repeated` DSTensor cache which `add_factors_`
updates, and truncating columns without it would silently over-count. **Measured
at `(24, 5)`, `kind=AUGMENT`: `repeated.slices` is EMPTY at every layer**, and
`get_repeated()` recomputes from the factors, so there is no stale cache to
corrupt. The module re-asserts that at runtime and refuses if it is ever
non-empty.

TIER
----
**DESIGN.** It re-reads the **P01** asset (`experiments/2026-08-26_p2_gt`) that D1
already spent, exactly as D2 did. `# seed-ledger: no block named`. It orients
priority and **promotes nothing**; any promotion needs a fresh block.

Run (GPU venv; submit through workerq):
    .venv-gpu\\Scripts\\python.exe local_sim/p2_augment_capped.py --verify-walk
    .venv-gpu\\Scripts\\python.exe local_sim/p2_augment_capped.py --gens 2 3 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

OUT_DIR = os.path.join(_ROOT, "experiments", "2026-09-09_slice4")
OUT = os.path.join(OUT_DIR, "augment_capped.json")
GT_DIR = "experiments/2026-08-26_p2_gt"

#: Pre-registered in the contract, before any datum.
SIMPLE_BASELINE_XMSE = 4.004685e-08     # reproduced to the digit three times
AUGMENT_UNCAPPED_XMSE = 1.912323e-08    # D1
#: The gate, chosen NOW and written on the difference the slice isolates:
#: capped AUGMENT must beat uncapped SIMPLE by >= 1.4x to reopen item 2.
GATE_RATIO = 1.4
GATE_XMSE = SIMPLE_BASELINE_XMSE / GATE_RATIO   # 2.860e-08

#: AUGMENT births 8n columns per layer after the first (3n at layer 0), measured.
BIRTH_FIRST, BIRTH_STEADY = 3, 8
SIMPLE_BIRTH = 2


def xmse(pred, A, B):
    """Unbiased two-half cross estimator. E[(p-A)(p-B)] = MSE(p) exactly."""
    return float(np.mean((pred - A) * (pred - B)))


def paired(a, b):
    """Paired arithmetic difference `a - b` with a studentised-t interval."""
    from competition_score import _t975
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    h = _t975(n - 1) * se if n > 1 else float("nan")
    r = np.asarray(a, float) / np.asarray(b, float)
    return {"paired_diff": m, "se": se, "ci95": [m - h, m + h], "n": n,
            "arithmetic_ratio": float(np.mean(a) / np.mean(b)),
            "geometric_ratio_DIAGNOSTIC": float(np.exp(np.mean(np.log(r)))),
            "n_a_worse": int((d > 0).sum())}


def _cap_factored(K3, keep_cols):
    """Youngest-`keep_cols` slice of a FactoredTensor, upstream's convention.

    REFUSES if the `repeated` cache is non-empty: truncating columns under a
    populated cache would silently over-count, and the measurement that this is
    safe (empty at every layer under `kind=AUGMENT`) is an observation about the
    current upstream, not a guarantee about every future one.
    """
    from mlp_kprop.factor_k3 import FactoredTensor

    if getattr(K3, "repeated", None) is not None and K3.repeated.slices:
        raise SystemExit(
            "REFUSED: FactoredTensor.repeated carries slices %s. Truncating "
            "columns under a populated repeated cache over-counts. The seam's "
            "safety was verified on an EMPTY cache and does not extend to this."
            % sorted(K3.repeated.slices))
    facs = K3.factors                       # already a defensive clone
    r = facs[0].shape[1]
    if keep_cols is None or r <= keep_cols:
        return K3
    return FactoredTensor(n=K3.n, d=K3.d,
                          factors=tuple(f[:, -keep_cols:] for f in facs),
                          device=K3.device, dtype=K3.dtype)


def walk(mlp, K_in, kind, gens=None, output_d_max=1, factor=True):
    """Upstream's own `mlp_kprop` loop, re-walked so a cap can sit between layers.

    Calls `linear_kprop` and `nonlin_kprop` UNMODIFIED. With `gens=None` this must
    reproduce `mlp_kprop` bit for bit -- `--verify-walk` asserts it.
    """
    import torch
    from mlp_kprop.kprop_harmonic import (
        WICK_COEF_D, coerce_input, linear_kprop, metric_needed, nonlin_kprop)

    n = mlp.Ws[0].weight.shape[1]
    wick = [WICK_COEF_D[nm] for nm in mlp.nonlin_names]
    init_scale = mlp.init_scale
    skip_metric = not metric_needed(3, kind)
    K = coerce_input(K_in, k_max=3, kind=kind)

    births = BIRTH_FIRST if kind.name == "AUGMENT" else SIMPLE_BIRTH
    steady = BIRTH_STEADY if kind.name == "AUGMENT" else SIMPLE_BIRTH
    keep_cols = None if gens is None else int(gens) * steady * n

    means = []
    for l, W_module in enumerate(mlp.Ws):
        if l >= len(mlp.nonlins):
            break
        metric = init_scale[l] if True else None
        if skip_metric:
            metric = 1.0
        K = linear_kprop(K, W_module.weight, k_max=3, set_metric=metric,
                         bias=W_module.bias)
        K = nonlin_kprop(K, nonlin_wick_coef=wick[l], k_max=3, kind=kind,
                         use_pK=True, factor=factor)
        if keep_cols is not None and 3 in K and hasattr(K[3], "factors"):
            K[3] = _cap_factored(K[3], keep_cols)
        obj = K[1]
        core = obj.core if hasattr(obj, "core") else obj
        means.append(torch.as_tensor(core).detach().to(torch.float64).cpu().numpy())
    _ = births
    return np.stack(means, axis=0)


def cmd_verify_walk(a) -> int:
    """`walk(gens=None)` must reproduce `mlp_kprop` BIT FOR BIT. No datum before this."""
    sys.path.insert(0, a.upstream)
    import torch
    from mlp_kprop.kprop_harmonic import SIMPLE, AUGMENT, mlp_kprop
    import p2_kprop_bridge as BRIDGE

    v = BRIDGE.verify_upstream(a.upstream)
    if not v["pass"]:
        raise SystemExit("REFUSED: upstream is not the pinned object: %r" % v)
    print("  upstream rev %s + recorded patch VERIFIED\n" % v["rev"][:12])

    worst = 0.0
    for (n, depth) in ((24, 5), (32, 4)):
        rng = np.random.default_rng(n * 7 + depth)
        W = [rng.standard_normal((n, n)) * (2.0 / n) ** 0.5 for _ in range(depth)]
        for kind in (SIMPLE, AUGMENT):
            mlp = BRIDGE.build_mlp(W, device=a.device, dtype=torch.float64)
            K_in = {1: torch.zeros(n, dtype=torch.float64, device=a.device),
                    2: torch.eye(n, dtype=torch.float64, device=a.device)}
            with torch.no_grad():
                ref = mlp_kprop(mlp, dict(K_in), k_max=3, output_all=True,
                                kind=kind, use_avg_metric=True, factor=True,
                                up_to_layer="act%d" % (depth - 1), output_d_max=1)
                ref = np.stack([
                    torch.as_tensor(
                        ref["act%d" % i][1].core
                        if hasattr(ref["act%d" % i][1], "core")
                        else ref["act%d" % i][1]
                    ).detach().to(torch.float64).cpu().numpy()
                    for i in range(depth)], axis=0)
                got = walk(mlp, dict(K_in), kind, gens=None, output_d_max=1)
            d = float(np.max(np.abs(ref - got)))
            worst = max(worst, d)
            print("    n=%-3d depth=%d  %-8s  max |walk - mlp_kprop| = %.3e  %s"
                  % (n, depth, kind.name, d, "BIT-EXACT" if d == 0.0 else "DRIFT"))
    ok = worst == 0.0
    print("\n  worst %.3e   %s" % (worst, "PASS" if ok else "FAIL"))
    if not ok:
        print("  The walk is a DIFFERENT ESTIMATOR from upstream's loop. Nothing")
        print("  measured through it may be reported. Fix the walk, not the gate.")
    return 0 if ok else 1


def _progress(frac):
    p = os.environ.get("WORKERQ_PROGRESS")
    if not p:
        return
    try:
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(str(min(max(frac, 0.0), 1.0)))
    except OSError:
        pass


def run(a) -> int:
    if cmd_verify_walk(a) != 0:
        raise SystemExit("REFUSED: the walk does not reproduce upstream. No data.")
    print()
    sys.path.insert(0, a.upstream)
    import torch
    from mlp_kprop.kprop_harmonic import SIMPLE, AUGMENT
    import p2_kprop_bridge as BRIDGE
    from mint_provenance import canonical_mint

    gt = os.path.join(_ROOT, GT_DIR)
    man = json.load(open(os.path.join(gt, "gt_manifest.json"), encoding="utf-8"))
    if not man.get("all_pass"):
        raise SystemExit("REFUSED: the P2-GT asset did not pass its own fitness "
                         "checks. An arm measured against a void label is not a "
                         "measurement.")
    zA = np.load(os.path.join(gt, "gt_halfA.npz"))
    zB = np.load(os.path.join(gt, "gt_halfB.npz"))
    seeds = [int(s) for s in zA["seeds"]]
    assert seeds == [int(s) for s in zB["seeds"]], "halves disagree about seeds"
    A_all, B_all = zA["means"], zB["means"]
    if a.limit:
        seeds, A_all, B_all = seeds[:a.limit], A_all[:a.limit], B_all[:a.limit]

    width, depth = man["width"], man["depth"]
    arms = [("simple", SIMPLE, None), ("augment", AUGMENT, None)]
    arms += [("augment_g%d" % g, AUGMENT, g) for g in a.gens]

    print("=" * 78)
    print("  CAPPED-GENERATION AUGMENT  --  %d x %d, %d nets"
          % (width, depth, len(seeds)))
    print("=" * 78)
    print("  asset            %s  (%s samples per half)"
          % (GT_DIR, format(man["samples_per_half"], ",")))
    print("  label noise/half %.4e" % man["gt_noise_per_half_mean"])
    print("  PRE-REGISTERED   gate: capped AUGMENT <= %.3e (%.1fx over SIMPLE's "
          "%.6e)" % (GATE_XMSE, GATE_RATIO, SIMPLE_BASELINE_XMSE))
    print("  arms             %s\n" % ", ".join(t for t, _, _ in arms))

    rows, t0 = [], time.perf_counter()
    for i, s in enumerate(seeds):
        w, _e, _ss = canonical_mint(width, depth, s)
        Af, Bf = A_all[i, -1, :], B_all[i, -1, :]
        row = {"seed": s}
        mlp = BRIDGE.build_mlp(w, device=a.device, dtype=getattr(torch, a.dtype))
        K_in = {1: torch.zeros(width, dtype=getattr(torch, a.dtype),
                               device=a.device),
                2: torch.eye(width, dtype=getattr(torch, a.dtype),
                             device=a.device)}
        for tag, kind, g in arms:
            try:
                with torch.no_grad():
                    m = walk(mlp, dict(K_in), kind, gens=g, output_d_max=1)
                row[tag] = xmse(m[-1], Af, Bf)
                row[tag + "_all_layers"] = xmse(m, A_all[i], B_all[i])
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                row[tag] = None
                row[tag + "_error"] = "%s: %s" % (type(exc).__name__, exc)
                torch.cuda.empty_cache()
        rows.append(row)
        _progress((i + 1) / len(seeds))
        drift = (row["simple"] - SIMPLE_BASELINE_XMSE
                 if row.get("simple") is not None else float("nan"))
        print("    %3d/%d seed %d  (%.0fs)  %s   [control drift %+.2e]"
              % (i + 1, len(seeds), s, time.perf_counter() - t0,
                 "  ".join("%s %.4e" % (t, row[t]) for t, _, _ in arms
                           if row.get(t) is not None), drift))

    doc = {
        "slice": "capped-generation AUGMENT",
        "contract": ("experiments/LOG.md 2026-09-09 CONTRACT (Rule 1): "
                     "capped-generation AUGMENT"),
        "tier": ("DESIGN. Re-reads the P01 asset D1 spent. Orients priority; "
                 "promotes NOTHING."),
        "asset": GT_DIR, "width": width, "depth": depth,
        "gt_noise_per_half": man["gt_noise_per_half_mean"],
        "arms": [t for t, _, _ in arms], "gens": a.gens,
        "gate": {"gate_xmse": GATE_XMSE, "gate_ratio_over_simple": GATE_RATIO,
                 "simple_baseline": SIMPLE_BASELINE_XMSE,
                 "augment_uncapped_d1": AUGMENT_UNCAPPED_XMSE},
        "seam": ("the walk re-implements mlp_kprop's loop and is asserted "
                 "BIT-EXACT against it at gens=None before any datum; upstream "
                 "is unmodified"),
        "rows": rows,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print("\n  wrote %s" % os.path.relpath(a.out, _ROOT))
    _progress(1.0)
    return report(a)


def report(a) -> int:
    with open(a.out, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    rows = doc["rows"]
    print("\n" + "=" * 78)
    print("  RESULT")
    print("=" * 78)
    ctl = [r["simple"] for r in rows if r.get("simple") is not None]
    if ctl:
        m = float(np.mean(ctl))
        print("  control SIMPLE   %.6e   vs pre-registered %.6e   drift %+.3e"
              % (m, SIMPLE_BASELINE_XMSE, m - SIMPLE_BASELINE_XMSE))
        print("  (prediction 3: if this does not reproduce, NOTHING else is read)")
    print()
    for arm in doc["arms"]:
        vals = [r[arm] for r in rows if r.get(arm) is not None]
        if not vals:
            print("      %-14s  --  absent" % arm)
            continue
        print("      %-14s  mean xmse %.4e   over %d nets   %.3fx SIMPLE"
              % (arm, float(np.mean(vals)), len(vals),
                 SIMPLE_BASELINE_XMSE / float(np.mean(vals))))
    print()
    for g in doc["gens"]:
        arm = "augment_g%d" % g
        ok = [(r[arm], r["simple"]) for r in rows
              if r.get(arm) is not None and r.get("simple") is not None]
        if len(ok) < 2:
            continue
        st = paired([x for x, _ in ok], [y for _, y in ok])
        mean_arm = float(np.mean([x for x, _ in ok]))
        passed = mean_arm <= GATE_XMSE
        print("      g=%d  mean %.4e   gate %.4e   %s" %
              (g, mean_arm, GATE_XMSE, "PASS -- item 2 REOPENS" if passed
               else "FAIL"))
        print("           paired (capped - SIMPLE) %+.4e  95%% [%+.4e, %+.4e]"
              % (st["paired_diff"], st["ci95"][0], st["ci95"][1]))
    print("\n  %s" % doc["tier"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gens", type=int, nargs="*", default=[2, 3, 4])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    # float32 IS THE COMPARABLE CHOICE, not merely the fast one. D1
    # (`p2_d1_augment_arm.py:235`) ran `float32`, and its `4.004685e-08` SIMPLE
    # baseline -- which this run's pre-registered prediction 3 gates on -- is an
    # f32 number. An f64 arm is measuring a different thing and could not
    # reproduce it.
    #
    # It is also ~46x faster here, and that is not incidental: a consumer RTX
    # 5090 runs f64 at a small fraction of its f32 rate, and f64 doubles the
    # state so a 23 GiB AUGMENT arm no longer fits. Job 453 was submitted at
    # f64, reached 8% in 2h33m against a 60m estimate, held 30.6 of 31.8 GiB,
    # and was cancelled. THAT IS WHY THIS DEFAULT IS f32 AND WHY IT IS WRITTEN
    # DOWN: the wrong default here is both a wrong measurement and a queue
    # outage for every other project on the box.
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    ap.add_argument("--verify-walk", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--upstream",
                    default=r"C:\Users\samsu\Documents\mlp_cumulant_propagation\src")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    if a.verify_walk:
        a.device = "cpu" if a.device == "cuda" and not _cuda() else a.device
        return cmd_verify_walk(a)
    if a.report:
        return report(a)
    return run(a)


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
