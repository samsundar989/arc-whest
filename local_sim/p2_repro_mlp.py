"""Run a candidate over ONE named network from a public split, and show the crash.

WHY
---
On 2026-09-16 `estimator_k3_v31.py` raised on **2 of the 100** public mini
networks. `whest run` names the count and not the networks, and finding out
which two by re-running the whole split costs two hours per attempt. A fix then
costs two more hours to test. That is not a diagnosis loop, it is a wait.

This module makes the loop about a minute. It loads the same split through
whestbench's own `load_dataset`/`mlp_at`, so the `MLP` object is built exactly
as the grader builds it -- flopscope-backed weights, the dataset's own seed --
and runs `predict` inside a real `BudgetContext`. No mint, no truth, no labels
read: this reads the WEIGHTS of a public network and never its targets, so it
is not a CONFIRM-tier read and decides nothing.

    uv run python local_sim/p2_repro_mlp.py --candidate estimator_k3_v31.py --scan
    uv run python local_sim/p2_repro_mlp.py --candidate estimator_k3_v31.py \
        --names lena-ruiz,otto-vance
    uv run python local_sim/p2_repro_mlp.py --candidate estimator_k3_v31.py \
        --from-report experiments/2026-09-16_slice7_mini/v31_mini.json

`--scan` walks the whole split and reports every network that raises, which is
the two-hour run; prefer `--from-report` or `--names` once you know who failed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# APPEND, never insert: local_sim holds numpy shims named `flopscope` and
# `whestbench`, and putting this directory first silently swaps the real
# packages for them. Every "counted FLOP" would then be fiction.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

DEFAULT_DATASET = "hf://aicrowd/arc-whestbench-public-2026@v2-phase2"
DEFAULT_SPLIT = "mini"


def load_candidate(path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("cand_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def failed_names_from_report(path: str):
    """Which MLPs did a whest run report as failed, and how."""
    with open(path, encoding="utf-8") as fh:
        rep = json.load(fh)
    rows = rep["results"]["per_mlp"]
    out = []
    for r in rows:
        err = r.get("error_code")
        state = r.get("state")
        flagged = bool(err) or (state and state != "completed")
        # A row that neither errored nor completed a scored prediction: the
        # zero-prediction MSE is ~0.9 while a healthy row is ~3e-8, so this
        # catches a silently-zeroed row even when no error_code survived.
        mse = r.get("final_layer_mse")
        if mse is not None and mse > 1e-3:
            flagged = True
        if flagged:
            out.append((r.get("mlp_index"), r.get("mlp_name"), err, state, mse))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--names", default="", help="comma-separated mlp_name values")
    ap.add_argument("--indices", default="", help="comma-separated 0-based indices")
    ap.add_argument("--from-report", default="",
                    help="a whest run --format json report; reproduce whatever it "
                         "recorded as failed")
    ap.add_argument("--scan", action="store_true",
                    help="run EVERY network in the split and report each that "
                         "raises. This is the expensive path.")
    ap.add_argument("--max-scan", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    import flopscope as flops  # noqa: E402
    from whestbench.dataset import load_dataset, mlp_at  # noqa: E402
    import official_config as oc  # noqa: E402

    rnd = oc.PHASE2
    print(f"candidate  {a.candidate}")
    print(f"dataset    {a.dataset}  split={a.split}")
    print(f"round      {rnd.name}  {rnd.width}x{rnd.depth}  B={rnd.flop_budget:,}")

    path, _, rev = a.dataset.partition("@")
    if path.startswith("hf://"):
        path = path[len("hf://"):]
    ds = load_dataset(path, revision=(rev or None), split=a.split)
    n = len(ds)
    names = [ds[i]["mlp_name"] for i in range(n)]
    print(f"loaded     {n} networks")

    wanted = []
    if a.from_report:
        rows = failed_names_from_report(a.from_report)
        print(f"\nfrom report: {len(rows)} flagged row(s)")
        for idx, nm, err, state, mse in rows:
            print(f"  [{idx}] {nm}  error_code={err!r} state={state!r} mse={mse}")
        wanted = [idx for idx, _nm, _e, _s, _m in rows if idx is not None]
    if a.names:
        for nm in a.names.split(","):
            nm = nm.strip()
            if nm not in names:
                raise SystemExit(f"no network named {nm!r} in {a.split}")
            wanted.append(names.index(nm))
    if a.indices:
        wanted += [int(x) for x in a.indices.split(",") if x.strip()]
    if a.scan:
        wanted = list(range(n if not a.max_scan else min(n, a.max_scan)))
    wanted = sorted(dict.fromkeys(wanted))
    if not wanted:
        raise SystemExit("nothing selected: pass --names, --indices, "
                         "--from-report or --scan")

    mod = load_candidate(os.path.join(_ROOT, a.candidate)
                         if not os.path.isabs(a.candidate) else a.candidate)

    # setup() once, outside any BudgetContext -- the grader's lifecycle.
    est = mod.Estimator()
    if hasattr(est, "setup"):
        import whestbench as _wb
        from whestbench.sdk import SetupContext
        # `api_version` is REQUIRED and has no default. Passing it wrong once
        # made this module skip setup() silently and print a NOTE -- i.e. it
        # ran a different lifecycle from the grader while saying so in a line
        # nobody would read as "this result is about a different program".
        est.setup(SetupContext(width=rnd.width, depth=rnd.depth,
                               flop_budget=rnd.flop_budget,
                               api_version=getattr(_wb, "__version__", "0.16.1"),
                               seed=0))
        print("  setup(context) called, as the grader calls it once per run")

    results = []
    print(f"\nrunning {len(wanted)} network(s)\n" + "-" * 70)
    for i in wanted:
        mlp = mlp_at(ds, i)
        ctx = flops.BudgetContext(flop_budget=rnd.flop_budget,
                                  wall_time_limit_s=rnd.wall_time_limit_s,
                                  quiet=True)
        t0 = time.perf_counter()
        rec = {"index": i, "name": names[i], "ok": None, "seconds": None,
               "exception": None, "traceback": None, "flops_used": None}
        try:
            with ctx:
                out = est.predict(mlp, rnd.flop_budget)
            arr = np.asarray(out, dtype=np.float64)
            rec["ok"] = True
            rec["shape"] = list(arr.shape)
            rec["finite"] = bool(np.all(np.isfinite(arr)))
            rec["n_nonfinite"] = int(np.sum(~np.isfinite(arr)))
            status = "ok" if rec["finite"] else f"NON-FINITE x{rec['n_nonfinite']}"
        except BaseException as exc:                      # noqa: BLE001
            rec["ok"] = False
            rec["exception"] = f"{type(exc).__name__}: {exc}"
            rec["traceback"] = traceback.format_exc()
            status = f"RAISED  {type(exc).__name__}"
        rec["seconds"] = time.perf_counter() - t0
        try:
            rec["flops_used"] = int(ctx.flops_used)
        except Exception:                                  # noqa: BLE001
            pass
        print(f"[{i:3d}] {names[i]:<28} {rec['seconds']:7.2f}s  "
              f"{(rec['flops_used'] or 0):>15,}  {status}")
        if rec["traceback"]:
            print(rec["traceback"])
        results.append(rec)

    bad = [r for r in results if not r["ok"]]
    nonfinite = [r for r in results if r["ok"] and not r.get("finite", True)]
    print("-" * 70)
    print(f"{len(results)} run, {len(bad)} raised, {len(nonfinite)} non-finite")
    for r in bad:
        print(f"  RAISED [{r['index']}] {r['name']}: {r['exception']}")
    for r in nonfinite:
        print(f"  NON-FINITE [{r['index']}] {r['name']}: {r['n_nonfinite']} cells")

    if a.json:
        d = os.path.dirname(os.path.abspath(a.json))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"candidate": a.candidate, "dataset": a.dataset,
                                 "split": a.split, "results": results,
                                 "reads_labels": False}, indent=2) + chr(10))
        print(f"  -> {a.json}")
    return 1 if (bad or nonfinite) else 0


if __name__ == "__main__":
    raise SystemExit(main())
