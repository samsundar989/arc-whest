"""Three residual instruments, one file, one shape -- and which of them decides.

P2-P0 (Phase 0 of `docs/GAME_PLAN.md`). Contract prediction B8 and `VALIDATION.md`
section 2b-ii: `experiments/LOG.md`, 2026-08-24.

WHY THIS FILE EXISTS
--------------------
Phase 2 sets `lambda = 0`, so `C == F` and residual wall time stops being a cost.
It survives as a **0.4-second per-MLP cliff** that FAILS the MLP -- zeroed
prediction, multiplier forced to 1.0.

That change makes the measurement path matter MORE, not less. A price is smooth:
being 20% out on residual is a 20% error in `C`. A cliff is not: being 20% out near
one is the difference between scoring and failing. And this repository measures
residual three different ways, which is three different numbers:

    whest run --runner subprocess    THE DEPLOYMENT AUTHORITY. What the grader's
                                     own runner reports, on the organisers' data.
    whest run --runner local         the same runner, in-process. The package
                                     itself says the host context is not
                                     authoritative (`scoring.py` prefers
                                     `last_predict_stats()`).
    lifecycle_eval in-process        our own BudgetContext, which is what every
                                     local research measurement uses.

`docs/GOTCHAS.md` records the PHASE-1 file at 70.85 ms local / 54.47 ms subprocess
/ 9.84 ms graded -- local ABOVE subprocess, and both far above the grader. That
ordering became a rule of thumb ("local bills ~7x the grader"), and a rule of thumb
about a cliff is worth checking before it is leaned on.

WHAT IT MEASURES
----------------
The same unmodified `estimator.py`, on the same official `v2-phase2` rows, through
all three instruments, with per-MLP figures rather than a single reading -- because
a single wall-time reading is not a measurement, and this repository has been
caught by that before.

Reports median and IQR per instrument, the pairwise ratios, and the headroom
`0.4 / observed` that the cliff actually cares about.

WHAT IT DOES NOT ESTABLISH
--------------------------
Nothing about the GRADER's seconds. No graded Phase-2 report exists, so the
subprocess path is the best available proxy and is labelled as one. Local absolutes
never transfer (`docs/GOTCHAS.md`); only the ordering and the ratios are the point.

It draws no seed block: it runs the organisers' own public rows, which are CONFIRM
tier and read-only, and it selects nothing.

Run:  uv run python local_sim/p2_residual_paths.py [--n-mlps 10]
"""

from __future__ import annotations

import argparse
import json
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

import official_config  # noqa: E402

OUT_DIR = os.path.join(_ROOT, "experiments", "2026-08-24_p2_p0")
OUT = os.path.join(OUT_DIR, "p2_residual_paths.json")

DATASET = "hf://aicrowd/arc-whestbench-public-2026@v2-phase2"


def _whest_exe() -> str:
    d = os.path.dirname(sys.executable)
    for name in ("whest.exe", "whest"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return "whest"


def official_run(runner: str, n_mlps: int, candidate: str) -> dict:
    """One `whest run --json`, returning per-MLP residual and wall seconds."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run(
        [_whest_exe(), "run", "--estimator", candidate, "--dataset", DATASET,
         "--split", "mini", "--runner", runner, "--n-mlps", str(n_mlps), "--json"],
        capture_output=True, text=True, cwd=_ROOT, env=env,
    )
    i = p.stdout.find("{")
    if i < 0:
        return {"error": (p.stderr or p.stdout)[-2000:]}
    d = json.loads(p.stdout[i:])
    if not d.get("results"):
        return {"error": json.dumps(d)[:2000]}
    rows = d["results"]["per_mlp"]
    return {
        "residual_s": [float(r["residual_wall_time_s"]) for r in rows],
        "wall_s": [float(r["wall_time_s"]) for r in rows],
        "flops": [float(r["flops_used"]) for r in rows],
        "n_failed_mlps": int(d["results"]["n_failed_mlps"]),
        "whestbench": d.get("whestbench_version"),
    }


def lifecycle_run(n_mlps: int, candidate: str) -> dict:
    """Our own in-process BudgetContext, over the SAME rows the official run used.

    The official split's rows are the organisers' own networks; `lifecycle_eval`
    mints from input seeds instead. Counted FLOPs on this dense path are
    value-independent, so the COST comparison is exact; the residual seconds are a
    property of the code path and the host, not of the particular weights. Both
    facts are what make this comparison legitimate at all, and neither is assumed
    silently.
    """
    import lifecycle_eval as le

    spec = le.LifecycleSpec.for_round("v2-phase2")
    seeds = [5_150_800 + i for i in range(n_mlps)]   # legacy-namespace throwaways
    truth = np.zeros((n_mlps, spec.width), dtype=np.float32)
    rows, prov = le.evaluate(candidate, seeds, truth, spec=spec)
    return {
        "residual_s": [float(r.residual_wall_time_s) for r in rows],
        "wall_s": [float(r.wall_time_s) for r in rows],
        "flops": [float(r.flops_used) for r in rows],
        "n_failed_mlps": int(sum(r.failed for r in rows)),
        "round_deviations": prov.get("round_deviations"),
    }


def _stats(xs) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "q25": float(np.quantile(a, 0.25)),
        "q75": float(np.quantile(a, 0.75)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-mlps", type=int, default=10)
    ap.add_argument("--candidate", default="estimator.py")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    r = official_config.LIVE
    cap = r.residual_wall_time_limit_s

    print("=" * 92)
    print("RESIDUAL PATHS -- the same file, the same shape, three instruments")
    print("=" * 92)
    print(f"  candidate {args.candidate}   shape {r.width}x{r.depth}   "
          f"n = {args.n_mlps}   cliff = {cap} s")
    print()

    paths = {}
    for label, fn in (
        ("whest --runner subprocess", lambda: official_run("subprocess", args.n_mlps,
                                                           args.candidate)),
        ("whest --runner local", lambda: official_run("local", args.n_mlps,
                                                      args.candidate)),
        ("lifecycle_eval in-process", lambda: lifecycle_run(args.n_mlps,
                                                            args.candidate)),
    ):
        print(f"  running {label} ...", flush=True)
        paths[label] = fn()

    bad = {k: v["error"] for k, v in paths.items() if "error" in v}
    if bad:
        for k, v in bad.items():
            print(f"  {k} FAILED: {v[:400]}")
        return 1

    print()
    print(f"  {'instrument':<28}{'median':>11}{'IQR':>21}{'max':>11}"
          f"{'headroom@median':>18}{'fail':>6}")
    summary = {}
    for label, v in paths.items():
        st = _stats(v["residual_s"])
        hr = cap / st["median"] if st["median"] > 0 else float("inf")
        hr_worst = cap / st["max"] if st["max"] > 0 else float("inf")
        summary[label] = {**st, "headroom_median": hr, "headroom_worst": hr_worst,
                          "n_failed_mlps": v["n_failed_mlps"],
                          "flops_median": float(np.median(v["flops"]))}
        print(f"  {label:<28}{st['median'] * 1e3:>9.2f}ms"
              f"{st['q25'] * 1e3:>10.2f}-{st['q75'] * 1e3:<9.2f}ms"
              f"{st['max'] * 1e3:>9.2f}ms{hr:>16.1f}x{v['n_failed_mlps']:>6}")

    sub = summary["whest --runner subprocess"]["median"]
    loc = summary["whest --runner local"]["median"]
    life = summary["lifecycle_eval in-process"]["median"]
    ratios = {
        "local / subprocess": loc / sub,
        "lifecycle / subprocess": life / sub,
        "lifecycle / local": life / loc,
    }
    print()
    print("  RATIOS of medians (the only part of a wall-time reading that transfers)")
    for k, v in ratios.items():
        print(f"    {k:<26} {v:.3f}x")

    print()
    print("  READ THIS BEFORE USING ANY NUMBER ABOVE")
    print("  - the DEPLOYMENT AUTHORITY is the subprocess row. The package itself")
    print("    prefers the worker's own stats over the host context.")
    print("  - no GRADED Phase-2 report exists, so even the subprocess row is a")
    print("    proxy for the grader, not the grader.")
    print("  - Phase 1 measured 70.85 ms local / 54.47 ms subprocess / 9.84 ms")
    print("    graded on a DIFFERENT shape. Do not carry its ~7x rule of thumb")
    print("    into Phase 2 without re-deriving it.")
    if life < sub:
        print("  - NOTE: the in-process figure is BELOW the subprocess one here, so")
        print("    a local headroom margin is NOT the conservative one. The sign of")
        print("    this bias is a measurement, not a constant.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "artifact": "p2_residual_paths",
            "artifact_version": 1,
            "candidate": args.candidate,
            "round": r.tag,
            "shape": {"width": r.width, "depth": r.depth},
            "residual_cap_s": cap,
            "n_mlps": args.n_mlps,
            "paths": paths,
            "summary": summary,
            "ratios": ratios,
            "note": (
                "Wall-time absolutes are host-specific and do not transfer. The "
                "subprocess row is the deployment authority and is itself a proxy: "
                "no graded Phase-2 report exists. Draws no seed block; the official "
                "rows are CONFIRM tier and read-only, and nothing is selected here."
            ),
        }, fh, indent=2)
    print(f"\n  wrote {os.path.relpath(args.out, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
