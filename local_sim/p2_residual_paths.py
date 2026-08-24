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

THE LEVEL, NOT ONLY THE RATIO -- added after the P2-P0 Skeptic review
---------------------------------------------------------------------
The first version of this file reported only the RATIO between instruments, and
P2-P0 promoted "the three instruments agree to within about +-7%" into
`VALIDATION.md` 2b-ii. An independent Skeptic pointed out that this is the wrong
quantity: a cliff compares against an **absolute level**, and an intra-session
instrument ratio says nothing about how much that level moves BETWEEN sessions.
The same file measured 45.9-53.2 ms in one session and ~69 ms in another.

So this file now reports the level and its between-session spread, and it takes
`--repeat` so a single invocation produces more than one reading. **A single
wall-time reading is not a measurement, and neither is a single session.**

WHY MORE THAN ONE CANDIDATE
---------------------------
`--candidates` takes several files. The one that matters is not the shipped
estimator: it is `local_sim/p2_arms/arm_k2p2.py`, the analytic reference the
target architecture in `docs/GAME_PLAN.md` is built on. Its residual, not the
shipped file's, is the number the carrier slice has to design against.

WHAT IT DOES NOT ESTABLISH
--------------------------
Nothing about the GRADER's seconds. No graded Phase-2 report exists, so the
subprocess path is the best available proxy and is labelled as one. Local absolutes
never transfer (`docs/GOTCHAS.md`); only the ordering and the ratios are the point.

It draws no seed block: it runs the organisers' own public rows, which are CONFIRM
tier and read-only, and it selects nothing.

Run:  uv run python local_sim/p2_residual_paths.py [--n-mlps 20] [--repeat 2]
      uv run python local_sim/p2_residual_paths.py --candidates \
          estimator.py,local_sim/p2_arms/arm_k2p1.py,local_sim/p2_arms/arm_k2p2.py
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
    ap.add_argument("--n-mlps", type=int, default=20)
    ap.add_argument("--candidates", default="estimator.py",
                    help="comma-separated candidate files")
    ap.add_argument("--repeat", type=int, default=1,
                    help="independent passes; between-session spread needs >= 2")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    r = official_config.LIVE
    cap = r.residual_wall_time_limit_s
    cands = [c.strip() for c in args.candidates.split(",") if c.strip()]

    print("=" * 100)
    print("RESIDUAL: the LEVEL against the 0.4 s cliff, and the instrument ratio")
    print("=" * 100)
    print(f"  shape {r.width}x{r.depth}   n = {args.n_mlps}   repeats = {args.repeat}"
          f"   cliff = {cap} s")
    print()

    runs = []
    for rep in range(args.repeat):
        for cand in cands:
            paths = {}
            for label, fn in (
                ("subprocess", lambda c=cand: official_run("subprocess", args.n_mlps, c)),
                ("local", lambda c=cand: official_run("local", args.n_mlps, c)),
                ("lifecycle", lambda c=cand: lifecycle_run(args.n_mlps, c)),
            ):
                print(f"  pass {rep}  {os.path.basename(cand):<18} {label} ...",
                      flush=True)
                paths[label] = fn()
            bad = {k: v["error"] for k, v in paths.items() if "error" in v}
            if bad:
                for k, v in bad.items():
                    print(f"    {k} FAILED: {v[:300]}")
                continue
            runs.append({"pass": rep, "candidate": cand,
                         "paths": paths,
                         "summary": {k: _stats(v["residual_s"])
                                     for k, v in paths.items()},
                         "flops": {k: float(np.median(v["flops"]))
                                   for k, v in paths.items()},
                         "n_failed": {k: v["n_failed_mlps"]
                                      for k, v in paths.items()}})

    if not runs:
        print("  no successful runs")
        return 1

    print()
    print(f"  {'candidate':<22}{'pass':>5}{'instrument':>12}{'median':>11}{'max':>11}"
          f"{'hr@med':>9}{'hr@worst':>10}{'over cap':>10}")
    for x in runs:
        for inst in ("subprocess", "local", "lifecycle"):
            st = x["summary"][inst]
            over = sum(1 for v in x["paths"][inst]["residual_s"] if v > cap)
            print(f"  {os.path.basename(x['candidate']):<22}{x['pass']:>5}{inst:>12}"
                  f"{st['median'] * 1e3:>9.1f}ms{st['max'] * 1e3:>9.1f}ms"
                  f"{cap / st['median']:>8.1f}x{cap / st['max']:>9.1f}x"
                  f"{over:>7}/{st['n']:<3}")

    # THE number the Skeptic asked for: how much the LEVEL moves between passes,
    # on the authority instrument, for the same candidate.
    print()
    print("  BETWEEN-PASS SPREAD of the ABSOLUTE LEVEL (subprocess = the authority)")
    spread = {}
    for cand in cands:
        med = [x["summary"]["subprocess"]["median"]
               for x in runs if x["candidate"] == cand]
        if len(med) < 2:
            print(f"    {os.path.basename(cand):<22} only {len(med)} pass(es) "
                  f"-- NO between-pass term measured. Re-run with --repeat 2.")
            spread[cand] = None
            continue
        lo, hi = min(med), max(med)
        spread[cand] = {"medians_s": med, "ratio_hi_lo": hi / lo}
        print(f"    {os.path.basename(cand):<22} "
              f"{'  '.join('%.1fms' % (m * 1e3) for m in med)}"
              f"   -> spread {hi / lo:.2f}x")

    print()
    print("  READ THIS BEFORE USING ANY NUMBER ABOVE")
    print("  - the DEPLOYMENT AUTHORITY is the subprocess row; and it is itself a")
    print("    PROXY -- no graded Phase-2 report exists.")
    print("  - the cliff tests an ABSOLUTE LEVEL. An instrument-to-instrument ratio")
    print("    does not bound it, and the between-pass spread above is the term")
    print("    that does. Quote headroom with that spread attached or not at all.")
    print("  - Phase 1's ~7x is local-vs-GRADED on a different shape. It has no")
    print("    Phase-2 counterpart and must not be used as a safety factor.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "artifact": "p2_residual_paths",
            "artifact_version": 2,
            "candidates": cands,
            "round": r.tag,
            "shape": {"width": r.width, "depth": r.depth},
            "residual_cap_s": cap,
            "n_mlps": args.n_mlps,
            "repeat": args.repeat,
            "runs": runs,
            "between_pass_spread": spread,
            "note": (
                "Wall-time absolutes are host-specific and do not transfer. The "
                "subprocess row is the deployment authority and is itself a proxy: "
                "no graded Phase-2 report exists. The cliff tests an absolute "
                "LEVEL, so the between-pass spread is reported alongside every "
                "headroom figure -- an instrument ratio does not bound a level. "
                "Draws no seed block; the official rows are CONFIRM tier and "
                "read-only, and nothing is selected here."
            ),
        }, fh, indent=2)
    print(f"\n  wrote {os.path.relpath(args.out, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
