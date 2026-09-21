"""THE evaluator. One candidate in, one RESULTS.md row out.

    python tools/ev.py <candidate.py> [--panel dev|fit|holdout] [--n 100]
                       [--baseline <file.py>] [--jobs K] [--note "..."]

PLAN §4. There is one evaluator, one results table and one decision rule; this
file is the first of the three.

What it reports
---------------
* **raw MSE** — mean final-layer MSE over the panel, against the organizers'
  1e9-sample labels (noise floor ~9e-11, three orders below anything we measure).
* **paired ratio vs the baseline** — mean of the per-net ADJUSTED score
  difference with a studentised-t 95% interval and wins/N, on common random
  numbers. The arithmetic difference is the competition's own identity; the
  geometric mean of per-net ratios is a stability diagnostic with no authority.
* **projected public score** — mean(MSE · max(0.1, F/2^41)) ÷ 1.037.
* **FLOPs, op calls, scalar materializations** — from the REAL flopscope 0.12.1
  meter on one net under the f64 read-only grader twin. Never from the shim,
  which reads ~18% low.
* **wall and peak RSS**.

The two paths, and why they are different processes
---------------------------------------------------
The **accuracy path** runs `tools/_predict_worker.py` in a pool of subprocesses
with the non-counting numpy shims in front of `sys.path`. It is fast and it
cannot report a FLOP, which is exactly the property that makes the shim safe to
use here. Predictions are cached under `cache/preds/<sha256(source)>/<panel>/`,
so a control is computed once and never again.

The **cost path** runs `local_sim/lifecycle_eval.evaluate` on ONE net with the
genuine flopscope package. FLOPs are net-independent (spread 0 across all 100
nets of every graded run we hold), so one net is the whole cost answer.

There is also a **cupy backend** (`tools/gpu_shim`, `WHEST_BACKEND=cupy`) at
~5.2 s/net against ~50 s/net on the CPU. It is a SCREENING instrument only:
both backends run the chain in float32, the GPU reduces in a different order,
and sixteen layers amplify that into ~1e-3 relative on a per-net MSE. `ev.py`
therefore always uses numpy; `tools/check_backends.py` quantifies the gap, and
the D2 fit is where the GPU earns its keep.

The server FLOP offset
----------------------
`docs/GOTCHAS.md` (2026-09-07) settles this: the offset depends on the opening
cast, and the f64 read-only grader twin reproduces it exactly — it predicted
submission 330212's `flops_used` to delta 0 out of sample. So this file adds
NOTHING to the twin's reading. `whest run --split mini` reads 2^25 low for a
different reason (its local dataset hands out float32 weights, so the opening
cast never runs), which is what PLAN §4's "+2^25" refers to; taking the number
from the twin makes the correction unnecessary rather than approximate.
`--flop-offset` exists to re-check that claim, not to be used routinely.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
LOCAL_SIM = ROOT / "local_sim"
TOOLS = ROOT / "tools"
CACHE = ROOT / "cache" / "preds"

# APPEND, never insert (CLAUDE.md): this process reports FLOPs, so the real
# flopscope and whestbench must win the import.
for p in (str(ROOT), str(TOOLS), str(LOCAL_SIM)):
    if p not in sys.path:
        sys.path.append(p)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# A Windows console is cp1252, and this file prints Delta, multiplication and
# section signs, and writes RESULTS.md rows containing them. Without this the
# run dies in its own report AFTER the compute -- the same class of failure as
# `whest`'s box-drawing output killing a cp1252 console inside its error
# printer (`docs/GOTCHAS.md`). errors="replace" so a console that still cannot
# encode something degrades a character instead of losing the whole result.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np  # noqa: E402

import panels  # noqa: E402

BUDGET = 2 ** 41
#: DEV 100-net mini score / this = predicted public score. Observed 1.0386 /
#: 1.0385 / 1.0339 on three submissions. Recalibrate in RESULTS.md, not here,
#: if a graded row ever lands more than 2% from its projection.
PUBLIC_FACTOR = 1.037
#: One-net grader-twin wall x this = server WORST-of-100 wall. Calibrated on the
#: two graded rows we hold, measured by THIS instrument (`_cost_worker.py`) so
#: the factor and the reading come from the same place:
#:
#:     F21  #330950   twin 58.295 s -> graded worst 93.1597 s   1.5981
#:     v31  #331317   twin 43.547 s -> graded worst 68.1319 s   1.5646
#:
#: The factor bundles two things -- the server being slower, and worst-of-100
#: against our one net -- and that is deliberate: the gate asks whether the
#: WORST net finishes, so the projection must answer that question. The larger
#: of the two anchors is used, because over-projecting wall costs nothing and
#: under-projecting costs the whole submission.
WALL_ANCHORS = {"f21": 93.1597 / 58.295410299964715,
                "v31": 68.1319 / 43.5474146999768}
WALL_FACTOR = max(WALL_ANCHORS.values())
#: Server residual seconds per op call, from the same two rows:
#:     F21  14,091 ops -> 0.156232 s   11.09 us/op
#:     v31  13,053 ops -> 0.140244 s   10.74 us/op
#: The high end is the one to quote, because the cliff FAILS the MLP outright.
US_PER_OP = (0.140244e6 / 13053, 0.156232e6 / 14091)
RESIDUAL_CLIFF_S = 0.4

#: Prediction workers. **Bounded by free RAM, not by cores.** One worker holds
#: ~3.8 GiB of chain at width 1024, and this box has 16 cores but only ~30 GiB
#: genuinely free (editors, browsers, other agents hold the rest). Six workers
#: (23 GiB) tipped the machine into paging on 2026-09-21: Windows trimmed their
#: working sets to ~0.5 GiB and they crawled on page faults while *looking* idle
#: in Get-Process -- low RSS and low CPU, which reads exactly like a deadlock and
#: is not one. Four is the number that fits. Raise it only after checking
#: Win32_OperatingSystem.FreePhysicalMemory, and never past free_GiB / 4.
DEFAULT_JOBS = 4


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# accuracy path
# --------------------------------------------------------------------------- #

def _cache_dir(sha: str, panel: str) -> Path:
    return CACHE / sha / panel


def predictions(candidate: Path, panel: str, n: int, jobs: int,
                force: bool = False, quiet: bool = False):
    """(preds, truth, meta) for `panel`, from cache or a fresh pooled run."""
    sha = sha256_file(candidate)
    cdir = _cache_dir(sha, panel)
    cdir.mkdir(parents=True, exist_ok=True)
    blob = cdir / f"n{n}.npz"
    meta_p = cdir / f"n{n}.json"
    if blob.exists() and meta_p.exists() and not force:
        d = np.load(blob)
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        meta["cached"] = True
        return d["preds"], d["truth"], meta

    total = len(panels.nets(panel, n))
    if total == 0:
        raise SystemExit(f"panel {panel!r} is empty")
    jobs = max(1, min(jobs, total))
    # Contiguous chunks: each worker pays the dataset open once.
    edges = [round(k * total / jobs) for k in range(jobs + 1)]
    spans = [(edges[k], edges[k + 1]) for k in range(jobs) if edges[k] < edges[k + 1]]

    tmp = cdir / "_parts"
    tmp.mkdir(exist_ok=True)
    env = dict(os.environ)
    # OFFLINE, and this is not an optimisation. Six workers opening the dataset
    # at once each resolve the revision against the HF hub; unauthenticated,
    # concurrent and rate-limited, five of the six sat at 4 CPU-seconds for
    # minutes while one proceeded. The panel data is already in the local cache,
    # so offline mode removes the hub from the path entirely. If this ever needs
    # to fetch, run `python tools/panels.py --build` first, online, once.
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    # TWO BLAS threads per worker, not `cpu_count // workers`. This chain barely
    # scales with threads -- one net took 50.02 s at 2 threads and 50.05 s at 4,
    # against 73 s with all sixteen -- so extra threads buy nothing and cost
    # contention. `cpu_count // workers` gave 4 each (16 total on 8 physical
    # cores) and three of four workers crawled through a single net for five
    # minutes while the first raced ahead; a flat 2 each ran all four uniformly
    # at 87.8 s/net. Measured 2026-09-21, both ways.
    per = int(os.environ.get("WHEST_BLAS_THREADS", "2"))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        env[var] = str(per)

    procs = []
    t0 = time.perf_counter()
    for lo, hi in spans:
        out = tmp / f"{lo}_{hi}.npz"
        procs.append(((lo, hi), out, subprocess.Popen(
            [sys.executable, str(TOOLS / "_predict_worker.py"), str(candidate),
             panel, str(lo), str(hi), str(out)],
            cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace")))
    if not quiet:
        print(f"  {len(spans)} workers x {per} BLAS threads over {total} nets ...",
              flush=True)

    parts, errors, peak = {}, [], 0.0
    for (lo, hi), out, pr in procs:
        so, se = pr.communicate()
        if pr.returncode != 0:
            raise SystemExit(f"worker [{lo}:{hi}] failed rc={pr.returncode}\n{se[-3000:]}")
        line = [l for l in so.strip().splitlines() if l.startswith("{")]
        info = json.loads(line[-1]) if line else {}
        errors += info.get("errors", [])
        peak = max(peak, float(info.get("peak_rss_mib", 0.0)))
        d = np.load(out)
        parts[lo] = (d["preds"], d["truth"])
    wall = time.perf_counter() - t0

    preds = np.concatenate([parts[lo][0] for lo in sorted(parts)], axis=0)
    truth = np.concatenate([parts[lo][1] for lo in sorted(parts)], axis=0)
    np.savez_compressed(blob, preds=preds, truth=truth)
    meta = {"sha256": sha, "panel": panel, "n": int(preds.shape[0]),
            "workers": len(spans), "blas_threads_each": per,
            "wall_s": round(wall, 1), "per_net_s": round(wall / total, 2),
            "worker_peak_rss_mib": peak, "failed_nets": errors,
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    meta_p.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    for _, out, _ in procs:
        out.unlink(missing_ok=True)
    meta["cached"] = False
    return preds, truth, meta


def per_net_mse(preds: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """float32, as the grader computes it (`scoring.py`)."""
    p = np.asarray(preds, dtype=np.float32)
    t = np.asarray(truth, dtype=np.float32)
    return ((p - t) ** 2).mean(axis=1, dtype=np.float32).astype(np.float64)


def multiplier(flops: float) -> float:
    return max(0.1, flops / BUDGET)


# --------------------------------------------------------------------------- #
# cost path -- the REAL meter, one net
# --------------------------------------------------------------------------- #

def cost(candidate: Path, panel: str = "dev", index: int = 0) -> Dict[str, float]:
    """FLOPs, op calls and scalar materializations on one net, f64 read-only.

    Runs in a subprocess so the real flopscope is imported in a process where
    the shim has never been on the path.
    """
    out = subprocess.run(
        [sys.executable, str(TOOLS / "_cost_worker.py"), str(candidate), panel, str(index)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise SystemExit(f"cost worker failed rc={out.returncode}\n{out.stderr[-4000:]}")
    line = [l for l in out.stdout.strip().splitlines() if l.startswith("{")]
    if not line:
        raise SystemExit(f"cost worker printed no result\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
    return json.loads(line[-1])


# --------------------------------------------------------------------------- #
# memory -- the only real 8 GiB test we have
# --------------------------------------------------------------------------- #

def memory_gate(candidate: Path, repeats: int = 2, out: Optional[Path] = None) -> Dict:
    """WSL, `RLIMIT_AS` 8,192 MB, f64 read-only, `repeats` predictions in ONE worker.

    Windows has no `RLIMIT_AS`, so a local peak RSS is accounting and never a
    test passed (`docs/GOTCHAS.md`). This shells out to the existing gate rather
    than reimplementing it; `repeats > 1` is what shows a cache that grows
    across a suite, which a single prediction cannot.
    """
    dest = out or (ROOT / "experiments" / "_ev" / f"mem_{candidate.stem}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [sys.executable, str(LOCAL_SIM / "linux_memory_gate.py"),
         "--candidate", str(candidate), "--limit-mb", "8192",
         "--weights-mode", "f64_readonly", "--repeats", str(repeats),
         "--json", str(dest)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if not dest.exists():
        return {"status": "UNAVAILABLE", "detail": (r.stderr or r.stdout)[-600:]}
    return json.loads(dest.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# the paired statistic
# --------------------------------------------------------------------------- #

def paired(cand_scores: np.ndarray, base_scores: np.ndarray) -> Dict[str, float]:
    """Paired arithmetic difference with a studentised-t 95% interval.

    PLAN §4 and `docs/VALIDATION.md` §2b-i: the binding statistic is the paired
    ARITHMETIC score difference. Positive means the candidate is better. The
    percentile bootstrap and BCa are refused here (measured false-promotion
    rates 6.8-8.2% and 9.1-11.5% against a nominal 5%).
    """
    import competition_score as cs  # noqa: PLC0415
    d = np.asarray(base_scores, dtype=np.float64) - np.asarray(cand_scores, dtype=np.float64)
    n = int(d.size)
    lo, hi = cs.paired_diff_ci(d, method="t")
    mean_base = float(np.mean(base_scores))
    mean_cand = float(np.mean(cand_scores))
    with np.errstate(divide="ignore", invalid="ignore"):
        geo = float(np.exp(np.mean(np.log(base_scores / cand_scores))))
    return {
        "n": n,
        "mean_difference": float(d.mean()),
        "relative_improvement": float(d.mean() / mean_base) if mean_base else float("nan"),
        "arithmetic_gain": mean_base / mean_cand if mean_cand else float("nan"),
        "t95_lo": float(lo), "t95_hi": float(hi),
        "wins": int((d > 0).sum()),
        "geometric_gain_diagnostic": geo,
    }


# --------------------------------------------------------------------------- #
# the row
# --------------------------------------------------------------------------- #

def _fmt(x, spec=".4g"):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def results_row(r: Dict) -> str:
    p = r["paired"]
    gain = (f"{p['arithmetic_gain']:.4f}× vs {r['baseline_name']} "
            f"(Δ {p['mean_difference']:+.3e}, t95 [{p['t95_lo']:+.3e}, {p['t95_hi']:+.3e}])"
            if p else "—")
    wins = f"{p['wins']}/{p['n']}" if p else "—"
    return ("| {date} | `{file}` | `{sha8}` | {panel} | {raw:.6e} | {fb:.5f} | {ops:,} | "
            "{wall:.3f} s (local) → {swall:.1f} s proj | {rss} | **{proj:.4e}** | {graded} | "
            "{gain} | {wins} | {note} |").format(
        date=r["date"], file=r["file"], sha8=r["sha8"], panel=r["panel"].upper(),
        raw=r["raw_mse"], fb=r["multiplier"], ops=int(r["ops"]),
        wall=r["worst_wall_local_s"], swall=r["projected_server_wall_s"],
        rss=(f"{r['peak_rss_mib']:.0f} MiB" if r.get("peak_rss_mib") else "—"),
        proj=r["projected_public"], graded=r.get("graded", "PENDING"),
        gain=gain, wins=wins, note=r.get("note", ""))


def gate_table(r: Dict, best_graded: float) -> List[tuple]:
    """PLAN §5, evaluated. Returns (name, threshold, reading, passed)."""
    proj = r["projected_public"]
    better = 1.0 - proj / best_graded
    res_hi = r["ops"] * US_PER_OP[1] * 1e-6
    return [
        ("DEV projected vs best graded", f"≥3% better than {best_graded:.4e}",
         f"{proj:.4e} = {better * 100:.1f}% better", better >= 0.03),
        ("op calls", "≤ 24,500", f"{int(r['ops']):,}", r["ops"] <= 24_500),
        ("projected server wall", "< 100 s",
         f"{r['projected_server_wall_s']:.1f} s", r["projected_server_wall_s"] < 100),
        ("setup", "< 2 s", f"{r['setup_s']:.2f} s", r["setup_s"] < 2.0),
        ("scalar materializations", "0", f"{int(r['materializations'])}",
         r["materializations"] == 0),
        ("failed nets on panel", "0", f"{len(r['failed_nets'])}", not r["failed_nets"]),
        ("projected server residual", f"< {RESIDUAL_CLIFF_S} s cliff (advisory)",
         f"{res_hi:.3f} s ({RESIDUAL_CLIFF_S / res_hi:.2f}× headroom)",
         res_hi < RESIDUAL_CLIFF_S),
    ] + ([] if not r.get("memory") else [
        ("WSL RLIMIT_AS 8 GiB gate", "PASS",
         "{} VmPeak {:.0f} MiB".format(
             r["memory"].get("status", "?"),
             (r["memory"].get("candidate_run", {}).get("vm_peak_bytes", 0) or 0) / 2 ** 20),
         r["memory"].get("status") == "PASS"),
    ])


def evaluate(candidate: Path, panel: str, n: int, jobs: int, baseline: Optional[Path],
             force: bool, do_cost: bool, quiet: bool, do_mem: bool = False) -> Dict:
    t_all = time.perf_counter()
    print(f"[ev] {candidate.name} on {panel.upper()} n={n}", flush=True)

    preds, truth, pmeta = predictions(candidate, panel, n, jobs, force, quiet)
    mse = per_net_mse(preds, truth)
    print(f"[ev] accuracy: {'cache' if pmeta.get('cached') else pmeta['wall_s']} "
          f"{'hit' if pmeta.get('cached') else 's'}, raw MSE {mse.mean():.6e}", flush=True)

    c = cost(candidate, panel) if do_cost else {}
    flops = float(c.get("flops", float("nan")))
    mult = multiplier(flops)
    scores = mse * mult
    projected = float(scores.mean()) / PUBLIC_FACTOR

    pair = None
    base_name = ""
    if baseline is not None:
        bpreds, btruth, _ = predictions(baseline, panel, n, jobs, False, quiet)
        if bpreds.shape != preds.shape:
            raise SystemExit("baseline was evaluated on a different panel size")
        if not np.array_equal(btruth, truth):
            raise SystemExit("baseline truth differs — not common random numbers")
        bc = cost(baseline, panel) if do_cost else {}
        bscores = per_net_mse(bpreds, btruth) * multiplier(
            float(bc.get("flops", float("nan"))))
        pair = paired(scores, bscores)
        base_name = baseline.stem

    worst_wall = float(pmeta.get("per_net_s", float("nan")))
    r = {
        "date": time.strftime("%Y-%m-%d", time.gmtime()),
        "file": candidate.name,
        "sha8": sha256_file(candidate)[:8],
        "panel": panel,
        "n": int(mse.size),
        "raw_mse": float(mse.mean()),
        "flops": flops,
        "multiplier": mult,
        "ops": float(c.get("ops", float("nan"))),
        "materializations": float(c.get("materializations", float("nan"))),
        "setup_s": float(c.get("setup_s", float("nan"))),
        "worst_wall_local_s": float(c.get("wall_s", worst_wall)),
        "projected_server_wall_s": float(c.get("wall_s", worst_wall)) * WALL_FACTOR,
        "peak_rss_mib": c.get("peak_rss_mib") or pmeta.get("worker_peak_rss_mib"),
        "adjusted_score_dev": float(scores.mean()),
        "projected_public": projected,
        "failed_nets": pmeta.get("failed_nets", []),
        "baseline_name": base_name,
        "paired": pair,
        "memory": memory_gate(candidate) if do_mem else None,
        "accuracy_meta": pmeta,
        "cost_meta": c,
        "ev_wall_s": round(time.perf_counter() - t_all, 1),
    }
    return r


def report(r: Dict, best_graded: float) -> None:
    print()
    print(f"  raw MSE            {r['raw_mse']:.6e}   (label noise floor ~9e-11)")
    print(f"  FLOPs / MLP        {r['flops']:,.0f}   = {r['multiplier']:.5f} x 2^41")
    print(f"  op calls           {r['ops']:,.0f}")
    print(f"  materializations   {r['materializations']:,.0f}")
    print(f"  adjusted (panel)   {r['adjusted_score_dev']:.6e}")
    print(f"  PROJECTED PUBLIC   {r['projected_public']:.6e}   (÷ {PUBLIC_FACTOR})")
    if r["paired"]:
        p = r["paired"]
        print(f"  paired vs {r['baseline_name']}:  gain {p['arithmetic_gain']:.4f}×, "
              f"Δ {p['mean_difference']:+.4e}, t95 [{p['t95_lo']:+.4e}, {p['t95_hi']:+.4e}], "
              f"wins {p['wins']}/{p['n']}")
        print(f"    (geometric {p['geometric_gain_diagnostic']:.4f}× — diagnostic only, "
              f"no authority over an outcome)")
    if r["failed_nets"]:
        print(f"  !! {len(r['failed_nets'])} FAILED NETS: {r['failed_nets'][:3]}")
    print()
    print("  PLAN §5 gates")
    allpass = True
    for name, thr, reading, ok in gate_table(r, best_graded):
        allpass &= bool(ok)
        print(f"    {'PASS' if ok else 'FAIL'}  {name:<32} {thr:<34} {reading}")
    print()
    print("  " + ("ALL GATES PASS — eligible for a canary under PLAN §5"
                  if allpass else "GATED — not submittable as it stands"))
    print()
    print("  RESULTS.md row:")
    print("  " + results_row(r))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("candidate")
    ap.add_argument("--panel", default="dev", choices=list(panels.PANELS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--baseline", default=None,
                    help="control candidate file; paired on common random numbers")
    ap.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
                    help=f"prediction worker processes (default {DEFAULT_JOBS}; each "
                         f"holds ~3.8 GiB of chain, so this is bounded by FREE RAM, "
                         f"not by cores)")
    ap.add_argument("--force", action="store_true", help="ignore the prediction cache")
    ap.add_argument("--no-cost", action="store_true",
                    help="skip the real-meter run (accuracy only; no score, no gates)")
    ap.add_argument("--mem", action="store_true",
                    help="also run the WSL RLIMIT_AS 8 GiB gate (2 predictions, ~3 min)")
    ap.add_argument("--best-graded", type=float, default=8.75203025190677e-09,
                    help="the score the §5 3%% gate is measured against")
    ap.add_argument("--json", default=None, help="also write the full result here")
    ap.add_argument("--append", action="store_true",
                    help="append the row to RESULTS.md")
    ap.add_argument("--note", default="")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    cand = Path(a.candidate)
    if not cand.is_absolute():
        cand = ROOT / cand
    if not cand.exists():
        raise SystemExit(f"{a.candidate} does not exist")
    base = None
    if a.baseline:
        base = Path(a.baseline)
        if not base.is_absolute():
            base = ROOT / base
        if not base.exists():
            raise SystemExit(f"baseline {a.baseline} does not exist")

    r = evaluate(cand, a.panel, a.n, a.jobs, base, a.force, not a.no_cost, a.quiet,
                 do_mem=a.mem)
    r["note"] = a.note
    report(r, a.best_graded)

    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1, default=float) + "\n",
                                encoding="utf-8")
    if a.append:
        rm = ROOT / "RESULTS.md"
        text = rm.read_text(encoding="utf-8")
        marker = "\n## Open items"
        row = results_row(r) + "\n"
        if marker in text:
            text = text.replace(marker, row + marker, 1)
        else:
            text += row
        rm.write_text(text, encoding="utf-8")
        print(f"  appended to {rm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
