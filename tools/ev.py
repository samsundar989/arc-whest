"""THE evaluator. One candidate in, one RESULTS.md row out.

    python tools/ev.py <candidate.py> [--panel dev|fit|val|fresh_a|fresh_lock|holdout_legacy]
                       [--n 100]
                       [--baseline <file.py>] [--jobs K] [--note "..."]

PLAN §4. There is one evaluator, one results table and one decision rule; this
file is the first of the three.

What it reports
---------------
* **raw MSE** — mean final-layer MSE over the panel, against the organizers'
  1e9-sample labels (noise floor at the scored layer 16 ~7.7e-11, ~0.3% of today's
  raw MSE; it is 6.8e-10 at layer 1 and falls with depth -- RESULTS.md Anatomy 2).
* **paired ratio vs the baseline** — mean of the per-net ADJUSTED score
  difference with a studentised-t 95% interval and wins/N, on common random
  numbers. The arithmetic difference is the competition's own identity; the
  geometric mean of per-net ratios is a stability diagnostic with no authority.
* **projected public score** — mean(MSE · max(0.1, F/2^41)) ÷ `PUBLIC_FACTOR` (1.038).
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
import tempfile
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
#: DEV 100-net mini score / this = predicted public score. EIGHT observations:
#: 1.0386 / 1.0385 / 1.0339 / 1.0426 / 1.0471 / 1.0458 / 1.0397 / 1.0361,
#: mean 1.0403. History: 1.037 -> 1.042 on 2026-09-22 because three rows in a
#: row graded BETTER than projected; LOWERED 1.042 -> 1.038 on 2026-09-25
#: because the next two graded WORSE than projected (v57 #332129 by 0.22%,
#: v60_ws400 #332223 by 0.54%), putting the implied factor at 1.0397 / 1.0361.
#: Neither breached PLAN §5's 2% recalibration trigger, but 1.042 made the 3%
#: submission gate ~0.4% LOOSER than intended, and a gate that is silently loose
#: is worse than one that is tight. At 1.038 projections come out slightly
#: pessimistic again, which is the side this constant exists to err on.
#: Change this and the matching block in RESULTS.md § Calibration TOGETHER.
PUBLIC_FACTOR = 1.038
#: FRESH score (floor-subtracted) / this = predicted public score. PLAN rev 18 §2:
#: FRESH-A replaces DEV as the acceptance panel. Set 2026-10-06 from Lane V's V4 table
#: (`docs/research/2026-10-06_laneV_report.md`): fresh / graded = 1.0177 (v73 FRESH-A),
#: 1.0197 (v68_ws FRESH-A), 1.0186 / 1.0213 (FRESH-D), mean 1.0193. One factor for all
#: three panels: they are draws from the same distribution. Recalibrate if a graded row
#: deviates from its FRESH-A projection by > 2 %.
FRESH_PUBLIC_FACTOR = {"fresh_a": 1.019, "fresh_lock": 1.019, "fresh_d": 1.019}
#: FRESH-LOCK is the designation lockbox: every read is logged, and a fourth read
#: is refused unless the reader says it is the designation read.
FRESH_LOCK_LOG = ROOT / "experiments" / "fresh_lock_reads.jsonl"
FRESH_LOCK_MAX_READS = 3
#: SERVER WALL -- REFIT 2026-09-23 on nine graded runs (six candidates), using
#: the per-net telemetry split the server reports: `flopscope_server_overhead_wall_s`
#: (per op call) and `flopscope_server_kernel_wall_s` (the arithmetic itself).
#:
#:     run          FLOPs    ops    median  worst | kernel ps/FLOP | overhead us/op
#:     F21 #330950  928.5G 14,091   89.5   93.2  |      90.0      |     408
#:     v31 #331317  662.0G 13,053   66.0   68.1  |      91.8      |     391
#:     C10 #331816  480.9G 23,421   83.1   87.6  |     154.8      |     361
#:     v50 #331863  473.4G 25,728   84.4   88.7  |     158.6      |     351
#:     v53 #331880  502.4G 25,140   92.8   96.5  |     166.2      |     360
#:     v50 #331939  (rerun)         85.0   87.7  |     160.1      |     350
#:     v55 #331967  475.4G 23,645   92.8  114.5  |     176.3      |     371   noisy
#:     v55 #331980  (rerun)         92.5  113.8  |     175.8      |     366   noisy
#:     v50 #331989  (rerun)         87.0  104.1  |     163.6      |     360   noisy
#:
#: Overhead is a flat ~0.36 ms per op call. KERNEL TIME PER FLOP IS NOT A
#: CONSTANT: it is ~91 ps with Strassen leaves of 64 (F21, v31: depth 4,
#: `_STRASSEN_MIN_BLOCK = 64`) and 155-176 ps with leaves of 32 (every depth-5
#: chain) -- small leaves run the server's BLAS at ~55% efficiency. The age-5
#: chains are a further ~10% slower per FLOP (memory traffic of the bigger
#: exact tier). That is why the old FLOPs+ops fit could not see either effect.
#:
#: The worst-of-100 is then the median times a TAIL FACTOR that depends on the
#: day, not the candidate: 1.03-1.055 on a quiet server, 1.20-1.235 on a noisy
#: one (v50 identical bytes: 87.7 s at 03:00 UTC, 104.1 s at 10:00 UTC Sep 23).
WALL_OVH_MS_PER_OP = 0.371          # worst observed median rate
KERNEL_PS_PER_FLOP = {              # keyed on (min leaf, age-5-or-more)
    (64, False): 92.0, (64, True): 94.0,   # (64, True): v56b #332063 measured 92.8
    (32, False): 164.0, (32, True): 177.0,
}
WALL_TAIL_QUIET = 1.055
WALL_TAIL_NOISY = 1.24


def wall_class(candidate: Path) -> tuple:
    """(min Strassen leaf, age>=5) read from the candidate's module constants.

    A file that routes some products at a different depth still reports its
    global constants, so a mixed file is classed by its SMALLEST leaf -- the
    conservative side. Its real wall must be read from one canary."""
    import re
    src = candidate.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^_STRASSEN_MIN_BLOCK\s*=\s*(\d+)", src, re.M)
    a = re.search(r"^_SHARED_AGE\s*=\s*(\d+)", src, re.M)
    leaf = int(m.group(1)) if m else 32
    return (64 if leaf >= 64 else 32, bool(a and int(a.group(1)) >= 5))


def projected_server_median_s(flops: float, ops: float, cls: tuple = (32, True)) -> float:
    return ops * WALL_OVH_MS_PER_OP * 1e-3 + flops * KERNEL_PS_PER_FLOP[cls] * 1e-12


def projected_server_wall_s(flops: float, ops: float, cls: tuple = (32, True),
                            noisy: bool = False) -> float:
    """Server worst-of-100 wall: median x the day's tail factor."""
    return projected_server_median_s(flops, ops, cls) * (
        WALL_TAIL_NOISY if noisy else WALL_TAIL_QUIET)


#: Server residual seconds per op call. Three graded anchors:
#:     F21  14,091 ops -> 0.156232 s   11.087 us/op   2.56x headroom
#:     v31  13,053 ops -> 0.140244 s   10.744 us/op   2.85x headroom
#:     C10  23,421 ops -> 0.236255 s   10.087 us/op   1.69x headroom
#: The rate FALLS as ops rise and is not even monotone between F21 and v31, so it
#: is a mix-dependent cost and three points do not define a line. Quote the WORST
#: rate: the cliff FAILS the MLP outright, so over-projecting is free.
US_PER_OP = (0.236255e6 / 23421, 0.156232e6 / 14091)

#: The per-MLP residual cliff. Exceeding it FAILS the MLP, which scores it as
#: zeros -- about 0.9 against a good score of 6e-9. There is no partial credit.
RESIDUAL_CLIFF_S = 0.4

#: What a candidate is actually gated on: PLAN D7's finals bound, not the cliff.
#: Gating canaries at the finals bound means no candidate is ever built on margin
#: a designated entry could not keep, which is the failure D7 exists to prevent.
RESIDUAL_GATE_S = 0.33

#: Op ceiling. RAISED 24,500 -> 26,000 on 2026-09-22, which PLAN §5 explicitly
#: provides for ("raise/lower after the first graded residual reading from
#: C10"). That reading: 23,421 ops billed 0.236255 s, i.e. 10.087 us/op and
#: 1.69x headroom -- comfortably better than the 0.252-0.260 s we had projected.
#: At the WORST of the three graded rates, 11.09 us/op, 26,000 ops is 0.288 s:
#: 1.39x headroom against the cliff and inside D7's 0.33 s bound. At C10's own
#: measured rate it is 0.262 s. The residual gate above is the binding
#: constraint; this ceiling is the cheap proxy that catches a regression early.
OP_CEILING = 26_000

#: Prediction workers. Bounded by free RAM: one worker holds ~3.8 GiB of chain
#: at width 1024, and this box has 16 logical / 8 physical cores but only ~30
#: GiB genuinely free. Four fits with room to spare.
#:
#: For most of 2026-09-21 the pool appeared not to scale past one worker, and
#: three separate causes were proposed and are recorded here because each was
#: WRONG: the HF hub's rate limiter, BLAS thread oversubscription, and the
#: Windows memory manager trimming working sets. The actual cause was a pipe
#: deadlock in this file -- see the comment on the Popen calls below. None of
#: the three hypotheses was tested against the one measurement that settled it
#: (per-process CPU seconds: a blocked worker had accumulated 3 of them).
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

#: What one worker holds at width 1024, used ONLY when the candidate's own
#: footprint has not been measured yet. Prefer the measured value.
WORKER_GIB = 5.2

#: Multiplier on a candidate's measured peak RSS to get what a worker really
#: needs. Peak RSS is a *settled* figure and badly understates the transient
#: allocations a Strassen chain makes: on 2026-09-22 two candidates with peak
#: RSS of 3.8 and 5.0 GiB both produced MemoryError on 4 workers, and the
#: failing allocations were 525 MiB Strassen batch tensors (7^5 x 256 x 32).
#: 2.5x is what makes the observed failures fit; it is empirical, not derived.
TRANSIENT_FACTOR = 2.5

#: Headroom left for the OS and everything else on the box.
RESERVE_GIB = 6.0


def _free_gib() -> float:
    """Physical memory actually free right now, or nan if it cannot be read."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", wintypes.DWORD),
                            ("dwMemoryLoad", wintypes.DWORD),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _MS()
            st.dwLength = ctypes.sizeof(st)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return float("nan")
            return st.ullAvailPhys / 2 ** 30
        except Exception:
            return float("nan")
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2 ** 20
    except Exception:
        pass
    return float("nan")


def _fit_pool_to_memory(jobs: int, quiet: bool, worker_gib: float = WORKER_GIB) -> int:
    """Shrink the pool to what free RAM can actually hold.

    On 2026-09-22 a 4-worker run of a candidate needing ~5 GiB each produced 51
    MemoryError nets out of 100 -- scored as zeros, which is exactly how a real
    failure looks. The gate caught it, but a run that merely came CLOSE to the
    limit would have produced quietly degraded numbers instead. So the pool is
    sized from measured free memory, not from a constant, and never silently:
    when it shrinks it says so.
    """
    free = _free_gib()
    if free != free:  # nan
        return jobs
    allowed = max(1, int((free - RESERVE_GIB) // worker_gib))
    if allowed >= jobs:
        return jobs
    if not quiet:
        print(f"  [mem] {free:.1f} GiB free, {worker_gib:.1f} GiB per worker -> "
              f"{allowed} workers, not {jobs}", flush=True)
    return allowed


def _cache_dir(sha: str, panel: str) -> Path:
    return CACHE / sha / panel


def predictions(candidate: Path, panel: str, n: int, jobs: int,
                force: bool = False, quiet: bool = False,
                worker_gib: float = WORKER_GIB):
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
    # REPAIR. A run with failed nets is never cached as a result, but its good nets are
    # kept in `.partial.npz`; the next run recomputes ONLY the failed indices (a local
    # MemoryError under commit pressure is not the candidate's fault, and re-running a
    # whole 100-net panel for two nets cost an hour twice on 2026-10-10).
    part_p = cdir / f"n{n}.partial.npz"
    repair = None
    if part_p.exists() and not force:
        with np.load(part_p) as d:
            repair = (np.array(d["preds"]), np.array(d["truth"]), [int(i) for i in d["failed"]])
        spans = [(i, i + 1) for i in repair[2]]
        jobs = _fit_pool_to_memory(max(1, min(jobs, len(spans))), quiet, worker_gib)
        if not quiet:
            print(f"  [repair] {len(spans)} failed nets from a partial run: {repair[2]}", flush=True)
    else:
        jobs = max(1, min(jobs, total))
        jobs = _fit_pool_to_memory(jobs, quiet, worker_gib)
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

    # EACH WORKER GETS ITS OWN LOG FILES, NOT A PIPE.
    #
    # `stdout=PIPE, stderr=PIPE` with a single `communicate()` at the end is a
    # deadlock, and it is the bug that ate the evening of 2026-09-21. The
    # candidate emits Python warnings with source context on stderr; once a
    # worker writes ~64 KiB the OS pipe buffer fills, and nothing drains it
    # until communicate(), which does not run until the process exits. The
    # worker blocks forever at ~0% CPU, a few hundred MiB resident, having just
    # entered predict -- which reads as "mysteriously slow" and was
    # mis-diagnosed three times (HF locks, BLAS oversubscription, the Windows
    # memory manager) before the process priorities and CPU times made it
    # obvious. Files have no buffer limit and keep the diagnostics.
    procs = []
    t0 = time.perf_counter()
    for lo, hi in spans:
        out = tmp / f"{lo}_{hi}.npz"
        fo = open(tmp / f"{lo}_{hi}.out", "w", encoding="utf-8")
        fe = open(tmp / f"{lo}_{hi}.err", "w", encoding="utf-8")
        procs.append(((lo, hi), out, (fo, fe), subprocess.Popen(
            [sys.executable, str(TOOLS / "_predict_worker.py"), str(candidate),
             panel, str(lo), str(hi), str(out)],
            cwd=str(ROOT), env=env, stdout=fo, stderr=fe)))
    if not quiet:
        print(f"  {len(spans)} workers x {per} BLAS threads over {total} nets ...",
              flush=True)

    parts, errors, peak = {}, [], 0.0
    for (lo, hi), out, (fo, fe), pr in procs:
        rc = pr.wait()
        fo.close()
        fe.close()
        so = (tmp / f"{lo}_{hi}.out").read_text(encoding="utf-8", errors="replace")
        if rc != 0:
            se = (tmp / f"{lo}_{hi}.err").read_text(encoding="utf-8", errors="replace")
            raise SystemExit(f"worker [{lo}:{hi}] failed rc={rc}\n{se[-3000:]}")
        line = [l for l in so.strip().splitlines() if l.startswith("{")]
        info = json.loads(line[-1]) if line else {}
        errors += info.get("errors", [])
        peak = max(peak, float(info.get("peak_rss_mib", 0.0)))
        # CLOSE IT. `np.load` on an .npz returns a lazy NpzFile that holds the
        # file handle open until closed, and on Windows an open handle makes the
        # later unlink fail with WinError 32 -- after the whole panel has been
        # computed. Copy the two arrays out and let the handle go.
        with np.load(out) as d:
            parts[lo] = (np.array(d["preds"]), np.array(d["truth"]))
    wall = time.perf_counter() - t0

    if repair is not None:
        preds, truth = repair[0].copy(), repair[1].copy()
        for lo in parts:
            preds[lo:lo + 1], truth[lo:lo + 1] = parts[lo]
    else:
        preds = np.concatenate([parts[lo][0] for lo in sorted(parts)], axis=0)
        truth = np.concatenate([parts[lo][1] for lo in sorted(parts)], axis=0)
    # NEVER CACHE A RUN THAT HAD FAILURES. A failed net is stored as the zeros
    # the grader would score, which is correct for reporting but poison to reuse:
    # the next run would hit the cache, skip the compute, and re-report a number
    # produced by a transient local OOM as though it were the candidate's.
    if not errors:
        np.savez_compressed(blob, preds=preds, truth=truth)
        part_p.unlink(missing_ok=True)
    else:
        np.savez_compressed(part_p, preds=preds, truth=truth,
                            failed=np.array(sorted(int(e["i"]) for e in errors)))
    meta = {"sha256": sha, "panel": panel, "n": int(preds.shape[0]),
            "workers": len(spans), "blas_threads_each": per,
            "wall_s": round(wall, 1), "per_net_s": round(wall / total, 2),
            "worker_peak_rss_mib": peak, "failed_nets": errors,
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    meta_p.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    # Keep the .err files: they are where a candidate's warnings live, and a
    # candidate that starts emitting new ones is telling you something.
    for (lo, hi), out, _fhs, _pr in procs:
        out.unlink(missing_ok=True)
        (tmp / f"{lo}_{hi}.progress").unlink(missing_ok=True)
    meta["cached"] = False
    return preds, truth, meta


def label_floors(panel: str, n: int) -> Optional[np.ndarray]:
    """Per-net expected label-noise MSE at the scored layer, or None (organizer
    panels: their N = 1e9 floor ~7.7e-11 is ~0.3% of raw and is not subtracted).

    Fresh labels are our N = 1e8 Monte Carlo, whose floor (~7.7e-10, ~3.6% of raw)
    must come off an ABSOLUTE number. It is common to both arms of a pair, so
    paired differences are reported raw.
    """
    if panel not in panels.FRESH:
        return None
    sys.path.append(str(TOOLS / "fresh"))
    import gen  # noqa: PLC0415
    return np.array([gen.load_label(panel, i)["floor"][-1]
                     for _s, i, _seed, _nm in panels.nets(panel, n)], dtype=np.float64)


def fresh_lock_reads() -> List[dict]:
    if not FRESH_LOCK_LOG.exists():
        return []
    return [json.loads(l) for l in FRESH_LOCK_LOG.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def fresh_lock_gate(candidate: Path, baseline: Optional[Path], designating: bool,
                    note: str) -> None:
    """Log this FRESH-LOCK read; refuse a 4th unless it is the designation read.

    A read is one ev.py invocation, cached or not -- a cached result still shows
    the number to whoever ran it."""
    prior = fresh_lock_reads()
    if len(prior) >= FRESH_LOCK_MAX_READS and not designating:
        raise SystemExit(
            f"FRESH-LOCK has been read {len(prior)} times ({FRESH_LOCK_LOG}); the limit is "
            f"{FRESH_LOCK_MAX_READS}. It exists to designate finals (PLAN rev 18 §2). "
            "Pass --i-am-designating only for the designation read itself.")
    FRESH_LOCK_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "read": len(prior) + 1, "candidate": candidate.name,
             "candidate_sha256": sha256_file(candidate),
             "baseline": baseline.name if baseline else None,
             "baseline_sha256": sha256_file(baseline) if baseline else None,
             "designating": bool(designating), "note": note}
    with open(FRESH_LOCK_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print(f"[ev] FRESH-LOCK read {entry['read']} of {FRESH_LOCK_MAX_READS} logged "
          f"-> {FRESH_LOCK_LOG}", flush=True)


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
    the shim has never been on the path. FLOPs are net-independent, so a fresh
    panel is costed on DEV net 0 (the cost worker reads organizer rows only).
    """
    if panel in panels.FRESH:
        panel, index = "dev", 0
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

def memory_gate(candidate: Path, repeats: int = 3, out: Optional[Path] = None) -> Dict:
    # 3, not 2: `linux_memory_gate.py` refuses `--repeats < 3` ("must be >= 3 for
    # lifecycle evidence"), and with 2 this silently reported UNAVAILABLE
    # (v55, 2026-09-23). Every passing gate on record used 3.
    """WSL, `RLIMIT_AS` 8,192 MB, f64 read-only, `repeats` predictions in ONE worker.

    Windows has no `RLIMIT_AS`, so a local peak RSS is accounting and never a
    test passed (`docs/GOTCHAS.md`). This shells out to the existing gate rather
    than reimplementing it; `repeats > 1` is what shows a cache that grows
    across a suite, which a single prediction cannot.
    """
    dest = out or (ROOT / "experiments" / "_ev" / f"mem_{candidate.stem}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected_sha = sha256_file(candidate)
    # A prior PASS must never survive a failed launch. Each invocation gets a
    # private output path; only its successful, hash-matching result is saved.
    with tempfile.TemporaryDirectory(prefix="memory_gate_", dir=dest.parent) as tmp:
        fresh = Path(tmp) / "result.json"
        try:
            r = subprocess.run(
                [sys.executable, str(LOCAL_SIM / "linux_memory_gate.py"),
                 "--candidate", str(candidate), "--limit-mb", "8192",
                 "--weights-mode", "f64_readonly", "--repeats", str(repeats),
                 "--json", str(fresh)],
                cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
        except OSError as exc:
            return {"status": "UNAVAILABLE", "detail": str(exc)}
        if r.returncode != 0:
            return {"status": "FAIL", "detail": f"memory gate exited {r.returncode}: "
                    + (r.stderr or r.stdout or "")[-600:]}
        if not fresh.exists():
            return {"status": "UNAVAILABLE", "detail": "memory gate produced no fresh JSON"}
        try:
            result = json.loads(fresh.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return {"status": "FAIL", "detail": f"invalid memory gate JSON: {exc}"}
        if not isinstance(result, dict) or result.get("candidate_sha256") != expected_sha:
            return {"status": "FAIL", "detail": "memory gate source SHA256 mismatch"}
        if sha256_file(candidate) != expected_sha:
            return {"status": "FAIL", "detail": "candidate changed during memory gate"}
        dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result


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
        date=r["date"], file=r["file"], sha8=r["sha8"],
        panel=r.get("panel_label") or r["panel"].upper(),
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
    memory = r.get("memory") or {}
    memory_hash_matches = (bool(r.get("candidate_sha256"))
                           and memory.get("candidate_sha256") == r["candidate_sha256"])
    return [
        # PLAN rev 13 decision 1 (Samuel, 2026-09-25): any projected improvement
        # over the best graded score qualifies; the 3% margin is retired.
        ("panel projected vs best graded", f"better than {best_graded:.4e}",
         f"{proj:.4e} = {better * 100:.2f}% better", better > 0.0),
        ("op calls", f"≤ {OP_CEILING:,}", f"{int(r['ops']):,}", r["ops"] <= OP_CEILING),
        # This absolute wall model is advisory only. PLAN rev 15 §5.8 also
        # requires manual code-path/parent evidence: data-only A <= ~115 s,
        # new-code-path A <= 105 s, B <= 95 s. This row cannot certify that.
        ("projected server wall (noisy day, A policy)", "≤ 115 s",
         f"{r.get('projected_server_wall_noisy_s', float('nan')):.1f} s (quiet day "
         f"{r['projected_server_wall_s']:.1f} s)",
         r.get("projected_server_wall_noisy_s", float("nan")) <= 115),
        ("setup", "< 2 s", f"{r['setup_s']:.2f} s", r["setup_s"] < 2.0),
        ("scalar materializations", "0", f"{int(r['materializations'])}",
         r["materializations"] == 0),
        ("failed nets on panel", "0", f"{len(r['failed_nets'])}", not r["failed_nets"]),
        # A REAL GATE, not advisory. The cliff FAILS the MLP outright and a
        # failed net scores ~0.9 against ~6e-9, so this is the one number where
        # being wrong is unrecoverable. Gated at PLAN D7's finals bound rather
        # than at the cliff, so a canary is never built on margin a final could
        # not keep.
        ("projected server residual", f"≤ {RESIDUAL_GATE_S} s (D7 finals bound)",
         f"{res_hi:.3f} s ({RESIDUAL_CLIFF_S / res_hi:.2f}× headroom vs the "
         f"{RESIDUAL_CLIFF_S} s cliff)",
         res_hi <= RESIDUAL_GATE_S),
        # PLAN rev 18 §2 (2026-10-06): acceptance is read on FRESH-A (100 fresh
        # nets), paired against the incumbent, with the memory gate on THIS hash.
        # DEV-100 is still accepted until the Lane V calibration is merged.
        ("full acceptance panel", "fresh_a (or dev), n = 100",
         f"{r['panel']}, n = {r['n']}", r["panel"] in ("fresh_a", "dev") and r["n"] == 100),
        ("paired vs incumbent", "--baseline given",
         r.get("baseline_name") or "no baseline", bool(r.get("paired"))),
        ("WSL RLIMIT_AS 8 GiB gate", "PASS on candidate SHA256 (--mem)",
         "NOT RUN" if not memory else "{} VmPeak {:.0f} MiB; source hash {}".format(
             memory.get("status", "?"),
             (memory.get("candidate_run", {}).get("vm_peak_bytes", 0) or 0) / 2 ** 20,
             "matches" if memory_hash_matches else "MISMATCH/MISSING"),
         memory.get("status") == "PASS" and memory_hash_matches),
    ]


def evaluate(candidate: Path, panel: str, n: int, jobs: int, baseline: Optional[Path],
             force: bool, do_cost: bool, quiet: bool, do_mem: bool = False) -> Dict:
    t_all = time.perf_counter()
    candidate_sha = sha256_file(candidate)
    print(f"[ev] {candidate.name} on {panel.upper()} n={n}", flush=True)

    # COST FIRST, deliberately. It is one net and about a minute, and it yields
    # this candidate's measured peak RSS -- which is what the prediction pool
    # must be sized from. Sizing from a constant put the box into MemoryError
    # twice on 2026-09-22 and scored 51 and 60 nets as zeros.
    c = cost(candidate, panel) if do_cost else {}
    rss_gib = float(c.get("peak_rss_mib") or 0.0) / 1024.0
    worker_gib = max(2.0, rss_gib * TRANSIENT_FACTOR) if rss_gib else WORKER_GIB
    if not quiet and rss_gib:
        print(f"[ev] cost: {c.get('flops', float('nan')):,.0f} FLOPs, "
              f"{c.get('ops', float('nan')):,.0f} ops, peak RSS {rss_gib:.2f} GiB "
              f"-> budget {worker_gib:.1f} GiB/worker", flush=True)

    preds, truth, pmeta = predictions(candidate, panel, n, jobs, force, quiet,
                                      worker_gib)
    mse = per_net_mse(preds, truth)
    print(f"[ev] accuracy: {'cache' if pmeta.get('cached') else pmeta['wall_s']} "
          f"{'hit' if pmeta.get('cached') else 's'}, raw MSE {mse.mean():.6e}", flush=True)

    flops = float(c.get("flops", float("nan")))
    mult = multiplier(flops)
    scores = mse * mult
    floors = label_floors(panel, n)
    if floors is None:
        abs_scores = scores
        projected = float(scores.mean()) / PUBLIC_FACTOR
    else:
        # ABSOLUTE numbers are floor-subtracted (PLAN rev 18 §2); the paired
        # statistic below uses `scores` -- the floor is common to both arms.
        abs_scores = (mse - floors) * mult
        projected = float(abs_scores.mean()) / FRESH_PUBLIC_FACTOR[panel]

    pair = None
    base_name = ""
    if baseline is not None:
        bc = cost(baseline, panel) if do_cost else {}
        b_rss = float(bc.get("peak_rss_mib") or 0.0) / 1024.0
        b_gib = max(2.0, b_rss * TRANSIENT_FACTOR) if b_rss else WORKER_GIB
        bpreds, btruth, _ = predictions(baseline, panel, n, jobs, False, quiet, b_gib)
        if bpreds.shape != preds.shape:
            raise SystemExit("baseline was evaluated on a different panel size")
        if not np.array_equal(btruth, truth):
            raise SystemExit("baseline truth differs — not common random numbers")
        bscores = per_net_mse(bpreds, btruth) * multiplier(
            float(bc.get("flops", float("nan"))))
        pair = paired(scores, bscores)
        base_name = baseline.stem

    worst_wall = float(pmeta.get("per_net_s", float("nan")))
    r = {
        "date": time.strftime("%Y-%m-%d", time.gmtime()),
        "file": candidate.name,
        "sha8": candidate_sha[:8],
        "candidate_sha256": candidate_sha,
        "panel": panel,
        "panel_label": panels.label(panel, n),
        "n": int(mse.size),
        "raw_mse": float(mse.mean()),
        "label_floor": float(floors.mean()) if floors is not None else None,
        "raw_mse_floor_free": float((mse - floors).mean()) if floors is not None else None,
        "score_cv": float(np.std(abs_scores, ddof=1) / np.mean(abs_scores)),
        "per_net_mse": [float(x) for x in mse],
        "per_net_floor": [float(x) for x in floors] if floors is not None else None,
        "flops": flops,
        "multiplier": mult,
        "ops": float(c.get("ops", float("nan"))),
        "materializations": float(c.get("materializations", float("nan"))),
        "setup_s": float(c.get("setup_s", float("nan"))),
        "worst_wall_local_s": float(c.get("wall_s", worst_wall)),
        "projected_server_wall_s": projected_server_wall_s(
            flops, float(c.get("ops", float("nan"))), wall_class(candidate)),
        "projected_server_wall_noisy_s": projected_server_wall_s(
            flops, float(c.get("ops", float("nan"))), wall_class(candidate), noisy=True),
        "peak_rss_mib": c.get("peak_rss_mib") or pmeta.get("worker_peak_rss_mib"),
        "adjusted_score_dev": float(abs_scores.mean()),
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
    if r.get("label_floor") is not None:
        print(f"  raw MSE            {r['raw_mse']:.6e}   (our N = 1e8 labels; mean floor "
              f"{r['label_floor']:.3e})")
        print(f"  raw MSE floor-free {r['raw_mse_floor_free']:.6e}")
    else:
        print(f"  raw MSE            {r['raw_mse']:.6e}   (label noise floor at layer 16 ~7.7e-11)")
    print(f"  FLOPs / MLP        {r['flops']:,.0f}   = {r['multiplier']:.5f} x 2^41")
    print(f"  op calls           {r['ops']:,.0f}")
    print(f"  materializations   {r['materializations']:,.0f}")
    print(f"  adjusted (panel)   {r['adjusted_score_dev']:.6e}")
    fac = FRESH_PUBLIC_FACTOR.get(r["panel"], PUBLIC_FACTOR)
    print(f"  per-net score CV   {r['score_cv']:.3f}")
    print(f"  PROJECTED PUBLIC   {r['projected_public']:.6e}   (÷ {fac})")
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
        # The absolute wall model is advisory; the separate PLAN rev 15 §5.8
        # check needs measured parent/code-path evidence (see below).
        advisory = name.startswith("projected server wall")
        if not advisory:
            allpass &= bool(ok)
        tag = ("ok  " if ok else "WARN") if advisory else ("PASS" if ok else "FAIL")
        print(f"    {tag}  {name:<32} {thr:<34} {reading}")
    print()
    print("  " + ("ALL CHECKABLE GATES PASS (PLAN rev 15 §5 items 3-7) — still needs "
                  "validate, parity, manual PLAN §5.8 wall evidence, "
                  "validate-package + source hash, ledger"
                  if allpass else "NOT ACCEPTED — gated or incomplete; not submittable"))
    r["accepted"] = bool(allpass)
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
                    help="also run the WSL RLIMIT_AS 8 GiB gate (3 predictions)")
    #: THE §5 BASELINE. v80_hyg #334659, graded 2026-10-07 (prev v79_dress #334576 3.7025275784494673e-09). Update this the moment a
    #: better score grades, or every later candidate is gated against a stale bar.
    ap.add_argument("--best-graded", type=float, default=3.0716825837413753e-09,
                    help="the best graded score, which the §5 gate must beat "
                         "(default: v80_hyg #334659)")
    ap.add_argument("--json", default=None, help="also write the full result here")
    ap.add_argument("--append", action="store_true",
                    help="append the row to RESULTS.md")
    ap.add_argument("--note", default="")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--i-am-designating", action="store_true",
                    help="permit a FRESH-LOCK read beyond the 3-read limit (designation only)")
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

    if panels.trained_fraction(a.panel, a.n) > 0:
        print(f"[ev] WARNING: {panels.label(a.panel, a.n)} -- these nets are in the "
              f"twin's TRAIN set; this is not held-out evidence", flush=True)
    if a.panel == "fresh_lock":
        fresh_lock_gate(cand, base, a.i_am_designating, a.note)
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
        # An explicit marker, not "the next heading": the file grows sections and
        # a row appended into the wrong one is a row nobody reads.
        marker = "\n<!-- ev.py --append inserts new rows directly above this marker."
        row = results_row(r) + "\n"
        if marker in text:
            text = text.replace(marker, row + marker, 1)
        else:
            raise SystemExit(
                "RESULTS.md has lost its --append marker; restore the "
                "'<!-- ev.py --append ... -->' comment under the Rows table.")
        rm.write_text(text, encoding="utf-8")
        print(f"  appended to {rm}")
    # Exit status is the acceptance verdict (0 only when every checkable gate ran
    # and passed), so no wrapper can mistake a completed run for an accepted one.
    return 0 if r.get("accepted") else 2


if __name__ == "__main__":
    sys.exit(main())
