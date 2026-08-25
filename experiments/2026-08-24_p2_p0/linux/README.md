# P2-P0 gate clause G4 — the memory path, provoked on Linux

G4 requires that the residual cliff, wall, **memory**, over-budget and
worker-restart paths be **PROVOKED, not asserted**. Four fire on Windows. The
8,192 MB memory limit cannot: Windows has no `resource` module, so
`whestbench/subprocess_worker.py:172-183` catches the `setrlimit` failure and the
worker runs with **no cap at all**.

That was recorded honestly and it was still not the check. **"This host cannot"
is not "the path fires."** These artifacts provoke it where the limit exists.

## Environment

| | |
|---|---|
| distro | Ubuntu 24.04.1 LTS (Noble Numbat), WSL2 |
| kernel | Linux 6.6.87.2-microsoft-standard-WSL2 x86_64 |
| python | 3.12.3 |
| whestbench | **0.16.0** |
| flopscope | **0.12.0** |
| numpy | 2.2.6 |
| `RLIMIT_AS` | **available** |
| host RAM | 30 GiB (never at risk — see below) |

Same official stack as Windows. The dataset is the Windows HF cache, reused via
`HF_HOME=/mnt/c/Users/samsu/.cache/huggingface`, so both platforms score the
organisers' identical `v2-phase2` bytes.

## Configuration exercised

| | |
|---|---|
| runner | `subprocess` — the only one that calls `setrlimit` |
| dataset | `hf://aicrowd/arc-whestbench-public-2026@v2-phase2`, split `mini`, 3 MLPs |
| shape | `1024 x 16` |
| `flop_budget` | `2,199,023,255,552` (`2**41`) |
| `lambda_flops_per_second` | `0.0` |
| `residual_wall_time_limit_s` | `0.4` |
| `wall_time_limit_s` | `120.0` |
| `memory_limit_mb` | `8192` |

## The provocation, and why it is safe

`RLIMIT_AS` bounds **virtual address space**, not resident memory. So the
sacrificial estimator requests an anonymous `mmap` of **9 GiB against the 8 GiB
cap and never writes to it**. The kernel refuses the reservation outright; no page
is ever faulted in, and **9 GiB is never filled**. Requesting address space is
exactly what the limit governs, which makes this the cheapest honest way to
exercise it.

**A failing `mmap` proves nothing on its own** — it could be a defect in the probe.
So a **256 MiB control mapping is requested first and must succeed**. Only then
does the 9 GiB refusal mean what it claims.

The worker's own limit is read and recorded **before** either attempt, so the cap
is proven applied rather than assumed.

## Result

From `rlimit_as_worker_report.json`, written inside the worker:

```json
{
  "rlimit_as_soft": 8589934592,          // exactly 8 GiB -- the cap WAS applied
  "rlimit_as_soft_gib": 8.0,
  "rlimit_as_is_infinity": false,
  "control_mmap": "OK",                   // 256 MiB succeeded: mechanism works
  "provocation_request_gib": 9.0,
  "provocation_mmap": "OSError: [Errno 12] Cannot allocate memory",
  "provocation_refused": true             // refused at mmap time, never touched
}
```

Failure classification and isolation, from `memory_gate_report.json`:

```text
n_failed_mlps = 1
failure_breakdown = {budget_exhausted: 0, time_exhausted: 0,
                     residual_wall_time_exhausted: 0,
                     combined_budget_exhausted: 0, error: 1}

  MLP 0   mse = 9.434763e-01   error = OSError: [Errno 12] Cannot allocate memory
  MLP 1   mse = 1.123911e+00   error = none      <- ran and was scored
  MLP 2   mse = 1.230982e+00   error = none      <- ran and was scored
```

`RLIMIT_AS` **refuses an allocation rather than killing the process**, so this
takes the grader's generic exception path: that MLP is zeroed and priced at
multiplier 1.0, the worker survives, and the following MLPs are scored normally.

## Both failure routes are covered, and they are different

| route | mechanism | what happens | where |
|---|---|---|---|
| **memory limit** | `RLIMIT_AS` refuses the allocation | exception in `predict`; **worker survives**; later MLPs scored | this directory |
| **worker death** | `os._exit(9)` | worker dies; **runner restarts it**; later MLPs scored | `local_sim/test_deployment_envelope.py` §6, run on both platforms |

The hard-exit test is preserved, not replaced. On Linux the full envelope suite
runs **21/21 with only `predict_response_timeout_s` unprovable** — and that one
never binds under Phase 2's 120 s wall.

## Files

| file | what |
|---|---|
| `probe_env.sh` | environment and `RLIMIT_AS` availability |
| `setup_venv.sh` | builds the Linux venv on the official stack |
| `sacrificial_memory.py` | **the estimator that provokes the limit.** Used by BOTH the standalone script and `test_deployment_envelope.py`, so the transcript and the shipped code path run byte-identical code |
| `run_memory_gate.sh` | the standalone provocation driver |
| `memory_gate_transcript.txt` | its raw transcript |
| `memory_gate_report.json` | the official `whest run --json` report |
| `memory_gate_stderr.txt` | worker stderr |
| `rlimit_as_worker_report.json` | what the worker saw, written before either mmap |
| `run_envelope_linux.sh` | the FULL envelope suite on Linux |
| `envelope_linux_transcript.txt` | its raw transcript — 21/21 |

## What this does not establish

Nothing about the **grader's** host. This proves the enforcement path in
whestbench 0.16 fires when `RLIMIT_AS` exists, on a Linux kernel, at the live
shape and configuration. The grader's own limit, hardware and headroom remain
unobserved — no graded Phase-2 report exists.

Peak memory for the shipped estimator on Linux measures **479.2 MB** against the
8,192 MB cap. As on Windows that is a **lower bound on the charged quantity**, so
the implied headroom is an **upper** bound, never a safety margin.
