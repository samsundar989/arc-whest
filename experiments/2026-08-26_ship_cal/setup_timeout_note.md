# The first P2-SHIP-CAL mini run died at SETUP_TIMEOUT, and that is worth keeping

First attempt, official settings, `--runner subprocess`, 2026-08-26:

```json
{"ok": false, "error": {"stage": "setup", "code": "SETUP_TIMEOUT",
                        "message": "worker setup timed out."},
 "whestbench_version": "0.16.0"}
```

Diagnosis, measured on the same host minutes later:

| quantity | measured |
|---|---:|
| fresh `import whestbench` + `import flopscope`, three trials | 4.35 s, 3.54 s, 4.22 s |
| importing `estimator_v2_7.py` (which pulls those transitively) | 4.28 s |
| `Estimator.setup()` itself | 0.278 s |
| official setup ceiling | **5.0 s** |
| host state | GPU 93% busy, ~23 GB VRAM in use by another workload; system RAM ~1 GB free |

**This is host contention, not an estimator defect.** The subprocess worker's
setup window has to cover the package imports, and on this loaded host those
alone consume 70-88% of the 5 s ceiling. `docs/CURRENT_STATE.md` already recorded
the risk: *"`import whestbench` alone is 1.3 s on a quiet host and 4.7-9.1 s under
load."* This is that sentence firing for real.

**Why it is worth keeping rather than dismissing.** Setup timeout does not fail
one MLP — it fails the **entire submission**, every MLP scored as zeros. It is the
sharpest cliff in the system and the only one whose margin we cannot measure
honestly from here, because the in-process timer measures a strictly smaller
window than the grader's.

**What was done about it, and what it costs.** The measurement run was repeated
with `--setup-timeout 120`. That deviation is recorded here rather than buried:

- it does **not** affect what the run measures. Setup happens once, off-budget,
  before any MLP; counted FLOPs, per-MLP MSE and per-MLP residual are unchanged;
- it **does** mean this run establishes **nothing about setup compliance**. Gate
  clause "setup <= 5 s" is UNTESTED by it and must be re-measured on a quiet host
  before any submission.

**Standing consequence for the port.** The planned pure-`fnp` port must keep its
own `setup()` far below the ceiling *and* assume the worker's imports may eat most
of it. A carrier that builds a point set in `setup()` is the design most exposed
to this, and `docs/CURRENT_STATE.md` already flags that. Target: our own setup
<= 0.5 s, leaving >= 4.5 s of the window for everything we do not control.
