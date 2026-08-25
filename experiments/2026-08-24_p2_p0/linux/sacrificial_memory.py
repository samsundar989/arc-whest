"""A sacrificial estimator that provokes RLIMIT_AS, without touching the memory.

P2-P0 gate clause **G4** — "the residual cliff, wall, memory, over-budget and
worker-restart paths are PROVOKED, not asserted". Every one of those fires on
Windows except the memory limit, because Windows has no `resource` module, so
`whestbench/subprocess_worker.py:172-183` catches the `setrlimit` failure and the
worker runs with **no cap at all**. This file provokes it where the limit exists.

HOW, AND WHY IT IS SAFE
-----------------------
`RLIMIT_AS` bounds a process's **virtual address space**, not its resident memory.
So the provocation is an anonymous `mmap` of **9 GiB against the configured 8 GiB
limit**, which is *never written to*: the kernel refuses the reservation outright
and returns `ENOMEM`. No page is ever faulted in, nothing is filled, and the host's
30 GiB of RAM is never at risk. Requesting address space is exactly what the limit
governs, so this is the cheapest possible way to exercise it honestly.

THE CONTROL, AND WHY IT IS NOT OPTIONAL
---------------------------------------
A failing `mmap` proves nothing on its own -- it could be a defect in this file. So
a **256 MiB** mapping is requested first and must SUCCEED. Only then is the 9 GiB
requested, and only then does its failure mean what it claims to mean: the
allocation was refused *because of the limit*, not because the mechanism is broken.

The worker's own `RLIMIT_AS` is read and recorded **before** either attempt, so the
transcript proves the cap was actually applied rather than assumed.

ONE MLP ONLY
------------
The trigger is a **sentinel file**, not a call counter. A counter lives in module
state, and if the worker dies and is replaced the counter restarts at zero and the
replacement sacrifices its own first MLP too -- every MLP fails and the runner
looks broken when the fault is in the test. That mistake was made once already in
this slice (`local_sim/test_deployment_envelope.py`) and is not repeated.

`RLIMIT_AS` refuses an allocation rather than killing the process, so the expected
outcome here is an **exception inside `predict`** -- the worker survives, that MLP
is zeroed and scored at multiplier 1.0, and the following MLPs run normally. That
is a different failure path from the hard-exit/worker-restart test, which is kept
separately and still exercises the death-and-restart route.

Driven by `experiments/2026-08-24_p2_p0/linux/run_memory_gate.sh`.
"""

import json
import mmap
import os
import resource

import flopscope.numpy as fnp
from whestbench.sdk import BaseEstimator

_SENTINEL = os.environ.get("ARC_WHEST_MEM_SENTINEL", "")
_REPORT = os.environ.get("ARC_WHEST_MEM_REPORT", "")
_REQUEST = int(os.environ.get("ARC_WHEST_MEM_REQUEST", str(9 * 1024 ** 3)))
_CONTROL = int(os.environ.get("ARC_WHEST_MEM_CONTROL", str(256 * 1024 ** 2)))

_GIB = float(1024 ** 3)


def _write(payload):
    if not _REPORT:
        return
    try:
        with open(_REPORT, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except Exception:
        pass


class Estimator(BaseEstimator):
    def setup(self, context) -> None:
        # Nothing here. The provocation belongs inside predict(), where the
        # grader's failure paths are.
        pass

    def predict(self, mlp, budget):
        if _SENTINEL and not os.path.exists(_SENTINEL):
            with open(_SENTINEL, "w", encoding="utf-8") as fh:
                fh.write(str(int(mlp.seed)))

            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            report = {
                "rlimit_as_soft": soft,
                "rlimit_as_hard": hard,
                "rlimit_as_soft_gib": (None if soft == resource.RLIM_INFINITY
                                       else soft / _GIB),
                "rlimit_as_is_infinity": soft == resource.RLIM_INFINITY,
                "control_request_bytes": _CONTROL,
                "provocation_request_bytes": _REQUEST,
                "provocation_request_gib": _REQUEST / _GIB,
                "pid": os.getpid(),
            }

            # --- CONTROL: a small mapping must succeed -----------------------
            try:
                m = mmap.mmap(-1, _CONTROL)
                m.close()
                report["control_mmap"] = "OK"
            except Exception as exc:
                report["control_mmap"] = f"{type(exc).__name__}: {exc}"
                _write(report)
                raise RuntimeError(
                    "control mmap failed -- the provocation below would prove "
                    f"nothing: {type(exc).__name__}: {exc}"
                ) from exc

            # --- PROVOCATION: 9 GiB of address space, never touched ----------
            try:
                big = mmap.mmap(-1, _REQUEST)
            except Exception as exc:
                report["provocation_mmap"] = f"{type(exc).__name__}: {exc}"
                report["provocation_refused"] = True
                _write(report)
                raise
            else:
                big.close()
                report["provocation_mmap"] = "OK -- NOT REFUSED"
                report["provocation_refused"] = False
                _write(report)
                raise RuntimeError(
                    f"RLIMIT_AS did NOT refuse a {_REQUEST / _GIB:.1f} GiB "
                    f"reservation (soft limit {soft}). The memory path is NOT "
                    "enforced in this configuration."
                )

        return fnp.zeros((len(mlp.weights), mlp.width), dtype=fnp.float32)
