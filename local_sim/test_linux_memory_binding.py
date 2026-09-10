"""The memory proof is bound to the workload it claims to test — mutation battery.

`docs/research/2026-09-10_validation_audit.md` finding 1. Stage 0 repair R2,
contract `experiments/2026-09-10_stage0_instrument/prediction.md` (committed at
`def1b00`, before the repair).

WHAT WAS WRONG. `certify_candidate.py` section 9 located a `linux_memory_gate`
report by the candidate's sha256 and then checked four things: the cap applied,
the negative control died, `predict_error == ""`, and `status == "PASS"`. Shape,
depth, budget, memory cap, weight representation and package versions are all
CLI flags on `linux_memory_gate.py` (`--width`, `--depth`, `--budget`,
`--limit-mb`, and now `--weights-mode`), and section 9 asked about NONE of them.
So a report for the right bytes, run at 64x4 under a 512 MB cap on float32
weights, satisfied every check and was printed as "the real 8-GiB enforcement".

No stored report is alleged to be misconfigured. This is a reachable
acceptance path, demonstrated below by mutating a GOOD report one field at a
time and requiring each mutation to be refused. A regression test that passes
against the defect tests nothing, so every case here fails at `0fe164c`.

Run: uv run python local_sim/test_linux_memory_binding.py
"""

import copy
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

from certify_candidate import (MEMORY_CAP_MB, MEMORY_REPORT_MIN_VERSION,  # noqa: E402
                              validate_memory_report)

RESULTS = []
SHA = "a" * 64
WIDTH, DEPTH, BUDGET = 1024, 16, 2 ** 41
STACK = {"whestbench": "0.16.1", "flopscope": "0.12.1"}
CAP_B = MEMORY_CAP_MB * 1024 * 1024


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {name:<64} {'ok ' if ok else 'FAILED'}  {detail}")
    return ok


def head(t):
    print(f"\n{t}\n{'-' * len(t)}")


def good_report():
    """A report that SHOULD certify. Every mutation below starts here."""
    return {
        "artifact": "linux_memory_gate",
        "artifact_version": 3,
        "candidate": "estimator_candidate.py",
        "candidate_sha256": SHA,
        "instrument": "local_sim/linux_memory_gate.py",
        "instrument_sha256": "b" * 64,
        "limit_mb": MEMORY_CAP_MB,
        "width": WIDTH, "depth": DEPTH, "budget": BUDGET, "seed": 90000000101,
        "weights_mode": "f64_readonly",
        "repeats": 2,
        "host_platform": "linux",
        "status": "PASS",
        "stack": dict(STACK, numpy="2.2.6", python="3.10.19"),
        "negative_control": {"cap_applied": True, "survived": False},
        "positive_control": {"survived": True},
        "candidate_run": {
            "stage": "done",
            "cap_applied": True,
            "cap_requested_bytes": CAP_B,
            "soft_limit_bytes": CAP_B,
            "predict_error": "",
            "weights_mode": "f64_readonly",
            "weight_dtype": "float64",
            "weight_writeable": False,
            "whestbench_file": "/home/x/.arcwhest-linux/lib/whestbench/__init__.py",
            "flopscope_file": "/home/x/.arcwhest-linux/lib/flopscope/__init__.py",
            "vm_peak_bytes": 1_200 * (1 << 20),
            "repeat_runs": [{"i": 0, "vm_peak_bytes": 1_200 * (1 << 20),
                             "flops_used": 1.376e11, "error": ""},
                            {"i": 1, "vm_peak_bytes": 1_201 * (1 << 20),
                             "flops_used": 1.376e11, "error": ""}],
            "repeat_error": "",
            "repeat_vm_peak_growth_bytes": 1 << 20,
            "repeat_flops_identical": True,
        },
    }


def verdict(lg):
    """(all checks passed, degraded reason, failed check names)."""
    checks, degraded = validate_memory_report(
        lg, candidate_sha256=SHA, width=WIDTH, depth=DEPTH, budget=BUDGET,
        cap_mb=MEMORY_CAP_MB, weights_mode="f64_readonly", stack=STACK)
    bad = [n for n, ok, _ in checks if not ok]
    return (not bad and not degraded), degraded, bad


def t_positive_control():
    head("0. the positive control — a correct report must still CERTIFY")
    ok, degraded, bad = verdict(good_report())
    check("an unmutated, correctly-configured report passes", ok,
          f"{len(bad)} failed; degraded={degraded!r}")


def _mut(name, fn, expect_substr=""):
    lg = good_report()
    fn(lg)
    ok, degraded, bad = verdict(lg)
    detail = (bad[0] if bad else degraded)[:74]
    hit = (not ok) and (not expect_substr or any(expect_substr in b for b in bad)
                        or expect_substr in degraded)
    check(name, hit, detail)


def t_workload_mutations():
    head("1. the WORKLOAD mutations — each was accepted before the repair")

    def small_shape(lg):
        lg["width"], lg["depth"] = 64, 4
    _mut("a report run at 64x4 is REFUSED", small_shape, "certified SHAPE")

    def wrong_depth(lg):
        lg["depth"] = 8
    _mut("...and one at the right width but the wrong depth", wrong_depth,
         "certified SHAPE")

    def loose_cap(lg):
        lg["limit_mb"] = 65536
        lg["candidate_run"]["cap_requested_bytes"] = 65536 * 1024 * 1024
        lg["candidate_run"]["soft_limit_bytes"] = 65536 * 1024 * 1024
    _mut("a report run under a 64 GiB cap is REFUSED", loose_cap,
         "FULL 8,192 MB cap")

    def cap_mismatch(lg):
        # the WORST case: the report DECLARES 8192 but the child read back more
        lg["candidate_run"]["soft_limit_bytes"] = 65536 * 1024 * 1024
    _mut("a report DECLARING the cap whose child read back a looser one",
         cap_mismatch, "READ BACK")

    def wrong_budget(lg):
        lg["budget"] = 272_000_000_000          # the Phase-1 budget
    _mut("a report run at the PHASE-1 budget is REFUSED", wrong_budget,
         "certified BUDGET")

    def f32_weights(lg):
        lg["weights_mode"] = "f32"
        lg["candidate_run"]["weights_mode"] = "f32"
        lg["candidate_run"]["weight_dtype"] = "float32"
        lg["candidate_run"]["weight_writeable"] = True
    _mut("a report run on WRITABLE float32 weights is REFUSED", f32_weights,
         "DEPLOYMENT weight")

    def writable_f64(lg):
        lg["candidate_run"]["weight_writeable"] = True
    _mut("...and one on f64 weights that were not actually read-only",
         writable_f64, "DEPLOYMENT weight")


def t_identity_and_stack():
    head("2. identity, instrument and stack")

    def stale_sha(lg):
        lg["candidate_sha256"] = "c" * 64
    _mut("a report for DIFFERENT bytes is REFUSED", stale_sha, "exact bytes")

    def no_sha(lg):
        lg.pop("candidate_sha256")
    _mut("...and one with no sha256 at all", no_sha, "exact bytes")

    # A version-2 report is DEGRADED: it cannot ANSWER the workload questions,
    # which is an absence of evidence rather than evidence of a violation. It
    # must not certify, and it must ALSO not fail -- the first draft emitted a
    # failing check here and made the certifier return REFUSED for a correct
    # candidate whose only sin was an old memory report. Caught by the full
    # mutation battery's positive control.
    lg = good_report()
    lg["artifact_version"] = 2
    for k in ("weights_mode", "stack", "instrument_sha256"):
        lg.pop(k, None)
    ok, degraded, bad = verdict(lg)
    check("a version-2 report does not certify", not ok, degraded[:60])
    check("...and is reported as DEGRADED, not as a FAILING check",
          bool(degraded) and not bad,
          f"{len(bad)} failing checks (a fail here would make a correct "
          f"candidate REFUSED)")
    check("...and the degraded reason names the version",
          f"artifact_version {2}" in degraded, "")

    def old_wb(lg):
        lg["stack"]["whestbench"] = "0.15.0"
    _mut("a report against a superseded whestbench is REFUSED", old_wb,
         "whestbench")

    def old_fs(lg):
        lg["stack"]["flopscope"] = "0.11.0"
    _mut("...and one against a superseded flopscope", old_fs, "flopscope")

    def no_instrument(lg):
        lg["instrument_sha256"] = ""
    _mut("a report that cannot name the instrument that made it", no_instrument,
         "names the instrument")


def t_controls_and_lifetime():
    head("3. the controls, and suite-lifetime stability")

    def neg_survived(lg):
        lg["negative_control"]["survived"] = True
    _mut("a negative control that SURVIVED withdraws the pass", neg_survived,
         "negative control")

    def pos_died(lg):
        lg["positive_control"]["survived"] = False
    _mut("...and a positive control that died means the machine, not the cap",
         pos_died, "positive control")

    def one_run(lg):
        cr = lg["candidate_run"]
        cr["repeat_runs"] = cr["repeat_runs"][:1]
        cr["repeat_vm_peak_growth_bytes"] = 0
    _mut("a SINGLE prediction cannot prove suite-lifetime memory", one_run,
         "MORE THAN ONCE")

    def growing(lg):
        cr = lg["candidate_run"]
        cr["repeat_vm_peak_growth_bytes"] = 400 * (1 << 20)
    _mut("address space that GROWS across predictions is refused", growing,
         "did not grow")

    def shim(lg):
        lg["candidate_run"]["whestbench_file"] = \
            "C:/repo/local_sim/whestbench/__init__.py"
    _mut("a run that loaded local_sim's SHIMS is refused", shim, "shims")

    def over_cap(lg):
        lg["candidate_run"]["vm_peak_bytes"] = CAP_B + 1
    _mut("a peak above the cap is refused", over_cap, "under the cap")

    def not_pass(lg):
        lg["status"] = "FAIL"
    _mut("a report whose own status is not PASS", not_pass, "under the real cap")


def t_purity():
    head("4. the validator is pure, so the battery above means something")
    lg = good_report()
    before = copy.deepcopy(lg)
    validate_memory_report(lg, candidate_sha256=SHA, width=WIDTH, depth=DEPTH,
                           budget=BUDGET, stack=STACK)
    check("validate_memory_report does not mutate its input", lg == before, "")
    checks, _ = validate_memory_report(None, candidate_sha256=SHA, width=WIDTH,
                                       depth=DEPTH, budget=BUDGET)
    check("a non-dict report is refused, not crashed on",
          bool(checks) and not checks[0][1], str(checks[0][0])[:50])


def main():
    print("=" * 92)
    print("THE MEMORY PROOF IS BOUND TO ITS WORKLOAD — mutation battery")
    print("=" * 92)
    for fn in (t_positive_control, t_workload_mutations, t_identity_and_stack,
               t_controls_and_lifetime, t_purity):
        fn()
    n_ok = sum(1 for r in RESULTS if r)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
