"""Mutation and replay tests for the workload-bound Linux memory consumer.

This is intentionally consumer-only and lightweight. It does not allocate a
live network or claim Linux enforcement; ``linux_memory_gate.py`` owns that job.
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

from certify_candidate import (MEMORY_CAP_MB, MEMORY_INSTRUMENT_POLICY,  # noqa: E402
                               MEMORY_REPORT_MIN_VERSION,
                               consume_memory_report)

RESULTS = []
SHA = "a" * 64
INSTRUMENT_SHA = "b" * 64
WIDTH, DEPTH, BUDGET = 1024, 16, 2 ** 41
STACK = {"whestbench": "0.16.1", "flopscope": "0.12.1"}
CAP_B = MEMORY_CAP_MB * 1024 * 1024
GROWTH_LIMIT = 64 * (1 << 20)


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {name:<68} {'ok ' if ok else 'FAILED'}  {detail}")
    return ok


def head(title):
    print(f"\n{title}\n{'-' * len(title)}")


def good_report():
    """A complete v4 report. Every mutation begins from this control."""
    return {
        "artifact": "linux_memory_gate",
        "artifact_version": MEMORY_REPORT_MIN_VERSION,
        "candidate": "estimator_candidate.py",
        "candidate_sha256": SHA,
        "instrument": "local_sim/linux_memory_gate.py",
        "instrument_sha256": INSTRUMENT_SHA,
        "instrument_policy": MEMORY_INSTRUMENT_POLICY,
        "limit_mb": MEMORY_CAP_MB,
        "width": WIDTH, "depth": DEPTH, "budget": BUDGET,
        "suite_n_mlps": 100,
        "seed": 90000000101, "weights_mode": "f64_readonly", "repeats": 3,
        "host_platform": "linux", "status": "PASS",
        "stack": dict(STACK, numpy="2.2.6", python="3.10.19",
                      source="importlib.metadata.version"),
        "negative_control": {"cap_applied": True, "survived": False},
        "positive_control": {"survived": True},
        "lifecycle_controls": {
            "growth_threshold_bytes": GROWTH_LIMIT,
            "stateless": {"completed": True, "growth_bytes": 1 << 20},
            "retaining": {"completed": True, "growth_bytes": 128 << 20},
        },
        "candidate_run": {
            "stage": "done", "ok": True, "cap_applied": True,
            "suite_n_mlps": 100,
            "cap_requested_bytes": CAP_B, "soft_limit_bytes": CAP_B,
            "predict_error": "", "setup_raised": None,
            "weights_mode": "f64_readonly", "weight_dtype": "float64",
            "weight_writeable": False,
            "whestbench_file": "/venv/site-packages/whestbench/__init__.py",
            "flopscope_file": "/venv/site-packages/flopscope/__init__.py",
            "vm_peak_bytes": 1_200 * (1 << 20),
            "repeat_runs": [
                {"i": i, "vm_size_settled_bytes": (900 + i) * (1 << 20),
                 "flops_used": 1.376e11 + i, "error": ""}
                for i in range(3)
            ],
            "repeat_error": "", "repeat_vm_size_growth_bytes": 2 << 20,
        },
    }


def consume(report, *, candidate_sha=SHA):
    return consume_memory_report(
        report, candidate_sha256=candidate_sha, width=WIDTH, depth=DEPTH,
        budget=BUDGET, cap_mb=MEMORY_CAP_MB, weights_mode="f64_readonly",
        stack=STACK, expected_instrument_sha256=INSTRUMENT_SHA)


def fixture_bundle():
    positive = good_report()
    incomplete = copy.deepcopy(positive)
    incomplete["artifact_version"] = 2
    for key in ("weights_mode", "stack", "instrument_sha256",
                "instrument_policy", "lifecycle_controls"):
        incomplete.pop(key, None)
    negative = copy.deepcopy(incomplete)
    negative["status"] = "FAIL"
    negative["candidate_run"]["predict_error"] = "MemoryError: deliberate"
    negative["candidate_run"]["cap_applied"] = False
    negative["negative_control"]["survived"] = True
    return {
        "positive": {"report": positive, "consumer": consume(positive)},
        "negative": {"report": negative, "consumer": consume(negative)},
        "incomplete": {"report": incomplete, "consumer": consume(incomplete)},
    }


def _mut(name, mutation, expected, needle=""):
    report = good_report()
    mutation(report)
    result = consume(report)
    rendered = json.dumps(result, sort_keys=True)
    check(name, result["verdict"] == expected and (not needle or needle in rendered),
          f"{result['verdict']}; {needle or result['incomplete'][:1]}")


def test_positive_negative_incomplete():
    head("1. positive, negative and incomplete producer fixtures")
    positive = consume(good_report())
    check("complete v4 positive control CERTIFIES",
          positive["verdict"] == "CERTIFIED" and positive["applicable"] is True,
          positive["verdict"])

    legacy = good_report()
    legacy["artifact_version"] = 2
    for key in ("weights_mode", "stack", "instrument_sha256",
                "instrument_policy", "lifecycle_controls"):
        legacy.pop(key, None)
    incomplete = consume(legacy)
    check("legacy missing v4 bindings is INCOMPLETE",
          incomplete["verdict"] == "INCOMPLETE"
          and incomplete["applicable"] is True,
          incomplete["incomplete"][-1][:72])

    negative = copy.deepcopy(legacy)
    negative["status"] = "FAIL"
    negative["candidate_run"]["predict_error"] = "MemoryError: deliberate"
    negative["candidate_run"]["cap_applied"] = False
    negative["negative_control"]["survived"] = True
    refused = consume(negative)
    check("applicable legacy explicit failures remain REFUSED",
          refused["verdict"] == "REFUSED"
          and len([c for c in refused["checks"] if not c[1]]) >= 3,
          f"{len(refused['checks'])} explicit negatives")


def test_binding_and_applicability():
    head("2. identity, workload, stack and instrument provenance")
    _mut("other candidate bytes are inapplicable, not a candidate failure",
         lambda r: r.__setitem__("candidate_sha256", "c" * 64),
         "INCOMPLETE", "different candidate bytes")
    _mut("missing candidate sha is incomplete path-only evidence",
         lambda r: r.pop("candidate_sha256"), "INCOMPLETE", "path-only")
    _mut("a small-shape report is inapplicable to the live workload",
         lambda r: r.__setitem__("width", 64), "INCOMPLETE", "certified shape")
    _mut("a Phase-1 budget report is inapplicable",
         lambda r: r.__setitem__("budget", 272_000_000_000),
         "INCOMPLETE", "certified budget")
    _mut("a looser cap report is inapplicable",
         lambda r: r.__setitem__("limit_mb", 65536),
         "INCOMPLETE", "full memory cap")
    _mut("a writable-f32 report is inapplicable",
         lambda r: r.__setitem__("weights_mode", "f32"),
         "INCOMPLETE", "deployment weight representation")
    _mut("a superseded WhestBench report is incomplete",
         lambda r: r["stack"].__setitem__("whestbench", "0.15.0"),
         "INCOMPLETE", "installed distribution whestbench")
    _mut("a non-current instrument hash cannot certify",
         lambda r: r.__setitem__("instrument_sha256", "c" * 64),
         "INCOMPLETE", "instrument sha256")
    _mut("a missing explicit instrument policy cannot certify",
         lambda r: r.pop("instrument_policy"),
         "INCOMPLETE", "instrument verification policy")


def test_applicable_failures_and_lifecycle():
    head("3. applicable explicit failures and lifecycle evidence")
    _mut("a mismatched child cap is REFUSED",
         lambda r: r["candidate_run"].__setitem__("soft_limit_bytes", 2 * CAP_B),
         "REFUSED", "cap read back exactly")
    _mut("a surviving negative control is REFUSED",
         lambda r: r["negative_control"].__setitem__("survived", True),
         "REFUSED", "negative control")
    _mut("a dead positive control is REFUSED",
         lambda r: r["positive_control"].__setitem__("survived", False),
         "REFUSED", "positive control")
    _mut("a candidate MemoryError is REFUSED",
         lambda r: r["candidate_run"].__setitem__("predict_error",
                                                   "MemoryError: deliberate"),
         "REFUSED", "MemoryError")
    _mut("one prediction cannot establish lifecycle stability",
         lambda r: r["candidate_run"].__setitem__("repeat_runs",
                                                   r["candidate_run"]["repeat_runs"][:1]),
         "REFUSED", "at least three")
    _mut("candidate settled address-space growth is REFUSED",
         lambda r: r["candidate_run"].__setitem__(
             "repeat_vm_size_growth_bytes", 256 << 20),
         "REFUSED", "settled address space")
    _mut("missing lifecycle controls is INCOMPLETE",
         lambda r: r.pop("lifecycle_controls"),
         "INCOMPLETE", "lifecycle controls are present")
    _mut("a stateless control that grows cannot certify the instrument",
         lambda r: r["lifecycle_controls"]["stateless"].__setitem__(
             "growth_bytes", 256 << 20),
         "INCOMPLETE", "stateless lifecycle positive control")
    _mut("a retaining control that does not grow cannot certify the instrument",
         lambda r: r["lifecycle_controls"]["retaining"].__setitem__(
             "growth_bytes", 1 << 20),
         "INCOMPLETE", "retaining lifecycle negative control")


def _file_sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def replay_stored_reports():
    paths = sorted(glob.glob(os.path.join(
        _ROOT, "experiments", "**", "linux_memory_gate*.json"), recursive=True))
    before = {os.path.relpath(p, _ROOT).replace("\\", "/"): _file_sha(p)
              for p in paths}
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        expected_sha = report.get("candidate_sha256") or "unverified-legacy-bytes"
        result = consume(report, candidate_sha=expected_sha)
        rel = os.path.relpath(path, _ROOT).replace("\\", "/")
        rows.append({
            "report": rel, "source_sha256": before[rel],
            "artifact_version": report.get("artifact_version"),
            "consumer_verdict": result["verdict"],
            "applicable": result["applicable"],
            "explicit_failed_checks": [n for n, ok, _ in result["checks"] if not ok],
            "incomplete_reasons": result["incomplete"],
        })
    after = {os.path.relpath(p, _ROOT).replace("\\", "/"): _file_sha(p)
             for p in paths}
    return {"artifact": "stage0_memory_report_replay", "artifact_version": 1,
            "consumer_min_version": MEMORY_REPORT_MIN_VERSION,
            "instrument_policy": MEMORY_INSTRUMENT_POLICY,
            "synthetic_fixtures": fixture_bundle(),
            "source_reports_unchanged": before == after,
            "n_reports": len(rows), "reports": rows}


def test_replay():
    head("4. non-rewriting replay of every stored Linux memory report")
    replay = replay_stored_reports()
    check("the expected seven historical reports were replayed",
          replay["n_reports"] == 7, str(replay["n_reports"]))
    check("every historical source artifact remained byte-identical",
          replay["source_reports_unchanged"], "sha256 before == after")
    verdicts = [row["consumer_verdict"] for row in replay["reports"]]
    check("all seven legacy reports are incomplete under the v4 consumer",
          verdicts == ["INCOMPLETE"] * 7, repr(verdicts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    for fn in (test_positive_negative_incomplete, test_binding_and_applicability,
               test_applicable_failures_and_lifecycle, test_replay):
        fn()
    replay = replay_stored_reports()
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(replay, handle, indent=2)
            handle.write("\n")
    n_ok = sum(RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
