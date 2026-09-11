"""Focused, truth-free tests for certification mutation shard receipts."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile

import aggregate_certify_mutation_shards as agg
import test_certify_candidate as battery

FAILED = []
TOTAL = 0


def check(name, ok):
    global TOTAL
    TOTAL += 1
    print(f"  {'ok' if ok else 'FAIL'}  {name}")
    if not ok:
        FAILED.append(name)


def _case(spec):
    expected = spec["expected_fail"]
    stdout, stderr = "FAIL  " + expected + "\n", ""
    return {
        "tag": spec["tag"], "kind": "mutation", "title": spec["title"],
        "section": spec["section"], "expected_fail": expected,
        "extra_args": spec["extra_args"], "injection_applied": True,
        "candidate_sha256": "a" * 64, "returncode": 1,
        "battery_red": True, "named_failure_matched": True,
        "fail_lines": ["FAIL  " + expected], "stdout": stdout,
        "stderr": stderr, "stdout_sha256": agg._text_sha(stdout),
        "stderr_sha256": agg._text_sha(stderr),
    }


def _positive():
    stdout, stderr = "NOT CHECKED\n", ""
    return {
        "tag": "B", "kind": "positive", "candidate_sha256": "d" * 64,
        "returncode": 3, "fail_lines": [], "stdout": stdout,
        "stderr": stderr, "stdout_sha256": agg._text_sha(stdout),
        "stderr_sha256": agg._text_sha(stderr), "not_refused": True,
        "no_fail_lines": True, "skips_named": True,
    }


def _receipt(provenance, registry, tags, *, include_positive=False,
             workerq_job_id="1001"):
    by_tag = {row["tag"]: row for row in registry}
    cases = ([_positive()] if include_positive else []) + [
        _case(by_tag[tag]) for tag in tags
    ]
    return {
        "artifact": "certification_mutation_shard",
        "artifact_version": battery.RECEIPT_VERSION,
        "provenance": provenance,
        "mutation_registry": registry,
        "mutation_registry_sha256": agg._registry_sha(registry),
        "selection": {"include_positive": include_positive,
                      "mutation_tags": tags},
        "cases": cases,
        "unit_check_count": len(battery.UNIT_CHECK_NAMES),
        "unit_checks_passed": True,
        "checks": [
            *[{"name": name, "ok": True}
              for name in battery.UNIT_CHECK_NAMES],
            {"name": "R: estimator.py sha256 unchanged", "ok": True},
            {"name": "R: the mutant directory is gone", "ok": True},
            {"name": "R: receipt provenance unchanged across execution",
             "ok": True},
        ],
        "overall_pass": True,
        "workerq_job_id": workerq_job_id,
    }


def _manifest(version, package, digit):
    files = [{"path": f"{package}/__init__.py", "size": 1,
              "sha256": digit * 64}]
    raw = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "version": version,
        "file_count": len(files),
        "total_bytes": sum(row["size"] for row in files),
        "manifest_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "files": files,
    }


def main():
    registry = battery.mutation_registry()
    tags = [row["tag"] for row in registry]
    live_provenance = battery.receipt_provenance()
    check("live installed package byte manifests validate",
          not agg._distribution_manifest_errors(live_provenance, "live")
          and all(live_provenance["distribution_bytes"][name]["file_count"] > 0
                  for name in ("whestbench", "flopscope")))
    provenance = {
        "test_certify_candidate_sha256": "1" * 64,
        "certify_candidate_sha256": "2" * 64,
        "estimator_sha256": "3" * 64,
        "distributions": {"whestbench": "0.16.1", "flopscope": "0.12.1"},
        "distribution_bytes": {
            "whestbench": _manifest("0.16.1", "whestbench", "4"),
            "flopscope": _manifest("0.12.1", "flopscope", "5"),
        },
    }

    sel = battery.shard_selection()
    check("default selection remains B plus every mutation",
          sel == {"include_positive": True, "mutation_tags": tags})
    sel = battery.shard_selection("M3,M1", skip_positive=True)
    check("mutation shard is canonical and omits B",
          sel == {"include_positive": False,
                  "mutation_tags": ["M1", "M3"]})
    sel = battery.shard_selection(positive_only=True)
    check("positive-only selects B and no mutation",
          sel == {"include_positive": True, "mutation_tags": []})
    for label, call in (
        ("unknown tag refuses", lambda: battery.shard_selection("NOPE")),
        ("duplicate tag refuses", lambda: battery.shard_selection("M1,M1")),
        ("empty skip-positive refuses",
         lambda: battery.shard_selection("", skip_positive=True)),
        ("conflicting positive flags refuse",
         lambda: battery.shard_selection("M1", skip_positive=True,
                                         positive_only=True)),
    ):
        try:
            call()
            refused = False
        except ValueError:
            refused = True
        check(label, refused)

    for value, env, wanted in (
        ("726", "726", True), ("726", "", True), ("", "", False),
        ("synthetic", "", False), ("726", "727", False),
    ):
        try:
            battery.validate_workerq_job_id(value, env)
            valid = True
        except ValueError:
            valid = False
        check(f"workerq id {value!r} against {env!r} validation", valid is wanted)

    receipts = [
        _receipt(provenance, registry, [], include_positive=True,
                 workerq_job_id="1001"),
        _receipt(provenance, registry, tags[:4], workerq_job_id="1002"),
        _receipt(provenance, registry, tags[4:8], workerq_job_id="1003"),
        _receipt(provenance, registry, tags[8:], workerq_job_id="1004"),
    ]
    wrapped = [{"path": f"r{i}.json", "sha256": "9" * 64,
                "payload": row} for i, row in enumerate(receipts)]
    good = agg.aggregate_payloads(
        wrapped, current_provenance=provenance, current_registry=registry)
    check("complete aggregate accepts", good["accepted"] is True)
    check("complete aggregate validates all thirteen required cases",
          good["coverage_complete"] is True
          and good["required_cases_covered"] == ["B"] + tags
          and [c["tag"] for c in good["cases"]] == ["B"] + tags)
    check("aggregate retains shard unit and repository evidence",
          all(row["unit_check_count"] == len(battery.UNIT_CHECK_NAMES)
              and [c["name"] for c in row["unit_checks"]]
              == list(battery.UNIT_CHECK_NAMES)
              and len(row["repository_checks"]) == 3
              for row in good["shards"]))

    adversarial = []
    bad = copy.deepcopy(wrapped)
    bad[0]["payload"]["artifact_version"] = True
    adversarial.append(("boolean artifact version refuses", bad,
                        "wrong artifact version"))

    bad = copy.deepcopy(wrapped)
    bad[0]["payload"]["provenance"]["distribution_bytes"]["whestbench"][
        "files"][0]["size"] = True
    adversarial.append(("boolean package file size refuses", bad,
                        "manifest rows are malformed"))

    bad = copy.deepcopy(wrapped)
    bad[1]["payload"]["cases"][0]["returncode"] = True
    adversarial.append(("boolean mutation return code refuses", bad,
                        "required red verdict"))

    bad = copy.deepcopy(wrapped)
    bad[1]["payload"]["selection"]["mutation_tags"][0] = {"tag": tags[0]}
    adversarial.append(("object selected tag refuses without raising", bad,
                        "selection fields are malformed"))

    bad = copy.deepcopy(wrapped)
    bad[1]["payload"]["cases"][0]["tag"] = [tags[0]]
    adversarial.append(("array case tag refuses without raising", bad,
                        "case tag must be a string"))

    for name, bad, expected_error in adversarial:
        out = agg.aggregate_payloads(
            bad, current_provenance=provenance, current_registry=registry)
        check(name, out["accepted"] is False
              and out["status"] == "REFUSED"
              and any(expected_error in error for error in out["errors"]))

    out = agg.aggregate_payloads(
        [[]], current_provenance=provenance, current_registry=registry)
    check("malformed top-level payload returns REFUSED instead of raising",
          out["accepted"] is False and out["status"] == "REFUSED"
          and any("malformed receipt payload" in error
                  for error in out["errors"]))

    bad = copy.deepcopy(wrapped)
    bad[-1]["payload"]["cases"].pop()
    bad[-1]["payload"]["selection"]["mutation_tags"].pop()
    out = agg.aggregate_payloads(
        bad, current_provenance=provenance, current_registry=registry)
    check("missing mutation refuses", not out["accepted"]
          and any("missing required cases" in e for e in out["errors"]))

    bad = copy.deepcopy(wrapped)
    bad[-1]["payload"]["cases"].append(copy.deepcopy(
        bad[1]["payload"]["cases"][0]))
    bad[-1]["payload"]["selection"]["mutation_tags"].append(tags[0])
    out = agg.aggregate_payloads(
        bad, current_provenance=provenance, current_registry=registry)
    check("duplicate mutation refuses", not out["accepted"]
          and any("duplicate executed case" in e for e in out["errors"]))

    bad = copy.deepcopy(wrapped)
    case = bad[1]["payload"]["cases"][0]
    case["named_failure_matched"] = False
    case["fail_lines"] = ["FAIL  a different check"]
    case["stdout"] = "FAIL  a different check\n"
    case["stdout_sha256"] = agg._text_sha(case["stdout"])
    out = agg.aggregate_payloads(
        bad, current_provenance=provenance, current_registry=registry)
    check("wrong named failure refuses", not out["accepted"]
          and any("named failure" in e for e in out["errors"]))

    bad = copy.deepcopy(wrapped)
    bad[2]["payload"]["provenance"]["estimator_sha256"] = "0" * 64
    out = agg.aggregate_payloads(
        bad, current_provenance=provenance, current_registry=registry)
    check("mixed provenance refuses", not out["accepted"]
          and any("provenance" in e for e in out["errors"]))

    bad = copy.deepcopy(wrapped)
    manifest = bad[2]["payload"]["provenance"]["distribution_bytes"]["flopscope"]
    manifest["files"][0]["sha256"] = "6" * 64
    raw = json.dumps(manifest["files"], sort_keys=True, separators=(",", ":"))
    manifest["manifest_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    out = agg.aggregate_payloads(
        bad, current_provenance=provenance, current_registry=registry)
    check("changed installed package bytes refuse", not out["accepted"]
          and any("provenance" in e for e in out["errors"]))

    bad = copy.deepcopy(wrapped)
    bad[2]["payload"]["provenance"]["distribution_bytes"]["flopscope"][
        "manifest_sha256"] = "0" * 64
    out = agg.aggregate_payloads(
        bad, current_provenance=provenance, current_registry=registry)
    check("malformed package byte manifest refuses", not out["accepted"]
          and any("byte manifest digest" in e for e in out["errors"]))

    bad = copy.deepcopy(wrapped)
    bad[1]["payload"]["workerq_job_id"] = ""
    out = agg.aggregate_payloads(
        bad, current_provenance=provenance, current_registry=registry)
    check("empty declared workerq job id refuses", not out["accepted"]
          and any("job id is absent" in e for e in out["errors"]))

    bad = copy.deepcopy(wrapped)
    bad[1]["payload"]["workerq_job_id"] = "1001"
    out = agg.aggregate_payloads(
        bad, current_provenance=provenance, current_registry=registry)
    check("duplicate declared workerq job id refuses", not out["accepted"]
          and any("duplicate declared" in e for e in out["errors"]))

    bad = copy.deepcopy(wrapped)
    bad.append({"path": "empty.json", "sha256": "8" * 64,
                "payload": _receipt(provenance, registry, [],
                                    workerq_job_id="1005")})
    out = agg.aggregate_payloads(
        bad, current_provenance=provenance, current_registry=registry)
    check("empty mutation-only shard refuses", not out["accepted"]
          and any("empty mutation-only" in e for e in out["errors"]))

    bad = copy.deepcopy(wrapped)
    bad[1]["payload"]["cases"][0] = "not an object"
    out = agg.aggregate_payloads(
        bad, current_provenance=provenance, current_registry=registry)
    check("non-object case refuses", not out["accepted"]
          and any("every case must be an object" in e for e in out["errors"]))

    bad = copy.deepcopy(wrapped)
    bad[1]["payload"]["unit_check_count"] -= 1
    out = agg.aggregate_payloads(
        bad, current_provenance=provenance, current_registry=registry)
    check("wrong unit check count refuses", not out["accepted"]
          and any("unit check count" in e for e in out["errors"]))

    bad = copy.deepcopy(wrapped)
    bad[1]["payload"]["checks"][0]["name"] = "renamed unit check"
    out = agg.aggregate_payloads(
        bad, current_provenance=provenance, current_registry=registry)
    check("wrong unit-check prefix refuses", not out["accepted"]
          and any("unit-check prefix" in e for e in out["errors"]))

    bad = copy.deepcopy(wrapped)
    bad[0]["payload"]["overall_pass"] = False
    bad[0]["payload"]["checks"][0]["ok"] = False
    out = agg.aggregate_payloads(
        bad, current_provenance=provenance, current_registry=registry)
    check("failed shard refuses", not out["accepted"]
          and any("verdict" in e for e in out["errors"]))

    with tempfile.TemporaryDirectory() as tmp:
        p1 = os.path.join(tmp, "shard.json")
        battery._write_json_exclusive(p1, {"ok": True})
        try:
            battery._write_json_exclusive(p1, {"ok": False})
            shard_refused = False
        except FileExistsError:
            shard_refused = True
        check("shard receipt is exclusive-create", shard_refused
              and json.load(open(p1, encoding="utf-8")) == {"ok": True})

        p2 = os.path.join(tmp, "aggregate.json")
        agg._write_exclusive(p2, {"accepted": True})
        try:
            agg._write_exclusive(p2, {"accepted": False})
            aggregate_refused = False
        except FileExistsError:
            aggregate_refused = True
        check("aggregate receipt is exclusive-create", aggregate_refused
              and json.load(open(p2, encoding="utf-8")) == {"accepted": True})

    print(f"\n{TOTAL - len(FAILED)}/{TOTAL} checks passed")
    if FAILED:
        print("FAILED: " + "; ".join(FAILED))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
