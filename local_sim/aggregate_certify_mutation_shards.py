"""Validate and combine receipt-bound certification mutation shards.

Acceptance validates receipt coverage equivalent to the default monolithic run:
one positive control and every registered mutation exactly once.  Receipts do
not authenticate execution.  Raw workerq job metadata and logs are the authority
that the declared jobs executed the production certifier entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_certify_candidate as battery  # noqa: E402

ARTIFACT_VERSION = 1


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _registry_sha(registry):
    raw = json.dumps(registry, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _is_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _text_sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_json_int(value):
    return type(value) is int


def _is_json_bool(value):
    return type(value) is bool


def _raw_streams_valid(case):
    stdout, stderr = case.get("stdout"), case.get("stderr")
    return (isinstance(stdout, str) and isinstance(stderr, str)
            and case.get("stdout_sha256") == _text_sha(stdout)
            and case.get("stderr_sha256") == _text_sha(stderr))


def _distribution_manifest_errors(provenance, label):
    """Validate the deterministic byte manifests embedded in provenance."""
    errors = []
    for key in ("test_certify_candidate_sha256", "certify_candidate_sha256",
                "estimator_sha256"):
        if not isinstance(provenance, dict) or not _is_sha(provenance.get(key)):
            errors.append(f"{label}: {key} is absent or invalid")
    versions = provenance.get("distributions") if isinstance(provenance, dict) else None
    manifests = provenance.get("distribution_bytes") if isinstance(provenance, dict) else None
    if not isinstance(versions, dict) or not isinstance(manifests, dict):
        return [f"{label}: installed distribution byte provenance is absent"]
    for name in ("whestbench", "flopscope"):
        manifest = manifests.get(name)
        if not isinstance(manifest, dict):
            errors.append(f"{label}: {name} byte manifest is absent")
            continue
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            errors.append(f"{label}: {name} byte manifest has no files")
            continue
        valid_rows = all(
            isinstance(row, dict)
            and isinstance(row.get("path"), str) and bool(row["path"])
            and _is_json_int(row.get("size")) and row["size"] >= 0
            and _is_sha(row.get("sha256"))
            for row in files
        )
        paths = [row.get("path") for row in files if isinstance(row, dict)]
        if not valid_rows:
            errors.append(f"{label}: {name} byte manifest rows are malformed")
            continue
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            errors.append(f"{label}: {name} byte manifest rows are malformed")
            continue
        raw = json.dumps(files, sort_keys=True, separators=(",", ":"))
        if (not isinstance(manifest.get("version"), str)
                or not isinstance(versions.get(name), str)
                or manifest.get("version") != versions.get(name)):
            errors.append(f"{label}: {name} manifest version mismatch")
        if (not _is_json_int(manifest.get("file_count"))
                or manifest.get("file_count") != len(files)):
            errors.append(f"{label}: {name} manifest file count mismatch")
        if (not _is_json_int(manifest.get("total_bytes"))
                or manifest.get("total_bytes") != sum(row["size"] for row in files)):
            errors.append(f"{label}: {name} manifest byte count mismatch")
        if manifest.get("manifest_sha256") != hashlib.sha256(
                raw.encode("utf-8")).hexdigest():
            errors.append(f"{label}: {name} byte manifest digest mismatch")
    return errors


def _aggregate_payloads(payloads, *, current_provenance=None,
                        current_registry=None):
    """Pure validator used by the CLI and focused mutation tests."""
    provenance = (battery.receipt_provenance() if current_provenance is None
                  else current_provenance)
    registry = (battery.mutation_registry() if current_registry is None
                else current_registry)
    registry_sha = _registry_sha(registry)
    expected = {row["tag"]: row for row in registry}
    errors = []
    all_cases = []
    shard_rows = []
    workerq_job_ids = set()

    errors.extend(_distribution_manifest_errors(provenance, "current provenance"))

    if not payloads:
        errors.append("no shard receipts supplied")

    for index, item in enumerate(payloads):
        label = str(item.get("path", f"receipt[{index}]"))
        receipt = item.get("payload")
        file_sha = item.get("sha256", "")
        shard_row = {"path": label, "sha256": file_sha}
        shard_rows.append(shard_row)
        if not _is_sha(file_sha):
            errors.append(f"{label}: receipt file sha256 is absent or invalid")
        if not isinstance(receipt, dict):
            errors.append(f"{label}: receipt is not an object")
            continue
        workerq_job_id = receipt.get("workerq_job_id")
        shard_row["declared_workerq_job_id"] = workerq_job_id
        if (not isinstance(workerq_job_id, str)
                or not workerq_job_id.isdecimal()
                or int(workerq_job_id) <= 0):
            errors.append(f"{label}: declared workerq job id is absent or invalid")
        elif workerq_job_id in workerq_job_ids:
            errors.append(f"{label}: duplicate declared workerq job id {workerq_job_id}")
        else:
            workerq_job_ids.add(workerq_job_id)
        if receipt.get("artifact") != "certification_mutation_shard":
            errors.append(f"{label}: wrong artifact kind")
        artifact_version = receipt.get("artifact_version")
        if (not _is_json_int(artifact_version)
                or artifact_version != battery.RECEIPT_VERSION):
            errors.append(f"{label}: wrong artifact version")
        errors.extend(_distribution_manifest_errors(
            receipt.get("provenance"), label))
        if receipt.get("provenance") != provenance:
            errors.append(f"{label}: provenance does not match current sources/packages")
        if receipt.get("mutation_registry") != registry:
            errors.append(f"{label}: mutation registry does not match current source")
        if receipt.get("mutation_registry_sha256") != registry_sha:
            errors.append(f"{label}: mutation registry digest mismatch")
        checks = receipt.get("checks")
        if not isinstance(checks, list) or not checks:
            errors.append(f"{label}: check outcomes are absent")
            checks = []
        elif any(not isinstance(c, dict) or c.get("ok") is not True for c in checks):
            errors.append(f"{label}: at least one outer battery check failed")
        unit_n = receipt.get("unit_check_count")
        expected_unit_names = list(battery.UNIT_CHECK_NAMES)
        if (not _is_json_int(unit_n)
                or unit_n != len(expected_unit_names)):
            errors.append(f"{label}: unit check count does not match current registry")
        unit_prefix = checks[:len(expected_unit_names)]
        prefix_names = [c.get("name") for c in unit_prefix
                        if isinstance(c, dict)]
        if prefix_names != expected_unit_names:
            errors.append(f"{label}: unit-check prefix does not match current registry")
        if (len(unit_prefix) != len(expected_unit_names)
                or any(c.get("ok") is not True for c in unit_prefix
                       if isinstance(c, dict))):
            errors.append(f"{label}: unit-check prefix did not pass")
        repository_checks = []
        for required_check in (
            "R: estimator.py sha256 unchanged",
            "R: the mutant directory is gone",
            "R: receipt provenance unchanged across execution",
        ):
            matches = [c for c in checks if isinstance(c, dict)
                       and c.get("name") == required_check]
            if len(matches) != 1 or matches[0].get("ok") is not True:
                errors.append(f"{label}: required repository check absent: "
                              f"{required_check}")
            else:
                repository_checks.append(matches[0])
        if receipt.get("unit_checks_passed") is not True:
            errors.append(f"{label}: unit checks did not pass")
        if receipt.get("overall_pass") is not True:
            errors.append(f"{label}: shard verdict is not PASS")
        shard_row.update({
            "unit_check_count": unit_n,
            "unit_checks": unit_prefix,
            "repository_checks": repository_checks,
            "checks": checks,
            "overall_pass": receipt.get("overall_pass"),
        })

        selection = receipt.get("selection")
        cases = receipt.get("cases")
        if not isinstance(selection, dict) or not isinstance(cases, list):
            errors.append(f"{label}: selection or cases are malformed")
            continue
        selected = selection.get("mutation_tags")
        include_b = selection.get("include_positive")
        shard_row["selection"] = selection
        if (not isinstance(selected, list)
                or not _is_json_bool(include_b)
                or any(not isinstance(tag, str) for tag in selected)):
            errors.append(f"{label}: selection fields are malformed")
            continue
        if include_b is False and not selected:
            errors.append(f"{label}: empty mutation-only shard is forbidden")
        known_order = [row["tag"] for row in registry]
        canonical = [tag for tag in known_order if tag in set(selected)]
        if selected != canonical or len(selected) != len(set(selected)):
            errors.append(f"{label}: selected tags are unknown, duplicated or out of order")
        if any(not isinstance(case, dict) for case in cases):
            errors.append(f"{label}: every case must be an object")
            case_tags = []
        else:
            case_tags = [case.get("tag") for case in cases]
            if any(not isinstance(tag, str) for tag in case_tags):
                errors.append(f"{label}: every case tag must be a string")
                case_tags = []
        wanted = (["B"] if include_b else []) + selected
        if case_tags != wanted:
            errors.append(f"{label}: executed cases {case_tags!r} != selection {wanted!r}")
        all_cases.extend((label, case) for case in cases if isinstance(case, dict))

    by_tag = {}
    for label, case in all_cases:
        tag = case.get("tag")
        if not isinstance(tag, str):
            errors.append(f"{label}: executed case tag must be a string")
            continue
        if tag not in {"B", *expected}:
            errors.append(f"{label}: unknown executed case {tag!r}")
            continue
        if tag in by_tag:
            errors.append(f"{label}: duplicate executed case {tag}")
            continue
        by_tag[tag] = case

    required = ["B"] + [row["tag"] for row in registry]
    missing = [tag for tag in required if tag not in by_tag]
    if missing:
        errors.append("missing required cases: " + ", ".join(missing))

    positive = by_tag.get("B")
    if positive is not None:
        if positive.get("kind") != "positive":
            errors.append("B: wrong case kind")
        if not (positive.get("not_refused") is True
                and positive.get("no_fail_lines") is True
                and positive.get("skips_named") is True
                and _is_json_int(positive.get("returncode"))
                and positive.get("returncode") == 3):
            errors.append("B: the three positive-control checks are not all proved")
        if positive.get("fail_lines") != []:
            errors.append("B: positive control carries FAIL lines")
        if not _raw_streams_valid(positive):
            errors.append("B: raw stdout/stderr digests do not verify")
        elif "NOT CHECKED" not in positive["stdout"]:
            errors.append("B: raw stdout does not name skipped certification")
        for key in ("candidate_sha256", "stdout_sha256", "stderr_sha256"):
            if not _is_sha(positive.get(key)):
                errors.append(f"B: {key} is not a sha256")

    for tag, spec in expected.items():
        case = by_tag.get(tag)
        if case is None:
            continue
        for key, wanted in (("kind", "mutation"), ("title", spec["title"]),
                            ("section", spec["section"]),
                            ("expected_fail", spec["expected_fail"]),
                            ("extra_args", spec["extra_args"])):
            if case.get(key) != wanted:
                errors.append(f"{tag}: {key} does not match the current registry")
        if case.get("injection_applied") is not True:
            errors.append(f"{tag}: mutation injection did not apply")
        if not (case.get("battery_red") is True
                and _is_json_int(case.get("returncode"))
                and case.get("returncode") == 1):
            errors.append(f"{tag}: certifier did not return the required red verdict")
        fail_lines = case.get("fail_lines")
        if not _raw_streams_valid(case):
            errors.append(f"{tag}: raw stdout/stderr digests do not verify")
            parsed_fail_lines = []
        else:
            parsed_fail_lines = battery._failed_lines(case["stdout"])
            if parsed_fail_lines != fail_lines:
                errors.append(f"{tag}: stored FAIL lines differ from raw stdout")
        named = (isinstance(fail_lines, list)
                 and any(spec["expected_fail"] in str(line) for line in fail_lines)
                 and any(spec["expected_fail"] in str(line)
                         for line in parsed_fail_lines))
        if case.get("named_failure_matched") is not True or not named:
            errors.append(f"{tag}: registered named failure is absent")
        for key in ("candidate_sha256", "stdout_sha256", "stderr_sha256"):
            if not _is_sha(case.get(key)):
                errors.append(f"{tag}: {key} is not a sha256")

    accepted = not errors
    ordered_cases = [by_tag[tag] for tag in required if tag in by_tag]
    return {
        "artifact": "certification_mutation_aggregate",
        "artifact_version": ARTIFACT_VERSION,
        "aggregator_sha256": _sha(__file__),
        "status": "PASS" if accepted else "REFUSED",
        "accepted": accepted,
        "errors": errors,
        "provenance": provenance,
        "mutation_registry": registry,
        "mutation_registry_sha256": registry_sha,
        "required_cases": required,
        "cases": ordered_cases,
        "required_cases_covered": [case["tag"] for case in ordered_cases],
        "coverage_complete": len(ordered_cases) == len(required),
        "execution_authority": (
            "Raw workerq job metadata and logs; receipt aggregation validates "
            "declared job IDs, provenance, checks, and case coverage but does not "
            "authenticate execution."
        ),
        "shards": shard_rows,
    }


def aggregate_payloads(payloads, *, current_provenance=None,
                       current_registry=None):
    """Return a refusal receipt for malformed input instead of raising."""
    try:
        return _aggregate_payloads(
            payloads, current_provenance=current_provenance,
            current_registry=current_registry)
    except Exception as exc:
        return {
            "artifact": "certification_mutation_aggregate",
            "artifact_version": ARTIFACT_VERSION,
            "aggregator_sha256": _sha(__file__),
            "status": "REFUSED",
            "accepted": False,
            "errors": [f"malformed receipt payload: {type(exc).__name__}: {exc}"],
            "required_cases": ["B"] + [row["tag"] for row in
                                          battery.mutation_registry()],
            "cases": [],
            "required_cases_covered": [],
            "coverage_complete": False,
            "shards": [],
        }


def _write_exclusive(path, payload):
    out = os.path.abspath(path)
    parent = os.path.dirname(out)
    if not os.path.isdir(parent):
        raise ValueError(f"output parent directory does not exist: {parent}")
    with open(out, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("receipts", nargs="+")
    ap.add_argument("--json", required=True, metavar="PATH",
                    help="exclusive-create aggregate receipt")
    a = ap.parse_args()
    if os.path.exists(os.path.abspath(a.json)):
        ap.error("aggregate receipt already exists; evidence is never overwritten")

    payloads = []
    for path in a.receipts:
        full = os.path.abspath(path)
        try:
            with open(full, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, ValueError) as exc:
            payload = {"_read_error": str(exc)}
        payloads.append({
            "path": os.path.relpath(full, _ROOT).replace("\\", "/"),
            "sha256": _sha(full) if os.path.isfile(full) else "",
            "payload": payload,
        })

    result = aggregate_payloads(payloads)
    _write_exclusive(a.json, result)
    print(f"{result['status']}: {len(result['cases'])}/13 cases; "
          f"{len(result['errors'])} error(s)")
    for error in result["errors"]:
        print("  " + error)
    print("  aggregate -> " + os.path.abspath(a.json))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
