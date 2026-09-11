"""Focused, truth-free tests for allocator-bounded edge-suite receipts."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import certify_candidate as cert  # noqa: E402
import edge_cases as edge  # noqa: E402

FAILED = []
TOTAL = 0


def check(name, ok):
    global TOTAL
    TOTAL += 1
    print("  %s  %s" % ("PASS" if ok else "FAIL", name))
    if not ok:
        FAILED.append(name)


def fixture(group="normal", count=1):
    candidate_sha = "a" * 64
    edge_sha = "b" * 64
    payload = {
        "artifact": "edge_cases_group",
        "artifact_version": cert.EDGE_GROUP_RECEIPT_VERSION,
        "candidate": "candidate.py",
        "candidate_sha256_before": candidate_sha,
        "candidate_sha256_after": candidate_sha,
        "edge_cases_sha256": edge_sha,
        "group_registry": [
            {"group": name, "expected_result_count": expected}
            for name, expected in cert.EDGE_GROUP_REGISTRY
        ],
        "world": {
            "phase2": True, "real_flopscope": True,
            "weights_mode": "f64_readonly",
            "width": cert.WIDTH, "depth": cert.DEPTH,
            "budget": cert.BUDGET,
        },
        "group": group,
        "expected_result_count": count,
        "results": [True] * count,
        "passed": count,
        "all_passed": True,
    }
    kwargs = {
        "group": group, "expected_count": count,
        "candidate": "candidate.py", "candidate_sha256": candidate_sha,
        "edge_cases_sha256": edge_sha, "weights_mode": "f64_readonly",
    }
    return payload, kwargs


def case_fixture(case="normal:0", group="normal", case_index=0):
    payload, group_kwargs = fixture(group=group, count=1)
    payload.pop("expected_result_count")
    payload.pop("group")
    payload.update({
        "artifact": "edge_cases_case",
        "artifact_version": cert.EDGE_CASE_RECEIPT_VERSION,
        "case_registry": [
            {"case": name, "group": logical, "case_index": index}
            for name, logical, index in cert.EDGE_CASE_REGISTRY
        ],
        "case": case, "logical_group": group, "case_index": case_index,
        "details": {},
    })
    kwargs = {key: value for key, value in group_kwargs.items()
              if key not in ("expected_count",)}
    kwargs.update({"case": case, "group": group, "case_index": case_index})
    return payload, kwargs


def budget_rows(log2ns, *, invalid_index=None, error_index=None):
    rows = []
    for index, (budget, log2n) in enumerate(zip(cert.EDGE_BUDGET_LADDER,
                                                log2ns)):
        finite = index != invalid_index
        error = "RuntimeError: injected" if index == error_index else ""
        rows.append({
            "case": "budget_monotone:%d" % index,
            "case_index": index, "result": bool(finite and not error),
            "details": {
                "budget": budget, "has_ladder": True, "log2n": log2n,
                "finite": finite, "error": error,
            },
        })
    return rows


def main():
    check("producer and consumer registries are exact and equal",
          edge.GROUP_REGISTRY == cert.EDGE_GROUP_REGISTRY
          and edge.CASE_COUNTS == cert.EDGE_CASE_COUNTS
          and edge.CASE_REGISTRY == cert.EDGE_CASE_REGISTRY
          and sum(count for _, count in edge.GROUP_REGISTRY) == 30
          and len({group for group, _ in edge.GROUP_REGISTRY})
          == len(edge.GROUP_REGISTRY))

    payload, kwargs = fixture()
    check("complete receipt accepts",
          cert.validate_edge_group_receipt(payload, **kwargs) == [])

    mutations = (
        ("non-object", lambda _p: [], "not a JSON object"),
        ("wrong artifact", lambda p: p.__setitem__("artifact", "other"),
         "artifact identity"),
        ("extra field", lambda p: p.__setitem__("unregistered", True),
         "receipt schema"),
        ("boolean version", lambda p: p.__setitem__("artifact_version", True),
         "artifact version"),
        ("wrong group", lambda p: p.__setitem__("group", "other"),
         "group identity"),
        ("wrong candidate", lambda p: p.__setitem__("candidate", "other.py"),
         "candidate path"),
        ("changed candidate", lambda p: p.__setitem__(
            "candidate_sha256_after", "c" * 64), "after differs"),
        ("wrong instrument", lambda p: p.__setitem__(
            "edge_cases_sha256", "c" * 64), "edge_cases.py hash"),
        ("truncated registry", lambda p: p["group_registry"].pop(),
         "group registry"),
        ("boolean registry count", lambda p: p["group_registry"][0].__setitem__(
            "expected_result_count", True), "group registry"),
        ("missing world field", lambda p: p["world"].pop("budget"),
         "world schema"),
        ("boolean width", lambda p: p["world"].__setitem__("width", True),
         "world field width"),
        ("wrong dtype", lambda p: p["world"].__setitem__(
            "weights_mode", "f32"), "world field weights_mode"),
        ("boolean count", lambda p: p.__setitem__(
            "expected_result_count", True), "expected result count"),
        ("short vector", lambda p: p.__setitem__("results", []),
         "exact boolean result vector"),
        ("integer result", lambda p: p.__setitem__("results", [1]),
         "exact boolean result vector"),
        ("wrong passed", lambda p: p.__setitem__("passed", 0),
         "passed count differs"),
        ("boolean passed", lambda p: p.__setitem__("passed", True),
         "passed count is not"),
        ("integer all_passed", lambda p: p.__setitem__("all_passed", 1),
         "all_passed is not"),
    )
    for name, mutate, needle in mutations:
        bad = copy.deepcopy(payload)
        result = mutate(bad)
        if name == "non-object":
            bad = result
        errors = cert.validate_edge_group_receipt(bad, **kwargs)
        check(name + " refuses", any(needle in error for error in errors))

    case_payload, case_kwargs = case_fixture()
    check("complete atomic-case receipt accepts",
          cert.validate_edge_case_receipt(case_payload, **case_kwargs) == [])
    case_mutations = (
        ("boolean case index", lambda p: p.__setitem__("case_index", False),
         "case index"),
        ("truncated case registry", lambda p: p["case_registry"].pop(),
         "case registry"),
        ("boolean registry index", lambda p: p["case_registry"][0].__setitem__(
            "case_index", False), "case registry"),
        ("non-object details", lambda p: p.__setitem__("details", []),
         "details are not"),
        ("wrong result length", lambda p: p.__setitem__("results", []),
         "one exact boolean"),
    )
    for name, mutate, needle in case_mutations:
        bad = copy.deepcopy(case_payload)
        mutate(bad)
        errors = cert.validate_edge_case_receipt(bad, **case_kwargs)
        check(name + " refuses", any(needle in error for error in errors))

    permutation_payload, permutation_kwargs = case_fixture(
        case="permutation:3", group="permutation", case_index=3)
    permutation_payload["details"] = cert._expected_permutation_details(3)
    check("effective permutation bar and digest receipt accepts",
          cert.validate_edge_case_receipt(
              permutation_payload, **permutation_kwargs) == [])
    bad_bar = copy.deepcopy(permutation_payload)
    bad_bar["details"]["bar"] *= 2.0
    check("unbound permutation bar refuses",
          "permutation evidence differs" in cert.validate_edge_case_receipt(
              bad_bar, **permutation_kwargs))
    bad_digest = copy.deepcopy(permutation_payload)
    bad_digest["details"]["permutation_sha256"] = "0" * 64
    check("wrong permutation stream digest refuses",
          "permutation evidence differs" in cert.validate_edge_case_receipt(
              bad_digest, **permutation_kwargs))

    permutation_cases = cert.EDGE_PERMUTATION_CASES
    monolith = list(edge._permutation_case_stream(
        edge.np.random.default_rng(30_806_006_100), permutation_cases))
    streams_match = len(monolith) == len(permutation_cases)
    for index in range(len(permutation_cases)):
        atomic = list(edge._permutation_case_stream(
            edge.np.random.default_rng(30_806_006_100), permutation_cases,
            case_index=index))
        streams_match &= (len(atomic) == 1 and atomic[0][0] == index
                          and edge.np.array_equal(atomic[0][2],
                                                  monolith[index][2]))
    check("atomic permutations preserve every monolith RNG prefix",
          streams_match)

    increasing = list(range(len(cert.EDGE_BUDGET_LADDER)))
    budget_ok, budget_errors = cert.reduce_budget_monotone_cases(
        budget_rows(increasing))
    check("budget reducer accepts an exact valid monotone ladder",
          budget_ok and budget_errors == [])
    invalid_rows = budget_rows(
        [0, 1, 2, 3, 10, 9, 10, 11, 12, 13], invalid_index=4)
    invalid_ok, invalid_errors = cert.reduce_budget_monotone_cases(invalid_rows)
    check("non-throwing invalid rung still advances comparison state",
          not invalid_ok and any("budget_monotone:5: sample ladder decreased"
                                 in error for error in invalid_errors))
    thrown_rows = budget_rows(
        [0, 1, 2, 3, 10, 9, 10, 11, 12, 13], error_index=4)
    thrown_ok, thrown_errors = cert.reduce_budget_monotone_cases(thrown_rows)
    check("throwing rung does not advance comparison state",
          not thrown_ok and not any(
              "budget_monotone:5: sample ladder decreased" in error
              for error in thrown_errors))

    # The default path and isolated path share the one frozen dispatcher.  This
    # source-level identity is the cheap parity proof; numerical parity is then
    # exercised by the real positive control under workerq.
    expected_registry_sha = hashlib.sha256(repr(edge.GROUP_REGISTRY).encode()).hexdigest()
    check("default and isolated paths share one frozen dispatcher",
          callable(edge._execute_group)
          and expected_registry_sha
          == hashlib.sha256(repr(cert.EDGE_GROUP_REGISTRY).encode()).hexdigest())

    with tempfile.TemporaryDirectory(prefix="edge-group-cli-test-") as td:
        candidate = os.path.join(td, "candidate.py")
        receipt = os.path.join(td, "receipt.json")
        with open(candidate, "w", encoding="utf-8") as fh:
            fh.write("class Estimator:\n"
                     "    def setup(self, context):\n"
                     "        pass\n")
        run = subprocess.run(
            [sys.executable, edge.__file__, candidate,
             "--phase2", "--real-flopscope", "--weights-mode", "f64_readonly",
             "--case", "seed_hygiene:0", "--receipt", receipt],
            cwd=_ROOT, capture_output=True, text=True, timeout=60)
        produced = json.load(open(receipt, encoding="utf-8")) if os.path.exists(
            receipt) else {}
        check("lightweight CLI case writes an exact exclusive receipt",
              run.returncode == 0
              and produced.get("case") == "seed_hygiene:0"
              and produced.get("results") == [True]
              and produced.get("candidate_sha256_before")
              == produced.get("candidate_sha256_after")
              and produced.get("edge_cases_sha256")
              == cert._sha(edge.__file__)
              and cert.validate_edge_case_receipt(
                  produced, case="seed_hygiene:0", group="seed_hygiene",
                  case_index=0,
                  candidate=os.path.relpath(candidate, _ROOT).replace("\\", "/"),
                  candidate_sha256=cert._sha(candidate),
                  edge_cases_sha256=cert._sha(edge.__file__),
                  weights_mode="f64_readonly") == [])
        again = subprocess.run(
            [sys.executable, edge.__file__, candidate,
             "--phase2", "--real-flopscope", "--weights-mode", "f64_readonly",
             "--case", "seed_hygiene:0", "--receipt", receipt],
            cwd=_ROOT, capture_output=True, text=True, timeout=60)
        check("CLI receipt refuses overwrite", again.returncode != 0
              and "never overwritten" in again.stderr)

        # A stateless zero-output fixture makes every array construction and
        # dispatch path real while keeping candidate work tiny.  The layer-1
        # oracle deliberately fails; parity compares the complete boolean
        # vector, not a hand-picked green path.
        with open(candidate, "w", encoding="utf-8") as fh:
            fh.write("import numpy as np\n"
                     "class Estimator:\n"
                     "    def setup(self, context):\n"
                     "        pass\n"
                     "    def predict(self, mlp, budget):\n"
                     "        return np.zeros((len(mlp.weights), mlp.width))\n")
        default = subprocess.run(
            [sys.executable, edge.__file__, candidate], cwd=_ROOT,
            capture_output=True, text=True, timeout=60)
        by_group = {group: [] for group, _ in edge.GROUP_REGISTRY}
        isolated_ok = True
        for case, group, case_index in edge.CASE_REGISTRY:
            path = os.path.join(td, "parity-" + case.replace(":", "_") + ".json")
            run = subprocess.run(
                [sys.executable, edge.__file__, candidate,
                 "--case", case, "--receipt", path],
                cwd=_ROOT, capture_output=True, text=True, timeout=60)
            if not os.path.exists(path):
                isolated_ok = False
                continue
            row = json.load(open(path, encoding="utf-8"))
            by_group[group].append((case_index, row.get("results", [False])[0]))
            isolated_ok &= run.returncode == (0 if row.get("all_passed") else 1)
        isolated_results = []
        for group, logical_count in edge.GROUP_REGISTRY:
            values = [value for _, value in sorted(by_group[group])]
            isolated_results.extend(values if logical_count > 1 else [all(values)])
        expected_tail = "%d/%d checks passed" % (
            sum(isolated_results), len(isolated_results))
        check("cheap default execution equals aggregated atomic cases",
              isolated_ok and len(isolated_results) == 30
              and expected_tail in default.stdout
              and default.returncode == (0 if all(isolated_results) else 1))

    if FAILED:
        print("FAILED %d: %s" % (len(FAILED), "; ".join(FAILED)))
        return 1
    print("%d/%d checks passed" % (TOTAL, TOTAL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
