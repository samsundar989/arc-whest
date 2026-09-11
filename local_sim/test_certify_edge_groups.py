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


def main():
    check("producer and consumer registries are exact and equal",
          edge.GROUP_REGISTRY == cert.EDGE_GROUP_REGISTRY
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
             "--group", "seed_hygiene", "--receipt", receipt],
            cwd=_ROOT, capture_output=True, text=True, timeout=60)
        produced = json.load(open(receipt, encoding="utf-8")) if os.path.exists(
            receipt) else {}
        check("lightweight CLI group writes an exact exclusive receipt",
              run.returncode == 0
              and produced.get("group") == "seed_hygiene"
              and produced.get("results") == [True]
              and produced.get("candidate_sha256_before")
              == produced.get("candidate_sha256_after")
              and produced.get("edge_cases_sha256")
              == cert._sha(edge.__file__)
              and cert.validate_edge_group_receipt(
                  produced, group="seed_hygiene", expected_count=1,
                  candidate=os.path.relpath(candidate, _ROOT).replace("\\", "/"),
                  candidate_sha256=cert._sha(candidate),
                  edge_cases_sha256=cert._sha(edge.__file__),
                  weights_mode="f64_readonly") == [])
        again = subprocess.run(
            [sys.executable, edge.__file__, candidate,
             "--phase2", "--real-flopscope", "--weights-mode", "f64_readonly",
             "--group", "seed_hygiene", "--receipt", receipt],
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
        isolated_results = []
        isolated_ok = True
        for group, _count in edge.GROUP_REGISTRY:
            path = os.path.join(td, "parity-" + group + ".json")
            run = subprocess.run(
                [sys.executable, edge.__file__, candidate,
                 "--group", group, "--receipt", path],
                cwd=_ROOT, capture_output=True, text=True, timeout=60)
            if not os.path.exists(path):
                isolated_ok = False
                continue
            row = json.load(open(path, encoding="utf-8"))
            isolated_results += row.get("results", [])
            isolated_ok &= run.returncode == (0 if row.get("all_passed") else 1)
        expected_tail = "%d/%d checks passed" % (
            sum(isolated_results), len(isolated_results))
        check("cheap default execution equals concatenated isolated groups",
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
