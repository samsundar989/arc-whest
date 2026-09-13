"""Tiny adversarial check of the actual Linux memory child warning filters."""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import warnings


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "local_sim" / "linux_memory_gate.py"


def child_filter_statements(source: str) -> list[ast.stmt]:
    outer = ast.parse(source)
    assignments = [
        node for node in outer.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "CHILD_PREDICT"
                for target in node.targets)
    ]
    assert len(assignments) == 1, "expected one CHILD_PREDICT definition"
    child = ast.parse(ast.literal_eval(assignments[0].value))
    statements = [
        node for node in child.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "warnings"
        and node.value.func.attr == "filterwarnings"
    ]
    assert len(statements) == 2, "expected only the symmetry and dtype filters"
    return statements


class FlopscopeWarning(Warning):
    pass


class SymmetryLossWarning(FlopscopeWarning):
    pass


MESSAGE = (
    "stats.norm.{op} promoted its float32 input to float64 to match "
    "scipy.stats. If float64 output was not intended, cast the result back "
    "with result.astype(np.float32) before downstream operations; float64 "
    "operations are billed at twice the float32 dtype rate."
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args()
    source_bytes = GATE.read_bytes()
    statements = child_filter_statements(source_bytes.decode("utf-8"))
    fragment = compile(ast.fix_missing_locations(ast.Module(body=statements,
                                                           type_ignores=[])),
                       str(GATE), "exec")
    flops = SimpleNamespace(FlopscopeWarning=FlopscopeWarning,
                            SymmetryLossWarning=SymmetryLossWarning)
    cases = [
        ("exact_cdf_typed", MESSAGE.format(op="cdf"), FlopscopeWarning,
         False, ""),
        ("exact_pdf_typed", MESSAGE.format(op="pdf"), FlopscopeWarning,
         False, ""),
        ("near_match_typed", MESSAGE.format(op="cdf") + " extra",
         FlopscopeWarning, True, "extra"),
        ("case_changed_typed", "Stats" + MESSAGE.format(op="cdf")[5:],
         FlopscopeWarning, True, "Stats.norm.cdf"),
        ("unrelated_typed", "unexpected flopscope numerical warning",
         FlopscopeWarning, True, "unexpected flopscope numerical warning"),
        ("same_message_wrong_type", MESSAGE.format(op="pdf"), UserWarning,
         True, "stats.norm.pdf"),
        ("symmetry_policy", "known symmetry loss", SymmetryLossWarning,
         False, ""),
    ]
    rows = []
    for name, message, category, visible, marker in cases:
        stderr = io.StringIO()
        with warnings.catch_warnings(), redirect_stderr(stderr):
            warnings.simplefilter("always")
            exec(fragment, {"warnings": warnings, "flops": flops})
            warnings.warn(message, category)
        actual = stderr.getvalue()
        passed = (bool(actual) == visible and (not visible or marker in actual))
        rows.append({"case": name, "passed": passed, "stderr": actual})

    stderr = io.StringIO()
    with warnings.catch_warnings(), redirect_stderr(stderr):
        warnings.simplefilter("always")
        exec(fragment, {"warnings": warnings, "flops": flops})
        print("arbitrary child stderr", file=__import__("sys").stderr)
    actual = stderr.getvalue()
    rows.append({"case": "arbitrary_stderr", "passed":
                 actual == "arbitrary child stderr\n", "stderr": actual})

    report = {
        "artifact": "linux_memory_warning_filter_mutations",
        "gate_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checks_total": len(rows),
        "checks_failed": [row["case"] for row in rows if not row["passed"]],
        "rows": rows,
    }
    report["status"] = "PASS" if not report["checks_failed"] else "FAIL"
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with args.json.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "checks_total", "checks_failed",
                       "gate_source_sha256", "test_source_sha256")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
