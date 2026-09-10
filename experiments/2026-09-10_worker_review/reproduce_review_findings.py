"""Read-only, tiny synthetic review probes. No network or truth file is loaded."""
from pathlib import Path
import ast
import json
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "local_sim"))
import numpy as np
import certify_candidate as cert
import p2_lab as lab
from test_linux_memory_binding import good_report, SHA, WIDTH, DEPTH, BUDGET, STACK

out = {"source_head_under_review": "bcca354", "uses_research_labels": False}
report = good_report()
report["artifact_version"] = 2
report["status"] = "FAIL"
report["candidate_run"]["cap_applied"] = False
report["candidate_run"]["predict_error"] = "MemoryError"
report["negative_control"]["survived"] = True
checks, degraded = cert.validate_memory_report(
    report, candidate_sha256=SHA, width=WIDTH, depth=DEPTH,
    budget=BUDGET, stack=STACK)
out["legacy_explicit_failure"] = {
    "failed_checks": [name for name, ok, detail in checks if not ok],
    "degraded": bool(degraded)}
out["actual_section_domains"] = {
    section: cert._domain(section) for section in ("5-8", "5b", "11")}
out["actual_stack"] = cert._certified_stack()
out["generated_arm_parse"] = {}
for name in ("k3_exact", "k3_k4coh", "k3_k4nobirth"):
    path = ROOT / "local_sim" / "p2_arms" / f"arm_{name}.py"
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        out["generated_arm_parse"][name] = "PARSES"
    except SyntaxError as exc:
        out["generated_arm_parse"][name] = {
            "error": exc.msg, "line": exc.lineno}

man = {"shape": {}, "budget": {}, "package_parity": {}}
z = {"means_r1": np.zeros((1, 1, 1)), "means_r2": np.zeros((1, 1, 1)),
     "gt_noise": np.zeros(1), "n_samples": 1}
class ArchiveLike(dict):
    @property
    def files(self):
        return list(self)

out["writer"] = {}
with tempfile.TemporaryDirectory(prefix="arc-review-") as tmp:
    for label, archive in (("actual_dict", z), ("legacy_archive_like", ArchiveLike(z))):
        path = str(Path(tmp) / f"{label}.json")
        try:
            lab._write(path, {}, {}, man, [0], archive,
                       SimpleNamespace(lam=0.0), [])
            out["writer"][label] = "WROTE"
        except Exception as exc:
            out["writer"][label] = f"{type(exc).__name__}: {exc}"
print(json.dumps(out, indent=2))
