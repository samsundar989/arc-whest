"""Truth-free final-origin-audit/report-publication counterexample.

This copies the registered lab and its real lock-first bootstrap into a
sacrificial tree via the existing tiny fixture. It changes only that copied
lab: immediately after its normal exclusive report write, it uses a direct
loader to register an unlisted repo-local module. The bootstrap's final audit
should refuse; this probe records whether the published report survives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from local_sim import test_p2_lab_locked_lifecycle as fixture  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="skeptic-late-origin-", dir=ROOT) as td:
        root = Path(td)
        target, receipt, truth = fixture._build(root)
        rogue = root / "rogue.py"
        rogue.write_text("X = 1\n", encoding="utf-8")
        with target.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(
                "\n_original_write_for_probe = _write\n"
                "def _write(*args, **kwargs):\n"
                "    result = _original_write_for_probe(*args, **kwargs)\n"
                "    import importlib.util\n"
                "    rogue_path = os.path.join(_ROOT, 'rogue.py')\n"
                "    rogue_spec = importlib.util.spec_from_file_location(\n"
                "        '_skeptic_rogue', rogue_path)\n"
                "    rogue_module = importlib.util.module_from_spec(rogue_spec)\n"
                "    sys.modules[rogue_spec.name] = rogue_module\n"
                "    rogue_spec.loader.exec_module(rogue_module)\n"
                "    return result\n"
            )
        fixture._refresh_fingerprint(target)
        out = root / "late_report.json"
        run = fixture._invoke(target, receipt, truth, out)
        report = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
        refusal = "loaded repo-local module _skeptic_rogue is absent from the locked registry"
        result = {
            "schema": "arc-whest.skeptic-late-origin-probe.v1",
            "truth": "two generated 2x2 sacrificial networks; no P01/P07/vault read",
            "source": "copied current p2_lab with probe hook after _write",
            "current_lab_sha256": sha(ROOT / "local_sim" / "p2_lab.py"),
            "current_bootstrap_sha256": sha(ROOT / "local_sim" / "p2_locked_bootstrap.py"),
            "copied_hooked_lab_sha256": sha(target),
            "copied_rogue_sha256": sha(rogue),
            "returncode": run.returncode,
            "final_origin_refused": refusal in run.stdout,
            "report_exists_after_refusal": out.exists(),
            "report_status_after_refusal": report.get("status") if report else None,
            "reported_arm_names": sorted(report.get("arms", {})) if report else [],
            "stdout_tail": run.stdout[-2000:],
            "stderr_tail": run.stderr[-1000:],
        }
        raw = json.dumps(result, indent=2, sort_keys=True) + "\n"
        with open(args.output, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(raw)
        print(raw)
        return 0 if (run.returncode == 2 and result["final_origin_refused"] and
                     result["report_exists_after_refusal"] and
                     result["report_status_after_refusal"] == "complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
