"""Truth-free direct-bootstrap caller-supplied pending-stage regression."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from local_sim import test_p2_lab_locked_lifecycle as fixture  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="skeptic-stage-spoof-", dir=ROOT) as td:
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
        requested = root / "requested_report.json"
        stage = root / "caller_chosen_complete_report.json"
        env = dict(os.environ, ARC_WHEST_P2_LAB_REPORT_STAGE=str(stage))
        command = [
            sys.executable, str(target.parent / "p2_locked_bootstrap.py"),
            "--target", str(target), "--expected-target-sha", sha(target), "--",
            "--truth", str(truth), "--freeze-receipt", str(receipt),
            "--out", str(requested), "--control", "exact",
            "--competition-cost-lambda", "0", "--residual-cap", "0",
            "--depth-subset", "2",
        ]
        run = subprocess.run(command, capture_output=True, text=True, cwd=root, env=env)
        staged = json.loads(stage.read_text(encoding="utf-8")) if stage.exists() else None
        result = {
            "schema": "arc-whest.skeptic-stage-spoof.v1",
            "truth": "two generated 2x2 sacrificial networks; no P01/P07/vault read",
            "current_lab_sha256": sha(ROOT / "local_sim" / "p2_lab.py"),
            "current_bootstrap_sha256": sha(ROOT / "local_sim" / "p2_locked_bootstrap.py"),
            "copied_hooked_lab_sha256": sha(target),
            "direct_bootstrap_returncode": run.returncode,
            "final_origin_refused": (
                "loaded repo-local module _skeptic_rogue is absent from the locked registry"
                in run.stdout),
            "requested_out_exists": requested.exists(),
            "caller_stage_exists": stage.exists(),
            "caller_stage_status": staged.get("status") if staged else None,
            "caller_stage_arms": sorted(staged.get("arms", {})) if staged else [],
            "truth_result_artifact": (
                staged["k4_freeze"]["protected_truth"]["reservation_receipt"]["result_artifact"]
                if staged else None),
            "requested_out_relative": requested.relative_to(root).as_posix(),
            "caller_stage_relative": stage.relative_to(root).as_posix(),
            "stdout_tail": run.stdout[-1800:],
            "stderr_tail": run.stderr[-600:],
        }
        raw = json.dumps(result, indent=2, sort_keys=True) + "\n"
        Path(args.output).open("x", encoding="utf-8", newline="\n").write(raw)
        print(raw)
        return 0 if (
            run.returncode == 2 and result["final_origin_refused"]
            and not result["requested_out_exists"]
            and result["caller_stage_exists"]
            and result["caller_stage_status"] == "pending"
            and result["truth_result_artifact"] == result["requested_out_relative"]
        ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
