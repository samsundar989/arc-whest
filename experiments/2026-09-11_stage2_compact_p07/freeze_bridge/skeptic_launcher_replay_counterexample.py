"""Truth-free counterexample for the bridge launcher's parent-authentication claim.

Exit zero means both real private CLI modes were replayed without a lock-owning
parent and a coordinated caller entered the lab worker with lifecycle version A
loaded while the accepted binding named on-disk version B.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parents[3]
LOCAL = REPO / "local_sim"
LAB_SOURCE = LOCAL / "p2_lab.py"
FREEZER_SOURCE = LOCAL / "p2_freeze_k4_arms.py"
OUT = Path(__file__).with_name("skeptic_launcher_replay_counterexample.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def direct_replay(module) -> dict:
    nonce = secrets.token_hex(32)
    bindings = module._binding_snapshot(module._cli_source_paths())
    payload = json.dumps({
        "schema": module._LOCKED_CHILD_SCHEMA,
        "launch_nonce": nonce,
        "bindings": bindings,
    }, sort_keys=True, separators=(",", ":"))
    completed = subprocess.run(
        [sys.executable, module.__file__, module._LOCKED_CHILD_ARG + "=" + nonce,
         "--help"],
        input=payload, text=True, capture_output=True, cwd=REPO)
    return {
        "caller_held_parent_locks": False,
        "caller_chose_nonce": True,
        "returncode": completed.returncode,
        "stdout_has_usage": "usage:" in completed.stdout,
        "stderr": completed.stderr,
    }


def coordinated_loaded_a_disk_b() -> dict:
    with tempfile.TemporaryDirectory(prefix="skeptic-launcher-a-b-") as raw_root:
        root = Path(raw_root)
        local = root / "local_sim"
        manifest_dir = root / "experiments" / "2026-08-21_phase2_manifest"
        local.mkdir()
        manifest_dir.mkdir(parents=True)
        copied_lab = local / "p2_lab.py"
        copied_freezer = local / "p2_freeze_k4_arms.py"
        lifecycle = local / "lifecycle_eval.py"
        shutil.copyfile(LAB_SOURCE, copied_lab)
        shutil.copyfile(FREEZER_SOURCE, copied_freezer)
        lifecycle.write_text("VERSION='A'\n", encoding="utf-8")
        (local / "competition_score.py").write_text(
            "# synthetic scorer\n", encoding="utf-8")
        (local / "phase2_manifest.py").write_text(
            "def load(): return {}\n", encoding="utf-8")
        for name in (
            "p2_tag_drift.py", "official_config.py", "harness.py",
            "mint_provenance.py", "seed_ledger.py", "seed_blocks.py",
            "truth_reads.py", "p2_k4_fit.py", "p2_k4_fit_producer.py",
            "p2_build_k4_arms.py", "p2_compact_candidate_template.py",
        ):
            (local / name).write_text(
                "# synthetic dependency\n", encoding="utf-8")
        (manifest_dir / "phase2_manifest.json").write_text(
            "{}\n", encoding="utf-8")

        old_path = list(sys.path)
        old_argv = list(sys.argv)
        old_stdin = sys.stdin
        try:
            sys.path.insert(0, str(local))
            module = load_source("_skeptic_launcher_loaded_a", copied_lab)
            assert module.le.VERSION == "A"
            lifecycle.write_text("VERSION='B'\n", encoding="utf-8")
            bindings = module._binding_snapshot(module._cli_source_paths())
            nonce = secrets.token_hex(32)
            payload = json.dumps({
                "schema": module._LOCKED_CHILD_SCHEMA,
                "launch_nonce": nonce,
                "bindings": bindings,
            }, sort_keys=True, separators=(",", ":"))

            # These are ordinary mutable Python process inputs. No parent or
            # inherited capability is needed to satisfy the private branch.
            module.__name__ = "__main__"
            sys.argv = [str(copied_lab), module._LOCKED_CHILD_ARG + "=" + nonce]
            sys.stdin = io.StringIO(payload)
            module.main = lambda _argv: 0 if module.le.VERSION == "A" else 9
            returncode = module.cli_entry()
        finally:
            sys.path[:] = old_path
            sys.argv[:] = old_argv
            sys.stdin = old_stdin

        disk_sha = sha256(lifecycle)
        return {
            "caller_held_parent_locks": False,
            "caller_chose_nonce": True,
            "returncode": returncode,
            "loaded_lifecycle_version": module.le.VERSION,
            "on_disk_lifecycle_version": lifecycle.read_text(encoding="utf-8").strip(),
            "recorded_lifecycle_sha256": bindings["lifecycle_eval"]["sha256"],
            "actual_disk_lifecycle_sha256": disk_sha,
            "accepted_loaded_A_while_binding_disk_B": (
                returncode == 0
                and module.le.VERSION == "A"
                and bindings["lifecycle_eval"]["sha256"] == disk_sha
            ),
        }


def main() -> int:
    sys.path.insert(0, str(LOCAL))
    import p2_freeze_k4_arms as freezer
    import p2_lab as lab

    result = {
        "schema": "arc-whest.skeptic-launcher-replay-counterexample.v1",
        "truth_or_labels_opened": False,
        "reviewed_lab_sha256": sha256(LAB_SOURCE),
        "reviewed_freezer_sha256": sha256(FREEZER_SOURCE),
        "direct_lab_private_mode": direct_replay(lab),
        "direct_freezer_private_mode": direct_replay(freezer),
        "coordinated_lab_loaded_A_disk_B": coordinated_loaded_a_disk_b(),
    }
    result["parent_authentication_replayable"] = (
        result["direct_lab_private_mode"]["returncode"] == 0
        and result["direct_freezer_private_mode"]["returncode"] == 0
        and result["coordinated_lab_loaded_A_disk_B"][
            "accepted_loaded_A_while_binding_disk_B"]
    )
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["parent_authentication_replayable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
