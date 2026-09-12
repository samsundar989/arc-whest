"""Truth-free reproducer for the p2_lab child-marker provenance bypass.

Exit zero means the reviewed implementation accepted a process that had loaded
dependency A while recording the hash of on-disk dependency B.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile


REPO = Path(__file__).resolve().parents[3]
LAB_SOURCE = REPO / "local_sim" / "p2_lab.py"
FREEZER_SOURCE = REPO / "local_sim" / "p2_freeze_k4_arms.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="skeptic-lab-child-marker-") as raw_root:
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
        (local / "competition_score.py").write_text("# synthetic scorer\n", encoding="utf-8")
        (local / "phase2_manifest.py").write_text("def load(): return {}\n", encoding="utf-8")
        for name in (
            "p2_tag_drift.py",
            "official_config.py",
            "harness.py",
            "mint_provenance.py",
            "seed_ledger.py",
            "seed_blocks.py",
            "truth_reads.py",
            "p2_k4_fit.py",
            "p2_k4_fit_producer.py",
            "p2_build_k4_arms.py",
            "p2_compact_candidate_template.py",
        ):
            (local / name).write_text("# synthetic dependency\n", encoding="utf-8")
        (manifest_dir / "phase2_manifest.json").write_text("{}\n", encoding="utf-8")

        spec = importlib.util.spec_from_file_location("_skeptic_lab_loaded_a", copied_lab)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        assert module.le.VERSION == "A"

        lifecycle.write_text("VERSION='B'\n", encoding="utf-8")
        bindings = module._binding_snapshot(module._cli_source_paths())
        os.environ[module._CLI_BINDINGS_ENV] = json.dumps(bindings)
        try:
            # This sentinel stands in for production main(), which later calls
            # the already imported lifecycle_eval module during scored execution.
            module.main = lambda argv: 0 if module.le.VERSION == "A" else 9
            return_code = module.cli_entry([])
        finally:
            os.environ.pop(module._CLI_BINDINGS_ENV, None)

        result = {
            "schema": "arc-whest.skeptic-child-marker-counterexample.v1",
            "truth_or_labels_opened": False,
            "reviewed_lab_sha256": sha256(LAB_SOURCE),
            "reviewed_freezer_sha256": sha256(FREEZER_SOURCE),
            "loaded_lifecycle_version": module.le.VERSION,
            "on_disk_lifecycle_version": lifecycle.read_text(encoding="utf-8").strip(),
            "recorded_lifecycle_sha256": bindings["lifecycle_eval"]["sha256"],
            "actual_disk_lifecycle_sha256": sha256(lifecycle),
            "cli_return_code": return_code,
            "accepted_loaded_A_while_binding_disk_B": (
                return_code == 0
                and module.le.VERSION == "A"
                and bindings["lifecycle_eval"]["sha256"] == sha256(lifecycle)
            ),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["accepted_loaded_A_while_binding_disk_B"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
