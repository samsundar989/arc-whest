"""Unmocked P14-reader provenance fixture; synthetic rows and truth bytes only."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

import design_sidecar as ds
import lifecycle_eval as le
import s18_birth_decide as m
from test_s18_birth_decide import TRUTH_SHA as UNUSED_SYNTHETIC_SHA, fixture


def refuses(call, fragment):
    try:
        call()
    except (ValueError, FileNotFoundError) as exc:
        assert fragment in str(exc), str(exc)
    else:
        raise AssertionError(f"accepted tampered {fragment}")


def main():
    with tempfile.TemporaryDirectory(prefix="s18_sidecar_", dir=m.ROOT) as tmp:
        top = Path(tmp)
        root = top / "snapshot"
        root.mkdir()
        for name in (m.CONTROL_NAME, m.CANDIDATE_NAME):
            shutil.copyfile(m.ROOT / name, root / name)
        dep = root / "numeric_dependency.py"
        dep.write_bytes(b"frozen synthetic dependency\n")
        truth = root / "truth_TEST.bin"
        truth.write_bytes(b"synthetic labels never interpreted\n")
        truth_sha = m._sha(truth)
        pred = top / "external" / "predictions"
        pred.mkdir(parents=True)
        score, side, paths = fixture(pred, root)
        man = side["manifest"]
        man["truth"] = score["truth"] = truth.name
        man["truth_sha256"] = score["truth_sha256"] = truth_sha
        man["packages"] = {"flopscope": "synthetic"}
        man["mint"] = {"fixture": True}
        man["dependency_sha256"] = {dep.name: m._sha(dep)}
        for arm in m.LABELS:
            info = man["arms"][arm]
            writer = ds.ArmWriter(str(pred), arm, width=1024,
                                  candidate=info["path"],
                                  candidate_sha256=info["candidate_sha256"])
            for full in info["evidence_rows"]:
                full["truth_sha256"] = truth_sha
                row = le.EvidenceRow(**full)
                writer.append(row, np.zeros(1024, dtype=np.float32))
            info["rows_hash"] = score["arms"][arm]["rows_hash"] = le.rows_hash(
                [le.EvidenceRow(**r) for r in info["evidence_rows"]])
        ds.write_manifest(str(pred), man)
        score_path = top / "external" / "score.json"
        score_path.write_text(json.dumps(score), encoding="utf-8")

        def read():
            return m.synthesize(score_path, pred, truth_sha, run_root=root)

        assert read()["artifact"] == "s18_birth_decide"
        # Tamper one actual byte at a time; restore before the next refusal.
        stream = Path(ds.stream_path(str(pred), "candidate"))
        original = stream.read_bytes()
        stream.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        refuses(read, "stored prediction hashes")
        stream.write_bytes(original)
        for path, fragment in ((root / m.CANDIDATE_NAME, "source bytes differ"),
                               (dep, "dependency"), (truth, "truth")):
            original = path.read_bytes()
            path.write_bytes(original + b"x")
            refuses(read, fragment)
            path.write_bytes(original)
        assert read()["truth_sha256"] == truth_sha
        print("PASS: unmocked P14 reader, prediction/source/dependency/truth byte refusals")


if __name__ == "__main__":
    main()
