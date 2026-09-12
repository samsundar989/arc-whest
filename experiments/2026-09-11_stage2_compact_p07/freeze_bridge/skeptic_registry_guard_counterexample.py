"""Truth-free adversarial check for cached namespace-package submodules.

The fixture preloads an unregistered repo-local namespace submodule during Python
startup, leaving the module as an attribute of its namespace package.  The
bootstrap purges the submodule from sys.modules but does not remove the stale
parent attribute.  A later normal ``from package import submodule`` can therefore
reuse the unregistered module without consulting the meta-path guard.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
BOOTSTRAP = ROOT / "local_sim" / "p2_locked_bootstrap.py"


def fingerprint(raw: bytes) -> bytes:
    pattern = re.compile(rb'(?m)^_SELF_FINGERPRINT = "[0-9a-f]{64}"')
    zeroed, count = pattern.subn(
        b'_SELF_FINGERPRINT = "' + b"0" * 64 + b'"', raw)
    assert count == 1
    digest = hashlib.sha256(zeroed.replace(b"\r\n", b"\n")).hexdigest()
    return pattern.sub(
        b'_SELF_FINGERPRINT = "' + digest.encode("ascii") + b'"', raw)


def target_text() -> bytes:
    raw = r'''from __future__ import annotations
import hashlib
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_BOOTSTRAP = os.path.join(_HERE, "p2_locked_bootstrap.py")
_SELF_FINGERPRINT = "0000000000000000000000000000000000000000000000000000000000000000"

def _attested_target_sha():
    raw = open(os.path.abspath(__file__), "rb").read()
    marker = re.compile(rb'(?m)^_SELF_FINGERPRINT = "[0-9a-f]{64}"')
    zeroed, count = marker.subn(
        b'_SELF_FINGERPRINT = "' + b"0" * 64 + b'"', raw)
    if count != 1:
        raise RuntimeError("bad fingerprint marker")
    if hashlib.sha256(zeroed.replace(b"\r\n", b"\n")).hexdigest() != _SELF_FINGERPRINT:
        raise RuntimeError("bad fingerprint")
    return hashlib.sha256(raw).hexdigest()

def _launch_locked_cli(args):
    return subprocess.run(
        [sys.executable, _BOOTSTRAP, "--target", os.path.abspath(__file__),
         "--expected-target-sha", _attested_target_sha(), "--"] + list(args),
        cwd=os.getcwd(), env=dict(os.environ)).returncode

if __name__ == "__main__":
    raise SystemExit(_launch_locked_cli(sys.argv[1:]))

if _HERE not in sys.path:
    sys.path.append(_HERE)
from stale_namespace import payload

_CLI_SOURCE_REGISTRY = {
    "locked_bootstrap": "local_sim/p2_locked_bootstrap.py",
    "p2_lab": "local_sim/p2_lab.py",
}
_BOUND_CLI_BINDINGS = None

def _cli_source_paths():
    return {name: os.path.abspath(os.path.join(_ROOT, *path.split("/")))
            for name, path in _CLI_SOURCE_REGISTRY.items()}

def _verify_cli_bindings(bindings, paths):
    if set(bindings) != set(paths):
        raise ValueError("binding mismatch")

def main(argv=None):
    Path = __import__("pathlib").Path
    Path(os.environ["SKEPTIC_PUBLICATION"]).write_text(payload.VALUE, encoding="utf-8")
    return 0
'''.encode("utf-8")
    return fingerprint(raw)


def run_one(command: list[str], root: Path, publication: Path) -> dict:
    marker = root / "preload_marker.txt"
    publication.unlink(missing_ok=True)
    marker.unlink(missing_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    env["SKEPTIC_PUBLICATION"] = str(publication)
    completed = subprocess.run(
        command, cwd=root, env=env, capture_output=True, text=True)
    return {
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "preload_executed": marker.exists(),
        "published": publication.read_text(encoding="utf-8")
        if publication.exists() else None,
        "disk_payload": (root / "stale_namespace" / "payload.py").read_text(
            encoding="utf-8"),
    }


def literal_registry(raw: bytes) -> dict[str, str]:
    tree = ast.parse(raw.decode("utf-8"))
    values = []
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Name) and
                node.targets[0].id == "_CLI_SOURCE_REGISTRY"):
            values.append(ast.literal_eval(node.value))
    assert len(values) == 1
    return values[0]


def sitecustomize_text(root: Path) -> str:
    preload_marker = str(root / "preload_marker.txt")
    payload_a = (
        "from pathlib import Path\n"
        f"Path({preload_marker!r}).write_text('A-loaded', encoding='utf-8')\n"
        "VALUE='A'\n")
    return (
        "from pathlib import Path\n"
        "import importlib\n"
        "p=Path(__file__).parent/'stale_namespace'/'payload.py'\n"
        f"p.write_text({payload_a!r}, encoding='utf-8')\n"
        "importlib.invalidate_caches()\n"
        "import stale_namespace.payload\n"
        "p.write_text(\"VALUE='B'\\n\", encoding='utf-8')\n")


def run_mutated_real_target(source_name: str) -> dict:
    with tempfile.TemporaryDirectory(
            prefix=f"arc-whest-skeptic-{source_name}-") as tmp:
        root = Path(tmp)
        local = root / "local_sim"
        namespace = root / "stale_namespace"
        local.mkdir()
        namespace.mkdir()
        original = (ROOT / "local_sim" / source_name).read_bytes()
        registry = literal_registry(original)
        for relative in registry.values():
            path = root.joinpath(*relative.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n" if path.suffix == ".json" else "# bound stub\n",
                            encoding="utf-8")
        shutil.copyfile(BOOTSTRAP, local / "p2_locked_bootstrap.py")

        injection = (
            "from stale_namespace import payload as _skeptic_payload\n"
            "with open(os.environ['SKEPTIC_PUBLICATION'], 'w', encoding='utf-8') as _skeptic_fh:\n"
            "    _skeptic_fh.write(_skeptic_payload.VALUE)\n\n").encode("utf-8")
        anchor = (b"sys.path =" if source_name == "p2_lab.py"
                  else b"_GENERATOR =")
        assert original.count(anchor) == 1
        mutated = original.replace(anchor, injection + anchor)
        target = local / source_name
        target.write_bytes(fingerprint(mutated))

        if source_name == "p2_lab.py":
            (local / "competition_score.py").write_text(
                "# registered import stub\n", encoding="utf-8")
            (local / "lifecycle_eval.py").write_text(
                "# registered import stub\n", encoding="utf-8")
            (local / "p2_freeze_k4_arms.py").write_text(
                "class FreezeRefused(ValueError): pass\n"
                "def _assert_loaded_source(expected=None): return expected\n",
                encoding="utf-8")
            (local / "phase2_manifest.py").write_text(
                "def load(): return {}\n", encoding="utf-8")

        (root / "sitecustomize.py").write_text(
            sitecustomize_text(root), encoding="utf-8")
        (namespace / "payload.py").write_text(
            "VALUE='initial'\n", encoding="utf-8")
        sha = hashlib.sha256(target.read_bytes()).hexdigest()
        direct = run_one([
            sys.executable, str(local / "p2_locked_bootstrap.py"),
            "--target", str(target), "--expected-target-sha", sha, "--", "--help",
        ], root, root / "direct_publication.txt")
        public = run_one(
            [sys.executable, str(target), "--help"],
            root, root / "public_publication.txt")
        return {
            "direct_bootstrap": direct,
            "real_public_script": public,
            "falsified": all(
                item["exit_code"] == 0 and item["preload_executed"] and
                item["published"] == "A" and "VALUE='B'" in item["disk_payload"]
                for item in (direct, public)),
        }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="arc-whest-skeptic-registry-") as tmp:
        root = Path(tmp)
        local = root / "local_sim"
        namespace = root / "stale_namespace"
        local.mkdir()
        namespace.mkdir()
        shutil.copyfile(BOOTSTRAP, local / "p2_locked_bootstrap.py")
        target = local / "p2_lab.py"
        target.write_bytes(target_text())
        sitecustomize = root / "sitecustomize.py"
        sitecustomize.write_text(
            sitecustomize_text(root),
            encoding="utf-8")
        payload = namespace / "payload.py"
        payload.write_text("VALUE='initial'\n", encoding="utf-8")
        sha = hashlib.sha256(target.read_bytes()).hexdigest()
        direct = run_one([
            sys.executable, str(local / "p2_locked_bootstrap.py"),
            "--target", str(target), "--expected-target-sha", sha,
        ], root, root / "direct_publication.txt")
        public = run_one(
            [sys.executable, str(target)], root, root / "public_publication.txt")
        result = {
            "schema": "arc-whest.skeptic-registry-guard-counterexample.v1",
            "truth_or_labels_opened": False,
            "minimal_supported_target": {
                "direct_bootstrap": direct,
                "real_public_script": public,
                "falsified": all(
                item["exit_code"] == 0 and item["preload_executed"] and
                item["published"] == "A" and "VALUE='B'" in item["disk_payload"]
                    for item in (direct, public)),
            },
        }
    result["copied_real_p2_lab"] = run_mutated_real_target("p2_lab.py")
    result["copied_real_p2_freeze_k4_arms"] = run_mutated_real_target(
        "p2_freeze_k4_arms.py")
    result["falsified"] = all(
        row["falsified"] for name, row in result.items()
        if name not in {"schema", "truth_or_labels_opened", "falsified"})
    rendered = json.dumps(result, indent=2) + "\n"
    Path(__file__).with_suffix(".json").write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 1 if not result["falsified"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
