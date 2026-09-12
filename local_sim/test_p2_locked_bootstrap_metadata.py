"""Truth-free protocol test for metadata discovery under the locked bootstrap."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BOOTSTRAP = os.path.join(_ROOT, "local_sim", "p2_locked_bootstrap.py")


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _target_source(target, result_path):
    registry = {
        "locked_bootstrap": _BOOTSTRAP,
        "p2_lab": target,
    }
    return '''import json
import os

_CLI_SOURCE_REGISTRY = %r
_BOUND_CLI_BINDINGS = None

def _cli_source_paths():
    return dict(_CLI_SOURCE_REGISTRY)

def _verify_cli_bindings(bindings, paths):
    if set(bindings) != set(paths):
        raise ValueError("binding names differ")
    for name, path in paths.items():
        if os.path.normcase(os.path.realpath(bindings[name]["path"])) != os.path.normcase(os.path.realpath(path)):
            raise ValueError("binding path differs")

def main(argv=None):
    import dill
    import importlib.metadata
    payload = {
        "distribution": "dill",
        "metadata_version": importlib.metadata.version("dill"),
        "module_version": dill.__version__,
        "module_file": os.path.realpath(dill.__file__),
    }
    with open(%r, "x", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True)
    return 0
''' % (registry, result_path)


def run_probe():
    with tempfile.TemporaryDirectory(
            prefix="p2-bootstrap-metadata-", dir=_ROOT) as root:
        target = os.path.join(root, "p2_lab.py")
        result_path = os.path.join(root, "result.json")
        with open(target, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(_target_source(target, result_path))
        command = [
            sys.executable, _BOOTSTRAP,
            "--target", target,
            "--expected-target-sha", _sha256(target),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        result = None
        if os.path.isfile(result_path):
            with open(result_path, encoding="utf-8") as fh:
                result = json.load(fh)
        expected_version = importlib.metadata.version("dill")
        passed = (
            completed.returncode == 0 and result is not None and
            result["metadata_version"] == expected_version and
            result["module_version"] == expected_version and
            os.path.isfile(result["module_file"])
        )
        return {
            "pass": passed,
            "expected_version": expected_version,
            "bootstrap_returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "result": result,
        }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    evidence = run_probe()
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "x", encoding="utf-8", newline="\n") as fh:
            json.dump(evidence, fh, indent=2, sort_keys=True)
            fh.write("\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
