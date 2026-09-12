"""Record HEAD and current Python/result bytes around live preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("start", "end"))
    args = parser.parse_args()
    tracked = subprocess.check_output(["git", "ls-files", "-z", "*.py"], cwd=ROOT)
    files = [ROOT / raw.decode("utf-8") for raw in tracked.split(b"\0") if raw]
    files.append(Path(__file__))
    hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
              for path in files}
    source = ROOT / "local_sim" / "test_p2_k4_increment_axis_gauss.py"
    result = HERE / "result.json"
    record = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                        cwd=ROOT, text=True).strip(),
        "source_sha256": sha(source), "result_sha256": sha(result),
        "python_hashes": hashes,
    }
    (HERE / f"byte_state_{args.phase}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    print(json.dumps({"head": record["head"],
                      "source_sha256": record["source_sha256"],
                      "result_sha256": record["result_sha256"],
                      "python_file_count": len(hashes)}, sort_keys=True))


if __name__ == "__main__":
    main()
