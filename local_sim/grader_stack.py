"""The grading server's package stack, DERIVED FROM EVIDENCE, not from belief.

Until 2026-09-05 the repository pinned `whestbench>=0.16.0,<0.17.0` and
`flopscope>=0.12.0,<0.13.0`, and `official_config.verify()` asserted only
`0.16.x` / `0.12.x`. On 2026-09-05 the grading server moved to
**0.16.1 / 0.12.1** underneath us while this host stayed on 0.16.0 / 0.12.0,
and nothing anywhere went red -- the range accepted both. The stack move was
not the cause of that day's seven failed submissions (a `copy=False` was), but
it was invisible, and an invisible difference between the measurement host and
the grading host is exactly the class of defect this repository exists to
refuse.

So the pin is no longer a human's opinion of the range. It is read out of the
newest `evaluation.runtime_environment` block in `experiments/run_reports/`,
frozen into `grader_stack.json` next to this file, and asserted for **equality**
by `official_config.verify()` -- which `scripts/preflight.ps1` step 1 runs.

Two properties this module deliberately has:

* **Derivation is separate from assertion.** `derive()` reads the reports;
  `load()` reads the frozen file; `verify()` in `official_config` compares the
  frozen file to the *installed* packages. Refreshing the file is a deliberate
  act (`python local_sim/grader_stack.py --write`), so the gate cannot quietly
  re-baseline itself onto whatever the server did last.
* **A missing file is RED, not green.** `load()` raises. A gate that degrades
  to "no evidence, therefore fine" is the failure mode of the old range check.

Usage:

    python local_sim/grader_stack.py            # show derived vs frozen vs installed
    python local_sim/grader_stack.py --write    # re-freeze from the newest report
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

STACK_PATH = os.path.join(_HERE, "grader_stack.json")
REPORTS_DIR = os.path.join(_ROOT, "experiments", "run_reports")

#: The runtime_environment keys we pin for EQUALITY. `participant_*` keys are
#: recorded for provenance but not asserted against the installed stack: this
#: host runs a different Python than the grader's participant shim, and a check
#: that can never pass is not a check.
PINNED_KEYS: Tuple[str, ...] = (
    "whestbench_version",
    "flopscope_version",
)

#: Recorded, reported on drift, but NOT gate-red. The grader's numpy/python are
#: not ours to match; when they move we want to see it, not to fail.
OBSERVED_KEYS: Tuple[str, ...] = (
    "flopscope_client_version",
    "flopscope_server_version",
    "numpy_version",
    "python_version",
    "participant_shim_version",
    "participant_python_version",
)


def _runtime_environment(doc: dict) -> Optional[dict]:
    ev = doc.get("evaluation")
    if isinstance(ev, dict):
        re_ = ev.get("runtime_environment")
        if isinstance(re_, dict):
            return re_
    re_ = doc.get("runtime_environment")
    return re_ if isinstance(re_, dict) else None


def _timestamp(doc: dict) -> str:
    ev = doc.get("evaluation") or {}
    for key in ("completed_at", "updated_at", "started_at"):
        v = ev.get(key)
        if isinstance(v, str) and v:
            return v
    v = doc.get("createdAt")
    return v if isinstance(v, str) else ""


def scan(reports_dir: str = REPORTS_DIR) -> List[dict]:
    """Every graded report that carries a runtime_environment, newest first."""
    out: List[dict] = []
    for path in sorted(glob.glob(os.path.join(reports_dir, "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        re_ = _runtime_environment(doc)
        if re_ is None:
            continue
        out.append({
            "source_report": os.path.basename(path),
            "submission_id": str(doc.get("id") or
                                 (doc.get("evaluation") or {}).get("submission_id") or ""),
            "observed_at": _timestamp(doc),
            "status": doc.get("status", ""),
            "runtime_environment": re_,
        })
    out.sort(key=lambda r: (r["observed_at"], r["submission_id"]), reverse=True)
    return out


def derive(reports_dir: str = REPORTS_DIR) -> dict:
    """The frozen-stack document implied by the NEWEST graded report.

    Raises if no report carries a `runtime_environment` -- there is then no
    evidence, and this module refuses to invent any.
    """
    rows = scan(reports_dir)
    if not rows:
        raise RuntimeError(
            f"no graded report under {reports_dir} carries an "
            f"evaluation.runtime_environment block -- there is no evidence to "
            f"pin against. Fetch one with local_sim/fetch_graded_report.py."
        )
    newest = rows[0]
    re_ = newest["runtime_environment"]
    missing = [k for k in PINNED_KEYS if not re_.get(k)]
    if missing:
        raise RuntimeError(
            f"{newest['source_report']} has a runtime_environment but is "
            f"missing {missing}; refusing to freeze a partial stack."
        )
    return {
        "_comment": (
            "DERIVED, do not hand-edit. Regenerate with "
            "`python local_sim/grader_stack.py --write` after fetching a newer "
            "graded report. official_config.verify() asserts PINNED == installed."
        ),
        "derived_from": {
            "source_report": newest["source_report"],
            "submission_id": newest["submission_id"],
            "observed_at": newest["observed_at"],
            "status": newest["status"],
            "n_reports_with_runtime_environment": len(rows),
        },
        "pinned": {k: re_[k] for k in PINNED_KEYS},
        "observed": {k: re_.get(k) for k in OBSERVED_KEYS},
    }


def load(path: str = STACK_PATH) -> dict:
    """The frozen document. Raises if absent -- no file means no gate."""
    if not os.path.exists(path):
        raise RuntimeError(
            f"{path} does not exist. The grader-stack pin is derived evidence, "
            f"not a default; run `python local_sim/grader_stack.py --write`."
        )
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    for k in PINNED_KEYS:
        if not (doc.get("pinned") or {}).get(k):
            raise RuntimeError(f"{path} is missing pinned key {k!r}")
    return doc


def pinned(path: str = STACK_PATH) -> Dict[str, str]:
    return dict(load(path)["pinned"])


def installed() -> Dict[str, str]:
    import importlib.metadata as md
    out: Dict[str, str] = {}
    for key, dist in (("whestbench_version", "whestbench"),
                      ("flopscope_version", "flopscope")):
        try:
            out[key] = md.version(dist)
        except Exception as exc:  # pragma: no cover - a missing dist is RED
            out[key] = f"MISSING ({type(exc).__name__})"
    return out


def drift(path: str = STACK_PATH) -> List[Tuple[str, str, str, bool]]:
    """`(key, pinned, installed, ok)` for each equality-pinned key."""
    want = pinned(path)
    have = installed()
    return [(k, want[k], have.get(k, "MISSING"), want[k] == have.get(k))
            for k in PINNED_KEYS]


def write(path: str = STACK_PATH, reports_dir: str = REPORTS_DIR) -> dict:
    doc = derive(reports_dir)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return doc


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="re-freeze grader_stack.json from the newest graded report")
    ap.add_argument("--reports", default=REPORTS_DIR)
    a = ap.parse_args(argv)

    derived = derive(a.reports)
    print("DERIVED from", derived["derived_from"]["source_report"],
          f"(submission {derived['derived_from']['submission_id']},",
          f"{derived['derived_from']['observed_at']})")
    for k, v in derived["pinned"].items():
        print(f"   pinned   {k:<28} {v}")
    for k, v in derived["observed"].items():
        print(f"   observed {k:<28} {v}")

    if a.write:
        write(a.reports and STACK_PATH or STACK_PATH, a.reports)
        print(f"\nwrote {STACK_PATH}")

    print()
    try:
        rows = drift()
    except RuntimeError as exc:
        print("FROZEN FILE:", exc)
        return 1
    bad = 0
    for k, want, have, ok in rows:
        print(f"   {'ok ' if ok else 'RED'}  {k:<28} pinned {want}   installed {have}")
        bad += 0 if ok else 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
