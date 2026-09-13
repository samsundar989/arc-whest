"""One label-free F21 lifecycle diagnostic on exposed P07 DESIGN identities.

This does not read truth, produce a score, or make a promotion decision.  The
all-zero target exists only because lifecycle_eval.evaluate requires an array.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() != _HERE]
for _path in (str(_HERE), str(_ROOT)):
    if _path not in sys.path:
        sys.path.append(_path)

import numpy as np  # noqa: E402
import lifecycle_eval as le  # noqa: E402
import seed_blocks  # noqa: E402
import seed_ledger  # noqa: E402


CANDIDATE = "estimator_k3_strassen_L4_f21.py"
CANDIDATE_SHA = "d77dab0e2c740cc9f6cd7d9172adacc8da731cc5b40ca87798f3fb006e475a94"
N = 128


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="new absolute path under experiments/")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if not out.is_relative_to(_ROOT / "experiments") or out.exists():
        parser.error("--out must be a new path under this repository's experiments/")
    if Path.cwd().resolve() != _ROOT:
        parser.error("run from repository root")

    candidate = _ROOT / CANDIDATE
    sources = {name: _sha(_ROOT / name) for name in
               (CANDIDATE, "local_sim/lifecycle_eval.py",
                "local_sim/mint_provenance.py", "local_sim/seed_blocks.py",
                "local_sim/f21_labelfree_stress.py")}
    if sources[CANDIDATE] != CANDIDATE_SHA:
        raise ValueError("F21 candidate SHA mismatch")
    seeds = seed_blocks.seeds("P07")[:N]
    if len(seeds) != N or len(set(seeds)) != N:
        raise ValueError("P07 identity selection mismatch")
    claims = seed_ledger.claims()
    if not any(c.block == "P07" and c.consumer == "local_sim/f21_labelfree_stress.py"
               and c.start == seeds[0] and c.n == N for c in claims):
        raise ValueError("P07 diagnostic identity span has not been ledger-claimed")

    spec = le.LifecycleSpec.for_round("v2-phase2", residual_wall_time_limit_s=None)
    spec_value = dataclasses.asdict(spec)
    if spec.deviations_from_round() != {
        "residual_wall_time_limit_s": [None, 0.4]
    } or spec.wall_time_limit_s != 120.0:
        raise ValueError("unexpected lifecycle spec")
    # Dummy targets are never read from disk and never published.  No gt_noise
    # argument is supplied; the evaluator's scored fields are discarded.
    dummy = np.zeros((N, spec.width), dtype=np.float32)
    rows, provenance = le.evaluate(str(candidate), seeds, dummy, spec=spec,
                                   weights_mode="f64_readonly", progress_every=8)
    if len(rows) != N or provenance["candidate_sha256"] != CANDIDATE_SHA:
        raise ValueError("incomplete or source-mismatched lifecycle evidence")
    for name, digest in sources.items():
        if _sha(_ROOT / name) != digest:
            raise ValueError(f"source changed during evaluation: {name}")

    fields = []
    counts: Counter[str] = Counter()
    for i, row in enumerate(rows):
        if row.index != i or row.input_seed != seeds[i] or row.candidate_sha256 != CANDIDATE_SHA:
            raise ValueError("unexpected row identity or source")
        failure = {
            "budget_exhausted": row.budget_exhausted,
            "time_exhausted": row.time_exhausted,
            "residual_exhausted": row.residual_exhausted,
            "combined_exhausted": row.combined_exhausted,
        }
        for key, value in failure.items():
            if value:
                counts[key] += 1
        if row.error_code:
            counts["error_code:" + row.error_code] += 1
        if row.failed:
            counts["any_failure"] += 1
        fields.append({
            "index": row.index,
            "input_seed": row.input_seed,
            "estimator_seed": row.estimator_seed,
            "failed": row.failed,
            **failure,
            "error_code": row.error_code,
            "wall_time_s": row.wall_time_s,
            "residual_wall_time_s": row.residual_wall_time_s,
            "flops_used": row.flops_used,
        })

    result = {
        "data_role": "diagnostic-exposed-design-label-free; no promotion claim",
        "candidate_sha256": CANDIDATE_SHA,
        "source_sha256": sources,
        "spec_sha256": _json_sha(spec_value),
        "seed_block": "P07",
        "rows": fields,
        "failure_categories": dict(sorted(counts.items())),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: a rerun can never overwrite the first observation.
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(f"published {N} label-free lifecycle rows; failures={counts['any_failure']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
