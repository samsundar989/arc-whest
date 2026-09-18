"""Label-free, paired full-recurrence audit of an exact and sketched query.

Run through workerq: this executes two complete 1024x16 estimators. It reads no
truth or registered seed block, and makes no accuracy decision. Source paths
are explicit so the selected exact package can be supplied after selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import p2_flop_anchor as anchor


ROOT = Path(__file__).resolve().parent.parent


def _hash(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def describe_predictions(exact: np.ndarray, approx: np.ndarray) -> dict:
    """Validate and describe full outputs, without labels or score proxies."""
    if exact.shape != approx.shape or exact.ndim != 2:
        raise ValueError("prediction shape mismatch")
    if not np.isfinite(exact).all() or not np.isfinite(approx).all():
        raise ValueError("nonfinite prediction")
    layers = []
    for i, (e, a) in enumerate(zip(exact, approx)):
        delta = a - e
        layers.append({
            "layer": i,
            "exact_sha256": _hash(e),
            "approx_sha256": _hash(a),
            "exact_l2": float(np.linalg.norm(e)),
            "approx_l2": float(np.linalg.norm(a)),
            "displacement_l2": float(np.linalg.norm(delta)),
            "displacement_linf": float(np.max(np.abs(delta))),
            "relative_displacement_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(e), 1e-300)),
            "exact_min": float(np.min(e)),
            "approx_min": float(np.min(a)),
        })
    return {"shape": list(exact.shape), "exact_sha256": _hash(exact),
            "approx_sha256": _hash(approx), "layers": layers}


def _source(path: Path) -> tuple[str, str]:
    resolved = path.resolve(strict=True)
    if resolved.parent != ROOT or resolved.suffix != ".py":
        raise ValueError("source must be a root-level Python file")
    return resolved.name, hashlib.sha256(resolved.read_bytes()).hexdigest()


def _predict(name: str, weights_np: list[np.ndarray], estimator_seed: int) -> tuple[np.ndarray, float]:
    import flopscope as flops
    import flopscope.numpy as fnp

    anchor._require_real_flopscope(flops)
    anchor._stub_whestbench()
    try:
        flops.configure(symmetry_warnings=False)
    except Exception:
        pass
    weights = [fnp.array(w) for w in weights_np]
    mlp = __import__("whestbench").MLP(weights=weights, width=anchor.WIDTH,
                                         depth=anchor.DEPTH, seed=estimator_seed)
    mod = anchor._load_candidate(name)
    est = mod.Estimator()

    class Context:
        width, depth, flop_budget = anchor.WIDTH, anchor.DEPTH, anchor.BUDGET
        seed = 0

    est.setup(Context())
    with flops.BudgetContext(flop_budget=anchor.BUDGET, quiet=True) as ctx:
        pred = anchor.materialise_like_grader(est.predict(mlp, anchor.BUDGET),
                                               anchor.DEPTH, anchor.WIDTH)
        counted = float(ctx.flops_used)
    arr = np.asarray(pred, dtype=np.float64)
    if arr.shape != (anchor.DEPTH, anchor.WIDTH):
        raise ValueError(f"{name}: wrong prediction shape {arr.shape}")
    return arr, counted


def run(exact_path: Path, approx_path: Path) -> dict:
    exact_name, exact_sha = _source(exact_path)
    approx_name, approx_sha = _source(approx_path)
    if exact_name == approx_name or exact_sha == approx_sha:
        raise ValueError("exact and approximate sources must differ")
    weights, seed = anchor._mint_weights()
    fixture_sha = hashlib.sha256(b"".join(np.ascontiguousarray(w).tobytes()
                                            for w in weights)).hexdigest()
    e, ef = _predict(exact_name, weights, seed)
    a, af = _predict(approx_name, weights, seed)
    return {
        "status": "complete", "role": "label-free continuation diagnostic",
        "fixture": {"width": anchor.WIDTH, "depth": anchor.DEPTH,
                    "probe_seed": anchor.PROBE_SEED, "estimator_seed": seed,
                    "weights_sha256": fixture_sha,
                    "mint_sha256": hashlib.sha256(Path(anchor.__file__).with_name("mint_provenance.py").read_bytes()).hexdigest()},
        "exact": {"source": exact_name, "sha256": exact_sha, "counted_flops": ef},
        "approx": {"source": approx_name, "sha256": approx_sha, "counted_flops": af},
        "predictions": describe_predictions(e, a),
        "instrument_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exact", type=Path, required=True)
    ap.add_argument("--approx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = run(args.exact, args.approx)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, sort_keys=True, indent=2, allow_nan=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
