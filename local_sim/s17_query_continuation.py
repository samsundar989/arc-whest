"""Label-free, paired full-recurrence audit of an exact and sketched query.

Run through workerq: this executes two complete 1024x16 estimators. It reads no
truth or registered seed block, and makes no accuracy decision. Source paths
are explicit so the selected exact package can be supplied after selection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
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


def _pinned_stack() -> dict:
    pinned_path = Path(anchor.__file__).with_name("grader_stack.json")
    pin = json.loads(pinned_path.read_text(encoding="utf-8"))["pinned"]
    installed = {"flopscope_version": importlib.metadata.version("flopscope"),
                 "whestbench_version": importlib.metadata.version("whestbench")}
    if installed != pin:
        raise RuntimeError(f"grader package stack drift: {installed} != {pin}")
    return {"installed": installed, "pin_sha256": hashlib.sha256(pinned_path.read_bytes()).hexdigest(),
            "anchor_sha256": hashlib.sha256(Path(anchor.__file__).read_bytes()).hexdigest()}


def _query_lines(source: Path) -> dict[int, str]:
    lines = source.read_text(encoding="utf-8").splitlines()
    markers = {
        "rank_fallback": "old = _mm(_mm(LL, FF.T), tw.Q.T)",
        "approximate": "old = _mm(U, _mm(_mm(_mm(U.T, LL), FF.T), tw.Q.T))",
    }
    rank_lines = [i for i, line in enumerate(lines, 1) if line.strip() == markers["rank_fallback"]]
    approx_lines = [i for i, line in enumerate(lines, 1) if line.strip() == markers["approximate"]]
    if len(rank_lines) != 2 or len(approx_lines) != 1 or not (rank_lines[0] < rank_lines[1] < approx_lines[0]):
        raise ValueError("approximate query branch signatures changed")
    return {rank_lines[0]: "rank_fallback", rank_lines[1]: "deficient_fallback",
            approx_lines[0]: "approximate"}


class QueryTrace:
    """Observe executed branch lines; never change the estimator's data path."""

    def __init__(self, source: Path):
        self.path = str(source.resolve())
        self.lines = _query_lines(source)
        self.events: list[dict] = []

    def trace(self, frame, event, arg):
        if event == "call":
            if frame.f_code.co_filename == self.path and frame.f_code.co_name == "_shared_contract_lean":
                return self.trace
            return None
        if event == "line" and frame.f_lineno in self.lines:
            caller = frame.f_back
            while caller is not None and not (caller.f_code.co_name == "_layer" and
                                               caller.f_code.co_filename == self.path):
                caller = caller.f_back
            if caller is None or not isinstance(caller.f_locals.get("layer_index"), int):
                raise RuntimeError("query trace could not bind a layer index")
            self.events.append({"layer": caller.f_locals["layer_index"],
                                "branch": self.lines[frame.f_lineno]})
        return self.trace

    def summary(self) -> dict:
        counts = {kind: sum(e["branch"] == kind for e in self.events)
                  for kind in ("rank_fallback", "deficient_fallback", "approximate")}
        if not self.events:
            raise RuntimeError("no old-query branch observed")
        first = next((e["layer"] for e in self.events if e["branch"] == "approximate"), None)
        return {"branch_counts": counts, "first_approximate_layer": first,
                "first_old_query_layer": min(e["layer"] for e in self.events),
                "events": self.events}


def _predict(name: str, weights_np: list[np.ndarray], estimator_seed: int,
             trace: QueryTrace | None = None) -> tuple[np.ndarray, float]:
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
    previous_trace = sys.gettrace()
    try:
        if trace is not None:
            sys.settrace(trace.trace)
        with flops.BudgetContext(flop_budget=anchor.BUDGET, quiet=True) as ctx:
            pred = anchor.materialise_like_grader(est.predict(mlp, anchor.BUDGET),
                                                   anchor.DEPTH, anchor.WIDTH)
            counted = float(ctx.flops_used)
    finally:
        if trace is not None:
            sys.settrace(previous_trace)
    arr = np.asarray(pred, dtype=np.float64)
    if arr.shape != (anchor.DEPTH, anchor.WIDTH):
        raise ValueError(f"{name}: wrong prediction shape {arr.shape}")
    return arr, counted


def run(exact_path: Path, approx_path: Path) -> dict:
    stack = _pinned_stack()
    exact_name, exact_sha = _source(exact_path)
    approx_name, approx_sha = _source(approx_path)
    if exact_name == approx_name or exact_sha == approx_sha:
        raise ValueError("exact and approximate sources must differ")
    weights, seed = anchor._mint_weights()
    fixture_sha = hashlib.sha256(b"".join(np.ascontiguousarray(w).tobytes()
                                            for w in weights)).hexdigest()
    trace = QueryTrace(ROOT / approx_name)
    e, ef = _predict(exact_name, weights, seed)
    a, af = _predict(approx_name, weights, seed, trace)
    if _source(exact_path) != (exact_name, exact_sha) or _source(approx_path) != (approx_name, approx_sha):
        raise RuntimeError("source changed during paired execution")
    if _pinned_stack() != stack:
        raise RuntimeError("meter/package stack changed during paired execution")
    return {
        "status": "complete", "role": "label-free continuation diagnostic",
        "fixture": {"width": anchor.WIDTH, "depth": anchor.DEPTH,
                    "probe_seed": anchor.PROBE_SEED, "estimator_seed": seed,
                    "weights_sha256": fixture_sha,
                    "mint_sha256": hashlib.sha256(Path(anchor.__file__).with_name("mint_provenance.py").read_bytes()).hexdigest()},
        "exact": {"source": exact_name, "sha256": exact_sha, "counted_flops": ef},
        "approx": {"source": approx_name, "sha256": approx_sha, "counted_flops": af},
        "predictions": describe_predictions(e, a),
        "query_trace": trace.summary(), "meter_stack": stack,
        "instrument_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def write_exclusive(out: Path, result: dict) -> None:
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out.parent,
                                         prefix=out.name + ".", suffix=".tmp",
                                         delete=False) as fh:
            tmp = Path(fh.name)
            json.dump(result, fh, sort_keys=True, indent=2, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        # Same-directory hard link is atomic and refuses an output created by
        # another process while these two full estimators were running.
        os.link(tmp, out)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exact", type=Path, required=True)
    ap.add_argument("--approx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = run(args.exact, args.approx)
    write_exclusive(args.out, result)
    print(args.out)


if __name__ == "__main__":
    main()
