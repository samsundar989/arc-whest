"""Float32 trajectory parity for the one-change L4 D21 fusion candidate.

This is an engineering check, not a score or cost measurement. It uses fixed
canonical_mint weights without labels or a decision-block claim. It compares the
actual flopscope states returned by _layer, then the public Estimator.predict
outputs. Both modules retain their configured L4 Strassen backend and generation
chooser; the effective Strassen depth is shape dependent.

# seed-ledger: no block named -- fixed canonical mints, no labels or selection.

Small CPU check (effective Strassen depth 1):
    uv run python local_sim/test_p2_d21_fusion_trajectory.py --out experiments/2026-09-12_f21/trajectory_128x4.json

Full L4 backend check (effective depth 4; submit through workerq):
    uv run python local_sim/test_p2_d21_fusion_trajectory.py --width 1024 --depth 4 --out experiments/2026-09-12_f21/trajectory_1024x4.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tarfile
import types


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# local_sim contains deliberately unmetered flopscope/whestbench shims. Import
# canonical_mint from there only after resolving the installed official stack.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != HERE]
for path in (str(ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.append(path)

PARENT_TAR = ROOT / "submission-k3strassen-L4.tar.gz"
PARENT_SHA = "052dd768e970c4bae831cff26f8ff15b0487f3b95eadf2fd3578e542aec34d58"
CANDIDATE = ROOT / "estimator_k3_strassen_L4_f21.py"
BUDGET = 2 ** 41
ENTRY_TOL = 2e-3
FINAL_RMS_TOL = 1e-2
CONFIG_NAMES = ("ALPHA_CAP", "_STRASSEN_DEPTH", "_STRASSEN_MIN_BLOCK",
                "_BUDGET_FRACTION", "_ELEM_PER_LAYER")


def _load_parent():
    with tarfile.open(PARENT_TAR, "r:gz") as archive:
        members = [m for m in archive.getmembers()
                   if m.isfile() and Path(m.name).name == "estimator.py"]
        if len(members) != 1:
            raise RuntimeError(f"expected one estimator.py tar member, found {len(members)}")
        stream = archive.extractfile(members[0])
        if stream is None:
            raise RuntimeError("cannot read parent estimator.py member")
        source = stream.read()
    got = hashlib.sha256(source).hexdigest()
    if got != PARENT_SHA:
        raise RuntimeError(f"L4 parent SHA mismatch: {got}")
    module = types.ModuleType("d21_l4_parent")
    module.__file__ = f"{PARENT_TAR}!estimator.py"
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module, got


def _load_candidate():
    source = CANDIDATE.read_bytes()
    sha = hashlib.sha256(source).hexdigest()
    spec = importlib.util.spec_from_file_location("d21_l4_f21", CANDIDATE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import candidate")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, sha


def _numeric(name, ref, new, np):
    if ref is None or new is None:
        ok = ref is None and new is None
        return {"field": name, "shape": None, "dtype_ref": None,
                "dtype_new": None, "max_normalized_entry_error": 0.0 if ok else None,
                "finite": ok, "pass": ok}
    r = np.asarray(ref)
    n = np.asarray(new)
    shape_ok = r.shape == n.shape
    dtype_ok = r.dtype == n.dtype
    finite = bool(np.isfinite(r).all() and np.isfinite(n).all())
    error = None
    if shape_ok and finite:
        rf = r.astype(np.float64)
        nf = n.astype(np.float64)
        error = float(np.max(np.abs(nf - rf) / np.maximum(1.0, np.abs(rf))))
    return {"field": name, "shape": list(r.shape), "dtype_ref": str(r.dtype),
            "dtype_new": str(n.dtype), "max_normalized_entry_error": error,
            "finite": finite, "pass": bool(shape_ok and dtype_ok and finite
                                           and error is not None and error <= ENTRY_TOL)}


def _tower_fields(layer, ref, new, np):
    if ref is None or new is None:
        ok = ref is None and new is None
        return [{"field": f"layer{layer}.tower", "pass": ok,
                 "both_none": ok}]
    rows = []
    rows.append({"field": f"layer{layer}.tower.k", "ref": ref.k, "new": new.k,
                 "pass": type(ref.k) is type(new.k) and ref.k == new.k})
    for field in ("A_D", "A_L", "M", "w2b"):
        rows.append(_numeric(f"layer{layer}.tower.{field}",
                             getattr(ref, field), getattr(new, field), np))
    return rows


def _state_fields(layer, ref, new, np):
    rows = [_numeric(f"layer{layer}.mu", ref[0], new[0], np),
            _numeric(f"layer{layer}.cov", ref[1], new[1], np)]
    rows.extend(_tower_fields(layer, ref[2], new[2], np))
    rows.append(_numeric(f"layer{layer}.c4", ref[3], new[3], np))
    return rows


def main(argv=None):
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from mint_provenance import canonical_mint

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1_000_921)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.width < 2 or args.depth < 2:
        ap.error("width and depth must both be at least 2")
    if HERE in Path(flops.__file__).resolve().parents:
        raise RuntimeError("local flopscope shim imported; official metering required")

    parent, parent_sha = _load_parent()
    candidate, candidate_sha = _load_candidate()
    config = {key: {"parent": getattr(parent, key),
                    "candidate": getattr(candidate, key)} for key in CONFIG_NAMES}
    config_ok = all(v["parent"] == v["candidate"] for v in config.values())
    chooser = {"parent": parent._fit_generations(args.width, args.depth, BUDGET),
               "candidate": candidate._fit_generations(args.width, args.depth, BUDGET)}
    chooser_ok = chooser["parent"] == chooser["candidate"]
    effective_depth = {"parent": parent._depth_for(args.width, args.width, args.width),
                       "candidate": candidate._depth_for(args.width, args.width, args.width)}
    backend_ok = (effective_depth["parent"] == effective_depth["candidate"]
                  and effective_depth["parent"] > 0)
    tables_ok = parent._tables() == candidate._tables()
    weights_np, estimator_seed, _sample_ss = canonical_mint(
        args.width, args.depth, args.seed)
    weights = [fnp.array(np.asarray(w, dtype=np.float32)) for w in weights_np]
    mlp = types.SimpleNamespace(width=args.width, depth=args.depth,
                                weights=weights, seed=estimator_seed)
    ctx = types.SimpleNamespace(width=args.width, depth=args.depth,
                                seed=estimator_seed)
    old_est, new_est = parent.Estimator(), candidate.Estimator()
    old_est.setup(ctx)
    new_est.setup(ctx)
    setup_ok = old_est._wick_program is not None and new_est._wick_program is not None

    rows = []
    if setup_ok:
        with flops.BudgetContext(flop_budget=10 * BUDGET, quiet=True):
            const_old = old_est._constants(args.width, fnp.float32)
            const_new = new_est._constants(args.width, fnp.float32)
            def initial():
                return (fnp.zeros(args.width, dtype=fnp.float32),
                        fnp.eye(args.width, dtype=fnp.float32), None, 0.0)

            state_old, state_new = initial(), initial()
            gens = chooser["parent"] if chooser["parent"] != -1 else 1
            for layer, weight in enumerate(weights):
                last = layer == args.depth - 1
                state_old = parent._layer(weight, state_old, parent._tables(),
                                          fnp.float32, const_old, 2.0, last, gens)
                state_new = candidate._layer(weight, state_new, candidate._tables(),
                                             fnp.float32, const_new, 2.0, last, gens)
                rows.extend(_state_fields(layer, state_old, state_new, np))
                # Match _tower's per-layer nonfinite sanitization before the
                # next layer; it is part of the working recurrence.
                mu_old = fnp.where(fnp.isfinite(state_old[0]), state_old[0], fnp.float32(0.0))
                mu_new = fnp.where(fnp.isfinite(state_new[0]), state_new[0], fnp.float32(0.0))
                state_old = (mu_old, state_old[1], state_old[2], state_old[3])
                state_new = (mu_new, state_new[1], state_new[2], state_new[3])

        with flops.BudgetContext(flop_budget=10 * BUDGET, quiet=True):
            out_old = old_est.predict(mlp, BUDGET)
        with flops.BudgetContext(flop_budget=10 * BUDGET, quiet=True):
            out_new = new_est.predict(mlp, BUDGET)
        rows.append(_numeric("predict.output", out_old, out_new, np))
        a = np.asarray(out_old, dtype=np.float64)
        b = np.asarray(out_new, dtype=np.float64)
        if a.shape == b.shape == (args.depth, args.width):
            final_rms = float(math.sqrt(np.mean((b[-1] - a[-1]) ** 2)
                                        / max(np.mean(a[-1] ** 2), 1e-300)))
            final_rms_ok = bool(math.isfinite(final_rms) and final_rms <= FINAL_RMS_TOL)
        else:
            final_rms = None
            final_rms_ok = False
        last_run_ok = old_est.last_run == new_est.last_run
    else:
        final_rms = None
        final_rms_ok = last_run_ok = False

    passed = bool(config_ok and chooser_ok and backend_ok and tables_ok and setup_ok
                  and last_run_ok and final_rms_ok and all(row["pass"] for row in rows))
    report = {
        "probe": "D21 L4 fusion float32 per-layer trajectory parity",
        "parent_tar": str(PARENT_TAR.relative_to(ROOT)), "parent_sha256": parent_sha,
        "candidate": str(CANDIDATE.relative_to(ROOT)), "candidate_sha256": candidate_sha,
        "flopscope_file": str(Path(flops.__file__).resolve()),
        "width": args.width, "depth": args.depth, "fixed_input_seed": args.seed,
        "estimator_seed": estimator_seed, "no_labels_or_decision_block": True,
        "entry_tolerance": ENTRY_TOL, "final_relative_rms_tolerance": FINAL_RMS_TOL,
        "config": config, "config_equal": config_ok, "table_equal": tables_ok,
        "generation_chooser": chooser, "generation_chooser_equal": chooser_ok,
        "effective_strassen_depth": effective_depth, "backend_equal_and_active": backend_ok,
        "setup_ok": setup_ok, "last_run_equal": last_run_ok,
        "final_relative_rms": final_rms, "final_relative_rms_pass": final_rms_ok,
        "fields": rows, "pass": passed,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "fields"}, indent=2))
    if args.out:
        path = Path(args.out)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
