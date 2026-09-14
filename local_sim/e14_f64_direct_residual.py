"""Installed WhestBench subprocess residual on canonical f64-readonly weights.

Targets are known zeros. No ground truth is minted or read; scores are invalid.
Run target shape only through workerq after the logged optimization review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

# The script directory contains local_sim stubs named whestbench/flopscope.
# Import the installed distributions first, while keeping lab modules available.
HERE = str(Path(__file__).resolve().parent)
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.append(HERE)

import numpy as np
import flopscope.numpy as fnp
from whestbench.cli import _RunnerEstimator
from whestbench.domain import MLP
from whestbench.runner import EstimatorEntrypoint, ResourceLimits, SubprocessRunner
from whestbench.scoring import ContestData, ContestSpec, evaluate_estimator
from whestbench.sdk import SetupContext

import p2_flop_anchor as anchor
from lifecycle_eval import build_weights
from mint_provenance import canonical_mint

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / 'estimator_e14_tree_f64.py'
OUTDIR = ROOT / 'experiments/e14_tree_f64'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    width, depth = (3, 4) if args.smoke else (anchor.WIDTH, anchor.DEPTH)
    if args.smoke:
        raw, estimator_seed, _ = canonical_mint(width, depth, anchor.PROBE_SEED)
    else:
        raw, estimator_seed = anchor._mint_weights()
    source_paths = (CANDIDATE, Path(__file__),
                    ROOT / 'local_sim/p2_flop_anchor.py',
                    ROOT / 'local_sim/lifecycle_eval.py',
                    ROOT / 'local_sim/mint_provenance.py')
    before = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p)
              for p in source_paths}
    weights = build_weights(raw, 'f64_readonly', fnp)
    raw_sha = hashlib.sha256(np.stack(raw).astype(np.float32).tobytes()).hexdigest()
    materialized = np.stack([np.asarray(W) for W in weights])
    weight_sha = hashlib.sha256(materialized.tobytes()).hexdigest()
    if materialized.dtype != np.float64 or materialized.shape != (depth, width, width):
        raise RuntimeError('supplied weights are not f64 target shape')
    if not np.array_equal(materialized.astype(np.float32), np.stack(raw)):
        raise RuntimeError('f64 conversion changed canonical values')
    mlp = MLP(width=width, depth=depth, weights=weights,
              seed=int(estimator_seed), name='E14-F64-DIRECT-RESIDUAL-DUMMY')
    mlp.validate()
    zeros = fnp.zeros((depth, width), dtype=fnp.float32)
    spec = ContestSpec(width=width, depth=depth, n_mlps=1,
                       flop_budget=anchor.BUDGET, ground_truth_samples=1,
                       setup_timeout_s=5.0, predict_timeout_s=660.0,
                       memory_limit_mb=8192, wall_time_limit_s=600.0,
                       residual_wall_time_limit_s=0.4, seed=None,
                       lambda_flops_per_second=0.0)
    spec.validate()
    data = ContestData(spec=spec, mlps=[mlp],
                       all_layer_targets=[zeros],
                       final_targets=[zeros[-1]], avg_variances=[0.0],
                       sampling_budget_breakdown=None)
    context = SetupContext(width=width, depth=depth,
                           flop_budget=anchor.BUDGET, api_version='1.0',
                           submission_dir=str(ROOT), seed=0)
    limits = ResourceLimits(setup_timeout_s=5.0, predict_timeout_s=660.0,
                            memory_limit_mb=8192, flop_budget=anchor.BUDGET,
                            wall_time_limit_s=600.0,
                            residual_wall_time_limit_s=0.4)
    runner = SubprocessRunner()
    entrypoint = EstimatorEntrypoint(file_path=CANDIDATE, class_name='Estimator')
    try:
        runner.start(entrypoint, context, limits)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            results = evaluate_estimator(_RunnerEstimator(runner), data)
        runner_stats = runner.last_predict_stats()
    finally:
        runner.close()
    after = {key: sha(ROOT / key) for key in before}
    if before != after:
        raise RuntimeError('source set changed during direct residual run')
    row = results['per_mlp'][0]
    if row.get('flops_used') is None or row.get('residual_wall_time_s') is None:
        raise RuntimeError('no executed residual row')
    stem = 'direct_residual_smoke' if args.smoke else 'direct_residual'
    OUTDIR.mkdir(parents=True, exist_ok=True)
    report = {
        'scope': 'installed WhestBench SubprocessRunner + CLI _RunnerEstimator '
                 'adapter, one canonical f64_readonly MLP, zero dummy targets; '
                 'local host only; diagnostic 600s wall and binding 0.4s residual',
        'do_not_use_scores': True,
        'source_sha256': before, 'raw_weights_sha256_f32': raw_sha,
        'supplied_weights_sha256_f64': weight_sha,
        'supplied_weight_dtype': str(materialized.dtype),
        'supplied_weight_writeable': [bool(np.asarray(W).flags.writeable)
                                      for W in weights],
        'width': width, 'depth': depth, 'estimator_seed': int(estimator_seed),
        'limits': {'flop_budget': anchor.BUDGET, 'memory_limit_mb': 8192,
                   'wall_time_limit_s': 600.0,
                   'residual_wall_time_limit_s': 0.4},
        'row': row, 'n_failed_mlps': results['n_failed_mlps'],
        'runner_stats': runner_stats,
        'warnings': [str(w.message) for w in caught],
        'evaluator': 'whestbench.scoring.evaluate_estimator / '
                     'whestbench.runner.SubprocessRunner',
    }
    path = OUTDIR / f'{stem}.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False, default=str),
                    encoding='utf-8')
    print(json.dumps({'out': str(path), 'flops_used': row['flops_used'],
                      'residual_wall_time_s': row['residual_wall_time_s'],
                      'residual_wall_time_exhausted': row['residual_wall_time_exhausted'],
                      'n_failed_mlps': results['n_failed_mlps']}))


if __name__ == '__main__':
    main()
