"""Run a label-free local official subprocess residual probe.

The baked dataset contains canonical unprotected weights and explicit ZERO
dummy targets. Its score is meaningless and must never be used as accuracy.
Run this target-shape instrument only through workerq.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import gc
from pathlib import Path
import subprocess
import sys
import tempfile

# Direct script execution puts local_sim's simulator packages ahead of the
# installed official stack. Keep local modules importable only after site-packages.
HERE = str(Path(__file__).resolve().parent)
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.append(HERE)

import numpy as np
from datasets import Dataset
from whestbench.dataset_io import (
    SCHEMA_FORMAT, SCHEMA_VERSION, make_features,
)
from whestbench.dataset import load_dataset, mlp_at
from whestbench.seeds import build_seed_protocol_metadata

import p2_flop_anchor as anchor

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / 'experiments/e14_tree_f64'
CANDIDATE = ROOT / 'estimator_e14_tree_f64.py'
SPLIT = 'probe'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='exercise only a tiny n=3, depth=4 local dataset')
    args = parser.parse_args()
    source_paths = [CANDIDATE, Path(__file__),
                    ROOT / 'local_sim/p2_flop_anchor.py',
                    ROOT / 'local_sim/mint_provenance.py']
    before = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p)
              for p in source_paths}
    width, depth = (3, 4) if args.smoke else (anchor.WIDTH, anchor.DEPTH)
    if args.smoke:
        from mint_provenance import canonical_mint
        raw_weights, estimator_seed, _ = canonical_mint(
            width, depth, anchor.PROBE_SEED)
    else:
        raw_weights, estimator_seed = anchor._mint_weights()
    weights = np.stack(raw_weights).astype(np.float32, copy=False)
    input_sha = hashlib.sha256(weights.tobytes(order='C')).hexdigest()
    zeros = np.zeros((depth, width), dtype=np.float32)
    ds = Dataset.from_dict({
        'mlp_id': [0],
        'mlp_name': ['E14-LABEL-FREE-RESIDUAL-PROBE'],
        'mlp_seed': [int(anchor.PROBE_SEED)],
        'weights': weights[None, ...],
        'all_layer_means': zeros[None, ...],
        'final_means': zeros[-1][None, ...],
        'avg_variance': [0.0],
        'sampling_budget_breakdown': [json.dumps({
            'flopscope_overhead_time_s': 0.0,
            'diagnostic_dummy_sampling': True,
        })],
    }, features=make_features(width=width, depth=depth))
    metadata = {
        'schema_version': SCHEMA_VERSION, 'format': SCHEMA_FORMAT,
        'backend': 'flopscope',
        'seed_protocol': build_seed_protocol_metadata(seed_protocol_version='3.0'),
        'split': SPLIT, 'config': 'default', 'n_mlps': 1, 'n_samples': 1,
        'width': width, 'depth': depth,
        'diagnostic_dummy_targets': True,
        'diagnostic_notice': 'Every target is zero; no ground truth was minted or read.',
    }
    dataset_dir = Path(tempfile.mkdtemp(prefix='arcwhest-e14-f64-residual-'))
    baked = dataset_dir / 'dataset'
    (baked / 'data').mkdir(parents=True)
    ds.to_parquet(str(baked / 'data' / f'{SPLIT}-00000-of-00001.parquet'))
    (baked / 'metadata.json').write_text(json.dumps(metadata, indent=2),
                                         encoding='utf-8')
    (baked / 'README.md').write_text(
        '# Label-free residual probe\n\n'
        'All target arrays are explicit zeros. Scores are meaningless; only '
        'the official subprocess residual and failure fields may be read.\n',
        encoding='utf-8')
    loaded = load_dataset(str(baked), split=SPLIT)
    mlp = mlp_at(loaded, 0)
    loaded_weights = np.stack([np.asarray(W) for W in mlp.weights])
    loaded_sha = hashlib.sha256(
        loaded_weights.astype(np.float32).tobytes(order='C')).hexdigest()
    loaded_dtype = str(loaded_weights.dtype)
    if loaded_sha != input_sha or loaded_weights.shape != weights.shape:
        raise RuntimeError('local dataset changed canonical weights')
    if int(mlp.seed) != int(estimator_seed):
        raise RuntimeError('local dataset changed canonical estimator seed')
    if not all(np.all(np.asarray(row) == 0) for row in
               (loaded[0]['all_layer_means'], loaded[0]['final_means'])):
        raise RuntimeError('dummy targets were changed')
    del ds, loaded, mlp, loaded_weights, weights, raw_weights
    gc.collect()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-c',
           'from whestbench.cli import main; raise SystemExit(main())',
           'run', '--estimator', str(CANDIDATE), '--dataset', str(baked),
           '--split', SPLIT, '--n-mlps', '1', '--runner', 'subprocess',
           '--format', 'json', '--wall-time-limit', '600',
           '--residual-wall-time-limit', '0.4', '--max-threads', '2']
    if args.smoke:
        cmd.append('--debug')
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    run = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                         text=True, encoding='utf-8', errors='replace',
                         timeout=1200)
    stem = 'residual_probe_smoke' if args.smoke else 'residual_probe'
    (OUTDIR / f'{stem}_stdout.txt').write_text(run.stdout, encoding='utf-8')
    (OUTDIR / f'{stem}_stderr.txt').write_text(run.stderr, encoding='utf-8')
    after = {key: sha(ROOT / key) for key in before}
    if before != after:
        raise RuntimeError('source set changed during residual probe')
    parquet = next((baked / 'data').glob('*.parquet'))
    result = {
        'scope': 'one canonical unprotected MLP; all-zero dummy targets; '
                 'official installed whest runner=subprocess; local host only',
        'do_not_use_score': True,
        'source_sha256': before,
        'dataset_dir': str(baked),
        'metadata_sha256': sha(baked / 'metadata.json'),
        'parquet_sha256': sha(parquet),
        'weights_sha256_f32': input_sha,
        'loaded_weights_sha256_f32': loaded_sha,
        'loaded_weight_dtype': loaded_dtype,
        'estimator_seed_from_canonical_mint': int(estimator_seed),
        'command': cmd, 'returncode': run.returncode,
        'stdout_path': f'experiments/e14_tree_f64/{stem}_stdout.txt',
        'stderr_path': f'experiments/e14_tree_f64/{stem}_stderr.txt',
        'report_parsed': False,
    }
    try:
        report = json.loads(run.stdout)
        result['report_parsed'] = True
        result['run_config'] = report.get('run_config')
        result['per_mlp'] = report.get('results', {}).get('per_mlp')
        result['n_failed_mlps'] = report.get('results', {}).get('n_failed_mlps')
    except json.JSONDecodeError:
        pass
    (OUTDIR / f'{stem}.json').write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'returncode': run.returncode,
                      'report_parsed': result['report_parsed'],
                      'n_failed_mlps': result['n_failed_mlps'] if
                      'n_failed_mlps' in result else None,
                      'weights_match': input_sha == loaded_sha,
                      'out': str(OUTDIR / f'{stem}.json')}))


if __name__ == '__main__':
    main()
