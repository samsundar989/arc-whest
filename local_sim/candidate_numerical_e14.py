"""Candidate-specific early/middle/late real-backend numerical checks.

Uses the existing atomic case worker and receipt verifier. Runs no labels.
Heavy target-branch runs belong in workerq. Each stdout is retained in full;
older certificates retained pass flags but discarded measured drift.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != HERE] + [str(HERE)]
import certify_candidate as cert


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--sha256', required=True)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    candidate = Path(a.candidate)
    if cert._sha(candidate) != a.sha256:
        ap.error('candidate source hash mismatch')
    os.environ['ARC_WHEST_RAW_MSE'] = '2e-8'
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=False)
    results = []
    for case in (0, 1, 3):
        receipt = out / ('permutation_%d.json' % case)
        command = [sys.executable, str(HERE / 'edge_cases.py'), str(candidate),
                   '--phase2', '--real-flopscope', '--weights-mode', 'f64_readonly',
                   '--case', 'permutation:%d' % case, '--receipt', str(receipt)]
        run = subprocess.run(command, capture_output=True, text=True, errors='replace')
        (out / ('permutation_%d.log' % case)).write_text(run.stdout + '\n' + run.stderr, encoding='utf-8')
        payload = json.loads(receipt.read_text()) if receipt.exists() else None
        errors = cert.validate_edge_case_receipt(
            payload, case='permutation:%d' % case, group='permutation', case_index=case,
            candidate=str(candidate).replace('\\', '/'), candidate_sha256=a.sha256,
            edge_cases_sha256=cert._sha(HERE / 'edge_cases.py'), weights_mode='f64_readonly') if payload else ['missing receipt']
        results.append({'case': case, 'returncode': run.returncode, 'receipt_path': str(receipt),
                        'receipt_sha256': cert._sha(receipt) if payload else None,
                        'receipt_errors': errors,
                        'declared_bar': 2e-10, 'raw_mse_scale': 2e-8,
                        'source_unchanged': cert._sha(candidate) == a.sha256,
                        'receipt': payload})
        (out / ('case_%d_result.json' % case)).write_text(json.dumps(results[-1], indent=2), encoding='utf-8')
    passed = all(r['returncode'] == 0 and not r['receipt_errors'] and r['source_unchanged'] and r['receipt']
                 and r['receipt']['details']['denom'] == 2e-8
                 and r['receipt']['results'] == [True] for r in results)
    summary = {'candidate': str(candidate), 'candidate_sha256': a.sha256,
               'instrument_sha256': cert._sha(__file__), 'raw_mse_scale': 2e-8,
               'bar': 2e-10, 'cases': results, 'passed': passed,
               'scope': 'Target-branch numerical checks only; no release or accuracy promotion'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
