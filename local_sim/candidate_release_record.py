"""Join pinned existing evidence without granting release or server-byte identity.

Input is a recipe with explicit candidate/baseline identities and hash-pinned
artifact references. No truth arrays, candidate execution or network access.
This record is an index of evidence, never a substitute for certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("MATH", "LOCAL", "SERVER_PREDICTED", "SERVER_MEASURED")
FLAGS = ("budget_exhausted", "time_exhausted", "residual_wall_time_exhausted",
         "combined_budget_exhausted")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pinned(ref, root):
    path = (root / ref['path']).resolve()
    if not path.is_relative_to(root.resolve()) or path.suffix == '.npz':
        raise ValueError('only repository source/evidence paths allowed; no truth arrays')
    if sha(path) != ref['sha256']:
        raise ValueError('artifact hash mismatch: ' + ref['path'])
    return path


def numerical_passes(summary, identity, scale, root):
    """Reuse the canonical receipt consumer and bind every referenced byte."""
    import certify_candidate as cert
    old_scale = os.environ.get('ARC_WHEST_RAW_MSE')
    try:
        cases = summary['cases']
        if not (summary['candidate_sha256'] == identity
                and summary['instrument_sha256'] == sha(root / 'local_sim/candidate_numerical_e14.py')
                and summary['raw_mse_scale'] == scale
                and summary['bar'] == .01 * scale and summary['passed'] is True
                and len(cases) == 3 and {r['case'] for r in cases} == {0, 1, 3}):
            return False
        candidate = pinned({'path': summary['candidate'], 'sha256': identity}, root)
        candidate_name = str(candidate.relative_to(root)).replace('\\', '/')
        edge_sha = sha(root / 'local_sim/edge_cases.py')
        for row in cases:
            if not (type(row['case']) is int and type(row['returncode']) is int
                    and row['returncode'] == 0 and row['source_unchanged'] is True
                    and row['receipt_errors'] == [] and row['declared_bar'] == .01 * scale
                    and row['raw_mse_scale'] == scale):
                return False
            path = pinned({'path': row['receipt_path'], 'sha256': row['receipt_sha256']}, root)
            payload = json.loads(path.read_text(encoding='utf-8'))
            # Compare JSON types too: Python's True == 1 must not hide mutations.
            if json.dumps(payload, sort_keys=True) != json.dumps(row['receipt'], sort_keys=True):
                return False
            raw_env = payload['details']['raw_mse_env']
            if type(raw_env) is not str or float(raw_env) != scale:
                return False
            os.environ['ARC_WHEST_RAW_MSE'] = raw_env
            errors = cert.validate_edge_case_receipt(
                payload, case='permutation:%d' % row['case'], group='permutation',
                case_index=row['case'], candidate=candidate_name, candidate_sha256=identity,
                edge_cases_sha256=edge_sha, weights_mode='f64_readonly')
            if errors or payload['all_passed'] is not True:
                return False
        return True
    except (KeyError, TypeError, ValueError, OSError):
        return False
    finally:
        if old_scale is None:
            os.environ.pop('ARC_WHEST_RAW_MSE', None)
        else:
            os.environ['ARC_WHEST_RAW_MSE'] = old_scale


def build(recipe, root=ROOT):
    candidate = pinned(recipe['candidate'], root)
    pinned(recipe['baseline'], root)
    for key in ('recurrence', 'representation', 'backend'):
        if not recipe.get(key):
            raise ValueError('missing explicit recipe: ' + key)
    scale = recipe['numerical_raw_mse_scale']
    if not math.isfinite(scale) or not 0 < scale <= 1.4e-6:
        raise ValueError('invalid tighten-only numerical scale')
    refs = recipe['evidence']
    docs = {k: json.loads(pinned(v, root).read_text(encoding='utf-8'))
            for k, v in refs.items()}
    cert = docs['certificate']
    identity = recipe['candidate']['sha256']
    if cert['candidate_sha256'] != identity:
        raise ValueError('certificate candidate mismatch')
    domains = {k: dict(cert['checks_by_domain'][k]) for k in DOMAINS}
    domains['SERVER_MEASURED'] = {'verdict': 'NOT_RUN', 'exact_byte_binding': 'UNKNOWN'}
    receipt = docs.get('package')
    package = None
    if receipt:
        package_ref = {'path': receipt['package_path'], 'sha256': receipt['package_sha256']}
        archive = pinned(package_ref, root)
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            expected = {m['name']: m for m in receipt['members']}
            if len(members) != len(expected) or {m.name for m in members} != set(expected):
                raise ValueError('package member inventory mismatch')
            for member in members:
                data = tar.extractfile(member).read()
                digest = hashlib.sha256(data).hexdigest()
                if digest != expected[member.name]['sha256'] or len(data) != expected[member.name]['size']:
                    raise ValueError('package member hash/size mismatch')
                if member.name == 'estimator.py' and digest != identity:
                    raise ValueError('package candidate mismatch')
            if 'estimator.py' not in expected:
                raise ValueError('package missing estimator')
        package = package_ref
    mini = docs.get('local_mini')
    local_mini = None
    if mini:
        cfg, res = mini['run_config'], mini['results']
        capped = (cfg['residual_wall_time_limit_s'] == .4 and
                  cfg['wall_time_limit_s'] == 120 and cfg['lambda_flops_per_second'] == 0)
        rows = res['per_mlp']
        complete = len(rows) == cfg['n_mlps'] and len({r['mlp_index'] for r in rows}) == len(rows)
        failures = sum(any(r.get(f) is not False for f in FLAGS) or bool(r.get('traceback')) for r in rows)
        local_mini = {'capped': capped, 'complete': complete, 'failed_rows': failures,
                      'reported_failed_rows': res['n_failed_mlps'],
                      'source_binding': 'RECIPE_PINNED_REPORT; report itself lacks source hash',
                      'verdict': 'PASS' if capped and complete and failures == res['n_failed_mlps'] == 0 else 'REFUSED'}
        if local_mini['verdict'] != 'PASS':
            domains['LOCAL']['verdict'] = 'REFUSED'
    graded = docs.get('server_report')
    if graded:
        ev = graded['evaluation']; rows = ev['results']['per_mlp']
        if str(graded['id']) != str(recipe['server_submission_id']) or str(ev['submission_id']) != str(graded['id']):
            raise ValueError('server submission identity mismatch')
        expected = ev['config_snapshot']['n_mlps']
        complete = (graded['status'] == 'GRADED' and ev['state'] == 'completed' and
                    not ev['error'] and len(rows) == expected and
                    len({r['mlp_index'] for r in rows}) == expected)
        failures = sum(r['state'] != 'completed' or bool(r['error_code']) or
                       any(r['telemetry'].get(f) is not False for f in FLAGS) for r in rows)
        telemetry = [r['telemetry'] for r in rows]
        residual = max((r['residual_wall_time_s'] for r in telemetry), default=None)
        wall = max((r['participant_predict_wall_s'] for r in telemetry), default=None)
        valid_times = (bool(telemetry) and all(
            math.isfinite(r['residual_wall_time_s']) and 0 < r['residual_wall_time_s'] <= .4
            and math.isfinite(r['participant_predict_wall_s']) and 0 < r['participant_predict_wall_s'] <= 120
            and math.isfinite(r['flops_used']) and 0 <= r['flops_used'] <= 2**41
            for r in telemetry))
        domains['SERVER_MEASURED'] = {
            'verdict': 'PASS' if complete and not failures and valid_times else 'REFUSED',
            'submission_id': graded['id'], 'rows': len(rows), 'failed_rows': failures,
            'exact_byte_binding': 'PROVISIONAL',
            'binding_limit': 'Pinned local package and attributed report; no independently verified server archive hash.',
            'worst_residual_s': residual, 'residual_headroom': .4 / residual if valid_times else None,
            'worst_predict_wall_s': wall, 'wall_headroom': 120 / wall if valid_times else None,
            'counted_flops_values': sorted(set(r['flops_used'] for r in telemetry)),
            'runtime_environment': ev.get('runtime_environment'),
        }
    inv = cert['sections'].get('invariants', {})
    numerical = docs.get('numerical')
    numerical_ok = False
    if numerical:
        numerical_ok = numerical_passes(numerical, identity, scale, root)
    memory = docs.get('linux_memory')
    if memory and memory['candidate_sha256'] != identity:
        raise ValueError('memory candidate mismatch')
    return {'artifact': 'candidate_release_evidence_index', 'version': 1,
            'recipe': recipe, 'candidate_source_verified': str(candidate.relative_to(root)),
            'package_verified': package, 'domains': domains, 'local_mini': local_mini,
            'numerical_requirement': {'raw_mse_scale': scale, 'bar': .01 * scale,
                                      'historical_invariant_metadata': inv.get('permutation_bar'),
                                      'status': ('THREE_TARGET_PERMUTATION_CASES_PASS' if numerical_ok else
                                                 'REQUIRES_CANDIDATE_SPECIFIC_EVIDENCE_AT_DECLARED_SCALE'),
                                      'evidence': refs.get('numerical'),
                                      'scope': 'Only the declared target-shape permutation cases; not full numerical/contract certification'},
            'official_packages': cert['sections']['package_identity']['distribution_versions'],
            'counted_ledger': cert['sections'].get('grader_twin_arm'),
            'linux_memory': {'historical_consumer': cert['sections'].get('linux_memory_gate'),
                             'report_status': memory.get('status') if memory else 'NOT_RUN',
                             'scope': 'Stored evidence applicability; no new process cap test'},
            'normal_release_ready': False,
            'release_scope': 'Evidence index only; complete candidate-specific release checks and human approval still required.',
            'unknowns': ['fresh private accuracy', 'prize eligibility', 'exact server-byte attribution'],
            'protected_truth_read': False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recipe', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    result = build(json.loads(Path(args.recipe).read_text(encoding='utf-8')))
    with Path(args.out).open('x', encoding='utf-8') as out:
        json.dump(result, out, indent=2, allow_nan=False); out.write('\n')
    print(json.dumps({k: v['verdict'] for k, v in result['domains'].items()}))


if __name__ == '__main__':
    main()
