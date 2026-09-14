"""Queued exact-live-byte preflight with before/after instrument freeze proof.

Git checkout normalizes historical CRLF files and invalidates their evidence
hashes. This wrapper does not modify files; workerq still arbitrates resources.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    paths = list(ROOT.glob('*.py'))
    for directory in ('local_sim', 'scripts'):
        for suffix in ('*.py', '*.ps1'):
            paths.extend((ROOT / directory).rglob(suffix))
    paths.extend(ROOT / path for path in ('AGENTS.md', 'docs/VALIDATION.md'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=False)
    before = source_hashes()
    (out / 'sources_before.json').write_text(json.dumps(before, indent=2))
    command = ['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy',
               'Bypass', '-File', 'scripts/preflight.ps1', '-Log', str(out/'verdicts.txt')]
    result = subprocess.run(command, cwd=ROOT)
    after = source_hashes()
    (out / 'sources_after.json').write_text(json.dumps(after, indent=2))
    frozen = before == after
    (out / 'receipt.json').write_text(json.dumps({
        'returncode': result.returncode, 'source_frozen': frozen,
        'changed_sources': sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k)),
        'passed': result.returncode == 0 and frozen, 'command': command,
        'scope': 'Default12/16 infrastructure preflight; not candidate Full certification',
    }, indent=2))
    return 0 if result.returncode == 0 and frozen else 1


if __name__ == '__main__':
    raise SystemExit(main())
