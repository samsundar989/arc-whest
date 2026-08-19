<#
.SYNOPSIS
  The gate that runs on every estimator change. Cheap, unconditional, ~2 minutes.

.DESCRIPTION
  Implements steps 1-3 of docs/VALIDATION.md section 7, plus a ship-file identity
  check that previously existed nowhere:

    1. mint provenance    - our MLPs are still bit-identical to the grader's
    2. invariant suite    - 30/30 on the SHIPPED file, not on a candidate
    3. whest validate     - the grader contract
    4. ship-file identity - estimator.py matches its version archive
    5. vault seal         - touch counter still agrees with the log
    6. seed ledger        - consumption is recorded, and no source file draws
                            seeds the ledger has never heard of

  What this does NOT do, because each costs something real:
    - the public mini run (CONFIRM tier, read-only, minutes)
    - a vault touch (one per estimator change, ever)
    - anything on the GPU venv
  Those stay explicit. See docs/VALIDATION.md section 7.

.PARAMETER Archive
  Version archive to compare estimator.py against. Default: the current shipped
  version. Update this default when a new version is promoted.

.EXAMPLE
  .\scripts\preflight.ps1
  .\scripts\preflight.ps1 -Archive estimator_v2_5r.py
#>
[CmdletBinding()]
param(
    [string]$Archive = "estimator_v2_6_1.py"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$failures = New-Object System.Collections.ArrayList

function Step {
    param([string]$Name, [scriptblock]$Body)
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    # Exit code is the only verdict. Windows PowerShell 5.1 wraps a native
    # command's stderr in an ErrorRecord, so with ErrorActionPreference=Stop a
    # benign warning on stderr fails the step -- which is exactly what happened
    # to `whest validate`'s SymmetryLossWarning on the first run of this script.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $global:LASTEXITCODE = 0
    try {
        & $Body
        $code = $LASTEXITCODE
        if ($code -ne 0) { throw "exit code $code" }
        Write-Host "  PASS  $Name" -ForegroundColor Green
    } catch {
        Write-Host "  FAIL  $Name -- $_" -ForegroundColor Red
        [void]$failures.Add($Name)
    } finally {
        $ErrorActionPreference = $prev
    }
}

Write-Host "preflight -- arc-whest" -ForegroundColor White
Write-Host "repo:    $repo"
Write-Host "shipped: estimator.py   archive: $Archive"

# --- 1. provenance: are our mints still the grader's mints? -------------------
Step "1. mint provenance" { uv run python local_sim/mint_provenance.py }

# --- 2. invariants on the SHIPPED file ----------------------------------------
Step "2. invariant suite (30/30)" { uv run python local_sim/edge_cases.py }

# --- 3. grader contract --------------------------------------------------------
Step "3. whest validate" { uv run whest validate --estimator estimator.py }

# --- 4. ship-file identity -----------------------------------------------------
# estimator.py must be byte-identical to its version archive modulo line endings.
# Nothing else in the repo asserts this, and the archive is what every graded
# number is attributed to.
Step "4. ship-file identity vs $Archive" {
    $py = @"
import hashlib, sys, pathlib
ship = pathlib.Path('estimator.py').read_bytes().replace(b'\r\n', b'\n')
arch = pathlib.Path(r'$Archive')
if not arch.exists():
    print('archive missing:', arch); sys.exit(1)
a = arch.read_bytes().replace(b'\r\n', b'\n')
print('  estimator.py  sha256(lf) =', hashlib.sha256(ship).hexdigest()[:32])
print('  %-13s sha256(lf) =' % arch.name, hashlib.sha256(a).hexdigest()[:32])
if ship != a:
    print('  MISMATCH: the shipped file is not the archived version.')
    print('  Either promote the candidate to a new archive, or revert estimator.py.')
    sys.exit(1)
print('  identical')
"@
    $py | uv run python -
}

# --- 5. vault seal -------------------------------------------------------------
# Counter must equal the number of touch-log lines. A quiet extra evaluation is
# exactly what the seal exists to make impossible.
Step "5. vault seal (counter vs log)" {
    $py = @"
import json, pathlib, sys
man = pathlib.Path('experiments/vault/manifest.json')
log = pathlib.Path('experiments/vault/touch_log.jsonl')
if not man.exists():
    print('  no vault manifest -- skipping'); sys.exit(0)
m = json.loads(man.read_text())
lines = [l for l in log.read_text().splitlines() if l.strip()] if log.exists() else []
counter = m.get('touches', m.get('touch_count'))
print('  manifest touches =', counter, ' log entries =', len(lines))
if counter is None:
    print('  manifest has no touch counter -- inspect vault.py'); sys.exit(1)
if counter != len(lines):
    print('  MISMATCH: the vault was evaluated without being logged, or vice versa.')
    sys.exit(1)
print('  seal consistent')
"@
    $py | uv run python -
}

# --- 6. seed ledger ------------------------------------------------------------
# check_disjoint() (invariant 12) proves no two blocks ALLOCATE the same seeds.
# It cannot see a probe with a seed hardcoded into it, which is how D16 was drawn
# 270 seeds dry while seed_blocks.py called it "Reusable" -- and how two probes
# ended up measuring two "independent" kills on the same 24 networks. This step
# re-validates the consumption ledger and statically scans the tree for seed
# literals and seeds() calls that the ledger does not record.
Step "6. seed ledger (consumption + source scan)" {
    uv run python local_sim/seed_ledger.py --check --scan
}

Write-Host ""
if ($failures.Count -eq 0) {
    Write-Host "PREFLIGHT GREEN -- 6/6" -ForegroundColor Green
    Write-Host ""
    Write-Host "Not covered by this gate, and still required before promotion:" -ForegroundColor Yellow
    Write-Host "  - public mini run (CONFIRM, read-only):  uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 --split mini --runner subprocess"
    Write-Host "  - GPU mirror parity:                     .venv-gpu\Scripts\python.exe local_sim/gpu_parity.py"
    Write-Host "  - vault touch (ONE per estimator change): .venv-gpu\Scripts\python.exe local_sim/vault.py evaluate --label <v> --reason estimator-change"
    Write-Host "  - Samuel's explicit approval before any submission."
    exit 0
} else {
    Write-Host "PREFLIGHT RED -- failed: $($failures -join ', ')" -ForegroundColor Red
    Write-Host "Do not promote, do not submit, do not spend a decision block." -ForegroundColor Red
    exit 1
}
