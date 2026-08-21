<#
.SYNOPSIS
  The gate that runs on every estimator change. Cheap, unconditional, ~2 minutes.

.DESCRIPTION
  Implements steps 1-3 of docs/VALIDATION.md section 7, plus a ship-file identity
  check that previously existed nowhere, plus the regression suites that used to
  pass ungated:

  DEFAULT TIER (10 of 12 steps, ~2-3 minutes) - runs on every change:
    1. mint provenance    - our MLPs are still bit-identical to the grader's
    2. invariant suite    - 30/30 on the SHIPPED file, not on a candidate
    3. whest validate     - the grader contract
    4. ship-file identity - estimator.py matches its version archive
    5. vault seal         - touch counter still agrees with the log
    6. seed ledger        - consumption is recorded, and no source file draws
                            seeds the ledger has never heard of
    7. canonical scorer   - 71/71, the competition identity is still bitwise
                            equal to the installed whestbench
    8. seed-ledger refusals - 44/44 negative tests: every guard still FIRES
    9. vault integrity    - array sha256s vs the manifest and the seed->weights
                            map, not merely the counter. Costs no touch.
   10. lifecycle evaluator - 37/37. The canonical grader lifecycle (one Estimator,
                            setup() once OFF-budget and TIMED), all six failure
                            paths PROVOKED not asserted, setup failure treated as
                            a run-level refusal the way the grader treats it, and
                            the immutable evidence rows every comparison uses.

  -Full TIER (adds 2 steps, ~10 minutes) - REQUIRED before promotion, submission
  or a vault touch:
   11. shape parity       - 27/27. The fallback path is v2.4.1 BIT FOR BIT. This
                            is the check that would have caught v2.6's odd-width
                            defect, which produced finite, plausible, wrong
                            numbers on 7 of 17 shapes.
   12. carrier strength   - 27/27 design-strength invariants of local_sim/carriers.py

  The split is by COST, not by importance. Steps 10-11 are ~594 s of the total
  and cannot fire unless estimator.py or carriers.py moved; a 10-minute
  unconditional gate is a gate people learn to skip.

  ACCOUNTING: the denominator is always TWELVE, and every step that did not run
  is named in a red NOT-CHECKED block. A default run therefore reports
  "GREEN 10/12, 2 SKIPPED", never "10/10". A check that could not run is not a
  pass, and a gate that says otherwise is worse than no gate -- this repository
  has lost four separate checks to assertions that reported a state rather than
  testing a property (docs/GOTCHAS.md).

  What this still does NOT do, because each costs something real:
    - the public mini run (CONFIRM tier, read-only, minutes)
    - a vault touch (one per estimator change, ever)
    - anything on the GPU venv
  Those stay explicit. See docs/VALIDATION.md section 7.

.PARAMETER Archive
  Version archive to compare estimator.py against. Default: the current shipped
  version. Update this default when a new version is promoted.

.PARAMETER Full
  Also run shape parity and carrier strength. Mandatory before promotion,
  submission, or spending a vault touch.

.EXAMPLE
  .\scripts\preflight.ps1
  .\scripts\preflight.ps1 -Full
  .\scripts\preflight.ps1 -Archive estimator_v2_5r.py
#>
[CmdletBinding()]
param(
    [string]$Archive = "estimator_v2_6_1.py",
    [switch]$Full
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$failures = New-Object System.Collections.ArrayList
$skipped  = New-Object System.Collections.ArrayList

function Step {
    param([string]$Name, [scriptblock]$Body)
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    # Exit code is the only verdict. Windows PowerShell 5.1 wraps a native
    # command's stderr in an ErrorRecord, so with ErrorActionPreference=Stop a
    # benign warning on stderr fails the step -- which is exactly what happened
    # to `whest validate`'s SymmetryLossWarning on the first run of this script.
    #
    # A step may report that it did not run, by setting $script:StepSkipped to a
    # reason string. That is NOT a pass: it is counted separately and printed in
    # the red NOT-CHECKED notice. An unrunnable check that reports green is the
    # precise failure this gate exists to prevent.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $global:LASTEXITCODE = 0
    $script:StepSkipped = $null
    try {
        & $Body
        $code = $LASTEXITCODE
        if ($code -ne 0) { throw "exit code $code" }
        if ($script:StepSkipped) {
            Write-Host "  SKIP  $Name -- $($script:StepSkipped)" -ForegroundColor Yellow
            [void]$skipped.Add("$Name -- $($script:StepSkipped)")
        } else {
            Write-Host "  PASS  $Name" -ForegroundColor Green
        }
    } catch {
        Write-Host "  FAIL  $Name -- $_" -ForegroundColor Red
        [void]$failures.Add($Name)
    } finally {
        $ErrorActionPreference = $prev
        $script:StepSkipped = $null
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
    if (-not (Test-Path "experiments/vault/manifest.json")) {
        Write-Host "  experiments/vault/manifest.json absent -- no vault to seal-check"
        $script:StepSkipped = "no vault manifest"
        return
    }
    $py = @"
import json, pathlib, sys
man = pathlib.Path('experiments/vault/manifest.json')
log = pathlib.Path('experiments/vault/touch_log.jsonl')
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

# --- 7. the canonical competition scorer ---------------------------------------
# local_sim/competition_score.py is the ONE place the grader's identity lives, and
# every gate downstream reads its numbers. 71/71 asserts it is still bitwise equal
# to the installed whestbench on the scalar primitives, on the aggregate, and on
# the real scoring loop INCLUDING both failure paths -- and carries the graded
# fixture where the arithmetic and geometric estimands pick opposite winners.
Step "7. canonical scorer (71/71)" {
    uv run python local_sim/test_competition_score.py
}

# --- 8. the seed ledger's refusals actually fire --------------------------------
# Step 6 proves the ledger is CONSISTENT. It cannot prove the guards still refuse.
# These are 44 negative tests: unregistered block, range leaving its block, any
# overlap in a DECISION block, a second claim on a SINGLE_USE one, a hand-edited
# ledger, an artifact rewritten after its claim.
Step "8. seed-ledger refusals (44/44)" {
    uv run python local_sim/test_seed_ledger.py
}

# --- 9. vault integrity, not merely the counter ---------------------------------
# Step 5 compares a number to a line count. This re-hashes every stored array
# against the manifest and reconstructs networks from their seeds to confirm the
# seed->weights map still holds. It reads no candidate predictions, so it COSTS NO
# TOUCH. The arrays are gitignored (33 MB), so a fresh clone legitimately has none
# and must not go red for it -- step 5 still covers the seal there.
Step "9. vault integrity (array hashes + seed->weights)" {
    if (-not (Test-Path "experiments/vault/vault_v1.npz")) {
        Write-Host "  experiments/vault/vault_v1.npz absent (gitignored)."
        Write-Host "  Run 'uv run python local_sim/vault.py mint' to restore it."
        $script:StepSkipped = "vault arrays absent (gitignored); step 5 still checked the seal"
        return
    }
    uv run python local_sim/vault.py verify
}

# --- 10. the canonical grader-lifecycle evaluator -------------------------------
# vault.py:282 and coverage_lab.py:226 build a fresh Estimator per MLP and never
# call setup(); the grader builds ONE and calls setup() once, OFF-budget. This
# suite pins that lifecycle, the six failure paths, and the immutable evidence
# rows -- the object every future candidate comparison is scored through.
Step "10. lifecycle evaluator (37/37)" {
    uv run python local_sim/lifecycle_eval.py --self-test
}

# --- 11-12. -Full only ----------------------------------------------------------
if ($Full) {
    # Shape parity: on any Strassen-INELIGIBLE shape the estimator falls back to
    # the dense chain, which is v2.4.1's algorithm exactly, and must reproduce it
    # BIT FOR BIT. This is the check that would have caught v2.6 deleting the whole
    # first layer's sampled path on every odd width -- 7 of 17 shapes wrong, 0.58
    # max relative error, and every output finite and plausible.
    Step "11. shape parity (27/27)" {
        uv run python local_sim/test_shape_parity.py
    }
    # Carrier design strength: the moment-error invariants that the whole design
    # lane's arithmetic rests on.
    Step "12. carrier strength (27/27)" {
        uv run python local_sim/test_carriers.py
    }
} else {
    [void]$skipped.Add("11. shape parity (27/27)      -- not run without -Full (479 s)")
    [void]$skipped.Add("12. carrier strength (27/27)  -- not run without -Full (104 s)")
}

# There are TWELVE checks in this gate. The default tier runs ten of them and
# skips two; a missing vault array skips a third. Denominating against the tier
# that happened to run would let an incomplete run report "10/10", which is the
# precise dishonesty this accounting exists to prevent -- so the denominator is
# always 12 and every skip is named below.
$total  = 12
$ran    = $total - $skipped.Count
$verdict = if ($skipped.Count -eq 0) { "$total/$total" }
           else { "$ran/$total, $($skipped.Count) SKIPPED" }

Write-Host ""
if ($failures.Count -eq 0) {
    Write-Host "PREFLIGHT GREEN -- $verdict" -ForegroundColor Green
    if ($skipped.Count -gt 0) {
        Write-Host ""
        Write-Host "NOT CHECKED by this run. A skipped step is NOT a pass." -ForegroundColor Red
        Write-Host "Clear every line below before promoting, submitting, or" -ForegroundColor Red
        Write-Host "spending a vault touch:" -ForegroundColor Red
        foreach ($s in $skipped) { Write-Host "  - $s" -ForegroundColor Red }
        Write-Host "  .\scripts\preflight.ps1 -Full" -ForegroundColor Red
    }
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
