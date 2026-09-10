<#
.SYNOPSIS
  The gate that runs on every estimator change. Cheap, unconditional, ~2 minutes.

.DESCRIPTION
  Implements steps 1-3 of docs/VALIDATION.md section 7, plus a ship-file identity
  check that previously existed nowhere, plus the regression suites that used to
  pass ungated:

  DEFAULT TIER (12 of 16 steps, ~3-4 minutes) - runs on every change:
    1. official stack     - the LIVE round, asserted field by field against the
                            INSTALLED whestbench. Added 2026-08-24 because this
                            gate ran ten green steps against whestbench 0.14 /
                            flopscope 0.10 while the competition graded on
                            0.16 / 0.12 -- and nothing in it looked.
    2. mint provenance    - our MLPs are still bit-identical to the grader's, at
                            the LIVE 1024x16 shape first, AND (WP0A, 2026-08-31)
                            local_sim/test_p2_tag_drift.py: the v2-phase2 pin
                            still describes the dataset those bytes came from,
                            and the loader still takes the route we compared.
                            Two commands, one step -- the denominator stays 16.
    3. invariant suite    - 30/30 on the SHIPPED file, not on a candidate,
                            AND (2026-09-03) invariant 9's mutation battery,
                            local_sim/test_edge_cases_inv9.py: nine deliberate
                            equivariance defects must still be REJECTED and two
                            EQUIVARIANT ones still accepted, after that
                            invariant's absolute float32 clause was replaced.
                            Two commands, one step -- the denominator stays 16.
    4. whest validate     - the grader contract, run with the child's text
                            encoding OWNED by this gate and ASSERTED, not
                            assumed. See the PYTHONIOENCODING block below.
    5. ship-file identity - estimator.py matches its version archive
    6. vault seal         - touch counter still agrees with the log
    7. seed ledger        - consumption is recorded, and no source file draws
                            seeds the ledger has never heard of, AND (WP0B,
                            2026-08-31) local_sim/test_enforcement_labels.py
                            --labels-only: every coverage count stated in THIS
                            docstring and in VALIDATION.md section 7 is checked
                            against the suite it names. Two commands, one step.
    8. canonical scorer   - 134/134, the competition identity is still bitwise
                            equal to the installed whestbench, AND decide() --
                            THE gate -- still refuses percentile/BCa, still
                            requires a pre-registered threshold, and still
                            follows arithmetic when the estimands disagree
    9. seed-ledger refusals - 66/66 negative tests: every guard still FIRES
   10. vault integrity    - array sha256s vs the manifest and the seed->weights
                            map, not merely the counter. Costs no touch.
   11. lifecycle evaluator - 56/56. The canonical grader lifecycle (one Estimator,
                            setup() once OFF-budget and TIMED), all six failure
                            paths PROVOKED not asserted, setup failure treated as
                            a run-level refusal the way the grader treats it, the
                            immutable evidence rows every comparison uses, and --
                            since 2026-08-24 -- a spec that NAMES its round and
                            records any deviation from it.
   12. counted-FLOP anchor - op by op against a real `whest run --profile --json`
                            report. THE check that says the local meter bills what
                            the grader bills. It REFUSES when the report is absent:
                            an absent comparison is not a pass.

  -Full TIER (adds 4 steps, ~11 minutes) - REQUIRED before promotion, submission
  or a vault touch:
   13. shape parity       - 27/27. The fallback path is v2.4.1 BIT FOR BIT. This
                            is the check that would have caught v2.6's odd-width
                            defect, which produced finite, plausible, wrong
                            numbers on 7 of 17 shapes.
   14. carrier strength   - 48/48 design-strength invariants of local_sim/carriers.py,
                            including the Kerdock fingerprint and the axial identity
   15. deployment envelope - what fails a Phase-2 MLP, AND the two constraints this
                            host cannot enforce at all. It prints those as NOT
                            PROVABLE rather than passing them. Since P1-PRE it also
                            runs local_sim/test_envelope_readonly.py, which proves
                            the suite's `--no-write` mode is read-only from OUTSIDE
                            it (child process, sha256, git) and, in the same run,
                            that a SKIPPED provocation never prints green.
   16. official Phase-2 smoke - DEPLOYMENT AUTHORITY. The grader's own subprocess
                            runner on the organisers' own v2-phase2 data, with
                            zero failures required and the live config asserted
                            from the report's own run_config.

  The split is by COST, not by importance. Steps 13-14 are ~594 s of the total
  and cannot fire unless estimator.py or carriers.py moved; a 10-minute
  unconditional gate is a gate people learn to skip.

  THE COUNTS ABOVE ARE UNDER TEST. `local_sim/test_enforcement_labels.py` parses
  every "N/N" in this docstring and in docs/VALIDATION.md section 7, runs the
  suite each one describes, and fails if they disagree. Three of them were wrong
  on 2026-08-31 -- steps 9 (44 vs 60), 11 (54 vs 56) and the step-8 comment
  (71 vs 98) -- and an audit had to find that by hand. Prose cannot enforce
  prose; that suite can. Update a count here and it is checked, not trusted.

  ACCOUNTING: the denominator is always SIXTEEN, and every step that did not run
  is named in a red NOT-CHECKED block. A default run therefore reports
  "GREEN 12/16, 4 SKIPPED", never "12/12". A check that could not run is not a
  pass, and a gate that says otherwise is worse than no gate -- this repository
  has lost four separate checks to assertions that reported a state rather than
  testing a property (docs/GOTCHAS.md).

  What this still does NOT do, because each costs something real:
    - the public Phase-2 mini run over all 100 rows (CONFIRM tier, read-only)
    - a vault touch (one per estimator change, ever)
    - anything on the GPU venv
    - the 8,192 MB memory limit, which is UNENFORCEABLE on Windows: there is no
      `resource` module, so the subprocess worker's RLIMIT_AS call is skipped and
      --runner local is advisory by design. Step 15 ACCOUNTS for it. Nothing here
      TESTS it, and no green run may be read as saying otherwise.
  Those stay explicit. See docs/VALIDATION.md section 7.

.PARAMETER Archive
  Version archive to compare estimator.py against. Default: the current shipped
  version. Update this default when a new version is promoted.

.PARAMETER Full
  Also run shape parity, carrier strength, the deployment envelope and the
  official Phase-2 smoke. Mandatory before promotion, submission, or spending a
  vault touch.

.PARAMETER Log
  Guarantee that this run's VERDICTS reach a file, whatever the caller does.

  The reason is a defect found on 2026-08-24: every preflight transcript this
  repository had ever committed as evidence was MISSING ITS VERDICTS. The step
  results and the final GREEN/RED/SKIPPED summary are written with `Write-Host`,
  which in Windows PowerShell 5.1 goes to the information stream -- so
  `.\preflight.ps1 2>&1 | Out-File` captures every child process's stdout and NONE
  of the gate's own conclusions. Three committed transcripts contained 900 lines
  of detail and zero verdict lines, and were cited as evidence of green runs.

  WHAT -Log DOES AND DOES NOT CAPTURE, measured rather than assumed:

    Start-Transcript (what -Log uses)   verdicts YES   child stdout NO
    .\preflight.ps1 2>&1  | Out-File    verdicts NO    child stdout YES
    .\preflight.ps1 *>&1  | Tee-Object  verdicts YES   child stdout YES

  In a `-NonInteractive` host with redirected stdio, Start-Transcript records the
  PowerShell host's own output and NOT the stdout of native child processes, so a
  `-Log` file carries the sixteen step results and the summary but not the suites'
  own output. That is the half that was missing and the half that matters for a
  claim of green.

  FOR A COMPLETE EVIDENCE TRANSCRIPT, use both streams explicitly:

    .\scripts\preflight.ps1 -Full *>&1 | Tee-Object -FilePath <path>

  Never `2>&1` -- it silently drops every verdict.

.EXAMPLE
  .\scripts\preflight.ps1
  .\scripts\preflight.ps1 -Full
  .\scripts\preflight.ps1 -Full -Log experiments/2026-08-24_p2_p0/verdicts.txt
  .\scripts\preflight.ps1 -Full *>&1 | Tee-Object -FilePath evidence.txt   # complete
  .\scripts\preflight.ps1 -Archive estimator_v2_5r.py
#>
[CmdletBinding()]
param(
    [string]$Archive = "estimator_v2_6_1.py",
    [switch]$Full,
    [string]$Log = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# --- the gate OWNS its children's text encoding --------------------------------
# `whest validate` prints a Unicode box-drawing table. On Windows, Python resolves
# a REDIRECTED stdout to the ANSI codepage (cp1252 here) regardless of `chcp
# 65001` or [Console]::OutputEncoding, so the CLI validates successfully, dies in
# its results printer (cli.py:2773), and then dies again in its error printer
# (cli.py:3655) -- exit 1, gate RED, estimator fine. A gate that reports the
# console's encoding as an estimator defect is a check reporting a STATE rather
# than testing a PROPERTY, which is the failure class this repo has lost four
# checks to (docs/GOTCHAS.md). So the gate sets it, asserts it in step 3, and
# restores whatever the caller had.
$script:PrevPyIoEnc = $env:PYTHONIOENCODING
$script:PrevPyUtf8  = $env:PYTHONUTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8       = "1"

$failures = New-Object System.Collections.ArrayList
$skipped  = New-Object System.Collections.ArrayList

$script:Transcribing = $false
if ($Log) {
    $logDir = Split-Path -Parent $Log
    if ($logDir -and -not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    }
    try {
        Start-Transcript -Path $Log -Force | Out-Null
        $script:Transcribing = $true
    } catch {
        # A transcript that silently did not start would produce exactly the
        # verdict-free evidence file this parameter exists to prevent.
        Write-Host "FAILED to start transcript at ${Log}: $_" -ForegroundColor Red
        exit 2
    }
}

function Restore-CallerEncoding {
    # preflight must not silently change the encoding of the shell that ran it.
    $env:PYTHONIOENCODING = $script:PrevPyIoEnc
    $env:PYTHONUTF8       = $script:PrevPyUtf8
    if ($script:Transcribing) {
        try { Stop-Transcript | Out-Null } catch { }
        $script:Transcribing = $false
    }
}

# A terminating error raised OUTSIDE a Step block -- with $ErrorActionPreference
# = "Stop" that is any unhandled failure in this script's own plumbing -- would
# skip the two `Restore-CallerEncoding` calls at the exit paths and leave the
# CALLER's shell with our encoding settings. A Skeptic pointed out that the
# restore was not on every path; this trap makes it so.
trap { Restore-CallerEncoding; break }

function Step {
    param([string]$Name, [scriptblock]$Body)
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    # Exit code is the only verdict. Windows PowerShell 5.1 wraps a native
    # command's stderr in an ErrorRecord, so with ErrorActionPreference=Stop a
    # benign warning on stderr fails the step -- which is exactly what happened
    # to `whest validate`'s SymmetryLossWarning on the first run of this script.
    #
    # STEP NAMES CARRY NO CHECK COUNTS. A count in a runtime label is a snapshot
    # that goes stale on the next test added -- this gate had three such bumps in
    # one day. The verdict is the exit code. Counts live in the header docstring.
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

# THE TRANSCRIPT WITNESSES ITS OWN COMMIT.
#
# Until 2026-08-25 a transcript recorded no SHA, so "GREEN 16/16 at HEAD <x>" was
# an assertion the artifact could not support -- an independent reviewer had to
# reconstruct the HEAD from file mtimes to check it, and a submission gate that
# names a HEAD is not mechanically checkable without this. Two lines fix it.
$gitHead = (& git rev-parse HEAD 2>$null)
$gitDirty = (& git status --porcelain 2>$null)
Write-Host "HEAD:    $gitHead"
if ($gitDirty) {
    Write-Host "worktree: DIRTY at start --" -ForegroundColor Yellow
    $gitDirty -split "`n" | ForEach-Object { if ($_) { Write-Host "    $_" } }
    Write-Host "  (steps 15/16 regenerate tracked evidence, so the tree is" -ForegroundColor Yellow
    Write-Host "   EXPECTED to be dirty at the END of a -Full run)" -ForegroundColor Yellow
} else {
    Write-Host "worktree: clean at start"
}

# --- 1. the OFFICIAL STACK, before anything is measured against it -------------
# Everything below is a measurement, and a measurement taken under the wrong
# packages is not a weaker measurement -- it is a different one. Until 2026-08-24
# this gate ran ten green steps against whestbench 0.14 / flopscope 0.10 while the
# competition graded on 0.16 / 0.12, and reported GREEN. It could not have done
# otherwise: nothing in it looked.
#
# `official_config.verify()` asserts the live round field by field against the
# INSTALLED package -- budget.RoundConfig for all three rounds, ContestSpec for the
# live one's limits, score_multiplier EXERCISED rather than read, and the
# run_config of a real `whest run --json`, which is what the runner actually used
# rather than what it declares. It also pins the trap that made the stale default
# invisible: the package still exports LAMBDA_FLOPS_PER_SECOND = 1e11 as a
# DEPRECATED ALIAS for the Phase-1 rate, while the live rate is 0.
Step "1. official stack + round configuration" {
    uv run python local_sim/official_config.py
}

# --- 2. provenance: are our mints still the grader's mints? -------------------
# Two commands, one step, because they answer the same question from opposite
# ends. mint_provenance asks "do WE still produce the grader's bytes"; the
# tag-drift regression asks "is the DATASET still the one those bytes came
# from" -- and, since 2026-08-31, "does the loader still take the route we
# compared". The second was added by WP0A after an independent Skeptic pointed
# out that a 70-object comparison nothing in the gate enforced is a comparison
# nobody re-runs. The denominator stays SIXTEEN; this is not a new step.
Step "2. mint provenance + dataset pin" {
    uv run python local_sim/mint_provenance.py
    if ($LASTEXITCODE -ne 0) { return }
    uv run python local_sim/test_p2_tag_drift.py
}

# --- 3. invariants on the SHIPPED file ----------------------------------------
# TWO commands, ONE step -- the denominator stays 16. The second is invariant
# 9's mutation battery: that invariant's absolute float32 clause was replaced on
# 2026-09-03 by a 1%-of-raw-MSE criterion, and an amendment that makes a failing
# gate PASS has to prove it still bites. The proof was written as a file NOTHING
# EXECUTED -- an independent Skeptic found it cited as "a gate" while this file
# and certify_candidate.py both ran only edge_cases.py. It is a gate now.
Step "3. invariant suite" {
    uv run python local_sim/edge_cases.py
    if ($LASTEXITCODE -ne 0) { return }
    uv run python local_sim/test_edge_cases_inv9.py estimator.py
}

# --- 4. grader contract --------------------------------------------------------
Step "4. whest validate" {
    # Assert the property, do not assume the export took. The export above is
    # unconditional, so a HOSTILE CALLER cannot trip this -- it is overwritten,
    # which is the point of the gate owning the setting. (An earlier version of
    # this comment claimed a caller could turn the gate RED here; an independent
    # Skeptic falsified it by running the gate with PYTHONIOENCODING=cp1252 and
    # getting GREEN. A comment asserting a property the code lacks is worse in
    # the enforcement layer than anywhere else.) What this DOES catch is a
    # toolchain that ignores the variable: a future venv, a different Python, or
    # a `uv run` that stops forwarding the environment -- each turns the gate RED
    # HERE with a readable reason, instead of sixty lines later inside someone
    # else's traceback.
    $enc = (uv run python -c "import sys; print(sys.stdout.encoding)") | Select-Object -Last 1
    Write-Host "  child stdout encoding: $enc"
    if ("$enc".Trim() -notmatch '^(?i)utf-?8$') {
        throw "child stdout encoding is '$enc', not utf-8 -- the gate failed to own it"
    }
    uv run whest validate --estimator estimator.py
}

# --- 5. ship-file identity -----------------------------------------------------
# estimator.py must be byte-identical to its version archive modulo line endings.
# Nothing else in the repo asserts this, and the archive is what every graded
# number is attributed to.
Step "5. ship-file identity vs $Archive" {
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

# --- 6. vault seal -------------------------------------------------------------
# Counter must equal the number of touch-log lines. A quiet extra evaluation is
# exactly what the seal exists to make impossible.
Step "6. vault seal (counter vs log)" {
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

# --- 7. seed ledger ------------------------------------------------------------
# check_disjoint() (invariant 12) proves no two blocks ALLOCATE the same seeds.
# It cannot see a probe with a seed hardcoded into it, which is how D16 was drawn
# 270 seeds dry while seed_blocks.py called it "Reusable" -- and how two probes
# ended up measuring two "independent" kills on the same 24 networks. This step
# re-validates the consumption ledger and statically scans the tree for seed
# literals and seeds() calls that the ledger does not record.
Step "7. seed ledger (consumption + source scan) + enforcement labels" {
    uv run python local_sim/seed_ledger.py --check --scan
    if ($LASTEXITCODE -ne 0) { return }
    # WP0B built local_sim/test_enforcement_labels.py so a stale coverage label
    # in THIS docstring or in VALIDATION.md section 7 is a red test rather than
    # documentation drift -- and then wired it into nothing, which is the exact
    # defect it was built to fix, on itself. An independent Skeptic pointed that
    # out and found two labels still false. It runs here now, as a second
    # command in an existing step, so the denominator stays SIXTEEN.
    #
    # --labels-only: the guard-mutation mirror (its section C) costs ~4 minutes
    # and rebuilds two suites in scratch. It is not gate-cheap. Run the full
    # suite deliberately: uv run python local_sim/test_enforcement_labels.py
    uv run python local_sim/test_enforcement_labels.py --labels-only
}

# --- 8. the canonical competition scorer ---------------------------------------
# local_sim/competition_score.py is the ONE place the grader's identity lives, and
# every gate downstream reads its numbers. 134/134 asserts it is still bitwise equal
# to the installed whestbench on the scalar primitives, on the aggregate, and on
# the real scoring loop INCLUDING both failure paths -- and carries the graded
# fixture where the arithmetic and geometric estimands pick opposite winners.
Step "8. canonical scorer + the binding gate" {
    uv run python local_sim/test_competition_score.py
}

# --- 9. the seed ledger's refusals actually fire --------------------------------
# Step 7 proves the ledger is CONSISTENT. It cannot prove the guards still refuse.
# These are 66 negative tests: unregistered block, range leaving its block, any
# overlap in a DECISION block, a second claim on a SINGLE_USE one, a hand-edited
# ledger, an artifact rewritten after its claim.
Step "9. seed-ledger refusals + the truth-read register" {
    uv run python local_sim/test_seed_ledger.py
    if ($LASTEXITCODE -ne 0) { return }
    # SECOND COMMAND, ONE STEP -- the denominator stays SIXTEEN.
    #
    # Added 2026-09-10 (Stage 0 R3). Two demonstrated defects: claim_range had
    # an unlocked read-decide-append window (twelve concurrent claimers on one
    # SINGLE_USE block, twelve succeeded), and nothing counted READS at all,
    # which is how P08 was claimed once and read three times. This drives
    # sacrificial ledgers in a temp directory and then checks the LIVE register
    # reports P08 as spent.
    uv run python local_sim/test_truth_reads.py
}

# --- 10. vault integrity, not merely the counter ---------------------------------
# Step 6 compares a number to a line count. This re-hashes every stored array
# against the manifest and reconstructs networks from their seeds to confirm the
# seed->weights map still holds. It reads no candidate predictions, so it COSTS NO
# TOUCH. The arrays are gitignored (33 MB), so a fresh clone legitimately has none
# and must not go red for it -- step 6 still covers the seal there.
Step "10. vault integrity (array hashes + seed->weights)" {
    if (-not (Test-Path "experiments/vault/vault_v1.npz")) {
        Write-Host "  experiments/vault/vault_v1.npz absent (gitignored)."
        Write-Host "  Run 'uv run python local_sim/vault.py mint' to restore it."
        $script:StepSkipped = "vault arrays absent (gitignored); step 6 still checked the seal"
        return
    }
    uv run python local_sim/vault.py verify
}

# --- 11. the canonical grader-lifecycle evaluator -------------------------------
# vault.py:282 and coverage_lab.py:226 build a fresh Estimator per MLP and never
# call setup(); the grader builds ONE and calls setup() once, OFF-budget. This
# suite pins that lifecycle, the six failure paths, and the immutable evidence
# rows -- the object every future candidate comparison is scored through.
Step "11. lifecycle evaluator + the certification consumers' mutation battery" {
    uv run python local_sim/lifecycle_eval.py --self-test
    if ($LASTEXITCODE -ne 0) { return }
    # SECOND COMMAND, ONE STEP -- the denominator stays SIXTEEN.
    #
    # Added 2026-09-10 (Stage 0 R2). certify_candidate.py section 9 used to
    # accept a linux_memory_gate report on four checks that said nothing about
    # WHAT the report had run: shape, depth, budget, memory cap and weight
    # representation are all CLI flags on that gate, and none was compared to
    # the candidate being certified. This battery mutates a good report one
    # field at a time and requires each mutation to be REFUSED. It is pure --
    # no subprocess, no candidate execution -- so it belongs in the cheap gate;
    # the expensive candidate run stays queued.
    uv run python local_sim/test_linux_memory_binding.py
}

# --- 12. the counted-FLOP anchor against the OFFICIAL runner --------------------
# The one check that says the local meter bills what the grader bills, OP BY OP.
# It caught two mechanisms worth 37,779,456 FLOPs between them: local weights that
# were raw NumPy rather than flopscope-backed (so every weight-METHOD call was free
# here and billed there), and whestbench 0.16's own in-window
# `materialise_predictions`, which is free for a float32 return and 2 FLOPs per
# element for a float64 one.
#
# It compares against a stored real `whest run --profile --json` report, and it
# REFUSES rather than skips when that report is absent -- an absent comparison is
# not a pass, and a matching TOTAL with a moved OPERATION is a coincidence, not
# parity.
Step "12. counted-FLOP anchor vs the official runner" {
    uv run python local_sim/p2_flop_anchor.py --gate
}

# --- 13-16. -Full only ----------------------------------------------------------
if ($Full) {
    # Shape parity: on any Strassen-INELIGIBLE shape the estimator falls back to
    # the dense chain, which is v2.4.1's algorithm exactly, and must reproduce it
    # BIT FOR BIT. This is the check that would have caught v2.6 deleting the whole
    # first layer's sampled path on every odd width -- 7 of 17 shapes wrong, 0.58
    # max relative error, and every output finite and plausible.
    Step "13. shape parity" {
        uv run python local_sim/test_shape_parity.py
    }
    # Carrier design strength: the moment-error invariants that the whole design
    # lane's arithmetic rests on.
    Step "14. carrier strength" {
        uv run python local_sim/test_carriers.py
    }
    # The deployment envelope, including the two constraints this host CANNOT
    # enforce. It prints those as NOT PROVABLE rather than passing them, which is
    # the whole reason it is a step: the 8,192 MB memory limit has no enforcement
    # path on Windows at all, and a gate that stayed silent about that would be
    # certifying a candidate against a limit nothing here measures.
    # ...and, since P1-PRE (2026-08-25), the proof that its own `--no-write`
    # mode is read-only -- run from OUTSIDE it, by a child process, sha256 and
    # `git status --porcelain`. The suite verifies that claim about itself
    # before printing it; a file that grades its own read-only mode is not
    # enough, because it was a file grading its own read-only mode that printed
    # "the artifact on disk was NOT touched" immediately after rewriting a
    # committed artifact twice. The same child skips section 6, so it also
    # covers the D1 skip path: a provocation that did not run must never print
    # green.
    #
    # Two commands, one step, and the first one's exit code is honoured -- the
    # Step wrapper only sees $LASTEXITCODE at the end of the block.
    Step "15. deployment envelope (+ what it cannot prove, + read-only proof)" {
        uv run python local_sim/test_deployment_envelope.py
        if ($LASTEXITCODE -ne 0) { return }
        uv run python local_sim/test_envelope_readonly.py
    }
    # DEPLOYMENT AUTHORITY. Everything else in this gate is a local MODEL of the
    # grader; this is the grader's own runner on the organisers' own Phase-2 data.
    # Zero failures is the bar, and `--runner subprocess` is not optional: the
    # in-process runner cannot reproduce worker death, and its residual figure is a
    # different instrument from the one the 0.4 s cliff is measured against.
    Step "16. official Phase-2 subprocess smoke (zero failures)" {
        $py = @"
import json, subprocess, sys, os
cmd = [os.path.join(os.path.dirname(sys.executable), 'whest.exe'), 'run',
       '--estimator', 'estimator.py',
       '--dataset', 'hf://aicrowd/arc-whestbench-public-2026@v2-phase2',
       '--split', 'mini', '--runner', 'subprocess', '--n-mlps', '2', '--json']
p = subprocess.run(cmd, capture_output=True, text=True)
i = p.stdout.find('{')
if i < 0:
    print('no JSON report:', (p.stderr or p.stdout)[-2000:]); sys.exit(1)
d = json.loads(p.stdout[i:])
if not d.get('results'):
    err = d.get('error') or {}
    code = err.get('code', '?')
    print('  run REFUSED at %s: %s -- %s' % (err.get('stage','?'), code,
                                             err.get('message','')))
    if code == 'SETUP_TIMEOUT':
        print('  This is an ENVIRONMENT condition, not an estimator defect.')
        print('  The 5 s setup ceiling covers worker spawn + IMPORTS + setup().')
        print('  On this host `import whestbench` alone is 1.3 s quiet and')
        print('  4.7-9.1 s under load; our own estimator setup is 0.0006 s.')
        print('  Re-run on a QUIET machine (close WSL/VMs/browsers).')
        print('  Reported as a FAILURE and not a skip: a gate that could not')
        print('  verify deployment must not certify it.')
    sys.exit(1)
r, cfg = d['results'], d['run_config']
print('  whestbench   %s' % d['whestbench_version'])
print('  dataset      %s' % cfg['dataset']['path'])
print('  shape        %sx%s   B = %s   lambda = %s' % (
    cfg['width'], cfg['depth'], format(cfg['flop_budget'], ','),
    cfg['lambda_flops_per_second']))
print('  residual cap %ss   wall %ss' % (
    cfg['residual_wall_time_limit_s'], cfg['wall_time_limit_s']))
print('  score        %.6e   raw MSE %.6e   mult %.4f' % (
    r['adjusted_final_layer_score'], r['final_layer_mse'],
    r['mean_score_multiplier']))
print('  C/B          %.7f   failures %d' % (
    r['mean_compute_utilization'], r['n_failed_mlps']))
worst = max(m['residual_wall_time_s'] for m in r['per_mlp'])
cap = cfg['residual_wall_time_limit_s']
print('  worst residual %.5f s of %.2f s cap  -> %.1fx headroom (GRADED PATH)' % (
    worst, cap, cap / worst if worst else float('inf')))
bad = []
if r['n_failed_mlps'] != 0:
    bad.append('n_failed_mlps = %d' % r['n_failed_mlps'])
if not str(cfg['dataset']['path']).endswith('@v2-phase2'):
    bad.append('not the v2-phase2 dataset')
if cfg['lambda_flops_per_second'] != 0.0:
    bad.append('lambda is not 0')
if cfg['residual_wall_time_limit_s'] != 0.4:
    bad.append('residual cap is not 0.4 s')
if worst > cap:
    bad.append('worst residual %.4f s exceeds the %.2f s cap' % (worst, cap))
if bad:
    print('  REFUSED: ' + '; '.join(bad)); sys.exit(1)
"@
        $py | uv run python -
    }
} else {
    [void]$skipped.Add("13. shape parity              -- not run without -Full (~480 s)")
    [void]$skipped.Add("14. carrier strength         -- not run without -Full (~130 s)")
    [void]$skipped.Add("15. deployment envelope      -- not run without -Full (~40 s)")
    [void]$skipped.Add("16. official Phase-2 smoke   -- not run without -Full (~30 s), and it is DEPLOYMENT AUTHORITY")
}

# There are SIXTEEN checks in this gate. The default tier runs twelve of them and
# skips four; a missing vault array skips another. Denominating against the tier
# that happened to run would let an incomplete run report "12/12", which is the
# precise dishonesty this accounting exists to prevent -- so the denominator is
# always 16 and every skip is named below.
$total  = 16
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
    Write-Host "  - public Phase-2 mini (CONFIRM, read-only -- FIT NOTHING ON IT):"
    Write-Host "      uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini --runner subprocess"
    Write-Host "    (v1-phase1 is the PHASE-1 split; use it only to replay a Phase-1 artifact)"
    Write-Host "  - the 8,192 MB memory limit is UNENFORCEABLE on this host (no RLIMIT_AS on"
    Write-Host "    Windows). Step 15 ACCOUNTS for it; nothing here TESTS it."
    Write-Host "  - GPU mirror parity:                     .venv-gpu\Scripts\python.exe local_sim/gpu_parity.py"
    Write-Host "  - vault touch (ONE per estimator change): .venv-gpu\Scripts\python.exe local_sim/vault.py evaluate --label <v> --reason estimator-change"
    Write-Host "  - Samuel's explicit approval before any submission."
    Restore-CallerEncoding
    exit 0
} else {
    Write-Host "PREFLIGHT RED -- failed: $($failures -join ', ')" -ForegroundColor Red
    Write-Host "Do not promote, do not submit, do not spend a decision block." -ForegroundColor Red
    Restore-CallerEncoding
    exit 1
}
