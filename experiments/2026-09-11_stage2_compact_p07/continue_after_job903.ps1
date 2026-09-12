param()

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$statePath = Join-Path $PSScriptRoot 'continuation_job903_state.json'
$logPath = Join-Path $PSScriptRoot 'continuation_job903.log'
$lockPath = Join-Path $PSScriptRoot 'continuation_job903.lock'
$observations = Join-Path $PSScriptRoot 'observations'
$predictions = Join-Path $observations 'predictions.npz'
$producerReceipt = Join-Path $observations 'producer_receipt.json'
$verificationReceipt = Join-Path $PSScriptRoot 'verification_receipt.json'
$fitReceipt = Join-Path $PSScriptRoot 'fit_receipt.json'
$frozenDir = Join-Path $PSScriptRoot 'frozen_arms'
$freezeReceipt = Join-Path $frozenDir 'freeze_receipt.json'
$p01Report = Join-Path $PSScriptRoot 'p01_report.json'

function Write-State([string]$stage, [string]$status, [string]$detail) {
    $row = [ordered]@{
        schema = 'arc-whest.stage2-continuation-state.v1'
        recorded_utc = [DateTime]::UtcNow.ToString('o')
        pid = $PID
        stage = $stage
        status = $status
        detail = $detail
    }
    $row | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statePath -Encoding utf8
    "$(Get-Date -Format o) [$status] $stage - $detail" |
        Add-Content -LiteralPath $logPath -Encoding utf8
}

function Require-Absent([string]$path, [string]$label) {
    if (Test-Path -LiteralPath $path) {
        throw "$label already exists: $path"
    }
}

function Run-Native([string]$stage, [string]$stdoutPath, [string[]]$command) {
    Write-State $stage 'RUNNING' ($command -join ' ')
    $executable = $command[0]
    $arguments = $command[1..($command.Count - 1)]
    & $executable @arguments 2>&1 | Tee-Object -FilePath $stdoutPath
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "$stage exited $code; see $stdoutPath"
    }
    Write-State $stage 'SUCCEEDED' "exit 0; output $stdoutPath"
}

$lock = $null
$ownsRun = $false
try {
    Require-Absent $statePath 'continuation state'
    Require-Absent $logPath 'continuation log'
    Require-Absent (Join-Path $PSScriptRoot 'job903_raw.log') 'job 903 raw log'
    Require-Absent (Join-Path $PSScriptRoot 'job903_metadata.json') 'job 903 metadata'
    Require-Absent (Join-Path $PSScriptRoot 'job903_manifest.json') 'job 903 manifest'
    Require-Absent $verificationReceipt 'verification receipt'
    Require-Absent $fitReceipt 'fit receipt'
    Require-Absent $frozenDir 'frozen arm directory'
    Require-Absent $p01Report 'P01 report'
    Require-Absent (Join-Path $PSScriptRoot 'verification_stdout.log') 'verification transcript'
    Require-Absent (Join-Path $PSScriptRoot 'fit_stdout.log') 'fit transcript'
    Require-Absent (Join-Path $PSScriptRoot 'freeze_stdout.log') 'freeze transcript'
    Require-Absent (Join-Path $PSScriptRoot 'p01_submit.txt') 'P01 submission record'

    $lock = [System.IO.File]::Open(
        $lockPath,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None)
    $ownsRun = $true
    Set-Location -LiteralPath $repo
    Write-State 'wait_job903' 'RUNNING' 'polling workerq job 903; no partial output access'

    while ($true) {
        $raw = workerq show 903 --json
        if ($LASTEXITCODE -ne 0) {
            throw 'workerq show 903 failed'
        }
        $job = $raw | ConvertFrom-Json
        if ($job.state -eq 'SUCCEEDED') {
            break
        }
        if ($job.state -in @('FAILED', 'CANCELLED', 'PREEMPTED')) {
            throw "job 903 ended $($job.state): $($job.error)"
        }
        Start-Sleep -Seconds 45
    }

    if ($job.id -ne 903 -or $job.project -ne 'arc-whest' -or
        $job.command_signature -ne '0d7eb36ac46deb90' -or
        $job.snapshot_mode -ne 'none' -or $job.requested_gpu_count -ne 0 -or
        (Resolve-Path -LiteralPath $job.execution_cwd).Path -ne $repo) {
        throw 'terminal job 903 metadata does not match the registered producer run'
    }

    Write-State 'capture_job903' 'RUNNING' 'terminal success observed; preserving queue evidence'
    $raw | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'job903_metadata.json') -Encoding utf8
    $jobLog = workerq logs 903
    $jobLogCode = $LASTEXITCODE
    $jobLog | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'job903_raw.log') -Encoding utf8
    if ($jobLogCode -ne 0) {
        throw 'workerq logs 903 failed'
    }
    Copy-Item -LiteralPath $job.manifest_path -Destination (Join-Path $PSScriptRoot 'job903_manifest.json')
    if (-not (Test-Path -LiteralPath $predictions) -or
        -not (Test-Path -LiteralPath $producerReceipt)) {
        throw 'job 903 succeeded without both immutable producer outputs'
    }
    Write-State 'capture_job903' 'SUCCEEDED' "peak $($job.peak_ram_mib) MiB; runtime $($job.runtime_seconds) s"

    $common = @(
        '--predictions', $predictions,
        '--producer-receipt', $producerReceipt,
        '--candidates', (Join-Path $repo 'experiments\2026-09-10_stage2_k4_design\candidates.json'),
        '--bounds', (Join-Path $repo 'experiments\2026-09-10_stage2_k4_design\bounds.json'),
        '--identities', (Join-Path $repo 'experiments\2026-09-10_stage2_k4_design\identities.json'),
        '--objective', 'final_layer_mse')

    $verifyCommand = @('uv', 'run', 'python', 'local_sim/p2_k4_fit.py') +
        $common + @('--verify-only', '--receipt', $verificationReceipt)
    Run-Native 'verify_producer' (Join-Path $PSScriptRoot 'verification_stdout.log') $verifyCommand

    $fitCommand = @('uv', 'run', 'python', 'local_sim/p2_k4_fit.py') +
        $common + @(
            '--truth', (Join-Path $repo 'experiments\2026-09-10_slice5\truth_P07.npz'),
            '--data-role', 'protected-design',
            '--receipt', $fitReceipt,
            '--block', 'P07',
            '--note', 'Registered compact K4 coefficient fit on exposed reusable P07 DESIGN')
    Run-Native 'fit_p07' (Join-Path $PSScriptRoot 'fit_stdout.log') $fitCommand

    Run-Native 'freeze_arms' (Join-Path $PSScriptRoot 'freeze_stdout.log') @(
        'uv', 'run', 'python', 'local_sim/p2_freeze_k4_arms.py',
        '--fit-receipt', $fitReceipt,
        '--out-dir', $frozenDir)
    if (-not (Test-Path -LiteralPath $freezeReceipt)) {
        throw 'freezer exited zero without freeze_receipt.json'
    }

    Write-State 'submit_p01' 'RUNNING' 'submitting registered exposed-DESIGN four-arm comparison'
    $submit = workerq submit --project arc-whest --label compact-p01-four-arm `
        --priority high --gpus 0 --ram 8 --vram 0 --cpus 2 --no-snapshot -- `
        uv run python local_sim/p2_lab.py `
        --truth experiments/2026-08-21_p2_cp1/truth_P01.npz `
        --out experiments/2026-09-11_stage2_compact_p07/p01_report.json `
        --freeze-receipt experiments/2026-09-11_stage2_compact_p07/frozen_arms/freeze_receipt.json `
        --n 0 --competition-cost-lambda 0 --depth-subset 8 --gate 1.10 `
        --block P01 --note 'Registered coherent-K4 mechanism comparison on exposed P01 DESIGN' `
        --control exact --official-samples 1e9 --residual-cap 0 2>&1
    $submitCode = $LASTEXITCODE
    $submit | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'p01_submit.txt') -Encoding utf8
    if ($submitCode -ne 0) {
        throw 'workerq P01 submission failed'
    }
    $joined = $submit -join "`n"
    $match = [regex]::Match($joined, 'job #(\d+) submitted')
    if (-not $match.Success) {
        throw 'P01 submission succeeded but its workerq job id was not parsed'
    }
    Write-State 'submit_p01' 'SUCCEEDED' "workerq job $($match.Groups[1].Value); 8 GiB RAM, 2 CPU, no GPU, high priority"
    Write-State 'continuation' 'SUCCEEDED' 'P07 verified/fitted/frozen and P01 submitted; driver must inspect terminal P01 evidence'
}
catch {
    if ($ownsRun) {
        Write-State 'continuation' 'FAILED' $_.Exception.Message
    } else {
        Write-Error $_.Exception.Message
    }
    exit 1
}
finally {
    if ($null -ne $lock) {
        $lock.Dispose()
    }
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
}
