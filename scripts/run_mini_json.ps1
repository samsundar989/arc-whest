<#
    Run the organisers' split and write the JSON report as UTF-8 WITHOUT a BOM.

    WHY THIS EXISTS, and it is not a convenience
    --------------------------------------------
    `uv run whest run --format json > report.json` under Windows PowerShell 5.1
    writes **UTF-16LE**. Nothing complains. The file looks fine in an editor.
    Then `json.load(open(path, encoding="utf-8"))` dies on `byte 0xff in position
    0`, and `local_sim/predict_public.py` cannot read its own inputs.

    On 2026-09-16 that swallowed 4h28m of paired subprocess runs before anyone
    looked at the first two bytes. `whest run` has no output-file flag, so the
    redirect is the only path and the redirect is the bug.

    `Out-File -Encoding utf8` is NOT the fix. PowerShell 5.1 writes a BOM, and
    Python's plain `utf-8` codec does not strip one -- it hands json a leading
    U+FEFF and json refuses it. Only `utf-8-sig` would, and the readers here do
    not use it. `[System.IO.File]::WriteAllText` with `UTF8Encoding($false)` is
    the one spelling that produces what the readers expect.

    THE OTHER THING THIS FIXES
    --------------------------
    `-NoResidualLimit` passes `--no-residual-wall-time-limit`. Handoff 09-14 §0.1
    deleted the local-seconds residual gate from every decision path: this host
    reads residual roughly 27x high per operation, so the graded 0.4 s cap fails
    every MLP locally and the run reports the zero-prediction MSE instead of the
    estimator's. Both arms of the 09-16 pair came back `n_failed_mlps: 100`,
    `failure_breakdown: {residual_wall_time_exhausted: 100}`, and an identical
    `final_layer_mse 0.9095` -- identical because it is not the estimator's
    number at all. FLOPs, ops and wall from such a run are real; the MSE is not.

    Op count against the 22,000 gate is the residual evidence. Local seconds are
    advisory and are never a gate here.

    Usage:
      powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_mini_json.ps1 `
        -Estimator estimator_k3_v31.py `
        -Out C:/Users/samsu/Documents/arc-whest/experiments/<slice>/v31_mini.json `
        -NoResidualLimit
#>
param(
    [Parameter(Mandatory = $true)][string]$Estimator,
    [Parameter(Mandatory = $true)][string]$Out,
    [switch]$NoResidualLimit,
    [string]$Split = "mini",
    [string]$Dataset = "hf://aicrowd/arc-whestbench-public-2026@v2-phase2",
    [string]$Runner = "subprocess",
    [int]$NMlps = 0
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$whestArgs = @(
    "run",
    "--estimator", $Estimator,
    "--dataset", $Dataset,
    "--split", $Split,
    "--runner", $Runner,
    "--detail", "full",
    "--format", "json"
)
if ($NoResidualLimit) { $whestArgs += "--no-residual-wall-time-limit" }
if ($NMlps -gt 0) { $whestArgs += @("--n-mlps", "$NMlps") }

Write-Host "whest $($whestArgs -join ' ')"
$lines = & uv run whest @whestArgs
if ($LASTEXITCODE -ne 0) { throw "whest run exited $LASTEXITCODE" }

$text = ($lines -join "`n")

# Refuse to write something that is not JSON. A truncated or warning-prefixed
# report that parses as nothing is the same silent failure in a new costume,
# and it is cheaper to fail here than four hours downstream.
try { $null = $text | ConvertFrom-Json }
catch { throw "whest run produced output that is not JSON ($($text.Length) chars). First 200: $($text.Substring(0, [Math]::Min(200, $text.Length)))" }

$dir = Split-Path -Parent $Out
if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force $dir | Out-Null }
[System.IO.File]::WriteAllText($Out, $text, (New-Object System.Text.UTF8Encoding($false)))

$bytes = [System.IO.File]::ReadAllBytes($Out)
if ($bytes.Length -ge 2 -and $bytes[0] -eq 0xFF -and $bytes[1] -eq 0xFE) { throw "wrote UTF-16 after all" }
if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) { throw "wrote a BOM after all" }

Write-Host "wrote $Out  ($($bytes.Length) bytes, UTF-8 no BOM, parses as JSON)"
