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

    CRASH-SAFE EVIDENCE (2026-09-16, validation audit P2)
    -----------------------------------------------------
    stdout used to be captured IN MEMORY and validated before anything was
    written, so a truncated/malformed report -- or this PowerShell dying mid-run
    -- lost every byte. Now the child's stdout and stderr go straight to two
    unique files from launch, through OS file handles (Start-Process
    redirection: raw bytes, no PowerShell re-encoding, nothing buffered here):

        <Out>.<stamp>.stdout.raw     <Out>.<stamp>.stderr.raw

    They are KEPT whatever happens. Afterwards:
      * valid JSON  -> published atomically (pending file + rename, never an
                       overwrite) to <Out>, UTF-8 without BOM;
      * invalid, or a nonzero exit -> <Out>.<stamp>.failure.json, a small
                       metadata record naming the raw files, their sizes and
                       SHA-256, the exit code and the reason.
    <Out> must not already exist; evidence is never overwritten, and it is
    cheaper to refuse that before a multi-hour run than after it.

    Usage:
      powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_mini_json.ps1 `
        -Estimator estimator_k3_v31.py `
        -Out C:/Users/samsu/Documents/arc-whest/experiments/<slice>/v31_mini.json `
        -NoResidualLimit

    -LauncherExe / -LauncherArgs are a TEST SEAM (default: `uv run whest`). The
    whest arguments are appended after -LauncherArgs.
#>
param(
    [Parameter(Mandatory = $true)][string]$Estimator,
    [Parameter(Mandatory = $true)][string]$Out,
    [switch]$NoResidualLimit,
    [string]$Split = "mini",
    [string]$Dataset = "hf://aicrowd/arc-whestbench-public-2026@v2-phase2",
    [string]$Runner = "subprocess",
    [int]$NMlps = 0,
    [switch]$DebugTracebacks,
    [string]$LauncherExe = "uv",
    [string[]]$LauncherArgs = @("run", "whest")
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Quote-Arg([string]$a) {
    # Start-Process joins -ArgumentList with spaces and does NOT quote, so an
    # argument containing whitespace or a quote would split. MSVCRT rules.
    if ($a -eq "") { return '""' }
    if ($a -notmatch '[\s"]') { return $a }
    $s = $a -replace '(\\*)"', '$1$1\"'
    $s = $s -replace '(\\+)$', '$1$1'
    return '"' + $s + '"'
}

function Write-NewFileAtomic([string]$Path, [string]$Text) {
    # Pending file, flushed to disk, then a rename that FAILS if the target
    # exists. Never a partial file at $Path, never an overwrite.
    $pending = "$Path.pending-" + [guid]::NewGuid().ToString("N")
    $bytes = $utf8NoBom.GetBytes($Text)
    $fs = New-Object System.IO.FileStream($pending, [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    try { $fs.Write($bytes, 0, $bytes.Length); $fs.Flush($true) } finally { $fs.Dispose() }
    try { [System.IO.File]::Move($pending, $Path) }
    catch { Remove-Item -LiteralPath $pending -Force -ErrorAction SilentlyContinue; throw }
}

function Assert-Utf8NoBom([string]$Path) {
    $b = [System.IO.File]::ReadAllBytes($Path)
    if ($b.Length -ge 2 -and $b[0] -eq 0xFF -and $b[1] -eq 0xFE) { throw "wrote UTF-16 after all: $Path" }
    if ($b.Length -ge 3 -and $b[0] -eq 0xEF -and $b[1] -eq 0xBB -and $b[2] -eq 0xBF) { throw "wrote a BOM after all: $Path" }
    return $b.Length
}

function Sha256-Hex([string]$Path) {
    # Plain .NET, streamed. NOT Get-FileHash: in Windows PowerShell 5.1 that is a
    # script function in the Microsoft.PowerShell.Utility module, auto-loaded by
    # PSModulePath. A driver that puts PowerShell 7's modules first (Codex does)
    # makes it unloadable, and every hash below fails. Validation review 09-17 §3.
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
    try { $h = $sha.ComputeHash($fs) } finally { $fs.Dispose(); $sha.Dispose() }
    return ([System.BitConverter]::ToString($h) -replace '-', '').ToLowerInvariant()
}

function Receipt-PathFor([string]$Report) {
    # CANONICAL receipt name: <report stem>.receipt.json -- the name
    # `local_sim/predict_public.py::find_receipt` discovers. (Until 2026-09-17
    # this wrote <report>.receipt.json, which the projector never found.)
    return [System.IO.Path]::ChangeExtension($Report, ".receipt.json")
}

function File-Record([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return @{ path = $Path; exists = $false } }
    return @{
        path   = $Path
        exists = $true
        bytes  = (Get-Item -LiteralPath $Path).Length
        sha256 = Sha256-Hex $Path
    }
}

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
if ($DebugTracebacks) { $whestArgs += "--debug" }

# RECEIPT (2026-09-16): bind the report to the SOURCE that produced it. Hash the
# estimator before launch and again after; a change in between refuses the receipt.
function Source-Sha([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    return Sha256-Hex $Path
}
$srcFull = [System.IO.Path]::GetFullPath($Estimator)
$srcShaBefore = Source-Sha $srcFull

$Out = [System.IO.Path]::GetFullPath($Out)
if (Test-Path -LiteralPath $Out) { throw "refusing to overwrite existing evidence: $Out" }
$receiptPath = Receipt-PathFor $Out
if ($receiptPath -eq $Out) { throw "the report may not itself be named *.receipt.json: $Out" }
if (Test-Path -LiteralPath $receiptPath) { throw "refusing to overwrite an existing receipt: $receiptPath" }
$dir = Split-Path -Parent $Out
if ($dir -and -not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Force $dir | Out-Null }

$stamp = "{0}-{1}" -f (Get-Date -Format "yyyyMMddTHHmmssfff"), ([guid]::NewGuid().ToString("N").Substring(0, 8))
$rawOut = "$Out.$stamp.stdout.raw"
$rawErr = "$Out.$stamp.stderr.raw"
$failMeta = "$Out.$stamp.failure.json"

$exe = (Get-Command $LauncherExe -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$allArgs = @($LauncherArgs) + $whestArgs
$argLine = ($allArgs | ForEach-Object { Quote-Arg $_ }) -join " "
$started = (Get-Date).ToUniversalTime().ToString("o")

Write-Host "$LauncherExe $($allArgs -join ' ')"
Write-Host "raw stdout -> $rawOut"
Write-Host "raw stderr -> $rawErr"

$proc = Start-Process -FilePath $exe -ArgumentList $argLine -NoNewWindow -PassThru `
    -RedirectStandardOutput $rawOut -RedirectStandardError $rawErr
$null = $proc.Handle   # PS 5.1: without touching Handle, ExitCode can come back $null
$proc.WaitForExit()
$whestExit = $proc.ExitCode
if ($null -eq $whestExit) { $whestExit = -1 }

# DO NOT throw on the exit code here. `whest run` exits 1 when ANY MLP raises --
# and it still evaluated the rest and printed a complete report naming which
# ones failed and why (job 1400, 2026-09-16: 98 good rows and the identity of
# the 2 bad ones, once deleted by trusting an exit code over the artifact).
# Evidence first; the exit code is re-raised at the very end.
if ($whestExit -ne 0) {
    Write-Host "NOTE: whest run exited $whestExit -- raw output is kept; validating it anyway."
}

$reason = $null
$text = $null
try {
    $rawBytes = [System.IO.File]::ReadAllBytes($rawOut)
    $strict = New-Object System.Text.UTF8Encoding($false, $true)
    $text = $strict.GetString($rawBytes)
    if ($text.Length -gt 0 -and $text[0] -eq [char]0xFEFF) { $text = $text.Substring(1) }
    $text = ($text -replace "`r`n", "`n").Trim()
} catch {
    $reason = "stdout is not readable UTF-8: $($_.Exception.Message)"
}

# Refuse to publish something that is not JSON. A truncated or warning-prefixed
# report is the same silent failure in a new costume.
if ($null -eq $reason) {
    if ($text.Length -eq 0) { $reason = "stdout is empty" }
    else {
        try { $null = $text | ConvertFrom-Json }
        catch { $reason = "stdout is not JSON ($($text.Length) chars): $($_.Exception.Message)" }
    }
}

$published = $false
if ($null -eq $reason) {
    Write-NewFileAtomic $Out $text
    $n = Assert-Utf8NoBom $Out
    $published = $true
    Write-Host "wrote $Out  ($n bytes, UTF-8 no BOM, parses as JSON)"
}

if (-not $published -or $whestExit -ne 0) {
    $meta = [ordered]@{
        artifact          = "run_mini_json_failure"
        artifact_version  = 1
        started_utc       = $started
        ended_utc         = (Get-Date).ToUniversalTime().ToString("o")
        exit_code         = $whestExit
        json_valid        = $published
        reason            = $(if ($reason) { $reason } else { "nonzero exit with a valid report" })
        report            = $(if ($published) { File-Record $Out } else { $null })
        stdout_raw        = File-Record $rawOut
        stderr_raw        = File-Record $rawErr
        launcher          = $LauncherExe
        arguments         = $allArgs
        head_200          = $(if ($text) { $text.Substring(0, [Math]::Min(200, $text.Length)) } else { "" })
    }
    Write-NewFileAtomic $failMeta ($meta | ConvertTo-Json -Depth 5)
    $null = Assert-Utf8NoBom $failMeta
    Write-Host "failure metadata -> $failMeta"
}

if ($published) {
    # arc-whest.run_receipt/1, the schema `local_sim/predict_public.py` validates.
    $srcShaAfter = Source-Sha $srcFull
    $ds = $null
    try { $ds = ($text | ConvertFrom-Json).run_config.dataset } catch { $ds = $null }
    $receipt = [ordered]@{
        schema         = "arc-whest.run_receipt/1"
        report_path    = $Out
        report_sha256  = Sha256-Hex $Out
        source_path    = $Estimator
        source_sha256  = $(if ($srcShaBefore -eq $srcShaAfter) { $srcShaBefore } else { $null })
        source_changed_during_run = ($srcShaBefore -ne $srcShaAfter)
        meter_path     = $(if ($Runner -eq "subprocess") { "whest_run_subprocess" } else { "unknown" })
        dataset        = $(if ($ds) { [ordered]@{ path = $ds.path; sha256 = $ds.sha256; n_mlps = $ds.n_mlps } } else { $null })
        host_load      = [ordered]@{ class = "unknown"; evidence = "runner-written receipt; queue overlap is not queried here" }
        provenance     = [ordered]@{ kind = "written_by_run_mini_json"; started_utc = $started;
                                     ended_utc = (Get-Date).ToUniversalTime().ToString("o"); arguments = $allArgs;
                                     exit_code = $whestExit }
    }
    Write-NewFileAtomic $receiptPath ($receipt | ConvertTo-Json -Depth 6)
    $null = Assert-Utf8NoBom $receiptPath
    Write-Host "receipt -> $receiptPath"
    if ($srcShaBefore -ne $srcShaAfter) {
        throw "the estimator changed during the run ($srcShaBefore -> $srcShaAfter); the receipt carries no source_sha256"
    }
}

if (-not $published) {
    throw "whest run produced no valid JSON report ($reason). Raw output kept: $rawOut / $rawErr; metadata: $failMeta"
}

# Only now, with the evidence durable, does a failed run report as failed.
if ($whestExit -ne 0) {
    throw "whest run exited $whestExit (the report WAS written to $Out -- read its failure_breakdown and per_mlp error_code)"
}
