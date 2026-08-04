# Windows setup → first submission (copy-paste guide)

*One-time setup ~10 minutes + one ~850 MB dataset download. All commands in PowerShell.*

## 0. Prereqs

You already have: AIcrowd account, joined the challenge. You need your **API key**: log in at aicrowd.com → https://www.aicrowd.com/participants/me/customize → copy the key.

## 1. Install uv (Python project manager — handles Python itself too)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen the terminal, then check: `uv --version`.

## 2. Install the project

```powershell
cd $env:USERPROFILE\Documents\arc-whest
uv sync
```

This downloads Python 3.10 + the two competition packages (`whestbench` = harness & CLI, `flopscope` = FLOP counter) into a local `.venv`. Nothing global is touched.

## 3. Log in to AIcrowd (one time)

```powershell
uv run whest login
```

Paste the API key when prompted.

## 4. Sanity checks (in order, cheap → expensive)

```powershell
# a) Contract check — runs your estimator on a toy MLP, verifies shapes/types (~seconds)
uv run whest validate --estimator estimator.py

# b) Real scoring on the public mini split (100 MLPs; downloads ~850 MB once)
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 --split mini --runner local

# c) Grader-parity mode (isolated subprocess, like the real thing)
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 --split mini --runner subprocess
```

What to look for in (b)/(c): `adjusted_final_layer_score` (expect ~2–3e-6 for our v0), `mean_compute_utilization` (expect <2%), `n_failed_mlps` (must be 0). If `validate` complains about any `fnp.` function name, tell Claude — one-line fixes.

## 5. Submit 🚀

```powershell
uv run whest submit --estimator estimator.py --watch
```

`--watch` streams grading progress. Afterwards, grab the **run report** (shown in the CLI / on the AIcrowd submission page) and paste it into the Claude session — it contains `admin_config` values (budget, wall-time cap, λ) that confirm Phase 2's exact settings.

## 6. Optional but recommended

```powershell
# Git for versioning (if not installed):
winget install --id Git.Git -e
cd $env:USERPROFILE\Documents\arc-whest
git init && git add -A && git commit -m "v0: calibrated covariance propagation"

# The official starter kit, for reference/docs (kept OUTSIDE our repo):
cd $env:USERPROFILE\Documents
git clone https://github.com/AIcrowd/whest-starterkit.git
```

## Troubleshooting quickies

- `uv: command not found` → reopen terminal (PATH refresh).
- `whest login` 302/redirect error → an old bug (forum thread 18031); make sure you clicked **Participate** on the challenge page and re-accepted rules for the current phase.
- Dataset download slow/fails → retry; it's Hugging Face-hosted (Xet). `uv run whest dataset download aicrowd/arc-whestbench-public-2026 --revision v1-phase1` prefetches.
- Score 10× worse on grader than local → you fell off the 0.1 multiplier floor (over 10% budget) or a wall-time surprise; send Claude the run report.
- Windows Defender/SmartScreen on uv installer → it's the official Astral installer (astral.sh); allow it.
