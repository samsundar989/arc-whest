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


---

## 7. Full command reference

*Moved here 2026-08-07 from `CLAUDE.md` so the top-level agent file stays a map.
Commands for lanes that are **killed** (Sprint 4/5 packed sparsity) are kept for the
record — do not rerun a spent decision block. Verdicts: `docs/GOTCHAS.md`.*


```powershell
uv sync                                          # env (Python 3.10 + flopscope + whestbench)
uv add --dev numpy scipy                         # dev-only deps for local_sim tools
uv run whest validate --estimator estimator.py   # contract check (seconds)
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 --split mini --runner subprocess
uv run whest submit --estimator estimator.py --watch
uv run python local_sim/validate_estimator.py    # offline held-out MSE check
uv run python local_sim/fit_calibration.py       # refit calibration (writes experiments/calibration.json)
uv run python local_sim/board_snapshot.py --top 30 --label <tag>   # archive a leaderboard (+ --diff a.json b.json)
uv run python local_sim/board_forensics.py                          # implied V*c, entry counts, score timeline
uv run python local_sim/twin_diff.py --all                          # suite identity across our graded reports
uv run python local_sim/phase2_checklist.py                         # rounds / dataset tags / boards
uv run python local_sim/export_split.py --split mini --n 100 --out <npz>   # dataset -> npz for the GPU venv
.venv-gpu\Scripts\python.exe local_sim/track_t_mint.py --data <npz> --out <npz> --draws 256
.venv-gpu\Scripts\python.exe local_sim/track_p_spectrum.py --data <npz> --mlps 100 --jensen --out <npz>
.venv-gpu\Scripts\python.exe local_sim/sprint3_probe.py --data <npz> --mlps 100   # on/kink FLOP allocation
.venv-gpu\Scripts\python.exe local_sim/robustness.py --mlps 1200 --truth 4194304  # fresh-MLP score distribution
.venv-gpu\Scripts\python.exe local_sim/robustness.py --stress                     # atypical-MLP safety
uv run python local_sim/edge_cases.py            # contract + INVARIANTS on the SHIPPED file (30/30)
uv run python local_sim/mint_provenance.py       # prove canonical_mint == whestbench.sample_mlp
uv run python local_sim/seed_blocks.py           # seed-block registry (allocate before any experiment)
uv run python local_sim/private_half.py          # public-private divergence self-check
uv run python local_sim/board_census.py          # honest-team census (PRIVATE)
.venv-gpu\Scripts\python.exe local_sim/gpu_parity.py                       # mirror bit-parity + fwd tolerance
.venv-gpu\Scripts\python.exe local_sim/vault.py verify                     # seal integrity + touch log
.venv-gpu\Scripts\python.exe local_sim/vault.py evaluate --label <v> --reason estimator-change
.venv-gpu\Scripts\python.exe local_sim/audit_constants.py --which all --mlps 200 --draws 32

# Sprint 4 — packed activation propagation (runs against REAL flopscope, not the shim)
.venv-gpu\Scripts\python.exe local_sim/packed_truth.py --block D11 --mlps 200   # labels for a decision block
uv run python local_sim/packed_sparse_probe.py --mode parity  --mlps 8          # S0: packed == dense
uv run python local_sim/test_packed_sparse.py                                    # 62/62 focused tests
uv run python local_sim/packed_sparse_probe.py --mode sweep    --block D11 --cost-only --targets 1.5,3,6 --strassens 0,1,2
uv run python local_sim/packed_sparse_probe.py --mode residual --mlps 3 --repeats 3   # QUIET MACHINE ONLY
uv run python local_sim/packed_sparse_probe.py --mode ablate   --block D11 --mlps 30 --shifts 8
uv run python local_sim/packed_sparse_probe.py --mode decide   --block D12 --residual-ratio <from residual mode>

# Sprint 5 — round-trip compression (LANE KILLED on D14; commands kept for the record)
uv run python local_sim/packed_sparse_probe.py --mode strassen --mlps 6        # A: batched vs recursive
uv run python local_sim/packed_sparse_probe.py --mode pareto  --block D13 --mlps 6 --repeats 3   # B: 24-carrier Pareto
uv run python local_sim/packed_sparse_probe.py --mode nsweep  --block D13 --mlps 24 --shifts 8 --log2ns 14,15,16 --carriers "dense:packing~0"
uv run python local_sim/packed_sparse_probe.py --mode decide5 --block D14 ...  # E: SPENT, do not rerun

# Moonshot G0 (GPU venv). MLP_KPROP_SRC must be set for --flow.
$env:MLP_KPROP_SRC="C:\Users\samsu\Documents\mlp_cumulant_propagation\src"
.venv-gpu\Scripts\python.exe local_sim/moonshot_g0.py --mlps 24 --depth 8 --samples 4194304
.venv-gpu\Scripts\python.exe local_sim/moonshot_g0.py --mlps 8 --depth 8 --flow --flow-kmax 2,3,4 --flow-rank 8
.venv-gpu\Scripts\python.exe local_sim/lowrank_cumulant_flow.py --smoke --rank 4 --width 16
```
