# Windows setup → Phase-2 validation (copy-paste guide)

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

Since 2026-08-24 the pins are the **official Phase-2 bake stack**: `whestbench 0.16.x` / `flopscope 0.12.x`. `preflight` step 1 refuses to run against anything else, so a stale environment fails the gate loudly instead of producing ten green steps against the wrong round — which is exactly what it did for three days.

Two auxiliary venvs, both gitignored, both optional:

| venv | contents | why it exists |
|---|---|---|
| `.venv-fs010` | flopscope 0.10 only | keeps the 0.10-vs-0.12 price comparison reproducible now that `.venv` has moved to 0.12. Only `p2_flop_anchor.py`'s two-sided run needs it; its `--gate` mode does not. |
| `.venv-gpu` | torch + CUDA, no whestbench/flopscope | the GPU truth mirror. Uses the numpy shims in `local_sim/`, so it must never measure counted FLOPs. |

Create the first with:

```powershell
uv venv .venv-fs010 --python 3.10
$env:VIRTUAL_ENV=".venv-fs010"; uv pip install "flopscope==0.10.0" "numpy==2.2.6"
```

## 3. Log in to AIcrowd (one time)

```powershell
uv run whest login
```

Paste the API key when prompted.

## 4. Sanity checks (in order, cheap → expensive)

```powershell
# a) Contract check — runs your estimator on a toy MLP, verifies shapes/types (~seconds)
uv run whest validate --estimator estimator.py

# b) Grader-parity Phase-2 smoke on one public-mini MLP
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini --n-mlps 1 --runner subprocess

# c) Full public-mini CONFIRM only after the candidate is frozen
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini --runner subprocess
```

What to look for in (b)/(c): `adjusted_final_layer_score`, exact counted FLOPs,
`residual_wall_time_s < 0.4`, and `n_failed_mlps = 0`. The current K2 path is only
an operational smoke and is not competitive. Never fit a constant or select a
candidate on the public mini split.

## 5. Submission is an approval gate

Do not keep a copy-paste submission command in the quick-start path. Submission
requires full preflight, official Phase-2 validate, a clean subprocess mini with
zero failures, a frozen candidate hash/evidence packet, and Samuel's explicit
approval. Follow `AGENTS.md` and `docs/GAME_PLAN.md` §6, *Promotion, deployment,
and nomination*. This setup guide is not submission authorization.

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
- Dataset download slow/fails → retry; it's Hugging Face-hosted (Xet). `uv run whest dataset download aicrowd/arc-whestbench-public-2026 --revision v2-phase2` prefetches.
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
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini --runner subprocess
# Submission command intentionally omitted; use the explicit approval gate.
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
uv run python local_sim/mint_provenance.py       # prove canonical_mint == whestbench.sample_mlp (1024x16 first)

# Phase-2 evaluator truthfulness (P2-P0, 2026-08-24)
uv run python local_sim/official_config.py                # the LIVE round, asserted against the installed package
uv run python local_sim/p2_flop_anchor.py                 # 3 arms x 2 flopscope versions; needs .venv-fs010
uv run python local_sim/p2_flop_anchor.py --gate          # op-by-op parity only; ONE venv (what preflight runs)
uv run python local_sim/test_deployment_envelope.py       # what fails an MLP, AND what this host cannot enforce
uv run python local_sim/test_deployment_envelope.py --no-write   # ...and this really writes nothing (P1-PRE)
uv run python local_sim/test_envelope_readonly.py         # proves that --no-write claim from OUTSIDE the suite
uv run python local_sim/p2_residual_paths.py --n-mlps 20  # 3 residual instruments; QUIET MACHINE ONLY
uv run python local_sim/phase2_manifest.py                # rebuild the manifest (package parity, live constants)
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

---

### 7.1 Heavy work goes through `gpuq` — the submit lines and declared footprints

*Added 2026-09-01. `~/.claude/CLAUDE.md` is the machine-wide policy; this is what
it means in this repository. Wiring lives in `.gpuq.toml` at the repo root, and
`AGENTS.md` § Commands carries the one-paragraph version.*

This box brokers expensive work across concurrent agents through `gpuq`, which
gates on **host RAM and CPU as well as VRAM**. Here RAM binds long before VRAM
does: the organisers' `mini` split is **~6.9 GB of float32 weights** across seven
parquet shards (100 MLPs x 16 x 1024 x 1024 x 4 B), and the residual soak reads
it three times. Submit heavy work and **keep working** — submission returns a job
id immediately, and a queued job runs the source exactly as it was at submission
time, so the tree can be edited straight away.

#### Three traps that are specific to this repository

1. **An artifact written to a relative path is LOST.** A queued job runs in a
   frozen git worktree under the gpuq state directory. `experiments/<slice>/` is
   tracked, so it exists *inside the snapshot*, and a report written there is
   deleted when `gpuq cleanup` expires the snapshot — while `experiments/LOG.md`
   in the live tree cites a path that never existed. **Always give a heavy job an
   absolute output path**, e.g.
   `--json C:/Users/samsu/Documents/arc-whest/experiments/<slice>/<name>.json`.
2. **`whest submit` and `vault.py evaluate` are NEVER submitted to the queue.**
   Their confirmation guards in `scripts/guard_paths.py` are wired into the agent
   tool layer (`.claude/hooks/`, `.codex/hooks/`) and into `pre-commit`. A gpuq
   job spawns processes outside all three, so the queue is a hole straight
   through the enforcement layer — and a vault touch written into a snapshot
   **spends the touch without recording it in the live touch log**, which
   `preflight.ps1` #6 then reads as a counter/log mismatch. Both are STRATEGIC,
   both need Samuel, and neither is heavy. Run them directly.
3. **Gitignored paths are invisible to a job unless `.gpuq.toml` lists them.**
   All four venvs (`.venv-gpu` is ignored by its own uv-written `.gitignore`),
   `local_sim/sobol_dirnum.npz`, the 3.2 GB `weights_P01.f32` cache and every
   truth `*.npz` are gitignored, and `local_sim` scripts resolve them from
   `_ROOT` — which is the *snapshot* root. **Add any new one to `.gpuq.toml`**,
   or the job dies with "No such file or directory". A passthrough is a junction
   to the LIVE path, so a job that writes through one writes into the real
   repository. A directory entry whose destination already exists in the snapshot
   is silently skipped, which is why the `experiments/` entries are per-file. The
   16 GB HuggingFace cache is outside the repo and needs no entry.

#### Declared footprints

`--ram`/`--vram` are **peak GiB, estimated high on purpose**: an under-declared
job is admitted when it should have waited, which is how this box falls over.
Over-declaring only makes a job wait slightly longer.

| workload | `--ram` | `--vram` | `--cpus` | basis |
|---|---:|---:|---:|---|
| `certify_candidate.py --residual-soak 3` | 20 | 0 | 6 | 3 x the row below, sequentially; ~3 h at the measured 37.7 s/MLP |
| `whest run --split mini --runner subprocess` | 20 | 0 | 6 | **MEASURED 2026-09-01**: harness `PeakWorkingSet64` **14.32 GiB** + a ~1.67 GiB estimator worker = ~16 GiB. See below |
| `linux_memory_gate.py` | 12 | 0 | 2 | **deliberately** requests 9 GiB in a WSL child to prove the 8,192 MB cap is in force |
| `p2_truth.py` | 8 | 6 | 4 | chunk `2**16` x 1024: fp32 draw + matmul + fp64 cast ~1.1 GiB, plus CUDA context |
| `p2_cal_gpu.py` | 8 | 8 | 4 | 3.2 GB fp32 memmap streamed a layer at a time in fp64 |
| `gpu_mint.py --mint 50000 --samples 65536` | 16 | 8 | 4 | writes a 497 MB npz; large host-side accumulation |
| `gpu_train_probe.py --data <npz>` | 12 | 6 | 4 | loads that 497 MB npz and trains on it |
| `robustness.py --mlps 1200 --truth 4194304` | 12 | 8 | 4 | 1200 fresh MLPs with `2**22`-sample ground truth |
| `vault.py mint --samples 33554432` | 8 | 6 | 4 | 500 MLPs x 256x32, `2**25` samples each |
| other GPU labs (below) | 8 | 6 | 4 | same 256x32 / 1024x16 chunked shape |

The mini-run figure is the one number here that was measured rather than
derived, because it was the one I could not read off the code — and it landed
well above a naive estimate. Three things it settled:

* **The dataset load is not the cost.** 6.3 GB and 100 MLPs resolve from the HF
  cache in **1.3 s**. The RAM is the harness holding scored state, not parquet.
* **Peak is reached in the first minutes and then flat** (13.29 GiB working set
  at MLP 18 and again at MLP 25; peak 14.32 GiB). So a future re-measurement
  wants `--n-mlps 5`, not a full split — the run measuring this was cancelled at
  25/100 once `PeakWorkingSet64` had stopped moving, because the remaining ~50
  minutes of a concurrency-1 queue would have bought nothing.
* **An 8 GiB declaration would have been badly wrong**, and the undeclared
  default of 4 GiB catastrophically so. That is the `docs/GOTCHAS.md` § host-commit
  trap reproducing itself exactly.

Local cost is **37.7 s/MLP** on this host with the official subprocess runner
(measured over 25 MLPs), which is what makes the soak a ~3-hour job.

"Other GPU labs" is `coverage_lab.py`, `bias_floor_lab.py`, `audit_constants.py`,
`packed_truth.py`, `track_t_mint.py`, `track_p_spectrum.py`, `sprint3_probe.py`,
`moonshot_g0.py`. Raise `--vram` to 8 when `--n-gt` goes above `2**22`.

#### The submit lines

```powershell
# The residual soak: 3 x 100 MLPs on the organisers' data. Rung-3 hardening.
gpuq submit --project arc-whest --ram 20 --vram 0 --gpus 0 --cpus 6 -- `
  C:/Users/samsu/Documents/arc-whest/.venv/Scripts/python.exe local_sim/certify_candidate.py `
  --candidate estimator_v2_7.py --residual-soak 3 `
  --json C:/Users/samsu/Documents/arc-whest/experiments/<slice>/certify.json

# One full official mini run. CONFIRM tier: it may CHECK, it may CHOOSE nothing.
gpuq submit --project arc-whest --ram 20 --vram 0 --gpus 0 --cpus 6 -- `
  C:/Users/samsu/Documents/arc-whest/.venv/Scripts/whest.exe run `
  --estimator estimator.py `
  --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 `
  --split mini --runner subprocess

# The 8,192 MB enforcement gate (WSL).
gpuq submit --project arc-whest --ram 12 --vram 0 --gpus 0 --cpus 2 -- `
  C:/Users/samsu/Documents/arc-whest/.venv/Scripts/python.exe local_sim/linux_memory_gate.py `
  --candidate estimator.py `
  --json C:/Users/samsu/Documents/arc-whest/experiments/<slice>/linux_memory_gate.json

# Phase-2 truth at 1024x16, two independent replicas. CLAIMS SEEDS.
gpuq submit --project arc-whest --ram 8 --vram 6 --cpus 4 -- `
  C:/Users/samsu/Documents/arc-whest/.venv-gpu/Scripts/python.exe local_sim/p2_truth.py `
  --block P02 --n-mlps 48 `
  --out C:/Users/samsu/Documents/arc-whest/experiments/<slice>/truth_P02.npz

# In-loop K2 calibration fit; streams the 3.2 GB float32 weight memmap.
gpuq submit --project arc-whest --ram 8 --vram 8 --cpus 4 -- `
  C:/Users/samsu/Documents/arc-whest/.venv-gpu/Scripts/python.exe local_sim/p2_cal_gpu.py

# B3 mint and its learnability probe.
gpuq submit --project arc-whest --ram 16 --vram 8 --cpus 4 -- `
  C:/Users/samsu/Documents/arc-whest/.venv-gpu/Scripts/python.exe local_sim/gpu_mint.py `
  --mint 50000 --samples 65536

# Fresh-MLP score distribution.
gpuq submit --project arc-whest --ram 12 --vram 8 --cpus 4 -- `
  C:/Users/samsu/Documents/arc-whest/.venv-gpu/Scripts/python.exe local_sim/robustness.py `
  --mlps 1200 --truth 4194304
```

Absolute interpreter paths are used throughout: a relative one would resolve through the snapshot's
`.venv` junction, which works only if the passthrough succeeded — a needless
dependency for the one thing that must always start.

#### What still runs directly

`preflight.ps1` and `preflight.ps1 -Full`: step 16's official smoke is
`--n-mlps 2`, not the full split, and the whole `-Full` gate is ~11 minutes well
under 4 GiB. Also `whest validate`, `official_config.py`,
`p2_flop_anchor.py --gate`, `seed_blocks.py`, `vault.py verify`, the `local_sim`
`test_*.py` files, `edge_cases.py`, and every doc/figure/board script — plus the
two commands trap 2 keeps off the queue.

#### Inspecting

`gpuq status` (`--json` to parse) · `gpuq top` · `gpuq show <id>` ·
`gpuq logs <id> [--follow]` · `gpuq report` · `gpuq cancel <id>`.

A QUEUED job is usually waiting deliberately and `gpuq status` prints the reason.
**Queue concurrency is 1 on this box**, so a long foreign job delays ours even
when ours needs no GPU. That is the design, not a fault: never work around it by
running the command directly, and never bypass the queue because `nvidia-smi`
looks idle — host RAM is the binding constraint and the queue is the only thing
that sees the whole machine. If a submission fails, it failed on purpose: report
it, run `gpuq doctor`, do not run the command yourself instead.
