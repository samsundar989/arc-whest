# AGENTS.md — arc-whest

**The cross-agent repository operating contract.** Every coding agent that drives this
repository obeys this file. Agent-specific files (`CLAUDE.md`, `.claude/`, `.codex/`)
are thin adapters around it and carry no project behaviour of their own.

*Rev 2026-09-21. This replaces the rules layer that stood until Sep 21 — pre-registered
contracts, seed blocks, DECIDE/CONFIRM/VAULT tiers, per-experiment Skeptic, the 2× gate,
`preflight.ps1` as the change gate, and the 55k-line experiment log. All of that now lives in
`archive/2026-09-21/`. It is history and must not be used for decisions.*

## Objective

Samuel's entry for the **ARC White-Box Estimation Challenge 2026** (AIcrowd account
`neopok919`). Given a random He-init ReLU MLP (Phase 2: 1024 wide × 16 deep, no biases),
predict every neuron's expected post-ReLU activation under x ~ N(0, I) as a
`(depth, width)` array. Score = mean over MLPs of
`final_layer_MSE × max(0.1, FLOPs/2**41)`, lower is better. Phase 2 closes
**Oct 17 23:59 UTC**; team freeze Oct 2; write-up Oct 24. Stack: WhestBench 0.16.1 /
flopscope 0.12.1, λ = 0, residual 0.4 s/MLP, wall 120 s/MLP, memory 8 GiB.

The prize is decided by a **private re-run of up to two designated submissions per team
on fresh MLPs**. Public board rank is not the target. A second, co-equal objective is the
**$20k contribution prize**, whose deliverable is the write-up.

## The rules

1. **Read `docs/PLAN.md` first. It is the only plan.** Every strategy, handoff, packet or
   review document older than 2026-09-21 is archived history. Where anything differs
   from PLAN, PLAN wins. `docs/HANDOFF_2026-09-21.md` turns PLAN into an ordered queue.
2. **Evaluate with `tools/ev.py`. Record one row per candidate in `RESULTS.md`.**
   Never write to the old experiment log (`archive/2026-09-21/experiments/LOG.md`);
   `RESULTS.md` carries the column schema and the live calibration constants.
3. **Submit when it is better.** Samuel has given standing approval (2026-10-12) for an
   agent to run `whest submit` on any **non-designated** candidate it judges better than
   the current best, without asking, **while more than 3 of today's 10 slots remain**.
   The **last 3 slots per UTC day are reserved**: `scripts/guard_paths.py` asks for
   Samuel's approval there, and also whenever the ledger cannot be read. Record every
   submission in `experiments/submissions.jsonl`
   (`python local_sim/submission_ledger.py --record ...`) immediately — the reserve is
   counted from that file, so an unrecorded submission silently spends Samuel's slots.
   An agent may **never** designate or
   nominate a final entry, contact the organizers, post on the forum, or reply to a
   forum request addressed to AI agents.
4. **Own code only.** `external/504aldo/` is a local oracle for parity and diagnostics.
   Never ship, copy or `import` its bytes. Record any mechanism re-implemented from it
   in `docs/PROVENANCE_504aldo.md` (file, line, what).
5. **Data.** The organizer public split `hf://aicrowd/arc-whestbench-public-2026@v2-phase2`
   is unlocked. **DEV** = the 100 mini nets (never fitted) · **TRAIN** = full rows 0–899
   (FIT 0–99 + the former holdout 100–899, authorised Sep 25) · **VAL** = full rows
   900–999 (checkpoint selection; a validation set, not a test). The Sep 21 "HOLDOUT"
   (full 100–999) is historical — `--panel holdout` is refused. Membership:
   PLAN §3, `data/panels/twin_train_split.json`, `tools/panels.py --check`. Fresh mints
   and seed blocks are retired.
6. **Process is minimal.** No pre-registered contracts, no prediction requests, no seed
   blocks, no SINGLE_USE/DECIDE tiers, no dispatch-plan commits, no producer sidecars, no
   instrument preflight, no per-experiment Skeptic, no `workerq`/`gpuq` for evaluations,
   no 2× gate. One evaluator, one results table, one decision rule (PLAN §4). Skeptic
   review is kept **only** for the two designated finals and for write-up numbers, and
   only when Samuel asks.
7. **Commits** are small, frequent and descriptive. One commit per candidate or result.
8. **Report to Samuel** (short message) after: each submission is graded; each deliverable
   completes; any gate fails twice; anything needs his decision. Otherwise keep going. If
   blocked on his input for more than 2 hours, take the conservative option, note it in
   `RESULTS.md`, and continue.

## Grader contract — these are still hard

1. A shipped estimator may import ONLY `flopscope`, `flopscope.numpy as fnp`,
   `whestbench`, and Python stdlib. No numpy/scipy/torch — the grader sandbox lacks them.
   Grader arrays are **immutable** (no `a[i]=`, no `+=` on arrays); pass `str` not `Path`;
   a single array must stay ≤ 4 GiB.
2. Determinism: randomness derives from `mlp.seed` (predict) / `context.seed` (setup) via
   `fnp.random.default_rng(...)`. Never `fnp.random.seed(...)`, never time-based.
3. **Never exploit the flopscope wall-time loophole** (routing math through raw
   uninstrumented arrays). The organizers call it out; we compete clean.
4. Never call `flops.budget_summary_dict()` / `budget_summary()` inside `predict` —
   it is O(op history) and costs a 157× residual blowup.
5. Secrets: the AIcrowd key lives only in `whest login`'s credential store. Never in
   tracked files or logs.
6. Keep the estimator shape-generic (`mlp.width`, `len(mlp.weights)`).
7. No bare `except` that returns a constant. A crash must reach the grader as a FAILED
   MLP, not as an invisible wrong answer; a per-net fallback must be a real predictor.

## Measurement

`docs/VALIDATION.md` remains the reference for *how a number is computed*; its
three-tier protocol, seed blocks and promotion t-gates are retired by rule 6 above.
What binds now:

- **Counted compares to counted; effective compares to effective**, always paired on
  common random numbers. Never quote a bare ratio of means.
- **Every MSE is quoted with its ground-truth noise floor, AT THE LAYER QUOTED.**
  Organizer label noise falls with depth: **≈6.8e-10 at layer 1, ≈7.7e-11 at the
  scored layer 16** (range 4.9e-11–1.1e-10 over 20 FIT nets), measured 2026-09-25 as
  MSE(aicrowd gt, keenanpepper full1000 mean)/2 — two independent N = 1e9 draws
  (`RESULTS.md` § "Anatomy 2"). Layer 1's value is also what our exact closed form
  reads there (6.88e-10 vs analytic 6.82e-10, § "Step 1 (HANDOFF 09-24)" 1a); the
  Sep 24 note that applied it to the FINAL layer was wrong by ~9×. At the scored
  layer the floor is ~0.3% of our raw MSE. For a floor-free MSE use the cross
  estimate mean((p − gt)(p − kp)) (`tools/anatomy2.py`).
- **Phase 2 sets λ = 0, so `C = F`.** Residual is not a cost — it is a 0.4 s per-MLP
  cliff that FAILS the MLP. Project it from **op calls** at 10.8–11.2 µs/op; the local
  in-process `residual_wall_time_s` is an instrument artefact and must never be quoted.
- **Never quote a local absolute wall time as a server figure.** Project server wall
  from the **wall law** (`tools/ev.py` `projected_server_wall_s`): median =
  0.36 ms × op calls + FLOPs × kernel rate (≈92–95 ps/FLOP at Strassen leaf 64,
  164–177 ps at leaf 32); worst-of-100 = median × 1.055 (quiet day) or × 1.24 (noisy
  day). The old ×1.3097 / ×1.5981 local-to-server factors are retired.
- **Predicted public score = DEV 100-net mini score ÷ 1.038** (eight graded
  observations, 1.0339–1.0471, mean 1.0403). Recalibrate if a graded row deviates by >2%.
- Never take FLOPs from the counting shim (it reads ~18% low). Use the real
  flopscope 0.12.1 f64 read-only grader twin, whose count matches the server exactly
  (no offset; only `whest run --split mini` reads 2^25 low — `RESULTS.md` § Calibration).
- The 8,192 MB limit is unenforceable on Windows; the WSL `RLIMIT_AS` gate
  (`local_sim/linux_memory_gate.py`) is the only real test.

`RESULTS.md` holds the current values of all three calibration constants. They are
updated there, not here, after every graded row.

## Commands

```powershell
uv sync                                                    # environment
uv run python tools/ev.py <candidate.py> [--panel dev|fit|holdout]   # THE evaluator
uv run whest validate --estimator <candidate.py>           # contract only (~100 ms)
uv run whest package --estimator <release-dir> --output <pkg>.tar.gz --yes
uv run whest validate-package <pkg>.tar.gz
uv run whest submit <pkg>.tar.gz --description "..." --watch    # PLAN §5 gates only
uv run python local_sim/official_config.py                 # the LIVE round
uv run python local_sim/linux_memory_gate.py               # WSL 8 GiB gate
uv run python scripts/verify_portability.py                # after any harness edit
```

`scripts/preflight.ps1` is **retired from the default path** (rule 6). It still runs and
its checks are still correct; it is simply not a gate any more.
**Run `whest` from PowerShell**, or set `PYTHONIOENCODING=utf-8` in a POSIX shell — the
CLI's box-drawing output kills a cp1252 console inside its own error printer.

Heavy *training / fitting* workloads still go through `workerq` per the machine policy;
**candidate evaluations do not** (rule 6) — they run directly, because the queue bypasses
`scripts/guard_paths.py` and because the loop has to be fast.

## Continuity — no agent may depend on another agent's chat history

| what | where |
|---|---|
| the plan and the ordered queue | `docs/PLAN.md`, `docs/HANDOFF_2026-09-21.md` |
| every candidate, result and graded row | `RESULTS.md` |
| where we stand right now | `docs/CURRENT_STATE.md` |
| traps not to rediscover | `docs/GOTCHAS.md` |
| how a number is computed | `docs/VALIDATION.md` |
| what we re-implemented and from where | `docs/PROVENANCE_504aldo.md` |
| plain-English task explainer | `docs/COMPETITION_BRIEF.md` |
| environment and heavy-job footprints | `docs/SETUP_WINDOWS.md` |
| the human's own beliefs | `docs/HUMAN_MODEL.md` (read-only to agents) |
| the independent-falsifier brief | `docs/SKEPTIC.md` |
| everything before 2026-09-21 | `archive/2026-09-21/` |

**Switching drivers is exactly:** start the new agent in the repository root and have it
read `docs/PLAN.md`, then `RESULTS.md`. If that is not enough, the previous session
failed to record something — a documentation defect, not a reason to read a transcript.

## Enforcement — mechanical, not prose

| invariant | implementation | wired into |
|---|---|---|
| `docs/HUMAN_MODEL.md` is never written by an agent | `scripts/guard_paths.py` | `.claude/hooks/`, `.codex/hooks/`, `scripts/git-hooks/pre-commit` |
| the vault seal is moved only by `vault.py` | `scripts/guard_paths.py` | same three |
| `whest submit` into the last 3 daily slots requires explicit confirmation | `scripts/guard_paths.py` | both agent hooks |
| the harness stays agent-swappable | `scripts/verify_portability.py` | run after any harness edit |

New invariants go into `scripts/guard_paths.py`, where both agents get them at once —
never into `settings.json` or a hook shim. The git pre-commit hook is the
agent-independent backstop; install it with `python scripts/install_git_hooks.py`.
`ARC_WHEST_PROTECTED_OK=1` is the conscious human override.

## Working style

Samuel has a CS degree but is new to ML/probability — explain what you are doing in plain
English in the final summary, not in jargon. Concentrate his understanding on the
decision-critical path (the score identity, the data panels, the residual model); do not
make him learn commodity code.

**Working-directory assumption: every agent is started from the repository root**
(`C:\Users\samsu\Documents\arc-whest`). All paths here are repo-relative.
There are no nested `AGENTS.md` files.

## Agent adapters

| agent | instruction discovery | subagents | hooks |
|---|---|---|---|
| any | `AGENTS.md` (this file) | `docs/SKEPTIC.md` | `scripts/guard_paths.py` + `scripts/git-hooks/` |
| Claude Code | `CLAUDE.md` (imports this file) | `.claude/agents/skeptic.md` | `.claude/settings.json` → `.claude/hooks/` |
| Codex | `AGENTS.md` directly | `.codex/agents/skeptic.md` | `.codex/hooks/` |
