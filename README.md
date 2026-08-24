# arc-whest — Samuel's entry, ARC White-Box Estimation Challenge 2026

**Challenge**: estimate every neuron's expected activation of a random Phase-2
`1024 x 16` ReLU MLP under `N(0,I)` inputs, from the weights, under a FLOP budget.
Lower `final_layer_MSE × max(0.1, FLOPs/2**41)` wins. **Phase 2 is live and closes
2026-10-17 23:59 UTC.** Start with [`docs/CURRENT_STATE.md`](docs/CURRENT_STATE.md),
then execute [`docs/GAME_PLAN.md`](docs/GAME_PLAN.md).
https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026

## Map

| Path | What |
|---|---|
| `docs/CURRENT_STATE.md` | **Where we stand right now**: baseline, closed lanes, open questions, next actions, the clock. **Start here to work.** |
| `docs/GAME_PLAN.md` | **The only active strategy:** target architecture, experiment order, gates, timeline, and completion definition. |
| `docs/CLAUDE_CODE_HANDOFF.md` | Copy-ready prompt for a new Claude Code Builder to execute the plan. |
| `docs/PHASE2_LIVE_AUDIT_2026-08-24.md` | Dated live rules, leaderboard, competitor, literature, and evaluator evidence behind the plan. |
| `AGENTS.md` | The cross-agent repository operating contract. Any coding agent reads this first; `CLAUDE.md` imports it. |
| `docs/SKEPTIC.md` | The independent-falsifier brief. `.claude/agents/skeptic.md` and `.codex/agents/skeptic.md` are thin adapters that load it. |
| `estimator.py` | **The submission** (v0: covariance propagation + offline per-layer calibration). Self-contained; only imports allowed in the grader sandbox. |
| `docs/COMPETITION_BRIEF.md` | Plain-English explainer: the competition, the math, what to learn. **Start here to understand.** |
| `docs/VALIDATION.md` | The measurement protocol and its proofs. Binding on anything measured. |
| `docs/GOTCHAS.md` | Standing operational knowledge — traps that must not be rediscovered. |
| `docs/HUMAN_MODEL.md` | Samuel's own mental model. Agents read and critique it; agents never write it. |
| `scripts/preflight.ps1` | The gate that runs on every estimator change (12 checks; 10 by default ~2-3 min, 12 under `-Full`). |
| `scripts/guard_paths.py` | The one implementation of the protected-path / irreversible-action rules. Both agents' hooks and the git pre-commit backstop call it. |
| `scripts/verify_portability.py` | Checks the harness is still swappable between agents (99 checks). |
| `docs/RESEARCH_NOTES.md` | Intel: rules, leaderboard, forum threads, papers, open questions. |
| `docs/SETUP_WINDOWS.md` | Copy-paste guide: install → validate → first submission. |
| `local_sim/` | Offline lab: numpy shims for `flopscope`/`whestbench` so the *exact* `estimator.py` runs anywhere; MLP generator; MC ground truth; calibration fitting; held-out validation. |
| `experiments/` | Results log + fitted calibration data. |

## Working on this project

**Read `docs/CURRENT_STATE.md`, then `experiments/LOG.md` from the bottom.** That is
the whole orientation procedure. **Switching agents is: start it in the repo root and
have it do exactly that.** No chat history transfers, and none is needed.

*There were six slash-command workflows (`session-start`, `next-slice`,
`review-slice`, `cold-start`, `routine`, `session-close`) plus a `slice-contract`
primitive. They were **removed on 2026-08-21** — they restated what a careful agent
does anyway, and in practice agents read the files rather than invoking them.*

**Two rules survived, because they were the part that actually caught mistakes**, and
they are stated directly in `AGENTS.md` § *How work is done*:

1. **Pre-register before you measure.** Question, hypothesis, mechanism, numbered
   predictions, the gate, the falsifier, the frozen residual model and the seed block
   — appended to `experiments/LOG.md` **and committed** before any data exists, plus
   Samuel's prediction before the result.
2. **An independent Skeptic** after every IMPORTANT or STRATEGIC slice, launched in a
   context that never saw the builder's reasoning, with raw artifact paths rather than
   a prose summary.

One-time setup for the agent-independent guard:

```bash
python scripts/install_git_hooks.py     # pre-commit backstop on protected paths
python scripts/verify_portability.py    # confirm the harness is still swappable
```

## Quick start (Windows)

```powershell
cd $env:USERPROFILE\Documents\arc-whest
uv sync
uv run whest login          # AIcrowd API key
uv run whest validate --estimator estimator.py
# Never submit from a quick-start command. Follow the explicit approval gate in AGENTS.md.
```

Full walkthrough: `docs/SETUP_WINDOWS.md`.

## Phase-2 lab status

```powershell
.\scripts\preflight.ps1
uv run python local_sim/phase2_checklist.py
uv run whest validate --estimator estimator.py
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini --n-mlps 1 --runner subprocess
```

The current preflight/checklist still uses stale package/config assumptions in some
paths. `GAME_PLAN.md` Phase 0 is the required repair before local results can claim
grader parity. The old calibration/Phase-1 experiment commands remain documented in
`SETUP_WINDOWS.md` only as historical lab references.

## Current method and target

The current Phase-2 path is basic Gaussian K2 propagation and is not competitive.
The target is a width-1024 Kerdock/MUB structured angular carrier with antipodes
and exact radial integration, supported by a compositional K2/augmented-K2 analytic
state and a compact checkpoint response closure. The evidence, module interfaces,
experiment gates, and two-candidate finish are in `docs/GAME_PLAN.md`.
