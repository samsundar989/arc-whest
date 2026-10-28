# arc-whest — Samuel's entry, ARC White-Box Estimation Challenge 2026

## Phase 2 write-up (submission #337601) — start here

- **Write-up (PDF):** [`docs/arc-whestbench-phase2-writeup-337601-neopok919.pdf`](docs/arc-whestbench-phase2-writeup-337601-neopok919.pdf) (source: `docs/writeup/writeup_source.md`). Graded entry **#337601**: adjusted 1.8751e-9, raw final-layer MSE 1.4035e-8 at 13.4 % of budget, 0/100 failures.
- **Graded source:** [`experiments/2026-10-21_L7/release_a5s_soupbc368/estimator.py`](experiments/2026-10-21_L7/release_a5s_soupbc368/estimator.py) — sha256 `180e488bfb2e…`, byte-identical to what was graded; the submitted package is `experiments/2026-10-21_L7/a5s-soupbc368-lam35-180e488b.tar.gz` (sha256 `2ee01ae90fe5…`). Depends only on `numpy`, `flopscope` 0.12.3 and the standard library.
- **Every number in the PDF, with its source file and line:** [`docs/writeup_datapack.md`](docs/writeup_datapack.md). Full ledger: `docs/WRITEUP_LEDGER_2026-10-21.md`. Nominee verification and rerun margins: `docs/DESIGNATION.md`. Per-experiment `RESULT.md` files under `experiments/2026-10-20_*` and `experiments/2026-10-21_*`; graded reports under `experiments/run_reports/`. Math notes: `docs/F_math.md`; architecture census and lever table: `docs/writeup/A_architecture.md`, `docs/writeup/B_levers.md`; sandbox reports (pooled carrier, coupling, split frontier, entry schedule, rank-k state, covariance faces, κ4 ceiling) and pre-registration specs: `docs/writeup/`; sandbox code and JSON results: `sandbox/`.
- **Reproduce the bill:** `python tools/bill_meter.py experiments/2026-10-21_L7/release_a5s_soupbc368/estimator.py` → 293,790,066,123 F / 20,991 ops (flopscope 0.12.3, numpy 2.2.6). FRESH panel tools: `tools/fresh/`; nominee verifier: `tools/verify_nominee.py`. `whest evaluate` on the public panel reproduces the graded score.
- **Provenance:** two mechanisms were re-implemented from 504aldo's public V29 (MIT) with our own fits — see `docs/PROVENANCE_504aldo.md` and PDF §8; the young-tree κ4 reader follows the physics in forum 18249; kai's two-term Q31 law is used as published. No third-party code is copied into this repository.
- **LLM use:** planning, analysis and the write-up text were produced with Anthropic's Claude (Cowork, Fable 5.1); code by Claude Code (Opus 5) and the author, who takes responsibility for every claim. Every change was gated by a pre-registered prediction, a byte-identical zero control, paired VAL/FRESH panels and a graded canary.
- **Branch note:** `writeup-337601` carries the release files on top of the public `main`; the full local development history (≈ 900 further commits) is being pushed to `main` separately.
- **Licence:** MIT (`LICENSE`).

---

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
| `docs/GAME_PLAN.md` | **The active implementation plan:** ideal architecture, module boundaries, work packages, tests, evidence sequence, kill rules, and completion definition. |
| `docs/CLAUDE_CODE_HANDOFF.md` | Copy-ready completion prompt for Claude Code to build every safe truth-free component and a runnable live-shape candidate. |
| `docs/PHASE2_FRONTIER_AUDIT_2026-08-31.md` | Current Phase-2 rules, board, discussion, literature, infrastructure verdict, and evidence behind the plan. |
| `AGENTS.md` | The cross-agent repository operating contract. Any coding agent reads this first; `CLAUDE.md` imports it. |
| `docs/SKEPTIC.md` | The independent-falsifier brief. `.claude/agents/skeptic.md` and `.codex/agents/skeptic.md` are thin adapters that load it. |
| `estimator.py` | **The shipped submission file** (v2.6.1). Self-contained; only imports allowed in the grader sandbox. Two Phase-2 submissions are already public; neither is a winning architecture. |
| `docs/COMPETITION_BRIEF.md` | Plain-English explainer: the competition, the math, what to learn. **Start here to understand.** |
| `docs/VALIDATION.md` | The measurement protocol and its proofs. Binding on anything measured. |
| `docs/GOTCHAS.md` | Standing operational knowledge — traps that must not be rediscovered. |
| `docs/HUMAN_MODEL.md` | Samuel's own mental model. Agents read and critique it; agents never write it. |
| `scripts/preflight.ps1` | The gate that runs on every estimator change (16 checks; 12 by default, all 16 under `-Full`). |
| `scripts/guard_paths.py` | The one implementation of the protected-path / irreversible-action rules. Both agents' hooks and the git pre-commit backstop call it. |
| `scripts/verify_portability.py` | Checks the harness is still swappable between agents (99 checks). |
| `docs/RESEARCH_NOTES.md` | Intel: rules, leaderboard, forum threads, papers, open questions. |
| `docs/SETUP_WINDOWS.md` | Copy-paste guide: install → validate → first submission. |
| `local_sim/` | Offline lab: numpy shims for `flopscope`/`whestbench` so the *exact* `estimator.py` runs anywhere; MLP generator; MC ground truth; calibration fitting; held-out validation. |
| `experiments/` | Results log + fitted calibration data. |

### Document authority

Only three files direct current work: `CURRENT_STATE.md` for status,
`GAME_PLAN.md` for the architecture and implementation order, and
`CLAUDE_CODE_HANDOFF.md` for the completion-oriented Builder prompt. Dated audits
are evidence snapshots, not action plans. `FRONTIER_ARCHITECTURE_PROPOSAL.md` and
`MOONSHOT.md` remain minimal archive redirects only because historical code and
experiment records cite those filenames. The redundant
`PHASE2_FRONTIER_ARCHITECTURE.md` redirect was removed on 2026-08-31; its original
content remains in git history.
`PHASE2_BRIEF_RECONCILIATION_2026-08-26.md` and
`PHASE2_LIVE_AUDIT_2026-08-24.md` are superseded dated evidence. The write-up and
organizer-email drafts are quarantined and must not be quoted or sent.

Old files are not silently deleted: their names are part of the experiment audit
trail. If an archive file conflicts with an active file, the active file wins and
the conflict is a documentation defect to repair.

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
   predictions, gate, falsifier, round-specific counted-FLOP model and seed block
   — appended to `experiments/LOG.md` **and committed** before data exists, plus
   Samuel's prediction. In Phase 2, residual is reported as headroom against the
   0.4-second cliff; it is not part of `C`.
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

The default gate is green: all 12 executed checks pass, including the live official
package/config assertions, 30 edge cases, scorer/seed/vault invariants, archive
identity, and the local official-package FLOP anchor. The four `-Full` checks are
deployment-envelope checks, not evidence already earned. Local operation counts match
the official package's subprocess report exactly; the grading server currently adds a
small `2**24` total-FLOP offset, and Windows cannot enforce the 8 GiB process limit.
The live `v2-phase2` tag has moved to a metadata/prepared-files commit while all 70
source Parquet files remain byte-identical; the checklist pin is stale and requires
a separate IMPORTANT repair before it can be called current.

## Current method and target

The shipped v2.6.1 path is basic Gaussian K2 propagation; the best public v2.7
adds a final scalar calibration. Neither is competitive. Full-tower exact W-EFF
and K2-prefix late-window K3 have both been measured and killed. The active plan
builds two different nominees: (1) an exact-radial structured angular carrier in
the live board's roughly 16k-direction operating envelope, and (2) corrected f32
K2 plus faithful augmented K2 and a query-directed K3 response. The exact module
boundaries, work packages, tests, falsifiers, decision order, and definition of
done are in `docs/GAME_PLAN.md`.
