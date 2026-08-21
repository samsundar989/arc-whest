# routine — arc-whest

*Canonical workflow. Agent-independent. Adapters: `.claude/skills/routine/`,
`.agents/skills/routine/`.*

For work whose success is mechanically checkable. **No contract, no prediction, no
Skeptic, no teach-back.** Ceremony that buys nothing is a tax, and this repository
already carries enough real process.

## In scope

- documentation edits, figure regeneration (`local_sim/make_figures.py`, `figure_data.py`)
- board snapshots and forensics (`board_snapshot.py`, `board_forensics.py`,
  `board_census.py`, `phase2_checklist.py`)
- fetching and filing grader reports (`fetch_graded_report.py` → `experiments/run_reports/`)
- adding tests, fixing a broken script, tidying output formatting
- committing, tagging, housekeeping
- lab scripts that touch **no seed block** and **no shipped file**

## The loop

```text
scope  →  implement  →  project check  →  targeted test  →  evidence
```

1. **Scope.** State in one line what will change and what "done" means. If you cannot,
   it is not routine.
2. **Implement.** Smallest change that does it.
3. **Project check.** Whatever is proportionate:
   - touched anything under `local_sim/` that the shipped path or the invariant suite
     imports → `.\scripts\preflight.ps1` (default tier, `9/11`; `-Full` for 11/11)
   - touched anything under `ai/workflows/`, `.claude/`, `.agents/`, or `.codex/` →
     `uv run python scripts/verify_portability.py` (and note that workflow edits are
     IMPORTANT, not routine — see the escalation table)
   - touched only docs or a standalone probe → just run the thing
4. **Targeted test.** Run the affected script and show real output, not a claim that
   it works. If it has tests (`local_sim/test_packed_sparse.py`, 93/93), run them.
5. **Evidence.** Command, output, files changed. One or two lines in
   `experiments/LOG.md` only if a future session would otherwise be confused — routine
   work usually does not earn a LOG entry.

## Escalate and stop when the task turns out to touch

Discovering this mid-task is normal and is a success, not a failure. Say what you
found, stop, and hand back to `next-slice`.

| trigger | becomes |
|---|---|
| `estimator.py` — any edit at all | **STRATEGIC** |
| allocating or spending a seed block | **STRATEGIC** |
| a vault touch (`vault.py evaluate`) | **STRATEGIC** — and the guard will prompt |
| a submission (`whest submit`) | **STRATEGIC** — needs Samuel's explicit approval |
| a gate, threshold, residual model, or the three-tier protocol | **STRATEGIC** |
| a claim destined for the write-up | **STRATEGIC** |
| reopening a lane marked CLOSED in `docs/CURRENT_STATE.md` | **STRATEGIC** |
| the instrument — `harness.py`, `mint_provenance.py`, `edge_cases.py`, `seed_blocks.py`, `vault.py`, a probe's math, grader-report parsing | **IMPORTANT** |
| the workflow layer — anything under `ai/workflows/`, or an enforcement script under `scripts/` | **IMPORTANT** |
| how any number is computed, anywhere | **IMPORTANT** |
| security or integrity posture (the wall-time accounting rule, credential handling) | **STRATEGIC** |

The instrument line is the one that catches people. A change to `edge_cases.py` or
`mint_provenance.py` looks like a small script edit and is not: a bug there does not
raise an error, it produces a plausible wrong number that survives review. Those get a
written prediction and a green preflight before they land.

## Standing rules that still apply

Even in routine work, none of these bend:

- Never write the AIcrowd key into a tracked file, a log, or `experiments/LOG.md`.
- `docs/BENCHMARK.md` and `RESEARCH_NOTES §2g` are **PRIVATE** — never the forum.
- `docs/HUMAN_MODEL.md` is Samuel's; propose, never edit.
- Never hand-edit `experiments/vault/manifest.json` or `touch_log.jsonl`.
- Run `whest` from PowerShell, or set `PYTHONIOENCODING=utf-8` in a POSIX shell — the
  CLI's box-drawing output kills a cp1252 console inside its own error printer, and
  that has already corrupted one run report.
