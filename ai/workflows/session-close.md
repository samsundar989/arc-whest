# session-close — arc-whest

*Canonical workflow. Agent-independent. Adapters: `.claude/skills/session-close/`,
`.agents/skills/session-close/`.*

`review-slice` answers *what did that experiment teach us*. This answers *what
survives me closing the laptop*. Run it at the end of a work session, before
switching to another project, **and before switching to a different driving agent.**

The test this workflow applies: **could a different agent, with no access to this
conversation, resume tomorrow from `session-start` alone?** If not, the session is
not closed.

**Do not re-run slice review here.** If a slice is finished but unreviewed, say so and
send Samuel to `review-slice` first — closing a session over an unreviewed strategic
result is exactly how a claim ends up in the write-up without an audit.

## 1. Nothing important exists only in chat

Scan the session for results, numbers, decisions, and surprises that were produced but
never written down. For each, either file it or say explicitly that it was discarded.

| kind | destination |
|---|---|
| a measurement, verdict, or gate result | `experiments/LOG.md` |
| a lane opened or closed, a baseline moved | `docs/CURRENT_STATE.md` |
| a trap that must not be rediscovered | `docs/GOTCHAS.md` |
| a strategy change | `docs/GAME_PLAN.md`, as a **new rev** |
| our numbers or a field anchor moved | `docs/BENCHMARK.md` |
| raw output | JSON/txt under `experiments/` — never pasted into a doc |

This project's rule is that chat is not the experiment database. A number that only
exists in a transcript is not evidence, will not be there next week, and is invisible
to whichever agent drives next.

## 2. `docs/CURRENT_STATE.md` is accurate

It is the first thing the next session reads, whoever is driving. Check and correct:

- **the clock** — round status, and whether the Phase-2 trigger has fired
  (`uv run python local_sim/phase2_checklist.py`)
- **standings** — graded / vault / CONFIRM numbers still current
- **closed lanes** — anything killed today added, with its gate and result
- **open questions** — reordered if priority moved; anything answered removed
- **standing risks** — anything resolved today struck out
- **next actions** — rewritten so the next session knows where to start

## 3. Active hypothesis and open blockers are recorded

One line each: what we currently believe, and what is in the way. If a slice is
mid-flight, record the block it is using, whether that block is single-use, and what
has already been spent — so nobody resumes by re-touching a spent block.

## 4. Working tree and seal state are clean

```powershell
git status --short
.venv-gpu\Scripts\python.exe local_sim/vault.py verify   # only if the vault was touched
```

- Uncommitted work: commit it, or state explicitly why it is being left. The last time
  this repository ran with 38 untracked files, a full day of decisive work was
  uncommitted and no LOG entry could cite a SHA. Do not repeat that.
- Vault touch counter must equal the `touch_log.jsonl` line count. `.\scripts\preflight.ps1`
  check #5 covers this.
- If `estimator.py` was touched at all, preflight must be GREEN before the session
  ends — never leave the shipped file in an unverified state overnight.
- If anything under `ai/workflows/`, `.claude/`, `.agents/`, or `.codex/` was touched,
  `uv run python scripts/verify_portability.py` must be GREEN — otherwise the *other*
  agent starts its next session with a broken control surface.

## 5. Temporary artifacts

- Large intermediate `.npz` caches belong in a scratch directory, not the repo — and
  `*.npz` is gitignored, so an important one must be regenerable from its seeds. Note
  the command that regenerates it.
- `_to_delete/` is untracked scratch. Leave it alone.
- Kill any background runs still going, or note that they are running and where their
  output lands.

## 6. Hand off

Close with a short block Samuel can act on cold — and that a *different agent* can act
on cold:

```text
SESSION CLOSE — <date>

Driver:         <which agent drove this session>
Did:            <2-3 lines, results not activities>
Learned:        <the one thing that changed a belief, or "nothing new">
State:          working tree <clean/dirty>, preflight <9/11|11/11|not run>, vault touches <n>
Open:           <blockers, unreviewed slices, in-flight blocks>
Next session:   session-start, then <the specific next move and why>
```

Naming the driver is not bookkeeping: it is how a later reader knows which
capabilities were in play when a judgement was made. It must never be needed to
*reconstruct* the work — that is what the durable docs are for.

If the human model was filled in or updated this session, note it. If it is still
blank, say so — `session-start` will open in FIRST-ORIENTATION MODE next time, and
that is the correct behaviour, not a problem to work around.
