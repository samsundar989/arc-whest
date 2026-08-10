# CURRENT_STATE.md — read this first

*Snapshot, not narrative. Last updated **2026-08-09** (submission 327743 graded).
Update this file when a lane opens or closes, a number is graded, or the phase
changes — not on every experiment. The narrative record is `experiments/LOG.md`.*

---

## The clock

| event | when | status |
|---|---|---|
| Round 1428 "Phase 1" | closes **2026-08-10 23:59 UTC** | **LIVE.** Deadline conflict RESOLVED 2026-08-09: the API round object itself moved from Aug 7 to Aug 10, matching organizer thread 18125 |
| Round 1429 "Phase 2" | opens **2026-08-18 00:00 UTC**, closes **2026-09-19 23:59 UTC** | UPCOMING, empty board, no `v1-phase2` HF tag. **Opening moved Aug 8 → Aug 18** (observed 2026-08-09): there is now a 7-day gap after Phase 1, not a same-day handover |
| Write-up due | **2026-08-17** (forum 18109) | the contribution-prize deliverable |
| Registration / team freeze | 2026-09-05 | |
| Results | 2026-10-01 | private re-run of ONE designated submission on fresh MLPs |

**Phase-2 opening is event-triggered, never date-triggered.** Poll with
`uv run python local_sim/phase2_checklist.py`. The runbook is at the tail of
`experiments/LOG.md` (§ *Phase-2 opening runbook*). First action on 1429 going
active is **frozen v2.4.1 for configuration telemetry, and only with Samuel's
explicit approval.**

---

## Where we stand

| instrument | tier | value | date |
|---|---|---|---|
| **graded, submission 327743 (v2.4.1 — THE SHIPPED FILE)** | graded | **2.6314e-7** adjusted · 1.3323e-6 MSE · mult 0.19896 · residual 10.022 ms/MLP · **0 of 100 failed** | 2026-08-09 |
| graded, submission 324420 (v2.4) | graded | 2.7194e-7 adjusted · 1.3748e-6 MSE · mult 0.1990 · residual 9.845 ms/MLP | 2026-08-05 |
| graded, 324094 (v1.1) | graded | 3.893e-7 | 2026-08-05 |
| vault, 500 sealed MLPs, v2.4.1 | VAULT | median **2.4651e-7** · mean 3.2669e-7 · p99 2.0535e-6 · max 3.0469e-6 · 0 non-finite | touch #1, 2026-08-06 |
| public mini (100), v2.4.1 | CONFIRM | 3.172e-7 adjusted · 1.4895e-6 MSE · mult 0.216 · 0 failed | 2026-08-06 |
| invariant suite on the shipped file | — | **30/30** | verified 2026-08-07 |
| board position | — | raw #90/200; **#82 of 192 plausible-honest** on 5 entries | 2026-08-06 10:03Z |

**Shipped file**: `estimator.py` == `estimator_v2_4_1.py` (byte-identical modulo
line endings — `scripts/preflight.ps1` asserts this). **v2.4.1 is now graded**
(327743, 2026-08-09, Phase 1): 0 failures, safe to designate. Vault touch counter: **1**.

**v2.4.1 vs v2.4 is a DRAW, not a 3.2% improvement.** The headline adjusted score
fell 2.7194e-7 → 2.6314e-7, but both were scored on the *same* 100 networks
(`mlp_name` lists identical), so the paired statistic applies and it says no effect:
geometric mean of per-MLP ratios **0.9665×, 95% CI [0.7969, 1.1721]**, v2.4.1 better
on **22 of 50** (sign test p = 0.839). **One network, `mlp_index 37`, contributes
140% of the entire aggregate delta.** Counted FLOPs agree to 0.02% and layers 1–31
are bit-identical; only the scored final layer differs. Quote v2.4.1 as *equivalent
to* v2.4, never as better. Full analysis: `experiments/LOG.md` 2026-08-09.

Field anchors: best published honest method 1.551e-7 · our declared honest sampling
frontier ~2.7e-7 · GT noise floor of the metric ~2e-10. Full table: `docs/BENCHMARK.md`.

---

## Closed lanes — do not reopen without new *external* evidence

Each was killed by a measurement, not by a preference. The reopening condition, if
any, is stated. Reopening one is STRATEGIC work and needs a Slice Contract.

| lane | verdict | gate vs result | evidence |
|---|---|---|---|
| Analytic family (B1 fusion / B2 tail / B3 learned residual) | **DEAD** | fusion net ~0 · tail −67…−214% · R²=0.012 | GAME_PLAN rev 3 |
| Track T — shrinkage-moment tail | **CLOSED** | 1.10× gate → 0.9998× | LOG 2026-08-05 |
| Track P — rank-k deviation propagation | **CLOSED** | 1.25× gate → 1.093× even with a free oracle | LOG 2026-08-05. Break-even rank = width/2 |
| Sprint 3 — structure-aware FLOP allocation | **FAIL** | → 1.0246× | LOG 2026-08-05 |
| Sprint 4/5 — packed activation sparsity | **KILLED** | pre-registered 1.15× → **1.1216×** [1.1188, 1.1244] on D14 | LOG 2026-08-06. D12 and **D14 are SPENT — never reuse** |
| N as an operating-point lever | **CLOSED** | `MSE×N` flat over {16k, 32k, 65k} at 8 shifts | the 14% "optimum" was one heavy-tailed realisation |
| Batched depth-2 Strassen | **OFF** | bitwise identical, −53% ops, **worse** residual | shipped behind `_STRASSEN_BATCHED`, tests kept |
| Score optimisation generally | **CLOSED at ~2.7e-7** | GAME_PLAN rev 6 | draw variance dwarfs any remaining ~2% gain |

**The mechanism behind the packed lane's death, which generalises**: flopscope
prices a gather at 4 FLOPs/element, and packing's entire saving is substituting
gathers for dense multiply-adds. A random-access gather costs far more *wall time*
per element than a BLAS multiply-add, and wall time is charged back at λ = 1e11.
The lane arbitrages the price table against physics, and loses.

---

## Open questions, ranked

1. **319341's variance advantage — the only open score question.** Their `MSE × N`
   is 0.018520 against our graded 0.022525: **1.216×**. Ruled out: arithmetic,
   packing, antithetic sampling. It is a *variance* mechanism and we do not know
   which one.
2. **Moonshot G1 — blocked on the D10 renormalization fit.** G0 (done 2026-08-07)
   proved the layer convention, built the registered calibrated-K2 baseline
   (depth 8 → 1.2067e-05; G1 must reach ≤4.02e-06), and read the oracle: full-rank
   K3 flow beats the control **12.786×**, but the pre-registered r ∈ {4, 8, 16} is
   *worse* than the control and the ≥3× gate needs r ≈ 128 = width/2 — exactly
   where Track P found rank-k factorisation stops paying. **Next step is the D10
   renormalization fit, not D08.** D08/D09 untouched.
3. **Phase-2 configuration.** Shape, `n_mlps`, budget, λ, floor are all unverified
   for round 1429. `wall_time_limit_s` has never been observable.
4. **The write-up (due Aug 17).** Sprint 5's negative result is unusually clean
   contribution material: a pre-registered gate, a conservative model frozen in
   advance that was later shown to be load-bearing, and a transferable mechanism.

---

## Standing risks

- **Uncommitted work — PARTIALLY resolved, and the remainder is load-bearing.**
  The *experiment* record landed across five commits through `7040f41` (mint
  provenance, vault, invariant suite, Sprints 4–5, Moonshot G0, records), so LOG
  entries can cite commits again. **What did not land is the cross-agent harness
  itself**: `AGENTS.md`, `ai/`, `.agents/`, `.codex/`, `scripts/guard_paths.py`,
  `scripts/verify_portability.py`, `scripts/git-hooks/`, `scripts/human_model_state.py`,
  `scripts/install_git_hooks.py`, `.claude/hooks/guard_hook.py`, plus modifications to
  all six SKILL.md adapters, `CLAUDE.md` and `README.md` — all **untracked** as of
  2026-08-09. `7040f41` committed only the Claude-side skills.
  This is sharper than the original risk: `CLAUDE.md` `@`-imports `AGENTS.md` as the
  operating contract, and `verify_portability.py` reports 174/174 against files that
  exist **only in the working tree**. Losing this tree loses the entire
  agent-independent layer and both agents' enforcement.
- **The cost half of the evaluator is not trustworthy.** See below. **New, 2026-08-09:
  the unreliable half is OUR measurement, not the grader's** — see the transfer-ratio
  note in § *Evaluator trust*.
- **Deadline conflict**, unresolved by the organizers since Aug 5 (thread 18130).
- **The metric has a heavy tail.** Fresh-MLP max/median = 12.9×; the worst 1% of
  MLPs carry 6.9% of the score. At a re-run of size R = 10 the p90 score is 50%
  above the median. Tail risk, not mean improvement, is what the re-run measures.

---

## Evaluator trust — E1/E2, and the split is the point

The score is `MSE × max(0.1, C/B)`, and its two halves are measured to completely
different standards. We are currently at `C/B ≈ 0.20`, well above the 0.1 floor, so
**both halves are live**.

| half | level | why |
|---|---|---|
| **Accuracy (MSE)** | **E2, approaching E3** | mint proven bit-identical to the grader (100/100, uint32); seed-block registry with machine-readable single-use blocks; gates pre-registered in LOG before the block is touched; paired common-random-number statistics with CIs; 32-shift averaging; every MSE quoted against its GT-noise floor; 30/30 invariants including permutation equivariance; touch-counted sealed vault |
| **Cost (`C` = counted FLOPs + λ·residual seconds)** | **E1 — directional only** | counted FLOPs are exact and reproducible. Uninstrumented wall time is not measurable on this machine: it bills the same file several× what the grader bills it. Every dense-vs-non-dense cost comparison therefore rests on a *model*, and the deciding model is a frozen assumption, not an observation |

**The transfer ratio is path-dependent — established 2026-08-09 (submission 327743).**
The grader side turns out to be the *stable* half: **9.845 ms/MLP (v2.4) → 10.022
ms/MLP (v2.4.1)** across four days and two files, paired ratio 1.0180× with a 95% CI
of [1.0119, 1.0241] over 100 networks. The local side is what moves. The **same
byte-identical file** billed **70.8 ms/MLP under `--runner local`** (2026-08-06) and
**42.94 ms/MLP under `--runner subprocess`** (2026-08-09) — putting local/grader at
**7.06× or 4.28×** purely as a function of which local path was used.

`AGENTS.md` § *Measurement law* states "~7×" as binding. That figure came from the
`local` runner; against the `subprocess` runner — which is what the grader uses — it
is ~4.3×. **Every cost verdict that scaled a local residual by ~7 may have used a
factor 65% too large, including Sprint 4 S2 and the D14 residual model.**

**Not yet established.** The two local readings differ in runner *and* in machine
state (Aug 6 vs Aug 9); one slice cannot separate them. **Do not edit the ~7× figure
in `AGENTS.md` until the discriminating run is done:** the same file, same mini
split, both runners, back to back, same day. It costs no seed block, no vault touch
and no submission, and it gates the credibility of every cost number in the write-up.

**Consequences, binding:**

- **No autonomous / Ralph-style optimization loop.** The framework rule is
  evaluator ≥ E2 before bounded autonomy; the composite here is not.
- Cost claims follow the three-model rule (`docs/VALIDATION.md` §2b): report three,
  freeze one *before* the decision block, decide on
  `max(local-ratio-on-graded-anchor, competitor-anchored)`. The `max` is
  load-bearing — the competitor anchor alone would have passed a carrier that
  did not deserve it.
- **Improving the cost evaluator outranks any further score optimization.** The only
  instrument that would do it is a graded run, which costs a submission.

---

## Harness — what is enabled and why

| capability | layer chosen | bottleneck it removes |
|---|---|---|
| Evaluation | the project's own scripts (`edge_cases`, `mint_provenance`, `vault`, `whest run`) | already best-in-class; wrapped, not replaced |
| Change gate | `scripts/preflight.ps1` | four commands that must all run on every estimator change, previously prose in `VALIDATION.md` §7; adds the ship-file identity check that existed nowhere |
| Pre-registration | skill `slice-contract` | the repo's #1 historical failure mode: a residual model or gate chosen after seeing the answer (Sprint 4 twice) |
| Independent falsification | agent `skeptic` | two "findings with tidy causal stories" turned out to be noise; the one independent audit (rev 11) corrected four claims |
| Orientation | this file + `docs/GOTCHAS.md` | a fresh session previously had to read ~150 KB across 12 stacked GAME_PLAN revisions to learn what was decided |
| Human comprehension | `docs/HUMAN_MODEL.md` (Samuel-owned, write-protected from agents) | no comprehension checkpoint existed anywhere in the record |
| Prompt friction | `.claude/settings.json` allowlist | read-only lab commands prompted on every invocation |
| **Agent portability** (audit 2026-08-07) | `AGENTS.md` + `ai/workflows/` + shared `scripts/` enforcement | the whole harness lived under `.claude/`. Exhausting one agent's quota meant losing the methodology, not just the session |

**The portability layer, in one paragraph.** `AGENTS.md` is now the cross-agent
operating contract and `CLAUDE.md` imports it, so no project behaviour is
agent-specific. The six workflow commands are defined once in `ai/workflows/`;
`.claude/skills/` and `.agents/skills/` (Codex) are thin adapters that load them, and
`scripts/verify_portability.py` (162 checks) fails if either side drifts, grows a rule
of its own, or names a product where it should name a role. The Skeptic exists for
both agents (`.claude/agents/`, `.codex/agents/`) over one shared brief. Enforcement is
one implementation, `scripts/guard_paths.py`, called by both agents' hooks *and* by
`scripts/git-hooks/pre-commit` — the backstop that holds even for an agent whose hooks
nobody has wired. **Switching drivers is: start the agent in the repo root, run
`session-start`, continue with `next-slice`.** Not verified in this environment: the
Codex hook *registration* schema, because Codex is not installed here — see
`.codex/hooks/README.md`. The pre-commit backstop covers the gap meanwhile.

**Deliberately not enabled** — each was considered and rejected for a stated reason:

| rejected | reason |
|---|---|
| Ralph / autonomous loops | cost half of the evaluator is E1. Forbidden until a graded anchor exists |
| MLflow / Phoenix / Langfuse | ~30 decisive experiments total, and the artifacts already exceed what these capture. None can model a single-use seed block or a touch-counted holdout — which are the properties that actually protect this project |
| Context7 | two pinned dependencies; the real API surface is already verified against the installed package and recorded |
| Serena | ~70 source files; `rg` + Read is sufficient |
| Repomix | the repo is local; subagents read it directly |
| Promptfoo | there is no LLM in the product |
| Playwright / Chrome DevTools / design skills | no UI anywhere |
| Exa / Firecrawl / last30days | the research surface is one forum and one leaderboard, already scripted (`board_snapshot`, `board_forensics`, `board_census`, `phase2_checklist`) |
| GitHub MCP | `git` + `gh` cover it |
| Symphony / Agent Deck / workflow orchestration | one operator, one working tree, evidence-gated work. Parallel agents here would create merge and provenance risk, not throughput |
| Google Drive / Slack / Gmail / Calendar MCPs | irrelevant to this repo; Gmail and Calendar are also unauthenticated in this environment |

---

## Next actions

1. **The runner-vs-machine-state discriminating run.** Same unmodified
   `estimator.py`, same mini split, `--runner local` and `--runner subprocess` back
   to back on the same day. Resolves whether the transfer factor is ~4.3× or ~7×.
   Free, and it gates every cost claim in the write-up. **Highest value in the
   repository right now.**
2. **`review-slice` on submission 327743** — the Skeptic has not yet seen the
   2026-08-09 slice.
3. **Write-up** — due **Aug 17**, eight days out, co-equal with the score lane. Two
   new pieces of material landed 2026-08-09: the aggregate-vs-paired dissolution on
   real grader data (with the responsible network named), and the path-dependent
   transfer ratio.
4. **Phase 1 closes Aug 10 23:59 UTC**; Phase 2 does not open until Aug 18. Decide
   before Aug 10 whether any further Phase-1 submission is wanted — after that there
   is a seven-day window with no live round.
5. **D10 renormalization fit** if the Moonshot lane continues; do not touch D08.
6. ~~Samuel fills in `docs/HUMAN_MODEL.md`.~~ **Done 2026-08-09.**
