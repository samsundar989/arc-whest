# CURRENT_STATE.md — read this first

*Snapshot, not narrative. Last updated **2026-08-10** (submission 327743 graded).
Update this file when a lane opens or closes, a number is graded, or the phase
changes — not on every experiment. The narrative record is `experiments/LOG.md`.*

---

## The clock

| event | when | status |
|---|---|---|
| Round 1428 "Phase 1" | closes **2026-08-10 23:59 UTC** | **LIVE.** Deadline conflict RESOLVED 2026-08-10: the API round object itself moved from Aug 7 to Aug 10, matching organizer thread 18125 |
| Round 1429 "Phase 2" | opens **2026-08-18 00:00 UTC**, closes **2026-09-19 23:59 UTC** | UPCOMING, empty board, no `v1-phase2` HF tag. **Opening moved Aug 8 → Aug 18** (observed 2026-08-10): there is now a 7-day gap after Phase 1, not a same-day handover |
| Write-up due | **2026-08-17** (forum 18109) | the contribution-prize deliverable |
| Registration / team freeze | 2026-09-05 | |
| Results | 2026-10-01 | private re-run of UP TO TWO designated submissions on fresh MLPs (forum 18143; Rules tab still says one — unresolved) |

**Phase-2 opening is event-triggered, never date-triggered.** Poll with
`uv run python local_sim/phase2_checklist.py`. The runbook is at the tail of
`experiments/LOG.md` (§ *Phase-2 opening runbook*). First action on 1429 going
active is **frozen v2.4.1 for configuration telemetry, and only with Samuel's
explicit approval.**

---

## Where we stand

| instrument | tier | value | date |
|---|---|---|---|
| **graded, submission 327743 (v2.4.1 — THE SHIPPED FILE)** | graded | **2.6314e-7** adjusted · 1.3323e-6 MSE · mult 0.19896 · residual 10.022 ms/MLP · **0 of 100 failed** | 2026-08-10 |
| graded, submission 324420 (v2.4) | graded | 2.7194e-7 adjusted · 1.3748e-6 MSE · mult 0.1990 · residual 9.845 ms/MLP | 2026-08-05 |
| graded, 324094 (v1.1) | graded | 3.893e-7 | 2026-08-05 |
| vault, 500 sealed MLPs, v2.4.1 | VAULT | median **2.4651e-7** · mean 3.2669e-7 · p99 2.0535e-6 · max 3.0469e-6 · 0 non-finite | touch #1, 2026-08-06 |
| public mini (100), v2.4.1 | CONFIRM | 3.172e-7 adjusted · 1.4895e-6 MSE · mult 0.216 · 0 failed | 2026-08-06 |
| invariant suite on the shipped file | — | **30/30** | verified 2026-08-07 |
| board position | — | raw #90/200; **#82 of 192 plausible-honest** on 5 entries | 2026-08-06 10:03Z |

**Shipped file**: `estimator.py` == `estimator_v2_4_1.py` (byte-identical modulo
line endings — `scripts/preflight.ps1` asserts this). **v2.4.1 is now graded**
(327743, 2026-08-10, Phase 1): 0 failures, safe to designate. Vault touch counter: **1**.

**v2.4.1 vs v2.4 is a DRAW, not a 3.2% improvement.** The headline adjusted score
fell 2.7194e-7 → 2.6314e-7, but both were scored on the *same* 100 networks
(`mlp_name` lists identical, `sampling_mse` identical to 17 digits), so the paired
statistic applies: geometric mean of per-MLP ratios **0.9665×, 95% CI [0.7969,
1.1721]**, better on **22 of 50** (exact p = 0.839). The aggregate delta is a small
residue of near-cancelling sums (+1.91e-5 of gains against −1.70e-5 of regressions);
three networks each individually flip its sign. Counted FLOPs agree to 0.02%, and
layers 0–30 are exactly float-equal on **all 50** networks — only the scored final
layer differs. **Quote v2.4.1 as *equivalent to* v2.4, never as better.**

*Precisely: "no **resolvable** effect." At n = 50 with sd(log ratio) = 0.696 this
design can only exclude |effect| > ~20%. The better-powered answer already existed
and agrees — 0.9959× over 32 draws/MLP (`docs/GOTCHAS.md`). The grader run confirmed
it; it did not discover it.* Full analysis and the independent audit that corrected
three claims in this section: `experiments/LOG.md` 2026-08-10.

Field anchors: best published honest method 1.551e-7 · our declared honest sampling
frontier ~2.7e-7 · GT noise floor of the metric ~2e-10. Full table: `docs/BENCHMARK.md`.

---

## Shipped file — v2.6 BUILT, NOT PROMOTED (2026-08-19)

`estimator.py` is **v2.6** (Strassen-Winograd carrier), archive
`estimator_v2_6.py`, preflight **GREEN 5/5**, invariants **30/30**, vault
touches still **1**. **It has not been gated, not been submitted, and no
decision block has been spent.**

The graded number of record is still **v2.4.1 = 2.6314e-7** (submission 327743).

**Why it is held.** The deciding quantity is the grader's residual price per
operation, which this document classifies **E1 — directional only**, and the
same byte-identical paired comparison reads:

| | cost ratio | net V.c |
|---|---|---|
| grader per-op price 7.0 us/op | 1.0990x | **1.0926x** |
| transfer the local residual ratio (conservative, quote this one) | 1.0740x | **1.0677x** |
| **this machine, as measured** | 0.9577x | **0.9522x** |

The sign flips between the local harness and the grader, because local residual
runs **4.37x** the grader's (42.99 ms/MLP vs 9.845) while the op count transfers
exactly (1,417 local vs 1,416 graded). Break-even is 55.5 us/op; the grader is
at 7.0, this machine at 30-51. **Only a graded run can decide it.** MSE is
settled and costs 0.51% (paired 0.9942x [0.9870, 1.0013]), all of it the
even-width trim — Strassen itself contributes nothing (LOG 2026-08-19).

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
| **Control variates (layer-1 exact-mean, LINEAR forms only)** | **DEAD** | **1.010×, 95% CI [0.979, 1.042]** on our Sobol+digital-shift path — CI includes 1.0. On plain MC the same correction gives 1.383× and the fitted optimum 1.584×: QMC already removes that variance | Skeptic probe 2026-08-11, 14 MLPs × 6 shifts, paired geo-mean, CONFIRM tier. **Covers recursive/per-layer forms** — any correction telescoping to a linear map of the layer-1 sample-mean error is in this class. **Does NOT cover** corrections using an *empirical* per-layer gain (`mean(z>0)`), which are nonlinear and unmeasured. The older "1.80× standalone" is a ratio of means over 3 MLPs and should not be quoted |
| Score optimisation generally | ~~CLOSED at ~2.7e-7~~ **REOPENED 2026-08-10** | — | GAME_PLAN rev 6's frontier described *our method*, not the field's. Board decomposition shows honest teams 4.8× better on MSE at our compute. See § *Open questions* #1 |

**The mechanism behind the packed lane's death, which generalises**: flopscope
prices a gather at 4 FLOPs/element, and packing's entire saving is substituting
gathers for dense multiply-adds. A random-access gather costs far more *wall time*
per element than a BLAS multiply-add, and wall time is charged back at λ = 1e11.
The lane arbitrages the price table against physics, and loses.

---

### Added 2026-08-19 — the carrier audit (LOG 2026-08-19)

| lane | verdict | number | note |
|---|---|---|---|
| **Point-set whitening (full 256x256 ZCA)** | **DEAD on our carrier** | **0.9972x, CI [0.986, 1.009]** | 24 MLPs x 24 shifts, paired, GT 2^23. Worth **1.23x on iid** — the device works, our Sobol already collects it. The competitor "2.46x" is a rung against an iid+antithetic baseline; that author's own ablation reads 0.97x / 0.994x in both orders and the census records 1.05x |
| **Diagonal moment standardisation** | **DEAD, and it was always a no-op** | **1.0000x, CI [0.999, 1.001]** | Our shipped Sobol's diagonal second moment is already **107x better than iid**; its off-diagonal is only 5.9x better. The 2026-08-05 kill and the radial-imbalance predictor both probed the already-exact half |
| **ReLU-homogeneity sphere projection** | **DEAD on our carrier** | 1.0010x, CI [0.998, 1.004] | Exact and free, but Sobol already balances the radius. Worth +4% *inside* a design carrier |
| **Tent / baker fold** | **DEAD alone** | 0.9810x, CI [0.928, 1.037] | Only `tent+white` clears, at **1.0931x [1.044, 1.148]** — the one surviving input-space combination |
| **Degree-3 structured design carrier** | **DEAD against the shipped stack** | **1.047x, CI [0.979, 1.120]** | Bare it is 1.182x [1.092, 1.290]; the rank-32 rotation collects it. **Does NOT close a true degree-4/5 design** (strength-4 OA / Kuperberg cubature), whose external ceiling is ~1.46x |
| ~~Strassen-Winograd~~ | **REOPENED** | **1.267x counted (depth 2), MSE-exact; ~1.10-1.16x net after residual** | Measured on the real flopscope 0.10.0. Our 2026-08-06 negative was on the *packed* v2.5 carrier (tiny irregular blocks). Now the largest bankable lever in the project |

**The governing mechanism, measured three ways today: SUBSTITUTION.** Whitening
vs Sobol, design vs rotation, and (previously) CV vs QMC. Two devices that
annihilate the same low-order structure do not compose — whichever is added first
takes the credit. **Read every competitor ladder for what its baseline arm
contains, never for its largest rung.**

**New standing number: 7.0 us/op = 695,268 FLOP-equivalents per grader operation**
(from 327743: 1,416 ops, 9.845 ms). Price any counted-FLOPs-for-op-count trade
before building it.

**Correction owed to a durable doc.** `docs/writeup-draft.md` and
`docs/RESEARCH_NOTES.md` attribute the 2.35x moment-standardisation regression to
"a first-order bias". Moment matching's bias is **O(1/N)** (Glasserman 4.5.1
p.245) and diagonal forcing measures at exactly 1.000x here and in forum 18151.
The number may be real; the mechanism is not. **Do not ship it.**

## Open questions, ranked

1. **THE ACCURACY GAP — reopened 2026-08-10 on new external evidence, and it is much
   larger than we thought.** Decomposing the public board via
   `score / scoreSecondary = C/B` shows honest teams are **2–3.6× better on score and
   4.8–10× better on raw MSE**. The decisive case is **J2W: C/B 0.194 against our
   0.198 — the same compute — with 4.8× lower MSE.** No operating-point confound.
   Teams with 18–52 entries (too few to probe the public set meaningfully) sit at
   2–3.6× better scores. Spending more compute does *not* close it: our N=65,536 arm
   projects to **0.999×** on the grader's own cost anatomy.
   **This supersedes "319341's 1.216× variance advantage"** — that was a corner of a
   much bigger gap. It also means `GAME_PLAN` rev 6's "honest sampling frontier
   ~2.7e-7" described *our method's* frontier, not the field's. Mechanism unknown;
   MSE alone cannot distinguish better variance reduction from a better analytic
   core. **Strategist Packet issued 2026-08-10.** Full analysis: `experiments/LOG.md`.

   **It is a VARIANCE gap, not a bias gap — MEASURED 2026-08-14 at DECIDE tier,
   question closed.** Block D16, 200 fresh canonical mints × 8 shifts, paired CRN,
   GT 2²⁴ samples/MLP, label noise **computed and subtracted**:

   | | |
   |---|---|
   | α (MSE ∝ N^−α) | **0.9928**, 95% CI [0.9586, 1.0274] |
   | σ_gt² (label noise) | 3.0925e-9 |
   | B² **after** subtracting it | **2.3727e-9**, 95% CI **[−2.46e-8, 2.84e-8]** — includes 0 |
   | **B² as share of MSE** | **0.574% → the residual is 99.4% VARIANCE** |

   **57% of the previously inferred "bias floor" was the noise in its own ground-truth
   labels** — σ_gt² and B² are perfectly confounded in an MSE-versus-N fit
   (`E[(pred−truth−η)²] = E[(pred−truth)²] + σ_gt²`), so any such fit that does not
   subtract σ_gt² is measuring its own labels. Artifact:
   `experiments/bias_floor_D16.json`; instrument: `local_sim/bias_floor_lab.py`.

   **Consequence:** J2W's 4.8× lower MSE at our compute is **4.8× lower variance per
   sample**. The next Strategist Packet targets **variance mechanisms**, and must
   carry the full kill list — the 2026-08-10 round trip happened because the
   control-variate lane was omitted from it.
   *(An earlier note here inferred the opposite — "probably mostly bias → attack the
   closure core" — from forum 18085's "≥58% bias". That figure describes
   radiant-allomancer's own estimator, 4.47e-7, not ours. Withdrawn 2026-08-11,
   closed by measurement 2026-08-14.)*
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
  2026-08-10. `7040f41` committed only the Claude-side skills.
  This is sharper than the original risk: `CLAUDE.md` `@`-imports `AGENTS.md` as the
  operating contract, and `verify_portability.py` reports 174/174 against files that
  exist **only in the working tree**. Losing this tree loses the entire
  agent-independent layer and both agents' enforcement.
- **The cost half of the evaluator is not trustworthy.** See below. **New, 2026-08-10:
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

**The local residual has no quotable level — established 2026-08-10 (submission
327743), corrected the same day after independent falsification.** Same
byte-identical `estimator.py`, same 100-MLP mini split:

| reading | ms/MLP | vs grader 10.022 |
|---|---|---|
| `--runner local`, 2026-08-06 | 70.85 | 7.07× |
| `--runner subprocess`, 2026-08-06 | **54.47** | **5.44×** |
| `--runner subprocess`, 2026-08-10 | 42.94 | 4.28× |

Runner effect, same day **1.301×**; machine-state effect, runner fixed **1.269×**;
they multiply to the observed 1.65×. So the spread is **half path, half time** — not
"purely the runner", and the overstatement is ~30%, not 65%.

`AGENTS.md` § *Measurement law* states "~7×" as binding and **must stay unedited**
until repeatability is measured. If three same-day repeats span >10%, the correct fix
is not "4.3× instead of 7×" but *"this machine has no quotable absolute residual
level"*, with the line replaced by a range carrying its measurement path.

**What this does NOT contaminate.** The claim that Sprint 4 S2 and the D14 residual
model "scaled by ~7" was **false and is withdrawn** — two different sevens were
conflated. D14's deciding model is *local **ratio** × graded anchor*
(`9.84 ms × 490.4/62.3`); the 7.87 there is a within-run candidate/control ratio,
invariant to any uniform rescaling of this machine, and the local→grader level never
enters. The local-absolute model is explicitly flagged "never decides". **The real
exposure, newly identified: that candidate/control ratio itself reads 5.47× in one
local harness and 7.87× in another — a 44% spread on the quantity that does decide.**

**Grader-side residual: a detected shift, not stability.** Paired 1.0180× [1.0119,
1.0241] excludes 1.0. But n = 1 submission per file, so the n = 100 CI is
pseudoreplication; and `residual_s` is a subtraction remainder whose components moved
~78 ms while the remainder moved 0.177 ms. Do not attribute it to the `eigh` path.

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

1. ~~VARIANCE-OR-BIAS on fresh canonical mints.~~ **DONE 2026-08-14 — 99.4% variance,
   B² CI includes 0.** See § *Open questions* #1. D16 spent (design block).
2. **THE WRITE-UP — due 2026-08-17 23:59 UTC, ~3 days.** Co-equal prize objective and
   now the tightest deadline in the project. `docs/writeup-draft.md` exists and is
   unmodified. Material banked since it was last touched, all of it unusually clean:
   the D14 pre-registered negative result and its price-table-vs-physics mechanism;
   the 327743 aggregate-vs-paired dissolution on real grader data; the path-dependent
   residual ratio; the board decomposition via `score/scoreSecondary`; and this
   slice's label-noise/bias confound. **Nothing here needs a new measurement.**
3. **Next Strategist Packet — variance mechanisms.** Must carry the FULL kill list
   including control variates (the 2026-08-10 round trip happened because it did not).
2. **Local-residual repeatability run.** Three `--runner subprocess` repeats plus one
   `--runner local`, unmodified `estimator.py`, mini split, same day, each saved with
   `--format json`. The runner arm is *already answered from disk* (1.301×); what is
   unmeasured is whether this machine's residual is repeatable at all. If the three
   repeats span >10%, no local absolute level is quotable and `AGENTS.md`'s "~7×"
   becomes a range. ~20 min, no seed block, no vault touch, no submission.
2. **Save the artifact for any local cost number.** The 42.94 ms/MLP reading has no
   saved JSON — it is internally consistent (multiplier 0.2117 with counted 5.33e10)
   but not reproducible by the next session. Contract said a mini-run JSON would be
   produced; it was not.
3. **Write-up** — due **Aug 17**, eight days out, co-equal with the score lane. Two
   new pieces of material landed 2026-08-10: the aggregate-vs-paired dissolution on
   real grader data (with the responsible network named), and the path-dependent
   transfer ratio.
4. **Phase 1 closes Aug 10 23:59 UTC**; Phase 2 does not open until Aug 18. Decide
   before Aug 10 whether any further Phase-1 submission is wanted — after that there
   is a seven-day window with no live round.
5. **D10 renormalization fit** if the Moonshot lane continues; do not touch D08.
6. ~~Samuel fills in `docs/HUMAN_MODEL.md`.~~ **Done 2026-08-10.**
