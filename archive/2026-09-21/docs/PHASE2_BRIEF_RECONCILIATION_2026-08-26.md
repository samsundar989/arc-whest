# PHASE2_BRIEF_RECONCILIATION_2026-08-26.md

> **HISTORICAL — superseded by `CURRENT_STATE.md`, `GAME_PLAN.md`, and
> `PHASE2_FRONTIER_AUDIT_2026-08-31.md`.** Do not execute present-tense actions in
> this file. Since it was written, Rules v14 resolved the two-designation/default
> rule, submissions #328662 and #328664 graded with zero failures, full K3 measured
> `4.0047e-8` on DESIGN (GT-noise `1.155e-9`) and `3.9249e-8` on organiser CONFIRM
> (GT-noise `7.478e-11`), and the active program became W-EFF price/parity followed
> by label-free layer-age diagnostics and a task-specific carrier proxy. This file
> preserves the 2026-08-26 audit trail only.

**A dated evidence report, not a roadmap.** `docs/GAME_PLAN.md` remains the only
active strategy; `docs/CURRENT_STATE.md` remains the only status snapshot. This
file records what happened when the externally-produced Phase-2 mission brief of
2026-08-26 was checked, item by item, against this repository and against the
installed official stack — what it corroborates, what it adds, what it gets
wrong, and what it left unsupported.

It exists because the brief opens by asserting priority over repository documents.
That assertion cannot be accepted or rejected wholesale: parts of the brief are
independently correct and genuinely new here, one part is a measured correction to
a durable repository record, and one part is an architecture pivot resting on
evidence that **does not exist on this machine**.

---

## 1. What the brief gets right, and what this repository had already established

Every Phase-2 constant the brief lists was re-asserted today against the installed
package by `local_sim/official_config.py` — all PASS:

| constant | brief | asserted against whestbench 0.16.0 / flopscope 0.12.0 |
|---|---|---|
| shape | 1024 x 16 | 1024 x 16 |
| budget `B` | `2**41` | `2,199,023,255,552` |
| lambda | 0 | `0.0` |
| residual | 0.4 s, gated not priced | `0.4`, `residual_mode = 'gated'` |
| wall | 120 s | `120.0` |
| memory | 8 GB | `8192` MB |
| setup | 5 s | `5.0` |
| multiplier floor | 0.1 | 0.1 |

**~~One live constraint the brief omits.~~ WITHDRAWN 2026-08-26 late — this was
my error, and the repository already had it right.** The first version of this
section claimed `ContestSpec.predict_timeout_s = 30.0` is a binding per-MLP
ceiling the brief had omitted. **It does not bind.** `whestbench/runner.py`
computes the per-MLP response timeout as

```python
max(limits.predict_timeout_s, limits.wall_time_limit_s + _PREDICT_RESPONSE_GRACE_S)
```

with `_PREDICT_RESPONSE_GRACE_S = 5.0`, so under Phase 2 it is
`max(30.0, 120.0 + 5.0) = 125.0 s` and **the 120 s wall is what binds**. Verified
by calling the function directly. `local_sim/phase2_manifest.py` had already
recorded exactly this, with the mechanism, and
`local_sim/p1pre_fault_inject.py` already carries a check asserting it — I quoted
the class default without reading the code that consumes it.

A strategy review independently reached the same conclusion from graded evidence
(a Phase-2 submission with COMPLETED per-MLP rows at wall ~78.5 s) and proposed
raising the local runner's `predict_timeout_s`. **That change should NOT be
made.** The local runner calls the same package function and already computes
125 s; raising the local value would move us *away* from grader parity, not
toward it. Nothing to fix — the correct action is no action.

The brief's leaderboard reading is confirmed. Re-crawled today: **50 entrants**
(48 at the 2026-08-25 snapshot), leader unchanged at `9.50e-9` adjusted /
`3.10e-8` raw. The brief's inference that the leader runs at multiplier ~0.31 is
arithmetically identical to the repository's independently derived `0.306`.

The account state is confirmed and, for the first time, **captured rather than
transcribed** — see §5.

The brief's judgement that sampling-primary is dead at this shape agrees with the
repository's own measurement, and its reading of the score identity is correct.

---

## 2. What the brief adds that this repository did not have

### 2a. The `flops.stats.*` float32 promotion — and it corrects a durable record

The brief warns that every `flops.stats.*` call silently promotes float32 to
float64, doubling downstream billing, and that the result must be cast back
immediately. **Verified directly today**: `flops.stats.norm.cdf` on a float32
input returns float64 and flopscope itself emits a `FlopscopeWarning` saying so.

This matters far beyond a style note, because `docs/CURRENT_STATE.md` carries this
durable claim:

> Coupled residual sampling is deprioritized. Its former 1.305x projection assumed
> a float32 K2 cost reduction that the frozen artifact does not show: measured
> cost fell only 3.1%, not 50%.

`local_sim/p2_arms/arm_k2p1_f32.py` calls `norm.cdf` and `norm.pdf` and **never
casts back**. From layer 0 onward its `Phi` and `phi` are float64, and so is every
expression that touches them — including `outer(Phi, Phi) * cov_pre * off`, where
essentially all the cost lives. The arm is float32 in name only.

Measured today, one network, `local_sim/p2_f32_price_probe.py`
(`experiments/2026-08-26_p2_cal/f32_price_probe.json`):

| arm | counted FLOPs | vs f64 | state dtype | rel. mean-square divergence on the scored row |
|---|---:|---:|---|---:|
| f64, the shipped spelling | `137,562,584,064` | 1.0000x | float64 | — |
| f32 as `arm_k2p1_f32.py` is written | `133,266,036,224` | **0.9688x** | **float64** | `1.0e-16` |
| f32 with the two stats results cast back | `68,782,589,440` | **0.5000x** | float32 | `3.3e-13` |

The middle row **reproduces the recorded 3.1% exactly**. The recorded measurement
was correct; the label on it was not. One line — `Phi = Phi.astype(fnp.float32)`
and the same for `phi` — recovers the full 2x.

The accuracy price is `3.3e-13` relative mean square on the scored row, against a
final-layer error we are fighting at roughly `5e-5` relative. That is **eight
orders of magnitude below the signal**, i.e. free — on one network. Whether it is
free on all 96, and whether any network drives the closure into a float32 failure
mode, is **not** established here and must be measured before anything is built
on it.

**This does not reopen the coupled-residual-sampling lane.** Reopening a closed
lane requires a contract with a stated reopening condition (`AGENTS.md`). What is
established is narrower and should be recorded as such: the cost half of that
lane's premise was measured on a mislabelled arm, so the reason given for closing
it no longer stands on its own.

### 2b. What the doubled headroom is actually worth

The Phase-2 multiplier is `max(0.1, F/B)`, so the first `0.1 * B =
219,902,325,555` counted FLOPs are **free**: spending them cannot move the score
except through accuracy. The shipped Phase-2 path spends `137,600,326,656` of
them (grader spelling) and leaves the rest unused.

| path | counted FLOPs | free FLOPs still unspent |
|---|---:|---:|
| shipped `_analytic_only`, f64 | `137,600,326,656` | `82,302,000,000` |
| the same closure, f32 with stats cast back | `~68,800,000,000` | `~151,100,000,000` |

A dense antipodal Kerdock/MUB basis at width 1024 costs exactly `2**36 =
68,719,476,736` (Phase 1A, exact). So the free zone holds **1.2 dense bases today
and 2.2 after the cast-back**. That is the whole practical content of §2a: not a
score improvement in itself, but roughly one extra free basis of room for whatever
family ends up winning.

### 2c. Rules and process items worth acting on

Genuinely new or sharper than the repository's record, and all of them
consequential:

- **Up to TWO submissions may be nominated** for the private re-run; the default
  is the two best public. `AGENTS.md` already flags that the Rules tab contradicts
  this and that the contradiction is unresolved on the forum. Plan for two,
  confirm before the deadline.
- **Team/registration freeze 2026-10-02**, up to 5 people. Currently solo. This is
  Samuel's decision and is surfaced, not taken.
- **Contribution prize**: PDF plus **exactly one successfully-graded submission
  ID**, due **2026-10-24**.
- **Winners open-source within 30 days.** Phase-1 winners' forced releases land
  around 2026-08-31 and are free method intel — worth a diary entry.
- **Enforcement continues after grading; invalidation can be retroactive.**
  Reinforces the existing prohibition on accounting games.
- **Borderline techniques can be pre-cleared by email** and written confirmation
  relied on.
- **10 submissions per team per UTC day.** See §5 — this specific number is still
  **not captured** from any live source.

---

## 3. What the brief asserts without evidence that exists here

### 3a. The benchmark bundle is not present

The brief's §3 — the measurements it calls "decisive", and the sole quantitative
basis for its architecture — refers to `kprop_bench_bundle.zip` as attached, to be
dropped into `experiments/kprop_bench/`. **It is not in this repository, and a
search of this machine found no file of that name, nor any `kprop_bench`
directory.** The only kprop artifact here is
`local_sim/mlp_kprop_py310.patch`.

So every figure in the brief's §3 — the K3 accuracy, the K3 FLOP breakdown, the
47.8% / 47.5% split between `get_dslice` and `contract_W`, the K2 calibration
result — is at present an **unreproduced external number**. They may well be
right. They are not evidence in this repository's sense, and `AGENTS.md` does not
let them become the basis of a lane decision until the artifacts arrive.

**Action: ask Samuel for the bundle.** It is the single cheapest thing that would
change the strategic picture.

### 3b. The K3 accuracy figure sits inside its own noise floor

Independent of the missing artifact, the brief's own table is self-limiting. It
reports K3 at `5.83e-8` mean final-layer MSE, marked "GT-noise-corrected", against
a stated ground-truth noise floor of `8.0e-8`. The corrected value is therefore
**smaller than the correction applied to it**: the raw reading was near `1.4e-7`
and more than half of it was subtracted as noise. The brief says as much and calls
for a remeasurement at `N = 1e8`.

That is the right call, and until it happens the number cannot carry an
architecture. For contrast, this repository's P01 truth runs `N = 2**22` per
replica with two independent replicas, giving a noise floor near `2.1e-8` against
a `4.1e-6` signal — **0.5%** — plus a two-replica cross-product estimator that is
unbiased by construction rather than by subtraction.

### 3c. The two candidate architectures, priced on the identity that decides

The brief's target is a rank-compressed factorized K3. This repository's active
plan is a structured Kerdock/MUB angular carrier with an analytic control branch.
Both are unmeasured at Phase-2 shape *by us*. Stating them on the score identity
makes the actual bet visible.

To beat the live leader's `9.50e-9`:

| operating point | multiplier | raw final-layer MSE needed |
|---|---:|---:|
| inside the free zone (`F <= 2.199e11`) | 0.100 | `< 9.50e-8` |
| at the leader's own cost (`F = 6.74e11`) | 0.306 | `< 3.10e-8` |
| at full budget | 1.000 | `< 9.50e-9` |

- **Compressed K3.** Claimed uncompressed: `5.83e-8` raw at `1.48 B`. At the floor
  that would score `5.83e-9` and win — **but only if a ~15x cost compression is
  accuracy-neutral**, which is exactly the thing no measurement anywhere yet
  addresses. The brief's own gate G1 asks for 5x at 1.5x accuracy loss; 5x
  compression lands at `~0.30 B`, i.e. leader parity, not a win. The plan's
  headline outcome and the plan's own success criterion are not the same point.
- **Structured carrier.** Cost is exact and known: `k * 2**36`. Accuracy at this
  shape is unmeasured. Phase 1A established the carrier is a valid real MUB and
  priced it exactly; it also measured a carrier-aligned fourth moment `21.69x` the
  spherical target, so validity does not imply the error behaves.

Neither is dismissible on today's evidence and this file does not choose between
them. What it records is that **both bets are about cost compression, and the
cheapest FLOPs available to either are the ones already inside the free zone** —
which is why the calibration slice (§4) came first.

### 3d. Where the brief's plan restates work that is already closed here

The brief's W0 asks for an environment upgrade, a Phase-2-truthful local scorer,
and a FLOP self-audit. All three exist and are closed: P2-P0 pinned the official
stack, replaced the Phase-1 lambda default, and reconciled the local meter to the
official runner **op by op, to zero** — under an independent ACCEPT. The brief's
suggested parity targets (2% cost, 5% score) are looser than what is already
achieved.

The brief's W0 also asks for an immediate canary submission. That is Track 0S,
which exists, is prepared, and is **blocked on the two items in §5** — not on
engineering.

---

## 4. The first thing the brief was right about that nobody here had tried

The brief's A0 is calibrated covariance propagation. The repository had per-layer
calibration for the Phase-1 width-256 architecture and **none at all at width
1024**: `estimator.py:867` `_analytic_only` runs the raw closure.

That gap was pre-registered as **P2-CAL** (`experiments/LOG.md`, 2026-08-26,
committed before any data existed) and measured the same day. The result and its
independent review are in the log; the headline is that an in-loop per-layer gain
costs `16,384` counted FLOPs — `1.2e-7` of the free zone — and is therefore free
in every sense that matters to the score.

---

## 5. Track 0S: one blocker moved, one did not

The submission was procedurally rejected for two reasons. Their state today:

- **Live signed-in account recheck — now CAPTURED.** An authenticated
  `whoami` + `eligibility` call returned, for `neopok919` (id 513543):
  `submissions_allowed: true`, `denied_reason: null`, `rules_accepted: true`,
  `participation_terms_accepted: true`, Phase 2 (round 1429) the active round,
  closing `2026-10-17T23:59Z`. Saved verbatim as
  `experiments/2026-08-26_p2_live/account_eligibility_capture.json`. This is a
  captured API response, not a transcription.
- **The 10-per-team-per-UTC-day allowance — STILL NOT CAPTURED.** No AIcrowd API
  endpoint reachable from here exposes a per-day count or a remaining balance, and
  the challenge rules page renders client-side, so an unauthenticated fetch
  returns chrome and no rules text. The figure remains transcribed. **The honest
  status is unchanged: this half of the blocker stands**, and closing it needs a
  browser session, which this environment does not have.

The second blocker — a focused independent ACCEPT on the corrected record —
is unaffected by anything in this file.

---

## 6. Standing corrections this file makes to durable records

1. `docs/CURRENT_STATE.md`'s "float32 K2 measured only 3.1%, not 50%" is
   reproduced and **explained**: the arm measured was float64 downstream of
   `flops.stats`. The correctly-cast arm measures exactly `0.5000x`. The sentence
   should be amended where it stands, and the coupled-residual-sampling lane's
   closure reason noted as resting on that mislabelled measurement.
2. The mission brief's `predict_timeout_s` omission (§1) should be carried into
   any deployment envelope that currently plans against 120 s per MLP.
3. The brief's §3 numbers must be marked SPECULATIVE wherever they are cited
   until the bundle arrives (§3a).

---

*Produced by the repo-local Builder (Claude Opus 5, 1M context), 2026-08-26.
Raw artifacts: `experiments/2026-08-26_p2_live/`, `experiments/2026-08-26_p2_cal/`.*
