# GAME_PLAN.md — Phase-2 winning architecture and execution plan

**Authoritative from 2026-08-24.** This is the only active strategy and roadmap.
`CURRENT_STATE.md` says where execution currently stands; `VALIDATION.md` governs
all measurements; `experiments/LOG.md` is the immutable narrative record. The dated
`PHASE2_LIVE_AUDIT_2026-08-24.md` is the evidence report behind this plan, not a
second roadmap.

Older Phase-1 and pre-launch Phase-2 plans are historical. They must not direct new
work. Their results survive in `experiments/LOG.md`, raw artifacts, and git history.

---

## 1. Objective and live score geometry

Win Phase 2 of the ARC White-Box Estimation Challenge while remaining fully within
the intended flopscope compute rules, and produce a contribution-prize-quality
explanation of the resulting method.

For each random He-initialized, bias-free `1024 x 16` ReLU MLP, predict every
post-ReLU mean under `x ~ N(0, I)`. Phase-2 score is:

```text
mean_MLP[final_layer_MSE * max(0.1, FLOPs / 2**41)]
```

The official live runtime is WhestBench 0.16 / flopscope 0.12 with lambda zero,
a 0.4-second residual cap, 120-second total wall cap, 8-GiB solution-memory limit,
and 5-second setup timeout.

At the live 2026-08-25 snapshot:

| system | adjusted score | raw final-layer MSE | counted FLOPs |
|---|---:|---:|---:|
| Puffi, rank 1 | `1.09e-8` | `3.55e-8` | published `~1.10e12` fingerprint |
| jamespayor, rank 2 | `1.17e-8` | `2.45e-8` | not captured in this snapshot |
| current repo Phase-2 K2 path, measured mini | `4.050566e-7` | `4.050566e-6` | `137,600,326,656` |

There were 46 entrants at the recheck; the `3.93e-7` K2 tie occupied about ranks
35--39, so the measured current path projects to about rank 40. The present gap
is about 37.2x in adjusted score and 114.1x in raw accuracy. The account has no
Phase-2 submission. Incremental 1.05x–1.30x improvements are useful controls but
are not a winning strategy.

### Outcome targets

- **Competitive milestone:** adjusted score below `5e-8`, zero failures.
- **Winning target:** adjusted score below `1.7e-8` on honest offline evidence,
  zero failures, and stable tails.
- **Robustness target:** both final nominees pass official subprocess evaluation,
  memory/code-policy audit, public CONFIRM, and an authorized Phase-2 vault decision.

These are engineering targets, not promotion statistics. Promotion remains the
paired arithmetic score difference returned by `competition_score.decide()` under
the preregistered gate for each slice.

---

## 2. What is known, inferred, and open

### Confirmed

1. The shipped `estimator.py` routes width greater than 256 to the basic analytic
   `k2p1` closure. There is no competitive Phase-2 candidate in the shipped path.
2. On 96 fresh 1024x16 networks, the tested K2p2 covariance-map refinement gained
   about 1.052x for negligible counted-cost change; tested IID and ordinary random
   orthogonal samplers lost badly.
3. P2-CP3R found that 94.61% of K2 closure error is accumulated mean/variance-state
   drift. Marginal skew/kurtosis in the scalar ReLU mean map recovered only 4.47%.
4. CP3R did **not** implement or bound recurrent augmented K2. It reset to true
   marginal moments at each layer; ARC augmented K2 propagates selected joint
   third/fourth harmonic and trace state forward.
5. P01 is a DESIGN block, not a DECIDE block. It has consumed 144/600 seeds and
   cannot support a binding global lane kill.
6. Phase-0 deployment parity is repaired and independently accepted: the official
   stack/config is asserted, local weights match the grader representation, the
   counted-FLOP anchor agrees op by op, and all five failure paths are provoked.
   Three non-gate instrument defects from that ACCEPT remain scheduled as P1-PRE.

### Strong inference

Puffi's exact Phase-2 cost is only 0.0264% above the leading dense cost of 32,768
full propagated directions, and:

```text
32,768 = 16 bases * 1,024 axes * 2 antipodes
```

Combined with Puffi's public Phase-1 Kerdock/Walsh mutually-unbiased-basis carrier,
antipodes, and exact radial integration, the highest-value explanation is a
16-basis width-1024 structured angular carrier. Phase-2 code is private, so this is
a mechanism fingerprint, not a claim of disclosure.

Its observed raw MSE is roughly 74x below the IID variance estimate at the same row
count. That external evidence exceeds the repository's own 2.7x condition for
reopening structured sampling. The earlier random-orthogonal and invalid lattice
arms do not bound this construction.

### Open questions, in priority order

1. Can a legal 16-basis Kerdock/MUB carrier with antipodes and exact radial
   integration reproduce the leader's cost and variance scale?
2. Does faithful recurrent augmented K2 materially repair the propagated state at
   Phase-2 shape while staying operationally safe?
3. Can sampled-versus-analytic checkpoint discrepancies be transported forward by
   a compact response closure?
4. Do remaining errors follow a low-dimensional shared-scale fluctuation that a
   fixed 2–3 component scale-mixture closure can capture?
5. Is a tiny permutation-equivariant, rollout-trained state corrector helpful after
   the mechanistic branches, without fitting public data or seed identities?

---

## 3. Target architecture

The target is a **structured angular carrier with a compositional analytic control
branch**. Sampling and analytic propagation are complements, not competing lanes.

```text
weights
  |
  +--> deterministic structured directions
  |      Kerdock/MUB bases -> antipodes -> exact radial nodes
  |      -> propagate through all 16 layers -> sampled state trajectory
  |
  +--> analytic state backbone
  |      exact layer 1 -> K2/K2p2 Gaussian state
  |      -> selected recurrent augmented-K2 joint state
  |
  +--> checkpoint predict/correct
         sampled state - analytic state at registered layer(s)
         -> compact response transport -> corrected later-layer state
         -> legal final all-layer means
```

### Module A — structured carrier

Purpose: replace `1/sqrt(N)` IID behavior with deterministic angular cancellation
while preserving the Gaussian radial measure.

Required properties:

- deterministic from `mlp.seed` / `context.seed` where randomness is used;
- exactly specified Kerdock/Walsh construction at dimension 1024;
- basis count configurable for research, with 16 bases the primary fingerprint;
- paired antipodes;
- radial integral separated from angular design and validated independently;
- a dense/full-batch float32 reference first (32,768 x 1,024 is 128 MiB per
  point buffer), followed by chunking only if measured memory or residual evidence
  requires it; no single allocation above 4 GiB;
- every meaningful operation through flopscope APIs;
- exact row-count, FLOP, residual, wall, and memory accounting;
- generator invariants: norms, orthogonality, mutual unbiasedness, uniqueness,
  determinism, and known low-degree moment identities.

Primary research arms:

- IID at matched row count;
- ordinary random orthogonal bases;
- Kerdock/MUB at 1, 2, 4, 8, and 16 bases;
- with/without antipodes;
- with exact radial integration versus Gaussian-radius sampling;
- multiple legal scrambles/shifts only if deterministic bias requires them.

### Module B — analytic backbone

Purpose: provide a stable, cheap description of each layer's state and an exact-mean
control/reference trajectory.

Start with K2p2, not the obsolete calibrated Phase-1 path. Preserve a minimal K2p1
fallback for numerical or operational safety.

The augmented branch must faithfully port the ARC joint-state recurrence. It must
not be represented by the existing CP3R marginal oracle. Small-shape tests must
compare every retained state component and layer against the official reference
implementation before Phase-2 evidence is collected.

Track only terms with a documented route into future mean/variance or the final
estimand. Price every spelling through official flopscope 0.12. Dense full K3 is not
an acceptable default: its projected cost is multiple budgets.

### Module C — checkpoint response closure

Purpose: repair accumulated compositional drift using information that the
structured carrier has already measured.

At preregistered checkpoint layers, compute compact differences between sampled
and analytic state. Candidate response features include:

- mean and diagonal-variance discrepancies;
- selected trace/full-trace cumulant summaries;
- shared-scale/norm fluctuation;
- projections onto registered analytic response modes;
- a small finite history of mixed responses if justified by ablation.

Transport the correction through later layers using analytic sensitivities or a
small fixed response basis. Any fitted map must use DESIGN-only networks and be
evaluated on its own rolled-forward trajectory. Teacher-forced one-step accuracy is
diagnostic only.

### Module D — optional compact state models

These are conditional branches, not starting points:

1. **Shared-scale mixture:** a fixed 2–3 node positive scale mixture if residuals
   correlate with layerwise norm/scale fluctuation.
2. **Equivariant corrector:** a tiny shared per-layer update using permutation-safe
   pooled/broadcast features, frozen coefficients, and rollout training.
3. **Sketched goal-directed K3:** only after recurrent augmented K2, and only if a
   sketch-bias test proves the required rank is small.
4. **Scrambled Sobol:** a bounded side test after the Kerdock reproduction. Prior
   QMC evidence was invalid, but high-dimensional nonsmooth QMC has no automatic
   advantage.

### Module E — integration and safety dispatch

The eventual `estimator.py` remains a single self-contained submission file. It
must expose no research dependencies. Offline-generated fixed tables or coefficients
must be embedded in an auditable, deterministic form.

The deployable candidate must:

- return shape `(depth, width)` for every supported MLP;
- keep exact layer-1 handling;
- avoid mutation of grader arrays;
- use only allowed imports and no code-policy loopholes;
- reserve operational headroom for the 0.4-second residual cliff and 8-GiB limit;
- fall back safely on numerical instability or projected over-budget behavior;
- report no in-predict budget summaries;
- pass official validation and subprocess mini with zero failures.

---

## 4. Execution sequence

Each slice is completed and synthesized before the next begins. IMPORTANT and
STRATEGIC requirements in `AGENTS.md` are part of the work, not optional review.

### Phase 0 — make the evaluator Phase-2 truthful — CLOSED

**Class:** IMPORTANT. **First action.** No seed block.

Deliverables:

1. Upgrade main/lock dependencies to WhestBench 0.16 and flopscope 0.12 while
   retaining explicit named Phase-1 replay constants.
2. Add one round-aware official configuration source: Phase 2 uses `B=2**41`,
   lambda `0`, residual `.4`, wall `120`, memory `8192`, setup `5`.
3. Build local MLP weights with official immutable flopscope-backed arrays.
4. Add exact operation/FLOP regressions against the official one-row subprocess
   anchor (`137,600,326,656` FLOPs for current K2p1 on the recorded row).
5. Make official subprocess runs the deployment authority. Label in-process runs
   research-only where memory and process death cannot be reproduced.
6. Test residual-cap failure, timeout, worker restart, native memory, over-budget
   fallback, all output shapes/dtypes, and 1024x16 CPU/GPU truth parity.
7. Refresh the manifest/checklist. Phase-2 preflight must assert the live stack,
   config, shape suite, official mini smoke, and zero failures.
8. Re-label old Phase-2 cost/failure claims provisional where they omitted the cap.

Exit gate:

- all official-stack unit/self-tests green;
- exact FLOP anchor reconciled or every remaining delta explained;
- official Phase-2 validate and subprocess smoke pass with zero failures;
- `scripts/preflight.ps1` and portability check green;
- raw evidence packet in the log;
- independent Skeptic accepts the instrument.

Phase 0 closed on the literal independent ACCEPT recorded in commit `7367fdb`.

### P1-PRE — corrections-only closure of the evaluator instrument

**Class:** IMPORTANT. No seed block, truth data, vault, or shipped-file change.

The three original carried defects are closed in code and survived six independent
attacks. The terminal implementation also closed all six findings from review 5:
15 mutations fire their named guards, the scratch mirror cannot destructively
restore repository files, the behavioural witnesses are pinned, preflight is
16/16, and portability is 99/99.

The sixth review remains PROVISIONAL on four bounded corrections. Finish exactly
these and do not expand the slice into the review's five explicitly non-blocking
meta-design observations:

1. make the harness's default `--out` resolve outside the repository, reject every
   in-repository output path with no tracked-transcript carve-out, include the old
   tracked transcript in the unchanged evidence, and measure repository state
   **after** the output write;
2. correct T7's recorded arithmetic and replace the false “ten of ten as written”
   headline with “nine as written, one corrected”; append corrections—never edit
   the preregistered text or prior result in place;
3. restore injection J, whose guard the reviewer independently confirmed still
   fires, and record the 11-to-15 mutation-set history truthfully;
4. reconcile `CURRENT_STATE.md`, `GOTCHAS.md` and the handoff so no current
   instruction describes the removed destructive path.

Preserve and re-demonstrate the original repairs:

1. replace the literal-`True` worker-restart witness with a structured result from
   the deployment-envelope suite itself; skipping the worker-death section must
   make that witness unprovable/failing rather than green;
2. make `verify_witness` resolve the `suite` it names to that suite's result
   artifact, and add negative tests for unknown suite, missing result, wrong check,
   failed check and detail mismatch;
3. make `--no-write` perform no regeneration and no tracked write, proven by
   before/after hashes and a clean worktree.

Exit gate: the corrected harness writes only to an external output by default;
the after-write state/hash proof covers the former tracked transcript; 15/15
injections are caught by their named guards; original D1–D3 evidence remains
green; full preflight and portability are green; the LOG contains explicit
corrections; and one fresh independent Skeptic returns **ACCEPT** against the
bounded four-item follow-up.

The reviewer may record residual limitations. It must not withhold acceptance
merely because no finite mutation set proves all possible future multi-edit
changes. A finding remains blocking only if it demonstrates a current false
green, possible evidence loss, recurrence of D1–D3, or a false durable claim.
If such a blocking finding remains after this pass, stop and return it to Samuel;
do not begin an open-ended seventh hardening cycle.

### Phase 0S — first Phase-2 calibration submission

**Class:** STRATEGIC because it is a submission. It changes no estimator and uses
no decision seed. It may be prepared alongside Phase 1A after P1-PRE ACCEPT.

The signed-in team/Phase-2 view showed zero submissions on 2026-08-25. Establish
one official calibration point with the current, frozen K2p1 file before the new
architecture is ready:

1. record the exact estimator hash and the measured official-subprocess mini
   result (`4.050566e-7` adjusted, about rank 40 at the current snapshot),
   explicitly labeling it a baseline rather than an improvement, and report its
   `7.47823932957716e-11` ground-truth noise floor alongside raw MSE;
2. run `whest validate` and a clean `v2-phase2` subprocess mini with zero failures;
3. verify official stack/config, ship-file identity, clean status and current
   daily submission allowance; preserve the signed-in account/quota evidence;
4. append and commit the full STRATEGIC submission contract, including Samuel's
   prediction and the fact that the live score is the 50-network feedback half,
   not access to the withheld final half;
5. obtain one focused independent ACCEPT, then show Samuel the exact-SHA packet
   and request explicit approval;
6. only if approved, submit once and record the official result without fitting
   any constant to it.

This checkpoint calibrates the grader, confirms account access and gives the team
a first Phase-2 record. It is not a promotion decision, not a final nominee, and
must not delay Phase 1A construction once P1-PRE is accepted.

### Phase 1A — construct and verify the carrier without truth data

**Class:** IMPORTANT because it changes a probe's mathematics. No seed block,
truth measurement, vault touch, or `estimator.py` edit.

Build a standalone width-1024 Kerdock/MUB mathematical specification, direct
generator, independent verifier, exact-radial rule, dense propagation reference,
and counted-cost/memory model. The existing length-256 implementation is
GR(4,7); width 1024 needs the known GR(4,9)/real-MUB construction and a direct way
to emit a chosen 1/2/4/8/16-basis subset without materialising the full
1,048,576-word code. This is a new implementation and proof obligation, not an
invitation to invent an unreferenced design.

Primary-source anchor: Calderbank, Cameron, Kantor and Seidel construct the real
Kerdock line set in dimension `N = 2^(m+1)` for odd `m`; therefore `N=1024`
corresponds to `m=9`, while the existing `N=256` code corresponds to `m=7`.
Use their binary quadratic-form/orthogonal-spread construction or prove an
equivalent GR(4,9) spelling; do not infer a parameter change from the current
implementation alone. Source: [Z4-Kerdock Codes, Orthogonal Spreads, and Extremal
Euclidean Line-Sets](https://doi.org/10.1112/S0024611597000403).

Required construction evidence, before any truth data:

1. cite and transcribe the exact algebraic construction, basis indexing and
   ordering; prove reduced dimensions against a simple reference implementation;
2. use different logic for generator and verifier so a shared mistake cannot make
   the suite self-consistent. Property checks over generator output do not meet
   this clause: add an independently generated finite-field Kerdock-matrix
   reference (CCKS Example 9.2) and compare reduced dimensions exhaustively and
   the live subset up to a declared fixed equivalence;
3. prove unit norms, within-basis orthogonality, cross-basis coherence exactly
   `1/sqrt(1024) = 1/32`, antipodal closure, expected unique row count and
   deterministic ordering for 1/2/4/8/16 bases;
4. preserve the known-bad multiplicative-rank spelling as an executable negative
   control and raw artifact, and correct the stale generator docstring/spectrum
   wording; test nested live prefixes `k=1/2/4/8/16`, float32 dtype and norms;
5. validate the bias-free ReLU network's positive-homogeneity/radial identity
   independently, including all layer prefixes and a guard that applies the
   chi-radial factor exactly once;
6. prove the exact second moment and report an exact degree-4 tensor/frame-potential
   diagnostic. For `N=1024,k=16`, a carrier-aligned fourth moment is
   **21.6881103515625x** the spherical target, so random probes are inadequate;
7. build the 32,768-direction dense/full-batch float32 reference first, then a
   submission-legal `fnp` construction/materialisation path. Measure its actual
   official-flopscope operation ledger and own overhead; the leader's unknown
   implementation is a clue, not an equality target;
8. measure fresh-subprocess setup time and peak/transient memory including
   construction, simultaneous point buffers, weights, temporaries and operation
   history. A point buffer is 128 MiB; Windows memory is accounting evidence, not
   an 8-GiB-limit test. Parameterize basis indices, or explicitly defer their
   choice and standard-basis inclusion to preregistered Phase 1B DESIGN.

Exit gate: zero algebraic/determinism/reference failures, exact explained row and
counted-cost accounting, full preflight, a LOG knowledge delta and an independent
Skeptic verdict. No score or variance claim is permitted in Phase 1A.

### Phase 1B — structured-carrier DESIGN evidence

**Class:** STRATEGIC when the new DESIGN block is claimed. The complete contract,
Samuel's restated S2 prediction and commit must precede the claim.

Use a new Phase-2 DESIGN block for basis count, radial nodes, chunking, checkpoint
placement, and other tuning. P01 may be used only for clearly labeled reusable
diagnostics consistent with its DESIGN registration; it is not the decision set.

Design success criteria:

- exact 16-basis, 32,768-direction leading-cost fingerprint reproduced;
- zero construction/determinism/moment invariant failures;
- no code-policy or memory violation;
- promising matched-cost variance reduction with stable layerwise behavior;
- **inside the residual budget below.**

### The operational budget, measured before the carrier exists (P2-P0, 2026-08-24)

Phase 0 measured what the carrier has to fit inside. These are constraints to
design against, not checks to run afterwards, and each needs a falsifier in the
Phase-1 contract.

| constraint | measured | what the carrier gets |
|---|---|---|
| residual, per MLP | `k2p2` spends **~95 ms** of the **400 ms** cap in the best observed session and **260.7 ms in the worst**, where its own worst MLP sat at **1.02x headroom with no carrier at all**. The level is not reproducible to better than **~2.4x** across runs | **~139 ms**, budgeted off the bad day. Designing against the best day is how a carrier passes here and fails on the grader |
| counted FLOPs | `k2p2` costs `1.379e11`; the leader's fingerprint is `1.0998e12`, `C/B = 0.51` | ~`9.6e11` before matching the leader's cost |
| `setup()` | the 5 s ceiling covers **worker spawn + imports + setup**. `import whestbench` alone is 1.3 s quiet, 4.7-9.1 s loaded; our estimator's own setup is **0.6 ms** | build the point set here — it is FREE of FLOPs on the grader — but the wall ceiling is shared with imports we do not control |
| memory | 8,192 MB, **unenforceable on Windows**; shipped file peaks at 918 MB commit (an upper-bound headroom of 8.9x) | one 32,768 x 1024 float32 point buffer is 128 MiB; start with the dense reference and chunk only if measured peak accounting or residual behavior requires it |

Two of these are new and neither was available before Phase 0:

- **the ~139 ms residual budget** is the binding operational number for a carrier
  that must propagate 32,768 directions through sixteen layers — and it is roughly
  a third of what the best-case reading suggested, which is the whole reason the
  measurement was worth arguing about;
- **`setup()` is FLOP-free but not wall-free.** The grader charges nothing for the
  point set built there, and the 5 s ceiling still applies to a window we mostly do
  not own (`import whestbench` alone is 1.3-9 s of it; our own setup is 0.6 ms).

**Quote residual against the worst observed run, never the best.** The same arm
reads 94.6 ms and 260.7 ms on different days, the cause of that 2.35x is measured
NOT to be the flopscope release and only ~19 ms of it the weight representation,
and the remainder is unexplained. See `docs/VALIDATION.md` 2b-ii.

**A carrier design that only fits in the best session is not a design.** If the
32,768-direction carrier cannot be built inside ~139 ms of residual, the honest
responses are to make it cheaper, to move work into `setup()` where the grader
charges no FLOPs, or to reduce the direction count -- not to re-measure until a
friendly number appears.

### Phase 2 — decide whether structured carrier is the main backbone

**Class:** STRATEGIC. New SINGLE_USE DECIDE block.

Preregister one primary comparison: Kerdock/MUB plus exact radial integration
versus matched-cost IID/ordinary orthogonal and the current K2 baseline. Freeze
all design choices before claiming the block.

Proposed falsifiers to freeze in the actual contract after Samuel's prediction:

- less than 2.7x MSE reduction versus IID at matched rows;
- no positive paired arithmetic-score improvement over K2;
- confidence interval crosses the registered score gate;
- residual/memory/failure behavior is not deployable.

If it passes, integrate a versioned conservative candidate and obtain the required
independent Skeptic verdict before any public confirmation or submission.

### Phase 3 — faithful recurrent augmented K2

**Class:** STRATEGIC when the decision block is spent or shipped code changes.

Sequence:

1. Port and verify official ARC recurrence at small shapes.
2. Instrument per-layer errors in mean, variance, retained joint state, and final
   mean.
3. Price the actual recurrence under flopscope 0.12.
4. Explore representation/rank only on DESIGN data.
5. Decide K2p2 versus recurrent augmented K2 on a fresh SINGLE_USE block.

The old CP3R result is not a kill. A reasonable decision falsifier is less than
1.10x raw gain, or less than 1.02x if the measured cost remains fully below the
score floor; the actual gate must be preregistered and tied to score difference.

Test both standalone and as the analytic reference under the carrier. A standalone
loss can still be useful if it materially improves checkpoint control at low cost,
but that combined estimand needs a separate contract.

### Phase 4 — compositional checkpoint control

**Class:** STRATEGIC.

Use DESIGN data to select no more than a small registered set of checkpoints and
response features. Fit with strict cross-fitting or disjoint fit/evaluation sets.
Then decide the frozen predict/correct closure on a new block.

Required diagnostics:

- rollout versus teacher-forced performance;
- per-layer state correction and final-score contribution;
- coefficient stability and tail sensitivity;
- ablation by checkpoint, feature group, and analytic reference;
- total official cost, residual, wall, memory, and failures.

Kill if the paired improvement is below the preregistered threshold, unstable in
the largest movers, or consumes more multiplier than it buys.

### Phase 5 — integrated frontier and bounded alternatives

Integrate only independently supported modules. Evaluate:

- carrier alone;
- carrier + K2p2 reference;
- carrier + recurrent augmented K2;
- carrier + best checkpoint response;
- best analytic-only candidate;
- one bounded optional branch only if residual diagnostics trigger it.

Run 300–600 fresh networks for robustness after architecture freeze. No coefficient,
rank, checkpoint, or threshold may be selected on the DECIDE or public split.

Promote two complementary nominees:

- **Candidate A — conservative:** simplest robust carrier/control combination;
- **Candidate B — aggressive:** the highest-scoring supported state/response hybrid.

### Phase 6 — confirmation, vault, and submission

1. Run public mini as CONFIRM only; fit nothing there.
2. Create/touch a Phase-2 vault only with Samuel's explicit authorization and a
   preregistered milestone contract. The Phase-1 vault is not reusable.
3. Run official subprocess tails, memory, determinism, code-policy, hash, and clean
   environment checks.
4. Freeze the two candidate hashes and evidence packets by October 12.
5. Submit only after `whest validate`, a clean Phase-2 mini with zero failures,
   candidate evidence shown to Samuel, and his explicit approval.
6. Nominate up to two final submissions before the competition deadline.
7. Complete the algorithmic-contribution write-up by October 24 using only claims
   supported by frozen artifacts and Skeptic-reviewed evidence.

---

## 5. Calendar

| window | required outcome |
|---|---|
| Aug 24–26 | evaluator parity; corrections-only P1-PRE ACCEPT; prepare first Phase-2 K2p1 calibration submission and ask Samuel |
| Aug 26–Sep 3 | Kerdock/MUB specification, generator, invariants, then DESIGN evidence |
| Sep 3–12 | carrier DECIDE; augmented-K2 small-shape parity and DESIGN evidence |
| Sep 12–24 | augmented-K2 DECIDE and checkpoint-response experiment |
| Sep 24–Oct 5 | integrated architecture decision and 300–600-network robustness |
| Oct 5–12 | public CONFIRM, authorized vault decision, official deployment audit |
| Oct 12–17 | freeze/submit/nominate two candidates; no late unmeasured changes |
| Oct 17–24 | contribution write-up finalized |

If the calendar slips, protect validation integrity and final operational safety;
drop optional branches before compressing DECIDE, Skeptic, or submission gates.

---

## 6. Validation and evidence contract

Every strategic experiment must contain, before data exists:

- question and exact estimand;
- confirmed/inferred/speculative evidence;
- hypothesis and mechanism;
- smallest discriminating intervention;
- numbered directional predictions;
- Samuel's prediction;
- gate on the isolated paired arithmetic score difference;
- falsifier;
- exact control, tier, and seed claim;
- the frozen **cost** model. Under Phase 2 that is a **counted-FLOP** model,
  because λ = 0 makes `C = F`; residual is reported as **headroom against the
  0.4 s cliff**, quoted against the **worst observed run**. The Phase-1
  three-residual-model rule (`VALIDATION.md` §2b) applies only to a Phase-1
  replay — see §2b-ii;
- confounders and what must not change;
- commit proving preregistration preceded data.

Every evidence packet must include raw artifact paths, hashes/provenance, ground-truth
noise floor, paired arithmetic score interval, ratio diagnostics, exact official
cost, residual/wall/memory/failures, tail sensitivity, and an independent Skeptic
verdict. A DESIGN result never becomes a DECIDE result by documentation wording.

---

## 7. Kill, pivot, and reopening discipline

- Ordinary IID and random-orthogonal sampling are controls, not winning backbones.
- The specific broken lattice remains dead; it does not close valid Kerdock/MUB or
  scrambled-Sobol constructions.
- Higher Price/Hermite orders inside the same Gaussian covariance map remain a
  low-value lane after the measured 1.052x plateau.
- Marginal final-mean Edgeworth correction remains closed at about 1.047x.
- Recurrent joint-state augmented K2 is open.
- Dense full K3 is deferred unless a goal-directed/sketched representation proves
  its bias and cost first.
- Coupled residual sampling is deprioritized until its float32 cost premise is
  measured in a real registered arm.
- Any lane may reopen only on new external evidence or a clearly demonstrated
  mismatch between the killed method and the proposed method, followed by a new
  STRATEGIC contract and Skeptic review.

---

## 8. Immediate next action

**Phase 0 is CLOSED.** Deployment parity was repaired on 2026-08-24: the official
stack is pinned, one round-aware config authority asserts every constant against
the installed package, local MLPs are flopscope-backed, the counted-FLOP anchor
reconciles to the official runner **op by op with residue zero**, and all five G4
paths are provoked — the memory limit on Linux, where `RLIMIT_AS` exists.

**The all-of gate G1–G7 is met**, on a literal independent **ACCEPT** at HEAD
`952bd61`, after four review passes (PROVISIONAL, PROVISIONAL, REJECT, ACCEPT).

**P1-PRE is CLOSED. The immediate action is the bounded Phase-1A closure while
Track 0S receives its cheap procedural correction.**

- **Phase 0S — calibration only.** The candidate is technically safe: the
  100-MLP official subprocess mini is zero-failure at `4.050566e-7`. Add the
  missing GT-noise floor, preserve live account/quota evidence, commit the
  submission contract + Samuel prediction, and obtain a focused ACCEPT before
  asking for exact-SHA approval. Expect about rank 40, not an improvement.
- Phase 1 then runs in two distinct stages:

- **Phase 1A — bounded closure, no truth.** Keep the valid 23/23 carrier, but do
  not call its property checker an independent reference. Add the finite-field
  reference, executable failed-mapping control, exact moment/radial evidence,
  ship-legal flopscope path, fresh-process time and peak/transient memory. No truth
  seed or estimator change. Ends only in preflight and a fresh independent ACCEPT.
- **Phase 1B — DESIGN evidence.** Contract reviewed before any block is claimed,
  on a **new** block (P02 rather than the already-exposed P01), judged on paired
  arithmetic competition-score differences with measured FLOPs, including a
  100-MLP soak reporting warm-up, residual drift, peak memory and worst-case 0.4 s
  headroom — with the unexplained ~2.35× process-state effect as an explicit
  falsifier.

The original Phase-0 prediction requirements and outcomes remain in
`experiments/LOG.md`; they are historical evidence, not current instructions.
