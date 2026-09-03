# GAME_PLAN.md — Phase-2 implementation plan and target architecture

> ## AMENDMENT, 2026-09-02 — read this before anything below it
>
> **This plan is dated 2026-08-31 and four slices have run since (WP13, WP14,
> `SKEPTIC WP14`, and today's follow-up). Sections 3 and 4 below no longer
> describe the live architecture, and a Builder who executed section 8's order
> literally would spend days on closed lanes.** The plan is amended here rather
> than rewritten, so the reasoning that produced it stays legible.
>
> ### What is now CLOSED or deprioritised, on measurements
>
> | package | status | measured reason |
> |---|---|---|
> | **WP1 / WP2** — the legal carrier constructor, objective and frozen panel | **CLOSED** | the carrier lane is closed; see `CURRENT_STATE.md` and Appendix B of the write-up |
> | **WP3** — faithful AUGMENT | **DEPRIORITISED** | not on the critical path; the SIMPLE tower reaches the floor without it |
> | the two-candidate ambition of the objective statement (a *structured angular-integration system* plus a compact analytic one) | **superseded** | the live pair is the regenerating K3 flow and shipped `v2.7`, chosen for diversity, not two new builds |
> | five column-weight selectors (WP13 `E1`), `Sigma`'s dominant range as a selector (WP14 `C1`) | **CLOSED** | every selector is `1.45x`-`2.10x` **worse** than not correcting; containment is `r/n` to three decimals, i.e. chance |
> | Strassen, symmetry tagging, `W-EFF` | **CLOSED / not needed** | the floor is reached in plain f32 with no fast matmul; symmetry tagging bills the identical figure |
>
> ### What actually directs work now
>
> **1. Finish the `fnp` port and present a canary.** `local_sim/p2_k3_fnp.py`.
> The layer exists, is metered and passes three closed-form checks; what remains
> is named in that module's `NOT_DONE` — parity against upstream at non-zero
> `kappa_3` (step 6, the one that can fail for an interesting reason), then an
> actual `Estimator` at live shape, then `certify_candidate` → a `gpuq`-brokered
> residual soak (worst run, `>= 4x` headroom) → the Linux memory gate → full-suite
> metered FLOPs including the `+2**24` server offset → an exact
> `submission.tar.gz` with its sha256 and a one-paragraph recommendation for
> **Samuel** to submit. **Never run `whest submit`, and never designate a
> submission.**
>
> **2. `CONTRACT WP15` (`experiments/LOG.md`, commit `1b6ac04`).** Two arms on the
> one remaining question — is there a MAP from the carried state
> `(mu, Sigma, S3, S21)` to the rank-`r` correction of BOTH `kappa_3` read
> slices. Arm M is mechanistic (measure `E21`'s memory depth, then a closed-form
> generator plus a rank-`r <= 192` sketch); Arm M's affordability horizon is
> `r <= 192`, **not** `512`. Arm L is a supervised permutation-equivariant map on
> `P05`, validated on `P06`. Seven gates, all thresholds already fixed.
>
> **3. The budget every design must now be priced against.** Measured, not
> projected: **`5.047` f32 `1024`-matmuls per layer**, `0.07886 B` over 16 layers,
> leaving **`0.02114 B` = `1.35` f32 matmuls per layer**. A design costing two
> extra `n^3` operations per layer does not fit and does not need measuring.
>
> **Kill line ~2026-09-20**: if neither WP15 arm reaches `>= 10x` end-to-end drift
> reduction on fresh networks, freeze, nominate the canary-confirmed uncorrected
> regenerating flow plus `v2.7`, and finish `docs/writeup-p2-draft.md`
> (due Oct 24, closing Oct 17).
>
> ### What in this plan is still exactly right, and should not be diluted
>
> Section 4's opening rule — *"each IMPORTANT or STRATEGIC package ends with
> synthesis and a fresh independent Skeptic before its output can feed the next
> implementation"* — and section 8's completion definition, especially its last
> paragraph: **a falsifier may legitimately kill a branch, and the required
> outcome is a clean evidence path, not a forced positive result.** WP13, WP14
> and WP15's own contract each predicted gates in themselves failing, and several
> did.
>
> **`WP0.2` is DONE and was still live today**: the stale preflight subtest
> labels it names were `60/60` in two places, and correcting them was part of
> repairing a change gate that had been RED for four commits.


**Authoritative from 2026-08-31. This is the sole active architecture and
implementation plan.** `CURRENT_STATE.md` records status, `VALIDATION.md` governs
measurement, and `PHASE2_FRONTIER_AUDIT_2026-08-31.md` contains the dated external
evidence. Historical plans and dated audits do not direct work.

The objective is not another incremental K2 variant. It is to produce, validate,
and package two materially different Phase-2 candidates capable of beating the
current public frontier on a fresh private rerun:

1. a structured angular-integration system in the leaders' observed compute
   envelope; and
2. a compact analytic system that preserves final-layer-relevant third-order
   information without transporting the full K3 tower.

The implementation must be usable before it is believed. Every truth-free module,
integration path, cost meter, deployment check, and falsifier should be completed
as early as possible. Labels then decide which already-working design survives;
they must not be used to finish the engineering.

## 1. Fixed problem, targets, and non-negotiable constraints

For a random bias-free `1024 x 16` He-initialized ReLU MLP, predict every layer's
post-ReLU mean under `x ~ N(0, I)`. Phase 2 scores

```text
mean_MLP[final_layer_MSE * max(0.1, FLOPs / 2**41)]
```

with lambda zero. Installed WhestBench 0.16 / flopscope 0.12 defaults are a
`0.4 s/MLP` residual cliff, a 120-second per-MLP wall default, 5-second setup, and
8-GiB memory. Public graded telemetry proves only that `>=104.766 s/MLP` passes;
the exact server ceiling is sealed.

The dated 2026-08-31 board target is J2W at `5.60e-9` adjusted / `2.25e-8` raw.
Our v2.7 is `1.374e-7` / `1.374e-6`. A nominal winning objective is `<4e-9`
adjusted on fresh offline evidence with interval and tail safety, not merely a
point estimate that matches the public leader. This is a program target, not a
promotion gate. Every actual gate is written and accepted before its data exists.

Hard constraints:

- shipped code may import only Python stdlib, `flopscope`,
  `flopscope.numpy as fnp`, and `whestbench`;
- arrays are immutable; no item assignment or in-place mutation;
- all numerical submission work is through instrumented `fnp` operations;
- predict-time randomness derives only from `mlp.seed`; setup randomness derives
  only from `context.seed`;
- no `bool`, `float`, `int`, assertion, or implicit conditional may materialize a
  flopscope value during `predict`;
- one array must remain below 4 GiB and total solution memory below 8 GiB;
- no public-mini fitting, no vault use without authorization, and no submission
  without Samuel's explicit active-session approval.

## 2. What is established and what is closed

### Baseline and evaluator

- v2.7's scalar calibration is real: approximately `2.9x` over basic K2, with
  local-to-graded transfer agreeing within 1.5%.
- More per-layer scalar gains are exhausted. Useful corrections must change
  neuron-specific shape or add information.
- Minting, scorer identity, paired arithmetic decisions, seed accounting, package
  parity, failure pricing, and the local official-package FLOP anchor are strong.
- Default preflight is green 12/16. The four full deployment stages remain
  necessary before promotion and are not current passes.
- Windows cannot prove the 8-GiB limit. A real Linux enforcement run is still
  required for a new candidate.
- The grading server adds an unexplained `+2**24` FLOPs to the current K2 file.
  Include it near a multiplier boundary; do not invent an operation attribution.

### Accuracy and cost evidence

- Full factorized K3 is the accuracy oracle: about `3.92e-8` raw on organizer
  networks.
- Its best verified whole-path spelling is about `1.0732 B`, giving projected
  adjusted score around `4.21e-8`. It is too expensive to win.
- W-EFF is closed: exact parity but `3.0664x` over its `<=0.35 B` screen.
- Late/age-window K3 is closed: even impossible best-case score floors are
  `2.09x`–`4.65x` worse than v2.7 because the K2 prefix already lost the state.
- A global positive scale mixture is mathematically redundant in a homogeneous
  bias-free network. Only neuron-specific shape/skew mixtures remain meaningful.
- Several leaders spend roughly `5.45e11–5.73e11` FLOPs, consistent with about
  16,384 dense directions. That is an operating envelope, not evidence of their
  point family.

Do not reopen W-EFF, late-window K3, global scale mixtures, generic Frobenius K3
compression, or further Gaussian pair-map refinements without new external
evidence that changes the mechanism.

## 3. Ideal architecture

The target system has shared deployment infrastructure and two independently
viable prediction branches. It must never require both branches to survive.

```text
                        deterministic setup assets
                     /                              \
        structured angular carrier              analytic state
        exact Gaussian radial factor             corrected f32 K2
        nested streamed directions               faithful AUGMENT state
        full-network propagation                 query-directed K3 response
                     \                              /
                      frozen, evidence-backed fusion
                         shape-safe final means
```

### 3A. Shared substrate

One shape-generic research interface owns configuration, per-MLP seed derivation,
chunking, result assembly, cost metadata, and fallbacks. It must expose each branch
alone and the frozen hybrid without duplicating math.

Expected research modules—names may change only if the existing layout makes a
different boundary materially clearer:

| module | responsibility |
|---|---|
| `local_sim/p2_carrier_fnp.py` | legal structured carrier construction and streamed propagation |
| `local_sim/p2_carrier_objective.py` | truth-free kernel discrepancy and frozen nested-policy selection |
| `local_sim/p2_augment.py` | faithful unfactored AUGMENT reference and deployable recurrence |
| `local_sim/p2_adjoint_k3.py` | layer-local K3 sources, final-output adjoints, direct contractions |
| `local_sim/p2_candidate_architecture.py` | branch configuration, integration, fallbacks, and research prediction API |
| corresponding `test_*.py` files | parity, metamorphic, cost, shape, determinism, and failure tests |

The existing NumPy/Torch references remain references. Submission-facing math must
also exist in stdlib/`fnp` form; wrapping a raw NumPy result is not legal evidence.
Do not touch `estimator.py` until a STRATEGIC promotion contract authorizes it.

### 3B. Nominee A — structured angular integration

Use a deterministic Kerdock/real-MUB family or the best surviving matched control,
paired antipodally and multiplied by the exact Gaussian radial expectation. Stream
nested blocks through all 16 layers and accumulate per-layer means.

Required design properties:

- the frozen `kerdock_mub.py`/`KERDOCK_MUB_SPEC.md` object is the parity oracle;
- basis ordering, Gray convention, scaling, antipodes, and radial normalization
  are explicit and tested;
- candidate policies are nested so added compute reuses the exact prefix;
- per-MLP randomization, if used, is a deterministic signed permutation or other
  measure-preserving symmetry derived from `mlp.seed` and implemented without
  scalar synchronization;
- direction chunks are chosen from memory accounting, not host convenience;
- controls include equal-row IID antithetic, ordinary orthogonal,
  scrambled-low-discrepancy, and at least one randomized orientation;
- 4/6/8/10/12 basis points are frozen before truth. Eight bases is a hypothesis,
  not an answer inferred from the board.

A carrier-only result is a complete candidate. K2 fusion is optional and must earn
its place on held-out error correlation and adjusted score.

### 3C. Nominee B — compact analytic response

Start from corrected float32 calibrated K2. Add two separately falsifiable layers:

1. **Faithful AUGMENT state.** Reproduce ARC's unfactored AUGMENT recurrence at
   small shape, component by component and layer by layer. `factor=True` is not a
   valid parity target because upstream drops unsupported diagrams.
2. **Query-directed K3.** Linearize the final-layer means with respect to
   layer-local third-cumulant source terms. Backpropagate final-output queries and
   contract each local source directly into those queries:

   ```text
   delta m_L ~= sum_l J_(L <- l) S_l^(3)
   ```

   The working implementation must avoid retaining the recurrent dense K3 tower.
   Its state size, rank growth, FLOPs, and memory must be explicit functions of
   width, depth, and query rank.

At small shape, compare every source contribution and final correction with exact
K3. Required falsifiers are source-amplitude linearity, sign/cosine agreement,
explained squared drift, covariance-feedback sensitivity, adjoint-rank growth, and
counted price. Kill the route if the exact comparison is poor or its scaling
reconstructs the full `Theta(n^3 L^2)` tower.

A tiny neuron-specific skew/shape mixture may be tested only after a supported
AUGMENT or adjoint state exists. It is an optional readout, never the core system.

### 3D. Fusion and fallbacks

The integrated research API supports:

- calibrated K2 control;
- carrier only;
- AUGMENT only;
- AUGMENT plus query-directed K3;
- one frozen carrier/analytic hybrid if its interaction is preregistered.

Every branch must have a deterministic, shape-generic fallback. Operational failure
must not silently select a different mathematical estimator. A deployable file must
either return its frozen candidate or fail validation loudly before submission.

The two eventual nominees should remain diverse by default:

- **Nominee A:** best legal structured carrier, with only an independently earned
  low-correlation analytic blend;
- **Nominee B:** corrected K2/AUGMENT plus supported query-directed K3, with only an
  independently earned neuron-specific readout.

## 4. Implementation work packages

Each IMPORTANT or STRATEGIC package ends with synthesis and a fresh independent
Skeptic before its output can feed the next implementation. Contracts and
predictions precede result data.

### WP0 — remove infrastructure blockers

Classify each change separately rather than hiding instrument work in a candidate
diff.

1. Repair the live Phase-2 manifest/checklist pin using the durable 70-object
   comparison. Preserve old commit provenance and add a regression for metadata-only
   tag movement.
2. Correct stale preflight subtest labels and the VALIDATION step mapping without
   changing executable gates.
3. Add candidate-selectable certification without weakening the ship-file gate.
   Prefer a separate read-only candidate command or a fully tested parameter whose
   default remains `estimator.py`. It must cover validation, lifecycle, FLOPs,
   shapes, residual, scalar synchronization, and memory accounting.
4. Add or finish a Linux/WSL memory-enforcement path that actually applies the
   8-GiB limit. If this host cannot execute it, produce the exact command and mark
   deployment blocked rather than claiming a pass.

Exit: the systems needed to price and certify a research candidate exist before
the candidate needs them. Default preflight and portability stay green.

### WP1 — legal carrier constructor and propagation

Implement the stdlib/`fnp` carrier and streaming forward path. Required tests:

- byte parity with the frozen reference at supported reduced sizes and live width;
- exhaustive defining properties at reduced sizes;
- repeated-process determinism;
- antipodal, row-order, scaling, dtype, and chunk-boundary mutations;
- per-layer propagation parity with the research reference;
- zero scalar materializations in `predict`;
- exact counted ledger, setup/first-call evidence, and peak/single-array memory.

Exit: a runnable carrier-only research candidate passes candidate certification on
the live shape without labels.

### WP2 — truth-free carrier objective and panel

Implement the deep-ReLU kernel discrepancy used to screen point sets. Prove its
algebraic nulls, permutation behavior, deterministic ordering, and global-rotation
invariance. Freeze the nested basis counts, controls, row matching, cost matching,
randomization policies, and deterministic proxy-to-policy rule before labels.

Exit: a content-addressed panel and one deterministic selection rule are ready for
DESIGN. Proxy success alone does not promote a policy.

### WP3 — faithful AUGMENT

Port an unfactored small-shape reference first, then a deployable state recurrence.
Test every retained state and layer against upstream; test dtype and symmetry
conventions; price the real float32 spelling; distinguish standalone AUGMENT from
its use as an adjoint/checkpoint substrate.

Exit: either a parity- and price-cleared AUGMENT branch plugs into the common
candidate API, or the lane is killed with the smallest discriminating artifact.

### WP4 — query-directed/adjoint K3

Implement an exact small-shape source oracle, backward query propagation, direct
source contractions, and rank/memory telemetry. Compare with full K3 before any
truth data. Test individual layers, sums of layers, seed determinism, source
scaling, covariance feedback, query-rank truncation, and pathological networks.

Exit: either a runnable analytic branch explains a preregistered fraction of
exact-K3 drift below its price/rank screens, or the lane is killed. No post-hoc
rank choice is allowed.

### WP5 — integrated research architecture

Wire all surviving branches into `p2_candidate_architecture.py`. The integration
must run at reduced and live shapes, expose branch-isolation controls, produce
per-MLP cost/failure/memory evidence, and be convertible to a single-file estimator
without importing local modules.

Required integration tests:

- shape matrix including `1024 x 16` first;
- exact layer-1 closed form and permutation equivariance;
- deterministic repeat and fresh-worker repeat;
- branch isolation and zero-weight blend identities;
- no mutation of grader arrays;
- no scalar synchronization;
- chunk and memory edges;
- official flopscope accounting and multiplier;
- subprocess lifecycle and first-call tails;
- deliberate failure mutations that prove every guard can fail.

Exit: at least one end-to-end, grader-compatible research architecture runs and is
fully priced before any new truth is read. This is the required engineering
completion state for the next Builder session.

## 5. Evidence-block and decision sequence

No Phase-2 decision block or independent DESIGN-replication block exists today.
P01 is DESIGN with 432 unused seeds, but it cannot serve as its own independent
replica.

After truth-free work fixes the power requirement and before labels:

1. run a separate IMPORTANT instrument contract to register a disjoint Phase-2
   DESIGN-replication block and a distinct SINGLE_USE DECIDE block;
2. test registry and ledger mutations, run default preflight, record the knowledge
   delta, and obtain an independent Skeptic;
3. under a STRATEGIC contract and Samuel's prediction, select on a preregistered
   DESIGN sample;
4. lock the policy and reproduce it on the disjoint DESIGN block;
5. synthesize, obtain a fresh Skeptic, and freeze architecture, coefficients,
   fallbacks, cost model, gate, and estimand;
6. only then claim the SINGLE_USE block and compare with exact v2.7 on common
   random networks using `competition_score.decide(gate=...)`.

The binding statistic is the paired arithmetic adjusted-score difference with a
studentised-t or basic-bootstrap interval. Percentile bootstrap and BCa are
refused. Report every MSE with its ground-truth noise floor, counted FLOPs and
multiplier, failures, residual and wall headroom, memory, and tail contribution.

The public mini is CONFIRM only and may fit no basis, coefficient, rank, checkpoint,
mixture, calibration, or threshold. A Phase-2 vault may be created or touched only
with explicit authorization and a STRATEGIC contract.

## 6. Promotion, deployment, and nomination

Promotion is a separate STRATEGIC change. Only after a decision passes may a
Builder create a new immutable `estimator_v*.py` archive and copy identical bytes
into `estimator.py`.

Required promotion packet:

- accepted contract, Samuel prediction, decision artifact, valid interval, and
  ground-truth noise;
- candidate git commit and exact SHA-256;
- ship/archive byte identity;
- official `whest validate`;
- default and full preflight, with a durable GREEN 16/16 transcript from the
  candidate-bearing commit;
- clean official Phase-2 mini CONFIRM with zero failures;
- three official-subprocess 100-MLP soaks with tail evidence;
- zero scalar materializations;
- counted FLOPs including the observed server offset near a boundary;
- residual and per-MLP wall headroom;
- real Linux 8-GiB enforcement;
- fresh independent Skeptic ACCEPT.

Samuel then approves or refuses the exact hash. Never submit, designate, post, or
touch an existing submission without that active-session approval.

The private prize reruns up to two designated entries, so nominate two diverse
candidates if both survive. If only one genuinely clears the evidence bar, do not
manufacture diversity by sending a weaker correlated variant.

## 7. Kill rules and pivots

| result | action |
|---|---|
| carrier cannot be built legally or fit memory | kill that construction; test the frozen matched controls, not an ad-hoc raw-array workaround |
| carrier proxy does not predict DESIGN ranking | discard the proxy; do not optimize it harder |
| carrier wins raw but loses adjusted | reduce only along the preregistered nested panel; otherwise kill |
| AUGMENT parity fails | repair the reference target once; kill if the exact unfactored target cannot be defined |
| AUGMENT costs more than its possible score gain | retain only as a truth-free oracle/checkpoint reference |
| adjoint K3 has weak exact-oracle agreement | kill; do not fit labels to conceal a wrong response model |
| adjoint rank/cost reconstructs the full tower | kill or restrict to a preregistered hard-neuron/query subset |
| fusion errors are highly correlated | keep the better standalone branch |
| DESIGN selection fails replication | do not spend SINGLE_USE; diagnose mechanism under a new contract |
| score passes but deployment fails | repair the operational defect under a new contract; do not submit |
| no branch survives | write a new Strategist Packet; do not reopen closed lanes without new evidence |

## 8. Session execution order and completion definition

The next Builder should work in this order and continue while safe work remains:

1. WP0 infrastructure blockers, as separately classified IMPORTANT slices.
2. WP1 legal carrier constructor and live-shape candidate path.
3. WP2 objective and frozen carrier panel.
4. WP3 faithful AUGMENT and WP4 adjoint K3, in parallel only when independent
   contexts and GPUs make that safe.
5. WP5 integrated research architecture and candidate certification.
6. Power analysis and the IMPORTANT Phase-2 block allocation.
7. DESIGN selection and disjoint replication only after the STRATEGIC contract,
   commit, and Samuel prediction.
8. SINGLE_USE decision, promotion, deployment, CONFIRM, nomination, and submission
   only through their explicit gates.

**A successful implementation session ends with:**

- all safe truth-free packages implemented, tested, synthesized, and independently
  reviewed;
- at least one runnable end-to-end research candidate at live shape;
- the carrier and analytic branches independently switchable and priced;
- no known systems/test dependency deferred merely because it is inconvenient;
- exact raw artifact paths and next gate in `experiments/LOG.md` and
  `CURRENT_STATE.md`;
- any unavoidable human, GPU, Linux, truth, vault, or submission dependency stated
  as one precise blocker with the exact command or decision required.

It does **not** mean claiming that an unmeasured architecture wins. A falsifier may
legitimately kill a branch. The required outcome is a working, testable architecture
and a clean evidence path—not a forced positive result.
