# CURRENT_STATE.md — read this first

**Current as of 2026-09-10 evening, after Stage 0 and Stage 1.** This replaces
the accumulated snapshot, preserved verbatim in [the pre-audit
archive](archive/CURRENT_STATE_before_2026-09-10_audit.md). Historical contracts
and verdicts remain binding. Then read [experiments/LOG.md](../experiments/LOG.md)
from the bottom.

## PICK UP HERE — the session ended on a usage limit, with a clean tree

Nothing is in flight. HEAD carries every artifact named below. Four things are
owed, in this order:

1. **Fit `theta` and `lam` on P07 only** (384 networks, DESIGN, reusable), and
   **freeze them before P01 is read**. Regenerate with
   `uv run python local_sim/p2_build_k4_arms.py --emit --theta <t> --lam <l>`.
   Expect `theta` far below 1: the three-network smoke at the placeholder
   `theta = 1.0` came out ~100x WORSE and diverging with depth, which bounds
   the fit range and is **not** evidence against the hypothesis.
2. **Then** run the arms on P01 through `p2_lab.py` with `--gate 1.10
   --control k3_exact --lam 0.0 --residual-cap 0` and an explicit `--ts`. The
   contract is committed at `08f787b`, before any data, with Samuel's
   predictions in it.
3. **Rule 2 — the independent Skeptic has NOT run** on Stage 0, Stage 1 or the
   Stage 2 slice. It is owed before any of it is believed. Raw artifacts:
   `experiments/2026-09-10_stage0_instrument/`,
   `experiments/2026-09-10_stage1_compact/`,
   `experiments/2026-09-10_stage2_k4/`.
4. **Full `scripts/preflight.ps1` has not been re-run since the Stage 0
   repairs.** Queue it, do not launch it.

**Samuel's standing decisions, which bound the scope:** rules §6 eligibility is
**"hold off entirely for now"** — no organizer question drafted, none sent, the
public reconstruction stays an **attributed research reference**, and Stage 3
is **closed by his decision**, not by an external blocker. `330566` (Strassen
L2, sha `390bc226…`) remains the standing candidate, untouched.

## Where we stand

- **Winning architecture not yet identified.** Persistent higher-cumulant propagation is the strongest concrete direction, but current SIMPLE-K3 is too expensive and leaves accuracy on the table.
- **New evidence:** 504aldo released compact all-history K3 with coherent K4 regeneration and shared old-source bases. Reported adjusted 5.40e-9/raw 2.13e-8/0.2526 budget. This supports specific new contracts, not blanket lane reopening. Public-fit constants and mutable-buffer assumptions are not our protocol. [Pinned code analysis](research/2026-09-10_literature_review.md).
- **Current public leader:** marius_binner #330500 adjusted 2.683e-9/raw 1.700e-8/0.1578 budget. **Our rank 44**, best #330290 adjusted 1.836e-8, a descriptive 6.84× gap. Official MSE noise floors unavailable; these ratios are not promotion statistics. [Chrome evidence](research/2026-09-10_competition_sources.md).
- **`estimator.py` remains v2.6.1**, matching its archive; it is the shipped baseline, not the active K3 candidate. No estimator/archive/instrument/gate/seed/vault changes in this audit. No promotion, submission or designation.
- **Next:** Stage 0 trust repairs and Stage 1 representation derivation in [GAME_PLAN](GAME_PLAN.md), then fresh preregistered research reconstruction. Before treating a participant-derived reconstruction as a submission candidate, resolve rules §6's restriction on derivation from another participant's submission. MIT attribution or a mathematical rewrite alone does not settle eligibility. No large training before a feasible representation and useful reachable-state target.

## Candidate state

| Role | File/submission | Evidence |
|---|---|---|
| Shipped baseline | `estimator.py`, v2.6.1 | Default preflight targets this file, not every nominee |
| Best public result | Strassen L4, #330290 | Public adjusted 1.836e-8/raw 3.779e-8, 0.4858 budget; worst displayed server wall 114.6798 s against 120 s |
| Prior compromise | Strassen L3, #330327 | Historical graded evidence/exact bytes in LOG/archive; no fresh certification here |
| More robust fallback | `estimator_k3_strassen_L2.py`, #330566 | **Now GRADED**; public adjusted 2.284e-8/raw 3.779e-8; certified multiplier 0.604217, 9,321 ops; worst displayed wall 33.0009 s |
| Earlier fast fallback | `estimator_k3_full.py`, #330212 | Historical evidence retained; pair rationale must use best eligible score, not average/worst-of-two |

L2 SHA256 `390bc226bbf517df0110facfc0dc91c78ef670e8a49368a536cabf630057d1db`. Public failures 0/50; all 100 visible statuses completed, private accuracy sealed. Exposed public raw/adjusted means and worst wall meet the corresponding prediction intervals; the full hidden-suite mean was not revealed. Prediction 9,304 ops already missed; residual ≤0.15 s remains unverified in the inspected UI. **Two historical certification gate failures remain failures.** Good grading does not amend their contract.

Observed cap/worst-wall ratios: L2 **3.64×**, L4 **1.05×**. These are dated server observations, separate from contention-adjusted planning margins; no future guarantee. Frozen comments still quote obsolete 1,022 matmul pricing; current SIMPLE shape model is 797. Correct manifests/docs without silently editing frozen bytes. [Architecture audit](research/2026-09-10_architecture_audit.md).

## Current competition facts

Phase 2 opened August 22, announcement August 23: 1024 × 16, Gaussian-input He-initialized bias-free ReLU. Budget 2^41, lambda 0; score is mean of per-network final MSE×max(0.1, F/B). Failed predictions use zero-output error without discount. Installed/current graded WhestBench 0.16.1, flopscope 0.12.1.

Rules explicitly publish **400 ms residual, 120 s wall, 5 s import/setup**, 8 GB participant memory; installed enforcement 8192 MiB. Participant one physical core/two vCPU, backend seven physical cores/fourteen vCPU. Local absolute wall/residual seconds are not server authority. Public mini 100/full 1000 with 1e9-sample truth is separate from leaderboard 50 public/50 sealed and fresh prize rerun.

**Rules v14 §7 resolves best-of-two:** only the participant's best-scoring eligible submission is ranked. §6 permits up to two, defaults to two best valid public submissions if none selected. The old §6-only “checked negative” and drafted clarification request are obsolete. Selection interface/deadline remains unpublished. Only Samuel designates; every submission needs his specific approval.

Deadlines: registration/team freeze **October 2**; submissions **October 17, 23:59 UTC**; write-up **October 24**; tentative winners November 15. Ten submissions/team/UTC day. [Official evidence](research/2026-09-10_competition_sources.md).

## Slice 5: supported scope

| Mechanism | Current status |
|---|---|
| Packing/symmetric tags on existing operands | Saving ~1.0067× failed its gate; not a representation-independent0.5B floor |
| Partial AUGMENT deletions, T1 | Failed; dependencies coupled; does not reject jointly derived K4 uses/births |
| Smooth final-readout correction, T2 | Gain ~1.0163× too small; corrected controls/ridge support that scoped negative |
| T2 P08 provenance | Read three times during repairs; spent, not pristine single-use confirmation |
| In-flight sensitivity arm | Never implemented/run; not settled by final readout |
| Three-generation carry/selectors | Scoped negatives stand; all-age shared coordinates are different |
| Old adjoint kill | Retracted; instrument broken; speculative, not ready to rerun unchanged |

**Whole score-lane closure was already retracted in LOG and must not be repeated.** New public mechanisms change the priority, without erasing tested negatives. New contracts must name the external evidence and exact distinction from a closed intervention.

Local T1 cross-MSE SIMPLE 4.004685e-8 versus AUGMENT 1.912323e-8, per-replica noise 1.1552e-9; full reference not demonstrated affordable. T2 per-replica noise 4.4988e-9. These are not official leaderboard errors. Raw paths and qualifications: [architecture audit](research/2026-09-10_architecture_audit.md).

## Stage 0 — the instrument repairs, DONE

Seven of the validation audit's eight findings are implemented, tested and
committed (`07e8f9b`, `42fd09f`, `06789bd`, `b3059a3`, `2b048eb`). Finding 8
(`out=` / mutable participant buffers) is deliberately **not** implemented: it
is an API question about the official participant shim whose answer is not in
this repository, and it stays a written question.

| what changed | why it mattered |
|---|---|
| arms pair on **MLP identity**, not list order | mispairing left `mean_difference` bit-identical and inflated the SE **8.55x** — the one failure shape no printed report exposes |
| `paired_diff_ci(method="t", alpha=…)` honours alpha | 0.05, 0.01 and 0.20 returned the same tuple bit for bit. Latent, so **no published interval moves** |
| the 8-GiB memory proof is bound to shape, budget, cap, weight mode and stack | a report at `64x4` under a 512 MB cap on float32 weights satisfied every old check. **Reachable, not exercised** — the seven stored reports were correctly configured and now read as DEGRADED |
| the seed claim is **atomic** | twelve concurrent claimers on a `SINGLE_USE` block **all** succeeded, every run. Now one succeeds, eleven are refused |
| a single-use block **counts its LOOKS** | `local_sim/truth_reads.py`. P08 now reports tier SINGLE_USE, cap 1, spent 3, **EXHAUSTED**. An aborted read still consumes the cap |
| label noise does **not** cancel from an adjusted difference | `(q_c − q_k)·v` survives pairing. `noise_corrected_compare` returns observed, true-function and official-precision forecast separately |
| MATH / LOCAL / SERVER are separate verdicts | check **names are unchanged**, so the two standing gate failures are not orphaned. Nobody has ruled on whether they were mis-specified and this does not either |
| the vault's candidate path runs through `lifecycle_eval` | its seal and touch counter are **byte-identical**, verified by re-hashing all six files |

**Test totals:** `competition_score` 98→**129**, `certify_candidate` unit tests
19→**26**, plus three new suites — `test_linux_memory_binding` **23/23**,
`test_truth_reads` **27/27**, `test_vault_lifecycle` **47/47**. Every new case
fails against `0fe164c`. `lifecycle_eval --self-test` 56/56,
`verify_portability` 103/103.

## Stage 1 — the compact representation is priced, and the price is a problem

`local_sim/p2_compact_reference.py`, **28/28**. Transport factors through a
shared basis exactly; the Hadamard does **not** factor, so old sources must be
reconstructed; the fourth-order diagonal identity is exact for its core and
demonstrably **not** for a general one.

**At the published shape (4 recent dense, shared rank 384) the compact form is
MORE expensive than our tower: `1.0023 B` against `0.9300 B`.** It is cheaper
only at rank ≤ 256, best at recent=1/rank=128 (`1.64x` calibrated). The **join
carries 22–53%** of the compact total everywhere, and its core step has **no
flopscope primitive**. The model prices our own tower at `1.1928x` its measured
figure and gives the compact side no equivalent concession, so every ratio is
an **upper bound** on the saving.

**The falsifiable consequence:** the configurations cheaper on our ledger are
precisely the ones the published work did **not** choose. Either their join is
materially cheaper than the incremental form priced here, or rank 384 buys
accuracy this cost screen cannot see, or the saving is somewhere this ledger
has not located. [Ledger](../experiments/2026-09-10_stage1_compact/ledger.json).

## Stage 2 experiment 1 — armed, not run

The shipped chain **already** carries a fourth-order state, and it is **one
float for all 1,024 neurons**. The intervention carries the per-neuron
deviation as well, transported by Stage 1's identity — and `diag(W^T C W)` is
`var`, already computed, so the channel is nearly free.

- `theta = 0` is **bit-identical** to `estimator_k3_full.py` in output **and**
  counted FLOPs: the control is nested exactly, not approximately.
- **Cost prediction RESOLVED:** the channel adds `145,977,344` FLOPs at
  `1024x16` — **`0.000066 B`** against a predicted "under `0.02 B`" and a Stage 1
  projection of `0.000038 B`. First metered confirmation of that projection.
- **Accuracy predictions 1, 2, 3, 5, 6 remain OPEN.** The fit is owed first.

## Validation and remaining repairs

Verified: configuration **96/96**, certifier aggregation **19/19**, default preflight **GREEN 12/16, four skipped**, shipped-file FLOP anchor 19 categories / zero disagreements. Queued job 539 declared 8 GiB RAM, 0 VRAM, 2 CPU, no GPU. [Evidence](research/2026-09-10_validation_audit.md).

Outstanding after Stage 0: the `out=` / mutable-participant-buffer question
(finding 8), which needs the official participant shim's exact semantics and is
not answerable from this repository; a full preflight re-run; and the
independent Skeptic. Everything else in the audit's list is implemented and
tested — see the Stage 0 table above.

Common-label noise cancels in raw MSE differences, not automatically in adjusted scores with unequal multipliers. Future reports need cross-MSE and official-precision noisy-score forecasts. This mathematical correction does not assert a particular past promotion changes outcome. [Repair details](research/2026-09-10_validation_audit.md).

## Durable operating state

AGENTS and VALIDATION remain binding. IMPORTANT instrumentation: written prediction, preflight, LOG delta, fresh-context Skeptic. STRATEGIC: committed preregistration, Samuel's predictions, fresh claimed blocks, binding arithmetic gate. This roadmap grants no seed/vault touch, gate amendment, submission or designation.

HUMAN_MODEL remains human-owned. Existing vault seal/touch counter intact, historical Phase 1. Heavy workloads through workerq with declared resources. No autonomous optimization loop, tracker or new orchestration framework needed. Parallel agents were user-requested for this audit; they do not replace the independent Skeptic.

Contribution write-up remains stale/do-not-submit pending claim review. Attribute prior public work and distinguish replication from new tested contributions. The audit is not a submitted novelty claim.

Current map: [GAME_PLAN](GAME_PLAN.md) → [research synthesis](PHASE2_RESEARCH_AUDIT_2026-09-10.md) → [VALIDATION](VALIDATION.md)/[GOTCHAS](GOTCHAS.md). September 5 strategy, old GAME_PLAN and future-named September 11 packet are historical; conflicting next-action/closure statements superseded. Original snapshot retained in [archive](archive/CURRENT_STATE_before_2026-09-10_audit.md); all contracts/results in [LOG](../experiments/LOG.md).
