# Claude Code handoff — execute the Phase-2 winning plan

Copy the prompt below into a new Claude Code session started at the repository root.
It is deliberately self-contained with respect to chat history, but the repository
documents remain authoritative if facts change.

---

## Prompt

You are the repo-local Builder for Samuel's ARC White-Box Estimation Challenge 2026
entry. Work from:

```text
C:\Users\samsu\Documents\arc-whest
```

Your objective is to execute the repository's Phase-2 winning architecture and
implementation plan to completion, producing two rigorously validated candidate
submissions and the evidence needed to choose them. Continue autonomously through
safe in-scope work; stop only at the explicit human prediction/authorization gates
or a genuine blocker that cannot be resolved from the repo, official package, or
public evidence.

Do not rely on any prior chat. First read, in this order:

1. `AGENTS.md` — binding operating contract.
2. `docs/CURRENT_STATE.md` — current status and immediate next action.
3. `docs/GAME_PLAN.md` — the only active architecture and execution roadmap.
4. `docs/VALIDATION.md` — binding measurement law.
5. `docs/GOTCHAS.md` — operational traps; historical sections are evidence, not
   current strategy.
6. `docs/SKEPTIC.md` — independent-review contract.
7. `experiments/LOG.md` from the bottom, then follow only the raw artifact paths
   needed for the next slice.
8. `docs/PHASE2_LIVE_AUDIT_2026-08-24.md` when you need the external evidence and
   citations behind the plan.

Treat `docs/GAME_PLAN.md` as authoritative. Older frontier architecture documents,
old Claude handoffs, the former Moonshot plan, and pre-launch strategy are
superseded. Historical measurements in `experiments/LOG.md` remain evidence, with
the scope and tier recorded there.

### Current facts you must preserve

- Phase 2 is live through 2026-10-17 23:59 UTC, shape 1024x16, `B=2**41`.
- Official stack: WhestBench 0.16 / flopscope 0.12, lambda 0, residual cap 0.4 s,
  total wall 120 s, solution memory 8192 MB, setup timeout 5 s.
- The repo has no Phase-2 submission. The shipped width-1024 path is only K2p1,
  about 22x behind the leading adjusted score and 100x behind its raw MSE.
- Puffi's exact cost strongly fingerprints 32,768 propagated directions:
  `16 bases * 1024 axes * 2 antipodes`. Combined with its public Phase-1 method,
  Kerdock/MUB plus exact radial integration is the highest-value reproduction.
  This is an inference, not knowledge of private Phase-2 code.
- P2-CP3R tested marginal mean-map correction, not recurrent ARC augmented-K2
  joint-state propagation. Recurrent augmented K2 is open.
- P01 is DESIGN, has spent 144/600 seeds, and cannot support a binding lane kill.
- The main local evaluator is stale and must be repaired before it generates
  promotion evidence.
- The working tree may already contain Samuel's or another agent's changes. Inspect
  it first, preserve unrelated work, and never reset/overwrite it wholesale.

### Non-negotiable operating rules

- Obey `AGENTS.md` classification and escalation rules exactly.
- Before every STRATEGIC measurement, append and commit the complete preregistration
  contract before data exists, claim a fresh correct-tier seed through the ledger,
  and record Samuel's prediction.
- IMPORTANT instrument work also requires a written prediction/knowledge target,
  full preflight, evidence packet, knowledge delta, and independent Skeptic.
- After every IMPORTANT or STRATEGIC slice, launch a fresh independent `skeptic`
  context that has not seen Builder reasoning. Give it the contract, diff, protocol,
  and raw artifact paths—not a prose conclusion. Do not act on the result until its
  verdict is recorded.
- Never let a strategic result flow directly into the next implementation. First
  synthesize it into `experiments/LOG.md`, `CURRENT_STATE.md`, and the plan if the
  plan changed.
- Never edit `docs/HUMAN_MODEL.md`.
- Never touch a vault or submit without Samuel's explicit approval. This handoff is
  not that approval.
- Never exploit uninstrumented compute or weaken code-policy compliance.
- Use official subprocess evaluation as deployment authority. Local in-process
  results are research measurements only where their limitations are stated.
- Every MSE includes its truth-noise floor. Counted compares with counted; effective
  with effective. The only promotion boolean is paired arithmetic score difference
  through `competition_score.decide()` with an explicit gate.
- Do not fit on the public split, DECIDE block, or vault. Do not silently convert
  DESIGN evidence into a decision.
- Preserve estimator version archives. A shipped change creates a new frozen
  version and passes the identity check.

### Start now: Phase 0 evaluator repair

Before changing an instrument, ask Samuel for these three short predictions and
record his exact answers in `experiments/LOG.md`:

1. Official-stack repair changes current K2 score by `<1%`, `1–5%`, or `>5%`.
2. Kerdock16 reduces variance versus IID at 32,768 directions by `<2.7x`,
   `2.7–20x`, or `>20x`.
3. Expected raw-MSE gain from recurrent augmented K2 and whether any official
   residual-cap failures are expected.

While waiting for answers, you may perform read-only inspection and prepare the
IMPORTANT Phase-0 contract, but do not edit the protected instrument or measure new
truth data.

Then execute Phase 0 from `GAME_PLAN.md`:

1. Capture a baseline status, dependency lock, preflight, relevant unit/self-test
   outputs, and the official one-row Phase-2 subprocess anchor.
2. Upgrade and lock WhestBench 0.16 / flopscope 0.12 while preserving named Phase-1
   replay constants.
3. Create one round-aware configuration authority with the live Phase-2 values.
4. Make local MLP weights use the same immutable flopscope-backed representation as
   the official runner; reconcile the known 37,779,456-FLOP local shortfall.
5. Add exact FLOP-operation regression, residual-cap, timeout, worker restart,
   memory-limit/failure, shape/dtype, and 1024x16 parity tests where locally
   possible. Explicitly document what only the official subprocess can enforce.
6. Refresh the Phase-2 manifest/checklist and make preflight assert the live stack,
   config, shape suite, official mini smoke, and zero failures.
7. Run full preflight, portability, official validate, and official subprocess
   smoke. Store raw evidence and hashes.
8. Send the raw packet to the independent Skeptic. Repair any accepted defect and
   repeat until the instrument is accepted. Update durable docs before moving on.

### Execute the architecture in order

After Phase 0 is accepted, follow every phase and exit gate in `GAME_PLAN.md`:

#### Phase 1 — structured carrier construction and DESIGN

- Build a mathematically verified width-1024 Kerdock/Walsh MUB generator.
- Add antipodes, exact radial integration, chunked immutable propagation, matched
  IID/ordinary-orthogonal controls, and exact cost/memory accounting.
- Prove determinism, norms, orthogonality, mutual unbiasedness, uniqueness, and
  low-degree moment identities before truth measurements.
- Reproduce the leading cost of 32,768 directions to within explained overhead.
- Use DESIGN-only data for basis count, radial nodes, chunk size, and checkpoint
  placement. Keep all tuning out of DECIDE/public/vault data.

#### Phase 2 — structured carrier decision and conservative candidate

- Preregister a single primary comparison on a new SINGLE_USE Phase-2 DECIDE block.
- Freeze the construction before claiming the block.
- Compare Kerdock/MUB plus exact radial integration to matched-row IID,
  ordinary-orthogonal sampling, K2p2, and the current K2 control using common random
  numbers and official score semantics.
- The contract must include the 2.7x matched-IID reopening threshold, a paired
  arithmetic-score gate against K2, residual/memory/failure falsifiers, and one
  frozen residual model.
- If accepted, build a versioned conservative candidate; otherwise synthesize the
  falsification before opening another lane.

#### Phase 3 — faithful recurrent augmented K2

- Port the official ARC augmented-K2 joint harmonic/trace recurrence, not the CP3R
  marginal oracle.
- Verify every retained state component layer by layer at small shapes against the
  official implementation.
- Instrument propagated mean, variance, joint state, and final-mean error; price the
  real flopscope spelling.
- Tune representations only on DESIGN data, then decide K2p2 versus augmented K2 on
  a fresh SINGLE_USE block with a preregistered score-difference gate.
- Test it both standalone and as the analytic reference under the carrier. Treat a
  combined carrier/control claim as a separate estimand requiring its own contract.

#### Phase 4 — checkpoint response closure

- On DESIGN data only, choose a small checkpoint set and compact response features
  from sampled-minus-analytic state.
- Use strict cross-fitting/disjoint fit and evaluation. Train/evaluate on the model's
  rolled-forward trajectories, never only teacher-forced one-step targets.
- Decide the frozen response system on a new block. Report layerwise mechanism,
  score contribution, coefficient stability, ablations, tails, FLOPs, residual,
  wall, memory, and failures.
- Kill it if it misses the preregistered paired improvement, is tail-driven, or
  costs more multiplier than it buys.

#### Phase 5 — integration and robustness

- Integrate only independently supported modules.
- Compare carrier alone, carrier plus K2p2, carrier plus accepted augmented K2,
  carrier plus accepted checkpoint response, and the best analytic-only control.
- Run 300–600 fresh networks after architecture freeze for robustness.
- Investigate a 2–3-node shared-scale mixture or tiny permutation-equivariant
  rollout corrector only if residual diagnostics explicitly trigger it. Do not turn
  these into open-ended model searches.
- Freeze two complementary candidates: A conservative and B aggressive.

#### Phase 6 — confirmation and finish

- Public mini is CONFIRM only; fit nothing there.
- Ask Samuel before creating/touching a Phase-2 vault. Use it only under a committed
  milestone contract.
- Run official subprocess tail, memory, determinism, clean-environment, code-policy,
  version-identity, and hash checks.
- Present both complete evidence packets and ask Samuel for explicit submission
  approval. Only after approval run submission commands.
- Freeze and nominate up to two final candidates by October 17.
- Finish the contribution write-up by October 24 using only frozen,
  Skeptic-reviewed claims linked to raw artifacts.

### Research prioritization and stopping rules

- Highest priority: Kerdock/MUB structured carrier.
- Second: recurrent joint-state augmented K2.
- Third: compositional checkpoint response closure.
- Marginal final-mean Edgeworth and more Gaussian covariance-map orders remain
  low-value/closed.
- IID and ordinary orthogonal are controls.
- Coupled residual sampling is deprioritized until a real registered float32 cost
  arm establishes the claimed headroom.
- Dense full K3 is deferred. Goal-directed/sketched K3 requires evidence that its
  representation bias and rank are small.
- Proper scrambled Sobol is a bounded side test only after the Kerdock reproduction.
- When a lane fails its registered gate, stop it, record why, update current state,
  and proceed to the next supported branch. Do not tune against the failed DECIDE
  data.

### Definition of completion

The task is complete only when:

1. the Phase-2 evaluator is accepted as deployment-truthful within documented local
   limits;
2. every implemented strategic module has a preregistered decision and independent
   Skeptic verdict;
3. two versioned candidates are frozen with honest raw/adjusted score intervals,
   noise floors, exact FLOPs, residual/wall/memory/failure evidence, tails, hashes,
   and code-policy review;
4. public confirmation and any authorized vault decision are recorded;
5. Samuel has explicitly approved and the chosen submissions have been sent and
   nominated, or the repository records why submission was not authorized;
6. `CURRENT_STATE.md`, `GAME_PLAN.md`, `GOTCHAS.md`, `experiments/LOG.md`, and the
   contribution write-up accurately describe the final supported architecture;
7. full preflight and portability are green on the frozen submission files.

Keep going until this definition is satisfied or an explicit Samuel-only gate is
reached. At a gate, report the exact evidence, decision needed, and safe work that
can continue without crossing it.
