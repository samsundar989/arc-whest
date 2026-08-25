# Claude Code handoff — begin P1-PRE, then Phase 1A

Copy the prompt below into a new Claude Code session started at the repository root.
It assumes no chat history. Repository state is authoritative if a SHA or fact has
changed since this handoff was written.

---

## Prompt

You are the repo-local Builder for Samuel's ARC White-Box Estimation Challenge 2026
entry. Work from:

```text
C:\Users\samsu\Documents\arc-whest
```

Continue autonomously through safe work, but stop at every Samuel-only prediction,
vault, submission or material scope-expansion gate. Never submit or touch a vault
without Samuel's explicit approval.

Do not rely on prior chat. First read, in this order:

1. `AGENTS.md` — binding operating contract and work classification.
2. `docs/CURRENT_STATE.md` — current status and immediate action.
3. `experiments/LOG.md` from the bottom — Phase-0 ACCEPT and carried D1–D3.
4. `docs/GAME_PLAN.md` — sole active architecture and execution plan.
5. `docs/VALIDATION.md` — binding measurement law.
6. `docs/GOTCHAS.md` — operational traps.
7. `docs/SKEPTIC.md` — independent-review contract.
8. Only the raw artifacts directly named by the slice you are executing.

Inspect the worktree before editing and preserve unrelated user changes. Do not
edit `docs/HUMAN_MODEL.md`, any frozen estimator archive, the seed ledger, vault
state or `estimator.py` during the work authorized below.

### Current state

- Phase 2 is live through 2026-10-17 23:59 UTC at width 1024, depth 16,
  `B = 2**41`.
- Official stack: WhestBench 0.16 / flopscope 0.12; lambda 0; residual cap 0.4 s;
  wall cap 120 s; solution memory class default 8192 MB; setup timeout 5 s.
- Phase 0 is **CLOSED**. A fresh independent Skeptic returned literal ACCEPT on
  G1–G7 for reviewed HEAD `952bd61`; closure commit is `7367fdb`.
- The local counted-FLOP meter matches the official subprocess report op by op:
  19 operations, zero disagreements, `137,600,326,656` FLOPs for current K2p1.
- No Phase-2 submission exists. The shipped width-1024 path is basic K2p1 and is
  far behind the leaders.
- The strongest frontier hypothesis is a 16-basis width-1024 Kerdock/Walsh MUB
  angular carrier with antipodes and exact radial integration. The leader's
  32,768-direction fingerprint supports this; the private implementation remains
  unknown.
- P01 is DESIGN and already exposed. Do not use it as independent evidence. No
  new block is authorized in this handoff.
- Phase-0's accepting review carried D1–D3. They do not reopen Phase 0, but they
  are known evaluator defects and must close before carrier construction.

## Start now — P1-PRE evaluator hardening

This is **IMPORTANT** instrument work under `AGENTS.md`, despite using no data.
Before editing, append a short written prediction/knowledge target to
`experiments/LOG.md`: each fix should change only refusal/read-only behavior, leave
all Phase-0 scientific and score evidence unchanged, and add no false-green path.
Record the baseline HEAD and relevant artifact hashes. Do not claim a seed.

Repair exactly these defects:

1. **D1 — literal-True worker-restart witness.** Remove the branch in
   `local_sim/test_deployment_envelope.py` that calls `check(..., True, ...)` for a
   `"this file"` witness. Emit a structured worker-restart result from this suite
   and verify it like every other witness. With `--skip-worker-death`, the witness
   must be explicitly unprovable/failing; it must never print green.
2. **D2 — suite resolution.** `verify_witness` must resolve the `suite` named by
   the witness to that suite's result artifact rather than always loading the
   lifecycle artifact. Refuse unknown suite, missing results, wrong check ID,
   failed check and detail mismatch. Keep a positive control. Tests must exercise
   the public resolver path rather than injecting `results=None` behind it.
3. **D3 — truthful read-only mode.** `--no-write` must not regenerate
   `lifecycle_self_test.json`, deployment evidence or any tracked file. Prove this
   with before/after hashes and a clean `git status`; do not print a read-only
   claim unless it is mechanically true.

Keep the intervention narrow. Do not refactor unrelated evaluator code or repeat
the Phase-0 research. Run focused positive and negative tests, then full
`preflight.ps1 -Full` on a quiet host and `verify_portability.py`. Store a complete
transcript, append the result and knowledge delta to `experiments/LOG.md`, and
update `CURRENT_STATE.md` only if the status changes.

Then launch a **fresh independent Skeptic** that has not seen Builder reasoning.
Give it the written target, exact diff, raw transcripts/artifacts and
`docs/SKEPTIC.md`. Require literal ACCEPT, REJECT or PROVISIONAL. Record the verdict.
Do not begin Phase 1A until P1-PRE is accepted.

## Next — Phase 1A mathematical construction, no truth data

Phase 1A is also **IMPORTANT** because it changes a probe's mathematics. Start it
as a separate slice after P1-PRE synthesis and ACCEPT. Record its own written
prediction/knowledge target before implementation. Claim no seed and do not touch
`estimator.py`.

The existing `local_sim/carriers.py` construction is length 256: GR(4,7), 65,536
signs. Width 1024 requires a direct implementation of the known GR(4,9)/real-MUB
construction. The primary target is a deterministic subset of 1, 2, 4, 8 or 16
bases, with antipodes, without materialising the full 1,048,576-word Kerdock code.

Do the work in this order:

1. **Write the mathematical specification first.** Cite the exact construction,
   basis indexing, primitive/basic polynomial or equivalent symplectic-spread
   representation, phase/sign convention and deterministic ordering. Separate
   established mathematics from implementation choices and from the inference
   that the leader uses 16 bases.
2. **Build two paths.** Implement the efficient direct generator and a genuinely
   independent verifier/reference using different logic. Use reduced dimensions
   where exhaustive comparison is cheap.
3. **Prove invariants.** For every supported basis count verify unit row norms,
   within-basis orthogonality, cross-basis absolute coherence
   `1/sqrt(1024) = 1/32`, antipodal closure, expected uniqueness and deterministic
   order. Check exact integer/Walsh correlations before float conversion where
   possible.
4. **Prove radial handling separately.** A bias-free ReLU MLP is positively
   homogeneous. Validate the exact Gaussian radial factor independently from the
   angular construction and show the all-layer mean convention is correct.
5. **Do not overclaim angular exactness.** Measure the 16-basis subset's degree-4
   angular error, including aligned/adversarial probes. Do not call the subset an
   exact spherical degree-4 design unless that identity is actually proved.
6. **Build a dense/full-batch reference first.** At 32,768 x 1,024, one float32
   point buffer is 128 MiB, far below the 4-GiB single-array cap. Reproduce the
   leading 32,768-forward cost fingerprint with explained overhead. Chunking is an
   optimization variable, not a correctness assumption; compare any chunked or
   algebraically cheaper spelling against the dense reference.
7. **Account honestly.** Record counted FLOPs, point/weight/temporary memory and
   setup behavior. Use flopscope-compatible immutable operations for any code that
   may later inform a submission, but keep this slice out of `estimator.py`.

Phase 1A's exit gate is mathematical and operational, not statistical:

- zero construction, reference, determinism and invariant failures;
- exact explained row counts and counted-cost model;
- no unproved degree-4 or leader-mechanism claim;
- full preflight and portability green;
- raw evidence and knowledge delta in the LOG;
- fresh independent Skeptic verdict.

No truth MSE, variance-reduction or score claim is permitted in Phase 1A.

## Stop before Phase 1B

After Phase 1A is independently accepted, synthesize the result into
`CURRENT_STATE.md` and `GAME_PLAN.md`, then stop and ask Samuel to review the
Phase-1B STRATEGIC contract and restate his S2 prediction before any block is
claimed.

The proposed Phase 1B design must use a new Phase-2 DESIGN block (P02, not P01),
pair common random numbers, report truth-noise floors, freeze counted FLOPs, and
report paired arithmetic competition-score differences without making a promotion
decision on DESIGN data. Residual is not priced in Phase 2; report worst-observed
`0.4 / residual` headroom and zero failures. Include
1/2/4/8/16-basis carriers, matched IID, ordinary orthogonal/Haar and K2 controls,
plus a persistent 100-MLP soak that retains the first/warm-up row and measures
residual drift and peak memory. Do not claim or run that block from this handoff.

## Later roadmap

After structured-carrier DESIGN and a fresh SINGLE_USE DECIDE block, proceed only
through supported branches in `GAME_PLAN.md`: faithful recurrent augmented K2,
checkpoint response closure, integrated robustness, public CONFIRM, an explicitly
authorized Phase-2 vault, and up to two frozen nominees. Every strategic slice is
preregistered before data and independently reviewed after it. Never fit public,
DECIDE or vault data. Never submit without Samuel's explicit approval after
official validation and a zero-failure subprocess mini run.

At every stopping point, report the exact evidence, current SHA, decision needed,
and safe work that can continue without crossing the gate.
