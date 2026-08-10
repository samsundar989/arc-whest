# session-start — arc-whest

*Canonical workflow. Agent-independent. Adapters: `.claude/skills/session-start/`,
`.agents/skills/session-start/`.*

**Read-only. Implement nothing.** The output is a control panel and, when needed, a
teaching pass. If the session needs work done, that is `next-slice`.

This is also **the entry point when the driving agent changes.** An incoming agent has
no access to the previous agent's chat history and must not ask for it. Everything
needed is in the repository; this workflow is how it gets reconstructed.

## Step 1 — read the durable state, in this order

Cheap first, expensive only if needed.

| source | what to extract |
|---|---|
| `docs/CURRENT_STATE.md` | the clock, standings, closed lanes, open questions, evaluator trust, next actions |
| `docs/HUMAN_MODEL.md` | **Samuel's own beliefs — and whether it is filled in at all** |
| `experiments/LOG.md` (tail, newest at bottom) | the last 2–3 entries: what was measured, on which block, with what verdict |
| `git log --oneline -8` | what actually landed, and whether the tree is clean |
| `docs/GAME_PLAN.md` (**rev 12 only**) | current strategy. Earlier revs are history — do not read them all |
| `docs/GOTCHAS.md` | only if a number is about to be quoted or a lane reconsidered |
| `docs/MOONSHOT.md` | only if the cumulant lane is in play |

Then check live state cheaply:

```powershell
uv run python local_sim/phase2_checklist.py    # is round 1429 live yet? dataset tags? boards?
git status --short
```

Report the vault touch counter and preflight status from `docs/CURRENT_STATE.md`
rather than spending a run, unless something suggests they have drifted.

## Step 2 — is the human model blank or stale?

Do not eyeball this — it is checked mechanically, identically under every agent:

```powershell
uv run python scripts/human_model_state.py
```

It prints `BLANK`, `PARTIAL`, `STALE`, or `FILLED` with the filled-slot count and the
reason. `BLANK` and `PARTIAL` mean most of the nine `>` answer slots in
`docs/HUMAN_MODEL.md` are empty. `STALE` means the file's stated bottleneck, next
experiment, or baseline disagrees with `docs/CURRENT_STATE.md`; the script flags the
candidates, and you confirm by reading both.

- **BLANK / PARTIAL / STALE → go to FIRST-ORIENTATION MODE below** and skip the
  control panel for now. An orientation panel aimed at someone who cannot yet read it
  is wasted.
- **FILLED → produce the control panel**, and separately note any point where
  Samuel's stated belief now conflicts with repository evidence. Say so plainly, cite
  the evidence, and **propose** the correction. Never edit that file.

## Step 3 — the control panel

Ten items, tight. Tag every claim **CONFIRMED** (measured here, with the artifact),
**INFERRED** (follows from measurements plus an assumption — name it), or
**SPECULATIVE**.

1. **Primary objective** — including that the score lane and the write-up lane are
   now co-equal, and which one the calendar favours today.
2. **Current validated baseline** — graded, vault, and CONFIRM numbers with tiers.
   Never quote a local absolute residual.
3. **Current bottleneck.**
4. **Evidence that it is the bottleneck** — the measurement, the block, the date.
5. **Critical execution/evaluation path involved** — real files and symbols.
6. **Most important recent learning** — usually the last LOG verdict.
7. **Three highest-value unresolved uncertainties.**
8. **Up to three mechanistically different hypotheses**, where the bottleneck admits
   them. Different *mechanisms*, not three tunings of one idea.
9. **Recommended next uncertainty-reducing move** — and say whether it is a
   diagnostic, an evaluation repair, an experiment, or research. It is often not code.
10. **What Samuel needs to understand to evaluate that move** — name the concept, and
    offer to teach it in three minutes.

Standing context that shapes every panel in this repo:

- The prize is a **private re-run of one designated submission on fresh networks**.
  Board rank is not the target.
- **Accuracy evaluation is E2; cost evaluation is E1.** Any recommendation resting on
  a cost delta must say so.
- Lanes closed in `docs/CURRENT_STATE.md` stay closed absent new *external* evidence.

End with the recommended next command. Usually `next-slice`; `cold-start` if the
model is filled but looks thin; `routine <task>` if the session's purpose is
mechanical.

---

# FIRST-ORIENTATION MODE

Triggered automatically when `scripts/human_model_state.py` reports BLANK, PARTIAL, or
STALE. Samuel has a CS degree and is new to ML and probability. **Do not dump the
repository.** Teach the ~15% that lets him direct the project, using real paths and one
real trace.

Budget roughly 15 minutes of his attention. Prefer a diagram and a traced example
over prose.

## What to teach, and nothing else

**1. What is being optimized — the score identity.** This is the Level-4 concept;
everything strategic follows from it.

```
score = final_layer_MSE  ×  max(0.1, C/B)
              ↑                    ↑
        how wrong we are     what we spent
                             C = counted FLOPs + 1e11 × uninstrumented seconds
                             B = 2.72e11
```

Per sample this is `V·c/B` — variance × cost. `N` largely cancels. **Consequence:
making a sample cheaper is worth exactly as much as making it less noisy.** That one
line explains why half this project's work is about cost.

We currently sit at `C/B ≈ 0.20`, above the 0.1 floor — so both factors are live.

**2. The critical execution path.** One trace, real symbols:

```
mlp.weights (32 layers of 256×256)
   │
   ▼  estimator.py:455  setup()          builds the Sobol point set once
   │
   ▼  estimator.py:477  predict(mlp, budget)
   │
   ├─ :682  _exact_first_layer(w0)       layer 1 is EXACTLY solvable, no sampling
   │                                      E[ReLU] = ‖W₁[:,j]‖/√(2π), error 1.8e-16
   ├─ :574  _closure(...)                Gaussian closure: propagate mean + covariance
   │                                      layer to layer through closed-form ReLU moments
   ├─ :632  _slice_weights(...)          prune neurons with α < _PRUNE_TAU (:119, = −2.5)
   ├─ :598  _alignment_basis(...)        rotate into the closure's own eigenbasis,
   │                                      rank _BASIS_RANK (:136, = 32)
   ├─         sample N=16,384 (:114) Sobol points through the survivors
   └─ :686  _analytic_only(...)          low-budget fallback (returns RAW closure)
   │
   ▼ (depth, width) array → grader
```

Show him `_exact_first_layer` and one closure step in the actual file. The single
most useful contrast: **layer 1 is exact to machine epsilon; layer 30 is not** — that
is the whole problem in one line.

**3. The evaluation path — the three tiers.** This is what protects the project from
fooling itself, and it is unusual enough to be worth his understanding:

```
DECIDE   fresh mints, NEW seed block per experiment, never reused   → choose things here
   │      local_sim/seed_blocks.py  (SINGLE_USE_BLOCKS is machine-readable)
   ▼
CONFIRM  the organizers' public split, 1e9-sample truth             → CHECK only, read-only
   │      no constant may ever be fitted here again
   ▼
VAULT    500 sealed networks, touch-counted, 3 permitted touches    → measure the level
          local_sim/vault.py:223 evaluate()
```

Why it exists: the public split steered a month of decisions, so its optimum is partly
a memory of our own search. And a constant chosen on seeds 1–500 then "confirmed" on
the same seeds is not independent evidence — it is the first result's noise, re-served.

**4. Where strategic behaviour lives, and where it does not.**

| worth his attention | not worth it |
|---|---|
| the score identity; the three tiers | the flopscope shim |
| `estimator.py` closure / prune / rotate | the GPU mirror plumbing |
| `docs/VALIDATION.md` (the protocol) | board-scraping tools |
| `local_sim/seed_blocks.py`, `vault.py` | `packed_sparse_probe.py` (112 KB, dead lane) |

**5. The invariants that must not break.** `estimator.py` may import only flopscope,
whestbench, stdlib. Grader arrays are immutable. Randomness derives from `mlp.seed`.
`scripts/preflight.ps1` checks five of these in ~2 minutes; `local_sim/edge_cases.py`
is 30/30 and includes permutation equivariance — the only check that catches a
transposed index producing perfectly plausible wrong numbers.

**6. The current bottleneck**, from `docs/CURRENT_STATE.md`, stated in plain English,
with why it is the bottleneck.

## Then ask at most three comprehension questions

Draw from `docs/HUMAN_MODEL.md`'s cold-start list. Suggested opening three:

1. If we halve the FLOPs a sample costs but the predictions get slightly noisier, can
   that be a win? Why?
2. Why is the public mini split read-only now?
3. Layer 1 is exact and layer 30 is not — what changed?

**"I don't know" is a fine answer.** Teach only that concept, using the real file,
ask one transfer question, move on. Do not lecture on the others.

## Close

- Tell Samuel **what to write into `docs/HUMAN_MODEL.md` himself** — a short list of
  the specific slots he can now fill. Do not edit it, do not draft prose for him to
  paste. The file is only worth something if the words are his. The protected-path
  guard blocks the edit under every agent; that is by design, not an obstacle.
- Name the **strategic question that should drive the next slice**, and stop.
