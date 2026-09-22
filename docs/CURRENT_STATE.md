# CURRENT_STATE — 2026-09-22

*The snapshot. Where we stand, what is running, what is next. Rewritten, not appended.*
*The plan is `docs/PLAN.md`; the queue is `docs/HANDOFF_2026-09-21.md`; every number is
in `RESULTS.md`. Everything before 2026-09-21 is in `archive/2026-09-21/`.*

## Board

Sep 22. **We are at 6.4914e-9** (C10, sub #331816, graded Sep 22 — 16 entries), up from
#71 at 8.752e-9; rank to be read off the board. #1 J2W 2.10e-9 · #2 marius_binner 2.40e-9 · #3 suliman_tadros /
a_s6 2.50e-9 · #5 Puffi / reds 2.80e-9 · #10 mliston 3.50e-9. Ranks 13–41 are trimmed
derivatives of 504aldo's public code at 4.2–5.3e-9; ranks 42–57 are sixteen exact clones
at 5.40e-9.

Phase 2 closes **Oct 17 23:59 UTC**. Team freeze Oct 2. Write-up due Oct 24.

## Best graded

**C10** — `estimator_k3_v43_complete_c10.py` (`fd8d497d`), sub #331816, **6.4914e-9**.
Raw 2.968645e-08, multiplier 0.21867, 100/100 with 0 failures, server residual 0.236255 s
of the 0.4 s cliff (1.69× headroom at 23,421 ops). Graded 0.53% better than its own
projection. **This is the §5 baseline every candidate must now beat by ≥3%**, i.e. a
projected public score ≤ 6.2967e-9.

Previous best: **v31** (`6dde5fd4`), sub #331317, 8.752e-9. Older fallback: **F21**
(`d77dab0e`), sub #330950, 1.5957e-8 — still the *designated* fallback until Samuel
designates otherwise. **No agent ever designates.**

## Best local

C10 is now graded, so the best *local* candidate and the best graded are the same file.
There is no untried candidate on the shelf — **the shelf is empty, and that is the state
D2/D3 exist to change.** C10's raw accuracy is 3% *worse* than v31; its entire gain was
cost (0.30103 → 0.21867 F/B).

Two facts that size the job: even at the 0.1 cost floor C10's predictions would score
2.97e-9, so **accuracy is mandatory**; and the public MIT baseline run on our own nets
(raw 2.38e-8 at 0.253 B) still beats C10, so **we are currently behind a free download**.

## What is running

| item | state |
|---|---|
| D0 submit C10 | **DONE — sub #331816 graded 6.4914e-9**, 100/100, 0 failed. New best. |
| D0b archive | **done** |
| D1 `tools/ev.py` | **done.** numpy reproduces the official DEV raw MSE exactly; cupy backend at 2.03 s/net for screening; panels built and hash-verified. |
| D4 diagram pricing | **done — recommend closing D4b.** Restoring the dropped diagrams is worth 1.19× at n=128 but 1.08× at n=256, i.e. shrinking with width. The same run priced SIMPLE→AUGMENT at ~2.5× and K3→K4 at 3.4–6.5× rising with width; both need a cost side before they mean anything. `experiments/2026-09-22_d4/`. |
| **D2 fitted coefficients** | **next.** The six profiles are already module-level constants in C10 (`RESULTS.md` § D2 prep), so D2(a) is a read-from-`coeffs.json` refactor, not a rewrite. |
| D3 billing polish | parallel, mechanical, each item an exact transform |

Submission slots: 1/10 used on 2026-09-22 UTC (331816); PLAN §5 caps agent canaries at 5/day.

**Nothing is currently submittable.** C10 is the baseline now, so a new candidate needs a
projected public score ≤ **6.2967e-9** to clear the §5 3% gate.

### Evaluating a candidate

```powershell
uv run python tools/ev.py <candidate.py> --panel dev --baseline estimator_k3_v31.py --append
```

numpy, 4 workers, ~40 min for 100 nets — the instrument of record. For screening
use the cupy backend (`WHEST_BACKEND=cupy`, one process, 3.4 min/panel): it
agrees with numpy to +3.2e-05 on a *panel mean* but only ~1e-3 *per net*, so it
ranks candidates and must never produce a submission number. Do not run both
backends at once. Calibration constants and the full account: `RESULTS.md`.

## Architecture, in one paragraph

Deterministic cumulant propagation — mean plus full covariance, a factorized third
cumulant with a young exact tier and a compressed old tier, regenerated κ4 (exact
diagonal plus λ·C_off), Strassen products, f32, single file. This is the leaders' family
and it is not in question. What changes is *how* we improve it: end-to-end **fitted
coefficients** on organizer labels (D2), billing polish (D3), and a cheaper or
regenerated **old tier** (D5) — not more exact-algebra archaeology against 504aldo's code
under a noisy instrument.

## Calibration constants (live)

- DEV 100-net mini score **÷ 1.037** = predicted public score. Four graded
  observations: 1.0386 / 1.0385 / 1.0339 / 1.0426. Deliberately at the low end, so a
  projection is slightly pessimistic and the gate slightly strict.
- One-net f64-twin wall **× 1.5981** = server worst-of-100 wall.
- Op calls **× 11.09 µs** (the worst of three graded anchors) = server residual, against
  a 0.4 s/MLP cliff that FAILS the MLP. 1.5× headroom corresponds to ~24,000 ops.
- FLOPs: **no offset** — the f64 read-only twin is the graded number, confirmed on three
  submissions (C10: 491 FLOPs of 4.8e11, 1e-9 relative).

Full derivations, the anchors and the observed spread: `RESULTS.md` § Calibration.

## Closed — do not reopen (PLAN §9)

Sampling / QMC / control-variate sidecars · Gaussian-mixture closures · generic post-hoc
correctors · exact augmented κ4 · the standalone (3,1)/q31 consumer · rank-32 old-query
readout (v44) · qEc/qEdc bases · CP/Tucker leg compression · f64 · E14 · further
"why does v29's chain differ from ours" archaeology · fresh-mint blocks · the 2× gate.

## Process reset of 2026-09-21

Pre-registered contracts, prediction capture, seed blocks, the DECIDE/CONFIRM/VAULT
tiers, per-experiment Skeptic, dispatch-plan commits, producer sidecars, `preflight.ps1`
as the change gate, `workerq` for evaluations and the 2× release gate are all **retired**
(`AGENTS.md` rule 6). Measurement was never the problem: mints are bit-identical to the
organizers', FLOPs match the grader exactly, and the DEV→graded factor has held to ±0.45%
over four submissions. The problem was throughput — 15 entries in 25 days against ~250
available, while the leaders ran 150–300 each. C10 had been ready and unsubmitted for
three days when this reset began; submitting it was worth 1.35×.

## Known deviations from PLAN §8

Three files PLAN listed for archiving were left in place: `docs/HUMAN_MODEL.md`
(Samuel-owned and mechanically write-protected — an agent relocating it is what that
guard exists to prevent), `docs/SKEPTIC.md` (PLAN §4 keeps the Skeptic for the
designated finals and write-up numbers, so archiving the brief would break a capability
the plan retains), and the `preflight.ps1` entry in `.claude/settings.json` (the agent's
own permission file; the entry is inert now that no contract names preflight as a gate).
