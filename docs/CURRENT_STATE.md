# CURRENT_STATE — 2026-09-21

*The snapshot. Where we stand, what is running, what is next. Rewritten, not appended.*
*The plan is `docs/PLAN.md`; the queue is `docs/HANDOFF_2026-09-21.md`; every number is
in `RESULTS.md`. Everything before 2026-09-21 is in `archive/2026-09-21/`.*

## Board

Sep 21, 184 teams. **We are #71 at 8.752e-9** (v31, sub #331317, Sep 16 — 15 entries
total, none since). #1 J2W 2.10e-9 · #2 marius_binner 2.40e-9 · #3 suliman_tadros /
a_s6 2.50e-9 · #5 Puffi / reds 2.80e-9 · #10 mliston 3.50e-9. Ranks 13–41 are trimmed
derivatives of 504aldo's public code at 4.2–5.3e-9; ranks 42–57 are sixteen exact clones
at 5.40e-9.

Phase 2 closes **Oct 17 23:59 UTC**. Team freeze Oct 2. Write-up due Oct 24.

## Best graded

**v31** — `estimator_k3_v31.py` (`6dde5fd4`), sub #331317, **8.752e-9**.
Raw 2.9074e-08, multiplier 0.30103, 100/100 with 0 failures, worst wall 68.13 s of 120,
worst residual 0.140 s of 0.4, VmPeak 5,434.6 MB. This is the §5 baseline every candidate
must beat by ≥3%.

Fallback: **F21** — `estimator_k3_strassen_L4_f21.py` (`d77dab0e`), sub #330950,
1.5957e-8. Still the designated fallback until Samuel designates otherwise.

## Best local

**C10** — `estimator_k3_v43_complete_c10.py` (`fd8d497d`), DEV 100-net mini
**6.767e-9** → projected public **6.526e-9** (25.4% better than v31), raw 3.0950e-08 at
multiplier 0.21865, 23,405 op calls, worst local wall 60.26 s, VmPeak 5,754.8 MiB.
Its raw accuracy is 3% *worse* than v31; the entire gain is cost.

Two facts that size the job: even at the 0.1 cost floor C10's predictions would score
2.97e-9, so **accuracy is mandatory**; and the public MIT baseline run on our own nets
(raw 2.38e-8 at 0.253 B) still beats C10, so **we are currently behind a free download**.

## What is running

| item | state |
|---|---|
| **C10 canary submission** | **PACKAGED AND GATED, NOT SENT.** `experiments/2026-09-21_c10_canary/c10-canary.tar.gz`. All seven PLAN §5 gates pass (table in `RESULTS.md`). The `whest submit` call was refused by the agent session's permission classifier, not by a project gate. **Needs Samuel to run it or to grant the permission.** |
| D0b archive | **done** |
| D1 `tools/ev.py` | **done, acceptance met.** C10 on DEV reads 6.767843e-09 against the official 6.767156e-09 (+3.2e-05). Panels built and hash-verified; cupy backend at 2.03 s/net; numpy panel confirmation running. |
| D4 diagram pricing | driver written and smoke-tested (`tools/d4_price_diagrams.py`); real runs due Sep 22 |
| D2 fitted coefficients | the next big item, Sep 23–26. The cupy backend is what makes its ~400 evaluations affordable at all. |

Submission slots: 0/10 used on 2026-09-21 UTC; PLAN §5 caps agent canaries at 5/day.

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

- DEV 100-net mini score **÷ 1.037** = predicted public score.
- Local worst wall **× 1.3097** = server worst wall.
- Op calls **× 10.8–11.2 µs** = server residual, against a 0.4 s/MLP cliff.

Full derivations and the observed spread: `RESULTS.md` § Calibration.

## Closed — do not reopen (PLAN §9)

Sampling / QMC / control-variate sidecars · Gaussian-mixture closures · generic post-hoc
correctors · exact augmented κ4 · the standalone (3,1)/q31 consumer · rank-32 old-query
readout (v44) · qEc/qEdc bases · CP/Tucker leg compression · f64 · E14 · further
"why does v29's chain differ from ours" archaeology · fresh-mint blocks · the 2× gate.

## Process reset of 2026-09-21

Pre-registered contracts, Samuel-prediction capture, seed blocks, the
DECIDE/CONFIRM/VAULT tiers, per-experiment Skeptic, dispatch-plan commits, producer
sidecars, `preflight.ps1` as the change gate, `workerq` for evaluations and the 2×
release gate are all **retired** (`AGENTS.md` rule 6). Measurement itself was never the
problem and is unchanged: mints are bit-identical to the organizers', FLOPs match the
grader, and the DEV→graded factor has held to ±0.3% over three submissions. The problem
was throughput — 15 entries in 25 days against ~250 available, while the leaders ran
150–300 each.

## Known deviations from PLAN §8

- `docs/HUMAN_MODEL.md` was **not** archived. It is Samuel-owned and mechanically
  write-protected by `scripts/guard_paths.py`; an agent relocating it is the exact thing
  that guard exists to prevent. One file of clutter, left for Samuel to move if he wants.
- `docs/SKEPTIC.md` was **not** archived. PLAN §4 keeps the Skeptic for the two
  designated finals and for write-up numbers, so archiving the brief while keeping the
  role would break a capability the plan retains.
- `.claude/settings.json` still allowlists `scripts/preflight.ps1`. That file is the
  agent's own permission configuration; the entry is inert now that no contract names
  preflight as a gate.
