# CURRENT_STATE — 2026-09-23

*The snapshot. Where we stand, what is running, what is next. Rewritten, not appended.*
*The plan is `docs/PLAN.md` (rev 12.1); the slice spec is `docs/HANDOFF_2026-09-22.md` +
`docs/STRATEGIST_RESPONSE_2026-09-22.md`; every number is in `RESULTS.md`. Everything
before 2026-09-21 is in `archive/2026-09-21/`.*

Phase 2 closes **Oct 17 23:59 UTC**. Team freeze Oct 2. Write-up due Oct 24.

## Best graded

**v55** — `estimator_k3_v55_p2r4.py` (`aac641b4`) = v53 + `_P2_RANK` 32→4.
**5.3200e-9**, graded twice identically (#331967, #331980), 100/100, 0 failures.
Raw 2.461e-08, multiplier 0.21618, 23,645 ops, residual ≤0.281 s.
**Wall: worst 114.5 s / 113.8 s** (median ~92.5 s) — see "Wall" below. **Not designated.**

The §5 baseline is now 5.3200e-9: a canary must project ≤ 5.1604e-9.

| candidate | graded | worst wall | note |
|---|---:|---:|---|
| v55 #331967/#331980 | **5.320e-9** | 114.5 / 113.8 s | best score, thinnest wall margin |
| v53 #331880 | 5.560e-9 | 96.5 s | age 5, P2 rank 32 |
| v50 #331863/#331939/#331989 | 6.192e-9 | 88.7 / 87.7 / 104.1 s | safest wall; the designate-B candidate |

**Designation is Samuel's alone. No agent ever designates.**

## What the Sep 22 handoff found (single-leg tower, T0–T3)

**Falsified.** On DEV 100 every "cheap old-source" arm loses one to two orders of
magnitude against the unshared exact chain T0 (raw 2.364e-08): T2 a=1 (old sources
reduced to their diagonal-born `M d(sb)` block) 48×; T1 a=3 (old sources out of D21)
15.5×; the pure two-leg block alone 14×; even the rank-32 `Z` residual alone 6.4×.
T3's fitted regeneration recovers 0.09% on DEV. The old diagonal block alone is worth
almost nothing; old content can be *compressed* (v53's shared basis: +7.7%) but not
*dropped or diagonalised*. Consistent with #329896 (memoryless κ3, raw ≈8.5e-7, 36×).
T4/T5 were not built (Step 6, third branch). Details: `RESULTS.md` "Single-leg kill tests".

## The fallback that shipped

Response §6's elasticity table on v53: of hubs, closure, shared rank, age and P2 rank,
only the **P2 residual rank** had score elasticity < 1. Rank 4: −4.49% score on DEV
(+0.94% raw, −5.38% FLOPs), FIT agrees (+0.98% raw). **Rank 2 is a cliff** (per-net up to
3.2×), rank 1 is +1,175% — do not go below 4. Age 6 on top buys only 0.25% for
~700 MiB. Closure and hubs pay for themselves by a wide margin (+17%, +5% if removed).

## Wall — the open risk

- The server got noisier on Sep 23: identical v50 bytes went from worst 87.7 s (03:00 UTC)
  to 104.1 s (~08:00 UTC), 15 nets > 100 s.
- Age-5 chains run ~5–8 s slower at the median than the FLOPs+ops model says (v55 median
  92.5 s ≈ v53's 92.7 s, ~5.5 s above v50 under the same server state). The model
  projected v55 at 91.3 s worst. It needs a memory/age term before it is trusted again.
- One net over 120 s is TIME_EXHAUSTED. v55 on a noisy day has ~6 s of margin; v50 ~16 s.

## Tools added Sep 23

- `tools/probe.py` — many knob-arms on one imported module, cupy, paired by construction,
  resumable. Screening only. `PROBE_GPU_LIMIT_GIB` caps the pool (default 20; the unshared
  chain needs >12).
- `tools/probe_fit.py` — CMA-ES on a linear knob vector on FIT (T3's ρ).
- `tools/variants.py` — emit single-constant variants of a candidate and meter each on the
  real f64 twin.
- `estimator_k3_p60_probe.py` — v53 + `_D21_AGE/_LBLOCK_AGE/_RHO/_KEEP_AGE/_LB_PART`.
- `tools/ev.py --mem` fixed (the WSL gate needs `--repeats ≥ 3`; 2 read UNAVAILABLE).

## Operational notes

- ev.py sizes its pool from free RAM (11 GiB/worker for these chains). Under a loaded box
  it drops to 1 worker (~2 h) and can still hit MemoryError; a run with failed nets is not
  cached and is not a row — rerun it.
- cupy's pool is uncapped by default and grew to the whole 32 GiB card over 19 nets once.

## Next

1. Samuel: designation (v55 score vs v50 wall margin), given the wall data above.
2. Refit the wall model with a memory/age term on the five v50/v53/v55 rows.
3. If wall margin matters more than 4%: v50 + P2 rank 4 (age 4) is the obvious candidate
   (cheaper than v50, same wall profile); measure it.
4. κ4-on-M (response §2) is untested on the exact tier of the v53/v55 chain; it is the one
   idea from the Sep 22 response that T2 did not refute.

Slots: 4 of 5 used on UTC day 2026-09-23.
