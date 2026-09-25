# CURRENT_STATE — 2026-09-25 01:30 UTC

*Where we stand right now. The plan is `docs/PLAN.md`; every candidate and result
is `RESULTS.md`; the open decisions are `docs/STRATEGIST_PACKET_2026-09-25.md`.*

## Best graded

**5.0443e-09** — `estimator_k3_v57_corr.py`, submission #332129, 100/100, 0 failed.
Was 5.3200e-09 at the start of the Sep 24 slice, so the slice was worth 1.0546x.

## The designation pair (Samuel designates; agents never do)

| role | file | graded | score | implied noisy worst |
|---|---|---|---:|---:|
| **A** | `estimator_k3_v60_ws400.py` | #332223 | 5.3387e-09 | **69.1 s** |
| B | `estimator_k3_v56b_wallsafe.py` | #332063 | 5.7076e-09 | ~68 s |

A dominates B, so the second slot currently buys nothing. See the packet §3-Q2.

## What the chain is

Deterministic cumulant propagation: exact mean and covariance; kappa3 as per-layer
factored lineages, those older than 5 layers compressed into a shared rank-256
basis; kappa4 regenerated with fitted per-layer lambda; Strassen products; float32;
stdlib + flopscope only. **Plus, since Sep 24, an in-loop learned mean corrector** —
a per-layer 33->32->32->1 tanh MLP, weights fitted by sequential teacher forcing,
embedded as base64 float32, costing +0.03% FLOPs and +541 ops.

## Blocked, and needing Samuel rather than measurement

1. **PLAN section 5's 3% gate is binding against our own best.**
   `estimator_k3_v59_corr400.py` would grade 4.9763e-09 (the best this project has
   produced, 1.35% better than #332129) and cannot be submitted.
2. **How do the two designated entries combine?** The Rules v14 text we hold says
   only "up to two designated submissions are rerun on fresh MLPs". The answer
   decides whether v59's extra 7% of score is worth its thinner wall margin.

## Closed by measurement — do not reopen

Single-leg, N1, tiered ranks, P3 at any rank, per-net lambda, the exact Gaussian
closure tail, joint refit, deeper Strassen (wall), K4 and unfactored AUGMENT
(priced out ~6x on score), adaptive age (-0.72% against a 2% gate), the recursive
corrector feature (no gain), hidden 64 (uniformly worse than 32), and the
formula-fix reading of the residual anatomy (concentration has no consistent sign).

## Two calibration facts corrected this slice

- The organizer label noise floor is **6.8e-10**, not 9e-11 (AGENTS.md was wrong by
  7.6x). Layer 1 is our exact closed form, so its MSE measures it directly.
- DEV->graded is **1.038**, lowered from 1.042 after two rows graded worse than
  projected. Change it in `RESULTS.md` and `tools/ev.py` together.
