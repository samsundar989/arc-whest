# Hand-over to Lane P: age 6 (M6.2), 2026-10-11

**Lever.** `_SHARED_AGE = 5 → 6`: each exact third-order lineage stays exact one more layer before
it joins the old tier.
- price.py name: `fold+age6` on the v83 stack (`age(6)` in `docs/research/2026-10-11_p6price/price.py`).
- The patch is one line. It applies to `estimator_k3_v83_fold.py`, and to the twin candidate
  (`experiments/2026-10-11_m6/refit/cand_age6.py`).

## Evidence (VAL-100 = full 900–999; all paired per net)

| step | arm | control | Δraw | per-net Δ ± se | wins |
|---|---:|---:|---:|---:|---:|
| unrefit, R2 files (price.py on v82c_r2) | 1.77337e-8 | fold 1.88375e-8 | −5.86 % | −5.82 ± 0.27 % | 99/100 |
| twin arm, R2 params (parity) | 1.773564e-8 | — | file +0.011 % | — | — |
| unrefit, R3 fold params | 1.738215e-8 | ctl 1.850971e-8 | −6.09 % | — | — |
| **refit, 1 epoch from R3 fold vs a matched control** | **1.735936e-8** | **1.848205e-8** | **−6.07 %** | **−6.04 ± 0.27 %** | **99/100** |

Refit recipe:
- R4's recipe (R2 mask J=2, the fold arm, S2tsf), at 1 epoch, lr 1e-4 cosine, accum 8;
- both arm and control start from `experiments/2026-10-11_p6/r3/fold/params_best.npz`;
- run locally, with the same code and hardware for arm and control.

Outputs:
- `experiments/2026-10-11_m6/refit/refit_{age6,ctl}/` (params_best.npz, train.log);
- per-net `refit/val/fin_{age6,ctl}.json`.

The refit keeps the whole unrefit gain. The learned parts don't absorb it, because it is exact
information.

## Cost (exact interpreter, price.py on v82c_r2 + fold)

| | F/B | ΔF | ops |
|---|---:|---:|---:|
| fold (v83) | 0.16554 | — | 22,618 |
| fold + age6 | 0.17354 | +4.83 % | 22,113 (−505) |

- Break-even Δraw: −4.61 %. **Net score: −1.54 %**: (1 − 0.0607) × 1.0483 − 1.
- Not priced: age6 stacked with od16. The two touch different things (old-tier age vs Strassen
  leaf depth), so a near-additive ΔF is expected, but price it.

## Memory and wall time

WSL `linux_memory_gate.py`, RLIMIT_AS 8 GiB:
- VmPeak 6,820–7,093 MB vs fold's 6,424 MB (+400 to +670 MB; v53 measured ~+700 MiB). Headroom is
  1.15–1.20×.
- **The lifecycle check is borderline.** With 3 repeats it FAILED (+622 MB, a one-off step on run 3).
  With 5 repeats it PASSED (+47 MB, non-monotone). I read this as allocator noise, not a leak,
  but it sits near that gate's noise.
- Files: `experiments/2026-10-11_m6/screen/memgate_*.json`.
- Wall time is not measured. F is +4.8 % and ops are −2.2 %; your `tools/wall_ab.py` decides.

## Fallbacks if memory or wall time fails a gate

These are nested variants that keep fewer lineages exact. Unrefit, on fold:

| variant | ΔF | Δraw | net |
|---|---:|---:|---:|
| lateage9 (bar 6 from layer 9) | +2.09 % | −2.83 % | −0.80 % |
| lateage11 (bar 6 from layer 11) | +0.89 % | −1.48 % | −0.61 % |

They are `m6levers.lateage(L0)` in `experiments/2026-10-11_m6/screen/`. lateage13 is a no-op
under J=2.

## Closed alongside

- age 7 for late layers on top of age 6: +0.46 % marginal.
- J4 and the jsvd + rank / age4 follow-ups: conditions not met.
- p2svd: flat.

The other M6.2 refits (jsvd, J3) and M6.3 (α-gains, the feature head) follow in the same folder.
