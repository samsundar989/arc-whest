# M6.2 / M6.3 refits: results (VAL-100, 1 epoch from R3 fold, R4 recipe, matched local control)

Control `ctl`: 1.848205e-8 at step 112 (start 1.850971e-8).

| arm | VAL at step 112 | Δ vs ctl | per-net (wins) | ΔF | net score | verdict |
|---|---:|---:|---|---:|---:|---|
| **age6** | 1.735936e-8 | **−6.07 %** | −6.04 ± 0.27 % (99/100) | +4.83 % | **−1.54 %** | **hand-over** (`../HANDOVER_M6_age6.md`) |
| jsvd | 1.820761e-8 | −1.48 % | −1.47 ± 0.19 % (77/100) | +1.07 % | −0.43 % | fallback only (substitutes for age6) |
| J3 (patience 3, stopped at 84) | 1.836513e-8 (start) | −0.78 % at matched step 84 | — | +0.33 % | ≈ −0.4 % | fallback only; full refit cancelled |
| acov, lr-mult 1000 | 1.932113e-8 | +4.54 % | — | 0 | — | step-size failure |
| acov, α² ≤ 9, lr-mult 100 | 1.848227e-8 | **+0.001 %** | — | 0 | 0 | **KILLED** (< 0.3 % bar) |
| cfeat, lr-mult 1 | 1.846790e-8 | −0.077 % | — | ≈ 0 | — | underpowered (head moved ≤ 5e-6) |
| cfeat, lr-mult 10 (√0 NaN fixed) | 1.849051e-8 | **+0.046 %** | — | ≈ 0 | 0 | **KILLED**: the head explored (worse at 28 and 56) and settled near 0 (|cf| ≤ 7e-6) |

## acov reading

- Learned ab ≈ −2.5e-4 per layer, with a consistent negative sign across layers 1–14. That is the
  sign of M5.2's physical (1 − 3v·α²).
- va ≈ 1e-5: it did not move.
- Neither buys anything. The late-band variance error the anatomy found (19 %) is network-specific,
  and a per-layer α² gain only reaches its average, which the refits already absorbed. This matches
  the review's rule that averaged physics does not transfer.

## cfeat reading

The five cheap features (C̃ row energy, D21 row sums, K22 row sums, dG/σ⁴, the star part)
carry no signal the refit can use on top of the corrector, which already reads d21e. This agrees
with B2 (local features explain 6–8 % of the orthogonal error) and the review's 'local features
≤ 7 %'.

**M6.3 closed: both arms killed.**
