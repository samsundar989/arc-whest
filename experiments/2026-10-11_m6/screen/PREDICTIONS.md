# M6.2 unrefit screen: predictions, recorded 2026-10-11 ~02:30 UTC, BEFORE any run

Files are the strategist's price.py patches applied to `estimator_k3_v82c_r2.py` (R2 constants),
as in P6.1's screen (`experiments/2026-10-11_p6/screen/mkvariants.py`). Panel: VAL-100
(`twin.py eval --raw --panel hold100`). Δraw is paired against the same-base control file.

**The base question.** On R2, unrefit fold+Habs is +65.6 % raw vs base (3.1528e-8 vs 1.9033e-8,
P6.1 `val_raw_*.json`). That is far off equilibrium, so a lever's unrefit Δ on it may not
transfer to the refit chain. Every lever is screened on both bases:
- `fold+Habs+X` vs `fold+Habs`, the handoff's base;
- `fold+X` vs `fold` (1.8838e-8), near equilibrium.

When the two disagree, the `fold` base is the one I trust for the refit decision, and the
refit (against a matched control) decides.

## Strategist (handoff)

| arm | break-even Δraw | prediction |
|---|---:|---|
| age 6 | −4.65 % | −4 to −6 % |
| jsvd | −1.08 % | −0.3 to −2 % |
| J3 | −0.24 % | −0.2 to −0.6 % |
| p2svd | < +0.08 % | 0 to −0.3 % |

## Mine (Lane M), Δraw vs the same-base control

| arm | on fold | on fold+Habs |
|---|---:|---:|
| age6 | −4 to −6 % | −3 to −7 % (noisier) |
| jsvd | −0.2 to −1.0 % | −0.5 to −2 % (H's dropped columns put more weight on the join basis) |
| J3 | −0.1 to −0.5 % | −0.1 to −0.6 % |
| p2svd | −0.3 to +0.3 % | −0.3 to +0.3 % |

Expected hand-overs after the unrefit screen:
- age6 is near break-even, so a refit decides it;
- jsvd most likely misses its −1.08 % break-even on fold;
- J3 most likely clears its −0.24 % break-even narrowly;
- p2svd will not reach −0.3 %, so it is not handed over.
