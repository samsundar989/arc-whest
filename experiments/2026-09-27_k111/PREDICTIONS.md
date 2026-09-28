# K111 diagnostic — predictions (written before the substantial run)

Tool: `tools/k111_diag.py` (step0 / run / report). Step 0 already passed (`step0.json`):
both identities to ~1e-15 on dense symmetric tensors; S transposed, multiplicity 2 and an
un-zeroed diagonal all fail at 0.1–10; W orientation confirmed (label var_next from Wᵀ C W at
1e-7, Wᵀ-swapped at 0.11–0.26).

## Hypothesis

The κ3 drift (Step 2 ceiling: exact D3+D21 at every layer is worth ~53% of the refit final MSE)
is not visible in the marginal K3 / pair K21 closure outputs (closure bench), so it should sit
in the all-distinct (AD) content as the next layer consumes it.

## Expected (point guesses, 4 FIT nets 0–3)

- AD is a large part of the next-layer slices: ‖AD‖/‖D3_next‖ 0.3–0.9 at mid depth, larger
  for D21 than D3.
- Label AD noise floor from K21 (rep1): 1–5% of ‖AD‖ for D3, lower relative for D21.
- Model AD error (free-running) 5–20% relative at mid/late depth, clearly above the floor
  (≥ 2×); in exact-input mode about half of that.
- AD error carries ≥ 70% of the model's D3_next / D21_next error; the rank-4 P2 repeated-slice
  mismatch is a minor term (< 20% of the error norm).
- Downstream (free, no refit): replacing AD at all layers recovers most of the FULL (exact
  D3+D21) gain; REP-only < 30% of it; CMP-only small (< 5% final MSE).
- Span: per-source-neuron gates on the existing two-leg / hub / inherited factors reduce the AD
  error by < 30% (most of the missing content lies outside their span); the next-neuron
  rescaling fit does no better.

## Resources

workerq, GPU: twin forward ~5 GiB VRAM + label/tower buffers → declare 10 GiB VRAM, 10 GiB RAM,
4 CPUs. Smoke (net 0, both modes + arms) first to measure time and peak VRAM; then nets 0–3.

## Stop condition

- If AD error ≤ 1.5× the measured noise floor at every layer ≥ 4 → stop at 4 nets: "not above
  noise".
- If the all-layer AD arm moves final MSE by < 2% → stop at 4 nets: "not score-relevant".
- Expand to 20 nets (0–19; rep1 exists only for 0–7) only if both are above those bars and the
  4-net picture is not already unambiguous.
