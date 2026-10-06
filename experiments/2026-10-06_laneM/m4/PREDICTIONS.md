# M4 — neighbor-feature corrector screen (twin only): prediction before running

Arm `--arm nbr` (tools/twin/twin.py, lane-m): a zero-initialised additive head on mu_out
beside the shipped corrector; 56 detached, per-net RMS-normalised features per neuron
(own α, Φ, φ, γ3; top-8 |ρ| partners' signed ρ, α, Φ, φ, γ3; similarity-kernel means
exp(−β(1−ρ²)), β ∈ {8, 32}, and log kernel mass; ridge (C⁺α)_i and (R⁺1)_i), MLP 56-32-1.
Stage N1: only nbr/* trains; init = twin-v73 (S2t3 step-736 leaves + G1bg gates).
1 epoch on TRAIN 900, accum 8, lr 3e-4, zero-scale 1.0, VAL (full 900-999) every 28 steps.

Kill rule (handoff): VAL raw improves < 2 % vs the init read → kill.

Worker prediction: VAL raw −1.0 % (80 % interval −0.2 … −2.5 %), i.e. P(≥ 2 %) ≈ 15 %.
Reason: the shipped corrector already sees D21 energy (Σ_j D21²) and the chain carries the
dense covariance; a neighbor's α/Φ is mostly a proxy for the neuron's own C-row, which the
chain propagates exactly. Gains, if any, should be at the deepest layers.
