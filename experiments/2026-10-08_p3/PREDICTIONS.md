# Lane P3 predictions (written before each run)

## P3.1a — Strassen leaf (written after the cost reads; no prediction was recorded before them)
Strategist: −1.5 to −3.5 % F at equal raw. Measured (real meter, DEV net 0): block 16 −0.57 %,
block 12 −0.59 %, block 12 + depth 6 −0.94 %, block 8 + depth 6 −0.72 %, block 8 + depth 7 +0.04 %;
ops +8 to +24 %. We already run leaf 32 on 1024-wide products plus an extra level on big ones, so
the public leaf-32→8 gain (504aldo's chain, no Strassen tuning) does not transfer.

## P3.2 — κ4-core loop, VAL-100 unrefit screen (written 2026-10-08 ~03:00 UTC, before the GPU run)
Already seen (CPU smoke, ONE net, full:900, baseline 2.238e-8): core + K211 exact + radial exact
3.40e-8 (×1.52); radial via hub i (memoryless λC) 1.17e-7 (×5.2); folded radial 3.1e-7 (×14);
core without K211 2.4e-6 (×108).
Worker, VAL-100 unrefit vs the dressed baseline:
| arm | prediction |
|---|---|
| (i) core + K211 exact + radial exact | +30 to +70 % |
| (i′) K211 truncated, radial exact | +40 to +120 % (K211 is what keeps the core bounded) |
| (ii) (i) + E21b | within ±10 % of (i) unrefit |
| (iv) (ii) old tier dropped | ×1.5 to ×3 of (ii) |
| (v) (ii) old tier rank 128 | +5 to +20 % over (ii) |
Refit (S2tk, lr 1e-4, 1 epoch TRAIN-900), vs a refit control continued from the same params:
(i) −3 % to +8 % [25 % that it is ≤ −3 %]; (ii) a further −1 to −5 %; (iv) +20 to +60 % vs v80.
Reason for pessimism: the chain's learned gains (λ, feed, tt, q31 modes, corrector) were fitted to the
memoryless core, and the unrefit +52 % on one net says the loop moves the state a lot; ARC's −11.6 %
E21b gain was measured in a learning-free chain.
Strategist (handoff): (i) −3 to −8 %; (i′) within 3 pp of (i); (ii) −8 to −18 %; (iii) further −2
to −6 %; (iv) +15 to +60 % vs v80.
