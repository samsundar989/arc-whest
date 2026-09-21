# SSC — Puffi's separate-universe κ₄ response closure, extracted

*Source: `docs/refs/algorithmic_contribution_puffi.pdf`, "Separate-Universe Response
Closure for Deep ReLU Networks: submission #327725", Cipo on behalf of Team Puffi.
Fetched and extracted 2026-09-08. This file exists so the next session does not
have to re-read a PDF; the PDF is committed beside it.*

**PRIVATE — competitor analysis. Never the forum.**

## What it is, in one line

Instead of carrying a K=4 state, **rebuild an approximate κ₄ after every ReLU from
`p ≤ 4` covariance-response matrices** with a small fitted coupling:

    κ₄,ijkl = Σ_{a,b=1..p} Λ_ab ( B_a,ij B_b,kl + B_a,ik B_b,jl + B_a,il B_b,jk )      (Eq. 2)

Each symmetric `B_a` is "a direction in which the current covariance could change"
— a background response, by analogy with super-sample covariance in cosmology. `Λ`
is a `p×p` coupling matrix, **fitted offline and frozen**. `Λ` may be indefinite,
so Eq. (2) is a *signed ansatz*, not a literal mixture. **Every contraction acts on
the `B_a` factors; the dense `n⁴` tensor is never formed.**

## The layer update (§3.2)

1. **Read the state** — `m`, `C`, structured κ₃ carriers, and the factored SSC
   response from the preceding layer.
2. **Apply `W`** — transports every represented cumulant exactly.
3. **Apply ReLU** — a Wick/Hermite expansion updates `m`, `C`, κ₃. This is the
   approximate deduction–projection step.
4. **Repair and refresh** — add fresh cross-family κ₃ atoms, apply the frozen
   late-layer drift corrector, construct up to four new `B_a` from the live state,
   reinstall Eq. (2), and retain **one signed response-memory term** from the
   previous update.

## THE PART THAT MATTERS MOST FOR US — the readout is a PER-NEURON SCALAR κ₄

With `w_j` the incoming weight vector of next-layer neuron `j`:

    q_aj = w_jᵀ B_a w_j                κ₄,j = 3 · q_jᵀ Λ q_j                        (Eq. 3)

**`κ₄,j` is a per-neuron scalar** — the marginal fourth cumulant of the next
preactivation. It is consumed as the **fourth-order Edgeworth contribution to the
next ReLU mean**.

**Our chain already computes the Hermite term that consumes it.**
`estimator_k3_full.py::_wick` builds `base(4) = (α² − 1) φ / σ³`, which is exactly
the `He₃`-weighted factor a κ₄ correction multiplies. We are a pure K=3 chain and
carry **no κ₄ at all**, so SSC's whole payload into our score is one extra
per-neuron vector times a term we already have.

**Consequence: handoff §3b (per-neuron diagonal κ₄) and §3d (SSC) are the SAME
lever with different κ₄ estimators.** §3b asks how to *consume* a per-neuron κ₄;
§3d is one way to *produce* one. That is the re-scope this document exists to
record.

## The offline fitting (§3.3) — all frozen before evaluation

| item | how it was fitted |
|---|---|
| the six coefficients of `Q` (Eq. C3) | layer by layer, ridge least squares, to the measured off-diagonal target `κ_iiij + κ_jjji` |
| **`Λ`** | ridge least squares to the **measured marginal fourth cumulant of the next preactivation**; the ridge value chosen on separate development networks by the MSE of the resulting fourth-order Edgeworth contribution to the next ReLU mean |
| response-memory coefficient | selected in a separate development study, frozen at **−0.3** |
| drift corrector | a `31 → 32 → 32 → 1` network over 31 deterministic summaries of `m`, `C`, κ₃; fitted on **160** Phase-1 networks, checked on 16 held out; teacher-forced target is the predictable one-step mean drift over a fixed late-layer range — **not** trained through the competition score |
| κ₃ repair | up to 64 fresh cross-family atoms, capped to the 4 highest-scoring before the next transport |

At inference it fits nothing to the test network and propagates no samples. The
four raw modes are applied directly and the resulting `4×4` coupling is
diagonalised at each layer.

## The measured result — and its scope

| claim | value |
|---|---|
| layer-32 MSE vs factorized **K=3** | **−77.2%** (paired-network bootstrap 95%: 68.9–83.2%) |
| layer-32 MSE vs factorized **K=4** | −48.2% (95%: 30.3–59.4%) |
| named-operation cost | `0.501 B` vs `0.373 B` for a parity-checked K=3 port → **+34.3% operations** |
| panel | 30 independently generated Phase-1 networks |
| depth-error exponent, free fit over `L = 8..32` | SSC **2.21** [1.97, 2.51] vs K=3 **2.69** [2.45, 2.94] vs K=4 4.13 |
| graded (#327725) | raw final-layer MSE `2.6597e-06`, adjusted `1.6533e-06`, at `0.622 B` |

**Every number above is at Phase-1 shape: width 256, depth 32.** Ours is **width
1024, depth 16** — four times wider and half as deep.

## WHAT THE PAPER ITSELF SAYS AGAINST IT

Recorded because it bears directly on the go/no-go, and because a competitor's own
caveats are the cheapest possible source of falsification:

- **"At layer 32, the submitted method therefore remains less accurate than the
  sampling references used here."** SSC's raw layer-32 MSE is **7.94× that of
  full-budget Kerdock QMC** on the same panel.
- **"SSC has no identified small expansion parameter and no corresponding bound on
  its truncation error."** The effective-field-theory analogy "does not validate it
  by itself".
- The endpoint score "measures the deployed system as a whole" — the response
  closure, the κ₃ repair, the pair-series terms, the response memory and the drift
  corrector all lie between the clean closure and the submitted number. **The
  −77.2% is NOT attributable to Eq. (2) alone.**
- "Because `n = 256` throughout, this figure tests depth dependence only, not joint
  `L/n` scaling" — so the depth exponent may not transfer to our width.

## A FREE IMPROVEMENT THEY PUBLISHED AND DID NOT SHIP

From the conclusion: a controlled audit kept the same four response modes but
**changed the regression target** — fitting `Λ` to *the mean correction consumed
after the next weight and ReLU update*, instead of to marginal fourth-cumulant
quantities. That **reduced layer-32 MSE by a further 13.6%** and won on 26 of 30
networks, and "the measured reduction came from changing the target, not from
adding modes or from orthogonalization alone." **It was never embedded in #327725.**

If we build any κ₄ closure, fit it to the mean correction it is consumed by, not
to κ₄ itself. That is free information.

## What this implies for our go/no-go, before anything is built

Naive depth transfer: if the MSE ratio scales as `L^(p_SSC − p_K3) = L^(−0.48)`,
then a `0.228` ratio at `L=32` becomes `0.228 × 2^0.48 ≈ 0.318` at `L=16` — a
**~3.1× improvement**, still well above the `≥1.5×` bar. **But the paper explicitly
refuses that extrapolation** (one fixed width cannot establish a joint `L/n` law),
and our width differs by 4×.

**So the transfer is unestablished in both dimensions, and the honest cheap test is
not to build SSC.** It is to measure the CEILING: give our chain an *oracle*
per-neuron κ₄ — the true value, computed offline at any cost — feed it through the
`base(4)` term the readout already has, and measure the final-layer MSE gain. That
bounds **every** κ₄ closure, SSC included, on OUR shape. If the ceiling is under
`1.5×`, §3d is dead without being built; if it is above, we know the target and the
remaining question is only how cheaply the κ₄ can be produced.
