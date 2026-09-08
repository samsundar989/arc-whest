# Strategist Packet — 2026-09-08

*Self-contained. Assumes no repository access and no shared chat history. Every
number is measured and its source is named. Written because repository evidence on
the cost half is now exhausted, and the accuracy half has had two of its three
planned arms priced or falsified.*

**PRIVATE. Contains competitor analysis. Never the forum.**

---

## 1. The task, in one paragraph

ARC White-Box Estimation Challenge 2026, Phase 2. Given the weights of a
bias-free He-init ReLU MLP (**width 1024, depth 16**), predict every neuron's
expected post-ReLU activation under `x ~ N(0, I)`, as a `(depth, width)` array, by
analysis rather than by sampling. Score = `final_layer_MSE × max(0.1, FLOPs/2**41)`,
**lower is better**. λ = 0, so cost is counted FLOPs only. Hard per-MLP limits:
**residual wall time `0.4 s`** (exceeding it FAILS the MLP, zeroing it at
multiplier 1.0), **total predict wall `120 s`**, memory `8,192 MB`. The prize is
decided by a **private re-run of up to two designated submissions on fresh MLPs**,
so nominee robustness matters more than public rank.

## 2. Where we are

| | |
|---|---|
| our best graded, nominee-grade | **`2.031e-08`** adjusted (`330327`) |
| our best graded, any | `1.835981e-08` (`330290`, not nominee-grade — §4) |
| two days ago | `8.501e-08`, rank `#47` |
| now | roughly **`#27`** |
| board `#1` | `3.80e-09` |
| the board's raw-error floor | **`1.87e-08 – 2.13e-08` across the entire top-12**, regardless of budget from `0.18 B` to `0.79 B` |

Our raw final-layer MSE is `3.78e-08`. **The leaders' raw floor is `~1.9e-08`.** We
are within `2x` of it on raw error and have spent the last two days on the cost
half of the identity.

## 3. What we have established, with measurements

**The architecture.** An exact factorized third-cumulant (K=3) chain, ported to
`flopscope.numpy`, carrying the whole third-cumulant tower exactly as wide `(n, kn)`
blocks. All-layers MSE (`1.559e-08`) is **better** than final-layer MSE — the
depth-propagating signature the whole top-8 shows and a final-row sampler cannot
produce.

**Cost engineering is finished at `~0.54 B` nominee-grade.**

| lever | result |
|---|---|
| `C_L = A_L diag(w2)` structural identities | `0.99937 B → 0.77970 B`, zero accuracy cost |
| **batched Strassen–Winograd** | `0.77970 B → 0.53745 B` at L=3, **exact**: graded raw `+0.0036%`, and the score gain equals the multiplier ratio to `4e-5` |
| aliased-operand self-contractions | **CLOSED — inapplicable.** The `0.5010x` discount requires literally the same buffer; every matmul here contracts two different arrays. No symmetry discount either (`as_symmetric` is unimplemented in flopscope 0.12.1 and would cost more than it saves) |
| generation truncation, last-layer capping | **CLOSED — losing trades**, measured 2026-09-05 |

**The binding constraint is now WALL, and it is new.** Strassen trades few huge
matmuls for thousands of small ones; arithmetic intensity collapsed `215 → 17
GFLOP/s` and kernel wall rose `7.97 → 51.8 s` *despite fewer FLOPs*. L=4 reached
`114.68 s` of a `120 s` cap. **The strategy canon's `≤0.35 B` target is unreachable
by this route.**

**Two measurement laws now hold, both gate-scanned.**

- *Residual is per-operation*, `10.7–11.2 µs/op` across four graded submissions
  spanning `2,690 → 15,393` operations (`4.3%` spread) and across a change of
  operation *type*.
- *Graded wall varies between runs of identical bytes*: v2.7 graded twice moved its
  median `−0.6%` and its **maximum `+51.8%`**, 16/100 rows. Any nominee margin must
  survive `1.60x`.

## 4. The accuracy half — two of three arms are down

The score is `raw × multiplier`. We have taken the multiplier from `0.78` to
`0.54`. **The remaining factor of ~5 to `#1` is almost entirely raw error**, and:

**§3a fitted renormalisation — FALSIFIED for the readout.** Four nested closed-form
arms, fit on 96 nets, quoted on 24 disjoint high-resolution nets:

| arm | params | final-layer MSE | gain |
|---|---:|---:|---:|
| none | 0 | `4.065813e-08` | `1.0000x` |
| one global scalar | 1 | `4.040971e-08` | `1.0061x` |
| scalar per layer | 16 | `4.036979e-08` | `1.0071x` |
| scalar + offset per layer | 32 | `4.035167e-08` | `1.0076x` |

A fitted scale removes **`0.75%`** against a predicted `20–50%`, and 31 extra
degrees of freedom buy another `0.15%`. **The residual is structure, not scale.**

**§3d the Puffi SSC κ₄ response closure — provisional NO-GO through its readout.**
SSC rebuilds κ₄ from `p ≤ 4` covariance-response modes with a ridge-fitted coupling;
its readout is a **per-neuron scalar κ₄** consumed as the fourth-order Edgeworth
term. Our chain already computes the Hermite factor that term multiplies. And our
own oracle, run 2026-08-21 with cumulants from an independent sample stream, prices
it:

| arm | share of the closure's error |
|---|---:|
| propagated `(μ, σ)` **state drift** | **`94.61%`** |
| 3rd-cumulant map | `4.01%` |
| **4th-cumulant map** | **`0.46%`** |

**A perfect per-neuron κ₄, handed free to the readout, is worth `0.46%`.** Puffi's
`−77.2%` is a whole-stack result at width 256 / depth 32, against a baseline `112x`
worse than our chain, and their paper states SSC remains `7.94x` worse than
full-budget sampling.

**What is NOT priced**: SSC's actual mechanism is *propagation* — κ₄ reinstalled
after every ReLU and transported through the next linear layer, changing the state
the next Wick update consumes. That is exactly where the `94.61%` lives, and
nothing has measured it. **The one experiment we intend next** is an `O4` oracle
arm carrying true κ₄ in flight; `O4 − O2 < 1.5x` kills the lane.

**Also known, from the same oracle:** with exact cumulants the Edgeworth map is
essentially perfect — `O3`'s residual is indistinguishable from its sample noise.
There is no headroom in a better cumulant *map*, only in a better *state*.

## 5. THE QUESTION FOR THE STRATEGIST

Everything above says the same thing from three directions: **the error is
propagated state drift, and every correction we have priced acts on the wrong
thing.**

- `94.61%` of the closure's error is `(μ, σ)` drift, not cumulant shape.
- Per-layer scalar calibration is exhausted at `2.96x`; the per-neuron
  scale-invariant state contains nothing beyond it; a fitted scale on the output
  buys `1.0076x`.
- The cumulant *map* is already perfect given exact cumulants.
- The whole top-12 sits on a shared raw floor of `1.87–2.13e-08` **regardless of
  budget spanning 4.4x** — which says the floor is a *bias* of the shared method
  class, not a variance anyone is buying down.

So:

1. **Is the `~1.9e-08` floor a property of cumulant-truncation estimators as a
   class?** If it is, no amount of K3/K4 refinement reaches `#1`, and the question
   becomes what the leaders are doing that is *not* in that class. If it is not,
   what is the specific term that breaks it?
2. **Given that `94.61%` of the error is propagated state drift and that a better
   cumulant map cannot touch it, what reduces the drift itself?** Candidates we can
   see: a better off-diagonal covariance map (K2's first approximation at width
   1024 is the Price order-1 truncation of the post-ReLU off-diagonal covariance,
   `c_ij Φ_i Φ_j`, and that error enters `var_pre = diag(WᵀΣW)` which sums over all
   of `Σ`); carrying κ₄ in flight; or an offline-learned corrector on chain
   features (§3e), the only remaining arm that *adds information*.
3. **Is the cost half worth any further effort at all?** We are at `0.54 B`
   nominee-grade with wall binding. Going to `0.35 B` would buy `1.53x`; closing
   half the raw gap would buy more. Where should the remaining slice go?
4. **Nominee strategy.** Two submissions may be designated. Is the right pair
   `{safe, aggressive}` — L=3 at `2.031e-08` with `1.94x` wall margin, and L=4 at
   `1.836e-08` with `0.65x` under measured contention — or two safe ones?

## 6. What we would need to be told to act

A named term, mechanism, or estimator class, with the reason it should reduce
*propagated state drift* rather than cumulant-map error — plus the cheapest
measurement that would falsify it on 24 networks before anything is built. This
project's failure mode has been building levers that act on the readout when the
error is upstream; the packet's purpose is to avoid a third one.

## 7. Confidence and known defects in our own record

Two errors were caught by an independent Skeptic in the last 48 hours and are
corrected in `experiments/LOG.md`: a graded report's **row 0 was quoted as the
aggregate** (propagating into three files), and a **gate was written on the paired
adjusted score when the multiplier is a per-arm constant**, making its interval
`250–2000x` coarser than the effect it nominally tested. Both are recorded as
errata. Treat single quoted figures here as needing that check; every number in
§§2–4 has been recomputed from raw artifacts since.
