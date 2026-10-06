# Deep review — 2026-10-07: workers validated, where the gap is, and new mathematics to close it

*Strategist session for Samuel. Inputs: the Oct 6 commits of both workers (main and lane-m), the live AIcrowd pages (Chrome, Oct 6 21:00 UTC), and two research agents with numerical toys (`docs/research/2026-10-07_R1_efficient_higher_order.md`, `docs/research/2026-10-07_R2_structure_and_subspace.md`). [M] measured / primary, [I] inference.*

## 1. Worker review (validated)
**Lane P (main) [M, graded scores re-read from the submission pages].**

| build | what | F/B | raw | graded | server wall med/max | residual max |
|---|---|---:|---:|---:|---|---:|
| v74_ws #334434 | v73 at leaf 64 (insurance B) | 0.23128 | 2.1515e-8 | 4.9761e-9 | 50.9 / 54.1 s | 0.206 s |
| v75_kas #334464 | dead-row pruning R2.0 + D21 rows Dz2.5, fixed K | 0.19020 | 2.2014e-8 | 4.1871e-9 | 81.4 / 86.9 s | 0.253 s |
| v76_kasws #334474 | v75 at block 40 (B) | 0.20416 | 2.2018e-8 | 4.4952e-9 | 51.7 / 56.3 s | 0.216 s |
| **v77_kash #334487** | v75 + dead birth columns (Hd2.5) | **0.17981** | 2.2021e-8 | **3.9596e-9** | 79.4 / 91.5 s | **0.306 s** |
| v78_kashws | v77 at block 40 (B) | 0.19296 | — | gated, waits for A/B slot (4.226e-9 projected) | — | — |

- Quality: careful and honest. Two twin bugs were found and fixed by parity checks, stale results were voided, the Strassen trap (K < 1024 loses the 5th level at block 32) and the 8-FLOPs-per-gather billing were caught. Every role sits at its threshold optimum (R 2.0, Dz 2.5, Hd 2.5, each with a cliff below); refits gave nothing.
- Against my P2 prediction (F/B 0.145–0.165): reality 0.180, for measured reasons — always-on columns carry κ4 hubs; the corrector's D21 feature limited D; fixed K and Strassen leaves eat part of the saving.
- **Watch:** v77's worst residual is 0.306 s (v73: 0.258 s; cap 0.4 s). Keep new builds ≤ 0.33 s.
- **Gap:** acceptance still used DEV+VAL because Lane V's FRESH panels were not merged into main. Fix first thing Oct 7.

**Lane V+M (lane-m) [M].**
- V: fresh-net generator reproduces organizer weights bit-for-bit; GPU labeller (N = 1e8, 66 s/net) validated unbiased against organizer labels and the exact layer-1 form; FRESH-A (100), FRESH-LOCK (200), FRESH-D (100) built; v73 reads 4.702e-9 on FRESH-A (+1.8 % vs graded), paired v73 − v68 = −9.00 % ± 0.45 pp vs graded −8.81 % — the panels predict the server.
- M: factored AUGMENT drops 4 of 91 terms; restoring them gains 26.6 / 18.6 / 15.6 / 16.8 % at n = 256 / 384 / 512 / 768 (flat), all from the κ3 × covariance-edge family (E21b ≈ 80 %; triangle ≈ 0). **On 16 DEV nets at n = 1024, ARC factored AUGMENT beats v73 raw 16/16 (1.79e-8 vs 2.32e-8) with no learned parts; unfactored ≈ 1.50e-8.** Closed on cost (young-only E21b keeps 2.5–8.7 % for +14–27 % FLOPs).
- M4 (neighbor-feature corrector): never beat v73 on VAL — closed.

## 2. The board (Oct 6 21:00 UTC) [M]
J2W 1.5e-9 (raw 1.47e-8, ~0.10 B), marius 1.6e-9 (1.14e-8), Luna 1.6e-9 (1.40e-8), mliston 1.7e-9 (1.46e-8), mk1981 1.8e-9 (1.72e-8), suliman 1.9e-9 (1.57e-8), oliver_chen 1.9e-9 (1.37e-8), thegamers / ben3 / AndreasHad04 2.1e-9. Top-10 ≤ 2.1e-9, top-20 ≤ 2.7e-9, top-25 ≈ 3.1e-9; us ≈ #43 at 4.0e-9 (display). **Nine teams at raw 1.37–1.57e-8** — the level of ARC's unfactored AUGMENT. Gap to #1: ×2.64 = accuracy ×1.50 × cost ×1.76.

## 3. Where the cost is [I, from the v73 census scaled by the v77 pruning; exact v77 census requested]
Recent exact tier ≈ 59 % (≈ 115 GF D21 readout + 117 GF transport), old tier ≈ 28 % (≈ 113 GF), covariance ≈ 9 % (36 GF), other ≈ 4 %. The recent tier is still the largest single cost; the old tier's cost still grows with depth. Engineering levers left: always-on D21 rows (−4 to −7 %), alive-block covariance (−2 to −3 %), join (−1 to −4 %), Strassen leaf for compact products (−1 to −3 %), layer-0 (−0.3 %): ≈ −10–15 % → F/B ≈ 0.155. Reaching 0.10 B needs a representation change nobody has found for us; the literature confirms compression of κ3 *content* cannot get there (R1 §1).

## 4. New mathematics found this session
1. **Dressed legs (Dyson resummation; R1 §5.1).** AUGMENT's κ3 births add eight heavy-edge families (κ3 (2,1) edge, κ3 repeated at the hub, κ4 (2,2) edge, doubled covariance edge; each in two mirror orientations). All share SIMPLE's hub leg, so to first order they fold into one dressed leg X̃ = w1∘C + H with H[a,j] = ½ w2_a D21[a,j] + (w3_j/w2_j)(½ w1_a D21[j,a] + ¼ w2_a D22[a,j] + ½ w2_a C_aj²). Births become Sym(X̃ ⊗ 3w2 e_j ⊗ X̃) = SIMPLE + all eight families + a small second-order spurious term — **AUGMENT births at SIMPLE cost (~0 FLOPs)**. Strategist verified each family's coefficient and hub weight against ARC's `fac1/fac3`; R1's toy (which omitted the hub-repeated D21 and κ4 families) put the net correction at 1.5–2 % of D21, i.e. ~5–8 % raw by 504aldo's ε² law, with partial sets possibly harmful (families cancel).
2. **Gain-mode ("super-sample") law (R2 §1.5).** A bias-free ReLU net is positively homogeneous; at depth the top covariance direction is the input-dependent overall gain (|cos(u₁, m̂)| = 0.98 at layer 16; gain CV ≈ 8 %, near log-normal). Modelling z = g·ẑ gives zero-parameter templates D3 = 6c²μσ², D21_ij = 2c²(μ_jσ_i² + 2μ_iC_ij), D4 = 12c²σ⁴, K22_ij = 4c²(σ_i²σ_j² + 2C_ij²), Q31_ij = 12c²σ_i²C_ij with c² = m̂ᵀCm̂/‖m‖² computable from the chain. They explain 52–77 % of the slices at layer 16 unfitted (83–92 % with one β). This explains *why* 504aldo's λ·C_off law works and why a (2,2)-slice regression on C collapses with depth (the right template is a product, σ_i²σ_j² + 2C_ij²). Use: a better κ4 closure at ~0 FLOPs; a write-up-grade mechanistic result.
3. **Readout is not the problem (R2 §2.3).** With exact per-neuron cumulants, Edgeworth to κ4 is exact at layer 16 (bias² 7.6e-11). Higher cumulants (κ5+) and subspace readouts are not needed; what matters is how accurately κ3/κ4 are *propagated*.
4. **Early layers (R2 §3).** Generic mean errors injected at l₀ ≤ 4 keep only 2–11 % of their energy by layer 16; errors along the mean/gain direction keep 27–64 %. Our early-layer contribution is likely ~1 % unless we carry coherent errors — a ½-day check decides.
5. **Dead ends confirmed:** Tucker/TT/HT/CP compression of content, DLRA/DO for content, c-index-only readout projection, sketching, streaming decompositions, conditional-Gaussian/sampling mixtures, κ5+ in the readout.

## 5. Plan (PLAN rev 19)
Accuracy first (it multiplies and is nearly free): dressed legs + gain-mode κ4 (Lane P2), with the ARC-b ablation telling us which AUGMENT ingredients carry the −23 % and whether dressing transfers (Lane M2); dense-κ3 oracle for κ3×edge with cheap forms (spike rank, gain-mode E21b, learned proxy); coherent early-error check. Cost second: always-on D21 rows, alive-block covariance, join, Strassen leaf. Target ≈ 2.4–3.0e-9 by Oct 13 (top-15 to top-25 today); stretch 2.0e-9.

## 6. Honest odds [I]
#1 ≤ 3 %; podium ≈ 5 %; top-10 ≈ 15 % (needs ~−45 % from here); top-20 ≈ 45 %. The write-up prize is live: pruning fingerprint, gain-mode law, dressed legs, κ3×edge measurement, anatomy and transfer law are mechanistic results ARC asks for.

## 7. Risks
Refit instability when physics changes (use lr 1e-4); partial-family inclusion making things worse; v77's residual (0.306 s) close to the gate; the flopscope fix/re-evaluation (we use no `out=`, `where=` or symmetry tags — low exposure); designation UI timing (~Oct 10).
