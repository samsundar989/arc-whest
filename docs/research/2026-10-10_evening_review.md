# Evening review — 2026-10-10: the ceiling, the leaders' walls, and what is left

*Strategist session for Samuel, Oct 10 23:10 UTC – Oct 11 00:45 UTC.*

*Inputs:*
- *both lanes' commits, RESULTS and packets (main @ c1b96ae, lane-m @ 34135b3);*
- *the live board (23:40 UTC);*
- *per-network telemetry of 16 leader submissions;*
- *a read-only audit of flopscope 0.12.1 and its GitHub issues, plus two billing probes against the real package;*
- *two new experiments on real nets and exact labels, run on the device VM;*
- *exact interpreter prices for every remaining lever on v82c_r1 (`docs/research/2026-10-11_p6price/`).*

*Tags: [M] measured, [I] inference or judgment.*

## 0. Bottom line

1. **The workers had a strong, clean day.** Three graded bests: v82a 3.2795e-9, then v82b 3.0717e-9, then **v82c_r1 3.0464e-9 (#335253, rank 26)**. That is −8.2 % vs v81d in a day, each landing within 0.15 % of its FRESH-A projection.
   - Lane M has two items ready to integrate:
     - the masked star-Q31 fold: −2.58 % raw vs a matched control (twin v81d); +0.27 % F on v82c_r1;
     - kink-active birth columns: +1.35 % raw (twin v81d); −5.89 % F on v82c_r1, −5.64 % stacked with the fold.
   - Stacked (joint refit −1.31 % raw), they give ≈ −6.9 %, so ≈ **2.84e-9 (≈ rank 24)**.
   - Everything else rev 22 proposed was tested with controls and closed: radial terms, paid κ₄ pieces, the hub move, dead-row prunings, covariance pruning.
2. **Yes, the marginal path has a ceiling.**
   - In our representation it ends around **2.6–2.7e-9 by Oct 17 (rank ~20–22)**, and around 2.3–2.5e-9 even with unlimited time.
   - The leaders (1.40–1.60e-9) sit below that asymptote. §2 explains why each source of gains is finite.
3. **The leaders' longer walls are legitimate on the evidence.**
   - J2W's kernel time is proportional to its billed FLOPs at about 300 ps per marginal FLOP, which is the price of ordinary small-leaf GEMMs. Every top team is 1.4–2.1× slower per FLOP than us.
   - flopscope does under-bill a few operations, with about 15 bug reports open and unanswered. Nothing in the telemetry requires them, we use none, and a fix would not move us.
   - What the walls do tell us: the leaders spend their FLOPs on many smaller, squarer products, i.e. a leaner representation than our wide exact lineages (§3).
4. **Two "cheaper third order" ideas were tested on real data today; both are dead** (§4).
   - (a) Low-rank D21 readouts. D21 is 85 % rank-2 by energy, but the remaining bulk still moves next-layer variances by 2.5e-3 at rank 64, about 25× our tolerance.
   - (b) Spherical-design cubature (kaileh57's August "Kerdock" branch). It is only ~2× better than plain Monte Carlo.
5. **Exact pricing of every remaining lever** (§5) found three levers on the H column set:
   - an **asymmetric H**: drop dead columns below α = −2.0 and keep always-on columns up to α = 3.0. It is 1.4 % cheaper than the planned τ2.5 cut and keeps the hub-carrying columns that cost most of H's accuracy.
   - a **deeper H cut** (τ2.0): −11.8 % F if its accuracy holds.
   - an **exact SVD basis for the old-tier join** at +1.1 % F. flopscope documents its top-k SVD price as a randomized SVD's, but computes the exact SVD.
   - It also corrected one of my estimates: more join skipping *raises* F (+0.24 %), it does not lower it.
   - Per-shape Strassen depth now fits under the 26,000-op gate.
6. **Plan rev 23:**
   - harvest: fold + H (with the H frontier), the spend items, refits, per-shape depth;
   - keep both entries robust for the private rerun;
   - run one diagnostic anatomy (Oct 11) to look for any attackable chunk ≥ 10 %.
   - **Odds (judgment):** top 20 ≈ 40 %, top 15 ≈ 8 %, top 10 ≈ 2 %.

---

## 1. What the workers did (Oct 10)

| item | result | evidence quality |
|---|---|---|
| v82a (`fnp.block` + `nan_to_num`) | graded 3.2795e-9, bit-identical to v81d, −1.22 % F; the server bills `block` as 0.12.1 does | meter = interpreter to the FLOP |
| P5.1 bundle | join-skip at L14–15 **−1.14 % raw** and −0.9 % F; fused advance; sandwich split; V8b (dead D21 columns, with a per-layer d21e energy ratio ρ = 1.12–1.68) −0.16 % unrefit | per-item twin + meter |
| closed in P5 | leg rows on the D set **+66 %**; K16 = 800 +6.8 %; old rebuild on D rows +10.6 %; covariance input pruning +24 % | one cause: dead neurons' D3 matters (their means sit in the ReLU tail, where κ₃ is first-order) |
| v82b | graded **3.0717e-9**, 0.11 % above FRESH-A | FRESH-A 1.0779× vs v81d, 100/100 |
| v82b_ws (B) | graded 3.1433e-9 | WSL PASS, wall 0.916 |
| R1 refit → v82c_r1 | **graded 3.0464e-9 (rank 26)** | FRESH-A 1.0083× vs v82b, 99/100 |
| M5.1 fold (recent lineages) | −1.23 % unrefit, **−2.58 % after refit vs matched control**; +0.25 % F and +243 ops on v81d, zero products | matched control; parity 4.8e-5 |
| M5.2 radial Sym(μ⊗κ₃) | real in the labels (fitted D4 coefficient = physical v) but +0.03 % after a properly powered refit → killed | |
| M5.3 regression | free pieces explain ΔR² ≈ 0.03–0.06 each; paid pieces ≤ 0.02 → none built; the κ₄ D31 channel in our chain is worth the fold's ~2.6 % | |
| M5.4 hub move | fidelity gate failed (age 1 R² 0.95–0.99; ages ≥ 2 0.5–0.8); but **kink-active H without the move: +1.35 % after refit**, and the fold survives it (joint −1.31 % vs control) | the count-table version is unmeasured (see P6.1) |

Assessment: disciplined and honest. Predictions were on record, every comparison had a matched control, and every build had a FRESH-A. Several rev-22 estimates were wrong, all in the optimistic direction:
- two-sided D's leg-row half;
- the "tiny" accuracy cost of K16 and the old rebuild;
- the hub move;
- the paid κ₄ pieces.

The workers measured and closed them correctly.

## 2. Is there a ceiling on marginal gains? Yes.

Each source of gains has a floor, and we are close to every one of them.

**Cost (F).**
- About 60 % of FLOPs are exact third-order lineages, moved and read at O(n³) per lineage per layer.
- No exact format is cheaper:
  - ARC's factorization is the floor;
  - coherent tiers lose ×4.8–6.6;
  - low-rank readouts fail, measured today (§4a).
- What could be pruned has been, or has been ruled out:
  - dead rows cost +6.8 to +66 %;
  - always-on rows cost +34–87 %;
  - the old tier sits at its elasticity optimum (rank 192 cost +21–44 % raw).
- Left (exact prices, §5):
  - kink-active columns: −5.6 % at τ2.5, up to −11.8 % at τ2.0 if accuracy allows;
  - per-shape depth: −1.3 %.
  - More join skipping raises F.
- **Floor ≈ 0.14–0.15 B.**

**Accuracy from physics.**
- The refit ceilings put the error in the κ₃ slices (53 %) and the covariance (40 %), and Step 3.1 places the κ₃ drift in the **all-distinct κ₃ births**.
- The missing pieces are n⁴ diagrams:
  - the covariance triangle;
  - κ₃×edge;
  - the full κ₄(a,a,b,c) → κ₃ feedback, which Lane M3 tied to K=4's per-neuron B_r blocks.
- The κ₄ D31 channel is exhausted at ~2.6 % (the fold).
- Physics that is "averaged" rather than network-specific does not transfer: E21b, the κ₄-core loop, exact dressing and the radial terms all lost their gain after refit. Learning has already absorbed their average effect.

**Accuracy from learning.**
- Local per-neuron features explain ≤ 7 % of each layer's residual, and the corrector already takes 5.2 %.
- More data adds nothing: 2,048 d8b nets.
- More capacity hurts: hidden 64 < 32.
- Learned closures are ≤ 0.75 %.
- Refits pay ~0.5–1 % per epoch, shrinking.

**Where this lands.** F ≈ 0.15 at raw ≈ 1.75–1.80e-8 gives **≈ 2.6–2.7e-9 (rank ~20–22)** by Oct 17. Every remaining lever is a one-off a few percent in size; they do not compound into a 2× jump.

**What would change this:** a representation that carries the all-distinct κ₃ (and the κ₄→κ₃ feedback) at third-order cost. We have not found one. §4 and the rev-22 work closed the candidates we could test.

## 3. The leaders' wall times [M]

**Telemetry** (100 nets per submission, Oct 10):

| team | billed GFLOP / kernel-s | r(kernel, FLOPs) | marginal ps / FLOP | intercept |
|---|---:|---:|---:|---:|
| ours (#334948) | 5.07 | — (constant FLOPs) | — | — |
| J2W (#335075) | 2.43 | 0.69 | 298 ± 32 | 26.7 s |
| Luna (#335138) | 3.00 | 0.68 | 270 | 16.0 s |
| oliver (#335098) | 3.49 | 0.63 | 322 | −10.2 s |
| mliston (#333607) | 3.73 | 0.68 | 214 | 13.8 s |
| mk1981 (#335113) | 2.92 | 0.60 | 297 | 10.7 s |
| marius (#335133) | 3.24 | 0.21 (FLOP CV only 1.2 %) | 104 | 59.8 s |
| Andreas (#334942) | 3.74 | 0.15 | 68 | 57.5 s |

Machine noise in kernel time is ≈ 2.9 % CV (ours, at constant FLOPs). With J2W's 3.1 % FLOP CV, pure proportionality predicts r ≈ 0.73; we observe 0.69. So J2W's kernel time is consistent with being proportional to its billed FLOPs.

**flopscope 0.12.1 audit** (installed source = the v0.12.1 tag):
- Intentional discounts:
  - **top-k SVD** bills 4mnk but runs the full SVD: 208× at 1024², k = 32;
  - **symmetric outputs** are billed by unique elements: aliased einsum Gram products 0.5×; `as_symmetric` operands.
- **Open, unanswered bugs:**
  - #273: `var`/`std` with a broadcast `mean=` bills O(n²) for O(n³) work;
  - #270: `polyval` bills by degree, not array size;
  - #275: `out=` broadcast replication is free;
  - stale symmetry tags: #285, #263–#265, #279, #284.
  - Issues #257–#285 (Sep 17 – Oct 5; #275 and #285 filed by AndreasHad04) have no maintainer reply. 0.12.1 (Aug 29) is still the latest release.
- `+=` and item assignment raise errors. Fancy-index gathers count as overhead, not kernel.
- **Two probes against the real 0.12.1 package (Oct 11):**
  - **complex64 is billed correctly.** A 64³ complex matmul bills 2,088,960 FLOPs against 520,192 for the real one (4×). Packing two real products into one complex product (A·(B + iC)) therefore costs 2× more, not less. This is not an explanation for slow-per-FLOP walls.
  - **The top-k SVD discount is documented, not a bug.** `flopscope/numpy/linalg/_svd.py` and `_flops.py:svd_cost` say it bills "the textbook cost of the operation (like matmul), not literal BLAS work", and it returns the exact economy SVD, sliced.
    - It is a legitimate tool. Its real cost lands in fixed per-shape wall time, which fits marius's and Andreas's large FLOP-independent kernel blocks as well as anything else.
    - We can use it too: §5 item 4.

**Reading:**
- J2W's ~300 ps per marginal FLOP is what undiscounted small-leaf GEMMs cost. Our server calibration: leaf 64 → 88–95 ps; leaf 32 → 154–177 ps.
- J2W was already ~340 ps/FLOP on Sep 18–24, before the bugs were reported.
- Andreas's and marius's large FLOP-independent kernel blocks fit fixed-shape work just as well as under-billed work. Nothing requires exploitation.
- Our estimator uses none of the affected calls, so a flopscope fix and re-grade would not move us.
- **The useful inference:** the leaders spend their FLOPs in many smaller, squarer products (e.g. ~12 square n³ products per layer), not in our few wide ones. That is a different and leaner representation, built over months. It is not a FLOP-for-time trade we could copy: deeper Strassen buys us ≤ 1.3 %.

## 4. New-method tests today [M]

### 4a. Is D21 low-rank where it matters? No.

**Setup:** true pre-activation D21 (FIT net 0, labels N = 1e9), its first-order covariance image E = sym(D_w2·D21·D_w1), and the next layer's weights. We truncate E to rank r and measure the induced error in the next layer's pre-activation variance.

| layer | E energy left at r = 2 / 16 / 64 / 256 | var error rms at r = 2 / 64 / 256 (relative) | E's whole contribution to var |
|---:|---|---|---:|
| 8 | 0.157 / 0.115 / 0.056 / 0.008 | 3.4e-3 / 2.6e-3 / 1.0e-3 | 8.1e-3 |
| 12 | 0.134 / 0.088 / 0.037 / 0.004 | 3.7e-3 / 2.5e-3 / 0.9e-3 | 9.2e-3 |
| 14 | 0.134 / 0.082 / 0.033 / 0.003 | 4.3e-3 / 2.5e-3 / 0.9e-3 | 1.0e-2 |

- Rank 2 (the radial u⊗m structure) holds ~85 % of E's energy.
- The bulk still moves variances by ~2.5e-3 at rank 64 and ~1e-3 at rank 256. The chain's whole per-layer variance drift is ~1e-4 (closure bench).
- So rank-r readouts (O(nkr) instead of O(n²k)) would cost far more accuracy than they save in F. Dead.
- Script: `docs/research/2026-10-10_evening_tests/d21_rank.py`.

### 4b. Spherical-design cubature? No.

kaileh57's public August branch (`agent/phase2-kerdock-frontier`) estimates E[h] on Kerdock bases plus antipodes, using homogeneity (E[h(x)] = E‖x‖·E_θ[h(θ)]). An orthonormal basis plus antipodes is a spherical 3-design. On FIT net 0 with exact labels:

| points | F/B | random orthonormal bases | randomized Hadamard | plain MC |
|---:|---:|---:|---:|---:|
| 2,048 | 0.031 | 1.59e-5 | 1.59e-5 | 3.08e-5 |
| 16,384 | 0.250 | 2.32e-6 | 2.00e-6 | 3.62e-6 |

- Only ~2× better than MC, and 100× worse than our chain at similar F.
- The network function has most of its variance in high-degree harmonics.
- Script: `docs/research/2026-10-10_evening_tests/cubature.py`.

## 5. What is left, ranked by expected score gain

Every F/B here is an exact interpreter price on today's files (`docs/research/2026-10-11_p6price/`). The base reproduces v82c_r1's real meter to the FLOP: F/B 0.16508, 22,375 ops.

**The H frontier** (with the fold; ΔF vs v82c_r1):

| # | lever | F/B | ΔF | ops | Δraw evidence | P | owner |
|---|---|---:|---:|---:|---|---:|---|
| 1 | **v83 = fold + kink-active H**, τ2.5 count table, ranked by \|α\| | 0.15577 | −5.64 % | 23,097 | joint refit −1.31 % vs control (twin v81d). The count/full-drop version is unmeasured: the twin arm kept the residual and S3 on always-on columns, which the estimator drops | 0.85 | P |
| 1b | **asymmetric H** (dead < −2.0 dropped, alive kept to +3.0): 1.4 % cheaper than 1 | 0.15362 | −6.95 % | 23,386 | keeps the always-on hubs, which carried most of H's cost (Hk2.5 +1.37 % vs Hkh2.5 +0.09 %, unrefit) | 0.6 that it beats 1 | P |
| 1c | asymmetric H, −2.0 / +2.5 | 0.15062 | −8.76 % | 23,386 | unmeasured | 0.4 | P |
| 1d | symmetric H τ2.0 | 0.14555 | −11.83 % | 23,360 | unmeasured | 0.25 | P |

**On top of fold + H** (ΔF vs 0.15577):

| # | lever | F/B | ΔF | ops | break-even Δraw | evidence / prediction | P | owner |
|---|---|---:|---:|---:|---:|---|---:|---|
| 2 | per-shape Strassen depth, leaf ≥ 12 | 0.15371 | −1.32 % | 25,473 | — | accuracy-neutral expected; the gate is wall (+15–25 %), **no op-gate change needed** | 0.5 | P |
| 3 | age 6 | 0.16337 | +4.88 % | 22,592 | −4.65 % | v53 measured −5.53 % (net −0.7 %) | 0.45 | M → P |
| 4 | exact top-k SVD join basis | 0.15746 | +1.09 % | 22,233 | −1.08 % | unmeasured; today's basis is one warm-started pass; +1–2 s wall | 0.3 | M → P |
| 5 | join skip J = 3 | 0.15614 | +0.24 % | 22,806 | −0.24 % | J = 2 gave −1.14 %; J = 3 unmeasured | 0.4 | M → P |
| 6 | refits R3+ | — | 0 | — | — | −0.5 to −1 % per round | 0.8 | P (pod) |
| 7 | α-dependent covariance gains + cheap corrector features | — | ~0 | — | — | −0.5 to −2 % | 0.3 | M |
| 8 | adaptive keep counts (Variant B) | — | ≈ 0 | — | — | ≈ −1 % | 0.3 | P (if time) |
| 9 | anatomy v2 → any chunk ≥ 10 % attackable in 3 days | ? | ? | ? | — | — | 0.15 | M |

**Closed by price plus history:**
- Rank 320: +6.26 % F; v67 measured a +2 % net.
- P2 rank 8: +0.93 % F; v53 measured −0.12 % raw.
- `_SHARED_PASSES = 2`: +3.65 % F; the SVD gets an exact basis for less.
- "Spend more on hard nets": per-net MSE CV is only 12 % on FRESH-A, and the worst 10 nets hold 12 % of the error.

**Neutral, retest only on condition:**
- J = 4: +1.09 % F; test it only if J = 3 clears its break-even by ≥ 0.5 %.
- Rank 288: +3.47 % F against v67's −3.45 % raw nets about −0.1 %, inside the noise; retest it only on top of an accepted SVD join.

**Projection:**
- ≈ 2.82–2.85e-9 after item 1, or ≈ 2.76–2.80e-9 if 1b wins (Oct 12);
- ≈ 2.65–2.8e-9 by Oct 15–16, depending on the H frontier, depth and the spend items;
- ≈ 2.55e-9 if 1c or 1d and items 3–4 all land.

**Board cuts now:** top 20 < 2.65e-9; top 15 < 2.35e-9; top 10 < 2.05e-9. The cuts will drift down by Oct 17 as others improve; marius and mliston both improved today.

## 6. Plan rev 23 (Oct 11–17)

Handoffs: `docs/HANDOFF_2026-10-11_LANE_P6.md`, `docs/HANDOFF_2026-10-11_LANE_M6.md`.

- **Lane P6:**
  - merge lane-m; twin support for H ranked by |α| or the asymmetric score, a full-drop mode and a shim SVD;
  - v83 = v82c_r1 + fold + the best H variant (1, 1b, 1c or 1d), with a joint refit;
  - integrate M6.2 winners (age 6, SVD join, J = 3);
  - refits;
  - per-shape depth if wall allows;
  - hardening, FRESH-LOCK, B build; freeze A at Oct 17 12:00 UTC.
- **Lane M6:**
  - anatomy v2 (Oct 11, time-boxed);
  - spend/join frontier screen in the twin (`--raw` on the price.py variant files);
  - α-dependent covariance gains and cheap corrector features;
  - a conditional build chosen from a menu by the anatomy.
- **Pod:** P's refit queue first.

## 7. Decisions for Samuel

1. Approve rev 23. If we hear nothing, the lanes proceed.
2. The pending gate change (ops ≤ 28,500 for per-shape depth) is probably no longer needed. On the H-pruned stack, per-shape depth measures 25,473 ops, under 26,000. Keep it as a standby in case the final stack needs it.
3. Keep one pod through Oct 16.
4. Optional, your call: the $20k algorithmic-contribution prize does not depend on score rank in the same way. ARC reviews "roughly the top 10 (possibly more)". Our κ₄ anatomy, star-κ₄ on κ₃ legs, homogeneity identity and today's negative results are real science. A write-up session could start Oct 14 without touching the score lanes.
