# Strategist response — 2026-09-10: AUGMENT's *channel*, not its tower; two cheap discriminators; nominee A moves

*Answers `docs/STRATEGIST_PACKET_2026-09-10.md`. Supersedes the Sep 9 response (its plan is void, as the packet says). Serves as the Slice 5 handoff (§5).*

## 0. Verdict on the slice: accepted; the measurements hold, the inversion is correct

The port's death is established by arithmetic on the lineage (8n births/layer vs 2n; 885 vs 240 block-layers; two rank-R contractions with no SIMPLE counterpart) and confirmed by ARC's own FLOP model; the "most generous floor" is still over the hard cap at every Strassen depth. §3.2 is the more important result: **AUGMENT's 2.09× is a property of the exact tower and vanishes under truncation** — capped AUGMENT is indistinguishable from capped SIMPLE. Together with the Sep 5 memory-depth ceiling and the Sep 8 collapse-beats-truncation result, this is now a three-part theorem of this project: *cumulant information in flight pays only when the tower that consumes it is exact*. That is the write-up's central mechanistic claim and it is stronger than any single positive result we have.

The wall lever is closed correctly (the glue is memory-bound; Strassen's arithmetic is a quarter of the wall). The exact-metric lever is closed correctly (1.07× at the live shape, decaying with width as the mechanism predicts). The ladder's over-read (1.2–1.4×) is a usable calibration, not a discard. Three Skeptic corrections changed conclusions; the record is trustworthy.

## 1. Q3 first, because everything else depends on it: what are the leaders running?

Two hypotheses survive the new facts, and they are discriminable in days, not weeks.

**H1 — AUGMENT-class κ₄ in flight through a cheap channel.** The ablation says all of AUGMENT's gain is κ₄'s (n,n) structure. In ARC's factored code that object couples into the flow through two channels: (i) the Wick updates of μ, Σ and the diagonal κ₃ slices — O(n²) per layer, essentially free — and (ii) births of new factored κ₃ columns (the 5 leg blocks + 2 degree-4 blocks) — the 8n/layer that kills the port. **Nobody has measured which channel carries the 2.09×.** If it is (i), then "AUGMENT-lite" — carry the (n,n) κ₄ trace object and its diagonal, transport them through W (two matmuls/layer), consume them in the nonlinear step's μ/Σ/diag-κ₃ updates, and birth nothing — gives AUGMENT's accuracy at SIMPLE's tower cost plus ~3 matmuls/layer. That is the one version of H1 consistent with 0.2 B, and the accuracy coincidence (our 1.91e-8 vs their 1.87–2.13e-8) is what makes H1 the prior favourite. It still requires a 2.7× cheaper tower than our SIMPLE lineage — which brings in the packed-spelling question (§2).

**H2 — a cheaper chain (K2/K3-lite or regenerating) plus a learned per-layer corrector trained on the public 1,100 networks.** Evidence for: Hilton's warm-up description ("factorized 3rd cumulant propagation combined with learned networks that consume the cumulant estimates as features"); Puffi's A716 (10-channel/90-coefficient per-layer fitted adjustment) in a #4 team's pipeline; and the telemetry — 25–35k ops at 5–10 GFLOP/s running to the wall cap is *exactly* what per-layer learned corrections spelled in fnp look like (many small elementwise ops, cheap FLOPs, expensive wall). Six teams converging on one floor is also explicable under H2: everyone trains on the same public labels, so everyone hits the same generalisation floor. Against: the accuracy match to AUGMENT would be a coincidence, and no team has published a corrector of that size.

The two are not exclusive (H1 chassis + H2 layer is the most likely truth). What matters is that **both discriminators are torch jobs on the 24-net set, no build, ≤3 GPU-days combined**, and each, if positive, is directly a shippable lever:

- **T1 (AUGMENT-lite ablation).** In ARC's torch flow, keep `kind=AUGMENT`'s κ₄ object and its use in the μ/Σ/diag-slice updates; suppress the κ₄-born κ₃ columns (or, equivalently, run AUGMENT and project the tower back to SIMPLE's column set each layer while keeping the κ₄ object live). Pre-registered prediction: retains 1.4–1.8× of the 2.09×. Gate: ≥1.5× ⇒ build the port at ~0.55–0.6 B (adjusted ~1.1e-8, #15). ≤1.2× ⇒ H1 is dead in every affordable form.
- **T2 (corrector on chain features).** Per-neuron equivariant head + pooled per-layer context over (α, σ, κ₃ slice diagonals, c₄, gate probability) at every layer of *our* SIMPLE chain, target = final residual, trained on the public 1,100 (labels 1e9) with cross-fitting, held out on the 24 high-resolution nets and on 100 fresh mints. Prediction: 1.15–1.4× (the board caps a *single* team's edge near 1.1×, but nobody has published this on an exact K3 chain). Gate: ≥1.15× held-out ⇒ ship (offline, unmetered; ~500 fnp ops if spelled per layer). Also log whether a per-*layer* in-flight corrector (DAgger-style, Puffi's form) beats the final-residual form — that is the H2 test proper.

## 2. Q1 — the packed spelling: one day of feasibility, not five of building

The premise needs sharpening. flopscope bills by the shapes you issue, so "does it bill the packed shape" is yes by construction; the real question is **whether any of our 797 matmuls has a symmetric operand at all**. The lineage's dominant products — Wᵀ·(n, kn) blocks and (n,n)@(n,kn) — contract *non-symmetric* factor blocks; the identity-leg saving W-EFF already took is the factored form's version of symmetry. The α ≈ 1/6 bound applies to the unfactored order-3 tensor we never materialise. So: **day 1, count the symmetric-operand contractions and compute the theoretical saving on the meter**; if ≥1.5× is on the table, take the remaining four days; if it is ≤1.2× (my expectation), close it and say so. Do not spend five days on a lever whose upper bound is unknown.

If it closes, cost is closed at ~0.5 B and the leaders' 0.2 B remains unexplained — which is itself a finding to state plainly in the write-up: "we can reproduce the leaders' accuracy and not their cost, and we have excluded the two public routes to it."

## 3. Q2 — the κ₄ ceiling: a write-up lane, not a score lane

The ladder says exact κ₄ is 9–10× better than AUGMENT (6.5–8.8× after the over-read discount). Real and important — it says the class's true ceiling is ~2e-9, not 1.9e-8 — but the same slice proved the thing that kills its exploitation: κ₄ information pays only in an exact tower, every κ₄ carrier priced is over the hard cap, and the one team that built a low-rank one is last of the six. With 5.3 weeks, building a κ₄ carrier that fits inside 2^41 and keeps its gain is not a credible path to points. **Score: trap. Write-up: lane** — run the Skeptic's rank-sweep on the ladder's top rung as a background GPU job (one job, no seed block) so the paper can state "the exact κ₄ ceiling at fixed L/n is 9×, and the fraction recovered at rank r is f(r)"; that curve, next to §3.2, is the cleanest statement anyone will have of why κ₄ is hard.

## 4. Q4 — nominees: move A now

The asymmetry is not arguable: one timed-out row costs ~9e-3 on the mean; the depth-3 edge is 2.5e-9; break-even is p ≈ 3e-7, and 1.21× margin under a *measured* 1.6× spread is a tail we cannot bound. **Nominee A moves to Strassen depth 2 (0.604 B, 33 s worst, 2.28×) today.** Then:

- If Rules v14 §6 scores each designated entry independently and ranks the team by the better one, **B = depth 3 (`330327`)** — free upside, and a timeout on B cannot hurt A. If the rule combines them, B = `330212`. The sentence has been requested three times; if it cannot be found in the captured rules text by reading, **post a neutral question on the forum** ("how are two designated submissions combined in the private re-evaluation — is the team ranked by its better one?") — it is a rules question, costs nothing, and the answer changes the pair.
- Depth-2 needs its own graded row before designation (it is priced, not graded): one slot, Samuel's approval.
- Re-designate whenever T1/T2 produce a certified candidate under the same ≥2× rule.

## 5. Q5 — effort split, and Slice 5 (Sep 10–23)

Split from now: **40% discriminators + their ports, 20% packed-spelling feasibility and nominations, 40% write-up.** The write-up is co-equal money and its spine is stronger than our board position; it should be near-final by Oct 3 and every section drafted from the LOG as results land. Its arc: (i) the exact factored K3 at 1024×16 on the counted meter and the three structural identities; (ii) the AUGMENT attribution ablation; (iii) the theorem — memory-depth ceiling, collapse beats truncation, AUGMENT's gain dies under truncation; (iv) the κ₄ ceiling on the ladder with the ladder's calibrated over-read; (v) the null results (exact metric decays with width; Strassen's wall is memory-bound glue; a 2× method that cannot fit its own budget); (vi) the board's shared floor as AUGMENT-class accuracy, with what we could and could not reproduce of the leaders' cost.

Execution order:
1. Day 1: nominee A → depth 2 canary (approval needed); packed-spelling feasibility count; T1 and T2 queued on the GPU.
2. Days 2–4: T1 result → if ≥1.5×, AUGMENT-lite port begins (same certification ladder, ≥2× margin); T2 training runs.
3. Days 4–7: T2 held-out result → if ≥1.15×, spell the head in fnp, certify, canary; the κ₄ rank-sweep runs in the background.
4. Days 7–13: integrate whatever landed; re-designate; write-up sections (i)–(vi) drafted.
5. Kill criteria: T1 ≤1.2× and T2 ≤1.1× ⇒ score is closed at nominee A's ~2.0–2.3e-8 (top-25 to top-30), and Sep 24–Oct 17 is hardening (1000 fresh mints through the nominee bytes, determinism twins, 3×100 official soaks), nominations by Oct 10, and the write-up.

Hand back by Sep 23: T1, T2, rank-sweep and packed-count numbers with pre-registered predictions; the depth-2 graded row; the §6 quote or the forum answer; the designated pair; write-up draft status per section; anything that contradicts this response.

## 6. What I am and am not claiming

Established by measurement: SIMPLE 3.78e-8 graded; AUGMENT 1.91e-8 in torch; AUGMENT unaffordable as factored; its gain dies under truncation; wall and exact-metric levers closed; ladder over-reads 1.2–1.4×; exact κ₄ is ~9× over AUGMENT at fixed L/n. Hypotheses: H1 (κ₄ via the cheap channel) and H2 (cheap chain + corrector) — T1 and T2 decide them. Retired from Sep 9: "the gap is cost and cost alone" (it is cost *and* an unidentified channel); the ≤0.25 B target (no priced lever reaches it); the Strassen-depth plan; the ≥2× margin rule stands and is now applied.
