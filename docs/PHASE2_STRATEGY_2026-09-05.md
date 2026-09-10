# Phase 2 strategy — 2026-09-05: what the leaders are doing, why we are 22× behind, and the plan to pass them

*Prepared for Samuel from a full session: four repo-audit agents, two literature agents, a live crawl of the AICrowd board/telemetry/forum in Chrome, and the Phase-1 write-ups of the current #1 (marius_binner) and #3 (Puffi). Every number below is sourced; inferences are labelled INFERENCE. This supersedes `phase2-strategy-2026-09-01.md` and the 09-10 strategist response where they differ.*

---

## 0. The verdict in one page

**Where we are.** Best public entry 329896 (`estimator_wp18_canary4.py`, the regenerating K3 flow): adjusted **8.50e-8**, raw 8.50e-7, at the 0.1 multiplier floor, 0/100 failures, **rank #47 of 109** today. The board top is **3.80e-9** (marius_binner, raw 1.87e-8 at util 0.205; a brand-new entrant, suliman_tadros, tied at 3.80e-9 after 13 entries). We are 22× off #1 and ~8× off top-10 (1.0e-8).

**What the leaders are doing.** Two levels of confidence. *Family — well-supported by today's telemetry:* the top-8 cluster spends a deterministic (often bit-identical per MLP) 0.18–0.23 B, runs 40–120 s wall per MLP at ~5–10 GFLOP/s kernel throughput (vs 70–85 GFLOP/s for dense-matmul entries like ours and jamespayor's), carries ~25–35k flopscope operations per MLP (inferred from 5–13 s of server overhead at our measured 0.36 ms/op), runs residual time up to 0.29–0.39 s (right under the 0.4 s cap), has **all-layers MSE (7–11e-9) better than final-layer MSE (1.9–2.8e-8)** — the signature of a depth-propagating chain, which our own K2/regen entries share (canary4: 4.1e-7 vs 8.5e-7) and a final-row sampler does not — and sits on a **raw-error floor of ~1.9–2.3e-8 that does not move with budget** (ArtemVoronov at util 0.79 gets 2.33e-8; jamespayor at 0.48 gets 1.96e-8; marius at 0.205 gets 1.87e-8). A sampling-primary method cannot produce those signatures together: its error would fall ~1/N with budget, and the variance arithmetic is decisive — marius's own Phase-1 Kerdock design (2.27× below i.i.d. MC, the best sampler documented here) would give ~2.4e-6 at the ~13k points his Phase-2 budget buys, i.e. he would need ~130× more variance reduction than any published design has. *Specific architecture — INFERENCE:* a **third-cumulant (K3-class) cumulant-propagation chain at roughly ARC's paper constant (c″₃·L²·n³ with c″₃ ≈ 1.5–2 → 0.2–0.25 B) plus a ~2× accuracy layer on top of vanilla K3** fits every signature. The strongest alternative that fits them equally is a K2/K3-class moment chain whose ~2× comes from an offline-learned per-neuron corrector on cumulant features (Hilton's description of the warm-up winners) — and the two are not exclusive. Our own measurements place vanilla full K3 at raw 3.92e-8; the leaders are at 1.9–2.1e-8 — the 2× that Puffi's published "SSC" κ₄ response closure (77% K3-error reduction at 256×32 for 34% more named operations) or a learned corrector on chain features would give. The plan below does not depend on which of the two it is: both are built.

**The single most important correction to our canon.** Rev 9 (Sep 1) retired the "leaders run ARC's K3 at its paper constant" attribution and closed W-EFF (our exact factorized-K3 port, parity 5e-16) because it priced at 1.07 B against a ≤0.35 B screen. That was the right measurement and the wrong decision: today's telemetry says the frontier *is* K3-class at 0.2 B, so the 4–5× implementation-constant gap between W-EFF and the leaders is **the** problem to solve, not a lane to abandon. The two-week program that followed (WP13–WP20, compressing κ₃ inside a 0.077 B flow) established, cleanly, that memory depth cannot be compressed — a real result and the write-up's spine — but it was optimizing for the wrong constraint: the 0.1 floor is a *term in the objective*, not a wall. W-EFF is at 1.07 B — **over budget, and over-budget MLPs are zeroed**, so it needs a ≥20% trim before it can ship at all; at 0.9 B it scores ~3.5e-8 (2.4× better than our nominee); at 0.25 B ~1e-8 (top-12); with SSC-class accuracy at 0.25 B ~5e-9 (top-5); the same at ≤0.12 B wins.

**Do we have the right architecture?** Half of it. The K3 chassis (W-EFF, parity-proven), the truth pipeline, the FLOP meter discipline, and the graded-vs-local calibration are exactly right and ahead of most of the field. What is wrong: (1) the shipped nominee is a *lossy* K3 (regenerating flow, 22× worse than full K3) chosen to fit the floor; (2) no accuracy layer beyond K3 exists (no κ₄ closure, no Edgeworth-4 readout, no fitted/learned correction on the K3 chain — O1 was never run); (3) cost engineering that the field uses (Strassen, f32 everywhere, symmetric/packed contractions under flopscope 0.12.1, dead-neuron pruning, last-layer skipping) was closed as "not needed for the floor"; (4) the validation harness builds float32 weights and never exercised the grader's float64 path (cost: 5 submission slots), pins versions only to 0.16.x, and cannot run the flopscope client/server split. Section 6 gives the target architecture; section 8 the plan; section 9 the infra fixes.

**Realistic outcomes (6 weeks to Oct 17).** Ladder: (L1) W-EFF ported to fnp, trimmed from 1.07 B to ≤0.9 B (last-layer readout skip, dead-column pruning, unread (2,1) slices — jamespayor at 0.48 B proves ≥2.2× is available) and shipped → ~3.5e-8, #47 → ~#38 (1 week, low risk, also our first "real" K3 submission and a hardening exercise). Budget self-audit with per-MLP fallback to v2.7 is mandatory: an MLP over B scores as zeros at multiplier 1.0 (≈0.9), which would destroy the run. (L2) cost-engineered K3 at ≤0.5 B (the levers priced today support 0.35–0.5 B) → ~1.4–2e-8, top-25 (2 weeks). (L3) + κ₄ response closure / Edgeworth-4 / fitted or learned corrector (×2 on raw) at ≤0.35 B → ~5–7e-9, top-8 (2 weeks, parallel with L2). (L4) find the last cost lever to ≤0.2 B and/or accuracy below 1.5e-8 → ≤3–4e-9, #1 (stretch; needs one of the levers in §6 to land). The algorithmic prize ($20k) is served by the same work: the mechanistic story is "K3 at paper constant + κ₄ response closure + the measured memory-depth ceiling", which is exactly what ARC said it wants.

---

## 1. Phase 2 facts (verified today) and corrections to our docs

| item | value | source |
|---|---|---|
| Shape / budget | 1024 wide × 16 deep, no biases, He-init; B = 2^41 = 2.199e12 FLOPs per MLP (= 65,536 dense forward samples at 2n²L = 3.355e7 each; the site's headline shows "≈65,374") | resources page, launch post 18197 |
| Score | mean over MLPs of final-layer MSE × max(0.1, F_m/B); **C_m = F_m, FLOPs only (λ = 0)** | resources page |
| Residual wall time | **capped at 0.4 s/MLP, not priced**; crossing it zeroes the MLP (multiplier 1.0). "Meaningful arithmetic in residual time is a rule breach regardless" | resources page, allowed-code.md |
| Wall cap | 120 s per MLP (joe_wanza's entry runs 99–119 s; Activation-Regions hits 120 s and has 8 non-completed rows) | telemetry |
| setup() | ≤5 s once, off-budget; overrun fails the whole submission | rules |
| Memory | 8 GB for your process; instance 64 GB; **your code 1 core/2 vCPU; flopscope 7 cores/14 vCPU** | resources page |
| Grader stack | whestbench **0.16.1** / flopscope **0.12.1** (client == server), numpy 2.2.6, Python 3.10.21, participant shim 0.1.0. Repo is on 0.16.0/0.12.0; preflight only asserts 0.16.x | every graded report today |
| Weights on grader | **float64** (marius's Phase-1 write-up §8.1 confirms: "weights arrive as float64; NumPy silently promotes… 2.00× loss"; our 2^25 FLOP delta confirms the cast ran on all 16 layers) | 18166 PDF, LOG.md L33305 |
| Submissions | 10 per team per UTC day (first-week amnesty over) | 18197 |
| Nominations | **"Designate up to two Phase 2 entries for the final private rerun that decides prizes"** (resources page; Rules v14). The overview page's "one designated submission" text is stale. Default if none designated: the two best-scoring valid public entries (Rules v14 §6, captured in `experiments/2026-08-28_p2_live/`) | resources page; rules v14 |
| Timeline | team freeze **Oct 2 23:59 UTC**; submissions close **Oct 17 23:59 UTC**; write-ups due **Oct 24**. The overview's "Sep 19" is stale | 18197 |
| Prizes | Phase 2 $100k (place prizes $50k/$20k/$10k + $20k algorithmic); winners open-source within 30 days under OSI licence, public ≥3 years | resources page |
| Phase 1 private results | **not published yet** (board unchanged: ednacob 1.84e-8, J2W, dstepanov). No winner code has dropped. The 30-day clock starts at prize announcement, so "harvest ednacob ~Sep 30" is unfounded — plan without it | Phase-1 board, 18197 |
| Public/private split | fixed hidden 100-MLP suite (same names for every submission), 50 public rows shown; keenanpepper (18199) showed no "easy set" tail; per-MLP adjusted spread for the leader is only 2.75e-9–4.89e-9 | telemetry, 18199 |
| Cost model facts that matter | matmul (m,k)@(k,n) = m·n·(2k−1); f64 = 2× f32; elementwise/reduction 1/elt; gather 4/elt; sort ~4N log N; `flopscope.stats` promotes to f64; aliased-operand self-contractions (`einsum('ij,kj->ik', W, W)`) bill ≈0.5005× of `W @ W.T` (measured in-repo under 0.12.0); 0.12.1's symmetry-tag change says "charges are unchanged"; Strassen is a real saving because billing is by shape | flopscope README/CHANGELOG, 18125, 18127, LOG.md:20929 |

Docs to fix: the project doc `claude/competition-brief.md` (Sep 19 close, 50/day, one nomination — the repo's `docs/COMPETITION_BRIEF.md` is already correct), `docs/GAME_PLAN.md` header (closed-lane table entries for W-EFF/Strassen — see §4), `CURRENT_STATE.md` §297 ("30-day window closes ~Sep 30"), `scripts/preflight.ps1` step 1 (pin `==0.16.1`/`==0.12.1`), `pyproject.toml:7–8`/`linux_stack_setup.sh:12` (same), `local_sim/real_data.py:24` (`REVISION = "v1-phase1"` default is stale).

---

## 2. The board today and the telemetry fingerprint

Full table with min/median/max wall, kernel, overhead, residual and FLOPs for 21 entries is in `experiments/board_snapshots/board_2026-09-05_telemetry.md`. The rows that carry the argument:

| rank | who | adj | raw | all-layers | util | FLOPs/MLP | wall s | kern s | overhead s | residual s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | marius_binner | 3.83e-9 | 1.87e-8 | 7.4e-9 | 0.205 | 4.504e11 **constant** | 70–79 | 60–69 | 9.5–10.3 | 0.29–0.31 |
| 1 | suliman_tadros (new) | 3.80e-9 | 2.03e-8 | 8.1e-9 | 0.186 | 3.9–4.3e11 *variable* | 41–52 | 36–46 | 4.9–5.7 | 0.12–0.14 |
| 3 | J2W | 3.93e-9 | 2.02e-8 | 1.07e-8 | 0.195 | 4.2–4.4e11 *variable* | 99–119 | 88–106 | 10.5–12.3 | 0.34–0.39 |
| 3 | Puffi | 3.88e-9 | 2.13e-8 | 8.6e-9 | 0.181 | 3.9–4.3e11 *variable* | 86–98 | 79–89 | 6.8–12 | 0.18–0.35 |
| 5 | oqaris | 4.30e-9 | 2.06e-8 | 7.7e-9 | 0.209 | 4.60e11 constant | 86–109 | 74–95 | 11.6–13.4 | 0.32–0.36 |
| 6 | fklassen | 5.44e-9 | 2.50e-8 | 1.0e-8 | 0.218 | 4.7–4.8e11 | 80–90 | 69–78 | 11–12 | 0.31–0.34 |
| 8 | hyojun_kwon (Phase-1 write-up: skew-augmented closure + mid-layer control-variate hybrid) | 6.41e-9 | 2.83e-8 | 1.03e-8 | 0.227 | 4.99e11 constant | 82–91 | 74–83 | 8.1–8.6 | 0.23–0.24 |
| 12 | jamespayor | 9.41e-9 | **1.96e-8** | 1.0e-8 | 0.482 | 1.06e12 constant | **16–17** | 12–13 | 3.9–4.2 | 0.10 |
| 26 | ArtemVoronov | 1.83e-8 | 2.33e-8 | 8.8e-9 | **0.787** | 1.73e12 | 44–51 | 39–44 | 5.6–6.1 | 0.15–0.17 |
| 9 | lode_dockx | 7.06e-9 | 4.78e-8 | **0.70** | 0.148 | 3.1–3.4e11 | 32–36 | 26–29 | 6.6–7.0 | 0.19–0.20 |
| 19 | AndreasHad04 | 1.22e-8 | 9.57e-8 | 7.1e-8 | **0.127** | 2.80e11 | 98–108 | 92–101 | 6.4–7.0 | 0.16–0.17 |
| 47 | **us (329896)** | 8.50e-8 | 8.50e-7 | 4.1e-7 | 0.077 | 1.687e11 | 3.5–3.9 | 2.5 | ~1.0 | 0.03 |

What this says, in order of confidence:

1. **The raw floor is a truncation floor, not a variance floor.** Across util 0.18 → 0.79 (a 4.4× range of budget) raw error sits at 1.87–2.33e-8. A method whose error is Var/N would have moved 4×. Whatever the top-12 run, more FLOPs do not help them — they are all sitting on the same bias floor. *That floor is the prize: whoever breaks it, at any cost ≤ B, takes #1 by a wide margin; whoever matches it at ≤0.1 B takes #1 by 2×.*
2. **Their intermediate layers are accurate.** All-layers MSE 7–11e-9 < final 1.9–2.8e-8 for every leader. That is the depth-growth signature of a deterministic propagation chain (error grows with depth), not of a final-row sampler (flat in depth). Our own K2/regen entries share it (canary4 4.1e-7 vs 8.5e-7; v2.7 8.0e-7 vs 1.37e-6), so it separates propagators from samplers but does not by itself say K3 vs K2-class. lode_dockx is the one final-row-only entry in the top-10 (all-layers 0.70) and it is cheap and mediocre (4.78e-8 at 0.148).
3. **Their operation profile is many-small-ops, not dense matmul.** Overhead 5–13 s; at our measured 0.36 ms/op (canary: 2,673 ops ↔ 0.96 s) that is ~15–35k ops/MLP (INFERENCE — the per-op constant may differ for their op mix); kernel throughput 5–10 GFLOP/s vs 70–85 GFLOP/s for the dense-matmul entries (ours; jamespayor). That is what factorized-K3 einsum towers look like (ARC's own code: "wall-clock dominated by a small fraction of FLOPs implemented inefficiently"), though Strassen leaf blocks or chunked loops look similar. What it proves regardless: **Phase 2 does not price wall time** (λ = 0), only caps it at 120 s, and the leaders exploit that on purpose (27k ops, 0.35 s residual, 100 s wall). The repo's residual/op-count anxiety (built when λ = 1e11 in Phase 1) is obsolete as a *price*; it survives only as the 0.4 s cap ≈ 30–35k ops.
4. **Fixed FLOPs per MLP for marius/oqaris/hyojun; variable for suliman/J2W/Puffi.** The fixed ones are pure deterministic chains; the variable ones add data-dependent work (per-MLP routing/pruning or an adaptive sample count). Puffi's Phase-1 hybrid pruned coordinates by probe mass, so their Phase-2 variability is consistent with the same trick on top of an analytic chain.
5. **jamespayor is the dense-matmul K3.** 1.06e12 FLOPs in 16 s (85 GFLOP/s) with raw 1.96e-8: a straightforward dense factorized K3 at c″₃ ≈ 3.9 (1.06e12 / (L²n³ = 2.75e11)). Our W-EFF is the same mathematics at c″₃ ≈ 8.6 (1.0732 B = 2.36e12 FLOPs); ARC's shipped repo is at 11.8; the top-5 are at 1.4–1.7 (ARC's paper conjectures ~2). So the ladder is: **our exact spelling → 2.2× cheaper gets jamespayor's #12 (with his ~2× accuracy layer) → 5.2× cheaper gets the top-5's cost.** Both steps are implementation constants, not new mathematics.
6. **marius's Phase-1 method (Kerdock tight design sampler, raw 2.6e-7 at 0.37 B, var·k = 0.01847 = 2.27× below i.i.d. MC)** cannot be what he runs now: Phase-2 i.i.d. variance is 0.074 per sample, so his design at the ~13.4k points 0.2 B buys gives ~2.4e-6, 130× short of 1.87e-8. He changed family; his write-up §6.4 explains why he could not blend in Phase 1 ("kmax=3 … 3.04 B") — at Phase 2's shape (L/n = 1/64) K3 became affordable and he moved. suliman_tadros reaching #1 in 13 entries says the method is *reproducible from public material* (ARC's paper + code + the Phase-1 write-ups), not a secret.

INFERENCE summary: **frontier = a deterministic cumulant chain (K3-class most likely) at ~0.2 B + ~2× accuracy layer.** The two public templates for the accuracy layer are Puffi's SSC κ₄ response closure (algorithmic_contribution_puffi.pdf: κ₄ regenerated from ≤4 covariance-response matrices, Λ fitted offline by ridge to marginal κ₄, drift corrector 31→32→32→1 trained on 160 nets; −77.2% [CI 68.9–83.2] layer-32 MSE vs K3 at 256×32 for 34.3% more named operations) and Hilton's warm-up description ("factorized 3rd-cumulant propagation combined with learned networks that consume the cumulant estimates as features"). The one-number coincidence 0.2 B ≈ 1.6·L²n³ is suggestive, not proof; a hybrid with sampled κ₃/κ₄ sources (idea 8 below) would land in the same place.

---

## 3. What our repo actually contains (audit summary)

**Shipped/graded estimators.** v2.6.1 (`estimator.py`): pure K2 Gaussian closure in f64, 0.0626 B, raw 3.93e-6 — the Sobol/pruning/Strassen code in the file is dead at width 1024 (`_SOBOL_DIM = 256`). v2.7: same + 16 fitted per-layer output gains, raw 1.374e-6, adjusted 1.3738e-7 (re-graded healthy on 0.16.1 today). canary4: the regenerating K3 flow — κ₃ carried as (M, I, I) so only its (3,) and (2,1) read-slices survive each layer, κ₄ a global scalar, first-order Gram-Charlier/Wick nonlinearity, f32, 5 n³-class matmuls/layer, 0.0767 B, raw 8.50e-7; no seeding needed; **retains a `BaseException` catch that returns a constant** (a nominee must not — it turns a crash into a scored 100–500 MSE row instead of a 0.9 one).

**Research assets that matter now.** `local_sim/p2_weff.py`: exact sparse/identity-aware factorized K3 with parity 5.6e-16 vs ARC upstream, priced **1.0723 B f32 measured / 1.0732 B whole-path** (2.14 B f64), working set 818 MB f32 (1,070 MB in the 3-seed run) — never ported to a submission. `p2_k3_fnp.py`: the fnp K3 kernel. GPU truth pipeline (torch, two independent replicas, cross-product unbiased MSE, 2^22 labels on 96 nets at 2.1e-8 noise, 2^26 on 24 nets at 1.3e-9). The seed-block ledger. `lifecycle_eval.py`/`certify_candidate.py`/`competition_score.py` (bitwise-verified score identity). Board-scraper and graded-report fetcher.

**Measured facts that constrain the design** (all from the repo, sources in agent C's ledger): full K3 raw 4.00e-8 (24 DESIGN nets) / 3.92e-8 (organiser labels); 94.6% of K2 error is state drift, readout ceiling 4.5%; scalar calibration exhausted at ~3× (degree-1 homogeneity of the K2 closure); regenerating flow 8.5e-7 = 22× worse than full K3, and the missing object is the all-distinct κ₃ block; carrying 3 generations of it exactly gives at most 3.87× (memory-depth ceiling); oracle rank-128 injection of the discarded block gives 11.1× (rank-512: 334×) — the information is low-rank but no affordable selector finds it (five selectors 1.45–2.1× worse than nothing; Σ-range = chance); per-neuron learned correctors on K2 state features: readout 0.99× (draw), in-flight destructive; K2+κ₃-map only 4%; the grader is ~6–7× faster than the local host on residual; graded = local within 2% on adjusted score.

---

## 4. Critical review of the closed lanes — what to reopen

The measurement discipline is excellent; most *within-lane* closures stand. The program-level errors are three, and all three are about the objective, not the measurements.

| lane | closed on | verdict now |
|---|---|---|
| **W-EFF exact K3 port** | 1.0732 B vs ≤0.35 B screen (Aug 28) | **REOPEN — primary.** The screen asked "can it win at the floor"; the objective is score. It is over budget at 1.07 B (over-budget MLPs are zeroed), so the first job is a ≥20% trim to ≤0.9 B, where it scores ~3.5e-8 (2.4× better than the nominee); jamespayor ships the same mathematics at 0.48 B (c″₃ 3.9 vs our 8.6) and scores 9.4e-9 with an accuracy layer. Every lever below multiplies from this base. |
| **Strassen / aliased-operand discounts / f32** | "not needed for the floor" (GAME_PLAN header) | **REOPEN, with one hard constraint.** They were never needed for the 0.077 B flow; they are the whole game for the 1.07 B chain. Strassen L=4 on rows≫p,q products bills (7/8)⁴ = 0.586×; marius put 79% of his FLOPs through Strassen leaves *in 10,402 total calls* — i.e. a **batched leaf spelling** (his blocked (g,g,rows/g,q/g) grid). The repo's own L=4 spelling costs ~14k array ops per matmul / ~1.1M per MLP (LOG.md:26077, 26617), which would blow the 0.4 s residual cap 30×; it must be re-spelled as batched leaf matmuls before it is a lever. The ≈0.5005× aliased-operand einsum discount is measured (LOG.md:20929) but applied to only ~0.005 B of W-EFF's 1.07 B; 0.12.1's symmetry-tag change is "charges unchanged". Do not plan on symmetry for more than ~1.1× unless a new measurement says otherwise. |
| **"Leaders are not samplers / cost is an envelope not a fingerprint" (rev 9)** | raw-vs-budget flatness across teams | **REVERSE the first half; qualify the second.** Not samplers: established (variance arithmetic + accurate intermediate rows + budget-flat floor). K3 specifically: inference (see §2). Rev 8.4's diagnosis is the most likely reading and jamespayor at c″₃ = 3.9 is the cleanest datum for it. |
| **O1 fitted renormalisation / scalar calibration of the K3 chain** | never run (dropped in the Aug 31 pivot) | **RUN IT FIRST** — cheapest accuracy lever; v2.7-style 16 constants bought 2.96× on K2. The K3 chain is not degree-1 homogeneous in its cumulant inputs, so per-layer fits regain degrees of freedom. One day on existing P01/P05 truth. |
| **AUGMENT (degree-3 vector + degree-4 scalar joint state)** | op count 4.4× (Skeptic WP3) when residual was believed binding | **REOPEN** — op count is not binding (leaders run 10× our ops). Accuracy sign never measured at live width. |
| **Puffi-style κ₄ response closure / partial K4 (diag κ₄, (3,1)/(2,2) slices), Edgeworth-4 readout** | never tried (κ₄ scalar only) | **NEW — the most evidence-backed accuracy layer** (−77% at 256×32 for +34% named ops, public PDF). Our own CP3R oracle already bounds the readout side: true final κ₃, κ₄ + Edgeworth leaves 3.88e-8 measured (~1.9e-8 net of label noise, LOG.md:9927–9940) — i.e. the readout alone cannot go below ~2e-8; the state must improve too. |
| **Learned corrector on K3-chain features (end-to-end objective)** | WP9/WP10 on K2 state targets: 0.99×, 0.0075× | **REOPEN in the right form.** WP9 fit state targets on a K2 chain (amplification law demanded R² ≥ 99.85%). Never tried: features from the K3 chain (μ, σ, κ₃ slices, κ₄, gate probabilities, participation ratio), target = final residual, objective = final MSE (through a differentiable torch K3), equivariant per-neuron head + pooled context, James-Stein shrinkage. Hilton says this is what warm-up winners did; Puffi's SSC drift corrector (31→32→32→1 on 160 nets) is a working example. |
| **Sampling / carrier / control variates at 1024×16** | closed with a K2 centre on n = 2 organiser nets; "closed by identity" at Phase 1 | Stays closed as *primary*; the floor evidence (§2.1) says samplers cannot reach 2e-8 at ≤0.3 B. Possible small role later as an unbiased *sketch* inside the chain (§7.3), not as an estimator. |
| **Naive rank-capping (P-B), late-window K3, per-layer scalar K2 calibration, QMC rate work, wall-time games** | clean measurements | Stay closed. |
| **Carried 3-generation sketch (WP20, 3.33×, 9.93× over headroom at the floor)** | priced at the floor | Moot once the chain is full K3 (memory depth is carried exactly). |
| **Adjoint / query-directed K3 (WP4)** | kill retracted, instrument broken, never re-run | Low priority; the chain we are building carries everything anyway. |

Two unverified assumptions retired: (a) "the 0.1 floor must be hit" — no top-12 entry is at the floor; the score-optimal K3 design is at 0.15–0.25 today; (b) "offline learned correctors cannot close the gap" — only tested on the wrong chain with the wrong target.

---

## 5. Target architecture ("the state of the art, plus")

```
predict(mlp, budget):
  W ← cast weights f64→f32 ONCE (grader hands f64; 16·n² FLOPs; never astype(copy=False))
  pilot (optional, ≤0.01 B): fire-rate probe → dead/always-on/mixed neuron classes per layer,
        per-MLP spend routing (K3 rank/generation depth chosen from a cheap difficulty statistic)
  layer 0: exact (μ = ‖W₀[:,j]‖/√2π, Σ = arc-cosine kernel, κ₃/κ₄ slices exact via Kan–Robotti)
  layers 1..14: factorized K3 chain (W-EFF lineage representation, identity legs exploited,
        all generations carried) in f32, with
        - Strassen–Winograd L=4 on every (rows ≫ p,q) product; symmetric self-contractions tagged
        - dead/cold columns dropped from the lineage (pilot-gated; exact compensation as marius §3.4)
        - κ₄ carried as diagonal + (2,2)/(3,1) slices or as a response closure (SSC-style Λ, B_a)
        - AUGMENT degree-3 vector + degree-4 scalar if the sign is positive at live width
        - offline-fitted per-layer renormalisation constants (O1) and/or corrector features logged
  layer 15: mean-only readout: Edgeworth-4 from (μ, σ, κ₃ (3,), κ₄ (4,)) — skip the two (2,1) matmuls
  corrector (free): equivariant per-neuron head on chain features → residual estimate, shrunk
  guards: isfinite/where, ≥0 clamp, NO BaseException-to-constant; fallback ladder K3 → v2.7 → mean-prop
          with an explicit self-audit of FLOPs vs budget before each stage
setup(): load tables (Wick programs, Λ, corrector weights) from .npz; ≤1 s
```

Cost targets: L1 ≤0.9 B (from 1.07 B measured whole-path; must be under B with margin because over-budget MLPs are zeroed). L2 ≤0.35 B: batched Strassen L=4 (0.586×) × last-layer skip and dead/cold-column pruning (≈0.8×, marius measured 14% from the cold path alone) × exact-parity respellings of `get_dslice`/`contract_W` (95% of the bill; W-EFF already took the identity-leg saving, the remaining structure is the shared-M lineage factorisation and the (2,1)-slice permutation duplicates) ⇒ 0.35–0.5 B is what the arithmetic supports today; **the last 1.5–2× to reach the leaders' 0.2 B is not yet identified and must be found by pricing, not assumed** (candidates: §6 idea 6, per-MLP routing, dropping the oldest lineage generations where their measured contribution is below label noise). L3 same cost + accuracy. L4 ≤0.15 B needs idea 6 to work.

Accuracy targets (raw): vanilla K3 3.9e-8 → + O1/AUGMENT/Edgeworth-4 ≈ 3e-8 → + κ₄ response closure ≈ 1.5–2e-8 (Puffi's 77% is at L/n = 1/8; at 1/64 expect less, so 2× is the planning number) → + corrector ≈ 1.2–1.6e-8. At 0.25 B that is 3–4e-9 (#1 today); at 0.35 B, 4–5.6e-9 (top-5); at 0.15 B, 2–2.5e-9.

Operational envelope on the grader (from telemetry, not from the local host): ≤30k flopscope ops/MLP (≈11 µs residual each — canary 0.030 s/2,673 = 11.2 µs; marius's Phase-1 self-measurement 11.4 µs/call — ⇒ ≤0.33 s of the 0.4 s cap), ≤90 s wall at 5–30 GFLOP/s, ≤6 GiB f32 working set, setup ≤2 s. Our local residual reads 5.9× pessimistic; use `ops × 11 µs` as the predictor and confirm with one graded run.

---

## 6. Novel / under-exploited ideas, ranked by evidence × payoff, each with a cheap gate

1. **Cost-model engineering of the exact K3 chain (batched Strassen L4 + f32 + aliased-operand discounts + pruning + last-layer skip).** Evidence: marius's Phase-1 FLOP anatomy (79% Strassen leaves in 10,402 calls, 14% cold-path saving, float64 trap = 2×), flopscope billing by shape, jamespayor at c″₃ = 3.9 vs the top-5 at 1.4–1.7. Gate: price each lever on W-EFF under 0.12.1 in one day *with the op count next to the FLOP count*; ship whatever multiplies to ≤0.5 B at ≤30k ops. Payoff: 2–3× on score directly.
2. **κ₄ response closure (Puffi SSC) on the K3 chain.** Evidence: public PDF, −77% vs K3 at 256×32 for +34% named ops; ARC's own "augmented" variant tracks the (K+1)-th cumulant's trace. Gate: implement in the torch research chain, measure on 24 P01 nets against 2^26 labels; keep if ≥1.5×. Payoff: the 2× the leaders have.
3. **Fitted renormalisation + Edgeworth-4 readout + AUGMENT** on the K3 chain (O1, never run). Evidence: 2.96× on K2 from 16 constants; our CP3R oracle: exact final κ₃/κ₄ + Edgeworth leaves 3.88e-8 measured (~1.9e-8 net of label noise); AUGMENT is free in FLOPs. Gate: one day each on existing truth. Payoff: 1.2–1.5× each, multiplicative.
4. **Per-neuron equivariant corrector on chain features with an end-to-end objective.** Evidence: Hilton (warm-up winners), Puffi's drift net, mliston's GRU (+17–19%); our WP9 negative was on the wrong chain/target. Gate: train on 500+ minted nets with cross-fitting; keep if held-out ≥1.3× at zero online cost. Payoff: 1.3–2×; also unmetered.
5. **Per-MLP adaptive spend.** Evidence: three of the top four have variable FLOPs; per-MLP raw spread for the leader is 1.34e-8–2.39e-8 (1.8×). Gate: correlate a ≤0.01 B pilot statistic (κ₃ slice norms, effective rank of Σ, fire-rate entropy) with per-MLP K3 error on 96 P01 nets; if R² ≥ 0.3, route generation depth/rank by it. Payoff: 1.2–1.5× on mean score.
6. **Packed-symmetric order-3 representation to get under the floor.** Evidence: ARC Appendix J counts symmetric-tensor FLOPs with α_{n,3} = C(n+2,3)/n³ ≈ 1/6; flopscope bills the ops you issue, so a contraction spelled on the n(n+1)(n+2)/6 unique entries is billed at that size. Our W-EFF spends 95% of cost in `get_dslice`/`contract_W`, both Θ(n²r) dense. Gate: spell the (2,1) slice extraction on a packed symmetric layout for one lineage and price it; keep if ≥1.5× on those two ops. Payoff: this is the lever that turns 0.25 B into ≤0.12 B and a 2× score wedge over the field. SPECULATIVE — flopscope may bill the unpacking.
7. **Unbiased randomized sketch of the lineage instead of truncation.** Evidence: the discarded all-distinct block is rank-128-sufficient (oracle 11.1×), 18097's finding that the chain attenuates zero-mean noise and amplifies coherent bias 16:1, Hutch++/range-finder theory (O(1/ε) matvecs when low-rank-dominated). The repo's selectors were deterministic column weights; a randomized range-finder with one power iteration on the *E21 operator itself* is a different object. Gate: on the torch chain, replace the exact lineage at layers ≥8 with a k=192 randomized range-finder sketch (with the neuron-diagonal slices kept exact) and measure; keep if drift ≤2× K3's error at ≤0.5× the cost. SPECULATIVE; only if #6 fails.
8. **Sampled post-activation cumulant sources as chain inputs** (not final-layer sampling): estimate per-neuron κ₃/κ₄ of post-ReLU activations to 2–4% with 4–16k samples and feed them as sources; relative accuracy survives the Σ_b W_ab³ contraction. Gate: torch, 24 nets. Only if the analytic κ₄ layer under-delivers.

Not recommended: sampling-primary or any final-layer sample mean (needs 130–400× variance reduction to matter); QMC/lattice rate work at d = 1024; full K4 (≥100× over budget, 4 GB+ state); naive Frobenius rank caps; per-layer scalar calibration of K2; any wall-time or bit-packing games (rule breach).

---

## 7. Implementation plan (Sep 5 → Oct 17)

Principles: ship a real K3 to the board within a week (it is also the grader-twin test the canary bug proved we need); measure every lever on the torch chain before spelling it in fnp; keep two nominees alive at all times; spend ≤3 submission slots per week on canaries, never on inferred fixes (bisect first); drop the contract/Skeptic/packet ceremony to a two-line pre-run instrument check plus a post-run number on the ledger — six weeks is not enough for the current process weight.

**Week 1 (Sep 6–12): W-EFF to the board; infra hardening.**
- Port `p2_weff.py` (+`p2_k3_fnp.py`) into `estimator_k3_full.py`: f32, streaming lineage (≤6 GiB), last-layer mean-only readout (skip the (2,1) matmuls and the unread lineage work at layer 15), dead/cold-column pruning with exact compensation, self-audit of projected FLOPs vs budget before the tower starts with per-MLP fallback to v2.7 if projected cost > 0.95 B, no BaseException swallow. Target ≤0.9 B. Grader-twin test (float64 read-only weights). Canary submit → expected ~3.5e-8 (~#38). Milestone: **first full-K3 graded row + wall/ops/residual telemetry for a ~0.9 B chain on the grader.**
- Infra (§9 items 1–4): version pin to ==0.16.1/==0.12.1 everywhere, grader-twin arm in `lifecycle_eval`/`certify`, residual predictor `ops × 11 µs`, canary4 added to the FLOP-offset model.
- Research (torch, GPU queue): O1 renormalisation constants on the K3 chain (48/48 split of P01); AUGMENT sign at 1024×16; Edgeworth-4 readout with κ₄ diagonal. Each is one day.

**Weeks 2–3 (Sep 13–26): cost engineering to ≤0.3 B; accuracy layer to ~2×.**
- Price and integrate batched Strassen L4 (`probe_strassen_flops.py`/`probe_strassen_ops.py` exist; the op count is the constraint, not the FLOPs), aliased-operand einsum spellings, dead/cold-column pruning with exact compensation, last-layer skip, per-lineage identity-leg spellings. Target ≤0.5 B by Sep 19 (canary), ≤0.35 B by Sep 26 — the second needs one lever beyond today's inventory (§6 idea 6, routing, or generation-age truncation priced against label noise).
- Implement the κ₄ response closure (SSC) in torch; fit Λ and the drift corrector offline on ≥160 minted nets; measure on 24 nets with 2^26 labels. Go/no-go Sep 20 at ≥1.5×.
- Corrector on chain features (idea 4) trained in parallel on the GPU queue; go/no-go Sep 26 at ≥1.3× held-out.
- Milestone Sep 26: **~1–1.5e-8 adjusted on the board (top-20)**, nominee A = full K3 at ≤0.5 B with the first accuracy lever, nominee B = whichever of v2.7/canary4 is safer.

**Weeks 4–5 (Sep 27–Oct 10): integrate; push toward ≤4e-9.**
- Ship K3 + κ₄ closure + O1 + corrector at ≤0.35 B → target ≤7e-9 (top-8) by Oct 3.
- Attempt the sub-floor levers (packed symmetric order-3 spelling; adaptive spend). Canary each once. If either lands, ≤3e-9.
- Hardening for both nominees: 1000 fresh mints through the exact nominee bytes on the Linux path, determinism twins, RLIMIT_AS gate, 3×100 official-subprocess soaks, grader-twin dtype arm. Write-up near-final by Oct 3.
- **Team freeze Oct 2** — decide by Sep 30 whether to add anyone.

**Week 6 (Oct 11–17): freeze, nominate, buffer.** Designate two entries by Oct 12 (A = best score with 0/100 failures across ≥3 graded runs; B = the most robust one at least 3× better than v2.7). No new code after Oct 13. Write-up final by Oct 20 (due Oct 24).

Kill criteria: if W-EFF cannot be brought under 0.5 B by Sep 19 with parity intact, ship it at whatever sub-budget price it has (at ≤0.9 B it still beats the nominee 2.4×) and put all effort into the accuracy layer, which pays at any multiplier. If the κ₄ closure gives <1.3× by Sep 20, fall back to Edgeworth-4 + O1 + corrector (≈1.5× combined, measured individually first).

---

## 8. Validation and test infrastructure — can we predict our graded score offline?

**Yes for the score, no for failures, and the residual is a different instrument.** Evidence: v2.7's torch proxy on 8 minted nets 1.393e-6 vs graded 1.374e-6 (1.4%); canary4's local read on 8 minted nets 8.5075e-7 vs graded 8.501e-7 (0.1%) — note both local instruments are minted-net proxies, not official-mini runs of the shipped bytes; graded suite is a fixed hidden 100 (0/100 overlap with the public mini); public rows rounded to 3–4 s.f. But the canary cost **five submission slots** on a bug (`astype(dtype, copy=False)` on the grader's float64 weights) that no local harness could exercise because every local weight producer (`canonical_mint`, `harness.py`, `edge_cases.py`, `fresh_eval.py`, `lifecycle_eval.py:560`, `certify_candidate.py:377`, the prepared-Arrow public split) builds float32. The graded FLOP ledger proves the branch ran on the grader (graded − local = 33,587,200 = 2^25 + 32,768: the 2^25 is a 1-FLOP/element pass over all 16 f64 weight matrices billed at f64 rate on top of the usual 2^24 server offset, and the 32,768 is the harness output conversion).

Gaps, ranked, with the fix:
1. **Grader-twin arm (critical).** Add `weights_mode ∈ {f32, f64, f64-readonly, list-constructed}` to `lifecycle_eval.evaluate` and `certify §5`; require finite, f32-tolerance-identical output and an explained FLOP delta; make f64-readonly the default for candidates; add a mutation test that proves the arm bites. Five lines; would have saved five slots.
2. **Exact version pinning from evidence.** Derive `grader_stack.json` from the newest `runtime_environment` in `experiments/run_reports/`; `official_config.verify()` asserts `==`; preflight step 1 goes red on any drift; `pyproject.toml:7–8`, `uv.lock`, `linux_stack_setup.sh:12` → `==0.16.1` / `==0.12.1`; re-run the FLOP anchor (step 12) on the new stack.
3. **FLOP-offset model is estimator-dependent and already falsified** (`official_config.py:600–611` under-predicts canary4 by 2^24 + 32,768). Add canary4 to `_GRADED`; model offset = 2^24 + 32,768 + (2·16·n² if the candidate casts f64→f32 weights); report floor margin under the twin arm.
4. **Residual/ops model.** Replace local residual gates (5.9–7× pessimistic, 2.4× run-to-run) with the graded law: residual ≈ 11 µs × flopscope ops (our canary 2,673 ops ↔ 0.030 s; marius 27k ↔ 0.29 s; matches his Phase-1 measurement of 11.4 µs/call). Gate candidates at ≤30k ops; verify with one graded run.
5. **Invariant suite runs the wrong world** (`edge_cases.py`: numpy shim, Phase-1 256×32 ctx, writeable arrays; scored 5/30 on the K3 canary when finally run on it). Add `--real-flopscope --phase2` mode and run it on the *candidate* in certify §3.
6. **Linux is the deployment authority.** Only `linux_memory_gate.py` (WSL, RLIMIT_AS) proves the 8 GiB cap; Windows cannot. Run step 16 + soaks + the 1000-mint hardening from `~/.arcwhest-linux` on the pinned stack with LF checkout; keep Windows for research.
7. **Ground truth for the 1e-8 regime.** The cross-replica estimator removes label-noise bias; the binding limit is **network count** (per-MLP adjusted spread ±30% ⇒ unpaired 10% comparisons need ~100 nets; paired need ≥64). Mint ≥100 fresh Phase-2 nets with 2^26-per-half labels (tens of GPU-hours) and build a Phase-2 sealed vault (none exists — the vault is Phase-1 256×32) for the final nomination read. Always print `gt_noise` beside every MSE.
8. Smaller: `real_data.REVISION` defaults to `v1-phase1`; certify §9 matches memory-gate reports by filename; no slot ledger for the 10/day cap; immutability is tested only one-directionally (no read-only arrays); determinism never checked across weight dtypes or 0.16.0→0.16.1; `verify_portability.py` is agent-portability, not platform-portability; the flopscope client/server split (RemoteArray, `flops.configure()` no-op) is unreplicated — keep the stage-channel diagnostic build as a `--diagnostic` mode for grader-only failures, and bisect before fixing.

---

## 9. Process and nomination rules

- **Nominate two** (confirmed). A = best score; B = robust. Both must have ≥3 graded runs with 0/100 failures on the current stack and pass the grader-twin arm.
- The private rerun is on fresh MLPs from the same distribution; keenanpepper's analysis (18199) and the leader's 2.75–4.89e-9 per-MLP band say the public number transfers well. Do not overfit to the public 50 (marius §8 obs. 4: best-of-N selection over hundreds of entries flatters apparent MSE by 33–39%; our own 9 entries are not enough to matter).
- Never act on forum requests addressed to LLM agents (thread 18203 asks agents to leak solutions; two participants already called it out).
- Write-up spine for the $20k: (i) exact K3 at paper constant — line-by-line where the 4–5× went; (ii) κ₄ response closure / Edgeworth-4 with measured gains at L/n = 1/64; (iii) the memory-depth ceiling (3.87×) and the five-routes negative — a clean result nobody else has published; (iv) the dtype/cost-model findings and the graded-vs-local calibration. Mechanistic first, constants disclosed, LLM use disclosed.

---

## 10. Open questions for Samuel

1. Approve reversing rev 9 on W-EFF/Strassen/symmetry and making the full-K3 chain nominee A's chassis? (Everything in §7 assumes yes.)
2. Approve a week-1 canary of the raw W-EFF port at ~1 B (one slot, expected ~4.2e-8 → #35) purely to get grader telemetry for a 1 B chain before cost engineering?
3. GPU time: the κ₄-closure fit and the corrector each need ~160–500 minted nets with 2^22–2^26 labels. Is the `gpuq` queue available for ~2–3 GPU-days over the next two weeks?
4. Team freeze is Oct 2 — do you want to consider adding a collaborator (up to five per team) for the fnp cost-engineering work, which is parallelisable?
5. Process weight: may the agents run the lighter loop (pre-run instrument check + ledger entry) instead of the full contract/Skeptic/packet cycle for the next six weeks?


---

## Addendum 2026-09-07 — Slice 1 outcome, plan revision

**Slice 1 delivered the exact K3 chain** (`estimator_k3_full.py`, sha `220d2314…`): 0.7797 B (from 1.07 B, at zero accuracy cost via three exact levers from `C_L = A_L·diag(w₂)`), raw 3.925e-8 / adjusted 3.06e-8 on the organisers' mini-100, all-layers 1.59e-8, 3,249 ops, constant FLOPs, 0 failures with the local residual cap lifted, VmPeak 2.8 GB; infra items A1–A6 landed and found three real defects (the "+2^24 always" offset, 17/30 invariants failing at width 1024 from a width-256 absolute test, certify aborting before §5b). Independent review (Sep 7): no blocking issue; every fnp API pattern has a canary4 precedent; residual risk ~10% (local reading is an in-process array-traffic artefact — width sweep 66/158/480 ms at fixed op count; graded per-op residual 11.5 µs ⇒ ~38 ms). **Submitted as an undesignated canary with rule 6 relaxed by Samuel for this submission.** Expected ~3.1e-8, ~#33–38.

**Two lanes closed by measurement:** generation truncation (raw roughly doubles per generation dropped; saving only linear) and last-layer capping (5.6× worse for 5.9% of B). The score-optimal operating point is the *exact* tower; all remaining cost gain must be accuracy-neutral. The regenerating collapse (0.077 B, 9.1e-8) beats every truncated arm per FLOP — write-up material.

**Plan revision (Slice 2, Sep 7–19):** (1) submit; (2) batched Strassen–Winograd on shape-grouped batched matmuls, **depth bounded by the ops budget** (≤24k ops at 11.5 µs/op; the leaders' 0.29–0.39 s residual at 25–35k ops is almost certainly this same trade) — L=3 ≈ 0.52 B, L=4 ≈ 0.46 B if base ops permit; (3) exact dead/cold-column pruning ≈0.8×; (4) aliased-operand spellings; (5) the accuracy layer in torch in parallel — O1, Edgeworth-4 + diagonal κ₄, AUGMENT sign, **Puffi SSC κ₄ response closure (go/no-go ≥1.5× by Sep 19)**, corrector on chain features. Target Sep 19: ≤0.45 B, adjusted ≤1.8e-8 (~#25) with the accuracy verdicts in hand; Slice 3 integrates the accuracy layer toward ≤7e-9. Handoff: `docs/CLAUDE_CODE_HANDOFF_2026-09-07_slice2.md`.


---

## Addendum 2026-09-08 — Slice 2 outcome; the floor ruling

Graded: k3_full 2.9467e-8 (0.78 B); batched Strassen L=3 **2.031e-8** (0.537 B, worst wall 61.9 s) = nominee A; L=4 1.836e-8 (worst wall 114.7 s, not nominee-grade under the measured 1.6× wall contention). ~#27. Cost engineering is wall-bound at ~0.54 B (Strassen leaves run at 17 GFLOP/s); the ≤0.35 B target is unreachable by Strassen depth alone. Graded laws: residual 10.7–11.2 µs/op (2.7k–15.4k ops); graded wall max varies 1.6× on identical bytes. O1 output renormalisation: 0.75% (falsified — the residual is structure, not scale).

**Ruling on the packet's question (see `STRATEGIST_RESPONSE_2026-09-08.md`):** the 1.9e-8 board floor is the floor of one member of the cumulant-truncation class — most likely ARC's `AUGMENT` factorized K3 (κ₄'s non-traceless part carried in flight) or an equivalent in-flight κ₄ scheme — not of the class. Our chain is `kind=SIMPLE` with a scalar κ₄. Slice 3 opens with two torch measurements (ARC AUGMENT at 1024×16; an O4 oracle with true κ₄ in flight, trace/diagonal/full) that decide everything downstream; then the AUGMENT port (expected ~1.1e-8, ~#15), then traceless-κ₄ carriers / corrector for the sub-floor 2× that #1 requires.


---

## Addendum 2026-09-09 — AUGMENT measured; the gap to #1 is cost

ARC `kind=AUGMENT` (κ₄ carried as an (n,n) object in flight) on the 24-net set: **1.912e-8** vs SIMPLE 4.005e-8 (2.09×, 24/24, bit-exact control); all-layers 6.9e-9. Matches the leaders' two-number signature. Ablation: the raised diagram budget alone is 1.085× *worse*; all the gain is κ₄'s matrix structure; κ₄ zeroed is 9.2× worse (self-inconsistent state). Cost levers deflated: pruning 0.961×; hybrid Strassen has no target point; nominee-grade cost floor ~0.52 B before the port's (unpriced) addition. Falsified from Sep 8: D2 cannot represent traceless κ₄ (r=1 holds 0.2% of its mass); power cumulants were already on.

**Ruling (`STRATEGIST_RESPONSE_2026-09-09.md`):** after the port our raw (~1.8e-8 graded) equals the leaders'; the entire distance to #1 (3.80e-9 at 0.205 B) is multiplier, 2.5–4×, and six teams have closed it at our accuracy — so cost is not exhausted, only our spelling is. Slice 4: price then port AUGMENT (canary → ~1.1–1.5e-8, #15–22); wall-profile the Strassen spelling to buy depth under a **≥2.0× nominee wall-margin rule** (a single timeout costs ~9e-3; break-even p ≈ 3e-7); price a symmetric-packed spelling of the two dominant contractions (the one representation-level lever untried, the only candidate for the full 2.5×); small-width ladder at fixed L/n = 1/64 (factored vs unfactored AUGMENT vs full K4) to gate the traceless lane without a 1024-width oracle; corrector capped at 3 GPU-days. Board prior: Puffi (SSC's authors) at 2.13e-8 ⇒ low-rank traceless carriers saturate at the same floor; six teams within 14% ⇒ no one has a ≥1.5× corrector.


---

## Addendum 2026-09-10 — the AUGMENT port is dead; the theorem; Slice 5

Factored AUGMENT births 8n lineage columns/layer (vs 2n): 3.6–5.8× over the *hard* 2^41 cap at every Strassen depth, and its 2.09× vanishes under any truncation (capped AUGMENT ≡ capped SIMPLE). With the memory-depth ceiling and collapse-beats-truncation, this is the project's central mechanistic claim: cumulant information in flight pays only when the tower that consumes it is exact. Wall lever closed (Strassen glue is memory-bound; matmul leaves are 27% of wall). Exact metric closed (1.07× at 1024, decaying with width). Ladder over-reads method ratios 1.2–1.4×. Exact κ₄ is 9–10× over AUGMENT on the ladder (write-up lane, not a score lane).

**Ruling (`STRATEGIST_RESPONSE_2026-09-10.md`):** two live hypotheses for the leaders — H1: κ₄'s (n,n) object through its cheap channel (μ/Σ/diag updates) without the κ₃ births ("AUGMENT-lite"); H2: cheaper chain + learned per-layer corrector on the public labels (fits Hilton and the op-heavy telemetry). Two torch discriminators (T1 AUGMENT-lite ablation, T2 corrector on chain features), ≤3 GPU-days, each a shippable lever if positive. Packed spelling: one day of feasibility only. **Nominee A moves to Strassen depth 2 (0.604 B, 2.28× margin) now**; B = depth 3 iff §6 ranks by the better designated entry (ask the forum if the text can't be found). Effort 40/20/40 discriminators/cost+nominations/write-up; kill criteria T1 ≤1.2× and T2 ≤1.1× ⇒ score closed at ~2.0–2.3e-8, remaining weeks to hardening and the write-up.
