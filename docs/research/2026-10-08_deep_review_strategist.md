# Deep review — 2026-10-08: is there a limit, how the leaders do it, and what we do next

*Strategist session for Samuel, Oct 7 evening (UTC). Inputs: the Oct 7 commits of both workers (main and lane-m), the live AIcrowd board and forum (Chrome, Oct 7 ~18:00 UTC), our fresh-label files, two research agents (`docs/research/2026-10-08_R3_arc_k4_inventory.md`, `docs/research/2026-10-08_R4_public_footprints.md`). [M] measured or primary source, [I] inference.*

## 1. Worker review (validated against the live submission pages) [M]

| build | what | F/B | raw | graded |
|---|---|---:|---:|---:|
| v78_kashws #334545 | wall-safe B twin of v77 | 0.19296 | 2.2023e-8 | 4.2496e-9 (new B) |
| v79_dress #334576 | v77 + symmetric dressed legs + refit | 0.17991 | 2.0580e-8 | 3.7025e-9 (−6.5 %) |
| **v80_hyg #334659** | v79 + layer-0 WᵀW, join via Strassen, extra Strassen level | **0.17602** | 2.0579e-8 | **3.6224e-9** (−2.2 %, board #34) |
| v80_hygws | B twin of v80 (block 40) | 0.18004 | — | gated for the Oct 8 A/B slot (≈ 3.66e-9 projected) |

- **Lane P2** (main): P2.0 done (rev 19 committed, lane-m merged, v77 census: recent tier 58.1 %, old 28.0 %, covariance 9.2 %; no `out=`/`where=`/symmetry tags). P2.1 dressed legs PASS (twin −6.81 % raw after refit; FRESH-A −7.4 %; graded −6.5 %). P2.2 gain-mode κ4 killed (+3.5 / +4.3 / +0.01 % after refit). P2.1b exact dressing correction closed (+0.3 to +1.9 % unrefit in our chain vs −41 % in ARC's). P2.4: always-on D21 rows (+34 % cliff) and alive-block covariance (×13) closed; three Strassen/join items shipped as v80. Calibration: FRESH-A ÷ 1.019 read ~1.2 % optimistic on v79/v80.
- **Lane M2** (lane-m): M5 — AUGMENT is **not separable**: removing any single ingredient makes ARC's chain 3–144× worse than SIMPLE (7–315× worse than AUGMENT); symmetric dressing loses 59 % of AUGMENT's gain; the exact asymmetric form Sym(X ⊗ e ⊗ (X + 2H)) reproduces AUGMENT to the last digit. M6 — the κ3×edge family moves D21 by only 0.2–0.9 % but the off-diagonal κ4 core by 13–25 %; E21b's gain-mode closed form cuts ARC's final MSE **−11.6 % ± 1.4 (16/16)** at ≈ +8 % FLOPs, but only through an off-diagonal κ4 core; the diagonal-only route keeps 12–16 % of it, so the v77 port was closed. M7 — early layers are not the lever (coherent errors 2–8× under budget); v77's variance level drifts −3e-5 per layer, same sign on all nets.
- Quality: careful, honest, both predictions recorded before runs. Watch item: v80's local wall (+8.6 % vs v79, noisy-implied ≈ 101 s vs our 105 s A gate).

## 2. The board (Oct 7 ~18:00 UTC) [M; cost = score ÷ raw, from rounded display values]

| rank | team | score | raw | cost (×B) |
|---|---|---:|---:|---:|
| 1 | J2W | 1.50e-9 | 1.47e-8 | 0.102 |
| 2 | marius_binner | 1.60e-9 | **1.14e-8** | 0.140 |
| 2 | Luna | 1.60e-9 | 1.40e-8 | 0.114 |
| 4 | mliston | 1.70e-9 | 1.46e-8 | 0.116 |
| 5 | mk1981 | 1.80e-9 | 1.72e-8 | 0.105 |
| 5 | oliver_chen | 1.80e-9 | **1.27e-8** (1.37e-8 on Oct 6) | 0.142 |
| 7 | suliman_tadros | 1.90e-9 | 1.57e-8 | 0.121 |
| 8 | ben3 | 2.00e-9 | 1.49e-8 | 0.134 |
| 9 | thegamers / AndreasHad04 | 2.10e-9 | 1.51e-8 | 0.139 |
| 11 | shiv_m / lode_dockx / Masato | 2.20e-9 | 1.52 / 2.21 / 1.46e-8 | 0.145 / 0.100 / 0.151 |
| 17 | kaileh57 | 2.50e-9 | 1.47e-8 | 0.170 |
| **34** | **us (v80)** | **3.62e-9** | **2.06e-8** | **0.176** |

Top-10 ≤ 2.1e-9, top-20 ≤ 2.6e-9, top-30 ≤ 3.4e-9. Eight of the top fifteen (nine with kaileh57 at #17) sit at raw **1.46–1.52e-8**; three are below that level: marius −24 %, oliver_chen −15 %, Luna −7 %.

## 3. Is there a limit? [M + I]
1. **The metric's floor.** Labels are Monte Carlo means over N = 1e9 inputs. From our 400 fresh-net label files, the mean per-neuron variance at layer 16 is **0.0747** (range 0.049–0.116) ⇒ label-noise floor **7.47e-11** raw (resolves BENCHMARK.md's open "6.1e-11 vs 2e-10"; Anatomy 2's cross estimate was 7.71e-11 on FIT nets). Minimum possible score ≈ 7.5e-12. The leaders' raw is 150–200× above it. *Disclosure: this read the label **variance** arrays of FRESH-A/D/LOCK; no estimator was evaluated and the LOCK read count is unchanged.*
2. **The cumulant ladder.** ARC (arXiv:2605.05179) conjectures MSE ≤ c_K (L/n)^K for K tracked cumulant orders, cost ∝ L² n^K (factorized). Our data at 1024×16 (L/n = 1/64):

| method | raw MSE | c_K = MSE·(n/L)^K | cost as written in ARC's code |
|---|---:|---:|---:|
| K=2 Gaussian closure | 4.1e-6 | 0.017 | ≈ 0.02–0.03 B (covariance only) |
| K=3 SIMPLE (ARC a) | 3.93e-8 | 0.010 | ≈ 1.4 B |
| K=3 AUGMENT factored (ARC b) | 1.79e-8 | 0.0047 | ≈ 11 B |
| K=3 AUGMENT unfactored (b × 0.835) | ≈ 1.50e-8 | 0.0039 | > 11 B |
| K=4 | unknown: ≈ 1e-10 by naive extrapolation, realistically 0.6e-10–3e-9 | — | ≈ 2,000 B (SIMPLE), ≈ 17,000 B (AUGMENT) |

   Width slopes from Lane M's Oct 6 data (8 nets per width, n = 256–1024): SIMPLE −2.9, factored AUGMENT −3.3, unfactored AUGMENT −3.2 — the n^−K law with K = 3. K=2 → K=3 was worth ~100–270× here; repeating that factor for K=3 → K=4 gives ~1e-10, but higher-order constants grow and depth compounds (ARC saw K=4 degrade with depth at n = 256), so the strategist's central guess is a 5–20× gain (≈ 1–3e-9); Lane M3 M9 measures it. Each order costs ~n× more. Competitive entries are compressed K=3 chains at 0.10–0.18 B.
3. **No proven lower bound** exists for the error achievable with a fixed FLOP budget (ARC §7.3 lists depth-independent/narrow-width performance as open). The private rerun uses fresh nets from the same distribution; our FRESH panels predict the server within ~1–2 %, so there is no luck factor in the ranking.

## 4. How the leaders do it [I unless marked]
- **Eight of the top fifteen are at the complete-K=3 level** (raw ≈ 1.5e-8 = ARC unfactored AUGMENT) and differ by cost (0.10–0.16). J2W runs it at the 0.10 floor. That is an efficient complete third-order method — optimization, not new mathematics.
- **Three teams are below that level:** marius (−24 %), oliver_chen (−15 %, improving) and Luna (−7 %). Consistent with calibrated corrections for the systematic part of the fourth-order residual, or partial fourth-order terms. No one shows a full fourth-order gain.
- **Public footprints** [M, R4]: no top-8 team has published anything. kaileh57 (rank 17, raw 1.46e-8) has public research notes from August: memoryless one-step κ3/κ4 corrections contracted against next-layer weights (≈ 2 n³ per layer), a κ4 ≈ Sym(M⊗C) matrix core closed under the linear step, and "the fourth-order state is load-bearing" (a K3-only free rollout diverged). AndreasHad04 (rank 9) files flopscope billing issues (#275 `out=` under-billing; #285 `where=` keeps symmetric tags, `exp` bills ~49 % low); the organizers' promised fixed release and re-evaluation have not appeared (v0.12.1 still latest).

## 5. Where we are [M + I]
- **Accuracy:** raw 2.06e-8 is 37 % above the complete-K=3 level. Anatomy 2 (v59, rolled exact substitution) splits our error into κ3 (−51 % if exact: D3 −22 %, D21 −12.5 %), covariance (−42 %), κ4 (−15 %, all Q31; exact D4 alone made it 1.3 % *worse* and exact D22 6.6 % worse, because the chain's κ4 parameters compensate); D21's error is births/closure, not compression. Our κ3/κ4 physics is incomplete (symmetric dressing, memoryless off-diagonal κ4 surrogate), and the learned gains compensate — which is why grafts transfer only partly (dressing −6.5 %; exact correction, gain-mode κ4 and E21b-diagonal hurt or vanish).
- **Cost:** 0.176 B, 1.76× the floor; every compression tried trades accuracy faster than cost (rank 192: −5.75 % F for +21 % raw; fold age 3: −5.4 % F for +31.6 %), and the wall (~101 s) blocks adding FLOPs.
- **Reading:** we are not yet at the "just optimizing" stage — the complete-K=3 physics is missing — but our architecture is near its own ceiling. A major jump needs complete third-order physics at lower cost, or fourth-order content at third-order cost.

## 6. New mathematics worth betting on
1. **Fourth-order births that ride on our factors** [M, R3; verified by numpy re-implementation]. K=3 AUGMENT discards the distinct-index κ4 births entirely (they are exactly traceless at birth): star (one kink hub, three legs; weight w3 = −αφ/σ²), path (two hubs of one lineage joined by C), and κ3×edge. The star uses exactly the transported legs of the κ3 lineage (A = M·X, B = M), so its slices read out with no extra transport: **D4[a] = 4·Σ_j w3_j A_aj³ B_aj — O(n²) per lineage**; D31 2 u, D22 1 u per lineage; the path's D4 costs one product (V = (A∘J)C̃). K=3 AUGMENT and K=4 SIMPLE use identical Wick terms for the mean and covariance, so the whole K=4 increment there is the traceless part of D4/D31/D22 plus κ3 differences. Exact factored K=4 is ≈ 2,000 B at n = 1024 (κ4 factors alone 0.27 TB), so only such shared-factor pieces are affordable. Lane M3 M9/M10 measure the headroom and the share these pieces carry.
2. **κ4 as memory** [hypothesis]. In ARC's chain the κ4 core's memory and the κ3↔κ4 loop are indispensable (memoryless core 17×, no K211 252×, no 211→111 314× worse). If the core can stand in for old κ3 lineages, it explains how complete-K3 teams afford 0.10–0.15 B ("old content at ≈ zero cost", our open question since Sep 22) and it would remove much of our 28 % old tier. Lane M3 M8 tests it directly (drop κ3 blocks older than A in ARC b+E21b, with and without the core).
3. Write-up framing: the cumulant-order map of the board (§3–4), the gain-mode law, dressed legs and the exact asymmetric identity, AUGMENT's non-separability, the κ3×edge measurement, and whichever of (1)/(2) holds.

## 7. Plan (PLAN rev 20) and odds
Samuel chose a **gated big bet** (the loop pays only through an old-tier saving, so M8 and Lane P3's arms iv/v are the crux). Lane M3: M8 (κ4 memory), M9 (K=4 headroom vs width, extrapolated to 1024), M10 (which fourth-order pieces carry it; the shared-factor arm). Lane P3: v80 B twin on the Oct 8 slot, bank (Strassen leaf, variance-level gain, star-D4 arm), build the κ4-core loop (+E21b) in the twin. **Oct 10 GO/no-GO on a v90 rebuild.** Write-up drafting starts after Oct 10 (Samuel's choice). Odds [I]: podium ≤ 3 %, top-10 ≈ 10 %, top-20 ≈ 35 %.

## 8. Risks
- The loop's cost — ≈ 9–10 raw n³ products per layer as specified (core sandwich 2, r = 1 projection ≥ 1, K211 3–4, radial 211→111 transport ≥ 2, E21b 1), ≈ +40–45 % F after Strassen, ≈ +30 % with truncated K211 and a folded radial term — needs −23 to −31 % raw just to break even, so it pays only if it lets the old tier shrink (M8; Lane P3 arms iv/v).
- Wall time is near our gate; any v90 must reduce F, not add it.
- flopscope re-evaluation (#275/#285) is still pending; we use none of the flagged patterns.
- Long-lived-worker memory growth (#18238, unanswered): Lane P3 measures it against the 8 GB limit.
- Refit instability when physics changes (lr 1e-4); partial grafts can hurt (non-separability).
