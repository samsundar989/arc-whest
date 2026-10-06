# PLAN — ARC White-Box Estimation Challenge, Phase 2 — rev 18 (2026-10-06)

**This is the only plan.** It supersedes rev 17 (Sep 28). Evidence: `claude/deep-review-2026-10-01.md` and the seven `claude/review-2026-10-01-*.md` agent reports (project), summarised in §0. Labels: **[M]** measured / primary source, **[I]** inference.

## Samuel's decisions (Oct 5–6)
1. **Two lanes in parallel.** Session 1 = Lane P (decided-neuron pruning, cost; owns main). Session 2 = Lane V+M (fresh-network validation panels, then the dropped-diagram measurement, accuracy).
2. Stay solo (team freeze passed Oct 2).
3. Build the wall-safe twin of v73 now (insurance B).
4. **Design from the 0.1 floor upward.** Target an estimator that bills ~0.10–0.11 B; add FLOPs above the floor only where accuracy pays for them (marginal rule, §1).
5. **Validate against fresh networks.** Decisions must predict performance on fresh, unseen networks (the prize rerun), not public-board readings or the graded nets. Build a fresh-network lockbox (§2).

## 0. Facts that govern decisions
- **Prizes [M]:** after Oct 17, ≤ 2 designated submissions are re-run on a **fresh suite drawn from private seeds, same distribution** (width 1024, depth 16, He N(0, 2/n) weights, no biases, x ~ N(0, I), budget 2^41); the sponsor picks the suite size and may add networks until the ranking is clear; only the better designate ranks (Rules v14 §5.4, §6, §7). The public board (50 nets) and the 50 sealed graded nets do not decide prizes. Default if nothing is designated: the two best public submissions.
- **Score [M]:** per net, final-layer MSE × max(0.1, F / 2^41); mean over nets.
- **Leaders, Oct 1 [M]:** J2W 1.630e-9 (raw 1.49e-8 × 0.109), marius_binner 1.674e-9 (1.14e-8 × 0.147), Luna 1.844e-9 (1.41e-8 × 0.131), suliman 1.97e-9, mliston 2.07e-9. Their per-network FLOPs vary and co-vary across 13 teams (r 0.8–0.99, uncorrelated with difficulty) → shared weight-dependent pruning [I]. Expected winning bar Oct 17: 1.3–1.5e-9 [I].
- **Us [M]:** A = v73_gb #332893 4.6199e-9 (2.1515e-8 × 0.21473); B = v68_ws #332544 5.0664e-9. Cost census: recent exact κ3 lineages 61.4 %, old tier 27.1 %, covariance 7.7 %, closure 1.1 %, corrector 0.03 %. Error: refit ceilings κ3 53 %, cov 41 %, κ4 28 %; κ3 error mostly all-distinct (25 % at depth).

## 1. Objective and the marginal rule
- Score = raw × max(0.1, c). **Below 0.1 B, FLOPs are free** — so at the floor, spend all of 0.1 B on accuracy. **Above the floor**, raising cost from c to c(1+Δ) pays only if raw falls by more than Δ/(1+Δ) (e.g. +10 % FLOPs needs ≥ 9.1 % lower raw).
- Our measured knob elasticities sit near break-even (age 5→6: −5.5 % raw for +5.2 % FLOPs; rank 256→320: −5.5 % for +7.9 %), so today's chain cannot reach the floor without losing more accuracy than it saves. Reaching ~0.10–0.11 B at our accuracy needs ≈ −50 % FLOPs from **representation** changes; pruning (Lane P) is the first evidenced piece.
- Targets: **T1** (Lane P) ≤ 0.15 B at ≤ +1.5 % raw by Oct 10 (≈ 3.2e-9). **T2** (Lane M) go/no-go on the dropped diagrams by Oct 8; if go, ≥ 10 % raw by Oct 13. **Stretch** ≤ 0.11 B at ≤ 1.9e-8 raw (≈ 2.1e-9).

## 2. Validation protocol (binding from Oct 7)
| panel | nets | use | never used for |
|---|---|---|---|
| TRAIN | organizer `full` 0–899 (+ the 2,048 d8b nets already used) | fitting | selection |
| VAL | `full` 900–999 | checkpoint selection | acceptance |
| DEV | `mini` 0–99 | quick screens | acceptance (retired as the acceptance panel) |
| **FRESH-A** (new) | 100 fresh nets | **acceptance** of canaries: paired vs incumbent, 95 % CI | fitting, checkpoint selection |
| **FRESH-LOCK** (new) | 200 fresh nets (+20 at N = 1e9) | **designation only**, ≤ 3 logged reads | everything else |

- Fresh nets come from **whestbench's own generator** (the grader's code path) with seeds from a reserved range; labels from our own GPU Monte Carlo, N = 1e8 (exact per-net noise floor σ²/N recorded and subtracted for absolute scores; it cancels in paired differences). 20 FRESH-LOCK nets also get N = 1e9 to validate the floor model.
- **Acceptance** = paired improvement on FRESH-A with the 95 % CI excluding 0 **and** projected score below the best graded **and** all hard gates (§5).
- **The public board is a parity check only** (local projection vs graded within 1 %); never a reason to choose between candidates. **Designation** is decided on FRESH-LOCK plus wall/residual margins.
- Expected precision: per-net score CV ≈ 14 %; paired differences resolve ≈ 0.3 pp on 100 nets — enough for every lever in this plan (each worth ≥ 5 %).

## 3. Lanes
- **Lane P** (session 1; owns main, `RESULTS.md`, `experiments/submissions.jsonl`, `docs/DESIGNATION.md`, `docs/CURRENT_STATE.md`, all `estimator_*` files): `docs/HANDOFF_2026-10-06_LANE_P.md`. P0 insurance B → P1 mask test → P2 compact pruned chain → P3 floor push.
- **Lane V+M** (session 2; worktree/branch `lane-m`; no submissions, no estimator edits on main): `docs/HANDOFF_2026-10-06_LANE_VM.md`. V fresh panels first → M dropped-diagram measurement → M4 fallback (neighbor-feature corrector screen).
- GPU: workerq; priority Lane P twin jobs > Lane V labelling > Lane M. RunPod (RTX PRO 5000, volume kept) is approved for Lane V/M when the local GPU is busy for > 2 h.

## 4. Timeline (UTC)
| date | Lane P | Lane V+M | strategist / Samuel |
|---|---|---|---|
| Oct 6 | P0 v74_ws built, gated, submitted | V1–V2 generator parity + MC labeller; first 100 labels | — |
| Oct 7 | P1 mask table + short refit | V3–V4 panels + calibration; M0–M1 | review P1 |
| Oct 8 | P2 build starts (if P1 passes) | M2–M3 decision | review M |
| Oct 9–10 | P2 priced, refit, FRESH-A accepted, canary (A + leaf-64 B) | M prototype (if go) or M4 | **Oct 10 decision point** |
| Oct 10–14 | P3 floor push; integrate M if go | support integration | write-up drafting (strategist) |
| Oct 14–16 | hardening; FRESH-LOCK read (≤ 3) | — | **Samuel designates A/B** |
| Oct 17 23:59 | close | | |
| Oct 24 23:59 | | | write-up due |

## 5. Standing rules
- **Hard gates** (exact source hash): `whest validate`; ops ≤ 26,000; projected residual ≤ 0.33 s; setup < 2 s; WSL 8 GiB PASS; 0 failed nets; wall — new code path ≤ 105 s noisy-implied (A), ≤ 95 s (B); twin↔export parity ≤ 1e-3; package hash = tested source.
- **Materializations:** zero by default. Data-dependent pruning may use ≤ 48 per net only if each is measured and the projected worst residual stays ≤ 0.30 s.
- ≤ 5 canaries per UTC day + 1 A/B slot. **Samuel designates;** agents never contact organizers or post.
- Every substantial run has a written prediction (worker and strategist) before it runs, and a stop rule.
- Shape safety: every per-layer table clamps or falls back for non-suite shapes (whestbench issue #149).
- Git hygiene: review agents use `git --no-optional-locks status`; push to origin daily (nothing pushed since Sep 17 as of Oct 1).

## 6. Not doing
Everything in rev 17 §6 (sampling hybrids; memoryless κ3; radial/harmonic old tier; compact-history B5 and E5 fallbacks; young-only D21; rank/age sweeps on the unpruned chain; tiered ranks; P3 feedback; per-net adaptive λ/age; learned closure; σ corrector; bigger corrector; gate growth; per-source κ4 legs; symmetric-core and q4/8-triangle corrections), plus: choosing between candidates on public-board readings; any fitting or selection on FRESH panels.
