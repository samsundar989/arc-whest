# HANDOFF — Lane V+M (session 2), 2026-10-06: fresh-network validation panels, then the dropped-diagram measurement

Repo `C:\Users\samsu\Documents\arc-whest`. Work in a **git worktree on branch `lane-m`** (`git worktree add ../arc-whest-lane-m -b lane-m`). Never edit `estimator_*` files on main, never submit, never touch `RESULTS.md`/ledger/DESIGNATION (Lane P owns them). Your outputs: `tools/fresh/`, `experiments/2026-10-06_laneV/`, `experiments/2026-10-06_laneM/`, `docs/research/2026-10-0X_laneV_report.md`, `docs/research/2026-10-0X_laneM_report.md`; hand the panel/evaluator changes to Lane P for merge.

Read: `docs/PLAN_rev18_2026-10-06.md` (§2 validation protocol), project docs `claude/deep-review-2026-10-01.md`, `claude/review-2026-10-01-C-literature.md` (§1.3: ARC term sets and the factored drops), `claude/review-2026-10-01-E-frontier-mechanisms.md` (§1, §6.2), `claude/review-2026-10-01-A2-cost-eval-infra.md` (§3–4: scoring, panels, ev.py).

GPU via workerq at lower priority than Lane P's twin jobs. If the local GPU is busy for > 2 h, use the RunPod box (RTX PRO 5000, volume kept; approved).

**Why V first.** Prizes are decided on a fresh suite from private seeds. Today every local panel has been used for fitting or selection, and 39 submissions have read the same 50 public nets. Samuel wants acceptance and designation decided on fresh networks that nothing has touched.

---

## V — Fresh-network panels (Oct 6–7)
- **V1 Generator parity.** `tools/fresh/gen.py`: build MLPs with **whestbench's own generator** (the function the grader and the HF bake use; start from `whestbench/generation.py`), float32 weights, seeds from a reserved range (e.g. 9_100_000 + i) recorded in `experiments/2026-10-06_laneV/seeds.json`. Verify on 5 organizer `mini` nets: regenerate from their published `mlp_seed` and check bit-identical weights. If the published seeds do not reproduce, prove distributional parity instead (per-layer weight mean/variance; our chain's per-net α and σ statistics vs `mini`).
- **V2 Labeller.** `tools/fresh/mc.py`: torch GPU Monte Carlo of **all-layer means** with N = 1e8 inputs per net — batched; **TF32 off**; float32 matmuls with float64 accumulation of sums and sums of squares; deterministic per-net seeds. Store means and per-neuron variances, so each net's exact noise floor (σ²/N, mean over the scored layer) is known. Target ≈ 1 min/net on the 5090. Validate on 10 organizer `mini` nets: MSE(ours vs organizer label) ≈ floor(1e8) + floor(1e9) ≈ 8.5e-10.
- **V3 Panels.** FRESH-A = 100 nets; FRESH-LOCK = 200 nets, plus 20 of them at N = 1e9 to validate the floor correction. Register in `tools/panels.py` (assert no overlap with TRAIN/VAL/DEV by seed and weight hash) and in `tools/ev.py`:
  - `--panel fresh_a`;
  - `--panel fresh_lock` appends every read to `experiments/fresh_lock_reads.jsonl` and refuses after 3 reads unless `--i-am-designating`;
  - absolute scores reported with the floor subtracted; paired differences reported raw.
- **V4 Calibration.** Evaluate v73_gb and v68_ws on FRESH-A: score, raw, paired difference, and the ratio to the DEV projection and to the graded public score. If the keenanpepper d8b corpus holds nets beyond the 2,048 already used for training, take 100 unused ones as a third independent panel (FRESH-D) and repeat.
- **Predictions (strategist):** v73 on FRESH-A within ±2 % of 4.62e-9; per-net CV ≈ 14 %; the v73 − v68 paired difference reproduces its graded value within 0.3 pp. Record yours.
- **Deliver:** panels + evaluator changes to Lane P for merge; `docs/research/2026-10-0X_laneV_report.md` with the calibration table. From then on, Lane P accepts candidates on FRESH-A.

## M — Dropped same-order diagrams (Oct 7–8)
Background: ARC's reference code warns that its *factored* AUGMENT drops non-hypertree diagrams (the covariance triangle ŵ2ŵ2ŵ2·C·C·C, and κ3 × covariance edge ŵ1_a ŵ2_b ŵ2_c κ3_abc C_bc with their (2,1,1) analogues) that "have the same Θ(n^-3) squared size as the extra terms that are kept" (`factor_k3.py` ~L291, `kprop_harmonic.py` `factored_keeps_term` ~L86). We tested one narrow sub-family (the q4/8 low-rank triangle: < 0.01 %). Our κ3 error is mostly all-distinct (25 % relative at depth, AD oracle −33 % MSE).
- **M0 Environment.** ARC repo at `external/mlp_cumulant_propagation`, `.venv-kprop`. Confirm entry points: `kprop_harmonic` kinds SIMPLE / AUGMENT with `factor=True/False`, `factored_keeps_term`, `scripts/kprop_by_width.py`, and how `tests/test_factor_k3.py` restricts the unfactored reference to the factored-kept terms (that filter is your tool for per-family toggles). Run on torch GPU if supported.
- **M1 Nets.** Widths n ∈ {256, 512} (also 1024 if the unfactored path fits in time and memory), depth 16, He init; 8 nets per width from V1's generator with a width override; MC labels N = 1e8 from V2.
- **M2 Variants per net:**
  - (a) factored SIMPLE, k_max = 3;
  - (b) factored AUGMENT, k_max = 3;
  - (c) unfactored AUGMENT, k_max = 3 (dense κ3);
  - (d) unfactored AUGMENT restricted to the factored-kept terms (must reproduce (b)), then with each dropped family added back alone.

  Report final-layer MSE (floor-corrected), all-layer MSE, per-layer D3/D21 relative error against MC where obtainable, and runtime.
- **Key quantities:** G_drop = (MSE_b − MSE_c)/MSE_b and each family's share of it; G_aug = (MSE_a − MSE_b)/MSE_a; the trend from n = 256 to 512.
- **Predictions (strategist):** G_drop at n = 512 is 5–25 % (50 % chance ≥ 10 %); the κ3 × edge family carries most of it. Record yours.
- **M3 Decision (by Oct 8).**
  - **GO** if G_drop ≥ 10 % at n = 512 and the trend is flat or growing. Derive the dominant family's cheap form for our chain — e.g. κ3 × edge: T_abc += c · ŵ2_a ŵ2_b ŵ1_c κ3_abc C_ab (+ perms); with C_off ≈ Σ_{k ≤ K} s_k v_k v_kᵀ (K = 1–4 top modes; at depth a rank-1 mean mode dominates) each existing lineage gains K row-scaled copies of its legs; the triangle needs K² legs. Price it in units against the Sep 28 census. Prototype it as a twin-only birth term with a fitted gain on the v73 path (later on Lane P's pruned path) and measure VAL raw. Hand it to Lane P if it gives ≥ 5 % raw at ≤ +5 % FLOPs.
  - **CLOSE** if G_drop < 10 %: record it as a write-up negative result and go to M4.
- **M4 (if M closes, Oct 8–12) — neighbor-feature corrector screen.** Add Puffi-style neighbor features to the corrector inputs in the twin: each neuron's top-8 |ρ| partners' α, Φ, φ, γ3; kernel-weighted averages exp(−β ρ²), β ∈ {8, 32}; (C⁺α)_i; (K⁺·1)_i. One epoch on TRAIN 900, select on VAL. Kill if VAL raw improves < 2 %.

## Do not
Submit; edit estimator files on main; fit or select on FRESH panels; read FRESH-LOCK outside the logged designation reads; contact organizers or post; occupy the GPU while a Lane P twin job is queued.

## Report
Daily one paragraph in `docs/STRATEGIST_PACKET_2026-10-0X.md` (append a "Lane V+M" section). After V4: the calibration table. After M2: the variant table with both predictions and the decision.
