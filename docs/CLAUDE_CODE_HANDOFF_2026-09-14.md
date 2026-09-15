# Claude Code handoff — 2026-09-14 (evening): Slice 6, "reach the public frontier, then N1"

*Companion to `phase2-strategist-response-2026-09-14.md` (read §§3–6 first) and `exp-lambda-e14-vs-504aldo-2026-09-14.md`. This replaces `docs/E14_REVISED_PLAN.md`, `docs/E14_TREE_CANDIDATE_DRAFT.md` and the Sep-12 handoff as the operative contract.*

**Samuel's decisions (recorded 2026-09-14 late evening, Chicago):**
1. **504aldo code:** re-implement the published mechanisms in our own files; v29 is a local parity oracle only; never submit v29 bytes; post the neutral rules question (§7) — the forum post itself still needs his approval of the exact wording before sending.
2. **Process:** all of §0 is authorized — delete the local-seconds residual gate from decision paths (op-count model gates), relative tolerances only, single-use DECIDE blocks only for candidates with fitted constants, daily fresh-truth cadence, Skeptic only for write-up numbers and designated files, build `predict_public.py` first. Per-submission approval and designation remain his.
3. **λ prediction (P1, Slice 6a):** recorded as **"I don't know"** for the own-torch-AUGMENT λ table vs 504aldo's table. The strategist's pre-registered prediction stands separately (within 15% per layer at l=1–14, R² ≥ 0.6 by layer 4).

## 0. Contract changes (apply first, ⟂)

1. **Delete the local-seconds residual gate from every decision path** (`certify_candidate.py §10`, `preflight.ps1` step 16, AGENTS hard rule 6 "clean mini n_failed=0"). Gate on op count ≤ 22,000 with the 13.4 µs/op server model (`official_config.PHASE2_OP_COUNT_GATE`); print local seconds as advisory only. Run the clean mini with the residual cap lifted so it measures accuracy.
2. **Tolerances**: tiny-shape conformance is relative, f32 candidate vs f64 oracle, 1e-4 rel on every compared field; absolute bounds are banned. Do not re-litigate E14's 2e-5 gate.
3. **Truth blocks**: single-use DECIDE only for candidates with fitted constants (λ tables, feedback ranks). Exact FLOP transformations are certified by ledger diff + tiny parity + one 24-net P01-class run; no fresh block.
4. **Cadence**: every candidate gets a 24-net fresh-truth score the day it compiles; one graded (non-designated) canary per material change. Log a one-line entry per result; no CONTRACT/SKEPTIC blocks for infrastructure.
5. **Build `local_sim/predict_public.py`** ⟂: `--candidate X --control 330950` → mini-100 subprocess run (f64_readonly, caps off) of both, `paired_compare_rows` ratio ± t-CI, projected public = graded(control) × ratio, multiplier {F, F+2²⁴}/2⁴¹, op count → residual forecast, wall projection × 1.60 via `p2_wall_ladder`, Linux memory-gate verdict, all tagged MATH / SERVER_PREDICTED / LOCAL. All pieces exist; ≤1 day.
6. **E14**: stop both E14 files (f32 and f64). Keep `_tree_core`'s consumer/feed plumbing for P1; discard the Gaussian-tree λ.

## 1. Oracles and provenance (⟂)

- Clone `https://github.com/504aldo/whest-p2-cumulant-k3` (MIT) into `external/504aldo/` as a **local parity oracle only**. Never submit its bytes. Add `docs/PROVENANCE_504aldo.md` listing which mechanisms are re-implemented and citing findings F63 (7→4 identity), F68 (κ4 regeneration law), F69 (D21 feedback), F72/F73 (age-gated basis), F86 (cost anatomy).
- Exact oracle for κ3 content: F21 (`estimator_k3_strassen_L4_f21.py`), unchanged.
- New candidate file: `estimator_k3_v30.py`, own code, F21 lineage, one mechanism per commit, tiny-shape parity against both oracles at each step.

## 2. Slice 6a (days 1–3): P1 — κ4 regeneration with our own λ

Goal: F21 raw 3.78e-8 → ≤2.3e-8 at unchanged FLOPs (prediction 2.1e-8 ± 10%).

1. **λ fit (GPU, one evening)**: extend `local_sim/p2_augment.py` (torch `kind=AUGMENT`, 1024×16) to dump per layer l: `G_pre_off(l+1)` (the augmented chain's off-diagonal κ4 core after W-transport), `C_pre_off(l+1)`, `diag g`, the (3,1) slice, `mean(var)`. Networks: 24 own mints (canonical mint, seeds from a new DESIGN block; no public labels touched). Compute λ_l = ⟨G_off, C_off⟩/‖C_off‖², R²_l, per-net spread; also the adaptive-rule observable mean(dG)/mean(var) per layer. Save `tables/lam_own_augment.npz`. **Report λ_l next to 504aldo's `lam_prefit_perdump.npy` (0.95×mean): prediction within 15% per layer, R² 0.6–0.97 rising with depth.** If R² < 0.5 at depth, stop and report.
2. **Consumers in v30** (port from `_tree_core`'s downstream, which already matches v29 l.1091–1112 term for term): `dG = (Wᵀ∘Wᵀ)@(g_prev − λ·diag C_out) + λ·diag C_pre`; `q4 = m·dG` into `pK[(p,)] += q4·w(4,p)/24`; `q31 = (m/2)·λ·C_pre_off`, `q22 = (m/6)(dG_i + dG_j)` into the bivariate `pK[(p,q)]`; post-ReLU `g = 6/(n+4)·(K4 + rowsum K22) − trace part`; κ4→κ3 feed `X3 = A·d(λw1) + P·d(w1²dG)` (E14's `_feed_slices` carries only the first term — add the second; both O(n²) per lineage). Keep `c4` scalar as the trace carrier or fold it into g (document which).
3. Tests: tiny-shape (n=4..16, L=4..6) f32 v30 vs f64 v30 relative 1e-4; v30-with-504aldo-table vs v29 `V17` output at tiny shape (relative 1e-3 — closures differ slightly in degree, so expect not exact); 24-net P01-class score vs F21 (paired). Then `predict_public.py`, then one graded canary at 0.42 B (prediction ≈ 8.5–9.5e-9).

## 3. Slice 6b (days 3–7): P2 + P3 + P4 + P5 — cost to ~0.25 B

- **P2 7→4 identity**: split `A_D = diag(S3) + 3·S21ᵀ` as `S21 = S_sep + R_res`, `S_sep = d(e)·C_off·d(w1)` with `e = w(1,2) − 2·pK1·w(1,1)`; put `3·S_sepᵀ` on the A leg (`3·a_b·d(e)`), carry `R_res ≈ Q·B` at rank 16 on a transported thin leg (range finder, sketch from `W[:, :16]`-style fixed columns); drop the dense `A_D` transport and the `(M∘M)@A_Dᵀ` contraction. Expected 432u → ~300u at +1.6% raw.
- **P3 D21 feedback rank 16**: `D21 ≈ Q_f B_f` (two QR passes), `X_t = 1.5·d(w2)·D21 = F1 R1`, `Y_t = 0.5·d(w1)·D21ᵀ·d(w3) = F2 R2`; births `X1 = 3a_b + X_t`, `Y1 = a_b·d(w2) + Y_t`; F1/F2 travel on the thin stack. Expected −8–10% raw.
- **P4 Strassen L5 / pooling / block-sym C_pre / final trim**: `_sb` → min block 32, depth 5, all products `out=` into persistent pooled buffers on the estimator instance (no `concatenate` in the hot loop; `gc.disable()` inside predict), C_pre via three block products of the 2×2 partition (0.75×), last layer: `var = rowsum((WC)∘W)`, D3 only, no D21/C_off. Gate: ops ≤ 22k; memory under `RLIMIT_AS` 8192 MiB with the grader's Python 3.10 / NumPy 2.2.6 in WSL.
- **P5 age-gated basis**: sources older than 4 transports → shared basis Q_c (n×384) by one-pass randomized range finder on the weighted leg Gram; older than 7 → nested 224; dense legs re-formed per layer from factors; contraction `[Σ LA F_Aᵀ + LP F_Pᵀ] Q_cᵀ`. Reproduce 504aldo's age law on 8 nets before shipping (r=256 at age 4 should cost ≈+22% raw).
- Canary after P2+P4 (≈0.28 B, ≈5.9e-9) and after P5 (≈0.25 B, ≈5.4e-9).

## 4. Slice 6c (week 2): N1 kill test, then port

Kill test on the **lean torch chain** (504aldo's `lean/` or our `p2_kprop_bridge`): at ages 5–8, compare raw vs rank r for (a) forward-energy basis (504aldo) and (b) sensitivity-weighted basis: Gram weighted by JᵀJ where J = Π_{k>l} D_k W_k, D_k = diag(Φ(α_k)) from the K2 chain (compute J by the transposed transport from the output, 1 matmul per layer, O(L·n³) total = 0.03 B; or restrict to its top-128 right singular vectors). Predict: (b) halves r at equal raw. If it passes, implement the **merged old core**: all sources older than the gate merged into one Tucker core T (r×r×r) in the sensitivity basis, transported by the r×r restricted propagator, (2,1)-slice read at r³/n² units per layer, rank chosen per MLP from the core spectrum (N7). Target total ≤185u (0.18 B). If it fails at r>200, fall back to the clones' trims (join overhead: warm-start the sketch from the rotated previous basis; second QR pass off; age gate 4→3 where accuracy allows) and Strassen L6.

## 5. Slice 6d (week 3): raw levers, each a one-day lean-chain ablation

N3 λ refit with target = consumed mean correction; N4 exact κ4 metric WᵀW on the diagonal channel (meter it); N2 thin κ4 memory legs (rank 16/32 per source) vs exact per-source κ4 (−9% ceiling); N5 universal diagonal prior; N6 exact bivariate ReLU covariance for top-64 pairs. Port only those with ≥3% raw at ≤5u.

## 6. Predictions to record before running (Samuel may replace any)

- λ_own vs 504aldo table: within 15% at l=1–14; R² ≥ 0.6 by l=4.
- F21+P1 on 24 nets: 2.1e-8 (range 1.9–2.4e-8).
- v30 after P2–P5: raw 2.15e-8 ± 0.1, 255–270u, ops 18–21k, residual forecast 0.25–0.30 s, graded ≈5.4e-9.
- N1: sensitivity-weighted basis reaches 504aldo's rank-384 accuracy at r = 128–192; merged core total cost 170–190u.

## 7. Rules question to post (neutral wording, Samuel's approval)

"Team 504aldo released their Phase-2 estimator under MIT and invited reuse; several accounts have since submitted it. For prize verification under §6 ('not derived from another Participant's submission'), is building on that code with attribution and substantial own changes acceptable, and where is the line?" Until answered: own re-implementation only, provenance file maintained, no v29 bytes submitted.
