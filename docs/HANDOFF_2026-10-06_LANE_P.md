# HANDOFF — Lane P (session 1), 2026-10-06: insurance B → decided-neuron pruning → floor push

Repo `C:\Users\samsu\Documents\arc-whest`. **First action:** replace `docs/PLAN.md` with `docs/PLAN_rev18_2026-10-06.md` (rev 18 is now the only plan), commit it with this handoff and `docs/HANDOFF_2026-10-06_LANE_VM.md`, and push to origin (nothing has been pushed since Sep 17).

Read: `docs/PLAN.md` (rev 18), the project docs `claude/deep-review-2026-10-01.md` (§6 evidence) and `claude/review-2026-10-01-A1-architecture.md` (code map of v73 with line numbers), `RESULTS.md` § Anatomy 2 and the Sep 28 cost census (`experiments/2026-09-28_cost_sites/`).

**Ownership.** You own main, `RESULTS.md`, `experiments/submissions.jsonl`, `docs/DESIGNATION.md`, `docs/CURRENT_STATE.md` and every `estimator_*` file. Session 2 (Lane V+M) works on branch `lane-m` and will hand you the FRESH-A/FRESH-LOCK panels (tools/panels.py + tools/ev.py changes) — review and merge them. GPU via workerq; your twin jobs have priority.

**Why this lane [I, evidence in the deep review §6].** 13 of the top teams bill a different FLOP count per network; their per-network counts correlate r = 0.8–0.99 with each other and ≈ 0 with difficulty; ours is a constant 472.2 G. On our FIT nets, at the last layer 19–28 % of neurons are dead (α < −3 / −2) and 38–57 % decided (|α| > 3 / 2). Dead rows carry ~2e-5 of the Φ²-weighted leg energy; decided hubs ~1.6e-5 (τ = 3) of the φ²-weighted birth energy. A crude cost model of skipping them reproduces the leaders' per-network spread (CV 2.2–2.4 %, max/min 1.12–1.14) and gives F/B ≈ 0.14–0.16 at near-equal raw.

---

## P0 — Insurance B (today, ≤ 3 h)
1. Build `estimator_k3_v74_ws.py` = `estimator_k3_v73_gb.py` with `_STRASSEN_MIN_BLOCK = 64` (the v68_wino → v68_ws recipe); nothing else changes.
2. Gates (PLAN §5): validate; DEV-100 paired vs v73 (expect ≤ ±0.01 % raw); real-meter f64 cost (expect ≈ 0.2313 B); ops; WSL memory; package hash.
3. Submit on the A/B slot. Expected graded ≈ 4.98e-9 (2.1515e-8 × 0.2313), median wall ≈ 52 s.
4. `DESIGNATION.md` evidence: A = v73_gb #332893, B = v74_ws once graded. Samuel designates.

## P1 — Accuracy-only mask test (Oct 6–7, ≤ 1 day, twin on GPU)
Goal: measure the raw-MSE cost of ignoring decided neurons in the κ3 machinery before any engineering. Add a twin flag (mask hook) on the v73 code path; α_l = μ_pre/σ per neuron at each layer.
- **R (dead rows → inner dimension of the next transport):** after the Wick scaling at layer l, zero rows i of every κ3 leg (exact tier A_L, M, Z, Y and hub legs; old-tier rows of Q) where α_l,i < −τ_R. Keep μ, C and **D3 for all neurons** (a dead-ish neuron's own D3 still moves its own mean by up to ~1e-4).
- **H (hub columns at birth):** when appending the newborn lineage at layer b, zero the **two-leg block** columns j (A_L[:, j], w2b_j and the hub contributions) where |α_b,j| > τ_H; zero the **slice-block** columns (sb_j, eb_j and the matching P2-residual rows) only where α_b,j < −τ_H — always-on neurons still carry the Gaussian C² repeated-index term (w2_c C_jc² from kink partners).
- **D (D21 readout):** zero D21 rows and columns i where α_l,i < −τ_D (the closure multiplies a dead neuron's pair terms by its Hermite weights). The last layer has no D21; leave it untouched.
- **Grid** (VAL-100 free-running raw vs the unmasked control on the same code): R τ ∈ {2.0, 2.5}; H τ ∈ {2.5, 3.0, 3.5}; D τ ∈ {2.5, 3.0}; each alone, then the combined **safe** set R2.0/H3.0/D3.0 and the **aggressive** set R2.0/H2.5/D2.5.
- **Log** kept fractions per layer and per net (rows, hubs, D21 rows) on FIT nets, and a **modeled FLOP factor**: take the job2309 per-site census and scale each site by its kept fractions (transport ∝ kept rows(l−1) × kept hub columns; D21 readout ∝ kept D21 rows(l)² × kept hub columns; old-tier sites ∝ kept rows).
- **Predictions (strategist):** R2.0 ≤ +0.5 % raw; H3.0 ≤ +0.5 %; D3.0 ≤ +0.5 %; safe set ≤ +1.5 % with modeled FLOPs −20 to −30 %; aggressive set +3 to +8 %. Record yours before running.
- Then a short refit of the safe set (S2-length: gains + corrector, lr 3e-4, TRAIN 900, select on VAL).
- **Kill:** safe set > +3 % raw after the refit ⇒ close the lane, record the table, go to P3 with the remaining time.

## P2 — Compact implementation `estimator_k3_v75_kas.py` (Oct 7–9)
- **Order inside a layer:** covariance transport → var → α_l (as today) → index selection → compacted κ3 work.
- **Selection, variant A (default, zero materializations):** per-layer, per-role fixed keep counts K_l^R, K_l^H, K_l^D from TRAIN-900 statistics (99.9th percentile of the kept count at the chosen τ, + 32, rounded up to a multiple of 32/64 for Strassen). Indices from `fnp.argsort` / `argpartition` of the role score (keep the K largest α for R and D; the K smallest |α| for the two-leg H columns), then `fnp.take`. Constant FLOPs. Verify that flopscope takes an integer index array device-side without a sync (`MaterializationCounter` = 0) and price the gathers (4 FLOPs/element).
- **Variant B (optional, like the leaders):** read the counts to the host once per layer (≤ 48 materializations/net), round to a multiple of 32, measure residual per sync; ship only if the projected worst residual ≤ 0.30 s.
- **Compact contractions:** transport `W_l[rows_kept, :]ᵀ @ legs[rows_kept, :]`; the newborn lineage stores only kept hub columns (so every later transport and readout shrinks); D21 only for the kept × kept block, with the closure's pair terms evaluated in compact index space (shrinks the O(n²) closure work too); old-tier Q rows compacted per layer. Scatter back to full n only where a full object is required (μ, C, D3, the output row).
- **Hygiene folded in:** layer-0 `C_pre = WᵀW` directly (not `_mm(Wᵀ, I)`, ≈ 0.27 % score); route the join's big products through `_mm` (≈ 0.7 %); test the P2 diagonal-leak fix (`sb = S3 − rowsum(Zb ∘ Lb)`) as a twin A/B only.
- **Shape safety:** per-layer K tables clamp, and fall back to no pruning for depth ≠ 16 or width ≠ 1024 (whestbench issue #149: the smoke test uses non-suite shapes).
- **Verification:** width-32 dense-tensor identity test with masks; twin-mask ↔ compact-estimator parity ≤ 1e-3 relative on the DEV mean; real-meter FLOPs; ops ≤ 26,000 (gathers add ops — budget them); WSL memory; wall-law projection.
- **Refit** through the twin on the compact chain (gains + corrector + gates, S2t-length), select on VAL.
- **Acceptance:** FRESH-A paired vs v73 (from the day Lane V delivers it; DEV-100 + VAL until then) plus all hard gates ⇒ canary (A) and its leaf-64 twin (B).
- **Prediction (strategist):** F/B 0.145–0.165 at raw ≤ +1.5 % ⇒ graded 3.2–3.6e-9.

## P3 — Floor push (Oct 10–14)
1. Re-measure the knob elasticities **on the pruned chain** (age 4/5/6, shared rank 192/256/320, P2 rank). Under pruning each knob's FLOP price falls; move toward 0.10–0.12 B only where the marginal rule (PLAN §1) says the score improves.
2. Integrate Lane M's term if it is a go (the strategist will specify the form).
3. One untested pilot if time allows — **train through truncation**: the age-3 fold (−12.8 % FLOPs at ×1.86 raw, never refit) refit end-to-end in the twin (corrector + gains), 1 epoch. Kill if it is not within 1.15× of the refit control's raw.

## Do not
Reopen PLAN §6; fit or select on FRESH panels; choose between candidates on public-board readings; designate, contact organizers, or post; run a twin job and a probe on the GPU at once.

## Report
Daily one paragraph in `docs/STRATEGIST_PACKET_2026-10-0X.md` + RESULTS rows. After P1: the mask table (raw Δ, kept fractions, modeled FLOPs) with both predictions. After P2: the cost table, FRESH-A paired result, graded row. Blocked on Samuel > 2 h ⇒ take the conservative option, note it, continue.
