# HANDOFF — Lane P5 (session 1, branch main), 2026-10-10: the cost front (about −18 % F), integrating the κ₄ pieces, daily canaries

**Repo and ownership.** Repo `C:\Users\samsu\Documents\arc-whest`, branch `main`. You own main, the estimator files, RESULTS, the ledger, `docs/DESIGNATION.md` (recommendations only; Samuel designates), CURRENT_STATE and the pod queue.

**Read first.**
- `docs/research/2026-10-10_pivot_rev22.md`: §0, §1.3, §2.2, §3–§6.
- `docs/research/2026-10-10_deep_review_strategist.md`: background.

This handoff replaces `HANDOFF_2026-10-10_LANE_P4.md`, which was not started; do not execute P4.

**Standing rules, unchanged:**
- PLAN rev 18 §5 hard gates:
  - ops ≤ 26,000 (28,500 only if Samuel approves it for P5.5);
  - projected residual ≤ 0.33 s;
  - wall ≤ 105 s for A and ≤ 95 s for B, noisy-implied;
  - setup < 2 s;
  - WSL 8 GiB PASS;
  - parity ≤ 1e-3;
  - hash = tested source.
- At most 5 canaries per UTC day plus 1 A/B slot.
- Record your prediction and the strategist's before every run.
- FRESH-A paired for any change that is not bit-identical.
- No fitting or selection on FRESH panels; FRESH-LOCK ≤ 3 reads, for designation only.
- No external messages. No `out=`, `where=` or symmetry tags.
- **No write-up work** until we are in the top 10.

## Why

v81d (#334948, 3.32003e-9 = raw 1.8833e-8 × F/B 0.17629) is #33. The top 20 needs F/B ≤ 0.1407 at our raw.

The exact cost model shows two pruning levers that this morning's plan missed:
- **two-sided D:** −8.84 % F alone, −8.0 % on top of the bundle;
- **H = kink-active columns** once the κ₄ hub leaves the legs: −6.03 % alone, −5.65 % on top.

With the billing bundle the joint stack measures **F/B 0.1455 (−17.5 %)**, and about −18.5 % with covariance pruning. Lane M5 runs the κ₄ D31 channel and the hub move; you integrate.

**Targets:** F/B ≈ 0.145 by Oct 15, and a final score of 2.3–2.6e-9.

## Tool: exact cost model

`docs/research/2026-10-10_flopmodel/` is a shape-only interpreter of flopscope 0.12.1. It reproduces the real meter to the FLOP and the op:
- v81d: 387,664,679,697 FLOPs and 25,741 ops;
- v77: 395,412,204,457 FLOPs;
- v81d_ws: F/B 0.18030 and 23,181 ops.

**How to run it:**
- Run `python census_v81d.py <path-to-estimator>` from inside that folder, in a fresh process. It installs fake `flopscope` and `whestbench` modules.
- It writes `.mm.json` and `.ledger.json` into the working directory, so run it from a copy if you care.
- The scripts first add `~/scratch` to `sys.path`. Delete that line, or make sure no stale copies exist there.
- `variants*.py` hold the measured patches (block, fused advance, join via `_mm`, skip joins, sandwich split, last-layer rows, old rebuild on kept rows, optimal depth, two-sided D "V8" and columns-only "V8b"). They form an import chain; `variants7.py` crashes on its third run.

Price every change with it **before** metering, then confirm on the real meter.

## P5.0 — today (Oct 10)

a. **Commit the strategist's files unchanged, in one commit:**
   - `docs/research/2026-10-10_pivot_rev22.md`
   - `docs/research/2026-10-10_deep_review_strategist.md`
   - `docs/research/2026-10-10_flopmodel/`
   - `docs/research/2026-10-10_k4x/`
   - `docs/HANDOFF_2026-10-10_LANE_P5.md` and `docs/HANDOFF_2026-10-10_LANE_M5.md`
   - In CURRENT_STATE, mark the P4/M4 handoffs superseded.
b. **A/B slot.** Gate and submit `estimator_k3_v81d_starall2ws.py` as B insurance. Expected ≈ 3.39–3.40e-9.
c. **v82a** = v81d + `fnp.block` for the output reassembly in `_sb` (est:1125) and in the sandwich (est:1361, 1372), + `nan_to_num(X, nan=0, posinf=0, neginf=0)` for the three n×n closure guards.
   - Expected −1.17 % and −0.045 % F, −1,440 ops.
   - Bit-identical on DEV nets, so no FRESH-A is needed.
   - Canary: compare the graded F with an interpreter run **on the exact v82a file**; it must match to the FLOP. This confirms the server bills `block` as 0.12.1 does.
d. **`docs/DESIGNATION.md`** (recommendation): A = #334948; B = today's v81d_ws once graded, else #334918.
e. **Start two-sided D now** (P5.2 step 1 takes minutes), so its refit can be queued on the pod tonight. It is the critical path.

## P5.1 — v82, the bundle (Oct 10–11)

One measured commit per item. Interpreter numbers are alone on v81d.

1. **Skip the old-tier joins at layers 14 and 15** (join call est:4293; its guard is 4292): −0.90 % F, −662 ops. Lineages stay exact longer, so this is more accurate, not less.
2. **Last-layer output rows K16 = 800**: patch est:4021, 4067 and 4111 only. Keep est:4028, the last-layer variance product, at full width. −0.79 %.
3. **Old-tier rebuild `Q·[FA|FM]` on D-kept rows only** (est:739): −1.30 %. This is also part of P5.2(ii); build it once.
4. **Fused Wᵀ_r products in `_advance`** (est:1181–1207: A_L, M, Z, Y and Q in one `_mm`, padded to a multiple of 32, outputs split by free slicing): −0.37 %, −2.1k ops.
   - Measure VmPeak (+0.3–0.5 GB) and wall. GOTCHAS records +2 % wall for an earlier `_advance` merge; split into two groups if needed.
5. **Two-level symmetric sandwich split:** −0.23 %, +585 ops.
6. **Covariance input pruning — new; add it to the interpreter first.** Drop dead (D set, τ 2.5) rows and columns of C_post before Wᵀ·C (est:1354) and the symmetric parts (est:1359/1360).
   - Estimate about −1.3 % F, accuracy +0–0.5 %. A dead neuron's variance is about 0.15–0.6 % of a live one's.
   - Build it only if the interpreter shows ≤ −1 %.
7. **Do not** route the join's plain `@` (est:839/845/854) through `_mm`. It saves −0.20 % but costs +3.2k ops (+2.3k after the joins are skipped). Keep that op headroom for Lane M's paid κ₄ pieces and for P5.5.

**Package.** Package v82 and v82ws. One FRESH-A paired run for the whole bundle is fine if each item is bit-identical or individually twin-checked. Then canary.

**Predictions.** Strategist: −4 to −5 % F (as the bundle measured, minus join-via-`_mm`, plus covariance pruning), with Δraw within ±0.5 %. Write yours first.

## P5.2 — v83, two-sided D (Oct 10–12)

**This reverses GOTCHAS #2** ("ship row-only D") on new evidence. Treat it as a bet, P ≈ 0.45.

**What the earlier evidence says** (RESULTS P1 and P2):
- P1's two-sided rows-and-columns mask (D2.5; v73; per-net thresholds) cost:
  - +10.55 % with the corrector reading the masked d21e;
  - +0.56 % with the corrector reading the unmasked feature (Dc2.5).
- Fixed keep counts add about 1.1 points.
- Leg-row pruning and dead-row D3 = 0 were never measured.
- The one refit of a two-sided mask (P2 safe-d) was cancelled at +7.8 %, on an lr 3e-4 S3 recipe whose own control spiked +6.8 %.

**Build:**
- (i) D21 columns restricted to the D set: extend the pad+take scatter-back at est:1494 to columns, and drop dead columns from the right operands of `_shared_contract_lean` (est:745–775);
- (ii) every leg transport in `_advance` computes only D-set output rows (the next layer's R rows are a subset of D, so its inner dimension is unaffected), and the old-tier rebuild runs on D rows;
- (iii) D3 and D4 = 0 on dead rows;
- (iv) the corrector's d21e (est:3878) over kept columns only.

**Interpreter:** −8.84 % alone; −8.0 % on top of the bundle (0.16758 → 0.15422).

**Step 1 — minutes, before any refit.** In the twin's mask arm (two-sided D2.5, with fixed keep counts):
- scale the kept-column d21e by a per-layer energy ratio ρ_ℓ = E[Σ_all D21²] / E[Σ_kept D21²], calibrated on FIT;
- measure VAL-100 unrefit;
- also run (ii) and (iii) on their own, to price them.

**Step 2 — one pod batch with matched epochs:**
- (a) the two-sided arm with the calibrated d21e;
- (b) columns-only (V8b, −3.55 % F). It has the same d21e failure mode, so it is not an independent fallback;
- (c) a same-recipe control.

Recipe: corrector + gains + sg, cosine schedule, lr ≤ 1e-4, starting from `experiments/2026-10-08_p3/refit_starall2/params_best.npz`. That file is untracked: use its absolute main-checkout path and add it to the pod tarball.

**Predictions** (strategist):
- Step 1: +1 to +4 % unrefit with the calibrated feature (+10 % without it).
- After the refit: +0.3 to +1.5 % vs the control.

**Kill:** > +1.5 % vs the control after the refit, for both (a) and (b). Then keep row-only D.

If it passes: FRESH-A, then canary.

## P5.3 — integrate Lane M5 (Oct 12–15)

Integrate in order of arrival:
- (a) the masked star-Q31 fold;
- (b) the radial Sym(μ_pre⊗κ₃) κ₄ terms;
- (c) the hub moved out of the legs, plus H = kink-active columns:
  - −6.03 % F alone;
  - KAS H table with 16 entries from the FIT q50 kink-active counts at τ 2.5: 1024, 1024, 1024, 992, 928, 896, 832, 800, 768, 736, 704, 640, 608, 576, 576, 544 — check against `kink_tables.json` in the flopmodel folder;
- (d) paid κ₄ pieces, within the op budget.

For each item:
1. port it from Lane M's twin arm;
2. check parity ≤ 1e-3 against the twin;
3. price it with the interpreter (F and ops);
4. run a joint refit on the pod (cosine restart from the current params) with a matched control;
5. run FRESH-A paired, then a canary.

Accept on the marginal rule: −Δraw > ΔF/(1+ΔF). Stack accepted items into v84 and v85; at least one canary a day.

## P5.4 — refits (continuous, pod)

Queue, in order:
- **R1 (tonight):** the P5.2 step 2 batch.
- **R2:** v82/v83 joint refit.
- **R3:** v84 with Lane M's free items.
- **R4:** v85 with the paid items.
- **R5:** a final polish at low lr.

Recipe: lane-m `docs/RUNPOD_ACCESS.md` (`git show lane-m:docs/RUNPOD_ACCESS.md`; fetch organizer data on the pod from HF).

Rules:
- Every comparison uses a same-recipe control with matched epochs; refits alone gain about 0.5 % per epoch.
- Select on VAL-100; accept only on FRESH-A paired.
- Stop a recipe after two consecutive restarts that gain < 0.3 %.
- Lane M's research refits fill gaps in the queue.

## P5.5 — spend freed wall on FLOPs (Oct 15–16)

After pruning has cut wall:
- **Per-shape optimal depth** (`_depth_for`, est:1153; leaf ≥ 12): −1.0 % F on the pruned stack, at about +3.1k ops (about 28.1k in total) and +15–25 % wall.
  - Build only with Samuel's approval for ops ≤ 28,500, and only if wall A stays ≤ 105 s on DEV net 3 (×2, alternating).
- **Adaptive per-network keep counts** (Variant B, `HANDOFF_2026-10-06_LANE_P.md`:33; host-read counts, ≤ 48 materializations, projected residual ≤ 0.30 s): about −1 % raw from accuracy, about 0 F.

## P5.6 — hardening and designation prep (Oct 16–17)

- FRESH-LOCK (≤ 3 reads in total) on the A and B finalists.
- B = the ws build of the final A (block 40, no ≥2³⁰ extra level). Projected VmPeak ≤ 7,800 MiB at 250 predicts in one WSL process under RLIMIT_AS 8,192 MiB.
- Package hashes = tested source.
- `docs/DESIGNATION.md` recommendation: A = best score; B = the wall- and memory-safe twin.
- Freeze A at Oct 17 12:00 UTC. Keep one slot for B insurance.

## Do not

- Reopen closed items: κ₄-core loop, gain-mode κ₄ templates, exact dressing correction, always-on D21 rows (Da +34 to +87 %), alive-block covariance, rank/age/fold changes, compact-history carriers, learned pair closures (Step 3.3, ≤ 0.75 %).
- Merge the `_dslice21` call sites (+28 % wall).
- Restrict D21 rows to kink-active neurons.
- **Trap:** `_layer` assigns a local named `_sb` (~est:4375–4394). Route new products through `_mm`.

## Report

- A daily paragraph in `docs/STRATEGIST_PACKET_2026-10-1X.md`.
- RESULTS rows with both predictions.
- A running ledger of cumulative ΔF, Δraw and ops against v81d.
- A canary graded > 2 % worse than projected stops the line until it is diagnosed.
- Flag immediately if the selection UI opens, a flopscope release or re-evaluation lands, or an organizer post appears.
