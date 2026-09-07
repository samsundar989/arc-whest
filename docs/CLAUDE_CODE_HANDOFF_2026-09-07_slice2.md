# Handoff to Claude Code — Slice 2 (Sep 7–19): submit the exact chain, then cost-engineer it to ≤0.45 B and start the accuracy layer

Read first: `docs/PHASE2_STRATEGY_2026-09-05.md` (canon, rev 10), the top of `docs/CURRENT_STATE.md` (Slice 1 outcome), `experiments/LOG.md` "THE RANKED COST-LEVER LIST FOR SLICE 2", and this file. Slice 1 is reviewed and accepted (see §0). Standing rules from the Slice 1 handoff are unchanged except where §1 says so.

## 0. Slice 1 review verdict (Cowork strategist, Sep 7)

Accepted. `estimator_k3_full.py` (sha256 `220d2314…`) is the exact factorized-K3 chain at 0.7797 B, raw 3.925e-8 on the organisers' mini-100, adjusted 3.06e-8, all-layers 1.59e-8, 3,249 ops/MLP, constant FLOPs, 0 failures with the local residual cap lifted, VmPeak 2.8 GB. Independent review found **no blocking issue at the live shape**: no `copy=False`, no in-place mutation, zero scalar materialisations, no exception swallowed in `predict`, no randomness, no file loads, budget self-audit at 0.95 B with 22% headroom, and every fnp API pattern it uses (`clip`, `isfinite`, `float64(0.0)`, `[None,:]`/`[:,None]` broadcasting, no-axis `.sum()`, `[:, -1:]`) has an exact precedent in canary4, which graded cleanly on 0.16.1.

**Rule 6 is relaxed for this one submission, by Samuel.** The local residual reading (0.45–0.51 s) is an in-process artefact: the width sweep at fixed op count (66 → 158 → 480 ms for 256/512/1024) shows it scales with array traffic, which the client/server grader attributes to `flopscope_server_kernel/overhead/comms`, not to residual. Graded per-op residual on canary4 was 11.1–11.6 µs; the top-8 run 25–35k ops at 0.29–0.39 s, i.e. ~11 µs/op with far larger operands than ours. Predicted graded residual for this file: ~38 ms (10.6× under the cliff). Residual instrument note goes in `docs/VALIDATION.md`: certify §10's local residual is a ratio-only instrument; the gate for candidates is `ops × 11.5 µs ≤ 0.3 s` plus one graded confirmation.

Hygiene to fix before the next candidate (not before this submission): `_regen` (lines ~980–999) is not a cheaper fallback — it is `gens=1` again and would guarantee an over-budget row at non-live shapes where `_fit_generations` returns −1; make it a real cheaper path (the regenerating collapse at 0.077 B, which the collapse-equivalence test already validates) or fall through to v2.7's path. Stale docstrings at lines ~52–68, 79–82, 353–358, 980–987 contradict the code and CURRENT_STATE. The memory-gate artefact in `experiments/2026-09-05_slice1/` is for sha `de328435…`; store the 220d2314 one beside it.

## 1. Submit (first action; one slot)

Samuel's approval for **this specific submission** is given in the message that delivered this file: submit `estimator_k3_full.py` sha256 `220d2314…` exactly as certified — nothing else, no edits. Steps: `whest validate` → `whest package` → verify the packaged inner sha matches → `whest submit` → append to `experiments/submissions.jsonl` with the approving message → poll and fetch the graded report (`local_sim/fetch_graded_report.py`) → record in `LOG.md` and `CURRENT_STATE.md`: adjusted, raw, all-layers, per-row FLOPs (check the offset model: expected 1,714,587,938,406 + 2^24 + 32,768), wall/kernel/overhead/comms/residual min/med/max, failures. Expected: adjusted ~3.1e-8, ~#33–38, residual ~38 ms, wall ~25–30 s.

If it FAILS: do not guess. One slot for a stage-channel diagnostic build (`estimator_wp18_stage.py` pattern, stages at: cast, layer 0, first tower layer, layer 8, last layer, assembly); then stop and report. If it fails on `residual_wall_time_exhausted`, that is the one outcome that changes the plan — report the per-row residual numbers immediately and do not build Strassen until the per-op residual model is re-fit.

## 2. Cost engineering to ≤0.45 B, at zero accuracy cost (the score lever)

The chain's 797 matmuls carry ~93% of the bill. Slice 1 proved generation truncation and last-layer capping are losing trades, so every lever here must be exact (parity ≤1e-6 relative in f32 on the slices, and the 24-net paired raw MSE within GT noise).

2a. **Batched Strassen–Winograd, bounded by the ops budget.** The constraint that matters is not FLOPs but flopscope operations: the residual cap ≈ 30k ops/MLP at 11.5 µs/op, and the leaders sit at 25–35k — that is almost certainly *why* their residual is 0.3–0.39 s. So: first group the tower's matmuls per layer by shape class and issue them as batched calls (the `(n, kn)` block spelling already makes `contract_W` three calls per layer; do the same for the `(2,1)`-slice products and the Wick assembly). Then apply Strassen–Winograd recursively **on the batched operands** (stack the 7 sub-products along a batch axis; form the 15 Winograd combinations as batched adds on views; reconstruct into a pre-allocated block grid — marius's Phase-1 write-up §4 describes exactly this spelling at 10,402 calls total). Measure, for L ∈ {1, 2, 3, 4}: FLOPs (expect (7/8)^L: 0.875 / 0.766 / 0.670 / 0.586), ops/MLP, parity, and the 24-net raw. Ship the deepest L whose ops ≤ 24k (leave 20% margin under the cliff). Prediction to record: L=3 lands at ~0.52 B with ~15–20k ops; L=4 is feasible only if the per-layer batching gets the base op count under ~4k.

2b. **Dead/cold-column pruning with exact compensation** (≈0.8×, unpriced). From the K2 state, neurons with Φ(α) below a threshold contribute κ₃ below label noise; drop their columns from the lineage and carry their means through the next layer's rows exactly (marius §3.4 damping ramp is one recipe; ours can be exact because we have the analytic mean). Price the FLOP saving vs the accuracy delta on 24 nets; keep if the delta is within GT noise at ≥10% saving.

2c. **Aliased-operand self-contractions** (≤1.1×) — spell every `X @ X.T`-shaped product as `einsum('ij,kj->ik', X, X)` where the operands are literally the same buffer; measured 0.5005× on those ops.

2d. **Per-MLP variable spend** — suliman_tadros, J2W and Puffi have variable FLOPs per MLP. After 2a/2b, check whether a cheap pilot statistic (Σ effective rank at layer 4, κ₃-slice norms) predicts per-MLP K3 error on the 96 P01 nets (R² ≥ 0.3 ⇒ route Strassen depth / pruning threshold per MLP).

Target by Sep 19: ≤0.45 B at raw 3.9–4.0e-8 ⇒ adjusted ≤1.8e-8 (~#25). Canary once when 2a lands (Samuel's approval per submission), once more when 2a+2b land.

## 3. Accuracy layer (torch, GPU queue, in parallel from day 1 — Part C of Slice 1 was never run and it is now the critical path)

Same protocol as before: predictions recorded before each run; P01 48/48 fit/held-out; confirm on the 24-net 2^26 set; cross-replica MSE with `gt_noise` printed.

3a. **O1 — fitted renormalisation of the K3 chain**: per-layer scalars on the mean, and separately on the (3,) and (2,1) κ₃ slices; sequential closed-form least squares. Prediction 1.2–1.5×.
3b. **Edgeworth-4 readout with per-neuron diagonal κ₄** (replace the global scalar κ₄). Prediction 1.1–1.4×.
3c. **AUGMENT sign** (ARC's degree-3 vector + degree-4 scalar joint state) at 1024×16. Prediction: unknown sign.
3d. **κ₄ response closure (Puffi SSC)** — read `algorithmic_contribution_puffi.pdf` (https://discourse.aicrowd.com/uploads/short-url/yQsl71y2a3zDgb4SpYQFnt9EpoE.pdf): κ₄ regenerated from ≤4 covariance-response matrices B_a with a ridge-fitted coupling Λ (fit offline to marginal κ₄ on ≥160 minted nets), response-memory coefficient, drift corrector 31→32→32→1. Implement on the torch K3 chain; measure on 24 nets. Go/no-go at ≥1.5× by Sep 19. This is the most evidence-backed route to the leaders' 2×; Puffi got −77% at 256×32.
3e. **Corrector on chain features** (per-neuron equivariant head + pooled context; features α, σ, κ₃ slices, κ₄, gate probability at every layer; target = final residual; cross-fitted; James–Stein-shrunk). Start logging the features from 3a onward; train once 3d's verdict is in.

Whatever lands in 3a–3d must be spelled in fnp at ≤5% FLOP overhead and re-certified; the readout-side items (3a, 3b) are near-free.

## 4. Nominee discipline

Keep two live candidates at all times: A = the best-scoring exact chain with ≥1 clean graded run on the current stack; B = canary4 (8.5e-8, 13× residual headroom) until A has ≥3 clean graded runs, then B = the previous A. Every candidate: twin arm default (f64 read-only), `edge_cases --real-flopscope --phase2` on the candidate, ops ≤24k, projected residual ≤0.3 s, Linux memory gate on the candidate's sha, no exception swallow.

## 5. Hand back (by Sep 19, in `CURRENT_STATE.md` + `LOG.md`)

Graded telemetry for every submission; the Strassen ops-vs-L table; the pruning price; the four accuracy numbers with CIs and pre-registered predictions; the SSC go/no-go; the updated ranked lever list for Slice 3 (Sep 20–Oct 3: integrate accuracy layer, push to ≤0.35 B, hardening, write-up near-final); anything that contradicts the strategy doc.
