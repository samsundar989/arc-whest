# Claude Code handoff — Slice 7 (2026-09-16 → 09-29)

You are the builder for the ARC White-Box Estimation Challenge Phase 2 repo at `C:\Users\samsu\Documents\arc-whest`. Read, in this order, before touching anything: `AGENTS.md`, `docs/CURRENT_STATE.md`, `docs/STRATEGIST_RESPONSE_2026-09-16.md` (the operative decision; §2 and §6 are your work list), `docs/STRATEGIST_RESPONSE_2026-09-14.md` §§4–6, `docs/CLAUDE_CODE_HANDOFF_2026-09-14.md` §0 (the process changes Samuel authorized), `docs/PROVENANCE_504aldo.md`, `experiments/2026-09-16_exp_q31/README.md`, and the last ~600 lines of `experiments/LOG.md`.

## Where we are

- Standing graded candidate: F21 `#330950`, public raw 3.78e-8, C/B 0.4222, adjusted 1.596e-8, rank ~67 this morning. Leaders 2.5–2.8e-9 (raw 1.7–1.9e-8 at 0.13–0.16 B). 504aldo's public MIT chain and ~25 clones occupy 4.2–5.4e-9.
- New candidate `estimator_k3_v31.py` (P1 κ4 regeneration with our label-free λ table + P2 separable (2,1) leg, rank-32 equivariant sketch): on the 24 P01 design nets raw 3.907e-8 vs F21 4.792e-8, C/B 0.30097 (metered), 12,665 ops, Linux 8 GiB gate 23/23, `--parity` five claims OK, budget guard over-predicts. Projected public ≈ 9.3e-9. Certification and the two mini runs were in flight at handoff time.
- Strategist's finding (proxy at 64×8 / 128×8, `experiments/2026-09-16_exp_q31/`): ~85–90% of 504aldo's κ4 gain rides on the λ term inside the transported κ4 diagonal, `dG = (W∘W)@(g_prev − var_prev·λ) + var_pre·λ`, consumed through the (4)/(2,2) rows; the (3,1) consumer is ~0–9%; the κ4→κ3 birth feed is ≤1.14× on the proxy but 504aldo's F68 says ~4× at 1024×16. Our v30/v31 gets 1.24× from the same channel where a clean K3 base should get ~1.75×. The missing factor is most likely one line in the dG transport or in the birth feed, not a missing table row.

## Decisions already made by Samuel (do not re-open)

1. 504aldo's code (`external/504aldo/`) is a **local parity/diagnostic oracle only**; re-implement mechanisms in our own files; never package or submit any of its bytes; keep `docs/PROVENANCE_504aldo.md` current with a timestamped timeline.
2. Process (handoff 09-14 §0): local-seconds residual gate is not a decision gate (op-count ≤22,000 at 13.4 µs/op is); relative tolerances only (f32 vs f64 oracle, 1e-4 rel); single-use DECIDE blocks only for candidates with fitted constants; Skeptic only for numbers headed to the write-up and for designated files; one fresh-truth score per candidate per day is the minimum cadence.
3. Every submission needs Samuel's explicit per-submission approval; you never designate. External messages (forum/email) need his approval of the exact wording.
4. Predictions: record Samuel's answer verbatim before each STRATEGIC experiment; "I don't know" is valid. The strategist's pre-registered numbers are in §6 of the 09-16 response — copy them into each contract as "strategist prediction", separate from Samuel's.

## Work list, in order. Do not reorder; do not start the next item while a gate is unresolved.

### 0. Today
a. Finish v31's evidence path: certification with §9 memory present, `local_sim/predict_public.py --candidate estimator_k3_v31.py --control 330950` on the finished mini pair. Report: paired ratio ± t-CI, projected public score, multiplier {F, F+2²⁴}/2⁴¹, op count → residual forecast, wall projection ×1.60, memory verdict. If all green and projected wall ×1.6 < 120 s: **request approval to submit v31 as a non-designated canary**. Log graded id, raw, multiplier, worst wall, worst residual when it grades.
b. Mint a fresh 24-network DESIGN block **P11** (canonical mint, 2×2²⁴ truth replicas as for P10). P01 has carried F21, the λ sweep, P1 and P2; use P11 for every accuracy read in this slice, P01 only for cost/parity.
c. Hand Samuel the rules question from `PROVENANCE_504aldo.md` with the P2 timeline attached, for approval and posting. Do not post it yourself.

### 1. Close the κ4 half (accuracy) — target 2–3 days
Gate for the slice: v31-fixed raw on P11 ≤ 0.67 × v31 raw on P11 (≥1.5×), paired t-CI excluding zero, cost unchanged within 0.5%. Strategist prediction: raw lands 2.2–2.4e-8 P01-scale-equivalent (55%).

**D1 — λ-table diagnostic (≤2 h).** Build a *diagnostic-only* arm of v31 with 504aldo's `LAM` table (their `estimator_v17.py` lines ~229–232, 0.95×mean, and their adaptive rule if trivially portable) substituted for ours; run on the 24 P11 nets against v31. This arm is never packaged, never submitted, and is labelled DIAGNOSTIC in LOG. Strategist prediction: moves raw by >15% (40%). If it does → the gap is our λ depth profile → fit our own per-layer λ end to end on P11 (coordinate descent over 4 layer groups, then per layer; report the profile next to theirs; fitted constants ⇒ this candidate needs a DECIDE block on a *different* fresh block before submission). If it does not → proceed to D2.

**D2 — trace comparison (≤½ day).** On one 1024×16 network, dump per layer from both `external/504aldo/…/estimator_v17.py` (run under the numpy shim pattern in `experiments/2026-09-16_exp_q31/`, gate forced open) and from v31: `dG`, `g_prev`, `var_pre`, `var_post`, `λ`, the (4) and (2,2) consumer inputs, and the κ4 inputs to the (3,) and (2,1) births. Compare layer by layer, relative. Localise the first ≥1.3× discrepancy and name the line. Strategist prediction: a ≥1.3× discrepancy exists in `dG` or the births (60%). Known prior bug in this port: `metric`/`init_scale` applied a second time in transport; check the trace/`c4` split for double counting.

**D3 — feed coefficient check (≤½ day, after D2).** On 8 P11 nets, zero the κ4 inputs to the (3,)/(2,1) births in v31 and measure the raw change. F68 predicts a large loss (their feed is worth ~4× at this shape). A small loss means our table-routed feed is not their feed at the right coefficient; derive and fix, with a tiny-shape f64 oracle.

Fix what D1–D3 locate, one mechanism per commit, tiny-shape parity (1e-4 rel vs f64) at each step, then the P11 screen against the gate. Canary on pass (approval). **Do not add the (3,1) table rows in this slice** — measured 0–9%; revisit only if the gate passes and raw is still >2.2e-8.

### 2. P3 — rank-16 D21 feedback into births (1 day)
`D21 ≈ Q_f B_f` (two QR passes, equivariant sketch — reuse `_p2_compress`'s column-norm selection, not `W[:, :r]`), `X_t = 1.5·d(w2)·D21 = F1 R1`, `Y_t = 0.5·d(w1)·D21ᵀ·d(w3) = F2 R2`, births `X1 = 3a_b + X_t`, `Y1 = a_b·d(w2) + Y_t`, F1/F2 on the thin stack. Gate: ≥1.06× raw on P11, cost ≤ +1%. Prediction: ×0.92 raw (504aldo F69). Canary on pass.

### 3. P4 — cost mechanics (2 days)
Strassen depth 5, min block 32, all products `out=` into persistent pooled buffers on the estimator instance (no `concatenate` in the hot loop; `gc.disable()` inside `predict` only if memory gate still passes), block-symmetric `C_pre` via three block products of the 2×2 partition, last layer `var = rowsum((WC)∘W)` + D3 only. Gate: C/B ≤ 0.27 at raw parity (dense-vs-dense identical to 1e-6 rel), ops ≤ 20,000, Linux memory gate pass, no `auto-routed` warnings. Canary on pass. Expected standing ≈ 5.5–6e-9.

### 4. N1 — sensitivity-weighted old-lineage basis (the research bet; 4–5 days)
**Kill test first (2 days), on our chain, 8 P11 nets, lineages older than 4 transports, ranks r ∈ {96,128,192,256,384}:** compare raw vs r for (a) the forward-energy Gram (504aldo F72: `G = A·d(dA)·Aᵀ + P·d(dP)·Pᵀ`) and (b) the same Gram weighted by JᵀJ where J = Π_{k>l} D_k W_k, D_k = diag(Φ(α_k)) from the K2 chain — compute by one transposed transport per layer from the output (or its top-128 right singular vectors), total ≤0.03 B. Pre-registered gate: (b) reaches (a)'s r=384 accuracy at r ≤ 192. Strategist prediction: 35%. Record Samuel's prediction first.
If it passes: merge all old lineages into one Tucker core in that basis (cost r³/n² units/layer, flat in lineage count), rank chosen per MLP from the core spectrum; transport by the r×r restricted propagator; read the (2,1) slice once per layer. Gate: C/B ≤ 0.20 at ≤ +5% raw. Canary. If it fails: implement P5 as published (age gates 4/7, ranks 384/224) → ~0.25 B, and write the negative result up properly (rank law for both bases).

### 5. If time remains (days 11–13)
Puffi-style λ refit target (fit λ to the *consumed mean correction* rather than to G_off) on P11 as a one-day accuracy probe; per-MLP adaptive rank from the join core spectrum; Strassen L6 only if ops stay ≤20k.

### Throughout
- Claims ledger `docs/CLAIMS_LEDGER.md`: one line per measured number headed for the write-up (what, where, block, date). ≤1 h/day. No write-up drafting before Oct 3.
- Every graded row: id, raw, multiplier, worst wall, worst residual, ops, into `experiments/submissions.jsonl` and LOG.
- Hand back a strategist packet when: the κ4 half is closed (or D1–D3 fail to locate it), or N1's kill test has a verdict, or anything contradicts the 09-16 response. Include the P11 table for every arm, the D2 trace summary, and the current board read.

## Constraints (unchanged)
Python-only, flopscope + stdlib + whestbench at eval; shape-generic and deterministic; no fitting on the public split; no randomness beyond the provided seed; heavy jobs through workerq with measured reservations; never edit frozen files or the shipped `estimator.py`; F21 remains the designated fallback until v31 has a clean graded row.
