# HANDOFF — Lane P7 (session 1, branch `main`), 2026-10-12: integrate S / K / O, refits, canaries, hardening

**Read first:** `docs/research/2026-10-12_deep_review.md` §0, §5, §6, §8. This handoff follows P6 and answers the three open items of `docs/STRATEGIST_PACKET_2026-10-12.md`.

**Ownership.** Unchanged: you own main, the estimator files, RESULTS, the ledger, `docs/DESIGNATION.md` (recommendations only; Samuel designates), CURRENT_STATE, pod 1's queue. Lanes S, K and O deliver patch atoms; you integrate, gate and submit.

## Answers to the Oct 12 packet
1. **Two-column-set H: GO**, built by Lane K (K3). Exact price −3.13 % F at τ 2.5 (`docs/research/2026-10-12_costfloor/`, scenario S2b); measured +1.22 % unrefit.
2. **A's wall margin:** spend nothing on it yet. Strassen v2 (Lane S) is predicted to *cut* wall by ~15–20 s at today's depths; re-decide od / age 7 / SVD join on the v85 canary's telemetry.
3. **P6.5 adaptive keep counts:** moved to Lane K (K5), optional, after K0–K3.

## Standing rules (changes in bold)
- PLAN rev 18 §5 hard gates stand: wall ≤ 105 s (A) / ≤ 95 s (B) noisy-implied; setup < 2 s; WSL 8 GiB PASS; parity ≤ 1e-3; hash = tested source.
- **Ops:** ≤ 26,000 until v85 is graded; then **projected residual ≤ 0.33 s at the per-op rate measured on v85's server telemetry** (Samuel's decision 3).
- **`out=` is allowed with an output of exactly the broadcast operand shape** (Samuel's decision 1). `where=` and symmetry tags stay banned.
- ≤ 5 canaries per UTC day + 1 A/B slot. FRESH-A paired for anything not bit-identical. FRESH-LOCK ≤ 3 reads, designation only. Predictions on record. No external messages. No write-up work.

## P7.0 — housekeeping (Oct 13, first hour)
a. Commit the strategist's files unchanged, one commit: `docs/research/2026-10-12_deep_review.md`, `docs/research/2026-10-12_strassen/`, `docs/research/2026-10-12_gain/`, `docs/research/2026-10-12_costfloor/`, and the four handoffs `docs/HANDOFF_2026-10-12_LANE_{S,K,O,P7}.md`. Mark P6/M6 complete and P7/S/K/O active in CURRENT_STATE.
b. A/B slot: submit `v83_r7ws` (packaged). B stays age-free until a B twin of the Strassen-v2 chain exists.
c. `tools/build_v85.py`: v84's builder plus slots for Lane S atoms (`sb2`, `depth16`, `depth8`, `sbks`), Lane K atoms (`k0`…`k4`) and Lane O atoms; '+'-joined as in `build_v83.py`.
d. Start pod 2 for Lanes K and O.

## P7.1 — v85 = v84 + `sb2` (Oct 13)
Bit-identical ⇒ no FRESH-A. Gates: Lane S's five checks reproduced on main (bit-identity on DEV 0–4, meter 0.16533 / 24,576 ops, WSL memory, wall A/B, `out=` shape audit). Canary. From its telemetry record: server wall mean/max, residual mean/max and per op, kernel seconds.
**Prediction (strategist):** graded 2.713e-9 ± 0.2 %; wall mean 62–70 s; residual max ≤ 0.27 s.

## P7.2 — v86 = v85 + depth (Oct 13–14)
Lane S2's leaf ≥ 16, then leaf ≥ 8 (or per-site depth) under the new residual rule. DEV-100 raw within ±0.3 %; FRESH-A paired (rounding changes). **Prediction:** F/B 0.156–0.160, raw +0.1 to +0.5 %, graded 2.57–2.64e-9.

## P7.3 — v87 = v86 + Lane K's stack (Oct 14–15)
Integrate K0–K3 as they arrive; joint refit R9 vs matched control on pod 1; FRESH-A; canary. Then K4 if it passed its screen. Recompute ρ (`compute_rho.py`) and the KAS tables on the agreed Strassen grid (multiples of 64 with Lane S4).
**Prediction:** F/B 0.136–0.142, raw 1.64–1.70e-8, graded 2.25–2.40e-9.

## P7.4 — v88 (Oct 15–16)
Variant C (`sbks`), tables on 64, Lane O winners (smaller old-tier rank, age 7 or SVD join by the marginal rule), refit polish (stop after two restarts < 0.3 %). **Prediction:** 2.0–2.2e-9.

## P7.5 — B and hardening (Oct 15–17)
- B = the block-40 twin of the best A with every margin ≥ 12 %: wall ≤ 95 s, residual ≤ 0.30 s, VmPeak ≤ 7,800 MiB at 250 predicts in one process. Use the A/B slot daily.
- FRESH-LOCK (≤ 3 reads) on the A and B finalists; `docs/DESIGNATION.md` recommendation; **freeze A at Oct 17 12:00 UTC**, keep one slot for B.
- Watch the Submissions page for the selection UI (`viewerCanSelect`) and the forum for an organizer post; flag either immediately.

## Canary budget
5 per day: one per integration step, never two builds that differ by more than one atom family unless both were FRESH-A'd. A canary graded > 2 % worse than projected stops the line.

## Report
Daily packet `docs/STRATEGIST_PACKET_2026-10-1X.md` with the ledger vs v84 (ΔF, Δraw, ops, wall, residual per build) and both predictions per row.
