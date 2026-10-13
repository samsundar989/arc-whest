# HANDOFF — Lane O (session 2, branch `lane-m`, follows M6), 2026-10-12: the old tier and the spend items

**Read first:** `docs/research/2026-10-12_deep_review.md` §0, §3, §5.3, §6, §7. Prices: `docs/research/2026-10-12_costfloor/x3_tables.txt` (every ages × old-tier × glue combination is priced exactly).

**Setup.** Merge main into lane-m. Work in the twin on the v84 arm (R8 parameters). Hand winners to Lane P as patch + twin parity + VAL-100 (unrefit; refit vs matched control) + interpreter F and ops. GPU: local 5090 through gpuq; pod 2 shared with Lane K (K's stack refit has priority on Oct 13–14).

**Standing rules.** As in M6: predictions on record; VAL-100 selection, FRESH-A acceptance by Lane P; matched controls; no external messages; no write-up work.

## Why
The rank-256 old tier costs 39 u = 0.0385 B — as much as three exact ages — and pruning barely shrinks it. With Lanes S and K landing, the chain is ≈ 0.126–0.138 B with the old tier and 0.096–0.116 B without it. The leaders at 0.105–0.116 B are in the second band. Whatever removes or halves the old tier at ≤ +5 % raw is worth 5–10 ranks; nothing else on the table is.

## O1 — exact-SVD join, then smaller ranks (Oct 13)
The pending M6.2 screens, now first in line. flopscope bills a top-k SVD at 4mnk while computing the exact SVD (documented in `_svd.py`).

| arm | ΔF vs v84 | break-even Δraw | history (1-pass basis) |
|---|---:|---:|---|
| SVD join, rank 256 | +1.1 % | −1.1 % | M6: −1.48 % raw unrefit |
| SVD join, rank 224 | ≈ −1.5 % | +1.5 % | +6.8 % (v67) |
| SVD join, rank 192 | ≈ −3.5 % | +3.6 % | +21 % (v67) |
| SVD join, rank 128 | ≈ −8 % | +8.7 % | untested |

Unrefit `twin.py eval --raw` on VAL-100 for each; refit only the arms inside 1.5× their break-even. First run the diagnostic: fraction of the top-256 energy of G = J·D·Jᵀ + Q·Sg·Qᵀ that the 1-pass basis captures, per join layer, on 2 FIT nets.
**Prediction (strategist):** rank 224 with SVD +2 to +5 % unrefit; rank 192 +10 to +18 %. P(any smaller rank passes) ≈ 0.25.

## O2 — separate ranks for the two legs (Oct 13)
The shared basis serves A_L and M together. Screen rank_A / rank_M ∈ {256/128, 256/192, 192/256, 128/256} with one basis per leg (the join's weighting unchanged). Price each with the interpreter before running (the rebuild and the readout split per leg).
**Prediction:** one leg tolerates 192 at ≤ +3 %; P ≈ 0.3. This is new: no earlier screen separated the legs.

## O3 — more exact ages, no old tier, C-mode tail (Oct 13–14)
Arms: exact ages A = 7, 8, 9 with content older than A (i) dropped, (ii) replaced by the harmonic tier of Sept 26–27 (Sym(u⊗C) + carried vector; the code exists) retired at age A. Prices on the v84 stack: no-old A = 7 / 8 / 9 → F/B 0.149 / 0.161 / 0.172; on the K-pruned stack with Strassen v2 they are ≈ 0.119 / 0.129 / 0.138 against 0.138 for age 6 + old tier.
- History: the harmonic tier at age 5 was ×5.9–6.6; tonight's labels explain it (the non-gain residual of D21 is 16–23 % of its energy, flat spectrum). The open question is only how fast that residual decays with age.
- **Kill:** A = 8 (ii) worse than +10 % unrefit ⇒ close the family for good.
**Prediction:** A = 8 (ii) +15 to +40 % unrefit. P(passes) ≈ 0.1. It is here because it is the one cheap test of the "old content at near-zero cost" reading of the leaders' bills.

## O4 — spend items on the pruned stack (Oct 14–15, after Lane K's first stack)
Re-price with `x3lib.py` on K's stack and screen by the marginal rule: age 7 (on C1 + glue: +4.0 % F), J = 3/4, SVD join at rank 256 (+1.1 %). Each exact age gets cheaper as K prunes, so the age-6 break-even (−5.8 % raw for +5.6 % F) moves in our favour.
**Prediction:** age 7 −3 to −4.5 % raw for ≈ +3.5 % F on the pruned stack: marginal; take it only if A's wall allows.

## O5 — anatomy on v84 (time-boxed half day, Oct 13)
Rerun `tools/anatomy2.py` (KAS off) on the v84 arm: var / C_off / D3 / D21 / κ4 / old-tier-exact / all-but-μ, full and by band. Age 6 removed ~6 % raw since the last anatomy; we need to know which channel it came out of and whether the covariance at layers 12–15 is still 19 points.

## Do not
- Full K = 4 transport, κ4-core loop, coherent κ3 tiers below age 7, learned pair closures, low-rank D21 readouts, cubature, sampling hybrids.
- Exact gain-mixture closure: estimated at a few percent at most (review §7); not this week.

## Report
A daily paragraph in `docs/STRATEGIST_PACKET_2026-10-1X_laneO.md`; tables under `experiments/2026-10-13_o/`.
