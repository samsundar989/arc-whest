# Strategist response — 2026-09-25: packet answers, verified facts, and the hand-over to PLAN rev 13

*Answers `docs/STRATEGIST_PACKET_2026-09-25.md`. The plan is `docs/PLAN.md` rev 13; the executable slice is `docs/HANDOFF_2026-09-25.md`. Labels: **[M]** measured / primary source read today, **[I]** inference.*

## 0. Review of the slice
Step 2 (in-loop corrector) is the best result this project has produced and the sequential-vs-single-pass finding (6.2 pp from exposure bias alone) is a real contribution — keep it for the write-up. Step 3 (adaptive age) and the recursive feature were killed cleanly. The anatomy, the wall law and the label-noise correction are all sound. My predictions for the slice were optimistic again on the closure side and the worker's were right; priors adjusted in rev 13.

## 1. Q1 — the 3% gate
**Dropped.** Samuel's decision today: submit any candidate whose projected score beats the best graded score; the hard gates (validate, ops ≤ 26,000, residual ≤ 0.33 s, WSL 8 GiB, setup < 2 s, 0 materializations, 0 failed nets) stay. Submit `estimator_k3_v59_corr400.py` today. The gate was written against a stale baseline; measured against our own fresh best it only guarantees that we never ship our best score.

## 2. Q2 — how two designated entries combine
**Resolved from the primary source.** Live Rules v14 §7, read in Chrome 2026-09-25 [M]: *"Where a Participant has more than one submission eligible for score-based prize consideration in a Phase, only that Participant's best-scoring such submission is ranked, so that each Participant occupies at most one position in that Phase's score-based ranking."* A net that times out receives the zero-prediction fallback (§5.5) — the entry stays valid, it just scores badly. So **the better of the two designates counts, and the aggressive one is a free option.** Policy: **A = best score (leaf 32 permitted, implied noisy worst ≤ ~115 s), B = wall-safe twin (≤ 95 s).** The Sep 10 source ledger (`docs/research/2026-09-10_competition_sources.md`) already recorded this sentence; nobody connected it to the packet. Forum: never answered there (/t/18143), so the rules page is the only source — quote it in `docs/DESIGNATION.md`. Samuel has not chosen to email the organizers; if he wants written confirmation, that is his call.

## 3. What changed on the board since the packet [M]
Luna (#5, 2.60e-9, raw 1.67e-8, 0.155 B, walls 40–47 s, all-layers 6.80e-9, per-net FLOPs ±6%, 7 entries since Sep 24). mliston 3.05 → 2.67e-9 in 9 h (Sep 24–25), raw 1.76e-8 at 0.152 B, walls 65 s. J2W #332199: 0.127 B, walls 104–117 s, 0 failures this time. jamespayor #332145: raw 1.56e-8 at 0.77 B. Insights: best raw 1.47e-8. Our #332223 telemetry: 0.232 B flat across nets, walls 52.3–57.6 s, all-layers 8.80e-9, growth L0→L15 33.5×, steepest L3 (1.74×). Per-net hardness matches J2W's ordering (amy-young worst for both; our/their raw ratio 1.34–1.52 across nets) — the gap is uniform, not a tail.

## 4. The packet's §4 flags
- **Per-net cost variation:** still unexplained; the leaders' ±6% is now also Luna's. No mechanism proposed here either; parked.
- **No noisy-day reading of the wall-safe twin:** still true. Under the best-of-two rule this matters less for A (B covers a failure) but B itself must be measured on a noisy day before Oct 10 — the daily A/B slot exists for that; take it opportunistically when a same-day row shows worst/median ≥ 1.15.

## 5. What this review adds (the case for rev 13's ordering)
1. The gap to #1 is two independent 1.4–1.7× gaps (accuracy and cost) and the leaders are more accurate at *every* layer, not only the last. Cost alone at the floor is ≈ #4; accuracy alone ≈ #12.
2. Our exact T0 chain (≈2.27e-8) is worse than 504aldo's *compressed* graded chain (2.13e-8): the closure/term set we run is leaving accuracy on the table before any new idea.
3. Exact per-layer pre-activation labels exist (keenanpepper full1000-N1e9: dense E[z_i z_j], E[z_i² z_j], E[z_i³ z_j], E[z_i² z_j²] + marginals to 4th order, N = 1e9, all 1000 full nets; bakev2 mini-supp: pre/post moments to 6th order + gate probabilities + dense post K21/K22 on our DEV nets). They let us measure *which carried object drifts* — the diagnosis this project has never had — and they are organizer-sanctioned training data.
4. End-to-end fitting through the chain has three independent pieces of evidence behind it (organizers' description of the warm-up winners; jamesrahenry's "step-local accuracy does not compose"; our own 79%-propagated-drift anatomy) and has never been built; the greedy corrector's −7% is its lower bound.
5. The multi-mode κ4 core (Puffi's SSC) and a σ-channel corrector are the two closure changes with primary-source evidence; the harmonic projection-at-fold old tier is the one cost idea not yet measured.

Odds and the stop rule are in `docs/PLAN.md` §3 and §6.
