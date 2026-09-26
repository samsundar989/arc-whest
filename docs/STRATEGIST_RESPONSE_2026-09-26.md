# Strategist response — 2026-09-26/27: the packet's decisions, and why the next slice is two representation experiments, not more fitting

*Answers `docs/STRATEGIST_PACKET_2026-09-26.md`. The plan is `docs/PLAN.md` rev 14; the executable slice is `docs/HANDOFF_2026-09-27.md`.*

## 0. Review
Rev 13 was executed completely and well: the twin (accepted at +0.006%), seven graded steps to 4.7039e-9, every arm priced with predictions on record, the Winograd wall finding, the noisy-day B reading, and Anatomy 2 — which is the most informative measurement this project has made. My predictions were wrong on the anatomy (cov share, D21 share, the 4-mode κ4 model) in the direction of overestimating κ3/κ4 structure and underestimating the covariance channel; the worker's were closer. Adjusted below.

## 1. Decision 1 — direction for Sep 27 – Oct 8
Not (b) alone. The twin's stages paid 0.3–2% each and the plateau is real; more capacity on the same parametrisation buys the 1–3% the packet estimates, which does not change the rank. The evidence points at two *representation* experiments, both testable in days:
- **Cost: the harmonic old tier** (PLAN §3 Step 1). The packet's (a) says "no concrete design is in hand". There is one: ARC's harmonic split of κ3 into Sym(u ⊗ C) + traceless. It was rev-12 H1, struck by inference and never run. It is the only candidate that (i) has a mechanism, (ii) would explain the leaders' bill (≈ ours minus the old tier), and (iii) matches ARC's stated direction. One-day test.
- **Accuracy: closure calibration** (PLAN §3 Steps 2–3). The anatomy's linear-in-depth growth is a closure-bias signature, and we hold both sides of the closure equation on 100 nets. First the refit ceilings (substitute-and-refit; substitution alone is misleading in a compensating chain — "exact κ4 makes it worse" proves it), then calibrate the closure on the channel with the largest ceiling, then fine-tune through the twin.
Then (c) from Oct 8 regardless.

## 2. Decision 2 — stop rule
Reset: decision point **Oct 5** on graded evidence; write-up starts **Oct 8** whatever the score; hardening Oct 12–16. Score work continues to Oct 15 only with a ≥ 10% jump in hand.

## 3. Decision 3 — team freeze
Noted; nothing to do.

## 4. Flags
- Machine contention: queue everything through workerq/gpuq; BelowNormal for other projects' jobs during this slice.
- Memory: the Winograd temporaries fix stands; keep the two-sided settle check but record allocator shrinkage as PASS-with-note.
- Wall law: refit it with the Winograd point as a glue-traffic term (ops × bytes), not ops alone; it will change the leaf-16 verdict.
- Board: Luna (raw 1.54e-8, 0.139 B, 55 s walls) is the team to model; J2W at 0.119 B.
