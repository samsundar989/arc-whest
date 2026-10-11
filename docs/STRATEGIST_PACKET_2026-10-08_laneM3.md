# Strategist packet — Lane M3 (2026-10-08)

Full report: `docs/research/2026-10-08_laneM3_report.md`. Raw: `experiments/2026-10-08_laneM3/`.

**M8 (2026-10-08) — κ4 is not a memory; old κ3 is compressible.** ARC b+E chain, 16 DEV nets,
n = 1024, organizer labels; old κ3 blocks dropped by age with their corrections split by tag
(exact, calibrated; A = ∞ reproduces b+E 1.583e-8). Dropping costs ×16.9 [15.8, 18.1] at A = 3,
×6.7 at A = 5, ×2.1 at A = 8 (16/16 each) — the handoff's GO bar (A = 3 ≤ 1.3×) fails by 13×. At
short memory b+E = SIMPLE (a / b+E 1.01–1.05): the core carries no old-κ3 content. **But folding
the old blocks into a shared rank-256 basis (re-based per layer, energy-ordered) with 5 exact recent
layers costs ×1.09 [1.06, 1.12]** and keeps AUGMENT's 2.4× edge over SIMPLE; rank 128 costs ×2.3.
For Lane P's Oct 10 decision: a complete-K3 chain can be cheap via our own 5-exact + rank-256 old
tier, not via the κ4 core; the fold's rank is the cost driver. No GO form for the core route.

**M9 (2026-10-08) — fourth order is worth ~7–9× at n = 1024.** ARC's sweep, 8 nets per width,
paired: K=4 SIMPLE / unfactored K=3 AUGMENT = 0.56 / 0.50 / 0.43 / 0.36 / 0.28 at n = 96 / 128 /
192 / 256 / 320 (8/8 at n ≥ 128); the ratio falls faster with width (log-slope −0.6 to −0.8) →
0.11–0.15 at 1024. Per-method slope extrapolation is 8× off at 1024 and was not used. K=3 at 256
matches the Oct 6 table. K=4 AUGMENT (factored) is unstable at depth 16.

**M10 (2026-10-08, final) — no cheap piece; NO-GO for Lane P; direction needed.** Birth-class and
correction-rank ablations of K=4 SIMPLE at n = 96–320 (8 nets each). At n ≤ 128 everything looked
cheap (corrections alone = all of the gain; rank-2 corrections better than exact). Both are
pre-asymptotic artefacts: by n = 320 corrections-only plateaus at 0.66 of K3 (exact K4 0.28),
rank-2/4 corrections are ×2.0/×1.5 worse than exact, and on top of the corrections no single birth
class adds a positive share (path +0.1, star −0.2 to −0.4, star+path −0.1 to −0.2, 3+2 ±). The
large-width gain needs full-rank repeated-slice transport and all birth classes together — n⁴-class
as far as we know. M10.2 (star/path shared-factor readouts) cannot meet the ≥ 40 % at ≤ +10 % F
rule and was not built. Open question for the strategist: is there any O(n³) representation of κ4
that tracks the asymptotic gain? I have no candidate; otherwise fourth order is write-up material
(the 7–9× headroom number and the ablation map).
