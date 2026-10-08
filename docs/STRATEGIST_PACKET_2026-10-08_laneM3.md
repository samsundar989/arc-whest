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

**M9 (2026-10-08) — fourth order is worth ~3–5× at n = 1024.** ARC's sweep, 8 nets per width,
paired: K=4 SIMPLE / unfactored K=3 AUGMENT = 0.56 / 0.50 / 0.43 / 0.36 at n = 96 / 128 / 192 / 256
(8/8 at n ≥ 128). Ratio trend → 0.18–0.36 at 1024 (2.8–5.5× headroom over unfactored K=3, ~3.5–7×
over ARC's factored K=3 level). Per-method slope extrapolation is 8× off at 1024 and was not used.
K=3 at 256 matches the Oct 6 table (1.10e-6 vs 1.21e-6 unfactored). K=4 AUGMENT (factored) is
unstable at depth 16. n = 320 point pending.

**M10 (2026-10-08) — the gain is not in star/path; direction needed.** Birth-class ablation of
K=4 SIMPLE (n = 96/128, 8 nets): with *no* top-slice births K=4 keeps all its gain (none/full 1.01
[0.67, 1.53]; none/K3 0.50, 8/8); star and star+path are worse than none; 3+2 alone ×2.5 worse;
no κ4 memory ×2.3 worse. The gain is the exact transport of the repeated-index κ4 blocks (K=3
AUGMENT keeps only their r = 1 trace), which ARC stores as an n × n matrix per column (~2 B per
block-layer). No shared-factor arm can meet the ≥ 40 % at ≤ +10 % F rule, so M10.2 was not built
and no GO goes to Lane P. Proposed next questions (minutes each on the M9 cache): the rank of the
transported correction blocks; which slice of them carries it ((2,1,1) feedback vs (2,2)/(3,1)/(4));
a gain-mode template for that content. Strategist's call whether any of this is worth the last week.
