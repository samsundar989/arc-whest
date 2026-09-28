# Claude Code handoff — execute PLAN rev17

Run this prompt from `C:\Users\samsu\Documents\arc-whest`. This handoff delegates the
canonical plan; `docs/PLAN.md` wins over any difference. No compact-history candidate
has yet been implemented, trained, evaluated or submitted.

---

You are the implementation worker for my ARC White-Box Estimation Challenge entry.
Work in `C:\Users\samsu\Documents\arc-whest`. Execute `docs/PLAN.md` rev17, especially
§4 B5 stages E0–E6, through our next genuinely better graded submission if the evidence
supports one. Implement and test the plan; do not stop at another proposal. Target a
20–30% lower competition score, with 15% as the structural research milestone. Lower
is better. These are objectives, not results to assume or manufacture.

Read in order: PLAN, AGENTS/CLAUDE, the latest RESULTS entries, CURRENT_STATE, then
`docs/research/2026-09-28_review_and_next_step.md` and
`experiments/2026-09-28_cost_sites/report.md`. PLAN rev17 supersedes older handoff
instructions. Preserve existing dirty work, checkpoints and logs. Use bounded agents
for disjoint algebra, implementation and measurement tasks; one integrator owns the
carrier/layer integration and canonical documents. No per-experiment Skeptic process.

Our verified incumbent is `estimator_k3_v72_gt.py`, SHA256
`3b6cc39790295d3be60a88f9dd9053915f695bdf0a0da2cf2e99587cce692bac`, submission332766,
score 4.620737758949154e-9. It costs 0.214733B; recent-history transport/readout consumes
61.373% of its counted FLOPs and resolved old history another 27.086%. The Sep 28
observed leader scored 1.748e-9. Refresh these anchors only if newer verified evidence
exists. The worker's incremental gains were real; the tested covariance-core residual
correction and small Gaussian triangle failed. Do not rerun those unchanged or train
a generic larger closure network.

The new hypothesis is to replace expensive carried history with a compact state trained
for the final prediction. Follow E0–E2 before substantial fitting:

- Reproduce the actual v72 teacher, including S2t3 step736 and G1tg step84. Verify the
  twin/export path; older v68/v71 diagnostic initialization is not this teacher.
- Implement exactly age3/q32 and age3/q64. Keep three young source lineages after each
  nonfinal birth and fold expired history into one symmetric core G plus thin basis U.
  Remove obsolete source arrays only after auditing every consumer. Include all source
  terms: D-block, two-leg/separable terms, P2 residual and active hubs.
- Derive transport, QR/core rotation, rebasing and readout from our own tensor algebra.
  Use the single label-free history-factor Gram basis specified in E1; freeze its
  algorithm for both capacities. Meter its entire construction. No inference teacher,
  per-network learned core, dense width-cubed tensor or uninstrumented math.
- Crucially, a lossy fold changes inherited slices. Compute closure from actual incoming
  state, transport/fold, then compensate the birth using the tensor actually retained.
  Prove full-P2 tiny identities and measure production rank-4 approximation separately.
- Pass E1's dense-reference, no-change, symmetry, gauge, determinism, shape and finite
  checks. Price the complete immutable-array shipping code with real flopscope before
  training. Target ≤0.12B and apply E2's explicit price/signal admission rules.

Then execute E3's bounded controlled pilot: TRAIN rows 0–79, exactly 60 optimizer steps with
accumulation 8, frozen v72 teacher means, final-label loss plus the specified early
teacher stabilization, identical data order/schedule for controls. Keep covariance,
closure and existing corrector fixed initially. Compare analytic students, learned
shared updates and continued v72. Implement the new training path explicitly; the
current twin CLI does not already support this carrier/loss. Verify initialization,
gradients, export parity and checkpoint/resume. Never fit DEV or VAL. VAL is full rows 900–999;
it is repeatedly used validation, not an untouched test.

Select by paired final MSE×max(0.1,F/2^41) using real counted cost, with intervals,
wins and tail diagnostics. A raw-error increase can be worthwhile: at 0.12B, about 43%
more error than v72 still gives 20% lower score. Full-VAL evidence for ≥15% improvement
and a win over the continued control admits E4's longer TRAIN 900 run. Expand only the
winner, using the bounded schedules and stopping rules in PLAN. An already qualifying
analytic student can advance directly. Do not optimize internal tensor energy as a
surrogate for final score or claim an oracle fit demonstrates learnability.

If an arm fails, use E5 to distinguish algebra, price, information retention and update
learning. Allow only the prescribed bounded diagnostic/correction, record the negative,
and change one mechanism. If both cores fail, investigate the specified recent-factor
compression fallback with an actual shipping cost/retention screen before another fit.
Do not silently widen ranks, bases, gates or checkpoint searches. Keep Oct5's strategy
decision and Oct8's write-up start. Report a precise blocker if evidence cannot support
a better submission; preserve the best valid incumbent.

Before each heavy research/training job, write the required optimization/resource review
and use workerq with conservative declared RAM/VRAM/CPUs. Use the one-network smoke to
measure peaks, stream data, cache compact teacher means and avoid concurrent teacher
autograd or a dense D21 corpus. Prefer 16–24 GiB host RAM when justified. Ordinary
candidate evaluation uses `tools/ev.py` directly, initially one process. Torch/shim
counts are never official cost; local time is never server wall.

For the selected export, complete DEV 100 and every PLAN §5 gate on the exact source hash,
including fresh WSL 8 GiB evidence, parity, real cost, op/residual limits, manual noisy-wall
projection, package validation/hash and the daily ledger. Standing approval permits
at most 5 non-designated canaries per UTC day when all gates pass. Submit an eligible
candidate, watch grading, inspect per-network failures/score/cost/wall, and update
RESULTS, CURRENT_STATE and calibration. Never designate final entries, contact organizers,
post on the forum, bypass guards, or write HUMAN_MODEL. If a hook actually blocks an
authorized canary, explain the exact action and reason rather than bypassing it.

Keep small descriptive commits limited to your changes. Report each completed stage,
graded submission, twice-failed gate or decision need. Finish with the actual graded
improvement (or exact remaining blocker), candidate/package hashes, reproducible artifact
paths, and the next evidence-backed step. Keep targets, local validation and graded
results clearly separate.
