# GOTCHAS.md — standing operational knowledge

*Extracted verbatim from `CLAUDE.md` on 2026-08-07 when the top-level agent file
was slimmed to a map. Nothing here was rewritten: these are the findings that must
not be rediscovered, in the words they were recorded in. Newest sections first;
sections marked "historical" were superseded by the ones above them and are kept
because the correction is itself the lesson.*

**Read this before**: touching `estimator.py`, quoting a number, comparing a cost,
reopening a closed lane, or trusting an offline measurement path.

Companion docs: `docs/VALIDATION.md` (how anything gets measured — the protocol),
`docs/CURRENT_STATE.md` (what is true right now), `experiments/LOG.md` (the full
narrative record), `docs/GAME_PLAN.md` (strategy, rev 12 is authoritative).

---

## 2026-08-21 — the Phase-2 shape moved, and what that costs a shape-specific result

*Full record: `experiments/LOG.md`, 2026-08-21.*

- **Phase 2 is 1024 wide x 16 deep.** Read from the HF dataset's own
  `metadata.json` at the **`v2-phase2`** tag, which appeared the day the round
  opened and was absent that morning. **Poll the dataset TAGS, not only the round
  list** — `local_sim/phase2_checklist.py` already prints them and that is where
  the shape surfaced first, before any forum post or email.

- **A working assumption is worth recording precisely so it can be falsified
  cleanly.** Samuel's "Phase 1 with different MLPs, maybe a little harder" was
  written down on 2026-08-20 **with its falsifiers listed in order of damage**.
  The first one — "a width that is not 256" — is what happened, less than 24 hours
  later. The assumption cost nothing because the falsifier was named in advance;
  the entry is kept rather than deleted.

- **`2^8 = 256` was doing more work than it looked.** Three separate results were
  width-256 objects, not general methods: the Kerdock/MUB-129 carrier
  (`256 = 4^4` makes the real-MUB bound `d/2+1 = 129` attainable; `GR(4,7)` because
  `2^(7+1) = 256`; the axial identity cancels because the sign set has exactly
  `d^2 = 2^16` points), the dual-BCH strength-5 array, and the 131,072 linear-floor
  proof via the Sidon-set obstruction. **When a construction's size is a power of
  the dimension, expect the whole thing to be dimension-specific**, and say so in
  the entry that records it.

- **The mathematics generalises even when the object does not.** At `1024 = 4^5`
  the analogue is `GR(4,9)`, 513 bases, and `2^20 + 2048 = 1,050,624` points; the
  axial identity `[(3-2S) + 2S]/(d^2 + 2d) = 3/(d(d+2))` is dimension-free. What
  changes is entirely the **cost**, which is why the derivation was written into
  `local_sim/carriers.py` rather than a blob.

- **Per-sample cost does not scale with parameter count.** 1024x16 has 4x the
  parameters of 256x32, but a dense layer costs `O(width^2)` per sample and there
  are half as many, so propagation is **~8x** per sample, not 4x. **Check the
  budget `B` before assuming the operating point survives.**

- **A shape change invalidates seed blocks, not just constants.** D16's 330 free
  seeds, D18's remainder, and the untouched single-use D17 and D19 are all
  allocations of **256x32** networks. They cannot be reused at 1024x16 — the seeds
  would mint different-shaped networks. The vault's 500 sealed networks are in the
  same position: a Phase-2 vault is a **new mint**, and touch #1 becomes historical
  rather than a baseline to difference against.

- **`L/n` is the number that decides whether the mechanistic lane is alive**, and
  it moved by 8x. ARC's cumulant propagation breaks even when `L^K < n`. Phase 1:
  `L/n = 32/256 = 1/8`, and `L^2 = 1024 > 256` — dead at every K. Phase 2:
  `16/1024 = 1/64`, and `L^2 = 256 < 1024` — **alive at K = 2**, still dead at
  K = 3 (`4096 > 1024`). This was pre-registered as the reopening condition in
  `docs/CURRENT_STATE.md` open question #1 before the shape was known, which is the
  only reason reopening the lane is legitimate rather than motivated.

---

## 2026-08-20 (CP0.3) — the arithmetic gate, and three traps found inside it

*Full record and the third Skeptic verdict: `experiments/LOG.md` 2026-08-20, CP0.3.*

- **The binding promotion statistic is now the paired ARITHMETIC score difference**
  with the studentised-t or basic-bootstrap interval, via
  `competition_score.decide()` — the only function here that returns a promotion
  boolean. Geometric is a **diagnostic**. Percentile and BCa are **refused**, not
  deprecated (measured false-promotion 6.8–8.2% and 9.1–11.5% vs a nominal 5%).
  `docs/VALIDATION.md` §2b-i.

- **A two-sided interval test in a one-sided decision passes REGRESSIONS.**
  `decide()` shipped with `excl = ci[0] > 0 or ci[1] < 0`. A candidate 30% worse on
  every network has an interval that excludes zero *two-sidedly*, so with a gate
  below 1.0 it returned `passed=True` **with an empty `reasons` list**. Two locks
  now: the test is one-sided (`ci[0] > 0`), and `gate` must be finite and ≥ 1.0.
  **When the decision has a direction, so must the interval test.**

- **`arith > geo` is NOT decided by the sign of `corr(candidate, ratio)`.** That
  "iff" was written into a durable doc and is false. The identity is
  `arith = uAM(ratio) + cov(cand, ratio)/mean(cand)` while `geo = uGM(ratio) ≤
  uAM(ratio)`, so positive correlation is **sufficient, never necessary** — the
  AM–GM gap of the ratios usually dominates. Measured: correlation **negative on
  every lane**, arithmetic above geometric on most. The rule was generalised from
  D14, the one lane whose ratios are tight enough (AM–GM gap **1.0001**) for the
  covariance to decide.

- **`arith = sum(cand·ratio)/sum(cand)` is a TAUTOLOGY.** It cancels to
  `sum(ctrl)/sum(cand)`. It was quoted as though it corroborated something. Second
  instance in one day of the CP0.1 trap: *before quoting an agreement, ask whether
  the two sides could ever have disagreed.*

- **The N-effect on `MSE×N` is NOT RESOLVABLE on 48 networks**, and the T2 headline
  hides it. Decomposed: the `MSE·N` leg reads arith **1.0591** / geo **0.8983**, t
  interval **spans zero**, sign p = 0.193, **one network carries 174.2%** of the
  delta, and leave-one-out gives **[0.9551, 1.1080] — crossing 1.0**. The
  counted-FLOPs-per-point leg is solid (1.1306, 48/48, p = 0.000, 2.1% mover).
  **Quote the cost leg; do not quote the MSE·N leg as a measurement.** Any K129
  N-correction bracket built on it is the span between "no correction" and "a
  correction that is a draw".

- **A live CLI can outlive the gate it implements.**
  `packed_sparse_probe.py --mode decide5` kept gating on the geometric mean with a
  1.96·SE-on-log interval — neither the binding statistic nor an eligible interval
  — and was the natural route for the next single-use block. It now **refuses to
  run** unless `ARC_WHEST_ALLOW_LEGACY_GEOMETRIC_GATE=1`, which exists only to
  reproduce D14's historical decision. **When a gate changes, grep for every path
  that returns a verdict, not just the docs that describe it.**

---

## 2026-08-20 (CP0.2) — what the grader does that we could not see

*Full record and the second independent Skeptic verdict: `experiments/LOG.md`
2026-08-20, CP0.2.*

- **A setup failure kills the WHOLE submission, not one MLP.** `cli.py:2357` calls
  `runner.start()` **outside** the `try` that wraps `evaluate_estimator`, and
  `runner.py:155-172` raises `RunnerError(SETUP_ERROR | SETUP_TIMEOUT)` out of it.
  Nothing is scored. Any evaluator that reports a per-MLP objective when `setup()`
  raised or overran is hiding the most expensive failure mode a carrier candidate
  has. `local_sim/lifecycle_eval.py` now raises `SetupFailure` instead.

- **The setup window is NOT the duration of `setup()`.** `runner.py:144` starts the
  clock **before** `load_estimator_from_path`; the subprocess path
  (`runner.py:281`) waits `setup_timeout_s` for the whole round-trip, so the worker
  process spawn and its imports are inside it too. Measured here: spawn +
  `import flopscope, flopscope.numpy` = **0.22–0.25 s**. A timer wrapped around
  `setup()` alone reported **176.9 ms** where the honest figure is **433 ms** —
  40% of the truth.

- **`import whestbench` costs ~1.05 s cold and ~0.3 ms warm.** First load of
  `estimator.py` measured 1052.5 ms, second 0.3 ms, third 0.2 ms; a raw `compile()`
  of its 58,641 bytes is only **1.9 ms**. So the estimator's module body is
  essentially free and the second is a cold package import (HF `datasets`) that the
  grader's worker has warm. **Do not attribute a cold-import second to your own
  code.**

- **We have never observed the grader's timing or memory limits.** All five durable
  graded reports publish `config_snapshot` with exactly `n_mlps`, `depth`, `width`,
  `flop_budget`. `setup_timeout_s` (5.0), `wall_time_limit_s` (60.0),
  `residual_wall_time_limit_s` (None), `predict_timeout_s` (30.0) and
  `memory_limit_mb` (65,536) are **`ContestSpec` package defaults**. Only λ = 1e11
  is independently corroborated, from per-MLP telemetry. Say "the package default
  is", never "the grader allows".

- **`memory_limit_mb = 65,536` is RLIMIT_AS under `--runner subprocess` and it
  already killed v2.5** at a 3.9 GB peak. For scale: a 66,048-point carrier on 256
  dimensions is **135 MB** in float64 for the point set alone, before propagation
  temporaries.

- **A local evaluator's `objective` is not a graded objective**, and the reason is
  one term. Counted FLOPs transfer exactly; `residual_wall_time_s` does not, and it
  enters `C = F + λR`. λ·R is ~10–13% of `C` under `--runner local` against 2–3% on
  the grader. Use ratios inside one measurement path; never quote a local absolute.

- **`mean(target²)` is a float32 quantity.** The grader computes the failure-path
  MSE in float32 (`scoring.py:846`); a float64 version of the "same" number differs
  by **1.96e-9** relative. Go through `competition_score.zero_prediction_mse`.

- **A `# seed-ledger: excludes <BLOCK>` pragma is not decoration.**
  `seed_ledger.py:461` reads it as `bid not in excluded`, so an `excludes V` pragma
  written "for tidiness" in a file that names no block **permanently disables the
  vault-seed-literal scan in that file**. Only write one to declare a block you are
  refusing.

---

## 2026-08-20 (CP0.1) — the fifth "snapshot instead of property", and it was inside the GATE

*Full record and the independent Skeptic verdict: `experiments/LOG.md` 2026-08-20.*

- **A check that could not run must never report a pass.** The first build of the
  two-tier preflight gate printed step 9's SKIP as a green `PASS` and counted it in
  `PREFLIGHT GREEN -- 9/9`, while `docs/VALIDATION.md` simultaneously claimed the
  gate "prints in red exactly which checks it skipped". Both were written in the
  same hour, by the same person, and the doc was false the moment it was committed.
  The fix is structural: **the denominator is always 11**, a skip is counted
  separately and named in red, and a default run reads `GREEN 9/11, 2 SKIPPED`.
  **Quote the tier when citing a green run** — "preflight GREEN" alone now means two
  different things.

- **`test_seed_ledger.py` asserted `consumed_seeds("D19") == 0`, and D19 exists in
  order to be spent.** Once that file was wired into the mandatory gate, the first
  legitimate claim on the reserved promotion block would have turned preflight RED
  at exactly the moment `-Full` is required, and the only way forward would have
  been to edit the gate to get past it. **This is the fifth instance of the same
  error** (after B4, R1, the rotation check, and the D18 assertion that went stale
  twice in one session) and the first one wired into an enforcement path. The
  durable property of a SINGLE_USE block is **"at most one recorded claim, ever"**,
  never "zero claims today". D16 had the same defect (`== 270` seeds, four exact
  gaps) on a block with 330 free seeds.

- **A borrowed constant is not a measurement, and the tell is usually already in
  your own output.** A K129 comparison was priced with `0.5082`, which is the board
  **#1** entry's multiplier — a different team, different method, unknown point
  count. K129's own is 0.51728. The builder's own script printed "recomputed
  1.1215e-7 vs published 1.1433e-7" and did not chase the 1.9% gap. **When a
  recomputation of someone else's published number misses, stop: the input is
  wrong, not the rounding.**

- **"The product reproduces the gap exactly" can be an algebraic identity.**
  `MSE x (C/B)` gap *is* (MSE·N ratio) x (per-sample-cost ratio), by definition. It
  cannot fail, so it corroborates nothing. Before quoting an agreement, ask whether
  the two sides could ever have disagreed.

- **"N is not a lever" is an approximation, and this repo measured it breaking.**
  T2 (2026-08-19, FALSIFIED) measured **1.1980x** at 8x the points on 48 networks:
  MSE·N **1.0591** (QMC beats the 1/N law) times counted-FLOPs-per-point **1.1306**
  (the fixed closure cost amortises). Any argument that assumes the flat law at a
  different operating point is assuming its own conclusion.

- **The affordable-carrier number is `C/B = 0.6932` counted at 66,048 points**,
  fitted as `C(N) = 6.8534e9 + 2.7508e6 N` on our own D18 arms — not the 0.8021 a
  flat per-sample extrapolation gives. The flat estimate is **pessimistic** because
  it charges the fixed closure cost per point.

---

## STAGE 0, 2026-08-19 (night) — seven things that must not be rediscovered

*Five slices repaired the instrument. Full record and contracts:
`experiments/LOG.md` 2026-08-19 (night). Every item below is measured.*

- **Layer-0 alphas are EXACTLY 0.0, for every neuron, on every network.** The
  input is `x ~ N(0, I)`, so the layer-1 pre-activation mean is zero and
  `alpha = mu/sigma = 0`. Verified as a single float64 bit pattern across 27
  (width, seed) combinations, widths 63–257. **Never compare alphas with a strict
  `>` against their own minimum**: `a > alpha[alive].min()` keeps *nothing* at
  layer 0. v2.6 did exactly that and deleted the entire first layer's sampled path
  on every odd-width network — 0.58 max relative error, 0.33 prediction MSE. The
  design probe it was translated from (`probe_strassen_carrier.py:113`) used
  `argsort`, which is tie-safe; the bug was introduced turning that into a
  threshold.

- **The competition objective is the ARITHMETIC MEAN of per-MLP scores over the
  FIFTY public MLPs** — not all 100 scored, and not a geometric mean. It is now
  reproducible offline **bitwise** from the durable graded reports:
  `2.719427529595e-07` (324420) and `2.631432177242e-07` (327743), exactly, from
  `mse_i` and `C_i` alone. `local_sim/competition_score.py` is the one place that
  arithmetic lives.

- **A failed MLP is scored at multiplier 1.0, not at the 0.1 floor** — exactly
  `1/mult_i`, which is 4.59–5.41x across the real graded suite — *and* against
  zeroed predictions. The exception path records `C = 0`; the exhaustion paths
  record the real `C` and still force the multiplier to 1.0.

- **The arithmetic and geometric estimands select OPPOSITE winners on our own
  graded data.** v2.4 -> v2.4.1, same 50 rows: arithmetic **1.0334x** (v2.4.1
  better), geometric **0.9666x** (v2.4 better), 22/50 wins. Neither is
  mis-computed; they answer different questions, and the competition asks the
  arithmetic one. *(`docs/GOTCHAS.md`'s own "ratio of means 1.0319" below is the
  raw-MSE frame and is CORRECT as written — 1.0334 is the adjusted-score frame.
  Do not "fix" one into the other; that is the frame-mixing `VALIDATION.md` §2b
  exists to prevent.)*

- **Common random numbers change the SHAPE of the estimand, not just its
  variance.** Each arm's per-MLP score is strongly right-skewed (worst/mean 3.59,
  top-1 tail share 7.2%), but the CRN *paired difference* is nearly symmetric —
  measured skew **+0.21, -0.55, -0.46** on 192 fresh networks. Do not transfer
  "this metric has a heavy tail" from the score to the difference.

- **Use the studentised t interval or the basic bootstrap. Do NOT gate on the
  percentile bootstrap or on BCa at n ~ 50.** Measured false-promotion rate
  against a true zero, nominal 5%: t **4.1–5.4%**, basic **4.0–5.2%**, percentile
  6.8–8.2%, **BCa 9.1–11.5% — roughly double**. This inverts the textbook
  ordering, for the reason above: BCa corrects a skew that CRN already removed.
  BCa only becomes the best method past skew ~6, which no real arm reaches.

- **Assert PROPERTIES, not snapshots — and never with a tolerance that ignores
  the estimator's own sampling error.** This bit four separate checks in a single
  session (2026-08-19): a 2% bar on counted FLOPs that have a **3.22%**
  per-network spread (v2.4.1 prunes; v1.1 does not and reads exactly 0.00%); a
  false-promotion rate read off a column whose population mean was not zero; a
  rotation check using an isotropic probe, which is rotation-invariant *by
  construction* and can never see alignment; and a seed-ledger assertion pinned to
  a snapshot of a block that grows every time an experiment claims from it. Before
  writing a check, ask what its own noise is, and whether the statistic can
  measure the thing being claimed.

- **A ±1 cube array is a degree-4 design only IN EXPECTATION OVER SPREAD
  DIRECTIONS, and the random rotation is what makes that true.** For such a point
  set `E[(a·u)⁴] = (3 − 2S)/65536` with `S = Σaᵢ⁴`, against the spherical
  `3/(256·258)` — equal exactly when `S = 3/258`, which is the expected `S` for a
  random unit vector in R²⁵⁶. Confine the direction and it degrades as predicted:
  0.00 / 0.77 / 2.27 / **5.15%** low at k = 256/128/64/32, measured 7.55e-04 /
  8.04e-03 / 2.32e-02 / **5.38e-02**. **Unrotated and confined to 32 dims the
  array is 4.6× WORSE than Sobol; rotated it is flat at 7.7e-04.** So a design
  cannot be aligned with the response's rank-32 subspace to save points — the
  alignment costs more than it saves, and balance on all 256 columns is required.

- **The grader's residual is a LINE, not a ratio — and we had the data for a
  month.** Fitted on all five graded submissions:
  **`residual ≈ 2.551 ms + 5.780 µs/op`** (R² = 0.898, leave-one-out 0.829 ms).
  **2.551 ms is fixed overhead charged before our first operation** — 25% of
  v2.4.1's whole residual. The old "7.0 µs/op" was a ratio *through the origin*
  from one submission and mis-prices any candidate whose op count differs.
  **Residual is only 1.9–3.3% of `C` on the grader**, so `C` is determined far
  better than the local measurement suggests — the local number reads 124.55
  ms/MLP for v2.6 where the grader projects 16.25 ms, **7.7×**, and that ratio is
  the entire reason the v2.6 sign used to flip. *Do not model the local residual;
  use graded telemetry.*

- **Residual tracks OPERATIONS, not FLOPs, and the proof is in our own reports.**
  v1.0 has **12.5× v0's counted FLOPs and LESS residual** (4.491 ms vs 6.743 ms).
  No monotone function of FLOPs can produce that. Ops explain residual at
  R² = 0.898; counted FLOPs at 0.387.

- **A tolerance that ignores the estimator's own sampling error is not a test.**
  Counted FLOPs are deterministic *given a network*, but v2.4+ prunes and the
  alive fraction varies: per-network sd **3.22%** for v2.4.1, **exactly 0.00%**
  for v1.1, which does not prune. Six seeds against a flat 2% bar reads FAILED on
  noise. Bit three separate checks in one session.

- **Allocating a seed block is not consuming it, and only allocation was ever
  checked.** `check_disjoint()` cannot see `SEED0 = 30_806_016_200` typed into a
  probe. D16 was called "Reusable" for nine days while five probes had drawn 270
  of its seeds — and **`probe_whitening` and `probe_design_carrier` drew the SAME
  24 networks**, so those two "independent" carrier kills are one sample. Take
  seeds with `seed_ledger.claim(...)`, never a literal, and write the returned
  range into the artifact.

---

## GRADED v2.4.1 (2026-08-10, submission 327743) — how a number is read

*Written 2026-08-10, then **corrected the same day after independent falsification**.
The struck claims are kept because the correction is the lesson: three of them were
falsified by artifacts that were already on disk and already cited in the slice's own
contract. Skeptic verdict PROVISIONAL; full audit in `experiments/LOG.md`.*

- **A 3.2% aggregate improvement on 50 real grader networks was a DRAW.** v2.4.1
  scored 2.6314e-7 against v2.4's 2.7194e-7 on the *identical* suite (all 100
  `mlp_name` values match in order, and `sampling_mse` is identical to 17 digits, so
  the design is genuinely paired). The mandated statistic — geometric mean of per-MLP
  ratios — is **0.9665×, 95% CI [0.7969, 1.1721]**, v2.4.1 better on **22 of 50**
  (exact one-sided p = 0.839). Ratio of means 1.0319, ratio of medians 1.0107.
  **Say "no *resolvable* effect", not "no effect".** With sd(log ratio) = 0.696 at
  n = 50 this design can only exclude |effect| > ~20%. A better-powered answer
  already existed and agrees: **0.9959× over 32 draws/MLP** (see the v2.4.1 entry
  below). The grader run *confirms* what the 32-shift local test already said; it did
  not discover it.
- ~~"The headline came from one network, `mlp_index 37`, which alone contributes 140%
  of the total delta."~~ **Overstated.** 140.2% recomputes correctly, but *three*
  networks each individually flip the sign of the total (idx 37 −8.55e-7, idx 14
  −6.48e-7, idx 1 −3.06e-8). The 2.1246e-6 total is the residue of **+1.9102e-5 in
  gains against −1.6978e-5 in regressions**. The honest description is *"the headline
  is a small residue of near-cancelling sums"*, not *"one network did it"*.
  **The diagnostic still earns its place, with a stated false-positive rate:** rank
  the per-MLP deltas and ask what fraction of the total the biggest mover carries.
  ≥100% means treat as a draw until proven otherwise. Simulating a *true* 5%
  improvement at the observed scatter fires it in ~10% of draws, so it is a screen,
  not a proof.
- **The local/grader residual ratio is path-dependent AND time-dependent, in roughly
  equal measure.** Same byte-identical `estimator.py`, same 100-MLP mini split:

  | reading | ms/MLP | vs grader 10.022 |
  |---|---|---|
  | `--runner local`, 2026-08-06 | 70.85 | 7.07× |
  | `--runner subprocess`, 2026-08-06 | **54.47** | **5.44×** |
  | `--runner subprocess`, 2026-08-10 | 42.94 | 4.28× |

  Runner effect, same day: **1.301×**. Machine-state effect, runner held fixed:
  **1.269×**. They multiply to the observed 1.65×. ~~"Purely a function of which
  local path was used"~~ is **false**, and ~~"~65% too large"~~ becomes **~30%** on
  the same-day comparison. The Aug-6 subprocess figure is recoverable from
  `2026-08-06_v2.4.1_mini_subprocess.json` and independently from its own recorded
  multiplier 0.216 with counted 5.330e10 — it was in the contract's evidence list the
  whole time.
- ~~"Sprint 4 S2 and the D14 residual model both scaled by ~7."~~ **False — two
  different sevens were conflated.** D14's deciding model is *local **ratio** ×
  graded anchor*: `r_cand = 9.84 ms × (490.4/62.3) = 9.84 × 7.87` (`experiments/LOG.md`,
  D14 residual-model table). That 7.87 is a **within-run candidate/control ratio**,
  invariant to any uniform rescaling of this machine; the local→grader *level* never
  enters. S2 is the same shape (`9.84 × 5.47`). Model 1, local absolute, is explicitly
  flagged "never decides". **The real exposure is different and is not yet recorded
  anywhere else: that candidate/control ratio itself reads 5.47× in one local harness
  and 7.87× in another — a 44% spread on the quantity that actually decides.**
- **`AGENTS.md` § *Measurement law* still says "~7×" and must stay unedited** until
  the repeatability run below. If three same-day repeats span >10%, the right fix is
  not "4.3× instead of 7×" — it is *"this machine has no quotable absolute residual
  level"*, and the line should become a range with its measurement path attached.
- **Only the final layer differs between v2.4 and v2.4.1 — verified on all 50, not
  one.** `per_layer_mse` has exactly one differing layer per network and it is
  *always* index 31; layers 0–30 are exactly float-equal across 50 × 31 comparisons.
  Good invariant candidate: a rotation/sampling change that alters an earlier layer
  has escaped its intended scope.
- **The +1.8% grader residual is a detected shift, not evidence of stability.** Paired
  1.0180× [1.0119, 1.0241] **excludes 1.0**, so calling the grader "steady to ±0.6%"
  mislabels a CI half-width as drift. But n = 1 submission per file: the n = 100 CI is
  pseudoreplication for a between-submission question, effective n = 1. `residual_s`
  is also a *subtraction remainder* (`wall − backend − overhead`), and the components
  moved far more than the remainder — paired backend 0.9417, overhead 1.0476, wall
  0.9629 — so a 0.177 ms difference is a 0.27% residue of ~78 ms of component motion.
  Do not attribute it to the `eigh` path.
- **Round dates changed under us mid-session** (two polls 30 min apart, 2026-08-10):
  Phase 1 end moved Aug 7 → **Aug 10 23:59 UTC**, Phase 2 open moved Aug 8 →
  **Aug 18**. Round metadata is live data, not a constant — re-poll
  `phase2_checklist.py` immediately before acting on a deadline, never trust a
  cached one. Submission 327743 went in at **19:00Z, ~5 h before close**, not with
  the day-and-a-half margin the slice contract assumed.
- **The driver misread the session date twice** (as Aug 7, then Aug 9; actually
  Aug 10), and wrote it into a filename, a LOG heading and two durable docs before it
  was caught. `Get-Date` was misread, and a "correction against two independent
  sources" cited a `git log` last-commit date, which is a *lower bound on the present*
  and cannot date anything. **Date the session from an artifact the grader or the
  filesystem stamped** — a graded report's `createdAt`, a commit you just made — never
  from an older commit.
- **Round dates changed under us mid-session** (two polls 30 min apart, 2026-08-10):
  Phase 1 end moved Aug 7 → **Aug 10 23:59 UTC**, Phase 2 open moved Aug 8 →
  **Aug 18**. Round metadata is live data, not a constant — re-poll
  `phase2_checklist.py` immediately before acting on a deadline, never trust a
  cached one.

---

## THE TRUTH HARNESS (2026-08-06) — read `docs/VALIDATION.md` before measuring anything

- **Our mints ARE the grader's mints now, and were not before.** `whestbench` builds
  every network (public splits *and* private re-run suites) as
  `weight_ss, sample_ss, est_seed = SeedSequence(input_seed).spawn(3)` →
  `sample_mlp(width, depth, default_rng(weight_ss), seed=est_seed)`, drawing in
  **float64** and casting once. Our old generator drew float32-natively (different
  bit stream ⇒ completely different numbers) and aliased `MLP.seed` to the weight
  seed — which mattered, because `estimator.py:532` derives the Sobol shift from
  `mlp.seed`, so our point set was a function of the network it scored.
  `local_sim/mint_provenance.py:canonical_mint` is proven **bit-identical, 100/100**
  (5 shapes × 20 seeds, uint32 comparison). **Always mint through it**; the old path
  survives as `harness.gen_mlp_legacy` for reproducing historical numbers only.
  GPU mirror mints on CPU and copies (torch cannot reproduce numpy's stream):
  generation bit-for-bit, forward pass agrees to 5.6e-17.
- **Three-tier data protocol, mandatory.** DECIDE on fresh mints (new seed block per
  experiment, `local_sim/seed_blocks.py`, never reused) → CONFIRM on the public split
  (**read-only; no constant may ever be fitted there again**) → VAULT for milestones.
  Legacy blocks L1–L5 were minted pre-fix and are not comparable to anything after.
- **The vault is touch-counted.** 500 sealed MLPs (block V, 20,260,806,000+), GT at
  2²⁵ samples, noise ~1.6e-9. `experiments/vault/manifest.json` + `touch_log.jsonl`
  are tracked in git; the `.npz` is not (weights aren't stored — the provenance proof
  makes seeds sufficient). Permitted touches: baseline, one per estimator change,
  designation day. **Never evaluate it to browse.**
- **`edge_cases.py` is now 30/30** and includes invariants, not just survival:
  layer-1 closed form (exact to 1.8e-16), **permutation equivariance** (catches
  transposed indices / swapped reduction axes — nothing else does), budget
  monotonicity, determinism ×3 + fresh instance, seed-block disjointness.
- **Vault baseline (touch #1, v2.4.1):** adjusted median **2.4651e-7**, mean
  **3.2669e-7**, p99 **2.0535e-6**, max 3.0469e-6; zero non-finite predictions.
  GT noise is 0.09% of measured raw MSE. The touch counter and log both read 1.
- **Constants audit (fresh blocks D01–D05, 200 MLPs × 32 shifts for the paired
  sweeps):** keep τ=-2.5; keep rotation rank 32 (rank 256's 1.0108× apparent gain
  is only 0.7σ); keep the v0 calibration table (≤0.07% drift). The old 3.8e-9
  pruned-value error bound is withdrawn and replaced by **1.043e-8** on canonical
  mints; the independent end-to-end τ sweep still selects -2.5. N=65,536 beat
  N=16,384 by 1.0408× at 3.2σ in the FLOP-only mint audit, but only ~1.007× on
  real local flopscope after residual billing, so **do not adopt**; retain as a
  Phase-2 telemetry candidate. The low-budget analytic fallback is currently raw
  closure—the audited calibration table is not wired into v2.4.1.
- **Private-half scores are NOT VISIBLE** (exhaustive key search, all 4 graded
  reports: 100 `per_mlp`, 50 `public_scores`, `aggregates` has only `public`). Do not
  claim an accuracy divergence. The **cost** ratio IS visible and is 0.9916–0.9999
  (v2.4 spends 0.84% less on the hidden 50) — that is the write-up's honest claim.
- **Board census** (`local_sim/board_census.py`, PRIVATE): 2026-08-06 10:03Z,
  round 1428 — 8 probing-signature, 192 plausible-honest, **us #82/192**
  (raw #90/200) on 5 entries.
  Criteria are pre-fixed; the screen is conservative, so #82 is an upper bound on how
  many real methods beat us.
- **Run `whest` from PowerShell, or set `PYTHONIOENCODING=utf-8` in Bash.** The CLI
  prints box-drawing characters; Git Bash's cp1252 console makes it die with
  `UnicodeEncodeError` *inside its own error printer*, which masks the real result
  (validate actually passes in 42 ms). The same class of contamination put 703
  characters of `uv : Warning: You are sending unauthenticated requests…` at the
  front of `2026-08-06_v2.4.1_mini_subprocess.json` and made it unparseable —
  repaired 2026-08-06. **Redirect `whest run --format json` output carefully.**
- **Phase-1 submission decision is conditional after new evidence (GAME_PLAN rev 9).**
  The original no-submit decision assumed the default top-2 selection was the only
  path. Organizer thread 18125 now says teams will receive nomination instructions
  and may explicitly choose two submissions. If Samuel approves before the live
  round closes, grade v2.4.1 and later nominate {v2.4.1, v1.1}; otherwise preserve
  the no-submit decision. **Never submit without Samuel's explicit approval.**

## SPRINT 5 VERDICT (2026-08-06) — the packed lane is KILLED. Authoritative plan is GAME_PLAN rev 12.

- **D14 (locked, single-use, SPENT): paired `V·C` = 1.1216× [1.1188, 1.1244]
  against a pre-registered 1.15× gate. FAIL.** Design and residual model were
  frozen in `experiments/LOG.md` before the block was touched; D13 predicted
  1.113% and D14 measured 1.1216×, so cost behaviour agrees across blocks. **Do
  not retune, do not reuse D14, do not request telemetry for any packed
  candidate.** `estimator.py` untouched, vault untouched (counter 1).
- **The counted arithmetic target IS reached.** Extrapolated to 319341's
  N = 84,992 our frozen carrier costs **2.137e6 counted FLOPs/sample against
  their 2.144e6** (0.4%). Always compare counted-to-counted; their 2.374e6 is
  *effective*.
- **Exact packing moves no variance — settled.** D14 control and candidate MSE
  agree to five significant figures (1.5586e-06). D12's 1.2079× is retired.
- **Operation count is NOT the residual driver here.** Batched depth-2 Strassen
  is *bitwise* identical to the recursion (shares `_ops7`/`_combine`), cuts the
  shipped file 10,263 → 4,823 calls, and measured **worse** residual at
  N = 16,384 / 32,768 / 65,536. Shipped **off** in `estimator_v2_5r.py` behind
  `_STRASSEN_BATCHED`; the implementation and its 93/93 tests stay.
- **N is not a lever.** `MSE × N` is flat over {16,384, 32,768, 65,536} at
  8 shifts × 24 MLPs (0.0366 / 0.0404 / 0.0367). A 1-shift pass said 32,768 was
  14% better; that was one heavy-tailed realisation. Residual is ~96% marginal
  in N (`r(N) = 31.9 ms + 9.27e-6 s/sample` for the packed carrier).
- **The mechanism, and it belongs in the write-up:** flopscope prices a gather at
  4 FLOPs/element, and packing's entire saving is substituting gathers for dense
  multiply-adds. A random-access gather costs far more *wall time* per element
  than a BLAS multiply-add, and flopscope charges wall time back at λ = 1e11.
  The lane arbitrages the price table against physics, and loses.
- **Residual-model rule, now standing (`docs/VALIDATION.md` §2b).** Report three
  models; freeze one before the decision block; the deciding model is
  `max(local-ratio-on-graded-anchor, 319341-anchored)`. The `max` is load-bearing
  — the competitor anchor *alone* is optimistic for Strassen arms and would have
  passed this carrier at ~1.21×.
- **Still open and now the top score question:** 319341's `MSE × N` = 0.018520
  against our graded 0.022525 (**1.216×**). Not arithmetic, not packing, not
  antithetic. It is a *variance* mechanism.
- **Next work is Moonshot G0 only** — layer-convention parity, a separate
  calibrated-K2 control, the low-rank flow as an accuracy oracle. **Do not touch
  D08; do not claim compressed cost.**

## MOONSHOT G0 — DONE 2026-08-07 (`local_sim/moonshot_g0.py`, block D15)

- **ARC `mlp_kprop` is now on this machine** (Samuel approved the fetch): pinned
  rev `93d091a4` at `C:\Users\samsu\Documents\mlp_cumulant_propagation`, with
  `local_sim/mlp_kprop_py310.patch` applied. **Set
  `MLP_KPROP_SRC=C:\Users\samsu\Documents\mlp_cumulant_propagation\src`** before
  running anything in this lane. Never installed into or imported by the
  participant process.
- **Layer-convention parity: PROVEN, twice.** The row convention (`z = Wᵀμ`)
  beats the column convention by 7,123× at layer 1 and 130,000× by layer 2, and
  ARC's own `k_max=2` flow reproduces our closure to **8.373e-15** max relative
  per-layer MSE. The adapter's `W.mT` is correct.
- **The registered calibrated-K2 baseline now exists** (it did not anywhere —
  v2.4.1's analytic path returns *raw* closure and the table lives only in
  `experiments/calibration.json`): **depth 8 → 1.2067e-05** (G1 must reach
  ≤4.02e-06); depth 32 → 2.4428e-05. Fixed on D15 before any D08 draw.
- **Fixed a real adapter defect:** at `k_max=3` upstream carries a degree-4
  `HTensor`, and the dense fallback tried to materialise 256⁴ = **34 GB** and
  OOM'd the GPU. `project_higher_cumulants` now takes `k_max` and refuses
  over-large dense projections. `k_max=3` runs in 0.3 s/MLP.
- **The oracle reading, unrenormalized, depth 8, vs the calibrated baseline:**
  r=4 **0.427×**, r=8 0.466×, r=16 0.545×, r=32 0.764×, r=64 1.300×,
  r=128 **2.996×**, full rank **12.786×**. So the mechanism is REAL (12.8× at
  full rank) but **MOONSHOT's pre-registered r ∈ {4,8,16} is worse than the
  control**, and the ≥3× gate needs **r ≈ 128 = width/2 — exactly where Track P
  showed a rank-k factorisation stops beating a dense matmul.**
- **NOT a G1 result.** Renormalization scalars were 1.0; the method requires them
  fitted offline (18097, block D10). **D08/D09 untouched, G1 not run.** No cost
  model or r³ claim. Next step is the D10 renormalization fit, not D08.

## INDEPENDENT AUDIT OVERRIDE (2026-08-06) — historical; superseded above for the packed lane

- **Authoritative plan is GAME_PLAN rev 11. Do not submit current
  `estimator_v2_5.py`.** There is useful offline carrier work left before any
  telemetry submission.
- **S1 is borderline/inconclusive, not a cost-only pass.** The 1.2026x number is
  a ratio of means. The pre-registered paired geometric cost statistic on D12,
  holding MSE neutral, is **1.1996x [1.1963, 1.2029]**, fractionally below the
  1.20 gate. The 1.4070x full result relies on the non-reproducing D12 MSE draw.
- **The subprocess timeout is not candidate-specific.** Independent repeated
  starts produced 2/6 `SETUP_TIMEOUT`s for shipped v2.4.1 and 0/6 for v2.5;
  v2.5 completed an isolated subprocess prediction. The local runner's five
  second cold-start deadline is flaky on this host. Do not call the 3.9 GB peak
  a 64 GB memory-cap failure without direct evidence.
- **Use the comparable variance gap: 1.216x.** Graded v2.4.1 has
  `MSE*N=0.022525`; 319341 has 0.018520. The D11/local 1.40x comparison crosses
  suites and must be labelled diagnostic only.
- **Do not compare our counted 2.359e6/sample with their effective
  2.374e6/sample.** Their counted figure is about 2.144e6/sample. With Sprint
  4's own 66 ms pre-registered residual, current v2.5 is about 2.762e6
  effective/sample. The arithmetic mechanism is reproduced; the published
  effective-cost target is not.
- **Sprint 5 is round-trip compression plus N amortisation.** Batch Strassen
  leaves, reduce bucket/control-flow operations, and sweep N=16k/32k/65k on
  fresh D13 design data; reserve D14 as the single-use decision block. A clean
  microbenchmark already made batched depth-2 Strassen bit-identical while
  reducing calls 219->173, residual 16%, and wall time 40% (+1% FLOPs).
- **Only an optimized, fully subprocess-safe candidate may request one Phase-2
  telemetry submission, and only after Samuel explicitly approves.** Frozen
  v2.4.1 remains the first Phase-2 configuration probe and the fallback.

## SPRINT 4 (2026-08-06 late) — historical report; corrected above

- **The mechanism reproduces.** Independent rebuild of 319341's described method
  (firing-rate column ordering → cold prefix + dense hot tail → row buckets on
  prefix support → exact zero-rectangle skipping → gathered corrections →
  Strassen) reaches **2.359e6 effective FLOPs/sample against their published
  2.374e6**, from first principles. Lab: `local_sim/packed_sparse_probe.py`.
- **S0 PASS.** The dense reference is **bit-identical** to shipped `estimator.py`;
  packing agrees to 1.5e-8 relative; MSE ratio 1.0000. `test_packed_sparse.py`
  **62/62**, including a 1e-20-activation-against-1e20-weight test that would fail
  if anything were thresholded rather than exactly zero.
- **S1 PASS, but quote ~1.23x, not the headline 1.41x.** D12 (locked, single-use,
  50 MLPs x 6 shifts) gave 1.4070x [1.2656, 1.5642] on the pre-registered cost
  model — but its MSE ratio of 1.2079 does **not** reproduce (D11 gives 1.0199,
  and no component ablation predicts it). **The cost half clears the gate alone at
  1.2026x with variance neutral**, which is the defensible claim.
- **S2 FAIL — the real stack bills the operations.** Full 100-MLP public mini:
  counted FLOPs 3.86e10/MLP (vs the probe's predicted 3.854e10 — 0.2%), but
  **residual wall time 387.6 ms/MLP against v2.4.1's 70.8 ms**, so effective
  compute 7.74e10 vs 6.04e10 and **adjusted 4.11e-7 vs 3.26e-7 — 26% WORSE**.
  Raw MSE 1.48e-6 vs 1.49e-6, i.e. **no variance gain**, which convicts D12's
  1.2079 as a draw. 0/100 failed. **The lane does NOT promote.**
  Caveat that cuts both ways: this machine bills v2.4.1 70.8 ms where the grader
  billed 9.84 ms (7.2x), so applying the measured 5.47x residual *ratio* to the
  grader's anchor predicts a 1.23x win instead. Only a graded run decides, and the
  `RemoteArray` client/server backend makes the pessimistic reading more likely.
- **S2 also BLOCKED — `SETUP_TIMEOUT` under `--runner subprocess`, twice.** Shipped
  v2.4.1 scores fine on the same command and machine, so it is the candidate.
  It is NOT slow setup (162 ms vs v2.4.1's 340 ms) and the estimator is sound
  (`whest validate` passes; `--runner local` scores it). Prime suspect is the
  worker's `memory_limit_mb=65536` enforced via `RLIMIT_AS`: v2.5's transient
  footprint is far larger (Strassen depth 2 holds 7 sub-products + 10 operand sums
  live per level; gathered corrections materialise `(rows, k, a_out)`), and a
  candidate process was seen at ~3.9 GB. **A killed worker scores zeros — this is
  a shipping blocker.** v2.4.1 remains the fallback and `estimator.py` is untouched.
- **The closure predicts its own sparsity to four decimals at every depth**, which
  is why the pilot component is dead: it cannot improve on that and costs a second
  plan build (~15%). Independent accuracy result worth the write-up.
- **Two "findings" this sprint were noise wearing a mechanism**: a 12-MLP run said
  antithetic costs 12% variance (30x8 says it *gains* 0.9%); D12 said the candidate
  wins 21% variance (D11 says 2%). Both had tidy causal stories on demand. Refuse
  any number no ablation predicted until it reproduces on another block.
- **Local residual is 7.2x the grader's** (70.8 ms/MLP measured by `whest run`
  here vs the graded 9.84 ms). Never quote the local level; transfer the ratio.
- Seed blocks **D11** (design, tuning allowed) and **D12** (locked, single-use,
  SPENT) — `seed_blocks.SINGLE_USE_BLOCKS`. Vault **not** touched (count still 1).

## Known unknowns / gotchas (updated 2026-08-06)

- **DEADLINE CONFLICT — plan against Aug 7.** Organizer thread 18125 (Aug 3) says Phase 1 closes **10 Aug 23:59 UTC**; the challenge page's round object still ends round 1428 at **2026-08-07 23:59 UTC** and opens 1429 at Aug 8 00:00. Thread 18130 flagged it Aug 5, no reply. Treat Aug 7 as operative and **trigger the Phase-2 runbook on round 1429 going active**, never on a date. Write-up due 17 Aug; registration/team freeze 5 Sep.
- **The residual-billing question is RULED ON — don't re-ask it.** 18122: eval servers ship `flopscope-client`, `fnp.ndarray` is a `RemoteArray` with no `.base`, so raw-array bypass is impossible there, and "it would be cheaper for you to use flopscope instead". 18132: "we will examine this during the private re-evaluation and prize due diligence"; "using residual pricing as an alternative arithmetic budget is not [legitimate]". **Our instrumented share is 98.2%** (5.31e10 FLOPs vs 9.84 ms residual on submission 324420) — quote it if due diligence ever asks.
- **Participant code runs on ONE physical core (2 vCPUs)**; the flopscope backend gets seven (14 vCPUs); λ still 1e11. All our graded numbers are post-0.10.0, so §3d pricing needs no revalidation.
- **v2.4.1 shipped** — same method and same score as v2.4 (0.9959× over 32 draws/MLP), but the rotation is now reproducible. JJᵀ is numerically rank ~68 (λ₉₆/λ₀ = 4.5e-17), so the raw eigenbasis is arbitrary past that: a **1e-15** relative weight perturbation moved it by 0.81 in max column distance and handed us an independent realisation. Canonical *signs* do not fix this (same 0.81) — the ambiguity is a rotation inside a near-degenerate subspace. Fix = keep the top 32 directions, pin signs, complete by `fnp.linalg.qr(mode="complete")` (2.0e6 FLOPs). Shim-vs-whest gap: **18% → 1.66%**.
- **A single 100-MLP mini run cannot resolve a <15% question.** Two realisations of the same estimator differ ~18% in the mean. For any A/B on the point set or the basis, average each MLP over many digital shifts (`local_sim/basis_parity.py`, 32 draws) instead. v2.4.1's mini run looks 6.7% better than v2.4 — that is luck, not method.

## Older gotchas (2026-08-05 night)

- **A probe living in `local_sim/` silently imports the flopscope SHIM, not the
  grader's cost model.** Python places a script's own directory at `sys.path[0]`,
  and `local_sim/flopscope/` is a numpy shim of the same name. `import flopscope`
  from any script in that directory resolves to the shim, whose `BudgetContext`
  exposes none of the real attributes — so it fails loudly if you read
  `flops_used` and **silently gives you nothing if you do not**. Any probe that
  needs real FLOP counts must evict its own directory from `sys.path` before the
  import and assert on `flops.__file__` (see `local_sim/probe_strassen_carrier.py`),
  exactly as `mint_provenance.py` already does for `whestbench`. Found 2026-08-19
  after it produced an `AttributeError` that looked like a flopscope API change.
- **Prizes are decided by a private re-run of UP TO TWO designated submissions on FRESH MLPs** (forum 18143, 2026-08-10; default = your top two public entries; the Rules tab still says one and the contradiction is unresolved).** Public-board rank is not a target; judge everything on the local 100-MLP mini split and on ≥1000 fresh minted MLPs (`local_sim/robustness.py`). The tail of the fresh-MLP score distribution is the real risk, not the mean.
- **Score optimisation is closed except Sprint 4's packed-sparsity reproduction**
  (GAME_PLAN rev 9). Sprint 3's static on/kink composition measured 1.0246×, but
  it did **not** test submission 319341's value-dependent cold-column packing,
  row buckets, exact zero-block skipping, or Strassen dense blocks. That public
  method regraded at 1.599e-7 and uses samples 1.39× cheaper than ours, so it is
  the one evidence-backed reopening condition. All older variance/analytic tracks
  remain closed; kill Sprint 4 unless paired fresh-mint `V*c` improves >=1.20×.
- **On-neuron reduction, verified twice** (keep — it is the write-up's theorem): for α large, E[ReLU(z)] = E[z], so at the scored layer an on-neuron needs no per-sample value, and at a feed-forward layer it is *linear* and composes (linearising every α>2.5 neuron in the last 8 layers costs 0.07% MSE). But **never mean-replace** a feed-forward on-neuron — that deletes its fluctuation and costs 48× at k=2, 3 orders of magnitude by k=8.
- **A constant fitted for one role does not transfer to another**: κ=1.363 (Track T's closure-variance rescale) makes the Sprint-3 rectification correction *worse*.
- **Fresh-re-run readiness: READY.** 1800 fresh + deformed MLPs, zero non-finite outputs, alive count never below 117/256, degenerate-layer fallback never fired, analytic patch bounded at 6.3e-3, α to ±12.4. Contract + invariant suite `edge_cases.py` **30/30** on the shipped file (zero layer, 1e4 dynamic range, seed 2⁶³−1, widths 64–256, depths 2–64, budget 0, input immutability, exact layer 1, permutation equivariance, budget monotonicity, determinism, seed hygiene, vault-noise floor).
- **The scored metric has a heavy tail, and it is mostly the metric.** Fresh-MLP adjusted score: median 2.19e-7, mean 3.01e-7, p99 1.65e-6, **max/median 12.9×**; worst 1% of MLPs carry 6.9% of the score. `corr(log per-MLP MSE, log mean(truth²)) = +0.548` — MSE is unnormalised and He draws vary ~5× in activation scale. **At a re-run size R=10 the p90 score is 50% above the median**; at R=100, 13%. Draw variance dwarfs any remaining ~2% estimator gain — which is the quantitative reason score work is closed.
- Minted MLPs are a faithful stand-in (GPU mirror: 1.6178e-6 fresh vs 1.5962e-6 on real mini, +1.4%) and if anything slightly *harder* in median (1.19e-6 vs 0.97e-6), so fresh-MLP numbers are conservative. The shipped file and the GPU mirror agree to 0.1% in median on the same fresh MLPs.
- **The public score is the mean over 50 FIXED MLPs (indices 0–49)**, and every submission returns those MLPs' exact per-MLP `final_layer_mse` + a 32-long `per_layer_mse` in `results.public_scores`. Verified bit-for-bit against the aggregates; the suite is identical across all four of our submissions. Consequences: (a) our graded A/B comparisons are **paired re-measurements**, not fresh samples — never read a graded improvement as population evidence; (b) the level carries a fixed 7–20% offset vs the local 100; (c) **decisions stay on the local 100-MLP mini split**, which is the honest proxy for the private re-run that sets the final ranking. Full record + the arithmetic: RESEARCH_NOTES §2g (**private — never the forum**). **We do not use that channel.**
- **The `sampling_mse` label-noise bound is WITHDRAWN.** 89 of 200 board entries show a final-layer MSE below 6.4695e-7, so it cannot be additive reference noise. Most likely a reference sampling baseline's MSE. Do not repeat the "nothing honest below 6.47e-8" claim.
- **Track T (shrinkage-moment tail) is CLOSED** — 0.9998× vs a 1.10× gate. Only `diag(WᵀΣW)` = `mean(z²)−mean(z)²` enters, so there is no covariance to estimate (no MP regime); v̂ is already 1.13% rms accurate and the exact-variance oracle still gives 0.611×. 41% of the analytic tail's error is irreducible model bias; ρ=1.000 with the sample mean in the α≥3 band that carries 76.5% of the MSE. Don't reopen without a new mechanism.
- **Track P (rank-k deviation propagation) is CLOSED** — 1.093× vs a 1.25× gate, even pricing an oracle Jensen correction at zero. Rank-k propagation *is* a rank-k weight factorisation; shape-based billing breaks even at k = width/2 = 128, which is also the rank truncation needs. (The spectrum result itself is real and write-up-worthy: effective rank 25.9→2.8 over depth, and the closure's own covariance eigenvectors find the subspace for free.)
- **⚠️ `local_sim/eval_estimator.py` (shim path) is 18% LOW vs `whest run` for rotation-using estimators.** Same `estimator.py`, same 100 mini MLPs: whest 1.6079e-6 (reproduced to 0.07% twice), GPU mirror 1.5962e-6, shim path 1.3624e-6. The shim agreed to 3% for v1.1, which had no rotation — the divergence arrived with `eig(JJᵀ)`. **Steer offline with whest-parity runs or the GPU mirror, never the shim's absolute level**; paired comparisons within one path are still fine. The "grader runs ~20% better than local" note for v2.4 is therefore unexplained, not draw evidence.
- **`fnp.linalg.eigh` on JJᵀ is only well-determined in its leading directions** — LAPACK and cuSOLVER disagree on 106 of 256 columns beyond sign. The rotation's *distribution* is unaffected but the realised per-MLP error is BLAS-dependent, so expect ~1% of unexplained local-vs-grader difference from this alone.
- **RQMC helps the first moment only**: measured effective N 32,870 for `mean(z)` (2.01×) but 18,719 for the sample variance (1.14×). Any scheme leaning on sample *variances* gets no QMC discount.

## Older gotchas (from 2026-08-04, after submission 323872)

- **We are in PHASE 1 until Aug 7 23:59 UTC.** Rounds (from the challenge page): 1425 warm-up (closed), **1428 "Phase 1" Jun 18 → Aug 7 23:59 — LIVE**, 1429 "Phase 2" Aug 8 → Sep 19 23:59 — UPCOMING (empty leaderboard, no `v1-phase2` HF tag). Earlier docs said Phase 2 opened Aug 1; wrong. Phase 1 was **extended one week** (Jul 31 → Aug 7) after a leaderboard re-score; write-up deadline moved to **Aug 17** (forum 18109). Phase 2 uses the same grader stack.
- **Live board (Aug 4)**: #1 4.87e-8, #10 9.85e-8, #200 6.40e-7. The old "Phase 1 #1 = 1.24e-8" note is superseded — see RESEARCH_NOTES §5b.
- **Never call `flops.budget_summary_dict()`/`budget_summary()` inside `predict`** — O(op history), measured 157× residual blowup (forum 18092). Size sample counts analytically. Freeing large temporaries also bills residual wall time (18101), so prefer **few large chunks over many small ops**. Full list: RESEARCH_NOTES §3c.
- **The score is `V·c/B`** — per-sample variance × per-sample cost; N cancels above the 0.1 floor. Cheapening a sample is worth exactly as much as de-noising it. Pricing that matters: **f16/int8/bool are rate 1.0, same as f32 — there is no dtype trick**; matmul bills `(2K−1)·M·N` **by shapes only**, so pruning pays only via smaller *dense* matrices; gather/where 4/elt; transcendentals 16/elt (`norm.ppf` 168/elt); thin SVD 26n³ ≈ 0.16% B, eigh 9n³. Full table: RESEARCH_NOTES §3d.
- **α = 1.00 is theory, not a bug** — for discontinuous integrands the RQMC rate bonus is 1/(4d−2) ≈ 0.001 at d=256 (He & Wang 2015). Sobol buys constants only; rate recovery needs kinks concentrated into few coordinates.
- **Decision policy: nothing is decided on fewer than the full 100-MLP mini split.** Three "wins" evaporated on re-measurement (LMS scrambling, C/B=0.5, a +33% rotation).
- **The analytic family is retired** (GAME_PLAN rev 3): B1 fusion net ~0, B2 tail −67%…−214%, B3 learned residual R²=0.012.
- **Phase 1 config confirmed** by the graded report: width 256 × depth 32, n_mlps 100, B = 2.72e11, λ = 1.000e11 (derived exactly from per-MLP telemetry), multiplier floor 0.1. **`admin_config` is admin-only** — participants never see it; `evaluation.config_snapshot` + per-MLP telemetry is what we get. `wall_time_limit_s` remains unknown (only bound: ≥0.6 s, our max was 0.592 s with no timeouts). **Phase 2's config stays unverified until a graded run in round 1429 (Aug 8+).**
- Real flopscope API matches `estimator.py` exactly — `whest validate` passed with zero changes (`fnp.diagonal/diag/outer/einsum`, `flops.stats.norm.cdf/pdf`, `flops.as_symmetric(data, *, symmetry=(0,1))` all correct, and `as_symmetric` accepts our width-256 covariance). If a future edit trips a name/signature, fix `estimator.py` minimally without changing the math, mirror into the shim, and log it.
- Grader reports are **not** returned by `whest submit --json` (it gives only id/score/score_secondary) nor by the `/api/v1/submissions/{id}` endpoint. The full report lives in the submission page's `data-react-props` payload → `props.data.submission.evaluation`.
- `whest login`/submit 302 or permission errors usually mean challenge rules need (re-)accepting in the browser for the current phase — stop and ask Samuel. (Did not occur for the Phase 1 round.)
- v0 grader baseline to beat: adjusted **2.4927e-6**, `mean_compute_utilization` 1.45%, 0 failures. Local mini-split predicts the grader to within ~7%. If a future run is >3× off local expectation, investigate before resubmitting.
