# Phase-2 frontier audit — 2026-08-31

This is the dated evidence report behind the current `GAME_PLAN.md`. It is not a
second roadmap. Competition observations were read in Samuel's Chrome session and
are timestamped; rival implementation details are private; method attributions are
labelled as disclosures, inferences, or hypotheses. No constant was fitted on the
public split, no decision/vault block was touched, and no submission was made.

## Executive verdict

The repository has a strong evaluator and a weak shipped estimator. The current
Phase-2 ship path is basic K2 covariance propagation; graded v2.7 adds only scalar
calibration. On the live board v2.7 is rank **53/83**, below the displayed median,
and trails the leader by **24.6x adjusted** and about **61x raw**. Winning requires
a different state or quadrature architecture, not another scalar correction.

Two new facts change the research order:

1. The four most informative leader reports spend roughly `5.45e11`, `5.73e11`,
   `5.55e11`, and `9.22e11` counted FLOPs while reaching raw final-layer MSEs
   between `1.83e-8` and `2.93e-8`. The first cluster is consistent with about
   **16,384 dense forward directions**. Eight antipodal Kerdock/MUB bases have that
   exact leading cost, but so do 16,384 Sobol, lattice, orthogonal, or other dense
   samples. The telemetry supports the operating envelope, **not a private method
   attribution**.
2. Installed WhestBench 0.16 applies its `wall_time_limit_s = 120` default **inside
   each MLP's prediction**, not once across the 100-MLP suite. The old `~1.2 s/MLP`
   planning interpretation was false. The four inspected graded ledgers report
   roughly 61–105 seconds per MLP and pass, proving a server allowance of at least
   `104.766 s/MLP` but not its exact ceiling. The separate installed `0.4 s/MLP`
   residual cliff remains.

The most evidence-backed near-term route is therefore a submission-legal,
exact-radial structured carrier with a frozen basis/sample-count frontier, while
faithful augmented K2 and a new goal-directed K3 response are developed as the
second architecture. Full K3 remains the accuracy oracle. Exact full-tower W-EFF
and K2-prefix late-window K3 remain closed on their measured negatives.

Cost notation inherited from the experiments uses `B` as the competition budget:
`1.0732 B` means `1.0732 × 2**41 = 2.360e12` counted FLOPs, not 1.0732 billion.

## Live competition truth

Captured at approximately **2026-08-31 14:57 UTC**. The board is public feedback,
not the private fresh-suite prize result.

| item | live value |
|---|---:|
| registered participants | `831` |
| submissions across the challenge | `19,532` |
| Phase-2 leaderboard entries | `83` |
| J2W, rank 1 | `5.60e-9` adjusted / `2.25e-8` raw |
| Puffi, rank 2 | `5.90e-9` / `2.26e-8` |
| Activation-Regions, rank 3 | `7.40e-9` / `2.93e-8` |
| marius_binner, rank 4 | `7.70e-9` / `1.83e-8` |
| neopok919 v2.7, rank 53 | `1.374e-7` / `1.374e-6` |
| median position, rank 42 | `1.13e-7` adjusted |

The top ten are all at or below `1.06e-8` adjusted. Their raw errors mostly occupy
the narrow `1.8e-8`–`3.4e-8` band while their multipliers differ. This is a real
multi-team frontier, not one lucky row. Our target remains `<=3e-9` on fresh
offline evidence for private-rerun margin; merely matching today's public leader is
not enough.

The Phase-1 board now shows its completed standings. At this capture the first
three displayed `1.84e-8`, `5.39e-8`, and `6.22e-8` adjusted; this repository's
Phase-1 entry is rank 104. Phase-1 methods and rankings are useful evidence but are
not a Phase-2 validation tier.

The challenge Insights page displayed a stale best score (`3.91e-8`) while the
authoritative Phase-2 leaderboard displayed `5.60e-9`. Use the round-specific
leaderboard and individual submission reports for live score/cost claims.

Primary pages:

- [challenge overview](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026)
- [Phase-2 leaderboard](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards?round=phase-2)
- [Phase-1 leaderboard](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards?round=phase-1)
- [rules](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/challenge_rules)
- [resources](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/dynamic_contents)
- [insights](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/insights)

Local manual transcriptions:

- `experiments/board_snapshots/2026-08-31T1457Z_phase2_frontier_board.md`
- `experiments/board_snapshots/2026-08-31T1457Z_phase2_frontier_submission_reports.md`

### Rules and dates that matter

- Phase 2 closes **2026-10-17 23:59 UTC**; team freeze is **2026-10-02**;
  contribution write-up is due **2026-10-24**.
- Daily allowance is 10 submissions per team per UTC day.
- Public grading uses 100 MLPs, with 50 contributing visible feedback and 50 sealed.
- Prizes use a fresh private rerun of up to two designated submissions. With none
  designated, Rules v14 selects the two best-scoring valid public entries.
- Live shape is `1024 x 16`; `B=2**41`; lambda is zero; score multiplier is
  `max(0.1, F/B)`.
- Participant code has one CPU core / two vCPUs; the numerical backend has seven
  cores / fourteen vCPUs. Installed/run-config defaults are 8 GiB memory, 5-second
  setup, **120-second wall per MLP**, and **0.4-second residual per MLP**. Graded
  config fields are sealed; public telemetry proves only the `>=104.766 s` wall
  lower bound.
- Phase-2 code may use Python stdlib, `flopscope`, and `whestbench`; numerical work
  through bundled NumPy/SciPy/BLAS, FFI, subprocesses, threads, or uninstrumented
  arrays is not allowed.

## What public submission telemetry does and does not reveal

| submission | adjusted | raw final MSE | mean compute | interpretation |
|---|---:|---:|---:|---|
| [J2W #329139](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/329139) | `5.580e-9` | `2.251e-8` | about `5.45e11` | approximately a 16k-direction dense-forward envelope |
| [Puffi #329181](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/329181) | `5.892e-9` | `2.262e-8` | about `5.73e11` | same broad envelope, higher overhead/cost |
| [Activation-Regions #328901](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/328901) | `7.404e-9` | `2.933e-8` | about `5.55e11` | final-only optimization; early-row error is much worse |
| [marius_binner #329126](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/329126) | `7.685e-9` | `1.833e-8` | about `9.22e11` | best raw of these four, but cost loses the adjusted trade |

Dense propagation of `16,384` rows through sixteen `1024 x 1024` layers costs
`2**39 = 5.497558e11` FLOPs before auxiliary work. That is also eight Kerdock/MUB
bases when each of 1,024 vectors is paired with its antipode. The J2W mean is close
enough to make this an important clue, but slightly below the leading arithmetic
and affected by per-MLP averaging/rounding. It cannot establish an exact row count,
point family, antithetic policy, or analytic branch.

Puffi publicly disclosed that its Phase-1 score system mixed analytic and QMC
methods and used Kerdock frames. That disclosure supports the family, not an
attribution of any Phase-2 entry. Current frontier code remains private.

The account has exactly the two known Phase-2 submissions:

| id | artifact | adjusted | raw | failures |
|---|---|---:|---:|---:|
| [328662](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/328662) | v2.6.1 K2 canary | `3.933e-7` | `3.933e-6` | `0` |
| [328664](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/328664) | v2.7 calibrated K2 | `1.374e-7` | `1.374e-6` | `0` |

A public failed submission exposed a `_kprop/wick.py` import error, which shows at
least one other participant explored Wick/cumulant machinery. It is not evidence
that a successful frontier entry uses that code.

## Latest discussion-board knowledge

The current discussion index has 29 topics. The most decision-relevant recent
threads are:

1. [Residual synchronization investigation](https://discourse.aicrowd.com/t/shipped-covariance-propagation-example-trips-the-phase-2-residual-cap-locally-0-16-0-but-grades-fine-what-happens-when-the-evaluator-upgrades/18202).
   `bool()`, `float()`, `int()`, implicit conditions, assertions, and similar
   materializations of flopscope values can force graph synchronization. Removing
   166 such points changed a five-failure run into a clean run with the same math,
   FLOPs, and roughly the same call count. New candidates require zero mid-predict
   scalar materializations. Very large stacked 3-D operands also produced an
   unresolved generic submission failure; avoid relying on them before a smoke test.
2. [Moment-matched sampling](https://discourse.aicrowd.com/t/phase-1-write-up-variance-reduction-beats-closure-refinement-a-moment-matched-sampling-estimator-submission-324497/18201).
   Antipodes and covariance whitening improved its sampler; exact Cho–Saul/Price/
   Drezner off-diagonal refinements did not. This supports variance-reduced sampling
   and confirms that improving the Gaussian pair map is not enough.
3. [Phase-2 complementary blend](https://discourse.aicrowd.com/t/phase-2-write-up-complementary-errors-beat-either-branch-a-fixed-0-75-0-25-covariance-whitened-antithetic-mc-blend-submission-328274/18200).
   A frozen covariance/sampler blend improved adjusted score by 24.85% with low
   residual-error correlation. The absolute score is not frontier, but the
   complementary-error mechanism is relevant after a strong carrier exists.
4. [The easy-leaderboard effect is gone](https://discourse.aicrowd.com/t/unusually-easy-leaderboard-set-effect-is-gone/18199).
   Phase-2 covariance-propagation difficulty appears ordinary rather than
   heavy-tailed. This agrees with the repo's unbiased public-mini transfer result.
5. [Phase-2 launch](https://discourse.aicrowd.com/t/phase-2-of-the-arc-white-box-estimation-challenge-is-live/18197).
   Confirms the official stack and intended clean-compute policy.
6. [Public Phase-1 frontier synthesis](https://discourse.aicrowd.com/t/phase-1-solution-ideas-for-phase-2-and-the-honest-public-frontier-all-methods-combined-and-explained/18193).
   The corrected discussion stresses that low participation ratio does not imply a
   low-dimensional error subspace; 99% of centered variance still required many
   modes. This supports query-directed corrections over generic rank truncation.

The newest topic asks Claude/GPT systems to reveal solutions. It is not technical
evidence and does not license sharing this repository's private benchmark or code.

## Repository and evaluator audit

Read-only checks began at clean HEAD `dff59df`; the recovery rerun occurred after
the audit documents were written, so the executable tree stayed unchanged while
the overall worktree was expectedly dirty with this documentation diff:

- default preflight: **GREEN 12/16**;
- official configuration: `73/73`;
- width-1024 mint parity: `20/20`;
- canonical score tests: `98/98`;
- lifecycle tests: `56/56`;
- official-package FLOP anchor: 19 operations, delta zero at
  `137,600,326,656` FLOPs;
- ship/archive identity: v2.6.1 exact;
- portability: `99/99` after removing one redundant pointer that exceeded the
  `AGENTS.md` size budget.

The four full-only stages were not rerun and must not be described as current
passes. The last stored full transcript is green from 2026-08-27; code since then
is unchanged, but that is provenance, not a new execution.

### Corrected wall semantics

Installed `whestbench.scoring.evaluate_estimator()` constructs a fresh
`BudgetContext` inside `for i, mlp in enumerate(data.mlps)`. The subprocess worker
enforces the same installed wall limit around that one `predict()` call and reports
it to the host. Therefore the installed `wall_time_limit_s=120` is **per MLP**. The
previous active-doc claim that 120 seconds was shared across all 100 networks was
false. Graded telemetry proves `104.766 s/MLP` passes but does not expose the exact
server setting.

Consequences:

- against the installed 120-second default, v2.7's roughly 0.92 seconds/MLP has
  about 130x headroom, not a nearly exhausted 100-MLP suite allowance; the public
  server evidence alone establishes only about 114x from its `>=104.766 s` lower
  bound and does not reveal the ceiling;
- the J2W/Puffi 65–91 second/MLP reports are legal and explain how dense 16k-row
  systems can grade;
- the `0.4 s` residual cap remains separate and is about Python glue,
  materialization, and non-backend time;
- W-EFF remains killed by the score multiplier, not by wall time;
- late-window K3 remains killed by its accuracy bound, not by wall time.

### Dataset-tag drift

`v2-phase2` moved from commit `ce926b7…` to `6a416f…`, so
`local_sim/phase2_checklist.py` now warns. A read-only comparison found all 70
source `data/*.parquet` files byte-identical and the README unchanged; 153 prepared
files were added and top-level `metadata.json` gained `prepared_splits`. Existing
mint/CONFIRM bytes remain valid. The exact 70-row LFS object/size comparison and
canonical manifest digest are preserved in
`experiments/2026-08-31_phase2_tag_drift/source_manifest.tsv`, rather than depending
on an agent transcript. However,
the pinned manifest/checklist is stale. Repairing that IMPORTANT instrument requires
its own prediction, committed contract, preflight, knowledge delta, and independent
Skeptic; this routine audit does not silently change it.

### Remaining infrastructure limits

- Windows still cannot enforce the 8-GiB virtual-address-space limit.
- The grading server still adds an unexplained `+2**24` FLOPs to v2.6.1; it exposes
  no operation ledger.
- Local residual seconds are not transferable to a new candidate.
- No Phase-2 vault, disjoint DESIGN-replication block, or SINGLE_USE decision block
  exists.
- Preflight certifies `estimator.py`, not v2.7 or arbitrary research candidates.
- Two instrument docstrings remain stale: `p2_window.py` states the wrong triangle
  bracket, and `competition_score.py` describes the geometric statistic as binding.
  Executable behavior is correct; changes require the IMPORTANT workflow.
- Preflight's descriptive subtest counts still say `44/44` seed-ledger and `54/54`
  lifecycle while the executable suites now return `60/60` and `56/56`;
  `VALIDATION.md` also assigns the `60/60` output to the wrong numbered stage.
  These are queued with the other IMPORTANT instrument-documentation repairs.

## Literature synthesis

### Established

1. ARC's cumulant propagation directly targets the quenched problem—one realized
   weight draw—and is the right higher-order reference. Its AUGMENT recurrence is
   the most supported compact extension of K2. [ARC paper](https://arxiv.org/abs/2605.05179),
   [reference code](https://github.com/alignment-research-center/mlp_cumulant_propagation).
2. Exact Hermite/Price Gaussian covariance maps improve the nonlinear readout only
   under a Gaussian incoming joint law. They cannot restore discarded non-Gaussian
   state. [Wright et al.](https://proceedings.mlr.press/v238/wright24a.html).
3. Finite-width cumulant/diagram theory supports layer-local sources followed by
   linear response, but most constants are ensemble/annealed rather than valid for
   a fixed network. [Hanin](https://jmlr.org/papers/v25/23-0643.html),
   [Yaida](https://proceedings.mlr.press/v107/yaida20a.html).
4. Kernel quadrature turns prior-averaged squared integration error into a kernel
   discrepancy. It is useful to screen point sets, but an infinite-width ReLU
   kernel cannot by itself certify finite-width, network-specific ranking.
   [Briol et al.](https://proceedings.mlr.press/v70/briol17a.html),
   [Cho and Saul](https://proceedings.neurips.cc/paper/2009/hash/5751ec3e9a4feab575962e78e006250d-Abstract.html).
5. Kerdock/MUB geometry provides large deterministic collections of mutually
   unbiased bases. The repo's width-1024 construction satisfies the defining Gram
   properties, but a fixed small prefix is not an exact spherical 4-design.
   [Calderbank et al.](https://doi.org/10.1112/S0024611597000403).

### Highest-value new hypothesis: query-directed adjoint K3

Full K3 proves third-order information is valuable, but transporting the complete
state is too expensive. Linearize the higher-order correction around the
K2/AUGMENT trajectory:

```text
delta m_L  ~=  sum_l  J_(L <- l)  S_l^(3)
```

Back-propagate adjoints for the final outputs (or a fixed hard-neuron subset) and
contract each locally born third-cumulant source directly into those queries. This
attacks the state damage that killed a late K3 window without retaining all K3
lineages.

This is a mechanism-backed research hypothesis, not an established algorithm.
Small-shape tests must compare it layer by layer with exact K3, measure explained
squared drift, check a source-scaling linearity falsifier, and price the real
flopscope spelling before any DESIGN truth.

### Other bounded hypotheses

- faithful float32 AUGMENT-K2;
- a two-component neuron-specific skew mixture driven by AUGMENT trace state;
- adjoint-weighted sampling of exact-K3 lineages, with unbiased weights;
- randomized signed-permutation/Hadamard rotations of antipodal MUB blocks;
- active-direction preintegration for a small fixed hard-neuron set;
- antipodal simplex blocks screened against matched MUB/orthogonal controls.

Generic EP sweeps, activation-region enumeration, more Gaussian Price terms,
global scale mixtures, generic Frobenius tensor compression, and a plain
TensorSketch of dense `W^(x3)` remain low value or structurally mismatched.

## Recommended architecture

Develop two genuinely different nominees.

### Nominee A — structured angular carrier

```text
stdlib/fnp Kerdock-MUB or matched structured generator in setup
  -> exact Gaussian radial factor
  -> nested 4/6/8/10/12-basis full-network propagation
  -> per-layer sample means (final row is the score target)
  -> optional frozen low-correlation K2 blend
  -> scalar calibration only if it adds held-out value
```

Controls must include matched-row IID, high-dimensional scrambled Sobol,
random-orthogonal blocks, and at least one randomized carrier orientation. Basis
count is selected by paired **adjusted score**, not raw MSE. The 8-basis clue is a
frozen panel point, not a fitted answer.

### Nominee B — compact analytic response

```text
corrected float32 calibrated K2
  -> faithful unfactorized AUGMENT-K2 state
  -> query-directed/adjoint K3 response
  -> optional neuron-specific skew mixture
  -> tiny permutation-equivariant rolled-forward correction
```

Full K3 is the accuracy oracle and diagnostic only. A final K3 window does not
return unless a better prefix supplies new external/mechanistic evidence and a new
contract explicitly reopens it.

## Execution order

1. **Routine continuity repair now.** Refresh active docs, board evidence, wall-cap
   semantics, and the Builder handoff. Do not alter historical result entries.
2. **IMPORTANT carrier deployment closure.** Build a stdlib/`fnp` constructor that
   is byte-identical to the NumPy reference for every supported `k`; validate fresh
   worker setup, zero scalar sync, memory accounting, and dense propagation parity.
3. **IMPORTANT carrier objective.** Freeze the truth-free kernel/objective and a
   carrier panel, but require it to predict finite-width ranking before using it to
   select policies.
4. **IMPORTANT evidence-block allocation.** Once truth-free results fix the power
   plan and before any labels, register a disjoint Phase-2 DESIGN-replication block
   and a separate SINGLE_USE DECIDE block; test the ledger, run default preflight,
   record the knowledge delta, and obtain an independent Skeptic. P01 cannot serve
   as its own replication and no decision block exists today.
5. **STRATEGIC carrier DESIGN.** After a Rule-1 contract, Samuel's prediction, and
   commit, evaluate the nested basis/sample frontier on a fresh DESIGN block using
   paired arithmetic score differences and measured total FLOPs. Replicate the
   deterministic choice on a disjoint DESIGN block.
6. **Parallel IMPORTANT analytic prototypes.** Faithful AUGMENT-K2 small-shape
   parity and a truth-free adjoint-K3 response prototype. Price before truth.
7. **STRATEGIC analytic DESIGN.** Test only parity- and price-cleared state/response
   arms on fresh DESIGN data; keep standalone-state and checkpoint estimands
   separate.
8. **Integrated SINGLE_USE decision.** After successful disjoint DESIGN replication,
   freeze all architecture, coefficients,
   carrier policies, costs, fallbacks, and two nominees before claiming a new
   Phase-2 decision block.
9. **Deployment.** Full candidate promotion into `estimator.py`, `preflight -Full`,
   three official-subprocess soaks, zero scalar materializations, Linux 8-GiB test,
   public CONFIRM, exact hashes, independent Skeptic, then explicit Samuel approval.

The live-shape `k3pre` attribution confirmation is scientifically useful but no
longer blocks these two candidate-producing tracks. Run it when a quiet ~30-GiB GPU
is available; do not let it delay the carrier constructor or AUGMENT parity.
