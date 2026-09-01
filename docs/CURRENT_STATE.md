# CURRENT_STATE.md — read this first

**Snapshot: 2026-08-31, updated after the WP0/WP1/WP5 engineering session.
Phase 2 is live, this account has two clean graded submissions, v2.7 is rank
53/83, full factorized K3 remains the accuracy oracle, and W-EFF and late-window
K3 are closed.**

**What changed this session, and it is all truth-free engineering.** WP0A–WP0D
are closed: the dataset pin now distinguishes a source-byte change from a
metadata-only tag move (and knows the loader prefers a `prepared/` Arrow tree),
the gate's stated coverage is scanned exhaustively and runs inside preflight,
any candidate file can be certified read-only without touching `estimator.py`,
and **the 8,192 MB limit is enforced for real on Linux for the first time —
`3.49x` headroom, not the `8.9x` Windows accounting suggested.** WP1 and WP5 are
closed: a **grader-legal Kerdock/MUB carrier, byte-identical to the frozen
reference at width 1024**, plugs into one shape-generic candidate API and builds
to a self-contained single file that is **CERTIFIED 36/36 at `1024 x 16`** with
zero failed MLPs on the official subprocess runner.

**The one accuracy datum that exists points AGAINST the branch, and an earlier
version of this paragraph wrongly said none existed.** The certification's
deployment smoke reads the organisers' `v2-phase2` mini split, so a real MSE was
produced. On the **same two organiser networks**, paired:

| | raw final MSE | multiplier | **adjusted** |
|---|---:|---:|---:|
| shipped v2.6.1 canary | `4.778624e-06` | `0.100000` | `4.778624e-07` |
| **carrier k=8** | `2.710315e-06` | `0.250125` | **`6.779175e-07`** |

Raw the carrier is **`1.763x` better**; break-even needs **`2.501x`**; adjusted
it is **`1.42x` WORSE than the canary**. **`n = 2` decides nothing** — max/median
over fresh MLPs is ~`12.9x` — and no GT-noise floor is quoted, so this is
neither a kill nor a signal. Nothing was selected on it; CONFIRM stays read-only.
It is recorded because the alternative is a durable record that says the question
is unmeasured while the first measurement sits in a committed artifact.

**The next gate is WP2**, the truth-free objective and the frozen nested basis
panel, because `k = 8` is one frozen panel point and not a selection. The
Skeptic's recommended discriminator for the break-even question is a paired
DESIGN panel on 24–32 fresh mints (P01 has 432 seeds left) through
`competition_score.decide()` — **STRATEGIC, needs a Rule-1 contract and Samuel's
prediction, and explicitly NOT more mini rows.**

This file says where execution stands;
[`GAME_PLAN.md`](GAME_PLAN.md) is the only active architecture and implementation plan;
[`VALIDATION.md`](VALIDATION.md) is binding for measurement; the newest entries at
the bottom of `experiments/LOG.md` contain the evidence narrative. The dated
competition/repository/literature evidence is
[`PHASE2_FRONTIER_AUDIT_2026-08-31.md`](PHASE2_FRONTIER_AUDIT_2026-08-31.md).

## One-minute orientation

- Phase 2 closes **2026-10-17 23:59 UTC**. Team/registration freeze is
  **2026-10-02 23:59 UTC**; contribution write-up is due **2026-10-24 23:59 UTC**.
- Shape is `1024 x 16`; budget is `2**41`; multiplier floor is `0.1`.
- Official stack is WhestBench 0.16 / flopscope 0.12. Lambda is zero. Its installed
  defaults are a 0.4-second residual cap and 120-second participant wall cap per
  MLP, plus 8 GiB memory; graded reports do not expose those exact config fields,
  while public ledgers prove that at least `104.766 s/MLP` passes.
- The live Phase-2 board had **83 entrants at 2026-08-31 14:57 UTC**. J2W led
  at displayed `5.60e-9` adjusted / `2.25e-8` raw with multiplier about `0.248`.
  `neopok919` v2.7 was **rank 53/83** at `1.374e-7` / `1.374e-6`: `24.6x`
  behind in adjusted score and about `61x` in raw MSE. Rank 42 (the median
  position) displayed `1.13e-7`, so this account is below the public median. This is a live-board
  observation, not a promotion dataset. Raw snapshot:
  `experiments/board_snapshots/2026-08-31T1457Z_phase2_frontier_board.md`.
  Older figures below remain dated historical observations; never quote a board
  number without its timestamp.
- **The `v2-phase2` tag moved from `ce926b7...` to `6a416f...`, and WP0A closed
   it on 2026-08-31.** A read-only comparison of **all 226 objects at both
   commits** shows: 70 source Parquet **unchanged**, README and `.gitattributes`
   unchanged, **`metadata.json` changed**, **153 `prepared/` objects added**,
   **nothing removed**. Both records are content-addressed and pinned in code
   (`experiments/2026-08-31_phase2_tag_drift/{source,loader}_manifest.tsv`,
   digests `3cb3e819...` and `29664dbb...`). The manifest is repinned to
   `6a416f...` with the old commit preserved in `dataset.commit_lineage`, and
   `local_sim/test_p2_tag_drift.py` now runs inside preflight step 2.
   **THE LOADER ROUTE MOVED, and the first version of this entry did not know
   it.** `whestbench 0.16.0` PREFERS a `prepared/<split>` Arrow tree over
   `data/*.parquet` whenever `metadata.json` advertises `prepared_splits`
   (`dataset.py:479-491`, `dataset_io.py:61-68`), and `whest run` is
   non-streaming by default — so **an official Phase-2 CONFIRM today reads the
   prepared tree, not the parquet**. On the one shard compared
   (`experiments/2026-08-31_wp0a_manifest/prepared_vs_parquet.json`) the two
   routes agree exactly: same schema, same 8 rows in the same order, `weights`
   and `all_layer_means` **bitwise identical**. **13 of 14 `mini` shards and the
   whole `full` split are still unverified.** Stored Phase-2 CONFIRM bytes were
   produced from the parquet path before the prepared tree existed and remain
   valid; a *new* CONFIRM takes the other route.
- **TWO PHASE-2 SUBMISSIONS EXIST AS OF 2026-08-27, both graded, both with zero
  failed MLPs.** This supersedes the "zero Phase-2 submissions" line that stood
  here on 2026-08-26.

  | id | artifact | sha256 | graded score |
  |---|---|---|---:|
  | **328662** | `estimator.py` = v2.6.1 (the canary) | `00240eb1…` | **`3.9330104345935976e-07`** |
  | **328664** | `estimator_v2_7.py` | `96ff522d…` | **`1.373807110667258e-07`** |

  Graded improvement `2.8629x`, against the local paired statistic's `2.9078x` —
  **agreement to 1.5%, so the offline DESIGN fitting loop is confirmed on the
  board and not merely in projection.** Reports:
  `experiments/run_reports/submission_3286{62,64}_*_graded_report.json`.

  **The telemetry these bought is the real prize** (full detail at the bottom of
  `experiments/LOG.md`):
  - **grading-server cost offset `+0.0122%`** — local official-package runner
    `137,600,326,656` vs graded
    `137,617,103,872` per MLP. `C == F` on all 100 rows on the real grader.
    The local 19-operation anchor is exact against the stored official-package
    subprocess report, **not** against the grading server's total; no graded
    operation ledger exists from which to attribute the `+2**24` offset.
  - **score parity: there is NO measurable bias. CORRECTED 2026-08-27, and the
    first version of this line was wrong.** The graded submission set and the
    public `mini` split share **zero** networks — they are two disjoint 100-MLP
    draws from the same distribution. The apparent `3.0%` (canary) and `1.4%`
    (v2.7) gaps are **`0.90 sigma` and `0.57 sigma`** of the pooled sampling
    error. **The local mini is an UNBIASED proxy for the graded score**; quote it
    with its sampling se (`1.3–1.8%` of the score) and **apply no correction
    factor** — doing so is fitting noise. Corroborated: an independent numpy K2
    closure scores the public 100 at `4.0506e-06`, matching the local `whest`
    mini's `4.050566e-06` to five significant figures, so the instrument
    reproduces exactly and only the network sample differs.
  - **residual: graded median `6.78`/`7.04 ms`, max `7.06`/`7.31 ms` → `54.7x`
    headroom** against the `0.4 s` cliff. The grader is `7.06x` faster than this
    host at the median for these files. `AGENTS.md` records this observation and
    correctly refuses to use it as a safety factor for a different candidate.
  - **setup `2.66–2.92 ms` across 5 workers, all SUCCESS → `1,712x` headroom**
    for these two submitted files. This proves their setup safety; it does not
    identify the cause of earlier local timeouts or clear a future candidate.
  - **feedback shape confirmed: 100 telemetry rows, 50 public score rows
    (indices 0–49), rounded to 3 significant figures.**
  - `sampling_mse = 1.133e-06` is the organiser's MC reference. v2.7 is within
    `1.21x` of it; the canary was `3.47x` off.
- **The residual-cliff worry is resolved for the two submitted K2 files.** Local
  measurement said v2.7 cost `+40.3 ms` over the baseline; the grader measured
  `+0.260 ms`. The reports do not identify the cause of that disagreement.
  Local residual deltas on this host are therefore not authoritative for a new
  architecture.
- **Open item created by the graded data:** the two files differ by `32,832`
  counted FLOPs locally and **`32,800` on the grader**. The 32-FLOP delta
  discrepancy is observed only in totals; no grading-server operation ledger
  exists to attribute it to an operation. It is `2.3e-10` of the total and changes
  no decision, but it belongs in graded-total regression fixtures.
- The current shipped file routes Phase-2 width to basic `k2p1`. Its official
  100-MLP subprocess mini measured `4.050566e-7` adjusted / `4.050566e-6` raw,
  with 0 failures, fixed `137,600,326,656` FLOPs/MLP and **4.29x LOCAL,
  one-session** worst-case residual headroom. Its graded canary is rank 57 at the
  current snapshot. It is a safe
  calibration baseline, not a materially improved Phase-2 system.
- The repo's scientific decision protocol is strong. P2-P0 repaired local
  deployment parity: the local meter reproduces the stored official-package
  subprocess runner op by op, to zero. The grading server still adds the small
  unexplained offset above; do not conflate those two claims.
- **Phase 0 is CLOSED.** Its all-of gate G1–G7 is met, with a literal independent
  **ACCEPT** on reviewed HEAD `952bd61`, recorded in closure commit `7367fdb` at
  the bottom of `experiments/LOG.md`.
  It took four review passes — PROVISIONAL, PROVISIONAL, REJECT, ACCEPT — and the
  first three each found something real, including a gate clause satisfied by a
  check that could not fail.
- **P1-PRE is CLOSED on a literal independent ACCEPT** (seventh pass, reviewed
  HEAD `6d05d77`). D1, D2 and D3 are closed; so are the six review-5 findings and
  the four review-6 blocking items. 15 mutations fire their named guard, the
  harness cannot write the repository (containment via `realpath`, measured after
  the write), preflight is 16/16 and portability 99/99.
  **Seven passes: PROVISIONAL x6, then ACCEPT — and five of the six found the
  defect in the repair rather than in the original defect.** Recorded limitations,
  none a false green today: the Linux branch and the child-`setrlimit` Linux
  answer have never executed, and the five design observations of the sixth pass
  were deliberately not pursued.
- **Track 0S is CLOSED — it shipped.** Its candidate is graded submission
  `328662` at `3.9330104345935976e-07`, zero failures; the gate went GREEN 16/16 at
  HEAD `4a4f575d` and Samuel's approval named the artifact. The paragraph below is
  retained as the historical record of how it got there.
  *(Historical, 2026-08-26:)* The
  frozen K2p1 candidate
  (`00240eb1e60810f5…`, byte-identical to `estimator_v2_6_1.py`) scores
  `4.050566369642183e-7` adjusted on the official 100-MLP subprocess mini with
  **zero failures**, fixed `137,600,326,656` FLOPs/MLP and a floor-pinned
  multiplier. The GT-noise floor is recorded (`7.47823932957716e-11`, 0.001846% of
  raw MSE, so the result is **54,165× above** the truth's own noise); the
  account/allowance evidence is preserved with **transcription honestly
  distinguished from capture**; the submission contract and **Samuel's prediction
  (`3.6e-7`, no failures, transfers)** are committed *before* any result. Two
  independent passes ran against an earlier tree, and every finding of both is
  closed.
  **The stale-witness defect is now FIXED:** `preflight -Full` is
  **GREEN 16/16 witnessed at HEAD `576ad46`** (transcript
  `experiments/2026-08-25_trackA/preflight_full_trackA.txt`, which records that
  SHA in its own header), committed as the single evidence commit `240f8df`, so
  the submitting HEAD differs from the witnessed HEAD **only by that commit** —
  nothing under `local_sim/`, `scripts/` or `estimator.py` differs. That is the
  shape the contract's reworded gate item 6 requires.
  **Still outstanding before approval may even be requested, and both are
  blocking:** (1) a **live signed-in recheck** of the account state and the
  10/team/UTC-day allowance — the committed figures are honestly labelled
  *transcribed, not captured*, and this session had no browser or credentials;
  (2) a **focused independent ACCEPT** on the corrected record. **Do not request
  approval or submit until both exist.**
- **P2-CAL (2026-08-26) measured a 2.96x accuracy gain for ~zero FLOPs, and its
  own falsifier fired.** Calibrating the K2 state improves held-out final-layer
  MSE from `4.141e-6` to `1.398e-6` on 96 DESIGN networks (two `2**22` replicas,
  GT noise `1.8e-8`), for `16,384` counted FLOPs against a `2.199e11` free zone —
  the multiplier stays pinned at the `0.1` floor. **But the in-loop, global and
  post-hoc arms are identical to twelve significant figures**, because the K2
  closure is exactly positively homogeneous of degree 1 in its state: scaling
  `(mu, cov) -> (g*mu, g^2*cov)` leaves `alpha` unchanged, so every downstream
  mean is scaled by exactly `g` (verified at `1.6e-30`). **Every purely
  multiplicative state correction is one output rescale in disguise**, whatever
  its parameter count. The contract's prediction 4 is REFUTED, the pre-registered
  falsifier fired, and because the continuation gate was conjunctive on it,
  **the gate is NOT met and no DECIDE block is authorised.** The 2.96x is real
  and holds on both replicas; turning it into a submission needs a NEW contract
  with a new gate and Samuel's prediction taken first. The remaining power is in
  corrections that change `alpha` and therefore do not commute — those are the
  live question, not this one.
- **P2-CAL part 2 closed that follow-on the same day: per-layer scalar
  calibration is EXHAUSTED at 2.96x.** The two corrections that genuinely change
  `alpha` and therefore do not commute — a per-layer gain on the post-ReLU
  **variance** (registered `C3`) and on the **off-diagonal covariance**
  (exploratory `C4`) — both fitted about 1% in-fit and **lost all of it out of
  sample**: held-out `1.0088x` and `0.9932x` against the scalar arm, **both
  intervals straddling zero**. The knobs are confirmed live (a `0.9` variance gain
  moves the scored row 12x more than the residual error), and both scans correctly
  reported layer 15 as flat, which it provably is. So the `~1.4e-6` residual is
  **not a per-layer scalar phenomenon at all**. The next correction must act
  **per neuron** rather than per layer, or **add information** (higher cumulants,
  sampled directions) rather than re-weight what K2 already computes — and option
  two costs FLOPs by construction, which is where the whole live top 12 sits.
- **WITHDRAWN, same day: "the free zone is a trap."** A board-forensics entry
  concluded that the `0.100` multiplier floor cannot be competitive. The
  measurement was right (33 of 57 entrants are floor-pinned; the best is rank 16)
  but the inference was wrong, and the strategy review rejected it correctly. That
  is a fact about today's *population*, not about the score identity. The floor
  caps **price**, not accuracy: `raw 3.0e-8` delivered at `F <= 0.1B` scores
  `3.0e-9` and wins outright. The counter-evidence was in the same entry — Puffi
  moved multiplier `0.3065 -> 0.2698` holding raw flat, i.e. **descending toward
  the floor**. The floor is the DESTINATION of the compression race.
  **Winning target, adopted: adjusted `<= 3e-9`** — raw `<= 3e-8` at `<= 0.1B`, or
  raw `<= 1.5e-8` at `<= 0.2B`. Puffi's own floor-limit, if their raw held, is
  `3.150e-9` (verified live, 2026-08-26 23:12 UTC).
  One review figure needed correcting in return: the best **raw** among
  floor-pinned entries is `7.956e-7` (best adjusted `7.960e-8`), not the `5.95e-8`
  the review quotes — a raw of `5.95e-8` at the floor would score `5.95e-9` and
  lead the board. The conclusion is unaffected; the honest distance from today's
  free-zone population to a winning free-zone entry is **`26.5x` in raw**.
- **Track 0S blocker (1) is half closed.** The account state is now **captured,
  not transcribed**: an authenticated `whoami` + `eligibility` call for
  `neopok919` (id 513543) returned `submissions_allowed: true`,
  `rules_accepted: true`, `participation_terms_accepted: true`, Phase 2 active
  until `2026-10-17T23:59Z`
  (`experiments/2026-08-26_p2_live/account_eligibility_capture.json`).
  **The `10 per team per UTC day` allowance is STILL NOT captured** — no reachable
  AIcrowd API exposes a per-day count or balance and the rules page renders
  client-side, so that half stands and needs a browser session.
- **HISTORICAL PRE-SUBMISSION RECORD for P2-SHIP-CAL.** v2.7 has since been
  submitted as #328664, graded `1.373807110667258e-7`, and completed with zero
  failures. The remaining indented text records the evidence and blockers as they
  stood before submission; it is **not a current action list**. `estimator_v2_7.py`
  (sha256 `96ff522d…`) = v2.6.1 plus sixteen
  literal output-row constants. Official 100-MLP subprocess mini:
  **`1.3929833e-7` adjusted / `1.3929833e-6` raw, 0 failures, multiplier `0.1`,
  `137,600,359,488` FLOPs** — a **`2.9078x`** improvement on the frozen K2p1
  baseline's `4.050566e-7`.
  **The binding statistic, which this entry originally omitted** (a bare ratio of
  means is what `VALIDATION.md` §2b-i forbids by name): paired arithmetic score
  difference **`2.657583e-07`**, se `7.2202e-09`, **`t = 36.81`**, studentised-t
  95% CI **`[2.514319e-07, 2.800847e-07]`**, basic bootstrap
  `[2.516508e-07, 2.797927e-07]`; **100/100 MLPs improve, zero regressions**;
  geometric-mean diagnostic `2.8780x`, per-MLP range `[1.5576, 4.5069]`. The
  **layer-0 control is bit-identical on 100/100 rows** (`_CAL_G[0]` is pinned to
  1.0), which establishes determinism, exact pairing and environmental stability
  in one measurement.
  **"All four pre-registered gate clauses pass" overstated what was tested.** An
  independent review found that only **G3** carries information: G4's first half
  cannot fail (the change is `2.4e-07` of budget; unpinning the `0.1` floor would
  need ~1.6x more FLOPs), G4's second half is redundant with G2, and G1
  (`whest validate`) exercises depth 2 and never reaches `_analytic_only`, so it
  is blind to the intervention. **G3 is genuine and passed by a wide margin**, so
  the substance stands.
  **The transfer hypothesis is confirmed and it is the real result:** the constants
  were fitted on freshly minted DESIGN networks and had never seen an organiser
  network, yet the DESIGN projection (`2.962x`) and the public measurement
  (`2.908x`) agree to **1.9%**. That licenses the offline fitting loop the whole
  Phase-2 plan rests on. The runner also confirms `C = F` directly:
  `mean_effective_compute == flops_used` exactly, on all 100 rows, with
  `lambda_flops_per_second = 0.0` in `run_config`.
  Residual: median `86.9 ms`, max `262.1 ms`, **0/100 over the `0.4 s` cap**
  (**`1.53x` LOCAL** headroom at the worst row, on a contended host — thin, and
  not convertible into a grader second).
  **THE RESIDUAL CLIFF IS WORSE THAN THIS PARAGRAPH USED TO IMPLY, and it is now
  the top deployment risk.** Read from `whestbench/scoring.py:898-912, 955-973`:
  an overrun row is **zeroed AND has its multiplier forced to `1.0`**, so it
  scores `mean(target^2) ~ 0.92` instead of `~1.4e-7`. One such row moves a
  100-MLP mean from `1.393e-7` to about `9.2e-3` — a **~66,000x loss**. It is not
  a penalty, it is annihilation of the submission.
  Against that: the **frozen baseline v2.6.1 is far safer than the candidate** on
  the same 100 paired networks — median `47.9 ms`, p90 `54.4`, max **`93.3 ms`**
  (**`4.29x` LOCAL**), zero rows past 150 ms, against the candidate's six rows
  past 150 ms and two past 200 ms. The paired delta is **`+44.3 ms` mean /
  `+40.3 ms` median / `2.81x` at the max on identical networks** — from a counted
  FLOP delta of `1.49e-08` of budget, which **cannot** cost 40 ms. So the `+44 ms`
  is host contention or something the FLOP counter cannot see, and the two runs
  were never executed back to back, so **the existing evidence cannot tell those
  apart.** The margin is smaller than the repository's own documented `2.35x`
  session-to-session residual spread.
  Lands about **rank 27 of 57** (2026-08-26 23:12 UTC snapshot, leader `8.50e-9`)
  — a baseline and a safe fallback nominee, never a contender; `16.4x` off that
  leader.
  **Historical limitation at that point:** the local run did not establish setup
  compliance and said nothing about a private rerun. The later grading report now
  proves setup success and zero failures for exact v2.7; it does not clear a future
  candidate or predict private-rerun accuracy.
  **HISTORICAL BLOCKERS — do not execute these as current instructions.** They are
  retained to preserve the pre-submission audit trail; #328664 is already graded.
  (i) **CLOSED.** Samuel supplied written SHA-naming approval for
  `96ff522d…` (v2.7) and standing approval for the frozen baseline
  `00240eb1…` (the canary). Hard rule 6 is satisfied for both artifacts.
  (iii) **NEW, and it makes gate item 6 unsatisfiable for v2.7 as written.**
  `scripts/preflight.ps1` hardcodes `estimator.py` (lines 299, 309, 469; steps
  4/12/13/15/16); only `-Archive` is a parameter. **A green 16/16 witness
  produced tomorrow would certify v2.6.1, not v2.7.** Two honest exits and no
  third: promote v2.7 into `estimator.py` under its own STRATEGIC contract and
  certify with `-Archive estimator_v2_7.py`, or give `preflight.ps1` an
  `-Estimator` parameter. Both touch the enforcement layer. *(The related claim
  that v2.7 cannot be submitted at all is FALSE and was checked:
  `whest submit --estimator estimator_v2_7.py --dry-run` packages that file,
  35.0 KB, 1 file. Submission does not require promotion; **certification does**.)*
  (iv) **NEW.** The Rule-2 Skeptic returned **REJECT on fitness to submit** —
  *"the submission should not go out"* — on the residual-cliff evidence above.
  Its pre-registered discriminating experiment and decision rule are recorded in
  the LOG entry. This was a pre-submission finding; the exact file has since graded
  cleanly as #328664.
  **Historical ordering at that time:** the canary was the artifact directly
  certifiable by `preflight.ps1`. The remaining current gap is candidate-selectable
  certification for future nominees, not permission to resubmit either file.
  (ii) **A
  `preflight -Full` 16/16 witness, which CANNOT be produced on this host today**:
  steps 1–14 PASS, steps **15 and 16 FAIL** on `SETUP_TIMEOUT` under contention
  (GPU 63–93% busy on an unrelated workload, RAM 93% loaded; bare
  `import whestbench`+`flopscope` measures 3.54–4.35 s against the 5.0 s ceiling).
  Verified not a regression — every executed file is byte-identical to session
  start, and four runs of the envelope suite gave three different outcomes, which
  is the diagnosis. The suite reports this as RED **by design** ("a gate that
  cannot verify deployment must not certify it"), and that is being respected:
  **the gate is not green and must not be treated as morally green.** Re-run on a
  quiet machine.** Samuel's prediction was also never taken; that is recorded as
  an open defect of the slice, flagged in the contract *before* the measurement so
  the gate could not be moved by it.
- **Phase 1A is mathematically valid but PROVISIONAL.** 34/34 verifier and 16/16
  probe. Exact `1/32` cross-basis coherence; the phase-degree distribution is
  exactly **`{0: 1, 2: 511}`**, and **all 130,816** pairwise differences have rank
  10; the rejected
  coordinate mapping is an executable negative control with a raw artifact.
  **K6 is WITHDRAWN** — the verifier is a property checker, not an independent
  reference; CCKS Example 9.2 was attempted and abandoned rather than tuned into
  agreement. **K8 is quantified**: the carrier-aligned fourth moment is exactly
  **21.6881103515625×** the spherical target while 64 random directions give
  `[0.968, 1.023]`, so the subset is **not** a 4-design and random probing cannot
  establish it. The dense carrier's official flopscope ledger is exactly
  `1,099,511,627,776 = 2**40`, hence `C/B = 0.5` and multiplier `0.5` before any
  analytic/control overhead: a **5x score penalty** relative to K2's `0.1` floor.
  The unseen leader's overhead is unknown. The current probe wraps a NumPy-built
  carrier in `fnp`; it is not yet a grader-legal construction path, and its setup
  and Windows memory checks cannot clear the official boundaries. **No seed
  block, no `estimator.py` edit, no truth measurement, and no performance claim.**
  Phase 1B stays blocked until the bounded repair and a fresh independent ACCEPT.

## The K3 lane — measured 2026-08-27, independently ACCEPTed, and confirmed on organiser data

> **READ THE 2026-08-28 UPDATE AT THE END OF THIS SECTION BEFORE ACTING ON
> ANYTHING IN IT.** The "PRICED OUT" and "honest ceiling `~5.9e-8`" framing
> below is the 2026-08-27 state. Three oracle slices ran on 2026-08-28 and an
> independent Skeptic returned **REJECT** on two of the three readings. The
> net correction: rank-`n` capping is **not competitive** (a measured
> negative, and the stronger "closed for good" is retracted), the tensor is
> `96.7%`-capturable at rank `n` and compressing it is still useless because
> the flow reads only two diagonal slices, and no affordable mechanism to
> exploit that is known.

**ARC's factorized K3 (`mlp_kprop`, `k_max=3, factor=True`, unmodified) reaches a
true final-layer MSE of `~3.9-4.0e-08` at 1024x16.** Two independent readings:

| reading | networks | label | mean | GT-noise floor | 95% CI |
|---|---|---|---:|---:|---:|
| **P-A** | 24 DESIGN mints (P01) | our `2**26`/half A/B asset | `4.0047e-08` | `1.155e-9` | `[3.794e-8, 4.215e-8]` |
| **P-A-CONFIRM** | **100 organiser** (`v2-phase2@mini`) | **organisers' `1e9`-sample** | **`3.9249e-08`** | `7.478e-11` | `[3.821e-8, 4.029e-8]` |

They agree to **`1.99%`**. It is not an artifact of our instrument, our
population, or our label. Independent Skeptic verdict: **ACCEPT** — *"the
4.0047e-08 figure should be used to make the architecture decision"* — after it
built and then refuted by measurement the one attack that could have killed it (a
**common** fp32 label bias, which an A/B estimator cannot see because both halves
share it; measured at `1.6e-18`, ten orders below the signal).

For historical scale: our best graded entry was `1.374e-07` and the
2026-08-28 16:40 UTC leader was displayed at `7.4e-9`. The current board snapshot
is at the top of this file.

**The whole question was price, not information — and the price has now been
measured. It is prohibitive.** See P-B-COST at the bottom of `experiments/LOG.md`
(HEAD `b784b6f`). **Rule 2 is satisfied: independent verdict ACCEPT**, after it
showed that both of the checks the entry called load-bearing could not have
caught a bug, built the ones that could (a binding null control, now in-tree as
`--nulltest`, exact to `3.3e-16`), and replicated the drift number from an
independent implementation to 4 significant figures. Six numbers were corrected;
the two that matter are folded in below. **Open scope, recorded:** this is
"priced out for the compressors tried", not "for the concept" — P-B-ORACLE is
pre-registered at the bottom of `experiments/LOG.md` to settle that, and its
decision rule is committed and awaiting Samuel's prediction.

**Rank-capping the factored K3 is PRICED OUT.** Cost falls with the rank exactly
as inferred — affine to `0.0012%`, `cost(r*) = 1.5751e11 + 1.8908e08 r*`, so
`r* = n/8` costs `0.0826 B`, *below* the `0.1` multiplier floor. But accuracy
falls faster. Bounded with **no ground truth and no seed block** by a triangle
inequality on the drift from the uncapped prediction:

| cell | multiplier | MSE bracket | **score bracket** |
|---|---:|---:|---:|
| `normsort n/8` | `0.0826` | `[2.29e-6, 3.66e-6]` | `[2.29e-07, 3.66e-07]` |
| `normsort n` | `0.1597` | `[1.65e-6, 2.83e-6]` | `[2.63e-07, 4.53e-07]` |
| `dsproj n` | `0.2977` | `[4.74e-7, 1.19e-6]` | `[1.41e-07, 3.53e-07]` |
| `normsort 8n` | `0.7055` | `[4.26e-8, 3.68e-7]` | `[3.00e-08, 2.60e-07]` |
| `normsort 16n` | `1.1515` | `[7.51e-9, 9.83e-8]` | `[8.65e-09, 1.13e-07]` |
| **uncapped** | `1.4796` | `4.005e-8` (measured) | **`5.93e-08`** |

**The best point on the whole lane is the UNCAPPED one, and compression buys
nothing anywhere — `r* = 16n` included.** Correcting this entry's first version:
`16n`'s `22%` cost cut is exactly eaten by a `28%` accuracy penalty
(`MSE ~ MSE_full + drift = 5.14e-8`, score `5.91e-8` against uncapped's
`5.93e-8`), a `0.2%` wash rather than the `1.29x` I first claimed by reading the
optimistic end of a bracket. **"K3-class raw at `~0.2 B`" is refuted: at `0.2 B`
the measured raw is `~2e-6`, fifty times worse.** The honest ceiling for this lane
is **`~5.9e-8`** — a **`2.32x`** improvement on our best graded entry
(`1.374e-07`), and `~7.7x` short of the leader (`7.70e-9`, Samuel's Contract 3
board read, 2026-08-27 ~19:20 UTC, 63 entries; the dated in-repo snapshots are
`8.50e-9` and `9.50e-9`). **Do not budget an `fnp` port against `4.6e-8` — that
target does not exist.**

**The independent Skeptic's own finding is a cleaner kill than the bracket.** At
`r* = n/8` the "compressed K3" is **14x closer to plain K2 than to K3** — it *is*
K2, to `7%` in squared distance. Structurally so: `n` of the `2n` columns added
per layer are `D[:,j] (x) e_j (x) e_j` (`factor_k3.py:257-267`), so capping below
`n` deletes whole neurons' `kappa_3` diagonal. It is **coordinate pruning, not
compression**, which is why the drift curve is flat from `r* = 32` to `128`. So
the cheap cells buy K2's `4.05e-6` raw at `0.083-0.094 B` against the shipped
K2p1's `0.0625 B` — **strictly dominated by what is already on the board**, with
no bracket or anchor needed.

Three corrections to what this section used to say, all from measurement:

1. **The rank is NOT "pinned at exactly `15n`".** It grows by exactly `2n` per
   layer, `2n` at layer 0 to `32n` at layer 15. `15n` is the *mean* over layers,
   which is what a per-layer cost model consumes — so nothing costed on it was
   wrong, but the lever is 120x, not 8x, and the old "1/4, 1/8, 1/16 of `15n`"
   table above measured a different axis than per-layer capping does.
2. **The selector-pricing worry (D9) does not bite, and the reason is worth
   keeping.** Capping *every* layer means the selector only ever sees rank
   `O(n)`, never `15n`, so an `O(nr)` norm-sort costs `0.04-0.12%` of total.
   D9 was right in general; it is not the thing that kills the lane.
3. **The `0.1` floor IS reachable by rank capping** — `r* = n/8` lands at
   `0.0826 B`. It simply does not help, because the accuracy there is `~3e-6`.

**What does NOT exist yet, and gates everything downstream:** no `fnp` port (K3
runs in torch on a GPU; the grader has neither), no flopscope price for any
operating point, and no residual or setup measurement. Torch peaks run
`3.9-22.3 GiB`, but the factors themselves are only `201 MB` at `r* = 16n`, so
**the `8 GiB` limit is NOT established as binding** and must not be quoted as a
second kill — a streaming port could plausibly clear it. fp32 is licensed for
**research runs only** — worst-network accuracy delta `0.037%` over 7 networks,
and P-B-COST's f64 control reproduced its f32 drift to four significant figures —
and any `fnp` spelling must be **priced, not assumed**. All P-B-COST FLOPs are
upstream's torch-side counter, **not flopscope 0.12 pricing**; the conclusion
would need a `>10x` pricing error to move.

**What survives, and is now the whole live program by elimination:** O1 (fitted
projection/renormalisation coefficients on the K3 chain), O2b (diagonal/trace
`kappa_4` readout correction) and O5 (per-neuron ridge corrector on the K3
residual). None of the three depends on rank capping. O2a (rank-capped `k3aug`)
and O3 (Richardson-in-rank) both presuppose the lever that just failed and are
undermined, though neither is formally refuted.

### UPDATE 2026-08-28 — the three oracles ran, and an independent Skeptic REJECTED two of the three headline readings

**Rule 2 IS satisfied and the verdict was REJECT.** The retraction entry at the
bottom of `experiments/LOG.md` is the record of correction; the two entries above
it contain claims that are struck there and must not be read on their own. The
corrected readings are folded in below. Nothing in this lane may enter the
write-up. Contracts and results are the four entries at the bottom of
`experiments/LOG.md` (`7effbc8`, `8549691`, `2fa22dc`, `222db96`, `b5442f8`);
raw artifacts in `experiments/2026-08-28_pb_oracle/`. `preflight` GREEN 12/16 at
`b5442f8`.

Mean drift from the uncapped prediction, three probe networks `1000201-203`,
`1024 x 16`, f32. Costs are **upstream's runtime counter, not flopscope 0.12.**

| cell | out rank | cost | drift | `/MSE_full` |
|---|---:|---:|---:|---:|
| `normsort:n` (naive) | `n` | `0.1597 B` | `2.724e-6` | `68.0` |
| `dsproj:n` (= ORACLE-3) | `n` | `0.2977 B` | `8.832e-7` | `22.1` |
| `oracle:n` (ORACLE-1) | `n` | `0.1597 B` | `7.531e-7` | `18.8` |
| `oracle_offdiag:n` (ORACLE-2, **defective**) | `2n` | `0.2477 B` | `3.960e-6` | `98.9` |
| `oracle_offdiag_exact:n` (ORACLE-2b) | `3n` | `0.3299 B` | `3.013e-7` | `7.52` |
| `oracle_offdiag_exact:2*n` | `4n` | `0.4121 B` | `1.510e-7` | `3.77` |
| **`oracle_offdiag_exact:4*n`** | `6n` | `0.5647 B` | **`3.689e-8`** | **`0.92`** |

1. **ORACLE-1 did NOT close the lane. That closure is RETRACTED, and the
   pre-registered rule it rested on was itself unsound.** `min` over *achievable*
   objects gives `opt <= 7.531e-7`; the gate concluded `opt >= 4.0e-7`. That
   infers a lower bound from an upper bound and is invalid at any margin — and the
   rule committed at `cbfe67f` already had the flaw, so pre-registration did not
   save it. The gated number is also soft: it falls **23%** to `5.789e-7` under a
   different column score and more sweeps, and drift is **non-monotone** in the
   sweep count (`6.493e-7` at 64, `6.623e-7` at 128), refuting the entry's own
   "more sweeps only tighten it" claim.
   **What stands is a strong measured negative:** the best of six rank-`n` objects
   drifts `5.789e-7 = 14.5x MSE_full`, so **rank-`n` capping in this basis is not
   competitive.** Whether *some* rank-`n` compressor beats `4.0e-7` is
   **unsettled and not settleable from here** — nothing measured is a lower bound.
   All three parties predicted PROVISIONAL, and PROVISIONAL is what it is.
2. **ORACLE-3 needed no GPU time: it is definitionally the `dsproj` cell**, already
   measured and committed at `b784b6f`. `8.832e-7`. The `< 5e-8` falsifier that
   would have implied a cheap near-`O(n^2 L)` "K3-diag" estimator **missed by
   `18x`**. There is no cheap diagonal-only K3. Recorded as a **postdiction** —
   the data pre-dates the prediction, so it carries no Rule-1 credit.
3. **The real finding is why compression failed, and it is not "the tensor is
   incompressible".** The best rank-`n` approximation captures **96.7% of the
   Frobenius norm** of a rank-`32n` tensor at depth (`sqrt(1 - 0.254^2)`; an
   earlier "~75%" here read `1 - 0.254` as a norm fraction, which it is not, and
   the residual is measurably orthogonal to the approximant). `kappa_3` is
   substantially
   compressible; compressing it was still useless, because **the flow reads
   `kappa_3` only through its `(3,)` and `(2,1)` diagonal slices**, and
   Frobenius-optimal compression spends rank on the all-distinct bulk — which
   carries most of the norm and none of the readout — and damages the diagonals on
   the way. The question was never "is `kappa_3` low-rank" but **"is there a
   compressor whose error is orthogonal to those two slices"**.
4. **ORACLE-2 as first specified was DEFECTIVE, not informative.** Keeping the
   diagonal summand exact does not keep the diagonal of the *sum* exact: the
   truncated remainder is not diagonal-free and leaked error worth **40% of the
   true `(2,1)` slice's norm by layer 15**. Confirmed three ways (layerwise
   measurement, control N4 at small `n`, and a `13.1x` recovery when removed). Its
   `3.960e-6` stands unamended in the log **and stays in control N4**, so the
   difference remains visible.
5. **The "diagonal preservation buys 4.5-5.5x" attribution is REFUTED at matched
   rank, and that comparison was rank-unmatched in my candidate's favour.** I
   compared it to `normsort` at matched *cost*; at matched *rank* plain `oracle`
   is **1.34-1.88x BETTER** than the diagonal-preserving construction (rank `3n`:
   `1.602e-7` vs `3.013e-7`). **The whole gain was the ALS oracle, not protecting
   the diagonals** — protecting them costs accuracy. Result-entry prediction 4 is
   re-scored HELD -> REFUTED.
   **And the oracle's own price is not ignorable:** measured `22.27 B` for
   `oracle:3*n` and `7.90 B` for `oracle_offdiag_exact:n`, `92-99%` of it in the
   selector, against `normsort:3*n`'s `0.0002 B`. Four orders of magnitude, and
   intrinsic (MTTKRP is `O(n r_in r*)`). Every `B` figure attached to an oracle
   cell is the price of the **resulting rank if some other mechanism produced
   it**, and no such mechanism is demonstrated.
6. **Status: WEAK OPEN, PENDING A MECHANISM — and the cheap half of the search
   is already negative.** Plain `oracle` reaches `1.204e-8` at rank `8n`, so the
   information IS recoverable; but better column scores buy nothing above rank `n`
   (`normsort:8n` with `d21` `1.644e-7` vs `frob` `1.651e-7`), leaving a `13.7x`
   gap to the oracle that no affordable mechanism closes. The adopted condition on
   this lane is the Skeptic's follow-up: cap inside `FactoredTensor.__add__` so
   `r_in` never exceeds `r* + 2n`, and bill the `rankcap`. **If that cannot get
   under `~0.01 B` while staying within a small constant of the oracle ladder, the
   lane is retired rather than laddered further.** It needs its own contract.
   Nothing is built; drift is not an error increment, every score column is a
   **bracket**, and every bracket straddles the uncapped `5.925e-8`. **W-EFF is
   unaffected by all of this and remains the centerpiece.**

**The `4.6e-8` warning above still stands and is reinforced**: do not budget an
`fnp` port against any number in the table, because none of them is a measured MSE
against a label and none is priced by flopscope.

**Instrument defect found and fixed** (`b5442f8`): `predict()` was building an
autograd graph in a module that differentiates nothing — `75.21 GiB` live on a
`31.84 GiB` card. Now under `torch.no_grad()`, asserted **bit-identical** on an
already-committed cell, and confirmed a second time independently.
**The claim that this also explained the full-width f64 `--selftest` OOM is
RETRACTED**: `cmd_selftest` never enters `predict()`, and it still OOMs at HEAD.
**The width-1024 self-test remains unproven — only width 256 is** — and that is
what licenses reading a capped run as "upstream plus a cap".
**Three of the five null controls (N1, N2, N3) are satisfied by construction** and
are documented as such in the source; only N4 and REG carry weight.

**Correction owed to Samuel's Contract 4 § 1.** Its `c''_3` arithmetic reproduces
exactly here (`L^2 n^3 = 2.7488e11`; `5.93e11 -> 2.157`; `3.2537e12 -> 11.837`;
ratio `5.49x`), but `11.8` is upstream's meter and `2.1` is flopscope's. Narrowed
by source reading rather than assumption: upstream's symmetry discount
(`flop_utils.slice_factor`, for a kernel its own docstring calls *"theoretical,
unimplemented"*) misses **both** dominant lines — `get_dslice` and `contract_W`
are billed at `factor = 1.0`, and only the `3.01%` `nonlin_sum` line carries it.
The `5.49x` is therefore an **affordability target obtained by comparing different
meters**, not a W-EFF saving. It must not be used as a projection or enter the
write-up until an exact W-EFF operating point is billed by flopscope. In SIMPLE
mode the two-leg block's middle factor is `torch.eye(n) * 3`, so of the `2n`
columns added per layer, `n` have **two** one-hot modes and the other `n` have
**one** — sparse-aware contraction has purchase on all of them, not half.

## W-EFF is MEASURED and the full-tower port is KILLED — 2026-08-28

**The cost question that gated everything is answered.** Contract `08bb4d5`
(before any data) → result `f461e5c` → independent **PROVISIONAL** whose central
finding was accepted in full and corrected at `966731b`. A **second** fresh pass
on that correction is in flight; nothing below may be treated as settled until it
returns.

- **Exact sparse/equality-aware K3 over the full 16-layer tower costs
  `1.0723 B` under flopscope 0.12** (whole-path `1.0732 B`), against the
  preregistered `<= 0.35 B` affordability screen. **FAIL by `3.0664x`.** The
  full-tower W-EFF port is killed. Raw evidence:
  `experiments/2026-08-28_weff/`.
- The saving over the dense spelling is `27.8%` and it is a **bounded constant
  factor**, not a change of order. The mechanism is **mode equality**, not
  sparsity: identity structure dies at the first `contract_W`, but
  `from_dstensor`'s `(A, I, I)` becomes `(., M, M)` and the two-leg block's mode
  2 stays exactly `3M` forever, because `contract_W` and `contract_wick_` apply
  the same operator to every mode. **The review verified this in upstream's real
  tower**, not just from source: `||B_D - C_D|| = 0` exactly and
  `||B_L - 3 B_D|| <= 3.24e-16` at every layer.
- The mechanism model predicts the saving from lineage counting alone —
  `contract_W -> 465/720 = 0.6458333`, `get_dslice -> 600/768 = 0.78125` — and the
  measurement returns **`0.6458333` and `0.78164`**. The saving is what the
  structure forces.
- Parity is exact: `5.6e-16` against upstream torch (mediated by an artifact over
  identical bytes), `5.6e-16` W-EFF against the dense `fnp` reference, and
  `4.4e-16` / `1.7e-16` on the `contract_W_weff` / `contract_wick_weff`
  **commutation** test that a review found missing. Zero scalar materializations;
  zero uninstrumented leaks (hard rule 3 is now *enforced*, not asserted in
  prose). Eight mutations detected, one recorded invariance not counted.
- **Two meters, and they now agree.** Once binned the way upstream bins them,
  `contract_W` `0.99951`, `get_dslice` `0.99884`, `nonlin_sum 111 factored`
  `0.98941`. The sole divergence is `contract_W_basic` at `+33.1%`, and **that is
  the sole cross-meter divergence. **A fourth pass REFUTED the first reading of
  it.** `flops.as_symmetric` on the core indeed buys nothing (`1.00171x` —
  declaring the symmetry *costs* `7,340,031` FLOPs), but flopscope's
  auto-detected `identity_pattern` **does** grant a discount: an aliased
  `fnp.einsum('ij,kj->ik', A, A)` is `0.5005x` of `A @ A.T`, and `W core W^T`
  via `cholesky` + aliased Gram is **`0.8336x`** (relerr `7.65e-07`),
  conditional on the core being SPD. So `~0.005 B` on the tower **is**
  recoverable. It does not move the screen; it widens the `m <= 3` window
  margin, which was the thin part of the table.
- **Corrected, and the first version of this claim was wrong.** The result entry
  said `~1.0e11` FLOPs of upstream's ledger were "not executed here at all" and
  quoted a `+6.1%` `get_dslice` divergence. Both were artifacts of a `flop_name`
  scope boundary — upstream bills the *fresh* two-leg block's `(2,1)` slice to
  `nonlin_sum 111 factored`, not to `get_dslice`. The genuinely unexecuted
  remainder is **`2,104,414,384` = `0.000957 B`**, wrong by a factor of `47.5`.
  *(That correction was itself corrected: the first attempt said `0.000222 B`,
  because upstream's `nonlin_sum 111 factored` bin also carries `contract_wick_`.
  It is no longer prose — `p2_weff.py --bincheck` proves the binning by
  measurement (`d(nonlin_sum) - 3n*2nl` constant at `31,072` over four
  increments) and closes the whole fnp-versus-upstream gap **exactly** at
  `13,006,070,096`.)*

### The durable deliverable: the measured late-window cost table

A final-`m`-layer exact-K3 window from a **zero** K3 checkpoint costs exactly the
first `m` layers of the tower — its rank trajectory inside the window is
`0, 2n, 4n, ...`, and that trajectory is verified against upstream *from a
zero-K3 start*, so it is the window's own, not the full tower's.

| `m` | K3 window | `+` K2p1 prefix | multiplier | margin to the `0.1` floor |
|---:|---:|---:|---:|---:|
| 2 | `0.0117 B` | `0.0743 B` | `0.1000` | **`25.7%`** |
| 3 | `0.0303 B` | `0.0929 B` | `0.1000` | **`7.1%`** |
| 4 | `0.0577 B` | `0.1203 B` | `0.1203` | over |
| 6 | `0.1388 B` | `0.2014 B` | `0.2014` | over |
| 8 | `0.2551 B` | `0.3177 B` | `0.3177` | over |

**`m <= 3` is free — but that is a MARGIN, not a category.** `7.1%` at `m = 3`
(`12.9%` if the acknowledged K2-prefix double count is removed), against an
un-executed remainder of `0.00004 B`, three orders below it. This replaces the
plan's `m^2/L^2` hypothesis with a measurement. **It says nothing about
accuracy** — that is the label-free drift panel of Track 2 (contract `f4461b4`).

### Corrected deployment constraints

Read from the installed package: `residual_wall_time_limit_s = 0.4` (**gated per
MLP**), installed `wall_time_limit_s = 120.0` (**participant predict wall per MLP**),
`predict_timeout_s = 30.0`,
`memory_limit_mb = 8192`.

**flopscope defines `residual_wall_time_s = wall - backend - overhead`**
(`_budget.py:1781-1799`). Work inside instrumented ops is *backend* time and is
**not** charged to the residual — which is why the shipped K2p1 spends
`137.6e9` counted FLOPs for a graded residual of `~7 ms`. So residual risk is
about **op count and Python glue**, not total FLOPs. **CORRECTED 2026-08-31:** the
installed 120-second wall limit is created inside the evaluator's per-MLP loop and enforced
by the subprocess worker around one `predict()` call. It is not a 100-MLP total
and must not be divided to `~1.2 s/MLP`. Graded v2.7 has roughly `130x` headroom
against this installed wall limit and `54.7x` against the residual cliff. Top public entries
use roughly 61–105 seconds per MLP and pass, proving a server allowance of at
least `104.766 s/MLP` but not exposing the exact server ceiling. A separate outer AIcrowd job timeout,
if any, remains unverified and must not be invented from this field.

### What this does NOT license

No accuracy claim. No deployability claim — **residual, setup and memory are
unmeasured for every W-EFF arm**, and there is no `fnp` port of the K3
recurrence. **A candidate cannot exist without that port**: SIMPLE mode needs 25
enumerable diagram terms plus the DS/harmonic algebra and the ReLU wick
coefficients, which is a multi-day IMPORTANT-then-STRATEGIC effort with its own
parity burden. Nothing here enters the write-up.

## GATE-W FAILS: the late-window lane is closed, and the reason was retracted — 2026-08-28

Contract `f4461b4` (before any arm ran) → result `f01c3e5` → independent
**PROVISIONAL** → addendum contract `7ca4285` → corrections at `eb016ac`.
**Four independent Skeptic passes ran across this session's two slices; all four
returned PROVISIONAL or REJECT, and all four were accepted.** Preflight
**GREEN 12/16** at `eb016ac`, worktree clean, same four full-only skips.

**Today's outcome: an evidence-backed kill. No truth block spent, no candidate
built, NO SUBMISSION.** `estimator.py` remains v2.6.1 (`00240eb1…`),
byte-identical to its archive. The seed ledger, vault seal and touch log,
`HUMAN_MODEL.md`, `competition_score.py` and `scripts/` are untouched across
`1f16203..eb016ac`.

### The verdict, and it is provable

Six fresh unregistered probes `1000401`–`1000406`, common random numbers, f32,
`1024x16`. `D` is RMS drift of the final-layer prediction from uncapped full K3.

| arm | mean `D` | provable score floor | vs graded v2.7 |
|---|---:|---:|---:|
| `late:0` (pure K2) | `2.0706e-03` | `2.901e-07` | **`2.11x` worse** |
| `late:2` | `2.0630e-03` | `2.874e-07` | **`2.09x` worse** |
| `late:4` | `2.0186e-03` | `3.215e-07` | **`2.34x` worse** |
| `late:8` | `1.8229e-03` | `6.387e-07` | **`4.65x` worse** |

**Every window arm PROVABLY CANNOT clear the `2x` bar** — not "was not shown
to". That is stronger than the result entry first claimed: `bracket()` had
dropped the `D - RMSE_full` branch of the triangle inequality, zeroing every
lower bound. Fixed; `p2_pb_rank.py:839` in this repository already had it right.

The gate is not rigged. Re-run maximally in the arm's favour — `MSE_FULL -> 0`
*and* the multiplier pinned at the floor for every `m`, both physically
impossible — G1 still fails by `2.20x`–`2.49x`. For `m = 2` to pass, the bar
would have to be `3.72x` worse than graded v2.7 and `1.26x` worse than the
shipped canary.

**Instrument validation:** `late:0` is pure K2, so `D^2 + MSE_full` should
reproduce the known K2 raw MSE. Measured `4.3267e-06` against the organiser
figure `4.050566e-06` — **`1.068x`**.

### RETRACTED: the reason, and the generalisation

The result entry attributed the failure to *"`kappa_3`'s contribution to the
final-layer mean is accumulated, not local"* and generalised to *"it equally
kills any future 'start the expensive machinery late' design on this state —
which is a family, not a single arm."* **Both are retracted.**

`late:m` changes **two** things at once: it discards `kappa_3` memory **and**
hands the window a `k_max=2` prefix whose `K[1]`/`K[2]` are already wrong. The
`k3pre:m` arm isolates them — full-K3 prefix, identical zero checkpoint. Over six
probes:

| `m` | `D(k3pre)^2 / D(late)^2` | **the prefix explains** |
|---:|---:|---:|
| 2 | `0.1079` | **`89.2%`** of `late:2`'s squared drift |
| 3 | `0.1546` | `84.5%` |
| 4 | `0.2013` | `79.9%` |

**The window failed mostly because its prefix is bad, not because `kappa_3` is
non-local.** The binding open question is therefore **the quality of the
`K[1]`/`K[2]` prefix** — already item 1 of the hypothesis stack below (f32 K2 /
faithful augmented-K2). This session did not open a lane; it removed a false
closure from one that was already ranked first.

**Deviation, declared:** `k3pre` ran at **width 256 on CPU, not the live shape**.
Two live-shape attempts OOM'd the 32 GiB card and other work needed the GPU.
`late:m` behaves differently across the two widths, so the *window* half does not
transfer; the *attribution* half holds at both widths over eight networks, which
is what the retraction rests on. **The live-shape six-network confirmation is
OPEN** — command recorded at the bottom of `experiments/LOG.md`, needs ~30 GiB
and a quiet card.

### What is NOT reopened

`normsort:8192` — an ordinary rank cap held across the whole tower — is the best
arm at `D = 4.36e-04`, `4.2x` better than `late:8`. **This does not reopen the
rank-capping lane**: it is internal evidence at a cell P-B already priced at
`0.7055 B` with a straddling bracket, `AGENTS.md` requires *external* evidence to
reopen, and it costs `~2.3e12` upstream. Its independence was also overstated —
`p2_window.py` imports `cap_factored` from `p2_pb_rank`, so it is the same
implementation, not a different code path.

### Still blocking any future windowed or K3 design

Residual against the `0.4 s` cliff (unmeasured for every arm); the per-MLP
installed **`120 s` participant wall default** (graded ledgers prove only `>=104.766 s`);
any separately sourced outer-job timeout; the
`8,192 MB` limit Windows cannot enforce; and
the absent `fnp` port of the K3 recurrence — 25 enumerable SIMPLE-mode diagram
terms plus the DS/harmonic algebra, a multi-day effort with its own parity
burden. **No same-day candidate was ever reachable**, and a passing gate would
only have licensed starting that port.

## Authoritative document map

| need | source |
|---|---|
| current status and next action | this file |
| active architecture, modules, work packages, tests, and decision sequence | `docs/GAME_PLAN.md` |
| current rules/leaderboard/research evidence | `docs/PHASE2_FRONTIER_AUDIT_2026-08-31.md` |
| historical pre-W-EFF/window frontier audit | `docs/PHASE2_FRONTIER_AUDIT_2026-08-28.md` |
| launch-week historical audit | `docs/PHASE2_LIVE_AUDIT_2026-08-24.md` |
| historical 2026-08-26 mission-brief reconciliation, superseded where current docs differ | `docs/PHASE2_BRIEF_RECONCILIATION_2026-08-26.md` |
| measurement protocol | `docs/VALIDATION.md` |
| operational traps | `docs/GOTCHAS.md` |
| contracts, results, Skeptic verdicts | bottom of `experiments/LOG.md` |
| independent review brief | `docs/SKEPTIC.md` |
| completion-oriented Claude implementation prompt | `docs/CLAUDE_CODE_HANDOFF.md` |
| contribution manuscript | `docs/writeup-draft.md` — STALE/DO NOT SUBMIT until a STRATEGIC claim review |

The old frontier packets and handoffs are superseded. The redundant Phase-2
frontier redirect was removed on 2026-08-31. Minimal stubs remain only where source
comments/tools link to their historical names; they redirect here and must not be
treated as plans.

## Current deployed and research state

### Shipped estimator

- `estimator.py` is still v2.6.1 and must remain byte-identical to its frozen archive
  unless a STRATEGIC contract authorizes a versioned change.
- The graded Phase-1 file remains v2.4.1.
- Phase-2 width takes `_analytic_only`; the effective method is K2p1.
- None of the Phase-1 rotation/Sobol/pruning/Strassen carrier is active at width
  1024.
- Official 0.16 validation passes.
- One official Phase-2 subprocess smoke produced:

| metric | one-row anchor |
|---|---:|
| score | `4.501854e-7` |
| final-layer MSE | `4.501854e-6` |
| multiplier | `0.1` |
| counted FLOPs | `137,600,326,656` |
| `C/B` | `0.0625734` |
| residual | `0.05368 s` |
| observed wall for this MLP | `0.35487 s` |
| failures | `0` |

This is an operational smoke, not a score estimate.

### Phase-2 local evidence that survives audit

P2-CP1 used 96 fresh 1024x16 networks and two independent `2**22` truth replicas.
There were no truth failures. Its narrow conclusions remain credible:

- K2p2 improved K2p1 about 1.052x for about 0.18% extra counted cost.
- Tested IID and ordinary random-orthogonal arms lost to K2.
- Higher Price/Hermite covariance-map orders plateaued.
- Those arms do not bound valid Kerdock/MUB structured quadrature.
- P01 was a DESIGN block, so it cannot support binding global lane kills.

P2-CP3R's durable mechanistic result:

- 94.61% of K2 error is accumulated mean/variance-state drift;
- marginal third/fourth corrections in the scalar ReLU mean map recover 4.47%;
- the registered O3-versus-O0 gate passed, then the recorded KILL used a different
  post-hoc O1-versus-O3 estimand;
- the oracle reset to true marginal state every layer and did not propagate ARC's
  augmented-K2 joint state.

Therefore marginal final-mean Edgeworth correction remains low value, but recurrent
joint-state augmented K2 is open.

### Seed and vault state

- P01 has consumed **168/600** seeds; 432 DESIGN seeds remain. The append-only
  seed ledger is authoritative; prose counts are only snapshots.
- P01 is not a DECIDE block and cannot serve as its own independent replication.
- No disjoint Phase-2 DESIGN-replication block or Phase-2 SINGLE_USE DECIDE block
  is registered. After the truth-free paths fix a power plan, allocating both is a
  separate IMPORTANT instrument slice with prediction, tests, preflight, knowledge
  delta, and independent review before any labels.
- Existing SINGLE_USE Phase-1 blocks keep their recorded state in the ledger.
- The Phase-1 vault cannot be used as a Phase-2 vault.
- No Phase-2 vault exists. Creating or touching one requires Samuel's explicit
  authorization and a STRATEGIC preregistration.

## Live frontier and research decision

**The sole active implementation program is the 2026-08-31 `GAME_PLAN.md`.** It
specifies module boundaries, work packages, tests, and falsifiable hypotheses; it
does not establish either nominee as a winning
architecture. Phase-2 leader code is private and no fixed basis count is confirmed.
The current cost cluster is consistent with roughly 16,384 dense forward
directions; eight antipodal MUB bases are one exact realization of that envelope,
not a method attribution.

The evidence-ordered hypothesis stack, **re-weighted 2026-08-31 after the live
board and wall-cap correction**:

1. **Submission-legal exact-radial structured carrier.** Public frontier telemetry
   and Puffi's disclosed Phase-1 family make the nested basis/sample frontier the
   shortest candidate-producing path. Close the existing grader-legal constructor,
   setup, memory, and scalar-sync gaps before truth.
2. **f32 K2 / faithful augmented-K2 plus query-directed K3 response.** Track 2
   measured that `89.2%` of a late window's error is its `k_max=2` prefix. Full K3
   proves the missing information matters; adjoint response tests whether only its
   final queries can be computed cheaply.
3. ~~exact sparse-aware K3 (W-EFF)~~ — **CLOSED, killed on a measured price:
   `1.0732 B` against its own `<= 0.35 B` screen, `3.0664x` over.**
4. ~~late-window or age-window exact K3~~ — **CLOSED, killed on a measured
   negative: every window arm's provable score floor is `2.09x`–`4.65x` worse
   than graded v2.7.**
5. a tiny neuron-specific mixture or permutation-equivariant rolled-forward
   correction only after a better carrier/state path exists;
6. two final nominees: a structured carrier and a compact analytic-response system.

**The immediate barrier was cost truth, and it has been paid.** Full factorized
K3 transfers at `3.9249e-8` raw MSE with a `7.478e-11` organiser-label noise
floor, and its exact W-EFF spelling is now **measured** at `1.0732 B` under
flopscope 0.12. The cross-meter `5.49x` "affordability target" is **dead** — it
compared two different meters and was never a saving.

**The whole problem, in one line.** Full K3 delivered at the `0.1` multiplier
floor would score `3.92e-09`, against the current leader's `5.60e-09` — it would
still lead. It costs **`10.7x` too much**, and W-EFF, the best exact full-tower
spelling, bought only `1.38x`. Meanwhile several public systems reach better raw
accuracy in the roughly 16k-direction dense-forward cost envelope. **The game is
to preserve third-order information or quadrature quality per billed FLOP.**

The unused free zone is `0.0374 B` (`82,301,998,899` FLOPs), and at corrected f32
the K2 path itself roughly halves, which would widen it to `~0.069 B`.
**Nothing measured so far is worth putting in it**: a 2-layer exact-K3 window
fits at `0.0117 B` and improves plain K2 by `0.37%`.

### Closed 2026-08-28 attempt

The W-EFF and late-window attempt ended in the two kills above. No truth block was
spent, no seed block claimed, no candidate built, and nothing submitted. Its
same-day execution plan has been removed from the active `GAME_PLAN.md`; the
contracts and results remain in `experiments/LOG.md` and git history.

> **OPEN RULE-2 GAP, recorded 2026-09-01. WP2, WP3 and WP4 have NOT been
> independently reviewed.** `AGENTS.md` Rule 2 requires a fresh Skeptic after
> every IMPORTANT slice before its result is believed or propagated. WP0A, WP0B
> and WP1+WP5 each got one, and **all three returned PROVISIONAL and each found
> a real defect** — including a flatly false *"no truth was read"* claim that
> had to be struck. The WP2/WP3/WP4 review was launched twice and killed twice
> by editor crashes before reporting.
>
> So the three NEWEST findings are the LEAST verified: the `MMD²` objective and
> its orthogonality-not-Kerdock negative, the AUGMENT **PORT** verdict at
> `1.03723x`, and the adjoint-K3 **KILL**. Treat all three as provisional until
> that review lands. The relaunch prompt is in the LOG entry for this gap.

**The next action, updated 2026-08-31 after WP0–WP5.** Every truth-free package
in `GAME_PLAN.md` §8 is now closed: WP0A–D (systems), WP1 (carrier), WP2
(objective + frozen panel), WP3 (AUGMENT, PORT), WP4 (adjoint K3, KILLED), WP5
(integrated candidate, certified). **The remaining steps all read truth and
therefore all need Samuel.**

**The single next decision, and the branches are no longer symmetric:**

| branch | multiplier | must beat v2.7 by | what is measured |
|---|---:|---|---|
| carrier k=8 | `0.2501` | **`2.501x` just to break even** | one `n=2` organiser datum says `1.42x` **worse** adjusted |
| **AUGMENT** | **`0.1`, pinned** | **nothing — it is free** | moves the scored row `1.8%` at `n=64`; sign unmeasured |
| adjoint K3 | — | — | **DEAD** (WP4) |

1. **The cheapest discriminator, and it needs no new allocation.** Score the
   carrier candidate against shipped v2.6.1 on **24–32 fresh mints from P01's
   432 unused DESIGN seeds**, common random numbers, one `2**22` truth replica,
   through `competition_score.decide()` with its GT-noise floor. It answers the
   `2.5x` break-even question before WP2's panel is spent on. **STRATEGIC: needs
   a Rule-1 contract and Samuel's prediction. It must NOT be answered by
   extending the `whest run` mini smoke past 2 MLPs — that turns a read-only
   CONFIRM tier into a selection tier.**
2. **The `fnp` port of AUGMENT**, now that its price is known and it is free at
   the floor. IMPORTANT, with its own parity burden (the harmonic/DS algebra and
   the Wick coefficients).
3. **WP0E**, prepared at `ea4b8fc` and awaiting Samuel's three answers.

The optional `k3pre` confirmation must not delay candidate-producing work.

The carrier, faithful augmented-K2 and response tracks remain part of the target
architecture, but none is allowed into a candidate without completing its own
evidence path. The recommended engineering bar—subject to Samuel
accepting it in the preregistration before data exists—is at least a `2x` adjusted
point improvement over v2.7 with the valid paired interval excluding zero. It is a
minimum “materially stronger” screen, not the winning target; the adopted winning
target remains `<=3e-9` on fresh offline evidence.

Coupled residual sampling is deprioritized. Its former 1.305x projection assumed a
float32 K2 cost reduction that the frozen artifact did not show: measured cost fell
only 3.1%, not 50%.

**CORRECTED 2026-08-26, and the correction was FIRST MADE 2026-08-21 — this
sentence was simply never updated.** `experiments/LOG.md` (2026-08-21, P2-CP1
prediction P6) already recorded the mechanism and the number, and that entry's own
independent review already reproduced it. This durable record contradicted the log
for five days. Independently re-reviewed and **ACCEPTED** 2026-08-26 (both cost and
accuracy), which also **answered the open caveat** the 2026-08-21 review had left:
float32 catastrophic cancellation in `var = max(ez2 - mu^2, 0)` **does** fire —
1,497 of 1,572,864 neuron-layers over 96 networks, worst relative 99.29%, most
negative `-7.72e-16` — and is **benign**: the `var_pre` clamp fires 0 times, there
are zero non-finite outputs on 96/96, and the float32 final-layer MSE against truth
is *marginally better* than float64 (`4.164689e-6` vs `4.164739e-6`; paired
`-4.97e-11`, 95% `[-9.94e-11, +5.2e-15]`, i.e. 0.28% of the GT noise floor).

**In grader spelling the ratio is `0.50002x`** (`137,600,363,520` ->
`68,802,510,336`), stable across five shapes and bit-identical across nine
networks — this arm does not prune, so its counted cost is network-independent.
`materialise_predictions` costs `32,768` on a float64 return and **`0`** on a
float32 one; the weight-representation term roughly halves (`37,746,688` ->
`19,920,896`).

The local-spelling table below is retained for the mechanism it shows.
**Label caution:** its "f64, shipped spelling" row is the *arm's* local spelling.
Three distinct numbers exist and must not be conflated — arm local
`137,562,584,064`, shipped `estimator.py` local `137,562,547,200`, grader
`137,600,326,656`.
`local_sim/p2_arms/arm_k2p1_f32.py` calls `flops.stats.norm.cdf` and `.pdf`, which
flopscope 0.12 **promotes from float32 to float64** (it emits a `FlopscopeWarning`
saying so), and the arm never casts back. `Phi`, `phi` and everything downstream of
them — including the covariance accumulator, where nearly all the cost lives — are
float64. The arm is float32 in name only. Measured today
(`local_sim/p2_f32_price_probe.py`, artifact
`experiments/2026-08-26_p2_cal/f32_price_probe.json`), one network:

| arm | counted FLOPs | vs f64 | state dtype |
|---|---:|---:|---|
| f64, shipped spelling | `137,562,584,064` | 1.0000x | float64 |
| f32 as the arm is written | `133,266,036,224` | **0.9688x** | **float64** |
| f32 with the two stats results cast back | `68,782,589,440` | **0.5000x** | float32 |

The recorded 3.1% is reproduced **exactly**; two `.astype` calls recover the full
2x. Divergence from the f64 control on the scored row is `3.3e-13` relative mean
square. **An earlier version of this paragraph called that "eight orders below a
signal of order 5e-5"; that arithmetic was invented.** The final row's mean square
is `0.924` and K2's own relative mean-square error is `4.51e-6`, so the divergence
is **7.1 orders** below the error being fought. Established across **all 96**
networks by the 2026-08-26 review, not one.

Consequence for design: the multiplier floor makes the first `219,902,325,555`
counted FLOPs free, and the shipped path spends `137,600,326,656` of them. At f64
the free zone holds `~82.3e9` unused FLOPs — **1.2** dense Kerdock/MUB bases, since
one costs exactly `2**36`. At corrected f32 it holds `~151.1e9` — **2.2** bases.

**The lane is NOT reopened by this**, and the reviewer pushed back on even
suggesting it. Reopening requires a STRATEGIC contract with a stated reopening
condition. What is established is narrower and is all that may be quoted: the cost
half of the lane's closure premise was measured on an arm that was not float32, so
the recorded reason no longer stands on its own. The 2026-08-24 deprioritisation
also rested on external board evidence this does not address, and its `1.305x`
projection was itself recorded as *"arithmetic, not measurement"* with three named
unmeasured inputs.

**Two things this does NOT license.** (a) The f32 arm's `--runner subprocess`
residual is **unmeasured** — lambda is 0 so it is not a cost, but the `0.4 s` cliff
is not smooth. (b) **The 2x dtype law is not universal.** `docs/GOTCHAS.md` records
flopscope-0.12 counterexamples in both directions — `where(mask, f32, scalar)` is
*half* price, `nansum`/`nanmean` on float went 2x *dearer*. `0.5000x` is a fact
about **this program**; any future port must be **priced, not assumed**.

## Evaluator state — P2-P0 repaired these, and the slice is closed

All nine Phase-0 blockers were repaired or explicitly bounded, deployment parity
is restored within the documented host limits, and the gate is closed on a literal
independent ACCEPT. Raw evidence:
`experiments/2026-08-24_p2_p0/`; contract, result, two reviews, a retraction, a
REJECT and the ACCEPT at the bottom of `experiments/LOG.md`.

**Four defects were carried forward** from the accepting review (D1–D4 in that
entry). None falsifies a Phase-0 gate clause. D4 — `VALIDATION.md` §7 describing
the old twelve-step gate — is fixed. **D1–D3 are CLOSED IN CODE and P1-PRE is
CLOSED on an independent ACCEPT** (seventh pass, reviewed HEAD `6d05d77`).

| carried defect | required repair | state |
|---|---|---|
| D1 — literal-`True` worker-restart witness | replace it with a structured result emitted by this suite; `--skip-worker-death` must make the witness fail/unprovable, never green | **closed** — deferred to a witness-closure section that runs after the provocations; injections A, D, G |
| D2 — witness `suite` field ignored | resolve the named suite to its own result artifact and refuse unknown/mismatched suites with executable negative tests | **closed** — `WitnessResolver`; also repaired on the artifact-writing path, where it had survived; injection B |
| D3 — `--no-write` rewrites lifecycle evidence | prevent regeneration and every tracked write; assert before/after hashes and a clean worktree | **closed** — a read-only latch that raises, plus an external proof by child process, sha256 and git; injection C |

**What P1-PRE cost, and the durable lesson:** six independent passes found six
successive instances of a claim outrunning the evidence behind it. D1–D3
themselves survived all six attacks; the overclaim was that a
pinned declaration proves the behaviour named by that declaration. It does not.
A useful check compares an independent expected property with observed behaviour.
Mutation probes add evidence that selected faults are detectable; they are not a
proof that every future edit is safe. The terminal pass therefore pins the exact
witness semantics and platform classification, measures what files/patterns were
actually scanned, and requires the named guard—not merely any non-zero exit—to
fire. It then stops. The repository will not build a second evaluator whose job
is to formally verify the first evaluator.

The historical destructive `git checkout --` path is gone at current HEAD, and
the T6 correction has landed: `local_sim/p1pre_fault_inject.py` **defaults its
transcript to a system temp directory, refuses any `--out` resolving inside the
repository** (tracked or not — the test is containment, not tracking status),
carries the old tracked transcript in its protected hash set, and takes the
untouched-repository measurement **after** the write rather than before it. It is
safe to run with no arguments. Copy its transcript into `experiments/` yourself
if you want it as committed evidence.

| # | blocker | state |
|---|---|---|
| 1 | `pyproject.toml` pinned WhestBench 0.14 / flopscope 0.10 | **fixed** — 0.16 / 0.12 pinned and locked; the lock diff is exactly those two packages |
| 2 | `competition_score.py` defaulted to `1e11` lambda | **fixed** — defaults come from `official_config`; live lambda is **0** |
| 3 | `lifecycle_eval.py` defaulted to Phase-1 wall/residual/memory/budget | **fixed** — `LifecycleSpec.for_round()` takes a round's six constants together and records deviations |
| 4 | local weights were raw NumPy, so weight-method ops were unmetered | **fixed** — `fnp.array`, as `MLP.from_row` builds them |
| 5 | local price low by 37,779,456 FLOPs vs the official anchor | **fixed** — reconciled to **zero**, op by op (see below) |
| 6 | the 0.4 s residual cliff was not enforced | **fixed** — armed by default and PROVOKED in the self-test |
| 7 | 8-GiB memory and worker death/restart not emulated | **partly** — worker restart is provoked for real through the official runner; **memory cannot be enforced on Windows at all** and is now an accounting constraint, not a test |
| 8 | manifest/checklist/preflight could be green on the stale stack | **fixed** — preflight step 1 asserts the live stack; the manifest reports `PACKAGE PARITY: ACHIEVED` |
| 9 | shape/GPU parity did not prove the 1024x16 live path | **fixed** — `mint_provenance` now checks 1024x16 FIRST, bit-identical against the installed 0.16 |

### The anchor, reconciled

The 37,779,456-FLOP shortfall had two mechanisms, and neither was the version bump
(flopscope 0.10 -> 0.12 moved **nothing** the shipped file executes):

```text
local, historical spelling      137,562,547,200
  + weight representation            37,746,688   fnp.array weights: astype 33,554,432
                                                  over 16 calls, plus multiply and sum
  + harness materialisation              32,768   whestbench 0.16 calls
                                                  materialise_predictions IN-window
full grader spelling            137,600,326,656   == official, unexplained 0
```

19 operations compared; every name, call count and FLOP figure agrees.

**A candidate whose final assembly is NATIVELY float32 avoids 32,768 FLOPs.**
`materialise_predictions` is free on a float32 return and 2 FLOPs/element on a
float64 one. The shipped file returns float64. The saving is CONDITIONAL: adding a
cast to reach float32 costs the same 2 FLOPs/element and nets zero, so it is only
free if the last array is built in float32. Taking it means changing
`estimator.py`, which is STRATEGIC and needs its own contract.

### What the evaluator still cannot do

- **Enforce the 8,192 MB memory limit — SUPERSEDED 2026-08-31 (WP0D). IT IS NOW
  ENFORCED, and the old figure was optimistic by 2.55x.**
  *Windows still cannot do it:* no `resource` module, so the subprocess worker's
  `RLIMIT_AS` call is skipped and `--runner local` is advisory by design. Windows
  accounting gave weights 64.0 MB, peak working set 440.8 MB, peak commit
  918.0 MB -> "at most 8.9x headroom, an UPPER BOUND", with the honest caveat
  that the true `RLIMIT_AS` headroom is smaller by an unmeasured factor.
  **The factor is now measured.** `local_sim/linux_memory_gate.py` applies
  `setrlimit(RLIMIT_AS, 8192 MB)` in a WSL2 Linux child exactly as
  `whestbench/subprocess_worker.py` does, **reads the soft limit back** (that
  worker catches four exception types and runs on uncapped, so "no exception" is
  not evidence), and runs the candidate's own `predict` at `1024 x 16` under it.
  A **negative control** requesting 9 GiB under the same cap dies with
  `MemoryError`; a **positive control** with no cap survives, so the death is the
  cap and not the machine.

  | file | VmPeak (what RLIMIT_AS caps) | headroom |
  |---|---:|---:|
  | `estimator.py` (v2.6.1) | `2,345.4 MB` | **`3.49x`** |
  | the WP1 carrier candidate, k=8 | `2,281.5 MB` | **`3.59x`** |

  Counted FLOPs matched Windows **exactly** in both cases, across two operating
  systems and two numpy versions. **Scope:** a WSL2 kernel applying the installed
  package's own default is not the grading server, whose `memory_limit_mb`
  appears in no `run_config` and no graded report this repository holds.
- **Observe the grader's own configuration.** Every limit above is a PACKAGE fact,
  and they are not equally well corroborated. A real `whest run --json`
  `run_config` records what the runner USED for `width`, `depth`, `flop_budget`,
  `lambda_flops_per_second`, `residual_wall_time_limit_s`, `wall_time_limit_s` and
  `setup_timeout_s` — and **not** for `memory_limit_mb` or `predict_timeout_s`,
  which remain `ContestSpec` class defaults only. So the **8,192 MB limit — the one
  constraint this host cannot test at all — is also the one with the weakest
  provenance**. The two Phase-2 graded reports publish observed setup/residual/
  FLOP telemetry for their exact files, but not an operation ledger or an
  enforceable local memory limit.
- **Transfer a local second.** The deployment authority for residual is
  `whest run --runner subprocess`, never the in-process figure.
- **Enforce the 5-second setup ceiling meaningfully.** `import whestbench` alone
  is 1.3 s on a quiet host and 4.7–9.1 s under load; our own estimator's setup is
  0.6 **ms**. The in-process timer measures a strictly smaller window than the
  grader's subprocess one, so its "419 ms of 5000 ms" is not a transferable
  margin. Matters directly: the carrier builds its point set in `setup()`.

### Historical launch-week residual measurements

Best case, one quiet session, `--runner subprocess` as the authority, 20 MLPs per
cell (`experiments/2026-08-24_p2_p0/p2_residual_paths.json`):

| candidate | median | worst reading | over the 0.4 s cap |
|---|---:|---:|---:|
| `estimator.py` (shipped K2p1) | 56.6 ms | 83.6 ms | 0/20 |
| `arm_k2p1` | 55.9 / 59.4 ms | 95.5 ms | 0/20 |
| `arm_k2p2` — the architecture's reference | 94.6 / 95.1 ms | 140.6 ms | 0/20 |

**Those are best-case and must not be used alone.** The same `arm_k2p2` reads
**260.7 ms median / 390.6 ms worst — 1.02x headroom** in the frozen P2-CP1 ladder
of 2026-08-21 — that figure is `lifecycle_eval` **in-process**, against the
subprocess table above, so the two are not directly comparable; the decomposition
is instrument-matched. A first write-up blamed machine contention; **that is retracted**.
Three experiments decompose the 2.35x: the weight-representation repair is a fixed
**~19 ms**, flopscope 0.10 vs 0.12 is **0.91x** (not the cause), and **~2.35x is
run/session state and remains unexplained.**

**There is no trustworthy additive carrier allowance.** Subtracting the 260.7-ms
median suggests ~139 ms, but the cap applies per MLP and that day's 390.6-ms
maximum leaves only **9.4 ms** on its row. Residual is not known to add linearly
between branches. Phase 1 must test the integrated candidate, include the observed
~2.35x session spread in a preregistered tail falsifier, and reject any design that
passes only in a friendly session.

Two cautions on the numbers above, both from an independent Skeptic: the "worst
MLP" is frequently the **first** MLP (a warm-up transient, which still fails an MLP
under a cliff but is not network heaviness), and **n = 20 is thin for a maximum**.

The paired arithmetic score test, common-random-number design, ground-truth noise
reporting, seed ledger, public-CONFIRM rule, and vault discipline remain sound.

## Active execution queue

| order | slice | class | status |
|---:|---|---|---|
| 0 | official-stack/evaluator repair | IMPORTANT | **CLOSED 2026-08-24** — G1–G7 all met, independent ACCEPT recorded |
| 0H | P1-PRE: D1–D3 evaluator hardening | **IMPORTANT** | **CLOSED** — seventh independent pass ACCEPT; 15 named mutations and current durable record verified |
| 0S | Phase-2 canary + calibrated submission | STRATEGIC | **CLOSED — SHIPPED AND GRADED.** #328662 and #328664, zero failures; v2.7 is rank 53/83 at the current snapshot |
| 0C | K2 scalar calibration | DESIGN | **EXHAUSTED.** 2.96x transfer is real; multiplicative state gains collapse to output scale and non-commuting layer scalars did not transfer |
| WP0A | Phase-2 manifest/tag-drift repair | IMPORTANT | **CLOSED 2026-08-31** (`79ec305`, repaired at `a4cae61`). Manifest repinned with both commits in `dataset.commit_lineage`; `p2_tag_drift.py` is the shared policy and runs in preflight step 2. Skeptic PROVISIONAL accepted in full: `data/` is NOT the whole input set — whestbench prefers `prepared/<split>` Arrow — so a 226-object loader record was added and four false-pass tampers closed |
| WP0B | enforcement-label repair | IMPORTANT | **CLOSED 2026-08-31** (`a4cae61`/`dfdf6a3`, repaired at `79b9fba`). Three gate labels were false; counts are now scanned EXHAUSTIVELY and the suite runs in preflight step 7. Skeptic PROVISIONAL accepted: its pre-registered prediction named the two remaining stale lines exactly |
| WP0C | candidate-selectable certification | IMPORTANT | **CLOSED 2026-08-31** (`51150ba`). `local_sim/certify_candidate.py --candidate <f>` certifies any file read-only in 11 sections; adds no path through `preflight.ps1`, so the ship-file gate is untouched. `test_certify_candidate.py` 25/25, ten mutations each failing its NAMED check |
| WP0D | real Linux 8-GiB enforcement | IMPORTANT | **CLOSED 2026-08-31 — ENFORCED FOR THE FIRST TIME** (`51150ba`). See the corrected memory figure below |
| WP0E | Phase-2 evidence-block allocation | IMPORTANT | **PREPARED, NOT EXECUTED** (`ea4b8fc`). No block registered, no seed drawn. Proposes P02/P03 DESIGN (96 each, disjoint) + P04 SINGLE_USE (100), and records the three questions Samuel is owed first. **A cheaper first step exists:** the carrier break-even question fits on P01's 432 unused DESIGN seeds with no new allocation |
| 1 | W-EFF exact K3 representative price/parity | IMPORTANT | **CLOSED 2026-08-28 — MEASURED AND KILLED.** `1.0723 B` measured / `1.0732 B` whole-path against a `<= 0.35 B` screen, FAIL by `3.0664x`. Parity exact, zero materializations, zero leaks, bin reconciliation closes exactly. **Two** independent PROVISIONAL passes, both accepted in full; the second found an error inside the first correction |
| 2 | late-window / age-window K3 imitation | IMPORTANT | **CLOSED 2026-08-28 — GATE-W FAILS, provably.** Every window arm's score floor is `2.09x`–`4.65x` worse than graded v2.7. Independent PROVISIONAL accepted: the *verdict* stands and is stronger; the *attribution* was retracted |
| WP1 | grader-legal carrier constructor | IMPORTANT | **CLOSED 2026-08-31** (`292fbab`). `p2_carrier_fnp.py` is stdlib + `fnp`, no numpy, **byte-identical to the frozen `kerdock_mub.py` at width 1024**. 117/117 |
| WP2 | truth-free carrier objective and frozen nested panel | IMPORTANT | **CLOSED 2026-08-31** (`3bae94b`). Deep-ReLU MMD² objective, 74/74 incl. live width. Panel frozen and content-addressed at `a9bd32a1…` with its rule, DESIGN falsifier and proposed gate inside the digest. **The finding runs against Kerdock:** `MMD² ∝ 1/k` (the plain Monte-Carlo rate, so structure buys a constant not a rate), and at matched rows random orthogonal blocks are only `1.026x`/`1.006x` worse while IID is `1.52–1.56x` worse — the advantage is mostly ORTHOGONALITY, and Kerdock adds `0.6–2.6%` shrinking with width |
| WP3 | faithful AUGMENT | IMPORTANT | **CLOSED 2026-08-31 — VERDICT PORT** (`a480ab6`). Unfactored AUGMENT costs **`1.03723x`** plain K2 at `1024x16`, `+3.72%`, **free at the `0.1` multiplier floor**. The excess halves per doubling of `n` — `Θ(n²)` on a `Θ(n³)` base. State: degree-3 is an `(n,)` VECTOR and degree-4 a SCALAR, `1,025` numbers on K2's `1,049,600`. 33/33. **No `fnp` port exists yet; the verdict is that one is worth building** |
| WP4 | query-directed / adjoint K3 | IMPORTANT | **CLOSED 2026-08-31 — KILLED on its own preregistered falsifier** (`18c5fa3`). The K3→mean response is neither linear nor additive: two kill rules fired at five shapes and it worsens with **DEPTH** (`0.277` at `32x3` → `0.938` at `32x10`; live depth is 16). Locality HELD — `kappa_3` IS local, the response to it is nonlinear |
| WP5 | integrated live-shape research architecture | IMPORTANT | **CLOSED 2026-08-31** (`292fbab`). `p2_candidate_architecture.py` + `p2_build_candidate.py`: a self-contained single file **CERTIFIED 36/36 at `1024 x 16`**, official subprocess smoke included, 0 failed MLPs. 50/50 integration tests |
| 5 | neuron-specific mixture / rolled-forward correction | STRATEGIC | pending after a deployable carrier or state path exists |
| 6 | integrated SINGLE_USE decision and robustness | STRATEGIC | pending |
| 7 | public CONFIRM / authorised Phase-2 vault / two nominees | STRATEGIC | pending |
| 8 | contribution write-up | STRATEGIC claim review | pending |

## Samuel's standing predictions — recorded 2026-08-24, before the data

Taken before the Phase-0 repair and before any carrier or augmented-K2 slice
exists. S2 and S3 are **not yet tested**; they will be restated in the contracts
that test them, and they must not be edited in the meantime.

| # | question | Samuel | outcome |
|---|---|---|---|
| S1 | official-stack repair changes the current K2 score by | `<1%` | **HIT** — exactly 0%; the multiplier is pinned at the 0.1 floor before and after |
| S2 | Kerdock16 vs matched-row IID at 32,768 directions | `2.7x`–`20x` | open historical carrier comparison; not today's gate |
| S3a | raw-MSE gain of faithful recurrent augmented K2 over K2p2, standalone | `<1.1x` | open; current `GAME_PLAN.md` Track 4 |
| S3b | official 0.4 s residual-cap failure risk | aug-K2 at risk, carrier safe | open |

S2 is still a valid historical prediction for **Kerdock16 versus matched-row IID**,
but it is not a promotion gate. The current leader at 2026-08-31 14:57 UTC implies
multiplier about `0.248`, close to the dense cost of 16,384 forward directions.
That equals eight antipodal MUB bases but also many other 16k-point samplers; it is
evidence for an operating envelope, not evidence that the leader uses Kerdock.

## Closed and open lanes

| lane | state | scope |
|---|---|---|
| basic K2/K2p2 | control / conservative branch | sound but not winning alone |
| higher Gaussian covariance-map orders | low value | measured plateau near 1.052x |
| marginal ReLU mean Edgeworth | closed | capped near 1.047x in CP3R |
| recurrent augmented K2 joint state | **open** | never tested by CP3R |
| exact sparse-aware K3 / W-EFF, FULL TOWER | **closed on a measured price** | `1.0732 B` whole-path, `3.0664x` over its own preregistered screen, after four independent passes. The exact operations survive |
| late-window / age-window K3 | **closed on a measured negative** | every window arm provably cannot clear the `2x` bar. It failed because its **K2 prefix** is bad (`89.2%` of the drift), not because `kappa_3` is non-local |
| cost-indexed structured carrier + exact radial | **highest-priority open branch** | Kerdock/MUB is valid but not yet grader-legal; select a frozen carrier/sample frontier by task-specific evidence and adjusted score, with 8 a telemetry clue and 16 a diagnostic endpoint |
| ordinary IID/random orthogonal | control only | tested arms lose |
| broken two-cycle lattice | closed | does not bound valid QMC/MUB |
| query-directed/adjoint K3 response | **open, speculative** | contract local K3 sources directly into final queries; prove small-shape response parity and price before truth |
| checkpoint response closure | **open after state path** | compositional mechanism; rollout/cross-fit required |
| neuron-specific skew/shape mixture | conditional | run only after supported AUGMENT state and matching residual evidence; a global scale mixture collapses by homogeneity |
| equivariant learned state corrector | conditional | rollout/cross-fit required |
| full factorized K3 | accuracy anchor, not deployable | organiser-confirmed `3.9249e-8` raw (GT-noise floor `7.478e-11`); best exact W-EFF spelling costs `1.0732 B` |
| arbitrary K3 rank capping | strong negative / weak open only with mechanism | rank-`n` measured poor; no impossibility theorem |
| coupled residual sampler | deprioritized | float32/headroom premise unsupported |

## Authority boundaries

- Do not edit `docs/HUMAN_MODEL.md`.
- Do not edit `estimator.py`, claim a seed block, reopen a killed lane, touch a
  vault, change a gate/residual model/protocol, or make a write-up claim without the
  required STRATEGIC sequence.
- Do not submit without Samuel's explicit approval **naming the exact estimator
  SHA**, and only after the **six-item operational gate** of the Track-0S
  submission contract: hash + archive identity, no dirty TRACKED files,
  `whest validate`, allowance, Samuel's SHA-naming approval, and
  **`preflight.ps1 -Full` GREEN 16/16 with a committed transcript grepped for
  its own verdict line**. `VALIDATION.md` §7 always required that last one;
  the contract first omitted it, and an independent review caught the
  omission.
- After every IMPORTANT or STRATEGIC slice, obtain an independent Skeptic verdict
  from raw artifacts before believing or propagating the result.
- Keep `CURRENT_STATE.md` updated when a lane, baseline, live rule, or next action
  changes. Do not stack another override on top of this file.
