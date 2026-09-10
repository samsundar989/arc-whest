# Stage 0 — written prediction, BEFORE any instrument edit

**Class: IMPORTANT** (instrument repair). Written 2026-09-10, before the first
diff. `AGENTS.md` § *Work classification* requires a written prediction first,
`scripts/preflight.ps1` green, and a LOG entry with a knowledge delta; Rule 2
requires a fresh-context Skeptic afterwards.

**Scope statement, so this cannot be read as more than it is.** Every repair
below tightens an *existing* consumer so it enforces what its own report already
claims. None of them changes a gate threshold, a data tier, the promotion
statistic, the three-tier protocol, or any historical verdict. Where a repair
would have changed a gate or the data protocol it is called out here and
deferred to a STRATEGIC contract rather than implemented.

Source of the defects: `docs/research/2026-09-10_validation_audit.md`,
findings 1-8. Nothing was implemented during that audit.

## What I predict, numbered, before writing any code

| # | Prediction | Direction / magnitude | How it is falsified |
|---:|---|---|---|
| 1 | The Linux memory report already on disk for `estimator_k3_strassen_L2.py` (or v2.6.1) is **missing at least three** of the fields the repaired certifier will require (package versions, weight mode, instrument hash) | it will therefore be **refused** by the strengthened consumer until regenerated | it already carries all of them, and the strengthened check passes unchanged |
| 2 | The strengthened consumer, driven by four mutation fixtures (wrong shape, wrong cap, wrong package version, stale SHA), **refuses all four**; the unmutated fixture passes | 5/5 | any mutation still certifies |
| 3 | `paired_diff_ci(method="t", alpha=0.01)` today returns **exactly** the same interval as `alpha=0.05` | bit-identical | the intervals differ |
| 4 | `paired_compare_rows` today accepts two arms whose rows are in **different MLP order** and returns a finite verdict, silently pairing the wrong networks | it will accept, and the returned `mean_difference` will differ from the correctly-ordered pairing | it already refuses, or order does not change the answer |
| 5 | After the pairing repair, **every existing caller** of `paired_compare_rows` in the repository still runs — because unnamed rows keep the positional path with an explicit recorded note, and only *conflicting* identities refuse | zero caller breakage; `test_competition_score.py` stays green | an existing caller raises |
| 6 | `seed_ledger.claim_range` has a **read-check-append** window with no interprocess lock, so two concurrent claimers can both pass the pre-append check | a scripted two-process race will produce **overlapping claims** against today's code, and **zero** overlaps after the repair | today's code already refuses the race |
| 7 | There is **no** record anywhere of the three P08 reads as *reads* — `seed_blocks.py` records them in prose only | the new read registry starts **empty**, and P08's three historical reads are entered as backfilled, explicitly-marked historical entries, not as fresh reads | a machine-readable read log already exists |
| 8 | For two arms with **unequal multipliers**, the common-label noise term does **not** cancel from the adjusted-score difference; a synthetic check with a known noise floor `v` will show a residual bias of exactly `(q_c - q_k) * v` to within Monte-Carlo error | bias present, matching the closed form to <5% at n=2000 | the bias is absent, i.e. noise does cancel |
| 9 | Splitting the certifier's verdict into three explicit states (contract / local runtime / server forecast) **does not change** the verdict of any stored certification report when replayed | the L2 candidate's stored REFUSED/CERTIFIED disposition is unchanged; the two historical gate failures remain FAILED | a stored report's disposition flips |
| 10 | Adding `*.py eol=lf` to `.gitattributes` changes **no** tracked Python file's committed bytes on this checkout | `git diff` empty after `git add --renormalize .` | some file renormalises |
| 11 | Full-repo cost of all Stage 0 repairs: **no** change to `estimator.py`, to any `estimator_v*.py`, to the vault seal, to the touch counter, or to any seed block's claimed state | all four unchanged; `preflight` step 5/6 stay green | any changes |

## What I expect to be hard, recorded before I find out

- **Finding 2 (vault lifecycle).** The vault is 256x32 Phase-1 and its runner is
  not `lifecycle_eval`. Rewiring it is the repair with the largest blast radius
  and the one most able to invalidate a historical artifact. Prediction: I will
  route the *consumer* through `lifecycle_eval` behind an explicit round-aware
  switch, verify it on **sacrificial synthetic fixtures only**, and touch
  neither the seal nor the counter. If that cannot be done without disturbing
  the seal, I stop and leave it as the recorded blocker.
- **Finding 8 (`out=` / mutable buffers).** No implementation. It is an API
  question, and the answer is not in this repository. It stays a written
  question.

## What would make me stop

Any repair that turns out to change a gate threshold, a tier boundary, or the
promotion statistic is **STRATEGIC**, not IMPORTANT. If I find one mid-slice I
stop, record it, and do not proceed on this prediction.
