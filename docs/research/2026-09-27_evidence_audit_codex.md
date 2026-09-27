# Evidence audit — 2026-09-27

Read-only audit of the repository's empirical state, followed by this new report only. No estimator edits, evaluations, training launches, submissions, or designations were performed. PLAN was read first; archived pre-Sep21 strategic documents were not used to make recommendations. Labels: **[M]** measured or directly inspected; **[I]** inference; **[U]** unresolved. Live competitor figures below were supplied by the coordinating agent's Chrome inspection, not independently browsed by this auditor.

## Executive finding

The implementation is a credible, substantially optimized member of the publicly described cumulant-propagation family. It is **not validated as a winning architecture**. Its latest major proposed accuracy and cost improvements have already been tested and failed their score goals, although PLAN still presents them as future work. The new evidence requires a strategy revision rather than another execution of the same experiments. [RESULTS.md:225](C:/Users/samsu/Documents/arc-whest/RESULTS.md:225), [Sep27 packet:1](C:/Users/samsu/Documents/arc-whest/docs/STRATEGIST_PACKET_2026-09-27.md:1), [PLAN.md:28](C:/Users/samsu/Documents/arc-whest/docs/PLAN.md:28).

All quoted final-layer raw MSE values have an organizer-label noise floor of approximately **7.7e-11** (measured range 4.9e-11–1.1e-10 on 20 FIT networks). It is about **0.35% of our current raw error**, so label noise does not explain the competitive gap. This is the scored layer's floor; the approximately 6.8e-10 floor belongs to layer 1. External Monte Carlo moment labels described in the repository as “exact” remain finite-sample estimates. [CURRENT_STATE.md:29](C:/Users/samsu/Documents/arc-whest/docs/CURRENT_STATE.md:29), [Anatomy 2](C:/Users/samsu/Documents/arc-whest/RESULTS.md:492).

## 1. Submitted artifacts and present competitive gap

The saved graded reports directly confirm the following. [A report](C:/Users/samsu/Documents/arc-whest/experiments/run_reports/v68_canary_332478_graded_report.json:1), [B report](C:/Users/samsu/Documents/arc-whest/experiments/run_reports/v68ws_canary_332544_graded_report.json:1).

| Artifact | Public adjusted score | Public final raw MSE | FLOPs / 2^41 | Counted FLOPs / MLP | Server wall median / worst |
|---|---:|---:|---:|---:|---:|
| Best A: v68_wino, #332478 | **4.703856080967253e-9** | 2.1905768576857555e-8 | 0.214731387509 | 472,199,314,830 | 84.00 / 93.86 s |
| Latest B: v68_ws, #332544 | **5.066365933886421e-9** | 2.1906279279448882e-8 | 0.231274597993 | 508,578,219,406 | 51.89 / 55.67 s |

Both completed 100/100 networks with no failures. A has 22,701 operation calls, maximum server residual 0.2521 s, and WSL VmPeak 6,736 MiB. B has 19,361 calls, residual 0.2063 s, and VmPeak 5,085 MiB. Their projected noisy-day worst walls, using measured median times 1.24, are 104.2 s and 64.3 s respectively; these projections are not observed private-run guarantees. [RESULTS.md:220](C:/Users/samsu/Documents/arc-whest/RESULTS.md:220), [DESIGNATION.md:29](C:/Users/samsu/Documents/arc-whest/docs/DESIGNATION.md:29).

B uses the same numerical approach with Strassen leaf 64 instead of leaf 32. It spends about 7.7% more counted arithmetic but runs much faster because the server executes the larger multiplication leaves more efficiently. The raw prediction accuracy is effectively equal within float32 arithmetic differences. Neither designation is performed by agents; these are the recorded recommended pair.

**Public/private distinction:** each saved report has 100 telemetry records and 100 completed MLPs, but only **50 public score records**, and the score aggregate is explicitly `aggregates.public`. The other 50 MSEs are sealed. The displayed leaderboard score must not be described as a publicly known 100-network mean. [A report](C:/Users/samsu/Documents/arc-whest/experiments/run_reports/v68_canary_332478_graded_report.json:1).

The coordinating agent's Sep27 live Chrome inspection reported J2W #332617 at **1.761e-9 adjusted, 1.615e-8 raw, 10.91% budget**, and Luna #332577 at **1.907e-9 adjusted, 1.422e-8 raw, 13.41% budget**. The leaderboard rounded display/link was stale relative to Insights and the newer submission. These supersede PLAN's Sep26 board snapshot; source details should be preserved in the coordinating agent's browser report. [Competition Insights](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/insights), [stale PLAN snapshot](C:/Users/samsu/Documents/arc-whest/docs/PLAN.md:11).

Using those rounded live figures, our adjusted score is about **2.67 times J2W's**: approximately **1.36 times its raw error** and **1.97 times its arithmetic cost**. This is descriptive cross-entry arithmetic, not a paired statistical estimate of method superiority. Closing the full gap requires about a 62.6% reduction in our score. At our present cost, beating 1.761e-9 requires raw MSE below about **8.20e-9**. Even reaching the 0.1 cost floor with our present raw MSE yields **2.1906e-9**, still behind J2W. Both error and cost matter.

Our score is below the widely publicized 504aldo V29 score of 5.40e-9, but its published raw 2.13e-8 is slightly lower than our graded 2.19e-8. A matched local benchmark is needed before claiming an accuracy advance over that public implementation; different measurements and historical submissions are not a paired comparison. The more recent leaders' internals are not established by their scores. [Public-frontier source ledger](C:/Users/samsu/Documents/arc-whest/docs/research/2026-09-25_E_literature_frontier.md:113).

## 2. What has actually been measured

### 2.1 Incremental optimization works, with diminishing returns

Seven graded steps took v59 from 4.9756e-9 to v68 at 4.7039e-9, a 5.46% score improvement. The differentiable twin imports the shipped estimator under a torch shim, so fitting changes parameters in the actual computation rather than an unrelated surrogate. It is checked against the official numerical path at panel level. [Twin implementation evidence](C:/Users/samsu/Documents/arc-whest/RESULTS.md:422).

| Recent change | Evidence and effect |
|---|---|
| End-to-end corrector fitting, S1 | DEV approximately 0.62% raw improvement; validation 0.76% |
| Per-layer profiles, S2 | DEV approximately 0.48% raw improvement; validation 0.42% |
| Per-term gains, S2t | DEV approximately 1.95% raw improvement; graded approximately 2.06% |
| Continued tuning, S2t2 | DEV approximately 0.71%; graded approximately 0.58% |
| Multi-mode fourth-order slice, S3 | DEV approximately 0.86%; graded approximately 0.98% |
| Strassen-Winograd | Approximately 0.79% fewer FLOPs, 1,947 fewer calls, raw essentially unchanged |

These local changes have transferred to graded submissions, which supports their reality. They do not add up to evidence that further parameter tuning alone can close a 2.67-fold score gap. [RESULTS.md:210](C:/Users/samsu/Documents/arc-whest/RESULTS.md:210), [RESULTS.md:439](C:/Users/samsu/Documents/arc-whest/RESULTS.md:439).

Winograd reduced measured median server wall by **6.7 s**, while the previous FLOPs-plus-calls model predicted only about 1.5 s. Memory traffic and temporary layout therefore deserve investigation, and the old wall model is incomplete for such changes. Earlier temporary release reduced the new implementation's WSL peak from 7,727 to 6,736 MiB without changing arithmetic. [RESULTS.md:220](C:/Users/samsu/Documents/arc-whest/RESULTS.md:220), [Winograd experiment](C:/Users/samsu/Documents/arc-whest/RESULTS.md:363).

### 2.2 Conventional representation knobs are locally exhausted

The rank sweep on the same 100 DEV networks is strong evidence that 256 is a local score optimum in the current representation:

| Shared rank | Change in raw MSE | Change in FLOPs | Change in score |
|---|---:|---:|---:|
| 192 | +21.0% | -5.75% | +14.0% |
| 224 | +6.77% | -1.81% | +4.84% |
| 256 | control | control | control |
| 288 | -3.45% | +3.87% | +0.29% |
| 320 | -5.47% | +7.89% | +1.99% |

Negative score change would be better; no tested alternative wins. Earlier birth-dependent folding also worsened score. These arms were not jointly refitted, so this is a measured local optimum under the tested protocol, not a proof over every possible training procedure. [Rank sweep](C:/Users/samsu/Documents/arc-whest/RESULTS.md:383), [fold sweep](C:/Users/samsu/Documents/arc-whest/RESULTS.md:401).

Single-leg and source-dropping experiments showed very large losses: keeping only simplified old diagonal content does not preserve the useful third-order information. Fully memoryless regeneration and the tested fitted regeneration failed. [Single-leg evidence](C:/Users/samsu/Documents/arc-whest/RESULTS.md:1037).

### 2.3 The harmonic radial-only replacement failed

Removing the old tier entirely saved about **28% FLOPs** but increased DEV100 raw MSE **17.8 times**. The best tested radial harmonic arm remained **5.93 times worse** on DEV100. The covariance-metric projection became severely ill-conditioned, reaching approximately 1e15 from near-dead neurons, so well-conditioned alternatives were tested. [Harmonic experiment](C:/Users/samsu/Documents/arc-whest/RESULTS.md:321).

A one-network diagnostic showed the radial component recovering about 70% of the old tier's excess error. That still leaves a large absolute error because the tier is so valuable. The justified conclusion is that **the tested radial-only summary loses indispensable information**. It does not close every harmonic representation, every low-rank traceless remainder, or every task-aware tensor approximation.

### 2.4 Oracle substitutions show remaining representational headroom

Replacing selected states with accurate external Monte Carlo moment labels at every layer, then refitting the remaining trainable parameters, gave:

| Substituted channel | Best refit final raw MSE | Error reduction versus control |
|---|---:|---:|
| Control | 2.158156e-8 | — |
| Covariance | 1.283910e-8 | 40.5% |
| Third cumulant | 1.005448e-8 | 53.4% |
| Fourth cumulant | 1.557046e-8 | 27.9% |
| All except mean | 1.845817e-9 | 91.4% |

These reductions overlap and must not be added. They are diagnostic ceilings involving unavailable-at-inference information, not deployable candidate scores. The twenty evaluation networks were withheld from this short refit but already used in previous training of the base estimator; see the split audit below. [Ceiling experiment](C:/Users/samsu/Documents/arc-whest/RESULTS.md:225).

### 2.5 The tested learned closure is score-negative

Offline residual prediction was promising: held-out network R-squared values ranged from roughly 0.6 to 0.87 for several scalar outputs. Yet deployment inside the complete recurrence changed the result:

| Arm | Validation final MSE | Raw change versus S3 | Cost implication |
|---|---:|---:|---|
| S3 base | 2.264823e-8 | — | reference |
| Continued S3 control | 2.260955e-8 | -0.17% | same architecture |
| Offline closure initialization | 2.379587e-8 | +5.1% | worse before in-loop fit |
| Both networks, in-loop fit S5 | 2.247849e-8 | -0.75% | +1.68% FLOPs; score-negative |
| Per-neuron network only S5u | 2.260988e-8 | -0.17% | essentially identical to continued control |

v69_clos reproduced the twin but was not submitted. Existing compensations plausibly absorb most of the tested closure correction's useful content; the experiment does not establish a universal impossibility result for learned corrections. [RESULTS.md:247](C:/Users/samsu/Documents/arc-whest/RESULTS.md:247).

The closure bench did not expose the large third-cumulant drift in the marginal and repeated-index outputs. The packet therefore locates the missing mechanism in all-distinct births/propagation. This is an evidence-informed hypothesis; a direct measurement of the complete distinct-index tensor error is still missing. [Closure bench](C:/Users/samsu/Documents/arc-whest/RESULTS.md:275), [packet interpretation](C:/Users/samsu/Documents/arc-whest/docs/STRATEGIST_PACKET_2026-09-27.md:29).

## 3. Current in-flight work

Read-only workerq inspection found job **2229** running S2t3 on **TRAIN900 plus 2,048 d8b networks**, approximately 48% complete at inspection. Declared resources were 16 GiB host RAM, 12 GiB VRAM, and 4 CPUs. No second experiment was launched by this audit.

The latest validation checkpoint observed was about **2.2476e-8**, approximately 0.76% below S3's 2.264823e-8. This is provisional, selected on validation, ungraded, and consistent with the worker's modest 0.3–0.8% prediction. The log remained in progress on the follow-up read. [Prediction record](C:/Users/samsu/Documents/arc-whest/experiments/2026-09-27_S2t3/PREDICTIONS.md:1), [training log](C:/Users/samsu/Documents/arc-whest/experiments/2026-09-27_S2t3/train.log:1).

## 4. Split integrity and evaluation semantics

**[M] Original split identities are sound.** The mini100 and full1000 have zero overlap, and recorded DEV/FIT weight hashes cover all 200 networks. Original membership was DEV=mini0–99, FIT=full0–99, HOLDOUT=full100–999, hence 900 holdout networks rather than the 800 in older prose. [Panel evidence](C:/Users/samsu/Documents/arc-whest/RESULTS.md:745), [panel code](C:/Users/samsu/Documents/arc-whest/tools/panels.py:64).

**[M] Training was subsequently expanded with recorded authorization.** Sep25 explicitly authorized TRAIN=FIT100 plus 800 former holdout networks. The pinned current split is full0–899 for training and full900–999 for validation, with no index overlap. DEV remains mini0–99 and is not directly trained on. The recent “TRAIN800” wording is stale; the actual code and S2t3 predictions consistently use 900. [Authorization record](C:/Users/samsu/Documents/arc-whest/docs/HANDOFF_2026-09-25.md:24), [pinned split](C:/Users/samsu/Documents/arc-whest/data/panels/twin_train_split.json:1), [twin panel selection](C:/Users/samsu/Documents/arc-whest/tools/twin/twin.py:124).

**[M] A stale evaluator route can mislabel training data as holdout.** `tools/panels.py` still reads original `holdout.json` membership full100–999; `tools/ev.py --panel holdout` defaults to n=100, hence full100–199, all in current training. Such a run is not independent validation for the current estimator. The files inspected contained 800 original holdout indices now in training. [Panel loader](C:/Users/samsu/Documents/arc-whest/tools/panels.py:126), [evaluator default](C:/Users/samsu/Documents/arc-whest/tools/ev.py:687).

**[M] HOLDOUT100 is a validation set.** Training repeatedly evaluates it and selects the best checkpoint/early stopping from those values. It is not an untouched final test. The comment “read once per stage” does not match the code's evaluation loop. DEV is also repeatedly used for candidate design and selection, although no direct gradient training on DEV was found. [Evaluation callback](C:/Users/samsu/Documents/arc-whest/tools/twin/twin.py:558), [checkpoint selection](C:/Users/samsu/Documents/arc-whest/tools/twin/twin.py:637).

**[M] Ceiling HOLDOUT20 is not globally unseen.** It is full80–99, withheld from fitting this diagnostic's 80-network refit but included in the earlier 900-network base training. The experiment remains useful for comparing mechanistic substitutions, with that limitation. [Ceiling protocol](C:/Users/samsu/Documents/arc-whest/RESULTS.md:227), [panel indices](C:/Users/samsu/Documents/arc-whest/tools/twin/twin.py:135).

There is no evidence here of prohibited training. The concern is misleading independence claims and an obsolete panel route. Before another claim of unseen-data transfer, names and routing should be reconciled, and validation selection should be distinguished from any final evaluation on genuinely new networks. This report does not authorize or perform a new data read or mint.

## 5. Reproducibility and documentation hazards

1. **PLAN is overtaken by results.** Steps 1–3 completed early and the two hoped-for jumps failed. The Sep27 packet records the need for a new direction. [PLAN](C:/Users/samsu/Documents/arc-whest/docs/PLAN.md:28), [packet](C:/Users/samsu/Documents/arc-whest/docs/STRATEGIST_PACKET_2026-09-27.md:1).
2. **CURRENT_STATE is stale.** Its timestamp says Sep25, references rev13, lists B=v64_ws, and calls completed anatomy/twin work in-flight. DESIGNATION's current table correctly has v68_ws, but later historical prose still discusses old v59/v60 choices. [CURRENT_STATE](C:/Users/samsu/Documents/arc-whest/docs/CURRENT_STATE.md:1), [DESIGNATION](C:/Users/samsu/Documents/arc-whest/docs/DESIGNATION.md:29).
3. **The evaluator's paired statistic is appropriate.** It computes per-network arithmetic score differences with a studentized-t interval, wins/count, and a ratio of the paired panel means. Keep per-network pairing when comparing methods; raw ratios of unrelated public averages do not supply uncertainty. [Paired implementation](C:/Users/samsu/Documents/arc-whest/tools/ev.py:480).
4. **Cost must use the real f64 grader twin.** Saved calibration verifies its counts against graded runs. The `whest run` float32-input path misses the opening cast cost of 2^25 FLOPs. The counting shim is not a billing authority. [Calibration](C:/Users/samsu/Documents/arc-whest/RESULTS.md:45).
5. **Backend parity is panel-level, not bit equality.** Float32 accumulation order produces per-network MSE differences around 1e-3 between backends while panel means agree much better. Percent-level screening is well supported; tiny effects require the official numerical path. [Backend evidence](C:/Users/samsu/Documents/arc-whest/RESULTS.md:752), [twin parity](C:/Users/samsu/Documents/arc-whest/RESULTS.md:422).
6. **Runtime is an empirical cliff.** Identical submitted bytes have shown materially different server tails. Current A/B measurements and observed noisy-day behavior should be preserved; a local absolute runtime is not a server runtime. [Wall calibration](C:/Users/samsu/Documents/arc-whest/RESULTS.md:65).
7. **Active experiment artifacts were present.** The working tree had changing ceiling checkpoint/log files and untracked S2t3/closure checkpoints; these were not altered. Any future code integration must account for the other ongoing worker.

## 6. Implications for the next design discussion

**Supported:** keep v68 as a functioning baseline, preserve its fast evaluator and faithful billing/memory tests, and separate an incremental canary from a genuinely new architecture experiment. Current cost/accuracy knobs and the two tested major proposals are mapped well enough to avoid rerunning them blindly.

**Most plausible remaining research target [I]:** distinct-index third-order information—how it is born, transported, and compressed. A low-rank traceless remainder, a representation chosen for the final estimator's contractions, or a different birth approximation could address the observed failure mode. None is validated here. The claim that leaders carry old sources nearly free remains a hypothesis from their telemetry, not confirmed knowledge of their implementations.

**Cost work remains useful but must be priced.** Layout can create runtime and memory headroom, yet leaf16 previously increased counted FLOPs in this implementation: 0.22014 B versus 0.21618 B, with 26,773 versus 23,645 calls. A rewritten layout could change that tradeoff, but deeper recursion must not be assumed to reduce billed cost. [Leaf16 result](C:/Users/samsu/Documents/arc-whest/RESULTS.md:1323).

**Do not overstate negative results.** Harmonic radial-only failure does not refute every harmonic or tensor method. A trained closure MLP's failure does not refute every learned state correction. Conversely, high offline R-squared or a large oracle ceiling is not evidence of a shippable score gain. The next proposal needs a specific preserved mathematical quantity, counted cost before a large run, and paired score improvement after integration.

The empirical record supports confidence in the existing baseline and skepticism about another round of minor tuning producing a win. It supports investigation of a structural change; it does not yet identify or validate the winning one.
