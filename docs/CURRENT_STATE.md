# CURRENT_STATE.md — read this first

> ## 2026-09-10 (Slice 5, day 1) — **THE COST LANE IS CLOSED AND NO AFFORDABLE κ₄ SPELLING EXISTS BY DELETION.** Two pre-registered gates failed the same day. The Skeptic returned PROVISIONAL on T1 and struck eight of my sentences; the gate outcome survived, the attribution did not.
>
> `estimator.py` untouched at **v2.6.1**. **Nothing designated. Nothing promoted.
> Nothing submitted.**
>
> ### THE FOUR-LINE ANSWER
>
> 1. **T1 — no affordable κ₄ spelling exists BY DELETION**, and that is the
>    narrowed claim the Skeptic left standing. Deleting κ₄'s κ₃ births while
>    keeping the `(n,n)` object live is `74x`–`109x` **worse than SIMPLE**, on
>    `0 of 24` nets improved. **But the ablated arms land on the GAUSSIAN
>    BASELINE** (`LITE` = K2 to within `4%`, interval through zero), so they say
>    nothing about *which* κ₄ channel carried the `2.094x` — and the two channels
>    are **not separable in this implementation** anyway, because κ₄'s own update
>    is computed from the blocks the ablation deletes. **The channel attribution
>    is struck as unidentified.**
> 2. **Packed spelling — closed at `1.0067x`** against a `1.5x` gate. Zero of the
>    chain's `797` matmuls has the redundancy this meter discounts. **Cost is now
>    closed at `~0.5 B`**, which meets the Slice-4 kill criterion thirteen days
>    early and fails it.
> 3. **Nominee A (Strassen depth 2) certifies on 6 of 8 gates.** The two failures
>    are one artefact measured twice, and it would refuse `330327` — which has a
>    real graded `0/100` row — for the same reason. **The canary is HELD** pending
>    Samuel, because I will not reinterpret my own failed gate.
> 4. **Rules v14 §6 is a checked negative.** Three independent public routes
>    tried; the sentence is not publicly stated and another team hit the same wall.
>    A neutral forum post is drafted and needs Samuel's account.
>
> ### T1 — THE NUMBERS, AND THE EXPLANATION THE EVIDENCE ACTUALLY IDENTIFIES
>
> | arm | births/layer | mean xmse | vs SIMPLE |
> |---|---:|---:|---:|
> | `SIMPLE` | `2n` | `4.004685e-08` | `1.000x` |
> | `AUGMENT` | `8n` | `1.912323e-08` | `2.094x` better |
> | `AUGMENT-LITE` | `5n` | `4.365096e-06` | `109x` WORSE |
> | `AUGMENT-MINIMAL` | **`2n`** | `2.976232e-06` | `74x` WORSE |
> | *Gaussian K2, same seeds/labels* | — | `4.196350e-06` | `105x` worse |
> | *calibrated K2* | — | `1.448599e-06` | `36x` worse |
>
> GT noise floor `1.1552e-09`. Controls are **bit-identical** (`0.000e+00` on all
> 24 rows, not the `~1e-15` I first wrote — that was rounding in my own contract).
>
> **Read the baselines before the arms.** `LITE` **is** K2: the paired difference
> straddles zero, `13/24`. `MINIMAL` is worse than *calibrated* K2. Deleting
> blocks from the carried lineage does not cost part of κ₄'s contribution — **it
> costs the entire κ₃ correction.**
>
> **The simpler explanation was already on disk, in an entry I wrote the day
> before.** Generation-capped SIMPLE — no κ₄ births anywhere — collapses to
> `1.651e-06`. **One statement covers every case measured: any deletion from the
> carried `(1,1,1)` lineage returns this chain to roughly Gaussian accuracy.**
>
> **A non-monotonicity I missed:** `MINIMAL` deletes strictly more than `LITE` and
> is `1.47x` **better**, `21/24`. My prediction 4's reasoning had the direction
> wrong, not just the magnitude.
>
> **STRUCK:** the width claim ("installs a width-independent error floor"). It
> contradicted itself inside one sentence, both axes moved between the two rows,
> and the small row was a 4-net smoke whose numbers **exist nowhere on disk** —
> and it had already reached the write-up. Removed from both.
>
> **In flight:** job `498` completes the 2x2 with the missing `legsonly` cell
> (`4n`, births kept, legs dropped) and the baseline columns. Pre-registered:
> `legsonly` also lands at `~1e-6` and `K2 - legsonly` straddles zero, `~80%`. If
> so the κ₄ attribution is dead and the Skeptic's statement is the finding.
>
> ### EVERY PREDICTION ABOUT THE SIZE WAS WRONG — THE STRATEGIST'S AND BOTH OF MINE
>
> The strategist's band was `1.4x–1.8x`; the answer is `0.0135x`, wrong by `~150x`.
> Mine were `[0.6, 1.1]` and `[0.35, 0.9]` — right in direction, wrong by two
> orders of magnitude, and prediction 4's stated **reasoning** was wrong in
> direction too. **The instructive failure:** I had the competing explanation on
> disk, in my own entry from the day before, and never tested my story against it.
> The instrument's premise — that the two channels are separable — was checkable
> in the upstream source in ten minutes, and I did not check it. **The Skeptic
> did, and found it false.**
>
> ### NOMINEE A — DEPTH 2, GATE BY GATE
>
> Candidate `estimator_k3_strassen_L2.py`, sha256 `390bc226…`, one character from
> `330327`'s bytes (`_STRASSEN_DEPTH 3 -> 2`). `0.604217 B`, `9,321` ops.
>
> | gate | state |
> |---|---|
> | (a) `whest validate` | **PASS** |
> | (b) certify §1–11 no FAIL | **FAIL**, §10 only, both clauses the artefact |
> | (c) grader-twin f64 read-only | **PASS**, `0.000e+00` relative, FLOP delta exactly one weight pass |
> | (d) `edge_cases --real-flopscope --phase2` | **PASS** |
> | (e) ops `<= 24,000` | **PASS**, `9,321` |
> | (f) projected residual `<= 0.3 s` | **PASS**, `0.125 s`, `3.20x` under the cliff |
> | (g) Linux 8-GiB gate on the sha | **PASS**, VmPeak `3,328 MB`, `2.46x`, matched on `390bc226…` |
> | (h) mini-100 `n_failed_mlps: 0` | **FAIL**, the artefact |
>
> **Three instruments agree on the price to `0.008%`**: the shape model, the real
> flopscope run, and the Linux gate's independently computed `0.6042021`.
>
> The nine mini-100 rows that "raised" were **the run, not the candidate**: `0 of 9`
> reproduce in isolation, all finite, all with byte-identical FLOP counts. Eight of
> the nine indices were contiguous while another project held 20 GiB for 41 minutes.
>
> **WHAT SAMUEL NEEDS TO DECIDE:** gates (b)-§10 and (h) are the same
> mis-specification — they judge the residual in local seconds, which this host
> reads `27x` high per operation, instead of on the op count that
> `VALIDATION.md` §2b-iii makes the gate. They would refuse `330327`. I wrote them
> before the data and they failed, so amending them now is choosing a gate after
> seeing the answer. **The canary is held until he acknowledges that.**
>
> ### T2 — CONTRACTED, AND THE HANDOFF'S CORPUS REFUSED
>
> The handoff says to train the corrector on the organisers' public 1,100.
> **`docs/VALIDATION.md` §0 forbids fitting anything on the public split** and
> `AGENTS.md` makes it binding, so the law wins over the packet. `P07` (384, the
> fit) and `P08` (128, the single held-out read) are allocated, claimed, and their
> labels are minting at `2^24` per half. The public split is read once at the end
> as a check that chooses nothing.
>
> **This changes what a negative T2 would mean and is flagged for the next packet.**
>
> **The label-precision cost I claimed was wrong, and the correction is in our
> favour.** P07's labels are minted and measured: noise `4.4988e-09` per replica,
> `2.2494e-09` in a two-replica mean, against a chain MSE of `4.0047e-08`. **The
> `R^2` ceiling is `0.944`, not the `~0.40` my own contract predicted.** I had
> assumed a post-ReLU activation variance of order 1; the measured value is
> `0.0755`, a factor of `13` in variance. `2^24` per half was therefore
> over-provisioned by about three GPU-hours. The replica-independence check passes
> at ratio `1.0076`, so the cross estimator is unbiased and **the gate does not
> move** — the ceiling governs only how an `R^2` is read.
>
> ### IN FLIGHT
>
> - `498` — T1's completed 2x2, the Skeptic's discriminating follow-up.
> - `493`/`494` — T2 labels for P07 and P08. **~3.3 h and ~1.5 h; my 22 s/net
>   estimate was `1.8x` low (measured `39.4 s/net`), and the ETAs are corrected
>   in the queue.**
>
> ### THE LEDGER GATE WENT RED MID-FLIGHT AND IS NOW GREEN AGAIN — kept as a note
>
> Claiming `P07`/`P08` with their artifact paths left `preflight` step 7
> (`seed_ledger.py --check --scan`) reporting **2 ERRORS** for several hours: an
> artifact named in a claim but never hashed. That is the designed order — the
> claim is written FIRST so an interrupted run leaves a claim rather than a silent
> draw — and the fix is `seed_ledger.finalize(...)` on the host once the file
> exists. **Both are finalized against their job-reported sha256
> (`b96f90a0…`, `4c7a8da9…`) and the check reads `CONSISTENT`.**
>
> Kept here because the window is real and a successor could land in it: **between
> claiming a block with an artifact path and finalizing it, preflight is red and
> nothing may be promoted.** Finalize on the host as soon as the job lands.
> - `497` — depth-2 mini-100 with the residual cap lifted: the replacement gate,
>   and the first accuracy read on the organisers' data (the capped run produced
>   **none**, because every row was zeroed).
> - Not yet queued: the κ₄ rank-sweep (write-up lane), the T2 fitter.
>
> ### PROCESS DEFECTS THIS SLICE
>
> - **I handed `predict` bare numpy weights in two probes.** `certify` hands
>   flopscope-backed arrays because *"operations invoked as METHODS ON A WEIGHT are
>   metered on the grader and free on bare numpy"* — a trap this repository had
>   already found, closed and written down on 2026-08-24. My figures under-read by
>   `144,472,064` FLOPs and `414` ops. **Read `build_weights` before writing a
>   probe.**
> - **A pre-registered prediction quoted a pricing instrument's number as the
>   file's**, with a zero-width interval: `9,304 +- 0` against an actual `9,321`.
> - **A third stale scalar** in `estimator_k3_strassen.py`'s docstrings: last-layer
>   pruning saves `4` matmuls per lineage and `60` of `857`, not "5 per lineage,
>   75 of 1,022". The file is NOT edited — its sha is graded bytes.
> - **`Add-Content -Encoding utf8` double-encoded 38 lines of `LOG.md`.** Repaired.
>   **Append to `LOG.md` from bash, never from PowerShell.**
> - **The prediction slot for the P07/P08 allocation is VOID** — I edited and ran
>   the tests before writing one.
> - **A latent statistics bug**, found by the Skeptic: `crit = 1.96 if n > 60 else
>   2.064` is `t` at `df=24` used for every `n` from 2 to 60. scipy is installed in
>   both venvs so it never fired and **no published interval is affected**, but
>   T2's `1.15x` gate would have inherited it. Replaced with an exact `df 1..30`
>   table plus a Cornish-Fisher tail, and the artifact now records which branch ran.
> - **I claimed controls at `~1e-15` that were bit-identical.** Understating a
>   control is still misreporting it.
>
> ### A PROVENANCE DEFECT WORTH FIXING PROPERLY
>
> A `workerq` git snapshot materialises **CRLF** under `core.autocrlf=true` while
> the working copy is **LF**. The first memory-gate run therefore certified a
> different sha for the same code, and `certify` §9 would have skipped. Fixed here
> with `--no-snapshot`. **The general form is worse: a fresh clone of this
> repository does not reproduce the graded submissions' shas.** Owed: a
> `.gitattributes` `*.py eol=lf` entry, as its own IMPORTANT change.


> ## 2026-09-10 (Slice 4, day 2) — **THE AUGMENT PORT IS DEPRIORITIZED ON EVERY AXIS.** Over the hard FLOP budget at full tower; its `2.09x` does not survive any approximation of the tower; and the one spelling that fits is beaten by the incumbent on arithmetic alone.
>
> `estimator.py` untouched at **v2.6.1**. **Nothing designated.** Nothing promoted.
> Three Rule-2 Skeptics ran on this slice. **All three found real defects and two
> changed a conclusion.** Their corrections are folded in below and in
> `experiments/LOG.md`; where this file and an older LOG entry differ, this file
> is the corrected one.
>
> ### THE THREE-LINE ANSWER
>
> 1. **Cost.** `kind=AUGMENT` births `8n` lineage columns per layer against
>    SIMPLE's `2n` and adds two rank-`R` contractions SIMPLE has no counterpart
>    for. That is **`3.6x`–`5.8x` over the HARD per-MLP FLOP budget** at every
>    Strassen depth. `B` is a cap, not only a multiplier: above `1.0 B` the
>    grader returns **zeros**.
> 2. **Accuracy under approximation.** Capped to the generations at which it
>    would fit, AUGMENT fails its pre-registered gate by `26x`–`58x`. And the
>    collapse is **not AUGMENT's** — capped AUGMENT is statistically
>    indistinguishable from capped SIMPLE at `g=2` and significantly *worse* at
>    `g=4`. **AUGMENT's `2.094x` does not survive truncation of the tower.**
> 3. **The spelling that actually fits.** Only `g=1` clears the budget, and there
>    `_fit_generations` drops to the **regenerating** tower. That closes on
>    arithmetic with no new data: grant regenerating AUGMENT its full `2.094x`
>    and assume it is **free**, and the multiplier floor still puts it at
>    `4.34e-08` against nominee A's graded `2.031e-08` — **`2.14x` worse in the
>    most generous case reachable.**
>
> **DEPRIORITIZED, not formally CLOSED**, and the distinction is deliberate:
> `experiments/2026-08-26_p2_gt/gt_manifest.json` says in writing *"Nothing
> measured on this asset may close a lane"*, and every input above is DESIGN-tier
> P01. A fresh block at a `2.14x` margin on the most generous possible assumption
> would be waste. **Do not later read deprioritized as disproven.**
>
> ### WHAT THIS DOES NOT SAY
>
> - **Not** that D1's `1.912e-8` is wrong. Both controls in the capped run are
>   **bit-identical to D1 on 24/24 rows**. Accuracy and cost are one object.
> - **Not** that `kappa_4`-in-flight is dead. It says *upstream's factored
>   spelling of it* has no viable operating point. The board reaches that
>   accuracy at `0.18–0.21 B`, roughly **`22x` cheaper than the only spelling we
>   have priced.** What representation does that is the open question.
>
> ### SLICE 4 STATUS AGAINST THE HANDOFF
>
> | item | state |
> |---|---|
> | 1. port price | **done** (Skeptic PROVISIONAL, two false sentences corrected) |
> | 2. port, certify, canary | **dropped.** Deprioritized on all three axes above |
> | 3. wall profile | **done**, cheap fix falsified, recommend closing at `~6%` |
> | 4. symmetric-packed spelling | **not started.** Now the whole cost lane |
> | 5. small-width ladder | **done x3.** Proxy check FAILED; dtype excluded |
> | 6. corrector | not started |
> | 7. Rules §6 | **closed as a dead end.** Needs Samuel |
>
> ### ITEM 3 — THE WALL PROFILE, AND A RULE WORTH CARRYING
>
> At Strassen `L=3`: Winograd adds are `37.2%` of backend wall for `1.3%` of
> FLOPs, concatenates `30.5%` for `1.6%`, and **the matmul leaves are `26.6%` of
> wall for `96.8%` of FLOPs.** That is the mechanism behind graded intensity
> collapsing `215 -> 22.8 GFLOP/s`.
>
> **The obvious fix is measured dead.** Merging the seven Strassen call sites a
> layer issues cuts operations `3.0x` and `3.7x` and moves wall by `2%` and
> `-28%`. **The glue is memory-bound: it costs bytes moved, not calls issued.**
> Carry this forward — *fewer, bigger operations buy RESIDUAL and buy nothing on
> WALL.* What remains is Winograd's 15-add form (`~6%`) and an unpriced
> batched/unbatched hybrid.
>
> ### ITEM 5 — THE LADDER CANNOT BE EXTRAPOLATED, AND ONE LEAD CAME OUT OF IT
>
> The pre-registered validity check **failed**: bootstrap CIs on `SIMPLE/AUGMENT`
> exclude the live shape's `2.094` at all three rungs (`2.75 / 3.00 / 2.91`). The
> ladder also **falsifies its own similarity premise from inside** — a method
> ratio moves with `n` at fixed `L/n` (`0.998 -> 0.977 -> 0.948`, non-overlapping
> CIs). Re-running at `float32` reproduced the `float64` numbers to 4–5
> significant figures, so **dtype is dead as the explanation**. Nothing in the
> ladder transfers to `1024 x 16`.
>
> Two readings survive as **orienting only**:
> - **full K4 is `9.0x`–`10.6x` better than AUGMENT** at all three rungs. The
>   arm is verified to be the exact `kappa_4` ceiling (factoring drops none of
>   its 171 terms; `FactoredTensor4`'s rank never truncates) and shares an
>   identical diagram budget with AUGMENT, so it is a clean *representation*
>   comparison. **A large ceiling above AUGMENT is a live hypothesis, not a
>   finding.**
> - **`use_avg_metric=False` is `~23%` more accurate**, flat across the rungs,
>   0 of 24 nets worse on any rung. **But it was measured on AUGMENT, the kind
>   with no budget.** Job `474` adds the shippable `k3_simple_exactmetric` arm.
>   It is not free: `metric.ndim == 2` switches on a ninth born block and breaks
>   our `_proj_coef_d4_r2` scalar reduction.
>
> **SIGN CORRECTION:** `unfactored/factored = 0.948` means `factor=True` **COSTS**
> `5.2%` of accuracy and the cost grows with `n`. It was first read the other way.
>
> ### WHAT THE SLICE NOW TURNS ON
>
> Cost is the whole lane and **item 4 is all of it.** The kill criterion
> ("if neither item 3 nor item 4 has a `>=1.3x` priced saving by Sep 23, cost is
> closed at `~0.5 B`") now rests entirely on item 4, because item 3 is closed at
> `~6%`.
>
> The strategist packet's question is no longer "port AUGMENT". It is: six teams
> reach raw `1.87–2.13e-8` at `0.18–0.21 B` while the only spelling of that
> accuracy we have priced is `22x` dearer, and Strassen depth alone cannot take
> our structure to `0.2 B`. **They are not running a 15-generation exact lineage.**
>
> ### THE EXACT-METRIC LEVER IS CLOSED, AND THE LADDER IS NOW CALIBRATED
>
> `use_avg_metric=False` at the LIVE shape: **`1.070x`** against a
> pre-registered `1.15x` gate. **FAIL, closed.** Real (`24/24` nets, CI clear
> of zero) and too small to buy against a ninth born block plus the loss of
> `_proj_coef_d4_r2`. Cost `41 s` of GPU because accuracy was measured before
> any cost model. Both pre-registered predictions confirmed; control drift
> `+1.3e-15`.
>
> **The ladder over-reads method ratios by `1.2x`-`1.4x`**, now measured twice:
>
>     SIMPLE/AUGMENT       ladder 2.909   live 2.094   over-read 1.39x
>     exact-metric gain    ladder 1.311   live 1.070   over-read 1.23x
>
> Discounting the full-K4 ceiling by the same factor leaves **`6.5x`-`8.8x`**
> at the live shape — still far above the `1.5x` lane threshold, so **the
> ceiling HYPOTHESIS survives calibration** and is the most valuable
> unmeasured quantity in the project. No kappa_4 ceiling has ever been
> measured at `1024 x 16`.
>
> ### IN FLIGHT
>
> - Nothing. **`docs/STRATEGIST_PACKET_2026-09-10.md` is written and is the
>   next action** — it supersedes the Sep 9 packet, whose plan is void.
>
> ### PROCESS DEFECTS THIS SLICE, ALL THE SAME SHAPE
>
> - **Two stale scalars read from prose instead of from the function.**
>   `CURRENT_STATE`'s `215 -> 17 GFLOP/s` (the `17` does not reproduce; `22.82`
>   does) and `_matmuls`' docstring "1,022 matmuls" (**the function returns
>   `797`**, which reproduces the graded `L=0` anchor to `0.17%`).
>   **`estimator_k3_strassen.py` MUST NOT be edited to fix it** — its sha256
>   `3ad05d75…` is submission `330327`'s bytes, nominee A.
> - **Two dtype defaults wrong in one session.** The capped run and the ladder
>   both defaulted to `float64` against `float32` anchors. The capped one reached
>   `8%` in `2h33m` against a `60m` estimate and starved the queue.
> - **Two numbers estimated that the repository already contained.** Prediction 2
>   quoted two rows of `generation_curve.json` and extrapolated past the five rows
>   below them, which held the answer exactly.
> - **My wall-time estimates were wrong twice**, by `28x` and by `4x`.


> ## 2026-09-09 — SLICE 3 STATUS: **Step 1 is done, the cost half is done, and the AUGMENT port is the next slice with its feasibility PROVEN.**
>
> `estimator.py` untouched at **v2.6.1**. **Nothing designated.** Nothing promoted
> — every measurement below is P01 DESIGN, re-read.
>
> ### THE RESULT THAT MATTERS
>
> | arm | budget | κ₄ as | mean xmse | 95% CI |
> |---|---|---|---:|---|
> | K3 `SIMPLE` | base (40 terms) | scalar | `4.004685e-08` | `[3.7943e-08, 4.2151e-08]` |
> | A5 | raised (142) | scalar | `4.3448e-08` | `[4.1498e-08, 4.5398e-08]` |
> | K3 **`AUGMENT`** | raised (142) | **`(n,n)` matrix** | **`1.912323e-08`** | **`[1.8229e-08, 2.0017e-08]`** |
> | A3 | raised (142) | zero | `3.6895e-07` | `[3.4443e-07, 3.9346e-07]` |
>
> `decide()` PASS, empty `reasons`, `2.094x`, `24/24` nets, control reproduced
> **bit-exactly** three times across two reboots. Quote it as **`1.9e-8`** — the CI
> is `±4.5%`, so it carries two significant figures.
>
> **The whole gain is κ₄'s `(n,n)` structure.** The raised diagram budget alone is
> *negative* (A5 is `1.085x` worse than SIMPLE). Both cheap ports are closed by
> measurement: budget-only by A3, scalar-κ₄ by A5.
>
> **All-layers fingerprint:** AUGMENT `6.906e-09` all-layers / `1.912e-08`
> final-layer, against the leaders' published `7–11e-9` / `1.9–2.8e-8`. It matches
> on **both** simultaneously. Caveat: the P01-24 path reads `+6.0%` hot vs
> graded-100 on an identical file, so this is good to a few percent.
>
> ### STEP 2 (the port) — FEASIBILITY PROVEN, DERIVATION OWED
>
> The blocker was the dense `n³` `(2,1,1)` object at **exactly** hard rule 1's
> `4 GiB` single-array cap. Both of its sites are now settled:
>
> 1. **`K_ds[(2,1,1)]` enters the `r=1` core ONLY through its axis-0 sum** —
>    `core = c(n)·sym(Σ_i slice[i,j,k])`, `c(1024) = 3/514`, verified against
>    upstream's own `proj_geq_r` at `2.2e-16`. `L²` of a `(2,1,1)` slice is
>    identically zero (no valid cap-graph), so only `L¹` survives.
> 2. **`pK[(2,1,1)]` carries no order-3 information.** Of 17 surviving vector
>    partitions, 15 have every block spanning ≤2 axes; the two that span three are
>    *degree-4* blocks, i.e. slices of `WK[4]`, which under AUGMENT is `r=1` with
>    an `(n,n)` core — `n³`-shaped views of `n²` data.
>
> **So the fused path fits `n²`; the largest array is `4 MB` at f32 against a
> `4 GiB` cap, by construction.** Owed next: the fused expression per partition;
> the two diagslices' closed form **read off upstream numerically, not derived**;
> verification against the dense path at `n = 32/64`; `p2_flop_anchor` pricing to
> replace the `~0.31 B` sanity estimate.
>
> ### COST HALF — BOTH ITEMS PRICED, BOTH DEFLATED
>
> | lever | claimed | **measured** |
> |---|---|---|
> | dead/cold-column pruning | `~0.80x` | **`0.961x`** free (`α < −4`). `0.845x` needs a compensation recovering `139x` our MSE |
> | hybrid Strassen | L=4 FLOPs at `≤75 s` | **the point does not exist** — Strassen's saving comes *from* depth |
>
> **But hybrid found a live nominee trade:** depth 2 (leaf 256) is `0.60420 B` at a
> projected `33.0 s` worst wall — **`2.28x` under the measured `1.60x` contention**,
> against nominee A's `1.21x` — for `12.4%` more score. **Choose the Strassen depth
> AFTER the port is priced**, on the combined cost. The pricing instrument
> reproduces the grader on both anchors (`+0.003%`), and its wall model validated
> **out of sample** (`118.2 s` predicted vs `114.68 s` graded, `+3.1%`).
>
> ### DEFERRED, WITH REASONS
>
> - **D2** — both its gates are dead (the lane gate is conjunctive and needs
>   `D1 ≥ 3.5e-8`; the carrier gate was demoted to a diagnostic). It is a Step 3/4
>   input, ~12 h of GPU, and Step 2 does not wait on it. Run when the machine idles.
>   Its **refusal gate bit on first use**: the default `2^22` would have let the
>   oracle's own noise push it toward its own kill; it must run at `2^24`.
> - **The corrector** (Step 4) — not started.
>
> ### NOMINEES
>
> **A = `330327` (L=3). B = `330212` (k3_full).** The Rules v14 §6 sentence on *how
> two designated submissions are scored* **does not exist in any capture we hold**;
> the live page is login-gated. Under the handoff's ambiguity branch, and on a
> maximin argument that holds either way, B is the safe one. **Samuel designates.**
>
> ### CORRECTIONS TO THE RECORD MADE THIS SLICE
>
> - **D3 is already banked** — power cumulants are upstream's `use_pK`, default
>   `True`. Contradicts the Sep 8 handoff's D3 row.
> - **D2 does not bracket SSC** — an `r=1` state holds `~0.2%` of κ₄'s Frobenius
>   mass. Contradicts the Sep 8 handoff.
> - **`expandable_segments` is inoperative on Windows** — it is NOT why AUGMENT
>   fits; capacity is.
> - **`215 → 17 GFLOP/s` does not reproduce.** `215.13` is exact; the L=3 row gives
>   `1181.9 GFLOP / 51.8 s = 22.82`, not 17. Trace or strike the 17.
> - **Two of my predictions missed** (A3 `f`: predicted `[0.5,0.8]`, got `16.72`;
>   A5: predicted `[2.2e-8,3.2e-8]`, got `4.34e-8`), both in the same direction, and
>   **two cost estimates were wrong by 4x and by everything**. Standing correction:
>   **on this chain, measure before estimating.**


> ## 2026-09-09 (later) — **THE ATTRIBUTION IS SETTLED: the whole `2.094x` is κ₄'s `(n,n)` STRUCTURE.** Both cheap ports are dead. D2 is running at `2^24`.
>
> `estimator.py` untouched at **v2.6.1**. **Nothing designated.** Nothing promoted
> — P01 is a re-read DESIGN block.
>
> ### FOUR ARMS, PAIRED IN ONE PROCESS, SAME 24 NETS
>
> | arm | budget | κ₄ carried as | mean xmse | vs SIMPLE |
> |---|---|---|---:|---:|
> | K3 `SIMPLE` | base (40 terms) | a scalar | `4.0047e-08` | `1.000x` |
> | **A5** | **raised (142)** | **a scalar** | **`4.3448e-08`** | **`1.085x` WORSE** |
> | K3 **`AUGMENT`** | raised (142) | an **`(n,n)` matrix** | **`1.9123e-08`** | **`0.478x`** |
> | A3 | raised (142) | zero | `3.6895e-07` | `9.21x` |
>
>     SIMPLE - A5    -3.4014e-09  95% [-4.9367e-09, -1.8662e-09]   A5 better on 4/24
>     A5 - AUGMENT   +2.4325e-08  95% [+2.2968e-08, +2.5682e-08]
>     f5 = 1.1626 (structure's share; >1 because the budget alone is net-negative)
>
> **The raised diagram budget on its own is worth NOTHING — it is slightly
> negative.** Every bit of the gain is κ₄'s `(n,n)` structure. The two are
> complementary, not additive.
>
> ### BOTH CHEAP PORTS ARE CLOSED BY MEASUREMENT
>
> | escape route | closed by | how badly |
> |---|---|---|
> | budget-only, no κ₄, never form the `4 GiB` array | **A3** | `9.2x` worse than SIMPLE |
> | carry κ₄ as a scalar, skip the `(n,n)` core | **A5** | `1.085x` worse than SIMPLE |
>
> **So the port MUST carry κ₄ as an `(n,n)` core, and Step 2's first task is
> unchanged: the fused `(2,1,1)` contraction** — now for a fully attributed
> reason. (A3's own verdict string claimed this for a reason its measurement did
> not support; it is A5 that establishes it.)
>
> One cell remains **untested**: *base budget + `(n,n)` κ₄*. Upstream exposes no
> flag for it and the injection seam cannot reach it. If it worked it would be a
> cheaper port.
>
> ### THE ALL-LAYERS FINGERPRINT — the leaders' signature on BOTH numbers
>
> | arm | all-layers | final-layer |
> |---|---:|---:|
> | `SIMPLE` | `1.5936e-08` | `4.0047e-08` |
> | A5 | `1.6078e-08` | `4.3448e-08` |
> | **`AUGMENT`** | **`6.9060e-09`** | **`1.9123e-08`** |
> | **leaders, published** | **`7–11e-9`** | **`1.9–2.8e-8`** |
>
> Matching the *pair* is a far harder test than the final-layer number alone.
> Caveats: ours sits a shade below the band, the P01-24 path reads `+6.0%` hot
> against graded-100 for an identical file, and the band is scraped telemetry — so
> good to a few percent, and the sign must not be read as "we are ahead".
>
> ### MY PREDICTIONS MISSED TWICE, IN THE SAME DIRECTION
>
>     A3   predicted f in [0.5, 0.8]     measured 16.72
>     A5   predicted [2.2e-8, 3.2e-8]    measured 4.3448e-08
>
> Both are the same error: I expected the raised budget to carry a meaningful
> share, and it carries none. **My priors about which term matters in this closure
> should be given little weight**; cheap arms should be measured, not reasoned
> about.
>
> ### THE REFUSAL GATE BIT ON ITS FIRST USE
>
>     2^20 ratio 0.424 REFUSE | 2^22 ratio 0.239 REFUSE | 2^24 ratio 0.111 PASS
>
> **`2^22` was the default a 24-net D2 would have used**, and its two-stream
> spread is `24%` of the gap the slice turns on. The oracle's noise biases
> `xmse_c` **upward — toward the pre-registered kill** — so D2 could have closed
> the κ₄ lane on its own noise. The first draft computed that noise and never
> gated on it; the Skeptic caught it, the ERRATUM pre-registered the refusal
> before any data, and it fired immediately. D2 therefore runs at **`2^24`**
> (`~2.5 h`, job `438`).
>
> ### CONTROLS NOW CLOSED
>
> - **Reversed arm order: BIT-IDENTICAL** (`0.0`) across all four arms.
> - **Zero** `"Snapping negative variance"` warnings at `1024x16`.
> - `SIMPLE` and `AUGMENT` reproduced D1's means **to the digit a third time**, in
>   a third process, across two reboots.


> ## 2026-09-09 — **D1 IS IN: `AUGMENT` TAKES THE CHAIN `4.0047e-08 → 1.9123e-08`, `2.094x`, ON 24/24 NETWORKS.** The floor ruling is confirmed.
>
> `estimator.py` untouched at **v2.6.1**. **Nothing is designated.** Nothing is
> promoted — P01 is a re-read DESIGN block.
>
> | arm | mean xmse | se | 95% CI |
> |---|---:|---:|---|
> | K3 `SIMPLE` (control) | `4.004685e-08` | `1.017e-09` | `[3.7943e-08, 4.2151e-08]` |
> | K3 **`AUGMENT`** | **`1.912323e-08`** | `4.320e-10` | **`[1.8229e-08, 2.0017e-08]`** |
>
> `paired SIMPLE − AUGMENT = +2.0924e-08`, 95% `[+1.9441e-08, +2.2406e-08]`.
> `decide()` **PASS with an empty `reasons` list**, gain `2.0941x` against the
> pre-registered gate `1.6019x`. `gt_noise/half 1.1552e-09`, `16.6x` below.
> Artifact `experiments/2026-09-08_slice3/d1_augment.json`, job `411`.
>
> **Why it is trustworthy, not merely good:** the control reproduced
> **BIT-EXACTLY** (`max |drift vs pa_k3.json| = 0.0` over all 24 nets, mean
> `4.004685e-08` digit for digit, across a reboot, a driver, `no_grad` and
> `expandable_segments`); the gain is **broad** (arithmetic `2.0941x` vs geometric
> `2.0905x`, agreeing to `0.2%`, `24/24` nets, no biggest-mover warning); and
> `kind` was the **only** difference between the arms.
>
> **`1.912e-8` lands inside the leaders' `1.87–2.13e-8` cluster.** That is the
> "same algorithm, different constants" fingerprint, reached independently. It is
> also a **LOWER bound** — factored AUGMENT drops same-order diagrams
> (`factor_k3.py:291-299`), and unfactored AUGMENT is `O(n⁴)`, unmeasurable here.
>
> **NOT a score.** AUGMENT's FLOP cost is **unpriced**. The `30x` torch wall ratio
> is the dense `n³` intermediate, *not* a cost signal. No projected adjusted score
> may be quoted until `p2_flop_anchor` prices a real fnp spelling.
>
> ### STEP 2 IS GATED OPEN — BUT ITS PRECONDITION IS MEMORY, NOT ACCURACY
>
> The degree-4 `(2,1,1)` slice is a dense `n³` object — **exactly `4.000 GiB` at
> width 1024 in f32, i.e. exactly hard rule 1's single-array cap**, and `8 GiB` in
> f64. An AUGMENT port that forms it cannot ship at any accuracy. So Step 2's
> first task is **not** the port:
>
> > Derive the fused `(2,1,1)` contraction (11 vector partitions; **9 survive**,
> > one is dropped by `factored_keeps_term`, one is the identity term; **none is
> > intrinsically order-3**), verify it against upstream's dense
> > `DS_pK_to_K` + `DS_harmonic_proj` at `n = 32/64`, and price it with
> > `p2_flop_anchor`. Rough estimate `~0.31 B` extra — a sanity check, **not** a
> > price.
>
> Same method already proven in this slice: the `core ↔ T` identity was derived,
> verified against upstream at small `n` to `2e-12`, then used at `n = 1024`.
>
> **It is also the practical unblocker.** Fused AUGMENT runs in `~2 GiB` instead of
> `22.97 GiB`, which is the difference between waiting on a card the desktop is
> using and running any time.
>
> ### IN FLIGHT  *(SUPERSEDED 2026-09-09 later: A3/A5 are in, D2 is running at 2^24)*
>
> | job | what | state |
> |---|---|---|
> | `413` | D2 two-stream REFUSAL, `2^20/22/24` | queued; **the 24-net D2 is refused unless it passes** |
> | — | 24-net D2 | after `413`; `--d1-xmse 1.912323e-08` is now available and REQUIRED |
> | — | Skeptic on D1 | running, before any port is started (Rule 2) |
>
> **`workerq` 1.3.0 changed under us** — see `docs/GOTCHAS.md`, four traps now.
> The two new ones: a **relative interpreter path** fails (the snapshot has no
> `python.exe`), and a **relative `--out` writes into the snapshot**, not the
> repository. Submit with an absolute interpreter and an absolute `--out`.
> AUGMENT needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — the earlier
> OOMs were fragmentation, not capacity.


> ## 2026-09-08 (later) — SLICE 3 STEP 1 IS PRE-REGISTERED, BUILT, SKEPTIC-CORRECTED AND QUEUED. NO DATUM EXISTS YET.
>
> **Read this before touching D1/D2.** Contract `432f7fd`, instruments `eff8367`
> / `366fed1`, **ERRATUM `7245bea`**, fixes `00f0d57`. `estimator.py` is still
> untouched at **v2.6.1**. **Nothing is designated.**
>
> ### THREE FINDINGS FROM SOURCE READING, BEFORE ANY GPU TIME
>
> 1. **The ruling's mechanism is confirmed exactly.** `get_d_max(3, ·) = 4` for
>    **both** kinds — AUGMENT adds no cumulant degree, it enriches degree 4 from
>    a **scalar** (`r_x = 2`, SIMPLE) to an **`(n,n)` matrix** (`r_x = 1`).
>    `kprop_harmonic.py:194-245`. `factor=True, kind=AUGMENT, k_max=3` is an
>    implemented upstream path (`:366-376`).
> 2. **D3 IS ALREADY BANKED — there is nothing to run.** Power cumulants are
>    upstream's `use_pK`, which **defaults to `True`**; `use_pK=False` is their
>    documented *ablation*. Both arms already carry them, and our fnp chain
>    implements "all 25 kept pK terms". **This contradicts the Sep 8 response's
>    D3 row** (predicted `1.1–1.5x`): the gain is already inside the `4.0e-8`
>    baseline. Checked, not asserted (`_assert_pk_default`).
> 3. **D1's number will be a LOWER BOUND.** `factor_k3.py:291-299` — factored
>    AUGMENT *drops* the non-hypertree diagrams, which carry the **same**
>    `Θ(n^-3)` squared size as those it keeps. Unfactored AUGMENT is `O(n^4)` and
>    unportable at `n = 1024`, so the factored variant is the only one worth
>    measuring — but it bounds AUGMENT-class information from **below**.
>
> ### THE SKEPTIC RETURNED **PROVISIONAL** AND IT PAID FOR ITSELF AGAIN
>
> Run against the *construction*, before the 24-net jobs. Every defect is fixed
> in `00f0d57`; the full list is the ERRATUM. The three that mattered:
>
> - **Rung (b) was mislabelled, and the thing it was named for is NOT MEASURABLE
>   HERE.** `diag(T)_i = Σ_k κ₄(i,i,k,k)` is a per-neuron **trace**, not
>   `κ₄(i,i,i,i)`. Measured against a dense κ₄: the per-neuron `κ₄(i,i,i,i)`
>   actually represented after rung (b) is `0.243/0.132/0.060` of truth at
>   `n = 16/32/64` — a `~1/n` law reaching **≈0.4% at `n = 1024`**.
>   **The strategist's D2(b) / §3b "per-neuron diagonal κ₄ in flight" needs Step
>   3, not Step 1** — an `r=1` HTensor has no per-neuron κ₄ degree of freedom.
> - **`D2(c)/K3` is confounded with D1** (`SIMPLE→AUGMENT` change *and* the truth
>   injection). In a 64×5 reproduction the representation change alone was
>   `2.24x` while the truth injection added `2.5%`. **`(a)→(c)` is the only
>   contrast isolating truth** and is now the headline.
> - **The CARRIER gate had the same hole as the LANE gate's caveat.**
>   `D2(c)/D2(a)` is measured *inside* the trace-matrix subspace and was being
>   used to decide a carrier for its **orthogonal complement**. Measured,
>   `‖κ₄ − P_{≥1}κ₄‖/‖κ₄‖` → **≈0.998 at `n = 1024`**: D2 bounds a subspace
>   holding **~0.2% of κ₄'s Frobenius mass**. **Demoted to a diagnostic; it may
>   not open or close the Step-3 lane in either direction.**
>
> Also fixed: the lane gate was implemented as half of its conjunctive self; all
> gates now run through `competition_score.decide()` on paired differences with
> intervals; the oracle's own noise scales as `~1.7·n/√N`, **not** `1/√N`
> (`--precision-check` was re-posed and job `397` cancelled before it printed a
> conclusion against a bar ~1000× too tight); JSON keys `c_over_a`/`c_over_b`
> held their reciprocals.
>
> **A new pre-registered REFUSAL** (before data): two independent oracle streams
> on one net must agree to **≤ 0.2 of the `(a)→(c)` gap** — the *smaller* gap —
> or the 24-net D2 run must not be submitted. The oracle's noise biases `xmse_c`
> **upward**, i.e. toward the pre-registered kill, and it was recorded but never
> gated.
>
> ### QUEUED — nothing has run  *(SUPERSEDED 2026-09-09: D1 has run; see the block above)*
>
> | job | what | gates |
> |---|---|---|
> | `399` | D1 feasibility probe, 1 net, both arms, post-`no_grad` peak VRAM | fixes the VRAM declaration for everything below |
> | `402` | D2 two-stream REFUSAL at `2^20/22/24` | **the 24-net D2 is refused unless this passes** |
> | — | 24-net D1 | submit after `399`; needs the corrected declaration |
> | — | 24-net D2 (`--d1-xmse` REQUIRED) | submit after `402` passes **and** D1 lands |
>
> `392`/`398` (24-net, declared `26 GiB`) were **cancelled**: upstream builds its
> MLP from `nn.Linear`, so the chain had been retaining a sixteen-layer autograd
> graph it never uses. `no_grad` (`cbc1a9c`) is numerics-neutral; `399` measures
> the real peak rather than re-guessing it. **Nothing was narrowed in width** — a
> width-512 proxy would answer a different question.
>
> ### NOMINEES — the sentence the handoff asks for DOES NOT EXIST in our captures
>
> Rules v14 establishes that up to **two** may be designated, and §6's fallback
> ("two best-scoring valid submissions from that Phase on that Phase's public
> leaderboard"). **Neither capture says how two designated submissions are
> SCORED** — both capture-scope fields say they were taken to settle *how many*.
> The live page is login-gated. So the handoff's ambiguity branch applies:
>
> **A = `330327` (L=3). B = `330212` (k3_full).**
>
> That is also **maximin**: `330290` (L=4) sits at `0.65x` of the wall cap under
> the measured `1.60x` contention, so it is a coin-flip on a fresh private suite
> and buys at most `1.106x` when it works; `330212` has `12.6x` wall headroom and
> cannot fail. **Samuel designates; nothing here does.** One logged-in read of §6
> settles it — look for whether the private re-run yields **one score per
> submission** (ranked by best) or **one per team** (averaged/combined).

> ## 2026-09-08 — NOMINEE A IS STRASSEN L=3 AT `2.031e-08`, FROM `8.501e-08` TWO DAYS AGO
>
> Three graded submissions in two days took the entry from **`#47`** to roughly
> **`#27`**. `estimator.py` is still untouched at **v2.6.1**. **Nothing is
> designated** — that is Samuel's alone.
>
> | submission | file | adjusted (AGGREGATE) | B | ops | wall worst | failed |
> |---|---|---:|---:|---:|---:|---:|
> | `330212` | `estimator_k3_full.py` | `2.9467e-08` | `0.77970` | 3,249 | `9.54 s` | 0/100 |
> | `330290` | Strassen **L=4** | **`1.835981e-08`** | `0.48578` | 15,393 | `114.68 s` | 0/100 |
> | **`330327`** | Strassen **L=3** | **`2.031e-08`** | `0.53745` | 12,357 | **`61.93 s`** | 0/100 |
>
> **A = `330327` (L=3).** Not the best score — the best *nominee*. `330290` scores
> `1.106x` better and its worst row cleared the `120 s` wall cap by `5.3 s`
> (`1.046x`, `1.23` sd). Against the **measured** `1.60x` contention factor it sits
> at `0.65x`, i.e. it would fail rows on a busy grading server, and a failed MLP is
> zeroed at multiplier 1.0 — a `~3e5` score penalty. L=3 sits at `1.94x` margin and
> `1.21x` under contention. **B = `330290`** as a public row; `330212` behind it.
>
> ### WHAT STRASSEN BOUGHT, AND WHAT IT COST
>
> Batched Strassen is **exact**: graded raw MSE `3.7794363e-08` against the
> control's `3.7793002e-08` — **`+0.0036%`** — and the whole score gain equals the
> multiplier ratio (`1.605120` vs `1.6050505`) **to `4e-5`**. That is what an exact
> transformation looks like.
>
> Its cost is **wall time**, which the cost model did not have: intensity collapsed
> `215 → 17 GFLOP/s` and kernel wall went `7.97 → 51.8 s` at L=3 *despite fewer
> FLOPs*. `official_config` now carries `GRADED_WALL_S_PER_MLP_OBSERVED`,
> `GRADED_WALL_S_WORST_OBSERVED` and `GRADED_WALL_CONTENTION_FACTOR = 1.60`, and
> `verify()` recomputes them from the raw reports (`93/93 → 96/96`).
>
> ### THE PER-OPERATION RESIDUAL LAW IS SETTLED — four graded points
>
>     submission   ops/MLP   residual median   per-op
>     329896         2,690        30.11 ms     11.190 us
>     330212         3,249        36.40 ms     11.204 us
>     330327        12,357       132.54 ms     10.726 us
>     330290        15,393       169.19 ms     10.991 us
>
> `4.3%` spread across a `5.7x` range in operation count, and `330290`'s op mix is
> concatenate/add glue rather than matmuls — so this is invariance in op TYPE as
> well as op COUNT. The Skeptic named this as the largest open measurement question
> on 2026-09-07; it is closed. **But the op gate is no longer the binding one for
> this architecture** — at 22,000 ops residual would be `~0.24 s` (fine) while wall
> would be `~150 s` (fail).
>
> ### COST ENGINEERING IS FINISHED AT `~0.54 B` NOMINEE-GRADE
>
> | lever | verdict |
> |---|---|
> | §2a batched Strassen | **SHIPPED at L=3.** L=4 is public-row only; L=5 excluded by the shape cap, memory AND wall |
> | §2c aliased self-contractions | **CLOSED — inapplicable.** The `0.5010x` einsum discount needs literally the same buffer; every matmul here contracts two different arrays. No symmetry discount either |
> | §2b pruning, §2d per-MLP spend | not started, and both are accuracy-affecting — the class Slice 1 already showed loses here |
>
> **The strategy canon's `≤0.35 B` target is unreachable by this route.** Wall
> binds. Further cost reduction must come from doing less work, not from computing
> the same work more cleverly.
>
> ### §3, THE ACCURACY LAYER — RE-SCOPED BY TWO MEASUREMENTS
>
> **§3a is falsified for the readout.** `WP22/O1`: a fitted scale removes `0.75%`
> of the error (32 per-layer parameters buy `1.0076x` against a predicted
> `1.2–1.5x`), and 31 extra degrees of freedom over one global scalar buy `0.15%`.
> **The residual is structure, not scale**, and not even layer-dependent scale.
>
> **§3d is a provisional NO-GO, on evidence the repository already had.** SSC's
> readout is a per-neuron scalar κ₄ (`docs/SSC_PUFFI.md`), and our chain already
> computes the Hermite factor it multiplies — so **§3b and §3d are one lever**. And
> `p2_cumulant_oracle.py` priced it on 2026-08-21 with oracle cumulants:
>
>     O1 + TRUE (mu, sigma)   94.61% of the error   <- propagated STATE drift
>     O2 + TRUE kappa_3        4.01%
>     O3 + TRUE kappa_4        0.46%                 <- the ceiling on SSC's readout
>
> A **perfect** per-neuron κ₄, handed free to the readout, is worth `0.46%`. Puffi's
> `−77.2%` is a whole-stack result at `256×32` against a baseline `112x` worse than
> our chain, and their own paper says SSC is still `7.94x` worse than sampling.
>
> **The lane survives only through PROPAGATION** — κ₄ carried in flight, changing
> the state the next Wick update consumes — which is where the `94.61%` lives and
> which nothing has measured.
>
> ### NEXT — one experiment, then the strategist
>
> 1. **Add an `O4` arm to `local_sim/p2_cumulant_oracle.py`: carry TRUE κ₄ IN
>    FLIGHT and measure against `O2`.** One GPU job, no new seed block, and it
>    bounds SSC and every other κ₄ closure on our shape. **If `O4` does not beat
>    `O2` by `≥1.5x`, §3d is dead and §3 collapses to §3e** (the corrector on chain
>    features — the only remaining arm that *adds information* rather than
>    re-expanding the same state).
> 2. **Then the Strategist Packet**, whose question has changed: cost engineering
>    is finished at `~0.54 B`; the `~1.9e-8` raw floor the whole top-12 shares is
>    all that separates us from `#1`; §3a is falsified and §3d is priced at
>    `0.46%` through its readout. `docs/STRATEGIST_PACKET_2026-09-08.md`.
> 3. Hygiene owed on the next candidate: `_regen` is still not a cheaper fallback;
>    stale docstrings; and the Strassen code is labelled "Strassen–Winograd" but
>    implements **classical Strassen** (18 additions, not 15) — the Winograd form
>    would lower the min-block threshold.


> ## 2026-09-07 — SUBMISSION `330212` IS GRADED AT `2.9467e-08`. NOMINEE A EXISTS.
>
> **`estimator_k3_full.py` (sha256 `220d2314…`) graded `2.9467e-08` adjusted on
> the official 100, `0/100` failed, residual `36.4 ms` against the `0.4 s` cliff.**
> That is **`2.885x` better than the previous graded nominee** `329896`
> (`8.501e-08`) and better than its own prediction (`~3.1e-08`). `estimator.py`
> is still untouched at **v2.6.1**. **Nothing is designated** — designation is
> Samuel's alone. Slot 1 of 10 used for 2026-09-07 UTC.
>
> | | graded `330212` |
> |---|---:|
> | **adjusted final-layer score** | **`2.9467e-08`** |
> | raw final-layer MSE | `3.7793e-08` |
> | **all-layers MSE** | **`1.5590e-08`** (better than final-layer — the leaders' signature) |
> | counted FLOPs / MLP | `1,714,587,938,406`, **bit-identical on all 100 rows** |
> | multiplier | `0.77970432` — off the `0.1` floor |
> | **n_failed_mlps** | **`0`** — budget 0, time 0, residual 0, combined 0, error 0 |
> | residual s/MLP (min/med/max) | `0.0355 / 0.0364 / 0.0393` → **`10.18x` worst-row headroom** |
> | predict wall s (min/med/max) | `8.95 / 9.22 / 9.54` against `120 s` → `12.6x` |
> | kernel / overhead / comms s | `7.97 / 1.23 / 0.130` (median) |
> | arithmetic intensity | `215.1 GFLOP/s` — **the highest on the board** |
>
> Projected rank against the 2026-09-05 snapshot: **~`#30`–`#34`, from `#47`**
> (between ArtemVoronov `#26` at `1.83e-08` and trim_qewas `#38` at `3.74e-08`).
> A fresh board snapshot is owed before that is quoted as fact.
>
> Report: `experiments/run_reports/k3_full_220d2314_330212_graded_report.json`.
> Ledger row: `experiments/submissions.jsonl`. Narrative: `experiments/LOG.md`,
> 2026-09-07.
>
> ### THE BLOCKER FROM 2026-09-05 IS CLOSED, AND IT WAS THE INSTRUMENT
>
> The local mini-100 failed `100/100` on `residual_wall_time_exhausted` at
> `0.4363/0.4486/0.4667 s` (`mini_k3_full_nocap.json`, the SUBMITTED bytes).
> The graded truth on those bytes is `0.0355/0.0364/0.0393 s` — **the local
> reading over-reads by `12.3x`.**
> Samuel relaxed hard rule 6 for that one submission on the measured argument;
> the argument was right.
>
> **The grader's residual is a PER-OPERATION CONSTANT.** Two graded candidates,
> `10.2x` apart in FLOPs and `3.2x` apart in arithmetic intensity, both counted
> in the GRADER'S weight convention (f64 read-only + `materialise_predictions` —
> the only one whose FLOP total reproduces the graded total):
>
>     candidate        ops    FLOPs    kernel   GFLOP/s   residual   PER-OP
>     canary4 329896  2,690   168.7 G   2.54 s     66.4    30.11 ms  11.192 us
>     k3_full 330212  3,249  1714.6 G   7.97 s    215.1    36.40 ms  11.205 us
>
> **`0.10%` apart.** Array traffic does not enter the grader's residual; it is
> billed to `flopscope_server_kernel_wall_s`, which moved only `3.1x` on that
> `10.2x` FLOP change and so does not track FLOPs either. **Operations predict
> residual better than any array-traffic proxy.** Corroborated across a `5.4x`
> op lever spanning the gate (`local_sim/p2_residual_law.py`, `R^2 = 0.939`).
> **`docs/VALIDATION.md` §2b-iii and `docs/GOTCHAS.md` carry this.**
>
> ### THE OPERATIONS GATE IS `22,000`
>
> Sized on the **maximum observed** per-op figure, which is the stated policy on
> that constant — and the maximum is not ours. Our own worst row is `12.096 us`
> (a single MLP), but two of the twenty other graded rows visible in
> `experiments/board_snapshots/` exceed it: neilarmstrong4 `13.31`, Camaro
> `12.17`.
>
> | | ops / MLP |
> |---|---:|
> | the `0.400 s` cliff at `13.4 us/op` | `29,850` |
> | **working gate, 25% margin (`0.295 s`, `1.36x`)** | **`22,000`** |
> | current chain | `3,249` — **`6.8x` of room** |
>
> An intermediate `24,000` was carried for part of the day and was **neither
> derived from the data nor conservative** — it came from the Slice-2 handoff
> before `330212` graded, and the entry that derived it was a retro-derivation.
> An independent Skeptic caught both. `official_config.verify()` now RECOMPUTES
> the law from the raw graded reports (`93/93`), so it is no longer prose.
>
> ### TWO INSTRUMENT FACTS FOUND BY THIS SUBMISSION
>
> 1. **The cast model of the server FLOP offset HELD OUT OF SAMPLE.** Graded
>    FLOPs equalled the local certify count **exactly**, delta `0` — which
>    `local_sim/official_config.py` predicted on 2026-09-05, before submission,
>    because the file opens with a genuine `f64 → f32` cast. (The `+2**24`
>    family is the one whose opening cast is a same-dtype f64 no-op: v2.6.1,
>    v2.7.) `AGENTS.md`'s flat `+2**24` line is the stale coarse form.
> 2. **An op count is meaningless without its weight convention, and there are
>    three.** f32 census `3,232` → `whest run` mini `3,233` → grader twin
>    `3,249`, and only the last reproduces the graded FLOP total. The ladder is
>    `+32,768` (`materialise_predictions`) and `+33,554,432` (the opening
>    `f64→f32` weight cast, which `whest run` never pays because the local
>    dataset hands float32 weights). **An earlier reading of this as an open
>    `2**25` mystery, with a matrix-vector mechanism, is RETRACTED** — the cause
>    was already asserted by `certify_candidate.py` and the mechanism was a
>    coincidence. Quote the grader-twin count, always.
>
> ### NOMINEE DISCIPLINE (handoff §4)
>
> **A = `estimator_k3_full.py`** — `2.9467e-08`, **1** clean graded run.
> **B = canary4 `329896`** — `8.501e-08`, `13.05x` residual headroom, retained
> until A has **3** clean graded runs.
>
> ### NEXT — Slice 2 (Sep 7–19)
>
> 1. **Cost engineering (handoff §2). Batched Strassen is PRICED and the
>    handoff's framing of it is corrected: the binding constraint is MEMORY, not
>    operations.** A fully batched Strassen level costs a flat 23 operations per
>    invocation, so even L=5 stays inside the gate — but the operand stacks grow
>    as `(7/4)^L` and a fully batched L=3 on the widest shape class raises
>    `MemoryError` against the `8,192 MB` limit. Expanding the top level into 7
>    sequential branches cuts peak memory ~6x for a few hundred operations. The
>    measured frontier (`local_sim/p2_strassen_plan.py`,
>    `experiments/2026-09-05_slice1/strassen_frontier.json`):
>
>    | L | chain B | ops (gate 22,000) | peak MB (limit 8,192) | adjusted |
>    |---:|---:|---:|---:|---:|
>    | 0 | `0.77969` | `3,232` | `2,828` | `2.9467e-08` |
>    | 2 | `0.60250` | `4,612` | `5,516` | `2.2770e-08` |
>    | 3 | `0.53528` | `13,702` | `3,750` | `2.0230e-08` |
>    | **4** | **`0.48281`** | **`18,532`** | **`4,534`** | **`1.8247e-08`** |
>
>    **L=4 hits the handoff's score target.** NOT YET BUILT — this prices a
>    candidate that does not exist, and the adjusted column assumes Strassen is
>    exact.
>
>    **The gating risk is measured and the `1e-6` parity bar is the WRONG
>    INSTRUMENT.** At the live shapes every arm, L=1 included, is over it — but
>    the plain float32 matmul is itself `4–8e-7` off float64 there, so a `1e-6`
>    bar allows only `1.3–2.5x` the error float32 already commits. Quote EXCESS
>    over the float32 baseline instead: L=1 `1.4–2.4x`, L=2 `2.7–6.1x`, L=3
>    `3.9–12x`, L=4 `5.4–30x`. Error is driven by the CONTRACTION LENGTH — `30x`
>    at `k=14336` against `6.5x` at `k=1024`, same array sizes — so a per-group
>    depth is an unpriced design option.
>
>    **Break-even is generous**: at L=4 the multiplier falls `0.619x`, so raw MSE
>    may rise to `6.10e-08` from `3.78e-08` before the trade stops paying. Whether
>    it does is a question for the **24-net paired raw MSE**, not for a per-matmul
>    relative error — and that is a STRATEGIC slice needing a pre-registered
>    contract and a seed block (`P06[..064..127]` unspent). Slice 1 closed two
>    accuracy-for-budget lanes, so the sign is genuinely open.
>
>    Then dead-column pruning, aliased self-contractions, per-MLP variable spend.
> 2. **The accuracy layer (handoff §3) is the critical path, and it is now
>    RE-SCOPED by measurement.** `WP22/O1` asked the cheap half of §3a first:
>    fit scalars on the RETURNED array, on P02, quoted on 24 disjoint high-res
>    nets.
>
>    | arm | params | final-layer MSE | gain |
>    |---|---:|---:|---:|
>    | none | 0 | `4.065813e-08` | `1.0000x` |
>    | one global scalar | 1 | `4.040971e-08` | `1.0061x` |
>    | scalar per layer | 16 | `4.036979e-08` | `1.0071x` |
>    | scalar + offset per layer | 32 | `4.035167e-08` | **`1.0076x`** |
>
>    **A fitted scale removes `0.75%` of the error, against §3a's predicted
>    `20–50%` — and 31 extra degrees of freedom buy another `0.15%`.** The
>    residual is **structure, not scale**, and not even layer-dependent scale.
>    §3a's prediction is FALSIFIED for the readout version. (The INTERNAL version
>    propagates and is not strictly this experiment; §3a's κ₃-slice half was not
>    measured at all. Both stay open — the evidence tilts hard, it does not
>    exclude.)
>
>    **So build the SHAPE corrections first.** Recommended amendment: promote
>    **§3d, the Puffi SSC κ₄-response closure** (`−77%` at 256×32; go/no-go
>    `≥1.5x` by Sep 19) ahead of §3a and §3b, and strike §3a's `1.2–1.5x` from
>    the Sep-19 arithmetic until the internal version is measured. Then §3b
>    (Edgeworth-4 with per-neuron diagonal κ₄), then §3e.
>
> 2b. **Lever §2c is CLOSED as inapplicable.** The aliased-operand discount is
>    real on 0.12.1 (`einsum('ij,kj->ik', X, X)` bills `0.5010x`) but needs the
>    einsum spelling AND literally the same buffer; `X @ X.T` gets nothing, and
>    every matmul in this chain contracts two genuinely different arrays. No
>    symmetry discount either — results are not auto-tagged and `as_symmetric` is
>    unimplemented in 0.12.1. One free micro-win banked for the next edit:
>    `einsum('ji,jk,kl->il', W, cov, W)` does `W^T cov W` in ONE operation
>    instead of two at identical FLOPs (`−16` ops/MLP).
> 3. Hygiene owed on the next candidate, not on the shipped one: `_regen` is not
>    a cheaper fallback (it is `gens=1` again and would blow the budget at
>    non-live shapes); stale docstrings at lines ~52–68, 79–82, 353–358, 980–987.
>
> ### WHAT THE 2026-09-07 SESSION DID NOT DO — read this before assuming
>
> The Slice-2 handoff covers Sep 7–19. One session closed §1 and §2a's pricing.
> **Everything below is untouched**, and none of it is blocked:
>
> | handoff | state |
> |---|---|
> | §1 submit | **DONE** — `330212` graded `2.9467e-08`, telemetry recorded |
> | §2a batched Strassen | **PRICED, NOT BUILT.** The frontier and its gating risk are measured; the next act is the pre-registered 24-net accuracy contract above, then the implementation |
> | §2b dead/cold-column pruning | **not started** — still unpriced |
> | §2c aliased-operand self-contractions | **not started** |
> | §2d per-MLP variable spend | **not started** |
> | §3a–3e the accuracy layer | **NOT STARTED, and it is the critical path.** Part C of Slice 1 was never run either, so this has now slipped twice. Nothing about it is blocked: it is torch work for the GPU queue and does not wait on any cost lever |
> | §4 nominee discipline | A = `k3_full` (1 clean graded run), B = canary4 until A has 3 |
> | §5 hand-back | this section |
>
> **If you can only do one thing next, do §3d (the Puffi SSC κ₄-response
> closure).** It is the most evidence-backed route to the leaders' 2× (Puffi
> measured −77% at 256×32), it is independent of every cost lever, and its
> go/no-go is due Sep 19. The cost work has already returned `2.9467e-08 →
> ~1.82e-08` on paper; the raw-error floor of `~1.9e-8` that the whole top-12
> shares is what actually separates us from #1, and no amount of Strassen touches
> it.


> ## 2026-09-05 (evening) — SLICE 1 PARTS A AND B: THE EXACT K3 CHAIN IS BUILT
>
> **`estimator_k3_full.py` scores `3.1224e-08` adjusted on 24 held-out nets —
> `2.72x` better than the graded nominee `329896` (`8.501e-08`) — at `0.77970 B`
> with `22.03%` budget headroom.** `estimator.py` is untouched at **v2.6.1**.
> **Nothing designated. Nothing submitted.** 3 of 10 slots remain for
> 2026-09-05 UTC (`experiments/submissions.jsonl`). No seed block drawn;
> `P06[..064..127]` unspent.
>
> | | |
> |---|---:|
> | raw final-layer MSE (24 P01 nets, `2^26` labels/half) | **`4.0045e-08`** |
> | GT noise per replica | `1.155e-09` |
> | counted FLOPs / MLP | `1,714,587,938,406` = **`0.77970 B`**, constant |
> | multiplier | `0.77970` — **off the `0.1` floor** |
> | **adjusted score** | **`3.1224e-08`** |
> | operations / MLP | `3,249` (gate `30,000`) → predicted graded residual `37.7 ms`, **`10.61x`** under the cliff |
> | Linux 8-GiB gate | 13/13, VmPeak `2,827.5 MB` (`2.90x`) |
> | `whest validate` | PASS, 48 ms |
>
> **AND ON THE ORGANISERS' OWN PUBLIC SPLIT** (`whest run --split mini --runner
> subprocess`, 100 rows, local residual cap lifted — see the blocker below):
>
> | | |
> |---|---:|
> | **adjusted final-layer score** | **`3.0603e-08`** |
> | raw final-layer MSE | `3.9251e-08` |
> | **all-layers MSE** | **`1.5937e-08`** — BETTER than final-layer, the leaders' depth-propagating signature |
> | **n_failed_mlps** | **`0`** — budget 0, time 0, residual 0, error 0 |
> | counted FLOPs | `1,714,554,383,974`, bit-identical on all 100 rows |
> | wall s (min/med/max) | `3.81 / 3.87 / 4.06` against a `120 s` cap |
>
> Projected rank against the 2026-09-05 snapshot: **~`#33`–`#38`, from `#47`.`
>
> Validated three ways: **collapse equivalence `1.93e-16`** (the lineage path,
> projected back to `(M, I, I)` each layer, reproduces the regenerating flow),
> **upstream parity `8.487e-10` against a matched control at `8.737e-10`**, and
> real labels. Full narrative: `experiments/LOG.md`, 2026-09-05 Slice 1.
>
> ### THE OPEN BLOCKER — and it is a MEASUREMENT problem, now measured
>
> **The official mini-100 fails 100/100 MLPs, every one of them on
> `residual_wall_time_exhausted` (local `0.453/0.469/0.509 s` against `0.400`).
> Nothing else fails** — no budget exhaustion, no time exhaustion, no error, and
> the FLOP count is bit-identical on all 100 rows. **Hard rule 6 requires a clean
> mini run before asking, and this is not one.**
>
> The local instrument is measuring a quantity the grader does not. Run the SAME
> candidate at three widths — identical operation count, only the arrays change:
>
>     width    ops    local residual   us/op   widest array
>       256  3,304          66.6 ms    20.2         3.8 MB
>       512  3,304         158.2 ms    47.9        15.0 MB
>      1024  3,304         480.4 ms   145.4        60.0 MB
>
> `7.2x` growth at a FIXED op count: the local reading absorbs in-process array
> traffic. On the grader that traffic lives on the server's seven cores and is
> reported separately — `329896`, 100 rows: kernel `2.39–2.83 s`, overhead
> `0.93–1.03 s`, comms `0.105–0.110 s`, **residual `0.0292–0.0307 s`, 0.8% of
> wall**. For this candidate to hit the cliff its graded per-op residual would
> have to be `121.0 us`, **`10.6x` canary4's measured `11.47 us`**.
> `docs/GOTCHAS.md` carries this. `certify` section 10 is deliberately left
> refusing on the local number. **Relaxing rule 6 is Samuel's call, not an
> agent's.**
>
> ### TWO LANES CLOSED BY MEASUREMENT TODAY
>
> **Generation truncation and last-layer capping are both LOSING TRADES**, on 24
> paired nets. Raw error roughly doubles per generation dropped while the budget
> saving is only linear; dropping the last layer's tower costs `5.6x` for `5.9%`
> of `B`. **The score-optimal operating point is the EXACT tower**, so the whole
> remaining gain must come from cost engineering that loses no accuracy.
>
> **Three such levers landed, taking the chain `0.99937 B -> 0.77970 B` at zero
> accuracy cost — all three out of one structural fact, `C_L = A_L diag(w2)`:**
>
> 1. `C_L` is never carried or advanced. The relation survives the lineage's
>    whole life because every later operation is a LEFT multiplication, and a
>    left multiplication commutes with a right diagonal. One matmul of nine.
> 2. The tower travels as wide `(n, k n)` blocks, so `contract_W` is THREE
>    matmuls whatever `k` is, and `sum_b X_b @ Y_b^T` is one contraction.
>    FLOP-neutral (`-0.0000038%`), operations `6,693 -> 3,305`.
> 3. In the `(2,1)` slice the two-leg block's `{0,1}|{2}` and `{1,2}|{0}`
>    permutations are IDENTICALLY the same product (`1.7e-16`), exactly as the
>    from_dstensor block's two already were. One matmul of eight; `902 -> 797`
>    matmuls per MLP.
>
> Unexpected, and for the write-up: **the regenerating collapse is a better
> approximation per FLOP than naive generation truncation** — `9.09e-08` adjusted
> at `0.077 B` beats every arm from `gens=1` through `gens=8`.
>
> ### NEXT
>
> 1. Finish certification; official mini (100) run; then **ask Samuel for the
>    canary**. Expected `~3.5e-08`, `~#38`.
> 2. Slice 2's ranked cost levers, all priced or measured on this chain, are in
>    `experiments/LOG.md`. Batched Strassen L=4 (`0.586x`) alone would take
>    `0.883 B → 0.517 B` and the score to `~2.1e-08`.
> 3. Part C's torch reads (O1, Edgeworth-4 + diagonal `kappa_4`, AUGMENT sign)
>    are NOT started. C4 was answered above instead, and in the shipped spelling.


> ## 2026-09-05 — STRATEGY CANON MOVED
>
> **`docs/PHASE2_STRATEGY_2026-09-05.md` (rev 10) is the strategy canon.** It
> supersedes rev 9 (`phase2-strategy-2026-09-01.md`), the 09-10 strategist
> response, and `docs/GAME_PLAN.md`'s amendment header wherever they differ.
> Read it before acting on anything below or in `GAME_PLAN.md`.
>
> The three reversals it makes, because they invert standing guidance here:
>
> 1. **W-EFF (the exact factorized-K3 port) is REOPENED and is nominee A's
>    chassis.** It was closed on 2026-08-28 for pricing at 1.07 B against a
>    `<= 0.35 B` screen. That screen asked "can it win at the 0.1 floor"; the
>    objective is SCORE, and the Sep-5 board telemetry says the whole top-8 is a
>    deterministic depth-propagating chain at 0.18-0.23 B with a raw floor of
>    ~1.9e-8. The 4-5x implementation-constant gap is the problem to solve, not
>    a lane to abandon.
> 2. **Strassen, aliased-operand discounts and symmetry tagging are REOPENED.**
>    They were never needed for a 0.077 B flow; they are the whole game for a
>    1.07 B chain.
> 3. **The 0.1 multiplier floor is a TERM IN THE OBJECTIVE, not a wall.** No
>    top-12 entry is at the floor. The score-optimal design today sits at
>    utilisation 0.15-0.25.
>
> Also corrected here: the "~Sep 30 ednacob harvest" watch in section 8 is
> withdrawn (Phase-1 private results are unpublished; see there).

> **2026-09-05 — THE CANARY IS GRADED. WORST-ROW RESIDUAL HEADROOM IS `13.047x`
> AGAINST A `>= 4x` BAR. RULING 2(a)'s FIRST BRANCH FIRES: *STOP, NO OP WORK,
> IT IS NOMINEE A AS-IS.***
>
> `estimator.py` untouched at **v2.6.1**. **Nothing designated.** `P06[..064..127]`
> unspent. No seed block drawn.
>
> ## THE NUMBER
>
> `329896` = `estimator_wp18_canary4.py`, AIcrowd, 100 rows, report
> `experiments/run_reports/canary4_329896_graded_report.json`:
>
> | | |
> |---|---:|
> | **worst-row residual headroom vs the `0.4 s` cliff** | **`13.047x`** |
> | mean headroom | `13.301x` |
> | failed MLPs | **`0 / 100`** |
> | raw final-layer MSE | **`8.501e-07`** |
> | **adjusted score** | **`8.501e-08`** |
> | mean compute utilization | `0.0767022` -> multiplier at the **`0.1` floor** |
>
> **`1.62x` better than v2.7**, which re-graded healthy at `1.3738e-07` the same
> day. Projected rank `22` of `67` against the 2026-08-28 snapshot, where v2.7
> sits at `41`.
>
> **THE OP-CUT WORK STREAM IS CANCELLED** by the pre-registered table, not
> deferred. § 4's `45.2%` was sized against a LOCAL residual and this morning's
> erratum withdrew it; the graded number is **`5.86x` better than the local
> `2.226x`**, consistent with the standing "~`7x` faster than this host" note.
>
> ## THE BUG THAT COST SEVEN SLOTS, AND IT WAS ONE KEYWORD
>
>     w = [x if x.dtype == dtype else x.astype(dtype, copy=False)   # <- copy=False
>          for x in mlp.weights]
>
> **Every local harness builds float32 weights, so `x.dtype == dtype`
> short-circuited and the cast never ran here. The grader's weights are not
> float32.** `copy=False` was redundant — the `if` already makes a no-op cast
> free — so it bought nothing and killed three submissions.
>
> Found by **two diagnostic submissions after three inferred fixes failed**:
> `329892` (flow replaced by zeros) PASSED, proving the fault is inside
> `predict`; `329894` returned its failing stage as a constant so the score
> decoded it — observed `final_layer_mse 2.689` against `2.6834` predicted for
> **stage 2**, neighbours at `0.82` and `6.54`.
>
> **Timing was read backwards for an hour:** the probe PASSED in `98.80 ms`, so
> the failures' `~95 ms` was never "died instantly" — that is simply what a cheap
> smoke costs. v2.7's `11,313 ms` is slow because v2.7 samples.
>
> ## OPEN, AND IT IS THE NOMINATION'S TO SETTLE
>
> 1. **`canary4` retains a `BaseException` catch that returns a constant** (the
>    stage channel). A safety net for a canary; **for a NOMINEE it converts a
>    crash into a scored-but-wrong answer instead of a visible failure.** Settle
>    it explicitly in the promotion contract — do not inherit it.
> 2. `13.047x` is ONE graded run. The standing soak language is the worst of
>    `3 x 100`; whether that still binds at `3.3x` above the bar is a judgement.
> 3. **Promotion of the flow into `estimator.py` is a STRATEGIC slice** and needs
>    its own contract. This session promoted nothing.
> 4. `preflight` step 1 asserts `0.16.x` / `0.12.x`; the grader is on
>    `0.16.1 / 0.12.1` and this repo on `0.16.0 / 0.12.0`. Unrelated to the bug,
>    still worth tightening.
>
> ## Today's submissions, all public and UNDESIGNATED
>
> | id | what | result |
> |---|---|---|
> | `329887` | canary v1 | FAILED |
> | `329889` | v2.7 unchanged, **control** | GRADED `1.3738e-07`, healthy |
> | `329890` | canary v2, best-effort `setup` | FAILED (fix was right, cause was not) |
> | `329891` | canary v3, width-2 `ZeroDivisionError` | FAILED (real defect, not the cause) |
> | `329892` | **bisection probe** | GRADED — fault is inside `predict` |
> | `329894` | **stage diagnostic** | GRADED — decoded stage 2 |
> | **`329896`** | **canary v4** | **GRADED `8.501e-08`, `0/100` failed, `13.047x`** |

> ## 0. RULE CHANGE, 2026-09-05 — an agent may now run `whest submit`
>
> **Samuel, verbatim:** *"change the rule for agents not to submit you should be
> allowed to submit ill approve"*.
>
> | | |
> |---|---|
> | **`whest submit`** | an agent **MAY** run it — **only** on Samuel's explicit approval of **that specific submission**. Approval is **per submission and never durable**; a previous yes carries to nothing |
> | **designation** | **UNCHANGED. No agent designates, ever.** That is the act that picks the entries for the private prize re-run, and it was not part of this change |
> | still binding | `whest validate` → clean mini run (`n_failed_mlps: 0`) → ask → submit. Record every agent-run submission in `experiments/LOG.md` with its artifact sha256 and the approving message |
>
> `AGENTS.md` hard rule 6 is the authority. `scripts/guard_paths.py` already
> implemented `whest submit` as **confirm**, not deny, so the enforcement layer
> needed no change — only the prose did.
>
> ## 1. THE CANARY WAS SUBMITTED AND IT **FAILED**. THE GRADING STACK HAS MOVED.
>
> **Submission `329887`, 2026-09-05 — `FAILED`.** Server smoke test,
> `error_code: ESTIMATOR_EXCEPTION`, `95 ms`, `0` of `100` MLPs evaluated. No
> score, no board movement, one of ten daily slots spent. Report:
> `experiments/run_reports/canary_failed_329887_graded_report.json`. Full entry in
> `experiments/LOG.md`.
>
> **THE GRADER IS ON `whestbench 0.16.1` / `flopscope 0.12.1`. THIS REPOSITORY IS
> ON `0.16.0` / `0.12.0`.** `preflight.ps1` step 1 asserts `0.16.x` / `0.12.x`, so
> it passes green on a point release and **could not have caught this**. Every
> local number here is measured one point release behind the grader.
>
> **The failure does NOT reproduce on the grader's own versions.** Built against
> `whestbench==0.16.1, flopscope==0.12.1, numpy==2.2.6`, the submitted bytes
> import, construct, `setup`, validate (`41 ms`), package to an identical manifest
> and inner sha256, pass `validate-package`, and **run at the live `1024 x 16` to
> finite output with identical `168,636,374,630` FLOPs / `2,673` operations.**
>
> **Leading hypothesis by ELIMINATION, not evidence: the participant shim and the
> flopscope CLIENT/SERVER split** (`flopscope_client` and `flopscope_server` are
> separate components in the report; every local run here uses the in-process
> backend). **That is the one environment this lab cannot replicate.**
>
> **THE OPEN QUESTION THAT MATTERS MOST, AND IT IS SAMUEL'S CALL:** v2.6.1 and
> v2.7 graded cleanly on the OLD stack. **Whether either still grades on `0.16.1`
> is unknown**, and both nominees are exposed, not just the canary. Finding out
> costs one submission slot.
>
> **The `>= 4x` graded per-row residual the canary exists to buy was NOT
> obtained** — the run never reached an MLP — so § 7's decision tree is still
> blocked on exactly the number it was blocked on before.
>
> ## 2. The ruling, and what it closes
>
> | # | question | ruling |
> |---|---|---|
> | **1** | second research contract on the carried-sketch lane? | **NO.** The `3.874x` exact-both-slices ceiling settles it. Carve-out, **deployment not research**: the `3.330x` cell as built could be shipped only under a promotion-class deployment contract, and **only if priced first at the multiplier floor with `+2**24`** |
> | **2** | `P06[..064..127]`? | **unspent** unless that deployment contract opens |
> | **3** | `P06`'s tier | **recorded as ACTUALLY USED. Never relabelled.** `seed_blocks.py:156` registers `P06` as **DESIGN**; `CONTRACT WP20` said DECIDE; the registry is the authority and the contract's label was the defect |
>
> **NEW STANDING RULE, effective now — the pre-run instrument check.** Before any
> job that scores a gate or spends a seed block: an **adversarial read that
> traces what the code does with each object the contract names**, confirming the
> cell answers the contract's question and not a narrower one. **It is distinct
> from the post-run Skeptic and does not replace it.**
>
> **AMENDED the same day, by the Skeptic that found what it would have missed.**
> The rule as first written would have caught the warm-up defect (it is in
> `slices_window`'s own docstring) and missed the `3 I` middle leg entirely —
> that lives in `factor_k3.py:400`, upstream, three files away, and is named by
> nothing in the contract. **Second clause: trace each named object to the
> UPSTREAM DEFINITION that produces it, not only to the place the contract names
> it.** It exists because
> `SKEPTIC WP20` found a cell where *nothing was broken* and every correctness
> check passed on the wrong construction. Its first binding application is the
> next contract; today's pricing was ROUTINE-by-the-table (no gate, no block) and
> took the post-run Skeptic instead.
>
> ## 3. THE `3.330x` CELL IS PRICED, AND IT IS DROPPED
>
> Instrument `local_sim/p2_wp20_price.py`; artifact
> `experiments/2026-09-04_wp20_price/wp20_price.json`. Executed arithmetic,
> flopscope 0.12, **f32**, real `BudgetContext`, live `1024 x 16`. Bar:
> `max(0.1, (168,636,407,398 + added + 2**24) / 2**41) == 0.1`, headroom
> **`51,249,140,941`** FLOPs — the module refuses to run if that disagrees with
> `CONTRACT WP20`'s own figure.
>
> **THESE ARE THE POST-AUDIT NUMBERS.** Rule 2 returned **`PROVISIONAL`** against
> pinned commit `af465fc`, the disposition survived over-determined, and **three
> supporting claims did not.** Every correction is applied and the artifact is
> regenerated. Full verdict and errata: `experiments/LOG.md`, *"SKEPTIC ON THE
> WP20 DEPLOYMENT PRICE"*.
>
> | spelling | added FLOPs | `x headroom` | |
> |---|---:|---:|---|
> | **`carry3_full`** — the object that measured `3.330x`, carried honestly | `608,251,203,072` | **`9.934`** | **OVER** |
> | `floor_full` — carry propagation only, `(n,n)` legs | `238,270,021,632` | `4.649` | OVER |
> | `floor_r` — the "unachievable lower bound" | `44,675,629,056` | `0.872` | **AT FLOOR** |
>
> **It does not fit. Per the ruling it is DROPPED and no further contract opens.**
> The verdict rests on `carry3_full` — **a factor of ten** — and the three
> structural savings available (identity legs, the window's warm-up,
> concatenation) were checked and none takes `9.9x` under `1x`.
>
> **The cause is structural.** `from_dstensor` returns `(diag(S3) + 3 S21^T, I,
> I)` — two identity legs, the regenerating flow's entire saving. A carried
> block's first `contract_W` is two products and a column scaling (its middle leg
> is still `3 I`, `factor_k3.py:400`); **every hop after that is three real
> matrices at the generic price.** Carrying is a different price regime, not a
> knob with a cost.
>
> **THREE CLAIMS ARE WITHDRAWN, and they are not in the table above:**
>
> 1. *"the unachievable lower bound is over too, so the gap is arithmetic, not
>    implementation"* — **FALSE.** `floor_r` over-charged twice (the window warms
>    up: `42` propagations, not `45`; and the age-1 middle leg is a scaling, not a
>    matmul). Corrected it is `0.872x` and **it fits**. It bounds nothing.
> 2. *"the op axis is NOT what killed it, `+885` ops, `1.331x`"* — **NOT
>    ESTABLISHED.** That added a LAB op increment to a SHIPPED op total. `regen`
>    bills `469` operations where the canary's own third-order increment is
>    `2,255` (`flow_price_wp17b.json`). Translated: about **`2.28x`** the canary,
>    against a cliff its worst graded row already sits `2.23x` under. **Op counts
>    are `3x` spelling-dependent inside identical arithmetic** and never transfer.
> 3. *"largest carried rank that stays at the floor: `64`"* — **WITHDRAWN.**
>    `carry3_r` prices a Tucker truncation, not a valid rank-`r` CP carry (a
>    `FactoredTensor`'s rank index is shared across all three legs). It
>    under-prices a fiction.
>
> **Control, and the audit strengthened it:** `regen` at f32 is **bit-exact half**
> WP13's committed f64 figure (`99,116,660,736` vs `198,233,321,472`), which also
> proves no silent NEP-50 promotion — and the reconciliation needed no `137.6e9/2`
> guess, because the measured f32 backbone `68,782,549,504` is on disk in
> `flow_price_wp17b.json`.
>
> ## 4. THE OP CENSUS — where the operations are. **The cut SIZE is NOT known.**
>
> Instrument `local_sim/p2_op_census.py` (new); artifacts
> `experiments/2026-09-04_wp20_price/op_census_canary.json`, `op_census_v261.json`.
>
> > **CORRECTED 2026-09-05.** This section first said *"a graded worst-row
> > headroom of `2.226x` already exists"* and sized a `45.2%` cut against it.
> > **`official_residual.json` is NOT graded** — it is `gpuq 182`, the official
> > *runner*, **on this host**. `AGENTS.md`: the grading server ran submitted K2
> > files **about `7x` faster than this host**. The Sep-3 ruling says *"size the
> > op-count work to the **graded** worst row, not to the local one"*, and the
> > canary's own bar is *local* `>= 1.5x` (met at `2.23x`) while the NOMINATION
> > bar is `>= 4x` **graded**. **WHICH BRANCH OF § 7 FIRES IS UNKNOWN, and
> > resolving it is the whole purpose of the canary.** Full erratum in
> > `experiments/LOG.md`.
>
> **What the census establishes, and it depends on no timing instrument:**
>
> | | |
> |---|---:|
> | canary operations, steady state | `2,673` (`2,674` in `certify_guarded.json`) |
> | **operations carrying the ENTIRE `168.6e9` FLOPs** | **`78` — `2.9%`** |
> | elementwise `O(n^2)`-and-smaller work | `2,595` — `97.1%` |
> | `estimator.py` (v2.6.1), metered directly | **`442`** operations |
>
> **The FLOP budget and the op budget are spent in almost disjoint places.** The
> three cheapest-looking groups — `103` scalar-by-scalar multiplies at **zero**
> FLOPs (`3.9%`), `265` vector-by-scalar (`9.9%`), `390` row/column broadcasts
> (`14.6%`) — are `28.4%` of the operation budget between them.
>
> **The fit in the LOG is a CALCULATOR, not a finding.** `residual ~ 0.85 ms +
> 57.77 us/op`, worst/median `1.169`. Substitute a real graded worst row when the
> canary returns one and it gives the cut size then. Substituted with the LOCAL
> `2.23x` it returns `45.2%`; **that is one scenario, not a requirement.**
>
> **Where the operations are, which is what makes `45.2%` targetable:** `78` of
> `2,673` operations carry the ENTIRE `168.6e9` FLOPs. The other `97.1%` are
> elementwise `O(n^2)`-and-smaller work — `103` scalar-by-scalar multiplies at
> **zero** FLOPs and `3.9%` of the op budget, `265` vector-by-scalar (`9.9%`),
> `390` row/column broadcasts (`14.6%`). **The FLOP budget and the op budget are
> spent in almost disjoint places.**
>
> **No cut has been made and none is authorised** — the branch opens only when the
> canary's own telemetry lands. Every change in it gets WP16's parity gate and a
> re-metered price.
>
> ## 5. The write-up is now PRIMARY WORK
>
> `docs/writeup-p2-draft.md`. **Near-final by Oct 3, final by Oct 20, due Oct 24**
> (exactly one graded submission ID). Today:
>
> - **§ 7.6 is new** — the fifth route closes a *second* time, on cost, for a
>   structural reason. It is the most transferable thing in § 7.
> - **The abstract's cost bullet was STALE and self-contradicting** — it still
>   stated *"a dense matmul at `4 n^3`, twice the textbook figure"*, the exact
>   claim § 9 already lists as retracted. Replaced with the dtype-correct table in
>   one consistent unit (f32 matmuls/layer): floor allows `6.40`, second-order
>   `2.00`, `+` regenerating K3 `4.91`, `+` the three-generation memory `6.49`.
> - Appendix B has a **twelfth** closed lane, the only one closed on cost.
> - §§ 8 and 9 carry the two new process lessons.
>
> ## 6. Next actions, in order
>
> 1. **Samuel: submit the canary** (or decline). Nothing else moves until it is
>    graded.
> 2. **When graded, read EVERY row's residual** and apply § 7's table below.
> 3. **Write-up** — continuous, and now the default use of a session.
> 4. **Hardening, after the draft**: `1000+` fresh mints through the exact
>    nominee files, determinism twin runs, the Linux memory gate, brokered `gpuq`
>    soaks — **for both nominees**.
> 5. **Nominations by Oct 10**, explicitly designated, never the Rules v14
>    default. **A** = the regenerating flow (the `3.330x` cell is NOT part of it);
>    **B** = `v2.7`. Present both to Samuel with SHAs and one paragraph each;
>    **Samuel designates.** Buffer Oct 10–17; close **Oct 17**.
>
> ## 7. The graded-residual decision table, unchanged and still binding
>
> | graded worst-row headroom | action |
> |---|---|
> | **`>= 4x`** | **stop. No op work.** The uncorrected flow is nominee A as-is |
> | **`2.5x`–`4x`** | reassociate to reach `4x`, **time-boxed `<= 4 days`**, WP16 parity gate (drift `<= 1%`, raw within noise) and a re-metered price on **every** change, then re-canary |
> | **`< 2.5x`** | **one** time-boxed round. If a `> 40%` op cut cannot be reached without drift, **abandon the flow as a nominee** and fall back to `v2.7` + a `v2.6.1`-class diverse pair |
>
> **The `3.330x` branch of that table is now dead** — it was conditional on the
> cell fitting, and it does not.
>
> ## 8. Standing watch, and it is the only one
>
> **Phase-1 winner code (`ednacob`) — the "~Sep 30 harvest" is WITHDRAWN
> (2026-09-05).** Phase-1 private results have not been published: the Phase-1
> board is unchanged (ednacob 1.84e-8, J2W, dstepanov) and no winner code has
> dropped. The 30-day open-sourcing clock starts at **prize announcement**, which
> has not happened, so "closes ~Sep 30" had no basis. **Plan without it.** If it
> does drop, cite it in the write-up's related work and do not build on it.
>
> Unchanged: AUGMENT deprioritised; carrier, output corrector, Arm L, Arm
> M-standalone and every closed lane stay closed; thresholds in flopscope FLOPs;
> nulls differ in **code path**, not parameter value; `artifact=` at every claim;
> committed preflight transcripts; soaks only through `gpuq`.

> **2026-09-04 — WP20 IS ANSWERED. THE CORRECTED CONSTRUCTION GIVES `3.330x`,
> THE `k = 3` CARRY FAMILY'S CEILING IS `3.874x`, AND THE RESULT LANDS IN THE
> PRE-REGISTERED `3`–`4x` NO-ACTION BAND. THE LANE IS NEITHER CLOSED NOR
> REOPENED.**
>
> `estimator.py` untouched at **v2.6.1**. **Nothing submitted, nothing
> designated.** `P06[..064..127]` **UNSPENT.**
>
> ## 1. The disposition, which `CONTRACT WP20` wrote down in advance
>
> | | threshold | measured | verdict |
> |---|---:|---:|---|
> | gate `G-WP20-E2E` | `>= 4.0x` | `2.003x` scored (`P06`), `3.330x` corrected (`P05`) | **FAIL, both** |
> | falsifier | `< 3.0x` | `3.330x` corrected | **does NOT fire** |
>
> > *"The gate fails and the falsifier does not fire. Pre-registered
> > disposition: nothing is promoted, no op work is spent on it, and the lane is
> > not reopened by me. It becomes a strategist decision with a measured number
> > attached."*
>
> **The earlier "closed for good" is formally WITHDRAWN.** So is *"below `3.0x`
> by a deductive argument"*. Both are recorded as errata in `experiments/LOG.md`
> rather than edited away.
>
> ## 2. What the audit changed, and why it mattered
>
> `SKEPTIC WP20` found the scored cell **never used the carried `(3,)` slice** —
> it re-derived the diagonal from truncation leakage while `D3` sat unused two
> lines above. That mechanism came from WP19, where the diagonal genuinely was
> not available. **Keeping it is worth `2.154x -> 3.330x`, paired `t = -20.59`,
> better on `16/16` networks**, and costs an `n`-vector the run had already
> computed. The Skeptic's response surface predicted `3.18x`; measured `3.330x`.
>
> ## 3. What is ESTABLISHED, and it is stronger than the closure would have been
>
> **No member of the `k = 3` carry family reaches `4x`.** The untruncated
> carry — both slices exact, a selftest-verified ceiling — is **`3.874x`** on 16
> fresh networks at the live shape. **The kill line is `>= 10x`.**
>
> | route | measured ceiling |
> |---|---|
> | select a subspace of what the flow holds | `1.45`–`2.10x` **worse** |
> | select by `Sigma`'s dominant range | `= r/n` — **chance** |
> | **predict** `E3` from carried scalars | held out `0.1002`, ceiling `0.2575` |
> | **predict** `E21` from 29 equivariant terms | held out `0.1436`, ceiling `0.1665` |
> | **carry** three generations, both slices exact | **`3.874x`** |
>
> The carried route is **the only one that recovers `E3` at all** (`0.79` against
> `0.10` and `0.14`). **The limit is the DEPTH OF THE MEMORY** — not the
> compression, not the predictability. Three generations of exact memory buys
> under `4x` against a `10x` requirement.
>
> **And `carry3_msketch` is DESTRUCTIVE** — `0.781x`, worse than not correcting,
> `E3` recovery `-1.702`. Two slices read separately must be carried separately.
> WP10's precedent recurring in a place nobody had looked.
>
> ## 4. Next actions
>
> 1. **Samuel: the canary.** Still the highest-value item on the board — the
>    nomination, the op work and Ruling 2(a)'s whole decision tree are
>    downstream of one graded number.
> 2. **A Strategist Packet is drafted** (`docs/STRATEGIST_PACKET_2026-09-04.md`)
>    on three questions this session cannot answer for itself.
> 3. Write-up § 7.5 carries the carried-route result.
> 4. `P06[..064..127]` stays unspent unless the strategist says otherwise.

> **2026-09-03 (LATE) — THE CANARY IS `30/30`, CERTIFIED `36/36`, AND
> PRESENTED. THE STRATEGIST'S TWO RULINGS ARE IN FORCE, THE ALLOCATION TO
> OCT 10 IS FIXED, AND `CONTRACT WP20` — THE LAST ACCURACY QUESTION THIS
> PROJECT WILL ASK — IS PRE-REGISTERED WITH ITS INSTRUMENT BUILT AND ITS FIRST
> JOB QUEUED.**
>
> `estimator.py` untouched at **v2.6.1** (`00240eb1`). **Nothing submitted,
> nothing designated.** `P06` unspent. Contract `40e88fe` precedes every WP20
> number that will exist.
>
> ## 1. RULING 1 EXECUTED — invariant 9's stale item is replaced, and the canary is green
>
> The red item was `d_k / max(scale, 1e-30) < 1e-6`: a max-absolute deviation
> against a fixed bar about `10x` float32 epsilon, written when the K2 flow had
> an f64 backbone. The live design runs the covariance backbone in **f32 on
> purpose** — that is what puts `C/B = 0.0767` on the `0.1` multiplier floor —
> and `SKEPTIC WP18-C` measured curing it at **exactly `2.0x` billing**, i.e.
> `1.53x` worse score. **The item did not ask a candidate to be correct. It
> asked it to be expensive.**
>
> It is replaced by the criterion the invariant's *surviving* clause already
> used, extended to the two deviations the old gate merely **printed**: every
> row's drift, in MSE units, `<= 1%` of the candidate's own raw MSE.
>
> | deviation | old | new |
> |---|---|---|
> | row `k` | `max/scale < 1e-6` | `mse <= 1%` of raw MSE |
> | **all other rows** | **printed, NOT gated** | `mse <= 1%` of raw MSE |
> | **final row** (the scored one) | `mse < 1.4e-8` | unchanged in value, restated |
>
> **A net widening in one place and a net tightening in two.** `d_other` on the
> canary is `1.91e-06` — *larger* than the `d_k` the old gate rejected on — and
> nothing was looking at it. The default denominator is the frozen `1.4e-6`, so
> **the bar on the shipped file did not move**; `ARC_WHEST_RAW_MSE` is clamped
> so an override can only ever tighten.
>
> **`local_sim/test_edge_cases_inv9.py` is the standing proof that the item
> still bites, and it is a gate, not a comment.** Six deliberate positional
> defects — a transposed index, a swapped reduction axis, an off-by-one on the
> depth axis, a basis on the wrong side, an adjacent-position swap, a
> closest-valued-position swap — are still **REJECTED at `5.3e4x`–`3.9e7x` the
> bar**. Two equivariant controls are still **accepted**, because invariant 9
> tests equivariance and not accuracy, and a battery in which everything fails
> proves nothing about specificity. The **detection floor is measured, not
> asserted**: a positional distortion above **`~0.04%` of a row** is caught,
> checked at `0.5x` and `2.0x` that amplitude.
>
> Writing the battery found **two errors in my own test before it found any in
> the invariant**: a value-selected swap is equivariant by construction and
> therefore invisible by design, and the first floor probe injected a second,
> larger defect and rejected at half the claimed floor. Both are recorded in the
> file rather than quietly fixed.
>
> | run | result |
> |---|---|
> | `edge_cases.py estimator_wp18_canary.py` | **`30/30`** |
> | the same with `ARC_WHEST_RAW_MSE=8.5075e-07` (**stricter**, the canary's own raw MSE) | **`30/30`** |
> | `edge_cases.py estimator.py` (v2.6.1) | **`30/30`**, row-k `1e-31` |
> | `scripts/preflight.ps1 -Full` | **`16/16` GREEN** |
> | `certify_candidate.py --candidate estimator_wp18_canary.py` | **`37/37` — CANDIDATE CERTIFIED**, and the `37th` is the battery, as check `3b` |
>
> **THE CHANGE GATE IS RED, AND FOR A KNOWN REASON THAT IS NOT THIS WORK.**
> `preflight -Full` after the Skeptic fixes is **`14 / 16`**: step 3 green *with
> the battery inside it*, and steps **7 and 9 RED on one cause** —
> `p2_range_lab.py::WP20 calib3` has **claimed `P05[..136..143]` and named an
> artifact `gpuq 227` has not produced yet**, so `A7: every artifact-naming
> claim in the REAL ledger is hashed` fails. **This is the claim-then-finalize
> protocol working, not a defect**; both steps go green the moment `227` lands
> and is `finalize`d. It is written here because this is the exact state that
> stayed red for nine commits in August and got rediscovered rather than read.
> The same single cause is the whole of `test_enforcement_labels`' `64/66`.
>
> **Rule 2 RAN and returned PROVISIONAL, and it was right to.** Pinned commit
> `40e88fe`; full entry in `experiments/LOG.md`. It found **four real things**,
> all now fixed:
>
> | finding | what it was |
> |---|---|
> | **a LIVE regression** | `perm_gate` used Python's `max`, which orders with `>`; `nan > x` is `False`, so **a NaN anywhere but the first position was DROPPED before the finiteness check saw it** — `worst = 0.0, ok = True`. The *pre-amendment* code rejected that input. The candidate family under test is the one that returns NaN on degenerate networks. Each component is now checked for finiteness **before** any ordering, with three non-finite cases in the battery |
> | **the "standing proof" ran nowhere** | the battery is called *"a gate, not a comment"* and **nothing executed it** — it was `??` (untracked) in the very preflight transcript meant to prove the change safe. It now runs in `preflight` step 3 (two commands, one step — **the denominator stays 16**) and as `certify` check `3b` |
> | **the cost was promised and not reported** | every mutation sat at `5.3e4x`–`3.9e7x` the bar, i.e. **far above the band the change moved**, so the battery was silent exactly where it mattered. It now measures both floors on the same ramp every run: **`1,541x` and `793x` given away** on `estimator.py` |
> | **and something LIVED in that band** | `VALIDATION.md` § 3.2 assigned invariant 9 a *second job* — *"an independent measurement of v2.4.1's rotation stability… `4.4e-8`"* — three orders of magnitude below the new floor. **The amendment retired a detector the binding protocol still advertised**, and § 3.2's table was stale by up to `75x` besides. § 3.2 is rewritten and now forbids citing invariant 9 for rotation stability |
>
> Three claims of mine were weaker than written and are labelled as such: the
> ruling's quoted wording says **f32-vs-f64** drift while I implemented
> **permutation** drift (the better choice, but an undisclosed divergence in an
> entry whose stated purpose was fidelity to the quote); *"measured at exactly
> `2.0x`"* is a rate-table fact **plus an inference**; and the newly-gated rows
> are a "tightening" that discriminates nothing at `1e5x`–`1e9x` slack.
>
> **One process failure, mine:** the tree moved under the audit — `HEAD` was
> `6d7aa50` against a pin of `40e88fe`, and `certify_guarded.json` was rewritten
> after the pin. Nothing is invalidated (the audited code is byte-identical at
> both commits) but it is the exact hazard this file already records against a
> previous Skeptic. **Rule adopted: once a commit is pinned for an audit, do not
> rewrite any artifact named in that audit's brief until the verdict lands.**
>
> **Deferred deliberately, conservative branch taken:** the Skeptic's follow-up
> probe (re-derive the alignment basis from a near-degenerate permuted
> covariance) is NOT run. § 3.2 is written **as if it had come back
> unfavourably** — the detector is declared retired. Running it could only
> upgrade that claim, never protect anything.
>
> Transcripts committed under `experiments/2026-09-03_inv9_amendment/`.
> Predictions 1, 2, 3 and 5 held; **prediction 4 held in direction but not in
> magnitude** — I said margins `>= 1e5x` and the weakest is `5.3e4x`.
>
> ## 2. THE CANARY IS PRESENTED, AND EVERY BAR ELEMENT IS ON AN ARTIFACT
>
> | | | artifact |
> |---|---|---|
> | archive | `experiments/2026-09-03_wp18_canary/submission_wp18_canary.tar.gz`, `13,858` bytes | |
> | archive sha256 | `e40f6b4de6a10cdcef74b729cc8dbc5eae9a930976e53ba37bee78904fb99960` | |
> | inner `estimator.py`, LF (the shipped bytes) | `252c3c947f130e75b71bdb976293af964ed93a377bfeb7fa8403e5e76a4acddc` | |
> | contents | `estimator.py` (`31,196` B) + `manifest.json` (`529` B) | |
> | **certify** | **`36/36`**, invariant section `30/30` | `certify_guarded.json` |
> | **Linux `8,192` MB gate** | `13/13`, `VmPeak 2266.8 MB`, both controls behaving | `linux_memory_gate.json` |
> | **local failures** | **`0 / 100`**, residual exhausted `0` | `official_residual.json` |
> | **multiplier** | `C/B = 0.0766870` -> **`0.100000`, the floor**, with `+2**24` | `certify_guarded.json` |
> | **local worst-row headroom** | **`2.23x`** against a `>= 1.5x` canary bar | `official_residual.json` |
>
> **Two caveats that travel with the headroom number rather than being smoothed
> over.** The `2.23x` was measured on snapshot `22db10a` at **`2,646`
> operations**, and the guarded file bills **`2,674`** (`+1.17%`). The guarded
> file's OWN graded-path read — `certify` § 10, mini split — is
> **`204.45 ms -> 2.0x`**. Both clear `1.5x`; **neither clears `4x`, which is the
> NOMINATION bar and is deliberately not what this is.**
>
> **The label is exact: *residual-measurement canary; nomination pending graded
> telemetry*. Submitting is Samuel's. ~~No agent runs `whest submit`~~ and no
> agent designates.**
>
> > **SUPERSEDED 2026-09-05 in part.** Samuel changed the rule: **an agent MAY run
> > `whest submit` once he has approved that specific submission.** `AGENTS.md`
> > hard rule 6 is the authority. **The designation half is UNCHANGED** — no agent
> > designates, ever.
>
> ## 3. RULING 2 — the allocation, fixed
>
> **(a) When the canary's graded telemetry returns, read EVERY row's residual.**
>
> | graded worst-row headroom | action |
> |---|---|
> | **`>= 4x`** | **stop. No op work.** It is nominee A as-is |
> | **`2.5x`–`4x`** | reassociate to reach `4x`, **time-boxed `<= 4 days`**, WP16 parity gate (drift `<= 1%`, raw within noise) and a re-metered price on **every** change, then re-canary |
> | **`< 2.5x`** | **one** time-boxed round. If a `> 40%` op cut cannot be reached without drift, **abandon the flow as a nominee** and fall back to `v2.7` + a `v2.6.1`-class diverse pair |
>
> **(b) The write-up is a co-equal `$20k` objective and becomes real work from
> ~Sep 21**, continuous before. Added today: **§7.4**, which states what
> §§7.1–7.3 add up to as one mechanistic result — the two discarded objects are
> one object; corrected jointly at rank `128` of `1024` the end-to-end drift
> falls **`11.1x`**, so the lane is blocked by **neither rank nor cost**; and
> neither slice is predictable from the carried state above `0.15` held out,
> both at ceilings within `0.03`–`0.16` of their own basis. **A regenerating
> cumulant flow's carried state does not determine what that flow discards** —
> the information is *destroyed* by the regeneration, not *compressed* by it.
> §6.8 already carried the graded 100-row result; §7.3 already carried the
> four-way closure and the `E3` convexity table.
>
> **(c) Nominations by Oct 10, explicitly designated.** **A** = the best
> certified flow with graded worst-row `>= 4x`. **B** = `v2.7`. Close **Oct 17**;
> write-up **Oct 24**.
>
> ## 4. CONTRACT WP20 — pre-registered, instrument built, first job queued
>
> **The question:** does a **CARRIED** three-generation sketch of the discarded
> all-distinct block, rank `r ∈ {128, 192}`, recover `E3` as well as `E21`, and
> what is the joint end-to-end drift reduction on fresh networks?
>
> **It is not a reopening.** Arm M was killed as a **standalone
> generate-and-contract** design on a `1`–`2`-step memory gate; Arm L died
> **predicting** the correction from carried state. A carried sketch is neither,
> and WP19's own findings motivate it — `E3` convexity (`5.06x` from `E3`'s
> quality alone), the joint `11.1x` oracle, and *"state does not determine what
> it discards"*.
>
> **What it actually measures is a CEILING, and the contract says so in
> advance.** The cell truncates *after* an exact three-layer propagation, where
> a deployable design would propagate the truncated state. So:
> **a ceiling that fails closes the lane strictly; a ceiling that passes
> promotes nothing and buys a second contract.**
>
> | | |
> |---|---|
> | **gate** | joint end-to-end **`>= 4.0x`** paired arithmetic through `decide()`; FLOPs `<= 51,249,140,941` added (`0.02330 B`, the repository's own `C/B` rather than the directive's `0.0236`); ops within the nomination residual bar the canary returns — **and that clause is UNSCORABLE until the telemetry exists, recorded `UNSCORED`, never `passed`** |
> | **falsifier** | `< 3.0x`, or residual-infeasible -> **the accuracy lane closes for good.** Strict, because the measured cell bounds the deployable one |
> | **the `3`–`4x` band** | pre-registered: gate fails, falsifier does not fire, **nothing is promoted, no op work, and I do not reopen it.** A strategist call with a number attached |
> | **block** | `P06[..000..063]`. `P06[..064..127]` was Arm L's and **stays unspent** — re-purposing a reserved half inside the contract that would benefit is the move this protocol exists to prevent |
> | **my prediction** | **`3.0x`, and I expect the gate to FAIL** (`P(>= 4x) ~ 0.15`, `P(< 3x) ~ 0.50`). Recorded deliberately: a contract written to be passed is not evidence |
>
> **The `(3,)` scalar is fitted on `P05` and FROZEN before `P06` is touched.**
> `P06`'s charter says *"No tuning of any kind here"* and that binds; `P05`'s
> says tuning is permitted there and nowhere else. `gpuq 227` is that
> calibration (8 nets, `P05[..136..143]`, claimed on the host). Its two
> constants go into `CARRY3_C` **in their own commit**, whose git timestamp
> precedes the `P06` job. `_parse_cells` **REFUSES** a `carry3_sketch` cell for
> which no frozen `c(r)` exists.
>
> **Two development smokes are DISCLOSED in the contract** (`256 x 4`,
> `256 x 6`, `f64`, out-of-block seeds, no gate scored). They earned their
> disclosure: they **corrected the null clause before it was committed** — a
> random rank-`r` sketch recovers `r/n` *by chance*, measured `0.2407` against
> `0.250`, which is WP14's containment result reappearing, and the original
> clause would have voided a valid run — and they **revised prediction P3 down**,
> since `E3` recovery trails `E21` recovery badly at `c = 1` (`0.525` vs
> `0.880`).
>
> ## 5. Next actions, in order
>
> 1. **Samuel: submit the canary** (or decline). Everything on the nomination
>    path is downstream of the graded per-row residual it buys.
> 2. `gpuq 227` -> freeze `c(128)`, `c(192)` -> commit -> submit the `P06` job.
>    Answer due **~Sep 14**.
> 3. Write-up continues; it is now a co-equal objective, not a trailing one.
> 4. `estimator.py` stays at v2.6.1 until a promotion contract exists.



> **2026-09-03 (END OF SESSION) — ARM L IS DEAD ON ITS OWN FALSIFIER. THE
> CORRECTION LANE IS CLOSED FOUR WAYS ON MEASURED CEILINGS. THE CANARY IS
> `29/30` AND PACKAGED. THE KILL LINE'S CONDITION IS MET THREE WEEKS EARLY, AND
> WHAT REMAINS IS A PORTFOLIO DECISION — WHICH IS THE STRATEGIST'S.**
>
> `estimator.py` untouched at v2.6.1. **Nothing submitted, nothing designated.**
> `P06[..000..127]` **released untouched** — its gate's falsifier fired upstream.
> Contract `49e2e42` precedes every number here.
>
> ## 1. ARM L IS DEAD, BY THE CONTRACT IT WAS WRITTEN UNDER
>
> `CONTRACT WP18` § B's falsifier: *"`R^2 < 0.90` on BOTH legs held out."*
>
> | leg | held-out `R^2` | **in-sample ceiling** | null |
> |---|---:|---:|---:|
> | `E3` — pointwise map on carried per-neuron scalars | **`0.1002`** | **`0.2575`** | `-2.01` |
> | `E21` — 29-term equivariant matrix basis | **`0.1436`** | **`0.1665`** | `-221,237` |
>
> **Both below `0.90`. The falsifier fires. No threshold was moved.** Each fit
> sits within `0.03`–`0.16` of its basis's own ceiling, so the shortfall is
> **representational, not statistical** — and both ceilings were verified against
> solver artifacts (E21: three solvers and five `rcond` settings agree to six
> decimals, Gram full rank `29/29`).
>
> **The other falsifier clauses did NOT fire**, which says where the failure is
> not: `G-DIAG-NOTWORSE` passed `0/8` at three ranks (the injection is **not**
> destructive — WP10's precedent does not recur), and both nulls are clean by
> enormous margins. The maps are not accidentally right and not harmful. They are
> not there.
>
> ## 2. The lane, closed four ways, with a ceiling on each
>
> | route | outcome | where |
> |---|---|---|
> | **select** a subspace of what the flow holds (5 column-weight rules) | `1.45x`–`2.10x` **worse** than not correcting | WP13 |
> | **select** by `Sigma`'s dominant range | containment `= r/n` to three decimals — **chance** | WP14 |
> | **predict `E3`** pointwise from carried scalars | held out `0.1002`, ceiling **`0.2575`** | WP18 § B |
> | **predict `E21`** from 29 equivariant terms | held out `0.1436`, ceiling **`0.1665`** | this session |
>
> **And the sensitivity that explains all of it** (WP19, 8 nets, live shape,
> `E21` held at **oracle**, `r = 192`):
>
> | `E3` quality | end-to-end |
> |---|---:|
> | absent | `2.223x` |
> | best free route (`R^2 = 0.453`) | `3.727x` |
> | exact | `18.873x` |
>
> **`10x` needs BOTH legs near-exact. Neither can be predicted above `0.15`.**
>
> ## 3. WP19: the one component that works, and why it is still unusable
>
> `E3` and `E21` are the **diagonal** and the **zeroed off-diagonal of one
> tensor**. Reading `E3` off a rank-192 reconstruction of `E21` — one fitted
> scalar, no new features, on an object already materialised — gives `R^2 = 0.4529`
> with `c = 1.019` (essentially **unbiased**), and is non-destructive on `8/8`
> nets at three ranks. **Both its gates still failed as written**: `G-DIAG-R2`
> `0.4529 < 0.50`, `G-DIAG-E2E` `3.727x < 5x`. `r = 256` would clear `5x` at
> `5.748x` and is **beyond the affordability horizon**; moving the gate there is
> refused in writing. The component is free and correct; it needs an `E21` we
> cannot produce.
>
> ## 4. The canary: `29/30`, packaged, awaiting a human decision
>
> | | |
> |---|---|
> | archive | `experiments/2026-09-03_wp18_canary/submission_wp18_canary.tar.gz`, `13,858` bytes |
> | archive sha256 | `e40f6b4de6a10cdcef74b729cc8dbc5eae9a930976e53ba37bee78904fb99960` |
> | inner `estimator.py` **LF** (shipped bytes) | `252c3c947f130e75b71bdb976293af964ed93a377bfeb7fa8403e5e76a4acddc` |
> | same content **CRLF** (what `certify_guarded.json` names) | `3faf21b3c816d72c…` |
> | multiplier | `0.100000` at the floor, `C/B = 0.0766870`, with `+2**24` |
> | steady-state price of the guards | `+31` ops (`+1.17%`), `+8,162,816` FLOPs |
> | Linux `8,192 MB` gate | `13/13`, `VmPeak 2266.8 MB`, `3.61x`, both controls behaving |
> | official smoke | `0 failed`, worst residual `245.52 ms` |
> | **the one red item** | invariant 9, permutation equivariance — float32 precision, verified at a `1e9` f32/f64 ratio |
>
> **`CONTRACT WP18` § A: any red item means it is not presented as gate-green.
> Submitting is a human override — Samuel's, not an agent's.** All three canary
> Skeptics say the bytes are operationally safe at `1024 x 16`.
>
> ## 5. Three Skeptics ran and all three returned PROVISIONAL
>
> Between them: the price quoted against the wrong row; **two false
> justifications written into a submission file's comment**, one revision apart;
> a claim true only at `1024 x 16` stated unqualified; a stated blocker that was
> false; an oracle table read one row off; **an endpoint check that was `1 - x/x`
> and inverted**; a price metered on the wrong array type (which resolved the
> `15 n^2` mystery two audits had left open); a guard that masked the regression
> it caused; and an accuracy figure describing an array the scorer never sees.
> **All fixed and recorded.** `SKEPTIC WP18`, `WP18-B`, `WP18-C` in
> `experiments/LOG.md`.
>
> ## 6. THE DECISION THAT IS NOT AN AGENT'S
>
> The `~Sep 24` kill line's condition — *no `>= 10x` end-to-end drift reduction
> on fresh nets* — **is met, three weeks early, on evidence rather than
> schedule.** The standing instruction on that condition is: freeze Arm L,
> nominate the canaried uncorrected flow (only if graded worst-row headroom
> `>= 4x`, else after a sized op cut) plus `v2.7` as the diverse pair, finish the
> write-up.
>
> **Two things now need a decision that repository evidence cannot make**, and a
> **Strategist Packet is drafted** (`docs/STRATEGIST_PACKET_2026-09-03.md`):
>
> 1. **Where do the remaining ~3 weeks go?** Op-count reduction is the only
>    remaining lever on the nomination and it is gated on a graded number we do
>    not have. The write-up is now the co-equal `$20k` objective and has a
>    complete four-ceiling negative result to carry.
> 2. **Does the canary go out at `29/30`**, given the one red item is a float32
>    property the graded distribution cannot observe, and given that the graded
>    per-row residual is the only instrument that can settle the nomination bar?
>
> ## 7. Next actions
>
> 1. **Samuel: the canary decision.** Everything else on the nomination path is
>    downstream of the graded per-row residual it would buy.
> 2. **Strategist packet** — drafted, awaiting hand-off.
> 3. Write-up is current: **§6.8, §7.1, §7.2, §7.3**. §7's status row now reads
>    *closed four ways on measured ceilings*.
> 4. `P06` is unspent and stays that way unless a lane reopens on **external**
>    evidence.


> **2026-09-03 (WP18, END OF SESSION) — THE CANARY IS `29/30`, REPACKAGED AFTER
> THREE INDEPENDENT AUDITS, AND WAITING ON ONE HUMAN DECISION. ARM L'S FIRST
> ROUTE DIED AT A CEILING; A CHEAPER ONE IT NEVER TRIED IS NOW THE NEXT
> MEASUREMENT.**
>
> `estimator.py` untouched at v2.6.1. **Nothing submitted, nothing designated.**
> `estimator_wp15_k3_canary.py` left byte-identical so the graded `8.75e-07`
> keeps naming an exact file. Contract `49e2e42` precedes every number here.
>
> **Three Skeptics ran today and all three returned PROVISIONAL. Between them
> they found nine things wrong, two of them in submission-file comments and one
> in a check that could not fail. Every one is fixed and recorded**
> (`SKEPTIC WP18`, `WP18-B`, `WP18-C` in `experiments/LOG.md`).
>
> ## 1. The strategist's three rulings, in force
>
> | | |
> |---|---|
> | **1** | the `>= 4x` residual bar **SPLITS**. A **canary** needs certify green, the Linux memory gate, `0/100` local failures, the FLOP floor with `+2**24`, worst-row headroom `>= 1.5x`. `>= 4x` moves to the **nomination** bar and is scored on the grader's per-row telemetry. The `3 x 100` soak is OFF the canary path |
> | **2** | **Arm M killed** as a standalone; its three-generation closed forms survive as Arm L features |
> | **3** | floor nominee and Arm L in **parallel**; Arm L starts with `E3` |
>
> Kill line **~2026-09-24**. Designate by **Oct 10**, close **Oct 17**, due **Oct 24**.
>
> ## 2. The canary: `29/30`, and the one red item is float32 precision
>
> Running `certify_candidate` on the canary **for the first time** returned
> `REFUSED` at **5 of 30** invariants: NaN output on degenerate networks,
> `151` of `1024` **scored-row** entries negative, and order-dependent FLOPs from
> a constants cache. Fixed in a new file `estimator_wp18_canary.py`, then fixed
> again twice more after audits.
>
> | | graded canary | **post-audit canary** |
> |---|---:|---:|
> | **invariants** | `5 / 30` | **`29 / 30`** |
> | counted f32 FLOPs (certify) | `168,629,787,238` | `168,636,407,398` |
> | `C/B` / multiplier | `0.0766836` / `0.1` | `0.0766870` / `0.1` — **floor, with `+2**24`** |
> | **steady state, GRADER-TYPE (`fnp`) arrays** | `2,642` ops / `168,628,211,814` | `2,673` ops / `168,636,374,630` |
> | **price of the guards** | | **`+31` ops (`+1.17%`), `+8,162,816` FLOPs (`+0.00484%`)** |
> | FLOPs network-independent | **NO** | **YES** |
> | unexplained value changes, rows 1.., **at `1024 x 16`** | — | **`0`** of `643` |
> | `depth 32 x width 64` final-row mean | `0.2103` | **`0.2103`** |
> | depth-1 MLP | OK | **OK** |
> | Linux `8,192 MB` gate | not run | **`13/13`**, `VmPeak 2266.8 MB`, `3.61x`, both controls behaving |
> | official smoke | `0 failed` | `0 failed`, worst residual `245.52 ms` |
>
> **THE ARCHIVE**
>
> | | |
> |---|---|
> | `experiments/2026-09-03_wp18_canary/submission_wp18_canary.tar.gz` | `13,858` bytes |
> | archive sha256 | `e40f6b4de6a10cdcef74b729cc8dbc5eae9a930976e53ba37bee78904fb99960` |
> | inner `estimator.py`, **LF** (the shipped bytes) | `252c3c947f130e75b71bdb976293af964ed93a377bfeb7fa8403e5e76a4acddc` |
> | the same content **CRLF**, which is what `certify_guarded.json` names | `3faf21b3c816d72c…` |
> | `validate_package` | `ok=True, issues=[]` |
> | label | **residual-measurement canary — NOT a nomination** |
>
> **The one red item: invariant 9, permutation equivariance.** `d_k/scale ~ 1.1e-6`
> against a `1e-6` bar, while the same invariant's implied-MSE clause passes by
> five orders of magnitude (`1.4e-13` against `1.4e-8`). `SKEPTIC WP18-C`
> verified the cause by a method this repository had not used — the same flow on
> the same networks, f32 against f64, gives `d_k/scale` `2.116e-06 -> 2.365e-15`.
> **A ratio of `~1e9` is the eps ratio, not an index bug.** Curing it needs
> float64 in the covariance backbone, measured at exactly `2.0x` billing, which
> moves `x = 0.0767 -> 0.153`, off the `0.1` floor, for a `1.53x` worse score.
> *(Earlier wording here said invariant 9 is "unmeetable by any competitive
> Phase-2 candidate". That is a universal the evidence cannot reach — what is
> established is the mechanism and the cost for THIS candidate.)*
>
> **`CONTRACT WP18` § A says any red item means the canary is not presented as
> gate-green, and that binds. Submitting is a human override of a pre-registered
> gate — Samuel's, not an agent's.** All three Skeptics say the bytes are
> operationally safe at `1024 x 16`.
>
> ## 3. What the three audits found, and it is worth reading before trusting anything here
>
> | audit | finding | status |
> |---|---|---|
> | **WP18** | price quoted against the graded file's **first MLP**, not its steady state | fixed; probe now prints a steady-state row |
> | **WP18** | `ALPHA_CAP = 40`'s justification **mathematically false** — `w(0,1)` is `E[ReLU]` and grows linearly in `alpha` forever | comment corrected; cap raised to `1e6` |
> | **WP18** | "every changed value was negative or non-finite" true at `1024 x 16`, **false at `64 x 32`** | every statement now carries its shape |
> | **WP18** | invariant 8's stated blocker **false** — `estimator.py` already returns float64 and already contains the fix | fix applied, `28/30 -> 29/30` |
> | **WP18-B** | the `10x` argument cited **single-slice** oracle cells; the joint cell `e3svd:128` is **`11.098x`**, above the gate | withdrawn everywhere; see §5 |
> | **WP18-B** | `E3` is the **diagonal** of `E21`'s own tensor; `diag(rank-192 E21)` + one scalar recovers `0.61`–`0.66` | now the next measurement, `CONTRACT WP19` |
> | **WP18-B** | my B1 endpoint check was `1 - x/x` — **could not fail, and was inverted** | replaced with a cross-path check; mutation battery `4/4` |
> | **WP18-C** | **`ALPHA_CAP = 1e6` made it WORSE** — zeroed the whole scored row at `64 x 32`; paired mean MSE `20.4` (cap 40) vs `10,957` (cap 1e6) | **reverted to `40`**, chosen empirically |
> | **WP18-C** | depth-1 MLPs **raise** — a hard-rule-7 regression the row-0 fix introduced | guarded; verified at depths 1, 2, 16 |
> | **WP18-C** | the price was metered on **raw numpy**, not `fnp` arrays — FLOP delta understated `4.36x` | fixed; **this also resolves the `15 n^2` mystery two audits left open** |
> | **WP18-C** | `1.826e-16` is **pre-coercion** — `materialise_predictions` casts to float32 inside the metered window | claim bounded; the fix is kept for consistency with the incumbent, and buys nothing scored |
>
> ## 4. ARM L: `G-L-FIT-E3` FAILS at `0.1002` — and the route that was never tried
>
> `gpuq` 191. `128` `P05` nets (claimed and **finalized**, ledger `66/66`),
> `1,441,792` labelled rows, `24` frozen features.
>
> | | |
> |---|---|
> | **`G-L-FIT-E3` layer-averaged held-out `R^2`** | **`0.1002` vs `0.90` — FAIL** |
> | per-layer, leave-networks-out | `0.1595` |
> | **per-block ceiling, IN SAMPLE** | **`0.2575`** (`0.2397` df-adjusted) |
> | permuted-coefficient null | `-2.0117` — `G-NULL` clean |
>
> **The ceiling is the finding** and `SKEPTIC WP18-B` verified it five independent
> ways (QR, two `lstsq` `rcond` settings, explicit SVD, normal equations), found
> rank `24/24` with no silent truncation, and showed adding 300 pairwise products
> and ten new row contractions moves honest CV only `0.292 -> 0.306`. **A
> pointwise map on the flow's carried per-neuron scalars cannot work.**
>
> **But *"Arm L's cheap leg is dead"* is OVERSTATED.** `E3` and `E21` are the
> diagonal and the zeroed off-diagonal of **one** discarded tensor (verified
> against upstream: `_factored_get_dslice(.,(3,)) == diag(T)` to `0.0`). Reading
> the diagonal off a rank-192 reconstruction of `E21`, with **one fitted scalar**,
> recovers `0.61`–`0.66` of `E3` — **`2.5x` better than the whole 24-column basis,
> free.** One route died; a cheaper one was never tried.
>
> ## 5. THE `10x` CLAIM I GOT WRONG, corrected
>
> `CONTRACT WP18` B5 cited `svd:*` cells as bounding the joint correction. They
> are **`(2,1)`-only**. Recomputed paired from `c2_e2e_oracle_rank.json`:
>
> | cell | x better than `none` | per-net range |
> |---|---:|---|
> | `e3` alone | `1.551` | 1.42–1.65 |
> | `svd:256` alone | `2.376` | 2.14–2.67 |
> | **`e3svd:128` — BOTH, at the affordable rank** | **`11.098`** | **9.86–12.63** |
> | `e3svd:512` — both | `333.8` | 302–373 |
>
> **The joint oracle CLEARS `G-L-E2E`'s `10x`**, and `experiments/LOG.md` said so
> on 2026-09-02 in plain words. **WITHDRAWN**: *"a fitted map cannot beat its own
> oracle, and the oracles are `2.534x` and `1.551x`"*, *"`10x` is very unlikely"*,
> and *"the kill line is the expected outcome rather than the contingency."*
> That was mood presented as arithmetic. What an *imperfect* `E21` does to the
> joint cell is **unmeasured**, and will not be asserted again before it is.
>
> ## 6. NEXT ACTIONS, in order
>
> 1. **Samuel decides on the canary** (`29/30`, invariant 9 only). On submission,
>    read the graded per-row residual against the **nomination** bar (`>= 4x`).
>    If `2.5x`–`4x`, reassociate and re-canary; if `< 2.5x`, cut operations hard.
>    **Sizing waits for the graded number.**
> 2. **`CONTRACT WP19` — `e3diag_svd:r`**, pre-registered and unrun: correct
>    `(2,1)` with its rank-`r` SVD and `(3,)` with **the diagonal of that same
>    reconstruction** times one scalar, `r in {128, 192, 256}`, `1024 x 16`, 8
>    nets, seeds `P05[40902005128..159]`. Gates `G-DIAG-R2 >= 0.50`,
>    `G-DIAG-E2E >= 5x`, and **`G-DIAG-NOTWORSE`** — a destructive injection is a
>    live possibility and WP10 is the precedent. It settles whether `10x` is
>    reachable with **no `E3` map at all**.
> 3. **`E21`'s Gram/moment pass** — basis declared before its moments are
>    computed. Still the thing `G-L-FIT-E21` needs.
> 4. Re-run the Linux memory gate **whenever the candidate changes** — § 9 finds
>    its report by FILENAME, so a stale report for the same filename is still read.
>
> ## 7. Write-up
>
> `docs/writeup-p2-draft.md` has **§6.8** (the 100-row graded confirmation and the
> residual-dispersion finding), **§7.1** (Arm M's memory depth) and **§7.2** (the
> `E3` ceiling, now carrying the corrected oracle table and the lesson that a
> component ablation whose parts do not sum to the headline means you have
> mis-read your own table). §7's stale *"the one experiment we would run next"* is
> replaced — it was run, and the answer is `r/n` to three decimals.


> **2026-09-03 (WP18) — THE STRATEGIST RULED, THE CANARY WAS CERTIFIED FOR THE
> FIRST TIME, AND IT FAILED 25 OF 30 INVARIANTS. THE GUARDED FILE IS 28/30 AND
> WAITING ON ONE DECISION THAT IS NOT AN AGENT'S.**
>
> `estimator.py` untouched at v2.6.1. Nothing submitted, nothing designated.
> `estimator_wp15_k3_canary.py` left **byte-identical** so the graded `8.75e-07`
> keeps naming an exact file. Contract `49e2e42`, guards + price `85f3055`.
>
> ## The three questions came back answered
>
> | ruling | what it says |
> |---|---|
> | **1** | **the `>= 4x` residual bar SPLITS.** A **canary** needs certify green, the Linux memory gate, `0/100` local failures, the FLOP floor with `+2**24`, and worst-row headroom `>= 1.5x`. The `>= 4x` moves to the **nomination** bar and is scored on the grader's own per-row telemetry, which is the only instrument that can settle it. The `3 x 100` soak is ruled OFF the canary path |
> | **2** | **Arm M is KILLED as a standalone.** `G-M-DEPTH` failed as written and the contract is honoured, not re-argued. Its three-generation closed forms survive as **Arm L features** |
> | **3** | **floor nominee and Arm L run in PARALLEL**, and Arm L starts with **`E3`**, which nothing in this project has ever targeted |
>
> Kill line moves to **~2026-09-24**. Designate explicitly by **Oct 10**, close
> **Oct 17**, due **Oct 24**.
>
> ## What running `certify_candidate` on the canary actually found
>
> **It had never been run on that file.** `CONTRACT WP18`'s prediction A1 said it
> would pass everything; it returned `CANDIDATE REFUSED — 3 failed` and the
> 30-invariant suite scored **5 of 30**.
>
> | defect | evidence |
> |---|---|
> | **NaN / non-finite output** on degenerate networks | `4,352` / `6,400` / `6,144` non-finite entries; `-4.037e+08` at `32 x 64`. Root cause: `alpha = mu/sigma` unbounded → Horner overflows to `inf` while `phi(alpha)` underflows to `0` → `inf * 0`. **A NaN fails the MLP and a failed MLP scores as zeros** |
> | **negative means on the SCORED row** | `151` of `1024` final-layer entries, worst `-2.831e-06` |
> | **order-dependent FLOPs** | the width-constant cache made MLP 1 pay `1,575,424` FLOPs and MLP 2..n pay none |
>
> Fixed in a **new** file `estimator_wp18_canary.py`: `ALPHA_CAP = 40`
> (`fnp.clip`), `fnp.where(isfinite)`, `fnp.maximum(out, 0)`, cache removed.
> The cap is **measured** — the uncapped file's worst `|alpha|` over 8 He-init
> nets at the live shape is `9.0400`, a `4.4x` margin (`alpha_range_1024x16.json`).
>
> | | graded file | guarded file |
> |---|---:|---:|
> | invariant suite | **5 / 30** | **28 / 30** |
> | counted f32 FLOPs | `168,629,787,238` | `168,629,918,310` (`+131,072`) |
> | `C/B` / multiplier | `0.0766836` / `0.1` | `0.0766840` / `0.1` — **floor, with `+2**24`** |
> | flopscope **operations** | `2,647` | `2,666` (**`+19`, `+0.72%`**) |
> | FLOPs equal on 4 nets in ONE process | **NO** | **YES** |
> | Linux `8,192 MB` gate | not run | **`13/13`**, `VmPeak 2258.8 MB`, `3.63x` |
>
> All `643` changed values were negative or non-finite in the graded file; **`0`
> unexplained**. ~~Headroom moves `2.23x -> 2.215x`~~ — **`2.215x` WITHDRAWN by
> `SKEPTIC WP18`**: three instruments disagree about the guards' residual price
> *in sign*, and `official_residual.json` is the LOCAL official runner, not
> graded telemetry. The conservative reading is `1.536x`, which clears the canary
> bar's `1.5x` by `2%`, not `48%`.
>
> ## THE ONE THING BLOCKING THE CANARY, AND IT IS SAMUEL'S CALL
>
> `28/30`. **Both remaining invariants are float32 measured against a float64
> threshold**, and the candidate cannot be float64 — an f64 flow bills at twice
> the rate, `x = 0.0767 -> 0.153`, the multiplier leaves the `0.1` floor, and the
> score is `1.53x` worse, which throws away the whole `4.63x` advantage and lands
> level with `v2.7`.
>
> | invariant | threshold | v2.6.1 | canary |
> |---|---|---:|---:|
> | **8** layer-1 closed form | `rel < 1e-12` | `1.7e-16` | `2.0e-07`–`4.4e-07` |
> | **9** permutation equivariance | `d_k/scale < 1e-6` **and** implied MSE `< 1.4e-8` | `2.2e-15` | `1.1e-6`, **implied MSE `1.4e-13`** |
>
> Invariant 9's own second clause passes by **five orders of magnitude**.
> Invariant 8 could be met alone by computing layer 1's variance in f64 (`~4`
> operations) — but that requires the returned array to become float64, an
> interface the grader has never seen from us, and **it was deliberately not done
> on the day the file ships** (`writeup` §6.7 is exactly that lesson). It would
> give `29/30` and would not touch invariant 9.
>
> **THE ARCHIVE IS BUILT AND WAITING.** `validate_package -> ok=True, issues=[]`.
>
> | | |
> |---|---|
> | archive | `experiments/2026-09-03_wp18_canary/submission_wp18_canary.tar.gz`, `11,364` bytes |
> | sha256 | `fd57cebf7c9856b7c54a53845caae5fb01c62006bb2f5a05828cb17d975fc33e` |
> | inner `estimator.py` sha256 | `d3accc63045fe2e643d83a95b15c2b2ac06ffa2134d44d90d56c366f08d7fae3` |
> | label | **residual-measurement canary — NOT a nomination** |
>
> ## NEXT ACTIONS, in order
>
> 1. **Samuel decides** whether `28/30` with those two float32 invariants is
>    submittable as an undesignated canary, or whether the f64 layer-1 row goes in
>    first (`29/30`), or whether thresholds 8 and 9 get qualified by dtype. **No
>    agent moves a threshold to make its own candidate pass.**
> 2. On submission: read the graded per-row residual against the **nomination**
>    bar (`>= 4x` worst-row headroom). If `2.5x`–`4x`, reassociate to that target
>    and re-canary; if `< 2.5x`, cut operations hard. Sizing waits for the graded
>    number and is not guessed from the local one.
> 3. **Arm L, in parallel and already moving.** `gpuq` 191 is queued: the oracle
>    `(E3, E21)` labels on **128 `P05` nets** (`40,902,005,000..127`, claimed).
>    `p2_wp18_labels.py` selftest is `3/3` at `128 x 4` in **both** f64 and f32,
>    including a permutation-equivariance check through the WEIGHTS that caught
>    its own first version using the wrong index convention. `p2_wp18_fit.py` is
>    written with all five fitting choices declared **before the labels exist**.
> 4. `G-L-FIT` is **SPLIT**: `G-L-FIT-E3` and `G-L-FIT-E21`, both `>= 0.90`
>    held-out, leave-networks-out. That is a tightening — the old gate could pass
>    while `E3`, ~40% of the error, went untested. First held-out `R^2` on record
>    by ~**Sep 13**.
> 5. `E21`'s Gram/moment pass is deliberately SECOND: declaring the basis and
>    then computing its moments cannot be reversed into a basis chosen after
>    seeing the target.
>
> ## Write-up is current
>
> `docs/writeup-p2-draft.md` gained **§6.8** (the 100-row graded confirmation and
> the residual-dispersion finding) and **§7.1** (Arm M's memory depth,
> `0.338/0.573/0.732`, the gate failing, and why `73%` of one slice is `~44%` of
> the error). §7's stale closing paragraph — *"the one experiment we would run
> next"* — is replaced: it was run, and the answer is `r/n` to three decimals.


> **2026-09-03 (END OF SESSION) — THE FIX IS CONFIRMED ON THE DEPLOYMENT
> AUTHORITY. THE RESIDUAL BAR IS THE ONLY THING BETWEEN US AND A NOMINATION, AND
> IT DOES NOT CLEAR.**
>
> `preflight.ps1` **GREEN 12/12** at `0358765`. `verify_portability` 103/103.
> Nothing submitted, nothing designated, `estimator.py` untouched at v2.6.1.
>
> ## The graded result, on 100 official rows with a matched control
>
> | | raw MSE | adjusted | utilization | failed |
> |---|---:|---:|---:|---:|
> | v2.6.1 (shipped), control | `4.05e-06` | `4.05e-07` | `0.06257` | 0/100 |
> | **fixed K3 canary** | **`8.75e-07`** | **`8.75e-08`** | `0.07668` | **0/100** |
> | | **`4.63x` better** | | multiplier `0.10000` both | |
>
> The control reproduces the `4.050566e-06` this repository has carried since
> August **to `0.01%`**. The retired canary scored `2.8192e-06`; this is `3.22x`
> better on five times the rows. **Every accuracy claim in this project now has
> graded backing.**
>
> ## The blocker, with its number
>
> | | |
> |---|---|
> | worst-MLP residual (100 rows, `--detail full`) | `0.1797 s` |
> | **worst-run headroom vs the `0.4 s` cliff** | **`2.23x`** |
> | standing soak requirement | **`>= 4x` on the worst of `3 x 100`** |
> | verdict | **DOES NOT CLEAR** |
>
> Operations are the right lever and now have a clean coefficient: the official
> mean-residual ratio to the control is `5.82x` where the local **operation**
> ratio is `5.98x`. Reaching `4x` needs worst `<= 0.1 s`, i.e. **`~1,470`
> operations per MLP (`~92` a layer) against today's `2,646` / `165`**.
>
> **The remaining reductions are not IEEE identities.** Everything done so far
> was byte-exact and verified as such; what is left — factoring
> `0.5 C11^2 X + 0.25 D22 X`, batching the thirteen two-dimensional terms through
> a 3-D stack — reassociates, and the stack materialises `54 MB` at `n = 1024`
> which forum 18101 says costs residual. Each needs a measured drift check.
>
> **AND THE `4x` BAR WAS CALIBRATED AGAINST NOISE THE AUTHORITY DOES NOT HAVE.**
> It was set when `VALIDATION.md` §2b-ii put run-to-run spread at `~2.4x` — a
> LOCAL figure from a contended host. On the authority the per-MLP spread across
> 100 rows is **`1.20x` min-to-max**, with the worst MLP `13%` above `p99`.
> **This is flagged and deliberately NOT acted on**: relaxing a standing policy
> is Samuel's decision and the strategist's, not the decision of the agent that
> would benefit from the answer.
>
> ## `G-M-DEPTH` FAILED
>
> `1024 x 16`, 16 `P01` nets, f32, f64 control agreeing to `1.6%`. One / two /
> three un-regenerated layers recover **`0.34` / `0.57` / `0.73`** of `E21`; the
> gate is written on two ages needing `0.70`. **Prediction P1 refuted** (it said
> `0.50` and `0.70`). **Arm M is not killed** — the falsifier is "`< 70%` at all
> three ages" and three give `0.73` — but `E21`'s memory is **three generations
> deep, not two**, and that is a cost case the arm did not previously need.
> Moving the gate to `k = 3` because `k = 3` passes is refused in writing.
>
> ## What is DONE, and what is NOT
>
> | the session's own definition of done | state |
> |---|---|
> | parity gate passed | **DONE** — both clauses, `SKEPTIC WP16` ACCEPT, and confirmed on graded data |
> | op count reduced and re-metered | **DONE** — `5,327 -> 2,646`, bit-exact, every figure artifact-backed |
> | `>= 4x` brokered soak | **NOT DONE, and now known to FAIL at `2.23x`.** The `3 x 100` soak was not run: a single official read already shows the bar is not met, and three runs cannot rescue a worst-case that fails on one |
> | canary presented | **NOT DONE, correctly.** Gated on the bar above |
> | both arm instruments built, first numbers on record | **PARTIAL.** Arm M built, ladder exact, `G-M-DEPTH` measured and FAILED. **Arm L not started** |
> | LOG / CURRENT_STATE current with the next gate | **DONE** |
>
> ## The three questions that are not the agent's to answer
>
> 1. **Is `>= 4x` still the right bar**, given it was calibrated on local noise
>    of `~2.4x` and the authority shows `1.20x` across 100 rows with `0/100`
>    failures? At `2.23x` the file is `2.2x` inside a cliff whose measured
>    dispersion is `1.2x`.
> 2. **Is Arm M worth a cost case** for a rank-`r <= 192` sketch of THREE
>    generations recovering `~73%` of `E21` only (and `E21` is not the whole
>    error — `E3` carries ~40%)?
> 3. **Where does the remaining time go** — into the reassociating op-count work
>    that is on the critical path to a nomination, or into Arm L, which is
>    upside on a file that already scores `4.63x` better than the shipped one?


> **2026-09-03 (LATE) — WHERE THIS STANDS, AND THE ONE NUMBER THAT DECIDES THE
> NEXT MOVE.**
>
> **Blocker A is CLOSED.** `NOT_DONE` step 6 ran, the port was wrong, the cause
> was one line, and `SKEPTIC WP16` returned **ACCEPT** — see the block below.
>
> **Blocker B is half done and its verdict is pending.** The operation count is
> `5,327 -> 2,646` per MLP, every float bit-identical, every FLOP figure now
> artifact-backed. **The residual has not been measured on anything that counts.**
>
> | | value | artifact |
> |---|---:|---|
> | per steady-state layer, f32, `n = 1024` | `10,813,796,920` | `layer_price_wp17.json` |
> | whole flow, 16 layers | `168,629,787,238` | `candidate_parity_1024x16.json` |
> | `x` budget / multiplier | `0.076684` / `0.10000` | same |
> | **flopscope operations** | **`2,646`** — lab and shipped candidate IDENTICAL | same |
> | end-to-end drift vs upstream, f64 | `1.692e-11` worst, 8/8 | `parity_1024x16.json` |
> | raw MSE vs minted truth | `8.5075e-07` (ratio `1.000x`) | `clause2_nosub.json` |
>
> **THE OPEN QUESTION, AND IT IS THE ONLY ONE THAT MATTERS THIS WEEK.**
> The retired canary had `1.21x` official residual headroom at `5,327`
> operations. The soak requires **`>= 4x` on the worst run**. If the residual
> were purely per-operation, `2,646 / 5,327` would buy about `2.4x` — **not
> enough**, and the remaining *exact* reductions are close to exhausted at ~165
> operations a layer against a `~120` target. **A pre-registered official read is
> queued** (`experiments/LOG.md`, "PRE-REGISTRATION: the official read"), with
> v2.6.1 as a matched control in the same split, band `[7.5e-07, 1.15e-06]`.
>
> **That run measures commit `ae44369` (`2,708` operations), not HEAD.** Every
> change since is bit-exact, so its **accuracy reading is identical** to the
> current file's, and its **residual reading is a conservative upper bound** —
> the current file is `62` operations cheaper. Stated here so the number is never
> quoted as HEAD's.
>
> **NEXT, in order.**
> 1. Read the official run: `n_failed_mlps`, worst per-MLP residual, raw MSE.
> 2. If headroom `>= 4x`: Linux memory gate -> `certify_candidate` -> brokered
>    `3 x 100`-MLP soak -> present the canary for **Samuel** to submit. ~~Never
>    `whest submit`~~ *(superseded 2026-09-05: an agent may, on his per-submission
>    approval — `AGENTS.md` hard rule 6)*, never designate.
> 3. If not: the next reductions require **reassociation** (factoring
>    `0.5 C11^2 X + 0.25 D22 X`, batching the 13 two-dimensional terms through a
>    3-D stack). Those are not identities, so each needs a measured drift check
>    against the golden reference and a note in the LOG — and the 3-D stack
>    materialises `54 MB` at `n = 1024`, which forum 18101 says costs residual.
>    **Deliberately left undone** so far: a change that is merely "small" needs a
>    measurement, a change that is an identity needs only a comparison.
> 4. `G-M-DEPTH` at `1024 x 16` is queued (16 `P01` nets + an f64 control). The
>    instrument is built and its ladder is exact; a smoke at the wrong shape gave
>    `k=1 0.41`, `k=2 0.66`, `k=3 0.82`, so **`k = 2` sits just under the `0.70`
>    gate** and Arm M may be killed by its own contract.
> 5. Arm L's instrument is **not built**. It needs `>= 128` minted networks on
>    `P05`, a permutation-equivariant basis of `<= 50` coefficients, and
>    leave-networks-out. Nothing about it has been started.
>
> **THE CHANGE GATE IS RED, AND FOR A KNOWN REASON**: `p2_wp15_depth.py` claimed
> `P01[..176..191]` naming `gm_depth.json`, and that artifact is still in the
> queue. Steps 7 and 9 go green the moment it lands and is `finalize`d. This is
> the claim-then-finalize protocol working, not a defect — but it is also the
> exact state that stayed red for nine commits in August, so it is written here
> rather than left to be rediscovered.

> **2026-09-03 — SKEPTIC WP16: ACCEPT. The `3.31x` was one line, and the Skeptic
> proved more than the slice did.**
>
> It re-derived `H1` from `factor_k3.py` by hand, and then went past it:
> **the FLOP delta reconciles in closed form** (`sub - nosub` is exactly
> `9n^2 + 4n` per layer, which at `n = 1024` is the claimed `9,441,280` to the
> FLOP, with an op delta of exactly `13`/layer and `56 - 49 = 7` audited arrays);
> **the reference is bitwise genuine upstream** and observing it through the
> caching accessor does not perturb it; and **the minted truth is not circular** —
> checked against the layer-1 closed form `||W1[:,j]| / sqrt(2 pi)` at rms-z
> `0.93-1.05`, which is the only check anyone has that is independent of *both*
> programs.
>
> **It also found one real gate defect and ten stale numbers, all now fixed.**
> `p2_k3_candidate_parity` gated counted FLOPs *exactly* and merely **printed**
> the operation count — it gated the axis `lambda = 0` makes free and left the
> axis that fails MLPs unguarded. It is a gate now, and running it at the live
> shape immediately exposed that its slack of `16` was exactly the delta at depth
> 16, because the weight `astype` was metered once per **layer**. Guarded by a
> Python `dtype` comparison; delta now `+0`; slack `4`.
>
> **Two corrections to my own numbers.** Clause 2 is **nearly redundant** with
> clause 1 — the torch dump scores `8.507508e-07` against the same truth, so
> `1.000x` is arithmetically forced once clause 1 passes, and I presented it as
> independent corroboration. And **`3.314x` is a cross-sample ratio** (20 graded
> rows over 8 offline networks); the paired figure is `3.161x`.
>
> **And a process hazard that is mine.** The Skeptic measured five different op
> counts for one command before realising I was **committing into the tree while
> it audited**. A `workerq` job it launched picked up a half-optimised snapshot
> and correctly FAILED the price check. *Any measurement taken on this repository
> without pinning a commit is unattributable*, and launching a Skeptic and then
> continuing to commit is a defect in how the slice was run.
>
> **What it could not settle, in its words:** whether the fix delivers
> `8.5075e-07` **on graded data**. The transfer inference is `~8.9-9.3e-07` from
> two estimators on 8 against 20 networks with no interval. And neither of us can
> rule out a defect shared by upstream, `run_flow` and `build_mlp` — golden parity
> is blind to it by construction, and clause 2's band came from the same program.
> The layer-1 closed form bounds that risk **at layer 1 only**. The queued
> official read is what would bound it at every layer.


> **2026-09-03 — THE `3.31x` WAS ONE LINE. `NOT_DONE` STEP 6 IS CLOSED, AND
> G-PARITY PASSES BOTH CLAUSES.**
> `CONTRACT WP16` (`7169392`, committed before any data existed), results
> `1e4d0f8`. Eight `P01` networks, f64, `1024 x 16`, against upstream rev
> `93d091a4`'s own regenerating flow.
>
> | | |
> |---|---|
> | **clause 1** — end-to-end drift vs the torch reference | **`1.692e-11`** worst, **8/8 nets**, gate `1e-2`. **PASS** |
> | **clause 2** — raw MSE vs already-minted truth | **`8.5075e-07`**, band `[7.0e-07, 1.1e-06]`. **PASS** — the WP14 C0-i figure to 4 s.f. |
> | the defect, kept as a named control (`--regen-mode sub`) | **`2.6890e-06`** = `3.161x`, where the grader measured `3.314x` on twenty other rows |
>
> **The defect.** `k3_layer` regenerated `M` from `K[(3,)] - p111_3` and
> `K[(2,1)] - p111_21`. Upstream subtracts `pK_111.get_repeated()` from the
> **stored DS part only**, then returns `K_out[3] = pK_111 + from_dstensor(K_ds[3])`
> — and `FactoredTensor.__add__` **discards the repeated cache** (its own TODO
> says so), so the regeneration hook's `get_repeated()` recomputes from the
> concatenated factors and recovers `K_ds3` **un-subtracted**. Measured at layer
> 0: `M(sub) - M(nosub)` is exactly `3 p111_21^T`, ratio **`1.000000`**, diagonal
> untouched because `p111_3` is exactly zero there. *That* is why `--layer-k2`,
> which only ever read the `(3,)` slice, passed throughout.
>
> **THE GATE THE BRIEF SPECIFIED WOULD HAVE PASSED THE BUG.** The defect's
> end-to-end drift is **`1.3046e-03` worst** (`1.064e-03` best) at `1024 x 16`
> and `2.6e-03` worst at `128 x 4` —
> it clears a `<= 1%` drift gate at both shapes. **Only clause 2, against minted
> truth, catches it.** A `0.12%` drift in the means is a `3.2x` miss in the
> score, because the means are `O(1)` and the error being chased is `O(1e-3)`
> RMS. **Relative drift in the means is not a proxy for score**, and any future
> gate written on drift alone is a gate that cannot see a `3x` regression.
>
> **THE STRUCTURAL FINDING: in a regenerating flow the whole `(1,1,1)` block is
> DEAD WORK.** Its all-distinct part is discarded by `from_dstensor` at the
> rebuild; its repeated part cancels exactly. `p111_3`, `p111_21`, `fac1`,
> `fac3`, `w1_sq` compute nothing the answer depends on. This sharpens rather
> than changes `CONTRACT WP15`'s premise: the regenerating flow's error *is* the
> discarded block, stated exactly.
>
> **PRECISION WAS NEVER THE EXPLANATION.** The fixed port at **f32** sits
> `4.1e-07` from the f64 reference and its raw MSE is `8.5072e-07` against f64's
> `8.5075e-07` — `0.004%`. Shipping f32 costs nothing in accuracy, which is the
> assumption the whole `0.1`-floor plan rests on.
>
> **THE PRICE, RE-METERED. EVERY EARLIER FIGURE IS SUPERSEDED.**
>
> | | before | after |
> |---|---:|---:|
> | one layer, f32, `n = 1024` | `10,837,935,618` | **`10,813,796,920`** |
> | whole flow, 16 layers | `173,381,273,195` | **`173,230,212,715`** |
> | `x` budget / multiplier | `0.078845` / `0.10000` | **`0.078776`** / **`0.10000`** |
> | flopscope **operations** | `5,327` | **`5,119`** (`320`/layer) |
>
> We are still at the floor. **The residual is still the blocker**: `3.9%` fewer
> operations is nowhere near the `>= 4x` headroom the soak requires, against the
> `1.21x` the deployment authority measured on the now-retired file.
>
> **THE MUTATION BATTERY: `0/7` -> `9/9` ON THE LIVE PATH.** A second detector
> that needs **no torch at run time** — a 46 KB committed, hashed golden dump of
> upstream's own answer at `128 x 4` (`golden_ref_128x4.npz`) — plus five new
> live-path mutations. `--layer-k2` alone still kills only `1/12`. The three
> survivors are exactly the three editing the dead `sub` branch, and each reports
> drift `3.617e-12`, the f64 baseline: not a gap, *proof the branch is inert*.
>
> **WHAT IS RESTORED, AND WHAT IS NOT.** Every accuracy claim resting on
> `8.5075e-07` is restored, including WP14 C0-i's paired `1.637x`-better-than-v2.7
> comparison and the `334x` two-slice oracle ceiling, whose PROVISIONAL was
> conditional on exactly this. **Nothing has been re-graded**: the canary that
> scored `2.8192e-06` is **RETIRED**, not amended, and no board number exists for
> the fixed file. `estimator.py` is untouched and remains v2.6.1. Nothing is
> submitted, nothing is designated, and the `mini` split was not re-read.
>
> **NEXT, unchanged in order: Blocker B.** `5,119` operations must come down
> toward `~1,920` (`~120`/layer): fuse elementwise chains, hoist MLP-independent
> constants into `setup()`, stack per-layer matmuls on concatenated operands,
> avoid materialising large temporaries, skip the `(2,1)` slice at layer 15.
> Then re-meter (never adjust), brokered `3 x 100`-MLP soak at `>= 4x` worst-run
> headroom, Linux memory gate, `certify_candidate`, and only then present a
> canary for Samuel to submit. Then WP15's two arms.


> *(**SUPERSEDED IN ITS DIAGNOSIS, NOT IN ITS MEASUREMENTS**, by the WP16 block
> above. Everything this block MEASURED stands — the grader numbers, the FLOP
> parity, `0/20` failed, the `1.21x` headroom, the `1.19x`-slower finding. What
> it left open, the `3.3x`, is answered: one line in `k3_layer`. The `1.21x`
> headroom and the `2.8192e-06` belong to a file that is now RETIRED, so both
> must be re-measured on the fixed file before either is quoted again.)*
>
> **2026-09-02 (LATE) — THE CANARY RAN ON THE DEPLOYMENT AUTHORITY. ZERO FAILED
> MLPs, `1.21x` RESIDUAL HEADROOM, FLOP PARITY TO THE FLOP — AND THE ACCURACY IS
> `3.3x` OFF ITS OWN PREDICTION.**
> `gpuq` 145/148, 20 rows of the public `v2-phase2 mini` split,
> `--runner subprocess`, matched control in the same session.
> **A CONFIRM read: nothing was fitted, nothing selected, no seed block drawn,
> and nothing may now be tuned against those 20 rows.** `estimator.py` is
> untouched and remains v2.6.1; nothing was submitted.
>
> | on the same 20 official rows | raw MSE | adjusted | FLOPs | worst residual | headroom | failed |
> |---|---:|---:|---:|---:|---:|---:|
> | v2.6.1 (shipped), control | `4.1805e-06` | `4.1805e-07` | `137,600,326,656` | `0.0590 s` | `6.78x` | 0/20 |
> | **K3 canary** | **`2.8192e-06`** | **`2.8192e-07`** | `173,381,273,195` | `0.3314 s` | **`1.21x`** | **0/20** |
>
> **THE RESIDUAL QUESTION IS ANSWERED: it does not fail MLPs**, so WP15's
> end-to-end gates are measurable. **But `1.21x` is not a margin** — the soak
> requirement is `>= 4x`, `VALIDATION.md` §2b-ii puts run-to-run spread at
> `~2.4x`, and `n = 20` is thin for a maximum. **Not nominable as it stands.**
> The `~140` removable operations per layer (`p2_k3_fnp.OP_COUNT_NOTES`) are now
> on the critical path, not a tidy-up.
>
> **AND THE LOCAL READING WAS OPTIMISTIC.** Local worst `0.2774 s` against
> official `0.3314 s`: **the grader is `1.19x` SLOWER for this file, not faster.**
> The local RATIO over-predicted too — `10.02x` locally, `5.62x` officially —
> because the subprocess runner costs the small K2 file proportionally more.
> The `~7x`-faster observation from two graded K2 reports transfers to nothing.
>
> **FLOP PARITY IS EXACT.** `mean_effective_compute = 173,381,273,195`, identical
> to the local meter; utilization `0.0788446747`. Every FLOP claim made today is
> confirmed by the deployment authority.
>
> **THE `3.3x` ACCURACY MISS, AND THE CONTROL MAKES IT A DIAGNOSIS.** WP14 C0-i
> put the regenerating flow's raw MSE at `8.5075e-07`; the grader says
> `2.8192e-06`. The control transfers to **`3.2%`** (`4.1805e-06` measured
> against the `4.050566e-06` this repository carries for the shipped K2p1), so
> the split, the networks and the instrument are all fine.
> **The discrepancy is in the K3 flow itself.**
>
> > **`NOT_DONE` step 6 — parity against upstream at non-zero `kappa_3` — stops
> > being hygiene and becomes the explanation for a `3.3x` gap.** `SKEPTIC
> > WP14-FOLLOWUP` named it: *"the priced program and the scored program have
> > never been shown equal"*, and its mutation battery had already shown that no
> > existing check could tell.
>
> **Every adjusted-score claim resting on `8.5075e-07` is therefore PROVISIONAL,
> including the `1.637x`-better-than-v2.7 paired comparison.** What can be said
> is narrow and true: **the canary beats the shipped uncalibrated file by
> `1.483x` on 20 official rows.** The v2.7 comparison must NOT be re-derived from
> these rows — v2.7 is calibrated and both files here are not.
>
> **NEXT, in order: (1)** `NOT_DONE` step 6 against upstream in the GPU venv —
> this is the `3.3x`; **(2)** the op-count reduction, then re-measure residual on
> the authority, with `--mutation-coverage` and `p2_k3_candidate_parity.py` still
> passing bit for bit; **(3)** only then WP15's two arms.

> **2026-09-02 (SKEPTIC WP14-FOLLOWUP — CLAIM A *PROVISIONAL*) — THE FLOP HALF
> SURVIVES EXECUTION AT `0.07886 B`. THE HALF THAT CAN ACTUALLY FAIL AN MLP WAS
> NOT MEASURED, AND IT IS CLOSE TO THE CLIFF.**
>
> > **λ = 0 MAKES FLOPs FREE BELOW THE FLOOR. I SPENT THE DAY MEASURING THE FREE
> > AXIS.** The flow issues **~5,300 flopscope operations per MLP** against the
> > shipped K2 closure's **453**, and its local in-process residual reads
> > `0.29`–`0.44 s` against the **`0.4 s` per-MLP cliff — over it on 2 of 6
> > readings**. **Reproduced in-tree on a `gpuq`-brokered quiet host**
> > (`flow_price_1024.json`, job 144, 3 reps, real canonical weights):
> >
> > | | counted FLOPs | flopscope **operations** | worst LOCAL residual | headroom vs `0.4 s` |
> > |---|---:|---:|---:|---:|
> > | shipped K2 closure | `68,782,549,504` | **`453`** | `0.0277 s` | **`14.4x`** |
> > | regenerating K3 flow | `173,381,273,195` | **`5,327`** | `0.2774 s` | **`1.4x`** |
> > | ratio | `2.52` | **`11.76`** | **`10.02`** | |
> >
> > It did not cross the cliff on that quiet run. **`1.4x` headroom is where the
> > shipped file has `14.4x`, and the standing soak requirement is `>= 4x` on the
> > worst run.** `GOTCHAS.md` already said why: residual tracks **op count and
> > Python glue, not FLOPs**. A residual excursion does not cost score, it
> > **fails the MLP and zeroes the prediction** — a `1e-7` row becomes `~0.92`.
> > The transferable form is the ratio to a matched control in the same process:
> > **`10`–`16x` on residual, `~11.6x` on operations.** `--flow-price` measures
> > all three together now and refuses to quote the absolute second.
> > **This is upstream of every WP15 gate**: a flow that fails MLPs makes
> > `G-M-E2E` and `G-L-E2E` unmeasurable. Next action, and it needs no seed
> > block: wrap the flow as a candidate and read the OFFICIAL per-MLP residual
> > and `n_failed_mlps` from `whest run --runner subprocess` on the `mini` split.
>
> **AND THE THREE CORRECTNESS CHECKS KILL NOTHING.** Seven targeted
> value-destroying mutations — the `(2,1)` coefficient, a dropped block
> permutation (**one matmul cheaper**), the `w(2)/w(1)` asymmetry that is the
> *exact* defect this module's docstring warns about, the rebuild's factor 3, the
> whole `(1,1,1)` block WP15 rests on, a sign, and the `(3,)` slice — **all seven
> pass all three checks identically. 0 of 7 killed**
> (`mutation_coverage.json`). At `M = 0` we have `P = Wᵀ M = 0`, so the two
> `(2,1)`-slice matmuls **execute on zeros** and the off-diagonal of `M_out` is
> never read: **the degenerate limit exercises the cheap half and prices the
> expensive half.** A `4.046`-matmul mutant (`0.06322 B`) passes everything the
> `5.047`-matmul original passes, so *"the price is a price of something"* was
> true of the checks and false of the program. Parity against upstream at
> non-zero `κ₃` (`NOT_DONE` step 6) is still the only thing that closes it.
>
> **THE FLOP MEASUREMENT ITSELF STANDS, AND ALMOST EVERYTHING ELSE SURVIVED THE
> ATTACK** — all three lean-but-exact choices re-derived from upstream by hand,
> all 16 Wick pairs to `≤ 1.3e-16`, `proj_coef` at `n = 4…4096` (the port is the
> *more* accurate of the two), all four frozen tables byte-identical on live
> re-enumeration, and the `(1,1,1)` block algebra confirmed — **so the mechanism
> `CONTRACT WP15` rests on is correct.**
> One K3 layer at `1024`, f32, executed as ONE program in `fnp` inside a real
> `BudgetContext`, is **`10,837,935,618` counted FLOPs = `5.047` dense f32
> `1024`-matmuls**. Sixteen of them are `173,406,969,888` = **`0.07886 B`**,
> against a floor of `0.1` — **headroom `46,495,355,667` = `0.02114 B` =
> `1.35` matmuls per layer.** The dtype audit checked **56 arrays, 0
> violations**, and recorded both `flops.stats.*` calls returning `float64` from
> a `float32` input and being cast back.
> `experiments/2026-09-02_wp14_fnp/layer_price_1024.json`.
>
> **CORRECTED: `f32 / f64` is `0.5000036376`, not "exactly `0.50000`", and I had
> the inference backwards.** The `78,848`-FLOP excess over half is *precisely*
> the two `flops.stats.*` calls, which are computed in `float64` in **both** arms
> and therefore billed at rate `2.0` in both and do not halve. **A ratio slightly
> ABOVE `0.5` is the signature OF the promotion**, not proof of its absence; I
> quoted a print-rounded figure as evidence for the opposite of what it shows.
> The 56-array audit is what actually carries "nothing promoted".
>
> *(**CLOSED 2026-09-03**: `NOT_DONE` step 6 ran, the numbers were NOT right,
> and the fix is the block at the top of this file. `SKEPTIC WP16` returned
> ACCEPT on the parity claim. Everything below is the state as it stood before
> that, kept because the reasoning is what led to the check.)*
>
> **`SKEPTIC WP14`'s PROVISIONAL on CLAIM A is discharged for the per-layer FLOP
> arithmetic and for nothing else — and `SKEPTIC WP14-FOLLOWUP` re-issued
> PROVISIONAL on the axis that decides.** `whest validate`, the Linux memory
> gate, an OFFICIAL residual reading and an end-to-end candidate are all still
> ahead, and the layer's NUMBERS were then unchecked against upstream at non-zero
> `kappa_3` (`NOT_DONE` step 6).
>
> **The projection was 3.3% optimistic, and the missing `0.157` matmuls/layer
> have a name.** `5.047` measured against `4.89` projected (`2.00` K2 `+ 2.89`
> K3) is the `O(n^2)` work that neither separately-measured half could charge
> for: the 25 `pK` terms, the 16 Wick coefficients, `pK -> K`, the `d = 4`
> harmonic projection, the `M = diag(S3) + 3 S21^T` rebuild and the diagonal
> masks. The review estimated that residue structurally at `0.13`. **It is
> `0.157`.** The margin narrows from `1.51` to `1.35`, and the oracle rank-512
> correction (`0.0073 B`) now fits `2.9x` over rather than `3.2x`.
>
> **THE PRICE IS A PRICE OF SOMETHING** — `p2_k3_fnp.py --layer-k2`, three
> checks at `kappa_3 = kappa_4 = 0`, where the answer is known in closed form:
> mean and variance against the shipped K2 closure (`0.00` / `4.9e-16`);
> covariance against K2 **plus the `k_max=3` edge term K2 does not have**
> (`5.1e-16`, and the check requires that term to be PRESENT — it is `5.49e-03`
> of the K2 covariance, and a layer reproducing bare K2 would be wrong); and the
> third-cumulant diagonal against independent Gauss-Legendre quadrature
> (**`1.07e-14`**).
>
> **THE MECHANISM WP15 RESTS ON IS NOW WRITTEN DOWN, NOT ARGUED.** Each layer
> the nonlinearity generates a fresh all-distinct `kappa_3` block as a rank-`n`
> factored tensor whose **middle leg is a multiple of the identity**, so its
> entire contribution to the next `(2,1)` slice is
> **`w(1)_i^2 w(2)_j C_ij^2`** — a Hadamard of the covariance with itself under
> two diagonal scalings, with **no `n^3` object in it**. The regenerating flow
> keeps it for one layer and discards the factored block at the rebuild. *That
> is the `E21` it is missing*, and it is why a MAP can exist where a selector
> cannot: a selector can only choose a subspace of something the flow already
> holds, and the flow does not hold this at all.
>
> **`K[4]` in this flow is ONE SCALAR per layer.** `r_x(4, 3, SIMPLE) = 2` and
> `4 - 2*2 = 0`, so the fourth cumulant's core is order zero; the `(3,1)` and
> `(2,1,1)` slices contribute exactly nothing to the `r = 2` projection, so the
> `n^3` object upstream materialises is never formed. Any design that budgeted
> for a fourth-order `n^2` or `n^3` object was budgeting for something that does
> not exist here.
>
> **THE PORT IS BOUNDED AND NOW HAS ITS COEFFICIENTS.**
> `local_sim/p2_k3_coefs.py` freezes, from upstream rev `93d091a4`, the four
> tables that were the open-ended part: the 25-term program with its scalars and
> per-axis Wick indices, `DS_part_sum`'s 19 vector partitions with their folded
> `pK -> K` coefficients, the two `d = 4` Laplacian coefficients, and the
> `_relu_wick_poly` polynomials for the 16 `(k, p)` pairs. `proj_coef(n, 4, 2)`
> is deliberately NOT frozen — it is a closed form in `n` and is evaluated.
>
> **THE `x 16` IS STILL AN UPPER BOUND, FOR ONE REASON AND NOT THE THREE I
> GAVE.** Layer 0 costs `10,837,936,642` — *more* than steady state, because
> nothing branches on `cov = I` or `M = 0` — and the last layer saves
> `2,098,176` FLOPs (`0.001` matmul), since `last_layer=True` skips only the `M`
> rebuild and **not** the two `(2,1)`-slice matmuls, which the `(2,1)` `pK` terms
> need at every layer. Only the `off_mask` amortisation was real. The real
> 16-layer flow measures `173,381,273,195` = `0.07884 B`, `25.7 M` FLOPs below
> the extrapolation, and `--flow-price` now runs it rather than extrapolating.
> **Confirmed in-tree: `173,381,273,195` = `0.078845 B`, `25,696,693` FLOPs below
> the extrapolation, to the FLOP of the review's independent build.**
>
> **TWO DEFECTS IN THIS SESSION'S OWN INSTRUMENT, FOUND BEFORE IT WAS
> BELIEVED.** (1) A **NaN reference passed a `>` gate silently** — `hermgauss`
> overflows past degree ~200 and `nan > tol` is `False`, so all three quadrature
> checks printed `nan` and the verdict printed PASS. *Second time this class has
> appeared here*; WP8's soak sorted never-executed rows to the bottom of a
> median. (2) **Gauss-Hermite is the wrong rule for a ReLU expectation** and it
> failed as the REFERENCE: the kink makes it converge algebraically, and at
> degree 160 it disagreed with the exact `sigma phi + mu Phi` by `1.1e-03` — an
> error that reads exactly like a bug in the port.
>
> **AND THE CHANGE GATE HAD BEEN RED SINCE `aa01fdd`, THROUGH NINE COMMITS —
> AND FOR THE FIRST TWO OF THEM SOMETHING *WAS* WRONG WITH THE LEDGER.**
> *(Corrected by `SKEPTIC WP14-FOLLOWUP`, which reconstructed the window from
> git rather than taking my word: I first wrote "since `3b38796`, four commits,
> nothing wrong with the ledger" and all three numbers were wrong. `3b38796`'s
> committed ledger passes both checks. Step 9 was red for nine commits. Step 7's
> red was a **different** defect — a truth artifact claimed with no hash, exactly
> the fail-open `SKEPTIC WP13` made us close — which lasted two commits and
> **self-healed** at `3084a7c` when `finalize()` was recorded, i.e. it was fixed
> by accident rather than by anyone reading the gate.)*
> WP14 C0-i recorded a deliberate seed
> overlap in its prose `note` and left **`paired_with` — the field the
> enforcement layer reads — empty**, and the check that caught it was a *count*
> of a growing record rather than a property. Repaired: `seed_ledger` gains a
> `"pair"` record kind (append-only; `declare_pair()` refuses to license an
> overlap that does not exist), the overlap report prints the declaring record's
> **own reason** instead of one of two canned tags (the P01 case is **both**
> deliberate and correlated, and a canned "one experiment, two roles" would have
> erased the warning), and `A1` becomes a property with a named accident list.
> `66/66`. **This repair LOOSENS an invariant and that must be said**: the rule
> moves from *"a deliberate overlap must be pre-declared"* to *"…must be
> acknowledged, possibly afterwards"*, and `declare_pair` cannot distinguish
> intent from retroactive rationalisation. **Nothing enforces "preflight was
> green" at commit time — that hole is real, is NOT repaired, and is recorded as
> open.**
>
> **`P05`/`P06` COULD NOT HAVE BEEN CLAIMED.** `1b6ac04` registered them in
> `seed_blocks.py` and not in `seed_ledger.TIERS`, so `block_span` worked and
> `tier_of` raised — **WP15's first instrument run would have died on its first
> claim.** Repaired, and the two registries are now asserted to agree in both
> directions. Likewise `k3_coefs_simple.json` was load-bearing with **no hash and
> no re-enumeration check**, unlike the term table beside it; it now has both,
> and its generator refuses to overwrite the committed path.
>
> **NEXT — `CONTRACT WP15` is pre-registered and committed (`1b6ac04`), before
> any data and before `P05`/`P06` are touched.** Two arms on the one remaining
> question, *is there a map from `(mu, Sigma, S3, S21)` to the rank-`r`
> correction of BOTH read-slices*: **Arm M** measures `E21`'s memory depth by
> age-selected exact injection and, if short, builds the closed-form generator
> plus a rank-`r` sketch; **Arm L** fits a permutation-equivariant
> polynomial-in-matrices map (<= 50 coefficients) on `P05`, leave-networks-out,
> and validates end to end in flight on `P06`. Seven gates, all thresholds fixed
> in the contract; twelve predictions, **five of which predict a gate in the
> contract failing**. Two disagreements with the strategist recorded before the
> data: **rank `512` is not affordable for a carried sketch** (`3.0`
> matmuls/layer against `1.35`; the horizon is `r <= 192`), and the `>= 90%`
> held-out `R^2` gate is not the binding constraint — the end-to-end requirement
> will bite first. Kill line **~2026-09-20**.


> **2026-09-02 (WP14 — CLOSED) — WE ARE AT THE MULTIPLIER FLOOR, AND WE GOT
> THERE BY FIXING A UNIT. THE FREE SELECTOR DOES NOT EXIST: `Sigma`'s DOMINANT
> RANGE CONTAINS `E21` EXACTLY AS MUCH AS A RANDOM SUBSPACE DOES.**
> Contract `d915f83` (twelve predictions, three of them predicting a gate in the
> contract failing, committed before any data); results `9b36521`. Six held,
> three refuted, two not run.
>
> **THE COST HALF IS SOLVED ON ARITHMETIC AND NOT YET ON EXECUTION** — see the
> PROJECTION paragraph below, and `SKEPTIC WP14`, which rated this claim
> PROVISIONAL for exactly that reason. flopscope does not bill a matmul at `4 n^3`. It
> bills the textbook `2 n^3` and scales every operation by a **per-dtype rate**
> from its own packaged table — `float64 = 2.0`, `float32 = 1.0`. WP13 built its
> probe arrays with a generator that returns f64, read that column, and wrote it
> down as a property of matmul; every cost statement in that slice is therefore a
> **float64** statement, quoted against accuracy evidence gathered in **float32**.
> Re-executed at f32 (not halved on paper — `np.float64(3.0)` scalars are
> *concrete* under NEP-50 and would have promoted the arm straight back):
>
> | composition | K2 backbone | K3 increment | total | `x budget` | multiplier |
> |---|---:|---:|---:|---:|---:|
> | all f64 — WP13's record | `137,600,326,656` | `198,233,321,472` | `335,833,648,128` | `0.15272` | `0.15272` |
> | **both f32** | `68,802,510,336` | `99,116,660,736` | **`167,919,171,072`** | **`0.07636`** | **`0.10000`** |
>
> **A regenerating full-K3 flow reaches the `0.1` floor with `0.02364 B`
> (`51,983,154,483` FLOPs, `24.2` f32 matmuls, `1.51` per layer over 16) to
> spare — no Strassen, no rank cap.** Both halves were already measured and independently reviewed in this
> repository; the f32 K2 figure had sat in this very file for six days. **Nobody
> had added them.** Strassen is no longer needed for the floor and `L = 4`'s
> `14,001` array ops/MLP stays what it always was: a table entry, not an
> operating point. **This is a PROJECTION** — two spellings that have never been
> executed as one program. `local_sim/p2_k3_fnp.py` exists to make it a
> measurement.
>
> **THE ACCURACY HALF IS NOT, AND BOTH CHEAP BRIDGES ARE NOW MEASURED DEAD.**
> `G-CONTAIN` FAILS at every gated layer and its falsifier FIRED. Containment of
> `E21`'s `r90`-energy by `Q_l` is `r/n` **to three decimals** at every depth and
> rank — `0.127 / 0.252 / 0.501 / 0.751` at `r = 128/256/512/768` — matching the
> random-orthonormal null (`t`-interval on the difference straddles zero from
> `r = 256` up) and the cross-network null, and falling **below** the null at
> three cells. `G-E2E` fails too, and its own `< 2x` falsifier fires: the
> projected correction buys `1.382x` where `5x` was required.
>
> **The suspicious cleanliness was checked before it was believed.** Exactly
> `r/n` is also what a broken range finder returns, and the plausible break was
> real — WP13 measured the rank collapse on the **post-ReLU** covariance while
> this hook builds `Q` from `cov_pre = W^T Sigma W`. Fifteen seconds of CPU
> (`local_sim/p2_cov_spectra.py`) refuted my own suspicion: `r90(cov_pre)` is
> `246 / 64 / 32` against `cov_post`'s `239 / 60 / 30` at layers `1 / 7 / 14`.
> The range finder was fed a genuinely low-rank covariance and found nothing.
>
> > **Two strongly low-rank objects, in the same flow, at the same layers, driven
> > by the same weights — and their subspaces are related no more than chance.**
>
> **WHAT PASSED, AND ONE OF IT CLEARS A STANDING DEBT.** `G-POS` holds at
> `1.08e-13` relative: injecting both true slices reproduces the uncapped flow, so
> the premise the whole lane rests on — **the scored means read `kappa_3` only
> through its two repeated slices** — is now demonstrated end to end at
> `1024 x 16` rather than argued. **`S2-1024` is BIT-IDENTICAL**, discharging the
> null-control debt `SKEPTIC WP13` left standing (it had only ever run at
> `256 x 4`). And **P8 held**: the in-flight injection is monotone in `r` and
> **not destructive**, unlike WP10's variance channel.
>
> **THE THING NOBODY HAD LOOKED AT.** Eleven closed lanes and two contracts all
> chased the `n x n` off-diagonal object. The `(3,)` slice beside it carries
> **~40%** of the regenerating flow's error, is an **`n`-vector**, and correcting
> it alone buys **`1.551x`** at `O(n)` per layer — a vector added to `S3` before
> `M = diag(S3) + 3 S21^T` is rebuilt. It is still an **oracle**: nothing predicts
> it from the flow's own state. It is the cheapest accuracy this project has
> found and it has never had a contract.
>
> **THE FALLBACK IS NOW A NUMBER, NOT A BRACKET.** Truth minted on the eight
> already-claimed `P01` networks (`gpuq` 121, `2^22` x 2 independent replicas,
> cross-product estimator, ledger finalised). **P2 HELD**: raw
> **`8.5075e-07`** against a pre-registered `[7.5e-07, 1.0e-06]`, so adjusted
> **`8.508e-08`**. WP13's `2.4x`-wide bracket collapses. Against v2.7 the
> comparison is now **PAIRED** on common networks (`SKEPTIC WP14` defect 6):
> arithmetic difference `5.4232e-07`, `t` 95% `[3.51e-07, 7.33e-07]`, basic
> bootstrap `[3.86e-07, 6.80e-07]`, **8 of 8 networks**, point **`1.637x`** —
> and the v2.7 proxy used lands within `1.4%` of the graded number it stands
> in for (`1.3931e-06` vs `1.3738e-06`). **The adjusted figure is contingent
> on the cost claim above, which `SKEPTIC WP14` rated PROVISIONAL: at WP13's
> `0.15272 B` the same raw scores `1.30e-07`, i.e. PARITY with v2.7.** GT noise is `1.72e-08`, `2%` of this arm and
> `43%` of full K3's own MSE, so the `uncapped` row (`3.7179e-08`) is a
> diagnostic and not a re-measurement of P-A.
>
> **WHERE WE ACTUALLY STAND.** At the floor the score is `raw / 10`:
>
> | | raw | adjusted | vs Puffi `4.5e-09` |
> |---|---:|---:|---:|
> | full factorized K3 **if it fit the floor** | `3.9e-08` | `3.9e-09` | **`1.15x` better** |
> | full K3 at its own measured f32 price (`1.0723 B`) | `3.9e-08` | `4.2e-08` | `9.3x` worse |
> | **regenerating flow at the floor** | `8.5e-07` | `8.5e-08` | `19x` worse |
> | graded v2.7 (shipped) | `1.37e-06` | `1.37e-07` | `31x` worse |
>
> *(The `1.0723 B` W-EFF price was checked before use and is **already an f32
> measurement** — f64 is `2.1446 B`. It does not halve again and W-EFF stays
> closed.)*
>
> > **Full-K3 accuracy is worth first place and costs `10.7x` the floor. The
> > regenerating flow buys the cost and loses `21x` of the accuracy. WP13 and
> > WP14 have between them closed both cheap bridges on measurements.**
>
> **`C4` (the low-rank-`Sigma` backbone) is NOT RUN, and that is a decision.**
> Its gate `G-FLOOR` required `<= 219,885,548,339` FLOPs; `C0-iii` measured
> `167,919,171,072` before `C4` existed. It was the route to the floor and the
> floor is reached. P11/P12 are recorded **NOT RUN**, not dropped.
>
> **NEXT ACTION — the `fnp` port, and it is now the only thing on the critical
> path.** `local_sim/p2_k3_fnp.py`. The port is **bounded**: upstream's term list
> depends on nothing but `(k_max, d_max, augment)`, so it is enumerated once and
> frozen — **35 terms, 25 kept in SIMPLE mode**, artifact
> `experiments/2026-09-02_wp14_fnp/k3_terms_simple.json`, `--check-terms`
> re-enumerates against the live checkout and refuses on drift. The **K2 half is
> finished and verified at the live shape** (`137,562,575,872` f64 /
> `68,782,549,504` f32, ratio `0.50001`, against the NumPy mirror to `2.2e-15`
> and `5.6e-07`). What remains is listed by name in the module's `NOT_DONE`.
> **Trap recorded there: `p2_slice_cost.py`'s spelling is cost-correct and
> numerically WRONG** (it uses `w1` for both wick factors where upstream uses
> `w(1)` and `w(2) * w(1)`), so it must not be promoted by copying.
>
> **THE CEILING OF THE CORRECTION LANE IS A BOARD-LEADING SCORE, AND IT IS
> AFFORDABLE.** The `e3svd` cell the first `C2` run omitted (`gpuq` 131,
> `c2_e2e_oracle_rank.json`) corrects BOTH slices at an oracle rank `r`:
>
> | oracle correction | drift | implied raw | adjusted at `0.1` | vs Puffi `4.5e-09` |
> |---|---:|---:|---:|---:|
> | none (the measured fallback) | `8.75e-07` | `8.51e-07` | `8.5e-08` | `19x` worse |
> | `e3svd:128` | `7.89e-08` | `~1.19e-07` | `~1.2e-08` | `2.6x` worse |
> | **`e3svd:512`** | `2.62e-09` | `~4.3e-08` | **`~4.3e-09`** | **`1.05x` BETTER** |
>
> Correcting only ONE slice saturates at `2.4x` however good the correction
> (`svd:256` = `2.376x`); correcting both at rank 128 gives `11.1x` and at rank
> 512 gives `333.8x`. A rank-512 correction costs `16,106,127,360` counted f32
> FLOPs (`0.0073 B`) against `0.0236 B` unspent — **it fits `3.2x` over.**
>
> > **The entire remaining problem is a MAP from the flow's own state to
> > `(E3, rank-512 E21)`.** WP13 killed five column-weight selectors; WP14 killed
> > `Sigma`'s dominant range. **Nothing has yet been tried that LEARNS the map**,
> > which was `C3`'s named backup and is now the only live route. It is an ORACLE
> > ceiling, not a result, and it needs its own contract.
>
> The contract's own **`DTYPE-F64` control landed** (`gpuq` 132) and changes
> nothing: containment `0.127 / 0.128 / 0.128` at `r = 128` and `0.500` at
> `r = 512`, identical to f32. The f64 control this repository has never had now
> exists, and `SKEPTIC WP13`'s open defect on that point is discharged.
>
> **Open, and needing a contract before anyone acts on it:** the learned map
> above; the `(3,)`-slice corrector (`1.55x`, `O(n)`, oracle-only today); and
> whether any closed lane was closed on a **float64** cost that a float32
> re-pricing would reopen — the W-EFF screen was checked and is unaffected, the
> rest have not been.

> **2026-09-02 (WP13 — CLOSED, E3 NO-GO) — THE INFORMATION IS LOW RANK,
> COLLAPSING WITH DEPTH, AND AFFORDABLE TO CARRY. NO SELECTOR CAN FIND IT, AND
> EVERY SELECTOR TRIED IS WORSE THAN DOING NOTHING.** The strategist's
> off-diagonal lane is answered on pre-registered numbers. Contract `8b46574`
> (nine predictions, four of them predicting failure, committed before any data);
> results `4561d32`; **`SKEPTIC WP13` PROVISIONAL, accepted in full**, with a
> re-score that **retracted my stated reason for the kill**.
>
> **The cost half is SOLVED, and this part stands.** `FactoredTensor.from_dstensor`
> returns factors `(diag(S3) + 3 S21^T, I, I)` — **two of three legs are the
> identity** — so a flow that *regenerates* `kappa_3` from its own two read-slices
> costs three matmuls a layer, not six. Measured in flopscope at `1024 x 16`:
> **`0.1527 B`**, and **`0.0943 B` at four Strassen levels, which pins the `0.1`
> multiplier floor**. Symmetry tagging buys exactly nothing (ratio `1.0000`).
> **G-COST PASS. G-TOL PASS.**
>
> **`E1` kills the lane, and it is the one result no model can rescue.** At every
> budget `k`, the best of five selectors leaves a `(2, 1)`-slice error **`1.45x`
> to `2.10x` WORSE than not correcting at all**. The slice-targeted weight beats
> the Frobenius weight by `1.0019x` — two parts in a thousand — so **there was no
> wrong norm to fix**: on this factorisation the two are collinear and the
> Dharangutte–Musco thesis has nothing to bite on. Randomised importance sampling
> is `2.48x` **worse** than deterministic truncation (`P6`, written against the
> strategist's Route A before the data, HELD). **G-SEL FAIL.**
>
> **`E2` refuted MY prediction and vindicated the strategist's mechanism.** The
> residual regeneration discards has `r90` falling `264 -> 78` over depth, while
> the covariance's own `r90` falls `237 -> 26`. Against an iid-Gaussian baseline
> of `r90 = 523` it is below isotropic even at layer 1 and `6.7x` below by layer
> 14, monotone on **8 of 8 seeds**. I had predicted `r90(L7)` in `[200, 600]` and
> **growing**; it is `147` and falling.
>
> **RETRACTED, on the Skeptic's finding:** *"it still does not fit in the
> budget."* `G-RANK`'s threshold `r <= 330` came from **upstream's torch-side
> counter — which the contract's own cost-model section forbids scoring a gate
> on.** That is the WP8 failure class repeating in a contract written to avoid it.
> Re-priced in flopscope, an oracle rank-`r` correction costs **one matmul per
> layer** (`S21 + U V^T`), so `r = 128` is still inside the floor and `r = 512`
> costs a `9%` multiplier penalty. At `r = 512` the composed excess is
> `3.92e-09` layer-averaged (**PASS**) and `1.10e-08` worst-layer. **`G-RANK` does
> not robustly fail.** `E3` is NO-GO on **`G-SEL` alone**.
>
> **Standing corrections to the record, all from the Skeptic and all verified:**
> `E0-TOL` ran on **3** networks, not the 8 I wrote (`E1`/`E2` used 8); the
> `8.832e-07` `dsproj` reference is in `2026-08-27_pb_rank/drift_dsproj.json`, NOT
> `2026-08-28_pb_oracle/drift_oracle.json`, which has no such row; the two
> collapses do **not** "track each other layer for layer" — the ratio grows
> `1.11 -> 2.94`; and **`eps = 0` is a determinism check, not a null control**
> (same deterministic call compared with itself). **The standing null-control debt
> is NOT discharged** — that is four consecutive slices — and the load-bearing
> null, S2, has only run at `256 x 4`.
>
> **What is CLOSED:** Route A (importance-sampled slice probing) and Route B's
> selector, both on measurements. **What is OPEN and must not be recorded as
> closed:** the residual's low-rank, depth-collapsing structure is real, new, and
> affordable to exploit *if* it can be found.
>
> **NEXT GATE — the one open question worth a contract.** `E21`'s leading subspace
> versus `Sigma`'s, which the flow already holds for free. It must be
> pre-registered as a **partial containment fraction**, not a yes/no, because
> `r90(E21) / r90(Sigma)` reaching `2.94` proves total containment is impossible
> at depth. One `gpuq` job on the **already-claimed** `P01` seeds; the composition
> rule (worst-layer or layer-averaged) named **before** running.
>
> **The fallback, bracketed and NOT a nomination.** The regenerating flow needs no
> correction to be a candidate: at the floor its raw MSE brackets
> `[5.20e-07, 1.26e-06]` (GT-noise floor `1.155e-09`, so `450x` below the bracket's
> low end), i.e. adjusted `[5.2e-08, 1.26e-07]` against graded v2.7's `1.374e-07`
> — **`1.1x` to `2.6x` better.** The bracket is too wide to justify a multi-day
> `fnp` port, its low end is barely an improvement, and it is a drift bracket, not
> a measurement: **no truth was minted.** Sharpening it costs one truth mint on
> the eight `P01` networks already claimed.
>
> **Standing cost model, and it replaces every hand estimate here:** flopscope
> bills a dense `(n, n) @ (n, n)` matmul at **`4 n^3`**, twice the textbook figure
> — so **at the `0.1` floor an entire `1024 x 16` estimator gets `3.2` dense
> matmuls per layer.** The shipped file spends `2.00`. `docs/GOTCHAS.md` carries
> it, along with the finding that a `gpuq` job submitted with **absolute paths does
> not run its snapshot**.
>
> **Track 2 is live:** `docs/writeup-p2-draft.md`, seven of ten sections
> evidence-complete. `docs/writeup-draft.md` (Phase 1) stays stale and untouched.

> **2026-09-02 — THE SOAK'S FIRST REAL RUN REFUSED THE SHIPPED FILE, AND THE
> INSTRUMENT COULD NOT SAY WHY.** WP6's `--residual-soak` ran for the first
> time, on **v2.6.1 — the exact file the grading server ran with ZERO failures**
> — and reported **45 failed MLPs of 300**, verdict `REFUSED`. Residual is not
> the cause: the worst row was `90.2 ms` against a `400 ms` cap, `4.43x`
> headroom. Two defects, both now repaired under **CONTRACT WP8** (`e7b83c2`,
> committed before the repair):
>
> 1. **The soak counted failures and discarded the reason.** The runner already
>    computes a five-way `failure_breakdown` and stamps every failed row with an
>    `error_code` and a traceback; none of it was read. It now is, including
>    **the deepest failing frame that lies in this repository** — the field that
>    separates "the host fell over" from "the candidate's own math raised".
> 2. **The pooled tail included rows that never executed.** `scoring.py`'s
>    exception path appends a failed row carrying `residual_wall_time_s = 0.0`.
>    Those zeros sort to the bottom, so the reported "median" at rank `150/300`
>    was the **41st percentile of executed work**. `worst` is a max and is
>    immune — which is exactly why the defect survived the number a reader
>    checks. The tail is now over executed rows only, with the contaminated
>    figures kept beside it and an explicit `median_bias_from_failed_rows`.
>
> **Why it survived review:** the arithmetic was fused with the subprocess loop,
> so the only way to exercise it was a three-hour official run — which means in
> practice it was never exercised. `_pool_residual` is now pure and 16 unit
> tests run in milliseconds. **The attributed re-measurement is in flight; until
> it returns, the 45 failures are UNEXPLAINED and no certification that used the
> old soak may be quoted.**
>
> **WP9 is open and pre-registered** (`9c961ce`): a per-neuron,
> scale-equivariant **in-flight state corrector**, gated on **rollout** and not
> on fit. **Its price is already MEASURED and the news is good** (`d004ed1`):
> correcting at *every* layer costs **36,433,980** counted flopscope FLOPs —
> `0.017%` of the free zone — leaving the multiplier pinned at `0.1` with
> `37.42%` of the free zone unused. Gate clause **G2 PASSES** and prediction
> **P6 HELD** with `13.7x` of margin. **Accuracy is now the only open question
> on this lane.** `P02` is being minted for it.
>
> **`p2_truth.py` recorded the post-ReLU variance at the FINAL LAYER ONLY**, so
> it could not have supplied an in-flight variance target at all. Now recorded
> at every layer, and the claim that the existing labels are bitwise unchanged
> is executed rather than asserted: `--check-allvar` ran the pre-WP9
> accumulation beside the current one and returned **bitwise-equal means and
> final-layer variance on both probe networks**.

> **2026-09-02 (WP10 + WP8 closure) — THE PER-NEURON FORK IS RESOLVED, AND THE
> SOAK MYSTERY DOES NOT REPRODUCE.**
>
> **WP8 is CLOSED.** The `3x100` discriminator — the same shape as the run that
> reported `45` failures, but submitted through `gpuq` at the declared `20 GiB`
> footprint — returned **`0` failed MLPs of `300`**, all five breakdown
> categories zero, worst executed residual `69.46 ms`, **`5.76x` headroom**,
> verdict `CERTIFIED`. Under an i.i.d. `15%` rate `P(0 of 300) ≈ 1e-21`, so the
> `45` are **not** a property of the file, the workload or the organisers'
> data. The one controlled difference is that the original does not appear in
> the `gpuq` history at all. **Recorded as environmental, cause unmeasured,
> non-reproducing under brokered submission** — it is circumstantial, not a
> measurement. The contract's *real* P5 (the one the Skeptic caught me swapping
> away) is now **HELD exactly**: counted FLOPs are `137,600,326,656` on all 300
> executed rows. **The standing consequence on G3 is lifted for the shipped
> file on this host, and for nothing else.**
>
> **WP10 CLOSES the in-flight mean lane on a number, not on a failure.** The
> oracle reproduced in-tree on all 96 networks (mean-only **`6.872x`**,
> variance-only `0.059x`, both `0.547x`). The fitted arm diverged `133x` — and
> so did both null controls, firing the contract's VOID clause. So the first
> job was to find out whether the harness leaks. **It does not:**
> `--sensitivity` injects a *known* `c_m`, the positive control at `c = 0`
> reproduces the baseline to `0.000e+00`, and a constant shift **at the fitted
> arm's own clamp** costs `1129x` on its own — arm C's `133x` is comfortably
> *inside* what the clamp permits.
>
> Excess MSE scales as **`c^2` across three decades** in both the coherent and
> incoherent regimes, so **the K2 tower is a linear amplifier of in-flight
> state error** — gain `~18x` coherent, `~4.2x` incoherent over 14 layers,
> ratio `4.3 ≈ sqrt(14)`. That converts the ceiling into an entry price:
>
> > **an in-flight mean corrector must reach held-out `R^2 >= 99.85%` on `c_m`
> > merely to MATCH the shipped file.** Arm C measured `-0.53`; WP9's readout
> > arm reached `+0.50` on an easier target. `(1 - R^2)` must fall `330x`.
>
> **P2-CAL's 2026-08-26 fork — *"act per neuron, or add information"* — is now
> resolved. Per neuron is CLOSED on all three channels, each with a measured
> reason:** the readout is exhausted by v2.7's sixteen constants (its single
> per-layer number predicts `c_m` at `R^2 = +0.51`, matching a 354-parameter
> network); the in-flight **variance** is *destructive* at zero fit error; the
> in-flight **mean** has a real `6.872x` ceiling behind a `99.85%` entry price.
> **Only "add information" remains** — the Price order-1 truncation of the
> post-ReLU **off-diagonal**, which has no label in `p2_truth.py` and needs
> either true covariances or an end-to-end gradient through the rollout.
>
> **A methodological debt, three slices running: every null control I have
> built has failed to do its job.** WP9's N1 shared the divergent channel;
> WP10's C0 turned out to be *the most damaging perturbation in the grid*
> presented as a null. **A null control must be verified to be null by
> measurement before its failure is allowed to mean anything.** The one valid
> null here is `c_m = 0`, pinned by self-test S10 at `0.000e+00`.
>
> **And the cheap lesson worth the most:** converting an oracle ceiling into an
> accuracy requirement costs **one sensitivity sweep** and would have retired
> this lane before `P02` was ever spent.

> **2026-09-02 (WP9/WP10) — A NEGATIVE CONCLUSION WAS RETRACTED, AND THE LANE
> IS OPEN.** WP9 spent `P02` on a per-neuron state corrector and I concluded
> *"the per-neuron half of that fork is empty."* **The sixth independent
> Skeptic pass overturned it** — the first pass this project to overturn a
> *negative*. Full entry at the bottom of `experiments/LOG.md`.
>
> WP9's in-flight arm moved the **mean** and the **variance** channel together,
> diverged `136.92x`, and I killed the family. The review reimplemented the
> closure independently — reproducing this repository's `B0` on **all 96
> networks to `5.4e-13`** — and ran the **oracle** cells I never ran, at
> **zero** fit error, layers 1–14:
>
> | in-flight correction, ZERO fit error | vs the shipped file |
> |---|---:|
> | truth in the **MEAN** channel only | **`6.42x` BETTER** |
> | truth in the **VARIANCE** channel only | `0.060x` — `16x` worse than doing nothing |
> | truth in **both** (WP9's arm B) | `0.56x` |
>
> **The two channels have opposite signs.** The mean channel's ceiling clears
> the `1.5x` gate by `4.3x` and is the one arm the study never ran. My stated
> mechanism is refuted too: I wrote that the variance walk *"amplifies whatever
> it gets wrong"* — it amplifies even when it gets it exactly **right**, and
> the clamp is exonerated to five significant figures.
>
> **New mechanistic fact, and assume it generalises:** correcting a closure's
> marginal variance *toward truth* is **destructive**. The Gaussian closure is
> an internally consistent approximation, not a set of independently improvable
> marginals — making `var` exact while the off-diagonal stays truncated at
> Price order 1 breaks a cancellation.
>
> **What WP9 did establish, confirmed and strengthened:** the **readout**
> corrector is exhausted by v2.7's sixteen constants. v2.7's *single* per-layer
> constant predicts the per-neuron residual at `R^2 = +0.5097`, matching a
> 354-parameter network's `0.499–0.522`. Arm A's `0.9945x` is a **draw**
> (`t` CI `[-1.7238e-08, +1.7762e-09]` straddles zero, sign `p = 0.610`), not
> the refutation the first write-up claimed.
>
> **`P02` WAS SPENT WITH NO LEDGER RECORD**, while the entry said *"ledger
> finalised."* A queued job runs in a frozen worktree and
> `experiments/seed_ledger.jsonl` is **tracked**, so `p2_truth.py`'s
> `claim()`/`finalize()` — called correctly — wrote into the **snapshot** and
> were discarded with it. 96 seeds were really drawn while the live ledger
> reported them FREE. **Backfilled** with the artifact sha256, and made
> mechanical: `seed_ledger._append()` now **refuses** to write a ledger under a
> `gpuq`/`workerq` snapshot.
>
> **`CONTRACT WP10` is open** (`30ce479`, addendum `a8ec186`): arm C, the
> **mean channel only**, in flight at layers 1..L-2, scored on rollout against
> the shipped file, with both null controls now sharing no channel with a
> divergent arm. Its own addendum records that the contract's first formulation
> was internally inconsistent — it put v2.7's gain inside the tower, which
> breaks homogeneity — and that **self-test S10 measured the damage at a
> relative `1.424` before any data was touched.**
>
> **Queued, not run** (`gpuq` 89, 90), behind the WP8 `3x100` soak. Until that
> soak returns, **G3 has nothing behind it but 12 rows** and no candidate may
> be certified on it.

> **2026-09-02 (later) — THE WP8 SKEPTIC RETURNED PROVISIONAL, AND TWO OF ITS
> FINDINGS ARE DEFECTS IN THE RECORD RATHER THAN IN THE CODE.** Fifth
> independent pass this project, fifth to find something real. Accepted in
> full; repairs and every number at the bottom of `experiments/LOG.md`.
>
> - **The attribution half was BROKEN.** `deepest_repo_frame` matched the
>   substrings `"estimator"` or `"arc-whest"` — and **`.venv/` lives inside
>   `C:/Users/samsu/Documents/arc-whest/`**, so every absolute path a real
>   `traceback.format_exc()` emits for a `whestbench` frame contains
>   `arc-whest`. A pure harness fault was reported as a candidate fault,
>   loudly. My guard test passed only because its fixture used a **relative**
>   path, which `format_exc()` never produces. Now `_candidate_frame()` matches
>   the candidate file's own absolute path with `.venv`/`site-packages`
>   segments excluded; three tests cover it.
> - **I SWAPPED A PRE-REGISTERED PREDICTION.** The contract's P5 was *"counted
>   FLOPs on every executed row remain exactly `137,600,326,656`"*; I scored a
>   residual-headroom proposition instead — **WP6's** P5, carried across. The
>   contract's P5 is **UNSCORED**, and the soak stored no per-row `flops_used`
>   so the artifact could not have answered it. It is answerable now
>   (`flops_executed_min/max/constant` plus a check-11 clause).
> - **"It is submitted and pending" was FALSE when committed.** The `1x100`
>   discriminator was cancelled — by me, to unblock WP9 at concurrency 1 — 34
>   minutes before the entry claiming it was pending. Re-queued as **`3x100`**
>   so it matches the original's shape.
> - `worst_all_rows` had been dropped while `median_all_rows` was kept, which
>   is backwards: only the *exception* path zeroes a residual, so excluding
>   failed rows makes `worst` **less** conservative. Also repaired: a
>   `ZeroDivisionError` that would kill the certifier before it wrote its JSON,
>   three `IndexError` sites that turned a mutation into a harness failure, and
>   **U1, which was inverted** and could not detect `whestbench` *adding* a
>   flag. Unit suite **19/19**, up from 16.
>
> **The Skeptic also settled something I had called untestable:** the 45
> failures **must** have been exception-path, by elimination —
> `residual_wall_time_exhausted` would have put a row above `0.4 s` and the
> artifact's worst is `0.09022`; `budget_exhausted` is impossible at a fixed
> `137,600,326,656` against `2**41`; `time_exhausted` needs `> 120 s` and
> leaves a real residual. **Why** they happened is still unmeasured, and the
> `gpuq` store showing an idle queue in that window is weak evidence *against*
> the brokered-neighbour story.
>
> **Binding until the `3x100` returns: WP9's gate clause G3 has nothing behind
> it but 12 rows**, and `deepest_candidate_frame` may not be read as evidence
> about prediction P2 on any artifact produced before this correction.
>
> **WP9 had the same class of defect and it is fixed.** Arm B0 was being scored
> as *uncorrected* K2, where the contract says **shipped calibrated K2
> (v2.7-equivalent)** — which would have credited the corrector with the
> `2.8629x` already on the board. On probe data that made arm A read `2.40x`
> when it is nearer parity with the shipped file. Corrected, and with it a
> structural rule: **the corrector REPLACES the calibration rather than
> stacking on it**, because `mu*g = sigma*g*h(alpha)` is exactly
> `c_m = (g-1)*h(alpha)` and `h(alpha)` is in the corrector's feature span.
> Self-test S9 pins both halves.

> **2026-09-01 — THE RULE-2 GAP IS CLOSED, AND IT OVERTURNED A CLOSED LANE.**
> The pending WP2/WP3/WP4 Skeptic ran in the foreground and returned
> **ACCEPT / PROVISIONAL / REJECT**. That is the fourth of four passes this
> session to find a real defect.
>
> - **WP4's adjoint-K3 KILL is RETRACTED.** Its linearity statistic **kills a
>   provably linear system at every depth** — the arm applies `eps` after every
>   layer, making the tangent deviation a degree-`L` polynomial regardless of
>   physics. The lane is **open again**, unproven, with a broken instrument.
>   Do not cite it as closed.
> - **WP3's PORT is PROVISIONAL.** The `1.03723x` price reproduces exactly and
>   gets *stronger* once its depth confound is corrected, but the contract froze
>   **no residual model** — and AUGMENT costs `4.38x` the op count and `1.26x`
>   the in-process wall. It authorises only the cheapest `fnp` probe.
> - **WP2 is ACCEPTed**, and its negative is *stronger*: at live width Kerdock
>   buys **`0.155%`** over random orthogonal blocks. The carrier stays demoted.
>
> Details, every number, and the repaired-instrument evidence: the last entry in
> `experiments/LOG.md`. The body below is the 2026-08-31 snapshot, corrected in
> place where these verdicts touch it.

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

- P01 has consumed **168/600** seeds; 432 DESIGN seeds remain. **P02 is now
  fully consumed (96/96)**; P03 and P04 remain untouched. The append-only
  seed ledger is authoritative; prose counts are only snapshots.
- P01 is not a DECIDE block and cannot serve as its own independent replication.
- **ALLOCATED 2026-09-01, all three unspent (0 seeds drawn).** The gap below is
  closed; Samuel's sign-off on the *allocation* is waived (solo), the Rule-1
  contract per slice is **not**.

  | block | tier | n | span | purpose |
  |---|---|---:|---|---|
  | **P02** | DESIGN | 96 | `40,901,002,000..095` | **SPENT 2026-09-02** on WP9/WP10 (truth `d21f81ad…`, ledger backfilled). Re-analysable within DESIGN; not independent |
  | **P03** | DESIGN | 96 | `40,901,003,000..095` | **disjoint replication** of whatever P02 selects |
  | **P04** | SINGLE_USE | 100 | `40,901,004,000..099` | the ONE paired DECIDE against graded v2.7 |

  Spans are provably disjoint from each other, from P01 and from every Phase-1
  block; ledger `60/60`, `PROBLEMS: NONE`. WP0E's *structure* was kept and its
  *stated purpose* replaced — it framed P02/P03 around carrier basis-count
  selection, and WP2's independently ACCEPTed finding then put the carrier's
  live-width margin at `0.155%`, which makes basis-count selection a poor use of
  96 fresh networks. **Claiming P03 for selection collapses the pair into one
  DESIGN block and defeats the only reason it exists.**
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

> **RULE-2 GAP CLOSED 2026-09-01 — and the review overturned one of the three.**
> The WP2/WP3/WP4 Skeptic finally ran in the foreground (the two background
> attempts died with the editor). Verdicts are **per slice**, and this was the
> **fourth of four** passes this session to find a real defect:
>
> | slice | verdict | consequence |
> |---|---|---|
> | **WP2** (`3bae94b`) | **ACCEPT** | the `MMD² ∝ 1/k` law is **algebra, not measurement** (exact closed form `C1/k + C0`, reproduced to float roundoff). The orthogonality-not-Kerdock negative **survives and strengthens**: re-run on a fresh seed family and extended to live width, Kerdock buys **`0.155%`** over random orthogonal blocks at `n = 1024` (`1248 σ` outside draw spread), down from `2.6%` at `n = 64` |
> | **WP3** (`a480ab6`) | **PROVISIONAL** | `1.03723x` reproduces to the digit and the depth confound, once corrected, **strengthens** it (matched `L = 16`: `1.14748 → 1.07423 → 1.03723`, halving to `0.15%`). But **the contract has no residual half at all**, which `VALIDATION.md` §2b-ii makes mandatory — and that is the axis where AUGMENT is expensive: `4.38x` the `aten` op count and `1.26x` the in-process wall, with **355 extra `_local_scalar_dense` readbacks** from a data-dependent branch in `mlp_kprop/wick.py:105` |
> | **WP4** (`18c5fa3`) | **REJECT — the KILL is retracted** | see below |
>
> **WP4's instrument cannot measure what it was named for.** `scale:eps`
> multiplies `K[3]` by `eps` after **every** layer
> (`p2_adjoint_k3.py:139-143`), so a source at layer `l` arrives attenuated by
> `eps^(L-1-l)` and `m(eps) - m(0)` is a degree-`L` polynomial *even in a
> perfectly linear system*. A **provably linear, provably additive** state-space
> null control, driven through WP4's exact arms and scored with WP4's exact
> statistics, is **KILLed at every depth** — `0.4325` at `L = 3` rising to
> `0.8786` at `L = 16`, the same range and the same monotone depth trend WP4
> attributed to physics. Three further record defects: all five kills came from
> the Q3 branch (so *"two kill rules fired"* is false — `verdict()` returns on
> `lin > 0.25` first), Q6's quoted `0.34–0.42` is `0.42–0.70` on disk, and the
> `32x10` "catastrophic" cell is **one draw** whose sibling sits *above* the bar.
>
> **The lane is retracted, not exonerated.** With the propagator kept intact on
> a full background, direction is fine everywhere (cosine `0.84–0.95`) and
> linearity is good at `L = 6` (`0.086–0.172`), but at `L = 10` it is
> **seed-dependent by 3.5x** (`0.262` vs `0.928`) and leave-one-out additivity
> over-sums badly (`-18.7`) — a double-counting signature, not a physics result.
> The adjoint lane returns to **open, speculative, with a known-broken
> instrument**. Full verdict and every number: bottom of `experiments/LOG.md`.

**The next action, updated 2026-08-31 after WP0–WP5.** Every truth-free package
in `GAME_PLAN.md` §8 is now closed: WP0A–D (systems), WP1 (carrier), WP2
(objective + frozen panel, ACCEPT), WP3 (AUGMENT, PORT — PROVISIONAL), WP4 (adjoint K3, kill **RETRACTED**), WP5
(integrated candidate, certified). **The remaining steps all read truth and
therefore all need Samuel.**

**The single next decision, and the branches are no longer symmetric:**

| branch | multiplier | must beat v2.7 by | what is measured |
|---|---:|---|---|
| carrier k=8 | `0.2501` | **`2.501x` just to break even** | one `n=2` organiser datum says `1.42x` **worse** adjusted |
| **AUGMENT** | **`0.1`, pinned** | **nothing — it is free** | moves the scored row `1.8%` at `n=64`; sign unmeasured |
| adjoint K3 | — | — | **NOT dead.** The WP4 kill was **retracted 2026-09-01** on a REJECT; the lane is open and unproven, and its instrument must be repaired before it is re-measured |

1. **The cheapest discriminator, and it needs no new allocation.** Score the
   carrier candidate against shipped v2.6.1 on **24–32 fresh mints from P01's
   432 unused DESIGN seeds**, common random numbers, one `2**22` truth replica,
   through `competition_score.decide()` with its GT-noise floor. It answers the
   `2.5x` break-even question before WP2's panel is spent on. **STRATEGIC: needs
   a Rule-1 contract and Samuel's prediction. It must NOT be answered by
   extending the `whest run` mini smoke past 2 MLPs — that turns a read-only
   CONFIRM tier into a selection tier.**
2. **The cheapest `fnp` AUGMENT probe — NOT the full port.** The 2026-09-01
   Skeptic left WP3 PROVISIONAL precisely because the residual axis was never
   in its contract, and AUGMENT is `4.38x` the op count there. So build **only**
   the degree-3 `(n,)` and degree-4 scalar update on top of the shipped K2p1,
   with arbitrary uncalibrated coefficients, and run `whest validate` plus a
   small `--runner subprocess` job. That converts a *projection* into a
   flopscope **price** and yields `0.4 / observed` residual headroom on the
   deployment-authoritative instrument, in one artifact, reading no truth and
   claiming no seed block. **Decision rule, fixed before the data:** PORT stands
   only if counted FLOPs stay at or below `2.199e11` **and** subprocess residual
   headroom is within a stated factor of frozen v2.6.1's. IMPORTANT, with its own
   parity burden (the harmonic/DS algebra, the Wick coefficients, and upstream's
   negative-variance clamp, which fires for AUGMENT and not for BASE).
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
| WP2 | truth-free carrier objective and frozen nested panel | IMPORTANT | **CLOSED 2026-08-31** (`3bae94b`). Deep-ReLU MMD² objective, 74/74 incl. live width. Panel frozen and content-addressed at `a9bd32a1…` with its rule, DESIGN falsifier and proposed gate inside the digest. **The finding runs against Kerdock:** `MMD² ∝ 1/k` (the plain Monte-Carlo rate, so structure buys a constant not a rate), and at matched rows random orthogonal blocks are only `1.026x`/`1.006x` worse while IID is `1.52–1.56x` worse — the advantage is mostly ORTHOGONALITY, and Kerdock adds `0.6–2.6%` shrinking with width. **Skeptic ACCEPT 2026-09-01**, with two corrections: the `1/k` law is an exact algebraic identity of the closed-form census, not a measurement; and at **live width the Kerdock margin is `0.155%`**, `1248 σ` outside draw spread — real, and smaller than recorded |
| WP3 | faithful AUGMENT | IMPORTANT | **CLOSED 2026-08-31 — VERDICT PORT** (`a480ab6`). Unfactored AUGMENT costs **`1.03723x`** plain K2 at `1024x16`, `+3.72%`, **free at the `0.1` multiplier floor**. The excess halves per doubling of `n` — `Θ(n²)` on a `Θ(n³)` base. State: degree-3 is an `(n,)` VECTOR and degree-4 a SCALAR, `1,025` numbers on K2's `1,049,600`. 33/33. **No `fnp` port exists yet.** **Skeptic PROVISIONAL 2026-09-01:** the price reproduces to the digit and the depth confound in the trend table, once corrected, *strengthens* it — but **the contract froze no residual model**, which `VALIDATION.md` §2b-ii requires, and AUGMENT costs `4.38x` the `aten` op count with **355 extra scalar readbacks**. PORT authorises only the cheapest `fnp` probe, not a full port |
| WP4 | query-directed / adjoint K3 | IMPORTANT | **REOPENED 2026-09-01 — the KILL was RETRACTED on an independent REJECT.** The 2026-08-31 kill (`18c5fa3`) fired on a statistic that **kills a provably linear system at every depth**: `scale:eps` applies `eps` at every layer, so the tangent deviation is a degree-`L` polynomial regardless of physics. All five kills came from the Q3 branch alone; Q6's quoted range was wrong; the worst cell was one draw. Repaired-instrument evidence is mixed and seed-dependent. **Lane state: open, speculative, instrument broken.** Locality still HELD |
| WP5 | integrated live-shape research architecture | IMPORTANT | **CLOSED 2026-08-31** (`292fbab`). `p2_candidate_architecture.py` + `p2_build_candidate.py`: a self-contained single file **CERTIFIED 36/36 at `1024 x 16`**, official subprocess smoke included, 0 failed MLPs. 50/50 integration tests |
| WP6 | grader-meter pricing + residual soak | IMPORTANT | **CLOSED 2026-09-01.** `official_config` models the grading server's per-MLP offset and `certify_candidate --residual-soak N` pools N official subprocess runs into a worst/p99/median tail. **The preregistered check FALSIFIED its own hypothesis:** the offset is *not* a bit-identical constant — `2**24` on 328662 and `2**24 - 32` on 328664 — so the MAXIMUM observed is carried as a conservative upper bound and the claim was corrected rather than kept. It costs `16,777,216` FLOPs of floor headroom and binds only above `C/B = 0.09999237`; v2.6.1 sits `82.3e9` FLOPs clear. 9 new checks against the STORED GRADED REPORTS, 5/5 mutations fire their named guard, `official_config` 82/82 |
| WP7 | corrector training-data source | ROUTINE | **OPEN QUESTION FOR SAMUEL, work not blocked.** The brief's § 8 names `aicrowd/arc-whestbench-public-2026` as a 500k-MLP training corpus. It is not: it has two configs (`mini`/`full`), `size_categories 1K<n<10K`, `n_mlps: 100` in mini, and **it IS the CONFIRM tier** — so fitting on it is exactly what § 6 of the same brief and `AGENTS.md` forbid. Proceeding on **own mints from P02** instead, which is disjoint from CONFIRM by construction, at the live shape, and with a GT-noise floor we can actually quote |
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
| per-neuron correction at the READOUT | **closed 2026-09-02 (WP9)** | exhausted by v2.7's sixteen constants; its single per-layer number predicts the residual at `R^2 = +0.51`, matching a 354-parameter network |
| per-neuron correction of the in-flight VARIANCE | **closed 2026-09-02 (WP9/WP10)** | destructive at ZERO fit error: truth in that channel is `17x` worse than no correction at all |
| per-neuron correction of the in-flight MEAN | **closed 2026-09-02 (WP10) on an entry price** | ceiling `6.872x` is real; break-even needs held-out `R^2 >= 99.85%`; best achieved `+0.50` on an easier target |
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
