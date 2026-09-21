# Strategist packet — after the graded F21 exact fusion

**Private decision request, 2026-09-14.** Answer this as if you have no repository
access and no chat history. We have now made and graded one exact, modest
improvement to our working estimator. We need a bounded next architecture
that can materially improve the private score, plus a clear fallback if the
first discriminator fails. Do not propose an open-ended search or assume the
public board reveals anyone else's method.

## Task, score and hard limits

The ARC White-Box Estimation Challenge 2026 Phase 2 asks for the expected
post-ReLU activation of every neuron in a random He-initialized, bias-free
MLP under Gaussian input. Live shape is width 1024, depth 16; output is
`(16,1024)`. The estimator sees weights, not input samples or truth. The
private objective is `mean_i[MSE_i * max(0.1, F_i / 2**41)]`, where `F_i`
is network `i`'s counted FLOPs; lower is better. Phase-2 lambda is zero:
residual time is **not** a priced cost, but exceeding 0.4 s per network
fails that network. Wall cap is 120 s/network, memory cap 8 GiB. The
installed/graded stack is WhestBench 0.16.1 and flopscope 0.12.1. Grader
arrays are immutable; allowed estimator imports are flopscope, its numpy
facade, whestbench and Python stdlib. Routing work through uninstrumented
arrays is forbidden. The prize uses a private rerun of up to two designated
submissions on fresh networks; no designation is authorized. The $20k
contribution write-up is a co-equal objective, currently stale/do-not-submit
pending a separate claim review.

## What the first strategy accomplished

The September-12 strategy was to preserve our **working** full Wick and
centering recurrence, all K3 history, source births, scalar K4, saturation,
precision, backend and conservative generation chooser, and fuse only two
`_dslice21` products sharing a right operand. Let `C=A_L*w2b` and `*` be
elementwise multiplication. The old portion was

```
2*(A_D*M) @ M.T + 3*(A_L*C) @ M.T
```

and the replacement is

```
[2*(A_D*M) + 3*(A_L*C)] @ M.T.
```

The other two products, division/mask, all nonlinear terms and state updates
were untouched. Fused L4 source SHA256 is
`d77dab0e2c740cc9f6cd7d9172adacc8da731cc5b40ca87798f3fb006e475a94`;
the own frozen L4 parent estimator member is
`052dd768e970c4bae831cff26f8ff15b0487f3b95eadf2fd3578e542aec34d58`.
The tiny independent tensor oracle and actual-precision full trajectory
passed predeclared tolerances; real-arithmetic equivalence was **not**
misrepresented as bit identity. Whole-program counted FLOPs fell from
`1,068,245,507,686` to `928,457,637,478` per live network, **13.0857438%**.
The candidate was frozen before score measurements. No coefficient was fit.

Fresh P10 DECIDE used 128 newly minted, paired networks and two independent
`2**24`-sample truth replicas. Its mean midpoint truth-noise floor was
`2.2363657945e-9` beside mean raw MSE about `4.1821e-8` for both L4 and
F21. The preregistered canonical **paired arithmetic** gate was a 1.05x
score gain over original L4 with a positive 95% t interval. F21 passed:
`1.1505662809x`, interval on L4-minus-F21 score
`[2.4582563842e-9, 2.5842673161e-9]`, 128/128 pairs favor F21. The
accuracy change is negligible relative to its floor; the gain is the
metered cost reduction. Original L2 was context, not an alternate selected
on P10. Independent Skeptics accepted the engineering and DECIDE claims.

Candidate-specific Linux-cap evidence passed at measured VmPeak 5,381.7 MiB
against 8,192 MiB. The exact-byte Full certificate nevertheless REFUSED
locally: its capped two-row official subprocess smoke failed 2/2 on the
0.4-s residual cap. A separate 100-row capped official local mini failed
100/100 on the same category, local residuals 7.312–9.633 s. No estimator,
FLOP-budget, total-wall or combined-budget failures appeared. This host's
absolute local residual does not transfer to the grading server; the
operation-count server forecast was 0.1888 s but based on sparse anchors.
The normal clean-mini gate was **not** silently waived. Samuel explicitly
authorized one diagnostic exception and one submission of the frozen tar
SHA256 `4f3cc6f440e4cb66ea86a0904c00e182eb7ecf5aaab0ee64c73efb7d22904775`.
No second submission or designation was authorized.

## New terminal graded evidence, with its limits

AIcrowd submission **#330950** is GRADED. The original official report
records 100/100 completed networks, 0 failed, no estimator/budget/wall/
residual failure flags, and exactly `928,457,637,478` counted FLOPs on
every row. Thus the predeclared **server-feasibility diagnostic passed for
this submission**. Contrary to a conditional cost forecast, this report
shows **no** `+2**24` server offset from the f64-readonly ledger. Worst
graded residual was `0.15623225` s, giving `0.4 / worst = 2.56029x`
headroom; worst predict wall was `93.1597153` s, giving `120 / worst =
1.28811x` headroom. The server smoke passed 11/11 worker attempts.
This does not guarantee the private rerun's failure rate or wall time.

The same 50 fixed exposed public score rows appear in archived original
L4 submission #330290 and F21 #330950, with matching network indices and
names. Their full-precision official public adjusted scores are L4
`1.835981420067277e-8` and F21 `1.595729544363834e-8`, a descriptive
**1.1505592702x** L4/F21 ratio, or 13.0857466% lower score; all 50
displayed paired adjusted rows favor F21. Public raw-MSE scores are nearly
identical (L4 `3.7794362910403833e-8`, F21
`3.779436170248118e-8`), but the official public truth-replica noise
floor is **unavailable**; the displayed `sampling_mse=1.132e-6` is not that
floor. Do not claim a public accuracy improvement from the tiny difference.
This public comparison is **read-only CONFIRM**, not a new DECIDE gate or
an independent sample from the private distribution. The P10 result above
is the binding fresh method decision.

An independent fresh-context Skeptic independently enumerated all graded
rows, checked the cost and timing identities and public pairing, and
accepted the zero-failure diagnostic in this scope. Its overall exact-byte
attribution remains **provisional**: the CLI upload receipt, graded page
(`submissionFiles=null`) and read-only submission-status API all omit a
server-provided archive/member hash. The locally checked tar/source hashes and exact
graded FLOP fingerprint strongly support the binding, but are not a
server-side cryptographic receipt. The pre-upload package receipt retains
historical `submitted:false`; the append-only slot ledger retains its
at-upload `pending` row. A separate final-status record and raw report now
state GRADED; neither historical record should be rewritten to disguise
the local Full/mini refusals. No private score or prize eligibility
determination is observable here.

At a read-only authenticated board snapshot on September 14, the displayed
leader was `2.5e-9`; our rounded F21 level was `1.6e-8`, rank 58 of 156.
This **6.38x public level gap** is dated and unpaired, not a promotion
statistic or the private prize result. It exposes a hard algebraic limit:
with F21's currently reported public raw error, even the minimum 0.1 cost
multiplier would score `3.7794e-9`, still **1.51x** above that dated leader.
Conditional on unchanged truth/leader, an estimator at the cost floor would
need at least **33.85% lower reported raw MSE**; at multiplier 0.2 it would
need about 66.9% lower. Therefore compression alone cannot close this
displayed gap unless it also helps accuracy, or the target changes.

## Research state and forbidden shortcuts

The credible route to larger cost reductions is a compact representation
of the **same working full recurrence** before compression. The older
compact `exact` arm does not reproduce our L2 nonlinear births and performed
poorly on exposed P01 DESIGN; that is not evidence that faithful
compression is bad. A full-rank bridge maps a working source as
`A=A_L, P=M, w2=w2b, s=e=0, Z=A_D, L=I`, without new K4-feed fields.
At n4/depth4, a separate full-rank interpreter/weighted-Gram/QR bridge
passed its tiny checks. At n8/depth6, B4 full-rank differential passed
756/841 checks and failed 85/841 numeric checks, first over tolerance
at layer-3 `wick.(4,1)` and worst normalized difference `4.04473e-4`.
Structural rank/activity checks passed. The frozen L2 side used f32 while
the compact interpreter used f64 on f32-quantized weights; the drift may
be amplified rounding or a real recurrence defect. The smallest **unrun**
discriminator is frozen L2 `_layer` in f64 versus compact f64 on that
same fixed fixture and the existing checkpoints/tolerance. A f64 pass
supports mixed-precision drift; a failure localizes a substantive mismatch.
Only after full-rank parity should one rank/storage approximation be
tested, preserving all source ages and charging every QR/transport/join/
reconstruction/query cost. No compressed method has yet passed a fresh
score or full-width cost gate.

The causal signed-law K4 project is **deferred**, not scientifically killed.
Its exact tiny accounting and memory inventories are preserved, but its
integrated n4 path passed only 93/172 checks with exploding K4 correction;
n8 passed 186/187, failing on negative variance because the K4 covariance
term drove variance negative. The bank was transported linearly through
nonlinear layers.
Do not make an exact signed distribution law, more symbolic memory
inventories, or another full-width unstable run a prerequisite for the
next candidate. A coherent **approximate** K4 or output-sensitive old-source
sketch may still be worthwhile if all consumers and births are consistent,
stability is explicit and the executed program is fully costed.

P01/P07 are already exposed reusable **DESIGN**, P08 is exhausted. Any
new promotion decision needs a newly claimed/preregistered single-use
fresh DECIDE block, predictions before results and the canonical paired
arithmetic score-difference interval. Public CONFIRM may check but never
fit/select. The sealed vault remains untouched without a milestone
contract. HUMAN_MODEL, shipped estimator and frozen archives are protected.
Every heavy or >30-minute job needs a recorded optimization review and
conservative `workerq` RAM/VRAM/CPU declaration. A fresh-context Skeptic
reviews every important or strategic result. Participant-derived code
has unresolved eligibility interpretation and must not be imported into
our own-parent candidate; source attribution and an OSI license for a
future public prize release also need separate review. No external
message, second submission or designation is authorized.

## Decision requested from the strategist

This packet deliberately does **not** reproduce the working estimator's
full Wick/centering coefficient table, all birth and final-readout code,
compact factor lifetimes, or flopscope's exact primitive tariff. Treat
that working nonlinear recurrence as an **immutable operator to wrap and
match**, not as a formula to invent from this summary. You can specify a
first differential and rank conceptual architectures without those
details. For a later compact method, use parameters `n=1024` (width),
`d=16` (depth), `h` (retained source ages), and proposed ranks `r`;
list dense products, elementwise work, QR/joins, materialized shapes and
co-live lifetimes symbolically. Do **not** claim a final exact FLOP count,
process peak, or drop-in implementation from this packet. The repo-local
Builder must audit the actual source, implement the selected slice and
meter its executed program before any score experiment.

1. **Choose one next bounded hypothesis** that can plausibly improve both
   score and competitive position beyond F21. Compare a faithful compact
   full-rank bridge followed by one compression choice with at most one
   alternative accuracy mechanism. Explain the causal route to lower
   final-layer error, its cost floor and server wall/residual risks; do not
   cite captured tensor energy as a substitute for output sensitivity.
2. For the compact route, specify the **single cheapest tiny f64
   differential** that separates B4 rounding from a recurrence defect,
   then the first full-rank gate and one subsequent compression
   intervention. State what must match (Wick/centering, D3/D21, births,
   scalar K4, all ages, final readout), what tolerance is legitimate at
   the actual target precision, and what result kills only that
   implementation. Give a **parameterized** operation/lifetime table and
   identify which source-level counts and primitive rates the Builder
   must measure to complete the cost and process-memory model before any
   live-shape score run. The tiny f64 differential is a proposed new
   IMPORTANT probe, not an already passed bridge.
3. If a larger gain needs approximate K4 or another nonlinear correction,
   provide one coherent update/readout design with source identities,
   all consumers/births, explicit approximation error and covariance
   stability. Name a cheap truth-free falsifier before exposed DESIGN.
   Do not require an exact finite signed-law proof; do not silently delete
   the existing scalar K4 or combine several interventions.
4. Give a **go/no-go threshold** for one preregistered DESIGN and fresh
   DECIDE experiment on a frozen candidate, including a counted-cost
   target, output-error target or sensitivity test, worst-row 0.4-s/120-s
   headroom plan, truth-floor requirement and paired arithmetic gate.
   Use F21 as primary comparator, original L4 as history and L2 as
   robustness context. Address whether the best near-term contribution
   write-up should document F21's exact identity/validation or await a
   larger method, with claims limited to established evidence.

Return a **ranked decision with one executable tiny first slice**, enough
pseudocode for that discriminator, a parameterized cost/liveness table
for the later architecture, a cheap falsifier, estimated resource envelope
and explicit stop condition. State which exact numbers depend on the
unprovided recurrence and tariff. If
neither route has a credible path past the score floor, say so and give a
bounded fallback. Do not recommend a new submission or designation from
public rank alone; those require separate Samuel authorization.

## Local evidence index for a verifier (not needed to answer)

`experiments/2026-09-13_p10/decide_P10.json` SHA256
`d868e4335168fa3ff1159c594be3166b61f81ebaba8b93faa846f0b77c23d79b`;
`experiments/run_reports/f21_diagnostic_330950_graded_report.json` SHA256
`4dc5c2f6f1d06220921efb74b71a3b5bee930e141e7319af0e0d55028100e5f3`;
`experiments/run_reports/k3_strassen_L4_330290_graded_report.json` SHA256
`65b5d9668620f1b17d6392198b408568287be1d6b3de6e009de43dfc687838e4`;
`experiments/2026-09-13_p10/f21_graded_skeptic.md` (independent verdict);
`experiments/2026-09-13_p10/board_after_f21_330950.json` SHA256
`ea0143329713035e2c6ea9b7ebeb9c42a03c1654f49a561cbd90fb27ec51c899`;
`experiments/2026-09-13_p10/f21_submission_330950_api_status.json`
SHA256 `125cd4c8d1821e915183a5d83285038e93d68e7b3c140fcfc5d15231f6c7eac1`;
`experiments/2026-09-12_b4_fullrank_deep.json` SHA256
`a5c6295c45695cc104f6db4d4f779cce059b18773184dfb418092953e420e52c`.
