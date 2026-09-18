# CURRENT_STATE.md — read this first

## September 18 — corrected query condition failed; P12 score and own fit running

Commit `8cae199` records the repaired stage-3 decision adapter, its query-evidence
producer, atomic/verified DESIGN sidecars, a candidate inventory, corrected
qE/qEd versioned files and the fixed-storage old-qEd comparator. Standard
preflight passed 12/12; focused adapter tests 10/10; independent Skeptic
accepted the final identity/score checks and DESIGN reuse. The complete
transcript and Skeptic verdicts are under `experiments/2026-09-18_*`.

Workerq #1668 **SUCCEEDED** on its frozen `553e9ac` snapshot. All 240 batch-B
rows and 10 prediction sidecars verify; P12 truth-read receipts from both jobs
are reconciled into the workspace register. See the paired saved-row analysis
[`synthesis.json`](../experiments/2026-09-17b_stage3/synthesis.json), the LOG
tail, and the [fresh scientific Skeptic](../experiments/2026-09-18_stage3_skeptic_verdict.md).
The Skeptic's verdict is **PROVISIONAL**: executed qE gains 1.0242× versus qoff
on the score-only official-precision comparison, but historical query artifacts
lack explicit E and cannot satisfy the full S3 gate. Isolated P3 hurts adjusted
score; PAIR+Q plus P3 improves conditionally and passes the stage-2 DESIGN score
gate uncompressed. The reference-lambda diagnostic reveals an interaction but
is not a shippable source. No candidate qualifies for P13.

Corrected qE/qEd versioned files passed tiny algebra and `whest validate`, with
no truth run. The source override for a complete label-free query audit is ready.
Samuel answered "i dont know" to all three requested corrected-objective
predictions; the LOG records this verbatim in commit `bc0c21b`, before new
measurement. The old and corrected full-shape label-free audits #1756/#1764
**SUCCEEDED** on the same network with complete source/identity/layer checks.
qEc raised complete recurrent E error from 0.0247135 to 0.0296860 and worsened
all five continuation checks, despite slightly higher same-state retained query
energy. A fresh Skeptic **REJECTED** the registered query subcondition; see
`experiments/2026-09-18_corrected_storage/query_skeptic.md`. The fixed four-arm
P12 DESIGN score comparison is workerq **#1765**, running at the last check;
do not interpret partial rows. No corrected storage arm can pass this
preregistered complete query screen, regardless of its eventual P12 score. P13
remains untouched, and no submission has been approved.

The original September17b branch-(b) contract also permits one bounded own
four-parameter fit. Its own-source generator and fixed 19-evaluation/12+12 P12
driver are source-only, targeted-tested and independently accepted in
`experiments/2026-09-18_own_fit_source_skeptic.md` and
`experiments/2026-09-18_fit_driver_skeptic.md`; neither has read fit labels.
The LOG tail records the dimensionless ridge objective, exact coordinate
schedule, 40-evaluation cap, and workerq optimization review before data.
Final standard preflight passed 12/16 on the scientific source bytes. The
registered fit is workerq **#1748**, submitted from commit `6c4c02a` with
0 GPUs, 12 GiB RAM and 4 CPUs. It started at **2026-09-18T03:05:45Z** after a
CPU-capacity wait; do not inspect or select from its partial output. A first mistaken queue request (#1747)
was cancelled before execution because workerq treated `--vram 0` as one GPU.
On #1748 completion, reconcile its P12 truth-read receipt from the immutable
snapshot, verify all 19 complete evaluations and paired held-out comparison,
then synthesize before any finalist choice. No fit accuracy result exists yet.
Read-only fit and corrected-storage result adapters now pass synthetic provenance
refusals (valid fit plus 17 forged cases; corrected storage 10/10) and independent
Skeptic review. Final-byte standard preflight is GREEN 12/16; see the LOG tail,
`experiments/2026-09-18_verifiers_skeptic.md` and
`experiments/2026-09-18_verifiers_preflight.log`. The fit reader has not seen a
real result, and the corrected reader has no new score or query artifact yet.

---

## September 18 UTC / September 17 evening — independent review: stage-3 interpretation needs repair

Read [strategy and implementation review](research/2026-09-18_strategy_review.md),
[validation audit](research/2026-09-18_validation_audit.md), and
[architecture audit](research/2026-09-18_architecture_audit.md) before interpreting the
active stage-3 screen. Stage 1 is complete; batch B job **1668 was RUNNING** at this
review. Its partial accuracy results were not analysed. The committed September17b
contract remains operative; no gates, candidate bytes, P13, vault or submissions changed.

**Confirmed instrument defects:** `s12_decide.py` gates observed rather than its
registered official-precision score, does not enforce paired identities, and does
not implement all stated query checks. Failure identity/fallback handling also needs
repair. Recompute only after a reviewed repair, using saved rows; no estimator rerun
is required to repair that arithmetic.
**Confirmed objective mismatches:** qE retains pre-Wick left factors at a post-Wick
join; qEd omits an A factor and uses the wrong lineage's birth weights after the
first join. Independent tiny algebra/source checks reproduce these discrepancies.
The executed files' label-free findings remain valid; they do not establish the
intended objectives or a scientific closure of query-directed compression.

The [independent Skeptic](research/2026-09-18_skeptic_verdict.md) accepts the bounded
repair/research plan and rejects the claim that stage 3 already implements its
registered gate/mechanism. There is no demonstrated winning configuration yet.
[Live Chrome snapshot](research/2026-09-18_competition_snapshot.md): latest submission
remains v31 #331317, rank **62/170**, adjusted **8.752e-9**. Clean leading reference
#331393 is **2.239e-9**; newer #331435 displays **2.220e-9** but includes a
TIME_EXHAUSTED row and inconsistent page summaries. Use the clean reference for
planning; do not infer failure-score semantics from that UI inconsistency.

---

## September 17 (evening) — slice 2026-09-17b IN PROGRESS: contract committed, panel minted, stage-1 batch running

**Operative:** the Rule-1 contract at the bottom of [experiments/LOG.md](../experiments/LOG.md)
(`CONTRACT (Rule 1): slice 2026-09-17b`) and its pre-data amendment, both committed before any
of this slice's data existed. Plan: [GAME_PLAN](GAME_PLAN.md), [strategist response](STRATEGIST_RESPONSE_2026-09-17_slice2.md),
[handoff](CLAUDE_CODE_HANDOFF_2026-09-17_slice2.md).

### The source finding this slice is built on

`local_sim/closure_audit/progswap.py`: **public v17's executed nonlinear program (87 oriented
terms) CONTAINS frozen v32's 27 exactly**, with identical rational coefficients and no term of
ours absent from theirs. The reciprocal swap is therefore pure addition one way and pure
deletion the other. No earlier arm ran that program: BUDGET 4 has 141 terms and omits the six
kappa_iiij rows, PAIR 114 and omits the same six, Q3q 57 and omits 33 while adding 3.

Structurally, the 60 terms the reference drops from our BUDGET-4 table are **every three- and
four-factor monomial** (27), `D22xD22` and `D4xD4` entirely (7), and a per-output subset of the
remaining two-factor families (26). That is an empirical mask and **it is not imported.** Our own
stated rule instead: every arm ever screened on this base was **mixed-order** — the BUDGET and
PAIR tables were generated under "kappa_iiij assumed zero" while carrying cost-4 blocks, and
`kappa_iiij = (m/2) lam C` has cost 3. `closure_budget4_q.json` (147) and `closure_pair_q.json`
(120) are that completion, derived here.

### State of the slice

| item | status |
|---|---|
| Rule-1 contract + Samuel's three predictions (all **"I don't know"**) | committed before data, `aba5373` / `90753cb` |
| Prediction sidecars (`design_sidecar.py`, `--pred-out`) | landed, **19/19** tests incl. five refusals |
| Seed blocks **P12** (DESIGN, 24 claimed) and **P13** (SINGLE_USE, untouched) | registered, disjoint, P11 retired from new accuracy selection |
| P12 truth minted | job 1650, `92a3228e…`, independence ratio **1.0121**, truth noise **4.4961e-9**/replica |
| Reciprocal chains + own closures, label-free gates | **69/69** (transplant) + **48/48** (own tables), three availability modes against the independent f64 reference |
| Stage-3 consumed-query basis | built; **17/17** label-free, including the cancelling case the old surrogate provably misses |
| Stage-1 batch (9 arms x 24 nets on P12) | **running**, job 1663 |
| Submission / designation / vault | none, and none authorised |

### Measured, label-free, at the LIVE 1024x16 shape (jobs 1658/1659)

The shared-coordinate join's basis objective, replaced by the actual consumed query's right
singular space, against the same chain's own diagonal surrogate:

| setting | arm | C/B | final-row displacement | D3 | D21 |
|---|---|---:|---:|---:|---:|
| age4 / rank256 | surrogate | 0.21623 | 8.142e-5 | 2.254e-2 | 3.069e-2 |
| age4 / rank256 | **query basis** | 0.22123 | **7.340e-5** | **1.847e-2** | **2.785e-2** |
| age3 / rank192 | surrogate | 0.18318 | 2.799e-4 | 7.471e-2 | 9.587e-2 |
| age3 / rank192 | **query basis** | 0.18869 | **2.203e-4** | **6.211e-2** | **8.591e-2** |

The query basis retains more of every consumer at both settings, for +2.3% and +3.0% of `C`.
The preregistered D3-protection arm (`qEd`) is **worse than the plain query basis at both** —
a clean negative on a secondary that was declared before the measurement. At n=512 with
r = n/2 the query basis was *worse*; the effect appears only where the compression is
actually aggressive.

### The release arithmetic, stated plainly

On one consistent panel scale, 2x over graded v31 means score <= **5.186e-9**. What that costs
in raw error at each achievable cost:

| C/B | raw MSE must be <= |
|---:|---:|
| 0.100 (the multiplier floor) | 5.186e-8 |
| 0.150 | 3.457e-8 |
| 0.183 (age3/rank192) | 2.834e-8 |
| 0.216 (the current cost base) | 2.398e-8 |
| 0.280 (v32 + P4 d5) | 1.852e-8 |

The three measured bundles score **8.75e-9 (C/B 0.216), 8.84e-9 (0.260), 9.32e-9 (0.280)** —
flat to 6% across a 30% cost range, which says the compression trade already sits near its own
optimum and **cost alone cannot reach the gate.** Closing it needs the raw error at the current
operating point to improve by about **1.69x**, which is 1.53x beyond public v29's raw on our own
networks. That is the honest size of the remaining problem, and the stage-1 batch is what
measures whether the program route moves it.

Latest graded control stays **v31 #331317, 8.7520303e-9**; F21 stays the designated fallback.

---


## September 17 latest strategist review — next slice defined; major release still unmet

**Active:** [GAME_PLAN](GAME_PLAN.md), [strategist response](STRATEGIST_RESPONSE_2026-09-17_slice2.md), and [Claude execution handoff](CLAUDE_CODE_HANDOFF_2026-09-17_slice2.md). Earlier sections below are historical snapshots, not instructions to replay their stages. No new estimator experiment, truth access, submission, designation or vault touch occurred during this review.

Four independent audits support the worker's measured DESIGN gains but reject broader lane closures. The repaired Q3q still misses **33 reference-program terms**; same-state captures do not compare full recurrent programs or establish negligible task error. The shared-source Gram omits cross-column/A-M query interactions. Rank256 and rank384 are effectively tied at official label precision; noise correction already in the score JSON reduces rank256's gain vs v32 from 1.1493x to about 1.1343x. Keep F5 failed and both controls retained.

**Next executable work:** committed bounded contract + new Samuel predictions; small prediction-sidecar extension; four reciprocal complete-program chains plus native v29 on one fresh DESIGN panel; our own coherent recurrence; one query-directed basis intervention and conditional cheaper setting; freeze one challenger for fresh DECIDE and exact release checks. No small canary by default. No new accuracy selection on P11 after its approximately 13 reads.

Latest graded control stays **v31 #331317, 8.7520303e-9**; F21 stays the designated fallback. The new best observed DESIGN bundle is **1.18585x** over v31, not a release. Remaining gain to the same-panel 2x target is **1.68655x**, not the packet's cross-scale 2x raw calculation. No universal cost floor was proved.

Chrome September17 refresh: rank **62/170**; new leader #331393 public **2.239e-9**, raw **1.655e-8**, mean compute **13.53%**. A4.4e-9 major release would be around rank13, not first. The active plan retains the 2x release gate and treats winning as the further objective.

## September 17 worker snapshot (historical): handoff-09-17 executed — release target UNMET

**Latest packet:** `docs/STRATEGIST_PACKET_2026-09-17_slice.md` (self-contained). Narrative and
pre-registered verdicts: `experiments/LOG.md` 2026-09-17 entries. Skeptic: `experiments/2026-09-17_skeptic/verdict.md`
(PROVISIONAL; its follow-ups resolved by job 1642). No submission, designation, vault touch or new seed block.

| step | result (P11 DESIGN, 24 nets, paired arithmetic) |
|---|---|
| 0 receipt name + PowerShell module path | fixed, tests 10/10, preflight GREEN 12/16 (job 1611); Skeptic ACCEPT |
| 1 closure audit | v32 == upstream cost-≤2 closure; v17/v29 == pruned PAIR (+6 κ_iiij rows), coefficients agree; f64 reference == live; same-state report: states near-identical except transported core dG and λ |
| 2 richer closure | BUDGET 4 0.835×, BUDGET 3 0.781×, PAIR 0.866× (κ_iiij = 0); readout-only null; protected D22 feed null; **with κ_iiij restored 1.189× over BUDGET 3 but still 0.929× of v32** |
| 3 shared old-lineage coordinates | age 2–3 ranks ≤256 destroy accuracy (0.26–0.79×); calibrated label-free proxy; **age 4 rank 256 lean + P4 depth 5 = 0.2163 B, raw ×1.22, adjusted 1.149× over v32 / 1.065× over P4 alone** |
| P4 depth 5 | exact: raw 1.0002, adjusted 1.079×, 6.53 GB Windows commit |

**Best supported cost base:** `estimator_k3_v37_a4r256_p4d5.py` — P11 score 8.746e-9 ≈ **1.19× over v31**
(projection ≈7.4e-9 public). 19,914 ops. Not DECIDE-tested, mini-run, Linux-memory-checked or certified.
**Not a release** under the 2× / 4.4e-9 criterion. Fallbacks unchanged: v31 `#331317` (latest graded),
F21 `#330950` (designated).

**Next:** strategist answers packet §4 (source of v29's raw edge via reverse ablation on v17 vs fitting our
own joint closure+core on a fresh DESIGN block; recent-lineage cost floor; whether a ≈1.19× canary is
worth the evidence path). Do not repeat the step-2/3 arms. P11 is read ~13 times: new accuracy
selection needs a new DESIGN block.

---


## September 17 review: nonlinear closure mismatch found — next slice is defined, no major release yet

> **Independent source/validation review:** principal P11 numbers reproduce, but the conclusion excluding κ4/closure interactions is rejected. Our actual nonlinear table has **25 terms**, public v17 **62+4**, including live mean products absent here. Protected-feed allocation before compression also differs. N1's tiny-width screen is exploratory and registration-deficient; it does not close live-shape compression. Stage 0 needs two integration fixes. See `docs/STRATEGIST_RESPONSE_2026-09-17.md` and its linked reviews.

**Active plan pointer:** `docs/CLAUDE_CODE_HANDOFF_2026-09-17.md` — independently derived
connected nonlinear closure → controlled protected feed → shared old-source factors
→ major-release evidence. `docs/STRATEGIST_PACKET_2026-09-17.md` has been corrected
to remove superseded causal/provenance claims. New slice not yet executed; register
its actual contracts and obtain required predictions before measurement.
No submission, designation, vault touch or public fitting occurred. v31 `#331317` remains the
latest graded file; F21 `#330950` the designated fallback.

| stage | result (P11 DESIGN, 24 nets, paired arithmetic) |
|---|---|
| 0 instrument repairs | Useful repairs landed; remaining receipt producer/discovery mismatch and child PowerShell/module compatibility must be fixed before release |
| 1 distinct-index κ4 feed (v32) | real but small: **1.032×** score, t95 > 0, +0.43% cost — slice gate 1.5× FAILS; supported improvement |
| 1b exact (3,1)+link bivariate κ4 (v34) | **worse: 0.924×** — falsified (also with 504aldo λ: 0.836×) |
| diagnostics | λ profile swap +2.4%; public v17 raw 1.22× better, **public v29 1.545× better** than v32; v17's (3,1) rows (term-identical to ours) +15.9% in theirs, −8% in ours; gap grows at layers 4–10 |
| 2 P3 κ3×C feed | derived exactly and built (13/13 parity); not screened — rank 32 costs ×1.069 |
| 2 P4 exact cost | Tested depth5+trims: C/B 0.2802, 17,483 ops, 6.0838 GiB Windows commit; paired raw parity and enforced Linux memory owed; no general cost floor proved |
| 3 N1 compression | Tested tiny merged policies deprioritized; n64 proxy informative but did not pay; 0.17 gate retracted; latest truth screen used unregistered seeds and contract committed after sampling began |

Best supported bundle ≈7.9e-9 (planning arithmetic; needs depth-5 raw parity, mini,
Linux 8-GiB, Skeptic before any approval request). Target was 4.0–4.4e-9.

**Live Chrome snapshot at this review:** rank **59/166**, top-ten boundary **3.8e-9**,
approximately **4.4e-9 at rank13**; leading detailed score previously **2.448e-9**
and current rounded board remains **2.4e-9**. No leader source is available.
**Noise convention:** P11 has 4.7634e-9 per truth replica, **2.3817e-9 for the
two-replica average used in these scores**. Historical sections below are dated
records, not instructions to repeat completed work.

---


**September 16 independent strategy review:**
[`research/2026-09-16_strategy_and_implementation_plan.md`](research/2026-09-16_strategy_and_implementation_plan.md)
contains the live Chrome competition audit, architecture/validation/literature reviews,
and independent Skeptic verdict. Proposed refinements: test full κ4-to-K3 birth plus
transport (repeated slices alone do not establish completeness), price the complete
N1 join/transport/query cycle, and repair bounded evidence/timing defects. This was
a read-only research review, not a new candidate or measured gain; Slice 7 item 1
remains unstarted. The existing execution contract remains operative pending a
recorded amendment. Live detailed leader score was `2.448e-9`; v31's gap is about
`3.58×`. One successful projection does not establish general half-percent accuracy.

## September 16: v31 GRADED at 8.752e-9 (#331317), Slice 7 item 0 closed, item 1 not started

**Operative contract:** `docs/CLAUDE_CODE_HANDOFF_2026-09-16.md` (Slice 7, Sep 16–29),
answering `docs/STRATEGIST_RESPONSE_2026-09-16.md`. **Latest state packet:**
`docs/STRATEGIST_PACKET_2026-09-16.md`. Narrative: `experiments/LOG.md` from the bottom.

| | adjusted | raw | C/B | worst wall | worst residual | failed |
|---|---|---|---|---|---|---|
| F21 `#330950` (designated fallback) | `1.596e-8` | `3.780e-8` | `0.4222` | `93.16 s` | `0.1562 s` | 0/100 |
| **v31 `#331317`, non-designated canary** | **`8.752e-9`** | **`2.907e-8`** | **`0.3010`** | **`68.13 s`** | **`0.1402 s`** | **0/100** |

**Graded against graded**, paired on the 50 public-board networks both reports
score: **`1.8233×`, better on 50/50**, t95 on the difference
`[+6.88e-9, +7.53e-9]`. Board ~rank **58/165** (was 69). Leader `2.4e-9`;
504aldo's family `4.2–5.3e-9`. Snapshot:
`experiments/board_snapshots/2026-09-16T1611Z_slice7_v31_canary_board.md`.

**`predict_public.py` projected `8.713e-9`; the server gave `8.752e-9` — +0.45%.**
First time the projection chain has been scored against a real row. It works.

### Slice 7 work list — where each item stands

| item | status |
|---|---|
| **0a** v31 evidence path + canary | **DONE.** Certified 60/0/1-skip, Linux 8 GiB CERTIFIED, mini-100 clean twice and bit-identical, graded clean |
| **0b** mint P11 | **DONE.** `P11[40,916,011,000..023]`, DESIGN, 24 nets × 2 replicas of 2²⁴, independence ratio `0.9998`, noise `4.76e-9`, ledger finalized, passed through in `.gpuq.toml` |
| **0c** rules question | **WITHDRAWN by Samuel** 2026-09-16. Do not re-raise. `PROVENANCE_504aldo.md` stays current as an internal record |
| **1** close the κ4 half: D1 → D2 → D3 | **NEXT. Not started.** Needs Samuel's prediction recorded first (strategist's: D1 moves raw >15%, 40%; D2 finds a ≥1.3× discrepancy, 60%; half closes to ≤2.4e-8 P01-scale, 55%) |
| 2 P3 · 3 P4 · 4 N1 · 5 | not started |

### Decisions pending, all Samuel's

1. **Whether v31 replaces F21 as the designated fallback.** The 09-16 handoff's
   condition (a clean graded row) is met. The 09-14 response's rule is three
   clean graded runs before designating. No agent designates.
2. **The six comment lines in `estimator_k3_v31.py` that name 504aldo** (lines
   28, 31, 605, 935, 1136–1137 — attributions, no code). Submitted unchanged
   because every piece of approved evidence was bound to that sha. Stripping
   them is five minutes; it must happen **before** a designation file's
   evidence is gathered, not after.

### Instrument facts learned today — read before quoting a wall or a residual

- **×1.60 on local worst wall is NOT a safety margin.** It was conservative for
  every contended local run and **under-predicted by 7% on an uncontended one**
  (63.2 s vs 68.1 s graded). Median-to-median graded/local is `1.46–1.49`
  contended, `1.82` alone. Until `predict_public.py` is repaired, quote graded
  wall as **1.9 × standalone local median**. The "1.062 anchor" is retracted.
- **The 13.4 µs/op residual model is ~20% conservative.** Graded rows sit at
  10.73–11.10 µs/op. Left frozen on purpose; quote real headroom (v31: 2.85×).
- **`whest run` subprocess bills exactly 2²⁵ FLOPs below the server** for both
  submissions; the grader-twin path matches the server exactly.
- **Mini runs must use `scripts/run_mini_json.ps1 -NoResidualLimit`.** A bare
  `whest run ... > file.json` writes UTF-16 and, without the flag, fails every
  row on this host's residual and reports the zero-prediction MSE. That cost
  4h28m on 09-16. The wrapper also no longer discards a report when an MLP
  raises (a second 1h55m loss the same day).
- **`local_sim/p2_repro_mlp.py`** runs one named public network through a
  candidate with the grader's own `MLP` construction and `setup()` — 45 s,
  instead of a 2-hour split.
- **v31 is deterministic to the bit** (100/100 networks, every layer, every
  FLOP, two runs). Job 1400's 2-of-100 raise never recurred locally or on the
  server; mechanism unknown, evidence destroyed. Open-low.

---


## September 15: P1 measured and attributed; P2 BUILT, and an invariant caught it

**P1 gives 1.2371x on 24 of 24 design networks** (raw `4.791516e-08` ->
`3.872559e-08`, t 95% `[+3.36e-9, +4.39e-9]`, sign p `0.0000`, cost unchanged at
`+0.000044 B`).

**The λ sweep settles the Skeptic's PROVISIONAL.** Gains against F21:

| λ-scale | gain |
|---:|---|
| 0.0 (bare core) | 1.0696× |
| 0.5 | 1.1964× |
| **1.0 (our table)** | **1.2371×** |
| 2.0 | 1.0344× |

Optimum at λ-scale **0.794**, peak **1.2506×** — so our label-free extrapolation
lands **within 1.1% of the peak**, despite the 8.89× window spread that
condemned its value. The uncertainty in λ does not propagate, because the
objective is flat near its optimum. **We are NOT retuning to 0.794**: that was
measured on P01, the corpus we score on, and 1.1% is not worth a constant
fitted on evaluation data.

Sensitivity is **asymmetric**: halving λ costs 3.3%, doubling costs 16.4%, and
λ=2 is worse than the bare core. Err low.

**The attribution is confirmed**: λ carries about two thirds of the log-gain and
the bare core 1.07×. The accurate headline is "κ4 regeneration with a label-free
λ", not "per-neuron core".

### P2 is implemented — `estimator_k3_v31.py`, and it carries P1 as well

Both identities were proved **before any estimator code existed**:
reconstruction (`1.9e-16` float64, 3.7 float32 ulps on real F21 state at its own
birth site) and the `(2,1)` regrouping (`6.8e-16`, arbitrary residual).
`local_sim/p2_build_v31.py` emits three arms from v30 by asserted-count patches;
the patches are **disjoint** from v30's, so this is layering, not merging.

**The price, MEASURED at the live shape.** An independent Skeptic metered both
arms at 1024x16 inside one `BudgetContext` rather than trusting the hand model,
and that measurement is now the number of record:

| arm | counted FLOPs | C/B |
|---|---:|---:|
| v30 (control) | `928,411,199,060` | `0.422208` |
| **v31, rank 32** | **`661,806,923,368`** | **`0.300970`** |
| v31 dense (full rank) | `1,660,083,133,052` | `0.754934` |

**Ratio `0.7128` — a 28.7% saving. Cost-only score gain `1.4028x`.** v30's C/B
reproduces F21's graded `0.4222` to five decimals, so the instrument is sound.

The count behind it is confirmed: F21 issues **6** k-scaling products a layer
(3 transports + 3 contractions), P2 issues **4**. Independently corroborated
without reading any source — v30's measured total against the shipped 7/6 model
implies a Strassen discount of `0.5406`, *below the theoretical depth-4 floor of
`0.5819`*, which is impossible; against 6/5 it implies `0.6222`, matching the
chain's own L4/L0 census `0.62303` to 0.13%.

> **I MISPRICED THIS A FOURTH TIME, and the Skeptic caught two errors that
> cancelled.** My hand model said 31.3% and `C/B 0.2999`. It under-billed the
> thin terms by exactly 2x (charging `k·n²·r` where flopscope's own convention
> is `n·n·(2n−1)`) and omitted `_p2_compress` entirely — eight cost centres, not
> the four the patch comment lists. Two further errors pushed the other way: a
> nominal ratio applied to a Strassen-discounted total, and the K2 anchor held
> at its nominal value when the measured total has it discounted too. Net error
> 0.35%. **Right answer, wrong working** — and a model that is wrong twice and
> cancels is the same hazard wearing a better coat. The corrected model gives
> 30.0%; the measurement gives 28.7%. **Quote the measurement.**
>
> The earlier mispricings (27.6%, 41.5%, 32.3%) came from reading the plan
> instead of the source. F21's own `_matmuls` bills an aged lineage **7**, a
> phantom left from before `_dslice21`'s two permutations were found to be the
> same product, and that is where "7→4" came from. **Count the products in the
> source, then meter them.**

### The finding worth remembering: a deterministic sketch is not a free win

The rank compression needs a sketch. 504aldo use `W[:, :r]`, the layer's leading
weight columns (`estimator_v16b.py:561`), and we copied it — at He init those
columns are i.i.d. Gaussian, so no seed is needed and hard rule 2 is satisfied
without sampling anything. **That reasoning was incomplete.** Relabelling the
hidden neurons permutes those columns, so "the first `r`" selects a different
subspace and the truncation error lands elsewhere. The estimator stops being
permutation-equivariant — a property the true answer has.

`local_sim/edge_cases.py::inv_permutation` caught it before the screen ran:

| shape | v30 | v31, `W[:, :r]` | v31, fixed |
|---|---|---|---|
| 256x32 permute layer 7 | `4.1e-13` | `2.7e-10` | `5.2e-13` |
| 256x32 permute layer 20 | `1.3e-13` | `2.2e-10` | `8.4e-13` |
| **128x8 permute layer 1** | `2.2e-14` | **`1.5e-08` = 1.07%, FAILED** | `1.5e-14` |
| 256x32 permute layer 30 | `1.3e-14` | `8.1e-12` | `1.5e-14` |

**The dense arm discriminated it.** Same code at full rank: 30/30, equivariant
to `3.0e-13`. At full rank the projector is the identity on `range(R)`, so the
sketch cannot reach the output — which exonerates the reconstruction, the
regrouping and every `Z`/`L` side convention in one run. The arm was specified
to answer a numerical question and answered a structural one.

**The fix is free**: sketch with the residual's own strongest columns,
`R[:, argsort((R*R).sum(0))[-r:]]`. Equivariant because the norms permute with
the columns, so the selected *set* is unchanged and only the projector reaches
the output. Still deterministic. Measured side by side under a random
permutation: **`8.8e-01` for the column slice, `3.2e-07` for this one.**

This is the first place our re-implementation is **better** than the oracle
rather than merely independent of it, and an instrument found it, not a closer
reading.

**What it is NOT: a better approximation.** I wrote that the rank-32 deviation
"got slightly smaller, which is the expected direction". The Skeptic ran the
ablation — three builds differing only in which columns are selected — and at
256x16 the *strongest* columns are the **worst** of the three. The power
iteration washes the sketch out; the selection is immaterial to accuracy and
the ordering flips with shape. That `2.0e-4 -> 1.7e-4` was one draw from a noise
band, and it is a story I would have told whichever way it went. **The
equivariance justification is the real one and it is sufficient.**

### v31's standing

- `--parity`, five claims: control == v30 **bitwise** at identical counted
  FLOPs; dense arm `1.5–1.9e-7`; rank-32 arm `1.7–2.6e-4`; both arms pick the
  same `gens` at the live shape; **equivariance within 1.3× of the control's own
  float32 floor** (bar 100×). The Skeptic rebuilt the old sketch and confirmed
  the gate bites: `5.4e+05×` and `1.8e+05×`, both ~2,000× past the bar.
- `whest validate` **81 ms**; `edge_cases` **30/30**.
- **No `auto-routed` warning** — v31 does not carry the release blocker that
  failed E14's memory gate. v31 adds **zero** warnings over v30; the one v30
  adds over F21 (`op 'diagonal' is not symmetry-aware`) is P1's κ4 transport.
- **Residual: prediction 5 is SUPPORTED, measured, before the screen.** v31
  bills **12,665 operations** against the `22,000` gate (1.74× margin) — **9.1%
  FEWER than v30's 13,936**. At the frozen `13.4 µs/op` that forecasts a graded
  residual of `0.1697 s` against the `0.4 s` cliff, **2.36× headroom**. The
  layout shuttles I was least confident about cost operations *down*, not up.
- **Accuracy not yet measured.** Contract and gate: `experiments/LOG.md`,
  2026-09-15, *CONTRACT (Rule 1): Slice 6b P2*, with its pre-data amendment.
  Screen queued as job **`1364`**; gate **`≥1.20×`** with the 95% studentised-t
  interval strictly above 0.

### v30's certification

Job `1352` ran 1h08m, **passed 46 checks**, and was REFUSED by one:
`this run changed no tracked source under estimator.py/scripts/local_sim` —
because I edited the P2 generator while it ran. That is my error, not v30's.
Two sections were named SKIPs and remain real gaps: the Linux 8-GiB
enforcement, and the official subprocess smoke (`--skip-smoke`). Re-queued on a
clean tree as job **`1365`**.

**Nothing here is a submission, a promotion, a DECIDE block, or a vault touch.**


**Next:** finish v30's evidence path (certify, Linux memory gate, mini run),
then **implement P2** — its algebra is proved, its price is known, and it is the
larger remaining lever.

---

## September 15 (earlier): Slice 6a P1 WORKS — 1.2371x on 24 of 24, Skeptic pending

| arm | raw MSE | C/B | score |
|---|---|---|---|
| F21 | `4.791516e-08` | 0.422214 | `2.023044e-08` |
| **v30** | **`3.872559e-08`** | 0.422265 | **`1.635247e-08`** |

Paired difference `+3.877963e-09`, t 95% `[+3.3625e-09, +4.3935e-09]`, better on
**24/24**, sign p `0.0000`. Cost essentially unchanged (`+0.000044 B`). Label
noise `-0.01%`. Design ratio `0.80831`; against F21's graded `1.5957e-8` that
projects to **`1.29e-8`** — a DESIGN-TIER projection, not a public run.

**The pre-registered prediction is still FALSIFIED**: raw `2.1e-8` predicted,
`3.87e-8` measured. The gain is real but **1.24x, not ~1.8x**. 504aldo report
**1.97x** from the same mechanism, so about half their gain is missing, and
both likely causes are identified: the **`q31` consumer is absent** (F21's Wick
table has no `(3,1)` entry to receive it) and **λ is an initialiser** whose
1024 value an earlier Skeptic rejected as window-dependent.

**What separated this from the earlier 9.2x regression** was one coefficient
family: the birth is the **r=1 matrix-core** harmonic projection, not a rescaled
r=2 scalar. Settled in one line of the parity oracle, cloned with approval and
recorded in [PROVENANCE](PROVENANCE_504aldo.md) with a consultation table.

**Standing:** Rule 2 Skeptic running. This licenses a DECIDE-tier contract and
nothing else — no promotion, submission, designation or public run. F21
`#330950` remains the standing candidate.

**Next, in order:** (1) Skeptic verdict; (2) the `q31` consumer, or P2 — see
[the P2 design note](../experiments/2026-09-14_slice6a_p1/p2_design_note.md),
which prices P2 at ~27.6% of the tower with their shipped rank 32 at +0.9% raw.

---

## September 15 (earlier): the P1 regression that located the projection bug

**v30 is 9.2x worse than F21 on 24 design networks.** Pre-registered raw
`2.1e-8` (range `1.9-2.4e-8`); measured `4.428424e-07` against F21's
`4.791516e-08`. Paired difference `-1.667665e-07`, t 95%
`[-1.7936e-07, -1.5417e-07]`, better on **0 of 24**, sign p `0.0000`. Label
noise contributes `0.00%`, so this is the true functions. The cost half was met
exactly: `+0.000044 B`.

**Cause located by measurement.** At `λ = 0` the per-neuron core is not inert —
it still differs by `5e-3`. Three normalisation guesses gave three residuals
(`5e-3`, `1.2e-3`, `3e-2`), so I stopped guessing and instrumented both files.
The ratio `mean(g_out)/c4_out` is **exactly 1.000000 at layer 0** and compounds
at **~1.5-1.7x per layer** — the transport applies `metric` a second time.

**F21 never transports `c4`**: it re-derives the scalar each layer and scales at
consumption. A per-neuron core that flows is a *different object*, and its
transport normalisation cannot be fixed by matching an untransported scalar.

**THE DECISION, and it is Samuel's.** Two routes:

1. **Derive** how `_proj_coef_d4_r2(n)` and the `metric` convention generalise
   to a transported per-neuron core. Self-contained, slower, real research.
2. **Clone the oracle.** Handoff §1 authorises `external/504aldo/` as a parity
   oracle only; their V17 carries this core with a working convention and would
   settle it in minutes. **Authorised in writing, not taken** — it is an
   outward-facing fetch of a competitor's code and I flagged earlier that I had
   left it alone deliberately.

This is the exact ambiguity the independent Skeptic named earlier: whether
upstream's degree-4 core is the same normalised object as 504aldo's consumed
`G` cannot be settled without their code in the repository.

**Unaffected and still standing:** the `+0.000044 B` cost model; P1b struck
because F21's term table already routes κ4 into the κ3 births (`1/24` and
`1/4`); P2's feasibility (separable part ~101% of `||S21||` and free, rank-16
remainder 1.7-7.7%).

---

## September 14 (night): Slice 6 is the operative plan; 6a step 1 is DONE

`docs/CLAUDE_CODE_HANDOFF_2026-09-14.md` is the operative contract. It replaces
`E14_REVISED_PLAN.md`, `E14_TREE_CANDIDATE_DRAFT.md` and every earlier Sep-14
next-action below. **E14 is stopped** — its label-free Gaussian-tree λ was
measured to be a different object from the one that produces the κ4 gain
(opposite sign from layer 4 on, R² ≤ 0.2, depth-profile Pearson r = −0.01;
`experiments/2026-09-14_exp_lambda/`). Its consumer and feed plumbing is kept
for P1; its λ is discarded.

**Where we stand on the board:** F21 `#330950`, adjusted **1.5957e-8**, rank 62.
Leaders 2.5–2.8e-9. A public MIT baseline (504aldo v29) sits at 5.4e-9 and its
clones at 4.2e-9. The family is right; the product is three times off.

### Done this session

| what | evidence |
|---|---|
| **§0.5 `local_sim/predict_public.py`** — project a public score from a paired local mini run against F21's graded anchor, every line tagged MATH / LOCAL / SERVER_PREDICTED | `test_predict_public.py` **39/39**; the real `#330950` report replays to `1.595740e-8` |
| **§2.1 our own κ4 λ, label-free, 24 fresh P03 nets** | job `1305`, `experiments/2026-09-14_slice6a_lambda/` |
| **§1 `docs/PROVENANCE_504aldo.md`** — pinned commit, five mechanism rows, what we refuse to take, and the rules §6 question **drafted and UNSENT** | — |

### The two results that change the plan

**1. The AUGMENT route to λ has a hard memory wall at about width 175.**
Unfactored AUGMENT at `k_max=3` keeps κ3 as a dense `(n,n,n)`. Measured peak
**retained state alone is ≥133 GiB at 1024×16 — 4.2× the card**, before any
transient, derived from the state inventory. The handoff's "dump the κ4 core at
1024×16" is not executable.

*(Three peak figures quoted here earlier — 1.61 GiB, 12.44 GiB and an OOM at
n=192 — were **withdrawn**: they came from an ad-hoc shell run that was never
saved as an artifact, and "the law is confirmed by the failure that tests it"
was false, since an OOM at n=192 is consistent with any exponent ≥ 2.33. The
retained-state figure needs none of them.)*

**2. λ: the mechanism is confirmed, the VALUE at 1024 is not. The original
headline here was RETRACTED after an independent Skeptic REJECT.**

What stands: the fit's own R² (κ4 core on `C_off`) is **0.624–0.843**, rising
with depth, essentially width-independent, per-net unimodal and tight, and
strengthened by a clean `n=160` arm. A coefficient explaining 62–84% of that
core's variance at every reachable width **is a real mechanism**, and the
pre-registered *"R² ≥ 0.6 by layer 4"* is **CONFIRMED**.

What was withdrawn: *"within a factor of two of 504aldo's at every layer"*. The
pre-registered gate was **15%**, it **FAILED**, and "factor of two" was a
criterion chosen after seeing the answer — which also cannot fail (a random
permutation of their table passes 14.5/15; a single constant passes 15/15).
λ(1024) moves by **8.89×** in the median across leave-one-out fit windows, so
its value is set by the window and not by the data.

A follow-up at `n=160` showed the extrapolation family is sound at a **1.25×
step** (median error 3.00%, 15/15 within 15%) and that dropping the "dirty"
`n=64` width makes it **worse**, not better. Both facts are in
`experiments/2026-09-14_slice6a_lambda/followup_result.md`.

**`lam_own.npz` must NOT be used as a fitted constant.** The route to a usable λ
is one scalar per layer fitted end to end against score on fresh design
networks, initialised from this shape.

| pre-registered prediction | verdict |
|---|---|
| strategist: λ within 15% per layer at l=1–14 | **FALSIFIED** — no window exceeds 6 of 15 |
| strategist: R² ≥ 0.6 by layer 4 | **CONFIRMED** — 0.692 at n=128, 0.744+ at n=160 |
| Samuel: "I don't know" | recorded as given |

### Exact next actions, in order

*Rewritten 2026-09-15. The previous list opened with "P1 in
`estimator_k3_v30.py` (does not exist yet)", which had been false for a day —
v30 exists, is measured, and is in certification. A stale next-action list is
worse than none, because it is the first thing the next driver reads.*

1. **Read job `1352`'s verdict** — v30's certification, the first time the
   evidence path has been walked for that file. Running since 2026-09-15
   16:29 UTC; at §4 of 12 after an hour.
2. **Read job `1361`'s verdict** — the P2 design screen, three arms
   (`v31_control` = v30, `v31_dense` = full rank, `v31` = rank 32) × 24 P01
   networks. Contract and gate: `experiments/LOG.md`, 2026-09-15, *CONTRACT
   (Rule 1): Slice 6b P2*. **Gate `>= 1.20x` with the 95% studentised-t
   interval strictly above 0**, chosen before the data existed.
3. **Rule 2 Skeptic on P2**, with the contract, the generator diff, the
   `--parity` transcript and the raw `score.json` — never a prose summary.
   P1's Skeptic returned PROVISIONAL and found a real shape-genericity crash;
   this one has a live target too, in the layout shuttles.
4. **Then v30's remaining evidence path**: the Linux memory gate (the 8,192 MB
   limit is unenforceable on Windows — it is accounting here, never a test
   passed) and a public mini run projected through `predict_public.py`.
   Only then is a graded canary worth Samuel's per-submission approval.
5. **Slices 6c (N1 kill test) and 6d (raw levers)** — untouched, and both are
   unblocked.
6. **The `q31` consumer** is a **four-row term-table extension plus a new
   `(3,1)` slice**, not one line. F21's table is upstream's frozen SIMPLE
   enumeration and does not contain `(3,1)`, so adding it diverges from the
   pinned reference and **needs its own contract**.
   `experiments/2026-09-14_slice6a_p1/q31_is_a_table_extension.md`.

**§0.1 is DONE** (verified 2026-09-15, all three sites): the local-seconds
residual no longer refuses in `certify_candidate.py` §10 (ADVISORY),
`preflight.ps1` step 16 ("THE LOCAL RESIDUAL NO LONGER REFUSES"), or AGENTS
hard rule 6 ("residual advisory, cost gates on ops"). The binding cost check
is the operation count against `official_config.PHASE2_OP_COUNT_GATE`
(22,000), which is asserted in that module. Historical refusals that quoted
the old line stand and are not amended.

**§0.6 is still partly owed**: E14 is stopped in this document, but both E14
files are still present and still referenced by older sections below.

**Rule 2 Skeptic on Slice 6a: DONE** — REJECT on λ, PROVISIONAL on P1.
Retraction and follow-up in LOG and `experiments/2026-09-14_slice6a_lambda/`.
The PROVISIONAL's main worry — that λ's 8.89x window spread invalidated the
gain — was answered by the λ sweep: the objective is flat near its optimum and
our label-free table sits within **1.1%** of peak.

### Standing constraints

Samuel approves every submission individually and designates alone. The rules
§6 question is drafted and **unsent**, pending his approval of the exact
wording. 504aldo's code is a **parity oracle only**; their fitted λ table is
never imported. No DECIDE block is spent, the vault is untouched, P08/P09/P10
stay spent, and the contribution write-up remains stale.

---

## September 14 full-float64 E14 tree: release hold after independent checks

Samuel's new prediction was “i dont know” for raw MSE, counted FLOPs, and residual/memory feasibility; it was committed before the separate full-float64 source. The candidate `estimator_e14_tree_f64.py` passes a bounded independent Gaussian-start dense recurrence (696 n3/4 checks, max 9.95e-14 against 1e-8), with exact frozen-F21 disabled output. This proves neither accuracy nor release feasibility.

At 1024x16 the complete local bill is **2,029,122,417,342 counted FLOPs** versus F21 **928,457,637,478** (2.18548x). It leaves 169.9 billion under 2**41, but needs a **54.24% raw-MSE reduction** to tie F21's score at equal targets; no fresh truth was measured. The generic chooser underprojects this actual bill by 72.19 billion FLOPs. The exact candidate **fails the enforced 8-GiB Linux address-space cap** at its first prediction: a 1.03-GiB float64 Strassen workspace allocation raises MemoryError with 8168.4 MiB VmPeak. Independent Skeptics accepted the bounded numerical and memory results, and the complete local cost only; see `experiments/LOG.md` and [release decision](E14_F64_RELEASE_DECISION.md).

Two label-free installed-subprocess probes report local residuals 16.91s and 16.73s against 0.4s, both failed rows. The direct parent supplied f64 weights, but the installed worker forcibly converts them to f32, then the candidate recasts them. A fresh Skeptic accepts this **f32-delivery/cast-path host diagnostic** and rejects its proposed f64-delivery interpretation. No server residual, truth-MSE, paired arithmetic score, public CONFIRM, vault touch or release pass exists. Frozen F21 remains the explicit comparator. A default Git-snapshot preflight #1283 was RED solely because checkout line endings changed exact-byte commitments; corrected live-worktree workerq #1284 is **GREEN 12/16 with four Full skips**, including the seed ledger and official FLOP anchor. This infrastructure pass cannot rescue the candidate's memory gate.

**Exact next decision:** review the [new strategist packet](research/2026-09-14_e14_f64_frontier_packet.md) for a separately contracted precision/workspace intervention, or keep E14 on hold and prioritize the contribution write-up. Any new intervention needs new Samuel predictions before implementation, complete cost and memory evidence, and only then a fresh committed DECIDE preregistration. Do not submit or designate this candidate.

## September 14 earlier f32 E14 tree candidate: complete cost, numerical gate held

**Earlier f32 precision follow-up.** The first observed f32-versus-dense difference on the saved n3 negative fixture is already layer-0 covariance multiplication (4.62649e-7), before any carried-state handoff. By layer 2, the base fourth moment differs 2.12720e-5 before the tree hyperedge and the final pK4 differs 2.69219e-5. A fresh Skeptic accepts this localization only provisionally because the replay does not emulate every active f32 operation. On the same saved f32-rounded weights, the unchanged active recurrence run in f64 agrees with an independently carried dense f64 recurrence across compared pK/K/K3 fields to 4.9738e-14; a separate Skeptic accepts that narrow result. It does **not** make the f32 candidate pass. The separate full-f64 implementation and its later failure are recorded above. Affected root preflight #1274 passed 12/16 with four Full skips after those earlier IMPORTANT probes.

Samuel answered “i dont know” for the prospective tree candidate's raw-MSE, counted-FLOP and residual-time predictions; the operative engineering contract records it before implementation. The uncompressed F21 tree candidate is `estimator_e14_tree.py`, kept separate from the frozen shipped file. Its actual 1024x16 label-free counted ledger on common canonical weights is **1,014,599,318,825 FLOPs**, +9.27793341% over F21's 928,457,637,478; the active path fits 2**41, but cost alone has no accuracy or residual-time implication. At equal target error it would need 8.4902% lower raw MSE just to tie F21 score, or 12.8478% lower to clear the proposed 1.05 arithmetic-gain gate. Independent Skeptic accepted this cost scope.

The tiny supplied-state recurrence, direct tree tensors, disabled F21 path and generic age/fallback checks have bounded passes. A later independent Gaussian-start dense n3/4 recurrence **fails the predeclared 2e-5 absolute f32 numerical bound** on the preserved negative-variance fixture: layer 2 pK(4,) index 1 differs by 2.692194e-5. The fresh Skeptic confirms the failure, while leaving its cause provisional because prior f32-versus-f64 state rounding may amplify. The exact operands and all intermediate artifacts are retained under `experiments/e14_tree_candidate/`; no tolerance was widened. The same-weights, layerwise f32-boundary diagnostic is the next scientific discriminator. Until a prospective numerical regime or corrected implementation passes independent review, this spelling is held from fresh DECIDE truth, CONFIRM, and release promotion. No new seed block, vault touch, public fitting, submission or designation has occurred. Workerq #1270 completed three Linux predictions under 8192 MiB (5574.9 MB peak address space) but its overall memory gate is FAIL from candidate stderr. An unrelated replay-test filter defect made root preflight #1271 red; the independently reviewed narrow fix `aa4fc22` restored #1272 to GREEN 12/16 with four Full skips. Neither result certifies this candidate. F21 remains the explicit score comparator, not a claim of competition success. The [release hold decision](E14_TREE_RELEASE_DECISION.md) is current.

## September 14 E14 revised execution: bounded proofs complete, candidate pending

The revised E14 independent prerequisites are durable in `experiments/LOG.md` and commits `2128550` (mixed residual falsifier) and `fb96636` (scalar-feed proof). Slice A's unchanged no-basis control replays frozen F21 on exposed B3/B4, and its full-rank f64 query algebra matches a dense oracle, but the mixed f32 operational path **fails** the unchanged B4 numerical allowance: final drift energy 0.3663149 versus 2e-10, with first consumed difference at layer 2. A fresh Skeptic rejected operational conformance. The rank64 615.26B FLOP figure remains a static upper model, with no executed compressed ledger, memory or residual-time result. No rank selection or numerical-regime change is justified.

Slice B's supplied-core scalar-metric J/U cancellation, shared A/P/y/r slices, active tiny f32 fused query and frozen disabled branch have a final **scoped independent ACCEPT**. The final 289-check raw probe independently reconstructs two frozen tiny F21 births and compares full direct/reduced supplied-Q births through two later supplied-state steps. Four earlier rejected/provisional revisions and their first falsifiers remain in `experiments/e14_slice_b/`. The accepted proof injects Q at a captured consumer boundary; it is not an active production estimator or a tested regeneration rule. No new MSE, counted candidate ledger, generic-shape budget fallback, memory or release certificate exists.

Affected root preflight workerq #1254 passed 12/16 enabled checks (four Full skips) with 315 unchanged before/after source hashes and 2,374.51 MiB measured peak under a 4 GiB/2 CPU reservation. Queued #1252 was canceled before launch solely to right-size its reservation from measured history. See `experiments/e14_revised_preflight/`. This is infrastructure regression evidence, not candidate certification.

**Exact next decision:** choose the specified tree-projected scalar-metric accuracy candidate on **uncompressed F21** as the first scientific intervention, separately from any shared-basis rank or numerical-regime change. The [prospective candidate draft](E14_TREE_CANDIDATE_DRAFT.md) makes the intervention, engineering falsifiers and proposed paired gate reviewable; it is not an operative preregistration. Before building or measuring it, complete a STRATEGIC contract with Samuel's new predictions, then retain all closure discarded terms and moment consumers, test active tiny/fallback paths, count the full actual ledger, and only then claim a new DECIDE block for paired arithmetic score evidence. Samuel's prediction response is pending. No fresh block, public fitting, vault touch, submission or designation occurred in this execution. F21 #330950 remains the accepted score comparator with its separate local release refusal; the contribution write-up remains stale.

## Next decision — independent E14 strategy review

Samuel requested a fresh review of the worker's decision point. The
[strategy assessment](research/2026-09-14_e14_strategy_review.md) accepts the
bounded worker evidence and supplies a [revised execution plan](E14_REVISED_PLAN.md).
It supersedes the remaining dependency order below, not the recorded failures.
The [independent worker audit](research/2026-09-14_e14_independent_review.md)
reproduced21 lightweight journal/release tests and309 source hashes. No new
estimator, rank or protected-data experiment was run in this review.

Two concrete next slices are independent: (A) packed f32 full residual plus
f64 shared-basis direct queries under the specified complete cost model;
(B) supplied-core scalar-metric K4 feed cancellation and sharing with existing
F21 factors, verified against full tiny tensors. Full-rank correctness does
not require full-rank cost savings. The accuracy identity need not wait for
compact QR. Both remain unimplemented proposals and require their normal
prediction/contract, tests and Skeptic before reliance. The
[strategy Skeptic](research/2026-09-14_e14_strategy_skeptic.md) accepts only this
research scope. Later rank, closure or numerical-regime selection is STRATEGIC
and needs new Samuel predictions. No historical tolerance or release rule is
changed. Full residual preservation is the first isolation step, not a permanent
winning constraint; later measured residual cost may motivate its own intervention.

F21#330950 remains primary. No new score, release pass, submission, designation,
vault touch or contribution claim exists. Stage0 is complete; further general
infrastructure work is not the critical path absent a concrete defect.

## Current execution — September 14 E14 prerequisites

Samuel authorized implementing the reviewed plan, including parallel bounded
tasks and local commits, while preserving new strategic predictions and per-entry
submission approval. See [E14 stage/status record](E14_EXECUTION_STATUS.md).
Stage0 repairs are implemented; base infrastructure is independently accepted.
The optional strict numerical-evidence consumer passes final affected preflight
workerq1230 (12 enabled steps,4skipped;309 source hashes unchanged). Production-truth pilot1219 and exact-byte default preflight1222
passed. B4 float64 mathematics passes841/841 and independent review. Original
actual-f32 compact conformance still fails; a mixed-f64 representation with
unchanged reference-order f32 operators reproduces all four fixed outputs,
but is a dense wrapper with no savings and imperfect exact-zero recovery.
The literal dense bridge and an ordinary-f64 direct-query spelling fail
necessary cost screens; independent review accepts those scoped rejections. No rank test
is justified by these results. The smallest missing artifact is a conformant
full-rank production-precision residual/query kernel with complete costs; see
[the self-contained strategist packet](research/2026-09-14_e14_strategist_packet.md).
The coherent connected-tree K4 definition passed static independent review;
its literal dense implementation exceeds the FLOP budget before all overhead.
No new candidate has earned promotion. Root estimator.py, frozen archives,
HUMAN_MODEL, spent blocks and vault touch state are unchanged. F21#330950
remains primary control; its prior local cap refusal is not waived.

## Reviewed strategy — September 14 research review

Samuel requested and received a multi-agent repository/validation/literature review
and a live competition review through his Chrome. The returned
[research strategy](RESEARCH_STRATEGY_2026-09-14.md) and
[implementation plan](IMPLEMENTATION_PLAN_2026-09-14.md) recommend keeping the
cumulant family while separating operator parity, all-age compression, and one
coherent accuracy extension. A winning implementation is **not demonstrated**.
The next proposed action is the existing fixed B4 float64-versus-float64 precision
discriminator, followed by actual-precision conformance; no new labels are needed.
Instrument work should be limited to candidate identity, numerical sensitivity,
crash-safe evidence and production-truth precision provenance. These are proposals:
apply the existing IMPORTANT/STRATEGIC contract before implementing or measuring.

F21 #330950 remains the latest successful graded candidate; root `estimator.py`
is the historical v2.6.1 baseline. Local capped-mini/Full refusal and the one-time
server diagnostic pass remain separate. The leader detail page refines the dated
target to `2.540e-9` (board rounds `2.50e-9`): F21 is about 6.28x higher. At its
unchanged public raw error, the 0.1 floor still gives `3.7794e-9`; reaching the
dated leader there requires about 32.79% lower raw error. Public truth noise and
private performance remain unknown. No code, gate, protected-data access,
submission, designation or write-up claim was changed by this review.

The supporting [competition notes](research/2026-09-14_competition_review.md),
[architecture audit](research/2026-09-14_architecture_audit.md),
[validation audit](research/2026-09-14_validation_audit.md), and
[frontier review](research/2026-09-14_frontier_research.md) preserve source paths,
primary citations, unknowns and concrete acceptance tests. See the separate
[research Skeptic verdict](research/2026-09-14_research_skeptic.md) for the scope
of independent acceptance. The entries below are dated supporting chronology;
older statements of pending status do not supersede the latest result.

## Dated supporting chronology

**September 14 F21 diagnostic #330950 GRADED, zero failures; strategist
decision now concerns larger gains and private selection.** Samuel
authorized one exception to the failed local clean-mini gate and one
submission without designation. The [official graded report](../experiments/run_reports/f21_diagnostic_330950_graded_report.json)
SHA256 `4dc5c2f6f1d06220921efb74b71a3b5bee930e141e7319af0e0d55028100e5f3`
records 100/100 completed MLPs, no estimator, budget, wall or residual-cap
failures, and exactly `928,457,637,478` counted FLOPs per MLP. Worst
measured graded residual is `0.15623225` s against 0.4 s, or **2.56029x
headroom**; worst predict wall is `93.1597153` s against 120 s. The
predeclared server-feasibility diagnostic therefore passes for #330950.
Original L4 #330290 and F21 share the 50 exposed public score rows:
official adjusted score improves from `1.8359814201e-8` to
`1.5957295444e-8`, a **1.150559x** descriptive paired ratio; 50/50
displayed adjusted rows favor F21. This is read-only public CONFIRM;
the independent fresh P10 DECIDE remains the binding method result.
The [fresh graded Skeptic](../experiments/2026-09-13_p10/f21_graded_skeptic.md)
ACCEPTS zero-failure grade and these scoped comparisons. Its exact-byte
server-attribution verdict is PROVISIONAL because the CLI and graded page
expose no uploaded archive hash; local tar SHA256
`4f3cc6f440e4cb66ea86a0904c00e182eb7ecf5aaab0ee64c73efb7d22904775`
and source/member SHA256
`d77dab0e2c740cc9f6cd7d9172adacc8da731cc5b40ca87798f3fb006e475a94`
are verified and the graded FLOP fingerprint matches. The one-time
exception does **not** rewrite the correct 100/100 local capped-mini
failure or Full REFUSED certificate. Private score, public truth floor and
prize eligibility remain unobserved. No second submission or designation
is authorized or made. The [dated board snapshot](../experiments/2026-09-13_p10/board_after_f21_330950.json)
shows 2.5e-9 leading and our rounded 1.6e-8 at rank58/156; board rank
is not the prize. The self-contained
[September-14 strategist packet](STRATEGIST_PACKET_2026-09-14.md), SHA256
`ee0977c21b727dc462be680c2b5cbbb197f0c694a61dacc6aba157c8e09fcf60`,
is ready for Samuel to take to a strategist. Its fresh independent
[packet Skeptic](../experiments/2026-09-13_p10/f21_strategist_packet_skeptic.md)
accepted the revised no-repo decision request at commit `c84ec7d` after
correcting the per-network score formula, MiB unit and parameterized
cost scope. Await the strategist's returned decision before selecting a
new strategic experiment; preregister and get Samuel's predictions before
any fresh block or source intervention.

**September 14 F21 passes the fresh paired score gate, but normal submission
release is BLOCKED: Full certificate REFUSED and the capped official mini
failed 100/100 networks on the local residual-time cap.**
Samuel's three new
"I don't know" predictions were committed at `ce853f7` before any P10 data.
The fresh SINGLE_USE 128-network truth claim was committed at `883bd45`;
two `2**24` replicas passed the pre-score floor screen and were finalized
at artifact SHA256
`9026b4ea66900695929cc9a32f1a801258733c176ae100b079e979907a04de30`.
The one protected P10 read is spent 1/1. Workerq 1185 completed all 128
original L4, frozen F21 and original L2 arms, exit 0, with no lifecycle
failure flags. Its exact [DECIDE result](../experiments/2026-09-13_p10/decide_P10.json)
SHA256 is
`d868e4335168fa3ff1159c594be3166b61f81ebaba8b93faa846f0b77c23d79b`.
The original-L4 versus F21 official-precision forecast at the conservative
observed `+2**24` server FLOP offset gives paired **arithmetic gain
`1.150566x`**, exceeding the predeclared `1.05` gate. The paired 95% t
interval on L4 minus F21 score is
`[2.4582563842e-9, 2.5842673161e-9]`, strictly positive; 128/128
pairs favor F21. Mean raw MSE is L4 `4.1821425170e-8` and F21
`4.1821088467e-8`, beside midpoint truth floor `2.2363657945e-9`;
the useful gain comes from `13.08574%` measured counted-FLOP saving.
L2 is robustness context only. The builder and fresh independent result
Skeptic `9c94b6e` separately recomputed the canonical gate from raw rows;
the Skeptic ACCEPTED only the DECIDE promotion evidence.
The candidate remains frozen at SHA256
`d77dab0e2c740cc9f6cd7d9172adacc8da731cc5b40ca87798f3fb006e475a94`.
P10's scorer disabled both local scoring timing caps exactly as registered;
this score is not a deployment pass. Workerq 1200's full exact-byte
certificate reported MATH 41/41 and an applicable Linux v6 memory pass,
but LOCAL REFUSED: its two-row capped official subprocess smoke failed 2/2
on local residual-cap exhaustion. The 14,091-op server-residual model
passed only as a forecast. The separate **clean capped** 100-row official
subprocess mini, workerq 1201, has completed. Its report records
`n_failed_mlps: 100`, all `residual_wall_time_exhausted` at local residuals
7.312–9.633 s against 0.4 s; the other four failure categories and estimator
tracebacks are zero. Its `928,424,083,046` f32-mini counted FLOPs are below
budget. Thus the required clean-mini gate `n_failed_mlps: 0` fails,
confirming the Full local refusal. The workerq exit 0 means only that a
complete report was produced; failed zeroed outputs are not accuracy
evidence. Local residual timing is not a graded-server measurement, and the
op-count server forecast cannot waive the observed gate. The exact
[mini report](../experiments/2026-09-13_p10/mini_F21_capped_official.json)
SHA256 is `c43eb8104047a2d2d0967c73d39f30b205bbe2389da5f66d2d81b28203d33808`;
the [workerq receipt](../experiments/2026-09-13_p10/workerq_1201_receipt.json)
and full terminal log are retained. F21 is a demonstrated score improvement,
but **not submission-ready** under the normal release contract. A no-cap mini
cannot substitute. No submission, external message or designation is authorized.
The fresh independent [mini Skeptic](../experiments/2026-09-13_p10/f21_mini_skeptic.md)
ACCEPTS the 100/100 local refusal and REJECTS normal release certification;
its verdict is commit `021f643`. The graded-server F21 residual and private
success remain unknown.
Next action: retain this scoped refusal. Any one-off release exception would
require Samuel's explicit decision on the exact evidence; otherwise use
standing L4/L2 while investigating a bounded residual-time remedy. No further
P10 or vault read is permitted.

**September 13 F21 lifecycle diagnostic closed; P09 still has no verdict.**
The preregistered, label-free stress on the first 128 exposed P07 DESIGN
identities used frozen F21 SHA `d77dab0e...`, the P09 scorer's Phase-2 spec
with only the residual cap disabled, canonical mint and `f64_readonly` weights.
Workerq 1150 completed 128/128. Its sole failure was row 27,
`time_exhausted=true`, local wall `17,914.856` s; the Windows System log
records a `17,892.563`-s host sleep during this run, so that row is strongly
environment-confounded. All other 127 rows succeeded, maximum local wall
`71.860` s, each with `928,457,637,478` counted FLOPs; no estimator exception
or budget flag reproduced. This does **not** reveal P09's discarded failure
category, establish graded-host wall or residual behavior, produce an accuracy
score, or promote F21. The [raw diagnostic](../experiments/2026-09-13_f21_labelfree_stress.json),
[host sleep events](../experiments/2026-09-13_f21_labelfree/host_sleep_events.json),
and [fresh Skeptic verdict](../experiments/2026-09-13_f21_labelfree/skeptic_verdict.md)
are preserved. The Skeptic ACCEPTED this narrow diagnostic only. The first
affected preflight was RED on an undeclared deliberate P07 overlap; an
append-only pairing declaration repaired the ledger refusal, and workerq
1152 passed the retry 12/12. Both transcripts are retained. F21 remains
**unpromoted and not submission-ready**.

**F21 engineering and DESIGN passed; P09 DECIDE produced no verdict, September 13.** The separate L4-derived
candidate [estimator_k3_strassen_L4_f21.py](../estimator_k3_strassen_L4_f21.py)
is committed at SHA256 `d77dab0e2c740cc9f6cd7d9172adacc8da731cc5b40ca87798f3fb006e475a94`.
Its [exact diff](../experiments/2026-09-12_f21/parent_to_f21.diff) changes only
the specified `_dslice21` contraction and its local explanation. The frozen
L4 parent is [pinned separately](../experiments/2026-09-12_f21/estimator_L4_parent.py).
The exact tiny oracle and installed-flopscope 1024x16 trajectory pass; all
125 sampled fields pass, maximum normalized error `6.09644e-5` against
`2e-3`, and final relative RMS `7.30922e-7` against `1e-2`. The corrected
f64-readonly counted ledger is **928,457,637,478 FLOPs** versus original L4
**1,068,245,507,686**, a **13.08574% measured saving** at one 1024x16
network. The parent exactly matches the stored graded L4 ledger; F21 still
needs its own official subprocess evidence. Independent
[cost](../experiments/2026-09-12_f21/cost_skeptic.md) and
[full-depth trajectory](../experiments/2026-09-12_f21/full16_skeptic.md)
Skeptics ACCEPT these narrow engineering claims. The F21 contract and
Samuel's three "I don't know" predictions were committed before results.
The exposed-P01 DESIGN screen passed: F21/L4 paired official-precision
forecast score ratio `0.8691817714` against the predeclared `1.05`
maximum, with 24/24 score pairs favoring F21. Raw MSE is essentially
unchanged/slightly worse, so the gain is the measured FLOP reduction. The
[fresh DESIGN Skeptic](../experiments/2026-09-12_f21/f21_design_skeptic.md)
ACCEPTED only that screen; it is not a promotion decision. The repaired
candidate [Linux cap report](../experiments/2026-09-12_f21/f21_linux_cap_v6.json)
passed 23/23 at 8,192 MiB, and its [independent review](../experiments/2026-09-12_f21/f21_cap_v6_skeptic.md)
accepted the three-repeat local memory gate. Candidate release checks and
submission approval remain open; F21 is **not a ready submission**.

**Current gate:** Full preflight 1092 passed 16/16, repaired default 1095
passed 12/12, and P09-registry default preflight 1097 passed 12/12. P09 was
claimed once (128 seeds); workerq **1098** completed the two-`2**24`-replica
truth mint. Replica disagreement ratio was 1.0102 and predicted midpoint
truth floor about `2.2442e-9`. Workerq **1099 FAILED exit 1** after original
L4 and fused L4 each reached 128/128: at least one fused lifecycle row
failed, but the scorer discarded failed-row flags before writing the JSON.
L2 and `competition_score.decide(gate=1.05, interval="t")` never ran. The
failed seed and cause cannot be recovered; the P09 truth read is spent 1/1
and may not be repeated. The fresh
[result Skeptic](../experiments/2026-09-12_f21/p09_result_skeptic.md) REJECTED
promotion evidence, without calling this a statistical loss or scientific
kill. The read-only
[forensic note](../experiments/2026-09-12_f21/p09_failure_forensics.md)
records what the terminal artifacts can and cannot show. A minimal scorer
diagnostic now logs only failure flags/timing before the unchanged abort;
workerq **1141** passed its affected default preflight 12/12, and the
[diagnostic Skeptic](../experiments/2026-09-12_f21/p09_failure_diag_skeptic.md)
ACCEPTED only that repair. The frozen F21
selection remains SHA `d77dab0e...` against original L4. The separate compact
B0/B1/B2/B3 probes establish n4/depth4 tiny full-rank recurrence parity only,
not compression or score. A separately preregistered n8/depth6 B4 extension
failed its fixed `1e-5` parity gate: 85/841 checks failed, worst normalized
error `4.04473e-4`; all age/activity/rank checks passed. The raw artifact and
contract are in LOG. B4 affected-tree preflight passed 12/12 and its fresh
Skeptic accepted only that failed tiny gate; no wider or scientific closure
follows. P08 remains exhausted. No vault/public F21 labels,
fresh DECIDE result or submission approval exists.

**September 12, 2026: strategist review of worker d124234.** Then read LOG from the bottom. The preceding detailed chronology is [archived verbatim](archive/CURRENT_STATE_before_worker_review_2026-09-12.md); all original contracts and results remain in experiments/LOG.md.

## Next discriminator: fresh F21 decision after the scoped diagnostic

P07's single observed timeout was confounded by host sleep; the other 127
rows have no local lifecycle failure. Do not rerun P09, infer its category
from P07, or treat this exposed DESIGN diagnostic as a score. A new F21
DECIDE requires a newly claimed and preregistered single-use block, new
Samuel predictions, the binding paired arithmetic gate and an explicit
local-wall handling plan in its contract. Only a passed fresh gate can
justify candidate-specific full certification, official mini and submission
review. The optional one-row no-sleep P07 check suggested by the Skeptic
would remain diagnostic, never a DECIDE substitute.

Use a new candidate file. The best-public L4 parent is inside `submission-k3strassen-L4.tar.gz`: estimator.py SHA256 `052dd768e970c4bae831cff26f8ff15b0487f3b95eadf2fd3578e542aec34d58`. Its relevant functions match L2 by independent AST comparison. Preserve the conservative generation/budget chooser, scalar K4, precision, saturation and shape behavior. Do not edit frozen files or shipped estimator.py.

Read [STRATEGIST_RESPONSE](STRATEGIST_RESPONSE_2026-09-12.md) for equations and evidence, [GAME_PLAN](GAME_PLAN.md) for execution, and [CODEX_EXECUTION_HANDOFF](CODEX_EXECUTION_HANDOFF.md) for the task boundaries. The F21 cost and score thresholds were preregistered in LOG; engineering and exposed DESIGN passed, but no fresh DECIDE result exists.

## Scope changed explicitly

Samuel answered **“Include eligibility and candidate preparation.”** The old Stage-3 hold is lifted for those activities. Actual submission still needs individual approval; external messages and designation are not authorized. [Current eligibility review and unsent clarification](ELIGIBILITY_REVIEW_2026-09-12.md). Do not let an unresolved participant-derived reconstruction block preparation of an independently documented own-code change.

## Accepted evidence and actual blockers

| Area | Current status |
|---|---|
| Stage 0 | Scoped independent ACCEPT; Linux positive plus M1–M12 aggregate 13/13. This is instrument coverage, not a complete future candidate certificate. Reuse it unless an affected path or new failure requires retesting. |
| Preflight | Workerq 1092 passed Full 16/16; default 1095, P09 registry 1097, B4 1100, F21 scorer diagnostic 1141 and label-free stress retry 1152 passed 12/12. Initial label-free 1148 was RED on undeclared P07 pairing; raw failure and append-only repair remain in LOG. Candidate-specific full checks remain conditional on a valid fresh DECIDE. |
| Compact supplied-core implementation | Algebra/cost evidence remains scoped: about 0.372208 B and about 1,014 MiB on its supplied-core program. Not the cost of a complete physical signed-source method. |
| Existing compact off control | Already evaluated on 96 exposed P01 DESIGN networks: raw mean MSE 2.11130e-5, averaged-truth noise 9.31914e-9, score 7.83730e-6. Saved arithmetic independently checks out. It is not a matched L2 comparison and is not the working L2 recurrence. |
| New K4 arms | Frozen P01 arms failed their promotion gate. The source and covariance defects explain why scientific causation remains incomplete; do not fit them again unchanged. |
| Integrated signed source | n4 93/172; n8 186/187 with negative variance. Later positive-feed optimization preserves that failure. This path cannot support a full-width readiness claim. |
| Memory inventory | 93/93 accepted sampled n8/n16 inventories; width1024 named-payload model 715,844,864 bytes. Not a process bound or 8-GiB proof. Further symbolic lifetime work is not the next score bottleneck. |

The [validation audit](../experiments/2026-09-12_worker_review/validation_skeptic.md) recomputes existing score rows and artifact hashes. [Architecture review](research/2026-09-12_architecture_review.md) identifies the contraction and compact-recurrence gaps. No protected labels, seed claims, vault evaluation, candidate edits or heavy jobs were used for this review.

## Architecture and priorities

1. **First better candidate:** preserve the working L2/L4 full Wick recurrence and scalar K4; remove redundant D21 work. Compare fused L4 against original L4, with L2 as a robustness control. A cheaper L2 alone may still lose to L4.
2. **Larger savings:** full-rank compact-state bridge to the working recurrence, then one declared compression approximation. The current compact template uses Gaussian-only moments in places where L2 includes incoming corrections; its self-oracles did not establish L2 parity.
3. **Research extension:** coherent approximate K4 after a competitive baseline. Exact causal signed-law compression is deferred as the next-candidate strategy, not scientifically killed. Do not require a solution to that larger problem before preparing an independent working-K3 improvement.

A winning architecture is not yet demonstrated. The immediate change is a modest improvement hypothesis; larger gains still require faithful compression and/or better controlled corrections. Latest [primary-source update](research/2026-09-12_frontier_update.md): 504aldo and ARC code pins unchanged; ARC paper v2 supersedes previously inspected v1. Public tiny-shape tests bypass regenerated K4 and do not validate its target branch.

## Submission and data state

| Role | Frozen evidence |
|---|---|
| Best public | L4 #330290, dated adjusted 1.836e-8, worst displayed wall 114.6798 s. Archive/hash above. |
| Robust standing fallback | L2 #330566, dated adjusted 2.284e-8, 0/50 public failures, worst displayed wall 33.0009 s; SHA 390bc226bbf517df0110facfc0dc91c78ef670e8a49368a536cabf630057d1db. |
| Shipped baseline | estimator.py v2.6.1, archive-identical, unchanged. |

Displayed official noise floors are unavailable for those public scores; dated wall observations are not guarantees. Chrome September12 shows our best 1.84e-8 at rank 52 and two leaders at rounded 2.60e-9. These unpaired rounded means imply roughly a sevenfold gap, not a promotion statistic. [Current source check](research/2026-09-12_competition_update.md).

P01/P07 remain exposed reusable DESIGN. P01 has at least the seven documented registered reads plus prior history; never describe it as untouched. P08 is exhausted. New DECIDE needs a fresh claimed block and committed contract. Public CONFIRM is read-only. The sealed Phase-1 vault and HUMAN_MODEL remain untouched. Previous “I don't know” predictions belong to their contracts, not all future work.

Phase2 remains 1024x16, budget2^41, lambda0, residual0.4s, wall120s, memory8192MiB enforcement. Use the paired arithmetic gate and explicit truth floors. Local timing is not server authority; current counted-package parity does not remove the separately observed server offset. Registration/team freeze October 2, submissions October 17 23:59UTC, write-up October 24.

Before every heavy or >30-minute job, record the workload optimization review required by AGENTS. Use workerq with conservative measured RAM/VRAM/CPU; the machine is RTX5090/64GiB host. Prefer 16–24GiB when appropriate without weakening evidence, not as a minimum for small work. Do not change queue priority or underdeclare. Recheck live jobs before launch.

Contribution write-up remains stale/do-not-submit until claims receive their own review. New plan supersedes old execution ordering; historical experiments and scoped negatives remain intact.
