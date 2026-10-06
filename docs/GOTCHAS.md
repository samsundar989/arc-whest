# GOTCHAS.md — standing operational knowledge

## 2026-09-08 / 09-09 — FOUR `workerq` TRAPS, ONE JOB EACH

**1. A `workerq` job requests ONE GPU by default, even when it is pure numpy.**
A CPU-only job submitted without `--gpus 0` sits in QUEUED behind whatever holds
the card, reporting `waiting for 1 free GPU(s)`, and `workerq requests` cannot fix
it — that subcommand takes `--ram`, `--vram` and `--cpus` but **not** `--gpus`, so
the only remedy is `cancel` and resubmit. Every probe in `local_sim/` that meters
flopscope is CPU-only. Submit them as:

    workerq submit --project arc-whest --ram 8 --cpus 4 --gpus 0 -- uv run python ...

**2. The job snapshot silently DROPS large gitignored files.** `.gitignore` line 5
is `*.npz`, so every truth artifact is untracked. Small ones are carried into the
snapshot anyway; large ones are not, and the job dies on `FileNotFoundError` inside
`~/.local/state/gpuq/snapshots/<id>/`. Measured the same day:

    experiments/2026-08-26_p2_gt/gt_halfA.npz          3 MB   -> IN the snapshot
    experiments/2026-09-01_wp9_corrector/truth_P02.npz 35 MB  -> DROPPED

`workerq submit` prints the excluded paths in its own output, and that list is
worth reading rather than scrolling past. **A job that needs a large `.npz` must
run locally**, or the file must be staged somewhere the snapshot keeps. This is
the second time the snapshot has bitten this repository — the 2026-09-05 entry
records it dropping an untracked *candidate* — so the general rule is: **check
what the snapshot actually contains before trusting a queued result.**

**3. The snapshot has no `python.exe`, so a RELATIVE interpreter path fails.**
Added 2026-09-09 on `workerq 1.3.0`. Job `406` died at 0 s with no traceback:

    worker-q: failed to start command: [WinError 2] The system cannot find the file specified
    worker-q: '.venv-gpu/Scripts/python.exe' was not found in the snapshot

`.venv-gpu` is gitignored. The snapshot carries the directory skeleton and even
`Scripts/activate`, but not the interpreter binary. Every GPU module in
`local_sim/` documents itself as `.venv-gpu\Scripts\python.exe local_sim/...`,
which is right by hand and **wrong on submit**. Use an absolute interpreter path;
keep the SCRIPT path relative so it still resolves inside the frozen copy, which
is the whole point of the snapshot:

    workerq submit --project arc-whest --ram 10 --vram 25 --cpus 4 -- \
      "C:/Users/samsu/Documents/arc-whest/.venv-gpu/Scripts/python.exe" \
      -u local_sim/<module>.py --out "<ABSOLUTE path>"

(1.3.0 now prints `Passthrough: .venv, .venv-gpu, ...` and may fix this on its
own. The absolute path costs nothing and does not depend on that holding.)

**4. A relative `--out` writes into the SNAPSHOT, not the repository.** The job
prints `artifact experiments/.../x.json` and exits 0, and the file is not there.
It is at `~/.local/state/gpuq/snapshots/<id>/job-<id>/experiments/.../x.json`,
and `cleanup` will eventually delete it. Job `409`'s D1 probe result had to be
copied back by hand. **Pass an absolute `--out`.** This is the same class as trap
2 and the 2026-09-05 candidate drop: *the snapshot is a different filesystem from
the one you are looking at.*

### `workerq 1.3.0` commands worth knowing before declaring

`doctor` (health; exit 0/1/2) · `reconcile` (**run after a reboot** — repairs job
metadata) · `gpu` (who holds memory) · `reserve` · `gpu-threshold` · `promote`
(queue-front, never preempts) · `config` · and `--env KEY=VALUE` on submit, which
is the supported way to pass e.g. `PYTORCH_CUDA_ALLOC_CONF` rather than baking it
into a module.


## 2026-09-07 — AN OP COUNT IS MEANINGLESS WITHOUT ITS WEIGHT CONVENTION. THERE ARE THREE.

**Never compare two operation counts, or two FLOP totals, without checking which
weight convention each was taken in.** Doing exactly that on 2026-09-07 made the
per-operation residual law look `5×` worse than it is, and an independent Skeptic
caught it. The three conventions, on the same bytes:

| convention | how you get it | k3_full | canary4 |
|---|---|---:|---:|
| f32 census | `p2_op_census.py`, float32 weights | `3,232` ops / `1,714,554,351,206` | `2,673` / `168,636,374,630` |
| `whest run` mini | `--runner subprocess` on the public split | `3,233` / `1,714,554,383,974` | `2,674` / `168,636,407,398` |
| **grader twin** | `lifecycle_eval.build_weights(..., 'f64_readonly')` | **`3,249` / `1,714,587,938,406`** | **`2,690` / `168,669,961,830`** |

The ladder between them is exact and fully explained:

    f32 census
      + 32,768      + 1 op    materialise_predictions, 2 FLOP/elt on the (16,1024) output
      + 33,554,432  + 16 ops  the opening f64->f32 weight cast, 2 FLOP/elt x 16 x 1024^2
    = grader twin   == THE GRADED TOTAL, bit-identical on all 100 rows, both submissions

**`whest run` reads low because the local dataset hands the candidate float32
weights and the grader hands float64** (`lifecycle_eval.py`), so the local run
never pays the opening cast. `certify_candidate.py` already asserts this — "the
f64-vs-f32 FLOP delta is an integer multiple of one pass over the weights
(`2*depth*width^2 = 33,554,432`)", and it reads `1.0000` passes.

> **RETRACTED, same day.** An earlier version of this section called the
> `2**25` gap an OPEN QUESTION and offered "precisely one `(n,n) @ (n,)`
> matrix-vector product per layer" as a candidate mechanism, on the arithmetic
> `2 n^2 L = 2**25`. **That was a numerical coincidence and it was wrong.** The
> root cause was already determined, already checked by an existing gate, and
> named in the very artifact the entry cited. `2 n^2 L` and
> `2 FLOP/elt × L × n^2` are the same number for a reason that has nothing to do
> with matrix-vector products.

## 2026-09-07 — THE CAST MODEL OF THE SERVER FLOP OFFSET PREDICTED `330212` EXACTLY, OUT OF SAMPLE

**Do not add a flat `+2**24` to a local FLOP count and call it the graded
number** — `AGENTS.md`'s one-line "unexplained `+2**24` FLOPs (`+0.0122%`)" is the
stale coarse reading. `local_sim/official_config.py` carries the refined one, and
on 2026-09-07 it was confirmed **out of sample**:

    graded flops_used (330212)  1,714,587,938,406
    local grader-twin arm       1,714,587,938,406      delta 0  -- EXACT

**The offset depends on how the candidate touches the grader's float64 weights,
and the model predicted this row before it was submitted** — the prediction is in
commit `fe2c6ed` (2026-09-05), two days ahead of the submission, and names
`estimator_k3_full.py` by filename.

| opening cast | on the grader | offset | submissions |
|---|---|---:|---|
| `x.astype(fnp.float64)` | a SAME-DTYPE f64 cast | `+2**24` (one FLOP per weight element, `16 · 1024²`) | `328662`, `328664` |
| `x.astype(fnp.float32)` | a GENUINE f64→f32 cast | **`0`, exact** | `329896`, **`330212`** |

The obvious confound — the `0.12.0 → 0.12.1` stack move — is killed by `329889`,
which re-graded v2.7's unchanged bytes on the new stack and still billed
`+2**24 − 32`.

**What this does NOT license.** It remains a MODEL, not a mechanism: no graded
operation ledger exists, so nothing is attributed to an operation and nothing here
claims server op-by-op parity. `n = 2` per branch, and the branches are confounded
with file vintage, which is not resolvable from what we hold.
`server_flop_offset_per_mlp` still carries the MAXIMUM observed (`2**24`) so
`graded_flops()` over-estimates — the safe direction for a floor-margin check and
the wrong direction for claiming parity.

## 2026-09-05 — LOCAL `residual_wall_time_s` IS DATA-BOUND, AND THE GRADER'S IS NOT

> **CONFIRMED 2026-09-07 by the cross-candidate control this section could only
> infer**, and the figures below are the CORRECTED ones (see the convention
> section above; the first write-up compared two conventions as one).
> `estimator_k3_full.py` was graded as `330212`. Its local residual on the
> SUBMITTED bytes was `0.4363 / 0.4486 / 0.4667 s` — 100/100 local failures. Its
> GRADED residual was `0.0355 / 0.0364 / 0.0393 s`, `0/100` failures: **the local
> instrument over-read by `12.3×`.** And the per-op constant held across a
> `10.2×` change in FLOPs:
>
>     candidate        ops    FLOPs    kernel   GFLOP/s   residual med   PER-OP
>     canary4 329896  2,690   168.7 G   2.54 s     66.4      30.11 ms   11.192 us
>     k3_full 330212  3,249  1714.6 G   7.97 s    215.1      36.40 ms   11.205 us
>
> **`0.10%` apart.** The array traffic that dominates the local reading is billed
> on the server to `flopscope_server_kernel_wall_s` — which moved only `3.1×` on
> that `10.2×` FLOP change, so it does not track FLOPs either, and **operations
> predict residual better than any array-traffic proxy**. Size the gate on the
> MAXIMUM observed: `13.4 µs/op`, from the widest inferred row of the visible
> board population, giving a working gate of `22,000` ops/MLP.
> Binding form: `docs/VALIDATION.md` §2b-iii. Corroboration across a `5.4×` op
> lever: `local_sim/p2_residual_law.py`.

**Do not gate a data-heavy candidate on the local residual reading, in-process or
under `--runner subprocess`. It measures a different quantity than the grader's.**

> **The width-sweep table below is from an EARLIER BUILD** (`1,940,198,469,222`
> FLOPs at `3,304` ops — the pre-collapse `0.88230 B` chain), not the submitted
> `0.77970 B` bytes. Its qualitative point — residual scales with array size at a
> FIXED operation count — is unaffected, and it is kept in the words it was
> recorded in. Do not quote its absolute seconds against the submitted file.

The experiment, because this one is a measurement rather than a caution.
`estimator_k3_full.py` run at three widths, depth 16, float32. The operation
count is IDENTICAL at all three — same code, same Python, same number of metered
calls — and only the array sizes change:

    width    ops     counted FLOPs        local residual   us/op   widest array
      256  3,304      30,465,280,998            66.6 ms    20.2         3.8 MB
      512  3,304     242,922,395,494           158.2 ms    47.9        15.0 MB
     1024  3,304   1,940,198,469,222           480.4 ms   145.4        60.0 MB

Residual grows `7.2x` across a `16x` growth in array size at a FIXED operation
count. It is therefore absorbing in-process array allocation and memory traffic,
not Python-between-calls.

**On the grading server that traffic is somewhere else and is reported
separately.** flopscope runs client/server there — the participant process holds
`RemoteArray` handles, and the arithmetic and the buffers live on the server's
seven cores. Submission `329896` (canary4), 100 graded rows:

    participant_predict_wall_s        3.3811 / 3.5461 / 3.8560
    flopscope_server_kernel_wall_s    2.3941 / 2.5392 / 2.8305
    flopscope_server_overhead_wall_s  0.9318 / 0.9731 / 1.0263
    flopscope_server_comms_wall_s     0.1046 / 0.1073 / 0.1098
    residual_wall_time_s              0.0292 / 0.0301 / 0.0307   <- 0.8% of wall

The same file reads `~234 ms` of residual on this host. The `0.4 s` cap is
applied to the last row only, and neither the kernel time nor the comms time
counts against it.

**What to gate on instead: the OPERATION COUNT**, through
`official_config.predicted_graded_residual_s()` (**`13.4 us/op` as of
2026-09-07** — the MAXIMUM observed across two of our own graded rows and twenty
inferred board rows; the figure this section originally carried, `11.6 us/op`,
is close to the MEDIAN, `11.2`, and is not the number a cliff is sized on).
Corroborated by marius_binner's Phase-1 self-measurement of `11.4 us/call`. The
op count is
exact, host-independent, and is the quantity the graded residual actually tracks.

**What this does NOT license.** `AGENTS.md` is still right that a local absolute
wall time is not quotable and that canary4's local:graded ratio is not a general
safety factor for a different candidate. The claim here is narrower and is
supported by the table above: the local number contains a term the graded number
does not, and that term grows with array size. `certify_candidate` section 10
still reports the local residual and still refuses on it — deliberately, because
it is the deployment-authority runner — so a candidate in this class must carry
BOTH numbers to the human who decides, not the convenient one.


*Extracted verbatim from `CLAUDE.md` on 2026-08-07 when the top-level agent file
was slimmed to a map. Nothing here was rewritten: these are the findings that must
not be rediscovered, in the words they were recorded in. Newest sections first;
sections marked "historical" were superseded by the ones above them and are kept
because the correction is itself the lesson.*

**Read this before**: touching `estimator.py`, quoting a number, comparing a cost,
reopening a closed lane, or trusting an offline measurement path.

Companion docs: `docs/VALIDATION.md` (how anything gets measured — the protocol),
`docs/CURRENT_STATE.md` (what is true right now), `experiments/LOG.md` (the full
narrative record), `docs/GAME_PLAN.md` (the sole active Phase-2 strategy).

> **Phase-2 reading rule, 2026-08-24:** later sections that name a numbered
> `GAME_PLAN` revision or call a Phase-1 sprint plan "authoritative" are preserved
> historical lessons, not current directions. `docs/GAME_PLAN.md` replaced every
> numbered revision. Do not infer a live lane state from an old GOTCHAS section;
> use `CURRENT_STATE.md`.

---


## 2026-09-04 (WP20 deployment price) — a regenerating flow's cheapness has a ONE-HOP HALF-LIFE, and price against the HEADROOM

> **AUDITED AND CORRECTED THE SAME DAY.** An independent Skeptic returned
> `PROVISIONAL` on the slice this section came from and flipped one of its rows.
> Items 1 and 2 stand as written. **Items 3 and 4 are rewritten below**, and the
> numbers they originally carried are recorded as errata in `experiments/LOG.md`
> rather than deleted. Full verdict: LOG, 2026-09-04, *"SKEPTIC ON THE WP20
> DEPLOYMENT PRICE"*.

**1. The identity legs are the whole saving, and one `contract_W` destroys them.**
`FactoredTensor.from_dstensor` returns `(diag(S3) + 3 S21^T, I, I)` — recorded in
the WP14 section above as *"worth `2.08x`"*. What that section does not say, and
what a carried design needs to know before it is designed:

> The saving belongs to the **regenerated** tensor and **does not survive being
> carried**. A generated all-distinct block is `(w1 . WK_11, 3 I, (w2 (x) w1 .
> WK_11)^T)`; after ONE `contract_W` its legs are `(W A1, 3 W, W A3)` — three
> real `(n, R)` matrices with no identity among them. From that hop onward it
> pays the generic price forever: `3` matmuls in `contract_W` instead of `1`, and
> `3` block permutations in the `(2,1)` slice instead of `2`.

**"Carry it instead of regenerating it" is therefore not a knob with a cost. It
is a different price regime**, entered on the first hop, at `2x` on `contract_W`
and `1.5x` on every slice read, **per carried generation**. Two qualifications
the first draft of this section got wrong, both found by audit:

- **the window warms up.** At layer `l` the live generations are those made at
  `l-1, l-2, l-3`, so the propagation counts are `1, 2, 3, 3, …` — **42** rounds
  over a `16`-layer flow, not `45`. `p2_wp15_depth.slices_window` says so in its
  own docstring and I billed as though it did not.
- **a generation's FIRST hop is cheaper than the rest.** Upstream's generated
  block has `fac2 = torch.eye(n) * 3` (`factor_k3.py:400`), so at age 1 the
  middle leg is a scaled identity and `W @ 3I` is a column scaling. Exactly one
  carried block is at age 1 in every round.

Measured, f32, live shape, **corrected**: carrying three generations at rank
`192` with propagation and **nothing else** — no slice read, no fresh block, no
truncation, no injection — is `44,675,629,056` FLOPs, `0.87x` the canary's
`51,249,140,941` headroom, i.e. **it fits**. The honest carry of the object that
was actually measured is `608,251,203,072`, **`9.93x` the headroom**. That is the
number the regime change is worth quoting on. `local_sim/p2_wp20_price.py`,
`experiments/2026-09-04_wp20_price/wp20_price.json`.

**2. Price against the HEADROOM, not against the budget.** This file's own
pricing table records *"thin SVD `26 n^3` ~ `0.16%` B"*. `0.16%` of `B` reads
harmless. The same call is **`54.5%` of a floor-resident candidate's spare
budget**, because a candidate at the `0.1` multiplier floor has `2.33%` of `B` to
spend and not `100%`. **A rate quoted against `B` is smallest exactly where it is
fatal.** Every price in a promotion decision belongs in units of
`0.1 * B - (candidate + 2**24)`.

**3. AN OPERATION COUNT IS A PROPERTY OF A PROGRAM, NOT OF A COMPUTATION — never
transfer one between spellings.** Measured by the audit, at the live shape, in
one flopscope: three separate rank-`192` blocks bill **`35`** operations; **one
concatenated rank-`576` block bills `11`** — *identical* counted FLOPs, values
agreeing to `5.1e-07` relative. **A `3x` range inside the same arithmetic.**

The concrete error this caused, and it is easy to repeat: a lab transcription of
the regenerating flow bills `469` operations where the canary's own third-order
increment is **`2,255`** (`flow_price_wp17b.json`, `2,708 - 453`) — the same
arithmetic in `4.81x` fewer array calls, because a lab spelling fuses what a
shipped file spells out. **Adding a lab op increment to a shipped-file op total
is the mixed-instrument error, and it understated a carried design's operation
cost by about `2x`.** Counted FLOPs on a dense path *do* transfer between
spellings; operation counts do not.

Meter both axes anyway. Which one binds is a property of the design — but say
which instrument each was metered on, in the same sentence as the number.

**4. A deliberately unachievable lower bound is a good technique and I executed
it wrongly, which is the more useful half of the lesson.** The idea is sound:
price an implementation that could not work, and if it is *already* over budget,
the negative verdict is independent of every implementation choice you made.
**But a lower bound has to actually be one.** Mine over-charged twice — the
warm-up and the age-1 identity leg, item 1 above — and came in at `1.060x` the
headroom on a `6.0%` margin. Corrected it is `0.872x`: **it fits, and it bounds
nothing.**

Two rules out of it. **Never let a decision rest on a row whose margin is
single-digit percent without saying so in the same paragraph** — `9.93x` needs no
audit, `1.06x` does. And **watch for the implementation choices that happen to
favour your own conclusion**: propagating three generations at every layer looked
like the safe, conservative direction, and it was the direction that made the
negative stronger.

---


## 2026-09-03 (WP18 / SKEPTIC WP18) — five traps, and three of them were in things I had already written down

**1. A `gpuq` job reports the CRLF hash. The working tree and the archive are LF.**
`certify_guarded.json` records `candidate_sha256 = 3d29e2d5a3e0f986…` for a file
that hashes to `3536cda3183ab429…` on disk and inside `submission.tar.gz`.
Confirmed by construction: `sha256(bytes.replace(b"\n", b"\r\n"))` is exactly the
certify hash. `gpuq` runs from a git checkout under
`~/.local/state/gpuq/snapshots/<id>/`, where `core.autocrlf` renders CRLF. **Same
content, two renderings.** So a certification hash never equals the archive hash,
and the chain from "certified" to "packaged" has to be joined by hand. Carry both
whenever a candidate is presented, or the provenance is not followable.

**2. `certify_candidate` § 9 MATCHES the memory-gate report on the report's own
`candidate` field, but FINDS it by filename.** It looks for
`linux_memory_gate_<candidate stem>.json` beside the candidate or in
`experiments/2026-08-31_wp0cd_certify/`. A report written anywhere else is
**SKIPPED, not failed**, and § 9 silently stays advisory while the run still says
`CANDIDATE REFUSED` for other reasons — so the missing enforcement is easy to
miss in the noise. It also matches on the FILENAME, not the content hash, so a
report generated for an earlier revision of the same filename is still read: re-run
the gate whenever the candidate changes.

**3. Quote a per-MLP price against the STEADY STATE, never against MLP 1.** Any
constant cached on the `Estimator` instance is paid by the first MLP of a run and
by none of the others, so the file has two prices and the artifact will contain
both. `SKEPTIC WP18` caught this packet quoting `+19` operations against the
baseline's first row when the honest figure against the row 99 of 100 graded MLPs
see is `+28`. The grader constructs ONE `Estimator`, calls `setup` **once, outside
any `BudgetContext`** (`whestbench/runner.py:175`, `subprocess_worker.py:207`),
and opens a **fresh `BudgetContext` per predict** (`scoring.py:709`). A probe that
builds a fresh instance per seed, or calls `setup` inside the meter, cannot see
the effect at all — and if `setup` resets the cache it will report the exact
opposite of the truth.

**4. `alpha = mu / sigma` is UNBOUNDED, and no Wick coefficient saturates in it.**
`w(0,1)` is `sigma * (phi(alpha) + alpha * Phi(alpha))` — that is `E[ReLU]`, and it
grows **linearly** in `alpha` forever, because a neuron that is always on has mean
`mu`. `w(0,2)` grows as `alpha^2`, `w(0,4)` as `alpha^4`. I wrote the opposite into
a submission file's comment ("every Wick coefficient has already reached its
limit beyond `13.2`") and used it to justify a cap 25,000 times more aggressive
than the real constraint. **The real constraint is float32 overflow of the highest
power the frozen table contains, `alpha^4`, i.e. `|alpha| < 4.295e9`**; past it
`alpha^4` is `inf` while `phi(alpha)` is exactly `0`, and `inf * 0` is the NaN that
fails the MLP. A clamp on `alpha` is a **lossy saturation**, not a limit-preserving
rewrite, and it must be described that way.

**5. An estimator's accuracy on the graded distribution says nothing about its
robustness, and the invariant suite must be run on THE CANDIDATE.** The K3 canary
scored `8.75e-07` with `0/100` failures on 100 graded rows *and* scored **5 of 30**
on `local_sim/edge_cases.py`, returning NaN on three of four degenerate cases and
`-4.037e+08` at `32 x 64`. It had never been run on that file — a certification of
`estimator.py` had been read as a property of the candidate. Run
`edge_cases.py <candidate>` before believing anything about a candidate's
deployment behaviour, and note that its guards convert *"fails the MLP"* into
*"answers absurdly"* on degenerate inputs, which is the right trade and is **not
the same as robustness**.

**And one methodological trap worth the same weight.** `local_sim/edge_cases.py`
invariants **8** (layer-1 closed form, `rel < 1e-12`) and **9** (permutation
equivariance) were calibrated beside an incumbent delivering `1.7e-16` and
`2.2e-15` — they encode a **float64** expectation. Phase 2 bills float64 at
`2.0x` on the operation this repository actually measured (matmul
`268,173,312 -> 536,346,624`), and since the covariance backbone is essentially
the whole cost, a full-f64 flow pushes `x` from `0.0767` to `0.153`, the
multiplier leaves the `0.1` floor, and the score is `1.53x` worse.

> **SUPERSEDED IN ITS CONCLUSION, 2026-09-03.** This paragraph used to end
> *"Invariant 9 is therefore unmeetable by any competitive Phase-2 candidate."*
> `CURRENT_STATE.md` retracted that universal once already — the evidence
> reaches the mechanism and the cost **for one candidate**, not a claim about
> every candidate — and the amendment below has now falsified it outright:
> invariant 9's absolute clause was replaced by "every row's drift, in MSE
> units, `<= 1%` of the candidate's own raw MSE", and the f32 canary reads
> `30/30`. See `docs/VALIDATION.md` § 3.2, which also records what the
> amendment **surrendered** (796x-1,552x of sensitivity, and a rotation-stability
> measurement that section used to advertise).
>
> **A second correction, same date.** *"exactly `2.0x` (measured)"* overstates
> what exists. What was measured is the **rate on one matmul**; that the whole
> backbone would scale at that rate is an **inference**, and no artifact shows a
> partial-f64 backbone being built and priced. The inference is defensible here
> because the backbone dominates the bill — but this repository has already
> recorded getting exactly this kind of attribution wrong once (see the sentence
> that follows), so it is labelled rather than repeated.

**Before reading a permutation-equivariance failure as an index bug, run the f64
control.** The deviation is a **dtype** effect: same flow, same networks, f32 vs
f64 moves `d_k/scale` from `2.116e-06` to `2.365e-15` (a ~1e9 ratio, which is
the eps ratio), and it **grows with width** (`1.105e-06` at `256x32` to
`2.992e-06` at `1024x16`) — which is what BLAS accumulation noise does and what
an index bug does not.

Invariant 8 is NOT in this category: row 0 has a closed form and `estimator.py`
has met the bar since v2.4 with three lines (`_exact_first_layer`), at about four
operations. Do not attach the `1.53x` full-flow penalty to a narrow fix — that
argument was made here and it was wrong.

---

## 2026-09-02 (WP13) — a `gpuq` job submitted with ABSOLUTE paths does not run the snapshot

`CLAUDE.md` and `AGENTS.md` both record, correctly, that *"a queued job runs the
source exactly as it was at submission time, so you may keep editing the
repository immediately after submitting."* **That guarantee is conditional on the
command being snapshot-relative, and nothing enforces it.**

Job 112 was submitted as

    workerq submit ... -- C:/.../arc-whest/.venv-gpu/Scripts/python.exe \
                          C:/.../arc-whest/local_sim/p2_slice_lab.py --all

Its log shows `cwd=C:\Users\samsu\.local\state\gpuq\snapshots\112\job-112` and
`snapshot=git b54a4d2` — the snapshot was made, and the job ignored it, because
the interpreter and the script were both named by **absolute paths into the live
working tree**. The module then derives `_ROOT` from `__file__`, so its outputs
and its `seed_ledger` reads also went to the live tree.

For job 112 this was harmless and was **checked, not assumed**: the snapshot copy
and the live file are byte-identical after CRLF normalisation
(`sha256 0cbd45da...` both sides). That was luck. Had the file been edited in the
19 minutes it sat in the queue, the run would have silently executed the edited
version while its log advertised a snapshot commit — the worst shape a provenance
defect can take, because the record would look right.

**The rule.** Either

- submit a snapshot-relative command (`python local_sim/foo.py`, resolved against
  the snapshot `cwd`), and the standing guarantee applies; or
- submit absolute paths and treat the repository as **frozen for that file until
  the job finishes**, and verify the hash afterwards.

There is a real reason to prefer absolute paths — artifacts land in the live
`experiments/` tree instead of being discarded with the snapshot, which is
exactly the WP9 defect where `P02` was drawn while the live ledger reported the
seeds FREE. So this is not "always use relative". It is: **know which one you
chose, and hash-check the file afterwards if you chose absolute.**

---

---

## 2026-09-02 (SKEPTIC WP14 follow-up) — three traps, two of them in a REFERENCE

### `nan > tol` is `False`, so a NaN reference PASSES a threshold check

The correctness check built for the `fnp` K3 layer compared it against
Gauss-Hermite quadrature. `numpy.polynomial.hermite.hermgauss` **overflows past
about degree 200 and returns NaN weights**; the comparison printed `nan` for all
three cumulants and the verdict printed **PASS**.

> **A `>` comparison is not a check. It silently admits the one value it cannot
> order.** Every threshold gate in this repository must reject non-finite input
> explicitly, before comparing.

Second appearance of this class: WP8's residual soak sorted never-executed rows
(`residual_wall_time_s = 0.0`) to the bottom of a pooled median, so the reported
"median at rank 150/300" was the 41st percentile of executed work. Both survived
because the *number a reader checks* looked normal.

### Gauss-Hermite is the wrong quadrature for a ReLU expectation

`max(z, 0)` has a kink at the origin, so Gauss-Hermite — which integrates over
the whole line against `exp(-t^2)` — converges **algebraically**, not spectrally.
At degree 160 it disagreed with the exact `sigma phi(alpha) + mu Phi(alpha)` by
`1.1e-03`, which reads exactly like a bug in the thing under test.

**Split the integral at the kink and use Gauss-Legendre** on
`[max(mu, 0), max(mu, 0) + 14 sigma]`: the integrand is analytic there, the
discarded tail is below `1e-42` of the mass, and agreement goes to `3e-15`.

> When a check fails, ask whether the REFERENCE is the thing that is wrong. Here
> the reference was wrong by a thousand times the tolerance.

### Under `lambda = 0` the FLOP budget is not the binding constraint — the OP COUNT is

**Read this before quoting any Phase-2 cost number, including the one in the next
section.** A FLOP below the `0.1` floor costs **nothing**: the multiplier is
already at its minimum. What can still ruin a run is the **`0.4 s` per-MLP
residual cliff**, and residual tracks **operation count and Python glue, not
FLOPs** — and an excursion does not cost score, it **fails the MLP and zeroes the
prediction**, turning a `1e-7` row into `~0.92`.

Measured with a matched control in the same process, same host, same run
(`local_sim/p2_k3_fnp.py --flow-price`):

| | counted FLOPs | flopscope **operations** | local residual |
|---|---:|---:|---:|
| shipped K2 closure | `68.8 B` | **`453`** | `0.018`-`0.034 s` |
| regenerating K3 flow | `173.4 B` | **`~5,300`** | **`0.29`-`0.44 s`** |

`2.5x` the FLOPs, **`11.6x` the operations, `10`-`16x` the residual** — and over
the `0.4 s` cliff on 2 of 6 readings.

> **Quote counted FLOPs, operation count and residual headroom in the same table,
> or you are measuring the axis that is free.** The absolute second is NOT
> quotable (authority is `--runner subprocess`); the RATIO to a matched control
> is what transfers.

This is the mechanism that killed the packed lane, seen again.

### A correctness check is scored by what it KILLS, not by what it passes

The three closed-form checks on the `fnp` K3 layer agree to `1e-14` and kill
**0 of 7** targeted value-destroying mutations — including one that is **a whole
matmul cheaper**, and including the exact `w(2)`-for-`w(1)` defect the module's
own docstring warns about. At `kappa_3 = kappa_4 = 0` the expensive operations
run on zeros.

> **The limit that makes an answer analytic is usually the limit that switches
> off the part you are trying to price.** Run a mutation battery beside any check
> that licenses a number; `p2_k3_fnp.py --mutation-coverage` is the pattern.

### The regenerating K3 flow's measured per-layer price, and it supersedes the projection

**The VALUES below ARE checked against upstream at non-zero `kappa_3`** as of
2026-09-03 (`p2_k3_parity.py`; `NOT_DONE` step 6 CLOSED). The FLOP figures in
this section predate that check and are superseded downward — one layer at f32
is `10,813,796,920`, the flow `168,629,787,238`. This is a price, and the checks that license it
kill nothing (previous section). Quote it as a cost and not as a method.

Measured, one program, one layer, at `1024` inside a real `BudgetContext`
(`local_sim/p2_k3_fnp.py --layer-price`):

| | counted FLOPs | dense `1024`-matmuls | `x 16` layers | `x budget` |
|---|---:|---:|---:|---:|
| `f64` | `21,675,713,540` | `5.047` | `346,811,416,640` | `0.15773` |
| **`f32`** | ~~`10,837,935,618`~~ **`10,813,796,920`** | ~~`5.047`~~ **`5.036`** | ~~`173,406,969,888`~~ **`168,629,787,238`** | ~~`0.07886`~~ **`0.076684`** |

> **SUPERSEDED 2026-09-03 (WP16/WP17).** The struck figures were measured on a
> layer that computed the `(1,1,1)` block, which WP16 showed is DEAD in a
> regenerating flow, and before WP17's operation-count reduction. The bold
> figures are current, and each has a committed artifact:
> `layer_price_wp17.json` for the per-layer figure and
> `candidate_parity_1024x16.json` for the whole flow, both under
> `experiments/2026-09-03_wp16_parity/`. Operations per MLP: **`2,646`**,
> lab and shipped candidate identical. The multiplier is `0.10000` either way.


The `f32 / f64` ratio is `0.5000036376`. **The excess over half is the tell, not
the halving**: it is exactly `78,848` FLOPs, precisely the two `flops.stats.*`
calls, which are computed in `float64` in BOTH arms and so are billed at rate
`2.0` in both and do not halve. A ratio slightly *above* `0.5` is the signature
**of** the promotion; what says nothing promoted downstream is the per-array
dtype audit. The five `n^3`-class operations per layer are `W.T @ cov @ W` (two),
`W.T @ M` (one) and the `(2,1)` slice's two surviving block permutations. The
remaining `0.047` matmuls/layer is every `O(n^2)` thing in the layer put
together.

> **Carry `5.047` f32 matmuls per layer, not `4.89`.** The composed projection
> `0.07636 B` was 3.3% optimistic; the margin under the `0.1` floor is
> `0.02114 B` = **`1.35` f32 matmuls per layer**, not `1.51`. One un-reducible
> extra `n^3` op per layer still fits. Two still do not.

And two structural facts worth not rediscovering:

- **`K[4]` in the SIMPLE `k_max = 3` flow is one SCALAR per layer.**
  `r_x(4, 3, SIMPLE) = 2` and `4 - 2*2 = 0`, so its core is order zero. The
  `(3,1)` and `(2,1,1)` slices contribute exactly nothing to the `r = 2`
  projection (`_lap_m_dslice` admits no multigraph meeting their arity
  condition), so the `n^3` object `(2,1,1)` that upstream materialises need never
  be formed.
- **The freshly generated all-distinct block's contribution to the next `(2,1)`
  slice is `w(1)_i^2 w(2)_j C_ij^2`** — its middle leg is a multiple of the
  identity, so every block permutation degenerates to a row scaling. No `n^3`
  object appears anywhere in generating it.

### Intent recorded in prose is not recorded

`seed_ledger` claims carry both a free-text `note` and a machine-readable
`paired_with`. WP14 C0-i wrote "DELIBERATE DESIGN-tier overlap" in the note and
left `paired_with` empty, and **preflight steps 7 and 9 were RED for four
commits** over a ledger that was substantively correct.

> If a rule is machine-checked, its exception must be machine-declared. Use
> `seed_ledger.declare_pair(...)` — a `"pair"` record — not a sentence.

**And say so when a repair loosens an invariant.** The pair record moves the rule
from *"a deliberate overlap must be pre-declared"* to *"…must be acknowledged,
possibly afterwards"*. `declare_pair` refuses to license an overlap that does not
exist, but it cannot distinguish intent from retroactive rationalisation.

*(Corrected 2026-09-02 by `SKEPTIC WP14-FOLLOWUP`: the window began at
`aa01fdd`, not `3b38796`, and ran **nine** commits, not four. Step 7's red was a
**different** defect — an artifact claimed with no hash — which lasted two
commits and self-healed. "Nothing was wrong with the ledger" was false for those
two. **Do not assert a git-reconstructible fact without reconstructing it**, least
of all inside a finding whose point is that prose is not evidence.)*

And **nothing enforces "preflight was green" at commit time.** The pre-commit
hook guards protected paths, not the gate. That hole is open.

---

## 2026-09-02 (WP14 / C0-iii) — a flopscope price is not well-formed without its DTYPE

**CORRECTS the WP13 entry that stood here for one day.** flopscope 0.12 does
not bill a dense `(n, n) @ (n, n)` matmul at `4 n^3`. It bills the textbook
`2 n^3` and then scales **every** operation by a per-dtype rate from its own
packaged weights, `flopscope/_weights.py::_ACTIVE_DTYPE_RATES`:

    float64 = 2.0   int64 = 2.0   uint64 = 2.0   complex128 = 2.0
    float32 = 1.0   float16 = 1.0   int32 = 1.0   complex64 = 1.0

Measured at `n = 1024`: `4,292,870,144` FLOPs at f64, `2,146,435,072` at f32.
WP13 read the f64 column because `np.random.default_rng().standard_normal()`
returns float64, and recorded it as a property of matmul. Every cost figure in
that slice is therefore a **float64** figure, quoted against accuracy evidence
gathered in **float32**.

**Three consequences.**

1. **Always name the dtype in the same sentence as the price.** "The floor
   affords 3.2 matmuls per layer" is true at f64 and wrong at f32, where it is
   6.4. `p2_slice_cost.py` now takes `--dtype` and stamps the billed dtype and
   the whole rate table into every artifact.
2. **`np.float64(3.0)` is a trap.** Under NEP-50 a numpy scalar is *concrete*,
   not weak, so one of them anywhere in an f32 expression promotes the result
   and flopscope bills the whole downstream chain at rate 2.0. A **Python**
   `3.0` stays weak and bills f32. `measure()` now asserts the carried state's
   dtype and refuses rather than reporting a silently-promoted number.
3. **This does NOT reopen the `f16`/`int8` note further down this file.** Those
   are rate `1.0`, the *same* as f32: there is no dtype trick below f32. The
   only dtype saving available is f64 -> f32, it is worth exactly `2x`, and for
   the K2 closure it was measured accuracy-neutral on 96/96 networks on
   2026-08-26.

*(Retained for the record: the WP13 heading read "flopscope bills a dense
matmul at `4 n^3`, not `2 n^3`". It was right about the number it saw and
wrong about what the number was.)*

Measured, not modelled, at the live width in `flopscope 0.12+np2.2.6` inside a
real `BudgetContext` (`local_sim/p2_slice_cost.py --> experiments/2026-09-02_wp13_slice/e0_cost.json`):

| primitive at `n = 1024` | counted FLOPs | `/ n^3` |
|---|---:|---:|
| `A @ B`, `A.T @ B` for `(n, n)` | `4,292,870,144` | **`3.998`** |
| `(A * B).sum(axis=1)` | `4,192,256` | `0.0039` |
| `A * B`, `A + B`, `A * v[:, None]` | `2,097,152` | `0.0020` |
| `A.T`, `fnp.diag(A)` | `0` | `0` |

**Every hand cost model that assumed the textbook `2 n^3` was a factor of two
optimistic.** Before quoting the price of a proposed architecture, count its
matmuls, not its multiply-adds.

The corollary is the more useful form, and it is the one to carry:

> The shipped `estimator.py` bills `137,600,326,656 / 16 = 8.60e9` per layer
> `= 2.00` matmuls — that is `W^T Sigma W` and essentially nothing else. At the
> `0.1` multiplier floor a `1024 x 16` estimator's ENTIRE budget is
> **`3.2` dense matmuls per layer.** A K3 flow that regenerates `kappa_3` from
> its two read-slices needs `5`, hence `0.1527 B`.

Two levers on that number exist and are untried: flopscope's **symmetry tagging**
(`as_symmetric()` — `Sigma` is symmetric and `W^T Sigma W` is two of the five)
and the shipped file's existing **Strassen** path. Neither is priced. Do not
assume either works; both were noticed while reading the meter, not measured.

Unrelated to pricing but found in the same read, and load-bearing for anything
touching `kappa_3`: `FactoredTensor.from_dstensor` returns factors
`(diag(S3) + 3 S21^T, I, I)` — **two of the three legs are exactly the
identity**. A generic `O(n^2 r)` spelling pays six matmuls a layer for what
three do. That single structural fact is worth `2.08x` and it cost one file
read and no seeds, after eleven closed lanes had tried to compress the tensor
instead.

---

## Do not read a suite result taken under heavy concurrent load

**Observed 2026-08-31 (WP0C).** `local_sim/test_enforcement_labels.py` ran while
`local_sim/test_certify_candidate.py` was executing eleven full candidate
certifications in the background. It reported `edge_cases.py` at **29/30** and
then flagged two documents as carrying STALE labels, because a documented
`30/30` no longer matched what the suite had just returned.

Run serially, `edge_cases.py` returns **30/30**, and the label suite returns
**39/39**. Three concurrent copies of `edge_cases.py` alone also return 30/30
each, so the trigger is heavier than simple parallelism.

**The cause was isolated later the same day, and it is not a check at all.**
Six consecutive runs gave `rc=1, rc=1, rc=0, rc=0, rc=1, rc=1` with

```
numpy._core._exceptions._ArrayMemoryError:
    Unable to allocate 4.00 MiB for an array with shape (8192, 128) ...
```

A **4 MiB** allocation failing is the tell: it is not the suite's working set.
Windows was at **72,241 MB committed against an 89,333 MB commit limit** with
35 GB of physical RAM still free — NumPy fails on *commit*, not on free
physical. It later reached **95,734 MB of a 95,801 MB limit: 67 MB of commit
headroom.**

**And the pressure was NOT this repository's.** A first reading blamed this
session's own load (a WSL2 VM, a 6.26 GiB Hugging Face download, several
certifications) and that was wrong. `Get-CimInstance Win32_Process` named the
holders: **other projects on this shared machine** — a `kalshi-bot` BTC
collector/replica report at 20,672 MB and a `biohub` `celltrack train` run at
13,459 MB whose own `--notes` read *"requeued after host OOM; resumable"*. This
repository's processes were not in the top six. Between the two readings a
single probe here allocated **12,800 MiB** without failing and `edge_cases.py`
returned 30/30 three times running, which is what a quiet window looks like.

Two consequences, and the second is the important one:

1. `~/.claude/CLAUDE.md` coordinates **GPU** work through `gpuq`. Host **commit**
   is an equally shared resource and *was* coordinated by nothing. A heavy suite
   here can push a neighbour's training job over the edge, and one had already
   gone over once. **Fixed 2026-09-01:** `gpuq` now gates on host RAM and CPU as
   well as VRAM, and this repository declares a footprint for every heavy
   workload — `docs/SETUP_WINDOWS.md` § 7.1, wiring in `.gpuq.toml`. The trap
   that remains is *under-declaring*: an undeclared job is charged a 4 GiB
   default and is admitted when it should have waited, which reproduces exactly
   the failure described here.
2. **Do not kill what you did not start.** The obvious "fix" is to free 34 GB by
   terminating two processes. They are someone's live trading collector and a
   resumable training run.

So the label suite was reporting a TRUE fact — the suite really did return
29/30 — about a transient it had no way to name. `_ArrayMemoryError` is the
string to grep for.

The trap is not the flake, it is the shape of the report: a label checker that
reads a suite's live output will call a *correct* document stale whenever the
suite under-reports for any reason. Two of the repository's four historical
lost checks were "a check reporting a state rather than testing a property";
this is the mirror image, a check reporting a *transient* as a defect.

### A SECOND contention mechanism, and memory does not explain it

Later the same day, with **23,649 MB of commit headroom** — no memory pressure
at all — a preflight run went **RED at step 7** while an independent Skeptic was
running the same suites concurrently:

```
lifecycle_eval.py --self-test   49/56      (serially: 56/56)
edge_cases.py                   crashed    (serially: 30/30)
```

Both are green when run alone, immediately before and after. So this is **not**
the `_ArrayMemoryError` mechanism above.

**The specific mechanism was NOT isolated**, and this note says so rather than
inventing one. The most likely candidate is that `lifecycle_eval --self-test`
PROVOKES its failure paths with real timers — a `1e-9 s` residual limit, a wall
limit, a 6-second setup refusal — and a contended CPU can flip a timing-provoked
check in either direction. That is the same family as the step-16
`SETUP_TIMEOUT` sensitivity `VALIDATION.md` §7 already documents. It is a
hypothesis; `--self-test` writes no shared file (`lifecycle_self_test.json` is
written only under `--emit-json`, and its mtime did not move), so a file
collision is ruled out and nothing else is confirmed.

**Rule: a gate verdict is only a verdict when the gate had the machine.** Run
`scripts/preflight.ps1` and the enforcement suites serially, on a quiet host,
exactly as `VALIDATION.md` §7 already requires for step 16's `SETUP_TIMEOUT`.
Check `Get-CimInstance Win32_OperatingSystem` for committed-vs-limit before
believing a memory-shaped failure — and note that **ample headroom does not
clear a run of contention**, because the second mechanism is CPU, not RAM.
Re-run serially before recording either.

## 2026-08-28 (P2-WEFF / P2-WINDOW) — four traps, three of which I walked into

Two IMPORTANT slices, four independent Skeptic passes, and both lanes closed on
measured negatives. These are the operational lessons, not the results
(`CURRENT_STATE.md` has those).

- **The installed `0.4 s` residual cliff and `120 s` PER-MLP wall cap are different
  constraints, and no document here had paired them.** flopscope defines
  `residual = wall − backend − overhead`, so work inside instrumented ops is
  *backend* time and is **not** charged to the residual — which is why the
  shipped K2p1 spends `137.6e9` counted FLOPs for a graded residual of `~7 ms`.
  **Residual risk therefore scales with op count and Python glue, not with
  FLOPs.** **CORRECTED 2026-08-31:** installed `wall_time_limit_s = 120.0` is enforced
  separately around each MLP's `predict()` call. It is not divided across the
  100-MLP suite. Installed `scoring.py` constructs a fresh `BudgetContext` inside
  `for i, mlp`; the subprocess worker enforces the same per-call limit. The old
  `~1.2 s/MLP` planning constraint was false. Public graded ledgers prove that
  `104.766 s/MLP` passes, consistent with the installed ceiling but not exposing
  its exact server value. A design can still be inside the residual cliff and
  exceed 120 seconds on one local official-runner MLP, so check both, but dense
  backend-heavy candidates have far more wall headroom than the old record implied.

- **flopscope grants an ALIASED-OPERAND discount, and it is easy to leave on the
  table.** `flops.as_symmetric(C, symmetry=(0,1))` buys **nothing** — it costs
  `7,340,031` FLOPs to declare and prices the product at `1.00171x`. But
  flopscope's auto-detected `identity_pattern` is real: `fnp.einsum('ij,kj->ik',
  A, A)` bills **`0.5005x`** of `A @ A.T`, and `W core W^T` spelled as
  `cholesky(core)` + `W @ L` + aliased Gram bills **`0.8336x`** (relerr
  `7.65e-07`, requires an SPD core — upstream itself prints "Snapping negative
  variance to zero", so that is not free). **Worth nothing at a floor-pinned
  operating point; worth `~0.005 B` to anything that is not.** I first recorded
  "no discount is recoverable" after testing *one* spelling. A review found the
  others. **Do not generalise a pricing claim from a single spelling.**

- **`max(0, RMSE_full − D)` is not the triangle inequality, and the error is
  self-serving in the worst way.** The reverse triangle inequality gives
  `|RMSE_full − D|`. Writing `max(0, ...)` discards the lower bound exactly when
  `D > RMSE_full` — i.e. whenever the candidate is bad — so a drift panel loses
  the only direction in which it can decide anything, and a decisive kill
  degrades into "undetermined". `local_sim/p2_pb_rank.py:839` already had it
  right; `p2_window.py` did not, and the entry built on it concluded "the panel's
  only decisive content is its failures", which was false. **A drift panel can
  kill decisively. Use the absolute value.**

- **Comparing two meters means comparing the same BINS, and a `flop_name` scope
  can silently move work between them.** Upstream bills the fresh two-leg block's
  `(2,1)` slice to `nonlin_sum 111 factored`, not to `get_dslice`, because
  `eval_part` populates `WK[3]`'s slice cache, `clone()` copies it, and
  `add_factors_` then takes its *incremental* branch inside
  `with flop_name('nonlin_sum 111 factored')` — where a bare module-level
  `_factored_get_dslice` inherits the enclosing name. `contract_wick_` is in that
  same bin. **Two successive prose accounts of this binning were wrong** — the
  first mis-attributed `~1.0e11` FLOPs, the second mis-stated the remainder by
  `4.31x` — and both looked completely reasonable. The fix was to stop arguing:
  `p2_weff.py --bincheck` proves the binning by measurement
  (`Δ(nonlin_sum) − 3n·2nl` constant to the FLOP across four increments) and
  closes the whole cross-meter gap **exactly**. **If a cross-meter claim matters,
  make it an assertion that closes to zero, not a table.**

- **A "control" that a symmetric object satisfies by construction is not a
  control** — the repository's signature failure, found twice more here.
  Permuting the *modes* of a lineage cannot change a symmetric tensor's diagonal
  slice (measured: `2.448e-16`, exactly the baseline), so it was relabelled an
  invariance and replaced by a *column* permutation. Separately, `late:16 == full`
  degenerates to the control (the checkpoint branch never fires) and is recorded
  as a labelled non-check. And prediction "counted cost is network-independent"
  was guaranteed before it ran, because the priced path has no data-dependent
  branching. **Before counting a check as evidence, ask what could make it fail.**

---

## 2026-08-25 (P1-PRE) — a falsification harness must not restore with Git

*Full record through the terminal-review PROVISIONAL verdict: bottom of
`experiments/LOG.md`.*

- **Historical warning, resolved after HEAD `0810537`:** the old
  `local_sim/p1pre_fault_inject.py` mutated repository source and then ran
  `git checkout --` on tracked evidence. The current implementation runs
  mutations in a scratch mirror and contains no executable destructive restore.
  Keep the lesson: a harness must never use version-control restoration as
  cleanup.
- **Resolved after HEAD `8e58db0`, and the shape of the defect is the lesson.**
  The harness's default `--out` was a *tracked* file; its in-repo refusal carved
  that one path out; and the "repository untouched" proof ran **before** the
  write over a protected list that omitted it. So a tool whose docstring said it
  never writes the repository wrote it on every invocation, and its own proof
  could not see the counterexample. **A guard scoped to "tracked files except
  mine" is not a guard, and a measurement taken before the event it certifies
  absent is not a measurement.** Now: the default is a temp path, *any* in-repo
  output is refused, the old transcript is in the protected set, and the
  measurement runs after the write.
- Run faulted code from temporary copies/scratch inputs. Snapshot repository status
  and tracked hashes before and after, and refuse any output path that aliases a
  tracked file. If cleanup is required for a test to be safe, the test is not yet
  safe enough to run in the shared workspace.
- A mutation is caught only when the **expected guard/reason** fires. A non-zero
  process exit can come from syntax damage, an incidental guard, or the harness
  itself; it is not proof that the intended property was tested.
- Mutation testing measures whether selected artificial faults are detected. It
  strengthens a suite but does not prove completeness. Stop after the named
  properties have independent behavioural oracles; do not recursively build a
  second framework to prove the first framework against arbitrary multi-edit
  sabotage.

---

## 2026-08-24 (P2-P0) — the evaluator was measuring a round that had ended

*Full record, contract and predictions: `experiments/LOG.md` 2026-08-24, P2-P0.
Raw artifacts: `experiments/2026-08-24_p2_p0/`.*

- **A constant can keep its spelling and change its meaning, and that is worse
  than being deleted.** whestbench 0.16 still exports `LAMBDA_FLOPS_PER_SECOND`
  and it is still `1e11` — as a **deprecated alias for the PHASE-1 rate**. The
  live default is `DEFAULT_LAMBDA_FLOPS_PER_SECOND = 0.0`. Our local
  `competition_score.LAMBDA_FLOPS_PER_SECOND = 1e11` therefore agreed with the
  package on the name and on the value while being wrong about the round. Neither
  a version bump nor a test that imports the name would have caught it. **When a
  phase changes, check what each constant MEANS, not whether it still resolves.**

- **λ = 0 in Phase 2, so `C == F`, and residual wall time stops being a cost.**
  It survives only as a **0.4 s per-MLP cliff** that FAILS the MLP — zeroed
  prediction, multiplier forced to 1.0. Everything in `VALIDATION.md` §2b about
  freezing a residual *price* model is Phase-1 law and decides nothing here
  (§2b-ii). What replaces it is headroom: `0.4 / observed`.

- **A price is smooth; a cliff is not.** Being 20 % out on a residual price is a
  20 % error in `C`. Being 20 % out near a cliff is the difference between scoring
  and failing. So the measurement path matters *more* now, not less.

- **At the Phase-2 shape the three residual instruments are INDISTINGUISHABLE, and
  a local margin is therefore not a safety factor.** Measured over three
  independent runs of 20 MLPs each on the shipped file
  (`local_sim/p2_residual_paths.py`), median residual per MLP:

  | instrument | run 1 | run 2 | run 3 |
  |---|---:|---:|---:|
  | `whest --runner subprocess` (authority) | 48.00 ms | 48.24 ms | 50.78 ms |
  | `whest --runner local` | 47.86 ms | 49.88 ms | 53.95 ms |
  | `lifecycle_eval` in-process | 51.26 ms | 45.87 ms | 53.20 ms |

  The lifecycle/subprocess ratio reads **1.068×, 0.951×, 1.048×** — it does not
  even hold its SIGN across runs, so the honest statement is that the three paths
  agree to within about ±7 % run-to-run variation, not that one is above another.
  **A single wall-time reading is not a measurement.** The first of these runs was
  taken while `preflight -Full` was on the same CPU and was discarded as
  contaminated before the others were run.

- **Do not use a universal ~7× residual rule.** Two Phase-2 graded K2 files ran
  about 7× faster than this host, but that ratio is evidence only for those exact
  files and sessions. A structurally different carrier/K3 candidate needs its own
  official-subprocess soak and graded telemetry when available. **The subprocess
  path remains the local deployment authority** because it matches the worker
  boundary, not because it is predictably slower or faster.

- **An instrument ratio does not bound a level, and only a level clears a cliff.**
  The ±7 % agreement above is real, and it was briefly used to characterise
  headroom against the 0.4 s cap. That was wrong: ±7 % is intra-session and
  instrument-to-instrument, while the cap tests an absolute second.

- **A residual level is not reproducible to better than ~2.4× across runs on this
  host, and the first explanation offered for that was wrong.** Best case, one
  quiet session, `--runner subprocess`, 20 MLPs per cell:

  | candidate | median | worst reading | over cap |
  |---|---:|---:|---:|
  | `estimator.py` (shipped K2p1) | 56.6 ms | 83.6 ms | 0/20 |
  | `arm_k2p1` | 55.9 / 59.4 ms | 95.5 ms | 0/20 |
  | `arm_k2p2` (the architecture's reference) | 94.6 / 95.1 ms | 140.6 ms | 0/20 |

  The same `arm_k2p2` reads **260.7 ms median / 390.6 ms worst — 1.02× headroom**
  in the frozen P2-CP1 ladder of 2026-08-21. **That figure is `lifecycle_eval`
  in-process** and the table above is subprocess, so the two are not directly
  comparable; the decomposition below is instrument-matched (lifecycle vs
  lifecycle) and that is where the 2.35× lives. **The gap was first attributed to
  machine contention. It is retracted.** Those are committed numbers from another
  day, not a contended re-run. Decomposed by three experiments:

  | candidate cause | measured | verdict |
  |---|---|---|
  | weight representation `np.asarray` → `fnp.array` | fixed **~19 ms** (18.9 / 19.7 ms, two arms) | real but small |
  | flopscope 0.10 → 0.12 | **0.91×** interleaved, 3 rounds, one session | **not the cause** |
  | run/session state | **~2.35×** | **the dominant term, and unexplained** |

  **Quote headroom against the worst observed run, not the best.**

- **Work the meter cannot see is billed to RESIDUAL, not to nothing.**
  `residual = wall − backend − overhead`, so before P2-P0 the sixteen 1024×1024
  float32→float64 weight conversions per `predict` — invisible on raw-NumPy
  weights — landed in residual. The repair moved that ~19 ms out of residual and
  into counted FLOPs. An uninstrumented operation is not free; it is merely
  charged somewhere you were not looking.

- **The "worst MLP" is often the FIRST MLP.** In three of five measured series the
  maximum is reading #1, and one decays visibly over ~7 MLPs
  (`95 82 77 93 94 94 74` → settles near `50`). Under a cliff a warm-up transient
  still fails an MLP, so do not discard it — but do not call it a heavy network
  either. And n = 20 is thin for a maximum: treat a worst-case headroom from 20
  draws as indicative, never as a bound.

- **The carrier's residual budget, and it is much tighter than the best case
  suggests.** `k2p2` costs ~95 ms of the 400 ms cap in the best observed session
  and **260.7 ms in the worst**, where its own worst MLP already sat at 1.02×
  headroom **with no carrier at all**. So the honest allowance for a carrier that
  must propagate 32,768 directions through sixteen layers is bounded by the bad
  day, not the good one: **~139 ms**, and near zero on that day's worst MLP.
  Residual is a first-class design constraint for Phase 1, not a box to tick.

- **Every preflight transcript this repository had ever committed was missing its
  verdicts, and they were cited as evidence of green runs.** `preflight.ps1` writes
  its step results and its final `PREFLIGHT GREEN / RED / n SKIPPED` summary with
  `Write-Host`. In Windows PowerShell 5.1 that goes to the **information stream**,
  which `2>&1 |` does not redirect. So `.\scripts\preflight.ps1 2>&1 | Out-File log.txt`
  captures every child process's stdout — hundreds of lines of convincing detail —
  and **none of the gate's own conclusions**. Audited 2026-08-24: three committed
  transcripts, 912 / 595 / 564 lines, `PREFLIGHT` verdict lines **0**, step
  `PASS` lines **0**.

  This is the failure class the gate's own accounting section warns about, arriving
  through the back door: not a check that reported a state instead of a property,
  but a check whose verdict never reached the file that was offered as proof of it.
  Found by an independent Skeptic, who could not locate a witness for a claimed
  "GREEN 16/16".

  **The three capture forms are not interchangeable, and none of them is obvious.**
  Measured, not assumed:

  | form | verdicts | child stdout |
  |---|---|---|
  | `2>&1 \| Out-File` | **no** | yes |
  | `-Log <path>` (Start-Transcript) | yes | **no** |
  | `*>&1 \| Tee-Object` | yes | yes |

  `Start-Transcript` in a `-NonInteractive` host with redirected stdio records the
  PowerShell host's own output but **not** the stdout of native child processes —
  so `-Log` guarantees the verdicts, which is the half that was missing, and not
  the suites' own output. **For a complete evidence transcript use
  `*>&1 | Tee-Object`.** Never `2>&1`.

  And whatever form you use: **when a transcript is committed as evidence, grep it
  for its own verdict line before believing it.**

- **A gate cannot catch what it does not look at.** For three days `preflight.ps1`
  reported GREEN 10/12 while running every check against whestbench 0.14 /
  flopscope 0.10 — the Phase-**1** stack — with the competition grading on
  0.16 / 0.12. Nothing in the gate examined its own environment. Step 1 now does,
  and step 12 asserts the counted-FLOP anchor op by op against a real graded-path
  report.

- **The flopscope 0.10 → 0.12 bump moved NOTHING the shipped file executes.**
  Measured on three arms: delta exactly 0 FLOPs, 0 operations moved. The four
  repriced primitives (`nansum`, `nanmean`, `where(mask, f32, python scalar)`) are
  ones it never calls. **No historical Phase-2 counted-FLOP number is invalidated
  by the version change** — the shortfall was entirely the two mechanisms above.

---

## 2026-08-21 (P2-CP0) — the Phase-2 stack, the floor, and a check that could not fail

*Full record and the fifth independent Skeptic verdict: `experiments/LOG.md`
2026-08-21, P2-CP0 and its review.*

- **The dataset CARD and `metadata.json` are different files, and the card carries
  the constants.** `metadata.json` has width, depth, N and the splits. **`B`, the
  multiplier floor, the no-bias / no-output-layer contract, the weight init, the
  input distribution and the forward-pass FLOP identity are only on the card.**
  This repository recorded "B and the floor are not in the dataset" for a day
  because it read one and not the other. **Read both, at the same commit.**

- **Mint parity can be proven against the ORGANISERS' PUBLISHED BYTES when the
  package is unavailable.** whestbench 0.16 exists on no public index, so the
  usual `canonical_mint == whestbench.sample_mlp` proof cannot reach the Phase-2
  release. It does not have to: under seed protocol 3.0 the parquet `mlp_seed`
  column is the **input** seed, so `canonical_mint(1024, 16, mlp_seed)` must
  reproduce the published `weights` bit for bit — and it does, 16/16 and 4/4 on a
  second shard. Those bytes *are* 0.16's output. **When you cannot get the tool,
  check against the tool's output.**

- **Score a residual against its own noise, not against a tolerance.** The
  published layer-1 mean is a Monte-Carlo mean over 1e9 draws, so "max relative
  error 2e-4" says nothing. Its noise is closed-form —
  `Var[ReLU(z)] = sigma^2 (1/2 - 1/(2 pi))` — so the residual must look like
  N(0,1), and it does: **rms z = 1.0017** over 16,384 neurons. That single number
  confirms the row-vector convention, no bias at layer 1, `x ~ N(0,I)`, and the
  truth-noise model at once. The **wrong** (column-vector) convention reads
  **rms z = 674.8**, so the statistic bites.

- **Do not quote an "expected maximum" from the Gaussian tail approximation.**
  `sqrt(2 log(2m/sqrt(2 pi)))` drops the `1/t` factor and overstates E[max]:
  4.354 against a simulated **4.136 (sd 0.286)** at m = 16,384. Use the **exact**
  order-statistic p-value `1 - erf(z/sqrt2)^m` instead. There is no reason to
  approximate a quantity that has a closed form.

- **At Phase 2 the multiplier floor binds for the whole K2 family, and that
  reframes the objective — but not the way it first looks.** The shipped analytic
  fallback measures counted `C/B` = **0.062556** and effective **0.071-0.079**,
  under the 0.1 floor, so its cost is free in score terms; residual would have to
  exceed **0.823 s/MLP** at lambda = 1e11 to lift it off. **It does NOT follow
  that the objective is "minimise MSE subject to C/B <= 0.1".** The objective is
  still `MSE x max(0.1, C/B)`, and above the floor the score is flat *only* for
  methods whose accuracy improves no faster than 1/compute. **An arm at C/B = 0.3
  must be scored, not rejected on sight.**

- **A whole estimator can price identically across two flopscope releases while
  the price table has genuinely moved.** `estimator.py` at 1024 x 16 costs
  **137,562,547,200 under both 0.10 and 0.12, op by op** — and 4 of 30 probed
  primitives still differ. The one that matters:
  **`where(mask, float32_array, 0.0)` is HALF price on 0.12**, because a Python
  scalar now follows NEP-50 weak promotion instead of being widened to 64-bit.
  That is exactly the spelling a mixed-precision path produces, and "a float32
  covariance halves `C/B`" is this project's headline cost lever. `nansum` and
  `nanmean` on float input went the other way, 2x dearer. **Price every arm under
  both releases** (`local_sim/p2_price_delta.py`). Three primitives agreeing is
  not a price table agreeing.

- **A check whose condition is the literal `True` is instance SIX of this
  repository's signature failure.** `p2_shape_smoke.py` shipped
  `check("S4 layer-1 closed form is exact", True, ...)`, computed a closed form,
  compared it to nothing, and had a comment describing a comparison against data
  the function never read. "8 of 8 passed" was 7 of 8. **Before writing a check,
  ask what input would make it fail.**

- **An evidence file that a documented invocation can shrink is not evidence.**
  Running the command printed in `phase2_mint_parity.py`'s own docstring
  (`--shard 1 --max-rows 4`) overwrote the committed 16-row artifact with a 4-row
  one. Name artifacts by what varies (`_shard0_rows16`), **refuse** to overwrite
  differing content, and never `os.remove()` past a refusal that exists on
  purpose.

- **A second implementation of the score identity hides inside "just reporting".**
  The smoke computed `max(floor, flops_used / budget)` for its artifact — counted
  FLOPs where the grader's multiplier is a function of **effective** compute
  `F + lambda R`. It disagreed with `competition_score` by 0.0626 vs 0.0684 and
  was hidden only because both were under the floor. At a candidate near 0.09 it
  would have written a false multiplier into the evidence.
  **Counted compares to counted; effective compares to effective; and only
  `competition_score` computes a multiplier.**

- **`claim()` records an artifact path BEFORE the artifact exists, so its hash is
  necessarily blank — and nothing used to fill it in.** `check_ledger` validated
  `if c.artifact and c.artifact_sha256`, so a blank hash passed silently and a
  truth set could be swapped after the arms were scored against it. Closed by
  `seed_ledger.finalize()`, which appends the hash after the file is written,
  refuses a file that does not exist, refuses an artifact no claim named, and
  refuses to re-close one whose hash moved. **Call it after writing every truth
  file**; an unfinalized artifact now turns the gate RED. An *absent* artifact is
  reported, never failed — `*.npz` is gitignored and a fresh clone has none.

- **The Phase-2 `full` split is 69.96 GB.** This machine has ~39 GB free, so
  `full` cannot be held here. Phase-2 CONFIRM work is `mini` (7.03 GB, seven
  1.119 GB shards of 16 MLPs each) unless disk is freed. One shard is enough for
  a provenance check or a smoke.

- **Published `mlp_seed` values are 63-bit.** Eight historical torch truth paths
  mask a sample seed with `& 0x7FFFFFFF`. They are deliberately **not** edited —
  they carry recorded Phase-1 results and their labels are unbiased — but every
  new Phase-2 truth path must use the full state. **Fix forward; do not rewrite
  the provenance of numbers already quoted.**

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
- **Prizes are decided by a private re-run of UP TO TWO designated submissions on FRESH MLPs.** Challenge Rules v14 states this explicitly; if none are designated, §6 selects the two best-scoring valid submissions from that Phase's public board. The old one-versus-two contradiction is resolved. Public-board rank is not a target; decide on fresh canonical mints and use the public mini only for CONFIRM under the binding three-tier protocol. The tail of the fresh-MLP score distribution is the real risk, not the mean.
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
- **The public score exposes 50 FIXED MLPs (indices 0–49)** with per-MLP feedback. Graded A/B comparisons on that suite are paired re-measurements, not population evidence. **Never decide or fit on it or on the public 100-MLP mini:** decisions use a newly claimed fresh canonical block; the public dataset is CONFIRM-only. Full historical record: RESEARCH_NOTES §2g (**private — never the forum**). **We do not use that channel.**
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

## Phase-2 gotchas added 2026-09-09 (Slice 4 day 1)

- **`B` IS A HARD PER-MLP CAP, NOT ONLY A MULTIPLIER — there is no "expensive but it still ships" regime.** `whestbench/scoring.py:709` wraps `estimator.predict` in `flops.BudgetContext(flop_budget=spec.flop_budget, ...)`; `:739` catches `flops.BudgetExhaustedError` and returns `fnp.zeros(...)` with `budget_exhausted = True`; `:1035` then forces that entry's multiplier to `1.0`. So a candidate priced above `1.0 B` does **not** score `raw × B` — it scores as **zeros**, roughly two orders of magnitude worse. `max(0.1, C/B)` is unbounded above only because under `λ = 0` a *valid* run cannot exceed `B`. A cost projection above `1.0 B` is a **failure**, not an expensive option, and any score arithmetic done on it is meaningless. This was got wrong once, on 2026-09-09, in the AUGMENT port price.
- **Fewer, bigger operations buy RESIDUAL and buy NOTHING on WALL.** Measured at `1024 × 16`, `k = 15` generations, on the shipped Strassen kernel: merging `_advance`'s three call sites into one cuts operations `210 → 71` and moves backend wall by `2%`; merging `_dslice21`'s four cuts `284 → 76` and makes wall **`28%` WORSE**. The Winograd glue is **memory-bound** — it costs *bytes moved*, not *calls issued* — so consolidating call sites moves the same traffic through fewer operations. Batching is a residual-only good. The repository had been treating "batch it" as a general good; `_Tower`'s batching is still right, but it was right for the op count and only for the op count.
- **At Strassen `L=3` the arithmetic is a quarter of the wall.** Winograd add/sub `37.2%` of backend wall for `1.3%` of FLOPs; concatenate `30.5%` for `1.6%`; the matmul leaves `26.6%` for `96.8%`. That is the mechanism behind graded intensity collapsing `215.13 → 22.82 GFLOP/s` between `L=0` and `L=3`, and the glue's share is *larger* on the grading server than on this host.
- **`estimator_k3_strassen._matmuls`' docstring is STALE and must not be fixed in place.** It says "1,022 matmuls"; the function returns **`797`**, and `797` reproduces the graded `L=0` anchor to `0.17%` while `1022` is `28%` high and describes an older build. **The file's sha256 `3ad05d75…` is submission `330327`'s submitted bytes — nominee A** — so editing even a comment breaks the property that the file on disk *is* the graded artifact. Correct it only in a successor file. This is the second orphaned scalar found in one slice (the first: `CURRENT_STATE`'s `215 → 17 GFLOP/s`, where the `17` does not reproduce). **Read constants from the function, never from prose about the function.**
- **The op-count gate is `22,000`, not `24,000`.** `docs/VALIDATION.md` §2b-iii retracted `24,000` as not conservative under the policy, after a Skeptic. `24,000` is still circulating in handoffs and in at least one instrument's prose.
- **`kind=AUGMENT` widens upstream's factored `K3` lineage from `2n` to `8n` columns per layer** (`5` legs + `2` live `211 → 111` blocks + `1` `from_dstensor`, against SIMPLE's `1 + 1`), and adds two rank-`R` contractions with no SIMPLE counterpart. Measured directly off `factors[0].shape[1]` at `(24,6)` and `(32,5)`. Any AUGMENT cost estimate that does not account for the lineage is pricing a few per cent of the bill — the fused `(2,1,1)` route everyone was worried about is `7.8%`.
- **Upstream's `factor_k4` needs `torch.no_grad()`.** It uses `out=` kwargs, and an `nn.Linear` weight still carrying `requires_grad` makes torch refuse them outright ("functions with `out=...` arguments don't support automatic differentiation"). Without `no_grad` a `k_max=4` arm does not fail loudly — **it does not run at all**, and a harness that catches the exception per-arm will silently report the arm as absent.

## Phase-2 gotchas added 2026-09-10 (Slice 5 day 1)

- **flopscope's ONLY matmul discount is object-identity aliasing in `einsum`, and it is worth exactly `2.00x`.** Measured across twelve spellings of the same arithmetic on the installed `0.12.1`: `fnp.einsum("ij,kj->ik", x, x)` with **one array passed twice** bills `0.5039x` dense. Everything else bills full price or worse — `x @ x.T` is `1.0039x`, `einsum("ij,kj->ik", x, x[:, :])` (a **view**, not the object) is `1.0020x`, and `einsum("ij,kj->ik", x*y, x)` (sharing a *factor*) is `1.0059x`. **Tagging an operand symmetric is a `1.8%` PENALTY, not a saving**: `as_symmetric(S) @ B` bills `1.0176x`, because cost is set by the output's unique elements and the contracted length, and `as_symmetric` adds a validation pass. If a future candidate ever produces a genuine `X Xᵀ`, spell it as the aliased einsum and pass the same object — the `@` spelling throws the discount away silently and nothing warns.
- **A `sha256` recorded by a `workerq` job is the sha of the SNAPSHOT's bytes, not of your working copy's.** `core.autocrlf=true` plus an LF index means a git snapshot materialises **CRLF**, so the same code hashes differently. The 2026-09-10 Linux memory gate certified `7d1be3b8…` for a file whose bytes on disk are `390bc226…`, and `certify_candidate` §9 — which matches on sha256 — would have SKIPPED. **For any check that must bind on the bytes you will package, submit with `--no-snapshot`.** The general form is worse and is still owed a fix: **a fresh clone of this repository does not reproduce the graded submissions' shas**, so every sha-based provenance claim in `LOG.md` is implicitly about *this working copy*. Owed: `.gitattributes` `*.py eol=lf`.
- **`certify_candidate` hands `predict` FLOPSCOPE-BACKED weights; a bare-numpy probe under-reads the meter.** `lifecycle_eval.build_weights` says why: *"flopscope-backed in every mode, because operations invoked as METHODS ON A WEIGHT are metered on the grader and free on bare numpy (that difference was 33.5M of the 37.8M FLOP shortfall the anchor closed on 2026-08-24)"*. Two probes written on 2026-09-10 handed bare numpy and under-read the depth-2 nominee by **`144,472,064` FLOPs and `414` operations** — `9,321` is the grader-faithful op count, `8,905` and `8,907` are not. **Read `build_weights` before writing any probe that quotes a FLOP or op count.** This trap was already found, closed and written down, and was walked into anyway.
- **The local subprocess runner reads residual `27x` high per operation, and TWO standard gates judge it in seconds.** On the depth-2 nominee this host measures `368 µs/op` against the grader's `13.4 µs/op`; `certify` §10's two clauses and any "mini-100 with `n_failed_mlps: 0`" gate therefore **refuse files the grader passes** — including `330327`, which holds a real graded `0/100` row. `VALIDATION.md` §2b-iii makes the **op count** the gate for exactly this reason. When you need a local mini-100 to mean anything, run it with `--no-residual-wall-time-limit`; with the cap armed every row is zeroed and **the run returns no accuracy signal at all**.
- **An ablation that deletes blocks is not necessarily a channel ablation.** In upstream's `factor_k3`, `K211_contrib` — κ₄'s own update, and the thing you would call the "cheap channel" — is computed from `pK_111.factors` (`factor_k3.py:473-497`), the very blocks a births-ablation deletes, and is added into `K_out[4].core` at `:531`. **So suppressing the births changes κ₄ itself from layer 2 onward.** Before claiming a component ablation isolates a channel, read whether any downstream term is a *functional* of the component you removed. A Skeptic found this on 2026-09-10 after the claim had already reached the write-up.
- **In this chain, ANY deletion from the carried `(1,1,1)` lineage costs essentially the WHOLE κ₃ correction, not a share of it.** Measured three ways on the same 24 networks: generation-capped SIMPLE `1.65e-06`; κ₄-births deleted `4.37e-06`; those plus the raised-budget legs deleted `2.98e-06` — against exact SIMPLE's `4.00e-08` and plain Gaussian K2's `4.20e-06`. **The ablated arms are the Gaussian baseline** (`K2 − LITE` straddles zero). **Always report `xmse_k2` and `xmse_k2cal` from `pa_k3.json` beside any ablated arm** — they are on disk for these exact seeds and cost nothing, and without them an arm sitting on the no-κ₃ baseline looks like a finding about whatever you removed.
- **The collapse is NOT monotone in how much you delete.** `AUGMENT-MINIMAL` removes strictly more diagrams than `AUGMENT-LITE` and is **`1.47x` better** (`2.98e-06` vs `4.37e-06`, 21/24 nets). Any reasoning of the form "it removes more, so it must be worse" is unsupported here.
- **`studentised_t`'s scipy-less fallback was `t` at the WRONG degrees of freedom.** `crit = 1.96 if n > 60 else 2.064` uses `df=24`'s value for every `n` from 2 to 60 (`2.064` against `2.0687` at `df=23`). scipy is installed in both venvs so it never fired and no published interval is affected — but it was one import failure away from a 12% error on an interval, and a `1.15x` gate would have inherited it. Fixed with an exact `df 1..30` table plus a Cornish–Fisher tail; **the artifact now records which branch produced the critical value.** An artifact that cannot say where its interval came from is not reproducible.
- **`Add-Content -Encoding utf8` from PowerShell DOUBLE-ENCODES UTF-8.** Two `LOG.md` appends on 2026-09-10 turned every `§` into `Â§` and every en-dash into `â€"`, 38 lines in total. **Append to `LOG.md` from bash (`cat >> file`), never from PowerShell.**
- **`seed_blocks.py` and `seed_ledger.TIERS` are two registries of one fact, and forgetting the second is a repeat offence.** Allocating `P07`/`P08` in `seed_blocks.py` alone left `tier_of('P07')` raising `KeyError` — the exact failure `SKEPTIC WP14-FOLLOWUP` found for `P05`/`P06`. The both-directions test added after that Skeptic **caught it before any instrument ran**. Add a block to both files in the same edit.

## Phase-2 gotchas added 2026-09-15 (Slice 6b)

**`certify_candidate.py` requires a QUIESCENT `local_sim/` for its whole run,
and it is right to.** Job `1352` certified `estimator_k3_v30.py` for 1h08m,
passed **46 checks**, and was then REFUSED by one:

    FAIL  R: this run changed no tracked source under estimator.py/scripts/local_sim
          THIS RUN DIRTIED: M local_sim/p2_build_v31.py

The dirtied file was an unrelated lab generator. The check does not and cannot
know that — `local_sim/` also holds `competition_score.py`, `lifecycle_eval.py`
and the mint, so an edit there genuinely can change a verdict mid-run. The cost
of the strictness is an occasional wasted hour; the cost of relaxing it is a
certificate that describes a tree that no longer exists.

**Operationally: commit, then certify, then do not touch `local_sim/` until it
finishes.** Editing anything under `estimator.py`, `scripts/` or `local_sim/`
while a certification is in flight throws the run away. Check with
`workerq status` before starting an edit session.

**A skipped section is not a pass, and the run says so.** The same job recorded

    SKIP  9: the real 8-GiB enforcement    no linux_memory_gate report FOR THIS CANDIDATE
    SKIP 10: official Phase-2 subprocess smoke    --quick/--skip-smoke; a skipped
             deployment check is NOT a pass

Both are named, both are real gaps in v30's evidence path, and neither is
closed by the 46 passes around them.

**`fnp.linalg.qr` does not raise on non-finite input.** It propagates NaN/Inf
like ordinary arithmetic and is fast doing it (`qr((1024,32))` on an all-NaN
matrix, 3 ms). So adding a LAPACK call to a path the invariant suite
deliberately floods with overflowed values does not introduce a new crash mode.
Verified because P2 puts a QR inside the birth, where sections 2 and 3 of
`edge_cases.py` push layers to `1e25` on purpose.

**Redirecting a long Python run to a file block-buffers its stdout.** Two
`edge_cases.py` runs sat at exactly 3,984 bytes — one buffer short of a flush —
and looked hung for minutes while progressing normally. Use `python -u` when
you intend to watch a redirected run, or you will debug a buffer.

**A `nohup ... &` inside the Bash tool does not survive its parent shell here.**
**A copied workerq snapshot cannot enforce a SINGLE_USE read cap by itself.**
Its `experiments/truth_reads.jsonl` and ledger lock are copies; two snapshots
can each see an unused block. For any future SINGLE_USE worker job, finish its
contract and validation package, then call
`truth_reads.py --prepare-snapshot` **on the host before snapshot dispatch**,
with absolute paths to the live authority register and ledger. Pass the returned
token, the exact frozen worker Git HEAD, consumer, purpose and those same live
paths to the worker's `truth_reads.load_snapshot_truth()` entrypoint. The
worker derives its own Git HEAD, checks truth and instrument bytes, activates
the token once under the host lock, and records completion or abort there.
Ordinary `truth_reads.reserve()` inside a
snapshot is unsafe for this tier. Preparation spends the one look even if the
job never runs, so do it only when the frozen job is ready to dispatch. The
sacrificial two-snapshot race and mutation checks live in
`local_sim/test_snapshot_reservation.py`; they touched no protected truth.

**A `nohup ... &` inside the Bash tool does not survive its parent shell here.**
One `edge_cases.py` run died silently at 68 lines. Use the tool's own
background mode, which does survive, and redirect inside the command if the
full transcript is wanted — the task output file keeps only the tail.

## 2026-09-25 — streaming the whole `full` split balloons the working set to ~36 GiB

`real_data.get(i, "full")` reads through HF `datasets.load_from_disk`, which MEMORY-MAPS
the split's arrow files. Touching all 1000 nets leaves ~36 GiB of mapped pages in the
process working set (private memory stays ~1.5 GiB), and worker-q then sees host RAM at
1–5% free and holds every other job. A writable `np.memmap` does the same on the write
side (worker-q 2134: 36 GiB on a 4 GiB declaration). Neither shows up as "private" RAM.
What to do: stream the split ONCE into `cache/twin/*_w.f32` (`tools/twin/export_weights.py`,
plain seek-and-write) and read nets with `np.fromfile(..., offset=)` (`tools/twin/twin.py
Data.net`). Anything that loops over hundreds of full-split nets through `real_data`
should declare `--ram 40` or use the export.

**workerq job outputs written relative to `__file__` land in the snapshot, not the repo**
(2026-09-28, jobs 2319/2325). A probe that saved `Path(__file__).parent / "x.json"` wrote
into `~/.local/state/gpuq/snapshots/<id>/job-<id>/...` and the live repo got nothing; copy
back from there, or write to an absolute repo path (`C:\Users\samsu\Documents\arc-whest\...`).

## 2026-10-06 — decided-neuron pruning (Lane P): four traps

1. **A compact inner dimension loses a Strassen level.** With `_STRASSEN_MIN_BLOCK = 32`, only
   width 1024 gets the 5th level. K = 992 then costs 4.4 % MORE than 1024. Compact chains ship
   block 20 (A) or 40 (B); see `experiments/2026-10-06_mask/strassen_probe.json`.
2. **The mean corrector reads `d21e = rowsum(D21²)/σ⁴`.** Any D21 mask changes that feature:
   two-sided masks cost +7–10 % raw through it alone. Ship row-only D plus no correction on
   the dropped rows (Dz). A twin diagnostic that forgets to feed the masked D21 to the
   corrector measures an unservable chain (this happened: commit `d36833f`, voided).
3. **Gathers bill 8 FLOPs per output element**, not 4. Selection (`argsort`) and `take` bill 0
   materializations. Scatter-back = zero-pad + `take` (never assign into an array).
4. **The `wall_class` law classes any block < 64 as leaf 32.** Block 40 builds are over-projected
   by ~45 %. Use a local alternating A/B against a parent with a measured server median.
