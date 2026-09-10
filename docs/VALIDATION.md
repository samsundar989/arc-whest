# VALIDATION.md — the truth harness

*Written 2026-08-06. This is the standing protocol for how anything in this repo
gets measured, and the record of the proofs behind it. It supersedes ad-hoc
practice everywhere it conflicts.*

The competition is decided by a private re-run of up to two designated submissions per team on
networks nobody has seen. Every number we quote is therefore a prediction about
unseen data, and a prediction is only as good as the instrument that produced it.
This document is the instrument's calibration certificate.

It exists because we found a real defect in the instrument. Until 2026-08-06 our
offline MLPs were **not** the grader's MLPs — same distribution, different
realisation, and with the estimator's random seed wrongly coupled to the weights.
Nothing we had measured was wrong, but nothing was reproducible against the
evaluator either, and the coupling was a live pathway for a spurious result. §1
is the fix and its proof.

---

## 0. The three-tier data protocol

Three tiers, three different jobs. **A number may only do the job of the tier it
came from.**

| tier | data | may be used to | may NOT be used to |
|---|---|---|---|
| **DECIDE** | freshly minted networks, a **new seed block per experiment**, never reused | choose constants, compare methods, kill or keep a track | — |
| **CONFIRM** | the real public split (mini 100 / full 1000, 1e9-sample truth) | check that a decision made on mints survives on the organizers' own data | choose *anything*. Read-only. **No constant may ever again be fitted here.** |
| **VAULT** | 500 sealed minted networks, high-precision truth, touch-counted | measure the level we expect on the private re-run | iterate. Three permitted touches, logged (§4). |

Why the middle tier is read-only now: the public mini split steered a month of
decisions. Its optimum is partly a memory of our own search, so a constant tuned
on it carries an unknown amount of our own noise into the private re-run. The
split is still valuable — it is the organizers' data with 1e9-sample truth — but
only as a *check*, never as an objective.

Why the first tier rotates seeds: if experiment A picks a constant on seeds
1..500 and experiment B re-measures it on the same seeds, B is not independent
evidence, it is A's noise re-served. Blocks are allocated once in
`local_sim/seed_blocks.py` and recorded in `experiments/LOG.md` with the entry
that consumed them.

**Standing rules.** Never tune on public data again. Decision seed-blocks are
never reused. Absolute claims need multi-shift or graded evidence. Every measured
MSE is reported alongside its ground-truth-noise floor (§3.5), so nobody
mistakes noise for signal.

---

## 1. Mint provenance — is our MLP the grader's MLP?

**Answer: it is now, bit-for-bit, and it was not before.**

### The authority

Both the public splits (`whestbench.dataset.bake_*`) and the private re-run
suites (`whestbench.scoring.make_contest`) build networks the same way:

```python
weight_ss, sample_ss, estimator_seed = SeedSequence(input_seed).spawn(3)
mlp = sample_mlp(width, depth, default_rng(weight_ss), seed=estimator_seed)
```

with `sample_mlp` drawing

```python
scale = float(sqrt(2.0 / width))
weights = [(rng.standard_normal((width, width)) * scale).astype(float32) ...]
```

Three substreams, deliberately independent: **spawn[0]** makes the weights,
**spawn[1]** draws the ground-truth Monte-Carlo samples, **spawn[2]** is the
`MLP.seed` handed to the estimator.

### What we were doing wrong

| | evaluator | ours, until 2026-08-06 |
|---|---|---|
| draw precision | float64 normal, float64 scale, **one cast at the end** | float32-native normal (`dtype=float32`), float32 scale |
| weight seed | `default_rng(SeedSequence(s).spawn(3)[0])` | `default_rng(s)` |
| `MLP.seed` | `spawn(3)[2]` — independent of the weights | **`s` — the weight seed itself** |
| GPU mirror | — | torch Philox, a different generator entirely |

The precision difference alone changes every number: `standard_normal(dtype=
float32)` consumes the bit stream by a different algorithm than the float64 path,
so the two agree in distribution and in nothing else.

The seed aliasing was the more serious one. `estimator.py:532` derives its Sobol
digital shift from `mlp.seed`. On our mints that seed *was* the weight seed, so
the point set was a deterministic function of the very network it was scoring —
a coupling the grader does not have, and exactly the kind of thing that
manufactures a result which then evaporates on unseen data.

### The proof

`local_sim/mint_provenance.py` reimplements the protocol in pure numpy (so it
also runs in the torch-only GPU venv) and checks it against the installed
`whestbench`:

```
uv run python local_sim/mint_provenance.py --seeds 20
```

| shape | seeds | weights bit-identical | `MLP.seed` identical | legacy generator identical |
|---|---|---|---|---|
| 256×32 | 20 | **20/20** | 20/20 | 0/20 |
| 256×8 | 20 | **20/20** | 20/20 | 0/20 |
| 128×16 | 20 | **20/20** | 20/20 | 0/20 |
| 64×4 | 20 | **20/20** | 20/20 | 0/20 |
| 256×2 | 20 | **20/20** | 20/20 | 0/20 |

Equality is checked on the raw `uint32` bit patterns, not on a tolerance.
`fnp.random.SeedSequence` **is** `numpy.random.SeedSequence` (identity), which is
what makes the reimplementation possible at all.

The legacy generator never matched, at a max relative elementwise divergence of
**1.868** — i.e. completely different numbers. It was, however, the right
*distribution*: over 12 × 256×8 networks, weight std / He target was 1.000329
(canonical) vs 1.000025 (legacy). **So past statistical results stand; past
per-seed reproducibility against the grader never existed.**

### The GPU mirror

`local_sim/gpu_parity.py` proves two separate claims to two different standards,
because they deserve different standards:

* **Generation is bit-for-bit** (equality). The mirror now mints on the CPU via
  `canonical_mint` and copies to the device; torch's own RNG cannot reproduce
  numpy's stream, only its distribution. 6/6 lossless float32 round-trips.
* **The forward pass agrees within a documented tolerance** (measurement). cuBLAS
  and OpenBLAS reduce a 256-term dot product in different orders and 32 layers
  compound it, so this cannot be an equality claim. Measured worst case over 6
  networks × 131,072 samples: max final-layer disagreement **5.55e-17**
  (relative 1.03e-16), implied MSE **1.2e-35** — 9e-30 of the ~1.4e-6 MSE we make
  decisions on.

### Consequences already banked

* Weights need not be stored anywhere. The vault keeps 500 seeds (4 KB) instead
  of 4.2 GB of float32, because the seed→weights map is now proven exact.
* `local_sim/harness.py:gen_mlp`, `robustness.py:mint` and `gpu_mint.py:he_mlp`
  all route through `canonical_mint`. The old generator survives as
  `gen_mlp_legacy` for reproducing historical numbers only.
* Ground truth is drawn from **spawn[1]**, so a vault network's truth is
  reproducible from its input seed alone, and is independent of both the weights
  and the estimator seed by construction rather than by assumption.

---

## 2. Seed-block registry

`local_sim/seed_blocks.py` is the single place that hands out input seeds.
Decimal digits are namespaces so a stray seed is readable at a glance.

| block | start | n | purpose |
|---|---|---|---|
| **V** | 20,260,806,000 | 500 | the sealed vault |
| D01–D05 | 30,806,001,000 … 30,806,005,000 | 600 each | constants audit (one per constant) |
| D06 | 30,806,006,000 | 600 | invariant-suite fresh draws |
| D07 | 30,806,007,000 | 600 | theory verification P1/P2/P3 |
| D08–D10 | 30,806,008,000 … 30,806,010,000 | 600 each | moonshot G1/G2 + renormalization fit |
| **D11** | 30,806,011,000 | 600 | **Sprint 4 DESIGN** — packed activation propagation. Prefix rule, bucket edges, Strassen depth, pilot size, on-neuron threshold. Tuning is permitted here and nowhere else. |
| **D12** | 30,806,012,000 | 600 | **Sprint 4 DECISION — LOCKED, SINGLE USE.** The S1 paired `V·c` gate, measured once against a frozen design. **SPENT 2026-08-06.** |
| **D13** | 30,806,013,000 | 600 | **Sprint 5 DESIGN** — round-trip compression. Batched-vs-recursive Strassen, bucket/segment Pareto, N ∈ {16384, 32768, 65536}, residual modelling. Tuning permitted here and nowhere else. |
| **D14** | 30,806,014,000 | 600 | **Sprint 5 DECISION — LOCKED, SINGLE USE.** The paired `V·C` gate (≥1.15×, 95% CI excluding 1.0), measured once against a design *and a residual model* frozen on D13. |
| L1–L5 | ≤ 9,000,000 | — | **legacy**, minted pre-fix; not comparable to anything after |

`check_disjoint()` is asserted by the invariant suite, so an overlapping block
fails CI rather than silently producing correlated "independent" evidence.

**But `check_disjoint()` proves only that blocks are ALLOCATED disjointly. It
cannot see a probe with a seed typed into it — and that is the failure that
actually happened.** For nine days `seed_blocks.py` said of D16 *"NO SEED WAS
EVER DRAWN. Reusable."* while five probes had drawn **270 of its 600 seeds**, and
**two of them had drawn the same 24 networks**: `probe_whitening` and
`probe_design_carrier`, whose kills (0.9972× and 1.047×) are recorded side by side
in `docs/CURRENT_STATE.md` under "the governing mechanism, measured three ways
today". Two of those three ways are one sample. Nothing was invalidated — both are
DESIGN tier and no gate was decided on either — but they are **correlated
evidence and must never be quoted as two independent readings**.

### 2a. The consumption ledger — added 2026-08-19 (P0.4)

Allocation is now separated from **consumption**. `local_sim/seed_ledger.py` and
the append-only `experiments/seed_ledger.jsonl` record who drew which range, when,
with which artifact and its sha256. Three layers:

| layer | what it stops |
|---|---|
| `claim()` / `claim_range()` | the sanctioned way to take seeds. Hard-fails on an unregistered block, a range leaving its block, **any** overlap in a DECISION block, or a second claim on a SINGLE_USE one. DESIGN overlap is permitted and always **reported** |
| `check_ledger()` | re-validates the whole ledger from scratch, so a hand-edited ledger cannot pass quietly. Catches out-of-block records, tier lies, and artifacts rewritten after their claim |
| `scan_sources()` | statically scans `local_sim/` and `scripts/` for seed literals **and** for `seed_blocks.seeds()` calls naming a block, with no ledger record. This is the layer that would have caught D16 |

Both run in **`scripts/preflight.ps1` step 7**, so the gate is executable rather
than prose. A file that names a block in order to *refuse* it — `moonshot_g0.py`
guards D08/D09/D10 — declares that with a `# seed-ledger: excludes ...` pragma.

Tiers govern strictness: **VAULT**, **SINGLE_USE** (one claim ever), **DECISION**
(overlap is a hard error), **DESIGN** (overlap permitted, always reported),
**INFRA**, **LEGACY**. `local_sim/test_seed_ledger.py` drives every refusal and
asserts it fires: **66/66**.

**Corrected block status, 2026-08-19.** D16 is **DESIGN, part-spent**: 270 seeds
drawn, 330 free in four gaps. D17 is **LOCKED SINGLE-USE** and verified never
drawn. **D18** (DESIGN) and **D19** (SINGLE_USE) were allocated 2026-08-19; D18
has since been drawn on for the interval-coverage study (192 seeds), D19 is held
for the v2.6 promotion gate so a Phase-2 shape change cannot burn the only clean
decision block. Ten of the twenty-one backfilled records are **upper bounds**
taken from CLI defaults where the consumer never wrote its range to disk —
over-recording consumption is the safe direction.

**Newly required of every experiment:** take seeds with
`seed_ledger.claim(block, n, consumer, ts=..., artifact=...)` and write the
returned range into the artifact. Never a literal.

`seed_blocks.SINGLE_USE_BLOCKS` names the blocks that may be consumed by exactly
one recorded experiment, so the rule is machine-readable instead of a convention
in a document. The LOG entry that spends such a block is its receipt: if a block
is listed there and appears in two entries, the second one is not evidence.

Labels for a decision block are produced by `local_sim/packed_truth.py`, which
draws ground truth from the evaluator's **own** sampling substream
(`seed_streams(input_seed)[1]`), so a block's truth is reproducible from its
input seeds alone — the same property the vault relies on. D11 and D12 each carry
200 labelled networks at 2²³ samples; measured GT-noise MSE **6.33e-9**, or 0.45 %
of the 1.4e-6 decision signal.

---

## 2b. Residual models — the rule that stops a cost claim from being a choice

> ### ⚠ SUPERSEDED FOR PHASE 2 — read §2b-ii before using anything below
>
> Everything in §2b is **Phase-1 law**. It exists to price residual wall time,
> because Phase 1 billed `C = F + λ·R` with **λ = 1e11**.
>
> **Phase 2 sets λ = 0.** `C == F` exactly. Residual wall time does not enter the
> multiplier by any path, so there is nothing left for a residual *price* model to
> decide. What replaced it is a **hard 0.4-second per-MLP cliff** that FAILS the
> MLP — zeroed predictions at multiplier 1.0 — and a cliff is not priced, it is
> cleared. Verified against the installed whestbench 0.16 on 2026-08-24
> (`local_sim/official_config.py::verify`, preflight step 1).
>
> §2b remains binding for **replaying a Phase-1 artifact**, and every word of it
> is still true of that round. Applying it to a Phase-2 decision would freeze a
> deciding model for a quantity the Phase-2 grader does not charge.

Counted FLOPs are measured. **Uninstrumented wall time is not measurable here at
all**: the grader bills `λ × residual_seconds` with λ = 1e11, and this machine
bills the *same file* ~7× what the grader billed it (70.8 ms/MLP locally against
9.84 ms graded for v2.4.1). So every cost comparison between a dense and a
non-dense carrier rests on a *model*, and a model chosen after seeing the answer
is not evidence.

Standing rule from Sprint 5: **three residual readings are reported, one is
frozen as the deciding model before the decision block is touched, and the
freeze is recorded in `experiments/LOG.md` before the number exists.**

| # | model | what it assumes | status |
|---|---|---|---|
| 1 | local absolute | this machine's level is the grader's | **never decides** — it is 7× off |
| 2 | local ratio × graded dense anchor (9.84 ms) | the candidate/control *ratio* transfers even though the level does not | plausible, unproven, **self-reported** |
| 3 | conservative packed model, anchored to submission 319341's published **185.8 ms/MLP at N = 84,992** | our packed carrier is no more residual-efficient per sample than the only packed carrier ever graded | **deciding model** |

Model 3 is conservative in a precise sense: it gives our implementation zero
credit for being better than someone else's, and it is anchored to a measurement
taken on the scoring hardware rather than on ours. Where the fixed/marginal
split of that 185.8 ms matters, the split is taken from **our own measured local
`r(N) = r₀ + r₁·N` fit**, so only the part that demonstrably amortises with N is
allowed to amortise.

Two arithmetic hygiene rules, both of which Sprint 4 broke:

* **Counted compares to counted; effective compares to effective.** 319341's
  published 2.374e6/sample is *effective* (residual included); its counted
  figure is ~2.144e6/sample. Comparing our counted number to their effective one
  manufactured a win that was not there.
* **Never a bare ratio of means quoted as a paired statistic.** On D12 the paired
  and unpaired readings were 1.1996× and 1.2026× — on opposite sides of the gate.
  Pair on common random numbers, always.

### 2b-ii. PHASE 2 — residual is a cliff, not a price (P2-P0, 2026-08-24)

`λ = 0`, so `C = F`. Three consequences, and each one retires a Phase-1 habit:

1. **There is no residual cost model to freeze for a Phase-2 decision.** The
   `max(local-ratio-on-graded-anchor, competitor-anchored)` rule of §2b decides
   nothing here, because the quantity it prices is multiplied by zero. A Phase-2
   contract that freezes one is answering a question the grader stopped asking.
   Freeze a **counted-FLOP** model instead; that is now the whole of `C`.
2. **What replaces it is a headroom margin.** The constraint is
   `residual_wall_time_s ≤ 0.4` per MLP, and exceeding it FAILS the MLP — the
   prediction is zeroed and the multiplier is forced to 1.0, which at our
   operating point is a ~10× multiplier penalty on top of a much larger MSE.
   Report `0.4 / observed` and call it headroom.
   `official_config.OfficialRound.residual_headroom()` is the only spelling.
3. **The measurement path matters more than it did, not less.** A price is
   smooth; a cliff is not. Being 20 % out on a price is a 20 % error in `C`;
   being 20 % out near a cliff is the difference between scoring and failing.
   This host's in-process residual is a *different instrument* from the graded
   one, so:

   - **the deployment authority is `whest run --runner subprocess`**, never the
     in-process figure, and never a local absolute;
   - a local headroom figure is quoted **as local**, with the word LOCAL in the
     sentence;
   - the direction of the local bias is **not assumed, and at this shape it is
     not resolvable from the launch-week local runs**: three runs of 20 MLPs give
     a lifecycle/subprocess ratio of 1.068×, 0.951×, 1.048×. Two later Phase-2
     graded K2 reports ran about 7× faster than this host, but that is evidence for
     those exact files, not a transferable safety factor for a new architecture.

   - **but an instrument ratio does not bound a level, and only a level clears a
     cliff.** The first version of this section quoted the ±7 % agreement above as
     if it characterised headroom. It does not, and an independent Skeptic was
     right to reject it. What the cliff compares against is the absolute second,
     so the absolute second is what must be reported, with its variation.

   **The measured law, and it took three experiments to get right**
   (`local_sim/p2_residual_paths.py`, `p2_weight_repr_residual.py`,
   `p2_residual_version.py`; artifacts under `experiments/2026-08-24_p2_p0/`).

   On one quiet host, one session, `--runner subprocess`, 20 MLPs per cell:

   | candidate | median | worst reading | over the cap |
   |---|---:|---:|---:|
   | `estimator.py` (shipped K2p1) | 56.6 ms | 83.6 ms | 0/20 |
   | `arm_k2p1` | 55.9 / 59.4 ms | 95.5 ms | 0/20 |
   | `arm_k2p2` — the architecture's reference | 94.6 / 95.1 ms | 140.6 ms | 0/20 |

   **Those are best-case numbers and must not be used alone.** The same `arm_k2p2`
   reads **260.7 ms median / 390.6 ms worst** in the frozen P2-CP1 ladder of
   2026-08-21 — **that figure is `lifecycle_eval` in-process, not subprocess**, so
   it is not directly comparable with the table above. The decomposition below IS
   instrument-matched (lifecycle against lifecycle: 260.7 ms then, 110.9 ms today
   at the same `np` weights), and that is where the 2.35× is measured (`experiments/2026-08-21_p2_cp1/ladder_frozen.json`,
   `per_mlp_residual_s`) — **1.02× headroom, effectively at the cliff.**

   That gap was first written up here as machine contention. **That was wrong, and
   it is retracted.** The frozen figures are a committed artifact from another day,
   not a contended re-run. Three experiments decompose the 2.35×:

   | candidate cause | measured | verdict |
   |---|---|---|
   | weight representation (`np.asarray` → `fnp.array`) | a fixed **~19 ms** slab, 18.9 / 19.7 ms across two arms | real, but small, and the wrong sign for this gap — both sides here are `np` |
   | flopscope 0.10 → 0.12 | **0.91×**, interleaved A/B, 3 rounds, one session | **not the cause**; 0.12 is marginally *slower* |
   | run/session state | **~2.35× unexplained** | **the dominant term** |

   The ~19 ms slab is itself a real finding: before P2-P0, sixteen 1024×1024
   float32→float64 weight conversions per `predict` were invisible to the meter,
   and since `residual = wall − backend − overhead`, invisible work is **billed to
   residual**. The repair moved that cost out of residual and into counted FLOPs.

   What remains unexplained is machine/session state and, plausibly, long-run
   process state — the ladder ran 11 arms × 96 networks in a single process.
   Nothing on disk separates those two, and neither did any experiment here.

   **So the rule is the opposite of what was first written:**

   > **A residual level is not reproducible to better than about 2.4× across runs
   > on this host.** Quote headroom against the **worst observed run**, not the
   > best, and never from a single session. A best-case reading is not a margin;
   > it is one draw from a distribution whose spread is larger than most of the
   > effects being measured.

   Two further cautions from the same data, both found by an independent Skeptic:

   - **The "worst MLP" is often the FIRST MLP, not a heavy network.** In three of
     five series the maximum is reading #1, and one series decays visibly over
     ~7 MLPs (`95 82 77 93 94 94 74` → settles near `50`). A warm-up transient
     still fails an MLP under a cliff, so it must not be discarded — but it must
     not be attributed to network heaviness either.
   - **n = 20 is thin for a tail statistic.** The cliff is a maximum, not a median,
     and a max over 20 draws has a wide sampling distribution. Treat a worst-case
     headroom from n = 20 as indicative, not as a bound.

   **ADDED 2026-09-02 (SKEPTIC WP14-FOLLOWUP) — the predictor is the OPERATION
   COUNT, and it must be reported in the same table as the FLOPs.**

   Everything above tells you how to *measure* residual. It does not tell you
   what *drives* it, and a whole slice was spent measuring the wrong axis because
   of that. Under `λ = 0` a FLOP below the floor costs **nothing**, so a cost
   table that contains only FLOPs is a table of the free variable. Measured with
   a matched control in one process, one host, one run
   (`local_sim/p2_k3_fnp.py --flow-price`, `gpuq` 144):

   | | counted FLOPs | flopscope **operations** | worst LOCAL residual | headroom |
   |---|---:|---:|---:|---:|
   | shipped K2 closure | `68.78 B` | **453** | `0.0277 s` | `14.4×` |
   | regenerating K3 flow | `173.38 B` | **5,327** | `0.2774 s` | **`1.4×`** |
   | ratio | `2.52×` | **`11.76×`** | **`10.02×`** | |

   > **SUPERSEDED IN ITS NUMBERS, NOT IN ITS LAW (2026-09-03, WP16/WP17).** The
   > K3 row above was measured on a flow that computed the `(1,1,1)` block —
   > which WP16 showed is **dead work** in a regenerating flow — and before
   > WP17's operation-count reduction. Current, same command,
   > `experiments/2026-09-03_wp16_parity/flow_price_wp17b.json`:
   >
   > | | counted FLOPs | flopscope **operations** | headroom |
   > |---|---:|---:|---:|
   > | regenerating K3 flow | `168.63 B` | **2,646** | *see below* |
   > | ratio to the K2 closure | `2.45×` | **`5.98×`** | |
   >
   > **The residual column is deliberately left blank, and that is the new law
   > here.** Four readings of the *matched K2 control* on this host on 2026-09-03
   > came back `0.0277`, `0.0322`, `0.0430` and `0.0340 s` — a 55% spread driven
   > by another project's jobs — and the K3/K2 **ratio moved with it** (`10.02`,
   > `11.98`, `12.76`, `7.28`). This section offered the ratio to a matched
   > control in the same process as the transferable quantity. **It transfers
   > only on a quiet host.** A local residual reading taken while anything else
   > is running is not evidence of either a level or a ratio, and the authority
   > remains `whest run --runner subprocess`.

   The paragraph below is unchanged and still correct: what tracks the residual
   is the operation count, and that is what a Phase-2 design must minimise.

   **`2.5×` the FLOPs buys `11.8×` the operations and `10×` the residual.** The
   two ratios that matter track each other and neither tracks the FLOPs. This is
   the same mechanism that killed the packed lane (`14,001` array ops per MLP)
   and the same one behind the `budget_summary_dict` ban.

   > **BINDING: every Phase-2 cost claim quotes counted FLOPs, flopscope
   > operation count and residual headroom against the worst observed run, in the
   > same table.** A claim with only the first is measuring the axis `λ = 0` made
   > free. `--flow-price` is the shape; `len(ctx.op_log)` is the count.

   The ratio to a matched control measured in the **same process, same host, same
   run** is the transferable form, because §2b-ii's own `2.4×` run-to-run spread
   applies to both arms at once and largely cancels. The absolute second still
   obeys every rule above: LOCAL, worst run, and never the deployment authority.

### 2b-iii. THE GRADED RESIDUAL IS PER-OPERATION — MEASURED, 2026-09-07 (`330212`)

*Corrected the same day, after an independent Skeptic. The first version of this
section compared two op counts taken in DIFFERENT WEIGHT CONVENTIONS, one of them
read off a file that never graded, and quoted a local residual from a superseded
build. Every number below is the corrected one; the errors and their effect are in
`experiments/LOG.md`, 2026-09-07 ERRATUM. Correcting them made the agreement
**better**, from 0.52% to 0.10%.*

§2b-ii said the measurement path matters more near a cliff than it does for a
price, and left open *which* path. It is now closed.

> **BINDING: the local `residual_wall_time_s` — in-process or under
> `--runner subprocess` — is a RATIO-ONLY instrument for a data-heavy candidate.
> It has no authority over a go/no-go. The gate is the OPERATION COUNT, in the
> grader's weight convention.**

#### The convention, first, because getting it wrong is how this went wrong

An op count is only comparable to another op count taken the same way. Three
local conventions exist and they differ:

| convention | k3_full | canary4 | what it is |
|---|---:|---:|---|
| f32 census | `3,232` | `2,673` | `p2_op_census.py`; float32 weights, no materialise |
| `whest run` mini | `3,233` | `2,674` | + `materialise_predictions` on the output |
| **grader twin (f64 read-only)** | **`3,249`** | **`2,690`** | **+ the opening `f64→f32` weight cast** |

Only the third reproduces the graded FLOP total, and it does so exactly:

    k3_full   f32 1,714,554,351,206 / 3,232
              + 32,768         + 1 op   materialise_predictions, 2 FLOP/elt on (16,1024)
              + 33,554,432    + 16 ops  the f64->f32 cast, 2 FLOP/elt x 16 x 1024^2
              = 1,714,587,938,406 / 3,249  == THE GRADED TOTAL, all 100 rows

    canary4   f32   168,636,374,630 / 2,673  -> 168,669,961,830 / 2,690  == GRADED

**Quote the grader-twin count, always.** `local_sim/lifecycle_eval.build_weights`
is what produces it offline.

#### The law

Two graded submissions on the same stack, `10.2×` apart in FLOPs:

| | ops/MLP | FLOPs | kernel wall | GFLOP/s | graded residual med | **per-op** |
|---|---:|---:|---:|---:|---:|---:|
| canary4 `329896` | 2,690 | `168.7 G` | `2.540 s` | 66.4 | `30.11 ms` | **`11.192 µs`** |
| k3_full `330212` | 3,249 | `1,714.6 G` | `7.970 s` | 215.1 | `36.40 ms` | **`11.205 µs`** |

`10.2×` the FLOPs and `3.2×` the arithmetic intensity move the per-op residual by
**`0.10%`**. Array traffic does not enter the grader's residual. Nor does it track
FLOPs by way of the kernel: `flopscope_server_kernel_wall_s` moved only `3.1×` on
that `10.2×` FLOP change, so **operations predict residual better than any
array-traffic proxy** — which is the actual claim, and it was tested rather than
assumed.

The local reading on the SUBMITTED bytes was `0.4363 / 0.4486 / 0.4667 s`
(`mini_k3_full_nocap.json`): it failed all 100 rows locally, passed all 100
graded, and **over-read by `12.3×`**.

#### Extrapolation — two points `1.21×` apart cannot license a `7×` gate

Our two direct points span `2,690 → 3,249` operations. A gate near 22,000 sits
far beyond both, in exactly the variable the law says matters. So the law is
corroborated against a third source, already committed and costing nothing:
`experiments/board_snapshots/board_2026-09-05_telemetry.md` carries median
overhead and residual for **twenty other graded submissions**.
`local_sim/p2_residual_law.py` calibrates overhead-per-operation on our own two
rows (`361.7` and `377.5 µs/op`, agreeing to `4.3%`) and inverts it:

    20 rows, inferred 6,223 - 33,818 operations   -- a 5.4x lever, STRADDLING the gate
    per-op residual   8.99 - 13.31 us,  median 10.14
    linear fit        residual = -0.023 s + 11.66 us/op * ops     R^2 = 0.939

A near-zero intercept and `R² = 0.939`: residual is proportional to operations
across a range five times wider than our own. **Those op counts are INFERRED and
are a model — never quote a row as another team's operation count.**

#### The gate, and how to size it

The policy on `GRADED_RESIDUAL_S_PER_OP` is *the maximum observed*, so it takes
the widest evidence available, not the most convenient:

    our worst single graded row                        12.096 us/op  (k3_full, one MLP)
    widest inferred row (neilarmstrong4 329639)        13.31  us/op
    carried figure                                     13.4   us/op
    ops at the 0.400 s cliff                           29,850
    **working gate, 25% margin**                       **22,000  (0.295 s, 1.36x)**

An earlier `24,000` was **not conservative** under this policy — two rows of the
visible population exceed the `12.1 µs` it was built on — and it was **not
derived from the data**: it was specified in the Slice 2 handoff before `330212`
was graded, and the entry that "derived" it was a retro-derivation. Both were
caught by the Skeptic. `local_sim/official_config.py::verify()` now RECOMPUTES the
per-op residual from the raw graded reports and refuses if the carried constant is
below any measured or inferred row, so this constant is scanned by a gate rather
than trusted as prose.

A candidate that passes the op gate but fails the local residual reading is
**admissible on the evidence above**, and that combination must be put to Samuel
with BOTH numbers — `certify_candidate.py` §10 continues to refuse on the local
number by design, because it is the deployment-authority runner and a silent pass
there would destroy the only conservative check we have. Relaxing hard rule 6 on
that basis remains Samuel's decision per submission, never an agent's.

**Scope.** Measured at the Phase-2 live shape (1024×16, float32) on whestbench
0.16.1 / flopscope 0.12.1 with a client/server backend. Our two direct points are
matmul-dominated files by one author; the twenty others reach us through an
overhead model an unusual operation mix could break (AndreasHad04 runs at `2.9
GFLOP/s` against our `215`). **Nothing here constrains curvature**, and no
experiment in this repository can — the grading server is not instrumentable from
here.

### 2b-i. THE BINDING STATISTIC — arithmetic, adopted 2026-08-20 (CP0.3)

**The primary promotion statistic is the paired ARITHMETIC score difference, with
the studentised-t or basic-bootstrap 95% interval. The geometric mean of per-MLP
ratios is a STABILITY DIAGNOSTIC and has no authority over any outcome.**

Samuel's decision, 2026-08-20. It replaces the rule — in force from 2026-08-06 to
2026-08-20 — that made the geometric mean binding.

**Why.** The competition scores `mean_i (mse_i × max(0.1, C_i/B))` over the public
MLPs. That is an arithmetic mean, it is reproducible offline **bitwise** from a
graded report, and it is what the private re-run will compute. The geometric mean
of ratios answers a different question: it discounts the heavy-tailed networks,
and the competition does not. On our own graded data the two **select opposite
winners** — v2.4 → v2.4.1 on the same 50 rows reads arithmetic **1.0334×** and
geometric **0.9666×**.

**A caveat that belongs next to that fixture, not buried:** the graded pair's own
paired t interval **spans zero** (sign test p = 0.48), so that comparison is a
*draw* under either estimand. It demonstrates that the two statistics *can* select
opposite winners on real data; it does not adjudicate which was right, and nothing
here claims it does. **The case for arithmetic is first-principles — the
competition computes an arithmetic mean — not empirical.**

**Which interval, and which are refused.** Measured, not assumed
(`local_sim/coverage_lab.py`, 192 fresh networks, block D18, 4,000 subsamples of
n = 50). False-promotion rate against a population recentred to a **true** mean of
exactly zero, nominal 5%:

| interval | measured false-promotion rate | status |
|---|---|---|
| studentised **t** | **4.1–5.4%** | **gate-eligible, default** |
| **basic** bootstrap | **4.0–5.2%** | **gate-eligible** |
| percentile bootstrap | 6.8–8.2% | **refused** |
| BCa | **9.1–11.5%** | **refused** — roughly double |

That inverts the textbook ordering, and the reason is measurable: common random
numbers make the paired *difference* nearly symmetric (skew +0.21, −0.55, −0.46)
even though each arm is strongly right-skewed, so BCa spends variance correcting a
skew that is not there. BCa only wins past skew ≈ 6, which no real arm reaches.

**The gate is executable, not prose.** `local_sim/competition_score.py::decide()`
is the only function in this repository that returns a promotion boolean. It
**requires** a pre-registered `gate` (no default), **raises** on `interval="percentile"`
or `"bca"`, and returns the geometric gain, the estimand-disagreement flag and the
biggest-mover share as diagnostics that cannot change the verdict. A pass requires
both the gain to clear the gate **and** the interval to exclude zero.

**Re-scoring the record — no verdict flipped.** Before adopting the change, every
recorded verdict with per-MLP data on disk was re-scored under both estimands
(`local_sim/rescore_verdicts.py`, artifact
`experiments/2026-08-20_rescore/rescore_verdicts.json`). Five reproduction checks
passed, including both graded aggregates bitwise and the D14 geometric headline to
1e-9. **No gated verdict crosses its gate.** In particular **D14** — the spent
single-use block that killed the packed lane at geometric 1.1216× against a 1.15×
gate — reads arithmetic **1.1177×**, i.e. *further below* the gate. The kill is
strengthened, not weakened.

---

## 3. The invariant suite

`local_sim/edge_cases.py` — runs on the **shipped file**, on every estimator
change. **30/30 passing** as of 2026-08-06 (was 24/24). The thirtieth check
asserts that the sealed vault's measured GT-noise floor stays below 10% of the
decision signal.

The original 24 asked *does it survive?* — degenerate pruning, a zero layer, 1e4
dynamic range, seed 2⁶³−1, widths 64–256, depths 2–64, budget 0, input
immutability. Finite output is necessary and nowhere near sufficient. The five
added classes ask *is it computing the right thing?*

### 3.1 Layer-1 closed form — exact

For x ~ N(0, I) the layer-1 pre-activation is exactly N(0, ‖W₁[:,j]‖²), so
E[ReLU] = ‖W₁[:,j]‖ / √(2π). No sampling, no closure, a known answer.

| shape | max relative error |
|---|---|
| 256×32 | 1.77e-16 |
| 128×8 | 1.64e-16 |
| 64×4 | 1.63e-16 |
| 256×2 | 1.72e-16 |

Machine epsilon. Any future drift here is arithmetic, not statistics.

### 3.2 Permutation equivariance — the test that catches silent math errors

Relabelling a hidden layer (permute the output columns of W_k and the matching
input rows of W_{k+1}) leaves the network's function identical. So the true
answer must satisfy `pred'[k] = pred[k][P]` and `pred'[l] = pred[l]` elsewhere.
This is the only check in the suite that catches a transposed index, an axis
swapped inside a reduction, or a basis applied on the wrong side — mistakes that
produce perfectly finite, perfectly plausible, wrong numbers.

> **AMENDED 2026-09-03, and one of this section's two jobs was RETIRED by the
> amendment.** Read the whole block before quoting any number below it. The
> change is recorded in `experiments/LOG.md` (*IMPORTANT (instrument)*), it was
> authorised by the strategist's Ruling 1, and an independent Skeptic returned
> **PROVISIONAL** on it — the finding that forced this rewrite is that the
> amendment retired a detector this section still advertised.

**The gate, as it now stands.** Every deviation a relabelling produces — the
permuted row, every other row, and the scored final row — is required to be, in
MSE units, **at most 1% of the candidate's own raw MSE**. The default
denominator is the frozen `1.4e-6` decision signal; `ARC_WHEST_RAW_MSE` may name
a candidate's own measured raw MSE and is clamped so it can only ever tighten.
Every component is tested for finiteness **before** any ordering.

**What was retired, and it is not a rounding detail.** The old gate also
required `max|pred'[k] - pred[k][P]| / mean|pred[-1]| < 1e-6` — roughly ten
times float32 epsilon. That is a bar only a float64 backbone can hold, and the
live design runs the covariance backbone in f32 deliberately, which is what puts
`C/B = 0.0767` on the `0.1` multiplier floor. **Measured on the same networks,
the amendment gives away a factor of 796×–1,552× of sensitivity** to positional
distortion (`local_sim/test_edge_cases_inv9.py` prints both floors on the same
ramp, every run):

| case | old clause rejected at | new clause rejects at | surrendered |
|---|---:|---:|---:|
| 128×8, layer 3 | `2.799e-07` of a row | `4.345e-04` | **1,552×** |
| 64×4, layer 1 | `4.884e-07` of a row | `3.887e-04` | **796×** |

**THE SECOND JOB THIS SECTION USED TO DO IS GONE.** The paragraph below used to
end by claiming this invariant *"doubles as an independent measurement of
v2.4.1's rotation stability… a full relabelling moves the final row by 4.4e-8."*
**That measurement lived entirely inside the surrendered band** — `4.4e-8` is
roughly three orders of magnitude below the new floor's implied deviation.
*(Written out rather than as a ratio: `test_enforcement_labels.py` scans this
document for `N` over `M` and reads any it finds as a suite-count claim, so a
bare ratio in prose turns the harness red for a reason that has nothing to do
with a suite.)* The invariant still *reports*
the deviation, and the number is still readable in its output; what it no longer
does is **gate** on it. **Do not cite invariant 9 as evidence of rotation
stability.** If that property needs a gate, it needs its own check with its own
threshold, and this document will say so when one exists.

**The measured deviations**, refreshed against today's `estimator.py` — the
previous table was stale by up to 75× in its *other rows* column, which is
exactly the rot a Skeptic caught and which is why the numbers are dated:

| case (v2.6.1, 2026-09-03) | permuted row | other rows | implied MSE | share of the 1.4e-6 signal |
|---|---|---|---|---|
| 256×32, layer 7 | 2.22e-15 | 7.75e-08 | 4.04e-16 | 3e-8 % |
| 256×32, layer 20 | 1.78e-15 | 1.93e-07 | 1.38e-15 | 1e-7 % |
| 128×8, layer 3 | 8.88e-16 | 1.06e-07 | 5.57e-16 | 4e-8 % |
| 256×32, layer 30 | 1.33e-15 | 1.59e-07 | 9.61e-16 | 7e-8 % |

The permuted row matches to floating point on the shipped file (`mse` at
`1e-31`). The residual in the *other* rows is not slop and is worth
understanding: reordering columns reorders the summation inside every downstream
matmul, so JJᵀ moves at the 1e-16 level and the alignment basis inherits it. We
therefore report the deviation as a share of the decision signal rather than
asserting an equality the hardware cannot deliver.

**A candidate at f32 reads three orders of magnitude higher on this invariant
than a candidate at f64, and that is a DTYPE fact, not an index bug.** Verified
two ways: the same flow on the same networks at f32 vs f64 moves `d_k/scale` from
`2.116e-06` to `2.365e-15` — a ~1e9 ratio, which is the eps ratio — and the
deviation **grows with width** (`1.105e-06` at 256×32 to `2.992e-06` at
1024×16), which is what BLAS accumulation noise does and what an index bug does
not. Before reading a permutation-equivariance failure as a math error, run the
f64 control.

**The standing proof that the item still bites** is
`local_sim/test_edge_cases_inv9.py`, a mutation battery wired into
`scripts/preflight.ps1` and `local_sim/certify_candidate.py`. Nine deliberate
defects — a transposed index, a swapped reduction axis, a depth off-by-one, a
basis on the wrong side, two positional swaps, and non-finite values on each of
the three gated rows — must all be REJECTED; two *equivariant* defects must be
ACCEPTED, because this invariant tests equivariance and not accuracy.

### 3.3 Budget monotonicity

More budget must never buy fewer samples, and every rung down to zero must
degrade rather than throw.

| budget | mode |
|---|---|
| 0 … 1e9 | analytic fallback |
| 1e10 | N = 2⁹ |
| 2.72e10 | N = 2¹¹ |
| **2.72e11 (Phase 1 B)** | **N = 2¹⁴** |
| 2.72e12 | N = 2¹⁷ |

Monotone, all outputs valid, no exceptions anywhere on the ladder.

### 3.4 Determinism

Three consecutive runs bit-identical, **plus** a freshly constructed `Estimator`
with its own `setup()` — which is the case that would catch state leaking
between MLPs through the cached Sobol table. Passes at 256×32 and 128×8.

### 3.4b The closure predicts its own sparsity — an accuracy check nobody asked for

Sprint 4 needed per-neuron firing rates to order columns, and had two sources: the
closure's `Phi(alpha)`, which is free because we already compute it, and an
empirical pilot pass. Comparing them turned into an independent accuracy check on
the closure itself. Measured post-ReLU zero fraction against the closure's
prediction, 256×32, canonical mints:

| layer | 1 | 4 | 8 | 15 | 23 | 30 |
|---|---|---|---|---|---|---|
| measured | 0.500 | 0.466 | 0.445 | 0.350 | 0.275 | 0.224 |
| closure | 0.500 | 0.466 | 0.445 | 0.350 | 0.276 | 0.225 |

Four decimals at every depth, including layer 30 where the closure's *means* are
18× worse than sampling. The Gaussian closure is badly calibrated in magnitude and
extremely well calibrated in **sign probability** — which is why pruning by
`alpha` works so well and why the analytic family died anyway. Any future design
needing a firing rate should take it from the closure and not pay for a pilot.

### 3.5 Ground-truth noise reporting

Every measured MSE is quoted with the MC noise of the truth it was measured
against, because an improvement smaller than that number is not an improvement.
For a truth of N samples the noise floor is `mean_i Var_i / N`, computed exactly
from the same pass. The vault's is **≈1.6e-9**, or 0.11 % of the 1.4e-6 raw
final-layer MSE we make decisions on.

---

## 4. The vault — our private-private test

500 networks, seed block V, ground truth at **2²⁵ = 33,554,432 samples each**,
built by `local_sim/vault.py`. It is the one set that stays unseen until it is
deliberately spent — the final re-run before the final re-run.

**Sealed** in the only sense available to a single-operator project: contents
hashed, hashes committed, and **every evaluation increments a counter and appends
a line to `experiments/vault/touch_log.jsonl`**. There is no quiet way to run
against it. If the counter says *n* and the log has *n* entries, the baseline you
are reading was not chosen by re-rolling until it looked good.

Permitted touches, and nothing else:

1. the v2.4.1 baseline, once, at seal time;
2. any future estimator change, once per change;
3. designation day.

Stored: `seeds`, `truth` (500, 32, 256), `truth_var`, `gt_noise` — **not
weights**, which §1 proves are a bit-exact function of the seeds. `manifest.json`
and `touch_log.jsonl` are tracked in git; the 33 MB array file is not (`*.npz` is
gitignored), so the seal travels with the repo and the bulk does not.

`vault.py verify` re-hashes every array against the manifest, reconstructs a
sample of networks from their seeds, and cross-checks the counter against the
log.

> **Baseline and touch record: see `experiments/vault/README.md`.**

---

## 5. Constants audit

Every frozen number in `estimator.py` was chosen at some point, on some data. The
audit re-measures each on freshly minted, never-reused seeds and asks one
question: *is the frozen value inside the measurement error of the fresh
optimum?* — not "is there a better value" (there always is, by noise).

Two design choices make that answerable at the ~1 % level, which a single
100-MLP run cannot reach (two realisations of the same estimator differ ~18 % in
mean MSE):

* **Paired, common random numbers.** Every configuration sees the same networks
  and the same digital shifts, so the comparison is on per-MLP *differences*.
* **Multi-shift averaging.** Each network is scored as the mean over 32
  independent digital shifts, removing point-set realisation noise instead of
  measuring it.

A constant "drifts" only if the fresh optimum beats it by more than ~2 standard
errors of the paired difference.

| item | chosen-on | fresh-mint verdict | action |
|---|---|---|---|
| pruning τ=-2.5 | public mini | **fresh optimum exactly -2.5** | keep |
| rotation rank 32 | 32-shift legacy-mint study | rank 256 appears 1.0108× better, only **0.7σ** | keep 32 |
| operating point N=16,384 | public mini + cost model | N=65,536 is 1.0408× better at **3.2σ** in the FLOP-only D03 audit | real-stack CONFIRM gave only ~1.007× after residual billing; do not adopt, retain for Phase-2 telemetry |
| pruned-value analytic-patch bound 3.8e-9 | legacy mint | **exceeded: 1.043e-8**, GT noise 1.65e-12 | withdraw old bound; end-to-end D01 still selects τ=-2.5 |
| v0 calibration table | legacy calibration mints | max shown fresh/frozen drift **0.07%** | keep table; note v2.4.1's low-budget fallback currently returns raw closure and does not apply it |

Full paired means are in `experiments/constants_audit_2026-08-06.json`; the
paired significance verdicts recovered after the interrupted output stream are
in `experiments/constants_audit_rank_2026-08-06.json` and
`experiments/constants_audit_cb_2026-08-06.json`. The narrative record is in
`experiments/LOG.md`. The public confirmation did not choose the candidate—the
choice came from D03—but measured the real residual-cost term omitted by the GPU
audit. Since the resulting gain was below the project's change threshold,
`estimator.py` was restored and the vault was not touched again.

---

## 6. Public–private divergence — NOT VISIBLE, and what is

Thread 18118 describes Phase 1 as 50 public + 50 private MLPs with
"public-private divergence" monitored. From the participant side the mechanism is
confirmed and the divergence is not observable:
`results.per_mlp` carries **100** entries, `results.public_scores` carries **50**
(indices 0–49), and `aggregates` has a `public` key with no sibling. An
exhaustive key search across all four graded reports
(`local_sim/private_half.py`) finds **no private score field of any kind**.

What *is* visible is the hidden half's compute telemetry, which answers a
narrower but real question — does the estimator behave the same on draws it was
never tuned against?

| submission | private-half state | errors | effective-compute ratio (private / public) | residual ratio |
|---|---|---|---|---|
| 323872 (v0) | completed | 0 | 0.9999 | 0.9992 |
| 324075 (v1.0) | completed | 0 | 0.9999 | 0.9885 |
| 324094 (v1.1) | completed | 0 | 0.9999 | 0.9940 |
| 324420 (v2.4) | completed | 0 | **0.9916** | 1.0006 |

v2.4 spends 0.84 % *less* compute on the hidden 50 — its pruning threshold prunes
marginally more there. That is the cost half of "no public–private divergence",
measured. **The accuracy half is not observable and we do not claim it.** For the
write-up's integrity section, state it exactly that way.

---

## 7. What runs on every estimator change

```powershell
uv run python local_sim/mint_provenance.py              # 1. provenance still proven
uv run python local_sim/edge_cases.py                   # 2. invariants, 30/30
uv run whest validate --estimator estimator.py          # 3. contract
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini --runner subprocess   # 4. Phase-2 CONFIRM (read-only)
.venv-gpu\Scripts\python.exe local_sim/gpu_parity.py    # 5. mirror still exact
.venv-gpu\Scripts\python.exe local_sim/vault.py evaluate --label "<version>" --reason estimator-change
```

Steps 1–3 are cheap and unconditional. Step 4 is a check, never an objective.
Step 6 costs a vault touch and is spent once per change, not once per idea. The
existing Phase-1 vault is not a Phase-2 vault; do not run that example for a
Phase-2 candidate unless Samuel has explicitly authorized creation/use of the
separate Phase-2 vault and the command has been updated accordingly.

**All of steps 1–3, plus the ship-file identity check, the vault seal and the
seed ledger, are wrapped by `scripts/preflight.ps1`, and it is the gate.** Run it
on every estimator change.

**Since 2026-08-20 the gate has TWO TIERS** (CP0.1; `experiments/LOG.md`). Four
suites and the vault's own hash verification used to pass ungated, so "preflight
GREEN" was a weaker claim than the entries quoting it assumed:

**Renumbered to SIXTEEN on 2026-08-24 (P2-P0).** Four steps were added and every
earlier step shifted by one. The table below is the current gate; where an older
LOG entry cites "step 6" or "10 of 12" it is describing the pre-2026-08-24 gate and
the number has moved.

| tier | steps run | when |
|---|---|---|
| **default** | **12 of 16** — **1** official stack + round configuration (96/96, asserted against the *installed* package; WP6 grew it by nine checks against the STORED GRADED REPORTS, for the grading server's per-MLP FLOP offset, and the Slice-2 work grew it again as the offset's cast model gained its second and third graded rows — 329896 and 330212, both offset-zero), **2** mint provenance (1024×16 first) **+ the dataset pin** (`local_sim/test_p2_tag_drift.py`, added 2026-08-31 by WP0A; two commands, one step, so the denominator stays 16), **3** invariant suite (30/30), **4** `whest validate`, **5** ship-file identity, **6** vault seal, **7** seed ledger consumption + source scan (no count; it reports CONSISTENT or refuses) **+ the enforcement labels** (`local_sim/test_enforcement_labels.py --labels-only`, added 2026-08-31 by WP0B; two commands, one step, so the denominator stays 16), **8** canonical scorer + the binding gate (134/134), **9** seed-ledger *refusals* (66/66), **10** vault integrity, **11** lifecycle evaluator (56/56), **12** counted-FLOP anchor vs the official runner | every change |
| **`-Full`** | **16 of 16** — adds **13** shape parity, **14** carrier strength, **15** deployment envelope (+ what it cannot prove) **and its read-only proof** (`local_sim/test_envelope_readonly.py`, added by P1-PRE 2026-08-25), **16** official Phase-2 subprocess smoke, zero failures | **required** before any promotion, submission, or vault touch |

> **Suite counts have been removed from the `-Full` row.** They moved three times
> in two days during P1-PRE and an independent Skeptic caught this table
> contradicting the transcript committed beside it twice — and then caught the
> *repair* of that row leaving it with two cells instead of three. **The
> authority is the gate's exit code**, exactly as the paragraph on step names
> below already says. The same reasoning retires line-number citations into any
> transcript that is regenerated on every run: those drift by construction, so
> cite a grep string instead. Counts survive in the default row because that row
> has not been the one moving; if it starts, it loses them too.

**Step 16 is DEPLOYMENT AUTHORITY** — the grader's own subprocess runner on the
organisers' own `v2-phase2` data — and it is **load-sensitive**: the 5 s setup
ceiling covers worker spawn *plus imports*, and `import whestbench` alone is 1.3 s
on a quiet host and 4.7–9.1 s under load. It has failed twice under contention.
Run the gate on a quiet machine, and read a `SETUP_TIMEOUT` as an environment
condition rather than an estimator defect — the gate says so itself, and still
fails, because a gate that could not verify deployment must not certify it.

*Wall-time figures recorded for earlier tiers (111 s for the nine-step gate,
~135–160 s for the twelve-step default, ~755 s for its `-Full`) are not comparable
with the sixteen-step gate and are kept only as history.*

The split is by **cost, not importance**.

**Step names carry no check counts.** A count in a runtime label is a snapshot and goes stale on the next test added; this gate had three such bumps in a single day. The verdict is the script's exit code.

**The counts in the table above are UNDER TEST, not merely documentation,
and the test RUNS IN THE GATE** — `scripts/preflight.ps1` step 7, second command.
`local_sim/test_enforcement_labels.py` runs the suite each count describes and
fails on a disagreement. Added 2026-08-31 (WP0B) because three counts were stale
and this row attached step 9's `60/60` to step **7**, which reports no count at
all.

It scans **exhaustively**, not from a hand-list. Every `N/N` and every prose
count (`60 negative tests`) anywhere in this file or in `preflight.ps1` must be
either matched by a checked claim or named in `ALLOWED_COUNTS` with a written
reason — step denominators, the mint-provenance table, quoted historical values.
The first version used an anchored hand-list and an independent Skeptic found
two counts it structurally could not see: `preflight.ps1`'s step-9 body comment
(*"44 negative tests"*, prose, no `N/N` to anchor on) and §2a's *"asserts it
fires: **39/39**"*, both against a suite that reports 60. A pattern that stops
matching is now a failure rather than a silent `continue`.

**The denominator is always 16, and a skipped step is never counted as a pass.**
A default run reports `GREEN 12/16, 4 SKIPPED` and names the four in a red
NOT-CHECKED block; if the vault arrays are also absent it reports `11/16, 5
SKIPPED`. The first build of this gate reported `9/11` as `9/9` and printed step
9's skip as a green PASS — caught by the independent Skeptic, and it is the same
failure this repository has lost four separate checks to: **a check that reports a
state rather than testing a property.** Quote the tier when citing a green run;
"preflight GREEN" alone is now ambiguous.

Step 10 reads no candidate predictions and therefore **costs no touch** — verified
twice over, by counter and log line count (1 → 1) and by the mtime of
`manifest.json` and `touch_log.jsonl`, both unmoved from 2026-08-06. It degrades to
a printed SKIP when `experiments/vault/vault_v1.npz` is absent, which is the normal
state of a fresh clone because the 33 MB array file is gitignored. Step 6 still
covers the seal in that case, because `manifest.json` and `touch_log.jsonl` *are*
tracked.

**Step 11 is `local_sim/lifecycle_eval.py`, the canonical grader-lifecycle
evaluator** (CP0.2). It is the object every candidate comparison is scored
through: one `Estimator`, `setup()` called once **off-budget** and timed, per-MLP
`BudgetContext`, all six failure paths **provoked rather than asserted**, float32
MSE, and immutable content-addressed evidence rows. It does not re-derive the
score identity — it produces `MlpRow`s and hands them to `competition_score.py`.

Three things about it are binding and easy to get wrong:

1. **The objective it prints is NOT a graded objective.** Every
   `residual_wall_time_s` is a `--runner local` figure read from the in-process
   `BudgetContext`; the grader deliberately prefers `last_predict_stats()`
   (`scoring.py:762-789`). λ·R is ~10–13% of `C` here against 2–3% on the grader.
   Counted FLOPs transfer exactly; the objective does not. **Ratios inside one
   path only.** The evidence bundle records `residual_is_graded: false`.
2. **A setup failure is a RUN-level refusal, not a per-MLP failure.** `cli.py:2357`
   calls `runner.start()` *outside* the try wrapping `evaluate_estimator`, and
   `runner.py:155-172` raises `SETUP_ERROR`/`SETUP_TIMEOUT` out of it — the whole
   submission dies. `evaluate()` therefore raises `SetupFailure` by default rather
   than returning a clean-looking objective. This is the property that lets the
   evaluator be trusted to price a candidate with an expensive `setup()`.
3. **`setup_timeout_s`, `wall_time_limit_s`, `residual_wall_time_limit_s`,
   `predict_timeout_s` and `memory_limit_mb` are INSTALLED-PACKAGE DEFAULTS, not
   observed grader configuration.** Every graded `config_snapshot` we hold carries
   exactly `n_mlps`, `depth`, `width`, `flop_budget`. Two of the six failure paths
   are therefore gated on constants we have never seen in a graded report — quote
   them as a prior, never as a fact.

   Updated 2026-08-24 (P2-P0): they are no longer *transcribed* priors. They come
   from `local_sim/official_config.py`, which asserts each against
   `whestbench.budget.RoundConfig` and `scoring.ContestSpec`, and additionally
   against the `run_config` block of a real `whest run --json` — which records
   what the runner **used**, not what it declares. That is a much stronger prior
   and it is still not a graded fact. The Phase-2 values are: setup 5 s, wall
   120 s, residual cap **0.4 s**, predict 30 s, memory **8,192 MB**, λ **0**.

   Note the trap the old wording sat on. λ = 1e11 *was* corroborated — for
   Phase 1. whestbench 0.16 still exports `LAMBDA_FLOPS_PER_SECOND = 1e11` as a
   **deprecated alias meaning the Phase-1 rate**, while the live default is
   `DEFAULT_LAMBDA_FLOPS_PER_SECOND = 0.0`. Same spelling, different meaning, and
   a local constant that agreed with the package on the name and not on the value
   is exactly how the stale default survived a phase change unnoticed.

Timings are quoted with an n and a range because the first measurement of the
default tier (103.9 s) did not replicate — an independent re-run measured 113.4 s,
and three replications give 111 s [110, 113]. Nothing turns on it against a target
of ~2 minutes, but a single unreplicated wall time is not a measurement.

Step 9 is not a duplicate of step 7 (they were steps 6 and 8 before the
2026-08-24 renumber). Step 7 proves the ledger is *consistent*;
step 9 proves its *refusals still fire* — unregistered block, range leaving its
block, overlap in a DECISION block, a second claim on a SINGLE_USE one, a
hand-edited ledger, an artifact rewritten after its claim.

**Two things changed on 2026-08-19 that this section has to carry:**

- **`vault.py evaluate` now takes `--candidate <file>`** and runs that file under
  a real `flopscope.BudgetContext`, so counted FLOPs are *measured* rather than
  modelled, and the touch record carries the candidate's **sha256**. `--dry-run`
  is **gone** — reading the truth *is* the touch. A separate `sacrificial`
  subcommand exercises the identical path against a non-vault truth set and
  records nothing. **Consequence: touch #1 (v2.4.1, 2026-08-06) was produced by
  the OLD evaluator — a torch mirror of v2.4 priced by a v2.4 cost model — so it
  is not comparable with any future touch.** See `docs/CURRENT_STATE.md`
  § *Next actions* #3.
- **The residual model of §2b is superseded for the grader.** `residual ≈
  2.551 ms + 5.780 µs/op`, R² = 0.898, LOO 0.829 ms, fitted on **five** graded
  submissions (`local_sim/residual_model.py`). The three-model rule remains the
  fallback where no graded anchor exists; where one does, use the line.
  **Both are Phase-1 statements.** Under Phase 2's λ = 0 the fitted line prices
  nothing; it survives only as a *predictor of seconds*, which is still useful for
  estimating headroom against the 0.4 s cliff (§2b-ii) and useless as a cost.
