# VALIDATION.md — the truth harness

*Written 2026-08-06. This is the standing protocol for how anything in this repo
gets measured, and the record of the proofs behind it. It supersedes ad-hoc
practice everywhere it conflicts.*

The competition is decided by a private re-run of one designated submission on
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

Both run in **`scripts/preflight.ps1` step 6**, so the gate is executable rather
than prose. A file that names a block in order to *refuse* it — `moonshot_g0.py`
guards D08/D09/D10 — declares that with a `# seed-ledger: excludes ...` pragma.

Tiers govern strictness: **VAULT**, **SINGLE_USE** (one claim ever), **DECISION**
(overlap is a hard error), **DESIGN** (overlap permitted, always reported),
**INFRA**, **LEGACY**. `local_sim/test_seed_ledger.py` drives every refusal and
asserts it fires: **39/39**.

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
* **The paired statistic is the geometric mean of per-MLP ratios**,
  `exp(mean(log(V·C_ctrl / V·C_cand)))`, not the ratio of the means. On D12 the
  two read 1.1996× and 1.2026× — on opposite sides of the gate.

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

| case | permuted row | other rows | implied MSE | share of signal |
|---|---|---|---|---|
| 256×32, layer 7 | 2.22e-15 | 4.38e-08 | 1.18e-16 | 8e-9 % |
| 256×32, layer 20 | 1.78e-15 | 2.07e-08 | 3.43e-17 | 2e-9 % |
| 128×8, layer 3 | 8.88e-16 | 1.42e-09 | 1.97e-19 | 1e-11 % |
| 256×32, layer 30 | 1.33e-15 | 4.96e-09 | 1.23e-18 | 9e-11 % |

The permuted row matches to floating point. The residual in the *other* rows is
not slop and is worth understanding: reordering columns reorders the summation
inside every downstream matmul, so JJᵀ moves at the 1e-16 level and the alignment
basis inherits it. We therefore report the deviation as a share of the decision
signal rather than asserting an equality the hardware cannot deliver. At ≤1.2e-16
implied MSE it is nine orders below anything we would act on.

This doubles as an independent measurement of v2.4.1's rotation stability: under
the pre-v2.4.1 raw eigenbasis, a 1e-15 weight perturbation moved the basis by
0.81 in max column distance. Here a full relabelling moves the final row by
4.4e-8.

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
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 --split mini --runner subprocess   # 4. CONFIRM (read-only)
.venv-gpu\Scripts\python.exe local_sim/gpu_parity.py    # 5. mirror still exact
.venv-gpu\Scripts\python.exe local_sim/vault.py evaluate --label "<version>" --reason estimator-change
```

Steps 1–3 are cheap and unconditional. Step 4 is a check, never an objective.
Step 6 costs a vault touch and is spent once per change, not once per idea.

**All of steps 1–3, plus the ship-file identity check, the vault seal and the
seed ledger, are wrapped by `scripts/preflight.ps1` — now SIX steps, and it is
the gate.** Run it on every estimator change.

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
