# Independent Skeptic — Slice 6a λ, 2026-09-14 (night)

Fresh context, never saw the builder's reasoning. Supplied with the contract,
the diff (`4ad0f02`, `f7778fd`), the instruments and the **raw artifact paths**;
it recomputed from `lambda_fit.npz` rather than from the logs.

## Verdicts

| claim | verdict |
|---|---|
| **λ: "an independent label-free fit lands within a factor of two of 504aldo's at every one of fifteen layers"** | **REJECT** |
| **memory: "the unfactored AUGMENT chain cannot be run at 1024×16"** | **PROVISIONAL** — conclusion accepted, the quoted numbers are not |

## Why the λ claim is rejected — the four findings I accept in full

**1. "Within a factor of two at every layer" cannot fail, so it is not a test.**
504aldo's table spans only `4.99e-3 → 1.085e-2`. Against our extrapolated
table the Skeptic ran controls:

| control | layers within 2× |
|---|---|
| a **random permutation** of their table | mean 14.5/15, `P(≥14 of 15) = 1.000` |
| a **single constant** at their table's mean | **15/15** — better than the real comparison |
| a deliberate **+1 layer index shift** | 14/14, median 0.775 — also better |

The criterion passes for anything of the right order of magnitude. And the
quantity with actual content — the depth *shape* — does **not** match: Pearson
`r = 0.102` over 15 layers.

**This is the finding that matters, and it is mine to own.** The pre-registered
gate was *within 15%*. It failed. I then reported "within a factor of two",
which is a criterion I invented after seeing the answer. That is precisely the
move Rule 1 exists to prevent, and no amount of caveat elsewhere repairs it.

**2. The agreement is a property of the fit window.** Leave-one-width-out,
recomputed with the same `powerfit` code:

| widths fitted | median ratio | range | within 2× |
|---|---|---|---|
| 64, 96, 128 (what I reported) | 0.744 | [0.499, 1.940] | 14/15 |
| **96, 128 (drop 64)** | **0.213** | **[0.046, 2.003]** | **3/15** |
| 64, 96 (drop 128) | 1.893 | [0.842, 5.727] | 8/15 |
| 64, 128 (drop 96) | 0.689 | [0.401, 1.946] | 13/15 |

At `l=9` the 64→96 slope is `p = 0.97` and the 96→128 slope is `p = 2.76`.
Over an 8× extrapolation `d(ln λ)/dp = −ln 8`, so that 1.79 in slope is a
factor of **41** in the answer. The single power law is absorbing real
curvature.

**3. `n=64` fails my own exclusion criterion, and it is load-bearing.** The
retained log was a *filtered* copy; the 1,113 removed lines were
`Snapping negative variance to zero`, and they are not uniform:

    n= 48   840 snaps   17/24 nets      <- excluded as "degenerate"
    n= 64   231 snaps    7/24 nets      <- KEPT, and it sets the low-width end
    n= 96     0 snaps     0/24
    n=128     0 snaps     0/24

I justified dropping `n=48` by that exact symptom. By the same criterion `n=64`
is degenerate too, and the only two numerically clean widths give median
**0.213** and **3 of 15**. At `n=64, l=14` the per-net R² min is **0.059** with
4 of 24 nets below 0.5 — the contract's own stop condition, triggered at the
width I kept.

Worse, the rule **as coded** (`p2_lambda_extrapolate.py`) is a *NaN detector*,
not a degeneracy detector. It fired on `n=48` only because of one NaN at l=14;
it would have admitted `n=48`'s `3.11e+04` at l=13 otherwise, and it cannot see
`n=64`'s problem at all.

**4. The literal sentence is already false.** The artifact prints
`range [0.499, 1.940]`. `0.499` is outside a factor of two.

Plus: the one layer with an external anchor (`l=0`, where an independent
n=1024 instrument gives `9.1423e-3` against our extrapolated `9.6814e-3`, a
genuine 5.9% out-of-sample hit **that I did not claim**) is the layer where we
*disagree* with 504aldo at 1.94×. My "we agree at the ends" omitted that `l=0`
is an end.

## What survived

- **The R² claim survives every attack the Skeptic made on it.** Per-net R² is
  unimodal and tight — at `n=128` the sd is 0.015–0.111 and only 2 of 360
  net×layer cells fall below 0.5. *"R² ≥ 0.6 by layer 4 → CONFIRMED"* is
  correct, and holds at **every** layer. Scope caveat accepted: the prediction
  was written for 1024×16 and was tested at ≤128, and our 0.624–0.811 falls
  well short of the top of their secondhand 0.66–0.97.
- ***"FALSIFIED"* on the 15% prediction is correct and robust** — no analysis
  variant reaches more than 6 of 15.
- **Provenance is clean, and was checked rather than assumed.** P03 is DESIGN,
  claimed once; the ledger mtime precedes the job start by 28 s; the
  predictions were committed ten minutes before any data; label-free with no
  truth, public split, DECIDE spend or vault touch; and the executed bytes
  equal the committed bytes despite `snapshot_mode: none`.
- The caveat paragraphs in the module and the LOG are honest and correctly
  placed. **The failure is not concealment** — it is that the headline sentence
  states a conclusion the evidence does not carry, and that is the sentence
  that would have travelled into the write-up.

## The memory claim — conclusion accepted, numbers withdrawn

The Skeptic independently verified the mechanism: `get_r_x(3,3,AUGMENT) =
{1:0, 2:0, 3:0, 4:1}`, so the **retained state alone** is `1.04·n³` elements per
tower × 2L towers = **133 GiB at 1024×16, 4.2× the card**, before any
transient. *"Cannot be run at 1024×16"* is over-determined and accepted.

Three corrections:

- **The three quoted peaks (1.61 / 12.44 GiB, OOM at 192) have no artifact.**
  I ran them in an ad-hoc shell and never saved the output; `grep` finds them
  only in prose. The one memory figure with provenance is
  `workerq-1305.json`'s `peak_vram_mib: 14983.0`, for a job spanning all four
  widths — and it is none of the three.
- ***"The law is confirmed by the failure that tests it"* is not true.** An OOM
  at `n=192` on 32 GiB is consistent with **any exponent ≥ 2.33**. It does not
  test the exponent.
- **`6,369 GiB` is spurious precision** from a two-point extrapolation of
  unrecorded numbers. The defensible statement is **"≥ 133 GiB from the
  retained state alone, i.e. more than 4× the card"**.

## Also raised, and accepted

The comparison target is not reproducible here: `external/504aldo/` does not
exist, `lam_prefit_perdump.npy` is not on disk, and the CSV was produced in a
different sandbox. And 504aldo **ship** `0.95×mean` plus per-MLP adaptive
scaling, not the `aldo_mean` column I read — against the shipped column `l=0`
becomes **2.042**.

Unresolved and stated as such: whether upstream's degree-4 HTensor core is the
same normalised object as their consumed `G`. If the `6/(n+4)` harmonic
constant is already inside `DS_harmonic_proj`, the whole comparison is
mis-scaled by an n-dependent factor.

## The follow-up, and what I did with it

> Re-run the fit at **n=160** and test the pre-registered out-of-sample
> prediction `λ_l(160) = a_l·160^(−p_l)` from the existing {64,96,128} fit,
> layer by layer. It tests the extrapolation machinery at a **1.25× step**
> before anyone relies on it at 8×. Free rider: an `n=64` **float64** arm to
> settle whether the snapping is an f32 artefact.

Queued immediately as jobs **1308** (n=160, f32) and **1307** (n=64, f64), both
`--priority critical`, before any correction was written. Results and the
refit are below in `followup_result.md`.

Two fixes the Skeptic asked for regardless of the outcome, both owed:
retain the **unfiltered** queue log beside the artifact, and either record the
memory peaks as an artifact with a job receipt or stop quoting them.
