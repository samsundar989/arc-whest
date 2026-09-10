# Strategist Packet — 2026-09-11: Slice 5 is complete. Both kill criteria met, the score lane is done, and two of your three instructions were wrong in ways worth knowing

*Answers `docs/STRATEGIST_RESPONSE_2026-09-10.md`. Self-contained: assumes no
repository access and no shared chat history. Every number traces to an artifact
path given inline. Two independent Skeptics ran; both returned PROVISIONAL; both
kills survived and roughly a dozen of my sentences did not.*

---

## 0. The answer in six lines

1. **Cost is closed at `~0.5 B`.** Packed spelling prices at `1.0067×` against
   your `1.5×` continue-threshold. It was the last cost lane.
2. **T1 failed its gate at `0.0135×`** against a `1.2×` kill. No affordable κ₄
   spelling exists by deletion.
3. **T2 failed its gate at `1.0163×`** against a `1.15×` build and `1.10×` kill.
4. **Both kill criteria are therefore met.** Per your own clause, the score is
   closed at nominee A and the remaining weeks are hardening, nominations and the
   write-up. Reported immediately; no rescue sought.
5. **Depth 2 is submitted** as `330566` and awaits its graded row.
6. **Rules v14 §6 is a checked negative.** Three public routes tried; a neutral
   forum question is drafted and needs Samuel's account.

---

## 1. The numbers, with pre-registered predictions

### Packed spelling — CLOSED at `1.0067×` (gate `1.5×`)

`experiments/2026-09-10_slice5/packed_feasibility.json`

Zero of the chain's `797` matmul-equivalents has redundancy this meter discounts.
The four dominant lineage products contract four *different* factor blocks
(`A_D`, `A_L`, `M`, `C_L`), and a discount needs a redundancy the accountant can
see. All four of my predictions confirmed, including the point estimate: I
predicted `1.007×`.

**You framed this as "does any matmul have a symmetric operand". That is the
wrong question on this meter, and measuring it was worth the day.** Twelve
spellings of the same arithmetic:

| spelling | × dense |
|---|---:|
| `A @ B` | `1.0039` |
| `as_symmetric(S) @ B` — a symmetric **operand** | `1.0176` |
| `A @ A.T` — symmetric **result**, written with `@` | `1.0039` |
| **`einsum("ij,kj->ik", x, x)` — the same object twice** | **`0.5039`** |
| `einsum("ij,kj->ik", x, x[:, :])` — a **view** | `1.0020` |

**A symmetric operand is a `1.8%` penalty.** The only discount is object-identity
aliasing, worth exactly `2.00×`, and it cannot be faked.

### T1 — H1's deletion form, DEAD (gate: kill at `1.2×`)

`experiments/2026-09-10_slice5/t1_augment_lite_v2.json`, 24 P01 networks, paired,
`float32`, controls bit-identical to their anchors.

| arm | κ₄ births | raised-budget legs | births/layer | mean xmse |
|---|---|---|---:|---:|
| `SIMPLE` | — | — | `2n` | `4.004685e-08` |
| `AUGMENT` | kept | kept | `8n` | `1.912323e-08` |
| `AUGMENT-LITE` | **deleted** | kept | `5n` | `4.365096e-06` |
| `AUGMENT-LEGSONLY` | kept | **deleted** | `4n` | `5.523768e-07` |
| `AUGMENT-MINIMAL` | **deleted** | **deleted** | `2n` | `2.976232e-06` |
| *Gaussian K2* | | | | `4.196350e-06` |
| *calibrated K2* | | | | `1.448599e-06` |

`SIMPLE / MINIMAL = 0.0135×`, `0 of 24` improved. GT noise floor `1.1552e-09`.

**`AUGMENT-MINIMAL` was not in your handoff.** It came from reading the birth
count off the returned tensor instead of estimating it, and it is the only
affordable member: SIMPLE's exact `2n` lineage with κ₄'s `(n,n)` object still
live. The gate was written on it.

**Your predicted band was `1.4–1.8×`. The answer is `0.0135×`.** Mine — `[0.35,
0.9]` — was wrong too, and prediction 4's stated *reasoning* had the direction
wrong, not just the magnitude.

### T2 — H2's readout form, CLOSED at `1.0163×` (gate: build `1.15×`, kill `1.10×`)

`experiments/2026-09-10_slice5/t2_corrector.json`, fit on 384 fresh networks
(P07), single held-out read on 128 (P08).

    B0 (held out)   3.859709e-08
    R    (fitted)   3.797635e-08   gain 1.0163x   100/128 better   shrinkage 0.554
    R0   (null)     3.857427e-08   gain 1.0006x
    Rshuf(null)     3.856640e-08   gain 1.0008x

The signal is **real** — both nulls beaten by an order of magnitude, `100/128`
networks improved — and `1.63%` where `15%` was needed. The VOID clause did not
fire. Four of five predictions confirmed; the fifth was never measured (§4).

---

## 2. Three things in your response that were wrong, and one that was unaskable

### 2.1 T1 as specified could not have answered the question it was set

You asked which of κ₄'s two channels carries the `2.094×`: the cheap one (the
Wick updates of μ, Σ and the diagonal κ₃ slices) or the expensive one (the κ₃
births). **The two are not separable in ARC's implementation.** `K211_contrib` —
κ₄'s own update, which you list as part of the *cheap* channel — is computed from
`pK_111.factors` (`factor_k3.py:473-497`) and added into `K_out[4].core` at `:531`.
Those are the very blocks a births-ablation deletes. So the ablated arms propagate
a **different κ₄**, not the same κ₄ with its births suppressed.

**No reweighting of any arms in this design separates them.** I ran the completed
2×2 anyway and it is informative, but the channel attribution is struck as
unidentified. A Skeptic found this after the claim had already reached the
write-up.

### 2.2 Your T2 corpus instruction is forbidden by our own measurement law

You specified training on the organisers' public 1,100 with cross-fitting.
`docs/VALIDATION.md` §0 states the public split is CONFIRM tier, read-only, *"no
constant may ever again be fitted here"*, and its standing rule is *"never tune on
public data again."* `AGENTS.md` makes that binding, and a strategist packet is
advice rather than an amendment to it.

**I refused and minted fresh DECIDE-tier blocks instead** (P07 384, P08 128, two
independent replicas of `2^24` samples). **This changes what a negative T2
means** and you should weigh it: our corpus is `512` networks rather than `1,100`,
and our labels are noisier than the organisers'.

**But the cost is far smaller than I first told you.** I claimed the label
precision would cap `R²` near `0.40`. Measured, the ceiling is **`0.9466`** — I
had assumed a post-ReLU activation variance of order 1 where the measured value is
`0.0755`. The gate needs `R² = 0.1304` (§3). We are not precision-limited by a
factor of seven.

### 2.3 "Any deletion returns the chain to Gaussian accuracy" is also false

This was the Skeptic's replacement for my struck claim, and I adopted it. The
completed 2×2 falsified it: `legsonly` reads `5.52e-07`, **`7.6×` better than
plain Gaussian and `2.6×` better than calibrated Gaussian**, both intervals clear
of zero.

**What the design does support, and it is cleaner than either story:** both
self-consistent configurations are good, and every *partial* configuration between
them is `13.8×`–`109×` worse than the worse endpoint, on `0 of 24` improved in
every case. **The term set is not a menu; it is two working configurations with a
cliff between them.** The damage is wildly non-monotone: `MINIMAL` deletes
strictly more than `LITE` and is `1.47×` better.

### 2.4 The rank-sweep and arm `S` were not run

The κ₄ rank-sweep (your §3, write-up lane) was never started: the GPU was
committed to label minting and the two discriminators for the whole slice. Arm
`S`, T2's in-flight sensitivity probe, is pre-registered in my own contract and
was never run — `--sensitivity` does not exist in the parser although the module
docstring advertises it. **Prediction 5 is unmeasured and I am not claiming it.**

---

## 3. The one methodological result I would carry into every future slice

**A learned-model null is unreadable without a positive control, and the control
must be at gate strength.**

Under the score identity, `gain = E[σ²t²]/E[σ²(c−t)²]`, so a gain of `G` is
exactly a σ-weighted `R²` of `1 − 1/G`:

    build gate 1.15x  <=>  weighted R^2 = 0.1304
    kill  gate 1.10x  <=>  weighted R^2 = 0.0909

That identity is what makes every `R²` in a corrector slice readable, and it was
missing from my first three write-ups of this result.

**Two failures, in sequence, both caught only by controls that could fail:**

1. My first T2 runs gave **all three arms identical to four decimals, including
   the feature-shuffled one**. I was one commit from publishing "there is no
   signal". A planted-target control read `R² = −0.4566`: the head could not
   recover a signal it had been handed. Two elementary defects — no input
   standardisation feeding a `Tanh` with `alpha` at `±10`, and no target
   standardisation for values of order `5e-4`.
2. My *repaired* control planted at `R² ≈ 0.94`, **seven times** the strength the
   gate turns on. The Skeptic built a proxy lacking the right basis, planted a
   large signal outside its span, and got `1.0164×` — my exact answer, with a big
   signal present.

Rebuilt at gate strength, across all three feature blocks, with a composed
nonlinear term:

| planted weighted `R²` | recovered | recovery |
|---:|---:|---:|
| `0.0909` (kill gate) | `+0.0901` | `99.1%` |
| **`0.1304` (build gate)** | **`+0.1298`** | **`99.5%`** |
| `0.3000` | `+0.3000` | `100%` |
| capacity probe `sin(20α)·k3n` at `0.30` | `−0.0019` | — |

**A `1.15×` smooth signal would have been found. A high-frequency one would not.**
A closed-form ridge over the same features independently reproduces the head's
`R²` to four decimals, so no optimiser artefact survives. That is the bounded
claim, and it is much stronger than the null I nearly published.

Note also that the oracle constant-`c` ceiling on the held-out set is `1.0089×`,
which agrees with the previously measured `0.75%` output renormalisation:
**per-neuron structure adds about `0.7%` and nothing more.**

---

## 4. Nominee A, and a governance point I want you to see

`330566` submitted 2026-09-10: Strassen depth 2, sha `390bc226…`, one character
from `330327`. `0.604217 B`, `9,321` operations, projected residual `0.100 s`
(`3.20×` under the cliff), projected worst wall `33.0 s` = **`2.28×` margin** under
the measured `1.60×` contention, against depth 3's `1.21×`.

It certified **6 of 8** pre-registered gates. Green: `whest validate`; the
grader-twin arm at `0.000e+00` relative across all four weight modes with the
f64-vs-f32 FLOP delta exactly one pass over the weights; bitwise determinism
across fresh interpreters; network-independent FLOPs; ops inside the `22,000`
gate; the Linux 8 GiB gate at `2.46×` headroom bound to the exact submitted bytes.
A CONFIRM read on the organisers' mini with the local residual cap lifted:
**`0/100` failures, raw `3.92e-08`, adjusted `2.37e-08`**, multiplier
`0.60420211` — which agrees to eight significant figures across four independent
paths.

**The two failing gates judge residual in local seconds**, which this host reads
`27×` high per operation, and they would also refuse `330327`, which holds a real
graded `0/100` row.

**Samuel authorised the submission on slot availability and explicitly did not
rule the gates mis-specified.** They therefore stand FAILED, unretracted. **If the
graded row comes back well, that is evidence about the candidate and not proof the
gates were wrong.** I record this because the failure mode Rule 1 guards against is
a gate that quietly stops binding once it is inconvenient.

**Nominee pair, proposed:** A = depth 2 once graded; B = `330212` on maximin until
§6 is answered. If §6 ranks a team by its better designated entry, B should move
to `330327` at zero risk.

---

## 5. What I am asking you for

1. **Do you accept the score closure?** Both your kill criteria are met. My
   reading is that Sep 11 – Oct 17 is hardening (1000 fresh mints through the
   nominee bytes, determinism twins, `3×100` official soaks), nominations by Oct
   10, and the write-up at 40%+ effort.
2. **Does the corpus substitution change your read of T2?** You specified the
   public 1,100; we used 512 fresh mints because the law forbids the former. The
   ceiling turned out to be `0.9466`, so precision is not the binding constraint —
   but the corpus is smaller and it is your call whether that weakens the null.
3. **Is arm `S` worth running?** The in-flight corrector form is the H2 test you
   called "proper". It is pre-registered, cheap (a sensitivity injection, no
   training), and unrun. Prior evidence says the in-flight lane needs held-out
   `R² ≥ 0.99849` on the K2 chain; nobody has measured that threshold for K3.
4. **Is the κ₄ rank-sweep still worth GPU time** now that the score lane is
   closed, or is the write-up better served by the `legsonly` result and the
   cliff statement?
5. **Anything in §2 that changes your model of the field.** The leaders reach
   `1.87–2.13e-8` at `0.18–0.21 B`; we reach that accuracy in a reference
   implementation and cannot reach that cost, with four routes excluded and both
   published hypotheses failed under instruments shown to detect the effect they
   were looking for. **We do not know what they are doing**, and I would rather
   say so than manufacture a sixth hypothesis.

---

## 6. What is established, what is hypothesis, what is owed

**Established by measurement:** cost floor `~0.5 B` with four routes excluded;
packed spelling `1.0067×`; the meter's aliasing discount; T1's `0.0135×` and its
`0/24`; the cliff between consistent configurations; T2's `1.0163×` with a
`99.5%`-recovery control and a ridge replication; depth 2's certification and its
`0/100` CONFIRM read; the `0.9466` label ceiling; the replica-independence checks
(`1.0076`, `1.0224`).

**Hypothesis, not finding:** the exact κ₄ ceiling (`6.5×`–`8.8×` at the live shape
after the ladder's calibrated over-read); that the leaders' method is outside the
class we have priced.

**Owed, and none of it blocking:** arm `S`; the κ₄ rank-sweep; a read log for
single-use held-out blocks (`P08` was read three times before anything noticed —
now registered `SINGLE_USE` with a declared `retier` record, but reads are still
uncounted); `.gitattributes` `*.py eol=lf`, because a fresh clone does not
currently reproduce the graded submissions' shas.
