# Strategist packet — 2026-09-16: v31 graded at 8.752e-9; Slice 7 item 0 closed

**State checkpoint, requested by Samuel after the canary graded.** This is *not*
one of the three hand-back triggers in the Slice 7 contract (κ4 half closed or
unlocatable, N1 kill-test verdict, or a contradiction of your 09-16 response).
**Nothing below contradicts your 09-16 response.** It reports item 0's outcome,
scores the predictions made before grading, records four instrument facts that
change how later numbers must be read, and asks three narrow questions. Answer
as if you have no repository access and no chat history.

Labels: **[G]** read off an official graded report · **[M]** measured locally ·
**[I]** inference.

---

## 1. Context in five lines

ARC White-Box Estimation Challenge 2026, Phase 2: predict every neuron's mean
post-ReLU activation of a random He-init bias-free ReLU MLP, width 1024 × depth
16, from the weights alone. `score = mean(final_layer_MSE × max(0.1, FLOPs/2⁴¹))`,
lower is better. Limits: 0.4 s residual (uninstrumented) time per network or it
fails, 120 s wall, 8 GiB. Prize decided by a private re-run of up to two
designated submissions. Our method: closed-form cumulant propagation to third
order, carried as a factored "tower" of thin legs, plus a κ4 regeneration
channel.

## 2. What item 0 produced

**v31** = F21 + P1 (κ4 regeneration with our label-free λ) + P2 (the separable
(2,1) leg identity with a rank-32 permutation-equivariant sketch). Submitted as a
**non-designated canary, #331317**, on Samuel's explicit approval.

| | adjusted | raw | C/B | worst wall | worst residual | failed |
|---|---|---|---|---|---|---|
| F21 #330950 (designated fallback) [G] | `1.596e-8` | `3.780e-8` | `0.4222` | `93.16 s` | `0.1562 s` | 0/100 |
| **v31 #331317** [G] | **`8.752e-9`** | **`2.907e-8`** | **`0.3010`** | **`68.13 s`** | **`0.1402 s`** | **0/100** |

**Graded against graded, paired per network** [G] — both reports expose scores
for the same 50 public-board networks, so this uses no local model at all:

    gain 1.8233x   better on 50/50   paired difference t95 [+6.88e-9, +7.53e-9]
    raw 3.7795e-8 -> 2.9074e-8  = 1.300x      cost 0.4222 -> 0.3010 = 1.403x

**Board, 16:11 UTC** [G]: 165 entries. v31 sits at ~**rank 58** (F21 was 69).
Leader `2.4e-9` (moved from 2.5e-9); top 10 all `≤3.8e-9` at C/B 0.13–0.20; the
published-competitor family (team A and ~25 accounts running its released code)
`4.2–5.3e-9`.

Also closed: a fresh 24-network design corpus, **P11**, minted with two
independent 2²⁴-sample truth replicas (independence ratio `0.9998`, truth noise
`4.76e-9` ≈ 12–16% of the raw MSEs it will measure). Every accuracy read in the
κ4-half slice will use P11, not the corpus that has already carried five reads.
The rules question to the organizers was **withdrawn by Samuel**.

## 3. Predictions written before grading, scored

| # | builder's prediction | graded | verdict |
|---|---|---|---|
| 1 | adjusted in `[8.45e-9, 8.97e-9]` (the local projection band) | `8.752e-9` | **HIT** — projection `8.713e-9`, error **+0.45%** |
| 2 | 0/100 failed | 0/100 | **HIT** |
| 3 | worst residual `< 0.20 s` | `0.140 s` | **HIT** |
| 4 | worst wall `50–65 s` | `68.1 s` | **MISS** (reasoning error, §4.1) |
| 5 | FLOPs = local + `0` or `2²⁴` | local + `2²⁵` | **MISS** in detail, 0.005%, §4.3 |

Your 09-16 response predicted ≈9.3e-9 and ~65–75 s wall. Both consistent.

## 4. Four instrument facts that change how later numbers must be read

**4.1 — the wall projection is load-dependent, and on a quiet host it is
anti-conservative.** [G vs M] The standing projection is *local worst wall ×
1.60*. Measured against three local 100-network runs:

| local run | graded/local **median** | graded/local **worst** | ×1.60 projection vs graded worst |
|---|---:|---:|---|
| F21, host busy | 1.459 | 1.011 | 147.5 vs 93.2 — conservative |
| v31, host busy | 1.488 | 1.310 | 83.2 vs 68.1 — conservative |
| **v31, host idle** | **1.824** | **1.724** | **63.2 vs 68.1 — under by 7%** |

The server is tight (worst/median 1.03–1.04); all the variance is the local
host. We had been treating ×1.60 as a margin; it was conservative only because
the machine was busy. Interim rule: **graded wall ≈ 1.9 × idle-host local
median**. This matters for P4 (more, smaller operations) and N1.

**4.2 — the residual model is ~20% conservative.** [G] Three graded rows now
give 10.73, 11.10 and 10.76 µs/operation; the frozen gate model uses 13.4.
v31's real residual headroom is **2.85×**, not the 2.29× we quoted. Left frozen
on purpose, since a gate should err safe.

**4.3 — FLOP accounting.** [G vs M] The server bills exactly **2²⁵** more than
the local `whest run` subprocess meter for both submissions, while our
grader-twin replay matches the server exactly. A difference between two local
paths, not a new server behaviour; 1.5×10⁻⁵ of the multiplier.

**4.4 — the local→public projection chain works.** [G vs M] Paired ratio over
100 public mini networks × F21's graded score projected `8.713e-9`; the server
said `8.752e-9`. This is the instrument every "did that help?" in the next two
weeks depends on, and this is its first test against a real row.

## 5. The one open anomaly

In one local 100-network run, under heavy host contention (two of our 13–14 GiB
runs plus two other projects' jobs), **2 of 100 networks raised during
`predict`**. The report that would have named them was destroyed by a wrapper
bug of ours. It has **not recurred**: two further local runs (one on an idle
host) were clean and **bit-identical to each other on every network, every
layer, every FLOP**, and the server graded 100/100 with smoke 8/8. A
deterministic estimator cannot raise on one run and not another on the same
weights, so the cause was outside the arithmetic [I] — most plausibly a worker
process lost under memory pressure. **Mechanism unknown.**

## 6. What happens next (unchanged from your 09-16 plan)

Item 1, not started: **D1** (substitute team A's λ table into a never-shipped
diagnostic arm, read on P11) → **D2** (layer-by-layer trace of `dG`, `g_prev`,
`var_pre/post`, λ and the (4)/(2,2) consumer inputs against team A's code on one
1024×16 network) → **D3** (zero the κ4 inputs to the (3,)/(2,1) births on 8 P11
networks). Gate: v31-fixed raw on P11 ≤ 0.67 × v31 raw, paired t-CI excluding
zero, cost within 0.5%. Your pre-registered probabilities (40% / 60% / 55%) are
recorded; Samuel's prediction is taken before D1 runs.

## 7. Questions — narrow, and only where your judgement changes what we do

**Q1 — Does the public-scale raw gain change the κ4-half target?** On the 24
design networks P1+P2 gave raw **1.226×** over F21 [M]; on the public networks
it gave **1.300×** graded [G] and 1.306× on the local mini [M]. Your gate is
written relative to v31 on P11 (≥1.5×) and your target in "P01-scale"
(2.2–2.4e-8). On public scale v31 is already at 2.907e-8, so a 1.5× fix would
land near **1.94e-8 raw → ~5.8e-9 adjusted** [I]. Should the target be restated
on public scale, and does a design-vs-public transfer ratio of 1.06 suggest P01
is harder than the public distribution in a way that could mislead D1–D3?

**Q2 — Designation risk of the §5 anomaly.** v31 now meets the handoff's
condition to become the fallback (one clean graded row). Your 09-14 plan says
three clean graded runs before designating. Is an unexplained, non-recurring,
contention-correlated 2-of-100 raise worth a **deliberate stress soak** before
designation — e.g. three concurrent 100-network runs under induced memory
pressure, label-free, no submission slots, ~3 hours — or is 2 clean local runs +
1 clean graded run already the evidence, and the soak a distraction from item 1?

**Q3 — Wall budget for P4 and N1, given §4.1.** v31 grades at 68 s worst with
the server nearly noiseless. P4 (Strassen depth 5, pooled `out=` buffers) and
N1 change the *operation mix*, which is exactly where our per-operation models
have evidence gaps. Is there a reason to expect server wall per FLOP to rise
sharply with more, smaller operations (per-call overhead is ~5.1 s over 13,037
calls, i.e. ~0.39 ms/call) such that P4 should be canaried alone before N1 is
stacked on it, or is stacking safe at ~55–65% of the 120 s cap?

**What we are not asking:** the allocation (settled in your 09-16 response), the
κ4 mechanism (D1–D3 will measure it), or anything that needs the public split
for fitting (refused by policy).
