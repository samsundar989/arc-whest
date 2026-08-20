# CURRENT_STATE.md — read this first

*Snapshot, not narrative. Last updated **2026-08-19** (Stage 0 — the instrument
repair; five slices, two independent Skeptic reviews). Update this file when a lane
opens or closes, a number is graded, or the phase changes — not on every
experiment. The narrative record is `experiments/LOG.md`.*

---

## The clock

*Last polled 2026-08-19 15:13 UTC.*

| event | when | status |
|---|---|---|
| Round 1428 "Phase 1" | closed **2026-08-10 23:59 UTC** | **CLOSED.** Our last submission is 327743 (v2.4.1) |
| Phase 1 submission selection | closed **2026-08-11 23:59 UTC** | **PASSED.** We nominated nothing, so the default applied: our **two** highest-ranked public entries |
| Phase 1 write-up | closed **2026-08-17 23:59 UTC** | **PASSED, MISSED.** No extension was issued. Write-ups posted after it (18184, 18185) drew no organiser objection, but that is not permission |
| Round 1429 "Phase 2" | opens **2026-08-20 23:59 UTC** | **UPCOMING** (re-polled 2026-08-19: leaderboard 1480 exists, **0 entries**; HF tags still `v1-phase1`/`v1-warmup`). Slipped twice: Aug 8 → Aug 18 → Aug 19 → Aug 20. Organiser 18186 (2026-08-18): *"still ironing out some details … the deadline for Phase 2 submissions will also be shifted"* |
| Phase 2 close | nominally 2026-09-19 23:59 UTC | **WILL MOVE** — organiser says so explicitly |
| Registration / team freeze | **2026-09-05 23:59 UTC** | 17 days out |
| Results | 2026-10-01 | private re-run of **UP TO TWO** designated submissions on fresh MLPs (forum 18143; the Rules tab still says one and the contradiction is unresolved) |

**Phase-2 opening is event-triggered, never date-triggered, and the date has now
moved three times.** Poll with `uv run python local_sim/phase2_checklist.py`.
**Its configuration is unpublished**: width, depth, biases, init, `n_mlps`, budget
`B`, λ and the multiplier floor are all unverified for round 1429, and no
`v1-phase2` HF dataset tag exists. Organiser says details ship by **email**, not
forum-first. The challenge page's timeline and its `max(0.5, ·)` worked example
are **stale Phase-1/warm-up copy — do not read constants off it**; the only
trustworthy live numbers are flopscope 0.10.0 / whestbench 0.14.0 / λ = 1e11 /
floor 0.1 from organiser thread 18125.

**First action when 1429 goes active:** re-fit `_PRUNE_TAU` and `_BASIS_RANK` if
the shape moved, *then* gate. Both are frozen at 256×32 and neither transfers.

## Where we stand

| instrument | tier | value | date |
|---|---|---|---|
| **graded, submission 327743 (v2.4.1 — THE SHIPPED FILE)** | graded | **2.6314e-7** adjusted · 1.3323e-6 MSE · mult 0.19896 · residual 10.022 ms/MLP · **0 of 100 failed** | 2026-08-10 |
| graded, submission 324420 (v2.4) | graded | 2.7194e-7 adjusted · 1.3748e-6 MSE · mult 0.1990 · residual 9.845 ms/MLP | 2026-08-05 |
| graded, 324094 (v1.1) | graded | 3.893e-7 | 2026-08-05 |
| vault, 500 sealed MLPs, v2.4.1 | VAULT | median **2.4651e-7** · mean 3.2669e-7 · p99 2.0535e-6 · max 3.0469e-6 · 0 non-finite | touch #1, 2026-08-06 |
| public mini (100), v2.4.1 | CONFIRM | 3.172e-7 adjusted · 1.4895e-6 MSE · mult 0.216 · 0 failed | 2026-08-06 |
| invariant suite on the shipped file | — | **30/30** | verified 2026-08-07 |
| board position | — | raw #90/200; **#82 of 192 plausible-honest** on 5 entries | 2026-08-06 10:03Z |

**Shipped file, corrected 2026-08-19**: `estimator.py` == **`estimator_v2_6_1.py`**
(byte-identical modulo line endings — `scripts/preflight.ps1` step 4 asserts this).
The **graded** file is `estimator_v2_4_1.py` (327743, 2026-08-10, Phase 1): 0
failures, safe to designate. **These are not the same file** — v2.6.1 is built,
green and deliberately un-gated, and the line above said "v2.4.1" until an
independent audit caught it. Vault touch counter: **1**.

**v2.4.1 vs v2.4 is a DRAW, not a 3.2% improvement.** The headline adjusted score
fell 2.7194e-7 → 2.6314e-7, but both were scored on the *same* 100 networks
(`mlp_name` lists identical, `sampling_mse` identical to 17 digits), so the paired
statistic applies: geometric mean of per-MLP ratios **0.9665×, 95% CI [0.7969,
1.1721]**, better on **22 of 50** (exact p = 0.839). The aggregate delta is a small
residue of near-cancelling sums (+1.91e-5 of gains against −1.70e-5 of regressions);
three networks each individually flip its sign. Counted FLOPs agree to 0.02%, and
layers 0–30 are exactly float-equal on **all 50** networks — only the scored final
layer differs. **Quote v2.4.1 as *equivalent to* v2.4, never as better.**

*Precisely: "no **resolvable** effect." At n = 50 with sd(log ratio) = 0.696 this
design can only exclude |effect| > ~20%. The better-powered answer already existed
and agrees — 0.9959× over 32 draws/MLP (`docs/GOTCHAS.md`). The grader run confirmed
it; it did not discover it.* Full analysis and the independent audit that corrected
three claims in this section: `experiments/LOG.md` 2026-08-10.

**Field anchors — corrected 2026-08-19.** Best *published and explained* honest
score is **9.10e-8** (Team Puffi, analytic+QMC hybrid); the best published raw MSE
is 1.546e-7. GT noise floor of the metric ~2e-10. **We are ~2.6× behind the best
published honest method, and behind roughly twelve of them.**

⚠️ **"Our declared honest sampling frontier ~2.7e-7" is WITHDRAWN** (it described
*our own method's* frontier, not the field's), and so is the framing that the
cost half of `V·c` is "empty" while variance is "crowded" — the 2026-08-19 carrier
audit measured our variance carrier at ~85–95% of the achievable ceiling for
input-space methods, and found the remaining headroom on the **cost** side.
Full table: `docs/BENCHMARK.md`.

---

## Shipped file — v2.6.1 BUILT, NOT PROMOTED (2026-08-19)

`estimator.py` is **v2.6.1** (Strassen-Winograd carrier + the P0.3 odd-width
repair), archive `estimator_v2_6_1.py`, preflight **GREEN 6/6**, invariants
**30/30**, shape parity **23/23**, vault touches still **1**. **It has not been
gated, not been submitted, and no decision block has been spent.**

**v2.6 had a network-destroying bug on every odd width and it is fixed.** The
Strassen trim ran *before* the eligibility test, and the trim itself
(`keep = a > alpha[alive].min()`) sheds every neuron tied at the minimum — and at
layer 0 **every alpha is exactly 0.0**, because `x ~ N(0, I)` makes the
pre-activation mean zero. So on any odd width v2.6 deleted the entire first
layer's sampled path: 7 of 17 shapes wrong, up to **0.58 max relative error** and
**0.33 prediction MSE** against v2.4.1. v2.6.1 decides eligibility from `n` and
`width` first, rebuilds untrimmed if trimming cannot make every dimension even,
and refuses any trim that would shed more than one neuron.

**Nothing recorded is invalidated**: the competition shape is 256x32, layer 0's
alive count there is 256 (even), the trim never fired, and v2.6.1 is bit-identical
to v2.6 at even widths **including its counted FLOP count to the unit**. The
independent Skeptic confirmed `use_str` was True on 100/100 public-mini networks.
Full record: `experiments/LOG.md` 2026-08-19 (night), P0.3.

The graded number of record is still **v2.4.1 = 2.6314e-7** (submission 327743).

**Why it is held.** The deciding quantity is the grader's residual price per
operation, which this document classifies **E1 — directional only**, and the
same byte-identical paired comparison reads:

| | cost ratio | net V.c |
|---|---|---|
| **fitted line on 5 graded submissions (2026-08-19 — QUOTE THIS)** | **1.1004x** | **1.0940x** |
| grader per-op price 7.0 us/op (ratio through origin) | 1.0990x | 1.0926x |
| transfer the local residual ratio (the old conservative choice) | 1.0740x | 1.0677x |
| **this machine, as measured** — never decides | 0.9577x | 0.9522x |

**The first row supersedes the rest.** Projected graded residual for v2.6.1 is
**16.251 ms** against v2.4.1's observed 10.022 ms; counted FLOPs fall 10.5%,
residual rises 62%, and because residual is only 2–3% of `C` on the grader the
counted saving wins. Known to well under 1% on the cost side, against the
17-point bracket below whose **sign flipped**. Remaining caveat: v2.6.1 sits at
2,370 ops and the fit is calibrated on 329–1,352 — a **1.75× extrapolation**,
which is now the load-bearing assumption and is much weaker than the old one.

The sign flips between the local harness and the grader, because local residual
runs **4.37x** the grader's (42.99 ms/MLP vs 9.845) while the op count transfers
exactly (1,417 local vs 1,416 graded). Break-even is 55.5 us/op; the grader is
at 7.0, this machine at 30-51. **Only a graded run can decide it.** MSE is
settled and costs 0.51% (paired 0.9942x [0.9870, 1.0013]), all of it the
even-width trim — Strassen itself contributes nothing (LOG 2026-08-19).

## Stage 0 — the instrument, repaired (2026-08-19 night)

Five slices, all recorded in `experiments/LOG.md` with contracts written before
the data existed. **Nothing was promoted, no decision block was spent, no vault
touch was taken, nothing was submitted.**

| # | what | state |
|---|---|---|
| **P0.1** | `local_sim/competition_score.py` — the grader's identity in one place, bitwise-equal to whestbench 0.14.0 on the scalar primitives, on the aggregate, and on the real scoring loop including both failure paths. **71/71** | done, Skeptic PROVISIONAL |
| **P0.4** | `local_sim/seed_ledger.py` + `experiments/seed_ledger.jsonl` — consumption recorded, enforced, and statically scanned. **39/39** negative tests. Preflight step **6** | done |
| **coverage** | `local_sim/coverage_lab.py` — which promotion interval actually covers 95%, on 192 fresh networks (D18) | done |
| **P0.3** | `estimator.py` -> **v2.6.1**, the odd-width repair. **23/23** shape parity | done, Skeptic **ACCEPT** |
| **P0.2** | `local_sim/vault.py` — evaluates a named candidate FILE under real flopscope, `--dry-run` deleted, touch bound to a sha256, maximum enforced, atomic seal update | done |

**Three things every future number depends on:**

1. **The competition objective is the ARITHMETIC MEAN of per-MLP scores over the
   FIFTY public MLPs**, and it is now reproducible offline **bitwise** from the
   durable graded reports — `2.719427529595e-07` (324420) and
   `2.631432177242e-07` (327743), exactly. A failure is priced at multiplier
   **1.0**, not the 0.1 floor: ~5x, on top of the zeroed prediction.

2. **The arithmetic mean and the geometric mean of per-MLP ratios select
   OPPOSITE winners on our own graded data.** v2.4 -> v2.4.1 on the same 50 rows:
   arithmetic **1.0334x** (v2.4.1 better), geometric **0.9666x** (v2.4 better),
   22/50 wins, bootstrap CI spanning zero. `docs/VALIDATION.md` 2b still makes the
   geometric mean the binding gate and **has not been changed** — that is a gate
   change, therefore STRATEGIC, and it needs its own contract.

3. **Use the studentised t interval or the basic bootstrap. Do NOT gate on the
   percentile bootstrap or on BCa at n ~ 50.** Measured false-promotion rate
   against a true zero (nominal 5%): t **4.1-5.4%**, basic **4.0-5.2%**,
   percentile 6.8-8.2%, **BCa 9.1-11.5% — roughly double**. The reason is that
   common random numbers make the *paired difference* nearly symmetric (skew
   +0.21, -0.55, -0.46) even though each arm's scores are strongly right-skewed,
   so BCa's skew correction is pure added variance. BCa only starts winning past
   skew ~6, which no real arm reaches.

**The v2.6 gate, re-derived under the right estimand** (read-only, from the two
durable CONFIRM artifacts; `local_sim/reconstruct_v26_gate.py`). The scorer
reproduces both artifacts' per-MLP and aggregate scores at **0.000e+00** relative
deviation:

| residual model | arithmetic gain | 95% CI (relative) | geometric gain |
|---|---|---|---|
| local, as measured — never decides | 0.9529x | [-5.77%, -4.11%] | 0.9521x |
| per-op at the graded 6.95 us/op | **1.0961x** | [+8.01%, +9.52%] | 1.0927x |
| transfer (conservative) | **1.0695x** | [+5.75%, +7.25%] | 1.0666x |
| counted FLOPs only | 1.1147x | [+9.54%, +11.03%] | 1.1109x |

**The estimand correction does not change this decision** — arithmetic and
geometric agree in sign in all four models, and the arithmetic reading is slightly
*more* favourable to v2.6. The sign still flips on the grader's residual price,
which is **E1**, and only a graded run settles it.

**Seed blocks after the ledger repair:** D16 **DESIGN, part-spent** (270 of 600
drawn by five probes; 330 free in four gaps). D17 **LOCKED SINGLE-USE**, verified
never drawn. D18 DESIGN, **192 drawn** by the coverage study, 408 free. D19
**SINGLE-USE, unspent**, held for the v2.6 promotion gate so a Phase-2 shape
change cannot burn the only clean decision block.

---

## Closed lanes — do not reopen without new *external* evidence

Each was killed by a measurement, not by a preference. The reopening condition, if
any, is stated. Reopening one is STRATEGIC work and needs a Slice Contract.

| lane | verdict | gate vs result | evidence |
|---|---|---|---|
| Analytic family (B1 fusion / B2 tail / B3 learned residual) | **DEAD** | fusion net ~0 · tail −67…−214% · R²=0.012 | GAME_PLAN rev 3 |
| Track T — shrinkage-moment tail | **CLOSED** | 1.10× gate → 0.9998× | LOG 2026-08-05 |
| Track P — rank-k deviation propagation | **CLOSED** | 1.25× gate → 1.093× even with a free oracle | LOG 2026-08-05. Break-even rank = width/2 |
| Sprint 3 — structure-aware FLOP allocation | **FAIL** | → 1.0246× | LOG 2026-08-05 |
| Sprint 4/5 — packed activation sparsity | **KILLED** | pre-registered 1.15× → **1.1216×** [1.1188, 1.1244] on D14 | LOG 2026-08-06. D12 and **D14 are SPENT — never reuse** |
| N as an operating-point lever | **CLOSED** | `MSE×N` flat over {16k, 32k, 65k} at 8 shifts | the 14% "optimum" was one heavy-tailed realisation |
| Batched depth-2 Strassen | **OFF** | bitwise identical, −53% ops, **worse** residual | shipped behind `_STRASSEN_BATCHED`, tests kept |
| **Control variates (layer-1 exact-mean, LINEAR forms only)** | **DEAD** | **1.010×, 95% CI [0.979, 1.042]** on our Sobol+digital-shift path — CI includes 1.0. On plain MC the same correction gives 1.383× and the fitted optimum 1.584×: QMC already removes that variance | Skeptic probe 2026-08-11, 14 MLPs × 6 shifts, paired geo-mean, CONFIRM tier. **Covers recursive/per-layer forms** — any correction telescoping to a linear map of the layer-1 sample-mean error is in this class. **Does NOT cover** corrections using an *empirical* per-layer gain (`mean(z>0)`), which are nonlinear and unmeasured. The older "1.80× standalone" is a ratio of means over 3 MLPs and should not be quoted |
| Score optimisation generally | ~~CLOSED at ~2.7e-7~~ **REOPENED 2026-08-10** | — | GAME_PLAN rev 6's frontier described *our method*, not the field's. Board decomposition shows honest teams 4.8× better on MSE at our compute. See § *Open questions* #1 |

**The mechanism behind the packed lane's death, which generalises**: flopscope
prices a gather at 4 FLOPs/element, and packing's entire saving is substituting
gathers for dense multiply-adds. A random-access gather costs far more *wall time*
per element than a BLAS multiply-add, and wall time is charged back at λ = 1e11.
The lane arbitrages the price table against physics, and loses.

---

### Added 2026-08-19 — the carrier audit (LOG 2026-08-19)

| lane | verdict | number | note |
|---|---|---|---|
| **Point-set whitening (full 256x256 ZCA)** | **DEAD on our carrier** | **0.9972x, CI [0.986, 1.009]** | 24 MLPs x 24 shifts, paired, GT 2^23. Worth **1.23x on iid** — the device works, our Sobol already collects it. The competitor "2.46x" is a rung against an iid+antithetic baseline; that author's own ablation reads 0.97x / 0.994x in both orders and the census records 1.05x |
| **Diagonal moment standardisation** | **DEAD, and it was always a no-op** | **1.0000x, CI [0.999, 1.001]** | Our shipped Sobol's diagonal second moment is already **107x better than iid**; its off-diagonal is only 5.9x better. The 2026-08-05 kill and the radial-imbalance predictor both probed the already-exact half |
| **ReLU-homogeneity sphere projection** | **DEAD on our carrier** | 1.0010x, CI [0.998, 1.004] | Exact and free, but Sobol already balances the radius. Worth +4% *inside* a design carrier |
| **Tent / baker fold** | **DEAD alone** | 0.9810x, CI [0.928, 1.037] | Only `tent+white` clears, at **1.0931x [1.044, 1.148]** — the one surviving input-space combination |
| **Degree-3 structured design carrier** | **DEAD against the shipped stack** | **1.047x, CI [0.979, 1.120]** | Bare it is 1.182x [1.092, 1.290]; the rank-32 rotation collects it. **Does NOT close a true degree-4/5 design** (strength-4 OA / Kuperberg cubature), whose external ceiling is ~1.46x |
| ~~Strassen-Winograd~~ | **REOPENED** | **1.267x counted (depth 2), MSE-exact; ~1.10-1.16x net after residual** | Measured on the real flopscope 0.10.0. Our 2026-08-06 negative was on the *packed* v2.5 carrier (tiny irregular blocks). Now the largest bankable lever in the project |

**The governing mechanism, measured three ways today: SUBSTITUTION.** Whitening
vs Sobol, design vs rotation, and (previously) CV vs QMC. Two devices that
annihilate the same low-order structure do not compose — whichever is added first
takes the credit. **Read every competitor ladder for what its baseline arm
contains, never for its largest rung.**

⚠️ **Correction, 2026-08-19 (P0.4): two of those three ways are ONE sample.**
`probe_whitening.py` and `probe_design_carrier.py` both hardcode
`SEED0 = 30_806_016_200` with `N_MLPS = 24` — **the same 24 networks**. The
whitening kill (0.9972x) and the degree-3 design-carrier kill (1.047x) are
therefore **correlated evidence, not two independent readings**. Both are DESIGN
tier and no gate was decided on either, so neither verdict changes; the claim of
independence does. The CV-vs-QMC leg is on different data and still stands. The
seed ledger now prints this every time anyone runs `preflight` step 6.

~~**New standing number: 7.0 us/op**~~ **SUPERSEDED 2026-08-19.** That was a
ratio *through the origin* from a single submission (327743: 1,416 ops, 9.845 ms).
Fitted on **all five** graded submissions the grader's residual is a **line**:

    residual_graded  ≈  2.551 ms  +  5.780 µs/op        R² = 0.898, LOO 0.829 ms

**2.551 ms is FIXED overhead charged to every MLP before our first operation** —
25% of v2.4.1's whole residual, 2.55e8 FLOP-equivalents. A ratio through the
origin cannot see it and mis-prices every candidate whose op count differs from
the one it was fitted on. The discriminating evidence: **v1.0 has 12.5× v0's
counted FLOPs and LESS residual** (4.491 ms vs 6.743 ms), which no function of
FLOPs can produce. Price any counted-FLOPs-for-op-count trade with the line, and
note that a candidate far outside 329–1,352 ops is an extrapolation.

**Correction owed to a durable doc.** `docs/writeup-draft.md` and
`docs/RESEARCH_NOTES.md` attribute the 2.35x moment-standardisation regression to
"a first-order bias". Moment matching's bias is **O(1/N)** (Glasserman 4.5.1
p.245) and diagonal forcing measures at exactly 1.000x here and in forum 18151.
The number may be real; the mechanism is not. **Do not ship it.**

## Open questions, ranked

*Re-ranked 2026-08-19. Questions 1 and 2 of the previous ranking are now answered
and have moved to § Closed lanes; what is left is genuinely short.*

1. **THE PHASE-2 CONFIGURATION — blocking, and it lands within hours.** Shape,
   `n_mlps`, `B`, λ and the floor are unpublished. This is the highest-ranked
   open question not because it is deep but because **everything downstream is
   conditioned on it**: `_PRUNE_TAU = −2.5` and `_BASIS_RANK = 32` are fitted at
   256×32, D17 is single-use, and any design-carrier work is width-specific.
   **If `L/n` falls materially — a wider or shallower Phase 2 — the mechanistic
   lane reopens wholesale**, because ARC's own cumulant propagation breaks even
   only when `L^K < n` (L < 16 for K=2, L < 6.3 for K=3). At 256×32 we sit at
   L/n = 1/8 and every variant of it is either over budget or worse than MC.

2. **Does v2.6 actually win?** Built, preflight GREEN, MSE settled at 0.9949×.
   The cost claim reads **1.068–1.093× projected on the grader** and **0.952×
   measured on this machine**, and the sign flips on the residual price — a
   quantity this document classifies **E1, directional only**. See § *Shipped
   file*. **Only a graded run decides it.** A Skeptic has not seen the build; the
   transfer argument (that v2.4.1's measured 4.34× per-op inflation carries to a
   file with 1.7× the operation count) is the load-bearing assumption and is
   exactly what an independent falsifier should attack.

2b. **The design lane is ALIVE, and the missing rung is now identified.**
   Measured 2026-08-19 (`local_sim/test_carriers.py`, 18/18): the shipped Sobol
   carrier's degree-4 moment error is **0.510× iid's** — it collects about half
   of degree 4 and none of degree 6. A Haar frame bank is **exactly a 3-design**
   (machine precision at k=1,2,3) and **0.700× iid at k=4**, which is the
   mechanistic explanation of the audit's degree-3 kill: it brings nothing the
   rotation had not already taken. **The remaining prize is specifically the
   degree-3 → degree-4/5 step**, and nothing above degree 4 is reachable with
   anything we have (every carrier sits at 0.76–1.09× iid at k=6). Two further
   mechanisms fell out: sphere projection is **angularly inert** (it moves the
   radius only, which is why it measured 1.0010×), and antipodal symmetrisation
   **costs ~2× at even degrees** because it uses n/2 distinct points.
   **BUILT, same day.** `local_sim/carriers.py:bch_oa` — the extended dual-BCH
   orthogonal array, `{Tr(ax + bx^3)}` plus the all-ones coset, 2^17 = 131,072
   rows on 256 coordinates. **Strength 5, verified combinatorially** (every random
   4- and 5-subset of columns perfectly balanced, 320/320). Its degree-4 moment
   error is **0.0305× iid — 17× better than the shipped carrier** — and **0.138×
   at degree 6**, where every other carrier we have sits at 0.76–1.09×. No
   Kerdock/Z₄ machinery was needed.
   **MEASURED END TO END, same night: the design is worth 1.4775x and we cannot
   afford it.** 48 fresh D18 networks, GT 2^24, arms matched at 131,072 points and
   at cost (counted FLOPs differ by 0.008%): **adjusted gain 1.4775×, studentised
   95% CI [+3.80e-08, +1.81e-07] excluding zero, relative [+14.4%, +45.9%], better
   on 33/48, sign-test p = 0.013.** Raw-MSE gain 1.4826×. **This is the largest
   measured effect in the project** — every other input-space device died between
   0.98× and 1.09× — and it lands within 1.5% of the ~1.46× ceiling derived
   independently from the harmonic spectrum.
   **The blocker is cost, and it is hard.** `C/B = 1.3509`: **48 of 48 MLPs over
   budget**, which the grader scores as a *failure* (multiplier 1.0 *and* zeroed
   predictions). The array only works whole — halving it gives a constant column
   and centroid norm 1.0 (41× worse end to end), and a *random* half is worse than
   Sobol at k=2 and k=4. **And FWHT is not the fix**: the first layer is only
   **4.7%** of the 32-layer chain (1.718e10 of 3.674e11), so even a free first
   layer leaves `C/B = 1.29`. That earlier speculation is withdrawn.
   **The construction question is SETTLED for linear codes: 131,072 is the
   floor.** Proven by exhaustive enumeration, not argued — the BCH parity-check
   columns `(x, x³)` form a Sidon set of size **255** in F₂¹⁶ (all 32,385 pairwise
   XORs distinct), and **all 65,535 nonzero vectors are 3-subset sums**, so a
   256th column always creates a weight-4 dual codeword. *Zero* candidates.
   **At width 255 the array would be affordable** (65,536 rows, `C/B ≈ 0.68`);
   width 256 rather than 2⁸−1 costs exactly **2×** on this lane.
   **Both known cost levers together are not enough**: depth-2 Strassen (1.267×)
   and an FWHT first layer (1.047×) leave `C/B = 1.0184` — still over, and far too
   close to a cliff that scores as a *failure*.
   **The most promising untried route is a weighted subspace cubature.** The
   estimator already rotates so `h`'s first 32 coordinates are the top-32
   eigendirections of `J Jᵀ`, and a degree-5 Gaussian cubature in **32** dimensions
   needs `O(d²) ≈ 2,049` nodes rather than 131,072. The obstruction is unequal
   weights, turning `h.mean(axis=0)` into a weighted mean — small, and
   flopscope-cheap. Nothing in the repo has tried it.
   If any route lands, `2.6314e-7 / 1.4775 = 1.781e-7` — roughly half the
   remaining gap to the best published honest score (9.10e-8), in one step.

3. **Where does the ~2.6× to the published honest frontier come from?**
   Unresolved and possibly unresolvable from public data. Their sample counts are
   not published, so their **variance advantage cannot be separated from their
   cost advantage**. Their write-ups attribute it to a strength-4 orthogonal-array
   design plus dead/on/kink routing plus FWHT. The one variance lane the carrier
   audit did *not* close is a **true degree-4/5 design** (BCH strength-4 OA, or
   Kuperberg's 2¹⁶+512-point cubature which is exact to degree 5 for the sphere
   *and* for N(0,I) simultaneously). External ceiling from the layer-32 harmonic
   spectrum (deg ≤5 share ≈ 0.36–0.39): **≈1.46×**. Width-specific — do not build
   it before #1 is answered.

4. **Moonshot G1 — blocked on the D10 renormalization fit, and it now has a
   deployable gate for the first time.** G0 (2026-08-07) proved the layer
   convention, built the calibrated-K2 baseline (depth 8 → 1.2067e-05, G1 must
   reach ≤4.02e-06), and read the oracle: full-rank K3 beats the control 12.786×,
   but pre-registered r ∈ {4, 8, 16} is *worse* and the ≥3× gate needs r ≈ 128 =
   width/2 — where Track P found factorisation stops paying. **New, better gate
   (2026-08-19): a closure whose mid-layer mean is 3–6× more accurate than the
   present one licenses the mid-layer control-variate lane** (`r_c = δ_c/SE_c < 1`
   at some depth where ρ² is large), which the oracle prices at up to ~100×. That
   is a *deployable* success criterion, unlike the old ≥3× accuracy bar which was
   never tied to a use. Kill-by-default **2026-09-01**. D08/D09/D10 untouched.

5. **Is there a Phase 2 write-up round?** Unverified and worth one email. The
   Phase 1 window closed 2026-08-17 and we missed it. The "Sep 26" figure in older
   docs is **uncorroborated by any source**. This matters more than it looks: the
   contribution prize is where our comparative advantage now sits (§ *Standing
   risks*), and the answer changes how the month is allocated.

---

## Standing risks

- **Uncommitted work — PARTIALLY resolved, and the remainder is load-bearing.**
  The *experiment* record landed across five commits through `7040f41` (mint
  provenance, vault, invariant suite, Sprints 4–5, Moonshot G0, records), so LOG
  entries can cite commits again. **What did not land is the cross-agent harness
  itself**: `AGENTS.md`, `ai/`, `.agents/`, `.codex/`, `scripts/guard_paths.py`,
  `scripts/verify_portability.py`, `scripts/git-hooks/`, `scripts/human_model_state.py`,
  `scripts/install_git_hooks.py`, `.claude/hooks/guard_hook.py`, plus modifications to
  all six SKILL.md adapters, `CLAUDE.md` and `README.md` — all **untracked** as of
  2026-08-10. `7040f41` committed only the Claude-side skills.
  This is sharper than the original risk: `CLAUDE.md` `@`-imports `AGENTS.md` as the
  operating contract, and `verify_portability.py` reports 174/174 against files that
  exist **only in the working tree**. Losing this tree loses the entire
  agent-independent layer and both agents' enforcement.
- **The cost half of the evaluator is not trustworthy.** See below. **New, 2026-08-10:
  the unreliable half is OUR measurement, not the grader's** — see the transfer-ratio
  note in § *Evaluator trust*.
- **Deadline conflict**, unresolved by the organizers since Aug 5 (thread 18130).
- **The metric has a heavy tail.** Fresh-MLP max/median = 12.9×; the worst 1% of
  MLPs carry 6.9% of the score. At a re-run of size R = 10 the p90 score is 50%
  above the median. Tail risk, not mean improvement, is what the re-run measures.

---

## Evaluator trust — E1/E2, and the split is the point

The score is `MSE × max(0.1, C/B)`, and its two halves are measured to completely
different standards. We are currently at `C/B ≈ 0.20`, well above the 0.1 floor, so
**both halves are live**.

| half | level | why |
|---|---|---|
| **Accuracy (MSE)** | **E2, approaching E3** | mint proven bit-identical to the grader (100/100, uint32); seed-block registry with machine-readable single-use blocks; gates pre-registered in LOG before the block is touched; paired common-random-number statistics with CIs; 32-shift averaging; every MSE quoted against its GT-noise floor; 30/30 invariants including permutation equivariance; touch-counted sealed vault |
| **Cost (`C` = counted FLOPs + λ·residual seconds)** | **E1 → E2 PROPOSED, 2026-08-19** | counted FLOPs are exact and reproducible. Local wall time is not measurable here — but **we never needed it**. The grader measured residual for us **five times** and the telemetry was sitting unused in `experiments/run_reports/`. Fitted on all five: **`residual ≈ 2.551 ms + 5.780 µs/op`**, R² = 0.898, leave-one-out 0.829 ms. Residual is only **1.9–3.3% of `C` on the grader**, so `C` is now determined to well under 1%. **The reclassification is STRATEGIC and is Samuel's to make** — it gates the "no autonomous loop while E1" rule. See `experiments/LOG.md` 2026-08-19 (night) |

**The local residual has no quotable level — established 2026-08-10 (submission
327743), corrected the same day after independent falsification.** Same
byte-identical `estimator.py`, same 100-MLP mini split:

| reading | ms/MLP | vs grader 10.022 |
|---|---|---|
| `--runner local`, 2026-08-06 | 70.85 | 7.07× |
| `--runner subprocess`, 2026-08-06 | **54.47** | **5.44×** |
| `--runner subprocess`, 2026-08-10 | 42.94 | 4.28× |

Runner effect, same day **1.301×**; machine-state effect, runner fixed **1.269×**;
they multiply to the observed 1.65×. So the spread is **half path, half time** — not
"purely the runner", and the overstatement is ~30%, not 65%.

`AGENTS.md` § *Measurement law* states "~7×" as binding and **must stay unedited**
until repeatability is measured. If three same-day repeats span >10%, the correct fix
is not "4.3× instead of 7×" but *"this machine has no quotable absolute residual
level"*, with the line replaced by a range carrying its measurement path.

**What this does NOT contaminate.** The claim that Sprint 4 S2 and the D14 residual
model "scaled by ~7" was **false and is withdrawn** — two different sevens were
conflated. D14's deciding model is *local **ratio** × graded anchor*
(`9.84 ms × 490.4/62.3`); the 7.87 there is a within-run candidate/control ratio,
invariant to any uniform rescaling of this machine, and the local→grader level never
enters. The local-absolute model is explicitly flagged "never decides". **The real
exposure, newly identified: that candidate/control ratio itself reads 5.47× in one
local harness and 7.87× in another — a 44% spread on the quantity that does decide.**

**Grader-side residual: a detected shift, not stability.** Paired 1.0180× [1.0119,
1.0241] excludes 1.0. But n = 1 submission per file, so the n = 100 CI is
pseudoreplication; and `residual_s` is a subtraction remainder whose components moved
~78 ms while the remainder moved 0.177 ms. Do not attribute it to the `eigh` path.

**Consequences, binding:**

- **No autonomous / Ralph-style optimization loop.** The framework rule is
  evaluator ≥ E2 before bounded autonomy; the composite here is not.
- Cost claims follow the three-model rule (`docs/VALIDATION.md` §2b): report three,
  freeze one *before* the decision block, decide on
  `max(local-ratio-on-graded-anchor, competitor-anchored)`. The `max` is
  load-bearing — the competitor anchor alone would have passed a carrier that
  did not deserve it.
- **Improving the cost evaluator outranks any further score optimization.** The only
  instrument that would do it is a graded run, which costs a submission.

---

## Harness — what is enabled and why

| capability | layer chosen | bottleneck it removes |
|---|---|---|
| Evaluation | the project's own scripts (`edge_cases`, `mint_provenance`, `vault`, `whest run`) | already best-in-class; wrapped, not replaced |
| Change gate | `scripts/preflight.ps1` | four commands that must all run on every estimator change, previously prose in `VALIDATION.md` §7; adds the ship-file identity check that existed nowhere |
| Pre-registration | skill `slice-contract` | the repo's #1 historical failure mode: a residual model or gate chosen after seeing the answer (Sprint 4 twice) |
| Independent falsification | agent `skeptic` | two "findings with tidy causal stories" turned out to be noise; the one independent audit (rev 11) corrected four claims |
| Orientation | this file + `docs/GOTCHAS.md` | a fresh session previously had to read ~150 KB across 12 stacked GAME_PLAN revisions to learn what was decided |
| Human comprehension | `docs/HUMAN_MODEL.md` (Samuel-owned, write-protected from agents) | no comprehension checkpoint existed anywhere in the record |
| Prompt friction | `.claude/settings.json` allowlist | read-only lab commands prompted on every invocation |
| **Agent portability** (audit 2026-08-07) | `AGENTS.md` + `ai/workflows/` + shared `scripts/` enforcement | the whole harness lived under `.claude/`. Exhausting one agent's quota meant losing the methodology, not just the session |

**The portability layer, in one paragraph.** `AGENTS.md` is now the cross-agent
operating contract and `CLAUDE.md` imports it, so no project behaviour is
agent-specific. The six workflow commands are defined once in `ai/workflows/`;
`.claude/skills/` and `.agents/skills/` (Codex) are thin adapters that load them, and
`scripts/verify_portability.py` (162 checks) fails if either side drifts, grows a rule
of its own, or names a product where it should name a role. The Skeptic exists for
both agents (`.claude/agents/`, `.codex/agents/`) over one shared brief. Enforcement is
one implementation, `scripts/guard_paths.py`, called by both agents' hooks *and* by
`scripts/git-hooks/pre-commit` — the backstop that holds even for an agent whose hooks
nobody has wired. **Switching drivers is: start the agent in the repo root, run
`session-start`, continue with `next-slice`.** Not verified in this environment: the
Codex hook *registration* schema, because Codex is not installed here — see
`.codex/hooks/README.md`. The pre-commit backstop covers the gap meanwhile.

**Deliberately not enabled** — each was considered and rejected for a stated reason:

| rejected | reason |
|---|---|
| Ralph / autonomous loops | cost half of the evaluator is E1. Forbidden until a graded anchor exists |
| MLflow / Phoenix / Langfuse | ~30 decisive experiments total, and the artifacts already exceed what these capture. None can model a single-use seed block or a touch-counted holdout — which are the properties that actually protect this project |
| Context7 | two pinned dependencies; the real API surface is already verified against the installed package and recorded |
| Serena | ~70 source files; `rg` + Read is sufficient |
| Repomix | the repo is local; subagents read it directly |
| Promptfoo | there is no LLM in the product |
| Playwright / Chrome DevTools / design skills | no UI anywhere |
| Exa / Firecrawl / last30days | the research surface is one forum and one leaderboard, already scripted (`board_snapshot`, `board_forensics`, `board_census`, `phase2_checklist`) |
| GitHub MCP | `git` + `gh` cover it |
| Symphony / Agent Deck / workflow orchestration | one operator, one working tree, evidence-gated work. Parallel agents here would create merge and provenance risk, not throughput |
| Google Drive / Slack / Gmail / Calendar MCPs | irrelevant to this repo; Gmail and Calendar are also unauthenticated in this environment |

---

## Next actions

*Rewritten 2026-08-19 for a cold start. The previous list had duplicated
numbering and four items whose deadlines had passed.*

**Start here.** `session-start`, then this list in order. Nothing below depends
on chat history; every number has an artifact under `experiments/`.

*Stage 0 (the instrument repair) is COMPLETE — see the section above. Items 2 and
3 of the previous list are done: the Skeptic has now seen the v2.6 build and
returned **ACCEPT**, after the odd-width defect it exposed was fixed.*

1. **Poll Phase 2 — it opens 2026-08-20 23:59 UTC and the config arrives by
   email.** `uv run python local_sim/phase2_checklist.py`. Record into
   `docs/RESEARCH_NOTES.md` §1: width, depth, `n_mlps`, `flop_budget`, λ, the
   multiplier floor, and the flopscope/whestbench versions. **Nothing else in
   this list should start before this is known**, because it decides whether the
   two frozen constants still hold and whether the mechanistic lane reopens.
   *Toolchain intel, 2026-08-19: **whestbench 0.15.0** shipped 2026-08-17 and
   requires **flopscope >=0.11.0**. Its `scoring.py`, `budget.py`, `domain.py`
   and `generation.py` are **byte-identical to the installed 0.14.0**, so the
   score identity, the 0.1 floor, λ = 1e11 and He-init/no-bias generation are
   unchanged in the newest published evaluator. **What flopscope 0.11 does to the
   PRICE TABLE is unknown** — that is the cost half, and it is our weakest half.*

2. **The gate-change contract** — adopt the paired arithmetic score difference as
   the primary promotion statistic, with the studentised t or basic bootstrap
   interval, demoting the geometric mean to a stability diagnostic. **STRATEGIC**:
   it needs its own Slice Contract, Samuel's prediction, and a Skeptic.
   `docs/VALIDATION.md` §2b is deliberately **unchanged** until then. The evidence
   it needs already exists (Stage 0, above): the two estimands disagree on our own
   graded data, and the coverage of every candidate interval is now measured.

3. **If the shape moved: re-fit before gating.** `_PRUNE_TAU` (constants audit
   selected −2.5 at 256×32; −3.0 is worse at 22.6σ, −2.0 at 13.3σ) and
   `_BASIS_RANK` (32, chosen for reproducibility not accuracy — rank 256 was
   1.0108× better at only 0.7σ). The alive-fraction optimum and the effective rank
   of `J Jᵀ` both depend on width and depth. Use a design block; **D17 is the
   single-use decision block and must not be spent on a shape that no longer
   exists.**

4. **Gate v2.6 on D17**, three-model residual rule, paired geometric mean, CI
   excluding 1.0. Then **one vault touch** (counter is at 1 of 3). Gate on `V·c`,
   not on MSE — MSE is already known to cost 0.51%.

5. **First Phase 2 submission — the instrument that decides #2 of the open
   questions.** It returns the grader's own residual telemetry for the new
   carrier, which is the only way to settle the local/graded disagreement.
   **Requires Samuel's explicit approval** (hard rule 6). Order is always
   `whest validate` → clean mini run (`n_failed_mlps: 0`) → ask. Consider
   submitting frozen **v2.4.1 first** purely for configuration telemetry, then
   v2.6, so the two are separable on the grader's own numbers.

6. **Two questions to the organisers** — `docs/email-draft-organizers.md` already
   exists. (a) Is there a Phase 2 write-up round, and when? (b) One designated
   submission or two — the announcement (18143) says two, the Rules tab says one,
   a participant raised it and it was not answered.

7. **Team freeze 2026-09-05 23:59 UTC**, 17 days out. Decide if this stays a
   solo entry.

8. **Not before #1:** the degree-4/5 design carrier (open question #3) and the
   Moonshot's D10 renormalization fit (open question #4, kill-by-default Sep 1).
   Both are real; both are conditioned on the Phase 2 shape.

**Explicitly NOT next**, each closed by measurement this month and recorded in
§ *Closed lanes*: any further input-space variance device (whitening, tent,
sphere, antithetic, degree-3 designs); mid-layer or layer-1 control variates;
MLMC over depth; more N-as-an-operating-point work; and chasing the sub-1e-8
board entries, which three independent forensic analyses fail to explain by any
constructible estimator.
