# CURRENT_STATE.md — read this first

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
> **`SKEPTIC WP14`'s PROVISIONAL on CLAIM A is discharged for the per-layer FLOP
> arithmetic and for nothing else — and `SKEPTIC WP14-FOLLOWUP` re-issued
> PROVISIONAL on the axis that decides.** `whest validate`, the Linux memory
> gate, an OFFICIAL residual reading and an end-to-end candidate are all still
> ahead, and the layer's NUMBERS are still unchecked against upstream at non-zero
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
