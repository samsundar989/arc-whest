---
title: "A first-order lineage chain for the means of random ReLU networks: what it gets right, what each part is worth, and where it stops"
subtitle: "ARC White-Box Estimation Challenge 2026 — Phase 2 algorithmic-contribution write-up"
author: "neopok919 (Samuel Sundararaman, solo) — submission #337601"
date: "28 October 2026"
---

**Submission ID: #337601** (graded 21 Oct 2026 19:02 UTC; adjusted score 1.8751 × 10⁻⁹; raw final-layer MSE 1.4035 × 10⁻⁸; 293.8 GFLOP per network = 13.4 % of budget; 0 failures / 100).
**Code:** MIT, https://github.com/samsundar989/arc-whest. **Numbers:** every figure in this document is traceable to a logged experiment file; the map is in Appendix C and in `docs/writeup_datapack.md` of the repository (cited below as *DP §x.y*).

---

## 0. Summary

**The task.** Take a random neural network — 16 layers, 1,024 neurons wide, ReLU activations, weights drawn the standard way (He initialisation) — feed it Gaussian noise, and predict the *average output of every neuron* without running it. The grader scores the mean-squared error of the last layer, multiplied by the fraction of a fixed compute budget (2⁴¹ ≈ 2.2 trillion floating-point operations per network) you spend, with a floor at 10 %. Spend less than 10 % and you gain nothing more; spend more and your error is scaled up in proportion.

**What we built.** A *cumulant chain*: instead of running the network, we track a short list of statistics of each layer — the means, the full covariance, the lopsidedness (third cumulant, κ3) and the tails (fourth cumulant, κ4) — and push them through each layer with formulas instead of samples. The expensive part is the lopsidedness: a full κ3 of a 1,024-wide layer is a billion numbers. We store it as *lineages*: each partially-on neuron throws a lopsided "shadow" with a simple hub-and-spokes shape; we keep each shadow as two vectors, carry it forward exactly for six layers, and then fold all older shadows into one shared 288-dimensional basis that *rides* along with the chain. Tails are handled by fitted laws plus cheap readers for the youngest tail terms born from κ3. A small learned corrector (34,832 weights, 0.12 % of the compute) fixes what the formulas miss, layer by layer. A planner chooses how to multiply the matrices (batched Strassen–Winograd) so that both the FLOP bill and the grader's per-operation overhead stay inside the limits.

**The result (#337601).** Final-layer MSE 1.40 × 10⁻⁸ at 13.4 % of budget ⇒ adjusted score **1.875 × 10⁻⁹**, 0/100 failures, rank 12 (tied) on the public board (read 27 Oct; DP §1.1). The grader's sampling reference MSE on the same panel is 1.13 × 10⁻⁶, so our raw error is roughly 80× below plain Monte Carlo at this budget (we take the grader's figure as given; its recipe is not documented, so treat the ratio as approximate). Each of our last nine graded submissions landed within ±0.5 % of the forecast we wrote down before submitting (DP §1.2). Throughout, *VAL* is our fixed 100-network development panel and *FRESH* a panel of 100 networks never used in development; "wins" counts how many of the 100 improved.

**What we claim as ours** (each measured on/off; §4 has the table, §8 the credits and the prior art we know of as of 27 Oct):

1. **The data law for the corrector** — one pass over ~2,000 freshly labelled networks lowers error by 0.4–0.6 % at zero FLOPs, reproduced four times (A4, A5a, A5b, A5c; DP §4.3–4.4) — and the observation that 12,000 publicly labelled networks had gone unused. No posted write-up reports a data-scaling measurement.
2. **A rerun-robustness rule:** we nominated by worst-day overhead headroom against the grader's slowest recorded day (82 graded runs, 23 days), not by best public score; the best-scoring entry (#337495, 1.853 × 10⁻⁹) was deliberately *not* nominated (§6).
3. **Two readers for κ4 terms born from κ3:** the repeated-index (2,1,1) birth term ("tp2"), whose age-1 reader reuses a product the chain already forms (the κ3 legs pulled back to the current layer, "Dx"); with two companion zero-FLOP terms it measured −0.30 % raw (77/100) at ≈ +0.15 % FLOPs in a matched refit, and −1.4 % VAL / −1.6 % FRESH at +0.5 % FLOPs as part of the NEW10 package shipped in #337601 (DP §4.2; there is no tp2-only number) — the term is derived, its coefficient fitted; and a nested ("Horner") pullback for ages 2–3 that costs one product per age-layer instead of re-pairing the full κ3 (−1.24 % raw, 100/100, at ≈ +4.4 % FLOPs on the real meter — not shipped; the score cannot pay for it at our spend; DP §6 N5, §3.4). Susupro's full K = 4 chain (18254) feeds the (2,1,1) slice into κ3; ours is the cheap memoryless version inside a K = 3 budget.
4. **A joint-objective planner:** every array call is priced at λ = 3.5 million FLOPs and each product is scheduled to minimise bill + λ·calls, because the grader fails a network whose non-FLOP overhead exceeds 0.4 s; λ comes from the per-call overhead law measured on flopscope 0.12.3 over 82 graded runs (DP §3.8, §5). Others report the same cliff and cap Strassen depth by product size (suliman_tadros 18289, pranay212 18273); the explicit joint objective and the measured day-by-day overhead law are ours.
5. **The riding band, as a variant:** old κ3 content in one shared rank-288 basis, entered at age 7 and never re-rotated. Shared bases for old κ3 are prior art (504aldo V22/P5; Artyom_Mazur 18249 "rides the transport", rank 448; mliston 18287 rank 384 with re-joins; dogus_ozel 18281 rank 256). What we add is the measurement: replacing a per-layer re-join by the riding band is −2.66 % score at equal error, 92/100 (join 20.6 → 7.4 GFLOP, DP §6 row 4); the age/rank frontier on the real meter (§5.3); and the cost structure — the band is readout-dominated, 0.058 % of the bill per column.
6. **An error anatomy on exact labels** (10⁹ samples per network, two orders of magnitude below the chain's error): about three quarters of the final error is mean error *inherited* from earlier layers; the κ3 slice error is spread out — not per-neuron-local, not pair-local, not inherited — and several tempting shortcuts have measured floors: pooled importance-capped carriers (5–10× worse than a basis), sketched "dressing", column cuts, zero-FLOP covariance faces (§5). Oracle maps with Monte-Carlo truth appear in 18281, 18284 and 18272; ours adds the exact-label floor and the negative structure tests.
7. **The coupling direction** between κ3 and κ4: on exact towers, κ4 content born from an incomplete κ3 *adds* error (+44–88 %); it pays only once κ3 is complete (§5.2). This gives a mechanism for what several teams observed empirically — that the "true" κ4 level makes their chains worse (18251 replies, 18290) — and it fixes the order of any future programme.

**What is not ours** (credited in §8 and in the code): the cumulant-propagation frame (ARC's reference); two mechanisms re-implemented from 504aldo's public V29 — the λ-law κ4 regeneration and the 7→4 leg identity — with our own λ fit and our own reconstruction identity; the physics of the young-tree κ4 reader (forum 18249, Artyom_Mazur / Kuznetsov), which we implemented and which is the single largest κ4 gain in the chain (−3.5 % raw at +0.95 % FLOPs); kai's two-term Q31 law (used as published); a shared basis for old κ3 content (504aldo V22; Artyom_Mazur 18249; mliston 18287; dogus_ozel 18281 — ours is a variant, see claim 5); dead-unit pruning; Strassen on metered products (others too); a learned per-layer corrector (Susupro, hyojun, mliston also use one). We did not copy code; §8 says exactly what was re-implemented and what we matched deliberately.

**Where it stops.** Every content lever we measured on this chain returns 0.25–0.65 % of error per 1 % of FLOPs; every cut costs 1.4 % to over 100 % of error per 1 % saved. At a 13.4 % spend the chain is at its knee: the content exists (a joint arm reached −3.9 % raw at +6.8 % FLOPs, DP §3.4) but the score cannot pay for it. §7 says what we would do next and why.

**LLM statement.** The project was directed by Samuel Sundararaman, who set the goals, approved the experiment programme and every nominated submission, and takes responsibility for every claim here. Research planning, experiment design, analysis and the first draft of this text were done with Anthropic's Claude (Claude in Cowork, model Fable 5.1, acting as strategist); the code was written by Claude Code (model Opus 5) from written handoffs, and by Samuel. Nothing was accepted on a model's word: every change had a pre-registered prediction, a byte-identical zero control, a paired 100-network panel, and — for anything shipped — a graded canary whose score had to match the forecast. A second, independent pass re-checked every number in this document against its source file.

---

## 1. The problem, explained from scratch

**A random network is a machine of 16 steps.** Each step takes 1,024 numbers, multiplies them by a 1,024 × 1,024 table of random weights, and then applies a *kink*: every negative number becomes zero (that is the ReLU). Feed in 1,024 random Gaussian numbers and you get 1,024 numbers out of each step. The question is: *averaged over all possible inputs, how bright is each of the 16 × 1,024 lamps?* The grader checks the last 1,024 lamps most strictly (the scored "final-layer MSE") and reports all 16 layers too.

**Why not just run it a lot?** You could feed in many random inputs and average. Each run costs about 34 million operations; the whole budget buys about 65,000 runs. Averaging 65,000 noisy numbers leaves noise of order 1/√65,000 ≈ 0.4 % of a lamp's natural spread, and the error that produces is roughly 80× larger than ours (the grader's sampling reference on this panel is 1.13 × 10⁻⁶ against our 1.40 × 10⁻⁸). So sampling is the wrong tool at this budget; we need formulas.

**Why formulas can work.** If a lamp's input were *exactly* Gaussian — a bell curve with a known centre μ and width σ — then its average brightness after the kink has a closed form that depends only on α = μ/σ ("how far the centre sits from the kink, in units of the spread"). A lamp far below zero is almost always off; far above, it is a straight pass-through; in between, the formula interpolates. The same holds for how two lamps co-vary. So if we can carry the *centres, widths and co-variations* from one layer to the next, we can predict the next layer's averages without a single sample.

**Why it is not that simple.** The input to layer 1 is Gaussian; the output of a kink is not. It is *lopsided* (more mass on one side) and its *tails* differ. The next weight table mixes 1,024 such outputs, which pulls the mixture back toward a bell curve — but not all the way. The leftover lopsidedness shifts the next layer's averages by a small amount, and because every layer feeds the next, small shifts compound over sixteen layers. The whole game is: track enough of the non-Gaussian leftovers to get the averages right, and spend as few operations as possible doing it.

**The vocabulary.** Statisticians call these leftovers *cumulants*. The first is the mean (centre). The second is the covariance (width and co-variation). The third, κ3, is lopsidedness (skew). The fourth, κ4, is how heavy the tails are. For 1,024 lamps the mean has 1,024 numbers, the covariance a million, κ3 a *billion* and κ4 a trillion. The budget pays for about 2 trillion operations *in total* over 16 layers. Carrying the full κ3 is already impossible; the craft is in carrying the part that matters.

**What the score rewards.** Score = (final-layer MSE) × max(0.1, FLOPs used / 2⁴¹). Two consequences shape everything below. First, below a 10 % spend the multiplier stops falling, so a method that spends 5 % and one that spends 10 % get the same multiplier; we spend 13.4 %, so every 1 % of FLOPs we save is worth about 1 % of score until we reach 10 %. Second, the grader fails a network outright if it takes more than 120 s of wall time, or if its *non-FLOP overhead* — time the meter cannot attribute to arithmetic, mostly per-operation bookkeeping — exceeds 0.4 s. A failed network counts as predicting zero. So the design has three dials — error, FLOPs, and operation count — and all three appear in what follows.

---

## 2. The chain, one layer at a time

Think of the method as a relay. At each layer it holds a *state* — a handful of statistics about the pre-kink numbers — uses the state to predict that layer's averages, and then transforms it into the next layer's state. Here is what the state contains and how each part moves. (Cost shares are for the shipped build, DP §3.1.)

### 2.1 Means and covariances — exactly (10 % of the bill)

The centres μ and the full 1,024 × 1,024 covariance C of the pre-kink numbers are carried *exactly*: multiply by the weight table, apply the kink formulas, multiply again. The kink step uses the "pair rule": for two lamps with known centres, widths and correlation, the average of the product of their outputs has a closed form (a bivariate Gaussian integral). We evaluate it for every pair — a million pairs per layer — with a 27-term polynomial closure. The matrix products use a two-level symmetric "sandwich" (Wᵀ C W computed once per symmetric half) and the first layer's product is special-cased. This part of the chain is the floor: switch everything else off and the means from this part alone already have the right shape; everything that follows is correction.

**Pruning (the KAS lists).** Many lamps are *decided*: their centre sits so far below the kink (α very negative) that they are off essentially always, or so far above that they are a pass-through. We rank lamps by |α| and keep, per layer, only the undecided ones in the expensive parts — rows for transport (1,024 at layer 1 shrinking to 736 at layer 16), columns for the readouts, and so on (the lists are constants in the code, DP §2). Cutting them further is sharply penalised: removing 128 rows at layers 2–5 saves 1.26 % of FLOPs and costs +5.7 % error (DP §3.2).

### 2.2 Lopsidedness (κ3) — as lineages (67 % of the bill)

**Where lopsidedness is born.** A lamp that is *partially* on — centre near the kink — produces a skewed output. How skewed depends on α through a weight w = σ·φ(α), where φ is the bell-curve density: lamps at the kink contribute most, decided lamps nothing. Crucially, the newborn skew has a *shape*: it is dominated by terms in which one index is the lamp itself (the **hub**) and the other two are lamps that co-vary with it (the **spokes**). We checked this in a sandbox against the exact third cumulant (n = 128, 8 layers): the diagonal captures 58 % of its effect on the means, hub-and-pair terms take it to 90 %, the full hub-star to 99.8 % (DP §6 row 1). Terms with a *cycle* (triangles) are the only ones that cannot be contracted with matrix products, and they are under 1 % of the hub term at this width (Appendix A).

**Storing a shadow as two vectors.** So instead of a billion-entry cube, each hub's contribution is a *lineage*: a weight w, a vector **u** (how the hub's output propagates into the next layer — a column of the layer Jacobian), and a vector **v** (how the hub co-varies with its spokes — a column of the cross-covariance). What the next layer's kink formulas need from κ3 is rebuilt from these: the diagonal slice D3 (each lamp's own skew, 3 Σ w·u·v²) and the pair slice D21 (how a lamp's skew couples to its partners, formed from (J∘X)Xᵀ + (X∘X)Jᵀ with the lineage matrices J and X). These two slices are all the kink formulas consume; D21 is formed only on the undecided rows and columns.

**Carrying shadows forward: ages.** When a lineage passes through a layer, its two vectors are multiplied by the weight table and reweighted by the kink's response ("transport"). We do this exactly for six layers — a lineage has an *age* from 1 to 6 — because each age-layer is one n³ matrix product; the transport family (the six ages plus the band) is 36 % of the bill. We measured the alternatives: five exact ages saves 4.5 % of FLOPs and costs +6.2 % error; seven costs +3.75 % FLOPs for −1.7 % error (DP §6 row 3, N2). Six is the knee.

**The riding band: what happens to old shadows.** At age 7, a lineage is folded into a shared basis of rank 288 that rides along with the chain: every older lineage is expressed in the same 288 directions, and from then on only the basis is transported. A shared basis for old κ3 content is not our idea (504aldo's age-gated basis; Artyom_Mazur's rank-448 basis "that rides the transport", 18249; mliston 18287; dogus_ozel 18281). Our variant keeps one band that is never re-rotated, so the join into it is cheap; replacing our own earlier per-layer re-join with it cut the join from 20.6 to 7.4 GFLOP — −2.66 % score at equal error, 92/100 networks (DP §6 row 4). The band's cost is dominated by *reading it out*, not carrying it: each column costs about 0.058 % of the bill, so raising the rank to 320 buys −1.2 % error for +1.86 % FLOPs — not worth it at our spend (DP §3.3).

**The young window is the whole story.** Births, exact-age transport and the young readout together are 67.5 % of the bill; the old tier (band rebuild, read and join) is 15 %; the exact covariance chain 10 %; everything else — closure, κ4 laws, corrector, bookkeeping — under 8 % (DP §3.1). Any plan to spend less has to touch the young κ3 window, and §5 shows why every such cut we tried loses more error than it saves FLOPs.

### 2.3 Reading the state into averages (the Edgeworth readout)

Given the centre, width, D3 and D4 (its own tail term, the diagonal of κ4) of a lamp, its average after the kink is the Gaussian formula plus small corrections proportional to D3 and D4 times derivatives of the bell curve (an Edgeworth expansion, Appendix A), with damping terms (θ) fitted so the series does not overshoot where the lopsidedness is large. The same expansion gives the covariance corrections, which is how skew changes *next* layer's covariances and, through them, the layer after. In the sandbox the direct final-layer readout removes about 21 % of the covariance-only error, the indirect route through intermediate means 58 %, both together 91 % (DP §6 row 7) — which is why old lineages matter mostly through the means they shift along the way, not through the last layer's readout.

### 2.4 Tails (κ4) — by laws and young readers (≈ 1 % of the bill)

A full fourth cumulant is out of reach, so we use three pieces:

- **A regeneration law.** The diagonal tail term each lamp needs is well approximated by λ_l times a quantity built from the covariance. This is 504aldo's published idea; we derived the law's form ourselves in a sandbox, fit λ per layer on 24 fresh networks with no labels, and found the fitted values agree with the first-principles ones in order of magnitude (DP §6 row 8). The sandbox also shows *why* the law works: every layer injects an emergent "scale mode" (fluctuations of the layer's overall size) that dominates κ4 at depth and is a first-order response to κ3 (Appendix A).
- **kai's two-term Q31 law** for the (3,1) slice, used as published (DP §6 row 9).
- **Young tail trees born from κ3.** When a skewed input meets a kink, the output's tail term inherits structure from the skew. The physics of reading these young trees off the κ3 legs is the 18249 team's (Artyom_Mazur / Kuznetsov, write-up #333534); our implementation of that reader is the largest single κ4 gain in the chain (−3.5 % raw at +0.95 % FLOPs, 94/100; DP §6 row 13). On top of it we added **tp2**, the repeated-index (2,1,1) birth term, whose age-1 reader reuses a product the chain already forms (the κ3 legs pulled back to the current layer, "Dx"). The term's form is derived from the born-κ4 expansion (Appendix A.6); its coefficient is a fitted gain, as all our κ4 coefficients are — we did not find a way to use first-principles values without harm, and §5.2 says why we think nobody has. Measured with two companion zero-FLOP terms (mean-field older ages, a var² D4 face) in a matched refit: −0.30 % raw, 77/100, ≈ +0.15 % FLOPs (DP §6 N4; no tp2-only number exists); shipped in #337601 inside the NEW10 package (those three plus six closure/metric gains) worth −1.4 % VAL / −1.6 % FRESH at +0.5 % FLOPs (DP §4.2). A naming note for readers of the code and ledger: the 18249-style reader's tables are `_YP_D4_GAIN`/`_YP_Q31_GAIN` (called the "tp2 Dx reader" in older notes); the (2,1,1) term claimed here is `_YP_TP_GAIN`/`_YP_TPQ_GAIN`. Older trees (ages 2–3) can be read by a **nested (Horner) pullback**: one product per age-layer instead of re-pairing the full κ3 — measured −1.24 % error (100/100) at ≈ +4.4 % FLOPs on the real meter (+3.4 % planned), which the score cannot afford at our spend, so it is off in the shipped build (DP §6 N5, §3.4). We also measured how faithful the shipped tail readers are to exact labels: the inputs they read are essentially exact (fidelity loss ≈ 0); what is missing is coefficients and coverage, worth +0.10–0.16 R² at the middle layers (DP §4.8).

### 2.5 The corrector (0.12 % of the bill)

At every layer, before the means are passed on, a tiny network (33 inputs → 32 → 32 → 1, tanh, one per layer, 34,832 weights in all) reads nine per-lamp numbers the chain already has — the centre, the width, the estimated skew and tail terms and a few of their products — and outputs a small adjustment. It was trained on the chain's own running state, layer by layer, against Monte-Carlo labels (10⁸ samples per network) for public random networks (the keenanpepper corpus, 14,048 networks; DP §4.3). It is a corrector, not the estimator: its first version was worth −5.6 % error on the development panel (DP §6 row 20), the later refits a further −1.4 % (R3), −1.4 % (A3, with the NEW10 tables), −0.6 % (A4) and −0.45 % (A5) (DP §4.1–4.4), and when the chain's κ3 and κ4 are replaced by exact values the corrector's benefit vanishes (DP §6 M3) — it is fixing the known shortfalls of the lineage approximation, not inventing the answer. (We do not have a clean corrector-off number on the shipped chain; it is on the gap list in §8.)

### 2.6 The planner: paying for operations, not only FLOPs

The grader's meter charges FLOPs for arithmetic but also spends real time per array operation, and it fails any network whose non-FLOP time exceeds 0.4 s. Strassen–Winograd multiplication lowers FLOPs (seven block products instead of eight, recursively) but multiplies the operation count. Our planner treats the operation count as a design variable: every product is scheduled to minimise FLOPs + λ_op × (operations), with λ_op = 3.5 million FLOPs per operation in the shipped build, recursion depth up to 6, leaf size 8, padding and trimming rules, and all seven sub-products of a level batched into single flat calls. The shipped build issues 20,991 operations per network at 293.8 GFLOP; its measured overhead is 0.29 s median on the grader, with +12.9 % headroom on the slowest grader day on record (DP §5). Other teams report the same cliff and manage it by capping Strassen depth by product size (pranay212 18273) or by counting round trips (suliman_tadros 18289, ≈ 4 µs per round trip); our per-call rate on flopscope 0.12.3 is 13.5–15.9 µs per logged operation and it drifted upward through October (DP §5), which is why we price calls explicitly rather than cap depth. The same chain at the cheapest-FLOP setting (λ_op = 0.6 million, 22,633 operations; A1 versus A1s) scores 1.7 % better on a quiet grader day but has only +2.5 % overhead headroom on the worst one; §6 explains why we nominated the safe planner.

### 2.7 Putting the layer together

One layer of the shipped chain, in order: transport the κ3 lineages and the band (matrix products); form D3 and D21 from them; apply the tail laws and the young-tree/tp2 readers; run the pair-rule closure on the covariance; read out means and covariances with the Edgeworth corrections; apply the corrector to the means; prune the lists for the next layer; hand the state on. Sixteen times. The last layer needs only means, so it skips what it can (4 % of the bill, DP §3.1).

---

## 3. How it was developed: a measurement-first loop

The method was not designed and then tuned; it was grown by a loop in which every proposed change had to *predict* its effect before it was allowed to *show* it. The loop had four fixed parts. (Build names used below: **8be** = the incumbent of 18 Oct; **A1** = 8be with the corrector refit on fresh labels; **A3** = A1 + the NEW10 tables; **A4**, **A5** = A3 after one and two more epochs of fresh labels; the suffix **s** = the safe planner, λ_op 3.5 × 10⁶; **B4** = A5's twin with a shallower, no-padding planner.)

**Exact labels.** For diagnosis we used networks whose per-neuron means were measured to 10⁹ samples (label noise ≈ 7.7 × 10⁻¹¹ MSE, two orders of magnitude below our error), so that an "error" is the chain's, not the label's. For training and panels we used the public keenanpepper corpus (10⁸ samples; label floor ≈ 7.5 × 10⁻¹⁰ on a 100-network panel, DP §6 M4).

**Paired panels and controls.** Every variant was compared on the same 100 networks (VAL, networks 900–999) and, before shipping, on 100 fresh networks it had never seen (FRESH-A). We report the paired mean change, the number of networks that improved (wins out of 100), and a 95 % t-interval. Every variant carried a *zero control*: with its new gains set to zero, its predictions had to be byte-identical to the parent's (two networks, equal BLAS threads) before the arm was allowed to train. This catches the commonest error in this kind of work — a change that "helps" because something else moved.

**Pre-registered predictions and keep rules.** Each handoff stated the predicted change with an 80 % interval and the rule for keeping the result (threshold, wins, interval excluding zero, FLOP price on the real meter) *before* the run. The ledger in §4 and Appendix B records predictions next to outcomes, including the ones that missed.

**Graded canaries.** Anything shipped was first submitted as a canary with its forecast written down. The forecast was anchored on FRESH-A and the real flopscope bill; a canary that missed by more than ±1.5 % would have been investigated, not nominated. None did:

| canary (DP §1.2) | forecast | graded | miss |
|---|---|---|---|
| 8be #336993 | 1.8956e-9 | 1.8962e-9 | +0.03 % |
| A1 #337456 | 1.863e-9 | 1.8656e-9 | +0.14 % |
| λ-twin #337470 | 1.8706e-9 | 1.8708e-9 | +0.01 % |
| A1s #337481 | 1.8970e-9 | 1.8974e-9 | +0.02 % |
| A3 #337495 | 1.8447e-9 | 1.8527e-9 | +0.43 % |
| A4s #337511 | 1.8819e-9 | 1.8800e-9 | −0.10 % |
| B-young #337462 / B4 #337562 | 2.0007e-9 / 1.9728e-9 | 2.0006e-9 / 1.9724e-9 | −0.01 % / −0.02 % |
| **A5s #337601** | **1.8715e-9** | **1.8751e-9** | **+0.19 %** |

The last day's chain of shipped builds shows the loop at work: 8be → A1 (corrector refit on fresh labels, −1.5 % FRESH) → A3 (+ NEW10, −1.6 % FRESH at +0.5 % FLOPs) → A4 (+ one epoch on 2,048 fresh networks, −0.6 %) → A5 (+ one more epoch, two arms averaged, −0.53 % FRESH). Each step was predicted, measured on VAL and FRESH, verified for bill, determinism, memory and 100-seed robustness, and graded (DP §4.1–4.4, §1.2).

---

## 4. What each piece is worth

The table lists the mechanisms in the shipped build with the measurement that justifies each one — its FLOP price and its error change on the stated panel — and who the idea belongs to. Negative Δraw is better. "Graded" means the change was confirmed on the public grader. Full rows, sources and dates are in DP §6; Appendix B lists what we tried and rejected.

| # | mechanism (shipped in #337601) | ΔFLOPs | Δ error (panel, wins) | graded | idea |
|--|------------------------------|----------|----------------------------|----------|----------------|
| 1 | hub-star κ3 lineages (diag + pair + (2,1,1) hub births, factored legs) | — | sandbox: 58 → 90 → 99.8 % of the κ3 effect; routes off on real chain: +5.4 %† | — | ours (frame: ARC) |
| 2 | six exact ages (vs five) | +5.55 % | −6.15 % VAL, −5.84 % FRESH | −1.54 % | ours (choice) |
| 3 | **riding band r288** (vs our per-layer re-join) | −2.59 % | −0.08 % (92/100) | −2.66 % | variant of shared-basis prior art (504aldo, 18249, 18287, 18281); measurement ours |
| 4 | 7→4 leg identity, rank 4 | −28.7 % (vs v30) | rank 2: +11 %† | — | 504aldo P2; reconstruction identity ours |
| 5 | Edgeworth readout + θ-damped closure extras | — | extras off: +18 %† | — | standard; damping ours |
| 6 | κ4 regeneration λ-law | ≈ 0 | sandbox residual −38 %; ≈ 90 % of the "augment" gain at zero n³† | — | 504aldo P1; λ fit + derivation ours |
| 7 | kai's two-term Q31 law | −1.09 % (in C1 cuts) | −0.19 % FRESH-D | — | kai (18246) |
| 8 | joint κ4 coefficient refit (Tier-1 law + Tier-1.5/2 births) | — | −4.21 % VAL vs matched control | −1.3 % | ours |
| 9 | fold κ4 births into D21 (abs(α) ≤ 2) | +0.26 % | −1.89 % VAL (83/100), FRESH 1.019× | −2.6 % | ours |
| 10 | masked star-Q31 fold | +0.27 % | graded raw −3.04 % (FRESH −2.84 %, 86/100) | — | ours |
| 11 | young-tree κ4 reader (`_YP_D4_GAIN`, `_YP_Q31_GAIN`) | +0.95 % | −3.53 % VAL (94/100) | −3.70 % | physics: 18249 (Artyom_Mazur / Kuznetsov); implementation ours |
| 12 | **tp2** (2,1,1) birth reader (+ 2 zero-FLOP companions), in NEW10 | ≈ +0.15 % (3 terms); +0.50 % NEW10 | −0.30 % (3 terms, 77/100); NEW10 −1.42 % VAL / −1.63 % FRESH (89/100) | A3 vs A1 −0.69 % | ours |
| 13 | dressed newborn legs; D21ᵀ dress 0.9 | 0 | −6.5 % FRESH (85/100); ×0.9: −0.38 % | — | ours |
| 14 | KAS pruning lists (several rounds) | −11.4 % … −3.6 % per round | +2.2 % … −0.16 % | −10.1 %, −1.63 % | dead-unit pruning: common; lists ours |
| 15 | exact K2 chain: symmetric sandwich, layer-0 hygiene | −0.23 %, −2.16 % | 0, +0.02 % | — | ours |
| 16 | batched Strassen–Winograd + planner options (sub-block trimming, padding, op diet) | −0.79, −4.73, −1.69, −1.04, −5.97 % | 0 … +0.07 % | −6.10 % (op diet) | Strassen: common; planner ours |
| 17 | **λ-priced flat-ops planner** (λ_op 3.5e6) | +1.68 % (A1 → A1s) | 0 (max abs Δ 7e-6) | +1.7 % score, +10.6 pts headroom | ours |
| 18 | in-loop corrector (first version); hub features; local-even | +0.06 % | −5.60 % DEV; −0.48 %; −0.16 % | — | learned corrector: common; features ours |
| 19 | corrector refits on fresh labels: R3, A4, A5 | 0 | −1.48 %, −0.58 %, −0.53 % FRESH | −1.61 %, —, −0.26 % | ours (data law) |

† Measured on an older version of the chain (September) and not re-measured on the shipped build; listed for completeness, not claimed as current.

Three things stand out. The κ3 carrier (rows 1–4) is where the FLOPs go and where the big error reductions came from. The κ4 readers (rows 6–12) are cheap and together worth roughly −13 % error (a rough compound of deltas measured against different parents, not an additive sum). And nothing after row 12 is a content lever: rows 13–19 are zero-FLOP or cost-side engineering, each small, each measured, and roughly −20 % of score in total — which is how a method that is "just" a first-order lineage chain ends at rank 12.

---

## 5. Where the error is, and why it stops here

### 5.1 Anatomy (exact labels, 10⁹ samples; DP §6 M1–M5, §4.7)

- **Three quarters of the final error is inherited.** Replace the chain's layer-15 means by exact values and the final MSE drops by 74–79 % (two networks). The last layer's own readout is not the problem; the state it receives is.
- **The covariance is half of that.** Replace the pre-activation covariance by exact values and the error drops 38–39 %; replace everything (κ3, κ4, covariance) and it drops to 6.5 % of itself.
- **The κ3 slices carry a growing, incoherent error.** The served D21 slice is 3.6 % wrong at layer 2, 35 % at layer 3 and 68 % by layer 14 (DP §4.7). We tested every cheap structure this error might have: it is not predictable per neuron, not per pair (the best pair-local model explains R² ≤ 0.26 of it; a scale factor alone explains 44 % of its energy at deep layers, but applying that scale without retraining the corrector makes the chain *worse* by +35 %), and it is not inherited from the previous layer's slice. A "transported-state" corrector built on the inheritance hypothesis was worth −0.84 % at +1.3 % FLOPs (DP §4.5) and nothing on the refit chain.
- **The covariance error has one coherent, zero-FLOP face** (ρ_t·μμᵀ, a depth law, DP §6 N11a): it explains 40–64 % of the unweighted covariance error at layers 9–15 but under 1 % once weighted by what the means feel — it is mean-neutral, and the fitted λ-law already captures its scale mode.

Other teams built similar oracle maps with Monte-Carlo truth (dogus_ozel 18281: exact carried means at layers 8–13 leave ×0.39; yeongjunyoo 18284; michael_anderson7 18272; megan_park 18282) and reach the same first conclusion. The picture: the chain's error is the *diffuse* remainder of a first-order approximation. No slice of it is local, sparse or low-rank. That is why the levers that remained were all "more of the same content" levers, and why they are all priced the same.

### 5.2 Floors with a mechanism (sandbox, exact towers)

- **Pooled carriers lose.** A single pool of hub columns under an importance cap — the only structural fact disclosed about some leaders' carriers — is 5–10× worse than exact ages + a shared basis at equal cost, at n = 128 and 256, under every importance rule (DP §6 N14). Hub content in a He-initialised network is *spread*, not sparse; compression must be by basis, not by selection. The split needs rank ∝ n to sit on its floor.
- **The κ3 × κ4 coupling has a direction.** On exact towers (n = 64, 8 layers), carrying the full κ4 on top of a κ3 truncated at age ≤ 2 makes the error *worse* by 44–88 %; at age ≤ 4 it is neutral; with the complete first-order κ3 it helps by 4–47 % (§L sandbox). κ4 content born from a deficient κ3 adds error. This is, we think, the mechanism behind an observation several teams made independently on their real chains — that imposing the *true* κ4 level makes the estimate worse (LuDoe, megan_park and yeongjunyoo in the 18251 thread: ×1.3–1.7 on 24/24 networks; foyshal_sk 18290) — and why fitted κ4 constants come out 2–5× the Monte-Carlo value: they are absorbing the κ3 deficit. Our real chain sits between the "age ≤ 4" and "complete" rows (a 6 % shape deficit on inherited content), which is why every κ4 reader measured on it came out small and additive, and why the joint arm J (NEW10 + nested readout + rank 320, refit together) was additive, not super-additive: −3.90 % raw against −3.85 % from the parts (DP §3.4).
- **Dressing is a bulk-trace effect.** The interaction of κ3 with the correlation pattern (κ3∘ρ, which the chain approximates with a fitted dress gain) is an n⁴ sum over cycles; it has no low-rank or sketched shortcut (every sketch has relative variance ≈ n/k, DP §6 N26).
- **Nothing cheaper carries the covariance error either.** A rank-k error state reproduces 88–99 % of the covariance error at n = 128 but only 20–50 % at n = 1,024 (DP §6 N11); a sphere (radial) formulation of the same chain gains 17 % at n = 64 and 9 % at n = 256, shrinking like 1/√n (N9a) — and our attempt at it on the real chain failed outright (+380 %, a specification error we did not have time to find).

### 5.3 The knee

Put every lever we measured on the real chain on one axis — error change per 1 % of FLOPs — and they line up (DP §3.3–3.6, §4.5, §4.8):

| lever (real meter, paired VAL) | ΔFLOPs | Δ error | error per 1 % FLOPs |
|------------------------------|--------|----------------|----------|
| rank 288 → 320 | +1.86 % | −1.18 % | 0.64 |
| nested readout ages 2–3 | +4.4 % (metered in J; +3.4 % planned) | −1.24 % | 0.28 |
| (5 ages, rank 384) reallocation | +2.21 % | −1.07 % | 0.49 |
| (4 ages, rank 512) | +7.70 % | −1.89 % | 0.25 |
| transported-state corrector (arm A; FLOP forecast) | +1.29 % | −0.84 % | 0.65 |
| joint arm J (all of the above refit together) | +6.78 % | −3.90 % | 0.58 |
| clean-Dx tail reader | +24 % | break-even needs −19 % | — |
| **cuts:** 128 rows off layers 2–5 | −1.26 % | +5.68 % | 4.5 |
| output rows → max(R, D, H) | −1.52 % | +6.15 % (refit: +5.83 %) | 4.0 (3.7) |
| no old-tier rebuild at layer 15 | −0.95 % | +107 % | 113 |
| no band read at layers 7–9 | −1.27 % | +60.9 % | 48 |
| five ages instead of six | −4.5 % | +6.15 % | 1.4 |
| **zero-FLOP:** one fresh epoch of corrector data | 0 | −0.38 … −0.60 % | ∞ |

Every content lever returns 0.25–0.65 % of error per 1 % of FLOPs; the score needs 1.0 to break even. Every cut costs from 1.4 % to over 100 % per 1 % saved. The only levers that pay are at zero FLOPs, and we used them (rows 19 of §4) until the day ran out. This is what we mean by "the chain is at its knee": not that nothing improves it, but that at a 13.4 % spend nothing improves it *at the price the score charges*. The joint arm J is the cleanest statement of it: −3.9 % error, 99/100 networks, and a score-equivalent of +2.9 %.

---

## 6. The rerun and the nominee choice

The prize rerun uses a fresh suite on the organisers' hardware, and a network that exceeds 120 s of wall time or 0.4 s of non-FLOP overhead scores as a zero prediction. We treated this as the dominant risk and measured it (DP §5):

- From 82 graded runs over 23 days we extracted the grader's per-operation overhead rate by day (10.2 µs/op in September, rising to 13.5–14.2 µs/op after 15 October; worst 15.9 µs/op), the worst within-run tail (a single network at 1.196× the run's median overhead; 1.242× for wall), and the worst day-to-day shift for identical code (1.029× overhead, 1.031× wall).
- For each candidate build we computed the overhead it would see on a replay of the worst day, with and without the day shift, and the *headroom* 1 − worst/0.4 s. Our rule: nominate only builds with ≥ 10 % headroom on both.

| build (DP §5) | ops | graded overhead med/max (s) | worst-day headroom (central / pessimistic) | slowdown to first failure |
|--------------------------|------|--------------|------------------|------------|
| A3 #337495 — best public score 1.853e-9 | 23,374 | 0.333 / 0.345 | **+0.4 % / −2.4 %** | 1.16× |
| A1 #337456 — 1.866e-9 | 22,633 | 0.326 / 0.335 | +2.5 % / −0.3 % | 1.19× |
| **A5s #337601 — nominated, 1.875e-9** | 20,991 | 0.291 / 0.302 | **+12.9 % / +10.4 %** | **1.33×** |
| **B4 #337562 — nominated, 1.972e-9** | 19,684 | 0.267 / 0.282 | **+20.0 % / +17.7 %** | **1.42×** |

A3 scores 1.2 % better than A5s on a quiet day and fails on a bad one. We nominated A5s (the same chain with the safe planner and two more corrector epochs) and B4 (a no-padding, shallower-Strassen twin that trades 5 % of score for 1.6× the headroom). For comparison, on a replay of our worst wall day 7 of the top-10 public entries would exceed 120 s by our estimate from their published wall times; by the same table both nominees tolerate a larger uniform slowdown than any of them (A5s 1.33×, B4 1.42×, versus ≤ 1.22×) — with the caveat that ours is set by the overhead cliff and theirs by the wall cliff, which are not the same quantity, and their overhead is not visible to us (DP §5, leaders table). We also rehearsed each nominee on 100–300 fresh seeds under the grader's memory limit (0 errors) and across the smoke shapes (256×4 to 2048×4; one known failure for weight scales ×0.5/×2 at 1024×16, outside the Phase 2 suite but noted in §8).

We think this is a method, not a trick: it turns "will it survive the rerun?" into two measured numbers with a stated rule, and it changed the decision.

---

## 7. What we would do next

In the order the measurements dictate:

1. **κ3 completeness first.** The coupling direction (§5.2) says κ4 readers cannot out-run the κ3 deficit they inherit. The cheapest route we know to a more complete κ3 at equal FLOPs is a better *band*: the readout, not the transport, dominates its cost (0.058 % per column), so a readout that forms D21 from the band without expanding it is the target.
2. **Young-κ4 coefficient refit.** The tree readers' inputs are already exact; their coefficients and coverage are worth +0.10–0.16 R² at layers 7–11 (DP §4.8) at zero FLOPs.
3. **Spend the unused labels.** About 3,300 labelled networks in the public corpus are still unseen by our corrector; the data law says ≈ −0.4 % per 2,000.
4. **The sphere formulation**, done right: measured 17.5 / 12 / 9 % at n = 64 / 128 / 256, shrinking roughly like 1/√n (so ≈ 4–6 % extrapolated to n = 1,024); worth one careful attempt on the real chain.

None of these changes the knee by itself. What would is a κ3 carrier with a different cost structure — one whose readout is linear in the band rank without re-expanding — and we did not find it in the time we had.

---

## 8. Reproducibility, code, provenance

**Code and licence.** https://github.com/samsundar989/arc-whest (MIT). The graded source is `experiments/2026-10-21_L7/release_a5s_soupbc368/estimator.py` (sha256 180e488b…; identical to `experiments/2026-10-21_L7/src/cand_a5s_soupbc368_180e488b.py`), 4,264 lines, depending only on `numpy`, `flopscope` 0.12.3 and the standard library. The repository also contains: `tools/bill_meter.py` (reproduces the per-network bill, 293,790,066,123 F / 20,991 ops, to the FLOP); `tools/fresh/` (the FRESH-A paired panel against a stored prediction set); `tools/verify_nominee.py` (determinism, memory, 100-seed sweep); the sandbox scripts and JSON results behind §5.2 and Appendix B (`sandbox/`); the full development log (`RESULTS.md`, 1,600+ commits of history); `docs/writeup_datapack.md` (every number in this document with its source file and line); `docs/WRITEUP_LEDGER_2026-10-21.md` (the full ledger, 60+ rows); `docs/PROVENANCE_504aldo.md`; and the per-experiment `RESULT.md` files cited in the data pack.

**To reproduce the headline.** `python tools/bill_meter.py experiments/2026-10-21_L7/release_a5s_soupbc368/estimator.py` prints the bill; running the estimator on VAL networks 900/901 reproduces the stored predictions to 7 × 10⁻⁶ max absolute difference (the cross-BLAS floor; parity was checked on those two networks, DP §1.2), and `whest evaluate` on the public panel reproduces the graded score. Training data: `hf://datasets/keenanpepper/arc-whestbench-p2-d8b-corpus-14k` (rev cc3405a9); the shards used by each refit are listed in DP §4.3.

**Provenance (what is re-implemented, what is ours).** 504aldo published V29 on 9–10 September 2026 (forum post 9 Sep; MIT repository, commit 18c17e2d of 10 Sep). We read it and re-implemented, in our own code: P1 the λ-law κ4 regeneration (our own label-free λ fit on 24 networks; their λ table not taken), P2 the 7→4 leg identity (the reconstruction identity A_D = M·d(s) + 3·A_L·d(e) + Z·Lᵀ is ours and was committed before the clone; the (2,1) regrouping and range finder we matched deliberately; they ship rank 32, we ship rank 4), P4 the cost-anatomy idea list (the batched flat-ops planner, sks trimming, padding and λ-pricing are ours), P5 the age-gated basis (ours is a single riding band, never re-rotated). P3 (D21→birth feedback) is implemented and off. Young-tree κ4 reader physics: the 18249 team (Artyom_Mazur / Kuznetsov; write-up #333534, which also has a rank-448 old-κ3 basis that "rides the transport"). Two-term Q31 law: kai (forum 18246). Shared old-κ3 bases: also mliston (18287) and dogus_ozel (18281). Residual-cliff management by depth caps or round-trip counts: pranay212 (18273), suliman_tadros (18289). Monte-Carlo oracle maps: dogus_ozel (18281), yeongjunyoo (18284), michael_anderson7 (18272), megan_park (18282). Per-operation cost behaviour of flopscope: issue #321 (Susupro) and #324–#327 (megan_park); our measurements are on 0.12.3. Cumulant-propagation frame and the power-cumulant trick: ARC's reference. Dead-unit pruning, Strassen on metered products and learned correctors are in common use on the forum. Everything else — the lineage carrier's spelling, the riding-band variant and its measurements, tp2, the nested readout, the joint-objective planner and the overhead law, the corrector design and data law, the exact-label anatomy and the floor tests, the coupling direction, the rerun rule — is ours, with the numbers above as the evidence. We read the public write-ups posted up to 27 October; if an earlier source for any of these exists, we will credit it in a revision.

**LLM use.** As stated in §0: planning, analysis and drafting with Claude (Cowork, Fable 5.1); code by Claude Code (Opus 5) and Samuel; every claim gated by pre-registered prediction, zero control, paired panels and graded canaries; numbers independently re-checked against source files.

**Known gaps (we did not measure these; do not read them as claims).** No derivation of the hub-star floor as a theorem; no Edgeworth-readout ablation on the real chain; no isolated λ-gain or Q31 ablation on the shipped build; no corrector-off ablation of the shipped chain; no single "with vs without KAS" read; the smoke failure at weight scales ×0.5/×2 (an index guard in the young-tree reader) is unfixed.

---

## Appendix A — The formulas (plain statements; derivations in `F_math.md`)

**A.1 Readout.** For z ~ nearly Gaussian with mean μ, std σ, α = μ/σ, third cumulant κ3 and fourth κ4:
E[ReLU(z)] = σφ(α) + μΦ(α) + c₃(α)·κ3/σ² + c₄(α)·κ4/σ³ + c₃₃(α)·κ3²/σ⁵ + …, with c₃ = −αφ(α)/6, c₄ = (α²−1)φ(α)/24, c₃₃ = (α⁴−6α²+3)φ(α)/72 (Stein's identity with f″ = σδ(x+α)). Sensitivities: ∂E/∂μ = Φ(α) (the largest — this is why inherited mean error dominates), ∂E/∂κ3 = c₃/σ², ∂E/∂κ4 = c₄/σ³.

**A.2 Born κ3 of h = ReLU(z).** With A_p(i) = E[h_i He_p(ẑ_i)] (A₁ = σΦ, A₂ = σφ, A₃ = −σαφ) and correlations ρ:
κ3(h)_ijk (distinct) = A₂(i)A₁(j)A₁(k)ρ_ij ρ_ik + 2 perms (the **hub-star**, O(ρ²)) + A₂A₂A₂ ρ_ij ρ_jk ρ_ki (triangle, O(ρ³)) + order-3 stars; κ3(h)_iik = B₁(i)A₁(k)ρ_ik + O(ρ²); κ3(h)_iii closed form. Every star diagram contracts with one matrix product per edge class; every cycle (triangle) is an n⁴ trace — not contractible. Verified against the brute-force tensor at n = 64 to MC noise.

**A.3 Lineages.** Hub j born at layer s: weight w_j = σ_jφ(α_j), legs u_j = W_{s+1}e_j (Jacobian column), v_j = W_{s+1}Φ_s C_s e_j/σ_j (cross-covariance column); both transported by G_l = W_{l+1}diag(Φ_l) each layer (gate linearisation, exact to O(1/√n) for distinct-index entries). Readouts: D3_i = 3Σ_j w_j u_ij v_ij²; D21_ik = Σ_j w_j(2u_ij v_ij v_kj + u_kj v_ij²), i.e. 2(J∘X)Xᵀ + (X∘X)Jᵀ on the undecided rows × columns. Forward lineages = backward adjoint = full n³ tensor tower (max |Δ| 4 × 10⁻¹⁶, n = 48, 6 layers).

**A.4 The band.** At age 7 a lineage's legs are projected on a fixed rank-288 basis Q (shared by all old lineages); thereafter only QᵀG_l… is transported and the old D3/D21 are read from the band coefficients. Cost per column ≈ 0.058 % of the bill, readout-dominated.

**A.5 κ4 laws.** Diagonal regeneration: D4_i ≈ λ_l × (covariance-built term), λ_l fitted per layer label-free; the sandbox identifies it with the emergent scale mode κ4(h)_iijj ⊃ 2X2_i T2_j κ3(z)_ijj (fitted v_eff ≈ 0.006–0.0095 ≫ the input radius 1/2n). (3,1) slice: kai's two-term law. Transport of κ4 pair entries uses T2_iT2_j, not g_i²g_j² (the gate-linearised factor is wrong by ≈ 2 per layer).

**A.6 Young trees.** When κ3(z) meets a kink, κ4(h) inherits terms linear in κ3(z) with repeated indices; the (2,1,1) term (tp2) is read at age 1 as a reweighting of the Dx the κ3 chain already forms; ages 2–3 by the nested pullback Dx₃ = ls[S3]·legsᵀ_{post(L−3)}, Dx₂ = Dx₃·B_{L−2} + ls[n2]·legsᵀ_{post(L−2)} — one product per age-layer.

**A.7 Planner.** Each product is scheduled to minimise F + λ_op·calls over Strassen–Winograd depth (≤ 6; B4 uses ≤ 5), leaf (8), padding/trim and batching choices; λ_op = 3.5 × 10⁶ F per call in #337601. The grader's overhead per call was measured at 13.5–15.9 µs/op (October); the 0.4 s cliff therefore sits at ≈ 25,000–29,600 calls, and the ≥ 10 % headroom rule at ≤ 20,889.

## Appendix B — What we tried and rejected (each with its number; DP §3, §4, §6)

| hypothesis | prediction | result | verdict |
|------------------------------|------------|----------------------------|--------|
| a transported-state corrector can remove inherited error (X1) | −4 % | −0.84 % at +1.29 % F; null after refit | stop |
| the D21 slice error is pair-local (L4) | P(R² ≥ 0.3) ≈ 0.10 | R² ≤ 0.26; scale-only unrefit +35 % | closed |
| zero-FLOP covariance faces from exact labels (J) | small | ρ·μμᵀ face R² 0.4–0.6 unweighted, < 1 % weighted; learned ρ ≤ 0.01× table | closed |
| rank-k covariance-error state (X6) | −2.5 % (80 %: −0.5…−7) | oracle 88–99 % at n = 128 → 20–50 % at 1024 | closed |
| SSC / scale-mixture κ4 faces (X4) | −2.5 % (80 %: −0.5…−6); ceiling read −0.2 % | var² face +2.3 % weighted; (2,2) face v = 0 | closed |
| sphere formulation on the real chain (X3) | −2 % (80 %: +1…−5) | +380 % (spec error) | unresolved |
| pooled importance-capped κ3 carrier (K) | ≥ 20 % better? | 5–10× worse | closed |
| birth-layer entry schedules for the band (N) | better frontier | = per-layer ages in disguise | closed |
| fewer ages / more rank (M, L4c) | −2 % score | +1.1 … +10 % score on the real meter | closed |
| hub threshold 3.0 | removes sandbox floor | null on the real chain (+0.01 %) | closed |
| drop the 25 % least important old columns (sandbox) | cheap | weighted D3 error ×1.9–2.1 at n = 128/256 | closed |
| cut output rows and refit (L3b) | elasticity < 1 | +5.8 % error for −1.6 % F (3.7) | closed |
| D21 column cut (L4d) | −6 % F at +1 % raw if columns were full (P ≈ 0.4) | columns already pruned (K1 × D); a two-sided cut is first-order error | closed |
| rank 320 + fresh epoch (J3) | −0.2 % score | predicted +0.15 %; not run | closed |
| joint content arm J | −3.5 % raw, +1.9 % score | −3.90 % raw (99/100), +2.88 % score | no-go (the knee) |
| sketching any expensive contraction | — | relative variance ≈ n/k | dead |
| exact adjoint for old ages | ≤ 12 % of remaining | ≈ 0 on six ages + band | not needed |

## Appendix C — Source map

Every number cited as *DP §x.y* is in `docs/writeup_datapack.md`, which gives the originating file and line (experiment `RESULT.md`, `RESULTS.md`, graded report JSON, or sandbox report). Graded numbers are from the AIcrowd graded reports (`experiments/run_reports/<id>_graded_report.json`); public submission IDs are given throughout so each can be checked on the leaderboard.
