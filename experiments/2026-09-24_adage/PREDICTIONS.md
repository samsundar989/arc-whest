# Step 3 (per-lineage adaptive age) — predictions BEFORE the sweep

Written 2026-09-24 ~02:50 UTC, after `estimator_k3_v60_adage.py` passed bit-parity
with v55 at `_ADAPT_TAU = None` and before any arm was evaluated.

## Strategist (response §4)

−2 to −5% score at equal wall [inference, low confidence]. Kill: < 2%.

## Worker (mine)

**−0% to −2%, i.e. I expect this to FAIL its 2% kill.** Three reasons, each from
a measurement already on record:

1. **The age curve is nearly flat at the top.** D5 measured age 4 -> 5 at −10.32%
   score and 5 -> 6 at only −10.91% (a further −0.66%). The marginal value of an
   extra exact layer has almost vanished by age 5. Adaptive age redistributes age
   around 5 — spending more on some lineages and less on others — so its upside is
   bounded by the CURVATURE of that curve, which is small where we sit.
2. **Energy already ranks contribution, which cuts both ways.** N1 measured
   Spearman(energy, downstream sensitivity) = 0.994. That is why energy is the
   right statistic — but it also means v55's fixed age is already folding
   lineages in almost exactly sensitivity order. Adaptivity can only help to the
   extent that the *timing* is wrong per net, not the ordering.
3. **Per-net variation in this chain has been measured small twice.** The
   adaptive-λ probe found r = mean(dG)/mean(var) varies only 1.8–9.0% CV across
   100 FIT nets and the [0.5, 2] clip never bound — recorded as "per-network
   adaptation is not where per-net variation lives in this problem". Lineage
   energies are built from the same weights.

**Specific numeric predictions:**
- Mean age will move roughly linearly with log(tau); I expect to need tau spanning
  ~1e-3 to ~1e-1 to cover mean age 4.0–5.5.
- Per-net FLOPs spread (max/min over 100 nets) **< 4%** — below the leaders'
  observed ±6%, because the energies are tightly distributed across nets.
- Best arm: **score between −0.5% and −2.0%** vs v55, and I expect the sweep's
  best to be at the LOW-age end (cheaper) rather than from smarter allocation.
- **If it does clear 2%, I predict it does so on COST, not accuracy** — i.e. raw
  MSE within +2% of v55 while FLOPs fall 3–6%.

## The gate this lever has that the others did not

Even a passing arm is not shippable on local evidence alone: the decision
materialises a scalar (16 `__bool__` per net, measured 32 ops / 32 FLOPs locally),
against a standing project invariant of ZERO materialisations and a grader whose
arithmetic runs server side (`flopscope_server_comms_wall_s` median 0.78 s on
tr64 #332088). A forced round trip is invisible locally. If an arm clears 2%, the
next step is a canary for the residual reading, NOT a designation.
