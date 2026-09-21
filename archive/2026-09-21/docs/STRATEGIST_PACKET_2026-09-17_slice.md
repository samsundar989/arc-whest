# Strategist packet — 17 September 2026 (after the closure / protected-feed / shared-factor slice)

> **Reviewed; historical packet.** The active decision is [STRATEGIST_RESPONSE_2026-09-17_slice2.md](STRATEGIST_RESPONSE_2026-09-17_slice2.md). It corrects the scope of the closure tests, the query-error identity, uncertainty/noise in the rank choice, and the mixed public/P11 arithmetic in section 3. The original packet below is preserved as the worker's record, not the current execution plan.

*Self-contained: assumes no repository access and no chat history. Labels: **[M]** measured this
slice with raw artifacts and pre-registered verdicts; **[R]** reported by others; **[I]** inference.
Answers `CLAUDE_CODE_HANDOFF_2026-09-17.md`. The major-release goal was **not met**; no submission was
requested or made; nothing was designated.*

## 1. Situation

ARC White-Box Estimation Challenge, Phase 2: predict every neuron's post-ReLU mean for random He-init
ReLU MLPs, 1024 wide × 16 deep, input x ~ N(0, I). Per-network score = final-layer MSE ×
max(0.1, counted FLOPs / 2^41); mean over networks; lower is better. Residual time is a 0.4 s cliff
per MLP (we gate on operation count ≤ 22,000); memory cap 8 GiB.

- Our latest graded file **v31: adjusted 8.752e-9, raw 2.907e-8, C/B 0.301**, 0/100 failures.
- Public MIT chain (team 504aldo) **v29: 5.40e-9 public** (raw 2.13e-8, C/B 0.2526) [R]. Leader ≈2.4e-9 [R].
- Requested next release: **≥2.0× paired adjusted gain over v31 and projected public ≤4.4e-9**.

Our method: moment propagation layer by layer — mean, full covariance, a factored third-cumulant
tower (one "lineage" of rank-structured source factors born per layer, transported by later
weights), a regenerated per-neuron fourth-cumulant core (diag g plus λ·C off-diagonal model), and a
nonlinear "closure" table that maps pre-activation cumulants to post-ReLU cumulants via Gaussian Wick
coefficients w(k,p) = E[∂^k ReLU^p].

All accuracy numbers below are on the same 24 design networks ("P11"; two independent 2^24-sample
truth replicas; noise of the averaged labels ≈2.38e-9), paired, arithmetic adjusted-score difference
with studentised-t 95% intervals, all local-seconds caps off. P11 has now been read ~13 times: it is
design-tier evidence only.

## 2. What this slice established

### Instrument and integration [M]
- Run-receipt name mismatch and a PowerShell-7 module-path hashing failure fixed and tested
  (10 wrapper tests incl. poisoned environments; default preflight green). Independent Skeptic: ACCEPT.
- A canonical audit of the nonlinear program each file *executes*: our v32 runs exactly the upstream
  ARC power-counted cost-≤2 closure (27 oriented terms). Public v17/v29 run a pruned "at most two
  cumulant factors" closure (54 extra terms + 6 κ_iiij rows); every shared coefficient agrees.
  An independently derived closure generator (BUDGET 2/3/4, PAIR) with polynomial, discrete-law and
  realizable-law oracles confirms coefficients against the pinned upstream enumerator.
- A float64 reference step reproduces the live layer to ≤1e-13; live float32 at 1024×16 sits at the
  base program's own float floor (no implementation defect).
- Same-state report on one 1024×16 network: v17 and v32 trajectories are nearly identical
  (C_off ≤0.5%, D3 ≤1.5%, D21 ≤2% apart) except the transported fourth-order core dG (up to 17%) and
  λ. Pre-activations are only mildly non-Gaussian (|skew| ≤0.075, |excess kurtosis| ≤0.056).

### Accuracy mechanisms [M]

| arm (vs v32 unless stated) | adjusted | raw | better |
|---|---:|---:|---:|
| BUDGET 4 closure every layer (κ_iiij = 0) | 0.835× | 0.843× | 0/24 |
| BUDGET 3 (κ_iiij = 0) | 0.781× | 0.783× | 0/24 |
| PAIR (κ_iiij = 0) | 0.866× | 0.872× | 0/24 |
| richer mean at final layer only | 0.999× | 0.999× | 13/24 |
| D22 feed protected from rank-32 compression | 0.997× | 1.000× | 9/24 |
| **BUDGET 3 + κ_iiij = (m/2)λC** vs BUDGET 3 | **1.189×** | 1.190× | 24/24 |
| BUDGET 3 + κ_iiij vs v32 | 0.929× | 0.932× | 7/24 |

Reading: most of the richer closure's harm was a mixed-order inconsistency (the cost-3 κ_iiij block
of our own fourth-order model was omitted); with it restored, the consistent closure is still 7% worse
than the 25-term closure on our chain, while the same families are part of a chain that is 1.29×
more accurate than ours (public v17/v29) [M,R]. The terms are not wrong; they do not help *this* chain.

### Cost mechanisms [M]

| arm | C/B | ops | raw vs v32 | adjusted vs v32 |
|---|---:|---:|---:|---:|
| v32 | 0.3023 | 13,850 | 1 | 1 |
| + P4: Strassen depth 5 + two exact trims | 0.2802 | 17,483 | 1.0002 | 1.079× |
| shared old-lineage coordinates, age 3, rank 256 (depth 4) | 0.2176 | 18,623 | 1.764 | 0.788× |
| … age 2 rank 192 | 0.1789 | 16,710 | 6.51 | 0.26× |
| **age 4 rank 256 lean + P4 depth 5** | **0.2163** | 19,914 | 1.216 | **1.149×** (1.065× over P4 alone) |
| age 4 rank 384 lean + P4 depth 5 | 0.2602 | 19,914 | 1.022 | 1.132× |

Shared coordinates: recent lineages stay exact; older lineages' left factors live in one shared rank-r
basis, transported exactly; the only approximation is a weighted range-finder join when a lineage ages
out. Its full-rank identity is exact. A label-free proxy (squared deviation from the exact chain on
unlabeled networks) predicted truth damage within 25%; join weighting/power passes change damage
≤15%; age and rank dominate. Structural limits: the exact recent lineages set a cost floor, and ranks
≥384 forfeit Strassen levels (rank 512 costs more than no sharing).

## 3. Best supported state

**Cost base:** `v37_a4r256_p4d5` — P11 score 8.746e-9 vs v31's 1.0372e-8 on the same networks ≈ **1.19×**;
naïve public projection ≈7.4e-9. Operations 19,914 (<22,000); peak Windows commit 4.47 GB (Linux
8-GiB enforcement not yet run). Not DECIDE-tested, not mini-run, not certified. **Far from the 2× /
4.4e-9 target**, and per the handoff not offered for submission. The graded v31 and F21 remain the
fallbacks; v32 + P4 depth 5 alone would be ≈1.11× over v31.

To reach 4.4e-9 at C/B ≈0.216 needs raw ≈2.0e-8 — 2.0× better raw than the cost base (4.04e-8), or
1.66× better than v32. Cost alone cannot get there: even at raw parity with v32, 4.4e-9 needs C/B ≈0.13.

## 4. The decision we need

1. **Where does v29's 1.29× raw advantage come from, given near-identical states?** Candidates not yet
   isolated: its transported-core recipe (dG differs up to 17%), its λ profile combined with its full
   pruned closure and κ_iiij rows *as a package on its own trajectory*, its rank-16 weight-column sketch
   and protected feed, and its D21 feedback. Single-mechanism transplants onto our base have all been
   neutral-to-negative, and identical terms change sign between chains — an interaction, not a missing
   term. Proposed discriminator: a *reverse* ablation on v17 itself (replace one of its components at a
   time with ours: closure table, core transport, sketch/rank, feedback) on P11, to find which of ITS
   parts carries the edge. Is this worth a design read, or should we instead fit and freeze our own
   joint closure+core recipe on a fresh DESIGN block?
2. **Cost:** the recent exact lineages are now the floor (~half the tower at age 4). Is a cheaper
   representation of *recent* lineages (e.g., low-rank legs from birth, or thin feedback that replaces
   exact recent contractions) the next cost question, or should cost work stop at ≈0.216 B until the
   accuracy gap is explained?
3. **Release policy:** is a ≈1.19× file worth a canary submission for server evidence (it would need a
   fresh DECIDE panel, public mini, Linux memory and certification first), or should submissions wait
   for a candidate nearer the 2× target? The prize is decided on up to two designated submissions.

## 5. Protocol notes and corrections [M]

- An independent Skeptic reproduced every number; it rejected two over-strong closures ("closure lane
  closed", "shared route closed") that were then retracted and resolved by the follow-up read above.
- Three screens were described as running on a git snapshot while their commands changed into the live
  repository; harmless (scored files never changed) and corrected for later jobs.
- A builder error regenerated two untracked `estimator_k3_v33*.py` files from their unchanged
  deterministic generator; byte identity with the originals is unverified; no evidence depends on them.
- Deadlines [R]: registration/team freeze October 2; submissions October 17 23:59 UTC; write-up October 24.
