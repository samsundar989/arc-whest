# Strategist response — 2026-09-08: the floor is a *member* of the class, not the class; carry κ₄ in flight; Slice 3

*Answers `docs/STRATEGIST_PACKET_2026-09-08.md`. Supersedes the Sep 7 Slice 2 handoff where they differ. Serves as the Slice 3 handoff (§5).*

## 0. Slice 2 verdict: accepted

Three graded submissions, all 0/100 failures: k3_full 2.9467e-8 (0.78 B, worst wall 9.5 s), Strassen L=4 1.836e-8 (worst wall 114.7 s), Strassen L=3 2.031e-8 (0.537 B, worst wall 61.9 s). 8.5e-8 → 2.03e-8 nominee-grade in two days, ~#47 → ~#27. The two errata (row-0-as-aggregate; a gate on the paired adjusted score) were caught and corrected; the record is trustworthy. Two measurement laws are now graded facts: residual is per-operation at 10.7–11.2 µs/op across 2.7k–15.4k ops (the local residual instrument is retired as a gate), and graded wall on identical bytes varies up to 1.6× at the maximum. Cost engineering via Strassen is wall-bound: L=4's 114.7 s worst row against a 120 s cap with a 1.6× contention factor is not nominee-grade. **Agreed: nominee A = L=3.**

One judgement I overrule: §3d/§3b were merged and priced as a *readout* lever using the K2-era oracle (94.61/4.01/0.46%). That oracle measured a K2 closure's readout, and the packet itself says the unpriced thing is κ₄ *in flight*. The lane is not "provisional NO-GO"; it is "not yet measured", and §1 below says why it is the most likely explanation of the leaders' 2×.

## 1. The answer to the question: is 1.9e-8 a property of cumulant-truncation estimators as a class?

**No. It is the floor of one specific member of the class, and we are running a different, worse member.** The argument:

1. **Our chain is ARC's `kind=SIMPLE` K3.** `estimator_k3_full.py` line 7 says so; κ₄ enters only as one global scalar `c4` (lines 690, 697). ARC's paper and code define two refinements we have never run at the live shape: **`AUGMENT`** — "track not merely the full trace of the (K+1)-th cumulant but the entire tensor except its traceless component" (§4.2, App. S.4.2), i.e. for K=3 the κ₄ trace matrix T_ij = Σ_k κ_ijkk (and its scalar) carried *in flight* through every linear and ReLU step — and **power cumulants** (Hermite expansions of powers of the activation, for repeated-index accuracy). The paper's Fig. 2 shows factorized-augmented beating factorized at K=3; the exact factor at 1024×16 is unmeasured by anyone we can see, including us. The Sep 5 and Sep 7 handoffs both asked for the AUGMENT sign; it has not been run.
2. **The physics says the missing term is κ₄ in flight, at exactly a ~2× scale.** ARC's conjecture MSE_K ≈ c_K (L/n)^K: the leading error of a K3 chain is the effect of the omitted κ₄ on the *propagated* (μ, Σ, κ₃) state — the state drift the packet correctly identifies. A readout-only κ₄ fixes one layer's contribution; an in-flight κ₄ fixes all sixteen. The 0.46% "κ₄ map" number is the readout share on a K2 chain; it says nothing about in-flight κ₄ on a K3 chain, and the packet already concedes this.
3. **The board fingerprint is "same algorithm, different constants".** Six teams at 1.87–2.13e-8 across 0.18–0.79 B is what you get when everyone runs the same public algorithm (ARC's factorized-augmented K3, or equivalent) with small private tweaks — not what you get from six independent inventions. Our SIMPLE K3 at 3.9e-8 is 2.0× above that cluster; augmentation is a 2×-class refinement; the arithmetic closes.
4. **The class itself has far more headroom.** By the same conjecture a full K4 chain sits near c₄(L/n)⁴ ≈ 5e-10 (100× over budget, but it bounds the class). So the 1.9e-8 floor is not where cumulant truncation stops; it is where *trace-only κ₄* stops. Anything that carries more of κ₄'s structure than its trace — Puffi's SSC (p ≤ 4 response modes B_a, coupling Λ: a low-rank model of the *traceless* part), per-neuron diagonal κ₄ in flight, power cumulants, or a learned in-flight corrector on chain features — is a candidate to go *below* the leaders. Puffi's −77% vs K3 at 256×32 (where K3 is far worse) should be read as "SSC carries more of κ₄ than augmented does", not as a number to transplant.

What I cannot claim: that the leaders are literally AUGMENT. hyojun_kwon's Phase-1 write-up (Gram–Charlier with Mehler co-skewness + fitted per-layer mean corrections) and Puffi's A716 prefix (marginal κ₃/κ₄ + a **10-channel learned in-flight adjustment per layer, 90 coefficients fitted to layer-8 residuals**) show two other routes to "κ₄-class information in flight", and Hilton's warm-up remark ("factorized 3rd-cumulant propagation + learned networks consuming cumulant estimates as features") is a third. All three are in-flight; none is a readout scalar. The cheapest falsifier is the same for all of them.

## 2. What reduces the drift (packet Q2), ranked by evidence and by cost to falsify

| # | term | why it acts on propagated drift | falsifier on 24 nets | prediction (raw, from 3.9e-8) |
|---|---|---|---|---|
| D1 | **ARC `AUGMENT` at k_max=3, factor=True** (κ₄ non-traceless part in flight) | the omitted κ₄ enters every layer's Wick update of (μ, Σ, κ₃) | run ARC upstream torch with `kind=AUGMENT` at 1024×16; compare to SIMPLE on the 24-net 2^26 set | **1.8–2.5e-8** (i.e. the leaders' floor). If ≥3.5e-8, augmentation is not the term |
| D2 | **O4 oracle: true κ₄ in flight** (from the independent sample stream): (a) trace part only, (b) per-neuron diagonal, (c) every κ₄ slice the Wick step reads | brackets every κ₄-in-flight scheme, including SSC | inject at each layer in the torch K3 chain; `O4/K3` per variant | (a) ≈ D1; (c) ≥ 3× — the ceiling of the κ₄ family. `O4(c)/K3 < 1.5×` kills the whole lane |
| D3 | **Power cumulants** (ARC §4.1 refinement) | repeated-index (self-correlation) accuracy in the ReLU step — a state-side term | ARC upstream flag, same 24 nets | 1.1–1.5× |
| D4 | **SSC-style low-rank traceless κ₄** (response modes) on top of D1 | carries what AUGMENT discards | build only if D2(c) ≫ D2(a) | up to D2(c) |
| D5 | **Learned in-flight corrector on chain features** (Puffi A716's 10-channel/90-coefficient per-layer adjustment; Hilton) | adds information the closure lacks; per-layer targets from the sample stream | fit on 96 P01 nets, held-out 24; DAgger rollout; the WP10 amplification law gives the entry price — recompute it on the K3(+AUGMENT) chain, it is not the K2 number | 1.2–2× on top of D1 |
| D6 | better off-diagonal covariance map | the K3 chain already carries the Price order-2 term; the residual off-diagonal error *is* the κ₄ cross term | subsumed by D2 | — |

Do not build anything before D1 and D2 have numbers. They are two GPU jobs, no seed block, ≤2 days, and they decide the rest of the competition.

## 3. The cost half (packet Q3)

Stop Strassen depth work. Everything on the board is wall-bound (the leaders run 70–119 s), so the real currency is **billed FLOPs saved per wall-second**, and Strassen's small leaves spend wall at 17 GFLOP/s. Two accuracy-neutral items only, and only after D1/D2:

- **Hybrid Strassen:** apply recursion only to matmuls above a size threshold (keep leaves ≥ 256 so arithmetic intensity stays > ~60 GFLOP/s); measure FLOPs vs worst wall at L ∈ {3, 4}. Target: L=4's FLOPs (0.46 B) at ≤ 75 s worst wall (1.6× margin). If it cannot, L=3 stays.
- **Dead/cold-column pruning with exact compensation** (≈0.8×, still unpriced). Accuracy-neutral by construction if the compensation is exact; price it.

Whatever AUGMENT costs in FLOPs (expect +5–15%: one extra n×n object per layer through W and the Wick step) is paid gladly.

## 4. Nominees (packet Q4)

A = L=3 (`330327`, 2.031e-8, 1.94× wall margin). **B = L=4 (`330290`, 1.836e-8) if and only if Rules v14 §6 scores each designated submission independently and ranks the team by its best** — then B is free upside and its failure risk cannot hurt A. Read the captured rules text (`experiments/2026-08-28_p2_live/*rules_v14*`) and quote the sentence; if the rule averages, combines, or is ambiguous, B = k3_full (`330212`, 2.95e-8, 9.5 s worst wall, the safest thing we own). Re-designate as D1/D2 land; keep every nominee at ≥3 clean graded runs and ≥1.6× wall margin.

## 5. Slice 3 (Sep 9–19) — execution order

1. **D1 and D2 in torch, first, in parallel** (GPU queue; the 24-net 2^26 set; predictions recorded before the runs; `gt_noise` printed). Deliver both numbers within 48 h. Include D3 if the upstream flag is a one-line change.
2. **If D1 ≤ 2.5e-8:** port AUGMENT into the fnp chain (`estimator_k3_aug.py`), same certification (twin arm, ops ≤ 24k, projected residual ≤ 0.3 s, Linux memory gate on the sha, 1.6× wall margin), canary once (Samuel's approval). Expected ≈ 2e-8 raw × 0.54 = **1.1e-8, ~#15**. Then pruning/hybrid-Strassen → 0.43 B → ~8.6e-9 (#11).
3. **If D2(c)/D2(a) ≥ 1.5:** build the traceless-κ₄ carrier (SSC-style p ≤ 4 modes, Λ ridge-fit offline on ≥160 mints; or per-neuron diagonal κ₄ in flight, whichever D2(b)/(c) favours). Target raw ≤ 1.3e-8 → **≤ 7e-9 (top-8)**.
4. **D5 corrector**, features logged from step 2 on, trained once D1's chain is fixed. Target 1.2–2× on top.
5. **If D1 ≥ 3.5e-8 and D2(c) < 1.5×:** the κ₄ lane is dead at this shape; the remaining information-adding arm is D5 alone, and the write-up's spine becomes the measured negative ("the state drift of a K3 chain at L/n = 1/64 is not κ₄-driven"). Report immediately; do not build anything else first.
6. Nominations updated at each graded improvement; write-up near-final by Oct 3; team freeze Oct 2.

Hand back by Sep 19: D1/D2/D3 numbers with CIs; the AUGMENT port's cost, ops, wall and graded row; the pruning price; the rules-§6 quote and the designated pair; the corrector's held-out number if reached.

## 6. Where this leaves the win condition

Board #1 is 3.80e-9. At 0.54 B we need raw ≤ 7e-9; at 0.43 B, ≤ 8.8e-9; at 0.35 B, ≤ 1.1e-8. The leaders' floor is 1.9e-8, so **matching their algorithm is not enough at any cost we can reach — #1 requires going below their floor by ~2×**, which is exactly what D2(c)−D2(a) measures the room for. Top-8 is reachable with D1 alone; top-3 needs D1 + one of D3/D4/D5; #1 needs two of them or a cost breakthrough we do not currently see. That is the honest shape of the remaining six weeks, and the $20k algorithmic prize is served by the same measurements either way.
