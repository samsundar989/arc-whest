# Strategist response — 2026-09-23: after the single-leg negative

*Answers the worker's Sep 23 stop report and RESULTS.md §"Single-leg kill tests" and §"Fallback". Supersedes `STRATEGIST_RESPONSE_2026-09-22.md` §2–§5; PLAN rev 12.2 is §6 below. Labels: **[M]** measured, **[I]** inference, **[?]** unknown.*

## 0. Where we are, plainly

Best graded: v55 #331967 = **5.320e-9** (raw 2.461e-8, 0.2162 B), ≈ rank 55–58, level with the public MIT baseline's clone pack. Leaders 2.10–2.50e-9 at 0.126–0.146 with raw 1.60–2.0e-8. Since Sep 21: 8.75 → 5.32e-9 (1.65×), 8 graded rows, one instrument that predicts the server to <1%. That is real, and it is also the end of what the current design gives: the remaining measured levers on this chain are single-digit percent each.

The single-leg hypothesis was mine and it was wrong by two orders of magnitude; the worker's probe framework settled it in a day, which is exactly what it is for. Three structural hypotheses for "old-source content at zero cost" have now been falsified by measurement: compressing it below rank ~300 (Sep 22), regenerating it memoryless (#329896 and T3), and reducing it to its diagonal-born part (T2). Anyone reading this should weight my next structural guess accordingly, and I have put the falsifiable part of it first (§3).

## 1. What the evidence now constrains about the leaders [M unless marked]

1. **Their old-source content is carried, and carried cheaply.** 504aldo's bill minus his old tier is 0.150 B; leaders run 0.126–0.146 with *better* raw than his f64 ceiling (1.94e-8). T1/T2/K prove old content cannot be dropped, diagonalised or regenerated (15–70× losses). So they carry it, at ≤ ~20 units total where ours costs 48 units for 7.7% loss.
2. **Their cost is data-dependent.** Per-MLP FLOPs vary ±6% for suliman_tadros (#330876) and J2W (#331502) [M, telemetry]; ours are bit-identical across nets. A deterministic exact chain has constant cost. Cost that varies per network means **tolerance-driven adaptive truncation** — they compress *something*, choose the rank per network, and lose ≤ a few percent. [I]
3. **What they compress therefore has a fast-decaying spectrum.** Our legs do not: the loss-vs-rank curve for the shared basis (rank 384/320/256/192/128/64 → 0/5/20/64/190/570%) is a smooth r^-3 law, i.e. a slow spectral tail, which is what products of 5–10 random matrices give. Objects in this problem that *are* low-rank: the P2 residual (rank 4, +0.9%), the D21 feedback (rank 16, no loss per 504aldo), and — by the same logic — anything born from a contraction rather than a transport. [I]
4. **Their accuracy edge over 504aldo (1.94 → 1.64e-8) is consistent with consumer-level work**: P3 feedback (−17% raw uncompressed here, −6% compressed), per-network adaptive λ (in v29, never tried on our chain), exact bivariate closures, joint refits. Each is a few percent; together ~1.2–1.3×. [I]
5. **Wall is a binding constraint for everyone and the server got 16 s noisier today** [M]: identical v50 bytes graded worst 87.7 s at 03:00 UTC and 104.1 s later; v55 worst 114 s twice with uncorrelated slow nets. Leaders sit at 100–117 s with `TIME_EXHAUSTED` failures on sealed nets. A failed net scores ~0.9.

## 2. Verdict

The family is right; the design is at 504aldo parity; the winning mechanism (item 1–3) is **not identified**. I do not have a mechanism I can show reaches 0.13 B at ≤5% loss, and I am not going to pretend the next guess is more than a guess. What I can give you is: (a) the one compression hypothesis that is untested at 1024 and has a principled reason to have a steeper spectrum, with a 1-day kill test; (b) the measured-but-unshipped accuracy levers that take us to ~4.3–4.6e-9 regardless; (c) a designation that survives a noisy server; (d) a write-up that is worth money now because of the probe framework. Odds, honestly: top-25 90%, top-15 55%, top-10 20%, top-3 ≤5%.

## 3. The one structural probe worth a day: sensitivity-weighted old-tier basis (N1, clean)

The shared basis `Q` is chosen by a consumption-weighted Gram of the legs — it keeps what is *large*. The score depends only on the final mean; the right basis keeps what the final mean is *sensitive to*. Balanced truncation in model reduction gives steeper spectra than energy truncation precisely when observability and controllability are misaligned, and here they are: the legs' energy is spread over ~n directions (Marchenko–Pastur-like), the final mean is an n-vector reached through ~10 more transports whose Jacobians contract most directions. N1 was "parked" on Sep 17 after a merged-core test at n=96; it has never been run as a basis test at 1024. [I; kill test below]

**Probe P-N1 (cupy probe framework, DEV 100 or 24):**
1. On v55, at the compression layer, compute the sensitivity of the final mean to the old-tier state by finite differences through the *actual* remaining chain: perturb `Q`'s coefficients along k = 64 random directions, read Δμ_L, form the k × k sensitivity Gram; or, cheaper and exact enough: run the chain once, then for each of the 256 basis vectors of a *rank-512 energy basis* measure ‖Δμ_L‖ from zeroing it (512 forward continuations of the tail on one net — ≈ 512 × 0.5 s on cupy). Rank the 512 directions by sensitivity and by energy; report the two spectra.
2. Keep the top-256 by *sensitivity*, run DEV, compare with the energy-256 arm (v55) and energy-384.
Predictions to record: strategist — sensitivity-256 within 3% of energy-384 (i.e. ≥15% raw better than v55 at equal cost); worker — record own. Kill: if the sensitivity and energy orderings agree (Spearman > 0.9) or the DEV gain is < 5%, N1 is dead for good and the compression lane is closed in this competition.
3. If it passes: make the basis adaptive (rank by tolerance on the sensitivity spectrum, per net — this is what item 2 in §1 predicts the leaders do), then push the age gate down (age 3–4) at the same loss. This is the only lane I can see to 0.15–0.17 B at unshared accuracy.

Second, cheaper (half a day): **tiered rank schedule** — the old tier's loss is dominated by its youngest members. Re-compress `Q` at ages 8 and 11 to 192 and 128 (the H3 note says this needs an explicit re-compression event, not a rank request). Expected −3–5% cost at ≤2% loss; worth a row, not a bet.

## 4. Measured, unshipped accuracy levers — ship as canaries, one variable at a time (Sep 24–27)

| lever | evidence | expected | cost |
|---|---|---|---|
| P3 D21-feedback at rank 16 (and 8) on v55 | −17% raw uncompressed (v42), −6% on C10 at rank 32 for +12% cost; 504aldo: rank 16 lossless, +8% efficiency | −4–8% score | +6% / +3% FLOPs, memory check |
| per-network adaptive λ (v29's `λ·clip(r/REF,0.5,2)^β`, own code, REF from FIT) | in v29; never tried here; v29 native beat our uncompressed arm on P12 | 0–8% raw [?] | ~0 |
| joint refit (λ table × P3 gain × θ) on the P3 chain | D2: 2.7% on v53; the fit has never seen P3 | 1–3% | 0 |
| κ4 on M for the exact tier (T4 as specified, on v55) | untested; 504aldo's κ4 memory legs 1.1% | 0–3% [?] | small |
| closure: exact bivariate ReLU moments where the PAIR table truncates Hermite (Kuang–Lin / Price) | closure is worth 18%; its truncation error is unmeasured | 0–5% [?] | elementwise |
| tiered old-tier ranks (§3, second) | rank curve | −3–5% cost | — |

Cumulative if the middle cases land: raw ≈ 2.1–2.2e-8 at ≈0.22 B → **4.3–4.6e-9, rank ~13–18**. Each is one DEV run and one canary; order them by expected gain per day: P3-r16 first (today), adaptive λ second, refit third.

## 5. Designation and wall — decide now, not Oct 11

The private rerun is on the same noisy grader. Policy: **designate A = best score whose worst wall ≤ 105 s on a noisy-day run; B = a wall-safe twin with worst ≤ 95 s on a noisy day.** Neither v55 (114) nor v50 (104) is B today. Build the twin now: same chain, fewer/larger ops — Strassen depth 4 on the two D21 products only (keeps the transport products at depth 5), which trades ~+4% FLOPs for a large wall cut (v31 at 13k ops and 0.30 B ran 68 s; wall follows op count and leaf size, not FLOPs). Measure on one canary. Then every later canary ships as an (A, B) pair when it beats the standing pair by ≥3%. Re-run designates once on a quiet day and once on a noisy day before Oct 15. Never trade B's margin for score.

## 6. PLAN rev 12.2 (replaces the Sep 24–Oct 10 hole)

- **Sep 24:** P3-r16/r8 rows and canary; wall-safe twin built and canaried; refit the wall model with the noisy-day data (use worst-of-two-runs as the statistic).
- **Sep 25:** P-N1 probe (§3) — the only bet; tiered-rank re-compression if N1 dies early.
- **Sep 26–27:** adaptive λ; joint refit on the P3 chain; canaries.
- **Sep 28–Oct 3:** if N1 passed: adaptive-rank basis, lower age gate, canaries toward ≤0.17 B. If not: κ4-on-M and closure items, then stop adding.
- **Oct 4–10:** write-up drafting starts (see §7), canaries only for measured ≥3% gains, two graded runs of each designate.
- **Oct 11–16:** D7 as amended in §5. **Oct 17–24:** D8.
- Standing rules, §5 gates (ops ≤ 26,000, residual hard gate), and the probe-before-inference rule stand. Commit `docs/PLAN.md`, `HANDOFF_2026-09-22.md`, `STRATEGIST_RESPONSE_2026-09-22.md` and this file together.

## 7. The write-up is now a co-equal objective, and we have the material

Nobody else on the board has: an exact unshared reference at 1024×16, a paired probe framework that evaluates 100 nets in minutes with the panel mean at 3e-5 of the grader, and a complete term-by-term/age-by-age ablation of a factorized K3 chain — what each block, slice, rank, age, closure term and fitted profile is worth, including three clean negatives (fitting saturates at 2.7%, richer term sets lose on score, old content is neither compressible below n/4 nor regenerable nor diagonalisable). "Anatomy of a factorized cumulant chain at 1024×16" is an honest contribution-prize entry whether or not §3 lands, and §3 either way adds a result. Start the claims ledger from RESULTS.md now (1 h/day from Oct 4), full drafting Oct 11–24, with explicit LLM-authorship and 504aldo provenance.

## 8. For Samuel

Two decisions are yours: (1) accept the (A, B) designation policy in §5, including shipping pairs; (2) accept that from Oct 4 the write-up gets real time, because the score odds above are what they are. Everything else the worker can run under the standing rules. If §3 fails, expect a final standing around 4.3–4.8e-9 (rank 13–20) and a strong write-up; if it passes, top-10 is back on the table.
