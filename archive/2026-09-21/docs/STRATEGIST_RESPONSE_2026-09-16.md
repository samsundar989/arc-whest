# Strategist response — 2026-09-16: answers to the Sep-15 packet, the submit decision, and Slice 7

*Answers `docs/STRATEGIST_PACKET_2026-09-15.md`. Read after `phase2-strategist-response-2026-09-14.md`; this refines its plan, it does not replace its verdict. New evidence tonight: a knob-by-knob ablation of 504aldo's v17 κ4 channel against Monte-Carlo truth at 64×8 and 128×8 (§2), and a fresh board read (Sep 16 ~05:00 UTC). Labels: **[M]** measured tonight or read from a primary source, **[R]** reported by the worker/competitor, **[I]** inference.*

---

## 0. Verdict on the slice, and the submit decision

The slice is good work and the numbers are trustworthy: P1 1.237× on 24/24, P2 cost 0.7128 metered by an independent Skeptic, v31 raw within +0.9% of its dense arm (504aldo's published +0.9% reproduced), 12,665 ops (2.36× residual headroom on the server model), 8 GiB gate 23/23, budget guard now over-predicts, identity suite honest 10/10 with mutation testing. The provenance correction is the right call and the rules question should now go out.

**Yes — submit v31 today, as a non-designated canary, on your approval.** Projected public ≈ 9.3e-9 (raw ≈3.08e-8 at C/B 0.3010), which lands around rank 52 on this morning's board (we have slipped to #67 at 1.60e-8 as more 504aldo clones arrive). Reasons: it is a 1.72× improvement over the standing candidate; it carries the first restructured tower (P2) and the first κ4 regeneration on the grader, and only a graded row gives authoritative residual/wall telemetry for that op structure; it becomes the fallback designate. Conditions before the slot: certification (job with §9 memory present) green, `predict_public.py` run on the finished mini pair with the paired ratio printed, wall projection ×1.6 < 120 s (v31 has 0.71× F21's FLOPs; F21 graded at 86–93 s, so expect ~65–75 s — check the mini's wall anyway), op count ≤ 22k (it is 12,665). Do not designate anything yet; F21 stays the designated fallback until v31 has a clean graded row.

## 1. Q1 — allocation

The decomposition in the packet is correct arithmetic (1.724 = 1.447 × 1.191 to 504aldo) but it frames the wrong opponent. Positions 11–45 on the board are now 504aldo and ~25 clones at 4.2–5.4e-9; the prize positions are 2.5–2.8e-9, i.e. **raw 1.7–1.9e-8 at 0.13–0.16 B**. Against the leader our gap is 1.62× accuracy × 2.25× cost. So: to 504aldo the deficit is mostly accuracy; to the prize it is mostly cost, and the accuracy deficit is a prerequisite either way.

Ranked allocation for the ~two person-weeks:

1. **Close the κ4 half (accuracy), 2–3 days.** Highest EV per day, because tonight's ablation (§2) says the missing 1.4× is almost certainly a *port discrepancy in one n² term*, not a missing table row. Expected: raw 3.08 → ~2.2–2.4e-8 public-scale at unchanged cost → ~7e-9.
2. **P3 (D21 feedback, rank 16), 1 day.** Measured by 504aldo at ×0.92 raw; independent of item 1. → ~6.3e-9.
3. **P4 (Strassen L5 + pooled buffers + block-symmetric C_pre + final trim), 2 days.** Mechanical, ~10% cost. → ~5.7e-9. Skip P5 as published.
4. **N1 instead of P5, 4–5 days (the research bet, and the only route to a prize).** Sensitivity-weighted basis for old lineages + merged old core, subsuming the age-gated basis. Kill test first (§4). If it halves the rank at equal error: 0.30 → ~0.18 B → ~3.5–4e-9 (top 8). If it fails, ship P5 as published (→ ~5e-9) and the score prize is out of reach; the write-up becomes the objective.
5. **Write-up: 1 hour/day of claims-ledger from now, real drafting after Oct 3.** Option (c) as a redirection *now* is wrong: nothing prize-worthy exists yet, and N1 is the thing that would make it so.

Not recommended: finishing P5 verbatim before N1 (it spends the days N1 needs and lands in the clone pack); any further work on the (3,1) table divergence (§2).

## 2. Q2 — where the 1.45× accuracy gap comes from **[M, small-shape proxy]**

I ran 504aldo's own v16b/v17 unmodified (numpy shim, no metering; their env kill-switches `V17_NO_FEED`, `V17_NO_WK431`, `V17_NO_REGEN`, `V17_LAM_SCALE`, plus two 3-line split knobs) against my own MC truth (4e6 inputs, noise floor ≤1% of MSE) on 8 He-init MLPs at 64×8 and 128×8. Their regen channel is gated behind `n==1024 and L==16`; I forced it open. Their shipped λ table is under-scaled off-suite (optimum ≈ ×10 at n=128, ≈ ×22 at n=64; λ_opt ∝ ~n⁻¹·¹), so the decisive rows are at tuned λ.

Multiplicative decomposition of the κ4 gain over the no-regen control (n=128 | n=64):

| component | gain | measured as |
|---|---|---|
| exact transported diagonal core g, λ=0 | ×1.04 | ×1.05 | control → λ=0 |
| diagonal part of the κ4→κ3 feed, λ=0 | ×1.01 | ×1.03 | |
| **λ inside the dG transport — `dG = (W∘W)@(g_prev − var_prev·λ) + var·λ`, i.e. λ·(diag C_pre − (W∘W)var_post) — consumed only through the (4) and (2,2) rows** | **×2.32** | **×3.95** | λ=0 → λ-in-dG-only |
| λ-carrying off-diagonal terms (wk431 + c1_b feed) | ×1.11 | ×1.18 | → full |
| — of which the (3,1) consumer wk431 alone | ×1.00 | ×1.09 | |
| — of which the c1_b feed alone | ≈×1.11 | ≈×1.08 | |

Per-layer MSE shows the λ effect is invisible at layers 2–3 and grows monotonically with depth: it is a *transport* effect. Ordering is identical on all 8 seeds at both shapes.

Reading for our port:

1. **The (3,1) slice is not the missing half.** 0–9% on the proxy; 504aldo's own F68 says "dropping (3,1) production cost nothing". Do not diverge from the frozen table for it. Strike that candidate from the packet's list.
2. **85–90% of the κ4 gain rides on one n² vector term** — the λ·(diag C_pre − (W∘W)·var_post) piece of the transported κ4 diagonal — consumed through the (4) and (2,2) rows that ARC's SIMPLE table already has. Our port has exactly those consumers. If v30/v31 gets 1.17× from λ where v17 gets ~2× at 1024 **[I from F68 + this proxy]**, the first place to look is *that one line*: is `dG` formed with both the `−(W∘W)·var_post·λ` subtraction and the `+var_pre·λ` addition, with the same λ per layer, before it enters the (4)/(2,2) rows and the (4)→(3,) / (2,2)→(2,1) births? A missing subtraction, a `metric` applied once too often (the bug already found once in this port), or the trace/`c4` split double-counting would each cost a factor of order 1.5 without breaking any parity test.
3. **The κ4→κ3 birth feed matters more at 1024 than the proxy shows.** F68 reports dropping the feed on the lean chain at 1024×16 costs 4× (2.23→9.23e-8); on the proxy it is ×1.1. So the second place to look is `feed_already_present.md`'s claim that the frozen table's (4)→(3,) and (2,2)→(2,1) routes *are* their feed with the *same coefficient*. That was argued, not measured.
4. **The baseline of their "1.97×" is inflated.** It is quoted against V16b with an over-fit CORR rider (LB 4.46e-8); against a clean K3 base their gain is ~1.75×. So the unexplained factor is ~1.4×, not 1.6×.
5. Ranked below: P3 feedback (×0.92, real, separate); f32 (504aldo: +0.004%, no); structural κ4 beyond diag+λC_off (their 9 extra modes: 1.1%, no); λ *value* (the sweep's flat optimum at 0.79 already excludes a scale error — but a global scale cannot detect a wrong *depth profile*, see D1).

**Discriminators, cheapest first (all label-free or DESIGN-tier, none needs a new block):**

- **D1 (2 hours):** run v31 on the 24 P01 design nets with 504aldo's `LAM` table substituted for ours — a *diagnostic arm, never shipped*. If raw drops toward ~2.2–2.4e-8 P01-scale, the gap is our λ depth-profile and the fix is a proper per-layer end-to-end fit on a fresh block (coordinate descent, one design run per layer group). If not, the gap is in the consumers/transport.
- **D2 (half a day):** trace comparison on one 1024×16 network: dump per layer `dG`, `g_prev`, `var_pre`, `var_post`, `λ`, and the (4)/(2,2) consumer inputs from `external/504aldo/…/estimator_v17_open.py` (numpy shim, minutes) and from v31; compare layer by layer. Agreement of `dG` to <5% with different outputs → consumers/births differ; disagreement → transport differs. This locates the line.
- **D3 (half a day, after D2):** verify the feed coefficient: on one network, zero the κ4 inputs to the (3,)/(2,1) births in v31 and measure the raw change on 8 design nets; F68 predicts a large loss (their feed is worth 4× at this shape). A small loss means our table-feed is not their feed.
- The `_q31` table extension: **not now**. Revisit only if D1–D3 close the gap and raw is still >2.2e-8.

Corpus hygiene: F21, the λ sweep, P1 and P2 have all been read on the same 24 P01 networks, and the λ optimum was read there too. Mint a fresh 24-net DESIGN block (P11) for D1/D3 and the κ4-half fix; keep P01 for cost-only and parity work.

## 3. Q3 — is a non-cumulant method at 2.5e-9?

No evidence of one, and positive evidence against **[M]**: the leader's per-layer error profile has the same shape as ours (all-layers/final = 7.82e-9/1.89e-8 = 0.41 for #330876; 1.56e-8/3.78e-8 = 0.41 for #330950) — error growing with depth exactly as a truncated cumulant chain does; a sampling or sampling-hybrid estimator has a depth-flat error. Their raw 1.7–1.9e-8 is our own torch AUGMENT floor (1.91e-8) to within noise. They complete 50/50 deterministically at 71–94 s wall for 0.134 B — many small ops, i.e. structured/compressed linear algebra, not sampling (QMC at 13k points would sit near 2e-6 raw). Learned correctors are capped at ~1.02× on a K3 base in three independent hands. The 2.15× beyond 504aldo decomposes as 1.13× raw × 1.9× cost: **their edge is cost — an old-lineage representation whose marginal cost per source is ~0 (504aldo: "leaders' bill = ours minus the old tier to three digits") — plus a little κ4 memory.** The bounded test that matters is therefore not "another method class" but N1's kill test (§4). If N1 fails, the honest conclusion is that we do not know their old-source trick and cannot reach a prize position by Oct 17.

## 4. N1 — the kill test, concretely

On our chain (v31 already has `_p2_compress`, a range finder over a weighted leg Gram) at 8 design nets, ages 5–8: compare raw vs rank r ∈ {96,128,192,256,384} for (a) the forward-energy Gram (504aldo's F72) and (b) a *sensitivity-weighted* Gram: weight by JᵀJ with J = Π_{k>l} D_k W_k, D_k = diag(Φ(α_k)) from the K2 chain (one transposed transport per layer, ~0.03 B, or its top-128 right singular vectors). Pre-registered prediction: (b) reaches (a)'s rank-384 accuracy at r ≤ 192. If it does, merge all old lineages into one Tucker core in that basis (cost r³/n² units per layer, flat in source count) with rank chosen per MLP from the core spectrum — the only representation consistent with the leaders' flat bill and ±6% per-MLP FLOP spread. If (b) does not halve r, N1 is dead and P5 ships as published.

## 5. Q4 — the contribution claim, plainly

As it stands it is a **methods note**, not a contribution-prize submission: an independently-derived reconstruction identity, a label-free λ whose value is ill-determined but whose effect is flat, an equivariance fix to a range finder, a measurement protocol, and negative results. Replicating P3–P5 adds nothing to that inventory.

Two things would change the answer. First, **N1 positive** — a new compression principle (output-sensitivity-weighted bases for cumulant states, ARC's "relevance" made concrete) with a measured rank law. Second, even without N1, an **"anatomy of the κ4 gain" study at 1024×16 that nobody has published**: using F21 as the *exact* K3 reference (we are the only team running one at this shape), report what each approximation costs — rank-32 M residual +0.9%, age gates, the κ4 channel decomposed term by term (tonight's finding that ~85–90% of the κ4 gain is one n² transport term and the (3,1) slice is ~0 is new information), the Gaussian-tree λ negative (κ3-*sourced* κ4 is where the gain lives), label-free λ and its flatness. That is a real, honest contribution if the 1024×16 numbers are measured rather than proxied — and D1–D3 produce exactly those numbers as a by-product. Be transparent in it about LLM authorship and about which mechanisms are 504aldo's (the provenance file already says so).

On provenance: the timeline correction is right. The reconstruction identity predates the clone; the (2,1) regrouping and the range finder do not. Under Samuel's Sep-14 decision (own re-implementation + ask), the rules question should go out now with the timeline attached; the drafted wording in `PROVENANCE_504aldo.md` is fine. Nothing derived from v29/v17 bytes is ever submitted; v31 is our own code path on our own tower and is submittable.

## 6. Slice 7 contract (Sep 16–29), replacing handoff §§3–4 for this window

Day 0: canary v31 (approval); send the rules question (approval); mint P11 (24 nets, DESIGN).
Days 1–2: D1, D2, D3 in that order; fix the located discrepancy; re-screen on P11 (gate: raw ≤ 2.6e-8 P01-scale, i.e. ≥1.5× over v31 raw; prediction: 2.2–2.4e-8). Canary if passed.
Day 3: P3 rank-16 D21 feedback (gate ≥1.06× raw). Canary.
Days 4–5: P4 (L5 Strassen with pooled `out=` buffers, block-symmetric C_pre, final-layer trim; gate: ≤0.27 B at raw parity, ops ≤ 20k). Canary. Expected standing ≈ 5.5–6e-9.
Days 6–10: N1 kill test, then the merged old core if it passes (gate ≤0.20 B at ≤+5% raw). Canary. Expected ≤4e-9 if positive.
Days 11–13: if N1 negative, P5 as published; else per-MLP adaptive rank; Puffi-style λ refit target (consumed mean correction) on P11 as a 1-day accuracy probe.
Throughout: claims ledger 1 h/day; every graded row logged with wall/residual/ops.
Predictions to record before running: D1 moves raw by >15% (I say 40% likely); D2 finds a ≥1.3× discrepancy in `dG` or the births (60%); κ4-half closes to ≤2.4e-8 P01-scale by day 2 (55%); N1 halves rank at equal error (35%). Samuel's predictions to be taken per contract.

## 7. What I am and am not claiming

Measured tonight: the v17 ablation at 64×8 and 128×8 (small-shape proxy; at 1024 F68 says the feed's share is larger, which is why D3 exists); the board. Read from the repo: all v30/v31 numbers. Inferred: that our port's missing factor is in the dG transport line or the birth feed (from the proxy plus F68); that the leaders' old-source representation is a merged low-rank core (from cost identities and telemetry). Unknown: whether N1 works; the private-rerun behaviour of anything.
