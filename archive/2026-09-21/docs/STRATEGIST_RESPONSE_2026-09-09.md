# Strategist response — 2026-09-09: after AUGMENT the gap to #1 is cost, and cost has an existence proof

*Answers `docs/STRATEGIST_PACKET_2026-09-09.md`. Supersedes the Sep 8 response where they differ. Serves as the Slice 4 handoff (§6).*

## 0. Verdict on Slice 3: accepted, and the central result stands

AUGMENT at 1.912e-8 (24/24 nets, CI [1.82, 2.00]e-8, bit-exact control) with all-layers 6.9e-9 reproduces the leaders' two-number signature. The ablation (raised diagram budget alone 1.085× *worse*; κ₄ zeroed 9.2× worse; all of the 2.09× is κ₄'s (n,n) structure) is the cleanest attribution this project has produced and is write-up material on its own. The three falsified claims from my Sep 8 response are accepted as falsified: D3 was already on; D2's r=1 injection cannot represent the traceless part (0.2% of κ₄'s Frobenius mass); both cost-lever estimates were wrong (pruning 0.961×, hybrid Strassen has no such point). The port's 4 GiB blocker being proven unnecessary three ways is exactly the kind of check that should precede a build.

## 1. Reframing: the remaining gap is not accuracy

Read the board with AUGMENT in hand. The six leaders are at raw 1.87–2.13e-8 — where we will be after the port — and at **0.18–0.21 B**. #1 is 3.80e-9 = 1.87e-8 × 0.205. Our nominee-grade cost floor is 0.52 B *before* the port's unpriced addition. So once AUGMENT ships:

    our raw ≈ 1.80e-8 (graded)   ≈ theirs
    our multiplier 0.52–0.86      = 2.5–4× theirs

**The entire distance to #1 is multiplier, and it is a distance six teams have demonstrably closed at our accuracy.** "Cost engineering is exhausted" is true of the levers we have priced; it is false of the problem, and the board is the proof. The packet's §4 arithmetic ("to reach #1 at 0.52 B we need raw ≤7.3e-9, a further 2.5×") is correct but it is the wrong axis: nobody on the board has that raw either. The right sentence is "to reach #1 at raw 1.8e-8 we need ≤0.21 B nominee-grade", and that is a spelling problem with a known answer somewhere in six other people's code.

What the leaders' telemetry says about that spelling (Sep 5 capture, unchanged): 25–35k ops/MLP (ours 12–15k), kernel throughput 5–10 GFLOP/s (ours 17 at L=3), wall 70–119 s (they run *to* the cap; J2W's max is 118.7 s). That is a deeper, op-heavier, wall-heavier spelling than ours at 2.5× fewer billed FLOPs. Two readings, not exclusive: (a) deeper Strassen with leaves executed efficiently enough that L=5–6 fits in ~100 s ((7/8)⁶ = 0.45 → 0.78 B × 0.45 = 0.35 B; with AUGMENT still ~0.4 B, so (a) alone is not enough); (b) a representation with fewer billed FLOPs per layer than our lineage-of-(n, kn)-blocks — ARC's paper constant c″₃ ≈ 2 (0.25 B) is stated with their symmetric-tensor FLOP accounting (App. J, α_{n,d}), and a symmetric-packed spelling of the dominant contractions is the one representation-level lever this repo has never priced. Either way the target is concrete: **K3+AUGMENT at ≤0.25 B with ≥2× wall margin.** I do not know which spelling gets there; the slice below is designed to find out cheaply before building.

## 2. Answers to §5

**Q1 — port first? Yes, unconditionally, but price before transcribing.** It is a measured 2× with zero research risk; every sub-floor or cost lever is built *on* the AUGMENT chain; a graded top-15 entry with the attribution story is the spine of the $20k case. Day 1 of the port is the price: spell the fused degree-4 route's ops on the meter, get the FLOP delta and op delta, and report them before writing the rest. If the delta is +60% (0.86 B), it still ships (1.55e-8 beats 2.03e-8) — but then the cost slice in §3 is the whole game.

**Q2 — what gates the traceless-carrier lane?** Two instruments, both cheap, neither requiring a new oracle at 1024:

- *The board.* Puffi — whose entire Phase-1 algorithmic contribution was SSC, a low-rank traceless-κ₄ carrier — sits at 2.13e-8, the *worst* of the six, at 0.18 B. If SSC-class structure bought ≥1.5× over AUGMENT at this shape, its authors would show it. That is a strong prior that low-rank traceless carriers saturate at the same floor.
- *The small-width ladder at fixed L/n = 1/64.* ARC's conjecture MSE_K ≈ c_K (L/n)^K makes (n, L) ∈ {(128, 2), (192, 3), (256, 4)} the affordable proxies for (1024, 16). At those widths run factored-AUGMENT, **unfactored AUGMENT** (O(n⁴) is fine there), **full K4**, and MC truth (cheap). The ratios give, directly: the factored-vs-unfactored gap (the packet's lever 3, "never measured at any width"), and the full-κ₄ ceiling over AUGMENT (the ceiling of *every* traceless carrier, SSC included). Extrapolate the ratios along the ladder; if full K4 over AUGMENT is <1.5× at (256, 4) and flat or falling with n, the lane is dead at 1024 and no representation-faithful oracle at 1024 is needed. Prediction to record: full K4 / AUGMENT ≈ 2–4× at (256, 4) (the conjecture's (L/n) factor is 1/64 but c₄/c₃ is unknown); unfactored/factored AUGMENT ≈ 1.1–1.3×. Two GPU-days at most; no seed block.

**Q3 — the learned corrector?** Bounded by the same board argument: with six teams inside a 14% band, nobody has a ≥1.5× corrector on top of AUGMENT, and marius's 1.87 vs the pack's 2.0–2.1 is the plausible size of one (~10%). Worth doing only because it is offline and unmetered: budget it at three GPU-days total, features logged from the AUGMENT torch chain, held-out on 24 nets, ship it if ≥1.1×, and never let it displace the cost work.

**Q4 — effort split.** Not "board vs write-up": the same work serves both. The write-up's spine is now (i) the exact factored K3 at 1024×16 on the fnp meter and the three structural identities; (ii) the AUGMENT attribution ablation — budget worthless without the structure it consumes, self-inconsistent state 9× worse than lower-order consistent; (iii) the memory-depth ceiling and the regenerating-collapse-beats-truncation result; (iv) the explanation of the board's shared floor as AUGMENT-class κ₄-in-flight, with the ladder measurement of where the class ceiling actually is. That is a mechanistic paper ARC will want to read regardless of rank. Split: **60% cost (§3), 25% ladder + corrector, 15% write-up** until Oct 3; then invert. #1 is not out of reach — it is a 2.5× cost problem with a public existence proof — but top-5 (≤4.3e-9 today) requires ≤0.24 B and I put that at maybe one in four; top-10 (≤7.8e-9) requires ≤0.43 B and is likely.

**Q5 — nominees.** The asymmetry decides it: one timed-out row on the fresh 100 costs ~9e-3 on the mean; 12.4% of score is 2.5e-9; break-even timeout probability is ~3e-7 per suite. No wall model resolves that tail. Rule: **nominee A must carry ≥2.0× projected worst-wall margin under the measured 1.6× contention** — so A = the deepest configuration that satisfies it (depth 2 at 0.604 B / 33 s if nothing intermediate exists; a per-shape-class hybrid between depth 2 and 3 probably lands ~0.57 B at ~45 s and should be priced). B depends on the rule the last handoff asked for and this packet still does not quote: if Rules v14 §6 scores each designated entry independently and ranks by the best, B = depth 3 (`330327`, 1.21×) as free upside; if it does not, B = `330212` (9.5 s). **Quote the sentence.** Re-designate as the AUGMENT port lands, under the same margin rule.

## 3. Slice 4 (Sep 10–23): the cost slice, with the port as its first deliverable

Ordered by value per day; each item has a stop rule.

1. **Port price (day 1).** Meter the fused AUGMENT route: FLOPs, ops, largest array, projected wall at Strassen depth 2 and 3. Report before transcription. Stop rule: none — this is information.
2. **Port, certify, canary (days 2–6).** `estimator_k3_aug.py`: same tower, AUGMENT nonlinearity, Strassen at the deepest depth with ≥2× margin. Twin arm, ops ≤24k, projected residual ≤0.3 s, Linux memory gate on the sha, official mini-100, one canary on Samuel's approval. Expected graded: raw ~1.8e-8, adjusted 1.06–1.55e-8 (#15–22).
3. **Wall profile of the Strassen spelling (days 3–5, parallel).** Where do the 51.8 s at L=3 go — leaf matmuls, the 15 Winograd adds per level, reshapes, or op overhead? If adds/reshapes dominate (they are memory-bound passes over ~(7^L × leaf²) arrays), a spelling that forms combinations in place and fuses reconstruction could halve wall at fixed FLOPs, which converts directly into one more Strassen level under the margin rule. Target: L=4's FLOPs (0.486 B) at ≤60 s worst; L=5 (0.425 B) at ≤90 s. Stop rule: if L=3's wall cannot be cut 25% by day 5, this lever is closed.
4. **Symmetric-packed spelling of the dominant contractions (days 5–12).** The (2,1)-slice extraction and `contract_W` on the lineage carry ~93% of the bill. Price a spelling that stores and contracts only the unique entries of the symmetric objects (ARC App. J's α_{n,3} ≈ 1/6 bound on order-3; the lineage's identity legs already capture part of it — measure what is left). Gate: parity 1e-6 on the slices; ≥1.5× on those two ops or close. This is the one representation-level lever untried and the only candidate for the full 2.5×.
5. **Small-width ladder (GPU, parallel, ≤2 days):** §2 Q2. Decides the traceless lane and the unfactored gap for the write-up.
6. **Corrector (GPU, background, ≤3 days):** §2 Q3.
7. **Every Friday:** nominee pair re-evaluated under the ≥2× rule; write-up sections (i)–(iv) drafted as results land; near-final Oct 3.

Kill criteria for the slice: if by Sep 23 neither item 3 nor item 4 has produced a ≥1.3× priced saving, cost is closed for real at ~0.5 B, the realistic finish is top-10 to top-15 on score, and Oct 3–17 goes to hardening, nominations, and the write-up.

## 4. Corrections carried

Retired from my Sep 8 response: D2 as the traceless gate; D3; the ≤0.35 B-by-Strassen and 0.8×-pruning projections. Standing: AUGMENT is the floor's mechanism (now measured); the leaders' 0.2 B is the target and it is a cost target; nominee margin ≥2.0×; the 24-net design set reads ~6% hot vs the graded 100 — quote graded numbers where they exist.
