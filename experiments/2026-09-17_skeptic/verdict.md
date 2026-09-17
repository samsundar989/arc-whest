# Skeptic verdict — handoff 09-17 slice (fresh context, docs/SKEPTIC.md)

*Saved verbatim by the Builder (the Skeptic has no write tool). 2026-09-17.*

## Skeptic verdict: PROVISIONAL (C1 PROVISIONAL · C2 ACCEPT · C3 PROVISIONAL · C4 ACCEPT · C5 ACCEPT)

(Not written to `experiments/2026-09-17_skeptic/verdict.md` by me — `.claude/agents/skeptic.md` / `docs/SKEPTIC.md` say a Skeptic does not write, and I have no write tool. Please save this block there verbatim. My only temp files were in the session scratchpad. No truth opened, no seeds claimed, no heavy jobs; working tree unchanged.)

**Independent recomputations done**
- From `experiments/2026-09-17_step2/rows.jsonl`, `…_step2b/rows.jsonl`, `…_step3acc/rows.jsonl`: per-network `mse·max(0.1,F/2^41)`, paired arithmetic, studentised t (df 23). Every LOG number reproduces to 4 digits: Q 0.8352× t95 [−2.465e-9,−1.501e-9] 0/24; QP 0.8474×; P 0.9966× [−7.67e-11,+8.35e-12] 9/24; R 0.9994× 13/24; Q3 0.7812× [−3.383e-9,−2.248e-9]; Qpair 0.8658× [−2.016e-9,−1.099e-9]; shared a3r256/a2r256/a3r192/a2r192 MSE ratios 1.7643/3.7270/2.9082/6.5141; p4d5 MSE ratio 1.0002, gain 1.0787×, t95 [+6.89e-10,+7.78e-10], 24/24. All arms 24 nets, 0 failures, single block P11; geometric agrees in direction everywhere; no single network carries >5.2% of any difference.
- Candidate SHAs: every row SHA == HEAD blob == live file; each scored arm file was committed exactly once and never modified.
- Commit order (commit UTC vs job start): step-2 contract a7a1a7a 07:03:53 < job 1607 07:05:02 and 1609 07:42:56; attribution registration 969307b 09:29:54 and arms 778f46f 09:30:14 < job 1635 09:30:24; step-3 contract 185884f 10:43:47 < job 1637 10:43:55; proxy rules 75a54ff 12:06:24 < job 1639 12:06:28. `s6_decide.py`/`s7_decide.py` unchanged since their contract commits; `p2_design_score.py`/`competition_score.py` last changed 09-16. No gate moved. Truth reads for 1609/1635/1637 registered in `experiments/truth_reads.jsonl` with real launch timestamps.
- `_fit_generations(1024,16,2^41)` returns None (exact tower) for v32, Q, QP, Q3, Qpair, P, p4d5 and the shared arms, so the guard's over-charge truncated nothing — not a confounder.
- C2 tables recomputed from `experiments/2026-09-17_closure/canon_programs.json` and `local_sim/closure/closure_*.json` (details below).
- C5: `uv run python -m unittest local_sim.test_run_mini_json` 10/10 here; positive control that the poisoned environment is really poisoned (`Get-FileHash` → "module could not be loaded", `ConvertTo-Json` fine).

---

### C1 — BUDGET-4/BUDGET-3/PAIR closures hurt; R and P null; "closure enrichment closed on this base" — **PROVISIONAL**

**Strongest challenge.** The measurements are real, but the harm may be the *chain the closures were dropped into*, not the closures. Two mixed-order inconsistencies sit exactly at the added order:
1. κ_iiij is set to zero while v32's own κ4 model makes it nonzero (`ref_step.blocks`: Q31 = (m/2)·λ·C). Under the generator's power counting Q31 has cost **3**, the same order as C·D21, C·D22, C·D3, C³. The generator's own README (lines 71–73) says that because κ(3,1) is dropped, BUDGET 3/4 "cannot improve the order past the scaling of those cumulants".
2. The all-distinct (1,1,1) birth is not enriched (`s6_build_closure.py` docstring), while hub route (i) still adds λC distinct-index content — so the tower is born with repeated slices at cost 3–4 and distinct entries at cost 2.
The observed pattern fits this at least as well as "these closures are wrong here": harm is **worst for BUDGET 3 (0.78×)**, exactly where the inconsistency begins, and adding cost-4 families recovers part (0.84×). The mirror experiment (the (3,1) consumer on the 25-term base, v34/job 1525) also hurt. **The joint arm — the added families plus the κ_iiij rows — was never run.** A smaller confounder: `_K4_LAM` is described as "Fitted end to end against score" (`estimator_k3_v32.py:71`) on the BUDGET-2 chain (though the 09-16 D1 swap moved only 2.4%, so λ alone cannot explain 16–22%).

**Evidence for it:** `local_sim/closure/README.md:15, 71-76`; `local_sim/closure_audit/ref_step.py:80` (Q31 block nonzero); `estimator_k3_v32.py:1188-1215` (hub routes/compensation); `local_sim/s1_build_v32.py:25-47`; step-2b result Q3 0.7812× vs Q 0.8352×.

**Implementation defect — no evidence, not fully excluded.** The (1,2) drop is safe (BUDGET 4 is i↔j symmetric — checked; tiny-shape parity uses the table's own (1,2) rows vs live (2,1).T). Normalised w^(k,p) with ALPHA_CAP 40 cannot overflow float32 at these polynomial degrees. But live Q was matched to the f64 reference only at tiny shapes (`s6_parity` 55/55); at 1024 `same_state.py` ran the *reference* on v32 boundaries, never compared it with the live float32 Q layer, and there is no count of zeroed non-finite entries at the live shape.

**Evidence for the original claim:** coefficients independently verified (v32's 27 terms == BUDGET-2 exactly, apart from B2's five (1,2) rows v32 never builds; v17 ⊂ B4 and ⊂ PAIR with 0 coefficient mismatches); unanimous sign (0/24 on every arm, tight intervals far from zero) on the binding statistic; R null correctly localises harm to recurrence; the registered consequence ("both hurt ≥5% → closed") was set before job 1635 and applied as written; pre-activations are only mildly non-Gaussian (|skew| ≤ 0.075), so the oracle's family-C warning does not apply.

**Where the wording overreaches.** "Closure enrichment by power-counting families is closed" should read *with κ_iiij = 0, unenriched (1,1,1) birth and v32's λ* — the step-2 scope paragraph says so, the step-2b consequence and synthesis drop it (the strategist response line 17 warns: "Reject those tested combinations, not every fourth-order/closure interaction"). "The public chain's better accuracy does not come from its closure on our state" is **untested**: no arm ran v17's actual program. PAIR has 63 terms v17 lacks (D21², D22², D22·D3/D4 in (2,1)/(2,2); D3²/D3D4/D4² in pK(2..4)), and v17 also carries 6 oriented Q31/Q13 rows.

**P (protected feed):** P genuinely changed computation (MSE 3.3241e-8 vs 3.3249e-8; FLOPs ×1.0036), protecting v32's two D22 columns. Nobody measured what fraction of the rank-32 residual's discarded energy that rank-2 content was; "null" may mean the range finder already captured it. QP > Q by 1.46% (t95 > 0), so protection matters when there is more content.

**Unresolved ambiguity.** Whether the harm is (a) the closure families on a correct chain, (b) the mixed-order inconsistency, or (c) a live-shape float32 defect — none of the existing artifacts separates them.

**Cheapest discriminating follow-up.** (1) Label-free, no block: on one captured 1024×16 boundary (the `same_state.py` machinery, seed 90_000_016_001), compare live `v35_Q` `_layer` pK/K/g_out in float32 against `ref_step` BUDGET-4 in f64 — rules (c) in or out. (2) Then one P11 arm (design block, fine): `v35_Q3` + the six v17-style single-factor Q31/Q13 rows with κ_iiij = (m/2)λC (consumers already exist from v34), vs v32 and v35_Q3. If it recovers to ≥ v32, the lane closure is wrong; if it stays ≤ 0.8×, C1 stands as worded.

**Checklist items triggered:** 2 (P11 only), 10/11 (P11 now read ~12 times; design tier only), 18 (mixed-order hidden confounder), reasoning ("cost-3 hurts most, cost-4 recovers" story post hoc and fits any order).

---

### C2 — executed-program audit — **ACCEPT** (two wording corrections)

Recomputed: v32 (27 terms) == BUDGET-2 **minus its 5 (1,2)-orientation rows** (B2 has 32). v17 live has 87 oriented terms: 54 non-Q extras beyond v32 plus **6** oriented Q31/Q13 rows (the LOG says "four"). v17-without-Q ⊂ B4 (99 B4 terms absent from v17) and ⊂ PAIR (63 absent), 0 coefficient mismatches; **not** ⊂ B3 (33 missing, e.g. D3², D12·D21). **Strongest challenge:** the agent's "independence" cannot be verified from artifacts — immaterial for correctness because the oracles match upstream `get_all_terms`. **Follow-up:** none; fix "four Q rows" and add the (1,2) qualifier wherever "exactly" is used.

---

### C3 — shared coordinates destroy accuracy at r ≤ 256; by calibrated proxy nothing cheaper than ~0.28 B stays under 1.7e-9; "route closed at affordable cost" — **PROVISIONAL**

**Accepted:** the P11 screen numbers; proxy calibration passes its own rule (P11 raw-MSE increase ÷ dev_mse = 1.246, 1.172, 1.178, 1.173, same order — recomputed from rows and `experiments/2026-09-17_step3/proxy.json`); the 1.7e-9 bar is the handoff's intermediate target (≤5% raw loss, `docs/CLAUDE_CODE_HANDOFF_2026-09-17.md:88`), not a builder choice.

**Strongest challenge:** "closed" is true only against the ≤5%-raw / 2× target. On the **binding objective (adjusted score)** the proxy's own numbers say otherwise. **a4r256 lean** — dev_mse 5.43e-9, C/B d4 0.2329, 15,384 ops — is not a "destroy" case and was never screened on P11 (screen covered ages 2–3 only). With the measured bundle/d4 price factor 0.929 (a3r256: 0.20200/0.21748), its bundle ≈ 0.216 B; raw ≈ 3.3255e-8 + 1.2 × 5.43e-9 = 3.98e-8; score ≈ 8.6e-9, i.e. **≈1.17× over v32 and ≈1.08× over v32+P4 d5** (a4r384 by the same arithmetic ≈1.145× / 1.06×). a4r256 beats v32+P4 d5 unless the proxy understates added error by > 1.8× (calibration: 1.17–1.25×). So the best *adjusted* grid point is a4r256, is ≤0.22 B, and forecasts a gain comparable to P4 d5 itself, which is being carried forward. "Nothing cheaper than 0.28 B" also only spans ages 3–4 × ranks 256/384/512 (age 5 at r256, ranks between 256 and 384 untried). Forecast weaknesses: dev_mse 5.4e-9 is ~4× below the smallest calibration point; two networks (per-net 4.1–6.8e-9); uncorrelated-error assumption (cross-term bound 2·sqrt(dev·mse) ≈ 2.7e-8 is not small).

**Is the join a fair test?** Mostly yes: weighting (consumption/sqrt/uniform) and 3 power passes move dev_mse ≤15%; age and rank dominate. a4r384 (~+1.5% raw by proxy) is not worse than the public F72 report (+8.4% raw at r384).

**Unresolved ambiguity:** proxy validity in the small-deviation regime; the true P11 added error of a4r256.

**Cheapest discriminating follow-up:** one P11 design read of `v37`-style a4r256 lean + P4 d5 (and optionally a4r384) against v32 and v32_p4d5 — after generating and label-free metering the a4r256 lean bundle (ops, peak commit). Until then reword the synthesis to "the intermediate and 2× targets are unreachable on the tested grid; best shared configuration forecasts ≈1.08× over v32+P4 d5 (unverified)".

**Checklist items:** 2, 3 (2-network proxy), 18 (conclusion framed on a surrogate instead of the objective).

---

### C4 — P4 at Strassen depth 5 keeps raw parity and improves adjusted 1.0787×, 24/24 — **ACCEPT** (P11 design tier)

Per-network MSE ratio min 0.99919 / median 1.00024 / max 1.00093; FLOPs 0.2802 vs 0.30232, whose ratio alone gives 1.0789. **Strongest challenge:** the 24/24 and tight interval are **mechanical** (a common FLOP cut on every network at equal MSE), not independent accuracy evidence — the claim is just "exact up to float reassociation, 7.3% cheaper", which it supports. Release caveats (outside this claim): depth-5 peak commit 6.53 GB (off-arm, job 1464) vs the 8,192 MB grader cap is unenforced locally; depth 5 adds ~4.8–5.0k ops, so the 22,000-op headroom must be re-metered on each bundle; still needs DECIDE-tier evidence before promotion. **Follow-up:** none for the claim; meter v32_p4d5 peak commit itself at the release gate.

---

### C5 — step-0 receipt name + module-independent hashing — **ACCEPT**

Diff (`scripts/run_mini_json.ps1`, `local_sim/predict_public.py`): all `Get-FileHash` replaced by streamed .NET SHA-256; receipt written to `<stem>.receipt.json`; pre-launch refusal if the receipt exists and if the report is itself named `*.receipt.json`; `find_receipt` keeps legacy discovery and refuses two differing receipts. Provenance checks unweakened — `load_receipt` still binds `report_sha256` to actual report bytes and checks source SHA, so a legacy receipt must still match its report. Tests 10/10 re-run here, with the positive control that the poisoned PSModulePath breaks `Get-FileHash` but not `ConvertTo-Json` (so the poisoned tests genuinely exercise the defect). Not re-run: the "old wrapper fails" mutation check (taken from the LOG).

---

### Protocol notes (not tied to one claim)
- **Snapshot vs live worktree.** Jobs 1609/1635/1637 were described as running on a "git snapshot containing this contract", but their commands `cd C:/Users/samsu/Documents/arc-whest`, so they executed on the **live worktree**; the snapshot commits (5ed0a6b…, a077380…, af7905c…) were not what ran. Harmless here (row SHAs == committed arm files, which never changed; scoring instruments predate the slice), but the snapshot provenance statement is inaccurate — fix the submit line.
- The v33 regeneration incident is irrelevant to C1–C5; no evidence depends on v33 bytes.
