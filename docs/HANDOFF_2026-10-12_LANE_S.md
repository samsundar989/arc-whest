# HANDOFF — Lane S (new session, branch `lane-s`), 2026-10-12: Strassen v2 — zero-copy glue, then depth

**Read first:** `docs/research/2026-10-12_deep_review.md` §0, §5.1, §6. Reference code and probes: `docs/research/2026-10-12_strassen/` (`swproto.py` holds the three prototypes, a verbatim copy of `_sb`, and exact cost models that equal the real meter to the FLOP; `probe_out.py`/`probe_out.log` are the billing probes; `estimate_v84.py`, `shape_table.py`, `ledger_summary.py` reproduce the tables).

**Setup.** `git worktree add ../arc-whest-lane-s -b lane-s main`. You own `_sb`, `_mm`, `_depth_for` and nothing else in the estimator. Lane P owns main and the estimator files; you deliver builder atoms for `tools/build_v85.py` (same style as `tools/build_v83.py`: exact text patches, each anchor checked to occur once). No canaries from this lane; Lane P submits.

**Standing rules.** Predictions on record before every run. No external messages. No write-up work. `out=` only with an output of exactly the broadcast operand shape (never larger: that is flopscope issue #275). No `where=`, no symmetry tags. Keep every buffer float32 (a float64 destination doubles the bill).

## Why
94 % of v84's FLOPs are matmuls through `_sb`; 18.7 u of 177.8 u is Winograd glue. Per recursion level our glue is 33 quarter-block areas (15 additions, two 7-block `concatenate` stacks, one `block`). Measured on the real flopscope 0.12.1: `fnp.add/subtract(a, b, out=view)`, `fnp.matmul(A, B, out=view)`, `fnp.copyto(view, a)`, `fnp.concatenate(..., out=view)` work on slices and non-contiguous quadrant views and bill exactly numel(out) (matmul (2k−1)·numel(out)), one op each; `fnp.empty` is 0 FLOPs and one op. Writing the operand combinations straight into the stacked buffer and the post-additions straight into output quadrants removes the copies. `RESULTS.md` § D3 item 3 ("INFEASIBLE") tested `out=buf` and slice assignment only; it is superseded.

| v84 with… | A (`sw_out`) | B (`sw_out(fused=True)`) | C (`sw_ks`) |
|---|---|---|---|
| glue per level (quarter areas) | 21 | 15 on the last level, 21 above | 12 + one-off transforms |
| same depths: ΔF / F/B / ops | −3.82 % / 0.16690 / 23,964 | **−4.73 % / 0.16533 / 24,576** | −5.52 % / 0.16395 / 24,775 |
| optimal depth, leaf ≥ 16 | −4.93 % / 0.16498 / 26,255 | −5.96 % / 0.16320 / 26,875 | −7.00 % / 0.16139 / 26,975 |
| optimal depth, leaf ≥ 8 | −7.29 % / 0.16088 / 29,017 | −8.70 % / 0.15844 / 30,017 | −10.74 % / 0.15491 / 30,025 |
| zero-pad to 2^d, leaf ≥ 8 | −8.46 % / 0.15886 / 29,330 | −10.09 % / 0.15603 / 30,401 | −12.87 % / 0.15120 / 30,489 |

A and B are bit-identical to `_sb` on five shapes (`np.array_equal`); C rounds differently (same error size).

## S1 — variant B at today's depths → atom `sb2` (Oct 13, first)
1. Port `sw_out(fused=True)` into the estimator behind `_mm`, same depth rule as today. Allocate buffers with `fnp.empty` inside `predict` (per-MLP cost must stay order-independent; no module-level pools).
2. Checks, in this order:
   - **bit-identity:** `np.array_equal` of the full (16, 1024) prediction vs v84 on DEV nets 0–4 under real flopscope. Any difference stops the line: find it before anything else.
   - **meter = model:** 0.16533 (363.6 G FLOPs), 24,576 ops at the same depths.
   - **memory:** WSL RLIMIT_AS 8 GiB gate, 5-repeat lifecycle; expect VmPeak to fall.
   - **wall:** `tools/wall_ab.py` vs v84 on DEV net 3 (×2, alternating). Prediction (strategist, modelled): 0.77–0.82× (≈ 63–67 s noisy-implied). **Residual:** record local residual per op for v84 and v85.
   - AST scan: every `out=` call site has `out` shape == broadcast operand shape (`audit_out.py` does this for the prototypes).
3. **Twin.** Do not teach the torch shim `out=`. The twin keeps the legacy `_sb` text through a builder switch; A/B are bit-identical, so file↔arm parity is unchanged.
4. Hand to Lane P: atom, the five checks, package hash. P submits it as v85 (bit-identical ⇒ no FRESH-A).

**Prediction (strategist):** v85 graded 2.713e-9 ± 0.2 % (raw identical to #335562), server wall mean 62–70 s, residual max ≤ 0.27 s.

## S2 — depth by the break-even rule (Oct 13–14)
One more level on a block (m, k, n) pays when k + 3 > g_A·k/m + g_B·k/n + g_C, with (7, 7, 7) for a stacked level and (4, 4, 7) for the fused last level. Replace `_depth_for` by this rule with a leaf floor parameter.
- Build leaf ≥ 16 and leaf ≥ 8. Prices above. Gates: wall A ≤ 105 s noisy-implied; projected residual ≤ 0.33 s at the per-op rate the v85 canary measures on the server (this replaces the 26,000-op cap — Samuel's decision 3; until v85 is graded, stay ≤ 26,000); memory; DEV-100 raw within ±0.3 % of v85 (float32 error doubles per level: 42× plain at depth 5, 85× at 6, 185× at 7).
- If leaf ≥ 8 fails a gate, ship per-site depth: deepest level only on the sites with the largest FLOP saving per added op (`tools/d3_strassen_table.py` logic).
- Ops trim available: order slots so each side's plain blocks are one `concatenate(out=)`, and let the leaf matmul allocate its own output (≈ +2 ops per level instead of +3).

**Prediction:** leaf ≥ 16 passes all gates (−6.0 % F total); leaf ≥ 8 passes wall, and passes residual only if pooled buffers cut the per-op residual by ≥ 15 %.

## S3 — variant C (Oct 14–15)
`sw_ks`: Winograd in the Karstadt–Schwartz alternative basis, Morton layout, in place; 12 quarter areas per level. Not bit-identical ⇒ DEV-100 raw within ±0.2 %, then FRESH-A paired by Lane P. Uses N-D batch axes and `[::-1]` views (both metered correctly in the probes). Worth −0.8 % F over B at same depths and −2 to −2.8 % at leaf ≥ 8.

## S4 — dims on multiples of 64 (with Lane K, Oct 14–15)
`_advance` (62 u) and `_shared_contract_lean` (57 u) are stuck at depth 5 because K_R, K_D, H and the padded widths are odd multiples of 32. Options: (i) zero-pad to 2^d (row "zero-pad" above); (ii) round the KAS tables to multiples of 64 at neutral volume (−10.0 % A to −14.6 % C at leaf ≥ 8 in the model). Lane K is rebuilding the tables this week: agree the rounding with K and re-price with `docs/research/2026-10-12_costfloor/x3lib.py`.

## Do not
- `out=` larger than the operands, `where=`, symmetry tags, float64 buffers, module-level pools.
- Change anything outside `_sb`/`_mm`/`_depth_for`.
- Reopen classic 18-add Strassen (504aldo's spelling is 22 quarter areas; A is 21, B 15, C 12).

## Report
A daily paragraph in `docs/STRATEGIST_PACKET_2026-10-1X_laneS.md`; atoms and check outputs under `experiments/2026-10-13_s/`.
