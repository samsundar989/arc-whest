# X2 — cost package v2 on #336993 (8be40031)

Spec: `_claude_snap/handoff_2026-10-19/HANDOFF_2026-10-19_X2_cost_package_v2.md`.
Status: predictions recorded before any run (~07:20 UTC). **Outcome 09:35 UTC: only D
survives. pkg2 = D alone, −0.26 % F at Δraw +0.006 %.** A, B1, B2, C and F fail their kill
lines; B1 and C fail catastrophically.

## Outcome (VAL-100 paired, GPU backend, unrefit, corrector on; job 3544, `val/gpu_summary.json`)

GPU control raw 1.534667e-8 vs cached CPU control 1.534734e-8: −0.004 % [−0.041, +0.035],
50 wins, so the instrument agrees with the canonical path. Scored-layer label floor ≈ 7.7e-11.

| item | predicted ΔF | measured ΔF | predicted Δraw | measured Δraw [95 % CI] | wins | kill line | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| A `_K2_O` → max(R,D,H) | −2 … −3.5 % | −1.521 % | +0 … +0.3 % | **+6.149 %** [+5.14, +6.82] | 4 | ≤ +0.608 % | KILL |
| B1 no old rebuild at l=15 | −0.9 … −1.3 % (B) | −0.946 % | +0 … +0.3 % | **+107.1 %** [+102.7, +111.4] | 0 | ≤ +0.378 % | KILL |
| B2 O[15] 704 → 640 | (in B) | −0.321 % | (in B) | +0.153 % [+0.084, +0.231] | 36 | ≤ +0.128 % | KILL (marginal; score-net ≈ −0.17 %, unclaimable) |
| C no Xold at l=7–9 | −1.5 … −2.5 % | −1.266 % | +0.3 … +1.0 % | **+60.87 %** [+57.4, +63.3] | 0 | ≤ +0.506 % | KILL |
| D λ 1e6 → 0.6e6 | −0.5 … −1.0 % | −0.260 % | 0 | +0.006 % [−0.029, +0.038] | 50 | residual ≤ 0.37 s | **KEEP** |
| E ⟨4,4,4;48⟩ | −1.5 … −3 % | not run | 0 | — | — | third worker only | not run |
| F K1 −128 at l=2–5 | −1 … −2 % | −1.257 % | +0.2 … +0.6 % | **+5.678 %** [+4.82, +6.65] | 9 | ≤ +0.503 % | KILL |
| all six composed | −5 % (A+B+D) | −5.367 % | ≤ +0.6 % | +183.2 % [+175.9, +189.2] | 0 | — | — |

D's residual: +398 ops at the served 0.3389 s / 22,235 ops (15.2 µs/op) gives ≈ 0.345 s
served-equivalent, inside 0.37 s by op projection. The six-call quiet protocol has not been run.

### What this says

- The old tier is not a small remainder at read time. Removing the old lineages' contribution
  to D3/D4 at the last layer (B1) doubles the final MSE, and dropping their D21 read at
  layers 7–9 (C) costs +61 %. The memo's "old-tier compression is 1.9 % of MSE" measures
  the *truncation* error of the band, not the band's *content*; the two were conflated.
- The row-set cuts (A, F) are not "near-exact". The transported D3 on the most-negative-α
  rows and the D21 rows at layers 2–5 carry ~6 % of raw each; the gain-mode fills do not
  replace them. Elasticity ≈ 4–5 (Δraw / |ΔF|), consistent with earlier capacity cuts
  (all ≥ 1).
- The A_architecture §3 "unmeasured exact-or-nearly-exact pools worth ≈ 15–20 GF" do not
  exist inside this representation. The only free item is the planner λ.

### pkg2 and R3

pkg2 = 8be + D (`src/vD_lam06.py`, sha 041e5465). D is a planner constant: it changes
Strassen depth choices only (Δraw +0.006 %, 50/50 wins), so it can be applied to R3's
selected export after the refit, with no need to compose it before. At −0.26 % F alone it is
below the ~1 % claimable threshold and is not worth a graded slot by itself.

R3 was not launched: the held binding's `training_ready` flip was refused twice by the
session's permission classifier and awaits Samuel. Disk C: was at 0.6–13 GB free during
this work (workerq snapshots 153 GB, `cache/` 182 GB), which R3's per-update checkpoints
also need.

## Schedule change (Samuel, 2026-10-20 ~07:15 UTC)

Lane 2 started at 07:06 UTC, after the spec's 06:00 composition freeze. Samuel chose to
**decouple**: R3 runs on the held local recipe (exact 8be parent, 736 updates, nine VAL
visits, step-0 fallback, workerq 16/8/3/1) without pkg2. X2 ablations are measured against
8be unrefit; keepers are then applied to R3's selected export and re-checked VAL-100 paired
before any canary. U1 NEW10 adds nine new parameter groups (not a rate-only change), so it is
not folded into the held R3 recipe.

## Pre-registered predictions and kill lines (copied from the spec, unchanged)

| item | prediction ΔF | prediction Δraw | keep if | measured ΔF | measured Δraw (VAL-100 paired, 95 % CI, wins) | kept |
|---|---|---|---|---|---|---|
| A. O∖R transport rows → R ∪ K1 ∪ {D3 weight > τ} | −2 … −3.5 % | +0.0 … +0.3 % | Δraw ≤ 0.4 × ΔF % | | | |
| B. drop L15 old-tier rebuild; also O 704 → 640 | −0.9 … −1.3 % | +0.0 … +0.3 % | same | | | |
| C. skip Xold / old LA,LP read at layers 7–9 | −1.5 … −2.5 % | +0.3 … +1.0 % | same; expect marginal | | | |
| D. `_SW_OP_FLOPS` 1e6 → 0.6e6 | −0.5 … −1.0 % | 0 | served-residual-equiv ≤ 0.37 s (six-call) | | | |
| E. ⟨4,4,4;48⟩ top-of-ladder step, ≥ 8 GF products | −1.5 … −3 % | 0 (bit-level) | only with a third worker; parity 3e-5 rel. Frobenius | | | |
| F. fewer K1 read-out rows for D21 at layers 2–5 | −1 … −2 % | +0.2 … +0.6 % | same rule as A | | | |

Strategist composite prediction: A+B+D −5 % F at ≤ +0.6 % raw, P 0.65.

## Implementation notes (decided before any accuracy read)

- **A as built (`vA_omax`)**: O must contain R, D and H, not just R ∪ K1, because the D21
  columns and birth columns are mapped into O-space through `_oinv`. R ⊂ O, since both are
  top-by-α in the same ordering, so O∖R is the most-negative-α rows, where K1 membership is
  rare and the D3 weight αφ(α)/σ² is small. Faithful cut within the current code:
  `_K2_O[l] = max(R, D, H)` for l = 0..14, giving −32 rows on most layers and −64 at
  layer 9. The memo's ≈ 10 GF "O∖R" pool overstates what this can reach unless the D/H
  column sets are also decoupled from O (a code change, not a table edit).
- **B1 (`vB1_l15noold`)**: at l = 15, `_full_legs_lean` substitutes zeros for
  `Q @ [FA|FM]`, so D3 and Tier-2 D4 read the exact lineages only. Q is still transported
  at l = 15 (a further ≈ 0.3 GF is available).
- **B2 (`vB2_o640`)**: `_K2_O[15]` 704 → 640.
- **C (`vC_noxold79`)**: `Xold` is skipped at l ∈ {7, 8, 9} and `tw._qold = None` there;
  old legs are still transported and rebuilt for D3/D4.
- **D (`vD_lam06`)**: `_SW_OP_FLOPS` 1e6 → 6e5.
- **F (`vF_k1m128`)**: the spec gives no size, so it was pre-registered here as −128 K1
  rows at layers 2–5 (1024/992/928/896 → 896/864/800/768).
- E is not attempted (no third worker).

## Bills: real flopscope 0.12.3 meter, `tools/_cost_worker.py`, DEV net 0, f64 read-only

| variant | FLOPs | ΔF vs control | ops | Δops |
|---|---:|---:|---:|---:|
| v0_control (sha 8be40031, byte-identical) | 286,868,957,368 | — | 22,235 | — |
| vD_lam06 | 286,123,309,398 | −0.260 % | 22,633 | +398 |
| vB2_o640 | 285,948,268,658 | −0.321 % | 22,235 | 0 |
| vA_omax | 282,504,610,375 | −1.521 % | 22,278 | +43 |
| vB1_l15noold | 284,153,403,562 | −0.946 % | 22,134 | −101 |
| vC_noxold79 | 283,237,526,017 | −1.266 % | 21,924 | −311 |
| vF_k1m128 | 283,261,914,327 | −1.257 % | 22,285 | +50 |

| **vALL = A+B1+B2+C+D+F** (`compose.py`, sha 43d789b6) | 271,471,929,471 | **−5.367 %** | 22,274 | +39 |

The composition is close to additive: the single cuts sum to −5.583 %. `compose.py`
reproduces each single variant byte-exactly when given one item.

Kill lines implied by these bills (keep if VAL-100 paired mean Δraw ≤ 0.4 × |ΔF|):
A ≤ +0.608 %, B1 ≤ +0.378 %, B2 ≤ +0.128 %, C ≤ +0.506 %, F ≤ +0.503 %. Break-even on
score alone is Δraw ≈ |ΔF|. All op changes are within ±400 of 22,235, so residual stays
near the measured 0.339 s served value (≈ 15 µs per op).

The control reproduces the official #336993 bill exactly (286,868,957,368 F / 22,235 ops).
The shape-only interpreter (`audit/census.py`, Oct 14) cannot run 8be (`fnp.linalg.inv`
unsupported), so the real meter is the only bill used.

## Accuracy instrument: GPU backend screen (changed 08:58 UTC)

The CPU `tools/ev.py` VAL reads (jobs 3539–3541, 2 workers × 2 BLAS threads) measured
about 1,000 CPU-seconds per prediction: 6 nets per variant in 55 minutes. Seven candidates × 100 nets would
need about 12 h of the whole machine. They were cancelled with no result. The screen now uses
the existing cupy backend (`tools/_predict_worker.py`, `WHEST_BACKEND=cupy`, ~4 s/net)
through `gpu_val.py`, workerq job 3544. The control and all candidates run on the same
backend, VAL 0–99, scored layer, paired. The GPU's float32 reduction order adds ~1e-3
relative per-net noise; the GPU control is also compared with the cached CPU 8be control.
Earlier attempts 3513–3533 never ran: they had a default GPU request, then hit failed snapshots on a full disk, then RAM waits.
Only the composed candidate chosen for shipping goes through the canonical numpy path (FRESH-A).

## Method

One-line gain/flag edits on a copy of the 8be source, with a byte-identical zero control. Each
variant is billed with the audit interpreter
(`docs/research/2026-10-15_exploration/audit/{fsmodel,census}.py`) and the real flopscope
0.12.3 meter on 2 nets. Accuracy is read VAL-100 paired through workerq, unrefit and
corrector-on, reporting mean Δraw, wins and 95 % CI.
