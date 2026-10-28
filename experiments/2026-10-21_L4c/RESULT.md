# L4c — tier reallocation on A4's export: fewer exact κ3 ages, higher band rank (constants only)

Handoff: `_claude_snap/handoff_2026-10-19/HANDOFF_2026-10-21_L4_TIER_REALLOC.md`. Session L4, started 16:10 UTC Oct 21.

## 0. Prediction and rule (handoff, before any number)
- Bill: (E, r) = (5, 384) −6…−9 % F; (4, 512) −10…−15 % F; (3, 576) −14…−20 % F if the band's real cost is ≈ linear in r.
- Unrefit VAL-100 paired vs A4's export: +1…+4 % raw. **Keep: score-equivalent = Δraw + ΔF ≤ −2 % unrefit** ⇒ refit 276 updates on
  fresh nets, corrector ×10; after refit keep if VAL ≤ −1.0 % score-equiv (CI excl. 0) and FRESH-A100 ≤ −0.8 % score-equiv ⇒ L2 bake (3.5e6).

## 1. Which constant is the entry age
`_SHARED_AGE` (A4 export line 96) is the age gate: `_shared_contract`/`_shared_join` record weights and move the oldest recent
lineage into the shared basis once the recent count `tw.k − tw.ko ≥ _SHARED_AGE`. So **E = `_SHARED_AGE`** recent lineages are held
exact when the join fires (6 today); `_SHARED_RANK` (line 97) is the band rank r (288 today; `_shared_rank` caps it at n).

## 2. Sources (`make_variants.py`; exactly the two constant lines change, asserted)
| tag | parent | (E, r) | sha256[:8] |
|---|---|---|---|
| a4x_E6_r288 (zero control) | A4 export s368 `67ed6090` | (6, 288) | 67ed6090 — byte-identical to parent |
| a4x_E5_r384 | A4 export | (5, 384) | b084d93b |
| a4x_E4_r512 | A4 export | (4, 512) | c8434238 |
| a4x_E3_r576 | A4 export | (3, 576) | e78fc7b5 |
| a4s_E6_r288 (zero control) | A4s λ 3.5e6 `7b0d6800` (shipped) | (6, 288) | 7b0d6800 — byte-identical to parent |
| a4s_E5_r384 / E4_r512 / E3_r576 | A4s | as above | 08d84904 / 0394906a / 9170818b |

## 3. Compute
Pod 14uah7xtcpq161 (RTX PRO 6000 Blackwell, $2.49/h; no RTX 5090 stock anywhere at 16:05), shared with L7 arm b.
- Bill: `pod_bill.py` = `2026-10-21_L4/l4_bill.py` with seed-regenerated weights (manifest fixtures checked), numpy 2.2.6 /
  flopscope 0.12.3 f64 read-only, nets 900/901. (Local workerq #3681 sat on RAM — 12 GiB free vs a 12.7 GiB measured peak — cancelled.)
- Unrefit VAL-100: the pod trainer's `--val-only` (`2026-10-21_L7/pod/a5_pod_train.py`, R3 dev backend), pod torch 2.8.0+cu128 /
  numpy 2.1.2 (PyPI unreachable from the pod host; torch 2.13's `nvidia-nccl-cu13==2.29.7` is not on any reachable index).
  **Twin zero control on the pod:** A4 export VAL0 = 1.483189e-8 vs A4's recorded 1.483245e-8 (−0.004 %, max |Δpred| 2.1e-6) — not
  byte-identical; the cross-torch-version f32 floor, as J's (+0.002 %). Δraw rows below are paired vs A4's recorded VAL.
- Refit corpus note: the ext shards hold 1,500 rows each, so "ext05/06 rows 1472..2047" is 56 nets. A refit would use the unused
  ext00 rows 100..1499 ∪ ext01/02 rows 1024..1499 (2,352 nets).

## 4. Real bill (flopscope 0.12.3 / numpy 2.2.6, f64 read-only, nets 900 and 901 identical; `bills/bill_0123_pod.jsonl`, 16:22 UTC)
**Zero control PASS:** a4s_E6_r288 bills **293,790,066,123 F / 20,991 ops** = A4s's asserted bill to the FLOP (and a4x_E6_r288 = the
export's 288,310,814,515 F / 22,976 ops). Residual projection at A4s's served 13.82 µs/op.

| variant (E, r) | F, A4s bake (λ 3.5e6) | ΔF vs A4s | ops | resid. proj. | F, export | ΔF vs export | ops |
|---|---:|---:|---:|---:|---:|---:|---:|
| (6, 288) A4s / A4 | 293,790,066,123 | — | 20,991 | 0.290 s | 288,310,814,515 | — | 22,976 |
| (5, 384) | 300,270,222,859 | **+2.206 %** | 21,453 | 0.296 s | 294,055,141,044 | +1.992 % | 23,477 |
| (4, 512) | 316,405,198,743 | **+7.698 %** | 22,497 | 0.311 s | 310,010,463,138 | +7.526 % | 24,211 |
| (3, 576) | 325,950,932,824 | **+10.947 %** | 23,014 | 0.318 s | 320,697,865,212 | +11.233 % | 24,564 |

(The a4s_E3 net-901 row was lost in the tty copy; both nets read 325,950,932,824 F / 23,014 ops on the pod.)
**No variant saves F; all three cost more.** The band's real cost grows faster in r than dropping exact ages saves — the sandbox's
"≈ linear in r" premise fails on the real chain. (Inference, not measured: extrapolating L3's rank-320 slope, r 288 → 384 alone would be
≈ +1.86 %·(96/32) ≈ +5.6 %, implying one dropped exact age buys back only ≈ 3.4 %.) The rule's "≥ 6 % F saved" filter admits none.

## 5. Unrefit VAL-100 (run anyway — the pod was already paid for arm b; `paired.py`, per-net in `pod_val/`)
Paired vs A4's recorded VAL (1.483245e-8); the pod's own A4 VAL0 column removes the cross-torch offset (−0.004 %).

| source | raw mean | Δraw vs A4 | wins | 95 % CI | Δraw vs pod A4 VAL0 | ΔF (A4s bake) | **score-equiv** |
|---|---|---|---|---|---|---|---|
| (5, 384) | 1.466217e-08 | −1.073 % | 74 | [−1.485, −0.661] % | −1.069 % | +2.206 % | **+1.133 %** |
| (4, 512) | 1.454152e-08 | −1.889 % | 83 | [−2.318, −1.461] % | −1.885 % | +7.698 % | **+5.809 %** |
| (3, 576) | 1.469267e-08 | −0.837 % | 70 | [−1.326, −0.348] % | −0.833 % | +10.947 % | **+10.110 %** |

Prediction was +1…+4 % raw unrefit: wrong sign — raw *improves* unrefit (the extra band rank more than covers the lost exact ages),
but at a FLOP price that score never repays (λ = 0, score = MSE × F).

## 6. Decision (16:30 UTC): NO-GO, no refit, nothing for L2
Best score-equivalent is +1.13 % (keep ≤ −2 %). For the write-up: on the real chain the κ3 tier frontier is cost-bound by the band,
not by exact ages; E = 4/r = 512 buys −1.9 % raw for +7.7 % F. Spend: no separate pod — bills (≈ 25 s × 2 nets) and three VAL reads
(≈ 3 min) ran beside L7 arm b on pod 14uah7xtcpq161; marginal cost ≈ $0.2, booked to L7 arm b's pod.

## 7. Addendum (17:20 handoff text, run 16:30–16:39 UTC): τ_hub 3.0
**Which constant.** The hub cut is `_KAS_K["H"]` (A4 export line ≈ 2730): per-layer hub-column **keep counts**, not a scalar τ and not a
fitted table. A4's list is exactly `experiments/2026-10-06_mask/ktab/aggr-r_m32.json` H ("aggr-r R2.0/**Hd2.5**/Dr2.5", median kept count,
nearest 32, `grid/fit_ctl.json`). The same builder at **Hd3.0** is `ktab/safe-r_m32.json` H, so τ_hub = 3.0 = that list
`[1024,1024,1024,1024,992,992,960,960,928,928,896,896,864,864,832,832]` (vs 2.5: `[…,992,992,960,928,928,896,864,864,832,832,800,800,800]`).
R (rows, `_KAS_K["R"]`, R2.0) and D (transport, `_KAS_K["D"]`, Dr2.5) unchanged. `make_hub.py`: exactly one line differs (asserted);
pod copies regenerated by the same replacement, sha256 equal (a4x c95a7192 / ec2dd700, a4s edc3a391 / 77863073).

| variant (E, r, τ_hub) | F (A4s bake) | ΔF vs A4s | ops | Δraw unrefit vs A4 | wins | 95 % CI | score-equiv | pays? (Δraw ≤ −ΔF − 2 %) |
|---|---:|---:|---:|---|---|---|---|---|
| (6, 288, 3.0) | 296,810,783,076 | +1.028 % | 20,753 | +0.011 % | 48 | [−0.153, +0.174] % | **+1.039 %** | no (needs ≤ −3.03 %) |
| (4, 512, 3.0) | 318,986,730,446 | +8.576 % | 22,274 | −1.889 % | 79 | [−2.355, −1.423] % | **+6.687 %** | no (needs ≤ −10.58 %) |

(Export bills: 291,307,887,467 F / 22,738 ops and 313,232,318,487 F / 23,948 ops.) The sandbox's 85 % hub-cut error reduction
does not show on the real chain at the scored layer: +10 % hub columns buys +1.03 % F and **no** raw change (CI spans 0). **NO-GO.**

## 8. Final table at 16:40 UTC (all five variants vs A4's export / A4s bill)
| variant | ΔF | Δraw unrefit | wins | score-equiv |
|---|---|---|---|---|
| (5, 384) | +2.206 % | −1.073 % | 74 | +1.133 % |
| (4, 512) | +7.698 % | −1.889 % | 83 | +5.809 % |
| (3, 576) | +10.947 % | −0.837 % | 70 | +10.110 % |
| (6, 288, τ 3.0) | +1.028 % | +0.011 % | 48 | +1.039 % |
| (4, 512, τ 3.0) | +8.576 % | −1.889 % | 79 | +6.687 % |
None clears −2 %; no refit; nothing for L2.
