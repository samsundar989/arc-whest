# Strategist packet — Lane P5, 2026-10-10

## Daily paragraph (Oct 10)

v82a (`fnp.block` + `nan_to_num`, bit-identical to v81d) graded **3.27946e-9 (#335165), new best**, exactly on
projection, so the server bills `block` as flopscope 0.12.1 does. **Two-sided D works once its feature is fixed:**
one scalar per layer (ρ, the d21e energy ratio over all vs kept columns, 1.12–1.68) turns the +8.6 % cost of
dropping dead D21 columns into −0.16 % unrefit. Three dead-row prunings all failed hard: leg output rows on the
D set (+66 %), last-layer K16 = 800 (+6.8 %) and the old-tier rebuild on D rows (+10.6 %). They share one cause:
**dead neurons' D3 matters**, because their means sit in the ReLU tail, where κ3 is a first-order correction.
Covariance input pruning is closed as well (+24 % / +11 %), consistent with P2.4.2. So V8 (−8.84 %) is replaced
by V8b (−3.55 %). Skipping the last two old-tier joins *improves* raw by 1.1 % and cuts F 0.9 %. The surviving
bundle, **v82b** (join-skip, fused advance, two-level sandwich split, V8b), meters F/B 0.16508 (−5.2 %) at
DEV raw −1.29 % (10/10 nets); its FRESH-A is running. R1 (V8b, v82b-arm and a matched control, 2 epochs
each) is running on the pod.

## Ledger vs v81d (#334948: F/B 0.17629, raw 1.8833e-8 graded, 25,741 ops)

| build | ΔF cumulative | Δraw (best evidence) | ops | status |
|---|---:|---:|---:|---|
| v82a | −1.22 % | 0 (bit-identical; graded raw identical) | 24,283 | graded 3.2795e-9 |
| v82b | −6.36 % (F/B 0.16508) | graded −1.20 % | 22,375 | **graded 3.0717e-9 (#335227)** |
| v82b_ws (B) | −4.16 % (F/B 0.16895) | graded −1.21 % | 20,269 | **graded 3.1433e-9 (#335250)** |
| v82c_r1 | −6.36 % | graded −2.01 % | 22,375 | **graded 3.0464e-9 (#335253), new best** |

## What the evidence changed in plan rev 22

- §2.2 "two-sided D −8.84 %" → **V8b −3.55 %** only. The leg-row half, (ii)+(iii), is closed (+66 %).
- §2.2 "last-layer rows K16 = 800 (tiny)" and "old rebuild on D-kept rows (tiny)" are closed (+6.8 % / +10.6 %).
- §2.2 covariance input pruning is closed (+24 %).
- The hub move + H kink-active (Lane M5, −6.03 %) zeroes *columns*, not dead rows' D3, so this evidence does
  not condemn it directly. Its Hkh2.5 check stands.
- Revised central path: v82b ≈ 3.07e-9 → refits → Lane M κ₄ items. The 0.145 F/B target now needs the hub move
  plus about 2 % more from elsewhere.

## Evening addendum (Oct 10, ~21:00 UTC)

- **v82b graded 3.0717e-9 (#335227), new best** (−7.5 % vs v81d), 0.11 % above its FRESH-A projection.
- **v82b_ws graded 3.1433e-9 (#335250), the new wall-safe B.** It took the A/B slot instead of v81d_ws:
  it is the ws build of the new A and projected 0.26e-9 better. Two v81d_ws FRESH-A runs lost nets to local
  commit-limit OOMs, and `tools/ev.py` now keeps partial caches so this cannot cost an hour again.
- R1 (pod): the V8b and v82b arms refit cleanly (−0.92 % / −1.73 % vs the control's best; the control
  early-stopped). The refit is baked into **v82c_r1** (FRESH-A 1.0083× vs v82b, 99/100), submitted as the
  third canary of the day. R2 (2 more epochs) is running on the pod.
- Next (P5.3): Lane M5's items as they arrive; the hub move + H kink-active is now the biggest remaining
  cost lever.
