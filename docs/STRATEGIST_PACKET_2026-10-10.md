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
| v82b | −6.36 % (F/B 0.16508) | −1.29 % DEV 0–9; twin VAL −0.89 % | 22,375 | FRESH-A running |

## What the evidence changed in plan rev 22

- §2.2 "two-sided D −8.84 %" → **V8b −3.55 %** only. The leg-row half, (ii)+(iii), is closed (+66 %).
- §2.2 "last-layer rows K16 = 800 (tiny)" and "old rebuild on D-kept rows (tiny)" are closed (+6.8 % / +10.6 %).
- §2.2 covariance input pruning is closed (+24 %).
- The hub move + H kink-active (Lane M5, −6.03 %) zeroes *columns*, not dead rows' D3, so this evidence does
  not condemn it directly. Its Hkh2.5 check stands.
- Revised central path: v82b ≈ 3.07e-9 → refits → Lane M κ₄ items. The 0.145 F/B target now needs the hub move
  plus about 2 % more from elsewhere.
