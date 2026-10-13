# O4 results: spend items on the pruned stack: J3 PASSES (net −0.22 %), the rest fail (2026-10-13)

Setup:
- **Prices:** on v87 = `fold+age6` + `sb2+sbtrim+depth8+k0:unrefit+k1:2.5_3.0+k2:3.0:unrefit+k3`, Lane K's
  out=-aware meter (`variants/prices.txt`; the base reproduces K's 0.14250 / 25,211 ops exactly).
- **Accuracy:** unrefit VAL-100, twin `--raw`, paired, on K's stack B over v84 (`est_B*.py`), workerq job 2682.
  - The first run (job 2675) on the full v87 files failed: the twin's torch shim has no `fnp.empty`, which sbtrim
    uses. sb2/sbtrim/depth8 only reorganise Strassen, and Lane K measured its stacks the same way.
  - The control reads 1.769215e-8 against K's recorded 1.767948e-8 (+0.07 %; K's own `tw.py` path). The comparison
    is paired, so this offset cancels.
- Predictions: `PREDICTIONS.md`.

| arm | raw MSE | Δraw | break-even | ΔF | **net** | per-net Δ ± se | wins | ops (Δ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **J3** (`o4`) | 1.76451e-08 | −0.27 % | −0.04 % | +0.04 % | **−0.22 %** | −0.27 ± 0.05 % | 75/100 | 24,780 (−431) |
| age 7 | 1.73837e-08 | −1.74 % | −3.61 % | +3.75 % | **+1.94 %** | −1.70 ± 0.17 % | 80/100 | 24,476 (−735) |
| J4 (`o4:4`) | 1.75970e-08 | −0.54 % | −0.87 % | +0.88 % | **+0.33 %** | −0.55 ± 0.09 % | 71/100 | 24,309 (−902) |
| SVD join (`o1`) | 1.76561e-08 | −0.20 % | −1.49 % | +1.51 % | **+1.30 %** | −0.21 ± 0.11 % | 61/100 | 23,782 (−1,429) |

## Reading

1. **J3 is nearly free on the pruned stack and pays.** K pruned the late joins' inputs, so skipping one more
   late join costs +0.04 % F (on v84 it was +0.36 %). It keeps the layer-12 lineage exact, worth −0.27 % raw at
   5 se, and it cuts 431 ops.
2. **Age 7 on the pruned stack is −1.7 %, not −3 to −4.5 %.** O5 explains why: age 6 already took κ3 and the
   old-tier error down 44 % and 77 %, so a 7th generation has little left to repair. It is net +1.9 %.
3. **J4 adds only −0.27 % over J3 for +0.84 % F,** and the SVD join is a substitute (as on v84). Both fail.

Versus predictions:
- mine: J3 −0.15 to −0.4 % ✓, P 0.85 ✓; age 7 −2.5 to −4 % ✗ (−1.74 %, smaller); J4 −0.3 to −0.7 % ✓, fails ✓;
  SVD −0.1 to −0.3 % ✓, fails ✓.
- strategist: age 7 −3 to −4.5 % ✗ (−1.74 %).

## Hand-over to Lane P

`HANDOVER_O4_J3.md` (this folder).
