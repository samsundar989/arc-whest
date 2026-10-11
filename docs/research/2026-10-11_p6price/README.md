# p6price: exact prices for the P6/M6 levers (Oct 11)

Built by the strategist on Oct 10–11. This is the Oct 10 flopmodel interpreter (`docs/research/2026-10-10_flopmodel/`) with two additions to `fakefs.py`, each following flopscope 0.12.1's own cost functions:
- `linalg.svd(…, k=…)` bills min(4mnk, economy);
- `linalg.eigh` bills 9n³.

`price.py` patches `estimator_k3_v82c_r1.py` with exact text replacements and meters each variant in a fresh process. In `run_census.py` the `~/scratch` path line is removed, so this folder's `fakefs.py` is the one imported.

## Run

```
cd docs/research/2026-10-11_p6price
python price.py                                # every variant; prints the table, updates prices.json
python price.py fold+Habs,fold+Hasym2030       # selected variants
python price.py --write-only fold+Habs         # only write variants/est_fold_Habs.py (twin --raw screens)
```

- `PRICE_SRC=<path>` changes the source file.
- `variants/` is git-ignored. Each file is about 380 KB.

## Validation

The base reproduces v82c_r1's real meter to the FLOP and the op: 363,022,283,745 FLOPs, F/B 0.16508, 22,375 ops. These are the values on the v82c_r1 FRESH-A json and the P5 ledger.

## Prices (Oct 11 00:30 UTC, all on v82c_r1)

F includes the harness's output cast (+32,768 FLOPs, +1 op).

### H frontier, with the fold

| variant | what | F/B | ΔF vs base | ops |
|---|---|---:|---:|---:|
| `base` | v82c_r1 | 0.16508 | — | 22,375 |
| `fold` | M5.1 masked star-Q31 fold (recent lineages) | 0.16554 | +0.27 % | 22,618 |
| `Habs` | H = kink-active count, τ2.5 table, ranked by α² | 0.15536 | −5.89 % | 22,854 |
| `fold+Habs` | both: **v83 as planned** | 0.15577 | −5.64 % | 23,097 |
| `fold+Hasym2030` | H ranked by max(−α/2.0, α/3.0), asymmetric count table | 0.15362 | −6.95 % | 23,386 |
| `fold+Hasym2025` | H ranked by max(−α/2.0, α/2.5) | 0.15062 | −8.76 % | 23,386 |
| `fold+Habs20` | H symmetric τ2.0 table | 0.14555 | −11.83 % | 23,360 |
| `fold+Habs30` | H symmetric τ3.0 table | 0.16317 | −1.16 % | 22,883 |

### Spend and structure levers, alone on v82c_r1

| variant | F/B | ΔF | ops |
|---|---:|---:|---:|
| `J3` (`_JOIN_SKIP_TAIL = 3`) | 0.16562 | +0.32 % | 22,084 |
| `J4` | 0.16715 | +1.25 % | 21,793 |
| `age6` | 0.17301 | +4.80 % | 21,870 |
| `rank288` | 0.17061 | +3.35 % | 22,411 |
| `rank320` | 0.17542 | +6.26 % | 23,293 |
| `p2r8` | 0.16661 | +0.93 % | 22,555 |

### On top of fold+Habs (0.15577)

| variant | F/B | ΔF vs fold+Habs | ops |
|---|---:|---:|---:|
| `+J3` | 0.15614 | +0.24 % | 22,806 |
| `+J4` | 0.15746 | +1.09 % | 22,515 |
| `+age6` | 0.16337 | +4.88 % | 22,592 |
| `+rank288` | 0.16117 | +3.47 % | 23,133 |
| `+od12` (per-shape depth, leaf ≥ 12) | 0.15371 | −1.32 % | 25,473 |
| `+od16` (leaf ≥ 16) | 0.15467 | −0.71 % | 25,167 |
| `+jsvd` (exact top-k SVD join basis) | 0.15746 | +1.09 % | 22,233 |
| `+pass2` (`_SHARED_PASSES = 2`) | 0.16145 | +3.65 % | 24,021 |
| `+p2svd` (P2 residual by exact top-4 SVD) | 0.15564 | −0.08 % | 22,977 |
| `+jsvd+rank224` | 0.15350 | −1.46 % | 21,945 |
| `+jsvd+rank192` | 0.14773 | −5.16 % | 21,945 |
| `+jsvd+age4` | 0.14910 | −4.28 % | 22,631 |

### Asymmetric count tables

Source: `kink_tables.json`, FIT q50, rounded to the nearest multiple of 32 (the convention of every KAS table: `active` and `keep` there are exactly the nearest-32 roundings of `act_raw` and `keep_raw`). Count = n − #(α < −a) − #(α > b) = keep_raw(a) − (keep_raw(b) − act_raw(b)).

| table | counts |
|---|---|
| −2.0 / +2.5 | 1024, 1024, 992, 960, 896, 832, 800, 736, 704, 672, 640, 608, 576, 544, 512, 480 |
| −2.0 / +3.0 | 1024, 1024, 992, 960, 928, 864, 832, 800, 736, 704, 672, 640, 608, 576, 544, 544 |

The −2.0 / +3.0 cut keeps 1–20 fewer columns than τ2.5 at every layer in raw FIT counts (11,904 vs 12,128 over layers 0–14 after rounding), so it is 1.4 % cheaper. It still keeps the always-on columns with α in (2.5, 3.0], which carry hubs because w1 ≈ 1, and drops more of the dead columns with α in (−2.5, −2.0), which carry almost nothing. It should be more accurate than τ2.5 as well as cheaper; the twin decides.

## Notes

- **The estimator drops everything on a dropped H column.** Its `_kh` gather removes A_L, w2b, hubs, e_b, the P2 residual rows (`Lb`) and S3 (`_sb`). The twin's H-literal mask keeps the residual and S3 on always-on columns (`_mSl` = the dead-only mask). Measure the shipped version with the twin's `Hfull=1` (P6.1 step 1).
- **The top-k SVD is a documented flopscope price.** `flopscope/numpy/linalg/_svd.py` and `_flops.py:svd_cost` bill "the textbook cost of the operation (like matmul), not literal BLAS work". The result is the exact economy SVD, sliced. Wall time is the real SVD's: about 0.36 s for 1024×1856 on a 2-core VM, and less on the 16-vCPU server.
- **complex64 is billed correctly**, checked against the real 0.12.1 package on Oct 11. A complex matmul bills 4× the real one (64³: 2,088,960 vs 520,192 FLOPs). Packing two real products into one complex product therefore costs 2×, not 0.5×.
