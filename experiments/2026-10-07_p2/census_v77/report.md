# v77_kash charged-FLOP profile

Total: **395,412,204,457 FLOPs**. Expected anchor: 395412204457. Exclusive sum reconciles: True.

Categories sum child-subtracted exclusive costs. Inclusive function totals overlap and must not be added. Recent/old labels identify measured matrix calls; mixed elementwise readout and Wick work remain explicitly mixed. This does not infer costs from tensor sizes or reuse an old version's percentage. No instrumented wall or residual value is a server estimate.

| category | exclusive FLOPs | share |
|---|---:|---:|
| recent.D21_readout | 119,655,090,514 | 30.261% |
| recent.transport | 109,888,305,854 | 27.791% |
| old.reconstruct_legs | 40,836,499,712 | 10.328% |
| old.compression | 36,687,479,128 | 9.278% |
| covariance.transport | 36,314,212,352 | 9.184% |
| old.D21_readout | 30,299,076,064 | 7.663% |
| closure.correction | 5,367,614,647 | 1.357% |
| old.transport | 3,076,096,640 | 0.778% |
| history.P2_readout_mixed | 2,943,902,208 | 0.745% |
| history.D21_readout_glue_mixed | 1,883,359,936 | 0.476% |
| layer.other_table_birth_covariance_glue | 1,743,415,966 | 0.441% |
| history.hub_readout_mixed | 1,292,276,736 | 0.327% |
| history.thin_D21_readout | 1,057,921,440 | 0.268% |
| history.shared_readout_glue | 1,044,133,056 | 0.264% |
| history.P2_transport | 763,171,072 | 0.193% |
| history.D3_readout_mixed | 708,295,680 | 0.179% |
| birth.P2_compression | 537,397,800 | 0.136% |
| history.hub_transport | 403,980,288 | 0.102% |
| kas.row_gather | 383,383,808 | 0.097% |
| mean_corrector | 154,763,776 | 0.039% |
| history.transport_glue | 150,110,208 | 0.038% |
| history.Wick_scale_mixed | 118,872,064 | 0.030% |
| history.bookkeeping | 52,180,224 | 0.013% |
| prediction.casts_constants_output | 41,925,444 | 0.011% |
| closure.Gaussian_moments | 4,669,440 | 0.001% |
| transport_gate | 4,070,400 | 0.001% |

| category | L-1 | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recent.D21_readout | 0.00 | 0.00 | 0.63 | 1.25 | 1.81 | 2.39 | 2.88 | 2.75 | 2.70 | 2.56 | 2.43 | 2.36 | 2.23 | 2.18 | 2.06 | 2.01 | 0.00 |
| recent.transport | 0.00 | 0.00 | 0.32 | 0.96 | 1.53 | 2.06 | 2.54 | 2.43 | 2.39 | 2.26 | 2.14 | 2.08 | 1.96 | 1.85 | 1.81 | 1.77 | 1.67 |
| old.reconstruct_legs | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 | 0.39 | 0.58 | 0.77 | 0.96 | 1.14 | 1.31 | 1.49 | 1.66 | 1.82 |
| old.compression | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.62 | 0.80 | 0.84 | 0.87 | 0.94 | 0.96 | 0.99 | 1.05 | 1.10 | 1.12 | 0.00 |
| covariance.transport | 0.00 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.32 |
| old.D21_readout | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.26 | 0.44 | 0.59 | 0.73 | 0.89 | 1.00 | 1.14 | 1.24 | 1.37 | 0.00 |
| closure.correction | 0.00 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.00 |
| old.transport | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.09 | 0.09 | 0.08 | 0.08 | 0.08 | 0.08 | 0.07 | 0.07 | 0.07 | 0.07 |
| history.P2_readout_mixed | 0.00 | 0.00 | 0.01 | 0.01 | 0.02 | 0.03 | 0.03 | 0.04 | 0.04 | 0.05 | 0.06 | 0.06 | 0.07 | 0.07 | 0.08 | 0.08 | 0.09 |
| history.D21_readout_glue_mixed | 0.00 | 0.00 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.04 | 0.04 | 0.04 | 0.05 | 0.05 | 0.06 | 0.06 | 0.00 |
| layer.other_table_birth_covariance_glue | 0.00 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.00 |
| history.hub_readout_mixed | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 | 0.04 | 0.04 |
| history.thin_D21_readout | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 | 0.00 |
| history.shared_readout_glue | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.00 |
| history.P2_transport | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| history.D3_readout_mixed | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| birth.P2_compression | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 |
| history.hub_transport | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| kas.row_gather | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 |
| mean_corrector | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| history.transport_glue | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| history.Wick_scale_mixed | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| history.bookkeeping | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| prediction.casts_constants_output | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| closure.Gaussian_moments | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| transport_gate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **layer total** | 0.01 | 0.72 | 1.70 | 2.98 | 4.15 | 5.29 | 6.90 | 7.41 | 7.75 | 7.88 | 8.03 | 8.29 | 8.38 | 8.61 | 8.78 | 9.04 | 4.08 |

| function | calls | inclusive FLOPs | exclusive FLOPs |
|---|---:|---:|---:|
| predict_and_materialize | 1 | 395,412,204,457 | 41,925,444 |
| _layer | 16 | 395,370,279,013 | 1,743,415,966 |
| _mm | 285 | 361,733,032,816 | 361,733,032,816 |
| _dslice21 | 14 | 153,939,581,010 | 1,883,359,936 |
| _shared_contract_lean | 14 | 150,998,299,634 | 1,044,133,056 |
| _advance | 15 | 114,281,664,062 | 150,110,208 |
| _dslice3 | 15 | 45,780,974,336 | 708,295,680 |
| _full_legs_lean | 87 | 40,836,499,712 | 232,570,880 |
| _shared_join_lean | 14 | 36,687,479,128 | 17,495,707,128 |
| _p4_sym_sandwich | 15 | 35,030,630,400 | 23,592,960 |
| _closure_apply | 16 | 5,367,614,647 | 5,354,572,983 |
| _p2_recon | 29 | 2,943,902,208 | 2,943,902,208 |
| _hub_parts | 29 | 1,292,276,736 | 1,292,276,736 |
| _p2_compress | 15 | 537,397,800 | 34,327,080 |
| _kas_rows | 13 | 383,383,808 | 383,383,808 |
| _corr_delta | 16 | 154,763,776 | 89,964,544 |
| _wick_tower | 14 | 118,872,064 | 118,872,064 |
| _corr_raw | 16 | 64,799,232 | 64,799,232 |
| _append | 15 | 52,180,224 | 52,180,224 |
| _closure_what | 16 | 13,041,664 | 13,041,664 |
| _wick | 16 | 4,669,440 | 4,669,440 |
| _gate_eval | 15 | 4,070,400 | 4,070,400 |
| _truncate | 15 | 0 | 0 |

Full JSON retains layer, original source call-site line, matrix shapes, and both counts. Prediction initialization/casts/output and materialization are covered by the root span, so no unassigned FLOPs are silently omitted.
