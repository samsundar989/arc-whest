# Current v72 charged-FLOP profile

Total: **472,202,879,721 FLOPs**. Expected anchor: 472,202,879,721. Exclusive sum reconciles: True.

Categories sum child-subtracted exclusive costs. Inclusive function totals overlap and must not be added. Recent/old labels identify measured matrix calls; mixed elementwise readout and Wick work remain explicitly mixed. This does not infer costs from tensor sizes or reuse an old version's percentage. No instrumented wall or residual value is a server estimate.

| category | exclusive FLOPs | share |
|---|---:|---:|
| recent.D21_readout | 147,288,383,488 | 31.192% |
| recent.transport | 142,515,572,736 | 30.181% |
| old.compression | 44,080,052,920 | 9.335% |
| old.reconstruct_legs | 42,210,590,720 | 8.939% |
| old.D21_readout | 37,761,417,216 | 7.997% |
| covariance.transport | 36,314,212,352 | 7.690% |
| closure.correction | 5,367,614,647 | 1.137% |
| old.transport | 3,849,216,000 | 0.815% |
| history.P2_readout_mixed | 3,020,513,280 | 0.640% |
| history.D21_readout_glue_mixed | 1,930,750,976 | 0.409% |
| layer.other_table_birth_covariance_glue | 1,620,821,150 | 0.343% |
| history.thin_D21_readout | 1,291,845,632 | 0.274% |
| history.hub_readout_mixed | 1,259,028,480 | 0.267% |
| history.P2_transport | 1,006,141,440 | 0.213% |
| history.D3_readout_mixed | 755,481,600 | 0.160% |
| birth.P2_compression | 537,397,800 | 0.114% |
| history.hub_transport | 503,070,720 | 0.107% |
| history.shared_readout_glue | 420,982,784 | 0.089% |
| mean_corrector | 154,763,776 | 0.033% |
| history.Wick_scale_mixed | 128,833,536 | 0.027% |
| history.bookkeeping | 68,936,704 | 0.015% |
| history.transport_glue | 67,108,864 | 0.014% |
| prediction.casts_constants_output | 41,925,300 | 0.009% |
| closure.Gaussian_moments | 4,669,440 | 0.001% |
| transport_gate | 3,548,160 | 0.001% |

| function | calls | inclusive FLOPs | exclusive FLOPs |
|---|---:|---:|---:|
| predict_and_materialize | 1 | 472,202,879,721 | 41,925,300 |
| _layer | 16 | 472,160,954,421 | 1,620,821,150 |
| _mm | 285 | 437,915,981,824 | 437,915,981,824 |
| _dslice21 | 14 | 188,693,380,096 | 1,930,750,976 |
| _shared_contract_lean | 14 | 185,470,783,488 | 420,982,784 |
| _advance | 15 | 147,941,109,760 | 67,108,864 |
| _dslice3 | 15 | 47,245,614,080 | 755,481,600 |
| _shared_join_lean | 14 | 44,080,052,920 | 19,134,962,360 |
| _full_legs_lean | 87 | 42,210,590,720 | 249,036,800 |
| _p4_sym_sandwich | 15 | 35,030,630,400 | 23,592,960 |
| _closure_apply | 16 | 5,367,614,647 | 5,354,572,983 |
| _p2_recon | 29 | 3,020,513,280 | 3,020,513,280 |
| _hub_parts | 29 | 1,259,028,480 | 1,259,028,480 |
| _p2_compress | 15 | 537,397,800 | 34,327,080 |
| _corr_delta | 16 | 154,763,776 | 89,964,544 |
| _wick_tower | 14 | 128,833,536 | 128,833,536 |
| _append | 15 | 68,936,704 | 68,936,704 |
| _corr_raw | 16 | 64,799,232 | 64,799,232 |
| _closure_what | 16 | 13,041,664 | 13,041,664 |
| _wick | 16 | 4,669,440 | 4,669,440 |
| _gate_eval | 15 | 3,548,160 | 3,548,160 |
| _truncate | 15 | 0 | 0 |

Full JSON retains layer, original source call-site line, matrix shapes, and both counts. Prediction initialization/casts/output and materialization are covered by the root span, so no unassigned FLOPs are silently omitted.
