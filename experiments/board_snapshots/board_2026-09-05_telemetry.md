# Live board + graded telemetry, captured 2026-09-05 ~11:40 CT via Chrome

Top of Phase 2 board (109 entries; we are #47 at 8.50e-8, raw 8.50e-7, floor):
| rank | team | adj | raw | util |
|---|---|---|---|---|
| 1 | marius_binner | 3.80e-9 | 1.87e-8 | 0.205 |
| 1 | suliman_tadros (NEW, 13 entries) | 3.80e-9 | 2.03e-8 | 0.186 |
| 3 | J2W (joe_wanza) | 3.90e-9 | 2.02e-8 | 0.195 |
| 3 | Puffi (andrew_epstein) | 3.90e-9 | 2.13e-8 | 0.181 |
| 5 | oqaris | 4.30e-9 | 2.06e-8 | 0.209 |
| 6 | Activation-Regions | 5.40e-9 | 2.80e-8 | 0.192 (8 rows non-completed!) |
| 6 | fklassen | 5.40e-9 | 2.50e-8 | 0.218 |
| 8 | hyojun_kwon | 6.40e-9 | 2.83e-8 | 0.227 |
| 9 | lode_dockx | 7.10e-9 | 4.78e-8 | 0.148 (all-layers MSE 0.70 = final-row only) |
| 10 | 504aldo | 7.80e-9 | 2.13e-8 | 0.367 |
| 11 | Camaro | 8.70e-9 | 2.73e-8 | 0.318 |
| 12 | jamespayor | 9.40e-9 | 1.96e-8 | 0.482 |
| 13 | neilarmstrong4 | 9.90e-9 | 3.31e-8 | 0.299 |
| 14 | noa_nabeshima | 1.00e-8 | 3.57e-8 | 0.280 |
| 14 | huang_chung_yi | 1.00e-8 | 2.70e-8 | 0.371 |
| 19 | AndreasHad04 | 1.22e-8 | 9.57e-8 | 0.127 |
| 22 | mliston | 1.44e-8 | 3.16e-8 | 0.455 |
| 26 | ArtemVoronov | 1.83e-8 | 2.33e-8 | 0.787 |
| 38 | trim_qewas | 3.74e-8 | 2.73e-7 | 0.137 |

Per-submission telemetry (min/median/max over 100 MLPs; wall = participant_predict_wall_s; kern = flopscope_server_kernel_wall_s; ovh = flopscope_server_overhead_wall_s (~0.36 ms/op => op count ~ ovh/0.00036); res = residual_wall_time_s, cap 0.4):

| sub | who | adj | raw | all-layers | FLOPs (e9) | util | wall s | kern s | ovh s | res s | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 329857 | marius_binner | 3.83e-9 | 1.87e-8 | 7.38e-9 | 450.4 const | 0.205 | 70/74/79 | 60/64/69 | 9.5/9.8/10.3 | .286/.291/.306 | setup 1.1 s; ~27k ops; 7 GFLOP/s |
| 329895 | suliman_tadros | 3.80e-9 | 2.03e-8 | 8.13e-9 | 389/410/431 VAR | 0.186 | 41/44/52 | 36/39/46 | 4.9/5.1/5.7 | .119/.124/.136 | adaptive FLOPs per MLP |
| 329787 | joe_wanza | 3.93e-9 | 2.02e-8 | 1.07e-8 | 421/428/436 VAR | 0.195 | 99/104/119 | 88/93/106 | 10.5/11/12.3 | .337/.354/.389 | at both caps (120 s, 0.4 s) |
| 329816 | andrew_epstein (Puffi) | 3.88e-9 | 2.13e-8 | 8.62e-9 | 387/397/431 VAR | 0.181 | 86/91/98 | 79/83/89 | 6.8/7.1/12 | .181/.193/.345 | |
| 329737 | oqaris | 4.30e-9 | 2.06e-8 | 7.69e-9 | 460 const | 0.209 | 86/91/109 | 74/79/95 | 11.6/12.2/13.4 | .315/.329/.363 | |
| 329501 | emil_partow (Activation-Regions) | 5.39e-9 | 2.80e-8 | 3.4e-4 | 403/422/422 | 0.192 | 102/108/120 | 90/96/106 | 1.3/12.5/13.6 | 0/.351/.377 | 8 non-completed rows; hits 120 s cap |
| 329689 | fklassen | 5.44e-9 | 2.50e-8 | 9.98e-9 | 473/479/483 | 0.218 | 80/83/90 | 69/71/78 | 11.1/11.6/12.2 | .307/.321/.341 | |
| 329812 | hyojun_kwon | 6.41e-9 | 2.83e-8 | 1.03e-8 | 499 const | 0.227 | 82/87/91 | 74/79/83 | 8.1/8.4/8.6 | .231/.237/.241 | setup 1.3 s |
| 329885 | lode_dockx | 7.06e-9 | 4.78e-8 | 0.699 | 312/325/338 | 0.148 | 32/34/36 | 26/27/29 | 6.6/6.8/7.0 | .186/.192/.200 | final-row only |
| 329644 | 504aldo | 7.82e-9 | 2.13e-8 | 8.70e-9 | 806 const | 0.367 | 31/32/37 | 29/30/34 | 2.2/2.3/2.7 | .059/.062/.072 | few big ops, 27 GFLOP/s |
| 329762 | Camaro | 8.70e-9 | 2.73e-8 | 1.03e-8 | 678/700/723 | 0.318 | 54/59/63 | 45/50/53 | 8.4/8.9/9.4 | .272/.293/.310 | |
| 329375 | jamespayor | 9.41e-9 | 1.96e-8 | 1.00e-8 | 1060 const | 0.482 | 16/16.6/17.3 | 12/12.5/13 | 3.9/4.0/4.2 | .099/.101/.105 | 85 GFLOP/s dense-matmul profile; best raw on board |
| 329691 | noa_nabeshima | 1.00e-8 | 3.57e-8 | 1.38e-8 | 591/615/641 | 0.280 | 62/69/74 | 56/63/68 | 5.6/6.0/6.4 | .146/.158/.168 | |
| 329874 | AndreasHad04 | 1.22e-8 | 9.57e-8 | 7.13e-8 | 280 const | 0.127 | 98/103/108 | 92/97/101 | 6.4/6.7/7.0 | .159/.165/.172 | 2.9 GFLOP/s: wall-expensive, FLOP-cheap ops |
| 329859 | ArtemVoronov | 1.83e-8 | 2.33e-8 | 8.78e-9 | 1730 const | 0.787 | 44/49/51 | 39/43/44 | 5.6/5.9/6.1 | .151/.161/.165 | |
| 329794 | trim_qewas | 3.74e-8 | 2.73e-7 | 2.05e-7 | 291/301/312 | 0.137 | 24/25/27 | 17/19/20 | 6.2/6.4/6.6 | .162/.167/.172 | |
| 329790 | mliston | 1.44e-8 | 3.16e-8 | 1.26e-8 | 985/1000/1040 | 0.455 | 77/81/85 | 69/73/77 | 7.5/7.8/8.1 | .208/.221/.229 | |
| 329879 | huang_chung_yi | 1.00e-8 | 2.70e-8 | 1.02e-8 | 815/816/818 | 0.371 | 87/92/102 | 80/84/94 | 7.1/7.4/7.9 | .196/.203/.213 | |
| 329639 | neilarmstrong4 | 9.90e-9 | 3.31e-8 | 1.28e-8 | 658 const | 0.299 | 87/93/110 | 77/83/98 | 9.4/10/11.7 | .344/.360/.398 | at residual cap |
| 329883 | SaltyTaro | 1.38e-8 | 2.99e-8 | 1.17e-8 | 1010 const | 0.459 | 33/35/36 | 29/31/32 | 3.9/4.0/4.2 | .102/.106/.109 | |
| 329896 | US canary4 | 8.50e-8 | 8.50e-7 | — | 168.7 const | 0.077 | 3.5/3.6/3.9 | 2.5 | ~1.0 | .029/.030/.031 | 2,673 ops, 70 GFLOP/s |

Grader stack on all: whestbench 0.16.1 / flopscope 0.12.1.

Observations:
- Entire top-8 cluster: util 0.18-0.23, wall 70-120 s (many at the 120 s cap), residual 0.29-0.39 s (right under the 0.4 s cap), overhead 10-13 s => ~25-35k flopscope ops per MLP, kernel throughput 5-7 GFLOP/s. This is NOT a dense-matmul profile (jamespayor's dense profile runs 85 GFLOP/s). It is a profile of very many small/cheap-per-FLOP operations and heavy Python orchestration -- consistent with chunked/sample-loop or per-neuron/per-block routing, sorting/gather, and elementwise-heavy work whose wall cost flopscope does not price (lambda=0 in Phase 2).
- suliman_tadros / joe_wanza / Puffi have VARIABLE FLOPs per MLP -> data-adaptive spend (per-MLP or per-neuron routing).
- All-layers MSE < final-layer MSE for all leaders (7-11e-9 vs 1.9-2.8e-8) => their intermediate rows are accurate: a depth-propagating chain, not a final-row-only method. (Our canary4 shares the signature: all-layers 4.07e-7 vs final 8.50e-7; v2.7 7.97e-7 vs 1.37e-6.) lode_dockx (all-layers 0.70) is the exception: final-row-only at raw 4.78e-8 for 0.148 util.
- The raw floor across the whole top-12 is 1.87e-8..2.1e-8 regardless of util from 0.18 to 0.79 (ArtemVoronov at 0.787 gets 2.33e-8). => a shared bias floor ~1.8-2e-8, NOT a variance floor. Whoever breaks that floor wins outright.
- AndreasHad04: 2.8e11 FLOPs but 103 s wall at 2.9 GFLOP/s => an extreme example of FLOP-cheap/wall-expensive operations (gather/sort/elementwise). Raw 9.6e-8 at util 0.127.
