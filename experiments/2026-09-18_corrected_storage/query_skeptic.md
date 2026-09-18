# Independent Skeptic: corrected-storage label-free query audit

## Verdict: REJECT the registered query subcondition

The primary old-qE to qEc comparison fails the contract's required reduction in complete recomputed E error and non-deterioration in D3, D21, next-state, and final-output displacement. All six registered aggregate errors worsen. The secondary old-qEd to qEdc diagnostic also worsens all six. This verdict concerns the query subcondition only; neither artifact contains truth labels or a P12 adjusted score.

**Raw evidence:** `experiments/2026-09-18_corrected_storage/query_old.json` and `experiments/2026-09-18_corrected_storage/query_corrected.json`. I read those artifacts, the corrected-storage contract in `experiments/LOG.md`, and `local_sim/s10_parity.py`, `local_sim/s12_decide.py`, and `local_sim/s16_corrected_storage_decide.py`. I did not inspect the active fit, score, P13, public, or vault results.

**Identity and coverage.** Both artifacts identify `s10_parity.query_error` with `label_free: true`, width 1024, depth 16, seed 90000016001, and overrides age 4/rank 256. Their uncompressed bills are equal at 620,599,862,355 FLOPs. Their complete qoff arm is identical, including its 480,624,740,791-FLOP bill. The old arms name the frozen qoff, qE, and fixed-stack old-qEd sources; the corrected arms name the same qoff, qEc, and qEdc sources. All five recorded source SHA-256 values match the current source files. The strict `_query()` reader accepts both artifacts. Each arm has layers 0–15; E and D21 are present on layers 1–14, D3 on 1–15, output mean on 0–15, and output covariance on 0–14. The final-row field is finite. Each arm records 11 retained-energy joins on layers 4–14.

| Registered relative error | Old qE | qEc | Old qEd | qEdc |
|---|---:|---:|---:|---:|
| Complete E, mean over eligible layers | 0.0247134803 | 0.0296859909 | 0.0276989035 | 0.0328282895 |
| D3 | 0.0135847348 | 0.0168040579 | 0.0156144105 | 0.0187655889 |
| D21 | 0.0199954988 | 0.0242806252 | 0.0225193072 | 0.0269596113 |
| Next-state output mean | 0.0000353907 | 0.0000450023 | 0.0000392548 | 0.0000485220 |
| Next-state output covariance | 0.0018414444 | 0.0022668834 | 0.0020815118 | 0.0025182409 |
| Final output row | 0.0000740522 | 0.0001030482 | 0.0000829222 | 0.0001110489 |

Primary E worsens 20.1%, D3 23.7%, D21 21.4%, mean 27.2%, covariance 23.1%, and final row 39.2%. For every registered per-layer metric, no active layer improves in either comparison; unchanged early layers precede the compression joins. `_query_compare()` requires qEc E to be strictly lower than old qE and each of the other five metrics to be no higher. It therefore returns false without needing a score artifact.

The favorable measure is **same-state retained query energy**: old qE 0.9992387137 to qEc 0.9996035097, and old qEd 0.9989999155 to qEdc 0.9994509269. This measures the immediate basis action at joins. The complete recomputed E and continuation fields show that this local gain did not carry through the recurrent chain on this network. It is the strongest competing explanation for a claim based on retained energy alone.

**Counted cost.** Old qE bills 491,609,541,047 FLOPs; qEc bills 491,635,245,495, an increase of 25,704,448 or 0.00523%. Old qEd bills 492,373,330,359; qEdc bills 492,410,569,143, an increase of 37,238,784 or 0.00756%. These are complete prediction FLOP bills from the same 1024×16 query instrument, not a residual-time certificate or P12 score.

**Inference limit and next decision.** The artifacts cannot determine whether qEc wins or loses P12 truth-based adjusted score; they contain no labels, ground-truth noise floor, or paired score interval. They also do not independently prove or disprove the intended post-Wick algebra: a corrected algebraic action can still allocate rank in a way that worsens complete recurrent error. Under the preregistered contract, failure of either the score or complete-query condition rejects this storage mechanism; the primary query failure already supplies that falsifier. No further experiment is needed to decide this subcondition. A different storage mechanism would need a new contract and DESIGN evidence.

**Checklist items:** 8 (counted cost kept separate from residual), 14 (one label-free query instrument), 17 (frozen source and raw artifact identity). Statistical-score and truth-noise items do not apply to this artifact.
