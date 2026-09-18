# Independent Skeptic: P12 complete-package factorial result

## Skeptic verdict: ACCEPT

**Scope.** ACCEPT the six-arm P12 **DESIGN** arithmetic, frozen C10 selection, and stopping the conditional complete-package fit. This is exposed-panel selection, not a fresh confirmation, release certification, or a P13 decision.

**Strongest challenge.** The apparent C11 improvement over C00 could invite a combined-mechanism claim, but P3 harms the fitted compressed arm. C10 official mean is 6.4866090e-9; C11 is 6.8083424e-9. Their paired official control-minus-candidate difference is -3.2173e-10, t95 approximately [-5.2547e-10, -1.1799e-10]. The factorial interaction, C11-C10-C01+C00, is +6.2390e-10 in official adjusted-score units, opposing an additive gain. C11 versus C00 passes only a weaker direct screen; it cannot replace the stronger C10 incumbent.

**Evidence for the original result.** I independently read `score.json`, `decision.json`, all six prediction sidecars and their manifest, the corrected `2026-09-18_s17_meter_grader_twin.json`, the immutable job-1882 snapshot at commit `7f374d2df6a24eeb6f1f9d9386f9a61eb9a379f0`, the registered LOG contract, and `s17_complete_package_decide.py`. Direct `design_sidecar.verify` with that snapshot root and full row/source/truth/dependency binding accepted six arms of 24 rows. The separate scalar stream has 144 rows. Score and manifest SHA-256 match the decision's `9048ac53...` and `a72ace62...`; all six snapshot source byte hashes match the score, manifest, reader role pins and decision. Each arm has the same ordered 24 P12 seeds, zero saved failures, and mean truth-noise floor 2.2480710e-9. The truth SHA is the registered `92a3228e...`. The corrected meter SHA is `fb96f44f...`; its six complete FLOP totals match the score rows exactly.

I recomputed per-network observed score as MSE times `max(0.1,F/2**41)` and official-precision score after replacing each row's truth-noise contribution by `noise * (2*2**24/1e9)`. Independent paired arithmetic t intervals reproduce the reader within the rounded t critical used in my calculation:

| Control → candidate | Observed mean difference | Official mean difference | Official t95, approximate | Official gate 1.0 |
|---|---:|---:|---:|---|
| C00 → C10 | +7.3837e-10 | +7.3841e-10 | [+6.1618e-10,+8.6065e-10] | pass |
| C00 → C01 | +2.4386e-10 | +3.0217e-10 | [+1.0123e-10,+5.0310e-10] | pass |
| C10 → C11 | -3.8005e-10 | -3.2173e-10 | [-5.2544e-10,-1.1803e-10] | fail |
| C01 → C11 | +1.1446e-10 | +1.1451e-10 | [-6.864e-12,+2.3589e-10] | fail |
| C00 → C11 | +3.5832e-10 | +4.1668e-10 | [+1.3831e-10,+6.9504e-10] | pass |
| C10 → C01 | -4.9451e-10 | -4.3624e-10 | [-6.1009e-10,-2.6240e-10] | fail |

Observed verdicts have the same pass/fail pattern. The registered visit order C00→C10→C01→C11 selects **C10**: C10 passes against C00; C01 and C11 both have higher official means than C10 and fail their direct C10 comparisons. Same-panel v31 means are 9.5146842e-9 observed and 8.8606652e-9 official; half targets are 4.7573421e-9 and 4.4303326e-9. C10 would need 31.664% and 31.700% respective reductions. Both exceed the preregistered 25% fit trigger, so the complete-package fit branch is closed. C10 versus v31 is only about 1.367x and 1.366x on these means, below the 2x release target.

FLOPs are C00 480.804B, C10 480.851B, C01 539.825B, C11 539.872B, dense sentinel 621.352B and v31 661.962B per MLP. Matched P3 cost is +59.021B. Raw mean MSEs are in `decision.json` beside the common 2.248071e-9 truth-noise floor. The run had disabled wall caps; neither zero saved failures nor modeled residual margin proves server feasibility. Standard snapshot-reader preflight is GREEN 12/16; four Full checks were skipped.

**Snapshot-root repair provenance.** The first strict read refused before interpretation because live checkout dependency bytes and the score's snapshot-relative sidecar path did not match the job root. Contract commit `27c815e` predicted a root-only repair; `cee6343` changed the reader and tests, with a separate fresh-context ACCEPT in `2026-09-18_s17_snapshot_reader_skeptic.md`. The score, source, sidecar, gate and scorer bytes were not changed by that repair. I used the immutable snapshot root, not live source bytes, for direct sidecar verification. Historical pre-repair refusal remains valid evidence of a fail-closed instrument, not a scored failure.

**Unresolved ambiguity.** P12 has been deliberately reused and is correlated with earlier choices. It cannot estimate fresh-panel reliability. Local disabled-cap timings and Windows memory accounting do not settle server residual or Linux 8 GiB feasibility. The registered single query-cost intervention remains a separate conditional question; this factorial result does not prejudge it.

**Cheapest discriminating follow-up.** Follow the already registered single rank32 old-D21 query cost screen on selected C10, using a label-free workerq audit first. Only if its complete counted-FLOP saving and numerical screen pass, spend the allowed paired P12 DESIGN accuracy comparison. Do not fit the complete package or spend P13 on this factorial result.

**Checklist items triggered.** 1, 2, 3, 4, 5, 6, 8, 10, 11, 13, 14, 17 and 18. The 24-network t intervals support narrow within-panel decisions only; multiple correlated screens and prior panel reuse limit extrapolation. No public or vault selection occurred in this audit.
