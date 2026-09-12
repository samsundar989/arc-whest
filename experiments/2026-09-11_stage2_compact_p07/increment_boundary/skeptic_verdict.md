## Skeptic verdict: REJECT

**Strongest challenge**  
The registered combined convergence gate fails exactly as its falsifier defines. Of 67 registered 32/48 comparisons, **48 fail** the `1e-8` gate. For the increment tensor:

- At `[1,1,1,1]`: N32 `0.0022470592875341033`, N48 `0.0022470790781579097`; absolute difference `1.9790623806414942e-8`, material-relative difference **`8.807266285723563e-6`**—about **881×** the gate.
- Maximum increment absolute difference: **`4.47939434433664e-8`** at `[1,1,1,0]`.
- Largest absolute 32/48 failure overall: `raw_U_04`, **`2.899732760575091e-7`** (`2.768507528101958` versus `2.768507238128682`).

The final result is therefore correctly **FAIL: 640/688 pass, 48 fail**. The preregistered falsifier says any numerical threshold failure rejects the combined gate.

**Evidence for the original claim**  
The boundary partition substantially repairs the earlier numerical method, but does not satisfy the registered 32/48 requirement:

- All **67** registered 48/64 comparisons pass. Worst material-relative difference is `6.583237642021994e-9`.
- N48 versus the conditional reference reaches `6.588178380515773e-9`; N64 reaches `4.940738526304726e-12`.
- The reference’s reported quadrature error is `9.847484958013634e-13`; its loose-versus-tight tolerance comparison differs by at most `7.185058279713619e-16`.

The mapped-axis diagnosis is supported independently. For the `X_04` axis contribution:

- N32: `0.18987757012046474`
- N48: `0.18987749639418858`
- N64: `0.18987749632528955`
- Independent continuous integral: `0.18987749632524198`

The quadrant contribution remains approximately `1.310122503674759`, isolating the order-32 error to the mapped-axis integration.

Mass and structural claims pass:

- Nodes: exactly `353`, `529`, `705`.
- Maximum pre-scaling mass errors: N32 `9.930251065881635e-12`, N48 `8.992806499463768e-14`, N64 `2.7755575615628914e-16`, all below `1e-8`.
- Maximum scaled support-mass error: `5.551115123125783e-17`, below `1e-12`.
- Maximum radial moment defect through degree five: `3.907985046680551e-14`, below `1e-12`.
- My independent reconstruction from raw moments used all 15 fourth-order set partitions and produced 81 transport/readout checks with maximum error `4.579669976578771e-16`. The fixture’s broader same-bank structural maximum is `3.219646771412954e-15`, below `1e-10`.
- At N64, `max|Delta| = 0.08756130902807811`, naive double-count gap `0.6927530692283623`, and identity discriminator `0.27970833835273834`; all exceed the strict `0.001` lower bound. Identity and negative-identity zero controls are exact.

A clean rerun of [the actual fixture](/C:/Users/samsu/Documents/arc-whest/local_sim/test_p2_k4_increment_boundary.py:33) reproduced [result.json](/C:/Users/samsu/Documents/arc-whest/experiments/2026-09-11_stage2_compact_p07/increment_boundary/result.json) byte for byte:

- Source SHA-256: `c9759cea80eda54302a9800ac8f960b0ff5e1ef5c1b4a530a21138d2dd849ceb`
- Result SHA-256: `6ec0a6ee32bd1a277dc352bf81f241c23b5860cd5aa42538e0e57bb9a02f5b18`

The [preregistered contract](/C:/Users/samsu/Documents/arc-whest/experiments/LOG.md:44787) is committed at `f4f1305`, before the source/result commit `46c77a2`. Its gates match the implementation.

The [queue verification](/C:/Users/samsu/Documents/arc-whest/experiments/2026-09-11_stage2_compact_p07/increment_boundary/queue_verification.json), [ZIP](/C:/Users/samsu/Documents/arc-whest/experiments/2026-09-11_stage2_compact_p07/increment_boundary/job985_raw.zip), and every retained ZIP member agree byte for byte. ZIP SHA-256 is `4e9a2aea6a77b0ba22e2b3758d91d9636b9f852fa3bee552b1414968fe5550ff`. Workerq 985 passed default preflight **12/16**, exit `0`, at HEAD `cffc4d3`, with peak RAM `2833.9609375 MiB` against a 5 GiB reservation.

I **ACCEPT** the negative numerical diagnosis, mass controls, finite-measure structural identities, and stated narrow scope. I **REJECT** the combined registered gate.

**Unresolved ambiguity**  
`local_sim/p2_k4_increment_boundary.py`, explicitly named in the request, does not exist at `f4f1305`, `46c77a2`, or in the current worktree. The implementation is entirely in `local_sim/test_p2_k4_increment_boundary.py`.

Workerq 985 ran only [default preflight](/C:/Users/samsu/Documents/arc-whest/experiments/2026-09-11_stage2_compact_p07/increment_boundary/job985_live_preflight.log:1109). It neither executed the boundary fixture nor byte-bound the untracked fixture at job start. Later hashes and exact rerun establish source-to-result reproducibility, but the queue archive itself does not.

The conditional reference is algorithmically independent of the discrete bank, but resides in the same source and shares its model constants and boundary specification. It cannot prove relevance to a real nonlinear history, scalable rank, challenge FLOPs, residual headroom, or accuracy. Those claims are properly excluded. There is no sampling or ground-truth noise uncertainty here; remaining uncertainty is numerical and specification-level.

**Cheapest discriminating follow-up**  
Under a new truth-free IMPORTANT contract, replace only the mapped-axis `u/(1-u)` quadrature with a preregistered split finite-interval/tail or moment-exact positive rule. Hold the quadrant sectors, radial rule, orders, maps, and `1e-8` gate fixed. Report per-support moment contributions before normalization. No seed block, public data, vault, or protected labels are needed.

**Checklist items triggered**  
13 checked: the gate did not move, and its registered falsifier fires.  
14: workerq preflight and the focused fixture are separate measurement paths.  
17: the queue record does not bind the untracked fixture bytes, the result is not yet recorded as a LOG knowledge delta, and the requested non-test source path is absent.  
18: mapped-axis order-32 underresolution is the concrete numerical counterexample.