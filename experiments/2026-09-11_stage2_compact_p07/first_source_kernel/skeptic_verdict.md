## Skeptic verdict: ACCEPT

I accept the package for the preregistered scope: one externally supplied, weighted-discrete first source; exact K4 transport; and its declared quadratic K3 feed at `n=1024`, `depth=16`, `S=64`.

**Strongest challenge**  
The evidence does not establish a complete estimator. Raw NumPy creates the synthetic nodes, maps, and coefficients outside the meter; physical ReLU source construction is absent. Job #986 also used a live worktree, and the ZIP preserves queue records but neither source bytes nor the 4,729 individual `op_log` entries. Exact reproduction from the committed source substantially addresses the byte-provenance concern, but server behavior remains unseen.

**Independent checks**

- The dense K4 oracle explicitly subtracts all three covariance pairings. The quadratic K3 factorization reproduces the positive and negative tensor placements, including asymmetric `D21[i,j] = F[i,i,j]`.
- I reran both committed tiny tests without bytecode writes. Both passed, covering the full K4/K3 tensors, D4/D22/D31/D3/D21 consumers, two signed transports, and the registered `rtol=1e-5`, `atol=1e-6` thresholds.
- The schedule follows directly from the layer indices:

  - K4 uses: `15`
  - K3 births: `14`
  - Feed transports/readouts: `14+…+1 = 105`
  - Full D21 readouts: `13+…+1 = 91`
  - Final D3 readouts: `14`

- The conventional dense prediction independently recomputes as  
  `646(2n²S) + 105(2n²) + 105(2n³) + 182(2n³)`  
  `= 703,252,660,224`.
- Applying flopscope’s exact `2mkq−mq` matmul accounting gives:

| Quantity | Verified value |
|---|---:|
| Dense subtotal | 702,496,398,336 |
| Nondense subtotal | 1,668,266,048 |
| Total official-package FLOPs | **704,164,664,384** |
| Dense difference from conventional | −756,261,888, or −0.10754% |
| Two dense negative-D21 products | **390,651,183,104** |
| Low-level operations | 4,729 |

Every dense family independently matches its expected matrix dimensions and call count; all family entries sum exactly to the reported total. A read-only live-shape rerun reproduced every figure exactly.

**Memory**  
Workerq #986 recorded **889.8828125 MiB** peak process RAM under an 8,192 MiB reservation. It had only two usage samples, so this is a sampled observed peak. The analytical payload is also small: 134,988,288 external-input bytes, 114,089,984 estimated internal array bytes, and a 4,194,304-byte largest array. Windows did not enforce an 8 GiB address-space limit.

**Preflight and provenance**

- Source SHA-256: `be73d8d0f35c050adb0c27a2c59fb13cb41dd79b81974ba1c6307574774fabdd`
- Test SHA-256: `a4f649852997831deff3c78ec53b3a9068f86db1c80729d73dfe5c38b2f31961`
- Both files remain byte-identical to commit `076031f`.
- I independently rehashed every member of the #986, #987, #988, and #990 ZIPs; all match their verification records.
- #987 is green but predates the final test bytes, so I did not use it to validate that test.
- #988 is genuinely **RED**: four absent artifacts, 22 snapshot byte/hash mismatches, and an estimator/report SHA mismatch caused steps 7, 9, and 12 to fail. It supplies no green evidence.
- #990 reached terminal **SUCCEEDED**, exit `0`, with **PREFLIGHT GREEN 12/16**. Its final log shows exact 19-operation anchor parity. The four skipped checks are explicitly Full-only deployment checks.

**Accepted scope**  
`704,164,664,384` is a local official-flopscope bill for this one-source research kernel. It cannot be added to the separate `818,494,496,296` supplied-core baseline without overlap analysis. It proves no later-source construction, all-history architecture, ordinary K3 backbone, S64 rank adequacy, server residual headroom, deployment memory compliance, MSE, or competition score. NumPy is legal in `local_sim`; this file itself is not a shippable estimator.

**Raw paths**

- [Kernel](C:/Users/samsu/Documents/arc-whest/local_sim/p2_k4_first_source_kernel.py:70)
- [Independent tiny tests](C:/Users/samsu/Documents/arc-whest/local_sim/test_p2_k4_first_source_kernel.py:54)
- [Meter ledger](C:/Users/samsu/Documents/arc-whest/experiments/2026-09-11_stage2_compact_p07/first_source_kernel/meter.json:1)
- [Job #986 receipt and memory](C:/Users/samsu/Documents/arc-whest/experiments/2026-09-11_stage2_compact_p07/first_source_kernel/job986_terminal_show.json:42)
- [Job #986 raw ZIP](C:/Users/samsu/Documents/arc-whest/experiments/2026-09-11_stage2_compact_p07/first_source_kernel/job986_raw.zip)
- [Job #988 RED log](C:/Users/samsu/Documents/arc-whest/experiments/2026-09-11_stage2_compact_p07/first_source_kernel/job988_log.log:469)
- [Job #990 terminal GREEN log](C:/Users/samsu/Documents/arc-whest/experiments/2026-09-11_stage2_compact_p07/first_source_kernel/job990_log.log)
- [Job #990 raw ZIP](C:/Users/samsu/Documents/arc-whest/experiments/2026-09-11_stage2_compact_p07/first_source_kernel/job990_raw.zip)

**Cheapest follow-up**  
Rerun the approximately four-second microkernel from a git snapshot while archiving the exact source, test, meter, and raw `ctx.op_log` in one content-addressed ZIP. This needs no seed block, labels, public data, or vault access.

**Checklist items triggered:** 6, 8, 9, 14, 17, 18.