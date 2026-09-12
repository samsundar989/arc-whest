# First-source-only weighted-discrete K4 cost: scoped PASS

Contract: `experiments/LOG.md` commit `cffc4d3`. This is truth-free instrument
evidence for one chosen discrete first source and its declared quadratic K3
feed. It uses neither a decision seed block nor challenge labels. No later K4
source, estimator, score, or server residual is represented.

The independent n=4 oracle enumerates the full centered fourth tensor and
full cubic feed tensor. It checks birth and two future nonconstant signed maps,
then separately compares the executing kernel's D4, D22, D31, D3 and oriented
D21 readouts. All two focused tests pass; see `focused_test.log`. The D21
symmetry warning is expected because D21[i,j] = F[i,i,j] is generally
asymmetric. The registered relative `1e-5` and near-zero absolute `1e-6`
tolerances are used throughout.

Workerq #986 used the live working tree with 8 GiB RAM, 2 CPUs and no GPU.
It succeeded at the frozen kernel SHA256
`be73d8d0f35c050adb0c27a2c59fb13cb41dd79b81974ba1c6307574774fabdd`.
The exact meter is `meter.json`, its source hash matches, and its 4,729 atomic
operations sum through every named family to **704,164,664,384 counted FLOPs**
(`0.3202170 * 2^41`). Dense products account for **702,496,398,336**; all
other counted operations account for **1,668,266,048**. The dense total is
0.10754% below the frozen `703,252,660,224` conventional prediction, within
the registered 5% gate. Flopscope counts `2mkq-mq` per dense product where the
conventional prediction used `2mkq`; the difference in dense billing is
756,261,888 FLOPs. The negative D21 pair of products bills
**390,651,183,104**, above the registered 350 billion threshold.

| Dense product family | Counted FLOPs |
|---|---:|
| Source map | 134,152,192 |
| K4 transport | 2,012,282,880 |
| K4 covariance Gram | 1,864,368,128 |
| K4 positive D22/D31 Grams | 3,728,736,256 |
| Shared A transport | 225,375,682,560 |
| Feed P/a transport | 14,306,073,600 |
| Feed Q reconstruction | 13,982,760,960 |
| Feed A+ reconstruction | 14,085,980,160 |
| Feed H Grams | 12,118,392,832 |
| Positive D21 products | 24,236,785,664 |
| Negative D21 products | 390,651,183,104 |
| **Total** | **702,496,398,336** |

The executed schedule is 15 K4 uses (14 full, one final D4-only), 14 K3
births, 105 future K3 transports/readouts, 91 full D21 readouts, 14 final
D3-only readouts, and 105 A transports. The single-source path has one feed
per birth, so sharing A among multiple K4 ages is not tested.

Workerq measured **889.883 MiB peak process RAM**. The ledger declares
134,988,288 bytes of external synthetic inputs, an estimated 114,089,984
bytes of persistent factors plus readout scratch, and a 4,194,304-byte
largest instrumented array. Its scratch allowance is a conservative array
payload plan, not a measured allocation peak. Windows does not enforce the
challenge's 8 GiB address-space limit, so the observed process peak is the
registered memory evidence. Original queue log, manifest, environment,
result and terminal receipt are copied byte-for-byte in `job986_raw.zip`;
`verification.json` checks their hashes and all ten frozen cost/schedule gates.

The synthetic input generator supplies raw nodes, positive weights, signed
dense maps and nonconstant response coefficients outside the meter, analogous
to externally supplied estimator inputs. Within the meter, source centering,
one source map, all factor formation and transports, both positive and
negative consumers, casts, output accumulations and final shortcuts are
billed. It does not price distributional integration or quadrature needed to
obtain a physical ReLU source. The ordinary K3 backbone's previously measured
818,494,496,296 FLOPs come from a different executable; shared work has not
been reconciled, so the two totals cannot be added. Local runtime is not
server residual authority. The repository's seed consistency/static scan
passes (`seed_scan.log`).

Workerq #987 passed prior-test-byte default preflight 12/16 with four Full-only
steps skipped; its original queue files are preserved in `job987_raw.zip`.
The full-tensor test was added afterward. A final current-byte
preflight and independent Skeptic verdict are still required before using this
cost result beyond this instrument's scope.
