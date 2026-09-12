# Truth-free K4 dense reference, 2026-09-12

Contract: `experiments/LOG.md` at `fbedd3b`; design memo and independent
correction: `repair_architecture.md` at `a8bb5e1` and
`repair_architecture_skeptic.md`. Run from the repository root:

```
uv run python local_sim/test_p2_k4_source_dense.py --output experiments/2026-09-11_stage2_compact_p07/source_dense/result.json
```

`result.json` records **PASS, 49 checks**. The largest relative error for
entries with absolute expected value above `1e-12` is `3.07447546077247e-14`;
the largest absolute error on zero/near-zero entries is
`5.551115123125783e-17`. Both meet the preregistered `1e-12` tolerances.
At n=4 and n=5, independent Gaussian ReLU coordinates start with a dense
diagonal fourth cumulant. A four-index dense contraction under signed,
nonconstant transport agrees entry by entry with the separate `(Q,k)` factor
path, all D4/D22/D31 slices, the symmetric oriented off-diagonal K11
covariance term, and all quadratic-truncation K3 feed entries, including
repeated indices. Signed fixtures expose the erroneous use of D31 without
transposition. A correlated Gaussian control uses the exact Isserlis identity
to set K4 and its K4-induced feed to zero; this is algebra, not empirical
evidence about nonlinear closure.

The enumerated `T(G,M)` tensor agrees with both-matrix transport and all
slice/feed factor paths. Keeping `M` fixed under a signed map changes the
fourth tensor by as much as `43.71184317211828` in this fixture; a shortcut
using the transported covariance off-diagonal in place of transported `G`
changes it by `12.622970412518244`. Zero G, permutations and signed `tI`
scaling pass. A singular rank-2 metric recovers G only on its support; the
cross-support residual has entry `E_0002=0.4`, and carrying it changes a
future D4 by `0.125`. An all-distinct symmetric residual has exactly zero
current D4/D22/D31 slices but becomes future D4 `1.5` under the prescribed
row. Present slices cannot replace a carried residual.

Workerq job **962** ran the default current-byte preflight from HEAD
`54925bc` with `--no-snapshot --live-worktree --ram 5 --cpus 2 --gpus 0`.
Its terminal result is **SUCCEEDED**, exit 0, **PREFLIGHT GREEN 12/16**, four
Full-only steps skipped, with measured peak **3592.53515625 MiB**. The log,
manifest, environment and result below are byte-for-byte copies from the
queue. `job962_raw.zip` contains those four exact files; every ZIP member
was compared to its copied file, and every copied file's SHA-256 matches the
queue original. The readable log can be newline-normalized by Windows Git;
the ZIP preserves the original bytes. `job962_terminal_show.json` records
the terminal peak and live snapshot mode.

| File | SHA-256 of disk bytes before commit |
|---|---|
| `local_sim/test_p2_k4_source_dense.py` | `1bf120cd5f4e53e4c72d6578b7ea742221542881f22f63cb5278933d84627b06` |
| `result.json` | `cd5b217732970d52ae5c5f053288480bf26a2b1aabc01863b1f592940fd2187f` |
| `job962_live_preflight.log` | `82d211b1b1ca7412312b3a2388818dbaf0f73e89198f4a68ad7704395745bc90` |
| `job962_live_manifest.json` | `0cc67d5bbefdc1fbefa34f6f98a88c36d22065749477c2bd17987ebd09217935` |
| `job962_live_environment.json` | `acbf4293111b56537db2fbd9e388c2acac5258c458ad4ed0b13f273639870c24` |
| `job962_live_result.json` | `21f864bf5d07db629887a52b1c8f2028d1688f5a9146af70518c16b5180476ab` |
| `job962_terminal_show.json` | `1cd7c3f3f6a73c27e0b85d3e21fe5117fd5d91bf225237cf794023c4b88a4bf9` |
| `job962_raw.zip` | `3c6d4e140775d570204a240e08fa66ca1c8273235f40363a4a5c27be479744e9` |

This result establishes these selected tensor identities and exposes two
invalid transport shortcuts. It does not establish a complete physical
mixed-ReLU source, efficient compression, counted flopscope cost, accuracy,
memory-limit compliance or server residual headroom. No P01/P07/vault truth,
registry seed block, scored estimator, frozen arm or source template was read
or changed by the test.
