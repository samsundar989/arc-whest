# Fixed four-coordinate correlated-ReLU fourth source, 2026-09-12

Contract: `experiments/LOG.md` at `6a873ba`, with the design-only method at
`mixed_source/four_distinct_method.md` (`e21c666`). This is a truth-free first-layer
Gaussian/ReLU identity fixture, not a score or candidate experiment. No label,
seed block, frozen estimator, shipped estimator, public split or vault was used.

The fixed covariance is `R=diag(1-a²)+aaᵀ`, `a=(0.4,0.5,0.6,0.7)`. The source
script computes all 70 raw moments through degree four twice: adaptive split
one-dimensional conditional-Gaussian integration, and independently inverted
support-aware marginal-density quadrature on `[0,12]`. The latter uses 32, 48,
and 64 Legendre nodes per active dimension, streams at most 4096 nodes per tile,
and evaluates respectively 1,185,920, 5,764,800, and 17,850,624 density
points over the nonempty supports. A zero-power coordinate is marginalized over
its full Gaussian law, retaining ReLU axis/origin mass. The two independent
fourth-cumulant reconstructions are set partitions of raw moments and centered
fourth moment minus all three covariance pairings.

`result.json` reports **PASS 1088/1088** at the preregistered gates: `K0012 =
0.014686836253094`, `K0123 = 0.010436951764302`, all 12 oriented three-distinct
classes and all 24 all-distinct permutations pass, as do the selected raw
moments, independent integration, signed 4×4 transport, D4/D22/D31 consumers,
and scalar moment reconstructions. The all-distinct term alone changes a
Hadamard future D4 by `0.0156554276464524`; rows with all plus versus one
minus give the required `+1.5 K0123` and `-1.5 K0123` contributions. Maximum
recorded absolute error is `2.66e-15`; maximum material relative error is
`2.68e-13`. The registered magnitude, `1e-8` quadrature and `1e-10` absolute
identity gates all pass without retuning.

The conservative analytic `[0,12]` tail bound is `8.64e-16` per raw moment.
Propagating the observed 48/64 raw differences, adaptive error estimates, that
tail and a `1e-14` roundoff allowance through absolute partition-term
sensitivities gives estimated decisive cumulant error bounds `7.69e-13`
(`K0012`) and `7.04e-13` (`K0123`). These are **estimated quadrature error
bounds**, not rigorous certified bounds: agreement between two finite orders
cannot prove their common error. The optional, non-gating workerq #979
diagnostic found maximum absolute raw changes `2.22e-16` for `[0,10]` versus
`[0,12]` at order 64 and `4.44e-15` for order 32 versus 48. The one-dimensional
reference also agrees between adaptive and separate 72/96 composite schemes.

Workerq **976** ran the substantial fixture with `--no-snapshot --live-worktree
--ram 2 --cpus 2 --gpus 0` and the exact command `uv run python
local_sim/test_p2_k4_four_distinct_source.py --output
experiments/2026-09-11_stage2_compact_p07/four_distinct_source/result.json`.
Terminal state SUCCEEDED, exit 0, measured process peak **1130.9 MiB**. This
exceeded the memo's 1 GiB process target by 106.9 MiB but stayed below the
2 GiB queue reservation. The script SHA-256 was
`4fee7ec37469f281dc7ee444b8e1628e4857aac85296cd43418b0e3cf06dcedd`
at submission and after terminal; the result SHA-256 is
`7b6c163474fed33e0e2bdb83474dba0868e22eb616e712091d7080b244e9dcc2`.
No source edit occurred while job 976 ran.

Workerq **977** then ran current-byte default `powershell -NoProfile
-ExecutionPolicy Bypass -File scripts/preflight.ps1` with `--no-snapshot
--live-worktree --ram 5 --cpus 2 --gpus 0`. Its start log binds HEAD
`e9eeac8a7b20f4a747b344f948cbf84b07c51189` with only the new fixture and
artifact directory untracked; HEAD and the script/result hashes remained fixed
through terminal. It ended SUCCEEDED, exit 0, **PREFLIGHT GREEN 12/16**, with
four Full-only steps skipped, at **3860.6 MiB** peak. Default preflight does not
run this standalone fixture; job 976 is its focused execution. Workerq **979**
ran the optional `tail_diagnostic.py` via the same live-worktree 2 GiB/2 CPU/0
GPU queue path, SUCCEEDED exit 0, peak **1129.1 MiB**. It did not change a gate.

All three jobs' log, manifest, environment and result files are exact byte
copies of their terminal queue originals; every raw ZIP member also equals its
copy and original. `verify_queue.py` recomputes that comparison, and
`queue_verification.json` records each original path, byte count, SHA-256 and
ZIP hash. Selected retained hashes:

| File | SHA-256 |
|---|---|
| `job976_raw.zip` | `43830fe15fc6c26d550f246384b2eb4183ccb41959500ca50ef4f1e3be6c5259` |
| `job977_live_preflight.log` | `fc9e00da76dda8088e39506c0e991304be8b6bdc30765b4a67c3343b4d4fec9d` |
| `job977_raw.zip` | `e2d8abb06abf3412916a9898147632793a16a16ca3aa5e990580ecb11dbce3fa` |
| `tail_diagnostic.py` | `9cdcc2a4dbc124f4868c1e81c69526cf1e64e150f1fbac1b580bab99663651a1` |
| `tail_diagnostic.json` | `c2d8b07972f89ab9da8930c44d94bd27a83a58303e2f5ce77a03966d918424b7` |
| `job979_raw.zip` | `5b2787fd73eecfa6f9850954210feddafdbaaa47d25506a2ebc2c77f00697ee6` |
| `queue_verification.json` | `eb307eb6611ea7c0ee30d84a18b11f5f4e6bd36c7c4c17b6b97ebe69b0bbadcf` |

This establishes only the fixed correlated Gaussian/ReLU fourth source and
tiny full-tensor linear transport at the stated precision. It does not prove a
cheap width-1024 representation, later non-Gaussian closure, score improvement,
counted flopscope cost, deployed 8 GiB compliance, or server residual headroom.
The 8 GiB Windows memory limit remains unenforceable in default preflight.
