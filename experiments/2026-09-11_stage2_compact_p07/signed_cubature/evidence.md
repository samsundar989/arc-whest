# Truth-free signed cubature factor fixture, 2026-09-12

Contract: `experiments/LOG.md` at `47d5061`. Run from repository root:

```
uv run python local_sim/test_p2_k4_signed_cubature.py --output experiments/2026-09-11_stage2_compact_p07/signed_cubature/result.json
```

`result.json` reports **PASS 83/83** at the fixed registered gates. For each
`rho=0,0.5`, the fixture integrates a complete signed Gaussian product grid,
splitting each coordinate into `[-10,0]` and `[0,10]` with 32 or 48 Legendre
nodes per half. It rectifies every node, including the axis/origin ReLU mass,
and normalizes the discrete weights. Weighted centered columns and their own
covariance produce a fourth cumulant by a rank-one fourth sum minus all three
covariance pairings. Independently enumerated set partitions of raw moments
from the **same weighted measure** match the factor tensor at both orders.
The maximum absolute error among the source, two-map, slice, covariance, K3
feed, and carried metric-residual identity checks is `3.5527136788005e-15`,
below `1e-10`. The largest material relative error across order convergence
and the separate order-96 continuous positive-quadrant reference is
`2.7956944288819e-14`, below `1e-8`.

At `rho=0`, mixed fourth cumulants are numerical zero. At `rho=0.5`, the
order-48 weighted-measure values are `K1112=0.11417442414038367`,
`K1122=0.10914008267227016`, and `K1222=0.11417442414038381`. The factor
and dense paths agree after the registered Hadamard map and a second signed,
nonconstant map. The second map makes D31 asymmetric; using D31 in place of
its transpose in one covariance term changes the off-diagonal result by
`0.004864073526488134` at `rho=0.5`. The metric core carries its full
residual through both maps; it is not claimed as a lossless compression alone.

At `rho=0.5`, omitting covariance subtraction changes a source entry by
`0.3485260584560599`. Retaining only positive-quadrant nodes changes a
source entry by `0.19704650026257436` and a future signed-map D4 by
`0.3470734820778318`. These pass the preregistered `>1e-3` discriminators.

Workerq job **971** ran current live bytes from HEAD
`54258abb05ced7ddb53b3ea3e9e9a074da0a780f` with
`--no-snapshot --live-worktree --ram 5 --cpus 2 --gpus 0` and the default
`scripts/preflight.ps1`. It ended **SUCCEEDED**, exit 0, **PREFLIGHT GREEN
12/16**, four Full-only steps skipped, and measured peak `3720.86328125 MiB`.
HEAD and the script/result hashes remained unchanged through terminal. The
default preflight does not execute this standalone oracle; its focused rerun
reproduced `result.json` byte for byte. The four queue files below are exact
copies of their queue originals, and every member of the ZIP equals its disk
copy byte for byte.

| File | SHA-256 of disk bytes before commit |
|---|---|
| `local_sim/test_p2_k4_signed_cubature.py` | `4ffd2704c0dc8f550f41f1327b28e0966d95ae7255c9a95545a2969a527b5db7` |
| `result.json` | `2ac1f3c15e52a048ec6dffce14d26936fa300c9a8cb5da9f5fd6abf3612410d5` |
| `job971_live_preflight.log` | `e4f7500e4ab365cdebc8ed826e63eadefbcf34a4487d10daf7b52d47f8ceec27` |
| `job971_live_manifest.json` | `b2b892c01d5cae60ec9898628316205f737af3f848fdb7b0f8fe8166ae476877` |
| `job971_live_environment.json` | `95f3a90fe9ad1cb72db3485e5795bbee0e973bc197ee2e0f0c14b6fca51512d5` |
| `job971_live_result.json` | `d52d00195344e6f3c735d63c38b2f97c8631ff86cd3acddd836ad446baa8c989` |
| `job971_terminal_show.json` | `b95a55851606f01cb638344d90465da01d620975ea0992a0b29a5845899bdafc` |
| `job971_raw.zip` | `d61142f3bd0463d1da0ada11a4228e140742cac927e81d3679337f24c5af9395` |

This establishes an exact weighted-discrete-measure cumulant representation
and its tiny signed linear transport, with quadrature agreement for the tested
Gaussian pair. It does not establish a small factor count at challenge width,
later non-Gaussian sources, counted flopscope cost, deployed memory or residual
headroom, or accuracy. No labels, seed blocks, frozen arms, shipped estimator,
or vault were touched.
