# Fixed two-layer increment fixture, 2026-09-12

The registered contract is the tail of `experiments/LOG.md` at `1e99494`.
This is a truth-free, two-coordinate finite-measure check. The focused fixture
returns **FAIL: 300/323 checks pass, 23 fail**. The failure is the registered
32/48 signed-grid convergence gate for quantities affected by the nonconstant
second ReLU. No gate was changed. The result cannot establish quadrature
accuracy for the stated later source.

For the same normalized full-sign `rho=0.5` measure, `X=ReLU(Z)`, the fixed
`B=[[0.8,-0.3],[-0.55,1.4]]`, and `Y=ReLU(BX)`, the script defines the toy old
carry exactly as `B^⊗4 K(X)=K(BX)` and the later increment exactly as
`Delta=K(Y)-K(BX)`. It independently computes all fourth cumulants from raw
moment set partitions. Its signed decomposition has positive centered fourth
factors of `Y`, negative centered fourth factors of `BX`, negative covariance
pairings of `Y`, and positive covariance pairings of `BX`. Both covariance
matrices are transported through two subsequent signed maps with their own
factors. This is an accounting identity for one finite discrete law, not an
efficient or complete nonlinear source model.

All finite-measure identity and structural controls pass. Maximum absolute
identity error is `7.216449660063518e-15` against the fixed `1e-10` gate.
The `B=I` increment is exactly zero, and the `B=-I` later total is exactly zero
with `Delta=-K(BX)`. At order 48 the nonconstant increment has
`max|Delta|=0.08756215922577214`, above the strict `1e-3` discriminator. The
naive old-plus-whole-later insertion has error `0.6927530692282082` for the
nonconstant map and `0.2797083383527048` even for identity, both above `1e-3`.
After each of two sequential signed maps, factor and separately enumerated
dense tensors agree in every D4/D22/D31 entry, the oriented off-diagonal
covariance, and all eight entries of the **declared one-quadratic/two-linear
K4-to-K3 feed**. Each transported total and increment also agrees with a fresh
raw-partition calculation of that map's output law.

The 4096-node order-32 and 9216-node order-48 *first-layer* source cumulants
converge to `2.22e-15` absolute. After `ReLU(BX)`, the means differ by up to
`9.02678390007905e-6` (`2.1843640157463572e-5` material relative), and the
increment tensor by up to `2.105534106192769e-5` (`8.48269946506e-4`
material relative). Across all failures the largest material relative
difference is `0.0018697216703024802`; the largest absolute difference is
`4.7736183096835916e-5`. The new oblique ReLU boundary cuts through the
fixed product-grid cells; this explains the observed slow order convergence.
It does not affect exact identities evaluated on a single order's measure.
The 23 failures are confined to nonconstant later-output and increment
convergence, including their transported consumers and feed.

The focused command was `uv run python local_sim/test_p2_k4_increment.py
--output experiments/2026-09-11_stage2_compact_p07/increment_source/result.json`.
It exited 1 by design because the registered gate failed and preserved the
complete result. Its source SHA-256 is
`c09860e6c96cf812ede6d3f1bf1d891205afa6e2517fae683427155ecbf5721e`;
the result SHA-256 is
`08468eabff7d9134bff172f3ee0d6af6b62110e8eea0dac892532507e9337d16`.
The script and result hashes were unchanged between live preflight submission
and terminal.

Workerq **983** ran current-byte default `powershell -NoProfile
-ExecutionPolicy Bypass -File scripts/preflight.ps1` with `--no-snapshot
--live-worktree --ram 5 --cpus 2 --gpus 0`. Its start log binds HEAD
`1e9949481add90627d97df9d33a77ab3211543f6` with the fixture and artifact
directory untracked. It ended **SUCCEEDED, exit 0, PREFLIGHT GREEN 12/16**;
four Full-only steps were skipped. Measured process peak was `2842.15625 MiB`
under the 5 GiB reservation. Default preflight does not execute this
standalone fixture; the focused command above supplies that result.

`archive_queue.py` copied job 983's terminal log, manifest, environment and
queue result byte for byte, captured the terminal `workerq show --json`
receipt, and verified every `job983_raw.zip` member against its retained copy.
`queue_verification.json` records every original path, member byte count and
SHA-256. ZIP SHA-256 is
`a7ac24d8ff0c48cc06ff66058b0521c6b6c2ad96f7dbed37912d877b7b086327`.
Git warns it may normalize the convenient log and terminal JSON text copies
on a future checkout; the ZIP holds their original bytes.

No labels, decision seed block, public split, vault, frozen estimator, shipped
estimator, or submission were touched. This failure leaves later-source
convergence unproven. It supplies no challenge score, measured flopscope cost,
width-1024 factor rank, all-history closure, or server residual headroom.
