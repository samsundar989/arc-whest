# Boundary-partitioned two-layer increment, 2026-09-12

The preregistered IMPORTANT contract is `experiments/LOG.md` at `f4f1305`.
The focused fixture exits **FAIL: 640/688 checks pass, 48 fail**. Every failure
is a fixed `N=32` versus `N=48` numerical comparison. No threshold, order,
partition, or result was changed after observing that outcome. This is a
negative result for the registered combined gate, not a rejection of the
finite-measure increment identity or a claim about challenge score.

The new positive-support construction uses the exact origin mass `1/3`, each
axis mass `1/6`, and the three positive-quadrant sectors split at
`atan(11/28)` and `atan(8/3)`. It uses a three-node Hankel radial rule and
fixed Gauss-Legendre angular/mapped-axis orders `32,48,64`, yielding exactly
`353,529,705` nodes. The radial moment defect through degree five is at most
`3.907985046680551e-14` against `1e-12`; the largest pre-scaling support-mass
error is `9.930251065881635e-12` against `1e-8`, and scaled support masses
agree to `5.56e-17`. The mapped-axis far-tail density underflows to zero in
float64 at 3, 5, and 7 nodes per axis as order increases. Those zero-weight
nodes are retained and counted; no negative or nonfinite weights occur.

All same-bank raw-partition, signed centered-factor/covariance, old-plus-Delta,
two-map D4/D22/D31, oriented-covariance, declared one-quadratic/two-linear
K4-to-K3 feed, and identity/negative-identity controls pass. Worst structural
error is `3.219646771412954e-15` against `1e-10`. At order 64 the
nonconstant `max|Delta|` is `0.08756130902807811`, above the fixed strict
`1e-3` discriminator. The naive old-plus-whole-later and identity controls
also clear their fixed `1e-3` discriminators.

The worst failed 32/48 material-relative gap is `8.807266285723563e-6`
(`Delta`), and worst failed absolute gap is `2.899732760575091e-7` (a raw
later/linear fourth moment), against the fixed `1e-8` material-relative or
near-zero absolute gate. The 48/64 comparisons reach at most
`6.583237642021994e-9` material-relative. An independently coded
conditional-Gaussian/truncated-normal continuous integral, with raw set
partitions independent of the discrete centered factors, agrees with order
48 to at most `6.588178380515773e-9` and order 64 to at most
`4.940738526304726e-12` material-relative. Those higher-order diagnostics
do **not** turn the combined fixed gate into a pass. The failure indicates
that the prescribed order-32 mapped-axis rule is underresolved for some
higher moments despite accurate support mass.

Focused command: `uv run python local_sim/test_p2_k4_increment_boundary.py
--output experiments/2026-09-11_stage2_compact_p07/increment_boundary/result.json`.
It exits 1 and retains the complete JSON. Source SHA-256 is
`c9759cea80eda54302a9800ac8f960b0ff5e1ef5c1b4a530a21138d2dd849ceb`;
result SHA-256 is
`6ec0a6ee32bd1a277dc352bf81f241c23b5860cd5aa42538e0e57bb9a02f5b18`.

Workerq **985** separately ran current-byte default preflight through
`--no-snapshot --live-worktree --ram 5 --cpus 2 --gpus 0`, at HEAD
`cffc4d371be2cd41cb63c98bc71953d88f4a5a14`. It ended **SUCCEEDED,
exit 0, PREFLIGHT GREEN 12/16**; four Full-only steps were skipped. Peak
process RAM was `2833.9609375 MiB` within the 5 GiB reservation. This
default preflight does not execute the focused boundary fixture. Its log
records HEAD and untracked files; the queue start does not independently
timestamp-bind the untracked fixture digest. Source and result hashes were
rechecked unchanged after terminal.

`archive_queue.py` copied the terminal log, manifest, environment, result,
and `workerq show --json` stdout byte for byte, then verified every retained
ZIP member. `queue_verification.json` records original paths, byte counts,
and member SHA-256 values. The ZIP SHA-256 is
`4e9a2aea6a77b0ba22e2b3758d91d9636b9f852fa3bee552b1414968fe5550ff`.

No protected labels, decision seed block, public split, vault, frozen or
shipped estimator, or submission were touched. This checks a fixed
two-coordinate source with a toy *linear* old carry. It does not establish a
real nonlinear response, width-1024 factor rank, counted challenge FLOPs,
server residual headroom, or accuracy. A fresh independent Skeptic must
review the negative result before synthesis or any further gate.
