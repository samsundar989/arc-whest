# Exact-axis two-layer increment fixture, 2026-09-12

The preregistered IMPORTANT contract is `experiments/LOG.md` at `4622426`;
its pre-result operational preflight amendment is `2b997b2`. The new fixture
copies the frozen boundary test and replaces only the half-axis quadrature
with a positive three-node Hankel rule from independently integrated moments
`mu0..mu5`. It retains `rho=.5`, `B=[[.8,-.3],[-.55,1.4]]`, the origin and
three oblique angular sectors, radial rule, fixed angular orders `32/48/64`,
signed maps, raw partitions, controls, conditional-Gaussian reference and
every numerical threshold. The old `increment_boundary` FAIL bytes remain
unchanged.

The focused command was `uv run python local_sim/test_p2_k4_increment_axis_gauss.py
--output experiments/2026-09-11_stage2_compact_p07/increment_axis_gauss/result.json`.
It exited zero: **PASS 715/715 registered checks**, with complete raw JSON.
The banks have `295/439/583` nodes. The six axis-moment defects are at most
`2.220446049250313e-16`, and the independent scalar quadrature's combined
reported/doubling/omitted-tail bound is `5.038759200361449e-14`, both below
`1e-12`. Axis nodes and weights are strictly positive. The largest
pre-scaling support-mass error is `2.7755575615628914e-16` against `1e-8`.
Each order records the *pre-scaling* axis-0, axis-1 and quadrant contribution
to every degree-four raw moment of `X`, `BX`, and `ReLU(BX)`.

The worst material-relative gap is `1.9762954105226713e-13` for 32/48,
`3.952590821046905e-13` for 48/64, `3.9525908210437806e-13` for
48/continuous-reference, and `7.905181642087561e-13` for
64/continuous-reference, all under the fixed `1e-8` gate. The maximum
same-bank structural identity error is `2.942091015256665e-15` against
`1e-10`. The signed old-plus-increment, two-map D4/D22/D31, oriented
covariance, declared quadratic K3 feed, identity/negative-identity and
naive-double-count controls all pass. At order 64 the nonconstant
`max|Delta|=0.08756130902805903`, above the strict `1e-3` discriminator.
This repairs the prescribed 32/48 *numerical integration* failure for the
fixed two-coordinate source; it does not infer a deployable nonlinear
old-history method or a challenge score.

Source SHA-256:
`1f2f421bcc696dec263a77847b076914982bc41eba08f251a4034925a6f4d1d9`.
Result SHA-256:
`3d297f29b0831c0958c3e86916eaf74b818114cf696f678ef033d32aabffc6b2`.
Both were committed at `df81223990bd46a875417f69fadff610760a1796`
before preflight. `byte_state_start.json` and `byte_state_end.json` are
exactly equal: they bind that HEAD, source/result hashes, and 332 Python
file hashes before submission and after terminal. A separate later commit
`70a2d80` changed HEAD *after* the terminal byte-state check.

Workerq **991** ran the amended live-worktree default preflight with
`--no-snapshot --live-worktree --ram 5 --cpus 2 --gpus 0` against tested
HEAD `df81223`. It ended **SUCCEEDED, exit 0, PREFLIGHT GREEN 12/16**;
four Full-only steps were skipped. Its measured peak process RAM was
`2827.03515625 MiB` under the 5 GiB reservation. Default preflight does
not execute the focused axis fixture; that result is the standalone run
above. The abandoned Git-snapshot path is recorded in the amendment as a
RED packaging artifact, not counted as axis preflight evidence.

`archive_queue.py` copied the terminal log, manifest, environment, queue
result and terminal `workerq show --json` stdout byte for byte, then checked
every retained `job991_raw.zip` member. `queue_verification.json` records
the originals, lengths and SHA-256 values and confirms byte-state equality.
The ZIP SHA-256 is
`569510bdee07c1340d28b70ae6f22981af08da65e1e6a5ec93ed4e3832e66ad9`.

No protected labels, decision seed block, public split, vault, frozen or
shipped estimator, or submission were touched. This remains a fixed
two-coordinate law with a toy *linear* old carry. It does not establish a
real nonlinear old-history response, multi-source closure, width-1024 rank,
counted challenge FLOPs, server residual headroom, or accuracy. A fresh
independent Skeptic must review the result before synthesis or promotion.
