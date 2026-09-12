# Frozen coherent K4 executable trace (truth-free)

Contract: `experiments/LOG.md` at commit `2164401`. Run
`uv run python experiments/2026-09-11_stage2_compact_p07/source_cumulant/executable_trace/trace_frozen_layer.py`
from the repository root. The run uses only deterministic width-8/depth-2
`W0=sqrt(2) I` and row-signed Hadamard He `W1`, plus a separate signed width-2
covariance state. It reads the byte-pinned frozen coherent source and exact-arm
control. `sys.settrace` copies locals from the actual `_layer` frame before
`Estimator.predict` can clean outputs. An untraced call's complete returned
state equals the traced call's returned state, including nested birth factors.

`trace_result.json` is PASS (14/14 checks). L1 has diagonal `C=0.681689978`,
`k4v=2.512936592`, off-diagonal `k22=0.837645531`, and
`gcore=1.256468296`. L2 has `dG=2.512936592` and `D4=5.025873184` in
every row. The independent cumulant contraction is `4*kappa4/8=0.559416677`.
Maximum reported relative error for the contracted source terms is below
`4e-7`, against the predeclared `1e-4` tolerance. The executed signed D31
increment has `[0,1]=0.0022620093` and `[1,0]=-0.0004265551`; both match
the independently calculated orientation. The frozen exact arm has no L2
`dG` or `D4` locals. The optional control initially stopped at a mistyped
expected exact-arm SHA before source execution; the corrected byte pin and
complete fixture passed before preflight submission.

Job 961 ran the default preflight with `--no-snapshot --live-worktree`, 5 GiB
declared RAM, two CPUs, and zero GPUs, starting at HEAD `ef503a2`. Its raw
transcript reports **GREEN 12/16**, four Full-only steps skipped. Workerq's
terminal record says SUCCEEDED, exit 0, measured peak 3275.398 MiB. The raw
queue files are copied here byte-for-byte as `job961_live_preflight.log`,
`job961_live_manifest.json`, `job961_live_environment.json`, and
`job961_live_result.json`; `job961_terminal_show.json` records the terminal
peak and live snapshot mode. Windows Git normalizes the readable log on add;
`job961_raw.zip` contains the exact original queue bytes of all four copied
files. Each ZIP member was checked equal to its queue copy before commit.

SHA-256 (disk bytes, before this packet was committed):

| File | SHA-256 |
|---|---|
| frozen coherent arm | `d3ec837b72f5dacb4f8f622394d3d635268649c741289fb9754197e993343665` |
| frozen exact arm | `f704f83b2063894c8dcd13120a8baada762229563f6d08bccee39d3aad2e6efc` |
| `trace_frozen_layer.py` | `635f7c4a0f6e6a0b93389d375c271f82e0d8b20b97d3418850776558d805a253` |
| `trace_result.json` | `a7f25fb4b20c3cbeccc98f3aa59c8158f1e2d2a2d3fec7c13cc948e531b8ef8a` |
| `job961_live_preflight.log` | `a593a6bb1e646ca853529843eced87bb540e1d68ed5c1a159479f7dbb1aeb643` |
| `job961_live_manifest.json` | `ce5012c2653102b7cdadd32b6e27b4da093de342e8b2f4e2d317bf64906f1a3f` |
| `job961_live_environment.json` | `205e4112003b908083a32397d06d63d4ec969166a6062e87a02d38ec4c85fbf7` |
| `job961_live_result.json` | `a7c8e42a84926fb69cc69daff4423665acd1f2a3e00b09b8367f6bbdcbf61412` |
| `job961_terminal_show.json` | `928bb6eba569bf22e403a634117686cf15ae390f87edeb86368c62850130e245` |
| `job961_raw.zip` | `d8bb0a7d09d747fb8297d8d8c8b3ae2beeda3935aae579be799c8a05d25fa4f0` |

This is an executable source-identity finding on an atypical synthetic
network. It does not measure accuracy, challenge score, counted FLOPs,
server residual headroom, or a repaired K4 method. The default preflight is
not a Full deployment gate and does not itself execute this trace script.
