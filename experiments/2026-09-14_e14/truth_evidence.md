# E14 production truth arithmetic instrument

IMPORTANT Builder prediction committed at `68cac46`; coordinator committed the
pilot protocol and resource review at `88d6002` before the pilot. No new seed
block, protected truth read, candidate change or promotion is involved.

Production `local_sim/p2_truth.py` pins highest f32 matmul precision, disables
TF32, refuses active autocast and checks for backend/runtime drift. New NPZs
embed JSON provenance and receive a SHA256-bound `.producer.json` sidecar.
Provenance distinguishes f32 forward, f64 sums/squared sums and returned means,
f32 exported means, and f64 pre-export-centered variances. Historical artifacts
remain unchanged and their missing numerical mode is not retrospectively filled.

The independent reference source is `local_sim/test_truth_math_e14.py::reference`.
It follows the installed official baker's f32 forward/f64 reduction/f32 export
precision. The official files are preserved verbatim in
`truth_official_simulation.py` (SHA256
`6b3b13e910c7a7e90b2e95c9add4d229976dcfd0490fc9cc0f326601b1684fc2`)
and `truth_official_simulation_torch.py` (SHA256
`31042ed31a011a18e2f0ff5104bfc15769e74dfd83e3d21b38c3ed79efe1b086`).
The official baker computes final variance around f32 exported means; this
producer retains its historical pre-export f64 centering. Both are reported
separately. Per-layer reference variance is explicitly a diagnostic extension.

`truth_cpu_tests.txt` records seven passing one-thread CPU tests: actual producer
versus independent reference, ragged chunks, unchanged default RNG routing,
autocast rejection, backend mutation rejection, source/library/device/chunk
binding refusal, input validation and sidecar/embedded metadata binding.
These tiny tests do not certify CUDA arithmetic at production shape.

Pilot command for the coordinator only:

```powershell
workerq submit --project arc-whest --ram 8 --vram 6 --cpus 1 -- .venv-gpu\Scripts\python.exe local_sim/test_truth_math_e14.py --pilot --out experiments/2026-09-14_e14/truth_pilot.json
```

Exactly one fixed 1024x16 canonical probe uses existing `--check-allvar` fixture
seed 1000051. A synthetic f32 NumPy input array, seed 140914, has 65537 rows and
is generated once. Exactly two predefined chunk settings, 65536 and 8192,
exercise the actual `_truth_one` producer against the independent reference.
The first includes a full production chunk and a one-row ragged tail. Inputs,
weights, code, reference sources and runtime provenance are hash-bound.

Before measurement the per-layer normalized max mean and variance differences
are required to be at most `2e-6`, normalized by `1+abs(reference)`. For final
exported mean squared arithmetic disagreement D, the conditional Cauchy bound
`2*sqrt(2e-8*D)+D` must be at most `1e-10`. This is at most 10% of a hypothetical
5% raw gain at error scale `2e-8`, provided reference residual MSE is at most
that scale and the multiplier is at most one. It is a finite-fixture arithmetic
screen, not an unconditional population-bias bound or promotion gate. Common
inputs remove independent sampling noise from the arithmetic difference; no
Monte Carlo truth accuracy MSE is measured.

Optimization review: one shared 256-MiB input array and 64-MiB weights; no
datasets/loaders/workers/prefetch/cache/model duplication. Torch and CPU BLAS
each use one thread. CPU reference and CUDA producer execute sequentially,
retaining layer moments and current activations only. Conservative host peak
below 4 GiB plus runtime is reserved at 8 GiB; CUDA temporaries/cache/context
are reserved at 6 GiB. No half precision, compilation, recovery weakening or
adaptive extra pass is used. Completed rows print immediately; the final JSON
is immutable. No resume or protected re-read semantics are introduced.

Default preflight, the queued pilot, and fresh independent Skeptic review are
the coordinator's remaining acceptance steps. No GPU work was run by this lane.
