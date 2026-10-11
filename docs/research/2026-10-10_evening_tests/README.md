# Oct 10 evening tests (strategist, run on the device VM)

| script | what it tests | result (review §3–§4) |
|---|---|---|
| `d21_rank.py [net] [layers]` | whether the true D21 and its covariance image E = sym(D_w2·D21·D_w1) are low-rank, measured by the next-layer variance error of a rank-r truncation | E is 85 % rank-2 by energy, but the variance error is 2.5e-3 at rank 64 and ~1e-3 at rank 256, against a ~1e-4 tolerance. Low-rank readouts are dead. |
| `cubature.py [nets] [K]` | orthonormal-basis + antipode spherical cubature (randomized Hadamard or random orthonormal bases) against plain MC, using E[h(x)] = E‖x‖·E_θ[h(θ)] | ~2× better than MC (2.0e-6 vs 3.6e-6 at 16,384 points, F/B 0.25); ~100× worse than the chain. Dead. |
| `probe_cplx.py` | how flopscope 0.12.1 bills complex64 matmuls and elementwise ops | complex matmul = 4× real (2,088,960 vs 520,192 at 64³). Packing two real products into one complex product costs 2×. |

- The scripts read `cache/twin_labels/fit_*.npz`, `cache/twin/full_w.f32` and `cache/twin/full_truth.npy` under the repo root. Set `WHEST_ROOT` to override the root.
- They are numpy-only, except `probe_cplx.py`, which needs the real flopscope 0.12.1.
