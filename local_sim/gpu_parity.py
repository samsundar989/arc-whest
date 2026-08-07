"""
GPU MIRROR PARITY — does the RTX box compute what the grader computes?

Two separate claims, proven separately, because they have different standards:

  1. **Generation is bit-for-bit.** The MLP the GPU mirror scores must be the
     exact float32 network `whestbench.sample_mlp` produces for the same input
     seed. This is an equality claim and it is checked as one — on the raw uint32
     bit patterns, not on a tolerance. It holds by construction now that the
     mirror mints on the CPU via `mint_provenance.canonical_mint` and copies to
     the device; it did NOT hold when the mirror drew from torch's own generator,
     which matches numpy in distribution and in nothing else.

  2. **The forward pass agrees within a documented tolerance.** This one cannot
     be an equality claim: cuBLAS and OpenBLAS reduce a 256-term dot product in
     different orders, and 32 layers of ReLU compound that. So we measure the
     disagreement rather than assert it away, and we report it against the thing
     that matters — the size of the signal we make decisions about (~2e-7
     adjusted, i.e. ~1.4e-6 raw final-layer MSE).

Run (GPU venv, which has torch but deliberately no whestbench):
    .venv-gpu\\Scripts\\python.exe local_sim/gpu_parity.py
    .venv-gpu\\Scripts\\python.exe local_sim/gpu_parity.py --mlps 8 --samples 262144
"""

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mint_provenance import canonical_mint, seed_streams  # noqa: E402


def forward_np(x, w):
    """CPU float32 forward, float64 column accumulation. Returns (depth, width)."""
    out = np.empty((len(w), w[0].shape[1]), dtype=np.float64)
    for li, wl in enumerate(w):
        x = np.maximum(x @ wl, np.float32(0.0))
        out[li] = x.mean(axis=0, dtype=np.float64)
    return out


def forward_torch(x, w, dev):
    out = torch.empty((w.shape[0], w.shape[-1]), dtype=torch.float64, device=dev)
    for li in range(w.shape[0]):
        x = torch.clamp(x @ w[li], min=0.0)
        out[li] = x.mean(dim=0, dtype=torch.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlps", type=int, default=6)
    ap.add_argument("--samples", type=int, default=131072)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--seed0", type=int, default=424242)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("=" * 92)
    print("GPU MIRROR PARITY")
    print("=" * 92)
    print(f"device       : {torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'}")
    print(f"torch        : {torch.__version__}   tf32 disabled")
    print(f"shape        : {args.width}x{args.depth}   {args.mlps} MLPs   "
          f"{args.samples:,} samples/MLP")
    print()

    # ---------------------------------------------------------------- claim 1
    print("CLAIM 1 — generation is bit-for-bit identical to the canonical mint")
    print("-" * 92)
    print(f"{'input seed':>12} {'bits equal':>12} {'dtype':>9} {'device roundtrip':>18} "
          f"{'est. seed':>12}")
    print("-" * 92)
    gen_ok = True
    mlps = []
    for k in range(args.mlps):
        s = args.seed0 + k
        w_np, est_seed, sample_ss = canonical_mint(args.width, args.depth, s)
        stacked = np.stack(w_np)
        w_gpu = torch.from_numpy(stacked).to(dev)
        back = w_gpu.cpu().numpy()
        bits = np.array_equal(back.view(np.uint32), stacked.view(np.uint32))
        gen_ok &= bits and back.dtype == np.float32
        mlps.append((s, stacked, w_gpu, est_seed, sample_ss))
        print(f"{s:>12} {str(bits):>12} {str(back.dtype):>9} "
              f"{'lossless' if bits else 'LOSSY':>18} {est_seed:>12}")
    print("-" * 92)
    print(f"generation bit-for-bit: {gen_ok}\n")

    # ---------------------------------------------------------------- claim 2
    print("CLAIM 2 — forward pass agrees within tolerance (BLAS reduction order)")
    print("-" * 92)
    print(f"{'input seed':>12} {'max|dGPU-CPU| final':>21} {'rel. to mean act.':>19} "
          f"{'implied MSE':>13} {'vs 1.4e-6 signal':>17}")
    print("-" * 92)
    worst_rel = 0.0
    worst_mse = 0.0
    for s, w_np, w_gpu, _est, sample_ss in mlps:
        gt_seed = int(sample_ss.generate_state(1)[0]) & 0x7FFFFFFF
        rng = np.random.default_rng(gt_seed)
        g = torch.Generator(device=dev).manual_seed(gt_seed)

        # identical INPUTS on both sides: draw once on CPU, copy to device, so the
        # only difference measured is the arithmetic, not the sample stream
        chunk = min(16384, args.samples)
        acc_np = np.zeros((args.depth, args.width))
        acc_t = torch.zeros((args.depth, args.width), dtype=torch.float64, device=dev)
        done = 0
        while done < args.samples:
            b = min(chunk, args.samples - done)
            x = rng.standard_normal((b, args.width), dtype=np.float32)
            acc_np += forward_np(x, list(w_np)) * b
            acc_t += forward_torch(torch.from_numpy(x).to(dev), w_gpu, dev) * b
            done += b
        cpu_m = acc_np / args.samples
        gpu_m = acc_t.cpu().numpy() / args.samples

        d = float(np.abs(gpu_m[-1] - cpu_m[-1]).max())
        scale = float(np.abs(cpu_m[-1]).mean())
        mse = float(((gpu_m[-1] - cpu_m[-1]) ** 2).mean())
        worst_rel = max(worst_rel, d / max(scale, 1e-30))
        worst_mse = max(worst_mse, mse)
        print(f"{s:>12} {d:>21.3e} {d / max(scale, 1e-30):>19.2e} {mse:>13.3e} "
              f"{mse / 1.4e-6:>16.1e}x")
        del g
    print("-" * 92)
    print(f"worst relative final-layer disagreement : {worst_rel:.2e}")
    print(f"worst implied MSE from BLAS alone       : {worst_mse:.3e}")
    print(f"  ... as a share of the 1.4e-6 raw MSE we make decisions on: "
          f"{worst_mse / 1.4e-6:.2e}")
    print()

    tol_ok = worst_mse < 1.4e-8  # 1% of the decision signal
    print("VERDICT")
    print(f"  generation bit-for-bit ................ {'PASS' if gen_ok else 'FAIL'}")
    print(f"  forward within 1% of decision signal .. {'PASS' if tol_ok else 'FAIL'}")
    sys.exit(0 if (gen_ok and tol_ok) else 1)


if __name__ == "__main__":
    main()
