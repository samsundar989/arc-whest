"""
Sprint 4 ground truth for a decision seed block (GPU; runs in .venv-gpu).

Mints canonical (grader-identical) MLPs for a registered seed block and labels
each one's final-layer post-ReLU means by high-sample Monte Carlo, drawing from
the **evaluator's own ground-truth substream** (`seed_streams(input_seed)[1]`,
i.e. spawn[1]) rather than an arbitrary generator seed. A block's truth is
therefore reproducible from its input seeds alone, exactly as the vault's is.

Also stores the per-neuron sample variance so every MSE measured against this
file can be quoted with its own noise floor (`docs/VALIDATION.md` §3.5), which
is the difference between an improvement and a rounding error.

Weights are NOT stored: `mint_provenance` proves they are a bit-exact function
of the input seed, so the seeds are the archive.

Run:
    .venv-gpu\\Scripts\\python.exe local_sim/packed_truth.py --block D11 --mlps 200
    .venv-gpu\\Scripts\\python.exe local_sim/packed_truth.py --block D12 --mlps 200 --samples 8388608
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mint_provenance import canonical_mint, seed_streams  # noqa: E402
from seed_blocks import seeds as block_seeds  # noqa: E402

OUT_DIR = os.path.join(_ROOT, "experiments", "packed")
WIDTH, DEPTH = 256, 32


def truth_final(w, n_samples, gen_seed_seq, dev, chunk=131072):
    """Final-layer post-ReLU mean and per-neuron variance, f64 accumulation.

    Draws are taken from numpy on the CPU using the evaluator's spawn[1]
    substream, then moved to the device -- torch cannot reproduce numpy's bit
    stream (gpu_parity.py), and reproducibility of the labels is worth more than
    the transfer time.
    """
    rng = np.random.default_rng(gen_seed_seq)
    n_out = w.shape[-1]
    s1 = torch.zeros(n_out, dtype=torch.float64, device=dev)
    s2 = torch.zeros(n_out, dtype=torch.float64, device=dev)
    done = 0
    while done < n_samples:
        b = min(chunk, n_samples - done)
        x = torch.from_numpy(rng.standard_normal((b, w.shape[-1]), dtype=np.float32)).to(dev)
        for li in range(w.shape[0]):
            x = torch.clamp(x @ w[li], min=0.0)
        xd = x.to(torch.float64)
        s1 += xd.sum(dim=0)
        s2 += (xd * xd).sum(dim=0)
        done += b
    mean = s1 / n_samples
    var = torch.clamp(s2 / n_samples - mean * mean, min=0.0)
    return mean, var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, help="seed block id, e.g. D11")
    ap.add_argument("--mlps", type=int, default=200)
    ap.add_argument("--samples", type=int, default=8_388_608)  # 2**23
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    all_seeds = block_seeds(args.block)
    if args.mlps > len(all_seeds):
        raise SystemExit(f"block {args.block} holds {len(all_seeds)} seeds, asked for {args.mlps}")
    sd = all_seeds[: args.mlps]

    out = args.out or os.path.join(OUT_DIR, f"{args.block}_truth_{args.width}x{args.depth}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; refusing to overwrite a labelled decision block.")

    print(f"device {torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'}  "
          f"block {args.block}  {len(sd)} MLPs  {args.width}x{args.depth}  "
          f"N_truth = {args.samples:,}", flush=True)

    truth = np.zeros((len(sd), args.width), dtype=np.float64)
    tvar = np.zeros((len(sd), args.width), dtype=np.float64)
    est_seeds = np.zeros(len(sd), dtype=np.uint64)
    t0 = time.perf_counter()
    for i, s in enumerate(sd):
        w_np, est_seed, sample_ss = canonical_mint(args.width, args.depth, int(s))
        w = torch.from_numpy(np.stack(w_np)).to(dev)
        m, v = truth_final(w, args.samples, sample_ss, dev)
        truth[i] = m.cpu().numpy()
        tvar[i] = v.cpu().numpy()
        est_seeds[i] = np.uint64(est_seed)
        if (i + 1) % 20 == 0:
            el = time.perf_counter() - t0
            print(f"  {i + 1}/{len(sd)}  {el:.0f}s  ({el / (i + 1):.2f}s/MLP)", flush=True)

    gt_noise = tvar.mean(axis=1) / float(args.samples)
    np.savez_compressed(out, seeds=np.array(sd, dtype=np.int64), est_seeds=est_seeds,
                        truth=truth, truth_var=tvar, gt_noise=gt_noise,
                        n_truth=args.samples, width=args.width, depth=args.depth,
                        block=args.block)
    print(f"\nwrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    print(f"GT-noise MSE: mean {gt_noise.mean():.3e}  max {gt_noise.max():.3e}  "
          f"({gt_noise.mean() / 1.4e-6 * 100:.3f}% of the 1.4e-6 decision signal)")
    print(f"mean(truth^2) median {np.median((truth ** 2).mean(axis=1)):.4f}")


if __name__ == "__main__":
    main()
