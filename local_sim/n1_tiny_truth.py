"""Monte Carlo truth for tiny self-minted networks (N1 continuous-cycle accuracy test).

Seeds are OUTSIDE every registered block (90_000_019_000 + 97 n + net); these
networks exist only for this label-free-method exploration and are never a DECIDE
or CONFIRM corpus. Two independent replicas, all layers' post-ReLU means, float64
accumulation of float32 forward passes. Runs in .venv-gpu (torch).
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(_HERE)
sys.path[:] = [q for q in sys.path if os.path.abspath(q or ".") != _HERE] + [R, _HERE]
import numpy as np
import torch
from mint_provenance import canonical_mint

def mc(ws, samples, seed, chunk=1 << 20):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev); g.manual_seed(seed)
    W = [torch.tensor(np.asarray(w), dtype=torch.float32, device=dev) for w in ws]
    n = W[0].shape[0]
    acc = torch.zeros(len(W), n, dtype=torch.float64, device=dev)
    done = 0
    while done < samples:
        b = min(chunk, samples - done)
        x = torch.randn(b, n, generator=g, device=dev)
        for l, w in enumerate(W):
            x = torch.relu(x @ w)          # row-vector convention: x_next = relu(x W), i.e. W^T x
            acc[l] += x.sum(0, dtype=torch.float64)
        done += b
    return (acc / samples).cpu().numpy()

def main():
    torch.set_float32_matmul_precision("highest"); torch.backends.cuda.matmul.allow_tf32 = False
    out = {}
    depth = 12
    for n in (64, 96):
        for net in range(4):
            seed = 90_000_019_000 + 97 * n + net
            w, es, _ = canonical_mint(n, depth, seed)
            r1 = mc(w, 1 << 27, seed * 2 + 1)
            r2 = mc(w, 1 << 27, seed * 2 + 2)
            out[f"n{n}_net{net}_seed"] = np.array(seed)
            out[f"n{n}_net{net}_r1"] = r1; out[f"n{n}_net{net}_r2"] = r2
            print(n, net, "replica disagreement final-layer mse %.3e" % np.mean((r1[-1] - r2[-1]) ** 2), flush=True)
    np.savez(os.path.join(R, "experiments", "2026-09-17_n1_cycle", "tiny_truth.npz"), **out)

if __name__ == "__main__":
    main()
