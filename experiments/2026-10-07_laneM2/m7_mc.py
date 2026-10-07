"""Lane M7, step 2 -- exact (MC) labels for v77's coherent early-layer error. `.venv-gpu`.

For each net in m7_state.npz: N fresh x ~ N(0, I) through layers 1..6 (h <- ReLU(h @ W[l]),
fp32 forward, TF32 off, fp64 accumulators). Per layer it accumulates the per-neuron mean and
variance and, for v77's 5 probe directions U = [m̂, u1..u4], the mean and second moment of
p = h @ U. Ten equal blocks give a standard error for every projected quantity.

    python experiments/2026-10-07_laneM2/m7_mc.py --state <abs m7_state.npz> --n 1e9 --out <abs.npz>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[1] / "tools" / "fresh"))
import gen  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_grad_enabled(False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--n", type=float, default=1e9)
    ap.add_argument("--chunk", type=int, default=1 << 16)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--nets", default=None, help="lo:hi rows of the state file")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    st = np.load(a.state)
    dev = torch.device("cuda")
    N = int(a.n)
    per_block = N // a.blocks
    n_chunks = per_block // a.chunk
    prog = os.environ.get("WORKERQ_PROGRESS")
    lo, hi = (int(x) for x in a.nets.split(":")) if a.nets else (0, len(st["seeds"]))
    nets = st["seeds"][lo:hi]
    L = st["U"].shape[1]
    out = {k: [] for k in ("S1", "S2", "P1", "P2")}
    for kk, seed in enumerate(nets):
        k = lo + kk
        w = torch.as_tensor(gen.mint_numpy(int(seed))[:L], device=dev)
        U = torch.as_tensor(st["U"][k], device=dev, dtype=torch.float32)   # (L, n, 5)
        n = w.shape[1]
        S1 = torch.zeros(a.blocks, L, n, dtype=torch.float64, device=dev)
        S2 = torch.zeros_like(S1)
        P1 = torch.zeros(a.blocks, L, 5, dtype=torch.float64, device=dev)
        P2 = torch.zeros(a.blocks, L, 5, 5, dtype=torch.float64, device=dev)
        g = torch.Generator(device=dev).manual_seed(int(seed) % (2**63 - 1) ^ 0x5EED7)
        for b in range(a.blocks):
            for _ in range(n_chunks):
                h = torch.randn(a.chunk, n, device=dev, generator=g)
                for l in range(L):
                    h = torch.relu(h @ w[l])
                    hd = h.double()
                    S1[b, l] += hd.sum(0)
                    S2[b, l] += (hd * hd).sum(0)
                    p = (h @ U[l]).double()
                    P1[b, l] += p.sum(0)
                    P2[b, l] += p.T @ p
            if prog:
                Path(prog).write_text(f"{(kk * a.blocks + b + 1) / (len(nets) * a.blocks):.3f}")
        cnt = float(n_chunks * a.chunk)
        out["S1"].append((S1 / cnt).cpu().numpy())
        out["S2"].append((S2 / cnt).cpu().numpy())
        out["P1"].append((P1 / cnt).cpu().numpy())
        out["P2"].append((P2 / cnt).cpu().numpy())
        print(f"net {k} seed {seed} done", flush=True)
        np.savez(a.out, **{kk: np.array(v) for kk, v in out.items()},
                 per_block=np.int64(n_chunks * a.chunk), seeds=nets[: kk + 1],
                 rows=np.arange(lo, k + 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
