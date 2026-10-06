"""GPU Monte Carlo labeller for fresh networks (Lane V, V2). Runs in `.venv-gpu`.

For each net: all-layer post-ReLU means AND per-neuron variances from N i.i.d.
x ~ N(0, I) inputs. The variances make each net's own label-noise floor exact:

    floor(net, layer) = mean_j Var[h_lj] / N

which is what `tools/ev.py` subtracts from an absolute fresh-panel MSE (paired
differences need no correction: the label noise is common to both arms).

Numerics (PLAN rev 18 §2 / handoff V2):
  * TF32 OFF -- full float32 matmuls (`allow_tf32 = False`, precision "highest").
  * float64 accumulation of sums and sums of squares across chunks (in-chunk
    float32 tree reduction, max relative error 2.3e-7 -- see `label`).
  * Deterministic: one torch CUDA Philox generator per (net, N tier), seeded from
    the net's SAMPLE SeedSequence child (the organizers' sample stream index) and
    a tier tag, so the N = 1e8 and N = 1e9 labels of one net are independent.

Weights are rebuilt from the seed by `gen.mint_numpy` (bit-identical to whestbench's
own mint -- `gen.py --parity`) and, for pinned panel nets, checked against the
recorded sha256 before a single sample is drawn.

    .venv-gpu/Scripts/python.exe tools/fresh/mc.py --panel fresh_a [--lo 0 --hi 100] [--n 1e8]
    .venv-gpu/Scripts/python.exe tools/fresh/mc.py --panel fresh_lock --n1e9-subset --n 1e9
    .venv-gpu/Scripts/python.exe tools/fresh/mc.py --mini 0:10              (validation)
    .venv-gpu/Scripts/python.exe tools/fresh/mc.py --lane-m 512:0:8         (width override)

Resumable: a net whose label file exists is skipped, so the job is preemptible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
import gen  # noqa: E402

TIER_TAG = {10**8: 1, 10**9: 2}


def setup_numerics() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def sample_seed(input_seed: int, n: int) -> int:
    """63-bit torch seed from the net's sample stream child and the N tier."""
    child = np.random.SeedSequence(int(input_seed)).spawn(3)[1]
    tagged = np.random.SeedSequence(child.generate_state(4).tolist() + [TIER_TAG.get(n, n)])
    return int(tagged.generate_state(2, dtype=np.uint64)[0] & ((1 << 63) - 1))


@torch.no_grad()
def label(w: np.ndarray, n: int, seed: int, chunk: int = 1 << 17, progress=None):
    """(means, variances) float64 (depth, width) from n samples."""
    dev = torch.device("cuda")
    W = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).to(dev)
    depth, width = W.shape[0], W.shape[2]
    s1 = torch.zeros((depth, width), dtype=torch.float64, device=dev)
    s2 = torch.zeros((depth, width), dtype=torch.float64, device=dev)
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    x_buf = torch.empty((chunk, width), dtype=torch.float32, device=dev)
    done = 0
    k = 0
    n_chunks = (n + chunk - 1) // chunk
    while done < n:
        b = min(chunk, n - done)
        x = x_buf[:b]
        torch.randn((b, width), out=x, generator=g)
        h = x
        for li in range(depth):
            h = torch.mm(h, W[li]).clamp_min_(0.0)
            # Within-chunk reductions in float32 (tree-reduced by torch: max relative
            # error 2.3e-7 on a 131k-row chunk, measured against a float64 sum --
            # 100x below the N = 1e9 MC noise), accumulated ACROSS chunks in float64.
            # A float64 in-chunk reduction halves throughput on a consumer GPU.
            s1[li] += h.sum(dim=0).double()
            s2[li] += torch.linalg.vector_norm(h, dim=0).double().square()
        done += b
        k += 1
        if progress is not None and k % 64 == 0:
            progress(k / n_chunks)
    m1 = (s1 / n).cpu().numpy()
    m2 = (s2 / n).cpu().numpy()
    return m1, np.maximum(m2 - m1 * m1, 0.0)


out_path = gen.label_path


def jobs_from_args(a) -> list:
    """[(tag, input_seed, width, expected_sha_or_None), ...]"""
    if a.panel:
        rec = gen.load_seeds()["panels"][a.panel]
        idx = rec.get("n1e9_subset") if a.n1e9_subset else range(len(rec["seeds"]))
        idx = [i for i in idx if a.lo <= i < (a.hi if a.hi is not None else 10**9)]
        return [(a.panel, rec["seeds"][i], gen.WIDTH, rec["weight_sha256"][i]) for i in idx]
    if a.mini:
        lo, hi = (int(t) for t in a.mini.split(":"))
        dev = json.loads((gen.ROOT / "data" / "panels" / "dev.json").read_text(encoding="utf-8"))
        return [("mini", dev["seeds"][i], gen.WIDTH,
                 dev["weight_sha256"][i] if "weight_sha256" in dev else None)
                for i in range(lo, hi)]
    if a.lane_m:
        width, lo, hi = (int(t) for t in a.lane_m.split(":"))
        return [(f"lane_m_w{width}", gen.lane_m_seed(width, i), width, None)
                for i in range(lo, hi)]
    raise SystemExit("choose --panel, --mini or --lane-m")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", choices=["fresh_a", "fresh_lock"])
    ap.add_argument("--n1e9-subset", action="store_true")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=None)
    ap.add_argument("--mini", help="lo:hi organizer mini rows (validation)")
    ap.add_argument("--lane-m", help="width:lo:hi width-override nets")
    ap.add_argument("--n", type=float, default=1e8)
    ap.add_argument("--chunk", type=int, default=1 << 17)
    ap.add_argument("--depth", type=int, default=gen.DEPTH)
    a = ap.parse_args(argv)
    n = int(a.n)
    setup_numerics()
    jobs = jobs_from_args(a)
    prog_path = os.environ.get("WORKERQ_PROGRESS")
    meta_common = {"torch": torch.__version__, "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0), "allow_tf32": False,
                   "matmul": "float32 highest", "accum": "float32 in-chunk, float64 across chunks", "chunk": a.chunk}
    print(json.dumps(dict(meta_common, jobs=len(jobs), n=n)), flush=True)
    for j, (tag, seed, width, sha) in enumerate(jobs):
        op = out_path(tag, seed, n)
        if op.exists():
            continue
        w = gen.mint_numpy(seed, width, a.depth)
        h = gen.sha256_weights(w)
        if sha is not None and h != sha:
            raise SystemExit(f"{tag} seed {seed}: weight sha {h[:12]} != recorded {sha[:12]}")
        ss = sample_seed(seed, n)

        def prog(f, j=j):
            if prog_path:
                try:
                    Path(prog_path).write_text(f"{(j + f) / len(jobs):.4f}")
                except OSError:
                    pass

        t0 = time.perf_counter()
        means, var = label(w, n, ss, a.chunk, prog)
        dt = time.perf_counter() - t0
        op.parent.mkdir(parents=True, exist_ok=True)
        tmp = op.with_name(op.stem + ".tmp.npz")
        floor_layers = var.mean(axis=1) / n
        np.savez(tmp, means=means, var=var, n=np.int64(n), input_seed=np.int64(seed),
                 sample_seed=np.int64(ss), width=np.int64(width), weight_sha256=h,
                 seconds=dt, meta=json.dumps(meta_common))
        os.replace(tmp, op)
        print(json.dumps({"tag": tag, "seed": seed, "width": width, "n": n,
                          "seconds": round(dt, 1), "floor_last": float(floor_layers[-1]),
                          "floor_l1": float(floor_layers[0]),
                          "mean_last": float(means[-1].mean())}), flush=True)
        if prog_path:
            Path(prog_path).write_text(f"{(j + 1) / len(jobs):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
