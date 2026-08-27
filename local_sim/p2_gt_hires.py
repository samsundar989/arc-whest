"""P2-GT: high-resolution A/B ground truth at 1024 x 16, stored as two separate halves.

Contract: `experiments/LOG.md`, entry "CONTRACT (Rule 1): P2-GT", committed in
`ca18439` **before** this module claimed a seed or drew a sample.

## Why this exists and why the existing asset is not enough

`experiments/2026-08-21_p2_cp1/truth_P01.npz` is `2**22` samples per replica, so
its own Monte-Carlo noise is `~2.1e-8`. That was fine when the arms under test
were separated at `~4.5e-6`. The accuracy band the live board's leaders occupy is
**`3e-8` raw final-layer MSE** -- the *same order as the old label's noise*. You
cannot measure a quantity with an instrument whose noise is the size of the
quantity, so the label has to get quieter before the next question can be asked
at all.

At `2**26 = 67,108,864` samples per half the per-neuron label noise is `~1.3e-9`,
23x below `3e-8`, and the unbiased cross estimator below is quieter still.

## The estimator this asset exists to serve

Every single-replica MSE is biased UPWARD by exactly the label's own variance:

    E[(p - y)^2] = MSE(p) + Var(y)

With two independent halves `A = t + e_A`, `B = t + e_B` the cross product

    mean_j (p_j - A_j)(p_j - B_j)

has expectation `MSE(p)` **exactly**, because `E[e_A e_B] = 0`. That is the
estimator that settled the K3 noise-floor question, and it is why the halves are
written to **two separate files**: a single file invites a consumer to average
the two into one label, which silently throws the estimator away and puts the
`1.3e-9` bias back.

## What is shared with the existing producer, deliberately

`_truth_one` and `_replica_seeds` are **imported** from `local_sim/p2_truth.py`,
not re-typed. That module's mint is `mint_provenance.canonical_mint`, proven
bit-identical to the organisers' published `v2-phase2` weights 16/16, and its
replica independence is a property of the evaluator's own substream
(`SeedSequence(s).spawn(3)[1]` spawned again into two children) rather than an
assumption. Two assets that disagree about their arithmetic would be worse than
one coarse asset.

Two differences from `p2_truth.py`, both deliberate:

- means and variances are stored in **float64**, not float32. At `2**22` the
  float32 quantisation was ~8 orders below the label noise and irrelevant; at
  `2**26` it is still ~8 orders below, but the asset is meant to outlive the
  next resolution increase and 4 MB per half is not a cost;
- the draw is **checkpointed per network** into a shard directory, so a host that
  dies 50 minutes into a 60-minute run loses one network rather than all of them.
  This host is contended and has already killed one measurement today.

## TF32 is asserted, not assumed

An fp32 matmul that silently runs on TF32 tensor cores has ~`9.8e-4` relative
error. At the `3e-8` scale that would make every number here fiction, and it
would look exactly like a real result. This module refuses to run if TF32 is on.

Run (GPU venv, and it CLAIMS SEEDS -- only inside a registered Slice Contract):

    .venv-gpu\\Scripts\\python.exe local_sim/p2_gt_hires.py --plan
    .venv-gpu\\Scripts\\python.exe local_sim/p2_gt_hires.py \\
        --block P01 --n-mlps 24 --log2-samples 26 \\
        --out-dir experiments/2026-08-26_p2_gt --ts 2026-08-26
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

from mint_provenance import canonical_mint  # noqa: E402
from p2_truth import DEPTH, WIDTH, _replica_seeds, _truth_one  # noqa: E402

# The four numbered predictions of the P2-GT contract, as executable bounds.
# They are here rather than in a report so that the run itself decides, and so
# that a future reader can see what was promised without trusting a summary.
PRED_RATIO = (0.80, 1.20)          # prediction 1: replica disagreement ratio
PRED_NOISE = (1.1e-9, 1.6e-9)      # prediction 2: mean truth noise per half
PRED_HALF_AGREE = (0.8, 1.2)       # prediction 4: x the A-PRIORI 2 * 0.087 / N
TARGET_MSE = 3.0e-8                # the quantity every downstream gate is on
SE_BAR = 3.0e-9                    # gate: cross-estimator SE must beat this


def _assert_fp32_is_fp32() -> dict:
    """Refuse to draw a label set on tensor cores pretending to be float32."""
    import torch

    facts = {
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    if facts["allow_tf32_matmul"] or facts["float32_matmul_precision"] != "highest":
        raise SystemExit(
            "REFUSED: TF32 is enabled. A TF32 matmul carries ~9.8e-4 relative "
            "error; this asset is meant to resolve 3e-8. Set "
            "torch.backends.cuda.matmul.allow_tf32 = False and "
            "torch.set_float32_matmul_precision('highest')."
        )
    # Direct evidence, not just a flag: fp32 eps is 1.19e-7, TF32 eps is 9.8e-4.
    g = torch.Generator(device="cuda"); g.manual_seed(11)
    a = torch.randn((2048, WIDTH), generator=g, device="cuda", dtype=torch.float32)
    b = torch.randn((WIDTH, WIDTH), generator=g, device="cuda", dtype=torch.float32)
    rel = float(((a @ b).double() - a.double() @ b.double()).abs()
                .div(( a.double() @ b.double()).abs() + 1e-12).median())
    facts["measured_median_rel_err"] = rel
    if rel > 1e-5:
        raise SystemExit(
            f"REFUSED: measured fp32 matmul relative error {rel:.3e} is far above "
            "fp32 eps (1.19e-7). Something is downcasting the arithmetic."
        )
    return facts


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _shard(shard_dir: str, i: int, seed: int) -> str:
    return os.path.join(shard_dir, f"net_{i:03d}_seed_{seed}.npz")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="P01")
    ap.add_argument("--n-mlps", type=int, default=24)
    ap.add_argument("--log2-samples", type=int, default=26)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--ts", default="", help="explicit date; this module never reads the clock")
    ap.add_argument("--consumer", default="local_sim/p2_gt_hires.py")
    ap.add_argument("--chunk", type=int, default=1 << 17)
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()

    n = 1 << a.log2_samples
    fwd = 2 * DEPTH * WIDTH * WIDTH
    work = 2 * a.n_mlps * n * fwd
    noise = 0.087 / n

    print("=" * 78)
    print("P2-GT  --  high-resolution A/B ground truth, 1024 x 16")
    print("=" * 78)
    print(f"  block            {a.block}")
    print(f"  networks         {a.n_mlps}")
    print(f"  samples/half     2^{a.log2_samples} = {n:,}   (2 independent halves each)")
    print(f"  total per MLP    {2 * n:,}")
    print(f"  predicted noise  ~0.087/N = {noise:.3e} per half")
    print(f"  cross-est SE at MSE={TARGET_MSE:.1e}: "
          f"{(2 * noise * TARGET_MSE / WIDTH) ** 0.5:.3e}  (bar {SE_BAR:.1e})")
    print(f"  host GPU work    {work:.3e} FLOPs")

    if a.plan:
        import seed_ledger as sl
        print(f"  free in {a.block}     {sl.free_ranges(a.block)}")
        print("\n  PLAN ONLY. Nothing claimed, nothing drawn, nothing written.")
        print("=" * 78)
        return 0

    if not a.out_dir or not a.ts:
        raise SystemExit("--out-dir and --ts are required unless --plan")

    out_dir = os.path.abspath(os.path.join(_ROOT, a.out_dir))
    half_a = os.path.join(out_dir, "gt_halfA.npz")
    half_b = os.path.join(out_dir, "gt_halfB.npz")
    manifest = os.path.join(out_dir, "gt_manifest.json")
    for p in (half_a, half_b, manifest):
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists. Truth is immutable.")
    shard_dir = os.path.join(out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    facts = _assert_fp32_is_fp32()
    print(f"  device           {facts['device']}   torch {facts['torch']}")
    print(f"  TF32             OFF (precision={facts['float32_matmul_precision']}, "
          f"measured rel err {facts['measured_median_rel_err']:.3e})")
    print()

    import seed_ledger as sl

    rel_manifest = os.path.relpath(manifest, _ROOT).replace("\\", "/")
    prior = [c for c in sl.claims() if c.consumer == a.consumer
             and c.artifact == rel_manifest]
    if prior:
        seeds = [prior[0].start + i for i in range(prior[0].n)]
        print(f"  RESUMING existing claim {a.block}"
              f"[{seeds[0]:,} .. {seeds[-1]:,}] -- shards are reused, not redrawn.")
    else:
        # Claim FIRST, so an interrupted run leaves a claim rather than a silent draw.
        seeds = sl.claim(a.block, a.n_mlps, a.consumer, ts=a.ts,
                         artifact=rel_manifest,
                         note=f"P2-GT hi-res truth, 2 halves x 2^{a.log2_samples}")
        print(f"  claimed          {a.block}[{seeds[0]:,} .. {seeds[-1]:,}]")

    t0 = time.perf_counter()
    device = "cuda"
    for i, s in enumerate(seeds):
        path = _shard(shard_dir, i, s)
        if os.path.exists(path):
            print(f"    {i + 1:>4}/{len(seeds)}   shard present, skipped")
            continue
        w, _est, _ss = canonical_mint(WIDTH, DEPTH, s)
        sa, sb = _replica_seeds(s)
        mA, vA = _truth_one(w, sa, n, device, a.chunk)
        mB, vB = _truth_one(w, sb, n, device, a.chunk)
        tmp = path + ".part"
        np.savez(tmp, seed=np.int64(s), means_A=mA, var_A=vA,
                 means_B=mB, var_B=vB, n_samples=np.int64(n))
        os.replace(tmp, path)
        el = time.perf_counter() - t0
        done = i + 1
        print(f"    {done:>4}/{len(seeds)}   {el:7.1f}s   "
              f"eta {el / done * (len(seeds) - done):7.1f}s")

    mA = np.zeros((len(seeds), DEPTH, WIDTH), dtype=np.float64)
    mB = np.zeros_like(mA)
    vA = np.zeros((len(seeds), WIDTH), dtype=np.float64)
    vB = np.zeros_like(vA)
    for i, s in enumerate(seeds):
        with np.load(_shard(shard_dir, i, s)) as z:
            if int(z["seed"]) != s or int(z["n_samples"]) != n:
                raise SystemExit(f"REFUSED: shard {i} carries the wrong seed or N.")
            mA[i], vA[i], mB[i], vB[i] = z["means_A"], z["var_A"], z["means_B"], z["var_B"]

    # ---- the contract's four predictions, decided by the run itself ----------
    diff = mA[:, -1, :] - mB[:, -1, :]
    obs = float((diff ** 2).mean())
    pred2vn = float((vA.mean() + vB.mean()) / n)
    ratio = obs / pred2vn
    gt_noise = ((vA + vB) / 2).mean(axis=1) / n
    n_nonfinite = int((~np.isfinite(mA)).sum() + (~np.isfinite(mB)).sum()
                      + (~np.isfinite(vA)).sum() + (~np.isfinite(vB)).sum())
    se_per_mlp = np.sqrt(2.0 * gt_noise * TARGET_MSE / WIDTH)

    checks = {
        "P1_ratio": (PRED_RATIO[0] <= ratio <= PRED_RATIO[1], ratio, PRED_RATIO),
        "P2_noise": (PRED_NOISE[0] <= float(gt_noise.mean()) <= PRED_NOISE[1],
                     float(gt_noise.mean()), PRED_NOISE),
        "P3_finite": (n_nonfinite == 0, n_nonfinite, (0, 0)),
        # P4 is NOT P1 restated. P1 asks whether the observed disagreement matches
        # the variance MEASURED on this very asset; P4 asks whether it matches the
        # A-PRIORI 2 * 0.087 / N written into the contract before any draw. P1 can
        # pass while P4 fails if the networks' average variance is not 0.087.
        "P4_half_agreement_vs_apriori": (
            PRED_HALF_AGREE[0] <= obs / (2 * 0.087 / n) <= PRED_HALF_AGREE[1],
            obs / (2 * 0.087 / n), PRED_HALF_AGREE),
        "GATE_cross_estimator_SE": (float(se_per_mlp.max()) < SE_BAR,
                                    float(se_per_mlp.max()), (0.0, SE_BAR)),
    }

    os.makedirs(out_dir, exist_ok=True)
    for path, m, v, tag in ((half_a, mA, vA, "A"), (half_b, mB, vB, "B")):
        np.savez_compressed(
            path, seeds=np.asarray(seeds, dtype=np.int64),
            means=m, var=v, n_samples=np.int64(n),
            width=np.int64(WIDTH), depth=np.int64(DEPTH), half=tag,
        )

    doc = {
        "slice": "P2-GT",
        "contract_commit": "ca18439",
        "ts": a.ts,
        "block": a.block,
        "seeds": [int(x) for x in seeds],
        "n_mlps": len(seeds),
        "samples_per_half": n,
        "samples_total_per_mlp": 2 * n,
        "width": WIDTH, "depth": DEPTH,
        "dtype_stored": "float64",
        "halves": {
            "A": {"path": os.path.relpath(half_a, _ROOT).replace("\\", "/"),
                  "sha256": _sha(half_a)},
            "B": {"path": os.path.relpath(half_b, _ROOT).replace("\\", "/"),
                  "sha256": _sha(half_b)},
        },
        "numerics": facts,
        "replica_disagreement_observed": obs,
        "replica_disagreement_predicted_2V_over_N": pred2vn,
        "replica_disagreement_ratio": ratio,
        "gt_noise_per_half_mean": float(gt_noise.mean()),
        "gt_noise_per_half_max": float(gt_noise.max()),
        "cross_estimator_se_at_target_mse": {
            "target_mse": TARGET_MSE,
            "max_over_mlps": float(se_per_mlp.max()),
            "mean_over_mlps": float(se_per_mlp.mean()),
            "bar": SE_BAR,
        },
        "nonfinite_values": n_nonfinite,
        "checks": {k: {"pass": bool(p), "value": val, "bound": list(b)}
                   for k, (p, val, b) in checks.items()},
        "all_pass": all(p for p, _v, _b in checks.values()),
        "estimator_note": (
            "Unbiased cross estimator: mean_j (p_j - A_j)(p_j - B_j) has "
            "expectation MSE(p) exactly. NEVER average A and B into one label -- "
            "that reintroduces the +Var(y) bias these two files exist to remove."
        ),
        "tier": "DESIGN -- P01. Nothing measured on this asset may close a lane.",
    }
    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)

    sl.finalize(a.consumer, rel_manifest, ts=a.ts,
                note="P2-GT manifest written; half shas inside")

    print()
    print("  contract predictions, decided by this run:")
    for k, (p, val, b) in checks.items():
        v = f"{val:.4e}" if isinstance(val, float) else str(val)
        print(f"    {'PASS' if p else 'FAIL'}  {k:<26} {v:>12}   bound {b}")
    print()
    print(f"  replica disagreement  observed {obs:.4e}  predicted 2V/N {pred2vn:.4e}"
          f"  ratio {ratio:.4f}")
    print(f"  mean truth noise/half {float(gt_noise.mean()):.4e}")
    print(f"  wrote  {doc['halves']['A']['path']}  {doc['halves']['A']['sha256'][:16]}")
    print(f"  wrote  {doc['halves']['B']['path']}  {doc['halves']['B']['sha256'][:16]}")
    print(f"  wrote  {rel_manifest}")
    print(f"  ledger finalized      yes")
    print(f"  elapsed               {time.perf_counter() - t0:.1f}s")
    print(f"  VERDICT               {'ALL PASS' if doc['all_pass'] else 'FAILED -- asset is VOID'}")
    print("=" * 78)
    return 0 if doc["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
