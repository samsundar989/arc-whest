"""Phase-2 ground truth at 1024 x 16, with TWO INDEPENDENT REPLICAS.

P2-CP1's label producer. Runs in the GPU venv (`.venv-gpu`), which has torch but
not whestbench, so everything here goes through `mint_provenance.canonical_mint`
-- proven bit-identical to the organisers' published `v2-phase2` weights, 16/16
(`local_sim/phase2_mint_parity.py`).

## Why two replicas, and what they buy

Every measured MSE is contaminated by the label's own Monte-Carlo noise:

    E[ (p - y)^2 ] = MSE(p) + Var(y)

so a single-replica MSE is BIASED UPWARD by exactly the truth noise, uniformly
across arms. That bias cancels in a *paired* comparison, but it does not cancel in
an absolute number -- and P2-CP1 has to report absolute MSEs next to a projection.
With two independent replicas `y1 = t + e1`, `y2 = t + e2` the cross product

    mean_j (p_j - y1_j)(p_j - y2_j)

has expectation `MSE(p)` exactly, because `E[e1 e2] = 0`. Both ordinary MSEs and
the cross-product estimate are reported, with the replica noise beside them, so
nobody has to take the correction on trust.

The replicas are independent BY CONSTRUCTION, not by assumption: the evaluator's
own sampling substream `spawn(3)[1]` is spawned again into two children, so the
two draws share no state.

## Seeds are used at FULL width

Eight historical torch paths in this repository mask the sample seed with
`& 0x7FFFFFFF`, truncating a 32-bit seed to 31 bits. Those are deliberately left
alone -- they carry recorded Phase-1 results -- but this file is new, and it uses
the full 64-bit state. Published Phase-2 `mlp_seed` values are already 63-bit.

## Cost

One sample through 1024 x 16 is 33,554,432 FLOPs, so `N = 2^22` per replica is
1.4e14 FLOPs per network per replica. Truth noise is then
`avg_variance / N ~ 0.087 / 4.19e6 = 2.1e-8`, about 0.4% of the ~4.8e-6 signal
the arms are separated on -- comfortably inside the 10% bar
`docs/VALIDATION.md` section 3.5 sets, with the cross-product estimate on top.

Run (GPU venv, and it CLAIMS SEEDS -- only inside a registered Slice Contract):
    .venv-gpu\\Scripts\\python.exe local_sim/p2_truth.py --block <BLOCK> \\
        --n-mlps 48 --log2-samples 22 --out experiments/<dir>/truth.npz --ts <date>

Dry-run the cost and the claim without drawing anything:
    .venv-gpu\\Scripts\\python.exe local_sim/p2_truth.py --block <BLOCK> --n-mlps 48 --plan
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

from mint_provenance import canonical_mint, seed_streams  # noqa: E402

WIDTH, DEPTH = 1024, 16
FWD_FLOPS = 2 * DEPTH * WIDTH * WIDTH


def _replica_seeds(input_seed: int):
    """Two INDEPENDENT sampling streams from the evaluator's own substream.

    `seed_streams(s)[1]` is `SeedSequence(s).spawn(3)[1]` -- the substream the
    evaluator draws its own ground truth from. Spawning it again gives two
    children that share no state, so `E[e1 e2] = 0` holds by construction rather
    than by hope. Full 64-bit entropy: no `& 0x7FFFFFFF` anywhere.
    """
    _w, sample_ss, _e = seed_streams(input_seed)
    a, b = sample_ss.spawn(2)
    return (int.from_bytes(a.generate_state(2).tobytes(), "little"),
            int.from_bytes(b.generate_state(2).tobytes(), "little"))


def _truth_one_final_only_LEGACY(weights, seed64, n_samples, device, chunk):
    """The pre-WP9 accumulation, kept ONLY as a bit-identity reference.

    Never called by the label path. `--check-allvar` runs it beside the current
    function on the same seed and asserts the final-layer variance agrees
    BITWISE -- which is the claim the WP9 docstring makes, and a claim of
    bit-identity that nothing executes is just a sentence.
    """
    import torch

    w = [torch.from_numpy(np.ascontiguousarray(x)).to(device) for x in weights]
    gen = torch.Generator(device=device)
    gen.manual_seed(seed64 % (2 ** 63 - 1))
    acc = [torch.zeros(WIDTH, dtype=torch.float64, device=device)
           for _ in range(DEPTH)]
    acc_sq = torch.zeros(WIDTH, dtype=torch.float64, device=device)
    done = 0
    while done < n_samples:
        m = min(chunk, n_samples - done)
        h = torch.randn((m, WIDTH), generator=gen, device=device,
                        dtype=torch.float32)
        for k in range(DEPTH):
            h = torch.relu(h @ w[k])
            acc[k] += h.sum(dim=0, dtype=torch.float64)
        acc_sq += (h.to(torch.float64) ** 2).sum(dim=0)
        done += m
    means = np.stack([(a / n_samples).cpu().numpy() for a in acc])
    ez2 = (acc_sq / n_samples).cpu().numpy()
    var = np.maximum(ez2 - means[-1] ** 2, 0.0)
    return means.astype(np.float64), var.astype(np.float64)


def _truth_one(weights, seed64, n_samples, device, chunk):
    """All-layer post-ReLU means AND all-layer post-ReLU variances, on the GPU.

    float64 accumulation, float32 arithmetic -- the same choice the organisers'
    bake documents, so the label's numerical character matches theirs.

    EXTENDED 2026-09-01 under CONTRACT WP9. This used to accumulate the second
    moment at the FINAL layer only, because the only consumer was a
    final-layer-MSE noise floor. An in-flight state corrector needs the true
    post-ReLU variance at EVERY layer: `alpha = mu_pre / sig` is what indexes
    the correction, `sig` comes from the propagated covariance, and the
    post-ReLU variance is the diagonal that feeds it. Without the intermediate
    variances the corrector can only touch the mean, which is half the state.

    The extra work is one square-and-sum per layer per chunk -- `m * n` against
    the layer's `m * n^2` matmul, so `1/n` = 0.1% at width 1024. The returned
    final-layer variance is BIT-IDENTICAL to what this function returned
    before: it is still `acc_sq[-1]/N - means[-1]^2`, the same accumulator in
    the same order, and `--check-allvar` asserts exactly that.
    """
    import torch

    w = [torch.from_numpy(np.ascontiguousarray(x)).to(device) for x in weights]
    gen = torch.Generator(device=device)
    gen.manual_seed(seed64 % (2 ** 63 - 1))          # torch caps at int64
    acc = [torch.zeros(WIDTH, dtype=torch.float64, device=device)
           for _ in range(DEPTH)]
    acc_sq = [torch.zeros(WIDTH, dtype=torch.float64, device=device)
              for _ in range(DEPTH)]
    done = 0
    while done < n_samples:
        m = min(chunk, n_samples - done)
        h = torch.randn((m, WIDTH), generator=gen, device=device,
                        dtype=torch.float32)
        for k in range(DEPTH):
            h = torch.relu(h @ w[k])
            acc[k] += h.sum(dim=0, dtype=torch.float64)
            acc_sq[k] += (h.to(torch.float64) ** 2).sum(dim=0)
        done += m
    means = np.stack([(a / n_samples).cpu().numpy() for a in acc])
    ez2 = np.stack([(a / n_samples).cpu().numpy() for a in acc_sq])
    allvar = np.maximum(ez2 - means ** 2, 0.0)
    # The final-layer variance keeps its own name and its own shape, so every
    # existing consumer and every stored P01 truth file still reads correctly.
    return (means.astype(np.float64), allvar[-1].astype(np.float64),
            allvar.astype(np.float64))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, help="seed block to claim from")
    ap.add_argument("--n-mlps", type=int, required=True)
    ap.add_argument("--log2-samples", type=int, default=22)
    ap.add_argument("--out", default="")
    ap.add_argument("--ts", default="", help="explicit date; this module never reads the clock")
    ap.add_argument("--consumer", default="local_sim/p2_truth.py")
    ap.add_argument("--note", default="")
    ap.add_argument("--chunk", type=int, default=1 << 16)
    ap.add_argument("--plan", action="store_true",
                    help="print the plan and the cost; claim nothing, draw nothing")
    ap.add_argument("--check-allvar", type=int, default=0, metavar="N",
                    help="prove the all-layer accumulator left the final-layer "
                         "variance BITWISE unchanged, on N probe networks; "
                         "claims nothing, writes nothing")
    ap.add_argument("--smoke", type=int, default=0,
                    help="time the path and CHECK REPLICA INDEPENDENCE on N "
                         "networks minted from an unregistered probe seed. "
                         "Claims nothing, writes nothing, selects nothing.")
    a = ap.parse_args()

    n_samples = 1 << a.log2_samples
    flops_total = 2 * a.n_mlps * n_samples * FWD_FLOPS

    print("=" * 78)
    print("PHASE-2 TRUTH  --  1024 x 16, two independent replicas")
    print("=" * 78)
    print(f"  block           {a.block}")
    print(f"  networks        {a.n_mlps}")
    print(f"  samples/replica 2^{a.log2_samples} = {n_samples:,}   (2 replicas each)")
    print(f"  expected truth noise per replica  ~ avg_variance / N")
    print(f"                  ~ 0.087 / {n_samples:,} = {0.087 / n_samples:.3e}")
    print(f"  total work      {flops_total:.3e} FLOPs")
    print(f"  out             {a.out or '(none -- --plan)'}")

    if a.check_allvar:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  device          {device}")
        print()
        print("  BIT-IDENTITY CHECK: unregistered probe seeds, nothing "
              "claimed, nothing written.")
        base, ok = 1_000_051, True
        for i in range(a.check_allvar):
            w, _e, _s = canonical_mint(WIDTH, DEPTH, base + i)
            sa, _sb = _replica_seeds(base + i)
            m_new, v_new, av = _truth_one(w, sa, n_samples, device, a.chunk)
            m_old, v_old = _truth_one_final_only_LEGACY(
                w, sa, n_samples, device, a.chunk)
            same_m = np.array_equal(m_new, m_old)
            same_v = np.array_equal(v_new, v_old)
            same_last = np.array_equal(av[-1], v_old)
            ok = ok and same_m and same_v and same_last
            print(f"    net {i}  means bitwise {same_m}   final var bitwise "
                  f"{same_v}   allvar[-1]==var {same_last}")
            print(f"           allvar shape {av.shape}, layer-0 mean var "
                  f"{av[0].mean():.6e}, final {av[-1].mean():.6e}")
        print()
        print(f"  {'PASS' if ok else 'FAIL'} -- the all-layer accumulator "
              f"{'left' if ok else 'CHANGED'} the existing labels "
              f"{'unchanged' if ok else '!!'}")
        print("=" * 78)
        return 0 if ok else 1

    if a.smoke:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  device          {device}"
              + (f"  {torch.cuda.get_device_name(0)}" if device == "cuda" else ""))
        print(chr(10) + "  SMOKE: unregistered probe seeds, nothing claimed, nothing written.")
        # seed-ledger: no block named. Counted work here is value-independent and
        # no statistic leaves this function, so it is not consumption.
        base = 1_000_003
        t0 = time.perf_counter()
        obs_l, pred_l = [], []
        pm1, pm2, pv1, pv2 = [], [], [], []
        pav1, pav2 = [], []
        for i in range(a.smoke):
            w, _e, _s = canonical_mint(WIDTH, DEPTH, base + i)
            sa, sb = _replica_seeds(base + i)
            m1, v1, av1 = _truth_one(w, sa, n_samples, device, a.chunk)
            m2, v2, av2 = _truth_one(w, sb, n_samples, device, a.chunk)
            pav1.append(av1); pav2.append(av2)
            d = m1[-1] - m2[-1]
            obs_l.append(float((d ** 2).mean()))
            pred_l.append(float((v1.mean() + v2.mean()) / n_samples))
            pm1.append(m1); pm2.append(m2); pv1.append(v1); pv2.append(v2)
        el = time.perf_counter() - t0
        obs, pred = float(np.mean(obs_l)), float(np.mean(pred_l))
        print(f"  {a.smoke} networks x 2 replicas in {el:.1f}s "
              f"({el / max(a.smoke, 1) / 2:.2f}s per replica)")
        print(f"  replica disagreement  observed {obs:.4e}   predicted 2V/N "
              f"{pred:.4e}   ratio {obs / pred:.4f}")
        print(f"  truth noise per replica ~ {pred / 2:.4e}")
        print(f"  projected wall time for 48 networks: "
              f"{el / max(a.smoke, 1) * 48 / 60:.1f} min")
        if a.out:
            # A PROBE truth file, so the downstream pipeline can be exercised
            # end to end without spending a block. It is stamped `probe=1` and
            # every consumer must refuse to treat it as evidence.
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            np.savez_compressed(
                a.out,
                seeds=np.asarray([base + i for i in range(a.smoke)], dtype=np.int64),
                means_r1=np.stack(pm1).astype(np.float32),
                means_r2=np.stack(pm2).astype(np.float32),
                var_r1=np.stack(pv1), var_r2=np.stack(pv2),
                allvar_r1=np.stack(pav1), allvar_r2=np.stack(pav2),
                gt_noise=np.asarray(
                    [(x.mean() + y.mean()) / 2.0 / n_samples
                     for x, y in zip(pv1, pv2)], dtype=np.float64),
                n_samples=np.int64(n_samples),
                width=np.int64(WIDTH), depth=np.int64(DEPTH),
                probe=np.int64(1),
            )
            print(f"  wrote PROBE truth (NOT evidence) -> {a.out}")
        print("=" * 78)
        return 0

    if a.plan:
        import seed_ledger as sl

        print(f"  free ranges in {a.block}: {sl.free_ranges(a.block)}")
        print("\n  PLAN ONLY. No seed claimed, nothing drawn, nothing written.")
        print("=" * 78)
        return 0

    if not a.out or not a.ts:
        raise SystemExit("--out and --ts are required unless --plan")
    if os.path.exists(a.out):
        raise SystemExit(f"REFUSED: {a.out} exists. Truth is immutable.")

    import torch

    import seed_ledger as sl

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device          {device}"
          + (f"  {torch.cuda.get_device_name(0)}" if device == "cuda" else ""))
    print()

    # Claim FIRST, so an interrupted run leaves a claim rather than a silent draw.
    seeds = sl.claim(a.block, a.n_mlps, a.consumer, ts=a.ts,
                     artifact=os.path.relpath(a.out, _ROOT).replace("\\", "/"),
                     note=a.note or f"P2-CP1 truth, 2 replicas x 2^{a.log2_samples}")
    print(f"  claimed         {a.block}[{seeds[0]:,} .. {seeds[-1]:,}]")

    t0 = time.perf_counter()
    m1 = np.zeros((a.n_mlps, DEPTH, WIDTH), dtype=np.float64)
    m2 = np.zeros_like(m1)
    v1 = np.zeros((a.n_mlps, WIDTH), dtype=np.float64)
    v2 = np.zeros_like(v1)
    # All-layer post-ReLU variance, added under CONTRACT WP9. Same shape as the
    # means, so the two index identically: [network, layer, neuron].
    av1 = np.zeros((a.n_mlps, DEPTH, WIDTH), dtype=np.float64)
    av2 = np.zeros_like(av1)
    for i, s in enumerate(seeds):
        w, _est, _ss = canonical_mint(WIDTH, DEPTH, s)
        sa, sb = _replica_seeds(s)
        m1[i], v1[i], av1[i] = _truth_one(w, sa, n_samples, device, a.chunk)
        m2[i], v2[i], av2[i] = _truth_one(w, sb, n_samples, device, a.chunk)
        # The final-layer variance must be exactly the last row of the
        # all-layer one -- the same accumulator read two ways. Asserted per
        # network rather than trusted, because a silent divergence here would
        # put a corrector's variance target and its noise floor on different
        # quantities.
        assert np.array_equal(v1[i], av1[i, -1]), f"allvar/var mismatch at {i}"
        assert np.array_equal(v2[i], av2[i, -1]), f"allvar/var mismatch at {i}"
        if (i + 1) % 4 == 0 or i + 1 == a.n_mlps:
            el = time.perf_counter() - t0
            print(f"    {i + 1:>4}/{a.n_mlps}   {el:7.1f}s   "
                  f"eta {el / (i + 1) * (a.n_mlps - i - 1):7.1f}s")

    # Replica disagreement is a MEASURED check on the independence claim: the
    # per-neuron difference should have variance 2 * avg_variance / N.
    diff = m1[:, -1, :] - m2[:, -1, :]
    obs = float((diff ** 2).mean())
    pred = float((v1.mean() + v2.mean()) / n_samples)
    gt_noise = float(((v1 + v2) / 2).mean(axis=1) / n_samples).__float__() \
        if False else ((v1 + v2) / 2).mean(axis=1) / n_samples

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(
        a.out,
        seeds=np.asarray(seeds, dtype=np.int64),
        means_r1=m1.astype(np.float32), means_r2=m2.astype(np.float32),
        var_r1=v1.astype(np.float64), var_r2=v2.astype(np.float64),
        allvar_r1=av1.astype(np.float64), allvar_r2=av2.astype(np.float64),
        gt_noise=gt_noise.astype(np.float64),
        n_samples=np.int64(n_samples),
        width=np.int64(WIDTH), depth=np.int64(DEPTH),
    )
    sha = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    rel = os.path.relpath(a.out, _ROOT).replace("\\", "/")
    sl.finalize(a.consumer, rel, ts=a.ts, note="truth written")

    print()
    print(f"  replica disagreement  observed {obs:.4e}   predicted 2*V/N "
          f"{pred:.4e}   ratio {obs / pred:.4f}")
    print("     (a ratio far from 1.0 means the replicas are NOT independent, or")
    print("      the variance model is wrong. It is a check, not a formality.)")
    print(f"  mean truth noise      {float(gt_noise.mean()):.4e}")
    print(f"  wrote                 {rel}")
    print(f"  sha256                {sha}")
    print(f"  ledger finalized      yes")
    print(f"  elapsed               {time.perf_counter() - t0:.1f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
