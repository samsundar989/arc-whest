"""
MINT PROVENANCE — is our MLP the grader's MLP?

Every offline number we have ever quoted rests on an unstated assumption: that a
network minted by `local_sim` is drawn from the same distribution, with the same
seed semantics, as one minted by the evaluator. This module removes the
assumption and replaces it with a proof.

The authority is `whestbench.generation.sample_mlp` driven by
`whestbench.seeds.derive_seed_streams`, which is what both `dataset.bake_*` (the
public splits) and `scoring.make_contest` (the private re-run suites) call:

    weight_ss, sample_ss, estimator_seed = SeedSequence(input_seed).spawn(3)
    mlp = sample_mlp(width, depth, default_rng(weight_ss), seed=estimator_seed)

Three facts make a pure-numpy reimplementation possible and checkable:

  * `fnp.random.SeedSequence` **is** `numpy.random.SeedSequence` (identity, not
    a lookalike), so the spawn tree is numpy's.
  * `fnp.random.default_rng` wraps a numpy `Generator` and forwards
    `standard_normal` to it unchanged.
  * `sample_mlp` draws in **float64** and casts once at the end:
    `(rng.standard_normal((w, w)) * scale).astype(float32)`.

That last point is where our old generator diverged: it drew natively in float32
(`standard_normal(..., dtype=float32)`), which consumes the bit stream by a
different algorithm and produces entirely different numbers — same distribution,
different realisation. It also used the *weight* seed as `MLP.seed`, whereas the
evaluator hands the estimator an **independent** substream (spawn[2]).

`canonical_mint` below is the reimplementation; `verify()` proves it equals the
real package bit-for-bit and shows the legacy generator does not.

Run:
    uv run python local_sim/mint_provenance.py            # the proof
    uv run python local_sim/mint_provenance.py --seeds 40
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


# --------------------------------------------------------------------------
# The canonical mint — pure numpy, no competition packages required.
# --------------------------------------------------------------------------

def seed_streams(input_seed):
    """Reimplementation of `whestbench.seeds.derive_seed_streams`.

    Returns (weight_ss, sample_ss, estimator_seed). The three substreams are
    independent: weights, ground-truth draws, and the seed the estimator sees.
    """
    weight_ss, sample_ss, estimator_ss = np.random.SeedSequence(int(input_seed)).spawn(3)
    return weight_ss, sample_ss, int(estimator_ss.generate_state(1)[0])


def canonical_weights(width, depth, weight_ss):
    """Reimplementation of `whestbench.generation.sample_mlp`'s weight draw.

    float64 normal draw, float64 scale multiply, single cast to float32 — the
    order of operations is load-bearing for bit-identity.
    """
    rng = np.random.default_rng(weight_ss)
    scale = float(np.sqrt(2.0 / width))
    return [
        (rng.standard_normal((width, width)) * scale).astype(np.float32)
        for _ in range(depth)
    ]


def canonical_mint(width, depth, input_seed):
    """A grader-identical MLP.

    Returns (weights, estimator_seed, sample_ss):
      weights        list of `depth` (width, width) float32 arrays, stored
                     (input, output) — forward is h <- ReLU(h @ W).
      estimator_seed the integer the evaluator puts in `MLP.seed`.
      sample_ss      the ground-truth SeedSequence, so our MC labels use the
                     evaluator's own sampling substream too.
    """
    weight_ss, sample_ss, estimator_seed = seed_streams(input_seed)
    return canonical_weights(width, depth, weight_ss), estimator_seed, sample_ss


def canonical_mlp(width, depth, input_seed):
    """`canonical_mint` wrapped in a whestbench-shim MLP object (for estimator runs)."""
    sys.path[:0] = [p for p in (_HERE, _ROOT) if p not in sys.path]
    from whestbench import MLP  # the local shim

    weights, est_seed, _ = canonical_mint(width, depth, input_seed)
    return MLP(width=width, depth=depth, weights=weights, seed=est_seed)


# --------------------------------------------------------------------------
# The proof
# --------------------------------------------------------------------------

def _import_real_whestbench():
    """Import the REAL whestbench, evicting the local shim from the path."""
    saved_path = list(sys.path)
    saved = {k: v for k, v in sys.modules.items()
             if k == "whestbench" or k.startswith("whestbench.")}
    sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
    for k in saved:
        del sys.modules[k]
    try:
        import whestbench.generation as wg
        import whestbench.seeds as ws

        return wg, ws
    finally:
        sys.path = saved_path
        for k, v in saved.items():
            sys.modules[k] = v


def _legacy_weights(width, depth, seed):
    """The generator we used until 2026-08-06 (`harness.gen_mlp`), for contrast."""
    rng = np.random.default_rng(seed)
    scale = np.float32(np.sqrt(2.0 / width))
    return [rng.standard_normal((width, width), dtype=np.float32) * scale
            for _ in range(depth)]


def verify(n_seeds=20,
           # 1024x16 FIRST: it is the LIVE Phase-2 shape, and until 2026-08-24
           # this list did not contain it -- the gate proved the mint on five
           # shapes the competition no longer uses. It is checkable directly
           # now only because the environment carries whestbench 0.16, the
           # release the Phase-2 data was actually baked with; before that the
           # hole had to be closed sideways, against the organisers' published
           # bytes (local_sim/phase2_mint_parity.py), which remains the
           # stronger end-to-end check and is still worth running.
           shapes=((1024, 16), (256, 32), (256, 8), (128, 16), (64, 4), (256, 2))):
    """Bit-identity of `canonical_mint` against the installed whestbench."""
    import flopscope.numpy as fnp

    wg, ws = _import_real_whestbench()

    print("=" * 88)
    print("MINT PROVENANCE PROOF — canonical_mint  vs  whestbench.sample_mlp")
    print("=" * 88)
    print(f"whestbench   : {getattr(wg, '__file__', '?')}")
    print(f"SeedSequence : fnp.random.SeedSequence is np.random.SeedSequence = "
          f"{fnp.random.SeedSequence is np.random.SeedSequence}")
    print()
    print(f"{'shape':>10} {'seeds':>6} {'weights bit-identical':>23} "
          f"{'MLP.seed identical':>20} {'legacy identical':>17}")
    print("-" * 88)

    all_ok = True
    legacy_ever_matched = False
    max_legacy_reldiff = 0.0
    for width, depth in shapes:
        w_ok = s_ok = 0
        legacy_ok = 0
        for k in range(n_seeds):
            input_seed = 1_000_003 * (k + 1) + 17 * width + depth

            # --- the authority ---
            weight_ss, _sample_ss, est_seed = ws.derive_seed_streams(input_seed)
            ref = wg.sample_mlp(width, depth, fnp.random.default_rng(weight_ss),
                                seed=est_seed)
            ref_w = [np.asarray(x) for x in ref.weights]

            # --- our reimplementation ---
            mine_w, mine_seed, _ = canonical_mint(width, depth, input_seed)

            if all(a.dtype == b.dtype and a.shape == b.shape
                   and np.array_equal(a.view(np.uint32), b.view(np.uint32))
                   for a, b in zip(ref_w, mine_w)):
                w_ok += 1
            else:
                all_ok = False
            if mine_seed == int(ref.seed) == est_seed:
                s_ok += 1
            else:
                all_ok = False

            # --- the generator we used before today ---
            leg = _legacy_weights(width, depth, input_seed)
            if all(np.array_equal(a, b) for a, b in zip(ref_w, leg)):
                legacy_ok += 1
                legacy_ever_matched = True
            d = max(float(np.abs(a - b).max() / (np.abs(a).max() + 1e-30))
                    for a, b in zip(ref_w, leg))
            max_legacy_reldiff = max(max_legacy_reldiff, d)

        print(f"{width}x{depth:<6} {n_seeds:>6} {f'{w_ok}/{n_seeds}':>23} "
              f"{f'{s_ok}/{n_seeds}':>20} {f'{legacy_ok}/{n_seeds}':>17}")

    print("-" * 88)
    print(f"canonical_mint is bit-identical to whestbench.sample_mlp : {all_ok}")
    print(f"legacy harness.gen_mlp ever matched                      : {legacy_ever_matched}")
    print(f"legacy max relative elementwise divergence               : {max_legacy_reldiff:.3f}")
    print()

    # --- distributional sanity: the legacy nets were the right *distribution* ---
    width, depth = 256, 8
    ref_all, leg_all = [], []
    for k in range(12):
        s = 7_777_777 + k
        ref_all.append(np.stack(canonical_mint(width, depth, s)[0]))
        leg_all.append(np.stack(_legacy_weights(width, depth, s)))
    ref_all, leg_all = np.concatenate(ref_all), np.concatenate(leg_all)
    target = np.sqrt(2.0 / width)
    print("DISTRIBUTIONAL CHECK (12 x 256x8 nets, He target std = %.6f)" % target)
    print(f"  canonical  mean {ref_all.mean():+.3e}   std {ref_all.std(ddof=1):.8f}   "
          f"std/target {ref_all.std(ddof=1) / target:.6f}")
    print(f"  legacy     mean {leg_all.mean():+.3e}   std {leg_all.std(ddof=1):.8f}   "
          f"std/target {leg_all.std(ddof=1) / target:.6f}")
    print("  -> same distribution, different realisation. Past *statistical* results")
    print("     stand; per-seed reproducibility against the grader did not exist.")
    print()

    # --- seed semantics: the part that could actually have biased a measurement ---
    print("SEED SEMANTICS")
    ex = 123456789
    _ws, _ss, es = seed_streams(ex)
    print(f"  input_seed        {ex}")
    print(f"  estimator seed    {es}   <- what MLP.seed holds on the grader")
    print(f"  legacy MLP.seed   {ex}   <- what our mints used (== the weight seed)")
    print("  The evaluator's estimator seed is an INDEPENDENT substream of the weight")
    print("  seed. Our mints coupled them, so any estimator that derives randomness")
    print("  from mlp.seed (v2.4.1 derives its digital shift from it) had a spurious")
    print("  functional dependence between its point set and the weights it scored.")
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    ok = verify(n_seeds=args.seeds)
    print()
    print("VERDICT:", "PROVEN — canonical_mint == whestbench.sample_mlp"
          if ok else "FAILED — do not use canonical_mint until this passes")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
