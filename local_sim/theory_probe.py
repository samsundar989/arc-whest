"""
THEORY VERIFICATION — P1 and P3 from the NNGP correlation-collapse account.

Write-up §7b has to explain a measured fact: the participation ratio of the
activation fluctuations falls 165 -> 25.9 -> 8.8 -> 4.4 -> 2.8 over 32 layers,
and the Gaussian closure's own covariance finds that collapsing subspace while
eigenvectors of W W^T find nothing. The proposed mechanism is NNGP correlation
growth: He init is ReLU-critical, so norms are preserved while input-input
correlations approach 1, and a correlation approaching 1 IS a rank-one
covariance.

**Standing rule for this file: disagreement is a result, not a bug to tune away.**
Nothing here has a free parameter, which is the point — P1 and P3 are either
right or they are not.

  P1  parameter-free correlation map. For He-init ReLU the uncentered
      correlation of two inputs' representations obeys

          c_{l+1} = R(c_l),   R(c) = (1/pi) [ sqrt(1-c^2) + c (pi - arccos c) ]

      the arc-cosine kernel (Cho & Saul 2009), with NO fitted constant: the 2/n
      scaling makes the map exactly norm-preserving. Measure c_l empirically at
      every layer from many input pairs and compare against iterating R from the
      measured c_0.

  P3  top-eigenvector / mean alignment. If the fluctuation covariance is
      collapsing to rank one because every input is being mapped to nearly the
      same representation, its leading eigenvector should align with the mean
      activation direction mu_l. Report |<v_l, mu_l>| / (|v_l| |mu_l|) by depth.

  P2  two-sector participation-ratio fit, held out at layers 12/20/28 —
      NOT IMPLEMENTED. Its functional form is specified in
      docs/nngp-theory-draft.md, which is not in the repo. Implementing a guess
      and calling it a verification would be worse than leaving it undone.

Seed block D07 (30,806,007,000+), fresh mints, never reused.

Run:
    .venv-gpu\\Scripts\\python.exe local_sim/theory_probe.py --mlps 64 --samples 8192
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mint_provenance import canonical_mint  # noqa: E402

WIDTH, DEPTH = 256, 32
BLOCK_D07 = 30_806_007_000
D07_N = 600          # size of the allocated block (seed_blocks.py)
P1_STARTS = 4        # c0 values; P1 consumes P1_STARTS*mlps, P3 the next mlps


def R_map(c):
    """The parameter-free ReLU correlation map. c in [-1, 1]."""
    c = min(1.0, max(-1.0, float(c)))
    return (math.sqrt(max(0.0, 1.0 - c * c)) + c * (math.pi - math.acos(c))) / math.pi


def _seed(offset):
    """A seed inside block D07, asserted. Straying outside an allocated block is
    exactly the failure the registry exists to prevent — including when it is our
    own probe doing the straying."""
    if not 0 <= offset < D07_N:
        raise AssertionError(
            f"theory_probe wants seed offset {offset}, outside block D07's "
            f"{D07_N} seeds. Reduce --mlps or enlarge D07 in seed_blocks.py.")
    return BLOCK_D07 + offset


def p1(mlps, n_pairs, dev, c0_targets=(0.0, 0.3, 0.6, 0.9)):
    """Empirical c_l vs R-iterates, for several starting correlations."""
    print("=" * 100)
    print("P1 — parameter-free correlation map  c_{l+1} = R(c_l)")
    print("=" * 100)
    print(f"{n_pairs} input pairs x {mlps} fresh MLPs per starting correlation, "
          f"no fitted constants\n")

    rows = []
    for ci, c0 in enumerate(c0_targets):
        acc = np.zeros(DEPTH)
        cnt = 0
        for j in range(mlps):
            w_np, _e, _s = canonical_mint(WIDTH, DEPTH, _seed(ci * mlps + j))
            w = torch.from_numpy(np.stack(w_np)).to(dev)
            g = torch.Generator(device=dev).manual_seed(7000 + j)

            # two input batches with exact target correlation, unit norm in expectation
            a = torch.randn(n_pairs, WIDTH, generator=g, device=dev)
            b = torch.randn(n_pairs, WIDTH, generator=g, device=dev)
            b = c0 * a + math.sqrt(max(0.0, 1.0 - c0 * c0)) * b
            ha, hb = a, b
            for li in range(DEPTH):
                ha = torch.clamp(ha @ w[li], min=0.0)
                hb = torch.clamp(hb @ w[li], min=0.0)
                # uncentered correlation, averaged over pairs
                num = (ha * hb).sum(dim=1)
                da = (ha * ha).sum(dim=1).sqrt()
                db = (hb * hb).sum(dim=1).sqrt()
                acc[li] += float((num / (da * db + 1e-30)).mean())
            cnt += 1
        rows.append((c0, acc / cnt))

    print(f"{'layer':>6} " + "".join(f"{'c0=' + str(c):>22}" for c, _ in rows))
    print(f"{'':>6} " + "".join(f"{'measured':>11}{'R-iterate':>11}" for _ in rows))
    print("-" * (6 + 22 * len(rows)))
    preds = {}
    for c0, _ in rows:
        seq, c = [], c0
        for _ in range(DEPTH):
            c = R_map(c)
            seq.append(c)
        preds[c0] = seq

    worst = 0.0
    for li in list(range(0, DEPTH, 4)) + [DEPTH - 1]:
        line = f"{li + 1:>6} "
        for c0, meas in rows:
            line += f"{meas[li]:>11.5f}{preds[c0][li]:>11.5f}"
            worst = max(worst, abs(meas[li] - preds[c0][li]))
        print(line)
    print("-" * (6 + 22 * len(rows)))
    print(f"max |measured - predicted| over all layers and starts: {worst:.5f}")
    verdict = "PASS" if worst < 0.02 else ("MARGINAL" if worst < 0.05 else "DISAGREE")
    print(f"P1 verdict: {verdict}  (threshold 0.02 absolute; parameter-free, so a")
    print("            disagreement is a fact about the theory, not a tuning failure)")
    return {"p1_max_abs_error": worst, "verdict": verdict}


def p3(mlps, n_samples, dev):
    """Alignment between the fluctuation covariance's top eigenvector and mu."""
    print("\n" + "=" * 100)
    print("P3 — top fluctuation eigenvector vs the mean activation direction")
    print("=" * 100)
    print(f"{mlps} fresh MLPs x {n_samples} samples\n")

    align = np.zeros(DEPTH)
    pr = np.zeros(DEPTH)
    frac1 = np.zeros(DEPTH)
    for j in range(mlps):
        w_np, _e, _s = canonical_mint(WIDTH, DEPTH, _seed(P1_STARTS * mlps + j))
        w = torch.from_numpy(np.stack(w_np)).to(dev)
        g = torch.Generator(device=dev).manual_seed(9100 + j)
        h = torch.randn(n_samples, WIDTH, generator=g, device=dev)
        for li in range(DEPTH):
            h = torch.clamp(h @ w[li], min=0.0)
            hd = h.double()
            mu = hd.mean(dim=0)
            X = hd - mu
            cov = (X.T @ X) / (n_samples - 1)
            ev, vec = torch.linalg.eigh(cov)
            v = vec[:, -1]
            align[li] += abs(float(v @ mu) / (float(mu.norm()) + 1e-30))
            lam = torch.clamp(ev, min=0.0)
            pr[li] += float((lam.sum() ** 2) / ((lam ** 2).sum() + 1e-300))
            frac1[li] += float(lam[-1] / (lam.sum() + 1e-300))
    align /= mlps
    pr /= mlps
    frac1 /= mlps

    print(f"{'layer':>6} {'|<v1, mu_hat>|':>16} {'participation ratio':>21} "
          f"{'lambda_1 share':>16}")
    print("-" * 62)
    for li in list(range(0, DEPTH, 4)) + [DEPTH - 1]:
        print(f"{li + 1:>6} {align[li]:>16.4f} {pr[li]:>21.2f} {frac1[li]:>16.4f}")
    print("-" * 62)
    print(f"alignment layer 1 -> {DEPTH}: {align[0]:.4f} -> {align[-1]:.4f}")
    print(f"PR         layer 1 -> {DEPTH}: {pr[0]:.1f} -> {pr[-1]:.2f}")
    print("\nReference from figure_data.py (independent measurement, older mint):")
    print("  PR by layer 1/8/16/24/32 = 165.0 / 25.9 / 8.8 / 4.4 / 2.8")
    rising = align[-1] > align[0]
    print(f"\nP3 verdict: alignment {'RISES' if rising else 'DOES NOT RISE'} with depth "
          f"({align[0]:.3f} -> {align[-1]:.3f}).")
    print("  A rising alignment supports the collapse-onto-the-mean-direction reading;")
    print("  a flat one would say the collapse is real but not toward mu, which would")
    print("  be a genuine correction to the section's expected argument.")
    return {"align_first": float(align[0]), "align_last": float(align[-1]),
            "pr": pr.tolist(), "align": align.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlps", type=int, default=64)
    ap.add_argument("--pairs", type=int, default=4096)
    ap.add_argument("--samples", type=int, default=8192)
    ap.add_argument("--which", default="all", choices=["all", "p1", "p3"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    res = {}
    if a.which in ("all", "p1"):
        res.update(p1(a.mlps, a.pairs, dev))
    if a.which in ("all", "p3"):
        res.update(p3(a.mlps, a.samples, dev))

    print("\n" + "=" * 100)
    print("P2 (two-sector participation-ratio fit, held out at layers 12/20/28): "
          "NOT IMPLEMENTED")
    print("=" * 100)
    print("  Its functional form lives in docs/nngp-theory-draft.md, which is not in")
    print("  the repo. P1 and P3 are parameter-free and could be run without it; P2")
    print("  cannot, and guessing a two-sector form then reporting it as a")
    print("  verification would invert the point of the exercise.")

    if a.out:
        import json
        json.dump(res, open(a.out, "w"), indent=2)
        print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
