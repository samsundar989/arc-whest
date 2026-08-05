"""
Track P, stage 1 verdict: price the rank-k scheme against the measured damage.

Reads `track_p_spectrum.py`'s output and turns the two measured curves (energy /
MSE versus rank) into the only number that matters, V*c.

The cost model, from flopscope's shape-based billing (RESEARCH_NOTES 3d):

  full layer        (N,256)@(256,256)          = (2*256-1)*N*256 = 130,816 N
  rank-k layer      (N,k)@(k,256)              = (2k-1)*N*256          )
                  + (N,256)@(256,k)  re-project = (2*256-1)*N*k        ) ~ 1023 k N
                  + ReLU / Jensen elementwise on N*256

Propagating the deviation in a k-dimensional subspace IS a rank-k factorisation
of each weight matrix, and a 256x256 matmul only gets cheaper as a product of two
factors when k < 128. That break-even is a property of the pricing, not of our
implementation.

Run: uv run python local_sim/track_p_cost.py <tp_mini.npz>
"""

import sys

import numpy as np

WIDTH = 256
DEPTH = 32
FULL = (2 * WIDTH - 1) * WIDTH          # per sample per layer
RELU = 0.5 * WIDTH
# stats.norm.cdf 96/elt and pdf 54/elt were measured for f64; halve for f32 and
# add ~5 arithmetic ops per element for the z*Phi + sd*phi combination.
JENSEN = (96 / 2 + 54 / 2 + 5) * WIDTH
PPF_AND_SOBOL = 171 * WIDTH + 30 * 3 * WIDTH


def layer_cost(k, jensen):
    if k >= WIDTH:
        return FULL + RELU
    return (2 * k - 1) * WIDTH + (2 * WIDTH - 1) * k + (JENSEN if jensen else RELU)


def main():
    path = sys.argv[1]
    d = np.load(path)
    layers = list(d["layers"])          # 0-based projection points
    ks = list(d["ks"])
    base = float(d["base_mse"].mean())
    prj = d["prj_mse"].mean(axis=0)
    jen = d["jen_mse"].mean(axis=0)

    print("break-even rank for the factorisation alone (ignoring the nonlinearity):")
    for k in (32, 64, 96, 128, 160):
        print(f"   k={k:>4}  rank-k layer = {(2 * k - 1) * WIDTH + (2 * WIDTH - 1) * k:>9,.0f}  "
              f"vs full {FULL:>9,.0f}  -> {FULL / ((2 * k - 1) * WIDTH + (2 * WIDTH - 1) * k):.3f}x")
    print(f"   the two factors cost the same as the dense matmul at k = {WIDTH // 2}\n")

    for tag, mse_tab, jflag in (("plain ReLU (no Jensen correction)", prj, False),
                                ("ORACLE Jensen correction", jen, True),
                                ("ORACLE Jensen correction, priced as FREE "
                                 "(bounds any cheaper approximation)", jen, False)):
        print("=" * 84)
        print(f"END-TO-END  V*c  RATIO vs full-width sampling  —  {tag}")
        print("=" * 84)
        print(f"{'project from':>13} " + " ".join(f"{'k=' + str(k):>21}" for k in ks))
        print(f"{'':>13} " + " ".join(f"{'cost x   MSE x   V*c x':>21}" for k in ks))
        print("-" * 84)
        for a_i, l0 in enumerate(layers):
            cells = []
            for k_i, k in enumerate(ks):
                m = float(mse_tab[a_i, k_i])
                if m == 0:
                    cells.append(f"{'--':>21}")
                    continue
                c_full = DEPTH * (FULL + RELU) + PPF_AND_SOBOL
                n_red = DEPTH - (l0 + 1)
                c_red = ((l0 + 1) * (FULL + RELU) + n_red * layer_cost(k, jflag) + PPF_AND_SOBOL)
                cost_x = c_full / c_red
                mse_x = base / m
                cells.append(f"{cost_x:>6.3f}x {mse_x:>6.3f}x {cost_x * mse_x:>6.3f}x")
            print(f"{'layer ' + str(l0 + 1):>13} " + " ".join(f"{c:>21}" for c in cells))
        print()

    best = 0.0
    for a_i, l0 in enumerate(layers):
        for k_i, k in enumerate(ks):
            for tab, jf in ((prj, False), (jen, True), (jen, False)):
                m = float(tab[a_i, k_i])
                if not m:
                    continue
                n_red = DEPTH - (l0 + 1)
                c_full = DEPTH * (FULL + RELU) + PPF_AND_SOBOL
                c_red = (l0 + 1) * (FULL + RELU) + n_red * layer_cost(k, jf) + PPF_AND_SOBOL
                best = max(best, (c_full / c_red) * (base / m))
    print(f"BEST V*c over the whole grid, using the ORACLE correction: {best:.3f}x")
    print(f"GATE (>=1.25x end-to-end): {'PASS' if best >= 1.25 else 'FAIL'}")


if __name__ == "__main__":
    main()
