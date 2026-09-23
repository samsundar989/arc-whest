"""Summarise a `tools/n1_probe.py` result: the two spectra and their rank agreement.

    python tools/n1_spectra.py experiments/2026-09-25_n1/net0_r512.json
"""
import json
import sys

import numpy as np


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main(path):
    d = json.load(open(path))
    s = np.array(d["sens"])
    n = len(s)
    e = np.array(d["gram_energy"])[:n]
    c = np.array(d["coef_energy"])[:n]
    dm = np.array(d["dmse"])
    print(f"net {d['net']} first join L{d['first_join_layer']} rank {d['rank']} "
          f"dirs {n} ref MSE {d['mse_ref']:.4e}")
    print(f"Spearman(gram energy, sensitivity) = {spearman(e, s):.3f}")
    print(f"Spearman(coef energy, sensitivity) = {spearman(c, s):.3f}")
    ss, es = np.sort(s)[::-1], np.sort(e)[::-1]
    print("| rank | energy (sorted) | sensitivity (sorted) | sens. of energy-rank dir |")
    print("|---:|---:|---:|---:|")
    for k in (1, 16, 64, 128, 256, 384, 512):
        if k <= n:
            print(f"| {k} | {es[k-1]:.3e} | {ss[k-1]:.3e} | {s[k-1]:.3e} |")
    tot = s.sum()
    order_s = np.argsort(s)[::-1]
    for K in (128, 192, 256, 384):
        if K <= n:
            print(f"top-{K}: sens captured by energy-order {s[:K].sum()/tot:.4f}, "
                  f"by sens-order {s[order_s[:K]].sum()/tot:.4f}; "
                  f"overlap {len(set(range(K)) & set(order_s[:K].tolist()))}/{K}")
    # dMSE: dropping a direction can HELP (negative) -- report how many
    print(f"directions whose removal lowers MSE: {(dm < 0).sum()}/{len(dm)}; "
          f"sum dMSE top-256-energy-dropped-set (256..511) {dm[256:].sum():+.3e}, "
          f"ref MSE {d['mse_ref']:.3e}")


if __name__ == "__main__":
    main(sys.argv[1])
