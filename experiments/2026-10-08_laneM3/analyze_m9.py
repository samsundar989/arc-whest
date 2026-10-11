"""M9/M10 tables from ARC's kprop_by_width outputs (`.venv-kprop`).

    python analyze_m9.py <data_dir>      data_dir = .../relu_w2_m9 (one sub-dir per run)

Per (run, n, trial): de-biased final-layer MSE = err(act15, d=1) − se2(act15). Ratios between
runs are paired by (n, trial) (same net, same labels); 95 % t-interval on the log ratio.
Slopes: least squares of log(mean MSE) on log n, over all widths and over n >= 128.
Extrapolation to n = 1024 uses each run's fitted line.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
_ARC = Path("C:/Users/samsu/Documents/arc-whest/external/mlp_cumulant_propagation")
sys.path += [str(_ARC / "scripts"), str(_ARC / "src")]   # the pickled cfg is a kprop_by_width class
T95 = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}
LAYER = "act15"


def load(d: Path):
    runs = {}
    for f in sorted(d.glob("*/kprop_by_width.pt")):
        _cfg, e, s = torch.load(f, weights_only=False)
        e, s = e[(e.l == LAYER) & (e.d == "1")], s[(s.l == LAYER) & (s.d == "1")]
        m = e.merge(s, on=["n", "t", "l", "d"])
        runs[f.parent.name] = {(int(r.n), int(r.t)): float(r.err - r.se2) for r in m.itertuples()}
        runs[f.parent.name + "__se2"] = {(int(r.n), int(r.t)): float(r.se2) for r in m.itertuples()}
    return runs


def gm_ci(x):
    x = np.log(np.asarray(x))
    k = len(x)
    h = T95.get(k, 2.0) * x.std(ddof=1) / math.sqrt(k)
    return math.exp(x.mean()), math.exp(x.mean() - h), math.exp(x.mean() + h)


def fit(ns, ys):
    b, a = np.polyfit(np.log(ns), np.log(ys), 1)
    return b, a


def main():
    runs = load(Path(sys.argv[1]))
    names = [k for k in runs if not k.endswith("__se2")]
    base = "k3aug_dense"
    out, L = {}, []
    L.append(f"Final-layer ({LAYER}) MSE, de-biased (err − se2), mean over trials; finite trials only\n")
    widths = sorted({n for k in names for (n, _t) in runs[k]})
    L.append("| run | " + " | ".join(f"n={n}" for n in widths) + " | slope (all) | slope (n≥128) | extrap. n=1024 |")
    L.append("|---" * (len(widths) + 4) + "|")
    fits = {}
    for k in names:
        means = {}
        for n in widths:
            v = [x for (nn, _t), x in runs[k].items() if nn == n and np.isfinite(x)]
            means[n] = (float(np.mean(v)), len(v)) if v else (float("nan"), 0)
        ok = [n for n in widths if means[n][1] and means[n][0] > 0 and means[n][0] < 1]
        row = {"means": {n: means[n][0] for n in widths}}
        cells = [f"{means[n][0]:.3e}" + (f" ({means[n][1]})" if means[n][1] not in (0, 8) else "") for n in widths]
        sl = sl2 = ex = float("nan")
        if len(ok) >= 2:
            sl, a = fit(ok, [means[n][0] for n in ok])
            ex = math.exp(a + sl * math.log(1024))
            big = [n for n in ok if n >= 128]
            if len(big) >= 2:
                sl2, a2 = fit(big, [means[n][0] for n in big])
                ex = math.exp(a2 + sl2 * math.log(1024))
        row.update(slope_all=sl, slope_big=sl2, extrap_1024=ex)
        fits[k] = row
        L.append(f"| {k} | " + " | ".join(cells) + f" | {sl:.2f} | {sl2:.2f} | {ex:.2e} |")
    out["fits"] = fits
    L.append(f"\nPaired ratio to {base} (geometric mean over trials, 95 %; wins = trials better than {base})\n")
    L.append("| run | " + " | ".join(f"n={n}" for n in widths) + " |")
    L.append("|---" * (len(widths) + 1) + "|")
    out["ratios"] = {}
    for k in names:
        if k == base or base not in runs:
            continue
        cells, rr = [], {}
        for n in widths:
            keys = [(nn, t) for (nn, t) in runs[k] if nn == n and (nn, t) in runs[base]]
            r = [runs[k][q] / runs[base][q] for q in keys
                 if np.isfinite(runs[k][q]) and runs[k][q] > 0 and runs[base][q] > 0]
            if len(r) >= 2:
                g, lo, hi = gm_ci(r)
                wins = sum(x < 1 for x in r)
                cells.append(f"{g:.3f} [{lo:.3f}, {hi:.3f}] {wins}/{len(r)}")
                rr[n] = (g, lo, hi, wins, len(r))
            else:
                cells.append("—")
        out["ratios"][k] = rr
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    se = runs.get(base + "__se2", {})
    if se:
        L.append("\nLabel noise se2 at act15 (mean): " + ", ".join(
            f"n={n}: {np.mean([v for (nn, _), v in se.items() if nn == n]):.1e}" for n in widths))
    txt = "\n".join(L) + "\n"
    print(txt)
    tag = Path(sys.argv[1]).name
    (HERE / f"m9_table_{tag}.md").write_text(txt, encoding="utf-8")
    (HERE / f"m9_summary_{tag}.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
