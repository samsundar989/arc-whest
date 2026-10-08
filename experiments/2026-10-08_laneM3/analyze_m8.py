"""M8 tables: final MSE and per-layer ratio to the undropped chain (A = ∞), per chain and A.

    python experiments/2026-10-08_laneM3/analyze_m8.py m8_all.json [...]  -> m8_table.md, m8_summary.json
Ratios: geometric mean over nets of MSE_A / MSE_inf (paired), 95 % t-interval on the log ratio.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T95 = {n: t for n, t in zip(range(2, 31), [12.71, 4.30, 3.18, 2.78, 2.57, 2.45, 2.36, 2.31, 2.26, 2.23, 2.20, 2.18,
                                             2.16, 2.14, 2.13, 2.12, 2.11, 2.10, 2.09, 2.09, 2.08, 2.07, 2.07, 2.06,
                                             2.06, 2.06, 2.05, 2.05, 2.05])}


def gm_ci(r):
    r = np.log(np.asarray(r))
    m, k = r.mean(), len(r)
    h = T95[k] * r.std(ddof=1) / math.sqrt(k) if k > 1 else float("nan")
    return math.exp(m), math.exp(m - h), math.exp(m + h)


def main():
    nets = []
    for f in sys.argv[1:]:
        nets += json.loads(Path(f).read_text())["nets"]
    nets = sorted({r["i"]: r for r in nets}.values(), key=lambda r: r["i"])
    arms = [k for k in nets[0] if "_A" in k]
    chains = sorted({k.split("_A")[0] for k in arms}, reverse=True)
    out, lines = {"n_nets": len(nets), "rows": []}, []
    lines.append(f"M8 ({len(nets)} DEV nets, n = 1024, organizer labels, floor-subtracted)\n")
    lines.append("| chain | A | final MSE (mean) | ratio to A = ∞ (gm, 95 %) | wins (A worse) | "
                 "per-layer ratio L4 / L8 / L12 / L16 | final cols |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in chains:
        ref = np.array([r[f"{c}_Ainf"]["mse_final"] for r in nets])
        refL = np.array([r[f"{c}_Ainf"]["mse_layers"] for r in nets])
        for k in [a for a in arms if a.startswith(c + "_A")]:
            v = np.array([r[k]["mse_final"] for r in nets])
            L = np.array([r[k]["mse_layers"] for r in nets])
            g, lo, hi = gm_ci(v / ref)
            pl = [L[:, l].mean() / refL[:, l].mean() for l in (3, 7, 11, 15)]
            row = dict(chain=c, A=k.split("_A")[1], mse=float(v.mean()), ratio=g, lo=lo, hi=hi,
                       worse=int((v > ref).sum()), per_layer=pl, cols=int(nets[0][k]["cols_final"]))
            out["rows"].append(row)
            lines.append(f"| {c} | {row['A']} | {v.mean():.3e} | ×{g:.2f} [{lo:.2f}, {hi:.2f}] | "
                         f"{row['worse']}/{len(nets)} | " + " / ".join(f"{x:.2f}" for x in pl) + f" | {row['cols']} |")
    if "bE_Ainf" in nets[0] and "a_Ainf" in nets[0]:
        lines.append("\nb+E vs a at equal A (ratio of means a / b+E):\n")
        for A in sorted({k.split("_A")[1] for k in arms}, key=lambda x: (x == "inf", int(x) if x != "inf" else 0)):
            a = np.mean([r[f"a_A{A}"]["mse_final"] for r in nets])
            b = np.mean([r[f"bE_A{A}"]["mse_final"] for r in nets])
            lines.append(f"- A = {A}: a {a:.3e}, b+E {b:.3e}, a / b+E = {a / b:.2f}")
    txt = "\n".join(lines) + "\n"
    print(txt)
    (HERE / "m8_table.md").write_text(txt, encoding="utf-8")
    (HERE / "m8_summary.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
