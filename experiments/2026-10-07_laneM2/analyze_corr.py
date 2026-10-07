"""Top-hub H⊗H correction table (Lane P's question): captured spurious-energy share per layer and
final MSE vs b / sym dressing, paired per net. Reads every m5_mini1024*.json.

    uv run python experiments/2026-10-07_laneM2/analyze_corr.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> int:
    nets: dict[int, dict] = {}
    for p in sorted(HERE.glob("m5_mini1024*.json")):
        for r in json.loads(p.read_text())["nets"]:
            nets.setdefault(r["i"], {}).update({k: v for k, v in r.items() if isinstance(v, dict)})
    arms = [("2_dressed", 0.0)] + [(k, f) for k, f in (("2g_dressed_corr0.05", .05), ("2h_dressed_corr0.15", .15),
                                                      ("2i_dressed_corr0.4", .4))]
    ids = sorted(i for i, d in nets.items() if all(k in d for k, _ in arms) and "b" in d)
    b = np.array([nets[i]["b"]["mse_final"] for i in ids])
    sym = np.array([nets[i]["2_dressed"]["mse_final"] for i in ids])
    lines = [f"Top-hub H⊗H correction ({len(ids)} DEV nets, n = 1024, ARC chain, no learned gains)", "",
             "| f (hubs corrected) | final MSE | ratio to b (mean of per-net) | share of sym−b gap closed | "
             "captured spurious energy, layers 2/4/8/12/16 | per-layer MSE ratio to b, layers 4/8/12/16 |",
             "|---|---|---|---|---|---|"]
    for k, f in arms + [("b", 1.0)]:
        x = np.array([nets[i][k]["mse_final"] for i in ids])
        closed = (sym.mean() - x.mean()) / (sym.mean() - b.mean())
        sh = ""
        if "corr_energy_share" in nets[ids[0]].get(k, {}):
            s = np.mean([nets[i][k]["corr_energy_share"] for i in ids], 0)   # per nonlinear step t (layer t+1)
            sh = " / ".join(f"{s[l - 1]:.2f}" for l in (2, 4, 8, 12, 16))
        pl = np.mean([np.array(nets[i][k]["mse_layers"]) / np.array(nets[i]["b"]["mse_layers"]) for i in ids], 0)
        lines.append(f"| {f:.2f} | {x.mean():.3e} | {np.mean(x / b):.3f} | {closed * 100:.0f} % | {sh} | "
                     + " / ".join(f"{pl[l - 1]:.2f}" for l in (4, 8, 12, 16)) + " |")
    txt = "\n".join(lines)
    print(txt)
    (HERE / "m5_corr_table.md").write_text(txt + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
