"""M5 table: merge the m5_mini1024*.json parts and the SIMPLE (a) column from m2_mini1024.json.

    uv run python experiments/2026-10-07_laneM2/analyze_m5.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M2 = HERE.parent / "2026-10-06_laneM" / "m2_mini1024.json"


def tci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan
    from math import sqrt
    t = {3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201,
         12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120}.get(len(x), 2.0)
    return t * x.std(ddof=1) / sqrt(len(x))


def main() -> int:
    nets: dict[int, dict] = {}
    for p in sorted(HERE.glob("m5_mini1024*.json")):
        for r in json.loads(p.read_text())["nets"]:
            d = nets.setdefault(r["i"], {"v73": r["v73_mse_final"]})
            d.update({k: v for k, v in r.items() if isinstance(v, dict)})
    for r in json.loads(M2.read_text())["nets"]:
        if r["i"] in nets:
            nets[r["i"]]["a"] = {"mse_final": r["variants"]["a"]["mse_final"], "mse_all": r["variants"]["a"]["mse_all"]}
    arms = [k for k in ("a", "b", "1_simple_legs", "2_dressed", "2f_dressed_s0.3", "2d_dressed_s0.5",
                        "2e_dressed_s0.7", "2c_asym", "3a_memoryless", "3b_gain", "3c_memoryless+gain",
                        "4_no211", "5_noK211", "45_no_k3k4_loop", "145_simple_legs_no_loop",
                        "3d_gain_from6", "3e_memoryless+gain_from6", "3f_memoryless_from6")]
    rows = ["| arm | nets | final MSE (mean) | Δ vs b (paired, 95 %) | loses (arm−b)/(a−b) | vs v73 (paired) | wins vs b | all-layer MSE |",
            "|---|---|---|---|---|---|---|---|"]
    summ = {}
    for k in arms:
        ids = sorted(i for i, d in nets.items() if k in d and "b" in d and "a" in d)
        if not ids:
            continue
        x = np.array([nets[i][k]["mse_final"] for i in ids])
        b = np.array([nets[i]["b"]["mse_final"] for i in ids])
        a = np.array([nets[i]["a"]["mse_final"] for i in ids])
        v = np.array([nets[i]["v73"] for i in ids])
        xa = np.array([nets[i][k]["mse_all"] for i in ids])
        dr = (x - b) / b
        lose = (x.mean() - b.mean()) / (a.mean() - b.mean())
        vr = (x - v) / v
        summ[k] = {"n": len(ids), "mse": x.mean(), "d_vs_b": dr.mean(), "loses": lose, "vs_v73": vr.mean()}
        rows.append(f"| {k} | {len(ids)} | {x.mean():.3e} | {dr.mean() * 100:+.1f} % ± {tci(dr) * 100:.1f} | "
                    f"{lose * 100:.0f} % | {vr.mean() * 100:+.1f} % ± {tci(vr) * 100:.1f} | {(x < b).sum()}/{len(ids)} | "
                    f"{xa.mean():.3e} |")
    txt = "\n".join(rows)
    print(txt)
    (HERE / "m5_table.md").write_text(txt + "\n", encoding="utf-8")
    (HERE / "m5_summary.json").write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8")
    # per-layer growth of the blow-ups (mean over nets): ratio to b at layers 1, 4, 8, 12, 16
    print("\nper-layer MSE ratio to b (mean over nets) at layers 1/4/8/12/16:")
    for k in arms:
        if k in ("a", "b"):
            continue
        ids = sorted(i for i, d in nets.items() if k in d and "b" in d)
        if not ids or "mse_layers" not in nets[ids[0]][k]:
            continue
        r = np.mean([np.array(nets[i][k]["mse_layers"]) / np.array(nets[i]["b"]["mse_layers"]) for i in ids], 0)
        print(f"  {k:22s} " + " ".join(f"{r[l - 1]:8.2f}" for l in (1, 4, 8, 12, 16)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
