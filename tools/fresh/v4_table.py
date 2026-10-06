"""V4 calibration table: v73_gb and v68_ws on the fresh panels (from ev.py's caches).

Per panel: floor-free score (mean (MSE - floor) x F/2^41), raw MSE with and without the
floor, per-net score CV, the paired v73 - v68 difference (raw, studentised t95), and the
ratios to each candidate's DEV projection and graded public score.

    python tools/fresh/v4_table.py [--panels fresh_d,fresh_a]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.append(str(HERE))
sys.path.append(str(ROOT / "tools"))
import gen  # noqa: E402

BUDGET = 2 ** 41
#: (file, F/2^41, DEV projected public, graded public) -- RESULTS.md rows.
CANDS = {
    "v73_gb": ("estimator_k3_v73_gb.py", 0.21473, 4.6090e-9, 4.6198973531323855e-9),
    "v68_ws": ("estimator_k3_v68_ws.py", 0.23127, 5.08e-9, 5.0664e-9),
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(panel: str, cand: str):
    f = ROOT / CANDS[cand][0]
    p = ROOT / "cache" / "preds" / sha(f) / panel / "n100.npz"
    if not p.exists():
        return None
    with np.load(p) as d:
        mse = ((d["preds"].astype(np.float32) - d["truth"].astype(np.float32)) ** 2).mean(
            1, dtype=np.float32).astype(np.float64)
    return mse


def t95(x):
    x = np.asarray(x)
    return 1.984 * x.std(ddof=1) / math.sqrt(len(x))     # t(99)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="fresh_d,fresh_a")
    a = ap.parse_args(argv)
    out, lines = {}, []
    for panel in a.panels.split(","):
        floors = np.array([gen.load_label(panel, i)["floor"][-1] for i in range(100)])
        rows = {}
        for c in CANDS:
            mse = load(panel, c)
            if mse is None:
                continue
            fb = CANDS[c][1]
            sc = (mse - floors) * fb
            rows[c] = {"raw": float(mse.mean()), "raw_floor_free": float((mse - floors).mean()),
                       "floor": float(floors.mean()), "score": float(sc.mean()),
                       "score_se": float(sc.std(ddof=1) / 10.0),
                       "cv": float(sc.std(ddof=1) / sc.mean()),
                       "vs_dev_projection": float(sc.mean() / CANDS[c][2]),
                       "vs_graded": float(sc.mean() / CANDS[c][3]), "_sc": sc, "_mse": mse}
        pr = {}
        if len(rows) == 2:
            d_sc = rows["v73_gb"]["_sc"] - rows["v68_ws"]["_sc"]
            d_raw = rows["v73_gb"]["_mse"] - rows["v68_ws"]["_mse"]
            b_sc, b_raw = rows["v68_ws"]["score"], rows["v68_ws"]["raw_floor_free"]
            graded_rel = CANDS["v73_gb"][3] / CANDS["v68_ws"][3] - 1
            pr = {"score_rel": float(d_sc.mean() / b_sc), "score_rel_t95": float(t95(d_sc) / b_sc),
                  "raw_rel": float(d_raw.mean() / b_raw), "raw_rel_t95": float(t95(d_raw) / b_raw),
                  "v73_wins": int((d_sc < 0).sum()), "graded_rel": float(graded_rel)}
        out[panel] = {"cands": {c: {k: v for k, v in r.items() if not k.startswith("_")}
                                for c, r in rows.items()}, "paired_v73_minus_v68": pr}
        lines.append(f"\n### {panel.upper()} (100 nets; mean label floor {floors.mean():.3e} subtracted)\n")
        lines.append("| candidate | score (floor-free) | ± se | raw MSE | raw floor-free | per-net CV | ÷ DEV projection | ÷ graded |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for c, r in rows.items():
            lines.append(f"| {c} | {r['score']:.4e} | {r['score_se']:.2e} | {r['raw']:.4e} | "
                         f"{r['raw_floor_free']:.4e} | {r['cv']*100:.1f} % | {r['vs_dev_projection']:.4f} | "
                         f"{r['vs_graded']:.4f} |")
        if pr:
            lines.append(f"\npaired v73 − v68 (raw arithmetic, t95): score {pr['score_rel']*100:+.2f} % "
                         f"± {pr['score_rel_t95']*100:.2f} pp, raw {pr['raw_rel']*100:+.2f} % ± "
                         f"{pr['raw_rel_t95']*100:.2f} pp, v73 wins {pr['v73_wins']}/100; graded "
                         f"{pr['graded_rel']*100:+.2f} %")
    txt = "\n".join(lines)
    print(txt)
    op = ROOT / "experiments" / "2026-10-06_laneV" / "v4_table.json"
    op.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    (op.with_suffix(".md")).write_text("# V4 calibration table (tools/fresh/v4_table.py)\n" + txt + "\n",
                                       encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
