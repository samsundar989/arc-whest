"""E2 retention screen: frozen v72 vs age3 history arms on TRAIN full rows 0..7.

    .venv-gpu/Scripts/python.exe experiments/2026-09-28_compact_history/retention/probe.py

Records, per arm and net, the final-layer label MSE (reference floor ~7.7e-11) and the
teacher distortion MSE(arm final mean, v72 final mean). Accuracy-only twin forward:
no counted cost is read here. TRAIN rows only; no DEV/VAL labels are touched.
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tools" / "twin"))
import twin  # noqa: E402  (installs the torch flopscope shim for this process)
import numpy as np  # noqa: E402
import torch  # noqa: E402

_names = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["age3_r64", "age3_r32", "age3_zero"])
ARMS = {"v72": ROOT / "estimator_k3_v72_gt.py", **{a: HERE / f"arm_{a}.py" for a in _names}}
NETS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2,3,4,5,6,7").split(",")]
TAG = sys.argv[3] if len(sys.argv) > 3 else ""

data = twin.Data()
finals, out = {}, {"nets": NETS, "arms": {}}
for name, path in ARMS.items():
    tw = twin.Twin(path, raw=True)
    rows, fin = [], []
    for i in NETS:
        w, truth = data.net("full", i)
        t0 = time.time()
        with torch.no_grad():
            mus = tw.forward(w)
        m = [float(x) for x in twin.mse_rows(mus, truth)]
        rows.append(m)
        fin.append(mus[-1].double().cpu().numpy())
        print(json.dumps({"arm": name, "net": i, "final_mse": m[-1], "s": round(time.time() - t0, 1)}),
              flush=True)
    finals[name] = np.stack(fin)
    out["arms"][name] = {"row_mse": rows, "final_mse": [r[-1] for r in rows]}
    del tw
    torch.cuda.empty_cache()

base = out["arms"]["v72"]["final_mse"]
for name in ARMS:
    a = out["arms"][name]
    a["teacher_dist"] = [float(((finals[name][k] - finals["v72"][k]) ** 2).mean()) for k in range(len(NETS))]
    a["mean_final_mse"] = float(np.mean(a["final_mse"]))
    a["ratio_vs_v72"] = float(np.mean(a["final_mse"]) / np.mean(base))
    a["wins_vs_v72"] = int(sum(x < b for x, b in zip(a["final_mse"], base)))
for name in [a for a in ("age3_r64", "age3_r32") if a in ARMS and "age3_zero" in ARMS]:
    z = out["arms"]["age3_zero"]["teacher_dist"]
    a = out["arms"][name]
    a["teacher_wins_vs_zero"] = int(sum(x < y for x, y in zip(a["teacher_dist"], z)))
np.savez_compressed(HERE / f"finals{TAG}.npz", **finals)
(HERE / f"retention{TAG}.json").write_text(json.dumps(out, indent=1))
for name in ARMS:
    a = out["arms"][name]
    print(name, "mean final", f"{a['mean_final_mse']:.4e}", "x v72", f"{a['ratio_vs_v72']:.3f}",
          "teacher", f"{np.mean(a['teacher_dist']):.3e}", a.get("teacher_wins_vs_zero", ""))
