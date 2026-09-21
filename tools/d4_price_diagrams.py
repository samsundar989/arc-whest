"""D4 — price the diagrams factored AUGMENT drops.

PLAN §6 D4. ARC's own `TODO.md` states that factored (factorized) cumulant
propagation with `kind=AUGMENT` is "the augmented term set minus the terms it
cannot afford (`kprop_harmonic.factored_keeps_term`: non-hypertree top-slice
diagrams, e.g. covariance cycles, and products of the factored top-degree
cumulant's all-distinct block with other blocks)... the dropped terms have the
same Theta(n^-k_max) squared size as the extra augmented terms that are kept."

Our shipped chain is a factored K3. If those dropped diagrams are worth a lot,
they are a principled accuracy source nobody in the field has published; if they
are worth little, the lane closes in one line and we stop wondering.

THE QUESTION, and the decision rule PLAN fixes in advance:

    unfactored AUGMENT MSE / factored AUGMENT MSE

    >= 1.3x at two shapes  -> schedule D4b (derive the dropped terms from
                              get_all_terms_iso minus factored_keeps_term,
                              implement with Sigma_off ~ U U^T at r = 64,
                              fitted gains via D2, budget 5 days, after D5)
    <  1.15x               -> close the lane in one line
    in between             -> report and let Samuel decide

Shapes are matched at **L/n = 1/64**, the competition's own ratio (1024 wide,
16 deep), because the truncation error the factorisation controls is a
depth-accumulated Theta(n^-K) effect and comparing it at the wrong aspect ratio
measures a different regime.

    python tools/d4_price_diagrams.py --shapes 128x2 256x4 --trials 8

Run it with `.venv-kprop` and `PYTHONPATH=external/mlp_cumulant_propagation/src`.
The MSE of every arm is quoted against the Monte-Carlo noise floor of the truth
it is measured against; a ratio between two arms that are both inside the noise
floor means nothing, and the report says so rather than printing it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, SIMPLE, AUGMENT

#: (kind, factor, k_max) -> the four arms PLAN names.
ARMS = [
    ("factored SIMPLE K3", SIMPLE, True, 3),
    ("factored AUGMENT K3", AUGMENT, True, 3),
    ("unfactored AUGMENT K3", AUGMENT, False, 3),
    ("factored K4", SIMPLE, True, 4),
]


def mc_truth(mlp: MLP, n: int, samples: int, batch: int, device, dtype):
    """E[output pre-activation] by sampling, with a bootstrap standard error.

    Returns (mean, se) where `se` is the standard error of each component's
    mean. The squared se, averaged over components, IS the noise floor of every
    MSE measured against this truth: an arm whose MSE is at or below it has not
    been measured, it has been guessed.
    """
    total = torch.zeros(mlp.output_dim, device=device, dtype=torch.float64)
    total_sq = torch.zeros(mlp.output_dim, device=device, dtype=torch.float64)
    done = 0
    while done < samples:
        b = min(batch, samples - done)
        x = torch.randn(b, n, device=device, dtype=dtype)
        y = mlp(x).out.to(torch.float64)
        total += y.sum(dim=0)
        total_sq += (y * y).sum(dim=0)
        done += b
    mean = total / samples
    var = (total_sq / samples - mean * mean).clamp_min(0.0)
    se = (var / samples).sqrt()
    return mean, se


def run_shape(n: int, L: int, trials: int, samples: int, batch: int,
              device, dtype) -> dict:
    rows = []
    for t in range(trials):
        torch.manual_seed(1000 * n + 17 * L + t)
        mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=L).to(device)
        if dtype == torch.float64:
            mlp = mlp.double()
        truth, se = mc_truth(mlp, n, samples, batch, device, dtype)
        noise_floor = float((se ** 2).mean())

        K_in = {1: torch.zeros(n, device=device, dtype=dtype),
                2: torch.eye(n, device=device, dtype=dtype)}
        row = {"trial": t, "noise_floor": noise_floor,
               "truth_rms": float(truth.pow(2).mean().sqrt())}
        for name, kind, factor, k_max in ARMS:
            t0 = time.perf_counter()
            try:
                K = mlp_kprop(mlp, K_in, k_max=k_max, kind=kind, factor=factor)
                est = K[1].to_tensor().to(torch.float64)
                mse = float((est - truth).pow(2).mean())
                row[name] = {"mse": mse, "seconds": round(time.perf_counter() - t0, 2)}
            except Exception as exc:
                row[name] = {"mse": None, "error": f"{type(exc).__name__}: {exc}"[:300]}
        rows.append(row)
        print(f"  n={n} L={L} trial {t + 1}/{trials}  floor {noise_floor:.3e}  "
              + "  ".join(
                  f"{nm.split()[0][:5]}:{(row[nm].get('mse') or float('nan')):.3e}"
                  for nm, *_ in ARMS), flush=True)
    return {"n": n, "L": L, "L_over_n": L / n, "trials": trials,
            "mc_samples": samples, "rows": rows}


def summarise(shape: dict) -> dict:
    """Per-arm mean MSE and the ratio the decision rule is written on."""
    rows = shape["rows"]
    out = {"n": shape["n"], "L": shape["L"], "arms": {}}
    floor = sum(r["noise_floor"] for r in rows) / len(rows)
    out["noise_floor"] = floor
    for name, *_ in ARMS:
        vals = [r[name]["mse"] for r in rows if r[name].get("mse") is not None]
        out["arms"][name] = {
            "mean_mse": (sum(vals) / len(vals)) if vals else None,
            "n_ok": len(vals),
            "above_noise_floor": bool(vals) and (sum(vals) / len(vals)) > 10 * floor,
        }
    fa = out["arms"]["factored AUGMENT K3"]["mean_mse"]
    ua = out["arms"]["unfactored AUGMENT K3"]["mean_mse"]
    out["unfactored_over_factored"] = (ua / fa) if (fa and ua) else None
    # The ratio the lane turns on is factored/unfactored > 1 when the dropped
    # diagrams HELP, so state it the way the decision rule reads.
    out["factored_over_unfactored"] = (fa / ua) if (fa and ua) else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shapes", nargs="+", default=["128x2", "256x4"],
                    help="NxL pairs; PLAN asks for 128x2, 256x4 and 512x8 if it fits")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--samples", type=int, default=4_000_000)
    ap.add_argument("--batch", type=int, default=32768)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--float64", action="store_true",
                    help="run truth and arms in float64 (slower, removes f32 noise)")
    ap.add_argument("--json", default="experiments/2026-09-22_d4/d4_diagrams.json")
    a = ap.parse_args(argv)

    torch.set_grad_enabled(False)
    device = torch.device(a.device)
    dtype = torch.float64 if a.float64 else torch.float32
    print(f"[d4] device={device} dtype={dtype} trials={a.trials} "
          f"mc_samples={a.samples:,}")

    shapes = []
    for s in a.shapes:
        n, L = (int(v) for v in s.lower().split("x"))
        print(f"[d4] shape n={n} L={L}  (L/n = 1/{n // L})")
        shapes.append(run_shape(n, L, a.trials, a.samples, a.batch, device, dtype))

    summaries = [summarise(s) for s in shapes]
    print()
    print("  arm mean MSE by shape (noise floor in brackets)")
    for s in summaries:
        print(f"    n={s['n']} L={s['L']}  [floor {s['noise_floor']:.3e}]")
        for name, *_ in ARMS:
            v = s["arms"][name]
            flag = "" if v["above_noise_floor"] else "   <-- AT OR BELOW NOISE FLOOR"
            mm = v["mean_mse"]
            print(f"      {name:<24} {('%.4e' % mm) if mm else 'FAILED':>12}{flag}")
        r = s["unfactored_over_factored"]
        print(f"      unfactored/factored AUGMENT = "
              f"{('%.3f' % r) if r else 'n/a'}x")

    # A ratio between two arms that both sit at the Monte-Carlo noise floor is
    # a ratio of two measurements of the noise, and it will read ~1.000x -- which
    # looks exactly like "the dropped diagrams do not matter". Only shapes where
    # BOTH AUGMENT arms are resolved above the floor may vote. The smoke case
    # (n=64, L=1) is precisely this trap: one linear layer has no nonlinear step
    # to truncate, every arm is identical, and an unguarded rule closes the lane
    # on no evidence.
    resolved = [s for s in summaries
                if s["arms"]["factored AUGMENT K3"]["above_noise_floor"]
                and s["arms"]["unfactored AUGMENT K3"]["above_noise_floor"]]
    unresolved = [s for s in summaries if s not in resolved]
    if unresolved:
        print("  NOT RESOLVED (arms at or below the MC noise floor; excluded "
              "from the verdict): "
              + ", ".join(f"n={s['n']} L={s['L']}" for s in unresolved))
        print("  -> raise --samples, or use --float64, before trusting these shapes.")

    ratios = [s["unfactored_over_factored"] for s in resolved
              if s["unfactored_over_factored"]]
    print()
    if not ratios:
        verdict = ("NO VERDICT: no shape resolved both AUGMENT arms above the "
                   "Monte-Carlo noise floor. Nothing was measured; raise "
                   "--samples or --float64 and rerun.")
        print(f"  VERDICT: {verdict}")
        out = Path(a.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"shapes": shapes, "summaries": summaries, "verdict": verdict,
             "device": str(device), "dtype": str(dtype)}, indent=1) + "\n",
            encoding="utf-8")
        print(f"  wrote {out}")
        return 1
    if len(ratios) >= 2 and sum(1 for r in ratios if r >= 1.3) >= 2:
        verdict = ("SCHEDULE D4b: unfactored AUGMENT beats factored by >=1.3x at "
                   "two shapes. The dropped diagrams are worth deriving.")
    elif ratios and max(ratios) < 1.15:
        verdict = ("CLOSE THE LANE: unfactored AUGMENT is within 1.15x of factored "
                   "at every shape measured. The dropped diagrams are not the gap.")
    else:
        verdict = ("INCONCLUSIVE against PLAN's thresholds — report to Samuel with "
                   "the per-shape ratios and let him decide.")
    print(f"  VERDICT: {verdict}")

    out = Path(a.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"shapes": shapes, "summaries": summaries, "verdict": verdict,
         "device": str(device), "dtype": str(dtype)}, indent=1) + "\n",
        encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
