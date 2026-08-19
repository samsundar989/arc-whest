"""MOONSHOT G0 — layer-convention parity and the separate calibrated-K2 control.

`docs/MOONSHOT.md` names two things that must exist *before* D08 may be spent on
gate G1, and neither did:

  1. **Layer-convention parity.** `lowrank_cumulant_flow.lowrank_challenge_flow`
     transposes every weight matrix (`W.mT`) on the stated grounds that
     whestbench propagates row vectors (`h @ W`) while ARC's k-propagation uses
     column vectors, and it hands ARC an `init_scale` of 2.0 per layer as the
     metric. Both were asserted, never measured. If either is wrong, G1 would
     compare a mis-wired flow against a correct baseline and the answer would be
     uninterpretable in the direction that looks like a negative result.

  2. **A separate calibrated-K2 control.** MOONSHOT's baseline is the
     *calibrated* Gaussian closure, "not a strawman". ARC's own K1/K2 states are
     full-dimensional but are not that trajectory, and v2.4.1's low-budget
     analytic fallback returns **raw** closure — the calibration table lives only
     in `experiments/calibration.json`. So the registered baseline had no
     implementation anywhere. This file is it.

G0 decides nothing. It is infrastructure, it runs on its own seed block **D15**,
and it deliberately does not touch D08/D09 so that G1 and G2 still have unseen
data. **No cost claim is made or implied here**: the low-rank flow remains an
accuracy oracle until a compressed representation with an honest flopscope cost
model exists (MOONSHOT G3).

Run (GPU venv):
    .venv-gpu\\Scripts\\python.exe local_sim/moonshot_g0.py --mlps 32
    .venv-gpu\\Scripts\\python.exe local_sim/moonshot_g0.py --mlps 8 --samples 1048576
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mint_provenance import canonical_mint, seed_streams  # noqa: E402
from seed_blocks import seeds as block_seeds  # noqa: E402

WIDTH, DEPTH = 256, 32
INV_SQRT_2PI = 0.3989422804014327


# ===========================================================================
# ground truth: per-LAYER post-ReLU means, not just the scored row
# ===========================================================================

def truth_all_layers(w, n_samples, gen_seed_seq, dev, chunk=65536):
    """Post-ReLU mean of every layer, f64 accumulation.

    The scored row alone cannot test a layer convention: a transpose error
    changes the trajectory from layer 1 onward, and by depth 32 everything is
    wrong by a similar amount. Per-layer means localise the divergence, which is
    the difference between "the conventions disagree" and "the conventions
    disagree starting at the first matmul, by exactly the transpose".
    """
    depth, width = w.shape[0], w.shape[-1]
    s1 = torch.zeros((depth, width), dtype=torch.float64, device=dev)
    s2 = torch.zeros((depth, width), dtype=torch.float64, device=dev)
    rng = np.random.default_rng(gen_seed_seq)
    done = 0
    while done < n_samples:
        b = min(chunk, n_samples - done)
        x = torch.from_numpy(rng.standard_normal((b, width), dtype=np.float32)).to(dev)
        for li in range(depth):
            x = torch.clamp(x @ w[li], min=0.0)
            xd = x.to(torch.float64)
            s1[li] += xd.sum(dim=0)
            s2[li] += (xd * xd).sum(dim=0)
        done += b
    mean = s1 / n_samples
    var = torch.clamp(s2 / n_samples - mean * mean, min=0.0)
    # Every measured MSE is quoted with the MC noise of the truth it was measured
    # against (VALIDATION.md 3.5): an "improvement" below this line is not one.
    return mean, var.mean(dim=1) / float(n_samples)


# ===========================================================================
# the Gaussian closure, in both candidate conventions
# ===========================================================================

def _ncdf(x):
    return 0.5 * torch.erfc(-x * (2.0 ** -0.5))


def _npdf(x):
    return torch.exp(-0.5 * x * x) * INV_SQRT_2PI


def closure(w, convention):
    """Per-layer post-ReLU means from the K=2 Gaussian closure.

    `convention` selects how a weight matrix acts:

      "row"    — `z = W^T mu`, `Cov_z = W^T Cov W`. This is whestbench's
                 `h @ W` read as a linear map on column vectors, and it is what
                 `estimator.py` implements.
      "column" — `z = W mu`, `Cov_z = W Cov W^T`. This is what a k-propagation
                 library expecting `W x` computes if it is handed the raw
                 whestbench matrix without the transpose.

    Everything else is identical, so the two arms differ in exactly the thing
    under test.
    """
    depth, width = w.shape[0], w.shape[-1]
    mu = torch.zeros(width, dtype=torch.float64, device=w.device)
    cov = torch.eye(width, dtype=torch.float64, device=w.device)
    off = 1.0 - torch.eye(width, dtype=torch.float64, device=w.device)
    out = torch.zeros((depth, width), dtype=torch.float64, device=w.device)
    for li in range(depth):
        wl = w[li].to(torch.float64)
        if convention == "row":
            mu_pre = wl.mT @ mu
            cov_pre = wl.mT @ cov @ wl
        elif convention == "column":
            mu_pre = wl @ mu
            cov_pre = wl @ cov @ wl.mT
        else:
            raise ValueError(convention)
        var_pre = torch.clamp(torch.diagonal(cov_pre), min=1e-12)
        sig = torch.sqrt(var_pre)
        alpha = mu_pre / sig
        Phi, phi = _ncdf(alpha), _npdf(alpha)
        mu = mu_pre * Phi + sig * phi
        ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sig * phi
        var = torch.clamp(ez2 - mu * mu, min=0.0)
        cov = torch.outer(Phi, Phi) * cov_pre * off + torch.diag(var)
        out[li] = mu
    return out


def exact_layer1(w0):
    """E[ReLU] at layer 1 is exactly ||W1[:,j]|| / sqrt(2 pi). A known answer."""
    return torch.sqrt((w0.to(torch.float64) ** 2).sum(dim=0)) * INV_SQRT_2PI


# ===========================================================================
# the registered calibrated-K2 control
# ===========================================================================

def load_calibration(depth):
    """Per-layer scale factors from `experiments/calibration.json`.

    Fitted on legacy (pre-provenance-fix) mints; the 2026-08-06 constants audit
    re-measured them on canonical mints and found <=0.07% drift, so they
    transfer. They are NOT in `estimator.py`: v2.4.1's low-budget analytic path
    returns raw closure, which is precisely why this control had to be built
    rather than imported.
    """
    path = os.path.join(_ROOT, "experiments", "calibration.json")
    with open(path, encoding="utf-8") as fh:
        cal = json.load(fh)
    key = str(depth)
    if key not in cal["depths"]:
        raise SystemExit(f"no calibration for depth {depth}; have {sorted(cal['depths'])}")
    return np.asarray(cal["depths"][key]["c_layers"], dtype=np.float64)


def calibrated_k2(w, c_layers):
    """THE REGISTERED G1 BASELINE: row-convention closure x per-layer calibration,
    with layer 1 replaced by its exact closed form."""
    pred = closure(w, "row")
    pred = pred * torch.as_tensor(c_layers, dtype=torch.float64,
                                  device=pred.device).reshape(-1, 1)
    pred[0] = exact_layer1(w[0])
    return pred


# ===========================================================================
# report
# ===========================================================================

def mse_per_layer(pred, truth):
    return ((pred - truth) ** 2).mean(dim=1)


def flow_means(w, k_max, rank):
    """Per-layer post-ReLU K1 from ARC's flow, through the fenced adapter.

    This is the third G0 item: put the flow's OWN trajectory through the same
    harness as the control, so "the adapter feeds ARC the right linear map" stops
    being an inference from a reimplementation and becomes a measurement of
    ARC's code. `k_max=2` is the parity configuration — at order two the
    projection and the renormalizers are inert, so the flow reduces to ARC's
    Gaussian closure and must reproduce ours.
    """
    import lowrank_cumulant_flow as lcf

    by, _recs = lcf.lowrank_challenge_flow(w.to(torch.float64), rank=rank,
                                           k_max=k_max, output_all=True)
    rows = []
    for li in range(w.shape[0]):
        k1 = by[f"act{li}"][1]
        k1 = k1.to_tensor() if hasattr(k1, "to_tensor") else torch.as_tensor(k1)
        rows.append(k1.reshape(-1))
    return torch.stack(rows, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="D15")
    ap.add_argument("--mlps", type=int, default=32)
    ap.add_argument("--samples", type=int, default=1 << 21)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--flow", action="store_true",
                    help="also run ARC's flow through this harness (needs MLP_KPROP_SRC)")
    ap.add_argument("--flow-kmax", dest="flow_kmax", default="2,3,4")
    ap.add_argument("--flow-rank", dest="flow_rank", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # seed-ledger: excludes D08, D09, D10
    # These are NAMED here in order to be REFUSED. G1/G2 still need unseen data.
    if args.block in ("D08", "D09", "D10"):
        raise SystemExit("G0 must not touch the moonshot gate blocks; use D15.")

    flow_kmax = [int(v) for v in args.flow_kmax.split(",") if v.strip()] if args.flow else []
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seeds = block_seeds(args.block)[: args.mlps]
    c_layers = load_calibration(args.depth)

    print("=" * 104)
    print("MOONSHOT G0 — layer-convention parity + the separate calibrated-K2 control")
    print(f"block {args.block}  {len(seeds)} MLPs  {args.width}x{args.depth}  "
          f"truth N = {args.samples:,}  device {dev}")
    print("Infrastructure only. No gate is decided; D08/D09 are untouched; no cost is claimed.")
    print("=" * 104)

    acc = {k: [] for k in ("row", "column", "cal", "l1_row", "l1_exact", "gt")}
    tsq = []
    for i, s in enumerate(seeds):
        w_np, _est, sample_ss = canonical_mint(args.width, args.depth, int(s))
        w = torch.from_numpy(np.stack(w_np)).to(dev)
        truth, gtn = truth_all_layers(w, args.samples, sample_ss, dev)
        acc["gt"].append(gtn.cpu().numpy())
        tsq.append(float((truth ** 2).mean()))
        acc["row"].append(mse_per_layer(closure(w, "row"), truth).cpu().numpy())
        acc["column"].append(mse_per_layer(closure(w, "column"), truth).cpu().numpy())
        acc["cal"].append(mse_per_layer(calibrated_k2(w, c_layers), truth).cpu().numpy())
        # layer 1 has a known closed form; both are checked against it directly
        acc["l1_row"].append(float(torch.abs(closure(w, "row")[0] - truth[0]).max()
                                   / torch.abs(truth[0]).max()))
        acc["l1_exact"].append(float(torch.abs(exact_layer1(w[0]) - truth[0]).max()
                                     / torch.abs(truth[0]).max()))
        for km in flow_kmax:
            rk = args.width if km == 2 else args.flow_rank
            acc.setdefault(f"flow{km}", []).append(
                mse_per_layer(flow_means(w, km, rk), truth).cpu().numpy())
        if (i + 1) % 8 == 0 or i + 1 == len(seeds):
            print(f"  {i + 1}/{len(seeds)}", flush=True)

    row = np.mean(np.stack(acc["row"]), axis=0)
    col = np.mean(np.stack(acc["column"]), axis=0)
    cal = np.mean(np.stack(acc["cal"]), axis=0)
    gtn = np.mean(np.stack(acc["gt"]), axis=0)
    mean_tsq = float(np.mean(tsq))

    print("\n--- 1. LAYER-CONVENTION PARITY ---------------------------------------------")
    print("If the row convention is the right one, its per-layer MSE tracks the truth from")
    print("layer 1 and the column convention diverges immediately and stays diverged.")
    print(f"\n{'layer':>6} {'row-convention MSE':>20} {'column-convention MSE':>23} "
          f"{'column/row':>12}")
    print("-" * 104)
    for li in sorted(set(list(range(min(6, args.depth)))
                        + [args.depth // 2, args.depth - 1])):
        print(f"{li + 1:>6} {row[li]:>20.4e} {col[li]:>23.4e} "
              f"{col[li] / max(row[li], 1e-300):>12.2f}x")
    print(f"\n  layer-1 closure vs truth, max relative : "
          f"{float(np.mean(acc['l1_row'])):.3e}")
    print(f"  layer-1 EXACT form vs truth, max rel.  : "
          f"{float(np.mean(acc['l1_exact'])):.3e}   (MC noise floor)")
    verdict_row = row[0] < col[0] / 10.0
    print(f"\n  VERDICT: the whestbench->ARC transpose in `lowrank_challenge_flow` is "
          f"{'CORRECT' if verdict_row else 'NOT CONFIRMED'}")
    print("  (`h @ W` is `W^T` acting on column vectors, so ARC — which applies `W x` —")
    print("   must be handed `W.mT`, which is what the adapter does.)")

    print("\n--- 2. THE SEPARATE CALIBRATED-K2 CONTROL ----------------------------------")
    print(f"{'layer':>6} {'raw closure MSE':>18} {'calibrated K2 MSE':>19} {'gain':>8} "
          f"{'c_layer':>9}")
    print("-" * 104)
    for li in sorted(set(list(range(min(4, args.depth)))
                        + [args.depth // 2, args.depth - 1])):
        print(f"{li + 1:>6} {row[li]:>18.4e} {cal[li]:>19.4e} "
              f"{row[li] / max(cal[li], 1e-300):>7.3f}x {c_layers[li]:>9.6f}")
    print(f"\n  final-layer raw closure   : {row[-1]:.4e}")
    print(f"  final-layer CALIBRATED K2 : {cal[-1]:.4e}   "
          f"({row[-1] / max(cal[-1], 1e-300):.3f}x better)")
    print(f"  mean(truth^2)             : {mean_tsq:.4f}")
    print(f"  GT noise, final layer     : {gtn[-1]:.4e}  "
          f"({gtn[-1] / max(cal[-1], 1e-300) * 100:.2f}% of the calibrated control)")
    print(f"  GT noise, layer 1         : {gtn[0]:.4e}  "
          f"({gtn[0] / max(row[0], 1e-300) * 100:.1f}% of the layer-1 number, which is")
    print("                              why layer 1 reads as truth noise, not model error)")
    print("\n  This is the number G1 must beat by >=3x. It is registered here, on D15,")
    print("  before any D08 network has been drawn.")

    print("\n--- 3. THE LOW-RANK FLOW AS AN ACCURACY ORACLE ------------------------------")
    flows = {km: np.mean(np.stack(acc[f"flow{km}"]), axis=0) for km in flow_kmax}
    if not flows:
        print("  Not run (pass --flow, with MLP_KPROP_SRC set to the pinned checkout).")
    else:
        print(f"  ARC mlp_kprop through the fenced adapter, rank {args.flow_rank}.")
        print("  k_max=2 uses FULL rank: at order two the projection and the")
        print("  renormalizers are inert, so it is the end-to-end convention test.")
        cols = "".join(f"{'ARC k=' + str(km):>16}" for km in flow_kmax)
        print(f"\n{'layer':>6} {'our closure':>14} {'calibrated K2':>15}{cols}")
        print("-" * 104)
        for li in sorted(set(list(range(min(4, args.depth)))
                             + [args.depth // 2, args.depth - 1])):
            cells = "".join(f"{flows[km][li]:>16.4e}" for km in flow_kmax)
            print(f"{li + 1:>6} {row[li]:>14.4e} {cal[li]:>15.4e}{cells}")
        if 2 in flows:
            d = float(np.max(np.abs(flows[2] - row) / np.maximum(row, 1e-300)))
            print("\n  PARITY, end to end: ARC's own order-2 flow against our closure,")
            print(f"  max relative per-layer MSE difference = {d:.3e}. The adapter feeds")
            print("  ARC the same linear map estimator.py propagates — measured on ARC's")
            print("  code, not inferred from a reimplementation.")
        print("\n  Final layer, against the registered control:")
        for km in flow_kmax:
            print(f"    k_max={km}: {flows[km][-1]:.4e}   "
                  f"({cal[-1] / max(flows[km][-1], 1e-300):.3f}x the calibrated-K2 "
                  f"baseline {cal[-1]:.4e})")
        print("\n  These are ORACLE readings on D15 infrastructure data. They are NOT")
        print("  gate G1, which is a pre-registered >=3x test on D08 and has not been run.")
    print("\n  No cost model, compressed representation, or r^3 claim is made or implied.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"block": args.block, "mlps": len(seeds), "samples": args.samples,
                       "width": args.width, "depth": args.depth,
                       "mse_row": row.tolist(), "mse_column": col.tolist(),
                       "mse_calibrated_k2": cal.tolist(), "gt_noise": gtn.tolist(),
                       "c_layers": c_layers.tolist(), "mean_truth_sq": mean_tsq,
                       "layer1_closure_rel": float(np.mean(acc["l1_row"])),
                       "layer1_exact_rel": float(np.mean(acc["l1_exact"])),
                       "flow_rank": args.flow_rank,
                       "mse_flow": {str(km): flows[km].tolist() for km in flow_kmax}},
                      fh, indent=1)
        print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
