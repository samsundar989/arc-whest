"""Lane M2 -- what do ARC's factored AUGMENT drops cost? Runs in `.venv-kprop`.

ARC's factored k_max = 3 AUGMENT (factor_k3) drops 4 of the 91 AUGMENT
(int_part, vec_part) terms -- the ones `kprop_harmonic.factored_keeps_term` rejects
(enumerated 2026-10-06, `m0_terms` below):

    T1    (1,1,1)  ((0,1,1),(1,0,1),(1,1,0))   x1   covariance triangle, top slice
    E3    (1,1,1)  ((0,1,1),(1,1,1))           x3   kappa3 x covariance edge, top slice
    E21a  (2,1,1)  ((0,1,1),(1,1,1))           x1   kappa3 x edge, (2,1,1) analogue (edge on the 1,1 pair)
    E21b  (2,1,1)  ((1,0,1),(1,1,1))           x2   kappa3 x edge, (2,1,1) analogue (edge touching the 2)

Variants per net (all k_max = 3, ReLU, avg metric 2 = He on EVERY layer, ours):

    a     factored SIMPLE
    b     factored AUGMENT
    c     unfactored AUGMENT (dense kappa3, all 91 terms)
    d0    unfactored AUGMENT restricted to the factored-kept 87 terms -> must equal b
    d+F   d0 plus family F alone, F in {T1, E3, E21a, E21b, E21 = E21a+E21b, E = E3+E21}

Metrics vs our MC labels (tools/fresh/mc.py --lane-m, N = 1e8): final-layer MSE minus
the label floor, all-layer MSE (floor-free), per-layer MSE; runtime.

The restriction uses the same mechanism as ARC's own `tests/test_factor_k3.py`
(`test_factored_kprop_augment`): swap `kprop_harmonic.get_all_terms_iso` for a
filtered copy during the unfactored nonlinear step. factor_k3 keeps its own binding.

    PYTHONPATH=external/mlp_cumulant_propagation/src .venv-kprop/Scripts/python.exe \
        experiments/2026-10-06_laneM/m2_variants.py --width 256 --nets 0:8 --out <abs.json>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.append(str(ROOT / "tools" / "fresh"))
import gen  # noqa: E402

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from mlp_kprop import kprop_harmonic as kh  # noqa: E402
from mlp_kprop.kprop_harmonic import AUGMENT, SIMPLE  # noqa: E402
from mlp_kprop.mlp import MLP  # noqa: E402

K_MAX = 3
FAMILIES = {
    "T1": [((1, 1, 1), ((0, 1, 1), (1, 0, 1), (1, 1, 0)))],
    "E3": [((1, 1, 1), ((0, 1, 1), (1, 1, 1)))],
    "E21a": [((2, 1, 1), ((0, 1, 1), (1, 1, 1)))],
    "E21b": [((2, 1, 1), ((1, 0, 1), (1, 1, 1)))],
}
FAMILIES["E21"] = FAMILIES["E21a"] + FAMILIES["E21b"]
FAMILIES["E"] = FAMILIES["E3"] + FAMILIES["E21"]

_REAL_ISO = kh.get_all_terms_iso


def m0_terms() -> dict:
    terms = _REAL_ISO(K_MAX, d_max=kh.get_d_max(K_MAX, AUGMENT), augment=True)
    dropped = [(ip, vp, c) for ip, vps in terms.items() for vp, c in vps.items()
               if not kh.factored_keeps_term(K_MAX, ip, vp)]
    listed = {t for fam in ("T1", "E3", "E21a", "E21b") for t in FAMILIES[fam]}
    assert {(ip, vp) for ip, vp, _c in dropped} == listed, f"family table stale: {dropped}"
    return {"n_terms": sum(len(v) for v in terms.values()), "dropped": [list(map(str, d)) for d in dropped]}


def filtered_iso(addback: set):
    def iso(k_max, d_max=None, use_mean_var=False, augment=False):
        ret = _REAL_ISO(k_max, d_max=d_max, use_mean_var=use_mean_var, augment=augment)
        return {ip: {vp: c for vp, c in vps.items()
                     if kh.factored_keeps_term(k_max, ip, vp) or (ip, vp) in addback}
                for ip, vps in ret.items()}
    return iso


def build_mlp(w: np.ndarray, dev, dtype) -> MLP:
    """Our net (16 x [W, ReLU], He 2/n everywhere) as ARC's MLP (17 linears, 16 ReLUs).

    The 17th linear is never reached: we stop at act15. init_scale -> 2.0 on every
    layer (ARC's He default gives the INPUT layer w_var 1; our layer 0 is 2/n too).
    """
    depth, n = w.shape[0], w.shape[1]
    m = MLP(hidden_dim=n, num_layers=depth + 1, nonlin="relu").to(dev, dtype=dtype)
    for l in range(depth):
        m.Ws[l].weight.data = torch.from_numpy(np.ascontiguousarray(w[l].T)).to(dev, dtype=dtype)
        assert m.Ws[l].bias is None
    m.init_scale = [2.0] * (depth + 1)
    return m


def run_variant(mlp: MLP, n: int, depth: int, variant: str, dev, dtype):
    kind = SIMPLE if variant == "a" else AUGMENT
    factor = variant in ("a", "b")
    K_in = {1: torch.zeros(n, device=dev, dtype=dtype), 2: torch.eye(n, device=dev, dtype=dtype)}
    restore = None
    if variant.startswith("d"):
        fam = variant[2:] if variant.startswith("d+") else None
        add = set(FAMILIES[fam]) if fam else set()
        restore = kh.get_all_terms_iso
        kh.get_all_terms_iso = filtered_iso(add)
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        Kl = kh.mlp_kprop(mlp, K_in, k_max=K_MAX, output_all=True, kind=kind,
                          factor=factor, up_to_layer=f"act{depth - 1}", output_d_max=1)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    finally:
        if restore is not None:
            kh.get_all_terms_iso = restore
    means = np.stack([Kl[f"act{l}"][1].core.double().cpu().numpy() for l in range(depth)])
    return means, dt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--nets", default="0:8")
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--variants", default="a,b,c,d0,d+T1,d+E3,d+E21a,d+E21b,d+E")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--no-labels", action="store_true", help="timing/bench only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dev = torch.device("cuda")
    dtype = getattr(torch, a.dtype)
    torch.set_default_dtype(dtype)
    lo, hi = (int(t) for t in a.nets.split(":"))
    variants = a.variants.split(",")
    out = {"m0": m0_terms(), "width": a.width, "depth": a.depth, "dtype": a.dtype,
           "k_max": K_MAX, "torch": torch.__version__, "nets": []}
    prog = os.environ.get("WORKERQ_PROGRESS")
    outp = Path(a.out)
    for i in range(lo, hi):
        seed = gen.lane_m_seed(a.width, i)
        w = gen.mint_numpy(seed, a.width, a.depth)
        rec = {"i": i, "seed": seed, "variants": {}}
        lab = None
        if not a.no_labels:
            with np.load(gen.label_path(f"lane_m_w{a.width}", seed, 10**8)) as d:
                assert str(d["weight_sha256"]) == gen.sha256_weights(w)
                lab = {"means": np.array(d["means"]), "floor": np.array(d["var"]).mean(1) / int(d["n"])}
            rec["floor_final"] = float(lab["floor"][-1])
        mlp = build_mlp(w, dev, dtype)
        for v in variants:
            torch.cuda.reset_peak_memory_stats()
            try:
                means, dt = run_variant(mlp, a.width, a.depth, v, dev, dtype)
            except torch.cuda.OutOfMemoryError as e:
                rec["variants"][v] = {"error": "OOM", "detail": str(e)[:200]}
                torch.cuda.empty_cache()
                continue
            r = {"seconds": dt, "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                 "means_final": means[-1].tolist()}
            if lab is not None:
                per_layer = ((means - lab["means"]) ** 2).mean(axis=1) - lab["floor"]
                r.update(mse_final=float(per_layer[-1]), mse_all=float(per_layer.mean()),
                         mse_layers=per_layer.tolist())
            rec["variants"][v] = r
            print(json.dumps({"width": a.width, "i": i, "v": v, "s": round(dt, 2),
                              "gib": round(r["peak_gib"], 2),
                              "mse_final": r.get("mse_final")}), flush=True)
            torch.cuda.empty_cache()
        if "b" in rec["variants"] and "d0" in rec["variants"] and "error" not in rec["variants"]["d0"]:
            rec["d0_vs_b_maxabs"] = float(np.max(np.abs(
                np.array(rec["variants"]["d0"]["means_final"]) -
                np.array(rec["variants"]["b"]["means_final"]))))
            print(json.dumps({"i": i, "d0_vs_b_maxabs": rec["d0_vs_b_maxabs"]}), flush=True)
        out["nets"].append(rec)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(out) + "\n", encoding="utf-8")
        if prog:
            Path(prog).write_text(f"{(i - lo + 1) / (hi - lo):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
