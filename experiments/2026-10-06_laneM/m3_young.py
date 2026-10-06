"""Lane M3 — can a CHEAP (youngest-lineage) κ3 × edge term keep the gain? `.venv-kprop`.

E21b (the dropped (2,1,1) κ3 × covariance-edge term, ~80 % of G_drop) written in ARC's
FACTORED representation, as an add-on to factored AUGMENT's κ4 core:

    B_jk = s * sym_jk[ W11_j W21_k  Σ_i W22_i C_ik κ3_ijk ],  diag(B) = 0,  K_out[4].core += B

with κ3 = Sym(Σ_r A⊗B⊗C) taken with repeated entries removed (ARC's rep-subtracted
factors, as in factor_k3.py:502-505) and C = off-diagonal input covariance. Each of the 6
role assignments (P→i, Q→j, R→k) is  Q @ (R ∘ (C @ (W22 ⊙ P)))ᵀ.

Modes:
  --calibrate   n = 64, depth 4, float64: find the scalar s for which factored b + E21b
                reproduces UNFACTORED d+E21b (m2_variants' filter), and report the residual.
  (default)     per net: b, b+E21b(all columns), b+E21b(young: only the κ3 columns born at
                the previous layer), and optionally young-2 (last two layers' births).

    PYTHONPATH=external/mlp_cumulant_propagation/src .venv-kprop/Scripts/python.exe \
        experiments/2026-10-06_laneM/m3_young.py --width 512 --nets 0:8 --out <abs.json>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
import m2_variants as m2  # noqa: E402
from m2_variants import gen, kh  # noqa: E402

import mlp_kprop.factor_k3 as fk3  # noqa: E402
from mlp_kprop.factor_k3 import FactoredTensor  # noqa: E402

_ORIG = fk3.factored_nonlin_kprop_k3


class E21bHook:
    def __init__(self, scale: float, young: int | None):
        self.scale = scale          # None/0 -> off
        self.young = young          # None = all columns; g = births of the last g layers
        self.births = []            # column offset where each layer's births start

    def __call__(self, K_in, nonlin_wick_coef, augment=False, base=False, use_pK=True):
        K_out = _ORIG(K_in=K_in, nonlin_wick_coef=nonlin_wick_coef, augment=augment,
                      base=base, use_pK=use_pK)
        R_in = K_in[3].factors[0].shape[1] if 3 in K_in else 0
        if self.scale and 3 in K_in and R_in > 0 and 4 in K_out:
            lo = 0
            if self.young is not None:
                starts = self.births[-self.young:] if self.births else [0]
                lo = starts[0] if starts else 0
            A, B, C = (f[:, lo:] for f in K_in[3].factors)
            ft = FactoredTensor(n=A.shape[0], d=3, factors=(A, B, C))
            rep = FactoredTensor.from_dstensor(ft.get_repeated()).factors
            F = (torch.cat([A, -rep[0]], 1), torch.cat([B, rep[1]], 1), torch.cat([C, rep[2]], 1))
            mean, var = K_in[1].core, K_in[2].core.diag()
            w = lambda k, p: nonlin_wick_coef(mean=mean, var=var, k=k, p=p)  # noqa: E731
            W22, W11, W21 = w(2, 2), w(1, 1), w(2, 1)
            Cm = K_in[2].core.clone()
            Cm.fill_diagonal_(0.0)
            X = 0.0
            for P, Q, R in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
                X = X + F[Q] @ (F[R] * (Cm @ (W22[:, None] * F[P]))).T
            X = X / 6.0
            Bm = W11[:, None] * X * W21[None, :]
            Bm = 0.5 * (Bm + Bm.T)
            Bm.fill_diagonal_(0.0)
            K_out[4].core = K_out[4].core + self.scale * Bm
        # this layer's births start at R_in in K_out[3] (old columns first, births appended)
        self.births.append(R_in)
        return K_out


def run(mlp, n, depth, variant, dev, dtype, hook=None):
    if hook is not None:
        fk3.factored_nonlin_kprop_k3 = hook
    try:
        return m2.run_variant(mlp, n, depth, variant, dev, dtype)
    finally:
        fk3.factored_nonlin_kprop_k3 = _ORIG


def calibrate(dev) -> dict:
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    n, depth = 64, 4
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((depth, n, n)) * np.sqrt(2.0 / n)).astype(np.float64)
    mlp = m2.build_mlp(w, dev, dtype)
    ref, _ = run(mlp, n, depth, "d+E21b", dev, dtype)
    d0, _ = run(mlp, n, depth, "d0", dev, dtype)
    base, _ = run(mlp, n, depth, "b", dev, dtype)
    out = {"n": n, "depth": depth, "|d+E21b - d0|": float(np.abs(ref - d0).max()),
           "|b - d0|": float(np.abs(base - d0).max()), "candidates": {}}
    for name, s in {"12/(2n+8)": 12 / (2 * n + 8), "24/(2n+8)": 24 / (2 * n + 8),
                    "6/(2n+8)": 6 / (2 * n + 8), "48/(2n+8)": 48 / (2 * n + 8),
                    "36/(2n+8)": 36 / (2 * n + 8), "72/(2n+8)": 72 / (2 * n + 8)}.items():
        got, _ = run(mlp, n, depth, "b", dev, dtype, E21bHook(s, None))
        out["candidates"][name] = float(np.abs(got - ref).max() / max(np.abs(ref - d0).max(), 1e-30))
    # least-squares scalar on the output shift (shift is ~linear in s at this size)
    one, _ = run(mlp, n, depth, "b", dev, dtype, E21bHook(1.0, None))
    dlt, tgt = (one - base).ravel(), (ref - base).ravel()
    s_ls = float(dlt @ tgt / (dlt @ dlt))
    got, _ = run(mlp, n, depth, "b", dev, dtype, E21bHook(s_ls, None))
    out["s_ls"] = s_ls
    out["s_ls_x(2n+8)"] = s_ls * (2 * n + 8)
    out["s_ls_rel_resid"] = float(np.abs(got - ref).max() / np.abs(ref - d0).max())
    print(json.dumps(out, indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--nets", default="0:8")
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--coef", type=float, default=None, help="s x (2n+8); from --calibrate")
    ap.add_argument("--young", default="1,2", help="comma list of young-generation counts")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    dev = torch.device("cuda")
    if a.calibrate:
        r = calibrate(dev)
        if a.out:
            Path(a.out).write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
        return 0
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    s = a.coef / (2 * a.width + 8)
    lo, hi = (int(t) for t in a.nets.split(":"))
    youngs = [int(g) for g in a.young.split(",") if g]
    res = {"width": a.width, "coef_x(2n+8)": a.coef, "nets": []}
    for i in range(lo, hi):
        seed = gen.lane_m_seed(a.width, i)
        w = gen.mint_numpy(seed, a.width, a.depth)
        with np.load(gen.label_path(f"lane_m_w{a.width}", seed, 10**8)) as d:
            lab_m = np.array(d["means"])
            fl = np.array(d["var"]).mean(1) / int(d["n"])
        mlp = m2.build_mlp(w, dev, dtype)
        rec = {"i": i}
        arms = [("b", None), ("b+E21b", E21bHook(s, None))] + \
               [(f"b+E21b_young{g}", E21bHook(s, g)) for g in youngs]
        for name, hook in arms:
            t0 = time.perf_counter()
            means, _dt = run(mlp, a.width, a.depth, "b", dev, dtype, hook)
            pl = ((means - lab_m) ** 2).mean(1) - fl
            rec[name] = {"mse_final": float(pl[-1]), "mse_all": float(pl.mean()),
                         "s": time.perf_counter() - t0}
            print(json.dumps({"i": i, "arm": name, "mse_final": rec[name]["mse_final"]}), flush=True)
        res["nets"].append(rec)
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    arms = [k for k in res["nets"][0] if k != "i"]
    b = np.array([r["b"]["mse_final"] for r in res["nets"]])
    for k in arms:
        v = np.array([r[k]["mse_final"] for r in res["nets"]])
        print(f"{k:22s} final {v.mean():.4e}  gain vs b {(b.mean() - v.mean()) / b.mean() * 100:+.1f}%"
              f"  wins {(v < b).sum()}/{len(v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
