"""Lane M6, end-to-end: E21 (κ3 × edge into the κ4 core) and its cheap forms in ARC's factored
AUGMENT chain at n = 1024, scored on final MSE against organizer labels. `.venv-kprop`.

E21a/E21b write an n × n κ4-core increment (m6_terms), so unlike E3 they can be injected into
the factored chain. Exact forms use the factored pre-activation κ3 with repeated entries removed
(ARC's rep-subtraction, as in factor_k3.py:502-505 and Lane M3):

    Σ_i u_i κ_ijk         = ⅓ sym[(uᵀA) B Cᵀ + (uᵀB) C Aᵀ + (uᵀC) A Bᵀ]
    Σ_i w22_i Cm_ik κ_ijk = ⅙ Σ_roles Q (R ∘ (Cm (w22 ∘ P)))ᵀ

Arms (all on top of "b"):
    E21b, E21 (= a + b)            exact
    E21b_gain, E21_gain             κ3 -> 2βc²·Sym(μ ⊗ C) closed forms (O(n²) + one n³ product)
    E21b_spike16, E21_spike16       C inside the edge -> top-16 eigenpairs
    E21b_gain_from6                 gain form only at steps t >= 5

    --calibrate: n = 64, depth 4, float64; exact E21 == unfactored d+E21 (m2 filter),
                 exact E21b == unfactored d+E21b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
import m5_ablate as m5  # noqa: E402
import m6_terms as mt  # noqa: E402
from m5_ablate import ARMS, STATS, factored_nonlin_kprop_k3_ablate, fk3, gen, m2  # noqa: E402

FactoredTensor = fk3.FactoredTensor
BETA = 0.7


def rep_factors(ft):
    A, B, C = ft._factors
    rep = FactoredTensor.from_dstensor(ft.get_repeated()).factors
    return torch.cat([A, -rep[0]], 1), torch.cat([B, rep[1]], 1), torch.cat([C, rep[2]], 1)


def contract_vec(F, u):
    A, B, C = F
    X = ((u @ A) * B) @ C.T + ((u @ B) * C) @ A.T + ((u @ C) * A) @ B.T
    return (X + X.T) / 6.0


def contract_edge(F, w22, Cm):
    X = 0.0
    for P, Q, R in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        X = X + F[Q] @ (F[R] * (Cm @ (w22[:, None] * F[P]))).T
    return X / 6.0


def spike(C, r):
    ev, evec = torch.linalg.eigh(C)
    return ((evec[:, -r:] * ev[-r:]) @ evec[:, -r:].T).fill_diagonal_(0.0)


def e21_cores(K_in, form: str, which: str):
    n = K_in[1].n
    mu, C = K_in[1].core, K_in[2].core
    Cm = C.clone().fill_diagonal_(0.0)
    w = mt.wicks(mu, C.diag())
    sa, sb = mt.SA_X / (2 * n + 8), mt.SB_X / (2 * n + 8)
    out = 0.0
    if form == "gain":
        c2 = float(mu @ C @ mu / (mu @ mu) ** 2)
        out = mt.e21b_gain_closed(w, mu, Cm, n, BETA, c2, mt.SB_X)
        if which == "E21":
            out = out + mt.e21a_gain_closed(w, mu, Cm, n, BETA, c2, mt.SA_X)
        return out
    F = rep_factors(K_in[3])
    Ce = spike(C, 16) if form == "spike16" else Cm
    Xb = contract_edge(F, w[(2, 2)], Ce)
    Bb = w[(1, 1)][:, None] * Xb * w[(2, 1)][None, :]
    out = (sb * 0.5 * (Bb + Bb.T)).fill_diagonal_(0.0)
    if which == "E21":
        Xa = contract_vec(F, w[(1, 2)])
        out = out + (sa * w[(2, 1)][:, None] * w[(2, 1)][None, :] * Ce * Xa).fill_diagonal_(0.0)
    return out


E2E = {
    "b": None,
    "E21b": ("exact", "E21b", 0),
    "E21": ("exact", "E21", 0),
    "E21b_gain": ("gain", "E21b", 0),
    "E21_gain": ("gain", "E21", 0),
    "E21b_spike16": ("spike16", "E21b", 0),
    "E21_spike16": ("spike16", "E21", 0),
    "E21b_gain_from6": ("gain", "E21b", 5),
}


def run(mlp, n, depth, arm, dev, dtype):
    spec = E2E[arm]
    STATS.clear()
    cnt = {"t": 0}

    def hook(K_in, nonlin_wick_coef, augment=False, base=False, use_pK=True):
        t = cnt["t"]
        cnt["t"] += 1
        out = factored_nonlin_kprop_k3_ablate(K_in, nonlin_wick_coef, augment=True, cfg=ARMS["b"])
        if spec is not None and 3 in K_in and t >= spec[2]:
            out[4].core = out[4].core + e21_cores(K_in, spec[0], spec[1])
        return out

    fk3.factored_nonlin_kprop_k3 = hook
    try:
        return m2.run_variant(mlp, n, depth, "b", dev, dtype)
    finally:
        fk3.factored_nonlin_kprop_k3 = m5._ORIG


def calibrate(dev):
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    n, depth = 64, 4
    rng = np.random.default_rng(7)
    w = rng.standard_normal((depth, n, n)) * np.sqrt(2.0 / n)
    mlp = m2.build_mlp(w, dev, dtype)
    d0, _ = m2.run_variant(mlp, n, depth, "d0", dev, dtype)
    out = {}
    for fam in ("E21b", "E21"):
        ref, _ = m2.run_variant(mlp, n, depth, f"d+{fam}", dev, dtype)
        got, _ = run(mlp, n, depth, fam, dev, dtype)
        out[f"{fam}: |got - unfactored| / |unfactored - d0|"] = float(np.abs(got - ref).max() / np.abs(ref - d0).max())
    print(json.dumps(out, indent=1))
    (HERE / "m6_e2e_calibrate.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--nets", default="0:16")
    ap.add_argument("--arms", default=",".join(E2E))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    dev = torch.device("cuda")
    torch.set_grad_enabled(False)
    if a.calibrate:
        calibrate(dev)
        return 0
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    n, depth = 1024, 16
    mini = np.load(gen.cache_dir() / "mini16_truth.npz")
    lo, hi = (int(x) for x in a.nets.split(":"))
    res = {"arms": a.arms.split(","), "beta": BETA, "nets": []}
    prog = os.environ.get("WORKERQ_PROGRESS")
    for i in range(lo, hi):
        seed = int(mini["seeds"][i])
        mlp = m2.build_mlp(gen.mint_numpy(seed, n, depth), dev, dtype)
        truth, floor = np.array(mini["truth"][i]), float(mini["avg_variance"][i]) / 1e9
        rec = {"i": i, "seed": seed, "v73_mse_final": float(mini["v73_mse"][i])}
        for arm in res["arms"]:
            t0 = time.perf_counter()
            means, _ = run(mlp, n, depth, arm, dev, dtype)
            pl = ((means - truth) ** 2).mean(1) - floor
            rec[arm] = {"mse_final": float(pl[-1]), "mse_all": float(pl.mean()), "mse_layers": pl.tolist(),
                        "s": time.perf_counter() - t0}
            print(json.dumps({"i": i, "arm": arm, "mse_final": rec[arm]["mse_final"]}), flush=True)
            torch.cuda.empty_cache()
        res["nets"].append(rec)
        if a.out:
            Path(a.out).write_text(json.dumps(res) + "\n", encoding="utf-8")
        if prog:
            Path(prog).write_text(f"{(i - lo + 1) / (hi - lo):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
