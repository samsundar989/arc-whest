"""Lane M5 -- which ingredients of ARC's factored AUGMENT carry its gain? `.venv-kprop`.

`factored_nonlin_kprop_k3_ablate` is a copy of ARC's `factor_k3.factored_nonlin_kprop_k3`
(external/mlp_cumulant_propagation, commit 93d091a, MIT) with one switch per ingredient.
With the default `Ablate()` it is ARC's augment path line for line (checked: `--calibrate`).

    legs   "aug"      ARC's five legs: light (1,1,C), heavy (2,1,D21/2), (1,2,D21ᵀ/2),
                      (2,2,K22/4), doubled edge (2,2,C²/2); light x heavy pairs only
           "simple"   legs = [(1,1,C)] only (SIMPLE births), everything else AUGMENT
           "dressed"  one dressed leg X̃ = w1∘C + H (R1 §5.1, handoff M5.2):
                        H[a,j] = ½ w2_a D21[a,j]
                               + (w3_j/w2_j)(½ w1_a D21[j,a] + ¼ w2_a K22[a,j] + ½ w2_a C_aj²)
                      births Sym(X̃ ⊗ 3 w2 e_j ⊗ X̃)  (includes the spurious H ⊗ H piece)
           "dressed_nospur"  dressed minus Sym(H ⊗ 3 w2 e_j ⊗ H): algebraically == "aug"
           "asym"     ONE triple Sym(X ⊗ 3 w2 e_j ⊗ (X + 2H)), X = w1∘C: algebraically == "aug"
                      (Sym is bilinear), same column count as "dressed", no spurious term
    k4     "full"     ARC's r = 1 κ4 core
           "memoryless"  drop every d = 4 pK term that reads the input κ4 (a block with
                      |v| = 4): the core is rebuilt from this layer's κ2/κ3 only. The
                      transported memoryless core still feeds the K22 leg and 211→111.
           "gain"     core := r = 1 projection of 4βc²·Sym₃(C ⊗ C) (3 pairings summed),
                      c² = mᵀCm/‖m‖⁴ of the post-ReLU state
           "memoryless+gain"  memoryless core + the gain template's projection
    no211  drop the 211 H(d=4, r=1) → 111 κ3 births
    noK211 drop the pK_111 / pK_211 → K_211 contributions to the κ4 core

Runs:
    --calibrate            n = 64, depth 4, float64: aug == ARC; dressed_nospur == aug;
                           closed-form gain projection == DS_harmonic_proj(dense template)
    --mini --nets 0:16     the 16 organizer DEV nets at n = 1024, organizer labels

    PYTHONPATH=external/mlp_cumulant_propagation/src .venv-kprop/Scripts/python.exe \
        experiments/2026-10-07_laneM2/m5_ablate.py --mini --nets 0:16 --out <abs.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.append(str(ROOT / "experiments" / "2026-10-06_laneM"))
import m2_variants as m2  # noqa: E402
from m2_variants import gen  # noqa: E402

import mlp_kprop.factor_k3 as fk3  # noqa: E402
from mlp_kprop.diagslice import DSTensor, DSTower, diagslice, eval_part, zero_repeated  # noqa: E402
from mlp_kprop.cumulants import DS_pK_to_K  # noqa: E402
from mlp_kprop.harmonic import HTensor, DS_harmonic_proj, proj_coef  # noqa: E402
from mlp_kprop.kprop_harmonic import factored_keeps_term, get_all_terms_iso, multiply_wicks  # noqa: E402
from mlp_kprop.partitions import check_vec_partition  # noqa: E402
from mlp_kprop.tensor_utils import symmetrize  # noqa: E402

FactoredTensor = fk3.FactoredTensor
_ORIG = fk3.factored_nonlin_kprop_k3


@dataclass(frozen=True)
class Ablate:
    legs: str = "aug"
    k4: str = "full"
    no211: bool = False
    noK211: bool = False
    beta: float = 0.7
    dress_s: float = 1.0   # "dressed": X̃ = w1∘C + s·H (spurious term × s², useful cross term × s)
    k4_from: int = 0       # the k4 variant applies only at nonlinear steps t >= k4_from (0-based)


ARMS = {
    "b": Ablate(),
    "1_simple_legs": Ablate(legs="simple"),
    "2_dressed": Ablate(legs="dressed"),
    "2c_asym": Ablate(legs="asym"),
    "2d_dressed_s0.5": Ablate(legs="dressed", dress_s=0.5),
    "2e_dressed_s0.7": Ablate(legs="dressed", dress_s=0.7),
    "2f_dressed_s0.3": Ablate(legs="dressed", dress_s=0.3),
    "3a_memoryless": Ablate(k4="memoryless"),
    "3b_gain": Ablate(k4="gain"),
    "3c_memoryless+gain": Ablate(k4="memoryless+gain"),
    "4_no211": Ablate(no211=True),
    "5_noK211": Ablate(noK211=True),
    "45_no_k3k4_loop": Ablate(no211=True, noK211=True),
    "145_simple_legs_no_loop": Ablate(legs="simple", no211=True, noK211=True),
    "3d_gain_from6": Ablate(k4="gain", k4_from=5),
    "3e_memoryless+gain_from6": Ablate(k4="memoryless+gain", k4_from=5),
    "3f_memoryless_from6": Ablate(k4="memoryless", k4_from=5),
}


def gain_core(mean: torch.Tensor, C: torch.Tensor, beta: float) -> tuple[torch.Tensor, float]:
    """r = 1 core of P_{>=1}[4βc² Sym₃(C⊗C)] with identity metric, without forming the n⁴ tensor.

    L T = 12 M, M = Σ_i T_iikl = 4βc²(tr C · C + 2 C²);  L² T = 24 tr M.
    P_{>=1} = a1 L + a2 R L²  (a = Σ_{r>=1} proj_coef(n, 4, r)) -> core = 12 a1 M + 24 a2 tr M · I.
    """
    n = C.shape[0]
    c2 = float(mean @ C @ mean / (mean @ mean) ** 2)
    M = 4.0 * beta * c2 * (torch.trace(C) * C + 2.0 * (C @ C))
    a = sum(proj_coef(n, 4, r) for r in (1, 2))
    core = 12.0 * float(a[1]) * M + 24.0 * float(a[2]) * torch.trace(M) * torch.eye(n, dtype=C.dtype, device=C.device)
    return core, c2


def _has_k4_block(vec_part) -> bool:
    return any(sum(v) == 4 for v in vec_part)


def factored_nonlin_kprop_k3_ablate(K_in, nonlin_wick_coef, augment=False, base=False, use_pK=True,
                                    cfg: Ablate = Ablate()):
    assert augment and not base and use_pK, "ablation is defined for factored AUGMENT only"
    step = STATS.get("step", 0)
    STATS["step"] = step + 1
    if step < cfg.k4_from and cfg.k4 != "full":
        cfg = Ablate(legs=cfg.legs, no211=cfg.no211, noK211=cfg.noK211, beta=cfg.beta, dress_s=cfg.dress_s)
    WK = K_in
    n = WK[1].n
    assert WK[1].r == 0 and WK[2].r == 0
    mean = WK[1].core
    var = WK[2].core.diag()

    cache = {}

    def get_wick_coef(k: int, p: int):
        if (k, p) not in cache:
            cache[(k, p)] = nonlin_wick_coef(mean=mean, var=var, k=k, p=p)
        return cache[(k, p)]

    # 3.1 dense pK slices (ARC lines 327-361), with the memoryless filter on d = 4 slices
    terms_iso = get_all_terms_iso(k_max=3, d_max=4, augment=True)
    terms_iso = [
        (int_part, vec_part, count)
        for int_part, vec_part_dict in terms_iso.items()
        for vec_part, count in vec_part_dict.items()
        if len(int_part) <= 3
        and int_part != (1, 1, 1)
        and (int_part, set(vec_part)) != ((2, 1, 1), {(1, 1, 1)})
        and factored_keeps_term(3, int_part, vec_part)
        and not (cfg.k4.startswith("memoryless") and sum(int_part) == 4 and _has_k4_block(vec_part))
    ]
    pK_slices = defaultdict(lambda: 0.0)
    for int_part, vec_part, count in terms_iso:
        term = eval_part(WK, vec_part, len(int_part), output_zero_repeated=True)
        if term is None:
            continue
        pK_slices[int_part] += count * multiply_wicks(
            term, check_vec_partition(vec_part, len(int_part)), p=int_part, wick_lookup=get_wick_coef)
    for int_part in pK_slices:
        pK_slices[int_part] = symmetrize(pK_slices[int_part], vec=int_part)

    # 3.2 factored (1,1,1) births (ARC lines 371-453)
    w = lambda k: get_wick_coef(k, 1)  # noqa: E731
    WK_11 = zero_repeated(WK[2].core)
    if 3 in WK:
        pK_111 = WK[3].clone()
        pK_111.contract_wick_(w(1))
    else:
        pK_111 = FactoredTensor(n=n, d=3, device=mean.device, dtype=mean.dtype)
    eye = torch.eye(n, device=mean.device, dtype=mean.dtype)
    fac2 = eye * 3
    D21 = diagslice(WK[3], (2, 1), output_zero_repeated=True) if 3 in WK else None
    K22 = diagslice(WK[4], (2, 2), output_zero_repeated=True) if 4 in WK else None

    if cfg.legs in ("aug", "simple"):
        legs = [(1, 1, WK_11, 1)]
        if cfg.legs == "aug":
            if D21 is not None:
                legs += [(2, 1, D21 / 2, 2), (1, 2, D21.T / 2, 2)]
            if K22 is not None:
                legs.append((2, 2, K22 / 4, 2))
            legs.append((2, 2, WK_11 * WK_11 / 2, 2))
        for a_l, b_l, s_l, c_l in legs:
            rights = [(a_r, b_r, s_r) for a_r, b_r, s_r, c_r in legs if c_l + c_r <= 3]
            fac1 = w(a_l)[:, None] * s_l
            fac3 = sum(w(b_l + a_r)[:, None] * w(b_r)[None, :] * s_r for a_r, b_r, s_r in rights)
            pK_111.add_factors_((fac1, fac2, fac3.T))
    else:  # dressed
        w1, w2, w3 = w(1), w(2), w(3)
        ratio = torch.where(w2 > 1e-30, w3 / w2.clamp_min(1e-30), -mean / var)  # = -μ/σ² (finite)
        heavy = 0.5 * w2[:, None] * WK_11 * WK_11  # ½ w2_a C_aj²   (hub ratio applied below)
        H = torch.zeros_like(WK_11)
        if D21 is not None:
            H = H + 0.5 * w2[:, None] * D21           # ½ w2_a D21[a,j]   (hub weight w2: no ratio)
            heavy = heavy + 0.5 * w1[:, None] * D21.T  # ½ w1_a D21[j,a]
        if K22 is not None:
            heavy = heavy + 0.25 * w2[:, None] * K22   # ¼ w2_a K22[a,j]
        H = H + heavy * ratio[None, :]
        if cfg.legs == "asym":
            # Sym is bilinear: light×light + light×heavy + heavy×light == Sym(X ⊗ e ⊗ (X + 2H)), exactly
            X = w1[:, None] * WK_11
            pK_111.add_factors_((X, fac2, ((X + 2.0 * H) * w2[None, :])))
        else:
            X = w1[:, None] * WK_11 + cfg.dress_s * H
            pK_111.add_factors_((X, fac2, (X * w2[None, :])))
            if cfg.legs == "dressed_nospur":
                pK_111.add_factors_((-H, fac2, (H * w2[None, :])))

    if 4 in WK and not cfg.no211:
        core, metric = WK[4].core, WK[4].metric
        if metric.ndim == 1:
            metric_full, metric_diag = metric.diagflat(), metric
        else:
            metric_full, metric_diag = metric, metric.diagonal()
        ones = torch.ones_like(metric_diag)
        pK_111.add_factors_((w(2)[:, None] * core.diagonal()[:, None] * ones[None, :],
                             w(1)[:, None] * metric_full, w(1)[:, None] * eye / 4))
        if metric.ndim == 2:
            pK_111.add_factors_((w(1)[:, None] * core, w(2)[:, None] * eye, w(1)[:, None] * metric_full))
        pK_111.add_factors_((w(1)[:, None] * core, w(1)[:, None] * eye,
                             w(2)[:, None] * metric_diag[:, None] * ones[None, :] / 4))

    # 4. pK -> K (ARC lines 464-467)
    pK_ds = DSTower.from_slices(pK_slices, autozero=True)
    K_ds = DS_pK_to_K(pK_ds, strict=False)
    K_ds[3] -= pK_111.get_repeated()

    # 4.1 K211 contributions (ARC lines 470-521)
    K211_contrib = 0.0
    if not cfg.noK211:
        A, B, C = pK_111.factors
        rep = list(FactoredTensor.from_dstensor(pK_111.get_repeated()).factors)
        A, B, C = torch.cat([A, -rep[0]], 1), torch.cat([B, rep[1]], 1), torch.cat([C, rep[2]], 1)
        pK1 = pK_ds[1].slices[(1,)]
        t = symmetrize(((pK1[:, None] * A).sum(0) * B) @ C.T + ((pK1[:, None] * B).sum(0) * C) @ A.T
                       + ((pK1[:, None] * C).sum(0) * A) @ B.T) / 3.0
        K211_contrib = K211_contrib + t * (-2 * 6 * 2 / (2 * n + 8))
        if 3 in WK:
            A, B, C = WK[3].factors
            rep = list(FactoredTensor.from_dstensor(WK[3].get_repeated()).factors)
            A, B, C = torch.cat([A, -rep[0]], 1), torch.cat([B, rep[1]], 1), torch.cat([C, rep[2]], 1)
            w1_, w2_ = get_wick_coef(1, 1), get_wick_coef(1, 2)
            t = symmetrize(((w2_[:, None] * A).sum(0) * w1_[:, None] * B) @ (w1_[:, None] * C).T
                           + ((w2_[:, None] * B).sum(0) * w1_[:, None] * C) @ (w1_[:, None] * A).T
                           + ((w2_[:, None] * C).sum(0) * w1_[:, None] * A) @ (w1_[:, None] * B).T) / 3
            K211_contrib = K211_contrib + t * (6 * 2 / (2 * n + 8))

    # 5. back to the factored tower
    K_out = {}
    K_out[3] = pK_111 + FactoredTensor.from_dstensor(K_ds[3])
    K_out[1] = HTensor(core=K_ds[1].to_tensor(), r=0)
    K_out[2] = HTensor(core=K_ds[2].to_tensor(), r=0)
    K_out[4] = DS_harmonic_proj(K_ds[4], r_out=1)
    K_out[4].core = K_out[4].core + K211_contrib
    if cfg.k4 in ("gain", "memoryless+gain"):
        g, c2 = gain_core(K_out[1].core, K_out[2].core, cfg.beta)
        STATS.setdefault("c2", []).append(c2)
        K_out[4].core = g if cfg.k4 == "gain" else K_out[4].core + g
    return K_out


STATS: dict = {}


def run_arm(mlp, n, depth, cfg: Ablate | None, dev, dtype):
    """cfg None -> ARC's own function (reference)."""
    STATS.clear()
    if cfg is not None:
        fk3.factored_nonlin_kprop_k3 = lambda **kw: factored_nonlin_kprop_k3_ablate(**kw, cfg=cfg)
    try:
        return m2.run_variant(mlp, n, depth, "b", dev, dtype)
    finally:
        fk3.factored_nonlin_kprop_k3 = _ORIG


def calibrate(dev) -> dict:
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    n, depth = 64, 4
    rng = np.random.default_rng(11)
    w = rng.standard_normal((depth, n, n)) * np.sqrt(2.0 / n)
    mlp = m2.build_mlp(w, dev, dtype)
    ref, _ = run_arm(mlp, n, depth, None, dev, dtype)
    a_s, _ = m2.run_variant(mlp, n, depth, "a", dev, dtype)
    scale = float(np.abs(ref - a_s).max())
    out = {"n": n, "depth": depth, "|b - a| (scale)": scale}
    for name in ("b", "dressed_nospur", "2c_asym", "2_dressed", "1_simple_legs", "3a_memoryless", "3b_gain"):
        cfg = Ablate(legs="dressed_nospur") if name == "dressed_nospur" else ARMS[name]
        got, _ = run_arm(mlp, n, depth, cfg, dev, dtype)
        out[f"|{name} - ARC b| / scale"] = float(np.abs(got - ref).max()) / scale
    # gain projection: closed form vs DS_harmonic_proj of the dense template
    X = torch.as_tensor(rng.standard_normal((n, 3 * n)), device=dev) / np.sqrt(3 * n)
    C = X @ X.T
    m = torch.as_tensor(rng.uniform(0.3, 1.0, n), device=dev)
    core, c2 = gain_core(m, C, 0.7)
    s = 4 * 0.7 * c2
    T = s * (torch.einsum("ij,kl->ijkl", C, C) + torch.einsum("ik,jl->ijkl", C, C)
             + torch.einsum("il,jk->ijkl", C, C))
    ref_core = DS_harmonic_proj(DSTensor.from_tensor(T), r_out=1).core
    out["gain_core rel err vs DS_harmonic_proj"] = float((core - ref_core).norm() / ref_core.norm())
    print(json.dumps(out, indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--nets", default="0:16")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    dev = torch.device("cuda")
    if a.calibrate:
        r = calibrate(dev)
        if a.out:
            Path(a.out).write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
        return 0
    assert a.mini
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    n, depth = 1024, 16
    mini = np.load(gen.cache_dir() / "mini16_truth.npz")
    lo, hi = (int(t) for t in a.nets.split(":"))
    arms = a.arms.split(",")
    res = {"arms": {k: asdict(ARMS[k]) for k in arms}, "nets": []}
    prog = os.environ.get("WORKERQ_PROGRESS")
    for i in range(lo, hi):
        seed = int(mini["seeds"][i])
        w = gen.mint_numpy(seed, n, depth)
        truth = np.array(mini["truth"][i])
        floor = float(mini["avg_variance"][i]) / 1e9
        mlp = m2.build_mlp(w, dev, dtype)
        rec = {"i": i, "seed": seed, "floor_final": floor, "v73_mse_final": float(mini["v73_mse"][i])}
        for k in arms:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            means, _ = run_arm(mlp, n, depth, ARMS[k], dev, dtype)
            pl = ((means - truth) ** 2).mean(1) - floor
            rec[k] = {"mse_final": float(pl[-1]), "mse_all": float(pl.mean()), "mse_layers": pl.tolist(),
                      "s": time.perf_counter() - t0, "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
            if "c2" in STATS:
                rec[k]["c2"] = STATS["c2"]
            print(json.dumps({"i": i, "arm": k, "mse_final": rec[k]["mse_final"], "s": round(rec[k]["s"], 1)}),
                  flush=True)
            torch.cuda.empty_cache()
        res["nets"].append(rec)
        if a.out:
            Path(a.out).write_text(json.dumps(res) + "\n", encoding="utf-8")
        if prog:
            Path(prog).write_text(f"{(i - lo + 1) / (hi - lo):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
