"""Lane M6 -- the κ3 × covariance-edge family (E3, E21a, E21b) as direct dense formulas. `.venv-kprop`.

All three read the PRE-activation state of one nonlinear step: mean μ, covariance C (used
off-diagonal, Cm), and κ3 restricted to its all-distinct entries (κ = zero_repeated(κ3)).
Wick coefficients w(k, p) = E_{N(μ,σ²)}[∂^k relu^p].

    E3   ΔK3 (post-act, all-distinct)  = c3 · Sym_ijk[ w11_i w21_j w21_k Cm_jk κ_ijk ]
    E21a Δcore_jk (κ4 r=1 core)        = sa · w21_j w21_k Cm_jk Σ_i w12_i κ_ijk
    E21b Δcore_jk                       = sb · sym_jk[ w11_j w21_k Σ_i w22_i Cm_ik κ_ijk ]

(diagonals of the cores zeroed). c3, sa, sb are fixed against ARC's own unfactored term
evaluation (eval_part + multiply_wicks + symmetrize + DS_harmonic_proj) by `--calibrate`.

Each formula takes the C inside the edge and the κ3 separately, so the cheap substitutes
of M6 (rank-r spike C, gain-mode κ3) go through the same code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from mlp_kprop.diagslice import DSTensor, eval_part, zero_repeated
from mlp_kprop.harmonic import HTensor, DS_harmonic_proj
from mlp_kprop.kprop_harmonic import get_all_terms_iso, multiply_wicks, get_d_max, AUGMENT
from mlp_kprop.partitions import check_vec_partition
from mlp_kprop.tensor_utils import symmetrize
from mlp_kprop.wick import relu_wick_coef

TERMS = {
    "E3": ((1, 1, 1), ((0, 1, 1), (1, 1, 1))),
    "E21a": ((2, 1, 1), ((0, 1, 1), (1, 1, 1))),
    "E21b": ((2, 1, 1), ((1, 0, 1), (1, 1, 1))),
}
# fixed by --calibrate (m6_calibrate.json); n-dependence of the cores is 1/(2n+8)
C3 = None
SA_X = None   # sa = SA_X / (2n + 8)
SB_X = None   # sb = SB_X / (2n + 8)
_CAL = Path(__file__).resolve().parent / "m6_calibrate.json"
if _CAL.exists():
    _c = json.loads(_CAL.read_text())
    C3, SA_X, SB_X = _c["c3"], _c["sa_x(2n+8)"], _c["sb_x(2n+8)"]


def wicks(mu, var):
    return {(k, p): relu_wick_coef(mean=mu, var=var, k=k, p=p) for k, p in ((1, 1), (2, 1), (1, 2), (2, 2))}


def sym3_(T: torch.Tensor) -> torch.Tensor:
    """Full symmetrisation over 3 indices with one extra n³ buffer."""
    S = T.clone()
    for p in ((0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        S.add_(T.permute(*p))
    return S.div_(6.0)


def e3_raw(w, Cm, kap, c3=1.0):
    """Unsymmetrised T_ijk = c3 w11_i w21_j w21_k Cm_jk κ_ijk (symmetric in j,k). Sym(T) is ΔK3."""
    T = kap * Cm[None, :, :]
    T.mul_((c3 * w[(1, 1)])[:, None, None]).mul_(w[(2, 1)][None, :, None]).mul_(w[(2, 1)][None, None, :])
    return T


def e21a_core(w, Cm, kap, n, sa_x=1.0):
    X = torch.einsum("i,ijk->jk", w[(1, 2)], kap)
    B = (sa_x / (2 * n + 8)) * w[(2, 1)][:, None] * w[(2, 1)][None, :] * Cm * X
    return B.fill_diagonal_(0.0)


def e21b_core(w, Cm, kap, n, sb_x=1.0):
    # X_jk = Σ_i w22_i Cm_ik κ_ijk  (n³: contract i with the vector-weighted column k)
    X = torch.einsum("ik,ijk->jk", w[(2, 2)][:, None] * Cm, kap)
    B = w[(1, 1)][:, None] * X * w[(2, 1)][None, :]
    B = (sb_x / (2 * n + 8)) * 0.5 * (B + B.T)
    return B.fill_diagonal_(0.0)


def gain_kappa3(mu, C, beta, c2=None):
    """2βc²·(μ_i C_jk + μ_j C_ik + μ_k C_ij), c² = μᵀCμ/‖μ‖⁴ (R2 §1.5)."""
    if c2 is None:
        c2 = float(mu @ C @ mu / (mu @ mu) ** 2)
    T = mu[:, None, None] * C[None, :, :]
    T = T + T.permute(1, 0, 2) + T.permute(1, 2, 0)
    return (2 * beta * c2) * T, c2


def e21b_gain_closed(w, mu, Cm, n, beta, c2, sb_x):
    """E21b with κ3 -> gain template, all-distinct i,j,k, in O(n²) + one n³ product (Cm diag(w22) Cm)."""
    w22 = w[(2, 2)]
    v = Cm @ (w22 * mu)                       # v_k = Σ_i Cm_ki w22_i μ_i
    q = (Cm * Cm).T @ w22                     # q_k = Σ_i w22_i Cm_ik²
    P = Cm @ (w22[:, None] * Cm)              # P_kj = Σ_i Cm_ki w22_i Cm_ij
    X = Cm * (v[None, :] - (w22 * mu)[:, None] * Cm)                 # i ≠ j exclusion: Cm_jk(v_k − w22_j μ_j Cm_jk)
    X = X + mu[:, None] * (q[None, :] - w22[:, None] * Cm * Cm)      # μ_j(q_k − w22_j Cm_jk²)
    X = X + mu[None, :] * P.T                                        # μ_k P_kj   (i = j term vanishes: Cm_jj = 0)
    X = (2 * beta * c2) * X
    B = w[(1, 1)][:, None] * X * w[(2, 1)][None, :]
    B = (sb_x / (2 * n + 8)) * 0.5 * (B + B.T)
    return B.fill_diagonal_(0.0)


def _arc_term(name, mu, C, K3):
    """ARC's own evaluation of one dropped term: symmetrised ΔpK slice (E3) or Δcore (E21*)."""
    n = mu.shape[0]
    var = C.diag()
    K = {1: HTensor(mu, r=0), 2: HTensor(C, r=0), 3: K3}
    iso = get_all_terms_iso(3, d_max=get_d_max(3, AUGMENT), augment=True)
    ip, vp = TERMS[name]
    count = iso[ip][vp]
    term = eval_part(K, vp, len(ip), output_zero_repeated=True)
    out = count * multiply_wicks(term, check_vec_partition(vp, len(ip)), p=ip,
                                 wick_lookup=lambda k, p: relu_wick_coef(mean=mu, var=var, k=k, p=p))
    out = symmetrize(out, vec=ip)
    if name == "E3":
        return out
    return DS_harmonic_proj(DSTensor({ip: out}, autozero=True), r_out=1).core


def calibrate(dev) -> dict:
    torch.set_default_dtype(torch.float64)
    n = 32
    g = torch.Generator(device="cpu").manual_seed(5)
    X = torch.randn(n, 2 * n, generator=g) / np.sqrt(2 * n)
    C = (X @ X.T).to(dev)
    mu = (0.3 * torch.randn(n, generator=g)).to(dev)
    A = torch.randn(n, 3 * n, generator=g).to(dev) * 0.05
    K3 = sym3_(torch.einsum("ir,jr,kr->ijk", A, A.roll(1, 1), A.roll(2, 1)))
    kap = zero_repeated(K3)
    Cm = C.clone().fill_diagonal_(0.0)
    w = wicks(mu, C.diag())
    out = {"n": n}
    ref = _arc_term("E3", mu, C, K3)
    mine = sym3_(e3_raw(w, Cm, kap))
    out["c3"] = float((ref * mine).sum() / (mine * mine).sum())
    out["c3_rel_resid"] = float((ref - out["c3"] * mine).norm() / ref.norm())
    for name, fn, key in (("E21a", e21a_core, "sa_x(2n+8)"), ("E21b", e21b_core, "sb_x(2n+8)")):
        ref = _arc_term(name, mu, C, K3)
        mine = fn(w, Cm, kap, n)
        s = float((ref * mine).sum() / (mine * mine).sum())
        out[key] = s
        out[f"{name}_rel_resid"] = float((ref - s * mine).norm() / ref.norm())
    # gain closed form vs dense template through the dense E21b formula
    gk, c2 = gain_kappa3(mu, C, 0.7)
    dense = e21b_core(w, Cm, zero_repeated(gk), n, out["sb_x(2n+8)"])
    closed = e21b_gain_closed(w, mu, Cm, n, 0.7, c2, out["sb_x(2n+8)"])
    out["e21b_gain_closed_rel_err"] = float((dense - closed).norm() / dense.norm())
    print(json.dumps(out, indent=1))
    _CAL.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    if "--calibrate" in sys.argv:
        calibrate(torch.device("cuda"))
