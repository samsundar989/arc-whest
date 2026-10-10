"""Check tools/twin/k4loop.py against ARC's factored_nonlin_kprop_k3 (AUGMENT, scalar metric).

    .venv-kprop/Scripts/python.exe experiments/2026-10-08_p3/k4loop_check.py

Random pre-activation state at n = 24, float64: mean, covariance, a factored kappa3, an r = 1
kappa4 core with metric m*1. Runs ARC's own nonlinear step (patched only to also return its pK /
K slices and pK_111), then rebuilds the OFF-diagonal output core from k4loop's formulas and the
same ingredients. Also checks k4loop.relu_wicks against ARC's relu_wick_coef.
"""
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "external" / "mlp_cumulant_propagation" / "src"))
sys.path.append(str(ROOT / "tools" / "twin"))

import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)

import mlp_kprop.factor_k3 as fk3  # noqa: E402
from mlp_kprop.factor_k3 import FactoredTensor  # noqa: E402
from mlp_kprop.harmonic import HTensor  # noqa: E402
from mlp_kprop.diagslice import diagslice, zero_repeated  # noqa: E402
from mlp_kprop.wick import relu_wick_coef  # noqa: E402
import k4loop as kl  # noqa: E402

src = inspect.getsource(fk3.factored_nonlin_kprop_k3)
src = src.replace("def factored_nonlin_kprop_k3(", "def _cap(")
tail = "            K_out[4].core += K211_contrib\n"
assert src.count(tail) == 1
src = src.rstrip()
assert src.endswith("return K_out")
src = src[: -len("return K_out")] + "return K_out, pK_ds, DS_pK_to_K(pK_ds, strict=not augment), pK_111\n"
ns = dict(fk3.__dict__)
exec(src, ns)
cap = ns["_cap"]


def main(n=24, seed=0, m=2.0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 2 * n, generator=g) / (2 * n) ** 0.5
    C = X @ X.T + 0.3 * torch.eye(n)
    mu = 0.4 * torch.randn(n, generator=g)
    R = 2 * n
    facs = tuple(0.15 * torch.randn(n, R, generator=g) for _ in range(3))
    K3 = FactoredTensor(n, 3, facs)
    core = torch.randn(n, n, generator=g) * 0.02
    core = 0.5 * (core + core.T) + torch.diag(0.05 + 0.02 * torch.rand(n, generator=g))
    K4 = HTensor(core=core, r=1, metric=m * torch.ones(n))
    K_in = {1: HTensor(mu, r=0), 2: HTensor(C, r=0), 3: K3, 4: K4}
    var = C.diagonal()
    wl = lambda mean, var, k, p: relu_wick_coef(mean=mean, var=var, k=k, p=p)
    out, pK_ds, K_full, pK_111 = cap(K_in, wl, augment=True)

    # wicks
    sig = var.sqrt()
    w = kl.relu_wicks(mu, sig, kl.WKEYS)
    werr = max(float((w[k] - relu_wick_coef(mean=mu, var=var, k=k[0], p=k[1])).abs().max()
                     / relu_wick_coef(mean=mu, var=var, k=k[0], p=k[1]).abs().max()) for k in kl.WKEYS)
    print(f"wicks max rel err {werr:.2e}")

    Cm = kl.zero_diag(C)
    D21 = K3.get_dslice((2, 1))
    D3 = K3.get_dslice((3,))
    D22 = zero_repeated(diagslice(K4, (2, 2)))
    D4 = diagslice(K4, (4,))
    Off = kl.zero_diag(core)
    pre = dict(Cm=Cm, D21=D21, D22=D22, Off=Off, D3=D3, D4=D4)
    pK1 = pK_ds[1].slices[(1,)]
    K2s = K_full[2]
    post = dict(pK1=pK1, K2=K2s.slices[(2,)], K11=kl.zero_diag(K2s.slices[(1, 1)]),
                K21=kl.zero_diag(K_full[3].slices[(2, 1)]))
    # kappa3 contractions, dense (n^3)
    T_pre = zero_repeated(K3.to_tensor())
    e = w[(1, 2)] - 2.0 * pK1 * w[(1, 1)]
    k3e = torch.einsum("k,kab->ab", e, T_pre) * (w[(1, 1)][:, None] * w[(1, 1)][None, :])
    T_post = zero_repeated(pK_111.to_tensor())
    w1 = w[(1, 1)]
    T_new = T_post - T_pre * w1[:, None, None] * w1[None, :, None] * w1[None, None, :]
    k3n = torch.einsum("k,kab->ab", pK1, T_new)
    mine, _ = kl.core_off(n, w, m, pre, post, k3_pre_e=k3e, k3_post_pk1=k3n)
    ref = kl.zero_diag(out[4].core)
    rel = float((mine - ref).norm() / ref.norm())
    print(f"core_off rel err vs ARC {rel:.3e}   |ref| {float(ref.norm()):.3e}")
    # ablations: how big is each part relative to the reference
    parts = {}
    for name in ("no_k3", "no_K11K11", "no_K31"):
        pass
    return rel


if __name__ == "__main__":
    r = [main(seed=s) for s in range(3)]
    print("PASS" if max(r) < 1e-10 else "FAIL", max(r))
