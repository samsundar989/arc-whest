"""Adapter parity: the canonical-table reference step == v32's LIVE `_layer`.

Label-free. v32's own `_layer` runs in float64 on an INJECTED boundary: the
tower's slice readers are replaced so `D3`/`D21` are exactly the supplied
arrays, `W = I` so `mu_pre`/`cov_pre` are the supplied ones, and `layer_index=0`
so the transported core is exactly the supplied `dG`. Everything the nonlinear
step computes is then compared with `ref_step.step(boundary, canon(v32))`:

    mu_out, cov_out (K2 diag + K11), g_out (K4 + rowsum K22), S3 = K3 - w1^3 D3,
    and K21 reconstructed from the compressor's input R = S21 - S_sep.

Also: the univariate FAST path and the generic loop are the same program
(`_wick_program` with `fast_1d` forced off must give identical outputs).

Run: uv run python local_sim/closure_audit/test_ref_step.py
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_LS = os.path.dirname(_HERE)
ROOT = os.path.dirname(_LS)
sys.path[:] = [q for q in sys.path if os.path.abspath(q or ".") not in (_HERE, _LS)] + [ROOT, _LS, _HERE]

import numpy as np  # noqa: E402
import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402

flops.configure(symmetry_warnings=False)

import k4feed_oracle as KO  # noqa: E402
import canon  # noqa: E402
import ref_step  # noqa: E402

RESULTS = []


def claim(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name, detail)


def fixture(n, seed, var_scale=1.0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)) / np.sqrt(n)
    cov = var_scale * (A @ A.T + 0.3 * np.diag(rng.uniform(0.5, 2.0, n)))
    mu = rng.normal(scale=0.7 * np.sqrt(var_scale), size=n)
    D3 = 0.05 * var_scale ** 1.5 * rng.normal(size=n)
    D21 = 0.02 * var_scale ** 1.5 * rng.normal(size=(n, n))
    np.fill_diagonal(D21, 0.0)
    dG = 0.01 * var_scale ** 2 * rng.uniform(0.2, 1.5, n)
    return {"mu_pre": mu, "cov_pre": cov, "D3": D3, "D21": D21, "dG": dG,
            "metric": 2.0, "lam": 0.0}


class _Sentinel:
    pass


def run_live(mod, bd, force_generic=False, keep_hubs=False, dtype=None, layer_index=0):
    dtype = fnp.float64 if dtype is None else dtype
    n = bd["mu_pre"].shape[0]
    cap = {}
    sent = _Sentinel()
    D3 = fnp.asarray(bd["D3"], dtype=dtype)
    D21 = fnp.asarray(bd["D21"], dtype=dtype)
    mod._advance = lambda WT, tw: sent
    mod._dslice3 = lambda tw: D3
    mod._dslice21 = lambda tw, off: D21 * off
    mod._wick_tower = lambda col, tw: sent

    def comp(R):
        cap["R"] = np.asarray(R)
        return R[:, :1], R[:, :1]
    mod._p2_compress = comp

    def app(tw, A_D, A_L, w2b, n_, Zb=None, Lb=None, sb=None, eb=None, *a):
        cap["sb"], cap["eb"], cap["A_L"] = np.asarray(sb), np.asarray(eb), np.asarray(A_L)
        return sent
    mod._append = app
    mod._truncate = lambda tw, g, n_: tw
    if not keep_hubs:
        mod._HUB_ROUTES = ()
    const = mod.Estimator()._constants(n, dtype)
    if force_generic:
        const["wick"] = dict(const["wick"], fast_1d=False)
    W = fnp.asarray(np.eye(n), dtype=dtype)
    st = (fnp.asarray(bd["mu_pre"], dtype=dtype), fnp.asarray(bd["cov_pre"], dtype=dtype),
          sent, 0.0, fnp.asarray(bd["dG"], dtype=dtype))
    with flops.BudgetContext(flop_budget=2 ** 55, quiet=True):
        mu_out, cov_out, _tw, _c4, g_out = mod._layer(
            W, st, mod._tables(), dtype, const, bd["metric"], last_layer=False,
            gens=None, layer_index=layer_index)
    cap.update(mu_out=np.asarray(mu_out), cov_out=np.asarray(cov_out), g_out=np.asarray(g_out))
    return cap


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def main():
    table32, _ = canon.canon_ours(os.path.join(ROOT, "estimator_k3_v32.py"))
    recs = canon.to_records(table32)
    for n, seed, vs in ((7, 1, 1.0), (9, 2, 0.15), (8, 3, 3.0)):
        bd = fixture(n, 20260917_00 + seed, vs)
        mod = KO._load(os.path.join(ROOT, "estimator_k3_v32.py"))
        live = run_live(mod, bd)
        mod2 = KO._load(os.path.join(ROOT, "estimator_k3_v32.py"))
        gen = run_live(mod2, bd, force_generic=True)
        ref = ref_step.step(bd, recs)
        K = ref["K"]
        mu, sigma = bd["mu_pre"], np.sqrt(np.diag(bd["cov_pre"]))
        w1 = ref_step.wick(1, 1, mu, sigma)
        w2 = ref_step.wick(2, 1, mu, sigma)
        off = 1.0 - np.eye(n)
        C11 = bd["cov_pre"] * off
        tag = f"n={n} var_scale={vs}"
        claim(f"{tag} mu_out", rel(live["mu_out"], ref["mu_out"]) < 1e-12, "%.1e" % rel(live["mu_out"], ref["mu_out"]))
        claim(f"{tag} cov_out", rel(live["cov_out"], ref["cov_out"]) < 1e-12, "%.1e" % rel(live["cov_out"], ref["cov_out"]))
        claim(f"{tag} g_out", rel(live["g_out"], ref["g_out"]) < 1e-12, "%.1e" % rel(live["g_out"], ref["g_out"]))
        S3 = K[(3,)] - w1 ** 3 * bd["D3"]
        claim(f"{tag} S3 = K3 - w1^3 D3", rel(live["sb"], S3) < 1e-12, "%.1e" % rel(live["sb"], S3))
        fac1 = w1[:, None] * C11
        p111_21 = (w1[:, None] ** 2 * w1[None, :]) * bd["D21"] + ((fac1 * fac1) * w2[None, :]) * off
        e_b = live["eb"]
        S_sep = (e_b[:, None] * C11) * w1[None, :]
        K21_live = live["R"] + S_sep + p111_21
        claim(f"{tag} K21 (via compressor input)", rel(K21_live, K[(2, 1)]) < 1e-12,
              "%.1e" % rel(K21_live, K[(2, 1)]))
        same = all(np.array_equal(live[k], gen[k]) for k in ("mu_out", "cov_out", "g_out", "sb", "R"))
        claim(f"{tag} fast_1d path == generic loop (bitwise)", same)
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
