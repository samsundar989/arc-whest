"""Lane M6 -- dense-κ3 one-step oracle for the κ3 × covariance-edge family at n = 1024. `.venv-kprop`.

Trajectory: ARC factored AUGMENT (m5_ablate, default arm "b" == "asym", i.e. SIMPLE births +
exact dressing). At the nonlinear step t that feeds pre-activation layer L = t + 2
(L ∈ {4, 8, 12, 16}) the factored pre-activation κ3 is materialised densely (n³, fp32)
and the dropped family is evaluated exactly (m6_terms, calibrated against ARC):

    E3   → Δκ3 (post-act t, all-distinct), transported with W_{L} to pre-act L: ΔD21, ΔD3
    E21a, E21b → Δ κ4-core (post-act t)

Then at step t + 1 (post-act layer L) the step is re-run with W Δcore Wᵀ added to the input κ4
core, which gives E21's ΔD21 one step later; E3 reaches the same point through the linear
w1⊗w1⊗w1 pass-through. Their sum is the family's ΔD21 "as a set".

Substitutes, through the same formulas:
    spike r  C inside the edge -> top-r eigenpairs of C (diag zeroed),  r ∈ {1, 4, 16, 64}
    gain     κ3 -> 2βc²(μ_i C_jk + μ_j C_ik + μ_k C_ij), β = 0.7, c² = μᵀCμ/‖μ‖⁴
capture = 1 − ‖sub − exact‖² / ‖exact‖²  (cos² = capture after the best scalar).

Proxy data: 40k random off-diagonal (a, b) per layer and net with the targets and cheap
features (ridge fitted offline, leave-one-net-out, analyze_m6.py).

    python experiments/2026-10-07_laneM2/run_kprop.py m6_oracle.py --nets 0:6 --out <abs.json>
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
from m5_ablate import ARMS, factored_nonlin_kprop_k3_ablate, fk3, gen, m2  # noqa: E402
from mlp_kprop.diagslice import zero_repeated  # noqa: E402
from mlp_kprop.harmonic import HTensor  # noqa: E402

RANKS = (1, 4, 16, 64)
BETA = 0.7
L_MEASURE = (4, 8, 12, 16)


def materialize(ft) -> torch.Tensor:
    A, B, C = ft._factors
    n, R = A.shape
    S = torch.zeros(n, n, n, device=A.device, dtype=A.dtype)
    Sv = S.view(n * n, n)
    for lo in range(0, R, 256):
        a, b, c = A[:, lo:lo + 256], B[:, lo:lo + 256], C[:, lo:lo + 256]
        Sv.addmm_((a[:, None, :] * b[None, :, :]).reshape(n * n, -1), c.T)
    out = mt.sym3_(S)
    del S
    return out


def transport(T: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    n = T.shape[0]
    T = (W @ T.reshape(n, -1)).reshape(n, n, n)
    T = (T.reshape(-1, n) @ W.T).reshape(n, n, n)
    return torch.matmul(W, T)


def d21_d3_sym(Y: torch.Tensor):
    """(2,1) slice (zero diag) and (3) slice of Sym(Y) for Y symmetric in its last two indices
    up to the unsymmetrised first index: Sym(Y)_aab = (Y_aab + Y_aba + Y_baa)/3."""
    n = Y.shape[0]
    ar = torch.arange(n, device=Y.device)
    Yaab = Y[ar, ar, :]            # [a, b]
    Yaba = Y[ar, :, ar]            # [a, b]
    Ybaa = Y[:, ar, ar].T          # [a, b] = Y_baa
    D21 = (Yaab + Yaba + Ybaa) / 3.0
    D3 = D21.diagonal().clone()
    return D21.fill_diagonal_(0.0), D3


def cap(sub, ex):
    e = float((ex * ex).sum())
    d = float(((sub - ex) ** 2).sum())
    cs = float((sub * ex).sum()) ** 2 / max(float((sub * sub).sum()) * e, 1e-300)
    return {"capture": 1.0 - d / e, "cos2": cs}


def rel(x, ref):
    return float(x.norm() / ref.norm())


class Oracle:
    def __init__(self, mlp, cfg, rng):
        self.mlp, self.cfg, self.rng = mlp, cfg, rng
        self.t = 0
        self.pending = {}     # t_E + 1 -> stash from step t_E
        self.rec = {}
        self.proxy = {}

    def step(self, K_in, nlw):
        return factored_nonlin_kprop_k3_ablate(K_in, nlw, augment=True, cfg=self.cfg)

    def __call__(self, K_in, nonlin_wick_coef, augment=False, base=False, use_pK=True):
        t = self.t
        self.t += 1
        L = t + 2
        stash = None
        if L in L_MEASURE:
            stash = self.oracle(K_in, t)
        out = self.step(K_in, nonlin_wick_coef)
        if stash is not None:
            core = out[4].core.clone().fill_diagonal_(0.0)
            r = self.rec[L]
            r["e21a_core_rel"] = rel(stash["E21a"]["exact"], core)
            r["e21b_core_rel"] = rel(stash["E21b"]["exact"], core)
            r["e21_core_rel"] = rel(stash["E21a"]["exact"] + stash["E21b"]["exact"], core)
            self.pending[t + 1] = stash
        if t in self.pending:
            self.post(K_in, nonlin_wick_coef, out, self.pending.pop(t), t)
        torch.cuda.empty_cache()
        return out

    # ---- step t_E: dense κ3 and the family's one-step outputs -------------------------------
    def oracle(self, K_in, t):
        n = K_in[1].n
        W = self.mlp.Ws[t + 1].weight
        mu, C = K_in[1].core, K_in[2].core
        var = C.diag()
        Cm = C.clone().fill_diagonal_(0.0)
        w = mt.wicks(mu, var)
        t0 = time.perf_counter()
        kap = zero_repeated(materialize(K_in[3]))
        t_mat = time.perf_counter() - t0
        ev, evec = torch.linalg.eigh(C)
        edges = {"exact": Cm}
        for r in RANKS:
            Cr = (evec[:, -r:] * ev[-r:]) @ evec[:, -r:].T
            edges[f"spike{r}"] = Cr.fill_diagonal_(0.0)
        out = {"E3": {}, "E21a": {}, "E21b": {}}
        for name, Ce in edges.items():
            T = mt.e3_raw(w, Ce, kap, mt.C3)
            out["E3"][name] = d21_d3_sym(transport(T, W))
            del T
            out["E21a"][name] = mt.e21a_core(w, Ce, kap, n, mt.SA_X)
            out["E21b"][name] = mt.e21b_core(w, Ce, kap, n, mt.SB_X)
        del kap
        torch.cuda.empty_cache()
        gk, c2 = mt.gain_kappa3(mu, C, BETA)
        gk = zero_repeated(gk)
        T = mt.e3_raw(w, Cm, gk, mt.C3)
        out["E3"]["gain"] = d21_d3_sym(transport(T, W))
        del T
        out["E21a"]["gain"] = mt.e21a_core(w, Cm, gk, n, mt.SA_X)
        out["E21b"]["gain"] = mt.e21b_core(w, Cm, gk, n, mt.SB_X)
        del gk
        closed = mt.e21b_gain_closed(w, mu, Cm, n, BETA, c2, mt.SB_X)
        rec = {"t": t, "c2": c2, "materialize_s": t_mat, "R": int(K_in[3]._factors[0].shape[1]),
               "e21b_gain_closed_rel_err": rel(closed - out["E21b"]["gain"], out["E21b"]["gain"]),
               "top_eig_share": [float(ev[-r:].sum() / ev.sum()) for r in RANKS]}
        for fam in ("E21a", "E21b"):
            ex = out[fam]["exact"]
            rec[f"{fam}_cap"] = {k: cap(v, ex) for k, v in out[fam].items() if k != "exact"}
        ex21, ex3 = out["E3"]["exact"]
        rec["E3_d21_cap"] = {k: cap(v[0], ex21) for k, v in out["E3"].items() if k != "exact"}
        rec["E3_d3_cap"] = {k: cap(v[1], ex3) for k, v in out["E3"].items() if k != "exact"}
        self.rec[t + 2] = rec
        return out

    # ---- step t_E + 1: baselines, E21 re-run, the set at post-act L --------------------------
    def post(self, K_in, nlw, out_base, stash, t):
        L = t + 1
        rec = self.rec[L]
        W = self.mlp.Ws[t].weight
        D21_pre = K_in[3].get_dslice((2, 1)).clone()
        D3_pre = K_in[3].get_dslice((3,)).clone()
        dE3_21, dE3_3 = stash["E3"]["exact"]
        rec["e3_d21_rel"] = rel(dE3_21, D21_pre)
        rec["e3_d3_rel"] = rel(dE3_3, D3_pre)
        rec["e3_d3_rel_to_d21"] = rel(dE3_3, D21_pre)
        w1 = mt.wicks(K_in[1].core, K_in[2].core.diag())[(1, 1)]
        lin = (w1 * w1)[:, None] * w1[None, :]
        D21_post = out_base[3].get_dslice((2, 1)).clone()
        m_post = out_base[1].core.clone()

        def rerun(dcore):
            K = dict(K_in)
            K[4] = HTensor(core=K_in[4].core + W @ dcore @ W.T, r=1, metric=K_in[4].metric)
            o = self.step(K, nlw)
            return o[3].get_dslice((2, 1)) - D21_post, o[1].core - m_post

        post = {}
        for key in ("exact", "gain", "spike16"):
            d21a, dma = rerun(stash["E21a"][key])
            d21b, dmb = rerun(stash["E21b"][key])
            e3 = lin * stash["E3"][key][0]
            post[key] = {"E3": e3, "E21a": d21a, "E21b": d21b, "set": e3 + d21a + d21b,
                         "dm_E21b": dmb, "dm_E21": dma + dmb}
        ex = post["exact"]
        rec["post_rel"] = {k: rel(v, D21_post) for k, v in ex.items() if not k.startswith("dm")}
        rec["post_dmean_rel_E21"] = rel(ex["dm_E21"], m_post)
        for key in ("gain", "spike16"):
            rec[f"post_set_cap_{key}"] = cap(post[key]["set"], ex["set"])
            rec[f"post_E21b_cap_{key}"] = cap(post[key]["E21b"], ex["E21b"])
        # proxy sample: features at pre-act L (state the chain already has)
        n = D21_pre.shape[0]
        C = K_in[2].core
        w2 = mt.wicks(K_in[1].core, C.diag())[(2, 1)]
        ev, evec = torch.linalg.eigh(C)
        P4 = evec[:, -4:] @ evec[:, -4:].T
        DC = D21_pre * C.clone().fill_diagonal_(0.0)
        feats = torch.stack([
            D21_pre, D21_pre.T, DC, DC.T,
            DC.sum(1)[:, None].expand(n, n), DC.sum(0)[None, :].expand(n, n),
            w2[:, None] * D21_pre, w2[None, :] * D21_pre,
            P4 @ D21_pre, D21_pre @ P4, P4 @ D21_pre @ P4, C,
        ], -1)
        idx = torch.as_tensor(self.rng.choice(n * n, 40000, replace=False), device=C.device)
        idx = idx[(idx // n) != (idx % n)]
        self.proxy[L] = {
            "X": feats.reshape(n * n, -1)[idx].double().cpu().numpy(),
            "y_e3_pre": dE3_21.reshape(-1)[idx].double().cpu().numpy(),
            "y_set_post": ex["set"].reshape(-1)[idx].double().cpu().numpy(),
            "y_e21b_post": ex["E21b"].reshape(-1)[idx].double().cpu().numpy(),
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", default="0:6")
    ap.add_argument("--arm", default="b")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dev = torch.device("cuda")
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    torch.set_grad_enabled(False)
    n, depth = 1024, 16
    mini = np.load(gen.cache_dir() / "mini16_truth.npz")
    lo, hi = (int(x) for x in a.nets.split(":"))
    res = {"arm": a.arm, "beta": BETA, "ranks": RANKS, "nets": []}
    outp = Path(a.out)
    prog = os.environ.get("WORKERQ_PROGRESS")
    proxy_all = {}
    for i in range(lo, hi):
        seed = int(mini["seeds"][i])
        mlp = m2.build_mlp(gen.mint_numpy(seed, n, depth), dev, dtype)
        orc = Oracle(mlp, ARMS[a.arm], np.random.default_rng(1000 + i))
        fk3.factored_nonlin_kprop_k3 = orc
        t0 = time.perf_counter()
        try:
            means, _ = m2.run_variant(mlp, n, depth, "b", dev, dtype)
        finally:
            fk3.factored_nonlin_kprop_k3 = m5._ORIG
        truth = np.array(mini["truth"][i])
        floor = float(mini["avg_variance"][i]) / 1e9
        res["nets"].append({"i": i, "seed": seed, "s": time.perf_counter() - t0,
                            "mse_final": float(((means[-1] - truth[-1]) ** 2).mean() - floor),
                            "layers": {str(k): v for k, v in orc.rec.items()}})
        for L, d in orc.proxy.items():
            for k, v in d.items():
                proxy_all[f"n{i}_L{L}_{k}"] = v
        print(json.dumps({"i": i, "s": round(time.perf_counter() - t0, 1),
                          "e3_d21_rel": {L: round(r.get("e3_d21_rel", -1), 4) for L, r in orc.rec.items()},
                          "post_set": {L: round(r["post_rel"]["set"], 4) for L, r in orc.rec.items()
                                       if "post_rel" in r}}), flush=True)
        outp.write_text(json.dumps(res) + "\n", encoding="utf-8")
        np.savez_compressed(outp.with_suffix(".proxy.npz"), **proxy_all)
        if prog:
            Path(prog).write_text(f"{(i - lo + 1) / (hi - lo):.3f}")
        del mlp, orc
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
