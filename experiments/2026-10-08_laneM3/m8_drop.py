"""Lane M3 / M8 -- is the κ4 core a memory of old κ3?  `.venv-kprop`.

Two chains, each with its old κ3 blocks dropped after A layers:

    bE   ARC factored AUGMENT + E21b gain-mode into the κ4 core (Lane M2 `m6_e2e` arm E21b_gain)
    a    ARC factored SIMPLE (control: no off-diagonal κ4 core)

Every κ3 factor column carries the birth step τ of its block. At the input of nonlinear step t
(pre-activation, right after transport) columns with τ < t − A are removed by building a new
FactoredTensor from the kept columns (the transport already cleared the slice cache).

Correction blocks. ARC's step ends with K3 = pK_111 + from_dstensor(K_ds[3]), where
K_ds[3] = (dense repeated slices born at t) − pK_111.get_repeated(). from_dstensor is linear and
its legs 2 and 3 are I, so the correction's first leg splits exactly by tag:

    D_total = D_new + Σ_g D_g,   D_g = −[rep_g(3,)·I + 3·rep_g(2,1)ᵀ],  rep_g = repeated slices
                                                                        of pK_111's columns in group g

D_g carries the tag of the block it corrects, so corrections die with their block (no
"−Rep ghosts"). Groups with the same drop step are merged (all of them when A = ∞, which is then
ARC's own single correction block).

    --calibrate   n = 64, depth 6, float64: (1) split-all A = ∞ == unmodified chain, both chains;
                  (2) at every step each old tag group (its columns + its corrections) has zero
                  repeated-index slices; (3) A = 3 runs.
    --nets 0:16   the 16 organizer DEV nets at n = 1024, organizer labels, floor-subtracted
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
M2DIR = HERE.parent / "2026-10-07_laneM2"
sys.path.append(str(M2DIR))
import m5_ablate as m5  # noqa: E402
import m6_e2e as m6  # noqa: E402
from m5_ablate import ARMS, fk3, gen, m2  # noqa: E402

FactoredTensor = fk3.FactoredTensor
INF = 10**9


def rep_leg(cols):
    """First leg of from_dstensor(-get_repeated()) for the factored tensor with these columns."""
    n = cols[0].shape[0]
    rep = FactoredTensor(n=n, d=3, factors=cols).get_repeated()
    eye = torch.eye(n, device=cols[0].device, dtype=cols[0].dtype)
    return -(rep.slices[(3,)][:, None] * eye + rep.slices[(2, 1)].T * 3)


class Dropper:
    """Wraps one chain's nonlinear step; keeps the per-column tag array of K[3]."""

    def __init__(self, chain: str, A: int, depth: int, mlp, split_all: bool = False, check: bool = False):
        self.chain, self.A, self.depth, self.mlp = chain, A, depth, mlp
        self.split_all, self.check = split_all, check
        self.t = 0
        self.tags = None          # np.int64 array, one entry per column of the current K[3]
        self.log = []             # per step: columns in / kept / out
        self.max_rep = 0.0        # calibration check (2)

    def drop_step(self, tau: int) -> int:
        """First step whose input no longer keeps tag tau."""
        if self.A >= INF:
            return INF
        d = tau + self.A + 1
        return d if d <= self.depth - 1 else INF

    def inner(self, K_in, nonlin_wick_coef):
        if self.chain == "a":
            return m5._ORIG(K_in=K_in, nonlin_wick_coef=nonlin_wick_coef, augment=False)
        out = m5.factored_nonlin_kprop_k3_ablate(K_in, nonlin_wick_coef, augment=True, cfg=ARMS["b"])
        if 3 in K_in:   # E21b gain-mode (m6_e2e arm "E21b_gain"): reads μ and C only
            out[4].core = out[4].core + m6.e21_cores(K_in, "gain", "E21b")
        return out

    def __call__(self, K_in, nonlin_wick_coef, augment=False, base=False, use_pK=True):
        t = self.t
        self.t += 1
        r_in = 0
        if 3 in K_in:
            F = K_in[3]._factors
            assert F[0].shape[1] == len(self.tags), (F[0].shape, len(self.tags))
            r_in = len(self.tags)
            keep = self.tags >= t - self.A if self.A < INF else np.ones(r_in, bool)
            if not keep.all():
                idx = torch.as_tensor(np.nonzero(keep)[0], device=F[0].device)
                K_in = dict(K_in)
                K_in[3] = FactoredTensor(n=K_in[3].n, d=3, factors=tuple(f[:, idx] for f in F))
            old_tags = self.tags[keep]
        else:
            old_tags = np.zeros(0, np.int64)
        out = self.inner(K_in, nonlin_wick_coef)
        F = out[3]._factors
        n = out[3].n
        R = F[0].shape[1]
        r_old = len(old_tags)
        n_birth = R - r_old - n
        assert n_birth >= 0
        if self.check:
            eye = torch.eye(n, device=F[0].device, dtype=F[0].dtype)
            assert torch.equal(F[1][:, -n:], eye) and torch.equal(F[2][:, -n:], eye)
        # group the old columns by drop step (split_all: one group per tag)
        key = (lambda tau: tau) if self.split_all else self.drop_step
        D_total = F[0][:, -n:]
        groups = {}
        for tau in np.unique(old_tags):
            groups.setdefault(key(int(tau)), []).append(int(tau))
        new_key = key(t)
        pieces, piece_tags = [], []
        D_rest = D_total
        for gk, taus in sorted(groups.items()):
            if gk == new_key:
                continue      # same drop step as this layer's births: stays in the new piece
            idx = torch.as_tensor(np.nonzero(np.isin(old_tags, taus))[0], device=F[0].device)
            Dg = rep_leg(tuple(f[:, idx] for f in F))
            pieces.append(Dg)
            piece_tags.append(min(taus))
            D_rest = D_rest - Dg
        eye = torch.eye(n, device=F[0].device, dtype=F[0].dtype)
        legs1 = [F[0][:, :R - n], D_rest] + pieces
        legs23 = [[F[k][:, :R - n]] + [eye] * (1 + len(pieces)) for k in (1, 2)]
        # old columns merged into the new piece (same drop step) keep their own tags: same drop step
        tags = np.concatenate([old_tags, np.full(n_birth + n, t)] + [np.full(n, pt) for pt in piece_tags])
        out[3] = FactoredTensor(n=n, d=3, factors=(torch.cat(legs1, 1), torch.cat(legs23[0], 1),
                                                     torch.cat(legs23[1], 1)))
        self.tags = tags.astype(np.int64)
        self.log.append((t, r_in, r_old, int(len(tags))))
        if self.check and pieces:
            self._check_groups(out[3], t, key, R - n)
        return out

    def _check_groups(self, K3, t, key, r_pre):
        """Each old drop group (columns + their corrections) must have zero repeated slices at birth."""
        F = K3._factors
        keys = np.array([key(int(x)) for x in self.tags])
        for gk in np.unique(keys):
            if gk == key(t):
                continue
            sel = np.nonzero(keys == gk)[0]
            idx = torch.as_tensor(sel, device=F[0].device)
            pre = torch.as_tensor(sel[sel < r_pre], device=F[0].device)   # without this step's correction
            rep = FactoredTensor(n=K3.n, d=3, factors=tuple(f[:, idx] for f in F)).get_repeated()
            rep0 = FactoredTensor(n=K3.n, d=3, factors=tuple(f[:, pre] for f in F)).get_repeated()
            for p, s in rep.slices.items():
                self.max_rep = max(self.max_rep, float(s.abs().max()) / max(float(rep0.slices[p].abs().max()), 1e-300))


def run(mlp, n, depth, chain, A, dev, dtype, **kw):
    d = Dropper(chain, A, depth, mlp, **kw)
    fk3.factored_nonlin_kprop_k3 = d
    try:
        means, dt = m2.run_variant(mlp, n, depth, "a" if chain == "a" else "b", dev, dtype)
    finally:
        fk3.factored_nonlin_kprop_k3 = m5._ORIG
    return means, dt, d


def reference(mlp, n, depth, chain, dev, dtype):
    if chain == "a":
        return m2.run_variant(mlp, n, depth, "a", dev, dtype)[0]
    return m6.run(mlp, n, depth, "E21b_gain", dev, dtype)[0]


def calibrate(dev):
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    n, depth = 64, 6
    rng = np.random.default_rng(5)
    mlp = m2.build_mlp(rng.standard_normal((depth, n, n)) * np.sqrt(2.0 / n), dev, dtype)
    out = {"n": n, "depth": depth}
    for chain in ("bE", "a"):
        ref = reference(mlp, n, depth, chain, dev, dtype)
        other = reference(mlp, n, depth, "a" if chain == "bE" else "bE", dev, dtype)
        scale = float(np.abs(ref - other).max())
        for label, kw, A in (("splitall_inf", dict(split_all=True, check=True), INF),
                             ("merged_inf", dict(check=True), INF),
                             ("A3", dict(check=True), 3), ("A1", dict(check=True), 1)):
            got, _, d = run(mlp, n, depth, chain, A, dev, dtype, **kw)
            out[f"{chain} {label}: max|got-ref| / |a-bE|"] = float(np.abs(got - ref).max()) / scale
            out[f"{chain} {label}: max rel repeated of old groups"] = d.max_rep
            out[f"{chain} {label}: columns (t, in, kept, out)"] = d.log
    print(json.dumps(out, indent=1))
    (HERE / "m8_calibrate.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--nets", default="0:16")
    ap.add_argument("--chains", default="bE,a")
    ap.add_argument("--A", default="inf,8,5,3,2,1")
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
    As = [INF if x == "inf" else int(x) for x in a.A.split(",")]
    res = {"chains": a.chains.split(","), "A": [("inf" if x >= INF else x) for x in As], "nets": []}
    prog = os.environ.get("WORKERQ_PROGRESS")
    for i in range(lo, hi):
        seed = int(mini["seeds"][i])
        mlp = m2.build_mlp(gen.mint_numpy(seed, n, depth), dev, dtype)
        truth, floor = np.array(mini["truth"][i]), float(mini["avg_variance"][i]) / 1e9
        rec = {"i": i, "seed": seed, "floor_final": floor}
        for chain in res["chains"]:
            for A in As:
                torch.cuda.reset_peak_memory_stats()
                means, dt, d = run(mlp, n, depth, chain, A, dev, dtype)
                pl = ((means - truth) ** 2).mean(1) - floor
                arm = f"{chain}_A{'inf' if A >= INF else A}"
                rec[arm] = {"mse_final": float(pl[-1]), "mse_layers": pl.tolist(), "s": dt,
                            "cols_final": d.log[-1][3], "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
                print(json.dumps({"i": i, "arm": arm, "mse_final": rec[arm]["mse_final"], "s": round(dt, 1),
                                  "cols": d.log[-1][3]}), flush=True)
                torch.cuda.empty_cache()
        res["nets"].append(rec)
        if a.out:
            Path(a.out).write_text(json.dumps(res) + "\n", encoding="utf-8")
        if prog:
            Path(prog).write_text(f"{(i - lo + 1) / (hi - lo):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
