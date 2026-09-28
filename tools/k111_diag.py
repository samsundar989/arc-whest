"""The all-distinct (K111) third-cumulant content AS THE NEXT LAYER CONSUMES IT: model vs labels.

    .venv-gpu/Scripts/python.exe tools/k111_diag.py step0 --out <dir>          numpy identities (CPU)
    .venv-gpu/Scripts/python.exe tools/k111_diag.py run --nets 0-3 --out <dir> twin, GPU (workerq)
    .venv-gpu/Scripts/python.exe tools/k111_diag.py report --out <dir>         tables from the JSONs

Research audit 2026-09-27 (docs/research/2026-09-27_architecture_audit_codex.md "Recommended next
diagnostic"; literature audit §4A). Convention: next pre z' = W^T a, W[a, j] = (current a, next j).
T = post-ReLU third cumulant at layer l, t[a] = T_aaa, S[a, b] = T_aab (first index repeated,
diagonal zeroed). Then, exactly (verified by `step0` on a dense symmetric tensor):

    D3_next[j]     = sum_abc W_aj W_bj W_cj T_abc
    rep3[j]        = sum_a W_aj^3 t_a + 3 sum_a W_aj^2 (S W)_aj
    D21_next[i, j] = sum_abc W_ai W_bi W_cj T_abc            (first output index twice)
    rep21          = (W*W)^T (t*W) + (W*W)^T (S W) + 2 (W*(S W))^T W
    AD             = next - rep                               (all-distinct projection)

LABELS: t = full1000 post K3 (N = 1e9), S = bench-supp s00 post_K21 (N = 1e8), D3/D21 next =
cache/twin_labels (full1000, N = 1e9). NOISE: nets 0..7 carry an independent rep1 re-bake of
post_K21; AD(s00) - AD(rep1) measures the K21 part of the label noise; the full1000 part is
an estimate from the bench README noise tables (N = 1e8 figures / sqrt 10).

MODEL: the twin (candidate v62_s1 + S3 parameters = v67/v68 accuracy). At layer l the chain's
outgoing tower `state[2]` IS the model's post T. Its own repeated slices are read by advancing
it with the identity (t_tw, S_tw); its next-layer D3/D21 by advancing with W_{l+1}^T (exactly
what the chain does at l+1). So AD_model = D3_next - rep(t_tw, S_tw, W) is the true all-distinct
projection of the model tensor, and rep(t_tw, S_tw) - rep(K3, K21) is the repeated-slice
mismatch left by the rank-4 P2 residual compression (and hubs), reported separately.
The tower is split, by capturing `_append`, into INHERITED (the wick-scaled, joined incoming
tower) + NEWBORN (two-leg block, hubs, D-block) -- a decomposition of the model's own tensor,
NOT a separation of new-birth error from inherited error in the truth.

Modes: `free` (the chain as it runs) and `exact` (every pre object substituted by its label at
every layer, as closure_bench.py; the model D3/D21 at l+1 are read BEFORE substitution).
Downstream arms (free mode): at one consumer layer L (or all), replace D3/D21 by
  AD   rep_tw + AD_lab        (only the all-distinct part corrected)
  REP  D - rep_tw + rep_lab   (only the repeated part corrected)
  CMP  D - rep_tw + rep_K     (only the compression mismatch removed)
  FULL D_lab
and read the mean MSE (vs the aicrowd truth) at L, L+1 and the final layer.
Span test: least-squares Grams of (AD_lab - AD_model) against the AD response to per-source-
neuron gates f(alpha, gamma3) on the newborn two-leg columns (w2b), the newborn hub columns
(cA, cB), and the inherited wick row scale (linearised by a central difference); plus a
next-neuron regression (AD_lab on AD_model x features of the next neuron).

Accuracy instrument only: no FLOP is ever quoted from here.
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

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "cache" / "hf" / "whest-p2-bakev2-bench-supp"
FULL = ROOT / "cache" / "hf" / "arc-whestbench-p2-full1000-N1e9" / "moments"
FEATS = ("one", "alpha", "alpha2", "gamma3", "Phi")
#: bench_supp README "MC noise floors", N = 1e8: pre kappa3 noise RMS per neuron and the
#: relative noise of the pre K21 sketch, per layer 0..15. At N = 1e9 divide by sqrt(10).
PRE_K3_NOISE_1E8 = [6.9e-04, 4.0e-04, 2.5e-04, 1.8e-04, 1.3e-04, 9.7e-05, 7.5e-05, 6.1e-05,
                    5.0e-05, 4.2e-05, 3.5e-05, 3.0e-05, 2.6e-05, 2.3e-05, 2.0e-05, 1.8e-05]
PRE_K21_REL_1E8 = [0.999, 0.033, 0.022, 0.016, 0.013, 0.010, 0.008, 0.007, 0.006, 0.006,
                   0.005, 0.005, 0.004, 0.004, 0.004, 0.003]


# ============================================================ the identities
def rep3(t, S, W):
    SW = S @ W
    return (W ** 3).T @ t + 3.0 * ((W * W) * SW).sum(0)


def rep21(t, S, W):
    SW = S @ W
    W2 = W * W
    return W2.T @ (t[:, None] * W) + W2.T @ SW + 2.0 * (W * SW).T @ W


def cmd_step0(a):
    """Dense float64 check of both identities on a random symmetric tensor."""
    out = {}
    rng = np.random.default_rng(20260927)
    for n, m in ((6, 6), (7, 5), (8, 9)):
        X = rng.standard_normal((n, n, n))
        T = sum(X.transpose(p) for p in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0),
                                          (2, 0, 1), (2, 1, 0))) / 6.0
        W = rng.standard_normal((n, m))
        idx = np.arange(n)
        t = T[idx, idx, idx]
        S = T[idx[:, None], idx[:, None], idx[None, :]].copy()     # S[a,b] = T[a,a,b]
        np.fill_diagonal(S, 0.0)
        a_, b_, c_ = np.meshgrid(idx, idx, idx, indexing="ij")
        dist = (a_ != b_) & (a_ != c_) & (b_ != c_)
        Tad = T * dist
        D3 = np.einsum("abc,aj,bj,cj->j", T, W, W, W)
        D3ad = np.einsum("abc,aj,bj,cj->j", Tad, W, W, W)
        D21 = np.einsum("abc,ai,bi,cj->ij", T, W, W, W)
        D21ad = np.einsum("abc,ai,bi,cj->ij", Tad, W, W, W)
        e3 = np.abs(D3 - rep3(t, S, W) - D3ad).max() / np.abs(D3).max()
        e21 = np.abs(D21 - rep21(t, S, W) - D21ad).max() / np.abs(D21).max()
        # wrong orientation / multiplicity must FAIL, or the check proves nothing
        e21_T = np.abs(D21 - rep21(t, S.T, W) - D21ad).max() / np.abs(D21).max()
        e3_T = np.abs(D3 - rep3(t, S.T, W) - D3ad).max() / np.abs(D3).max()
        e3_2 = np.abs(D3 - (rep3(t, S, W) - ((W * W) * (S @ W)).sum(0)) - D3ad).max() / np.abs(D3).max()
        Sd = S.copy()
        np.fill_diagonal(Sd, t)                                     # diagonal NOT zeroed
        e3_d = np.abs(D3 - rep3(t, Sd, W) - D3ad).max() / np.abs(D3).max()
        # the chain convention: D21[i,j] = kappa(z_i, z_i, z_j) is the (i,i,j) slice of W.T
        Tn = np.einsum("abc,ai,bj,ck->ijk", T, W, W, W)
        conv = np.abs(D21 - np.einsum("iij->ij", Tn)).max()
        out[f"n{n}_m{m}"] = {"D3_identity_maxrel": e3, "D21_identity_maxrel": e21,
                             "D3_with_S_transposed": e3_T, "D21_with_S_transposed": e21_T,
                             "D3_with_multiplicity_2": e3_2, "D3_with_S_diag_not_zeroed": e3_d,
                             "D21_is_kappa_iij_of_next": conv}
    ok = all(v["D3_identity_maxrel"] < 1e-12 and v["D21_identity_maxrel"] < 1e-12
             and v["D21_with_S_transposed"] > 1e-3 and v["D3_with_multiplicity_2"] > 1e-3
             and v["D3_with_S_diag_not_zeroed"] > 1e-3 for v in out.values())
    out["identities_ok"] = bool(ok)
    # ---- DATA orientation checks (one FIT net, float64, CPU) --------------------------
    g = 0
    lab = np.load(ROOT / "cache" / "twin_labels" / f"fit_{g:03d}.npz")
    z = np.load(FULL / f"mlp_{g:05d}.npz")
    wfile = ROOT / "cache" / "twin" / "full_w.f32"
    per = 16 * 1024 * 1024
    Wall = np.fromfile(wfile, dtype=np.float32, count=per, offset=g * per * 4).reshape(16, 1024, 1024)
    data = {}
    for l in (2, 8, 13):
        W = Wall[l + 1].astype(np.float64)
        am = np.asarray(z["mean"][l], np.float64)
        C = np.asarray(z["M11"][l], np.float64) - am[:, None] * am[None, :]
        v = np.asarray(lab["var"][l + 1], np.float64)
        vW = np.einsum("aj,ab,bj->j", W, C, W)
        vWT = np.einsum("ja,ab,jb->j", W, C, W)
        # K21 orientation: bench post_K21 vs the full1000 raw moments cannot be compared (no
        # post M21 in full1000), so read the bench's own README convention off its sketches:
        # post_K21_O = K21 @ Omega must equal the dense post_K21 @ Omega untransposed.
        b = np.load(BENCH / "s00" / f"net_{g:05d}" / f"layer_{l:02d}.npz")
        K21 = np.asarray(b["post_K21"], np.float64)
        Om = np.asarray(b["Omega"], np.float64)
        o = np.asarray(b["post_K21_O"], np.float64)
        oT = np.asarray(b["post_K21T_O"], np.float64)
        data[f"l{l}"] = {
            "var_next_rel_err_W": float(np.linalg.norm(vW - v) / np.linalg.norm(v)),
            "var_next_rel_err_WT": float(np.linalg.norm(vWT - v) / np.linalg.norm(v)),
            "K21_O_vs_dense_untransposed": float(np.linalg.norm(K21 @ Om - o) / np.linalg.norm(o)),
            "K21_O_vs_dense_transposed": float(np.linalg.norm(K21.T @ Om - o) / np.linalg.norm(o)),
            "K21T_O_vs_denseT": float(np.linalg.norm(K21.T @ Om - oT) / np.linalg.norm(oT)),
            "K21_diag_over_postK3": float(np.linalg.norm(np.diag(K21)) / np.linalg.norm(
                np.asarray(z["post_m3"][l]) - 3 * am * np.asarray(z["post_m2"][l]) + 2 * am ** 3)),
        }
    out["data_orientation_net0"] = data
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "step0.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0 if ok else 1


# ============================================================ the twin side
def _twin():
    sys.path.insert(0, str(ROOT / "tools" / "twin"))     # the torch shim, this process only
    import twin as T  # noqa: E402
    return T


class Hook:
    """`_A2`: record pre-substitution values, optionally substitute (exact mode) and/or
    replace D3/D21 (downstream arms). Records only what is asked for."""

    def __init__(self, sub=None, record=True):
        self.sub = sub                  # T.Subst or None
        self.record = record
        self.rec = {}
        self.edit = None                # fn(l, name, value) -> value, after substitution

    def __call__(self, l, name, value, *aux):
        if name == "K":
            if self.record and (3,) in value and (2, 1) in value:
                self.rec[(l, "K3")] = value[(3,)].detach()
                self.rec[(l, "K21")] = value[(2, 1)].detach()
            return value
        if self.record and name in ("D3", "D21"):
            self.rec[(l, name)] = value.detach()
        out = value
        if self.sub is not None:
            if l < 1:
                if name == "mu":
                    out = self.sub.t["mu"][0].to(value.dtype)
            else:
                out = self.sub(l, name, value, *aux)
        if self.edit is not None and name in ("D3", "D21"):
            out = self.edit(l, name, out)
        if self.record and name in ("mu", "var", "D3"):
            self.rec[(l, name + "_used")] = out.detach()
        return out


class Diag:
    def __init__(self, a):
        T = self.T = _twin()
        import torch
        self.torch = torch
        self.tw = T.Twin(Path(a.cand))
        self.tw.install(T.load_params(a.params))
        self.mod = self.tw.mod
        self.data = T.Data()
        self.n, self.depth = self.tw.width, self.tw.depth
        self.dev = T.DEV
        self.off64 = 1.0 - torch.eye(self.n, dtype=torch.float64, device=self.dev)
        self.I32 = torch.eye(self.n, dtype=torch.float32, device=self.dev)
        self.h = a.h
        self.max_layer = a.max_layer          # debug only: stop the chain early
        self.gains = json.loads(Path(a.gain_file).read_text()) if a.gain_file else None
        # capture `_append`'s arguments (the inherited tower + the newborn pieces)
        self.cap = {}
        orig = self.mod._append

        def _append(tw, A_D, A_L, w2b, n, Zb=None, Lb=None, sb=None, eb=None,
                    Y=None, CA=None, CB=None):
            if self.capture:
                self.cap[self.mod._CUR_LAYER] = dict(inh=tw, A_L=A_L, w2b=w2b, Zb=Zb, Lb=Lb,
                                                     sb=sb, eb=eb, Y=Y, CA=CA, CB=CB)
            return orig(tw, A_D, A_L, w2b, n, Zb, Lb, sb, eb, Y, CA, CB)

        self.mod._append = _append
        self.capture = False

    # ---------------------------------------------------------------- tower helpers
    def tcopy(self, tw):
        o = object.__new__(type(tw))
        for s in type(tw).__slots__:
            setattr(o, s, getattr(tw, s, None))
        o._p2 = o._hub = o._legs = o._p3 = None
        return o

    def adv_full(self, WT, tw):
        """Transport an ALREADY-ADVANCED tower (M covers every lineage) by WT."""
        m = self.mod._mm
        o = self.tcopy(tw)
        o.A_L = m(WT, tw.A_L)
        o.M = m(WT, tw.M)
        if tw.A_D is not None:
            o.A_D = m(WT, tw.A_D)
        if tw.Z is not None:
            o.Z = m(WT, tw.Z)
        if tw.Y is not None:
            o.Y = m(WT, tw.Y)
        if tw.Q is not None:
            o.Q = m(WT, tw.Q)
        if tw.F is not None:
            o.F = m(WT, tw.F)
        return o

    def slices(self, tw, off32):
        """(3,) and off-diagonal (2,1) slices of a tower whose M is complete, float64."""
        c = self.tcopy(tw)
        return (self.mod._dslice3(c).double(), self.mod._dslice21(c, off32).double())

    def state_slices(self, st_tw, WT, off32):
        """Own slices (identity advance) and next-layer slices of an appended state tower."""
        s_own = self.slices(self.mod._advance(self.I32, st_tw), off32)
        s_next = self.slices(self.mod._advance(WT, st_tw), off32)
        return s_own, s_next

    def newborn(self, c, which, f=None):
        """A one-lineage tower holding only part of the block born at this layer."""
        torch, mod = self.torch, self.mod
        n = self.n
        one = torch.ones(n, dtype=torch.float32, device=self.dev) if f is None else f.float()
        zero = torch.zeros(n, dtype=torch.float32, device=self.dev)
        Zb = c["Zb"]
        keepD = which in ("all", "dblock")
        w2b = c["w2b"] * one if which in ("all", "twoleg") else zero
        sb = c["sb"] if keepD else zero
        eb = c["eb"] if keepD else zero
        Z = Zb if keepD else Zb * 0.0
        hub = which in ("all", "hub") and c["Y"] is not None
        tw = mod._append(None, None, c["A_L"], w2b, n, Z, c["Lb"], sb, eb,
                         c["Y"] if hub else None,
                         (c["CA"] * one) if hub else None, (c["CB"] * one) if hub else None)
        return tw

    # ---------------------------------------------------------------- labels
    def labels(self, g):
        torch = self.torch
        dev = self.dev
        lab = np.load(ROOT / "cache" / "twin_labels" / f"fit_{g:03d}.npz")
        L = {k: torch.from_numpy(np.asarray(lab[k], np.float64)).to(dev)
             for k in ("mu", "var", "D3")}
        L["D21"] = torch.from_numpy(np.asarray(lab["D21"])).to(dev)      # float32 (16,n,n)
        z = np.load(FULL / f"mlp_{g:05d}.npz")
        a = np.asarray(z["mean"], np.float64)
        m2, m3 = np.asarray(z["post_m2"], np.float64), np.asarray(z["post_m3"], np.float64)
        L["t"] = torch.from_numpy(m3 - 3 * a * m2 + 2 * a ** 3).to(dev)
        L["S"], L["S_rep1"] = {}, {}
        for l in range(self.depth - 1):
            L["S"][l] = self._k21(BENCH / "s00" / f"net_{g:05d}" / f"layer_{l:02d}.npz")
            p = BENCH / "rep1" / "s00" / f"net_{g:05d}" / f"layer_{l:02d}.npz"
            if p.exists():
                L["S_rep1"][l] = self._k21(p)
        return L

    def _k21(self, p):
        S = self.torch.from_numpy(np.asarray(np.load(p)["post_K21"], np.float64)).to(self.dev)
        return S * self.off64

    def label_parts(self, L, l, W):
        """Label rep and AD of the next-layer D3 / D21 for source layer l."""
        off = self.off64
        t, S = L["t"][l], L["S"][l]
        D3n = L["D3"][l + 1]
        D21n = L["D21"][l + 1].double() * off
        r3, r21 = rep3(t, S, W), rep21(t, S, W) * off
        out = {"D3": D3n, "D21": D21n, "rep3": r3, "rep21": r21,
               "AD3": D3n - r3, "AD21": D21n - r21}
        if l in L["S_rep1"]:
            S1 = L["S_rep1"][l]
            out["AD3_rep1"] = D3n - rep3(t, S1, W)
            out["AD21_rep1"] = D21n - rep21(t, S1, W) * off
        return out

    # ---------------------------------------------------------------- one layer's analysis
    def analyse(self, l, st_tw, K3, K21, W, lp, feats_src, feats_next):
        torch = self.torch
        off, n = self.off64, self.n
        off32 = off.float()
        WT = W.T.contiguous()
        Wd = W.double()
        (t_tw, S_tw), (D3m, D21m) = self.state_slices(st_tw, WT, off32)
        S_tw = S_tw * off
        D21m = D21m * off
        r3_tw, r21_tw = rep3(t_tw, S_tw, Wd), rep21(t_tw, S_tw, Wd) * off
        K21d = K21.double() * off
        r3_K, r21_K = rep3(K3.double(), K21d, Wd), rep21(K3.double(), K21d, Wd) * off
        AD3m, AD21m = D3m - r3_tw, D21m - r21_tw
        nrm = lambda x: float(x.norm())
        res = {}
        for tag, Dm, rtw, rK, ADm, key in (("D3", D3m, r3_tw, r3_K, AD3m, "3"),
                                            ("D21", D21m, r21_tw, r21_K, AD21m, "21")):
            Dl, rl, ADl = lp[tag], lp["rep" + key], lp["AD" + key]
            e, eAD, eR = Dm - Dl, ADm - ADl, rtw - rl
            ee = float((e * e).sum())
            r = {"n_next_lab": nrm(Dl), "n_rep_lab": nrm(rl), "n_AD_lab": nrm(ADl),
                 "n_next_mod": nrm(Dm), "n_rep_mod": nrm(rtw), "n_AD_mod": nrm(ADm),
                 "relerr_next": nrm(e) / nrm(Dl), "relerr_AD": nrm(eAD) / nrm(ADl),
                 "relerr_rep": nrm(eR) / nrm(rl),
                 "corr_AD": float(torch.corrcoef(torch.stack([ADm.flatten(), ADl.flatten()]))[0, 1]),
                 "gain_AD": float((ADm * ADl).sum() / (ADm * ADm).sum()),
                 "err_norm_AD_over_err": nrm(eAD) / math.sqrt(ee),
                 "err_norm_rep_over_err": nrm(eR) / math.sqrt(ee),
                 "err_share_AD_proj": float((e * eAD).sum()) / ee,
                 # repeated-slice mismatch of the model's own tower vs its closure K
                 "cmp_mismatch_over_err": nrm(rtw - rK) / math.sqrt(ee),
                 "cmp_mismatch_over_AD_lab": nrm(rtw - rK) / nrm(ADl),
                 "repK_err_over_err": nrm(rK - rl) / math.sqrt(ee)}
            if ("AD" + key + "_rep1") in lp:
                nz = nrm(ADl - lp["AD" + key + "_rep1"]) / math.sqrt(2.0)
                r["noise_K21_abs"] = nz
                r["noise_K21_rel_AD"] = nz / nrm(ADl)
            # full1000 (N = 1e9) part of the label noise, from the README tables
            L1 = l + 1
            if tag == "D3":
                nzF = PRE_K3_NOISE_1E8[L1] / math.sqrt(10.0) * math.sqrt(n)
            else:
                nzF = PRE_K21_REL_1E8[L1] / math.sqrt(10.0) * nrm(Dl)
            r["noise_full1000_est_abs"] = nzF
            r["noise_full1000_est_rel_AD"] = nzF / nrm(ADl)
            res[tag] = r
        self._pending_D3 = D3m
        res["_vec"] = {"AD3_lab": lp["AD3"].float().cpu().numpy(),
                       "AD3_mod": AD3m.float().cpu().numpy(),
                       "D3_mod": D3m.float().cpu().numpy(), "D3_lab": lp["D3"].float().cpu().numpy()}
        # ---- decomposition + span test (needs the `_append` capture of layer l)
        c = self.cap.pop(l, None)
        if c is not None:
            res["span"] = self.span(l, c, W, WT, off32, AD3m, AD21m, lp, feats_src, feats_next)
        return res, (r3_tw, r21_tw, r3_K, r21_K)

    def ad_of(self, tw, WT, W, off32, adv_state=False):
        """(AD3, AD21) of a tower: next slices minus rep of its own slices."""
        off = self.off64
        Wd = W.double()
        if adv_state:
            (t, S), (D3, D21) = self.state_slices(tw, WT, off32)
        else:
            t, S = self.slices(tw, off32)
            D3, D21 = self.slices(self.adv_full(WT, tw), off32)
        S = S * off
        return D3 - rep3(t, S, Wd), (D21 - rep21(t, S, Wd)) * off

    def span(self, l, c, W, WT, off32, AD3m, AD21m, lp, fs, fn):
        cap, self.capture = self.capture, False      # `newborn` calls _append itself
        try:
            return self._span(l, c, W, WT, off32, AD3m, AD21m, lp, fs, fn)
        finally:
            self.capture = cap

    def _span(self, l, c, W, WT, off32, AD3m, AD21m, lp, fs, fn):
        torch = self.torch
        out = {}
        # decomposition checks: inherited + newborn(all) == model; D-block AD == 0
        nb3, nb21 = self.ad_of(self.newborn(c, "all"), WT, W, off32, adv_state=True)
        d3, d21 = self.ad_of(self.newborn(c, "dblock"), WT, W, off32, adv_state=True)
        if c["inh"] is not None:
            i3, i21 = self.ad_of(c["inh"], WT, W, off32)
        else:
            i3, i21 = torch.zeros_like(nb3), torch.zeros_like(nb21)
        out["check_additivity_D3"] = float((i3 + nb3 - AD3m).norm() / AD3m.norm())
        out["check_additivity_D21"] = float((i21 + nb21 - AD21m).norm() / AD21m.norm())
        out["check_dblock_AD_zero_D3"] = float(d3.norm() / AD3m.norm())
        out["check_dblock_AD_zero_D21"] = float(d21.norm() / AD21m.norm())
        out["share_inh_D3"] = float(i3.norm() / AD3m.norm())
        out["share_new_D3"] = float(nb3.norm() / AD3m.norm())
        out["share_inh_D21"] = float(i21.norm() / AD21m.norm())
        out["share_new_D21"] = float(nb21.norm() / AD21m.norm())
        B3, B21, names = self.gate_basis(c, W, WT, off32, fs)
        out["names"] = names
        y3, y21 = lp["AD3"] - AD3m, lp["AD21"] - AD21m
        X3 = torch.stack(B3, 1)                          # (n, p)
        X21 = torch.stack([b.flatten() for b in B21], 1)  # (n^2, p)
        out["gram"] = {"D3": {"XtX": (X3.T @ X3).cpu().tolist(), "Xty": (X3.T @ y3).cpu().tolist(),
                              "yty": float(y3 @ y3), "ADlab2": float((lp["AD3"] ** 2).sum())},
                       "D21": {"XtX": (X21.T @ X21).cpu().tolist(),
                               "Xty": (X21.T @ y21.flatten()).cpu().tolist(),
                               "yty": float((y21 * y21).sum()),
                               "ADlab2": float((lp["AD21"] ** 2).sum())}}
        del X21
        return self._next_reg(out, AD3m, AD21m, lp, fn)

    def gate_basis(self, c, W, WT, off32, fs):
        """AD response to per-source-neuron gates f on the newborn two-leg / hub columns and
        (linearised) the inherited wick row scale."""
        B3, B21, names = [], [], []
        for fam in ("twoleg", "hub"):
            if fam == "hub" and c["Y"] is None:
                continue
            for k, f in zip(FEATS, fs):
                b3, b21 = self.ad_of(self.newborn(c, fam, f), WT, W, off32, adv_state=True)
                B3.append(b3)
                B21.append(b21)
                names.append(f"{fam}:{k}")
        if c["inh"] is not None:
            for k, f in zip(FEATS, fs):
                col_p = (1.0 + self.h * f).float()[:, None]
                col_m = (1.0 - self.h * f).float()[:, None]
                p3, p21 = self.ad_of(self.mod._wick_tower(col_p, self.tcopy(c["inh"])), WT, W, off32)
                m3, m21 = self.ad_of(self.mod._wick_tower(col_m, self.tcopy(c["inh"])), WT, W, off32)
                B3.append((p3 - m3) / (2 * self.h))
                B21.append((p21 - m21) / (2 * self.h))
                names.append(f"inh:{k}")
        return B3, B21, names

    def _next_reg(self, out, AD3m, AD21m, lp, fn):
        torch = self.torch
        # ---- next-neuron regression: AD_lab ~ AD_mod x features of the NEXT neuron
        Xn3 = torch.stack([AD3m * f for f in fn], 1)
        Xn21 = torch.stack([(AD21m * fn[k][:, None]).flatten() for k in range(1, 5)]
                           + [(AD21m * fn[k][None, :]).flatten() for k in range(1, 5)]
                           + [AD21m.flatten()], 1)
        out["next_gram"] = {
            "D3": {"XtX": (Xn3.T @ Xn3).cpu().tolist(), "Xty": (Xn3.T @ lp["AD3"]).cpu().tolist(),
                   "yty": float((lp["AD3"] ** 2).sum())},
            "D21": {"XtX": (Xn21.T @ Xn21).cpu().tolist(),
                    "Xty": (Xn21.T @ lp["AD21"].flatten()).cpu().tolist(),
                    "yty": float((lp["AD21"] ** 2).sum())}}
        return out

    # ---------------------------------------------------------------- runs
    def feats(self, mu, var, D3):
        torch = self.torch
        sig = var.double().clamp_min(1e-30).sqrt()
        al = (mu.double() / sig).clamp(-float(self.mod.ALPHA_CAP), float(self.mod.ALPHA_CAP))
        g3 = D3.double() / sig ** 3
        return [torch.ones_like(al), al, al * al, g3, torch.special.ndtr(al)]

    def chain_run(self, w, hook, analyse=None, start=None, cache_states=False):
        """Drive the chain layer by layer. `analyse(l, state)` after each non-final layer."""
        torch = self.torch
        mod = self.mod
        mod._A2 = hook
        if start is None:
            state = (self.T.fnp.zeros(self.n), self.T.fnp.eye(self.n), None, 0.0,
                     self.T.fnp.zeros(self.n))
            l0 = 0
        else:
            l0, state = start
        mus, states = {}, {}
        with torch.no_grad():
            for l in range(l0, min(self.depth, self.max_layer)):
                state = self.tw._step(l, w[l], state)
                mus[l] = torch.clamp_min(state[0], 0.0).double()
                if cache_states:
                    states[l] = state
                if analyse is not None and l < self.depth - 1:
                    analyse(l, state)
        mod._A2 = None
        return mus, states

    def mse(self, mus, truth):
        t = self.torch.from_numpy(np.asarray(truth, np.float64)).to(self.dev)
        return {l: float(((m - t[l]) ** 2).mean()) for l, m in mus.items()}

    def run_net(self, g, a):
        torch = self.torch
        t0 = time.time()
        w, truth = self.data.net("full", g)
        L = self.labels(g)
        lps = {l: self.label_parts(L, l, w[l + 1].double()) for l in range(self.depth - 1)}
        res = {"net": g, "modes": {}}
        vecs = {}
        self._base_K = {}
        for mode in a.modes.split(","):
            sub = self.T.Subst(g, ("mu", "var", "C11", "D3", "D21", "dG", "D22", "Q31n")) \
                if mode == "exact" else None
            hook = Hook(sub=sub)
            rows = {}
            prev = [None]
            self.capture = True
            self.cap = {}

            def analyse(l, state, hook=hook, rows=rows, mode=mode, prev=prev):
                rec = hook.rec
                fs = self.feats(rec[(l, "mu_used")], rec[(l, "var_used")], rec[(l, "D3_used")]) \
                    if l >= 1 else self.feats(rec[(l, "mu_used")], rec[(l, "var_used")],
                                              torch.zeros_like(rec[(l, "var_used")]))
                fn = self.feats(L["mu"][l + 1], L["var"][l + 1], L["D3"][l + 1])
                r, reps = self.analyse(l, state[2], rec[(l, "K3")], rec[(l, "K21")], w[l + 1],
                                       lps[l], fs, fn)
                vecs.update({f"{mode}__{k}__{l}": v for k, v in r.pop("_vec").items()})
                # the chain's own pre-substitution D3 at layer l == our transport from l-1
                if l >= 1 and (l - 1) in rows:
                    rows[l - 1]["check_tap_D3"] = float(
                        (rec[(l, "D3")].double() - prev[0]).norm() / prev[0].norm())
                prev[0] = self._pending_D3
                # K21 SEMANTIC orientation: the closure's K[(2,1)] (T_aab by construction of
                # p111_21) against bench post_K21 as loaded and transposed
                Km = rec[(l, "K21")].double() * self.off64
                Sl = L["S"][l]
                r["K21_vs_lab_untransposed"] = float((Km - Sl).norm() / Sl.norm())
                r["K21_vs_lab_transposed"] = float((Km.T - Sl).norm() / Sl.norm())
                rows[l] = r
                if mode == "free":
                    self._base_K[l] = (rec[(l, "K3")], rec[(l, "K21")])
                # keep memory flat: the recorded pre-substitution D21 of old layers
                for key in [k for k in rec if k[0] < l]:
                    del rec[key]

            mus, states = self.chain_run(w, hook, analyse, cache_states=(mode == "free"))
            self.capture = False
            self.cap = {}
            res["modes"][mode] = {"layers": rows, "mse": self.mse(mus, truth)}
            if mode == "free":
                base_states = states
            print(f"  net {g} mode {mode}: {time.time() - t0:.0f}s", flush=True)
        if "free" in a.modes.split(",") and not a.no_arms:
            res["arms"] = self.arms(g, w, truth, L, lps, base_states, res["modes"]["free"]["mse"],
                                    a)
        np.savez_compressed(Path(a.out) / f"vec_{g:03d}.npz", **vecs)
        res["seconds"] = round(time.time() - t0, 1)
        return res

    def arms(self, g, w, truth, L, lps, base_states, base_mse, a):
        """Downstream: replace parts of D3/D21 at consumer layer(s) L, free-running chain."""
        torch = self.torch
        off = self.off64
        off32 = off.float()
        ctx = {}

        def prepare(Lc, state, K3, K21, gate=None):
            W = w[Lc]
            Wd = W.double()
            (t_tw, S_tw) = self.slices(self.mod._advance(self.I32, state[2]), off32)
            S_tw = S_tw * off
            K21d = K21.double() * off
            lp = lps[Lc - 1]
            ctx[Lc] = {"r3_tw": rep3(t_tw, S_tw, Wd), "r21_tw": rep21(t_tw, S_tw, Wd) * off,
                       "r3_K": rep3(K3.double(), K21d, Wd),
                       "r21_K": rep21(K3.double(), K21d, Wd) * off, "lp": lp}
            if gate is not None:                  # (capture c, source feats fs)
                c, fs = gate
                WT = W.T.contiguous()
                cap, self.capture = self.capture, False
                try:
                    (t2, S2), (D3m, D21m) = self.state_slices(state[2], WT, off32)
                    S2 = S2 * off
                    a3 = D3m - rep3(t2, S2, Wd)
                    a21 = (D21m * off - rep21(t2, S2, Wd)) * off
                    B3, B21, _ = self.gate_basis(c, W, WT, off32, fs)
                finally:
                    self.capture = cap
                n3, n21 = float((lp["AD3"] ** 2).sum()), float((lp["AD21"] ** 2).sum())
                X3 = torch.stack(B3, 1)
                X21 = torch.stack([b.flatten() for b in B21], 1)
                A = X3.T @ X3 / n3 + X21.T @ X21 / n21
                b = X3.T @ (lp["AD3"] - a3) / n3 + X21.T @ (lp["AD21"] - a21).flatten() / n21
                th = torch.from_numpy(_solve(A.cpu().numpy(), b.cpu().numpy())).to(A.device)
                ctx[Lc]["gate3"] = X3 @ th
                ctx[Lc]["gate21"] = ((X21 @ th).reshape(self.n, self.n)) * off
                del X21

        def make_edit(arm, layers):
            def edit(l, name, v):
                if l not in layers or l not in ctx:
                    return v
                c = ctx[l]
                lp = c["lp"]
                k = "3" if name == "D3" else "21"
                vd = v.double()
                if arm == "AD" or (arm == "AD_D3only" and k == "3") or (arm == "AD_D21only" and k == "21"):
                    x = c["r" + k + "_tw"] + lp["AD" + k]
                elif arm in ("AD_D3only", "AD_D21only"):
                    return v
                elif arm == "ADSCALE":            # oracle per-layer scalar on the model's AD
                    adm = vd - c["r" + k + "_tw"]
                    gg = float((adm * lp["AD" + k]).sum() / (adm * adm).sum())
                    x = c["r" + k + "_tw"] + gg * adm
                elif arm == "AD_orth":            # only the error part orthogonal to model AD
                    adm = vd - c["r" + k + "_tw"]
                    e = lp["AD" + k] - adm
                    e = e - float((e * adm).sum() / (adm * adm).sum()) * adm
                    x = c["r" + k + "_tw"] + adm + e
                elif arm == "ADGATE":             # oracle per-net gate fit (15 responses, joint)
                    x = vd + c["gate" + k]
                elif arm == "ADSCALE_fix":        # a FIXED per-layer gain profile (--gain-file)
                    adm = vd - c["r" + k + "_tw"]
                    x = c["r" + k + "_tw"] + self.gains["D" + k][l - 1] * adm
                elif arm == "AD_rep1":
                    x = c["r" + k + "_tw"] + lp["AD" + k + "_rep1"]
                elif arm == "REP":
                    x = vd - c["r" + k + "_tw"] + lp["rep" + k]
                elif arm == "CMP":
                    x = vd - c["r" + k + "_tw"] + c["r" + k + "_K"]
                elif arm == "FULL":
                    x = lp["D" + k]
                else:
                    raise ValueError(arm)
                if k == "21":
                    x = x * off
                return x.to(v.dtype)
            return edit

        out = {"single": {}, "all": {}}
        arms = ["AD", "REP", "CMP", "FULL"]
        has_rep1 = all("AD3_rep1" in lps[l] for l in range(self.depth - 1))
        # ALL layers: the prepare step must run on THIS run's state before each layer
        extra = (["AD_D3only", "AD_D21only", "ADSCALE", "AD_orth", "ADGATE"]
                 + (["ADSCALE_fix"] if self.gains else []))
        for arm in arms + extra + (["AD_rep1"] if has_rep1 else []):
            hook = Hook(record=(arm == "ADGATE"))
            krec = {}
            self.capture = arm == "ADGATE"
            self.cap = {}

            def khook(l, name, value, *aux, _h=hook, _k=krec):
                if name == "K" and (3,) in value and (2, 1) in value:
                    _k[l] = (value[(3,)].detach(), value[(2, 1)].detach())
                    return value
                return _h(l, name, value, *aux)

            layers = set(range(1, self.depth))
            hook.edit = make_edit(arm, layers)
            ctx.clear()

            def analyse(l, state, _k=krec, _h=hook, _arm=arm):
                gate = None
                if _arm == "ADGATE":
                    rec = _h.rec
                    fs = self.feats(rec[(l, "mu_used")], rec[(l, "var_used")], rec[(l, "D3_used")])
                    gate = (self.cap.pop(l), fs)
                    for key in [q for q in rec if q[0] < l]:
                        del rec[key]
                prepare(l + 1, state, *_k[l], gate=gate)

            mus, _ = self.chain_run(w, khook, analyse)
            self.capture = False
            self.cap = {}
            m = self.mse(mus, truth)
            out["all"][arm] = {"final": m[self.depth - 1], "rows": m}
        # SINGLE consumer layer Lc, restarted from the cached base state after Lc-1
        for arm in arms:
            out["single"][arm] = {}
            for Lc in range(1, self.depth):
                st = base_states[Lc - 1]
                # K of layer Lc-1 in the base run: recompute cheaply from the base tap
                ctx.clear()
                K3, K21 = self._base_K[Lc - 1]
                prepare(Lc, st, K3, K21)
                hook = Hook(record=False)
                hook.edit = make_edit(arm, {Lc})
                mus, _ = self.chain_run(w, hook, start=(Lc, st))
                m = self.mse(mus, truth)
                r = {"L": m[Lc] / base_mse[Lc] - 1.0, "final": m[self.depth - 1] / base_mse[self.depth - 1] - 1.0}
                if Lc + 1 < self.depth:
                    r["L+1"] = m[Lc + 1] / base_mse[Lc + 1] - 1.0
                out["single"][arm][Lc] = r
        out["base_final"] = base_mse[self.depth - 1]
        return out


def cmd_run(a):
    import torch
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = Diag(a)
    lo, hi = (int(x) for x in a.nets.split("-"))
    progress = os.environ.get("WORKERQ_PROGRESS")
    nets = list(range(lo, hi + 1))
    for i, g in enumerate(nets):
        p = out / f"net_{g:03d}.json"
        if p.exists():
            continue
        torch.cuda.reset_peak_memory_stats()
        r = d.run_net(g, a)
        r["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        p.write_text(json.dumps(r))
        print(f"net {g}: {r['seconds']}s peak VRAM {r['peak_vram_gib']:.2f} GiB", flush=True)
        if progress:
            try:
                Path(progress).write_text(str((i + 1) / len(nets)))
            except OSError:
                pass
    return 0


# ============================================================ report
def _solve(A, b, rcond=1e-8):
    """Normal equations, column-normalised, truncated-SVD solve (the gate features
    1, alpha, alpha^2, Phi(alpha) are nearly collinear)."""
    A, b = np.asarray(A, np.float64), np.asarray(b, np.float64)
    d = np.sqrt(np.maximum(np.diag(A), 1e-300))
    u = np.linalg.lstsq(A / d[:, None] / d[None, :], b / d, rcond=rcond)[0]
    return u / d


def _fit(XtX, Xty, yty, idx):
    A = np.asarray(XtX)[np.ix_(idx, idx)]
    b = np.asarray(Xty)[idx]
    th = _solve(A, b)
    return max(yty - 2 * th @ b + th @ A @ th, 0.0), th


def _pooled_lono(grams, idx):
    """Leave-one-net-out: fit on the others' summed Grams, residual on the held-out net."""
    out = []
    for i, g in enumerate(grams):
        A = sum(np.asarray(h["XtX"]) for j, h in enumerate(grams) if j != i)
        b = sum(np.asarray(h["Xty"]) for j, h in enumerate(grams) if j != i)
        A, b = A[np.ix_(idx, idx)], b[idx]
        th = _solve(A, b)
        Ai, bi = np.asarray(g["XtX"])[np.ix_(idx, idx)], np.asarray(g["Xty"])[idx]
        out.append(max(g["yty"] - 2 * th @ bi + th @ Ai @ th, 0.0))
    return out


def cmd_report(a):
    out = Path(a.out)
    nets = [json.loads(p.read_text()) for p in sorted(out.glob("net_*.json"))]
    if not nets:
        print("no nets")
        return 1
    med = lambda xs: float(np.median([x for x in xs if x is not None and np.isfinite(x)])) if xs else float("nan")
    lines = [f"# K111 diagnostic report ({len(nets)} nets: {[n['net'] for n in nets]})", ""]
    summ = {"nets": [n["net"] for n in nets]}
    for mode in ("free", "exact"):
        if mode not in nets[0]["modes"]:
            continue
        lines += [f"## mode `{mode}` -- medians over nets; l = source post layer, consumer l+1", ""]
        for tag in ("D3", "D21"):
            lines += [f"### {tag}", "",
                      "| l | ‖next‖ lab | ‖rep‖ lab | ‖AD‖ lab | ‖AD‖ mod | relerr next | relerr AD | "
                      "noise AD (K21) | noise AD (f1000 est) | corr AD | gain lab/mod | "
                      "‖eAD‖/‖e‖ | ‖eRep‖/‖e‖ | AD proj share | cmp/‖e‖ | ‖repK−rep lab‖/‖e‖ |",
                      "|" + "---|" * 16]
            for l in range(15):
                rs = [n["modes"][mode]["layers"][str(l)][tag] for n in nets]
                g = lambda k: med([r.get(k) for r in rs])
                lines.append(
                    f"| {l} | {g('n_next_lab'):.3e} | {g('n_rep_lab'):.3e} | {g('n_AD_lab'):.3e} | "
                    f"{g('n_AD_mod'):.3e} | {g('relerr_next'):.2e} | {g('relerr_AD'):.2e} | "
                    f"{g('noise_K21_rel_AD'):.2e} | {g('noise_full1000_est_rel_AD'):.2e} | "
                    f"{g('corr_AD'):.4f} | {g('gain_AD'):.3f} | {g('err_norm_AD_over_err'):.3f} | "
                    f"{g('err_norm_rep_over_err'):.3f} | {g('err_share_AD_proj'):.3f} | "
                    f"{g('cmp_mismatch_over_err'):.3f} | {g('repK_err_over_err'):.3f} |")
                summ.setdefault(mode, {}).setdefault(tag, {})[l] = {
                    k: g(k) for k in rs[0] if isinstance(rs[0][k], float)}
            lines.append("")
        # span test
        lines += [f"### span test ({mode}): residual ‖AD_lab − fit‖/‖AD_lab‖, medians; "
                  "in-sample per net and leave-one-net-out pooled per layer", "",
                  "| l | tgt | model | scalar gains (new2leg, hub, inh) | all 15 gates (in) | "
                  "all 15 (LONO) | twoleg only 5 | hub only 5 | inh only 5 | next-neuron scalar | "
                  "next-neuron feats | share inh | share new | checks add/dblock |", "|" + "---|" * 14]
        for l in range(15):
            sps = [n["modes"][mode]["layers"][str(l)].get("span") for n in nets]
            if any(s is None for s in sps):
                continue
            names = sps[0]["names"]
            fam = lambda pre: [i for i, s in enumerate(names) if s.startswith(pre)]
            scal = [i for i, s in enumerate(names) if s.endswith(":one")]
            for tgt in ("D3", "D21"):
                G = [s["gram"][tgt] for s in sps]
                rel = lambda r2, gg: math.sqrt(r2 / gg["ADlab2"])
                base = med([math.sqrt(gg["yty"] / gg["ADlab2"]) for gg in G])
                allidx = list(range(len(names)))
                fin = lambda idx: med([rel(_fit(gg["XtX"], gg["Xty"], gg["yty"], idx)[0], gg) for gg in G])
                lono = med([rel(r2, gg) for r2, gg in zip(_pooled_lono(G, allidx), G)]) if len(G) > 2 else float("nan")
                NG = [s["next_gram"][tgt] for s in sps]
                ns = med([math.sqrt(_fit(q["XtX"], q["Xty"], q["yty"], [0] if tgt == "D3" else [len(q["Xty"]) - 1])[0] / q["yty"]) for q in NG])
                nf = med([math.sqrt(_fit(q["XtX"], q["Xty"], q["yty"], list(range(len(q["Xty"]))))[0] / q["yty"]) for q in NG])
                sh = f"{med([s['share_inh_' + tgt] for s in sps]):.2f}"
                sn = f"{med([s['share_new_' + tgt] for s in sps]):.2f}"
                ck = (f"{max(s['check_additivity_' + tgt] for s in sps):.1e}/"
                      f"{max(s['check_dblock_AD_zero_' + tgt] for s in sps):.1e}")
                lines.append(f"| {l} | {tgt} | {base:.3f} | {fin(scal):.3f} | {fin(allidx):.3f} | "
                             f"{lono:.3f} | {fin(fam('twoleg')):.3f} | "
                             f"{fin(fam('hub')) if fam('hub') else float('nan'):.3f} | "
                             f"{fin(fam('inh')) if fam('inh') else float('nan'):.3f} | {ns:.3f} | {nf:.3f} | "
                             f"{sh} | {sn} | {ck} |")
        lines.append("")
    if "arms" in nets[0]:
        lines += ["## downstream (free mode): relative change of mean MSE vs the unmodified twin", "",
                  "ALL consumer layers 1..15 (final-layer MSE change, per net then median):", ""]
        for arm in nets[0]["arms"]["all"]:
            ch = [n["arms"]["all"][arm]["final"] / n["arms"]["base_final"] - 1.0
                  for n in nets if arm in n["arms"]["all"]]
            lines.append(f"- **{arm}**: median {med(ch):+.2%}  per net "
                         + ", ".join(f"{x:+.2%}" for x in ch))
            summ.setdefault("arms_all", {})[arm] = ch
        lines += ["", "SINGLE consumer layer L (median over nets: ΔMSE at L / L+1 / final):", "",
                  "| L | " + " | ".join(nets[0]["arms"]["single"]) + " |",
                  "|---|" + "---|" * len(nets[0]["arms"]["single"])]
        for Lc in range(1, 16):
            cells = []
            for arm in nets[0]["arms"]["single"]:
                rs = [n["arms"]["single"][arm][str(Lc)] for n in nets]
                cells.append(f"{med([r['L'] for r in rs]):+.2%} / "
                             f"{med([r.get('L+1', float('nan')) for r in rs]):+.2%} / "
                             f"{med([r['final'] for r in rs]):+.2%}")
            lines.append(f"| {Lc} | " + " | ".join(cells) + " |")
        lines.append("")
    txt = "\n".join(lines)
    (out / "report.md").write_text(txt, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summ))
    sys.stdout.reconfigure(encoding="utf-8")
    print(txt)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["step0", "run", "report"])
    ap.add_argument("--cand", default=str(ROOT / "estimator_k3_v62_s1.py"))
    ap.add_argument("--params", default=str(ROOT / "experiments" / "2026-09-25_twin" / "S3"
                                            / "params_best.npz"))
    ap.add_argument("--nets", default="0-3")
    ap.add_argument("--modes", default="free,exact")
    ap.add_argument("--no-arms", action="store_true")
    ap.add_argument("--h", type=float, default=0.05, help="central-difference step, inherited gates")
    ap.add_argument("--gain-file", default="", help='json {"D3": [15], "D21": [15]} per source layer')
    ap.add_argument("--max-layer", type=int, default=99, help="debug: run layers < this only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"step0": cmd_step0, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
