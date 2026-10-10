"""M5.3 -- the Q31 / D4 channel regression (HANDOFF 2026-10-10 Lane M5).

    .venv-gpu/Scripts/python.exe experiments/2026-10-10_m5/k4reg.py --nets 0:20 --out <dir>
    python experiments/2026-10-10_m5/k4reg.py --report <dir>      (CPU: fits + tables)

Runs the v81d chain in the TWIN (v62_gb + mask+dress+bank star,starall + refit_starall2 params;
twin arm vs the shipped file DEV 1.2e-4) on FIT nets, layer by layer, and at every layer
l = 2..14 (0-based; l = 15 is the last layer, which reads no off-diagonal object) records
weighted sufficient statistics for:

  Q31: target y = TRUE kappa_aaab / (s_a^3 s_b), s = the chain's sigma, off-diagonal pairs.
       baseline S0 = the chain's own normalised Q31 (the closure's nb["Q31"], learned modes
       included); residual r = y - S0. Pair weight (w3_a w1_b s_a^3 s_b)^2 -- the residual is
       scaled by how Q31 enters the covariance, (1/6) w3_a w1_b kappa_aaab.
  D4:  target y = TRUE kappa_aaaa / s_a^4 (= 2 dG_label); S0 = the chain's D4 (m dG + star);
       weight (w4_a s_a^4)^2.

Statistics per (net, layer): G = X^T W X, b = X^T W r, sums for centring, and the
Gaussian-approximation label-noise energy (Var kappa_aaab ~ 6 s^6 s^2 / N, Var kappa_aaaa
~ 24 s^8 / N, N = 1e9). Fits: per layer on nets 0..9, scored on nets 10..19.

Candidates (all normalised as the target):
  Q31: rad (3 a_a D21n_ab + a_b g_a) with the chain's D21/D3 and with the TRUE ones; its two
       halves; star-Q31 per lineage age (1 = youngest .. 6, 7+ lumped) and all ages x sg;
       path-Q31 per age (1..4) with the birth layer's C11; (2,2)-block per age (1..3) with
       the birth layer's post K22; rho, rho^2, C.C, (W^T diag(v) W), Sym6(M (x) C) with
       M in {diag dG, W^T diag(v) W}, Puffi's D3 m^T + m D3^T, (D_m D21^T + D21 D_m)/2,
       v v^T, alpha alpha, 1.
  D4:  12 mu D3 (chain / true), star-D4 per age, path-D4 per age, (2,2)-block D4 per age,
       dG, 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAIN = Path("C:/Users/samsu/Documents/arc-whest")
NS = 1e9
QLAYERS = range(2, 15)
AGES_STAR, AGES_PATH, AGES_K22 = 6, 4, 3

BASE = dict(
    cand=ROOT / "experiments" / "2026-09-27_gates" / "estimator_k3_v62_gb.py",
    mask="R=2.0,Hd=2.5,Dz=2.5,K=" + str(MAIN / "experiments/2026-10-06_mask/ktab/aggr_q50.json"),
    params=MAIN / "experiments/2026-10-08_p3/refit_starall2/params_best.npz")

REC_PATCH = ("    if last_layer:\n        S = {(1,): mu_pre, (3,): D3, (4,): D4}\n",
             "    if _REC is not None:\n"
             "        _REC(layer_index, adv=adv, bk=_bkst, w=w, alpha=alpha, sigma=sigma,\n"
             "             mu_pre=mu_pre, cov_pre=cov_pre, cov_in=cov, W=W, dG=dG, D4=D4, D3=D3)\n"
             "    if last_layer:\n        S = {(1,): mu_pre, (3,): D3, (4,): D4}\n")


def q31_names():
    n = ["rad", "rad_true", "radA", "radB"]
    n += [f"star{i}" for i in range(1, AGES_STAR + 1)] + ["star_old", "star_all"]
    n += [f"path{i}" for i in range(1, AGES_PATH + 1)]
    n += [f"k22_{i}" for i in range(1, AGES_K22 + 1)]
    n += ["rho", "rho2", "CC", "NV", "k31_dG", "k31_N", "D3m", "puffi21", "vv", "aa", "one"]
    return n


def d4_names():
    n = ["radD4", "radD4_true"]
    n += [f"star{i}" for i in range(1, AGES_STAR + 1)] + ["star_old"]
    n += [f"path{i}" for i in range(1, AGES_PATH + 1)]
    n += [f"k22_{i}" for i in range(1, AGES_K22 + 1)]
    n += ["dG", "one"]
    return n


def collect(a):
    os.environ.setdefault("TWIN_CACHE", str(MAIN / "cache" / "twin"))
    sys.path.insert(0, str(ROOT / "tools" / "twin"))
    import numpy as np
    import torch
    import twin as T

    T.ARMS["rec"] = [("def _layer(W, state, table, dtype, const, init_scale, last_layer, gens,\n",
                      "_REC = None\n\n\ndef _layer(W, state, table, dtype, const, init_scale, "
                      "last_layer, gens,\n"), REC_PATCH]
    tw = T.Twin(Path(BASE["cand"]), arm="mask+dress+bank+rec")
    tw.install(T.load_params(BASE["params"]))
    m = tw.mod
    m._MASK = T.parse_mask(BASE["mask"])
    m._DRESS = T.parse_dress("all")
    m._BANK = {"star": 1, "starall": 1}
    sg = m._TWG["sg"]
    data = T.Data()
    dev = T.DEV
    f64 = torch.float64
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = (int(x) for x in a.nets.split(":"))
    QN, DN = q31_names(), d4_names()

    for gi in range(lo, hi):
        fp = out / f"net_{gi:03d}.npz"
        if fp.exists():
            continue
        t0 = time.time()
        _z = np.load(MAIN / "cache" / "twin_labels" / f"fit_{gi:03d}.npz")
        lab = {k_: _z[k_] for k_ in ("Q31", "D21", "D3", "dG")}
        m._MASK = T.parse_mask(BASE["mask"])
        w, _truth = data.net("full", gi)
        rec, hook = {}, {}

        def R(l, **kw):
            rec.clear()
            rec.update(kw)

        def A2(l, name, value, *aux):
            if name == "Q31n":
                hook["Q31n"] = value
            elif name in ("D21", "K"):
                hook[name] = value
            return value

        m._REC, m._A2 = R, A2
        births = {}            # layer -> (C11 birth, post K22 at birth)
        stats = {}
        n = tw.width
        off = ~torch.eye(n, dtype=torch.bool, device=dev)
        state = (T.fnp.zeros(n), T.fnp.eye(n), None, 0.0, T.fnp.zeros(n))
        with torch.no_grad():
            for l in range(tw.depth):
                hook.clear()
                state = tw._step(l, w[l], state)
                last = l == tw.depth - 1
                if not last:
                    births[l] = ((rec["cov_pre"] * off).to(f64), hook["K"][(2, 2)].to(f64))
                if l in QLAYERS and rec["adv"] is not None:
                    stats[l] = layer_stats(l, rec, hook, births, lab, sg, m, off, torch, np, QN, DN)
                rec.clear()
        np.savez(fp, **{f"{l}_{k}": v for l, d in stats.items() for k, v in d.items()})
        print(f"[k4reg] net {gi}: {len(stats)} layers in {time.time() - t0:.0f} s", flush=True)
        prog = os.environ.get("WORKERQ_PROGRESS")
        if prog:
            Path(prog).write_text(str((gi - lo + 1) / (hi - lo)))
    return 0


def layer_stats(l, rec, hook, births, lab, sg, m, off, torch, np, QN, DN):
    dev = rec["sigma"].device
    f64 = torch.float64
    t = lambda x: torch.as_tensor(np.asarray(x), device=dev).to(f64)
    s = rec["sigma"].to(f64)
    al = rec["alpha"].to(f64)
    mu = rec["mu_pre"].to(f64)
    var = s * s
    C = (rec["cov_pre"].to(f64)) * off
    D21 = hook["D21"].to(f64)
    D3 = rec["D3"].to(f64)
    dG = rec["dG"].to(f64)
    w = rec["w"]
    w1, w3 = w[(1, 1)].to(f64), w[(3, 1)].to(f64)
    w4 = w[(4, 1)].to(f64)
    n = s.shape[0]
    norm = (s ** 3)[:, None] * s[None, :]
    # ---- targets ----
    Qt = t(lab["Q31"][l]) / norm
    D21t, D3t = t(lab["D21"][l]), t(lab["D3"][l])
    S0 = hook["Q31n"].to(f64)
    # ---- the tower, full legs (old first), age 1 = youngest ----
    adv = rec["adv"]
    A, M = m._full_legs_lean(adv)
    A, M = A.to(f64), M.to(f64)
    k = A.shape[1] // n
    assert k == l, (k, l)
    w3b = [x.to(f64) for x in rec["bk"]["w3b"]][-k:]
    w2b = adv.w2b.to(f64)
    assert w2b.shape[0] == k * n

    def cols(age):
        i = k - age
        return slice(i * n, (i + 1) * n)

    F = {}
    g3 = D3 / s ** 3
    D21n = D21 / ((s ** 2)[:, None] * s[None, :])
    F["radA"] = 3.0 * al[:, None] * D21n
    F["radB"] = al[None, :] * g3[:, None]
    F["rad"] = F["radA"] + F["radB"]
    D21tn = D21t / ((s ** 2)[:, None] * s[None, :])
    F["rad_true"] = 3.0 * al[:, None] * D21tn + al[None, :] * (D3t / s ** 3)[:, None]
    # star-Q31: sum_j c4_j (A_aj^3 M_bj + 3 M_aj A_aj^2 A_bj)
    star_all = torch.zeros((n, n), dtype=f64, device=dev)
    star_old = torch.zeros((n, n), dtype=f64, device=dev)
    d4star = {}
    d4star_old = torch.zeros(n, dtype=f64, device=dev)
    for age in range(1, k + 1):
        c = cols(age)
        Aa, Ma, c4 = A[:, c], M[:, c], w3b[k - age]
        A2c = (Aa * Aa) * c4[None, :]
        Qs = ((A2c * Aa) @ Ma.T + 3.0 * ((A2c * Ma) @ Aa.T)) / norm
        d4 = 4.0 * ((A2c * Aa) * Ma).sum(1) / s ** 4
        star_all = star_all + Qs * float(sg[l])
        if age <= AGES_STAR:
            F[f"star{age}"] = Qs
            d4star[age] = d4
        else:
            star_old = star_old + Qs
            d4star_old = d4star_old + d4
    F["star_old"], F["star_all"] = star_old, star_all
    for age in range(k + 1, AGES_STAR + 1):
        F[f"star{age}"] = torch.zeros((n, n), dtype=f64, device=dev)
    # path (birth C11 of that lineage) and (2,2)-block (birth post K22), youngest ages
    d4path, d4k22 = {}, {}
    for age in range(1, AGES_PATH + 1):
        if age > k:
            F[f"path{age}"] = torch.zeros((n, n), dtype=f64, device=dev)
            d4path[age] = torch.zeros(n, dtype=f64, device=dev)
            continue
        c = cols(age)
        Cb = births[l - age][0]
        J = M[:, c] * w2b[c][None, :]
        AJ = A[:, c] * J
        V = AJ @ Cb
        F[f"path{age}"] = ((J * V) @ A[:, c].T + (A[:, c] * V) @ J.T) / norm
        d4path[age] = (V * AJ).sum(1) / s ** 4
    for age in range(1, AGES_K22 + 1):
        if age > k:
            F[f"k22_{age}"] = torch.zeros((n, n), dtype=f64, device=dev)
            d4k22[age] = torch.zeros(n, dtype=f64, device=dev)
            continue
        P = M[:, cols(age)]
        K22 = births[l - age][1] * off
        PPK = (P * P) @ K22
        F[f"k22_{age}"] = ((PPK * P) @ P.T) / norm
        d4k22[age] = (PPK * (P * P)).sum(1) / s ** 4
    rho = C / (s[:, None] * s[None, :])
    F["rho"], F["rho2"] = rho, rho * rho
    F["CC"] = (C * C) / norm
    Win = rec["W"].to(f64)
    v_in = torch.diagonal(rec["cov_in"]).to(f64)
    NV = (Win.T * v_in[None, :]) @ Win
    NVo = NV * off
    F["NV"] = NVo / norm
    F["k31_dG"] = (dG[:, None] * C) / norm
    F["k31_N"] = (torch.diagonal(NV)[:, None] * C + var[:, None] * NVo) / norm
    F["D3m"] = (D3[:, None] * mu[None, :] + mu[:, None] * D3[None, :]) / norm
    F["puffi21"] = 0.5 * (mu[:, None] * D21.T + mu[None, :] * D21) / norm
    F["vv"] = (var[:, None] * var[None, :]) / norm
    F["aa"] = al[:, None] * al[None, :]
    F["one"] = torch.ones((n, n), dtype=f64, device=dev)
    # ---- Q31 stats ----
    ww = ((w3[:, None] * w1[None, :]) * norm) ** 2
    sel = off
    X = torch.stack([F[k_][sel] for k_ in QN], 1)
    y = Qt[sel]
    r = (Qt - S0)[sel]
    W_ = ww[sel]
    out = suff(X, y, r, W_, S0[sel], noise=6.0 / NS, torch=torch)
    # ---- D4 stats ----
    yD = 2.0 * t(lab["dG"][l]) / s ** 4
    S0D = rec["D4"].to(f64) / s ** 4
    FD = {"radD4": 12.0 * mu * D3 / s ** 4, "radD4_true": 12.0 * mu * D3t / s ** 4,
          "star_old": d4star_old, "dG": dG / s ** 4, "one": torch.ones(n, dtype=f64, device=dev)}
    for age in range(1, AGES_STAR + 1):
        FD[f"star{age}"] = d4star.get(age, torch.zeros(n, dtype=f64, device=dev))
    for age in range(1, AGES_PATH + 1):
        FD[f"path{age}"] = d4path[age]
    for age in range(1, AGES_K22 + 1):
        FD[f"k22_{age}"] = d4k22[age]
    XD = torch.stack([FD[k_] for k_ in DN], 1)
    outD = suff(XD, yD, yD - S0D, (w4 * s ** 4) ** 2, S0D, noise=24.0 / NS, torch=torch)
    res = {f"q_{k_}": v for k_, v in out.items()}
    res.update({f"d_{k_}": v for k_, v in outD.items()})
    return res


def suff(X, y, r, W, S0, noise, torch):
    Xw = X * W[:, None]
    return {"G": (Xw.T @ X).cpu().numpy(), "b": (Xw.T @ r).cpu().numpy(),
            "sx": Xw.sum(0).cpu().numpy(),
            "rr": float((W * r * r).sum()), "sr": float((W * r).sum()),
            "yy": float((W * y * y).sum()), "sy": float((W * y).sum()), "sw": float(W.sum()),
            "s0s0": float((W * S0 * S0).sum()), "noise": float(W.sum()) * noise,
            "npair": int(W.numel())}


# ------------------------------------------------------------------ report

def report(a):
    import numpy as np
    d = Path(a.report)
    nets = sorted(int(p.stem[4:]) for p in d.glob("net_*.npz"))
    Z = {i: np.load(d / f"net_{i:03d}.npz") for i in nets}
    tr = [i for i in nets if i < 10]
    te = [i for i in nets if 10 <= i < 20]
    lines = [f"# M5.3 regression -- train nets {tr}, test nets {te}", ""]
    summary = {}
    for tag, names in (("q", q31_names()), ("d", d4_names())):
        lines += [f"## {'Q31' if tag == 'q' else 'D4'} (out-of-sample weighted R^2 on test nets)", ""]
        layers = sorted({int(k.split("_")[0]) for k in Z[nets[0]].files})

        def agg(ids, l):
            o = {}
            for key in ("G", "b", "sx", "rr", "sr", "yy", "sy", "sw", "noise"):
                o[key] = sum(Z[i][f"{l}_{tag}_{key}"] for i in ids)
            return o

        hdr = "| l | R2_S0 | noise-ceiling | " + " | ".join(names) + " | greedy (set: dR2) |"
        lines += [hdr, "|" + "---|" * (len(names) + 4)]
        summary[tag] = {}
        for l in layers:
            A, B = agg(tr, l), agg(te, l)
            sst = B["yy"] - B["sy"] ** 2 / B["sw"]
            r2s0 = 1.0 - B["rr"] / sst
            ceil = 1.0 - B["noise"] / sst

            def oos(S):
                S = list(S)
                if not S:
                    return 0.0
                G = A["G"][np.ix_(S, S)]
                if np.any(np.diag(G) <= 0):
                    return 0.0
                dsc = 1.0 / np.sqrt(np.diag(G))
                Gs = G * dsc[:, None] * dsc[None, :]
                beta = np.linalg.lstsq(Gs + 1e-10 * np.eye(len(S)), A["b"][S] * dsc,
                                       rcond=1e-12)[0] * dsc
                ssr = B["rr"] - 2 * beta @ B["b"][S] + beta @ B["G"][np.ix_(S, S)] @ beta
                return (B["rr"] - ssr) / sst

            single = [oos([j]) for j in range(len(names))]
            chosen, cur, path = [], 0.0, []
            for _ in range(6):
                best = max(((oos(chosen + [j]), j) for j in range(len(names)) if j not in chosen),
                           default=None)
                if best is None or best[0] - cur < 0.002:
                    break
                cur = best[0]
                chosen.append(best[1])
                path.append(f"{names[best[1]]}:{cur:+.3f}")
            summary[tag][l] = {"r2_s0": r2s0, "ceiling": ceil,
                               "single": dict(zip(names, single)), "greedy": path, "greedy_r2": cur}
            lines.append(f"| {l} | {r2s0:.3f} | {ceil:.3f} | "
                         + " | ".join(f"{x:+.3f}" for x in single) + " | " + ", ".join(path) + " |")
        lines.append("")
    (d / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", default="0:20")
    ap.add_argument("--out", default="C:/Users/samsu/Documents/arc-whest-lane-m/experiments/2026-10-10_m5/k4reg")
    ap.add_argument("--report", default="")
    a = ap.parse_args()
    return report(a) if a.report else collect(a)


if __name__ == "__main__":
    raise SystemExit(main())
