"""Markdown tables from a `tools/anatomy2.py` result file.

    python tools/anatomy2.py --report experiments/2026-09-25_anatomy2/fit20.json

Every MSE here is the CROSS estimate mean((p - gt)(p - kp)) -- the noise-free MSE,
unbiased because aicrowd's and keenanpepper's labels are independent N = 1e9
draws -- averaged over nets. "Recovered" is 1 - MSE(arm) / MSE(base), computed on
the net-averaged MSEs (the score is a mean of MSEs, so that is the weighting that
matters).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _mean_over(nets, getter):
    vals = [getter(r) for r in nets.values()]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _g(d, *ks):
    for k in ks:
        if d is None:
            return None
        d = d.get(str(k)) if isinstance(d, dict) else None
    return d


def _layers(nets):
    any_net = next(iter(nets.values()))
    return sorted(int(k) for k in any_net["chain"])


def table_objects(nets, arch=None):
    Ls = _layers(nets)
    keys = ["mu", "var", "C11", "D3", "D21", "dG", "D22", "Q31", "cov_out"]
    hdr = "| layer (1-based) | " + " | ".join(keys) + " | cov err: diag / trace / traceless share |"
    out = [hdr, "|" + "---|" * (len(keys) + 2)]
    for l in Ls:
        src = (lambda r: r["arch"][arch]["objects"]) if arch else (lambda r: r["objects"])
        cells = []
        for k in keys:
            v = _mean_over(nets, lambda r: _g(src(r), l, k, "rel"))
            cells.append("—" if np.isnan(v) else f"{v:.3g}")
        d = _mean_over(nets, lambda r: _g(src(r), l, "cov_split", "diag_share"))
        t = _mean_over(nets, lambda r: _g(src(r), l, "cov_split", "trace_share"))
        tl = _mean_over(nets, lambda r: _g(src(r), l, "cov_split", "traceless_share"))
        split = "—" if np.isnan(d) else f"{d:.2f} / {t:.3f} / {tl:.2f}"
        out.append(f"| {l + 1} | " + " | ".join(cells) + f" | {split} |")
    return "\n".join(out)


def table_substitution(nets, kind, base, arms):
    Ls = [l for l in _layers(nets) if l >= (2 if kind == "twostep" else 1)]
    out = ["| layer | chain MSE | " + base + " MSE | "
           + " | ".join(f"{a} rec." for a in arms) + " |",
           "|" + "---|" * (len(arms) + 3)]
    for l in Ls:
        ch = _mean_over(nets, lambda r: _g(r, "chain", l)[1] if _g(r, "chain", l) else None)
        b = _mean_over(nets, lambda r: (_g(r, kind, base, l) or [None, None])[1])
        cells = []
        for a in arms:
            m = _mean_over(nets, lambda r: (_g(r, kind, a, l) or [None, None])[1])
            cells.append(f"{(1 - m / b) * 100:+.0f}%")
        out.append(f"| {l + 1} | {ch:.3e} | {b:.3e} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_rolled(nets):
    Ls = _layers(nets)
    last = Ls[-1]
    ch = _mean_over(nets, lambda r: _g(r, "chain", last)[1])
    keys = list(next(iter(nets.values()))["rolled"].keys())
    out = [f"Chain final-layer MSE (x-est, mean of {len(nets)} nets): **{ch:.4e}**", "",
           "| substituted at layers 2..P+1 (1-based) | final MSE | share of chain final MSE removed | MSE at layer 9 | at layer 13 |",
           "|---|---:|---:|---:|---:|"]
    for k in keys:
        m = _mean_over(nets, lambda r: _g(r, "rolled", k, last)[1])
        m8 = _mean_over(nets, lambda r: (_g(r, "rolled", k, 8) or [None, None])[1])
        m12 = _mean_over(nets, lambda r: (_g(r, "rolled", k, 12) or [None, None])[1])
        obj, P = k.split("@")
        lo, _, hi = P.rpartition("-")      # band keys "obj@lo-hi" (0-based)
        span = f"{int(lo) + 1}..{int(hi) + 1}" if lo else f"2..{int(P) + 1}"
        out.append(f"| {obj} @ {span} | {m:.4e} | {(1 - m / ch) * 100:+.1f}% | "
                   f"{m8:.3e} | {m12:.3e} |")
    return "\n".join(out)


def _gram(stats, cols, const):
    """Summed (A, v, yy, sy, n) with optional intercept as column 0."""
    A = sum(np.array(s["XtX"]) for s in stats)[np.ix_(cols, cols)]
    v = sum(np.array(s["Xty"]) for s in stats)[cols]
    yy = sum(s["yty"] for s in stats)
    sy = sum(s["sy"] for s in stats)
    n = sum(s["n"] for s in stats)
    if const:
        sx = sum(np.array(s["sx"]) for s in stats)[cols]
        A = np.block([[np.array([[n]]), sx[None, :]], [sx[:, None], A]])
        v = np.concatenate([[sy], v])
    return A, v, yy, sy, n


def _r2(stats, cols, const, b):
    A, v, yy, sy, n = _gram(stats, cols, const)
    return 1 - (yy - 2 * b @ v + b @ A @ b) / (yy - sy * sy / n)


def _fit_eval(stats_tr, stats_ev, cols, const=True, ridge=1e-9):
    """Least squares on the train nets, R^2 (about the eval mean) on the eval nets."""
    A, v, *_ = _gram(stats_tr, cols, const)
    lam = ridge * np.trace(A) / A.shape[0]
    beta = np.linalg.solve(A + lam * np.eye(A.shape[0]), v)
    return beta, _r2(stats_tr, cols, const, beta), _r2(stats_ev, cols, const, beta)


def _fixed_r2(stats, col):
    """R^2 of y = X[:, col] with coefficient exactly 1 (the chain's own form)."""
    return _r2(stats, [col], False, np.array([1.0]))


def table_k4(nets):
    keys = sorted(nets, key=int)
    half = len(keys) // 2
    tr, ev = (keys[:half], keys[half:]) if half else (keys, keys)   # 1 net: in-sample
    Ls = sorted({int(l) for k in keys for l in nets[k]["k4"]})
    out = [f"Fitted on nets {tr[0]}..{tr[-1]}, R² on nets {ev[0]}..{ev[-1]} (normalised "
           "slices, off-diagonal entries; R² is about the eval-set mean, and every fitted "
           "model has an intercept unless marked).", "",
           "κ4_iijj / (σ_i²σ_j²). B = (κ4_iiii + κ4_jjjj)/6 from TRUE marginals = the chain's "
           "form (m/6)(dG_i+dG_j) with exact dG:", "",
           "| layer | mean target | B, coef 1 | λ*·ρ (no intercept): λ*, R² | const + λ*ρ | const + B + λ*ρ | const + 4-mode {ρ, ρ², ḡρ, α_iα_j} | const + B + 4-mode | T (transported diagonal), coef 1 | const + T | const + B + 4-mode + T |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for l in Ls:
        S = lambda ks: [nets[k]["k4"][str(l)]["k22"] for k in ks]
        ybar = sum(s["sy"] for s in S(ev)) / sum(s["n"] for s in S(ev))
        fixed = _fixed_r2(S(ev), 0)
        b1, _, r1 = _fit_eval(S(tr), S(ev), [1], const=False)
        _, _, rc = _fit_eval(S(tr), S(ev), [1])
        _, _, rb = _fit_eval(S(tr), S(ev), [0, 1])
        _, _, r4 = _fit_eval(S(tr), S(ev), [1, 2, 3, 4])
        _, _, r5 = _fit_eval(S(tr), S(ev), [0, 1, 2, 3, 4])
        tf = _fixed_r2(S(ev), 5)
        _, _, rt = _fit_eval(S(tr), S(ev), [5])
        _, _, r6 = _fit_eval(S(tr), S(ev), [0, 1, 2, 3, 4, 5])
        out.append(f"| {l + 1} | {ybar:.2e} | {fixed:.3f} | {b1[0]:.4f}, {r1:.3f} | {rc:.3f} | "
                   f"{rb:.3f} | {r4:.3f} | {r5:.3f} | {tf:.3f} | {rt:.3f} | {r6:.3f} |")
    out += ["", "κ4_iiij / (σ_i³σ_j). The chain's form is (m/2)·λ·C_ij (no intercept); "
            "\"fitted coef\" is the best (m/2)λ:", "",
            "| layer | chain form: fitted (m/2)λ, R² | chain (m/2)λ used | λ*·ρ (no intercept) R² | const + 4-mode {ρ, ρ², g_iρ, α_iα_j} | const + C-form + 4-mode | T, coef 1 | const + T | const + C-form + 4-mode + T |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for l in Ls:
        S = lambda ks: [nets[k]["k4"][str(l)]["k31"] for k in ks]
        b0, _, r0 = _fit_eval(S(tr), S(ev), [0], const=False)
        _, _, r1 = _fit_eval(S(tr), S(ev), [1], const=False)
        _, _, r4 = _fit_eval(S(tr), S(ev), [1, 2, 3, 4])
        _, _, r5 = _fit_eval(S(tr), S(ev), [0, 1, 2, 3, 4])
        lam_used = nets[keys[0]]["k4"][str(l)].get("lam_chain", float("nan"))
        tf = _fixed_r2(S(ev), 5)
        _, _, rt = _fit_eval(S(tr), S(ev), [5])
        _, _, r6 = _fit_eval(S(tr), S(ev), [0, 1, 2, 3, 4, 5])
        out.append(f"| {l + 1} | {b0[0]:.4f}, {r0:.3f} | {lam_used:.4f} | {r1:.3f} | {r4:.3f} | "
                   f"{r5:.3f} | {tf:.3f} | {rt:.3f} | {r6:.3f} |")
    return "\n".join(out)


def report(path: Path) -> str:
    res = json.loads(Path(path).read_text(encoding="utf-8"))
    nets = res["nets"]
    parts = [f"# Anatomy 2 — {res['candidate']} on {res['panel'].upper()} ({len(nets)} nets)", ""]
    parts += ["## 1. Object errors (relative Frobenius, mean over nets)", "", table_objects(nets), ""]
    for arm in ("T0", "T0x"):
        if all("objects" in r.get("arch", {}).get(arm, {}) for r in nets.values()):
            parts += [f"### {arm}", "", table_objects(nets, arm), ""]
            last = _layers(nets)[-1]
            parts.append(f"{arm} final MSE (x-est): "
                         f"{_mean_over(nets, lambda r: r['arch'][arm]['chain'][str(last)][1]):.4e}")
            parts.append("")
    parts += ["## 2. κ4 model check", "", table_k4(nets), ""]
    parts += ["## 3a. One-step (substituted at l, mean read at l)", "",
              table_substitution(nets, "onestep", "base", ["var", "D3", "D4", "marg"]), ""]
    parts += ["## 3b. Two-step (substituted at l−1, mean read at l)", "",
              table_substitution(nets, "twostep", "base2",
                                 ["var", "C11", "cov", "D3", "D21", "D4", "D22", "Q31",
                                  "offdiag", "all"]), ""]
    parts += ["## 3c. Rolled", "", table_rolled(nets), ""]
    return "\n".join(parts)
