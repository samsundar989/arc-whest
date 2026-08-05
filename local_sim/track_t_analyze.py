"""
Track T analysis: does denoising the last step's moments beat sampling it?

Reads the per-draw statistics minted by `track_t_mint.py` and answers, in order:

  1. **The four arms.** Final-layer MSE for the direct sample mean (= v2.4) and
     for the analytic rectified-Gaussian tail evaluated at every combination of
     {sampled, reference} x {mean, variance}. The (reference, reference) arm is
     the pure model bias — Gaussian closure of a non-Gaussian z, plus pruning.
     If that arm does not beat sampling, no amount of moment denoising can.

  2. **Bias/variance split across draws.** With R independent digital shifts per
     MLP, every estimator's error splits into a per-neuron bias (constant across
     draws) and a variance. Bias is uncorrectable by blending; variance is not.
     The bias^2 estimate is corrected for the R draws' own noise (var/R).

  3. **Shrinkage.** v_lambda = lambda * v_closure + (1-lambda) * v_hat, with the
     Ledoit-Wolf analytic intensity as a reference point. A scale factor kappa on
     the closure target is fitted too, because the closure's variance is known to
     be biased and shrinking toward a biased target injects that bias.

  4. **The blend.** final = beta*analytic + (1-beta)*sampling. Both estimators are
     built from the SAME draw, so their errors are correlated and the textbook
     precision weights are wrong; beta is fitted empirically and cross-fitted.

  5. **The gate.** V*c/B against v2.4, with the added ops priced.

Run:
    uv run python local_sim/track_t_analyze.py --data <tt_mini.npz>
    uv run python local_sim/track_t_analyze.py --data <tt_mini.npz> --fit <tt_full.npz>
"""

import argparse

import numpy as np
from scipy.special import ndtr

INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)
BUDGET = 2.72e11


def npdf(x):
    return INV_SQRT_2PI * np.exp(-0.5 * x * x)


def rect_gauss(mu, var):
    """E[ReLU(X)] for X ~ N(mu, var), elementwise."""
    sig = np.sqrt(np.maximum(var, 1e-30))
    a = mu / sig
    return mu * ndtr(a) + sig * npdf(a)


class Data:
    def __init__(self, path):
        d = np.load(path)
        self.m1 = d["m1"]            # (M, R, n)
        self.m2 = d["m2"]
        self.s = d["s"]
        self.mu_c = d["mu_c"]        # (M, n) closure pre-activation mean
        self.v_c = d["v_c"]
        self.p_c = d["p_c"]
        self.t = d["truth"]
        self.n_alive = d["n_alive"]
        self.cost = d["cost"]
        self.R = int(d["draws"])
        self.N = int(d["n"])
        self.M, _, self.n = self.m1.shape
        self.vhat = self.m2 - self.m1**2
        # pooled reference over all R draws == R*N samples
        self.m1_ref = self.m1.mean(axis=1)
        self.v_ref = self.m2.mean(axis=1) - self.m1_ref**2
        self.s_ref = self.s.mean(axis=1)

    def mse(self, est):
        """est: (M,R,n) or (M,n) -> mean squared error against truth."""
        t = self.t[:, None, :] if est.ndim == 3 else self.t
        return float(np.mean((est - t) ** 2))


def bias_var(est, t, R):
    """Split MSE into bias^2 (corrected for the R-draw noise) and variance."""
    m = est.mean(axis=1)
    v = est.var(axis=1, ddof=1)
    b2 = (m - t) ** 2 - v / R
    return float(np.mean(b2)), float(np.mean(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="evaluation set (mini)")
    ap.add_argument("--fit", default="", help="disjoint fit set (full split)")
    args = ap.parse_args()

    D = Data(args.data)
    print(f"eval set: {D.M} MLPs x {D.R} draws x N={D.N}  (reference = {D.R * D.N:,} samples)")
    print(f"mean alive at layer 31: {D.n_alive[:, -2].mean():.1f}/{D.n} "
          f"| v2.4 per-sample matmul cost {D.cost.mean():,.0f}\n")

    # ---------------------------------------------------------------- 1. arms
    print("=" * 78)
    print("1. THE FOUR ARMS  (final-layer MSE, mean over MLPs and draws)")
    print("=" * 78)
    a_ss = rect_gauss(D.m1, D.vhat)                                   # sampled mean, sampled var
    a_rs = rect_gauss(D.m1_ref[:, None, :], D.vhat)                   # reference mean, sampled var
    a_sr = rect_gauss(D.m1, D.v_ref[:, None, :])                      # sampled mean, reference var
    a_rr = rect_gauss(D.m1_ref, D.v_ref)                              # both reference
    base = D.mse(D.s)
    rows = [
        ("direct sample mean  (= v2.4)", base),
        ("analytic (sampled mu, sampled v)", D.mse(a_ss)),
        ("analytic (REFERENCE mu, sampled v)", D.mse(a_rs)),
        ("analytic (sampled mu, REFERENCE v)", D.mse(a_sr)),
        ("analytic (REFERENCE mu, REFERENCE v) = model bias", D.mse(a_rr)),
        ("closure only (v0 analytic prediction)", D.mse(D.p_c)),
        ("reference direct mean (pruned model, R*N samples)", D.mse(D.s_ref)),
    ]
    print(f"{'arm':<52} {'MSE':>12} {'vs v2.4':>9}")
    print("-" * 78)
    for label, v in rows:
        print(f"{label:<52} {v:>12.4e} {base / v:>8.3f}x")

    # -------------------------------------------------- 2. bias/variance split
    print("\n" + "=" * 78)
    print("2. BIAS / VARIANCE SPLIT ACROSS DRAWS")
    print("=" * 78)
    print(f"{'estimator':<40} {'bias^2':>12} {'variance':>12} {'bias share':>11}")
    print("-" * 78)
    for label, est in (("direct sample mean (v2.4)", D.s),
                       ("analytic (sampled mu, sampled v)", a_ss),
                       ("analytic (ref mu, sampled v)", a_rs),
                       ("analytic (sampled mu, ref v)", a_sr)):
        b2, v = bias_var(est, D.t, D.R)
        print(f"{label:<40} {b2:>12.4e} {v:>12.4e} {b2 / (b2 + v):>10.1%}")

    # ------------------------------------------------- 3. error decomposition
    print("\n" + "=" * 78)
    print("3. WHERE THE ANALYTIC ARM'S ERROR COMES FROM")
    print("=" * 78)
    e_model = (a_rr - D.t)[:, None, :] + np.zeros_like(a_ss)
    e_mean = a_sr - a_rr[:, None, :]      # moving mu from reference to sampled
    e_var = a_rs - a_rr[:, None, :]       # moving v  from reference to sampled
    e_tot = a_ss - D.t[:, None, :]
    inter = e_tot - (e_model + e_mean + e_var)
    parts = [("model bias (closure + pruning)", e_model), ("mean-estimation noise", e_mean),
             ("variance-estimation noise", e_var), ("interaction", inter)]
    print(f"{'term':<40} {'mean square':>13} {'share of total':>15}")
    print("-" * 78)
    tot = float(np.mean(e_tot**2))
    for label, e in parts:
        print(f"{label:<40} {float(np.mean(e**2)):>13.4e} {float(np.mean(e**2)) / tot:>14.1%}")
    print(f"{'TOTAL (with cross terms)':<40} {tot:>13.4e} {1.0:>14.1%}")
    for (la, ea), (lb, eb) in ((parts[0], parts[1]), (parts[0], parts[2]), (parts[1], parts[2])):
        c = float(np.mean(ea * eb))
        print(f"    cross-term 2*<{la[:22]}, {lb[:22]}> = {2 * c:+.4e}")

    # ------------------------------------------------------- 3b. sampling arm
    e_s = D.s - D.t[:, None, :]
    rho = float(np.mean(e_s * e_tot) / np.sqrt(np.mean(e_s**2) * np.mean(e_tot**2)))
    print(f"\n  corr(analytic error, sampling error) over the same draws = {rho:+.4f}")
    print("  (the blend's ceiling is set by this: rho -> 1 means the two carry the same error)")

    # -------------------------------------------------------- 4. shrinkage fit
    print("\n" + "=" * 78)
    print("4. SHRINKING THE VARIANCE ESTIMATE TOWARD THE CLOSURE")
    print("=" * 78)
    kappa = float(np.mean(D.v_ref) / np.mean(D.v_c))
    print(f"  closure variance vs reference: kappa = mean(v_ref)/mean(v_closure) = {kappa:.5f}")
    print(f"  rms relative error of v_closure (after kappa): "
          f"{float(np.sqrt(np.mean((kappa * D.v_c / D.v_ref - 1) ** 2))):.4f}")
    var_vhat = float(np.mean(D.vhat.var(axis=1, ddof=1)))
    print(f"  rms relative error of v_hat (N={D.N}):          "
          f"{float(np.sqrt(np.mean((D.vhat / D.v_ref[:, None, :] - 1) ** 2))):.4f}")
    denom = float(np.mean((D.vhat - (kappa * D.v_c)[:, None, :]) ** 2))
    lw = var_vhat / denom if denom else 0.0
    print(f"  Ledoit-Wolf analytic intensity (reduced form)  = {lw:.4f}")
    print(f"\n  {'lambda':>7} {'MSE analytic':>14} {'vs v2.4':>9}")
    print("  " + "-" * 34)
    best = (None, np.inf)
    for lam in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, lw):
        v_l = lam * (kappa * D.v_c)[:, None, :] + (1 - lam) * D.vhat
        m = D.mse(rect_gauss(D.m1, v_l))
        tag = "  <- LW" if lam == lw else ""
        print(f"  {lam:>7.4f} {m:>14.4e} {base / m:>8.3f}x{tag}")
        if m < best[1]:
            best = (lam, m)
    print(f"  best lambda = {best[0]:.4f} -> {best[1]:.4e} ({base / best[1]:.3f}x)")

    # ------------------------------------------------------------ 5. the blend
    print("\n" + "=" * 78)
    print("5. PRECISION BLEND WITH THE DIRECT SAMPLE MEAN")
    print("=" * 78)
    lam_star = best[0]
    v_l = lam_star * (kappa * D.v_c)[:, None, :] + (1 - lam_star) * D.vhat
    a_best = rect_gauss(D.m1, v_l)
    va, vs = float(np.mean((a_best - D.t[:, None, :]) ** 2)), base
    cov = float(np.mean((a_best - D.t[:, None, :]) * e_s))
    beta_txt = (vs - cov) / (va + vs - 2 * cov)
    print(f"  Var(analytic)={va:.4e}  Var(sampling)={vs:.4e}  Cov={cov:.4e}  "
          f"rho={cov / np.sqrt(va * vs):+.4f}")
    print(f"  textbook (independent-error) weight would be {vs / (va + vs):.4f}; "
          f"correlation-aware optimum is {beta_txt:.4f}")
    print(f"\n  {'beta':>7} {'MSE':>13} {'vs v2.4':>9}")
    print("  " + "-" * 33)
    bb = (0.0, base)
    for beta in list(np.arange(0.0, 1.01, 0.1)) + [beta_txt]:
        m = D.mse(beta * a_best + (1 - beta) * D.s)
        print(f"  {beta:>7.4f} {m:>13.4e} {base / m:>8.3f}x")
        if m < bb[1]:
            bb = (float(beta), m)
    print(f"  best beta = {bb[0]:.4f} -> {bb[1]:.4e} ({base / bb[1]:.3f}x)")

    # ------------------------------------------- 5b. per-alpha-band blending
    print("\n" + "=" * 78)
    print("5b. IS THE BLEND WORTH MORE INSIDE ALPHA BANDS?")
    print("=" * 78)
    alpha_c = D.mu_c / np.sqrt(np.maximum(D.v_c, 1e-30))
    bands = [(-99, -3), (-3, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 3), (3, 99)]
    ea = a_best - D.t[:, None, :]
    tot_base = np.mean(e_s**2)
    gained = 0.0
    print(f"  {'alpha band':>12} {'share of MSE':>13} {'rho':>7} {'beta*':>7} {'band gain':>10}")
    print("  " + "-" * 56)
    for lo, hi in bands:
        m = (alpha_c >= lo) & (alpha_c < hi)
        if not m.any():
            continue
        sel = m[:, None, :] & np.ones_like(ea, dtype=bool)
        es, eaa = e_s[sel], ea[sel]
        vs_, va_ = float(np.mean(es**2)), float(np.mean(eaa**2))
        c_ = float(np.mean(es * eaa))
        den = va_ + vs_ - 2 * c_
        b_ = (vs_ - c_) / den if den > 0 else 0.0
        b_ = min(max(b_, 0.0), 1.0)
        new = b_**2 * va_ + (1 - b_) ** 2 * vs_ + 2 * b_ * (1 - b_) * c_
        w = es.size / e_s.size
        gained += w * (vs_ - new)
        print(f"  {f'[{lo},{hi})':>12} {w * vs_ / tot_base:>12.1%} {c_ / np.sqrt(va_ * vs_):>7.3f} "
              f"{b_:>7.4f} {vs_ / new:>9.4f}x")
    print(f"  oracle-per-band blend over the whole layer: {tot_base / (tot_base - gained):.4f}x")

    # ---------------------------------------------------------- 6. cross-fit
    if args.fit:
        print("\n" + "=" * 78)
        print("6. CROSS-FIT: parameters from the disjoint fit set, applied here")
        print("=" * 78)
        F = Data(args.fit)
        kf = float(np.mean(F.v_ref) / np.mean(F.v_c))
        gridl = np.arange(0.0, 1.001, 0.02)
        fb = (0.0, 0.0, np.inf)
        for lam in gridl:
            v_l = lam * (kf * F.v_c)[:, None, :] + (1 - lam) * F.vhat
            af = rect_gauss(F.m1, v_l)
            for beta in np.arange(0.0, 1.001, 0.02):
                m = F.mse(beta * af + (1 - beta) * F.s)
                if m < fb[2]:
                    fb = (float(lam), float(beta), m)
        lam_f, beta_f, m_f = fb
        print(f"  fit set ({F.M} MLPs): kappa={kf:.5f} lambda={lam_f:.3f} beta={beta_f:.3f} "
              f"-> fit-set MSE {m_f:.4e} ({F.mse(F.s) / m_f:.3f}x in-sample)")
        v_l = lam_f * (kf * D.v_c)[:, None, :] + (1 - lam_f) * D.vhat
        est = beta_f * rect_gauss(D.m1, v_l) + (1 - beta_f) * D.s
        m_x = D.mse(est)
        print(f"  applied to eval set: MSE {m_x:.4e}  ({base / m_x:.4f}x over v2.4)")
        # draw-0 only = the single realisation the grader would see
        est0 = est[:, 0, :]
        print(f"  draw-0 only (one realisation, as graded): {D.mse(est0):.4e} vs "
              f"v2.4 {D.mse(D.s[:, 0, :]):.4e} ({D.mse(D.s[:, 0, :]) / D.mse(est0):.4f}x)")

        # ------------------------------------------------------------ 7. gate
        print("\n" + "=" * 78)
        print("7. GATE  (V*c/B, with the added ops priced)")
        print("=" * 78)
        # v2.4 marginal per-sample cost as minted, plus Track T's additions:
        #   mean(z): 1/elt, mean(z^2): 2/elt over the (N x width) pre-activations
        add_per_sample = 3.0 * D.n
        c0 = float(D.cost.mean()) + 0.5 * D.n * 32 + 1.0 * D.n * 32 + 30 * 3 * D.n + 171 * D.n
        c1 = c0 + add_per_sample
        v0, v1 = base * D.N, m_x * D.N
        adj0, adj1 = v0 * c0 / BUDGET, v1 * c1 / BUDGET
        print(f"  per-sample cost   v2.4 {c0:,.0f}   +T {c1:,.0f}  (+{100 * (c1 / c0 - 1):.3f}%)")
        print(f"  per-sample var    v2.4 {v0:.4e}   +T {v1:.4e}")
        print(f"  adjusted (V*c/B)  v2.4 {adj0:.4e}   +T {adj1:.4e}   => {adj0 / adj1:.4f}x")
        print(f"\n  GATE (>=1.10x): {'PASS' if adj0 / adj1 >= 1.10 else 'FAIL'}")


if __name__ == "__main__":
    main()
