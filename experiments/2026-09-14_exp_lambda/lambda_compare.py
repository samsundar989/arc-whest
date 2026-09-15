#!/usr/bin/env python3
"""TASK A: E14 `_tree_core` label-free lambda vs 504aldo's fitted LAM table.

Plain float64 numpy Gaussian covariance propagation through He-init ReLU MLPs
(n=1024, L=16, bias-free), replicating estimator_e14_tree.py's Gaussian-order
closure (the frozen F21 term table evaluated with all cumulant slices = 0), and at
every non-final layer the exact `_tree_core` contraction (ported line by line).

Outputs (in this directory):
  lam_post_e14.csv        E14 lambda (post-ReLU C_out definition), MLP x layer
  lam_pre_*.csv           pre-activation-aligned LS coefficients (504aldo's definition)
  lam_compare_summary.csv per-layer means / ratios / correlations
"""
import base64, json, re, sys, time, zlib, os
import numpy as np
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
E14_PATH = "/mnt/user-data/uploads/arc-whest/estimator_e14_tree.py"
ALDO_DIR = ("/tmp/claude-0/-home-claude/f77c3f4b-531a-5071-9a15-1bd37ed4507c/"
            "scratchpad/whest-p2-cumulant-k3/generators")
N, DEPTH = 1024, 16
ALPHA_CAP = 40.0
CLOSURE = os.environ.get("CLOSURE", "e14")   # "e14" (table, deg-2 off-diag) | "deg4"


# ----------------------------------------------------------------------------- table
def load_table():
    src = open(E14_PATH).read()
    m = re.search(r'_TABLES_B64 = \((.*?)\)\n', src, re.S)
    b = ''.join(re.findall(r'"([^"]*)"', m.group(1)))
    return json.loads(zlib.decompress(base64.b64decode(b)))


def parse_ip(key):
    return tuple(int(x) for x in key.strip("()").split(",") if x.strip())


def wick(mean, sigma, alpha, table):
    """Port of estimator_e14_tree._wick (float64)."""
    Phi = ndtr(alpha)
    phi = np.exp(-0.5 * alpha * alpha) / np.sqrt(2 * np.pi)
    out = {}

    def base(k):
        if k == 0:
            return sigma * phi + mean * Phi
        if k == 1:
            return Phi
        if k == 2:
            return phi / sigma
        if k == 3:
            return -alpha * phi / sigma ** 2
        if k == 4:
            return (alpha * alpha - 1.0) * phi / sigma ** 3
        raise ValueError(k)

    for _key, spec in table["wick"].items():
        k, p = int(spec["k"]), int(spec["p"])
        if spec["branch"] == "poly":
            P1 = np.polynomial.polynomial.polyval(alpha, np.array(spec["P1"], float))
            P2 = np.polynomial.polynomial.polyval(alpha, np.array(spec["P2"], float))
            out[(k, p)] = sigma ** (p - k) * (P1 * phi + P2 * Phi)
        elif spec["branch"] == "base":
            out[(k, p)] = base(k)
        else:  # reduce
            out[(k, p)] = float(spec.get("factorial_p", 1.0)) * base(int(spec.get("k_reduced", k)))
    return out


def get_pk_block(pK, b, d):
    nz = tuple(i for i, x in enumerate(b) if x > 0)
    key = tuple(b[i] for i in nz)
    f = pK[tuple(sorted(key, reverse=True))]
    if len(key) == 2 and key[0] < key[1]:
        f = f.T
    if len(nz) != d:
        f = f[:, None] if nz == (0,) else f[None, :]
    return f


def gaussian_K(table, mu_pre, var, C11, w, off_mask):
    """E14's pK -> K pipeline with every incoming cumulant slice zero (Gaussian order).
    Returns K dict with (1,), (2,), (1,1), (4,), (2,2) (and (3,), (2,1))."""
    n = mu_pre.shape[0]
    Z = np.zeros((n, n))
    z = np.zeros(n)
    S = {(1,): mu_pre, (2,): var, (1, 1): C11, (2, 1): Z, (1, 2): Z,
         (3,): z, (4,): z, (2, 2): Z}
    pK = {}
    for row in table["terms"]:
        ip = tuple(row["int_part"])
        d = len(ip)
        scalar = float(row["scalar"])
        term = None
        for blk in row["blocks"]:
            f = S[tuple(blk["vec"])]
            term = f if term is None else term * f
        if term is None:
            term = scalar
        elif scalar != 1.0:
            term = term * scalar
        for axis, kp in enumerate(row["wick"]):
            wk = w[(int(kp[0]), int(kp[1]))]
            term = term * (wk if d == 1 else (wk[:, None] if axis == 0 else wk[None, :]))
        pK[ip] = term if ip not in pK else pK[ip] + term
    for ip in list(pK):
        if len(ip) == 2 and ip[0] == ip[1]:
            pK[ip] = (pK[ip] + pK[ip].T) * 0.5
    K = {}
    for ip_key, rows in table["pK_to_K"].items():
        ip = parse_ip(ip_key)
        d = len(ip)
        acc = None
        for r in rows:
            prod = None
            for b in r["blocks"]:
                f = get_pk_block(pK, tuple(int(x) for x in b), d)
                prod = f if prod is None else prod * f
            term = prod * float(r["coef"])
            acc = term if acc is None else acc + term
        K[ip] = acc
    return K


# ----------------------------------------------------------------------- tree core
def tree_core(Cpre, Cout, K4, K22, b, a, d, off_mask):
    """Faithful float64 port of estimator_e14_tree._tree_core."""
    C = (Cpre + Cpre.T) * 0.5
    v = np.diagonal(C)
    t = (C * C) @ (b * b)
    Z = (C * (d * b * v)[None, :]) @ C
    Rbb = (C * (b * b)[None, :]) @ C
    Raa = (C * (a * a * v)[None, :]) @ C
    E = ((C * C) * (a * b)[None, :]) @ C
    r = C @ (a * b * v)
    star = C * (d * t)[:, None] * b[None, :]
    edge = C * (a * r)[:, None] * b[None, :] + E * a[:, None] * b[None, :]
    H = (star + star.T + 2.0 * (b[:, None] * b[None, :]) * Z
         + 2.0 * (a[:, None] * a[None, :]) * C * Rbb
         + 2.0 * (b[:, None] * b[None, :]) * Raa
         + 2.0 * (edge + edge.T))
    k22 = ((K22 + K22.T) * 0.5) * off_mask
    anchor = K4 + k22.sum(axis=1)
    H = H * off_mask + np.diag(anchor)
    n = C.shape[0]
    h = np.diagonal(H).sum() / float(n)
    g = 6.0 * (np.diagonal(H) - h) / float(n + 4)
    Cout_s = (Cout + Cout.T) * 0.5
    O = Cout_s * off_mask
    denom = (O * O).sum()
    numerator = ((H * off_mask) * O).sum()
    lam = 0.0 if denom == 0.0 else (6.0 / float(n + 4)) * numerator / denom
    return g, lam, Cout_s, H


def ls_coef(G_off, C_off):
    return float((G_off * C_off).sum() / (C_off * C_off).sum())


# ---------------------------------------------------------------------------- MLPs
def gen_mlp(seed, width=N, depth=DEPTH):
    """He-init MLP, float32 draws (the harness `gen_mlp_legacy` recipe: same
    distribution as the organisers' mint, NOT the same realisation)."""
    rng = np.random.default_rng(seed)
    scale = np.float32(np.sqrt(2.0 / width))
    return [rng.standard_normal((width, width), dtype=np.float32) * scale
            for _ in range(depth)]


def run_mlp(weights, table, verbose=True):
    n = weights[0].shape[0]
    off_mask = 1.0 - np.eye(n)
    mu = np.zeros(n)
    cov = np.eye(n)
    rec = dict(lam_post=[], lam_pre_off=[], lam_pre_diag=[], lam_pre_e14model=[],
               r2_post=[], r2_pre_off=[], mean_var=[], mean_g=[], mean_dG=[],
               cos_pre_off=[])
    prev = None   # (H_off_scaled, g, lam_post, Cout)
    for li, w in enumerate(weights):
        W = np.asarray(w, dtype=np.float64)     # stored (in, out): pre = W^T h
        WT = W.T
        mu_pre = WT @ mu
        cov_pre = WT @ cov @ W if li > 0 else WT @ W
        var = np.maximum(np.diagonal(cov_pre), 1e-12)
        sigma = np.sqrt(var)
        alpha = np.clip(mu_pre / sigma, -ALPHA_CAP, ALPHA_CAP)
        wk = wick(mu_pre, sigma, alpha, table)
        b, a, d = wk[(1, 1)], wk[(2, 1)], wk[(3, 1)]

        # --- 504aldo-aligned LS at THIS layer's pre-activation (uses previous layer's core)
        if prev is not None:
            Hoff_s, g_prev, lam_prev, Cout_prev = prev
            Cpre_off = ((cov_pre + cov_pre.T) * 0.5) * off_mask
            T = WT @ Hoff_s @ W                      # transported off-diagonal core
            T_off = T * off_mask
            rec["lam_pre_off"].append(ls_coef(T_off, Cpre_off))
            rec["cos_pre_off"].append(float((T_off * Cpre_off).sum()
                                            / np.sqrt((T_off ** 2).sum() * (Cpre_off ** 2).sum())))
            D = (WT * g_prev[None, :]) @ W           # W^T diag(g) W
            rec["lam_pre_diag"].append(ls_coef(D * off_mask, Cpre_off))
            # E14's own regenerated model transported: G_post = diag(g) + lam*Cout_off
            M = (WT * g_prev[None, :]) @ W + lam_prev * (WT @ (Cout_prev * off_mask) @ W)
            rec["lam_pre_e14model"].append(ls_coef(M * off_mask, Cpre_off))
            # E14 _layer_tree l.928 (== v29 l.1108): transported kappa4 diagonal
            dG = (WT * WT) @ (g_prev - lam_prev * np.diagonal(Cout_prev)) + lam_prev * var
            rec["mean_dG"].append(float(dG.mean()))
            rec["mean_var"].append(float(var.mean()))

        if li == len(weights) - 1:
            break
        C11 = cov_pre * off_mask
        K = gaussian_K(table, mu_pre, var, C11, wk, off_mask)
        mu_out = K[(1,)]
        if CLOSURE == "deg4":
            e4 = wk[(4, 1)]
            cov_out = (np.diag(K[(2,)]) + C11 * np.outer(b, b) + C11 ** 2 / 2 * np.outer(a, a)
                       + C11 ** 3 / 6 * np.outer(d, d) + C11 ** 4 / 24 * np.outer(e4, e4))
        else:
            cov_out = np.diag(K[(2,)]) + K[(1, 1)]
        g, lam, Cout_s, H = tree_core(cov_pre, cov_out, K[(4,)], K[(2, 2)], b, a, d, off_mask)
        Hoff_s = (6.0 / (n + 4)) * (H * off_mask)
        O = Cout_s * off_mask
        resid = Hoff_s - lam * O
        rec["r2_post"].append(1.0 - (resid ** 2).sum() / (Hoff_s ** 2).sum())
        rec["lam_post"].append(lam)
        rec["mean_g"].append(float(g.mean()))
        prev = (Hoff_s, g, lam, Cout_s)
        if verbose:
            print(f"    layer {li:2d}: lam_post={lam:.5e}  R2_post={rec['r2_post'][-1]:.3f}"
                  f"  mean var_pre={var.mean():.4f}"
                  + (f"  lam_pre_off={rec['lam_pre_off'][-1]:.5e}" if prev and rec['lam_pre_off'] else ""),
                  flush=True)
        mu, cov = mu_out, cov_out
    return rec


def main():
    table = load_table()
    seeds = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4,5,6,7").split(",")]
    aldo = np.load(os.path.join(ALDO_DIR, "lam_prefit_perdump.npy"))     # (8, 15)
    allrec = []
    t0 = time.time()
    for s in seeds:
        print(f"== MLP seed {s}", flush=True)
        t1 = time.time()
        rec = run_mlp(gen_mlp(s), table)
        print(f"   done in {time.time()-t1:.1f}s", flush=True)
        allrec.append(rec)
    tag = "" if CLOSURE == "e14" else "_" + CLOSURE
    keys = ["lam_post", "lam_pre_off", "lam_pre_diag", "lam_pre_e14model", "r2_post",
            "cos_pre_off", "mean_var", "mean_dG", "mean_g"]
    arrs = {k: np.array([r[k] for r in allrec]) for k in keys}
    for k, A in arrs.items():
        np.savetxt(os.path.join(HERE, f"{k}{tag}.csv"), A, delimiter=",",
                   header="rows=MLP seeds " + ",".join(map(str, seeds)) + "; cols=layer index li (E14 layer li / 504aldo LAM[li])")
    np.savez(os.path.join(HERE, f"lambda_results{tag}.npz"), seeds=np.array(seeds), **arrs)

    np.set_printoptions(precision=4, linewidth=220, suppress=False)
    lam_post = arrs["lam_post"]; lam_pre = arrs["lam_pre_off"]; lam_pre_tot = arrs["lam_pre_off"] + arrs["lam_pre_diag"]
    am = aldo.mean(0)
    print("\n=== per-layer table (li = E14 layer index = 504aldo LAM index) ===")
    hdr = f"{'li':>3} {'aldo_mean':>10} {'aldo_sd%':>8} {'0.95xmean':>10} | {'E14_post':>10} {'sd%':>6} {'ratio':>7} | {'E14_pre_off':>11} {'ratio':>7} | {'pre_off+diag':>12} {'ratio':>7} | {'e14model_pre':>12} {'R2post':>7} {'cos_pre':>7}"
    print(hdr)
    rows = []
    for li in range(15):
        p = lam_post[:, li]; q = lam_pre[:, li]; qt = lam_pre_tot[:, li]; em = arrs["lam_pre_e14model"][:, li]
        print(f"{li:3d} {am[li]:10.4e} {100*aldo[:,li].std()/am[li]:8.1f} {0.95*am[li]:10.4e} | "
              f"{p.mean():10.4e} {100*p.std()/abs(p.mean()):6.1f} {p.mean()/am[li]:7.3f} | "
              f"{q.mean():11.4e} {q.mean()/am[li]:7.3f} | {qt.mean():12.4e} {qt.mean()/am[li]:7.3f} | "
              f"{em.mean():12.4e} {arrs['r2_post'][:,li].mean():7.3f} {arrs['cos_pre_off'][:,li].mean():7.3f}")
        rows.append([li, am[li], aldo[:, li].std(), 0.95 * am[li], p.mean(), p.std(), p.mean() / am[li],
                     q.mean(), q.std(), q.mean() / am[li], qt.mean(), qt.mean() / am[li], em.mean(),
                     arrs['r2_post'][:, li].mean(), arrs['cos_pre_off'][:, li].mean()])
    np.savetxt(os.path.join(HERE, f"lam_compare_summary{tag}.csv"), np.array(rows), delimiter=",",
               header="li,aldo_mean,aldo_sd,aldo_095mean,e14_post_mean,e14_post_sd,ratio_post,e14_pre_off_mean,e14_pre_off_sd,ratio_pre_off,e14_pre_offdiag_mean,ratio_pre_offdiag,e14model_pre_mean,r2_post,cos_pre_off")
    for name, A in (("E14_post", lam_post), ("E14_pre_off", lam_pre), ("E14_pre_off+diag", lam_pre_tot)):
        r = A.mean(0) / am
        print(f"\n{name}: ratio to 504aldo mean -- layers 1..14 mean {r[1:].mean():.3f}, sd {r[1:].std():.3f}, "
              f"min {r.min():.3f} max {r.max():.3f}; layer0 ratio {r[0]:.3f}")
        # shape similarity across depth (Pearson between layer profiles)
        c = np.corrcoef(A.mean(0), am)[0, 1]
        c1 = np.corrcoef(A.mean(0)[1:], am[1:])[0, 1]
        print(f"   depth-profile Pearson r (all 15 layers) = {c:.4f}; layers 1..14 = {c1:.4f}")
        # best single scale factor (LS through origin) and residual
        k = (A.mean(0) * am).sum() / (am * am).sum()
        rel = np.sqrt((((A.mean(0) - k * am) / am) ** 2).mean())
        print(f"   LS scale factor E14 = k * aldo: k = {k:.4f}, rms relative residual after scaling = {100*rel:.1f}%")
    print(f"\nTotal wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
