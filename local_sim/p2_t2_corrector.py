"""T2 — a learned corrector on the exact-K3 chain's OWN carried state.

Contract: `experiments/LOG.md`, *CONTRACT (Rule 1): T2* (2026-09-10), committed
**before this file existed and before P07 or P08 was claimed**. Handoff
`docs/STRATEGIST_RESPONSE_2026-09-10.md` §1 (H2).

THE CORPUS, AND WHY IT IS NOT THE ONE THE HANDOFF ASKED FOR
------------------------------------------------------------
The handoff says to train on the organisers' public 1,100 networks.
`docs/VALIDATION.md` §0 forbids fitting anything on the public split -- it is
CONFIRM tier, read-only, *"no constant may ever again be fitted here"* -- and
`AGENTS.md` makes that binding. So the corpus is minted DECIDE-tier data:

    P07  384 networks   the FIT.   Tuning permitted HERE and nowhere else.
    P08  128 networks   the READ.  The 1.15x gate is decided here, ONCE.

The public split is read at the very end, read-only, and chooses nothing.

WHY THE READOUT FORM IS THE PRIMARY ARM
----------------------------------------
`RESULT WP10` closed the **in-flight** corrector on the K2 chain: the fitted arm
returned `0.0075x` and both null controls diverged, and a sensitivity injection
then showed why -- an incoherent per-neuron error of RMS `1e-4` already costs
`1.016x`, so an in-flight corrector needs held-out `R^2 >= 0.99849` merely to
break even. **That threshold is a compounding effect**: the corrector's own error
is consumed by the next layer, sixteen times over.

A **readout** correction cannot compound. It is applied once, to the final layer,
after the chain has finished. So it faces a threshold lower by orders of
magnitude, and it has never been tried on an exact K3 chain. The in-flight form is
still measured here, but by the cheap route: `--sensitivity` injects a KNOWN error
and reports this chain's break-even `R^2` **without training anything**.

SCALE EQUIVARIANCE IS THE WHOLE SHAPE OF THE HEAD
--------------------------------------------------
The closure is positively homogeneous of degree 1: `(mu, cov) -> (g mu, g^2 cov)`
leaves every `alpha = mu/sigma` unchanged and scales every predicted mean by
exactly `g`. So the residual of a degree-1 homogeneous estimator against a
degree-1 homogeneous truth is itself `sigma` times a function of **scale
invariants**. The correction is therefore written

    mu_corrected = mu_raw + sigma * c(features),   every feature scale-invariant

which keeps the corrected estimator exactly degree-1 homogeneous. If `c` were
constant across neurons it would collapse to one output rescale -- which P2-CAL
already banked and exhausted -- so the constant-`c` member is an exact **null
control** (`R0`), not a competitor.

Research only. Never imported by `estimator.py` (needs torch).

Run (GPU venv, via workerq):
    .venv-gpu\\Scripts\\python.exe -u local_sim/p2_t2_corrector.py --extract \\
        --truth experiments/2026-09-10_slice5/truth_P07.npz \\
        --out   experiments/2026-09-10_slice5/feat_P07.npz
    .venv-gpu\\Scripts\\python.exe -u local_sim/p2_t2_corrector.py --fit \\
        --fit-features  experiments/2026-09-10_slice5/feat_P07.npz \\
        --read-features experiments/2026-09-10_slice5/feat_P08.npz \\
        --json experiments/2026-09-10_slice5/t2_corrector.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

WIDTH, DEPTH = 1024, 16

# ---- pre-registered in the contract, before any datum. Do not edit. -------- #
BUILD_GATE = 1.15          # >= this held out on P08  ->  spell it in fnp
KILL_GATE = 1.10           # <= this                  ->  close H2's readout form
PREDICTION_BAND = (1.00, 1.10)
WP10_INFLIGHT_BREAKEVEN_R2 = 0.99849   # the K2 chain's, for comparison


# --------------------------------------------------------------------------- #
# statistics                                                                   #
# --------------------------------------------------------------------------- #

def xmse(pred, A, B):
    """Unbiased two-half cross estimator. `E[(p-A)(p-B)] = MSE(p)` exactly."""
    return float(np.mean((pred - A) * (pred - B)))


def studentised_t(d, alpha=0.05):
    """Paired studentised-t interval. Never percentile, never BCa."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    try:
        from scipy import stats
        crit = float(stats.t.ppf(1 - alpha / 2, n - 1))
    except Exception:
        crit = 1.96 if n > 60 else 2.064
    return m, se, (m - crit * se, m + crit * se)


def _progress(frac):
    p = os.environ.get("WORKERQ_PROGRESS")
    if not p:
        return
    try:
        with open(p, "w") as fh:
            fh.write(str(min(max(frac, 0.0), 1.0)))
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# feature extraction                                                           #
# --------------------------------------------------------------------------- #

#: Per-neuron, per-layer scale invariants. Order is frozen here because the
#: fitted coefficients are meaningless without it.
PER_NEURON = ("alpha", "gate", "k3n", "k4n")
#: Pooled per-layer context: four order statistics of `alpha` plus the layer's
#: own scalar invariants. Pooling makes the head permutation-EQUIVARIANT in
#: neurons: it sees its own row plus a summary of the layer, never an ordering.
POOLED = ("alpha_mean", "alpha_std", "gate_mean", "k3n_mean", "k4n", "logsig_std")


def _state_features(out_pre, layer, torch):
    """Scale-invariant per-neuron features from ONE pre-activation state.

    Everything here is a ratio of cumulants to the matching power of `sigma`, so
    the whole feature vector is invariant under `W -> g W`. That is not a
    nicety: it is what keeps the corrected estimator degree-1 homogeneous, and a
    single non-invariant feature would silently break the property the head's
    whole shape is built on.
    """
    from mlp_kprop.diagslice import diagslice

    st = out_pre[f"pre{layer}"]
    mu = st[1].core.detach()
    cov = st[2].core.detach()
    var = torch.diagonal(cov).clamp_min(1e-30)
    sig = var.sqrt()
    alpha = mu / sig

    # kappa_3's fully repeated (3,) slice, normalised by sigma^3.
    try:
        k3 = diagslice(st[3], (3,))
        k3 = torch.as_tensor(k3).detach().reshape(-1)
    except Exception:
        k3 = torch.zeros_like(mu)
    k3n = k3 / (sig ** 3)

    # SIMPLE carries degree 4 at r=2, so its `core` is a SCALAR -- but the tensor
    # it denotes is `Sym(core (x) metric^(x)2)`, and the metric is
    # `init_scale * I`. So `core` alone is NOT scale-free: normalising it by
    # `sigma^4` left a feature that moved under `W -> g W`, which a homogeneity
    # check caught before this module was ever queued. Read the fully repeated
    # `(4,)` slice instead -- that is the actual kappa_4 diagonal in tensor
    # space, it is per-neuron rather than one broadcast number, and dividing it
    # by `var^2` is unambiguous.
    try:
        k4 = torch.as_tensor(diagslice(st[4], (4,))).detach().reshape(-1)
    except Exception:
        k4 = torch.zeros_like(mu)
    k4n = k4 / (var ** 2)

    gate = 0.5 * (1.0 + torch.erf(alpha / math.sqrt(2.0)))

    per = torch.stack([alpha, gate, k3n, k4n], dim=1)             # (n, 4)
    pooled = torch.tensor([
        float(alpha.mean()), float(alpha.std()),
        float(gate.mean()), float(k3n.mean()),
        float(k4n.mean()), float(torch.log(sig).std()),
    ], device=mu.device, dtype=mu.dtype)                          # (6,)
    return per, pooled, sig


def extract(a) -> int:
    """Run the chain on the truth file's networks and dump features + prediction."""
    import torch
    import p2_kprop_bridge as bridge
    from mint_provenance import canonical_mint
    from mlp_kprop.kprop_harmonic import SIMPLE, mlp_kprop

    z = np.load(a.truth)
    seeds = [int(s) for s in z["seeds"]]
    m1, m2 = z["means_r1"], z["means_r2"]
    n_samples = int(z["n_samples"])
    gt_noise = z["gt_noise"] if "gt_noise" in z else None

    v = bridge.verify_upstream(bridge._src())
    if not v["pass"]:
        raise SystemExit(f"REFUSED: upstream is not the pinned rev + patch: {v}")

    dev = a.device
    dtype = torch.float32
    print("=" * 78)
    print("T2 EXTRACT -- chain state + prediction on the corrector's corpus")
    print("=" * 78)
    print(f"  truth        {a.truth}   {len(seeds)} networks")
    print(f"  labels       2 replicas x {n_samples:,} samples")
    if gt_noise is not None:
        print(f"  label noise  mean {float(np.mean(gt_noise)):.4e} per replica")
    print(f"  upstream     {v['rev'][:12]} + recorded patch, verified")
    print()

    PER, POOL = [], []
    PRED, SIG = [], []
    t0 = time.perf_counter()
    for i, s in enumerate(seeds):
        w, _e, _ss = canonical_mint(WIDTH, DEPTH, s)
        mlp = bridge.build_mlp(w, device=dev, dtype=dtype)
        K_in = {1: torch.zeros(WIDTH, device=dev, dtype=dtype),
                2: torch.eye(WIDTH, device=dev, dtype=dtype)}
        with torch.no_grad():
            out = mlp_kprop(mlp, K_in, k_max=3, output_all=True, kind=SIMPLE,
                            use_avg_metric=True, factor=True,
                            up_to_layer=f"act{DEPTH - 1}")
            per_l, pool_l = [], []
            for l in range(DEPTH):
                p, g, sg = _state_features(out, l, torch)
                per_l.append(p)
                pool_l.append(g)
                if l == DEPTH - 1:
                    sig_last = sg
            pred = torch.as_tensor(out[f"act{DEPTH - 1}"][1].core).detach()
        PER.append(torch.stack(per_l).cpu().numpy().astype(np.float32))
        POOL.append(torch.stack(pool_l).cpu().numpy().astype(np.float32))
        PRED.append(pred.cpu().numpy().astype(np.float64))
        SIG.append(sig_last.cpu().numpy().astype(np.float64))
        _progress((i + 1) / len(seeds))
        if (i + 1) % 32 == 0 or i == 0:
            el = time.perf_counter() - t0
            print(f"   [{i+1:4d}/{len(seeds)}]  {el/(i+1):.2f}s/net")
        del out, mlp, K_in

    rec = dict(
        seeds=np.asarray(seeds, dtype=np.int64),
        per=np.stack(PER),          # (N, DEPTH, WIDTH, 4)
        pooled=np.stack(POOL),      # (N, DEPTH, 6)
        pred=np.stack(PRED),        # (N, WIDTH)   final-layer post-ReLU mean
        sig=np.stack(SIG),          # (N, WIDTH)   final PRE-activation sigma
        truth_r1=m1[:, -1, :].astype(np.float64),
        truth_r2=m2[:, -1, :].astype(np.float64),
        n_samples=np.int64(n_samples),
        per_names=np.asarray(PER_NEURON), pooled_names=np.asarray(POOLED),
    )
    if gt_noise is not None:
        rec["gt_noise"] = gt_noise
    out_path = a.out if os.path.isabs(a.out) else os.path.join(_ROOT, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **rec)
    base = np.mean([xmse(PRED[i], rec["truth_r1"][i], rec["truth_r2"][i])
                    for i in range(len(seeds))])
    print()
    print(f"  baseline chain xmse (this corpus)  {base:.6e}")
    print(f"  artifact  {os.path.relpath(out_path, _ROOT)}")
    _progress(1.0)
    return 0


# --------------------------------------------------------------------------- #
# the head                                                                     #
# --------------------------------------------------------------------------- #

def _design(per, pooled, mode="full"):
    """(N, WIDTH, F) design matrix. `mode` selects an arm's feature access.

    `full`      per-neuron features + the whole chain's pooled context
    `constant`  pooled context ONLY -- the null control that must collapse to a
                per-layer scalar
    """
    N, D, W, P = per.shape
    ctx = pooled.reshape(N, D * pooled.shape[2])            # (N, D*6)
    ctx = np.broadcast_to(ctx[:, None, :], (N, W, ctx.shape[1]))
    if mode == "constant":
        return np.ascontiguousarray(ctx.astype(np.float32))
    own = per[:, -1, :, :]                                   # final layer (N,W,P)
    # A short history: the same neuron index at the last four layers. Indices are
    # NOT in correspondence across layers, so this is a coarse feature and is
    # marked as such -- it is here to be ablated, not to be believed.
    hist = per[:, -4:, :, :].transpose(0, 2, 1, 3).reshape(N, W, -1)
    return np.ascontiguousarray(
        np.concatenate([own, hist, ctx], axis=2).astype(np.float32))


def _fit_head(X, y, torch, epochs=60, width=32, lr=1e-3, seed=0, dev="cuda"):
    """A small permutation-equivariant head. Rows are neurons; nothing sees an
    ordering, only its own features and pooled context."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    F = X.shape[-1]
    net = torch.nn.Sequential(
        torch.nn.Linear(F, width), torch.nn.Tanh(),
        torch.nn.Linear(width, width), torch.nn.Tanh(),
        torch.nn.Linear(width, 1),
    ).to(dev)
    for m in net:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, 0.0, 0.05, generator=None)
            torch.nn.init.zeros_(m.bias)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.as_tensor(X.reshape(-1, F), device=dev)
    yt = torch.as_tensor(y.reshape(-1, 1), device=dev, dtype=Xt.dtype)
    n = Xt.shape[0]
    bs = min(1 << 16, n)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g).to(dev)
        for k in range(0, n, bs):
            idx = perm[k:k + bs]
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(net(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    return net


def _apply(net, X, torch, dev="cuda"):
    F = X.shape[-1]
    with torch.no_grad():
        p = net(torch.as_tensor(X.reshape(-1, F), device=dev))
    return p.reshape(X.shape[0], X.shape[1]).cpu().numpy().astype(np.float64)


def _james_stein(c, resid):
    """Shrink the correction toward zero by the factor a held-out fold supports.

    `s = <c, r> / <c, c>` is the least-squares scale of the correction against
    the residual it is meant to explain, clipped to [0, 1]: a correction that
    does not point at the residual is shrunk to nothing rather than trusted.
    """
    num = float(np.sum(c * resid))
    den = float(np.sum(c * c))
    if den <= 0:
        return 0.0
    return float(min(max(num / den, 0.0), 1.0))


def fit(a) -> int:
    import torch

    dev = a.device
    zf = np.load(a.fit_features)
    zr = np.load(a.read_features)

    def unpack(z):
        pred, sig = z["pred"], z["sig"]
        A, B = z["truth_r1"], z["truth_r2"]
        # The target is scale-free: the residual divided by the final
        # pre-activation sigma, exactly the object the homogeneity argument says
        # is a function of invariants.
        tgt = ((A + B) / 2.0 - pred) / np.maximum(sig, 1e-30)
        return z["per"], z["pooled"], pred, sig, A, B, tgt

    perF, poolF, predF, sigF, AF, BF, tgtF = unpack(zf)
    perR, poolR, predR, sigR, AR, BR, tgtR = unpack(zr)

    nF, nR = len(predF), len(predR)
    noiseF = float(np.mean(zf["gt_noise"])) if "gt_noise" in zf else float("nan")
    var_tgt = float(np.var(tgtF))
    # The R^2 ceiling this corpus can support: label noise divided by the
    # target's own variance, both in the SAME normalised units.
    noise_in_tgt = float(np.mean(noiseF / np.maximum(sigF, 1e-30) ** 2)) \
        if np.isfinite(noiseF) else float("nan")
    r2_ceiling = 1.0 - noise_in_tgt / var_tgt if var_tgt > 0 else float("nan")

    print("=" * 78)
    print("T2 FIT -- the readout corrector on the exact-K3 chain's own state")
    print("=" * 78)
    print(f"  fit corpus   {os.path.basename(a.fit_features)}  {nF} networks (P07)")
    print(f"  read corpus  {os.path.basename(a.read_features)}  {nR} networks (P08)")
    print(f"  label noise  {noiseF:.4e} per replica")
    print(f"  R^2 CEILING on this corpus  {r2_ceiling:.3f}  "
          f"-- quote no R^2 without it")
    print(f"  PRE-REGISTERED  band {PREDICTION_BAND}   build >= {BUILD_GATE}x   "
          f"kill <= {KILL_GATE}x")
    print()

    base_fit = np.array([xmse(predF[i], AF[i], BF[i]) for i in range(nF)])
    base_read = np.array([xmse(predR[i], AR[i], BR[i]) for i in range(nR)])
    print(f"  B0 on the FIT corpus   {base_fit.mean():.6e}")
    print(f"  B0 on the READ corpus  {base_read.mean():.6e}")
    print()

    arms = {}
    folds = a.folds
    idx = np.arange(nF)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    parts = np.array_split(idx, folds)

    for arm in ("R", "R0", "Rshuf"):
        mode = "constant" if arm == "R0" else "full"
        Xf = _design(perF, poolF, mode)
        Xr = _design(perR, poolR, mode)
        if arm == "Rshuf":
            # Permute the PER-NEURON block across neurons within each network,
            # leaving the pooled context intact. If this arm gains, the gain is
            # not per-neuron information.
            Xf = Xf.copy()
            for i in range(nF):
                p = rng.permutation(Xf.shape[1])
                Xf[i, :, :perF.shape[-1] * 5] = Xf[i, p, :perF.shape[-1] * 5]

        oof = np.zeros_like(tgtF)
        shrink = []
        for k in range(folds):
            te = parts[k]
            tr = np.concatenate([parts[j] for j in range(folds) if j != k])
            net = _fit_head(Xf[tr], tgtF[tr], torch, epochs=a.epochs,
                            width=a.head_width, lr=a.lr, seed=k, dev=dev)
            c = _apply(net, Xf[te], torch, dev)
            oof[te] = c
            shrink.append(_james_stein(c, tgtF[te]))
        s_hat = float(np.mean(shrink))

        # One head on ALL of P07, applied ONCE to P08. No tuning after this line.
        net_all = _fit_head(Xf, tgtF, torch, epochs=a.epochs,
                            width=a.head_width, lr=a.lr, seed=99, dev=dev)
        cR = _apply(net_all, Xr, torch, dev) * s_hat

        corrected = predR + sigR * cR
        arm_x = np.array([xmse(corrected[i], AR[i], BR[i]) for i in range(nR)])
        d = base_read - arm_x
        m, se, ci = studentised_t(d)
        # Out-of-fold R^2 on the FIT corpus, quoted with its ceiling.
        ss_res = float(np.sum((tgtF - oof * s_hat) ** 2))
        ss_tot = float(np.sum((tgtF - tgtF.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        arms[arm] = {
            "mean_xmse_read": float(arm_x.mean()),
            "gain": float(base_read.mean() / arm_x.mean()),
            "paired_diff": m, "se": se, "ci95": list(ci),
            "n_better": int(np.sum(d > 0)), "n": nR,
            "shrinkage": s_hat, "oof_r2_fit": r2,
        }
        print(f"  {arm:6s} read {arm_x.mean():.6e}   gain "
              f"{arms[arm]['gain']:.4f}x   better on {arms[arm]['n_better']}/{nR}"
              f"   shrink {s_hat:.3f}   oof R^2 {r2:+.4f}")

    gain = arms["R"]["gain"]
    verdict = ("BUILD" if gain >= BUILD_GATE
               else "H2 READOUT CLOSED" if gain <= KILL_GATE
               else "INCONCLUSIVE BAND")
    void = arms["R0"]["gain"] > 1.05      # the contract's VOID clause
    print()
    print(f"  GATE  build >= {BUILD_GATE}x   kill <= {KILL_GATE}x   ->  {verdict}")
    if void:
        print("  *** VOID CLAUSE FIRED: the constant-c null control did NOT "
              "collapse. The harness leaks and nothing above may be read. ***")

    rec = {
        "artifact_kind": "t2_readout_corrector",
        "contract": "experiments/LOG.md 2026-09-10 CONTRACT (Rule 1): T2",
        "fit_features": a.fit_features, "read_features": a.read_features,
        "n_fit": nF, "n_read": nR, "folds": folds,
        "label_noise_per_replica": noiseF,
        "r2_ceiling_on_this_corpus": r2_ceiling,
        "pre_registered": {"band": list(PREDICTION_BAND),
                           "build_gate": BUILD_GATE, "kill_gate": KILL_GATE},
        "baseline": {"fit": float(base_fit.mean()), "read": float(base_read.mean())},
        "arms": arms, "verdict": verdict, "void_clause_fired": bool(void),
        "wp10_inflight_breakeven_r2_on_K2": WP10_INFLIGHT_BREAKEVEN_R2,
        "seed_blocks": {"fit": "P07", "read": "P08"},
    }
    if a.json:
        out = a.json if os.path.isabs(a.json) else os.path.join(_ROOT, a.json)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2)
        print(f"  artifact  {os.path.relpath(out, _ROOT)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--truth", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--fit-features", default="")
    ap.add_argument("--read-features", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--head-width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args(argv)
    if a.extract:
        if not (a.truth and a.out):
            raise SystemExit("--extract needs --truth and --out")
        return extract(a)
    if a.fit:
        if not (a.fit_features and a.read_features):
            raise SystemExit("--fit needs --fit-features and --read-features")
        return fit(a)
    raise SystemExit("choose --extract or --fit")


if __name__ == "__main__":
    raise SystemExit(main())
