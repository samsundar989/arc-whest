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


#: Which branch produced the last critical value. An artifact that cannot say
#: whether its interval came from scipy or from a fallback is not reproducible,
#: and the old fallback was WRONG: `2.064` is t at df=24, used for every n
#: from 2 to 60. A Skeptic found it latent on 2026-09-10 (scipy is installed in
#: both venvs, so it never fired) and it is fixed before a 1.15x gate uses it.
_LAST_T_CRIT = {"value": None, "source": None}

#: Two-sided 97.5% t critical values, df 1..30. Exact to 4 decimals, so a
#: missing scipy degrades the interval by <0.05% instead of by 12%.
_T975 = (
    12.7062, 4.3027, 3.1824, 2.7764, 2.5706, 2.4469, 2.3646, 2.3060, 2.2622,
    2.2281, 2.2010, 2.1788, 2.1604, 2.1448, 2.1314, 2.1199, 2.1098, 2.1009,
    2.0930, 2.0860, 2.0796, 2.0739, 2.0687, 2.0639, 2.0595, 2.0555, 2.0518,
    2.0484, 2.0452, 2.0423,
)


def _t_crit(df, alpha=0.05):
    """The two-sided critical value, and where it came from."""
    if df < 1:
        return float("nan"), "df<1"
    try:
        from scipy import stats
        return float(stats.t.ppf(1 - alpha / 2, df)), "scipy"
    except Exception:
        pass
    if abs(alpha - 0.05) > 1e-12:
        return 1.96, "normal-approx (alpha != 0.05, no scipy)"
    if df <= len(_T975):
        return _T975[df - 1], "table"
    # Beyond the table, Cornish-Fisher on the normal quantile: error < 0.1%.
    z = 1.959963984540054
    return z + (z ** 3 + z) / (4.0 * df), "cornish-fisher"


def studentised_t(d, alpha=0.05):
    """Paired studentised-t interval. Never percentile, never BCa."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    crit, src = _t_crit(n - 1, alpha)
    _LAST_T_CRIT["value"], _LAST_T_CRIT["source"] = crit, src
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
    import truth_reads

    try:
        block = truth_reads.resolve_block(a.truth, a.block)
    except truth_reads.ReadRefused as exc:
        raise SystemExit(f"REFUSED before truth access: {exc}") from exc

    import torch
    import p2_kprop_bridge as bridge
    from mint_provenance import canonical_mint

    # `bridge._src()` is what puts the pinned upstream checkout on `sys.path`
    # (`p2_kprop_bridge.py:94`), so **nothing may import `mlp_kprop` before it has
    # been called**. The first version of this module imported
    # `mlp_kprop.kprop_harmonic` on the line above and died with
    # `ModuleNotFoundError` two seconds into a queued job. Verify first, import
    # second -- which is also the order that guarantees no upstream symbol is
    # bound before the rev has been checked.
    src = bridge._src()
    v = bridge.verify_upstream(src)
    if not v["pass"]:
        raise SystemExit(f"REFUSED: upstream is not the pinned rev + patch: {v}")

    from mlp_kprop.kprop_harmonic import SIMPLE, mlp_kprop  # noqa: E402

    try:
        z = truth_reads.load_truth(
            block, a.truth, consumer="local_sim/p2_t2_corrector.py",
            purpose="extract exact-K3 chain features from protected truth",
            instrument=os.path.abspath(__file__), result_artifact=a.out, ts=a.ts)
    except truth_reads.ReadRefused as exc:
        raise SystemExit(f"REFUSED before truth access: {exc}") from exc
    seeds = [int(s) for s in z["seeds"]]
    m1, m2 = z["means_r1"], z["means_r2"]
    n_samples = int(z["n_samples"])
    gt_noise = z["gt_noise"] if "gt_noise" in z else None

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
        # PER-NEURON label variances, so the R^2 ceiling can be computed exactly.
        # Without these the ceiling falls back to `gt_noise`, a per-network MEAN
        # over neurons, which the 2026-09-10 T2 Skeptic showed overstates the
        # label noise by 4.06%.
        var_r1=z["var_r1"].astype(np.float64),
        var_r2=z["var_r2"].astype(np.float64),
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


class _Head:
    """A fitted head plus the standardisation it was fitted under.

    STANDARDISATION IS NOT COSMETIC HERE, and its absence is what a positive
    control caught on 2026-09-10. The design matrix mixes `alpha` spanning
    `+-10` with a gate probability in `[0, 1]` and cumulant ratios of order
    `0.1`, and the first layer is a `Tanh`: unstandardised, the net saturates on
    `alpha` and effectively sees one feature. The target is of order `5e-4`, so
    an unstandardised regression is asking a zero-initialised output layer to
    reach a value four orders below its own weight scale. With both fixed the
    head recovers a signal it is given; without them it returned held-out
    `R^2 = -0.4566` on a target that was a KNOWN function of its own inputs.

    Statistics are computed on the TRAINING fold only and applied to the held-out
    fold, so no fold leaks into its own prediction.
    """

    def __init__(self, net, xm, xs, ym, ys):
        self.net, self.xm, self.xs, self.ym, self.ys = net, xm, xs, ym, ys


def _fit_head(X, y, torch, epochs=60, width=32, lr=1e-3, seed=0, dev="cuda"):
    """A small permutation-equivariant head. Rows are neurons; nothing sees an
    ordering, only its own features and pooled context."""
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    F = X.shape[-1]

    Xf = X.reshape(-1, F).astype(np.float32)
    yf = y.reshape(-1, 1).astype(np.float32)
    xm = Xf.mean(axis=0, keepdims=True)
    xs = Xf.std(axis=0, keepdims=True)
    xs[xs < 1e-12] = 1.0                       # a constant column contributes nothing
    ym = float(yf.mean())
    ys = float(yf.std()) or 1.0

    net = torch.nn.Sequential(
        torch.nn.Linear(F, width), torch.nn.Tanh(),
        torch.nn.Linear(width, width), torch.nn.Tanh(),
        torch.nn.Linear(width, 1),
    ).to(dev)                                   # PyTorch's own init, not an ad-hoc 0.05

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.as_tensor((Xf - xm) / xs, device=dev)
    yt = torch.as_tensor((yf - ym) / ys, device=dev)
    n = Xt.shape[0]
    bs = min(1 << 16, n)
    for _ep in range(epochs):
        perm = torch.randperm(n, generator=g).to(dev)
        for k in range(0, n, bs):
            idx = perm[k:k + bs]
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(net(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    return _Head(net, xm, xs, ym, ys)


def _apply(head, X, torch, dev="cuda"):
    """Predict in the TARGET's units, undoing the fit-time standardisation."""
    F = X.shape[-1]
    Xf = X.reshape(-1, F).astype(np.float32)
    with torch.no_grad():
        p = head.net(torch.as_tensor((Xf - head.xm) / head.xs, device=dev))
    out = p.cpu().numpy().astype(np.float64) * head.ys + head.ym
    return out.reshape(X.shape[0], X.shape[1])


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
    dev = a.device
    import truth_reads
    try:
        fit_block = truth_reads.resolve_block(a.fit_features, a.fit_block)
        zf = truth_reads.load_truth(
            fit_block, a.fit_features, consumer="local_sim/p2_t2_corrector.py",
            purpose="fit T2 corrector from truth-bearing extracted features",
            instrument=os.path.abspath(__file__), result_artifact=a.json, ts=a.ts)
    except truth_reads.ReadRefused as exc:
        raise SystemExit(f"REFUSED before feature-label access: {exc}") from exc
    # CONTROLS-ONLY reads the FIT corpus and nothing else. It exists because the
    # 2026-09-10 Skeptic established that P08 had already been read three times
    # (the two broken runs computed `gain` on it), and re-reading a single-use
    # held-out block to repair a control would compound that, not fix it.
    if a.controls_only:
        zr = zf
    else:
        try:
            read_block = truth_reads.resolve_block(a.read_features, a.read_block)
            zr = truth_reads.load_truth(
                read_block, a.read_features,
                consumer="local_sim/p2_t2_corrector.py",
                purpose="evaluate T2 corrector on truth-bearing held-out features",
                instrument=os.path.abspath(__file__), result_artifact=a.json, ts=a.ts)
        except truth_reads.ReadRefused as exc:
            raise SystemExit(f"REFUSED before feature-label access: {exc}") from exc

    import torch

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
    var_tgt = float(np.var(tgtF))
    # The R^2 ceiling this corpus can support.
    #
    # `gt_noise` is the PER-REPLICA noise variance of a per-neuron mean, on the
    # MSE scale, one number per network. The target uses the AVERAGE of two
    # independent replicas, so the noise entering it is `gt_noise / 2` -- and it
    # must be divided by `sigma^2` to reach the target's own normalised units.
    # The first version of this line omitted the factor of two and applied a
    # corpus-mean noise against a per-network sigma. Both are fixed.
    #
    # The contract's own estimate of this ceiling (`~0.40`) was badly wrong and
    # is corrected in the LOG: it assumed a post-ReLU activation variance of
    # order 1, where the measured value is `0.0755`. The real ceiling is `0.944`.
    #
    # SECOND CORRECTION, from the 2026-09-10 T2 Skeptic: `gt_noise` is a
    # per-network MEAN over neurons (`p2_truth.py:400`), and applying it against
    # a PER-NEURON sigma overstates the label noise by 4.06%. The per-neuron
    # variances are on disk as `var_r1`/`var_r2`; use them.
    if "var_r1" in zf and "var_r2" in zf:
        v = 0.5 * (np.asarray(zf["var_r1"], dtype=np.float64)
                   + np.asarray(zf["var_r2"], dtype=np.float64))
        gn_pn = v / float(zf["n_samples"])              # per-neuron, per replica
        noiseF = float(gn_pn.mean())
        noise_in_tgt = float(np.mean((gn_pn / 2.0) / np.maximum(sigF, 1e-30) ** 2))
    elif "gt_noise" in zf:
        gn = np.asarray(zf["gt_noise"], dtype=np.float64).reshape(-1, 1)
        noiseF = float(gn.mean())
        noise_in_tgt = float(np.mean((gn / 2.0) / np.maximum(sigF, 1e-30) ** 2))
    else:
        noiseF, noise_in_tgt = float("nan"), float("nan")
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
    # ---- POSITIVE CONTROLS, and they run BEFORE the real arms ------------- #
    #
    # THE GATE'S ARITHMETIC EQUIVALENT, which makes every R^2 below readable.
    # The score identity gives  gain = E[s^2 t^2] / E[s^2 (c-t)^2], so a gain of
    # G is exactly a SIGMA-WEIGHTED R^2 of 1 - 1/G:
    #
    #     build gate 1.15x  <=>  weighted R^2 = 0.1304
    #     kill  gate 1.10x  <=>  weighted R^2 = 0.0909
    #
    # The first version of this control planted a signal at R^2 ~ 0.94 -- SEVEN
    # TIMES the explained variance the gate turns on -- built from two columns of
    # `own` and using none of `hist` or `ctx`. The 2026-09-10 Skeptic showed that
    # is not a control for the failure mode that would fake this null: it built a
    # proxy lacking the right basis, planted a large signal OUTSIDE its span, and
    # got oof R^2 +0.0178 / gain 1.0164x -- this slice's exact answer, with a big
    # signal present. So the control now plants at GATE STRENGTH, from features
    # spanning all three blocks, with a term the head must compose.
    #
    # A second target is planted that the head should struggle with (a
    # high-frequency oscillation), so "cannot learn" and "nothing to learn" are
    # separated rather than conflated.
    pc = None
    if not a.skip_positive_control:
        Xpc = _design(perF, poolF, "full")
        F_ = Xpc.shape[-1]
        al = perF[:, -1, :, 0].astype(np.float64)
        gt = perF[:, -1, :, 1].astype(np.float64)
        k3 = perF[:, -1, :, 2].astype(np.float64)
        k4 = perF[:, -1, :, 3].astype(np.float64)
        al8 = perF[:, -4, :, 0].astype(np.float64)          # a `hist` column
        cx = np.broadcast_to(poolF[:, :, 0].mean(axis=1, keepdims=True),
                             al.shape).astype(np.float64)   # a `ctx` column
        w2 = (sigF ** 2)                                    # the score's weights

        def _wr2(t, c):
            num = float(np.sum(w2 * (c - t) ** 2))
            den = float(np.sum(w2 * (t - np.average(t, weights=w2)) ** 2))
            return 1.0 - num / den if den > 0 else float("nan")

        def _plant(f, target_wr2):
            """Scale `f` and add noise so its own weighted R^2 is `target_wr2`."""
            f = (f - f.mean()) / (f.std() or 1.0)
            e = np.random.default_rng(11).normal(0.0, 1.0, size=f.shape)
            # weighted R^2 of the noiseless part = var(af) / (var(af)+var(e))
            a_ = math.sqrt(target_wr2 / max(1.0 - target_wr2, 1e-12))
            t = a_ * f + e
            return t * float(np.std(tgtF)) / float(np.std(t))

        smooth = (0.6 * np.tanh(al) + 1.5 * k3 + 0.8 * k4
                  - 0.5 * np.tanh(al * k3) + 0.4 * al8 + 0.7 * cx)
        rough = np.sin(20.0 * al) * k3

        pc = {"gate_weighted_r2": {"build_1.15x": 1 - 1 / BUILD_GATE,
                                   "kill_1.10x": 1 - 1 / KILL_GATE},
              "curve": [], "capacity_probe": None}
        print("  POSITIVE CONTROLS -- the gate is weighted R^2 "
              f"{1 - 1 / BUILD_GATE:.4f} (build) / {1 - 1 / KILL_GATE:.4f} (kill)")
        for lvl in (0.0909, 0.1304, 0.30):
            tp = _plant(smooth, lvl)
            oof = np.zeros_like(tp)
            for k in range(folds):
                te = parts[k]
                tr = np.concatenate([parts[q] for q in range(folds) if q != k])
                net = _fit_head(Xpc[tr], tp[tr], torch, epochs=a.epochs,
                                width=a.head_width, lr=a.lr, seed=k, dev=dev)
                oof[te] = _apply(net, Xpc[te], torch, dev)
            rec_w = _wr2(tp, oof)
            gain_implied = 1.0 / max(1.0 - rec_w, 1e-12)
            pc["curve"].append({"planted_weighted_r2": lvl,
                                "recovered_weighted_r2": rec_w,
                                "implied_gain": gain_implied})
            print(f"     planted wR^2 {lvl:.4f} -> recovered {rec_w:+.4f}"
                  f"   implied gain {gain_implied:.4f}x")
        tp = _plant(rough, 0.30)
        oof = np.zeros_like(tp)
        for k in range(folds):
            te = parts[k]
            tr = np.concatenate([parts[q] for q in range(folds) if q != k])
            net = _fit_head(Xpc[tr], tp[tr], torch, epochs=a.epochs,
                            width=a.head_width, lr=a.lr, seed=k, dev=dev)
            oof[te] = _apply(net, Xpc[te], torch, dev)
        pc["capacity_probe"] = {"planted_weighted_r2": 0.30,
                                "recovered_weighted_r2": _wr2(tp, oof),
                                "target": "sin(20*alpha)*k3n -- deliberately hard"}
        print(f"     CAPACITY PROBE  sin(20a)*k3n planted 0.30 -> recovered "
              f"{pc['capacity_probe']['recovered_weighted_r2']:+.4f}"
              f"   (a LOW number here means the head has a basis limit, not that"
              f" the chain has no signal)")
        # THE PASS CONDITION IS A RECOVERY FRACTION, NOT AN ABSOLUTE GAIN.
        # The first version demanded `implied_gain >= BUILD_GATE` on a signal
        # planted at exactly the weighted R^2 that PRODUCES `BUILD_GATE` -- so
        # only a literally perfect learner could pass, and a 99.5% recovery was
        # reported as a failure. What the control has to establish is that a
        # gate-strength signal would be FOUND, i.e. that recovery is near unity.
        at_gate = [c for c in pc["curve"]
                   if abs(c["planted_weighted_r2"] - (1 - 1 / BUILD_GATE)) < 1e-3]
        frac = (at_gate[0]["recovered_weighted_r2"] / at_gate[0]["planted_weighted_r2"]
                if at_gate else float("nan"))
        pc["recovery_fraction_at_gate"] = frac
        pc["passes"] = bool(frac >= 0.90)
        print(f"     GATE-STRENGTH CONTROL  recovery {frac:.4f} of planted   "
              f"{'PASS -- a 1.15x signal WOULD have been found' if pc['passes'] else '*** FAIL -- a 1.15x signal would NOT have been found; the arms below are unreadable ***'}")
        print()

        if a.controls_only:
            rec_pc = {"artifact_kind": "t2_positive_controls_only",
                      "fit_features": a.fit_features,
                      "note": "P08 IS NOT READ BY THIS RUN. Controls are fitted "
                              "and read entirely within P07.",
                      "label_noise_per_replica": noiseF,
                      "noise_in_target_units": noise_in_tgt,
                      "r2_ceiling_on_this_corpus": r2_ceiling,
                      "positive_control": pc,
                      "t_crit": dict(_LAST_T_CRIT)}
            if a.json:
                out = a.json if os.path.isabs(a.json) else os.path.join(_ROOT, a.json)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "w", encoding="utf-8") as fh:
                    json.dump(rec_pc, fh, indent=2)
                print(f"  artifact  {os.path.relpath(out, _ROOT)}")
            return 0

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
        "positive_control": pc,
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
    ap.add_argument("--block", default="",
                    help="truth block id; filename identity wins and mismatches are refused")
    ap.add_argument("--ts", default="",
                    help="historical event timestamp only; omit for a runtime receipt")
    ap.add_argument("--out", default="")
    ap.add_argument("--fit-features", default="")
    ap.add_argument("--read-features", default="")
    ap.add_argument("--fit-block", default="",
                    help="block for truth-bearing fit features; inferred from feat_<BLOCK>.npz")
    ap.add_argument("--read-block", default="",
                    help="block for truth-bearing read features; inferred from feat_<BLOCK>.npz")
    ap.add_argument("--json", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--head-width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--controls-only", action="store_true",
                    help="Run ONLY the positive controls, on the fit corpus. Does not read the held-out block at all.")
    ap.add_argument("--skip-positive-control", action="store_true",
                    help="DIAGNOSTIC ONLY. The control is what makes a null readable.")
    a = ap.parse_args(argv)
    if a.extract:
        if not (a.truth and a.out):
            raise SystemExit("--extract needs --truth and --out")
        return extract(a)
    if a.fit:
        if not a.fit_features:
            raise SystemExit("--fit needs --fit-features")
        if not a.read_features and not a.controls_only:
            raise SystemExit("--fit needs --read-features (or --controls-only)")
        return fit(a)
    raise SystemExit("choose --extract or --fit")


if __name__ == "__main__":
    raise SystemExit(main())
