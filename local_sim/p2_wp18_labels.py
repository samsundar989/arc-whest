"""WP18 / Arm L, step B1: the ORACLE LABELS `(E3, E21)` and the `E3` feature set.

Contract: `experiments/LOG.md`, 2026-09-03, "PRE-REGISTRATION (Rule 1): CONTRACT
WP18", § B, committed at `49e2e42` **before this file existed**. Seed block
`P05`, DESIGN tier, tuning permitted here and nowhere else in the project.

Research only. **Never imported by `estimator.py`**: this needs torch and the
grader sandbox has none.

WHAT A LABEL IS HERE
--------------------
The regenerating flow rebuilds `kappa_3` from its own two repeated slices every
layer and throws the all-distinct block away. What it throws away, read at
pre-layer `p`, is exactly

    E3_p  = uncapped(3,)_p   - regen(3,)_p        an n-vector
    E21_p = uncapped(2,1)_p  - regen(2,1)_p       n x n

and both are computed here by the SAME window machinery `G-M-DEPTH` used, whose
`--selftest` S1/S2 assert that `k = 0` is the regenerating flow and `k >= p` is
the uncapped one. This module re-asserts both endpoints per run rather than
trusting them (`--endpoint-check`), because `CONTRACT WP18`'s prediction B1 is
written on them and a labeller that is silently computing something else would
produce a beautiful `R^2` for the wrong target.

WHY `E3` IS THE CHEAP LEG, AND ONE MECHANISTIC FACT THAT SHAPES ITS FEATURES
---------------------------------------------------------------------------
`E3` is an `n`-vector, so a *pointwise* map -- neuron `i`'s correction from
neuron `i`'s own carried scalars -- is permutation-equivariant **by
construction**. Equivariance is free rather than imposed, which is the whole
reason `CONTRACT WP18` puts `E3` first.

And a fact worth writing down because it decided the feature list: the fresh
all-distinct block's OWN `(3,)` slice is **exactly zero**. Its factors are
`(w1 . C, 3 I, (w2 (x) w1 . C)^T)` with `C = zero_repeated(Sigma)`, so

    slice3_i = sum_j (w1_i C_ij)(3 delta_ij)(w2_j w1_i C_ji) = 3 w1_i^2 w2_i C_ii^2 = 0

because `C`'s diagonal is zero by definition. **The discarded block therefore
contributes nothing to the `(3,)` slice at the layer that generates it; it
reaches `(3,)` only after a linear step mixes the indices.** So the features are
taken from the POST-LINEAR state at pre-layer `p` -- where `W`'s effect is
already folded into `Sigma_p = W Sigma_{p-1} W^T` -- and they are row
contractions of that `Sigma`, not of the pre-linear one. A feature set built on
the pre-nonlinearity state of the generating layer would have been looking in
the wrong place, and the zero above is what says so.

WHAT THIS MODULE DOES NOT DO
----------------------------
* **It stores no dense `E21`.** 128 nets x 11 layers x `1024^2` f32 is `4.6 GB`
  and the fit does not need it: for a basis that is LINEAR in its coefficients,
  ridge needs only the Gram matrix `<T_t, T_s>`, the moments `<T_t, E21>` and
  `||E21||^2`. Those are computed by `--e21-moments` in a SECOND pass, once the
  `E21` basis is declared, which is also the honest order -- pre-declaring the
  basis and then computing its moments cannot be reversed into a basis chosen
  after seeing the target. This pass stores `||E21||^2` and the top singular
  values as diagnostics only.
* **It fits nothing.** `p2_wp18_fit.py` does the ridge and the leave-networks-out.
* **It reads no truth and no CONFIRM split.** `E3`/`E21` are differences between
  two of our own flows; the minted networks are DESIGN tier.

Run (GPU venv; heavy -- submit through gpuq, never launch):
    .venv-gpu\\Scripts\\python.exe local_sim/p2_wp18_labels.py --selftest --width 128 --depth 4
    .venv-gpu\\Scripts\\python.exe local_sim/p2_wp18_labels.py --labels --nets 128 \\
        --out experiments/2026-09-03_wp18_armL/labels_P05.npz
"""

from __future__ import annotations

import argparse
import json
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

#: `CONTRACT WP15`'s composition rule, inherited unchanged by `CONTRACT WP18`:
#: layers 1-3 are excluded because the flow has barely accumulated a `kappa_3`
#: there; layer 15 is excluded because nothing downstream reads its `(2,1)`
#: slice. The pre-layers actually read are `l + 1` for `l` in here.
GATE_LAYERS = tuple(range(4, 15))

#: `E3`'s pointwise feature set, DECLARED HERE AND FROZEN before any label
#: exists. Every entry is an `n`-vector computed from the regenerating flow's
#: own POST-LINEAR state at pre-layer `p` -- so a deployed map needs nothing the
#: flow does not already hold. Order is load-bearing: `p2_wp18_fit.py` indexes
#: coefficients by it and reports them by name.
#:
#: `C = zero_repeated(Sigma)` throughout (the off-diagonal covariance), `var =
#: diag(Sigma)`, `sd = sqrt(var)`, `alpha = mu / sd`, `w_k = relu_wick_coef(mu,
#: var, k, 1)` from upstream -- never transcribed, which is the WP16 lesson.
E3_FEATURES = (
    "one",             # intercept
    "mu",
    "sd",
    "alpha",           # mu / sd -- the Wick coefficients' own argument
    "S3",              # the (3,) slice the flow already carries
    "w1", "w2", "w3",  # upstream's relu Wick coefficients
    "rowC2",           # sum_j C_ij^2
    "rowC3",           # sum_j C_ij^3
    "rowC2w1",         # sum_j C_ij^2 w1_j
    "rowC2w2",         # sum_j C_ij^2 w2_j
    "rowC2sd",         # sum_j C_ij^2 sd_j
    "rowC2alpha",      # sum_j C_ij^2 alpha_j
    "rowS21",          # sum_j S21_ij
    "colS21",          # sum_j S21_ji
    "w1_rowC2",        # w1_i * rowC2_i
    "w2_rowC2",        # w2_i * rowC2_i
    "w1sq_rowC2",      # w1_i^2 * rowC2_i
    "w1w2_rowC2",      # w1_i w2_i * rowC2_i   -- the fresh block's own shape
    "alpha_rowC2",
    "alphasq_rowC2",
    "S3_w1",           # S3_i w1_i
    "sd3",             # sd_i^3, the natural scale of a third cumulant
)

#: How many singular values of `E21` to keep as a DIAGNOSTIC. Not a fit target:
#: `CONTRACT WP15` fixed Arm M's affordability horizon at `r <= 192`, so 256 is
#: enough to see whether the deployable rank captures the energy.
E21_SV_KEEP = 256


def _upstream():
    import p2_kprop_bridge as bridge

    src = bridge._src()
    ver = bridge.verify_upstream(src)
    if not ver["pass"]:
        raise SystemExit(
            f"REFUSED: upstream is not rev {bridge.UPSTREAM_REV[:8]} plus exactly "
            f"the recorded patch: {ver}")
    return bridge


# ==========================================================================
# the state and the labels at one pre-layer
# ==========================================================================

def regen_state(mlp, snap, p):
    """The regenerating flow's POST-LINEAR state at pre-layer `p`.

    Identical in construction to `p2_wp15_depth.slices_window(..., k=0)` -- it
    restarts from `snap[p - 1]` and takes layer `p`'s linear step without its
    nonlinearity -- but returns `(mu, Sigma, S3, S21)` as well as the slices,
    because the features live on the same state the slices are read from and
    reading them from two different places is how a subtle mismatch gets in.
    """
    import torch

    from mlp_kprop.kprop_harmonic import clone_tower

    import p2_wp15_depth as wp15

    K = clone_tower(snap[p - 1])
    with torch.no_grad():
        K = wp15._run_layers(mlp, K, p, p + 1, regenerate=False, read_pre=p)
    ds = K[3].get_repeated()
    mu = (K[1].core if hasattr(K[1], "core") else K[1]).clone()
    Sig = (K[2].core if hasattr(K[2], "core") else K[2]).clone()
    return (ds.slices[(3,)].clone(), ds.slices[(2, 1)].clone(), mu, Sig)


def e3_features(mu, Sig, S3, S21):
    """The frozen `E3_FEATURES`, as an `(n, n_feat)` float64 tensor.

    Everything is `O(n^2)` at worst (four `Sigma`-row contractions) and the
    deployed version of each is one `fnp` reduction, which is why `CONTRACT
    WP18`'s prediction B7 puts the whole map under `1e8` FLOPs.
    """
    import torch
    from mlp_kprop.diagslice import zero_repeated
    from mlp_kprop.wick import relu_wick_coef

    var = Sig.diagonal()
    w1 = relu_wick_coef(mean=mu, var=var, k=1, p=1)
    w2 = relu_wick_coef(mean=mu, var=var, k=2, p=1)
    w3 = relu_wick_coef(mean=mu, var=var, k=3, p=1)

    mu, var = mu.double(), var.double()
    w1, w2, w3 = w1.double(), w2.double(), w3.double()
    S3, S21 = S3.double(), S21.double()
    C = zero_repeated(Sig).double()

    sd = var.clamp_min(1e-300).sqrt()
    alpha = mu / sd
    C2, C3 = C * C, C * C * C
    rowC2 = C2.sum(dim=1)
    cols = {
        "one": torch.ones_like(mu),
        "mu": mu,
        "sd": sd,
        "alpha": alpha,
        "S3": S3,
        "w1": w1, "w2": w2, "w3": w3,
        "rowC2": rowC2,
        "rowC3": C3.sum(dim=1),
        "rowC2w1": C2 @ w1,
        "rowC2w2": C2 @ w2,
        "rowC2sd": C2 @ sd,
        "rowC2alpha": C2 @ alpha,
        "rowS21": S21.sum(dim=1),
        "colS21": S21.sum(dim=0),
        "w1_rowC2": w1 * rowC2,
        "w2_rowC2": w2 * rowC2,
        "w1sq_rowC2": w1 * w1 * rowC2,
        "w1w2_rowC2": w1 * w2 * rowC2,
        "alpha_rowC2": alpha * rowC2,
        "alphasq_rowC2": alpha * alpha * rowC2,
        "S3_w1": S3 * w1,
        "sd3": sd * sd * sd,
    }
    missing = set(E3_FEATURES) - set(cols)
    extra = set(cols) - set(E3_FEATURES)
    if missing or extra:
        raise AssertionError(
            f"feature set drifted from the frozen declaration: "
            f"missing {sorted(missing)}, extra {sorted(extra)}")
    return torch.stack([cols[name] for name in E3_FEATURES], dim=1)


# ==========================================================================
# --labels
# ==========================================================================

def cmd_labels(a) -> int:
    """Per net, per pre-layer: exact `E3`, the frozen features, `E21` diagnostics."""
    import torch

    import p2_wp15_depth as wp15

    dtype = torch.float64 if a.dtype == "f64" else torch.float32
    pre = sorted({l + 1 for l in GATE_LAYERS if l + 1 <= a.depth - 1})
    n_feat = len(E3_FEATURES)

    E3 = np.zeros((len(a.seeds), len(pre), a.width), dtype=np.float64)
    FEAT = np.zeros((len(a.seeds), len(pre), a.width, n_feat), dtype=np.float64)
    E21_SQ = np.zeros((len(a.seeds), len(pre)), dtype=np.float64)
    E3_SQ = np.zeros((len(a.seeds), len(pre)), dtype=np.float64)
    REF3_SQ = np.zeros((len(a.seeds), len(pre)), dtype=np.float64)
    REF21_SQ = np.zeros((len(a.seeds), len(pre)), dtype=np.float64)
    SV = np.zeros((len(a.seeds), len(pre), min(E21_SV_KEEP, a.width)),
                  dtype=np.float64)
    endpoints = []
    prog = os.environ.get("WORKERQ_PROGRESS")

    for i, seed in enumerate(a.seeds):
        t0 = time.perf_counter()
        mlp, K_in = wp15._build(seed, a.width, a.depth, a.device, dtype)
        unc = wp15.uncapped_slices(mlp, K_in, a.depth, pre)
        snap = wp15.regen_snapshots(mlp, K_in, a.depth)

        for j, p in enumerate(pre):
            r3, r21 = unc[p][0].double(), unc[p][1].double()
            s3, s21, mu, Sig = regen_state(mlp, snap, p)
            e3 = r3 - s3.double()
            e21 = r21 - s21.double()

            E3[i, j] = e3.cpu().numpy()
            FEAT[i, j] = e3_features(mu, Sig, s3, s21).cpu().numpy()
            E3_SQ[i, j] = float((e3 * e3).sum())
            E21_SQ[i, j] = float((e21 * e21).sum())
            REF3_SQ[i, j] = float((r3 * r3).sum())
            REF21_SQ[i, j] = float((r21 * r21).sum())
            sv = torch.linalg.svdvals(e21)[:SV.shape[2]]
            SV[i, j, :sv.numel()] = sv.cpu().numpy()

            # -- B1's endpoint check, on the FIRST net only (it is a form check
            #    on the labeller, not a per-net property, and it costs a full
            #    extra window flow per layer).
            if i == 0 and a.endpoint_check:
                w0 = wp15.slices_window(mlp, snap, p, 0)[1].double()
                wfull = wp15.slices_window(mlp, snap, p, a.depth)[1].double()
                den = max(float((r21 - w0).norm() ** 2), 1e-300)
                endpoints.append({
                    "pre_layer": p,
                    "explained_k0": 1.0 - float((r21 - w0).norm() ** 2) / den,
                    "explained_kdepth": 1.0 - float((r21 - wfull).norm() ** 2) / den,
                    "regen_matches_window0_rel": float((s21.double() - w0).norm()
                                                       / max(float(w0.norm()), 1e-300)),
                })

        del mlp, unc, snap
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"    net {i + 1:>4}/{len(a.seeds)}  seed {seed}  "
              f"[{time.perf_counter() - t0:.1f}s]", flush=True)
        if prog:
            try:
                with open(prog, "w") as fh:
                    fh.write(str((i + 1) / len(a.seeds)))
            except OSError:
                pass

    # ---- the endpoint verdict, which is B1 -------------------------------
    ep_ok = True
    if endpoints:
        worst_k0 = max(abs(r["explained_k0"]) for r in endpoints)
        worst_kd = max(abs(1.0 - r["explained_kdepth"]) for r in endpoints)
        worst_id = max(r["regen_matches_window0_rel"] for r in endpoints)
        ep_ok = (worst_k0 < 1e-12) and (worst_kd < 1e-12) and (worst_id < 1e-12)
        print()
        print("  B1 endpoint check on the first net")
        print(f"    |explained(k=0)|        worst {worst_k0:.3e}   (must be 0)")
        print(f"    |1 - explained(k=depth)| worst {worst_kd:.3e}   (must be 0)")
        print(f"    regen state vs window-0  worst rel {worst_id:.3e}")
        print(f"    -> {'PASS' if ep_ok else 'FAIL -- the labeller is not computing E21'}")

    rel3 = float(np.mean(np.sqrt(E3_SQ / np.maximum(REF3_SQ, 1e-300))))
    rel21 = float(np.mean(np.sqrt(E21_SQ / np.maximum(REF21_SQ, 1e-300))))
    print()
    print(f"  labels: {len(a.seeds)} nets x {len(pre)} pre-layers x {a.width} neurons")
    print(f"    mean rel_E3  (||E3|| / ||uncapped(3,)||)   {rel3:.4f}")
    print(f"    mean rel_E21 (||E21|| / ||uncapped(2,1)||) {rel21:.4f}")
    print(f"    features: {n_feat} frozen columns, {', '.join(E3_FEATURES[:6])}, ...")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        np.savez_compressed(
            a.out,
            seeds=np.asarray(a.seeds, dtype=np.int64),
            pre_layers=np.asarray(pre, dtype=np.int64),
            feature_names=np.asarray(E3_FEATURES),
            E3=E3, FEAT=FEAT, E3_SQ=E3_SQ, E21_SQ=E21_SQ,
            REF3_SQ=REF3_SQ, REF21_SQ=REF21_SQ, E21_SV=SV,
            width=a.width, depth=a.depth, dtype=a.dtype,
            endpoint_pass=np.asarray([ep_ok]),
            endpoints=np.asarray([json.dumps(endpoints)]),
            contract=np.asarray(["experiments/LOG.md 2026-09-03 CONTRACT WP18 "
                                 "§ B, HEAD 49e2e42"]),
        )
        print(f"  wrote {os.path.relpath(a.out, _ROOT)} "
              f"({os.path.getsize(a.out) / 2**20:.1f} MiB)")
    return 0 if ep_ok else 1


# ==========================================================================
# --selftest
# ==========================================================================

def cmd_selftest(a) -> int:
    """Three checks, each of which fails loudly under a specific defect.

    S1  **The labeller's `E21` is `G-M-DEPTH`'s `E21`.** `regen_state`'s `(2,1)`
        slice must equal `slices_window(..., k=0)`'s bitwise. They restart from
        the same snapshot and take the same linear step, so anything but exact
        equality means one of them is running a different flow.
    S2  **The endpoints.** `explained(0) = 0` and `explained(depth) = 1`, which
        is `CONTRACT WP18`'s prediction B1 at small shape.
    S3  **Equivariance is real, not asserted.** Permute the network's neurons at
        one layer and the features and `E3` must permute with them. This is the
        property `CONTRACT WP15` called "not decoration", and it has never been
        tested in this repository -- it is tested here by permuting the WEIGHT
        MATRICES, which is a different code path from permuting the features.
    """
    import torch

    import p2_wp15_depth as wp15

    dtype = torch.float64 if a.dtype == "f64" else torch.float32
    p = min(a.depth - 1, max(2, a.depth // 2))
    seed = 90_000_000_301  # outside every registered block: a selftest, no truth
    mlp, K_in = wp15._build(seed, a.width, a.depth, a.device, dtype)
    unc = wp15.uncapped_slices(mlp, K_in, a.depth, [p])
    snap = wp15.regen_snapshots(mlp, K_in, a.depth)
    r3, r21 = unc[p][0].double(), unc[p][1].double()
    s3, s21, mu, Sig = regen_state(mlp, snap, p)

    w0_3, w0_21 = wp15.slices_window(mlp, snap, p, 0)
    s1_3 = float((s3.double() - w0_3.double()).abs().max())
    s1_21 = float((s21.double() - w0_21.double()).abs().max())
    s1_ok = (s1_3 == 0.0) and (s1_21 == 0.0)
    print(f"    S1  regen_state vs slices_window(k=0): max abs diff "
          f"(3,) {s1_3:.3e}  (2,1) {s1_21:.3e}   "
          f"{'PASS' if s1_ok else 'FAIL'}")

    den = max(float((r21 - w0_21.double()).norm() ** 2), 1e-300)
    wf = wp15.slices_window(mlp, snap, p, a.depth)[1].double()
    ex0 = 1.0 - float((r21 - w0_21.double()).norm() ** 2) / den
    exd = 1.0 - float((r21 - wf).norm() ** 2) / den
    s2_ok = abs(ex0) < 1e-12 and abs(1.0 - exd) < 1e-12
    print(f"    S2  explained(k=0) {ex0:.3e}   explained(k=depth) {exd:.12f}   "
          f"{'PASS' if s2_ok else 'FAIL'}")

    # ---- S3: permutation equivariance, through the weights ----------------
    from mint_provenance import canonical_mint

    bridge = _upstream()
    w, _e, _s = canonical_mint(a.width, a.depth, seed)
    g = torch.Generator().manual_seed(12345)
    perm = torch.randperm(a.width, generator=g).numpy()
    # Permuting layer `p`'s OUTPUT neurons means permuting W_p's rows and
    # W_{p+1}'s columns. Everything at pre-layer `p` must then permute.
    # `build_mlp` copies `W.T` into an `(out, in)` `nn.Linear`, so the MINT's
    # convention is `(in, out)`: output neuron `j` of layer `l` owns COLUMN `j`
    # of `w[l]`. Permuting layer `p`'s outputs is therefore a column permutation
    # of `w[p]` and a row permutation of `w[p + 1]`. Getting this backwards is
    # exactly what the first run of this selftest did, and it FAILED at rel
    # `0.97` -- which is the check doing its job.
    wp = [np.array(m, copy=True) for m in w]
    wp[p] = wp[p][:, perm]
    if p + 1 < len(wp):
        wp[p + 1] = wp[p + 1][perm, :]
    mlp2 = bridge.build_mlp(wp, device=a.device, dtype=dtype)
    K_in2 = {1: torch.zeros(a.width, device=a.device, dtype=dtype),
             2: torch.eye(a.width, device=a.device, dtype=dtype)}
    unc2 = wp15.uncapped_slices(mlp2, K_in2, a.depth, [p])
    snap2 = wp15.regen_snapshots(mlp2, K_in2, a.depth)
    s3b, s21b, mu2, Sig2 = regen_state(mlp2, snap2, p)

    e3a = (r3 - s3.double()).cpu().numpy()
    e3b = (unc2[p][0].double() - s3b.double()).cpu().numpy()
    fa = e3_features(mu, Sig, s3, s21).cpu().numpy()
    fb = e3_features(mu2, Sig2, s3b, s21b).cpu().numpy()
    sc3 = max(float(np.abs(e3a).max()), 1e-300)
    scf = np.maximum(np.abs(fa).max(axis=0), 1e-300)
    d3 = float(np.abs(e3a[perm] - e3b).max()) / sc3
    df = float(np.max(np.abs(fa[perm] - fb) / scf))
    tol = 1e-9 if a.dtype == "f64" else 2e-4
    s3_ok = (d3 < tol) and (df < tol)
    print(f"    S3  permutation equivariance: E3 rel {d3:.3e}  features rel "
          f"{df:.3e}   tol {tol:.1e}   {'PASS' if s3_ok else 'FAIL'}")

    ok = s1_ok and s2_ok and s3_ok
    print(f"\n  -> {'PASS' if ok else 'FAIL'}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"check": "WP18 Arm L labeller selftest",
                       "width": a.width, "depth": a.depth, "dtype": a.dtype,
                       "pre_layer": p,
                       "S1_regen_vs_window0": {"d3": s1_3, "d21": s1_21,
                                               "pass": s1_ok},
                       "S2_endpoints": {"explained_k0": ex0,
                                        "explained_kdepth": exd, "pass": s2_ok},
                       "S3_equivariance": {"E3_rel": d3, "feat_rel": df,
                                           "tol": tol, "pass": s3_ok},
                       "pass": ok}, fh, indent=1)
        print(f"  wrote {os.path.relpath(a.out, _ROOT)}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--dtype", choices=("f64", "f32"), default="f32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--nets", type=int, default=128)
    ap.add_argument("--block", default="P05")
    #: seed-ledger: block P05. Claimed on the HOST by this module and passed
    #: back with `--already-claimed` when the run is brokered, because a gpuq
    #: snapshot cannot write the ledger (the WP9 defect).
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--already-claimed", action="store_true")
    ap.add_argument("--endpoint-check", action="store_true", default=True)
    ap.add_argument("--no-endpoint-check", dest="endpoint_check",
                    action="store_false")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    print("=" * 78)
    print("WP18 / Arm L  --  step B1: the oracle labels (E3, E21)")
    print("=" * 78)
    _upstream()
    if a.selftest:
        print("\n-- the labeller must BE what it claims to be ----------------")
        return cmd_selftest(a)
    if a.labels:
        import seed_ledger as sl

        if a.already_claimed:
            if not a.seed:
                ap.error("--already-claimed needs --seed")
            a.seeds = [a.seed + i for i in range(a.nets)]
            print(f"  seeds {a.seeds[0]:,}..{a.seeds[-1]:,} (already claimed)")
        else:
            a.seeds = sl.claim(
                a.block, a.nets, "local_sim/p2_wp18_labels.py",
                ts="2026-09-03", artifact=a.out,
                note="WP18 Arm L step B1: oracle (E3, E21) labels and the "
                     "frozen E3 feature set. DESIGN tier; CONTRACT WP18 § B "
                     "permits tuning on P05 and nowhere else.")
            print(f"  claimed {a.seeds[0]:,}..{a.seeds[-1]:,} from {a.block}")
        print("\n-- uncapped minus regenerating, at every gate pre-layer -----")
        return cmd_labels(a)
    ap.error("one of --selftest / --labels")


if __name__ == "__main__":
    raise SystemExit(main())
