"""All-layer residual anatomy against the organizer's 16-layer labels.

    python tools/anatomy.py <candidate.py> --panel dev --n 100 \
        --out experiments/<dir>/anatomy.json [--layers 8,12,15,16] [--bins 10]

HANDOFF 2026-09-24 Step 1. The organizer split carries `all_layer_means`, a
(depth, width) float64 ground truth at N = 1e9 samples for EVERY layer, not just
the scored one. This tool reads all sixteen rows and answers four questions:

1. MSE by layer, and the growth ratio MSE_l / MSE_{l-1} -- where the error grows.
2. At the requested layers, the residual `r_i = mu_hat_i - mu_i` binned into
   quantile bins of four axes -- alpha = mu_pre/sigma_pre, D3/sigma^3, D4/sigma^4
   and the incoming weight-row norm ||W[:, i]|| -- with per-bin bias (mean) and
   RMS, and the share of total squared error in the top-2 bins of each axis.
3. The ONE-STEP residual: at layer l, one chain step taken from the *labelled*
   mu_{l-1} instead of the chain's own, everything else (covariance, kappa3
   tower, kappa4 core) left exactly as the chain has it. Reported beside the
   accumulated MSE, this separates local closure error from propagated drift.
4. Pearson correlation of the one-step residual with each binning feature, per
   layer.

WHAT THE ONE-STEP NUMBER DOES AND DOES NOT ISOLATE
--------------------------------------------------
It substitutes the mean channel only. The covariance, the carried third-cumulant
tower and the fourth-order core going into that step are still the chain's own,
and they have drifted for l-1 layers. So the one-step residual is *not* the pure
local closure error: it is "local closure error + the part of cumulant drift that
the mean update reads". It is therefore an UPPER bound on the local term, which
is the conservative direction for the question being asked (is the error local or
propagated?) -- a small one-step number is strong evidence for propagation.

HOW THE CHAIN IS INSTRUMENTED, AND WHY THAT IS SAFE
---------------------------------------------------
The features this needs (mu_pre, sigma, D3, D4, D21) are locals of `_layer` and
are not returned. Rather than recompute them outside -- which would duplicate the
dG transport formula and let the instrument drift from the chain -- this writes a
PATCHED COPY of the candidate next to the output with exactly two insertions,
both anchored on a string that occurs once in the file:

  * `_layer`: a call to `_ANATOMY_HOOK(...)` immediately after `mu_out` is
    formed -- the same point at which the v57 corrector is applied, so the
    features recorded here are exactly the values it will see at inference.
  * `_Tower`'s driving loop in `Estimator._tower`: when `_ANATOMY_ONESTEP` holds
    the labelled means, one extra `_layer` call per layer on the substituted
    state, whose output mu is recorded and then thrown away.

The patched copy is kept, so the diff is auditable. `_layer` never mutates its
input state -- `_advance` and `_append` both build a fresh `_Tower` and only the
fresh one is ever cached into (`_p2`, `_hub`, `_p3`, `_legs`, `_wts`) -- so the
extra call cannot perturb the chain that continues. That is asserted rather than
assumed: `--verify` runs the patched module with the hook OFF against the
unpatched module and requires the final-layer prediction to agree BIT FOR BIT,
and runs the one-step pass as well and requires the *accumulated* prediction to
be bit-identical to the pass without it. An instrument that changes the thing it
measures is not an instrument.

Accuracy backend only -- this reports no FLOP and no op count, so it runs on the
cupy shims (WHEST_BACKEND=cupy, ~2 s/net) like `tools/probe.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL_SIM = ROOT / "local_sim"
GPU_SHIM = ROOT / "tools" / "gpu_shim"

BACKEND = os.environ.get("WHEST_BACKEND", "cupy").lower()
# Same deliberate insert as tools/probe.py: this process reports accuracy only,
# never a FLOP, so putting the shims in front is the point. (AGENTS.md forbids
# the insert for anything that quotes a counted FLOP; nothing here does.)
sys.path.insert(0, str(LOCAL_SIM))
if BACKEND == "cupy":
    sys.path.insert(0, str(GPU_SHIM))
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "tools"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np  # noqa: E402

BUDGET = 2 ** 41

# ---------------------------------------------------------------- the patch

#: Anchor in `_layer`, immediately after `mu_out` is formed. This is the SAME
#: point at which the v57 corrector is applied, and that is deliberate: the
#: features the corrector is fitted on must be the exact values it will see at
#: inference. An earlier anchor (before `pK`) had every feature except `mu_out`
#: and `dmu`, and taking those two from the returned prediction array instead
#: made the layer-1 and layer-2 features inconsistent with the chain, because
#: `predict` replaces row 0 with its exact closed form. Asserted unique.
_ANCHOR_LAYER = """
    mu_out = K[(1,)]
    if last_layer:
        return (mu_out, None, None, 0.0, dG)
"""

_INJECT_LAYER = '''
    mu_out = K[(1,)]
    # --- tools/anatomy.py instrumentation (inert unless _ANATOMY_HOOK is set)
    if _ANATOMY_HOOK is not None:
        _ANATOMY_HOOK(layer_index, mu_pre, sigma, D3, D4, W,
                      None if last_layer else D21, mu, mu_out)
    if last_layer:
        return (mu_out, None, None, 0.0, dG)
'''

#: Anchor in `Estimator._tower`: the one loop that drives the chain.
_ANCHOR_TOWER = """        mus = []
        for l in range(depth):
            state = _layer(w[l], state, table, dtype, const, 2.0,
                           last_layer=(l == depth - 1), gens=gens,
                           layer_index=l, depth=depth)
"""

_INJECT_TOWER = """        mus = []
        for l in range(depth):
            # --- tools/anatomy.py: ONE step from the LABELLED mu_{l-1}, with
            # this layer's covariance / kappa3 / kappa4 left exactly as the
            # chain has them. Runs BEFORE the real step and is discarded. The
            # hook is muted so the recorded features stay those of the real
            # chain. `_layer` does not mutate its input state, so the chain
            # that continues below is untouched -- see the module docstring.
            if _ANATOMY_ONESTEP is not None and l > 0:
                global _ANATOMY_HOOK
                _saved_hook = _ANATOMY_HOOK
                _ANATOMY_HOOK = None
                try:
                    _alt = _layer(w[l],
                                  (_ANATOMY_ONESTEP[l - 1], state[1], state[2],
                                   state[3], state[4]),
                                  table, dtype, const, 2.0,
                                  last_layer=(l == depth - 1), gens=gens,
                                  layer_index=l, depth=depth)
                finally:
                    _ANATOMY_HOOK = _saved_hook
                _ANATOMY_ONESTEP_OUT.append(_alt[0])
                _alt = None
            state = _layer(w[l], state, table, dtype, const, 2.0,
                           last_layer=(l == depth - 1), gens=gens,
                           layer_index=l, depth=depth)
"""

_INJECT_GLOBALS = '''

# ---------------------------------------------------------------------------
# tools/anatomy.py instrumentation. Both are None in the shipped chain and this
# copy is NEVER submitted; it exists so the anatomy reads the chain's own
# internal features rather than a reimplementation of them.
_ANATOMY_HOOK = None
_ANATOMY_ONESTEP = None
_ANATOMY_ONESTEP_OUT = []
'''


def patch_source(src: str) -> str:
    """Insert the two hooks. Raises unless each anchor occurs exactly once."""
    for name, anchor in (("_layer", _ANCHOR_LAYER), ("_tower", _ANCHOR_TOWER)):
        if src.count(anchor) != 1:
            raise SystemExit(
                f"anatomy: the {name} anchor occurs {src.count(anchor)} times, "
                "expected exactly 1 -- the candidate's spelling has changed and "
                "the patch must be re-derived rather than applied blind")
    src = src.replace(_ANCHOR_LAYER, _INJECT_LAYER, 1)
    src = src.replace(_ANCHOR_TOWER, _INJECT_TOWER, 1)
    # The globals go after the imports but before first use; module top is safe.
    marker = "\nBUDGET_UNITS" if "\nBUDGET_UNITS" in src else None
    if marker:
        src = src.replace(marker, _INJECT_GLOBALS + marker, 1)
    else:
        # After the last top-level import line.
        lines = src.splitlines(keepends=True)
        last = max(i for i, l in enumerate(lines)
                   if l.startswith("import ") or l.startswith("from "))
        lines.insert(last + 1, _INJECT_GLOBALS)
        src = "".join(lines)
    return src


# ------------------------------------------------------------------- harness


class _MLP:
    __slots__ = ("weights", "width", "depth", "seed")

    def __init__(self, weights, width, depth, seed):
        self.weights, self.width, self.depth, self.seed = weights, width, depth, seed


class _Ctx:
    def __init__(self, width, depth, budget, seed):
        self.width, self.depth = width, depth
        self.flop_budget, self.seed = budget, seed
        self.api_version, self.scratch_dir, self.submission_dir = "1.0", None, None


def load_module(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _host(a, xp):
    """Bring one array back to the host as float64 numpy."""
    if a is None:
        return None
    if xp is not None:
        a = xp.asnumpy(a)
    return np.asarray(a, dtype=np.float64)


#: The feature names the hook records, IN ORDER. The fitted corrector weights
#: depend on this order, so it is part of the contract between this file,
#: `tools/corr_ceiling.py` and the shipped estimator -- do not reorder.
#: `alpha` is the chain's own clipped mu_pre/sigma only inside the closure; here
#: it is the raw ratio, which is what a binning axis and a feature want.
FEATURES = ("mu_pre", "sigma", "alpha", "d3n", "d4n", "wnorm", "d21e",
            "mu_out", "dmu")


def make_hook(store: list, xp):
    """Record one row of per-neuron features per layer, on the host."""
    def hook(layer_index, mu_pre, sigma, D3, D4, W, D21, mu_prev, mu_out):
        mp = _host(mu_pre, xp)
        sg = _host(sigma, xp)
        d3 = _host(D3, xp)
        d4 = _host(D4, xp)
        # Neuron i of this layer reads the COLUMN W[:, i] (`mu_pre = W.T @ mu`),
        # so its incoming weight-row norm is ||W[:, i]||.
        wn = np.sqrt((_host(W, xp) ** 2).sum(axis=0))
        if D21 is None:
            d21e = np.full_like(sg, np.nan)
        else:
            # Row energy sum_k D21[i, k]^2, reduced on the device before the
            # copy -- D21 is (n, n) and copying 100 x 16 of them is pointless.
            d21 = D21 if xp is None else D21
            d21e = _host((d21 * d21).sum(axis=1), xp)
        mo = _host(mu_out, xp)
        with np.errstate(divide="ignore", invalid="ignore"):
            store.append({
                "layer": int(layer_index),
                "mu_pre": mp, "sigma": sg,
                "alpha": mp / sg,
                "d3n": d3 / sg ** 3,
                "d4n": d4 / sg ** 4,
                "wnorm": wn,
                "d21e": d21e / sg ** 4,
                "mu_out": mo,
                # The chain's own previous mean, NOT the prediction array's row
                # (which `predict` overwrites at layer 1 with a closed form).
                "dmu": mo - _host(mu_prev, xp).mean(),
            })
    return hook


def run_net(mod, est, wf, width, depth, seed, xp, truth, onestep: bool):
    """One predict call. Returns (pred (depth,width), features, onestep mus)."""
    store: list = []
    mod._ANATOMY_HOOK = make_hook(store, xp)
    mod._ANATOMY_ONESTEP_OUT = []
    if onestep:
        # The labelled means, in the chain's own dtype and on its own device.
        # truth[l] is layer l+1 (1-indexed); the step at layer index l consumes
        # the mean of layer l, i.e. truth[l - 1].
        mod._ANATOMY_ONESTEP = [
            (xp.asarray(truth[i].astype(np.float32)) if xp is not None
             else truth[i].astype(np.float32)) for i in range(depth)]
    else:
        mod._ANATOMY_ONESTEP = None
    try:
        raw = est.predict(_MLP(wf, width, depth, int(seed)), BUDGET)
        pred = _host(raw, xp)
        alt = [_host(a, xp) for a in mod._ANATOMY_ONESTEP_OUT]
    finally:
        mod._ANATOMY_HOOK = None
        mod._ANATOMY_ONESTEP = None
        mod._ANATOMY_ONESTEP_OUT = []
    return pred, store, alt


# ------------------------------------------------------------- aggregation


def bin_stats(axis: np.ndarray, resid: np.ndarray, nbins: int) -> dict:
    """Quantile-bin `resid` by `axis`; per-bin bias, RMS and squared-error share."""
    ok = np.isfinite(axis) & np.isfinite(resid)
    if ok.sum() < nbins * 4:
        return {"usable": int(ok.sum())}
    a, r = axis[ok], resid[ok]
    edges = np.quantile(a, np.linspace(0, 1, nbins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, nbins - 1)
    tot = float((r ** 2).sum())
    bins = []
    for b in range(nbins):
        m = idx == b
        rb = r[m]
        if rb.size == 0:
            bins.append(None)
            continue
        se = float((rb ** 2).sum())
        bins.append({
            "n": int(rb.size),
            "axis_lo": float(a[m].min()), "axis_hi": float(a[m].max()),
            "bias": float(rb.mean()),
            "rms": float(np.sqrt((rb ** 2).mean())),
            "se_share": se / tot if tot > 0 else float("nan"),
        })
    shares = sorted((b["se_share"] for b in bins if b), reverse=True)
    return {"usable": int(ok.sum()), "total_se": tot, "bins": bins,
            "top2_se_share": float(sum(shares[:2]))}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return float("nan")
    a, b = x[ok], y[ok]
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("candidate")
    ap.add_argument("--panel", default="dev")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="8,12,15,16",
                    help="1-indexed layers to bin the residual at")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--onestep", action="store_true", default=True)
    ap.add_argument("--no-onestep", dest="onestep", action="store_false")
    ap.add_argument("--verify", action="store_true",
                    help="prove the patch is inert, then exit")
    ap.add_argument("--dump", default="",
                    help="also write per-neuron features and residuals to this "
                         ".npz, for fitting the corrector's ceiling offline")
    a = ap.parse_args(argv)

    import panels
    import real_data as rd

    xp = None
    if BACKEND == "cupy":
        import cupy as xp
        xp.get_default_memory_pool().set_limit(
            size=int(float(os.environ.get("PROBE_GPU_LIMIT_GIB", "20")) * 2 ** 30))

    cand = Path(a.candidate)
    if not cand.is_absolute():
        cand = ROOT / cand
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    # The patched copy lives beside the output and is kept for audit.
    patched_path = out.parent / f"_patched_{cand.stem}.py"
    patched_path.write_text(patch_source(cand.read_text(encoding="utf-8")),
                            encoding="utf-8", newline="\n")
    print(f"[anatomy] patched copy -> {patched_path.relative_to(ROOT)}", flush=True)

    mod = load_module(patched_path, "_anatomy_candidate")
    est = mod.Estimator()

    rows = panels.nets(a.panel)[a.lo:a.lo + a.n]

    def load(i_split_idx):
        split, idx, seed, _nm = i_split_idx
        w, tr, _ = rd.get(idx, split, dtype=np.float32)
        depth, width = w.shape[0], w.shape[1]
        wf = []
        for x in w:
            arr = np.asarray(x, dtype=np.float64)
            arr.setflags(write=False)
            wf.append(xp.asarray(arr) if xp is not None else arr)
        return wf, width, depth, int(seed), np.asarray(tr, dtype=np.float64)

    # ------------------------------------------------ the inertness proof
    if a.verify:
        base = load_module(cand, "_anatomy_base")
        best = base.Estimator()
        bad = 0
        for k in range(2):
            wf, width, depth, seed, truth = load(rows[k])
            best.setup(_Ctx(width, depth, BUDGET, 0))
            est.setup(_Ctx(width, depth, BUDGET, 0))
            ref = _host(best.predict(_MLP(wf, width, depth, seed), BUDGET), xp)
            # (a) hook off, one-step off -> must be bit-identical
            mod._ANATOMY_HOOK = None
            mod._ANATOMY_ONESTEP = None
            p_off = _host(est.predict(_MLP(wf, width, depth, seed), BUDGET), xp)
            # (b) hook ON and one-step ON -> the ACCUMULATED chain must still be
            #     bit-identical, i.e. the extra _layer call perturbs nothing.
            p_on, store, alt = run_net(mod, est, wf, width, depth, seed, xp,
                                       truth, onestep=True)
            for tag, p in (("hook off", p_off), ("hook+onestep on", p_on)):
                same = np.array_equal(ref, p)
                mx = float(np.abs(ref - p).max())
                print(f"[verify] net {k} {tag:18s} bit-identical={same} "
                      f"max|diff|={mx:.3e}", flush=True)
                if not same:
                    bad += 1
            print(f"[verify] net {k} features recorded for {len(store)} layers, "
                  f"{len(alt)} one-step means", flush=True)
            if len(store) != depth or len(alt) != depth - 1:
                print(f"[verify] net {k} WRONG COUNTS: expected {depth} / "
                      f"{depth - 1}", flush=True)
                bad += 1
        print("VERIFY PASS" if not bad else f"VERIFY FAIL ({bad})")
        return 1 if bad else 0

    # ------------------------------------------------------------ the run
    res = {"candidate": cand.name, "patched": patched_path.name,
           "panel": a.panel, "n": len(rows), "backend": BACKEND,
           "bins": a.bins, "onestep": bool(a.onestep),
           "layers_binned": [int(x) for x in a.layers.split(",")]}

    depth = width = None
    acc_se = None       # accumulated squared error, summed over nets, per layer
    one_se = None       # one-step squared error, same
    lbl_sq = None       # label energy per layer (for a relative scale)
    # Per-layer pooled feature/residual columns, for bins and correlations.
    pool_feat: dict = {}
    pool_acc: dict = {}
    pool_one: dict = {}
    pool_tgt: dict = {}
    setup_done = False
    t0 = time.time()

    for i, row in enumerate(rows):
        wf, width, depth, seed, truth = load(row)
        if not setup_done:
            est.setup(_Ctx(width, depth, BUDGET, 0))
            setup_done = True
            acc_se = np.zeros(depth)
            one_se = np.zeros(depth)
            lbl_sq = np.zeros(depth)
            dump_feats = FEATURES
            for l in range(depth):
                pool_feat[l] = {f: [] for f in dump_feats}
                pool_acc[l] = []
                pool_one[l] = []
                pool_tgt[l] = []

        pred, store, alt = run_net(mod, est, wf, width, depth, seed, xp,
                                   truth, onestep=a.onestep)
        if pred.shape != truth.shape:
            raise SystemExit(f"net {i}: pred {pred.shape} vs truth {truth.shape}")
        if len(store) != depth:
            raise SystemExit(f"net {i}: hook fired {len(store)} times, want {depth}")

        r_acc = pred - truth                       # (depth, width)
        acc_se += (r_acc ** 2).sum(axis=1)
        lbl_sq += (truth ** 2).sum(axis=1)
        if a.onestep:
            for l in range(1, depth):
                one_se[l] += float(((alt[l - 1] - truth[l]) ** 2).sum())

        for rec in store:
            l = rec["layer"]
            for f in FEATURES:
                pool_feat[l][f].append(rec[f].astype(np.float32))
            pool_acc[l].append(r_acc[l].astype(np.float32))
            # label - the CHAIN's own mu_out (rec["mu_out"]), which is what the
            # corrector adds to. Not label - pred[l]: see the dump comment.
            pool_tgt[l].append((truth[l] - rec["mu_out"]).astype(np.float32))
            if a.onestep and l > 0:
                pool_one[l].append((alt[l - 1] - truth[l]).astype(np.float32))

        if xp is not None:
            xp.get_default_memory_pool().free_all_blocks()
        if (i + 1) % 5 == 0 or i == 0:
            el = time.time() - t0
            print(f"[anatomy] net {i + 1}/{len(rows)} ({el:.0f}s) "
                  f"final-layer MSE {float((r_acc[-1] ** 2).mean()):.4e}", flush=True)

    nnet, n = len(rows), width
    tot = nnet * n

    if a.dump:
        # (layer, net*neuron, feature) plus both residuals, so the corrector's
        # ceiling can be fitted offline without re-running the chain. A net id
        # column goes with it: a fit that splits by NET rather than by neuron is
        # the only honest held-out split -- neurons of one net share its weights.
        dpath = Path(a.dump)
        if not dpath.is_absolute():
            dpath = ROOT / dpath
        names = list(dump_feats)
        X = np.stack([np.stack([np.concatenate(pool_feat[l][f]) for f in names],
                               axis=1) for l in range(depth)])
        netid = np.repeat(np.arange(nnet, dtype=np.int32), n)
        one = np.stack([(np.concatenate(pool_one[l]) if (a.onestep and l)
                         else np.zeros(nnet * n, dtype=np.float32))
                        for l in range(depth)])
        acc = np.stack([np.concatenate(pool_acc[l]) for l in range(depth)])
        # THE CORRECTOR'S TRAINING TARGET, recorded explicitly rather than
        # derived from `accumulated`. They differ in two places and both would
        # be silent: `accumulated` is chain-MINUS-label (opposite sign), and it
        # is built from the PREDICTION array, whose row 0 `predict` overwrites
        # with an exact closed form and whose every row is clamped at zero. The
        # corrector corrects the chain's own `mu_out`, so that is what this
        # subtracts from the label.
        tgt = np.stack([np.concatenate(pool_tgt[l]) for l in range(depth)])
        np.savez_compressed(dpath, X=X.astype(np.float32), features=np.array(names),
                            onestep=one.astype(np.float32),
                            accumulated=acc.astype(np.float32),
                            target=tgt.astype(np.float32), netid=netid,
                            panel=a.panel, candidate=cand.name)
        print(f"[anatomy] dump -> {dpath.relative_to(ROOT)}  X{X.shape}", flush=True)

    # ---- 1. MSE by layer and the growth ratio
    mse = acc_se / tot
    mse_one = one_se / tot
    by_layer = []
    for l in range(depth):
        by_layer.append({
            "layer": l + 1,
            "mse": float(mse[l]),
            "ratio_prev": float(mse[l] / mse[l - 1]) if l else None,
            "label_ms": float(lbl_sq[l] / tot),
            "nmse": float(mse[l] / (lbl_sq[l] / tot)),
            "onestep_mse": float(mse_one[l]) if (a.onestep and l) else None,
            "onestep_over_acc": float(mse_one[l] / mse[l]) if (a.onestep and l) else None,
        })
    res["by_layer"] = by_layer

    # ---- 2. binned residual at the requested layers, and 4. correlations
    axes = ("alpha", "d3n", "d4n", "wnorm")
    binned: dict = {}
    corr: dict = {}
    for l in range(depth):
        feats = {f: np.concatenate(pool_feat[l][f]).astype(np.float64)
                 for f in FEATURES}
        r_a = np.concatenate(pool_acc[l]).astype(np.float64)
        r_o = (np.concatenate(pool_one[l]).astype(np.float64)
               if (a.onestep and l) else None)
        corr[l + 1] = {
            "accumulated": {f: pearson(feats[f], r_a) for f in FEATURES},
            "onestep": ({f: pearson(feats[f], r_o) for f in FEATURES}
                        if r_o is not None else None),
        }
        if (l + 1) in res["layers_binned"]:
            binned[l + 1] = {
                "accumulated": {ax: bin_stats(feats[ax], r_a, a.bins) for ax in axes},
                "onestep": ({ax: bin_stats(feats[ax], r_o, a.bins) for ax in axes}
                            if r_o is not None else None),
            }
        del feats, r_a, r_o
    res["binned"] = binned
    res["corr"] = corr

    out.write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"\n[anatomy] -> {out.relative_to(ROOT)}")

    print(f"\n  layer      MSE     ratio    one-step   one/acc")
    for r in by_layer:
        rp = f"{r['ratio_prev']:7.3f}" if r["ratio_prev"] else "      -"
        om = f"{r['onestep_mse']:.4e}" if r["onestep_mse"] else "         -"
        oo = f"{r['onestep_over_acc']:7.3f}" if r["onestep_over_acc"] else "      -"
        print(f"  {r['layer']:5d}  {r['mse']:.4e} {rp}  {om}  {oo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
