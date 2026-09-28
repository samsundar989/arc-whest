"""Gaussian triangle alignment screen; research only, never a score meter.

The connected quadratic-Hermite term is
    T_ijk = f2_i f2_j f2_k C_ij C_jk C_ki = tr(H_i H_j H_k),
where C ~= U U.T is a PSD approximation to the FULL source pre-covariance,
f2 = phi(mu / sigma) / sigma, and H_i = f2_i u_i u_i.T. Its coefficient is +1:
eight connected Wick pairings cancel the three Hermite factors of 1/2.
This is one omitted Gaussian diagram, not the whole third cumulant.

Source repeated-index terms are subtracted before comparison with next-layer
all-distinct label error. Every reported scalar fit is diagnostic, not a trained
submission. FIT networks were used in base fitting; no unseen-score claim follows.

    python tools/k111_triangle_probe.py check                 # tiny CPU only
    python tools/k111_triangle_probe.py run --nets 0-1         # workerq required
    python tools/k111_triangle_probe.py report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def triangle_slices(U, f2, W, xp=np):
    """Compensated next (3,) and off-diagonal (2,1), numpy or torch arrays.

    No cubic tensor: only n*q*q factors and O(n*n) slice matrices. W[a,j]
    maps source neuron a to next neuron j. D21[i,j] repeats output i.
    """
    from k111_diag import rep3, rep21

    n, q = U.shape
    m = W.shape[1]
    H = f2[:, None, None] * U[:, :, None] * U[:, None, :]
    H2 = H @ H
    t = (H2 * H).sum(-1).sum(-1)  # H is symmetric, so this is tr(H^3).
    B = H2.reshape(n, q * q) @ H.reshape(n, q * q).T
    Boff = B - xp.diag(t)
    Hn = (W.T @ H.reshape(n, q * q)).reshape(m, q, q)
    Hn2 = Hn @ Hn
    D3 = (Hn2 * Hn).sum(-1).sum(-1)
    D21 = Hn2.reshape(m, q * q) @ Hn.reshape(m, q * q).T
    A3 = D3 - rep3(t, Boff, W)
    A21 = D21 - rep21(t, Boff, W)
    # Its diagonal equals A3; the normal D21 consumer has zero diagonal.
    diag_error = A21.diagonal() - A3
    A21 = A21 - xp.diag(A21.diagonal())
    return A3, A21, diag_error


def alignment(target, feature):
    """Uncentered inner-product projection, with positive coefficient = add."""
    ee = float((target * target).sum())
    xx = float((feature * feature).sum())
    xy = float((feature * target).sum())
    if not all(math.isfinite(v) for v in (ee, xx, xy)):
        raise FloatingPointError("nonfinite alignment Gram")
    return {
        "target_energy": ee, "feature_energy": xx, "dot": xy,
        "feature_over_target_norm": math.sqrt(xx / ee) if ee else None,
        "cosine": xy / math.sqrt(ee * xx) if ee and xx else None,
        "theory_coefficient": 1.0,
        "theory_energy_removed": (2 * xy - xx) / ee if ee else None,
        "oracle_scalar": xy / xx if xx else None,
        "oracle_energy_removed": xy * xy / (xx * ee) if xx and ee else None,
    }


def metrics(target, feature, model):
    # Orthogonal diagnostics remove model-parallel components from BOTH vectors.
    # Coefficient one there is not the raw additive triangle: it also adjusts
    # a uniform model component. Only the raw column tests the unchanged term.
    mm = float((model * model).sum())
    if not math.isfinite(mm):
        raise FloatingPointError("nonfinite model Gram")
    target_orth = target - (float((target * model).sum()) / mm) * model if mm else target
    feature_orth = feature - (float((feature * model).sum()) / mm) * model if mm else feature
    return {"raw": alignment(target, feature),
            "orthogonal_to_model": alignment(target_orth, feature_orth),
            "orthogonal_target_energy_fraction": float((target_orth ** 2).sum()) /
            max(float((target ** 2).sum()), 1e-300)}


def check():
    """Tiny float64 dense tensor truth, plus wrong-compensation negative control."""
    from k111_diag import rep3, rep21

    rng = np.random.default_rng(20260928)
    rows = []
    for n, m, q in ((6, 7, 3), (7, 5, 4), (5, 6, 5)):
        U = rng.normal(size=(n, q)) / math.sqrt(q)
        f2 = np.exp(rng.normal(size=n))
        W = rng.normal(size=(n, m)) / math.sqrt(n)
        C = U @ U.T
        H = f2[:, None, None] * U[:, :, None] * U[:, None, :]
        T = np.einsum("iab,jbc,kca->ijk", H, H, H)
        law = (f2[:, None, None] * f2[None, :, None] * f2[None, None, :] *
               C[:, :, None] * C[None, :, :] * C.T[:, None, :])
        idx = np.arange(n)
        a, b, c = np.meshgrid(idx, idx, idx, indexing="ij")
        Tad = T * ((a != b) & (a != c) & (b != c))
        D3 = np.einsum("abc,aj,bj,cj->j", Tad, W, W, W)
        D21 = np.einsum("abc,ai,bi,cj->ij", Tad, W, W, W)
        np.fill_diagonal(D21, 0.0)
        got3, got21, diag = triangle_slices(U, f2, W)
        t = T[idx, idx, idx]
        S = T[idx[:, None], idx[:, None], idx[None, :]].copy()
        np.fill_diagonal(S, 0.0)
        full3 = np.einsum("abc,aj,bj,cj->j", T, W, W, W)
        full21 = np.einsum("abc,ai,bi,cj->ij", T, W, W, W)
        wrong21 = full21 - rep21(t, S.T, W)
        np.fill_diagonal(wrong21, 0.0)
        row = {
            "n": n, "m": m, "q": q,
            "trace_matches_coefficient_one_cycle": float(np.max(abs(T - law))),
            "symmetry_max_error": float(np.max(abs(T - T.transpose(2, 1, 0)))),
            "D3_relative_error": float(np.linalg.norm(got3 - D3) / np.linalg.norm(D3)),
            "D21_relative_error": float(np.linalg.norm(got21 - D21) / np.linalg.norm(D21)),
            "D21_diagonal_matches_D3": float(np.linalg.norm(diag) / np.linalg.norm(got3)),
            "uncompensated_D3_relative_error": float(np.linalg.norm(full3 - D3) / np.linalg.norm(D3)),
            "wrong_repeated_orientation_D21_error": float(np.linalg.norm(wrong21 - D21) / np.linalg.norm(D21)),
        }
        assert row["trace_matches_coefficient_one_cycle"] < 1e-10
        assert row["symmetry_max_error"] < 1e-10
        assert row["D3_relative_error"] < 1e-11
        assert row["D21_relative_error"] < 1e-11
        assert row["D21_diagonal_matches_D3"] < 1e-11
        assert row["uncompensated_D3_relative_error"] > 1e-3
        assert row["wrong_repeated_orientation_D21_error"] > 1e-3
        rows.append(row)
    # Orthogonal projection must remove a feature parallel to the old model.
    m = np.array([1., 2., 3.])
    assert metrics(np.array([2., 3., 4.]), m, m)["orthogonal_to_model"]["feature_energy"] == 0
    return {"pass": True, "coefficient": 1.0, "dense_cases": rows}


def analyse_layer(d, rec, state, labels, layer, W, ranks):
    from k111_diag import rep3, rep21

    torch = d.torch
    off = d.off64
    Wd = W.double()
    (t, S), (D3, D21) = d.state_slices(state[2], W.T.contiguous(), off.float())
    model3 = D3 - rep3(t, S * off, Wd)
    model21 = (D21 - rep21(t, S * off, Wd)) * off
    lp = d.label_parts(labels, layer, Wd)
    error3, error21 = lp["AD3"] - model3, lp["AD21"] - model21
    var = rec["var"].double()
    if not bool((var > 0).all()):
        raise FloatingPointError(f"nonpositive pre-variance at source layer {layer}")
    C = rec["C11"].double() + torch.diag(var)
    symmetry = float((C - C.T).norm() / C.norm())
    C = (C + C.T) * .5
    eig, V = torch.linalg.eigh(C)
    sig = var.sqrt()
    alpha = rec["mu"].double() / sig
    f2 = torch.exp(-.5 * alpha * alpha) / (math.sqrt(2 * math.pi) * sig)
    result = {
        "source_layer": layer, "consumer_layer": layer + 1,
        "covariance_symmetry_relative_error": symmetry,
        "covariance_min_eigenvalue": float(eig[0]),
        "covariance_negative_energy_fraction": float((eig.clamp_max(0) ** 2).sum() /
                                                        (eig ** 2).sum()),
        "D3_AD_relative_error": float(error3.norm() / lp["AD3"].norm()),
        "D21_AD_relative_error": float(error21.norm() / lp["AD21"].norm()),
        "ranks": {},
    }
    for q in ranks:
        if q > d.n:
            raise ValueError("requested covariance rank exceeds width")
        top = eig[-q:].clamp_min(0)
        U = V[:, -q:] * top.sqrt()[None, :]
        X3, X21, diag = triangle_slices(U, f2, Wd, xp=torch)
        if not bool(torch.isfinite(X3).all() and torch.isfinite(X21).all()):
            raise FloatingPointError("nonfinite triangle projection")
        row = {
            "positive_modes": int((top > 0).sum()),
            "covariance_energy_fraction": float((top ** 2).sum() / (eig ** 2).sum()),
            "D21_diagonal_identity_relative_error": float(diag.norm() / X3.norm().clamp_min(1e-30)),
            "D3": metrics(error3, X3, model3), "D21": metrics(error21, X21, model21),
        }
        # Independent K21 re-bake changes the target, not the proposed feature.
        if "AD3_rep1" in lp:
            row["D3_rep1"] = metrics(lp["AD3_rep1"] - model3, X3, model3)
            row["D21_rep1"] = metrics(lp["AD21_rep1"] - model21, X21, model21)
        result["ranks"][str(q)] = row
    return result


def run(a):
    import torch
    from k111_diag import Diag

    torch.set_num_threads(2)
    start, end = map(int, a.nets.split("-"))
    if start < 0 or end < start or end > 99:
        raise ValueError("this diagnostic is restricted to FIT full rows 0-99")
    layers = sorted({int(x) for x in a.layers.split(",")})
    ranks = sorted({int(x) for x in a.ranks.split(",")})
    if not ranks or min(ranks) < 1:
        raise ValueError("ranks must be positive")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cand, params = (ROOT / a.cand).resolve(), (ROOT / a.params).resolve()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    metadata = {
        # Stable input names plus hashes allow recovery from a new queue snapshot
        # whose absolute checkout path changed but whose source bytes did not.
        "candidate": a.cand, "candidate_sha256": sha(cand),
        "params": a.params, "params_sha256": sha(params),
        "probe_sha256": sha(Path(__file__)),
        "diag_sha256": sha(ROOT / "tools/k111_diag.py"),
        "twin_sha256": sha(ROOT / "tools/twin/twin.py"),
        "layers_source_zero_based": layers, "ranks": ranks,
        "coefficient": 1.0, "meaning": "Gaussian quadratic-Hermite triangle only",
        "panel": "FIT/TRAIN; not independent validation", "device": str(torch.cuda.get_device_name()),
    }
    meta_path = out / "metadata.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise SystemExit("existing metadata differs; use a new output directory")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    args = SimpleNamespace(cand=str(cand), params=str(params), h=.05, max_layer=99, gain_file="")
    d = Diag(args)
    if not layers or min(layers) < 1 or max(layers) >= d.depth - 1:
        raise ValueError("source layers must have pre-covariance hooks and a following layer")
    d.capture = False  # No saved old towers or all-layer copies.
    for g in range(start, end + 1):
        dest = out / f"net_{g:03d}.json"
        if dest.exists():
            print(f"REUSE net={g}", flush=True)
            continue
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        w, truth = d.data.net("full", g)
        labels = d.labels(g)
        rec = {}
        repairs = {"checked_hook_values": 0}

        def hook(layer, name, value, *aux):
            if name == "K":
                arrays = value.values()
            elif name in ("mu", "var", "C11", "D3", "D21"):
                arrays = (value,)
            else:
                arrays = ()
            for array in arrays:
                if not bool(torch.isfinite(array).all()):
                    raise FloatingPointError(f"nonfinite {name} before repair, net={g}, layer={layer}")
                repairs["checked_hook_values"] += 1
            if layer in layers and name in ("mu", "var", "C11"):
                rec[name] = value.detach()
            return value

        d.mod._A2 = hook
        result = {"net": g, "layers": {}}
        state = (d.T.fnp.zeros(d.n), d.T.fnp.eye(d.n), None, 0.0, d.T.fnp.zeros(d.n))
        try:
            with torch.no_grad():
                for layer in range(d.depth):
                    rec.clear()
                    state = d.tw._step(layer, w[layer], state)
                    if layer in layers:
                        if set(rec) != {"mu", "var", "C11"}:
                            raise RuntimeError(f"missing pre-birth source hooks at layer {layer}: {set(rec)}")
                        result["layers"][str(layer)] = analyse_layer(d, rec, state, labels, layer,
                                                                    w[layer + 1], ranks)
                        print(f"net={g} source={layer} elapsed={time.time()-t0:.2f}s", flush=True)
                pred = state[0].clamp_min(0).double()
                gt = torch.from_numpy(np.asarray(truth[-1], np.float64)).to(d.dev)
                result["baseline_final_mse"] = float(((pred - gt) ** 2).mean())
        finally:
            d.mod._A2 = None
        result.update(seconds=time.time() - t0, finite_hooks=repairs,
                      peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
                      final_layer_label_noise_floor=7.7e-11)
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        tmp.replace(dest)
        progress = os.environ.get("WORKERQ_PROGRESS")
        if progress:
            Path(progress).write_text(str((g - start + 1) / (end - start + 1)))
        print(f"COMPLETE net={g} seconds={result['seconds']:.2f} peak={result['peak_vram_gib']:.2f}GiB", flush=True)
        del labels, w, state, rec, pred, gt
    report(out)


def report(out):
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("net_*.json"))]
    if not rows:
        raise SystemExit("no completed triangle networks")
    lines = ["# Gaussian triangle alignment screen", "",
             "Coefficient +1 is the theoretical connected quadratic-Hermite term. Negative energy "
             "removed means the correction worsens the projected moment error. Oracle scalar fitting "
             "uses the target and is only an optimistic diagnostic. Cross-network fitting below uses "
             "the other completed FIT networks; the base estimator was trained on FIT, so this is "
             "not independent validation or evidence of final score gain. The orthogonal columns "
             "project BOTH residual and feature off the model AD; their coefficient-one result "
             "therefore includes a compensating model-parallel adjustment, unlike the raw columns.", "",
             "| source | q | channel | target | median theory energy removed | median oracle energy removed | "
             "median oracle scalar | median cross-network energy removed |",
             "|---:|---:|---|---|---:|---:|---:|---:|"]
    for layer in rows[0]["layers"]:
        for q in rows[0]["layers"][layer]["ranks"]:
            for channel in ("D3", "D21"):
                for target in ("raw", "orthogonal_to_model"):
                    grams = [r["layers"][layer]["ranks"][q][channel][target] for r in rows]
                    cross = []
                    for i, g in enumerate(grams):
                        other = [v for j, v in enumerate(grams) if j != i]
                        xx = sum(v["feature_energy"] for v in other)
                        if xx and g["target_energy"]:
                            coef = sum(v["dot"] for v in other) / xx
                            cross.append((2 * coef * g["dot"] - coef**2 * g["feature_energy"]) /
                                         g["target_energy"])
                    def med(key):
                        values = [v[key] for v in grams if v[key] is not None]
                        return f"{np.median(values):.5f}" if values else "n/a"
                    cross_text = f"{np.median(cross):.5f}" if cross else "n/a"
                    lines.append(f"| {layer} | {q} | {channel} | {target} | {med('theory_energy_removed')} | "
                                 f"{med('oracle_energy_removed')} | {med('oracle_scalar')} | {cross_text} |")
    lines += ["", "Final-layer reference MSE noise floor is approximately 7.7e-11. The alignment "
              "targets are third-cumulant projections, whose noise is different; per-net JSON includes "
              "the available independent K21-label replicate check.", "",
              "Baseline final MSEs: " + str({r["net"]: r["baseline_final_mse"] for r in rows})]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("check", "run", "report"))
    ap.add_argument("--nets", default="0-1")
    ap.add_argument("--layers", default="2,5,8,11,14", help="zero-based source layers")
    ap.add_argument("--ranks", default="4,8")
    ap.add_argument("--cand", default="estimator_k3_v62_s1.py")
    ap.add_argument("--params", default="experiments/2026-09-27_S2t3/params_step736.npz")
    ap.add_argument("--out", default=str(ROOT / "experiments/2026-09-28_triangle"))
    args = ap.parse_args()
    if args.command == "check":
        result = check()
        print(json.dumps(result, indent=2, allow_nan=False))
        # Explicit output only: the default check touches no experiment directory.
        if "--out" in os.sys.argv:
            dest = Path(args.out)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "cpu_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    elif args.command == "run":
        run(args)
    else:
        report(Path(args.out))
