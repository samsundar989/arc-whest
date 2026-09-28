"""Compare value-only and slice-consistent K111 diagnostic interventions.

This is an offline diagnostic, never a submitted estimator. The original oracle replaces
D3/D21 after _advance, then subtracts those replaced values from the new repeated-slice
birth while _wick_tower still carries the original tensor. This probe changes ONLY those
two compensation operands back to the actual advanced tensor's slices. Closure outputs
and direct mean consumers still see the injected values. No shared source is modified.

  .venv-gpu/Scripts/python.exe tools/k111_consistent_probe.py small --out <directory>
  workerq submit ... -- .venv-gpu/Scripts/python.exe tools/k111_consistent_probe.py run \
      --nets 0-1 --arms AD,rank32,FULL --out <absolute-directory>

Optimization review: one net at a time, no autograd or stored layer states, existing
labels only, CPU thread cap, JSON checkpoint per net. Expected conservative envelope
16 GiB host RAM / 10 GiB VRAM / 2 CPUs for the default FIT-2 comparison. Small mode is
CPU-only n=24 and uses no dataset. A positive diagnostic is not a realizable predictor.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch_layer(mod, out):
    """Install a second _layer function in the existing module's globals.

    Capture the actual tensor slices immediately before their diagnostic hooks, and
    use those only in p111 compensation. The hook outputs remain the closure inputs.
    """
    src = inspect.getsource(mod._layer)
    replacements = [
        ("    D3 = _a2('D3', D3)\n",
         "    _comp_D3 = D3\n    D3 = _a2('D3', D3)\n"),
        ("        D21 = _a2('D21', D21)\n",
         "        _comp_D21 = D21\n        D21 = _a2('D21', D21)\n"),
    ]
    for old, new in replacements:
        if src.count(old) != 1:
            raise RuntimeError(f"compensation capture anchor occurs {src.count(old)}x: {old!r}")
        src = src.replace(old, new, 1)
    src, n3 = re.subn(r"(p111_3 = [^\n]+\* )D3(?=\n)", r"\1_comp_D3", src)
    src, n21 = re.subn(r"(p111_21 = [^\n]+\* )D21(?=\n)", r"\1_comp_D21", src)
    if (n3, n21) != (1, 1):
        raise RuntimeError(f"compensation anchors: D3={n3}, D21={n21}")
    legacy = mod._layer
    path = Path(out) / "_consistent_layer.py"
    path.write_text(src, encoding="utf-8")
    exec(compile(src, str(path), "exec"), mod.__dict__)
    consistent = mod._layer
    mod._layer = legacy
    return legacy, consistent


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def small(a):
    """Actual source regression: full-rank P2, no shared compression, tiny CPU arrays."""
    os.environ["TWIN_DEVICE"] = "cpu"
    import sys
    sys.path.insert(0, str(ROOT / "tools" / "twin"))
    import twin as T
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mod = T.load_module(Path(a.cand), name="_consistent_small")
    mod._P2_RANK = 0
    mod._SHARED_ON = False
    mod._STRASSEN_DEPTH = 0
    mod._CORR_ON = False
    legacy, consistent = patch_layer(mod, out)
    n, depth = 24, 4
    dtype = T.fnp.float64
    est = mod.Estimator()
    est.setup(T._Ctx(n, depth, T.BUDGET, 0))
    est._wick_program = None
    const, table = est._constants(n, dtype), mod._tables()
    off = const["off_mask"]
    gen = torch.Generator(device="cpu").manual_seed(20260928)
    weights = [torch.randn((n, n), generator=gen, dtype=torch.float64) * (2.0 / n) ** .5
               for _ in range(depth)]
    st0 = (T.fnp.zeros(n, dtype=dtype), T.fnp.eye(n, dtype=dtype), None, 0.0,
           T.fnp.zeros(n, dtype=dtype))
    mod._A2 = None
    st = legacy(weights[0], st0, table, dtype, const, 2.0, False, None, 0, depth)
    idx = torch.arange(n, dtype=torch.float64)
    delta3 = 0.003 * torch.cos(idx)
    delta21 = 0.0003 * torch.sin(idx[:, None] * .7 + idx[None, :] * .2) * off

    def norm(x):
        return float(torch.linalg.vector_norm(x))

    rows = {}
    outputs = {}
    for name, fn, inject in (("base", legacy, False), ("consistent_identity", consistent, False),
                             ("legacy_inject", legacy, True), ("consistent_inject", consistent, True)):
        rec = {}

        def hook(l, key, value, *aux):
            if key == "K":
                rec["K"] = {k: v.detach().clone() for k, v in value.items()}
            elif key in ("D3", "D21", "mu", "var"):
                rec[key] = value.detach().clone()
            if inject and key == "D3":
                return value + delta3
            if inject and key == "D21":
                return value + delta21
            return value

        mod._A2 = hook
        s = fn(weights[1], st, table, dtype, const, 2.0, False, None, 1, depth)
        adv = mod._advance(T.fnp.eye(n, dtype=dtype), s[2])
        k3, k21 = mod._dslice3(adv), mod._dslice21(adv, off)
        K = rec["K"]
        sig = rec["var"].sqrt()
        al = (rec["mu"] / sig).clamp(-mod.ALPHA_CAP, mod.ALPHA_CAP)
        w1 = mod._wick(rec["mu"], sig, al, table, dtype, const["wick"])[(1, 1)]
        predicted3 = -(w1 ** 3) * delta3
        predicted21 = -(w1[:, None] ** 2) * w1[None, :] * delta21
        rows[name] = {
            "K3_slice_error_abs": norm(k3 - K[(3,)]),
            "K21_slice_error_abs": norm(k21 - K[(2, 1)]),
            "K3_slice_error_relative": norm(k3 - K[(3,)]) / norm(K[(3,)]),
            "K21_slice_error_relative": norm(k21 - K[(2, 1)]) / norm(K[(2, 1)]),
            "legacy_formula_K3_residual": norm((k3 - K[(3,)]) - predicted3),
            "legacy_formula_K21_residual": norm((k21 - K[(2, 1)]) - predicted21),
        }
        outputs[name] = (s[0], s[1], k3, k21, K)
    b, c = outputs["base"], outputs["consistent_identity"]
    x, y = outputs["legacy_inject"], outputs["consistent_inject"]
    checks = {
        "identity_mean_bit_equal": torch.equal(b[0], c[0]),
        "identity_cov_bit_equal": torch.equal(b[1], c[1]),
        "identity_K3_bit_equal": torch.equal(b[2], c[2]),
        "identity_K21_bit_equal": torch.equal(b[3], c[3]),
        "injected_mean_bit_equal": torch.equal(x[0], y[0]),
        "injected_cov_bit_equal": torch.equal(x[1], y[1]),
        "injected_closure_K_bit_equal": all(torch.equal(x[4][k], y[4][k]) for k in x[4]),
        "legacy_mismatch_observed": rows["legacy_inject"]["K21_slice_error_abs"] > 1e-5,
        "legacy_formula_verified": rows["legacy_inject"]["legacy_formula_K3_residual"] < 1e-12
                                   and rows["legacy_inject"]["legacy_formula_K21_residual"] < 1e-12,
        "consistent_slices_preserved": rows["consistent_inject"]["K3_slice_error_abs"] < 1e-12
                                       and rows["consistent_inject"]["K21_slice_error_abs"] < 1e-12,
    }
    report = {"width": n, "depth": depth, "P2_rank": "full", "shared": False,
              "candidate": str(a.cand), "rows": rows, "checks": checks, "pass": all(checks.values())}
    write_json(out / "small.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


def run(a):
    import k111_diag as K
    import torch
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    a.h, a.max_layer, a.gain_file = .05, 99, ""
    d = K.Diag(a)
    legacy, consistent = patch_layer(d.mod, out)
    off, off32 = d.off64, d.off64.float()
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    allowed = {"AD", "FULL", "REP", "CMP", "AD_D3only", "AD_D21only"}
    if any(s not in allowed and not re.fullmatch(r"rank\d+", s) for s in arms):
        raise ValueError(f"unknown arms: {arms}")
    lo, hi = map(int, a.nets.split("-"))
    for gi in range(lo, hi + 1):
        path = out / f"net_{gi:03d}.json"
        if path.exists():
            print(f"reuse {path.name}", flush=True)
            continue
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        w, truth = d.data.net("full", gi)
        labels = d.labels(gi)
        lp = {l: d.label_parts(labels, l, w[l + 1].double()) for l in range(d.depth - 1)}
        result = {"net": gi, "candidate": str(a.cand), "params": str(a.params),
                  "final_label_noise_floor_approx": 7.7e-11, "semantics": {}}
        for semantics, function in (("legacy", legacy), ("consistent", consistent)):
            d.mod._layer = function
            scores = {}
            for arm in ["base"] + arms:
                ctx, kr, mismatch = {}, {}, {}

                def prepare(l, state):
                    Wd = w[l + 1].double()
                    t, S = d.slices(d.mod._advance(d.I32, state[2]), off32)
                    S = S * off
                    K3, K21 = kr["K"]
                    ctx.clear()
                    ctx.update(r3=K.rep3(t, S, Wd), r21=K.rep21(t, S, Wd) * off,
                               rk3=K.rep3(K3.double(), K21.double() * off, Wd),
                               rk21=K.rep21(K3.double(), K21.double() * off, Wd) * off,
                               lp=lp[l])
                    mismatch[l] = {
                        "K3_relative": float((t - K3).norm() / K3.norm()),
                        "K21_relative": float((S - K21).norm() / K21.norm()),
                    }

                def edit(l, name, v):
                    if arm == "base" or not ctx or l == 0:
                        return v
                    suffix = "3" if name == "D3" else "21"
                    vd, p = v.double(), ctx["lp"]
                    if arm.startswith("rank"):
                        if suffix == "3":
                            ctx["e3"] = p["AD3"] - (vd - ctx["r3"])
                            return (vd + ctx["e3"]).to(v.dtype)
                        rank = int(arm[4:])
                        error = (p["AD21"] - (vd * off - ctx["r21"])) * off
                        if rank == 0:
                            correction = error * 0.0
                        elif rank >= d.n:
                            correction = error
                        else:
                            U, s, Vh = torch.linalg.svd(error + torch.diag(ctx["e3"]))
                            correction = ((U[:, :rank] * s[:rank]) @ Vh[:rank]) * off
                        value = vd + correction
                    elif arm == "FULL":
                        value = p["D" + suffix]
                    elif arm == "REP":
                        value = vd - ctx["r" + suffix] + p["rep" + suffix]
                    elif arm == "CMP":
                        value = vd - ctx["r" + suffix] + ctx["rk" + suffix]
                    elif arm == "AD" or arm == "AD_D3only" and suffix == "3" or arm == "AD_D21only" and suffix == "21":
                        value = ctx["r" + suffix] + p["AD" + suffix]
                    else:
                        return v
                    return (value if suffix == "3" else value * off).to(v.dtype)

                hook = K.Hook(record=False)
                hook.edit = edit

                def khook(l, name, value, *aux):
                    if name == "K" and (3,) in value:
                        kr["K"] = (value[(3,)].detach(), value[(2, 1)].detach())
                    return hook(l, name, value, *aux)

                mus, _ = d.chain_run(w, khook, prepare)
                mse = d.mse(mus, truth)
                scores[arm] = {"final": mse[d.depth - 1], "rows": mse,
                               "outgoing_slice_mismatch": dict(mismatch)}
                print(f"net {gi} {semantics} {arm}: final {mse[d.depth - 1]:.9e}, {time.time()-t0:.0f}s", flush=True)
                del mus
            base = scores["base"]["final"]
            for item in scores.values():
                item["relative_final_change"] = item["final"] / base - 1.0
            result["semantics"][semantics] = scores
        result["base_relative_difference"] = result["semantics"]["consistent"]["base"]["final"] / result["semantics"]["legacy"]["base"]["final"] - 1.0
        result["seconds"] = time.time() - t0
        result["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
        write_json(path, result)
        del labels, lp, w
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("small", "run"))
    ap.add_argument("--cand", default=str(ROOT / "estimator_k3_v62_s1.py"))
    ap.add_argument("--params", default=str(ROOT / "experiments/2026-09-25_twin/S3/params_best.npz"))
    ap.add_argument("--nets", default="0-1")
    ap.add_argument("--arms", default="AD,rank32,FULL")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"small": small, "run": run}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
