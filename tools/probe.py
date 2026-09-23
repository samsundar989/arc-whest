"""Multi-arm accuracy probe on the cupy backend. Screening instrument.

    python tools/probe.py <candidate.py> --arms arms.json --panel dev [--n 100]
                          --out experiments/<dir>/<name>.json [--ref T0]

`arms.json` is `{"T0": {}, "T2a1": {"_LBLOCK_AGE": 1}, ...}`: each arm is a set
of module-level overrides applied to ONE imported copy of the candidate. Every
net is loaded once and every arm runs on it, so the arms are paired on common
random numbers by construction and the dataset cost is paid once.

Why not `tools/ev.py`: ev.py keys its cache on the source sha, so a knob sweep
would need one file per arm, and it runs numpy at ~50 s/net. Probe arms are
accuracy-only kill tests; their cost is irrelevant (the real cost is measured
once on the rebuilt chain). cupy agrees with numpy to 3.2e-05 on a 100-net
panel mean (RESULTS.md, "Two accuracy backends"), which is ~1000x inside any
decision these probes feed.

Writes per-net final-layer MSE for every arm, the panel mean, and -- against
`--ref` -- the paired mean ratio of raw MSE, a t-95% interval on the per-net
difference, and wins/N. Resumable: nets already in `--out` are skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL_SIM = ROOT / "local_sim"
GPU_SHIM = ROOT / "tools" / "gpu_shim"

BACKEND = os.environ.get("WHEST_BACKEND", "cupy").lower()
# Same deliberate insert as _predict_worker.py / fit.py: this process reports
# accuracy only, never a FLOP, so the shims in front are the point.
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


class _MLP:
    __slots__ = ("weights", "width", "depth", "seed")

    def __init__(self, weights, width, depth, seed):
        self.weights, self.width, self.depth, self.seed = weights, width, depth, seed


class _Ctx:
    def __init__(self, width, depth, budget, seed):
        self.width, self.depth = width, depth
        self.flop_budget, self.seed = budget, seed
        self.api_version, self.scratch_dir, self.submission_dir = "1.0", None, None


def load_module(path: Path, name: str = "_probe_candidate"):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def apply_arm(mod, defaults: dict, arm: dict) -> None:
    for k, v in defaults.items():
        setattr(mod, k, v)
    for k, v in arm.items():
        setattr(mod, k, v)


def paired(c: np.ndarray, r: np.ndarray) -> dict:
    d = c - r
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    half = 1.96 * sd / math.sqrt(n) if n > 1 else float("nan")
    rm = float(r.mean())
    return {"ratio": float(c.mean() / rm), "gain_pct": float((c.mean() / rm - 1) * 100),
            "ci95_pct": float(half / rm * 100), "wins": int((c < r).sum()), "n": n}


def summarise(res: dict, ref: str | None) -> dict:
    out = {}
    arms = res["arms"]
    for a, per in arms.items():
        v = np.array([per[k] for k in sorted(per, key=int)])
        s = {"raw_mse": float(v.mean()), "n": len(v)}
        if ref and ref in arms and a != ref:
            common = sorted(set(per) & set(arms[ref]), key=int)
            s.update(paired(np.array([per[k] for k in common]),
                            np.array([arms[ref][k] for k in common])))
        out[a] = s
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("candidate")
    ap.add_argument("--arms", required=True, help="JSON file or inline JSON object")
    ap.add_argument("--panel", default="dev")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None)
    a = ap.parse_args(argv)

    import panels
    import real_data as rd

    xp = None
    if BACKEND == "cupy":
        import cupy as xp
        # CAP THE POOL. Uncapped, cupy's pool grew to the whole 32 GiB card over
        # 19 nets and the run wedged on CUDA_ERROR_OUT_OF_MEMORY (2026-09-23).
        xp.get_default_memory_pool().set_limit(
            size=int(float(os.environ.get('PROBE_GPU_LIMIT_GIB', '20')) * 2**30))

    arms_src = a.arms
    arms = json.loads(Path(arms_src).read_text() if Path(arms_src).exists() else arms_src)
    cand = Path(a.candidate)
    if not cand.is_absolute():
        cand = ROOT / cand
    mod = load_module(cand)
    keys = sorted({k for arm in arms.values() for k in arm})
    defaults = {k: getattr(mod, k) for k in keys}

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {"candidate": cand.name, "panel": a.panel, "backend": BACKEND,
           "arms_spec": arms, "arms": {k: {} for k in arms}}
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        for k in arms:
            res["arms"][k] = old.get("arms", {}).get(k, {})

    rows = panels.nets(a.panel)[a.lo:a.lo + a.n]
    est = mod.Estimator()
    t0 = time.time()
    setup_done = False
    for i, (split, idx, seed, _name) in enumerate(rows):
        key = str(a.lo + i)
        todo = [k for k in arms if key not in res["arms"][k]]
        if not todo:
            continue
        w, tr, _ = rd.get(idx, split, dtype=np.float32)
        depth, width = w.shape[0], w.shape[1]
        if not setup_done:
            try:
                est.setup(_Ctx(width, depth, BUDGET, 0))
            except Exception as exc:
                print(f"[probe] setup failed ({type(exc).__name__})", flush=True)
            setup_done = True
        wf = []
        for x in w:
            arr = np.asarray(x, dtype=np.float64)
            arr.setflags(write=False)
            wf.append(xp.asarray(arr) if xp is not None else arr)
        t = tr[-1].astype(np.float32)
        line = []
        for k in todo:
            apply_arm(mod, defaults, arms[k])
            raw = est.predict(_MLP(wf, width, depth, int(seed)), BUDGET)
            p = np.asarray(xp.asnumpy(raw) if xp is not None else raw, dtype=np.float32)[-1]
            if not np.all(np.isfinite(p)):
                raise SystemExit(f"arm {k} net {key}: non-finite prediction")
            mse = float(((p - t) ** 2).mean(dtype=np.float32))
            res["arms"][k][key] = mse
            line.append(f"{k}={mse:.4e}")
        apply_arm(mod, defaults, {})
        if xp is not None:
            xp.get_default_memory_pool().free_all_blocks()
        res["summary"] = summarise(res, a.ref)
        out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        el = time.time() - t0
        print(f"[probe] net {i + 1}/{len(rows)} ({el:.0f}s) " + " ".join(line), flush=True)

    res["summary"] = summarise(res, a.ref)
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print()
    for k, s in res["summary"].items():
        extra = ""
        if "ratio" in s:
            extra = (f"  vs {a.ref}: {s['gain_pct']:+.2f}% raw (+/-{s['ci95_pct']:.2f})"
                     f"  wins {s['wins']}/{s['n']}")
        print(f"  {k:14s} raw {s['raw_mse']:.6e}  n={s['n']}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
