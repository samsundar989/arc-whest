"""D2b — fit the coefficient layer on FIT with CMA-ES.

    python tools/fit.py --candidate estimator_k3_v47_fit.py --budget 400

Minimises mean final-layer MSE over a **FIT** minibatch. FIT is the only panel a
coefficient may ever be fitted on: DEV is what every candidate is scored on and
HOLDOUT is read at most twice a week (`AGENTS.md` rule 5). The two are disjoint
from FIT by seed *and* by sha256 of the weight tensors, checked over all 200
nets, so nothing fitted here has seen a net it will later be judged on.

**The evaluator is the cupy backend.** Per-net it disagrees with numpy by ~1e-3,
which is far too coarse to *report* a score — but a fit only needs the ranking
of nearby coefficient vectors, and on a 25-net minibatch the panel-mean
disagreement is ~3e-5. Anything the fit finds is re-measured on numpy before it
becomes a row. This is the whole reason the GPU backend was built: 1.37 s/net
against 50 s/net makes a 400-evaluation fit an overnight job instead of a
fortnight.

Runs as ONE persistent process: the FIT weights and truth are loaded once and
the candidate module is imported once, after which an evaluation is just
`mod._COEFFS = ...` plus a forward pass. Every evaluation is appended to a JSONL
log, so a killed run is resumable and the search is auditable.

The minibatch is FIXED (a rotation seeded once) rather than resampled per
evaluation: CMA-ES compares candidates against each other, and a batch that
moves underneath it turns sampling noise into a gradient signal.
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
sys.path.insert(0, str(LOCAL_SIM))
if BACKEND == "cupy":
    sys.path.insert(0, str(GPU_SHIM))
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "tools"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np  # noqa: E402

#: Which profiles are fitted, in order. Each contributes 4 knots.
PROFILES = ("lam", "theta", "feed")
KNOTS = 4

#: Knots are searched in log space around 1.0 so the optimiser cannot propose a
#: negative or zero gain, and so a 10% change costs the same step size whether
#: the profile currently sits at 0.5 or at 2.0.
LOG_BOUND = float(os.environ.get("FIT_LOG_BOUND", "0.7"))  # e^0.7 ~ 2.0x


def vec_to_coeffs(v):
    out = {}
    for i, name in enumerate(PROFILES):
        out[name] = [float(np.exp(np.clip(x, -LOG_BOUND, LOG_BOUND)))
                     for x in v[i * KNOTS:(i + 1) * KNOTS]]
    return out


class _MLP:
    __slots__ = ("weights", "width", "depth", "seed")

    def __init__(self, weights, width, depth, seed):
        self.weights, self.width, self.depth, self.seed = weights, width, depth, seed


class _Ctx:
    def __init__(self, width, depth, budget, seed):
        self.width, self.depth = width, depth
        self.flop_budget, self.seed = budget, seed
        self.api_version, self.scratch_dir, self.submission_dir = "1.0", None, None


BUDGET = 2 ** 41


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidate", default="estimator_k3_v47_fit.py")
    ap.add_argument("--panel", default="fit")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--budget", type=int, default=400, help="max evaluations")
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out", default="experiments/2026-09-22_d2/fit_run.jsonl")
    ap.add_argument("--kill-after", type=int, default=300,
                    help="PLAN's kill rule: stop if <5%% gain after this many evals")
    a = ap.parse_args(argv)

    import importlib.util
    import panels
    import real_data as rd

    xp = None
    if BACKEND == "cupy":
        import cupy as xp  # noqa: F401

    cand = Path(a.candidate)
    if not cand.is_absolute():
        cand = ROOT / cand
    spec = importlib.util.spec_from_file_location("_fit_candidate", cand)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_fit_candidate"] = mod
    spec.loader.exec_module(mod)

    rows = panels.nets(a.panel)
    rng = np.random.default_rng(a.seed)
    pick = sorted(rng.choice(len(rows), size=min(a.batch, len(rows)), replace=False).tolist())
    batch = [rows[i] for i in pick]
    print(f"[fit] {cand.name} on {a.panel.upper()} batch of {len(batch)} "
          f"(fixed, seed {a.seed}), backend {BACKEND}", flush=True)

    # Load the batch ONCE. Weights go to the device in the dtype the grader uses.
    W, T = [], []
    for split, idx, seed, _name in batch:
        w, tr, _ = rd.get(idx, split, dtype=np.float32)
        wf = []
        for x in w:
            arr = np.asarray(x, dtype=np.float64)
            arr.setflags(write=False)
            wf.append(xp.asarray(arr) if xp is not None else arr)
        W.append((wf, int(seed)))
        T.append(tr[-1].astype(np.float32))
    depth, width = w.shape[0], w.shape[1]
    T = np.stack(T)
    print(f"[fit] batch loaded: depth={depth} width={width}", flush=True)

    est = mod.Estimator()
    try:
        est.setup(_Ctx(width, depth, BUDGET, 0))
    except Exception as exc:
        print(f"[fit] setup failed ({type(exc).__name__}), continuing", flush=True)

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.open("a", encoding="utf-8")
    state = {"n": 0, "best": float("inf"), "best_v": None, "t0": time.time()}

    def evaluate(v) -> float:
        coeffs = vec_to_coeffs(v)
        mod._COEFFS = dict(mod._COEFFS, **coeffs)
        se = 0.0
        try:
            for (wf, seed), t in zip(W, T):
                raw = est.predict(_MLP(wf, width, depth, seed), BUDGET)
                p = np.asarray(xp.asnumpy(raw) if xp is not None else raw,
                               dtype=np.float32)[-1]
                se += float(((p - t) ** 2).mean())
            mse = se / len(W)
            if not np.isfinite(mse):
                mse = float("inf")
        except Exception as exc:
            print(f"[fit] eval raised {type(exc).__name__}: {exc}", flush=True)
            mse = float("inf")
        state["n"] += 1
        better = mse < state["best"]
        if better:
            state["best"], state["best_v"] = mse, list(map(float, v))
        log.write(json.dumps({"i": state["n"], "mse": mse, "best": state["best"],
                              "coeffs": coeffs, "t": round(time.time() - state["t0"], 1)}) + "\n")
        log.flush()
        if better or state["n"] % 10 == 0:
            print(f"[fit] {state['n']:4d}/{a.budget}  mse {mse:.6e}  "
                  f"best {state['best']:.6e}  {'*' if better else ''}", flush=True)
        return mse

    # The baseline: the frozen constants, i.e. every profile exactly 1.0.
    mod._COEFFS = {k: None for k in mod._COEFFS}
    base = evaluate(np.zeros(len(PROFILES) * KNOTS))
    print(f"[fit] baseline (frozen constants) mse = {base:.9e}", flush=True)

    import cma
    es = cma.CMAEvolutionStrategy(
        [0.0] * (len(PROFILES) * KNOTS), a.sigma,
        {"seed": a.seed, "maxfevals": a.budget, "verbose": -9,
         "bounds": [-LOG_BOUND, LOG_BOUND]})
    while not es.stop() and state["n"] < a.budget:
        xs = es.ask()
        es.tell(xs, [evaluate(x) for x in xs])
        if state["n"] >= a.kill_after and state["best"] > base * 0.95:
            print(f"[fit] KILL RULE: {state['n']} evaluations, best is only "
                  f"{(1 - state['best'] / base) * 100:.2f}% better than the frozen "
                  f"constants (PLAN wants >=5%). Stopping.", flush=True)
            break

    gain = (1.0 - state["best"] / base) if base else 0.0
    print()
    print(f"[fit] evaluations   {state['n']}")
    print(f"[fit] baseline mse  {base:.9e}")
    print(f"[fit] best mse      {state['best']:.9e}")
    print(f"[fit] raw gain      {gain:+.2%}   (PLAN D2 gate: >=10% on DEV)")
    best = ROOT / a.out.replace(".jsonl", "_best.json")
    best.write_text(json.dumps(
        {"candidate": cand.name, "panel": a.panel, "batch_indices": pick,
         "evaluations": state["n"], "baseline_mse": base, "best_mse": state["best"],
         "raw_gain": gain, "coeffs": vec_to_coeffs(state["best_v"] or [0.0] * 12),
         "profiles": list(PROFILES), "seed": a.seed}, indent=1) + "\n",
        encoding="utf-8")
    print(f"[fit] wrote {best}")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
