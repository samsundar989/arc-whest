"""Fit a probe knob vector on FIT with CMA-ES. Screening instrument (cupy).

    python tools/probe_fit.py <candidate.py> --base '{"_LBLOCK_AGE": 1}'
        --form vec16|exp2 --budget 150 --out experiments/<dir>/<name>.jsonl

T3 (HANDOFF 2026-09-22 Step 4): `_RHO` is a 16-vector added on top of the base
arm. Two parametrisations:

  vec16  rho[l] free for every layer that has an old lineage (the rest stay 0).
  exp2   rho[l] = c * sum_{a = A+1}^{l-1} exp(-a / tau), A = _LBLOCK_AGE:
         one age-weighted proxy per old lineage, summed -- the handoff's
         2-parameter form, which needs no per-lineage storage because every old
         lineage is proxied by the same age-0 L-block.

Unlike `tools/fit.py` the knobs are LINEAR (rho can be negative or large), the
search starts at rho = 0 (== the base arm exactly), and the batch is FIT only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe  # noqa: E402  (sets up sys.path for the shims)

import numpy as np  # noqa: E402


def rho_from(v, form, depth, age):
    rho = [0.0] * depth
    if form == "vec16":
        act = [l for l in range(depth) if l - 1 > age]      # layers with an old lineage
        for l, x in zip(act, v):
            rho[l] = float(x)
    else:
        c, ltau = float(v[0]), float(v[1])
        tau = math.exp(ltau)
        for l in range(depth):
            rho[l] = c * sum(math.exp(-a / tau) for a in range(age + 1, l))
    return rho


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("candidate")
    ap.add_argument("--base", default='{"_LBLOCK_AGE": 1}')
    ap.add_argument("--form", default="vec16", choices=["vec16", "exp2"])
    ap.add_argument("--panel", default="fit")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import panels
    import real_data as rd
    import cupy as xp
    # CAP THE POOL. Uncapped, cupy's pool grew to the whole 32 GiB card over
    # 19 nets and the run wedged on CUDA_ERROR_OUT_OF_MEMORY (2026-09-23).
    xp.get_default_memory_pool().set_limit(
        size=int(float(os.environ.get('PROBE_GPU_LIMIT_GIB', '20')) * 2**30))
    import cma

    base = json.loads(a.base)
    age = int(base.get("_LBLOCK_AGE", 1))
    cand = Path(a.candidate)
    if not cand.is_absolute():
        cand = probe.ROOT / cand
    mod = probe.load_module(cand)
    for k, v in base.items():
        setattr(mod, k, v)

    rows = panels.nets(a.panel)
    rng = np.random.default_rng(a.seed)
    pick = sorted(rng.choice(len(rows), size=min(a.batch, len(rows)), replace=False).tolist())
    W, T = [], []
    for i in pick:
        split, idx, seed, _ = rows[i]
        w, tr, _ = rd.get(idx, split, dtype=np.float32)
        W.append(([xp.asarray(np.asarray(x, dtype=np.float64)) for x in w], int(seed)))
        T.append(tr[-1].astype(np.float32))
    depth, width = w.shape[0], w.shape[1]
    est = mod.Estimator()
    est.setup(probe._Ctx(width, depth, probe.BUDGET, 0))
    print(f"[pfit] {cand.name} base {base} form {a.form} on {a.panel.upper()} "
          f"batch {len(W)} (seed {a.seed})", flush=True)

    out = Path(a.out)
    if not out.is_absolute():
        out = probe.ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.open("a", encoding="utf-8")
    st = {"n": 0, "best": float("inf"), "best_rho": None, "t0": time.time()}

    def evaluate(v) -> float:
        rho = rho_from(v, a.form, depth, age)
        mod._RHO = rho if any(rho) else None
        se = 0.0
        for (wf, seed), t in zip(W, T):
            raw = est.predict(probe._MLP(wf, width, depth, seed), probe.BUDGET)
            p = np.asarray(xp.asnumpy(raw), dtype=np.float32)[-1]
            se += float(((p - t) ** 2).mean(dtype=np.float32))
        xp.get_default_memory_pool().free_all_blocks()
        mse = se / len(W)
        if not np.isfinite(mse):
            mse = float("inf")
        st["n"] += 1
        if mse < st["best"]:
            st["best"], st["best_rho"] = mse, rho
        log.write(json.dumps({"i": st["n"], "mse": mse, "best": st["best"], "v": list(map(float, v)),
                              "rho": rho, "t": round(time.time() - st["t0"], 1)}) + "\n")
        log.flush()
        print(f"[pfit] {st['n']:4d}  mse {mse:.6e}  best {st['best']:.6e}", flush=True)
        return mse

    dim = sum(1 for l in range(depth) if l - 1 > age) if a.form == "vec16" else 2
    x0 = [0.0] * dim if a.form == "vec16" else [0.0, math.log(2.0)]
    base_mse = evaluate([0.0] * dim if a.form == "vec16" else [0.0, math.log(2.0)])
    es = cma.CMAEvolutionStrategy(x0, a.sigma, {"seed": a.seed, "maxfevals": a.budget,
                                                "verbose": -9})
    while not es.stop() and st["n"] < a.budget:
        xs = es.ask()
        es.tell(xs, [evaluate(x) for x in xs])
    res = {"candidate": cand.name, "base": base, "form": a.form, "panel": a.panel,
           "batch_indices": pick, "evaluations": st["n"], "base_mse": base_mse,
           "best_mse": st["best"], "gain": 1 - st["best"] / base_mse, "rho": st["best_rho"]}
    Path(str(out).replace(".jsonl", "_best.json")).write_text(json.dumps(res, indent=1) + "\n",
                                                              encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
