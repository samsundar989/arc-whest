"""P2-CP1: the Phase-2 baseline ladder, through ONE canonical evaluator.

Runs every arm over the same networks under `lifecycle_eval.evaluate` -- one
`Estimator`, `setup()` once and off-budget, a per-MLP `BudgetContext`, all six
grader failure paths live, immutable evidence rows. No arm gets a private path.

## What it reports, and why each piece is there

Per arm, per MLP:

  * **canonical arithmetic score** `mse * max(0.1, (F + lambda R)/B)` from
    `competition_score`, which is the only place that identity lives;
  * **ordinary MSE against each replica**, and the **cross-product estimate**
    `mean_j (p_j - y1_j)(p_j - y2_j)`, whose expectation is the prediction's MSE
    because `E[e1 e2] = 0`. A single-replica MSE is biased upward by exactly the
    truth noise; the cross product removes that bias without pretending the
    labels are exact;
  * **truth noise** beside every MSE (`docs/VALIDATION.md` section 3.5);
  * **MSE by depth**, which is the mechanism curve. ARC's propagation predicts
    `MSE_K ~ c_K (L/n)^K`, so the shape of error against depth is the thing that
    says whether the analytic family is behaving as the theory claims -- and it
    is measurable here for free because every arm returns all layers;
  * **exact counted cost under BOTH flopscope releases**, because four primitives
    reprice between 0.10 and 0.12 (`local_sim/p2_price_delta.py`);
  * residual, peak memory, failures, and the paired interval with its
    biggest-mover share.

Comparisons are **paired on common random numbers** -- every arm sees the same
networks and the same truth -- and the binding statistic is the paired
ARITHMETIC score difference with a studentised-t or basic-bootstrap interval,
via `competition_score.decide()`. Percentile and BCa are refused there, not here.

## What it will NOT do

It will not choose anything on public mini/full: it reads a fresh DESIGN truth
file and nothing else. It will not spend a block -- `p2_truth.py` does that, once,
inside the contract. And `decide()` requires a pre-registered gate, so this module
cannot invent one.

Run:
    uv run python local_sim/p2_lab.py --truth experiments/<dir>/truth.npz \\
        --out experiments/<dir>/ladder.json
    uv run python local_sim/p2_lab.py --truth ... --arms k1,k2p1,k2p2 --n 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

import competition_score as cs  # noqa: E402
import lifecycle_eval as le  # noqa: E402
from phase2_manifest import load as load_manifest  # noqa: E402

ARMS_DIR = os.path.join(_HERE, "p2_arms")

DEFAULT_ARMS = [
    "k1", "k2p1", "k2p2", "k2p3", "k2p4", "k2p6", "k2p1_f32",
    "sample_iid_max", "sample_iid_floor", "sample_orth", "sample_lattice",
]

#: The arm every other arm is compared against. `k2p1` is BITWISE the shipped
#: `estimator.py` analytic path at this shape (`p2_test_arms.py` T2), so the
#: control is the thing being controlled, not a re-derivation of it.
CONTROL = "k2p1"


def _cross_mse(pred, y1, y2):
    """mean_j (p - y1)(p - y2). Unbiased for MSE(p) when E[e1 e2] = 0."""
    return float(((pred - y1) * (pred - y2)).mean())


def _run_arm(name, seeds, truth_final, spec, gt_noise):
    path = os.path.join(ARMS_DIR, f"arm_{name}.py")
    if not os.path.exists(path):
        raise SystemExit(f"no such arm: {path}. Run p2_build_arms.py first.")
    return le.evaluate(path, seeds, truth_final, spec=spec, gt_noise=gt_noise)


def _cost_under_012(name, width, depth, budget):
    """Counted cost of one predict under flopscope 0.12, in the isolated venv.

    Returns None when `.venv-fs012` is absent. Four primitives reprice between
    0.10 and 0.12 and one of them (`where` with a Python scalar) HALVES, so a
    cost quoted from 0.10 alone is not a Phase-2 cost.
    """
    exe = os.path.join(_ROOT, ".venv-fs012", "Scripts", "python.exe")
    if not os.path.exists(exe):
        return None
    code = (
        "import sys,os,importlib.util as u,types,numpy as np\n"
        "import flopscope as flops\n"
        "wb=types.ModuleType('whestbench')\n"
        "class MLP:\n"
        " def __init__(s,weights,width,depth,seed):\n"
        "  s.weights,s.width,s.depth,s.seed=weights,width,depth,seed\n"
        "class BaseEstimator:\n"
        " def setup(s,c):pass\n"
        "wb.MLP=MLP;wb.BaseEstimator=BaseEstimator;wb.__path__=[]\n"
        "sys.modules['whestbench']=wb\n"
        f"W,D,B={width},{depth},{budget}\n"
        "ss=np.random.SeedSequence(1000003).spawn(3)\n"
        "r=np.random.default_rng(ss[0]);sc=float(np.sqrt(2.0/W))\n"
        "wt=[(r.standard_normal((W,W))*sc).astype(np.float32) for _ in range(D)]\n"
        "m=MLP(wt,W,D,int(ss[2].generate_state(1)[0]))\n"
        f"sp=u.spec_from_file_location('a',r'{os.path.join(ARMS_DIR, f'arm_{name}.py')}')\n"
        "mo=u.module_from_spec(sp);sp.loader.exec_module(mo);e=mo.Estimator()\n"
        "class C:width,depth,flop_budget,seed=W,D,B,0\n"
        "e.setup(C())\n"
        "with flops.BudgetContext(flop_budget=B,quiet=True) as c:\n"
        "    e.predict(m,B)\n"
        "    print('@@',float(c.flops_used))\n"
    )
    r = subprocess.run([exe, "-c", code], capture_output=True, text=True,
                       cwd=_ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    for ln in r.stdout.splitlines():
        if ln.startswith("@@"):
            return float(ln[2:])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--n", type=int, default=0, help="0 = every network in the truth file")
    ap.add_argument("--gate", type=float, default=None,
                    help="pre-registered promotion gate; REQUIRED to print a verdict")
    a = ap.parse_args()

    man = load_manifest()
    width, depth = man["shape"]["width"], man["shape"]["depth"]
    budget = man["budget"]["flop_budget_per_mlp"]
    floor = man["budget"]["multiplier_floor"]
    parity = man["package_parity"]

    z = np.load(a.truth)
    seeds = [int(s) for s in z["seeds"]]
    n = len(seeds) if a.n == 0 else min(a.n, len(seeds))
    seeds = seeds[:n]
    y1 = z["means_r1"][:n, -1, :].astype(np.float64)
    y2 = z["means_r2"][:n, -1, :].astype(np.float64)
    all1 = z["means_r1"][:n].astype(np.float64)
    all2 = z["means_r2"][:n].astype(np.float64)
    gt_noise = z["gt_noise"][:n].astype(np.float64)
    y_mid = ((y1 + y2) / 2.0).astype(np.float32)

    is_probe = bool(int(z["probe"])) if "probe" in z.files else False
    if is_probe:
        print("!" * 92)
        print("!!  PROBE TRUTH -- unregistered seeds, no block claimed. This run is a")
        print("!!  PIPELINE SMOKE. Nothing it prints may select an arm, a constant, a")
        print("!!  rank, or a gate, and no number from it belongs in a durable doc.")
        print("!" * 92)

    spec = le.LifecycleSpec(width=width, depth=depth, flop_budget=budget, floor=floor)
    arms = [x for x in a.arms.split(",") if x]

    print("=" * 92)
    print("P2-CP1 LADDER  --  one canonical evaluator, paired on common random numbers")
    print("=" * 92)
    print(f"  truth      {os.path.relpath(a.truth, _ROOT)}   {n} networks   "
          f"2 replicas x {int(z['n_samples']):,} samples")
    print(f"  truth noise per replica   mean {gt_noise.mean():.4e}")
    print(f"  shape      {width} x {depth}   B = {budget:,}   floor {floor}")
    print(f"  stack      flopscope {parity['installed_here']['flopscope']}"
          f"   PARITY {parity['status']}")
    if parity["limitation_stamp"]:
        print(f"  STAMP      {parity['limitation_stamp'][:74]}...")
    print()

    results = {}
    for name in arms:
        t0 = time.perf_counter()
        rows, prov = _run_arm(name, seeds, y_mid, spec, list(gt_noise))
        suite = le.score(rows, spec=spec)
        preds = None                       # re-derive per-MLP predictions cheaply
        # lifecycle_eval scores the final layer; for the depth curve and the
        # cross-product estimate we need the arm's own arrays, so re-run the
        # predict outside the budget (it is deterministic, so this is the same
        # numbers, and no score is computed from this path).
        preds, alld = _predict_all(name, seeds, spec)
        cross = np.array([_cross_mse(preds[i], y1[i], y2[i]) for i in range(n)])
        mse1 = np.array([float(((preds[i] - y1[i]) ** 2).mean()) for i in range(n)])
        mse2 = np.array([float(((preds[i] - y2[i]) ** 2).mean()) for i in range(n)])
        by_depth = np.array([
            [float((((alld[i][d] - (all1[i][d] + all2[i][d]) / 2.0)) ** 2).mean())
             for d in range(depth)] for i in range(n)])
        c012 = _cost_under_012(name, width, depth, budget)
        results[name] = dict(
            rows=rows, prov=prov, suite=suite, cross=cross, mse1=mse1, mse2=mse2,
            by_depth=by_depth, cost012=c012, secs=time.perf_counter() - t0,
        )
        f = float(np.mean([r.flops_used for r in rows]))
        eff = float(np.mean([cs.effective_compute(r.flops_used,
                                                  r.residual_wall_time_s, spec.lam)
                             for r in rows]))
        print(f"  {name:<18} score {suite.objective:.5e}   "
              f"MSE(x-prod) {cross.mean():.4e}   MSE(r1) {mse1.mean():.4e}   "
              f"F/B {f / budget:.4f}   eff {eff / budget:.4f}   "
              f"mult {suite.mean_multiplier:.4f}   fail {suite.n_failed}/{n}   "
              f"{results[name]['secs']:.0f}s"
              + ("" if c012 is None else
                 f"   [0.12: {c012 / budget:.6f}"
                 + ("" if abs(c012 - f) < 1 else f"  MOVED {c012 / f:.6f}x") + "]"))

    # ---- paired comparisons against the control ---------------------------
    print()
    print(f"  PAIRED vs {CONTROL} -- arithmetic score difference, "
          f"studentised-t 95% (the binding statistic, docs/VALIDATION.md 2b-i)")
    print(f"  {'arm':<18} {'gain':>8} {'95% t CI (relative)':>26} {'better':>9} "
          f"{'sign p':>8}")
    ctrl_rows = results[CONTROL]["rows"]
    paired = {}
    for name in arms:
        if name == CONTROL:
            continue
        res = cs.paired_compare_rows(
            [r.to_row() for r in ctrl_rows],
            [r.to_row() for r in results[name]["rows"]],
            lam=spec.lam, floor=floor,
        )
        paired[name] = res
        # The BINDING interval is the studentised t on the ABSOLUTE difference
        # (docs/VALIDATION.md 2b-i). It is shown relative to the control's mean
        # purely for readability; the decision is made by `decide()`, which takes
        # the pre-registered gate and refuses percentile/BCa.
        lo, hi = res.t_ci_difference
        m = res.mean_control or float("nan")
        print(f"  {name:<18} {res.arithmetic_gain:>8.4f} "
              f"[{lo / m:>+10.2%}, {hi / m:>+10.2%}] {res.n_candidate_better:>4}/{res.n:<4} "
              f"p={res.sign_test_p:>6.3f}")

    # ---- the mechanism curve ----------------------------------------------
    print()
    print("  MSE BY DEPTH -- the (L/n)^K mechanism curve, cross-replica mean truth")
    hdr = "  " + f"{'arm':<18}" + "".join(f"{'L' + str(d + 1):>11}"
                                          for d in (0, 1, 3, 7, 11, 15))
    print(hdr)
    for name in arms:
        bd = results[name]["by_depth"].mean(axis=0)
        print("  " + f"{name:<18}" + "".join(f"{bd[d]:>11.3e}"
                                             for d in (0, 1, 3, 7, 11, 15)))

    if a.out:
        _write(a.out, results, paired, man, seeds, z, spec, arms)
        print(f"\n  artifact -> {os.path.relpath(a.out, _ROOT)}")
    print("=" * 92)
    return 0


def _predict_all(name, seeds, spec):
    """Deterministic re-run outside the budget, for the depth curve only.

    No score is computed from this path -- `lifecycle_eval` owns every scored
    number. This exists because the evaluator keeps the final layer and the
    mechanism curve needs all of them.
    """
    import flopscope as flops

    import harness
    from mint_provenance import canonical_mint

    mod, _ = le.load_candidate(os.path.join(ARMS_DIR, f"arm_{name}.py"))
    est = mod.Estimator()
    est.setup(le.SetupContext(spec))
    finals, alls = [], []
    for s in seeds:
        w, es, _ = canonical_mint(spec.width, spec.depth, int(s))
        mlp = harness.MLP(weights=[np.asarray(x) for x in w], width=spec.width,
                          depth=spec.depth, seed=es)
        with flops.BudgetContext(flop_budget=spec.flop_budget, quiet=True):
            p = np.asarray(est.predict(mlp, spec.flop_budget), dtype=np.float64)
        alls.append(p)
        finals.append(p[-1])
    return np.asarray(finals), np.asarray(alls)


def _write(path, results, paired, man, seeds, z, spec, arms):
    if os.path.exists(path):
        raise SystemExit(f"REFUSED: {path} exists. Evidence is immutable.")
    payload = {
        "artifact": "p2_cp1_ladder",
        "is_probe_not_evidence": bool(int(z["probe"])) if "probe" in z.files else False,
        "artifact_version": 1,
        "shape": man["shape"],
        "budget": man["budget"],
        "package_parity": man["package_parity"],
        "truth": {
            "n_samples_per_replica": int(z["n_samples"]),
            "n_networks": len(seeds),
            "replicas": 2,
            "gt_noise_mean": float(z["gt_noise"].mean()),
        },
        "seeds": [int(s) for s in seeds],
        "lambda_used": spec.lam,
        "lambda_provenance": "PHASE-1 value 1e11; Phase-2 lambda is UNPUBLISHED",
        "residual_is_graded": False,
        "control": CONTROL,
        "arms": {},
    }
    for name in arms:
        r = results[name]
        payload["arms"][name] = {
            "objective": r["suite"].objective,
            "mean_mse_r1": float(r["mse1"].mean()),
            "mean_mse_r2": float(r["mse2"].mean()),
            "mean_mse_cross": float(r["cross"].mean()),
            "per_mlp_cross": [float(x) for x in r["cross"]],
            "mean_flops": float(np.mean([x.flops_used for x in r["rows"]])),
            "flops_012": r["cost012"],
            "n_failed": r["suite"].n_failed,
            "mean_multiplier": r["suite"].mean_multiplier,
            "mse_by_depth": [float(x) for x in r["by_depth"].mean(axis=0)],
            "rows_hash": le.rows_hash(r["rows"]),
            "setup_seconds": r["prov"]["setup_seconds"],
        }
        if name in paired:
            p = paired[name]
            payload["arms"][name]["paired_vs_control"] = {
                "arithmetic_gain": p.arithmetic_gain,
                "mean_difference": p.mean_difference,
                "t_ci_difference": list(p.t_ci_difference),
                "boot_ci_difference": list(p.boot_ci_difference),
                "relative_improvement": p.relative_improvement,
                "geometric_gain_DIAGNOSTIC_ONLY": p.geometric_gain,
                "n_candidate_better": p.n_candidate_better,
                "n": p.n,
                "sign_test_p": p.sign_test_p,
                "estimands_disagree": p.estimands_disagree,
            }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, indent=1, default=str) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
