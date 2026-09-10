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
  * residual **and its ratio to the control**, because counted FLOPs and residual
    wall time are priced by completely different mechanisms and an arm can be
    cheap in one and ruinous in the other -- that is what killed the packed lane
    (`docs/GOTCHAS.md`) and it is what the P2-CP1 counted-only freeze hid;
  * peak memory, MEASURED via `tracemalloc` rather than promised. An earlier
    build of this docstring listed peak memory and the code never recorded it,
    which an independent Skeptic caught;
  * failures, and the paired interval with its biggest-mover share.

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
import re
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

import tracemalloc  # noqa: E402

import competition_score as cs  # noqa: E402
import lifecycle_eval as le  # noqa: E402
from phase2_manifest import load as load_manifest  # noqa: E402

ARMS_DIR = os.path.join(_HERE, "p2_arms")

DEFAULT_ARMS = [
    "k1", "k2p1", "k2p2", "k2p3", "k2p4", "k2p6", "k2p1_f32",
    "sample_iid_max", "sample_iid_floor", "sample_orth", "sample_lattice",
]

#: The arm every other arm is compared against.
#:
#: `p2_test_arms.py` T2 proves `k2p1` reproduces the shipped `estimator.py`
#: `_analytic_only` path **bitwise in its OUTPUT** at this shape. That makes it a
#: sound ACCURACY control.
#:
#: It is NOT a cost-identical control, and this comment used to imply it was.
#: Measured 2026-08-24 (`p2_flop_anchor.py --gate --candidate
#: local_sim/p2_arms/arm_k2p1.py`): 137,600,363,520 against the shipped file's
#: 137,600,326,656 -- **+36,864 FLOPs over three operations** (one extra `astype`,
#: a `concatenate`, a `reshape`) from the arm's own scaffolding. Negligible at
#: 0.000027%, and irrelevant while the multiplier is pinned at the 0.1 floor --
#: but "bitwise identical" and "identical cost" are different claims, and this
#: ladder reports both accuracy and cost.
CONTROL = "k2p1"


def _replica_noise(y1, y2):
    """mean_j ((y1 - y2)/2)^2, per network. The AVERAGED replica's own noise.

    This is the whole cross-product correction, and it needs no predictions:

        mean[(p - y1)(p - y2)] = mean[(p - ybar)^2] - mean[((y1 - y2)/2)^2]

    with `ybar = (y1 + y2)/2` -- expand both sides and the cross terms cancel
    identically. So the noise-corrected MSE is the ordinary MSE against the
    averaged replica minus a constant that depends only on the truth.

    Three things follow, and they are why the driver runs each arm ONCE:
      * the evaluator already computes the first term, so no second pass is
        needed to recover per-MLP predictions;
      * the subtracted constant IS `V / (2N)`, so it is a measured quantity with
        a predicted value to check it against;
      * the bias is IDENTICAL across arms, so it cancels exactly in the paired
        difference. The correction changes no comparison -- it only makes the
        ABSOLUTE MSEs quotable next to a projection.
    """
    return (((y1 - y2) / 2.0) ** 2).mean(axis=1)


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



_BLOCK_RE = re.compile(r"truth_([A-Z]\d{2})")


def _block_from_path(path: str) -> str:
    """The block id a truth filename carries, or "" if it carries none."""
    m = _BLOCK_RE.search(os.path.basename(path))
    return m.group(1) if m else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--n", type=int, default=0, help="0 = every network in the truth file")
    ap.add_argument("--lam", type=float, default=None,
                    help="residual price. Pass 0.0 for the COUNTED-FLOPS-ONLY "
                         "model, which is what the P2-CP1 contract froze as the "
                         "deciding model. Default: the Phase-1 lambda = 1e11, "
                         "reported alongside but NOT the frozen model.")
    ap.add_argument("--depth-subset", type=int, default=8,
                    help="networks used for the MSE-by-depth mechanism curve")
    ap.add_argument("--gate", type=float, default=None,
                    help="pre-registered promotion gate; REQUIRED to print a verdict")
    ap.add_argument("--block", default="",
                    help="the seed block this truth belongs to. Inferred from "
                         "the filename when it carries a block id; REQUIRED "
                         "otherwise, because guessing is how a single-use "
                         "block gets read twice.")
    ap.add_argument("--ts", default="",
                    help="UTC timestamp for the read register. REQUIRED: this "
                         "module must never read the clock, so a replay carries "
                         "the historical date.")
    ap.add_argument("--note", default="", help="recorded on the read line")
    ap.add_argument("--control", default=CONTROL,
                    help="the arm every other arm is paired against")
    ap.add_argument("--residual-cap", type=float, default=None,
                    help="override the 0.4 s cliff for an ACCURACY-ONLY run. "
                         "<=0 disables it entirely. The relaxation is printed "
                         "and recorded; such a run is never residual evidence.")
    a = ap.parse_args()
    if not a.ts:
        print("REFUSED: --ts is required. The read register records WHEN a "
              "protected artifact was looked at, and a module that reads the "
              "clock cannot replay a historical run honestly.")
        return 2

    man = load_manifest()
    width, depth = man["shape"]["width"], man["shape"]["depth"]
    budget = man["budget"]["flop_budget_per_mlp"]
    floor = man["budget"]["multiplier_floor"]
    parity = man["package_parity"]

    # THE TRUTH READ IS REGISTERED, NOT MERELY PERFORMED.
    #
    # Stage 0 R3 (2026-09-10): `seed_ledger` refuses a second CLAIM on a
    # single-use block and has never been able to refuse a second np.load of
    # the truth that claim produced. P08 was claimed once and read three times
    # before anything noticed. `local_sim/truth_reads.py` counts looks; this is
    # the call site that makes the lab honest, and the block is REQUIRED rather
    # than guessed at when the filename does not carry it.
    import truth_reads as _tr
    block = a.block or _block_from_path(a.truth)
    if not block:
        print("REFUSED: cannot tell which seed block "
              f"{os.path.basename(a.truth)!r} belongs to. Pass --block. "
              "Guessing here is how a single-use block gets read twice.")
        return 2
    with _tr.reserve(block, a.truth, consumer="local_sim/p2_lab.py",
                     purpose=f"arms {a.arms} at the live shape",
                     ts=a.ts, instrument=os.path.abspath(__file__),
                     result_artifact=a.out,
                     note=a.note):
        z = dict(np.load(a.truth))
    seeds = [int(s) for s in z["seeds"]]
    n = len(seeds) if a.n == 0 else min(a.n, len(seeds))
    seeds = seeds[:n]
    y1 = z["means_r1"][:n, -1, :].astype(np.float64)
    y2 = z["means_r2"][:n, -1, :].astype(np.float64)
    all1 = z["means_r1"][:n].astype(np.float64)
    all2 = z["means_r2"][:n].astype(np.float64)
    gt_noise = z["gt_noise"][:n].astype(np.float64)
    y_mid = ((y1 + y2) / 2.0).astype(np.float32)

    is_probe = bool(int(z["probe"])) if "probe" in z else False
    if is_probe:
        print("!" * 92)
        print("!!  PROBE TRUTH -- unregistered seeds, no block claimed. This run is a")
        print("!!  PIPELINE SMOKE. Nothing it prints may select an arm, a constant, a")
        print("!!  rank, or a gate, and no number from it belongs in a durable doc.")
        print("!" * 92)

    # The DECIDING model is whatever the contract froze, and it is passed in
    # explicitly rather than defaulted. P2-CP1 froze COUNTED FLOPS ONLY (lam=0):
    # every analytic arm sits at C/B ~0.063, a factor of 1.6 under the 0.1 floor,
    # so no residual model that does not add 0.8 s/MLP can reorder them -- and the
    # sampling arms are separated by counted cost by construction. The local
    # residual is a --runner local figure that does not transfer (2-3% of C on the
    # grader against 10-13% here), so letting it into the deciding number would
    # import this machine's noise into a Phase-2 conclusion.
    # CHANGED UNDER YOU, 2026-08-24 (P2-P0). This call takes LifecycleSpec's
    # DEFAULTS for everything it does not name, and two of those defaults moved:
    #
    #   residual_wall_time_limit_s   None  ->  0.4 s     the cliff is now ARMED
    #   lam                          1e11  ->  0.0       the live Phase-2 rate
    #
    # The lambda change is a no-op for this slice: P2-CP1 froze lam = 0 explicitly
    # and it turns out that is not merely conservative, it is EXACTLY the Phase-2
    # rule -- the round sets lambda to zero, so C == F. The frozen model was right.
    #
    # The cliff is not a no-op. P2-CP1 ran with it DISABLED, so its failure
    # accounting could not see a 0.4 s violation, and re-running this ladder today
    # may fail an arm that passed then. That is the correct behaviour and the
    # difference is a finding, not a regression -- see the P2-P0 correction in
    # experiments/LOG.md before comparing a new run to the recorded one.
    # THE RESIDUAL CLIFF, AND WHY A DESIGN-TIER ACCURACY RUN MAY RELAX IT.
    #
    # `--residual-cap` exists because this host reads ~27x high per operation
    # on residual (docs/VALIDATION.md 433-542, docs/GOTCHAS.md 126), so the
    # armed 0.4 s cliff fails the exact K3 chain here on machine speed rather
    # than on the candidate's behaviour -- measured 726 ms on one network at
    # 1024x16 against a graded 330566 whose worst SERVER wall was 33 s of 120.
    # A failed MLP scores as zero-prediction MSE, so leaving it armed would
    # replace an accuracy comparison with a comparison of nothing.
    #
    # It is NOT disabled silently and it is NOT a policy change. The cap is
    # printed, recorded in the report, and any run that relaxes it is labelled
    # ACCURACY-ONLY. Deployment residual remains `certify_candidate` section 7
    # and the official subprocess runner's job, and no relaxed run may be
    # quoted as residual evidence.
    _res = {} if a.residual_cap is None else {
        "residual_wall_time_limit_s": (None if a.residual_cap <= 0
                                       else a.residual_cap)}
    spec = le.LifecycleSpec(width=width, depth=depth, flop_budget=budget,
                            floor=floor, **_res,
                            **({} if a.lam is None else {"lam": a.lam}))
    if a.residual_cap is not None:
        print("  *** RESIDUAL CAP RELAXED -- this run is ACCURACY-ONLY. Its "
              "failure accounting")
        print("      cannot see a 0.4 s violation and it is NOT residual "
              "evidence for anything.")
    print(f"  deciding model   lambda = {spec.lam:.4g}"
          + ("   (COUNTED FLOPS ONLY -- the frozen model)" if spec.lam == 0.0
             else "   (Phase-1 lambda; NOT the frozen P2-CP1 model)"))
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

    # The averaged replica's noise, measured once. Every arm's MSE against `y_mid`
    # is inflated by exactly this, identically, so it cancels in every paired
    # comparison and is subtracted only for the ABSOLUTE numbers.
    rep_noise = _replica_noise(y1, y2)
    print(f"  replica noise of the AVERAGED truth   measured {rep_noise.mean():.4e}"
          f"   predicted V/2N {gt_noise.mean() / 2:.4e}"
          f"   ratio {rep_noise.mean() / (gt_noise.mean() / 2):.4f}")
    print("     (the ratio is a live check on replica independence; far from 1.0")
    print("      means the two draws share state and every corrected MSE is wrong)")
    print()

    n_depth = min(a.depth_subset, n)
    results = {}
    for name in arms:
        t0 = time.perf_counter()
        # ONE scored pass. `lifecycle_eval` owns every scored number.
        rows, prov = _run_arm(name, seeds, y_mid, spec, list(gt_noise))
        suite = le.score(rows, spec=spec)
        raw = np.array([r.mse for r in rows])
        cross = raw - rep_noise
        # The depth curve needs full prediction arrays, which the evaluator does
        # not return. A mechanism curve does not need 96 networks, so it is
        # computed on a fixed SUBSET and labelled as such rather than pretending
        # otherwise.
        tracemalloc.start()
        _f, alld = _predict_all(name, seeds[:n_depth], spec)
        _cur, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        by_depth = np.array([
            [float(((alld[i][d] - (all1[i][d] + all2[i][d]) / 2.0) ** 2).mean())
             for d in range(depth)] for i in range(n_depth)])
        c012 = _cost_under_012(name, width, depth, budget)
        results[name] = dict(
            rows=rows, prov=prov, suite=suite, cross=cross, raw=raw,
            by_depth=by_depth, cost012=c012, secs=time.perf_counter() - t0,
            peak_mb=peak_bytes / 2 ** 20, n_depth=n_depth,
        )
        f = float(np.mean([r.flops_used for r in rows]))
        eff = float(np.mean([cs.effective_compute(r.flops_used,
                                                  r.residual_wall_time_s, spec.lam)
                             for r in rows]))
        print(f"  {name:<18} score {suite.objective:.5e}   "
              f"MSE(corr) {cross.mean():.4e}   MSE(raw) {raw.mean():.4e}   "
              f"F/B {f / budget:.4f}   eff {eff / budget:.4f}   "
              f"mult {suite.mean_multiplier:.4f}   fail {suite.n_failed}/{n}   "
              f"resid {1000 * float(np.mean([r.residual_wall_time_s for r in rows])):.0f}ms   "
              f"{results[name]['secs']:.0f}s"
              + ("" if c012 is None else
                 f"   [0.12: {c012 / budget:.6f}"
                 + ("" if abs(c012 - f) < 1 else f"  MOVED {c012 / f:.6f}x") + "]"))

    # ---- paired comparisons against the control ---------------------------
    print()
    print(f"  PAIRED vs {a.control} -- arithmetic score difference, "
          f"studentised-t 95% (the binding statistic, docs/VALIDATION.md 2b-i)")
    print(f"  {'arm':<18} {'gain':>8} {'95% t CI (relative)':>26} {'better':>9} "
          f"{'sign p':>8}")
    ctrl_rows = results[a.control]["rows"]
    paired = {}
    for name in arms:
        if name == a.control:
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

    # ---- does any arm leave the floor once residual is charged? --------------
    # The frozen deciding model is counted-FLOPs-only, and that is defensible for
    # RANKING here. It is NOT a licence to ignore residual: an arm can be pinned
    # at the floor on counted cost and off it on effective cost, and the two are
    # priced by different mechanisms. flopscope charges the RELEASE of a large
    # temporary as residual wall time (~100-120 us/op, forum 18101) while pricing
    # the operation itself at O(n^2) FLOPs -- so an elementwise-heavy arm is cheap
    # in the frozen model and expensive in reality. That arbitrage is exactly what
    # killed the packed lane. Report it loudly rather than letting the freeze hide
    # it, because the next lane (K3/Edgeworth) is elementwise-heavy in the same way.
    print()
    print("  FLOOR CHECK -- counted vs effective, at the PHASE-1 lambda = 1e11")
    print("  (local residual does NOT transfer; the RATIO to the control is the")
    print("   signal, not the level)")
    ctl_r = float(np.mean([r.residual_wall_time_s for r in results[a.control]["rows"]]))
    for name in arms:
        rws = results[name]["rows"]
        f = float(np.mean([r.flops_used for r in rws]))
        rr = float(np.mean([r.residual_wall_time_s for r in rws]))
        eff = [(r.flops_used + 1e11 * r.residual_wall_time_s) / budget for r in rws]
        off = f / budget < floor <= max(eff)
        print(f"    {name:<18} F/B {f / budget:.4f}   eff mean {np.mean(eff):.4f}"
              f"   eff max {max(eff):.4f}   resid {1000 * rr:>6.0f} ms"
              f"  ({rr / ctl_r if ctl_r else float('nan'):>5.2f}x control)"
              + ("   *** OFF THE FLOOR once residual is charged ***" if off else ""))

    # ---- THE GATE -----------------------------------------------------------
    # The first build of this driver took a `--gate` argument and never used it,
    # leaving the verdict to whoever read the table above. That is precisely the
    # failure `competition_score.decide()` exists to prevent: it is the ONLY
    # function in this repository that returns a promotion boolean, it REQUIRES a
    # pre-registered gate (no default, so a gate supplied after the data is
    # visible shows up in the diff), it is ONE-SIDED, and it refuses percentile
    # and BCa outright.
    verdicts = {}
    if a.gate is None:
        print()
        print("  NO GATE SUPPLIED -- no verdict is printed. The numbers above are")
        print("  descriptive. Pass --gate <the pre-registered threshold> to get a")
        print("  decision, and only the threshold registered BEFORE the data.")
    else:
        print()
        print(f"  VERDICT vs {a.control}, pre-registered gate {a.gate:.4f}x, "
              f"studentised-t, one-sided")
        ctrl_scores = cs.score_suite([r.to_row() for r in ctrl_rows],
                                     lam=spec.lam, floor=floor).scores
        for name in arms:
            if name == a.control:
                continue
            cand_scores = cs.score_suite(
                [r.to_row() for r in results[name]["rows"]],
                lam=spec.lam, floor=floor).scores
            v = cs.decide(ctrl_scores, cand_scores, gate=a.gate, interval="t")
            verdicts[name] = v
            print(f"  {name:<18} {'PASS' if v.passed else 'fail':<5} "
                  f"gain {v.gain:>7.4f}x   "
                  f"t CI on the difference [{v.ci[0]:>+.3e}, {v.ci[1]:>+.3e}]"
                  f"   excl 0: {str(v.ci_excludes_zero):<5}"
                  f"   {v.n_candidate_better}/{v.n}"
                  f"   mover {v.biggest_mover_share * 100:>6.1f}%"
                  + ("   *** ESTIMANDS DISAGREE ***" if v.estimands_disagree else ""))
            for why in v.reasons:
                print(f"       - {why}")

    # ---- the mechanism curve ----------------------------------------------
    print()
    print(f"  MSE BY DEPTH -- the (L/n)^K mechanism curve, on {n_depth} of {n} "
          f"networks (a mechanism curve does not need all of them)")
    print("  L1 is exact for EVERY arm, so that column reads the truth-noise "
          "floor -- a built-in calibration")
    hdr = "  " + f"{'arm':<18}" + "".join(f"{'L' + str(d + 1):>11}"
                                          for d in (0, 1, 3, 7, 11, 15))
    print(hdr)
    for name in arms:
        bd = results[name]["by_depth"].mean(axis=0)
        print("  " + f"{name:<18}" + "".join(f"{bd[d]:>11.3e}"
                                             for d in (0, 1, 3, 7, 11, 15)))

    if a.out:
        _write(a.out, results, paired, man, seeds, z, spec, arms, verdicts, a.gate)
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


def _write(path, results, paired, man, seeds, z, spec, arms,
           verdicts=None, gate=None):
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
            "replica_noise_of_average_measured": float(_replica_noise(
                z["means_r1"][:len(seeds), -1, :].astype(np.float64),
                z["means_r2"][:len(seeds), -1, :].astype(np.float64)).mean()),
            "n_samples_per_replica": int(z["n_samples"]),
            "n_networks": len(seeds),
            "replicas": 2,
            "gt_noise_mean": float(z["gt_noise"].mean()),
        },
        "seeds": [int(s) for s in seeds],
        "lambda_used": spec.lam,
        "deciding_model": ("counted FLOPs only (lam=0), as frozen in the P2-CP1 "
                           "contract" if spec.lam == 0.0 else
                           "effective F + lam*R at the PHASE-1 lambda; this is "
                           "NOT the model the contract froze"),
        "lambda_provenance": "PHASE-1 value 1e11; Phase-2 lambda is UNPUBLISHED",
        "residual_is_graded": False,
        "control": a.control,
        "residual_cap_override": a.residual_cap,
        "accuracy_only": a.residual_cap is not None,
        "pre_registered_gate": gate,
        "gate_note": (
            "the gate is registered in experiments/LOG.md BEFORE any data "
            "existed; decide() requires it explicitly and has no default"
        ),
        "noise_correction_identity": (
            "mean[(p-y1)(p-y2)] = mean[(p-ybar)^2] - mean[((y1-y2)/2)^2]; the "
            "subtracted term is the averaged replica's own noise and is identical "
            "across arms, so it cancels exactly in every paired difference"
        ),
        "arms": {},
    }
    for name in arms:
        r = results[name]
        payload["arms"][name] = {
            "objective": r["suite"].objective,
            "mean_mse_raw_vs_averaged_replica": float(r["raw"].mean()),
            "mean_mse_noise_corrected": float(r["cross"].mean()),
            "per_mlp_mse_raw": [float(x) for x in r["raw"]],
            "per_mlp_mse_noise_corrected": [float(x) for x in r["cross"]],
            "mean_flops": float(np.mean([x.flops_used for x in r["rows"]])),
            "flops_012": r["cost012"],
            "n_failed": r["suite"].n_failed,
            "mean_multiplier": r["suite"].mean_multiplier,
            "mse_by_depth": [float(x) for x in r["by_depth"].mean(axis=0)],
            "mse_by_depth_n_networks": int(r["n_depth"]),
            "peak_python_heap_mb": r["peak_mb"],
            "peak_memory_note": ("tracemalloc counts PYTHON allocations only; the "
                                 "grader's memory_limit_mb is RLIMIT_AS over the "
                                 "whole address space, so this is a LOWER BOUND"),
            "mean_residual_wall_time_s": float(np.mean(
                [x.residual_wall_time_s for x in r["rows"]])),
            "max_residual_wall_time_s": float(np.max(
                [x.residual_wall_time_s for x in r["rows"]])),
            "rows_hash": le.rows_hash(r["rows"]),
            "setup_seconds": r["prov"]["setup_seconds"],
            "candidate_sha256": r["prov"]["candidate_sha256"],
            # Per-MLP scores, so the verdict is reproducible from this file alone
            # without re-running a single arm.
            "per_mlp_score": [float(x) for x in cs.score_suite(
                [x.to_row() for x in r["rows"]],
                lam=spec.lam, floor=man["budget"]["multiplier_floor"]).scores],
            "per_mlp_flops": [float(x.flops_used) for x in r["rows"]],
            "per_mlp_residual_s": [float(x.residual_wall_time_s) for x in r["rows"]],
            "per_mlp_failed": [bool(x.failed) for x in r["rows"]],
        }
        if verdicts and name in verdicts:
            v = verdicts[name]
            payload["arms"][name]["verdict"] = {
                "passed": bool(v.passed),
                "gain": v.gain,
                "gate": v.gate,
                "ci_on_paired_difference": list(v.ci),
                "ci_excludes_zero": bool(v.ci_excludes_zero),
                "interval": v.interval,
                "n": v.n,
                "n_candidate_better": v.n_candidate_better,
                "sign_test_p": v.sign_test_p,
                "geometric_gain_DIAGNOSTIC": v.geometric_gain,
                "estimands_disagree_DIAGNOSTIC": bool(v.estimands_disagree),
                "biggest_mover_share_DIAGNOSTIC": v.biggest_mover_share,
                "reasons": list(v.reasons),
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
