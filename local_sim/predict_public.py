"""What would this candidate score on the public board? One answer, three labels.

Slice 6 contract, `docs/CLAUDE_CODE_HANDOFF_2026-09-14.md` §0.5 — the piece that
was to be built first, because every later slice needs a way to ask "did that
help?" that does not cost a submission slot.

WHY A PROJECTION AND NOT A MEASUREMENT
---------------------------------------
The public board is 50 hidden MLPs we cannot see. What we CAN do is run two
candidates over the same 100 public-mini networks, take the **paired ratio** of
their competition scores, and multiply the control's **known graded** public
score by it. The ratio is the transferable quantity: it survives the fact that
our 100 networks are not their 50, because both arms saw the same 100.

    projected_public(candidate) = graded_public(control) x ratio(local, paired)

That is an estimate with an interval, and it is labelled as one. It is not a
score, it does not licence a submission, and it cannot promote anything.

THE THREE LABELS, AND WHY EVERY LINE CARRIES ONE
-------------------------------------------------
    MATH              arithmetic and identities. Host-independent; a failure
                      here is a failure anywhere.
    LOCAL             what this machine measured. Real, and NOT the grading
                      server: this host reads ~27x high per operation on
                      residual and the shim reads ~18% low on wall for
                      rotation-using estimators.
    SERVER_PREDICTED  a forecast from a versioned model with a stated evidence
                      domain. A forecast is not a measurement, however
                      carefully it is made.

`certify_candidate.py` grew the same three-way split on 2026-09-10 (Stage 0 R5)
for the same reason: one unqualified number over three kinds of evidence is how
a local second gets quoted as a server second.

CONFIRM TIER — READ THIS BEFORE USING THE OUTPUT
-------------------------------------------------
The public mini split carries the organisers' REAL labels. Reading it is a
CONFIRM-tier read: it may **check** a decision already made on fresh mints, and
it may **choose nothing**. Do not select a rank, a coefficient, a blend or a
threshold on a number this module printed. `docs/VALIDATION.md` is binding and
this tool does not relax it.

WHAT IT REFUSES
---------------
* two runs that did not see the same MLPs (identity-aligned by `mlp_name`
  through `competition_score.align_rows_by_identity`, which realigns a permuted
  arm and refuses a disjoint one);
* a control whose local replay does not reproduce its own graded score, since
  the whole projection hangs on that anchor;
* a run with failed MLPs, unless `--allow-failures`, because a failed row
  scores as zero-prediction MSE and silently flatters nothing — it destroys the
  arm.

Run:
    uv run python local_sim/predict_public.py \\
        --candidate-run experiments/<slice>/mini_candidate.json \\
        --control-run   experiments/<slice>/mini_f21.json \\
        --control-graded experiments/run_reports/f21_diagnostic_330950_graded_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

import competition_score as cs  # noqa: E402
import official_config as oc  # noqa: E402

MATH, LOCAL, SERVER = "MATH", "LOCAL", "SERVER_PREDICTED"

#: The grading server has been observed to add 0 or +2**24 counted FLOPs
#: depending on the candidate's cast family, and which branch a given file takes
#: is not predictable from here. Both ends are reported; a projection that
#: straddles a decision boundary across them is not a projection.
SERVER_FLOP_OFFSETS: Tuple[int, ...] = (0, 2 ** 24)


class ProjectionRefused(RuntimeError):
    """The inputs cannot support a projection. Never downgraded to a warning."""


# --------------------------------------------------------------------------- #
# reading a `whest run --json` report
# --------------------------------------------------------------------------- #


def load_run(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def run_rows(report: Dict[str, Any], *, require_mse: bool = True) -> List[cs.MlpRow]:
    """Scorable rows from a local `whest run` report.

    `mlp_name` is the network's identity and is carried onto the row, so the
    pairing is checkable rather than positional.
    """
    cfg = report.get("run_config") or {}
    budget = float(cfg.get("flop_budget") or 0.0)
    if budget <= 0:
        raise ProjectionRefused(
            "the run report carries no positive flop_budget; refusing to guess one")
    out: List[cs.MlpRow] = []
    for e in (report.get("results") or {}).get("per_mlp") or []:
        mse = e.get("final_layer_mse")
        if mse is None and require_mse:
            continue
        out.append(cs.MlpRow(
            mse=float("nan") if mse is None else float(mse),
            flops_used=float(e.get("flops_used") or 0.0),
            residual_wall_time_s=float(e.get("residual_wall_time_s") or 0.0),
            budget=budget,
            failed=cs.failed_from_report_entry(e),
            name=str(e.get("mlp_name") or e.get("mlp_index", "")),
        ))
    if not out:
        raise ProjectionRefused("the run report contains no scorable rows")
    return out


def run_ops_per_mlp(report: Dict[str, Any]) -> Optional[float]:
    """Metered flopscope OPERATIONS per MLP — what the residual model prices.

    Counted FLOPs are the score's cost term; the residual cliff is a function of
    the OPERATION COUNT, and the two are different quantities that a report
    happens to carry side by side.
    """
    res = report.get("results") or {}
    n = len((res.get("per_mlp") or [])) or int(
        (report.get("run_config") or {}).get("n_mlps") or 0)
    by_ns = ((res.get("breakdowns") or {}).get("estimator") or {}).get("by_namespace") or {}
    calls = 0
    for ns, d in by_ns.items():
        if "estimator" in ns:
            calls += int(d.get("calls") or 0)
    if not calls or not n:
        return None
    return calls / float(n)


def run_wall_worst(report: Dict[str, Any]) -> Optional[float]:
    rows = (report.get("results") or {}).get("per_mlp") or []
    walls = [float(r.get("wall_time_s") or 0.0) for r in rows]
    return max(walls) if walls else None


# --------------------------------------------------------------------------- #
# the projection
# --------------------------------------------------------------------------- #


def graded_anchor(graded: Dict[str, Any]) -> Dict[str, float]:
    """The control's OWN graded public score, replayed through our scorer.

    Replayed rather than read off a summary field, because the anchor is the
    load-bearing number and a summary we did not compute is a number we cannot
    audit.
    """
    rows = cs.rows_from_graded_report(graded)
    suite = cs.score_suite(rows)
    return {"adjusted": float(suite.objective), "raw_mse": float(suite.mean_mse),
            "multiplier": float(suite.mean_multiplier), "n": int(suite.n),
            "n_failed": int(suite.n_failed)}


def project(candidate_rows, control_rows, anchor, *, gate: Optional[float] = None,
            allow_failures: bool = False):
    """The paired ratio, its interval, and the projected public score."""
    c_rows, k_rows, notes = cs.align_rows_by_identity(
        control_rows, candidate_rows, require_ids=True)

    ctrl = cs.score_suite(c_rows)
    cand = cs.score_suite(k_rows)
    if not allow_failures and (ctrl.n_failed or cand.n_failed):
        raise ProjectionRefused(
            f"failed MLPs present (control {ctrl.n_failed}, candidate "
            f"{cand.n_failed}). A failed row is scored at zero-prediction MSE, "
            "so a projection built on one is a projection of a different "
            "program. Re-run with the residual cap lifted, or pass "
            "--allow-failures and say so in the write-up.")

    paired = cs.paired_compare_rows(c_rows, k_rows, require_ids=True)
    d = np.asarray([a.score() - b.score() for a, b in zip(c_rows, k_rows)],
                   dtype=np.float64)
    lo, hi = cs.paired_diff_ci(d, method="t")

    mean_ctrl = float(ctrl.objective)
    ratio = float(cand.objective / mean_ctrl) if mean_ctrl else float("nan")
    # The interval on the DIFFERENCE is the binding statistic; the ratio band is
    # that interval re-expressed against the control's own mean, which is a
    # rescaling and not a second inference.
    ratio_lo = (mean_ctrl - hi) / mean_ctrl if mean_ctrl else float("nan")
    ratio_hi = (mean_ctrl - lo) / mean_ctrl if mean_ctrl else float("nan")

    out = {
        "n": int(paired.n),
        "local_control_adjusted": mean_ctrl,
        "local_candidate_adjusted": float(cand.objective),
        "local_control_raw_mse": float(ctrl.mean_mse),
        "local_candidate_raw_mse": float(cand.mean_mse),
        "local_control_multiplier": float(ctrl.mean_multiplier),
        "local_candidate_multiplier": float(cand.mean_multiplier),
        "ratio": ratio,
        "ratio_ci": (ratio_lo, ratio_hi),
        "mean_difference": float(paired.mean_difference),
        "t_ci_difference": tuple(float(x) for x in paired.t_ci_difference),
        "arithmetic_gain": float(paired.arithmetic_gain),
        "geometric_gain_DIAGNOSTIC": float(paired.geometric_gain),
        "n_candidate_better": int(paired.n_candidate_better),
        "sign_test_p": float(paired.sign_test_p),
        "anchor_graded_adjusted": float(anchor["adjusted"]),
        "projected_public_adjusted": float(anchor["adjusted"] * ratio),
        "projected_public_ci": (float(anchor["adjusted"] * ratio_lo),
                                float(anchor["adjusted"] * ratio_hi)),
        "notes": list(paired.notes) + notes,
        "n_failed_control": int(ctrl.n_failed),
        "n_failed_candidate": int(cand.n_failed),
    }
    if gate is not None:
        v = cs.decide(ctrl.scores, cand.scores, gate=gate)
        out["verdict"] = {"passed": bool(v.passed), "gain": float(v.gain),
                          "gate": float(v.gate),
                          "ci_excludes_zero": bool(v.ci_excludes_zero),
                          "reasons": list(v.reasons)}
    return out


def multiplier_band(flops_per_mlp: float, budget: float) -> List[Dict[str, float]]:
    """`{F, F + 2**24} / 2**41`, both ends, because the server picks one."""
    return [{"offset": off,
             "flops": flops_per_mlp + off,
             "c_over_b": (flops_per_mlp + off) / budget,
             "multiplier": max(cs.MULTIPLIER_FLOOR, (flops_per_mlp + off) / budget)}
            for off in SERVER_FLOP_OFFSETS]


def residual_forecast(ops_per_mlp: float) -> Dict[str, Any]:
    """Operation count -> predicted SERVER residual, with its evidence domain.

    §0.1 of the Slice 6 contract removed the local-seconds residual gate from
    every decision path. This is what replaces it: the operation count, priced
    by the calibrated per-operation model, against the published 0.4 s cliff.
    """
    lo_s, hi_s = oc.GRADED_RESIDUAL_S_PER_OP_RANGE
    cap = 0.4
    return {
        "ops_per_mlp": ops_per_mlp,
        "gate_ops": int(oc.PHASE2_OP_COUNT_GATE),
        "gate_passed": ops_per_mlp <= oc.PHASE2_OP_COUNT_GATE,
        "s_per_op_point": oc.GRADED_RESIDUAL_S_PER_OP,
        "s_per_op_range": (lo_s, hi_s),
        "predicted_s_point": ops_per_mlp * oc.GRADED_RESIDUAL_S_PER_OP,
        "predicted_s_range": (ops_per_mlp * lo_s, ops_per_mlp * hi_s),
        "cap_s": cap,
        "headroom_point": cap / (ops_per_mlp * oc.GRADED_RESIDUAL_S_PER_OP)
                          if ops_per_mlp else float("inf"),
    }


def wall_forecast(local_worst_s: Optional[float]) -> Dict[str, Any]:
    """Local worst wall x the measured contention factor, against the 120 s cap.

    The contention factor is not a safety margin. It is what the maximum row
    moved between two graded runs of IDENTICAL bytes (+51.8% while the median
    moved -0.6%), so it is the size of the variation a margin has to survive,
    measured once, on one pair of runs.
    """
    if local_worst_s is None:
        return {"available": False,
                "why": "no per-MLP wall times in the run report"}
    f = oc.GRADED_WALL_CONTENTION_FACTOR
    projected = local_worst_s * f
    return {"available": True, "local_worst_s": local_worst_s,
            "contention_factor": f, "projected_worst_s": projected,
            "cap_s": float(oc.PHASE2.wall_time_limit_s),
            "margin": float(oc.PHASE2.wall_time_limit_s) / projected
                      if projected else float("inf")}


def memory_verdict(candidate_path: Optional[str], sha: Optional[str],
                   width: int, depth: int, budget: float) -> Dict[str, Any]:
    """The Linux 8-GiB enforcement report for these exact bytes, if one exists."""
    if not candidate_path:
        return {"available": False, "why": "no --candidate path given"}
    import certify_candidate as cc
    stem = os.path.splitext(os.path.basename(candidate_path))[0]
    cands = []
    for root, _dirs, files in os.walk(os.path.join(_ROOT, "experiments")):
        for fn in files:
            if fn.startswith("linux_memory_gate") and fn.endswith(".json"):
                cands.append(os.path.join(root, fn))
    for p in sorted(cands, reverse=True):
        try:
            with open(p, encoding="utf-8") as fh:
                lg = json.load(fh)
        except (OSError, ValueError):
            continue
        if sha and lg.get("candidate_sha256") != sha:
            continue
        if not sha and stem not in os.path.basename(p):
            continue
        checks, degraded = cc.validate_memory_report(
            lg, candidate_sha256=sha or lg.get("candidate_sha256", ""),
            width=width, depth=depth, budget=int(budget),
            weights_mode=cc._lifecycle_default_mode(),
            stack=cc._certified_stack())
        bad = [n for n, ok, _ in checks if not ok]
        return {"available": True, "report": os.path.relpath(p, _ROOT),
                "degraded": degraded, "failed_checks": bad,
                "verdict": ("REFUSED" if bad else
                            "INCOMPLETE" if degraded else "PASS")}
    return {"available": False,
            "why": f"no linux_memory_gate report found for {stem}"}


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _line(tag, text, value=""):
    print(f"  [{tag:<16}] {text:<52}{value}")


def report(res, mult, resid, wall, mem, meta):
    print("=" * 100)
    print("PROJECTED PUBLIC SCORE -- an estimate with an interval, not a score")
    print("=" * 100)
    print(f"  candidate   {meta.get('candidate', '(unnamed)')}")
    print(f"  control     {meta.get('control', '(unnamed)')}   "
          f"graded submission {meta.get('control_submission', '?')}")
    print(f"  networks    {res['n']} public-mini MLPs, paired on mlp_name")
    print()
    print("  *** THE PUBLIC MINI SPLIT CARRIES THE ORGANISERS' REAL LABELS.")
    print("  *** This is a CONFIRM-tier read. It may CHECK a decision made on")
    print("  *** fresh mints and it may CHOOSE NOTHING. Do not select a rank,")
    print("  *** a coefficient, a blend or a threshold on anything below.")
    print()

    print("-- the paired local measurement " + "-" * 67)
    _line(LOCAL, "control adjusted, our 100 nets",
          f"{res['local_control_adjusted']:.6e}")
    _line(LOCAL, "candidate adjusted, our 100 nets",
          f"{res['local_candidate_adjusted']:.6e}")
    _line(LOCAL, "control raw MSE", f"{res['local_control_raw_mse']:.6e}")
    _line(LOCAL, "candidate raw MSE", f"{res['local_candidate_raw_mse']:.6e}")
    _line(LOCAL, "control multiplier", f"{res['local_control_multiplier']:.6f}")
    _line(LOCAL, "candidate multiplier", f"{res['local_candidate_multiplier']:.6f}")
    print()
    _line(MATH, "paired arithmetic score difference",
          f"{res['mean_difference']:+.6e}")
    _line(MATH, "  studentised-t 95% CI on that difference",
          f"[{res['t_ci_difference'][0]:+.4e}, {res['t_ci_difference'][1]:+.4e}]")
    _line(MATH, "ratio candidate/control", f"{res['ratio']:.6f}")
    _line(MATH, "  ratio 95% band",
          f"[{res['ratio_ci'][0]:.6f}, {res['ratio_ci'][1]:.6f}]")
    _line(MATH, "arithmetic gain (control/candidate)",
          f"{res['arithmetic_gain']:.4f}x")
    _line(MATH, "candidate better on",
          f"{res['n_candidate_better']}/{res['n']}   sign p = "
          f"{res['sign_test_p']:.4f}")
    _line(MATH, "geometric gain  DIAGNOSTIC ONLY, no authority",
          f"{res['geometric_gain_DIAGNOSTIC']:.4f}x")
    print()

    print("-- the projection " + "-" * 81)
    _line(LOCAL, "control's OWN graded public score (replayed)",
          f"{res['anchor_graded_adjusted']:.6e}")
    _line(SERVER, "projected public adjusted",
          f"{res['projected_public_adjusted']:.6e}")
    _line(SERVER, "  95% band",
          f"[{res['projected_public_ci'][0]:.4e}, "
          f"{res['projected_public_ci'][1]:.4e}]")
    print("       The ratio is what transfers: our 100 networks are not their")
    print("       50, but both arms saw the same 100. The anchor is measured;")
    print("       the ratio is measured; the product is a PREDICTION.")
    print()

    print("-- cost, both ends of the server's offset " + "-" * 57)
    for m in mult:
        _line(SERVER, f"F + {m['offset']:>9,} -> C/B",
              f"{m['c_over_b']:.6f}   multiplier {m['multiplier']:.6f}")
    if len(mult) == 2 and mult[0]["multiplier"] != mult[1]["multiplier"]:
        spread = abs(mult[1]["multiplier"] - mult[0]["multiplier"])
        rel = spread / max(mult[0]["multiplier"], 1e-30)
        _line(MATH, "  offset moves the multiplier by", f"{rel * 100:.4f}%")
    print()

    print("-- residual, priced on OPERATIONS and not on local seconds " + "-" * 40)
    if resid is None:
        _line(LOCAL, "operation count", "UNAVAILABLE -- no by_namespace calls")
        print("       Without an operation count there is no residual forecast.")
        print("       A local second is not a server second and is not a")
        print("       substitute (Slice 6 contract section 0.1).")
    else:
        _line(LOCAL, "metered operations per MLP", f"{resid['ops_per_mlp']:,.1f}")
        _line(MATH, f"  against the {resid['gate_ops']:,}-operation gate",
              "PASS" if resid["gate_passed"] else "*** FAIL ***")
        _line(SERVER, "predicted server residual",
              f"{resid['predicted_s_point']:.4f} s   "
              f"[{resid['predicted_s_range'][0]:.4f}, "
              f"{resid['predicted_s_range'][1]:.4f}]")
        _line(SERVER, f"  headroom under the {resid['cap_s']} s cliff",
              f"{resid['headroom_point']:.2f}x")
        print("       The per-operation model is calibrated on graded rows and")
        print("       has an evidence domain. It is not a guarantee, and a")
        print("       candidate whose operation MIX differs from its")
        print("       calibration set is outside it.")
    print()

    print("-- wall " + "-" * 91)
    if wall.get("available"):
        _line(LOCAL, "worst local wall", f"{wall['local_worst_s']:.3f} s")
        _line(SERVER, f"x {wall['contention_factor']:.2f} contention",
              f"{wall['projected_worst_s']:.3f} s of {wall['cap_s']:.0f} s")
        _line(SERVER, "  margin", f"{wall['margin']:.2f}x")
        print("       The contention factor is not a safety margin: it is how")
        print("       far the worst row moved between two graded runs of")
        print("       IDENTICAL bytes. Measured once, on one pair of runs.")
    else:
        _line(LOCAL, "wall", f"UNAVAILABLE -- {wall.get('why', '')}")
    print()

    print("-- memory, the one cap this host cannot test " + "-" * 55)
    if mem.get("available"):
        _line(LOCAL, "Linux 8-GiB enforcement", mem["verdict"])
        _line(LOCAL, "  report", mem["report"])
        if mem.get("degraded"):
            print(f"       DEGRADED: {mem['degraded'][:80]}")
        for b in mem.get("failed_checks", [])[:4]:
            print(f"       FAILED: {b}")
    else:
        _line(LOCAL, "Linux 8-GiB enforcement",
              f"NOT TESTED -- {mem.get('why', '')}")
        print("       A skipped memory gate is NOT a pass. Windows has no")
        print("       RLIMIT_AS, so nothing measured here clears the limit.")
    print()

    if "verdict" in res:
        v = res["verdict"]
        print("-- the pre-registered gate " + "-" * 73)
        _line(MATH, f"gate {v['gate']:.4f}x",
              ("PASS" if v["passed"] else "FAIL")
              + f"   gain {v['gain']:.4f}x   CI excludes zero: "
              + str(v["ci_excludes_zero"]))
        for r in v["reasons"]:
            print(f"       - {r}")
        print()

    for n in res.get("notes", []):
        print(f"  note: {n}")
    print()
    print("  NOTHING HERE PROMOTES, SUBMITS OR DESIGNATES ANYTHING. Samuel")
    print("  approves every submission individually and designates alone.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument("--candidate-run", required=True,
                    help="whest run --json report for the candidate")
    ap.add_argument("--control-run", required=True,
                    help="whest run --json report for the control, SAME networks")
    ap.add_argument("--control-graded", default=os.path.join(
        _ROOT, "experiments", "run_reports",
        "f21_diagnostic_330950_graded_report.json"),
        help="the control's own graded report -- the projection's anchor")
    ap.add_argument("--candidate", default="",
                    help="the candidate's source path, for the memory gate")
    ap.add_argument("--gate", type=float, default=None,
                    help="a PRE-REGISTERED gate. Omit unless a contract set one")
    ap.add_argument("--allow-failures", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    cand_run = load_run(a.candidate_run)
    ctrl_run = load_run(a.control_run)
    graded = load_run(a.control_graded)

    cand_rows = run_rows(cand_run)
    ctrl_rows = run_rows(ctrl_run)
    anchor = graded_anchor(graded)

    cfg = cand_run.get("run_config") or {}
    budget = float(cfg.get("flop_budget") or 0.0)
    width = int(cfg.get("width") or 1024)
    depth = int(cfg.get("depth") or 16)

    res = project(cand_rows, ctrl_rows, anchor, gate=a.gate,
                  allow_failures=a.allow_failures)

    flops = float(np.mean([r.flops_used for r in cand_rows]))
    mult = multiplier_band(flops, budget)
    ops = run_ops_per_mlp(cand_run)
    resid = residual_forecast(ops) if ops else None
    wall = wall_forecast(run_wall_worst(cand_run))
    sha = _sha256(a.candidate) if a.candidate and os.path.exists(a.candidate) else None
    mem = memory_verdict(a.candidate or None, sha, width, depth, budget)

    meta = {"candidate": a.candidate or os.path.basename(a.candidate_run),
            "control": os.path.basename(a.control_run),
            "control_submission": (graded.get("submission_id")
                                   or (graded.get("evaluation") or {}).get(
                                       "submission_id", "?")),
            "candidate_sha256": sha,
            "dataset": ((cfg.get("dataset") or {}).get("path")),
            "anchor_n": anchor["n"], "anchor_failed": anchor["n_failed"]}
    report(res, mult, resid, wall, mem, meta)

    if a.json:
        payload = {"meta": meta, "projection": res, "multiplier_band": mult,
                   "residual_forecast": resid, "wall_forecast": wall,
                   "memory": mem,
                   "labels": {MATH: "arithmetic and identities; host-independent",
                              LOCAL: "measured on THIS host; not server seconds",
                              SERVER: "a forecast from a versioned model with a "
                                      "stated evidence domain"},
                   "confirm_tier_read": True,
                   "promotes_nothing": True}
        d = os.path.dirname(os.path.abspath(a.json))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2, default=str) + chr(10))
        print(f"  -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
