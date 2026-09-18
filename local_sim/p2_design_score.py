"""Score one candidate against one control on DESIGN truth. The cadence tool.

Slice 6 contract §0.4: *"every candidate gets a 24-net fresh-truth score the day
it compiles."* This is that. It is deliberately small and general — two file
paths, one truth file, a paired arithmetic score difference — because the thing
that has been missing since September 7 is not an instrument, it is a habit.

WHY NOT `p2_lab.py`
--------------------
`p2_lab.py` was locked on 2026-09-14 to a specific four-arm preregistered
mechanism experiment: it demands a freeze receipt naming exactly
`("exact", "coherent", "consumers-only", "births-only")` and a `--control` of
`"exact"`. Those locks are correct for that experiment and this is a different
one, so this module exists beside it rather than loosening it. **A lock you
route around is not a lock.**

WHAT IT IS AND IS NOT
----------------------
DESIGN tier. It reads a reusable design truth file through the read register,
so the look is counted. It produces the paired arithmetic score difference with
a studentised-t interval — the binding statistic — and, when the truth file
carries `gt_noise`, the noise-corrected true-function difference and the
official-precision forecast beside it, because label noise does **not** cancel
from an adjusted difference when the two arms have different multipliers.

It **decides nothing**. `--gate` is accepted only so a preregistered contract
can have its verdict printed; with no gate it prints the statistics and stops.
A design-tier pass licenses a DECIDE-block contract, never a promotion.

Run:
    uv run python local_sim/p2_design_score.py \\
        --control estimator_k3_strassen_L4_f21.py \\
        --candidate estimator_k3_v30.py \\
        --truth experiments/2026-08-21_p2_cp1/truth_P01.npz \\
        --n 24 --ts <utc> --out experiments/<slice>/score.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import dataclasses

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402


def _pkg_ver(name: str) -> str:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--control", required=True)
    ap.add_argument("--candidate", required=True,
                    help="one path, or several comma-separated. The control is "
                         "run ONCE and every candidate is paired against it, "
                         "which is what makes a parameter sweep affordable: "
                         "k candidates cost k+1 arms, not 2k.")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--ts", required=True,
                    help="UTC timestamp for the read register. REQUIRED: this "
                         "module never reads the clock, so a replay carries "
                         "the historical date.")
    ap.add_argument("--block", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--gate", type=float, default=None,
                    help="a PRE-REGISTERED gate. Omit unless a contract set "
                         "one; this tool does not invent thresholds.")
    ap.add_argument("--wall-caps", default="off", choices=("off", "official"),
                    help="ALL local-seconds caps -- residual, wall, predict "
                         "timeout -- together. Default off, which is what an "
                         "ACCURACY-ONLY run needs; 'official' arms them and "
                         "makes the run a deployment check instead.")
    ap.add_argument("--official-samples", type=float, default=1e9)
    ap.add_argument("--out", default="")
    ap.add_argument("--rows-jsonl", default="",
                    help="stream every completed row (arm label, candidate sha256, "
                         "row) to this JSONL, flushed and fsynced as it lands, so "
                         "a crash keeps every finished network")
    ap.add_argument("--pred-out", default="",
                    help="DIRECTORY for per-network prediction sidecars: the "
                         "FINAL-LAYER array each arm actually had scored, plus "
                         "the run's source/config/truth/mint/package identity "
                         "(`local_sim/design_sidecar.py`). Written append-only "
                         "and fsynced per row, so a crash keeps every finished "
                         "network. Costs ~96 KiB per arm at 24x1024 and is what "
                         "the exact prediction-change decomposition needs.")
    ap.add_argument("--reuse-control", default="",
                    help="DESIGN only: reuse a complete frozen control from a "
                         "prior --pred-out directory after strict full-identity "
                         "verification. Refuses legacy/incomplete manifests; "
                         "never reuses timing or DECIDE evidence.")
    a = ap.parse_args(argv)

    import flopscope as flops
    flops.configure(symmetry_warnings=False)
    import competition_score as cs
    import design_sidecar as ds
    import lifecycle_eval as le
    import truth_reads as tr
    if a.reuse_control and not a.pred_out:
        ap.error("--reuse-control requires --pred-out for a new evidence record")

    block = a.block or tr.resolve_block(a.truth)
    with tr.reserve(block, a.truth, consumer="local_sim/p2_design_score.py",
                    purpose="design-tier paired score: %s vs %s"
                            % (os.path.basename(a.candidate),
                               os.path.basename(a.control)),
                    ts=a.ts, instrument=os.path.abspath(__file__),
                    result_artifact=a.out, note=a.note) as truth_path:
        z = dict(np.load(truth_path))

    truth_sha = ds.sha_file(a.truth)
    truth_meta = {k: (z[k].tolist() if hasattr(z[k], "tolist") else z[k])
                  for k in ("mint", "mint_version", "provenance", "generator")
                  if k in z}

    seeds = [int(s) for s in z["seeds"]][:a.n]
    n = len(seeds)
    width = int(z["width"])
    depth = int(z["depth"])
    n_samples = float(z["n_samples"])
    y = ((z["means_r1"][:n, -1, :].astype(np.float64)
          + z["means_r2"][:n, -1, :].astype(np.float64)) / 2.0)
    gt = z["gt_noise"][:n].astype(np.float64) if "gt_noise" in z else None
    # The AVERAGED truth has half the per-replica noise.
    gt_avg = None if gt is None else gt / 2.0

    # EVERY LOCAL-SECONDS CAP GOES TOGETHER, OR NONE DOES.
    #
    # The first version of this module disabled the residual cliff and left the
    # 120 s WALL cap armed. On this host F21 at Strassen depth 4 averages ~73 s
    # per MLP, so three of twenty-four networks blew the wall, were scored at
    # zero-prediction MSE, and dragged the control arm's raw error from ~4e-8
    # to 1.03e-01. The comparison measured the host, not the candidate.
    #
    # Section 0.1 removed LOCAL SECONDS from every decision path. The residual
    # cliff is local seconds; so is the wall cap; so is the predict timeout.
    # Disabling one and arming the others is the worst of both, because the
    # failure it produces looks like an accuracy result.
    #
    # `--wall-caps official` arms all three, which makes the run a deployment
    # check rather than an accuracy comparison. It is never the default.
    accuracy_only = (a.wall_caps == "off")
    caps = ({"residual_wall_time_limit_s": None,
             "wall_time_limit_s": None,
             "predict_timeout_s": 1.0e9,
             "setup_timeout_s": 1.0e9}
            if accuracy_only else {})
    spec = le.LifecycleSpec(width=width, depth=depth,
                            flop_budget=2 ** 41, lam=0.0, **caps)

    print("=" * 96)
    print("DESIGN-TIER PAIRED SCORE -- licenses a contract, never a promotion")
    print("=" * 96)
    print(f"  control    {a.control}")
    print(f"  candidate  {a.candidate}")
    print(f"  truth      {os.path.relpath(a.truth, _ROOT)}   block {block}   "
          f"{n} nets   {width}x{depth}   2 replicas x {n_samples:,.0f}")
    if accuracy_only:
        print("  *** ALL LOCAL-SECONDS CAPS OFF -- ACCURACY-ONLY. Residual,")
        print("  *** wall and predict timeout are all disarmed together. This")
        print("  *** run is NOT deployment evidence for any of them, and its")
        print("  *** failure accounting cannot see a timing violation.")
        print("  *** (Armed, this host's ~73 s/MLP blows the 120 s wall on")
        print("  *** some networks and scores them at zero-prediction MSE --")
        print("  *** which measures the host, not the candidate.)")
    print()

    cand_paths = [x.strip() for x in a.candidate.split(",") if x.strip()]
    labels = [("control", a.control)] + [
        ("cand%d" % i if len(cand_paths) > 1 else "candidate", pth)
        for i, pth in enumerate(cand_paths)]

    import hashlib

    def _sink_for(label, path):
        if not a.rows_jsonl:
            return None
        with open(path, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()

        def sink(row):
            rec = {"arm": label, "candidate": os.path.basename(path),
                   "candidate_sha256": sha, "block": block,
                   "row": row.to_row() if hasattr(row, "to_row") else str(row)}
            with open(a.rows_jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + chr(10))
                fh.flush()
                os.fsync(fh.fileno())
        return sink

    # ---- prediction sidecars (stage 0, handoff 2026-09-17b) ---------------- #
    # Purely additive: the scorer's numbers are unchanged, and with no
    # --pred-out nothing below runs. What it stores is the array the grader
    # lifecycle ACTUALLY SCORED, after every zeroing, so a failed network
    # contributes the zeros it was scored on.
    pred_dir = os.path.abspath(a.pred_out) if a.pred_out else ""
    pred_manifest = {
        "control": a.control, "candidate": a.candidate,
        "truth": os.path.relpath(a.truth, _ROOT).replace(chr(92), "/"),
        "truth_sha256": truth_sha, "block": block, "n": n,
        "width": width, "depth": depth,
        "n_samples_per_replica": n_samples,
        "truth_replicas": 2, "tier": "DESIGN",
        "mint": dict(truth_meta) or {"mint": "mint_provenance.canonical_mint"},
        "config": {"accuracy_only": accuracy_only,
                   "local_seconds_caps": a.wall_caps,
                   "official_samples": float(a.official_samples),
                   "weights_mode": le.DEFAULT_WEIGHTS_MODE,
                   "gate": a.gate,
                   "ts": a.ts,
                   "spec": le.dataclasses.asdict(spec)},
        "packages": {"flopscope": _pkg_ver("flopscope"),
                     "whestbench": _pkg_ver("whestbench"),
                     "numpy": np.__version__,
                     "python": sys.version.split()[0]},
        "instrument_sha256": ds.sha_file(os.path.abspath(__file__)),
        "sidecar_module_sha256": ds.sha_file(ds.__file__),
        "ordered_network_seeds": seeds,
        # Conservatively bind the entire local evaluator source tree. This can
        # cause a harmless cache miss after an unrelated lab edit, never an
        # unsafe reuse after a transitive import changes.
        "dependency_sha256": {
            os.path.relpath(os.path.join(_HERE, name), _ROOT).replace(chr(92), "/"):
                ds.sha_file(os.path.join(_HERE, name))
            for name in sorted(os.listdir(_HERE)) if name.endswith(".py")},
        "numerical_environment": {
            "platform": platform.platform(), "machine": platform.machine(),
            "python": sys.version, "executable": sys.executable,
            "numpy_build": str(np.__config__.CONFIG),
            "threads": {key: os.environ.get(key) for key in
                        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                         "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}},
        "arms": {},
    }

    arms = {}
    for label, path in labels:
        t0 = time.perf_counter()
        writer = None
        preds = None
        sink = _sink_for(label, path)
        reused = False
        if label == "control" and a.reuse_control:
            if not accuracy_only or spec.lam != 0:
                raise ValueError("control reuse is DESIGN accuracy-only with lambda zero")
            old = ds.verify(a.reuse_control, root=_ROOT,
                            require_present_sources=True,
                            require_full_row_binding=True)
            om = old["manifest"]
            if om.get("tier") != "DESIGN":
                raise ValueError("frozen control reuse requires a DESIGN source tier")
            if om.get("run_complete") is not True:
                raise ValueError("frozen control source run is incomplete")
            for key in ("truth_sha256", "block", "n", "width", "depth",
                        "n_samples_per_replica", "truth_replicas", "mint",
                        "packages", "instrument_sha256",
                        "sidecar_module_sha256", "ordered_network_seeds",
                        "dependency_sha256", "numerical_environment"):
                if key not in om or om[key] != pred_manifest[key]:
                    raise ValueError(f"frozen control identity mismatch: {key}")
            # Timestamp is a read-register event, not evaluator configuration.
            # All numerical/protocol controls, including the registered gate,
            # must match. No automatic DECIDE recovery is provided.
            old_cfg = dict(om["config"])
            new_cfg = dict(pred_manifest["config"])
            old_cfg.pop("ts", None); new_cfg.pop("ts", None)
            if old_cfg != new_cfg:
                raise ValueError("frozen control configuration mismatch")
            old_arm = om["arms"].get("control", {})
            if (old_arm.get("candidate_sha256") != ds.sha_file(path)
                    or old_arm.get("evidence_rows") is None
                    or old_arm.get("rows_hash") is None):
                raise ValueError("frozen control lacks exact source or complete evidence rows")
            rows = [le.EvidenceRow(**r) for r in old_arm["evidence_rows"]]
            if (len(rows) != n or [r.input_seed for r in rows] != seeds
                    or le.rows_hash(rows) != old_arm["rows_hash"]):
                raise ValueError("frozen control evidence rows mismatch")
            if any(r.failed for r in rows):
                raise ValueError("failed frozen control is not reusable")
            reused = True
        if pred_dir:
            with open(path, "rb") as fh:
                import hashlib as _h
                _sha = _h.sha256(fh.read()).hexdigest()
            writer = ds.ArmWriter(pred_dir, label, width=width,
                                  candidate=path, candidate_sha256=_sha)
            preds = []
            _inner = sink

            def sink(row, _w=writer, _p=preds, _i=_inner):
                if _i is not None:
                    _i(row)
                _w.append(row, _p[-1])
            pred_manifest["arms"][label] = {
                "path": path, "candidate_sha256": _sha, "n_written": 0}
            ds.write_manifest(pred_dir, pred_manifest)
        if reused:
            for row, prediction in zip(rows, old["arms"]["control"]["predictions"]):
                preds.append(prediction)
                if sink is not None:
                    sink(row)
            prov = {"candidate_sha256": ds.sha_file(path),
                    "reused_numerical_evidence": True}
        else:
            rows, prov = le.evaluate(path, seeds, y, spec=spec,
                                     gt_noise=None if gt_avg is None
                                     else list(gt_avg), progress_every=0,
                                     predictions_out=preds,
                                     evidence_sink=sink)
        if writer is not None:
            pred_manifest["arms"][label]["n_written"] = writer.n
            pred_manifest["arms"][label]["rows_hash"] = le.rows_hash(rows)
            pred_manifest["arms"][label]["evidence_rows"] = [
                dataclasses.asdict(row) for row in rows]
            pred_manifest["arms"][label]["reused_numerical_evidence"] = reused
            pred_manifest["arms"][label]["n_failed"] = int(
                sum(1 for r in rows if r.failed))
            ds.write_manifest(pred_dir, pred_manifest)
        suite = le.score(rows, spec=spec)
        arms[label] = {"rows": rows, "suite": suite, "prov": prov,
                       "secs": None if reused else time.perf_counter() - t0,
                       "reused": reused}
        f = float(np.mean([r.flops_used for r in rows]))
        wall = [r.wall_time_s for r in rows]
        print(f"  {label:<10} score {suite.objective:.6e}   "
              f"raw MSE {suite.mean_mse:.6e}   C/B {f / 2 ** 41:.6f}   "
              f"mult {suite.mean_multiplier:.6f}   failed {suite.n_failed}/{n}"
              + ("   reused frozen numerical evidence" if reused else
                 f"   {arms[label]['secs']:.0f}s"))
        if not reused:
            print(f"             ADVISORY local wall: median "
                  f"{float(np.median(wall)):.1f}s  worst {float(np.max(wall)):.1f}s"
                  f"   -- LOCAL seconds, gates nothing, not the server's")
        if suite.n_failed:
            print(f"             *** {suite.n_failed} FAILED. With caps off "
                  "this is a real failure, not a timing artefact -- read the "
                  "flags before trusting any number above.")

    if pred_dir:
        pred_manifest["run_complete"] = True
        ds.write_manifest(pred_dir, pred_manifest)

    c_rows = [r.to_row() for r in arms["control"]["rows"]]
    comparisons = {}
    for label, _pth in labels[1:]:
        k_rows = [r.to_row() for r in arms[label]["rows"]]
        print()
        print("=" * 96)
        print(f"  {label}  vs  control")
        print("=" * 96)
        nac = None
        if gt_avg is not None:
            nac = cs.noise_corrected_compare(
                c_rows, k_rows, local_samples=n_samples * 2.0,
                official_samples=a.official_samples, require_ids=True)
            print(nac)
            paired = nac.observed
        else:
            paired = cs.paired_compare_rows(c_rows, k_rows, require_ids=True)
            print("  no gt_noise in the truth file: observed difference only")

        print()
        print("  PAIRED arithmetic score difference (the binding statistic)")
        print(f"    mean difference   {paired.mean_difference:+.6e}")
        print(f"    studentised-t 95% [{paired.t_ci_difference[0]:+.4e}, "
              f"{paired.t_ci_difference[1]:+.4e}]")
        print(f"    arithmetic gain   {paired.arithmetic_gain:.4f}x   "
              f"candidate better on {paired.n_candidate_better}/{paired.n}   "
              f"sign p {paired.sign_test_p:.4f}")
        print(f"    geometric gain    {paired.geometric_gain:.4f}x   "
              "DIAGNOSTIC ONLY, no authority")

        # THE RATIO ON THE NOISE-CORRECTED SCALE, which is the one a public
        # forecast needs. `noise_bias` being ~0 says the two MULTIPLIERS are
        # close; it says nothing about the ratio, because the label floor sits
        # in BOTH arms' MSE and cancels from the difference while inflating the
        # denominator. An independent Skeptic caught exactly that conflation.
        if nac is not None:
            tf = nac.true_function
            print(f"    noise-corrected gain {tf.arithmetic_gain:.4f}x   "
                  f"(observed {paired.arithmetic_gain:.4f}x)")
            print(f"      control true-function mean {tf.mean_control:.6e}, "
                  f"candidate {tf.mean_candidate:.6e}")
            print("      The label floor sits in both arms and cancels from the")
            print("      DIFFERENCE, not from the RATIO. Quote this one for a")
            print("      public forecast; quote the observed one for this run.")

        verdict = None
        if a.gate is not None:
            v = cs.decide(arms["control"]["suite"].scores,
                          arms[label]["suite"].scores, gate=a.gate)
            print()
            print(v)
            verdict = {"passed": bool(v.passed), "gain": float(v.gain),
                       "gate": float(v.gate),
                       "ci_excludes_zero": bool(v.ci_excludes_zero),
                       "biggest_mover_share": float(v.biggest_mover_share),
                       "reasons": list(v.reasons)}
        comparisons[label] = (paired, nac, verdict)
    if a.gate is None:
        print()
        print("  no --gate given, so no verdict. This tool does not invent "
              "thresholds.")

    if a.out:
        d = os.path.dirname(os.path.abspath(a.out))
        if d:
            os.makedirs(d, exist_ok=True)
        payload = {
            "control": a.control, "candidate": a.candidate,
            "truth": os.path.relpath(a.truth, _ROOT), "block": block,
            "n": n, "width": width, "depth": depth,
            "n_samples_per_replica": n_samples,
            "accuracy_only": accuracy_only,
            "local_seconds_caps": a.wall_caps,
            "tier": "DESIGN", "promotes_nothing": True,
            "truth_sha256": truth_sha,
            "prediction_sidecars": (os.path.relpath(pred_dir, _ROOT).replace(
                chr(92), "/") if pred_dir else None),
            "arms": {k: {"path": dict(labels)[k],
                         "objective": float(v["suite"].objective),
                         "raw_mse": float(v["suite"].mean_mse),
                         "multiplier": float(v["suite"].mean_multiplier),
                         "n_failed": int(v["suite"].n_failed),
                         "flops_mean": float(np.mean(
                             [r.flops_used for r in v["rows"]])),
                         "rows_hash": le.rows_hash(v["rows"]),
                         "candidate_sha256": v["prov"].get("candidate_sha256",
                                                           ""),
                         "seconds": v["secs"],
                         "reused_numerical_evidence": v["reused"],
                         # PER-MLP ROWS, not just their hash. Without these
                         # nobody can recompute the interval, find the biggest
                         # mover, or inspect a guarded row -- a Skeptic could
                         # only take the printed summary on trust.
                         "rows": [{"index": r.index, "seed": r.input_seed,
                                   "mse": float(r.mse),
                                   "flops_used": float(r.flops_used),
                                   "gt_noise": float(r.gt_noise),
                                   "wall_time_s": (None if v["reused"] else
                                                   float(r.wall_time_s)),
                                   "residual_wall_time_s":
                                       (None if v["reused"] else
                                        float(r.residual_wall_time_s)),
                                   "failed": bool(r.failed),
                                   "error_code": r.error_code,
                                   "nonfinite": bool(r.nonfinite)}
                                  for r in v["rows"]]}
                     for k, v in arms.items()},
            "comparisons": {
                lab: {
                    "mean_difference": float(pr.mean_difference),
                    "t_ci": [float(x) for x in pr.t_ci_difference],
                    "arithmetic_gain": float(pr.arithmetic_gain),
                    "geometric_gain_DIAGNOSTIC": float(pr.geometric_gain),
                    "n_candidate_better": int(pr.n_candidate_better),
                    "sign_test_p": float(pr.sign_test_p),
                    "notes": list(pr.notes),
                    "verdict": vd,
                    "noise_aware": (None if na is None else {
                        "true_function_difference":
                            float(na.true_function.mean_difference),
                        "true_function_gain":
                            float(na.true_function.arithmetic_gain),
                        "true_function_mean_control":
                            float(na.true_function.mean_control),
                        "true_function_mean_candidate":
                            float(na.true_function.mean_candidate),
                        "forecast_difference":
                            float(na.forecast.mean_difference),
                        "forecast_gain":
                            float(na.forecast.arithmetic_gain),
                        "noise_bias": float(na.noise_bias),
                        "noise_bias_share": float(na.noise_bias_share),
                        "n_negative_corrected": int(na.n_negative_corrected),
                    }),
                }
                for lab, (pr, na, vd) in comparisons.items()},
        }
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2, default=str) + "\n")
        print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
