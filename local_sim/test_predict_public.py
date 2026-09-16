"""The public-score projector does the arithmetic it claims, and refuses the rest.

Slice 6 contract, `docs/CLAUDE_CODE_HANDOFF_2026-09-14.md` §0.5; repaired
2026-09-16 against `docs/research/2026-09-16_validation_audit.md`.

The projector's whole job is to answer "did that help?" without spending a
submission slot, so the ways it can be WRONG are the ways a slot gets spent on
a candidate that did not help:

  * pairing two runs that did not see the same networks, or not the same dataset;
  * silently dropping rows from a truncated report;
  * projecting from a failed or incomplete anchor;
  * a local control produced by different bytes than the graded control;
  * the wrong FLOP offset for the meter path;
  * calling a transformed difference interval a prediction interval;
  * a wall forecast that under-predicts the server on a quiet host.

Synthetic reports plus trimmed copies of REAL reports under
`local_sim/fixtures/predict_public/` (each carries `_fixture.source_report_sha256`
of its original). No truth is read, no network is minted, no block is claimed.

Run: uv run python local_sim/test_predict_public.py
"""

import copy
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

import competition_score as cs  # noqa: E402
import official_config as oc  # noqa: E402
import predict_public as pp  # noqa: E402

RESULTS = []
B = 2.0 ** 41
FIX = os.path.join(_HERE, "fixtures", "predict_public")
F21_SHA = "d77dab0e2c740cc9f6cd7d9172adacc8da731cc5b40ca87798f3fb006e475a94"
V31_SHA = "6dde5fd43ad41e8ec7199b742dd554d22d506c9cf85a532027e86a590d0a787c"
DS_SHA = "264fa1f416d16a40821fb5e8e94f5d2da4698a201d40da999616225b38b464f1"
TMP = tempfile.mkdtemp(prefix="test_predict_public_")


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {name:<70} {'ok ' if ok else 'FAILED'}  {detail}")
    return ok


def head(t):
    print(f"\n{t}\n{'-' * len(t)}")


def refuses(fn, exc=pp.ProjectionRefused, needle=""):
    try:
        fn()
    except exc as e:
        return (needle.lower() in str(e).lower()), str(e)[:90]
    return False, "did not refuse"


def make_run(n=40, mse=4.0e-8, cb=0.42, seed=1, ops_per_mlp=9000,
             wall=30.0, fail_idx=(), names=None, ds_sha=DS_SHA, ops_by_idx=None):
    """A synthetic `whest run --json` report with everything the tool reads."""
    rs = np.random.default_rng(seed)
    per = []
    for i in range(n):
        ops = int((ops_by_idx or {}).get(i, ops_per_mlp))
        per.append({
            "mlp_index": i,
            "mlp_name": (names[i] if names else "net-%03d" % i),
            "final_layer_mse": float(mse * rs.uniform(0.85, 1.15)),
            "flops_used": int(cb * B),
            "wall_time_s": float(wall * rs.uniform(0.9, 1.1)),
            "residual_wall_time_s": 0.12,
            "budget_exhausted": False, "time_exhausted": False,
            "residual_wall_time_exhausted": i in fail_idx,
            "combined_budget_exhausted": False, "traceback": None,
            "breakdowns": {"estimator": {"by_namespace": {
                "estimator.estimator-client": {"calls": ops}}}},
        })
    return {
        "results": {
            "per_mlp": per,
            "n_failed_mlps": len(fail_idx),
            "breakdowns": {"estimator": {"by_namespace": {
                "estimator.estimator-client": {
                    "calls": sum(p["breakdowns"]["estimator"]["by_namespace"]
                                 ["estimator.estimator-client"]["calls"] for p in per)}}}},
        },
        "run_meta": {"host": {"hostname": "h", "python_version": "3.10.19",
                              "numpy_version": "2.2.6", "flopscope_version": "0.12.1"}},
        "run_config": {"flop_budget": int(B), "width": 1024, "depth": 16,
                       "n_mlps": n, "wall_time_limit_s": 120.0,
                       "residual_wall_time_limit_s": None,
                       "lambda_flops_per_second": 0.0,
                       "dataset": {"path": "hf://x@v2-phase2", "sha256": ds_sha,
                                   "n_mlps": n}},
        "whestbench_version": "0.16.1",
    }


def fake_graded(adjusted=1.5957e-8, n=100, fail_idx=(), drop_public=()):
    """A graded report whose replay lands on a known public score."""
    mult = 0.422214
    mse = adjusted / mult
    pm, ps = [], []
    for i in range(n):
        pm.append({"mlp_index": i, "mlp_name": "graded-%03d" % i,
                   "state": "completed", "error_code": None,
                   "telemetry": {"flops_used": mult * B, "residual_wall_time_s": 0.1,
                                 "participant_predict_wall_s": 60.0,
                                 "residual_wall_time_exhausted": i in fail_idx}})
        if i < n // 2 and i not in drop_public:
            ps.append({"mlp_index": i, "scores": {"final_layer_mse": mse}})
    return {"evaluation": {"config_snapshot": {"flop_budget": B, "n_mlps": n,
                                               "width": 1024, "depth": 16},
                           "submission_id": "330950", "state": "completed",
                           "progress": {"mlps_failed": len(fail_idx)},
                           "results": {"per_mlp": pm, "public_scores": ps}}}


def fixture(name):
    return os.path.join(FIX, name)


def write_json(obj, name):
    p = os.path.join(TMP, name)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    return p


def write_receipt(report_path, source_sha, meter="whest_run_subprocess",
                  load="standalone", name=None):
    rec = {"schema": pp.RECEIPT_SCHEMA, "report_sha256": pp._sha256(report_path),
           "source_path": "x.py", "source_sha256": source_sha, "meter_path": meter,
           "host_load": {"class": load, "evidence": "test"},
           "provenance": {"kind": "test"}}
    return write_json(rec, name or (os.path.basename(report_path) + ".rcpt.json"))


def write_ledger(rows, name="ledger.jsonl"):
    p = os.path.join(TMP, name)
    with open(p, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


LEDGER = write_ledger([{"submission_id": "330950", "inner_estimator_sha256": F21_SHA},
                       {"submission_id": "331317", "inner_estimator_sha256": V31_SHA}])
ANCHOR = pp.graded_anchor(fake_graded())


def args(**kw):
    base = dict(candidate_run="", control_run="", control_graded="",
                candidate_receipt="", control_receipt="", submissions_ledger=LEDGER,
                meter_path="", unverified_control_source=False,
                unverified_candidate_source=False, candidate="", gate=None,
                allow_failures=False, json="")
    base.update(kw)
    return types.SimpleNamespace(**base)


# --------------------------------------------------------------------------- #


def t_anchor():
    head("1. the anchor is REPLAYED, and only from a clean, complete report")
    check("the synthetic anchor replays to its intended score",
          abs(ANCHOR["adjusted"] - 1.5957e-8) / 1.5957e-8 < 1e-9,
          f"{ANCHOR['adjusted']:.6e}")
    check("...over all 50 public rows of 100, none failed",
          ANCHOR["n"] == 50 and ANCHOR["n_rows"] == 100 and ANCHOR["n_failed"] == 0, "")
    a = pp.graded_anchor(pp.load_run(fixture("f21_graded_330950.json")))
    check("the REAL F21 #330950 report (trimmed) replays to 1.5957e-8",
          abs(a["adjusted"] - 1.595740e-8) < 1e-13,
          f"{a['adjusted']:.6e}  wall max {a['wall']['max']:.2f} s")
    real = os.path.join(_ROOT, "experiments", "run_reports",
                        "f21_diagnostic_330950_graded_report.json")
    if os.path.exists(real):
        b = pp.graded_anchor(pp.load_run(real))
        check("...and the trimmed fixture replays identically to the original",
              b["adjusted"] == a["adjusted"], "")

    ok, msg = refuses(lambda: pp.graded_anchor(fake_graded(fail_idx=(7,))),
                      needle="failed MLPs")
    check("NEGATIVE: a graded anchor with a failed MLP is refused", ok, msg)
    ok, msg = refuses(lambda: pp.graded_anchor(fake_graded(fail_idx=(80,))),
                      needle="failed MLPs")
    check("...even when the failed row is in the PRIVATE half", ok, msg)
    ok, msg = refuses(lambda: pp.graded_anchor(fake_graded(drop_public=(3,))),
                      needle="public score rows")
    check("NEGATIVE: a graded anchor missing a public score row is refused", ok, msg)
    g = fake_graded()
    g["evaluation"]["results"]["per_mlp"] = g["evaluation"]["results"]["per_mlp"][:99]
    ok, msg = refuses(lambda: pp.graded_anchor(g), needle="per_mlp rows")
    check("NEGATIVE: a graded anchor with a missing per_mlp row is refused", ok, msg)
    g = fake_graded()
    g["evaluation"]["state"] = "failed"
    ok, msg = refuses(lambda: pp.graded_anchor(g), needle="state")
    check("NEGATIVE: an evaluation that did not complete is refused", ok, msg)


def t_projection_arithmetic():
    head("2. projection = anchor x paired ratio; the band is labelled for what it is")
    ctrl = make_run(seed=1, mse=4.0e-8)
    cand = copy.deepcopy(ctrl)
    for e in cand["results"]["per_mlp"]:
        e["final_layer_mse"] *= 0.8
    res = pp.project(pp.run_rows(cand), pp.run_rows(ctrl), ANCHOR)
    check("a uniform 20% MSE improvement gives ratio 0.8",
          abs(res["ratio"] - 0.8) < 1e-12, f"{res['ratio']:.9f}")
    check("...and arithmetic gain 1.25x", abs(res["arithmetic_gain"] - 1.25) < 1e-9,
          f"{res['arithmetic_gain']:.6f}")
    want = ANCHOR["adjusted"] * 0.8
    check("the projected public score is anchor x ratio",
          abs(res["projected_public_adjusted"] - want) / want < 1e-12,
          f"{res['projected_public_adjusted']:.6e} vs {want:.6e}")
    # REPLACED 2026-09-16. The old test asserted "a uniform ratio still leaves a
    # real interval, and it brackets 0.8" as if the band were a ratio CI. For an
    # EXACTLY constant ratio a ratio-estimator CI would have zero width; this one
    # does not, which proves it is a transformed DIFFERENCE interval. Assert that,
    # and the label, instead.
    lo, hi = res["transformed_band_ratio"]
    check("an exactly constant ratio still gives a NONZERO band -> not a ratio CI",
          hi - lo > 0, f"[{lo:.6f}, {hi:.6f}]")
    check("...and the band is exactly the t difference interval, transformed",
          abs(lo - (res["local_control_adjusted"] - res["t_ci_difference"][1])
              / res["local_control_adjusted"]) < 1e-12, "")
    check("the band carries the conditional label, verbatim",
          res["interval_label"] == pp.INTERVAL_LABEL
          and "conditional" in pp.INTERVAL_LABEL
          and "excludes hidden-network transfer uncertainty" in pp.INTERVAL_LABEL, "")
    check("...and declares it is NOT a hidden-board prediction interval",
          res["interval_is_hidden_board_prediction_interval"] is False
          and "ratio_ci" not in res and "projected_public_ci" not in res,
          "old keys removed")
    check("the label never says 'prediction interval' or '95%'",
          "prediction interval" not in pp.INTERVAL_LABEL and "95" not in pp.INTERVAL_LABEL, "")
    same = pp.project(pp.run_rows(ctrl), pp.run_rows(ctrl), ANCHOR)
    check("identical arms project the anchor back, exactly",
          same["projected_public_adjusted"] == ANCHOR["adjusted"], "")


def t_cost_enters_the_score():
    head("3. cost is half the objective, and the projector knows it")
    ctrl = make_run(seed=3, mse=4.0e-8, cb=0.42)
    cheap = make_run(seed=3, mse=4.0e-8, cb=0.21)
    res = pp.project(pp.run_rows(cheap), pp.run_rows(ctrl), ANCHOR)
    check("halving C/B at equal MSE halves the projected score",
          abs(res["ratio"] - 0.5) < 1e-9, f"ratio {res['ratio']:.9f}")
    floored = make_run(seed=3, mse=4.0e-8, cb=0.02)
    r2 = pp.project(pp.run_rows(floored), pp.run_rows(ctrl), ANCHOR)
    check("...and the 0.1 multiplier FLOOR stops it going further",
          abs(r2["ratio"] - (0.1 / 0.42)) < 1e-9, f"ratio {r2['ratio']:.6f}")


def t_refusals():
    head("4. identity, completeness and failures -- each with a passing control")
    ctrl = make_run(seed=5)
    cand = make_run(seed=6)
    ok_res = pp.project(pp.run_rows(cand), pp.run_rows(ctrl), ANCHOR)
    check("the positive control projects", math.isfinite(ok_res["ratio"]), "")
    check("...and the pair identity check passes",
          pp.check_pair_identity(cand, ctrl) == [], "")

    other = make_run(seed=6, names=["other-%03d" % i for i in range(40)])
    ok, msg = refuses(lambda: pp.check_pair_identity(other, ctrl), needle="not the same MLPs")
    check("two runs over DIFFERENT networks are refused", ok, msg)
    try:
        pp.project(pp.run_rows(other), pp.run_rows(ctrl), ANCHOR)
        ok = False
    except cs.PairingError:
        ok = True
    check("...and project() itself still refuses them", ok, "")

    colliding = make_run(seed=6, ds_sha="0" * 64)
    ok, msg = refuses(lambda: pp.check_pair_identity(colliding, ctrl),
                      needle="different datasets")
    check("NEGATIVE: COLLIDING names on a different dataset sha256 are refused", ok, msg)
    nosha = make_run(seed=6)
    nosha["run_config"]["dataset"].pop("sha256")
    ok, msg = refuses(lambda: pp.check_pair_identity(nosha, ctrl), needle="sha256")
    check("a report with no dataset sha256 cannot be bound and is refused", ok, msg)

    fewer = make_run(n=39, seed=6)
    fewer["run_config"]["dataset"]["n_mlps"] = 40   # same dataset identity
    ok, msg = refuses(lambda: pp.check_pair_identity(fewer, ctrl), needle="not the same MLPs")
    check("a different COUNT of networks is refused", ok, msg)

    truncated = make_run(seed=6)
    truncated["results"]["per_mlp"] = truncated["results"]["per_mlp"][:-3]
    ok, msg = refuses(lambda: pp.run_rows(truncated), needle="TRUNCATED")
    check("NEGATIVE: a TRUNCATED report (rows < n_mlps) is refused", ok, msg)

    nomse = make_run(seed=6)
    nomse["results"]["per_mlp"][4]["final_layer_mse"] = None
    ok, msg = refuses(lambda: pp.run_rows(nomse), needle="final_layer_mse")
    check("NEGATIVE: a row missing its MSE is refused, not skipped", ok, msg)
    nan = make_run(seed=6)
    nan["results"]["per_mlp"][4]["final_layer_mse"] = float("nan")
    ok, msg = refuses(lambda: pp.run_rows(nan), needle="final_layer_mse")
    check("...and so is a NaN MSE", ok, msg)

    stack = make_run(seed=6)
    stack["whestbench_version"] = "0.16.0"
    ok, msg = refuses(lambda: pp.check_pair_identity(stack, ctrl), needle="whestbench_version")
    check("arms on different evaluator versions are refused", ok, msg)
    host = make_run(seed=6)
    host["run_meta"]["host"]["hostname"] = "other"
    notes = pp.check_pair_identity(host, ctrl)
    check("...a different HOST is only a timing note", any("hostname" in n for n in notes), "")

    perm = copy.deepcopy(cand)
    perm["results"]["per_mlp"] = list(reversed(perm["results"]["per_mlp"]))
    res_p = pp.project(pp.run_rows(perm), pp.run_rows(ctrl), ANCHOR)
    check("a PERMUTED arm is realigned, not mispaired",
          abs(res_p["ratio"] - ok_res["ratio"]) < 1e-12
          and abs(res_p["t_ci_difference"][0] - ok_res["t_ci_difference"][0]) < 1e-24, "")

    failed = make_run(seed=6, fail_idx=(3, 11))
    ok, msg = refuses(lambda: pp.project(pp.run_rows(failed), pp.run_rows(ctrl), ANCHOR),
                      needle="failed MLPs")
    check("an arm with FAILED MLPs is refused by default", ok, msg)
    res_f = pp.project(pp.run_rows(failed), pp.run_rows(ctrl), ANCHOR, allow_failures=True)
    check("...but --allow-failures proceeds and REPORTS the count",
          res_f["n_failed_candidate"] == 2, "")

    nob = make_run(seed=6)
    nob["run_config"]["flop_budget"] = 0
    ok, _ = refuses(lambda: pp.run_rows(nob))
    check("a report with no budget is refused, not defaulted", ok, "")


def t_source_binding():
    head("5. the local control must be the bytes that were graded")
    ctrl_p = write_json(make_run(seed=5), "ctrl.json")
    good = pp.load_receipt(write_receipt(ctrl_p, F21_SHA), ctrl_p, label="control")
    g = fake_graded()
    sha, ev = pp.graded_source_sha256(g, LEDGER)
    check("the graded source hash is resolved from the ledger", sha == F21_SHA, ev)
    b = pp.bind_control_source(good, sha, ev, allow_unverified=False)
    check("matching control source -> VERIFIED", b["status"] == "VERIFIED", "")

    wrong = pp.load_receipt(write_receipt(ctrl_p, V31_SHA, name="wrong.json"),
                            ctrl_p, label="control")
    ok, msg = refuses(lambda: pp.bind_control_source(wrong, sha, ev, allow_unverified=False),
                      needle="WRONG CONTROL SOURCE")
    check("NEGATIVE: a control run from DIFFERENT source bytes is refused", ok, msg)
    ok, msg = refuses(lambda: pp.bind_control_source(wrong, sha, ev, allow_unverified=True),
                      needle="WRONG CONTROL SOURCE")
    check("...and --unverified-control-source does NOT override a mismatch", ok, msg)

    ok, msg = refuses(lambda: pp.bind_control_source(None, sha, ev, allow_unverified=False),
                      needle="--unverified-control-source")
    check("no control receipt -> refused, naming the flag", ok, msg)
    u = pp.bind_control_source(None, sha, ev, allow_unverified=True)
    check("...with the flag -> proceeds labelled UNVERIFIED", u["status"] == "UNVERIFIED",
          u["evidence"][:60])
    nosha, nev = pp.graded_source_sha256(g, write_ledger([], "empty.jsonl"))
    ok, msg = refuses(lambda: pp.bind_control_source(good, nosha, nev,
                                                     allow_unverified=False),
                      needle="graded source hash unavailable")
    check("no graded source hash -> refused as well", ok, msg)

    real_ledger = os.path.join(_ROOT, "experiments", "submissions.jsonl")
    if os.path.exists(real_ledger):
        rs, rev = pp.graded_source_sha256(
            pp.load_run(fixture("f21_graded_330950.json")), real_ledger)
        check("REAL ledger: #330950 resolves to F21's bytes via its package tarball",
              rs == F21_SHA, rev[:70])
        rs, rev = pp.graded_source_sha256(
            pp.load_run(fixture("v31_graded_331317.json")), real_ledger)
        check("REAL ledger: #331317 resolves to v31's bytes", rs == V31_SHA, rev[:70])

    other_p = write_json(make_run(seed=9), "other.json")
    ok, msg = refuses(lambda: pp.load_receipt(write_receipt(ctrl_p, F21_SHA, name="r2.json"),
                                              other_p, label="control"),
                      needle="different report bytes")
    check("a receipt for DIFFERENT report bytes is refused", ok, msg)

    head("5b. the candidate hash used for memory lookup must be the run's")
    cand_src = os.path.join(TMP, "cand_src.py")
    with open(cand_src, "wb") as fh:
        fh.write(b"# candidate\n")
    csha = pp._sha256(cand_src)
    cand_p = write_json(make_run(seed=6), "cand.json")
    crec = pp.load_receipt(write_receipt(cand_p, csha), cand_p, label="candidate")
    check("matching --candidate bytes -> VERIFIED",
          pp.bind_candidate_source(crec, cand_src, allow_unverified=False)["status"]
          == "VERIFIED", "")
    bad = pp.load_receipt(write_receipt(cand_p, V31_SHA, name="cbad.json"), cand_p,
                          label="candidate")
    ok, msg = refuses(lambda: pp.bind_candidate_source(bad, cand_src, allow_unverified=True),
                      needle="WRONG CANDIDATE SOURCE")
    check("NEGATIVE: --candidate bytes not matching the run are refused (no override)",
          ok, msg)
    ok, msg = refuses(lambda: pp.bind_candidate_source(None, cand_src, allow_unverified=False),
                      needle="--unverified-candidate-source")
    check("no candidate receipt -> refused unless the flag is passed", ok, msg)


def t_meter_paths():
    head("6. FLOPs to graded are PATH-specific -- regressions on real graded pairs")
    sub = pp.graded_flops(0.25 * B, B, "whest_run_subprocess")
    check("whest_run_subprocess adds exactly 2**25",
          sub["status"] == "VERIFIED" and [e["offset"] for e in sub["entries"]] == [2 ** 25], "")
    twin = pp.graded_flops(0.25 * B, B, "grader_twin_f64_readonly")
    check("grader_twin_f64_readonly adds NOTHING",
          [e["offset"] for e in twin["entries"]] == [0], "")
    unk = pp.graded_flops(0.25 * B, B, "unknown")
    check("unknown -> UNVERIFIED band {0, 2**25}",
          unk["status"] == "UNVERIFIED"
          and [e["offset"] for e in unk["entries"]] == [0, 2 ** 25], "")
    check("2**24 is no longer an offset anywhere",
          2 ** 24 not in pp.UNVERIFIED_OFFSET_BAND
          and 2 ** 24 not in pp.METER_PATH_TO_GRADED_OFFSET.values(), "")

    for loc, gr in (("f21_mini_job1401.json", "f21_graded_330950.json"),
                    ("v31_mini_job1412.json", "v31_graded_331317.json"),
                    ("v31_mini_confirm_job1413.json", "v31_graded_331317.json")):
        L, G = pp.load_run(fixture(loc)), pp.load_run(fixture(gr))
        lf = {e["flops_used"] for e in L["results"]["per_mlp"]}
        gf = {e["telemetry"]["flops_used"] for e in G["evaluation"]["results"]["per_mlp"]}
        out = pp.graded_flops(float(next(iter(lf))), B, "whest_run_subprocess")
        check(f"{loc}: subprocess + offset == graded, exactly",
              len(lf) == 1 and len(gf) == 1
              and out["entries"][0]["flops"] == next(iter(gf)),
              f"{out['entries'][0]['flops']:,.0f} vs {next(iter(gf)):,.0f}")
    twins = pp.load_run(fixture("grader_twin_meters.json"))["meters"]
    graded_by_sub = {"330950": "f21_graded_330950.json", "331317": "v31_graded_331317.json"}
    for m in twins:
        G = pp.load_run(fixture(graded_by_sub[m["graded_submission"]]))
        g = G["evaluation"]["results"]["per_mlp"][0]["telemetry"]["flops_used"]
        out = pp.graded_flops(m["flops_per_mlp"], B, m["meter_path"])
        check(f"{m['candidate']} grader twin f64_readonly + 0 == graded, exactly",
              out["entries"][0]["flops"] == g, f"{g:,.0f}")

    c_p = write_json(make_run(seed=6), "mp_c.json")
    k_p = write_json(make_run(seed=5), "mp_k.json")
    rc = pp.load_receipt(write_receipt(c_p, V31_SHA, meter="whest_run_subprocess"),
                         c_p, label="candidate")
    rk = pp.load_receipt(write_receipt(k_p, F21_SHA, meter="grader_twin_f64_readonly"),
                         k_p, label="control")
    ok, msg = refuses(lambda: pp.resolve_meter_paths(rc, rk), needle="different paths")
    check("arms on DIFFERENT meter paths are refused", ok, msg)
    check("no receipts -> meter path 'unknown'", pp.resolve_meter_paths(None, None) == "unknown", "")


def t_residual_is_operations_on_the_max():
    head("7. residual is priced on OPERATIONS, gated on the per-network MAX")
    r = pp.residual_forecast(9000.0)
    check("the forecast uses the calibrated per-operation model",
          abs(r["predicted_s_point"] - 9000.0 * oc.GRADED_RESIDUAL_S_PER_OP) < 1e-15, "")
    check("9,000 operations passes the 22,000 gate", r["gate_passed"], "")
    check("the model is labelled ~20% conservative, observed 10.73-11.10 us/op",
          "20%" in r["model_note"] and r["graded_us_per_op_observed"] == (10.73, 11.10), "")
    spiky = make_run(n=40, ops_per_mlp=13_000, ops_by_idx={17: 23_000})
    nets = pp.network_summary(spiky)
    check("per-network max and WHICH network are reported",
          nets["ops"]["max"] == 23_000 and nets["ops"]["max_network"] == "net-017",
          f"mean {nets['ops']['mean']:,.0f} max {nets['ops']['max']:,.0f}")
    rf = pp.residual_forecast(nets["ops"]["max"], ops_mean=nets["ops"]["mean"],
                              max_network=nets["ops"]["max_network"])
    check("one network over 22,000 FAILS the gate although the mean passes",
          not rf["gate_passed"] and nets["ops"]["mean"] < oc.PHASE2_OP_COUNT_GATE,
          f"gate basis: {rf['gate_basis']}")
    bare = make_run(n=40)
    for e in bare["results"]["per_mlp"]:
        e.pop("breakdowns")
    check("no per-network call counts -> ops None, no gate (aggregate mean kept)",
          pp.network_summary(bare)["ops"] is None
          and pp.network_summary(bare)["ops_mean_aggregate"] == 9000.0, "")
    L = pp.load_run(fixture("v31_mini_confirm_job1413.json"))
    n = pp.network_summary(L)
    check("REAL v31: 13,037 ops on every network; wall/flops maxima named",
          n["ops"]["max"] == 13037 and n["wall_s"]["max_network"] == "richard-davis",
          f"wall max {n['wall_s']['max']:.2f} s")
    check("the trimmed report has no per-op breakdown -> sync counts ABSENT",
          n["sync_scalar"]["available"] is False and "ABSENT" in n["sync_scalar"]["why"],
          n["sync_scalar"]["why"][:60])
    withops = make_run(n=4)
    for e in withops["results"]["per_mlp"]:
        e["breakdowns"]["estimator"]["by_namespace"]["estimator.estimator-client"][
            "operations"] = {"matmul": {"calls": 3}}
    s = pp.sync_scalar_counts(withops)
    check("per-op breakdown without a sync counter says ABSENT, not zero",
          s["available"] is False and "not zero" in s["why"], "")
    withops["results"]["per_mlp"][2]["breakdowns"]["estimator"]["by_namespace"][
        "estimator.estimator-client"]["operations"]["item"] = {"calls": 5}
    s = pp.sync_scalar_counts(withops)
    check("...and a present counter is summed with its max network",
          s["available"] and s["max_per_network"]["item"]["network"] == "net-002", "")


def t_wall():
    head("8. wall: the x1.60 rule is retired; 1.9 x median is EMPIRICAL and advisory")
    L = pp.load_run(fixture("v31_mini_confirm_job1413.json"))
    G = pp.load_run(fixture("v31_graded_331317.json"))
    nets = pp.network_summary(L)
    graded_max = max(e["telemetry"]["participant_predict_wall_s"]
                     for e in G["evaluation"]["results"]["per_mlp"])
    old = nets["wall_s"]["max"] * oc.GRADED_WALL_CONTENTION_FACTOR
    check("REGRESSION: old rule (standalone local worst x1.60) UNDER-predicts 68.13 s",
          old < graded_max and abs(graded_max - 68.13) < 0.01,
          f"{nets['wall_s']['max']:.2f} x 1.60 = {old:.2f} s < graded {graded_max:.2f} s")
    w = pp.wall_forecast(nets["wall_s"], {"class": "standalone", "evidence": "job 1413"})
    check("new output: 1.9 x standalone median",
          abs(w["empirical_forecast_s"] - 1.9 * nets["wall_s"]["median"]) < 1e-9,
          f"{w['empirical_forecast_s']:.2f} s")
    check("...labelled EMPIRICAL_FORECAST, advisory, not a gate",
          w["label"] == pp.EMPIRICAL and w["advisory"] is True and w["is_gate"] is False
          and "not a safety proof" in w["caveat"].lower(), "")
    check("...and no 'projected_worst_s' / contention factor in the output",
          "projected_worst_s" not in w and "contention_factor" not in w, "")
    check("local median AND worst (with network) are both reported",
          w["local_worst_network"] == "richard-davis" and w["local_median_s"] < w["local_worst_s"], "")
    wc = pp.wall_forecast(nets["wall_s"], {"class": "concurrent"})
    check("a contended run is flagged OUT of the evidence domain",
          wc["in_evidence_domain"] is False and w["in_evidence_domain"] is True, "")
    wu = pp.wall_forecast(nets["wall_s"], None)
    check("no receipt -> host load 'unknown', stated", wu["host_load"]["class"] == "unknown", "")
    check("a report with no wall times says so rather than assuming",
          pp.wall_forecast(None)["available"] is False, "")


def t_end_to_end():
    head("9. end to end on trimmed REAL reports: v31 (job 1413) vs F21 + #330950")
    cand_p = fixture("v31_mini_confirm_job1413.json")
    ctrl_p = fixture("f21_mini_job1401.json")
    graded_p = fixture("f21_graded_330950.json")
    rc = write_receipt(cand_p, V31_SHA, load="standalone", name="e2e_c.json")
    rk = write_receipt(ctrl_p, F21_SHA, load="concurrent", name="e2e_k.json")
    out = pp.build(args(candidate_run=cand_p, control_run=ctrl_p, control_graded=graded_p,
                        candidate_receipt=rc, control_receipt=rk))
    res = out["res"]
    check("projected v31 = 8.7133e-9 (graded #331317 was 8.752e-9)",
          abs(res["projected_public_adjusted"] - 8.713275557293036e-09) < 1e-15,
          f"{res['projected_public_adjusted']:.6e}")
    check("...binding difference and t interval unchanged from the stored replay",
          abs(res["mean_difference"] - 7.522869220696936e-09) < 1e-17, "")
    check("provenance VERIFIED on both sources and the meter path",
          out["meta"]["control_source"]["status"] == "VERIFIED"
          and out["cost"]["status"] == "VERIFIED", "")
    check("graded-FLOP cross-check moves the projection by < 1e-4 relative",
          abs(res["projected_public_at_graded_flops"] / res["projected_public_adjusted"] - 1)
          < 1e-4, f"{res['projected_public_at_graded_flops']:.6e}")
    check("residual forecast is gated on the max network",
          out["resid"]["ops_max"] == 13037 and out["resid"]["gate_passed"], "")

    jpath = os.path.join(TMP, "should_not_exist.json")
    wrong = write_receipt(ctrl_p, V31_SHA, name="e2e_wrong.json")
    code = pp.main(["--candidate-run", cand_p, "--control-run", ctrl_p,
                    "--control-graded", graded_p, "--candidate-receipt", rc,
                    "--control-receipt", wrong, "--submissions-ledger", LEDGER,
                    "--json", jpath])
    check("CLI: wrong control source exits 2 and writes no JSON",
          code == 2 and not os.path.exists(jpath), f"exit {code}")
    code = pp.main(["--candidate-run", cand_p, "--control-run", ctrl_p,
                    "--control-graded", graded_p, "--candidate-receipt", rc,
                    "--submissions-ledger", LEDGER, "--json", jpath])
    check("CLI: no control receipt and no flag exits 2", code == 2, f"exit {code}")
    bad_g = pp.load_run(graded_p)
    bad_g["evaluation"]["results"]["per_mlp"][60]["telemetry"]["time_exhausted"] = True
    code = pp.main(["--candidate-run", cand_p, "--control-run", ctrl_p,
                    "--control-graded", write_json(bad_g, "bad_graded.json"),
                    "--candidate-receipt", rc, "--control-receipt", rk,
                    "--submissions-ledger", LEDGER, "--json", jpath])
    check("CLI: a failed graded anchor exits 2", code == 2, f"exit {code}")
    trunc = pp.load_run(cand_p)
    trunc["results"]["per_mlp"] = trunc["results"]["per_mlp"][:97]
    tp = write_json(trunc, "trunc.json")
    code = pp.main(["--candidate-run", tp, "--control-run", ctrl_p,
                    "--control-graded", graded_p, "--control-receipt", rk,
                    "--submissions-ledger", LEDGER, "--json", jpath])
    check("CLI: a truncated candidate report exits 2", code == 2, f"exit {code}")


def main():
    print("=" * 100)
    print("THE PUBLIC-SCORE PROJECTOR — anchor x paired ratio, labelled, and refusing")
    print("=" * 100)
    try:
        for fn in (t_anchor, t_projection_arithmetic, t_cost_enters_the_score,
                   t_refusals, t_source_binding, t_meter_paths,
                   t_residual_is_operations_on_the_max, t_wall, t_end_to_end):
            fn()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    n_ok = sum(1 for r in RESULTS if r)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
