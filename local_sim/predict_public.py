"""What would this candidate score on the public board? One answer, labelled by kind.

Slice 6 contract, `docs/CLAUDE_CODE_HANDOFF_2026-09-14.md` §0.5 — the piece that
was to be built first, because every later slice needs a way to ask "did that
help?" that does not cost a submission slot. Repaired 2026-09-16 against
`docs/research/2026-09-16_validation_audit.md` (P0 wall model, P1 cost offsets,
P1 report provenance, P2 interval labelling) after #331317 graded.

WHY A PROJECTION AND NOT A MEASUREMENT
---------------------------------------
The public board is 50 hidden MLPs we cannot see. What we CAN do is run two
candidates over the same 100 public-mini networks, take the **paired ratio** of
their competition scores, and multiply the control's **known graded** public
score by it.

    projected_public(candidate) = graded_public(control) x ratio(local, paired)

It is not a score, it does not licence a submission, and it promotes nothing.
One forecast landing +0.45% from the graded row (#331317) is one data point;
it does not calibrate an interval.

THE LABELS, AND WHY EVERY LINE CARRIES ONE
------------------------------------------
    MATH               arithmetic and identities. Host-independent.
    LOCAL              what this machine measured. Real, and NOT the server.
    SERVER_MEASURED    a number the grading server itself reported.
    SERVER_PREDICTED   a forecast from a versioned model with a stated evidence
                       domain.
    EMPIRICAL_FORECAST a rule of thumb fitted to very few graded rows. Advisory,
                       family-specific, and NOT a safety proof.

THE INTERVAL — READ BEFORE QUOTING IT
-------------------------------------
The binding statistic is the paired ARITHMETIC score difference with its
studentised-t interval (`VALIDATION.md` §2b-i). The band printed next to the
projection is that difference interval **transformed** through the observed
control mean and the graded anchor. It is a *transformed local
paired-difference interval, conditional on these 100 networks*; it ignores the
randomness of the denominator, the different hidden networks, and model-transfer
error. It is NOT a 95% prediction interval for the hidden-board score.

WHAT IT REFUSES (exit 2, no number printed)
-------------------------------------------
* local reports whose dataset identity differs (`run_config.dataset.path`,
  `.sha256`, `.n_mlps`), whose MLP name sets or counts differ, or whose
  evaluator stack differs (whestbench / flopscope / numpy version, budget,
  shape, caps);
* a local report with ANY expected row missing (`len(per_mlp) != n_mlps`) or
  missing a finite `final_layer_mse` — these used to be skipped silently;
* a graded anchor that is not `completed`, has any failed or non-completed
  row, or lacks any expected public score row;
* a local control run not bound to the source that was graded: the control's
  run receipt `source_sha256` must equal the graded submission's estimator
  sha256 (from `experiments/submissions.jsonl`, verified against the package
  tarball when present). A MISMATCH is never overridable. A MISSING hash is
  refused unless `--unverified-control-source`, and the output then says
  UNVERIFIED;
* `--candidate <path>` whose bytes do not hash to the candidate receipt's
  `source_sha256` (same rule; override for a missing hash only with
  `--unverified-candidate-source`);
* arms metered on different meter paths;
* failed local MLPs, unless `--allow-failures`.

RUN RECEIPTS (what the reports lack)
------------------------------------
A `whest run --format json` report carries dataset path/sha256, MLP names,
package versions and host name — but NO source hash, NO runner/meter path and
NO host-load metadata. Those come from a sidecar receipt, schema
`arc-whest.run_receipt/1`, which binds itself to the report bytes by
`report_sha256`:

    {"schema": "arc-whest.run_receipt/1",
     "report_sha256": "<sha256 of the report file>",
     "source_path": "estimator_k3_v31.py", "source_sha256": "<64 hex>",
     "meter_path": "whest_run_subprocess" | "grader_twin_f64_readonly" | "unknown",
     "host_load": {"class": "standalone" | "concurrent" | "unknown", "evidence": "..."},
     "provenance": {"kind": "...", ...}}

Found via `--candidate-receipt/--control-receipt`, else `<report stem>.receipt.json`
beside the report. Backfilled receipts for the 2026-09-16 slice-7 minis live in
`experiments/2026-09-16_stage0/predict_public_receipts/` (source hashes taken from
the workerq execution snapshots and cross-checked against git blobs).

Run:
    uv run python local_sim/predict_public.py \\
        --candidate-run experiments/<slice>/mini_candidate.json \\
        --control-run   experiments/<slice>/mini_f21.json \\
        --control-graded experiments/run_reports/f21_diagnostic_330950_graded_report.json

CONFIRM TIER: the public mini split carries the organisers' REAL labels. This
output may CHECK a decision already made on fresh mints and may CHOOSE NOTHING.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tarfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

import competition_score as cs  # noqa: E402
import official_config as oc  # noqa: E402

MATH, LOCAL, SERVER = "MATH", "LOCAL", "SERVER_PREDICTED"
SERVER_MEASURED = "SERVER_MEASURED"
EMPIRICAL = "EMPIRICAL_FORECAST"

RECEIPT_SCHEMA = "arc-whest.run_receipt/1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# --------------------------------------------------------------------------- #
# meter paths — FLOPs are path-specific, not a server increment
# --------------------------------------------------------------------------- #

#: Counted FLOPs to ADD to a meter path's per-MLP figure to reach the grader's
#: figure. The offset is the f64 weight pass (16 x 1024**2 x 2 = 2**25), which
#: the `whest run` subprocess runner does not bill and the grader does.
METER_PATH_TO_GRADED_OFFSET: Dict[str, int] = {
    "whest_run_subprocess": 2 ** 25,
    "grader_twin_f64_readonly": 0,
}
METER_PATH_EVIDENCE: Dict[str, str] = {
    "whest_run_subprocess": (
        "graded - local subprocess = 33,554,432 exactly for F21 #330950 "
        "(928,457,637,478 - 928,424,083,046) and v31 #331317 "
        "(661,961,618,994 - 661,928,064,562); two pairs, one representation"),
    "grader_twin_f64_readonly": (
        "lifecycle/p2_flop_anchor f64_readonly already equals the server: F21 "
        "928,457,637,478, v31 661,961,618,994. Adding an offset double-counts"),
}
#: Printed when the meter path is unknown. Both ends, labelled UNVERIFIED.
UNVERIFIED_OFFSET_BAND: Tuple[int, ...] = (0, 2 ** 25)

# --------------------------------------------------------------------------- #
# wall — the x1.60-on-local-worst rule is RETIRED
# --------------------------------------------------------------------------- #

#: EMPIRICAL_FORECAST: graded wall ~= 1.9 x STANDALONE local median.
#: Evidence (LOG 2026-09-16 16:09 UTC): v31 job 1413, alone on the queue,
#: graded median / local median = 1.824, graded worst / local worst = 1.724.
#: Rounded up. Family-specific (K3 Strassen), a handful of candidates, NOT a
#: safety proof, and never a gate. `oc.GRADED_WALL_CONTENTION_FACTOR` x local
#: worst is not used here: on the same standalone run it forecast 63.25 s and
#: the server measured 68.13 s.
EMPIRICAL_WALL_MEDIAN_FACTOR: float = 1.9
WALL_MODEL_VERSION = "wall-empirical-median-x1.9 / 2026-09-16"

#: Graded residual per operation actually observed (us/op): F21 11.10,
#: v31 10.76, Sep-8 L3 10.73. The frozen 13.4 us/op model is ~20% conservative.
GRADED_RESIDUAL_US_PER_OP_OBSERVED: Tuple[float, float] = (10.73, 11.10)

#: Operation names that would indicate a device synchronisation or a scalar
#: materialisation, if a report ever carries them. Checked, never assumed.
SYNC_OR_SCALAR_OP_NAMES: Tuple[str, ...] = (
    "item", "tolist", "__float__", "__int__", "__bool__", "__index__",
    "float", "int", "bool", "synchronize", "materialize")

INTERVAL_LABEL = ("transformed local paired-difference interval "
                  "(conditional; excludes hidden-network transfer uncertainty)")


class ProjectionRefused(RuntimeError):
    """The inputs cannot support a projection. Never downgraded to a warning."""


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_run(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# local report identity and rows
# --------------------------------------------------------------------------- #


def _per_mlp(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list((report.get("results") or {}).get("per_mlp") or [])


def dataset_identity(report: Dict[str, Any]) -> Dict[str, Any]:
    """`run_config.dataset` — path, sha256, n_mlps. Refuses if unbindable."""
    ds = ((report.get("run_config") or {}).get("dataset")) or {}
    ident = {"path": ds.get("path"), "sha256": ds.get("sha256"),
             "n_mlps": ds.get("n_mlps")}
    if not ident["path"] or not ident["sha256"]:
        raise ProjectionRefused(
            "the run report carries no run_config.dataset path+sha256; two runs "
            "cannot be shown to have read the same dataset without it")
    return ident


def run_rows(report: Dict[str, Any], *, label: str = "run") -> List[cs.MlpRow]:
    """EVERY expected row of a local `whest run` report, or a refusal.

    A row without a finite `final_layer_mse` used to be skipped, which shrinks
    the paired design silently. Now the report must carry exactly
    `run_config.n_mlps` rows, each named and each with a finite MSE.
    """
    cfg = report.get("run_config") or {}
    budget = float(cfg.get("flop_budget") or 0.0)
    if budget <= 0:
        raise ProjectionRefused(
            f"{label}: the run report carries no positive flop_budget; refusing "
            "to guess one")
    per = _per_mlp(report)
    if not per:
        raise ProjectionRefused(f"{label}: the run report contains no rows")
    expected = [cfg.get("n_mlps"), ((cfg.get("dataset") or {}).get("n_mlps"))]
    expected = [int(x) for x in expected if x is not None]
    if not expected:
        raise ProjectionRefused(
            f"{label}: no run_config.n_mlps; cannot tell a complete report from "
            "a truncated one")
    if any(e != len(per) for e in expected):
        raise ProjectionRefused(
            f"{label}: TRUNCATED -- {len(per)} per_mlp rows, expected "
            f"{sorted(set(expected))}")
    missing = [str(e.get("mlp_name") or e.get("mlp_index")) for e in per
               if e.get("final_layer_mse") is None
               or not math.isfinite(float(e.get("final_layer_mse")))]
    if missing:
        raise ProjectionRefused(
            f"{label}: {len(missing)} expected row(s) have no finite "
            f"final_layer_mse (e.g. {missing[:5]}); a missing row is not skipped")
    out: List[cs.MlpRow] = []
    for e in per:
        out.append(cs.MlpRow(
            mse=float(e["final_layer_mse"]),
            flops_used=float(e.get("flops_used") or 0.0),
            residual_wall_time_s=float(e.get("residual_wall_time_s") or 0.0),
            budget=budget,
            failed=cs.failed_from_report_entry(e),
            name=str(e.get("mlp_name") or e.get("mlp_index", "")),
        ))
    return out


#: Keys that must agree between the two arms, and where they live.
_RUNTIME_KEYS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("whestbench_version", ("whestbench_version",)),
    ("flopscope_version", ("run_meta", "host", "flopscope_version")),
    ("numpy_version", ("run_meta", "host", "numpy_version")),
    ("flop_budget", ("run_config", "flop_budget")),
    ("width", ("run_config", "width")),
    ("depth", ("run_config", "depth")),
    ("wall_time_limit_s", ("run_config", "wall_time_limit_s")),
    ("residual_wall_time_limit_s", ("run_config", "residual_wall_time_limit_s")),
    ("lambda_flops_per_second", ("run_config", "lambda_flops_per_second")),
)


def _dig(d: Dict[str, Any], path: Sequence[str]) -> Any:
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def check_pair_identity(cand: Dict[str, Any], ctrl: Dict[str, Any]) -> List[str]:
    """Same dataset, same networks (set AND count), same evaluator stack."""
    notes: List[str] = []
    dc, dk = dataset_identity(cand), dataset_identity(ctrl)
    if dc != dk:
        raise ProjectionRefused(
            f"the arms read different datasets: candidate {dc} vs control {dk}. "
            "Colliding MLP names are not the same networks")
    names_c = [str(e.get("mlp_name") or "") for e in _per_mlp(cand)]
    names_k = [str(e.get("mlp_name") or "") for e in _per_mlp(ctrl)]
    for lab, names in (("candidate", names_c), ("control", names_k)):
        if any(not n for n in names):
            raise ProjectionRefused(f"{lab}: a row has no mlp_name")
        if len(set(names)) != len(names):
            raise ProjectionRefused(f"{lab}: duplicate mlp_name rows")
    if len(names_c) != len(names_k) or set(names_c) != set(names_k):
        only_c = sorted(set(names_c) - set(names_k))[:3]
        only_k = sorted(set(names_k) - set(names_c))[:3]
        raise ProjectionRefused(
            f"the arms are not the same MLPs: counts {len(names_c)} vs "
            f"{len(names_k)}; candidate-only {only_c}, control-only {only_k}")
    for key, path in _RUNTIME_KEYS:
        a, b = _dig(cand, path), _dig(ctrl, path)
        if a != b:
            raise ProjectionRefused(
                f"the arms ran different evaluator configurations: {key} "
                f"{a!r} (candidate) vs {b!r} (control)")
    for key in ("hostname", "python_version"):
        a = _dig(cand, ("run_meta", "host", key))
        b = _dig(ctrl, ("run_meta", "host", key))
        if a != b:
            notes.append(f"arms differ in {key}: {a!r} vs {b!r} (timing only)")
    return notes


# --------------------------------------------------------------------------- #
# receipts and source binding
# --------------------------------------------------------------------------- #


def find_receipt(report_path: str, explicit: str = "") -> Optional[str]:
    if explicit:
        if not os.path.exists(explicit):
            raise ProjectionRefused(f"receipt not found: {explicit}")
        return explicit
    # Canonical: <report stem>.receipt.json, which scripts/run_mini_json.ps1
    # writes since 2026-09-17. Before that it wrote <report>.receipt.json and this
    # never found it (validation review 09-17 §2); that legacy name is still
    # accepted, and two receipts that disagree are refused rather than chosen.
    canonical = os.path.splitext(report_path)[0] + ".receipt.json"
    legacy = report_path + ".receipt.json"
    found = [p for p in (canonical, legacy) if os.path.exists(p)]
    if len(found) == 2 and _sha256(canonical) != _sha256(legacy):
        raise ProjectionRefused(
            f"two different receipts for {report_path}: {canonical} and "
            f"{legacy}. Remove the wrong one or pass the receipt explicitly")
    return found[0] if found else None


def load_receipt(receipt_path: Optional[str], report_path: str,
                 *, label: str) -> Optional[Dict[str, Any]]:
    """Validate a sidecar receipt against the report bytes it claims."""
    if not receipt_path:
        return None
    with open(receipt_path, encoding="utf-8") as fh:
        rec = json.load(fh)
    if rec.get("schema") != RECEIPT_SCHEMA:
        raise ProjectionRefused(
            f"{label} receipt {receipt_path}: schema {rec.get('schema')!r}, "
            f"expected {RECEIPT_SCHEMA!r}")
    actual = _sha256(report_path)
    if rec.get("report_sha256") != actual:
        raise ProjectionRefused(
            f"{label} receipt {receipt_path} is for different report bytes: "
            f"receipt {str(rec.get('report_sha256'))[:16]}... vs report "
            f"{actual[:16]}...")
    src = rec.get("source_sha256")
    if src is not None and not _HEX64.match(str(src)):
        raise ProjectionRefused(
            f"{label} receipt: source_sha256 {src!r} is not a sha256")
    mp = rec.get("meter_path", "unknown")
    if mp not in METER_PATH_TO_GRADED_OFFSET and mp != "unknown":
        raise ProjectionRefused(
            f"{label} receipt: meter_path {mp!r} is not a known path "
            f"({sorted(METER_PATH_TO_GRADED_OFFSET)} or 'unknown')")
    rec = dict(rec)
    rec["_path"] = receipt_path
    return rec


def graded_source_sha256(graded: Dict[str, Any], ledger_path: str
                         ) -> Tuple[Optional[str], str]:
    """The estimator sha256 the grader actually ran, from the submissions ledger.

    Sources, in order, and they must agree: the row's `inner_estimator_sha256`;
    `estimator.py` inside the row's `artifact` tarball, when that tarball exists
    and its sha256 equals the row's `sha256`. Returns (sha or None, evidence).
    """
    sid = str(_graded_submission_id(graded) or "")
    if not sid:
        return None, "graded report carries no submission id"
    if not ledger_path or not os.path.exists(ledger_path):
        return None, f"submissions ledger not found: {ledger_path}"
    rows = []
    with open(ledger_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if str(r.get("submission_id")) == sid:
                rows.append(r)
    if not rows:
        return None, f"no ledger row for submission {sid}"
    found: Dict[str, str] = {}
    for r in rows:
        inner = r.get("inner_estimator_sha256")
        if inner:
            found.setdefault(inner, f"ledger inner_estimator_sha256 (#{sid})")
        art = r.get("artifact")
        if art:
            ap = art if os.path.isabs(art) else os.path.join(_ROOT, art)
            if os.path.exists(ap) and _sha256(ap) == r.get("sha256"):
                with tarfile.open(ap, "r:*") as tf:
                    try:
                        m = tf.getmember("estimator.py")
                    except KeyError:
                        m = None
                    if m is not None:
                        fh2 = tf.extractfile(m)
                        if fh2 is not None:
                            h = _sha256_bytes(fh2.read())
                            found.setdefault(
                                h, f"estimator.py inside {art} (tarball sha256 "
                                   f"matches ledger)")
    if len(found) > 1:
        raise ProjectionRefused(
            f"submission {sid}: the ledger and its package disagree on the "
            f"estimator sha256: {sorted(found)}")
    if not found:
        return None, (f"ledger row for {sid} has no inner_estimator_sha256 and "
                      "no verifiable package tarball")
    (h, ev), = found.items()
    return h, ev


def bind_control_source(ctrl_receipt: Optional[Dict[str, Any]],
                        graded_sha: Optional[str], graded_evidence: str,
                        *, allow_unverified: bool) -> Dict[str, Any]:
    local_sha = (ctrl_receipt or {}).get("source_sha256")
    if local_sha and graded_sha:
        if local_sha != graded_sha:
            raise ProjectionRefused(
                "WRONG CONTROL SOURCE: the local control run was produced by "
                f"{local_sha[:16]}... but the graded anchor ran {graded_sha[:16]}... "
                f"({graded_evidence}). Not overridable")
        return {"status": "VERIFIED", "sha256": local_sha,
                "evidence": f"control receipt == {graded_evidence}"}
    why = []
    if not local_sha:
        why.append("no control run receipt with source_sha256")
    if not graded_sha:
        why.append(f"graded source hash unavailable: {graded_evidence}")
    if not allow_unverified:
        raise ProjectionRefused(
            "control source provenance cannot be verified (" + "; ".join(why)
            + "). Pass --unverified-control-source to proceed with the "
            "projection labelled UNVERIFIED")
    return {"status": "UNVERIFIED", "sha256": local_sha or graded_sha,
            "evidence": "; ".join(why)}


def bind_candidate_source(cand_receipt: Optional[Dict[str, Any]],
                          candidate_path: Optional[str],
                          *, allow_unverified: bool) -> Dict[str, Any]:
    rec_sha = (cand_receipt or {}).get("source_sha256")
    if not candidate_path:
        return {"status": "RECEIPT_ONLY" if rec_sha else "NOT_REQUESTED",
                "sha256": rec_sha, "file_sha256": None,
                "evidence": "no --candidate path; memory lookup skipped"}
    if not os.path.exists(candidate_path):
        raise ProjectionRefused(f"--candidate not found: {candidate_path}")
    file_sha = _sha256(candidate_path)
    if rec_sha:
        if rec_sha != file_sha:
            raise ProjectionRefused(
                f"WRONG CANDIDATE SOURCE: {candidate_path} hashes to "
                f"{file_sha[:16]}... but the candidate run was produced by "
                f"{rec_sha[:16]}.... Not overridable")
        return {"status": "VERIFIED", "sha256": file_sha, "file_sha256": file_sha,
                "evidence": "--candidate bytes == candidate receipt source_sha256"}
    if not allow_unverified:
        raise ProjectionRefused(
            "candidate source provenance cannot be verified: no candidate run "
            "receipt with source_sha256, so the memory lookup for "
            f"{candidate_path} would not be tied to the run. Pass "
            "--unverified-candidate-source to proceed labelled UNVERIFIED")
    return {"status": "UNVERIFIED", "sha256": file_sha, "file_sha256": file_sha,
            "evidence": "no candidate receipt; file hash is not tied to the run"}


def resolve_meter_paths(cand_receipt, ctrl_receipt, cli: str = "") -> str:
    got = {lab: (r or {}).get("meter_path", "unknown")
           for lab, r in (("candidate", cand_receipt), ("control", ctrl_receipt))}
    if cli:
        if cli not in METER_PATH_TO_GRADED_OFFSET:
            raise ProjectionRefused(f"--meter-path {cli!r} is not a known path")
        for lab, mp in got.items():
            if mp != "unknown" and mp != cli:
                raise ProjectionRefused(
                    f"--meter-path {cli} contradicts the {lab} receipt ({mp})")
        return cli
    known = {mp for mp in got.values() if mp != "unknown"}
    if len(known) > 1:
        raise ProjectionRefused(
            f"the arms were metered on different paths {got}; their FLOPs are "
            "not comparable")
    if got["candidate"] == "unknown":
        return "unknown"
    return got["candidate"]


# --------------------------------------------------------------------------- #
# graded anchor
# --------------------------------------------------------------------------- #


def _graded_submission_id(graded: Dict[str, Any]) -> Any:
    return (graded.get("submission_id")
            or (graded.get("evaluation") or {}).get("submission_id")
            or graded.get("id"))


def graded_anchor(graded: Dict[str, Any]) -> Dict[str, Any]:
    """The control's OWN graded public score, replayed — only from a clean report.

    Refuses: evaluation not `completed`; row count != config n_mlps; any failed
    or non-completed row; `progress.mlps_failed` or aggregate `n_failed_mlps`
    nonzero; public score rows != n_mlps // 2 (the published public half), or a
    public row without a matching per_mlp row or without `final_layer_mse`.
    """
    ev = graded.get("evaluation", graded)
    state = ev.get("state")
    if state is not None and state != "completed":
        raise ProjectionRefused(f"graded anchor: evaluation state {state!r}")
    res = ev.get("results") or {}
    per = res.get("per_mlp") or []
    n_cfg = (ev.get("config_snapshot") or {}).get("n_mlps")
    if n_cfg is None or int(n_cfg) != len(per):
        raise ProjectionRefused(
            f"graded anchor: {len(per)} per_mlp rows vs config n_mlps {n_cfg}")
    bad = [str(e.get("mlp_name") or e.get("mlp_index")) for e in per
           if cs.failed_from_report_entry(e)
           or (e.get("state") not in (None, "completed"))]
    prog_failed = int(((ev.get("progress") or {}).get("mlps_failed")) or 0)
    agg_failed = int((((res.get("aggregates") or {}).get("public") or {})
                      .get("n_failed_mlps")) or 0)
    if bad or prog_failed or agg_failed:
        raise ProjectionRefused(
            f"graded anchor has failed MLPs ({len(bad)} rows e.g. {bad[:3]}; "
            f"progress.mlps_failed {prog_failed}; aggregate {agg_failed}). A "
            "failed anchor row is a zero-prediction score, not the control's")
    idx = {int(e.get("mlp_index", -1)) for e in per}
    ps = res.get("public_scores") or []
    expected_public = int(n_cfg) // 2
    ps_idx = [int(r.get("mlp_index", -1)) for r in ps]
    no_mse = [i for i, r in zip(ps_idx, ps)
              if (r.get("scores") or {}).get("final_layer_mse") is None]
    if (len(ps) != expected_public or len(set(ps_idx)) != len(ps_idx)
            or not set(ps_idx) <= idx or no_mse):
        raise ProjectionRefused(
            f"graded anchor: {len(ps)} public score rows (expected "
            f"{expected_public}, unique, each matching a per_mlp row with an "
            f"MSE); rows without MSE {no_mse[:5]}")
    rows = cs.rows_from_graded_report(graded)
    if len(rows) != expected_public:
        raise ProjectionRefused(
            f"graded anchor replayed {len(rows)} rows, expected {expected_public}")
    suite = cs.score_suite(rows)
    walls = [(str(e.get("mlp_name")), float((e.get("telemetry") or {})
              .get("participant_predict_wall_s") or float("nan"))) for e in per]
    resid = [(str(e.get("mlp_name")), float((e.get("telemetry") or {})
              .get("residual_wall_time_s") or float("nan"))) for e in per]
    flops = [(str(e.get("mlp_name")), float((e.get("telemetry") or {})
              .get("flops_used") or 0.0)) for e in per]
    return {"adjusted": float(suite.objective), "raw_mse": float(suite.mean_mse),
            "multiplier": float(suite.mean_multiplier), "n": int(suite.n),
            "n_failed": int(suite.n_failed), "n_rows": len(per),
            "submission_id": str(_graded_submission_id(graded)),
            "wall": _named_summary(walls), "residual": _named_summary(resid),
            "flops": _named_summary(flops),
            "config": ev.get("config_snapshot") or {}}


# --------------------------------------------------------------------------- #
# per-network summaries
# --------------------------------------------------------------------------- #


def _named_summary(pairs: Sequence[Tuple[str, float]]) -> Optional[Dict[str, Any]]:
    vals = [(n, v) for n, v in pairs if v is not None and math.isfinite(v)]
    if not vals:
        return None
    xs = [v for _, v in vals]
    worst = max(vals, key=lambda t: t[1])
    return {"n": len(xs), "mean": float(statistics.fmean(xs)),
            "median": float(statistics.median(xs)), "max": float(worst[1]),
            "max_network": worst[0], "min": float(min(xs))}


def per_network_ops(report: Dict[str, Any]) -> List[Tuple[str, int]]:
    """Per-network estimator operation counts from per-MLP breakdowns."""
    out = []
    for e in _per_mlp(report):
        by_ns = (((e.get("breakdowns") or {}).get("estimator") or {})
                 .get("by_namespace")) or {}
        calls = sum(int(d.get("calls") or 0) for ns, d in by_ns.items()
                    if "estimator" in ns)
        if by_ns:
            out.append((str(e.get("mlp_name") or e.get("mlp_index")), calls))
    return out


def run_ops_per_mlp(report: Dict[str, Any]) -> Optional[float]:
    """MEAN metered operations per MLP from the aggregate breakdown (legacy)."""
    res = report.get("results") or {}
    n = len((res.get("per_mlp") or [])) or int(
        (report.get("run_config") or {}).get("n_mlps") or 0)
    by_ns = ((res.get("breakdowns") or {}).get("estimator") or {}).get("by_namespace") or {}
    calls = sum(int(d.get("calls") or 0) for ns, d in by_ns.items() if "estimator" in ns)
    if not calls or not n:
        return None
    return calls / float(n)


def sync_scalar_counts(report: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronisation / scalar-materialisation counts, if the report has any."""
    have_ops = False
    counts: Dict[str, int] = {}
    per_net_max: Dict[str, Tuple[str, int]] = {}
    for e in _per_mlp(report):
        by_ns = (((e.get("breakdowns") or {}).get("estimator") or {})
                 .get("by_namespace")) or {}
        name = str(e.get("mlp_name") or e.get("mlp_index"))
        for d in by_ns.values():
            ops = d.get("operations")
            if ops is None:
                continue
            have_ops = True
            for op, od in ops.items():
                if op in SYNC_OR_SCALAR_OP_NAMES:
                    c = int(od.get("calls") or 0)
                    counts[op] = counts.get(op, 0) + c
                    if c > per_net_max.get(op, ("", -1))[1]:
                        per_net_max[op] = (name, c)
    if not have_ops:
        return {"available": False,
                "why": "the report carries no per-operation breakdown, so "
                       "synchronisation/scalar-materialisation counts are ABSENT"}
    if not counts:
        return {"available": False,
                "why": "per-operation breakdown present but it contains no "
                       "synchronisation/scalar-materialisation counter "
                       f"(checked {', '.join(SYNC_OR_SCALAR_OP_NAMES)}); ABSENT, "
                       "not zero"}
    return {"available": True, "total_calls": counts,
            "max_per_network": {k: {"network": v[0], "calls": v[1]}
                                for k, v in per_net_max.items()}}


def network_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    per = _per_mlp(report)
    nm = lambda e: str(e.get("mlp_name") or e.get("mlp_index"))  # noqa: E731
    flops = _named_summary([(nm(e), float(e.get("flops_used") or 0.0)) for e in per])
    wall = _named_summary([(nm(e), float(e["wall_time_s"])) for e in per
                           if e.get("wall_time_s") is not None])
    resid = _named_summary([(nm(e), float(e["residual_wall_time_s"])) for e in per
                            if e.get("residual_wall_time_s") is not None])
    ops_pairs = per_network_ops(report)
    ops = _named_summary([(n, float(c)) for n, c in ops_pairs]) if ops_pairs else None
    return {"flops": flops, "wall_s": wall, "residual_s_local": resid,
            "ops": ops, "ops_mean_aggregate": run_ops_per_mlp(report),
            "sync_scalar": sync_scalar_counts(report)}


# --------------------------------------------------------------------------- #
# the projection
# --------------------------------------------------------------------------- #


def project(candidate_rows, control_rows, anchor, *, gate: Optional[float] = None,
            allow_failures: bool = False, flop_offset: Optional[int] = None):
    """The paired difference, its t interval, and the projected public score."""
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
    # The t interval on the DIFFERENCE is the binding statistic. The band below
    # is that interval pushed through the OBSERVED control mean and the anchor:
    # a transformation, conditional on these networks, not a ratio CI and not a
    # prediction interval for the hidden board.
    band_lo = (mean_ctrl - hi) / mean_ctrl if mean_ctrl else float("nan")
    band_hi = (mean_ctrl - lo) / mean_ctrl if mean_ctrl else float("nan")

    out = {
        "n": int(paired.n),
        "local_control_adjusted": mean_ctrl,
        "local_candidate_adjusted": float(cand.objective),
        "local_control_raw_mse": float(ctrl.mean_mse),
        "local_candidate_raw_mse": float(cand.mean_mse),
        "local_control_multiplier": float(ctrl.mean_multiplier),
        "local_candidate_multiplier": float(cand.mean_multiplier),
        "ratio": ratio,
        "mean_difference": float(paired.mean_difference),
        "t_ci_difference": tuple(float(x) for x in paired.t_ci_difference),
        "interval_label": INTERVAL_LABEL,
        "interval_is_hidden_board_prediction_interval": False,
        "transformed_band_ratio": (band_lo, band_hi),
        "arithmetic_gain": float(paired.arithmetic_gain),
        "geometric_gain_DIAGNOSTIC": float(paired.geometric_gain),
        "n_candidate_better": int(paired.n_candidate_better),
        "sign_test_p": float(paired.sign_test_p),
        "anchor_graded_adjusted": float(anchor["adjusted"]),
        "projected_public_adjusted": float(anchor["adjusted"] * ratio),
        "projected_public_transformed_band": (float(anchor["adjusted"] * band_lo),
                                              float(anchor["adjusted"] * band_hi)),
        "notes": list(paired.notes) + notes,
        "n_failed_control": int(ctrl.n_failed),
        "n_failed_candidate": int(cand.n_failed),
    }
    if flop_offset:
        # Both arms moved to the grader's FLOP figure. A cross-check on the
        # as-metered ratio, not a replacement for it.
        shift = lambda rows: [dataclasses.replace(r, flops_used=r.flops_used  # noqa: E731
                                                  + flop_offset) for r in rows]
        cg, kg = cs.score_suite(shift(c_rows)), cs.score_suite(shift(k_rows))
        rg = float(kg.objective / cg.objective) if cg.objective else float("nan")
        out["ratio_at_graded_flops"] = rg
        out["projected_public_at_graded_flops"] = float(anchor["adjusted"] * rg)
    if gate is not None:
        v = cs.decide(ctrl.scores, cand.scores, gate=gate)
        out["verdict"] = {"passed": bool(v.passed), "gain": float(v.gain),
                          "gate": float(v.gate),
                          "ci_excludes_zero": bool(v.ci_excludes_zero),
                          "reasons": list(v.reasons)}
    return out


def graded_flops(flops_per_mlp: float, budget: float, meter_path: str
                 ) -> Dict[str, Any]:
    """Meter-path FLOPs -> the grader's FLOPs, by path, never by guesswork."""
    def entry(off):
        f = flops_per_mlp + off
        return {"offset": off, "flops": f, "c_over_b": f / budget,
                "multiplier": max(cs.MULTIPLIER_FLOOR, f / budget)}
    if meter_path in METER_PATH_TO_GRADED_OFFSET:
        return {"meter_path": meter_path, "status": "VERIFIED",
                "evidence": METER_PATH_EVIDENCE[meter_path],
                "entries": [entry(METER_PATH_TO_GRADED_OFFSET[meter_path])]}
    return {"meter_path": meter_path or "unknown", "status": "UNVERIFIED",
            "evidence": "meter path unknown: no receipt and no --meter-path; "
                        "both documented offsets shown",
            "entries": [entry(o) for o in UNVERIFIED_OFFSET_BAND]}


def residual_forecast(ops_max: float, *, ops_mean: Optional[float] = None,
                      max_network: Optional[str] = None) -> Dict[str, Any]:
    """Operation count -> predicted SERVER residual. Gated on the MAX network."""
    lo_s, hi_s = oc.GRADED_RESIDUAL_S_PER_OP_RANGE
    cap = 0.4
    return {
        "ops_max": ops_max,
        "ops_max_network": max_network,
        "ops_mean": ops_mean,
        "gate_ops": int(oc.PHASE2_OP_COUNT_GATE),
        "gate_basis": "per-network MAX operation count",
        "gate_passed": ops_max <= oc.PHASE2_OP_COUNT_GATE,
        "s_per_op_point": oc.GRADED_RESIDUAL_S_PER_OP,
        "s_per_op_range": (lo_s, hi_s),
        "graded_us_per_op_observed": GRADED_RESIDUAL_US_PER_OP_OBSERVED,
        "model_note": ("frozen 13.4 us/op is ~20% conservative against graded "
                       "rows (10.73-11.10 us/op); kept frozen deliberately"),
        "predicted_s_point": ops_max * oc.GRADED_RESIDUAL_S_PER_OP,
        "predicted_s_range": (ops_max * lo_s, ops_max * hi_s),
        "cap_s": cap,
        "headroom_point": cap / (ops_max * oc.GRADED_RESIDUAL_S_PER_OP)
                          if ops_max else float("inf"),
    }


def wall_forecast(wall_summary: Optional[Dict[str, Any]],
                  host_load: Optional[Dict[str, Any]] = None,
                  host_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Local wall facts plus the EMPIRICAL 1.9 x standalone-median forecast.

    Never a gate. The retired rule (x1.60 on local worst) is not computed.
    """
    if not wall_summary:
        return {"available": False, "why": "no per-MLP wall times in the run report"}
    load_class = (host_load or {}).get("class", "unknown")
    in_domain = load_class == "standalone"
    fc = EMPIRICAL_WALL_MEDIAN_FACTOR * wall_summary["median"]
    cap = float(oc.PHASE2.wall_time_limit_s)
    return {
        "available": True,
        "local_median_s": wall_summary["median"],
        "local_mean_s": wall_summary["mean"],
        "local_worst_s": wall_summary["max"],
        "local_worst_network": wall_summary["max_network"],
        "host_load": host_load or {"class": "unknown",
                                   "evidence": "no receipt; the report carries "
                                               "hostname/cpu count only"},
        "host_meta": {k: v for k, v in (host_meta or {}).items() if v is not None},
        "model_version": WALL_MODEL_VERSION,
        "label": EMPIRICAL,
        "empirical_factor": EMPIRICAL_WALL_MEDIAN_FACTOR,
        "empirical_forecast_s": fc,
        "in_evidence_domain": in_domain,
        "advisory": True,
        "is_gate": False,
        "cap_s": cap,
        "advisory_margin": cap / fc if fc else float("inf"),
        "caveat": ("family-specific (K3 Strassen), fitted to very few graded "
                   "rows, NOT a safety proof. A new arithmetic family, Strassen "
                   "depth or allocation/synchronisation pattern needs a canary"),
    }


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
            stack=cc._certified_stack(),
            # REQUIRED: `consume_memory_report` binds the instrument sha as
            # `bool(expected) and expected == got`; omitting it is a FAILED
            # binding, not a skipped one (2026-09-16).
            expected_instrument_sha256=cc._memory_instrument_sha256())
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


def _line(tag, text, value=""):
    print(f"  [{tag:<18}] {text:<52}{value}")


def _max_line(tag, label, s, fmt):
    if not s:
        _line(tag, label, "UNAVAILABLE")
        return
    _line(tag, label, f"mean {format(s['mean'], fmt)}   max {format(s['max'], fmt)}"
                      f" ({s['max_network']})")


def report(res, cost, resid, wall, mem, meta, nets, anchor):
    print("=" * 100)
    print("PROJECTED PUBLIC SCORE -- an estimate, not a score")
    print("=" * 100)
    print(f"  candidate   {meta.get('candidate', '(unnamed)')}")
    print(f"  control     {meta.get('control', '(unnamed)')}   "
          f"graded submission {meta.get('control_submission', '?')}")
    print(f"  networks    {res['n']} public-mini MLPs, paired on mlp_name; dataset "
          f"{meta['dataset']['path']} sha256 {str(meta['dataset']['sha256'])[:16]}...")
    print()
    print("  *** THE PUBLIC MINI SPLIT CARRIES THE ORGANISERS' REAL LABELS.")
    print("  *** This is a CONFIRM-tier read. It may CHECK a decision made on")
    print("  *** fresh mints and it may CHOOSE NOTHING.")
    print()

    print("-- provenance " + "-" * 85)
    cs_ = meta["control_source"]
    _line(MATH, "control run source == graded source", cs_["status"])
    print(f"       {cs_['evidence']}")
    ks = meta["candidate_source"]
    _line(MATH, "candidate source binding", ks["status"])
    print(f"       {ks['evidence']}")
    _line(MATH, "anchor", f"{anchor['n_rows']} rows completed, 0 failed, "
                          f"{anchor['n']} public scores")
    _line(MATH, "meter path", f"{cost['meter_path']} ({cost['status']})")
    print()

    print("-- the paired local measurement " + "-" * 67)
    _line(LOCAL, "control adjusted", f"{res['local_control_adjusted']:.6e}")
    _line(LOCAL, "candidate adjusted", f"{res['local_candidate_adjusted']:.6e}")
    _line(LOCAL, "control raw MSE", f"{res['local_control_raw_mse']:.6e}")
    _line(LOCAL, "candidate raw MSE", f"{res['local_candidate_raw_mse']:.6e}")
    _line(LOCAL, "control multiplier", f"{res['local_control_multiplier']:.6f}")
    _line(LOCAL, "candidate multiplier", f"{res['local_candidate_multiplier']:.6f}")
    print()
    _line(MATH, "paired arithmetic score difference  (BINDING)",
          f"{res['mean_difference']:+.6e}")
    _line(MATH, "  studentised-t 95% CI on that difference",
          f"[{res['t_ci_difference'][0]:+.4e}, {res['t_ci_difference'][1]:+.4e}]")
    _line(MATH, "ratio candidate/control", f"{res['ratio']:.6f}")
    _line(MATH, "  transformed difference band (ratio scale)",
          f"[{res['transformed_band_ratio'][0]:.6f}, "
          f"{res['transformed_band_ratio'][1]:.6f}]")
    _line(MATH, "arithmetic gain (control/candidate)",
          f"{res['arithmetic_gain']:.4f}x")
    _line(MATH, "candidate better on",
          f"{res['n_candidate_better']}/{res['n']}   sign p = "
          f"{res['sign_test_p']:.4f}")
    _line(MATH, "geometric gain  DIAGNOSTIC ONLY, no authority",
          f"{res['geometric_gain_DIAGNOSTIC']:.4f}x")
    print()

    print("-- the projection " + "-" * 81)
    _line(SERVER_MEASURED, f"control's graded public score #{anchor['submission_id']}",
          f"{res['anchor_graded_adjusted']:.6e}  (replayed)")
    _line(SERVER, "projected public adjusted",
          f"{res['projected_public_adjusted']:.6e}")
    if "projected_public_at_graded_flops" in res:
        _line(MATH, "  same, both arms at graded FLOPs",
              f"{res['projected_public_at_graded_flops']:.6e}")
    _line(SERVER, "  transformed band",
          f"[{res['projected_public_transformed_band'][0]:.4e}, "
          f"{res['projected_public_transformed_band'][1]:.4e}]")
    print(f"       That band is a {INTERVAL_LABEL}.")
    print("       It is NOT a 95% prediction interval for the hidden-board score.")
    print()

    print("-- cost: FLOPs by meter path " + "-" * 70)
    _max_line(LOCAL, "counted FLOPs per network", nets["flops"], ",.0f")
    for m in cost["entries"]:
        _line(SERVER if cost["status"] == "VERIFIED" else "UNVERIFIED",
              f"mean F + {m['offset']:>10,} -> C/B",
              f"{m['c_over_b']:.6f}   multiplier {m['multiplier']:.6f}")
    print(f"       {cost['status']}: {cost['evidence']}")
    print()

    print("-- residual, priced on OPERATIONS (gate on the per-network MAX) " + "-" * 35)
    ops = nets["ops"]
    if resid is None:
        _line(LOCAL, "operation count", "UNAVAILABLE -- no per-network call counts")
        if nets["ops_mean_aggregate"]:
            _line(LOCAL, "  aggregate mean only (NOT gated)",
                  f"{nets['ops_mean_aggregate']:,.1f}")
        print("       Without per-network counts there is no residual gate.")
    else:
        _max_line(LOCAL, "metered operations per network", ops, ",.0f")
        _line(MATH, f"  max against the {resid['gate_ops']:,}-operation gate",
              "PASS" if resid["gate_passed"] else "*** FAIL ***")
        _line(SERVER, "predicted server residual, max network",
              f"{resid['predicted_s_point']:.4f} s   "
              f"[{resid['predicted_s_range'][0]:.4f}, "
              f"{resid['predicted_s_range'][1]:.4f}]")
        _line(SERVER, f"  headroom under the {resid['cap_s']} s cliff",
              f"{resid['headroom_point']:.2f}x")
        print(f"       {resid['model_note']}.")
    _max_line(LOCAL, "local residual s (advisory, NOT server s)",
              nets["residual_s_local"], ".3f")
    ss = nets["sync_scalar"]
    if ss["available"]:
        _line(LOCAL, "sync/scalar materialisations", json.dumps(ss["total_calls"]))
    else:
        _line(LOCAL, "sync/scalar materialisations", "ABSENT")
        print(f"       {ss['why']}")
    if anchor.get("residual"):
        a = anchor["residual"]
        _line(SERVER_MEASURED, "control graded residual",
              f"median {a['median']:.4f} s   max {a['max']:.4f} s ({a['max_network']})")
        if anchor.get("control_ops_max"):
            _line(MATH, "  graded residual per op, control max/max",
                  f"{a['max'] / anchor['control_ops_max'] * 1e6:.2f} us/op at "
                  f"{anchor['control_ops_max']:,.0f} ops")
    print()

    print("-- wall (advisory; no gate) " + "-" * 71)
    if wall.get("available"):
        _line(LOCAL, "local median / mean",
              f"{wall['local_median_s']:.3f} s / {wall['local_mean_s']:.3f} s")
        _line(LOCAL, "local worst", f"{wall['local_worst_s']:.3f} s "
                                    f"({wall['local_worst_network']})")
        hl = wall["host_load"]
        n_ov = len(hl.get("overlapping_queue_jobs") or [])
        _line(LOCAL, "host load", f"{hl.get('class', 'unknown')}"
              + (f"   ({n_ov} overlapping queue jobs)" if "overlapping_queue_jobs" in hl else ""))
        print(f"       {hl.get('evidence', '')}")
        _line(EMPIRICAL, f"{wall['empirical_factor']} x local median",
              f"{wall['empirical_forecast_s']:.2f} s of {wall['cap_s']:.0f} s  "
              f"(advisory margin {wall['advisory_margin']:.2f}x)")
        if not wall["in_evidence_domain"]:
            print("       OUT OF EVIDENCE DOMAIN: the factor was fitted to a "
                  "STANDALONE run;")
            print(f"       this run's host load is '{hl.get('class')}'.")
        print(f"       {wall['model_version']}: {wall['caveat']}.")
    else:
        _line(LOCAL, "wall", f"UNAVAILABLE -- {wall.get('why', '')}")
    if anchor.get("wall"):
        a = anchor["wall"]
        _line(SERVER_MEASURED, "control graded wall",
              f"median {a['median']:.2f} s   max {a['max']:.2f} s ({a['max_network']})")
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
        print("       A skipped memory gate is NOT a pass.")
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


def build(a) -> Dict[str, Any]:
    """Everything except printing. Raises ProjectionRefused / PairingError."""
    cand_run = load_run(a.candidate_run)
    ctrl_run = load_run(a.control_run)
    graded = load_run(a.control_graded)

    notes = check_pair_identity(cand_run, ctrl_run)
    cand_rows = run_rows(cand_run, label="candidate")
    ctrl_rows = run_rows(ctrl_run, label="control")
    anchor = graded_anchor(graded)

    cfg = cand_run.get("run_config") or {}
    budget = float(cfg.get("flop_budget") or 0.0)
    width = int(cfg.get("width") or 1024)
    depth = int(cfg.get("depth") or 16)
    gcfg = anchor["config"]
    for key, local in (("flop_budget", budget), ("width", width), ("depth", depth)):
        if gcfg.get(key) is not None and float(gcfg[key]) != float(local):
            raise ProjectionRefused(
                f"graded anchor {key} {gcfg[key]} != local {local}")

    cand_rec = load_receipt(find_receipt(a.candidate_run, a.candidate_receipt),
                            a.candidate_run, label="candidate")
    ctrl_rec = load_receipt(find_receipt(a.control_run, a.control_receipt),
                            a.control_run, label="control")
    g_sha, g_ev = graded_source_sha256(graded, a.submissions_ledger)
    ctrl_src = bind_control_source(ctrl_rec, g_sha, g_ev,
                                   allow_unverified=a.unverified_control_source)
    cand_src = bind_candidate_source(cand_rec, a.candidate or None,
                                     allow_unverified=a.unverified_candidate_source)
    meter = resolve_meter_paths(cand_rec, ctrl_rec, a.meter_path)
    offset = METER_PATH_TO_GRADED_OFFSET.get(meter)

    res = project(cand_rows, ctrl_rows, anchor, gate=a.gate,
                  allow_failures=a.allow_failures, flop_offset=offset)
    res["notes"] = res["notes"] + notes

    nets = network_summary(cand_run)
    ctrl_ops = network_summary(ctrl_run)["ops"]
    anchor["control_ops_max"] = ctrl_ops["max"] if ctrl_ops else None
    cost = graded_flops(nets["flops"]["mean"], budget, meter)
    cost["max_network_flops"] = nets["flops"]["max"]
    cost["max_network"] = nets["flops"]["max_network"]
    ops = nets["ops"]
    resid = (residual_forecast(ops["max"], ops_mean=ops["mean"],
                               max_network=ops["max_network"]) if ops else None)
    wall = wall_forecast(nets["wall_s"], (cand_rec or {}).get("host_load"),
                         (cand_run.get("run_meta") or {}).get("host"))
    mem = memory_verdict(a.candidate or None, cand_src.get("file_sha256"),
                         width, depth, budget)

    meta = {"candidate": a.candidate or os.path.basename(a.candidate_run),
            "control": os.path.basename(a.control_run),
            "control_submission": anchor["submission_id"],
            "candidate_source": cand_src, "control_source": ctrl_src,
            "candidate_receipt": (cand_rec or {}).get("_path"),
            "control_receipt": (ctrl_rec or {}).get("_path"),
            "dataset": dataset_identity(cand_run),
            "anchor_n": anchor["n"], "anchor_failed": anchor["n_failed"]}
    return {"res": res, "cost": cost, "resid": resid, "wall": wall, "mem": mem,
            "meta": meta, "nets": nets, "anchor": anchor}


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
    ap.add_argument("--candidate-receipt", default="",
                    help=f"{RECEIPT_SCHEMA} sidecar for the candidate run")
    ap.add_argument("--control-receipt", default="",
                    help=f"{RECEIPT_SCHEMA} sidecar for the control run")
    ap.add_argument("--submissions-ledger", default=os.path.join(
        _ROOT, "experiments", "submissions.jsonl"))
    ap.add_argument("--meter-path", default="",
                    choices=[""] + sorted(METER_PATH_TO_GRADED_OFFSET))
    ap.add_argument("--unverified-control-source", action="store_true",
                    help="proceed when no control source hash exists (never "
                         "on a mismatch); output is labelled UNVERIFIED")
    ap.add_argument("--unverified-candidate-source", action="store_true")
    ap.add_argument("--candidate", default="",
                    help="the candidate's source path, for the memory gate")
    ap.add_argument("--gate", type=float, default=None,
                    help="a PRE-REGISTERED gate. Omit unless a contract set one")
    ap.add_argument("--allow-failures", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    try:
        out = build(a)
    except (ProjectionRefused, cs.PairingError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        print("No projection was computed.", file=sys.stderr)
        return 2

    report(out["res"], out["cost"], out["resid"], out["wall"], out["mem"],
           out["meta"], out["nets"], out["anchor"])

    if a.json:
        payload = {"meta": out["meta"], "projection": out["res"],
                   "cost": out["cost"], "network_summary": out["nets"],
                   "residual_forecast": out["resid"], "wall": out["wall"],
                   "anchor": {k: v for k, v in out["anchor"].items() if k != "config"},
                   "memory": out["mem"],
                   "labels": {MATH: "arithmetic and identities; host-independent",
                              LOCAL: "measured on THIS host; not server seconds",
                              SERVER_MEASURED: "reported by the grading server",
                              SERVER: "a forecast from a versioned model with a "
                                      "stated evidence domain",
                              EMPIRICAL: "advisory rule of thumb; few candidates; "
                                         "not a safety proof"},
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
