"""Paired F21 accuracy screen and decision on already-minted Phase-2 truth.

This is deliberately one comparison, not a configurable experiment framework.
The L4 parent and L2 control are frozen own-code files.  The fused L4 SHA is
supplied after the engineering gate freezes its bytes.  All three arms use the
same canonical mints and the grader-twin lifecycle evaluator.

Local residual timing is not server timing.  This score run explicitly relaxes
the local 0.4 s residual cliff so host speed cannot replace an accuracy result
with zeroed predictions.  Subprocess and Linux memory gates remain separate.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() != _HERE]
for _p in (str(_HERE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

import competition_score as cs  # noqa: E402
import lifecycle_eval as le  # noqa: E402
import official_config  # noqa: E402
import seed_blocks  # noqa: E402
import seed_ledger  # noqa: E402
import truth_reads  # noqa: E402


L4_SHA = "052dd768e970c4bae831cff26f8ff15b0487f3b95eadf2fd3578e542aec34d58"
L2_SHA = "390bc226bbf517df0110facfc0dc91c78ef670e8a49368a536cabf630057d1db"
SERVER_OFFSET_MAX = official_config.PHASE2.server_flop_offset_per_mlp
OFFICIAL_SAMPLES = 1_000_000_000


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _source(path: str, expected: str) -> tuple[Path, str]:
    if Path(path).is_absolute():
        raise ValueError("candidate sources must be repo-relative snapshot paths")
    full = Path(os.path.abspath(_ROOT / path))
    if not full.is_relative_to(_ROOT):
        raise ValueError("candidate source leaves the executing snapshot")
    if not full.is_file():
        raise ValueError(f"missing candidate file: {full}")
    actual = _sha(full)
    if actual != expected:
        raise ValueError(f"source SHA mismatch for {full}: {actual} != {expected}")
    return full, actual


def _read(path: str, block: str, purpose: str, out: str,
          live_reads: Path, live_ledger: Path) -> tuple[dict, dict]:
    if Path(path).is_absolute():
        raise ValueError("truth must be a repo-relative snapshot path")
    full = Path(os.path.abspath(_ROOT / path))
    if not full.is_relative_to(_ROOT):
        raise ValueError("truth path leaves the executing snapshot")
    if not full.is_file():
        raise ValueError(f"missing truth file: {full}")
    canonical = full.relative_to(_ROOT).as_posix()
    prior = {r.reservation for r in truth_reads.reads(path=str(live_reads))}
    data = truth_reads.load_truth(
        block, str(full), consumer="local_sim/f21_score.py", purpose=purpose,
        instrument=str(Path(__file__).resolve()), result_artifact=str(Path(out).resolve()),
        path=str(live_reads), ledger_path=str(live_ledger))
    actual_sha = _sha(full)
    records = truth_reads.reads(path=str(live_reads))
    mine = [r for r in records if r.reservation not in prior
            and r.consumer == "local_sim/f21_score.py"
            and r.artifact == canonical and r.result_artifact == str(Path(out).resolve())]
    reserved = [r for r in mine if r.status == "reserved"]
    complete = [r for r in mine if r.status == "completed"]
    if len(reserved) != 1 or len(complete) != 1 or (
        reserved[0].reservation != complete[0].reservation
        or reserved[0].artifact_sha256 != actual_sha
        or complete[0].artifact_sha256 != actual_sha
    ):
        raise ValueError(f"truth read receipt failed to record canonical path {canonical}")
    return data, {"path": canonical, "snapshot_path": str(full),
                  "sha256": actual_sha, "live_register": str(live_reads),
                  "read_receipt_before_count": len(prior),
                  "read_receipt_after_count": len({r.reservation for r in records}),
                  "read_receipt_reservation": reserved[0].reservation,
                  "read_receipt_path": reserved[0].artifact,
                  "read_receipt_status": complete[0].status}


def _labels(a: argparse.Namespace, live_root: Path, live_reads: Path,
            live_ledger: Path) -> tuple[list[int], np.ndarray, np.ndarray, dict]:
    """Return seeds, float32 midpoint labels, predicted midpoint noise, provenance."""
    if a.tier == "design":
        canonical = (_ROOT / "experiments" / "2026-08-26_p2_gt")
        if Path(os.path.abspath(_ROOT / a.truth_a)) != canonical / "gt_halfA.npz" or (
            Path(os.path.abspath(_ROOT / a.truth_b)) != canonical / "gt_halfB.npz"
        ):
            raise ValueError("DESIGN must use the registered 24-network high-resolution P01 pair")
        da, pa = _read(a.truth_a, "P01", "F21 exposed DESIGN half A", a.out,
                       live_reads, live_ledger)
        db, pb = _read(a.truth_b, "P01", "F21 exposed DESIGN half B", a.out,
                       live_reads, live_ledger)
        required = {"seeds", "means", "var", "n_samples"}
        if required - da.keys() or required - db.keys():
            raise ValueError("high-resolution halves lack required arrays")
        if not np.array_equal(da["seeds"], db["seeds"]):
            raise ValueError("high-resolution halves identify different networks")
        n_samples = int(da["n_samples"])
        if n_samples != 2**26 or int(db["n_samples"]) != n_samples:
            raise ValueError("DESIGN truth must be two 2**26-sample halves")
        seeds = [int(s) for s in da["seeds"]]
        if len(seeds) != 24 or len(set(seeds)) != 24:
            raise ValueError("DESIGN truth must contain 24 unique networks")
        if not set(seeds) <= set(seed_blocks.seeds("P01")):
            raise ValueError("DESIGN truth seeds leave P01")
        y1 = np.asarray(da["means"][:, -1, :], dtype=np.float64)
        y2 = np.asarray(db["means"][:, -1, :], dtype=np.float64)
        v1 = np.asarray(da["var"], dtype=np.float64)
        v2 = np.asarray(db["var"], dtype=np.float64)
        # Mean final-neuron variance / (2*N): midpoint of independent halves.
        noise = ((v1 + v2) / 2).mean(axis=1) / (2 * n_samples)
        files = [pa, pb]
    else:
        truth_path = Path(os.path.abspath(_ROOT / a.truth))
        truth_rel = os.path.relpath(truth_path, _ROOT).replace("\\", "/")
        truth_sha = _sha(truth_path)
        if not any(c.block == "P09" and c.artifact == truth_rel
                   and c.artifact_sha256 == truth_sha
                   for c in seed_ledger.finalizations(path=str(live_ledger))):
            raise ValueError("P09 truth must be finalized in the seed ledger before reading")
        d, p = _read(a.truth, "P09", "F21 fresh DECIDE truth", a.out,
                     live_reads, live_ledger)
        required = {"seeds", "means_r1", "means_r2", "gt_noise", "n_samples"}
        if required - d.keys():
            raise ValueError("P09 truth lacks required arrays")
        n_samples = int(d["n_samples"])
        if n_samples != 2**24:
            raise ValueError("DECIDE truth must use 2**24 samples per replica")
        seeds = [int(s) for s in d["seeds"]]
        if len(seeds) != 128 or len(set(seeds)) != 128:
            raise ValueError("DECIDE truth must contain 128 unique networks")
        if seeds != seed_blocks.seeds("P09")[:128]:
            raise ValueError("DECIDE truth must contain the registered P09 span")
        claims = seed_ledger.claims(path=str(live_ledger))
        if not all(any(c.block == "P09" and c.start <= s < c.start + c.n
                       for c in claims) for s in seeds):
            raise ValueError("all P09 seeds must be claimed before truth is scored")
        y1 = np.asarray(d["means_r1"][:, -1, :], dtype=np.float64)
        y2 = np.asarray(d["means_r2"][:, -1, :], dtype=np.float64)
        noise = np.asarray(d["gt_noise"], dtype=np.float64) / 2.0
        files = [p]
    if y1.shape != (len(seeds), 1024) or y2.shape != y1.shape:
        raise ValueError("truth has the wrong final-layer shape")
    if noise.shape != (len(seeds),) or not np.all(np.isfinite(noise)) or np.any(noise <= 0):
        raise ValueError("missing, nonfinite or nonpositive truth noise")
    if not np.all(np.isfinite(y1)) or not np.all(np.isfinite(y2)):
        raise ValueError("nonfinite truth")
    empirical_noise = np.mean(((y1 - y2) / 2) ** 2, axis=1)
    midpoint = ((y1 + y2) / 2).astype(np.float32)
    return seeds, midpoint, noise, {
        "files": files, "n_samples_per_replica": n_samples,
        "mean_predicted_midpoint_noise": float(noise.mean()),
        "mean_empirical_midpoint_noise": float(empirical_noise.mean()),
        "empirical_to_predicted_noise_ratio": float(empirical_noise.mean() / noise.mean()),
        "local_samples_total": 2 * n_samples,
    }


def _score_vectors(rows: list[cs.MlpRow], local_samples: int) -> dict[str, np.ndarray]:
    v_official = np.array([r.gt_noise for r in rows]) * (local_samples / OFFICIAL_SAMPLES)
    return {
        "observed": cs.score_suite(rows, lam=0).scores,
        "corrected": cs.score_suite(
            [dataclasses.replace(r, mse=r.mse - r.gt_noise) for r in rows], lam=0).scores,
        "forecast": cs.score_suite(
            [dataclasses.replace(r, mse=r.mse - r.gt_noise + float(v))
             for r, v in zip(rows, v_official)], lam=0).scores,
        "forecast_offset_max": cs.score_suite(
            [dataclasses.replace(r, mse=r.mse - r.gt_noise + float(v),
                                 flops_used=r.flops_used + SERVER_OFFSET_MAX)
             for r, v in zip(rows, v_official)], lam=0).scores,
    }


def _pair(a: np.ndarray, b: np.ndarray) -> dict:
    r = cs.paired_compare(a, b)
    return {
        "mean_control": r.mean_control, "mean_candidate": r.mean_candidate,
        "mean_difference": r.mean_difference, "arithmetic_gain": r.arithmetic_gain,
        "t_ci_difference": list(r.t_ci_difference), "n": r.n,
        "geometric_gain_diagnostic": r.geometric_gain,
        "biggest_mover_share": float((a - b)[np.argmax(np.abs(a - b))] / (a - b).sum())
        if float((a - b).sum()) != 0 else None,
    }


def _report_lifecycle_failures(name: str, evidence: list[le.EvidenceRow]) -> bool:
    """Emit only failure metadata, never labels or partial score rows."""
    failed = [row for row in evidence if row.failed]
    for row in failed:
        diagnostic = {
            "arm": name,
            "index": row.index,
            "input_seed": row.input_seed,
            "budget_exhausted": row.budget_exhausted,
            "time_exhausted": row.time_exhausted,
            "residual_exhausted": row.residual_exhausted,
            "combined_exhausted": row.combined_exhausted,
            "error_code": row.error_code,
            "wall_time_s": row.wall_time_s,
            "residual_wall_time_s": row.residual_wall_time_s,
        }
        print("lifecycle failure " + json.dumps(diagnostic, sort_keys=True),
              file=sys.stderr, flush=True)
    return bool(failed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", choices=("design", "decide"), required=True)
    ap.add_argument("--truth-a", help="repo-relative snapshot DESIGN half A")
    ap.add_argument("--truth-b", help="repo-relative snapshot DESIGN half B")
    ap.add_argument("--truth", help="repo-relative snapshot DECIDE P09 NPZ")
    ap.add_argument("--l4", required=True, help="repo-relative frozen original L4 parent")
    ap.add_argument("--fused", required=True, help="repo-relative fused L4 candidate")
    ap.add_argument("--fused-sha", required=True, help="expected candidate SHA256")
    ap.add_argument("--l2", required=True, help="repo-relative frozen original L2 context")
    ap.add_argument("--out", required=True)
    ap.add_argument("--live-register-root", required=True,
                    help="absolute live repo root for truth-read and seed-ledger receipts")
    a = ap.parse_args()

    if (a.tier == "design" and (not a.truth_a or not a.truth_b or a.truth)) or (
        a.tier == "decide" and (not a.truth or a.truth_a or a.truth_b)):
        ap.error("DESIGN needs --truth-a/--truth-b; DECIDE needs --truth")
    live_root = Path(a.live_register_root)
    if not live_root.is_absolute():
        ap.error("--live-register-root must be absolute")
    live_root = live_root.resolve()
    git_root = subprocess.run(
        ["git", "-C", str(live_root), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True).stdout.strip()
    if Path(git_root).resolve() != live_root:
        ap.error("--live-register-root is not a Git repository root")
    if Path.cwd().resolve() != _ROOT:
        ap.error("run from the executing snapshot repository root")
    live_reads = live_root / "experiments" / "truth_reads.jsonl"
    live_ledger = live_root / "experiments" / "seed_ledger.jsonl"
    if not live_reads.is_file() or not live_ledger.is_file():
        ap.error("live truth-read and seed-ledger files must already exist")
    truth_paths = [a.truth] if a.tier == "decide" else [a.truth_a, a.truth_b]
    if any(Path(p).is_absolute() for p in truth_paths):
        ap.error("truth paths must be repo-relative snapshot paths")
    if a.tier == "decide" and not Path(a.truth).name.startswith("truth_P09"):
        ap.error("DECIDE truth filename must identify P09")
    out = Path(a.out).resolve()
    if not Path(a.out).is_absolute():
        ap.error("--out must be an absolute live path")
    if not out.is_relative_to(live_root / "experiments"):
        ap.error("--out must be inside the live experiments directory")
    if out.exists():
        ap.error("output already exists; evidence is immutable")
    if not a.fused_sha or len(a.fused_sha) != 64:
        ap.error("--fused-sha needs the frozen 64-character digest")

    paths = {
        "L4": _source(a.l4, L4_SHA),
        "fused_L4": _source(a.fused, a.fused_sha.lower()),
        "L2": _source(a.l2, L2_SHA),
    }
    if len({p for p, _ in paths.values()}) != 3:
        raise ValueError("all three arms must be distinct files")
    for name, (path, digest) in paths.items():
        live_copy = live_root / path.relative_to(_ROOT)
        if not live_copy.is_file() or _sha(live_copy) != digest:
            raise ValueError(f"live workspace does not contain snapshot-pinned {name} bytes")
    seeds, truth, noise, truth_prov = _labels(a, live_root, live_reads, live_ledger)
    spec = le.LifecycleSpec.for_round("v2-phase2", residual_wall_time_limit_s=None)
    print(f"F21 {a.tier.upper()}: {len(seeds)} paired networks; local residual cap explicitly relaxed", flush=True)
    print(f"midpoint truth floor {noise.mean():.6e}; official forecast N={OFFICIAL_SAMPLES:,}", flush=True)

    arms = {}
    for name in ("L4", "fused_L4", "L2"):
        path, expected_sha = paths[name]
        print(f"arm {name} {expected_sha}...", flush=True)
        evidence, prov = le.evaluate(
            str(path), seeds, truth, spec=spec, gt_noise=noise,
            weights_mode="f64_readonly", progress_every=8)
        if prov["candidate_sha256"] != expected_sha or any(
            row.candidate_sha256 != expected_sha for row in evidence):
            raise ValueError(f"{name} evaluated unexpected source bytes")
        if _sha(path) != expected_sha:
            raise ValueError(f"{name} source changed during evaluation")
        rows = [r.to_row() for r in evidence]
        if _report_lifecycle_failures(name, evidence):
            raise ValueError(f"{name} had a local non-residual lifecycle failure")
        vectors = _score_vectors(rows, truth_prov["local_samples_total"])
        arms[name] = {"evidence": evidence, "provenance": prov, "rows": rows,
                      "vectors": vectors}

    # Identity alignment is a mandatory pair check even though the generator
    # currently visits seeds in order.  All three arms share exact truth bytes.
    for name in ("fused_L4", "L2"):
        cs.align_rows_by_identity(arms["L4"]["rows"], arms[name]["rows"], require_ids=True)
        if [r.name for r in arms["L4"]["rows"]] != [r.name for r in arms[name]["rows"]]:
            raise ValueError(f"{name} rows are not in the same paired order as L4")
    comparisons = {}
    for name in ("fused_L4", "L2"):
        c, k = arms["L4"], arms[name]
        noise_compare = cs.noise_corrected_compare(
            c["rows"], k["rows"], local_samples=truth_prov["local_samples_total"],
            official_samples=OFFICIAL_SAMPLES, lam=0, require_ids=True)
        comparisons[name] = {
            "observed": _pair(c["vectors"]["observed"], k["vectors"]["observed"]),
            "corrected": _pair(c["vectors"]["corrected"], k["vectors"]["corrected"]),
            "forecast": _pair(c["vectors"]["forecast"], k["vectors"]["forecast"]),
            "forecast_offset_max": _pair(c["vectors"]["forecast_offset_max"],
                                         k["vectors"]["forecast_offset_max"]),
            "noise_bias_in_observed_difference": noise_compare.noise_bias,
        }

    # The preregistered gate uses the conservative observed maximum server
    # offset.  The offset-zero forecast remains a sensitivity analysis only.
    c = arms["L4"]["vectors"]["forecast_offset_max"]
    k = arms["fused_L4"]["vectors"]["forecast_offset_max"]
    if a.tier == "design":
        design_ratio = float(k.mean() / c.mean())
        verdict = {"kind": "DESIGN accuracy screen, official precision and max observed server offset",
                   "forecast_offset_max_score_ratio": design_ratio,
                   "max_allowed_ratio": 1.05, "passed": bool(design_ratio <= 1.05),
                   "promotion": False}
    else:
        v = cs.decide(c, k, gate=1.05, interval="t")
        verdict = {"kind": "DECIDE primary L4 comparison at official truth precision and max observed server offset",
                   "passed": v.passed, "gain": v.gain, "gate": v.gate,
                   "interval": v.interval, "ci_difference": list(v.ci),
                   "ci_excludes_zero": v.ci_excludes_zero,
                   "geometric_gain_diagnostic": v.geometric_gain,
                   "n": v.n, "n_candidate_better": v.n_candidate_better,
                   "biggest_mover_share": v.biggest_mover_share,
                   "reasons": list(v.reasons)}

    payload = {
        "experiment": "F21 one shared-right D21 fusion", "tier": a.tier.upper(),
        "data_role": "exposed DESIGN" if a.tier == "design" else "fresh DECIDE",
        "n": len(seeds), "seeds": seeds, "truth": truth_prov,
        "live_register_root": str(live_root),
        "live_truth_reads": str(live_reads), "live_seed_ledger": str(live_ledger),
        "score_model": {"lambda": 0, "budget": 2**41, "floor": 0.1,
                        "official_samples": OFFICIAL_SAMPLES,
                        "server_offset_max_observed": SERVER_OFFSET_MAX,
                        "offset_is_fixed": False,
                        "binding_gate_score": "forecast_offset_max",
                        "offset_zero_forecast_is_sensitivity_only": True,
                        "local_residual_cap_relaxed_accuracy_only": True,
                        "subprocess_deployment_certification": False},
        "source_hashes": {name: {"path": str(path), "sha256": digest}
                          for name, (path, digest) in paths.items()},
        "arms": {}, "comparisons_vs_L4": comparisons, "verdict": verdict,
    }
    for name, result in arms.items():
        ev = result["evidence"]
        vv = result["vectors"]
        payload["arms"][name] = {
            "provenance": result["provenance"],
            "mean_raw_mse": float(np.mean([r.mse for r in ev])),
            "mean_truth_floor": float(noise.mean()),
            "mean_flops": float(np.mean([r.flops_used for r in ev])),
            "worst_local_residual_s_accuracy_only": max(r.residual_wall_time_s for r in ev),
            "mean_scores": {key: float(x.mean()) for key, x in vv.items()},
            "rows": [{**dataclasses.asdict(row),
                      "score_observed": float(vv["observed"][i]),
                      "score_corrected": float(vv["corrected"][i]),
                      "score_forecast": float(vv["forecast"][i]),
                      "score_forecast_offset_max": float(vv["forecast_offset_max"][i])}
                     for i, row in enumerate(ev)],
        }
    for path, digest in paths.values():
        if _sha(path) != digest:
            raise ValueError(f"candidate changed before report publication: {path}")
    if out.exists():
        raise ValueError("output appeared during evaluation")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, allow_nan=False)
        f.write("\n")
    print(f"{a.tier.upper()} {'PASS' if verdict['passed'] else 'FAIL'}: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
