"""
Save the full grader report for a submission to experiments/run_reports/.

Neither `whest submit --json` nor `GET /api/v1/submissions/{id}` returns the
report — they give only id/score/score_secondary. The real thing (config_snapshot,
per-MLP telemetry, runtime_environment) is embedded in the submission page's
`data-react-props` payload, which is what this pulls out.

`admin_config` itself is admin-only and absent for participants; lambda and the
multiplier floor are derived from per-MLP telemetry instead (see --derive).

The API key is handled only by whestbench's own credential loader — never read,
printed, or written here.

Usage: uv run python local_sim/fetch_graded_report.py <submission_id> [label]
"""

import html
import json
import os
import re
import sys

import httpx

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

from whestbench import aicrowd_config  # noqa: E402

SLUG = "arc-white-box-estimation-challenge-2026"
OUT = os.path.join(_ROOT, "experiments", "run_reports")


def fetch(sub_id: int) -> dict:
    tok = aicrowd_config.resolve_api_key(None)
    page = httpx.get(
        f"https://www.aicrowd.com/challenges/{SLUG}/submissions/{sub_id}",
        headers={"Authorization": f"Token {tok}"},
        timeout=90,
    ).text
    for m in re.finditer(r'data-react-props="(.*?)"(?=\s|>)', page, re.S):
        try:
            p = json.loads(html.unescape(m.group(1)))
        except Exception:
            continue
        sub = (p.get("props") or {}).get("data", {}).get("submission")
        if isinstance(sub, dict) and "evaluation" in sub:
            return sub
    raise SystemExit(f"no evaluation payload found for submission {sub_id}")


def derive(sub: dict) -> dict:
    """Recover the grader config that admin_config would have told us."""
    ev = sub["evaluation"]
    agg = (ev.get("results", {}).get("aggregates", {}) or {}).get("public", {})
    lam, flops_, resid = [], [], []
    for row in ev.get("results", {}).get("per_mlp", []) or []:
        t = row.get("telemetry") or {}
        f, e, r = t.get("flops_used"), t.get("effective_compute"), t.get("residual_wall_time_s")
        if None not in (f, e, r) and r:
            lam.append((e - f) / r)
            flops_.append(f)
            resid.append(r)
    out = dict(ev.get("config_snapshot") or {})
    if lam:
        out["lambda_derived"] = sum(lam) / len(lam)
        out["flops_per_mlp"] = sum(flops_) / len(flops_)
        out["residual_s_per_mlp"] = sum(resid) / len(resid)
        out["residual_share_of_billed"] = 1.0 - sum(flops_) / sum(
            f + 1e11 * r for f, r in zip(flops_, resid)
        )
    if agg.get("score_secondary"):
        out["multiplier_floor_check"] = agg["score"] / agg["score_secondary"]
    out["runtime_environment"] = ev.get("runtime_environment")
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    sub_id = int(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else f"submission_{sub_id}"
    sub = fetch(sub_id)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{label}_{sub_id}_graded_report.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(sub, fh, indent=2)
    agg = (sub["evaluation"].get("results", {}).get("aggregates", {}) or {}).get("public", {})
    print(f"saved {path} ({os.path.getsize(path)} bytes)")
    print(f"\nstatus={sub.get('status')}  round={sub.get('round')}")
    for k in (
        "adjusted_final_layer_score",
        "final_layer_mse",
        "all_layers_mse",
        "mean_compute_utilization",
        "mean_score_multiplier",
        "n_failed_mlps",
        "best_mlp_adjusted_final_layer_score",
        "worst_mlp_adjusted_final_layer_score",
        "sampling_mse",
    ):
        if k in agg:
            print(f"  {k:<38} {agg[k]}")
    print("\nderived config:")
    print(json.dumps(derive(sub), indent=2, default=str))


if __name__ == "__main__":
    main()
