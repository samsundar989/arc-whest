"""Read-only diagnostic replay of the frozen red n4 integrated fixture."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "local_sim"))
import p2_k4_integrated_first_source as mod
from test_p2_k4_integrated_first_source import ordinary_tensor
from test_p2_k4_two_birth_compact import direct_birth_feed, direct_carry


def main():
    _, f = mod._real_flopscope()
    label = [""]
    qr_inputs, array_stats = [], []
    family_calls = {}
    original_qr = f.linalg.qr
    original_run = mod.Meter.run

    def audit_qr(y, *args, **kwargs):
        qr_inputs.append((label[0], np.asarray(y).copy()))
        return original_qr(y, *args, **kwargs)

    def audit_run(self, family, fn, *args, **kwargs):
        label[0] = family
        out = original_run(self, family, fn, *args, **kwargs)
        family_calls[family] = family_calls.get(family, 0) + 1
        if hasattr(out, "shape") and np.prod(out.shape) <= 64:
            a = np.asarray(out)
            if a.size:
                array_stats.append((family, family_calls[family],
                                    bool(np.isfinite(a).all()),
                                    float(np.nanmax(np.abs(a)))))
        return out

    f.linalg.qr = audit_qr
    mod.Meter.run = audit_run
    try:
        result = mod.integrated_metered_program(
            n=4, depth=16, source_rank=3, rank=4, recent=4,
            ordinary_residual=4, capture_tiny=True)
    finally:
        f.linalg.qr = original_qr
        mod.Meter.run = original_run

    traces, weights = result["_trace"], result["_inputs"][0]
    rows = []
    for layer in (14, 15):
        row = traces[layer]
        positive = {layer - s.age - 1: s for kind, s in row["sources"]
                    if kind == "feed_positive"}
        negative = {layer - s.age - 1: s for kind, s in row["sources"]
                    if kind == "feed_negative"}
        pair_gaps = []
        got_total = np.zeros((4, 4, 4))
        want_total = np.zeros((4, 4, 4))
        for birth in sorted(positive):
            sp, sn = positive[birth], negative[birth]
            got = ordinary_tensor(sp) + ordinary_tensor(sn)
            born = traces[birth]
            want = direct_birth_feed(born["V"], born["p"],
                                     born["w1"], born["w2"])
            for later in range(birth + 1, layer + 1):
                transform = np.asarray(weights[later], dtype=np.float32)
                if later > birth + 1:
                    transform = transform * traces[later - 1]["w1"][None, :]
                want = direct_carry(want, transform)
            pair_gaps.append({"birth": birth,
                              "max_abs_gap": float(np.max(np.abs(got - want))),
                              "max_abs_got": float(np.max(np.abs(got))),
                              "max_abs_want": float(np.max(np.abs(want)))})
            got_total += got
            want_total += want
        rows.append({"layer": layer, "feed_pair_gaps": pair_gaps,
                     "max_feed_pair_gap": max(x["max_abs_gap"] for x in pair_gaps),
                     "max_feed_sum_gap": float(np.max(np.abs(got_total - want_total))),
                     "max_feed_sum": float(np.max(np.abs(got_total))),
                     "max_ordinary_tensor": float(np.max(np.abs(sum(
                         (ordinary_tensor(s) for kind, s in row["sources"]
                          if kind == "ordinary"), np.zeros((4, 4, 4))))))})
    qrs = []
    for family, y in qr_inputs:
        singular = np.linalg.svd(y, compute_uv=False)
        qrs.append({"family": family,
                    "condition": (float(singular[0] / singular[-1])
                                  if singular[-1] else float("inf")),
                    "smallest_singular": float(singular[-1]),
                    "largest_singular": float(singular[0])})
    report = {
        "source_sha256": result["source_sha256"],
        "input_sha256": result["input_sha256"],
        "rows": rows,
        "qr_worst": sorted(qrs, key=lambda x: x["condition"], reverse=True)[:12],
        "qr_by_family": {name: {
            "count": sum(x["family"] == name for x in qrs),
            "max_condition": max((x["condition"] for x in qrs
                                  if x["family"] == name), default=0)}
            for name in sorted(set(x["family"] for x in qrs))},
        "nonfinite_families": sorted(set(name for name, _, finite, _ in array_stats
                                         if not finite)),
        "nonfinite_occurrences": [
            {"family": name, "call": call} for name, call, finite, _
            in array_stats if not finite],
        "largest_family_values": sorted(
            ({"family": name, "call": call, "max_abs": magnitude}
             for name, call, finite, magnitude in array_stats if finite),
            key=lambda x: x["max_abs"], reverse=True)[:12],
        "tracked_by_call": [
            {"family": name, "call": call, "finite": finite,
             "max_abs": magnitude}
            for name, call, finite, magnitude in array_stats
            if name in ("backbone_cov_linear_right", "nonlinear_safe_variance",
                        "nonlinear_gaussian_mean", "nonlinear_mean_add",
                        "nonlinear_mean_k3", "nonlinear_mean_k4",
                        "k4_D4_negative", "D3_contraction",
                        "nonlinear_gaussian_e3", "nonlinear_gaussian_k3",
                        "source_s", "source_e", "source_dP",
                        "source_residual_power")],
    }
    target = Path(__file__).with_name("tiny_diagnostic.json")
    target.write_bytes((json.dumps(report, indent=2, sort_keys=True)
                        + "\n").encode("utf-8"))
    print("feed gaps", [(x["layer"], x["max_feed_pair_gap"],
                         x["max_feed_sum_gap"], x["max_feed_sum"])
                        for x in rows])
    print("QR", report["qr_by_family"])
    print("nonfinite", report["nonfinite_families"])
    print("largest", report["largest_family_values"][:4])


if __name__ == "__main__":
    main()
