"""End-to-end synthetic regressions for the bounded coherent-K4 fit path."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile
from argparse import Namespace
import numpy as np
from numpy import load as _load_npz
import p2_k4_fit as fit
import p2_k4_fit_producer as producer

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(value, fh, sort_keys=True); fh.write("\n")


def _rewrite_npz(path, mutate):
    with _load_npz(path, allow_pickle=False) as z:
        values = {name: z[name] for name in z.files}
    mutate(values); np.savez(path, **values)


def _swap_source_order(values):
    order = [1, 0, 2]
    values["candidate_source_paths"] = np.asarray(
        values["candidate_source_paths"])[order]
    values["candidate_source_hashes"] = np.asarray(
        values["candidate_source_hashes"])[order]


def _registered_manifest():
    seeds = producer.seed_blocks.seeds("P07")
    return {"kind": "registered-design-identities-v1", "block": "P07",
            "width": 1024, "depth": 16,
            "identities": [{"id": str(seed), "seed": seed} for seed in seeds]}


def _registered_load(root, manifest):
    path = os.path.join(root, "registered-identities.json")
    _write_json(path, manifest)
    return producer._load_identities(path)


def _fixture(root):
    candidates, bounds = os.path.join(root, "candidates.json"), os.path.join(root, "bounds.json")
    identities, produced = os.path.join(root, "identities.json"), os.path.join(root, "candidate_observations")
    truth, receipt = os.path.join(root, "truth.npz"), os.path.join(root, "receipt.json")
    _write_json(candidates, {"parameters": ["lam"], "candidates":
                [{"lam": 0.0}, {"lam": 0.25}, {"lam": 0.5}]})
    _write_json(bounds, {"lam": {"min": 0.0, "max": 0.5}})
    _write_json(identities, {"kind": "synthetic-identities-v1", "width": 4,
                "depth": 3, "identities": [{"id": "synthetic-0", "seed": 7001},
                                             {"id": "synthetic-1", "seed": 7002}]})
    result = producer.produce(Namespace(candidates=candidates, bounds=bounds,
                                        identities=identities, out_dir=produced))
    with _load_npz(result["predictions"], allow_pickle=False) as z:
        ids, target = z["identities"], np.asarray(z["predictions"])[1]
    manifest = {"kind": "synthetic", "fixture": "producer-lifecycle-v1",
                "labels_protected": False, "identity_count": 2}
    np.savez(truth, identities=ids, truth=target,
             manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)))
    args = Namespace(predictions=result["predictions"], truth=truth,
                     data_role="synthetic", candidates=candidates, bounds=bounds,
                     identities=identities, objective="final_layer_mse",
                     receipt=receipt, block="", ts="", note="")
    return args, result


def _temp(prefix):
    return tempfile.TemporaryDirectory(prefix=prefix, dir=_ROOT)


def run_checks():
    rows = []
    def check(name, condition, detail):
        rows.append({"check": name, "pass": bool(condition), "detail": detail})
        print("%-52s %s  %s" % (name, "PASS" if condition else "FAIL", detail))

    with _temp("p2-k4-fit-p07-registry-") as root:
        manifest = _registered_manifest()
        loaded, ids, seeds, width, depth = _registered_load(root, manifest)
        check("exact P07 design identity registry is accepted",
              loaded == manifest and ids == [str(x) for x in seeds] and
              len(seeds) == 384 and (width, depth) == (1024, 16),
              "%d ordered identities at %dx%d" % (len(seeds), width, depth))

    registered_mutations = (
        ("wrong registered block", lambda m: m.__setitem__("block", "P08"),
         "exact block P07"),
        ("wrong registered seed", lambda m: m["identities"][10].__setitem__(
         "seed", m["identities"][10]["seed"] + 1), "exactly match P07"),
        ("wrong registered order", lambda m: m["identities"].__setitem__(
         slice(0, 2), m["identities"][1::-1]), "exactly match P07"),
        ("wrong registered shape", lambda m: m.__setitem__("width", 512),
         "width=1024 and depth=16"))
    for label, mutation, expected in registered_mutations:
        with _temp("p2-k4-fit-p07-refusal-") as root:
            manifest = _registered_manifest(); mutation(manifest)
            try: _registered_load(root, manifest); refused = False
            except producer.ProducerRefused as exc: refused = expected in str(exc)
            check("%s refuses without evaluation" % label, refused,
                  "validation-only; no lifecycle or truth call")

    with _temp("p2-k4-fit-e2e-") as root:
        a, produced = _fixture(root); receipt = fit.fit(a)
        check("producer to selector freezes synthetic optimum",
              receipt["selection"]["frozen_values"] == {"lam": 0.25}, str(receipt["selection"]))
        check("candidate source paths and hashes are verified",
              len(receipt["candidate_sources"]) == 3 and all(x["verified"] for x in receipt["candidate_sources"]),
              "%d generated sources" % len(receipt["candidate_sources"]))
        check("canonical lifecycle costs preserve identities",
              receipt["counted_costs"] == produced["counted_costs"] and
              len({tuple(x) for x in receipt["counted_costs"]}) == 1, str(receipt["counted_costs"]))
        before = open(a.receipt, "rb").read()
        try: fit.fit(a); refused = False
        except fit.FitRefused as exc: refused = "already exists" in str(exc)
        check("existing receipt is immutable", refused and open(a.receipt, "rb").read() == before,
              "exclusive create; bytes unchanged")

    with _temp("p2-k4-fit-source-bytes-") as root:
        a, made = _fixture(root); source = os.path.join(_ROOT, made["candidate_sources"][0])
        with open(source, "a", encoding="utf-8") as fh: fh.write("\n# mutation\n")
        calls = []
        try: fit.fit(a, truth_loader=lambda _a: calls.append(True)); refused = False
        except fit.FitRefused as exc: refused = "hash mismatch" in str(exc)
        check("mutated candidate source refuses before truth", refused and not calls, "loader calls=%d" % len(calls))

    with _temp("p2-k4-fit-source-hash-") as root:
        a, _ = _fixture(root)
        _rewrite_npz(a.predictions, lambda d: d.__setitem__("candidate_source_hashes",
                     np.asarray(["0" * 64] + list(d["candidate_source_hashes"])[1:])))
        calls = []
        try: fit.fit(a, truth_loader=lambda _a: calls.append(True)); refused = False
        except fit.FitRefused as exc: refused = "hash mismatch" in str(exc)
        check("mutated recorded hash refuses before truth", refused and not calls, "loader calls=%d" % len(calls))

    with _temp("p2-k4-fit-source-order-") as root:
        a, _ = _fixture(root); _rewrite_npz(a.predictions, _swap_source_order); calls = []
        try: fit.fit(a, truth_loader=lambda _a: calls.append(True)); refused = False
        except fit.FitRefused as exc: refused = "registered coherent configuration" in str(exc)
        check("mutated source order refuses before truth", refused and not calls,
              "loader calls=%d" % len(calls))

    mutations = (
        ("cost", lambda d: d.__setitem__("counted_costs", np.asarray(d["counted_costs"]) +
                                         np.asarray([[0], [1], [0]])), "value-independent"),
        ("candidate order", lambda d: d.__setitem__("candidate_values",
         np.asarray(d["candidate_values"])[[1, 0, 2]]), "exactly match"),
        ("identity", lambda d: d.__setitem__("identities", np.asarray(d["identities"])[::-1]), "identit"))
    for label, mutation, expected in mutations:
        with _temp("p2-k4-fit-mutation-") as root:
            a, _ = _fixture(root); _rewrite_npz(a.predictions, mutation); calls = []
            try: fit.fit(a, truth_loader=lambda _a: calls.append(True)); refused = False
            except fit.FitRefused as exc: refused = expected in str(exc)
            check("mutated %s refuses before truth" % label, refused and not calls,
                  "loader calls=%d" % len(calls))

    with _temp("p2-k4-fit-role-") as root:
        a, _ = _fixture(root); a.data_role, a.block = "protected-design", "P01"; calls = []
        try: fit.fit(a, truth_loader=lambda _a: calls.append(True)); refused = False
        except fit.FitRefused as exc: refused = "exact block P07" in str(exc)
        check("protected fitting requires exact P07", refused and not calls, "loader calls=%d" % len(calls))

    with _temp("p2-k4-fit-synthetic-role-") as root:
        a, _ = _fixture(root)
        np.savez(a.truth, seeds=np.asarray([1, 2]), means_r1=np.zeros((2, 3, 4)), means_r2=np.zeros((2, 3, 4)))
        try: fit.fit(a); refused = False
        except fit.FitRefused as exc: refused = "declared data role" in str(exc)
        check("synthetic role rejects protected schema", refused and not os.path.exists(a.receipt),
              "truth_reads cannot be bypassed")

    with _temp("p2-k4-fit-cli-") as root:
        a, _ = _fixture(root)
        command = [sys.executable, os.path.join(_HERE, "p2_k4_fit.py"), "--predictions", a.predictions,
                   "--truth", a.truth, "--data-role", "synthetic", "--candidates", a.candidates,
                   "--bounds", a.bounds, "--identities", a.identities, "--objective", a.objective,
                   "--receipt", a.receipt]
        proc = subprocess.run(command, capture_output=True, text=True)
        check("documented selector CLI executes", proc.returncode == 0 and os.path.isfile(a.receipt),
              "exit=%d receipt_created=%s" % (proc.returncode, os.path.isfile(a.receipt)))

    with _temp("p2-k4-producer-cli-") as root:
        candidates = os.path.join(root, "candidates.json")
        bounds = os.path.join(root, "bounds.json")
        identities = os.path.join(root, "identities.json")
        out_dir = os.path.join(root, "observations")
        _write_json(candidates, {"parameters": ["lam"], "candidates": [{"lam": 0.0}]})
        _write_json(bounds, {"lam": {"min": 0.0, "max": 0.0}})
        _write_json(identities, {"kind": "synthetic-identities-v1", "width": 3,
                    "depth": 2, "identities": [{"id": "cli-synthetic", "seed": 7011}]})
        command = [sys.executable, os.path.join(_HERE, "p2_k4_fit_producer.py"),
                   "--candidates", candidates, "--bounds", bounds,
                   "--identities", identities, "--out-dir", out_dir]
        proc = subprocess.run(command, capture_output=True, text=True)
        second = subprocess.run(command, capture_output=True, text=True)
        check("documented producer CLI is immutable", proc.returncode == 0 and
              os.path.isfile(os.path.join(out_dir, "producer_receipt.json")) and
              second.returncode == 2 and "already exists" in second.stdout,
              "first=%d second=%d" % (proc.returncode, second.returncode))
    return {"kind": "synthetic coherent-K4 producer-selector conformance", "labels_read": False,
            "checks": rows, "n_checks": len(rows), "n_failed": sum(not x["pass"] for x in rows),
            "pass": all(x["pass"] for x in rows)}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--json", default=""); ap.add_argument("--fixture-dir", default="")
    a = ap.parse_args(argv)
    if a.fixture_dir:
        os.mkdir(a.fixture_dir); fixture, _ = _fixture(a.fixture_dir); receipt = fit.fit(fixture)
        print("synthetic fixture selected %s" % json.dumps(receipt["selection"]["frozen_values"], sort_keys=True)); return 0
    result = run_checks()
    if a.json:
        if os.path.lexists(a.json): raise SystemExit("REFUSED: evidence already exists: %s" % a.json)
        with open(a.json, "x", encoding="utf-8", newline="\n") as fh: json.dump(result, fh, indent=2, sort_keys=True); fh.write("\n")
    print("%d/%d passed" % (result["n_checks"] - result["n_failed"], result["n_checks"]))
    return 0 if result["pass"] else 1


if __name__ == "__main__": raise SystemExit(main())
