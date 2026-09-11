"""End-to-end synthetic regressions for the bounded coherent-K4 fit path."""
from __future__ import annotations
import argparse, copy, dataclasses, json, os, subprocess, sys, tempfile
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
    args = Namespace(predictions=result["predictions"],
                     producer_receipt=os.path.join(produced, "producer_receipt.json"),
                     truth=truth,
                     data_role="synthetic", candidates=candidates, bounds=bounds,
                     identities=identities, objective="final_layer_mse",
                     receipt=receipt, block="", ts="", note="")
    return args, result


def _verify_args(a, receipt):
    return Namespace(predictions=a.predictions,
                     producer_receipt=a.producer_receipt,
                     truth="", data_role="", candidates=a.candidates,
                     bounds=a.bounds, identities=a.identities,
                     objective=a.objective, receipt=receipt,
                     block="", ts="", note="")


def _temp(prefix):
    return tempfile.TemporaryDirectory(prefix=prefix, dir=_ROOT)


def _mock_clean_evidence(source_path, seeds, target, spec, predictions_out):
    if len(seeds) != 1:
        raise AssertionError("mock helper expects one identity")
    source_hash = producer._sha256(source_path)
    candidate_path = os.path.relpath(source_path, _ROOT).replace("\\", "/")
    prediction = np.zeros(spec.width, dtype=np.float32)
    predictions_out.append(prediction)
    row = producer.le.EvidenceRow(
        index=0, input_seed=seeds[0],
        estimator_seed=producer.mint_provenance.seed_streams(seeds[0])[2],
        candidate=candidate_path, candidate_sha256=source_hash,
        truth_sha256=producer.le._array_sha(target), width=spec.width,
        depth=spec.depth, budget=float(spec.flop_budget), flops_used=1.0,
        n_ops=1.0, mse=0.0, zero_prediction_mse=0.0, gt_noise=0.0,
        residual_wall_time_s=0.0, wall_time_s=0.0)
    flops = producer.le._require_real_flopscope()
    provenance = {
        "candidate": candidate_path,
        "candidate_sha256": source_hash,
        "truth_sha256": producer.le._array_sha(target),
        "n_mlps": 1,
        "spec": producer.dataclasses.asdict(spec),
        "round_tag": spec.round_tag,
        "round_deviations": spec.deviations_from_round(),
        "lifecycle": "grader",
        "weights_mode": producer.le.DEFAULT_WEIGHTS_MODE,
        "weights_mode_is_grader_default": True,
        "setup_called": True,
        "setup_seconds": 0.0,
        "setup_timeout_s": spec.setup_timeout_s,
        "setup_timed_out": False,
        "setup_error": "",
        "host_cold_start_s": producer.le.HOST_COLD_START_S,
        "residual_measurement_path": "runner=local (in-process BudgetContext)",
        "residual_is_graded": False,
        "objective_is_graded": False,
        "backend": {"flopscope": getattr(flops, "__version__", "?"),
                    "flopscope_path": os.path.dirname(flops.__file__ or ""),
                    "numpy": np.__version__, "python": sys.version.split()[0]},
        "rows_hash": producer.le.rows_hash([row]),
    }
    return [row], provenance


def run_checks():
    rows = []
    def check(name, condition, detail):
        rows.append({"check": name, "pass": bool(condition), "detail": detail})
        print("%-52s %s  %s" % (name, "PASS" if condition else "FAIL", detail))

    with _temp("p2-k4-fit-p07-registry-") as root:
        manifest = _registered_manifest()
        loaded, ids, seeds, width, depth, _ = _registered_load(root, manifest)
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
         "width=1024 and depth=16"),
        ("string registered seed", lambda m: m["identities"][0].__setitem__(
         "seed", str(m["identities"][0]["seed"])), "integer seed"),
        ("integer registered id", lambda m: m["identities"][0].__setitem__(
         "id", m["identities"][0]["seed"]), "string id"),
        ("extra registered manifest key", lambda m: m.__setitem__("extra", 1),
         "keys must be exactly"))
    for label, mutation, expected in registered_mutations:
        with _temp("p2-k4-fit-p07-refusal-") as root:
            manifest = _registered_manifest(); mutation(manifest)
            try: _registered_load(root, manifest); refused = False
            except (producer.ProducerRefused, fit.FitRefused) as exc:
                refused = expected in str(exc)
            check("%s refuses without evaluation" % label, refused,
                  "validation-only; no lifecycle or truth call")

    with _temp("p2-k4-fit-verify-") as root:
        a, produced = _fixture(root)
        verify_path = os.path.join(root, "verification.json")
        v = _verify_args(a, verify_path)
        calls = []
        original_loader = fit._load_truth
        fit._load_truth = lambda _a: calls.append(True)
        try:
            verification = fit.verify(v)
        finally:
            fit._load_truth = original_loader
        expected_sources = {"fit_interface", "fit_producer", "arm_generator",
                            "arm_exact", "arm_coherent", "arm_consumers_only",
                            "arm_births_only"}
        check("verify-only binds producer evidence without truth",
              (not calls and verification["labels_read"] is False
               and verification["verification_only"] is True
               and set(verification["source_hashes"]) == expected_sources
               and verification["inputs"]["producer_receipt"]["sha256"] ==
               fit._sha256(a.producer_receipt)),
              "loader calls=%d sources=%d" %
              (len(calls), len(verification["source_hashes"])))
        before = open(verify_path, "rb").read()
        try: fit.verify(v); refused = False
        except fit.FitRefused as exc: refused = "already exists" in str(exc)
        check("verification receipt is immutable",
              refused and open(verify_path, "rb").read() == before,
              "exclusive create; bytes unchanged")

        truthy = _verify_args(a, os.path.join(root, "truthy-verification.json"))
        truthy.truth = a.truth
        try: fit.verify(truthy); refused = False
        except fit.FitRefused as exc: refused = "does not accept truth" in str(exc)
        check("verify-only refuses truth arguments",
              refused and not os.path.exists(truthy.receipt),
              "no truth path accepted or resolved")

        original_receipt_bytes = open(a.producer_receipt, "rb").read()
        original_receipt_sha = fit.hashlib.sha256(original_receipt_bytes).hexdigest()
        replacement = os.path.join(root, "replacement-producer-receipt.json")
        with open(replacement, "wb") as fh:
            fh.write(original_receipt_bytes + b"\n")
        swap_request = _verify_args(a, os.path.join(root, "swap-verification.json"))
        original_reader = fit._read_json
        def swap_after_parse(path, label):
            snapshot = original_reader(path, label)
            if os.path.normcase(os.path.abspath(path)) == os.path.normcase(
                    os.path.abspath(a.producer_receipt)):
                os.replace(replacement, a.producer_receipt)
            return snapshot
        fit._read_json = swap_after_parse
        try:
            try:
                fit.verify(swap_request)
                refused = False
            except fit.FitRefused as exc:
                refused = "producer receipt changed" in str(exc)
        finally:
            fit._read_json = original_reader
        current_receipt_sha = fit._sha256(a.producer_receipt)
        check("receipt replacement after snapshot refuses at closing revalidation",
              (refused and original_receipt_sha != current_receipt_sha
               and not os.path.exists(swap_request.receipt)),
              "parsed=%s current=%s" %
              (original_receipt_sha[:12], current_receipt_sha[:12]))
        with open(a.producer_receipt, "wb") as fh:
            fh.write(original_receipt_bytes)

        original_prediction_bytes = open(a.predictions, "rb").read()
        original_prediction_sha = fit.hashlib.sha256(
            original_prediction_bytes).hexdigest()
        replacement = os.path.join(root, "replacement-predictions.npz")
        with open(replacement, "wb") as fh:
            fh.write(original_prediction_bytes + b"\n")
        swap_request = _verify_args(a, os.path.join(root, "npz-swap-verification.json"))
        original_bytes_reader = fit._read_bytes
        def swap_after_read(path, label):
            snapshot = original_bytes_reader(path, label)
            if os.path.normcase(os.path.abspath(path)) == os.path.normcase(
                    os.path.abspath(a.predictions)):
                os.replace(replacement, a.predictions)
            return snapshot
        fit._read_bytes = swap_after_read
        try:
            try:
                fit.verify(swap_request)
                refused = False
            except fit.FitRefused as exc:
                refused = "predictions changed" in str(exc)
        finally:
            fit._read_bytes = original_bytes_reader
        current_prediction_sha = fit._sha256(a.predictions)
        check("NPZ replacement after snapshot refuses at closing revalidation",
              (refused and original_prediction_sha != current_prediction_sha
               and not os.path.exists(swap_request.receipt)),
              "parsed=%s current=%s" %
              (original_prediction_sha[:12], current_prediction_sha[:12]))
        with open(a.predictions, "wb") as fh:
            fh.write(original_prediction_bytes)

        with open(a.producer_receipt, encoding="utf-8") as fh:
            pristine = json.load(fh)
        receipt_mutations = (
            ("extra top-level key", lambda r: r.__setitem__("extra", 1)),
            ("schema", lambda r: r.__setitem__("schema", "wrong")),
            ("immutable type", lambda r: r.__setitem__("immutable", 1)),
            ("prediction path", lambda r: r.__setitem__("predictions", a.candidates)),
            ("prediction path escape", lambda r: r.__setitem__("predictions", "../../outside.npz")),
            ("prediction hash", lambda r: r.__setitem__("predictions_sha256", "0" * 64)),
            ("candidate source paths", lambda r: r.__setitem__("candidate_sources", [])),
            ("candidate source hashes", lambda r: r.__setitem__("candidate_source_hashes", [])),
            ("receipt costs", lambda r: r.__setitem__("counted_costs", [])),
            ("receipt cost type", lambda r: r["counted_costs"][0].__setitem__(0, True)),
            ("manifest extra key", lambda r: r["manifest"].__setitem__("extra", 1)),
            ("labels-read flag", lambda r: r["manifest"].__setitem__("labels_read", True)),
            ("identity role", lambda r: r["manifest"].__setitem__("identity_role", "wrong")),
            ("candidate registry", lambda r: r["manifest"].__setitem__("candidate_registry", [])),
            ("bounds", lambda r: r["manifest"].__setitem__("bounds", {})),
            ("identity manifest", lambda r: r["manifest"].__setitem__("identity_manifest", {})),
            ("input bindings", lambda r: r["manifest"].__setitem__(
                "input_bindings", {})),
            ("manifest candidate sources", lambda r: r["manifest"].__setitem__("candidate_sources", [])),
            ("manifest costs", lambda r: r["manifest"].__setitem__("counted_costs", [])),
            ("manifest cost type", lambda r: r["manifest"]["counted_costs"][0].__setitem__(0, True)),
            ("producer path", lambda r: r["manifest"]["producer"].__setitem__("path", "local_sim/wrong.py")),
            ("producer hash", lambda r: r["manifest"]["producer"].__setitem__("sha256", "0" * 64)),
            ("generator path", lambda r: r["manifest"]["generator"].__setitem__("path", "local_sim/wrong.py")),
            ("generator hash", lambda r: r["manifest"]["generator"].__setitem__("sha256", "0" * 64)),
            ("upstream hash", lambda r: r["manifest"]["generator"].__setitem__("upstream_source_sha256", "0" * 64)),
            ("lifecycle width", lambda r: r["manifest"]["lifecycle_spec"].__setitem__("width", 5)),
            ("lifecycle depth", lambda r: r["manifest"]["lifecycle_spec"].__setitem__("depth", 4)),
            ("lifecycle budget", lambda r: r["manifest"]["lifecycle_spec"].__setitem__("flop_budget", 1)),
            ("lifecycle mode", lambda r: r["manifest"]["lifecycle_spec"].__setitem__("weights_mode", "f32")),
            ("lifecycle residual cap", lambda r: r["manifest"]["lifecycle_spec"].__setitem__("residual_cap", 0.4)),
            ("row coverage", lambda r: r["manifest"].__setitem__("row_hashes", [])),
            ("row hash type", lambda r: r["manifest"]["row_hashes"].__setitem__(0, True)),
            ("operation coverage", lambda r: r["manifest"].__setitem__("n_ops", [])),
            ("operation count type", lambda r: r["manifest"]["n_ops"][0].__setitem__(0, True)),
            ("lifecycle provenance", lambda r: r["manifest"].__setitem__(
                "lifecycle_provenance", [])),
            ("lifecycle provenance mode", lambda r: r["manifest"]
             ["lifecycle_provenance"][0].__setitem__("lifecycle",
                                                       "per_mlp_no_setup")),
        )
        for i, (label, mutation) in enumerate(receipt_mutations):
            mutated = copy.deepcopy(pristine)
            mutation(mutated)
            receipt_path = os.path.join(root, "mutated-%02d.json" % i)
            _write_json(receipt_path, mutated)
            request = _verify_args(a, os.path.join(root, "rejected-%02d.json" % i))
            request.producer_receipt = receipt_path
            try: fit.verify(request); refused = False
            except fit.FitRefused: refused = True
            check("mutated producer receipt refuses: " + label,
                  refused and not os.path.exists(request.receipt),
                  "truth-free refusal")

        bad_for_fit = copy.deepcopy(pristine)
        bad_for_fit["predictions_sha256"] = "0" * 64
        bad_fit_receipt = os.path.join(root, "bad-for-fit.json")
        _write_json(bad_fit_receipt, bad_for_fit)
        fit_request = copy.copy(a)
        fit_request.producer_receipt = bad_fit_receipt
        fit_request.receipt = os.path.join(root, "must-not-fit.json")
        calls = []
        try:
            fit.fit(fit_request, truth_loader=lambda _a: calls.append(True))
            refused = False
        except fit.FitRefused:
            refused = True
        check("normal fit revalidates producer receipt before truth",
              refused and not calls and not os.path.exists(fit_request.receipt),
              "loader calls=%d" % len(calls))

    identity_mutations = (
        ("string seed", lambda m: m["identities"][0].__setitem__(
         "seed", str(m["identities"][0]["seed"])), "integer seed"),
        ("integer id", lambda m: m["identities"][0].__setitem__(
         "id", m["identities"][0]["seed"]), "string id"),
        ("extra manifest key", lambda m: m.__setitem__("extra", 1),
         "keys must be exactly"))
    for label, mutation, expected in identity_mutations:
        with _temp("p2-k4-fit-identity-type-") as root:
            a, _ = _fixture(root)
            with open(a.identities, encoding="utf-8") as fh:
                manifest = json.load(fh)
            mutation(manifest)
            _write_json(a.identities, manifest)
            request = _verify_args(a, os.path.join(root, "must-not-verify.json"))
            try: fit.verify(request); refused = False
            except fit.FitRefused as exc: refused = expected in str(exc)
            check("verifier refuses identity %s" % label,
                  refused and not os.path.exists(request.receipt),
                  "strict shared manifest parser")

    registry_mutations = (
        ("extra candidate-registry key", "candidates",
         lambda value: value.__setitem__("extra", 1), "keys must be exactly"),
        ("string bound numeric", "bounds",
         lambda value: value["lam"].__setitem__("max", "0.5"),
         "must be JSON numbers"),
        ("boolean false candidate numeric", "candidates",
         lambda value: value["candidates"][0].__setitem__("lam", False),
         "must be a JSON number"))
    for label, target, mutation, expected in registry_mutations:
        with _temp("p2-k4-fit-registry-type-") as root:
            a, _ = _fixture(root)
            path = a.candidates if target == "candidates" else a.bounds
            with open(path, encoding="utf-8") as fh:
                value = json.load(fh)
            mutation(value)
            _write_json(path, value)
            calls = []
            try:
                fit.fit(a, truth_loader=lambda _a: calls.append(True))
                refused = False
            except fit.FitRefused as exc:
                refused = expected in str(exc)
            check("fit refuses %s before truth" % label,
                  refused and not calls and not os.path.exists(a.receipt),
                  "loader calls=%d" % len(calls))

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
        check("normal fit records producer receipt revalidation",
              (receipt["validation"]["producer_receipt_revalidated_before_truth_load"] is True
               and receipt["inputs"]["producer_receipt"]["sha256"] ==
               fit._sha256(a.producer_receipt)),
              receipt["inputs"]["producer_receipt"]["sha256"][:12])
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
        except fit.FitRefused: refused = True
        check("mutated candidate source refuses before truth", refused and not calls, "loader calls=%d" % len(calls))

    with _temp("p2-k4-fit-source-closing-race-") as root:
        a, made = _fixture(root)
        source = os.path.join(_ROOT, made["candidate_sources"][0])
        request = _verify_args(a, os.path.join(root, "must-not-verify.json"))
        original_problem = producer._provenance_problem
        mutated = [False]
        def mutate_source_during_provenance(*args, **kwargs):
            result = original_problem(*args, **kwargs)
            if not mutated[0]:
                with open(source, "a", encoding="utf-8") as fh:
                    fh.write("\n# closing-race mutation\n")
                mutated[0] = True
            return result
        producer._provenance_problem = mutate_source_during_provenance
        try:
            try:
                fit.verify(request)
                refused = False
            except fit.FitRefused as exc:
                refused = "candidate source 0 changed" in str(exc)
        finally:
            producer._provenance_problem = original_problem
        check("verifier closing recheck catches source changed during provenance",
              refused and mutated[0] and not os.path.exists(request.receipt),
              "refused=%s receipt=%s" %
              (refused, os.path.exists(request.receipt)))

    with _temp("p2-k4-fit-registry-closing-race-") as root:
        a, _ = _fixture(root)
        request = _verify_args(a, os.path.join(root, "must-not-verify.json"))
        original_sha = fit._sha256
        mutated = [False]
        denied = [False]
        def mutate_registry_after_its_closing_read(path):
            result = original_sha(path)
            if (not mutated[0] and
                    os.path.normcase(os.path.abspath(path)) ==
                    os.path.normcase(os.path.abspath(a.candidates))):
                mutated[0] = True
                try:
                    with open(a.candidates, "a", encoding="utf-8") as fh:
                        fh.write("\n")
                except PermissionError:
                    denied[0] = True
            return result
        fit._sha256 = mutate_registry_after_its_closing_read
        try:
            try:
                verified = fit.verify(request)
                accepted = True
            except fit.FitRefused:
                accepted = False
        finally:
            fit._sha256 = original_sha
        check("verifier lock blocks registry write during closing scan",
              (accepted and mutated[0] and denied[0]
               and fit._sha256(a.candidates) ==
                   verified["inputs"]["candidates"]["sha256"]
               and os.path.exists(request.receipt)),
              "accepted=%s denied=%s" % (accepted, denied[0]))

    with _temp("p2-k4-fit-source-hash-") as root:
        a, _ = _fixture(root)
        _rewrite_npz(a.predictions, lambda d: d.__setitem__("candidate_source_hashes",
                     np.asarray(["0" * 64] + list(d["candidate_source_hashes"])[1:])))
        calls = []
        try: fit.fit(a, truth_loader=lambda _a: calls.append(True)); refused = False
        except fit.FitRefused: refused = True
        check("mutated recorded hash refuses before truth", refused and not calls, "loader calls=%d" % len(calls))

    with _temp("p2-k4-fit-source-order-") as root:
        a, _ = _fixture(root); _rewrite_npz(a.predictions, _swap_source_order); calls = []
        try: fit.fit(a, truth_loader=lambda _a: calls.append(True)); refused = False
        except fit.FitRefused: refused = True
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
            except fit.FitRefused: refused = True
            check("mutated %s refuses before truth" % label, refused and not calls,
                  "loader calls=%d" % len(calls))

    with _temp("p2-k4-fit-role-") as root:
        a, _ = _fixture(root); a.data_role, a.block = "protected-design", "P01"; calls = []
        try: fit.fit(a, truth_loader=lambda _a: calls.append(True)); refused = False
        except fit.FitRefused as exc: refused = "exact block P07" in str(exc)
        check("protected fitting requires exact P07", refused and not calls, "loader calls=%d" % len(calls))

    with _temp("p2-k4-fit-role-confusion-") as root:
        a, _ = _fixture(root)
        a.data_role, a.block = "protected-design", "P07"
        calls = []
        try: fit.fit(a, truth_loader=lambda _a: calls.append(True)); refused = False
        except fit.FitRefused as exc:
            refused = "registered-design-identities-v1" in str(exc)
        check("protected P07 refuses synthetic identity package",
              refused and not calls and not os.path.exists(a.receipt),
              "loader calls=%d" % len(calls))

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
                   "--producer-receipt", a.producer_receipt,
                   "--truth", a.truth, "--data-role", "synthetic", "--candidates", a.candidates,
                   "--bounds", a.bounds, "--identities", a.identities, "--objective", a.objective,
                   "--receipt", a.receipt]
        proc = subprocess.run(command, capture_output=True, text=True)
        check("documented selector CLI executes", proc.returncode == 0 and os.path.isfile(a.receipt),
              "exit=%d receipt_created=%s" % (proc.returncode, os.path.isfile(a.receipt)))

    with _temp("p2-k4-fit-verify-cli-") as root:
        a, _ = _fixture(root)
        verification = os.path.join(root, "verification.json")
        command = [sys.executable, os.path.join(_HERE, "p2_k4_fit.py"),
                   "--verify-only", "--predictions", a.predictions,
                   "--producer-receipt", a.producer_receipt,
                   "--candidates", a.candidates, "--bounds", a.bounds,
                   "--identities", a.identities, "--objective", a.objective,
                   "--receipt", verification]
        proc = subprocess.run(command, capture_output=True, text=True)
        check("documented verify-only CLI executes without truth",
              proc.returncode == 0 and os.path.isfile(verification)
              and "without truth access" in proc.stdout,
              "exit=%d receipt_created=%s" %
              (proc.returncode, os.path.isfile(verification)))

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
              not os.path.exists(os.path.join(out_dir, "failure_diagnostic.json")) and
              second.returncode == 2 and "already exists" in second.stdout,
              "first=%d second=%d" % (proc.returncode, second.returncode))

    with _temp("p2-k4-producer-failure-") as root:
        candidates = os.path.join(root, "candidates.json")
        bounds = os.path.join(root, "bounds.json")
        identities = os.path.join(root, "identities.json")
        out_dir = os.path.join(root, "observations")
        _write_json(candidates, {"parameters": ["lam"], "candidates": [{"lam": 0.0}]})
        _write_json(bounds, {"lam": {"min": 0.0, "max": 0.0}})
        _write_json(identities, {"kind": "synthetic-identities-v1", "width": 3,
                    "depth": 2, "identities": [{"id": "failed-synthetic", "seed": 7021}]})
        original_input_hashes = {
            "candidates_sha256": producer._sha256(candidates),
            "bounds_sha256": producer._sha256(bounds),
            "identities_sha256": producer._sha256(identities),
        }
        expected_provenance = {}
        original_evaluate = producer.le.evaluate
        def failing_evaluate(source_path, seeds, target, *, spec, gt_noise,
                             predictions_out):
            row = producer.le.EvidenceRow(
                index=0, input_seed=seeds[0],
                estimator_seed=producer.mint_provenance.seed_streams(seeds[0])[2],
                candidate=os.path.relpath(source_path, _ROOT).replace("\\", "/"),
                candidate_sha256=producer._sha256(source_path),
                truth_sha256=producer.le._array_sha(target),
                width=spec.width, depth=spec.depth,
                budget=float(spec.flop_budget), flops_used=float(spec.flop_budget + 1),
                n_ops=17.0, mse=0.0, zero_prediction_mse=0.0,
                gt_noise=0.0, residual_wall_time_s=0.1, wall_time_s=0.2,
                budget_exhausted=True, error_code="", nonfinite=False)
            predictions_out.append(np.zeros(spec.width, dtype=np.float32))
            expected_provenance.update({
                "candidate": row.candidate,
                "candidate_sha256": row.candidate_sha256,
                "truth_sha256": producer.le._array_sha(target),
                "n_mlps": 1,
                "spec": producer.dataclasses.asdict(spec),
                "round_tag": spec.round_tag,
                "round_deviations": spec.deviations_from_round(),
                "lifecycle": "grader",
                "weights_mode": producer.le.DEFAULT_WEIGHTS_MODE,
                "weights_mode_is_grader_default": True,
                "setup_called": True,
                "setup_seconds": 0.125,
                "setup_timeout_s": spec.setup_timeout_s,
                "setup_timed_out": False,
                "setup_error": "",
                "host_cold_start_s": producer.le.HOST_COLD_START_S,
                "residual_measurement_path":
                    "runner=local (in-process BudgetContext)",
                "residual_is_graded": False,
                "objective_is_graded": False,
                "backend": {
                    "flopscope": getattr(producer.le._require_real_flopscope(),
                                         "__version__", "?"),
                    "flopscope_path": os.path.dirname(
                        producer.le._require_real_flopscope().__file__ or ""),
                    "numpy": np.__version__,
                    "python": sys.version.split()[0]},
                "rows_hash": producer.le.rows_hash([row]),
            })
            return [row], dict(expected_provenance)
        producer.le.evaluate = failing_evaluate
        try:
            try:
                producer.produce(Namespace(candidates=candidates, bounds=bounds,
                                           identities=identities, out_dir=out_dir))
                refused = False
            except producer.ProducerRefused:
                refused = True
        finally:
            producer.le.evaluate = original_evaluate
        diagnostic_path = os.path.join(out_dir, "failure_diagnostic.json")
        with open(diagnostic_path, encoding="utf-8") as fh:
            diagnostic = json.load(fh)
        failure_row = diagnostic["rows"][0]
        check("failed producer writes exclusive truth-free row evidence",
              (refused and diagnostic["immutable"] is True
               and diagnostic["complete_candidate"] is False
               and diagnostic["labels_read"] is False
               and diagnostic["failure_stage"] == "predict"
               and diagnostic["evaluated_rows"] == 1
               and diagnostic["captured_predictions"] == 1
               and all(
                   diagnostic["inputs"][name]["frozen_sha256"] ==
                       original_input_hashes[name + "_sha256"]
                   and diagnostic["inputs"][name]["current_sha256"] ==
                       producer._sha256(path)
                   and diagnostic["inputs"][name]["unchanged_after_evaluation"] is True
                   and diagnostic["inputs"][name]["observation_error"] == ""
                   for name, path in (("candidates", candidates),
                                      ("bounds", bounds),
                                      ("identities", identities)))
               and diagnostic["candidate"] == {"lam": 0.0}
               and diagnostic["identity_manifest"]["identities"][0] ==
                   {"id": "failed-synthetic", "seed": 7021}
               and diagnostic["lifecycle_spec"] ==
                   producer.dataclasses.asdict(producer.le.LifecycleSpec(
                       width=3, depth=2, residual_wall_time_limit_s=None))
               and diagnostic["lifecycle_provenance"] == expected_provenance
               and failure_row["budget_exhausted"] is True
               and failure_row["error_code"] == ""
               and failure_row["flops_used"] == float(2 ** 41 + 1)
               and failure_row["mse"] == 0.0
               and not os.path.exists(os.path.join(out_dir, "predictions.npz"))
               and not os.path.exists(os.path.join(out_dir, "producer_receipt.json"))),
              "refused=%s rows=%d no_success_artifacts=%s" %
              (refused, len(diagnostic["rows"]),
               not os.path.exists(os.path.join(out_dir, "predictions.npz"))))

    with _temp("p2-k4-producer-setup-failure-") as root:
        candidates = os.path.join(root, "candidates.json")
        bounds = os.path.join(root, "bounds.json")
        identities = os.path.join(root, "identities.json")
        out_dir = os.path.join(root, "observations")
        _write_json(candidates, {"parameters": ["lam"], "candidates": [{"lam": 0.0}]})
        _write_json(bounds, {"lam": {"min": 0.0, "max": 0.0}})
        _write_json(identities, {"kind": "synthetic-identities-v1", "width": 3,
                    "depth": 2, "identities": [{"id": "setup-failure", "seed": 7031}]})
        original_evaluate = producer.le.evaluate
        def setup_failure(*args, **kwargs):
            raise producer.le.SetupFailure("SETUP_ERROR: deliberate fixture")
        producer.le.evaluate = setup_failure
        try:
            try:
                producer.produce(Namespace(candidates=candidates, bounds=bounds,
                                           identities=identities, out_dir=out_dir))
                refused = False
            except producer.ProducerRefused as exc:
                refused = "SetupFailure" in str(exc)
        finally:
            producer.le.evaluate = original_evaluate
        diagnostic_path = os.path.join(out_dir, "failure_diagnostic.json")
        with open(diagnostic_path, encoding="utf-8") as fh:
            diagnostic = json.load(fh)
        check("setup failure writes classified zero-row evidence",
              (refused and diagnostic["failure_stage"] ==
               "setup_or_lifecycle_exception"
               and diagnostic["evaluated_rows"] == 0
               and diagnostic["captured_predictions"] == 0
               and diagnostic["rows"] == []
               and "SetupFailure: SETUP_ERROR: deliberate fixture" in
                   diagnostic["lifecycle_provenance"]["setup_error"]
               and diagnostic["lifecycle_provenance"]["candidate_sha256"] is None
               and diagnostic["lifecycle_provenance"]["setup_called"] is None
               and diagnostic["candidate_source"]["unchanged_after_evaluation"] is True
               and diagnostic["candidate_source"]["current_sha256"] ==
                   diagnostic["candidate_source"]["sha256"]
               and diagnostic["lifecycle_provenance"]["rows_hash"] ==
                   producer.le.rows_hash([])
               and diagnostic["lifecycle_spec"] ==
                   producer.dataclasses.asdict(producer.le.LifecycleSpec(
                       width=3, depth=2, residual_wall_time_limit_s=None))
               and not os.path.exists(os.path.join(out_dir, "predictions.npz"))
               and not os.path.exists(os.path.join(out_dir, "producer_receipt.json"))),
              "refused=%s rows=%d classified=%s" %
              (refused, len(diagnostic["rows"]), diagnostic["failure_stage"]))

    with _temp("p2-k4-producer-incomplete-rows-") as root:
        candidates = os.path.join(root, "candidates.json")
        bounds = os.path.join(root, "bounds.json")
        identities = os.path.join(root, "identities.json")
        out_dir = os.path.join(root, "observations")
        _write_json(candidates, {"parameters": ["lam"], "candidates": [{"lam": 0.0}]})
        _write_json(bounds, {"lam": {"min": 0.0, "max": 0.0}})
        _write_json(identities, {"kind": "synthetic-identities-v1", "width": 3,
                    "depth": 2, "identities": [
                        {"id": "incomplete-0", "seed": 7041},
                        {"id": "incomplete-1", "seed": 7042}]})
        original_evaluate = producer.le.evaluate
        def incomplete_rows(source_path, seeds, target, *, spec, gt_noise,
                            predictions_out):
            source_hash = producer._sha256(source_path)
            predictions_out.extend([
                np.zeros((spec.depth, spec.width), dtype=np.float32),
                np.zeros((spec.depth, spec.width), dtype=np.float32)])
            row = producer.le.EvidenceRow(
                index=0, input_seed=seeds[0],
                estimator_seed=producer.mint_provenance.seed_streams(seeds[0])[2],
                candidate=os.path.relpath(source_path, _ROOT).replace("\\", "/"),
                candidate_sha256=source_hash, truth_sha256="synthetic-zero",
                width=spec.width, depth=spec.depth, budget=float(spec.flop_budget),
                flops_used=1.0, n_ops=1.0, mse=0.0, zero_prediction_mse=0.0,
                gt_noise=0.0, residual_wall_time_s=0.0, wall_time_s=0.0)
            return [row], {"candidate": row.candidate,
                           "candidate_sha256": source_hash,
                           "truth_sha256": producer.le._array_sha(target),
                           "n_mlps": len(seeds),
                           "spec": producer.dataclasses.asdict(spec),
                           "round_tag": spec.round_tag,
                           "round_deviations": spec.deviations_from_round(),
                           "lifecycle": "grader",
                           "weights_mode": producer.le.DEFAULT_WEIGHTS_MODE,
                           "weights_mode_is_grader_default": True,
                           "setup_called": True,
                           "setup_seconds": 0.0,
                           "setup_timeout_s": spec.setup_timeout_s,
                           "setup_timed_out": False,
                           "setup_error": "",
                           "host_cold_start_s": producer.le.HOST_COLD_START_S,
                           "residual_measurement_path":
                               "runner=local (in-process BudgetContext)",
                           "residual_is_graded": False,
                           "objective_is_graded": False,
                           "backend": {
                               "flopscope": getattr(
                                   producer.le._require_real_flopscope(),
                                   "__version__", "?"),
                               "flopscope_path": os.path.dirname(
                                   producer.le._require_real_flopscope().__file__ or ""),
                               "numpy": np.__version__,
                               "python": sys.version.split()[0]},
                           "rows_hash": producer.le.rows_hash([row])}
        producer.le.evaluate = incomplete_rows
        try:
            try:
                producer.produce(Namespace(candidates=candidates, bounds=bounds,
                                           identities=identities, out_dir=out_dir))
                refused = False
            except producer.ProducerRefused as exc:
                refused = "row count 1 != identity count 2" in str(exc)
        finally:
            producer.le.evaluate = original_evaluate
        diagnostic_path = os.path.join(out_dir, "failure_diagnostic.json")
        with open(diagnostic_path, encoding="utf-8") as fh:
            diagnostic = json.load(fh)
        check("incomplete lifecycle rows can never produce success artifacts",
              (refused and diagnostic["failure_stage"] ==
               "incomplete_or_misbound_evidence"
               and diagnostic["evaluated_rows"] == 1
               and diagnostic["captured_predictions"] == 2
               and not os.path.exists(os.path.join(out_dir, "predictions.npz"))
               and not os.path.exists(os.path.join(out_dir, "producer_receipt.json"))),
              "refused=%s rows=%d predictions=%d" %
              (refused, diagnostic["evaluated_rows"],
               diagnostic["captured_predictions"]))

    with _temp("p2-k4-producer-forged-provenance-") as root:
        candidates = os.path.join(root, "candidates.json")
        bounds = os.path.join(root, "bounds.json")
        identities = os.path.join(root, "identities.json")
        out_dir = os.path.join(root, "observations")
        _write_json(candidates, {"parameters": ["lam"], "candidates": [{"lam": 0.0}]})
        _write_json(bounds, {"lam": {"min": 0.0, "max": 0.0}})
        _write_json(identities, {"kind": "synthetic-identities-v1", "width": 3,
                    "depth": 2, "identities": [{"id": "forged", "seed": 7051}]})
        original_evaluate = producer.le.evaluate
        def forged_provenance(source_path, seeds, target, *, spec, gt_noise,
                              predictions_out):
            source_hash = producer._sha256(source_path)
            candidate_path = os.path.relpath(source_path, _ROOT).replace("\\", "/")
            prediction = np.zeros(spec.width, dtype=np.float32)
            predictions_out.append(prediction)
            row = producer.le.EvidenceRow(
                index=0, input_seed=seeds[0],
                estimator_seed=producer.mint_provenance.seed_streams(seeds[0])[2],
                candidate=candidate_path, candidate_sha256=source_hash,
                truth_sha256=producer.le._array_sha(target),
                width=spec.width, depth=spec.depth, budget=float(spec.flop_budget),
                flops_used=1.0, n_ops=1.0, mse=0.0, zero_prediction_mse=0.0,
                gt_noise=0.0, residual_wall_time_s=0.0, wall_time_s=0.0)
            flops = producer.le._require_real_flopscope()
            provenance = {
                "candidate": candidate_path,
                "candidate_sha256": source_hash,
                "truth_sha256": producer.le._array_sha(target),
                "n_mlps": 1,
                "spec": producer.dataclasses.asdict(spec),
                "round_tag": "forged-round",
                "round_deviations": {},
                "lifecycle": "per_mlp_no_setup",
                "weights_mode": "f32",
                "weights_mode_is_grader_default": False,
                "setup_called": False,
                "setup_seconds": 0.0,
                "setup_timeout_s": spec.setup_timeout_s,
                "setup_timed_out": True,
                "setup_error": "forged clean setup",
                "host_cold_start_s": -1.0,
                "residual_measurement_path": "forged",
                "residual_is_graded": True,
                "objective_is_graded": True,
                "backend": {"flopscope": getattr(flops, "__version__", "?"),
                            "flopscope_path": "forged", "numpy": np.__version__,
                            "python": sys.version.split()[0]},
                "rows_hash": producer.le.rows_hash([row]),
            }
            return [row], provenance
        producer.le.evaluate = forged_provenance
        try:
            try:
                producer.produce(Namespace(candidates=candidates, bounds=bounds,
                                           identities=identities, out_dir=out_dir))
                refused = False
            except producer.ProducerRefused as exc:
                refused = "provenance round mismatch" in str(exc)
        finally:
            producer.le.evaluate = original_evaluate
        diagnostic_path = os.path.join(out_dir, "failure_diagnostic.json")
        with open(diagnostic_path, encoding="utf-8") as fh:
            diagnostic = json.load(fh)
        check("forged clean lifecycle provenance cannot produce success",
              (refused and diagnostic["failure_stage"] ==
               "incomplete_or_misbound_evidence"
               and diagnostic["lifecycle_provenance"]["lifecycle"] ==
                   "per_mlp_no_setup"
               and diagnostic["lifecycle_provenance"]["weights_mode"] == "f32"
               and not os.path.exists(os.path.join(out_dir, "predictions.npz"))
               and not os.path.exists(os.path.join(out_dir, "producer_receipt.json"))),
              "refused=%s forged_lifecycle=%s" %
              (refused, diagnostic["lifecycle_provenance"]["lifecycle"]))

    invalid_measurements = (
        ("forged estimator seed and fractional counts", 123, 1.5, 2.5,
         "estimator seed mismatch"),
        ("nonfinite operation count", None, 1.0, float("nan"),
         "FLOP/operation count"),
    )
    for label, estimator_seed, flops_used, n_ops, expected_error in invalid_measurements:
        with _temp("p2-k4-producer-invalid-row-") as root:
            candidates = os.path.join(root, "candidates.json")
            bounds = os.path.join(root, "bounds.json")
            identities = os.path.join(root, "identities.json")
            out_dir = os.path.join(root, "observations")
            seed = 7059
            _write_json(candidates, {"parameters": ["lam"],
                        "candidates": [{"lam": 0.0}]})
            _write_json(bounds, {"lam": {"min": 0.0, "max": 0.0}})
            _write_json(identities, {"kind": "synthetic-identities-v1",
                        "width": 3, "depth": 2,
                        "identities": [{"id": "invalid-row", "seed": seed}]})
            original_evaluate = producer.le.evaluate
            def invalid_row(source_path, seeds, target, *, spec, gt_noise,
                            predictions_out):
                source_hash = producer._sha256(source_path)
                candidate_path = os.path.relpath(source_path, _ROOT).replace("\\", "/")
                prediction = np.zeros(spec.width, dtype=np.float32)
                predictions_out.append(prediction)
                actual_estimator_seed = producer.mint_provenance.seed_streams(
                    seeds[0])[2]
                row = producer.le.EvidenceRow(
                    index=0, input_seed=seeds[0],
                    estimator_seed=(actual_estimator_seed if estimator_seed is None
                                    else estimator_seed),
                    candidate=candidate_path, candidate_sha256=source_hash,
                    truth_sha256=producer.le._array_sha(target),
                    width=spec.width, depth=spec.depth,
                    budget=float(spec.flop_budget), flops_used=flops_used,
                    n_ops=n_ops, mse=0.0, zero_prediction_mse=0.0,
                    gt_noise=0.0, residual_wall_time_s=0.0, wall_time_s=0.0)
                flops = producer.le._require_real_flopscope()
                provenance = {
                    "candidate": candidate_path,
                    "candidate_sha256": source_hash,
                    "truth_sha256": producer.le._array_sha(target),
                    "n_mlps": 1,
                    "spec": producer.dataclasses.asdict(spec),
                    "round_tag": spec.round_tag,
                    "round_deviations": spec.deviations_from_round(),
                    "lifecycle": "grader",
                    "weights_mode": producer.le.DEFAULT_WEIGHTS_MODE,
                    "weights_mode_is_grader_default": True,
                    "setup_called": True,
                    "setup_seconds": 0.0,
                    "setup_timeout_s": spec.setup_timeout_s,
                    "setup_timed_out": False,
                    "setup_error": "",
                    "host_cold_start_s": producer.le.HOST_COLD_START_S,
                    "residual_measurement_path":
                        "runner=local (in-process BudgetContext)",
                    "residual_is_graded": False,
                    "objective_is_graded": False,
                    "backend": {"flopscope": getattr(flops, "__version__", "?"),
                                "flopscope_path": os.path.dirname(flops.__file__ or ""),
                                "numpy": np.__version__,
                                "python": sys.version.split()[0]},
                    "rows_hash": producer.le.rows_hash([row]),
                }
                return [row], provenance
            producer.le.evaluate = invalid_row
            try:
                try:
                    producer.produce(Namespace(candidates=candidates, bounds=bounds,
                                               identities=identities, out_dir=out_dir))
                    refused = False
                except producer.ProducerRefused as exc:
                    refused = expected_error in str(exc)
            finally:
                producer.le.evaluate = original_evaluate
            diagnostic_path = os.path.join(out_dir, "failure_diagnostic.json")
            with open(diagnostic_path, encoding="utf-8") as fh:
                diagnostic = json.load(fh)
            check(label + " is diagnosed before conversion",
                  (refused and diagnostic["failure_stage"] ==
                   "incomplete_or_misbound_evidence"
                   and diagnostic["evaluated_rows"] == 1
                   and not os.path.exists(os.path.join(out_dir, "predictions.npz"))
                   and not os.path.exists(os.path.join(out_dir,
                                                       "producer_receipt.json"))),
                  "refused=%s flops=%r n_ops=%r" %
                  (refused, diagnostic["rows"][0]["flops_used"],
                   diagnostic["rows"][0]["n_ops"]))

    with _temp("p2-k4-producer-input-byte-swap-") as root:
        candidates = os.path.join(root, "candidates.json")
        bounds = os.path.join(root, "bounds.json")
        identities = os.path.join(root, "identities.json")
        out_dir = os.path.join(root, "observations")
        _write_json(candidates, {"parameters": ["lam"], "candidates": [{"lam": 0.0}]})
        _write_json(bounds, {"lam": {"min": 0.0, "max": 0.0}})
        _write_json(identities, {"kind": "synthetic-identities-v1", "width": 3,
                    "depth": 2, "identities": [{"id": "byte-swap", "seed": 7061}]})
        original_evaluate = producer.le.evaluate
        original_input_problem = producer._input_evidence_problem
        def swap_inputs_after_clean_evaluation(source_path, seeds, target, *, spec,
                                               gt_noise, predictions_out):
            return _mock_clean_evidence(
                source_path, seeds, target, spec, predictions_out)
        swapped = [False]
        def swap_before_input_revalidation(a, input_hashes):
            if not swapped[0]:
                for path in (candidates, bounds, identities):
                    with open(path, "a", encoding="utf-8") as fh:
                        fh.write("\n")
                swapped[0] = True
            return original_input_problem(a, input_hashes)
        producer.le.evaluate = swap_inputs_after_clean_evaluation
        producer._input_evidence_problem = swap_before_input_revalidation
        try:
            try:
                producer.produce(Namespace(candidates=candidates, bounds=bounds,
                                           identities=identities, out_dir=out_dir))
                refused = False
            except producer.ProducerRefused as exc:
                refused = "registry bytes changed" in str(exc)
        finally:
            producer.le.evaluate = original_evaluate
            producer._input_evidence_problem = original_input_problem
        with open(os.path.join(out_dir, "failure_diagnostic.json"),
                  encoding="utf-8") as fh:
            diagnostic = json.load(fh)
        check("semantic input byte replacement cannot produce success",
              (refused and diagnostic["failure_stage"] ==
               "incomplete_or_misbound_evidence"
               and all(not diagnostic["inputs"][name]
                           ["unchanged_after_evaluation"]
                       for name in ("candidates", "bounds", "identities"))
               and not os.path.exists(os.path.join(out_dir, "predictions.npz"))
               and not os.path.exists(os.path.join(out_dir,
                                                   "producer_receipt.json"))),
              "refused=%s changed=%s" %
              (refused, [name for name, value in diagnostic["inputs"].items()
                         if not value["unchanged_after_evaluation"]]))

    inconsistent_limits = (
        ("FLOP", {"flops_used": float(2 ** 41 + 1)}, "exceeds FLOP limit"),
        ("wall", {"wall_time_s": 121.0}, "exceeds wall limit"),
    )
    for label, changes, expected_error in inconsistent_limits:
        with _temp("p2-k4-producer-inconsistent-limit-") as root:
            candidates = os.path.join(root, "candidates.json")
            bounds = os.path.join(root, "bounds.json")
            identities = os.path.join(root, "identities.json")
            out_dir = os.path.join(root, "observations")
            _write_json(candidates, {"parameters": ["lam"],
                        "candidates": [{"lam": 0.0}]})
            _write_json(bounds, {"lam": {"min": 0.0, "max": 0.0}})
            _write_json(identities, {"kind": "synthetic-identities-v1",
                        "width": 3, "depth": 2,
                        "identities": [{"id": "limit", "seed": 7071}]})
            original_evaluate = producer.le.evaluate
            def inconsistent_limit(source_path, seeds, target, *, spec, gt_noise,
                                   predictions_out, _changes=changes):
                rows_out, provenance = _mock_clean_evidence(
                    source_path, seeds, target, spec, predictions_out)
                row = dataclasses.replace(rows_out[0], **_changes)
                provenance["rows_hash"] = producer.le.rows_hash([row])
                return [row], provenance
            producer.le.evaluate = inconsistent_limit
            try:
                try:
                    producer.produce(Namespace(candidates=candidates, bounds=bounds,
                                               identities=identities, out_dir=out_dir))
                    refused = False
                except producer.ProducerRefused as exc:
                    refused = expected_error in str(exc)
            finally:
                producer.le.evaluate = original_evaluate
            with open(os.path.join(out_dir, "failure_diagnostic.json"),
                      encoding="utf-8") as fh:
                diagnostic = json.load(fh)
            check("clean row above %s cap is diagnosed" % label,
                  (refused and diagnostic["failure_stage"] ==
                   "incomplete_or_misbound_evidence"
                   and not os.path.exists(os.path.join(out_dir, "predictions.npz"))
                   and not os.path.exists(os.path.join(out_dir,
                                                       "producer_receipt.json"))),
                  "refused=%s flags=%s" %
                  (refused, {name: diagnostic["rows"][0][name] for name in
                             ("budget_exhausted", "time_exhausted")}))

    with _temp("p2-k4-producer-none-provenance-") as root:
        candidates = os.path.join(root, "candidates.json")
        bounds = os.path.join(root, "bounds.json")
        identities = os.path.join(root, "identities.json")
        out_dir = os.path.join(root, "observations")
        _write_json(candidates, {"parameters": ["lam"], "candidates": [{"lam": 0.0}]})
        _write_json(bounds, {"lam": {"min": 0.0, "max": 0.0}})
        _write_json(identities, {"kind": "synthetic-identities-v1", "width": 3,
                    "depth": 2, "identities": [{"id": "none-prov", "seed": 7081}]})
        original_evaluate = producer.le.evaluate
        def none_provenance(source_path, seeds, target, *, spec, gt_noise,
                            predictions_out):
            rows_out, _ = _mock_clean_evidence(
                source_path, seeds, target, spec, predictions_out)
            return rows_out, None
        producer.le.evaluate = none_provenance
        try:
            try:
                producer.produce(Namespace(candidates=candidates, bounds=bounds,
                                           identities=identities, out_dir=out_dir))
                refused = False
            except producer.ProducerRefused as exc:
                refused = "provenance keys mismatch" in str(exc)
        finally:
            producer.le.evaluate = original_evaluate
        with open(os.path.join(out_dir, "failure_diagnostic.json"),
                  encoding="utf-8") as fh:
            diagnostic = json.load(fh)
        check("missing provenance is diagnosed without a serializer crash",
              (refused and diagnostic["lifecycle_provenance"] is None
               and diagnostic["failure_stage"] ==
                   "incomplete_or_misbound_evidence"
               and not os.path.exists(os.path.join(out_dir, "predictions.npz"))
               and not os.path.exists(os.path.join(out_dir,
                                                   "producer_receipt.json"))),
              "refused=%s provenance=%r" %
              (refused, diagnostic["lifecycle_provenance"]))

    with _temp("p2-k4-producer-bool-provenance-") as root:
        candidates = os.path.join(root, "candidates.json")
        bounds = os.path.join(root, "bounds.json")
        identities = os.path.join(root, "identities.json")
        out_dir = os.path.join(root, "observations")
        _write_json(candidates, {"parameters": ["lam"],
                    "candidates": [{"lam": 0.0}]})
        _write_json(bounds, {"lam": {"min": 0.0, "max": 0.0}})
        _write_json(identities, {"kind": "synthetic-identities-v1", "width": 3,
                    "depth": 2,
                    "identities": [{"id": "bool-prov", "seed": 7082}]})
        original_evaluate = producer.le.evaluate
        def boolean_identity_count(source_path, seeds, target, *, spec, gt_noise,
                                   predictions_out):
            rows_out, provenance = _mock_clean_evidence(
                source_path, seeds, target, spec, predictions_out)
            provenance["n_mlps"] = True
            return rows_out, provenance
        producer.le.evaluate = boolean_identity_count
        try:
            try:
                producer.produce(Namespace(candidates=candidates, bounds=bounds,
                                           identities=identities,
                                           out_dir=out_dir))
                refused = False
            except producer.ProducerRefused as exc:
                refused = "identity count mismatch" in str(exc)
        finally:
            producer.le.evaluate = original_evaluate
        with open(os.path.join(out_dir, "failure_diagnostic.json"),
                  encoding="utf-8") as fh:
            diagnostic = json.load(fh)
        check("boolean lifecycle identity count cannot equal integer one",
              (refused and diagnostic["lifecycle_provenance"]["n_mlps"] is True
               and not os.path.exists(os.path.join(out_dir, "predictions.npz"))
               and not os.path.exists(os.path.join(out_dir,
                                                   "producer_receipt.json"))),
              "refused=%s type=%s" %
              (refused, type(diagnostic["lifecycle_provenance"]
                             ["n_mlps"]).__name__))

    with _temp("p2-k4-producer-prior-source-race-") as root:
        candidates = os.path.join(root, "candidates.json")
        bounds = os.path.join(root, "bounds.json")
        identities = os.path.join(root, "identities.json")
        out_dir = os.path.join(root, "observations")
        _write_json(candidates, {"parameters": ["lam"], "candidates": [
                    {"lam": 0.0}, {"lam": 0.25}]})
        _write_json(bounds, {"lam": {"min": 0.0, "max": 0.25}})
        _write_json(identities, {"kind": "synthetic-identities-v1", "width": 3,
                    "depth": 2,
                    "identities": [{"id": "source-race", "seed": 7091}]})
        original_evaluate = producer.le.evaluate
        calls = [0]
        def mutate_prior_source(source_path, seeds, target, *, spec, gt_noise,
                                predictions_out):
            calls[0] += 1
            result = _mock_clean_evidence(
                source_path, seeds, target, spec, predictions_out)
            if calls[0] == 2:
                prior = os.path.join(out_dir,
                                     "candidate_000_coherent.py")
                with open(prior, "a", encoding="utf-8") as fh:
                    fh.write("\n# later-candidate mutation\n")
            return result
        producer.le.evaluate = mutate_prior_source
        try:
            try:
                producer.produce(Namespace(candidates=candidates, bounds=bounds,
                                           identities=identities,
                                           out_dir=out_dir))
                refused = False
            except producer.ProducerRefused as exc:
                refused = "candidate bytes changed" in str(exc)
        finally:
            producer.le.evaluate = original_evaluate
        with open(os.path.join(out_dir, "failure_diagnostic.json"),
                  encoding="utf-8") as fh:
            diagnostic = json.load(fh)
        check("producer closing recheck catches prior candidate source mutation",
              (refused and diagnostic["failure_stage"] ==
               "dependency_changed_before_success_write"
               and diagnostic["candidate_index"] == 0
               and not diagnostic["candidate_source"]
                   ["unchanged_after_evaluation"]
               and not os.path.exists(os.path.join(out_dir, "predictions.npz"))
               and not os.path.exists(os.path.join(out_dir,
                                                   "producer_receipt.json"))),
              "refused=%s current_match=%s" %
              (refused, diagnostic["candidate_source"]
               ["unchanged_after_evaluation"]))

    with _temp("p2-k4-producer-registry-closing-race-") as root:
        candidates = os.path.join(root, "candidates.json")
        bounds = os.path.join(root, "bounds.json")
        identities = os.path.join(root, "identities.json")
        out_dir = os.path.join(root, "observations")
        _write_json(candidates, {"parameters": ["lam"],
                    "candidates": [{"lam": 0.0}]})
        _write_json(bounds, {"lam": {"min": 0.0, "max": 0.0}})
        _write_json(identities, {"kind": "synthetic-identities-v1", "width": 3,
                    "depth": 2,
                    "identities": [{"id": "registry-race", "seed": 7092}]})
        original_evaluate = producer.le.evaluate
        original_observed = producer._observed_binding
        mutated = [False]
        denied = [False]
        producer.le.evaluate = lambda source_path, seeds, target, *, spec, gt_noise, predictions_out: _mock_clean_evidence(source_path, seeds, target, spec, predictions_out)
        def mutate_registry_during_source_scan(path, frozen_sha256):
            result = original_observed(path, frozen_sha256)
            if (not mutated[0] and
                    os.path.basename(path) == "candidate_000_coherent.py"):
                mutated[0] = True
                try:
                    with open(candidates, "a", encoding="utf-8") as fh:
                        fh.write("\n")
                except PermissionError:
                    denied[0] = True
            return result
        producer._observed_binding = mutate_registry_during_source_scan
        try:
            try:
                result = producer.produce(Namespace(
                    candidates=candidates, bounds=bounds,
                    identities=identities, out_dir=out_dir))
                accepted = True
            except producer.ProducerRefused:
                accepted = False
        finally:
            producer.le.evaluate = original_evaluate
            producer._observed_binding = original_observed
        check("producer lock blocks registry write during source scan",
              (accepted and mutated[0] and denied[0]
               and producer._sha256(candidates) ==
                   result["manifest"]["input_bindings"]["candidates"]
                   ["current_sha256"]
               and os.path.exists(os.path.join(out_dir, "predictions.npz"))
               and os.path.exists(os.path.join(out_dir,
                                               "producer_receipt.json"))),
              "accepted=%s denied=%s" % (accepted, denied[0]))

    with _temp("p2-k4-producer-loader-source-lock-") as root:
        candidates = os.path.join(root, "candidates.json")
        bounds = os.path.join(root, "bounds.json")
        identities = os.path.join(root, "identities.json")
        out_dir = os.path.join(root, "observations")
        _write_json(candidates, {"parameters": ["lam"],
                    "candidates": [{"lam": 0.0}]})
        _write_json(bounds, {"lam": {"min": 0.0, "max": 0.0}})
        _write_json(identities, {"kind": "synthetic-identities-v1", "width": 3,
                    "depth": 2,
                    "identities": [{"id": "loader-lock", "seed": 7093}]})
        original_loader = producer.le.load_candidate
        attempted = [False]
        denied = [False]
        def attempt_source_swap_before_load(path):
            attempted[0] = True
            try:
                with open(path, "r+b") as fh:
                    raw = fh.read()
                    replacement = raw.replace(b"_K4_LAM = 0.0",
                                              b"_K4_LAM = 0.25")
                    fh.seek(0)
                    fh.write(replacement)
                    fh.truncate()
            except PermissionError:
                denied[0] = True
            return original_loader(path)
        producer.le.load_candidate = attempt_source_swap_before_load
        try:
            result = producer.produce(Namespace(
                candidates=candidates, bounds=bounds,
                identities=identities, out_dir=out_dir))
        finally:
            producer.le.load_candidate = original_loader
        source_path = os.path.join(_ROOT, result["candidate_sources"][0])
        check("producer lifecycle lock blocks source swap before module load",
              (attempted[0] and denied[0]
               and producer._sha256(source_path) ==
                   result["candidate_source_hashes"][0]
               and result["manifest"]["lifecycle_provenance"][0]
                   ["candidate_sha256"] == result["candidate_source_hashes"][0]),
              "attempted=%s denied=%s" % (attempted[0], denied[0]))
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
