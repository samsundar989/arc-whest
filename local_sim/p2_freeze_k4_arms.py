"""Freeze a protected P07 K4 fit selection into four immutable mechanism arms.

This bridge reads no research labels.  It accepts only the strict receipt emitted
by ``p2_k4_fit.py`` for the registered protected-design P07 fit, revalidates all
of that receipt's nontruth dependencies, executes the exact locked generator
bytes, and creates a fresh output directory containing four arms and a receipt.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import ctypes
import hashlib
import json
import math
import os
import re
import shutil


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_GENERATOR = os.path.join(_HERE, "p2_build_k4_arms.py")
_TEMPLATE = os.path.join(_HERE, "p2_compact_candidate_template.py")
_FIT = os.path.join(_HERE, "p2_k4_fit.py")
_PRODUCER = os.path.join(_HERE, "p2_k4_fit_producer.py")
_TRUTH_READS_SOURCE = os.path.join(_HERE, "truth_reads.py")
_TRUTH_READS_LEDGER = os.path.join(_ROOT, "experiments", "truth_reads.jsonl")
_DEFAULT_ARMS = os.path.join(_HERE, "p2_arms")
_ARMS = ("exact", "coherent", "consumers_only", "births_only")
_HASH = re.compile(r"[0-9a-f]{64}")
_FIT_SCHEMA = "arc-whest.coherent-k4-fit-receipt.v1"
_FREEZE_SCHEMA = "arc-whest.coherent-k4-arm-freeze-receipt.v1"


class FreezeRefused(ValueError):
    pass


def _sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def _read_bytes(path, label):
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise FreezeRefused("cannot read %s: %s" % (label, exc)) from exc
    return raw, _sha256_bytes(raw)


def _read_json(path, label):
    raw, sha = _read_bytes(path, label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FreezeRefused("invalid %s JSON: %s" % (label, exc)) from exc
    return value, sha


def _exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise FreezeRefused("%s keys must be exactly %s, got %s" %
                            (label, sorted(keys), got))


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _valid_hash(value):
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _repo_path(raw, label):
    if not isinstance(raw, str) or not raw:
        raise FreezeRefused("%s must be a nonempty path string" % label)
    path = os.path.realpath(os.path.abspath(
        raw if os.path.isabs(raw) else os.path.join(_ROOT, raw)))
    try:
        inside = os.path.commonpath([os.path.realpath(_ROOT), path]) == os.path.realpath(_ROOT)
    except ValueError:
        inside = False
    if not inside:
        raise FreezeRefused("%s escapes the repository" % label)
    return path


def _relative(path):
    return os.path.relpath(os.path.abspath(path), _ROOT).replace("\\", "/")


def _same_path(left, right):
    return os.path.normcase(os.path.realpath(left)) == os.path.normcase(os.path.realpath(right))


@contextlib.contextmanager
def _stable_files(paths):
    unique, seen = [], set()
    for raw in paths:
        path = os.path.normcase(os.path.realpath(os.path.abspath(raw)))
        if path not in seen:
            seen.add(path)
            unique.append(path)
    if os.name == "nt":
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel32.CreateFileW
        create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                           wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                           wintypes.HANDLE)
        create.restype = wintypes.HANDLE
        close = kernel32.CloseHandle
        close.argtypes = (wintypes.HANDLE,)
        handles = []
        try:
            for path in unique:
                handle = create(path, 0x80000000, 0x00000001, None, 3,
                                0x00000080, None)
                if handle == wintypes.HANDLE(-1).value:
                    raise FreezeRefused("cannot lock dependency %s: WinError %d" %
                                        (path, ctypes.get_last_error()))
                handles.append(handle)
            yield
        finally:
            for handle in reversed(handles):
                close(handle)
        return
    import fcntl
    files = []
    try:
        for path in unique:
            fh = open(path, "rb")
            files.append(fh)
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    except (OSError, BlockingIOError) as exc:
        raise FreezeRefused("cannot lock freeze dependencies: %s" % exc) from exc
    finally:
        for fh in reversed(files):
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()


def _binding_path(binding, label):
    _exact(binding, {"path", "sha256"}, label)
    if not _valid_hash(binding["sha256"]):
        raise FreezeRefused("%s sha256 is invalid" % label)
    return _repo_path(binding["path"], label)


def _parse_constants(raw, path):
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=path)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise FreezeRefused("arm source does not parse: %s" % exc) from exc
    values = {}
    wanted = {"_K4_ACTIVE", "_K4_CONSUMERS", "_K4_BIRTHS", "_K4_LAM"}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Name) and
                node.targets[0].id in wanted):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError as exc:
                raise FreezeRefused("arm constant is not literal") from exc
    if set(values) != wanted:
        raise FreezeRefused("arm constants are incomplete")
    return values


def _registry(candidate_path, bounds_path):
    raw_candidates, candidate_sha = _read_json(candidate_path, "candidate registry")
    raw_bounds, bounds_sha = _read_json(bounds_path, "bounds")
    _exact(raw_candidates, {"parameters", "candidates"}, "candidate registry")
    if raw_candidates["parameters"] != ["lam"]:
        raise FreezeRefused("candidate registry must expose exactly lam")
    candidates = raw_candidates["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise FreezeRefused("candidate registry must be nonempty")
    _exact(raw_bounds, {"lam"}, "bounds")
    _exact(raw_bounds["lam"], {"min", "max"}, "lam bounds")
    lo, hi = raw_bounds["lam"]["min"], raw_bounds["lam"]["max"]
    if not _number(lo) or not _number(hi) or not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or float(lo) > float(hi):
        raise FreezeRefused("lam bounds must be ordered finite JSON numbers")
    normalized = []
    for index, candidate in enumerate(candidates):
        _exact(candidate, {"lam"}, "candidate %d" % index)
        value = candidate["lam"]
        if (not _number(value) or not math.isfinite(float(value)) or
                not float(lo) <= float(value) <= float(hi)):
            raise FreezeRefused("candidate %d lam is invalid or outside bounds" % index)
        normalized.append({"lam": float(value)})
    if len({row["lam"] for row in normalized}) != len(normalized):
        raise FreezeRefused("candidate registry contains duplicate lam values")
    return normalized, {"lam": {"min": float(lo), "max": float(hi)}}, candidate_sha, bounds_sha


def _identities(path):
    raw, sha = _read_json(path, "identity registry")
    _exact(raw, {"kind", "block", "width", "depth", "identities"},
           "identity registry")
    if (raw["kind"] != "registered-design-identities-v1" or
            raw["block"] != "P07" or raw["width"] != 1024 or raw["depth"] != 16):
        raise FreezeRefused("identity registry must be registered P07 at 1024x16")
    rows = raw["identities"]
    expected = list(range(40_910_007_000, 40_910_007_384))
    if (not isinstance(rows, list) or len(rows) != len(expected) or
            any(not isinstance(row, dict) or set(row) != {"id", "seed"}
                or row["seed"] != seed or row["id"] != str(seed)
                for row, seed in zip(rows, expected))):
        raise FreezeRefused("identity registry does not exactly match ordered P07")
    return raw, [str(seed) for seed in expected], sha


def _initial_fit_paths(fit_path, receipt):
    """Discover lock targets from untrusted input; validation happens later."""
    paths = [fit_path, os.path.abspath(__file__), _FIT, _PRODUCER,
             _GENERATOR, _TEMPLATE, _TRUTH_READS_SOURCE]
    if isinstance(receipt, dict):
        for value in (receipt.get("inputs") or {}).values():
            if isinstance(value, dict) and isinstance(value.get("path"), str):
                # Truth is deliberately metadata-only: never open or lock labels.
                if value is not (receipt.get("inputs") or {}).get("truth"):
                    paths.append(_repo_path(value["path"], "fit input"))
        for value in (receipt.get("source_hashes") or {}).values():
            if isinstance(value, dict) and isinstance(value.get("path"), str):
                paths.append(_repo_path(value["path"], "fit source"))
        for value in receipt.get("candidate_sources") or []:
            if isinstance(value, dict) and isinstance(value.get("path"), str):
                paths.append(_repo_path(value["path"], "candidate source"))
    return paths


def _row_sha(row):
    raw = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(raw)


def _reservation_evidence(embedded):
    """Prove the embedded completion exists behind its earlier reservation."""
    try:
        rows = []
        with open(_TRUTH_READS_LEDGER, encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                row = json.loads(line)
                if row.get("reservation") == embedded["reservation"]:
                    rows.append((line_number, row))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeRefused("cannot validate live truth-read ledger: %s" % exc) from exc
    reserved = [(line, row) for line, row in rows if row.get("status") == "reserved"]
    completed = [(line, row) for line, row in rows if row.get("status") == "completed"]
    if len(rows) != 2 or len(reserved) != 1 or len(completed) != 1:
        raise FreezeRefused(
            "live truth-read ledger must contain exactly one reserved and one completed row for the fit reservation")
    reserved_line, reserved_row = reserved[0]
    completed_line, completed_row = completed[0]
    if reserved_line >= completed_line:
        raise FreezeRefused("completed truth-read row does not follow its reservation")
    if completed_row != embedded:
        raise FreezeRefused("embedded completed reservation does not exactly match the live ledger")
    immutable = {"block", "artifact", "artifact_sha256", "consumer",
                 "instrument_sha256", "purpose", "ts", "reservation",
                 "result_artifact", "note", "backfilled", "timestamp_source",
                 "receipt_created_utc"}
    if (not isinstance(reserved_row, dict) or set(reserved_row) != set(embedded) or
            any(reserved_row.get(key) != embedded.get(key) for key in immutable) or
            reserved_row.get("receipt_completed_utc") != "" or
            reserved_row.get("elapsed_seconds") is not None):
        raise FreezeRefused("reserved truth-read row conflicts with its completed row")
    return {"ledger_path": _relative(_TRUTH_READS_LEDGER),
            "reservation": embedded["reservation"],
            "reserved_row_sha256": _row_sha(reserved_row),
            "completed_row_sha256": _row_sha(completed_row)}


def _load_generator(raw):
    namespace = {"__name__": "_locked_p2_build_k4_arms",
                 "__file__": _GENERATOR, "__package__": None}
    try:
        code = compile(raw, _GENERATOR, "exec", dont_inherit=True)
        exec(code, namespace)
    except Exception as exc:
        raise FreezeRefused("locked generator bytes cannot execute: %s" % exc) from exc
    for name in ("ARMS", "SOURCE", "build"):
        if name not in namespace:
            raise FreezeRefused("locked generator is missing %s" % name)
    if not _same_path(namespace["SOURCE"], _TEMPLATE):
        raise FreezeRefused("locked generator names the wrong template")
    if tuple(row[0] for row in namespace["ARMS"]) != _ARMS:
        raise FreezeRefused("locked generator arm registry is wrong")
    return namespace


def _validate_fit_locked(fit_path):
    receipt, fit_sha = _read_json(fit_path, "fit receipt")
    keys = {"schema", "immutable", "data_role", "objective", "parameter_names",
            "generator_binding", "candidate_registry", "bounds", "identities",
            "truth_manifest", "reservation_receipt", "inputs", "source_hashes",
            "candidate_sources", "counted_costs", "results", "selection",
            "validation"}
    _exact(receipt, keys, "fit receipt")
    if receipt["schema"] != _FIT_SCHEMA or receipt["immutable"] is not True:
        raise FreezeRefused("fit receipt schema/immutability is invalid")
    if receipt["data_role"] != "protected-design":
        raise FreezeRefused("fit receipt must have protected-design data role")
    objective = receipt["objective"]
    _exact(objective, {"name", "direction", "definition", "validity"}, "objective")
    if objective != {"name": "final_layer_mse", "direction": "minimize",
                      "definition": "mean identity final-layer neuron MSE",
                      "validity": "registered coefficient candidates have identical counted cost for every identity"}:
        raise FreezeRefused("fit objective is not the registered final-layer MSE")
    if receipt["parameter_names"] != ["lam"] or receipt["generator_binding"] != {
            "lam": "generated arm module constant _K4_LAM", "arms": list(_ARMS)}:
        raise FreezeRefused("fit generator binding is invalid")

    inputs = receipt["inputs"]
    _exact(inputs, {"predictions", "producer_receipt", "truth", "candidates",
                    "bounds", "identities"}, "fit inputs")
    paths = {}
    for name, binding in inputs.items():
        paths[name] = _binding_path(binding, "fit input %s" % name)
        if name != "truth":
            _raw, actual = _read_bytes(paths[name], "fit input %s" % name)
            if actual != binding["sha256"]:
                raise FreezeRefused("fit input %s hash mismatch" % name)
    candidates, bounds, candidate_sha, bounds_sha = _registry(paths["candidates"], paths["bounds"])
    identity_manifest, identities, identity_sha = _identities(paths["identities"])
    if (receipt["candidate_registry"] != candidates or receipt["bounds"] != bounds or
            receipt["identities"] != identities or
            inputs["candidates"]["sha256"] != candidate_sha or
            inputs["bounds"]["sha256"] != bounds_sha or
            inputs["identities"]["sha256"] != identity_sha):
        raise FreezeRefused("fit registry/bounds/identity bindings do not match current bytes")

    source_paths = {"fit_interface": _FIT, "fit_producer": _PRODUCER,
                    "arm_generator": _GENERATOR, "arm_template": _TEMPLATE}
    source_paths.update({"arm_" + name: os.path.join(_DEFAULT_ARMS, "arm_%s.py" % name)
                         for name in _ARMS})
    sources = receipt["source_hashes"]
    if not isinstance(sources, dict) or set(sources) != set(source_paths):
        raise FreezeRefused("fit source bindings are incomplete")
    for name, expected_path in source_paths.items():
        path = _binding_path(sources[name], "source %s" % name)
        if not _same_path(path, expected_path):
            raise FreezeRefused("source %s has the wrong registered path" % name)
        raw, actual = _read_bytes(path, "source %s" % name)
        if actual != sources[name]["sha256"]:
            raise FreezeRefused("source %s no longer matches fit receipt" % name)
        try:
            ast.parse(raw.decode("utf-8"), filename=path)
        except (UnicodeDecodeError, SyntaxError) as exc:
            raise FreezeRefused("source %s does not parse" % name) from exc

    generator_raw, generator_sha = _read_bytes(_GENERATOR, "arm generator")
    template_raw, template_sha = _read_bytes(_TEMPLATE, "arm template")
    if generator_sha != sources["arm_generator"]["sha256"] or template_sha != sources["arm_template"]["sha256"]:
        raise FreezeRefused("generator/template binding mismatch")
    generator = _load_generator(generator_raw)
    template_text = template_raw.decode("utf-8")

    candidate_sources = receipt["candidate_sources"]
    if not isinstance(candidate_sources, list) or len(candidate_sources) != len(candidates):
        raise FreezeRefused("fit candidate source coverage is invalid")
    coherent = next(row for row in generator["ARMS"] if row[0] == "coherent")
    for index, (binding, candidate) in enumerate(zip(candidate_sources, candidates)):
        _exact(binding, {"path", "sha256", "verified"}, "candidate source %d" % index)
        if binding["verified"] is not True or not _valid_hash(binding["sha256"]):
            raise FreezeRefused("candidate source %d binding is invalid" % index)
        path = _repo_path(binding["path"], "candidate source %d" % index)
        raw, actual = _read_bytes(path, "candidate source %d" % index)
        expected = generator["build"](*coherent, template_text, template_sha,
                                      candidate["lam"]).encode("utf-8")
        if actual != binding["sha256"] or raw != expected:
            raise FreezeRefused("candidate source %d is not its exact registered regeneration" % index)

    n, c = len(identities), len(candidates)
    costs = receipt["counted_costs"]
    if (not isinstance(costs, list) or len(costs) != c or
            any(not isinstance(row, list) or len(row) != n or
                any(not _is_int(x) or x < 0 for x in row) for row in costs) or
            any(row != costs[0] for row in costs[1:])):
        raise FreezeRefused("fit counted costs do not prove value-independent candidates")
    results = receipt["results"]
    if not isinstance(results, list) or len(results) != c:
        raise FreezeRefused("fit results must cover every candidate")
    objectives = []
    for index, (row, candidate) in enumerate(zip(results, candidates)):
        _exact(row, {"candidate", "objective", "per_identity"}, "result %d" % index)
        value, per_identity = row["objective"], row["per_identity"]
        if (row["candidate"] != candidate or not _number(value) or
                not math.isfinite(float(value)) or float(value) < 0 or
                not isinstance(per_identity, list) or len(per_identity) != n or
                any(not _number(x) or not math.isfinite(float(x)) or float(x) < 0
                    for x in per_identity)):
            raise FreezeRefused("fit result %d is invalid" % index)
        recomputed = math.fsum(float(x) for x in per_identity) / n
        if not math.isclose(float(value), recomputed, rel_tol=2e-15, abs_tol=2e-15):
            raise FreezeRefused("fit result %d objective does not equal its arithmetic mean" % index)
        objectives.append(float(value))

    selection = receipt["selection"]
    _exact(selection, {"registered_index", "frozen_values", "objective", "tie_break"},
           "fit selection")
    selected = selection["registered_index"]
    first_best = min(range(c), key=lambda i: objectives[i])
    if (not _is_int(selected) or not 0 <= selected < c or selected != first_best or
            selection["frozen_values"] != candidates[selected] or
            not _number(selection["objective"]) or
            float(selection["objective"]) != objectives[selected] or
            selection["tie_break"] != "first in explicitly registered order"):
        raise FreezeRefused("fit selected index/value/objective are inconsistent")

    truth_manifest = receipt["truth_manifest"]
    _exact(truth_manifest, {"kind", "block", "replicas", "n_samples"}, "truth manifest")
    if (truth_manifest["kind"] != "protected-design" or truth_manifest["block"] != "P07" or
            truth_manifest["replicas"] != 2 or
            (truth_manifest["n_samples"] is not None and
             (not _is_int(truth_manifest["n_samples"]) or truth_manifest["n_samples"] <= 0))):
        raise FreezeRefused("truth manifest is not completed protected P07 evidence")
    reservation = receipt["reservation_receipt"]
    reservation_keys = {"block", "artifact", "artifact_sha256", "consumer",
                        "instrument_sha256", "purpose", "ts", "status",
                        "reservation", "result_artifact", "note", "backfilled",
                        "timestamp_source", "receipt_created_utc",
                        "receipt_completed_utc", "elapsed_seconds"}
    _exact(reservation, reservation_keys, "reservation receipt")
    truth_rel = _relative(paths["truth"])
    result_path = _repo_path(reservation["result_artifact"],
                             "reservation result artifact")
    if (reservation["block"] != "P07" or reservation["artifact"] != truth_rel or
            reservation["artifact_sha256"] != inputs["truth"]["sha256"] or
            reservation["consumer"] != "local_sim/p2_k4_fit.py" or
            reservation["instrument_sha256"] != sources["fit_interface"]["sha256"] or
            reservation["purpose"] != "bounded coherent-K4 coefficient fit" or
            reservation["status"] != "completed" or
            not isinstance(reservation["reservation"], str) or not reservation["reservation"] or
            not _same_path(result_path, fit_path) or
            reservation["backfilled"] is not False or
            reservation["timestamp_source"] not in {"automatic_runtime", "supplied_historical"} or
            not isinstance(reservation["receipt_created_utc"], str) or not reservation["receipt_created_utc"] or
            not isinstance(reservation["receipt_completed_utc"], str) or not reservation["receipt_completed_utc"] or
             not _number(reservation["elapsed_seconds"]) or float(reservation["elapsed_seconds"]) < 0):
        raise FreezeRefused("fit receipt lacks matching completed P07 reservation evidence")
    reservation_evidence = _reservation_evidence(reservation)
    truth_reads_raw, truth_reads_sha = _read_bytes(
        _TRUTH_READS_SOURCE, "truth-read instrument source")
    try:
        ast.parse(truth_reads_raw.decode("utf-8"), filename=_TRUTH_READS_SOURCE)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise FreezeRefused("truth-read instrument source does not parse") from exc

    validation = receipt["validation"]
    _exact(validation, {"completed_before_truth_load", "producer_receipt_revalidated_before_truth_load",
                        "candidate_count", "identity_count", "all_finite"}, "fit validation")
    if validation != {"completed_before_truth_load": True,
                      "producer_receipt_revalidated_before_truth_load": True,
                      "candidate_count": c, "identity_count": n, "all_finite": True}:
        raise FreezeRefused("fit validation evidence is incomplete")
    return {"receipt": receipt, "sha256": fit_sha, "paths": paths,
            "candidates": candidates, "bounds": bounds,
            "identity_manifest": identity_manifest, "sources": sources,
            "generator": generator, "generator_raw": generator_raw,
            "generator_sha": generator_sha, "template_raw": template_raw,
            "template_sha": template_sha, "selected": selected,
            "lam": candidates[selected]["lam"],
            "reservation_evidence": reservation_evidence,
            "extra_sources": {"truth_reads_interface": {
                "path": _relative(_TRUTH_READS_SOURCE), "sha256": truth_reads_sha}}}


def _validated_fit(fit_path):
    fit_path = _repo_path(fit_path, "fit receipt")
    initial, _ = _read_json(fit_path, "fit receipt")
    paths = _initial_fit_paths(fit_path, initial)
    with _stable_files(paths):
        with _stable_files([_TRUTH_READS_LEDGER]):
            validated = _validate_fit_locked(fit_path)
        yield_paths = list(paths)
        # The validated set must equal or be contained in the initially locked set.
        required = _initial_fit_paths(fit_path, validated["receipt"])
        normalized = {os.path.normcase(os.path.realpath(x)) for x in yield_paths}
        if any(os.path.normcase(os.path.realpath(x)) not in normalized for x in required):
            raise FreezeRefused("fit receipt changed its dependency set before validation")
        yield validated


@contextlib.contextmanager
def validated_fit_receipt(fit_path):
    """Yield a strict P07 fit receipt while every nontruth dependency is locked."""
    yield from _validated_fit(fit_path)


def _write_exclusive(path, raw):
    try:
        with open(path, "xb") as fh:
            fh.write(raw)
    except FileExistsError as exc:
        raise FreezeRefused("output already exists: %s" % path) from exc


def freeze(a):
    fit_path = _repo_path(a.fit_receipt, "fit receipt")
    out_dir = _repo_path(a.out_dir, "output directory")
    parent = os.path.dirname(out_dir)
    if not os.path.isdir(parent):
        raise FreezeRefused("output parent directory does not exist: %s" % parent)
    if os.path.lexists(out_dir):
        raise FreezeRefused("output directory already exists: %s" % out_dir)
    created = False
    try:
        with validated_fit_receipt(fit_path) as valid:
            os.mkdir(out_dir)
            created = True
            lam = valid["lam"]
            template_text = valid["template_raw"].decode("utf-8")
            arm_rows = []
            for index, arm in enumerate(valid["generator"]["ARMS"]):
                name, consumers, births, blurb = arm
                rendered = valid["generator"]["build"](
                    name, consumers, births, blurb, template_text,
                    valid["template_sha"], lam).encode("utf-8")
                path = os.path.join(out_dir, "arm_%s.py" % name)
                _write_exclusive(path, rendered)
                constants = _parse_constants(rendered, path)
                expected_constants = {"_K4_ACTIVE": bool(consumers or births),
                                      "_K4_CONSUMERS": bool(consumers),
                                      "_K4_BIRTHS": bool(births),
                                      "_K4_LAM": lam}
                if constants != expected_constants:
                    raise FreezeRefused("generated arm %s constants are wrong" % name)
                arm_rows.append({"index": index, "name": name,
                                 "path": _relative(path),
                                 "sha256": _sha256_bytes(rendered),
                                 "constants": constants})

            freezer_raw, freezer_sha = _read_bytes(os.path.abspath(__file__), "freezer source")
            source_bindings = dict(valid["sources"])
            source_bindings.update(valid["extra_sources"])
            source_bindings["arm_freezer"] = {"path": _relative(os.path.abspath(__file__)),
                                               "sha256": freezer_sha}
            nontruth_inputs = {name: valid["receipt"]["inputs"][name]
                               for name in ("predictions", "producer_receipt", "candidates",
                                            "bounds", "identities")}
            payload = {
                "schema": _FREEZE_SCHEMA, "immutable": True, "labels_read": False,
                "data_role": "protected-design",
                "objective": valid["receipt"]["objective"],
                "fit_receipt": {"path": _relative(fit_path), "sha256": valid["sha256"]},
                "selection": {"registered_index": valid["selected"],
                              "frozen_values": {"lam": lam}, "k4_lam": lam},
                "generator": {"path": _relative(_GENERATOR),
                              "sha256": valid["generator_sha"],
                              "executed_sha256": _sha256_bytes(valid["generator_raw"])},
                "template": {"path": _relative(_TEMPLATE), "sha256": valid["template_sha"]},
                "reservation_evidence": valid["reservation_evidence"],
                "input_bindings": nontruth_inputs,
                "source_bindings": source_bindings,
                "arms": arm_rows,
                "validation": {"protected_p07_fit_validated": True,
                               "completed_reservation_validated": True,
                               "selection_matches_registered_candidate": True,
                               "generator_snapshot_executed": True,
                               "dependencies_locked_through_publication": True,
                               "truth_opened_by_freezer": False}}
            receipt_path = os.path.join(out_dir, "freeze_receipt.json")
            arm_paths = [os.path.join(_ROOT, row["path"].replace("/", os.sep))
                         for row in arm_rows]
            with _stable_files(arm_paths):
                for row, path in zip(arm_rows, arm_paths):
                    if _read_bytes(path, "frozen arm")[1] != row["sha256"]:
                        raise FreezeRefused("frozen arm changed before publication")
                raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
                _write_exclusive(receipt_path, raw)  # publication is last
                with _stable_files([receipt_path]):
                    if _read_bytes(receipt_path, "freeze receipt")[0] != raw:
                        raise FreezeRefused("freeze receipt changed during publication")
                    for row, path in zip(arm_rows, arm_paths):
                        if _read_bytes(path, "frozen arm")[1] != row["sha256"]:
                            raise FreezeRefused("frozen arm changed during publication")
            return payload
    except Exception:
        if created:
            shutil.rmtree(out_dir, ignore_errors=True)
        raise


def _validate_freeze_locked(path, frozen, frozen_sha, valid, arm_paths):
    """Validate a freeze while its complete discovered dependency set is held."""
    keys = {"schema", "immutable", "labels_read", "data_role", "objective",
            "fit_receipt", "selection", "generator", "template",
            "reservation_evidence", "input_bindings", "source_bindings", "arms",
            "validation"}
    _exact(frozen, keys, "freeze receipt")
    fit_path = _binding_path(frozen["fit_receipt"], "freeze fit receipt")
    if (frozen["schema"] != _FREEZE_SCHEMA or frozen["immutable"] is not True or
            frozen["labels_read"] is not False or frozen["data_role"] != "protected-design" or
            frozen["fit_receipt"] != {"path": _relative(fit_path), "sha256": valid["sha256"]} or
            frozen["objective"] != valid["receipt"]["objective"] or
            frozen["selection"] != {"registered_index": valid["selected"],
                                    "frozen_values": {"lam": valid["lam"]},
                                    "k4_lam": valid["lam"]}):
        raise FreezeRefused("freeze receipt does not match its validated P07 fit")
    expected_inputs = {name: valid["receipt"]["inputs"][name]
                       for name in ("predictions", "producer_receipt", "candidates",
                                    "bounds", "identities")}
    freezer_binding = {"path": _relative(os.path.abspath(__file__)),
                       "sha256": _read_bytes(os.path.abspath(__file__), "freezer source")[1]}
    expected_sources = dict(valid["sources"])
    expected_sources.update(valid["extra_sources"])
    expected_sources["arm_freezer"] = freezer_binding
    if (frozen["input_bindings"] != expected_inputs or
            frozen["source_bindings"] != expected_sources or
            frozen["reservation_evidence"] != valid["reservation_evidence"]):
        raise FreezeRefused("freeze input/source/reservation bindings are stale or incomplete")
    if frozen["generator"] != {"path": _relative(_GENERATOR),
                               "sha256": valid["generator_sha"],
                               "executed_sha256": valid["generator_sha"]} or frozen["template"] != {
                                   "path": _relative(_TEMPLATE), "sha256": valid["template_sha"]}:
        raise FreezeRefused("freeze generator/template execution binding is invalid")
    if not isinstance(frozen["arms"], list) or len(frozen["arms"]) != 4 or len(arm_paths) != 4:
        raise FreezeRefused("freeze receipt must contain four arms")
    template = valid["template_raw"].decode("utf-8")
    for index, (row, spec) in enumerate(zip(frozen["arms"], valid["generator"]["ARMS"])):
        _exact(row, {"index", "name", "path", "sha256", "constants"},
               "frozen arm %d" % index)
        name, consumers, births, blurb = spec
        expected_path = os.path.join(os.path.dirname(path), "arm_%s.py" % name)
        actual_path = _repo_path(row["path"], "frozen arm %d" % index)
        expected_constants = {"_K4_ACTIVE": bool(consumers or births),
                              "_K4_CONSUMERS": bool(consumers),
                              "_K4_BIRTHS": bool(births),
                              "_K4_LAM": valid["lam"]}
        expected_raw = valid["generator"]["build"](
            name, consumers, births, blurb, template,
            valid["template_sha"], valid["lam"]).encode("utf-8")
        actual_raw, actual_sha = _read_bytes(actual_path, "frozen arm %d" % index)
        if (row != {"index": index, "name": name, "path": _relative(expected_path),
                    "sha256": _sha256_bytes(expected_raw), "constants": expected_constants} or
                not _same_path(actual_path, expected_path) or
                actual_sha != row["sha256"] or actual_raw != expected_raw):
            raise FreezeRefused("frozen arm %s is stale, mixed, or misbound" % name)
    expected_validation = {"protected_p07_fit_validated": True,
                           "completed_reservation_validated": True,
                           "selection_matches_registered_candidate": True,
                           "generator_snapshot_executed": True,
                           "dependencies_locked_through_publication": True,
                           "truth_opened_by_freezer": False}
    if frozen["validation"] != expected_validation:
        raise FreezeRefused("freeze validation evidence is incomplete")
    return {"receipt": frozen, "receipt_path": path,
            "receipt_sha256": frozen_sha, "fit_receipt_path": fit_path,
            "fit_receipt_sha256": valid["sha256"], "k4_lam": valid["lam"],
            "arm_paths": arm_paths}


def _discover_freeze(path, frozen):
    fit_path = _binding_path(frozen["fit_receipt"], "freeze fit receipt")
    initial_fit, _ = _read_json(fit_path, "fit receipt")
    arm_paths = []
    if isinstance(frozen.get("arms"), list):
        for row in frozen["arms"]:
            if isinstance(row, dict) and isinstance(row.get("path"), str):
                arm_paths.append(_repo_path(row["path"], "frozen arm"))
    paths = _initial_fit_paths(fit_path, initial_fit) + [path] + arm_paths
    return fit_path, initial_fit, arm_paths, paths


def revalidate_held_freeze(result):
    """Closing check for callers while ``hold_freeze_receipt`` still owns locks."""
    state = result.get("_held_state") if isinstance(result, dict) else None
    if not isinstance(state, dict) or not state.get("active"):
        raise FreezeRefused("freeze revalidation requires an active held context")
    frozen, frozen_sha = _read_json(result["receipt_path"], "freeze receipt")
    with _stable_files([_TRUTH_READS_LEDGER]):
        valid = _validate_fit_locked(result["fit_receipt_path"])
    checked = _validate_freeze_locked(result["receipt_path"], frozen, frozen_sha,
                                      valid, result["arm_paths"])
    if (checked["receipt_sha256"] != result["receipt_sha256"] or
            checked["fit_receipt_sha256"] != result["fit_receipt_sha256"]):
        raise FreezeRefused("freeze or fit receipt changed during held execution")
    return checked


@contextlib.contextmanager
def hold_freeze_receipt(path):
    """Yield validated normalized bindings while all dependencies remain locked."""
    path = _repo_path(path, "freeze receipt")
    frozen, frozen_sha = _read_json(path, "freeze receipt")
    _exact(frozen, {"schema", "immutable", "labels_read", "data_role", "objective",
                    "fit_receipt", "selection", "generator", "template",
                    "reservation_evidence", "input_bindings", "source_bindings",
                    "arms", "validation"}, "freeze receipt")
    fit_path, initial_fit, arm_paths, paths = _discover_freeze(path, frozen)
    with _stable_files(paths):
        again, again_sha = _read_json(path, "freeze receipt")
        if again != frozen or again_sha != frozen_sha:
            raise FreezeRefused("freeze receipt changed before locks were acquired")
        with _stable_files([_TRUTH_READS_LEDGER]):
            valid = _validate_fit_locked(fit_path)
        required = _initial_fit_paths(fit_path, valid["receipt"]) + [path] + arm_paths
        held = {os.path.normcase(os.path.realpath(item)) for item in paths}
        if any(os.path.normcase(os.path.realpath(item)) not in held for item in required):
            raise FreezeRefused("freeze dependency set changed before validation")
        result = _validate_freeze_locked(path, frozen, frozen_sha, valid, arm_paths)
        result["dependency_paths"] = paths
        state = {"active": True}
        result["_held_state"] = state
        try:
            yield result
            revalidate_held_freeze(result)
        finally:
            state["active"] = False


validated_freeze_receipt = hold_freeze_receipt


def validate_freeze_receipt(path):
    """One-shot validation wrapper; use ``hold_freeze_receipt`` around execution."""
    with hold_freeze_receipt(path) as result:
        public = {key: value for key, value in result.items() if key != "_held_state"}
    return public


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-receipt", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    try:
        result = freeze(args)
    except FreezeRefused as exc:
        parser.error(str(exc))
    print("frozen K4 lambda %.17g from registered candidate %d" %
          (result["selection"]["k4_lam"], result["selection"]["registered_index"]))
    print("receipt: %s" % os.path.join(args.out_dir, "freeze_receipt.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
