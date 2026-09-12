"""P2-CP1: the Phase-2 baseline ladder, through ONE canonical evaluator.

Runs every arm over the same networks under `lifecycle_eval.evaluate` -- one
`Estimator`, `setup()` once and off-budget, a per-MLP `BudgetContext`, all six
grader failure paths live, immutable evidence rows. No arm gets a private path.

## What it reports, and why each piece is there

Per arm, per MLP:

  * **canonical arithmetic score** `mse * max(0.1, (F + lambda R)/B)` from
    `competition_score`, which is the only place that identity lives;
  * **ordinary MSE against each replica**, and the **cross-product estimate**
    `mean_j (p_j - y1_j)(p_j - y2_j)`, whose expectation is the prediction's MSE
    because `E[e1 e2] = 0`. A single-replica MSE is biased upward by exactly the
    truth noise; the cross product removes that bias without pretending the
    labels are exact. The report separately applies each arm's multiplier,
    because unequal multipliers stop label noise cancelling from score;
  * **truth noise** beside every MSE (`docs/VALIDATION.md` section 3.5);
  * **MSE by depth**, which is the mechanism curve. ARC's propagation predicts
    `MSE_K ~ c_K (L/n)^K`, so the shape of error against depth is the thing that
    says whether the analytic family is behaving as the theory claims -- and it
    is measurable here for free because every arm returns all layers;
  * **exact counted cost under BOTH flopscope releases**, because four primitives
    reprice between 0.10 and 0.12 (`local_sim/p2_price_delta.py`);
  * residual **and its ratio to the control**, because counted FLOPs and residual
    wall time are priced by completely different mechanisms and an arm can be
    cheap in one and ruinous in the other -- that is what killed the packed lane
    (`docs/GOTCHAS.md`) and it is what the P2-CP1 counted-only freeze hid;
  * peak memory, MEASURED via `tracemalloc` rather than promised. An earlier
    build of this docstring listed peak memory and the code never recorded it,
    which an independent Skeptic caught;
  * failures, and the paired interval with its biggest-mover share.

Comparisons are **paired on common random numbers** -- every arm sees the same
networks and the same truth -- and the binding statistic is the paired
ARITHMETIC score difference with a studentised-t or basic-bootstrap interval,
via `competition_score.decide()`. Percentile and BCa are refused there, not here.

## What it will NOT do

It will not choose anything on public mini/full. It accepts only the already
exposed P01 DESIGN truth and only four arms bound by a strict P07 fit/freeze
receipt. It cannot accept free-form arm paths or confuse the fitted ``k4_lam``
with Phase 2's zero ``competition_cost_lambda``. ``decide()`` requires a
pre-registered gate, so this module cannot invent one.

Run:
    uv run python local_sim/p2_lab.py --truth experiments/<dir>/truth.npz \\
        --out experiments/<dir>/ladder.json \\
        --freeze-receipt experiments/<freeze>/freeze_receipt.json --control exact \
        --competition-cost-lambda 0
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import ctypes
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

import tracemalloc  # noqa: E402

import competition_score as cs  # noqa: E402
import lifecycle_eval as le  # noqa: E402
import p2_freeze_k4_arms as k4_freezer  # noqa: E402
from phase2_manifest import load as load_manifest  # noqa: E402

ARMS_DIR = os.path.join(_HERE, "p2_arms")
_ACTIVE_ARM_PATHS = None
_CLI_BINDINGS_ENV = "ARC_WHEST_P2_LAB_LOCKED_BINDINGS"
_SELF_FINGERPRINT = "1ea7ed02f95ab64cf7b4819f31945adec5ed1593ce4bce87b7eec989bdf123de"
_LOCKED_CHILD_ARG = "--arc-whest-lab-locked-child"
_LOCKED_CHILD_SCHEMA = "arc-whest.p2-lab-locked-child.v1"
_BOUND_CLI_BINDINGS = None
_ALLOW_IN_PROCESS_TEST = False

MECHANISM_ARMS = ("exact", "coherent", "consumers-only", "births-only")

#: The arm every other arm is compared against.
#:
#: `p2_test_arms.py` T2 proves `k2p1` reproduces the shipped `estimator.py`
#: `_analytic_only` path **bitwise in its OUTPUT** at this shape. That makes it a
#: sound ACCURACY control.
#:
#: It is NOT a cost-identical control, and this comment used to imply it was.
#: Measured 2026-08-24 (`p2_flop_anchor.py --gate --candidate
#: local_sim/p2_arms/arm_k2p1.py`): 137,600,363,520 against the shipped file's
#: 137,600,326,656 -- **+36,864 FLOPs over three operations** (one extra `astype`,
#: a `concatenate`, a `reshape`) from the arm's own scaffolding. Negligible at
#: 0.000027%, and irrelevant while the multiplier is pinned at the 0.1 floor --
#: but "bitwise identical" and "identical cost" are different claims, and this
#: ladder reports both accuracy and cost.
CONTROL = "exact"


def _sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def _normalized_self_sha(raw):
    marker = re.compile(
        rb'(?m)^_SELF_FINGERPRINT = "[0-9a-f]{64}"')
    normalized, count = marker.subn(
        b'_SELF_FINGERPRINT = "' + (b'0' * 64) + b'"', raw)
    if count != 1:
        raise k4_freezer.FreezeRefused(
            "lab source fingerprint marker is missing or ambiguous")
    return _sha256_bytes(normalized.replace(b"\r\n", b"\n"))


def _assert_loaded_source(expected_exact_sha=None):
    try:
        with open(os.path.abspath(__file__), "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise k4_freezer.FreezeRefused("cannot read lab source: %s" % exc) from exc
    exact_sha = _sha256_bytes(raw)
    if _normalized_self_sha(raw) != _SELF_FINGERPRINT:
        raise k4_freezer.FreezeRefused(
            "loaded lab source does not match the locked disk snapshot")
    if expected_exact_sha is not None and exact_sha != expected_exact_sha:
        raise k4_freezer.FreezeRefused(
            "lab source differs from the parent-locked snapshot")
    return exact_sha


@contextlib.contextmanager
def _bootstrap_stable_files(paths):
    """Hold CLI sources before a fresh child imports any of them."""
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
        close.restype = wintypes.BOOL
        handles = []
        try:
            for path in unique:
                handle = create(path, 0x80000000, 0x00000001, None, 3,
                                0x00000080, None)
                if handle == wintypes.HANDLE(-1).value:
                    raise k4_freezer.FreezeRefused(
                        "cannot lock lab CLI dependency %s: WinError %d" %
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
            fcntl.flock(fh.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield
    except (OSError, BlockingIOError) as exc:
        raise k4_freezer.FreezeRefused(
            "cannot lock lab CLI dependencies: %s" % exc) from exc
    finally:
        for fh in reversed(files):
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()


def _cli_source_paths():
    return {
        "p2_lab": os.path.abspath(__file__),
        "lifecycle_eval": os.path.abspath(le.__file__),
        "competition_score": os.path.abspath(cs.__file__),
        "arm_freezer": os.path.abspath(k4_freezer.__file__),
        "phase2_manifest": os.path.join(_HERE, "phase2_manifest.py"),
        "p2_tag_drift": os.path.join(_HERE, "p2_tag_drift.py"),
        "phase2_manifest_data": os.path.join(
            _ROOT, "experiments", "2026-08-21_phase2_manifest",
            "phase2_manifest.json"),
        "official_config": os.path.join(_HERE, "official_config.py"),
        "harness": os.path.join(_HERE, "harness.py"),
        "mint_provenance": os.path.join(_HERE, "mint_provenance.py"),
        "seed_ledger": os.path.join(_HERE, "seed_ledger.py"),
        "seed_blocks": os.path.join(_HERE, "seed_blocks.py"),
        "truth_reads_interface": os.path.join(_HERE, "truth_reads.py"),
        "fit_interface": os.path.join(_HERE, "p2_k4_fit.py"),
        "fit_producer": os.path.join(_HERE, "p2_k4_fit_producer.py"),
        "arm_generator": os.path.join(_HERE, "p2_build_k4_arms.py"),
        "arm_template": os.path.join(_HERE, "p2_compact_candidate_template.py"),
    }


def _binding_snapshot(paths):
    result = {}
    for name, path in paths.items():
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
        except OSError as exc:
            raise k4_freezer.FreezeRefused(
                "cannot read lab CLI source %s: %s" % (name, exc)) from exc
        result[name] = {
            "path": os.path.relpath(path, _ROOT).replace("\\", "/"),
            "sha256": _sha256_bytes(raw)}
    return result


def _verify_cli_bindings(bindings, paths):
    current = _binding_snapshot(paths)
    if not isinstance(bindings, dict) or bindings != current:
        raise k4_freezer.FreezeRefused(
            "parent-locked lab source bindings are invalid or changed")
    _assert_loaded_source(bindings["p2_lab"]["sha256"])
    k4_freezer._assert_loaded_source(bindings["arm_freezer"]["sha256"])


def _locked_child_entry(args, paths, launch_nonce):
    """Accept bindings only from the real script child's dedicated stdin."""
    if __name__ != "__main__":
        print("REFUSED: locked lab child mode requires script execution")
        return 2
    try:
        encoded = sys.stdin.read()
        payload = json.loads(encoded)
        if not isinstance(payload, dict) or set(payload) != {
                "schema", "launch_nonce", "bindings"}:
            raise k4_freezer.FreezeRefused(
                "locked lab child payload keys are invalid")
        if payload["schema"] != _LOCKED_CHILD_SCHEMA:
            raise k4_freezer.FreezeRefused("locked lab child schema is invalid")
        if (not isinstance(launch_nonce, str) or
                re.fullmatch(r"[0-9a-f]{64}", launch_nonce) is None or
                payload["launch_nonce"] != launch_nonce):
            raise k4_freezer.FreezeRefused(
                "locked lab child launch nonce is invalid")
        bindings = payload["bindings"]
    except (OSError, json.JSONDecodeError, k4_freezer.FreezeRefused) as exc:
        print("REFUSED: invalid locked lab child payload: %s" % exc)
        return 2
    global _BOUND_CLI_BINDINGS
    try:
        with _bootstrap_stable_files(paths.values()):
            _verify_cli_bindings(bindings, paths)
            _BOUND_CLI_BINDINGS = bindings
            return main(args)
    except k4_freezer.FreezeRefused as exc:
        print("REFUSED: %s" % exc)
        return 2
    finally:
        _BOUND_CLI_BINDINGS = None


def cli_entry(argv=None):
    """Run production in a fresh child while the parent holds source locks."""
    from_sys_argv = argv is None
    args = list(sys.argv[1:] if from_sys_argv else argv)
    paths = _cli_source_paths()
    child_prefix = _LOCKED_CHILD_ARG + "="
    if args and (args[0] == _LOCKED_CHILD_ARG or
                 args[0].startswith(child_prefix)):
        if not from_sys_argv or not args[0].startswith(child_prefix):
            print("REFUSED: locked lab child mode is private to the CLI launcher")
            return 2
        return _locked_child_entry(args[1:], paths,
                                   args[0][len(child_prefix):])

    try:
        with _bootstrap_stable_files(paths.values()):
            _assert_loaded_source()
            bindings = _binding_snapshot(paths)
            env = dict(os.environ)
            # A legacy marker is replayable process state, not launch evidence.
            # Strip it so even a pre-set/forged value cannot authorize this
            # already-imported parent to execute production core.
            env.pop(_CLI_BINDINGS_ENV, None)
            launch_nonce = secrets.token_hex(32)
            payload = json.dumps({
                "schema": _LOCKED_CHILD_SCHEMA,
                "launch_nonce": launch_nonce,
                "bindings": bindings,
            }, sort_keys=True, separators=(",", ":"))
            with tempfile.TemporaryDirectory(prefix="p2-lab-pycache-") as cache:
                completed = subprocess.run(
                    [sys.executable, "-X", "pycache_prefix=" + cache,
                     os.path.abspath(__file__),
                     child_prefix + launch_nonce] + args,
                    cwd=os.getcwd(), env=env, input=payload, text=True)
            return int(completed.returncode)
    except k4_freezer.FreezeRefused as exc:
        print("REFUSED: %s" % exc)
        return 2


def _revalidate_instruments(bindings):
    for name, binding in bindings.items():
        path = k4_freezer._repo_path(binding["path"], f"lab instrument {name}")
        try:
            with open(path, "rb") as fh:
                closing = hashlib.sha256(fh.read()).hexdigest()
        except OSError as exc:
            raise k4_freezer.FreezeRefused(
                f"lab instrument {name} unavailable at closing publication: {exc}") from exc
        if closing != binding["sha256"]:
            raise k4_freezer.FreezeRefused(
                f"lab instrument {name} changed before closing publication")


def _arm_path(name):
    """Map the report-facing arm name to its generated Python source."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        raise ValueError(f"invalid arm name {name!r}")
    if _ACTIVE_ARM_PATHS is not None:
        try:
            return _ACTIVE_ARM_PATHS[name]
        except KeyError as exc:
            raise ValueError(f"freeze receipt has no arm {name!r}") from exc
    return os.path.join(ARMS_DIR, f"arm_{name.replace('-', '_')}.py")


def _parse_arms(raw):
    arms = [x.strip() for x in raw.split(",") if x.strip()]
    if len(arms) != len(set(arms)):
        raise ValueError("freeze receipt arm set contains duplicates")
    if set(arms) != set(MECHANISM_ARMS) or len(arms) != 4:
        raise ValueError(
            "freeze receipt must name exactly: " + ",".join(MECHANISM_ARMS))
    return arms


def _validate_request(a, man, frozen):
    """Validate everything knowable without opening protected truth."""
    freeze_rows = frozen["receipt"]["arms"]
    arms = _parse_arms(",".join(row["name"].replace("_", "-")
                                for row in freeze_rows))
    arm_paths = {row["name"].replace("_", "-"):
                 k4_freezer._repo_path(row["path"], "frozen arm")
                 for row in freeze_rows}
    if a.control != CONTROL or a.control not in arms:
        raise ValueError(f"--control must be {CONTROL!r} and appear in the frozen arms")
    if not a.out:
        raise ValueError("--out is required")
    if os.path.lexists(a.out):
        raise ValueError(f"output already exists: {a.out}")
    if os.path.abspath(a.out) == os.path.abspath(a.truth):
        raise ValueError("--out must not be the truth path")
    if not os.path.isfile(a.truth):
        raise ValueError(f"truth file does not exist: {a.truth}")
    if a.n < 0 or a.depth_subset <= 0:
        raise ValueError("--n must be non-negative and --depth-subset positive")
    for label, value in (("competition cost lambda", a.competition_cost_lambda),
                         ("gate", a.gate),
                         ("official samples", a.official_samples),
                         ("residual cap", a.residual_cap)):
        if value is not None and not np.isfinite(value):
            raise ValueError(f"{label} must be finite")
    if a.gate is not None and a.gate <= 0:
        raise ValueError("--gate must be positive")
    if a.official_samples <= 0:
        raise ValueError("--official-samples must be positive")
    if a.competition_cost_lambda != 0.0:
        raise ValueError("Phase-2 --competition-cost-lambda must be exactly zero")
    try:
        shape, budget = man["shape"], man["budget"]
        if int(shape["width"]) <= 0 or int(shape["depth"]) <= 0:
            raise ValueError("manifest shape must be positive")
        if float(budget["flop_budget_per_mlp"]) <= 0:
            raise ValueError("manifest FLOP budget must be positive")
        if float(budget["multiplier_floor"]) <= 0:
            raise ValueError("manifest multiplier floor must be positive")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"manifest is missing required configuration: {exc}") from exc
    for name in arms:
        path = arm_paths[name]
        if not os.path.isfile(path):
            raise ValueError(f"generated arm is missing: {path}")
        try:
            with open(path, encoding="utf-8") as fh:
                ast.parse(fh.read(), filename=path)
        except (OSError, SyntaxError) as exc:
            raise ValueError(f"generated arm {name!r} does not parse: {exc}") from exc
    return arms, arm_paths


def _replica_noise(y1, y2):
    """mean_j ((y1 - y2)/2)^2, per network. The AVERAGED replica's own noise.

    This is the whole cross-product correction, and it needs no predictions:

        mean[(p - y1)(p - y2)] = mean[(p - ybar)^2] - mean[((y1 - y2)/2)^2]

    with `ybar = (y1 + y2)/2` -- expand both sides and the cross terms cancel
    identically. So the noise-corrected MSE is the ordinary MSE against the
    averaged replica minus a constant that depends only on the truth.

    Three things follow, and they are why the driver runs each arm ONCE:
      * the evaluator already computes the first term, so no second pass is
        needed to recover per-MLP predictions;
      * the subtracted constant IS `V / (2N)`, so it is a measured quantity with
        a predicted value to check it against;
      * the MSE bias is identical across arms, while the SCORE bias is weighted
        by each arm's per-network multiplier and therefore cancels only when
        those multipliers are equal.
    """
    return (((y1 - y2) / 2.0) ** 2).mean(axis=1)


def _run_arm(name, seeds, truth_final, spec, gt_noise):
    path = _arm_path(name)
    if not os.path.exists(path):
        raise SystemExit(f"no such arm: {path}. Run p2_build_arms.py first.")
    return le.evaluate(path, seeds, truth_final, spec=spec, gt_noise=gt_noise)


def _cost_under_012(name, width, depth, budget):
    """Counted cost of one predict under flopscope 0.12, in the isolated venv.

    Returns None when `.venv-fs012` is absent. Four primitives reprice between
    0.10 and 0.12 and one of them (`where` with a Python scalar) HALVES, so a
    cost quoted from 0.10 alone is not a Phase-2 cost.
    """
    exe = os.path.join(_ROOT, ".venv-fs012", "Scripts", "python.exe")
    if not os.path.exists(exe):
        return None
    code = (
        "import sys,os,importlib.util as u,types,numpy as np\n"
        "import flopscope as flops\n"
        "wb=types.ModuleType('whestbench')\n"
        "class MLP:\n"
        " def __init__(s,weights,width,depth,seed):\n"
        "  s.weights,s.width,s.depth,s.seed=weights,width,depth,seed\n"
        "class BaseEstimator:\n"
        " def setup(s,c):pass\n"
        "wb.MLP=MLP;wb.BaseEstimator=BaseEstimator;wb.__path__=[]\n"
        "sys.modules['whestbench']=wb\n"
        f"W,D,B={width},{depth},{budget}\n"
        "ss=np.random.SeedSequence(1000003).spawn(3)\n"
        "r=np.random.default_rng(ss[0]);sc=float(np.sqrt(2.0/W))\n"
        "wt=[(r.standard_normal((W,W))*sc).astype(np.float32) for _ in range(D)]\n"
        "m=MLP(wt,W,D,int(ss[2].generate_state(1)[0]))\n"
        f"sp=u.spec_from_file_location('a',r'{_arm_path(name)}')\n"
        "mo=u.module_from_spec(sp);sp.loader.exec_module(mo);e=mo.Estimator()\n"
        "class C:width,depth,flop_budget,seed=W,D,B,0\n"
        "e.setup(C())\n"
        "with flops.BudgetContext(flop_budget=B,quiet=True) as c:\n"
        "    e.predict(m,B)\n"
        "    print('@@',float(c.flops_used))\n"
    )
    r = subprocess.run([exe, "-c", code], capture_output=True, text=True,
                       cwd=_ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    for ln in r.stdout.splitlines():
        if ln.startswith("@@"):
            return float(ln[2:])
    return None



def main(argv=None) -> int:
    if _BOUND_CLI_BINDINGS is None and not _ALLOW_IN_PROCESS_TEST:
        print("REFUSED: production P01 reporting requires the locked CLI entrypoint")
        return 2
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--freeze-receipt", default="",
                    help="immutable P07 coefficient-to-four-arm freeze receipt")
    ap.add_argument("--n", type=int, default=0, help="0 = every network in the truth file")
    ap.add_argument("--competition-cost-lambda", type=float, default=0.0,
                    help="Phase-2 score-cost lambda; must be zero and is distinct "
                         "from the fitted K4 coefficient")
    ap.add_argument("--depth-subset", type=int, default=8,
                    help="networks used for the MSE-by-depth mechanism curve")
    ap.add_argument("--gate", type=float, default=None,
                    help="pre-registered promotion gate; REQUIRED to print a verdict")
    ap.add_argument("--block", default="",
                    help="the seed block this truth belongs to. Inferred from "
                         "the filename when it carries a block id; REQUIRED "
                         "otherwise, because guessing is how a single-use "
                         "block gets read twice.")
    ap.add_argument("--ts", default="",
                    help="historical event timestamp only; omit for an automatic "
                         "runtime receipt")
    ap.add_argument("--note", default="", help="recorded on the read line")
    ap.add_argument("--control", required=True,
                    help=f"the registered control; must be {CONTROL!r}")
    ap.add_argument("--official-samples", type=float, default=1e9,
                    help="official truth samples used for the precision forecast")
    ap.add_argument("--residual-cap", type=float, default=None,
                    help="override the 0.4 s cliff for an ACCURACY-ONLY run. "
                         "<=0 disables it entirely. The relaxation is printed "
                         "and recorded; such a run is never residual evidence.")
    a = ap.parse_args(argv)

    instrument_paths = {
        "p2_lab": os.path.abspath(__file__),
        "lifecycle_eval": os.path.abspath(le.__file__),
        "competition_score": os.path.abspath(cs.__file__),
        "harness": os.path.join(_HERE, "harness.py"),
        "mint_provenance": os.path.join(_HERE, "mint_provenance.py"),
        "phase2_manifest": os.path.join(_HERE, "phase2_manifest.py"),
        "p2_tag_drift": os.path.join(_HERE, "p2_tag_drift.py"),
        "phase2_manifest_data": os.path.join(
            _ROOT, "experiments", "2026-08-21_phase2_manifest",
            "phase2_manifest.json"),
        "official_config": os.path.join(_HERE, "official_config.py"),
        "seed_ledger": os.path.join(_HERE, "seed_ledger.py"),
        "seed_blocks": os.path.join(_HERE, "seed_blocks.py"),
    }
    out_preexisting = os.path.lexists(a.out)
    global _ACTIVE_ARM_PATHS
    try:
        with k4_freezer.hold_freeze_receipt(a.freeze_receipt) as frozen:
            with k4_freezer._stable_files(list(instrument_paths.values()) +
                                          [os.path.abspath(a.truth)]):
                instrument_bindings = {}
                for name, path in instrument_paths.items():
                    with open(path, "rb") as fh:
                        raw = fh.read()
                    if path.endswith(".py"):
                        ast.parse(raw.decode("utf-8"), filename=path)
                    else:
                        json.loads(raw.decode("utf-8"))
                    instrument_bindings[name] = {
                        "path": os.path.relpath(path, _ROOT).replace("\\", "/"),
                        "sha256": hashlib.sha256(raw).hexdigest()}
                for name in ("truth_reads_interface", "arm_freezer"):
                    binding = frozen["receipt"]["source_bindings"][name]
                    instrument_bindings[name] = binding
                if _BOUND_CLI_BINDINGS is not None:
                    for name, binding in instrument_bindings.items():
                        if _BOUND_CLI_BINDINGS.get(name) != binding:
                            raise k4_freezer.FreezeRefused(
                                f"executed lab dependency {name} does not match its parent lock")
                man = load_manifest()
                arms, arm_paths = _validate_request(a, man, frozen)
                _ACTIVE_ARM_PATHS = arm_paths
                rc = _execute_held(a, man, frozen, arms, arm_paths,
                                   instrument_bindings)
                k4_freezer.revalidate_held_freeze(frozen)
                _revalidate_instruments(instrument_bindings)
                return rc
    except (ValueError, k4_freezer.FreezeRefused) as exc:
        if not out_preexisting and os.path.isfile(a.out):
            os.remove(a.out)
        print(f"REFUSED: {exc}")
        return 2
    finally:
        _ACTIVE_ARM_PATHS = None


def _execute_held(a, man, frozen, arms, arm_paths, instrument_bindings) -> int:
    width, depth = man["shape"]["width"], man["shape"]["depth"]
    budget = man["budget"]["flop_budget_per_mlp"]
    floor = man["budget"]["multiplier_floor"]
    parity = man["package_parity"]
    frozen_hashes = {row["name"].replace("_", "-"): row["sha256"]
                     for row in frozen["receipt"]["arms"]}

    # THE TRUTH READ IS REGISTERED, NOT MERELY PERFORMED.
    #
    # Stage 0 R3 (2026-09-10): `seed_ledger` refuses a second CLAIM on a
    # single-use block and has never been able to refuse a second np.load of
    # the truth that claim produced. P08 was claimed once and read three times
    # before anything noticed. `local_sim/truth_reads.py` counts looks; this is
    # the call site that makes the lab honest, and the block is REQUIRED rather
    # than guessed at when the filename does not carry it.
    import truth_reads as _tr
    try:
        block = _tr.resolve_block(a.truth, a.block)
        if block != "P01":
            raise _tr.ReadRefused(
                "the receipt-driven four-arm mechanism report is restricted to exposed DESIGN block P01")
        read_status = _tr.status(block)
        if read_status.get("tier") != "DESIGN":
            raise _tr.ReadRefused(
                "P01 must remain registered as exposed DESIGN data")
        before_reservations = {row["reservation"] for row in read_status["reads"]}
        z = _tr.load_truth(
            block, a.truth, consumer="local_sim/p2_lab.py",
            purpose=f"four-arm mechanism report: {','.join(arms)}",
            ts=a.ts, instrument=os.path.abspath(__file__),
            result_artifact=a.out, note=a.note)
        after_rows = _tr.status(block)["reads"]
        def is_this_report(row):
            try:
                return k4_freezer._same_path(
                    k4_freezer._repo_path(row["result_artifact"],
                                          "P01 result artifact"),
                    os.path.abspath(a.out))
            except (KeyError, k4_freezer.FreezeRefused):
                return False
        new_rows = [row for row in after_rows
                    if row["reservation"] not in before_reservations and
                    row["consumer"] == "local_sim/p2_lab.py" and
                    row.get("purpose") == f"four-arm mechanism report: {','.join(arms)}" and
                    is_this_report(row)]
        ids = {row["reservation"] for row in new_rows}
        if (len(ids) != 1 or len(new_rows) != 2 or
                {row["status"] for row in new_rows} != {"reserved", "completed"}):
            raise k4_freezer.FreezeRefused(
                "P01 truth access lacks one unique reserved/completed receipt pair")
        truth_reservation = next(row for row in new_rows if row["status"] == "completed")
        truth_reserved = next(row for row in new_rows if row["status"] == "reserved")
        immutable_reservation_fields = {
            "block", "artifact", "artifact_sha256", "consumer",
            "instrument_sha256", "purpose", "ts", "reservation",
            "result_artifact", "note", "backfilled", "timestamp_source",
            "receipt_created_utc"}
        if (not k4_freezer._same_path(
                k4_freezer._repo_path(truth_reservation["result_artifact"],
                                      "P01 result artifact"),
                os.path.abspath(a.out)) or
                truth_reservation["block"] != "P01" or
                truth_reservation.get("instrument_sha256") !=
                instrument_bindings["p2_lab"]["sha256"] or
                truth_reservation.get("backfilled") is not False or
                any(truth_reserved.get(key) != truth_reservation.get(key)
                    for key in immutable_reservation_fields) or
                truth_reserved.get("receipt_completed_utc") != "" or
                truth_reserved.get("elapsed_seconds") is not None):
            raise k4_freezer.FreezeRefused(
                "P01 truth receipt does not bind this report path")
        with open(a.truth, "rb") as fh:
            truth_sha = hashlib.sha256(fh.read()).hexdigest()
        if truth_reservation["artifact_sha256"] != truth_sha:
            raise k4_freezer.FreezeRefused(
                "P01 truth bytes do not match the completed reservation")
        truth_binding = {
            "data_role": "exposed-design", "block": "P01",
            "path": os.path.relpath(os.path.abspath(a.truth), _ROOT).replace("\\", "/"),
            "sha256": truth_sha, "reservation_receipt": truth_reservation}
    except _tr.ReadRefused as exc:
        print(f"REFUSED before truth access: {exc}")
        return 2
    required = {"seeds", "means_r1", "means_r2", "gt_noise", "n_samples"}
    missing = required - set(z)
    if missing:
        raise ValueError(f"truth archive is missing required arrays: {sorted(missing)}")
    seeds = [int(s) for s in z["seeds"]]
    n = len(seeds) if a.n == 0 else min(a.n, len(seeds))
    seeds = seeds[:n]
    y1 = z["means_r1"][:n, -1, :].astype(np.float64)
    y2 = z["means_r2"][:n, -1, :].astype(np.float64)
    all1 = z["means_r1"][:n].astype(np.float64)
    all2 = z["means_r2"][:n].astype(np.float64)
    gt_noise_per_replica = z["gt_noise"][:n].astype(np.float64)
    gt_noise = gt_noise_per_replica / 2.0
    y_mid = ((y1 + y2) / 2.0).astype(np.float32)

    is_probe = bool(int(z["probe"])) if "probe" in z else False
    if is_probe:
        print("!" * 92)
        print("!!  PROBE TRUTH -- unregistered seeds, no block claimed. This run is a")
        print("!!  PIPELINE SMOKE. Nothing it prints may select an arm, a constant, a")
        print("!!  rank, or a gate, and no number from it belongs in a durable doc.")
        print("!" * 92)

    # The DECIDING model is whatever the contract froze, and it is passed in
    # explicitly rather than defaulted. P2-CP1 froze COUNTED FLOPS ONLY (lam=0):
    # every analytic arm sits at C/B ~0.063, a factor of 1.6 under the 0.1 floor,
    # so no residual model that does not add 0.8 s/MLP can reorder them -- and the
    # sampling arms are separated by counted cost by construction. The local
    # residual is a --runner local figure that does not transfer (2-3% of C on the
    # grader against 10-13% here), so letting it into the deciding number would
    # import this machine's noise into a Phase-2 conclusion.
    # CHANGED UNDER YOU, 2026-08-24 (P2-P0). This call takes LifecycleSpec's
    # DEFAULTS for everything it does not name, and two of those defaults moved:
    #
    #   residual_wall_time_limit_s   None  ->  0.4 s     the cliff is now ARMED
    #   lam                          1e11  ->  0.0       the live Phase-2 rate
    #
    # The lambda change is a no-op for this slice: P2-CP1 froze lam = 0 explicitly
    # and it turns out that is not merely conservative, it is EXACTLY the Phase-2
    # rule -- the round sets lambda to zero, so C == F. The frozen model was right.
    #
    # The cliff is not a no-op. P2-CP1 ran with it DISABLED, so its failure
    # accounting could not see a 0.4 s violation, and re-running this ladder today
    # may fail an arm that passed then. That is the correct behaviour and the
    # difference is a finding, not a regression -- see the P2-P0 correction in
    # experiments/LOG.md before comparing a new run to the recorded one.
    # THE RESIDUAL CLIFF, AND WHY A DESIGN-TIER ACCURACY RUN MAY RELAX IT.
    #
    # `--residual-cap` exists because this host reads ~27x high per operation
    # on residual (docs/VALIDATION.md 433-542, docs/GOTCHAS.md 126), so the
    # armed 0.4 s cliff fails the exact K3 chain here on machine speed rather
    # than on the candidate's behaviour -- measured 726 ms on one network at
    # 1024x16 against a graded 330566 whose worst SERVER wall was 33 s of 120.
    # A failed MLP scores as zero-prediction MSE, so leaving it armed would
    # replace an accuracy comparison with a comparison of nothing.
    #
    # It is NOT disabled silently and it is NOT a policy change. The cap is
    # printed, recorded in the report, and any run that relaxes it is labelled
    # ACCURACY-ONLY. Deployment residual remains `certify_candidate` section 7
    # and the official subprocess runner's job, and no relaxed run may be
    # quoted as residual evidence.
    _res = {} if a.residual_cap is None else {
        "residual_wall_time_limit_s": (None if a.residual_cap <= 0
                                       else a.residual_cap)}
    spec = le.LifecycleSpec(width=width, depth=depth, flop_budget=budget,
                            floor=floor, **_res,
                            lam=a.competition_cost_lambda)
    if a.residual_cap is not None:
        print("  *** RESIDUAL CAP RELAXED -- this run is ACCURACY-ONLY. Its "
              "failure accounting")
        print("      cannot see a 0.4 s violation and it is NOT residual "
              "evidence for anything.")
    print(f"  deciding model   lambda = {spec.lam:.4g}"
          + ("   (COUNTED FLOPS ONLY -- the frozen model)" if spec.lam == 0.0
             else "   (Phase-1 lambda; NOT the frozen P2-CP1 model)"))
    print("=" * 92)
    print("P2-CP1 LADDER  --  one canonical evaluator, paired on common random numbers")
    print("=" * 92)
    print(f"  truth      {os.path.relpath(a.truth, _ROOT)}   {n} networks   "
          f"2 replicas x {int(z['n_samples']):,} samples")
    print(f"  truth noise per replica   mean {gt_noise_per_replica.mean():.4e}")
    print(f"  shape      {width} x {depth}   B = {budget:,}   floor {floor}")
    print(f"  stack      flopscope {parity['installed_here']['flopscope']}"
          f"   PARITY {parity['status']}")
    if parity["limitation_stamp"]:
        print(f"  STAMP      {parity['limitation_stamp'][:74]}...")
    print()

    # The averaged replica's noise, measured once. Every arm's MSE against `y_mid`
    # is inflated by exactly this, identically, so it cancels in every paired
    # comparison and is subtracted only for the ABSOLUTE numbers.
    rep_noise = _replica_noise(y1, y2)
    print(f"  replica noise of the AVERAGED truth   measured {rep_noise.mean():.4e}"
          f"   predicted V/2N {gt_noise.mean():.4e}"
          f"   ratio {rep_noise.mean() / gt_noise.mean():.4f}")
    print("     (the ratio is a live check on replica independence; far from 1.0")
    print("      means the two draws share state and every corrected MSE is wrong)")
    print()

    n_depth = min(a.depth_subset, n)
    results = {}
    for name in arms:
        t0 = time.perf_counter()
        # ONE scored pass. `lifecycle_eval` owns every scored number.
        rows, prov = _run_arm(name, seeds, y_mid, spec, list(gt_noise))
        expected_hash = frozen_hashes[name]
        row_hashes = {row.candidate_sha256 for row in rows}
        if prov.get("candidate_sha256") != expected_hash or row_hashes != {expected_hash}:
            raise k4_freezer.FreezeRefused(
                f"scored lifecycle for {name} did not execute its frozen candidate bytes")
        suite = le.score(rows, spec=spec)
        raw = np.array([r.mse for r in rows])
        cross = raw - rep_noise
        corrected = raw - gt_noise
        # The depth curve needs full prediction arrays, which the evaluator does
        # not return. A mechanism curve does not need 96 networks, so it is
        # computed on a fixed SUBSET and labelled as such rather than pretending
        # otherwise.
        tracemalloc.start()
        _f, alld, depth_hash = _predict_all(name, seeds[:n_depth], spec)
        if depth_hash != expected_hash:
            raise k4_freezer.FreezeRefused(
                f"depth lifecycle for {name} did not execute its frozen candidate bytes")
        _cur, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        by_depth = np.array([
            [float(((alld[i][d] - (all1[i][d] + all2[i][d]) / 2.0) ** 2).mean())
             for d in range(depth)] for i in range(n_depth)])
        c012 = _cost_under_012(name, width, depth, budget)
        results[name] = dict(
            rows=rows, prov=prov, suite=suite, cross=cross,
            corrected=corrected, raw=raw,
            by_depth=by_depth, cost012=c012, secs=time.perf_counter() - t0,
            peak_mb=peak_bytes / 2 ** 20, n_depth=n_depth,
        )
        f = float(np.mean([r.flops_used for r in rows]))
        eff = float(np.mean([cs.effective_compute(r.flops_used,
                                                  r.residual_wall_time_s, spec.lam)
                             for r in rows]))
        print(f"  {name:<18} score {suite.objective:.5e}   "
              f"MSE(corr) {corrected.mean():.4e}   MSE(raw) {raw.mean():.4e}   "
              f"F/B {f / budget:.4f}   eff {eff / budget:.4f}   "
              f"mult {suite.mean_multiplier:.4f}   fail {suite.n_failed}/{n}   "
              f"resid {1000 * float(np.mean([r.residual_wall_time_s for r in rows])):.0f}ms   "
              f"{results[name]['secs']:.0f}s"
              + ("" if c012 is None else
                 f"   [0.12: {c012 / budget:.6f}"
                 + ("" if abs(c012 - f) < 1 else f"  MOVED {c012 / f:.6f}x") + "]"))

    # ---- paired comparisons against the control ---------------------------
    print()
    print(f"  PAIRED vs {a.control} -- arithmetic score difference, "
          f"studentised-t 95% (the binding statistic, docs/VALIDATION.md 2b-i)")
    print(f"  {'arm':<18} {'gain':>8} {'95% t CI (relative)':>26} {'better':>9} "
          f"{'sign p':>8}")
    ctrl_rows = results[a.control]["rows"]
    paired = {}
    noise_aware = {}
    local_samples = 2.0 * float(z["n_samples"])
    for name in arms:
        if name == a.control:
            continue
        res = cs.paired_compare_rows(
            [r.to_row() for r in ctrl_rows],
            [r.to_row() for r in results[name]["rows"]],
            lam=spec.lam, floor=floor,
        )
        paired[name] = res
        noise_aware[name] = cs.noise_corrected_compare(
            [r.to_row() for r in ctrl_rows],
            [r.to_row() for r in results[name]["rows"]],
            local_samples=local_samples,
            official_samples=a.official_samples,
            lam=spec.lam, floor=floor, require_ids=True)
        # The BINDING interval is the studentised t on the ABSOLUTE difference
        # (docs/VALIDATION.md 2b-i). It is shown relative to the control's mean
        # purely for readability; the decision is made by `decide()`, which takes
        # the pre-registered gate and refuses percentile/BCa.
        lo, hi = res.t_ci_difference
        m = res.mean_control or float("nan")
        print(f"  {name:<18} {res.arithmetic_gain:>8.4f} "
              f"[{lo / m:>+10.2%}, {hi / m:>+10.2%}] {res.n_candidate_better:>4}/{res.n:<4} "
              f"p={res.sign_test_p:>6.3f}")

    print()
    print("  NOISE-AWARE ADJUSTED DIFFERENCES -- observed / true-function / "
          "official precision")
    for name in arms:
        if name == a.control:
            continue
        q = noise_aware[name]
        print(f"  {name:<18} {q.observed.mean_difference:+.4e}  "
              f"{q.true_function.mean_difference:+.4e}  "
              f"{q.forecast.mean_difference:+.4e}   "
              f"noise term {q.noise_bias:+.3e}")

    # ---- does any arm leave the floor once residual is charged? --------------
    # The frozen deciding model is counted-FLOPs-only, and that is defensible for
    # RANKING here. It is NOT a licence to ignore residual: an arm can be pinned
    # at the floor on counted cost and off it on effective cost, and the two are
    # priced by different mechanisms. flopscope charges the RELEASE of a large
    # temporary as residual wall time (~100-120 us/op, forum 18101) while pricing
    # the operation itself at O(n^2) FLOPs -- so an elementwise-heavy arm is cheap
    # in the frozen model and expensive in reality. That arbitrage is exactly what
    # killed the packed lane. Report it loudly rather than letting the freeze hide
    # it, because the next lane (K3/Edgeworth) is elementwise-heavy in the same way.
    print()
    if spec.lam == 0.0:
        print("  COST CHECK -- Phase 2 lambda = 0, so effective C equals counted F")
        print("  (local residual is shown separately as a cliff-risk diagnostic;")
        print("   it is not priced and its absolute level does not transfer)")
    else:
        print(f"  COST CHECK -- effective C = F + lambda*R at lambda = {spec.lam:g}")
        print("  (local residual does NOT transfer; only the within-instrument")
        print("   ratio is descriptive, and this is not a Phase-2 cost model)")
    ctl_r = float(np.mean([r.residual_wall_time_s for r in results[a.control]["rows"]]))
    for name in arms:
        rws = results[name]["rows"]
        f = float(np.mean([r.flops_used for r in rws]))
        rr = float(np.mean([r.residual_wall_time_s for r in rws]))
        eff = [(r.flops_used + 1e11 * r.residual_wall_time_s) / budget for r in rws]
        off = f / budget < floor <= max(eff)
        print(f"    {name:<18} F/B {f / budget:.4f}   eff mean {np.mean(eff):.4f}"
              f"   eff max {max(eff):.4f}   resid {1000 * rr:>6.0f} ms"
              f"  ({rr / ctl_r if ctl_r else float('nan'):>5.2f}x control)"
              + ("   *** OFF THE FLOOR once residual is charged ***" if off else ""))

    # ---- THE GATE -----------------------------------------------------------
    # The first build of this driver took a `--gate` argument and never used it,
    # leaving the verdict to whoever read the table above. That is precisely the
    # failure `competition_score.decide()` exists to prevent: it is the ONLY
    # function in this repository that returns a promotion boolean, it REQUIRES a
    # pre-registered gate (no default, so a gate supplied after the data is
    # visible shows up in the diff), it is ONE-SIDED, and it refuses percentile
    # and BCa outright.
    verdicts = {}
    if a.gate is None:
        print()
        print("  NO GATE SUPPLIED -- no verdict is printed. The numbers above are")
        print("  descriptive. Pass --gate <the pre-registered threshold> to get a")
        print("  decision, and only the threshold registered BEFORE the data.")
    else:
        print()
        print(f"  VERDICT vs {a.control}, pre-registered gate {a.gate:.4f}x, "
              f"studentised-t, one-sided")
        ctrl_scores = cs.score_suite([r.to_row() for r in ctrl_rows],
                                     lam=spec.lam, floor=floor).scores
        for name in arms:
            if name == a.control:
                continue
            cand_scores = cs.score_suite(
                [r.to_row() for r in results[name]["rows"]],
                lam=spec.lam, floor=floor).scores
            v = cs.decide(ctrl_scores, cand_scores, gate=a.gate, interval="t")
            verdicts[name] = v
            print(f"  {name:<18} {'PASS' if v.passed else 'fail':<5} "
                  f"gain {v.gain:>7.4f}x   "
                  f"t CI on the difference [{v.ci[0]:>+.3e}, {v.ci[1]:>+.3e}]"
                  f"   excl 0: {str(v.ci_excludes_zero):<5}"
                  f"   {v.n_candidate_better}/{v.n}"
                  f"   mover {v.biggest_mover_share * 100:>6.1f}%"
                  + ("   *** ESTIMANDS DISAGREE ***" if v.estimands_disagree else ""))
            for why in v.reasons:
                print(f"       - {why}")

    # ---- the mechanism curve ----------------------------------------------
    print()
    print(f"  MSE BY DEPTH -- the (L/n)^K mechanism curve, on {n_depth} of {n} "
          f"networks (a mechanism curve does not need all of them)")
    print("  L1 is exact for EVERY arm, so that column reads the truth-noise "
          "floor -- a built-in calibration")
    depth_columns = [d for d in (0, 1, 3, 7, 11, 15) if d < depth]
    if depth - 1 not in depth_columns:
        depth_columns.append(depth - 1)
    hdr = "  " + f"{'arm':<18}" + "".join(f"{'L' + str(d + 1):>11}"
                                          for d in depth_columns)
    print(hdr)
    for name in arms:
        bd = results[name]["by_depth"].mean(axis=0)
        print("  " + f"{name:<18}" + "".join(f"{bd[d]:>11.3e}"
                                             for d in depth_columns))

    if a.out:
        run_config = {
            "control": a.control,
            "residual_cap_override": a.residual_cap,
            "accuracy_only": a.residual_cap is not None,
            "official_samples": float(a.official_samples),
            "historical_timestamp_supplied": bool(a.ts),
        }
        bridge = {
            "fit_receipt": {"path": os.path.relpath(
                frozen["fit_receipt_path"], _ROOT).replace("\\", "/"),
                "sha256": frozen["fit_receipt_sha256"]},
            "freeze_receipt": {"path": os.path.relpath(
                frozen["receipt_path"], _ROOT).replace("\\", "/"),
                "sha256": frozen["receipt_sha256"]},
            "k4_lam": frozen["k4_lam"],
            "competition_cost_lambda": spec.lam,
            "protected_truth": truth_binding,
            "frozen_arm_hashes": frozen_hashes,
            "instrument_bindings": instrument_bindings,
            "locked_cli_source_bindings": (
                dict(_BOUND_CLI_BINDINGS) if _BOUND_CLI_BINDINGS is not None else
                {"mode": "explicit-in-process-test"}),
        }
        def before_publish():
            k4_freezer.revalidate_held_freeze(frozen)
            _revalidate_instruments(instrument_bindings)
        _write(a.out, results, paired, man, seeds, z, spec, arms, verdicts,
               a.gate, run_config=run_config, noise_aware=noise_aware,
               bridge=bridge, before_publish=before_publish)
        print(f"\n  artifact -> {os.path.relpath(a.out, _ROOT)}")
    print("=" * 92)
    return 0


def _predict_all(name, seeds, spec):
    """Deterministic re-run outside the budget, for the depth curve only.

    No score is computed from this path -- `lifecycle_eval` owns every scored
    number. This exists because the evaluator keeps the final layer and the
    mechanism curve needs all of them.
    """
    import flopscope as flops

    import harness
    from mint_provenance import canonical_mint

    mod, source_hash = le.load_candidate(_arm_path(name))
    est = mod.Estimator()
    est.setup(le.SetupContext(spec))
    finals, alls = [], []
    for s in seeds:
        w, es, _ = canonical_mint(spec.width, spec.depth, int(s))
        mlp = harness.MLP(weights=[np.asarray(x) for x in w], width=spec.width,
                          depth=spec.depth, seed=es)
        with flops.BudgetContext(flop_budget=spec.flop_budget, quiet=True):
            p = np.asarray(est.predict(mlp, spec.flop_budget), dtype=np.float64)
        alls.append(p)
        finals.append(p[-1])
    return np.asarray(finals), np.asarray(alls), source_hash


def _paired_payload(p):
    return {
        "mean_control": p.mean_control,
        "mean_candidate": p.mean_candidate,
        "arithmetic_gain": p.arithmetic_gain,
        "mean_difference": p.mean_difference,
        "t_ci_difference": list(p.t_ci_difference),
        "boot_ci_difference": list(p.boot_ci_difference),
        "relative_improvement": p.relative_improvement,
        "geometric_gain_DIAGNOSTIC_ONLY": p.geometric_gain,
        "n_candidate_better": p.n_candidate_better,
        "n": p.n,
        "sign_test_p": p.sign_test_p,
        "estimands_disagree": p.estimands_disagree,
    }


def _write(path, results, paired, man, seeds, z, spec, arms,
           verdicts=None, gate=None, *, run_config=None, noise_aware=None,
           bridge=None, before_publish=None):
    if os.path.lexists(path):
        raise SystemExit(f"REFUSED: {path} exists. Evidence is immutable.")
    run_config = dict(run_config or {})
    noise_aware = noise_aware or {}
    if not isinstance(bridge, dict):
        raise k4_freezer.FreezeRefused(
            "a validated K4 freeze bridge is required for report publication")
    control = run_config.get("control", CONTROL)
    n_samples = float(z["n_samples"])
    gt_per_replica = np.asarray(z["gt_noise"][:len(seeds)], dtype=np.float64)
    gt_average = gt_per_replica / 2.0
    rep_noise = _replica_noise(
        z["means_r1"][:len(seeds), -1, :].astype(np.float64),
        z["means_r2"][:len(seeds), -1, :].astype(np.float64))
    official_samples = float(run_config.get("official_samples", 1e9))
    payload = {
        "artifact": "p2_cp1_ladder",
        "artifact_version": 3,
        "status": "complete" if (set(arms) == set(MECHANISM_ARMS)
                                   and set(results) == set(arms)) else "incomplete",
        "is_probe_not_evidence": bool(int(z["probe"])) if "probe" in z else False,
        "shape": man["shape"],
        "budget": man["budget"],
        "package_parity": man["package_parity"],
        "truth": {
            "replica_noise_of_average_measured_mean": float(rep_noise.mean()),
            "replica_noise_of_average_measured_per_mlp": rep_noise.tolist(),
            "noise_floor_per_replica_mean": float(gt_per_replica.mean()),
            "noise_floor_of_two_replica_average_mean": float(gt_average.mean()),
            "noise_floor_of_two_replica_average_per_mlp": gt_average.tolist(),
            "noise_floor_half_factor": 0.5,
            "replica_independence_ratio_measured_to_predicted": (
                float(rep_noise.mean() / gt_average.mean())
                if gt_average.mean() else float("nan")),
            "n_samples_per_replica": int(n_samples),
            "local_effective_samples": int(2 * n_samples),
            "official_samples_for_forecast": official_samples,
            "n_networks": len(seeds),
            "replicas": 2,
        },
        "seeds": [int(s) for s in seeds],
        "k4_freeze": bridge,
        "competition_cost_lambda": spec.lam,
        "deciding_model": ("counted FLOPs only (lam=0), as frozen in the P2-CP1 "
                           "contract" if spec.lam == 0.0 else
                           "effective F + lam*R at the PHASE-1 lambda; this is "
                           "NOT the model the contract froze"),
        "competition_cost_lambda_provenance": "Phase 2 sets competition cost lambda to zero",
        "residual_is_graded": False,
        "control": control,
        "run_configuration": run_config,
        "pre_registered_gate": gate,
        "gate_note": (
            "the gate is registered in experiments/LOG.md BEFORE any data "
            "existed; decide() requires it explicitly and has no default"
        ),
        "noise_correction_identity": (
            "the averaged truth has half the per-replica floor; adjusted-score "
            "noise cancels only where per-network multipliers are equal"
        ),
        "arms": {},
    }
    for name in arms:
        r = results[name]
        payload["arms"][name] = {
            "objective": r["suite"].objective,
            "observed_objective": r["suite"].objective,
            "mean_mse_raw_vs_averaged_replica": float(r["raw"].mean()),
            "mean_mse_noise_corrected": float(r["corrected"].mean()),
            "mean_mse_cross_product_estimate": float(r["cross"].mean()),
            "per_mlp_mse_raw": [float(x) for x in r["raw"]],
            "per_mlp_mse_noise_corrected": [float(x) for x in r["corrected"]],
            "per_mlp_mse_cross_product_estimate": [float(x) for x in r["cross"]],
            "mean_flops": float(np.mean([x.flops_used for x in r["rows"]])),
            "flops_012": r["cost012"],
            "n_failed": r["suite"].n_failed,
            "mean_multiplier": r["suite"].mean_multiplier,
            "mse_by_depth": [float(x) for x in r["by_depth"].mean(axis=0)],
            "mse_by_depth_n_networks": int(r["n_depth"]),
            "peak_python_heap_mb": r["peak_mb"],
            "peak_memory_note": ("tracemalloc counts PYTHON allocations only; the "
                                 "grader's memory_limit_mb is RLIMIT_AS over the "
                                 "whole address space, so this is a LOWER BOUND"),
            "mean_residual_wall_time_s": float(np.mean(
                [x.residual_wall_time_s for x in r["rows"]])),
            "max_residual_wall_time_s": float(np.max(
                [x.residual_wall_time_s for x in r["rows"]])),
            "rows_hash": le.rows_hash(r["rows"]),
            "setup_seconds": r["prov"]["setup_seconds"],
            "candidate_sha256": r["prov"]["candidate_sha256"],
            # Per-MLP scores, so the verdict is reproducible from this file alone
            # without re-running a single arm.
            "per_mlp_score": [float(x) for x in cs.score_suite(
                [x.to_row() for x in r["rows"]],
                lam=spec.lam, floor=man["budget"]["multiplier_floor"]).scores],
            "per_mlp_ids": [x.to_row().name for x in r["rows"]],
            "per_mlp_flops": [float(x.flops_used) for x in r["rows"]],
            "per_mlp_counted_multiplier": [float(x.to_row().multiplier(
                0.0, man["budget"]["multiplier_floor"])) for x in r["rows"]],
            "per_mlp_effective_multiplier": [float(x.to_row().multiplier(
                spec.lam, man["budget"]["multiplier_floor"])) for x in r["rows"]],
            "per_mlp_residual_s": [float(x.residual_wall_time_s) for x in r["rows"]],
            "per_mlp_failed": [bool(x.failed) for x in r["rows"]],
            "per_mlp_error_code": [x.error_code for x in r["rows"]],
            "per_mlp_zero_prediction_mse": [float(x.zero_prediction_mse)
                                              for x in r["rows"]],
        }
        q = np.asarray(payload["arms"][name]["per_mlp_effective_multiplier"])
        official_floor = gt_average * (2.0 * n_samples / official_samples)
        payload["arms"][name]["true_function_objective_estimate"] = float(
            np.mean(q * r["corrected"]))
        payload["arms"][name]["official_precision_objective_forecast"] = float(
            np.mean(q * (r["corrected"] + official_floor)))
        if verdicts and name in verdicts:
            v = verdicts[name]
            payload["arms"][name]["verdict"] = {
                "passed": bool(v.passed),
                "gain": v.gain,
                "gate": v.gate,
                "ci_on_paired_difference": list(v.ci),
                "ci_excludes_zero": bool(v.ci_excludes_zero),
                "interval": v.interval,
                "n": v.n,
                "n_candidate_better": v.n_candidate_better,
                "sign_test_p": v.sign_test_p,
                "geometric_gain_DIAGNOSTIC": v.geometric_gain,
                "estimands_disagree_DIAGNOSTIC": bool(v.estimands_disagree),
                "biggest_mover_share_DIAGNOSTIC": v.biggest_mover_share,
                "reasons": list(v.reasons),
            }
        if name in paired:
            payload["arms"][name]["paired_vs_control"] = _paired_payload(
                paired[name])
        if name in noise_aware:
            nc = noise_aware[name]
            payload["arms"][name]["noise_aware_vs_control"] = {
                "observed": _paired_payload(nc.observed),
                "true_function": _paired_payload(nc.true_function),
                "official_precision_forecast": _paired_payload(nc.forecast),
                "noise_bias": nc.noise_bias,
                "noise_bias_share": nc.noise_bias_share,
                "local_samples": nc.local_samples,
                "official_samples": nc.official_samples,
                "n_negative_corrected": nc.n_negative_corrected,
                "notes": list(nc.notes),
            }
    build_ok = (payload["status"] == "complete" and
                all(payload["arms"][name]["n_failed"] == 0 for name in arms))
    payload["outcomes"] = {
        "build": {
            "kind": "build",
            "status": "PASS" if build_ok else "FAIL",
            "criterion": ("all four requested arms parsed, ran through the "
                          "canonical lifecycle, and produced no failed rows"),
        },
        "kill": {
            "kind": "kill",
            "status": "NOT_EVALUATED",
            "reason": ("this command has no separately registered kill rule; "
                       "failure to pass a promotion gate is not a kill"),
        },
        "mechanism": {
            "kind": "mechanism",
            "status": "DESCRIPTIVE",
            "arms": [name for name in ("consumers-only", "births-only")
                     if name in payload["arms"]],
            "reason": ("component-arm comparisons are reported separately and "
                       "do not return a promotion or kill boolean"),
        },
        "promotion": {
            "kind": "promotion",
            "status": ("NOT_EVALUATED" if gate is None else "EVALUATED"),
            "authority": "competition_score.decide(gate=..., interval='t')",
        },
    }
    if before_publish is None:
        raise k4_freezer.FreezeRefused(
            "closing dependency revalidation is required before report publication")
    before_publish()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        with open(path, "x", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=1, default=str) + "\n")
    except FileExistsError as exc:
        raise SystemExit(
            f"REFUSED: {path} appeared while the run was executing. Evidence "
            "is immutable; choose a new path.") from exc


if __name__ == "__main__":
    raise SystemExit(cli_entry())
