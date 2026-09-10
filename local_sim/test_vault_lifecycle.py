"""The vault's candidate path runs the GRADER's lifecycle, or it is not evidence.

Stage 0 repair R7. `docs/research/2026-09-10_validation_audit.md` finding 2.

WHAT WAS WRONG. `vault.py::_run_candidate` was a second prediction loop, written
before `local_sim/lifecycle_eval.py` existed, and it got four things wrong at
once. Every one of them leaned the same way -- towards a number kinder than the
grader's:

  1. it built weights with `np.asarray(x)`: writeable float32 bare NumPy. The
     grader hands over float64 flopscope arrays that are READ-ONLY, and an
     operation invoked as a method on a weight is metered there and free on bare
     NumPy. The 2026-09-05 dtype incident cost seven submission slots to exactly
     this gap;
  2. it constructed a fresh `mod.Estimator()` for EVERY network and never called
     `setup()`. The grader builds one instance and calls `setup()` once,
     off-budget, so everything setup does is free -- and anything a candidate
     holds in instance state was billed 500 times here and zero times there;
  3. it never ran the official output materialisation, which whestbench 0.16
     performs INSIDE the metered window. On a float64 return that is a dtype
     conversion at 2 FLOPs/element -- 32,768 counted FLOPs at 1024x16, and 96 at
     the 3x16 fixture below;
  4. it had no failure paths at all. An exception in `predict` propagated and
     killed the run, and `_score_block` called `score_suite(on_over_budget=
     "fail")`, a mode the canonical scorer itself labels optimistic because it
     forces the failure MULTIPLIER while RETAINING the candidate's own MSE. A
     real failed MLP is scored against ZEROED predictions.

Separately, `vault.py:71` fixed `WIDTH, DEPTH = 256, 32` and defaulted the budget
to `272_000_000_000` -- both Phase 1's, with nothing recording that, and Phase 2
moved width, depth, budget, lambda, the wall limit and the residual limit all at
once.

Everything below runs on a SACRIFICIAL synthetic fixture in a temp directory:
four 16x3 networks, made-up truth, throwaway seeds. It never opens the vault's
arrays, never calls `vault.py evaluate`, and section 7 re-hashes every file under
`experiments/vault/` to prove the seal did not move. This suite measures
LIFECYCLE and FAILURE SEMANTICS, not accuracy, so the truth values only need to
be fixed and non-degenerate -- the same rationale `lifecycle_eval`'s own
self-test states.

WHY EACH CHECK DISCRIMINATES. A regression test that passes against the defect
tests nothing. At `0fe164c` this file cannot even import the repaired module --
`build_spec`, `RoundMismatch` and `_load_sacrificial` do not exist and
`_run_candidate` has a different signature -- so sections 1-4 additionally carry
a NEGATIVE CONTROL that reproduces the old behaviour through
`lifecycle_eval`'s own historical arms and shows the two answers side by side.
The contrast is measured here, not asserted from memory.

Run: uv run python local_sim/test_vault_lifecycle.py
"""

import dataclasses
import hashlib
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# APPEND, never insert. `local_sim/` holds numpy shims named `flopscope` and
# `whestbench`; prepending this directory makes them shadow the real packages,
# and every counted FLOP below would become fiction.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

import lifecycle_eval  # noqa: E402
import vault  # noqa: E402
from competition_score import (MlpRow, align_rows_by_identity,  # noqa: E402
                               score_suite, zero_prediction_mse)

RESULTS = []

#: The fixture. Small enough to run in a second, real enough to exercise the
#: whole lifecycle: `canonical_mint` is shape-generic, so these are genuine
#: grader-distribution networks at a shape no round uses.
WIDTH, DEPTH, N_NETS = 16, 3, 4
#: The round the fixture stands in for. Phase 1 on purpose: its residual limit is
#: `None`, so nothing in this suite can fail on a wall-clock measurement that
#: `docs/GOTCHAS.md` classifies E1 on this host.
FIXTURE_ROUND = "v1-phase1"
#: Legacy-namespace throwaway seeds, the `lifecycle_eval.py` convention: this
#: file measures MECHANICS, decides nothing, and must not draw on a registered
#: block. 5,160,000+ is outside every registered span (nearest are L3 at 900,000
#: and L2 at 7,000,000).
SEED0 = 5_160_000
#: The truth, and the value the "accurate" probe returns. Chosen so the
#: candidate's own MSE is EXACTLY 0 and the zeroed-prediction MSE is EXACTLY
#: 0.0625 -- both representable in float32, so section 4's contrast is exact
#: arithmetic rather than a tolerance.
TRUTH_VALUE = 0.25


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {name:<66} {'ok ' if ok else 'FAILED'}  {detail}")
    return ok


def head(t):
    print(f"\n{t}\n{'-' * len(t)}")


def _src(lines):
    """Build a throwaway probe's source from a line list.

    Deliberately no escape sequences: `lifecycle_eval.py` records that an earlier
    revision built these with embedded newline escapes and a tooling round-trip
    turned them into real newlines, producing a file that would not parse.
    """
    nl = chr(10)
    return nl.join(lines) + nl


# --------------------------------------------------------------------------- #
# the sacrificial fixture and the probes
# --------------------------------------------------------------------------- #


def _sandbox():
    return tempfile.mkdtemp(prefix="vault_lifecycle_")


def _fixture(d, *, round_tag=FIXTURE_ROUND, omit=()):
    """A sacrificial truth set, in the format `vault._load_sacrificial` requires.

    `width`, `depth` and `round_tag` are stored WITH the arrays. The old format
    carried only `seeds`/`truth`/`sigma_gt2`, so the shape came from `vault.py`'s
    Phase-1 literals and the rulebook came from a default argument -- which is
    how a fixture silently becomes evidence about a different experiment.
    `omit` drops fields, so section 6 can show an undeclared fixture is refused.
    """
    path = os.path.join(d, "sacrificial_truth.npz")
    fields = {
        "seeds": np.arange(SEED0, SEED0 + N_NETS, dtype=np.int64),
        "truth": np.full((N_NETS, WIDTH), TRUTH_VALUE, dtype=np.float64),
        "sigma_gt2": np.full(N_NETS, 1e-12, dtype=np.float64),
        "width": np.int64(WIDTH),
        "depth": np.int64(DEPTH),
        "round_tag": np.array(round_tag),
    }
    for k in omit:
        fields.pop(k)
    np.savez(path, **fields)
    return path


def _write_probe(d, name, *, ret="zeros_f64", burn=False):
    """A candidate FILE that reports what the lifecycle handed it.

    It writes one line per lifecycle event to `$ARC_WHEST_PROBE_LOG`, which is
    the only way to observe `__init__` and `setup` from outside: an estimator
    that is constructed four times and never set up looks, from its predictions
    alone, exactly like one that is constructed once and set up once.
    """
    body = [
        "import os",
        "import flopscope.numpy as fnp",
        "",
        "_LOG = os.environ['ARC_WHEST_PROBE_LOG']",
        "",
        "",
        "def _note(kind, detail=''):",
        "    with open(_LOG, 'a', encoding='utf-8') as fh:",
        "        fh.write(kind + chr(9) + detail + chr(10))",
        "",
        "",
        "class Estimator:",
        "    def __init__(self):",
        "        _note('init')",
        "",
        "    def setup(self, ctx):",
        "        _note('setup', str(ctx.width) + 'x' + str(ctx.depth))",
        "",
        "    def predict(self, mlp, budget):",
        "        w = mlp.weights[0]",
        "        _note('predict', '|'.join([str(mlp.seed), str(w.dtype),",
        "                                   str(bool(w.flags.writeable)),",
        "                                   type(w).__module__]))",
    ]
    if burn:
        # 2 * 16**3 = 8,192 counted FLOPs, so a declared budget of 1,000 raises
        # `BudgetExhaustedError` here rather than anywhere subtler.
        body.append("        _ = w @ w")
    # `fnp.zeros` is billed NOTHING, so a zeros-returning probe's entire counted
    # cost is the grader's in-window output materialisation and nothing else.
    # `fnp.full` is billed 2 FLOPs/element, which is why it is not used for the
    # materialisation measurement in section 3.
    body.append({
        "zeros_f64": "        return fnp.zeros((len(mlp.weights), mlp.width), "
                     "dtype=fnp.float64)",
        "zeros_f32": "        return fnp.zeros((len(mlp.weights), mlp.width), "
                     "dtype=fnp.float32)",
        "full_f64": "        return fnp.full((len(mlp.weights), mlp.width), "
                    f"{TRUTH_VALUE}, dtype=fnp.float64)",
    }[ret])
    path = os.path.join(d, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_src(body))
    return path


def _run(probe, log, *, fixture_dir, budget=None, weights_mode=None,
         lifecycle="grader"):
    """The vault's repaired candidate path, or the historical one, on the fixture.

    Both go through `lifecycle_eval`, which is the point: the historical arm is a
    NEGATIVE CONTROL driven by the canonical evaluator's own `_lifecycle`
    switch, not a reimplementation of the defect.
    """
    os.environ["ARC_WHEST_PROBE_LOG"] = log
    seeds = list(range(SEED0, SEED0 + N_NETS))
    truth = np.full((N_NETS, WIDTH), TRUTH_VALUE, dtype=np.float64)
    gt = np.full(N_NETS, 1e-12, dtype=np.float64)
    spec = vault.build_spec(FIXTURE_ROUND, width=WIDTH, depth=DEPTH,
                            budget=budget, sacrificial_shape=True)
    if lifecycle == "grader":
        rows, prov, preds = vault._run_candidate(
            probe, seeds, truth, spec, gt, weights_mode=weights_mode,
            progress_every=0)
    else:
        rows, prov = lifecycle_eval.evaluate(
            probe, seeds, truth, spec=spec, gt_noise=gt,
            weights_mode=weights_mode or lifecycle_eval.DEFAULT_WEIGHTS_MODE,
            _lifecycle=lifecycle)
        preds = None
    return rows, prov, preds, spec


def _events(log):
    if not os.path.exists(log):
        return []
    out = []
    with open(log, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip("\n"):
                continue
            k, _, detail = line.rstrip("\n").partition("\t")
            out.append((k, detail))
    return out


# --------------------------------------------------------------------------- #
# 1. one estimator, one setup, for the whole suite
# --------------------------------------------------------------------------- #


def t_setup_once():
    head("1. setup() is called ONCE for the suite, not never and not per network")
    d = _sandbox()
    probe = _write_probe(d, "probe_zeros64.py")

    _rows, prov, _p, _s = _run(probe, os.path.join(d, "grader.log"),
                               fixture_dir=d)
    ev = _events(os.path.join(d, "grader.log"))
    n_init = sum(1 for k, _ in ev if k == "init")
    n_setup = sum(1 for k, _ in ev if k == "setup")
    n_pred = sum(1 for k, _ in ev if k == "predict")

    check("the candidate is constructed EXACTLY ONCE for 4 networks",
          n_init == 1, f"{n_init} __init__ calls")
    check("setup() is called EXACTLY ONCE", n_setup == 1,
          f"{n_setup} setup calls")
    check("...and it is called BEFORE any prediction",
          bool(ev) and ev[0][0] in ("init", "setup")
          and [k for k, _ in ev].index("setup") < [k for k, _ in ev].index("predict"),
          "the grader sets up off-budget, then predicts")
    check("...with the SPEC's shape, not a transcribed literal",
          n_setup == 1 and [v for k, v in ev if k == "setup"][0] == f"{WIDTH}x{DEPTH}",
          [v for k, v in ev if k == "setup"][:1])
    check("all 4 networks were predicted", n_pred == N_NETS, f"{n_pred}")
    check("...and the provenance SAYS setup ran, rather than leaving it inferred",
          prov["setup_called"] is True and prov["lifecycle"] == "grader",
          f"lifecycle={prov['lifecycle']}")

    # NEGATIVE CONTROL: the historical vault lifecycle, through the canonical
    # evaluator's own switch. This is what `_run_candidate` did before the repair.
    _rows, oprov, _p, _s = _run(probe, os.path.join(d, "old.log"),
                                fixture_dir=d, lifecycle="per_mlp_no_setup")
    oev = _events(os.path.join(d, "old.log"))
    o_init = sum(1 for k, _ in oev if k == "init")
    o_setup = sum(1 for k, _ in oev if k == "setup")
    check("NEGATIVE CONTROL: the old path built one estimator PER NETWORK",
          o_init == N_NETS and o_setup == 0,
          f"{o_init} __init__, {o_setup} setup — a carrier built in setup() "
          f"would have been billed {N_NETS} times and the grader bills it 0")


# --------------------------------------------------------------------------- #
# 2. the grader's weight objects
# --------------------------------------------------------------------------- #


def t_weights_are_grader_twins():
    head("2. weights reach the estimator as READ-ONLY float64 flopscope arrays")
    d = _sandbox()
    probe = _write_probe(d, "probe_zeros64.py")

    _rows, prov, _p, _s = _run(probe, os.path.join(d, "twin.log"), fixture_dir=d)
    seen = [v.split("|") for k, v in _events(os.path.join(d, "twin.log"))
            if k == "predict"]
    dtypes = {s[1] for s in seen}
    writeable = {s[2] for s in seen}
    modules = {s[3] for s in seen}

    check("every weight arrives as float64", dtypes == {"float64"}, str(dtypes))
    check("every weight arrives READ-ONLY", writeable == {"False"},
          "an in-place write is a crash on the grader and a silent success on a "
          "writeable copy")
    check("every weight is flopscope-backed, not bare NumPy",
          modules == {"flopscope._ndarray"}, str(modules))
    check("...and the run DECLARES the arm it used",
          prov["weights_mode"] == "f64_readonly"
          and prov["weights_mode_is_grader_default"] is True,
          prov["weights_mode"])

    # NEGATIVE CONTROL: the historical local world, which is what `np.asarray(x)`
    # produced. Kept as an arm so its delta is measured rather than asserted.
    _rows, _prov, _p, _s = _run(probe, os.path.join(d, "f32.log"),
                                fixture_dir=d, weights_mode="f32")
    old = [v.split("|") for k, v in _events(os.path.join(d, "f32.log"))
           if k == "predict"]
    check("NEGATIVE CONTROL: the historical arm hands over WRITEABLE float32",
          {s[1] for s in old} == {"float32"} and {s[2] for s in old} == {"True"},
          "the exact pair a `x.dtype == dtype` guard short-circuits on")


# --------------------------------------------------------------------------- #
# 3. the in-window output materialisation
# --------------------------------------------------------------------------- #


def t_materialisation_is_billed():
    head("3. predictions are materialised, and the FLOPs match lifecycle_eval's")
    d = _sandbox()
    p64 = _write_probe(d, "probe_zeros64.py", ret="zeros_f64")
    p32 = _write_probe(d, "probe_zeros32.py", ret="zeros_f32")

    r64, prov64, _p, spec = _run(p64, os.path.join(d, "m64.log"), fixture_dir=d)
    r32, _prov32, _p, _s = _run(p32, os.path.join(d, "m32.log"), fixture_dir=d)

    # `fnp.zeros` is billed nothing, so a zeros-returning probe's ENTIRE counted
    # cost is the materialisation whestbench 0.16 runs inside the metered window.
    expected = 2 * DEPTH * WIDTH
    f64 = {r.flops_used for r in r64}
    f32 = {r.flops_used for r in r32}
    check("a float64 return is billed EXACTLY 2 FLOPs/element for the cast",
          f64 == {float(expected)},
          f"{sorted(f64)} vs 2*{DEPTH}*{WIDTH} = {expected}")
    check("a float32 return is billed NOTHING, because no cast happens",
          f32 == {0.0}, f"{sorted(f32)}")
    check("...so the materialisation surcharge is exactly the dtype conversion",
          (max(f64) - max(f32)) == float(expected),
          f"{max(f64) - max(f32):,.0f} FLOPs — 32,768 at the live 1024x16 shape")

    # The reuse claim, checked rather than described: the vault's candidate path
    # and a direct call to the canonical evaluator produce the same rows, bit for
    # bit. `rows_hash` covers every DECIDED quantity -- flops, MSE, failure
    # flags, identity -- and excludes only the wall-clock fields.
    direct_rows, direct_prov = lifecycle_eval.evaluate(
        p64, list(range(SEED0, SEED0 + N_NETS)),
        np.full((N_NETS, WIDTH), TRUTH_VALUE, dtype=np.float64),
        spec=spec, gt_noise=np.full(N_NETS, 1e-12, dtype=np.float64))
    check("the vault path and lifecycle_eval produce the SAME rows, bit for bit",
          prov64["rows_hash"] == direct_prov["rows_hash"],
          f"{prov64['rows_hash'][:24]}...")
    check("...and the same counted FLOPs per network",
          [r.flops_used for r in r64] == [r.flops_used for r in direct_rows],
          "one lifecycle, one implementation, one answer")
    check("...and the evaluator names the op ledger it measured",
          all(np.isfinite(r.n_ops) and r.n_ops > 0 for r in r64),
          f"worst {max(r.n_ops for r in r64):.0f} ops")


# --------------------------------------------------------------------------- #
# 4. the grader's real failure semantics
# --------------------------------------------------------------------------- #


def t_failure_semantics():
    head("4. a failed network is scored against ZEROED predictions, not its own MSE")
    d = _sandbox()
    good = _write_probe(d, "probe_accurate.py", ret="full_f64")
    burn = _write_probe(d, "probe_burn.py", ret="full_f64", burn=True)

    g_rows, _gprov, _p, g_spec = _run(good, os.path.join(d, "good.log"),
                                      fixture_dir=d)
    check("the accurate probe scores EXACTLY zero MSE against its own truth",
          all(r.mse == 0.0 for r in g_rows), f"{[r.mse for r in g_rows][:1]}")

    # A declared budget DEVIATION: 1,000 FLOPs against a 16x16 matmul that costs
    # 8,192. The point is the failure, not the number.
    b_rows, bprov, _p, b_spec = _run(burn, os.path.join(d, "burn.log"),
                                     fixture_dir=d, budget=1_000)
    zmse = zero_prediction_mse(np.full(WIDTH, TRUTH_VALUE, dtype=np.float32))

    check("a network that blows the budget is FAILED, on the budget path",
          all(r.budget_exhausted and r.failed for r in b_rows),
          f"{sum(r.budget_exhausted for r in b_rows)}/{N_NETS}")
    check("...its MSE is EXACTLY mean(target^2), i.e. the prediction was zeroed",
          all(r.mse == zmse for r in b_rows),
          f"{b_rows[0].mse:.6f} == {zmse:.6f}")
    check("...and it is priced at multiplier EXACTLY 1.0, not the 0.1 floor",
          all(r.to_row().multiplier(b_spec.lam, b_spec.floor) == 1.0
              for r in b_rows),
          "a failure is not a cheap MLP")

    real = vault._score_block(b_rows, bprov, b_spec, {})
    check("the vault's own report counts the failures and names the path",
          real["n_failed"] == N_NETS
          and real["failure_paths"]["budget_exhausted"] == N_NETS,
          f"{real['failure_paths']}")
    check("...and reports the failed objective, mean(target^2) * 1.0",
          real["objective"] == zmse, f"{real['objective']:.6f}")

    # NEGATIVE CONTROL, and the exact shape of the old defect. `_score_block`
    # built `MlpRow`s carrying the CANDIDATE's MSE with `failed` unset, then
    # called `score_suite(on_over_budget="fail")`. That mode forces the failure
    # multiplier and keeps the MSE, which `competition_score` labels optimistic in
    # its own docstring -- it cannot zero a prediction it never saw. Here the
    # candidate's own MSE is 0, so the optimistic path reports a PERFECT score for
    # a submission the grader flattens.
    optimistic = score_suite(
        [MlpRow(mse=g.mse, flops_used=float(b_spec.flop_budget) * 2.0,
                residual_wall_time_s=0.0, budget=float(b_spec.flop_budget))
         for g in g_rows],
        lam=b_spec.lam, floor=b_spec.floor, on_over_budget="fail")
    check("NEGATIVE CONTROL: the retained-MSE path reports a PERFECT score here",
          optimistic.objective == 0.0 and real["objective"] > 0.0,
          f"optimistic {optimistic.objective:.6f} vs real "
          f"{real['objective']:.6f} — the same rows, one of them fiction")
    # And the mode is gone behaviourally, not just textually: hand `_score_block`
    # a row that is over budget and NOT flagged -- the exact shape the old code
    # quietly rewrote -- and it must refuse rather than score it.
    smuggled = [dataclasses.replace(
        r, flops_used=float(b_spec.flop_budget) * 2.0, budget_exhausted=False,
        time_exhausted=False, residual_exhausted=False,
        combined_exhausted=False, error_code="") for r in b_rows]
    refused = False
    try:
        vault._score_block(smuggled, bprov, b_spec, {})
    except ValueError:
        refused = True
    check("an unflagged over-budget row is REFUSED, not rewritten as optimistic",
          refused,
          "the old path took on_over_budget='fail' and scored it with its own MSE")


# --------------------------------------------------------------------------- #
# 5. rows carry a network identity
# --------------------------------------------------------------------------- #


def t_rows_have_identity():
    head("5. every row carries the network's identity, so two arms can be paired")
    d = _sandbox()
    probe = _write_probe(d, "probe_zeros64.py")
    rows, _prov, _p, _spec = _run(probe, os.path.join(d, "id.log"), fixture_dir=d)

    names = [r.to_row().name for r in rows]
    expected = [f"seed:{SEED0 + i}" for i in range(N_NETS)]
    check("each row is named by its INPUT SEED, which is what the network is",
          names == expected, str(names[:2]))
    check("...and the names are distinct", len(set(names)) == N_NETS, "")

    # The identity is only worth having if the pairing machinery will use it.
    # A permuted candidate arm must realign, because equal length is not common
    # random numbers -- mispairing leaves the mean untouched and inflates the
    # standard error by a median 8.55x (competition_score.align_rows_by_identity).
    arm = [r.to_row() for r in rows]
    permuted = [arm[i] for i in (2, 0, 3, 1)]
    ctrl, cand, _notes = align_rows_by_identity(arm, permuted, require_ids=True)
    check("a PERMUTED second arm realigns by name under require_ids=True",
          [c.name for c in ctrl] == [k.name for k in cand],
          "pairing follows the network, not the list position")

    # NEGATIVE CONTROL: the rows the old `_score_block` built carried no `name`
    # at all, so a contract that demands identities has nothing to align on.
    unnamed = [MlpRow(mse=float(r.mse), flops_used=float(r.flops_used),
                      residual_wall_time_s=0.0, budget=float(r.budget))
               for r in rows]
    refused = False
    try:
        align_rows_by_identity(unnamed, unnamed, require_ids=True)
    except Exception:
        refused = True
    check("NEGATIVE CONTROL: unnamed rows are REFUSED by an identified design",
          refused, "the old rows had no identity to refuse on")


# --------------------------------------------------------------------------- #
# 6. a round mismatch is refused
# --------------------------------------------------------------------------- #


def t_round_mismatch_is_refused():
    head("6. a shape, a corpus and a rulebook that disagree are REFUSED")

    def _refuses(fn, what):
        try:
            fn()
        except vault.RoundMismatch as exc:
            return True, str(exc).splitlines()[0][:58]
        except Exception as exc:  # noqa: BLE001
            return False, f"raised {type(exc).__name__}, not RoundMismatch"
        return False, "IT WAS PERMITTED"

    ok, why = _refuses(
        lambda: vault.build_spec("v2-phase2", width=256, depth=32),
        "phase-1 shape under phase-2")
    check("256x32 networks under the LIVE round are refused", ok, why)

    ok, why = _refuses(
        lambda: vault.build_spec("v1-phase1", width=1024, depth=16),
        "phase-2 shape under phase-1")
    check("...and 1024x16 under Phase 1 is refused too, in both directions",
          ok, why)

    ok, why = _refuses(lambda: vault.build_spec("v3-imaginary", width=16, depth=3),
                       "unknown round")
    check("a round nobody has heard of is refused, not defaulted", ok, why)

    # The SEALED entry point, checked on a synthetic manifest so the real one is
    # never opened. This is the refusal that keeps Phase-1 arrays from being
    # reported as Phase-2 evidence.
    ok, why = _refuses(
        lambda: vault._refuse_unless_corpus_matches_live_round(
            {"width": 256, "depth": 32, "n_mlps": 500,
             "created_utc": "2026-08-06T00:00:00Z", "seed_block": "V"}),
        "phase-1 corpus")
    check("a PHASE-1 sealed corpus refuses evaluation under the live round",
          ok, why)
    check("...and the refusal says a Phase-2 vault needs a STRATEGIC contract",
          ok and "STRATEGIC" in _refusal_text(),
          "a refusal that does not say what would unblock it is a dead end")

    live = vault.official_config.LIVE
    permitted = True
    try:
        vault._refuse_unless_corpus_matches_live_round(
            {"width": live.width, "depth": live.depth, "n_mlps": 1,
             "created_utc": "", "seed_block": "?"})
    except Exception:  # noqa: BLE001
        permitted = False
    check("...but a corpus AT the live shape is permitted, so the gate is not a wall",
          permitted, f"{live.width}x{live.depth} passes")

    # A fixture that declares one round and is asked for another.
    d = _sandbox()
    fx = _fixture(d, round_tag="v1-phase1")
    ok, why = _refuses(lambda: vault._load_sacrificial(fx, "v2-phase2"),
                       "fixture round mismatch")
    check("a fixture asked for a round it does not declare is refused", ok, why)

    same = vault._load_sacrificial(fx, "v1-phase1")
    check("...and the same fixture loads under the round it DOES declare",
          same[5] == "v1-phase1" and same[3] == WIDTH and same[4] == DEPTH,
          f"{same[3]}x{same[4]} @ {same[5]}")

    bare = _fixture(_sandbox(), omit=("round_tag", "width", "depth"))
    undeclared = False
    try:
        vault._load_sacrificial(bare, None)
    except SystemExit:
        undeclared = True
    check("a fixture that declares NO shape and NO round is refused",
          undeclared,
          "the old format had none of the three and inherited Phase-1 literals")

    # A declared budget deviation must travel, not vanish.
    dev = vault.build_spec(FIXTURE_ROUND, width=WIDTH, depth=DEPTH,
                           budget=1_000, sacrificial_shape=True)
    devs = dev.deviations_from_round()
    check("a synthetic shape and a budget override are RECORDED as deviations",
          devs.get("flop_budget", [None])[0] == 1_000
          and devs.get("width", [None])[0] == WIDTH
          and devs.get("depth", [None])[0] == DEPTH,
          f"{sorted(devs)}")
    clean = vault.build_spec(FIXTURE_ROUND, width=256, depth=32)
    check("...and a clean round reports NO deviation at all",
          clean.deviations_from_round() == {}
          and clean.flop_budget == 272_000_000_000,
          f"B={clean.flop_budget:,} from the round, not from a default argument")


def _refusal_text():
    try:
        vault._refuse_unless_corpus_matches_live_round(
            {"width": 256, "depth": 32, "n_mlps": 500,
             "created_utc": "", "seed_block": "V"})
    except vault.RoundMismatch as exc:
        return str(exc)
    return ""


# --------------------------------------------------------------------------- #
# 7. end to end, and the seal never moves
# --------------------------------------------------------------------------- #


def _vault_dir_state():
    """sha256 of every file under experiments/vault/, plus its listing."""
    out = {}
    for root, _dirs, files in os.walk(vault.VAULT_DIR):
        for f in sorted(files):
            p = os.path.join(root, f)
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out[os.path.relpath(p, vault.VAULT_DIR)] = h.hexdigest()
    return out


def t_end_to_end(before):
    head("7. the whole sacrificial command, and the seal it must not touch")
    d = _sandbox()
    fx = _fixture(d)
    probe = _write_probe(d, "probe_accurate.py", ret="full_f64")
    os.environ["ARC_WHEST_PROBE_LOG"] = os.path.join(d, "e2e.log")

    res = vault.evaluate("sacrificial", "seal", probe, None, 0, fx,
                         round_tag=FIXTURE_ROUND, evidence_dir=d)
    check("the sacrificial command runs the repaired path end to end",
          res["n_mlps"] == N_NETS and res["setup_called"] is True
          and res["lifecycle"] == "grader", f"{res['n_mlps']} networks")
    check("...it took NO touch and recorded none", res["touch"] is None,
          "sacrificial: nothing recorded, seal untouched")
    check("...it names its round and its deviations from it",
          res["round_tag"] == FIXTURE_ROUND
          and set(res["round_deviations"]) == {"width", "depth"},
          f"{res['round_tag']}, deviations {sorted(res['round_deviations'])}")
    check("...it reports lambda from the SPEC, not from a module default",
          res["lambda"] == vault.official_config.by_tag(
              FIXTURE_ROUND).lambda_flops_per_second,
          f"{res['lambda']:.3e}")
    ev = [f for f in os.listdir(d) if f.startswith("lifecycle_evidence_")]
    check("...and it wrote an append-only evidence bundle beside the FIXTURE",
          len([f for f in ev if f.endswith(".json")]) == 1
          and len([f for f in ev if f.endswith("_preds.npz")]) == 1,
          f"{sorted(ev)}")

    after = _vault_dir_state()
    check("NOTHING under experiments/vault/ changed, added or vanished",
          after == before,
          f"{len(before)} files, sha256 identical"
          if after == before
          else f"CHANGED: {sorted(set(before) ^ set(after)) or 'contents'}")


def main():
    print("=" * 96)
    print("THE VAULT'S CANDIDATE PATH IS THE GRADER'S LIFECYCLE — regression suite")
    print("=" * 96)
    before = _vault_dir_state()
    t_setup_once()
    t_weights_are_grader_twins()
    t_materialisation_is_billed()
    t_failure_semantics()
    t_rows_have_identity()
    t_round_mismatch_is_refused()
    t_end_to_end(before)
    n_ok = sum(1 for r in RESULTS if r)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
