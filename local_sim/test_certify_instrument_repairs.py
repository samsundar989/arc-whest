"""Light tests for the 2026-09-16 certify_candidate instrument repairs.

Validation audit 2026-09-16:
  P0  the official smoke/soak must disable ONLY the local residual cap, and a
      capped, residual-zeroed report must not pass as functional evidence --
      while every non-residual failure still fails;
  P1  run integrity is dependency CONTENT identity, not `git status` equality.

No official run, no dataset, no truth, no candidate predict. `_run` is mocked
where a command is exercised. Small CPU-only.

Run: uv run python local_sim/test_certify_instrument_repairs.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# local_sim ships numpy shims NAMED flopscope/whestbench: APPEND, never insert.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import certify_candidate as cc  # noqa: E402

FLAG = "--no-residual-wall-time-limit"


def _row(i, failed_how=None, resid=0.9):
    m = {"mlp_index": i, "mlp_name": "mlp_%d" % i,
         "residual_wall_time_s": resid, "flops_used": 137600326656,
         "wall_time_s": 3.0, "budget_exhausted": False,
         "time_exhausted": False, "residual_wall_time_exhausted": False,
         "combined_budget_exhausted": False, "traceback": None}
    if failed_how == "error":
        m.update({"error_code": "ESTIMATOR_ERROR", "error": "boom",
                  "residual_wall_time_s": 0.0, "flops_used": 0,
                  "traceback": "Traceback (most recent call last):\n"
                               "ValueError: boom"})
    elif failed_how:
        m[failed_how] = True
    return m


def _report(rows, cap, mse=3.9e-8):
    failed = [m for m in rows if cc._row_failed(m)]
    return {
        "run_config": {"residual_wall_time_limit_s": cap,
                       "wall_time_limit_s": 120.0,
                       "flop_budget": 2 ** 41,
                       "lambda_flops_per_second": 0.0,
                       "width": 1024, "depth": 16,
                       "dataset": {"path": cc.OFFICIAL_DATASET}},
        "results": {"per_mlp": rows, "n_failed_mlps": len(failed),
                    "failure_breakdown": {}, "final_layer_mse": mse,
                    "mean_score_multiplier": 0.1,
                    "mean_compute_utilization": 0.06},
    }


class _Proc:
    def __init__(self, payload):
        self.stdout = json.dumps(payload)
        self.stderr = ""
        self.returncode = 0


class OfficialRunPath(unittest.TestCase):
    def test_canonical_command_disables_only_the_local_residual_cap(self):
        cmd = cc.official_run_cmd("cand.py", 2, exe="whest")
        self.assertEqual(cmd.count(FLAG), 1)
        self.assertEqual(cmd[:2], ["whest", "run"])
        for token in ("--residual-wall-time-limit", "--wall-time-limit",
                      "--flop-budget", "--lambda-flops-per-second",
                      "--setup-timeout"):
            self.assertNotIn(token, cmd)
        self.assertIn("subprocess", cmd)
        self.assertEqual(cmd[cmd.index("--dataset") + 1], cc.OFFICIAL_DATASET)

    def test_smoke_and_soak_build_their_command_through_it(self):
        seen = []

        def fake_run(argv, timeout=0, env=None):
            seen.append(list(argv))
            return _Proc(_report([_row(0, resid=0.9), _row(1, resid=0.8)],
                                 cap=None))

        with mock.patch.object(cc, "_run", fake_run):
            smoke = cc._smoke("cand.py")
            soak = cc._residual_soak("cand.py", reps=2, n_mlps=2)
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(FLAG in argv for argv in seen), seen)
        self.assertTrue(smoke["functional_verdict"]["ok"], smoke)
        self.assertIsNone(smoke["residual_cap"])
        self.assertEqual(smoke["server_residual_cap_s"], 0.4)
        # Over-cap LOCAL seconds do not fail a cap-disabled soak; they are
        # recorded as advisory against the server cap.
        self.assertTrue(soak["ok"], soak)
        self.assertEqual(soak["n_failed_mlps"], 0)
        self.assertTrue(soak["local_residual_cap_disabled_every_rep"])
        self.assertEqual(soak["residual_cap"], 0.4)
        self.assertLess(soak["headroom_worst"], 1.0)

    def test_capped_residual_only_report_is_not_the_corrected_path(self):
        rows = [_row(0, "residual_wall_time_exhausted"),
                _row(1, "residual_wall_time_exhausted")]
        verdict = cc.official_functional_verdict(_report(rows, cap=0.4))
        self.assertFalse(verdict["ok"])
        self.assertFalse(verdict["local_residual_cap_disabled"])
        self.assertTrue(verdict["server_residual_cap_exercised"])
        self.assertTrue(any("ARMED" in p for p in verdict["problems"]))
        # ...and through `_smoke`, the same capped output is refused.
        with mock.patch.object(cc, "_run", lambda *a, **k: _Proc(
                _report(rows, cap=0.4))):
            smoke = cc._smoke("cand.py")
        self.assertFalse(smoke["functional_verdict"]["ok"])

    def test_residual_exhaustion_with_cap_disabled_is_inconsistent(self):
        verdict = cc.official_functional_verdict(
            _report([_row(0, "residual_wall_time_exhausted")], cap=None))
        self.assertFalse(verdict["ok"])
        self.assertTrue(any("inconsistent" in p for p in verdict["problems"]))

    def test_clean_cap_disabled_report_passes(self):
        verdict = cc.official_functional_verdict(
            _report([_row(0), _row(1)], cap=None))
        self.assertTrue(verdict["ok"], verdict["problems"])
        self.assertFalse(verdict["server_residual_cap_exercised"])

    def test_estimator_exception_still_fails(self):
        verdict = cc.official_functional_verdict(
            _report([_row(0), _row(1, "error")], cap=None))
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["n_failed_mlps"], 1)
        self.assertEqual(verdict["failure_categories"],
                         {"estimator_error:ESTIMATOR_ERROR": 1})

    def test_budget_wall_and_nonfinite_failures_still_fail(self):
        for how in ("budget_exhausted", "time_exhausted",
                    "combined_budget_exhausted"):
            verdict = cc.official_functional_verdict(
                _report([_row(0, how)], cap=None))
            self.assertFalse(verdict["ok"], how)
            self.assertEqual(verdict["failure_categories"], {how: 1})
        verdict = cc.official_functional_verdict(
            _report([_row(0)], cap=None, mse=float("nan")))
        self.assertFalse(verdict["ok"])

    def test_declared_failure_count_must_match_rows(self):
        rep = _report([_row(0)], cap=None)
        rep["results"]["n_failed_mlps"] = 1
        self.assertFalse(cc.official_functional_verdict(rep)["ok"])


def _git(cwd, *args):
    return subprocess.run(["git", "-C", cwd] + list(args), capture_output=True,
                          text=True, check=True).stdout


class DependencyIdentity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="certify_identity_")
        self.lab = os.path.join(self.tmp, "local_sim")
        os.makedirs(os.path.join(self.lab, "flopscope"))  # a shim PACKAGE
        self._w("local_sim/root_tool.py",
                "import os\nimport helper\nimport flopscope\n")
        self._w("local_sim/helper.py",
                "def f():\n    from deep import g\n    return g()\n")
        self._w("local_sim/deep.py", "def g():\n    return 1\n")
        self._w("local_sim/unrelated_generator.py", "X = 1\n")
        self._w("local_sim/flopscope/__init__.py", "")
        self._w("candidate.py", "class Estimator: pass\n")
        _git(self.tmp, "init", "-q")
        _git(self.tmp, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
        _git(self.tmp, "-c", "user.email=t@t", "-c", "user.name=t",
             "commit", "-q", "-m", "fixture")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _w(self, rel, text):
        with open(os.path.join(self.tmp, rel), "w", encoding="utf-8",
                  newline="\n") as fh:
            fh.write(text)

    def _identity(self):
        return cc.dependency_identity(
            os.path.join(self.tmp, "candidate.py"), roots=("root_tool.py",),
            base=self.lab, root_dir=self.tmp)

    def _status(self):
        return _git(self.tmp, "status", "--porcelain")

    def test_closure_is_traced_transitively_and_skips_shim_packages(self):
        files = cc.instrument_dependency_files(("root_tool.py",), self.lab)
        self.assertEqual([os.path.basename(p) for p in files],
                         ["deep.py", "helper.py", "root_tool.py"])

    def test_unrelated_edit_does_not_invalidate(self):
        before = self._identity()
        status_before = self._status()
        self._w("local_sim/unrelated_generator.py", "X = 2\n")
        after = self._identity()
        self.assertNotEqual(self._status(), status_before)  # git WOULD refuse
        self.assertEqual(cc.dependency_identity_changes(before, after), [])

    def test_already_dirty_dependency_edited_during_run_invalidates(self):
        self._w("local_sim/deep.py", "def g():\n    return 2\n")  # dirty first
        before = self._identity()
        status_before = self._status()
        self._w("local_sim/deep.py", "def g():\n    return 3\n")
        after = self._identity()
        self.assertEqual(self._status(), status_before)  # git would NOT see it
        self.assertEqual(cc.dependency_identity_changes(before, after),
                         ["instruments.local_sim/deep.py"])

    def test_candidate_env_and_new_import_invalidate(self):
        before = self._identity()
        self._w("candidate.py", "class Estimator:\n    x = 1\n")
        self.assertIn("candidate.sha256",
                      cc.dependency_identity_changes(before, self._identity()))
        before = self._identity()
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "3"}):
            changed = cc.dependency_identity_changes(before, self._identity())
        self.assertEqual(changed, ["runtime.numerical_env.OMP_NUM_THREADS"])
        before = self._identity()
        self._w("local_sim/root_tool.py",
                "import os\nimport helper\nimport unrelated_generator\n")
        changed = cc.dependency_identity_changes(before, self._identity())
        self.assertIn("instruments.local_sim/unrelated_generator.py", changed)

    def test_missing_root_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            cc.instrument_dependency_files(("absent.py",), self.lab)

    def test_real_closure_binds_what_certify_executes(self):
        ident = cc.dependency_identity(os.path.join(_ROOT, "estimator.py"))
        keys = set(ident["instruments"])
        for name in cc.INSTRUMENT_ROOTS + ("harness.py", "seed_blocks.py",
                                           "competition_score.py",
                                           "grader_stack.py"):
            self.assertIn("local_sim/" + name, keys)
        self.assertFalse(any("/flopscope/" in k or "/whestbench/" in k
                             for k in keys))
        self.assertTrue(ident["packages"]["whestbench"]["version"])
        self.assertTrue(ident["packages"]["flopscope"]["version"])
        self.assertEqual(cc.dependency_identity_changes(
            ident, cc.dependency_identity(os.path.join(_ROOT, "estimator.py"))),
            [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
