"""Crash-evidence tests for `scripts/run_mini_json.ps1`, with a FAKE whest.

No real `whest run`, no dataset, no truth: the launcher seam points the script
at a tiny Python program that prints a canned report. Validation audit
2026-09-16 P2: raw stdout/stderr must survive a truncated report and a nonzero
exit, and a valid report must still publish as UTF-8 without a BOM.

Run: uv run python local_sim/test_run_mini_json.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.append(_HERE)
SCRIPT = os.path.join(_ROOT, "scripts", "run_mini_json.ps1")

FAKE = r'''
import json, sys
mode = sys.argv[1]
args = sys.argv[2:]
sys.stderr.write("fake whest stderr line\n")
report = {"run_config": {"residual_wall_time_limit_s": None},
          "results": {"n_failed_mlps": 0, "per_mlp": []},
          "argv": args, "note": "café"}
text = json.dumps(report, indent=2, ensure_ascii=False)
if mode == "truncated":
    sys.stdout.write(text[: len(text) // 2])
    sys.stdout.flush()
    sys.exit(3)
if mode == "valid_exit1":
    sys.stdout.write(text + "\n")
    sys.exit(1)
sys.stdout.write(text + "\n")
'''


def _ps_quote(s):
    return "'" + s.replace("'", "''") + "'"


# A PowerShell-7-only Microsoft.PowerShell.Utility module. Put first on the
# child's PSModulePath it reproduces the Codex driver defect (validation review
# 2026-09-17 section 3): Windows PowerShell 5.1 auto-loads Get-FileHash from it
# and fails.
_CORE_ONLY_UTILITY = (
    '@{\n'
    'GUID="1DA87E53-152B-403E-98DC-74D7B4D63D59"\n'
    'ModuleVersion="7.0.0.0"\n'
    'CompatiblePSEditions=@("Core")\n'
    'PowerShellVersion="7.0"\n'
    'NestedModules="Microsoft.PowerShell.Commands.Utility.dll"\n'
    'CmdletsToExport="Get-FileHash"\n'
    '}\n')
_WINPS_MODULES = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"),
                              "System32", "WindowsPowerShell", "v1.0", "Modules")


def _child_env(tmp, kind):
    # os.environ upper-cases keys on Windows; one spelling only, or the child
    # sees two PSModulePath entries and Start-Process refuses the environment.
    env = {k: v for k, v in os.environ.items() if k.upper() != "PSMODULEPATH"}
    inherited = os.environ.get("PSModulePath", _WINPS_MODULES)
    env["PSModulePath"] = inherited
    if kind == "inherited":
        return env
    if kind == "clean":
        env["PSModulePath"] = _WINPS_MODULES
        return env
    assert kind == "poisoned", kind
    mods = os.path.join(tmp, "ps7mods")
    mdir = os.path.join(mods, "Microsoft.PowerShell.Utility")
    os.makedirs(mdir, exist_ok=True)
    with open(os.path.join(mdir, "Microsoft.PowerShell.Utility.psd1"), "w",
              encoding="utf-8") as fh:
        fh.write(_CORE_ONLY_UTILITY)
    env["PSModulePath"] = mods + os.pathsep + inherited
    return env


def _run_script(tmp, mode, out, estimator="est with space.py", env_kind="inherited"):
    fake = os.path.join(tmp, "fake_whest.py")
    with open(fake, "w", encoding="utf-8") as fh:
        fh.write(FAKE)
    command = ("& %s -Estimator %s -Out %s -NoResidualLimit -LauncherExe %s "
               "-LauncherArgs @(%s, %s)" % (
                   _ps_quote(SCRIPT), _ps_quote(estimator),
                   _ps_quote(out), _ps_quote(sys.executable),
                   _ps_quote(fake), _ps_quote(mode)))
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy",
         "Bypass", "-Command", command],
        capture_output=True, text=True, errors="replace", timeout=180,
        env=_child_env(tmp, env_kind))


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


@unittest.skipUnless(sys.platform == "win32", "Windows PowerShell 5.1 script")
class RunMiniJsonEvidence(unittest.TestCase):
    def test_truncated_json_nonzero_exit_keeps_raw_evidence(self):
        with tempfile.TemporaryDirectory(prefix="run_mini_json_") as tmp:
            out = os.path.join(tmp, "sub dir", "report.json")
            p = _run_script(tmp, "truncated", out)
            self.assertNotEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertFalse(os.path.exists(out), "invalid JSON was published")
            raw_out = glob.glob(out + ".*.stdout.raw")
            raw_err = glob.glob(out + ".*.stderr.raw")
            meta = glob.glob(out + ".*.failure.json")
            self.assertEqual((len(raw_out), len(raw_err), len(meta)), (1, 1, 1),
                             os.listdir(os.path.dirname(out)))
            with open(raw_out[0], "rb") as fh:
                raw = fh.read()
            self.assertTrue(raw.startswith(b"{"), raw[:40])
            with self.assertRaises(ValueError):
                json.loads(raw.decode("utf-8"))
            with open(raw_err[0], "rb") as fh:
                self.assertIn(b"fake whest stderr line", fh.read())
            with open(meta[0], "rb") as fh:
                meta_bytes = fh.read()
            self.assertFalse(meta_bytes.startswith(b"\xef\xbb\xbf"))
            record = json.loads(meta_bytes.decode("utf-8"))
            self.assertEqual(record["exit_code"], 3)
            self.assertFalse(record["json_valid"])
            self.assertIn("not JSON", record["reason"])
            self.assertEqual(record["stdout_raw"]["sha256"], _sha(raw_out[0]))
            self.assertEqual(record["stdout_raw"]["bytes"], len(raw))
            self.assertIn("--no-residual-wall-time-limit", record["arguments"])
            self.assertFalse(glob.glob(out + ".pending-*"))

    def test_valid_json_publishes_utf8_without_bom(self):
        with tempfile.TemporaryDirectory(prefix="run_mini_json_") as tmp:
            out = os.path.join(tmp, "report.json")
            p = _run_script(tmp, "valid", out)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            with open(out, "rb") as fh:
                data = fh.read()
            self.assertFalse(data.startswith(b"\xef\xbb\xbf"))
            self.assertFalse(data.startswith(b"\xff\xfe"))
            report = json.loads(data.decode("utf-8"))
            self.assertEqual(report["note"], "café")
            # The argument with a space survived Start-Process as ONE argument,
            # and the residual switch reached the launcher.
            self.assertIn("est with space.py", report["argv"])
            self.assertIn("--no-residual-wall-time-limit", report["argv"])
            self.assertEqual(len(glob.glob(out + ".*.stdout.raw")), 1)
            self.assertEqual(len(glob.glob(out + ".*.stderr.raw")), 1)
            self.assertEqual(glob.glob(out + ".*.failure.json"), [])
            # the run receipt is bound to the published report's bytes
            with open(os.path.join(tmp, "report.receipt.json"), "rb") as fh:
                rec = json.loads(fh.read().decode("utf-8"))
            self.assertEqual(rec["schema"], "arc-whest.run_receipt/1")
            self.assertEqual(rec["report_sha256"], _sha(out))
            self.assertEqual(rec["meter_path"], "whest_run_subprocess")
            self.assertFalse(rec["source_changed_during_run"])

    def test_valid_json_nonzero_exit_publishes_then_fails(self):
        with tempfile.TemporaryDirectory(prefix="run_mini_json_") as tmp:
            out = os.path.join(tmp, "report.json")
            p = _run_script(tmp, "valid_exit1", out)
            self.assertNotEqual(p.returncode, 0)
            self.assertTrue(os.path.isfile(out))
            meta = glob.glob(out + ".*.failure.json")
            self.assertEqual(len(meta), 1)
            with open(meta[0], encoding="utf-8") as fh:
                record = json.load(fh)
            self.assertTrue(record["json_valid"])
            self.assertEqual(record["exit_code"], 1)
            self.assertEqual(record["report"]["sha256"], _sha(out))

    def test_existing_output_is_refused_before_launch(self):
        with tempfile.TemporaryDirectory(prefix="run_mini_json_") as tmp:
            out = os.path.join(tmp, "report.json")
            with open(out, "w", encoding="utf-8") as fh:
                fh.write("{}")
            p = _run_script(tmp, "valid", out)
            self.assertNotEqual(p.returncode, 0)
            self.assertEqual(glob.glob(out + ".*.stdout.raw"), [])
            with open(out, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "{}")


# A real, existing estimator: hashing must actually run, and its receipt must be
# discoverable by the projector with NO explicit receipt path and NO override.
REAL_CANDIDATE = os.path.join(_ROOT, "estimator_k3_v31.py")


@unittest.skipUnless(sys.platform == "win32", "Windows PowerShell 5.1 script")
class CrossToolReceipt(unittest.TestCase):
    """run_mini_json.ps1 output -> predict_public default discovery -> source binding."""

    def _success_then_project(self, env_kind):
        import predict_public as pp
        with tempfile.TemporaryDirectory(prefix="run_mini_json_x_") as tmp:
            out = os.path.join(tmp, "cand mini.json")
            p = _run_script(tmp, "valid", out, estimator=REAL_CANDIDATE, env_kind=env_kind)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertFalse(os.path.exists(out + ".receipt.json"), "legacy name written")
            found = pp.find_receipt(out)
            self.assertEqual(found, os.path.join(tmp, "cand mini.receipt.json"))
            rec = pp.load_receipt(found, out, label="candidate")
            self.assertEqual(rec["source_sha256"], _sha(REAL_CANDIDATE))
            bound = pp.bind_candidate_source(rec, REAL_CANDIDATE, allow_unverified=False)
            self.assertEqual(bound["status"], "VERIFIED")
            # a different file is refused, and the override does not rescue it
            with self.assertRaises(pp.ProjectionRefused):
                pp.bind_candidate_source(rec, SCRIPT, allow_unverified=True)

    def test_inherited_environment(self):
        self._success_then_project("inherited")

    def test_clean_child_environment(self):
        self._success_then_project("clean")

    def test_poisoned_ps7_module_path(self):
        self._success_then_project("poisoned")

    def test_poisoned_failure_paths_still_hash(self):
        import predict_public as pp
        with tempfile.TemporaryDirectory(prefix="run_mini_json_x_") as tmp:
            out = os.path.join(tmp, "report.json")
            p = _run_script(tmp, "truncated", out, estimator=REAL_CANDIDATE,
                            env_kind="poisoned")
            self.assertNotEqual(p.returncode, 0)
            meta = glob.glob(out + ".*.failure.json")
            self.assertEqual(len(meta), 1, p.stdout + p.stderr)
            with open(meta[0], encoding="utf-8") as fh:
                record = json.load(fh)
            raw = glob.glob(out + ".*.stdout.raw")[0]
            self.assertEqual(record["stdout_raw"]["sha256"], _sha(raw))
            self.assertNotIn("Get-FileHash", p.stdout + p.stderr)
            self.assertIsNone(pp.find_receipt(out), "no receipt for an unpublished report")
        with tempfile.TemporaryDirectory(prefix="run_mini_json_x_") as tmp:
            out = os.path.join(tmp, "report.json")
            p = _run_script(tmp, "valid_exit1", out, estimator=REAL_CANDIDATE,
                            env_kind="poisoned")
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("exited 1", p.stdout + p.stderr)
            with open(glob.glob(out + ".*.failure.json")[0], encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["report"]["sha256"], _sha(out))
            rec = pp.load_receipt(pp.find_receipt(out), out, label="candidate")
            self.assertEqual(rec["provenance"]["exit_code"], 1)

    def test_existing_receipt_is_refused_before_launch(self):
        with tempfile.TemporaryDirectory(prefix="run_mini_json_x_") as tmp:
            out = os.path.join(tmp, "report.json")
            with open(os.path.join(tmp, "report.receipt.json"), "w") as fh:
                fh.write("{}")
            p = _run_script(tmp, "valid", out, estimator=REAL_CANDIDATE)
            self.assertNotEqual(p.returncode, 0)
            self.assertFalse(os.path.exists(out))
            self.assertEqual(glob.glob(out + ".*.stdout.raw"), [])


class ReceiptDiscoveryCompat(unittest.TestCase):
    """Legacy <report>.receipt.json stays discoverable; conflicting pairs refuse."""

    def test_legacy_and_conflict(self):
        import predict_public as pp
        with tempfile.TemporaryDirectory(prefix="receipt_compat_") as tmp:
            rep = os.path.join(tmp, "r.json")
            open(rep, "w").close()
            self.assertIsNone(pp.find_receipt(rep))
            legacy = rep + ".receipt.json"
            with open(legacy, "w") as fh:
                fh.write('{"a": 1}')
            self.assertEqual(pp.find_receipt(rep), legacy)
            canonical = os.path.join(tmp, "r.receipt.json")
            with open(canonical, "w") as fh:
                fh.write('{"a": 1}')
            self.assertEqual(pp.find_receipt(rep), canonical)
            with open(canonical, "w") as fh:
                fh.write('{"a": 2}')
            with self.assertRaises(pp.ProjectionRefused):
                pp.find_receipt(rep)
            self.assertEqual(pp.find_receipt(rep, explicit=legacy), legacy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
