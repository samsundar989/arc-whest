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


def _run_script(tmp, mode, out):
    fake = os.path.join(tmp, "fake_whest.py")
    with open(fake, "w", encoding="utf-8") as fh:
        fh.write(FAKE)
    command = ("& %s -Estimator %s -Out %s -NoResidualLimit -LauncherExe %s "
               "-LauncherArgs @(%s, %s)" % (
                   _ps_quote(SCRIPT), _ps_quote("est with space.py"),
                   _ps_quote(out), _ps_quote(sys.executable),
                   _ps_quote(fake), _ps_quote(mode)))
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy",
         "Bypass", "-Command", command],
        capture_output=True, text=True, errors="replace", timeout=180)


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
            with open(out + ".receipt.json", "rb") as fh:
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
