"""Emit knob variants of a candidate as files, and meter each on the real flopscope.

    python tools/variants.py <base.py> --arms arms.json --dir <outdir> [--cost]

Each arm `{"name": {"_CONST": value, ...}}` becomes `<outdir>/<stem>__<name>.py`,
the base with those single-line module constants rewritten (`_X = <repr>`).
With `--cost`, every file is metered by `tools/_cost_worker.py` (one net, f64
read-only grader twin -- the instrument of record for FLOPs and op calls) and the
table is written to `<outdir>/cost.json`.

The accuracy of the same arms is measured on the GPU by `tools/probe.py`, which
sets the same constants on one imported module; the two must agree on what an
arm is, so both read the same arms.json.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def emit(base: Path, name: str, over: dict, outdir: Path) -> Path:
    src = base.read_text(encoding="utf-8")
    for k, v in over.items():
        pat = re.compile(r"^%s\s*=.*$" % re.escape(k), re.M)
        if len(pat.findall(src)) != 1:
            raise SystemExit(f"{k}: expected exactly one single-line definition in {base.name}")
        src = pat.sub(lambda _m: f"{k} = {v!r}", src, count=1)
    dst = outdir / f"{base.stem}__{name}.py"
    dst.write_text(src, encoding="utf-8", newline="\n")
    return dst


def meter(path: Path) -> dict:
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "_cost_worker.py"), str(path), "dev", "0"],
                       cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if not line:
        return {"error": (r.stderr or r.stdout)[-800:]}
    return json.loads(line[-1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--cost", action="store_true")
    a = ap.parse_args(argv)
    base = (ROOT / a.base) if not Path(a.base).is_absolute() else Path(a.base)
    arms = json.loads(Path(a.arms).read_text() if Path(a.arms).exists() else a.arms)
    outdir = ROOT / a.dir
    outdir.mkdir(parents=True, exist_ok=True)
    table = {}
    cpath = outdir / "cost.json"
    if cpath.exists():
        table = json.loads(cpath.read_text(encoding="utf-8"))
    for name, over in arms.items():
        f = emit(base, name, over, outdir) if over else base
        print(f"[variants] {name}: {f.name}", flush=True)
        if a.cost and name not in table:
            c = meter(f)
            table[name] = {k: c.get(k) for k in ("flops", "multiplier", "ops", "materializations",
                                                  "wall_s", "peak_rss_mib", "error")}
            print(f"   F/B {c.get('multiplier')}  ops {c.get('ops')}  wall {c.get('wall_s')}"
                  f"  err {c.get('error')!r}", flush=True)
            cpath.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
