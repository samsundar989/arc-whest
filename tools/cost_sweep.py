"""Per-NET cost on the real f64 grader twin, for candidates whose FLOPs vary by net.

    python tools/cost_sweep.py <dir-of-variants>|<file.py> [<file2.py> ...] \
        --panel dev --indices 0,1,2,3,4 --out experiments/<dir>/cost_sweep.json

WHY THIS EXISTS
---------------
`tools/_cost_worker.py` meters ONE net, and every candidate this project has
shipped made that sufficient: "FLOPs are net-independent: every graded run we hold
reports spread 0 across all 100 nets". The adaptive-age candidate
(`estimator_k3_v60_adage.py`) deliberately breaks that property -- its fold
schedule depends on the lineage energies, so its FLOPs are a function of the net,
which is the per-MLP cost variation the leaders' telemetry shows and ours never
had. A one-net reading of such a candidate is not its cost; it is one sample of
its cost, and the SCORE depends on the mean while the BUDGET depends on the max.

So this loops the real meter over several nets per variant and reports
min/mean/max FLOPs and ops, which is what HANDOFF 09-24 Step 3 asks for.

It calls `tools/_cost_worker.py` as a subprocess exactly as `tools/variants.py`
does, so the instrument of record is unchanged: real flopscope 0.12.1, f64
read-only grader-twin weights, `local_sim` APPENDED so the shims cannot win.
Never take these numbers from the counting shim -- it reads ~18% low.

Each net is a full twin `predict` (~70-95 s), so cost scales as
variants x nets x 90 s. Pick `--indices` deliberately.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def _rel(p) -> str:
    """`p` relative to the repo root for LOGGING, never raising.

    `Path.relative_to` raises when the path is not under `ROOT`, and under
    worker-q `ROOT` is the job's snapshot directory while an output path is
    deliberately absolute and outside it (`.gpuq.toml`: "Always give heavy jobs
    an ABSOLUTE output path"). That combination failed jobs 2090 and 2091 on a
    print statement AFTER all their work was written -- a cosmetic line that
    cost two exit-code-1 job reports.
    """
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)



def meter(path: Path, panel: str, idx: int) -> dict:
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "_cost_worker.py"), str(path),
         panel, str(idx)],
        cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if not line:
        return {"error": (r.stderr or r.stdout)[-600:]}
    return json.loads(line[-1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("targets", nargs="+",
                    help="variant .py files, or a directory of them")
    ap.add_argument("--panel", default="dev")
    ap.add_argument("--indices", default="0,1,2,3,4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    files: list[Path] = []
    for t in a.targets:
        p = Path(t)
        if not p.is_absolute():
            p = ROOT / p
        if p.is_dir():
            files.extend(sorted(x for x in p.glob("*.py")
                                if not x.name.startswith("_")))
        else:
            files.append(p)
    idxs = [int(x) for x in a.indices.split(",")]
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    res: dict = {"panel": a.panel, "indices": idxs, "variants": {}}
    if out.exists():
        res = json.loads(out.read_text(encoding="utf-8"))
        res.setdefault("variants", {})

    BUDGET = 2 ** 41
    for f in files:
        key = f.stem
        res["variants"].setdefault(key, {})
        for i in idxs:
            if str(i) in res["variants"][key]:
                continue
            d = meter(f, a.panel, i)
            res["variants"][key][str(i)] = d
            out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
            if d.get("error"):
                print(f"[cost] {key} net {i}: ERROR {d['error'][:200]}", flush=True)
            else:
                print(f"[cost] {key} net {i}: {d['flops']:,.0f} FLOPs "
                      f"({d['multiplier']:.5f})  {d['ops']:.0f} ops", flush=True)

    print(f"\n{'variant':44s} {'n':>2} {'FLOPs min':>16} {'mean':>16} "
          f"{'max':>16} {'spread':>7} {'ops mean':>9}")
    for key, per in res["variants"].items():
        vals = [v for v in per.values() if not v.get("error")]
        if not vals:
            continue
        fl = sorted(v["flops"] for v in vals)
        ops = [v["ops"] for v in vals]
        mean = sum(fl) / len(fl)
        print(f"{key:44s} {len(fl):>2} {fl[0]:>16,.0f} {mean:>16,.0f} "
              f"{fl[-1]:>16,.0f} {fl[-1] / fl[0] - 1:>6.2%} "
              f"{sum(ops) / len(ops):>9.0f}")
        per["_summary"] = {"n": len(fl), "flops_min": fl[0], "flops_mean": mean,
                           "flops_max": fl[-1], "spread": fl[-1] / fl[0] - 1,
                           "mult_mean": mean / BUDGET,
                           "ops_mean": sum(ops) / len(ops),
                           "ops_max": max(ops)}
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"\n[cost] -> {_rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
