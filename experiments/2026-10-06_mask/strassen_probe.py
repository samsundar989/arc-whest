"""P2: does a compact inner dimension K < 1024 lose a Strassen level, and does a lower
`_STRASSEN_MIN_BLOCK` get it back? Real flopscope meter, v73's own `_mm`.

    uv run python experiments/2026-10-06_mask/strassen_probe.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "local_sim"))

import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402

spec = importlib.util.spec_from_file_location("v73", ROOT / "estimator_k3_v73_gb.py")
v73 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v73)

rng = fnp.random.default_rng(0)
n, cols = 1024, 5 * 1024
rows = []
for K in (1024, 992, 928, 864, 800, 736):
    A = rng.standard_normal((n, K))
    B = rng.standard_normal((K, cols))
    for mb in (32, 24, 20, 16):
        v73._STRASSEN_MIN_BLOCK = mb
        with flops.BudgetContext(flop_budget=2 ** 45, quiet=True) as b:
            v73._mm(A, B)
        d = v73._depth_for(n, K, cols)
        rows.append({"K": K, "min_block": mb, "depth": d, "flops": int(b.flops_used),
                     "per_K": b.flops_used / K})
        print(json.dumps(rows[-1]), flush=True)
(Path(__file__).parent / "strassen_probe.json").write_text(json.dumps(rows, indent=1))
