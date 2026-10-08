"""P3.0.2 — long-lived-worker memory growth (forum #18238).

Runs N sequential predicts of one candidate in ONE Linux process under the
8 GiB RLIMIT_AS cap, exactly as whestbench 0.16.1's subprocess_worker does
(fresh BudgetContext per predict, materialise in-window, summary_dict after),
and records per predict: VmPeak, VmHWM, settled VmRSS / VmSize, and the size of
flopscope's session-wide accumulator (records, op-log entries held).

    wsl.exe -d Ubuntu --exec ~/.arcwhest-linux/bin/python \
        experiments/2026-10-08_p3/mem_growth.py <candidate.py> <n> <out.json>

No flopscope.budget_reset() is called anywhere (rules' plain reading).
"""
import gc, json, os, resource, sys, time, warnings

CAND, N, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
LIMIT = 8192 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (LIMIT, LIMIT))
assert resource.getrlimit(resource.RLIMIT_AS)[0] == LIMIT

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.join(REPO, "local_sim")):   # APPEND, never insert
    if p not in sys.path:
        sys.path.append(p)

import flopscope as flops
import flopscope.numpy as fnp
from flopscope import _budget as _fb
from whestbench import MLP
assert "local_sim" not in flops.__file__
warnings.filterwarnings("ignore")
from mint_provenance import canonical_mint
from lifecycle_eval import build_weights, materialise_predictions

W, D, B = 1024, 16, 2 ** 41


def proc(key):
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024
    return 0


import importlib.util as iu
spec = iu.spec_from_file_location("cand", CAND)
mod = iu.module_from_spec(spec)
spec.loader.exec_module(mod)
est = mod.Estimator()


class Ctx:
    width, depth, flop_budget, seed, n_mlps = W, D, B, 12345, 200


est.setup(Ctx())
rows = []
t00 = time.time()
for i in range(N):
    w_np, s, _ = canonical_mint(W, D, 20261008 + 1000 * i)
    mlp = MLP(weights=build_weights(w_np, "f64_readonly", fnp), width=W, depth=D, seed=s)
    del w_np
    ctx = flops.BudgetContext(flop_budget=B, quiet=True)
    t0 = time.perf_counter()
    with ctx:
        pred = est.predict(mlp, B)
        arr = materialise_predictions(pred, depth=D, width=W)
        del pred
    _ = ctx.summary_dict(by_namespace=True)      # as _budget_payload does
    dt = time.perf_counter() - t0
    fl, nops = ctx.flops_used, len(ctx.op_log)
    del arr, mlp, ctx, _
    gc.collect()
    acc = _fb._accumulator
    held = sum(len(r.op_log) for r in acc._records)
    rows.append(dict(i=i, wall_s=round(dt, 2), flops=fl, ops=nops,
                     vm_peak=proc("VmPeak"), vm_hwm=proc("VmHWM"),
                     rss=proc("VmRSS"), vm_size=proc("VmSize"),
                     acc_records=len(acc._records), acc_ops_held=held))
    r = rows[-1]
    print(f"{i:3d} wall {dt:6.1f}s  rss {r['rss']/2**20:7.0f} MiB  size {r['vm_size']/2**20:7.0f}"
          f"  peak {r['vm_peak']/2**20:7.0f}  hwm {r['vm_hwm']/2**20:7.0f}  acc {len(acc._records)} recs {held} ops",
          flush=True)
    json.dump(dict(candidate=CAND, n=N, rows=rows, elapsed_s=time.time() - t00), open(OUT, "w"), indent=1)
