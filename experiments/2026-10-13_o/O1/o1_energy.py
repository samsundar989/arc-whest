"""O1 diagnostic: how much of the top-r energy of the join Gram G = J D J^T + Q Sg Q^T the shipped
one-pass basis captures, per join, plus G's spectrum at the candidate smaller ranks.

    <main>/.venv-gpu/Scripts/python.exe o1_energy.py <cand.py> <out.json> [nets=0,1]

Runs the candidate in the main tree's twin (`--raw`: constants baked in the file), FIT rows `full:<i>`.
Per join it records: layer, ko (old lineages already in the tier), n (joiner width), r, and
  cap_r      = tr(Qn^T G Qn) / sum(top-r eig G)   (1 = the basis is the exact top-r eigenspace)
  frac_k     = sum(top-k eig G) / tr G            for k in 128, 192, 224, 256
"""
import json
import sys
from pathlib import Path

MAIN = Path("C:/Users/samsu/Documents/arc-whest")
sys.path.append(str(MAIN / "tools/twin"))
import twin  # noqa: E402  (puts the torch shim first on the path)
import torch  # noqa: E402

cand, out = Path(sys.argv[1]), Path(sys.argv[2])
nets = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0,1").split(",")]
tw = twin.Twin(cand, raw=True)
mod = tw.mod
assert mod._SHARED_WEIGHTS == "cons" and not mod._QUERY_BASIS, "diagnostic assumes the v84 join branch"
orig = mod._shared_join_lean
recs = []


def wrap(t):
    n = mod._lin_widths(t)[t.ko]
    r = mod._shared_rank(n)
    joins = bool(r) and (t.k - t.ko) >= mod._SHARED_AGE and t._wts is not None
    o = orig(t)
    if not joins or o is t:
        return o
    J = torch.cat([t.A_L[:, :n], t.M[:, :n]], dim=1).double()
    dJ = torch.cat(list(t._wts)).double()
    G = (J * dJ[None, :]) @ J.T
    if t.ko:
        Q = t.Q.double()
        G = G + Q @ t.Sg.double() @ Q.T
    ev = torch.linalg.eigvalsh(G).flip(0).clamp_min(0)
    Qn = o.Q.double()
    cap = float(torch.trace(Qn.T @ G @ Qn) / ev[:r].sum())
    tot = float(ev.sum())
    recs.append({"net": NET, "layer": int(mod._CUR_LAYER), "ko": int(t.ko), "n": int(n), "r": int(r),
                 "cap_r": cap, **{f"frac_{k}": float(ev[:k].sum()) / tot for k in (128, 192, 224, 256)}})
    return o


mod._shared_join_lean = wrap
data = twin.Data()
with torch.no_grad():
    for i in nets:
        NET = i
        w, truth = data.net("full", i)
        mse = float(twin.mse_rows(tw.forward(w), truth)[-1])
        print(f"net {i}: final MSE {mse:.6e}", flush=True)
out.write_text(json.dumps(recs, indent=1))
print(f"{'net':>3} {'L':>2} {'ko':>2} {'n':>5} {'r':>3} {'cap_r':>7} {'f128':>6} {'f192':>6} {'f224':>6} {'f256':>6}")
for x in recs:
    print(f"{x['net']:3d} {x['layer']:2d} {x['ko']:2d} {x['n']:5d} {x['r']:3d} {x['cap_r']:7.4f} "
          f"{x['frac_128']:6.4f} {x['frac_192']:6.4f} {x['frac_224']:6.4f} {x['frac_256']:6.4f}")
