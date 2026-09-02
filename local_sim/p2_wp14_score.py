"""Score CONTRACT WP14's C1 and C2 gates exactly as pre-registered.

Every ratio gate is written as a LINEAR CONTRAST so no bare ratio of means is
quoted, and every interval is the studentised-t on the paired difference --
percentile and BCa are refused (measured false-promotion rates 6.8-8.2% and
9.1-11.5% against a nominal 5%).
"""
import json, os, sys, collections
import numpy as np

ROOT = r"c:\Users\samsu\Documents\arc-whest"
sys.path.append(os.path.join(ROOT, "local_sim"))
from competition_score import paired_diff_ci

D = os.path.join(ROOT, "experiments", "2026-09-02_wp14_range")
c1 = json.load(open(os.path.join(D, "c1_containment.json")))
c2 = json.load(open(os.path.join(D, "c2_e2e.json")))

print("=" * 76)
print("C1 -- G-CONTAIN")
print("=" * 76)
rows = c1["rows"]
print(f"  {'layer':>5} {'r':>5} {'contain':>9} {'null_rand':>10} {'null_xnet':>10} "
      f"{'r/n':>6} {'ratio':>7} {'r90(E21)':>9}")
by = collections.defaultdict(list)
for r in rows:
    by[(r["layer"], r["r"])].append(r)
for (lay, rr) in sorted(by):
    g = by[(lay, rr)]
    m = lambda k: float(np.mean([x[k] for x in g]))
    c, nr = m("contain_r90"), m("null_rand_r90")
    nx = float(np.mean([x["null_xnet_r90"] for x in g if "null_xnet_r90" in x]))
    print(f"  {lay:>5} {rr:>5} {c:>9.4f} {nr:>10.4f} {nx:>10.4f} "
          f"{g[0]['null_expected_r_over_n']:>6.3f} {c/nr:>7.3f} {m('r90_E21'):>9.1f}")

print()
print("  G-CONTAIN at r = 512, scored per layer on the paired contrast")
print("  d_i = contain_i - max(threshold, 1.5 * null_i);  PASS iff t-lower > 0")
print(f"  {'layer':>5} {'thresh':>7} {'mean d':>11} {'t-lower':>11} {'t-upper':>11}  verdict")
verdicts = {}
for lay in sorted({r["layer"] for r in rows}):
    g = [r for r in rows if r["layer"] == lay and r["r"] == 512]
    if not g:
        continue
    thr = 0.60 if lay <= 4 else (0.80 if lay >= 10 else None)
    if thr is None:
        continue
    d = np.array([x["contain_r90"] - max(thr, 1.5 * x["null_rand_r90"]) for x in g])
    lo, hi = paired_diff_ci(d, method="t")
    ok = lo > 0
    verdicts[lay] = ok
    print(f"  {lay:>5} {thr:>7.2f} {d.mean():>11.4f} {lo:>11.4f} {hi:>11.4f}  "
          f"{'PASS' if ok else 'FAIL'}")
print(f"\n  G-CONTAIN (conjunctive over the gated layers): "
      f"{'PASS' if all(verdicts.values()) else 'FAIL'}")

# the null-equality question, stated as its own contrast
print("\n  Is containment DISTINGUISHABLE from the random-orthonormal null at all?")
print(f"  {'r':>5} {'mean(c - null)':>15} {'t-lower':>11} {'t-upper':>11}")
for rr in sorted({r["r"] for r in rows}):
    g = [r for r in rows if r["r"] == rr]
    d = np.array([x["contain_r90"] - x["null_rand_r90"] for x in g])
    lo, hi = paired_diff_ci(d, method="t")
    print(f"  {rr:>5} {d.mean():>15.5f} {lo:>11.5f} {hi:>11.5f}")

print()
print("=" * 76)
print("C2 -- G-E2E and G-POS")
print("=" * 76)
rows2 = c2["rows"]
cells = []
for r in rows2:
    if r["cell"] not in cells:
        cells.append(r["cell"])
seeds = sorted({r["seed"] for r in rows2})
drift = {(r["seed"], r["cell"]): r["drift_final_layer"] for r in rows2}
base = np.array([drift[(s, "none")] for s in seeds])
print(f"  uncorrected regenerating arm: mean drift {base.mean():.4e}  "
      f"(WP13 committed {c2['regen_drift_reference']:.4e})")
print()
print(f"  {'cell':<14}{'mean drift':>13}{'x better':>10}{'t-lower(x)':>12}")
for cell in cells:
    v = np.array([drift[(s, cell)] for s in seeds])
    if cell == "none":
        print(f"  {cell:<14}{v.mean():>13.4e}{1.0:>10.3f}{'-':>12}")
        continue
    # linear contrast for "at least k-fold": d_i = base_i - k * v_i
    lo_k = None
    for k in np.arange(1.0, 400.0, 0.01):
        d = base - k * v
        lo, _ = paired_diff_ci(d, method="t")
        if lo <= 0:
            lo_k = k - 0.01
            break
    print(f"  {cell:<14}{v.mean():>13.4e}{base.mean()/v.mean():>10.3f}"
          f"{(lo_k if lo_k else float('nan')):>12.2f}")

print()
d = base - 5.0 * np.array([drift[(s, "proj:512")] for s in seeds])
lo, hi = paired_diff_ci(d, method="t")
print(f"  G-E2E  d_i = drift_none - 5 * drift_proj512")
print(f"         mean {d.mean():.4e}   t 95% [{lo:.4e}, {hi:.4e}]   "
      f"{'PASS' if lo > 0 else 'FAIL'}")
fal = base - 2.0 * np.array([drift[(s, "proj:512")] for s in seeds])
lo2, _ = paired_diff_ci(fal, method="t")
print(f"  falsifier (< 2x): {'CLEARED' if lo2 > 0 else 'FIRED'}")

ex = np.array([drift[(s, "exact")] for s in seeds])
print(f"\n  G-POS  exact-injection drift mean {ex.mean():.4e}, max {ex.max():.4e}")
print(f"         relative to the uncorrected arm: {ex.mean()/base.mean():.3e}   "
      f"{'PASS' if ex.mean()/base.mean() <= 1e-12 else 'see note'}")
z = np.array([drift[(s, "proj:0")] for s in seeds])
print(f"  determinism  proj:0 == none exactly on all seeds: "
      f"{bool(np.array_equal(z, base))}")
sp = [r["slice_probe"] for r in rows2 if r["cell"] == "exact" and r["slice_probe"]]
if sp:
    print(f"  S-SLICE in flight (exact cell, first layer): {sp[0][0]}")
