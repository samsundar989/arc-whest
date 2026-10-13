"""Per-net, per-layer alpha class counts from exact FIT labels (cache/twin_labels/fit_XXX.npz).
alpha = mu/sqrt(var) of the pre-activation. Writes alpha_stats.npz: dead[tau][net,layer], on[tau][net,layer]."""
import numpy as np, glob, os, json
R = os.path.expanduser("~/mnt/arc-whest/cache/twin_labels")
files = sorted(glob.glob(R + "/fit_*.npz"))
TAUS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
dead = np.zeros((len(TAUS), len(files), 16), int); on = np.zeros_like(dead)
alphas = np.zeros((len(files), 16, 1024), np.float32)
for i, f in enumerate(files):
    z = np.load(f)
    a = z["mu"] / np.sqrt(np.maximum(z["var"], 1e-12))
    alphas[i] = a
    for t, tau in enumerate(TAUS):
        dead[t, i] = (a < -tau).sum(1); on[t, i] = (a > tau).sum(1)
np.savez("alpha_stats.npz", taus=np.array(TAUS), dead=dead, on=on, alphas=alphas)
print(len(files), "nets")
for t, tau in enumerate(TAUS):
    keep = 1024 - dead[t]; act = 1024 - dead[t] - on[t]
    print("tau", tau)
    print("  keep q50", np.median(keep, 0).tolist())
    print("  act  q50", np.median(act, 0).tolist())
    print("  keep q10", np.percentile(keep, 10, 0).round(0).tolist())
    print("  keep q90", np.percentile(keep, 90, 0).round(0).tolist())
    print("  act  q10", np.percentile(act, 10, 0).round(0).tolist())
    print("  act  q90", np.percentile(act, 90, 0).round(0).tolist())
