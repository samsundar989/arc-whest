"""Per-layer class counts on the 100 FIT nets (exact labels): q50 [p10-p90] of dead / kink-active / always-on."""
import numpy as np
z = np.load("alpha_stats.npz"); taus = [float(t) for t in z["taus"]]; dead = z["dead"]; on = z["on"]
q = lambda a, p: np.percentile(a, p, axis=0)
for t in (2.0, 2.5, 3.0, 3.5, 4.0):
    i = taus.index(t); d = dead[i]; o = on[i]; k = 1024 - d - o
    print("tau %.1f   (fractions of 1024, %%; q50 [p10-p90])" % t)
    for nm, a in (("dead  a<-t", d), ("kink |a|<=t", k), ("on    a>+t", o), ("non-dead", 1024 - d)):
        print("  %-12s" % nm + " ".join("%5.1f[%4.1f-%4.1f]" % (np.median(a, 0)[l] / 10.24, q(a, 10)[l] / 10.24, q(a, 90)[l] / 10.24) for l in (2, 4, 6, 8, 10, 12, 14, 15)))
print("layers shown: 2 4 6 8 10 12 14 15")
k25 = 1024 - dead[taus.index(2.5)] - on[taus.index(2.5)]
print("CV of kink-active(2.5) count across nets by layer:", " ".join("%.1f%%" % (100 * k25[:, l].std() / k25[:, l].mean()) for l in range(16)))
