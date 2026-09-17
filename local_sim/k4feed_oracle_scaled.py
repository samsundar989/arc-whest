"""Skeptic follow-up (A): the kappa4 all-distinct route check at NON-UNIT variance.

Sigma = s (I + eps C). The scale mixture's kappa4 is then d^2 pairings(Sigma), which
equals the estimator's model m Sym(G (x) I) with g = 1.5 d^2 s^2 and lam = 3 d^2 s
(lam multiplies C_pre_off = s eps C). Checks route (i), (ii), (i)+(ii) against exact
for s in {0.15, 0.5, 2.0} and two mean vectors. Label-free.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(_HERE)
sys.path[:] = [q for q in sys.path if os.path.abspath(q or ".") != _HERE] + [R, _HERE]
import numpy as np
from scipy.special import ndtr
import flopscope as flops
flops.configure(symmetry_warnings=False)
import k4feed_oracle as K

mod = K._load(os.path.join(R, "estimator_k3_v31_dense.py"))
rng = np.random.default_rng(20260917)
n = 3
C = rng.normal(size=(n, n)); C = (C + C.T) * 0.5; np.fill_diagonal(C, 0.0)
m = 2.0
rows = []
for mu in (np.array([0.35, -0.2, 0.1]), np.array([-0.6, 0.4, 0.05])):
    for s in (0.15, 0.5, 2.0):
        mu_s = mu * np.sqrt(s)                       # keep alpha = mu/sigma comparable
        sig = np.sqrt(s)
        al = mu_s / sig
        w1, w2, w3 = ndtr(al), K._phi(al) / sig, -al * K._phi(al) / s

        def exact(d, e):
            return K._mixture_cumulants(mu_s, s * (np.eye(n) + e * C), d)["k3_012"]

        def est(d, e, routes):
            d2 = d * d
            out = K.run_layer(mod, np.eye(n), mu_s, s * (np.eye(n) + e * C),
                              np.full(n, 1.5 * d2 * s * s), 3.0 * d2 * s)
            T = K.tower_tensor(out[2], n)
            Cp = s * e * C
            if "i" in routes:
                T = T + K._compensated(K.feed_tensor(3.0 * d2 * s, m, w1, w2, Cp))
            if "ii" in routes:
                dG = np.full(n, 1.5 * d2 * s * s)      # W=I: dG = gk4
                D22 = (m / 6.0) * (dG[:, None] + dG[None, :]) * (1.0 - np.eye(n))
                T = T + K._compensated(K.link_tensor(D22, w1, w2, w3, Cp))
            return T[0, 1, 2]

        def odd(fn, d=0.02, e=0.01):
            return ((fn(d, e) - fn(0.0, e)) - (fn(d, -e) - fn(0.0, -e))) / (2 * d * d * e)
        ex = odd(exact)
        r = {k: odd(lambda a, b, k=k: est(a, b, k)) for k in ((), ("i",), ("i", "ii"))}
        rel = lambda x: abs(x - ex) / abs(ex)
        print(f"mu={mu.tolist()} s={s:<5} exact {ex:+.5e}  v31 {r[()]:+.5e}  "
              f"(i) {r[('i',)]:+.5e} [{rel(r[('i',)]):.2e}]  (i)+(ii) {r[('i','ii')]:+.5e} [{rel(r[('i','ii')]):.2e}]",
              flush=True)
        rows.append((s, rel(r[('i',)]), rel(r[('i', 'ii')])))
ok = all(b < 0.05 for _, _, b in rows)
print("VERDICT (i)+(ii) within 5% of exact at every variance:", ok)
