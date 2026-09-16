"""N1 label-free scaling read: energy-basis one-shot compression error of the old
tier, E(r), relative to deleting the tier, E0, against r/n at several widths.

Same definitions as `n1_relevance_oracle.py` (one-shot projection at layer l0 of
every lineage with age >= a; exact continuation; final-layer mean MSE). Decides
nothing: it says whether the rank FRACTION needed shrinks with width, which is
what would make r ~ 128-150 at n = 1024 plausible before any truth is spent.
Networks: canonical mint, seeds outside every registered block.
"""
import json, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]
import numpy as np
import n1_relevance_oracle as O

def main():
    import flopscope as flops
    from mint_provenance import canonical_mint
    flops.configure(symmetry_warnings=False)
    widths = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "24,40,64").split(",")]
    out = {}
    for n in widths:
        depth, l0, age = 12, 7, 3
        rows = []
        for net in range(3):
            w, _e, _ = canonical_mint(n, depth, 90_000_018_000 + 97 * n + net)
            Ws = [np.asarray(x, dtype=np.float64) for x in w]
            mod, st = O.load_lineage(os.path.join(_ROOT, "estimator_k3_v32_dense.py"))
            snap = {}
            def cap(l, s):
                if l == l0 and s[2] is not None:
                    snap["old"] = sum(T for b, T in s[2].items if l - b >= age)
            mu = O.run(mod, st, Ws, capture=cap)
            E0 = float(np.mean((O.run(mod, st, Ws, l0=l0, age=age, project=np.zeros((n, n))) - mu) ** 2))
            fr = {}
            for f in (1/16, 1/8, 3/16, 1/4, 3/8):
                r = max(1, int(round(f * n)))
                P = O.basis_energy(snap["old"], r)
                fr[f"{f:.4f}"] = float(np.mean((O.run(mod, st, Ws, l0=l0, age=age, project=P) - mu) ** 2)) / E0
            rows.append({"net": net, "E0": E0, "E_over_E0": fr,
                         "final_mean_sq": float(np.mean(mu ** 2))})
            print(n, net, f"E0 {E0:.2e} (vs mean^2 {np.mean(mu**2):.2e})",
                  " ".join(f"r/n={k}:{v:.3f}" for k, v in fr.items()), flush=True)
        out[n] = rows
    p = os.path.join(_ROOT, "experiments", "2026-09-16_stage3_n1", "rank_scaling.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

if __name__ == "__main__":
    main()
