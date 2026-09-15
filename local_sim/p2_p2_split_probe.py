"""Is the 7->4 split real? Measure it before writing a line of estimator code.

Slice 6b, `docs/CLAUDE_CODE_HANDOFF_2026-09-14.md` §3, mechanism **P2**. This is
the feasibility screen that has to pass before P2 is worth building, and it is
label-free: no truth, no public split, no DECIDE spend.

WHAT P2 CLAIMS
--------------
F21 births its dense leg as

    A_D = diag(S3) + 3 * S21.T          (`estimator_k3_strassen_L4_f21.py:944`)

P2 says `S21` splits into a **separable** part and a small residual:

    S21 = S_sep + R_res
    S_sep[i,j] = e_i * C_off[i,j] * w1_j,     e = w(1,2) - 2 * pK1 * w(1,1)

The separable part needs no new operand: since `C_off` is symmetric and F21
already carries `a_b = fac1 = w1[:,None] * C11`,

    (3 * S_sep).T[i,j] = 3 * e_j * C_off[j,i] * w1_i = 3 * fac1[i,j] * e_j

so `3 * S_sep.T` is `3 * a_b * d(e)` — **a column scale on a leg that already
exists**. That is the whole saving: if `R_res` is small or low rank, the dense
`A_D` transport and the `(M.M) @ A_D.T` contraction can go.

WHAT THIS MODULE MEASURES, AND WHAT WOULD KILL P2
--------------------------------------------------
Per layer, on real chain state:

  * `||S_sep|| / ||S21||` — how much of the object the free part already is;
  * the singular spectrum of `R_res`, and the relative error of truncating it
    to rank 16, which is the rank the plan proposes;
  * the same for the quantity that actually matters, `3 * S21.T`, since that is
    what enters the tower.

**P2 dies here** if `R_res` needs a rank comparable to `n` to be represented —
then the "thin transported leg" is not thin and the saving is fictional. It
also dies if `S_sep` captures so little that the split buys nothing.

This screen says nothing about accuracy. A rank that represents `R_res` well in
Frobenius norm can still change the score, because what the chain consumes is a
contraction, not a matrix. That is a later, separate measurement.

Run:
    uv run python local_sim/p2_p2_split_probe.py --width 192 --depth 8 --nets 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

SOURCE = os.path.join(_ROOT, "estimator_k3_strassen_L4_f21.py")

#: Capture the layer internals P2 needs, immediately before the tower birth.
#: Appended rather than edited: the arithmetic above the hook is untouched, so
#: the state we measure is the state the estimator actually produces.
_HOOK = '''    _P2_CAPTURE.append({
        "layer": int(layer_index_probe[0]),
        "S21": np.asarray(S21, dtype=np.float64),
        "S3": np.asarray(S3, dtype=np.float64),
        "C11": np.asarray(C11, dtype=np.float64),
        "w11": np.asarray(w[(1, 1)], dtype=np.float64),
        "w12": np.asarray(w[(1, 2)], dtype=np.float64),
        "pK1": np.asarray(pK[(1,)], dtype=np.float64),
        "fac1": np.asarray(fac1, dtype=np.float64),
    })
    layer_index_probe[0] += 1
'''


def _instrumented():
    """F21 with a capture hook, written to a scratch module. Never committed."""
    with open(SOURCE, "rb") as fh:
        raw = fh.read()
    src = raw.decode("utf-8")
    anchor = "    out_tw = _append(adv, fnp.diag(S3) + 3.0 * S21.T, fac1, w2b_vec, n)"
    if src.count(anchor) != 1:
        raise SystemExit("F21 has moved: the birth anchor is not unique")
    src = src.replace(anchor, _HOOK + anchor, 1)
    # numpy is not imported by the estimator (it may not import it), so the
    # probe supplies it. This module is a LAB INSTRUMENT and never shipped.
    src = src.replace("import flopscope as flops",
                      "import numpy as np\nimport flopscope as flops", 1)
    src = src.replace("_TABLES = None",
                      "_P2_CAPTURE = []\nlayer_index_probe = [0]\n_TABLES = None", 1)
    return src, hashlib.sha256(raw).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--nets", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=40901003000,
                    help="P03, the block already claimed for Slice 6a")
    ap.add_argument("--rank", type=int, default=16,
                    help="the rank the plan proposes for R_res")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    import importlib.util
    import tempfile

    import flopscope as flops
    flops.configure(symmetry_warnings=False)
    import harness
    from mint_provenance import canonical_mint

    src, sha = _instrumented()
    tmpd = tempfile.mkdtemp(prefix="p2split_")
    path = os.path.join(tmpd, "f21_probe.py")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("f21_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    print("=" * 96)
    print("P2 FEASIBILITY: does S21 split into a free separable part plus a thin residual?")
    print("=" * 96)
    print(f"  F21 sha256 {sha[:16]}...   {a.width}x{a.depth}   {a.nets} nets   "
          f"proposed rank {a.rank}")
    print("  LABEL-FREE: no truth, no public split, no DECIDE spend.")
    print()

    est = mod.Estimator()

    class _Ctx:
        width, depth, flop_budget, seed, n_mlps = a.width, a.depth, 2 ** 41, 0, 1
    est.setup(_Ctx())

    per_layer = {}
    for k in range(a.nets):
        mod._P2_CAPTURE.clear()
        mod.layer_index_probe[0] = 0
        w, es, _ = canonical_mint(a.width, a.depth, a.seed_start + k)
        mlp = harness.MLP(weights=[np.asarray(x) for x in w], width=a.width,
                          depth=a.depth, seed=es)
        with flops.BudgetContext(flop_budget=2 ** 41, quiet=True):
            est.predict(mlp, 2 ** 41)
        for rec in mod._P2_CAPTURE:
            l = rec["layer"]
            S21, C11 = rec["S21"], rec["C11"]
            w11, w12, pK1 = rec["w11"], rec["w12"], rec["pK1"]
            e = w12 - 2.0 * pK1 * w11
            S_sep = e[:, None] * C11 * w11[None, :]
            R = S21 - S_sep

            nS = float(np.linalg.norm(S21))
            nSep = float(np.linalg.norm(S_sep))
            nR = float(np.linalg.norm(R))
            sv = np.linalg.svd(R, compute_uv=False)
            r = min(a.rank, sv.size)
            tail = float(np.sqrt(max((sv[r:] ** 2).sum(), 0.0)))
            # rank needed to hold 99% of the residual's energy
            cum = np.cumsum(sv ** 2)
            need99 = int(np.searchsorted(cum, 0.99 * cum[-1]) + 1) if cum[-1] > 0 else 0

            d = per_layer.setdefault(l, {"sep_frac": [], "res_frac": [],
                                         "rank16_err": [], "need99": []})
            d["sep_frac"].append(nSep / nS if nS else np.nan)
            d["res_frac"].append(nR / nS if nS else np.nan)
            d["rank16_err"].append(tail / nS if nS else np.nan)
            d["need99"].append(need99)

    print(f"  {'layer':>5} {'||S_sep||/||S21||':>18} {'||R||/||S21||':>15} "
          f"{'rank-%d resid err' % a.rank:>18} {'rank for 99pct':>14}")
    rows = []
    for l in sorted(per_layer):
        d = per_layer[l]
        row = (l, float(np.mean(d["sep_frac"])), float(np.mean(d["res_frac"])),
               float(np.mean(d["rank16_err"])), float(np.mean(d["need99"])))
        rows.append(row)
        print(f"  {row[0]:>5} {row[1]:>18.4f} {row[2]:>15.4f} "
              f"{row[3]:>18.4e} {row[4]:>14.1f}")

    print()
    if rows:
        worst_err = max(r[3] for r in rows)
        worst_rank = max(r[4] for r in rows)
        print(f"  worst rank-{a.rank} residual error over layers: {worst_err:.4e}")
        print(f"  worst rank needed for 99% of the residual: {worst_rank:.0f} "
              f"of n={a.width}")
        print()
        # THE CRITERION THIS PRINTED FIRST WAS WRONG, and it is worth saying
        # why. It read "rank for 99% of the residual > n/4 => P2 is in
        # trouble", which pronounced on layer 0 -- where the residual is 2.7%
        # of S21 and its 99%-rank is therefore the rank of numerical noise,
        # not of structure. A rank that is large in proportion to a residual
        # that is nearly zero costs nothing. The bar the PLAN proposed is
        # rank 16 at +1.6% raw, so that is what gets judged.
        print("  READ THESE TWO TOGETHER, not separately:")
        print(f"    the separable part carries ~{100 * np.mean([r[1] for r in rows]):.0f}% "
              "of ||S21|| at every layer, and it is FREE")
        print(f"    rank-{a.rank} on the remainder leaves {worst_err:.2%} of "
              "||S21|| unrepresented, worst layer")
        print()
        print("  The 99%-rank column is what FULL fidelity of the residual")
        print("  would cost, and it is only meaningful where the residual is")
        print("  itself large: at layer 0 the residual is "
              f"{100 * rows[0][2]:.1f}% of ||S21||, so its rank is the rank of")
        print("  noise. Weight it by the residual fraction before quoting it.")
        print()
        print("  THIS SCREEN CANNOT SAY WHETHER THE SCORE SURVIVES. What the")
        print("  chain consumes is a contraction, not a matrix, and a few")
        print("  percent of Frobenius norm is not a few percent of MSE. The")
        print("  plan's +1.6% raw is a PREDICTION to be tested end to end.")
        print()
        print("  SCOPE: measured at n=%d. The rank a residual needs can grow"
              % a.width)
        print("  with width, and this says nothing about 1024 on its own.")

    if a.out:
        dd = os.path.dirname(os.path.abspath(a.out))
        if dd:
            os.makedirs(dd, exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(
                {"f21_sha256": sha, "width": a.width, "depth": a.depth,
                 "nets": a.nets, "rank": a.rank, "label_free": True,
                 "rows": [{"layer": r[0], "sep_frac": r[1], "res_frac": r[2],
                           "rank_err": r[3], "rank_for_99pct": r[4]}
                          for r in rows]}, indent=2) + "\n")
        print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
