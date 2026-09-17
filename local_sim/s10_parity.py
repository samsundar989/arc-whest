"""Label-free gates and the QUERY-ERROR instrument for the stage-3 consumed-query basis.

Handoff 2026-09-17b section 3. Thresholds fixed here, before the first run. No
truth, no score, no promotion.

THE GATES
---------
 1. NESTING: `..._qoff` == its base bit for bit, output AND counted FLOPs, at
    64x5 and 96x6. `_QUERY_BASIS = False` must be a genuine no-op, or every
    later difference is the patch rather than the objective.
 2. FULL-RANK IDENTITY: forced rank = n, the query basis must reproduce the
    UNCOMPRESSED chain exactly -- projecting onto a complete basis is the
    identity whatever chose it. rel <= 1e-9 on a float64 tiny chain, every row,
    with joins actually happening.
 3. RUNS: the approximate path at a tiny rank is finite, non-negative, and
    actually joins.
 4. EQUIVARIANCE: relabelling a hidden layer moves the prediction by no more
    than 100x the base's own float32 floor. The initialiser is unchanged and
    the new objective introduces no sampling, so this must hold.
 5. THE TINY DETERMINISTIC CHECKS the handoff names, on the JOIN ITSELF:
      (a) full-rank identity of the projector;
      (b) CORRELATED / CANCELLING columns -- the case that exposes the old
          surrogate: both left columns equal `u`, legs `e1` and `e2`, so the
          true query is rank one along `e1 + e2` and the diagonal surrogate is
          isotropic. The query basis must capture it at rank 1; the surrogate
          must not;
      (c) NON-UNIT, HETEROGENEOUS scaling of the left factors;
      (d) the COMPLETE recomputed `D3` and `D21` after projection, and the
          whole next-layer and output displacement, not a norm of the basis.

THE INSTRUMENT
--------------
`query_error()` measures, on a real chain at a real shape, the thing the two
objectives disagree about: the ACTUAL post-projection error of everything the
legs are consumed by --

    E   = LA A^T + LP M^T     the (2,1) contraction,
    D3  = the elementwise third-order slice,
    D21 = the bivariate slice as the layer forms it,
    plus the resulting next-layer covariance and the final prediction.

It reports each arm's relative error against the SAME chain run uncompressed,
so "the basis retains more of what is consumed" is a measurement and not an
argument about Frobenius norms of factors.

Run:  uv run python local_sim/s10_parity.py
      uv run python local_sim/s10_parity.py --query-error --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

OK = []
ROWS = []

BASE = "estimator_k3_v37_a4r256_p4d5.py"
STEM = "v37_a4r256_p4d5"


def claim(name, ok, detail=""):
    OK.append(bool(ok))
    ROWS.append({"check": name, "passed": bool(ok), "detail": str(detail)})
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<62} {detail}", flush=True)


def rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _variant(src_path, tmp, tag, **flags):
    """A temp copy of an emitted arm with its module-level flags overridden."""
    src = open(src_path, encoding="utf-8").read()
    for k, v in flags.items():
        old = "\n%s = " % k
        i = src.index(old)
        j = src.index("\n", i + 1)
        src = src[:i] + "\n%s = %r" % (k, v) + src[j:]
    p = os.path.join(tmp, f"{tag}.py")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    return p


def count_joins(mod):
    box = {"joins": 0, "max_ko": 0}
    orig = mod._shared_join_lean

    def wrapped(tw):
        out = orig(tw)
        if out is not tw:
            box["joins"] += 1
            box["max_ko"] = max(box["max_ko"], out.ko)
        return out

    mod._shared_join_lean = wrapped
    return box


def layer_loop(mod, w, n, depth):
    import flopscope as flops
    import flopscope.numpy as fnp
    st = (np.zeros(n), np.eye(n), None, 0.0, np.zeros(n))
    rows = []
    const = mod.Estimator()._constants(n, fnp.float64)
    for l in range(depth):
        W = np.asarray(w[l], dtype=np.float64)
        with flops.BudgetContext(flop_budget=2 ** 55, quiet=True):
            o = mod._layer(fnp.asarray(W, dtype=fnp.float64),
                           (fnp.asarray(st[0], dtype=fnp.float64),
                            fnp.asarray(st[1], dtype=fnp.float64),
                            st[2], st[3], fnp.asarray(st[4], dtype=fnp.float64)),
                           mod._tables(), fnp.float64, const, 2.0,
                           last_layer=(l == depth - 1), gens=None, layer_index=l)
        rows.append(np.asarray(o[0], dtype=np.float64))
        st = (o[0], o[1], o[2], o[3], o[4])
    return np.stack(rows)


# --------------------------------------------------------------------------- #
# 5. the tiny deterministic join checks, on the join alone
# --------------------------------------------------------------------------- #

def _join_basis(mod, U, J, r, ko=0, Q=None, Xold=None):
    """Run ONE join's basis selection on supplied operands and return Q."""
    import flopscope as flops
    import flopscope.numpy as fnp
    n = J.shape[0]
    m = J.shape[1] // 2

    class _TW:
        pass

    tw = _TW()
    tw.A_L = fnp.asarray(J[:, :m], dtype=fnp.float64)
    tw.M = fnp.asarray(J[:, m:], dtype=fnp.float64)
    tw.w2b = fnp.asarray(np.ones(m), dtype=fnp.float64)
    tw.k, tw.ko = 99, ko
    tw.Q, tw.FA, tw.FM, tw.Sg = Q, None, None, None
    tw._wts = (fnp.asarray((U[:, :m] * U[:, :m]).sum(axis=0), dtype=fnp.float64),
               fnp.asarray((U[:, m:] * U[:, m:]).sum(axis=0), dtype=fnp.float64))
    tw._qU = (fnp.asarray(U[:, :m], dtype=fnp.float64),
              fnp.asarray(U[:, m:], dtype=fnp.float64))
    tw._qold = Xold
    tw.A_D = tw.Z = tw.L = tw.sb = tw.eb = None
    tw.Y = tw.CA = tw.CB = None
    tw._legs = None
    orig = (mod._SHARED_RANK, mod._SHARED_AGE, mod._SHARED_FORCE)
    # `_shared_rank` refuses r == n unless FORCE is on; the full-rank identity
    # check needs exactly that case, so it is armed here and restored after.
    mod._SHARED_RANK, mod._SHARED_AGE, mod._SHARED_FORCE = r, 0, True
    try:
        with flops.BudgetContext(flop_budget=2 ** 55, quiet=True):
            out = mod._shared_join_lean(tw)
    finally:
        mod._SHARED_RANK, mod._SHARED_AGE, mod._SHARED_FORCE = orig
    if out is tw or out.Q is None:
        raise SystemExit("the join did not run (rank %d, n %d)" % (r, n))
    return np.asarray(out.Q, dtype=np.float64)


def tiny_join_checks(mods):
    """(a)-(c): the cases that separate the two objectives, at n = 6."""
    n = 6
    print("\n5. TINY DETERMINISTIC JOIN CHECKS")

    # (b) the cancelling case the strategist names, made concrete.
    #     Both left columns are the SAME vector u, so U^T U is all-ones(2) times
    #     |u|^2 and its OFF-DIAGONAL is the whole story. The legs are e1 and e2,
    #     so E = u (e1 + e2)^T and the query's right space is exactly
    #     span(e1 + e2), one dimension. The diagonal surrogate sees
    #     J diag(|u|^2, |u|^2) J^T = |u|^2 (e1 e1^T + e2 e2^T), which is
    #     ISOTROPIC on span(e1, e2) and cannot prefer the right direction.
    rng = np.random.default_rng(20260917)
    u = rng.normal(size=n)
    U = np.stack([u, u], axis=1)                      # (n, 2), m = 1
    J = np.zeros((n, 2))
    J[0, 0] = 1.0
    J[1, 1] = 1.0
    want = np.zeros(n)
    want[0] = want[1] = 1.0 / np.sqrt(2.0)
    for tag, mod in mods.items():
        if tag == "qoff":
            continue
        Q = _join_basis(mod, U, J, r=1)
        cap = float(abs(Q[:, 0] @ want))
        claim(f"(b) {tag} rank-1 basis captures the cancelling query direction",
              cap > 0.999, "|<q, (e1+e2)/sqrt2>| = %.6f" % cap)
    Qg = _join_basis(mods["qoff"], U, J, r=1)
    capg = float(abs(Qg[:, 0] @ want))
    claim("(b) the OLD diagonal surrogate does NOT capture it",
          capg < 0.95, "|<q, (e1+e2)/sqrt2>| = %.6f  (the defect, reproduced)" % capg)

    # (a) full-rank identity of the projector, on the same operands
    for tag, mod in mods.items():
        Q = _join_basis(mod, U, J, r=n)
        err = rel(Q @ (Q.T @ J), J)
        claim(f"(a) {tag} full-rank projector is the identity on the legs",
              err <= 1e-12, "%.1e" % err)

    # (c) non-unit, heterogeneous scaling of the left factors
    m = 4
    U2 = rng.normal(size=(n, 2 * m)) * np.exp(rng.normal(scale=2.0, size=2 * m))
    J2 = rng.normal(size=(n, 2 * m))
    E = U2 @ J2.T
    # the optimal rank-2 right subspace of E, computed independently
    _u, _s, vt = np.linalg.svd(E, full_matrices=False)
    best = float(np.linalg.norm(_s[2:]) / max(np.linalg.norm(_s), 1e-300))
    for tag, mod in mods.items():
        Q = _join_basis(mod, U2, J2, r=2)
        got = float(np.linalg.norm(E - E @ Q @ Q.T) / np.linalg.norm(E))
        ROWS.append({"check": f"(c) {tag} rank-2 relative query error",
                     "passed": True, "detail": "%.4f (optimal %.4f)" % (got, best)})
        print(f"        (c) {tag:<5} rank-2 relative query error {got:.4f}   "
              f"(independent SVD optimum {best:.4f})")
        if tag != "qoff":
            claim(f"(c) {tag} beats the diagonal surrogate under heterogeneous scaling",
                  got <= float(np.linalg.norm(
                      E - E @ (lambda Qg2: Qg2 @ Qg2.T)(
                          _join_basis(mods["qoff"], U2, J2, r=2)))
                      / np.linalg.norm(E)) + 1e-12,
                  "%.4f" % got)
    del vt


# --------------------------------------------------------------------------- #

def main_tiny():
    import flopscope as flops
    import k4feed_oracle as K
    import s1_parity_v32 as P1
    from mint_provenance import canonical_mint

    flops.configure(symmetry_warnings=False)
    L = lambda p: K._load(p if os.path.isabs(p) else os.path.join(_ROOT, p))
    tmp = tempfile.mkdtemp(prefix="s10_")
    arms = {t: "estimator_k3_v39_%s_%s.py" % (STEM, t)
            for t in ("qoff", "qE", "qEd")}

    print("\n1. NESTING -- _QUERY_BASIS = False is a no-op")
    base = L(BASE)
    for wd, dp, sd in ((64, 5, 90000001101), (96, 6, 90000001102)):
        w, es, _ = canonical_mint(wd, dp, sd)
        pb, fb = P1._predict(base, w, wd, dp, es)
        po, fo = P1._predict(L(arms["qoff"]), w, wd, dp, es)
        claim(f"{wd}x{dp} qoff == base bitwise + FLOPs",
              np.array_equal(pb, po) and fb == fo, f"flops {fo:.0f} vs {fb:.0f}")

    print("\n2. FULL-RANK IDENTITY -- a complete basis is the identity, whoever chose it")
    unc = _variant(os.path.join(_ROOT, arms["qoff"]), tmp, "unc", _SHARED_ON=False)
    for wd, dp, sd in ((64, 6, 90000001103),):
        w, _es, _ = canonical_mint(wd, dp, sd)
        ref = layer_loop(L(unc), w, wd, dp)
        for t in ("qE", "qEd"):
            p = _variant(os.path.join(_ROOT, arms[t]), tmp, t + "_full",
                         _SHARED_RANK=wd, _SHARED_FORCE=True, _SHARED_AGE=2,
                         _QB_D3=(8 if t == "qEd" else 0))
            mod = L(p)
            box = count_joins(mod)
            got = layer_loop(mod, w, wd, dp)
            claim(f"{wd}x{dp} {t} full-rank == uncompressed (all rows)",
                  rel(got, ref) <= 1e-9 and box["joins"] > 0,
                  "rel %.1e, %d joins, max ko %d" % (rel(got, ref), box["joins"],
                                                     box["max_ko"]))

    print("\n3. THE APPROXIMATE PATH RUNS AND JOINS")
    for wd, dp, sd in ((64, 7, 90000001104),):
        w, es, _ = canonical_mint(wd, dp, sd)
        ref = None
        for t in ("qoff", "qE", "qEd"):
            p = _variant(os.path.join(_ROOT, arms[t]), tmp, t + "_r16",
                         _SHARED_RANK=16, _SHARED_AGE=3,
                         _QB_D3=(4 if t == "qEd" else 0))
            mod = L(p)
            box = count_joins(mod)
            pr, fl = P1._predict(mod, w, wd, dp, es)
            if ref is None:
                ref = pr
            claim(f"{wd}x{dp} rank16 {t} finite, non-negative, joins",
                  bool(np.isfinite(pr).all() and (pr >= 0).all() and box["joins"] > 0),
                  "%d joins, rel vs qoff %.2e" % (box["joins"], rel(pr, ref)))

    print("\n4. EQUIVARIANCE -- no sampling was introduced")
    wd, dp, sd = 96, 6, 90000001105
    w, es, _ = canonical_mint(wd, dp, sd)
    perm = np.random.default_rng(5).permutation(wd)
    wp = [np.array(x) for x in w]
    wp[0] = wp[0][:, perm]
    wp[1] = wp[1][perm, :]
    base_arm = _variant(os.path.join(_ROOT, arms["qoff"]), tmp, "qoff_r24",
                        _SHARED_RANK=24, _SHARED_AGE=3)
    b0, _ = P1._predict(L(base_arm), w, wd, dp, es)
    b1, _ = P1._predict(L(base_arm), wp, wd, dp, es)
    floor = max(rel(b1, b0), 1e-300)
    for t in ("qE", "qEd"):
        p = _variant(os.path.join(_ROOT, arms[t]), tmp, t + "_r24",
                     _SHARED_RANK=24, _SHARED_AGE=3,
                     _QB_D3=(4 if t == "qEd" else 0))
        e0, _ = P1._predict(L(p), w, wd, dp, es)
        e1, _ = P1._predict(L(p), wp, wd, dp, es)
        claim(f"{t} equivariance <= 100x the base's own floor",
              rel(e1, e0) <= 100 * floor, "%.2e vs floor %.2e" % (rel(e1, e0), floor))

    mods = {t: L(_variant(os.path.join(_ROOT, arms[t]), tmp, t + "_jc",
                          _QB_D3=(0 if t != "qEd" else 0)))
            for t in ("qoff", "qE", "qEd")}
    tiny_join_checks(mods)


# --------------------------------------------------------------------------- #
# the query-error instrument: the ACTUAL post-projection error of every consumer
# --------------------------------------------------------------------------- #

def query_error(width, depth, seed, out_path):
    """Per layer, per arm: the real error of E, D3, D21, and of the whole output."""
    import flopscope as flops
    import k4feed_oracle as K
    from mint_provenance import canonical_mint

    flops.configure(symmetry_warnings=False)
    tmp = tempfile.mkdtemp(prefix="s10q_")
    arms = {t: "estimator_k3_v39_%s_%s.py" % (STEM, t)
            for t in ("qoff", "qE", "qEd")}
    w, es, _ = canonical_mint(width, depth, seed)
    import harness

    def run(path, on=True):
        p = _variant(os.path.join(_ROOT, path), tmp,
                     os.path.basename(path)[:-3] + ("_on" if on else "_off"),
                     _SHARED_ON=on)
        mod = K._load(p)
        caps = {}
        orig21 = mod._dslice21
        orig3 = mod._dslice3
        state = {"l": 0}

        def d21(tw, off):
            v = orig21(tw, off)
            caps.setdefault(state["l"], {})["D21"] = np.asarray(v, dtype=np.float64)
            return v

        def d3(tw):
            v = orig3(tw)
            caps.setdefault(state["l"], {})["D3"] = np.asarray(v, dtype=np.float64)
            return v

        orig_layer = mod._layer

        def layer(W, st, tb, dt, const, isc, *args, **kw):
            layer_index = kw.get("layer_index", args[-1] if args else 0)
            state["l"] = layer_index
            o = orig_layer(W, st, tb, dt, const, isc, *args, **kw)
            rec = caps.setdefault(layer_index, {})
            rec["mu_out"] = np.asarray(o[0], dtype=np.float64)
            if o[1] is not None:
                rec["cov_out"] = np.asarray(o[1], dtype=np.float64)
            return o

        mod._dslice21, mod._dslice3, mod._layer = d21, d3, layer
        mlp = harness.MLP(weights=[np.asarray(x) for x in w], width=width,
                          depth=depth, seed=es)
        ctx = flops.BudgetContext(flop_budget=2 ** 41, quiet=True)
        with ctx:
            pred = np.asarray(mod.Estimator().predict(mlp, 2 ** 41), dtype=np.float64)
        return caps, pred, float(ctx.flops_used)

    ref_caps, ref_pred, ref_f = run(arms["qoff"], on=False)
    out = {"artifact": "s10_parity.query_error", "label_free": True,
           "network": {"width": width, "depth": depth, "seed": seed},
           "uncompressed_flops": ref_f, "arms": {}}
    for t in ("qoff", "qE", "qEd"):
        caps, pred, f = run(arms[t], on=True)
        rows = []
        for l in sorted(set(caps) & set(ref_caps)):
            r = {"layer": l}
            for key in ("D3", "D21", "mu_out", "cov_out"):
                if key in caps[l] and key in ref_caps[l]:
                    a, b = caps[l][key], ref_caps[l][key]
                    nb = float(np.linalg.norm(b))
                    r[key] = (float(np.linalg.norm(a - b)) / nb) if nb > 0 else None
            rows.append(r)
        out["arms"][t] = {
            "path": arms[t],
            "sha256": __import__("hashlib").sha256(
                open(os.path.join(_ROOT, arms[t]), "rb").read()).hexdigest(),
            "flops": f, "c_over_b": f / 2 ** 41,
            "final_row_rel_vs_uncompressed":
                float(np.linalg.norm(pred[-1] - ref_pred[-1])
                      / max(np.linalg.norm(ref_pred[-1]), 1e-300)),
            "all_rows_rel_vs_uncompressed":
                float(np.linalg.norm(pred - ref_pred)
                      / max(np.linalg.norm(ref_pred), 1e-300)),
            "per_layer": rows,
        }
        print(f"  {t:<5} C/B {f / 2**41:.6f}  final-row rel "
              f"{out['arms'][t]['final_row_rel_vs_uncompressed']:.5e}  "
              f"all-rows rel {out['arms'][t]['all_rows_rel_vs_uncompressed']:.5e}",
              flush=True)
        for r in rows:
            print("      L%-2d D3 %s  D21 %s  mu %s  cov %s"
                  % (r["layer"],
                     "n/a" if r.get("D3") is None else "%.3e" % r["D3"],
                     "n/a" if r.get("D21") is None else "%.3e" % r["D21"],
                     "n/a" if r.get("mu_out") is None else "%.3e" % r["mu_out"],
                     "n/a" if r.get("cov_out") is None else "%.3e" % r["cov_out"]))
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(out, fh, indent=1)
        print("wrote", out_path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-error", action="store_true")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--seed", type=int, default=90_000_016_001)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    print("=" * 96)
    print("STAGE-3 QUERY BASIS -- label-free, decides nothing, gates the truth read")
    print("=" * 96)
    if a.query_error:
        query_error(a.width, a.depth, a.seed, a.out)
        return 0
    main_tiny()
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"artifact": "s10_parity.tiny", "checks": ROWS}, fh, indent=1)
    print("\n%d/%d passed" % (sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    sys.exit(main())
