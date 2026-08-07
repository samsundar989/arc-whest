"""
Focused tests for the Sprint-4 packed activation path.

`packed_sparse_probe --mode parity` compares whole predictions and is the S0
gate. These tests are the other half: they aim at the specific ways a packed
implementation goes wrong *without* producing an obviously wrong answer.

The failure mode this file exists to prevent is a packing scheme that quietly
drops small activations. That would look like a free speed-up on typical
networks, survive an end-to-end MSE check inside sampling noise, and then be an
approximation we had claimed was exact. So the central test drives a *tiny*
activation against a *huge* weight row and demands the contribution survive.

Run: uv run python local_sim/test_packed_sparse.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

import flopscope as flops  # noqa: E402  isort:skip
import flopscope.numpy as fnp  # noqa: E402  isort:skip

import numpy as np  # noqa: E402

for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

flops.configure(symmetry_warnings=False)

import packed_sparse_probe as PP  # noqa: E402
from mint_provenance import canonical_mint  # noqa: E402
from packed_sparse_probe import (BUDGET, Config, _bucket_rows,  # noqa: E402
                                 _partition_rows, build_plan, forward,
                                 layer_packed, run_mlp, strassen,
                                 undo_final_order)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {name:<52} {'ok ' if ok else 'FAILED'}  {detail}")
    return ok


def _np(x):
    return np.asarray(x)


def _packed_vs_dense(h_np, w_np, p, cfg):
    """Run one packed layer and un-permute it back to the caller's row order."""
    with flops.budget(10 ** 15, quiet=True):
        h = fnp.array(h_np)
        w = fnp.array(w_np)
        z = layer_packed(h, w, p, cfg)
        supp = (h[:, :p] > fnp.float32(0.0)).sum(axis=1)
        order, _segs = _partition_rows(supp, cfg, int(h.shape[0]))
        inv = fnp.argsort(order)
        z_back = fnp.take(z, inv, axis=0)
        ref = h @ w
    return _np(z_back), _np(ref), _np(order)


# ---------------------------------------------------------------- 1. exactness
def t_packed_equals_dense():
    print("\n1. PACKED == DENSE  (the reorganisation must be arithmetic, not approximate)")
    rng = np.random.default_rng(4001)
    ok = True
    for n, a, o, p, dens in ((512, 96, 64, 32, 0.30), (1024, 205, 205, 64, 0.55),
                             (256, 120, 33, 40, 0.10), (768, 200, 128, 100, 0.65)):
        raw = rng.standard_normal((n, a)).astype(np.float32)
        h = np.where(rng.random((n, a)) < dens, np.abs(raw), np.float32(0.0)).astype(np.float32)
        w = (rng.standard_normal((a, o)) * np.float32(np.sqrt(2.0 / a))).astype(np.float32)
        got, ref, _ = _packed_vs_dense(h, w, p, Config(packing=True))
        rel = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30))
        ok &= check(f"n={n} a={a} o={o} P={p} density={dens}", rel < 1e-6, f"max rel {rel:.3e}")
    return ok


def t_no_thresholding():
    """A tiny-but-nonzero activation must NEVER be dropped.

    Row r has exactly two prefix nonzeros: one enormous, one 1e-20. If the
    implementation truncated by magnitude, the small one would vanish; we make
    its weight row 1e20 so its contribution is O(1) and impossible to miss.
    """
    print("\n2. NO MAGNITUDE THRESHOLDING  (only exact 0.0 may be skipped)")
    n, a, o, p = 64, 40, 8, 24
    h = np.zeros((n, a), dtype=np.float32)
    w = np.zeros((a, o), dtype=np.float32)
    h[:, p:] = 1.0                                     # hot tail, all firing
    w[p:, :] = 0.01
    h[0, 3] = np.float32(1e-20)                        # the mouse
    h[0, 7] = np.float32(1e6)                          # the elephant
    w[3, :] = np.float32(1e20)                         # ... with a loud voice
    w[7, :] = np.float32(1e-6)
    for k in (1, 2, 4, 8):
        cfg = Config(packing=True, buckets=(k,))
        got, ref, _ = _packed_vs_dense(h, w, p, cfg)
        contrib = float(got[0, 0] - (h[0, p:] @ w[p:, 0]))
        check(f"bucket set ({k},): row 0 keeps BOTH nonzeros", abs(contrib - 1.0 - 1.0) < 1e-3,
              f"prefix contribution {contrib:.6f}, expected 2.0")
        check(f"bucket set ({k},): whole layer exact",
              float(np.abs(got - ref).max() / np.abs(ref).max()) < 1e-6, "")
    return all(RESULTS[-8:])


def t_bucket_boundaries():
    """Supports exactly at, one below and one above every bucket edge."""
    print("\n3. BUCKET BOUNDARIES  (0, 1, 2, 3/4, 5..8, 9.. and the whole prefix)")
    a, o, p = 48, 16, 32
    edges = (1, 2, 4, 8)
    supports = [0, 1, 2, 3, 4, 5, 7, 8, 9, 16, p]
    n = len(supports) * 3
    rng = np.random.default_rng(4002)
    h = np.zeros((n, a), dtype=np.float32)
    for i in range(n):
        s = supports[i % len(supports)]
        cols = rng.choice(p, size=s, replace=False) if s else []
        for c in cols:
            h[i, c] = np.float32(rng.random() + 0.05)
    h[:, p:] = np.abs(rng.standard_normal((n, a - p))).astype(np.float32)
    w = (rng.standard_normal((a, o)) * 0.1).astype(np.float32)
    cfg = Config(packing=True, buckets=edges)
    got, ref, order = _packed_vs_dense(h, w, p, cfg)
    rel = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30))
    ok = check("every support size 0..P is exact", rel < 1e-6, f"max rel {rel:.3e}")

    with flops.budget(10 ** 15, quiet=True):
        supp = (fnp.array(h)[:, :p] > fnp.float32(0.0)).sum(axis=1)
        od, segs = _bucket_rows(supp, edges, n)
    sup = _np(supp)
    seen = np.zeros(n, dtype=int)
    good = True
    for k, s, e in segs:
        rows = _np(od)[s:e]
        seen[rows] += 1
        lo = 0 if k in (0, 1) else max([x for x in edges if x < k], default=0)
        if k == 0:
            good &= bool((sup[rows] == 0).all())
        elif k == -1:
            good &= bool((sup[rows] > max(edges)).all())
        else:
            good &= bool(((sup[rows] > lo) & (sup[rows] <= k)).all())
    ok &= check("partition covers every row exactly once", bool((seen == 1).all()),
                f"{len(segs)} segments")
    ok &= check("each segment holds only its own support class", good, "")
    return ok


def t_all_zero_all_hot():
    print("\n4. DEGENERATE BLOCKS  (all-zero prefix, all-hot prefix, empty buckets)")
    a, o, p = 64, 24, 32
    rng = np.random.default_rng(4003)
    w = (rng.standard_normal((a, o)) * 0.1).astype(np.float32)
    cases = {
        "prefix identically zero": np.concatenate(
            [np.zeros((128, p), np.float32),
             np.abs(rng.standard_normal((128, a - p))).astype(np.float32)], axis=1),
        "prefix fully dense": np.abs(rng.standard_normal((128, a))).astype(np.float32) + 0.1,
        "entire h zero": np.zeros((128, a), np.float32),
        "single row": np.abs(rng.standard_normal((1, a))).astype(np.float32),
    }
    ok = True
    for name, h in cases.items():
        got, ref, _ = _packed_vs_dense(h, w, p, Config(packing=True))
        denom = max(float(np.abs(ref).max()), 1e-30)
        rel = float(np.abs(got - ref).max() / denom)
        ok &= check(name, rel < 1e-6 and np.isfinite(got).all(), f"max rel {rel:.3e}")
    return ok


def t_odd_sizes():
    print("\n5. ODD AND AWKWARD SIZES")
    rng = np.random.default_rng(4004)
    ok = True
    for n, a, o, p in ((101, 77, 13, 31), (33, 41, 41, 9), (7, 129, 1, 17), (2048, 199, 199, 103)):
        h = np.where(rng.random((n, a)) < 0.4,
                     np.abs(rng.standard_normal((n, a))).astype(np.float32), np.float32(0)).astype(np.float32)
        w = (rng.standard_normal((a, o)) * 0.1).astype(np.float32)
        got, ref, _ = _packed_vs_dense(h, w, p, Config(packing=True))
        rel = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30))
        ok &= check(f"n={n} a={a} o={o} P={p}", rel < 1e-6, f"max rel {rel:.3e}")
    return ok


# ---------------------------------------------------------------- 2. Strassen
def t_strassen():
    print("\n6. STRASSEN  (algebraically exact; padded, not skipped, on odd sizes)")
    rng = np.random.default_rng(4005)
    ok = True
    for m, k, nn, d in ((512, 205, 205, 1), (512, 141, 33, 1), (1024, 256, 199, 2),
                        (2048, 150, 205, 3), (64, 9, 7, 2)):
        A = (rng.standard_normal((m, k)) * 0.1).astype(np.float32)
        B = (rng.standard_normal((k, nn)) * 0.1).astype(np.float32)
        with flops.budget(10 ** 15, quiet=True) as b1:
            ref = _np(fnp.array(A) @ fnp.array(B))
        with flops.budget(10 ** 15, quiet=True) as b2:
            got = _np(strassen(fnp.array(A), fnp.array(B), d))
        rel = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30))
        saved = float(b1.flops_used) / max(float(b2.flops_used), 1.0)
        ok &= check(f"({m},{k})@({k},{nn}) depth {d}", got.shape == ref.shape and rel < 2e-5,
                    f"max rel {rel:.2e}, cost {saved:.3f}x dense")
    return ok


# ------------------------------------------------- 3. permutation correspondence
def t_permutation():
    print("\n7. PERMUTATION CORRESPONDENCE  (column order is folded into BOTH sides)")
    width, depth = 256, 32
    w_np, est_seed, _ = canonical_mint(width, depth, 30_806_011_500)
    cfg_d = Config()
    cfg_p = Config(packing=True, dyn_order=True)
    _, _, base = run_mlp(w_np, est_seed, cfg_d, width, depth, n_shifts=1)

    rng = np.random.default_rng(4006)
    ok = True
    for k in (5, 17, 30):
        P = rng.permutation(width)
        w2 = [np.array(x) for x in w_np]
        w2[k] = w2[k][:, P]
        w2[k + 1] = w2[k + 1][P, :]
        _, _, perm = run_mlp(w2, est_seed, cfg_p, width, depth, n_shifts=1)
        rel = float(np.abs(perm - base).max() / np.abs(base).max())
        ok &= check(f"relabel hidden layer {k}: scored row unchanged", rel < 1e-5,
                    f"max rel {rel:.3e}")

    # the packed path's own ordering must be undone exactly at the scored layer
    with flops.budget(10 ** 15, quiet=True):
        wf = [fnp.array(x) for x in w_np]
        plan = build_plan(wf, Config(on_compose=True), width, depth)
        v = fnp.arange(width).astype(fnp.float64)
        forder = plan.final_split[3]
        back = _np(undo_final_order(fnp.take(v, forder), plan, width))
    ok &= check("undo_final_order inverts the class permutation",
                bool(np.array_equal(back, np.arange(width))), "")
    return ok


# ------------------------------------------------- 4. classification boundaries
def t_classification():
    print("\n8. DEAD / ON / KINK BOUNDARIES  (classes partition the layer, no gaps or overlaps)")
    width, depth = 128, 8
    w_np, est_seed, _ = canonical_mint(width, depth, 30_806_011_501)
    ok = True
    for a_on in (0.0, 2.5, 3.0, 12.0, -2.5):
        cfg = Config(on_compose=True, alpha_on=a_on)
        with flops.budget(10 ** 15, quiet=True):
            plan = build_plan([fnp.array(x) for x in w_np], cfg, width, depth)
        nd, no, nk = plan.final_split[:3]
        forder = _np(plan.final_split[3])
        ok &= check(f"alpha_on={a_on:>5}: classes sum to width and are a permutation",
                    nd + no + nk == width and len(set(forder.tolist())) == width,
                    f"dead {nd} on {no} kink {nk}")
    # the boundary is >= alpha_on for 'on' and <= tau for 'dead': a neuron sitting
    # exactly on either edge must land in exactly one class, which the sum above proves.
    for a_on in (2.5, 3.0):
        _, _, out = run_mlp(w_np, est_seed, Config(on_compose=True, alpha_on=a_on),
                            width, depth, n_shifts=1)
        ok &= check(f"alpha_on={a_on}: output finite and non-negative",
                    bool(np.isfinite(out).all() and (out >= -1e-12).all()),
                    f"min {out.min():.3e}")
    return ok


# ------------------------------------------------- 5. antithetic accounting
def t_antithetic():
    print("\n9. ANTITHETIC SAMPLE ACCOUNTING  (N rows, not N/2; the reflection is exact)")
    width, depth = 256, 32
    w_np, est_seed, _ = canonical_mint(width, depth, 30_806_011_502)
    cfg = Config(antithetic=True, log2_n=12)
    n = 1 << cfg.log2_n
    with flops.budget(10 ** 15, quiet=True):
        wf = [fnp.array(x) for x in w_np]
        plan = build_plan(wf, cfg, width, depth)
        shift = fnp.array(np.random.default_rng(int(est_seed)).integers(
            0, 1 << 30, size=width, dtype=np.uint32))
        h = PP._points(n, width, shift, cfg)
        z = h @ plan.sliced[0]
        z_full = fnp.concatenate([z, -z], axis=0)
    ok = check("point set holds N/2 rows before reflection", int(h.shape[0]) == n // 2,
               f"{int(h.shape[0])} of {n}")
    ok &= check("reflected block restores N rows", int(z_full.shape[0]) == n, "")
    zf = _np(z_full)
    ok &= check("column sums of layer-1 pre-activations vanish exactly",
                float(np.abs(zf.sum(axis=0)).max()) < 1e-3,
                f"max |sum| {float(np.abs(zf.sum(axis=0)).max()):.2e} over {n} rows")
    ok &= check("z(-x) == -z(x) elementwise", bool(np.array_equal(zf[n // 2:], -zf[: n // 2])), "")
    _, bills, out = run_mlp(w_np, est_seed, cfg, width, depth, n_shifts=1)
    ok &= check("full estimate stays finite and correctly shaped",
                out.shape == (width,) and bool(np.isfinite(out).all()), "")
    return ok


# ------------------------------------------------- 6. determinism / hygiene
def t_determinism():
    print("\n10. DETERMINISM  (same seed, bit-identical, three times, fresh state)")
    width, depth = 256, 32
    w_np, est_seed, _ = canonical_mint(width, depth, 30_806_011_503)
    ok = True
    for lbl, cfg in (("dense", Config()),
                     ("packed", Config(packing=True, dyn_order=True)),
                     ("combined", Config(packing=True, dyn_order=True, antithetic=True,
                                         strassen=1, on_compose=True, pilot=256))):
        runs = [run_mlp(w_np, est_seed, cfg, width, depth, n_shifts=1)[2] for _ in range(3)]
        same = all(np.array_equal(runs[0], r) for r in runs[1:])
        PP._BASE_POINTS.clear()                       # the cached Sobol net must not matter
        again = run_mlp(w_np, est_seed, cfg, width, depth, n_shifts=1)[2]
        ok &= check(f"{lbl}: 3x identical + cold cache identical",
                    same and np.array_equal(runs[0], again), "")
    return ok


def t_no_mutation():
    print("\n11. GRADER-ARRAY IMMUTABILITY  (inputs must come back untouched)")
    width, depth = 256, 32
    w_np, est_seed, _ = canonical_mint(width, depth, 30_806_011_504)
    before = [x.copy() for x in w_np]
    ok = True
    for lbl, cfg in (("dense", Config()),
                     ("combined", Config(packing=True, dyn_order=True, antithetic=True,
                                         strassen=1, on_compose=True, pilot=256))):
        run_mlp(w_np, est_seed, cfg, width, depth, n_shifts=1)
        ok &= check(f"{lbl}: weights unmodified",
                    all(np.array_equal(a, b) for a, b in zip(before, w_np)), "")
    return ok


def t_nonfinite():
    print("\n12. NON-FINITE AND DEGENERATE NETWORKS")
    width, depth = 256, 16
    rng = np.random.default_rng(4007)
    scale = np.float32(np.sqrt(2.0 / width))
    cfg = Config(packing=True, dyn_order=True, antithetic=True, strassen=1, on_compose=True)
    ok = True
    cases = {}
    base = [(rng.standard_normal((width, width)) * scale).astype(np.float32)
            for _ in range(depth)]
    cases["plain"] = base
    z = [x.copy() for x in base]
    z[5] = np.zeros_like(z[5])
    cases["layer 5 identically zero"] = z
    b = [x.copy() for x in base]
    b[7] = b[7] * np.float32(1e4)
    cases["layer 7 scaled 1e4"] = b
    s = [x.copy() for x in base]
    s[3] = s[3] * np.float32(1e-4)
    cases["layer 3 scaled 1e-4"] = s
    o = [x.copy() for x in base]
    o[9] = o[9] + np.float32(3.0) * scale
    cases["layer 9 mean-shifted (all on)"] = o
    for name, w in cases.items():
        try:
            _, _, out = run_mlp(w, 12345, cfg, width, depth, n_shifts=1)
            good = out.shape == (width,) and bool(np.isfinite(out).all()) and (out >= -1e-9).all()
            ok &= check(name, good, f"range [{out.min():.2e}, {out.max():.2e}]")
        except Exception as e:                          # a raise on the grader scores zero
            ok &= check(name, False, f"RAISED {type(e).__name__}: {e}")
    return ok


def t_shape_and_memory():
    print("\n13. RESULT-ARRAY AND MEMORY LIMITS")
    width, depth = 256, 32
    w_np, est_seed, _ = canonical_mint(width, depth, 30_806_011_505)
    cfg = Config(packing=True, dyn_order=True, antithetic=True, strassen=1, on_compose=True)
    n = 1 << cfg.log2_n
    with flops.budget(10 ** 15, quiet=True):
        wf = [fnp.array(x) for x in w_np]
        plan = build_plan(wf, cfg, width, depth)
        shift = fnp.array(np.random.default_rng(int(est_seed)).integers(
            0, 1 << 30, size=width, dtype=np.uint32))
        out = forward(plan, cfg, shift, width, depth)
        full = fnp.stack(plan.rows[:-1] + [undo_final_order(out, plan, width)], axis=0)
    a = _np(full)
    ok = check("full answer is (depth, width) float64", a.shape == (depth, width), str(a.dtype))
    ok &= check("answer finite and non-negative",
                bool(np.isfinite(a).all()) and bool((a >= -1e-12).all()), "")
    # largest live array: the packed path never exceeds the dense pass's (N, width),
    # plus the (rows, k, out) gather, whose worst bucket is the widest one.
    biggest = max(n * width * 4, n * max(cfg.buckets) * width * 4)
    ok &= check("largest single array well under the 4 GiB cap",
                biggest < 4 * 1024 ** 3, f"{biggest / 1e6:.1f} MB worst case")
    ok &= check("Sobol cache holds one entry per (n, width)",
                len(PP._BASE_POINTS) <= 4, f"{len(PP._BASE_POINTS)} entries")
    return ok


def t_oracle():
    print("\n14. COST ORACLE vs INSTRUMENTED BILL  (the model must not flatter us)")
    width, depth = 256, 32
    w_np, est_seed, _ = canonical_mint(width, depth, 30_806_011_506)
    ok = True
    for lbl, cfg in (("dense", Config()),
                     ("packed", Config(packing=True, dyn_order=True)),
                     ("packed+strassen", Config(packing=True, dyn_order=True, strassen=1)),
                     ("antithetic", Config(antithetic=True))):
        _, bills, _ = run_mlp(w_np, est_seed, cfg, width, depth, n_shifts=1)
        r = bills[0]["oracle"] / bills[0]["flops_sampling"]
        ok &= check(f"{lbl}: oracle / counted sampling FLOPs", 0.97 < r < 1.03, f"{r:.4f}")
    return ok


# ------------------------------------------------- 7. SPRINT 5: batched Strassen
def t_strassen_batched():
    """The batched leaf dispatch must be BITWISE identical to the recursion.

    Both paths share `_ops7` and `_combine`, so they perform the same additions
    in the same order and differ only in whether the 7 (or 49) leaf products are
    issued as separate matmuls or one stacked contraction. Floating-point
    addition is not associative, so `array_equal` -- not a tolerance -- is the
    right assertion: any tolerance here would hide a genuinely different
    algorithm behind a plausible number.
    """
    print("\n15. BATCHED STRASSEN == RECURSIVE STRASSEN (bitwise)")
    rng = np.random.default_rng(4008)
    ok = True
    cases = ((512, 205, 205, 1), (512, 205, 205, 2), (1024, 199, 187, 2),
             (2048, 256, 256, 2), (256, 141, 33, 1), (64, 9, 7, 2),
             (4096, 200, 200, 2), (512, 205, 205, 3))
    for m, k, nn, d in cases:
        A = (rng.standard_normal((m, k)) * 0.1).astype(np.float32)
        B = (rng.standard_normal((k, nn)) * 0.1).astype(np.float32)
        with flops.budget(10 ** 15, quiet=True) as bd:
            ref = _np(fnp.array(A) @ fnp.array(B))
        with flops.budget(10 ** 15, quiet=True) as br:
            rec = _np(strassen(fnp.array(A), fnp.array(B), d, "recursive"))
        with flops.budget(10 ** 15, quiet=True) as bb:
            bat = _np(strassen(fnp.array(A), fnp.array(B), d, "batched"))
        cr = sum(v["calls"] for v in br.summary_dict()["operations"].values())
        cb = sum(v["calls"] for v in bb.summary_dict()["operations"].values())
        rel = float(np.abs(bat - ref).max() / max(np.abs(ref).max(), 1e-30))
        ok &= check(f"({m},{k})@({k},{nn}) d={d}: bitwise == recursive",
                    bool(np.array_equal(rec, bat)) and bat.shape == ref.shape,
                    f"calls {cr}->{cb}, rel vs dense {rel:.2e}, "
                    f"flops {float(bb.flops_used) / float(bd.flops_used):.4f}x")
    return ok


def t_strassen_batched_edges():
    print("\n16. BATCHED STRASSEN EDGES  (depth 0, odd dims, fallback, determinism)")
    rng = np.random.default_rng(4009)
    A = (rng.standard_normal((1024, 205)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((205, 205)) * 0.1).astype(np.float32)
    with flops.budget(10 ** 15, quiet=True):
        fa, fb = fnp.array(A), fnp.array(B)
        d0 = _np(strassen(fa, fb, 0, "batched"))
        dn = _np(fa @ fb)
    ok = check("depth 0 is the plain matmul, bitwise", bool(np.array_equal(d0, dn)), "")
    with flops.budget(10 ** 15, quiet=True):
        reps = [_np(strassen(fnp.array(A), fnp.array(B), 2, "batched")) for _ in range(3)]
    ok &= check("depth 2 batched deterministic over 3 runs",
                all(np.array_equal(reps[0], r) for r in reps[1:]), "")
    for m, k, nn in ((1020, 199, 187), (1024, 3, 3), (13, 205, 205), (1024, 205, 1)):
        with flops.budget(10 ** 15, quiet=True):
            fa = fnp.array((rng.standard_normal((m, k)) * .1).astype(np.float32))
            fb = fnp.array((rng.standard_normal((k, nn)) * .1).astype(np.float32))
            ref = _np(fa @ fb)
            got = _np(strassen(fa, fb, 2, "batched"))
        rel = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30))
        ok &= check(f"({m},{k})@({k},{nn}) d=2 batched: shape, finiteness, value",
                    got.shape == ref.shape and bool(np.isfinite(got).all()) and rel < 2e-5,
                    f"rel {rel:.2e}")
    return ok


# ------------------------------------------------- 8. SPRINT 5: bucket designs
def t_bucket_designs():
    """Every bucket layout must be exact, including the degenerate ones.

    `buckets=()` is the zero-rectangle-only design: no gathered corrections at
    all, just skip the rows whose entire prefix is 0.0 and run the rest dense.
    `buckets=(1,)` is the narrowest possible gather. Both are corner cases where
    an off-by-one in the segment arithmetic would show up as dropped rows rather
    than as a wrong number, which is why row coverage is asserted separately.
    """
    print("\n17. BUCKET DESIGNS  (0/1/2/3/5 classes x 2 partitioners x 2 gathers)")
    rng = np.random.default_rng(4010)
    layouts = [(), (1,), (8,), (16,), (4, 16), (8, 16), (2, 8, 16), (1, 2, 4, 8, 16)]
    ok = True
    for n, a, o, p, dens in ((1024, 205, 205, 64, 0.55), (777, 120, 61, 40, 0.15)):
        raw = rng.standard_normal((n, a)).astype(np.float32)
        h = np.where(rng.random((n, a)) < dens, np.abs(raw),
                     np.float32(0.0)).astype(np.float32)
        w = (rng.standard_normal((a, o)) * np.float32(np.sqrt(2.0 / a))).astype(np.float32)
        worst = 0.0
        for bk in layouts:
            for mode in ("flatnonzero", "segment"):
                for gm in ("einsum", "slotloop"):
                    cfg = Config(packing=True, buckets=bk, bucket_mode=mode,
                                 gather_mode=gm)
                    got, ref, _ = _packed_vs_dense(h, w, p, cfg)
                    rel = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30))
                    worst = max(worst, rel)
                    if rel >= 1e-6:
                        check(f"n={n} b={bk} {mode[:4]}/{gm[:4]}", False,
                              f"max rel {rel:.3e}")
                        ok = False
        ok &= check(f"n={n} a={a} P={p}: all 32 layout/mode combinations exact",
                    worst < 1e-6, f"worst max rel {worst:.3e}")
    return ok


def t_segment_partition():
    """`segment` and `flatnonzero` must induce the SAME partition into classes."""
    print("\n18. SEGMENT PARTITION  (one ordering op reproduces the mask sweep)")
    rng = np.random.default_rng(4011)
    ok = True
    for n, p, edges in ((4096, 64, (1, 2, 4, 8, 16)), (4096, 64, (8,)), (512, 30, ()),
                        (300, 24, (1,)), (1024, 48, (2, 8, 16))):
        h = np.where(rng.random((n, p)) < 0.08,
                     np.abs(rng.standard_normal((n, p))).astype(np.float32),
                     np.float32(0.0)).astype(np.float32)
        with flops.budget(10 ** 15, quiet=True):
            supp = (fnp.array(h) > fnp.float32(0.0)).sum(axis=1)
            o1, s1 = _partition_rows(supp, Config(buckets=edges), n)
            o2, s2 = _partition_rows(supp, Config(buckets=edges,
                                                  bucket_mode="segment"), n)
        cls1 = {k: set(_np(o1)[s:e].tolist()) for k, s, e in s1}
        cls2 = {k: set(_np(o2)[s:e].tolist()) for k, s, e in s2}
        cover = sorted(sum((list(v) for v in cls2.values()), [])) == list(range(n))
        ok &= check(f"n={n} P={p} edges={edges if edges else '()'}: identical classes",
                    cls1 == cls2 and cover,
                    f"{len(s1)} vs {len(s2)} segments, coverage {cover}")
    return ok


def t_never_truncate():
    """A row with MORE nonzeros than its bucket width must never be truncated.

    With `buckets=(1,)` a row holding two nonzeros cannot be served by the
    one-slot gather, so it must be routed to the dense remainder. If it were
    served anyway, `argpartition` would keep the larger entry and silently drop
    the other -- the exact failure this lane must not have.
    """
    print("\n19. NO ROW IS EVER TRUNCATED  (over-full rows take the dense remainder)")
    a, o, p = 40, 8, 24
    n = 96
    rng = np.random.default_rng(4012)
    ok = True
    for bk in ((1,), (2,), (4,), (1, 2)):
        h = np.zeros((n, a), dtype=np.float32)
        h[:, p:] = np.abs(rng.standard_normal((n, a - p))).astype(np.float32)
        for i in range(n):
            for c in rng.choice(p, size=(i % (max(bk) + 3)), replace=False):
                h[i, c] = np.float32(1.0 + rng.random())
        w = (rng.standard_normal((a, o)) * 0.3).astype(np.float32)
        worst = 0.0
        for mode in ("flatnonzero", "segment"):
            for gm in ("einsum", "slotloop"):
                cfg = Config(packing=True, buckets=bk, bucket_mode=mode, gather_mode=gm)
                got, ref, _ = _packed_vs_dense(h, w, p, cfg)
                worst = max(worst, float(np.abs(got - ref).max()
                                         / max(np.abs(ref).max(), 1e-30)))
        ok &= check(f"b={bk}: supports 0..{max(bk) + 2} exact in all 4 modes",
                    worst < 1e-6, f"max rel {worst:.3e}")
    return ok


def t_sprint5_oracle():
    print("\n20. COST ORACLE COVERS THE SPRINT-5 PATHS")
    width, depth = 256, 32
    w_np, est_seed, _ = canonical_mint(width, depth, 30_806_013_500)
    ok = True
    arms = (("strassen1 batched", Config(strassen=1, strassen_mode="batched")),
            ("strassen2 batched", Config(strassen=2, strassen_mode="batched")),
            ("pack b=(8,)", Config(packing=True, dyn_order=True, buckets=(8,))),
            ("pack segment", Config(packing=True, dyn_order=True,
                                    bucket_mode="segment")),
            ("pack slotloop", Config(packing=True, dyn_order=True, buckets=(8,),
                                     gather_mode="slotloop")),
            ("pack+str2 batched", Config(packing=True, dyn_order=True, strassen=2,
                                         strassen_mode="batched")))
    for lbl, cfg in arms:
        _, bills, _ = run_mlp(w_np, est_seed, cfg, width, depth, n_shifts=1)
        r = bills[0]["oracle"] / bills[0]["flops_sampling"]
        ok &= check(f"{lbl}: oracle / counted sampling FLOPs", 0.97 < r < 1.03, f"{r:.4f}")
    return ok



def main():
    print("=" * 96)
    print("PACKED ACTIVATION PROPAGATION — focused tests")
    print("=" * 96)
    for fn in (t_packed_equals_dense, t_no_thresholding, t_bucket_boundaries,
               t_all_zero_all_hot, t_odd_sizes, t_strassen, t_permutation,
               t_classification, t_antithetic, t_determinism, t_no_mutation,
               t_nonfinite, t_shape_and_memory, t_oracle,
               t_strassen_batched, t_strassen_batched_edges, t_bucket_designs,
               t_segment_partition, t_never_truncate, t_sprint5_oracle):
        fn()
    n_ok = sum(1 for r in RESULTS if r)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
