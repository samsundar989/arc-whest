"""WP16 / Blocker A: is the `fnp` port the SAME FUNCTION as upstream at non-zero
`kappa_3`?

Contract: `experiments/LOG.md`, 2026-09-03, "CONTRACT (Rule 1): WP16", HEAD
`7169392`, committed before any data existed.

WHY THIS EXISTS
---------------
`NOT_DONE` step 6 of `p2_k3_fnp.py` has been open since the port was written, and
on 2026-09-02 it stopped being hygiene: the canary built from that port scored
`2.8192e-06` on 20 official rows where the torch reference had measured
`8.5075e-07` -- a `3.31x` miss -- while the matched v2.6.1 control transferred to
`3.2%`. The split, the networks and the instrument all transfer. **The port and
the reference are not the same function**, and every WP15 gate is written against
the reference's accuracy.

WHAT IT COMPARES, AND WHY IT IS A LADDER RATHER THAN ONE NUMBER
----------------------------------------------------------------
A single end-to-end drift says "they differ" and nothing else. This walks the two
programs side by side and compares, at EVERY layer, at TWO points:

    lin   after `linear_kprop`      mu_pre, cov_pre, kappa_3's (3,) and (2,1)
                                    repeated slices, K[4]'s core
    nl    after the nonlinearity,   mu, cov, the (3,) and (2,1) repeated slices
          BEFORE regeneration       of the OUTGOING kappa_3, the rebuilt M, K[4]

so the first layer and the first component at which they part company is a
readout, not an inference. Every `(n, n)` object is compared BOTH ways round --
`X` against `Y` and `X` against `Y^T` -- because "which of `kappa_3[i,i,j]` and
`kappa_3[i,j,j]` is stored" is one of the named suspects and an orientation error
is otherwise indistinguishable from a coefficient error.

TWO VENVS, AND THEY CANNOT BE MERGED
-------------------------------------
The port needs real flopscope and must never see torch; the reference needs torch
and upstream's checkout and has no flopscope. They therefore run as two separate
processes that exchange `.npz` files, and `--compare` refuses unless both sides
carry the SAME sha256 over the minted weights. "Same networks" is proved here,
not assumed.

    uv run python local_sim/p2_k3_parity.py --port --width 128 --depth 4 --nets 2 \
        --dtype f64 --regen-mode sub   --out experiments/.../port_sub.npz
    .venv-gpu\\Scripts\\python.exe local_sim/p2_k3_parity.py --ref --width 128 \
        --depth 4 --nets 2 --dtype f64 --device cpu --out experiments/.../ref.npz
    uv run python local_sim/p2_k3_parity.py --compare experiments/.../port_sub.npz \
        experiments/.../ref.npz

SEEDS
-----
`P01[40,821,001,168 .. 40,821,001,175]` -- the same eight networks WP13 E2 and
WP14 C0-i used, imported from `p2_range_lab.P01_SLICE_SEEDS` rather than written
again as a literal. Re-driving exactly those networks is the point: their truth
is already minted, so the fixed port can be scored against it without spending a
fresh seed. The overlap is DELIBERATE, is declared in the ledger with
`declare_pair`, and makes this result CORRELATED evidence with WP13 E2 and WP14
C0-i -- never a third independent one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# APPEND, never insert: `local_sim/` holds numpy shims named `flopscope` and
# `whestbench`, and prepending this directory would swap the real packages for
# them on the port side.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

WIDTH, DEPTH = 1024, 16

#: Which components are cheap enough to carry for EVERY layer (vectors and
#: scalars), against those only kept for `--full-layers` (the `(n, n)` objects).
_VEC = {"mu_pre", "s3", "c4_in", "mu", "K3", "S3", "c4", "p111_3"}
_MAT = {"cov_pre", "s21", "cov", "K21", "S21", "M", "p111_21"}


# ==========================================================================
# provenance -- "the same networks" is proved, not asserted
# ==========================================================================

def _seeds(n_nets: int, width: int, depth: int):
    """The P01 networks, or a DERIVED smoke range at any other shape.

    At the live shape the seeds are `P01`'s, imported from `p2_range_lab` so the
    literal exists in exactly one place. At any other shape they cannot be P01 --
    a seed names a network only together with its shape -- so a clearly-labelled
    smoke range is used and NOTHING measured there may be quoted as a result on
    the claimed networks.
    """
    if (width, depth) == (WIDTH, DEPTH):
        import p2_range_lab as rl

        seeds = list(rl.P01_SLICE_SEEDS[:n_nets])
        assert len(seeds) == n_nets, f"P01 holds 8 slice seeds, asked for {n_nets}"
        return seeds, "P01[40,821,001,168..40,821,001,175]"
    return ([1_000_921 + i for i in range(n_nets)],
            f"SMOKE (not P01) at {width}x{depth} -- shape differs, never quotable "
            "as a result on the claimed networks")


def _mint(width, depth, seeds):
    from mint_provenance import canonical_mint

    return [canonical_mint(width, depth, s)[0] for s in seeds]


def _weights_sha(nets) -> str:
    """One hash over every weight of every network, f32, C-order."""
    h = hashlib.sha256()
    for w in nets:
        for a in w:
            h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes())
    return h.hexdigest()


def _np(x):
    """Anything -> a float64 numpy array. Works for fnp arrays and torch tensors."""
    if x is None:
        return None
    if hasattr(x, "detach"):                       # torch
        return x.detach().to("cpu").double().numpy()
    return np.asarray(x, dtype=np.float64)


# ==========================================================================
# side A -- the port, in fnp, under a real BudgetContext
# ==========================================================================

def run_port(width, depth, seeds, *, dtype_name, regen_mode, full_layers):
    import flopscope as flops
    import flopscope.numpy as fnp

    import p2_k3_fnp as port

    port._require_real_flopscope(flops)
    table = port.load_coefs()
    np_dt = np.float64 if dtype_name == "f64" else np.float32
    fnp_dt = fnp.float64 if dtype_name == "f64" else fnp.float32

    nets = _mint(width, depth, seeds)
    blob, meta = {}, []
    for i, w_np in enumerate(nets):
        rec = {}

        def probe(l, tag, d, _rec=rec, _i=i):
            # `(n, n)` objects for NET 0 ONLY: at 1024 they are 8 MB apiece and
            # eight networks x sixteen layers would be ~7 GiB of ladder for a
            # table that is read one network at a time. The `(n,)` components are
            # kept for every network at every layer.
            keep = _VEC | (_MAT if (l in full_layers and _i == 0) else set())
            for k, v in d.items():
                if k in keep and v is not None:
                    _rec[f"n{i}_l{l}_{tag}_{k}"] = _np(v)

        audit = port.DtypeAudit(str(np.dtype(np_dt)))
        w = [fnp.array(np.asarray(x, dtype=np_dt)) for x in w_np]
        with flops.BudgetContext(flop_budget=100 * port.BUDGET, quiet=True) as ctx:
            stacked, _st = port.k3_flow(fnp, flops, w, table, fnp_dt, audit,
                                        regen_mode=regen_mode, probe=probe)
        blob[f"n{i}_means"] = _np(stacked)
        blob.update(rec)
        meta.append({"net": i, "seed": int(seeds[i]),
                     "flops": float(ctx.flops_used),
                     "ops": len(getattr(ctx, "op_log", []) or []),
                     "dtype_violations": len(audit.summary()["violations"])})
    return blob, {"side": "port", "regen_mode": regen_mode, "dtype": dtype_name,
                  "per_net": meta}


# ==========================================================================
# side B -- upstream's own flow, in torch, with WP13's regeneration hook
# ==========================================================================

def run_ref(width, depth, seeds, *, dtype_name, device, full_layers):
    import torch

    import p2_kprop_bridge as bridge
    import p2_range_lab as rl

    # `_src()` is what puts upstream's checkout on the path, and it also refuses
    # if `MLP_KPROP_SRC` is not a directory. Import of upstream must follow it.
    src = bridge._src()
    ver = bridge.verify_upstream(src)
    if not ver["pass"]:
        raise SystemExit(
            f"REFUSED: upstream checkout is not rev {bridge.UPSTREAM_REV[:8]} plus "
            f"exactly the recorded patch: {ver}")
    from mlp_kprop.factor_k3 import FactoredTensor

    dt = torch.float64 if dtype_name == "f64" else torch.float32
    nets = _mint(width, depth, seeds)
    blob = {}

    def _core(K, k):
        if k not in K:
            return None
        o = K[k]
        return getattr(o, "core", o)

    def _rep(K):
        """kappa_3's two repeated slices, or zeros before any kappa_3 exists."""
        if 3 not in K or not hasattr(K[3], "_factors"):
            z1 = torch.zeros(width, device=device, dtype=dt)
            return None, z1, torch.zeros(width, width, device=device, dtype=dt)
        ds = K[3].get_repeated()
        return ds, ds.slices[(3,)], ds.slices[(2, 1)]

    for i, w_np in enumerate(nets):
        rec = {}

        def store(l, tag, d, _rec=rec, _i=i):
            # Net 0 only for the `(n, n)` objects -- see `run_port`.
            keep = _VEC | (_MAT if (l in full_layers and _i == 0) else set())
            for k, v in d.items():
                if k in keep and v is not None:
                    _rec[f"n{_i}_l{l}_{tag}_{k}"] = _np(v)

        def hook_lin(l, K):
            _ds, s3, s21 = _rep(K)
            store(l, "lin", {"mu_pre": _core(K, 1), "cov_pre": _core(K, 2),
                             "s3": s3, "s21": s21, "c4_in": _core(K, 4)})
            return K

        def hook_nl(l, K, W_next):
            # `get_repeated()` here is what WP13's regeneration hook consumes,
            # and under CONTRACT WP16's H1 it equals the port's UN-subtracted
            # `K[(3,)]` / `K[(2,1)]`. Computed once and reused for the rebuild so
            # this observer cannot change the arm it observes.
            ds, s3, s21 = _rep(K)
            rec_d = {"mu": _core(K, 1), "cov": _core(K, 2),
                     "K3": s3, "K21": s21, "c4": _core(K, 4)}
            if ds is not None:
                K[3] = FactoredTensor.from_dstensor(ds)
                rec_d["M"] = K[3].factors[0]
            store(l, "nl", rec_d)
            return K

        mlp = bridge.build_mlp(w_np, device=device, dtype=dt)
        K_in = {1: torch.zeros(width, device=device, dtype=dt),
                2: torch.eye(width, device=device, dtype=dt)}
        with torch.no_grad():
            out = rl.run_flow(mlp, K_in, hook_lin=hook_lin, hook_nl=hook_nl,
                              up_to_layer=f"act{depth - 1}", output_d_max=1)
        blob[f"n{i}_means"] = np.stack(
            [bridge._degree1(out[f"act{l}"][1]) for l in range(depth)], axis=0)
        blob.update(rec)
        del mlp, out
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return blob, {"side": "ref", "regen_mode": "dsproj (upstream regen hook)",
                  "dtype": dtype_name, "device": device}


# ==========================================================================
# the comparison
# ==========================================================================

def _rel(a, b):
    """||a - b|| / ||b||, with a zero-reference convention that cannot flatter."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    nb = float(np.linalg.norm(b))
    na = float(np.linalg.norm(a))
    if nb == 0.0:
        return 0.0 if na == 0.0 else float("inf")
    return float(np.linalg.norm(a - b) / nb)


def cmd_compare(a) -> int:
    A = np.load(a.compare[0], allow_pickle=False)
    B = np.load(a.compare[1], allow_pickle=False)
    ma = json.loads(str(A["__meta__"]))
    mb = json.loads(str(B["__meta__"]))
    print(f"  A  {os.path.relpath(a.compare[0], _ROOT)}")
    print(f"     {ma['side']}  dtype={ma['dtype']}  regen={ma['regen_mode']}")
    print(f"  B  {os.path.relpath(a.compare[1], _ROOT)}")
    print(f"     {mb['side']}  dtype={mb['dtype']}  regen={mb['regen_mode']}")
    if ma["weights_sha256"] != mb["weights_sha256"]:
        print("\n  REFUSED: the two sides did not run on the same networks.")
        print(f"    A {ma['weights_sha256'][:16]}   B {mb['weights_sha256'][:16]}")
        return 1
    print(f"  weights sha256   {ma['weights_sha256'][:16]}  IDENTICAL on both sides")
    print(f"  shape            {ma['width']} x {ma['depth']}, {ma['n_nets']} nets")
    print(f"  seed provenance  {ma['seed_note']}")

    nets = int(ma["n_nets"])
    print("\n  END TO END -- the (depth, width) means, per network, no averaging")
    print(f"    {'net':>4}{'seed':>16}{'rel drift':>14}{'final layer':>14}")
    e2e, fin = [], []
    for i in range(nets):
        r = _rel(A[f"n{i}_means"], B[f"n{i}_means"])
        rf = _rel(A[f"n{i}_means"][-1], B[f"n{i}_means"][-1])
        e2e.append(r)
        fin.append(rf)
        print(f"    {i:>4}{ma['seeds'][i]:>16,}{r:>14.3e}{rf:>14.3e}")
    print(f"    {'min':>4}{'':>16}{min(e2e):>14.3e}{min(fin):>14.3e}")
    print(f"    {'max':>4}{'':>16}{max(e2e):>14.3e}{max(fin):>14.3e}")

    print("\n  THE LADDER -- net 0, first divergence wins. `T` = same vs TRANSPOSE")
    order = [("lin", "mu_pre"), ("lin", "cov_pre"), ("lin", "s3"), ("lin", "s21"),
             ("nl", "mu"), ("nl", "cov"), ("nl", "K3"), ("nl", "K21"),
             ("nl", "S3"), ("nl", "S21"), ("nl", "M"), ("nl", "c4")]
    print(f"    {'layer':>5} {'point':>5} {'component':>10}{'rel':>13}{'rel vs T':>13}")
    rows = []
    for l in range(int(ma["depth"])):
        for tag, name in order:
            k = f"n0_l{l}_{tag}_{name}"
            if k not in A or k not in B:
                continue
            va, vb = A[k], B[k]
            r = _rel(va, vb)
            rt = _rel(va, vb.T) if va.ndim == 2 else float("nan")
            rows.append({"layer": l, "point": tag, "component": name,
                         "rel": r, "rel_vs_transpose": rt})
            flag = ""
            if r > 1e-9:
                flag = "  <-- DIVERGES"
                if np.isfinite(rt) and rt < r * 0.5:
                    flag += " (TRANSPOSED?)"
            print(f"    {l:>5} {tag:>5} {name:>10}{r:>13.3e}"
                  f"{rt:>13.3e}{flag}")
        if any(x["layer"] == l and x["rel"] > 1e-6 for x in rows) and not a.all_layers:
            print("    (stopping at the first diverging layer; --all-layers for the rest)")
            break

    ok = max(e2e) <= a.gate
    print(f"\n  G-PARITY clause 1: worst end-to-end drift {max(e2e):.3e} "
          f"vs gate {a.gate:.3e}   {'PASS' if ok else 'FAIL'}")
    print("  Clause 2 (raw MSE reproduces 8.5075e-07 within [7.0e-07, 1.1e-06])")
    print("  is scored separately against the minted truth -- agreement between")
    print("  two programs is not evidence that either is the right function.")
    if a.out:
        _dump(a.out, {
            "check": "WP16 Blocker A: fnp port vs upstream torch reference, "
                     "component ladder at two points per layer",
            "contract": "experiments/LOG.md 2026-09-03 CONTRACT WP16, HEAD 7169392",
            "A": ma, "B": mb,
            "e2e_rel_per_net": e2e, "final_layer_rel_per_net": fin,
            "e2e_worst": max(e2e), "e2e_best": min(e2e),
            "gate": a.gate, "gate_clause1_pass": bool(ok),
            "ladder_net0": rows,
            "caveat": "Clause 1 only. Two programs agreeing is not proof either "
                      "is correct; clause 2 ties the port to minted truth.",
        })
    return 0 if ok else 1


# ==========================================================================
# G-PARITY clause 2 -- the port against ALREADY-MINTED truth
# ==========================================================================

def cmd_score(a) -> int:
    """Score a `--port` dump's final-layer means against `truth_P01_wp13nets.npz`.

    Clause 1 only says the two programs agree. **Two programs can agree and both
    be the wrong function**, so clause 2 ties the port back to the number the
    write-up quotes: WP14 C0-i's `8.5075e-07`, measured on exactly these eight
    networks with two independent `2^22`-sample replicas.

    The estimator is the **cross product** `mean((p - y1)(p - y2))`, which is
    unbiased for the true MSE because the two replicas' Monte-Carlo errors are
    independent — the ordinary `mean((p - y)^2)` carries the replica's own noise
    as a positive bias. Both are printed; the cross product is the one quoted.
    This reads an existing artifact and mints nothing.
    """
    P = np.load(a.score[0], allow_pickle=False)
    mp = json.loads(str(P["__meta__"]))
    T = np.load(a.score[1], allow_pickle=False)
    if "probe" in T.files and int(T["probe"]) == 1:
        print("  REFUSED: that is a PROBE truth file, not evidence.")
        return 1
    tseeds = [int(x) for x in T["seeds"]]
    if [int(s) for s in mp["seeds"]] != tseeds:
        print("  REFUSED: the prediction dump and the truth file are not the "
              f"same networks.\n    preds {mp['seeds'][:3]}...\n    truth {tseeds[:3]}...")
        return 1
    m1, m2 = T["means_r1"], T["means_r2"]
    noise = np.asarray(T["gt_noise"], dtype=np.float64)
    print(f"  predictions  {os.path.relpath(a.score[0], _ROOT)}")
    print(f"               {mp['side']}  dtype={mp['dtype']}  regen={mp['regen_mode']}")
    print(f"  truth        {os.path.relpath(a.score[1], _ROOT)}")
    print(f"               2^{int(round(np.log2(int(T['n_samples']))))} samples "
          f"x 2 replicas, GT noise {noise.mean():.4e} per replica")
    print(f"\n    {'net':>4}{'seed':>16}{'cross':>14}{'vs r1':>14}{'vs r2':>14}")
    cross = []
    for i, s in enumerate(tseeds):
        p = np.asarray(P[f"n{i}_means"], dtype=np.float64)[-1]
        y1 = np.asarray(m1[i, -1], dtype=np.float64)
        y2 = np.asarray(m2[i, -1], dtype=np.float64)
        c = float(((p - y1) * (p - y2)).mean())
        cross.append(c)
        print(f"    {i:>4}{s:>16,}{c:>14.4e}"
              f"{float(((p - y1) ** 2).mean()):>14.4e}"
              f"{float(((p - y2) ** 2).mean()):>14.4e}")
    mean_cross = float(np.mean(cross))
    lo, hi = a.band
    ok = lo <= mean_cross <= hi
    print(f"\n    {'mean':>4}{'':>16}{mean_cross:>14.4e}")
    print(f"\n  WP14 C0-i reference (torch, f32)      {a.reference:.4e}")
    print(f"  this dump                             {mean_cross:.4e}   "
          f"ratio {mean_cross / a.reference:.3f}x")
    print(f"  GT noise floor                        {noise.mean():.4e}  "
          f"({100 * noise.mean() / mean_cross:.1f}% of the number above)")
    print(f"  the canary the grader actually scored  2.8192e-06   "
          f"ratio {2.8192e-06 / a.reference:.3f}x")
    print(f"\n  G-PARITY clause 2: [{lo:.2e}, {hi:.2e}]   "
          f"{'PASS' if ok else 'FAIL'}")
    if a.out:
        _dump(a.out, {
            "check": "WP16 G-PARITY clause 2: the port's final-layer means against "
                     "WP14 C0-i's minted truth, cross-product estimator",
            "contract": "experiments/LOG.md 2026-09-03 CONTRACT WP16, HEAD 7169392",
            "preds": mp, "truth": os.path.relpath(a.score[1], _ROOT),
            "seeds": tseeds, "mse_cross_per_net": cross,
            "mse_cross_mean": mean_cross,
            "reference_wp14_c0i": a.reference,
            "band": [lo, hi], "pass": bool(ok),
            "gt_noise_mean": float(noise.mean()),
            "graded_canary_raw_mse": 2.8192e-06,
            "caveat": "DESIGN tier, and CORRELATED with WP13 E2 and WP14 C0-i -- "
                      "the same eight networks. Not an independent confirmation.",
        })
    return 0 if ok else 1


def _dump(path, doc):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print(f"  wrote {os.path.relpath(path, _ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", action="store_true")
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--compare", nargs=2, default=None)
    ap.add_argument("--score", nargs=2, default=None,
                    metavar=("PORT_NPZ", "TRUTH_NPZ"))
    ap.add_argument("--band", nargs=2, type=float, default=(7.0e-07, 1.1e-06))
    ap.add_argument("--reference", type=float, default=8.5075e-07)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--nets", type=int, default=8)
    ap.add_argument("--dtype", choices=("f64", "f32"), default="f64")
    ap.add_argument("--regen-mode", choices=("sub", "nosub"), default="sub")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--full-layers", default="0,1,2")
    ap.add_argument("--all-layers", action="store_true")
    ap.add_argument("--gate", type=float, default=0.01)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    print("=" * 78)
    print("WP16 / Blocker A  --  port vs upstream at non-zero kappa_3")
    print("=" * 78)
    if a.compare:
        return cmd_compare(a)
    if a.score:
        return cmd_score(a)

    full = {int(x) for x in a.full_layers.split(",") if x.strip() != ""}
    seeds, note = _seeds(a.nets, a.width, a.depth)
    sha = _weights_sha(_mint(a.width, a.depth, seeds))
    if a.port:
        blob, meta = run_port(a.width, a.depth, seeds, dtype_name=a.dtype,
                              regen_mode=a.regen_mode, full_layers=full)
    elif a.ref:
        blob, meta = run_ref(a.width, a.depth, seeds, dtype_name=a.dtype,
                             device=a.device, full_layers=full)
    else:
        ap.error("one of --port / --ref / --compare")
    meta.update({"width": a.width, "depth": a.depth, "n_nets": a.nets,
                 "seeds": [int(s) for s in seeds], "seed_note": note,
                 "weights_sha256": sha, "full_layers": sorted(full)})
    blob["__meta__"] = np.array(json.dumps(meta))
    if not a.out:
        ap.error("--out is required for --port / --ref")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **blob)
    print(f"  {meta['side']}: {a.nets} nets, {a.width} x {a.depth}, {a.dtype}")
    print(f"  seeds     {note}")
    print(f"  weights   sha256 {sha[:16]}")
    print(f"  wrote     {os.path.relpath(a.out, _ROOT)}  ({len(blob) - 1} arrays)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
