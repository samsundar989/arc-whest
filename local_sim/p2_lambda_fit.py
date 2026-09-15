"""Our own κ4 regeneration λ, fitted from the unfactored AUGMENT chain.

Slice 6a step 1, `docs/CLAUDE_CODE_HANDOFF_2026-09-14.md` §2.1.

WHAT λ IS
---------
504aldo's V17 regenerates the fourth cumulant each layer as

    G = diag(g) + λ_l · C_off

with a 16-entry λ table fitted by least squares of the **dense augmented K3
chain's** off-diagonal κ4 core onto `C_pre_off`. That mechanism is reported to
halve their raw MSE. We do not adopt their fitted numbers — they were fitted on
eight public MLPs, and our validation policy fits on fresh mints only — so we
have to fit our own. This module is that fit.

For each layer `l`, on the pre-activation state of layer `l+1`:

    λ_l = ⟨G_off, C_off⟩ / ‖C_off‖²          (zero-intercept least squares)
    R²_l = cos²(G_off, C_off)                 (that LS fit's own R²)

The indexing follows 504aldo's: their model is `G_pre_off(l+1) = LAM[l] ·
C_pre_off(l+1)`, so `λ_l` is read at `pre{l+1}` and indexed by `l`.

THE THING THIS MODULE FOUND, WHICH THE PLAN DID NOT ANTICIPATE
---------------------------------------------------------------
The handoff says to run this at **1024×16**. That is not possible, and the
reason is a hard wall rather than a tuning problem. Unfactored AUGMENT at
`k_max=3` keeps the third cumulant as a **dense (n,n,n) tensor**:

    get_d_max(3, AUGMENT) = 4,  r_x = {1:0, 2:0, 3:0, 4:1}
      degree 3 at r=0 -> 3 free indices -> (n, n, n)
      degree 4 at r=1 -> 2 free indices -> (n, n)     <- the core we want

MEASURED peak VRAM on an RTX 5090 (32 GiB), float32, depth 16:

    n =  64      1.61 GiB
    n = 128     12.44 GiB          growth 7.73x, against the n^3 law's 8.00x
    n = 192     OUT OF MEMORY      the n^3 law predicts 42.0 GiB

so the law is confirmed by the failure that tests it. Extrapolated:
`n=256` 99.5 GiB, `n=512` 796 GiB, **`n=1024` 6,369 GiB**. The largest width
this card can hold is about **175**.

So the fit cannot be performed at the live width, and the scientific question
becomes whether it NEEDS to be: **is λ_l width-stable?** `--widths` runs the
fit at several widths so that question is answered with a measurement instead
of an assumption. If λ_l converges in width, a fit at 128 transfers to 1024 and
the wall costs us nothing. If it does not, the AUGMENT route to λ is closed at
the live shape and a different estimator of the same quantity is needed.

LABEL-FREE
----------
No truth is read, no public data is touched, and no DECIDE block is spent.
This fit consumes DESIGN networks only, claimed through `seed_ledger` in the
live repository before any queued job starts — a claim written inside a job
snapshot is discarded with the snapshot while the seeds are really drawn.

Run:
    .venv-gpu/Scripts/python.exe local_sim/p2_lambda_fit.py \\
        --widths 64,96,128 --depth 16 --n-mlps 8 --seed-start <claimed> \\
        --out experiments/<slice>/lambda_fit.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

#: The 504aldo per-layer λ means, as recorded in our own comparison artifact on
#: 2026-09-14. Read from OUR file, not from their repo: this is a comparison
#: target, and it is never used as a fitted constant in any candidate.
ALDO_CSV = os.path.join(_ROOT, "experiments", "2026-09-14_exp_lambda",
                        "lam_compare_summary.csv")


def aldo_lambda():
    """`(li, mean)` from the stored comparison. Returns None if absent."""
    if not os.path.exists(ALDO_CSV):
        return None
    rows = np.genfromtxt(ALDO_CSV, delimiter=",", names=True, comments="#")
    names = rows.dtype.names
    li = rows[names[0]]
    mean = rows[names[1]]
    return np.asarray(li, dtype=int), np.asarray(mean, dtype=np.float64)


def _off(a):
    """The off-diagonal part, as a flat vector."""
    n = a.shape[0]
    m = ~np.eye(n, dtype=bool)
    return a[m]


def fit_one(weights, *, k_max=3, device="cuda", dtype=None):
    """Per-layer λ and its diagnostics for ONE network.

    Returns a dict of `(depth-1,)` arrays: `lam`, `r2`, `cos`, plus the
    observables the adaptive rule would need (`mean_var`, `mean_dg`) and the
    norms, so a later reader can tell a small λ from a small signal.
    """
    import torch

    import p2_augment as pa
    pa._ensure_upstream()
    dtype = dtype or torch.float32
    depth = len(weights)
    out = pa.run_augment([np.asarray(x) for x in weights], k_max=k_max,
                         kind="AUGMENT", factor=False, device=device,
                         dtype=dtype)

    lam, r2, cos, mv, mg, ng, nc = [], [], [], [], [], [], []
    for l in range(depth - 1):
        key = "pre%d" % (l + 1)
        st = out.get(key)
        if st is None or 4 not in st or 2 not in st:
            break
        G = getattr(st[4], "core", st[4]).detach().to(torch.float64).cpu().numpy()
        C = getattr(st[2], "core", st[2]).detach().to(torch.float64).cpu().numpy()
        # Symmetrise: the core is symmetric in exact arithmetic and the fit
        # should not be sensitive to which triangle a backend happened to fill.
        G = 0.5 * (G + G.T)
        C = 0.5 * (C + C.T)
        go, co = _off(G), _off(C)
        denom = float(co @ co)
        num = float(go @ co)
        l_hat = num / denom if denom > 0 else float("nan")
        gg = float(go @ go)
        lam.append(l_hat)
        # Zero-intercept LS R^2 is exactly the squared cosine.
        c = num / np.sqrt(gg * denom) if gg > 0 and denom > 0 else float("nan")
        cos.append(c)
        r2.append(c * c)
        mv.append(float(np.mean(np.diag(C))))
        mg.append(float(np.mean(np.diag(G))))
        ng.append(float(np.sqrt(gg)))
        nc.append(float(np.sqrt(denom)))
    del out
    try:
        import torch as _t
        _t.cuda.empty_cache()
    except Exception:
        pass
    return {"lam": np.asarray(lam), "r2": np.asarray(r2),
            "cos": np.asarray(cos), "mean_var": np.asarray(mv),
            "mean_g": np.asarray(mg), "norm_G_off": np.asarray(ng),
            "norm_C_off": np.asarray(nc)}


def fit_many(width, depth, seeds, *, device="cuda", dtype=None, k_max=3,
             progress=True):
    from mint_provenance import canonical_mint

    per = []
    t0 = time.perf_counter()
    for i, s in enumerate(seeds):
        w, _e, _s = canonical_mint(width, depth, int(s))
        per.append(fit_one(w, k_max=k_max, device=device, dtype=dtype))
        if progress:
            print(f"    n={width} net {i + 1}/{len(seeds)} seed {s} "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)
    keys = per[0].keys()
    stack = {k: np.stack([p[k] for p in per], axis=0) for k in keys}
    return stack


def summarise(stack):
    lam = stack["lam"]
    return {"lam_mean": lam.mean(axis=0), "lam_sd": lam.std(axis=0, ddof=1)
            if lam.shape[0] > 1 else np.zeros(lam.shape[1]),
            "r2_mean": stack["r2"].mean(axis=0),
            "cos_mean": stack["cos"].mean(axis=0),
            "mean_var": stack["mean_var"].mean(axis=0),
            "n_nets": int(lam.shape[0])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--widths", default="64,96,128",
                    help="comma-separated. The dense AUGMENT state is O(n^3); "
                         "this card OOMs above about 175.")
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--n-mlps", type=int, default=8)
    ap.add_argument("--seed-start", type=int, required=True,
                    help="first seed of the CLAIMED design range. Required: a "
                         "literal seed that no ledger recorded is how a block "
                         "gets drawn dry while the registry calls it free.")
    ap.add_argument("--block", default="",
                    help="the block those seeds came from, for the record")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    import torch
    dtype = getattr(torch, a.dtype)
    widths = [int(x) for x in a.widths.split(",") if x]
    seeds = [a.seed_start + i for i in range(a.n_mlps)]

    print("=" * 96)
    print("OUR OWN kappa-4 REGENERATION LAMBDA, from the unfactored AUGMENT chain")
    print("=" * 96)
    print(f"  widths   {widths}   depth {a.depth}   {a.n_mlps} nets   {a.dtype}")
    print(f"  seeds    {seeds[0]}..{seeds[-1]}"
          + (f"   block {a.block}" if a.block else "   *** NO BLOCK RECORDED ***"))
    print("  LABEL-FREE: no truth, no public split, no DECIDE block.")
    print()

    aldo = aldo_lambda()
    results = {}
    for w in widths:
        print(f"  -- width {w} " + "-" * 60)
        try:
            stack = fit_many(w, a.depth, seeds, device=a.device, dtype=dtype)
        except Exception as exc:                       # noqa: BLE001
            print(f"    REFUSED at n={w}: {type(exc).__name__}: "
                  f"{str(exc)[:120]}")
            print("    The dense AUGMENT state is O(n^3). This is the wall, "
                  "not a tuning problem.")
            results[w] = {"failed": f"{type(exc).__name__}: {str(exc)[:200]}"}
            break
        results[w] = summarise(stack)
        results[w]["_raw"] = stack

    print()
    print("  l    " + "".join(f"{('n=%d' % w):>14}" for w in results
                              if "failed" not in results[w])
          + ("        504aldo" if aldo is not None else ""))
    ws = [w for w in widths if w in results and "failed" not in results[w]]
    if ws:
        L = min(len(results[w]["lam_mean"]) for w in ws)
        for l in range(L):
            line = f"  {l:<4d} "
            for w in ws:
                line += f"{results[w]['lam_mean'][l]:>14.6e}"
            if aldo is not None and l < len(aldo[1]):
                line += f"{aldo[1][l]:>15.6e}"
            print(line)
        print()
        print("  R^2 of that fit (cos^2), same layout")
        for l in range(L):
            line = f"  {l:<4d} "
            for w in ws:
                line += f"{results[w]['r2_mean'][l]:>14.4f}"
            print(line)

        if len(ws) >= 2:
            print()
            print("  WIDTH STABILITY -- the question the memory wall forces")
            a_w, b_w = ws[-2], ws[-1]
            la, lb = results[a_w]["lam_mean"][:L], results[b_w]["lam_mean"][:L]
            rel = np.abs(lb - la) / np.maximum(np.abs(la), 1e-30)
            print(f"    n={a_w} -> n={b_w}: median |rel change| "
                  f"{np.median(rel):.4f}, worst {np.max(rel):.4f}")
            print("    A fit at a feasible width transfers to 1024 only if this")
            print("    is small AND still shrinking as width grows. One pair of")
            print("    widths cannot establish that; it can only refute it.")
            if aldo is not None:
                al = aldo[1][:L]
                r = lb / np.where(np.abs(al) > 0, al, np.nan)
                print(f"    ours / 504aldo at n={b_w}: median {np.nanmedian(r):.4f}, "
                      f"range [{np.nanmin(r):.4f}, {np.nanmax(r):.4f}]")

    if a.out:
        d = os.path.dirname(os.path.abspath(a.out))
        if d:
            os.makedirs(d, exist_ok=True)
        payload = {}
        for w, r in results.items():
            if "failed" in r:
                continue
            for k, v in r.items():
                if k == "_raw":
                    for kk, vv in v.items():
                        payload[f"n{w}_raw_{kk}"] = vv
                else:
                    payload[f"n{w}_{k}"] = np.asarray(v)
        payload["widths"] = np.asarray(ws)
        payload["seeds"] = np.asarray(seeds, dtype=np.int64)
        payload["depth"] = np.int64(a.depth)
        np.savez_compressed(a.out, **payload)
        meta = {"widths_requested": widths, "widths_completed": ws,
                "depth": a.depth, "n_mlps": a.n_mlps, "seeds": seeds,
                "block": a.block, "dtype": a.dtype, "k_max": 3,
                "kind": "AUGMENT", "factor": False,
                "label_free": True,
                "failures": {str(w): r["failed"] for w, r in results.items()
                             if "failed" in r}}
        with open(os.path.splitext(a.out)[0] + "_meta.json", "w",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(meta, indent=2) + "\n")
        print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
