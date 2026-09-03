"""WP18 § A: what the canary's GUARDS cost, and what they change.

Contract: `experiments/LOG.md`, 2026-09-03, "CONTRACT WP18" § A, committed at
`49e2e42`. This is the re-metering half of the strategist's process rule --
*"every reassociating change gets WP16's parity gate and a re-metered price"* --
applied to a change that is not a reassociation but a set of range guards.

WHAT IT ANSWERS, IN THE ORDER THE DECISION NEEDS THEM
-----------------------------------------------------
`--pair A B`   Two estimator files, metered inside a real `BudgetContext` at the
               live shape, on the same canonically-minted network:

               * counted f32 FLOPs and the multiplier `max(0.1, F/B)` for each,
                 with and without the grading server's unexplained `+2**24`;
               * flopscope OPERATIONS for each -- the binding axis under
                 `lambda = 0`, because operations are what the `0.4 s` residual
                 cliff is paid in;
               * the worst absolute and relative difference between the two
                 predictions, per layer and on the SCORED final layer alone;
               * whether the FLOP total is the same for four different networks
                 in ONE process, which is `certify_candidate` § 6's check and the
                 thing the constants cache was silently breaking.

`--alpha-range` The worst `|mu / sigma|` the flow actually reaches, over minted
               networks at the live shape, measured from the LAB flow rather
               than from the candidate -- so it is an independent statement
               about the quantity `ALPHA_CAP` clamps, not a self-report. This is
               what licenses "the cap is never approached on He-init networks",
               or refuses it.

WHY THE DIFFERENCE COLUMN IS NOT A PARITY GATE ON ITS OWN
---------------------------------------------------------
The guards are DESIGNED to change values: `maximum(out, 0)` moves every negative
entry to zero. So "bit-identical" is the wrong test and passing it would mean the
guard is not wired. The right test is the one this prints: **every difference is
a clamped negative or a substituted non-finite, and the scored final layer is
unchanged except at entries that were negative.** A difference anywhere else is a
defect, and the per-layer table is what would show it.

`# seed-ledger: no block named` -- `canonical_mint` is called on fixed seeds
outside every registered block for a determinism and price check. No block is
drawn, no truth is read, and nothing is selected.

Run:
    uv run python local_sim/p2_wp18_price.py --pair estimator_wp15_k3_canary.py estimator_wp18_canary.py
    uv run python local_sim/p2_wp18_price.py --alpha-range --nets 8
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# local_sim ships numpy shims NAMED flopscope/whestbench: APPEND, never insert.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

WIDTH, DEPTH = 1024, 16
BUDGET = 2 ** 41
FLOOR = 0.1
#: The grading server bills an unexplained `+2**24` (`+0.0122%`) that no local
#: ledger reproduces. `docs/GOTCHAS.md`. Included as a separate column, never
#: folded into the measured figure.
SERVER_OFFSET = 2 ** 24
#: Seeds OUTSIDE every registered block: this module measures price and range,
#: never accuracy against truth, so it claims nothing.
#: seed-ledger: no block named
SEEDS = (90_000_000_401, 90_000_000_402, 90_000_000_403, 90_000_000_404)


def _load(path):
    name = "cand_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mint(seed, width, depth):
    from mint_provenance import canonical_mint

    w, _e, _s = canonical_mint(width, depth, seed)
    return [np.ascontiguousarray(np.asarray(x, dtype=np.float32)) for x in w]


class _Ctx:
    """The minimum `SetupContext` the candidates read: `seed` and two paths."""

    def __init__(self, seed, tmp):
        self.seed = seed
        self.scratch_dir = tmp
        self.submission_dir = _ROOT


def _run(mod, weights, seed, *, budget=BUDGET, est=None):
    """One metered `setup` + `predict`, inside a real `BudgetContext`.

    `est` is passed in ON PURPOSE. The grader constructs ONE `Estimator` and
    calls `predict` on every MLP of the split, so any state cached on the
    instance is paid for by the FIRST MLP and by none of the others. A probe
    that builds a fresh instance per seed cannot see that, and the first version
    of this module did exactly that and reported the constants cache as
    network-INDEPENDENT -- the opposite of what `certify_candidate` measured with
    a shared instance. The check was vacuous and the run that exposed it is in
    the LOG.
    """
    import flopscope
    import whestbench

    if est is None:
        est = mod.Estimator()
        est.setup(_Ctx(seed, _ROOT))
    mlp = whestbench.MLP(weights=weights, width=weights[0].shape[0],
                         depth=len(weights), seed=seed)
    with flopscope.BudgetContext(budget) as flops:
        out = est.predict(mlp, budget)
        used = int(flops.flops_used)
        # The op ledger the way `certify_candidate` reads it. NEVER
        # `budget_summary_dict()` inside `predict` -- hard rule 4, a 157x
        # residual blowup. This is outside it.
        ns = (flops.summary_dict(by_namespace=True).get("by_namespace") or {})
        n_ops = int(sum(float(v.get("calls", 0.0)) for v in ns.values()))
    return np.asarray(out, dtype=np.float64), used, n_ops


def cmd_pair(a) -> int:
    import flopscope

    rows = []
    print(f"  shape {a.width}x{a.depth}   B = {BUDGET:,}   floor {FLOOR}\n")
    preds, prices = {}, {}
    for path in (a.pair[0], a.pair[1]):
        mod = _load(path)
        per_seed = []
        first_pred = None
        # ONE instance, set up ONCE, exactly as the grader does -- and `setup`
        # OUTSIDE the metered region, because the grader gives it its own 5 s
        # window. Calling `setup` per seed is what made the first version of
        # this check vacuous: the old canary's `setup` did `self._const = {}`,
        # so re-running it cleared the very cache the check exists to detect.
        est = mod.Estimator()
        est.setup(_Ctx(SEEDS[0], _ROOT))
        for s in SEEDS:
            w = _mint(s, a.width, a.depth)
            out, used, n_ops = _run(mod, w, s, budget=a.budget, est=est)
            per_seed.append({"seed": s, "flops": used, "n_ops": n_ops})
            if first_pred is None:
                first_pred = out
        preds[path] = first_pred
        flops0 = per_seed[0]["flops"]
        indep = len({r["flops"] for r in per_seed}) == 1
        # THE STEADY-STATE ROW, and it is the one a per-MLP price must be quoted
        # against. `SKEPTIC WP18` caught the first version of this packet
        # comparing the guarded file to the baseline's FIRST MLP -- the single
        # most expensive row the baseline ever produces, because MLP 1 pays for
        # the constants cache and MLP 2..100 do not. Against the row that 99 of
        # 100 graded MLPs actually see, the price is larger. The modal row is
        # the steady state; for an order-independent file it equals row 0.
        _fv = [r["flops"] for r in per_seed]
        _ov = [r["n_ops"] for r in per_seed]
        steady = max(set(_fv), key=_fv.count)
        steady_ops = max(set(_ov), key=_ov.count)
        mult = max(FLOOR, flops0 / a.budget)
        mult_off = max(FLOOR, (flops0 + SERVER_OFFSET) / a.budget)
        prices[path] = {"flops": flops0, "n_ops": per_seed[0]["n_ops"],
                        "flop_independent": indep,
                        "per_seed": per_seed,
                        "flops_steady_state": steady,
                        "n_ops_steady_state": steady_ops,
                        "x": flops0 / a.budget,
                        "x_with_server_offset": (flops0 + SERVER_OFFSET) / a.budget,
                        "multiplier": mult,
                        "multiplier_with_server_offset": mult_off}
        print(f"  {path}")
        print(f"    counted f32 FLOPs           {flops0:,}")
        print(f"    x = F/B                     {flops0 / a.budget:.7f}"
              f"   multiplier {mult:.6f}"
              f"{'  (PINNED at the floor)' if flops0 / a.budget < FLOOR else ''}")
        print(f"    x with the +2**24 offset    "
              f"{(flops0 + SERVER_OFFSET) / a.budget:.7f}"
              f"   multiplier {mult_off:.6f}")
        print(f"    flopscope OPERATIONS        {per_seed[0]['n_ops']:,}"
              f"   <- the binding axis under lambda = 0")
        spread = "YES" if indep else "NO -- " + ", ".join(
            f"{r['flops']:,}" for r in per_seed)
        print(f"    FLOPs equal on {len(SEEDS)} networks in ONE process   {spread}")
        print(f"    STEADY-STATE FLOPs / operations   {steady:,} / {steady_ops:,}"
              f"{'   <- the row 99 of 100 graded MLPs see' if not indep else ''}")
        print()

    A, B = a.pair[0], a.pair[1]
    pa, pb = preds[A], preds[B]
    d = pb - pa
    scale = max(float(np.abs(pa).max()), 1e-300)
    neg_a = pa < 0.0
    print(f"  {B}  vs  {A}   on seed {SEEDS[0]}")
    print(f"    {'layer':>6}{'worst |d|':>14}{'rel':>12}{'n changed':>11}"
          f"{'n were < 0':>12}")
    per_layer = []
    for l in range(pa.shape[0]):
        chg = int((d[l] != 0.0).sum())
        per_layer.append({"layer": l, "worst_abs": float(np.abs(d[l]).max()),
                          "rel": float(np.abs(d[l]).max() / scale),
                          "n_changed": chg,
                          "n_negative_in_A": int(neg_a[l].sum())})
        print(f"    {l:>6}{np.abs(d[l]).max():>14.3e}"
              f"{np.abs(d[l]).max() / scale:>12.3e}{chg:>11}"
              f"{int(neg_a[l].sum()):>12}")
    # ROW 0 IS CHECKED AGAINST THE CLOSED FORM, NOT AGAINST THE BASELINE.
    # The candidate replaces row 0 with `||W[:, j]|| / sqrt(2 pi)` in float64, so
    # every one of its `n` entries differs from the baseline BY DESIGN and the
    # negative-or-non-finite predicate cannot explain them -- which it did not,
    # and it said so with `1,024` unexplained changes the first time this ran.
    # That is the predicate doing its job on an intentional change, so the
    # intentional change gets its OWN check rather than a widened predicate.
    w0 = _mint(SEEDS[0], a.width, a.depth)[0].astype(np.float64)
    exact0 = np.sqrt((w0 * w0).sum(axis=0)) / np.sqrt(2.0 * np.pi)
    sc0 = max(float(np.abs(exact0).max()), 1e-300)
    rel_A0 = float(np.abs(pa[0] - exact0).max()) / sc0
    rel_B0 = float(np.abs(pb[0] - exact0).max()) / sc0
    row0_ok = (rel_B0 < 1e-12) and (rel_B0 < rel_A0)
    print()
    print("    row 0 against the exact closed form ||W[:,j]|| / sqrt(2 pi)")
    print(f"      baseline  rel {rel_A0:.3e}")
    print(f"      candidate rel {rel_B0:.3e}   "
          f"{'STRICTLY BETTER and inside 1e-12' if row0_ok else 'NOT better'}")

    # THE claim, for rows 1.. : every change is a clamped negative or a
    # substituted non-finite.
    changed = (d != 0.0)
    changed[0] = False
    explained = changed & (neg_a | ~np.isfinite(pa))
    unexplained = int((changed & ~explained).sum())
    print()
    print(f"    rows 1..{pa.shape[0] - 1}:")
    print(f"      entries changed                        {int(changed.sum()):,}")
    print(f"      of which were negative or non-finite   {int(explained.sum()):,}")
    print(f"      UNEXPLAINED changes                    {unexplained:,}"
          f"   {'<- must be 0' if unexplained else '(none)'}")
    fl = pa.shape[0] - 1
    print(f"    scored final layer: worst |d| {np.abs(d[fl]).max():.3e}, "
          f"{int((d[fl] != 0.0).sum())} of {pa.shape[1]} entries changed, "
          f"all of them negative in {A}: "
          f"{bool(np.all(neg_a[fl][d[fl] != 0.0])) if (d[fl] != 0.0).any() else True}")

    dops = prices[B]["n_ops_steady_state"] - prices[A]["n_ops_steady_state"]
    dfl = prices[B]["flops_steady_state"] - prices[A]["flops_steady_state"]
    print()
    print("  PRICE, STEADY STATE against STEADY STATE -- the honest comparison")
    print(f"    operations  {prices[A]['n_ops_steady_state']:,} -> "
          f"{prices[B]['n_ops_steady_state']:,}   {dops:+,}   "
          f"({dops / max(prices[A]['n_ops_steady_state'], 1) * 100:+.2f}%)")
    print(f"    f32 FLOPs   {prices[A]['flops_steady_state']:,} -> "
          f"{prices[B]['flops_steady_state']:,}   {dfl:+,}   "
          f"({dfl / max(prices[A]['flops_steady_state'], 1) * 100:+.5f}%)")

    ok = (unexplained == 0) and prices[B]["flop_independent"] and row0_ok
    print()
    print(f"  -> {'PASS' if ok else 'FAIL'}   (row 0 exact; no unexplained "
          f"change in rows 1.., AT THIS SHAPE; B order-independent)")
    print("     SCOPE: 'no unexplained change' is a claim about THIS shape. The")
    print("     alpha cap binds where the flow is ALREADY numerically broken --")
    print("     SKEPTIC WP18 measured 311 unexplained changes at 64x32, where")
    print("     the uncapped |alpha| reaches 2.4e14. That is the guard working,")
    print("     and it is a LOSS there rather than a no-op.")
    if a.out:
        _dump(a.out, {"check": "WP18 § A: the guarded canary's price and its "
                               "value difference from the graded file",
                      "contract": "experiments/LOG.md 2026-09-03 CONTRACT WP18 "
                                  "§ A, HEAD 49e2e42",
                      "width": a.width, "depth": a.depth, "budget": a.budget,
                      "server_offset": SERVER_OFFSET,
                      "baseline": A, "candidate": B,
                      "price": prices, "per_layer": per_layer,
                      "steady_state_delta_ops": int(dops),
                      "steady_state_delta_flops": int(dfl),
                      "row0_rel_baseline": rel_A0,
                      "row0_rel_candidate": rel_B0,
                      "row0_exact_pass": bool(row0_ok),
                      "scope_caveat": "unexplained == 0 is a claim about THIS shape only. The alpha cap binds where the flow is already numerically broken; SKEPTIC WP18 measured 311 unexplained changes at 64x32.",
                      "entries_changed": int(changed.sum()),
                      "entries_explained": int(explained.sum()),
                      "unexplained": unexplained,
                      "pass": bool(ok)})
    return 0 if ok else 1


def cmd_alpha_range(a) -> int:
    """The worst `|mu / sigma|` the flow reaches, read from the UNCAPPED file.

    `ALPHA_CAP` is a claim about a quantity, and the honest way to check a clamp
    is to measure the quantity where the clamp does not exist. So this runs
    `estimator_wp15_k3_canary.py` -- the file the grader scored, which has no
    cap -- and wraps its `_wick`, whose third argument IS `alpha`. Nothing about
    the capped file is consulted except the cap's value.
    """
    base = _load(a.baseline)
    seen = []
    real_wick = base._wick

    def spy(mean, sigma, alpha, table, dtype, const):
        arr = np.abs(np.asarray(alpha, dtype=np.float64))
        seen.append(float(arr.max()))
        return real_wick(mean, sigma, alpha, table, dtype, const)

    base._wick = spy
    rows = []
    est = base.Estimator()
    est.setup(_Ctx(SEEDS[0], _ROOT))
    for i in range(a.nets):
        seed = 90_000_000_500 + i          # seed-ledger: no block named
        seen.clear()
        w = _mint(seed, a.width, a.depth)
        _run(base, w, seed, budget=a.budget, est=est)
        rows.append({"seed": seed, "n_layers_seen": len(seen),
                     "worst_abs_alpha": max(seen) if seen else None})
        print(f"    seed {seed}  {len(seen)} layers  "
              f"worst |alpha| {rows[-1]['worst_abs_alpha']:.4f}")
    base._wick = real_wick

    top = max(r["worst_abs_alpha"] for r in rows)
    cand = _load(a.candidate)
    cap = float(getattr(cand, "ALPHA_CAP", float("nan")))
    print()
    print(f"  worst |alpha| over {len(rows)} networks at {a.width}x{a.depth}"
          f"   {top:.4f}")
    print(f"  ALPHA_CAP in {a.candidate}   {cap:.1f}")
    print(f"  margin   {cap / max(top, 1e-300):.1f}x"
          f"   -> the cap is {'NEVER approached' if cap > 4 * top else 'CLOSE'} "
          f"on He-init networks at the live shape")
    if a.out:
        _dump(a.out, {"check": "WP18 § A: the worst |mu/sigma| the UNCAPPED "
                               "flow reaches at the live shape",
                      "contract": "experiments/LOG.md 2026-09-03 CONTRACT WP18 "
                                  "§ A, HEAD 49e2e42",
                      "measured_on": a.baseline, "cap_from": a.candidate,
                      "width": a.width, "depth": a.depth, "rows": rows,
                      "worst_abs_alpha": top, "alpha_cap": cap,
                      "margin": cap / max(top, 1e-300)})
    return 0


def _dump(path, doc):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print(f"  wrote {os.path.relpath(path, _ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, metavar=("BASELINE", "CANDIDATE"))
    ap.add_argument("--alpha-range", action="store_true")
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--nets", type=int, default=8)
    ap.add_argument("--baseline", default="estimator_wp15_k3_canary.py")
    ap.add_argument("--candidate", default="estimator_wp18_canary.py")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    print("=" * 78)
    print("WP18 / § A  --  the guards' price, and what they change")
    print("=" * 78)
    if a.pair:
        return cmd_pair(a)
    if a.alpha_range:
        return cmd_alpha_range(a)
    ap.error("one of --pair / --alpha-range")


if __name__ == "__main__":
    raise SystemExit(main())
