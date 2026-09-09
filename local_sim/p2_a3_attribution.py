"""A3 — does κ₄ carry D1's `2.094x`, or does the raised diagram budget?

Contract: `experiments/LOG.md`, *ERRATUM ON THE D1 ENTRY, AND CONTRACT (Rule 1)
FOR ARM A3* (2026-09-09), committed at `2e4e9c2` **before this file existed**.

## Why this exists

D1 measured `kind=SIMPLE -> kind=AUGMENT` as `4.0047e-08 -> 1.9123e-08`, and the
entry claimed the gain was "attributable to the κ₄ representation and to nothing
else" because `kind` is a single flag. **One flag is not one mechanism.**

    kprop_harmonic.py:59      budget = k_max - 1 + augment

sits inside `get_vec_cond`, which gates every vec-partition at **every** degree.
Measured on the pinned upstream, `get_all_terms(3, d_max=4)` returns **40** terms
at `augment=False` and **142** at `augment=True`; `(1,1,1)` alone grows `4 -> 31`.
And of the four legs `augment` adds to the factored `(1,1,1)` block
(`factor_k3.py:389-399`), **three contain no κ₄**: `WK_21` twice (a κ₃ `(2,1)`
slice) and `WK_11 * WK_11 / 2` (a doubled κ₂ edge).

This is not pedantry about wording. Step 2's memory precondition — the dense `n³`
`(2,1,1)` slice at *exactly* hard rule 1's `4 GiB` single-array cap — is justified
by the κ₄ attribution. If the κ₂/κ₃ legs carry the gain, a port could raise the
budget **without** carrying κ₄'s `(n,n)` object and never form that array.

## The arm

**A3 = `kind=AUGMENT` with `WK[4]` forced to ZERO at every nonlinear output.**

It keeps the raised budget and removes every κ₄ *information* surface. It is an
ablation, not a deployable estimator, and it does **not** reduce memory — the
slice is still formed, only its content is zeroed.

The injection uses the seam built and proven for D2
(`p2_d2_k4_oracle.kprop_means_injected`, variant `"zero"`): upstream is never
modified, `verify_upstream` still passes, and `test_p2_d2_seam.py` shows the
no-op arm is bit-identical while `3*T` and `T=0` move the chain — so the seam is
known to be live, not merely wired.

## The split, fixed in the contract BEFORE this file existed

    f = (xmse_A3 - xmse_AUGMENT) / (xmse_SIMPLE - xmse_AUGMENT)

the fraction of the paired gain κ₄ carries. `f = 1` all of it, `f = 0` none.

    f >= 0.7   kappa_4 carries the gain -> the (2,1,1) memory work is EARNED
    f <= 0.3   the raised BUDGET carries it -> the memory work is aimed at the
               wrong object; try budget-only first
    else       both matter

Pre-registered prediction: **`f` in `[0.5, 0.8]`**.

`f` is a descriptive three-way split, not a promotion gate, so it is reported as
a ratio of the paired mean differences (the competition's own arithmetic
identity) **with both underlying paired differences and their studentised-t
intervals printed beside it** — those are the binding statistics. A per-net `f`
distribution is reported as a stability diagnostic only.

## Two controls the Skeptic asked for, both nearly free

1. **Reversed arm order** on the first `--reverse-nets` networks. SIMPLE ran
   first in every run on disk; this is the last untested control. Bit-equality is
   asserted against the same-net forward-order rows.
2. **All 16 layers' xmse in every arm.** `gt_halfA["means"]` is `(24, 16, 1024)` —
   the labels already exist and D1 discarded 15/16 of them. The leaders' published
   all-layers (`7-11e-9`) against final-layer (`1.9-2.8e-8`) is a far harder
   fingerprint than the final-layer number alone.

Also counts upstream's *"Snapping negative variance to zero"* warnings, which
fire freely at `64x5` and did not fire at all in job `411` — currently unrecorded.

Research only. Never imported by `estimator.py`.
`# seed-ledger: no block named` — same P01 DESIGN asset, same 24 nets.

Run (GPU venv, via workerq, ABSOLUTE interpreter and --out; see docs/GOTCHAS.md):
    <abs>/.venv-gpu/Scripts/python.exe -u local_sim/p2_a3_attribution.py \\
        --out <abs>/experiments/2026-09-08_slice3/a3_attribution.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

# Pre-registered at 2e4e9c2, before any A3 datum. Do not edit.
F_KAPPA4_EARNED = 0.7      # f >= this  -> the (2,1,1) memory work is earned
F_BUDGET_CARRIES = 0.3     # f <= this  -> aimed at the wrong object
F_PREDICTION = (0.5, 0.8)


def xmse(pred, A, B):
    """Unbiased two-half cross estimator. E[(p-A)(p-B)] = MSE(p) exactly."""
    return float(np.mean((pred - A) * (pred - B)))


def studentised_t(d, alpha=0.05):
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    try:
        from scipy import stats
        crit = float(stats.t.ppf(1 - alpha / 2, n - 1))
    except Exception:
        crit = 1.96 if n > 60 else 2.064
    return m, se, (m - crit * se, m + crit * se)


def _progress(frac):
    p = os.environ.get("WORKERQ_PROGRESS")
    if not p:
        return
    try:
        with open(p, "w") as fh:
            fh.write(str(min(max(frac, 0.0), 1.0)))
    except Exception:
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="experiments/2026-08-26_p2_gt")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reverse-nets", type=int, default=3,
                    help="run the arms in REVERSED order on this many networks "
                         "and assert bit-equality with the forward-order rows")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    import torch
    import p2_kprop_bridge as bridge
    import p2_d2_k4_oracle as D2
    from mint_provenance import canonical_mint

    v = bridge.verify_upstream(bridge._src())
    if not v["pass"]:
        raise SystemExit(f"REFUSED: upstream is not the pinned rev + patch: {v}")
    from mlp_kprop.kprop_harmonic import SIMPLE, AUGMENT, get_all_terms

    # The premise of this whole slice, checked rather than quoted from the entry.
    # get_all_terms returns a LIST of terms (verified 2026-09-09: 40 -> 142).
    n_simple = len(get_all_terms(3, d_max=4, augment=False))
    n_augment = len(get_all_terms(3, d_max=4, augment=True))
    assert n_augment > n_simple, (
        f"REFUSED: augment does not raise the term count here "
        f"({n_simple} -> {n_augment}); the attribution question is malformed.")

    gt = os.path.join(_ROOT, a.gt_dir)
    man = json.load(open(os.path.join(gt, "gt_manifest.json"), encoding="utf-8"))
    if not man.get("all_pass"):
        raise SystemExit("REFUSED: the P2-GT asset did not pass its own fitness checks.")
    zA = np.load(os.path.join(gt, "gt_halfA.npz"))
    zB = np.load(os.path.join(gt, "gt_halfB.npz"))
    seeds = [int(s) for s in zA["seeds"]]
    A_all, B_all = zA["means"], zB["means"]
    if a.limit:
        seeds, A_all, B_all = seeds[:a.limit], A_all[:a.limit], B_all[:a.limit]
    depth = A_all.shape[1]

    print("=" * 78)
    print("A3 -- does kappa_4 carry D1's 2.094x, or the raised diagram budget?")
    print("=" * 78)
    print(f"  asset            {a.gt_dir}  ({len(seeds)} networks)")
    print(f"  gt_noise/half    {man['gt_noise_per_half_mean']:.4e}")
    print(f"  upstream         {v['rev'][:12]} + recorded patch, verified")
    print(f"  diagram terms    augment=False {n_simple}  ->  augment=True {n_augment}"
          f"   (the bundle this arm splits)")
    print(f"  arms             SIMPLE | AUGMENT | A3 = AUGMENT with WK[4] := 0")
    print(f"  PRE-REGISTERED   f in [{F_PREDICTION[0]}, {F_PREDICTION[1]}];  "
          f"f>={F_KAPPA4_EARNED} kappa4 earns the memory work, "
          f"f<={F_BUDGET_CARRIES} it does not")
    print()

    def run(kind_or_zero, w):
        """One arm. Returns (depth, width) means and telemetry."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if kind_or_zero == "A3":
                m = D2.kprop_means_injected(w, None, "zero", device=a.device,
                                            dtype=torch.float32)
            else:
                with torch.no_grad():
                    m = bridge.kprop_means(w, k_max=3, factor=True,
                                           kind=kind_or_zero, device=a.device,
                                           dtype=torch.float32)
            snaps = sum(1 for c in caught if "Snapping negative variance" in str(c.message))
        tel = {"seconds": time.perf_counter() - t0, "snap_warnings": snaps,
               "finite": bool(np.isfinite(m).all())}
        if torch.cuda.is_available():
            tel["cuda_peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        return m, tel

    rows, t0 = [], time.perf_counter()
    for i, s in enumerate(seeds):
        w, _e, _ss = canonical_mint(man["width"], man["depth"], s)
        Af, Bf = A_all[i], B_all[i]          # (depth, width) -- ALL layers
        row = {"seed": s}
        preds = {}
        for name, arg in (("simple", SIMPLE), ("augment", AUGMENT), ("a3", "A3")):
            m, tel = run(arg, w)
            preds[name] = m
            row[name] = tel
            row[f"xmse_{name}"] = xmse(m[-1], Af[-1], Bf[-1])
            row[f"xmse_all_layers_{name}"] = [
                xmse(m[d], Af[d], Bf[d]) for d in range(depth)]

        # CONTROL: reversed arm order must not change a single value.
        if i < a.reverse_nets:
            rev = {}
            for name, arg in (("a3", "A3"), ("augment", AUGMENT), ("simple", SIMPLE)):
                m, _ = run(arg, w)
                rev[name] = float(np.max(np.abs(m - preds[name])))
            row["reverse_order_max_abs_diff"] = rev

        rows.append(row)
        den = row["xmse_simple"] - row["xmse_augment"]
        fi = (row["xmse_a3"] - row["xmse_augment"]) / den if den else float("nan")
        el = time.perf_counter() - t0
        print(f"    {i+1:>3}/{len(seeds)}  seed {s}   "
              f"SIMPLE {row['xmse_simple']:.4e}  AUGMENT {row['xmse_augment']:.4e}  "
              f"A3 {row['xmse_a3']:.4e}   f_i {fi:+.3f}   "
              f"eta {el/(i+1)*(len(seeds)-i-1)/60:.0f}m")
        _progress((i + 1) / len(seeds))
        if a.out:
            _dump(a, rows, man, v, {"simple": n_simple, "augment": n_augment},
                  depth, partial=True)

    return _report(a, rows, man, v, {"simple": n_simple, "augment": n_augment}, depth)


def _summary(rows, depth):
    if not rows:
        return {}
    g = lambda k: np.array([r[k] for r in rows], dtype=np.float64)
    xs, xa, x3 = g("xmse_simple"), g("xmse_augment"), g("xmse_a3")
    out = {"n": len(rows)}
    for k, arr in (("simple", xs), ("augment", xa), ("a3", x3)):
        m, se, ci = studentised_t(arr)
        out[k] = {"mean_xmse": m, "se": se, "ci95": list(ci)}
    d_total, se_t, ci_t = studentised_t(xs - xa)     # SIMPLE - AUGMENT, the D1 gain
    d_k4, se_k, ci_k = studentised_t(x3 - xa)        # A3 - AUGMENT, kappa_4's share
    d_bud, se_b, ci_b = studentised_t(xs - x3)       # SIMPLE - A3, the budget's share
    out["paired_simple_minus_augment"] = {"diff": d_total, "se": se_t, "ci95": list(ci_t)}
    out["paired_a3_minus_augment_KAPPA4_SHARE"] = {"diff": d_k4, "se": se_k, "ci95": list(ci_k)}
    out["paired_simple_minus_a3_BUDGET_SHARE"] = {"diff": d_bud, "se": se_b, "ci95": list(ci_b)}
    f = d_k4 / d_total if d_total else float("nan")
    per_net = (x3 - xa) / np.where(xs - xa == 0, np.nan, xs - xa)
    out["f_kappa4_share"] = f
    out["f_per_net_diagnostic"] = {
        "median": float(np.nanmedian(per_net)),
        "q25": float(np.nanpercentile(per_net, 25)),
        "q75": float(np.nanpercentile(per_net, 75)),
        "n_negative": int(np.nansum(per_net < 0)),
    }
    out["verdict"] = ("KAPPA4 CARRIES IT -- the (2,1,1) memory work is EARNED"
                      if f >= F_KAPPA4_EARNED else
                      "THE RAISED BUDGET CARRIES IT -- the memory work is aimed at "
                      "the wrong object; price a budget-only port first"
                      if f <= F_BUDGET_CARRIES else
                      "BOTH MATTER -- port needs both; a budget-only arm is worth "
                      "pricing as a cheaper fallback")
    out["prediction_f_held"] = bool(F_PREDICTION[0] <= f <= F_PREDICTION[1])
    # all-layers fingerprint
    for k in ("simple", "augment", "a3"):
        arr = np.array([r[f"xmse_all_layers_{k}"] for r in rows], dtype=np.float64)
        out[f"all_layers_mean_{k}"] = arr.mean(axis=0).tolist()
        out[f"all_layers_grand_mean_{k}"] = float(arr.mean())
    rev = [r["reverse_order_max_abs_diff"] for r in rows if "reverse_order_max_abs_diff" in r]
    if rev:
        out["reverse_order_worst_abs_diff"] = {
            k: max(r[k] for r in rev) for k in rev[0]}
    out["snap_warnings_total"] = int(sum(
        r[k]["snap_warnings"] for r in rows for k in ("simple", "augment", "a3")))
    return out


def _dump(a, rows, man, v, terms, depth, partial=False):
    if not a.out:
        return
    out = os.path.join(_ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rec = {
        "slice": "A3 -- attribute D1's 2.094x between kappa_4 and the raised diagram budget",
        "contract": "experiments/LOG.md, ERRATUM + CONTRACT (Rule 1) ARM A3, commit 2e4e9c2",
        "asset": a.gt_dir, "n_mlps": len(rows),
        "label_noise_per_half": man["gt_noise_per_half_mean"],
        "upstream": v, "diagram_terms": terms, "partial": partial,
        "f_thresholds": {"kappa4_earned": F_KAPPA4_EARNED,
                         "budget_carries": F_BUDGET_CARRIES,
                         "prediction": list(F_PREDICTION)},
        "tier": "DESIGN (P01), re-read. Orients priority; promotes NOTHING.",
        "arm_note": ("A3 is an ABLATION, not a deployable estimator: it keeps the "
                     "raised diagram budget and zeroes every kappa_4 information "
                     "surface. It does not reduce memory -- the (2,1,1) slice is "
                     "still formed, only its content is zeroed."),
        "cost_note": "No cost claim. Torch meter, not flopscope.",
        "summary": _summary(rows, depth), "rows": rows,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)
    return rec


def _report(a, rows, man, v, terms, depth) -> int:
    s = _summary(rows, depth)
    print()
    print(f"  gt_noise per half   {man['gt_noise_per_half_mean']:.4e}")
    for k, lbl in (("simple", "SIMPLE"), ("augment", "AUGMENT"), ("a3", "A3 (AUGMENT, k4:=0)")):
        e = s[k]
        print(f"  {lbl:<22s} {e['mean_xmse']:.4e}   se {e['se']:.2e}   "
              f"95% [{e['ci95'][0]:.4e}, {e['ci95'][1]:.4e}]")
    print()
    print("  THE BINDING STATISTICS -- paired arithmetic differences, studentised-t:")
    for key, lbl in (("paired_simple_minus_augment", "SIMPLE - AUGMENT  (D1's total gain)"),
                     ("paired_a3_minus_augment_KAPPA4_SHARE", "A3 - AUGMENT      (kappa_4's share)"),
                     ("paired_simple_minus_a3_BUDGET_SHARE", "SIMPLE - A3       (budget's share)")):
        e = s[key]
        print(f"    {lbl:<36s} {e['diff']:+.4e}   95% [{e['ci95'][0]:+.4e}, {e['ci95'][1]:+.4e}]")
    d = s["f_per_net_diagnostic"]
    print()
    print(f"  f = kappa_4's share of the gain   {s['f_kappa4_share']:+.4f}")
    print(f"    per-net diagnostic  median {d['median']:+.3f}  "
          f"IQR [{d['q25']:+.3f}, {d['q75']:+.3f}]  negative on {d['n_negative']}/{s['n']}")
    print(f"    prediction f in [{F_PREDICTION[0]}, {F_PREDICTION[1]}] -> "
          f"{'HELD' if s['prediction_f_held'] else 'MISSED'}")
    print()
    print(f"  VERDICT  {s['verdict']}")
    print()
    print("  ALL-LAYERS fingerprint (grand mean over layers; leaders publish 7-11e-9)")
    for k, lbl in (("simple", "SIMPLE"), ("augment", "AUGMENT"), ("a3", "A3")):
        print(f"    {lbl:<10s} {s[f'all_layers_grand_mean_{k}']:.4e}")
    if "reverse_order_worst_abs_diff" in s:
        r = s["reverse_order_worst_abs_diff"]
        ok = all(x == 0.0 for x in r.values())
        print()
        print(f"  CONTROL  reversed arm order, worst |diff|: "
              + "  ".join(f"{k} {x:.2e}" for k, x in r.items())
              + f"   -> {'BIT-IDENTICAL' if ok else 'ORDER MATTERS -- investigate'}")
    print(f"  upstream 'Snapping negative variance' warnings: {s['snap_warnings_total']}")
    _dump(a, rows, man, v, terms, depth)
    if a.out:
        print(f"\n  artifact  {a.out}")
    _progress(1.0)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
