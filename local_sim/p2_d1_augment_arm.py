"""D1 — does ARC's `kind=AUGMENT` carry κ₄ far enough to reach the board floor?

Contract: `experiments/LOG.md`, *CONTRACT (Rule 1): SLICE 3 STEP 1* (2026-09-08),
committed at `432f7fd` **before this file existed**. Handoff:
`docs/STRATEGIST_RESPONSE_2026-09-08.md` §5 step 1.

## The one thing that changes between the arms

`kind`. Nothing else. Same nets, same order, same process, same dtype, same
`factor=True`, same `use_pK` (upstream's default `True` — see below). A one-flag
diff is the entire point: it makes the measured difference attributable to the
κ₄ representation and to nothing else.

    SIMPLE   get_r_x(4, 3) = 2  ->  degree 4 has 4 - 2*2 = 0 free indices  -> a SCALAR
    AUGMENT  get_r_x(4, 3) = 1  ->  degree 4 has 4 - 2*1 = 2 free indices  -> an (n, n) MATRIX

`get_d_max(3, ·) = 4` for **both**, so AUGMENT does not add a cumulant degree.
It enriches the one that is already tracked. That is the whole hypothesis.

## What this number is a LOWER bound on, and why that is not a defect

Upstream refuses to pretend otherwise (`factor_k3.py:291-299`): with
`augment=True` the factored path **drops** the non-hypertree diagrams — the
covariance triangle and the κ₃ × edge products — and those have *the same*
`Θ(n^-3)` squared size as the augmented diagrams it keeps. So factored-AUGMENT is
strictly less than unfactored-AUGMENT.

WP3 (`local_sim/p2_augment.py`) concluded from the same warning that unfactored
is the only admissible **parity** target, and that was right *for a parity
target*. This is not a parity slice. Unfactored AUGMENT is `O(n^4)` and cannot be
ported to the grader at `n = 1024` at any price, so the question worth money is
what the **portable** variant is worth. If the portable variant already reaches
the band, the lane is open; if it does not, the unfactored arm remains a stated
upper bound that nothing here has measured. Both readings are recorded rather
than one being quietly adopted.

## Power cumulants (the handoff's D3) are already on

Upstream `use_pK` defaults to `True`; `use_pK=False` is its documented ablation
("doesn't get good MSE"). Neither this module nor `p2_kprop_bridge` passes it, so
**both arms already include power cumulants** and D3 is not an additional lever.
`_assert_pk_default()` below turns that from a claim into a check, because a
silently changed upstream default would make D3 look like a discovery.

## The estimator

The two-half cross estimator, never the plain MSE:

    xmse(p) = mean_j (p_j - A_j)(p_j - B_j)

`E[(p - y)^2] = MSE(p) + Var(y)` biases every single-replica MSE upward by the
label's own variance. With independent halves `E[e_A e_B] = 0`, so the cross
product is unbiased **exactly**. The plain half-A MSE is printed beside it so the
size of the correction stays visible instead of being taken on trust.

## The instrument check that is wired in rather than promised

The SIMPLE arm is **re-run**, not read from `pa_k3.json`, so the comparison is
paired within one process. The re-run is then compared *against* the stored P-A
numbers per network. If they disagree beyond `--drift-tol`, the run **halts**:
that means the asset, the mint, the upstream checkout or the bridge moved since
2026-08-26, and every conclusion downstream of it would be about the drift rather
than about κ₄.

## Memory is a deliverable here, not telemetry

Whether the port is feasible at all is decided by this run. Peak CUDA allocation
and reservation are recorded per arm per network. **An OOM is a reported
result.** It is never handled by narrowing the width — a width-512 proxy would
answer a different question than the one asked, and answering a narrower question
while looking successful is the exact failure mode this repository has a standing
instrument-check rule about.

Research only. Never imported by `estimator.py` (needs torch; the grader has none).
`# seed-ledger: no block named` — reads the existing P01 DESIGN asset.

Run (GPU venv, via workerq — this is a GPU job):
    .venv-gpu\\Scripts\\python.exe -u local_sim/p2_d1_augment_arm.py \\
        --out experiments/2026-09-08_slice3/d1_augment.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

# Pre-registered in the contract at 432f7fd, before any datum. Do not edit.
D1_BAND = (1.8e-8, 2.5e-8)          # prediction 1
D1_FALSIFIER = 3.5e-8               # prediction 2 / handoff §5.5
D1_BUILD_GATE = 2.5e-8              # handoff §5.2 -- port only if xmse <= this
PA_STORED = "experiments/2026-08-26_p2_gt/pa_k3.json"


def xmse(pred, A, B):
    """Unbiased two-half cross estimator. E[(p-A)(p-B)] = MSE(p) exactly."""
    return float(np.mean((pred - A) * (pred - B)))


def studentised_t(d, alpha=0.05):
    """Paired studentised-t interval. The binding statistic; never percentile/BCa."""
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


def _assert_pk_default():
    """D3's premise, checked rather than asserted in prose.

    If upstream's `use_pK` default ever became False, both arms would silently
    lose power cumulants and 'turning D3 on' would look like a new lever when it
    was really a regression being undone.
    """
    import inspect
    from mlp_kprop.kprop_harmonic import mlp_kprop, nonlin_kprop
    out = {}
    for fn in (mlp_kprop, nonlin_kprop):
        p = inspect.signature(fn).parameters.get("use_pK")
        out[fn.__name__] = None if p is None else p.default
    assert all(v is True for v in out.values()), (
        f"REFUSED: upstream use_pK defaults changed: {out}. Power cumulants are "
        "assumed ON in both arms by the contract's Evidence table.")
    return out


def _mem_reset(torch):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _mem_read(torch):
    if not torch.cuda.is_available():
        return {}
    return {
        "cuda_peak_alloc_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
        "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30,
    }


def _host_rss_gib():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 30
    except Exception:
        return None


def _progress(frac):
    """worker-q turns a written fraction into a measured ETA instead of a guess."""
    p = os.environ.get("WORKERQ_PROGRESS")
    if not p:
        return
    try:
        with open(p, "w") as fh:
            fh.write(str(min(max(frac, 0.0), 1.0)))
    except Exception:
        pass


def _run_arm(bridge, torch, w, kind, dtype):
    """One arm on one network. Returns (means, telemetry). OOM is DATA, not a crash."""
    _mem_reset(torch)
    t0 = time.perf_counter()
    try:
        means = bridge.kprop_means(w, k_max=3, factor=True, kind=kind, dtype=dtype)
    except torch.cuda.OutOfMemoryError as e:
        tel = {"seconds": time.perf_counter() - t0, "oom": True,
               "error": str(e)[:400], **_mem_read(torch)}
        _mem_reset(torch)
        return None, tel
    tel = {"seconds": time.perf_counter() - t0, "oom": False,
           "finite": bool(np.isfinite(means).all()),
           "host_rss_gib": _host_rss_gib(), **_mem_read(torch)}
    return means, tel


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="experiments/2026-08-26_p2_gt")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    ap.add_argument("--drift-tol", type=float, default=1e-12,
                    help="max |re-run SIMPLE - stored P-A| per network before HALT")
    ap.add_argument("--probe-only", type=int, default=0,
                    help="run only this many networks; a feasibility probe, not a result")
    a = ap.parse_args(argv)

    import torch
    import p2_kprop_bridge as bridge
    from mint_provenance import canonical_mint

    dtype = getattr(torch, a.dtype)
    gt = os.path.join(_ROOT, a.gt_dir)
    man = json.load(open(os.path.join(gt, "gt_manifest.json"), encoding="utf-8"))
    if not man.get("all_pass"):
        raise SystemExit("REFUSED: the P2-GT asset did not pass its own fitness "
                         "checks. An arm measured against a void label is not a "
                         "measurement.")

    zA = np.load(os.path.join(gt, "gt_halfA.npz"))
    zB = np.load(os.path.join(gt, "gt_halfB.npz"))
    seeds = [int(s) for s in zA["seeds"]]
    assert seeds == [int(s) for s in zB["seeds"]], "halves disagree about seeds"
    A_all, B_all = zA["means"], zB["means"]
    n_all = len(seeds)
    if a.limit:
        seeds, A_all, B_all = seeds[:a.limit], A_all[:a.limit], B_all[:a.limit]
    if a.probe_only:
        seeds, A_all, B_all = seeds[:a.probe_only], A_all[:a.probe_only], B_all[:a.probe_only]

    # -- instrument checks, all before the first datum -----------------------
    v = bridge.verify_upstream(bridge._src())
    if not v["pass"]:
        raise SystemExit(f"REFUSED: upstream checkout is not the pinned rev + patch: {v}")
    pk = _assert_pk_default()

    from mlp_kprop.kprop_harmonic import SIMPLE, AUGMENT, get_d_max, get_r_x
    shape_check = {
        "d_max_simple": get_d_max(3, SIMPLE), "d_max_augment": get_d_max(3, AUGMENT),
        "r_x_4_simple": get_r_x(4, 3, SIMPLE), "r_x_4_augment": get_r_x(4, 3, AUGMENT),
    }
    assert shape_check["d_max_simple"] == shape_check["d_max_augment"] == 4, shape_check
    assert shape_check["r_x_4_simple"] == 2 and shape_check["r_x_4_augment"] == 1, (
        f"REFUSED: the contract's mechanism claim does not hold in this upstream: "
        f"{shape_check}. SIMPLE must carry degree 4 as a scalar (r=2) and AUGMENT "
        f"as an (n,n) core (r=1).")

    stored = {}
    sp = os.path.join(_ROOT, PA_STORED)
    if os.path.exists(sp):
        stored = {int(r["seed"]): float(r["xmse_k3"])
                  for r in json.load(open(sp, encoding="utf-8"))["rows"]}

    print("=" * 78)
    print("D1  --  ARC kind=AUGMENT vs kind=SIMPLE, factorized K3 at 1024 x 16")
    print("=" * 78)
    print(f"  asset             {a.gt_dir}  ({len(seeds)} of {n_all} networks, "
          f"{man['samples_per_half']:,} samples per half)")
    print(f"  label noise/half  {man['gt_noise_per_half_mean']:.4e}")
    print(f"  upstream          {v['rev'][:12]} + recorded patch, verified")
    print(f"  use_pK defaults   {pk}  (power cumulants ON in BOTH arms -> D3 is not a lever)")
    print(f"  degree-4 shape    SIMPLE r={shape_check['r_x_4_simple']} (scalar)   "
          f"AUGMENT r={shape_check['r_x_4_augment']} (n x n)   d_max=4 for both")
    print(f"  dtype             {a.dtype}")
    print(f"  PRE-REGISTERED    band [{D1_BAND[0]:.1e}, {D1_BAND[1]:.1e}]   "
          f"falsifier >= {D1_FALSIFIER:.1e}   build gate <= {D1_BUILD_GATE:.1e}")
    if a.probe_only:
        print(f"  *** PROBE ONLY ({a.probe_only} net(s)) -- feasibility, NOT a result ***")
    print()

    rows, t0 = [], time.perf_counter()
    for i, s in enumerate(seeds):
        w, _e, _ss = canonical_mint(man["width"], man["depth"], s)
        Af, Bf = A_all[i, -1, :], B_all[i, -1, :]
        row = {"seed": s}

        m_s, tel_s = _run_arm(bridge, torch, w, SIMPLE, dtype)
        row["simple"] = tel_s
        if m_s is not None:
            row["xmse_simple"] = xmse(m_s[-1], Af, Bf)
            row["plain_mse_simple_halfA"] = float(np.mean((m_s[-1] - Af) ** 2))
            if s in stored:
                row["drift_vs_pa_k3"] = row["xmse_simple"] - stored[s]

        m_a, tel_a = _run_arm(bridge, torch, w, AUGMENT, dtype)
        row["augment"] = tel_a
        if m_a is not None:
            row["xmse_augment"] = xmse(m_a[-1], Af, Bf)
            row["plain_mse_augment_halfA"] = float(np.mean((m_a[-1] - Af) ** 2))

        rows.append(row)
        el = time.perf_counter() - t0
        d = row.get("drift_vs_pa_k3")
        print(f"    {i + 1:>3}/{len(seeds)}  seed {s}   "
              f"SIMPLE {row.get('xmse_simple', float('nan')):.4e}   "
              f"AUGMENT {row.get('xmse_augment', float('nan')):.4e}   "
              f"[{tel_s['seconds']:.0f}s/{tel_s.get('cuda_peak_alloc_gib', 0):.1f}G, "
              f"{tel_a['seconds']:.0f}s/{tel_a.get('cuda_peak_alloc_gib', 0):.1f}G]"
              + (f"  drift {d:+.2e}" if d is not None else "")
              + (f"  eta {el / (i + 1) * (len(seeds) - i - 1) / 60:.0f}m"))
        _progress((i + 1) / len(seeds))

        # Halt on drift the moment it is visible, not after 24 networks of it.
        if d is not None and abs(d) > a.drift_tol:
            _dump(a, rows, man, v, pk, shape_check, seeds,
                  halted=f"SIMPLE re-run drifted from pa_k3.json by {d:+.3e} on "
                         f"seed {s} (tol {a.drift_tol:.1e}). The asset, the mint, "
                         f"the upstream checkout or the bridge has moved since "
                         f"2026-08-26; every number downstream would be about that "
                         f"drift and not about kappa_4.")
            raise SystemExit(1)

        if a.out:  # incremental: a preemption or an OOM still leaves the data
            _dump(a, rows, man, v, pk, shape_check, seeds, partial=True)

    return _report(a, rows, man, v, pk, shape_check, seeds)


def _summary(rows):
    ok = [r for r in rows if "xmse_simple" in r and "xmse_augment" in r]
    if not ok:
        return {}
    xs = np.array([r["xmse_simple"] for r in ok])
    xa = np.array([r["xmse_augment"] for r in ok])
    ms, ses, cis = studentised_t(xs)
    ma, sea, cia = studentised_t(xa)
    dm, sed, cid = studentised_t(xs - xa)     # positive => AUGMENT is better
    return {
        "n_paired": len(ok),
        "simple_mean_xmse": ms, "simple_se": ses, "simple_ci95": list(cis),
        "augment_mean_xmse": ma, "augment_se": sea, "augment_ci95": list(cia),
        "paired_simple_minus_augment": dm, "paired_se": sed, "paired_ci95": list(cid),
        "augment_better_on": int((xs > xa).sum()),
        "ratio_diagnostic_simple_over_augment": ms / ma if ma else None,
        "geo_mean_ratio_diagnostic": float(np.exp(np.mean(np.log(xs / xa)))),
        "prediction_1_band_held": bool(D1_BAND[0] <= ma <= D1_BAND[1]),
        "prediction_2_falsifier_fired": bool(ma >= D1_FALSIFIER),
        "build_gate_passed": bool(ma <= D1_BUILD_GATE),
    }


def _dump(a, rows, man, v, pk, shape_check, seeds, partial=False, halted=""):
    if not a.out:
        return
    out = os.path.join(_ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rec = {
        "slice": "D1 -- ARC kind=AUGMENT vs SIMPLE, factorized K3, 1024x16",
        "contract": "experiments/LOG.md, CONTRACT (Rule 1) Slice 3 step 1, commit 432f7fd",
        "handoff": "docs/STRATEGIST_RESPONSE_2026-09-08.md section 5 step 1",
        "asset": a.gt_dir, "n_mlps_requested": len(seeds),
        "samples_per_half": man["samples_per_half"],
        "label_noise_per_half": man["gt_noise_per_half_mean"],
        "dtype": a.dtype, "upstream": v, "use_pK_defaults": pk,
        "degree4_shape_check": shape_check,
        "prediction_band": list(D1_BAND), "falsifier": D1_FALSIFIER,
        "build_gate": D1_BUILD_GATE,
        "partial": partial, "halted": halted,
        "probe_only": bool(a.probe_only),
        "tier": ("DESIGN (P01), re-read. Orients priority; promotes NOTHING. "
                 "A port needs a fresh block plus the official mini-100."),
        "scope_caveat": ("factor=True only. The factored AUGMENT path DROPS the "
                         "non-hypertree diagrams (factor_k3.py:291-299), which have "
                         "the same Theta(n^-3) squared size as those it keeps. This "
                         "number is therefore a LOWER BOUND on what AUGMENT-class "
                         "information is worth. Unfactored AUGMENT is O(n^4) and "
                         "unreachable at n=1024, so it is not portable either way."),
        "cost_note": ("No cost claim. Upstream's NamedFlopCounter is torch's meter, "
                      "not flopscope's. Any FLOP figure for a port is a PROJECTION "
                      "until p2_flop_anchor prices the fnp spelling."),
        "summary": _summary(rows), "rows": rows,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)
    return rec


def _report(a, rows, man, v, pk, shape_check, seeds) -> int:
    s = _summary(rows)
    ooms = [r["seed"] for r in rows if r.get("augment", {}).get("oom")]
    print()
    if ooms:
        print(f"  *** AUGMENT OOMed on {len(ooms)} network(s): {ooms[:5]} ***")
        print("      Reported as a result. NOT retried at a narrower width -- a")
        print("      width-512 proxy would answer a different question.")
    if not s:
        print("  NO PAIRED ROWS. Nothing is concluded.")
        _dump(a, rows, man, v, pk, shape_check, seeds)
        return 1

    gt_noise = man["gt_noise_per_half_mean"]
    print(f"  gt_noise per half         {gt_noise:.4e}   (quoted beside every MSE)")
    print(f"  SIMPLE   mean xmse        {s['simple_mean_xmse']:.4e}   "
          f"se {s['simple_se']:.3e}   95% [{s['simple_ci95'][0]:.4e}, {s['simple_ci95'][1]:.4e}]")
    print(f"  AUGMENT  mean xmse        {s['augment_mean_xmse']:.4e}   "
          f"se {s['augment_se']:.3e}   95% [{s['augment_ci95'][0]:.4e}, {s['augment_ci95'][1]:.4e}]")
    print()
    print(f"  PAIRED SIMPLE - AUGMENT   {s['paired_simple_minus_augment']:+.4e}   "
          f"95% [{s['paired_ci95'][0]:+.4e}, {s['paired_ci95'][1]:+.4e}]   "
          f"<- THE BINDING STATISTIC")
    print(f"  AUGMENT better on         {s['augment_better_on']}/{s['n_paired']} networks")
    print(f"  ratio (diagnostic only)   {s['ratio_diagnostic_simple_over_augment']:.4f}x   "
          f"geo-mean {s['geo_mean_ratio_diagnostic']:.4f}x")
    print()
    print(f"  PREDICTION 1  AUGMENT in [{D1_BAND[0]:.1e}, {D1_BAND[1]:.1e}] -> "
          f"{'HELD' if s['prediction_1_band_held'] else 'MISSED'}")
    print(f"  PREDICTION 2  falsifier >= {D1_FALSIFIER:.1e} -> "
          f"{'FIRED -- augmentation is NOT the missing term' if s['prediction_2_falsifier_fired'] else 'not fired'}")
    print(f"  BUILD GATE    xmse <= {D1_BUILD_GATE:.1e} -> "
          f"{'PASSED -- port AUGMENT (handoff 5.2)' if s['build_gate_passed'] else 'NOT PASSED -- do not port'}")
    if a.probe_only:
        print()
        print("  *** PROBE ONLY. These verdicts are NOT the slice's answer. ***")

    ms = max((r.get("simple", {}).get("cuda_peak_alloc_gib", 0) for r in rows), default=0)
    ma = max((r.get("augment", {}).get("cuda_peak_alloc_gib", 0) for r in rows), default=0)
    ts = sum(r.get("simple", {}).get("seconds", 0) for r in rows)
    ta = sum(r.get("augment", {}).get("seconds", 0) for r in rows)
    print()
    print(f"  FEASIBILITY   peak CUDA alloc  SIMPLE {ms:.2f} GiB   AUGMENT {ma:.2f} GiB"
          + (f"   ({ma / ms:.2f}x)" if ms else ""))
    print(f"                total wall       SIMPLE {ts:.0f}s   AUGMENT {ta:.0f}s"
          + (f"   ({ta / ts:.2f}x)" if ts else ""))
    _dump(a, rows, man, v, pk, shape_check, seeds)
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
