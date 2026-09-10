"""T1 — AUGMENT-LITE: which CHANNEL of kappa_4 carries the 2.094x?

Contract: `experiments/LOG.md`, *CONTRACT (Rule 1): T1* (2026-09-10), committed
**before this file existed**. Handoff `docs/STRATEGIST_RESPONSE_2026-09-10.md`
§1 (H1) and §5 step 1.

THE QUESTION, AND WHY IT IS THE WHOLE LANE
------------------------------------------
Slice 3 settled that **all** of `kind=AUGMENT`'s `2.094x` is kappa_4's `(n, n)`
structure (the raised diagram budget alone is `1.085x` *worse*). Slice 4 settled
that the factored spelling of that structure is `3.6x-5.8x` over the hard budget,
because it births `8n` lineage columns per layer against SIMPLE's `2n`.

**Nobody has measured which of kappa_4's two channels carries the accuracy.** In
upstream's factored code the `(n, n)` object couples into the flow twice:

  (i)  the CHEAP channel -- `WK[4]` is read by the `terms_iso` sum that builds
       `pK_slices`, i.e. the Wick updates of mu, Sigma and the diagonal kappa_3
       slices, and by `K211_contrib`, kappa_4's own update. All `O(n^2)`.
  (ii) the EXPENSIVE channel -- kappa_4 BIRTHS new factored kappa_3 columns:
       the `(2, 2)` leg `diagslice(WK[4], (2,2))` and the two surviving
       `211 H(d=4,r=1) -> 111` blocks. These are the columns that kill the port.

If the accuracy lives in (i), an affordable "AUGMENT-lite" exists: carry the
`(n, n)` object, transport it through `W`, consume it in the mu/Sigma/diag
updates, and **birth nothing**. If it lives in (ii), H1 is dead in every
affordable form and the score lane closes with it.

HOW THE ABLATION IS APPLIED, AND WHY NOT BY COPYING THE FUNCTION
-----------------------------------------------------------------
The upstream checkout is pinned at rev `93d091a4` plus the recorded patch and is
**never edited** -- `p2_kprop_bridge.verify_upstream` refuses the run otherwise.
The ablation is therefore applied to a *copy of the source text*, through two
substitutions whose search strings are **exact anchors**. An anchor that no
longer matches raises `AnchorError` and the run refuses; upstream cannot drift
under this module silently, which a hand-copied 130-line function would let it
do.

The patched `factored_nonlin_kprop_k3` is installed on `mlp_kprop.factor_k3`.
`kprop_harmonic` imports that symbol **inside** its own function body
(`kprop_harmonic.py:368`), so the rebinding is seen on the next call and the
un-ablated path is byte-identical to upstream when the flag is off.

THE FIDELITY CHECK THAT MAKES THE ABLATION READABLE
----------------------------------------------------
Before any datum, one network is run through the AUGMENT arm **twice**: once on
upstream's own function and once on the patched module with the flag OFF. They
must agree **bit for bit**. If they do not, the patch changed something other
than what it claims and nothing downstream is read. That check is the reason
this is an ablation rather than a rewrite.

WHAT IS MEASURED
----------------
Three arms, paired within one process on the same 24 P01 networks, same labels,
same order, `float32` (D1's anchor dtype -- and Slice 4 lost two runs to a
float64 default, so it is stated rather than assumed):

    SIMPLE          control, must reproduce 4.004685e-08
    AUGMENT         control, must reproduce 1.912323e-08
    AUGMENT-LITE    kappa_4's births suppressed, its cheap channel live

plus, free, the **lineage column count per layer per arm** -- the number that
decides affordability -- read off the returned `FactoredTensor` rather than
estimated.

Research only. Never imported by `estimator.py` (needs torch).
`# seed-ledger: no block named` -- re-reads the existing P01 DESIGN asset, whose
manifest forbids closing a lane on it.

Run (GPU venv, via workerq):
    .venv-gpu\\Scripts\\python.exe -u local_sim/p2_t1_augment_lite.py \\
        --out experiments/2026-09-10_slice5/t1_augment_lite.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402

# ---- pre-registered in the contract, before any datum. Do not edit. -------- #
SIMPLE_ANCHOR = 4.004685e-08          # pa_k3.json, D1's re-run control
AUGMENT_ANCHOR = 1.912323e-08         # D1, bit-exact across two reboots
FULL_GAIN = SIMPLE_ANCHOR / AUGMENT_ANCHOR      # 2.0942x
#: The gate is on RETENTION -- the share of the full gain the lite arm keeps.
BUILD_GATE = 1.5                      # >= 1.5x of SIMPLE  ->  price and build
KILL_GATE = 1.2                       # <= 1.2x of SIMPLE  ->  H1 is dead
PREDICTION_BAND = (1.4, 1.8)          # the strategist's pre-registered band
PA_STORED = "experiments/2026-08-26_p2_gt/pa_k3.json"

#: (tag, kind, ablate kappa_4 births, ablate the raised budget's extra legs).
#: The order is the order they run in, per network, inside one process.
#: The 2x2 is now COMPLETE. The first version ran (F,F), (F,F), (T,F), (T,T) and
#: never ran (births kept, legs dropped) -- a missing cell the 2026-09-10 Skeptic
#: found, and the one that decides whether the collapse is about kappa_4 at all.
ARMS = (
    ("simple",   "SIMPLE",  False, False),
    ("augment",  "AUGMENT", False, False),
    ("lite",     "AUGMENT", True,  False),
    ("legsonly", "AUGMENT", False, True),
    ("minimal",  "AUGMENT", True,  True),
)
ARM_LABEL = {
    "simple":   "SIMPLE (2n)",
    "augment":  "AUGMENT (8n)",
    "lite":     "AUGMENT-LITE (5n)",
    "legsonly": "AUGMENT-LEGSONLY (4n)",
    "minimal":  "AUGMENT-MINIMAL (2n)",
}

#: The Gaussian baselines, on the SAME 24 seeds and the SAME labels, already on
#: disk in `pa_k3.json`. They cost nothing to report and they are the reference
#: the first version of this module omitted -- without them an arm sitting on the
#: no-kappa_3 baseline looks like a finding about kappa_4.
BASELINE_COLUMNS = ("xmse_k2", "xmse_k2cal")


class AnchorError(RuntimeError):
    """An ablation anchor no longer matches the pinned upstream source."""


# --------------------------------------------------------------------------- #
# the ablation                                                                 #
# --------------------------------------------------------------------------- #

#: (name, exact search text, replacement). Both anchors must occur EXACTLY once.
_ANCHORS = [
    (
        "k4_leg_block",
        "            if 4 in WK:\n"
        "                assert isinstance(WK[4], HTensor)\n"
        "                WK_22 = diagslice(WK[4], (2, 2), output_zero_repeated=use_pK) / 4\n"
        "                legs.append((2, 2, WK_22, 2))\n",
        "            if 4 in WK and not _ABLATE_K4_BIRTHS:\n"
        "                assert isinstance(WK[4], HTensor)\n"
        "                WK_22 = diagslice(WK[4], (2, 2), output_zero_repeated=use_pK) / 4\n"
        "                legs.append((2, 2, WK_22, 2))\n",
    ),
    (
        "k4_211_to_111_blocks",
        "        if augment and 4 in WK:\n",
        "        if augment and 4 in WK and not _ABLATE_K4_BIRTHS:\n",
    ),
    (
        "raised_budget_legs",
        "        legs = [(1, 1, WK_11, 1)]\n"
        "        if augment:\n",
        "        legs = [(1, 1, WK_11, 1)]\n"
        "        if augment and not _ABLATE_EXTRA_LEGS:\n",
    ),
]


def build_ablated_module(src_path):
    """Exec a text-patched copy of `factor_k3.py`. Refuses on any anchor miss."""
    with open(src_path, encoding="utf-8") as fh:
        src = fh.read()
    original_sha = hashlib.sha256(src.encode("utf-8")).hexdigest()

    patched = src
    applied = []
    for name, find, repl in _ANCHORS:
        hits = patched.count(find)
        if hits != 1:
            raise AnchorError(
                f"REFUSED: ablation anchor {name!r} matched {hits} times in "
                f"{src_path} (expected exactly 1). The pinned upstream source "
                f"has moved under this module; re-derive the anchor rather than "
                f"loosening it.")
        patched = patched.replace(find, repl)
        applied.append(name)

    # The flag itself, injected after the imports so the module body can read it.
    marker = "FacHTower = dict[int, FactoredTensor | HTensor]"
    if patched.count(marker) != 1:
        raise AnchorError(f"REFUSED: flag-injection anchor missing in {src_path}")
    patched = patched.replace(
        marker,
        "_ABLATE_K4_BIRTHS = False\n_ABLATE_EXTRA_LEGS = False\n\n" + marker)

    mod = types.ModuleType("mlp_kprop.factor_k3_ablated")
    mod.__file__ = src_path + " [ABLATED COPY -- the file itself is untouched]"
    code = compile(patched, mod.__file__, "exec")
    exec(code, mod.__dict__)                                   # noqa: S102
    return mod, {
        "source": src_path,
        "source_sha256": original_sha,
        "patched_sha256": hashlib.sha256(patched.encode("utf-8")).hexdigest(),
        "anchors_applied": applied,
    }


class _Ablation:
    """Install the patched `factored_nonlin_kprop_k3`; flag it per arm."""

    def __init__(self, mod):
        self.mod = mod
        import mlp_kprop.factor_k3 as upstream
        self.upstream = upstream
        self.original = upstream.factored_nonlin_kprop_k3

    def install(self):
        self.upstream.factored_nonlin_kprop_k3 = \
            self.mod.factored_nonlin_kprop_k3

    def restore(self):
        self.upstream.factored_nonlin_kprop_k3 = self.original

    def set(self, births: bool, legs: bool = False):
        """`births` suppresses kappa_4's kappa_3 births; `legs` the raised
        budget's kappa_3/kappa_2 leg blocks. Both off is upstream AUGMENT."""
        self.mod._ABLATE_K4_BIRTHS = bool(births)
        self.mod._ABLATE_EXTRA_LEGS = bool(legs)


# --------------------------------------------------------------------------- #
# statistics -- the binding ones, and only those                               #
# --------------------------------------------------------------------------- #

def xmse(pred, A, B):
    """Unbiased two-half cross estimator. `E[(p-A)(p-B)] = MSE(p)` exactly."""
    return float(np.mean((pred - A) * (pred - B)))


#: Which branch produced the last critical value. An artifact that cannot say
#: whether its interval came from scipy or from a fallback is not reproducible,
#: and the old fallback was WRONG: `2.064` is t at df=24, used for every n
#: from 2 to 60. A Skeptic found it latent on 2026-09-10 (scipy is installed in
#: both venvs, so it never fired) and it is fixed before a 1.15x gate uses it.
_LAST_T_CRIT = {"value": None, "source": None}

#: Two-sided 97.5% t critical values, df 1..30. Exact to 4 decimals, so a
#: missing scipy degrades the interval by <0.05% instead of by 12%.
_T975 = (
    12.7062, 4.3027, 3.1824, 2.7764, 2.5706, 2.4469, 2.3646, 2.3060, 2.2622,
    2.2281, 2.2010, 2.1788, 2.1604, 2.1448, 2.1314, 2.1199, 2.1098, 2.1009,
    2.0930, 2.0860, 2.0796, 2.0739, 2.0687, 2.0639, 2.0595, 2.0555, 2.0518,
    2.0484, 2.0452, 2.0423,
)


def _t_crit(df, alpha=0.05):
    """The two-sided critical value, and where it came from."""
    if df < 1:
        return float("nan"), "df<1"
    try:
        from scipy import stats
        return float(stats.t.ppf(1 - alpha / 2, df)), "scipy"
    except Exception:
        pass
    if abs(alpha - 0.05) > 1e-12:
        return 1.96, "normal-approx (alpha != 0.05, no scipy)"
    if df <= len(_T975):
        return _T975[df - 1], "table"
    # Beyond the table, Cornish-Fisher on the normal quantile: error < 0.1%.
    z = 1.959963984540054
    return z + (z ** 3 + z) / (4.0 * df), "cornish-fisher"


def studentised_t(d, alpha=0.05):
    """Paired studentised-t interval. Never percentile, never BCa."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    crit, src = _t_crit(n - 1, alpha)
    _LAST_T_CRIT["value"], _LAST_T_CRIT["source"] = crit, src
    return m, se, (m - crit * se, m + crit * se)


# --------------------------------------------------------------------------- #
# plumbing                                                                     #
# --------------------------------------------------------------------------- #

def _mem_reset(torch):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _mem_read(torch):
    if not torch.cuda.is_available():
        return {}
    return {"cuda_peak_alloc_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
            "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30}


def _progress(frac):
    p = os.environ.get("WORKERQ_PROGRESS")
    if not p:
        return
    try:
        with open(p, "w") as fh:
            fh.write(str(min(max(frac, 0.0), 1.0)))
    except Exception:
        pass


class _ColumnCounter:
    """Record the lineage column count each layer hands on. Costs nothing.

    This is the affordability number -- SIMPLE births `2n` per layer, factored
    AUGMENT `8n` -- and reading it off the returned object is the difference
    between a measured port price and an estimated one.
    """

    def __init__(self, ablation):
        self.ab = ablation
        self.inner = None
        self.cols = []

    def __enter__(self):
        target = self.ab.upstream
        self.inner = target.factored_nonlin_kprop_k3

        def wrapped(*a, **kw):
            out = self.inner(*a, **kw)
            try:
                k3 = out.get(3)
                f = getattr(k3, "factors", None)
                self.cols.append(int(f[0].shape[1]) if f is not None else -1)
            except Exception:
                self.cols.append(-1)
            return out

        target.factored_nonlin_kprop_k3 = wrapped
        return self

    def __exit__(self, *exc):
        self.ab.upstream.factored_nonlin_kprop_k3 = self.inner
        return False


def _run_arm(bridge, torch, w, kind, dtype, ablation=None, ablate=False,
             ablate_legs=False, count=None):
    """One arm on one network. Returns (means, telemetry). OOM is DATA."""
    if ablation is not None:
        ablation.set(ablate, ablate_legs)
    _mem_reset(torch)
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            if count is not None:
                with count:
                    means = bridge.kprop_means(w, k_max=3, factor=True,
                                               kind=kind, dtype=dtype)
            else:
                means = bridge.kprop_means(w, k_max=3, factor=True, kind=kind,
                                           dtype=dtype)
    except torch.cuda.OutOfMemoryError as e:
        tel = {"seconds": time.perf_counter() - t0, "oom": True,
               "error": str(e)[:400], **_mem_read(torch)}
        _mem_reset(torch)
        return None, tel
    tel = {"seconds": time.perf_counter() - t0, "oom": False,
           "finite": bool(np.isfinite(means).all()), **_mem_read(torch)}
    return means, tel


# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-dir", default="experiments/2026-08-26_p2_gt")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    ap.add_argument("--drift-tol", type=float, default=1e-11,
                    help="max |re-run SIMPLE - stored P-A| per net before HALT. "
                         "Same derivation as p2_d1_augment_arm.py: ~25x the f32 "
                         "non-determinism floor, 4 orders below the effect.")
    ap.add_argument("--skip-fidelity", action="store_true",
                    help="DIAGNOSTIC ONLY. Skips the bit-exactness check that "
                         "makes the ablation readable; the artifact records it.")
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

    # ---- instrument checks, every one of them before the first datum -------
    v = bridge.verify_upstream(bridge._src())
    if not v["pass"]:
        raise SystemExit(f"REFUSED: upstream is not the pinned rev + patch: {v}")

    from mlp_kprop.kprop_harmonic import SIMPLE, AUGMENT, get_d_max, get_r_x
    shape_check = {
        "d_max_simple": get_d_max(3, SIMPLE), "d_max_augment": get_d_max(3, AUGMENT),
        "r_x_4_simple": get_r_x(4, 3, SIMPLE), "r_x_4_augment": get_r_x(4, 3, AUGMENT),
    }
    assert shape_check["d_max_simple"] == shape_check["d_max_augment"] == 4, shape_check
    assert shape_check["r_x_4_simple"] == 2 and shape_check["r_x_4_augment"] == 1, (
        f"REFUSED: the mechanism claim does not hold in this upstream: {shape_check}")

    import mlp_kprop.factor_k3 as _fk3
    ab_mod, ab_meta = build_ablated_module(_fk3.__file__)
    ablation = _Ablation(ab_mod)

    stored, baselines = {}, {}
    sp = os.path.join(_ROOT, PA_STORED)
    if os.path.exists(sp):
        _pa = json.load(open(sp, encoding="utf-8"))["rows"]
        stored = {int(r["seed"]): float(r["xmse_k3"]) for r in _pa}
        baselines = {int(r["seed"]): {c: float(r[c]) for c in BASELINE_COLUMNS
                                      if c in r} for r in _pa}

    print("=" * 78)
    print("T1  --  AUGMENT-LITE: which CHANNEL of kappa_4 carries the 2.094x?")
    print("=" * 78)
    print(f"  asset             {a.gt_dir}  ({len(seeds)} of {n_all} nets, "
          f"{man['samples_per_half']:,} samples/half)")
    print(f"  label noise/half  {man['gt_noise_per_half_mean']:.4e}")
    print(f"  upstream          {v['rev'][:12]} + recorded patch, verified")
    print(f"  ablated copy      anchors {ab_meta['anchors_applied']}")
    print(f"                    upstream sha {ab_meta['source_sha256'][:16]}...")
    print(f"  dtype             {a.dtype}")
    print(f"  PRE-REGISTERED    full gain {FULL_GAIN:.4f}x   band "
          f"[{PREDICTION_BAND[0]}, {PREDICTION_BAND[1]}]x   "
          f"build >= {BUILD_GATE}x   kill <= {KILL_GATE}x")
    print()

    # ---- the fidelity check: the patched module, flag OFF, must be AUGMENT --
    fidelity = {"skipped": bool(a.skip_fidelity)}
    if not a.skip_fidelity:
        print("-- fidelity: patched module with the flag OFF must be bit-exact --")
        w0, _e, _ss = canonical_mint(man["width"], man["depth"], seeds[0])
        m_ref, _ = _run_arm(bridge, torch, w0, AUGMENT, dtype)
        ablation.install()
        try:
            m_pat, _ = _run_arm(bridge, torch, w0, AUGMENT, dtype,
                                ablation=ablation, ablate=False)
        finally:
            ablation.restore()
        if m_ref is None or m_pat is None:
            raise SystemExit("REFUSED: the fidelity check OOMed. Nothing below "
                             "can be read without it.")
        exact = bool(np.array_equal(m_ref, m_pat))
        maxabs = float(np.max(np.abs(m_ref - m_pat)))
        fidelity.update({"bit_exact": exact, "max_abs_diff": maxabs,
                         "seed": seeds[0]})
        print(f"   bit-exact {exact}   max |diff| {maxabs:.3e}")
        if not exact:
            raise SystemExit(
                "REFUSED: the ablated module is not bit-identical to upstream "
                "with the flag OFF. The patch changes something other than what "
                "it claims, so the lite arm would not be a clean ablation.")
        print()

    ablation.install()
    rows, t0 = [], time.perf_counter()
    col_counts = {}
    try:
        for i, s in enumerate(seeds):
            w, _e, _ss = canonical_mint(man["width"], man["depth"], s)
            Af, Bf = A_all[i, -1, :], B_all[i, -1, :]
            row = {"seed": s}

            for tag, kind_name, ablate, ablate_legs in ARMS:
                kind = {"SIMPLE": SIMPLE, "AUGMENT": AUGMENT}[kind_name]
                counter = _ColumnCounter(ablation) if i == 0 else None
                m, tel = _run_arm(bridge, torch, w, kind, dtype,
                                  ablation=ablation, ablate=ablate,
                                  ablate_legs=ablate_legs, count=counter)
                row[tag] = tel
                if counter is not None:
                    col_counts[tag] = counter.cols
                if m is not None:
                    row[f"xmse_{tag}"] = xmse(m[-1], Af, Bf)
                    row[f"plain_mse_{tag}_halfA"] = float(np.mean((m[-1] - Af) ** 2))
                del m

            if "xmse_simple" in row and s in stored:
                row["drift_vs_pa_k3"] = row["xmse_simple"] - stored[s]
                if abs(row["drift_vs_pa_k3"]) > a.drift_tol:
                    print(f"   *** HALT: seed {s} SIMPLE drifted "
                          f"{row['drift_vs_pa_k3']:+.3e} from the stored P-A "
                          f"(tol {a.drift_tol:.0e}). The asset, the mint, the "
                          f"upstream checkout or the bridge has moved.")
                    rows.append(row)
                    raise SystemExit(2)

            rows.append(row)
            _progress((i + 1) / len(seeds))
            el = time.perf_counter() - t0
            print("   [%2d/%d] seed %d  %s   %.1fs/net" % (
                i + 1, len(seeds), s,
                "  ".join("%s %.4e" % (t, row.get("xmse_" + t, float("nan")))
                          for t, _k, _b, _l in ARMS),
                el / (i + 1)))
    finally:
        ablation.restore()

    # ---- the result ---------------------------------------------------------
    tags = [t for t, _k, _b, _l in ARMS]
    ok = [r for r in rows if all(f"xmse_{t}" in r for t in tags)]
    # Attach the Gaussian baselines for the SAME networks, in the same order.
    have_base = baselines and all(r["seed"] in baselines for r in ok)
    arr = {t: np.array([r[f"xmse_{t}"] for r in ok], dtype=np.float64)
           for t in tags}

    res = {}
    for t in tags:
        m, se, ci = studentised_t(arr[t])
        res[t] = {"mean_xmse": m, "se": se, "ci95": list(ci), "n": len(arr[t])}

    # The binding statistic is the paired ARITHMETIC difference with its
    # studentised-t interval. The gain RATIOS below are reported because the
    # gate is written on one, and they are ratios OF MEANS -- never a mean of
    # ratios, which `docs/VALIDATION.md` §2b-i refuses.
    for t in tags[1:]:
        d = arr["simple"] - arr[t]
        m, se, ci = studentised_t(d)
        res[f"simple_minus_{t}"] = {"mean_diff": m, "se": se, "ci95": list(ci),
                                    "n_better": int(np.sum(d > 0)), "n": len(d)}
    for t in ("lite", "legsonly", "minimal"):
        d = arr[t] - arr["augment"]
        m, se, ci = studentised_t(d)
        res[f"{t}_minus_augment"] = {"mean_diff": m, "se": se, "ci95": list(ci),
                                     "n_better": int(np.sum(d < 0)), "n": len(d)}

    if have_base:
        for c in BASELINE_COLUMNS:
            arr[c] = np.array([baselines[r["seed"]][c] for r in ok],
                              dtype=np.float64)
            m, se, ci = studentised_t(arr[c])
            res[c] = {"mean_xmse": m, "se": se, "ci95": list(ci), "n": len(ok)}
        # THE COMPARISON THE FIRST VERSION OMITTED: is an ablated arm simply the
        # Gaussian baseline? An arm sitting on it carries no information about
        # which kappa_4 channel carried anything.
        for t in ("lite", "legsonly", "minimal"):
            for c in BASELINE_COLUMNS:
                d = arr[c] - arr[t]
                m, se, ci = studentised_t(d)
                res[f"{c}_minus_{t}"] = {
                    "mean_diff": m, "se": se, "ci95": list(ci),
                    "arm_better_than_baseline": int(np.sum(d > 0)), "n": len(d),
                    "ratio_of_means": float(arr[c].mean() / arr[t].mean()),
                    "straddles_zero": bool(ci[0] < 0.0 < ci[1]),
                }

    gain = {t: float(arr["simple"].mean() / arr[t].mean()) for t in tags[1:]}
    gain_augment = gain["augment"]

    def retention_of(t):
        return ((gain[t] - 1.0) / (gain_augment - 1.0)
                if gain_augment > 1 else float("nan"))

    retention = {t: retention_of(t) for t in ("lite", "legsonly", "minimal")}

    # THE GATE IS ON `minimal`, and the contract says why: it is the ONLY arm
    # that births SIMPLE's `2n` columns and can therefore be afforded. `lite` is
    # the channel attribution -- it births `5n` and is priced, not shipped.
    g = gain["minimal"]
    verdict = ("BUILD" if g >= BUILD_GATE
               else "H1 DEAD" if g <= KILL_GATE
               else "INCONCLUSIVE BAND")

    print()
    print("-" * 78)
    for t in tags:
        print(f"   {ARM_LABEL[t]:26s} {res[t]['mean_xmse']:.6e}  "
              f"95% [{res[t]['ci95'][0]:.4e}, {res[t]['ci95'][1]:.4e}]")
    print()
    print(f"   SIMPLE / AUGMENT           {gain_augment:.4f}x   "
          f"(anchor {FULL_GAIN:.4f}x)")
    for t in ("lite", "legsonly", "minimal"):
        print(f"   SIMPLE / {ARM_LABEL[t]:22s} {gain[t]:.4f}x   "
              f"retains {retention.get(t, float('nan')):+.3f} of the full gain")
    if have_base:
        print()
        print("   AGAINST THE GAUSSIAN BASELINES, same seeds, same labels:")
        for c in BASELINE_COLUMNS:
            print(f"     {c:12s} {res[c]['mean_xmse']:.6e}")
        for t in ("lite", "legsonly", "minimal"):
            for c in BASELINE_COLUMNS:
                k = f"{c}_minus_{t}"
                v = res[k]
                print(f"     {c} - {t:9s} {v['mean_diff']:+.4e}  "
                      f"95% [{v['ci95'][0]:+.3e}, {v['ci95'][1]:+.3e}]  "
                      f"ratio {v['ratio_of_means']:.4f}  "
                      f"{'STRADDLES ZERO -- the arm IS this baseline' if v['straddles_zero'] else ''}")
    print(f"   GATE (on MINIMAL, the affordable arm)   build >= {BUILD_GATE}x"
          f"   kill <= {KILL_GATE}x   ->  {verdict}")
    print()
    if col_counts:
        print("   lineage columns handed on, per layer (net 0):")
        for tag, cols in col_counts.items():
            print(f"     {tag:12s} {cols}")
    print()

    control = {
        "simple_anchor": SIMPLE_ANCHOR,
        "simple_measured": res["simple"]["mean_xmse"],
        "simple_drift": res["simple"]["mean_xmse"] - SIMPLE_ANCHOR,
        "augment_anchor": AUGMENT_ANCHOR,
        "augment_measured": res["augment"]["mean_xmse"],
        "augment_drift": res["augment"]["mean_xmse"] - AUGMENT_ANCHOR,
    }
    print(f"   control drift  SIMPLE {control['simple_drift']:+.3e}   "
          f"AUGMENT {control['augment_drift']:+.3e}")

    rec = {
        "artifact_kind": "t1_augment_lite_channel_ablation",
        "contract": "experiments/LOG.md 2026-09-10 CONTRACT (Rule 1): T1",
        "gt_dir": a.gt_dir,
        "gt_manifest": man,
        "upstream": v,
        "ablation": ab_meta,
        "fidelity": fidelity,
        "dtype": a.dtype,
        "pre_registered": {"full_gain": FULL_GAIN, "band": list(PREDICTION_BAND),
                           "build_gate": BUILD_GATE, "kill_gate": KILL_GATE,
                           "gate_is_on": "minimal"},
        "arms": [{"tag": t, "kind": k, "ablate_k4_births": b,
                  "ablate_extra_legs": l, "label": ARM_LABEL[t]}
                 for t, k, b, l in ARMS],
        "results": res,
        "gain": gain,
        "retention": retention,
        "verdict": verdict,
        "controls": control,
        "lineage_columns_per_layer": col_counts,
        "gaussian_baselines_attached": bool(have_base),
        "t_crit": dict(_LAST_T_CRIT),
        "rows": rows,
        "seed_ledger": "no block named",
    }
    if a.out:
        out = a.out if os.path.isabs(a.out) else os.path.join(_ROOT, a.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2)
        print(f"   artifact  {os.path.relpath(out, _ROOT)}")
    _progress(1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
