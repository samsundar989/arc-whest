"""SLICE 4, ITEM 5 -- the small-width ladder at fixed L/n = 1/64.

*Strategist handoff `docs/STRATEGIST_RESPONSE_2026-09-09.md` section 2 Q2 and
section 3 item 5.*

WHAT THIS DECIDES, AND WHY IT NEEDS NO 1024-WIDTH ORACLE
---------------------------------------------------------
Two questions have been open since Slice 3 and both were blocked on an oracle at
the live shape that nobody can afford:

1. **Does a traceless kappa_4 carrier (SSC-class) have a ceiling above AUGMENT?**
   `AUGMENT` carries only the part of kappa_4 an `r=1` HTensor can hold. D2's own
   scope caveat is explicit that a small `(c)/K3` kills the trace-matrix family
   and does **not** kill a traceless carrier. The ceiling of *every* kappa_4
   carrier, traceless included, is **full K4** -- so `full-K4 / AUGMENT` is the
   number that bounds the lane.
2. **How much does upstream's `factor=True` cost in accuracy?** Upstream warns
   (`kprop_harmonic.py:538`) that factored AUGMENT drops the augmented diagrams
   that do not factor, so it is NOT the same estimator as unfactored AUGMENT.
   That gap has never been measured at any width, and `docs/GAME_PLAN.md` WP3
   calls the unfactored path the only admissible parity target.

ARC's conjecture `MSE_K ~ c_K (L/n)^K` makes the RATIO between methods a function
of `L/n` rather than of `n` alone. So `(n, L)` in `{(128, 2), (192, 3), (256, 4)}`
holds `L/n = 1/64` fixed -- the live shape's value -- while `n^4` stays
affordable. If `full-K4 / AUGMENT` is small at `(256, 4)` and flat or falling
along the ladder, the traceless lane is dead at `1024 x 16` and no
representation-faithful oracle at 1024 is needed.

TIER AND SEEDS -- read this before quoting anything from here
--------------------------------------------------------------
**DESIGN. It orients priority and it promotes NOTHING**, exactly as D2 was
labelled. `# seed-ledger: no block named` -- and it cannot contaminate one,
because every shape here is a PROXY shape (`128/192/256` wide, `2/3/4` deep) and
no live-shape (`1024 x 16`) seed is consumed. The seeds are a fixed, documented
range so the ladder is reproducible.

**Every ratio here is measured at L/n = 1/64 and at n <= 256. Extrapolating it to
n = 1024 is an inference, and the ladder's own trend across three rungs is the
only evidence offered for that inference.** If the ratio moves with `n` along the
ladder, say so and do not extrapolate.

LABELS
------
Two INDEPENDENT Monte-Carlo replicas per network, and the two-half cross
estimator `xmse(p) = mean_j (p_j - A_j)(p_j - B_j)`, whose expectation is
`MSE(p)` exactly because `E[e_A e_B] = 0`. A single-replica MSE is biased upward
by the label's own variance, uniformly across arms; that bias cancels in a paired
comparison but not in the absolute numbers this module prints. The replica noise
is printed beside every MSE, per the measurement law.

Run (GPU venv; submit through workerq, see docs/SETUP_WINDOWS.md section 7):
    .venv-gpu\\Scripts\\python.exe local_sim/p2_small_width_ladder.py --nets 24
    .venv-gpu\\Scripts\\python.exe local_sim/p2_small_width_ladder.py --report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

OUT_DIR = os.path.join(_ROOT, "experiments", "2026-09-09_slice4")
OUT = os.path.join(OUT_DIR, "small_width_ladder.json")

#: L/n = 1/64 at every rung -- the live shape's ratio.
RUNGS = ((128, 2), (192, 3), (256, 4))

#: Fixed, documented, and deliberately outside every registered live-shape block.
SEED_BASE = 9_100_000

#: MC samples per replica. SET FROM A MEASUREMENT, not from habit: a 2^16 smoke
#: run read per-replica label noise `6.0e-06` against an AUGMENT signal of
#: `~5e-08` at `(256, 4)`, and the cross estimator went NEGATIVE -- unbiased, and
#: swamped. Noise falls as `1/N`, so `2^24` puts it at `~2.3e-08`, below the
#: signal, and 24 nets x n neurons then resolve a 1.3x ratio. `2^22` would NOT
#: have been enough, and the same trap fired on D2's refusal gate this slice.
SAMPLES = 2 ** 24
CHUNK = 2 ** 16

#: Arms. `full K4` is the CEILING of every kappa_4 carrier, traceless included --
#: and that is CHECKED, not assumed: `factored_keeps_term(4, ...)` is True for
#: 171/171 SIMPLE terms (factoring drops nothing outside AUGMENT) and
#: `FactoredTensor4`'s rank only ever grows by `torch.cat`, never truncates.
#: `get_vec_cond`'s budget is `k_max - 1 + augment`, so k3-AUGMENT and k4-SIMPLE
#: share the IDENTICAL diagram budget of 3; the only difference between their
#: term sets is the all-distinct degree-4 slice `(1,1,1,1)` (0 terms vs 29).
#: So this is a clean representation comparison and not a raised-budget one --
#: which is exactly the over-attribution that bit D1's first reading.
#:
#: `use_avg_metric` ASYMMETRY, added as its own arm after a Skeptic:
#: `metric_needed(4, SIMPLE)` is False, so the k4 arm never pays the
#: `E[WW^T] = 2I` approximation that BOTH k3 arms pay. Left unmeasured, that is
#: a confound in the headline ratio, so `k3_augment_exactmetric` isolates it.
ARMS = (
    ("k3_simple", dict(k_max=3, factor=True, kind="SIMPLE")),
    ("k3_augment", dict(k_max=3, factor=True, kind="AUGMENT")),
    ("k3_augment_unfactored", dict(k_max=3, factor=False, kind="AUGMENT")),
    ("k3_augment_exactmetric",
     dict(k_max=3, factor=True, kind="AUGMENT",
          extra={"use_avg_metric": False})),
    # THE ARM THAT CAN ACTUALLY SHIP. The first pass measured the exact metric
    # only on AUGMENT, which a Skeptic pointed out is the one kind with no
    # budget -- so a 23% lever was read off an arm that cannot be built.
    # `metric_needed(3, SIMPLE)` is True, so the exact metric is live for SIMPLE
    # too and this is where the question actually is.
    ("k3_simple_exactmetric",
     dict(k_max=3, factor=True, kind="SIMPLE",
          extra={"use_avg_metric": False})),
    ("k4_factored", dict(k_max=4, factor=True, kind="SIMPLE")),
)


def xmse(pred, A, B):
    """Unbiased two-half cross estimator. E[(p-A)(p-B)] = MSE(p) exactly."""
    return float(np.mean((pred - A) * (pred - B)))


def _t975(df):
    from competition_score import _t975 as t
    return t(df)


def paired(a, b):
    """Paired arithmetic difference `a - b` with a studentised-t interval.

    The binding promotion statistic in this repository is the paired ARITHMETIC
    difference (`docs/VALIDATION.md` section 2b-i); the geometric mean of
    per-net ratios is a stability DIAGNOSTIC with no authority. Both are
    reported and labelled.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    h = _t975(n - 1) * se if n > 1 else float("nan")
    ratios = np.asarray(a, float) / np.asarray(b, float)
    return {
        "paired_diff": m, "se": se, "ci95": [m - h, m + h],
        "n": n, "n_a_worse": int((d > 0).sum()),
        "arithmetic_ratio": float(np.mean(a) / np.mean(b)),
        "geometric_ratio_DIAGNOSTIC": float(np.exp(np.mean(np.log(ratios)))),
    }


def _truth_two_replicas(weights, sample_ss, width, depth, device, samples, chunk):
    """All-layer post-ReLU means from two INDEPENDENT sampling streams."""
    import torch

    w = [torch.from_numpy(np.ascontiguousarray(x)).to(device) for x in weights]
    # The evaluator's own sampling substream, spawned into two disjoint children.
    kids = sample_ss.spawn(2)
    out = []
    for ss in kids:
        seed64 = int(ss.generate_state(2, dtype=np.uint64)[0] % (2 ** 63 - 1))
        gen = torch.Generator(device=device)
        gen.manual_seed(seed64)
        acc = [torch.zeros(width, dtype=torch.float64, device=device)
               for _ in range(depth)]
        acc_sq = [torch.zeros(width, dtype=torch.float64, device=device)
                  for _ in range(depth)]
        done = 0
        while done < samples:
            m = min(chunk, samples - done)
            h = torch.randn((m, width), generator=gen, device=device,
                            dtype=torch.float32)
            for k in range(depth):
                h = torch.relu(h @ w[k])
                acc[k] += h.sum(dim=0, dtype=torch.float64)
                acc_sq[k] += (h.to(torch.float64) ** 2).sum(dim=0)
            done += m
        means = np.stack([(a / samples).cpu().numpy() for a in acc])
        ez2 = np.stack([(a / samples).cpu().numpy() for a in acc_sq])
        out.append((means, np.maximum(ez2 - means ** 2, 0.0)))
    (mA, vA), (mB, vB) = out
    # Per-replica noise on the MEAN, which is what the labels are.
    noise = float(np.mean(0.5 * (vA[-1] + vB[-1])) / samples)
    return mA, mB, noise


def _progress(frac):
    p = os.environ.get("WORKERQ_PROGRESS")
    if not p:
        return
    try:
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(str(min(max(frac, 0.0), 1.0)))
    except OSError:
        pass


def run(a) -> int:
    sys.path.insert(0, a.upstream)
    import torch
    import p2_kprop_bridge as BRIDGE
    from mint_provenance import canonical_mint
    from mlp_kprop.kprop_harmonic import SIMPLE, AUGMENT

    kinds = {"SIMPLE": SIMPLE, "AUGMENT": AUGMENT}
    v = BRIDGE.verify_upstream(a.upstream)
    if not v["pass"]:
        raise SystemExit("REFUSED: upstream is not the pinned object: %r" % v)
    print("  upstream rev %s VERIFIED\n" % v["rev"][:8])

    total = len(RUNGS) * a.nets
    done_units = 0
    rungs_out = []
    for (width, depth) in RUNGS:
        print("=" * 74)
        print("  RUNG  n=%d  L=%d   L/n = %.5f" % (width, depth, depth / width))
        print("=" * 74)
        rows = []
        for i in range(a.nets):
            seed = SEED_BASE + 1000 * width + i
            weights, _est, sample_ss = canonical_mint(width, depth, seed)
            t0 = time.perf_counter()
            A, B, noise = _truth_two_replicas(
                weights, sample_ss, width, depth, a.device, a.samples, a.chunk)
            row = {"seed": seed, "gt_noise_per_replica": noise}
            for name, kw in ARMS:
                if name in a.skip:
                    continue
                try:
                    # `no_grad` is REQUIRED, not an optimisation: upstream's
                    # `factor_k4` uses `out=` kwargs, and an `nn.Linear` weight
                    # still carrying `requires_grad` makes torch refuse them
                    # outright ("functions with out=... arguments don't support
                    # automatic differentiation"). Without this the k4 arm does
                    # not fail quietly -- it does not run at all.
                    with torch.no_grad():
                        m = BRIDGE.kprop_means(
                            weights, k_max=kw["k_max"], factor=kw["factor"],
                            kind=kinds[kw["kind"]], device=a.device,
                            dtype=getattr(torch, a.dtype),
                            **kw.get("extra", {}))
                except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                    row[name] = None
                    row[name + "_error"] = "%s: %s" % (type(exc).__name__, exc)
                    torch.cuda.empty_cache()
                    continue
                row[name] = xmse(m[-1], A[-1], B[-1])
                row[name + "_all_layers"] = xmse(m, A, B)
            rows.append(row)
            done_units += 1
            _progress(done_units / total)
            got = " ".join("%s %.3e" % (k, row[k]) for k, _ in ARMS
                           if row.get(k) is not None)
            print("    seed %d  (%.1fs)  gt_noise %.2e  %s"
                  % (seed, time.perf_counter() - t0, noise, got))
        rungs_out.append({"width": width, "depth": depth,
                          "L_over_n": depth / width, "rows": rows})

    doc = {
        "slice": "Slice 4 item 5 -- small-width ladder at fixed L/n = 1/64",
        "handoff": "docs/STRATEGIST_RESPONSE_2026-09-09.md section 2 Q2 / section 3 item 5",
        "tier": ("DESIGN. Orients priority; promotes NOTHING. Proxy shapes only; "
                 "no live-shape (1024x16) seed consumed."),
        # PROVENANCE. The first artifact this module wrote recorded no dtype, no
        # device, no timestamp and no job id, which is how an f64 run came to be
        # compared against an f32 anchor with nothing on disk to reveal it.
        "dtype": a.dtype, "device": a.device,
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workerq_job": os.environ.get("WORKERQ_JOB_ID"),
        "live_shape_anchor": {
            "SIMPLE_over_AUGMENT_at_1024x16": 2.0941,
            "source": "experiments/2026-09-08_slice3/d1_augment.json, dtype float32",
            "role": ("prediction 3's validity check. The rungs must reproduce it "
                     "IN THE SAME DTYPE or nothing here extrapolates."),
        },
        "seed_base": SEED_BASE, "samples_per_replica": a.samples,
        "nets_per_rung": a.nets, "arms": [n for n, _ in ARMS],
        "upstream": v, "rungs": rungs_out,
        "predictions_pre_registered": {
            "source": "Samuel / strategist, recorded before any data existed",
            "full_K4_over_AUGMENT_at_256x4": [2.0, 4.0],
            "unfactored_over_factored_AUGMENT_at_256x4": [1.1, 1.3],
        },
        "scope_caveat": ("Every ratio is measured at L/n = 1/64 and n <= 256. "
                         "Extrapolation to n = 1024 is an INFERENCE; the ladder's "
                         "own trend across three rungs is the only evidence for it."),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print("\n  wrote %s" % os.path.relpath(a.out, _ROOT))
    _progress(1.0)
    return report(a)


def report(a) -> int:
    with open(a.out, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    print("\n" + "=" * 74)
    print("  LADDER  --  ratios along fixed L/n = 1/64")
    print("=" * 74)
    pred = doc["predictions_pre_registered"]
    print("  pre-registered: full-K4/AUGMENT %.1f-%.1fx at (256,4); "
          "unfactored/factored %.1f-%.1fx"
          % (pred["full_K4_over_AUGMENT_at_256x4"][0],
             pred["full_K4_over_AUGMENT_at_256x4"][1],
             pred["unfactored_over_factored_AUGMENT_at_256x4"][0],
             pred["unfactored_over_factored_AUGMENT_at_256x4"][1]))
    # The label says which way a ratio BELOW 1 reads, because the first pass at
    # this table read `unfactored/factored = 0.948` as "the factored arm is
    # slightly better". It is the opposite: the numerator is the better arm, so
    # 0.948 means `factor=True` COSTS 5.2% of accuracy. A ratio without its
    # direction spelled out is a sign error waiting to happen.
    pairs = (
        ("k3_simple", "k3_augment", "SIMPLE / AUGMENT       (>1: AUGMENT wins)"),
        ("k3_augment", "k4_factored", "AUGMENT / full-K4      (>1: full K4 wins)"),
        ("k3_augment_unfactored", "k3_augment",
         "unfactored / factored  (<1: factor=True COSTS)"),
        ("k3_augment_exactmetric", "k3_augment",
         "exact-metric / avg AUG (<1: use_avg_metric COSTS)"),
        ("k3_simple_exactmetric", "k3_simple",
         "exact-metric / avg SIM (<1: use_avg_metric COSTS)  <- the shippable one"),
    )
    for rung in doc["rungs"]:
        rows = rung["rows"]
        print("\n  n=%d L=%d   %d nets   gt_noise %.3e"
              % (rung["width"], rung["depth"], len(rows),
                 float(np.mean([r["gt_noise_per_replica"] for r in rows]))))
        for name in doc["arms"]:
            vals = [r[name] for r in rows if r.get(name) is not None]
            if not vals:
                err = next((r.get(name + "_error") for r in rows
                            if r.get(name + "_error")), "absent")
                print("      %-24s  --   %s" % (name, err))
                continue
            print("      %-24s  mean xmse %.4e   over %d nets"
                  % (name, float(np.mean(vals)), len(vals)))
        for a_key, b_key, label in pairs:
            va = [r.get(a_key) for r in rows]
            vb = [r.get(b_key) for r in rows]
            ok = [(x, y) for x, y in zip(va, vb) if x is not None and y is not None]
            if len(ok) < 2:
                continue
            st = paired([x for x, _ in ok], [y for _, y in ok])
            print("      %-24s  arithmetic %.3fx   paired diff %+.3e  "
                  "95%% [%+.3e, %+.3e]   geo %.3fx (diagnostic)"
                  % (label, st["arithmetic_ratio"], st["paired_diff"],
                     st["ci95"][0], st["ci95"][1],
                     st["geometric_ratio_DIAGNOSTIC"]))
    print("\n  %s" % doc["scope_caveat"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nets", type=int, default=24)
    ap.add_argument("--samples", type=int, default=SAMPLES)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--device", default="cuda")
    # float32 IS THE COMPARABLE CHOICE. The whole ladder is validated against
    # the `2.094x` SIMPLE/AUGMENT measured at the live 1024x16 shape by D1, and
    # `d1_augment.json` records `"dtype": "float32"`. The first run of this
    # module hard-coded float64 and then compared its 2.9 against that 2.09 --
    # a cross-instrument comparison, and the artifact did not even record which
    # instrument it used. A Skeptic caught it. Record the dtype in the artifact.
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    ap.add_argument("--skip", nargs="*", default=[],
                    help="arm names to skip (e.g. k4_factored)")
    ap.add_argument("--report", action="store_true",
                    help="re-print from the stored artifact, run nothing")
    ap.add_argument("--upstream",
                    default=r"C:\Users\samsu\Documents\mlp_cumulant_propagation\src")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    return report(a) if a.report else run(a)


if __name__ == "__main__":
    raise SystemExit(main())
