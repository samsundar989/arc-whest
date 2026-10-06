"""P1 modeled FLOP factor of decided-neuron pruning (HANDOFF 2026-10-06 Lane P).

    uv run python tools/mask_flops.py --stats experiments/2026-10-06_mask/grid/fit_ctl.json

Scales every site of the v72/v73 per-site census (`experiments/2026-09-28_cost_sites/
profile.json`, one net, 472.2 G, line numbers identical in v73) by the kept fractions a
compact chain would see, per net, from the alpha statistics `twin.py eval --mask-stats`
logs. A MODEL, not a meter: it says what shrinks, assuming every compacted product is
billed at its kept shape, and charges nothing for gathers/scatters.

Rules (l = the layer the site runs at; b = a lineage's birth layer):
  R(l)   rows kept after layer l's Wick scale (alpha_l >= -tau_R)
  A(b)   A_L columns kept: H -> |alpha_b| <= tau_H ; Hk -> alpha_b >= -tau_H
  M(b)   M columns kept (two-leg middle or slice column): alpha_b >= -tau_H
  D(l)   D21 rows kept: alpha_l >= -tau_D
  recent transport A_L        R(l-1) x mean A(b) over the recent lineages
  recent transport M (older)  R(l-1) x mean M(b)
  thin Z / Y / old Q transport R(l-1)
  recent D21 readout          D(l) Dc(l) x mean (A + M)/2 over the recent lineages
  thin D21 readout            D(l) Dc(l)
  old D21 readout (764)       D(l) x mean old (A + M)/2 ; (766) D(l) Dc(l)
  Dc(l) = D(l) when rows AND columns are masked, 1 for row-only D (Dr)
  old leg reconstruction      mean old (A + M)/2   (rows stay full: D3 reads every row)
  old compression (join)      R(l) x (A + M)/2 of the joiner
  everything else             1 (covariance, closure, corrector, gates, P2 birth,
                              mixed elementwise D3/P2/hub readouts -- conservative)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CENSUS = ROOT / "experiments" / "2026-09-28_cost_sites" / "profile.json"
N = 1024

CONFIGS = {
    "R2.0": {"R": 2.0}, "R2.5": {"R": 2.5},
    "H2.5": {"H": 2.5}, "H3.0": {"H": 3.0}, "H3.5": {"H": 3.5},
    "Hk2.5": {"H": 2.5, "Hk": 1}, "Hk3.0": {"H": 3.0, "Hk": 1}, "Hk3.5": {"H": 3.5, "Hk": 1},
    "D2.5": {"D": 2.5}, "D3.0": {"D": 3.0},
    "safe": {"R": 2.0, "H": 3.0, "D": 3.0}, "safeK": {"R": 2.0, "H": 3.0, "Hk": 1, "D": 3.0},
    "aggr": {"R": 2.0, "H": 2.5, "D": 2.5}, "aggrK": {"R": 2.0, "H": 2.5, "Hk": 1, "D": 2.5},
    # Hd: every H part on dead columns only -> A = M = alpha >= -tau (same cost as Hk)
    "Hd2.5": {"H": 2.5, "Hk": 1}, "Hd3.0": {"H": 3.0, "Hk": 1},
    "R2.0/Hd3.0/D3.0": {"R": 2.0, "H": 3.0, "Hk": 1, "D": 3.0},
    "R2.0/Hd2.5/D2.5": {"R": 2.0, "H": 2.5, "Hk": 1, "D": 2.5},
    "R2.0/H2.5/D3.0": {"R": 2.0, "H": 2.5, "D": 3.0},
    # row-only D21 (Dr): readouts scale by kept rows, not rows squared
    "Dr2.5": {"D": 2.5, "Dr": 1}, "Dr3.0": {"D": 3.0, "Dr": 1},
    "safe-r R2.0/Hd3.0/Dr3.0": {"R": 2.0, "H": 3.0, "Hk": 1, "D": 3.0, "Dr": 1},
    "aggr-r R2.0/Hd2.5/Dr2.5": {"R": 2.0, "H": 2.5, "Hk": 1, "D": 2.5, "Dr": 1},
    "lean-r R2.5/Hd2.5/Dr2.5": {"R": 2.5, "H": 2.5, "Hk": 1, "D": 2.5, "Dr": 1},
    # no H: R and row-only D keep the uniform lineage width (cheap to build)
    "rd R2.0/Dr2.5": {"R": 2.0, "D": 2.5, "Dr": 1},
    "rd3 R2.0/Dr3.0": {"R": 2.0, "D": 3.0, "Dr": 1},
    "rdl R2.5/Dr2.5": {"R": 2.5, "D": 2.5, "Dr": 1},
}


def fractions(stats, cfg, depth=16):
    """Per-layer kept fractions (R, A, M, D) for one net's alpha statistics."""
    by_l = {s["l"]: s for s in stats}

    def dead(l, t):
        return by_l[l][f"dead{t:.1f}"] if l in by_l else 0.0

    def dec(l, t):
        return by_l[l][f"dec{t:.1f}"] if l in by_l else 0.0

    R = [1.0 - dead(l, cfg["R"]) if "R" in cfg else 1.0 for l in range(depth)]
    if "H" in cfg:
        M = [1.0 - dead(b, cfg["H"]) for b in range(depth)]
        A = M[:] if cfg.get("Hk") else [1.0 - dec(b, cfg["H"]) for b in range(depth)]
    else:
        A = M = [1.0] * depth
    D = [1.0 - dead(l, cfg["D"]) if "D" in cfg else 1.0 for l in range(depth)]
    Dc = [1.0] * depth if cfg.get("Dr") else D      # row-only D keeps every column
    return R, A, M, D, Dc


def fixed_k(frs, q=99.9, margin=32, mult=64):
    """Per-layer kept fraction -> the fixed count a shape-only table would ship."""
    out = []
    for role in zip(*frs):                           # role: per-net lists of per-layer fractions
        arr = np.array(role)                         # (nets, depth)
        k = np.percentile(arr * N, q, axis=0) + margin
        k = np.minimum(np.ceil(k / mult) * mult, N)
        out.append([float(x) / N for x in k])
    return tuple(out)


def factor(sites, R, A, M, D, Dc):
    tot = kept = 0.0
    recent_k = {}
    for s in sites:                                  # recent count per layer, from 755
        if s["line"] == 755 and s["caller"] == "_shared_contract_lean":
            recent_k[s["layer"]] = s["matrix_shapes"][0][1] // (2 * N)
    for s in sites:
        f, l, line = s["exclusive_flops"], s["layer"], s["line"]
        tot += f
        if l is None or l < 1:
            kept += f
            continue
        k = recent_k.get(l, 0)
        rec = list(range(max(0, l - k), l))          # recent births (incl. youngest)
        old = list(range(0, max(0, l - k)))
        am = lambda bs: float(np.mean([(A[b] + M[b]) / 2 for b in bs])) if bs else 1.0
        if line == 1163:
            nk = s["matrix_shapes"][1][1] // N
            g = R[l - 1] * float(np.mean([A[b] for b in range(l - nk, l)]))
        elif line == 1164:
            nk = s["matrix_shapes"][1][1] // N
            g = R[l - 1] * float(np.mean([M[b] for b in range(l - 1 - nk, l - 1)]))
        elif line in (1170, 1174, 1177):
            g = R[l - 1]
        elif line == 755:
            g = D[l] * Dc[l] * am(rec)
        elif line in (1425, 1429, 766):
            g = D[l] * Dc[l]
        elif line == 764:
            g = D[l] * am(old)
        elif line == 739:
            g = am(old)
        elif s["category"] == "old.compression":
            g = R[l] * am([max(0, l - k)])
        else:
            g = 1.0
        kept += f * g
    return kept / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True, help="a mask_grid JSON with per-net stats")
    ap.add_argument("--json", default="")
    ap.add_argument("--q", type=float, default=99.9, help="--fixed-k percentile")
    ap.add_argument("--margin", type=float, default=32)
    ap.add_argument("--mult", type=int, default=64)
    ap.add_argument("--round", type=int, default=0,
                    help="P2 variant B: per-net count rounded UP to this multiple")
    ap.add_argument("--ktab-cfg", default="", help="write the median-K table for this config")
    ap.add_argument("--ktab-out", default="")
    ap.add_argument("--fixed-k", action="store_true",
                    help="P2 variant A: per-layer keep count = ceil64(q99.9(kept) + 32) "
                         "over these nets, the same for every net")
    a = ap.parse_args()
    census = json.loads(CENSUS.read_text())
    sites = census["call_sites"]
    for s in sites:
        s["layer"] = None if s.get("layer") in (None, "") else int(s["layer"])
    st = json.loads(Path(a.stats).read_text())["stats"]
    if a.ktab_cfg:
        cfg = CONFIGS[a.ktab_cfg]
        R, A, M, D, _ = zip(*[fractions(v, cfg) for v in st.values()])
        def tab(fr):                                  # nearest multiple of --mult, <= N
            med = np.median(np.array(fr) * N, axis=0)
            return [int(min(N, max(a.mult, round(m / a.mult) * a.mult))) for m in med]
        K = {}
        if "R" in cfg:
            K["R"] = tab(R)
        if "H" in cfg:
            K["H"] = tab(M)                           # dead-only H keeps alpha >= -tau
        if "D" in cfg:
            K["D"] = tab(D)
        Path(a.ktab_out).write_text(json.dumps({"config": a.ktab_cfg, "stats": a.stats,
                                                "rule": f"median kept count, nearest {a.mult}",
                                                "K": K}, indent=1))
        print(json.dumps(K))
        return
    base_b = census["total_flops"] / 2 ** 41
    out = {}
    print(f"census {census['total_flops']:,} FLOPs = {base_b:.5f} B; {len(st)} nets")
    print("| config | modeled F factor (mean) | F/B | CV across nets | max/min |")
    print("|---|---:|---:|---:|---:|")
    for name, cfg in CONFIGS.items():
        fr = [fractions(v, cfg) for v in st.values()]
        if a.fixed_k:                                # one table for all nets
            fr = [fixed_k(fr, a.q, a.margin, a.mult)] * len(fr)
        elif a.round:
            fr = [tuple([min(1.0, math.ceil(x * N / a.round) * a.round / N) for x in r]
                        for r in f) for f in fr]
        fs = np.array([factor(sites, *f) for f in fr])
        out[name] = {"mean": float(fs.mean()), "cv": float(fs.std() / fs.mean()),
                     "maxmin": float(fs.max() / fs.min()), "per_net": fs.tolist()}
        print(f"| {name} | {fs.mean():.4f} | {fs.mean() * base_b:.4f} | "
              f"{100 * fs.std() / fs.mean():.2f} % | {fs.max() / fs.min():.3f} |")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
