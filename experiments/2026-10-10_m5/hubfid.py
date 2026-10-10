"""M5.4 Step 1 -- fidelity of moving the route-(i) kappa4->kappa3 hub out of the lineage legs.

    .venv-gpu/Scripts/python.exe experiments/2026-10-10_m5/hubfid.py --nets 0:5 --out <dir>
    python experiments/2026-10-10_m5/hubfid.py --report <dir>

Route (i) (v62_gb `_layer` hub birth; v81d est:4322-4397): each lineage carries a thin transported
y (= w2 at birth) and a static column scale c_i = (m/8) lam w1 (times the feed gain and the H
mask). Its readouts, in the code's convention (`_dslice21` / `_dslice3` with `_hub_parts`):

    S_h  = A diag(c) M^T + M diag(c) A^T            (the lineage's columns only)
    D21 += (2/3) (y_a S_ab + y_b S_aa / 2)          off-diagonal
    D3  += y_a S_aa

The move replaces S_h by gamma C~_pre (surrogate 1) or g1 C~_pre + g2 W^T diag(v) W (surrogate 2),
v = the incoming post-ReLU variance. Per (net, layer l = 4..14, age) we store the weighted Grams
of [sur1, sur2-C, sur2-N] against the exact contribution, for D21 (weight (w2_a w1_b)^2, how D21
enters the covariance) and D3 (weight w3_a^2). The route-(i) share of the tower's total column
scale is recorded at birth (cA_i / (cA_i + cA_ii)), so the H mask and feed gain carried in CA apply.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from k4reg import BASE, MAIN, REC_PATCH  # noqa: E402

LAYERS = range(4, 15)
HUB_PATCH = ('            _sf = _OWN_S_FEED * _prof_at("feed", layer_index, depth)\n',
             "            if _HUBREC is not None:\n"
             "                _HUBREC[layer_index] = (w1 * (_hm * _lam_h), cA)\n"
             '            _sf = _OWN_S_FEED * _prof_at("feed", layer_index, depth)\n')


def collect(a):
    os.environ.setdefault("TWIN_CACHE", str(MAIN / "cache" / "twin"))
    sys.path.insert(0, str(ROOT / "tools" / "twin"))
    import numpy as np
    import torch
    import twin as T

    T.ARMS["rec"] = [("def _layer(W, state, table, dtype, const, init_scale, last_layer, gens,\n",
                      "_REC = None\n_HUBREC = None\n\n\ndef _layer(W, state, table, dtype, const, "
                      "init_scale, last_layer, gens,\n"), REC_PATCH, HUB_PATCH]
    tw = T.Twin(Path(BASE["cand"]), arm="mask+dress+bank+rec")
    tw.install(T.load_params(BASE["params"]))
    m = tw.mod
    m._DRESS = T.parse_dress("all")
    m._BANK = {"star": 1, "starall": 1}
    data = T.Data()
    f64 = torch.float64
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = (int(x) for x in a.nets.split(":"))
    for gi in range(lo, hi):
        fp = out / f"net_{gi:03d}.json"
        if fp.exists():
            continue
        t0 = time.time()
        m._MASK = T.parse_mask(BASE["mask"])
        w, _ = data.net("full", gi)
        rec, hub = {}, {}
        m._REC = lambda l, **kw: (rec.clear(), rec.update(kw))
        m._HUBREC = hub
        n = tw.width
        dev = T.DEV
        off = ~torch.eye(n, dtype=torch.bool, device=dev)
        state = (T.fnp.zeros(n), T.fnp.eye(n), None, 0.0, T.fnp.zeros(n))
        res = {}
        with torch.no_grad():
            for l in range(tw.depth):
                state = tw._step(l, w[l], state)
                if l not in LAYERS or rec.get("adv") is None:
                    continue
                adv = rec["adv"]
                A, M = m._full_legs_lean(adv)
                A, M = A.to(f64), M.to(f64)
                k = A.shape[1] // n
                Y = adv.Y.to(f64)
                CA = adv.CA.to(f64)
                C = rec["cov_pre"].to(f64)
                Win = rec["W"].to(f64)
                v_in = torch.diagonal(rec["cov_in"]).to(f64)
                N = (Win.T * v_in[None, :]) @ Win
                w1, w2, w3 = (rec["w"][kk].to(f64) for kk in ((1, 1), (2, 1), (3, 1)))
                W21 = ((w2[:, None] * w1[None, :]) ** 2) * off
                W3 = w3 ** 2
                per = {}
                tot = {"ex21": 0.0, "ex3": 0.0}
                for age in range(1, k + 1):
                    i = k - age
                    b = l - age                       # birth layer
                    ci_raw, ca_raw = (x.to(f64) for x in hub[b])
                    share = torch.where(ca_raw.abs() > 0, ci_raw / torch.where(
                        ca_raw.abs() > 0, ca_raw, torch.ones_like(ca_raw)), torch.zeros_like(ca_raw))
                    c = CA[i * n:(i + 1) * n] * share
                    Ac, Mc = A[:, i * n:(i + 1) * n], M[:, i * n:(i + 1) * n]
                    y = Y[:, 2 * i]
                    S = (Ac * c[None, :]) @ Mc.T
                    S = S + S.T
                    ex21 = (2.0 / 3.0) * (y[:, None] * S + 0.5 * torch.diagonal(S)[:, None] * y[None, :])
                    ex3 = y * torch.diagonal(S)
                    feats21, feats3 = [], []
                    for Sx in (C, N):
                        feats21.append((2.0 / 3.0) * (y[:, None] * Sx
                                                       + 0.5 * torch.diagonal(Sx)[:, None] * y[None, :]))
                        feats3.append(y * torch.diagonal(Sx))
                    X21 = torch.stack([f[off] for f in feats21], 1)
                    X3 = torch.stack(feats3, 1)
                    e21, e3 = ex21[off], ex3
                    w21, w3v = W21[off], W3
                    per[age] = {
                        "G21": ((X21 * w21[:, None]).T @ X21).cpu().numpy().tolist(),
                        "b21": ((X21 * w21[:, None]).T @ e21).cpu().numpy().tolist(),
                        "e21": float((w21 * e21 * e21).sum()),
                        "G3": ((X3 * w3v[:, None]).T @ X3).cpu().numpy().tolist(),
                        "b3": ((X3 * w3v[:, None]).T @ e3).cpu().numpy().tolist(),
                        "e3": float((w3v * e3 * e3).sum()),
                        "old": bool(i < adv.ko)}
                    tot["ex21"] += float((w21 * e21 * e21).sum())
                    tot["ex3"] += float((w3v * e3 * e3).sum())
                # the hub's size relative to the whole D21 / D3 the chain consumes
                D21 = m._dslice21(adv, T.fnp.ones((n, n)) * off).to(f64)
                tot["D21"] = float((W21 * D21 * D21).sum())
                tot["D3"] = float((W3 * rec["D3"].to(f64) ** 2).sum())
                res[l] = {"ages": per, "tot": tot, "k": k, "ko": int(adv.ko)}
                rec.clear()
        fp.write_text(json.dumps(res))
        print(f"[hubfid] net {gi}: {time.time() - t0:.0f} s", flush=True)
    return 0


def report(a):
    import numpy as np
    d = Path(a.report)
    R = [json.loads(p.read_text()) for p in sorted(d.glob("net_*.json"))]
    lines = ["# M5.4 step 1 -- route-(i) hub fidelity (weighted R^2 of the exact per-age contribution",
             f"# by gamma C~_pre (sur1) and g1 C~ + g2 W^T diag(v) W (sur2); {len(R)} nets pooled)", "",
             "| l | age | D21 sur1 | D21 sur2 | D3 sur1 | D3 sur2 | hub/D21 energy | old |",
             "|---|---|---|---|---|---|---|---|"]
    out = {}
    for l in sorted({int(x) for r in R for x in r}):
        ages = sorted({int(x) for r in R for x in r[str(l)]["ages"]})
        for age in ages:
            acc = {}
            for r in R:
                p = r[str(l)]["ages"][str(age)]
                for kk in ("G21", "b21", "e21", "G3", "b3", "e3"):
                    acc[kk] = acc.get(kk, 0) + np.asarray(p[kk])
            hub_e = sum(r[str(l)]["ages"][str(age)]["e21"] for r in R)
            d21_e = sum(r[str(l)]["tot"]["D21"] for r in R)

            def r2(G, b, e, S):
                G, b = G[np.ix_(S, S)], b[S]
                beta = np.linalg.lstsq(G, b, rcond=None)[0]
                return 1.0 - (e - 2 * beta @ b + beta @ G @ beta) / e, beta.tolist()
            row = {}
            row["d21_1"], g1 = r2(acc["G21"], acc["b21"], acc["e21"], [0])
            row["d21_2"], g2 = r2(acc["G21"], acc["b21"], acc["e21"], [0, 1])
            row["d3_1"], _ = r2(acc["G3"], acc["b3"], acc["e3"], [0])
            row["d3_2"], _ = r2(acc["G3"], acc["b3"], acc["e3"], [0, 1])
            row["gamma1"], row["gamma2"] = g1, g2
            row["hub_share"] = hub_e / max(d21_e, 1e-300)
            old = R[0][str(l)]["ages"][str(age)]["old"]
            out.setdefault(l, {})[age] = row
            lines.append(f"| {l} | {age} | {row['d21_1']:.3f} | {row['d21_2']:.3f} | {row['d3_1']:.3f} "
                         f"| {row['d3_2']:.3f} | {row['hub_share']:.2e} | {'Q' if old else ''} |")
    (d / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (d / "summary.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", default="0:5")
    ap.add_argument("--out", default="C:/Users/samsu/Documents/arc-whest-lane-m/experiments/2026-10-10_m5/hubfid")
    ap.add_argument("--report", default="")
    a = ap.parse_args()
    return report(a) if a.report else collect(a)


if __name__ == "__main__":
    raise SystemExit(main())
