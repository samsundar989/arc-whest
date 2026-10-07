"""P2.2 diagnostic: per-layer magnitude of the chain's kappa4 slices vs the gain-mode templates.

Runs the v77-equivalent twin (no dressing, no gm) on a few VAL nets and, at each layer, records
c^2 = m^T C m / |m|^4 (post-ReLU state), the chain's D4 = m*dG and additive D22, and the
templates 12 c^2 s^4 and 4 c^2 (s_i^2 s_j^2 + 2 C_ij^2) (beta = 1), plus the exact labels' D4 /
D22 where FIT labels exist (cache/twin_labels, FIT rows 0..99 only -> uses full rows 0..2).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path(r"C:/Users/samsu/Documents/arc-whest")
sys.path.insert(0, str(ROOT / "tools" / "twin"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import twin  # noqa: E402

CAND = ROOT / "experiments" / "2026-09-27_gates" / "estimator_k3_v62_gb.py"
INIT = MAIN / "experiments" / "2026-09-27_gates" / "G1bg" / "params_best.npz"
MASK = "R=2.0,Hd=2.5,Dz=2.5,K=" + str(MAIN / "experiments" / "2026-10-06_mask" / "ktab" / "aggr_q50.json")

tw = twin.Twin(CAND, arm="mask")
tw.install(twin.load_params(INIT))
tw.mod._MASK = twin.parse_mask(MASK)
rows = []
orig = tw.mod._closure_apply


def spy(pK, cc, sigma, alpha, blocks, off_mask, last_layer, dtype):
    if not last_layer:
        st = SPY["state"]
        mu, cov = st
        mm = float((mu * mu).sum())
        c2 = float((mu * (cov @ mu)).sum()) / max(mm * mm, 1e-30)
        var = sigma * sigma
        C = blocks["C11"]
        D22 = blocks["D22"]
        tpl22 = ((var[:, None] * var[None, :]) + 2.0 * C * C) * off_mask * (4.0 * c2)
        tpl4 = 12.0 * c2 * var * var
        d = {"l": twin_layer[0], "c2": c2,
             "D4_mean": float(blocks["D4"].mean()), "tpl4_mean": float(tpl4.mean()),
             "D22_absmean": float(D22.abs().mean()), "tpl22_absmean": float(tpl22.abs().mean()),
             "D22_rowsum_mean": float(D22.sum(1).mean()), "tpl22_rowsum_mean": float(tpl22.sum(1).mean()),
             "corr_D22_tpl": float(torch.corrcoef(torch.stack([D22.flatten(), tpl22.flatten()]))[0, 1]),
             "lam": float(blocks["lam"]), "metric": float(blocks["metric"]),
             "Q31raw_absmean": float((C.abs() * blocks["lam"]).mean()),
             "Q31tpl_absmean": float((12.0 * c2 * var[:, None] * C.abs()).mean())}
        rows.append(d)
    return orig(pK, cc, sigma, alpha, blocks, off_mask, last_layer, dtype)


SPY = {}
twin_layer = [0]
orig_layer = tw.mod._layer


def layer_spy(W, state, *a, **k):
    SPY["state"] = (state[0], state[1])
    twin_layer[0] = k.get("layer_index", 0)
    return orig_layer(W, state, *a, **k)


tw.mod._closure_apply = spy
tw.mod._layer = layer_spy
data = twin.Data()
out = {}
with torch.no_grad():
    for i in (900, 901, 902):
        rows.clear()
        w, truth = data.net("full", i)
        tw.forward(w)
        out[i] = list(rows)
        for d in rows:
            print(i, json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}), flush=True)
Path(MAIN / "experiments" / "2026-10-07_p2" / "gm_diag.json").write_text(json.dumps(out))
