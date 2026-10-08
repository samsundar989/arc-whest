"""P3.1b: star-D4 on a v80-family file, with the twin refit (stage S2ts) baked in.

    uv run python tools/build_star.py --base estimator_k3_v80_hyg.py \
        --params experiments/2026-10-08_p3/refit_star/params_best.npz --out estimator_k3_v81_star.py

The free fourth-order births (R3 §3.1): a recent kappa3 lineage with legs A = M X (our A_L) and
B = M, hub weight w3 of its birth layer, carries the star kappa4 whose (4,) slice is
D4_star[a] = 4 sum_j w3_j A_aj^3 M_aj -- O(n^2) per lineage, no transport. It is added to the
chain's D4 with a learned per-layer gain (twin arm `bank`, key `star`, gain/sg; init 0). The
birth w3 of each RECENT lineage (KAS H columns only) rides in the chain's unused `c4` slot.

The refit's S2t tables and corrector replace the base's (tools/build_dress.build, dressing off
because the base is already dressed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_dress  # noqa: E402

ANCHOR = "def _layer(W, state, table, dtype, const, init_scale, last_layer, gens,\n"
PATCHES = [
    ("    D4 = metric * dG\n",
     "    D4 = metric * dG\n"
     "    if (isinstance(c4, list) and adv is not None and layer_index < len(_STAR_GAIN)\n"
     "            and _STAR_GAIN[layer_index] != 0.0):\n"
     "        # P3.1b: D4_star = 4 sum_j w3b_j A_L^3 M over the recent lineages (O(n^2 k))\n"
     "        _w3c = c4[0] if len(c4) == 1 else fnp.concatenate(c4)\n"
     "        _AL = adv.A_L\n"
     "        D4 = D4 + (((_AL * _AL) * (_AL * adv.M)) @ _w3c) * (4.0 * _STAR_GAIN[layer_index])\n"),
    ("    return (mu_out, cov_out, out_tw, c4_out, g_out)\n",
     "    # P3.1b: carry the birth w3 of every RECENT lineage (the joiner leaves with the join)\n"
     "    _w3n = w[(3, 1)] if _kh is None else fnp.take(w[(3, 1)], _kh)\n"
     "    c4_out = ((list(c4) if isinstance(c4, list) else []) + [_w3n])[-(out_tw.k - out_tw.ko):]\n"
     "    return (mu_out, cov_out, out_tw, c4_out, g_out)\n"),
]


def build(base: Path, params: Path) -> str:
    src = build_dress.build(base, params, dress=(0.0, 0.0, 0.0))
    g = np.load(params)["gain__sg"].astype(float).tolist()
    const = ("# ---- P3.1b STAR-D4 (tools/build_star.py; R3 §3.1) -- per-layer gain, twin-fitted\n"
             f"_STAR_GAIN = {g!r}\n\n\n")
    assert src.count(ANCHOR) == 1
    src = src.replace(ANCHOR, const + ANCHOR, 1)
    for a, b in PATCHES:
        if src.count(a) != 1:
            raise SystemExit(f"anchor occurs {src.count(a)}x:\n{a}")
        src = src.replace(a, b, 1)
    return src


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).write_text(build(Path(a.base), Path(a.params)), encoding="utf-8", newline="\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
