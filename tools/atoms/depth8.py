"""Atom `depth8` -- Lane S, HANDOFF 2026-10-12 Lane S step S2: `_depth_for` by exact price, leaf >= 8.

NOT bit-identical (deeper Strassen on some products changes float32 rounding): DEV-100 raw within +-0.3 % of v85.
Default arm: leaf 8, cap 6, lam 1.2e6 FLOPs/op (fits the interim 26,000-op cap). Arms: `depth8:<lam>[/<cap>]`.
Replay price on v84 (exact model == meter; `experiments/2026-10-13_s/s2_lambda_scan.log`):
  lam 1.2e6 cap 6 -> F/B 0.15972 (-7.96 % vs v84), 25,201 ops.   lam 0 cap 7 -> 0.15844, 30,017 ops.
Requires `sb2` earlier in --atoms. Anchor: v84's `_depth_for`, verbatim. Twin arm: none yet.
"""
from tools.atoms._lane_s_depth import arm, patch


def apply(src: str, z) -> str:
    return patch(src, 8, 6, 1200000)


def make(arg: str):
    return arm(8, arg)
