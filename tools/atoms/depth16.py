"""Atom `depth16` -- Lane S, HANDOFF 2026-10-12 Lane S step S2: `_depth_for` by exact price, leaf >= 16.

NOT bit-identical (deeper Strassen on some products changes float32 rounding): DEV-100 raw within +-0.3 % of v85.
Default arm: leaf 16, cap 6, lam 8e5 FLOPs/op. Arms: `depth16:<lam>[/<cap>]`.
Replay price on v84 (exact model == meter; `experiments/2026-10-13_s/s2_lambda_scan.log`):
  lam 8e5 -> F/B 0.16370 (-5.67 % vs v84), 24,875 ops.   lam 0 -> 0.16320, 26,875 ops.
Requires `sb2` earlier in --atoms. Anchor: v84's `_depth_for`, verbatim. Twin arm: none yet.
"""
from tools.atoms._lane_s_depth import arm, patch


def apply(src: str, z) -> str:
    return patch(src, 16, 6, 800000)


def make(arg: str):
    return arm(16, arg)
