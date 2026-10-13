"""Lane O, HANDOFF 2026-10-12 Lane O step O4: spend items on the pruned stack. The one that pays is J3.

    o4        J3: `_JOIN_SKIP_TAIL = 3` (no old-tier join in the last 3 layers before the final one; v84 has 2)
    o4:<k>    `_JOIN_SKIP_TAIL = k` (o4:4 = J4, screened and NOT recommended)

Not bit-identical. One constant (price.py `J(k)`), so order-independent: it touches no line any S, K or O atom
anchors on. `v87 + o4` rebuilds the screened file byte for byte.

On v87 = fold+age6 + sb2+sbtrim+depth8+k0:unrefit+k1:2.5_3.0+k2:3.0:unrefit+k3 (R8, Lane K's out=-aware meter):
  v87        F/B 0.14250, 25,211 ops
  v87 + o4   F/B 0.14256 (+0.04 %), 24,780 ops (-431)
Accuracy, unrefit VAL-100 in the twin, paired, on stack B over v84 (the twin cannot run sbtrim's `fnp.empty`;
sb2/sbtrim/depth8 only reorganise Strassen): -0.27 % raw (-0.27 +/- 0.05 % per net, 75/100 wins), net -0.22 %.
Twin arm: the built file `--raw`. Results: `experiments/2026-10-13_o/O4/RESULTS.md`.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "docs/research/2026-10-11_p6price"))
import price  # noqa: E402


def apply(src, z):
    return price.J(3)(src)


def make(arg):
    return lambda src, z: price.J(int(arg))(src)
