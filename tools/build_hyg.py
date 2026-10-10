"""P2.4 cost hygiene on a v79-family file (HANDOFF 2026-10-07 Lane P2, P2.4.4 / P2.4.5).

    uv run python tools/build_hyg.py --base estimator_k3_v79_dress.py --l0 --block 16 --out X.py

--l0     Layer 0's input covariance is exactly I (x ~ N(0, I); `predict` starts the state at
         `fnp.eye`), so `W^T I W = W^T W`: the sandwich's first product `W^T @ I` is skipped and
         the block-symmetric second product is applied to `W^T` directly. Exact in real
         arithmetic (float32 drops one rounded product).
--join   P2.4.3: the join's old-coordinate re-rotation `T @ [FA|FM]` goes through `_mm` (Strassen).
--bigvol One extra Strassen level for products with m*k*n >= this (P2.4.5, ops-efficient).
--block  `_STRASSEN_MIN_BLOCK` (v77/v79: 20). Smaller blocks give compact (K < 1024) products
         their 5th Strassen level more often: fewer FLOPs, more operations.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

L0_FN = '''

def _p4_sym_rest(X, W):
    """The second half of `_p4_sym_sandwich` given `X = W^T C` (P2.4.4: layer 0, C = I)."""
    n = X.shape[0]
    if n % 2:
        return _mm(X, W)
    h = n // 2
    top = _mm(X[:h], W)
    b22 = _mm(X[h:], W[:, h:])
    return fnp.concatenate([top, fnp.concatenate([top[:, h:].T, b22], axis=1)], axis=0)
'''

L0_SITE = ("        cov_pre = (_p4_sym_sandwich(WT, cov, W) if _P4_BLOCKSYM\n"
           "                   else _mm(_mm(WT, cov), W))\n")
L0_NEW = ("        if layer_index == 0 and adv is None and _P4_BLOCKSYM:\n"
          "            # P2.4.4: the input covariance is I, so W^T I W = W^T W (no W^T @ I)\n"
          "            cov_pre = _p4_sym_rest(WT, W)\n"
          "        else:\n"
          "            cov_pre = (_p4_sym_sandwich(WT, cov, W) if _P4_BLOCKSYM\n"
          "                       else _mm(_mm(WT, cov), W))\n")


DEPTH_SITE = ("    while d < _STRASSEN_DEPTH and (lo >> (d + 1)) >= _STRASSEN_MIN_BLOCK:\n"
              "        d += 1\n"
              "    return d\n")
DEPTH_NEW = ("    while d < _STRASSEN_DEPTH and (lo >> (d + 1)) >= _STRASSEN_MIN_BLOCK:\n"
             "        d += 1\n"
             "    # P2.4.5: one more level for BIG products whose leaf would stay >= the big block.\n"
             "    # Python ints only (no flopscope op); FLOPs per added op are best on these.\n"
             "    if (d < _STRASSEN_DEPTH and m * k * n >= _STRASSEN_BIG_VOL\n"
             "            and (lo >> (d + 1)) >= _STRASSEN_BIG_BLOCK):\n"
             "        d += 1\n"
             "    return d\n")


JOIN_SITE = "        FF = T @ fnp.concatenate([tw.FA, tw.FM], axis=1)\n"
JOIN_NEW = ("        # P2.4.3: the old-coordinate re-rotation (r x r @ r x 2C, C up to ~17k) through _mm\n"
            "        FF = _mm(T, fnp.concatenate([tw.FA, tw.FM], axis=1))\n")


def build(src: str, l0: bool, block: int | None, bigvol: int | None = None,
          bigblock: int = 16, join: bool = False) -> str:
    if join:
        assert src.count(JOIN_SITE) == 1
        src = src.replace(JOIN_SITE, JOIN_NEW, 1)
    if bigvol is not None:
        assert src.count(DEPTH_SITE) == 1
        src = src.replace(DEPTH_SITE, DEPTH_NEW, 1)
        src, k = re.subn(r"^(_STRASSEN_MIN_BLOCK = \d+)$",
                         rf"\1\n_STRASSEN_BIG_VOL = {bigvol}\n_STRASSEN_BIG_BLOCK = {bigblock}",
                         src, flags=re.M)
        assert k == 1
    if l0:
        a = "\n\ndef _p3_compress(D):"
        assert src.count(a) == 1 and src.count(L0_SITE) == 1
        src = src.replace(a, L0_FN + a, 1).replace(L0_SITE, L0_NEW, 1)
    if block is not None:
        src, k = re.subn(r"^_STRASSEN_MIN_BLOCK = \d+$", f"_STRASSEN_MIN_BLOCK = {block}", src,
                         flags=re.M)
        assert k == 1
    return src


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--l0", action="store_true")
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--bigvol", type=int, default=None, help="extra Strassen level at m*k*n >= this")
    ap.add_argument("--bigblock", type=int, default=16)
    ap.add_argument("--join", action="store_true", help="route the join's re-rotation via _mm")
    a = ap.parse_args()
    Path(a.out).write_text(build(Path(a.base).read_text(encoding="utf-8"), a.l0, a.block,
                                 a.bigvol, a.bigblock, a.join),
                           encoding="utf-8", newline="\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
