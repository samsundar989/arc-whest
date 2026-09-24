"""Generate `estimator_k3_v57_corr.py`: v55 plus the in-loop learned corrector.

    python tools/build_v57.py --verify                 parity check, no weights
    python tools/build_v57.py --params <params.npz>     inject a fit and emit
    python tools/build_v57.py --params ... --out <file>

HANDOFF 2026-09-24 Step 2. The corrector is a per-layer `33 -> 32 -> 32 -> 1`
tanh MLP that predicts `mu_l(label) - mu_l(chain)` from the layer's own state and
adds it to `mu_out` **before it becomes the next layer's input**, so corrections
propagate. It is generated rather than hand-edited so that retraining is one
command and the diff against v55 stays exactly three insertions.

WHY A GENERATOR AND NOT AN EDIT
------------------------------
The weights are ~36,000 float32 values. Hand-maintaining them in a 100 KB source
file is how a stale coefficient ships. Here the only editable artifacts are this
file and the params `.npz`; `estimator_k3_v57_corr.py` is output.

THE PARITY INVARIANT
--------------------
`_CORR_ON = False` makes every inserted line dead, and `--verify` asserts the
generated file is then **bit-identical to v55** on real nets. With weights
injected, `_CORR_ON = True`, and an all-zero layer is still exactly inert
(`tanh(0) = 0` through both hidden layers, then `+ b3 = 0`), so a partially
trained fit degrades to the base chain layer by layer instead of misbehaving.

WEIGHTS ARE BASE64 float32, NOT DECIMAL LITERALS
------------------------------------------------
36k decimal reprs is ~700 KB of source and rounds unless every value carries 9
significant digits. Base64 of the raw float32 bytes is ~190 KB, decodes EXACTLY
(verified max error 0.0), and uses only `base64` and `array` from the stdlib,
which the grader contract allows. Decoding happens in pure Python on first use --
no flopscope operation, so it is not billed and cannot depend on MLP order.

COST, AND WHY THE ARRAYS ARE REBUILT EVERY `predict`
----------------------------------------------------
v55's `_constants` deliberately rebuilds its arrays for every MLP, because
caching makes the first MLP of a run cost more FLOPs than the rest and
`certify_candidate` section 6 refuses an order-dependent per-MLP cost. The
corrector follows that rule: the eight stacked weight arrays are built inside
`_constants` (2 ops each, 16 ops per MLP) rather than cached on the instance.
Measured per-layer cost of the corrector itself: ~8 MFLOP and ~30 op calls, i.e.
~128 MFLOP (0.027% of v55's 475 G) and ~500 ops (+2.1%) over 16 layers.
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "estimator_k3_v55_p2r4.py"
OUT = ROOT / "estimator_k3_v57_corr.py"

# --------------------------------------------------------------- the anchors

_A_IMPORTS = "import flopscope.numpy as fnp\n"

_A_LAYER_DEF = ("\ndef _layer(W, state, table, dtype, const, init_scale, "
                "last_layer, gens,\n")

_A_APPLY = """
    mu_out = K[(1,)]
    if last_layer:
        return (mu_out, None, None, 0.0, dG)
"""

_I_APPLY = """
    mu_out = K[(1,)]
    # ---- THE IN-LOOP MEAN CORRECTION (v57) --------------------------------
    # Applied HERE, to the state, not to the output array: `mu_out` is what the
    # next layer transports as `mu_pre = W^T mu`, so a correction added here
    # propagates through the rest of the chain. The covariance and kappa3
    # channels are deliberately left alone. Inert when `_CORR_ON` is False or a
    # layer's weights are zero.
    if _CORR_ON and const.get("corr") is not None:
        mu_out = mu_out + _corr_delta(
            layer_index, const["corr"], mu_pre, sigma, D3, D4, W,
            None if last_layer else D21, mu, mu_out, n, dtype)
    if last_layer:
        return (mu_out, None, None, 0.0, dG)
"""

_A_CONST = """                "closure": (_closure_consts(dtype)
                            if (_CLOSURE_SCOPE or _PROTECT_FEED) else None)}
"""

_I_CONST = """                "closure": (_closure_consts(dtype)
                            if (_CLOSURE_SCOPE or _PROTECT_FEED) else None),
                "corr": _corr_arrays(dtype)}
"""

# The corrector block, inserted immediately before `def _layer(`.
_BLOCK = '''
# =========================== THE IN-LOOP LEARNED MEAN CORRECTOR ==============
# HANDOFF 2026-09-24 Step 2, and the answer to the anatomy in RESULTS.md
# "Step 1 (HANDOFF 09-24)": 79% of the final-layer error is drift carried in from
# earlier layers, a ReLU layer CONTRACTS incoming mean error (A_l = 0.60-0.88),
# and all the growth in MSE is newly created error. So the lever is to shrink
# each layer's own increment, in the loop, where the correction can propagate.
#
# What the corrector may read: the chain's own state at this layer, and nothing
# else. No neuron index, no net identity, no label-derived quantity, no
# randomness. Every feature below is a function of `W_l` and the incoming state,
# so the estimator stays deterministic and shape-generic.
#
# Why a NONLINEAR model. Measured on FIT 100 (RESULTS.md, "Step 2 priced"):
# every single-feature Pearson correlation with the residual is < 0.10 and every
# single-axis nonparametric ceiling is < 0.7%, yet a 33-feature MLP reaches a
# net-wise held-out R^2 of 0.057-0.119 against a ridge's 0.016-0.024. The signal
# lives entirely in feature INTERACTIONS; a linear corrector cannot see it.

#: Master switch. False makes every line in this block dead and this file
#: bit-identical to v55 -- asserted by `tools/build_v57.py --verify`.
_CORR_ON = False

#: The nine raw per-neuron channels, in the order the fitted weights assume:
#: mu_pre, sigma, alpha, D3/s^3, D4/s^4, ||W[:,i]||, D21energy/s^4, mu_out, dmu.
#: This order is a CONTRACT shared with `tools/anatomy.py` FEATURES and
#: `tools/corr_ceiling.py` expand(). Reordering it silently invalidates the fit.
_CORR_NF = 9

#: Pairwise products of the first six raw channels, i < j -- the expansion the
#: ridge and MLP ceilings were measured with.
_CORR_PI = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4]
_CORR_PJ = [1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5]

#: name -> (base64 of C-order float32 bytes, shape). Injected by the builder.
#: Empty here, which is what keeps the un-fitted file inert.
_CORR_B64 = {}

_CORR_DECODED = None


def _corr_decode():
    """Base64 -> Python lists. PURE PYTHON, so it bills nothing and cannot make
    the first MLP of a run cost more than the rest."""
    global _CORR_DECODED
    if _CORR_DECODED is None:
        _CORR_DECODED = {
            k: (list(_array.array("f", _base64.b64decode(blob))), tuple(shape))
            for k, (blob, shape) in _CORR_B64.items()}
    return _CORR_DECODED


def _corr_arrays(dtype):
    """The eight stacked weight arrays, REBUILT EVERY `predict`.

    Not cached on the instance, for the same reason `_constants` is not: an
    order-dependent per-MLP cost is refused by certification, and it would price
    a candidate near a multiplier boundary on the wrong side. Two operations per
    array, sixteen per MLP.
    """
    if not _CORR_ON or not _CORR_B64:
        return None
    return {k: fnp.reshape(fnp.asarray(v, dtype=dtype), shape)
            for k, (v, shape) in _corr_decode().items()}


def _corr_delta(li, corr, mu_pre, sigma, D3, D4, W, D21, mu_prev, mu_out, n,
                dtype):
    """The fitted mean correction for layer `li`, as an `(n,)` array.

    The target's mean and standard deviation are folded into `W3`/`b3` at export
    time, so there is no de-standardisation step to pay for here.
    """
    inv = dtype(1.0) / sigma
    inv2 = inv * inv
    if D21 is None:
        # The LAST layer never forms D21 -- skipping it is worth 5 matmuls per
        # lineage and is why `last_layer` pruning exists. Its energy channel is
        # therefore a constant zero, which is exactly how the corrector was
        # FITTED for that layer, not an approximation made at inference.
        d21e = mu_out * dtype(0.0)
    else:
        d21e = (D21 * D21).sum(axis=1) * (inv2 * inv2)
    X = fnp.stack([mu_pre, sigma, mu_pre * inv,
                   D3 * (inv2 * inv), D4 * (inv2 * inv2),
                   fnp.sqrt((W * W).sum(axis=0)), d21e,
                   mu_out, mu_out - mu_prev.mean()], axis=1)
    X = fnp.concatenate([X, X * X, X[:, _CORR_PI] * X[:, _CORR_PJ]], axis=1)
    X = (X - corr["fmu"][li]) * corr["fisd"][li]
    h = fnp.tanh(X @ corr["W1"][li] + corr["b1"][li])
    h = fnp.tanh(h @ corr["W2"][li] + corr["b2"][li])
    return (h @ corr["W3"][li] + corr["b3"][li])[:, 0]


'''


def build(params: Path | None) -> str:
    src = BASE.read_text(encoding="utf-8")
    for name, anchor in (("imports", _A_IMPORTS), ("_layer def", _A_LAYER_DEF),
                         ("apply point", _A_APPLY), ("_constants", _A_CONST)):
        if src.count(anchor) != 1:
            raise SystemExit(f"build_v57: anchor {name!r} occurs "
                             f"{src.count(anchor)} times, expected 1")

    block = _BLOCK
    if params is not None:
        d = np.load(params)
        enc = {}
        for k in ("fmu", "fisd", "W1", "b1", "W2", "b2", "W3", "b3"):
            a = np.ascontiguousarray(d[k], dtype=np.float32)
            enc[k] = (base64.b64encode(a.tobytes()).decode("ascii"), a.shape)
        lines = ["_CORR_B64 = {"]
        for k, (blob, shape) in enc.items():
            lines.append(f'    "{k}": (')
            for i in range(0, len(blob), 96):
                lines.append(f'        "{blob[i:i + 96]}"')
            lines.append(f"    , {tuple(int(x) for x in shape)}),")
        lines.append("}")
        block = block.replace(
            "#: Empty here, which is what keeps the un-fitted file inert.\n"
            "_CORR_B64 = {}",
            "#: Injected by `tools/build_v57.py --params`.\n" + "\n".join(lines))
        block = block.replace("_CORR_ON = False", "_CORR_ON = True", 1)

    src = src.replace(_A_IMPORTS,
                      _A_IMPORTS + "\nimport array as _array\nimport base64 as _base64\n", 1)
    src = src.replace(_A_LAYER_DEF, block + _A_LAYER_DEF, 1)
    src = src.replace(_A_APPLY, _I_APPLY, 1)
    src = src.replace(_A_CONST, _I_CONST, 1)
    return src


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--params", default="")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--verify", action="store_true",
                    help="emit with no weights and prove bit-parity with v55")
    ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args(argv)

    params = Path(a.params) if a.params else None
    if params is not None and not params.is_absolute():
        params = ROOT / params
    src = build(params)
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.write_text(src, encoding="utf-8", newline="\n")
    on = "_CORR_ON = True" in src
    print(f"[build_v57] -> {out.relative_to(ROOT)}  ({len(src):,} bytes, "
          f"_CORR_ON={on})")

    if a.verify:
        if on:
            raise SystemExit("--verify needs the weightless build (drop --params)")
        sys.path.insert(0, str(ROOT / "tools"))
        import subprocess
        r = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "_parity.py"),
             str(BASE), str(out), str(a.n)],
            cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        print(r.stdout.strip())
        if r.returncode != 0:
            print(r.stderr[-2000:])
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
