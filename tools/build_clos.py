"""Step 3.3 (HANDOFF 09-27): embed the closure-correction nets in an estimator file.

    python tools/build_clos.py --base estimator_k3_v62_s1.py --weights <clos.npz> --out <file.py>

The closure (`_wick`, the term table, `_closure_apply`, `pK -> K`) has a residual on EXACT inputs
that is a smooth state function (RESULTS.md § Step 3.1–3.2: mlp R² 0.63–0.87 on held-out nets).
This adds, right after `K` is formed and before anything reads it:

  per neuron  (alpha, gamma3 = D3/s^3, gamma4 = D4/s^4, l/16) -> tanh 16 -> tanh 16 -> 4
              K1 += s o0     K2 += var o1     K3 += s^3 o2     K4 += s^4 o3   (last layer: K1 only)
  per pair    (rho, alpha_i, alpha_j, g3_i, g3_j, d21, d12, d22, l/16) -> tanh 16 -> 2
              K11 += s_i s_j sym(o0)       K21 += s_i^2 s_j o1                    (off-diagonal)

so the corrected K1/K2/K11 reach the mean and covariance, K3/K21 the kappa3 births, K4 the kappa4
core. Weights are the npz of `tools/closure_fit.py` (with standardisers `fmu`/`fisd` and output
scale `osd`, FOLDED here into the first/last layers) or a twin parameter file (already folded).
The pair net's first layer is split by linearity -- row features, column features, pair
features -- so no (n, n) copy of a per-neuron feature is ever formed. Cost ~0.3 GFLOP/layer.
"""
from __future__ import annotations

import argparse
import base64
import json
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
KEYS = ("uW1", "ub1", "uW2", "ub2", "uW3", "ub3", "bW1", "bb1", "bW2", "bb2")

_CODE = '''

# ==========================================================================
# STEP 3.3 (HANDOFF 09-27, tools/build_clos.py): the calibrated closure
# ==========================================================================
_CLOS_ON = True
_CLOS_B64 = (
{b64}
)


def _clos_decode():
    return json.loads(zlib.decompress(base64.b64decode("".join(_CLOS_B64))).decode("ascii"))


def _clos_arrays(dtype):
    """The closure-correction weights, rebuilt every `predict` (as `_corr_arrays`)."""
    if not _CLOS_ON or not _CLOS_B64:
        return None
    return {{k: fnp.reshape(fnp.asarray(v, dtype=dtype), shape)
            for k, (v, shape) in _clos_decode().items()}}


def _clos_apply(K, c, sigma, var, alpha, D3, D4, S, off_mask, last_layer, layer_index, dtype):
    """K corrected by the fitted closure residual nets (see tools/build_clos.py)."""
    inv = fnp.divide(dtype(1.0), sigma)
    inv2 = inv * inv
    g3 = D3 * (inv2 * inv)
    g4 = D4 * (inv2 * inv2)
    lf = float(layer_index) / 16.0
    X = fnp.stack([alpha, g3, g4], axis=1)
    h = fnp.tanh(X @ c["uW1"][:3] + (c["uW1"][3] * lf + c["ub1"]))
    h = fnp.tanh(h @ c["uW2"] + c["ub2"])
    o = h @ c["uW3"] + c["ub3"]
    K = dict(K)
    K[(1,)] = K[(1,)] + o[:, 0] * sigma
    if last_layer:
        return K
    K[(2,)] = K[(2,)] + o[:, 1] * var
    K[(3,)] = K[(3,)] + o[:, 2] * (var * sigma)
    K[(4,)] = K[(4,)] + o[:, 3] * (var * var)
    ii = inv[:, None]
    jj = inv[None, :]
    P = fnp.stack([S[(1, 1)] * (ii * jj), S[(2, 1)] * (inv2[:, None] * jj),
                   S[(2, 1)].T * (ii * inv2[None, :]),
                   S[(2, 2)] * (inv2[:, None] * inv2[None, :])], axis=2)
    W1 = c["bW1"]
    rowv = alpha[:, None] * W1[1] + g3[:, None] * W1[3]
    colv = alpha[:, None] * W1[2] + g3[:, None] * W1[4]
    Wp = fnp.stack([W1[0], W1[5], W1[6], W1[7]], axis=0)
    H = fnp.tanh(P @ Wp + rowv[:, None, :] + colv[None, :, :] + (W1[8] * lf + c["bb1"]))
    O = H @ c["bW2"] + c["bb2"]
    ss = sigma[:, None] * sigma[None, :]
    o11 = O[:, :, 0]
    K[(1, 1)] = K[(1, 1)] + ((o11 + o11.T) * 0.5 * ss) * off_mask
    K[(2, 1)] = K[(2, 1)] + (O[:, :, 1] * (ss * sigma[:, None])) * off_mask
    return K
'''

_PATCHES = [
    ('                "corr": _corr_arrays(dtype)}',
     '                "corr": _corr_arrays(dtype), "clos": _clos_arrays(dtype)}'),
    ("    mu_out = K[(1,)]\n",
     "    if const.get(\"clos\") is not None:\n"
     "        K = _clos_apply(K, const[\"clos\"], sigma, var, alpha, D3, D4, S, off_mask,\n"
     "                        last_layer, layer_index, dtype)\n"
     "    mu_out = K[(1,)]\n"),
]


def fold(z) -> dict:
    """closure_fit.py npz -> the ten folded arrays; a twin params file is passed through."""
    if "clos__uW1" in z:                         # twin parameter file: already folded
        return {k: np.asarray(z[f"clos__{k}"], np.float32) for k in KEYS}
    out = {}
    for p in ("u", "b"):
        W1, b1 = z[f"{p}W1"].astype(np.float64), z[f"{p}b1"].astype(np.float64)
        mu, isd = z[f"{p}fmu"].astype(np.float64), z[f"{p}fisd"].astype(np.float64)
        out[f"{p}W1"] = W1 * isd[:, None]                     # (x - mu) * isd @ W1
        out[f"{p}b1"] = b1 - (mu * isd) @ W1
        last = "3" if p == "u" else "2"
        osd = z[f"{p}osd"].astype(np.float64)
        if p == "u":
            out["uW2"], out["ub2"] = z["uW2"], z["ub2"]
        out[f"{p}W{last}"] = z[f"{p}W{last}"].astype(np.float64) * osd[None, :]
        out[f"{p}b{last}"] = z[f"{p}b{last}"].astype(np.float64) * osd
    return {k: np.asarray(out[k], np.float32) for k in KEYS}


def encode(W: dict) -> str:
    js = json.dumps({k: [np.asarray(v, np.float32).ravel().tolist(), list(v.shape)]
                     for k, v in W.items()})
    b = base64.b64encode(zlib.compress(js.encode("ascii"), 9)).decode("ascii")
    return "\n".join(f'    "{b[i:i + 96]}"' for i in range(0, len(b), 96))


def build(src: str, W: dict) -> str:
    for a, b in _PATCHES:
        if src.count(a) != 1:
            raise SystemExit(f"anchor occurs {src.count(a)}x:\n{a}")
        src = src.replace(a, b, 1)
    anchor = "\n\ndef _layer(W, state, table, dtype, const, init_scale, last_layer, gens,\n"
    if src.count(anchor) != 1:
        raise SystemExit("no _layer anchor")
    return src.replace(anchor, _CODE.format(b64=encode(W)) + anchor, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zero", action="store_true", help="all-zero nets (an exact identity)")
    ap.add_argument("--zero-last", action="store_true",
                    help="keep the fitted hidden layers, zero the output layers: an exact identity "
                         "at init whose features are the offline fit's (the twin then learns only "
                         "the correction that survives in-loop)")
    a = ap.parse_args()
    W = fold(np.load(a.weights))
    if a.zero:
        W = {k: v * 0 for k, v in W.items()}
    if a.zero_last:
        for k in ("uW3", "ub3", "bW2", "bb2"):
            W[k] = W[k] * 0
    src = build((ROOT / a.base).read_text(encoding="utf-8"), W)
    Path(a.out).write_text(src, encoding="utf-8", newline="\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
