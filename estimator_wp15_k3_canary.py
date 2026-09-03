"""K3 CANARY -- the regenerating k_max=3 cumulant flow, as a grader-runnable file.

**NOT A SUBMISSION AND NOT SHIPPED.** `estimator.py` is untouched and remains
v2.6.1. This file exists to answer ONE question that no local instrument can:

    SKEPTIC WP14-FOLLOWUP measured this flow at ~5,300 flopscope operations per
    MLP against the shipped closure's 453, with a LOCAL in-process residual of
    0.29-0.44 s against the 0.4 s per-MLP cliff -- over it on 2 of 6 readings.
    A residual excursion does not cost score: it FAILS the MLP and zeroes the
    prediction. Under lambda = 0 that is the only cost that can still ruin a run,
    and `docs/GOTCHAS.md` forbids quoting a local absolute wall time.

    The authority is `whest run --runner subprocess`. This file is what it runs.

Read from its report: the OFFICIAL per-MLP residual and `n_failed_mlps`. Nothing
else. No score is quoted, no seed block is drawn, nothing is selected -- the
`mini` split is CONFIRM tier and read-only, and this run fits nothing on it.

WHAT IT COMPUTES
----------------
The SIMPLE k_max=3 tower, with kappa_3 carried in the REGENERATING form: factors
`(diag(S3) + 3 S21^T, I, I)`, exactly what `FactoredTensor.from_dstensor` builds,
rebuilt from its own two read-slices each layer so the rank never grows. The
all-distinct block is generated and then discarded at the rebuild; that discarded
part is the known error and is the subject of CONTRACT WP15.

Five n^3-class operations per layer, and no more:
    cov_pre = W.T @ cov @ W        two   (the K2 backbone)
    P       = W.T @ M              one   (contract_W; legs 2 and 3 are I)
    (P.Q) @ Q.T  and  (Q.Q) @ P.T  two   (the (2,1) slice's block perms)
Measured after WP16 removed the dead `(1,1,1)` block: 10,828,494,338 counted f32
FLOPs per layer at 1024, 173,230,212,715 over 16 layers = 0.078776 x budget,
multiplier 0.10000 (`experiments/2026-09-03_wp16_parity/flow_price_nosub.json`).
The pre-WP16 figures 10,837,935,618 / 173,381,273,195 are SUPERSEDED.

WHAT IS NOT ESTABLISHED, AND MUST BE READ BESIDE ANY NUMBER THIS PRODUCES
------------------------------------------------------------------------
The VALUES ARE checked against upstream at non-zero kappa_3 as of 2026-09-03
(`local_sim/p2_k3_parity.py`; `NOT_DONE` step 6 CLOSED): end-to-end f64 drift
`1.692e-11` on 8/8 P01 nets at 1024x16, raw MSE `8.5075e-07` against minted
truth. **THE VERSION OF THIS FILE THE GRADER SCORED AT `2.8192e-06` IS RETIRED**
-- it carried the `(1,1,1)` repeated-part subtraction WP16 removed, worth
`3.31x`. Nothing here has been re-graded. The three closed-form checks that
do exist still kill only 1 of 12 targeted mutations; the golden-parity detector
WP16 added kills 9 of 9 on the live code path. **This file has not been graded
since the fix**, so its accuracy is established against upstream and against
minted truth, and NOT against the grader.

The frozen combinatorial tables are embedded below because the grader sandbox has
no `experiments/`. `local_sim/p2_k3_candidate_parity.py` asserts this file agrees
with `p2_k3_fnp.k3_flow` to f32 rounding, so the copy cannot drift in silence.

Contract: `experiments/LOG.md`, 2026-09-02, CONTRACT WP15 (HEAD 1b6ac04) and
SKEPTIC WP14-FOLLOWUP. `# seed-ledger: no block named` -- no seed is drawn here.
"""

import base64
import json
import math
import zlib

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator

#: zlib+base64 of the `terms` / `wick` / `pK_to_K` / `lap2_d4` tables from
#: `experiments/2026-09-02_wp14_fnp/k3_coefs_simple.json`, sha256
#: 894185e450533765f633312b36345ab7a29bfb4219a9ab5ee0c36a12d5988fa7, enumerated
#: from upstream rev 93d091a4. Every table depends only on the partition
#: combinatorics -- not on n, not on the weights, not on the data.
_TABLES_B64 = (
    "eNq1l8GOmzAQhl9l5dOu5ETxzGxU5RX2svcKIQLsNkoaECGtqoh3r22KwYDBlDaXBBj/38w/dowf7BLlECbEDg/2"
    "DPwJXtSv07UM86goT+Upu4Zxln6wA3JWpsX3Gzt8fbA4yvM0kT93XASc1RFftjvOPoso/yYfCP2gSG+n5B5dtJq8"
    "G1RBxdkzcSdGjGMMBKhLAQdDQvK3sMzCN12Y4E/iRSseL1l8VuI6QaMqtrtKXsjAQdgwCEbVZIq25gZkPO8GAR9X"
    "g76adnVMk/qSdSRMcxshmA4SPhXAWAVD03jXuI0Y6AxV0KFiKQ2zVtV3SNgPwCGJJkh23vsxXA8pRnPqYDc0rH4m"
    "aeolXXUXhQnTMfdrqRdNs5bU0pMPbnF0iQo9nLOfp/jcTJbAIj1Ykn4WaaqX9480lrcwqHyEd1ux731aEE6AqAGR"
    "L2hHPRS1JBqQXJrgdgVWuQL+rsAqV2CBK+DpCrpdwVWuoL8ruMoVXOAKerpCbldolSvk7wqtcoUWuOImQUNS24KK"
    "GX/gXMbcXsivLRXafcuL7YewetXZ0OY6prbGDgKWIcDn3w5sxLRRsMKpxV2CqS7BP+kSTFkIf9kl7yrEbBXYdmmN"
    "UWL1RIDJiQD/eyLA3ESA1RMB5iYCrFyuswjw2YZnugSzXVKMoLn9YPLVS329C5X+Vh1n3kGda2R69eWxiK6xPNKw"
    "PLv8YpzJUTLzXKLly598RTGj2yFaQRXooQJaBY0KWMNMLuinhlqNrJxeh4qN2t5PlaSqqH0yMcfoltYxwrghOm6A"
    "nT+4CMK4AE4CGALUBBNTpMk9VlEfUVxmxUkdNnV8Peoc1gFJl4NODhoOLuCgxYEOBz009iMawvQSO71UB2zLUnJ5"
    "iqZr5KyVTK20oFay8kRTKy2olUb8Qq1BPnnQiIj4U3BV/QZ7Vwq4"
)

_TABLES = None


def _tables():
    global _TABLES
    if _TABLES is None:
        _TABLES = json.loads(zlib.decompress(base64.b64decode(_TABLES_B64)))
    return _TABLES


def _parse_int_part(key):
    return tuple(int(x) for x in key.strip("()").split(",") if x.strip())


def _evalpoly(coefs, x):
    """Horner with PYTHON floats.

    `np.float64(c)` would be CONCRETE under NEP-50 and would promote an f32 array
    on contact, and flopscope bills float64 at twice the float32 rate -- so a
    single numpy scalar here doubles the bill for the whole downstream graph.
    """
    y = x * 0.0 + float(coefs[-1])
    for c in reversed(coefs[:-1]):
        y = y * x + float(c)
    return y


def _wick(mean, sigma, alpha, table, dtype):
    """`relu_wick_coef` for the 16 (k, p) pairs the layer needs. O(n) each.

    The two `flops.stats.*` calls are the only place a float64 can enter --
    flopscope promotes their inputs "to match scipy.stats" -- and both are cast
    back immediately. Without the casts the covariance accumulator, where nearly
    all the cost lives, is float64 in an arm that calls itself float32.
    """
    Phi = flops.stats.norm.cdf(alpha).astype(dtype)
    phi = flops.stats.norm.pdf(alpha).astype(dtype)
    he = {0: alpha * 0.0 + 1.0, 1: alpha, 2: alpha * alpha - 1.0}

    def base(k):
        if k == 0:
            return sigma * phi + mean * Phi
        if k == 1:
            return Phi
        den = sigma
        for _ in range(k - 2):
            den = den * sigma
        return ((-1.0) ** (k - 2)) * he[k - 2] * phi / den

    out = {}
    for _key, spec in table["wick"].items():
        k, p = int(spec["k"]), int(spec["p"])
        if spec["branch"] == "base":
            v = base(k)
        elif spec["branch"] == "reduce":
            v = float(spec["factorial_p"]) * base(int(spec["k_reduced"]))
        else:
            s_pow = alpha * 0.0 + 1.0
            for _ in range(p - k):
                s_pow = s_pow * sigma
            v = s_pow * (_evalpoly(spec["P1"], alpha) * phi
                         + _evalpoly(spec["P2"], alpha) * Phi)
        out[(k, p)] = v
    return out


def _proj_coef_d4_r2(n):
    """`harmonic.proj_coef(n, 4, 2)[2]`, evaluated so a width change cannot
    silently invalidate it. Verified against upstream at n = 4..4096."""
    d, r, j = 4, 2, 2
    c = d - r + n / 2.0 - 1
    rising = 1.0
    for m in range(j):
        rising *= (1 - c + m)
    return ((-1.0) ** r * (c - r) / 4.0 ** j / math.factorial(r)
            / math.factorial(j - r) / c / rising)


def _get_pk_block(pK, b, d):
    """`DS_part_sum.get_block`: the pK slice a vector-partition block reads."""
    nz = tuple(i for i, x in enumerate(b) if x > 0)
    key = tuple(b[i] for i in nz)
    f = pK[tuple(sorted(key, reverse=True))]
    if len(key) == 2 and key[0] < key[1]:
        f = f.T
    if len(nz) != d:
        f = f[:, None] if nz == (0,) else f[None, :]
    return f


def _layer(W, state, table, dtype, off_mask, init_scale, last_layer):
    """One layer: the linear step, then the nonlinear step. See the module docstring."""
    n = W.shape[0]
    mu, cov, M, c4 = state[0], state[1], state[2], state[3]

    mu_pre = W.T @ mu
    cov_pre = W.T @ cov @ W                                   # 2 matmuls
    P = W.T @ M                                               # 1 matmul
    Q = W.T                                                   # the identity legs
    metric = float(init_scale)

    var = fnp.maximum(fnp.diagonal(cov_pre), dtype(1e-12))
    sigma = fnp.sqrt(var)
    alpha = mu_pre / sigma
    w = _wick(mu_pre, sigma, alpha, table, dtype)

    C11 = cov_pre * off_mask
    PQ = P * Q
    D3 = (PQ * Q).sum(axis=1)
    D21 = ((2.0 * (PQ @ Q.T) + ((Q * Q) @ P.T)) / 3.0) * off_mask   # 2 matmuls
    ones_n = fnp.ones(n, dtype=dtype)
    D4 = (c4 * metric * metric) * ones_n
    D22 = (c4 * metric * metric / 3.0) * fnp.outer(ones_n, ones_n) * off_mask

    S = {(1,): mu_pre, (2,): var, (1, 1): C11, (2, 1): D21, (1, 2): D21.T,
         (3,): D3, (4,): D4, (2, 2): D22}

    pK = {}
    for row in table["terms"]:
        ip = tuple(row["int_part"])
        d = len(ip)
        term = None
        for blk in row["blocks"]:
            f = S[tuple(blk["vec"])]
            term = f if term is None else term * f
        if term is None:
            term = ones_n if d == 1 else fnp.outer(ones_n, ones_n)
        term = term * float(row["scalar"])
        for axis, kp in enumerate(row["wick"]):
            wk = w[(int(kp[0]), int(kp[1]))]
            term = term * (wk if d == 1 else (wk[:, None] if axis == 0 else wk[None, :]))
        pK[ip] = term if ip not in pK else pK[ip] + term
    for ip in list(pK):
        if len(ip) == 2 and ip[0] == ip[1]:
            pK[ip] = (pK[ip] + pK[ip].T) * 0.5

    # WP16 (2026-09-03): the (1,1,1) block USED to be computed here and its
    # repeated part subtracted below. Both are gone, because in a REGENERATING
    # flow the block is dead: `from_dstensor` discards its all-distinct part at
    # the rebuild, and upstream's subtraction of its repeated part from the DS
    # part is undone by adding the block straight back into `K_out[3]` -- whose
    # `__add__` drops the repeated cache, so the regeneration hook reads the
    # UN-subtracted slices. Removing it is worth `4,726,784` FLOPs and 112
    # operations per 256-layer, and it is the whole of the `3.31x` the graded
    # canary missed by. Parity: `local_sim/p2_k3_parity.py`, contract WP16.

    K = {}
    for ip_key, rows in table["pK_to_K"].items():
        ip = _parse_int_part(ip_key)
        d = len(ip)
        acc = None
        for r in rows:
            prod = None
            for b in r["blocks"]:
                f = _get_pk_block(pK, tuple(int(x) for x in b), d)
                prod = f if prod is None else prod * f
            term = prod * float(r["coef"])
            acc = term if acc is None else acc + term
        K[ip] = acc

    mu_out = K[(1,)]
    cov_out = fnp.diag(K[(2,)]) + K[(1, 1)]
    M_out = M if last_layer else (fnp.diag(K[(3,)]) + 3.0 * K[(2, 1)].T)
    lap = table["lap2_d4"]
    c4_out = _proj_coef_d4_r2(n) * (
        float(lap["(4,)"]["terms"][0]["coef"])
        * float(lap["(4,)"]["int_partition_coef"]) * K[(4,)].sum()
        + float(lap["(2, 2)"]["terms"][0]["coef"])
        * float(lap["(2, 2)"]["int_partition_coef"]) * K[(2, 2)].sum())
    return (mu_out, cov_out, M_out, c4_out)


class Estimator(BaseEstimator):
    """The regenerating k_max=3 cumulant flow. A residual probe, not a submission."""

    def setup(self, context) -> None:
        _tables()          # decompress once, OFF-BUDGET

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        table = _tables()
        width = mlp.width
        depth = len(mlp.weights)
        dtype = fnp.float32
        # `copy=False` so an ALREADY-float32 weight costs nothing: the grader's
        # Phase-2 weights are float32, and a plain `.astype` bills n^2 per layer
        # for a cast that does nothing (measured: 65,536 FLOPs at n = 256, 0 with
        # copy=False). The cast must still be here unconditionally -- if a round
        # ever handed float64 weights, flopscope would bill this entire flow at
        # the float64 rate 2.0 and the price would silently double.
        w = [x.astype(dtype, copy=False) for x in mlp.weights]
        off_mask = fnp.ones((width, width), dtype=dtype) - fnp.eye(width, dtype=dtype)
        state = (fnp.zeros(width, dtype=dtype),
                 fnp.eye(width, dtype=dtype),
                 fnp.zeros((width, width), dtype=dtype),
                 0.0)
        mus = []
        for l in range(depth):
            state = _layer(w[l], state, table, dtype, off_mask, 2.0,
                           last_layer=(l == depth - 1))
            mus.append(state[0])
        return fnp.stack(mus)
