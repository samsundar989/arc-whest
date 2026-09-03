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
    """Horner with PYTHON floats, seeded from a float rather than a vector.

    `np.float64(c)` would be CONCRETE under NEP-50 and would promote an f32 array
    on contact, and flopscope bills float64 at twice the float32 rate -- so a
    single numpy scalar here doubles the bill for the whole downstream graph.

    The seed used to be `x * 0.0 + float(coefs[-1])`, two metered operations to
    materialise a constant vector. Seeding from the float is the same value --
    `0.0 * x` is signed zero for finite `x`, and `+/-0 + c` is `c` exactly.
    """
    y = float(coefs[-1])
    for c in reversed(coefs[:-1]):
        y = y * x + float(c)
    return y


def _horner_batch(C, A):
    """Horner for a stack of polynomials at once. `C` is `(m, L)`, `A` is `(1, n)`.

    Rows are padded at the HIGH end with zeros, which is bit-exact: a padded row
    starts at `0`, and `0 * a + 0 == 0` then `0 * a + c == c` in IEEE. Twelve
    polynomials cost `2 (L - 1)` operations instead of `2 sum_i (L_i - 1)`.
    """
    y = C[:, -1:]
    for j in range(C.shape[1] - 2, -1, -1):
        y = y * A + C[:, j:j + 1]
    return y


def _wick_program(table, dtype):
    """The width- and weight-INDEPENDENT half of `_wick`. Built ONCE, in `setup`.

    Under `lambda = 0` the binding cost is the per-MLP OPERATION COUNT, not the
    arithmetic, so anything that does not depend on the MLP must not be inside
    `predict`. This depends on the frozen `(k, p)` table alone.
    """
    poly_keys, P1s, P2s, pow_deg, other = [], [], [], [], []
    for _key, spec in table["wick"].items():
        k, p = int(spec["k"]), int(spec["p"])
        if spec["branch"] == "poly":
            poly_keys.append((k, p))
            P1s.append([float(c) for c in spec["P1"]])
            P2s.append([float(c) for c in spec["P2"]])
            pow_deg.append(p - k)
        else:
            other.append((k, p, spec["branch"],
                          float(spec.get("factorial_p", 1.0)),
                          int(spec.get("k_reduced", k))))
    L1 = max(len(c) for c in P1s)
    L2 = max(len(c) for c in P2s)

    def pad(rows, L):
        return [r + [0.0] * (L - len(r)) for r in rows]

    prog = {"poly_keys": poly_keys, "pow_deg": pow_deg, "other": other,
            "C1": fnp.array(pad(P1s, L1), dtype=dtype),
            "C2": fnp.array(pad(P2s, L2), dtype=dtype),
            "max_sigma_pow": max(pow_deg + [3])}

    # The one-dimensional fast path, decided here against the frozen table so
    # `predict` never re-derives it. See `_layer`.
    ones_d = [r for r in table["terms"] if len(r["int_part"]) == 1]
    ps = sorted({r["int_part"][0] for r in ones_d})
    want = sorted([((), 1.0), (((3,),), round(1.0 / 6.0, 12)),
                   (((4,),), round(1.0 / 24.0, 12))])
    ok = len(ones_d) == 3 * len(ps) and all(
        sorted((tuple(tuple(b["vec"]) for b in r["blocks"]),
                round(float(r["scalar"]), 12))
               for r in ones_d if r["int_part"][0] == p) == want for p in ps)
    prog["fast_1d"] = ok
    prog["ps"] = ps
    if ok:
        by = {}
        for r in ones_d:
            key = () if not r["blocks"] else tuple(r["blocks"][0]["vec"])
            by[(r["int_part"][0], key)] = (int(r["wick"][0][0]), int(r["wick"][0][1]))
            if key == (3,):
                prog["s3c"] = float(r["scalar"])
            elif key == (4,):
                prog["s4c"] = float(r["scalar"])
        prog["w0_keys"] = [by[(p, ())] for p in ps]
        prog["w3_keys"] = [by[(p, (3,))] for p in ps]
        prog["w4_keys"] = [by[(p, (4,))] for p in ps]
    return prog


def _wick(mean, sigma, alpha, table, dtype, const):
    """`relu_wick_coef` for the 16 (k, p) pairs the layer needs. O(n) each.

    The two `flops.stats.*` calls are the only place a float64 can enter --
    flopscope promotes their inputs "to match scipy.stats" -- and both are cast
    back immediately. Without the casts the covariance accumulator, where nearly
    all the cost lives, is float64 in an arm that calls itself float32.

    **This used to be 144 of a layer's 315 metered operations**, to produce
    sixteen `(n,)` vectors. Four bit-exact changes take it to 40: `sigma^j` built
    once in the same left-associated order the per-pair loop used, `base(k)`
    memoised instead of recomputed for each consumer, the `1.0 * ones *` factors
    dropped as the identities they are, and the twelve polynomials evaluated as
    one padded batched Horner.
    """
    Phi = flops.stats.norm.cdf(alpha).astype(dtype)
    phi = flops.stats.norm.pdf(alpha).astype(dtype)

    sig_pow = {1: sigma}
    for j in range(2, max(const["max_sigma_pow"], 3) + 1):
        sig_pow[j] = sig_pow[j - 1] * sigma

    cache = {}

    def base(k):
        if k in cache:
            return cache[k]
        if k == 0:
            v = sigma * phi + mean * Phi
        elif k == 1:
            v = Phi
        elif k == 2:
            v = phi / sig_pow[1]
        elif k == 3:
            v = (-1.0) * alpha * phi / sig_pow[2]
        elif k == 4:
            v = (alpha * alpha - 1.0) * phi / sig_pow[3]
        else:
            raise ValueError(k)
        cache[k] = v
        return v

    A = alpha[None, :]
    P1 = _horner_batch(const["C1"], A)
    P2 = _horner_batch(const["C2"], A)
    S = fnp.stack([sig_pow[d] for d in const["pow_deg"]])
    V = S * (P1 * phi[None, :] + P2 * Phi[None, :])

    out = {}
    for i, kp in enumerate(const["poly_keys"]):
        out[kp] = V[i]
    for k, p, branch, fac, k_red in const["other"]:
        out[(k, p)] = base(k) if branch == "base" else fac * base(k_red)
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


def _layer(W, state, table, dtype, const, init_scale, last_layer):
    """One layer: the linear step, then the nonlinear step. See the module docstring.

    `const` carries everything that does not depend on the MLP -- the diagonal
    kill mask, the all-ones vector and the batched Wick coefficient stacks. Under
    `lambda = 0` the binding cost is the OPERATION COUNT per MLP, so anything
    rebuilt per layer that could have been built once is pure loss.

    **Last-layer pruning.** The grader reads post-ReLU MEANS. At the final layer
    there is no next layer, so the only output ever read is
    `mu_out = K[(1,)] = pK[(1,)]`, which needs `w(0,1)`, `w(3,1)`, `w(4,1)`, `D3`
    and `D4` and nothing else. The `(2,1)` slice's two matmuls, `C11`, `D22`,
    twenty-two of the twenty-five pK terms, six of the seven `pK -> K` slices,
    the covariance, `K[4]` and the rebuild are all skipped there.
    """
    n = W.shape[0]
    off_mask, ones_n = const["off_mask"], const["ones_n"]
    mu, cov, M, c4 = state[0], state[1], state[2], state[3]

    mu_pre = W.T @ mu
    cov_pre = W.T @ cov @ W                                   # 2 matmuls
    P = W.T @ M                                               # 1 matmul
    Q = W.T                                                   # the identity legs
    metric = float(init_scale)

    var = fnp.maximum(fnp.diagonal(cov_pre), dtype(1e-12))
    sigma = fnp.sqrt(var)
    alpha = mu_pre / sigma
    w = _wick(mu_pre, sigma, alpha, table, dtype, const["wick"])

    PQ = P * Q
    D3 = (PQ * Q).sum(axis=1)
    D4 = (c4 * metric * metric) * ones_n
    if last_layer:
        S = {(1,): mu_pre, (3,): D3, (4,): D4}
    else:
        C11 = cov_pre * off_mask
        D21 = ((2.0 * (PQ @ Q.T) + ((Q * Q) @ P.T)) / 3.0) * off_mask  # 2 matmuls
        # `outer(ones, ones)` is exactly an all-ones matrix, so scaling the mask
        # directly is the same floats for two fewer operations.
        D22 = (c4 * metric * metric / 3.0) * off_mask
        S = {(1,): mu_pre, (2,): var, (1, 1): C11, (2, 1): D21, (1, 2): D21.T,
             (3,): D3, (4,): D4, (2, 2): D22}

    pK = {}
    # THE ONE-DIMENSIONAL SLICES, ALL FOUR AT ONCE. The frozen table gives every
    # `(p,)` the same three terms:
    #
    #     pK[(p,)] = w(0,p) + (1/6) D3 w(3,p) + (1/24) D4 w(4,p)
    #
    # so the four slices are one `(4, n)` expression sharing `(1/6) D3` and
    # `(1/24) D4`. Twelve terms and eight accumulations become nine operations,
    # and because every float is multiplied and added in the order the per-term
    # loop used, the result is bit-identical -- a regrouping, not a
    # reassociation. `prog["fast_1d"]` is decided ONCE, in `setup`, against the
    # frozen table; if a table ever fell out of this shape the generic loop below
    # still computes it.
    prog = const["wick"]
    if prog["fast_1d"]:
        u3 = S[(3,)] * prog["s3c"]
        u4 = S[(4,)] * prog["s4c"]
        W0 = fnp.stack([w[kp] for kp in prog["w0_keys"]])
        W3 = fnp.stack([w[kp] for kp in prog["w3_keys"]])
        W4 = fnp.stack([w[kp] for kp in prog["w4_keys"]])
        PK1 = W0 + u3[None, :] * W3
        PK1 = PK1 + u4[None, :] * W4
        for i, p in enumerate(prog["ps"]):
            pK[(p,)] = PK1[i]

    blocks_seen = {}
    for row in table["terms"]:
        ip = tuple(row["int_part"])
        d = len(ip)
        if last_layer and ip != (1,):
            continue
        if d == 1 and prog["fast_1d"]:
            continue
        # The same block product appears in several int_parts -- `C11*C11` in all
        # three two-dimensional slices, `D22` likewise -- and was recomputed each
        # time. Memoised on the block tuple: a reuse, not a reassociation.
        bkey = tuple(tuple(blk["vec"]) for blk in row["blocks"])
        term = blocks_seen.get(bkey)
        if term is None and bkey:
            for blk in row["blocks"]:
                f = S[tuple(blk["vec"])]
                term = f if term is None else term * f
            blocks_seen[bkey] = term
        scalar = float(row["scalar"])
        if term is None:
            # The empty vec_part is the all-ones tensor; starting from the
            # scalar is the same value for fewer operations. Every such term in
            # this table is d = 1.
            term = scalar
        elif scalar != 1.0:
            term = term * scalar
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
    # UN-subtracted slices. It is the whole of the `3.31x` the graded canary
    # missed by. Parity: `local_sim/p2_k3_parity.py`, contract WP16.

    K = {}
    for ip_key, rows in table["pK_to_K"].items():
        ip = _parse_int_part(ip_key)
        if last_layer and ip != (1,):
            continue
        d = len(ip)
        acc = None
        for r in rows:
            prod = None
            for b in r["blocks"]:
                f = _get_pk_block(pK, tuple(int(x) for x in b), d)
                prod = f if prod is None else prod * f
            # `* 1.0` (7 of 19 rows) and `* -1.0` (1 row) are identities; the
            # second folds into a subtraction, `a + (-b) == a - b` in IEEE.
            coef = float(r["coef"])
            if coef == 1.0:
                acc = prod if acc is None else acc + prod
            elif coef == -1.0 and acc is not None:
                acc = acc - prod
            else:
                term = prod * coef
                acc = term if acc is None else acc + term
        K[ip] = acc

    mu_out = K[(1,)]
    if last_layer:
        return (mu_out, None, None, 0.0)
    cov_out = fnp.diag(K[(2,)]) + K[(1, 1)]
    M_out = fnp.diag(K[(3,)]) + 3.0 * K[(2, 1)].T
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
        # Decompress the frozen tables and build the width-INDEPENDENT half of
        # the Wick program, both OFF-BUDGET and once for the whole run.
        # `SetupContext` carries only `seed`, `scratch_dir` and `submission_dir`,
        # so the width-dependent constants cannot be built here; they are cached
        # per width on the first `predict` that sees that width.
        self._const = {}
        _tables()
        self._wick_program = _wick_program(_tables(), fnp.float32)

    def _constants(self, width, dtype):
        """The MLP-independent arrays, built once per width and then reused.

        The mask and the ones vector depend on the width alone, never on the
        weights or the seed, so rebuilding them for every MLP -- or worse, for
        every layer -- is operations spent on a constant. One `1024 x 1024`
        float32 mask is 4 MB held for the run, against an 8,192 MB cap.
        """
        c = self._const.get(width)
        if c is None:
            c = {"off_mask": (fnp.ones((width, width), dtype=dtype)
                              - fnp.eye(width, dtype=dtype)),
                 "ones_n": fnp.ones(width, dtype=dtype),
                 "wick": self._wick_program}
            self._const[width] = c
        return c

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
        const = self._constants(width, dtype)
        state = (fnp.zeros(width, dtype=dtype),
                 fnp.eye(width, dtype=dtype),
                 fnp.zeros((width, width), dtype=dtype),
                 0.0)
        mus = []
        for l in range(depth):
            state = _layer(w[l], state, table, dtype, const, 2.0,
                           last_layer=(l == depth - 1))
            mus.append(state[0])
        return fnp.stack(mus)
