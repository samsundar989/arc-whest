"""Compact all-history K3/K4 candidate template.

This is the grader-facing spelling of the independently checked total-core
algorithm in ``p2_compact_reference.py``. The four mechanism arms differ only
in the literal constants below. ``p2_build_k4_arms.py`` owns those replacements.

Every K3 source is retained. The four most recent sources remain dense; older
sources share a rank-384 weighted-Gram basis and keep rank-16 residuals. The
implementation uses only grader packages and never reads labels or protected data.
"""

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator


_K4_CONSUMERS = False  # __K4_CONSUMERS__
_K4_BIRTHS = False  # __K4_BIRTHS__
_K4_ACTIVE = False  # __K4_ACTIVE__
_K4_LAM = 0.33333333333333331  # __K4_LAM__

_RECENT = 4
_SHARED_RANK = 384
_RESIDUAL_RANK = 16
_INV_SQRT_2PI = 0.39894228040143267794


def _transport_source(source, W, wick):
    A, P, Z, L, s, e, w2, dA, dP, c1, c2, Y, age = source
    return (W @ (wick[:, None] * A), W @ (wick[:, None] * P),
            W @ (wick[:, None] * Z), L, s, e, w2, dA, dP, c1, c2,
            None if Y is None else W @ (wick * Y), age + 1)


def _insert_birth(birth, W):
    a, R, L, s, e, w2, dA, dP, c1, c2, y = birth
    return (W @ a, fnp.copy(W), W @ R, L, s, e, w2, dA, dP,
            c1, c2, None if y is None else W @ y, 0)


def _join_and_transport(shared, source, W, wick, rank, dtype):
    """Move one dense source into the shared weighted-Gram basis."""
    A, P, Z, L, s, e, w2, dA, dP, c1, c2, Yfeed, age = source
    A = wick[:, None] * A
    P = wick[:, None] * P
    omega = W[:, :rank]
    Qp = None if shared is None else wick[:, None] * shared[0]
    sketch = A @ (dA[:, None] * (A.T @ omega))
    sketch = sketch + P @ (dP[:, None] * (P.T @ omega))
    if shared is not None:
        sketch = sketch + Qp @ (shared[2] @ (Qp.T @ omega))
    Qn = fnp.linalg.qr(sketch)[0][:, :rank]
    FAj = Qn.T @ A
    FPj = Qn.T @ P
    if shared is None:
        factors = []
        gram = fnp.zeros((rank, rank), dtype=dtype)
    else:
        _Q0, old_factors, old_gram = shared
        rotation = Qn.T @ Qp
        factors = []
        for fact in old_factors:
            (FA, FP, Zi, Li, si, ei, w2i, dAi, dPi,
             c1i, c2i, Yi, ai) = fact
            factors.append((rotation @ FA, rotation @ FP, Zi, Li, si, ei,
                            w2i, dAi, dPi, c1i, c2i, Yi, ai))
        gram = rotation @ (old_gram @ rotation.T)
    gram = gram + ((FAj * dA[None, :]) @ FAj.T
                   + (FPj * dP[None, :]) @ FPj.T)
    factors.append((FAj, FPj, Z, L, s, e, w2, dA, dP,
                    c1, c2, Yfeed, age))
    Qout = W @ Qn
    transported = []
    for fact in factors:
        (FA, FP, Zi, Li, si, ei, w2i, dAi, dPi,
         c1i, c2i, Yi, ai) = fact
        Yout = None if Yi is None else W @ (wick * Yi)
        transported.append((FA, FP, W @ (wick[:, None] * Zi), Li, si, ei,
                            w2i, dAi, dPi, c1i, c2i, Yout, ai + 1))
    return (Qout, transported, gram)


def _transport_shared(shared, W, wick):
    if shared is None:
        return None
    Q, factors, gram = shared
    Qout = W @ (wick[:, None] * Q)
    transported = []
    for fact in factors:
        (FA, FP, Z, L, s, e, w2, dA, dP,
         c1, c2, Yfeed, age) = fact
        Yout = None if Yfeed is None else W @ (wick * Yfeed)
        transported.append((FA, FP, W @ (wick[:, None] * Z), L, s, e,
                            w2, dA, dP, c1, c2, Yout, age + 1))
    return (Qout, transported, gram)


def _history_slices(young, shared, n, rank, dtype, last):
    """Evaluate D3/D21 with one shared return for all compressed sources."""
    sources = [(False, source) for source in young]
    if shared is not None:
        Q, factors, _gram = shared
        for fact in factors:
            (FA, FP, Z, L, s, e, w2, dA, dP,
             c1, c2, Yfeed, age) = fact
            A = Q @ FA
            P = Q @ FP
            sources.append((True, (A, P, Z, L, s, e, w2, dA, dP,
                                   c1, c2, Yfeed, age, FA, FP)))
    coord = (fnp.zeros((n, rank), dtype=dtype)
             if shared is not None and not last else None)
    residual = (fnp.zeros((n, n), dtype=dtype)
                if shared is not None and not last else None)
    D3 = fnp.zeros(n, dtype=dtype)
    D21 = None if last else fnp.zeros((n, n), dtype=dtype)
    for compressed, source in sources:
        A, P, Z, L, s, e, w2, _dA, _dP, c1, c2, Yfeed = source[:12]
        AP = A * P
        PP = P * P
        R = Z @ L.T
        MP = R * P + PP * s[None, :] + 3.0 * AP * e[None, :]
        LA = 2.0 * AP * w2[None, :] + PP * e[None, :]
        LP = ((A * A) * w2[None, :] + PP * (s[None, :] / 3.0)
              + (2.0 / 3.0) * MP)
        RF = None
        if Yfeed is not None:
            RF = AP @ c1 + PP @ c2
            LP = LP + (A * c1[None, :] + P * c2[None, :]) * (
                (2.0 / 3.0) * Yfeed[:, None])
        part3 = (3.0 * A * AP * w2[None, :] + MP * P).sum(axis=1)
        if Yfeed is not None:
            part3 = part3 + RF * Yfeed
        D3 = D3 + part3
        if not last:
            if compressed:
                FA, FP = source[13], source[14]
                coord = coord + LA @ FA.T + LP @ FP.T
                residual = residual + ((PP @ L) @ Z.T) / 3.0
                if Yfeed is not None:
                    residual = residual + (RF[:, None] * Yfeed[None, :]) / 3.0
            else:
                part21 = LA @ A.T + LP @ P.T + ((PP @ L) @ Z.T) / 3.0
                if Yfeed is not None:
                    part21 = part21 + (RF[:, None] * Yfeed[None, :]) / 3.0
                D21 = D21 + part21
    if shared is not None and not last:
        D21 = D21 + coord @ shared[0].T + residual
    return D3, D21


def _layer(W, state, n, rank, residual_rank, dtype, last):
    mu, C, wick, young, shared, newborn, gcore = state
    C_old = C
    mu_pre = W @ mu
    C_pre = (W @ C) @ W.T
    var_pre = fnp.diagonal(C_pre)
    joining = bool(not last and young and young[0][12] + 1 >= _RECENT)
    if joining:
        joiner, young = young[0], young[1:]
        shared = _join_and_transport(shared, joiner, W, wick, rank, dtype)
    else:
        shared = _transport_shared(shared, W, wick)
    young = [_transport_source(source, W, wick) for source in young]
    if newborn is not None:
        young.append(_insert_birth(newborn, W))
    D3, D21 = _history_slices(young, shared, n, rank, dtype, last)

    safe_var = fnp.maximum(var_pre, dtype(1.0e-8))
    sigma = fnp.sqrt(safe_var)
    alpha = mu_pre / sigma
    phi = flops.stats.norm.pdf(alpha).astype(dtype)
    Phi = flops.stats.norm.cdf(alpha).astype(dtype)
    w1 = Phi
    w2 = phi / sigma
    w3 = -alpha * phi / safe_var
    w4 = ((alpha * alpha - 1.0) * phi / (safe_var * sigma)
          if _K4_ACTIVE and _K4_CONSUMERS else None)
    base = sigma * phi + mu_pre * Phi
    C_off = C_pre - fnp.diag(var_pre)

    dG = None
    if _K4_ACTIVE:
        if gcore is None:
            dG = fnp.zeros(n, dtype=dtype)
        else:
            WW = W * W
            old_var = fnp.diagonal(C_old)
            dG = WW @ (gcore - _K4_LAM * old_var) + _K4_LAM * var_pre
    if _K4_ACTIVE and _K4_CONSUMERS:
        D4 = 2.0 * dG
        D22 = (dG[:, None] + dG[None, :]) / 3.0
        D31 = (C_off / 3.0 if _K4_LAM == 1.0 / 3.0
               else _K4_LAM * C_off)
        mu = base + D3 * w3 / 6.0 + D4 * w4 / 24.0
    else:
        D4 = None
        mu = base + D3 * w3 / 6.0
    if last:
        return (mu, None, None, young, shared, None, None)

    e2 = ((safe_var + mu_pre * mu_pre) * Phi + mu_pre * sigma * phi)
    var_out = fnp.maximum(e2 - base * base, dtype(1.0e-8))
    K11 = (C_off * (w1[:, None] * w1[None, :])
           + 0.5 * (C_off * C_off) * (w2[:, None] * w2[None, :]))
    K11 = K11 + 0.25 * (D21 * (w2[:, None] * w1[None, :])
                         + D21.T * (w1[:, None] * w2[None, :]))
    if _K4_ACTIVE and _K4_CONSUMERS:
        K11 = (K11 + D22 * (w2[:, None] * w2[None, :]) / 4.0
               + (D31 + D31.T)
               * (w3[:, None] * w1[None, :]) / 12.0)
    off_mask = (fnp.ones((n, n), dtype=dtype)
                - fnp.eye(n, dtype=dtype))
    C = fnp.diag(var_out) + K11 * off_mask

    e3 = ((mu_pre * mu_pre * mu_pre + 3.0 * mu_pre * safe_var) * Phi
          + (mu_pre * mu_pre * sigma + 2.0 * sigma * safe_var) * phi)
    k3v = e3 - 3.0 * base * e2 + 2.0 * base * base * base
    next_gcore = None
    if _K4_ACTIVE:
        e4 = ((mu_pre**4 + 6.0 * mu_pre * mu_pre * safe_var
               + 3.0 * safe_var * safe_var) * Phi
              + (mu_pre**3 * sigma + 5.0 * mu_pre * sigma * safe_var) * phi)
        if _K4_ACTIVE and _K4_CONSUMERS:
            k4v = (e4 - 4.0 * base * e3 + 6.0 * base * base * e2
                   - 3.0 * base**4 + D4 * w1**4)
            k22 = (((k4v[:, None] + k4v[None, :]) / 6.0
                    + D22 * (w1[:, None] * w1[None, :])) * off_mask)
        else:
            k4v = (e4 - 4.0 * base * e3 + 6.0 * base * base * e2
                   - 3.0 * base**4)
            k22 = ((k4v[:, None] + k4v[None, :]) / 6.0) * off_mask
        h4 = k4v + k22.sum(axis=1)
        next_gcore = ((6.0 / (n + 4.0)) * h4
                      - (3.0 / ((n + 2.0) * (n + 4.0))) * h4.sum()) / 2.0

    a = w1[:, None] * C_off
    local21 = ((w1[:, None] * w1[:, None]) * w2[None, :] * (C_off * C_off)
               + 2.0 * (w2 * w1 * var_pre)[:, None]
               * w1[None, :] * C_off)
    old21 = (w1[:, None] * w1[:, None]) * D21 * w1[None, :]
    if _K4_ACTIVE and _K4_BIRTHS:
        w1sq = w1 * w1
        y = 0.5 * w2
        dgw = w1sq * dG
        u = y * dG
        c1 = (w1 / 3.0 if _K4_LAM == 1.0 / 3.0
              else _K4_LAM * w1)
        c2 = dgw
        feed21 = (dgw[:, None] * y[None, :] / 3.0
                  + (2.0 / 3.0) * y[:, None] * (a * c1[None, :])
                  + w1sq[:, None] * u[None, :] / 3.0)
        feed3 = dgw * y + u * w1sq
    else:
        y = None
        c1 = None
        c2 = None
    if _K4_ACTIVE and _K4_BIRTHS:
        rep21 = old21 + (a * a) * w2[None, :] + feed21
        K21 = old21 + local21 + feed21
        s = k3v - D3 * (w1**3) - feed3
    else:
        rep21 = old21 + (a * a) * w2[None, :]
        K21 = old21 + local21
        s = k3v - D3 * (w1**3)
    e = w2 - 2.0 * mu * w1
    residual = K21 - rep21 - e[:, None] * C_off * w1[None, :]
    omega = W[:, :residual_rank]
    Qr = fnp.linalg.qr(residual @ omega)[0][:, :residual_rank]
    Qr = fnp.linalg.qr(residual @ (residual.T @ Qr))[0][:, :residual_rank]
    R = (Qr.T @ residual).T * 3.0
    dA = 9.0 + w2 * w2 + 9.0 * e * e
    dP = 1.0 + s * s
    newborn = (a, R, Qr, s, e, w2, dA, dP, c1, c2, y)
    return (mu, C, w1, young, shared, newborn, next_gcore)


class Estimator(BaseEstimator):
    """Compact deterministic all-history estimator."""

    last_run = None

    def setup(self, context) -> None:
        self.last_run = None

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        width = mlp.width
        depth = len(mlp.weights)
        dtype = fnp.float32
        rank = min(_SHARED_RANK, width)
        residual_rank = min(_RESIDUAL_RANK, width)
        state = (fnp.zeros(width, dtype=dtype),
                 fnp.eye(width, dtype=dtype),
                 fnp.ones(width, dtype=dtype), [], None, None, None)
        outputs = []
        for layer in range(depth):
            raw = mlp.weights[layer]
            W = raw.T if raw.dtype == dtype else raw.T.astype(dtype)
            state = _layer(W, state, width, rank, residual_rank, dtype,
                           layer == depth - 1)
            mu = fnp.where(fnp.isfinite(state[0]), state[0], dtype(0.0))
            state = (mu, state[1], state[2], state[3], state[4],
                     state[5], state[6])
            outputs.append(mu)

        raw0 = mlp.weights[0]
        w64 = raw0 if raw0.dtype == fnp.float64 else raw0.astype(fnp.float64)
        exact0 = fnp.sqrt((w64 * w64).sum(axis=0)) * _INV_SQRT_2PI
        row0 = fnp.reshape(exact0, (1, width))
        out = row0 if depth == 1 else fnp.concatenate(
            [row0, fnp.stack(outputs[1:]).astype(fnp.float64)], axis=0)
        out = fnp.where(fnp.isfinite(out), out, fnp.float64(0.0))
        out = fnp.maximum(out, fnp.float64(0.0))
        self.last_run = {"mode": "compact", "width": width, "depth": depth,
                         "recent": _RECENT, "rank": rank,
                         "residual_rank": residual_rank,
                         "k4_consumers": _K4_CONSUMERS,
                         "k4_births": _K4_BIRTHS, "k4_lam": _K4_LAM}
        return out
