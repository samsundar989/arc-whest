"""Separate E14 actual-f32 compact arithmetic backend.

Copied function ASTs from compact_reference (hash recorded by driver), with
only explicit f32 allocation and immutable reassignment replacing +=.
All arithmetic, including QR, is executed through real flopscope.numpy.
No coefficient/age/source recurrence change and no frozen file edit.
"""
import flopscope.numpy as np
from p2_compact_reference import Source, Factors, Shared

def materialize(source):
    return source.P * source.s[None, :] + 3.0 * source.A * source.e[None, :] + source.Z @ source.L.T

def reconstruct(shared):
    if shared.Q is None:
        return []
    return [Source(shared.Q @ f.FA, shared.Q @ f.FP, f.Z, f.L, f.s, f.e, f.w2, f.dA, f.dP, f.c1, f.c2, f.Y, f.age) for f in shared.factors]

def compact_slices(shared, need_d21=True):
    """Reassociate every Q-trailing product before one shared return."""
    if shared.Q is None or not shared.factors:
        return (None, None)
    n, r = shared.Q.shape
    d3 = np.zeros(n, dtype=np.float32)
    coord = np.zeros((n, r), dtype=np.float32) if need_d21 else None
    residual = np.zeros((n, n), dtype=np.float32) if need_d21 else None
    for f in shared.factors:
        A, P = (shared.Q @ f.FA, shared.Q @ f.FP)
        H = f.FP * f.s[None, :] + 3.0 * f.FA * f.e[None, :]
        R = f.Z @ f.L.T
        M = shared.Q @ H + R
        AP, PP, MP = (A * P, P * P, M * P)
        d3 = d3 + (3.0 * A * AP * f.w2[None, :] + MP * P).sum(axis=1)
        has_birth_feed = f.c1 is not None and f.c2 is not None and (f.Y is not None)
        rf = AP @ f.c1 + PP @ f.c2 if has_birth_feed else None
        if has_birth_feed:
            d3 = d3 + rf * f.Y
        if need_d21:
            LA = 2.0 * AP * f.w2[None, :] + PP * f.e[None, :]
            LP = A * A * f.w2[None, :]
            LP = LP + (PP * (f.s[None, :] / 3.0) + 2.0 / 3.0 * MP)
            if has_birth_feed:
                LP = LP + (A * f.c1[None, :] + P * f.c2[None, :]) * (2.0 / 3.0 * f.Y[:, None])
            coord = coord + (LA @ f.FA.T + LP @ f.FP.T)
            residual = residual + PP @ f.L @ f.Z.T / 3.0
            if has_birth_feed:
                residual = residual + rf[:, None] * f.Y[None, :] / 3.0
    return (d3, None if not need_d21 else coord @ shared.Q.T + residual)

def weighted_gram_join(shared, source, rank, omega):
    """Join one dense source using weighted-Gram sketch and thin QR."""
    n = source.A.shape[0]
    rank = min(int(rank), n, omega.shape[1])
    Y = source.A @ (source.dA[:, None] * (source.A.T @ omega))
    Y = Y + source.P @ (source.dP[:, None] * (source.P.T @ omega))
    if shared.Q is not None:
        Y = Y + shared.Q @ (shared.gram @ (shared.Q.T @ omega))
    Qn, _ = np.linalg.qr(Y, mode='reduced')
    Qn = Qn[:, :rank]
    FAj, FPj = (Qn.T @ source.A, Qn.T @ source.P)
    if shared.Q is not None:
        T = Qn.T @ shared.Q
        rotated = [Factors(T @ f.FA, T @ f.FP, f.Z, f.L, f.s, f.e, f.w2, f.dA, f.dP, f.c1, f.c2, f.Y, f.age) for f in shared.factors]
        gram = T @ shared.gram @ T.T
    else:
        rotated, gram = ([], np.zeros((rank, rank), dtype=np.float32))
    fj = Factors(FAj, FPj, source.Z, source.L, source.s, source.e, source.w2, source.dA, source.dP, source.c1, source.c2, source.Y, source.age)
    gram = gram + FAj * source.dA[None, :] @ FAj.T
    gram = gram + FPj * source.dP[None, :] @ FPj.T
    return Shared(Qn, rotated + [fj], gram)

def transport_shared(shared, W, wick):
    if shared.Q is None:
        return shared
    return Shared(W @ (wick[:, None] * shared.Q), [Factors(f.FA, f.FP, W @ (wick[:, None] * f.Z), f.L, f.s, f.e, f.w2, f.dA, f.dP, f.c1, f.c2, None if f.Y is None else W @ (wick * f.Y), f.age + 1) for f in shared.factors], shared.gram)
