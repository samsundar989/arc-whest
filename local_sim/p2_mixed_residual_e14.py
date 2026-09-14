"""E14 Slice A: full f32 residual and common f64 A/P basis.

This module substitutes only F21's tower primitives in a separately loaded copy
of the frozen module. Its nonlinear, Wick, centering, scalar and birth program
remains the frozen `_layer` function. No residual is decoded through a basis.
"""
from __future__ import annotations

from dataclasses import dataclass

import flopscope.numpy as fnp


@dataclass
class MixedTower:
    D: object                 # f32, every source, oldest first
    Q: object                 # f64 common old A/P basis, or None
    FA: object                # f64 packed old coordinates, or None
    FP: object
    G: object                 # f64 weighted coordinate Gram, or None
    young_A: object           # f32 newest A
    young_P: object           # None at birth (symbolic I), otherwise f32
    w: object                 # f32 birth weights, all ages
    k: int


def install(frozen, rank=None):
    """Install the mixed primitives into a private loaded F21 module instance."""
    original = {key: getattr(frozen, key) for key in
                ('_advance', '_wick_tower', '_dslice3', '_dslice21',
                 '_append', '_truncate')}
    query_cache = {}

    def advance(WT, tw):
        query_cache.clear()
        n = WT.shape[0]
        D = frozen._mm(WT, tw.D)
        Q = None if tw.Q is None else WT.astype(fnp.float64) @ tw.Q
        A = frozen._mm(WT, tw.young_A)
        P = WT if tw.young_P is None else frozen._mm(WT, tw.young_P)
        return MixedTower(D, Q, tw.FA, tw.FP, tw.G, A, P, tw.w, tw.k)

    def slices(tw, off_mask=None):
        n, k = tw.D.shape[0], tw.k
        m = k - 1
        cached = query_cache.get(id(tw))
        if cached is None:
            yd = tw.D[:, m*n:]
            yw = tw.w[m*n:]
            young = frozen._Tower(yd, tw.young_A, tw.young_P, yw, 1)
            y3 = original['_dslice3'](young)
            if m:
                # The only dense reconstructions are the two shared-basis legs.
                A = tw.Q @ tw.FA
                P = tw.Q @ tw.FP
                R = tw.D[:, :m*n].astype(fnp.float64)
                w = tw.w[:m*n].astype(fnp.float64)
                C = A * w
                PP = P * P
                old3 = (R*P*P).sum(axis=1) + 3.0*(A*P*C).sum(axis=1)
                all3 = (old3 + y3.astype(fnp.float64)).astype(fnp.float32)
            else:
                A=P=R=w=C=PP=None
                all3=y3
            cached=(young,A,P,R,w,C,PP,all3)
            query_cache[id(tw)] = cached
        young,A,P,R,w,C,PP,all3=cached
        if off_mask is None:
            return all3, None
        y21 = original['_dslice21'](young, off_mask)
        if m == 0:
            return all3, y21
        LA = 2.0*(A*P)*w
        LP = A*C + (2.0/3.0)*(R*P)
        X = LA @ tw.FA.T + LP @ tw.FP.T
        # Both operands are f32, including the residual's left operand.
        Y32 = frozen._mm(PP.astype(fnp.float32), tw.D[:, :m*n].T)
        old21 = (X @ tw.Q.T + Y32.astype(fnp.float64)/3.0)
        old21 = old21 * off_mask.astype(fnp.float64)
        all21 = (old21 + y21.astype(fnp.float64)).astype(fnp.float32)
        return all3, all21

    def dslice3(tw):
        return slices(tw)[0]

    def dslice21(tw, off_mask):
        return slices(tw, off_mask)[1]

    def wick(col, tw):
        # Advanced old Q and all D get this layer's Wick once. The newborn has
        # not yet been appended and receives no scale here.
        return MixedTower(tw.D*col, None if tw.Q is None else
                          tw.Q*col.astype(fnp.float64), tw.FA, tw.FP, tw.G,
                          tw.young_A*col, tw.young_P*col, tw.w, tw.k)

    def append(tw, A_D, A_L, w2b, n):
        if tw is None:
            return MixedTower(A_D, None, None, None, None, A_L, None, w2b, 1)
        r = n if rank is None else min(int(rank), n)
        Q, FA, FP, G = tw.Q, tw.FA, tw.FP, tw.G
        B = tw.young_A.astype(fnp.float64)
        E = tw.young_P.astype(fnp.float64)
        a = 9.0 + tw.w[-n:].astype(fnp.float64)**2
        Omega = fnp.eye(n, dtype=fnp.float64)[:, :r]
        H = B @ (a[:, None]*(B.T @ Omega)) + E @ (E.T @ Omega)
        if Q is not None:
            H = H + Q @ (G @ (Q.T @ Omega))
        Qnew = fnp.linalg.qr(H, mode='reduced')[0]
        FAnew = Qnew.T @ B
        FPnew = Qnew.T @ E
        Gnew = (FAnew*a[None, :]) @ FAnew.T + FPnew @ FPnew.T
        if Q is not None:
            T = Qnew.T @ Q
            FAold = T @ FA
            FPold = T @ FP
            Gnew = Gnew + T @ G @ T.T
            FAnew = fnp.concatenate([FAold, FAnew], axis=1)
            FPnew = fnp.concatenate([FPold, FPnew], axis=1)
        return MixedTower(fnp.concatenate([tw.D, A_D], axis=1), Qnew,
                          FAnew, FPnew, Gnew, A_L, None,
                          fnp.concatenate([tw.w, w2b]), tw.k+1)

    def truncate(tw, gens, n):
        # Exact all-age initial discriminator. A later budget fallback must
        # synchronize D, factors and w; it is not silently inherited here.
        if gens is not None and tw.k > gens:
            raise RuntimeError('mixed all-age kernel has no generation fallback')
        return tw

    frozen._advance = advance
    frozen._wick_tower = wick
    frozen._dslice3 = dslice3
    frozen._dslice21 = dslice21
    frozen._append = append
    frozen._truncate = truncate
    return original
