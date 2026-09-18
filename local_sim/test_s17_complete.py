"""Tiny label-free checks for the complete P3/shared-gram composition."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from k4feed_oracle import _load  # noqa: E402
from mint_provenance import canonical_mint  # noqa: E402
from s1_parity_v32 import _predict  # noqa: E402
from s6_parity import layer_loop  # noqa: E402


def close(a, b, tol=1e-9):
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) <= tol * max(1.0, np.max(np.abs(b)))


def test_tensor():
    mod = _load(str(ROOT / "estimator_k3_v43_complete_c01.py"))
    rng = np.random.default_rng(271828)
    n, ko, recent, r, p2r = 5, 2, 2, 3, 2
    k = ko + recent
    Q, _ = np.linalg.qr(rng.normal(size=(n, r)))
    FA, FM = rng.normal(size=(r, ko*n)), rng.normal(size=(r, ko*n))
    Ar, Mr = rng.normal(size=(n, recent*n)), rng.normal(size=(n, recent*n))
    tw = mod._Tower(None, Ar, Mr, rng.normal(size=k*n), k,
                    rng.normal(size=(n, k*p2r)), rng.normal(size=(k, n, p2r)),
                    rng.normal(size=k*n), rng.normal(size=k*n))
    tw.Q, tw.FA, tw.FM, tw.ko = Q, FA, FM, ko
    tw.F = rng.normal(size=(n, 2*k*r))
    tw.R3, tw.R4 = rng.normal(size=(k,n,r)), rng.normal(size=(k,n,r))
    tw.w3b = rng.normal(size=k*n)
    A = np.concatenate([Q @ FA, Ar], axis=1)
    M = np.concatenate([Q @ FM, Mr], axis=1)
    Gp, TT, d3 = mod._p3_parts(tw)
    # Independent explicit symmetric rank-one tensor expansion. No generator
    # helper contributes to this oracle.
    T = np.zeros((n,n,n))
    for j in range(k):
        for c in range(n):
            ix = j*n+c
            f1 = tw.F[:,2*j*r:2*j*r+r] @ tw.R3[j,c]
            f2 = tw.F[:,2*j*r+r:2*(j+1)*r] @ tw.R4[j,c]
            for f, w in ((f1, tw.w3b[ix]), (f2, tw.w2b[ix])):
                a, m, ac = f, M[:,ix], A[:,ix]*w
                T += (np.einsum('a,b,c->abc',a,m,ac)
                      + np.einsum('a,b,c->abc',a,ac,m)
                      + np.einsum('a,b,c->abc',m,a,ac)
                      + np.einsum('a,b,c->abc',m,ac,a)
                      + np.einsum('a,b,c->abc',ac,a,m)
                      + np.einsum('a,b,c->abc',ac,m,a)) / 2.0
    assert close(d3, np.einsum('aaa->a', T))
    off = 1-np.eye(n)
    blank = mod._Tower(None, Ar, Mr, tw.w2b, k, tw.Z, tw.L, tw.sb, tw.eb)
    blank.Q, blank.FA, blank.FM, blank.ko = Q, FA, FM, ko
    got = np.asarray(mod._dslice21(tw, off)) - np.asarray(mod._dslice21(blank, off))
    expected = np.einsum('aab->ab', T)*off
    assert close(got, expected, 3e-8), np.max(np.abs(got-expected))


def test_join_and_disabled():
    base = _load(str(ROOT / "estimator_k3_v41_pairq_a4r256_gram_p4d5.py"))
    off = _load(str(ROOT / "estimator_k3_v43_complete_c00.py"))
    on = _load(str(ROOT / "estimator_k3_v43_complete_c01.py"))
    w, es, _ = canonical_mint(24, 7, 90000000695)
    a, fa = _predict(base, w, 24, 7, es)
    b, fb = _predict(off, w, 24, 7, es)
    assert np.array_equal(a,b) and fa == fb
    on._SHARED_AGE, on._SHARED_RANK = 2, 12
    joins = []
    original = on._shared_join_lean
    def observe(tw):
        out = original(tw)
        if out is not tw:
            joins.append((out.ko, out.R3.shape[0], out.F.shape[1]))
        return out
    on._shared_join_lean = observe
    c, _ = _predict(on, w, 24, 7, es)
    assert np.isfinite(c).all() and len(joins) >= 2
    assert all(ko <= k for ko,k,_ in joins)
    # A tiny budget forces the established fallback. It must remain finite.
    assert np.isfinite(on._projected_flops(24,7,None))


def test_birth_compensation_and_truncation():
    mod = _load(str(ROOT / "estimator_k3_v43_complete_c01.py"))
    rng = np.random.default_rng(314159)
    n, k, r = 7, 4, 3
    D = rng.normal(size=(n,n))  # deliberately nonsymmetric
    U,V = mod._p3_compress(D)
    Dhat = np.asarray(U @ V.T)
    w1,w2,w3 = (rng.normal(size=n) for _ in range(3))
    C = rng.normal(size=(n,n)); C = (C+C.T)/2
    C *= 1-np.eye(n)
    removed = ((w1*w1)[:,None]*Dhat.T*C)*w3[None,:]
    removed += (((w1*w2)[:,None]*Dhat)*C)*w2[None,:]
    # Explicit newborn P3 dense tensor repeated (2,1) content at birth.
    f1 = w1[:,None]*np.asarray(V)
    f2 = w2[:,None]*np.asarray(U)
    repeated = np.zeros((n,n))
    for a in range(n):
        for b in range(n):
            if a != b:
                # I-mode newborn, C-leg = w1 C w3 or w1 C w2.
                repeated[a,b] = (w1[a]**2*Dhat[b,a]*C[a,b]*w3[b]
                                 +w1[a]*w2[a]*Dhat[a,b]*C[a,b]*w2[b])
    assert close(removed,repeated)
    tw = mod._Tower(None, rng.normal(size=(n,k*n)),rng.normal(size=(n,k*n)),
                    rng.normal(size=k*n),k,rng.normal(size=(n,k*r)),
                    rng.normal(size=(k,n,r)),rng.normal(size=k*n),rng.normal(size=k*n))
    tw.F = rng.normal(size=(n,2*k*r))
    tw.R3,tw.R4 = rng.normal(size=(k,n,r)),rng.normal(size=(k,n,r))
    tw.w3b = rng.normal(size=k*n)
    cut = mod._truncate(tw,2,n)
    assert cut.k == 2 and cut.F.shape == (n,4*r)
    assert np.array_equal(cut.F,tw.F[:,-4*r:])
    assert np.array_equal(cut.R3,tw.R3[-2:])
    assert np.array_equal(cut.R4,tw.R4[-2:])
    assert np.array_equal(cut.w3b,tw.w3b[-2*n:])


def test_complete_recurrence():
    """Independent frozen dense-history source versus multiple full-rank joins."""
    ref = _load(str(ROOT / "estimator_k3_v42_pairq_p3_uncompressed.py"))
    got = _load(str(ROOT / "estimator_k3_v43_complete_c01.py"))
    got._SHARED_AGE, got._SHARED_RANK, got._SHARED_FORCE = 2, 16, True
    joins = []
    orig = got._shared_join_lean
    def observe(tw):
        out = orig(tw)
        if out is not tw:
            joins.append(out.ko)
        return out
    got._shared_join_lean = observe
    for seed in (90000000696,90000000697):
        w,_,_ = canonical_mint(16,8,seed)
        a = layer_loop(ref,w,16,8)
        b = layer_loop(got,w,16,8)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        # Relative to each row rather than the huge final row.
        row_rel = np.max(np.abs(a-b),axis=1)/np.maximum(1,np.max(np.abs(a),axis=1))
        assert np.max(row_rel) <= 2e-10, row_rel
    assert len(joins) >= 4 and max(joins) >= 2, joins
    # Budget fallback deliberately excludes carried generations and still runs.
    w,es,_ = canonical_mint(16,5,90000000698)
    budget = 2000000
    assert got._fit_generations(16,5,budget) == -1
    fallback,used = _predict(got,w,16,5,es,budget=budget)
    assert np.isfinite(fallback).all() and used <= budget


if __name__ == "__main__":
    test_tensor()
    print("P3 dense tensor D3/D21 parity: PASS")
    test_join_and_disabled()
    print("disabled identity and multiple gram joins: PASS")
    test_birth_compensation_and_truncation()
    print("birth compensation and truncation: PASS")
    test_complete_recurrence()
    print("full-rank complete recurrence and fallback: PASS")
