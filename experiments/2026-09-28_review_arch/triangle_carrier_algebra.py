"""Tiny CPU identities for a covariance-triangle K111 residual carrier.

No labels, no training, no performance claim. Verifies a candidate representation only.
The source dimension q is small; each source uses r=q(q+1)/2 extra P2 columns. Those
columns carry the packed symmetric H matrices as well as the repeated-slice canceler.
"""
from pathlib import Path
import json
import numpy as np


def pack(H):
    q = H.shape[-1]
    i, j = np.triu_indices(q)
    return H[..., i, j] * np.where(i == j, 1.0, np.sqrt(2.0))


def sym_leg(A, M):
    X = np.einsum("aj,bj,cj->abc", A, M, M)
    return (X + X.transpose(1, 0, 2) + X.transpose(2, 1, 0)) / 3.0


def triangle(H):
    return np.einsum("aij,bjk,cki->abc", H, H, H)


def report():
    rng = np.random.default_rng(20260928)
    n, q, m = 7, 3, 6
    U = rng.normal(size=(n, q))
    f = rng.uniform(.2, .9, n)
    C = U @ U.T
    H = f[:, None, None] * U[:, :, None] * U[:, None, :]
    T = triangle(H)
    explicit = (f[:, None, None] * f[None, :, None] * f[None, None, :]
                * C[:, :, None] * C[None, :, :] * C[:, None, :])
    R, Q = pack(H), pack(H @ H)
    B = Q @ R.T
    t = np.einsum("aii->a", H @ H @ H)
    Z, L, sb = -3 * R, Q, 2 * t
    M = np.eye(n)
    A = M * sb + Z @ L.T
    Td = T + sym_leg(A, M)
    ij = np.arange(n)
    rep = Td[ij[:, None], ij[:, None], ij[None, :]]
    W = rng.normal(size=(n, m))
    phi = rng.uniform(.2, .95, m)
    P = phi[:, None] * W.T
    Hnext = np.einsum("ia,ajk->ijk", P, H)
    Mn, Zn = P @ M, P @ Z
    An = Mn * sb + Zn @ L.T
    represented = triangle(Hnext) + sym_leg(An, Mn)
    exact = np.einsum("abc,ia,jb,kc->ijk", Td, P, P, P)
    Rn, Qn = pack(Hnext), pack(Hnext @ Hnext)
    tn = np.einsum("aii->a", Hnext @ Hnext @ Hnext)
    next_b = Qn @ Rn.T
    ids = {
        "triangle_trace_equals_covariance_product": float(np.max(np.abs(T - explicit))),
        "packed_pair_readout": float(np.max(np.abs(B - T[ij[:, None], ij[:, None], ij[None, :]]))),
        "packed_diagonal_readout": float(np.max(np.abs(t - T[ij, ij, ij]))),
        "birth_repeated_slices_cancel": float(np.max(np.abs(rep))),
        "packed_H_reuses_Z_transport": float(np.max(np.abs(-Zn / 3 - Rn))),
        "transport_and_wick_full_tensor": float(np.max(np.abs(represented - exact))),
        "next_pair_readout": float(np.max(np.abs(next_b - triangle(Hnext)[np.arange(m)[:, None], np.arange(m)[:, None], np.arange(m)[None, :]]))),
        "next_diagonal_readout": float(np.max(np.abs(tn - triangle(Hnext)[np.arange(m), np.arange(m), np.arange(m)]))),
    }
    scale = max(float(np.max(np.abs(exact))), 1.0)
    result = {"n": n, "q": q, "next_width": m, "extra_P2_columns_per_source": q*(q+1)//2,
              "checks_absolute": ids, "transport_scale": scale,
              "next_D3_nonzero_norm": float(np.linalg.norm(np.einsum("iii->i", exact))),
              "pass": all(v < 1e-10 * scale for v in ids.values())}
    p = Path(__file__).with_suffix(".json")
    p.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(report())
