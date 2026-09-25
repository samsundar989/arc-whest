"""Pre-activation cumulant slices from keenanpepper's full1000-N1e9 raw moments.

Card (cache/hf/arc-whestbench-p2-full1000-N1e9/README.md): raw moments, row-vector
convention z_l = a_{l-1} @ W_l, layer index l = 0..15. `pre_M21[i,j] = E[z_i^2 z_j]`
and `pre_M31[i,j] = E[z_i^3 z_j]` carry the power on the FIRST index -- the same
orientation as the chain's `D21[i,j] = kappa(z_i, z_i, z_j)`.

All conversions in float64 (xp = cupy or numpy). Diagonals of the pairwise slices
are meaningless (they duplicate marginals) and are zeroed.

    C_ij       = M11 - mu_i mu_j
    k3_iij     = M21 - 2 mu_i M11 - mu_j m2_i + 2 mu_i^2 mu_j                (card)
    c22_ij     = M22 - 2 mu_j M21_ij - 2 mu_i M21_ji + mu_j^2 m2_i + mu_i^2 m2_j
                 + 4 mu_i mu_j M11 - 3 mu_i^2 mu_j^2
    k4_iijj    = c22 - C_ii C_jj - 2 C_ij^2
    c31_ij     = M31 - mu_j m3_i - 3 mu_i M21_ij + 3 mu_i mu_j m2_i + 3 mu_i^2 M11 - 3 mu_i^3 mu_j
    k4_iiij    = c31 - 3 C_ii C_ij
    k3_iii     = m3 - 3 mu m2 + 2 mu^3
    k4_iiii    = c4 - 3 c2^2,  c4 = m4 - 4 mu m3 + 6 mu^2 m2 - 3 mu^4
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FULL_DIR = ROOT / "cache" / "hf" / "arc-whestbench-p2-full1000-N1e9" / "moments"


def path(global_index: int) -> Path:
    return FULL_DIR / f"mlp_{global_index:05d}.npz"


class NetLabels:
    """Lazy per-layer cumulants for one full-split net. Dense slices are float64 on `xp`."""

    def __init__(self, global_index: int, xp=np):
        self.xp = xp
        self.z = np.load(path(global_index))
        assert int(self.z["global_index"]) == global_index, "label file / index mismatch"
        self.name = str(self.z["mlp_name"])
        self.marg = {k: np.asarray(self.z[k], dtype=np.float64)
                     for k in ("pre_mean", "pre_m2", "pre_m3", "pre_m4",
                               "mean", "post_m2", "post_m3", "post_m4")}
        self.gt_mean = np.asarray(self.z["gt_mean"], dtype=np.float64)
        self._dense = {}

    def dense(self, key: str) -> np.ndarray:
        """One (16, n, n) float32 member, read from the npz ONCE per net."""
        if key not in self._dense:
            self._dense[key] = self.z[key]
        return self._dense[key]

    # ---- marginals (numpy, float64, (16, n)) --------------------------------
    def pre_marginals(self):
        m = self.marg
        mu, m2, m3, m4 = m["pre_mean"], m["pre_m2"], m["pre_m3"], m["pre_m4"]
        c2 = m2 - mu * mu
        k3 = m3 - 3 * mu * m2 + 2 * mu ** 3
        c4 = m4 - 4 * mu * m3 + 6 * mu * mu * m2 - 3 * mu ** 4
        return {"mu": mu, "var": c2, "k3": k3, "k4": c4 - 3 * c2 * c2}

    def post_k4(self):
        """kappa_4 of each post-activation a_l = relu(z_l), (16, n) float64."""
        m = self.marg
        mu, m2, m3, m4 = m["mean"], m["post_m2"], m["post_m3"], m["post_m4"]
        c2 = m2 - mu * mu
        c4 = m4 - 4 * mu * m3 + 6 * mu * mu * m2 - 3 * mu ** 4
        return c4 - 3 * c2 * c2

    def post_mean(self):
        return self.marg["mean"]

    # ---- dense slices at one layer ------------------------------------------
    def layer(self, l: int, want=("C", "k3_iij", "k4_iijj", "k4_iiij")):
        xp = self.xp
        f = lambda k: xp.asarray(self.dense(k)[l], dtype=xp.float64)
        mu = xp.asarray(self.marg["pre_mean"][l])
        m2 = xp.asarray(self.marg["pre_m2"][l])
        m3 = xp.asarray(self.marg["pre_m3"][l])
        mi, mj = mu[:, None], mu[None, :]
        M11 = f("pre_M11")
        C = M11 - mi * mj
        Cd = xp.diagonal(C).copy()
        out = {}
        need21 = any(k in want for k in ("k3_iij", "k4_iijj", "k4_iiij"))
        M21 = f("pre_M21") if need21 else None
        if "C" in want:
            out["C"] = C
        if "k3_iij" in want:
            out["k3_iij"] = M21 - 2 * mi * M11 - mj * m2[:, None] + 2 * mi * mi * mj
        if "k4_iijj" in want:
            M22 = f("pre_M22")
            c22 = (M22 - 2 * mj * M21 - 2 * mi * M21.T + mj * mj * m2[:, None]
                   + mi * mi * m2[None, :] + 4 * mi * mj * M11 - 3 * mi * mi * mj * mj)
            out["k4_iijj"] = c22 - Cd[:, None] * Cd[None, :] - 2 * C * C
            del M22, c22
        if "k4_iiij" in want:
            M31 = f("pre_M31")
            c31 = (M31 - mj * m3[:, None] - 3 * mi * M21 + 3 * mi * mj * m2[:, None]
                   + 3 * mi * mi * M11 - 3 * mi ** 3 * mj)
            out["k4_iiij"] = c31 - 3 * Cd[:, None] * C
            del M31, c31
        n = C.shape[0]
        eye = xp.eye(n, dtype=bool)
        for k in ("k3_iij", "k4_iijj", "k4_iiij"):
            if k in out:
                out[k][eye] = 0.0
        return out
