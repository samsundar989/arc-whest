"""Atom k0 (Lane K, HANDOFF 2026-10-12 Lane K, K0): gain-fill the D21 rows KAS already drops.

Not bit-identical (accuracy change). Twin arm: `mask+dress+bank+fold+kgain --kg rows=dead`
(tools/twin/kgain_arm.py); the two compute the same numbers (same KAS D set, same formula, same
d21e rescaling).

What changes, per non-final layer with a KAS D table:
  * after the D3 readout, the in-chain gain scalar
        v = sum_a g3_a alpha_a / (6 sum_a alpha_a^2)  over |alpha_a| <= 2.5,   g3 = D3 / sigma^3,
    times the learned per-layer gain `_K0_VG[l]` (twin leaf gain/kv, init 1);
  * the dropped D21 rows (KAS D, not formed by the readout) get the gain formula on the kept
    columns:  D21_ac = 2 v (2 mu_a C_ac + mu_c sigma_a^2)  (off-diagonal);
  * the corrector delta stays on for those rows (their d21e is now defined): `_kz` is cleared.
Cost: ~10 n^2 elementwise FLOPs per layer, ~5 extra ops per layer (<0.05 % F).

`apply` bakes `_K0_VG` from the params key `gain__kv` (fails if absent); `make('unrefit')` uses 1.0.
Anchors: v84 base text; independent of the S atoms. k1 / k2 build on this atom (apply k0 first).
"""
from tools.atoms import rep


def _patch(src, vg):
    src = rep(src,
              "_KAS_K = {\"R\": ",
              "#: Lane K k0: learned per-layer gain on the in-chain gain scalar v (twin gain/kv).\n"
              f"_K0_VG = {vg!r}\n"
              "_KAS_K = {\"R\": ")
    src = rep(src,
              "    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n",
              "    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n"
              "    _k0v = None\n"
              "    if _kz is not None and layer_index < len(_K0_VG):\n"
              "        # Lane K: the gain mode's scalar from the chain's own D3 on the kink neurons\n"
              "        _k0a = fnp.where(alpha * alpha <= dtype(6.25), alpha, dtype(0.0))\n"
              "        _k0v = (((D3 / (var * sigma)) * _k0a).sum()\n"
              "                / fnp.maximum((_k0a * alpha).sum() * 6.0, dtype(1e-30))) * _K0_VG[layer_index]\n")
    src = rep(src,
              "        D21 = (_dslice21(adv, off_mask) if adv is not None\n"
              "               else fnp.zeros((n, n), dtype=dtype))\n"
              "        _FOLD_V = None\n",
              "        D21 = (_dslice21(adv, off_mask) if adv is not None\n"
              "               else fnp.zeros((n, n), dtype=dtype))\n"
              "        _FOLD_V = None\n"
              "        if _k0v is not None:\n"
              "            # Lane K k0: the KAS-dropped rows get the gain formula on the kept columns,\n"
              "            # D21_ac = 2 v (2 mu_a C_ac + mu_c sigma_a^2), and keep their corrector delta\n"
              "            _k0G = ((mu_pre[:, None] * C11) * 2.0 + mu_pre[None, :] * var[:, None]) * (_k0v * 2.0)\n"
              "            D21 = D21 + ((_k0G * off_mask) * (1.0 - _kz)[:, None]) * _kz[None, :]\n"
              "            _kz = None\n")
    return src


def apply(src, z):
    if "gain__kv" not in z.files:
        raise SystemExit("atom k0: params have no gain__kv (refit with stage ...k), or use k0:unrefit")
    return _patch(src, [float(x) for x in z["gain__kv"]])


def make(arg):
    if arg != "unrefit":
        raise SystemExit(f"atom k0: unknown arm {arg!r} (k0 | k0:unrefit)")
    return lambda src, z: _patch(src, [1.0] * 16)
