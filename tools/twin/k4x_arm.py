"""Twin arms `fold` and `rad` (HANDOFF 2026-10-10 Lane M5, M5.1 / M5.2) on the v62_gb chain.

Apply AFTER `bank` (`--arm mask+dress+bank+fold+rad`, `--bank star,starall`). Both are inert
while their spec is None (exact identity).

fold (M5.1) -- the masked star-Q31 fold. The star kappa4 born with each lineage (code
  convention: star-D4 = 4 sum_j c4_j A_aj^3 M_aj, scaled by gain/sg) has the D31 readout
      Qs_ab = sum_j c4_j (A_aj^3 M_bj + 3 M_aj A_aj^2 A_bj),
  whose covariance term (1/6) w3_a w1_b Qs_ab = 1/2 w2_a w1_b (r_a / 3) Qs_ab, r = w3/w2 = -alpha/sigma.
  So D21's left operands get
      LP += g om_a (r_a / 3) (A.A.A) diag(c4),     LA += g om_a r_a (M.A.A) diag(c4),
  om_a = 1{|alpha_a| <= tau}, g = gain/fg[l] (init = gain/sg[l]: D4 and D31 start consistent).
  `_FOLD` keys:
    tau   the om threshold (2.0)
    mode  ns  : exact, does not steer the old-tier join (`_qold` stored unfolded; the old part
                of the fold costs one extra thin product in a shipped build)
          st  : zero products; the old-tier join's `_qold` sees the fold (steers)
          rec : zero products, non-steering; recent lineages only (old columns not folded)
    learn 1 -> g = gain/fg (trained); else g = gain/sg x scale
    scale multiplies g (default 1; 0 = an exact no-op check)
    stats 1 -> per-layer contamination diagnostics into `_FOLDS['stats']` (eval only; runs
          `_dslice21` twice)

rad (M5.2) -- the radial Sym(mu_pre (x) kappa3) kappa4 terms of a random overall gain with
  variance v, in v' = 3v (normalised units):
      Q31n += v'_q (3 alpha_a D21n_ab + alpha_b g3_a)          (closure nb["Q31"])
      D22  += 2 v'_22 (mu_a D21_ba + mu_b D21_ab)                (raw, off-diagonal)
      D4   += 4 v'_4  mu_a D3_a                                  (raw)
  with D21_ab = kappa3(a,a,b) (the chain's own), D3 = kappa3(a,a,a), g3 = D3 / sigma^3.
  v'_l from the quenched gain law (deep review §6.2: n Var log|h|^2 = 2 + 5 sum_j [1 - c(rho_j)],
  scaled so v_16 = 0.0045 as in the handoff; v_4 = 0.0025, v_8 = 0.0036). Learned per layer as
  gain/rq, gain/r22, gain/r4. `_RAD` keys: q, d22, d4 (1 = on, default all on), scale.
"""
import math

#: f(l) = 2 + 5 sum_{j<l} [1 - c(rho_j)], l = 0..16 (rho_0 = 0, rho_{j+1} = k1(rho_j))
_LAW = (2.0, 7.0, 11.0, 14.2, 16.8, 19.0, 20.9, 22.5, 23.9, 25.1, 26.2, 27.2, 28.0, 28.8,
        29.5, 30.2, 30.8)


def vprime(depth=16):
    """Physical v' = 3 v per code layer (v_16 = 0.0045)."""
    return [3.0 * 0.0045 * _LAW[min(l, 16)] / _LAW[16] for l in range(depth)]


HELPERS = r'''
_FOLD = None
_FOLDS = {}
_RAD = None
_RAD_V = %r


def _rad_coef(key, li):
    if _TWG is not None and key in _TWG:
        return _TWG[key][li] * _RAD.get('scale', 1.0)
    return _RAD_V[li] * _RAD.get('scale', 1.0)


''' % (vprime(),)

# eval-only diagnostics (torch), part of the helpers
STATS = r'''

def _fold_stats(li, D21u, D21f, w, alpha, sigma, adv=None, fv=None, off=None):
    """M5.1 contamination: the fold's change dD of D21 as seen by each consumer."""
    import torch as _t
    with _t.no_grad():
        dD = (D21f - D21u).double()
        D = D21u.double()
        w1, w2 = w[(1, 1)].double(), w[(2, 1)].double()
        r = (-(alpha / sigma)).double()
        s4 = (sigma.double() ** 4)
        nrm = lambda x: float(x.pow(2).sum().sqrt())
        H = 0.5 * (w2[:, None] * D + (w1[:, None] * D.T) * r[None, :])
        dH = 0.5 * (w2[:, None] * dD + (w1[:, None] * dD.T) * r[None, :])
        e0 = (D * D).sum(1) / s4
        e1 = (D21f.double() ** 2).sum(1) / s4
        cov = 0.5 * w2[:, None] * w1[None, :] * dD          # the intended covariance term
        brt = (w1 * w1)[:, None] * w1[None, :] * dD         # births: p111_21's w1^2 w1 D21
        _FOLDS.setdefault('stats', []).append({
            "l": li, "rel_D21": nrm(dD) / max(nrm(D), 1e-300),
            "rel_H": nrm(dH) / max(nrm(H), 1e-300),
            "rel_d21e": float((e1 - e0).abs().mean() / e0.abs().mean().clamp_min(1e-300)),
            "birth_over_cov": nrm(brt) / max(nrm(cov), 1e-300),
            "rows_on": float((dD.abs().sum(1) > 0).double().mean())})
        if adv is not None and fv is not None:
            # exactness: dD must equal (rv/3) sum_j c4_j (A^3 M^T + 3 M A^2 A^T), off-diagonal
            A, M = _full_legs_lean(adv)
            A, M = A.double(), M.double()
            rv = fv[0].double()
            c4 = _t.cat([x.double() for x in fv[1][-(A.shape[1] // A.shape[0]):]])
            A2c = (A * A) * c4[None, :]
            Q = (A2c * A) @ M.T + 3.0 * ((A2c * M) @ A.T)
            ex = (rv[:, None] / 3.0) * Q * off.double()
            _FOLDS['stats'][-1]["exact_err"] = nrm(dD - ex) / max(nrm(ex), 1e-300)
'''
HELPERS = HELPERS + STATS.lstrip() + "\n\n"

PATCHES = [
    ("def _layer(W, state, table, dtype, const, init_scale, last_layer, gens,\n",
     HELPERS + "def _layer(W, state, table, dtype, const, init_scale, last_layer, gens,\n"),
    # ---- fold: set the per-row factor before D21 is read; D21 itself is formed below ----
    ("        C11 = cov_pre * off_mask\n",
     "        _FOLDS['v'] = None\n"
     "        if (_FOLD is not None and adv is not None and _bkst.get('w3b') and _TWG is not None):\n"
     "            _fg = (_TWG['fg'][layer_index] if _FOLD.get('learn')\n"
     "                   else _TWG['sg'][layer_index] * _FOLD.get('scale', 1.0))\n"
     "            _om = fnp.where(alpha * alpha <= _FOLD.get('tau', 2.0) ** 2, dtype(1.0), dtype(0.0))\n"
     "            _FOLDS['v'] = ((-(alpha / sigma)) * _om * _fg, _bkst['w3b'])\n"
     "        C11 = cov_pre * off_mask\n"),
    ("        D21 = (_dslice21(adv, off_mask) if adv is not None\n"
     "               else fnp.zeros((n, n), dtype=dtype))\n",
     "        _D21u = None\n"
     "        if _FOLD is not None and _FOLD.get('stats') and _FOLDS.get('v') is not None:\n"
     "            _fv = _FOLDS['v']\n"
     "            _FOLDS['v'] = None\n"
     "            _D21u = _dslice21(adv, off_mask)\n"
     "            _FOLDS['v'] = _fv\n"
     "        D21 = (_dslice21(adv, off_mask) if adv is not None\n"
     "               else fnp.zeros((n, n), dtype=dtype))\n"
     "        _FOLDS['v'] = None\n"
     "        if _D21u is not None:\n"
     "            _fold_stats(layer_index, _D21u, D21, w, alpha, sigma, adv, _fv, off_mask)\n"),
    # inside _dslice21 (P2 branch): build the fold's left-operand increments
    ("        acc = _shared_contract_lean(tw, LA, LP, A_L, M)\n",
     "        _FOLDS['d'] = None\n"
     "        if _FOLDS.get('v') is not None:\n"
     "            _rv, _w3l = _FOLDS['v']\n"
     "            _nn = A_L.shape[0]\n"
     "            _w3c = fnp.concatenate(_w3l[-(A_L.shape[1] // _nn):])\n"
     "            _A2c = (A_L * A_L) * _w3c[None, :]\n"
     "            _FOLDS['d'] = ((_A2c * M) * _rv[:, None],\n"
     "                           (_A2c * A_L) * (_rv * (1.0 / 3.0))[:, None],\n"
     "                           _FOLD.get('mode', 'ns'))\n"
     "        acc = _shared_contract_lean(tw, LA, LP, A_L, M)\n"
     "        _FOLDS['d'] = None\n"),
    # inside _shared_contract_lean: after the joiner's weights / query views are taken
    ("    rec = _mm(fnp.concatenate([LA[:, c:], LP[:, c:]], axis=1),\n",
     "    _fd = _FOLDS.get('d')\n"
     "    if _fd is not None:\n"
     "        _dA, _dP, _fm = _fd\n"
     "        if _fm == 'rec' and c:\n"
     "            _dA = fnp.concatenate([_dA[:, :c] * 0.0, _dA[:, c:]], axis=1)\n"
     "            _dP = fnp.concatenate([_dP[:, :c] * 0.0, _dP[:, c:]], axis=1)\n"
     "        _LA0, _LP0 = LA, LP\n"
     "        LA, LP = LA + _dA, LP + _dP\n"
     "    rec = _mm(fnp.concatenate([LA[:, c:], LP[:, c:]], axis=1),\n"),
    ("    Xold = _mm(fnp.concatenate([LA[:, :c], LP[:, :c]], axis=1), FF.T)\n"
     "    tw._qold = Xold\n",
     "    if _fd is not None and _fm == 'ns':\n"
     "        Xold = _mm(fnp.concatenate([_LA0[:, :c], _LP0[:, :c]], axis=1), FF.T)\n"
     "        tw._qold = Xold\n"
     "        Xold = Xold + _mm(fnp.concatenate([_dA[:, :c], _dP[:, :c]], axis=1), FF.T)\n"
     "    else:\n"
     "        Xold = _mm(fnp.concatenate([LA[:, :c], LP[:, :c]], axis=1), FF.T)\n"
     "        tw._qold = Xold\n"),
    # ---- rad: D4 (every layer, the last included) and D22 ----
    ("    if last_layer:\n        S = {(1,): mu_pre, (3,): D3, (4,): D4}\n",
     "    if _RAD is not None and _RAD.get('d4', 1) and layer_index >= 1:\n"
     "        D4 = D4 + (mu_pre * D3) * (4.0 * _rad_coef('r4', layer_index))\n"
     "    if last_layer:\n        S = {(1,): mu_pre, (3,): D3, (4,): D4}\n"),
    ("        S = {(1,): mu_pre, (2,): var, (1, 1): C11, (2, 1): D21, (1, 2): D21.T,\n",
     "        if _RAD is not None and _RAD.get('d22', 1) and layer_index >= 1:\n"
     "            D22 = D22 + ((mu_pre[:, None] * D21.T + mu_pre[None, :] * D21) * off_mask) * (\n"
     "                2.0 * _rad_coef('r22', layer_index))\n"
     "        S = {(1,): mu_pre, (2,): var, (1, 1): C11, (2, 1): D21, (1, 2): D21.T,\n"),
    # ---- rad: Q31 in the closure's normalised units ----
    ('        nb["Q13"] = nb["Q31"].T\n',
     "        if _RAD is not None and _RAD.get('q', 1) and _CUR_LAYER >= 1:\n"
     "            nb['Q31'] = nb['Q31'] + ((alpha[:, None] * D21n) * 3.0 + alpha[None, :] * D3n[:, None]\n"
     "                                     ) * off_mask * _rad_coef('rq', _CUR_LAYER)\n"
     '        nb["Q13"] = nb["Q31"].T\n'),
]

def parse_fold(spec):
    """'tau=2,mode=ns,learn=0,scale=1,stats=0' -> dict; '' -> None."""
    if not spec:
        return None
    out = {"tau": 2.0, "mode": "ns"}
    for part in filter(None, spec.split(",")):
        k, v = part.split("=")
        if k == "mode":
            assert v in ("ns", "st", "rec"), v
            out[k] = v
        elif k in ("tau", "learn", "scale", "stats"):
            out[k] = float(v)
        else:
            raise SystemExit(f"--fold: unknown key {k}")
    return out


def parse_rad(spec):
    """'q=1,d22=1,d4=1,scale=1' -> dict; 'all' = everything on; '' -> None."""
    if not spec:
        return None
    out = {"q": 1.0, "d22": 1.0, "d4": 1.0}
    for part in filter(None, spec.split(",")):
        if part == "all":
            continue
        k, v = part.split("=")
        if k not in ("q", "d22", "d4", "scale"):
            raise SystemExit(f"--rad: unknown key {k}")
        out[k] = float(v)
    return out
