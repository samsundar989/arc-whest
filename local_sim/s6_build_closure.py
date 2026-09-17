"""Generate the handoff-09-17 slice arms from FROZEN v32: richer closure (R, Q) and protected feed (P).

Contract: `experiments/LOG.md`, 2026-09-17, CONTRACT (Rule 1) for this slice.
Algebra audit: `local_sim/closure_audit/` (canonical programs, float64 reference
step with live-v32 adapter parity). Independent closure derivation:
`local_sim/closure/` (generated tables, oracles).

WHAT CHANGES, AND WHAT DOES NOT
-------------------------------
Every arm is v32 (`estimator_k3_v32.py`, sha256 10869a6c...) plus Python-level
flags that default to OFF. With all flags off the emitted file must be v32 in
output and counted FLOPs, bit for bit (`--parity`).

(R) READOUT ONLY -- `_CLOSURE_SCOPE = "last"`: the declared closure's extra
    UNIVARIATE mean terms are added at the final layer only. The trajectory is
    v32's exactly; this isolates the immediate readout effect.
(Q) RECURRENT -- `_CLOSURE_SCOPE = "all"`: the declared closure's extra terms
    are added to every pK output at every layer (and the final mean). Centering
    (pK -> K), the regenerated core (K4, K22), the source birth (K3, K21), the
    covariance and the hubs all consume the enriched pK through v32's unchanged
    code. The all-distinct (1,1,1) birth program (SIMPLE two-leg block + hubs)
    is NOT enriched -- a declared mixed-order limitation.
(P) PROTECTED FEED -- `_PROTECT_FEED = True`: the part of the newborn K21 driven
    only by D22 (optionally times univariate D3/D4 blocks) is EXACTLY low rank
    before the diagonal kill, e.g. for the SIMPLE table

        (m/24)(dG_i + dG_j) [w(2,2)_i - 2 pK1_i w(2,1)_i] w(2,1)_j      (rank 2).

    It is removed from the residual BEFORE `_p2_compress` and appended to the
    thin leg as exact columns (Z += 3V, L += U), its diagonal taken out of `sb`.
    The compressed residual keeps rank 32 and the same norm-selected sketch; the
    uncompressed tensor is unchanged (full-rank identity, `--parity`).

EXTRA TERMS ARE EVALUATED NORMALISED. With s = sigma, a monomial's blocks are
divided by their sigma powers (C/(s_i s_j), D21/(s_i^2 s_j), D3/s^3, ...) and
multiplied by  w^(k,p) = s^k w(k,p) = s^p (P1(alpha) phi + P2(alpha) Phi),
which stays finite for k up to 8 where s^-(k-p) alone would overflow float32.
A non-finite result is zeroed (no fitted clipping).

Run:  uv run python local_sim/s6_build_closure.py --table <canonical json> --emit
      uv run python local_sim/s6_build_closure.py --table <canonical json> --parity
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import sys
import zlib
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
SOURCE = os.path.join(_ROOT, "estimator_k3_v32.py")
V32_SHA = "10869a6ca7b9b95005db3d63ea0cf63c44f070dd059553f492ad1cf12105657c"

_BIV = ("C", "D21", "D12", "D22", "D3i", "D3j", "D4i", "D4j", "Q31", "Q13")
_LOWRANK = {"D22", "D3i", "D3j", "D4i", "D4j"}


# --------------------------------------------------------------------------- #
# closed forms of  w^(k,p) = sigma^k w(k,p) = sigma^p (P1(a) phi + P2(a) Phi)
# --------------------------------------------------------------------------- #

def _pmul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def _padd(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)]


def _hermite(r):
    """He_r coefficients, low to high."""
    h0, h1 = [1], [0, 1]
    if r == 0:
        return h0
    for k in range(1, r):
        h0, h1 = h1, _padd(_pmul([0, 1], h1), [-k * c for c in h0])
    return h1


def what_poly(k, p):
    """(P1, P2) integer coefficient lists with w^(k,p) = sigma^p (P1 phi + P2 Phi)."""
    if k > p:
        r = k - p - 1
        s = math.factorial(p) * (-1) ** r
        return [s * c for c in _hermite(r)], [0]
    q = p - k
    # E_r := E[z^r 1{z > -a}] = A_r(a) phi + B_r(a) Phi
    A, B = [[0], [1]], [[1], [0]]
    for rr in range(2, q + 1):
        mono = [0] * (rr - 1) + [(-1) ** (rr - 1)]          # (-a)^(rr-1)
        A.append(_padd(mono, [(rr - 1) * c for c in A[rr - 2]]))
        B.append([(rr - 1) * c for c in B[rr - 2]])
    P1, P2 = [0], [0]
    for rr in range(q + 1):
        apow = [0] * (q - rr) + [math.comb(q, rr)]
        P1 = _padd(P1, _pmul(apow, A[rr]))
        P2 = _padd(P2, _pmul(apow, B[rr]))
    f = math.factorial(p) // math.factorial(q)
    return [f * c for c in P1], [f * c for c in P2]


# --------------------------------------------------------------------------- #
# the closure program: what v32 lacks, grouped for the runtime
# --------------------------------------------------------------------------- #

def load_table(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    terms = data["terms"] if isinstance(data, dict) else data
    out = {}
    for t in terms:
        k = (tuple(t["output"]), tuple(sorted(t["factors"])), tuple(t["derivs"]))
        out[k] = out.get(k, Fraction(0)) + Fraction(t["coef"])
    return {k: c for k, c in out.items() if c != 0}


def closure_program(table, v32_terms):
    """Split `table` into v32's own terms (must be a subset, equal coefficients) and extras."""
    missing = [k for k in v32_terms if table.get(k) != v32_terms[k]]
    if missing:
        raise SystemExit("declared closure does not contain v32's program exactly: %r" % missing[:5])
    # v32 never forms a (1,2) output: `_get_pk_block` reads it as (2,1).T
    extra = {k: c for k, c in table.items() if k not in v32_terms and k[0] != (1, 2)}
    for (out, factors, derivs) in extra:
        for f in factors:
            if f not in _BIV and f not in ("D3", "D4"):
                raise SystemExit("unavailable block %r in the declared closure" % f)
    uni, biv = {}, {}
    for (out, factors, derivs), c in sorted(extra.items()):
        if len(out) == 1:
            uni.setdefault((factors, derivs[0]), {})[out[0]] = float(c)
        else:
            biv.setdefault((factors, derivs[0], derivs[1]), {})["%d%d" % out] = float(c)
    has_q = any(f in ("Q31", "Q13") for (fs, ki, kj) in biv for f in fs)
    return {"has_q": has_q,
            "uni": [[list(f), k, cs] for (f, k), cs in uni.items()],
            "biv": [[list(f), ki, kj, cs] for (f, ki, kj), cs in biv.items()]}


def protect_program(table):
    """Column recipes for the exactly low-rank, D22-driven part of K21 = pK21 - 2 pK1_i pK11.

    A recipe is (coef, u_atoms, u_w, v_atoms, v_w): column u = coef * prod(u_atoms) * w^(u_w)
    [* -2 pK1 when from pK11], v = prod(v_atoms) * w^(v_w). Atoms are normalised
    vectors: "is2" = 1/s^2, "dGn" = dG/s^2 (so D22/(s_i^2 s_j^2) = (m/6)(dGn_i is2_j + is2_i dGn_j)),
    "D3n" = D3/s^3, "D4n" = m dG / s^4, and "one".
    """
    cols = []
    for (out, factors, derivs), c in sorted(table.items()):
        if out not in ((2, 1), (1, 1)) or "D22" not in factors or not set(factors) <= _LOWRANK:
            continue
        # expand the product of rank-one-sums into outer products
        expansions = [((), ())]
        for f in factors:
            if f == "D22":
                opts = [(("dGn",), ("is2",)), (("is2",), ("dGn",))]
                scale = None
            elif f in ("D3i", "D4i"):
                opts = [(("D3n" if f == "D3i" else "D4n",), ())]
            else:
                opts = [((), ("D3n" if f == "D3j" else "D4n",))]
            expansions = [(u + ou, v + ov) for (u, v) in expansions for (ou, ov) in opts]
        m6 = factors.count("D22")
        for u, v in expansions:
            cols.append({"coef": float(c), "d22_count": m6, "u": sorted(u), "v": sorted(v),
                         "u_w": [derivs[0], out[0]], "v_w": [derivs[1], out[1]],
                         "times_m2pk1": out == (1, 1)})
    # group by the v recipe so identical right vectors share one column
    groups = {}
    for col in cols:
        key = (tuple(col["v"]), tuple(col["v_w"]))
        groups.setdefault(key, []).append(col)
    return [{"v": list(k[0]), "v_w": list(k[1]),
             "u_terms": [{"coef": g["coef"], "d22_count": g["d22_count"], "u": g["u"],
                          "u_w": g["u_w"], "times_m2pk1": g["times_m2pk1"]} for g in gs]}
            for k, gs in sorted(groups.items())]


# --------------------------------------------------------------------------- #
# emitted runtime code
# --------------------------------------------------------------------------- #

_RUNTIME = r'''
# ==========================================================================
# HANDOFF 09-17: the declared closure's EXTRA terms and the protected feed
# ==========================================================================
# GENERATED by `local_sim/s6_build_closure.py`. The closure table and its
# derivation are recorded there; this file only evaluates it.

def _closure_consts(dtype):
    """MLP-independent program for the extra terms. Built per predict, like `_constants`."""
    prog = json.loads(zlib.decompress(base64.b64decode(_CLOSURE_B64)))
    keys = []
    for fs, k, cs in prog["uni"]:
        for p in cs:
            keys.append((k, int(p)))
    for fs, ki, kj, cs in prog["biv"]:
        for ab in cs:
            keys.append((ki, int(ab[0])))
            keys.append((kj, int(ab[1])))
    for col in prog["protect"]:
        keys.append(tuple(col["v_w"]))
        for ut in col["u_terms"]:
            keys.append(tuple(ut["u_w"]))
    keys = sorted(set(keys))
    polys = prog["what"]
    P1s = [polys["%d,%d" % kp][0] for kp in keys]
    P2s = [polys["%d,%d" % kp][1] for kp in keys]
    L = max(max(len(r) for r in P1s), max(len(r) for r in P2s), 1)
    pad = lambda rows: [[float(c) for c in r] + [0.0] * (L - len(r)) for r in rows]
    return {"prog": prog, "keys": keys, "row": {kp: i for i, kp in enumerate(keys)},
            "C1": fnp.array(pad(P1s), dtype=dtype), "C2": fnp.array(pad(P2s), dtype=dtype),
            "ppow": [kp[1] for kp in keys]}


def _closure_what(cc, sigma, alpha, dtype):
    """(R, n) stack of  w^(k,p) = sigma^p (P1(alpha) phi + P2(alpha) Phi)."""
    Phi = flops.stats.norm.cdf(alpha).astype(dtype)
    phi = flops.stats.norm.pdf(alpha).astype(dtype)
    A = alpha[None, :]
    P1 = _horner_batch(cc["C1"], A)
    P2 = _horner_batch(cc["C2"], A)
    sp = {0: fnp.ones_like(sigma)}
    for j in range(1, max(cc["ppow"]) + 1):
        sp[j] = sigma if j == 1 else sp[j - 1] * sigma
    S = fnp.stack([sp[p] for p in cc["ppow"]])
    return S * (P1 * phi[None, :] + P2 * Phi[None, :])


def _closure_apply(pK, cc, sigma, alpha, blocks, off_mask, last_layer, dtype):
    """Add the declared closure's extra terms to `pK` in place. Returns `(H, nb)`."""
    prog = cc["prog"]
    H = _closure_what(cc, sigma, alpha, dtype)
    row = cc["row"]
    inv = 1.0 / sigma
    inv2 = inv * inv
    D3n = blocks["D3"] * (inv2 * inv)
    D4n = blocks["D4"] * (inv2 * inv2)
    uni_n = {"D3": D3n, "D4": D4n}
    # ---- univariate outputs -------------------------------------------------
    ups = (1,) if last_layer else (1, 2, 3, 4)
    for p in ups:
        sel = [(fs, k, cs[str(p)]) for fs, k, cs in prog["uni"] if str(p) in cs]
        if not sel:
            continue
        mons = []
        for fs, k, c in sel:
            m = uni_n[fs[0]]
            for f in fs[1:]:
                m = m * uni_n[f]
            mons.append(m)
        Mu = fnp.stack(mons)
        Hs = fnp.stack([H[row[(k, p)]] for fs, k, c in sel])
        cvec = fnp.array([c for fs, k, c in sel], dtype=dtype)
        X = fnp.einsum("ti,ti,t->i", Mu, Hs, cvec)
        X = fnp.where(fnp.isfinite(X), X, dtype(0.0))
        pK[(p,)] = pK[(p,)] + X
    nb = None
    if last_layer or not prog["biv"]:
        return H, nb
    # ---- bivariate outputs, normalised blocks -------------------------------
    Cn = blocks["C11"] * (inv[:, None] * inv[None, :])
    D21n = blocks["D21"] * (inv2[:, None] * inv[None, :])
    D22n = blocks["D22"] * (inv2[:, None] * inv2[None, :])
    nb = {"C": Cn, "D21": D21n, "D12": D21n.T, "D22": D22n,
          "D3i": D3n[:, None], "D3j": D3n[None, :], "D4i": D4n[:, None], "D4j": D4n[None, :]}
    if "lam" in blocks and prog.get("has_q"):
        # kappa_iiij of the carried model kappa4 = m Sym(G (x) I), G = diag(dG) + lam C_off:
        # Q31_ij = (m/2) lam C_ij, normalised by s_i^3 s_j (Skeptic follow-up, 2026-09-17)
        nb["Q31"] = Cn * inv2[:, None] * (0.5 * blocks["metric"] * blocks["lam"])
        nb["Q13"] = nb["Q31"].T
    mons = []
    for fs, ki, kj, cs in prog["biv"]:
        m = nb[fs[0]]
        for f in fs[1:]:
            m = m * nb[f]
        mons.append(m)
    Mb = fnp.stack(mons)
    for out, sym in (((1, 1), True), ((2, 1), False), ((2, 2), True)):
        tag = "%d%d" % out
        sel = [(i, ki, kj, cs[tag]) for i, (fs, ki, kj, cs) in enumerate(prog["biv"]) if tag in cs]
        if not sel:
            continue
        Mt = Mb if len(sel) == len(prog["biv"]) else fnp.stack([mons[i] for i, *_ in sel])
        Li = fnp.stack([H[row[(ki, out[0])]] for i, ki, kj, c in sel])
        Rj = fnp.stack([H[row[(kj, out[1])]] * c for i, ki, kj, c in sel])
        X = fnp.einsum("tij,ti,tj->ij", Mt, Li, Rj)
        X = fnp.where(fnp.isfinite(X), X, dtype(0.0))
        if sym:
            X = (X + X.T) * 0.5
        pK[out] = pK[out] + X * off_mask
    return H, nb


def _protect_factors(cc, H, sigma, dG, D3, metric, pk1, dtype):
    """Exact (U, V), (n, q) each, with the D22-driven K21 content = (U V^T) before the diagonal kill."""
    row = cc["row"]
    inv = 1.0 / sigma
    inv2 = inv * inv
    atoms = {"is2": inv2, "dGn": dG * inv2, "D3n": D3 * (inv2 * inv),
             "D4n": (dG * metric) * (inv2 * inv2)}
    Us, Vs = [], []
    for col in cc["prog"]["protect"]:
        v = H[row[tuple(col["v_w"])]]
        for a in col["v"]:
            v = v * atoms[a]
        u = None
        for ut in col["u_terms"]:
            t = H[row[tuple(ut["u_w"])]] * (ut["coef"] * (metric / 6.0) ** ut["d22_count"])
            for a in ut["u"]:
                t = t * atoms[a]
            if ut["times_m2pk1"]:
                t = t * (pk1 * (-2.0))
            u = t if u is None else u + t
        Us.append(u)
        Vs.append(v)
    return fnp.stack(Us, axis=1), fnp.stack(Vs, axis=1)
'''


def _patches(scope, protect):
    P = []

    def add(old, new, why, count=1):
        P.append((old, new, why, count))

    # flags + table, next to the other module constants
    add('''_HUB_COMPENSATE = True
''', '''_HUB_COMPENSATE = True

#: HANDOFF 09-17. `None` (v32), `"last"` (R: final mean only) or `"all"` (Q).
_CLOSURE_SCOPE = %r
#: Protect the exactly low-rank D22-driven K21 content from the rank-r compressor (P).
_PROTECT_FEED = %r
''' % (scope, protect), "flags")

    # constants: the closure program rides with the Wick program
    add('''        return {"off_mask": (fnp.ones((width, width), dtype=dtype)
                             - fnp.eye(width, dtype=dtype)),
                "ones_n": fnp.ones(width, dtype=dtype),
                "wick": prog}''',
        '''        return {"off_mask": (fnp.ones((width, width), dtype=dtype)
                             - fnp.eye(width, dtype=dtype)),
                "ones_n": fnp.ones(width, dtype=dtype),
                "wick": prog,
                "closure": (_closure_consts(dtype)
                            if (_CLOSURE_SCOPE or _PROTECT_FEED) else None)}''', "constants")

    # the extra terms, after v32's own pK and its symmetrisation
    add('''    for ip in list(pK):
        if len(ip) == 2 and ip[0] == ip[1]:
            pK[ip] = (pK[ip] + pK[ip].T) * 0.5
''', '''    for ip in list(pK):
        if len(ip) == 2 and ip[0] == ip[1]:
            pK[ip] = (pK[ip] + pK[ip].T) * 0.5

    _cl_H = None
    if _CLOSURE_SCOPE == "all" or (_CLOSURE_SCOPE == "last" and last_layer):
        _cl_blocks = {"D3": S[(3,)], "D4": S[(4,)]}
        if not last_layer:
            _cl_blocks.update({"C11": S[(1, 1)], "D21": S[(2, 1)], "D22": S[(2, 2)],
                               "lam": float(_lam_prev), "metric": metric})
        _cl_H, _cl_nb = _closure_apply(pK, const["closure"], sigma, alpha, _cl_blocks,
                                       off_mask, last_layer, dtype)
''', "extras")

    # protected feed, inside the hub birth branch, just before compression
    add('''            Zb, Lb = _p2_compress(resid)
            out_tw = _append(adv, None, fac1, w2b_vec, n, Zb, Lb, S3, e_b,
                             fnp.stack([yA, yB], axis=1), cA, cB)''',
        '''            if _PROTECT_FEED:
                # HANDOFF 09-17 (P): the D22-driven K21 content is exact rank q
                # before the diagonal kill. Take it out of the residual, carry it
                # as exact thin columns, and remove its diagonal through `sb`.
                _cc = const["closure"]
                if _cl_H is None:
                    _cl_H = _closure_what(_cc, sigma, alpha, dtype)
                _U, _V = _protect_factors(_cc, _cl_H, sigma, dG, D3, metric, pK[(1,)], dtype)
                resid = resid - (_U @ _V.T) * off_mask
                Zb, Lb = _p2_compress(resid)
                Zb = fnp.concatenate([Zb, _V * 3.0], axis=1)
                Lb = fnp.concatenate([Lb, _U], axis=1)
                _sb = S3 - (_U * _V).sum(axis=1) * 3.0
            else:
                Zb, Lb = _p2_compress(resid)
                _sb = S3
            out_tw = _append(adv, None, fac1, w2b_vec, n, Zb, Lb, _sb, e_b,
                             fnp.stack([yA, yB], axis=1), cA, cB)''', "protect")

    # budget guard: over-charge the extra work
    add('''        if _HUB_ROUTES and l < depth - 1:
            thin += 12 * n * n                     # birth-time hub construction
    return base + thin''',
        '''        if _HUB_ROUTES and l < depth - 1:
            thin += 12 * n * n                     # birth-time hub construction
        if _CLOSURE_SCOPE == "all" and l < depth - 1:
            thin += (_CLOSURE_NBIV * 40 + 200) * n * n   # extra-term stacks + 3 einsums
        if _PROTECT_FEED:
            q = _PROTECT_Q
            thin += k * n * n * (2 * q) + n * (k * q) * (2 * n - 1) + k * n * q * (2 * n - 1)
            thin += n * n * (2 * k * q) + (4 * q + 8) * n * n
    return base + thin''', "cost-model")
    return P


def build(src, scope, protect, rank, prog_json, title):
    body = src
    for old, new, why, count in _patches(scope, protect):
        found = body.count(old)
        if found != count:
            raise SystemExit("PATCH ANCHOR COUNT WRONG (%s): expected %d, found %d.\n%s"
                             % (why, count, found, old[:300]))
        body = body.replace(old, new, count)
    anchor = "_P2_RANK = 32\n"
    if body.count(anchor) != 1:
        raise SystemExit("rank anchor not unique")
    body = body.replace(anchor, "_P2_RANK = %d\n" % rank, 1)
    b64 = base64.b64encode(zlib.compress(json.dumps(prog_json, sort_keys=True).encode(), 9)).decode()
    chunks = "\n".join('    "%s"' % b64[i:i + 88] for i in range(0, len(b64), 88))
    consts = ("\n#: zlib+base64 of the closure program (extras, protect recipes, w^ polynomials).\n"
              "#: Source table sha256 %s.\n_CLOSURE_B64 = (\n%s\n)\n"
              "_CLOSURE_NBIV = %d\n_PROTECT_Q = %d\n"
              % (prog_json["table_sha256"], chunks, len(prog_json["biv"]), len(prog_json["protect"])))
    fa = "_TABLES = None\n"
    if body.count(fa) != 1:
        raise SystemExit("cannot place closure constants")
    body = body.replace(fa, fa + consts, 1)
    fb = "def _proj_coef_d4_r2(n):"
    if body.count(fb) != 1:
        raise SystemExit("cannot place runtime")
    body = body.replace(fb, _RUNTIME.lstrip("\n") + "\n\n" + fb, 1)
    i = body.find('"""')
    j = body.find('"""', i + 3)
    doc = body[i + 3:j]
    head = ('"""%s\n\nGENERATED by `local_sim/s6_build_closure.py` from `estimator_k3_v32.py`\n'
            '(sha256 %s). DO NOT EDIT; edit the generator and re-emit.\n'
            'Contract: experiments/LOG.md 2026-09-17. Not approved for submission.\n\n'
            '--------------------------------------------------------------------------\n'
            'INHERITED FROM v32\n\n%s\n"""' % (title, V32_SHA, doc.strip()))
    return head + body[j + 3:]


def program_json(table_path, scope):
    """The program one arm embeds. PROTECTED CONTENT COMES FROM THE CLOSURE THE ARM
    EXECUTES: with `scope="all"` that is the declared table, otherwise v32's own
    program -- protecting a term the arm never adds would subtract content that is
    not there."""
    sys.path.append(os.path.join(_HERE, "closure_audit"))
    import canon
    v32_terms, _ = canon.canon_ours(SOURCE)
    table = load_table(table_path)
    prog = closure_program(table, v32_terms)
    prog["protect"] = protect_program(table if scope == "all" else v32_terms)
    prog["protect_source"] = "declared table" if scope == "all" else "v32 program"
    keys = set()
    for fs, k, cs in prog["uni"]:
        keys |= {(k, int(p)) for p in cs}
    for fs, ki, kj, cs in prog["biv"]:
        keys |= {(ki, int(ab[0])) for ab in cs} | {(kj, int(ab[1])) for ab in cs}
    for col in prog["protect"]:
        keys.add(tuple(col["v_w"]))
        keys |= {tuple(ut["u_w"]) for ut in col["u_terms"]}
    prog["what"] = {"%d,%d" % kp: what_poly(*kp) for kp in sorted(keys)}
    with open(table_path, "rb") as fh:
        prog["table_sha256"] = hashlib.sha256(fh.read()).hexdigest()
    prog["table_path"] = os.path.relpath(table_path, _ROOT).replace("\\", "/")
    return prog


ARMS = (
    # name, scope, protect, rank, title
    ("estimator_k3_v35_off.py", None, False, 32,
     "v35_off PARITY: every handoff-09-17 flag off; must equal v32 bit for bit."),
    ("estimator_k3_v35_R.py", "last", False, 32,
     "v35_R DIAGNOSTIC: v32 + the declared closure's extra MEAN terms at the final layer only."),
    ("estimator_k3_v35_Q.py", "all", False, 32,
     "v35_Q: v32 + the declared closure at every layer (recurrent)."),
    ("estimator_k3_v35_P.py", None, True, 32,
     "v35_P: v32 + the D22-driven K21 content protected from the rank-32 compressor."),
    ("estimator_k3_v35_QP.py", "all", True, 32,
     "v35_QP: the recurrent closure and the protected feed together."),
    ("estimator_k3_v35_P_dense.py", None, True, 0,
     "v35_P_dense PARITY: protected feed at full residual rank (must equal v32_dense)."),
    ("estimator_k3_v35_QP_dense.py", "all", True, 0,
     "v35_QP_dense PARITY: QP at full residual rank (must equal v35_Q_dense)."),
    ("estimator_k3_v35_Q_dense.py", "all", False, 0,
     "v35_Q_dense PARITY: Q at full residual rank."),
)


def emit(table_path, out_dir=_ROOT, arms=ARMS):
    with open(SOURCE, "rb") as fh:
        raw = fh.read()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != V32_SHA:
        raise SystemExit("estimator_k3_v32.py is not the frozen v32 (%s)" % sha)
    src = raw.decode("utf-8")
    paths, progs = [], {}
    for name, scope, protect, rank, title in arms:
        prog = program_json(table_path, scope)
        progs[name] = prog
        print("  %s: %d univariate + %d bivariate extra monomials, %d protected columns (%s)"
              % (name, len(prog["uni"]), len(prog["biv"]), len(prog["protect"]), prog["protect_source"]))
        text = build(src, scope, protect, rank, prog, title)
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        paths.append(path)
        print("  wrote %-30s scope=%-5s protect=%-5s rank=%-2d sha256 %s"
              % (name, scope, protect, rank, hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]))
    return paths, progs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--out-dir", default=_ROOT)
    a = ap.parse_args()
    if a.emit:
        emit(a.table, a.out_dir)
