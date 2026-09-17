"""Live-width SAME-STATE closure report: v32 vs public v17 vs BUDGET 4, reciprocal boundaries.

Handoff 2026-09-17 step 1 deliverable. LABEL-FREE: one canonical 1024x16 network
(seed 90_000_016_001, outside every registered block -- `# seed-ledger: no block
named`), no truth, no score. Diagnostic only; decides nothing.

1. Run frozen v32 (`predict`, float32, the live branch) and capture, at chosen
   layers, the O(n^2) boundary its nonlinear step reads:
   mu_pre, C_off, var, D3, D21 (= the tower's slices), dG (transported core), lam.
2. Run public v17 at its live 1024x16 branch (regen on, NO_WK431 off, NO_CORR on)
   from a DIAGNOSTIC COPY whose only change is two extra keys in its own
   `V17_DEBUG` record (mu_pre = the post-linear `mu`, C_off). No arithmetic changes.
3. Feed EACH boundary through each canonical program in float64
   (`ref_step`): T_v32 (27 terms), T_v17 (its executed mode-1 program incl. the
   (3,1) rows at its lam), T_b4 (independent BUDGET 4, no Q31).
4. Report, per layer: boundary differences (v17 vs v32 state); per boundary the
   relative change of mu_out, off-diagonal cov_out, K3, K21, g_out between
   programs; and the largest term families absent from v32, by Frobenius norm.
5. Adapter checks: ref_step(T_v32, B_v32).mu_out vs v32's own output row, and
   ref_step(T_v17, B_v17).mu_out vs v17's own row (float32-level agreement expected).

Run (heavy-ish, workerq): uv run python local_sim/closure_audit/same_state.py --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_LS = os.path.dirname(_HERE)
ROOT = os.path.dirname(_LS)
sys.path[:] = [q for q in sys.path if os.path.abspath(q or ".") not in (_HERE, _LS)] + [ROOT, _LS, _HERE]

import numpy as np  # noqa: E402

SEED = 90_000_016_001          # seed-ledger: no block named (same network as s2_cost_meter)
WIDTH, DEPTH = int(os.environ.get("SS_W", 1024)), int(os.environ.get("SS_D", 16))
LAYERS = [int(x) for x in os.environ.get("SS_LAYERS", "1,3,5,8,11,14").split(",")]
#: test-only override of v17's 1024x16 regeneration gate; recorded in the output
FORCE_SHAPE_GATE = os.environ.get("SS_FORCE_V17_GATE", "0") == "1"


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _mlp(w):
    import harness
    return harness.MLP(weights=[np.asarray(x) for x in w], width=WIDTH, depth=DEPTH, seed=SEED)


def capture_v32(w):
    import flopscope as flops
    import k4feed_oracle as KO
    mod = KO._load(os.path.join(ROOT, "estimator_k3_v32.py"))
    orig_layer, orig_d3, orig_d21 = mod._layer, mod._dslice3, mod._dslice21
    cur = {}
    caps = {}

    def d3(tw):
        v = orig_d3(tw)
        cur["D3"] = np.asarray(v, dtype=np.float64)
        return v

    def d21(tw, off):
        v = orig_d21(tw, off)
        cur["D21"] = np.asarray(v, dtype=np.float64)
        return v

    def layer(W, state, table, dtype, const, init_scale, last_layer, gens, layer_index=0):
        cur.clear()
        out = orig_layer(W, state, table, dtype, const, init_scale, last_layer, gens, layer_index)
        l = layer_index
        if l in LAYERS:
            Wn = np.asarray(W, dtype=np.float64)
            mu, cov, gk4 = (np.asarray(state[0], dtype=np.float64), np.asarray(state[1], dtype=np.float64),
                            np.asarray(state[4], dtype=np.float64))
            mu_pre = Wn.T @ mu
            cov_pre = Wn.T @ cov @ Wn
            var = np.maximum(np.diag(cov_pre), 1e-12)
            li = l - 1
            lam = float(mod._K4_LAM[li]) if 0 <= li < len(mod._K4_LAM) else 0.0
            dG = (Wn.T * Wn.T) @ (gk4 - np.diag(cov) * lam) + var * lam
            n = mu.shape[0]
            caps[l] = {"mu_pre": mu_pre, "cov_pre": cov_pre, "D3": cur.get("D3", np.zeros(n)),
                       "D21": cur.get("D21", np.zeros((n, n))), "dG": dG, "lam": lam,
                       "metric": float(init_scale), "mu_out_live": np.asarray(out[0], dtype=np.float64)}
        return out

    mod._layer, mod._dslice3, mod._dslice21 = layer, d3, d21
    est = mod.Estimator()
    with flops.BudgetContext(flop_budget=2 ** 41, quiet=True):
        est.setup(None) if False else None
        pred = np.asarray(est.predict(_mlp(w), 2 ** 41), dtype=np.float64)
    return caps, pred


def capture_v17(w, tmpdir, source=None):
    """Capture a v17-SHAPED chain's own boundaries.

    `source` selects which v17-shaped file to instrument; it defaults to the
    pinned public v17. The reciprocal program-swap experiment points it at its
    own diagnostic copy so the copy is measured on ITS OWN native boundaries
    rather than on the reference's.
    """
    import flopscope as flops
    import k4feed_oracle as KO
    src_path = source or os.path.join(ROOT, "external", "504aldo", "estimators", "estimator_v17.py")
    src = open(src_path, encoding="utf-8").read()
    old = "DEBUG.append(dict(layer=li, dG=dG, g_prev=g_prev, var_prev=var_prev,\n                                          lam=lam_prev, var=var, D3=D3, D21=D21))"
    new = ("DEBUG.append(dict(layer=li, dG=dG, g_prev=g_prev, var_prev=var_prev,\n                                          lam=lam_prev, var=var, D3=D3, D21=D21,"
           " mu_pre=mu, C_off=C_off))")
    if src.count(old) != 1:
        raise SystemExit("v17 DEBUG anchor not found exactly once")
    src = src.replace(old, new)
    if FORCE_SHAPE_GATE:
        # TEST-ONLY, RECORDED: v17 enables regeneration only at 1024x16. A small-shape
        # smoke must override that gate explicitly or it silently tests the wrong branch.
        g_old = "riders = (n == 1024 and L == len(CORR_BETA))"
        if src.count(g_old) != 1:
            raise SystemExit("v17 shape-gate anchor not found exactly once")
        src = src.replace(g_old, "riders = (L <= len(CORR_BETA))")
    diag = os.path.join(tmpdir, "estimator_v17_debugcopy.py")
    with open(diag, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    os.environ["V17_DEBUG"] = "1"
    for k in ("V17_NO_FEED", "V17_NO_WK431", "V17_NO_REGEN", "V17_NO_CORR", "V17_LAM_SCALE"):
        os.environ.pop(k, None)
    mod = KO._load(diag)
    mod.DEBUG.clear()
    est = mod.Estimator()
    with flops.BudgetContext(flop_budget=2 ** 41, quiet=True):
        pred = np.asarray(est.predict(_mlp(w), 2 ** 41), dtype=np.float64)
    caps = {}
    for rec in mod.DEBUG:
        l = int(rec["layer"])
        if l not in LAYERS:
            continue
        C_off = np.asarray(rec["C_off"], dtype=np.float64)
        var = np.asarray(rec["var"], dtype=np.float64)
        caps[l] = {"mu_pre": np.asarray(rec["mu_pre"], dtype=np.float64),
                   "cov_pre": C_off + np.diag(var), "D3": np.asarray(rec["D3"], dtype=np.float64),
                   "D21": np.asarray(rec["D21"], dtype=np.float64), "dG": np.asarray(rec["dG"], dtype=np.float64),
                   "lam": float(rec["lam"]), "metric": 2.0}
    return caps, pred, sha(diag), sha(src_path)


def relnorm(a, b):
    nb = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b)) / nb if nb > 0 else float(np.linalg.norm(a))


def summarize_programs(bd, tables):
    import ref_step
    outs = {name: ref_step.step(bd, t, want_families=(name != "v32")) for name, t in tables.items()}
    n = bd["mu_pre"].shape[0]
    off = 1.0 - np.eye(n)
    base = outs["v32"]

    def objs(o):
        return {"mu_out": o["mu_out"], "cov_off": o["cov_out"] * off, "K3": o["K"][(3,)],
                "K21": o["K"][(2, 1)], "K4": o["K"][(4,)], "g_out": o["g_out"]}

    b = objs(base)
    report = {"norms_v32": {k: float(np.linalg.norm(v)) for k, v in b.items()}}
    for name in tables:
        if name == "v32":
            continue
        o = objs(outs[name])
        report[f"rel_change_{name}_vs_v32"] = {k: relnorm(o[k], b[k]) for k in b}
        fams = []
        v32_keys = {(tuple(t["output"]), tuple(t["factors"]), tuple(t["derivs"])) for t in tables["v32"]}
        for key, val in outs[name]["families"].items():
            if key in v32_keys:
                continue
            fams.append({"output": list(key[0]), "factors": list(key[1]), "derivs": list(key[2]),
                         "fro": float(np.linalg.norm(val))})
        fams.sort(key=lambda r: -r["fro"])
        report[f"top_families_absent_from_v32_{name}"] = fams[:12]
    return report, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import tempfile
    import canon
    import s6_build_closure as B
    from mint_provenance import canonical_mint

    t0 = time.time()
    w, es, _ = canonical_mint(WIDTH, DEPTH, SEED)
    tables = {"v32": canon.to_records(canon.canon_ours(os.path.join(ROOT, "estimator_k3_v32.py"))[0]),
              "v17": canon.to_records(canon.canon_public(os.path.join(ROOT, "external", "504aldo",
                                                                      "estimators", "estimator_v17.py"))[0]),
              "b4": [dict(output=list(k[0]), factors=list(k[1]), derivs=list(k[2]), coef=str(c))
                     for k, c in B.load_table(os.path.join(_LS, "closure", "closure_budget4.json")).items()
                     if k[0] != (1, 2)]}
    print("capturing v32 ...", flush=True)
    b32, p32 = capture_v32(w)
    print("  done %.0fs" % (time.time() - t0), flush=True)
    print("capturing v17 ...", flush=True)
    b17, p17, diag_sha, v17_sha = capture_v17(w, tempfile.mkdtemp(prefix="same_state_"))
    print("  done %.0fs" % (time.time() - t0), flush=True)

    result = {"artifact": "closure_audit.same_state", "version": 1, "label_free": True,
              "network": {"width": WIDTH, "depth": DEPTH, "seed": SEED, "mint": "canonical_mint"},
              "v17_shape_gate_overridden": FORCE_SHAPE_GATE,
              "sources": {"estimator_k3_v32.py": sha(os.path.join(ROOT, "estimator_k3_v32.py")),
                          "estimator_v17.py": v17_sha, "v17_debug_copy": diag_sha,
                          "closure_budget4.json": sha(os.path.join(_LS, "closure", "closure_budget4.json")),
                          "same_state.py": sha(os.path.abspath(__file__))},
              "final_row": {"v32_mean": float(p32[-1].mean()), "v17_mean": float(p17[-1].mean()),
                            "rel_v17_vs_v32": relnorm(p17[-1], p32[-1])},
              "layers": {}}
    for l in LAYERS:
        if l not in b32 or l not in b17:
            result["layers"][str(l)] = "missing capture"
            continue
        B32, B17 = b32[l], b17[l]
        n = WIDTH
        off = 1.0 - np.eye(n)
        entry = {"boundary_v17_vs_v32": {
            "mu_pre": relnorm(B17["mu_pre"], B32["mu_pre"]),
            "C_off": relnorm(B17["cov_pre"] * off, B32["cov_pre"] * off),
            "var": relnorm(np.diag(B17["cov_pre"]), np.diag(B32["cov_pre"])),
            "D3": relnorm(B17["D3"], B32["D3"]), "D21": relnorm(B17["D21"] * off, B32["D21"] * off),
            "dG": relnorm(B17["dG"], B32["dG"]), "lam_v32": B32["lam"], "lam_v17": B17["lam"]},
            "standardised_scales_v32": {
                "median_abs_skew": float(np.median(np.abs(B32["D3"]) / np.diag(B32["cov_pre"]) ** 1.5)),
                "median_abs_exkurt": float(np.median(np.abs(2.0 * B32["dG"]) / np.diag(B32["cov_pre"]) ** 2)),
                "median_abs_corr": float(np.median(np.abs(B32["cov_pre"] * off)
                                                   / np.sqrt(np.outer(np.diag(B32["cov_pre"]),
                                                                      np.diag(B32["cov_pre"]))))),
            }}
        for tag, bd in (("B_v32", B32), ("B_v17", B17)):
            rep, outs = summarize_programs(bd, tables)
            entry[tag] = rep
            if tag == "B_v32":
                entry["adapter_v32_mu_out_rel"] = relnorm(outs["v32"]["mu_out"], B32["mu_out_live"])
            else:
                entry["adapter_v17_mu_out_rel_vs_row"] = relnorm(outs["v17"]["mu_out"], p17[l])
            del outs
        # which part of the program difference is the STATE and which the PROGRAM:
        entry["decomposition_mu_out"] = {
            "program_on_v32_state": entry["B_v32"]["rel_change_v17_vs_v32"]["mu_out"],
            "program_on_v17_state": entry["B_v17"]["rel_change_v17_vs_v32"]["mu_out"],
        }
        result["layers"][str(l)] = entry
        print("layer %d done %.0fs" % (l, time.time() - t0), flush=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(result, fh, indent=1)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=1)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
