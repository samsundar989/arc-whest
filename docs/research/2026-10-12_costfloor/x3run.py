"""Run one (patched) estimator through the shape-only interpreter and write a categorised ledger.

    python3 x3run.py <estimator.py> [out.json]

Prints `TOTAL <flops incl. harness cast> ops <ops incl. cast>`. The JSON holds per-layer x per-component
FLOPs, the `_mm` call list (shapes, depth, flops) and the transport column composition per layer."""
import sys, os, json, difflib
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_census as rc
import fakefs

ORIG = os.path.join(HERE, "estimator_k3_v84_age6.py")
U = 2 * 1024 ** 3 - 1024 ** 2
B = 2 ** 41


def linemap(path):
    a = open(ORIG, encoding="utf-8").read().split("\n")
    b = open(path, encoding="utf-8").read().split("\n")
    if a == b:
        return None
    m = {}
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        for j in range(j1, j2):
            m[j + 1] = (i1 + (j - j1) + 1) if tag == "equal" else (i1 + 1)
    return m


def cat(func, line, key0):
    if func in ("_p4_sym_sandwich", "_p4_sym_rest2", "_p4_sym_rest"): return "cov sandwich"
    if func == "_layer":
        if line in (4109, 4111): return "cov sandwich"
        if 4099 <= line <= 4126: return "layer misc"
        if 4127 <= line <= 4147: return "KAS select/gathers"
        if 4149 <= line <= 4181: return "k4 core"
        if 4182 <= line <= 4193: return "star D4"
        if 4194 <= line <= 4209: return "layer misc"
        if 4210 <= line <= 4262: return "ReLU table"
        if 4263 <= line <= 4272: return "closure"
        if 4273 <= line <= 4320: return "birth"
        if 4321 <= line <= 4340: return "ReLU table"
        if 4341 <= line <= 4363: return "corrector"
        if 4364 <= line <= 4374: return "layer misc"
        if 4375 <= line <= 4387: return "tower bookkeeping"
        if 4388 <= line <= 4500: return "birth"
        return "k4 core" if line > 4500 else "layer misc"
    if func == "_advance":
        return "ADV" if key0 == "_mm" else "transport glue"
    if func == "_shared_contract_lean":
        if key0 == "_mm":
            return {787: "recent D21 readout", 795: "old D21 readout", 797: "old D21 readout"}.get(line, "D21?%d" % line)
        if 776 <= line <= 786: return "fold (elementwise)"
        return "D21 glue"
    if func == "_full_legs_lean": return "old rebuild"
    if func == "_shared_join_lean": return "old join"
    if func == "_dslice21":
        return "thin D21 readout" if key0 == "_mm" else "D21 glue"
    if func in ("_p2_recon", "_cat1"): return "P2 recon"
    if func == "_hub_parts": return "hubs"
    if func == "_dslice3": return "D3 readout"
    if func in ("_closure_apply", "_closure_what", "_own_extra", "<listcomp>"): return "closure"
    if func in ("_corr_delta", "_corr_raw"): return "corrector"
    if func == "_gate_eval": return "gates"
    if func == "_kas_rows": return "KAS select/gathers"
    if func == "_p2_compress": return "birth"
    if func == "_dress_H": return "birth"
    if func in ("_wick_tower", "_append", "_l_append", "_truncate"): return "tower bookkeeping"
    if func in ("_wick", "_horner_batch"): return "wick coefficients"
    return "predict casts/constants"


def main(path, out=None):
    lm = linemap(path)
    ADVC = {}
    def patch(mod):
        orig = mod._advance
        def _advance(WT, tw, *a, **k):
            layer = sys._getframe(1).f_locals.get("layer_index", -1)
            hc = (a[1] if len(a) > 1 else k.get("hc", 0)) or 0
            ADVC[layer] = dict(AL=tw.A_L.shape[1] - hc, M=0 if tw.M is None else tw.M.shape[1],
                               Z=0 if tw.Z is None else tw.Z.shape[1], Y=0 if tw.Y is None else tw.Y.shape[1],
                               Q=0 if tw.Q is None else tw.Q.shape[1], rows=tw.A_L.shape[0], k=tw.k, ko=tw.ko)
            return orig(WT, tw, *a, **k)
        mod._advance = _advance
    mod, tot, ops = rc.run(path, patch=patch)
    L = fakefs.LEDGER
    C = defaultdict(lambda: defaultdict(int)); OPS = defaultdict(int)
    leaf = glue = 0
    for op, fl, layer, key, shp in L:
        line = key[2] if lm is None else lm.get(key[2], key[2])
        c = cat(key[1], line, key[0])
        if key[0] in ("_mm", "_sb"):
            if op == "matmul": leaf += fl
            else: glue += fl
        if c == "ADV":
            d = ADVC[layer]
            t = d["AL"] + d["M"] + d["Z"] + d["Y"] + d["Q"]; pad = (-t) % 32; T = t + pad
            for nm, wd in (("recent transport", d["AL"] + d["M"]), ("thin transport", d["Z"] + d["Y"] + pad),
                           ("old transport (Q)", d["Q"])):
                C[nm][layer] += fl * wd / T
            OPS["recent transport"] += 1
            continue
        C[c][layer] += fl; OPS[c] += 1
    C["predict casts/constants"][-1] += 32768
    tot += 32768; ops += 1
    print("TOTAL", tot, "ops", ops, "F/B %.5f" % (tot / B), "units %.3f" % (tot / U))
    res = dict(flops=tot, ops=ops, fb=tot / B, units=tot / U, leaf=leaf, glue=glue,
               comp={c: {str(l): v for l, v in d.items()} for c, d in C.items()},
               cops=dict(OPS), adv={str(k): v for k, v in ADVC.items()},
               mm=[dict(caller=r["caller"], line=(r["line"] if lm is None else lm.get(r["line"], r["line"])),
                        A=list(r["A"]), B=list(r["B"]), depth=r["depth"], flops=r["flops"], ops=r["ops"],
                        leaf=r["leaf"], layer=r["layer"]) for r in rc.MM])
    if out:
        json.dump(res, open(out, "w"))
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
