import json, sys
from collections import defaultdict
U = 2 * 1024**3 - 1024**2
def cat(func, line, mm):
    if func in ("_p4_sym_sandwich", "_p4_sym_rest"): return "covariance"
    if func == "_layer" and line in (4028, 4030): return "covariance"
    if func == "_advance":
        if line in (1191, 1193, 1189, 1192): return "recent.transport"
        if line in (1199, 1203): return "thin.transport(Z,Y)"
        if line == 1206: return "old.transport"
        return "recent.transport"
    if func == "_shared_contract_lean":
        if line in (763, 764, 761, 762, 755, 756): return "recent.D21_readout"
        return "old.D21_readout"
    if func == "_full_legs_lean": return "old.rebuild_legs"
    if func == "_shared_join_lean": return "old.join"
    if func == "_dslice21":
        if line in (1483, 1484, 1486, 1490): return "thin.D21_readout(Z,Y)"
        return "D21_glue(LA/LP,scatter)"
    if func in ("_p2_recon", "_cat1"): return "P2_recon"
    if func == "_hub_parts": return "hubs"
    if func == "_dslice3": return "D3"
    if func == "_layer" and line in (4109, 4111): return "star"
    if func in ("_closure_apply", "_closure_what", "_own_extra", "<listcomp>"): return "closure"
    if func in ("_corr_delta", "_corr_raw"): return "corrector"
    if func == "_gate_eval": return "gates"
    if func == "_kas_rows": return "kas.gathers"
    if func == "_layer" and line in (4051, 4056, 4057, 4062, 4064, 4065, 4382, 4383, 4384, 4387, 4388, 4389, 4390, 4391, 4392, 4393, 4456): return "kas.gathers"
    if func in ("_p2_compress",): return "birth.P2_compress"
    if func == "_dress_H": return "birth/table(dress)"
    if func in ("_wick_tower", "_append", "_l_append", "_truncate"): return "tower.bookkeeping"
    if func in ("predict", "_constants", "<dictcomp>", "_closure_consts", "_tower", "harness"): return "predict.casts/constants"
    return "layer.table/birth/wick"
if __name__ == "__main__":
    path = sys.argv[1]
    L = json.load(open(path + ".ledger.json"))
    tot = sum(r[1] for r in L) + 32768
    C = defaultdict(lambda: defaultdict(int)); OPS = defaultdict(int)
    for op, fl, layer, key, shp in L:
        mm = key[0] in ("_mm", "_sb")
        c = cat(key[1], key[2], mm)
        C[c][layer] += fl; OPS[c] += 1
    C["predict.casts/constants"][-1] += 32768
    cats = sorted(C, key=lambda c: -sum(C[c].values()))
    layers = list(range(-1, 16))
    print("TOTAL %d = %.2f u" % (tot, tot / U))
    print("%-26s %8s %6s %6s " % ("component", "u", "%", "ops") + " ".join("%5s" % ("L%d" % (l + 1) if l >= 0 else "pre") for l in layers))
    for c in cats:
        s = sum(C[c].values())
        print("%-26s %8.2f %5.2f%% %6d " % (c, s / U, 100 * s / tot, OPS[c]) + " ".join("%5.2f" % (C[c][l] / U) for l in layers))
    print("%-26s %8.2f %6s %6s " % ("layer total", tot / U, "", "") + " ".join("%5.2f" % (sum(C[c][l] for c in cats) / U) for l in layers))
