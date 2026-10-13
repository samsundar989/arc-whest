"""Task 4: apply the exact (meter-validated) cost models to every _mm call of the v84 ledger."""
import pickle, sys, types
from collections import defaultdict
# cost models only (no flopscope needed): stub the imports of swproto
sys.modules.setdefault("flopscope", types.ModuleType("flopscope")); sys.modules.setdefault("flopscope.numpy", types.ModuleType("flopscope.numpy"))
sys.modules["flopscope"].numpy = sys.modules["flopscope.numpy"]
import swproto as sp
MM, LED = pickle.load(open("census/v84.pkl", "rb"))
U = 2 * 1024 ** 3 - 1024 ** 2
F0 = sum(r[1] for r in LED) + 32768; OPS0 = len(LED) + 1
print("v84 ledger: F=%d (%.2f u, F/B %.5f), ops=%d" % (F0, F0 / U, F0 / 2 ** 41, OPS0))
# validate the reference model against the ledger, call by call
bad = 0
for c in MM:
    m, k = c["A"]; n = c["B"][1]
    f, o = sp.cost_ref(m, k, n, c["depth"])
    bad += (f != c["flops"]) or (o != c["ops"])
print("cost_ref == ledger for %d/%d _mm calls" % (len(MM) - bad, len(MM)))
def v2(x):
    d = 0
    while x % 2 == 0: x //= 2; d += 1
    return d
def eff_depth(m, k, n, d):
    return min(d, v2(m), v2(k), v2(n))
def cost(design, m, k, n, d):
    d = eff_depth(m, k, n, d)
    if design == "ref": return sp.cost_ref(m, k, n, d)
    if design == "out": return sp.cost_out(m, k, n, d)
    if design == "fused": return sp.cost_out(m, k, n, d, True)
    if design == "ks": return sp.cost_ks(m, k, n, d)
    if design == "best":   # cheapest of the three new spellings at this depth
        return min((cost(x, m, k, n, d) for x in ("out", "fused", "ks")), key=lambda t: t[0])
def total(design, depth_rule, floor=0):
    """depth_rule: 'same' or 'opt'. floor: smallest allowed leaf dimension under 'opt'."""
    dF = 0; dO = 0; hist = defaultdict(int); leafF = 0; glueF = 0
    for c in MM:
        m, k = c["A"]; n = c["B"][1]
        if depth_rule == "same":
            f, o = cost(design, m, k, n, c["depth"]); d = eff_depth(m, k, n, c["depth"])
        else:
            cands = []
            for d in range(0, 11):
                if d > min(v2(m), v2(k), v2(n)): break
                if d > c["depth"] and min(m, k, n) >> d < floor: break
                cands.append((cost(design, m, k, n, d), d))
            (f, o), d = min(cands, key=lambda t: t[0][0])
        dF += f - c["flops"]; dO += o - c["ops"]; hist[d] += 1
        lf = 7 ** d * (m >> d) * (n >> d) * (2 * (k >> d) - 1); leafF += lf; glueF += f - lf
    F = F0 + dF
    return F, OPS0 + dO, dict(sorted(hist.items())), leafF, glueF
print("\n%-34s %8s %8s %8s %7s %7s %7s  %s" % ("variant", "F (u)", "F/B", "dF %", "leaf u", "glue u", "ops", "depth histogram (#calls)"))
rows = [("v84 as is (ref, same depths)", "ref", "same", 0)]
for des, nm in (("out", "A: W-out="), ("fused", "B: A + fused leaf"), ("ks", "C: KS/Morton"), ("best", "best of A/B/C per call")):
    rows.append(("(i) %s, same depths" % nm, des, "same", 0))
rows.append(("ref spelling, opt depth leaf>=16", "ref", "opt", 16))
rows.append(("ref spelling, opt depth leaf>=8", "ref", "opt", 8))
for fl in (16, 8, 4, 0):
    for des, nm in (("out", "A"), ("fused", "B"), ("ks", "C"), ("best", "best")):
        rows.append(("(ii) %s, opt depth, leaf>=%d" % (nm, fl), des, "opt", fl))
for nm, des, rule, fl in rows:
    F, ops, hist, lf, gl = total(des, rule, fl)
    print("%-34s %8.2f %8.5f %+8.2f %7.2f %7.2f %7d  %s" % (nm, F / U, F / 2 ** 41, 100 * (F - F0) / F0, lf / U, gl / U, ops, hist))

# ---- (iii) what divisibility costs: allow zero-padding each dim up to a multiple of 2^d (pad copies billed),
#      and (iii') the "re-rounded tables" potential (dims rounded to the nearest multiple of 2^d, no pad cost)
def rup(x, q): return -(-x // q) * q
def rnear(x, q): return max(q, int(round(x / q)) * q)
def total_pad(design, floor, mode):
    dF = 0; dO = 0; hist = defaultdict(int)
    for c in MM:
        m, k = c["A"]; n = c["B"][1]
        cands = []
        for d in range(0, 9):
            q = 1 << d
            if d > c["depth"] and min(m, k, n) >> d < floor: break
            if mode == "pad":
                M, K, N = rup(m, q), rup(k, q), rup(n, q)
                f, o = cost(design, M, K, N, d)
                if (M, K) != (m, k): f += M * K; o += 2
                if (K, N) != (k, n): f += K * N; o += 2
            else:
                M, K, N = rnear(m, q), rnear(k, q), rnear(n, q)
                f, o = cost(design, M, K, N, d)
                # charge/credit the volume change at the call's own rate so only the depth effect is counted
                f = f * (m * k * n) / (M * K * N)
            cands.append((f, o, d))
        f, o, d = min(cands)
        dF += f - c["flops"]; dO += o - c["ops"]; hist[d] += 1
    F = F0 + dF
    return F, OPS0 + dO, dict(sorted(hist.items()))
print()
for mode, label in (("pad", "(iii) zero-pad to 2^d"), ("near", "(iii') tables re-rounded to 2^d")):
    for fl in (16, 8):
        for des, nm in (("ref", "ref"), ("out", "A"), ("fused", "B"), ("ks", "C")):
            F, ops, hist = total_pad(des, fl, mode)
            print("%-44s %8.2f %8.5f %+8.2f %7d  %s" % ("%s %s leaf>=%d" % (label, nm, fl), F / U, F / 2 ** 41, 100 * (F - F0) / F0, ops, hist))

# ---- Task 5: rough wall model (INFERRED).  T = 82 s + g*dGlue + c_op*dOps + a*dLeafFLOPs + c_leaf*dLeafCalls
#  g = 1.1 ns per billed glue element and a = 12 ps per leaf FLOP: two-point fit to the server figures
#  (90 ps/FLOP at leaf 64, 165 ps/FLOP at leaf 32; glue share of a 1024^3 product 7.2 % / 14.1 %);
#  c_op = 0.36 ms; c_leaf = 100 ns per leaf matrix (VM: ~40 ns, server ~2.5x slower than the VM per leaf FLOP).
G_NS, A_PS, C_OP, C_LEAF = 1.1e-9, 12e-12, 0.36e-3, 100e-9
def wall(design, rule, floor=0, mode=None):
    dG = dO = dL = dN = 0
    for c in MM:
        m, k = c["A"]; n = c["B"][1]
        d0 = eff_depth(m, k, n, c["depth"])
        leaf0 = 7 ** d0 * (m >> d0) * (n >> d0) * (2 * (k >> d0) - 1)
        if rule == "same":
            (f, o), d, (M, K, N) = cost(design, m, k, n, c["depth"]), d0, (m, k, n)
        else:
            cands = []
            for d in range(0, 9):
                q = 1 << d
                if d > c["depth"] and min(m, k, n) >> d < floor: break
                if mode == "pad":
                    M, K, N = rup(m, q), rup(k, q), rup(n, q)
                    f, o = cost(design, M, K, N, d)
                    if (M, K) != (m, k): f += M * K; o += 2
                    if (K, N) != (k, n): f += K * N; o += 2
                else:
                    if d > min(v2(m), v2(k), v2(n)): break
                    M, K, N = m, k, n
                    f, o = cost(design, m, k, n, d)
                cands.append((f, o, d, (M, K, N)))
            f, o, d, (M, K, N) = min(cands)
        leaf = 7 ** d * (M >> d) * (N >> d) * (2 * (K >> d) - 1)
        dG += (f - leaf) - (c["flops"] - leaf0); dO += o - c["ops"]; dL += leaf - leaf0; dN += 7 ** d - 7 ** d0
    return 82 + G_NS * dG + C_OP * dO + A_PS * dL + C_LEAF * dN, G_NS * dG, C_OP * dO, A_PS * dL + C_LEAF * dN
print("\n%-40s %7s   %s" % ("variant", "wall s", "(glue passes, op overhead, leaf kernel) deltas in s"))
for nm, des, rule, fl, mode in (
        ("(i) A same depths", "out", "same", 0, None), ("(i) B same depths", "fused", "same", 0, None), ("(i) C same depths", "ks", "same", 0, None),
        ("(ii) A opt depth leaf>=16", "out", "opt", 16, None), ("(ii) B opt depth leaf>=16", "fused", "opt", 16, None), ("(ii) C opt depth leaf>=16", "ks", "opt", 16, None),
        ("(ii) A opt depth leaf>=8", "out", "opt", 8, None), ("(ii) B opt depth leaf>=8", "fused", "opt", 8, None), ("(ii) C opt depth leaf>=8", "ks", "opt", 8, None),
        ("(ii) C opt depth no floor", "ks", "opt", 0, None),
        ("(iii) A zero-pad leaf>=8", "out", "opt", 8, "pad"), ("(iii) C zero-pad leaf>=8", "ks", "opt", 8, "pad"),
        ("ref spelling opt depth leaf>=8", "ref", "opt", 8, None)):
    w, a, b, c = wall(des, rule, fl, mode)
    print("%-40s %7.1f   %+6.1f %+5.1f %+5.1f" % (nm, w, a, b, c))
