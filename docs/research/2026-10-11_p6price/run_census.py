import sys, os, importlib.util, json, shutil
from collections import defaultdict
import fakefs
flops, fnp, wb = fakefs.install()

def load(path, name):
    base = os.path.basename(path)
    fakefs.EST_FILES.add(base)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

MM = []
def wrap_mm(mod):
    orig = mod._mm
    def _mm(A, B):
        before = len(fakefs.LEDGER)
        d = 0 if mod._STRASSEN_DEPTH <= 0 else mod._depth_for(A.shape[0], A.shape[1], B.shape[1])
        out = orig(A, B)
        recs = fakefs.LEDGER[before:]
        # caller site = the frame calling _mm
        f = sys._getframe(1)
        MM.append(dict(caller=f.f_code.co_name, line=f.f_lineno, A=A.shape, B=B.shape, depth=d,
                       flops=sum(r[1] for r in recs), ops=len(recs),
                       leaf=sum(r[1] for r in recs if r[0] == "matmul"),
                       layer=recs[0][2] if recs else -1))
        return out
    mod._mm = _mm

def run(path, name="est", n=1024, depth=16, patch=None):
    fakefs.LEDGER.clear(); MM.clear()
    mod = load(path, name)
    if patch: patch(mod)
    wrap_mm(mod)
    est = mod.Estimator()
    est.setup(None)
    fakefs.LEDGER.clear()
    W = [fakefs.FA((n, n), fakefs.float64) for _ in range(depth)]
    mlp = wb.MLP(n, W)
    out = est.predict(mlp, 2 ** 41)
    tot = sum(r[1] for r in fakefs.LEDGER)
    return mod, tot, len(fakefs.LEDGER)

if __name__ == "__main__":
    path = sys.argv[1]
    mod, tot, ops = run(path)
    print("TOTAL", tot, "ops", ops, "F/B", tot / 2 ** 41)
