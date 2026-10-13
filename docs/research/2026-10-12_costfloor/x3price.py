"""x3 scenario pricer. spec -> cfg (x3lib) -> patched estimator -> exact shape-only meter (x3run).

A spec is a dict:  s1=tau_r | None, s2=tau_h | None, s2b=tau_h | None, s3=(tau_o, tau_i) | None,
                   age=int, old=bool, glue='std'|'out'|'min', depth='std'|('opt', leaf), lastk=None|K, olast=bool
Counts come from class-count tables `tabs` = {'keep': {tau: [16]}, 'act': {tau: [16]}} (raw counts), rounded
to the nearest multiple of `g` (default 32, python round = the kink_tables.json convention).
Results are cached in res/<name>.json.
"""
import json, os, subprocess, sys, hashlib
import numpy as np
import x3lib
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "res"); VAR = os.path.join(HERE, "variants")
os.makedirs(RES, exist_ok=True); os.makedirs(VAR, exist_ok=True)
U = 2 * 1024 ** 3 - 1024 ** 2
B = 2 ** 41
F0 = 381613795475
KEEP_VARIANTS = False
TAUS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


def load_stats():
    z = np.load(os.path.join(HERE, "alpha_stats.npz"))
    taus = [float(t) for t in z["taus"]]
    return taus, z["dead"], z["on"]          # [tau, net, layer]


class Tabs(object):
    """Class-count tables. count(a, b) = stat over nets of #( -a <= alpha <= b ) per layer (b None = no upper
    cut, i.e. 'non-dead at a'; a None = no lower cut). stat = 'q50' | 'qNN' | int net index (adaptive)."""
    def __init__(self, dead, on, taus, stat="q50"):
        self.dead, self.on, self.taus, self.stat = dead, on, list(taus), stat
        self.adaptive = isinstance(stat, int)
    def count(self, a, b=None):
        c = np.full(self.dead.shape[1:], 1024.0)
        if a is not None: c = c - self.dead[self.taus.index(float(a))]
        if b is not None: c = c - self.on[self.taus.index(float(b))]
        if isinstance(self.stat, int): return c[self.stat].tolist()
        if self.stat == "q50": return np.median(c, axis=0).tolist()
        return np.percentile(c, float(self.stat[1:]), axis=0).tolist()


def rnd(xs, g=32, mode="near"):
    out = []
    for x in xs:
        if mode == "near": v = int(round(x / g)) * g
        elif mode == "floor": v = int(np.floor(x / g)) * g
        elif mode == "ceil": v = int(np.ceil(x / g)) * g
        else: v = int(round(x)); g = 1
        out.append(max(g, min(1024, v)))
    return out


def _ab(x):
    """tau -> (tau, tau) symmetric kink-active; (a, b) -> asymmetric  -a <= alpha <= b."""
    return (x, x) if not isinstance(x, (tuple, list)) else tuple(x)


def mk(spec, tabs, g=32, mode="near", override=None):
    """spec + count tables -> x3lib cfg."""
    cfg = {}
    C = lambda a, b=None: rnd(tabs.count(a, b), g, mode)
    if spec.get("s1") is not None:
        cfg["Dr"] = C(*_ab(spec["s1"])); cfg["Dc"] = C(2.5); cfg["drabs"] = True
    elif tabs.adaptive:
        cfg["Dr"] = C(2.5); cfg["Dc"] = C(2.5)
    if spec.get("s2") is not None:
        cfg["H"] = C(*_ab(spec["s2"])); cfg["habs"] = True
    elif spec.get("s2b") is not None:
        wA = C(*_ab(spec["s2b"])); wM = C(2.5)
        wA = [min(a, m) for a, m in zip(wA, wM)]
        cfg["H"] = [(a + m) // 2 for a, m in zip(wA, wM)]
        cfg["HC"] = [(m - a) // 2 for a, m in zip(wA, wM)]
        cfg["habs"] = True
    elif tabs.adaptive:
        cfg["H"] = C(2.5)
    if spec.get("s3") is not None:
        to, ti = spec["s3"]
        if to is not None: cfg["O"] = C(to)
        cfg["R"] = C(ti)
        cfg["olast"] = spec.get("olast", True)
    elif tabs.adaptive:
        cfg["R"] = C(2.0)
    for k in ("age", "old", "glue", "depth", "lastk", "rank"):
        if k in spec and spec[k] is not None:
            cfg[k] = spec[k]
    if override:
        cfg.update(override)
    return cfg


def _t(x):
    return "%g" % x if not isinstance(x, (tuple, list)) else "%g_%g" % tuple(x)


def name_of(spec):
    p = []
    if spec.get("s1") is not None: p.append("S1r" + _t(spec["s1"]))
    if spec.get("s2") is not None: p.append("S2h" + _t(spec["s2"]))
    if spec.get("s2b") is not None: p.append("S2bh" + _t(spec["s2b"]))
    if spec.get("s3") is not None:
        p.append("S3o%si%g%s" % ("%g" % spec["s3"][0] if spec["s3"][0] is not None else "all", spec["s3"][1],
                                 "" if spec.get("olast", True) else "nl"))
    if spec.get("age", 6) != 6 or spec.get("old", True) is False:
        p.append("A%d%s" % (spec.get("age", 6), "" if spec.get("old", True) else "noold"))
    if spec.get("rank") is not None: p.append("rk%d" % spec["rank"])
    if spec.get("glue", "std") != "std": p.append("g" + spec["glue"])
    if spec.get("depth", "std") != "std": p.append("od%d" % spec["depth"][1])
    if spec.get("lastk") is not None: p.append("lk%d" % spec["lastk"])
    return "+".join(p) or "base"


def run_cfg(cfg, name, keep_json=True, force=False):
    out = os.path.join(RES, name + ".json")
    if os.path.exists(out) and not force:
        return json.load(open(out))
    # unique file name per variant and `-B`: a reused name + same size + same mtime second would
    # silently load STALE bytecode from __pycache__ (caught on the first S3 batch).
    h = hashlib.md5((name + json.dumps(cfg, sort_keys=True)).encode()).hexdigest()[:12]
    fn = os.path.join(VAR, "est_%s.py" % h)
    open(fn, "w", encoding="utf-8").write(x3lib.build(cfg))
    tmp = out + ".tmp_" + h
    r = subprocess.run([sys.executable, "-B", os.path.join(HERE, "x3run.py"), fn, tmp], capture_output=True, text=True, cwd=HERE)
    if "TOTAL" not in r.stdout:
        raise RuntimeError(name + "\n" + r.stderr[-3000:])
    d = json.load(open(tmp)); os.remove(tmp)
    if not KEEP_VARIANTS:
        os.remove(fn)
    d["cfg"] = cfg; d["name"] = name
    if not keep_json:
        d.pop("mm", None)
    with open(out + ".part", "w") as fh:
        json.dump(d, fh)
    os.replace(out + ".part", out)          # atomic: a killed batch never leaves a truncated cache entry
    return d


_T = None
def q50tabs():
    global _T
    if _T is None:
        taus, dead, on = load_stats()
        _T = Tabs(dead, on, taus, "q50")
    return _T


def price(spec, force=False):
    return run_cfg(mk(spec, q50tabs()), name_of(spec), force=force)


def line(d, ref=F0):
    return "%-44s %8.2f u  F/B %.5f  dF %+6.2f%%  ops %6d" % (d["name"], d["units"], d["fb"], 100 * (d["flops"] / ref - 1), d["ops"])


if __name__ == "__main__":
    T = q50tabs()
    print("check tables vs shipped:")
    print(" R  ", rnd(T.count(2.0)) == x3lib.BASE["R"])
    print(" D/H", rnd(T.count(2.5)) == x3lib.BASE["D"])
    for t in TAUS:
        print(" keep %.1f" % t, rnd(T.count(t)))
        print(" act  %.1f" % t, rnd(T.count(t, t)))
    print(line(price({})))
