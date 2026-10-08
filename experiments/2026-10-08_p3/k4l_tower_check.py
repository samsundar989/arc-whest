"""Check k4l_arm's tower contraction against the chain's own readouts (twin, one net).

    TWIN_DEVICE=cpu .venv-gpu/Scripts/python.exe experiments/2026-10-08_p3/k4l_tower_check.py

At layers 2, 6, 10: adv = _advance(W^T, tower). Checks (a) k4l's D21 formula == _dslice21(adv),
(b) the contraction with e_a reproduces D21's row a (C(e_a)[a, c] = kappa3_aac), (c) for a
lineage subset, (a) and (b) hold for the subset alone (b is checked; a needs no chain twin).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "twin"))
import torch  # noqa: E402
import twin  # noqa: E402

MAIN = Path(r"C:/Users/samsu/Documents/arc-whest")
CAND = ROOT / "experiments" / "2026-09-27_gates" / "estimator_k3_v62_gb.py"
MASK = ("R=2.0,Hd=2.5,Dz=2.5,K=" + str(MAIN / "experiments" / "2026-10-06_mask" / "ktab" / "aggr_q50.json"))
PARAMS = MAIN / "experiments" / "2026-10-07_p2" / "refit_dress_all" / "params_best.npz"


def main():
    tw = twin.Twin(CAND, arm="mask+dress+k4l")
    tw.install(twin.load_params(PARAMS))
    tw.mod._DRESS = twin.parse_dress("all")
    tw.mod._MASK = twin.parse_mask(MASK)
    m = tw.mod
    w, _ = twin.Data().net("full", 900)
    n = tw.width
    state = (m.fnp.zeros(n), m.fnp.eye(n), None, 0.0, m.fnp.zeros(n))
    off = tw.const["off_mask"]
    with torch.no_grad():
        for l in range(11):
            if l in (2, 6, 10):
                adv = m._advance(w[l].T, state[2])
                D_chain = m._dslice21(adv, off)
                v = torch.randn(n, dtype=torch.float32)
                C, D = m._k4l_tower_cd(adv, v)
                ea = (D * off - D_chain).norm() / D_chain.norm()
                rows = [3, 100, 517]
                errs = []
                for a in rows:
                    e = torch.zeros(n)
                    e[a] = 1.0
                    Ca, _ = m._k4l_tower_cd(adv, e)
                    r = (Ca[a] * off[a] - D_chain[a])
                    errs.append(float(r.norm() / D_chain[a].norm()))
                sym = float((C - C.T).norm() / C.norm())
                lm = torch.zeros(adv.k)
                lm[-1] = 1.0
                errs_s = []
                for a in rows:
                    e = torch.zeros(n)
                    e[a] = 1.0
                    Ca, Da = m._k4l_tower_cd(adv, e, lm)
                    errs_s.append(float((Ca[a] * off[a] - Da[a] * off[a]).norm() / (Da[a] * off[a]).norm()))
                print(f"layer {l}: k={adv.k} ko={adv.ko}  D21 formula vs chain {float(ea):.2e}  "
                      f"row checks {[f'{x:.1e}' for x in errs]}  subset rows {[f'{x:.1e}' for x in errs_s]}  "
                      f"C asym {sym:.1e}", flush=True)
            state = tw._step(l, w[l], state)


if __name__ == "__main__":
    main()
