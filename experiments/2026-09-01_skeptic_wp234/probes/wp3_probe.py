import os, sys, time, json
ROOT = r"C:\Users\samsu\Documents\arc-whest"
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "local_sim"))
import p2_augment as A
import p2_kprop_bridge as BRIDGE
A._ensure_upstream()
from mint_provenance import canonical_mint
import torch
from mlp_kprop.flop_utils import NamedFlopCounter

SEED = 90000000701  # same as test_p2_augment SEEDS[0]? verify separately

def meas(weights, kind):
    with torch.no_grad(), NamedFlopCounter() as ctr:
        A.run_augment(weights, k_max=2, kind=kind, factor=False,
                      dtype=torch.float64, device="cpu")
    accum = sum(int(v) for v in ctr._accum.values())
    raw = sum(int(v) for v in ctr._raw_accum.values())
    tot = int(ctr.mode.get_total_flops())
    return {"accum": accum, "raw": raw, "mode_total": tot,
            "resid": tot - raw, "keys": {k: int(v) for k, v in ctr._accum.items()}}

rows = []
for (n, L) in [(256,4),(256,16),(1024,4),(1024,16),(512,4),(512,16)]:
    w,_,_ = canonical_mint(n, L, SEED)
    t0=time.time()
    b = meas(w, "BASE"); a = meas(w, "AUGMENT")
    rows.append({"n":n,"L":L,
                 "base_accum":b["accum"],"aug_accum":a["accum"],
                 "ratio_accum": a["accum"]/b["accum"],
                 "base_raw":b["raw"],"aug_raw":a["raw"],
                 "ratio_raw": a["raw"]/b["raw"],
                 "base_mode":b["mode_total"],"aug_mode":a["mode_total"],
                 "ratio_mode": a["mode_total"]/b["mode_total"],
                 "base_resid":b["resid"],"aug_resid":a["resid"],
                 "secs": round(time.time()-t0,1)})
    print(json.dumps(rows[-1]))
    sys.stdout.flush()
print("\nBASE keys @256x4:"); 
w,_,_ = canonical_mint(256,4,SEED)
print(json.dumps(meas(w,"BASE")["keys"], indent=1))
print("AUGMENT keys @256x4:")
print(json.dumps(meas(w,"AUGMENT")["keys"], indent=1))
