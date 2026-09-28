"""Gate checks through the twin (torch shim). Run with .venv-gpu.

    TWIN_DEVICE=cpu .venv-gpu/Scripts/python.exe experiments/2026-09-27_gates/gate_check.py small
        tiny float64 world (n=64, depth 6, random He nets): repeated-slice preservation of the
        outgoing tower with gates != 1, at FULL-rank P2 residual and at the shipped rank 4.
    .venv-gpu/Scripts/python.exe experiments/2026-09-27_gates/gate_check.py gpu   (workerq)
        1024 x 16 on FIT full rows 0, 1: identity parity (raw v68 files and hooked v62 files +
        S3 params), gradient check, and the rank-4 slice leak on real state.

Writes JSON next to this file (absolute paths: worker-q runs a snapshot).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("C:/Users/samsu/Documents/arc-whest")
OUT = ROOT / "experiments" / "2026-09-27_gates"
sys.path.append(str(ROOT / "tools" / "twin"))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import twin as T  # noqa: E402  (puts the torch shim first -- this process only)

fnp = T.fnp
S3 = ROOT / "experiments" / "2026-09-25_twin" / "S3" / "params_best.npz"
SCALE = {"birth": 0.5, "trans": 0.25}


class Tap:
    """`_A2` spy: keeps the layer's K dict, changes nothing."""

    def __init__(self):
        self.K = None
        self.g = None

    def __call__(self, l, name, value, *aux):
        if name == "K":
            self.K = value
        return value


def load(path, raw, tag):
    mod = T.load_module(Path(path), name=f"_gc_{tag}", raw=raw)
    tap = Tap()
    mod._A2 = tap
    orig = mod._gate_eval

    def spy(*a):
        g = orig(*a)
        tap.g = g
        return g

    mod._gate_eval = spy
    return mod, tap


def slices(mod, tw, n, dtype, off):
    """(3,) and (2,1) slices of the tower AS OUTPUT by a layer (youngest M = I)."""
    advI = mod._advance(fnp.eye(n, dtype=dtype), tw)
    return mod._dslice3(advI), mod._dslice21(advI, off)


def rel(a, b):
    return float(torch.linalg.norm((a - b).double()) / max(float(torch.linalg.norm(b.double())), 1e-300))


def perturb_gate(g, amp, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    g = dict(g)
    for k in ("W2", "b2"):
        g[k] = (torch.randn(g[k].shape, generator=gen, dtype=torch.float64) * amp).to(
            g[k].device, g[k].dtype)
    return g


def pad_tower(mod, tw, r_new):
    """Zero-pad every lineage's P2 residual leg from rank r to r_new (exact)."""
    if tw is None or tw.Z is None:
        return tw
    n, k, r = tw.Z.shape[0], tw.k, tw.L.shape[2]
    if r == r_new:
        return tw
    Z3 = torch.zeros((n, k, r_new), dtype=tw.Z.dtype, device=tw.Z.device)
    Z3[:, :, :r] = tw.Z.reshape(n, k, r)
    L = torch.zeros((k, n, r_new), dtype=tw.L.dtype, device=tw.L.device)
    L[:, :, :r] = tw.L
    out = mod._Tower(tw.A_D, tw.A_L, tw.M, tw.w2b, tw.k, Z3.reshape(n, k * r_new), L,
                     tw.sb, tw.eb)
    out.Y, out.CA, out.CB = tw.Y, tw.CA, tw.CB
    mod._carry_shared(out, tw, tw.Q)
    return out


def layer(mod, tap, W, state, table, dtype, const, l, depth, rank):
    old = mod._P2_RANK
    mod._P2_RANK = rank
    try:
        s = mod._layer(W, state, table, dtype, const, 2.0, last_layer=(l == depth - 1),
                       gens=None, layer_index=l, depth=depth)
    finally:
        mod._P2_RANK = old
    return s, tap.K


# ------------------------------------------------------------------ small (CPU, f64)

def cmd_small():
    n, depth = 64, 6
    dtype = fnp.float64
    rep = {"n": n, "depth": depth, "dtype": "float64", "arms": {}}
    for arm in ("birth", "trans"):
        mod, tap = load(OUT / f"estimator_k3_v70g_{arm}.py", raw=True, tag=f"s_{arm}")
        mod._SHARED_ON = False                       # depth 6 < age 5 + 1 anyway
        est = mod.Estimator()
        est.setup(T._Ctx(n, depth, T.BUDGET, 0))
        table = mod._tables()
        const = est._constants(n, dtype)
        off = const["off_mask"]
        const_id = dict(const, gate=None)
        const_g = dict(const, gate=perturb_gate(const["gate"], 1.0, 11))
        gen = torch.Generator(device="cpu").manual_seed(5)
        Ws = [(torch.randn((n, n), generator=gen, dtype=torch.float64) * (2.0 / n) ** 0.5)
              .to(T.DEV) for _ in range(depth)]
        rows = []
        for rank in (0, 4):
            # the chain is run GATED at this rank; at each layer the identity layer is run
            # from the same incoming state for the comparison
            st = (fnp.zeros(n, dtype=dtype), fnp.eye(n, dtype=dtype), None, 0.0,
                  fnp.zeros(n, dtype=dtype))
            for l in range(depth - 1):
                sg, Kg = layer(mod, tap, Ws[l], st, table, dtype, const_g, l, depth, rank)
                g = tap.g
                d3g, d21g = slices(mod, sg[2], n, dtype, off)
                si, Ki = layer(mod, tap, Ws[l], st, table, dtype, const_id, l, depth, rank)
                d3i, d21i = slices(mod, si[2], n, dtype, off)
                # the next layer's consumed D3 from each outgoing tower (what the gate moves)
                nxt = Ws[l + 1].T
                D3n_g = mod._dslice3(mod._advance(nxt, sg[2]))
                D3n_i = mod._dslice3(mod._advance(nxt, si[2]))
                rows.append({
                    "rank": "full" if rank == 0 else rank, "layer": l,
                    "gate_dev_max": float((g - 1).abs().max()),
                    "gated_slice3_rel": rel(d3g, Kg[(3,)]),
                    "gated_slice21_rel": rel(d21g, Kg[(2, 1)] * off),
                    "ident_slice3_rel": rel(d3i, Ki[(3,)]),
                    "ident_slice21_rel": rel(d21i, Ki[(2, 1)] * off),
                    "K_same": rel(Kg[(2, 1)], Ki[(2, 1)]),
                    "next_D3_gate_effect_rel": rel(D3n_g, D3n_i),
                })
                print(json.dumps({"arm": arm, **rows[-1]}), flush=True)
                st = (sg[0], sg[1], sg[2], sg[3], sg[4])
        rep["arms"][arm] = rows
    (OUT / "check_small.json").write_text(json.dumps(rep, indent=1))
    print("wrote", OUT / "check_small.json")


# ------------------------------------------------------------------ gpu (1024 x 16)

def forward_all(tw, w):
    with torch.no_grad():
        return tw.forward(w)


def cmd_gpu():
    rep = {}
    data = T.Data()
    nets = [0, 1]
    ws = {i: data.net("full", i) for i in nets}
    t0 = time.time()

    # --- (1) identity parity, raw (shipped v68 arithmetic)
    par = {}
    preds = {}
    for tag, f in (("v68", ROOT / "estimator_k3_v68_wino.py"),
                   ("v70g_birth", OUT / "estimator_k3_v70g_birth.py"),
                   ("v70g_trans", OUT / "estimator_k3_v70g_trans.py")):
        tw = T.Twin(f, raw=True)
        preds[tag] = {}
        for i in nets:
            w, truth = ws[i]
            mus = forward_all(tw, w)
            preds[tag][i] = torch.stack(mus).cpu()
            par.setdefault(tag, {})[i] = float(T.mse_rows(mus, truth)[-1])
        del tw
        torch.cuda.empty_cache()
    for tag in ("v70g_birth", "v70g_trans"):
        for i in nets:
            par[f"{tag}_vs_v68_net{i}"] = {
                "final_mse_rel": par[tag][i] / par["v68"][i] - 1.0,
                "max_abs_pred_diff": float((preds[tag][i] - preds["v68"][i]).abs().max()),
                "bit_equal": bool(torch.equal(preds[tag][i], preds["v68"][i]))}
    rep["parity_raw"] = par
    print(json.dumps({"parity_raw": par}), flush=True)

    # --- (1b) identity parity, hooked (the TRAINING path: v62 base + S3 params)
    S3p = T.load_params(S3)
    parh = {}
    predh = {}
    for tag, f in (("v62_s1", ROOT / "estimator_k3_v62_s1.py"),
                   ("v62_gb", OUT / "estimator_k3_v62_gb.py"),
                   ("v62_gt", OUT / "estimator_k3_v62_gt.py")):
        tw = T.Twin(f)
        tw.install(S3p)
        predh[tag] = {}
        for i in nets:
            w, truth = ws[i]
            mus = forward_all(tw, w)
            predh[tag][i] = torch.stack(mus).cpu()
            parh.setdefault(tag, {})[i] = float(T.mse_rows(mus, truth)[-1])
        del tw
        torch.cuda.empty_cache()
    for tag in ("v62_gb", "v62_gt"):
        for i in nets:
            parh[f"{tag}_vs_v62_s1_net{i}"] = {
                "final_mse_rel": parh[tag][i] / parh["v62_s1"][i] - 1.0,
                "max_abs_pred_diff": float((predh[tag][i] - predh["v62_s1"][i]).abs().max()),
                "bit_equal": bool(torch.equal(predh[tag][i], predh["v62_s1"][i]))}
    for i in nets:
        parh[f"v62_s1+S3_vs_v68_raw_net{i}"] = parh["v62_s1"][i] / par["v68"][i] - 1.0
    rep["parity_hooked"] = parh
    print(json.dumps({"parity_hooked": parh}), flush=True)

    # --- (2) gradient check through the twin (training path), net 0
    grads = {}
    for arm, f in (("birth", OUT / "estimator_k3_v62_gb.py"),
                   ("trans", OUT / "estimator_k3_v62_gt.py")):
        tw = T.Twin(f)
        for phase in ("identity", "W2_perturbed"):
            params = dict(S3p)
            for k in T._GATE_LEAVES:
                v = tw.leaf(k).clone()
                if phase == "W2_perturbed" and k.endswith("W2"):
                    gen = torch.Generator(device="cpu").manual_seed(3)
                    v = (torch.randn(v.shape, generator=gen) * 0.05).to(v.device)
                params[k] = v.requires_grad_(True)
            tw.install(params)
            w, truth = ws[0]
            mus = tw.forward(w, ckpt=True)
            rows = T.mse_rows(mus, truth)
            loss = rows[-1] + 0.1 * torch.stack(rows[:-1]).mean()
            (loss * 1e8).backward()
            grads[f"{arm}/{phase}"] = {
                "loss_final_mse": float(rows[-1]),
                **{k: float(params[k].grad.norm()) if params[k].grad is not None else None
                   for k in T._GATE_LEAVES},
                "finite": all(bool(torch.isfinite(params[k].grad).all())
                              for k in T._GATE_LEAVES if params[k].grad is not None)}
            print(json.dumps({f"grad {arm}/{phase}": grads[f"{arm}/{phase}"]}), flush=True)
            del mus, rows, loss
        del tw
        torch.cuda.empty_cache()
    rep["grad"] = grads
    rep["peak_vram_gib_after_grad"] = torch.cuda.max_memory_allocated() / 2 ** 30

    # --- (3) rank-4 slice leak on real state (raw v70g files = shipped arithmetic), net 0
    leak = {}
    amp = {"birth": 0.1, "trans": 0.2}
    for arm in ("birth", "trans"):
        mod, tap = load(OUT / f"estimator_k3_v70g_{arm}.py", raw=True, tag=f"g_{arm}")
        est = mod.Estimator()
        n, depth = 1024, 16
        est.setup(T._Ctx(n, depth, T.BUDGET, 0))
        table = mod._tables()
        dtype = fnp.float32
        const = est._constants(n, dtype)
        off = const["off_mask"]
        cid = dict(const, gate=None)
        cg = dict(const, gate=perturb_gate(const["gate"], amp[arm], 11))
        w, _ = ws[0]
        st = (fnp.zeros(n), fnp.eye(n), None, 0.0, fnp.zeros(n))
        rows = []
        with torch.no_grad():
            for l in range(depth - 1):
                out = {}
                for rk, rank in (("r4", 4), ("full", 0)):
                    s_in = st if rank == 4 else (st[0], st[1], pad_tower(mod, st[2], n),
                                                 st[3], st[4])
                    for gk, cc in (("id", cid), ("g", cg)):
                        s, K = layer(mod, tap, w[l], s_in, table, dtype, cc, l, depth, rank)
                        if gk == "g":
                            gv = tap.g
                        d3, d21 = slices(mod, s[2], n, dtype, off)
                        advn = mod._advance(w[l + 1].T, s[2])
                        out[(rk, gk)] = {"e3": rel(d3, K[(3,)]), "e21": rel(d21, K[(2, 1)] * off),
                                         "D3n": mod._dslice3(advn),
                                         "D21n": mod._dslice21(advn, off),
                                         "state": s if (rk, gk) == ("r4", "id") else None}
                        del advn
                ref = out[("full", "id")]
                row = {"layer": l, "gate_dev_rms": float(((gv - 1) ** 2).mean().sqrt()),
                       "gate_dev_max": float((gv - 1).abs().max())}
                for key, v in out.items():
                    row[f"{key[0]}_{key[1]}_slice3_rel"] = v["e3"]
                    row[f"{key[0]}_{key[1]}_slice21_rel"] = v["e21"]
                # intended (exact) gate effect on what the next layer consumes
                row["gate_effect_D3n_rel_full"] = rel(out[("full", "g")]["D3n"], ref["D3n"])
                row["gate_effect_D21n_rel_full"] = rel(out[("full", "g")]["D21n"], ref["D21n"])
                # compression error on the consumed objects, identity vs gated
                for gk in ("id", "g"):
                    row[f"r4_compress_err_D3n_{gk}"] = rel(out[("r4", gk)]["D3n"],
                                                           out[("full", gk)]["D3n"])
                    row[f"r4_compress_err_D21n_{gk}"] = rel(out[("r4", gk)]["D21n"],
                                                            out[("full", gk)]["D21n"])
                rows.append(row)
                print(json.dumps({"leak": arm, **row}), flush=True)
                s = out[("r4", "id")]["state"]
                st = (torch.where(torch.isfinite(s[0]), s[0], torch.zeros_like(s[0])),
                      s[1], s[2], s[3], s[4])
                del out
                torch.cuda.empty_cache()
        leak[arm] = rows
    rep["leak"] = leak
    rep["seconds"] = round(time.time() - t0, 1)
    rep["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    (OUT / "check_gpu.json").write_text(json.dumps(rep, indent=1))
    print("wrote", OUT / "check_gpu.json", flush=True)


if __name__ == "__main__":
    {"small": cmd_small, "gpu": cmd_gpu}[sys.argv[1]]()
