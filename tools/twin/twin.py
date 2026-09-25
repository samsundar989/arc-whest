"""The differentiable twin: the SHIPPED chain source, run on torch, trained end to end.

    .venv-gpu/Scripts/python.exe tools/twin/twin.py parity   [--cand F] [--nets 0,1,2]
    .venv-gpu/Scripts/python.exe tools/twin/twin.py eval     --panel dev|hold100|train [--params P]
    .venv-gpu/Scripts/python.exe tools/twin/twin.py train    --stage S1 --out DIR [--epochs 2]

HANDOFF 2026-09-25 Step 2 / PLAN rev 13 §3 Step 2.

WHY NOT A RE-IMPLEMENTATION. The handoff asks for a PyTorch re-implementation of
v59's forward math and accepts it at 1e-4 relative on per-net MSE. Rather than
write the Wick program, closure tables and tower a second time -- which is how a
twin silently stops being a twin -- this runs the estimator file ITSELF with
`flopscope` replaced by a torch shim (`tools/twin/shim`). Every array is a
`torch.Tensor`, so autograd flows through exactly the arithmetic that ships:
Strassen products, `_advance`, `_dslice3/21`, `_shared_join_lean`, hubs, P2
compress (`torch.linalg.qr`), kappa4 regeneration, the corrector pages. Parity is
then a float32 round-off question, checked by `parity` against the cupy probe.

Memory: one `torch.utils.checkpoint` per layer (non-reentrant), one net per
forward, gradients accumulated over `--accum` nets per optimiser step.

PARAMETERS are a flat dict of leaf tensors (`Params`) so any subset can be frozen:
  S1  corrector pages W1,b1,W2,b2,W3,b3 (the standardisers fmu/fisd stay frozen)
  (S2+ -- lambda/theta/feed profiles, term gains, multi-mode kappa4, sigma page --
  are added by patching the module's scalar hooks; see `Params.install`.)

Objective: mean final-layer MSE over TRAIN nets, + `--w-all` x mean MSE of rows
1..depth-2 (row 0 is an exact closed form in `predict`; the last row is the
objective). Outputs are clamped at 0 as `predict` does.

Panels (`data/panels/twin_train_split.json`): TRAIN = full rows 0..899, HOLDOUT-100
= rows 900..999, DEV = mini 0..99 (never trained on, read for selection).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHIM = ROOT / "tools" / "twin" / "shim"
sys.path.insert(0, str(SHIM))                     # flopscope -> torch (this process only)
sys.path.append(str(ROOT / "local_sim"))          # whestbench shim (MLP, BaseEstimator)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

import flopscope.numpy as fnp  # noqa: E402  (the torch shim)

assert "twin" in fnp.__file__, f"wrong flopscope on the path: {fnp.__file__}"
torch.backends.cuda.matmul.allow_tf32 = False     # float32 means float32
torch.backends.cudnn.allow_tf32 = False

BUDGET = 2 ** 41
DEV = torch.device("cuda")
CACHE = ROOT / "cache" / "twin"
DEFAULT_CAND = ROOT / "estimator_k3_v59_corr400.py"
TRAINABLE = {"S1": ("W1", "b1", "W2", "b2", "W3", "b3")}


# --------------------------------------------------------------------- data

class Data:
    def __init__(self):
        self.w, self.truth, self.meta = {}, {}, {}
        for split, n in (("full", 1000), ("mini", 100)):
            p = CACHE / f"{split}_w.f32"
            if p.exists():
                self.w[split] = np.memmap(p, dtype=np.float32, mode="r",
                                          shape=(n, 16, 1024, 1024))
                self.truth[split] = np.load(CACHE / f"{split}_truth.npy")
                self.meta[split] = json.loads((CACHE / f"{split}_meta.json").read_text())

    def net(self, split, i):
        if not self.meta[split]["done"][i]:
            raise RuntimeError(f"{split} row {i} not exported yet")
        w = [torch.from_numpy(np.array(self.w[split][i, l])).to(DEV) for l in range(16)]
        return w, self.truth[split][i]


def panel(name):
    s = json.loads((ROOT / "data" / "panels" / "twin_train_split.json").read_text())
    if name == "train":
        return [("full", i) for i in s["train"]["indices"]]
    if name == "hold100":
        return [("full", i) for i in s["holdout_100"]["indices"]]
    if name == "dev":
        return [("mini", i) for i in range(100)]
    if name.startswith("fit"):
        return [("full", i) for i in range(int(name[3:] or 100))]
    raise SystemExit(f"unknown panel {name}")


# -------------------------------------------------------------------- model

class _Ctx:
    def __init__(self, width, depth, budget, seed):
        self.width, self.depth = width, depth
        self.flop_budget, self.seed = budget, seed
        self.api_version, self.scratch_dir, self.submission_dir = "1.0", None, None


def load_module(path: Path, name="_twin_candidate"):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Twin:
    def __init__(self, cand: Path, width=1024, depth=16):
        self.mod = load_module(cand)
        self.est = self.mod.Estimator()
        self.est.setup(_Ctx(width, depth, BUDGET, 0))
        self.width, self.depth = width, depth
        self.dtype = fnp.float32
        self.table = self.mod._tables()
        self.gens = self.mod._fit_generations(width, depth, BUDGET)
        assert self.gens is None, "generation cap -- not the exact tower"
        self.const = self.est._constants(width, self.dtype)
        self.corr0 = {k: v.detach().clone() for k, v in self.const["corr"].items()}

    def install(self, params: dict):
        """Point the chain's corrector at `params` (leaf tensors) + frozen rest."""
        corr = dict(self.corr0)
        corr.update(params)
        self.const["corr"] = corr

    def _step(self, l, W, state):
        s = self.mod._layer(W, state, self.table, self.dtype, self.const, 2.0,
                            last_layer=(l == self.depth - 1), gens=self.gens,
                            layer_index=l, depth=self.depth)
        mu = torch.where(torch.isfinite(s[0]), s[0], torch.zeros_like(s[0]))
        return (mu, s[1], s[2], s[3], s[4])

    def forward(self, w, ckpt=False):
        """Rows 1..depth-1 of the prediction (clamped at 0), as a list of tensors."""
        n = self.width
        state = (fnp.zeros(n), fnp.eye(n), None, 0.0, fnp.zeros(n))
        mus = []
        for l in range(self.depth):
            if ckpt:
                state = checkpoint(self._step, l, w[l], state, use_reentrant=False)
            else:
                state = self._step(l, w[l], state)
            mus.append(torch.clamp_min(state[0], 0.0))
        return mus


def mse_rows(mus, truth):
    """Per-row MSE (float64) for rows 1..depth-1."""
    t = torch.from_numpy(np.asarray(truth)).to(DEV)
    return [((m.double() - t[l]) ** 2).mean() for l, m in enumerate(mus) if l >= 1]


# ------------------------------------------------------------------ commands

def cmd_parity(a):
    """Twin forward vs the cupy probe (anatomy2's plain chain MSE on the same nets)."""
    tw = Twin(Path(a.cand))
    data = Data()
    ref = json.loads(Path(a.ref).read_text())["nets"] if a.ref else {}
    out = []
    for i in [int(x) for x in a.nets.split(",")]:
        w, truth = data.net("full", i)
        t0 = time.time()
        with torch.no_grad():
            mus = tw.forward(w)
        torch.cuda.synchronize()
        dt = time.time() - t0
        m = [float(x) for x in mse_rows(mus, truth)]
        r = ref.get(str(i), {}).get("chain", {}).get(str(tw.depth - 1))
        rel = (m[-1] / r[0] - 1) if r else float("nan")
        out.append({"net": i, "twin_final_mse": m[-1], "cupy_final_mse": r[0] if r else None,
                    "rel": rel, "seconds": round(dt, 2)})
        print(json.dumps(out[-1]), flush=True)
    return out


def cmd_eval(a, tw=None, params=None, quiet=False):
    tw = tw or Twin(Path(a.cand))
    if params is None and a.params:
        params = {k: torch.from_numpy(v).to(DEV) for k, v in np.load(a.params).items()}
    if params is not None:
        tw.install(params)
    data = Data()
    rows = panel(a.panel)[: a.n or None]
    fin = []
    with torch.no_grad():
        for split, i in rows:
            w, truth = data.net(split, i)
            fin.append(float(mse_rows(tw.forward(w), truth)[-1]))
    r = {"panel": a.panel, "n": len(fin), "raw_mse": float(np.mean(fin)), "per_net": fin}
    if not quiet:
        print(json.dumps({k: v for k, v in r.items() if k != "per_net"}), flush=True)
    return r


def cmd_train(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tw = Twin(Path(a.cand))
    names = TRAINABLE[a.stage]
    params = {k: tw.corr0[k].clone().requires_grad_(True) for k in names}
    if a.init:
        for k, v in np.load(a.init).items():
            if k in params:
                params[k] = torch.from_numpy(v).to(DEV).requires_grad_(True)
    tw.install(params)
    opt = torch.optim.Adam(list(params.values()), lr=a.lr)
    data = Data()
    train = panel("train")
    rng = np.random.default_rng(20260925)
    steps_per_epoch = len(train) // a.accum
    total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    log = open(out / "train.log", "a", encoding="utf-8")

    def evaluate(tag):
        ev = argparse.Namespace(panel="hold100", n=a.hold_n, cand=a.cand, params=None)
        r = cmd_eval(ev, tw=tw, params=params, quiet=True)
        msg = f"[{tag}] HOLDOUT-100 raw {r['raw_mse']:.6e} (n={r['n']})"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
        return r["raw_mse"]

    best = evaluate("init")
    np.savez(out / "params_best.npz", **{k: v.detach().cpu().numpy() for k, v in params.items()})
    bad = 0
    step = 0
    progress = os.environ.get("WORKERQ_PROGRESS")
    t0 = time.time()
    for ep in range(a.epochs):
        order = rng.permutation(len(train))
        for s in range(steps_per_epoch):
            opt.zero_grad(set_to_none=True)
            lsum = 0.0
            for j in order[s * a.accum:(s + 1) * a.accum]:
                split, i = train[j]
                w, truth = data.net(split, i)
                mus = tw.forward(w, ckpt=True)
                rows = mse_rows(mus, truth)
                loss = rows[-1] + a.w_all * torch.stack(rows[:-1]).mean()
                (loss * (1e8 / a.accum)).backward()
                lsum += float(rows[-1])
                del mus, rows, loss, w
            opt.step()
            sched.step()
            step += 1
            msg = (f"ep {ep} step {step}/{total} train final {lsum / a.accum:.4e} "
                   f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
            log.write(msg + "\n")
            if step % 10 == 0:
                print(msg, flush=True)
                log.flush()
            if progress:
                try:
                    Path(progress).write_text(str(step / total))
                except OSError:
                    pass
            if step % a.eval_every == 0 or step == total:
                h = evaluate(f"step {step}")
                if h < best:
                    best, bad = h, 0
                    np.savez(out / "params_best.npz",
                             **{k: v.detach().cpu().numpy() for k, v in params.items()})
                else:
                    bad += 1
                    if bad >= a.patience:
                        log.write(f"early stop at step {step}\n")
                        print(f"early stop at step {step}", flush=True)
                        return
    log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["parity", "eval", "train"])
    ap.add_argument("--cand", default=str(DEFAULT_CAND))
    ap.add_argument("--nets", default="0,1,2")
    ap.add_argument("--ref", default=str(ROOT / "experiments" / "2026-09-25_anatomy2" / "fit20.json"))
    ap.add_argument("--panel", default="dev")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--params", default="")
    ap.add_argument("--stage", default="S1")
    ap.add_argument("--init", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-all", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=28)
    ap.add_argument("--hold-n", type=int, default=100)
    ap.add_argument("--patience", type=int, default=3)
    a = ap.parse_args()
    {"parity": cmd_parity, "eval": cmd_eval, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
