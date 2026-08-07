"""
THE VAULT — a sealed 500-MLP holdout. Our private-private test.

Why it exists
-------------
The prize is decided by re-running one designated submission on MLPs nobody has
seen. Everything else we measure on has been seen: the public mini/full splits
have steered a month of decisions, and the fresh-mint blocks get consumed by the
experiments that request them. The vault is the one set that stays unseen until
it is deliberately spent.

It is sealed in the only sense that matters for a single-operator project: the
contents are hashed, the hash is written down, and **every evaluation increments
a counter and appends a line to the log**. There is no way to run against the
vault quietly. If the counter says 6 and the log says 6, the baseline you are
reading was not chosen by re-rolling until it looked good.

Permitted touches (anything else is a protocol violation, and the log will show it):
    (i)   the v2.4.1 baseline, once, at seal time
    (ii)  any future estimator change, once per change
    (iii) designation day

What is stored
--------------
Seeds and truth — **not weights**. `mint_provenance` proves the weights are a
bit-exact function of the input seed, so storing 4.2 GB of float32 would add
nothing a 4 KB seed array does not already pin down. That is the provenance
proof paying rent.

    seeds        (500,)          int64   input seeds, block V
    truth        (500, 32, 256)  float64 post-ReLU means, N samples/MLP
    truth_var    (500, 256)      float64 final-layer per-neuron sample variance
    gt_noise     (500,)          float64 per-MLP GT-noise MSE = mean_i Var_i / N

Run:
    .venv-gpu\\Scripts\\python.exe local_sim/vault.py mint --samples 33554432
    .venv-gpu\\Scripts\\python.exe local_sim/vault.py verify
    .venv-gpu\\Scripts\\python.exe local_sim/vault.py evaluate --label "v2.4.1 baseline" --reason seal
"""

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mint_provenance import canonical_mint, seed_streams  # noqa: E402
from seed_blocks import VAULT_BLOCK  # noqa: E402

VAULT_DIR = os.path.join(_ROOT, "experiments", "vault")
ARRAYS = os.path.join(VAULT_DIR, "vault_v1.npz")
MANIFEST = os.path.join(VAULT_DIR, "manifest.json")
TOUCHLOG = os.path.join(VAULT_DIR, "touch_log.jsonl")

WIDTH, DEPTH = 256, 32
_, SEED0, N_MLPS, _ = VAULT_BLOCK


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _now():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------- mint ---
def mint(n_samples, chunk=131072):
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(VAULT_DIR, exist_ok=True)
    if os.path.exists(ARRAYS):
        raise SystemExit(f"{ARRAYS} already exists — the vault is sealed. Refusing to remint.\n"
                         f"(Deleting it would destroy the audit trail; if a remint is truly "
                         f"intended, move the old vault aside by hand and say so in the log.)")

    print(f"device {torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'}   "
          f"{N_MLPS} MLPs x {WIDTH}x{DEPTH}   N = {n_samples:,} samples/MLP", flush=True)
    print(f"seed block V: {SEED0:,} .. {SEED0 + N_MLPS - 1:,}\n", flush=True)

    seeds, truth, tvar, noise = [], [], [], []
    t0 = time.perf_counter()
    for i in range(N_MLPS):
        s = SEED0 + i
        w_np, _est, sample_ss = canonical_mint(WIDTH, DEPTH, s)
        w = torch.from_numpy(np.stack(w_np)).to(dev)
        gt_seed = int(sample_ss.generate_state(1)[0]) & 0x7FFFFFFF
        g = torch.Generator(device=dev).manual_seed(gt_seed)

        acc = torch.zeros((DEPTH, WIDTH), dtype=torch.float64, device=dev)
        acc_sq = torch.zeros(WIDTH, dtype=torch.float64, device=dev)
        done = 0
        while done < n_samples:
            b = min(chunk, n_samples - done)
            h = torch.randn(b, WIDTH, generator=g, device=dev, dtype=torch.float32)
            for li in range(DEPTH):
                h = torch.clamp(h @ w[li], min=0.0)
                acc[li] += h.sum(dim=0, dtype=torch.float64)
            acc_sq += (h.double() ** 2).sum(dim=0)
            done += b

        m = (acc / n_samples).cpu().numpy()
        v = (acc_sq / n_samples).cpu().numpy() - m[-1] ** 2
        v = np.maximum(v, 0.0)
        seeds.append(s)
        truth.append(m)
        tvar.append(v)
        noise.append(float(v.mean() / n_samples))
        if (i + 1) % 25 == 0:
            el = time.perf_counter() - t0
            print(f"  {i + 1:>4}/{N_MLPS}   {el:7.1f}s   eta {el / (i + 1) * (N_MLPS - i - 1):7.1f}s   "
                  f"gt-noise {np.mean(noise):.3e}", flush=True)

    seeds = np.asarray(seeds, dtype=np.int64)
    truth = np.stack(truth).astype(np.float64)
    tvar = np.stack(tvar).astype(np.float64)
    noise = np.asarray(noise, dtype=np.float64)

    np.savez(ARRAYS, seeds=seeds, truth=truth, truth_var=tvar, gt_noise=noise)
    manifest = {
        "created_utc": _now(),
        "seed_block": "V",
        "seed_start": int(SEED0),
        "n_mlps": int(N_MLPS),
        "width": WIDTH,
        "depth": DEPTH,
        "gt_samples_per_mlp": int(n_samples),
        "mint_protocol": "whestbench.sample_mlp via mint_provenance.canonical_mint "
                         "(proven bit-identical, 100/100)",
        "gt_stream": "seed_streams(input_seed)[1] -> torch generator",
        "sha256": {"seeds": _sha(seeds), "truth": _sha(truth),
                   "truth_var": _sha(tvar), "gt_noise": _sha(noise)},
        "gt_noise_mse_mean": float(noise.mean()),
        "gt_noise_mse_max": float(noise.max()),
        "touches": 0,
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    open(TOUCHLOG, "a").close()

    print(f"\nsealed {ARRAYS}")
    print(f"  truth   sha256 {manifest['sha256']['truth'][:32]}...")
    print(f"  seeds   sha256 {manifest['sha256']['seeds'][:32]}...")
    print(f"  GT-noise MSE: mean {noise.mean():.3e}   max {noise.max():.3e}")
    print(f"  (decision signal is ~1.4e-6 raw final-layer MSE -> GT noise is "
          f"{noise.mean() / 1.4e-6 * 100:.2f}% of it)")


# --------------------------------------------------------------- verify ---
def load(check=True):
    if not os.path.exists(ARRAYS):
        raise SystemExit("vault not minted yet — run `vault.py mint` first")
    d = np.load(ARRAYS)
    man = json.load(open(MANIFEST))
    if check:
        for k in ("seeds", "truth", "truth_var", "gt_noise"):
            got = _sha(d[k])
            if got != man["sha256"][k]:
                raise SystemExit(f"VAULT CORRUPT: {k} sha256 {got[:16]} != "
                                 f"manifest {man['sha256'][k][:16]}")
    return d, man


def verify():
    d, man = load(check=True)
    print("=" * 88)
    print("VAULT INTEGRITY")
    print("=" * 88)
    print(f"created            {man['created_utc']}")
    print(f"shape              {man['n_mlps']} MLPs, {man['width']}x{man['depth']}")
    print(f"GT samples/MLP     {man['gt_samples_per_mlp']:,}")
    print(f"GT-noise MSE       mean {man['gt_noise_mse_mean']:.3e}  "
          f"max {man['gt_noise_mse_max']:.3e}")
    print(f"touches so far     {man['touches']}")
    for k in ("seeds", "truth", "truth_var", "gt_noise"):
        print(f"  {k:<10} sha256 OK  {man['sha256'][k][:48]}...")

    # weights are NOT stored — prove they are still recoverable from the seeds
    n_spot = 5
    print(f"\nreconstructing {n_spot} MLPs from seeds to confirm the seed->weights map:")
    for i in np.linspace(0, man["n_mlps"] - 1, n_spot).astype(int):
        s = int(d["seeds"][i])
        w, est, _ = canonical_mint(man["width"], man["depth"], s)
        print(f"  [{i:>3}] seed {s:,}  est_seed {est:>12}  "
              f"w[0] sha {_sha(np.stack(w))[:16]}...  ||w||_F {np.linalg.norm(np.stack(w)):.4f}")

    n = 0
    if os.path.exists(TOUCHLOG):
        with open(TOUCHLOG) as f:
            lines = [json.loads(x) for x in f if x.strip()]
        n = len(lines)
        print(f"\nTOUCH LOG ({n} entries)")
        print("-" * 88)
        for e in lines:
            print(f"  #{e['touch']:>2} {e['utc']}  {e['reason']:<12} {e['label']}")
            print(f"      median {e['median']:.4e}  mean {e['mean']:.4e}  "
                  f"p99 {e['p99']:.4e}  max {e['max']:.4e}")
    if n != man["touches"]:
        print(f"\n  !! counter/log mismatch: manifest says {man['touches']}, log has {n}")
        return False
    print(f"\ncounter and log agree ({n}).")
    return True


# ------------------------------------------------------------- evaluate ---
def _quantile(x, q):
    s = np.sort(x)
    return float(s[min(len(s) - 1, int(math.ceil(q * len(s)) - 1))])


def evaluate(label, reason, n_points=16384, limit=0, dry_run=False):
    """Score the shipped pipeline on the vault. Increments the touch counter."""
    import torch

    from robustness import FIXED_FLOPS, run_v24
    from track_t_mint import sobol_base

    if reason not in ("seal", "estimator-change", "designation"):
        raise SystemExit(f"reason {reason!r} is not a permitted vault touch "
                         f"(seal | estimator-change | designation)")

    d, man = load(check=True)
    seeds, truth, gt_noise = d["seeds"], d["truth"], d["gt_noise"]
    if limit:
        seeds, truth, gt_noise = seeds[:limit], truth[:limit], gt_noise[:limit]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    base_pts = sobol_base(n_points, WIDTH, dev)

    print(f"VAULT EVALUATION — {label}   ({len(seeds)} MLPs, reason={reason})", flush=True)
    mses, adjs, cbs, nonfinite = [], [], [], 0
    preds_out = []
    t0 = time.perf_counter()
    for i, s in enumerate(seeds):
        w_np, est_seed, _ = canonical_mint(WIDTH, DEPTH, int(s))
        w = torch.from_numpy(np.stack(w_np)).to(dev)
        pred, cps, _diag = run_v24(w, est_seed, base_pts, dev, n_points)
        preds_out.append(pred.cpu().numpy())
        if not bool(torch.isfinite(pred).all()):
            nonfinite += 1
            continue
        t = torch.from_numpy(truth[i, -1]).to(dev)
        mse = float(((pred - t) ** 2).mean())
        cb = (n_points * cps + FIXED_FLOPS) / 2.72e11
        mses.append(mse)
        cbs.append(cb)
        adjs.append(mse * max(0.1, cb))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(seeds)}  {time.perf_counter() - t0:.0f}s", flush=True)

    if preds_out:
        np.savez(os.path.join(VAULT_DIR, "mirror_preds.npz"),
                 seeds=seeds, preds=np.stack(preds_out).astype(np.float64))

    a = np.asarray(adjs)
    m = np.asarray(mses)
    res = {
        "touch": man["touches"] + 1,
        "utc": _now(),
        "label": label,
        "reason": reason,
        "n_mlps": int(len(a)),
        "n_points": int(n_points),
        "nonfinite": int(nonfinite),
        "median": _quantile(a, 0.5),
        "mean": float(a.mean()),
        "p90": _quantile(a, 0.90),
        "p99": _quantile(a, 0.99),
        "max": float(a.max()),
        "mse_mean": float(m.mean()),
        "mse_median": _quantile(m, 0.5),
        "cb_mean": float(np.mean(cbs)),
        "gt_noise_mean": float(gt_noise.mean()),
        "gt_noise_share_of_mse": float(gt_noise.mean() / m.mean()),
    }

    print("\n" + "=" * 88)
    print(f"VAULT RESULT — {label}")
    print("=" * 88)
    print(f"  adjusted   median {res['median']:.4e}   mean {res['mean']:.4e}   "
          f"p90 {res['p90']:.4e}")
    print(f"             p99    {res['p99']:.4e}   max  {res['max']:.4e}   "
          f"max/med {res['max'] / res['median']:.2f}x")
    print(f"  raw MSE    mean   {res['mse_mean']:.4e}   median {res['mse_median']:.4e}")
    print(f"  C/B mean   {res['cb_mean']:.4f}    non-finite predictions {nonfinite}")
    print(f"  GT noise   {res['gt_noise_mean']:.3e} = "
          f"{res['gt_noise_share_of_mse'] * 100:.2f}% of measured MSE  "
          f"<- any claim smaller than this is noise")

    if dry_run:
        print("\n  --dry-run: touch NOT recorded")
        return res
    with open(TOUCHLOG, "a") as f:
        f.write(json.dumps(res) + "\n")
    man["touches"] = res["touch"]
    with open(MANIFEST, "w") as f:
        json.dump(man, f, indent=2)
    print(f"\n  touch #{res['touch']} recorded in {os.path.relpath(TOUCHLOG, _ROOT)}")
    return res


def write_readme():
    """Regenerate experiments/vault/README.md from the manifest and touch log."""
    _d, man = load(check=True)
    touches = []
    if os.path.exists(TOUCHLOG):
        with open(TOUCHLOG) as f:
            touches = [json.loads(x) for x in f if x.strip()]

    L = []
    a = L.append
    a("# The vault — sealed holdout")
    a("")
    a("*Generated by `vault.py readme`. Do not hand-edit — edit the manifest or add")
    a("a touch and regenerate, so this file can never drift from the seal it reports.*")
    a("")
    a("This is the private-private test: the final re-run before the final re-run.")
    a("Protocol and rationale in `docs/VALIDATION.md` §4.")
    a("")
    a("## Contents")
    a("")
    a("| field | value |")
    a("|---|---|")
    a(f"| created (UTC) | {man['created_utc']} |")
    a(f"| seed block | **{man['seed_block']}**, input seeds "
      f"{man['seed_start']:,} … {man['seed_start'] + man['n_mlps'] - 1:,} |")
    a(f"| networks | {man['n_mlps']} × {man['width']}×{man['depth']} |")
    a(f"| ground truth | {man['gt_samples_per_mlp']:,} samples/MLP |")
    a(f"| GT-noise MSE | mean **{man['gt_noise_mse_mean']:.3e}**, "
      f"max {man['gt_noise_mse_max']:.3e} |")
    a(f"| mint protocol | {man['mint_protocol']} |")
    a(f"| GT stream | {man['gt_stream']} |")
    a("")
    a("**Weights are not stored.** `mint_provenance.py` proves the seed→weights map")
    a("is bit-exact against `whestbench.sample_mlp`, so the seed array pins down what")
    a("would otherwise be 4.2 GB of float32. `vault.py verify` re-hashes every array")
    a("and reconstructs a sample of networks from their seeds.")
    a("")
    a("## SHA-256")
    a("")
    a("| array | sha256 |")
    a("|---|---|")
    for k, v in man["sha256"].items():
        a(f"| `{k}` | `{v}` |")
    a("")
    a(f"## TOUCH COUNTER: **{man['touches']}**")
    a("")
    a("Permitted touches, and nothing else: (i) the v2.4.1 baseline at seal time,")
    a("(ii) one per estimator change, (iii) designation day. Every evaluation appends")
    a("to `touch_log.jsonl` and increments this counter, so a reported baseline can be")
    a("checked against the number of attempts that produced it.")
    a("")
    if touches:
        a("| # | UTC | reason | label | n | median | mean | p99 | max | GT-noise share of MSE |")
        a("|---|---|---|---|---|---|---|---|---|---|")
        for e in touches:
            a(f"| {e['touch']} | {e['utc']} | {e['reason']} | {e['label']} | "
              f"{e['n_mlps']} | {e['median']:.4e} | {e['mean']:.4e} | "
              f"{e['p99']:.4e} | {e['max']:.4e} | "
              f"{e.get('gt_noise_share_of_mse', float('nan')) * 100:.2f}% |")
    else:
        a("*No touches yet — the vault is sealed and unread.*")
    a("")

    os.makedirs(VAULT_DIR, exist_ok=True)
    path = os.path.join(VAULT_DIR, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        for line in L:
            print(line, file=f)
    print(f"wrote {path}  (touch counter {man['touches']})")


def shipped_check(limit=50):
    """Shipped estimator.py vs the GPU mirror on vault networks.

    Deliberately reads NO ground truth, so it costs no touch: this compares two
    implementations to each other, which is a parity question, not a measurement.
    Run from the MAIN venv — the GPU venv has no flopscope/whestbench shims.
    """
    import importlib.util

    d, _man = load(check=True)
    mp = os.path.join(VAULT_DIR, "mirror_preds.npz")
    if not os.path.exists(mp):
        raise SystemExit("no mirror_preds.npz — run `vault.py evaluate` first")
    blob = np.load(mp)
    seeds, mirror = blob["seeds"], blob["preds"]

    spec = importlib.util.spec_from_file_location(
        "shipped_est", os.path.join(_ROOT, "estimator.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["shipped_est"] = mod
    spec.loader.exec_module(mod)
    est = mod.Estimator()

    class Ctx:
        width, depth, flop_budget, seed = WIDTH, DEPTH, 272_000_000_000, 0

    est.setup(Ctx())
    from whestbench import MLP  # the local shim

    n = min(limit, len(seeds))
    absd, rel = [], []
    for i in range(n):
        w, est_seed, _ = canonical_mint(WIDTH, DEPTH, int(seeds[i]))
        mlp = MLP(width=WIDTH, depth=DEPTH, weights=w, seed=est_seed)
        p = np.asarray(est.predict(mlp, 272_000_000_000), dtype=np.float64)[-1]
        m = mirror[i]
        absd.append(float(np.abs(p - m).max()))
        rel.append(float(np.abs(p - m).max() / max(float(np.abs(m).max()), 1e-30)))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n}", flush=True)

    print("=" * 88)
    print(f"SHIPPED estimator.py vs GPU MIRROR, {n} vault networks "
          f"(no truth read — costs no touch)")
    print("=" * 88)
    print(f"  max  abs difference   {max(absd):.3e}")
    print(f"  max  rel difference   {max(rel):.3e}")
    print(f"  mean abs difference   {np.mean(absd):.3e}")
    print("  The two paths share the method but not the BLAS or the eigensolver, so")
    print("  agreement here is what licenses quoting mirror numbers for the shipped file.")
    return {"n": n, "max_abs": max(absd), "max_rel": max(rel)}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mint")
    m.add_argument("--samples", type=int, default=33_554_432)
    m.add_argument("--chunk", type=int, default=131072)
    sub.add_parser("verify")
    sub.add_parser("readme")
    sc = sub.add_parser("shipped-check")
    sc.add_argument("--limit", type=int, default=50)
    e = sub.add_parser("evaluate")
    e.add_argument("--label", required=True)
    e.add_argument("--reason", required=True,
                   choices=["seal", "estimator-change", "designation"])
    e.add_argument("--points", type=int, default=16384)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.cmd == "mint":
        mint(a.samples, a.chunk)
    elif a.cmd == "verify":
        sys.exit(0 if verify() else 1)
    elif a.cmd == "readme":
        write_readme()
    elif a.cmd == "shipped-check":
        shipped_check(a.limit)
    else:
        evaluate(a.label, a.reason, a.points, a.limit, a.dry_run)
        write_readme()


if __name__ == "__main__":
    main()
