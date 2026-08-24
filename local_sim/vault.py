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
# APPEND, never insert. `local_sim/` holds numpy shims named `flopscope` and
# `whestbench` for the torch-only GPU venv; prepending this directory makes them
# shadow the real packages, and the repaired evaluator MEASURES counted FLOPs
# with the real flopscope. `_run_candidate` asserts it got the real one.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

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


MAX_TOUCHES = 3
_REASONS = ("seal", "estimator-change", "designation")


def _file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_candidate(candidate, seeds, truth, budget, limit_note=""):
    """Run an actual estimator FILE on the given networks, under real flopscope.

    Returns per-MLP (mse, counted flops, residual seconds) and the predictions.

    This replaces the v2.4 torch mirror that `evaluate` used to call. The mirror
    was a second implementation of a program that no longer ships, priced by a
    hardcoded v2.4 cost model, and nothing checked that either still matched the
    file being attributed. Here the candidate file IS the program, and counted
    FLOPs are MEASURED by flopscope rather than modelled — the same quantity the
    grader bills. No GPU is involved: the vault stores its truth, so evaluation
    is a forward pass and a subtraction.
    """
    import importlib.util

    import flopscope as flops

    if os.path.abspath(os.path.dirname(flops.__file__ or "")).startswith(
            os.path.abspath(_HERE)):
        raise SystemExit(
            "REFUSED: `flopscope` resolved to the local numpy SHIM in local_sim/, "
            "not the real package. Counted FLOPs measured against the shim are not "
            "the grader's FLOPs. Run this from the MAIN venv (`uv run python "
            "local_sim/vault.py ...`), not the GPU venv."
        )

    spec = importlib.util.spec_from_file_location("_vault_candidate", candidate)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_vault_candidate"] = mod
    spec.loader.exec_module(mod)

    import harness

    mses, fl, rs, preds, nonfinite = [], [], [], [], 0
    t0 = time.perf_counter()
    for i, s in enumerate(seeds):
        w_np, est_seed, _ = canonical_mint(WIDTH, DEPTH, int(s))
        mlp = harness.MLP(weights=[np.asarray(x) for x in w_np],
                          width=WIDTH, depth=DEPTH, seed=est_seed)
        ctx = flops.BudgetContext(flop_budget=budget, quiet=True)
        with ctx:
            pred = mod.Estimator().predict(mlp, budget)
        p = np.asarray(pred, dtype=np.float64)
        preds.append(p[-1])
        fl.append(float(ctx.flops_used))
        rs.append(float(ctx.residual_wall_time_s or 0.0))
        if not np.all(np.isfinite(p)):
            nonfinite += 1
            mses.append(float("inf"))
            continue
        mses.append(float(np.mean((p[-1] - truth[i, -1]) ** 2)))
        if (i + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print(f"  {i + 1}/{len(seeds)}  {el:.0f}s  "
                  f"eta {el / (i + 1) * (len(seeds) - i - 1):.0f}s{limit_note}", flush=True)
    return np.asarray(mses), np.asarray(fl), np.asarray(rs), np.asarray(preds), nonfinite


def _score_block(mses, fl, rs, budget, gt_noise, nonfinite, extra):
    """Everything the vault reports, through the canonical scorer."""
    from competition_score import (DEFAULT_LAMBDA_FLOPS_PER_SECOND, MlpRow,
                                   effective_compute, score_suite)

    rows = [MlpRow(mse=float(m), flops_used=float(f), residual_wall_time_s=0.0,
                   budget=float(budget))
            for m, f in zip(mses, fl)]
    s = score_suite(rows, on_over_budget="fail")
    a, m = s.scores, s.mses
    local_c = np.asarray([effective_compute(f, r) for f, r in zip(fl, rs)])
    out = {
        "n_mlps": int(s.n),
        "nonfinite": int(nonfinite),
        # PRIMARY. Accuracy is what the vault can actually measure.
        "mse_mean": float(m.mean()),
        "mse_median": _quantile(m, 0.5),
        "gt_noise_mean": float(gt_noise.mean()),
        "gt_noise_share_of_mse": float(gt_noise.mean() / m.mean()),
        # SECONDARY, and model-labelled. `C` here is COUNTED FLOPS ONLY: this
        # machine's residual wall time is classified E1 in docs/CURRENT_STATE.md,
        # so folding it in would make the vault a measurement of this laptop.
        "cost_model": "counted FLOPs only; residual EXCLUDED (local residual is E1)",
        "objective": float(s.objective),
        "median": _quantile(a, 0.5),
        "mean": float(a.mean()),
        "p90": _quantile(a, 0.90),
        "p99": _quantile(a, 0.99),
        "max": float(a.max()),
        "cb_mean": float(s.multipliers.mean()),
        "counted_flops_mean": float(np.mean(fl)),
        # Reported so the exclusion above is visible rather than silent.
        "local_residual_s_median": float(np.median(rs)),
        "local_effective_compute_mean_INFORMATIONAL": float(local_c.mean()),
        # The rate this block ACTUALLY used, not a constant that happens to be in
        # scope. Until 2026-08-24 this recorded `LAMBDA_FLOPS_PER_SECOND` while
        # `effective_compute` above inherited the module default -- and P2-P0 moved
        # those apart: the name became a DEPRECATED ALIAS for the Phase-1 rate
        # (1e11) while the live default went to 0. A touch record is one of three
        # that will ever exist; it must not state a rate it did not use.
        "lambda": DEFAULT_LAMBDA_FLOPS_PER_SECOND,
    }
    out.update(extra)
    return out


def evaluate(label, reason, candidate="estimator.py", budget=272_000_000_000,
             limit=0, sacrificial=""):
    """Evaluate an explicitly named candidate FILE. Increments the touch counter.

    `sacrificial` is reached ONLY through the separate `sacrificial` subcommand,
    never through `evaluate`. Both agent hooks and the git pre-commit backstop
    guard the string "vault.py evaluate" (scripts/guard_paths.py), so the test
    path deliberately does not wear that name — and, more importantly, it never
    opens the vault's arrays at all.
    """
    if reason not in _REASONS:
        raise SystemExit(f"reason {reason!r} is not a permitted vault touch "
                         f"({' | '.join(_REASONS)})")
    cand_path = candidate if os.path.isabs(candidate) else os.path.join(_ROOT, candidate)
    if not os.path.exists(cand_path):
        raise SystemExit(f"candidate {candidate!r} does not exist")
    cand_sha = _file_sha(cand_path)

    if sacrificial:
        sp = sacrificial if os.path.isabs(sacrificial) else os.path.join(_ROOT, sacrificial)
        d = np.load(sp)
        seeds, truth_final, gt_noise = d["seeds"], d["truth"], d["sigma_gt2"]
        truth = truth_final.reshape(len(seeds), 1, WIDTH)
        truth = np.broadcast_to(truth, (len(seeds), DEPTH, WIDTH))
        man = {"touches": None}
        print(f"SACRIFICIAL RUN on {sacrificial} — {len(seeds)} NON-VAULT networks.")
        print("Nothing is recorded and no touch is taken.\n", flush=True)
    else:
        d, man = load(check=True)
        seeds, truth, gt_noise = d["seeds"], d["truth"], d["gt_noise"]
        if man["touches"] >= MAX_TOUCHES:
            raise SystemExit(
                f"REFUSED: the vault has already been touched {man['touches']} times "
                f"and the maximum is {MAX_TOUCHES} (seal, one estimator change, "
                f"designation). Re-read experiments/vault/README.md before doing "
                f"anything that would need a fourth."
            )
    if limit:
        seeds, truth, gt_noise = seeds[:limit], truth[:limit], gt_noise[:limit]

    print(f"VAULT EVALUATION — {label}   ({len(seeds)} MLPs, reason={reason})")
    print(f"  candidate  {candidate}")
    print(f"  sha256     {cand_sha}")
    print(f"  budget     {budget:,}\n", flush=True)

    mses, fl, rs, preds, nonfinite = _run_candidate(cand_path, seeds, truth, budget)
    np.savez(os.path.join(VAULT_DIR,
                          "sacrificial_preds.npz" if sacrificial else "candidate_preds.npz"),
             seeds=seeds, preds=preds.astype(np.float64))

    res = _score_block(
        mses, fl, rs, budget, gt_noise, nonfinite,
        {
            "touch": (man["touches"] + 1) if man["touches"] is not None else None,
            "utc": _now(),
            "label": label,
            "reason": reason,
            "candidate": candidate,
            "candidate_sha256": cand_sha,
            "budget": int(budget),
            "sacrificial": sacrificial or None,
        },
    )
    print("\n" + "=" * 88)
    print(f"VAULT RESULT — {label}")
    print("=" * 88)
    print(f"  PRIMARY (accuracy, what this instrument can measure)")
    print(f"    raw MSE   mean {res['mse_mean']:.4e}   median {res['mse_median']:.4e}")
    print(f"    GT noise  {res['gt_noise_mean']:.3e} = "
          f"{res['gt_noise_share_of_mse'] * 100:.2f}% of measured MSE  "
          f"<- any claim smaller than this is noise")
    print(f"  SECONDARY (score, MODEL-DEPENDENT: {res['cost_model']})")
    print(f"    adjusted  median {res['median']:.4e}   mean {res['mean']:.4e}   "
          f"p90 {res['p90']:.4e}")
    print(f"              p99    {res['p99']:.4e}   max  {res['max']:.4e}   "
          f"max/med {res['max'] / res['median']:.2f}x")
    print(f"    C/B mean  {res['cb_mean']:.4f}   counted "
          f"{res['counted_flops_mean']:.4e}   non-finite {nonfinite}")
    print(f"    local residual {res['local_residual_s_median'] * 1e3:.2f} ms/MLP "
          f"— EXCLUDED from C above and shown only so the exclusion is visible")

    if sacrificial:
        print("\n  SACRIFICIAL: nothing recorded, no touch taken, seal untouched.")
        return res

    # Atomic-ish seal update. The manifest goes first, via os.replace, so a crash
    # can leave the counter BEHIND the log but never ahead of it — and preflight
    # step 5 already fails on a mismatch in either direction. The old order
    # (append log, then rewrite manifest) could leave the counter behind too, but
    # with a partially written manifest; this cannot.
    man["touches"] = res["touch"]
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, MANIFEST)
    with open(TOUCHLOG, "a") as f:
        f.write(json.dumps(res) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(f"\n  touch #{res['touch']} of {MAX_TOUCHES} recorded in "
          f"{os.path.relpath(TOUCHLOG, _ROOT)}")
    print(f"  bound to {res['candidate']} sha256 {res['candidate_sha256'][:16]}...")
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
    e.add_argument("--reason", required=True, choices=list(_REASONS))
    e.add_argument("--candidate", default="estimator.py",
                   help="the estimator FILE to evaluate. Recorded with its sha256 "
                        "in the touch log, so a touch names a program, not a mood.")
    e.add_argument("--budget", type=int, default=272_000_000_000)
    e.add_argument("--limit", type=int, default=0)
    # NOTE: there is deliberately no --dry-run. Reading the truth IS the touch;
    # a flag that reads it and records nothing is exactly the hole the seal
    # exists to close. See experiments/LOG.md 2026-08-19, P0.2 finding V3.
    sac = sub.add_parser(
        "sacrificial",
        help="run the EVALUATE code path against a NON-vault truth set. Records "
             "nothing, takes no touch, and never opens the vault's arrays. This "
             "is how the evaluator is tested without spending what it evaluates.")
    sac.add_argument("--truth", required=True,
                     help="npz with seeds / truth (M,width) / sigma_gt2")
    sac.add_argument("--candidate", default="estimator.py")
    sac.add_argument("--budget", type=int, default=272_000_000_000)
    sac.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.cmd == "mint":
        mint(a.samples, a.chunk)
    elif a.cmd == "verify":
        sys.exit(0 if verify() else 1)
    elif a.cmd == "readme":
        write_readme()
    elif a.cmd == "shipped-check":
        shipped_check(a.limit)
    elif a.cmd == "sacrificial":
        evaluate("sacrificial", "seal", a.candidate, a.budget, a.limit, a.truth)
    else:
        evaluate(a.label, a.reason, a.candidate, a.budget, a.limit)
        write_readme()


if __name__ == "__main__":
    main()
