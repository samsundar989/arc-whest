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

Which round it is
-----------------
The stored arrays are PHASE 1 -- 500 networks at 256x32, minted 2026-08-06. That
is a historical fact and it is preserved, not repurposed. The live round is
Phase 2 at 1024x16, so `evaluate` REFUSES on the sealed corpus until a Phase-2
sealed set exists under its own STRATEGIC contract. See
`_refuse_unless_corpus_matches_live_round` for why neither rulebook can be
borrowed for the other.

The candidate is run by `local_sim/lifecycle_eval.py`, which is the one place the
grader's lifecycle lives. This file used to carry its own prediction loop; it got
the weight dtype, the setup call, the output materialisation and every failure
path wrong (`docs/research/2026-09-10_validation_audit.md`, finding 2).

Run:
    .venv-gpu\\Scripts\\python.exe local_sim/vault.py mint --samples 33554432
    .venv-gpu\\Scripts\\python.exe local_sim/vault.py verify
    uv run python local_sim/vault.py sacrificial --truth <fixture.npz>
    .venv-gpu\\Scripts\\python.exe local_sim/vault.py evaluate --label "v2.4.1 baseline" --reason seal
"""

import argparse
import contextlib
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

import official_config  # noqa: E402
from mint_provenance import canonical_mint, seed_streams  # noqa: E402
from seed_blocks import VAULT_BLOCK  # noqa: E402

VAULT_DIR = os.path.join(_ROOT, "experiments", "vault")
ARRAYS = os.path.join(VAULT_DIR, "vault_v1.npz")
MANIFEST = os.path.join(VAULT_DIR, "manifest.json")
TOUCHLOG = os.path.join(VAULT_DIR, "touch_log.jsonl")

# The August 6 seal evaluation predates the shared truth-read register.  It is
# preserved in manifest.json/touch_log.jsonl and counts against the same
# three-look cap. Every later reservation is written by truth_reads before the
# arrays are opened, so this fixed adoption baseline must not be replaced by
# the live manifest counter (doing that would double-count registered reads).
VAULT_HISTORICAL_READS_OUTSIDE_REGISTER = 1

#: THE ROUND THE SEALED CORPUS WAS MINTED UNDER, named rather than implied.
#:
#: Until 2026-09-10 this file carried `WIDTH, DEPTH = 256, 32` and a
#: `budget=272_000_000_000` default with nothing recording which rulebook they
#: belonged to. They are Phase 1's. Phase 2 moved width, depth, budget, lambda,
#: the wall limit and the residual limit ALL AT ONCE and added a constraint that
#: did not previously exist, so an evaluation that inherited some of those and
#: kept the transcribed rest would have been scored under a rulebook the
#: competition has never used -- and would have printed a clean objective while
#: doing it. `official_config` is the single authority for all six; nothing here
#: transcribes them.
VAULT_ROUND_TAG = "v1-phase1"
VAULT_ROUND = official_config.by_tag(VAULT_ROUND_TAG)
WIDTH, DEPTH = VAULT_ROUND.width, VAULT_ROUND.depth
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


class RoundMismatch(SystemExit):
    """A corpus, a shape and a rulebook that do not describe one experiment.

    A `SystemExit` SUBCLASS, so a command-line run still dies with the message
    printed and nothing swallows it by accident; a NAMED class, so a regression
    test can prove a refusal fired for THIS reason rather than because something
    else happened to raise.
    """


def build_spec(round_tag, *, width, depth, budget=None, seed=0,
               sacrificial_shape=False):
    """The `LifecycleSpec` for an explicitly NAMED round, or a refusal.

    Every constant comes from `official_config` through
    `LifecycleSpec.for_round`, which hands over a round's fields TOGETHER --
    width, depth, budget, lambda, wall limit, residual cliff, memory cap,
    multiplier floor. Picking them up one at a time is how a run ends up with
    Phase 2's lambda and Phase 1's budget, which is a rulebook that has never
    existed.

    `budget` is an explicit DEVIATION here, not a default. When it is passed the
    spec's `deviations_from_round()` names it, and that dict travels in the
    provenance blob of every run this file produces.

    `sacrificial_shape` is the ONLY route to a shape the named round does not
    have, and it exists solely for synthetic fixtures: a four-network 16x3
    fixture is not Phase 1, and quietly calling it Phase 1 is exactly what this
    function refuses. The deviation is recorded rather than hidden, so a report
    built on a fixture can never read as a report built on the round.
    """
    from lifecycle_eval import LifecycleSpec

    try:
        r = official_config.by_tag(str(round_tag))
    except KeyError as exc:
        raise RoundMismatch(
            f"REFUSED: {round_tag!r} is not a round this repository knows "
            f"({sorted(official_config.ROUNDS)}). An evaluation has to name the "
            f"rulebook it runs under; there is deliberately no default."
        ) from exc

    overrides = {}
    if (int(width), int(depth)) != (r.width, r.depth):
        if not sacrificial_shape:
            raise RoundMismatch(
                f"REFUSED: {int(width)}x{int(depth)} networks under round "
                f"{r.tag!r}, which is {r.width}x{r.depth}. That pairing is not a "
                f"round; it is a shape from one experiment and a budget, a lambda "
                f"and a residual limit from another -- and the estimator is "
                f"budget-adaptive, so it would not even do the same work. Name "
                f"the round the arrays were actually minted under."
            )
        overrides["width"] = int(width)
        overrides["depth"] = int(depth)
    if budget is not None:
        overrides["flop_budget"] = int(budget)
    return LifecycleSpec.for_round(r, seed=seed, **overrides)


def _refuse_unless_corpus_matches_live_round(man):
    """The sealed corpus must be the LIVE round's shape, or nothing is evaluated.

    Checked against the MANIFEST, and deliberately BEFORE `load()` opens the
    arrays: reading the truth IS the touch, so a refusal that happens after the
    read is not a refusal.

    The stored corpus is 500 networks at 256x32, minted 2026-08-06 under Phase 1.
    The live round is Phase 2 at 1024x16, with a different budget, lambda 0 and a
    0.4 s residual cliff. There is no honest way to evaluate one under the other:

      * under the PHASE-1 rulebook the arithmetic is fine, but the result is a
        Phase-1 measurement of a round that is over, and it would still spend one
        of three touches that will ever exist;
      * under the PHASE-2 rulebook the shape is wrong. `predict(mlp, budget)`
        sizes its work to the budget it is handed, so 256x32 networks under
        2**41 do several times the work they were minted for. That is a
        measurement of a rulebook that has never existed.

    So the sealed path refuses until a PHASE-2 sealed corpus exists. Building one
    is a STRATEGIC slice -- a pre-registered contract in `experiments/LOG.md`,
    Samuel's prediction recorded before the result, and an explicit authorised
    touch plan (`AGENTS.md` section *Work classification*, `docs/VALIDATION.md`
    section 4). No agent may create it as a side effect of repairing an
    evaluator, and this function is not where that decision gets made.

    The Phase-1 seal is preserved exactly as it stands: it is historical evidence
    and stays readable. `verify`, `readme` and `shipped-check` are untouched by
    this refusal, because none of them reads the truth.
    """
    live = official_config.LIVE
    w, dp = int(man.get("width", -1)), int(man.get("depth", -1))
    if (w, dp) == (live.width, live.depth):
        return
    nl = chr(10)
    raise RoundMismatch(
        f"REFUSED: the sealed corpus is {man.get('n_mlps', '?')} networks at "
        f"{w}x{dp} (minted {man.get('created_utc', '?')}, seed block "
        f"{man.get('seed_block', '?')}, round {VAULT_ROUND_TAG}). The LIVE round "
        f"is {live.tag} at {live.width}x{live.depth}." + nl +
        f"  Evaluating these arrays under {live.tag}'s budget and lambda would "
        f"report a rulebook that has never existed; evaluating them under "
        f"{VAULT_ROUND_TAG} would spend one of {MAX_TOUCHES} permitted touches on "
        f"a round that is over." + nl +
        f"  A Phase-2 sealed corpus is a STRATEGIC slice: a pre-registered "
        f"contract in experiments/LOG.md, Samuel's prediction recorded before the "
        f"result, and an authorised touch plan. It does not exist yet, and no "
        f"agent may create it while repairing an evaluator." + nl +
        f"  The Phase-1 seal is untouched and stays readable -- `vault.py verify`, "
        f"`vault.py readme`. To exercise this code path, use `vault.py "
        f"sacrificial --truth <fixture.npz>`, which never opens the vault's "
        f"arrays."
    )


@contextlib.contextmanager
def _reserve_vault_truth(
    *,
    label,
    reason,
    candidate_sha256,
    artifact=ARRAYS,
    reads_path=None,
    ledger_path=None,
):
    """Reserve a future vault look before exposing any label bytes.

    `truth_reads.reserve` writes the reservation under the seed-ledger lock.
    A process crash can therefore leave a `reserved` line, but that line still
    consumes the cap; concurrent evaluators cannot both receive the final look.
    The optional paths exist solely for sacrificial regression fixtures.
    """
    import truth_reads

    kw = {
        "consumer": "local_sim/vault.py",
        "purpose": f"vault {reason}: {label}",
        "instrument": os.path.abspath(__file__),
        "note": ("pre-read reservation; candidate sha256 "
                 + str(candidate_sha256)),
        "prior_spent": VAULT_HISTORICAL_READS_OUTSIDE_REGISTER,
    }
    if reads_path is not None:
        kw["path"] = reads_path
    if ledger_path is not None:
        kw["ledger_path"] = ledger_path
    with truth_reads.reserve("V", artifact, **kw) as resolved:
        yield resolved


def _run_candidate(candidate, seeds, truth_final, spec, gt_noise,
                   weights_mode=None, progress_every=50):
    """Run a candidate FILE over `seeds` through THE canonical lifecycle evaluator.

    This used to be a second prediction loop, and it got the lifecycle wrong in
    four separate ways that all leaned the same direction -- towards a number
    kinder than the grader's:

      * **bare NumPy weights.** `np.asarray(x)` gave writeable float32. The
        grader hands over float64 flopscope arrays that are READ-ONLY, and an
        operation invoked as a METHOD ON A WEIGHT is metered there and free on
        bare NumPy. The 2026-09-05 dtype incident cost seven submission slots to
        precisely this gap: an `x.dtype == dtype` guard short-circuited on every
        local weight producer, so the branch the grader takes was never once
        exercised offline.
      * **a fresh `Estimator()` per network, `setup()` never called.** The grader
        builds ONE instance and calls `setup()` once, OFF-budget, so everything
        setup does is free. Anything a candidate builds in INSTANCE state was
        billed 500 times here and zero times there -- which is the entire cost
        question for a carrier candidate.
      * **no output materialisation.** whestbench 0.16 runs
        `materialise_predictions` INSIDE the metered window; on a float64 return
        at 1024x16 that dtype conversion is 2 FLOPs/element = 32,768 counted
        FLOPs the grader bills and this file did not.
      * **no failure paths at all.** An exception inside `predict` propagated and
        killed the whole run. The grader zeroes that MLP, prices it at multiplier
        1.0, and carries on with the rest.

    `local_sim/lifecycle_eval.py` is the ONE place all of that lives -- 56/56 in
    its own suite, and anchored op-for-op against a real official-package
    subprocess run. Two implementations of one lifecycle is how a repository ends
    up with two answers, so this is now a call, not a copy.

    Returns `(rows, provenance, predictions)`. `predictions` are the FINAL-LAYER
    arrays as they were actually SCORED, so a failed MLP contributes the zeros
    the grader scored rather than whatever the candidate returned before it was
    disqualified.
    """
    import lifecycle_eval

    preds = []
    rows, prov = lifecycle_eval.evaluate(
        candidate,
        [int(x) for x in seeds],
        truth_final,
        spec=spec,
        gt_noise=gt_noise,
        progress_every=progress_every,
        weights_mode=weights_mode or lifecycle_eval.DEFAULT_WEIGHTS_MODE,
        predictions_out=preds,
    )
    return rows, prov, np.asarray(preds, dtype=np.float64)


def _score_block(rows, prov, spec, extra):
    """Everything the vault reports, through the canonical scorer.

    `on_over_budget` is no longer `"fail"`. That mode is labelled optimistic
    inside `competition_score.score_suite` itself, and the label is exact: it
    forces the failure MULTIPLIER while RETAINING the candidate's own MSE,
    because a scorer handed nothing but rows cannot zero a prediction it never
    saw. A real failed MLP is scored against ZEROED predictions, so its MSE is
    `mean(target**2)` -- typically orders of magnitude larger than the
    candidate's own. The old path could therefore report a survivable score for a
    submission the grader would have flattened.

    The rows arriving here come from `lifecycle_eval.evaluate`, which zeroes a
    prediction where the grader zeroes it and sets `failed` where the grader sets
    it. The failure is already IN the row, so `"raise"` is the correct setting,
    and it firing would mean the evaluator handed over a row it had not
    classified.

    `lifecycle_eval.score` owns the call, so the score identity still lives in
    exactly one module.
    """
    from competition_score import effective_compute
    from lifecycle_eval import score as lifecycle_score

    suite = lifecycle_score(rows, spec=spec)
    a, m = suite.scores, suite.mses
    fl = np.asarray([r.flops_used for r in rows], dtype=np.float64)
    rs = np.asarray([r.residual_wall_time_s for r in rows], dtype=np.float64)
    gt = np.asarray([r.gt_noise for r in rows], dtype=np.float64)
    ops = np.asarray([r.n_ops for r in rows], dtype=np.float64)
    gt_mean = float(np.nanmean(gt)) if np.any(np.isfinite(gt)) else float("nan")
    local_c = np.asarray([effective_compute(f, r, spec.lam) for f, r in zip(fl, rs)])
    out = {
        "n_mlps": int(suite.n),
        "nonfinite": int(sum(1 for r in rows if r.nonfinite)),
        # PRIMARY. Accuracy is what the vault can actually measure.
        "mse_mean": float(m.mean()),
        "mse_median": _quantile(m, 0.5),
        "gt_noise_mean": gt_mean,
        "gt_noise_share_of_mse": (gt_mean / float(m.mean())
                                  if m.mean() else float("nan")),
        # THE GRADER'S FAILURE ACCOUNTING, itemised. A single `n_failed` hides
        # which cliff fired, and on this host one of them -- residual -- is an E1
        # instrument that reads several times pessimistic, so a residual failure
        # here is a statement about this laptop and a budget failure is not.
        "n_failed": int(suite.n_failed),
        "failure_paths": {
            "budget_exhausted": int(sum(1 for r in rows if r.budget_exhausted)),
            "time_exhausted": int(sum(1 for r in rows if r.time_exhausted)),
            "residual_exhausted": int(sum(1 for r in rows if r.residual_exhausted)),
            "combined_exhausted": int(sum(1 for r in rows if r.combined_exhausted)),
            "predict_error": int(sum(1 for r in rows if r.error_code)),
        },
        # SECONDARY, and model-labelled.
        "cost_model": (f"C = F + lam*R at lam={spec.lam:g} ({spec.round_tag}); "
                       f"R is this HOST's in-process residual, classified E1 in "
                       f"docs/CURRENT_STATE.md"),
        "objective": float(suite.objective),
        "median": _quantile(a, 0.5),
        "mean": float(a.mean()),
        "p90": _quantile(a, 0.90),
        "p99": _quantile(a, 0.99),
        "max": float(a.max()),
        "cb_mean": float(suite.multipliers.mean()),
        "counted_flops_mean": float(np.mean(fl)),
        # Reported so the residual's status is visible rather than silent.
        "local_residual_s_median": float(np.median(rs)),
        "local_effective_compute_mean_INFORMATIONAL": float(local_c.mean()),
        "n_ops_worst": (float(np.nanmax(ops)) if np.any(np.isfinite(ops))
                        else float("nan")),
        # The rate this block ACTUALLY used, not a constant that happens to be in
        # scope. Until 2026-08-24 this recorded `LAMBDA_FLOPS_PER_SECOND` while
        # `effective_compute` inherited the module default -- and P2-P0 moved
        # those apart. It now comes from the SPEC, which comes from the named
        # round, so the three cannot drift from one another at all.
        "lambda": float(spec.lam),
        # THE RULEBOOK, and every field of it this run did not honour. An empty
        # `round_deviations` means a clean round; anything in it means a number
        # from this run must say so wherever it is quoted.
        "round_tag": spec.round_tag,
        "round_deviations": prov.get("round_deviations", {}),
        # THE LIFECYCLE, recorded rather than assumed. `setup_called` false, or
        # `lifecycle` != "grader", would mean the defect this repair removed had
        # come back.
        "lifecycle": prov.get("lifecycle"),
        "setup_called": prov.get("setup_called"),
        "setup_seconds": prov.get("setup_seconds"),
        "weights_mode": prov.get("weights_mode"),
        "weights_mode_is_grader_default": prov.get("weights_mode_is_grader_default"),
        "rows_hash": prov.get("rows_hash"),
        "truth_sha256": prov.get("truth_sha256"),
        "residual_is_graded": False,
        "objective_is_graded": False,
    }
    out.update(extra)
    return out


def _load_sacrificial(path, round_tag):
    """A SACRIFICIAL fixture, and the round it declares. Never the vault.

    A fixture has to say what it is. The old format stored `seeds`, `truth` and
    `sigma_gt2` and nothing else, so its shape came from this module's Phase-1
    literals and its rulebook came from a default argument -- the same silence
    the round refusals above exist to remove, just on the test path. `width`,
    `depth` and `round_tag` are therefore REQUIRED fields, and a fixture without
    them is refused rather than guessed at.
    """
    d = np.load(path)
    missing = [k for k in ("seeds", "truth", "sigma_gt2", "width", "depth",
                           "round_tag") if k not in d]
    if missing:
        raise SystemExit(
            f"REFUSED: {path} is missing {missing}. A sacrificial fixture must "
            f"declare its own shape and the round it stands in for; inferring "
            f"those from this module's constants is how a fixture silently "
            f"becomes evidence about a different experiment."
        )
    fixture_round = str(np.asarray(d["round_tag"]).item())
    if round_tag is not None and str(round_tag) != fixture_round:
        raise RoundMismatch(
            f"REFUSED: {path} declares round {fixture_round!r} and you asked for "
            f"{str(round_tag)!r}. The fixture's truth was produced under one "
            f"rulebook; scoring it under another is not a run of that other "
            f"round, it is a mislabelled run of this one."
        )
    return (d["seeds"], np.asarray(d["truth"]), np.asarray(d["sigma_gt2"]),
            int(np.asarray(d["width"]).item()),
            int(np.asarray(d["depth"]).item()), fixture_round)


def evaluate(label, reason, candidate="estimator.py", budget=None,
             limit=0, sacrificial="", round_tag=None, weights_mode=None,
             evidence_dir=None, progress_every=50):
    """Evaluate an explicitly named candidate FILE. Increments the touch counter.

    THE ROUND IS NAMED, NOT INHERITED. `budget` was `272_000_000_000` here until
    2026-09-10 -- Phase 1's number, sitting in a default argument with nothing
    saying so. It is now `None` by default and means "whatever the named round
    says"; passing it is a declared DEVIATION that travels in the result's
    `round_deviations`.

    `sacrificial` is reached ONLY through the separate `sacrificial` subcommand,
    never through `evaluate`. Both agent hooks and the git pre-commit backstop
    guard the string "vault.py evaluate" (`scripts/guard_paths.py`), so the test
    path deliberately does not wear that name -- and, more importantly, it never
    opens the vault's arrays at all.

    The prediction arrays used to be written to `candidate_preds.npz`, which was
    OVERWRITTEN on every touch (`experiments/LOG.md`, defect 6 of the 2026-08-19
    audit). A once-ever touch is now recorded as an append-only
    `lifecycle_eval.write_evidence` bundle -- per-MLP rows, failure flags,
    counted FLOPs, the round and its deviations -- beside an `_preds.npz` of the
    final-layer arrays as they were SCORED. `write_evidence` refuses to
    overwrite, so a second run cannot quietly replace the first one's record.
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
        seeds, truth_final, gt_noise, f_w, f_d, f_round = _load_sacrificial(
            sp, round_tag)
        # A synthetic fixture is allowed a shape no round has -- and only here.
        # The deviation is recorded in `round_deviations`, so a fixture's numbers
        # can never read as the round's. When the fixture IS the round's shape
        # this flag adds nothing.
        spec = build_spec(f_round, width=f_w, depth=f_d, budget=budget,
                          sacrificial_shape=True)
        man = {"touches": None}
        out_dir = evidence_dir or os.path.dirname(os.path.abspath(sp))
        print(f"SACRIFICIAL RUN on {sacrificial} — {len(seeds)} NON-VAULT networks.")
        print("Nothing is recorded and no touch is taken.\n", flush=True)
    else:
        # THE MANIFEST FIRST, and the refusal before it. `load()` opens the truth,
        # and opening the truth is the touch; a refusal that happens after the
        # read is not a refusal.
        man = json.load(open(MANIFEST))
        _refuse_unless_corpus_matches_live_round(man)
        live = official_config.LIVE
        if round_tag is not None and str(round_tag) != live.tag:
            raise RoundMismatch(
                f"REFUSED: the sealed path evaluates under the LIVE round "
                f"({live.tag}) and nothing else. You asked for {round_tag!r}. A "
                f"sealed corpus is spent once; it is not a place to replay an old "
                f"rulebook."
            )
        # The cap is checked before the reservation, and the reservation is
        # written before `load()` opens the arrays.  The shared register also
        # knows about crashed and concurrent attempts; manifest.json alone did
        # not, because the old path updated it only after a successful run.
        if man["touches"] >= MAX_TOUCHES:
            raise SystemExit(
                f"REFUSED: the vault has already been touched {man['touches']} times "
                f"and the maximum is {MAX_TOUCHES} (seal, one estimator change, "
                f"designation). Re-read experiments/vault/README.md before doing "
                f"anything that would need a fourth."
            )
        spec = build_spec(live.tag, width=int(man["width"]),
                          depth=int(man["depth"]), budget=budget)
        with _reserve_vault_truth(
                label=label, reason=reason, candidate_sha256=cand_sha):
            d, man = load(check=True)
        seeds, truth_final, gt_noise = (d["seeds"], d["truth"][:, -1, :],
                                        d["gt_noise"])
        out_dir = evidence_dir or VAULT_DIR
    if limit:
        seeds, truth_final, gt_noise = (seeds[:limit], truth_final[:limit],
                                        gt_noise[:limit])

    dev = spec.deviations_from_round()
    print(f"VAULT EVALUATION — {label}   ({len(seeds)} MLPs, reason={reason})")
    print(f"  candidate  {candidate}")
    print(f"  sha256     {cand_sha}")
    print(f"  round      {spec.round_tag}   {spec.width}x{spec.depth}   "
          f"B {spec.flop_budget:,}   lambda {spec.lam:g}")
    print(f"  deviations {dev or 'none — a clean round'}\n", flush=True)

    rows, prov, preds = _run_candidate(cand_path, seeds, truth_final, spec,
                                       gt_noise, weights_mode=weights_mode,
                                       progress_every=progress_every)

    res = _score_block(
        rows, prov, spec,
        {
            "touch": (man["touches"] + 1) if man["touches"] is not None else None,
            "utc": _now(),
            "label": label,
            "reason": reason,
            "candidate": candidate,
            "candidate_sha256": cand_sha,
            "budget": int(spec.flop_budget),
            "sacrificial": sacrificial or None,
        },
    )

    # Append-only evidence, named by WHEN and by the content hash of the rows, so
    # two runs cannot collide and neither can overwrite the other.
    import lifecycle_eval

    stem = ("sacrificial" if sacrificial else f"touch{res['touch']}")
    stamp = _now().replace("-", "").replace(":", "")
    base = os.path.join(out_dir,
                        f"lifecycle_evidence_{stem}_{stamp}_{prov['rows_hash'][:8]}")
    lifecycle_eval.write_evidence(base + ".json", rows, prov,
                                  lifecycle_eval.score(rows, spec=spec))
    np.savez(base + "_preds.npz", seeds=np.asarray(seeds),
             preds=preds.astype(np.float64))

    print("\n" + "=" * 88)
    print(f"VAULT RESULT — {label}")
    print("=" * 88)
    print(f"  PRIMARY (accuracy, what this instrument can measure)")
    print(f"    raw MSE   mean {res['mse_mean']:.4e}   median {res['mse_median']:.4e}")
    print(f"    GT noise  {res['gt_noise_mean']:.3e} = "
          f"{res['gt_noise_share_of_mse'] * 100:.2f}% of measured MSE  "
          f"<- any claim smaller than this is noise")
    print(f"  LIFECYCLE (the grader's, through local_sim/lifecycle_eval.py)")
    print(f"    setup called {res['setup_called']} once in "
          f"{(res['setup_seconds'] or 0.0) * 1e3:.1f} ms   weights "
          f"{res['weights_mode']}"
          + ("  (GRADER TWIN)" if res["weights_mode_is_grader_default"]
             else "  <- NOT the grader's dtype; a control, not evidence"))
    print(f"    failures  {res['n_failed']}/{res['n_mlps']}   "
          f"{res['failure_paths']}")
    print(f"  SECONDARY (score, MODEL-DEPENDENT: {res['cost_model']})")
    print(f"    adjusted  median {res['median']:.4e}   mean {res['mean']:.4e}   "
          f"p90 {res['p90']:.4e}")
    # `max/med` is a dispersion diagnostic, and a suite whose median score is
    # exactly zero is not a bug -- a probe that returns its own truth reaches it.
    # Printing "n/a" beats dying in the reporter after the run is already done.
    _ratio = (f"{res['max'] / res['median']:.2f}x" if res['median'] else "n/a")
    print(f"              p99    {res['p99']:.4e}   max  {res['max']:.4e}   "
          f"max/med {_ratio}")
    print(f"    C/B mean  {res['cb_mean']:.4f}   counted "
          f"{res['counted_flops_mean']:.4e}   non-finite {res['nonfinite']}")
    print(f"    local residual {res['local_residual_s_median'] * 1e3:.2f} ms/MLP "
          f"— an E1 RATIO instrument, never deployment headroom")
    print(f"    rows_hash {res['rows_hash'][:32]}...")
    print(f"    evidence  {os.path.relpath(base + '.json', _ROOT)}")

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

    # The vault's OWN round, not a transcribed literal. This path is a parity
    # comparison against a stored mirror, so it stays on the shim MLP and float32
    # weights it was built with -- which is a real scope limit, not a claim: it
    # says the two implementations agree, and says nothing about what the grader's
    # float64 read-only arrays would do to either of them.
    class Ctx:
        width, depth, flop_budget, seed = (
            WIDTH, DEPTH, VAULT_ROUND.flop_budget, 0)

    est.setup(Ctx())
    from whestbench import MLP  # the local shim

    n = min(limit, len(seeds))
    absd, rel = [], []
    for i in range(n):
        w, est_seed, _ = canonical_mint(WIDTH, DEPTH, int(seeds[i]))
        mlp = MLP(width=WIDTH, depth=DEPTH, weights=w, seed=est_seed)
        p = np.asarray(est.predict(mlp, VAULT_ROUND.flop_budget),
                       dtype=np.float64)[-1]
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
    # NO DEFAULT BUDGET. It was 272,000,000,000 -- Phase 1's -- sitting in a
    # default argument with nothing saying which round it belonged to. The round
    # supplies it now, and passing this is a DECLARED deviation that shows up in
    # `round_deviations` on the result.
    e.add_argument("--budget", type=int, default=0,
                   help="DEVIATE from the named round's own FLOP budget. Recorded "
                        "as a deviation; omit it for a clean round.")
    e.add_argument("--round", dest="round_tag", default=None,
                   help="the round to evaluate under. The sealed path accepts "
                        "only the LIVE round; naming another is refused rather "
                        "than quietly honoured.")
    e.add_argument("--weights-mode", default=None,
                   help="lifecycle_eval.WEIGHTS_MODES. Defaults to f64_readonly, "
                        "which is what the grading server actually hands over; "
                        "anything else is a control, not evidence.")
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
                     help="npz with seeds / truth (M,width) / sigma_gt2 / width / "
                          "depth / round_tag. The last three are REQUIRED: a "
                          "fixture that does not declare its own shape and "
                          "rulebook cannot be evaluated without guessing.")
    sac.add_argument("--candidate", default="estimator.py")
    sac.add_argument("--budget", type=int, default=0,
                     help="DEVIATE from the fixture's round's FLOP budget.")
    sac.add_argument("--round", dest="round_tag", default=None,
                     help="assert the fixture's declared round. A mismatch is "
                          "refused.")
    sac.add_argument("--weights-mode", default=None)
    sac.add_argument("--evidence-dir", default=None,
                     help="where the append-only evidence bundle goes. Defaults "
                          "to the fixture's own directory, NEVER the vault.")
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
        evaluate("sacrificial", "seal", a.candidate, a.budget or None, a.limit,
                 a.truth, round_tag=a.round_tag, weights_mode=a.weights_mode,
                 evidence_dir=a.evidence_dir)
    else:
        evaluate(a.label, a.reason, a.candidate, a.budget or None, a.limit,
                 round_tag=a.round_tag, weights_mode=a.weights_mode)
        write_readme()


if __name__ == "__main__":
    main()
