"""P2-CP3R: how much of the closure's error can ANY cumulant correction reach?

P2-CP1 established that ~95% of the Gaussian closure's error is its Gaussian
assumption, and that improving the covariance map is capped at 1.052x. Both
remaining research checkpoints -- P2-CP3 (Edgeworth) and P2-CP4 (goal-directed
K3) -- assume a cumulant correction can *reach* that 95%. **Nothing had measured
whether it can.** This module measures the ceiling before any production spelling
is built, exactly as Moonshot G0's oracle did at Phase 1.

## The identity being oracled

For a pre-activation `z` with mean `mu`, sd `sigma`, `alpha = mu/sigma`, third
central moment `k3` and excess fourth cumulant `k4`, the Edgeworth-corrected ReLU
mean is

    E[ReLU(z)] = mu*Phi(a) + sigma*phi(a)
                 - (k3 / (6 sigma^2))  * a         * phi(a)      <- 3rd cumulant
                 + (k4 / (24 sigma^3)) * (a^2 - 1) * phi(a)      <- 4th cumulant

Both correction terms were derived by integrating the Edgeworth density against
`(u+alpha)^+` using `int_{-a}^inf He_n(u) phi(u) du = He_{n-1}(a) phi(a)`, and
both were verified against Monte Carlo on skewed gammas and kurtotic t's before
this module was written.

**Two structural facts drive the design.** The third-cumulant term carries the
factor `alpha`, so it **vanishes identically wherever the pre-activation is
zero-mean** -- which at layer 1 is every neuron, exactly. The fourth-cumulant
term carries `(alpha^2 - 1)` and is at its *maximum* there. Measured on a real
network, `|a phi(a)|` averages 0.07-0.16 across depth while `|(a^2-1) phi(a)|`
averages 0.09-0.40. **A K3-only correction is the wrong shape for this problem**,
which is why this oracle prices the two terms separately.

## The ladder

    O0  closure's own (mu, sigma)   Gaussian map      == arm_k2p1, honest
    O1  TRUE (mu, sigma)            Gaussian map      isolates STATE error
    O2  TRUE (mu, sigma, k3)        + 3rd term
    O3  TRUE (mu, sigma, k3, k4)    + 3rd + 4th term

`O1 - O0` is the state error; `O2 - O1` and `O3 - O2` are what the cumulant terms
buy on top of a perfect state.

**None of these is deployable.** They all consume Monte-Carlo truth. That is the
point: they bound what any spelling that must *estimate* the cumulants can ever
achieve. A ceiling measured with an oracle is worth more than a floor measured
with a guess.

**The oracle's cumulants and the truth come from INDEPENDENT sample streams**, and
that is not a detail. The first build of this module accumulated both from the
same draw, and O3 then scored an MSE of 3.6e-10 against a truth-noise floor of
7.2e-8 -- a prediction cannot be closer to a Monte-Carlo mean than that mean's own
noise, and the tell was exactly that impossibility. Feeding an Edgeworth formula
the *sample* moments of the same draw that produced the target reproduces the
target's noise, in-sample, and reports it as accuracy. Two streams spawned from
the evaluator's own sampling substream fix it: the cumulants come from stream A,
the truth from stream B, and `E[e_A e_B] = 0` by construction.

Run (GPU venv; CLAIMS SEEDS, only inside the registered contract):
    .venv-gpu\\Scripts\\python.exe local_sim/p2_cumulant_oracle.py \\
        --block P01 --n-mlps 48 --log2-samples 22 --ts 2026-08-21 \\
        --out experiments/2026-08-21_p2_cp3r/oracle.json

Smoke it on unregistered probe seeds, claiming and writing nothing:
    .venv-gpu\\Scripts\\python.exe local_sim/p2_cumulant_oracle.py --smoke 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

from mint_provenance import canonical_mint, seed_streams  # noqa: E402

WIDTH, DEPTH = 1024, 16
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _Phi(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _phi(x):
    return np.exp(-0.5 * x * x) / _SQRT2PI


def _relu_mean(mu, sig, k3=None, k4=None):
    """Edgeworth-corrected E[ReLU(z)]. `k3`/`k4` None -> that term is omitted."""
    a = mu / sig
    P, p = _Phi(a), _phi(a)
    out = mu * P + sig * p
    if k3 is not None:
        out = out - (k3 / (6.0 * sig * sig)) * a * p
    if k4 is not None:
        out = out + (k4 / (24.0 * sig ** 3)) * (a * a - 1.0) * p
    return out


def _closure_state(w):
    """The honest closure's (mu, sigma) for every layer's PRE-activation.

    Exactly `arm_k2p1`'s recursion, in plain numpy: this is O0's state and it uses
    no truth of any kind.
    """
    n = w[0].shape[0]
    mu = np.zeros(n)
    cov = np.eye(n)
    off = ~np.eye(n, dtype=bool)
    out = []
    for wk in w:
        mu_pre = wk.T @ mu
        cov_pre = wk.T @ cov @ wk
        vp = np.maximum(np.diag(cov_pre), 1e-12)
        sig = np.sqrt(vp)
        out.append((mu_pre.copy(), sig.copy()))
        a = mu_pre / sig
        P, p = _Phi(a), _phi(a)
        mu = mu_pre * P + sig * p
        ez2 = (mu_pre * mu_pre + vp) * P + mu_pre * sig * p
        var = np.maximum(ez2 - mu * mu, 0.0)
        cov = np.outer(P, P) * cov_pre * off + np.diag(var)
    return out


def _true_moments(weights, seed64, n_samples, device, chunk):
    """Per-neuron central moments 1-4 of every layer's PRE-activation, on the GPU.

    Accumulated in float64. Also returns the post-ReLU mean at every layer.

    **Call this TWICE with independent seeds.** Using one call's moments to
    predict the same call's post-ReLU mean is an in-sample fit that reports the
    draw's own noise as accuracy -- see the module docstring.
    """
    import torch

    w = [torch.from_numpy(np.ascontiguousarray(x)).to(device) for x in weights]
    g = torch.Generator(device=device)
    g.manual_seed(seed64 % (2 ** 63 - 1))
    D, W = len(w), w[0].shape[0]
    s1 = [torch.zeros(W, dtype=torch.float64, device=device) for _ in range(D)]
    s2 = [torch.zeros(W, dtype=torch.float64, device=device) for _ in range(D)]
    s3 = [torch.zeros(W, dtype=torch.float64, device=device) for _ in range(D)]
    s4 = [torch.zeros(W, dtype=torch.float64, device=device) for _ in range(D)]
    r1 = [torch.zeros(W, dtype=torch.float64, device=device) for _ in range(D)]
    done = 0
    while done < n_samples:
        m = min(chunk, n_samples - done)
        h = torch.randn((m, W), generator=g, device=device, dtype=torch.float32)
        for k in range(D):
            z = h @ w[k]
            zd = z.to(torch.float64)
            s1[k] += zd.sum(dim=0)
            s2[k] += (zd * zd).sum(dim=0)
            s3[k] += (zd ** 3).sum(dim=0)
            s4[k] += (zd ** 4).sum(dim=0)
            h = torch.relu(z)
            r1[k] += h.to(torch.float64).sum(dim=0)
        done += m
    N = float(n_samples)
    mu, sig, k3, k4, post = [], [], [], [], []
    for k in range(DEPTH if len(w) == DEPTH else len(w)):
        m1 = (s1[k] / N)
        m2 = (s2[k] / N)
        m3 = (s3[k] / N)
        m4 = (s4[k] / N)
        c2 = m2 - m1 ** 2
        c3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
        c4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4 - 3 * c2 ** 2
        mu.append(m1.cpu().numpy())
        sig.append(torch.sqrt(torch.clamp(c2, min=1e-300)).cpu().numpy())
        k3.append(c3.cpu().numpy())
        k4.append(c4.cpu().numpy())
        post.append((r1[k] / N).cpu().numpy())
    return (np.stack(mu), np.stack(sig), np.stack(k3), np.stack(k4),
            np.stack(post))


def _arms(w, tm):
    """O0..O3 predicted post-ReLU means, per layer, from stream A's moments.

    The caller scores these against stream B's post-ReLU means. Mixing the two
    streams is the whole point; see the module docstring.
    """
    mu_t, sig_t, k3_t, k4_t, _post = tm
    st = _closure_state(w)
    o0 = np.stack([_relu_mean(m, s) for m, s in st])
    o1 = np.stack([_relu_mean(mu_t[k], sig_t[k]) for k in range(len(w))])
    o2 = np.stack([_relu_mean(mu_t[k], sig_t[k], k3=k3_t[k]) for k in range(len(w))])
    o3 = np.stack([_relu_mean(mu_t[k], sig_t[k], k3=k3_t[k], k4=k4_t[k])
                   for k in range(len(w))])
    return {"O0": o0, "O1": o1, "O2": o2, "O3": o3}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="")
    ap.add_argument("--n-mlps", type=int, default=0)
    ap.add_argument("--log2-samples", type=int, default=22)
    ap.add_argument("--out", default="")
    ap.add_argument("--ts", default="")
    ap.add_argument("--consumer", default="local_sim/p2_cumulant_oracle.py")
    ap.add_argument("--chunk", type=int, default=1 << 15)
    ap.add_argument("--start", type=int, default=0,
                    help="claim from this offset inside the block, to stay "
                         "DISJOINT from an earlier draw")
    ap.add_argument("--smoke", type=int, default=0,
                    help="run on UNREGISTERED probe seeds; claims nothing, "
                         "writes nothing, selects nothing")
    a = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_samples = 1 << a.log2_samples

    print("=" * 84)
    print("P2-CP3R CUMULANT ORACLE  --  the CEILING of the cumulant lane")
    print("=" * 84)
    print(f"  device      {device}"
          + (f"  {torch.cuda.get_device_name(0)}" if device == "cuda" else ""))
    print(f"  samples     2^{a.log2_samples} = {n_samples:,} per network")
    print(f"  arms        O0 honest closure | O1 true state | O2 +3rd | O3 +3rd+4th")
    print()

    if a.smoke:
        seeds = [1_000_003 + i for i in range(a.smoke)]   # seed-ledger: no block
        print("  SMOKE: unregistered probe seeds. Nothing claimed, nothing written.")
    else:
        if not (a.block and a.n_mlps and a.ts and a.out):
            raise SystemExit("--block --n-mlps --ts --out are required unless --smoke")
        if os.path.exists(a.out):
            raise SystemExit(f"REFUSED: {a.out} exists. Evidence is immutable.")
        import seed_blocks as sb
        import seed_ledger as sl

        lo, _n, _why = {b[0]: (b[1], b[2], b[3]) for b in sb.DECISION_BLOCKS}[a.block]
        start = lo + a.start
        seeds = sl.claim_range(
            a.block, start, a.n_mlps, a.consumer, ts=a.ts,
            artifact=os.path.relpath(a.out, _ROOT).replace("\\", "/"),
            note="P2-CP3R cumulant oracle; DISJOINT from the P2-CP1 draw")
        print(f"  claimed     {a.block}[{seeds[0]:,} .. {seeds[-1]:,}]  "
              f"({a.n_mlps} seeds)")
    print()

    t0 = time.perf_counter()
    rows = []
    for i, s in enumerate(seeds):
        w_np, _e, _ss = canonical_mint(WIDTH, DEPTH, s)
        w = [np.asarray(x, dtype=np.float32) for x in w_np]
        # TWO independent streams from the evaluator's own sampling substream:
        # A supplies the oracle's cumulants, B supplies the truth it is scored
        # against. Same-stream reuse is an in-sample fit (module docstring).
        _ws, sample_ss, _es = seed_streams(s)
        ca, cb = sample_ss.spawn(2)
        sd_a = int.from_bytes(ca.generate_state(2).tobytes(), "little")
        sd_b = int.from_bytes(cb.generate_state(2).tobytes(), "little")
        tm = _true_moments(w, sd_a, n_samples, device, a.chunk)      # stream A
        tb = _true_moments(w, sd_b, n_samples, device, a.chunk)      # stream B
        truth = tb[4]
        # The replica disagreement is the live check that the streams really are
        # independent -- the same check p2_truth.py runs.
        rep_ok = float(((tm[4][-1] - tb[4][-1]) ** 2).mean())
        w64 = [x.astype(np.float64) for x in w_np]
        arms = _arms(w64, tm)
        mse = {k: float(((v[-1] - truth[-1]) ** 2).mean()) for k, v in arms.items()}
        by_depth = {k: [float(((v[d] - truth[d]) ** 2).mean())
                        for d in range(DEPTH)] for k, v in arms.items()}
        # excess kurtosis of the true pre-activation, per layer -- Q5
        exk = [float((tm[3][d] / tm[1][d] ** 4).mean()) for d in range(DEPTH)]
        skw = [float((tm[2][d] / tm[1][d] ** 3).mean()) for d in range(DEPTH)]
        rows.append({"seed": int(s), "mse": mse, "by_depth": by_depth,
                     "excess_kurtosis": exk, "skewness": skw,
                     "replica_disagreement_final": rep_ok})
        if (i + 1) % 4 == 0 or i + 1 == len(seeds):
            el = time.perf_counter() - t0
            print(f"    {i + 1:>3}/{len(seeds)}  {el:7.1f}s  "
                  f"eta {el / (i + 1) * (len(seeds) - i - 1):7.1f}s   "
                  f"O0 {mse['O0']:.3e} -> O3 {mse['O3']:.3e}")

    m = {k: float(np.mean([r["mse"][k] for r in rows])) for k in ("O0", "O1", "O2", "O3")}
    rec = {k: 1.0 - m[k] / m["O0"] for k in ("O1", "O2", "O3")}
    print()
    print(f"  {'arm':<5}{'final-layer MSE':>18}{'MSE recovered vs O0':>22}")
    for k in ("O0", "O1", "O2", "O3"):
        r = "" if k == "O0" else f"{100 * rec[k]:>20.1f}%"
        print(f"  {k:<5}{m[k]:>18.4e}{r:>22}")
    print()
    print(f"  state error  (O0 -> O1)   {100 * rec['O1']:>6.1f}% of O0's MSE")
    print(f"  3rd cumulant (O1 -> O2)   {100 * (m['O1'] - m['O2']) / m['O0']:>6.1f}%")
    print(f"  4th cumulant (O2 -> O3)   {100 * (m['O2'] - m['O3']) / m['O0']:>6.1f}%")
    print()
    rep = float(np.mean([r["replica_disagreement_final"] for r in rows]))
    print(f"  stream A/B disagreement at the final layer  {rep:.4e}")
    print(f"  -> truth noise per stream ~ {rep / 2:.4e}. Any arm scoring BELOW")
    print("     this is fitting noise in-sample, not predicting.")
    print()
    exk = np.mean([r["excess_kurtosis"] for r in rows], axis=0)
    skw = np.mean([r["skewness"] for r in rows], axis=0)
    print("  true pre-activation shape by layer (mean over neurons and networks):")
    print(f"    {'layer':>6}{'skewness':>12}{'excess kurtosis':>18}")
    for d in (0, 1, 3, 7, 11, 15):
        print(f"    {d + 1:>6}{skw[d]:>12.4f}{exk[d]:>18.4f}")

    if a.out:
        art = {"artifact": "p2_cp3r_cumulant_oracle", "artifact_version": 1,
               "shape": {"width": WIDTH, "depth": DEPTH},
               "n_samples_per_network": n_samples, "n_networks": len(seeds),
               "block": a.block, "seeds": [int(x) for x in seeds],
               "mean_mse": m, "fraction_recovered_vs_O0": rec,
               "mean_skewness_by_layer": [float(x) for x in skw],
               "mean_excess_kurtosis_by_layer": [float(x) for x in exk],
               "rows": rows,
               "deployable": False,
               "note": "O1-O3 consume Monte-Carlo truth for the cumulants. They "
                       "are CEILINGS on what any deployable spelling can reach, "
                       "not estimators. O0 is the honest closure and uses no truth."}
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(art, indent=1) + "\n")
        import hashlib

        import seed_ledger as sl
        sl.finalize(a.consumer,
                    os.path.relpath(a.out, _ROOT).replace("\\", "/"), ts=a.ts,
                    note="oracle written")
        print(f"\n  artifact -> {os.path.relpath(a.out, _ROOT)}")
        print(f"  sha256      {hashlib.sha256(open(a.out,'rb').read()).hexdigest()}")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
