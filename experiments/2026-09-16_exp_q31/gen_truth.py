"""He-init bias-free ReLU MLPs (float32 weights, x@W convention) + MC truth in float64."""
import sys, time, json
import numpy as np

def build_mlp(n, L, seed):
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, np.sqrt(2.0 / n), size=(n, n)).astype(np.float32) for _ in range(L)]

def mc_truth(ws, N, chunk=250_000, seed=12345):
    n = ws[0].shape[0]; L = len(ws)
    rng = np.random.default_rng(seed)
    acc = np.zeros((L, n)); acc2 = np.zeros((L, n)); done = 0
    w64 = [w.astype(np.float64) for w in ws]
    while done < N:
        m = min(chunk, N - done)
        h = rng.standard_normal((m, n))
        for l, w in enumerate(w64):
            h = np.maximum(h @ w, 0.0)
            acc[l] += h.sum(0); acc2[l] += (h * h).sum(0)
        done += m
    mean = acc / N
    var = acc2 / N - mean ** 2
    return mean, var / N  # per-neuron mean and its estimator variance (noise floor)

if __name__ == "__main__":
    n = int(sys.argv[1]); L = int(sys.argv[2]); seeds = range(int(sys.argv[3])); N = int(float(sys.argv[4]))
    out = {}
    for s in seeds:
        t0 = time.time()
        ws = build_mlp(n, L, s)
        mean, nv = mc_truth(ws, N)
        np.savez(f"mlp_n{n}_L{L}_s{s}.npz", weights=np.stack(ws), truth=mean, noise_var=nv)
        print(f"seed {s}: {time.time()-t0:.1f}s  final-layer mean^2={np.mean(mean[-1]**2):.4e}  noise floor (mean var/N) final={nv[-1].mean():.3e}", flush=True)
