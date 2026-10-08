"""Lane M3 / M9 -- how much is fourth order worth?  ARC's own width sweep (R3 §5.2).

Copy into ARC's `scripts/` and run from there (output paths are cwd-relative):

    cd mlp_cumulant_propagation/scripts && python m9_sweep.py [run ...]

16 ReLU layers act0..act15, every layer W ~ N(0, 2/n) (num_layers = 17, init 'manual', w_var 2),
use_avg_metric = True (as in Lane M's Oct 6 / M2 / M8 chains), grad off (set by the import),
fp32, TF32 off. Means only (sample_k_max = output_d_max = 1). Each run writes
../data/relu_w2_m9/<name>/kprop_by_width.pt = (cfg, errs, sample_se2); all runs share one
sample cache, so the nets and labels are paired across methods by (n, trial).

Deviation from ARC's script: the sampler skips pre-activation layers (include_pre=False) --
only act* layers are scored, and it halves the sampling cost.
"""
import functools
import logging
import os
import sys

import torch

import kprop_by_width as kbw
from kprop_by_width import HKind, KPropByWidthCfg, kprop_by_width

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def fast_sample_means(mlp, samples, batch_size=100_000, return_stderr=False, batches_per_bs=5, *, k_max,
                      int_cond=None, d_max=None, include_pre=False, include_act=True):
    """Means-only stand-in for kbw.get_sample_cumulants (k_max = d_max = 1, act layers only).

    ARC's generic cumulant generator costs ~20 min per n = 256 trial at N = 2^30 against ~1.5 min
    of forward passes. Same estimator: per-bootstrap-block means accumulated in float64, mean of
    blocks, SE² = Var_blocks(ddof=1) / num_blocks (ARC's Welford M2 / (B−1) / B). Same cache format.
    """
    assert k_max == 1 and (d_max or 1) == 1 and include_act
    n = mlp.input_dim
    nb = samples // batch_size
    nbs = nb // batches_per_bs
    assert samples % (batch_size * batches_per_bs) == 0
    acc = None
    for b in range(nb):
        act = mlp(torch.randn(batch_size, n, device=kbw.device), output_acts=True).act   # (batch, L, n)
        s = act.sum(0, dtype=torch.float64)
        if acc is None:
            acc = torch.zeros((nbs,) + s.shape, dtype=torch.float64, device=s.device)
        acc[b // batches_per_bs] += s
    acc /= batch_size * batches_per_bs
    mean, se = acc.mean(0), acc.std(0, unbiased=True) / nbs ** 0.5
    mk = lambda v: kbw.DSTower({1: kbw.DSTensor({(1,): v}, n=n, d=1)})  # noqa: E731
    K = {f"act{l}": mk(mean[l]) for l in range(mean.shape[0])}
    S = {f"act{l}": mk(se[l]) for l in range(se.shape[0])}
    return K, (S if return_stderr else None)


kbw.get_sample_cumulants = fast_sample_means

COMMON = dict(
    samples=2**30, batch_size=2**17, batches_per_bs=2**7,       # 64 bootstraps; floor ~1e-9 << MSE
    trials=8, base_seed=0,
    sample_k_max=1, output_d_max=1,
    sampK_cache_name="relu_w2_m9/cache/sampK_L16",
    mlp_kwargs=dict(num_layers=17, nonlin="relu", init_kind="manual", w_var=2.0),
)
NS = (64, 96, 128, 192, 256)
RUNS = {  # name: (k_maxs, kprop_kwargs, widths)
    "k3aug_dense": ((3,), dict(kind=HKind.AUGMENT, factor=False), NS),
    "k3aug_fac": ((3,), dict(kind=HKind.AUGMENT, factor=True), NS),
    "k3simple_fac": ((3,), dict(kind=HKind.SIMPLE, factor=True), NS),
    "k4simple_fac": ((4,), dict(kind=HKind.SIMPLE, factor=True), NS),
    "k4aug_fac": ((4,), dict(kind=HKind.AUGMENT, factor=True), (64, 96, 128)),
}
if os.environ.get("M9_NS"):      # extra widths, written to <name>_n<widths> so earlier runs are kept
    _ns = tuple(int(x) for x in os.environ["M9_NS"].split(","))
    RUNS = {f"{k}_n{'_'.join(map(str, _ns))}": (v[0], v[1], _ns) for k, v in RUNS.items()}
if os.environ.get("M9_SMOKE"):   # tiny end-to-end check of the whole pipeline
    COMMON.update(samples=2**18, batch_size=2**13, batches_per_bs=2**2, trials=2,
                  sampK_cache_name="relu_w2_m9smoke/cache/sampK_L16")
    RUNS = {k: (v[0], v[1], (16, 24)) for k, v in RUNS.items()}

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        force=True)
    logging.getLogger("kprop_by_width").setLevel(logging.INFO)
    names = sys.argv[1:] or list(RUNS)
    for name in names:
        k_maxs, kk, ns = RUNS[name]
        tag = "relu_w2_m9smoke" if os.environ.get("M9_SMOKE") else "relu_w2_m9"
        cfg = KPropByWidthCfg(k_maxs=k_maxs, ns=ns, name=f"{tag}/{name}/kprop_by_width",
                              kprop_kwargs=dict(**kk, use_avg_metric=True), **COMMON)
        print(f"=== {name} {ns}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        kprop_by_width(cfg)
        print(f"=== {name} done, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
