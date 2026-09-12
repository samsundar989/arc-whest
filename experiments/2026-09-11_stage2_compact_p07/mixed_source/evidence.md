# Truth-free correlated-ReLU fourth source, 2026-09-12

Contract: `experiments/LOG.md` at `d695633`. Run from repository root:

```
uv run python local_sim/test_p2_k4_mixed_source.py --output experiments/2026-09-11_stage2_compact_p07/mixed_source/result.json
```

`result.json` reports **PASS, 80 checks** at the fixed preregistered gates. For each unit Gaussian pair (`rho=0`, `rho=0.5`), the script integrates ReLU raw moments through total order four on `[0,10]` with 64- and 96-node Gauss-Legendre quadrature. Single-coordinate moments integrate the full Gaussian marginal, including both signs of the unused coordinate. It builds all 16 fourth-tensor entries with the raw-moment set-partition cumulant formula, then separately reconstructs them from central fourth moments and three covariance pairings. It checks analytic univariate ReLU moments and fourth cumulants. Across the registered numeric comparisons, the largest absolute error is `1.6653345369377348e-15` and the largest relative error for material terms is `3.662261922909072e-15`, within `1e-8`.

At `rho=0`, all three distinct mixed fourth cumulants are zero to displayed precision, and the future full versus diagonal-only D4 gap is at most `5.551115123125783e-17`. At `rho=0.5`, `K1112=0.11417442414038237`, `K1122=0.10914008267226893`, and `K1222=0.11417442414038226`. Under fixed `B=[[1,1],[1,-1]]/sqrt(2)`, the future D4 values are `[0.5319131414655188, 0.07521544490398993]`; dropping mixed source entries changes these by `[+0.3920589722891677, -0.0646387242723612]`. Both registered lower bounds (`K1122>1e-3`, a future D4 gap `>1e-3`) pass.

For the explicitly declared `M=2I`, the script computes the contraction-matching `G` and retains the entire residual `E=K-T(G,M)`. The metric contraction matches to numerical precision. The transported core D4 is `[0.5199851640494928, 0.06328746748796399]`, and the explicit transported residual adds about `0.011927977416026` to each. At `rho=0`, projection alone also misses D4 by about `0.069927084588176`, despite zero mixed source. These are current tiny-tensor diagnostics, not a compression or challenge-cost certificate.

Workerq job **964** ran current live bytes from HEAD `d695633bcd44ac82b987b8ef6a96004a167f0af9` using `--no-snapshot --live-worktree --ram 5 --cpus 2 --gpus 0` and `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/preflight.ps1`. Terminal state **SUCCEEDED**, exit 0; **PREFLIGHT GREEN 12/16**, four Full-only steps skipped; measured peak **3911.1171875 MiB**. The log, manifest, environment, and queue result are byte-for-byte copies from the queue. Every member of `job964_raw.zip` equals its copy and original. The default preflight does not execute this new oracle; a focused rerun reproduced `result.json` byte for byte, and the script/result SHA-256 values were unchanged from preflight start through terminal.

| File | SHA-256 of disk bytes before commit |
|---|---|
| `local_sim/test_p2_k4_mixed_source.py` | `00f5485b44769d3545a9485a30165c11c4e77ad62cac1ba49e493edea80b43b1` |
| `result.json` | `6490c583d40125335e33ad30ed2c13a0e656f6e78548123a6cda09db6f27de70` |
| `job964_live_preflight.log` | `bfa4eb16901cb26f6c936ff41ae971b03deb9d3dd251453fc53d071df3d77f4a` |
| `job964_live_manifest.json` | `4ee80389af607fd252711645b60ab2b6246e24b543b967f48327b438f4782ec6` |
| `job964_live_environment.json` | `a13e440f0618295f30dd7f9765248c8734f8df509db8d3c3c56f28049788b7a4` |
| `job964_live_result.json` | `6dfb5c437be86623ae6d0695f4941a3352de05a7608c2f6de69951386965bf1e` |
| `job964_terminal_show.json` | `8f9b58d62da01dcab1d869636cc4a1c79d45d5d812f23d4d293cb9e286e8b159` |
| `job964_raw.zip` | `394ed199aec9436c2cb7ad7072e4157a116f875cf05f2c293a449ca43cda48d0` |

This establishes only the exact Gaussian-pair post-ReLU fourth source at the stated quadrature precision and its tiny signed linear transport. It does not establish a non-Gaussian later-layer source, a compact transportable residual, accuracy, counted flopscope cost, memory-limit compliance, or server residual headroom. No P01/P07/public/vault labels, registry seed block, frozen arm, or shipped estimator were accessed or changed by this source test.
