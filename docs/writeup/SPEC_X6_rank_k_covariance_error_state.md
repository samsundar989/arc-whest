# Worker spec X6 — rank-k transported covariance-error state with quadratic readout (the off-diagonal channel X1 could not carry)

**Owner:** the X1 lane (code, dumps and trainer exist: `experiments/2026-10-20_x1/{dump.py,train.py,eval_val.py}`, trajectories in `cache/x1/{train,val}`). Local 5090 through workerq, alongside R3 (two processes fit; the X1 trainer used 12 min).
**Clock:** trainer change +1.5 h; two arms × 2 seeds +1 h; VAL read +0.5 h. Decision by +3.5 h. Then FRESH-A dumps (the cancelled jobs 3537/3538) if it passes.

## Why this and not more of X1
- X1 confirmed that ~75–80 % of late-layer error variance is *transported* (Φ∘(Wᵀd) with gain 1.00, R² 0.74–0.81 at layers 11–15) and that the module's per-layer gain decays from −7.6 % at L4 to −0.8 % at L16: it corrects what the in-loop corrector already handles and cannot predict the late innovations.
- The late innovation is dominated by the **variance channel**: the round-3 corr study found 49–51 % of the one-step mean-error energy at L12–16 is the first-order response to δσ, that δσ is ≥ 97 % sourced from the **off-diagonal covariance error through the sandwich** (δvar_{l+1,i} = Σ_{j≠k} W_ij W_ik δC_{l,jk}), and that no per-neuron corrector can see it (R² ≤ 0.031 even with the oracle *diagonal* error as input). X1's state (an n-vector per channel, transported by W and W∘W) carries exactly the diagonal — the W∘W message is diag(W diag(s) Wᵀ) — and so could not carry this channel either.
- The exact carrier of an off-diagonal covariance error δC ≈ U Uᵀ − V Vᵀ (rank k) through a layer is **U_{l+1} = Φ_{l+1}∘(W_{l+1} U_l)** (gate-linearised), and its effect on the next variance is the **quadratic readout** q_i = Σ_k (U_{l+1})²_{ik} − Σ_k (V_{l+1})²_{ik}. A learned U_l (n×k) with this transport and this readout is the smallest module that can represent the channel. Cost at inference: 2·n²·k per layer per state (k = 16: 34 MFLOP/layer, ≈ 0.55 GF total = 0.19 % of the bill for U and V together); ops: two matmuls per layer.
- This is not in any public write-up (mliston's GRU and hyojun's/Susupro's modules transport vectors, not low-rank matrices with quadratic readouts).

## 0a. Sandbox evidence (added 18:10 UTC, /home/claude/whest_reports/I_x6_sandbox.md and sandbox/results/x6_an_*.txt)
In exact numpy towers vs MC truth (n = 128/256, L = 8/16), the post-ReLU covariance error O_l of every truncated chain tested is **low-rank and coherent**: participation ratio 1.3–4 for age-truncation and for the gate-transport error (the closest proxy to our dressing deficit), ≈ 10–40 for the missing-hub error; the top-16 eigen-pairs carry 84–98 % of ‖O_l‖_F² and reproduce R² 0.80–0.95 of the induced next-layer variance error (a GOE matrix of the same size: 0.21–0.38). The oracle-innovation rank-k state S_l = Trunc_k[transport(S_{l−1}) + I_l] reproduces the variance error with R² 0.88–0.99 at k = 8–16 (age/gate errors) and 0.65–0.93 (missing-hub error). Per-neuron / diagonal messages of the X1 kind explain only 13–25 %. The innovation I_l itself has PR ≈ 3–4 (fro16 ≈ 0.88), i.e. it lies in a few coherent directions (scale-mode-like, built from m_l, var_l, g_l, D3_l) — which is what a per-neuron MLP can emit. **Consequences for the build:** run k = 8 and k = 16 before k = 32; seed the first column of U with the chain's own m_l (the scale-mode direction) so the MLP only learns its amplitude; the √T2 gate variant is worth one arm. The unknown is only whether the real 1024×16 chain's covariance error is as coherent as these towers' — the VAL read decides.

## 0. Pre-registered prediction and kill line
- Prediction (strategist): VAL-100 paired, real chain, corrector-on: **−2.5 % raw (80 %: −0.5 … −7 %)**; P(≤ −2 %) 0.35. Attribution: if the quadratic-readout arm beats X1-A by < 0.3 pp, the channel is not learnable at rank ≤ 32 either — record and stop.
- Keep if ≤ −2.0 % raw with CI excluding 0 **and** score-equivalent (raw ⊕ real bill) ≤ −1.5 %; then FRESH-A100 ≤ −1.5 %.

## 1. Model (minimal change to X1's arm A/B code)
State per layer: u_l ∈ ℝⁿ (mean error, as X1), U_l ∈ ℝ^{n×k} and V_l ∈ ℝ^{n×k} (signed low-rank covariance-error factors), k = 16 (arm K16) and k = 32 (arm K32).
- Transport (exact linear algebra, no parameters): ũ = Φ_l∘(W_l u_{l−1}); Ũ = Φ_l∘(W_l U_{l−1}) (column-wise, i.e. diag(Φ)·W·U); Ṽ likewise. Use the same Φ gate as X1 (or √T2 = √(Φ − αφ) for the covariance factors, as the sandbox found for coherent κ4 faces — make it a flag; default Φ).
- Readout features per neuron (added to X1's 14 local features): q_i = Σ_k Ũ_ik² − Σ_k Ṽ_ik² (the predicted variance innovation), r_i = Σ_k Ũ_ik·ũ_i-type cross term (optional), and the k-dim rows Ũ_i, Ṽ_i themselves (so the MLP can also use them linearly).
- MLP g (shared across layers, 3 × 128 GELU as X1-A): inputs [14 local ‖ ũ_i ‖ q_i ‖ Ũ_i (k) ‖ Ṽ_i (k) ‖ onehot(l)] → outputs: g_l (scalar innovation of the mean error), ΔU_i (k), ΔV_i (k) (the covariance-error innovation this layer adds, low-rank); U_l = Ũ + ΔU, V_l = Ṽ + ΔV, u_l = ũ + g_l. Final prediction = b_16 + u_16, clamped at ≥ 0.
- Targets: as X1 (ε_l innovation targets from the all-layer mean labels, w_16 = 4). **No variance labels are needed** — U, V are latent; the mean loss through the recurrence trains them. If FIT 0–99 exact per-layer variances are convenient (`cache/twin_labels/fit_0xx.npz`), add an auxiliary loss 0.1·‖q_l − δvar_l‖² on those 100 nets only (arm K16v) — it should speed convergence but is optional.
- Stabilisation as in X1 (tanh-bounded state scale, output scale 0.1, warmup + cosine); 10–20 epochs; TRAIN 0–849 / 850–899 early-stop; VAL 900–999 read once per model on the real numpy chain.

## 2. Arms
A (X1-A rerun, reference) · K16 · K32 · K16v. Two seeds each if time allows (K16 first).

## 3. Read and price
- VAL-100 paired real-chain table as X1; per-layer gains at L4/L8/L12/L16 — the diagnostic is whether the L12–L16 gain rises above X1-A's −2.2/−0.8 %.
- Real flopscope bill of the module alone on 2 nets (matmuls `W @ U` are billed at 2n²k; the MLP as in X1); ops count.

## 4. Deliverable
`experiments/2026-10-20_x6/RESULT.md` (prediction vs outcome, table, bill), one `RESULTS.md` line, exported constants if kept. Integration path = X1 §4 (post-hoc `fnp` module after the chain; byte-identical zero control at scale 0).
