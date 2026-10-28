# I_x6_sandbox — Can a rank-k transported covariance-error state (worker spec X6) carry the off-diagonal variance channel?

Sandbox test of SPEC_X6 (`/home/claude/whest_reports/SPEC_X6_rank_k_covariance_error_state.md`) in the numpy cumulant tower (`/home/claude/sandbox/`, 2 cores, 7 GB).
Scripts: `x6_chains.py` (runs reference + truncated chains, dumps per-layer μ_l, C_l, m_h, Cov_h), `x6_analysis.py` (all diagnostics A–H, D2–D4 below), `x6_null.py` (GOE null), `x6_squeeze.py` (squeezed-GOE null, runs at n = 1024 directly), `x6_summary.py`, `x6_tables.py`. One additive line was added to `tower.py` (records `m_h`, `Cov_h` per layer). Raw outputs: `results/x6_*.{npz,json,txt,log}`, tables in `results/x6_tables.md`.

Notation. z_l pre-activation (l = 1..L), h_l = ReLU(z_l), z_{l+1} = W_{l+1} h_l. δC_l = C_l(truncated) − C_l(reference) (pre-activation covariance error, n×n), δvar_l = diag(δC_l). δCovh_l = Covh_l(trunc) − Covh_l(ref) is the post-ReLU covariance error the chain actually propagates (recorded exactly, W δCovh_l Wᵀ = δC_{l+1} to 1e-12), split into D_l (diagonal) + O_l (off-diagonal). Then δvar_{l+1} = (W∘W)·diag(D_l) [diagonal channel, X1's message] + diag(W O_l Wᵀ) [off-diagonal sandwich, X6's target]. Gate g = Φ(μ/σ) from the reference chain; gate-linearised transport of a covariance error is δC_{l+1} ≈ W diag(g) δC_l diag(g) Wᵀ, which is exactly what X6's U ← W(g∘U) carries. "GOE" = random symmetric matrix of the same Frobenius norm; "scr" = O_l with its entries sign-scrambled (same magnitudes, coherence destroyed).

---

## 0. Verdict

**Representable, not learnable as specified; and the representable fraction at n = 1024 is set by the age of the error, not by k alone.**

1. In the sandbox every covariance error we can produce — dropped old lineages (age ≤ 3), missing hub-born κ3, gate-transport error, the K2-only chain, and even the *second-order* residual of the complete first-order tower against MC — is **low-rank at depth**: at n = 128, L = 16 the participation ratio of O_l is 3–4 (hub-missing) or 1.3–1.7 (lineage/K2) at l ≥ 12, the top-16 eigenpairs hold 96–98 % of ‖O_l‖_F² and reproduce 95–99 % of the induced next-layer variance error (ndR²₁₆), against 0.35–0.45 for a GOE matrix and 0.65–0.9 for the sign-scrambled matrix (T2). An oracle rank-16 state with a fresh oracle innovation every layer reproduces 88–93 % (hub-missing) / 99 % (lineage) of δvar_l at l = 12–16 at n = 128; rank 32: 98–100 % (D2). At n = 256 the hub-missing numbers are already 53–71 % / 83–87 % (T5).
2. Two mechanisms make it low rank, and they scale differently with n:
   (a) **coherent lineage content** — the non-random-sign part of every born κ3 pattern (Σ_m d_m V_im² V_jm ≈ ⟨V²⟩(V d)_j, i.e. a rank-2 "a_i w_j + w_i a_j" footprint per birth, cos² with m_h up to 0.9): n-independent rank, dominates the age-truncation, K2 and second-order errors (top-1/2 eigenpairs carry 50–90 %, flat or *growing* with n from 64 to 256 — 0.63 → 0.79 → 0.92 for the second-order residual, T4);
   (b) **transport squeezing** — products of gated W's compress any old error into their top singular directions. This part obeys a k/n law: the squeezed-GOE null (T6, computed at n = 1024 directly) has PR = 1.8, 4.5, 9.5, 14.7 at n = 128, 256, 512, 1024 for a single incoherent innovation 14 layers old, i.e. PR ∝ n. At n = 1024, L = 16 a 12-layer-old incoherent error is captured 70–76 % (k = 16) / 89–92 % (k = 32); a *steady stream* of incoherent innovations (transported share 65–72 %, like the real chain's 75–80 %) is captured only **18–22 % (k = 16) / 30–35 % (k = 32)** of ‖δC‖² at l = 12–16 (25–30 % / 37–44 % of the variance error by R²), versus 65–74 % / 82–87 % at n = 128 — the sandbox over-states the incoherent part by 3×.
3. The real chain's late-layer error is, by the round-3/X1 facts, 75–80 % transported and has no per-neuron predictor (R² ≤ 0.03). The sandbox's hub-missing error is the closest analogue (R²(loc) 0.1–0.3 at n = 128 falling with n, innovation share 0.14–0.32 at l ≥ 12). Its fixed-k fractions are flat from n = 64 to 128 at L = 16 (fro₁₆ 0.98 → 0.97 at l = 14) but drop from 128 to 256 (0.97 → 0.85; PR 3–4 → 10–11; oracle-innovation rank-16 ceiling D2 0.89 → 0.53–0.71, rank-32 0.99 → 0.83–0.87; T5) and at L = 8 (fro₁₆ 0.96 → 0.87 → 0.65 for n = 64 → 128 → 256 at l = 7, T4): it is a mixture of (a) and (b) in which (b) dominates by n = 256. **Extrapolated to 1024×16 at l = 12–16: a rank-16 (32) state can represent ≈ 25–50 % (40–70 %) of the off-diagonal variance-error energy if the real error is lineage-like as in the sandbox, and ≈ 20 % (32 %) if it is a stream of fresh incoherent innovations; with a perfect innovation every layer the rank-16 (32) state reproduces ≈ 0.3–0.45 (0.5–0.7) of δvar.** The strategist's premise ("δC ≈ UUᵀ − VVᵀ with k = 16") is therefore half right: the directions exist, but k = 16 is not "the" rank — the effective rank of the incoherent part is ≈ n/t and only the coherent lineage part is O(1).
4. **What X6 as specified cannot do.** (i) A state initialised once and transported explains nothing after ≈ 6 layers (pure-transport R² → 0 by l0 + 6 in every run; the energy gain of the gated covariance transport is ≈ 0.67/layer and the directions rotate; C). (ii) Per-layer innovation is therefore mandatory, and the innovation's *directions* are not the problem — injecting 10 arbitrary vectors per layer (local features or pure random, D4) with an oracle (k+10)×(k+10) core already reaches 0.5–0.8 at k = 16, because squeezing supplies the directions. (iii) The *amplitudes* (the k×k core) are the problem: the error is not in the span of per-neuron features (local vectors hold 1–10 % of ‖δC_l‖², H), local-born innovations with an oracle core give ≤ 0.19 (D3), and the needed core is the net-specific random-sign lineage content. X6's MLP sees only per-neuron features and its own state rows, so it can produce per-neuron gains and layer-constant k×k mixing; its inputs are the per-neuron quantities the round-3 study already showed carry R² ≤ 0.03 of this channel. The sandbox gives no evidence that the core is learnable; it gives positive evidence that it is net-specific (sign-scrambling the entries of O_l, which keeps every per-neuron magnitude, removes 30 % of the captured fraction at n = 128 and 50 % at n = 256, T2/T5).
5. Hence: **a rank-k transported state can carry part of the off-diagonal variance channel in principle — at 1024×16, k = 16 holds ≈ 20–50 % and k = 32 ≈ 30–70 % of it (the lower ends if the real error is a stream of fresh incoherent innovations, the upper ends if it is squeezed lineage content), i.e. below the ≥ 50 % at k = 16–32 the task set as the "can work" line and above the < 15 % "cannot" line only for k = 32 in the lineage case — and only with per-layer innovations whose amplitudes are supplied from outside the per-neuron feature set. The X6 module as written (per-neuron MLP → ΔU) has no such input and should be expected to behave like X1 (gain concentrated at early layers, ≤ 1 % at L12–16).** Before spending the 3.5 h trainer budget, run the 1-hour pre-check in §7 on two real nets: it measures the only unknown (PR of the real δC_l and the D2 ceiling) directly.

---

## 1. Set-up: chains and their MSE ratios

Nets W_l ~ N(0, 2/n), x ~ N(0, I), MC truth from `results/mc_n{n}_L8_s{seed}.npz` (L = 8 only; L = 16 chains have no MC and are compared to the full tower). Reference = complete first-order κ3 tower (`run_tower` defaults, k4 = False). Truncated chains:

| chain | n=64 s0 | n=64 s1 | n=128 s0 | n=256 s0 |
|---|---|---|---|---|
| full tower (reference) | 9.56e-05 (1.00×) | 4.93e-05 (1.00×) | 1.07e-05 (1.00×) | 4.72e-06 (1.00×) |
| ages ≤ 2 (direct+indirect) | 2.77e-04 (2.90×) | 6.08e-05 (1.23×) | 3.07e-05 (2.86×) | – |
| **ages ≤ 3** | 1.67e-04 (1.75×) | 5.72e-05 (1.16×) | 2.00e-05 (**1.87×**) | 8.82e-06 (**1.87×**) |
| ages ≤ 4 | 1.05e-04 (1.10×) | 5.59e-05 (1.13×) | 1.40e-05 (1.30×) | 7.14e-06 (1.51×) |
| direct all, indirect ages ≤ 3 | 1.43e-04 (1.49×) | 4.31e-05 (0.87×) | 1.77e-05 (1.65×) | – |
| **born diag+pair only (no hub)** | 1.30e-04 (1.36×) | 5.62e-05 (1.14×) | 2.20e-05 (**2.05×**) | 8.09e-06 (**1.71×**) |
| gate transport | 8.77e-05 (0.92×) | 4.49e-05 (0.91×) | 1.12e-05 (1.04×) | – |
| K2 only | 8.00e-04 (8.37×) | 2.82e-04 (5.72×) | 1.18e-04 (11.0×) | 4.16e-05 (8.81×) |

Two "almost right" chains were analysed in full, because they produce the two archetypes of error:
* **age3** (ages ≤ 3 in both channels; 1.87× at n = 128 and 256): drops the *indirect shifts of old lineages* — a coherent error (the lineage footprint "a_i w_j + w_i a_j"), locally predictable (R²(loc) 0.2–0.9).
* **dp** (diag+pair born only, hub-born dropped; 1.7–2.05×): drops a *random-sign born pattern* — the most incoherent first-order error available, the only one with R²(loc) ≤ 0.3 like the real chain. Note the dp chain also matches the real chain in that ≥ 85 % of δvar_{l+1} comes through the off-diagonal sandwich (off-ch share 0.85–1.0, T2) and 70–85 % of δC_l is transported at l ≥ 12.
K2-only, gate-transport, ind3/age4 and the full tower vs MC (second-order residual) were run as brackets. At L = 16 (no MC) the chains were run at n = 64 (2 seeds), 128 and 256 (dp only at 256, merged births).

---

## 2. Experiment 1–2: size and rank of δC_l; fraction of the induced variance error captured by rank k

Per layer (T2, n = 128, L = 16, dp = hub-missing; the full tables for all chains, n and L are in `results/x6_tables.md` and the `results/x6_an_*.txt` files):

| l | off-ch share | PR(O_l) | fro1 | fro2 | fro4 | fro8 | fro16 | fro32 | ndR²4 | ndR²16 | ndR²32 | GOE R²16/32 | scr R²16/32 | D2 k=8/16/32 | D4 k=16 (loc/rnd) | I/dC | span1_16 | core1_16 | R²(loc) | tpl | win4 / null |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.941 | 29.5 | 0.076 | 0.150 | 0.255 | 0.427 | 0.641 | 0.857 | 0.339 | 0.703 | 0.895 | 0.34/0.60 | 0.43/0.75 | 0.44/0.66/0.88 | 0.06/0.12 | 1.00 | – | – | 0.08 | 0.01 | 0.02 / 0.01 |
| 4 | 0.859 | 19.5 | 0.151 | 0.229 | 0.356 | 0.519 | 0.718 | 0.896 | 0.435 | 0.687 | 0.913 | 0.43/0.67 | 0.51/0.74 | 0.56/0.72/0.92 | 0.30/0.17 | 0.70 | 0.32 | 0.43 | 0.07 | 0.02 | 0.08 / 0.01 |
| 6 | 0.877 | 11.7 | 0.214 | 0.326 | 0.492 | 0.669 | 0.829 | 0.951 | 0.539 | 0.861 | 0.960 | 0.38/0.63 | 0.63/0.84 | 0.63/0.80/0.94 | 0.32/0.34 | 0.64 | 0.38 | 0.29 | 0.22 | 0.04 | 0.23 / 0.03 |
| 8 | 0.961 | 8.4 | 0.257 | 0.411 | 0.607 | 0.774 | 0.892 | 0.969 | 0.697 | 0.933 | 0.980 | 0.34/0.58 | 0.55/0.79 | 0.66/0.85/0.97 | 0.50/0.45 | 0.45 | 0.60 | 0.59 | 0.19 | 0.06 | 0.40 / 0.06 |
| 10 | 0.937 | 5.5 | 0.357 | 0.514 | 0.722 | 0.838 | 0.932 | 0.983 | 0.700 | 0.923 | 0.985 | 0.35/0.63 | 0.62/0.89 | 0.72/0.82/0.98 | 0.49/0.46 | 0.48 | 0.59 | 0.63 | 0.13 | 0.09 | 0.33 / 0.06 |
| 12 | 1.035 | 2.9 | 0.569 | 0.689 | 0.818 | 0.894 | 0.958 | 0.991 | 0.815 | 0.951 | 0.986 | 0.22/0.53 | 0.70/0.90 | 0.63/0.89/0.98 | 0.47/0.50 | 0.32 | 0.71 | 0.68 | 0.28 | 0.17 | 0.45 / 0.04 |
| 13 | 0.997 | 3.0 | 0.550 | 0.691 | 0.809 | 0.899 | 0.961 | 0.989 | 0.789 | 0.966 | 0.993 | 0.36/0.64 | 0.69/0.88 | 0.70/0.88/0.98 | 0.57/0.59 | 0.24 | 0.81 | 0.73 | 0.12 | 0.15 | 0.57 / 0.02 |
| 14 | 0.974 | 3.1 | 0.515 | 0.721 | 0.868 | 0.923 | 0.970 | 0.992 | 0.833 | 0.974 | 0.993 | 0.40/0.61 | 0.69/0.89 | 0.77/0.88/0.99 | 0.75/0.60 | 0.21 | 0.80 | 0.81 | 0.17 | 0.18 | 0.54 / 0.03 |
| 15 | 0.913 | 4.0 | 0.416 | 0.651 | 0.841 | 0.918 | 0.966 | 0.991 | 0.904 | 0.972 | 0.994 | 0.34/0.58 | 0.65/0.88 | 0.78/0.90/0.99 | 0.78/0.58 | 0.14 | 0.88 | 0.84 | 0.25 | 0.13 | 0.59 / 0.04 |

Columns: *off-ch share* = ‖diag(W O_l Wᵀ)‖²/‖δvar_{l+1}‖²; *PR* = (Σλ²)²/Σλ⁴ of O_l; *fro_k* = share of ‖O_l‖_F² in the top-k eigenpairs (by |λ|); *ndR²_k* = R² of diag(W T_k Wᵀ) with the diagonal of the rank-k truncation T_k removed, against δvar^off_{l+1} (this is X6's quadratic readout once the (W∘W)·diag(UUᵀ) piece is left to the diagonal channel; with the diagonal kept the R² is erratic at small k, e.g. −0.21 at k = 2, l = 5 — a practical point for the trainer, §7); *GOE / scr* = same R² for a GOE matrix / sign-scrambled O_l; *D2* = R² on δvar_l of S_l = Trunc_k[transport(S_{l−1}) + I_l] (rank-k state, oracle innovation every layer); *D4* = same when new directions may enter the state only through 10 vectors per layer — local features (loc) or random (rnd) — with an oracle core; *I/dC* = ‖I_l‖²/‖δC_l‖², I_l = δC_l − W G δC_{l−1} G Wᵀ; *span1/core1* = one-step persistence of the top-16 directions of δC_{l−1} after transport (span fraction of δC_l / oracle-core R² on δvar_l); *R²(loc)* = OLS of δvar_l on [var, μ, μ², g, φ]; *tpl* = share of ‖O_l‖² in m_h m_hᵀ + Cov_h templates; *win4/null* = share of ‖δC_l‖² in the span of 4 local vectors × last 6 layers transported, and a random span of the same dimension.

Same table for **age3** (coherent lineage error), n = 128, L = 16, l = 8/12/14/15: PR 2.0/1.4/1.6/1.7; fro1 0.70/0.84/0.79/0.76; fro16 0.973/0.984/0.983/0.981; ndR²₁₆ 0.978/0.990/0.987/0.986; GOE R²₁₆ 0.45/0.42/0.29/0.39; D2 k=16 0.99/0.99/0.99/0.99; D4 loc/rnd 0.85/0.60, 0.96/0.89, 0.94/0.84, 0.93/0.81; I/dC 0.29/0.12/0.25/0.21; R²(loc) 0.60/0.87/0.72/0.70; tpl 0.57/0.77/0.67/0.63. **K2-only** is the same picture with PR 1.2–1.4 and R²(loc) 0.85–0.95. **Full tower vs MC** (second-order residual, n = 128, L = 8, noise ≤ 0.4 % of ‖δC‖²): PR 1.6–2.3, fro1 0.66–0.80, fro16 0.95–0.98, cos²(u₁, m_h) 0.5–0.85 (the emergent scale mode m mᵀ of F_math §7.3), R²(loc) 0.4–0.55.

Reading. (i) In every case the off-diagonal sandwich carries 85–100 % (dp) or 60–80 % (age3/K2) of ‖δvar_{l+1}‖², as in the real chain. (ii) At n = 128 rank 16 reproduces 95–99 % of the induced variance error at l ≥ 10 for every error type; the dp error starts broad (PR 30, fro16 = 0.64 at l = 3, essentially the sign-scrambled null) and is squeezed to PR 3–4 by l = 12. (iii) The GOE null at n = 128 is itself 0.35–0.45 at k = 16 (k/n = 1/8), so the sandbox numbers cannot be read at face value for n = 1024 — see §6.

L = 8, n = 128 (last post-ReLU layer l = 7), all chains, fro16 / ndR²₁₆ / D2₁₆: age3 0.97/0.98/0.98; age4 0.97/0.98/0.99; ind3 = age3; dp 0.87/0.87/0.87; gate 0.94/0.95/0.95; K2 0.96/0.98/0.96; full-vs-MC 0.98/0.98/0.93 (full rows in `results/x6_tables.md` T3).

---

## 3. Experiment 3: transport persistence (state initialised once, no innovation)

Top-k eigenpairs of δC_{l0} (l0 = first non-zero layer) transported as U ← W(g∘U) (and with √T2 gates), read at later layers (section C of the `.txt` files). n = 128:

| chain, L | l0 → l | pure R²₁₆ (transport only) | span₁₆ (fraction of ‖δC_l‖² in the transported span) | core₁₆ (oracle k×k refit in that span) | fullT (full-rank transported δC_{l0}, no truncation) | in-layer ceiling eig₁₆ |
|---|---|---|---|---|---|---|
| dp, L=16 | 3 → 4 / 6 / 8 / 12 / 16 | 0.24 / 0.07 / 0.03 / −0.00 / −0.00 | 0.54 / 0.35 / 0.41 / 0.44 / 0.51 (span) | 0.43 / 0.35 / 0.41 / 0.44 / 0.51 | 0.28 / 0.03 / 0.00 / −0.01 / −0.00 | 0.95–0.98 |
| age3, L=16 | 6 → 7 / 9 / 12 / 16 | 0.36 / 0.20 / −0.06 / −0.04 | – | 0.58 / 0.79 / 0.86 / 0.83 | 0.33 / 0.18 / −0.06 / −0.05 | 0.98–0.99 |
| age3, L=8 | 6 → 7 / 8 | 0.36 / 0.27 | 0.54 / 0.65 | 0.58 / 0.68 | 0.33 / 0.26 | 0.98 / 0.99 |
| dp, L=8 (n=256) | 3 → 5 / 8 | 0.08 / −0.00 | 0.14 / 0.07 (core) | 0.14 / 0.07 | 0.10 / 0.00 | 0.65 / – |
| dp, L=16 (n=256) | 3 → 8 / 12 / 16 | −0.00 / 0.01 / −0.00 | – | 0.07 / 0.30 / 0.37 | −0.00 / 0.01 / 0.00 | 0.72 / 0.87 / 0.89 |

√T2 gates are uniformly slightly worse than Φ gates (e.g. age3 L=8: pure₁₆ 0.363 → 0.325, fullT 0.334 → 0.290); the exact first-order response of Cov_h,ij to δC_ij is g_i g_j, √T2 belongs to κ4 faces. **A state transported without innovation explains nothing of the late-layer error** — not because of the rank cut (the full-rank transported δC_{l0} is just as dead) but because the gated covariance transport attenuates energy by ≈ 0.67/layer (T6: ‖S‖² 1 → 0.004 over 14 layers) while new error is injected every layer. The transported *span* keeps 40–50 % (dp) to 80–86 % (age3) of the later error at n = 128 because the squeezing re-aligns everything with the same top singular directions — but that fraction is 7–14 % at n = 256 (dp, L = 8), i.e. it is a k/n effect.

---

## 4. Experiment 4: innovation rank

I_l = δC_l − W G δC_{l−1} G Wᵀ (gate-linearised; the exact one-step propagation through `cov_relu` gives the same numbers to ±0.01, so gate linearisation is not the issue). n = 128, L = 16, dp:

| l | ‖I_l‖²/‖δC_l‖² | ‖diag I_l‖²/‖δvar_l‖² | R² of transported δvar (R2tr) | PR(I_l) | fro16(I_l) | fro32(I_l) | sandwich R²₁₆ of rank-16 I_l at l+1 |
|---|---|---|---|---|---|---|---|
| 4 | 0.70 | 0.72 | 0.28 | 20.0 | 0.75 | 0.93 | 0.84 |
| 8 | 0.45 | 0.48 | 0.52 | 9.8 | 0.89 | 0.98 | 0.90 |
| 12 | 0.32 | 0.32 | 0.68 | 4.1 | 0.96 | 0.99 | 0.97 |
| 14 | 0.21 | 0.19 | 0.81 | 3.9 | 0.95 | 0.99 | 0.96 |
| 16 | 0.18 | 0.14 | 0.86 | 4.3 | 0.95 | 0.99 | – |

age3: innovation share 0.12–0.29 at l ≥ 8, PR(I) 2–4. The transported share at l ≥ 12 (0.68–0.86) matches the real chain's X1 figure (0.74–0.81). The innovation is itself low-rank at depth *in the sandbox* because the dropped κ3 content is lineage content that was transported before it acts (PR(I) ∝ age-squeezing, §6); a genuinely fresh innovation has PR ≈ 0.5 n (GOE, PR = 64 at n = 128).

Oracle-innovation rank-k state (D2, n = 128, L = 16, R² on δvar_l): dp k = 4/8/16/32 at l = 12: 0.67/0.63/0.89/0.98, l = 14: 0.74/0.77/0.88/0.99, l = 16: 0.81/0.84/0.93/0.99; age3 k = 8/16: 0.98/0.99 from l = 8 on; K2: 0.99/0.99. So *if the innovation were available*, rank 16 is enough at n = 128.

Local-born innovations (D3: innovation restricted to F_l A F_lᵀ, F_l = 10 per-neuron features, oracle 10×10 core): dp k = 16 at l = 12/14/16: 0.07/0.02/0.07 (vs D2 0.89/0.88/0.93); D4 (new directions through 10 local or 10 random vectors, oracle core, k = 16): loc 0.47/0.75/0.79, rnd 0.50/0.60/0.73. **The directions are free (random vectors do as well as local ones); the core is what is missing.**

---

## 5. Experiment 5: what the diagonal channel and per-neuron features explain (section E)

OLS R² of δvar_{l+1} on (W∘W)·δVar_h,l (X1's exact diagonal message), on the X1-type message (W∘W)(g²δvar_l), on local features [var, μ, μ², g, φ]_{l+1}, and on all; n = 128:

| chain, L | l+1 | R²(fdiag) | R²(g² msg) | R²(loc) | R²(all) | off-channel share |
|---|---|---|---|---|---|---|
| dp, 16 | 6 / 10 / 12 / 14 / 16 | 0.18 / 0.11 / 0.02 / 0.01 / 0.10 | 0.15 / 0.11 / 0.01 / 0.01 / 0.09 | 0.22 / 0.13 / 0.28 / 0.17 / 0.07 | 0.24 / 0.16 / 0.28 / 0.17 / 0.14 | 0.88 / 0.94 / 1.00 / 0.97 / 0.91 |
| age3, 16 | 8 / 12 / 16 | 0.25 / 0.19 / 0.11 | 0.24 / 0.18 / 0.11 | 0.60 / 0.87 / 0.78 | 0.61 / 0.87 / 0.79 | 0.68 / 0.70 / 0.64 |
| age3, 8 | 6 / 7 / 8 | 0.14 / 0.25 / 0.20 | 0.13 / 0.24 / 0.19 | 0.23 / 0.44 / 0.60 | 0.24 / 0.45 / 0.61 | 0.90 / 0.77 / 0.81 |
| dp, 8 (n=256) | 4 / 6 / 8 | 0.02 / 0.03 / 0.05 | 0.02 / 0.03 / 0.05 | 0.02 / 0.05 / 0.06 | 0.03 / 0.06 / 0.10 | 0.96 / 0.97 / 1.02 |

The diagonal channel (what X1 carries) explains ≤ 0.25 of δvar anywhere and ≤ 0.1 at late layers; the gap to the off-diagonal channel is 0.75–1.0 of the energy. Local features explain the coherent lineage error (age3: 0.6–0.9) but not the hub-type error (dp: 0.1–0.3 at n = 128, 0.02–0.06 at n = 256 → consistent with the real chain's ≤ 0.03 at n = 1024). The sandbox analogue of the real chain's channel is therefore the dp-type error, and its coherent template part (tpl) is only 0.01–0.18.

---

## 6. n-scaling and the extrapolation to n = 1024

### 6.1 Sandbox scaling, L = 8 (last post-ReLU layer l = 7; T4)

| error | n | PR | fro1 | fro2 | fro4 | fro8 | fro16 | fro32 | ndR²₁₆ | ndR²₃₂ | GOE null fro16 / fro32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dp | 64 (s0/s1) | 5.5 / 8.6 | 0.31 / 0.22 | 0.55 / 0.38 | 0.76 / 0.61 | 0.89 / 0.81 | 0.96 / 0.94 | 1.00 / 0.99 | 0.95 / 0.89 | 1.00 / 0.99 | 0.62 / 0.89 |
| dp | 128 | 9.8 | 0.22 | 0.39 | 0.56 | 0.73 | 0.87 | 0.96 | 0.87 | 0.94 | 0.38 / 0.63 |
| dp | 256 | 27.2 | 0.10 | 0.17 | 0.29 | 0.45 | 0.65 | 0.84 | 0.70 | 0.86 | 0.21 / 0.38 |
| age3 | 64 (s0/s1) | 1.9 / 3.5 | 0.73 / 0.45 | 0.82 / 0.72 | 0.88 / 0.85 | 0.94 / 0.93 | 0.98 / 0.98 | 1.00 | 0.99 / 0.98 | 1.00 | 0.62 / 0.89 |
| age3 | 128 | 1.9 | 0.71 | 0.84 | 0.89 | 0.94 | 0.97 | 0.99 | 0.98 | 0.99 | 0.38 / 0.63 |
| age3 | 256 | 3.0 | 0.54 | 0.73 | 0.78 | 0.85 | 0.91 | 0.96 | 0.93 | 0.97 | 0.21 / 0.38 |
| K2 only | 64 / 128 / 256 | 1.6–2.2 / 1.4 / 1.6 | 0.80, 0.66 / 0.84 / 0.79 | – | – | – | 0.99, 0.98 / 0.96 / 0.93 | 1.00 / 0.99 / 0.97 | 0.99 / 0.98 / 0.94 | – | – |
| full vs MC (2nd order) | 64 / 128 / 256 | 2.4, 2.3 / 1.6 / 1.2 | 0.62, 0.64 / 0.79 / 0.92 | – | – | – | 0.99 / 0.98 / 0.98 | 1.00 / 1.00 / 0.99 | 0.99 / 0.98 / 0.97 | – | – |

Fixed k, dp: fro16 falls 0.95 → 0.87 → 0.65 per doubling of n (PR 7 → 10 → 27, roughly ∝ n); at fixed k/n = 1/16 (k = 4, 8, 16) it is 0.69 → 0.73 → 0.65, i.e. ≈ constant: **the hub-type error follows a k/n law at L = 8**. The age3 / K2 / second-order errors are dominated by 1–2 directions whose share is flat or *growing* with n (fro1 of the second-order residual 0.63 → 0.79 → 0.92): **coherent, n-independent rank**.

### 6.2 Sandbox scaling, L = 16 (T5; n = 256 only for dp)

| error | n | l | PR | fro1 | fro2 | fro4 | fro8 | fro16 | fro32 | D2 k=16 | D2 k=32 | D4 k=16 | I/dC | span1₁₆ | R²(loc) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dp | 64 (s0 / s1) | 14 | 2.4 / 4.3 | 0.61 / 0.38 | 0.82 / 0.65 | 0.90 / 0.80 | 0.95 / 0.92 | 0.985 / 0.979 | 0.999 | 0.995 / 0.986 | 1.00 | 0.96 / 0.92 | 0.19 / 0.40 | 0.87 / 0.81 | 0.68 / 0.29 |
| dp | 128 | 14 | 3.1 | 0.52 | 0.72 | 0.87 | 0.92 | 0.970 | 0.992 | 0.88 | 0.99 | 0.75 | 0.21 | 0.80 | 0.17 |
| dp | 256 | 12 / 14 / 15 | 10.0 / 10.9 / 10.9 | 0.21 / 0.20 / 0.19 | 0.40 / 0.36 / 0.36 | 0.54 / 0.53 / 0.52 | 0.71 / 0.70 / 0.71 | 0.850 / 0.853 / 0.856 | 0.944 / 0.947 / 0.946 | 0.71 / 0.53 / 0.54 | 0.87 / 0.83 / 0.84 | 0.37 / 0.32 / 0.38 | 0.23 / 0.26 / 0.26 | 0.69 / 0.66 / 0.67 | 0.04 / 0.14 / 0.14 |
| age3 | 64 (s0 / s1) | 14 | 1.2 / 1.4 | 0.90 / 0.83 | 0.93 / 0.88 | 0.96 / 0.93 | 0.98 / 0.97 | 0.994 / 0.995 | 1.00 | 0.999 / 0.977 | 1.00 | 0.99 / 0.97 | 0.03 / 0.15 | 0.98 / 0.95 | 0.98 / 0.75 |
| age3 | 128 | 14 | 1.6 | 0.79 | 0.87 | 0.93 | 0.96 | 0.983 | 0.995 | 0.99 | 1.00 | 0.94 | 0.25 | 0.80 | 0.72 |

Between n = 128 and 256 at L = 16 the hub-type (dp) error moves decisively towards the k/n law: PR at l = 12–15 goes 3–4 → 10–11, fro16 0.97 → 0.85, fro32 0.99 → 0.945, ndR²₁₆ 0.95–0.97 → 0.83–0.88 (GOE null 0.19–0.24, scrambled 0.34–0.44), the oracle-innovation rank-16 ceiling D2₁₆ 0.88–0.90 → 0.53–0.71 (D2₃₂ 0.98–0.99 → 0.83–0.87), one-step span persistence 0.80–0.88 → 0.66–0.69, D4(k=16) 0.75–0.78 → 0.32–0.38, R²(loc) 0.17–0.28 → 0.04–0.14. At matched k/n the fraction falls by 0.15–0.2 per doubling (k/n = 1/64: 0.69–0.72 → 0.53; 1/32: 0.82–0.87 → 0.70), i.e. the coherent, n-independent part of this error is at most ≈ 0.2–0.3 of its energy and the rest is squeezed-incoherent content whose effective rank grows with n. The age3 error (coherent lineage footprint) stays rank 1–2 at every n.

### 6.3 The squeezed-GOE null, computed at n = 1024 (T6; `x6_squeeze.py`)

Pure geometry of the He-init chain: a unit-norm GOE innovation (zero diagonal) injected at pre-activation layer l0 and transported by W diag(g) (·) diag(g) Wᵀ with the gates of the K2 chain of the same random net; "single" = one injection at l0 = 2, "mixture" = an independent equal-norm injection at every layer (transported share of ‖δC_l‖² at l = 12–16: 0.65–0.72, close to the real chain's 0.75–0.80). fro_k = share of ‖δC_l‖_F² in the top k, R²_k = R² of the in-layer quadratic readout diag(T_k) on δvar_l.

| injection | n | l | PR | fro8 | fro16 | fro32 | fro64 | R²₁₆ | R²₃₂ | R²₆₄ |
|---|---|---|---|---|---|---|---|---|---|---|
| single (l0 = 2) | 128 | 8 / 12 / 14 / 16 | 3.8 / 2.6 / 1.8 / 1.8 | 0.96 / 0.99 / 0.99 / 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| single | 256 | 8 / 12 / 14 / 16 | 9.9 / 4.8 / 5.0 / 4.5 | 0.78 / 0.90 / 0.93 / 0.96 | 0.93 / 0.99 / 0.99 / 1.00 | 0.99 / 1.00 | 1.00 | 0.92 / 0.98 / 0.99 / 1.00 | 0.99 / 1.00 | 1.00 |
| single | 512 | 8 / 12 / 14 / 16 | 20.5 / 13.0 / 10.9 / 9.5 | 0.54 / 0.69 / 0.74 / 0.79 | 0.77 / 0.89 / 0.92 / 0.95 | 0.93 / 0.98 / 0.99 / 1.00 | 0.99 / 1.00 | 0.76 / 0.89 / 0.92 / 0.94 | 0.93 / 0.98 / 0.99 / 1.00 | 0.99 / 1.00 |
| **single** | **1024** | 8 / 12 / 14 / 16 | **42 / 25 / 20 / 15** | 0.34 / 0.47 / 0.54 / 0.62 | **0.53 / 0.70 / 0.76 / 0.82** | **0.76 / 0.89 / 0.92 / 0.95** | 0.93 / 0.98 / 0.99 / 1.00 | 0.52 / 0.70 / 0.77 / 0.81 | 0.75 / 0.89 / 0.92 / 0.95 | 0.92 / 0.98 / 0.99 / 1.00 |
| mixture | 128 | 8 / 12 / 14 / 16 | 31 / 27 / 20 / 15 | 0.43 / 0.47 / 0.55 / 0.59 | 0.61 / 0.65 / 0.71 / 0.74 | 0.79 / 0.82 / 0.86 / 0.87 | 0.94 / 0.96 / 0.96 / 0.97 | 0.64 / 0.78 / 0.77 / 0.81 | 0.74 / 0.87 / 0.87 / 0.88 | 0.91 / 0.95 / 0.95 / 0.96 |
| mixture | 256 | 8 / 12 / 14 / 16 | 74 / 44 / 39 / 41 | 0.23 / 0.34 / 0.37 / 0.36 | 0.38 / 0.51 / 0.54 / 0.53 | 0.56 / 0.69 / 0.70 / 0.70 | 0.75 / 0.84 / 0.85 / 0.84 | 0.47 / 0.63 / 0.66 / 0.68 | 0.62 / 0.78 / 0.78 / 0.79 | 0.71 / 0.85 / 0.88 / 0.87 |
| mixture | 512 | 8 / 12 / 14 / 16 | 139 / 97 / 89 / 77 | 0.14 / 0.20 / 0.21 / 0.24 | 0.25 / 0.32 / 0.34 / 0.38 | 0.39 / 0.48 / 0.51 / 0.54 | 0.57 / 0.65 / 0.68 / 0.70 | 0.29 / 0.42 / 0.39 / 0.49 | 0.50 / 0.60 / 0.56 / 0.64 | 0.63 / 0.75 / 0.75 / 0.79 |
| **mixture** | **1024** | 8 / 12 / 14 / 16 | **283 / 220 / 196 / 171** | 0.08 / 0.10 / 0.11 / 0.13 | **0.14 / 0.18 / 0.20 / 0.22** | **0.24 / 0.29 / 0.32 / 0.35** | 0.39 / 0.45 / 0.48 / 0.52 | 0.17 / 0.23 / 0.25 / 0.30 | 0.31 / 0.37 / 0.40 / 0.44 | 0.46 / 0.55 / 0.58 / 0.62 |

Plain GOE null (no transport, `x6_null.py`): fro_k depends on k/n only — 0.058 / 0.112 / 0.21 / 0.38 / 0.62 at k/n = 1/64 / 1/32 / 1/16 / 1/8 / 1/4; at n = 1024, k = 16 → 0.058, k = 32 → 0.11.

Validation of the null against the tower: at n = 128 the dp error sits between "single" (fro16 = 1.00 at l ≥ 12) and "mixture" (0.65–0.74) — measured 0.96–0.97 — because its innovations are themselves transported lineage content (PR(I) 4) rather than fresh GOE; the age3/K2 errors are above "single" because of their n-independent coherent rank.

### 6.4 Extrapolation to 1024×16

PR of every incoherent component grows ∝ n (sandbox dp at L = 8: 7 → 10 → 27; null: 1.8 → 4.5 → 9.5 → 14.7 single, 15 → 41 → 77 → 171 mixture). Only the coherent lineage/template directions (rank 2–4) keep an n-independent share. Therefore at n = 1024, l = 12–16, the share of the off-diagonal variance-error energy that a rank-k truncation can hold is:

| error composition | k = 16 | k = 32 | k = 64 |
|---|---|---|---|
| stream of fresh incoherent innovations, 65–75 % transported (mixture null) | 0.18–0.22 (R² 0.23–0.30) | 0.29–0.35 (R² 0.37–0.44) | 0.45–0.52 (R² 0.55–0.62) |
| single old (≥ 10 layers) incoherent error | 0.70–0.82 | 0.89–0.95 | 0.98–1.00 |
| lineage-like error (sandbox dp at L = 16: fro16/fro32 = 0.97/0.99 at n = 128 → 0.85/0.945 at n = 256; at matched k/n 0.69–0.72/0.82–0.87 → 0.53/0.70, falling 0.15–0.2 per doubling towards the mixture floor; D2 ceiling 0.89/0.99 → 0.6/0.85) | ≈ 0.25–0.5 (D2 ceiling ≈ 0.3–0.45) | ≈ 0.4–0.7 (D2 ≈ 0.5–0.7) | ≈ 0.65–0.85 |
| coherent template content (age-truncation, scale mode) | 0.8–0.9 (rank 2–4 suffices) | 0.9–0.95 | ≥ 0.95 |

The real chain's channel has R²(loc) ≤ 0.03 and 75–80 % transported share, so it is not the fourth row; the sandbox cannot decide between rows 1 and 3 (both are compatible with those two facts), which is exactly what the pre-check in §7 measures. The strategist's kill line "if the quadratic-readout arm beats X1-A by < 0.3 pp, the channel is not learnable at rank ≤ 32" conflates representability (rows 1–3: 20–85 %) with learnability (§4: the core is not a function of per-neuron inputs); a null result from the trainer would not distinguish them, the pre-check does.

---

## 7. Recommendations for the X6 lane

1. **Pre-check on the real chain before any training (≈ 1 h, no GPU).** On 2 FIT nets with MC covariance labels C_l^MC (16 × 8 MB per net; the twin-label variance files are not enough — the diagonal is X1's channel), compute δC_l = C_l(chain) − C_l^MC and run `x6_analysis.py`-style sections B (PR, fro_k, ndR²_k with GOE/scrambled baselines) and D2 (oracle-innovation rank-k ceiling) at l = 8, 12, 14, 16. Decision: if PR(δC_l) at l ≥ 12 is ≲ 30 and D2 at k = 32 ≥ 0.6, the channel is representable (rows 2–3 above) and the question becomes the innovation; if PR ≳ 100 and D2(k = 32) ≈ 0.3–0.4 (row 1), stop — no rank ≤ 64 state carries it. The MC covariance noise is negligible for this (noise/‖δC‖² ≤ 0.004 at n = 128 with 5e7 samples; at n = 1024 with 1e6 samples the per-entry noise σ²/√N ≈ 2e-3 must be compared with the measured rms(δC) first).
2. **If the trainer is run anyway**: (a) readout q_i = Σ_k (WU)²_ik − ((W∘W)(U∘U))_i, i.e. remove the diagonal of UUᵀ from the quadratic readout (one extra Hadamard matmul per layer) — with the diagonal included the small-k readout is erratic in the sandbox (R² −0.2 to +0.5 at k = 2–4 vs monotone 0.5–0.9 without it); (b) Φ gates, not √T2; (c) the innovation head must receive non-local information or the module collapses to X1 — the one cheap candidate consistent with the sandbox is to feed the MLP the *rows of the transported κ3-diagonal state* X1 already has (the lineage legs are the directions the core lives on), and to allow a learned layer-wise k×k mixing matrix A_l (the MLP is shared across neurons, so U ← Ũ A_l + F_l B_l is implementable); (d) k = 32 rather than 16 (the k/n law: at n = 1024 rank 32 buys 1.5× the fraction of rank 16 in every row of §6.4).
3. **The alternative the sandbox does support**: the coherent part of the missing lineage content (the "a_i w_j + w_i a_j" footprint, with a local and w = transported local vector) is cheap in closed form (O(n²) per birth given the legs: F_math §4/§7.5) and is exactly what makes the age-truncation error rank-2 per birth. If the real chain drops old lineages, carrying their coherent pair footprint analytically removes 60–85 % of that δC without any learned state. This is a different lane (X4/κ3 template) and should be priced against X6.

---

## 8. Files

* `/home/claude/sandbox/x6_chains.py` — `python3 x6_chains.py n L seed [variants]` → `results/x6_chains_n{n}_L{L}_s{seed}.npz` (per-layer μ, C, m_h, Cov_h, y; MSE vs MC if available). Variants: full, age2, age3, age4, ind3, dp, gate, k2. For n³·L·8 B > 1.5 GB the births are merged (identical chain for variants without age truncation).
* `/home/claude/sandbox/x6_analysis.py` — `python3 x6_analysis.py n L seed trunc [ref=full|mc]` → `results/x6_an_{trunc}_vs_{ref}_n{n}_L{L}_s{seed}.{json,txt}`: sections A (sizes, channel split), B (spectrum/sandwich R² with GOE & scrambled baselines), B2 (pre-activation δC), C (transport persistence, Φ and √T2), D (innovation), D2 (oracle-innovation rank-k state), D3 (local-born innovations), D4 (direction ceiling, local vs random vectors), E (diagonal channel / local-feature OLS), F (template content), G (one-step persistence), H (local and transported-local spans).
* `/home/claude/sandbox/x6_null.py` → `results/x6_null.json` (GOE fractions vs k/n); `x6_squeeze.py n L seed` → `results/x6_squeeze_n{n}_L16_s0.json` (n = 128…1024); `x6_summary.py`, `x6_tables.py` → `results/x6_tables.md` (all tables, all runs).
* Runs: n = 64 (s0, s1), 128, 256 at L = 8 (all variants, MC-validated); n = 64 (s0, s1), 128 at L = 16 (full, age3, age4, ind3, dp, k2); n = 256, L = 16 (full, dp). Every run ≤ 5 min; the n = 256, L = 16 full tower needs merged births (15 separate n³ births OOM at 7 GB).
