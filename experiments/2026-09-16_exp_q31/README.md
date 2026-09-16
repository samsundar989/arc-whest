# exp_q31 — knob-by-knob ablation of 504aldo's v17 κ4 channel (2026-09-16, small-shape proxy)

Run in the Cowork cloud container, plain numpy, no metering. 504aldo's `estimator_v16b.py` and `estimator_v17.py` (MIT, local oracle only) run unmodified through shims except for four tagged edits (`v17_open_knobs.diff`): a `V17_FORCE_RIDERS` env to open the `n==1024 and L==16` gate off-suite, and two split knobs `LAM_DIAG` (λ inside the dG transport) / `LAM_OFF` (λ in `wk431` and the `c1_b` feed). Their own kill-switches `V17_NO_FEED`, `V17_NO_WK431`, `V17_NO_REGEN`, `V17_LAM_SCALE` used as-is.

Truth: 8 He-init MLPs per shape (seeds 0–7, float32 weights, `x@W`), 4e6 Gaussian inputs, float64; noise floor 4.2–4.3e-8 (≤1% of every MSE). Shapes 64×8 and 128×8. Their shipped λ table is under-scaled off-suite (optimum ≈×10 at n=128, ≈×22 at n=64; λ_opt ∝ ~n⁻¹·¹), so the decisive rows use tuned λ.

Result (gain over the no-regen control at tuned λ; n=128 | n=64): exact diagonal core at λ=0 ×1.04 | ×1.05; diagonal feed at λ=0 ×1.01 | ×1.03; **λ inside `dG = (W∘W)@(g_prev − var_prev·λ) + var·λ`, consumed only via the (4)/(2,2) rows: ×2.32 | ×3.95**; λ-carrying off-diagonal terms (wk431 + c1_b) ×1.11 | ×1.18, of which the (3,1) consumer wk431 alone ×1.00 | ×1.09. Ordering identical on all 8 seeds. Per-layer MSE: the λ effect is absent at layers 2–3 and grows with depth (transport effect).

Caveat: F68 (504aldo, 1024×16 lean chain) says dropping the feed costs 4× there; on this proxy it is ≤×1.14. The feed's share likely grows with n. Decisive versions of these ablations belong at 1024×16 (D2/D3 in `STRATEGIST_RESPONSE_2026-09-16.md`).

Files: `results_all.json` (every arm: final MSE, per-seed, per-layer, env), `run_arms.py`, `gen_truth.py`, `v17_open_knobs.diff`. MLP/truth `.npz` and shims remain in the session scratchpad.
