# X1 — frozen-chain recurrent error-transport corrector: RESULT (2026-10-20 08:20 UTC)

**Verdict: STOP (kill line missed). Not integrated, no FRESH-A read, no bill, no submission.**
Spec: `_claude_snap/handoff_2026-10-19/HANDOFF_2026-10-19_X1_transported_corrector.md`.

## Prediction vs outcome
- Pre-registered (strategist): VAL-100 paired raw **−4 %** (80 %: −1.5 … −8 %).
- Keep line: ≤ −2.0 % raw on VAL-100 with 95 % CI excluding 0, **and** FRESH-A100 ≤ −1.5 %.
- Outcome: best arm **A0 −0.84 % raw, 95 % CI [−1.17, −0.52] %, 68/100 wins**. Below the
  80 % interval; misses the keep line by 2.4×. FRESH-A gate not reached (by rule).

## VAL-100 paired table (real numpy chain, 8be, final layer, raw MSE)
Chain alone: mean raw MSE **1.53509e-8** (organizer L16 label floor ≈ 7.7e-11; paired, so
the floor cancels). CI = paired bootstrap over nets, 10k resamples, ratio of means.

| arm | seed | params | raw Δ | 95 % CI | wins | per-layer Δ L4 / L8 / L12 / L16 | forecast cost | score-equiv Δ |
|---|---|---|---|---|---|---|---|---|
| A (state 32, 3×128) | 0 | 49,697 | **−0.840 %** | [−1.167, −0.516] | 68 | −7.6 / −4.5 / −2.2 / −0.8 % | ≈3.7 GF (+1.29 %) | **+0.44 % (worse)** |
| A | 1 | 49,697 | −0.741 % | [−1.069, −0.414] | 67 | −7.6 / −4.4 / −2.0 / −0.7 % | +1.29 % | +0.54 % (worse) |
| B (state 1, 2×64) | 0 | 6,530 | −0.329 % | [−0.608, −0.044] | 56 | −4.8 / −3.1 / −1.5 / −0.3 % | ≈0.32 GF (+0.11 %) | −0.22 % |
| B | 1 | 6,530 | −0.512 % | [−0.798, −0.225] | 64 | −4.7 / −2.9 / −1.5 / −0.5 % | +0.11 % | −0.40 % |
| C (no learned state) | 0 | 37,377 | −0.476 % | [−0.712, −0.242] | 65 | −6.2 / −3.1 / −1.4 / −0.5 % | ≈1.27 GF (+0.44 %) | −0.04 % |
| C | 1 | 37,377 | −0.446 % | [−0.689, −0.206] | 62 | −6.3 / −3.0 / −1.4 / −0.4 % | +0.44 % | −0.01 % |

Cost column is an **arithmetic forecast** (MLP MACs + state/u matvecs + W∘W, 16 layers,
vs the 8be bill 286,868,957,368 F), **not** a real-flopscope bill — none was taken because
no arm passed. Score-equiv = (1+raw)(1+cost)−1 at λ = 0. Ops and wall not measured.
Internal TRAIN hold-out (rows 850–899) agreed with VAL: A −0.79 %, C −0.63 %, B −0.19/−0.39 %.

## What was learned (for the write-up)
1. **The transport premise is right.** On TRAIN nets the true per-layer error d_l is
   explained by Φ(α_l)∘(W_lᵀ d_{l−1}) with gain 1.00 ± 0.03 and R² ≈ 0.45–0.5 at
   layers 2–4 rising to **0.74–0.81 at layers 11–15** (label noise dominates layer 1).
   So ~75 % of late-layer error variance is transported, ~25 % is new innovation.
2. **The module cannot predict the late innovations from these inputs.** Its per-layer
   gain is large mid-chain (−6…−8 % at L4) and decays to < 1 % at L16. It learns the
   early/mid errors (where the chain's in-loop corrector leaves structure), but those are
   a small part of what reaches layer 16; the late-layer innovations remain unexplained.
3. **Learned transported state (A vs C) adds ≈ 0.35 pp raw**, but costs ≈ 2.4 GF, so on
   the score it is a net loss. The cheapest arm (B) is the only one with a clearly
   negative score-equivalent (−0.2…−0.4 %), far too small to justify integration risk
   on the last upload day.
4. The public-team gains (−5.6 … −17 %) do not transfer to our chain at this input set:
   our in-loop corrector already removes most of the locally predictable mean error.

## Deviations from the spec (all disclosed; none tuned on VAL)
- Compute: all local via workerq (cupy shim dump 1 s/net, training 12 min on the 5090),
  not Kaggle — faster than setting up a kernel inside the 3 h clock.
- Dumps: TRAIN from the **cupy shim** of the exact 8be source (not the torch twin);
  VAL-100 from the **real numpy shim chain**, so the VAL read is the real-chain read
  (the module is post hoc and never feeds back into the chain).
- Data: TRAIN rows 0–849 for fitting, 850–899 for early stopping; **no d8b 0–2047**
  (needs the Torch-2.8 weight generator). VAL read exactly once per model.
- Features: 14 per neuron (μ, σ, α, Φ, φ, D3/σ³, D4/σ⁴, ‖W col‖, d21e, b_l, in-loop
  correction, b_l−E1, E2−E1², log σ) + transported u/state + layer one-hot, not 35.
- Training: 10 epochs (spec 30), Adam 5e-4 with 100-step warmup + cosine, output scale
  0.1 and tanh-bounded state (needed for stability: the unbounded recurrence diverged
  in smoke tests). Loss and layer weights as specified (w_16 = 4).
- Corrector-off chain + module: not run (stopped first).

## Files
- `dump.py` (hook recorder), `train.py`, `eval_val.py`, `estimator_8be.py` (sha 8be40031…).
- `run1/summary.json`, `run1/eval_val_900_1000.json`, `run1/norm.pt`, `run1/*.pt`, `run1/train.log`.
- Raw trajectories (not committed, ~0.66 MB/net): `cache/x1/{train,val}/netNNNN.npz`.
- Jobs: dumps 3507/3512 (TRAIN), 3518/3520/3526/3534 (VAL); train 3536; VAL read 3543
  (3542 computed the same read and failed only at JSON save, no result printed).
  FRESH-A dumps 3537/3538 cancelled once the stop was decided.
- No exported constants file (nothing kept).
