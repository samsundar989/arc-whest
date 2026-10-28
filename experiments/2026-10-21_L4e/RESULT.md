# L4e — where the κ4 gain goes: young-tree fidelity on the real chain vs exact labels (A4 export, FIT 0–9, layers 6–15)

Handoff: `_claude_snap/handoff_2026-10-19/HANDOFF_2026-10-21_L4_K4_TREE_FIDELITY.md` (L4 + L1). Started 18:34 UTC Oct 21.
Split: **L4** — step 1 on A4 (`fidelity_a4.py`), step 2(iii) the exact-Dx ceiling (`mc_dx.py`, `exact_tree.py`), the attribution
table. **L1** — the chain dump with served and clean (unfolded) Dx (`dump_clean.py` → `cache/l4e_clean/`), the tree rebuild from the
chain's and the clean Dx (`tree.py`), and the price (`price.py`); L1's section is below under its own heading.
Nothing here ships tonight.

## 0. Definitions (common to every row)
- **Label:** `cache/twin_labels/fit_g.npz` `dG` = κ4_ii / 2 (metric 2), N = 1e9; FIT net g = full row g. Chain counterpart: the served
  `D4 / 2` (what the closure reads); "transport dG" = the chain's `dG` before any law/tree term.
- **Weight:** s_i = ((α_i² − 1) φ(α_i) / σ_i³)² at the exact α, σ (the D4 term of E[relu]). **R²w** = 1 − Σ s (y − ŷ)² / Σ s (y − ȳ_s)²,
  pooled over nets per layer (L1's `fidelity.py` definition, identical in `fidelity_a4.py`). "at gain" = the best single coefficient
  on the young part, to separate shape from scale.
- **Young trees** (A4 `_yp_contract`, coefficient 1, scattered exactly as served): yp = 12 Σ_j (DxH·ratio) M A (the young (2,1,1)/3+2
  term, served × `_YP_D4_GAIN`); tp2 = 12 Σ_j DxH·srh·M² (U1 age-1 tp2, served × `_YP_TP_GAIN`).
  Arms: noyoung = served D4 − served young terms; shipped = served D4; chain1/clean1/exact1 = noyoung + (yp + tp2) from the chain's
  Dx / the unfolded Dx / the exact Dx.
- **Exact Dx (2(iii)).** The source defines Dx = w1t_j · Dxc, Dxc(x, j) = κ3_pre(e_j; t_x, t_x): the layer-(l−1) pre-activation κ3
  with one leg on neuron j and two on the transported row t_x[k] = W_l[k, x] Φ(α_k). `mc_dx.py` computes it with no closure: true
  forward MC on the GPU, z̃ = z_{l−1} − μ_label, u = (Φ ∘ z̃) W_l, K[j, x] = κ3(z̃_j, u_x, u_x) (centred-moment formula), N = 2²⁵,
  Φ from the exact α. Smoke (#3697, N = 2²⁰, FIT 0): MC means/vars at all 16 layers match the labels to 0.005 σ / 0.5 %
  (convention and layer indexing confirmed). `exact_tree.py` takes DxH_exact(x, h) = Φ_j K[j, x] at the reader's rows (kr through
  `_orow`) and, per newborn H column, the neuron j whose exact column best matches the chain's DxH column (match quality reported).
- **Caveat on "exact".** This is the exact *input* the shipped tree formulas are written for (the transport t_x is the chain's
  first-order one). It is not the census's exact-κ4 tree catalogue (that script is not in the repo; it ran on the strategist VM).

## 1. Attribution table (19:50 UTC) — R²w vs the dG label, pooled over FIT 0–9, K1 rows (the rows the young reader forms)
Sources: step 1 `fidelity_a4.py` → `cache/l4e_clean/fidelity_a4.json` (transport dG and Tier-law columns are pooled over **all** rows);
L1 `tree.py` → `tree_a4.json` (clean1); L4 `exact_tree.py` → `exact_tree.json` (shipped, no young, chain1, exact1). chain1 here is
`exact_tree.py`'s own rebuild with `tree.py`'s formulas and scatter; it reproduces L1's chain1 (served `_YP_D4` sanity in `tree_a4.json`).

| layer | transport dG | Tier law | **shipped D4/2** | no young | chain1 (coef 1) | clean1 (coef 1) | exact1 (coef 1) | chain1 @ best gain (g) | clean1 @ gain | exact1 @ gain (g) | coefficient cost = chain1@g − shipped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 0.338 | 0.664 | **0.779** | 0.614 | 0.646 | 0.645 | 0.634 | 0.807 (0.52) | 0.807 | 0.808 (0.51) | +0.028 |
| 7 | 0.416 | 0.719 | **0.728** | 0.596 | 0.752 | 0.750 | 0.741 | 0.831 (0.63) | 0.831 | 0.831 (0.62) | +0.103 |
| 8 | 0.394 | 0.670 | **0.780** | 0.634 | 0.660 | 0.659 | 0.638 | 0.780 (0.52) | 0.780 | 0.780 (0.50) | +0.000 |
| 9 | 0.388 | 0.625 | **0.693** | 0.528 | 0.771 | 0.772 | 0.764 | 0.795 (0.77) | 0.795 | 0.796 (0.74) | +0.103 |
| 10 | 0.421 | 0.636 | **0.780** | 0.650 | 0.717 | 0.717 | 0.707 | 0.781 (0.59) | 0.781 | 0.782 (0.57) | +0.002 |
| 11 | 0.398 | 0.637 | **0.623** | 0.433 | 0.785 | 0.786 | 0.789 | 0.788 (1.08) | 0.788 | 0.790 (1.04) | +0.164 |
| 12 | 0.453 | 0.664 | **0.767** | 0.674 | 0.715 | 0.716 | 0.706 | 0.768 (0.57) | 0.769 | 0.769 (0.55) | +0.001 |
| 13 | 0.398 | 0.620 | **0.730** | 0.610 | 0.751 | 0.752 | 0.748 | 0.767 (0.76) | 0.767 | 0.768 (0.74) | +0.037 |
| 14 | 0.419 | 0.661 | **0.758** | 0.674 | 0.720 | 0.721 | 0.716 | 0.755 (0.60) | 0.755 | 0.757 (0.59) | -0.003 |
| 15 | 0.475 | 0.639 | **0.724** | 0.668 | 0.641 | 0.641 | 0.632 | 0.705 (0.43) | 0.705 | 0.706 (0.42) | -0.019 |

Exact-Dx check (`exact_tree.json` → `match`): chain DxH vs exact DxH per live column, uncentred correlation median **0.996–0.998**
(L6 → L15), chain/exact scale **0.966 → 0.938**; the H ids from the chain's own rule agree with the best-correlation match on 100 % of
live columns. MC: N = 2²⁵ per net (#3700, 105 s/net), forward sanity mean ≤ 0.0008 σ, var ≤ 0.11 % vs labels on all 10 nets.
(First run at 19:40 used the best-match column also for the chain's zero-pad H columns, which then took neuron 0's exact column and
produced exact1 R²w down to −2.0; replaced by the rule-derived ids at 19:45. Pad columns carry < 1 % of the exact norm.)

### Reading — which stage loses the κ4 gain
- **Dx fidelity (κ3 deficit) — ≈ 0.** The exact young-reader input (MC, no closure) and the chain's Dx agree to 0.997 per column; the
  trees built from them differ by ≤ 0.02 R²w at coefficient 1 (exact slightly *lower*: its 3–6 % larger scale overshoots further) and by
  ≤ 0.002 at the best gain. **The κ4 channel is not limited by its κ3 inputs.**
- **Dressing / quartic fold — ≈ 0** (L1: clean − chain within ±0.001; the fold moves Dx 1.9–2.8 %). A clean-Dx reader buys nothing and
  would cost +24.1 % F (L1's price) — **not a build.**
- **Coefficients — the only recoverable piece measured here: 0 … +0.16 R²w.** With the base law fixed, the best single young gain is
  0.42–0.77 (1.04–1.08 at L11), not 1: the fitted Tier law and transport already carry part of what the trees add, so at coefficient 1
  the trees overshoot (chain1 < shipped at L6, L8, L10, L12, L14, L15). Against the shipped gains, the best joint gain gains +0.10
  (L7, L9) and +0.16 (L11); elsewhere ≈ 0. These are in-sample on the same 10 nets (one coefficient per layer, so optimism is small).
- **Readout / coverage — the rest.** Even with exact inputs and the best coefficient the trees reach R²w 0.71–0.83; the census's exact-κ4
  tree catalogue closes D4 at 0.83–0.86. The residual 0.03–0.15 sits in trees A4 does not read (ages ≥ 2 beyond age-1 tp2, 3+2 at older
  ages, the old tier) — a coverage/readout limit, not an input-fidelity one.
  *Caveat:* 0.83–0.86 is the census's number as quoted in the handoff (script not in the repo; its weight is not confirmed to be
  this one), so this residual is indicative, not a paired measurement.
- **What it means for the 0.797× D4 channel:** the measured recoverable part on A4 is the coefficient piece (≤ +0.16 R²w at three layers,
  ≈ 0 at the rest), reachable by a refit at 0 F (the U1/NEW10 gains already exist), and the coverage piece needs new trees whose price
  is the open question (U2b: +3.4 % F for ages 2–3). The "5–10× less realised than the census implies" is **not** explained by Dx
  fidelity; it is explained by (i) the trees' content overlapping the fitted law (best gain ≈ 0.5, i.e. half the tree is already in the
  law) and (ii) coverage. A clean-Dx tree reader is **not** the post-deadline build.


## L1 — clean (unfolded) Dx and the price of a clean-Dx reader

### Price (step 4) — real meter, one net (`price.py`, workerq #3699, `price_a4_fit0.json`)
A clean-Dx reader keeps the served Dx for D21 and forms beside it Dx_clean = Dx + [ΔLA | ΔLP] @ [pre.A_L | pre.M]ᵀ (Δ = the K9.2
fold's change to the recent left factors), then reads yp and tp2 from it. The only new matmul has exactly the shapes of the served
young-tier recent block, so it is billed by running a same-shape product (+ two K1 × h reads) inside every young-reader call.
A4 export 67ed6090, FIT 0, f64 read-only, flopscope **0.12.1** (project `.venv`; the absolute bill is 0.12.1's, the Δ is a same-meter
ratio): served 288,310,814,515 F / 22,976 ops → **ΔF = +69,571,264,442 F = +24.1 % of the bill (+3.16 % of B), +1,556 ops**;
predictions bit-identical (measurement does not perturb the chain).

| layer (1-based) | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ΔF (GF) | 2.11 | 4.34 | 5.54 | 6.85 | 7.37 | 7.23 | 6.54 | 5.89 | 5.23 | 4.81 | 4.25 | 4.37 | 5.04 |
| recent block K1 × inner (A / P) | 1024 × 1024 | 992 × 2048 | 928 × 3072 | 896 × 4064 | 832 × 4992 / 5056 | 800 × 4864 / 4992 | 768 × 4672 / 4896 | 736 × 4448 / 4800 | 704 × 4224 / 4704 | 640 × 4032 / 4576 | 608 × 3840 / 4480 | 576 × 4480 / 5312 | 576 × 5088 / 6144 |

Why it is expensive: the fold rewrites the left factors of **every** recent lineage (ages ≤ 6, inner width 4–6k), not only the
newborn's, so un-folding Dx re-does the whole young-tier product. At λ = 0 score ∝ F, so a clean-Dx reader at this price needs a raw
gain ≈ −19 % (1 − 1/1.241) just to break even; restricting it to L ≥ 9 still costs +36.1 GF (+12.5 %, break-even raw ≈ −11 %).

### Clean vs chain Dx (step 2 (i)/(ii)) — `dump_clean.py` (workerq #3698) → `tree.py` → `tree_a4.json`
A4 export 67ed6090, FIT 0–9 (10 nets, 70–132 s each), real numpy chain f64 read-only. `--verify`: the wrapped run (clean pass +
served pass) gives predictions **bit-identical** to an unwrapped run (net 0). Sanity: yp rebuilt from the recorded Dx = served
`_YP_D4` to 5.4e-7 relative (max over nets × layers); the recovered tp2 gains 0.165–0.297 = `_YP_TP_GAIN` (mean-field off).
**The fold changes Dx by only 1.9–2.8 %** (‖Dx − Dx_clean‖ / ‖Dx_clean‖, every net, L6–15).

R²w on the rows the young reader forms (K1 rows; all-rows values within 0.01), pooled over the 10 nets. "gain" = best single
coefficient on (yp + tp2) jointly (on the served young part for `shipped`):

| layer | n | noyoung | shipped | shipped at gain (g) | chain1 (coef 1) | chain1 at gain (g) | clean1 (coef 1) | clean1 at gain | clean − chain |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|
| 6 | 8,960 | 0.614 | 0.779 | 0.782 (1.14) | 0.646 | 0.807 (0.52) | 0.645 | 0.807 | −0.001 |
| 7 | 8,320 | 0.596 | 0.728 | 0.730 (1.13) | 0.752 | 0.831 (0.63) | 0.750 | 0.831 | −0.001 |
| 8 | 8,000 | 0.634 | 0.780 | 0.783 (1.17) | 0.660 | 0.780 (0.52) | 0.659 | 0.780 | −0.001 |
| 9 | 7,680 | 0.528 | 0.693 | 0.701 (1.28) | 0.771 | 0.795 (0.77) | 0.772 | 0.795 | +0.000 |
| 10 | 7,360 | 0.650 | 0.780 | 0.782 (1.17) | 0.717 | 0.781 (0.59) | 0.717 | 0.781 | +0.000 |
| 11 | 7,040 | 0.433 | 0.623 | 0.645 (1.46) | 0.785 | 0.788 (1.08) | 0.786 | 0.788 | +0.000 |
| 12 | 6,400 | 0.674 | 0.767 | 0.768 (1.10) | 0.715 | 0.768 (0.57) | 0.716 | 0.769 | +0.001 |
| 13 | 6,080 | 0.610 | 0.730 | 0.734 (1.21) | 0.751 | 0.767 (0.76) | 0.752 | 0.767 | +0.001 |
| 14 | 5,760 | 0.674 | 0.758 | 0.758 (1.06) | 0.720 | 0.755 (0.60) | 0.721 | 0.755 | +0.001 |
| 15 | 5,760 | 0.668 | 0.724 | 0.726 (0.83) | 0.641 | 0.705 (0.43) | 0.641 | 0.705 | +0.001 |

**Readings (L1).**
1. **Dressing / fold cost = 0.** Clean − chain is within ±0.001 R²w at every layer, at coefficient 1 and at the best gain. The fold
   moves Dx by ~2 %, and that 2 % is not where the κ4 diagonal's error is. With the price above (+24.1 % F), **a clean-Dx reader is
   not a post-deadline build.**
2. **The young trees do carry D4 content.** noyoung → young trees at the best gain: +0.10…+0.36 R²w (largest at L9, L11 — exactly
   the layers where the served `noyoung` is weakest, 0.53 / 0.43).
3. **Coefficient structure, not inputs, separates shipped from chain1-at-gain.** The served weights (yp × `_YP_D4_GAIN` 0.78–1.03,
   tp2 × `_YP_TP_GAIN` 0.16–0.33) score ≥ a single joint gain on (yp + tp2) at L6, 8, 10, 12, 14, 15, and **lose to it by +0.09
   (L7), +0.10 (L9), +0.14 (L11), +0.03 (L13)**. The joint coefficient-1 trees overshoot (best g 0.43–0.77 except L11 1.08), while
   the served part undershoots (best g on it 1.06–1.46). This is a zero-FLOP reweighting question (yp vs tp2 balance per layer), but
   R²w is a κ4-diagonal fidelity, not a score: the served gains were fitted end-to-end on VAL through the closure and corrector,
   so a re-fit on dG labels must be screened on VAL before it means anything.
4. Remaining question for L4's exact arm (2(iii)): whether yp/tp2 from the *exact* Dx close the remaining gap to the census ceiling
   (0.83–0.86). Our chain-Dx trees at their best gain reach 0.71–0.83 on these rows.
