# F — Fact verification (ARC White-Box Estimation Challenge 2026, Phase 2)

Verified 2026-09-24. Method: discourse `/t/<id>.json`, `/user_actions.json`, and `/c/.../l/latest.json` endpoints, HF `raw/main/README.md` cards, the AIcrowd leaderboard HTML, and GitHub raw files, all via WebFetch (direct curl to discourse/youtube is blocked by the egress proxy; `search.json` and GitHub `/tree/` are robots-blocked; YouTube watch pages returned 429; the AIcrowd rules and submissions pages are JS-rendered and unreadable). Quotes below are as returned by the fetch layer; where it paraphrased rather than quoted I say so.

---

## 1. keenanpepper's public HF datasets — CONFIRMED (with corrections)

Full inventory from `https://huggingface.co/api/datasets?author=keenanpepper`. Fields verbatim from each card.

### Phase 2 (1024 wide × 16 deep, He init, no bias, x ~ N(0, I_1024))

| HF repo id | nets | N / net | what it holds |
|---|---|---|---|
| `keenanpepper/whest-p2-bakev2-d8b-sketch-g00` (2026-09-13, 1,242 dl) | 2,048 random nets, family `d8b`, indices 0–2047, weights regenerable: `W = torch.randn(16,1024,1024, generator=torch.Generator("cpu").manual_seed(770000+idx)) * sqrt(2/1024)` (fp32), `w_sha256` check | 1e8 | sketch tier only (no dense blocks) |
| `keenanpepper/whest-p2-bakev2-bench-supp` (2026-09-13, 80 dl) | 100 nets = "bench 0..99 (aicrowd config full, rows 0–99)" | 1e8 | supp tier: marg + sketches + **dense 1024×1024 blocks** |
| `keenanpepper/whest-p2-bakev2-mini-supp` (2026-09-13, 167 dl) | 100 nets = aicrowd split `mini` (disjoint from full) | 1e8 | supp tier, same as above |
| `keenanpepper/arc-whestbench-p2-full1000-N1e9` (2026-09-13, 1,060 dl) | all 1000 nets of aicrowd config `full`, rev `v2-phase2`, "taken verbatim from the aicrowd dataset row `global_index`" | **1e9** | raw marginal moments to 4th order + **dense pairwise** |
| `keenanpepper/arc-whestbench-p2-higher-moments-2026` (2026-08-25) | 100 `mini`-split nets | 1e8 | same field set as full1000 (~322 MB/net) + `cache/` feature files |
| `keenanpepper/arc-whestbench-p2-d8b-corpus-14k` (2026-08-31) | 14,048 (2,048 + 12,000 ext.), seed `W_SEED0 = 770000`, `manual_seed(W_SEED0 + idx)` | 1e8 | post-ReLU marginal `moments` (128,4,16,1024) m1..m4 per shard, `final_mean` (128,1024), "teacher cumulant state" `k4d` (κ4 diag), `k22_u`/`k22_v` (rank-8 pairwise factors), `noise_rep_final` (~9.0e-10 floor). Weights not stored. |

**bakev2 `marg.npz` fields (all three bakev2 repos), verbatim from card:**
- `pre_shift, post_shift | (16,1024) | f32 | pilot shifts c for z / a`
- `pre_s, post_s | (6,16,1024) | f64 | s[p-1] = E[(z−c)^p], p = 1..6`
- `gate_p | (16,1024) | f64 | P(z > 0)`
- `pre_mean, pre_var, post_mean, post_var` (16,1024) f64 (derived), `gt_mean` (16,1024) f32 ("aicrowd reference means")
- plus metadata (seeds, sample counts, weight checksums, hardware).

So: **yes, per-layer per-neuron PRE- and POST-activation statistics to 6th order for both z and a=relu(z)** — stored as shifted central moments; κ1..κ6 are obtained via the card's `cumulants_from_central`. Gate probabilities `gate_p = P(z>0)` per neuron are included.

**bakev2 `layer_LL.npz`:**
- Dense (supp tier only, (1024,1024) f32): `post_K21`, `post_K22`, `gate_GG` ("P(z_i>0, z_j>0)"), `gate_GX`. (The fetch layer described `gate_GX` as a gate×activation cross statistic; I could not confirm its exact definition verbatim.) Pre-activation dense blocks are *not* in bakev2 — the card points to `arc-whestbench-p2-full1000-N1e9` for those.
- Single Ω-sketches (all tiers, (1024,128) f32): `pre_K21_O, pre_K21T_O, pre_K31_O, pre_K31T_O, pre_K22_O, gate_GX_O`; optional `pre_K11_O, post_K21_O, post_K21T_O`. Ω_l = G/√128 with `G ~ N(0,1)` (1024,128) from a CPU generator; `K21_O[i,k] = Σ_j K21[i,j] Ω[j,k]`; "E[ΩΩᵀ] = I, so ‖AΩ‖²_F estimates ‖A‖²_F".
- Double sketches ((1024,32,32) f32): `pre_K3ad_PP`, `post_K3ad_PPp` = "Σ_{j≠k, j≠i, k≠i} κ(z_i,z_j,z_k) Ψ_ja Ψ_kb", Ψ = 2Ω[:,:32] (all-distinct third cumulants).
- Noise: "fp32 storage everywhere (fp16 would eat the 1e-4 relative MC floor at N = 1e8)"; worst relative noise ~0.36 (`pre_K31_O`, early layers), ~0.02–0.04 by layer 15. Round-1 cost "≈ 520 GPU-h, ≈ 470 GB" on H200.

**`arc-whestbench-p2-full1000-N1e9` fields, verbatim:** `pre_mean, pre_m2, pre_m3, pre_m4` (16,1024) f64 = "raw marginal moments `E[z^p]`, p = 1..4, of the pre-activations"; `mean, post_m2, post_m3, post_m4` (16,1024) f64 = "raw marginal moments `E[a^p]` of the post-activations"; `pre_M11` "E[z_i z_j]", `pre_M21` "E[z_i² z_j]", `pre_M31` "E[z_i³ z_j]", `pre_M22` "E[z_i² z_j²] (symmetric)", `M11` post-activation E[a_i a_j] — all **(16,1024,1024) f32 dense**; `gt_mean` (16,1024) f32; `global_index, mlp_name, n_samples, sample_seed`. "one uncompressed np.savez per network, 336.7 MB" (so ≈337 GB total). Seed `20260824 ^ ((idx · 2654435761) mod 2⁶³)`, Philox, "24 × H200". Card calls it "the high-power replacement for the 100-network `mini` moment sets". No explicit gate probabilities and no K21/K22 cumulant slices — but K21/K22 for pre-activations are derivable from `pre_M21/pre_M22/pre_M11` and marginals.

### Phase 1 (256 × 32)
- `keenanpepper/whestbench-relu-mlp-means-500k` (2026-07-30): "500,000 train + 2,000 low-noise validation bias-free ReLU MLPs (width 256, depth 32, He init, input x ~ N(0, I))"; targets from "scrambled-Sobol QMC pass (fp64 forward)", K=2²¹ train / 2²⁵ val. Files: `alm_*` post-ReLU means f32 (targets), `mu_*`, `sig_*` pre-activation mean/std f16, `g3_*`, `g4_*` standardized pre-activation skew / excess kurtosis f16, `wfp_*` weight fingerprints, `seeds_*`. Per-layer, no pairwise, no post-activation higher moments, no gate probs.
- `keenanpepper/whestbench-relu-mlp-moments-10k` (2026-07-01): "10,500 random ReLU MLPs (10,000 train + 500 held-out test)", "N = 1,000,000 input samples per MLP (fp32 matmuls, fp64 accumulation)"; arrays `pre_mean, pre_var, pre_k3, pre_k4, post_mean, post_var, post_k3, post_k4` (N,32,256) plus weights (N,32,256,256). 
- `keenanpepper/whestbench-relu-mlp-jointfeats-10k` (joint pairwise / K3 objects, companion), `keenanpepper/arc-whestbench-higher-moments-2026` (1000 Phase-1 benchmark nets), `keenanpepper/whest-k3-tensors-2026` (full post-activation third-cumulant tensors for 10 nets).

### Is training on external MC data allowed? — CONFIRMED (allowed, but discounted for the algorithmic prize)
Townhall summary, `https://discourse.aicrowd.com/t/18078` (aicrowd_team, 2026-07-17), verbatim:
> "Offline training on external Monte Carlo data (e.g. published MLP moment datasets) is within the rules and untracked, but is seen as less likely to earn the algorithmic contribution prize than a more theoretical/mechanistic approach."

and in the Q&A section:
> "Training estimators offline on external Monte Carlo data (including publicly shared MLP moment datasets) is within the rules. There's no way to track offline compute, so it doesn't count toward anything computationally, but a more theoretical/mechanistic approach is seen as more likely to earn the algorithmic contribution prize than a purely learned/black-box estimator."

Phase 2 launch thread `/t/18197` (mohanty, 2026-08-23): "Data files remain permitted, including weights, lookup tables, and precomputed artifacts. … Where the Sponsor considers that a bundled file is not genuinely data, the Sponsor may decline to treat it as data and may disqualify the submission." Prize guidelines `/t/18041` (mohanty, 2026-06-23): methods "relying heavily on sampling, fine-tuned constants, careful performance optimization, and opaque LLM-optimized code" are less preferred (fetch-layer paraphrase).

---

## 2. "jamespayor's" Phase 1 write-up — PARTLY (wrong attribution)

**The author of `/t/18097` is `jamesrahenry` (display name "James Henry"), not `jamespayor`.** Title: "Phase 1 write-up: Stabilizing cumulant propagation at depth 32 — a trajectory-calibrated moment chain, with an error budget (submission #314695)", created 2026-07-24, 4 posts, all by jamesrahenry. The Substack is `https://jamesrahenry.substack.com/p/predicting-a-network-without-running` ("Predicting a Network Without Running It", James Henry, 2026-08-18, announced in `/t/18187`), written explicitly for non-technical friends; it does not describe his own method beyond the general landscape (analytic ≈ sampling parity at 256×32, Kerdock 66,048-direction design "within 0.023% of the best possible rule", the ~0.992 mystery constant), and says only that "ARC is much more interested in the algorithmic contributions for Phase 2 than in Phase 1's raw leaderboard".

`jamespayor` is a separate forum user (James Payor; account created 2021-08-17, last posted 2026-07-06, last seen 2026-09-24). His only forum posts are flopscope cost-model feedback in `/t/18025` (Jul 6: input-sparsity savings, complex64 billing) and `/t/17981` (Jun 4–23: smoke-test budget, wall-time penalty, "final layer MSE" column bug). He has posted nothing about Phase 2 on the forum or Substack. On the Phase 2 leaderboard he is **rank 70, adjusted 6.3e-9, Final Layer MSE 1.99e-8, 24 entries, last submission Thu 24 Sep 2026 03:17**. The claim "raw 1.56e-8 at 0.77× budget" is NOT FOUND — the submissions page is JS-rendered; note 1.56e-8 is marius_binner's displayed raw MSE (rank 3).

**What the method actually is** (WRITEUP.md in `https://github.com/jamesrahenry/arc-whitebox-replication`, MIT):
- Chain state: "(mean, covariance), propagated by a Hermite-series bivariate Gaussian ReLU closure" (K=10 Hermite terms), i.e. a k=2 closure.
- Per-layer linear correction: "(m̂, Ŝ) + per-layer linear corrector on those features ──► (μ, Σ) for l+1". Features: "8 mean + 6 variance + 2 off-diagonal features" — those are the **16 correction slots**; 16 × 32 layers = 512 coefficients ("its stabilizer coefficient table: 512 doubles indexed by layer number").
- Rolled-forward fitting: "Fitting is sequential per layer (DAgger-style): fit layer l's coefficients on training states rolled forward with corrections 0..l−1 already applied." Why not fit on true cumulants: "Step-local accuracy does not compose in an error-compensating dynamical system" — the chain's covariance errors partially cancel its mean bias, so correcting covariance alone "makes the estimate worse". One-step closure with true inputs: 1.16e-6 (vs sampling 1.61e-6) but chained: "6.0e-5 (52×)".
- Probe: "MC probe (once per net, N=4096, ~6% of the FLOP budget): x ~ N(0, I) forwarded through all 32 layers", measuring per layer "κ3(l), κ4(l), and the pair field E[zc²ᵢ zcⱼ](l)".
- Numbers: "7.4e-6 held-out at seed-resampled expectation (5.2e-6 on the fixed development probe seeds)", "~8.4× better than the plain-kprop baseline"; graded raw 5.89e-6 (adjusted 1.77e-6 after flopscope 0.10.0 float64 repricing, multiplier 0.300).
- **State-keyed corrections** (post 3, 2026-08-11; `post_phase1_state_keyed/`): "I refit the same 16 correction slots as a function of measured per-layer state — participation ratio of the propagated covariance (…), the ReLU operating point (mean/std of μ/σ), and off-diagonal correlation mass — with no layer index anywhere and 128 parameters instead of 512." State: "q = PR(Σ_pre)/w (participation-ratio fraction, computed as tr²/‖·‖²_F, no eigendecomposition), mean/std of a = μ/σ, and mean off-diagonal |ρ|". Same "sequential fit-on-own-trajectory structure"; a pooled fit "diverges within two iterations". Results: depth-32 held-out 4.74e-6 vs 5.21e-6 (5/10 wins); depth 48: 5.85e-6 vs 8.63e-6, depth 64 parity, because "the chain's state trajectory saturates past roughly L30"; 200-net tail test 6.92e-6 vs 7.60e-6 (125/200 wins), "1.20× improvement in the p90+ tail"; post 4: "state-keyed corrections improve the intrinsic tail 1.32× vs 1.07× in the body".
- Phase 2: jamesrahenry is rank 77 (adj 7.0e-9, MSE 2.18e-8, 15 entries, last Sep 22 21:56) and posted in `/t/18218` on Sep 13 (see §5). No Phase 2 write-up from him yet.

---

## 3. Luna / mliston / suliman_tadros / a_s6 / dipdecir / jacob_stavrianos — PARTLY

Leaderboard (`https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards`, read 2026-09-24; columns Adjusted / Final Layer MSE / Entries / Last):

| rank | name | adj | raw MSE | entries | last |
|---|---|---|---|---|---|
| 1 | J2W (team: `joe_wanza`, 1/5 members) | 2.0e-9 | 1.61e-8 | 327 | Sep 24 08:55 |
| 2 | suliman_tadros | 2.1e-9 | 1.59e-8 | 53 | Sep 24 19:57 |
| 3 | marius_binner | 2.3e-9 | 1.56e-8 | 158 | Sep 23 06:46 |
| 4 | a_s6 | 2.5e-9 | 2.00e-8 | 108 | Sep 21 13:55 |
| 5 | **Luna** | 2.6e-9 | 1.67e-8 | 7 | Sep 24 03:32 |
| 6 | Puffi / reds | 2.8e-9 | 1.94e-8 / 1.73e-8 | 78 / 60 | Sep 8 / Sep 21 |
| 8 | mliston | 2.9e-9 | 1.77e-8 | 45 | Sep 24 22:04 |
| 35 | jacob_stavrianos | 5.1e-9 | 1.97e-8 | 11 | Sep 24 18:51 |
| 41 | dipdecir | 5.2e-9 | 2.40e-8 | 16 | Sep 24 22:18 |

- **Luna**: team page `/teams/Luna` shows one member, **`dogus_ozel` (Team Organizer)**, no description. `/participants/dogus_ozel` → 404; discourse `user_actions.json?username=dogus_ozel` → empty list (no posts); web search for the name finds no GitHub/paper trace. NOT FOUND beyond the username. Implied utilisation ≈ 0.156 (2.6e-9/1.67e-8), i.e. near the 0.1 floor like the other leaders.
- **mliston**: forum posts are all Phase 1 — `/t/18170` "Phase 1 Write Up - Bias–Variance Tradeoff Across Control Layers (submission #327723)" (Aug 15: "a control variate plus a bag of tricks that Claude and Codex found"), `/t/18108`, `/t/18122`, `/t/18025`, `/t/18051`. **No Phase 2 posts.** The "2.67e-9 on Sep 25" figure is not visible: the board shows 2.9e-9 as of Sep 24 22:04.
- **suliman_tadros**, **a_s6**: discourse accounts exist, zero posts. **dipdecir**, **jacob_stavrianos**: no discourse account (404). No public statements from any of them.

---

## 4. "Interesting talks by the organizers" — CONFIRMED thread, PARTLY on content

`https://discourse.aicrowd.com/t/18206`, by **Cipo**, 2026-09-01 20:47 UTC, 3 posts. Post 1 verbatim: "Some interesting talks by the organizers in front of the math community. In case you have a free evening :slight_smile: - https://www.youtube.com/watch?v=_BPsuomAcoI - https://www.youtube.com/watch?v=N0hhsESjKEs". Replies (marius_binner Sep 3, ZITOUNI Sep 4) are jokes about Terence Tao joining; no technical content.

Via YouTube oEmbed the videos are:
1. `_BPsuomAcoI` — **"Paul Christiano - Mechanistic Estimation - IPAM at UCLA"** (channel: Institute for Pure & Applied Mathematics).
2. `N0hhsESjKEs` — **"Jacob Hilton - Miniature mathematical mysteries: uninterpreted models w/ fewer than 1,500 parameters"** (IPAM) — the AlgZoo material (ARC blog 2026-01-26).

**No Michael Winer talk is linked**, and the watch pages/transcripts were unreachable (429); IPAM's 2025/2026 video listings do not index these talks, so I could not summarise slide content. The organizer technical hints you listed actually come from the **Townhall** (`/t/18078`), verbatim: "As network depth increases, the effective rank of the activation covariance matrix decreases (eigenvalues become more skewed). ARC is internally exploring spending more compute on higher-variance directions rather than tracking every direction equally"; "cumulant propagation methods will probably beat quasi-Monte Carlo in the long run"; "Pushing cumulant propagation to higher-order moments improves results initially but eventually diverges. ARC has explored Borel resummation as a possible fix but hasn't fully solved it. They believe this issue matters more for very narrow networks (e.g. width 16) than for the current challenge parameters (width 256)". Winer's "Matching Sampling Principle" framing is in the ARC blog "A Mike's-Eye View of ARC's Research" (2026-06-09): "The general strategy is something we call cumulant propagation."

---

## 5. Discussion listing, activity after 2026-09-15 — CONFIRMED

Category id **2991**, slug `white-box-estimation-challenge-2026`, 93 topics (`/c/white-box-estimation-challenge-2026/2991/l/latest.json`). Only three topics were created or bumped after Sep 15:

| id | title | author | created / last | technical content |
|---|---|---|---|---|
| 18218 | "Everything we tried: a factorized K=3 cumulant-propagation estimator at 0.25 x B, where its FLOPs go, and 25 measured dead ends (team 504aldo, rank ~10)" | 504aldo | Sep 10 / **Sep 22** (5 posts) | raw 2.13e-8, C/B 0.2526, adj 5.40e-9; exact 7→4 units per source layer, κ4 regeneration `G_off = lambda_l * C_off` (one scalar/layer), rank-≤16 D21 feedback, shared rank-128 subspace for old sources, per-MLP λ, Strassen-Winograd (sponsor confirmed 2026-09-11 "permitted with no recursion-level limit"). Repo `github.com/504aldo/whest-p2-cumulant-k3` (MIT). jamesrahenry (Sep 13): "13 different exact submissions under other people's accounts … Your disclosure definitely got me out of a local minima." 504aldo (Sep 14): "someone trimmed 7-22% of the chain's cost (4.2e-9 to 5.2e-9) without losing accuracy". **AndreasHad04 (Sep 22)**: measured R² of G_off vs C decays 0.9977 (L1) → 0.2232 (L8) → 0.0454 (L15) on 8 nets, contradicting "R^2 about 0.95 AT DEPTH". **504aldo (Sep 22)**: different objects — his G is "the r=1 matrix core of the augmented chain", not the (2,2) slice of κ4; code uses the pre-activation law `G_pre_off = lambda * C_pre_off` with "R^2 0.48-0.52 at layer 0, ≥ 0.95 from layer 3 and 0.985-0.99 at depth". |
| 18219 | "[Phase 2] The compute budget is not a lever, and the scoring rule is why: `adjusted = mse x max(0.1, util)`" | nkosi_ndwandwe | **Sep 17** (1 post) | Sobol sweep 1,621→3,647 points: adjusted flat at ~1.337e-6 (MC rate, "the low-discrepancy advantage has collapsed"); analytic exact-Price cov-prop 3.7599e-6 MSE × util 0.0653 shows as 3.7599e-7 ⇒ floor `max(0.1, util)` binding ("Below the floor, the leaderboard is a pure MSE ranking"); the visible ~2.13e-8 cluster is NOT the stock `03_covariance_propagation` baseline (3.933e-7). |
| 18145 | "[Phase 1 write-up] Fixed grid of directions instead of random sampling (submission #320802)…" | skye_nygaard | Aug 10 / bumped Sep 22 by `mcdvoice007` | The Sep 22 reply is absent from the JSON (deleted; mcdvoice007 also left a content-free reply in 18198) — spam bump. |

Earlier September threads for context: 18217 "Errors are leaking information" (marius_binner, Sep 9; mohanty: "the global submission message is currently leaking some message, and we will fix it right away"); 18216 yujigon Phase 2 write-up #330001 (zero fitted constants, rank-4,608 CP channel for inherited κ3, 1.066e-7 public, 9.89% budget; "The constants are absorbing accumulated drift, not local skew"); 18214 venture_agent_ai (exact bivariate ReLU covariance via Price/Plackett + per-layer ridge calibration, 1.32e-7 adj; feeding calibration back "made things 69x worse"); 18206 (talks); 18198 keenanpepper "Covariance propagation baseline scores 3.93e-7/3.93e-6" ("decisively beats sampling in Phase 2"); 18199 keenanpepper "'Unusually easy leaderboard set' effect is gone" (Phase 2 per-MLP difficulty "max/median 1.4×, skew 0.03"; "Local scores now transfer to the leaderboard ~1:1"). No organizer topic since Aug 23 (18197); no Phase 1 results announcement exists yet.

---

## 6. How two designated submissions are combined — NOT FOUND on the forum

- `/t/18197` (Phase 2 live) contains **no** sentence about designation/selection; the only "private" mention is "The Phase 1 private re-evaluations are in progress."
- Townhall `/t/18078`: "You can keep submitting throughout Phase 2, and any submission during that window can be designated as your final entry. The designation locks at the actual Phase 2 deadline." (singular "final entry") and "The private rerun evaluates designated entries on a completely fresh set of MLPs. No public-split score carries over."
- Phase 1 wording (`/t/18125`, `/t/18118`, `/t/18143`): "Each team may select up to two submissions for the Phase 1 private re-evaluation." / "If you select none, we will use your two highest-ranked submissions on the Phase 1 public leaderboard." / "Prize rankings will be determined exclusively by the post-Phase 1 private re-evaluation results". **Nowhere does the forum state how the two are combined** (best-of-two vs both ranked).
- `/t/18143`: **inverted_donkey's question (Aug 10) was never answered**: "The rules specify a single submission per team for both phases, mention a singular 'submission' a dozen times throughout the text, and don't seem to contain any clause allowing for arbitrary changes in this regard. Am I missing something?" thylinao's three write-up questions in the same thread were also not answered there (they had been answered on Jul 6 in `/t/18041` post 4: "A PDF technical write-up plus exactly one submission ID").
- The rules page is JS-rendered, so I could not read v14 §7 myself. Verdict: the forum is silent — it neither confirms nor contradicts "only that Participant's best-scoring such submission is ranked".

---

## 7. ARC publications in September 2026 — NOT FOUND

- ARC blog index: newest post is "A Mike's-Eye View of ARC's Research" (Winer, 2026-06-09); nothing in Aug–Sep 2026. Michael Winer's author page has the same.
- LessWrong "Announcing the ARC White-Box Estimation Challenge": no comments after ~Jul 2026; ARC staff comments (paulfchristiano, Jacob_Hilton, Sharada Mohanty) are Phase 1 logistics only. "Mechanistic estimation for wide random MLPs" (May 7): no Aug/Sep comments. Jacob Hilton's LW profile shows nothing new.
- Web search for X/Twitter posts by Hilton / Wu / Lecomte / Winer on Phase 2: nothing.
- The phrase "learned networks consuming cumulant estimates" does not appear in any public ARC/organizer source I could find; the closest organizer language is the Townhall's "purely learned/black-box estimator" contrast and the prize guideline's discount on "fine-tuned constants". Phase 1 prize results are still unpublished ("in progress" as of Aug 23; no results topic as of Sep 24).

---

### Corrections to the brief
1. `/t/18097` and the Substack belong to **jamesrahenry (James Henry)**, not jamespayor; jamespayor is a different, silent-in-Phase-2 participant at rank 70.
2. The "Interesting talks" links are **Christiano (Mechanistic Estimation)** and **Hilton (AlgZoo)** at IPAM, not a Winer talk.
3. `arc-whestbench-p2-full1000-N1e9` is dense pairwise (16×1024×1024 per field) at **336.7 MB per net** (~337 GB total), not 336.7 MB total.
4. bakev2 stores shifted central moments `E[(z−c)^p]`, p=1..6 (κ derived), plus `gate_p`; dense `post_K21/post_K22/gate_GG/gate_GX` exist only in the two `*-supp` repos (100 `full` + 100 `mini` nets); pre-activation dense blocks live in full1000 / p2-higher-moments.
