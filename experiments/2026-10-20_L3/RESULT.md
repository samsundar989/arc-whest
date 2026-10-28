# L3 — gain-group screen (RunPod) → nominee B (B-young)

Handoff: `_claude_snap/handoff_2026-10-19/HANDOFF_2026-10-20_L3_GAINGROUP_SCREEN_BYOUNG.md`. Gain groups as defined in
L2's handoff §2. All times UTC.

## Shared gain source (L2 applies this exact file to R3's export)

`experiments/2026-10-20_L3/gaingroups/est_8be_p2.py`, sha256 **52dc31e81e6fd2aa68830883c6ced6e319f98395595dd9ef67387604ee9ade13**.
Built as a clean 3-way merge (`git merge-file`, 0 conflicts): base = A4 (`a4_fixedplanner_prerequisite/estimator.py`, 010786d7),
ours = exact 8be (8be40031), theirs = A4+P2 (`a4_p2_integration/A4_P2_template.py`, 6d674f38, the U1 P2 forum hunks);
then one added gain group, the ρ-face: `_RHO` (16 zeros) and, after `var` is formed in `_layer`,
`if _fz_on(_RHO[l]): var = var + mu_pre**2 * _RHO[l]`. All gains default 0 (`_D4_LAW_SCALE` 1) and both training flags
(`_U1_TRAIN`, `_FZ_TRAIN`) False, so every new path is gated off in production.

| group | tables (each a 16-entry per-layer list) | training flag | NEW rate |
|---|---|---|---|
| G1 = U1 NEW10 | `_YP_TP_GAIN _YP_TPQ_GAIN _K4T1_CV _D4_LAW_SCALE _TMET_GAIN _MARG2_GAIN _MARG2_C2_GAIN _CL_C3_GAIN _CL_CQ_GAIN` | `_U1_TRAIN`, `_FZ_TRAIN` | 1e-2 (job 3472's) |
| G2 = ρ-face | `_RHO`, init = J §6 table at layer_index 2..14 (L3..L15) | `_FZ_TRAIN` | 1e-6 (Adam steps ≈ lr; the table is 1e-5…1e-4) |
| G3 = marg2 + metric + closure pairs | `_TMET_GAIN _MARG2_GAIN _MARG2_C2_GAIN _CL_C3_GAIN _CL_CQ_GAIN` — **a subset of G1** as the handoffs define them | `_FZ_TRAIN` | 1e-2 |
| ALL | G1 ∪ G2 | both | — |
| ALL-R | ALL + `_SHARED_RANK` 288 → 320 | both | — |

Trainer: `gaingroups/l3_train.py` — the R3 stack unchanged (pair `run.py` + literal-8be parser-repair adapter + linalg shim),
TRAIN = full 0..899 (seed-20260925 permutation, batch 8, 112 updates, 4 rows unused), cosine over 112, corrector LR ×10,
VAL 900..999 at 0/28/56/84/112, Strassen-0 dev backend; the parser guard's reference is the arm's own starting source.

## 0. Predictions on record (written 2026-10-20 17:50, before any arm ran)

| arm (from exact 8be, 112 updates, corrector ×10) | prediction raw vs CONTROL | keep if |
|---|---|---|
| G1 = U1 NEW10 nine groups | −0.9 % | ≤ −0.3 %, ≥ 60/100, CI excl. 0 |
| G2 = ρ-face (`var_t += _RHO[t]·mu_t²`, init J table) | −0.3 % (80 %: −0.1 … −0.6 %) | same; and fitted `_RHO` at L ≥ 9 stays within 0.3–2× the table |
| G3 = marg2 + per-neuron metric + closure pair diagrams | −0.4 % | same |
| ALL = G1+G2+G3 | −1.3 % | — (L2's second stage runs this if all keep) |
| ALL-R = ALL + `_SHARED_RANK` 320 | −1.8 % raw, +0.5 % F | score-equiv ≤ −0.2 % vs ALL |
| B-young canary | 1.95 – 2.15e-9 | graded inside the interval; per-MLP residual correlation with A < 0.9 |

## Pod / spend log

| time | event |
|---|---|
| 17:44:39 | `young-final` (wk8wfy9vtygcwv, staged volume) could not start: host GPUs taken. Created `arc-whest-l3-screen` `a0y274h7apvvu2`, RTX 5090 Secure, EUR-IS-1, $1.19/h, 150 GB volume. |
| 18:05 | a0y274h7apvvu2 never started its container (host DNS failure pulling the image) → terminated (≈ $0.42). Created `arc-whest-l3-screen2` zfrzn3dosnueme (EUR-NO-1): container never created in 17 min → terminated 18:23 (≈ $0.35). |
| 18:23 | Started stopped pod `arc-whest-c1-direct-proof` gjv6ablp1xm7ce: torch cannot init CUDA on that host ("CUDA unknown error", also with 2.13+cu130, also after restart) → stopped 18:31 (≈ $0.16). |
| 18:31 | Started stopped pods `arc-whest-subset-proof` **zrdhrgb88enc0w** (EUR-NO-1, 5090, $1.19/h) and `arc-whest-c1-generated-proof` cmaprcxx73kry8. Both CUDA OK. cmaprcxx73kry8 cannot reach PyPI (timeouts) → stopped 18:47 (≈ $0.32). **All screen arms run on zrdhrgb88enc0w** (Torch 2.13.0+cu130, NumPy 2.2.6, whestbench 0.16.1, flopscope 0.12.3; weights regenerated per net from the seed, fixture-hash checked). |

## 1. Screen

**Runtime facts (pod zrdhrgb88enc0w)**
- CONTROL VAL0 (100 nets) = **1.534575e-8**, equal to L2's R3 VAL0 on the local box (exact 8be, 1.534575e-8) — the parent is exact 8be, reproduced cross-host.
- Zero control: G1 with all nine tables at 0 gives VAL predictions **byte-identical to CONTROL at all 16 layers** (smoke, 2 nets); ALL VAL0 = CONTROL VAL0 exactly (100 nets).
- **ρ table falsified as an initial value.** Starting `_RHO` at the J §6 table makes step-0 VAL **7.8× worse** than CONTROL (1.287e-7 vs 1.647e-8, 2 VAL nets; deviation starts at layer index 2 = L3 and grows to 1.3e-3 max-abs at the output). A face that is "mean-neutral" by J's analysis cannot do that at its own fitted values, so the table does not match this chain's `var`/`mu_pre` scale or indexing. The screen therefore starts `_RHO` at 0 (byte-identical start) and lets it learn at lr 1e-6. L2 is doing the same (ack 18:47).
- ALL-R step 0 (rank 320, everything else unchanged) = 1.516442e-8: **−1.18 % raw before any update**.
- Speed: ≈ 22–25 s/update with three arms sharing the GPU (98 % util); VAL-100 ≈ 90 s.

Launch: CONTROL, ALL, ALL-R at 18:42:30 (112 updates each); G1, G2, G3 queued to start when those three complete (≈ 19:35).

### Screen tables (VAL-100 paired vs CONTROL at the same step; `gaingroups/analyze.py runs ALL ALL-R`)

CONTROL raw by step: 0 1.534575e-8 · 28 1.549221e-8 · 56 1.539980e-8 · 84 1.532927e-8 · **112 1.531575e-8** (best; −0.20 % vs step 0).

| step | ALL Δraw | wins | 95 % CI | ALL-R Δraw | wins | 95 % CI |
|---|---|---|---|---|---|---|
| 0 | 0.000 % | — | (identical) | −1.182 % | 86 | [−1.440, −0.924] % |
| 28 | +5.728 % | 14 | [+4.626, +6.829] % | +3.434 % | 25 | [+2.344, +4.523] % |
| 56 | +0.973 % | 27 | [+0.547, +1.399] % | −0.041 % | 52 | [−0.549, +0.467] % |
| 84 | +0.289 % | 46 | [+0.045, +0.532] % | −0.906 % | 70 | [−1.272, −0.540] % |
| **112** | **+0.018 %** | **52** | **[−0.220, +0.255] %** | **−1.175 %** | **77** | **[−1.507, −0.843] %** |

Best-vs-best = step 112 vs step 112 for both. ALL-R vs ALL at 112: raw ratio 0.98807 (−1.19 %), real bill +1.864 % F
(292,214,932,424 vs 286,868,957,368; ops unchanged 22,235; nets 900/901) ⇒ **score-equivalent +0.65 % (worse)**. The −1.18 % is
already there at step 0 — it is the rank, not the fit.

Fitted `_RHO` (ALL, step 112, layer_index 2..14): −2.5e-6 −2.6e-6 −1.5e-6 −9.0e-7 −2.2e-6 −1.2e-6 −7.7e-7 1.1e-7 7.1e-7 5.5e-7
−4.0e-7 −9.7e-7 −1.2e-6; ratio to the J table at L9..L15 = −0.01 … +0.01 (ALL-R the same). The optimizer holds ρ at ≈ 0, two
orders below J's 0.7–1.1e-4: ⇒ below the 0.3× bound ⇒ DROP by the handoff's own rule (and the table start is 7.8× worse).

### Verdict (posted 19:42 UTC; G1/G2/G3 attribution arms running since 19:38, done ≈ 20:32 — update below)

| group | verdict | evidence |
|---|---|---|
| G1 U1 NEW10 | **DROP** (pending G1 arm) | inside ALL: ALL = CONTROL at 112 (+0.018 %, 52/100, CI spans 0) |
| G2 ρ-face | **DROP** | learned ρ ≈ 0 (≤ 0.01× table); table start 7.8× worse |
| G3 marg2/metric/closure | **DROP** (pending G3 arm) | ⊂ G1; inside ALL, null |
| ALL | **DROP** | +0.018 % vs CONTROL at 112, CI [−0.22, +0.26] % |
| R = `_SHARED_RANK` 320 | **DROP** | −1.19 % raw vs ALL but +1.86 % F ⇒ score +0.65 % (rule: ≤ −0.2 %) |

**For L2:** no gain group earns its place in 112 updates from 8be ⇒ per your handoff's fallback, **bake the R3 export alone**
(no second stage) unless the G1 arm alone contradicts ALL by 22:30. Predictions vs outcome: G1 −0.9 % / G2 −0.3 % / G3 −0.4 % /
ALL −1.3 % / ALL-R −1.8 % predicted; ALL measured +0.02 %, ALL-R −1.18 % raw but +0.65 % score. All wrong in the improving direction.

### Attribution update (20:40 UTC; G1/G2/G3 finished 20:33) — verdict unchanged, all DROP

| arm vs CONTROL | 28 | 56 | 84 | **112** | wins @112 | 95 % CI @112 | bill (net 900) | score-equiv |
|---|---|---|---|---|---|---|---|---|
| G1 U1 NEW10 | +4.781 % | +1.073 % | +0.234 % | **+0.032 %** | 50 | [−0.191, +0.255] % | — | DROP |
| G2 ρ-face | +0.260 % | +0.060 % | +0.003 % | **−0.001 %** | 54 | [−0.013, +0.011] % | — | DROP |
| G3 marg2+metric+closure | −0.110 % | −0.145 % | −0.198 % | **−0.209 %** | 74 | [−0.274, −0.144] % | 288,244,803,047 F (+0.480 %), 22,735 ops (+500) | **+0.27 %** ⇒ DROP |

G3 is a real but sub-threshold raw effect (negative at every VAL step, CI excludes 0) that costs more FLOPs than it saves; inside
G1/ALL it is cancelled by the U1-only tables (`_YP_TP_GAIN _YP_TPQ_GAIN _K4T1_CV _D4_LAW_SCALE`), which on their own make
G1 ≈ +0.24 % worse than G3 at step 112. Its step-112 export is kept at `pod_artifacts/G3_estimator_step112.py` (6861ed59) in case
a later composition wants it. G1 does not contradict ALL ⇒ **no 22:30 change for L2**.

Artifacts: `pod_artifacts/` (every VAL json with per-net vectors and fitted tables, train logs, identities, bills).

**Pod spend:** a0y274h7apvvu2 ≈ $0.42 · zfrzn3dosnueme ≈ $0.36 · gjv6ablp1xm7ce ≈ $0.18 · cmaprcxx73kry8 ≈ $0.28 ·
zrdhrgb88enc0w 18:31–20:37 ≈ $2.45 ⇒ **≈ $3.7 total** (≤ $20). All four pre-existing pods back to EXITED, both pods created by
this lane terminated.

## 2. B-young on 8be (pipeline proof before A1)

Real meter = flopscope 0.12.3 + whestbench 0.16.1, f64 read-only weights, `gaingroups/l3_bill.py` (the `tools/_cost_worker.py`
path with seed-regenerated weights) on pod zrdhrgb88enc0w CPUs. **Meter check: exact 8be bills 286,868,957,368 F / 22,235 ops** on
nets 900 and 901 = the recorded #336993 bill.

Planner sweep from 8be (only `_SW_LEAF`, `_SW_CAP`, `_SW_PAD`, `_SW_OP_FLOPS` change; `_STRASSEN_MIN_BLOCK` is not read by
8be's `_sw_plan`). Bills are shape-determined (identical on nets 900 and 901). Residual projection = ops × 15.24 µs (#336993's
served worst: 0.3389 s / 22,235 ops); wall projection = ops × 0.371 ms + F × 255 ps (fits #336993's served max 81.4 s).

| build (cap, pad, leaf, λ_op) | FLOPs | ΔF vs 8be | ops | resid proj | wall proj |
|---|---:|---:|---:|---:|---:|
| 8be (6, pad, 8, 1e6) | 286,868,957,368 | — | 22,235 | 0.339 s | 81.4 s |
| 5, no pad, 8, 1e6 | 297,744,578,986 | +3.79 % | 21,773 | 0.332 | 84.0 |
| 5, no pad, 8, 2e6 | 300,103,663,483 | +4.61 % | 20,509 | 0.313 | 84.1 |
| 5, no pad, 8, 3e6 | 301,082,756,245 | +4.96 % | 20,225 | 0.308 | 84.3 |
| 5, no pad, 8, 4e6 | 303,256,906,679 | +5.71 % | 19,671 | 0.300 | 84.6 |
| **5, no pad, 8, 5e6 = B-young** | **306,840,769,522** | **+6.96 %** | **18,943** | **0.289** | **85.3** |
| 5, no pad, 8, 6e6 | 313,954,921,534 | +9.44 % | 17,741 | 0.270 | 86.6 |
| 5, no pad, 8, 8e6 | 319,279,604,852 | +11.30 % | 17,045 | 0.260 | 87.7 |
| 5, no pad, 12 or 16, 1e6 / 2e6 | 301.5 G / 303.4 G | +5.10 / +5.75 % | 21,341 / 20,239 | 0.325 / 0.308 | 84.8 |
| 5, no pad, 40, 1e6 / 4e6 (literal "block-40") | 343.1 G / 344.5 G | +19.6 / +20.1 % | 18,525 / 17,645 | 0.282 / 0.269 | 94.4 / 94.4 |

Choice: **cap 5, no pad, leaf 8, λ_op 5e6** — the cheapest build under 20 k ops with margin to the 0.30 s residual bound (λ 4e6
sits exactly on 0.300 s at the worst served rate). A literal leaf-40 costs +20 % F and is rejected. The handoff's "+1–2 % F" is not
reachable without pad: dropping pad alone is +3.8 %.

**Parity vs 8be.** Nets 900/901: max |Δ| 7.9e-6 (1.5–1.6e-6 of the output scale; > the 1e-6 target). For scale: 8be against
itself at 2 vs 4 BLAS threads moves by 1.1e-5 / 6.7e-6 — B's reordering is inside 8be's own thread-count variation.
**VAL-100 (real numpy chain, 2 threads):** 8be 1.534588e-8, B-young 1.534875e-8 ⇒ **Δraw +0.019 %** (95 % CI [−0.015, +0.052] %,
B wins 48/100), max |Δpred| 3.4e-5 ⇒ **passes |Δraw| ≤ 0.05 %**. Per-MLP final-layer error correlation with A: mean 0.99992,
min 0.99933 — **the pre-registered "< 0.9" is false**: B is the same arithmetic in another order, so it fails like A on accuracy;
it fails differently only on runtime (no pad, depth ≤ 5, 15 % fewer op calls, ≈ 0.05 s lower residual). Score cost ≈ +7 % (F).

**`tools/verify_nominee.py` on B-young-8be (workerq #3577, CPU-only, 20:36 UTC): PASS.** checkout (sha f9878ec0; WARN not yet in
DESIGNATION), imports (AST + client-only venv, numpy not loaded), static (WARN: reviewed helper `_pkk_fast_ok:1318`, same as 8be),
real client↔server rig at 1024×16 / 512×16 / 256×4 finite, meter DEV0–7 = 306,840,769,522 F exactly, 18,943 ops, 2-process
determinism bit-identical, served-equivalent residual **0.2887 s**, wall **83.0 s**. Memory gate (100 calls, WSL 8 GiB) queued as #3580.

**Incident 20:43 UTC.** The B-young-8be memory gate #3580 (WSL) started alongside L1's #3578 and tripped R3 #3556's owner guard
(R3 Windows memory + whole WSL VM ≤ 16 GiB): R3 died at step 330/736; L2 resumes from the step-330 checkpoint after both exit
⇒ A1 slips to ≈ 03:30 UTC. L3 runs no local WSL/heavy job until L2 confirms R3 finished; B on A1 is planned on a RunPod pod.

**Memory gate on B-young-8be (#3580, WSL RLIMIT_AS 8 GiB, 100 calls): PASS** — 0 errors, FLOPs identical on every call,
**VmPeak 3,691 MiB** (≤ 7,200), settled growth call 10 → 100 **+32.0 MiB** (≤ 64). (8be: 4,570 MiB per L1's verifier.)

## 3. B-young on A1 (provisional A1 = L2's `cand_r3s552_lam06.py`, sha 56f08e9e; R3 step 552 + λ 0.6e6)

`byoung_a1/estB_r3s552.py` = `build_byoung.py` on A1 (3 lines: `_SW_CAP` 6→5, `_SW_PAD` True→False, `_SW_OP_FLOPS` 6e5→5e6),
sha **e58966ff**, `whest validate` pass. Runs on pod gjv6ablp1xm7ce (its GPU is broken; CPU-only work; started 23:41 UTC — CPU pods had no
capacity) with flopscope 0.12.3 / whestbench 0.16.1 / NumPy 2.2.6, 2 BLAS threads for both A and B.

| | A1 | B | Δ |
|---|---:|---:|---:|
| real bill (nets 900, 901) | 286,123,309,398 F / 22,633 ops (= L2's) | **306,840,769,522 F / 18,943 ops** | **+7.241 % F**, −16.3 % ops |
| VAL-100 raw | 1.514349e-8 | 1.514188e-8 | **−0.011 %** [−0.046, +0.024], B wins 53 |
| FRESH-A100 raw (N = 1e8 labels; weights hash-checked vs laneV seeds.json) | 1.547433e-8 | 1.547729e-8 | **+0.019 %** [−0.012, +0.050], B wins 40 |
| per-MLP error corr (VAL / FRESH) | | | 0.99993 / 0.99993 (min 0.9998) |
| max \|Δpred\| (VAL / FRESH) | | | 1.13e-5 / 1.23e-5 |

Both |Δraw| ≤ 0.05 % ✔. Projected served residual 0.289 s (≤ 0.30 ✔), wall ≈ 83 s (≤ 90 ✔) — same shapes as B-on-8be, whose
verify_nominee projection was 0.2887 s / 83.0 s. Memory gate (native Linux RLIMIT_AS 8 GiB, 100 calls) running on the pod for B and A1.
**Expected B score = A1 graded × 1.0724 × (1 ± 0.0005)** (FLOP ratio × raw ratio; λ = 0 so score ∝ F × raw).

### Final A1 = `cand_r3s736_lam06.py` (7d381fc5, R3 step 736 + λ; L2 commit c141312) ⇒ **B = `byoung_a1/estB_r3s736.py`, sha 7a146f84** (commit 1784014)

| | A1 (7d381fc5) | B (7a146f84) | Δ |
|---|---:|---:|---:|
| real bill | 286,123,309,398 F / 22,633 ops | 306,840,769,522 F / 18,943 ops | +7.241 % F |
| VAL-100 raw | 1.513847e-8 | 1.513700e-8 | **−0.010 %** [−0.048, +0.028], B wins 47 |
| FRESH-A100 raw | 1.546965e-8 | 1.546842e-8 | **−0.008 %** [−0.051, +0.035], B wins 49 |
| per-MLP error corr (VAL / FRESH) | | | 0.99993 / 0.99992 |

Both ✔ (|Δraw| ≤ 0.05 %). **Canary forecast (written before submission): B graded = A1 graded × 1.0724 (± 0.05 %).**
If A1 grades at L2's point 1.83e-9 ⇒ B ≈ 1.962e-9 (inside the pre-registered 1.95–2.15e-9).

**Gates on B (A1 chain).** Memory = shape-only, run on the 552 bake (same file except fitted tables): native Linux
`local_sim/linux_memory_gate.py`, RLIMIT_AS 8,192 MB applied (negative/positive/lifecycle controls PASS), real whestbench 0.16.1 /
flopscope 0.12.3, f64 read-only weights, 100 predicts in one worker: all finite, **VmPeak 3,667.7 MB** (2.23× headroom), settled
growth **−12.1 MB** over 100 ⇒ PASS (the script then raised in an incidental "same file as repo-root estimator.py" anchor comparison
that does not exist on the pod — after every check had passed). verify_nominee #3611 on 7a146f84: checkout/imports/static PASS,
meter exact 306,840,769,522 F / 18,943 ops, 2-process determinism PASS, residual-eq 0.2887 s, wall-eq 83.0 s; **rig FAILED on a
client handshake timeout** while the box was RAM-exhausted (L2's concurrent #3608 hit `_ArrayMemoryError` at 00:57) — rerun pending
a free box. (B-on-8be passed rig at 1024×16 / 512×16 / 256×4 with the identical planner code, #3577.)

**verify_nominee on final B 7a146f84:** #3611 checkout/imports/static/meter/project PASS (meter exact 306,840,769,522 / 18,943,
determinism, residual-eq 0.2887 s, wall-eq 83.0 s); its rig failed under host RAM exhaustion (00:57) — rerun #3625 (imports,rig):
**rig PASS** at 1024×16 (78.8 s), 512×16, 256×4. **All B gates green.** `docs/DESIGNATION.md` B section added (score pending canary).

**Native-Linux memory table (pod gjv6ablp1xm7ce, same host, `linux_memory_gate.py`, RLIMIT_AS 8,192 MB, 100 predicts, all controls PASS)**

| file | VmPeak | settled growth / 100 | verdict |
|---|---:|---:|---|
| exact 8be (8be40031) — control | 4,527.9 MB | +204.2 MB | FAIL (> 64) |
| A1 lam06 (552 bake; same code as 7d381fc5) | 4,564.2 MB | +200.7 MB | FAIL |
| A1 lam10 (73321458) | 4,535.1 MB | +131.5 MB | FAIL |
| **B (no pad, cap 5)** | **3,667.7 MB** | **−12.1 MB** | **PASS** |

8be reads +0.0 MiB on WSL (L1's verifier) and #336993 served cleanly, so the growth is a host/glibc behaviour of the padded,
depth-6 planner on native Linux, not λ ⇒ L2 keeps A1 = lam06 (7d381fc5). B is the only build flat on native glibc — one more way it
fails differently from A. Logs: `pod_artifacts/mem/`. Pod gjv6ablp1xm7ce stopped 03:26 UTC (23:41–03:26 ≈ $4.4) ⇒ **L3 RunPod total ≈ $8.1**.

## 4. Canary B — forecast (written 03:30 UTC Oct 21, before submission)

A1 = #337456 (7d381fc5, submitted 03:24:25Z by L2, L2 forecast 1.863e-9). B package `b-young-r3s736-7a146f84.tar.gz`
(estimator 7a146f84; `whest validate-package` pass). Forecast rule: **B graded = A1 graded × 1.0724** (F ratio 1.07241 ×
raw ratio 0.9999 on VAL / FRESH-A). On L2's A1 forecast: **B = 1.998e-9** (pre-registered L3 band 1.95–2.15e-9). Served:
wall max ≈ 83 s and residual max ≈ 0.289 s expected (A1 ≈ 0.345 s). Will re-anchor on A1's graded score before submitting.
**Re-anchored 03:5x UTC (before submitting):** A1 #337456 graded 1.8656192469864824e-9 ⇒ **B forecast = 2.0007e-9**
(1.8656192e-9 × 306,840,769,522 / 286,123,309,398), interval ± 0.5 % ⇒ **1.991–2.011e-9**; served residual max ≈ 0.335 × 18,943/22,633 ≈ 0.28 s.

### Canary B outcome — #337462 GRADED (04:22 UTC Oct 21)

| | forecast (written first) | graded | A1 #337456 |
|---|---|---|---|
| score | 2.0007e-9 (1.991–2.011e-9); pre-registered band 1.95–2.15e-9 | **2.0005580379785525e-9** (−0.007 % vs forecast) ✔ | 1.8656192469864824e-9 |
| raw (final-layer MSE) | = A1 × 0.9999 | 1.4337337e-8 | 1.4338385e-8 |
| B / A1 | 1.07241 | **1.07233** | — |
| served FLOPs per MLP | 306,840,769,522 | 306,840,355,133 (−414,389 = graded offset) | — |
| served residual max / mean | ≈ 0.28 s | **0.2760 / 0.2610 s** | 0.3349 / 0.3262 s |
| served wall max / mean | ≈ 83 s (law) | **61.06 / 54.79 s** | 83.60 / 78.47 s |
| failed MLPs | 0 | **0/100** | 0/100 |
| per-MLP corr with A1, served residual / predict wall | < 0.9 | **0.248 / 0.078** ✔ | — |

B lands inside its interval and **fails differently from A on runtime** (per-MLP served residual correlation 0.25; 18 % lower residual
max, 27 % lower wall max, no pad path, depth ≤ 5, 16 % fewer op calls, flat on native glibc) while matching A's accuracy
(VAL/FRESH Δraw −0.010 / −0.008 %). The wall law over-projected B (83 s vs 61 s served): unpadded depth-5 leaves run faster than its
leaf-32 rate assumes. Report: `experiments/run_reports/canaryB_L3_337462_graded_report.json`. Ledger: 337462 recorded.

WSL memory gate on final B (#3634) cancelled at Samuel's request (04:38 UTC) for a restart; B's memory evidence = B-on-8be WSL
(VmPeak 3,691 MiB, +32 MiB) and final-chain B native Linux (3,668 MB, −12.1 MB). Lane L3 closed.
