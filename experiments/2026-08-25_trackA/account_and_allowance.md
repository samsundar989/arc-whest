# Account, allowance and board state — Track 0S durable artifact

**Purpose.** The submission contract needs these as artifacts, not as prose in a
narrative entry. This file is that artifact.

**Provenance, stated exactly, because it is not uniform.** Two different kinds of
fact are recorded below and they are *not* equally well evidenced.

---

## 1. What the Builder verified directly, in this repository

| fact | value | how |
|---|---|---|
| candidate sha256 | `00240eb1e60810f5e9a6f743b8ad89a1533cfd8671c08e07525f1ed1de54d9ec` | `sha256sum estimator.py` |
| identical to archive | yes, `estimator_v2_6_1.py` | LF-normalised comparison, preflight step 5 |
| official mini result | adjusted `4.050566369642183e-7` | `experiments/2026-08-25_trackA/mini_100_subprocess.json`, 100 rows |
| failures | `0`, all five breakdown counters `0` | same report |
| counted FLOPs/MLP | `137,600,326,656` | same report; equals the anchor, delta `+0` |
| worst residual | `0.09327980 s` of `0.4 s` → `4.288x` | same report |
| **GT-noise floor** | **`7.47823932957716e-11`** | `gt_noise_floor.txt`, recomputed from the organisers' published parquet rows |
| dataset revision | `hf://aicrowd/arc-whestbench-public-2026@v2-phase2`, sha256 `aab61cf11271d201…` | the report's own `run_config` |

## 2. What the Builder did NOT verify, and cannot from here

**This session has no browser and no AIcrowd credentials.** Everything in this
section is transcribed from the signed-in recheck recorded in the
`2026-08-25 — DECISION REVIEW` entry of `experiments/LOG.md`, performed by the
plan reviewers, not by the Builder. It is recorded here so the contract has a
durable artifact, and it is labelled so nobody later mistakes a transcription for
a capture.

| fact | recorded value | source |
|---|---|---|
| Phase-2 submissions by this team | **zero** | signed-in team/Phase-2 filter, 2026-08-25 |
| submission allowance | **10 per team per UTC day** | official challenge overview, 2026-08-25 |
| board size | 46 entrants | live leaderboard, 2026-08-25 |
| rank 1 | Puffi, `1.09e-8` adjusted / `3.55e-8` raw | same |
| the `3.93e-7` covariance-propagation tie | roughly ranks 35–39 | same |
| our measured baseline projects to | about **rank 40** | derived: `4.0506e-7` sits just behind that tie |
| gap to the leader | **37.2x** adjusted, **114.1x** raw | derived from the two figures above |

**If any of these has moved since 2026-08-25, the contract's operational
assumptions move with it.** A live recheck before submitting is cheap and is
Samuel's to run; nothing in the contract depends on the *board* being current,
only on the allowance being ≥ 1 and the account being able to submit.

---

## 3. What the visible score after submission will and will not be

**The live submission score is feedback on the 50 public networks only.** The
other 50 are withheld and surface only in the final result.

This corrects the earlier readiness packet, which said the graded submission
"runs the organisers' public+private evaluation" and implied the returned number
reflects both halves. It does not. `docs/VALIDATION.md` §6 already establishes
that the private half's **accuracy** is not observable in any graded report — only
its compute telemetry is — and the packet's wording contradicted that.

Consequence for the contract: the expected-score band below is a prediction about
**the feedback half**, and agreement there is weaker evidence than it looks,
because it is 50 networks rather than 100.
