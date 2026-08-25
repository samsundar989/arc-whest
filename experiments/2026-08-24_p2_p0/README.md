# P2-P0 evidence — the Phase-2 evaluator repair

Slice: **P2-P0**, 2026-08-24. Contract, result and Skeptic review: `experiments/LOG.md`
(search `P2-P0`). Preregistration commit `e8afcbf`, which precedes every measurement here.

Class IMPORTANT. **No seed block was claimed or touched**, `estimator.py` is
byte-identical throughout (`00240eb1e60810f5e9a6f743b8ad89a1533cfd8671c08e07525f1ed1de54d9ec`),
the vault was not touched, and no promotion statistic was computed.

## What each file is

| file | what it is | read it for |
|---|---|---|
| `baseline_before.txt` | git SHA, file hashes and package versions **before** any change | provenance of the starting state |
| `uv.lock.before` | the lock file before the upgrade | that the lock diff is exactly two packages |
| `official_anchor_1row.json` | `whest run --runner subprocess --profile --detail full --json`, 1 MLP, `v2-phase2` | **the anchor.** The official op-by-op table the local meter is reconciled against, plus the `run_config` the runner actually used |
| `official_anchor_1row_localrunner.json` | the same under `--runner local` | that the 32,768-FLOP `asarray` is charged on **both** runner paths, so it is not an IPC artifact |
| `p2_flop_anchor.json` | 3 arms × 2 flopscope versions | the reconciliation: `37,746,688` weight representation + `32,768` harness materialisation = `37,779,456`, residue **0** |
| `deployment_envelope.json` | enforceability matrix, memory accounting, worker death/restart | what this host **cannot** enforce — the 8,192 MB limit has no enforcement path on Windows at all |
| `p2_residual_paths.json` | 5 candidate-passes × 3 instruments × 20 MLPs | the residual **level** in ONE session. `k2p2` = 94.6/95.1 ms median, 140.6 ms worst, 0/20 over cap. Best case; not a bound |
| `p2_weight_repr_residual.json` | same host, same session, `np.asarray` vs `fnp.array` weights | the repair moved a fixed **~19 ms** out of residual and into counted FLOPs. Uninstrumented work is billed to residual, not to nothing |
| `p2_residual_version.json` | flopscope 0.10 vs 0.12, interleaved A/B, 3 rounds | **0.91×** — the stack release is NOT the cause of the gap against 2026-08-21 |
| `gpu_parity_1024x16.txt` | GPU/CPU truth parity at the live shape | generation bit-for-bit 4/4; forward disagreement 2.3e-11 of the decision signal |

## The preflight transcripts, and why there are several

`preflight.ps1` writes its verdicts with `Write-Host`, and the three capture forms
do **not** capture the same things. This was discovered here, by an independent
Skeptic, after the first three transcripts turned out to contain **zero** verdict
lines while being cited as evidence of green runs.

| form | verdicts | child stdout |
|---|---|---|
| `2>&1 \| Out-File` | **no** | yes |
| `-Log <path>` (Start-Transcript) | yes | **no** |
| `*>&1 \| Tee-Object` | yes | yes |

| file | capture form | verdict | use |
|---|---|---|---|
| **`preflight_full_evidence.txt`** | `*>&1 \| Tee-Object` | **`PREFLIGHT GREEN -- 16/16`** | **THE canonical evidence.** 986 lines, all 16 verdicts *and* all suite output |
| `preflight_full_witnessed.txt` | `-Log` | `PREFLIGHT GREEN -- 16/16` | the first transcript in this repo to contain its own conclusion |
| `preflight_full_complete.txt` | `-Log` | `PREFLIGHT GREEN -- 16/16` | an independent repeat of the above |
| `preflight_before_stale_stack.txt` | `2>&1` | *(none — see above)* | the BEFORE state, on whestbench 0.14 / flopscope 0.10. Detail only |
| `preflight_after_repair_run1.txt` | `2>&1` | *(none)* | an intermediate run. Detail only; superseded |
| `preflight_full_after_repair.txt` | `2>&1` | *(none)* | **a RED run, kept deliberately.** Its last line is step 16 failing with `SETUP_TIMEOUT` under CPU contention — the evidence that the setup ceiling is load-sensitive, and that a contended host produces failures as well as inflated seconds |

**Verification state: three consecutive `PREFLIGHT GREEN -- 16/16` on a quiet host,
exit code 0, nothing skipped.**

## Two things this evidence does NOT support

1. **Nothing about the grader's own configuration or seconds.** Every limit here is
   a package fact; `run_config` corroborates seven of them and **not**
   `memory_limit_mb` or `predict_timeout_s`. No graded Phase-2 report exists.
2. **No residual figure here is a bound.** The same `arm_k2p2` reads 94.6 ms in
   this session and **260.7 ms** in the frozen P2-CP1 ladder of 2026-08-21
   (`experiments/2026-08-21_p2_cp1/ladder_frozen.json`, key `per_mlp_residual_s`).
   An earlier draft of the LOG attributed that to contention; **that was retracted**
   — the older figures are a committed artifact from another day. Measured causes:
   weight representation **~19 ms**, flopscope 0.10→0.12 **0.91× (not the cause)**,
   and **~2.35× unexplained run/session state**. Quote headroom against the worst
   observed run. See `p2_weight_repr_residual.json` and `p2_residual_version.json`.
