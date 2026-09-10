# Independent instrument Skeptic — September 10, 2026

Reviewed source range: `0fe164c` through `bcca354`. This reviewer received the contract, protocol, diff and raw artifact paths in a fresh context, without the Builder's reasoning. No instrument files were edited. No research labels were opened and no vault evaluation was run. Findings below refer to source at `bcca354`.

## Skeptic verdict: REJECT

**Strongest challenge**

“Stage 0 is complete and trusted” overstates the implementation. Several unit suites test hand-built inputs rather than the real reporting paths; those paths still misclassify evidence, omit a package-version check, and discard existing failures.

**Evidence for it**

1. **[P1] Runtime evidence is incorrectly tagged MATH.** `local_sim/certify_candidate.py:131–147` maps sections `5`, `6`, `7`, `8`, but actual headings are `5-8` at line 694 and `11` at line 1166. Both fall through to MATH. SERVER receives no checks from the combined cost section. Local residual/wall failures therefore become “host-independent” mathematical failures. U9 manually constructs correctly tagged rows instead of exercising `head()` followed by `check()`.

2. **[P1] WhestBench version binding is absent in the actual environment.** `certify_candidate.py:327` and `linux_memory_gate.py:159` use `whestbench.__version__`, which is absent. `_certified_stack()` actually returns only `{'flopscope': '0.12.1+np2.2.6'}`. The consumer silently omits WhestBench. Tests supply a populated `STACK`. Use installed distribution metadata in both producer and consumer, and fail closed if required version information is unavailable.

3. **[P2] `8f7047e` suppresses explicit failures in legacy reports.** The early return at `certify_candidate.py:380–393` precedes checks of existing cap/control/prediction/status fields. A v2 report with `status=FAIL`, `predict_error=MemoryError`, `cap_applied=False`, and a surviving negative control produces zero failed checks, only DEGRADED. Missing new evidence can be SKIP; applicable existing negative evidence must remain FAIL. U10 (`test_certify_candidate.py:585`) only recalculates each stored verdict from its own stored failure lists; it does not replay the new consumer. Applicability matters: a failure on an unrelated workload is not proof that the current workload fails.

4. **[P2] The repeated-memory test retains its own extra networks.** `linux_memory_gate.py:181–189` retains the original float32 mint, float64 weights, and MLP. Lines 255–257 create another mint and MLP without releasing the first. At 1024×16 that adds approximately 192 MiB of network arrays before candidate state, against a 64 MiB permitted growth threshold at line 608. A small-memory candidate can fail because the instrument retains arrays. This is source-derived allocation accounting, not a measured live-shape result. Identical FLOPs across distinct networks (line 611) is also not a universal contract for value-dependent estimators.

5. **[P2] Noise accounting is a library helper, not integrated reporting.** `noise_corrected_compare` has no production callers. `p2_lab.py` still gates observed scores, and at line 349 assigns per-replica `gt_noise` to rows scored against the average of two replicas, whose noise floor is half that amount. Wiring the helper without correcting this would over-subtract noise by 2×. No historical decision reversal is established.

6. **Read counting remains partial.** Reservation and atomic claiming are implemented, but the original P08 consumer, `p2_corrector.py:357`, still directly calls `np.load` without the registry. The audit detects this but does not prevent another read. Describe the registry as an opt-in path until protected consumers are routed through it.

7. **Lower-priority numerical/provenance limits.** `_t_ppf` restricts its search to ±10,000 (`competition_score.py:516`): at df=1 and alpha=1e-6, it returns 10,000 versus SciPy's 636,619.7724192955. Ordinary tested alphas work. Memory-report instrument “binding” (`certify_candidate.py:397`) checks only a nonempty string: `arbitrary-non-hash` passes. It records a claimed identifier rather than verifying a known instrument version.

**Evidence for the original claim**

The following independently executed lightweight checks passed:

- Memory binding: 25/25.
- Truth reads and claim-race fixtures: 27/27.
- Competition scorer: 129/129.
- Certifier unit tests: 26/26. This is not the full mutation battery.
- Vault lifecycle sacrificial-only subset: 30/30, covering setup once, readonly float64 weights, materialization billing, failed-prediction zeroing, and identity pairing. This is not a rerun of the complete 47-check suite or a sealed-vault evaluation.

Identity-based alignment, ordinary-alpha correction, the unequal-multiplier noise formula, atomic `claim_range`, reserve-before-read behavior, and the vault's shared lifecycle call survive those bounded checks.

Exact read-only counterexample, run from the repository root through `uv run python -`:

```python
import sys
sys.path.append("local_sim")
import certify_candidate as c
from test_linux_memory_binding import (
    good_report, SHA, WIDTH, DEPTH, BUDGET, STACK
)

r = good_report()
r["artifact_version"] = 2
r["status"] = "FAIL"
r["candidate_run"]["cap_applied"] = False
r["candidate_run"]["predict_error"] = "MemoryError"
r["negative_control"]["survived"] = True
checks, degraded = c.validate_memory_report(
    r, candidate_sha256=SHA, width=WIDTH, depth=DEPTH,
    budget=BUDGET, stack=STACK
)
print("failed", [n for n, ok, detail in checks if not ok])
print("degraded", bool(degraded))
print("domains", {s: c._domain(s) for s in ("5-8", "5b", "11")})
print("actual_stack", c._certified_stack())
```

Observed:

```text
failed []
degraded True
domains {'5-8': 'MATH', '5b': 'MATH', '11': 'MATH'}
actual_stack {'flopscope': '0.12.1+np2.2.6'}
```

Additional exact read-only probes, after the imports above:

```python
r = good_report()
r["instrument_sha256"] = "arbitrary-non-hash"
checks, degraded = c.validate_memory_report(
    r, candidate_sha256=SHA, width=WIDTH, depth=DEPTH,
    budget=BUDGET, stack=STACK
)
print("arbitrary_instrument_accepted", all(x[1] for x in checks) and not degraded)
import competition_score as cs
from scipy.stats import t
print("alpha_1e-6_df1", cs._t_ppf(1 - 1e-6 / 2, 1), t.ppf(1 - 1e-6 / 2, 1))
```

Observed:

```text
arbitrary_instrument_accepted True
alpha_1e-6_df1 10000.0 636619.7724192955
```

The executed vault subset called only `t_setup_once`, `t_weights_are_grader_twins`, `t_materialisation_is_billed`, `t_failure_semantics`, and `t_rows_have_identity` from `test_vault_lifecycle`. No full candidate/live-shape suite was launched by this reviewer.

**Unresolved ambiguity**

Post-worker preflight and the separately observed full mutation battery remain unverified in this review. Root reported workerq 574 queued for default preflight with 8 GiB RAM, 0 VRAM, 2 CPU, no GPU and no source snapshot. This is not `-Full`; eventual evidence must be attributed to actual executed source. No regenerated v3 Linux memory report was demonstrated here.

The existing Phase-1 vault is currently blocked from live-round evaluation. Its dormant evaluation path still checks the touch cap after loading truth and increments the counter only after successful completion (`vault.py:633–637`, `:728`). The shared lifecycle repair does not establish crash-safe or concurrent touch accounting. Repair those semantics on sacrificial fixtures before future enablement; do not open a new vault for this purpose.

**Cheapest discriminating follow-up**

One synthetic integration battery, no seed block required: exercise real `head()`/`check()` tagging, actual installed-version extraction, legacy failed and legacy merely-incomplete memory reports, and a stateless two-network memory control after releasing instrument-owned arrays. Run substantial memory work through workerq. Include production-path assertions for replica noise and protected-read routing before declaring those repairs complete.

**Checklist items triggered**

1, 4, 8, 10, 12, 14, 17, 18.

## Documentation follow-up verdict: ACCEPT within instrument-review scope

Reviewed `docs/WORKER_REVIEW_2026-09-10.md`, `docs/CURRENT_STATE.md`, `docs/GAME_PLAN.md`, and `docs/CODEX_EXECUTION_HANDOFF.md` after the original verdict. This second pass read documentation only and ran no fresh tests or label-consuming work.

The documents accurately retain the instrument REJECT, distinguish the bounded passing fixtures from production integration defects, and prescribe repairs rather than claiming they have been implemented. The independent totals are accurately scoped, including 26 certifier unit checks and the 30-check sacrificial subset. Pending default preflight 574 and the unverified separate full mutation result remain explicit. The memory-retention figure is correctly labeled source accounting rather than an observed peak. Legacy failure applicability, exposed data, current Phase-1 refusal, and future sacrificial-only vault repairs are stated appropriately. No material misstatement of this instrument review was found.

One wording clarification is recommended in WORKER_REVIEW's opening scope paragraph: “No estimator, instrument, gate, seed registry, protected truth, or vault was changed/read by this review” conflates source reading with mutation and label access. Source code was necessarily read. Prefer separate statements that no estimator/instrument/gate/registry/vault state was changed and no protected truth labels were opened. This does not alter the verdict or require an instrument change.

This documentation acceptance is not implementation acceptance, a new preflight result, or independent verification of the mathematical Skeptic's separate K4/compact-algorithm findings. Those findings are outside this follow-up's instrument scope. The original implementation verdict remains REJECT until production repairs, relevant evidence and independent acceptance exist.
