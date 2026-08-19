"""
One-shot backfill of `experiments/seed_ledger.jsonl` — the durable record of how
the ledger was seeded, kept in the repo so the reconstruction is auditable.

Every record below is read out of the SOURCE that drew the seeds, not out of a
document. Each consumer is cited with the line that fixes its range. Run once;
the ledger is append-only, so a second run duplicates records and
`seed_ledger --check` will then report the duplicates as DESIGN overlaps.

  uv run python experiments/2026-08-19_seed_ledger_backfill.py --dry-run
  uv run python experiments/2026-08-19_seed_ledger_backfill.py --write
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "local_sim"))

from seed_ledger import LEDGER_PATH, claim_range, describe  # noqa: E402

# (block, start, n, consumer, ts, artifact, note, paired_with)
BACKFILL = [
    (
        "D16", 30_806_016_000, 200, "local_sim/bias_floor_lab.py", "2026-08-11",
        "experiments/bias_floor_D16.json",
        "BLOCK = 'D16' (bias_floor_lab.py:45). Artifact carries 200 seeds, "
        "contiguous from the block start. LOG 2026-08-11 VARIANCE-OR-BIAS.",
    ),
    (
        "D16", 30_806_016_200, 24, "local_sim/probe_whitening.py", "2026-08-19",
        "experiments/2026-08-19_carrier_probes/white_probe.json",
        "SEED0 = 30_806_016_200 (probe_whitening.py:21), N_MLPS = 24 (line 20). "
        "LOG 2026-08-19 CARRIER AUDIT: point-set whitening DEAD, 0.9972x.",
    ),
    (
        "D16", 30_806_016_200, 24, "local_sim/probe_design_carrier.py", "2026-08-19",
        "experiments/2026-08-19_carrier_probes/design_probe.json",
        "SEED0 = 30_806_016_200 (probe_design_carrier.py:23), N_MLPS = 24 (line 22). "
        "*** SAME 24 NETWORKS AS probe_whitening. *** LOG 2026-08-19 CARRIER AUDIT: "
        "degree-3 design carrier DEAD against the shipped stack, 1.047x. That kill "
        "and the whitening kill are CORRELATED evidence and must never be quoted "
        "as two independent readings of the substitution mechanism.",
    ),
    (
        "D16", 30_806_016_300, 20, "local_sim/probe_carrier_transfer.py", "2026-08-19",
        "",
        "SEED0 = 30_806_016_300 (probe_carrier_transfer.py:20), N_MLPS = 20 (line 19). "
        "No JSON artifact was written; the result survives only in LOG 2026-08-19.",
    ),
    (
        "D16", 30_806_016_400, 20, "local_sim/probe_midlayer_cv.py", "2026-08-19",
        "experiments/2026-08-19_carrier_probes/midlayer_cv.json",
        "SEED0 = 30_806_016_400 (probe_midlayer_cv.py:52), N_MLPS = 20 (line 51). "
        "LOG 2026-08-19 STAGE 0: the mid-layer CV dies by identity.",
    ),
    (
        "D16", 30_806_016_500, 6, "local_sim/probe_strassen_carrier.py", "2026-08-19",
        "",
        "SEED0 = 30_806_016_500 (probe_strassen_carrier.py:61), range(6) (line 212). "
        "No JSON artifact; result in LOG 2026-08-19 (Strassen REOPENED, 1.267x counted).",
    ),
    # ----------------------------------------------------------------------- #
    # Found by seed_ledger.scan_sources() AFTER the six above were recorded —
    # which falsified prediction A3 ("exactly the six, no seventh"). These are
    # older consumptions nobody had written down anywhere.
    #
    # UPPER BOUNDS, deliberately. Where the count comes from a CLI argument the
    # documented/default run size is recorded. Over-recording consumption is the
    # SAFE direction: it can only stop a seed being re-drawn, never license one.
    # Each note names the line that fixes the start.
    # ----------------------------------------------------------------------- #
    (
        "D01", 30_806_001_000, 200, "local_sim/audit_constants.py", "2026-08-06",
        "experiments/constants_audit_2026-08-06.json",
        "BLOCKS['tau'] (audit_constants.py:65); --mlps default 200 (line 464). "
        "Constants audit: pruning tau sweep. UPPER BOUND at the default run size.",
    ),
    (
        "D02", 30_806_002_000, 200, "local_sim/audit_constants.py", "2026-08-06",
        "experiments/constants_audit_rank_2026-08-06.json",
        "BLOCKS['rank'] (audit_constants.py:65). Rotation-rank sweep. UPPER BOUND.",
    ),
    (
        "D03", 30_806_003_000, 200, "local_sim/audit_constants.py", "2026-08-06",
        "experiments/constants_audit_cb_2026-08-06.json",
        "BLOCKS['cb'] (audit_constants.py:65). C/B operating point. UPPER BOUND.",
    ),
    (
        "D04", 30_806_004_000, 200, "local_sim/audit_constants.py", "2026-08-06", "",
        "BLOCKS['patch'] (audit_constants.py:66). Pruned-patch bias bound. UPPER BOUND.",
    ),
    (
        "D05", 30_806_005_000, 200, "local_sim/audit_constants.py", "2026-08-06", "",
        "BLOCKS['cal'] (audit_constants.py:66). v0 calibration table. UPPER BOUND.",
    ),
    (
        "D06", 30_806_006_000, 200, "local_sim/edge_cases.py", "2026-08-06", "",
        "Individual literals at +1..+4, +11..+14, +21, +31..+32, +100 "
        "(edge_cases.py:199-292). Recorded as the first 200 of the block: an UPPER "
        "BOUND that covers every literal actually used. INFRA tier, nothing decided.",
    ),
    (
        "D07", 30_806_007_000, 320, "local_sim/theory_probe.py", "2026-08-06", "",
        "BLOCK_D07 (theory_probe.py:57) with P1_STARTS=4 and --mlps default 64: P1 "
        "consumes 4*64 = 256, P3 the next 64. 320 total. INFRA tier.",
    ),
    (
        "D11", 30_806_011_000, 500, "local_sim/packed_sparse_probe.py", "2026-08-06",
        "",
        "Two literal offsets: 30_806_011_000 + i (line 1160) and +300 + i (line 1438), "
        "--mlps default 20 but the recorded design runs used 200 "
        "(docs/VALIDATION.md 2). Recorded as 011000..011499, an UPPER BOUND covering "
        "both arms. Sprint 4 DESIGN.",
    ),
    (
        "D11", 30_806_011_500, 7, "local_sim/test_packed_sparse.py", "2026-08-06", "",
        "Seven exact literals 011500..011506 (test_packed_sparse.py:216-397).",
    ),
    (
        "D12", 30_806_012_000, 200, "local_sim/packed_sparse_probe.py --mode decide4",
        "2026-08-06", "",
        "SINGLE-USE, SPENT. Sprint 4 S1 paired V*c gate, 1.1996x. Reached via "
        "_bs_seeds(args.block)[:args.mlps] (line 1850) — no literal, so the literal "
        "scan cannot see it. 200 labelled networks per docs/VALIDATION.md 2.",
    ),
    (
        "D13", 30_806_013_000, 200, "local_sim/packed_sparse_probe.py", "2026-08-06", "",
        "_bs_seeds('D13')[:args.mlps] (line 1692). Sprint 5 DESIGN. UPPER BOUND at 200.",
    ),
    (
        "D13", 30_806_013_500, 1, "local_sim/test_packed_sparse.py", "2026-08-06", "",
        "One exact literal 30_806_013_500 (test_packed_sparse.py:568).",
    ),
    (
        "D14", 30_806_014_000, 200, "local_sim/packed_sparse_probe.py --mode decide5",
        "2026-08-06", "",
        "SINGLE-USE, SPENT 2026-08-06: FAIL at 1.1216x [1.1188, 1.1244], packed lane "
        "killed. Reached via _bs_seeds(args.block) — invisible to the literal scan.",
    ),
    (
        "D15", 30_806_015_000, 200, "local_sim/moonshot_g0.py", "2026-08-07",
        "experiments/2026-08-06_moonshot_g0.json",
        "block_seeds('D15') via --block default 'D15' (moonshot_g0.py:209). INFRA "
        "tier; G0 decides nothing. UPPER BOUND at 200.",
    ),
    (
        "D11", 30_806_011_000, 200, "local_sim/packed_truth.py", "2026-08-06", "",
        "block_seeds(block) — labels D11's 200 networks at 2^23 samples "
        "(docs/VALIDATION.md 2). Overlaps packed_sparse_probe by design: this is "
        "the TRUTH for the very networks that probe scored, not a second sample.",
        "local_sim/packed_sparse_probe.py",
    ),
]


def main():
    dry = "--write" not in sys.argv
    if dry:
        print("DRY RUN — pass --write to append to the ledger\n")
    elif os.path.exists(LEDGER_PATH) and os.path.getsize(LEDGER_PATH) > 0:
        print(f"REFUSING: {LEDGER_PATH} already exists and is non-empty.")
        print("The ledger is append-only; a second backfill would duplicate records.")
        return 1
    for rec in BACKFILL:
        block, start, n, consumer, ts, artifact, note = rec[:7]
        paired_with = rec[7] if len(rec) > 7 else ""
        claim_range(
            block, start, n, consumer, ts=ts, artifact=artifact, note=note,
            backfilled=True, paired_with=paired_with, dry_run=dry,
        )
        print(f"  {'would record' if dry else 'recorded'} "
              f"{block}[{start:,}..{start + n - 1:,}]  {consumer}")
    if not dry:
        print()
        describe()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
