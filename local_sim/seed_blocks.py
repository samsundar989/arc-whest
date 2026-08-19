"""
SEED-BLOCK REGISTRY — one place that hands out MLP input seeds.

Standing rule (2026-08-06): **decision seeds are never reused across
experiments.** If experiment A picks a constant on seeds 1..500 and experiment B
re-measures it on the same 1..500, B is not independent evidence — it is A's
noise re-served. Reusing a block is the cheapest way to manufacture a result
that will not survive the private re-run, so blocks are allocated here, once,
and recorded in `experiments/LOG.md` with the entry that consumed them.

Every seed is an **input seed** in the evaluator's sense: the root that
`mint_provenance.seed_streams` expands into (weights, ground-truth draws,
estimator seed). Two blocks that do not overlap therefore produce networks,
truth samples, and estimator randomness that are all independent.

Layout — decimal digits are used as namespaces so a stray seed is readable at a
glance:

    2_026_08_06_xxx    reserved: the sealed vault (block V, 500 seeds)
    3_0dd_nnn_xxx      decision blocks: dd = day-of-month, nnn = block index
    9_0xx_xxx_xxx      legacy pre-2026-08-06 blocks (numbers NOT comparable to
                       anything minted after the provenance fix)

Legacy blocks are listed so they are never accidentally re-issued, but any
number measured on them used the pre-fix generator and is a different
realisation of the distribution.
"""

VAULT_BLOCK = ("V", 20_260_806_000, 500, "sealed 500-MLP holdout, 2026-08-06")

# --- retired: minted with the pre-provenance-fix generator -------------------
LEGACY_BLOCKS = [
    ("L1", 9_000_000, 64, "validate_estimator.py held-out check"),
    ("L2", 7_000_000, 50_000, "gpu_mint.py B3/G1 training mint (Phase C, retired)"),
    ("L3", 900_000, 6_000, "robustness.py fresh + stress distributions"),
    ("L4", 1_000, 200, "fit_calibration.py v0 per-layer calibration"),
    ("L5", 777_001, 1, "fit_calibration.py validation net"),
]

# --- live: allocated after the provenance fix; append, never edit ------------
DECISION_BLOCKS = [
    ("D01", 30_806_001_000, 600, "2026-08-06 constants audit: prune tau sweep"),
    ("D02", 30_806_002_000, 600, "2026-08-06 constants audit: rotation rank sweep"),
    ("D03", 30_806_003_000, 600, "2026-08-06 constants audit: C/B operating point"),
    ("D04", 30_806_004_000, 600, "2026-08-06 constants audit: pruned-patch bias bound"),
    ("D05", 30_806_005_000, 600, "2026-08-06 constants audit: v0 calibration table"),
    ("D06", 30_806_006_000, 600, "2026-08-06 invariant suite fresh-mint draws"),
    ("D07", 30_806_007_000, 600, "2026-08-06 theory verification P1/P2/P3"),
    ("D08", 30_806_008_000, 600, "moonshot G1: depth-8 low-rank cumulant flow (MOONSHOT.md)"),
    ("D09", 30_806_009_000, 600, "moonshot G2: depth-32 bias-vs-r sweep (MOONSHOT.md)"),
    ("D10", 30_806_010_000, 600, "moonshot: per-layer renormalization fitting set"),
    ("D11", 30_806_011_000, 600, "Sprint 4 DESIGN block: packed activation propagation "
                                 "(prefix sizing, bucket edges, Strassen depth, pilot size, "
                                 "on-neuron threshold). Tuning is PERMITTED here."),
    ("D12", 30_806_012_000, 600, "Sprint 4 DECISION block: LOCKED, SINGLE-USE. The S1 paired "
                                 "V*c gate is measured here ONCE against a frozen design. "
                                 "No constant may be changed after a D12 number is seen. SPENT "
                                 "2026-08-06 (S1 borderline 1.1996x cost-only; see LOG)."),
    ("D13", 30_806_013_000, 600, "Sprint 5 DESIGN block: round-trip compression and fixed-cost "
                                 "amortisation. Batched-vs-recursive Strassen, bucket/segment "
                                 "Pareto, N in {16384, 32768, 65536}, residual modelling. "
                                 "Tuning is PERMITTED here and nowhere else."),
    ("D14", 30_806_014_000, 600, "Sprint 5 DECISION block: LOCKED, SINGLE-USE. The paired V*C "
                                 "gate (>=1.15x, 95% CI excluding 1.0) is measured here ONCE "
                                 "against a design frozen on D13. Do not inspect until the "
                                 "design AND the residual model are frozen. SPENT 2026-08-06 "
                                 "(FAIL, 1.1216x; packed lane killed)."),
    ("D15", 30_806_015_000, 600, "Moonshot G0: layer-convention parity and the separate "
                                 "calibrated-K2 control (MOONSHOT.md). Infrastructure only -- "
                                 "no gate is decided here, and D08/D09 stay untouched so G1/G2 "
                                 "still have unseen data."),
    ("D16", 30_806_016_000, 600, "DESIGN block. Allocated 2026-08-10 for the layer-local "
                                 "control-variate design phase and wrongly recorded as released "
                                 "unspent. CORRECTED 2026-08-19: it is SPENT in part -- 270 of "
                                 "600 seeds drawn by five probes, and two of them "
                                 "(probe_whitening, probe_design_carrier) drew the SAME 24 "
                                 "networks, so those two kills are correlated evidence and must "
                                 "never be quoted as independent. Exact ranges, consumers and "
                                 "artifact hashes: experiments/seed_ledger.jsonl. 330 seeds "
                                 "remain free; use seed_ledger.claim() to take them."),
    ("D17", 30_806_017_000, 600, "DECISION block, LOCKED SINGLE-USE 2026-08-19. Allocated "
                                 "2026-08-10 and never drawn -- verified by static scan of the "
                                 "whole tree (seed_ledger.scan_sources) and by the empty ledger. "
                                 "Reserved for the next single-shot gate; see SINGLE_USE_BLOCKS."),
    ("D18", 30_806_018_000, 600, "DESIGN block, allocated 2026-08-19 for the interval-coverage "
                                 "study of the paired arithmetic promotion statistic (Skeptic "
                                 "finding D5 on the P0.1 slice). Tuning permitted; no gate is "
                                 "decided here."),
    ("D19", 30_806_019_000, 600, "DECISION block, LOCKED SINGLE-USE, allocated 2026-08-19 and "
                                 "unspent. Held for the v2.6 promotion gate once the Phase-2 "
                                 "shape is known -- kept distinct from D17 so a shape change "
                                 "does not burn the only clean decision block."),
]

# --- single-use blocks -------------------------------------------------------
# Listed so the rule is machine-readable rather than a convention in a doc.
# A block named here may be consumed by exactly one recorded experiment; the
# LOG.md entry that spends it is the receipt.
SINGLE_USE_BLOCKS = {
    "D12": "Sprint 4 / S1 paired decision (GAME_PLAN rev 9) — SPENT 2026-08-06",
    "D14": "Sprint 5 / packed-carrier paired decision (GAME_PLAN rev 11) — SPENT 2026-08-06",
    "D17": "reserved single-shot gate — allocated 2026-08-10, never drawn, LOCKED 2026-08-19",
    "D19": "v2.6 promotion gate once the Phase-2 shape is known — allocated 2026-08-19, unspent",
}

# Consumption — as opposed to allocation — is recorded in `seed_ledger.py` and
# `experiments/seed_ledger.jsonl`. `check_disjoint()` below proves that no two
# blocks ALLOCATE the same seeds; it cannot see a probe with a seed hardcoded
# into it, which is exactly how D16 was drawn dry while this file called it
# reusable. Take seeds with `seed_ledger.claim()`, never by writing a literal.


def _all():
    return [VAULT_BLOCK] + LEGACY_BLOCKS + DECISION_BLOCKS


def seeds(block_id):
    """The input seeds of a named block, as a list."""
    for bid, start, n, _why in _all():
        if bid == block_id:
            return [start + i for i in range(n)]
    raise KeyError(f"unknown seed block {block_id!r}; known: "
                   f"{[b[0] for b in _all()]}")


def check_disjoint():
    """Assert no two live blocks overlap. Called by the invariant suite."""
    live = [VAULT_BLOCK] + DECISION_BLOCKS
    spans = sorted((s, s + n - 1, bid) for bid, s, n, _ in live)
    for (a0, a1, ai), (b0, b1, bi) in zip(spans, spans[1:]):
        if b0 <= a1:
            raise AssertionError(f"seed blocks {ai} [{a0},{a1}] and {bi} [{b0},{b1}] overlap")
    return True


def describe():
    print(f"{'id':>5} {'start':>16} {'n':>7} {'end':>16}  purpose")
    print("-" * 96)
    for bid, start, n, why in _all():
        tag = "  (legacy)" if any(bid == b[0] for b in LEGACY_BLOCKS) else ""
        print(f"{bid:>5} {start:>16,} {n:>7,} {start + n - 1:>16,}  {why}{tag}")
    print("-" * 96)
    print(f"live blocks disjoint: {check_disjoint()}")


if __name__ == "__main__":
    describe()
