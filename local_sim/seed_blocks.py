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
    3_0dd_nnn_xxx      PHASE-1 decision blocks (256 x 32): dd = day-of-month,
                       nnn = block index
    4_0mdd_nnn_xxx     PHASE-2 blocks (1024 x 16). A DIFFERENT SHAPE IS A
                       DIFFERENT EXPERIMENT: the same seed mints a
                       different-shaped network, so a Phase-1 block cannot be
                       reused at Phase 2 and the leading digit says which world a
                       stray seed belongs to at a glance.
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
    # ---- PHASE 2, 1024 x 16. A new shape is a new world: every block above
    # allocates 256x32 networks and none of them is reusable here.
    ("P01", 40_821_001_000, 600, "PHASE-2 DESIGN block, allocated 2026-08-21 for "
                                 "P2-CP1, the 1024x16 baseline ladder (analytic "
                                 "Price/Hermite rungs vs sampling arms) under the "
                                 "contract registered in experiments/LOG.md that "
                                 "day. Tuning and implementation choices are "
                                 "PERMITTED here; no promotion gate is decided on "
                                 "it. 96 seeds claimed by the truth generator; the "
                                 "remaining 504 are held for a DISJOINT P2-CP2 "
                                 "draw so the deployment gate is not measured on "
                                 "the networks that chose the candidate."),
    # ---- PHASE 2 evidence blocks, allocated 2026-09-01 under WP0E's structure
    # (LOG 2026-08-31) but NOT its stated purpose. WP0E framed P02/P03 around
    # carrier basis-count selection; the WP2 slice, independently ACCEPTed on
    # 2026-09-01, then measured the carrier's advantage over random orthogonal
    # blocks at LIVE WIDTH as 0.155%, which demotes the carrier to a support arm
    # and makes basis-count selection a poor use of 96 fresh networks. The
    # allocation is kept, the purpose is restated. Samuel's sign-off on the
    # allocation is WAIVED (solo, 2026-09-01); the Rule-1 contract per slice is
    # NOT waived.
    ("P02", 40_901_002_000, 96,  "PHASE-2 DESIGN block. Selection over the Rung-1 "
                                 "state/corrector architecture: the offline learned "
                                 "cumulant-feature corrector and the AUGMENT-K2 "
                                 "diagonal-cumulant state. Tuning, hyper-parameters "
                                 "and implementation choices are PERMITTED here; NO "
                                 "promotion gate is decided on it, and nothing "
                                 "selected here may be reported without P03."),
    ("P03", 40_901_003_000, 96,  "PHASE-2 DESIGN block, DISJOINT from P02 by "
                                 "construction. Holds the independent replication of "
                                 "whatever P02 selects, so a selected configuration "
                                 "is never reported on the networks that chose it. "
                                 "Claiming P03 for selection instead of replication "
                                 "defeats the only reason it exists."),
    ("P04", 40_901_004_000, 100, "PHASE-2 DECIDE block, LOCKED SINGLE-USE. The one "
                                 "paired arithmetic comparison of the final candidate "
                                 "against graded v2.7, through "
                                 "competition_score.decide(gate=...) with a gate "
                                 "fixed in a committed contract BEFORE the block is "
                                 "touched. One claim, ever. Spending it on a "
                                 "rehearsal, a debug run, or a configuration still "
                                 "under selection destroys the only clean Phase-2 "
                                 "decision this project has."),
    # --- CONTRACT WP15, allocated 2026-09-02 BEFORE any WP15 data exists ------
    # The remaining Phase-2 problem is a MAP from the regenerating flow's own
    # carried state to the rank-r correction of both kappa_3 read-slices. Two
    # arms need three disjoint roles, and conflating any two of them is the
    # failure this repository has already paid for: a map fitted and validated on
    # the same networks is not evidence, and two arms scored on one validation
    # set is one number quoted twice.
    ("P05", 40_902_005_000, 160, "PHASE-2 DESIGN block, allocated 2026-09-02 for "
                                 "CONTRACT WP15 Arm L: the SUPERVISED FIT of the "
                                 "state -> correction map, leave-networks-out. "
                                 "Tuning, basis selection, coefficient counts and "
                                 "regularisation are PERMITTED here and NOWHERE "
                                 "else. No gate is decided on it, and nothing "
                                 "fitted here may be reported without P06."),
    ("P06", 40_902_006_000, 128, "PHASE-2 DESIGN block, allocated 2026-09-02 for "
                                 "CONTRACT WP15: the end-to-end validation of BOTH "
                                 "arms on networks neither arm was built on. "
                                 "DISJOINT from P05 by construction and to be "
                                 "claimed in TWO disjoint halves of 64 -- "
                                 "P06[..000..063] for Arm M, P06[..064..127] for "
                                 "Arm L -- so the two arms are never scored on the "
                                 "same networks. No tuning of any kind here."),
    # --- CONTRACT T2, allocated 2026-09-10 BEFORE any T2 data exists ----------
    # T2 asks whether a learned corrector on the exact-K3 chain's own carried
    # state is worth >= 1.15x held-out. The Slice-5 handoff specified training on
    # the organisers' public 1,100 networks, and `docs/VALIDATION.md` section 0
    # forbids exactly that: the public split is CONFIRM tier, read-only, and "no
    # constant may ever again be fitted here". The rule wins over the handoff, so
    # the corpus is minted DECIDE-tier networks and the public split is read at
    # the end as a check that chooses nothing. Two blocks because a map fitted
    # and validated on the same networks is not evidence -- the same reason P05
    # and P06 are two.
    ("P07", 40_910_007_000, 384, "PHASE-2 DESIGN block, allocated 2026-09-10 for "
                                 "CONTRACT T2: the SUPERVISED FIT of the "
                                 "chain-state -> final-residual corrector on the "
                                 "exact factored K3 chain. Tuning, feature "
                                 "choice, head width, cross-fitting folds and "
                                 "shrinkage are PERMITTED here and NOWHERE else. "
                                 "No gate is decided on it, and nothing fitted "
                                 "here may be reported without P08. Replaces the "
                                 "handoff's public-split corpus, which "
                                 "VALIDATION.md section 0 forbids."),
    ("P08", 40_910_008_000, 128, "PHASE-2 DESIGN block, allocated 2026-09-10 for "
                                 "CONTRACT T2: the held-out read of whatever P07 "
                                 "fits, on networks the fit never saw. DISJOINT "
                                 "from P07 by construction. The 1.15x gate is "
                                 "read HERE and nowhere else. No tuning of any "
                                 "kind, no feature selection, no re-fitting -- "
                                 "claiming P08 for anything but the single "
                                 "held-out read defeats the only reason it "
                                 "exists."),
    ("P09", 40_912_009_000, 128, "PHASE-2 fresh single-use DECIDE block for "
                                 "the preregistered F21 fused-L4 versus original-L4 "
                                 "paired arithmetic score gate. Original L2 is "
                                 "robustness context only; no fitting or selection. "
                                 "Register after exposed P01 DESIGN passed and "
                                 "independent Skeptic accepted; claim once before mint."),
    ("P10", 40_913_010_000, 128, "PHASE-2 fresh single-use DECIDE block for "
                                 "the preregistered post-P09 F21 versus original-L4 "
                                 "paired arithmetic score gate. Original L2 is "
                                 "robustness context only; no fitting or selection. "
                                 "Claim once before mint; local score-only wall "
                                 "and residual limits are disabled per the P10 contract."),
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
    "P04": "the one Phase-2 DECIDE comparison against graded v2.7 — allocated 2026-09-01, unspent",
    # Added 2026-09-10 after the T2 Skeptic. P08's own registry text already said
    # "claiming P08 for anything but the single held-out read defeats the only
    # reason it exists" — and that was PROSE, so nothing stopped the block being
    # read three times while the instrument was being repaired. Registering it
    # here makes a second CLAIM refuse. It does NOT count re-READS of an
    # already-claimed block, which is the deeper gap: see READ-COUNTED ARTIFACTS
    # below.
    "P08": "the one Phase-2 T2 held-out read — allocated and claimed 2026-09-10, SPENT",
    "P09": "the one fresh Phase-2 F21 DECIDE read — registered 2026-09-12, unspent",
    "P10": "the one fresh Phase-2 F21 DECIDE read — registered 2026-09-13, unspent",
}

# --- READ-COUNTED ARTIFACTS, owed --------------------------------------------
# `SINGLE_USE_BLOCKS` protects a block from being CLAIMED twice. It does not
# protect a claimed block's artifact from being READ twice, and on 2026-09-10 the
# T2 held-out corpus was read three times: two runs of a broken instrument, then
# the repaired one. Nothing recorded that, and the LOG entry said "single
# held-out read" in good faith.
#
# The repository already has the right machinery in exactly one place —
# `local_sim/vault.py`, which is touch-counted with a permanent log and a
# hard cap. A single-use held-out block deserves the same shape:
#   * a `<block>_reads.jsonl` beside the artifact, appended by whatever reads it;
#   * the consumer, the instrument's sha256 and the timestamp on every line;
#   * a cap, and a refusal past it.
# BUILT 2026-09-10 (Stage 0 R3): `local_sim/truth_reads.py`, register at
# `experiments/truth_reads.jsonl`, tests in `local_sim/test_truth_reads.py`,
# wired into preflight step 9. It reserves BEFORE the labels are exposed and
# counts an aborted read, because a process that died after `np.load` returned
# has still seen the numbers. P08's three historical reads are backfilled and
# marked as such, so the block now reports EXHAUSTED to anything that asks.
# This tuple stays as the record of what was owed and when it was paid.
READ_COUNTED_OWED = ("P08",)

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
