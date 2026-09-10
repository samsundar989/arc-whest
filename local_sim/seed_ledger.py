"""
SEED LEDGER — append-only record of which seeds were actually drawn, by whom.

Stage 0 / P0.4 of `docs/FRONTIER_ARCHITECTURE_PROPOSAL.md`. Contract and
predictions A1-A6: `experiments/LOG.md`, 2026-08-19 (night).

WHY THIS FILE EXISTS
--------------------
`seed_blocks.py` allocates ranges. Nothing recorded *consumption*, so for nine
days the registry said of D16 "NO SEED WAS EVER DRAWN. Reusable." while five
probes had drawn 270 of its seeds and two of them had drawn the SAME 24. Nothing
failed, because nothing was checking. `check_disjoint()` proves allocations do
not overlap; it cannot see a probe with `SEED0` typed into it.

Three layers, weakest to strongest:

  1. `claim()` / `claim_range()` — the sanctioned way to obtain seeds. Hard-fails
     on an unregistered block, a range that leaves its block, any overlap inside a
     DECISION block, or a second claim on a single-use block. Writes the ledger.
  2. `check_ledger()` — re-validates the whole ledger from scratch. Run by
     preflight, so a hand-edited ledger cannot pass quietly.
  3. `scan_sources()` — static scan of the tree for integer literals inside a
     registered block that have no ledger record. This is the layer that catches
     the failure mode that actually happened: a seed hardcoded into a probe.

The ledger is APPEND-ONLY by convention and by check: `check_ledger()` fails if a
record is malformed, and the file is tracked in git so a rewrite is visible in the
diff. It is not cryptographically sealed — that is the vault's job, and the vault
is what a seed ledger cannot substitute for.

Run:  uv run python local_sim/seed_ledger.py            # show the ledger
      uv run python local_sim/seed_ledger.py --check    # what preflight runs
      uv run python local_sim/seed_ledger.py --scan     # static source scan
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.append(_HERE)

from seed_blocks import (DECISION_BLOCKS, LEGACY_BLOCKS,  # noqa: E402
                         SINGLE_USE_BLOCKS, VAULT_BLOCK)

LEDGER_PATH = os.path.join(_ROOT, "experiments", "seed_ledger.jsonl")

#: Tier governs how strictly a block's ranges are policed.
#:   VAULT     — touch-counted elsewhere (`vault.py`); the ledger only records.
#:   DECISION  — overlap is a HARD ERROR. A decision measured twice on the same
#:               networks is one measurement quoted twice.
#:   SINGLE_USE— DECISION, plus: at most one claim, ever.
#:   DESIGN    — overlap is PERMITTED (tuning re-runs are legitimate) but always
#:               REPORTED, because two "independent" design kills on the same
#:               networks are correlated evidence and must not read as two.
#:   INFRA     — parity/provenance/invariant draws; nothing is decided on them.
#:   LEGACY    — pre-provenance-fix; never re-issue.
TIERS: Dict[str, str] = {
    "V": "VAULT",
    "D01": "DESIGN", "D02": "DESIGN", "D03": "DESIGN", "D04": "DESIGN",
    "D05": "DESIGN", "D06": "INFRA", "D07": "INFRA",
    "D08": "DESIGN", "D09": "DESIGN", "D10": "DESIGN",
    "D11": "DESIGN", "D12": "SINGLE_USE",
    "D13": "DESIGN", "D14": "SINGLE_USE",
    "D15": "INFRA",
    "D16": "DESIGN",
    "D17": "SINGLE_USE",
    "D18": "DESIGN",
    "D19": "SINGLE_USE",
    # ---- Phase 2, 1024 x 16 ----
    "P01": "DESIGN",
    # Allocated 2026-09-01. P02 selects, P03 replicates what P02 selected, P04
    # decides ONCE. Claiming P03 for selection collapses the pair into one
    # DESIGN block and is the failure this split exists to prevent.
    "P02": "DESIGN",
    "P03": "DESIGN",
    "P04": "SINGLE_USE",
    # Allocated 2026-09-02 with CONTRACT WP15. P05 FITS the state->correction
    # map (tuning permitted); P06 validates BOTH arms on networks neither was
    # built on, in two disjoint halves of 64 so the arms never share a network.
    #
    # These two are here because SKEPTIC WP14-FOLLOWUP found them registered in
    # `seed_blocks.py` and NOT here: `block_span('P05')` worked, `tier_of('P05')`
    # raised, and WP15's first instrument run would have died on its first claim.
    # `test_seed_ledger.py` now asserts the two registries agree in BOTH
    # directions, so a block can never again exist in one and not the other.
    "P05": "DESIGN",
    "P06": "DESIGN",
    # Allocated 2026-09-10 with CONTRACT T2. P07 FITS the chain-state ->
    # final-residual corrector (tuning permitted); P08 is the single held-out
    # read where the 1.15x gate is decided, on networks the fit never saw.
    # The pair exists for the same reason P05/P06 do, and it is minted DECIDE
    # data rather than the handoff's public split because `docs/VALIDATION.md`
    # section 0 makes the public split read-only.
    "P07": "DESIGN",
    # P08 is registered SINGLE_USE, not DESIGN, from 2026-09-10: it is the one
    # held-out read of the T2 gate and its own registry text always said so. It
    # was prose until the T2 Skeptic pointed out that nothing enforced it.
    "P08": "SINGLE_USE",
}

#: Strictness ordering, so a tier CHANGE can be judged as a tightening or a
#: loosening rather than merely "different". Only tightening is permitted after a
#: claim has been recorded.
TIER_STRICTNESS: Dict[str, int] = {
    "LEGACY": 0, "INFRA": 1, "DESIGN": 2, "DECISION": 3,
    "SINGLE_USE": 4, "VAULT": 5,
}



@dataclass(frozen=True)
class Claim:
    block: str
    start: int
    n: int
    consumer: str
    tier: str
    ts: str
    note: str = ""
    artifact: str = ""
    artifact_sha256: str = ""
    commit: str = ""
    backfilled: bool = False
    #: Name the consumer this claim is DELIBERATELY on the same networks as.
    #: Truth labelling and the probe that scores against it are one experiment,
    #: not two correlated ones, and the overlap report must say so.
    paired_with: str = ""
    #: "claim" (the default, and every historical record), "finalize", or "pair".
    #:
    #: A claim necessarily records its artifact PATH before the artifact exists,
    #: so its `artifact_sha256` is necessarily blank at claim time -- and nothing
    #: used to fill it in. That is a fail-OPEN hole: a truth set could be swapped
    #: after the arms were scored against it and the ledger would still validate.
    #: A "finalize" record closes one claim's artifact by appending its hash
    #: after the file is written. Finalize records are NOT consumptions: they are
    #: excluded from overlap and SINGLE_USE accounting, so closing a claim can
    #: never look like a second claim on the same block.
    #:
    #: A "pair" record declares, AFTER the fact, that two consumers drawing the
    #: same range did so deliberately. `paired_with` can only be set at claim
    #: time, and the second consumer of a deliberate overlap is by definition not
    #: known when the first claims -- so a deliberate overlap whose intent was
    #: recorded only in prose reads to the machine as an accident. That is
    #: exactly what happened to `P01[40,821,001,168..175]` on 2026-09-02, and it
    #: held `preflight` RED for four commits. Pair records are NOT consumptions
    #: and are excluded from every accounting the same way finalizations are.
    kind: str = "claim"

    @property
    def end(self) -> int:
        return self.start + self.n - 1

    def overlaps(self, other: "Claim") -> bool:
        return self.block == other.block and self.start <= other.end and other.start <= self.end


# --------------------------------------------------------------------------- #
# block metadata
# --------------------------------------------------------------------------- #


def _blocks() -> Dict[str, Tuple[int, int, str]]:
    out: Dict[str, Tuple[int, int, str]] = {}
    for bid, start, n, why in [VAULT_BLOCK] + DECISION_BLOCKS:
        out[bid] = (start, n, why)
    return out


def block_span(block: str) -> Tuple[int, int]:
    b = _blocks()
    if block not in b:
        raise KeyError(
            f"unregistered seed block {block!r}. Allocate it in seed_blocks.py "
            f"first; known: {sorted(b)}"
        )
    start, n, _ = b[block]
    return start, start + n - 1


def tier_of(block: str) -> str:
    if block in TIERS:
        return TIERS[block]
    if any(block == b[0] for b in LEGACY_BLOCKS):
        return "LEGACY"
    raise KeyError(f"block {block!r} has no tier in seed_ledger.TIERS")


# --------------------------------------------------------------------------- #
# ledger I/O
# --------------------------------------------------------------------------- #


def ledger(path: str = LEDGER_PATH) -> List[Claim]:
    if not os.path.exists(path):
        return []
    out: List[Claim] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                d = json.loads(line)
                out.append(Claim(**d))
            except Exception as exc:
                raise ValueError(f"{path}:{lineno} is not a valid ledger record: {exc}") from exc
    return out


#: Set to "1" to allow a ledger write from inside a job-queue snapshot.
#: There is no legitimate use; it exists so the refusal can be exercised.
_SNAPSHOT_OVERRIDE = "ARC_WHEST_LEDGER_ALLOW_SNAPSHOT"


def _in_job_snapshot(path: str) -> bool:
    """Is this ledger inside a `gpuq`/`workerq` job snapshot, not the repo?

    A queued job runs in a FROZEN GIT WORKTREE under the queue's state
    directory. `experiments/seed_ledger.jsonl` is tracked, so it exists
    inside that snapshot -- and a claim written there is deleted with the
    snapshot while the live ledger never hears about it.

    THIS ALREADY HAPPENED. On 2026-09-02 the WP9 truth run was submitted to
    `gpuq`; `p2_truth.py` called `claim()` and `finalize()` exactly as it
    should, both wrote into the snapshot, and 96 `P02` seeds were really
    drawn while the live ledger still reported them FREE. An independent
    review found it. The failure is silent, it defeats the one guarantee
    this module exists to provide -- that no two experiments share networks
    -- and `docs/SETUP_WINDOWS.md` 7.1 had warned about exactly this class
    of loss for ARTIFACTS without anyone noticing the ledger is one.
    """
    norm = os.path.normcase(os.path.abspath(path)).replace("\\", "/")
    return "/gpuq/snapshots/" in norm or "/workerq/snapshots/" in norm


def _append(claim: Claim, path: str = LEDGER_PATH) -> None:
    if _in_job_snapshot(path) and os.environ.get(_SNAPSHOT_OVERRIDE) != "1":
        raise RuntimeError(
            "REFUSED: this ledger is inside a job-queue SNAPSHOT\n"
            f"  {path}\n"
            "A claim written here is discarded with the snapshot, while the "
            "seeds are really drawn -- so the live ledger reports them FREE "
            "and a later experiment can silently reuse the same networks.\n"
            "Claim seeds in the live repository BEFORE submitting, and pass "
            "the explicit seed range to the queued job."
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(claim), sort_keys=True) + "\n")


def _sha256(path: str) -> str:
    p = path if os.path.isabs(path) else os.path.join(_ROOT, path)
    if not os.path.exists(p):
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# consumption view
# --------------------------------------------------------------------------- #


def consumed_ranges(block: str, path: str = LEDGER_PATH) -> List[Tuple[int, int, str]]:
    return sorted(
        (c.start, c.end, c.consumer)
        for c in ledger(path)
        if c.block == block and c.kind == "claim"
    )


def claims(path: str = LEDGER_PATH) -> List[Claim]:
    """Consumption records only. `kind == "finalize"` is metadata, not a draw."""
    return [c for c in ledger(path) if c.kind == "claim"]


def finalizations(path: str = LEDGER_PATH) -> List[Claim]:
    return [c for c in ledger(path) if c.kind == "finalize"]


def pairings(path: str = LEDGER_PATH) -> List[Claim]:
    """Records declaring an overlap deliberate. Not consumptions."""
    return [c for c in ledger(path) if c.kind == "pair"]


def is_paired(a: Claim, b: Claim, path: str = LEDGER_PATH) -> bool:
    """Is this overlap DECLARED deliberate -- at claim time or afterwards?

    Two routes, and both are explicit machine-readable declarations rather than
    prose in a `note`:
      * either claim named the other in `paired_with` when it was written; or
      * a later `kind == "pair"` record names both consumers on a range covering
        the overlap.
    """
    if (b.consumer in a.paired_with) or (a.consumer in b.paired_with):
        return True
    lo, hi = max(a.start, b.start), min(a.end, b.end)
    for p in pairings(path):
        if p.block != a.block or p.start > lo or p.end < hi:
            continue
        named = [x.strip() for x in p.consumer.split("|") if x.strip()]
        if a.consumer in named and b.consumer in named:
            return True
    return False


def pairing_reason(a: Claim, b: Claim, path: str = LEDGER_PATH) -> str:
    """WHY this overlap is deliberate, in the words of the record that says so.

    A binary deliberate/accidental tag was not enough and the P01 case is why:
    that overlap is BOTH deliberate AND correlated -- reusing WP13's eight
    networks to mint truth was a choice, and the two results still must never be
    quoted as independent. A canned "one experiment, two roles" would have
    erased the warning. So a `pair` record's own note is the tag.
    """
    if (b.consumer in a.paired_with) or (a.consumer in b.paired_with):
        return "SAME NETWORKS BY DESIGN (one experiment, two roles)"
    found = ""
    lo, hi = max(a.start, b.start), min(a.end, b.end)
    for p in pairings(path):
        if p.block != a.block or p.start > lo or p.end < hi:
            continue
        named = [x.strip() for x in p.consumer.split("|") if x.strip()]
        if a.consumer in named and b.consumer in named:
            found = "DECLARED DELIBERATE: " + (p.note or "no reason recorded")
    # The LAST matching declaration wins. The ledger is append-only, so a pair
    # note that turns out to be WRONG cannot be edited -- it can only be
    # superseded by a later one. The first P01 note asserted a red-gate window
    # that `SKEPTIC WP14-FOLLOWUP` then disproved from git, and this is how that
    # correction reaches a reader without rewriting history.
    return found



def declare_pair(block: str, start: int, n: int, consumers: List[str], *,
                 note: str, ts: str, path: str = LEDGER_PATH) -> None:
    """Append a `pair` record. Refuses unless the overlap it describes is REAL.

    A pairing that does not correspond to an actual overlap in the record would
    be a licence issued against nothing, so the overlap is checked first.
    """
    if len(consumers) < 2:
        raise ValueError("a pairing names at least two consumers")
    tier = tier_of(block)
    lo, hi = block_span(block)
    end = start + n - 1
    if start < lo or end > hi:
        raise ValueError(f"pair {block}[{start:,}..{end:,}] leaves its block")
    real = [(a, b) for a, b in collisions(path)
            if a.block == block and a.consumer in consumers
            and b.consumer in consumers
            and max(a.start, b.start) >= start and min(a.end, b.end) <= end]
    if not real:
        raise ValueError(
            f"REFUSED: no recorded overlap on {block}[{start:,}..{end:,}] between "
            f"{consumers}. A pairing declares an existing overlap deliberate; it "
            "cannot license one that does not exist.")
    rec = Claim(block=block, start=start, n=n, consumer=" | ".join(consumers),
                tier=tier, ts=ts, note=note, commit=_git_commit(), kind="pair")
    _append(rec, path)


def consumed_seeds(block: str, path: str = LEDGER_PATH) -> int:
    """DISTINCT seeds consumed — overlapping claims are counted once."""
    covered = set()
    for s, e, _ in consumed_ranges(block, path):
        covered.update(range(s, e + 1))
    return len(covered)


def free_ranges(block: str, path: str = LEDGER_PATH) -> List[Tuple[int, int]]:
    lo, hi = block_span(block)
    taken = set()
    for s, e, _ in consumed_ranges(block, path):
        taken.update(range(max(s, lo), min(e, hi) + 1))
    out, run_start = [], None
    for x in range(lo, hi + 2):
        if x <= hi and x not in taken:
            if run_start is None:
                run_start = x
        elif run_start is not None:
            out.append((run_start, x - 1))
            run_start = None
    return out


def collisions(path: str = LEDGER_PATH) -> List[Tuple[Claim, Claim]]:
    """Every pair of CLAIM records that draw the same seed.

    `finalize` records carry their claim's range for traceability but are not
    consumptions, so they can never collide with anything.
    """
    recs = claims(path)
    out = []
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            if recs[i].overlaps(recs[j]):
                out.append((recs[i], recs[j]))
    return out


# --------------------------------------------------------------------------- #
# claiming — the sanctioned path
# --------------------------------------------------------------------------- #


def claim_range(
    block: str,
    start: int,
    n: int,
    consumer: str,
    *,
    note: str = "",
    artifact: str = "",
    ts: str,
    backfilled: bool = False,
    paired_with: str = "",
    path: str = LEDGER_PATH,
    dry_run: bool = False,
) -> List[int]:
    """Record a consumption and return the seeds. Hard-fails rather than warns.

    `ts` is required and explicit: this module must never read the clock, so a
    replay of a historical claim carries the historical date.
    """
    lo, hi = block_span(block)
    tier = tier_of(block)
    if n <= 0:
        raise ValueError("n must be positive")
    end = start + n - 1
    if start < lo or end > hi:
        raise ValueError(
            f"claim {block}[{start:,}..{end:,}] leaves its block [{lo:,}..{hi:,}]"
        )

    existing = claims(path)
    mine = [c for c in existing if c.block == block]
    cand = Claim(
        block=block, start=start, n=n, consumer=consumer, tier=tier, ts=ts,
        note=note, artifact=artifact, artifact_sha256=_sha256(artifact) if artifact else "",
        commit=_git_commit(), backfilled=backfilled, paired_with=paired_with,
    )

    if tier == "SINGLE_USE" and mine:
        raise PermissionError(
            f"{block} is SINGLE_USE and was already claimed by "
            f"{mine[0].consumer!r} on {mine[0].ts}. A second measurement on the same "
            f"networks is the same measurement quoted twice. Allocate a new block."
        )
    if tier in ("DECISION", "SINGLE_USE", "VAULT"):
        clash = [c for c in mine if c.overlaps(cand)]
        if clash:
            raise PermissionError(
                f"{block} is {tier}: claim [{start:,}..{end:,}] overlaps "
                f"{clash[0].consumer!r} [{clash[0].start:,}..{clash[0].end:,}]. "
                f"Free ranges: {free_ranges(block, path)}"
            )
    else:
        clash = [c for c in mine if c.overlaps(cand)]
        if clash and not dry_run:
            print(
                f"  NOTE {block} is {tier}: [{start:,}..{end:,}] overlaps "
                f"{clash[0].consumer!r}. Permitted, but the two results are "
                f"CORRELATED evidence and must not be quoted as independent.",
                file=sys.stderr,
            )

    if not dry_run:
        _append(cand, path)
    return [start + i for i in range(n)]


def claim(
    block: str,
    n: int,
    consumer: str,
    *,
    ts: str,
    note: str = "",
    artifact: str = "",
    path: str = LEDGER_PATH,
    dry_run: bool = False,
) -> List[int]:
    """Claim the next `n` unconsumed seeds of `block`, contiguous."""
    for lo, hi in free_ranges(block, path):
        if hi - lo + 1 >= n:
            return claim_range(
                block, lo, n, consumer, note=note, artifact=artifact, ts=ts,
                path=path, dry_run=dry_run,
            )
    raise RuntimeError(
        f"{block} has no free contiguous run of {n} seeds; free: {free_ranges(block, path)}"
    )


# --------------------------------------------------------------------------- #
# checks — what preflight runs
# --------------------------------------------------------------------------- #


def finalize(
    consumer: str,
    artifact: str,
    *,
    ts: str,
    note: str = "",
    path: str = LEDGER_PATH,
) -> str:
    """Close a claim's artifact by appending its sha256. Returns the hash.

    Call this AFTER the artifact is written. It refuses to record a hash for a
    file that does not exist, so it cannot manufacture provenance for an artifact
    that was never produced, and it refuses to finalize an artifact no claim
    named, so a stray file cannot acquire the authority of a claimed one.
    """
    recs = ledger(path)
    owners = [c for c in recs if c.kind == "claim"
              and c.consumer == consumer and c.artifact == artifact]
    if not owners:
        raise PermissionError(
            f"no claim by {consumer!r} names artifact {artifact!r}. finalize() "
            "closes an existing claim; it does not create provenance."
        )
    if not os.path.exists(os.path.join(_ROOT, artifact)) and not os.path.exists(artifact):
        raise FileNotFoundError(
            f"REFUSED: {artifact} does not exist. A hash cannot be recorded for "
            "an artifact that was never written."
        )
    digest = _sha256(artifact)
    if not digest:
        raise FileNotFoundError(f"REFUSED: could not hash {artifact}")

    prior = [c for c in recs if c.kind == "finalize"
             and c.consumer == consumer and c.artifact == artifact]
    for q in prior:
        if q.artifact_sha256 != digest:
            raise PermissionError(
                f"REFUSED: {artifact} was already finalized by {consumer!r} at "
                f"{q.artifact_sha256[:12]} and now hashes to {digest[:12]}. The "
                "artifact changed after it was closed; that is the exact event "
                "this record exists to make impossible to hide."
            )
        return digest                                   # idempotent re-finalize

    o = owners[0]
    _append(Claim(
        block=o.block, start=o.start, n=o.n, consumer=consumer, tier=o.tier,
        ts=ts, note=note or "artifact finalized", artifact=artifact,
        artifact_sha256=digest, commit=_git_commit(), kind="finalize",
    ), path)
    return digest



def _retier_covers(recs, block: str, old_tier: str, new_tier: str) -> bool:
    """Is this tier change DECLARED, and is it a tightening?

    Both conditions, not either. A declared LOOSENING is still an error: the
    point of the declaration is to make a deliberate hardening visible, never to
    provide a route around a block's own rules.
    """
    if TIER_STRICTNESS.get(new_tier, -1) <= TIER_STRICTNESS.get(old_tier, -1):
        return False
    for r in recs:
        if (r.kind == "retier" and r.block == block
                and r.note and new_tier in r.note and old_tier in r.note):
            return True
    return False


def retier(block: str, old_tier: str, new_tier: str, reason: str, *, ts: str,
           path: str = LEDGER_PATH) -> None:
    """Declare that `block` was TIGHTENED after claims against it existed.

    Appends, never rewrites. Refuses a loosening outright: a block's rules may
    only ever get stricter after it has been drawn from.
    """
    if TIER_STRICTNESS.get(new_tier, -1) <= TIER_STRICTNESS.get(old_tier, -1):
        raise ValueError(
            f"REFUSED: {old_tier!r} -> {new_tier!r} is not a tightening. A block "
            f"that has already been claimed may only be made STRICTER.")
    lo, _hi = block_span(block)
    _append(Claim(block=block, start=lo, n=0, consumer="seed_ledger.retier",
                  tier=new_tier, ts=ts, kind="retier",
                  note=f"tier {old_tier} -> {new_tier}: {reason}"),
            path=path)


def check_ledger(path: str = LEDGER_PATH, *, verbose: bool = True) -> List[str]:
    """Re-validate the whole ledger. Returns a list of ERRORS (empty == good)."""
    errors: List[str] = []
    absent: List[str] = []
    #: Non-fatal observations that must still be VISIBLE. A tier tightened after
    #: a claim is legitimate, but it is exactly the kind of thing that must never
    #: happen silently.
    notes: List[str] = []
    recs = ledger(path)
    for c in recs:
        try:
            lo, hi = block_span(c.block)
        except KeyError as exc:
            errors.append(f"{c.consumer}: {exc}")
            continue
        # A `retier` record is a DECLARATION about a block, not a range, so it
        # carries `n = 0` deliberately. Every other kind must span at least one
        # seed.
        if c.n <= 0 and c.kind != "retier":
            errors.append(f"{c.consumer}: non-positive n={c.n}")
        if c.start < lo or c.end > hi:
            errors.append(
                f"{c.consumer}: {c.block}[{c.start:,}..{c.end:,}] outside [{lo:,}..{hi:,}]"
            )
        try:
            # A recorded tier that DISAGREES with the registry is drift -- unless
            # the registry has been TIGHTENED since the claim. Tightening is the
            # deliberate hardening of a block after the fact and is safe: the
            # claim happened under weaker rules and every rule now in force is
            # stricter. Loosening is the dangerous direction (it would let a
            # DECISION block be claimed under DESIGN's overlap rules) and stays a
            # hard error. Added 2026-09-10 when P08 was promoted DESIGN ->
            # SINGLE_USE after the T2 Skeptic found its single-use rule was prose.
            _now = tier_of(c.block)
            if c.tier != _now:
                # A DECLARED tightening is legitimate; an undeclared mismatch is
                # drift, and the two are indistinguishable from the tier alone.
                # A record claiming DESIGN on a SINGLE_USE block is exactly how
                # one would dodge the single-use rule, so "tightening is safe"
                # cannot be inferred -- it must be DECLARED, the same way a
                # deliberate overlap is declared by a `pair` record. A `retier`
                # record names the block, the old tier and the new one, and is
                # appended to the same append-only ledger.
                if _retier_covers(recs, c.block, c.tier, _now):
                    notes.append(
                        f"{c.consumer}: {c.block} was TIGHTENED {c.tier!r} -> "
                        f"{_now!r} by a declared `retier` record. The claim stands "
                        f"and every rule now in force is stricter."
                    )
                else:
                    errors.append(
                        f"{c.consumer}: recorded tier {c.tier!r} != registry tier "
                        f"{_now!r} for {c.block}"
                    )
        except KeyError as exc:
            errors.append(str(exc))
        if c.artifact and c.artifact_sha256:
            now = _sha256(c.artifact)
            if now and now != c.artifact_sha256:
                errors.append(
                    f"{c.consumer}: artifact {c.artifact} changed since it was claimed "
                    f"({c.artifact_sha256[:12]} -> {now[:12]})"
                )
        elif c.artifact and c.kind == "claim":
            # THE FAIL-OPEN, CLOSED. A record that names an artifact and carries
            # no hash used to pass silently, which would let a truth set be
            # swapped after the arms were scored against it.
            #
            # A claim's hash is necessarily blank at claim time (the artifact
            # does not exist yet), so the hash may come from a `finalize`
            # record instead -- but it must come from somewhere.
            closed = [q for q in recs if q.kind == "finalize"
                      and q.consumer == c.consumer and q.artifact == c.artifact
                      and q.artifact_sha256]
            if not closed:
                errors.append(
                    f"{c.consumer}: names artifact {c.artifact} but no hash was "
                    f"ever recorded for it. Call seed_ledger.finalize({c.consumer!r}, "
                    f"{c.artifact!r}, ts=...) after writing it."
                )
            else:
                now = _sha256(c.artifact)
                if now and now != closed[-1].artifact_sha256:
                    errors.append(
                        f"{c.consumer}: artifact {c.artifact} changed since it was "
                        f"finalized ({closed[-1].artifact_sha256[:12]} -> {now[:12]})"
                    )
                elif not now:
                    # An ABSENT artifact is reported, never failed. `*.npz` is
                    # gitignored, so a fresh clone legitimately has none of them,
                    # and a gate that goes red for that is reporting a state
                    # rather than testing a property -- the failure class this
                    # repository has lost five checks to.
                    absent.append(f"{c.consumer}: {c.artifact}")

    seen_single: Dict[str, Claim] = {}
    for c in recs:
        if c.kind != "claim":
            continue
        if c.tier == "SINGLE_USE":
            if c.block in seen_single:
                errors.append(
                    f"{c.block} is SINGLE_USE but claimed twice: "
                    f"{seen_single[c.block].consumer!r} and {c.consumer!r}"
                )
            else:
                seen_single[c.block] = c

    permitted, hard = [], []
    for a, b in collisions(path):
        (hard if a.tier in ("DECISION", "SINGLE_USE", "VAULT") else permitted).append((a, b))
    for a, b in hard:
        errors.append(
            f"{a.tier} block {a.block}: {a.consumer!r} [{a.start:,}..{a.end:,}] and "
            f"{b.consumer!r} [{b.start:,}..{b.end:,}] draw the same seeds"
        )

    if verbose:
        # The breakdown must SUM to the total, or a reader silently loses a
        # category. It stopped summing the moment "pair" became a third kind --
        # SKEPTIC WP14-FOLLOWUP caught it in the same pass that added the kind.
        print(f"  ledger records                     {len(recs)}"
              f"  ({sum(1 for c in recs if c.kind == 'claim')} claims, "
              f"{sum(1 for c in recs if c.kind == 'finalize')} finalizations, "
              f"{sum(1 for c in recs if c.kind == 'pair')} pairings)")
        _kinds = {c.kind for c in recs}
        if _kinds - {"claim", "finalize", "pair", "retier"}:
            errors.append(f"unknown ledger record kind(s): "
                          f"{sorted(_kinds - {'claim', 'finalize', 'pair', 'retier'})}")
        print(f"  claims naming an artifact          "
              f"{sum(1 for c in recs if c.kind == 'claim' and c.artifact)}"
              f"  ({len(absent)} absent on this host, hash recorded)")
        for a in absent:
            print(f"    - not on disk here (gitignored?): {a}")
        if notes:
            print(f"  tier tightenings                   {len(set(notes))}"
                  f"  (permitted; recorded so they are never silent)")
            for n in sorted(set(notes)):
                print(f"    - {n}")
        print(f"  hard collisions (DECISION/VAULT)   {len(hard)}")
        print(f"  permitted overlaps (DESIGN)        {len(permitted)}")
        for a, b in permitted:
            why = pairing_reason(a, b, path)
            tag = ("<- " + why if why else
                   "<- CORRELATED EVIDENCE, never quote as two independent results")
            print(
                f"    - {a.block}[{a.start:,}..{a.end:,}]  {a.consumer}  vs  {b.consumer}"
                f"   {tag}"
            )
    return errors


_SEED_LITERAL = re.compile(r"\b(\d[\d_]{7,})\b")
_BLOCK_NAME = re.compile(r"""["'](D\d\d|V)["']""")
# A consumer can also reach seeds WITHOUT writing a literal, by calling
# `seed_blocks.seeds(block)`. Those are invisible to the literal scan — which is
# why the second pass below looks for a block NAME in any file that imports the
# registry's `seeds` helper.
_SEEDS_HELPER = re.compile(r"from seed_blocks import[^\n]*\bseeds\b|seed_blocks\.seeds")
# A file may NAME a block precisely in order to refuse it — `moonshot_g0.py`
# names D08/D09/D10 in the guard that stops G0 touching the moonshot gate blocks.
# Naming a block to protect it is the opposite of consuming it, so the file
# declares that explicitly and the scan honours the declaration:
#     # seed-ledger: excludes D08, D09, D10
_EXCLUDES = re.compile(r"#\s*seed-ledger:\s*excludes\s+([^\n]*)")


def scan_sources(roots: Sequence[str] = ("local_sim", "scripts"), path: str = LEDGER_PATH
                 ) -> List[str]:
    """Find seed consumption the ledger does not record. Two passes.

    1. Integer literals inside a registered block. This is the layer that would
       have caught D16: `SEED0 = 30_806_016_200` typed into two probes while the
       registry still called the block reusable.
    2. Files that import `seed_blocks.seeds` and name a block. Those draw real
       seeds without ever writing one down, so pass 1 cannot see them.
    """
    blocks = _blocks()
    recs = ledger(path)
    known = set()
    for c in recs:
        known.update(range(c.start, c.end + 1))
    findings: List[str] = []
    skip = {os.path.abspath(os.path.join(_ROOT, "local_sim", "seed_blocks.py")),
            os.path.abspath(os.path.join(_ROOT, "local_sim", "seed_ledger.py"))}
    for root in roots:
        base = os.path.join(_ROOT, root)
        for dirpath, _dirs, files in os.walk(base):
            if "__pycache__" in dirpath:
                continue
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                full = os.path.join(dirpath, fn)
                if os.path.abspath(full) in skip:
                    continue
                rel = os.path.relpath(full, _ROOT).replace("\\", "/")
                # A consumer string may carry the invocation that spent the block
                # ("path.py --mode decide4"), so match on the path prefix.
                recorded_here = {
                    c.block for c in recs if c.consumer == rel or c.consumer.startswith(rel + " ")
                }
                with open(full, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
                excluded = set()
                for m in _EXCLUDES.finditer(text):
                    excluded.update(re.findall(r"D\d\d|V", m.group(1)))
                for lineno, line in enumerate(text.splitlines(), 1):
                    for m in _SEED_LITERAL.finditer(line):
                        val = int(m.group(1).replace("_", ""))
                        for bid, (start, n, _why) in blocks.items():
                            if (start <= val <= start + n - 1
                                    and val not in known and bid not in excluded):
                                findings.append(
                                    f"{rel}:{lineno} uses {val:,} which is inside "
                                    f"block {bid} but has no ledger record"
                                )
                if _SEEDS_HELPER.search(text):
                    for bid in sorted({m.group(1) for m in _BLOCK_NAME.finditer(text)}):
                        if bid in blocks and bid not in recorded_here and bid not in excluded:
                            findings.append(
                                f"{rel} calls seed_blocks.seeds() and names block {bid}, "
                                f"but has no ledger record for it"
                            )
    return findings


def describe(path: str = LEDGER_PATH) -> None:
    print(f"{'block':>6} {'tier':>11} {'allocated':>7} {'consumed':>9} {'free':>7}  ranges")
    print("-" * 108)
    for bid, (start, n, _why) in _blocks().items():
        try:
            t = tier_of(bid)
        except KeyError:
            t = "?"
        used = consumed_seeds(bid, path)
        fr = free_ranges(bid, path)
        span = "" if used == 0 else ", ".join(
            f"{s:,}..{e:,}({c})" for s, e, c in consumed_ranges(bid, path)
        )
        print(f"{bid:>6} {t:>11} {n:>7,} {used:>9,} {n - used:>7,}  {span}")
        if used and fr:
            print(f"{'':>6} {'':>11} {'':>7} {'':>9} {'':>7}  free: "
                  + ", ".join(f"{s:,}..{e:,}" for s, e in fr))


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--check" in argv or "--scan" in argv:
        rc = 0
        if "--check" in argv:
            print("seed ledger — consistency")
            errs = check_ledger()
            for e in errs:
                print(f"  ERROR {e}")
            print(f"  {'CONSISTENT' if not errs else f'{len(errs)} ERROR(S)'}")
            rc |= 1 if errs else 0
        if "--scan" in argv:
            print("\nseed ledger — static source scan")
            f = scan_sources()
            for x in f:
                print(f"  UNRECORDED {x}")
            print(f"  {'no unrecorded seed literals' if not f else f'{len(f)} UNRECORDED'}")
            rc |= 1 if f else 0
        return rc
    describe()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
