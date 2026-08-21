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
    #: "claim" (the default, and every historical record) or "finalize".
    #:
    #: A claim necessarily records its artifact PATH before the artifact exists,
    #: so its `artifact_sha256` is necessarily blank at claim time -- and nothing
    #: used to fill it in. That is a fail-OPEN hole: a truth set could be swapped
    #: after the arms were scored against it and the ledger would still validate.
    #: A "finalize" record closes one claim's artifact by appending its hash
    #: after the file is written. Finalize records are NOT consumptions: they are
    #: excluded from overlap and SINGLE_USE accounting, so closing a claim can
    #: never look like a second claim on the same block.
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


def _append(claim: Claim, path: str = LEDGER_PATH) -> None:
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


def check_ledger(path: str = LEDGER_PATH, *, verbose: bool = True) -> List[str]:
    """Re-validate the whole ledger. Returns a list of ERRORS (empty == good)."""
    errors: List[str] = []
    absent: List[str] = []
    recs = ledger(path)
    for c in recs:
        try:
            lo, hi = block_span(c.block)
        except KeyError as exc:
            errors.append(f"{c.consumer}: {exc}")
            continue
        if c.n <= 0:
            errors.append(f"{c.consumer}: non-positive n={c.n}")
        if c.start < lo or c.end > hi:
            errors.append(
                f"{c.consumer}: {c.block}[{c.start:,}..{c.end:,}] outside [{lo:,}..{hi:,}]"
            )
        try:
            if c.tier != tier_of(c.block):
                errors.append(
                    f"{c.consumer}: recorded tier {c.tier!r} != registry tier "
                    f"{tier_of(c.block)!r} for {c.block}"
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
        print(f"  ledger records                     {len(recs)}"
              f"  ({sum(1 for c in recs if c.kind == 'claim')} claims, "
              f"{sum(1 for c in recs if c.kind == 'finalize')} finalizations)")
        print(f"  claims naming an artifact          "
              f"{sum(1 for c in recs if c.kind == 'claim' and c.artifact)}"
              f"  ({len(absent)} absent on this host, hash recorded)")
        for a in absent:
            print(f"    - not on disk here (gitignored?): {a}")
        print(f"  hard collisions (DECISION/VAULT)   {len(hard)}")
        print(f"  permitted overlaps (DESIGN)        {len(permitted)}")
        for a, b in permitted:
            intended = (b.consumer in a.paired_with) or (a.consumer in b.paired_with)
            tag = ("<- SAME NETWORKS BY DESIGN (one experiment, two roles)"
                   if intended else
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
