"""A single-use block is single-use when its LABELS are counted, not its claim.

Stage 0 repair R3. `docs/research/2026-09-10_validation_audit.md` finding 3;
contract `experiments/2026-09-10_stage0_instrument/prediction.md`, committed at
`def1b00` before the repair.

WHY THIS FILE EXISTS
--------------------
`seed_ledger.claim_range` refuses a second CLAIM on a `SINGLE_USE` block. It
has never been able to refuse a second `np.load` of the truth that claim
produced, and on 2026-09-10 that gap was spent: **P08 was read three times**
while the instrument reading it was being repaired, and the winning instrument
was committed in the same commit as the result it produced. `seed_blocks.py`
recorded the rule in PROSE --

    "claiming P08 for anything but the single held-out read defeats the only
     reason it exists"

-- which is precisely the failure shape `AGENTS.md` § *Enforcement* exists to
prevent: a load-bearing rule that is not executable. `seed_blocks.READ_COUNTED_OWED`
has named this as owed since that day. This is it.

THE MODEL IS `vault.py`, WHICH ALREADY GOT THIS RIGHT
-----------------------------------------------------
The vault counts three permitted touches, logs each one with its consumer and
its reason, and refuses the fourth. A held-out seed block needs exactly that,
and nothing more elaborate.

RESERVE BEFORE YOU LOOK, AND AN ABORTED LOOK STILL COUNTS
---------------------------------------------------------
The reservation is written **before** the labels are handed over, and it is
NOT rolled back if the consumer then crashes. That is deliberate and it is the
whole point: a process that died after `np.load` returned has still seen the
numbers, and a registry that forgave crashes would turn "my script raised" into
an unlimited supply of looks. An aborted read is recorded as `aborted` and
still consumes the cap. If a genuine infrastructure failure burns a read, that
is a fact about the block, to be recorded and lived with -- exactly as a vault
touch spent on a crashed run would be.

WHAT THIS IS NOT
----------------
This is an **audit discipline on a single-operator filesystem**, not a security
boundary. Anyone with the repository can `np.load` the file directly, and
`audit_direct_loads()` below is a static scan that finds the obvious ways to do
it by accident, not a sandbox. It makes the honest path the easy one and makes
the dishonest one leave a hole in the record.

Run:
    uv run python local_sim/truth_reads.py                 # the register
    uv run python local_sim/truth_reads.py --audit         # scan for direct loads
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.append(_HERE)

from seed_ledger import LEDGER_PATH, ledger_lock, tier_of  # noqa: E402

#: One registry, beside the claim ledger, in the same JSON-Lines format.
READS_PATH = os.path.join(_ROOT, "experiments", "truth_reads.jsonl")

#: How many reads a block of each tier may ever have. A DESIGN block is
#: reusable by construction, so it is counted but not capped; the point of
#: counting it is that a report can say how much a design corpus has been
#: mined, which is a different question from whether it was allowed.
CAPS: Dict[str, Optional[int]] = {
    "SINGLE_USE": 1,
    "VAULT": 3,          # matches vault.py's own counter; vault.py remains the
                         # enforcer for the vault itself -- this is the mirror.
    "DECISION": None,
    "DESIGN": None,
    "LEGACY": None,
}

#: Statuses a read line can carry. `reserved` is written before the labels are
#: exposed; exactly one of `completed`/`aborted` follows it.
_OPEN, _DONE, _ABORT = "reserved", "completed", "aborted"


class ReadRefused(PermissionError):
    """The cap is spent. Never a warning, never overridable by a flag."""


_BLOCK_IN_NAME = re.compile(
    r"(?:^|[^A-Za-z0-9])(?:truth|feat|features)_([A-Z]\d{2})(?:[^A-Za-z0-9]|$)",
    re.I)
_VAULT_IN_NAME = re.compile(r"(?:^|[^A-Za-z0-9])vault_v\d+\.npz(?:$|[^A-Za-z0-9])", re.I)


def resolve_block(artifact: str, supplied: str = "") -> str:
    """Bind a caller's block id to the identity carried by an artifact path.

    A supplied id may fill in an intentionally generic filename. It may never
    contradict a filename that already identifies a block. Vault artifacts are
    always block ``V``. This check is deliberately performed before a read is
    reserved, so a mismatched request neither exposes labels nor spends a look.
    """
    name = os.path.basename(os.fspath(artifact))
    match = _BLOCK_IN_NAME.search(name)
    inferred = match.group(1).upper() if match else ("V" if _VAULT_IN_NAME.search(name) else "")
    requested = supplied.strip().upper()
    if inferred and requested and inferred != requested:
        raise ReadRefused(
            f"artifact {name!r} identifies block {inferred}, but the caller "
            f"supplied {requested}; refusing before truth access")
    block = inferred or requested
    if not block:
        raise ReadRefused(
            f"cannot identify the seed block for {name!r}; use a truth_<BLOCK>.npz "
            "filename or supply --block explicitly")
    try:
        tier_of(block)
    except KeyError as exc:
        raise ReadRefused(f"unknown seed block {block!r}; refusing before truth access") from exc
    return block


@dataclass(frozen=True)
class Read:
    """One look at a protected artifact's labels."""

    block: str
    artifact: str
    artifact_sha256: str
    consumer: str
    instrument_sha256: str
    purpose: str
    ts: str
    status: str = _OPEN
    reservation: str = ""
    result_artifact: str = ""
    note: str = ""
    backfilled: bool = False
    timestamp_source: str = "legacy_unclassified"
    receipt_created_utc: str = ""
    receipt_completed_utc: str = ""
    elapsed_seconds: Optional[float] = None

    @property
    def counts_against_cap(self) -> bool:
        """A reservation counts the moment it is written.

        `aborted` is included on purpose -- see the module docstring.
        """
        return self.status in (_OPEN, _DONE, _ABORT)


def _sha256_file(path: str) -> str:
    if not path:
        return ""
    p = path if os.path.isabs(path) else os.path.join(_ROOT, path)
    if not os.path.isfile(p):
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: str) -> str:
    try:
        return os.path.relpath(os.path.abspath(path), _ROOT).replace("\\", "/")
    except ValueError:
        return path.replace("\\", "/")


def reads(path: str = READS_PATH) -> List[Read]:
    out: List[Read] = []
    if not os.path.exists(path):
        return out
    fields = {f.name for f in dataclasses.fields(Read)}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            d = json.loads(line)
            out.append(Read(**{k: v for k, v in d.items() if k in fields}))
    return out


def _append(rec: Read, path: str = READS_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(rec), sort_keys=True) + "\n")


def spent(block: str, path: str = READS_PATH) -> int:
    """How many reads this block has already had, aborted ones included.

    Counts RESERVATIONS, not lines. Each read writes two lines -- the
    `reserved` that precedes the labels and the `completed`/`aborted` that
    follows them -- and the first draft of this counted both, so one read read
    as two. The test suite caught it, which is the only reason this comment
    exists rather than a silently doubled cap.
    """
    return len({r.reservation for r in reads(path) if r.block == block
                and r.status in (_OPEN, _DONE, _ABORT)})


def cap_for(block: str) -> Optional[int]:
    return CAPS.get(tier_of(block))


def status(block: str, path: str = READS_PATH) -> Dict[str, Any]:
    used, cap = spent(block, path), cap_for(block)
    return {
        "block": block,
        "tier": tier_of(block),
        "cap": cap,
        "spent": used,
        "remaining": (None if cap is None else max(0, cap - used)),
        "exhausted": (False if cap is None else used >= cap),
        "reads": [asdict(r) for r in reads(path) if r.block == block],
    }


@contextlib.contextmanager
def reserve(
    block: str,
    artifact: str,
    *,
    consumer: str,
    purpose: str,
    ts: str = "",
    instrument: str = "",
    result_artifact: str = "",
    note: str = "",
    path: str = READS_PATH,
    ledger_path: str = LEDGER_PATH,
    prior_spent: int = 0,
):
    """Reserve one read of `artifact`, or refuse. Yields the resolved path.

    The reservation is appended under the ledger's lock BEFORE this yields, so
    two concurrent consumers cannot both be handed the last read of a
    single-use block. `instrument` is the path of the code doing the reading;
    its sha256 is recorded, so a repaired instrument cannot later be confused
    with the one that produced a stored result. ``prior_spent`` is a fixed
    adoption baseline for historical exposures absent from this registry. It
    must not be a live counter that later includes these same reservations.
    """
    if prior_spent < 0:
        raise ValueError("prior_spent must be non-negative")
    cap = cap_for(block)
    receipt_created = (_dt.datetime.now(_dt.timezone.utc)
                       .isoformat(timespec="seconds").replace("+00:00", "Z"))
    event_ts = ts or receipt_created
    timestamp_source = ("supplied_historical" if ts else
                        "automatic_runtime")
    # UNIQUE per reservation, not per (block, consumer, ts). The first draft
    # hashed only those, so the same consumer reading a reusable design block
    # twice produced the same id and the register counted one read. `spent()`
    # counts distinct reservations, so a colliding id undercounts the cap --
    # the failure direction that matters.
    rid = uuid.uuid4().hex[:16]

    with ledger_lock(ledger_path):
        registered = spent(block, path)
        used = prior_spent + registered
        if cap is not None and used >= cap:
            prior = [r for r in reads(path) if r.block == block]
            raise ReadRefused(
                f"{block} is {tier_of(block)} and its {cap} permitted "
                f"read(s) are SPENT ({used}: {prior_spent} historical "
                f"outside this registry + {registered} registered). Prior reads: "
                + "; ".join(f"{r.consumer} on {r.ts} ({r.status})"
                            for r in prior[:4])
                + ". A repaired instrument does not restore a pristine "
                  "held-out block. Allocate genuinely untouched networks."
            )
        rec = Read(
            block=block, artifact=_rel(artifact),
            artifact_sha256=_sha256_file(artifact),
            consumer=consumer, instrument_sha256=_sha256_file(instrument),
            purpose=purpose, ts=event_ts, status=_OPEN, reservation=rid,
            result_artifact=result_artifact, note=note,
            timestamp_source=timestamp_source,
            receipt_created_utc=receipt_created,
        )
        _append(rec, path)

    ok = False
    t0 = time.perf_counter()
    try:
        yield (artifact if os.path.isabs(artifact)
               else os.path.join(_ROOT, artifact))
        ok = True
    finally:
        receipt_completed = (_dt.datetime.now(_dt.timezone.utc)
                             .isoformat(timespec="seconds")
                             .replace("+00:00", "Z"))
        done = dataclasses.replace(
            rec, status=(_DONE if ok else _ABORT),
            receipt_completed_utc=receipt_completed,
            elapsed_seconds=float(time.perf_counter() - t0),
            note=(rec.note if ok else
                  (rec.note + " | consumer raised before completing; the "
                   "labels were already exposed, so this read STILL COUNTS")
                  .strip(" |")))
        with ledger_lock(ledger_path):
            _append(done, path)


def load_truth(block: str, artifact: str, **kw):
    """The ONE way protected truth enters a process. Reserves, then loads.

    Returns whatever `numpy.load` returns, with the archive already read into
    memory so the reservation cannot outlive a lazy handle.
    """
    import numpy as np

    block = resolve_block(artifact, block)

    with reserve(block, artifact, **kw) as p:
        with np.load(p) as z:
            return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- #
# the audit: who loads protected artifacts without going through here?
# --------------------------------------------------------------------------- #

#: A load of one of these outside `load_truth` is what the scan looks for. It
#: is a heuristic over source text: it finds the accidental path, not a
#: determined one.
_LOAD_CALL = re.compile(r"\b(?:np|numpy)\.load\s*\(")
#: A file that mentions any of these is in the business of reading labels.
#: Matching on the CALL SITE alone finds almost nothing, because every real
#: consumer passes a variable: `np.load(a.truth)`. Matching on the file is the
#: only version of this scan that reports anything true.
_TRUTH_CONTEXT = re.compile(
    r"truth[_A-Za-z0-9]*\.npz|vault_v\d+\.npz|truth_P\d+|--truth"
    r"|gt_noise|means_r1")
_PROTECTED_TIERS = ("SINGLE_USE", "VAULT")


def audit_direct_loads(root: str = _ROOT) -> List[Dict[str, Any]]:
    """Source sites that load a truth-shaped artifact without reserving.

    Deliberately reports on the FILE's context, not the call site's arguments:
    every real consumer here loads a path held in a variable, so a call-site
    regex reports zero and means nothing. A hit says "this module reads labels
    and does not go through the register" -- which is true of most of the lab
    and is not by itself wrong. It is wrong when the block is SINGLE_USE or
    VAULT, and that is a judgement for the reader, which is why the block is
    not guessed at here.
    """
    hits: List[Dict[str, Any]] = []
    skip_files = {"truth_reads.py", "test_truth_reads.py"}
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, "local_sim")):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", "flopscope")]
        for fn in sorted(filenames):
            if not fn.endswith(".py") or fn in skip_files:
                continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p, encoding="utf-8") as fh:
                    text = fh.read()
            except OSError:
                continue
            if not _TRUTH_CONTEXT.search(text):
                continue
            lines = text.splitlines()
            uses_reserve = "truth_reads" in text
            for i, ln in enumerate(lines, 1):
                if _LOAD_CALL.search(ln):
                    hits.append({"file": _rel(p), "line": i,
                                 "text": ln.strip()[:110],
                                 "imports_truth_reads": uses_reserve})
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audit", action="store_true",
                    help="scan local_sim/ for direct loads of truth artifacts")
    ap.add_argument("--block", default="", help="report on one block")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if a.audit:
        hits = audit_direct_loads()
        if a.json:
            print(json.dumps(hits, indent=2))
            return 0
        print(f"direct truth-artifact loads: {len(hits)}")
        for h in hits:
            flag = "" if h["imports_truth_reads"] else "   <- does NOT reserve"
            print(f"  {h['file']}:{h['line']}{flag}")
            print(f"      {h['text']}")
        print()
        print("  A hit is not automatically wrong: a module may legitimately")
        print("  load an UNPROTECTED design artifact. It is wrong when the")
        print("  block is SINGLE_USE or VAULT. This scan is an audit discipline")
        print("  on a single-operator filesystem, not a boundary.")
        return 0

    if a.block:
        st = status(a.block)
        print(json.dumps(st, indent=2) if a.json else
              f"{st['block']}  tier {st['tier']}  cap {st['cap']}  "
              f"spent {st['spent']}  exhausted {st['exhausted']}")
        return 0

    rs = reads()
    by_block: Dict[str, int] = {}
    for r in rs:
        by_block[r.block] = by_block.get(r.block, 0) + (1 if r.status != _DONE or True else 0)
    print(f"truth read register  --  {READS_PATH}")
    print(f"  {len(rs)} lines")
    for b in sorted({r.block for r in rs}):
        st = status(b)
        mark = "  SPENT" if st["exhausted"] else ""
        print(f"  {b:<6} tier {st['tier']:<11} cap {str(st['cap']):<5} "
              f"spent {st['spent']}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
