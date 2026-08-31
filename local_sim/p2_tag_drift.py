"""Phase-2 dataset tag drift — does a moved tag mean the SOURCE BYTES moved?

`v2-phase2` moved from `ce926b7d…` to `6a416fa5…` on or before 2026-08-31.
`local_sim/phase2_checklist.py` compared one scalar field and concluded *"the
dataset was re-baked … re-check every shape-dependent result"*. That conflates
**the tag moved** with **the bytes moved**, which is the failure class this
repository has lost four separate checks to: reporting a state instead of testing
a property.

WHAT ACTUALLY DETERMINES OUR EVIDENCE
-------------------------------------
WhestBench materialises MLP rows from the `data/*.parquet` objects. Every byte a
mint, a CONFIRM score, or any shape-dependent result depends on is a function of
**those objects** and of the installed package — never of the repository's file
count, its prepared-split artifacts, or the `prepared_splits` key in top-level
`metadata.json`. So a commit move whose source objects are byte-identical cannot
change any of them, and a commit move that touches even one of them invalidates
all of them.

THE RECORD, AND WHY IT IS A FILE AND NOT A SENTENCE
---------------------------------------------------
`experiments/2026-08-31_phase2_tag_drift/source_manifest.tsv` is a read-only
Hugging Face tree comparison captured at that date: 70 `data/*.parquet` rows with
their LFS sha256 at BOTH commits, the README git blob at both commits, and the
file counts. This module never trusts its `canonical_data_manifest_sha256`
header — it **recomputes** the digest from the rows and refuses if they disagree.
A recorded digest that is not re-derived is a declaration, and a declaration is
not evidence (`docs/GOTCHAS.md`).

VERDICTS
--------
``PINNED``             the live commit IS the commit the manifest was built
                       against. Nothing moved.
``SOURCE_EQUIVALENT``  the live commit is recorded here, and all 70 source
                       objects are byte-identical to the pinned commit's.
                       Shape-dependent evidence stays valid; the executable pin
                       is stale and should be refreshed, which is bookkeeping.
``SOURCE_CHANGED``     the live commit is recorded here and at least one source
                       object differs. Every shape-dependent result is suspect.
``UNKNOWN``            the live commit is not in the record at all. This is NOT
                       a pass. It is the same alarm the old code raised, and it
                       is what makes ``SOURCE_EQUIVALENT`` falsifiable: an
                       arbitrary commit cannot obtain it.

Run:  uv run python local_sim/p2_tag_drift.py
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

#: The durable read-only comparison. Committed; this module never writes it.
RECORD_PATH = os.path.join(
    _ROOT, "experiments", "2026-08-31_phase2_tag_drift", "source_manifest.tsv"
)

#: The only prefix whose bytes any of our evidence depends on.
SOURCE_PREFIX = "data/"

PINNED = "PINNED"
SOURCE_EQUIVALENT = "SOURCE_EQUIVALENT"
SOURCE_CHANGED = "SOURCE_CHANGED"
UNKNOWN = "UNKNOWN"

#: Verdicts under which shape-dependent evidence (mints, CONFIRM bytes, FLOP
#: anchors) taken at the pinned commit remains valid.
_EVIDENCE_VALID = (PINNED, SOURCE_EQUIVALENT)


class DriftRecordError(ValueError):
    """The durable record is malformed, truncated, or internally inconsistent."""


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

def _parse_header(comment_lines: List[str]) -> Dict[str, str]:
    """`# key: value` comment lines, as a dict. Unkeyed comments are ignored."""
    out: Dict[str, str] = {}
    for line in comment_lines:
        body = line.lstrip("#").strip()
        if ":" not in body:
            continue
        key, _, value = body.partition(":")
        key = key.strip()
        # Only accept bare identifier-ish keys. Prose sentences that happen to
        # contain a colon ("Canonical digest input is …: …") must not become
        # header fields and silently shadow a real one.
        if key and all(c.isalnum() or c == "_" for c in key):
            out[key] = value.strip()
    return out


class DriftRecord:
    """The parsed source-manifest comparison. Immutable after construction."""

    def __init__(self, text: str, path: str = RECORD_PATH):
        self.path = path
        lines = text.splitlines()
        comments = [l for l in lines if l.startswith("#")]
        body = [l for l in lines if l and not l.startswith("#")]
        if not body:
            raise DriftRecordError(f"{path}: no data rows")

        self.header = _parse_header(comments)
        self.column_header = body[0]
        expected_cols = ["path", "size", "old_lfs_sha256", "new_lfs_sha256"]
        if self.column_header.split("\t") != expected_cols:
            raise DriftRecordError(
                f"{path}: column header is {self.column_header!r}, "
                f"expected {chr(9).join(expected_cols)!r}"
            )
        self.rows_raw: Tuple[str, ...] = tuple(body[1:])
        if not self.rows_raw:
            raise DriftRecordError(f"{path}: header present but zero rows")

        parsed = []
        for i, raw in enumerate(self.rows_raw):
            parts = raw.split("\t")
            if len(parts) != 4:
                raise DriftRecordError(
                    f"{path}: row {i} has {len(parts)} columns, expected 4"
                )
            p, size, old, new = parts
            if not p.startswith(SOURCE_PREFIX):
                raise DriftRecordError(
                    f"{path}: row {i} path {p!r} is not a {SOURCE_PREFIX!r} object; "
                    "this record is defined over SOURCE objects only"
                )
            try:
                size_i = int(size)
            except ValueError as exc:  # pragma: no cover - malformed record
                raise DriftRecordError(f"{path}: row {i} size {size!r}") from exc
            for h in (old, new):
                if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
                    raise DriftRecordError(f"{path}: row {i} bad sha256 {h!r}")
            parsed.append((p, size_i, old, new))
        self.rows: Tuple[Tuple[str, int, str, str], ...] = tuple(parsed)

        paths = [r[0] for r in self.rows]
        if len(set(paths)) != len(paths):
            raise DriftRecordError(f"{path}: duplicate source paths in the record")

    # -- identity ---------------------------------------------------------- #
    @property
    def old_commit(self) -> Optional[str]:
        return self.header.get("old_commit")

    @property
    def new_commit(self) -> Optional[str]:
        return self.header.get("new_commit")

    @property
    def recorded_digest(self) -> Optional[str]:
        return self.header.get("canonical_data_manifest_sha256")

    @property
    def old_file_count(self) -> Optional[int]:
        return _as_int(self.header.get("old_file_count"))

    @property
    def new_file_count(self) -> Optional[int]:
        return _as_int(self.header.get("new_file_count"))

    @property
    def added_count(self) -> Optional[int]:
        return _as_int(self.header.get("added_count"))

    @property
    def removed_count(self) -> Optional[int]:
        return _as_int(self.header.get("removed_count"))

    # -- the two measurements ---------------------------------------------- #
    def canonical_digest(self) -> str:
        """Recomputed from the rows: the 70 data rows in path order, LF-joined,
        no trailing LF. Never read from the header."""
        return hashlib.sha256("\n".join(self.rows_raw).encode("utf-8")).hexdigest()

    def digest_ok(self) -> bool:
        rec = self.recorded_digest
        return rec is not None and rec == self.canonical_digest()

    def differing_rows(self) -> Tuple[Tuple[str, int, str, str], ...]:
        """Source objects whose bytes moved between the two commits."""
        return tuple(r for r in self.rows if r[2] != r[3])

    # -- the verdict -------------------------------------------------------- #
    def classify(self, live_commit: str, pinned_commit: str) -> "DriftVerdict":
        """Where does `live_commit` stand relative to `pinned_commit`?

        The record only ever licenses ONE transition: the one it measured. A
        live commit that is neither the pinned one nor this record's
        `new_commit` is `UNKNOWN`, whatever the rows say.
        """
        live = (live_commit or "").strip().lower()
        pinned = (pinned_commit or "").strip().lower()
        if not self.digest_ok():
            return DriftVerdict(
                UNKNOWN, self, live, pinned,
                reason="the durable record's canonical digest does not "
                       "reproduce from its own rows; the record is untrusted",
            )
        if live and live == pinned:
            return DriftVerdict(PINNED, self, live, pinned,
                                reason="the live tag is the pinned commit")
        old, new = (self.old_commit or "").lower(), (self.new_commit or "").lower()
        if not (live and old and new and live == new and pinned == old):
            return DriftVerdict(
                UNKNOWN, self, live, pinned,
                reason="this record measures %s -> %s only; it says nothing "
                       "about %s -> %s" % (old[:12] or "?", new[:12] or "?",
                                           pinned[:12] or "?", live[:12] or "?"),
            )
        diff = self.differing_rows()
        if diff:
            return DriftVerdict(
                SOURCE_CHANGED, self, live, pinned,
                reason="%d of %d source objects changed bytes" % (len(diff), len(self.rows)),
            )
        return DriftVerdict(
            SOURCE_EQUIVALENT, self, live, pinned,
            reason="all %d source objects are byte-identical; %s objects were "
                   "added and %s removed" % (len(self.rows),
                                             self.added_count, self.removed_count),
        )


class DriftVerdict:
    """A classification plus everything needed to defend or attack it."""

    def __init__(self, status: str, record: DriftRecord, live: str, pinned: str,
                 reason: str):
        self.status = status
        self.record = record
        self.live_commit = live
        self.pinned_commit = pinned
        self.reason = reason

    @property
    def evidence_valid(self) -> bool:
        """May shape-dependent results taken at the pinned commit still be
        quoted? `UNKNOWN` is deliberately False — an unmeasured move is not a
        safe move."""
        return self.status in _EVIDENCE_VALID

    @property
    def pin_is_stale(self) -> bool:
        """The executable pin should be refreshed (bookkeeping, not evidence)."""
        return self.status != PINNED

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "evidence_valid": self.evidence_valid,
            "pin_is_stale": self.pin_is_stale,
            "reason": self.reason,
            "pinned_commit": self.pinned_commit,
            "live_commit": self.live_commit,
            "record_path": os.path.relpath(self.record.path, _ROOT).replace("\\", "/"),
            "source_objects": len(self.record.rows),
            "source_objects_changed": len(self.record.differing_rows()),
            "canonical_data_manifest_sha256": self.record.canonical_digest(),
            "old_file_count": self.record.old_file_count,
            "new_file_count": self.record.new_file_count,
            "added_count": self.record.added_count,
            "removed_count": self.record.removed_count,
        }

    def report_lines(self) -> List[str]:
        d = self.as_dict()
        out = [
            "  tag-drift verdict: %s  (source objects %d, changed %d)"
            % (d["status"], d["source_objects"], d["source_objects_changed"]),
            "    %s" % self.reason,
            "    evidence at the pinned commit remains valid: %s"
            % ("YES" if d["evidence_valid"] else "NO"),
            "    canonical source digest %s" % d["canonical_data_manifest_sha256"][:16],
            "    record: %s" % d["record_path"],
        ]
        if self.status == SOURCE_EQUIVALENT:
            out.append(
                "    NOTE: only the executable pin is stale. Prepared/metadata "
                "objects were added (%s -> %s files); no source byte moved."
                % (d["old_file_count"], d["new_file_count"])
            )
        elif self.status == SOURCE_CHANGED:
            out.append("    *** SOURCE BYTES MOVED. Re-derive every "
                       "shape-dependent Phase-2 result. ***")
        elif self.status == UNKNOWN:
            out.append("    *** UNMEASURED TAG MOVE. Treat as invalidating until "
                       "a fresh read-only tree comparison is captured. ***")
        return out


def _as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def load_record(path: str = RECORD_PATH) -> DriftRecord:
    with open(path, encoding="utf-8") as fh:
        return DriftRecord(fh.read(), path)


def classify_live_commit(live_commit: str, pinned_commit: str,
                         path: str = RECORD_PATH) -> DriftVerdict:
    return load_record(path).classify(live_commit, pinned_commit)


def lineage_block(path: str = RECORD_PATH) -> dict:
    """The `dataset.commit_lineage` block for `phase2_manifest.json`.

    It preserves BOTH commit IDs and the canonical digest, so a later rebuild of
    the manifest cannot lose the provenance of the commit our stored CONFIRM
    bytes were produced under.
    """
    rec = load_record(path)
    v = rec.classify(rec.new_commit or "", rec.old_commit or "")
    return {
        "pinned_commit": rec.old_commit,
        "source_equivalent_commit": rec.new_commit,
        "status": v.status,
        "source_objects": len(rec.rows),
        "source_objects_changed": len(rec.differing_rows()),
        "canonical_data_manifest_sha256": rec.canonical_digest(),
        "canonical_digest_reproduces_header": rec.digest_ok(),
        "old_file_count": rec.old_file_count,
        "new_file_count": rec.new_file_count,
        "added_count": rec.added_count,
        "removed_count": rec.removed_count,
        "evidence_path": os.path.relpath(path, _ROOT).replace("\\", "/"),
        "meaning": (
            "Shape-dependent Phase-2 evidence depends only on the data/*.parquet "
            "source objects. This block records a measured, content-addressed "
            "comparison of all of them across the tag move. A commit that is not "
            "named here is UNKNOWN and must be treated as invalidating."
        ),
    }


def main() -> int:
    rec = load_record()
    print("Phase-2 dataset tag drift")
    print("  record          %s" % os.path.relpath(rec.path, _ROOT).replace("\\", "/"))
    print("  old commit      %s" % rec.old_commit)
    print("  new commit      %s" % rec.new_commit)
    print("  source objects  %d   (changed %d)"
          % (len(rec.rows), len(rec.differing_rows())))
    print("  files           %s -> %s   (+%s / -%s)"
          % (rec.old_file_count, rec.new_file_count,
             rec.added_count, rec.removed_count))
    print("  digest recomputed %s" % rec.canonical_digest())
    print("  digest recorded   %s" % rec.recorded_digest)
    print("  digest reproduces %s" % rec.digest_ok())
    print()
    v = rec.classify(rec.new_commit or "", rec.old_commit or "")
    for line in v.report_lines():
        print(line)
    return 0 if rec.digest_ok() else 1


if __name__ == "__main__":
    raise SystemExit(main())
