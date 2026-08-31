"""Phase-2 dataset tag drift — did a moved tag move anything the LOADER reads?

`v2-phase2` moved from `ce926b7d…` to `6a416fa5…` on or before 2026-08-31.
`local_sim/phase2_checklist.py` compared one scalar field and concluded *"the
dataset was re-baked … re-check every shape-dependent result"*. That conflates
**the tag moved** with **the bytes moved**, which is the failure class this
repository has lost four separate checks to: reporting a state instead of
testing a property.

WHAT ACTUALLY FEEDS A MEASUREMENT — CORRECTED 2026-08-31, AND THE FIRST VERSION
OF THIS PARAGRAPH WAS WRONG
------------------------------------------------------------------------------
The first version of this module said every shape-dependent result "is a
function of the `data/*.parquet` objects and of the installed package — never of
the repository's file count, its prepared-split artifacts, or the
`prepared_splits` key in top-level `metadata.json`."

**An independent Skeptic falsified that against the installed package, and it
matters, because the tag move added exactly the thing the sentence called
inert.** Read it yourself:

* `whestbench/dataset_io.py:61-68` — *"Optional sibling tree of pre-prepared
  HuggingFace Arrow artifacts. When present, `whestbench.load_dataset` **prefers
  it over the parquet path**…"*
* `whestbench/dataset.py:468, 479-491` — reads `md.get("prepared_splits")`, and
  when `not streaming and split in prepared_splits` calls
  `_try_load_prepared_split(...)` and **returns without ever touching `data/`**.
  `whest run` defaults to non-streaming (`cli.py:3127-3131`), so an official
  Phase-2 CONFIRM today takes the prepared path.

So the loader-visible input set is `data/**` **plus** `metadata.json` **plus**
`prepared/**`, and this module now compares all of it
(`experiments/2026-08-31_phase2_tag_drift/loader_manifest.tsv`, 226 rows).

WHAT THE TWO RECORDS SAY, MEASURED
----------------------------------
| kind | status | count |
|---|---|---|
| `data/**` source parquet | unchanged | 70 |
| `README.md` | unchanged | 1 |
| `.gitattributes` | unchanged | 1 |
| `metadata.json` | **changed** (gained `prepared_splits`) | 1 |
| `prepared/**` | **added** | 153 |
| anything | removed | **0** |

And the two paths were compared functionally, not merely reasoned about:
`experiments/2026-08-31_wp0a_manifest/prepared_vs_parquet.json` reads
`prepared/mini/data-00000-of-00014.arrow` against
`data/mini-00000-of-00007.parquet` and finds the same schema, the same 8 rows in
the same order, and `weights` / `all_layer_means` **bitwise identical**
(`max|diff| = 0.0`). One shard of fourteen — it does not verify the rest.

VERDICTS
--------
``PINNED``            the live commit IS the manifest's pin. Nothing moved.
``SOURCE_EQUIVALENT`` the live commit is recorded here, every source object is
                      byte-identical, and nothing was removed. Evidence taken at
                      the pinned commit stays valid. `loader_path_changed` says
                      separately whether the *route* the loader takes moved.
``SOURCE_CHANGED``    at least one source object differs, or something was
                      removed. Every shape-dependent result is suspect.
``UNKNOWN``           the move is not covered by these records, or a record
                      failed its integrity check. This is NOT a pass. It is the
                      alarm the old code raised, and it is what makes
                      ``SOURCE_EQUIVALENT`` falsifiable.

WHY THE PINNED CONSTANTS BELOW EXIST
------------------------------------
The same Skeptic showed that recomputing a digest from a file's own rows and
comparing it to that file's own header proves nothing: rewrite the header's
`new_commit`, or re-stamp the digest after a tamper, and the check passes. Four
constructed tampers obtained `SOURCE_EQUIVALENT`. So the expected digests, the
commit pair, the row counts and the object counts are **pinned here, in code**,
and the record is checked against them. A record cannot certify itself.

Run:  uv run python local_sim/p2_tag_drift.py
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

_DRIFT_DIR = os.path.join(_ROOT, "experiments", "2026-08-31_phase2_tag_drift")

#: The 70 `data/*.parquet` objects at both commits. Captured 2026-08-31.
RECORD_PATH = os.path.join(_DRIFT_DIR, "source_manifest.tsv")
#: EVERY object at both commits, classified. Captured 2026-08-31 after the
#: prepared-Arrow fast path was found. `capture_loader_manifest.py` beside it
#: regenerates it from Hugging Face, read-only.
LOADER_RECORD_PATH = os.path.join(_DRIFT_DIR, "loader_manifest.tsv")

#: Functional comparison of the two loader routes on one shard.
PREPARED_CHECK_PATH = os.path.join(
    _ROOT, "experiments", "2026-08-31_wp0a_manifest", "prepared_vs_parquet.json")

#: Only these bytes decide a mint, a CONFIRM score, or any shape-dependent
#: result. `metadata.json` is here because it selects the ROUTE.
SOURCE_PREFIX = "data/"

PINNED = "PINNED"
SOURCE_EQUIVALENT = "SOURCE_EQUIVALENT"
SOURCE_CHANGED = "SOURCE_CHANGED"
UNKNOWN = "UNKNOWN"

_EVIDENCE_VALID = (PINNED, SOURCE_EQUIVALENT)

# --------------------------------------------------------------------------- #
# PINNED EXPECTATIONS — the record is checked against these, never against
# itself. Changing one is a deliberate act with its own contract.
# --------------------------------------------------------------------------- #
EXPECTED_OLD_COMMIT = "ce926b7d1ced65faa668c01b94a43f8c4f43783d"
EXPECTED_NEW_COMMIT = "6a416fa5fb45cade1e9ad34320204853f5d4a5f6"

#: TWO HUGGING FACE APIS RESOLVE `v2-phase2` TO DIFFERENT COMMIT IDS, and this
#: module (and `phase2_checklist.py`) consult only the first:
#:
#:   list_repo_refs().tags      -> 6a416fa5...   <- what is pinned here
#:   repo_info(revision=TAG).sha -> aa99830f...  (branch `main`,
#:                                  "Advertise prepared/ in metadata.json")
#:
#: An independent Skeptic found this and checked the consequence: the two trees
#: are **object-identical on all 226 paths** -- same size, same LFS sha256, same
#: blob id -- so nothing is wrong today and no number depends on which one is
#: read. It is recorded because a future move could separate them, and a drift
#: detector that consults one of two authorities should say so rather than
#: appear to consult the only one.
ALT_TAG_RESOLUTION = {
    "list_repo_refs_tags": EXPECTED_NEW_COMMIT,
    "repo_info_sha": "aa99830fdc09fad15407b10e8e3459d3e18bba0a",
    "trees_object_identical_2026_08_31": True,
    "consulted_here": "list_repo_refs_tags",
}

EXPECTED_SOURCE_ROWS = 70
EXPECTED_SOURCE_DIGEST = (
    "3cb3e81907069725662edc011638b095b399dafe42d35b97ea5456ece0cb8c66")

EXPECTED_LOADER_ROWS = 226
EXPECTED_LOADER_DIGEST = (
    "29664dbb64a02b8d8e26a213fcf741ee19d308a338520dbcea6c5b7d91901f2d")

#: What the full-tree comparison must show, kind by kind. A record that says
#: anything else describes a different move and licenses nothing.
EXPECTED_LOADER_COUNTS = {
    ("source", "unchanged"): 70,
    ("readme", "unchanged"): 1,
    ("other", "unchanged"): 1,
    ("metadata", "changed"): 1,
    ("prepared", "added"): 153,
}

EXPECTED_OLD_OBJECTS = 73
EXPECTED_NEW_OBJECTS = 226


class DriftRecordError(ValueError):
    """A record is malformed, truncated, or inconsistent with the pinned facts."""


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

def _parse_header(comment_lines: List[str]) -> Dict[str, str]:
    """`# key: value` comment lines, as a dict. Prose comments are ignored.

    Only bare identifier-ish keys are accepted, so a sentence that happens to
    contain a colon cannot become a header field and shadow a real one.
    """
    out: Dict[str, str] = {}
    for line in comment_lines:
        body = line.lstrip("#").strip()
        if ":" not in body:
            continue
        key, _, value = body.partition(":")
        key = key.strip()
        if key and all(c.isalnum() or c == "_" for c in key):
            out[key] = value.strip()
    return out


def _as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _split_tsv(text: str, path: str, columns: List[str]):
    lines = text.splitlines()
    comments = [l for l in lines if l.startswith("#")]
    body = [l for l in lines if l and not l.startswith("#")]
    if not body:
        raise DriftRecordError(f"{path}: no data rows")
    if body[0].split("\t") != columns:
        raise DriftRecordError(
            f"{path}: column header is {body[0]!r}, expected "
            f"{chr(9).join(columns)!r}")
    rows = tuple(body[1:])
    if not rows:
        raise DriftRecordError(f"{path}: header present but zero rows")
    return _parse_header(comments), rows


def _digest(rows) -> str:
    """The rows in file order, LF-joined, no trailing LF."""
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


class DriftRecord:
    """The `data/**` comparison. Immutable after construction."""

    COLUMNS = ["path", "size", "old_lfs_sha256", "new_lfs_sha256"]

    def __init__(self, text: str, path: str = RECORD_PATH):
        self.path = path
        self.header, self.rows_raw = _split_tsv(text, path, self.COLUMNS)
        parsed = []
        for i, raw in enumerate(self.rows_raw):
            parts = raw.split("\t")
            if len(parts) != 4:
                raise DriftRecordError(
                    f"{path}: row {i} has {len(parts)} columns, expected 4")
            p, size, old, new = parts
            if not p.startswith(SOURCE_PREFIX):
                raise DriftRecordError(
                    f"{path}: row {i} path {p!r} is not a {SOURCE_PREFIX!r} "
                    "object; this record is defined over SOURCE objects only")
            try:
                size_i = int(size)
            except ValueError as exc:
                raise DriftRecordError(f"{path}: row {i} size {size!r}") from exc
            for h in (old, new):
                if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
                    raise DriftRecordError(f"{path}: row {i} bad sha256 {h!r}")
            parsed.append((p, size_i, old, new))
        self.rows: Tuple[Tuple[str, int, str, str], ...] = tuple(parsed)
        paths = [r[0] for r in self.rows]
        if len(set(paths)) != len(paths):
            raise DriftRecordError(f"{path}: duplicate source paths in the record")

    old_commit = property(lambda self: self.header.get("old_commit"))
    new_commit = property(lambda self: self.header.get("new_commit"))
    recorded_digest = property(
        lambda self: self.header.get("canonical_data_manifest_sha256"))
    old_file_count = property(lambda self: _as_int(self.header.get("old_file_count")))
    new_file_count = property(lambda self: _as_int(self.header.get("new_file_count")))
    added_count = property(lambda self: _as_int(self.header.get("added_count")))
    removed_count = property(lambda self: _as_int(self.header.get("removed_count")))

    def canonical_digest(self) -> str:
        return _digest(self.rows_raw)

    def digest_ok(self) -> bool:
        """Against the PINNED constant, not against this file's own header.

        Comparing a file's rows to that file's header lets an attacker re-stamp
        both; four constructed tampers passed that way. The header is checked
        too, but the pinned constant is the authority.
        """
        return self.canonical_digest() == EXPECTED_SOURCE_DIGEST

    def differing_rows(self):
        return tuple(r for r in self.rows if r[2] != r[3])

    def integrity(self) -> List[str]:
        """Everything about this record that disagrees with the pinned facts."""
        v = []
        if self.canonical_digest() != EXPECTED_SOURCE_DIGEST:
            v.append("row digest %s != pinned %s"
                     % (self.canonical_digest()[:16], EXPECTED_SOURCE_DIGEST[:16]))
        if self.recorded_digest != EXPECTED_SOURCE_DIGEST:
            v.append("header digest %s != pinned %s"
                     % (str(self.recorded_digest)[:16], EXPECTED_SOURCE_DIGEST[:16]))
        if len(self.rows) != EXPECTED_SOURCE_ROWS:
            v.append("row count %d != pinned %d"
                     % (len(self.rows), EXPECTED_SOURCE_ROWS))
        if (self.old_commit or "").lower() != EXPECTED_OLD_COMMIT:
            v.append("old_commit %s != pinned" % str(self.old_commit)[:12])
        if (self.new_commit or "").lower() != EXPECTED_NEW_COMMIT:
            v.append("new_commit %s != pinned" % str(self.new_commit)[:12])
        if self.old_file_count != EXPECTED_OLD_OBJECTS:
            v.append("old_file_count %s != pinned %d"
                     % (self.old_file_count, EXPECTED_OLD_OBJECTS))
        if self.new_file_count != EXPECTED_NEW_OBJECTS:
            v.append("new_file_count %s != pinned %d"
                     % (self.new_file_count, EXPECTED_NEW_OBJECTS))
        if self.removed_count != 0:
            v.append("removed_count is %s; a REMOVAL is never source-equivalent"
                     % self.removed_count)
        if self.added_count != EXPECTED_NEW_OBJECTS - EXPECTED_OLD_OBJECTS:
            v.append("added_count %s does not close the object accounting"
                     % self.added_count)
        return v

    # -- the verdict -------------------------------------------------------- #
    def classify(self, live_commit: str, pinned_commit: str,
                 loader: Optional["LoaderRecord"] = None) -> "DriftVerdict":
        live = (live_commit or "").strip().lower()
        pinned = (pinned_commit or "").strip().lower()
        viol = self.integrity()
        # AN ABSENT LOADER RECORD IS A VIOLATION, NOT SILENCE.
        #
        # This branch used to read `if loader is not None:` and skip every
        # loader clause when the file was missing -- so DELETING
        # `loader_manifest.tsv` returned SOURCE_EQUIVALENT with
        # `evidence_valid=True`, dropped the "THE LOADER ROUTE MOVED" warning,
        # and exited 0. An independent Skeptic constructed exactly that as a
        # fifth path to a false pass. It is the repository's signature failure:
        # a check that could not run reporting a pass. The loader record is now
        # REQUIRED, and its absence is named.
        if loader is None:
            viol.append(
                "the loader record is ABSENT (%s). data/ is not the whole input "
                "set: whestbench 0.16.0 prefers prepared/<split> Arrow whenever "
                "metadata.json advertises prepared_splits, so a source-only "
                "comparison cannot license anything."
                % _rel(LOADER_RECORD_PATH))
        else:
            viol += ["loader record: " + x for x in loader.integrity()]
        if viol:
            return DriftVerdict(
                UNKNOWN, self, loader, live, pinned,
                reason="the durable record fails its pinned integrity check: "
                       + "; ".join(viol[:3]))
        if live and live == pinned:
            return DriftVerdict(PINNED, self, loader, live, pinned,
                                reason="the live tag is the pinned commit")
        old, new = EXPECTED_OLD_COMMIT, EXPECTED_NEW_COMMIT
        if not (live == new and pinned == old):
            return DriftVerdict(
                UNKNOWN, self, loader, live, pinned,
                reason="these records measure %s -> %s only; they say nothing "
                       "about %s -> %s" % (old[:12], new[:12],
                                           pinned[:12] or "?", live[:12] or "?"))
        diff = self.differing_rows()
        if diff:
            return DriftVerdict(
                SOURCE_CHANGED, self, loader, live, pinned,
                reason="%d of %d source objects changed bytes"
                       % (len(diff), len(self.rows)))
        return DriftVerdict(
            SOURCE_EQUIVALENT, self, loader, live, pinned,
            reason="all %d source objects are byte-identical and nothing was "
                   "removed" % len(self.rows))


class LoaderRecord:
    """The FULL-tree comparison — everything `whestbench.load_dataset` can read."""

    COLUMNS = ["path", "kind", "status", "old_size", "new_size",
               "old_lfs_sha256", "new_lfs_sha256", "old_blob_id", "new_blob_id"]

    def __init__(self, text: str, path: str = LOADER_RECORD_PATH):
        self.path = path
        self.header, self.rows_raw = _split_tsv(text, path, self.COLUMNS)
        self.rows = []
        for i, raw in enumerate(self.rows_raw):
            parts = raw.split("\t")
            if len(parts) != len(self.COLUMNS):
                raise DriftRecordError(
                    f"{path}: row {i} has {len(parts)} columns, "
                    f"expected {len(self.COLUMNS)}")
            self.rows.append(dict(zip(self.COLUMNS, parts)))
        paths = [r["path"] for r in self.rows]
        if len(set(paths)) != len(paths):
            raise DriftRecordError(f"{path}: duplicate paths in the record")

    old_commit = property(lambda self: self.header.get("old_commit"))
    new_commit = property(lambda self: self.header.get("new_commit"))
    recorded_digest = property(
        lambda self: self.header.get("canonical_loader_manifest_sha256"))

    def canonical_digest(self) -> str:
        return _digest(self.rows_raw)

    def counts(self) -> Dict[Tuple[str, str], int]:
        out: Dict[Tuple[str, str], int] = {}
        for r in self.rows:
            k = (r["kind"], r["status"])
            out[k] = out.get(k, 0) + 1
        return out

    def integrity(self) -> List[str]:
        v = []
        if self.canonical_digest() != EXPECTED_LOADER_DIGEST:
            v.append("row digest %s != pinned %s"
                     % (self.canonical_digest()[:16], EXPECTED_LOADER_DIGEST[:16]))
        if self.recorded_digest != EXPECTED_LOADER_DIGEST:
            v.append("header digest != pinned")
        if len(self.rows) != EXPECTED_LOADER_ROWS:
            v.append("row count %d != pinned %d"
                     % (len(self.rows), EXPECTED_LOADER_ROWS))
        if (self.old_commit or "").lower() != EXPECTED_OLD_COMMIT:
            v.append("old_commit != pinned")
        if (self.new_commit or "").lower() != EXPECTED_NEW_COMMIT:
            v.append("new_commit != pinned")
        c = self.counts()
        if c != EXPECTED_LOADER_COUNTS:
            v.append("kind/status census %s != pinned %s"
                     % (sorted(c.items()), sorted(EXPECTED_LOADER_COUNTS.items())))
        removed = [r["path"] for r in self.rows if r["status"] == "removed"]
        if removed:
            v.append("%d object(s) REMOVED, e.g. %s" % (len(removed), removed[0]))
        return v

    @property
    def loader_path_changed(self) -> bool:
        """Did the ROUTE `whestbench.load_dataset` takes move, even though the
        source bytes did not?

        True here: `metadata.json` changed and a `prepared/` tree appeared, and
        `whestbench/dataset.py:479-491` prefers that tree over the parquet.
        """
        c = self.counts()
        return bool(c.get(("metadata", "changed"))) or \
            bool(c.get(("prepared", "added"))) or \
            bool(c.get(("prepared", "changed")))


class DriftVerdict:
    """A classification plus everything needed to defend or attack it."""

    def __init__(self, status, record, loader, live, pinned, reason):
        self.status = status
        self.record = record
        self.loader = loader
        self.live_commit = live
        self.pinned_commit = pinned
        self.reason = reason

    @property
    def evidence_valid(self) -> bool:
        """May shape-dependent results taken at the pinned commit still be
        quoted? `UNKNOWN` is deliberately False."""
        return self.status in _EVIDENCE_VALID

    @property
    def pin_is_stale(self) -> bool:
        return self.status != PINNED

    @property
    def loader_path_changed(self) -> Optional[bool]:
        return None if self.loader is None else self.loader.loader_path_changed

    def as_dict(self) -> dict:
        d = {
            "status": self.status,
            "evidence_valid": self.evidence_valid,
            "pin_is_stale": self.pin_is_stale,
            "loader_path_changed": self.loader_path_changed,
            "reason": self.reason,
            "pinned_commit": self.pinned_commit,
            "live_commit": self.live_commit,
            "record_path": _rel(self.record.path),
            "source_objects": len(self.record.rows),
            "source_objects_changed": len(self.record.differing_rows()),
            "canonical_data_manifest_sha256": self.record.canonical_digest(),
            "old_file_count": self.record.old_file_count,
            "new_file_count": self.record.new_file_count,
            "added_count": self.record.added_count,
            "removed_count": self.record.removed_count,
        }
        if self.loader is not None:
            d["loader_record_path"] = _rel(self.loader.path)
            d["canonical_loader_manifest_sha256"] = self.loader.canonical_digest()
            d["loader_census"] = {f"{k}:{s}": n
                                  for (k, s), n in sorted(self.loader.counts().items())}
        return d

    def report_lines(self) -> List[str]:
        d = self.as_dict()
        out = [
            "  tag-drift verdict: %s  (source objects %d, changed %d)"
            % (d["status"], d["source_objects"], d["source_objects_changed"]),
            "    %s" % self.reason,
            "    evidence at the pinned commit remains valid: %s"
            % ("YES" if d["evidence_valid"] else "NO"),
            "    canonical source digest %s"
            % d["canonical_data_manifest_sha256"][:16],
            "    record: %s" % d["record_path"],
        ]
        if self.loader is not None:
            out.append("    loader census: %s"
                       % ", ".join(f"{k}={v}" for k, v in d["loader_census"].items()))
            if self.loader.loader_path_changed:
                out.append(
                    "    *** THE LOADER ROUTE MOVED. metadata.json now advertises "
                    "prepared_splits, and whestbench 0.16.0 PREFERS prepared/<split> "
                    "Arrow over data/*.parquet (dataset.py:479-491). Source bytes "
                    "are unchanged; the path to them is not. ***")
                out.append("    functional check on one shard: %s"
                           % _prepared_check_summary())
        if self.status == SOURCE_EQUIVALENT:
            out.append(
                "    NOTE: the executable pin is stale (bookkeeping). "
                "Prepared/metadata objects were added (%s -> %s); no source "
                "byte moved and nothing was removed."
                % (d["old_file_count"], d["new_file_count"]))
        elif self.status == SOURCE_CHANGED:
            out.append("    *** SOURCE BYTES MOVED. Re-derive every "
                       "shape-dependent Phase-2 result. ***")
        elif self.status == UNKNOWN:
            out.append("    *** UNMEASURED OR UNTRUSTED. Treat as invalidating "
                       "until a fresh read-only tree comparison is captured. ***")
        return out


def _rel(p: str) -> str:
    return os.path.relpath(p, _ROOT).replace("\\", "/")


def _prepared_check_summary() -> str:
    """One line about the functional prepared-vs-parquet comparison, or the
    honest absence of one. Never asserts a pass it cannot read."""
    import json
    try:
        with open(PREPARED_CHECK_PATH, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return ("NOT PRESENT -- the two routes have never been compared here. "
                "Run experiments/2026-08-31_wp0a_manifest/"
                "prepared_vs_parquet_probe.py")
    arr = d.get("array_columns", {})
    bits = ", ".join(
        "%s %s/%s bitwise" % (k, v.get("bitwise_identical_rows"), v.get("rows"))
        for k, v in sorted(arr.items()) if v.get("present"))
    return "%s (%s; %s)" % (d.get("verdict", "?"), bits, d.get("scope", "")[:60])


# --------------------------------------------------------------------------- #
# loading and the manifest block
# --------------------------------------------------------------------------- #

def load_record(path: str = RECORD_PATH) -> DriftRecord:
    with open(path, encoding="utf-8") as fh:
        return DriftRecord(fh.read(), path)


def load_loader_record(path: str = LOADER_RECORD_PATH) -> Optional[LoaderRecord]:
    try:
        with open(path, encoding="utf-8") as fh:
            return LoaderRecord(fh.read(), path)
    except FileNotFoundError:
        return None


def classify_live_commit(live_commit: str, pinned_commit: str,
                         path: str = RECORD_PATH,
                         loader_path: str = LOADER_RECORD_PATH) -> DriftVerdict:
    return load_record(path).classify(live_commit, pinned_commit,
                                      load_loader_record(loader_path))


def lineage_block(path: str = RECORD_PATH,
                  loader_path: str = LOADER_RECORD_PATH) -> dict:
    """The `dataset.commit_lineage` block for `phase2_manifest.json`.

    Field names say what they are. The first version called the record's OLD
    endpoint `pinned_commit`, which stopped being true the moment the manifest
    was repinned to the new one — a reviewer read
    `commit_lineage.pinned_commit != dataset.commit` and was right to flag it.
    `manifest_pin` is filled in by the manifest writer, which is the only place
    that knows it.
    """
    rec = load_record(path)
    loader = load_loader_record(loader_path)
    v = rec.classify(EXPECTED_NEW_COMMIT, EXPECTED_OLD_COMMIT, loader)
    block = {
        "record_old_commit": EXPECTED_OLD_COMMIT,
        "record_new_commit": EXPECTED_NEW_COMMIT,
        "manifest_pin": None,
        "status": v.status,
        "source_objects": len(rec.rows),
        "source_objects_changed": len(rec.differing_rows()),
        "canonical_data_manifest_sha256": rec.canonical_digest(),
        "record_passes_pinned_integrity": not rec.integrity(),
        "old_file_count": rec.old_file_count,
        "new_file_count": rec.new_file_count,
        "added_count": rec.added_count,
        "removed_count": rec.removed_count,
        "evidence_path": _rel(path),
        "meaning": (
            "Shape-dependent Phase-2 evidence depends on the data/*.parquet "
            "SOURCE objects. This block records a measured, content-addressed "
            "comparison of all of them across the tag move. A commit not named "
            "here is UNKNOWN and must be treated as invalidating."
        ),
    }
    if loader is not None:
        block["loader"] = {
            "evidence_path": _rel(loader.path),
            "canonical_loader_manifest_sha256": loader.canonical_digest(),
            "objects_compared": len(loader.rows),
            "census": {f"{k}:{s}": n for (k, s), n in sorted(loader.counts().items())},
            "loader_path_changed": loader.loader_path_changed,
            "why_this_exists": (
                "whestbench 0.16.0 PREFERS a prepared/<split> Arrow tree over "
                "data/*.parquet whenever metadata.json advertises "
                "prepared_splits (dataset.py:479-491, dataset_io.py:61-68), and "
                "this tag move added exactly that tree. A data/-only comparison "
                "does not cover what the loader reads."
            ),
            "functional_check": _rel(PREPARED_CHECK_PATH),
        }
    return block


def main() -> int:
    rec = load_record()
    loader = load_loader_record()
    print("Phase-2 dataset tag drift")
    print("  source record   %s" % _rel(rec.path))
    print("  loader record   %s" % (_rel(loader.path) if loader else "ABSENT"))
    print("  old commit      %s" % rec.old_commit)
    print("  new commit      %s" % rec.new_commit)
    print("  source objects  %d   (changed %d)"
          % (len(rec.rows), len(rec.differing_rows())))
    print("  files           %s -> %s   (+%s / -%s)"
          % (rec.old_file_count, rec.new_file_count,
             rec.added_count, rec.removed_count))
    print("  source digest   %s" % rec.canonical_digest())
    print("  integrity       %s"
          % ("OK against the pinned constants" if not rec.integrity()
             else "; ".join(rec.integrity())))
    if loader is not None:
        print("  loader digest   %s" % loader.canonical_digest())
        print("  loader census   %s"
              % ", ".join(f"{k}:{s}={n}" for (k, s), n in sorted(loader.counts().items())))
        print("  loader integ.   %s"
              % ("OK against the pinned constants" if not loader.integrity()
                 else "; ".join(loader.integrity())))
    print()
    v = rec.classify(EXPECTED_NEW_COMMIT, EXPECTED_OLD_COMMIT, loader)
    for line in v.report_lines():
        print(line)
    # `loader is None` used to make this expression TRUE, so a missing
    # loader record exited 0. It is a refusal now, matching classify().
    ok = (not rec.integrity() and loader is not None
          and not loader.integrity()
          and v.status in (PINNED, SOURCE_EQUIVALENT))
    if loader is None:
        print()
        print('  REFUSED: the loader record is absent. Regenerate it with')
        print('    uv run python experiments/2026-08-31_phase2_tag_drift/'
              'capture_loader_manifest.py')
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
