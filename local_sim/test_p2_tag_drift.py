"""Regression for `local_sim/p2_tag_drift.py` — the source-vs-metadata pin.

Contract: `experiments/LOG.md`, *CONTRACT WP0A* (2026-08-31), committed at
`43b8f23` before this file existed. Predictions P1–P5 appear below as named
checks; P6 (no executable measurement moves) is preflight's job, not this
suite's.

WHAT THIS SUITE MUST BE ABLE TO DO
----------------------------------
Prove that the new `SOURCE_EQUIVALENT` verdict **can fail**. A guard that
returns "fine" for every input is worth less than no guard, and this repository
has shipped one of those before (`docs/CURRENT_STATE.md`: a gate clause
satisfied by a check that could not fail). Sections M1–M6 are therefore
mutations of an in-memory copy of the durable record, each of which must flip
the verdict or raise.

The real record on disk is opened READ-ONLY and never rewritten. Every mutation
is applied to a string.

Run: uv run python local_sim/test_p2_tag_drift.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# local_sim ships shims NAMED flopscope/whestbench: APPEND, never insert.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

from p2_tag_drift import (  # noqa: E402
    PINNED, SOURCE_CHANGED, SOURCE_EQUIVALENT, UNKNOWN,
    DriftRecord, DriftRecordError, RECORD_PATH, lineage_block, load_record,
)

_FAILED = []
_TOTAL = [0]

#: The value the contract predicts (P1). Written down so the suite tests the
#: record against a stated expectation rather than against whatever it emits.
EXPECTED_DIGEST = "3cb3e81907069725662edc011638b095b399dafe42d35b97ea5456ece0cb8c66"
EXPECTED_OLD = "ce926b7d1ced65faa668c01b94a43f8c4f43783d"
EXPECTED_NEW = "6a416fa5fb45cade1e9ad34320204853f5d4a5f6"
EXPECTED_ROWS = 70


def check(name, ok, detail=""):
    _TOTAL[0] += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<64}{detail}")
    if not ok:
        _FAILED.append(name)


def raises(name, fn, exc=DriftRecordError, detail=""):
    try:
        fn()
    except exc as e:
        check(name, True, detail or f"{type(e).__name__}: {str(e)[:60]}")
        return
    except Exception as e:  # wrong exception type is still a failure
        check(name, False, f"raised {type(e).__name__}, expected {exc.__name__}")
        return
    check(name, False, "no exception raised")


def _raw_text():
    with open(RECORD_PATH, encoding="utf-8") as fh:
        return fh.read()


def main():
    print("WP0A — Phase-2 tag-drift record, verdicts, and their mutations\n")
    text = _raw_text()
    before = hashlib.sha256(text.encode("utf-8")).hexdigest()
    rec = load_record()

    # ---------------------------------------------------------------- P1/P2 --
    print("P1/P2  the durable record reproduces its own digest, and no source moved")
    check("P1: canonical digest recomputes from the rows",
          rec.canonical_digest() == EXPECTED_DIGEST,
          rec.canonical_digest()[:24])
    check("P1: recomputed digest equals the recorded header digest",
          rec.digest_ok(), f"header {str(rec.recorded_digest)[:24]}")
    check("P2: source objects that changed bytes", len(rec.differing_rows()) == 0,
          f"{len(rec.differing_rows())} of {len(rec.rows)}")
    check("record covers the expected object count", len(rec.rows) == EXPECTED_ROWS,
          f"{len(rec.rows)} rows")
    check("every row is a data/ SOURCE object",
          all(p.startswith("data/") for p, _, _, _ in rec.rows),
          "prepared/metadata objects are excluded by construction")
    check("commits recorded are the two known ones",
          rec.old_commit == EXPECTED_OLD and rec.new_commit == EXPECTED_NEW,
          f"{str(rec.old_commit)[:8]} -> {str(rec.new_commit)[:8]}")
    check("the file-count delta is prepared-only, nothing removed",
          rec.removed_count == 0 and rec.added_count == 153
          and rec.new_file_count - rec.old_file_count == rec.added_count,
          f"{rec.old_file_count} -> {rec.new_file_count}, +{rec.added_count}")
    # A prose comment line containing a colon must not become a header field.
    check("prose comment lines do not become header fields",
          "Canonical digest input is the 70 data rows in path order, joined by LF "
          "with no trailing LF" not in rec.header,
          f"{len(rec.header)} header keys: {sorted(rec.header)[:3]}...")

    # ------------------------------------------------------------------- P3 --
    print("\nP3  the live commit classifies as SOURCE_EQUIVALENT, and evidence holds")
    v = rec.classify(EXPECTED_NEW, EXPECTED_OLD)
    check("P3: live 6a416f… against pinned ce926b7… is SOURCE_EQUIVALENT",
          v.status == SOURCE_EQUIVALENT, v.status)
    check("P3: shape-dependent evidence stays valid", v.evidence_valid is True,
          v.reason[:52])
    check("P3: and the executable pin is nonetheless reported STALE",
          v.pin_is_stale is True, "bookkeeping, separate from evidence validity")
    same = rec.classify(EXPECTED_OLD, EXPECTED_OLD)
    check("no-move classifies PINNED, not SOURCE_EQUIVALENT",
          same.status == PINNED and same.pin_is_stale is False, same.status)

    # ------------------------------------------------------------------- P4 --
    print("\nP4  an unrecorded commit can NEVER obtain SOURCE_EQUIVALENT")
    for label, live, pinned in (
        ("an arbitrary unrecorded commit", "0" * 40, EXPECTED_OLD),
        ("the OLD commit presented as live against the NEW pin",
         EXPECTED_OLD, EXPECTED_NEW),
        ("a third commit against the recorded pin", "f" * 40, EXPECTED_OLD),
        ("empty live commit", "", EXPECTED_OLD),
        ("live is right but the manifest pin is not the recorded old commit",
         EXPECTED_NEW, "1" * 40),
        ("a prefix of the real commit is not the real commit",
         EXPECTED_NEW[:12], EXPECTED_OLD),
    ):
        u = rec.classify(live, pinned)
        check(f"P4: {label} -> UNKNOWN", u.status == UNKNOWN, u.status)
        check(f"P4: {label} -> evidence NOT asserted valid",
              u.evidence_valid is False, "UNKNOWN is not a pass")
    check("the record licenses exactly ONE transition",
          rec.classify(EXPECTED_NEW, EXPECTED_OLD).status == SOURCE_EQUIVALENT
          and rec.classify(EXPECTED_OLD, EXPECTED_NEW).status == UNKNOWN,
          "direction matters; a reverse move is unmeasured")

    # ---------------------------------------------------------------- P5/M* --
    print("\nP5/M  mutations of an in-memory copy must each break something")
    lines = text.splitlines()
    body_idx = [i for i, l in enumerate(lines) if not l.startswith("#")]
    first_row_i = body_idx[1]          # [0] is the column header
    last_row_i = body_idx[-1]

    # M1 -- one flipped source hash: SOURCE_CHANGED *and* a broken digest.
    m = list(lines)
    p, size, old, new = m[first_row_i].split("\t")
    new_mut = ("0" if new[0] != "0" else "1") + new[1:]
    m[first_row_i] = "\t".join([p, size, old, new_mut])
    r1 = DriftRecord("\n".join(m), "<M1>")
    check("M1: one flipped new_lfs_sha256 -> digest no longer reproduces",
          not r1.digest_ok(), r1.canonical_digest()[:16])
    check("M1: ...and one differing source row is seen",
          len(r1.differing_rows()) == 1, f"{len(r1.differing_rows())} rows differ")
    check("M1: with the header digest ALSO repaired, the verdict is SOURCE_CHANGED",
          _repair_digest(m).classify(EXPECTED_NEW, EXPECTED_OLD).status
          == SOURCE_CHANGED,
          "an attacker who fixes the digest still cannot get SOURCE_EQUIVALENT")

    # M2 -- an untampered-looking record whose header digest was forged.
    m2 = list(lines)
    hdr_i = [i for i, l in enumerate(m2)
             if l.startswith("# canonical_data_manifest_sha256")][0]
    m2[hdr_i] = "# canonical_data_manifest_sha256: " + "a" * 64
    r2 = DriftRecord("\n".join(m2), "<M2>")
    check("M2: a forged header digest is refused, not trusted",
          not r2.digest_ok(), "recomputation is the authority")
    check("M2: ...and every classification degrades to UNKNOWN",
          r2.classify(EXPECTED_NEW, EXPECTED_OLD).status == UNKNOWN,
          "an untrusted record licenses nothing")

    # M3 -- a dropped row. Digest breaks; count breaks.
    m3 = [l for i, l in enumerate(lines) if i != last_row_i]
    r3 = DriftRecord("\n".join(m3), "<M3>")
    check("M3: a dropped source row breaks the digest",
          not r3.digest_ok(), f"{len(r3.rows)} rows")
    check("M3: ...and is visible as a count change",
          len(r3.rows) == EXPECTED_ROWS - 1, f"{len(r3.rows)}")
    check("M3: even with a repaired digest, the coverage claim is now false",
          len(_repair_digest(m3).rows) != EXPECTED_ROWS,
          "the suite pins 70; a silent shrink fails here")

    # M4 -- a smuggled non-source object.
    m4 = list(lines) + ["prepared/mini-000.parquet\t1\t" + "b" * 64 + "\t" + "c" * 64]
    raises("M4: a non-data/ path in the record is refused",
           lambda: DriftRecord("\n".join(m4), "<M4>"),
           detail="the record is defined over SOURCE objects only")

    # M5 -- structural corruption.
    m5 = list(lines)
    m5[first_row_i] = m5[first_row_i].replace("\t", " ", 1)
    raises("M5: a row with the wrong column count is refused",
           lambda: DriftRecord("\n".join(m5), "<M5>"))
    m5b = list(lines)
    m5b[body_idx[0]] = "path\tsize\tsha\tsha2"
    raises("M5: a renamed column header is refused",
           lambda: DriftRecord("\n".join(m5b), "<M5b>"))
    m5c = list(lines)
    p, size, old, new = m5c[first_row_i].split("\t")
    m5c[first_row_i] = "\t".join([p, size, old[:-1], new])
    raises("M5: a truncated sha256 is refused",
           lambda: DriftRecord("\n".join(m5c), "<M5c>"))

    # M6 -- a duplicated row would let a record cover 70 names with 69 objects.
    m6 = list(lines)
    m6.insert(last_row_i + 1, lines[first_row_i])
    raises("M6: a duplicated source path is refused",
           lambda: DriftRecord("\n".join(m6), "<M6>"))

    # ------------------------------------------------------- lineage block --
    print("\nL  the manifest lineage block preserves BOTH commits")
    lb = lineage_block()
    check("L: pinned commit preserved", lb["pinned_commit"] == EXPECTED_OLD,
          str(lb["pinned_commit"])[:16])
    check("L: source-equivalent commit recorded",
          lb["source_equivalent_commit"] == EXPECTED_NEW,
          str(lb["source_equivalent_commit"])[:16])
    check("L: canonical digest carried", lb["canonical_data_manifest_sha256"]
          == EXPECTED_DIGEST, lb["canonical_data_manifest_sha256"][:16])
    check("L: status is SOURCE_EQUIVALENT", lb["status"] == SOURCE_EQUIVALENT,
          lb["status"])
    check("L: source vs prepared counts are separate fields",
          lb["source_objects"] == EXPECTED_ROWS
          and lb["added_count"] == 153 and lb["source_objects_changed"] == 0,
          f"{lb['source_objects']} source, +{lb['added_count']} prepared")
    check("L: evidence path exists on disk",
          os.path.exists(os.path.join(_ROOT, lb["evidence_path"])),
          lb["evidence_path"])
    check("L: the block is JSON-serialisable", _json_ok(lb), "round-trips")

    # ------------------------------------------------- the manifest on disk --
    print("\nD  the manifest on disk is pinned at a RECORDED endpoint")
    man_path = os.path.join(_ROOT, "experiments", "2026-08-21_phase2_manifest",
                            "phase2_manifest.json")
    with open(man_path, encoding="utf-8") as fh:
        man = json.load(fh)
    ds = man["dataset"]
    check("D: the manifest carries dataset.commit_lineage",
          "commit_lineage" in ds, "both commit IDs survive a rebuild")
    lin2 = ds.get("commit_lineage", {})
    check("D: the pin is one of the record's two endpoints",
          ds["commit"] in (lin2.get("pinned_commit"),
                           lin2.get("source_equivalent_commit")),
          f"pinned at {ds['commit'][:12]}")
    check("D: the lineage preserves the OLD commit our stored evidence used",
          lin2.get("pinned_commit") == EXPECTED_OLD,
          "provenance of every pre-2026-08-31 CONFIRM/mint byte")
    check("D: the lineage carries the canonical 70-row digest",
          lin2.get("canonical_data_manifest_sha256") == EXPECTED_DIGEST,
          str(lin2.get("canonical_data_manifest_sha256"))[:16])
    check("D: source objects and prepared additions are SEPARATE fields",
          lin2.get("source_objects") == EXPECTED_ROWS
          and lin2.get("source_objects_changed") == 0
          and lin2.get("added_count") == 153,
          "70 source unchanged, 153 prepared added")
    check("D: shape and budget are unchanged by the repin",
          man["shape"]["width"] == 1024 and man["shape"]["depth"] == 16
          and man["budget"]["flop_budget_per_mlp"] == 2 ** 41
          and man["budget"]["multiplier_floor"] == 0.1,
          "1024x16, B = 2**41, floor 0.1")

    # ------------------------------------------------------------ read-only --
    print("\nR  the durable record was not modified by this suite")
    after = hashlib.sha256(_raw_text().encode("utf-8")).hexdigest()
    check("R: source_manifest.tsv sha256 unchanged", before == after,
          before[:16])

    print()
    if _FAILED:
        print(f"FAILED {len(_FAILED)}: " + ", ".join(_FAILED))
        raise SystemExit(1)
    total = _TOTAL[0]
    print(f"{total}/{total} checks passed")


def _repair_digest(mutated_lines):
    """Re-stamp the header digest of a mutated record, so a mutation test can
    isolate the ROW check from the DIGEST check instead of conflating them."""
    lines = list(mutated_lines)
    rows = [l for l in lines if l and not l.startswith("#")][1:]
    d = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    for i, l in enumerate(lines):
        if l.startswith("# canonical_data_manifest_sha256"):
            lines[i] = "# canonical_data_manifest_sha256: " + d
    return DriftRecord("\n".join(lines), "<repaired>")


def _json_ok(obj):
    try:
        return json.loads(json.dumps(obj)) == obj
    except Exception:
        return False


if __name__ == "__main__":
    main()
