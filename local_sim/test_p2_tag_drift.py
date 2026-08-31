"""Regression for `local_sim/p2_tag_drift.py` — the source/loader tag-drift pin.

Contract: `experiments/LOG.md`, *CONTRACT WP0A* (`43b8f23`) and its addendum
*CONTRACT WP0A-FIX* — both committed before the code they describe.

WHAT THIS SUITE MUST BE ABLE TO DO
----------------------------------
Prove that `SOURCE_EQUIVALENT` **can fail**. A guard that returns "fine" for
every input is worth less than no guard, and this repository has shipped one of
those before (`docs/CURRENT_STATE.md`: a gate clause satisfied by a check that
could not fail).

The first version of this suite believed it had done that, and an independent
Skeptic broke it four ways. Every one of those four is now section **T**:

| tamper | first version | now |
|---|---|---|
| rewrite one header line (`# new_commit:`) | SOURCE_EQUIVALENT | UNKNOWN |
| change BOTH hash columns of a row identically, re-stamp the digest | SOURCE_EQUIVALENT | UNKNOWN |
| truncate to 5 of 70 rows, re-stamp the digest | SOURCE_EQUIVALENT | UNKNOWN |
| set `# removed_count: 999` | SOURCE_EQUIVALENT | UNKNOWN |

The fix is structural, not a longer list of checks: **a record can no longer
certify itself.** The expected digests, commit pair, row counts and object
counts are pinned in `p2_tag_drift.py` and the record is checked against those.

Both records on disk are opened READ-ONLY. Every mutation is applied to a string.

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

import p2_tag_drift as TD  # noqa: E402
from p2_tag_drift import (  # noqa: E402
    PINNED, SOURCE_CHANGED, SOURCE_EQUIVALENT, UNKNOWN,
    DriftRecord, DriftRecordError, LoaderRecord,
    LOADER_RECORD_PATH, PREPARED_CHECK_PATH, RECORD_PATH,
    lineage_block, load_loader_record, load_record,
)

_FAILED = []
_TOTAL = [0]

EXPECTED_DIGEST = "3cb3e81907069725662edc011638b095b399dafe42d35b97ea5456ece0cb8c66"
EXPECTED_LOADER_DIGEST = (
    "29664dbb64a02b8d8e26a213fcf741ee19d308a338520dbcea6c5b7d91901f2d")
EXPECTED_OLD = "ce926b7d1ced65faa668c01b94a43f8c4f43783d"
EXPECTED_NEW = "6a416fa5fb45cade1e9ad34320204853f5d4a5f6"
EXPECTED_ROWS = 70
EXPECTED_LOADER_ROWS = 226


def check(name, ok, detail=""):
    _TOTAL[0] += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<68}{detail}")
    if not ok:
        _FAILED.append(name)


def raises(name, fn, exc=DriftRecordError, detail=""):
    try:
        fn()
    except exc as e:
        check(name, True, detail or f"{type(e).__name__}: {str(e)[:56]}")
        return
    except Exception as e:
        check(name, False, f"raised {type(e).__name__}, expected {exc.__name__}")
        return
    check(name, False, "no exception raised")


def _read(p):
    with open(p, encoding="utf-8") as fh:
        return fh.read()


def _restamp_source(lines):
    """Re-stamp a mutated source record's header digest — what an attacker who
    knows how the check works would do."""
    rows = [l for l in lines if l and not l.startswith("#")][1:]
    d = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    out = [("# canonical_data_manifest_sha256: " + d)
           if l.startswith("# canonical_data_manifest_sha256") else l
           for l in lines]
    return DriftRecord("\n".join(out), "<restamped>")


def main():
    print("WP0A - the tag-drift records, their verdicts, and their tampers\n")
    src_text, ldr_text = _read(RECORD_PATH), _read(LOADER_RECORD_PATH)
    before = {p: hashlib.sha256(_read(p).encode("utf-8")).hexdigest()
              for p in (RECORD_PATH, LOADER_RECORD_PATH)}
    rec = load_record()
    loader = load_loader_record()

    # ---------------------------------------------------------------- P1/P2 --
    print("P1/P2  the source record reproduces its pinned digest; no source moved")
    check("P1: canonical digest recomputes from the rows",
          rec.canonical_digest() == EXPECTED_DIGEST, rec.canonical_digest()[:24])
    check("P1: it matches the PINNED constant, not merely its own header",
          rec.digest_ok() and TD.EXPECTED_SOURCE_DIGEST == EXPECTED_DIGEST,
          "a record cannot certify itself")
    check("P2: source objects that changed bytes", len(rec.differing_rows()) == 0,
          f"{len(rec.differing_rows())} of {len(rec.rows)}")
    check("the record passes every pinned integrity clause",
          rec.integrity() == [], "; ".join(rec.integrity())[:56] or "no violations")
    check("record covers the expected object count", len(rec.rows) == EXPECTED_ROWS,
          f"{len(rec.rows)} rows")
    check("every row is a data/ SOURCE object",
          all(p.startswith("data/") for p, _, _, _ in rec.rows), "by construction")
    check("prose comment lines do not become header fields",
          not any(" " in k for k in rec.header),
          f"{len(rec.header)} header keys, all identifier-ish")

    # ------------------------------------------------------------- LOADER ---
    print("\nL  the FULL loader-visible tree, because data/ is NOT the whole input")
    check("L: the loader record exists", loader is not None, TD._rel(LOADER_RECORD_PATH))
    check("L: it covers every object at the new commit",
          len(loader.rows) == EXPECTED_LOADER_ROWS, f"{len(loader.rows)} rows")
    check("L: its digest recomputes to the pinned constant",
          loader.canonical_digest() == EXPECTED_LOADER_DIGEST,
          loader.canonical_digest()[:24])
    check("L: it passes every pinned integrity clause",
          loader.integrity() == [], "; ".join(loader.integrity())[:56] or "no violations")
    census = loader.counts()
    check("L: 70 source objects unchanged", census.get(("source", "unchanged")) == 70,
          str(census.get(("source", "unchanged"))))
    check("L: README and .gitattributes unchanged",
          census.get(("readme", "unchanged")) == 1
          and census.get(("other", "unchanged")) == 1, "1 + 1")
    check("L: metadata.json CHANGED", census.get(("metadata", "changed")) == 1,
          "it gained prepared_splits")
    check("L: 153 prepared objects ADDED", census.get(("prepared", "added")) == 153,
          "the tree whestbench now prefers")
    check("L: NOTHING was removed",
          not any(s == "removed" for _, s in census), sorted(census))
    check("L: the verdict reports that the LOADER ROUTE moved",
          loader.loader_path_changed is True,
          "whestbench dataset.py:479-491 prefers prepared/<split>")
    check("L: source-equivalence and loader-route change are SEPARATE fields",
          rec.classify(EXPECTED_NEW, EXPECTED_OLD, loader).status == SOURCE_EQUIVALENT
          and rec.classify(EXPECTED_NEW, EXPECTED_OLD, loader).loader_path_changed
          is True,
          "the bytes did not move; the path to them did")

    # the functional comparison of the two routes
    print("\nF  the two loader routes were compared functionally, on real bytes")
    if os.path.exists(PREPARED_CHECK_PATH):
        with open(PREPARED_CHECK_PATH, encoding="utf-8") as fh:
            pc = json.load(fh)
        check("F: prepared/mini shard 0 vs data/mini shard 0 was measured",
              pc.get("rows_compared", 0) > 0, f"{pc.get('rows_compared')} rows")
        for col in ("weights", "all_layer_means"):
            a = pc.get("array_columns", {}).get(col, {})
            check(f"F: '{col}' is BITWISE identical on every compared row",
                  a.get("present") and a.get("bitwise_identical_rows") == a.get("rows")
                  and a.get("max_abs_difference") == 0.0,
                  f"{a.get('bitwise_identical_rows')}/{a.get('rows')}, "
                  f"max|diff|={a.get('max_abs_difference')}")
        check("F: mlp_id/mlp_seed agree IN ORDER, not merely as sets",
              all(pc["scalar_columns"][c].get("identical_in_order")
                  for c in ("mlp_id", "mlp_seed", "mlp_name", "avg_variance")),
              "row order decides which 50 rows are publicly scored")
        check("F: the schemas match exactly",
              pc.get("columns_added_by_prepared") == []
              and pc.get("columns_missing_from_prepared") == [], "no column drift")
        check("F: the record states its own scope honestly",
              "One of 14 prepared shards" in pc.get("scope", ""),
              "13 shards and the full split remain unverified")
    else:
        check("F: the functional comparison artifact exists", False,
              TD._rel(PREPARED_CHECK_PATH))

    # ------------------------------------------------------------------- P3 --
    print("\nP3  the live commit classifies as SOURCE_EQUIVALENT, and evidence holds")
    v = rec.classify(EXPECTED_NEW, EXPECTED_OLD, loader)
    check("P3: live 6a416f against pinned ce926b7 is SOURCE_EQUIVALENT",
          v.status == SOURCE_EQUIVALENT, v.status)
    check("P3: shape-dependent evidence stays valid", v.evidence_valid is True,
          v.reason[:48])
    check("P3: and the executable pin is nonetheless reported STALE",
          v.pin_is_stale is True, "bookkeeping, separate from evidence validity")
    same = rec.classify(EXPECTED_OLD, EXPECTED_OLD, loader)
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
        ("live is right but the pin is not the recorded old commit",
         EXPECTED_NEW, "1" * 40),
        ("a prefix of the real commit is not the real commit",
         EXPECTED_NEW[:12], EXPECTED_OLD),
    ):
        u = rec.classify(live, pinned, loader)
        check(f"P4: {label} -> UNKNOWN", u.status == UNKNOWN, u.status)
        check(f"P4: {label} -> evidence NOT asserted valid",
              u.evidence_valid is False, "UNKNOWN is not a pass")
    check("the records license exactly ONE transition",
          rec.classify(EXPECTED_OLD, EXPECTED_NEW, loader).status == UNKNOWN,
          "the reverse move is unmeasured")

    # -------------------------------------------------------------------- T --
    print("\nT  the four tampers an independent Skeptic used to obtain a false pass")
    lines = src_text.splitlines()
    body_idx = [i for i, l in enumerate(lines) if not l.startswith("#")]
    first_row_i, last_row_i = body_idx[1], body_idx[-1]

    # T1 -- rewrite one header line. Nothing in the rows moves.
    t1 = [("# new_commit: " + "d" * 40) if l.startswith("# new_commit") else l
          for l in lines]
    r1 = DriftRecord("\n".join(t1), "<T1>")
    check("T1: a rewritten `# new_commit` header -> UNKNOWN",
          r1.classify(EXPECTED_NEW, EXPECTED_OLD, loader).status == UNKNOWN,
          "the commit pair is pinned in code")
    check("T1: ...and the row digest alone would NOT have caught it",
          r1.canonical_digest() == EXPECTED_DIGEST,
          "which is exactly why the header is pinned too")

    # T2 -- change BOTH hash columns of a row identically, then re-stamp.
    t2 = list(lines)
    p, size, old, new = t2[first_row_i].split("\t")
    forged = ("0" if old[0] != "0" else "1") + old[1:]
    t2[first_row_i] = "\t".join([p, size, forged, forged])
    r2 = _restamp_source(t2)
    check("T2: both columns forged identically + re-stamped digest -> UNKNOWN",
          r2.classify(EXPECTED_NEW, EXPECTED_OLD, loader).status == UNKNOWN,
          "old == new, so `differing_rows` sees nothing; the digest does")
    check("T2: ...and `differing_rows` really is blind to it",
          len(r2.differing_rows()) == 0, "the naive check would have passed")

    # T3 -- truncate to 5 rows, then re-stamp.
    t3 = [l for l in lines if l.startswith("#")] + [lines[body_idx[0]]] \
        + lines[first_row_i:first_row_i + 5]
    r3 = _restamp_source(t3)
    check("T3: truncation to 5 of 70 rows + re-stamped digest -> UNKNOWN",
          r3.classify(EXPECTED_NEW, EXPECTED_OLD, loader).status == UNKNOWN,
          f"{len(r3.rows)} rows against a pinned {EXPECTED_ROWS}")
    check("T3: ...and coverage is enforced in classify(), not in a test literal",
          any("row count" in x for x in r3.integrity()),
          "; ".join(r3.integrity())[:56])

    # T4 -- claim a removal in the header.
    t4 = [("# removed_count: 999") if l.startswith("# removed_count") else l
          for l in lines]
    r4 = DriftRecord("\n".join(t4), "<T4>")
    check("T4: `removed_count: 999` -> UNKNOWN",
          r4.classify(EXPECTED_NEW, EXPECTED_OLD, loader).status == UNKNOWN,
          "a REMOVAL is never source-equivalent")
    check("T4: ...and the reason names the removal, not something else",
          any("REMOVAL" in x or "removed_count" in x for x in r4.integrity()),
          "; ".join(r4.integrity())[:56])

    # T5 -- the same class of attack on the LOADER record.
    l5 = [("# canonical_loader_manifest_sha256: " + "a" * 64)
          if l.startswith("# canonical_loader_manifest_sha256") else l
          for l in ldr_text.splitlines()]
    lr5 = LoaderRecord("\n".join(l5), "<T5>")
    check("T5: a forged loader digest is refused",
          any("header digest" in x for x in lr5.integrity()), "pinned in code")
    check("T5: ...and it degrades the SOURCE verdict to UNKNOWN too",
          rec.classify(EXPECTED_NEW, EXPECTED_OLD, lr5).status == UNKNOWN,
          "an untrusted loader record licenses nothing")
    l6 = [l for l in ldr_text.splitlines()]
    for i, l in enumerate(l6):
        if l.startswith("prepared/") and "\tadded\t" in l:
            l6[i] = l.replace("\tadded\t", "\tremoved\t", 1)
            break
    lr6 = LoaderRecord("\n".join(l6), "<T6>")
    check("T6: a smuggled REMOVED row in the loader record is refused",
          any("REMOVED" in x or "census" in x for x in lr6.integrity()),
          "; ".join(lr6.integrity())[:56])

    # ------------------------------------------------------------------ M* --
    print("\nM  structural corruption is refused before any verdict is formed")
    m = list(lines)
    p, size, old, new = m[first_row_i].split("\t")
    m[first_row_i] = "\t".join([p, size, old, ("0" if new[0] != "0" else "1") + new[1:]])
    r = _restamp_source(m)
    check("M1: one asymmetric flipped hash + repaired digest -> SOURCE_CHANGED",
          r.classify(EXPECTED_NEW, EXPECTED_OLD, loader).status
          in (SOURCE_CHANGED, UNKNOWN),
          r.classify(EXPECTED_NEW, EXPECTED_OLD, loader).status)
    m4 = list(lines) + ["prepared/mini-000.arrow\t1\t" + "b" * 64 + "\t" + "c" * 64]
    raises("M4: a non-data/ path in the SOURCE record is refused",
           lambda: DriftRecord("\n".join(m4), "<M4>"))
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
    m6 = list(lines)
    m6.insert(last_row_i + 1, lines[first_row_i])
    raises("M6: a duplicated source path is refused",
           lambda: DriftRecord("\n".join(m6), "<M6>"))
    l7 = ldr_text.splitlines()
    l7.append(l7[-1])
    raises("M7: a duplicated path in the loader record is refused",
           lambda: LoaderRecord("\n".join(l7), "<M7>"))

    # ------------------------------------------------- the lineage block ----
    print("\nD  the manifest lineage block, and the field names a reviewer read wrong")
    lb = lineage_block()
    check("D: the record's two endpoints are named for what they ARE",
          lb["record_old_commit"] == EXPECTED_OLD
          and lb["record_new_commit"] == EXPECTED_NEW,
          "not 'pinned_commit', which stopped being true after the repin")
    check("D: `manifest_pin` is a separate field the manifest writer fills",
          "manifest_pin" in lb, str(lb["manifest_pin"]))
    check("D: canonical digest carried",
          lb["canonical_data_manifest_sha256"] == EXPECTED_DIGEST,
          lb["canonical_data_manifest_sha256"][:16])
    check("D: the block records that the record passed pinned integrity",
          lb["record_passes_pinned_integrity"] is True, "not self-certified")
    check("D: the loader half is carried too",
          lb.get("loader", {}).get("loader_path_changed") is True
          and lb["loader"]["objects_compared"] == EXPECTED_LOADER_ROWS,
          f"{lb.get('loader', {}).get('objects_compared')} objects")
    check("D: source vs prepared counts are SEPARATE fields",
          lb["source_objects"] == EXPECTED_ROWS
          and lb["source_objects_changed"] == 0 and lb["added_count"] == 153,
          "70 source unchanged, 153 prepared added")
    check("D: both evidence paths exist on disk",
          os.path.exists(os.path.join(_ROOT, lb["evidence_path"]))
          and os.path.exists(os.path.join(_ROOT, lb["loader"]["evidence_path"])),
          lb["evidence_path"])
    check("D: the block is JSON-serialisable",
          json.loads(json.dumps(lb)) == lb, "round-trips")

    man_path = os.path.join(_ROOT, "experiments", "2026-08-21_phase2_manifest",
                            "phase2_manifest.json")
    with open(man_path, encoding="utf-8") as fh:
        man = json.load(fh)
    ds = man["dataset"]
    lin = ds.get("commit_lineage", {})
    check("D: the manifest carries dataset.commit_lineage", bool(lin), "present")
    check("D: the pin is one of the record's two endpoints",
          ds["commit"] in (lin.get("record_old_commit"),
                           lin.get("record_new_commit")),
          f"pinned at {ds['commit'][:12]}")
    check("D: `manifest_pin` in the stored block equals dataset.commit",
          lin.get("manifest_pin") == ds["commit"],
          "the field a reviewer found self-contradictory is now filled")
    check("D: the OLD commit our stored evidence used is preserved",
          lin.get("record_old_commit") == EXPECTED_OLD, "provenance survives")
    check("D: shape and budget are unchanged by the repin",
          man["shape"]["width"] == 1024 and man["shape"]["depth"] == 16
          and man["budget"]["flop_budget_per_mlp"] == 2 ** 41
          and man["budget"]["multiplier_floor"] == 0.1,
          "1024x16, B = 2**41, floor 0.1")

    # ------------------------------------------------------------ read-only --
    print("\nR  neither durable record was modified by this suite")
    for p_, h in before.items():
        check(f"R: {os.path.basename(p_)} sha256 unchanged",
              hashlib.sha256(_read(p_).encode("utf-8")).hexdigest() == h, h[:16])

    print()
    if _FAILED:
        print(f"FAILED {len(_FAILED)}: " + ", ".join(_FAILED))
        raise SystemExit(1)
    print(f"{_TOTAL[0]}/{_TOTAL[0]} checks passed")


if __name__ == "__main__":
    main()
