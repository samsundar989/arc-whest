"""
Does the seed ledger actually BITE? — the negative tests.

Stage 0 / P0.4. Contract and predictions A1-A6: `experiments/LOG.md`,
2026-08-19 (night).

`seed_ledger --check` passing on the real ledger proves the ledger is
self-consistent. It does not prove the enforcement would ever stop anything. Every
check below drives the module into the failure it exists to prevent and demands it
refuse. They run against a TEMPORARY ledger file; the real
`experiments/seed_ledger.jsonl` is opened read-only by the last two checks and is
never written.

Run: uv run python local_sim/test_seed_ledger.py

The seed numbers below are written into a TEMPORARY ledger to prove the refusals
fire, and D16's free gaps are asserted as literals. No network is ever minted from
any of them and no truth is drawn, so this is not consumption — but the scan cannot
know that, so the file declares it in the pragma below. The declaration is itself
checked: `t_real_ledger_readonly` asserts the real ledger records zero consumption
for D17/D19 and exactly 270 for D16, which would fail if this file ever did mint
from them. D18 is LIVE and grows, so it is checked by invariant (accounting closes,
every claim names a real consumer inside the block) rather than by snapshot — two
earlier snapshot assertions went stale within hours of being written.
"""

# seed-ledger: excludes D16, D17, D18, D19

import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import seed_ledger as SL  # noqa: E402
from seed_blocks import SINGLE_USE_BLOCKS, check_disjoint  # noqa: E402

RESULTS = []
TS = "2026-08-19"


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {name:<62} {'ok ' if ok else 'FAILED'}  {detail}")
    return ok


def head(t):
    print(f"\n{t}\n{'-' * len(t)}")


def _raises(fn, exc=Exception):
    try:
        fn()
        return False
    except exc:
        return True
    except Exception:
        return False


def t_single_use():
    head("1. a SINGLE_USE block may be claimed exactly once, ever")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "l.jsonl")
        SL.claim("D17", 50, "first.py", ts=TS, path=p)
        check("first claim on D17 succeeds", len(SL.ledger(p)) == 1)
        check(
            "a SECOND claim on D17 is refused even on disjoint seeds",
            _raises(lambda: SL.claim("D17", 50, "second.py", ts=TS, path=p), PermissionError),
        )
        check(
            "an explicit non-overlapping range is refused too",
            _raises(
                lambda: SL.claim_range("D17", 30_806_017_400, 10, "third.py", ts=TS, path=p),
                PermissionError,
            ),
        )
        check("the refused claims wrote nothing", len(SL.ledger(p)) == 1)
        check("D17 and D19 are registered single-use", {"D17", "D19"} <= set(SINGLE_USE_BLOCKS))


def t_design_overlap_allowed_but_reported():
    head("2. a DESIGN block permits overlap — and always reports it")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "l.jsonl")
        SL.claim_range("D18", 30_806_018_000, 24, "probe_a.py", ts=TS, path=p)
        SL.claim_range("D18", 30_806_018_000, 24, "probe_b.py", ts=TS, path=p)
        check("the overlapping DESIGN claim is allowed", len(SL.ledger(p)) == 2)
        check("and it is reported as a collision", len(SL.collisions(p)) == 1)
        check(
            "check_ledger does NOT error on a DESIGN overlap",
            SL.check_ledger(p, verbose=False) == [],
        )
        check(
            "distinct-seed accounting counts the shared range once",
            SL.consumed_seeds("D18", p) == 24,
            f"{SL.consumed_seeds('D18', p)} distinct from 48 claimed",
        )


def t_bounds_and_unknown_blocks():
    head("3. ranges that leave their block, and blocks that do not exist")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "l.jsonl")
        check(
            "a range running past the end of the block is refused",
            _raises(
                lambda: SL.claim_range("D18", 30_806_018_590, 20, "over.py", ts=TS, path=p),
                ValueError,
            ),
        )
        check(
            "a range starting before the block is refused",
            _raises(
                lambda: SL.claim_range("D18", 30_806_017_990, 20, "under.py", ts=TS, path=p),
                ValueError,
            ),
        )
        check(
            "an unregistered block is refused",
            _raises(lambda: SL.claim("D99", 10, "ghost.py", ts=TS, path=p), KeyError),
        )
        check("n <= 0 is refused", _raises(lambda: SL.claim_range(
            "D18", 30_806_018_000, 0, "empty.py", ts=TS, path=p), ValueError))
        check("nothing was written", SL.ledger(p) == [])


def t_exhaustion_and_free_ranges():
    head("4. free-range accounting, and running a block dry")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "l.jsonl")
        SL.claim_range("D18", 30_806_018_000, 100, "a.py", ts=TS, path=p)
        SL.claim_range("D18", 30_806_018_200, 100, "b.py", ts=TS, path=p)
        fr = SL.free_ranges("D18", p)
        check(
            "free ranges are the gaps, exactly",
            fr == [(30_806_018_100, 30_806_018_199), (30_806_018_300, 30_806_018_599)],
            f"{fr}",
        )
        got = SL.claim("D18", 100, "c.py", ts=TS, path=p)
        check("claim() takes the FIRST gap that fits", got[0] == 30_806_018_100)
        check(
            "a request larger than any gap is refused",
            _raises(lambda: SL.claim("D18", 400, "d.py", ts=TS, path=p), RuntimeError),
        )


def t_ledger_tamper():
    head("5. a hand-edited ledger does not pass quietly")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "l.jsonl")
        SL.claim_range("D18", 30_806_018_000, 10, "a.py", ts=TS, path=p)
        with open(p, "a", encoding="utf-8") as fh:  # out of bounds, written by hand
            fh.write('{"block":"D18","start":30806019999,"n":5,"consumer":"hand.py",'
                     '"tier":"DESIGN","ts":"2026-08-19"}\n')
        errs = SL.check_ledger(p, verbose=False)
        check("an out-of-block hand-written record is caught", any("outside" in e for e in errs),
              f"{len(errs)} error(s)")
        with open(p, "a", encoding="utf-8") as fh:  # wrong tier
            fh.write('{"block":"D17","start":30806017000,"n":5,"consumer":"lie.py",'
                     '"tier":"DESIGN","ts":"2026-08-19"}\n')
        errs = SL.check_ledger(p, verbose=False)
        check("a record claiming the wrong tier is caught", any("tier" in e for e in errs))
        with open(p, "a", encoding="utf-8") as fh:
            fh.write("not json\n")
        check("a malformed line raises rather than being skipped",
              _raises(lambda: SL.ledger(p), ValueError))


def t_hard_collision_on_decision():
    head("6. a DECISION-tier overlap is a hard error, not a note")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "l.jsonl")
        # Write two overlapping SINGLE_USE records by hand — claim() would refuse
        # the second, so this is the only way the state can arise at all.
        for consumer in ("a.py", "b.py"):
            with open(p, "a", encoding="utf-8") as fh:
                fh.write('{"block":"D19","start":30806019000,"n":50,"consumer":"%s",'
                         '"tier":"SINGLE_USE","ts":"2026-08-19"}\n' % consumer)
        errs = SL.check_ledger(p, verbose=False)
        check("claimed twice is caught", any("claimed twice" in e for e in errs))
        check("the seed overlap is ALSO caught as a hard collision",
              any("draw the same seeds" in e for e in errs), f"{len(errs)} error(s)")


def t_artifact_hash():
    head("7. an artifact that changes after the claim is caught")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "l.jsonl")
        art = os.path.join(d, "result.json")
        with open(art, "w", encoding="utf-8") as fh:
            fh.write('{"result": 1}')
        rel = os.path.relpath(art, _ROOT)
        SL.claim_range("D18", 30_806_018_000, 10, "a.py", ts=TS, artifact=rel, path=p)
        check("the claim recorded a hash", bool(SL.ledger(p)[0].artifact_sha256))
        check("unchanged artifact passes", SL.check_ledger(p, verbose=False) == [])
        with open(art, "w", encoding="utf-8") as fh:
            fh.write('{"result": 2}')
        errs = SL.check_ledger(p, verbose=False)
        check("a rewritten artifact is caught", any("changed since" in e for e in errs))


def t_artifact_finalization():
    head("7b. an artifact CLAIMED but never hashed is caught -- the fail-open")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "l.jsonl")
        art = os.path.join(d, "truth.npz")
        rel = os.path.relpath(art, _ROOT)

        # The real pattern: claim names the artifact BEFORE it exists, so the
        # claim's own hash is necessarily blank. This used to pass silently.
        SL.claim_range("D18", 30_806_018_400, 10, "lab.py", ts=TS,
                       artifact=rel, path=p)
        check("a claim made before its artifact exists records NO hash",
              SL.ledger(p)[0].artifact_sha256 == "")
        errs = SL.check_ledger(p, verbose=False)
        check("an unhashed artifact is now an ERROR, not silence",
              any("no hash was ever recorded" in e for e in errs), f"{len(errs)} error(s)")

        # L2: finalize refuses to invent provenance for a file that never existed.
        try:
            SL.finalize("lab.py", rel, ts=TS, path=p)
            check("finalize REFUSES an artifact that does not exist", False)
        except FileNotFoundError:
            check("finalize REFUSES an artifact that does not exist", True)

        # Now write it and close the claim.
        with open(art, "wb") as fh:
            fh.write(b"truth-v1")
        SL.finalize("lab.py", rel, ts=TS, path=p)
        check("after finalize the ledger is clean",
              SL.check_ledger(p, verbose=False) == [])
        check("finalize is idempotent",
              SL.finalize("lab.py", rel, ts=TS, path=p) is not None
              and SL.check_ledger(p, verbose=False) == [])

        # L3: a finalize record is NOT a consumption.
        check("finalize does not consume seeds",
              SL.consumed_seeds("D18", p) == 10,
              f"{SL.consumed_seeds('D18', p)} seeds after 1 claim + 2 finalize calls")
        check("finalize does not collide with its own claim",
              SL.collisions(p) == [])
        check("claims() and finalizations() split the ledger",
              len(SL.claims(p)) == 1 and len(SL.finalizations(p)) == 1)

        # The whole point: the artifact cannot be swapped after the fact.
        with open(art, "wb") as fh:
            fh.write(b"truth-v2-SWAPPED")
        errs = SL.check_ledger(p, verbose=False)
        check("an artifact swapped AFTER finalization is caught",
              any("changed since it was finalized" in e for e in errs))
        try:
            SL.finalize("lab.py", rel, ts=TS, path=p)
            check("re-finalizing a swapped artifact is REFUSED", False)
        except PermissionError:
            check("re-finalizing a swapped artifact is REFUSED", True)

        # finalize() closes an existing claim; it does not create provenance.
        try:
            SL.finalize("nobody.py", rel, ts=TS, path=p)
            check("finalize REFUSES an artifact no claim named", False)
        except PermissionError:
            check("finalize REFUSES an artifact no claim named", True)


def t_unbound_claim_binding():
    head("7d. a claim recorded WITHOUT an artifact name can still be closed -- by block, once")
    # `claim_range`'s `artifact` is optional, so a claim can be recorded without
    # naming the file it will produce. P12 was claimed that way on 2026-09-17 and
    # could not be finalized at all: the seeds were spoken for and the truth that
    # came out of them could never be bound to them. That is a hole in the
    # provenance chain, and closing it must not open a route for a stray file to
    # acquire a claimed block's authority.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "l.jsonl")
        art = os.path.join(d, "truth_X.npz")
        rel = os.path.relpath(art, _ROOT)
        with open(art, "wb") as fh:
            fh.write(b"truth-unbound")

        SL.claim_range("D18", 30_806_018_500, 8, "mint.py", ts=TS, path=p)
        check("the claim carries no artifact name", SL.ledger(p)[0].artifact == "")
        try:
            SL.finalize("mint.py", rel, ts=TS, path=p)
            check("without a block the unbound claim still REFUSES", False)
        except PermissionError:
            check("without a block the unbound claim still REFUSES", True)

        digest = SL.finalize("mint.py", rel, ts=TS, block="D18", path=p)
        check("binding by block records the real hash",
              digest == SL._sha256(rel) and len(digest) == 64)
        check("the binding is recorded as such",
              any("bound by block" in f.note for f in SL.finalizations(p)))
        check("after binding the ledger is clean",
              SL.check_ledger(p, verbose=False) == [])
        check("binding does not consume seeds", SL.consumed_seeds("D18", p) == 8)

        art2 = os.path.join(d, "truth_Y.npz")
        with open(art2, "wb") as fh:
            fh.write(b"a different file")
        rel2 = os.path.relpath(art2, _ROOT)
        try:
            SL.finalize("mint.py", rel2, ts=TS, block="D18", path=p)
            check("a SECOND artifact cannot bind to the same block", False)
        except PermissionError:
            check("a SECOND artifact cannot bind to the same block", True)

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "l.jsonl")
        art = os.path.join(d, "truth_Z.npz")
        rel = os.path.relpath(art, _ROOT)
        with open(art, "wb") as fh:
            fh.write(b"z")
        SL.claim_range("D18", 30_806_018_520, 4, "mint.py", ts=TS, path=p)
        SL.claim_range("D18", 30_806_018_530, 4, "mint.py", ts=TS, path=p)
        try:
            SL.finalize("mint.py", rel, ts=TS, block="D18", path=p)
            check("TWO unbound claims in one block is ambiguous and REFUSED", False)
        except PermissionError:
            check("TWO unbound claims in one block is ambiguous and REFUSED", True)

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "l.jsonl")
        rel = os.path.relpath(os.path.join(d, "never_written.npz"), _ROOT)
        SL.claim_range("D18", 30_806_018_540, 4, "mint.py", ts=TS, path=p)
        try:
            SL.finalize("mint.py", rel, ts=TS, block="D18", path=p)
            check("binding by block still REFUSES a file that never existed", False)
        except FileNotFoundError:
            check("binding by block still REFUSES a file that never existed", True)


def t_absent_artifact_is_reported_not_failed():
    head("7c. a gitignored artifact absent from a fresh clone is REPORTED, not failed")
    # *.npz is gitignored, so a fresh clone legitimately has none of them. A gate
    # that goes RED for that is reporting a STATE rather than testing a PROPERTY
    # -- the failure class this repository has lost five checks to. The hash stays
    # recorded and pinned; only the re-check is skipped.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "l.jsonl")
        art = os.path.join(d, "gone.npz")
        rel = os.path.relpath(art, _ROOT)
        with open(art, "wb") as fh:
            fh.write(b"payload")
        SL.claim_range("D18", 30_806_018_420, 5, "lab2.py", ts=TS,
                       artifact=rel, path=p)
        SL.finalize("lab2.py", rel, ts=TS, path=p)
        check("present + finalized is clean", SL.check_ledger(p, verbose=False) == [])
        digest = SL.finalizations(p)[0].artifact_sha256
        os.remove(art)
        check("an ABSENT finalized artifact does not fail the gate",
              SL.check_ledger(p, verbose=False) == [])
        check("and its hash is still pinned in the record",
              SL.finalizations(p)[0].artifact_sha256 == digest and len(digest) == 64)


def t_scan_catches_a_new_literal():
    head("8. the source scan catches the failure that actually happened")
    tmp_dir = os.path.join(_ROOT, "local_sim", "_ledger_scan_tmp")
    # Clear any corpse first. This test writes a real seed literal into the real
    # tree; a hard kill between the write and the `finally` leaves it there, and
    # since 2026-08-20 preflight's ledger step (--scan) runs BEFORE its refusals
    # step -- step 7 before step 9 after the 2026-08-24 renumber -- so the
    # leftover would fail the gate on every subsequent run. The scan's message
    # names the file, so it is self-diagnosing rather than mysterious -- but
    # clearing it here means the next run repairs it without anyone reading that
    # message. The directory is gitignored so it can never be committed.
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    probe = os.path.join(tmp_dir, "sneaky_probe.py")
    try:
        with open(probe, "w", encoding="utf-8") as fh:
            # The literal must sit in a range the ledger does NOT record. D18's
            # first 192 seeds are claimed by the coverage study, and a literal
            # inside a RECORDED range is correctly not flagged — that is the scan
            # working, not failing. 30_806_018_500 is in D18's free tail.
            fh.write("SEED0 = 30_806_018_500  # typed straight in, D16-style\n")
        found = SL.scan_sources(("local_sim",))
        check(
            "a hardcoded seed inside a registered block is flagged",
            any("sneaky_probe.py" in f and "D18" in f for f in found),
            f"{len(found)} finding(s)",
        )
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("from seed_blocks import seeds\nS = seeds('D18')\n")
        found = SL.scan_sources(("local_sim",))
        check(
            "a seeds() call with no literal is ALSO flagged",
            any("sneaky_probe.py" in f and "D18" in f for f in found),
            "the blind spot that hid D12/D14/D15",
        )
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("# seed-ledger: excludes D18\n"
                     "from seed_blocks import seeds\n"
                     "if b in ('D18',): raise SystemExit('refused')\n")
        found = SL.scan_sources(("local_sim",))
        check(
            "an explicit 'excludes' pragma suppresses it (naming a block to refuse it)",
            not any("sneaky_probe.py" in f for f in found),
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    check("the temporary probe is gone", not os.path.exists(tmp_dir))


def _refuses(fn) -> bool:
    """True when `fn` raises. A guard that does not fire is not a guard."""
    try:
        fn()
    except Exception:
        return True
    return False


def t_real_ledger_readonly():
    head("9. the REAL ledger — read-only assertions (predictions A1, A2, A6)")
    before = os.path.getsize(SL.LEDGER_PATH)
    errs = SL.check_ledger(SL.LEDGER_PATH, verbose=False)
    check("the committed ledger is consistent", errs == [], f"{len(errs)} error(s)")
    check("allocations are still disjoint (invariant 12)", check_disjoint())

    # The two registries are separate files and drifted: CONTRACT WP15 added
    # P05/P06 to seed_blocks.py and not to seed_ledger.TIERS, so block_span()
    # worked and tier_of() raised -- WP15's first claim would have died. Assert
    # agreement in BOTH directions so the drift cannot recur silently.
    import seed_blocks as SB
    # `_all()` is the registry: VAULT + LEGACY + DECISION. Read it directly and
    # let an AttributeError be loud -- a fallback to `SL.TIERS` here would make
    # both checks below compare a set with itself and pass vacuously, which is
    # the failure class this file exists to catch.
    reg = {b[0] for b in SB._all()}
    legacy = {b[0] for b in SB.LEGACY_BLOCKS}
    check("the block registry is non-empty and was really read", len(reg) > 20,
          f"{len(reg)} blocks")
    # Legacy blocks predate the ledger and are deliberately untiered.
    reg = reg - legacy
    check(
        "every registered block has a tier in seed_ledger.TIERS",
        not (reg - set(SL.TIERS)),
        f"missing: {sorted(reg - set(SL.TIERS))}",
    )
    check(
        "and every tier names a registered block",
        not (set(SL.TIERS) - reg),
        f"orphaned: {sorted(set(SL.TIERS) - reg)}",
    )
    check(
        "a tier lookup succeeds for every registered block",
        all(SL.tier_of(b) for b in sorted(reg)),
    )

    cols = SL.collisions()
    # An overlap is UNDECLARED when neither claim named the other at claim time
    # and no later `pair` record declares it. Counting undeclared overlaps was
    # the original assertion here and it was a SNAPSHOT: the record legitimately
    # grew a second one (P01, WP14 C0-i) whose intent lived only in a prose note,
    # and preflight steps 7 and 9 went RED for four commits with nothing wrong
    # with the ledger except an unset field. The test now asserts the PROPERTY --
    # every undeclared overlap is on a named list, with its reason -- so a new
    # accident still fires and a newly DECLARED overlap does not.
    accidental = [(a, b) for a, b in cols if not SL.is_paired(a, b)]
    KNOWN_UNDECLARED = {
        ("D16", frozenset({"local_sim/probe_whitening.py",
                           "local_sim/probe_design_carrier.py"})):
            "2026-08-10, found 2026-08-19: two probes drew the same 24 D16 "
            "networks by accident. Both kills are CORRELATED evidence.",
    }
    unknown = [(a, b) for a, b in accidental
               if (a.block, frozenset({a.consumer, b.consumer}))
               not in KNOWN_UNDECLARED]
    check(
        "A1: every UNDECLARED overlap is one of the known accidents",
        not unknown,
        "; ".join(f"{a.block} {a.consumer} vs {b.consumer}" for a, b in unknown),
    )
    check(
        "A1: and the D16 accident is still in the record, not quietly resolved",
        any(a.block == "D16"
            and {a.consumer, b.consumer}
            == {"local_sim/probe_whitening.py", "local_sim/probe_design_carrier.py"}
            for a, b in accidental),
        f"{len(accidental)} undeclared overlap(s)",
    )
    check(
        "A1: the P01 WP13/WP14 overlap is DECLARED, and its reason still warns "
        "that the evidence is correlated",
        any(a.block == "P01"
            and "CORRELATED" in SL.pairing_reason(a, b).upper()
            for a, b in cols
            if {a.consumer, b.consumer}
            == {"local_sim/p2_slice_lab.py::WP13 E0-TOL/E1/E2",
                "local_sim/p2_truth.py"}),
    )
    check(
        "A1: a pairing cannot be declared for an overlap that does not exist",
        # The start is READ from the registry, never written as a literal: a
        # seed-shaped constant in this file is exactly what the source scan is
        # for, and widening the exclusion pragma to silence it would blunt the
        # scan to make a test pass.
        _refuses(lambda: SL.declare_pair(
            "P03", SL.block_span("P03")[0], 4,
            ["local_sim/p2_truth.py", "local_sim/p2_slice_lab.py"],
            note="fabricated", ts="2026-09-02", path=SL.LEDGER_PATH)),
    )
    # D17 and D19 are SINGLE_USE and currently unspent. Asserting "zero
    # consumption" is a SNAPSHOT, and D19 exists precisely in order to be spent
    # once -- on the v2.6 promotion gate. Since 2026-08-20 this file runs inside
    # the mandatory preflight gate, so that snapshot would have turned preflight
    # RED at the exact moment the block was used as designed, and the only way
    # forward would have been to edit the gate. That is the failure mode
    # docs/GOTCHAS.md records four times: an assertion written against a moment
    # rather than against a property. The durable property of a SINGLE_USE block
    # is that it is claimed AT MOST ONCE, ever.
    for blk in ("D17", "D19"):
        claims = SL.consumed_ranges(blk)
        lo, hi = SL.block_span(blk)
        check(
            f"A2: {blk} is SINGLE_USE and has at most one recorded claim",
            len(claims) <= 1,
            f"{len(claims)} claim(s), {SL.consumed_seeds(blk)} seeds"
            + ("  <- UNSPENT" if not claims else
               f"  <- SPENT by {claims[0][2]}"),
        )
        check(
            f"A2: any {blk} claim names a real consumer inside the block",
            all(c and lo <= a and b <= hi for a, b, c in claims),
            "vacuously true while unspent" if not claims else f"{claims}",
        )
    # D18 is a LIVE design block and grows as experiments claim from it. Two
    # earlier versions of this check asserted a SNAPSHOT ("zero consumption",
    # then "exactly 192 by coverage_lab") and both went stale within hours. A
    # test of a growing ledger must assert INVARIANTS, not a moment in time.
    d18 = SL.consumed_ranges("D18")
    lo, hi = SL.block_span("D18")
    check(
        "D18 accounting closes: consumed + free == the block",
        SL.consumed_seeds("D18") + sum(e - s + 1 for s, e in SL.free_ranges("D18"))
        == hi - lo + 1,
        f"{SL.consumed_seeds('D18')} consumed, "
        f"{sum(e - s + 1 for s, e in SL.free_ranges('D18'))} free of {hi - lo + 1}",
    )
    check(
        "every D18 claim names a real consumer inside the block",
        all(c and lo <= a and b <= hi for a, b, c in d18) and len(d18) >= 1,
        f"{len(d18)} claim(s): {[c for _a, _b, c in d18]}",
    )
    # D16 is a LIVE, part-spent DESIGN block with 330 seeds still free, so its
    # consumed count and its gap list are both moving targets -- the same
    # snapshot trap as D17/D19 above. Assert the accounting instead.
    d16 = SL.consumed_ranges("D16")
    lo16, hi16 = SL.block_span("D16")
    check(
        "A6: D16 accounting closes: consumed + free == the block",
        SL.consumed_seeds("D16")
        + sum(e - s + 1 for s, e in SL.free_ranges("D16")) == hi16 - lo16 + 1,
        f"{SL.consumed_seeds('D16')} consumed, "
        f"{sum(e - s + 1 for s, e in SL.free_ranges('D16'))} free in "
        f"{len(SL.free_ranges('D16'))} gaps of {hi16 - lo16 + 1}",
    )
    check(
        "A6: every D16 claim names a real consumer inside the block",
        all(c and lo16 <= a and b <= hi16 for a, b, c in d16) and len(d16) >= 1,
        f"{len(d16)} claim(s)",
    )
    check(
        "A6: free ranges are disjoint, ascending, and inside the block",
        all(s <= e for s, e in SL.free_ranges("D16"))
        and all(SL.free_ranges("D16")[i][1] < SL.free_ranges("D16")[i + 1][0]
                for i in range(len(SL.free_ranges("D16")) - 1))
        and all(lo16 <= s and e <= hi16 for s, e in SL.free_ranges("D16")),
    )
    check("A3: the tree has no unrecorded seed use", SL.scan_sources() == [])
    # Every claim that names an artifact must have a hash from somewhere -- the
    # fail-open this suite exists to keep closed. Asserted as a PROPERTY over the
    # live ledger, so it holds for records that do not exist yet.
    unhashed = [
        c.consumer for c in SL.claims()
        if c.artifact and not c.artifact_sha256
        and not any(q.consumer == c.consumer and q.artifact == c.artifact
                    and q.artifact_sha256 for q in SL.finalizations())
    ]
    check("A7: every artifact-naming claim in the REAL ledger is hashed",
          unhashed == [], f"unhashed: {unhashed}")
    check("A7: D17 and D19 are still unspent after the finalize change",
          SL.consumed_seeds("D17") == 0 and SL.consumed_seeds("D19") == 0,
          f"D17 {SL.consumed_seeds('D17')}, D19 {SL.consumed_seeds('D19')}")
    check("no test wrote to the real ledger", os.path.getsize(SL.LEDGER_PATH) == before)


def main():
    print("=" * 104)
    print("SEED LEDGER — does the enforcement bite?")
    print("=" * 104)
    for fn in (
        t_single_use,
        t_design_overlap_allowed_but_reported,
        t_bounds_and_unknown_blocks,
        t_exhaustion_and_free_ranges,
        t_ledger_tamper,
        t_hard_collision_on_decision,
        t_artifact_hash,
        t_artifact_finalization,
        t_unbound_claim_binding,
        t_absent_artifact_is_reported_not_failed,
        t_scan_catches_a_new_literal,
        t_real_ledger_readonly,
    ):
        fn()
    n_ok = sum(1 for r in RESULTS if r)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
