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
for D17/D18/D19 and exactly 270 for D16, which would fail if this file ever did
mint from them.
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


def t_scan_catches_a_new_literal():
    head("8. the source scan catches the failure that actually happened")
    tmp_dir = os.path.join(_ROOT, "local_sim", "_ledger_scan_tmp")
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


def t_real_ledger_readonly():
    head("9. the REAL ledger — read-only assertions (predictions A1, A2, A6)")
    before = os.path.getsize(SL.LEDGER_PATH)
    errs = SL.check_ledger(SL.LEDGER_PATH, verbose=False)
    check("the committed ledger is consistent", errs == [], f"{len(errs)} error(s)")
    check("allocations are still disjoint (invariant 12)", check_disjoint())

    cols = SL.collisions()
    accidental = [
        (a, b) for a, b in cols
        if not ((b.consumer in a.paired_with) or (a.consumer in b.paired_with))
    ]
    check(
        "A1: exactly ONE accidental collision in the whole record",
        len(accidental) == 1,
        "; ".join(f"{a.block} {a.consumer} vs {b.consumer}" for a, b in accidental),
    )
    check(
        "A1: and it is probe_whitening vs probe_design_carrier on D16",
        accidental
        and accidental[0][0].block == "D16"
        and {accidental[0][0].consumer, accidental[0][1].consumer}
        == {"local_sim/probe_whitening.py", "local_sim/probe_design_carrier.py"},
    )
    check("A2: D17 has zero consumption", SL.consumed_seeds("D17") == 0)
    check("A2: D19 (the reserved single-use gate block) has zero consumption",
          SL.consumed_seeds("D19") == 0)
    # D18 was clean when A2 was written; the interval-coverage study has since
    # claimed its first 192 seeds through claim(). Assert the claim is exactly
    # what that study says it took, rather than that the block is untouched.
    d18 = SL.consumed_ranges("D18")
    check(
        "D18 shows exactly one claim, 192 seeds, by coverage_lab",
        SL.consumed_seeds("D18") == 192 and len(d18) == 1
        and d18[0][2] == "local_sim/coverage_lab.py",
        f"{SL.consumed_seeds('D18')} consumed by {[c for _s, _e, c in d18]}",
    )
    check(
        "A6: D16 shows 270 distinct seeds consumed, 330 free",
        SL.consumed_seeds("D16") == 270,
        f"{SL.consumed_seeds('D16')} consumed, "
        f"{600 - SL.consumed_seeds('D16')} free in {len(SL.free_ranges('D16'))} gaps",
    )
    check(
        "A6: the four free gaps are exactly the ones predicted",
        SL.free_ranges("D16") == [
            (30_806_016_224, 30_806_016_299),
            (30_806_016_320, 30_806_016_399),
            (30_806_016_420, 30_806_016_499),
            (30_806_016_506, 30_806_016_599),
        ],
    )
    check("A3: the tree has no unrecorded seed use", SL.scan_sources() == [])
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
        t_scan_catches_a_new_literal,
        t_real_ledger_readonly,
    ):
        fn()
    n_ok = sum(1 for r in RESULTS if r)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
