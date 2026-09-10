"""Backfill P08's three historical reads into the new register.

They are recorded as BACKFILLED and as HISTORICAL FACT. They are not fresh
reads and they are not permission for another one: the point of writing them
down is that P08's cap reads as spent to anything that asks, instead of the
prose in seed_blocks.py that nothing could enforce.
"""
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "local_sim"))
import truth_reads as tr  # noqa: E402

ART = "experiments/2026-09-10_slice5/truth_P08.npz"

HISTORY = [
    dict(consumer="local_sim/p2_corrector.py (broken instrument, run 1)",
         purpose="CONTRACT T2 held-out gain -- the first of two runs whose "
                 "`gain` was computed on P08 by an instrument later repaired",
         note="Found by the T2 Skeptic on 2026-09-10, after the fact. This "
              "read was not recorded anywhere at the time."),
    dict(consumer="local_sim/p2_corrector.py (broken instrument, run 2)",
         purpose="CONTRACT T2 held-out gain -- the second broken run",
         note="Second uncounted look. The LOG entry that said 'single held-out "
              "read' was written in good faith and has been struck."),
    dict(consumer="local_sim/p2_corrector.py (repaired instrument)",
         purpose="CONTRACT T2 held-out gain -- the run that produced the "
                 "1.0163x figure",
         note="The winning instrument was committed in the SAME commit as this "
              "result, so no committed state of it predates its P08 read. That "
              "is why the 1.0163x stands as a kill and not as a promotion."),
]


def main():
    existing = [r for r in tr.reads() if r.block == "P08"]
    if existing:
        print(f"P08 already has {len(existing)} lines; refusing to double-write.")
        return 1
    # Written directly, NOT through reserve(): reserve() would refuse the second
    # and third, which is correct behaviour for a fresh read and wrong for a
    # historical record. Backfill is the one path that bypasses the cap, it is
    # marked as such on every line, and it exists once.
    for i, h in enumerate(HISTORY, 1):
        rec = tr.Read(
            block="P08", artifact=ART,
            artifact_sha256=tr._sha256_file(ART),
            consumer=h["consumer"],
            instrument_sha256="",
            purpose=h["purpose"],
            ts="2026-09-10",
            status="completed",
            reservation=f"backfill-{i}",
            result_artifact="",
            note=h["note"],
            backfilled=True,
        )
        tr._append(rec)
    st = tr.status("P08")
    print(f"P08  tier {st['tier']}  cap {st['cap']}  spent {st['spent']}  "
          f"exhausted {st['exhausted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
