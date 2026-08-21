"""Close the two D18 truth artifacts that were claimed but never hashed.

P2-CP0.5. `claim()` records an artifact PATH before the artifact exists, so its
`artifact_sha256` is necessarily blank at claim time -- and nothing ever filled
it in. `check_ledger` validated a hash only `if c.artifact and c.artifact_sha256`,
so a blank one passed silently: a truth set could be swapped after the arms were
scored against it and the ledger would still validate.

`seed_ledger.finalize()` closes that. This script is the one-off backfill for the
two records that predate it. It is idempotent -- a second run finalizes nothing
and refuses if a hash ever disagrees.

The timestamp is TODAY, deliberately, and the note says so: these hashes were
taken on 2026-08-21, not on 2026-08-19 when the artifacts were written. Recording
the claim date would assert a provenance chain that does not exist.

Run:  uv run python experiments/2026-08-21_seed_ledger_finalize.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "local_sim"))

import seed_ledger as sl  # noqa: E402

TS = "2026-08-21"
NOTE = ("backfilled 2026-08-21 (P2-CP0.5). The hash was taken on this date, not "
        "when the artifact was written on 2026-08-19; nothing certifies the file "
        "was unchanged in between. It is pinned from here on.")

TARGETS = [
    ("local_sim/coverage_lab.py",
     "experiments/2026-08-19_coverage/truth_D18.npz"),
    ("local_sim/probe_oa_carrier.py",
     "experiments/2026-08-19_carriers/oa_truth_D18.npz"),
]


def main() -> int:
    for consumer, artifact in TARGETS:
        digest = sl.finalize(consumer, artifact, ts=TS, note=NOTE)
        print(f"  finalized  {consumer:<34} {artifact}")
        print(f"             sha256 {digest}")
    errors = sl.check_ledger(verbose=True)
    print()
    if errors:
        for e in errors:
            print(f"  ERROR {e}")
        return 1
    print("  ledger clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
