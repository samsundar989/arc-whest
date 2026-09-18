# Independent Skeptic: snapshot-root reader repair

**Verdict: ACCEPT for the narrow instrument repair.** Reviewed contract
commit `27c815e`, the reader/test diff, and `design_sidecar.verify`. The
focused synthetic test passed: a valid fixture and 17 existing provenance
tamper refusals, plus snapshot-relative external output resolution, wrong-path
refusal, and tampered snapshot source refusal.

**Strongest challenge:** the new test mocks `ds.verify`, so the test itself
does not demonstrate truth and dependency verification. Source inspection of
`design_sidecar.py` lines 259-281 shows that the supplied root is used for
source, truth, and dependency hashes. The reader forwards that root and also
uses it for registered role paths and the recorded sidecar path.

**Scope:** no real P12 reader execution or scored-row inspection was part of
this review. Full verification of the saved artifacts belongs to the
registered next read.
