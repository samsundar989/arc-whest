# Fresh-context Skeptic verdict — selector provenance repair

**Verdict: ACCEPT.**

The production producer no longer imports the selector before binding it. It
locks `p2_k4_fit.py`, reads one byte snapshot, compiles and executes a private
module from that buffer, hashes the same buffer, and holds the lock through
receipt publication. The verifier requires that producer-time selector binding
to equal its current `fit_interface` bytes before truth access.

The Skeptic independently checked the implementation, raw 97/97 suite and
workerq 893. Job 893 ran current live bytes at exact code commit `0b93512`, passed
default preflight 12/16 with four Full-only steps skipped, and peaked at 2,966.6
MiB under its 5 GiB declaration. Candidate math, registries, lifecycle, seeds,
estimator files and data roles did not change. No protected, public or vault
labels were opened.

The strongest limitation is that most programmatic fixtures use the explicit
test-only `_SELECTOR_OVERRIDE`; the direct marker fixture and documented producer
subprocess exercise the locked production loader. Repository search found no
non-test programmatic caller. POSIX locking is advisory, though exact-snapshot
execution and closing hashes still catch persistent substitution. The real P07
batch remains unrun, and Full-only preflight steps remain outside this repair gate.

The cheapest optional follow-up is another truth-free subprocess fixture that
opens its positive receipt, asserts the private loader's selector binding, and
runs standalone verification. The Skeptic did not require that follow-up before
the registered P07 producer.
