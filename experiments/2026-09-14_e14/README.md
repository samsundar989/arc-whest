# E14 execution evidence

**No new candidate earned promotion. F21 remains the strongest validated
implementation.** Instrument repairs and mathematical diagnostics are distinct
from a validated algorithmic improvement or release clearance.

Start with [stage status](../../docs/E14_EXECUTION_STATUS.md), the
[release evidence index](f21_release_record_v3.json), and the bounded
[strategist packet](../../docs/research/2026-09-14_e14_strategist_packet.md).
Contracts and knowledge deltas are committed at the bottom of LOG.md.

## Candidate and release recommendation

Keep the existing frozen F21 candidate and its already graded #330950 as the
competitive reference. Do not replace root estimator.py with an unvalidated
compact or K4 prototype, and do not resubmit unchanged F21 solely because
these instrument repairs passed.

- Source: `estimator_k3_strassen_L4_f21.py`, SHA256
  `d77dab0e2c740cc9f6cd7d9172adacc8da731cc5b40ca87798f3fb006e475a94`.
- Reusable single-file package with license/manifest:
  `experiments/2026-09-12_f21/f21-with-notice.tar.gz`, SHA256
  `4f3cc6f440e4cb66ea86a0904c00e182eb7ecf5aaab0ee64c73efb7d22904775`.
  Its estimator member hash equals the candidate hash. The new evidence-index
  consumer verifies the archive inventory and every member's size/hash.
- Counted official cost: **928,457,637,478 FLOPs per MLP**. This is existing
  measured evidence; no compression cost saving was measured in E14.
- Existing measured server run: 100/100 completed, residual headroom **2.5603**
  against 0.4 seconds, wall headroom **1.2881** against 120 seconds. These apply
  to that run only. Exact server-byte attribution is provisional.
- Existing applicable Linux memory evidence: VmPeak **5,381.7 MiB** under
  **8,192 MiB**, margin **2,810.3 MiB**; settled address-space growth 39.2 MiB.
  The index reuses the recorded bound test; E14 did not run a new Linux cap.
- **LOCAL remains REFUSED.** The existing capped mini failed 100/100 on local
  residual exhaustion. The one-time prior submission exception does not extend
  to future entries. `normal_release_ready` remains false.

No submission approval is requested: there is no newly improved release-ready
candidate. No submission, designation, vault evaluation or external message
was performed. Exact-byte attribution and contribution-prize eligibility
remain open questions, not conditions this engineering work resolves.

## Evidence disposition

| Work | Result and scope | Raw/review evidence |
|---|---|---|
| Candidate identity and release index | Explicit candidate/package identity; local/forecast/measured domains remain separate; semantic mutations refuse | release_tests_v5.txt, release_supplement_skeptic.md, f21_release_recipe_v3.json |
| Candidate numerical sensitivity | Three actual 1024x16 F21 permutations pass at raw-error scale 2e-8 and drift bar 2e-10 | f21_numerical/, workerq_1217_receipt.json |
| Crash-safe row evidence | Complete rows survive crashes; incomplete/failure/malformed/duplicate evidence cannot promote; no new protected read enabled | journal_artifacts_final/, journal_tests_skeptic_fix.log, infrastructure_skeptic.md |
| Production-truth arithmetic | Explicit math mode and provenance; shared-input production-shape pilot passes two declared chunk comparisons | truth_cpu_tests.txt, truth_pilot.json, truth_evidence.md, workerq_1219_receipt.json |
| Float64 full-rank B4 | 841/841 pass; worst normalized discrepancy 9.7388e-13 under unchanged 1e-5 | b4_f64_v2.json, b4_f64_skeptic_final.md |
| Actual-f32 compact interpreter | Output conformance refused in 3/4 fixed cases; original failures preserved | b4_real_backend/, b4_execution_report.md |
| Mixed-f64 representation control | Zero final-output drift in four fixed reference-order cases; exact source/zero recovery still fails; dense wrapper saves no work | b4_mixed_qr/, b4_mixed_qr_skeptic.md |
| Faithful compression cost screen | Dense-decoding bridge cannot save cost; ordinary-f64 residual/query subset alone exceeds20%-saving ceiling; no rank measurement | ../../docs/research/2026-09-14_e14_compression_cost.md |
| Additional coherent K4 | Symbolic definition accepted; literal dense implementation exceeds budget; stability and accuracy untested | accuracy_skeptic_final.md, linked research derivations |

Initial shim runs, serialization failures, rejected journal/receipt variants
and failed snapshot preflight remain present. The authoritative installed-backend
B4 results are explicitly named v2 / b4_real_backend. Synthetic `mse` fields in
journal fixtures are test values, not estimator accuracy measurements.

## Fresh score evidence and truth noise

**No E14 fresh paired score experiment occurred.** No block was claimed, and
P10's spent protected truth was not read. The actual-precision/cost prerequisites
prevented a frozen new candidate from reaching a justified decision experiment.
No promotion gain, private-network result or winning algorithm is claimed.

For continuity only, the already accepted P10 comparison used 128 paired
networks: F21 raw MSE 4.1821088467e-8 versus L4 4.1821425170e-8, beside truth
midpoint noise floor 2.2363657945e-9. The existing official-precision forecast
has arithmetic L4-minus-F21 score 95% t interval
[2.4582563842e-9, 2.5842673161e-9], gain 1.150566x at gate 1.05. These are
historical accepted results, not an E14 reanalysis or cap pass. Public-label
truth noise remains unavailable.

E14 permutation drift and B4 discrepancy measure numerical differences without
labels. The truth pilot bounds arithmetic perturbation at an assumed error
scale; it does not measure candidate MSE or a truth-noise floor. Its conditional
score-error bounds 6.4183421023e-12 and 4.5053995651e-12 satisfy the declared
1e-10 bar on one fixed shared-input fixture. No population claim follows.

## Resource and verification scope

All heavy work used workerq with committed optimization reviews. Numerical
job1217 reserved 8 GiB RAM / 0 VRAM / 2 CPUs; peak host 3,738 MiB. Truth job1219
reserved 8 GiB RAM / 6 VRAM / 1 CPU; peaks host 4,557 MiB / VRAM 2,462 MiB.
Default preflight job1222 reserved 6 GiB RAM / 0 VRAM / 2 CPUs; peak host
2,077 MiB, passed 12 enabled steps with 4 intentionally skipped and unchanged
source bytes. Job1230 covers the later consumer/mixed-control files; its final
receipt is the final affected-gate authority: **passed12/16,4skipped**, all309
source hashes unchanged; peak host2,375MiB within the6GiB reservation. Default preflight is not candidate
Full certification. No local host seconds are presented as server forecasts.

The contribution write-up remains **stale / do not submit**. F21 #330950 is a
successfully graded anchor, but the new derivations are not demonstrated
accuracy improvements or publishable novelty claims. A strategic claim review,
source attribution and eligibility resolution are still required.

## Exact-byte preservation and reproducibility

The accompanying E14 evidence capsule preserves the exact current bytes of the
raw records and source instruments with a per-member SHA256 inventory. Git text
normalization changes some historical CRLF bytes on checkout; the capsule avoids
silently attributing a normalized file to a raw-file hash. It contains source and
unprotected diagnostics/reports, not challenge truth arrays or a new submission.
The existing F21 tarball remains the candidate package. Reproducing an experiment
that accesses protected truth still requires the repository's separate protocol;
the capsule grants no additional data access or resumption.

## Commit account and final review

Starting commit was4e0a987 with a clean tree. No unrelated changes existed.
The substantive local history through the source freeze is:

```text
68cac46 Preregister E14 important implementation prerequisites
70819fc Declare E14 production precision conformance regime
791fcea Add pinned candidate evidence index and explicit numerical sensitivity
88d6002 Preregister common-input production truth arithmetic pilot
b8f8062 Pin production truth arithmetic and preserve runtime provenance
c2bcfba Declare fullrank precision error localization controls
fd3bd0f Journal scorer rows durably and preserve separate release domains
7491d72 Preserve initial B4 float64 parity diagnostic and raw operands
2c6c503 Preserve initial f32 shim conformance failures before backend correction
456c61a Reject malformed journal evidence and enforce read-cap ordering
ce680a3 Run queued preflight with exact live bytes and source freeze checks
832c0d6 Record independent repairs review and targeted QR stabilization prediction
046fa94 Verify B4 on installed backend and isolate precision failures
8f600fb Declare bounded faithful compression cost bridge prediction
daed828 Bind numerical receipts strictly and preserve reviewed E14 validation evidence
```

The subsequent documentation/evidence and capsule commits record the final
accepted state. [Final independent Skeptic](final_skeptic.md): **ACCEPT**,
restricted to these instrument results and two implementation cost rejections.
It grants no algorithmic promotion or release clearance. All source changes
are covered by the final309-file freeze; later edits are evidence/documentation.
