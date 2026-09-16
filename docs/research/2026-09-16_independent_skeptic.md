# Independent Skeptic — 16 September 2026 strategy audit

The reviewer was launched in a fresh context without Builder history or the three new audit narratives. Inputs were `docs/SKEPTIC.md`, the binding validation/current-state/gotchas documents, the September 16 contracts, raw v31 and public v17/v29 source, `feed_already_present.md`, stored graded reports and validation source. No experiments, label reads, seed claims, edits or heavy execution were performed. This file records its returned findings.

## Skeptic verdict: PROVISIONAL

**Strongest challenge:** matching repeated-index slices does not establish complete tensor feedback; N1's rank test does not establish its total computational economics or the relevance of its weighting metric.

**Evidence for the challenge:**

1. `estimator_k3_v31.py:1080–1103` builds repeated-support residuals while the hub at line 1035 lacks explicit κ4 feedback. `external/504aldo/estimators/estimator_v17.py:585–598` adds a κ4 hub whose off-diagonal part produces all-distinct entries. Lines 643–646 subtract its repeated slices before residual construction; lines 735–753 consume its transported content. The earlier feed note did not derive this mapping. Adding an uncompensated hub double-counts; compensated insertion is coherent.
2. D3 as written tests the repeated-slice route. Neither a large nor a small response by itself determines whether the distinct-index route exists or whether a coefficient is wrong. One dense transport exposes the difference.
3. A general dense Tucker readout costs approximately `2*n*r^3 + 2*n^2*r`, excluding joining, basis construction, rebasing and transport. At width 1024, rank 192 is about 14.9 billion ordinary FLOPs per readout; rank 384 about 116.8 billion. A generic three-mode core rotation adds about `6*r^4`. These are algebraic counts, not measured flopscope prices. v29 retains per-source factor stacks and reconstructs dense old legs; it does not demonstrate the proposed single-core economics.
4. `J=product(DW)` describes a frozen mean-path perturbation. It omits the injection of K3 perturbations into covariance, moments and later births. Weighting a Gram by `J.T@J` needs a precise objective/projector; a generic product is not symmetric.
5. `predict_public.py` does not enforce the local-control provenance check promised by its prose. It transforms a paired-difference interval into a band without full ratio/transfer uncertainty. Its candidate SHA is not tied to the run that produced rows. Average operation counts do not certify adaptive per-MLP behavior. Its old offset and timing bands are stale. Candidate smoke still enables the invalid local residual gate before the advisory check.

**Evidence for the original direction:** raw stored reports show F21 at adjusted 1.59573e-8 and v31 at 8.75203e-9, both 0/100 failures. Worst graded wall/residual are 93.16 seconds/0.15623 seconds for F21 and 68.13 seconds/0.14024 seconds for v31. The v31 projection missed by about 0.45%. Public truth-noise floors are not established by those summaries. This supports actual progress and one successful forecast, not a universal transfer guarantee.

**Unresolved ambiguity:** the missing representational component's score effect is unknown. Sufficiently low-rank merging might still meet 0.20B, but the current price claim does not demonstrate it. No winning implementation is established.

**Cheapest discriminating follow-up:** one label-free tiny float64 birth/transport identity test with identical incoming state and nonzero off-diagonal κ4. Match repeated slices through compensation, inspect distinct entries, then mix with an arbitrary dense matrix and compare consumed slices. No seed block or truth labels are required. Price a complete N1 update/join/readout before its P11 accuracy screen. P11 is DESIGN; fitted choices need the separately required fresh decision evidence.

**Checklist items triggered:** 1, 4, 6, 8, 9, 11, 14, 17, 18.
