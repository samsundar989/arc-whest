# E14 tree candidate tiny evidence provenance

The first `tiny_results.json` was generated from the provisional source with
SHA256 `57618eafd0e2d5aafdaf93da4967466079708c2caf3d72df7d0a4af0632cc806`.
That source incorrectly used `Cout` for both the connected-tree H and the
projection. The subsequent corrected-source run overwrote that JSON. Its exact
bytes were not separately saved; this is an evidence-retention gap. The defect
and revoked hash were reported to the root agent before any heavy job.

The corrected source separated tree `Cpre` from projection `Cout`; later source
edits added debug capture without changing active arithmetic. The current
`tiny_results_v3.json` preserves the rerun and the first large drift operands.
`layer_oracle_v1.json` is a separate independent dense Q/manifest/centering/K3
check on that source. Neither file contains truth labels or a score outcome.

At n=4, depth=3, layer 1, the active covariance diagonal contains
`-1.176324725151062`, while frozen F21 on the same tiny weights contains
`-4.18978214263916`. Both are inadmissible as covariance matrices on this
fixture; the active negative value alone does not isolate a new regression.
The active mean difference at that layer is `0.8563736081123352` maximum.

During the second Skeptic follow-up, `layer_oracle_v2.json` first held the
negative-fixture-only two-layer audit. The subsequent admissible-controlled
extension reused that filename and overwrote its exact bytes before a separate
copy was made. That intermediate version is also an evidence-retention gap.
The current eight-case output is copied to `layer_oracle_v3.json`, and the
oracle now writes v3. `layer_oracle_v1.json` remains intact from the first
Skeptic review. No result from an overwritten version is used as a gate.

The Gaussian-start oracle added a distinct revision chain. `full_oracle_v1.json`
is its first complete run; an earlier incomplete exception receipt at that same
path was overwritten during local debugging. `full_oracle_v2.json` used original
float64 generated weights while the active path consumed their float32 cast;
the resulting first reported discrepancy was an input-representation mismatch.
`full_oracle_v3.json` corrected the shared input bytes and found a later
2e-5-bound failure, but its short `first_falsifier` summary still selected
layer 0. `full_oracle_v4.json` retains the exact first bound-crossing layer and
full operands. Prior versions are preserved for audit; v4 is the conformance
verdict. No acceptance threshold was changed.
