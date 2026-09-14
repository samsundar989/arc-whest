# E14 B4 precision execution and bounded localization

Contracts: initial IMPORTANT prediction `68cac46`; actual-f32 regime `70819fc`;
localization prediction `c2bcfba`. No labels, seed claim/read, vault operation,
rank measurement, candidate edit, submission or promotion occurred.

## Findings

The real-package float64 diagnostic passes all **841** original checkpoints at
`1e-5`, worst normalized difference `9.738773765525043e-13`.
`b4_f64_v2.json` records the exact source/dependency hashes and installed
`flopscope`/`fnp` import paths. It supports precision as an explanation of the
historical mismatch, not production numerical conformance.

Actual-f32 fullrank conformance is **REFUSED**. The preregistered final-output
squared-drift allowance is `2e-10`, unchanged. Both arms use identical
float32-quantized weights from the existing B4 or B3/B1 fixed fixture.

| Fixed fixture / reference | Real-f32 output squared drift | Original 1e-5 failures | Declared field-envelope failures | Final-output verdict |
|---|---:|---:|---:|---|
| B4 n8/d6 / L2 | 7.260882557602599e-5 | 66 | 0 | REFUSED |
| B4 n8/d6 / forced F21 | 2.306408673473925 | 210 | 0 | REFUSED |
| B3 n4/d4 / L2 | 6.274092356761685e-12 | 0 | 0 | PASS |
| B3 n4/d4 / forced F21 | 4.2012704426497294e-10 | 8 | 0 | REFUSED |

Raw records: `b4_real_backend/b4_f32_summary.json` and its four case artifacts.
All captured compact arrays are verified float32. The envelope remains
`max_abs(compact32-reference32) <= 4*max_abs(reference32-reference64) +
64*eps32*max(max_abs(reference64),max_abs(reference32))`, with exact equality at
zero scale. Its pass does not override the original tolerance or output failure.
The identity output check passes, and a `4e-10` injected squared drift fails.
These are prediction arithmetic differences, not errors against truth; ground
truth and its noise floor are not involved.

## Where the numerical difference enters

The unchanged reference `_layer` was executed after fullrank identity or QR
encoding/decoding, preserving its operation order. The independent one-step
operator was also replayed from the exact reference input state at every layer.

| Fixture / reference | Identity, reference order | QR, reference order | Same-input independent final step |
|---|---:|---:|---:|
| B4 / L2 | 0 | 4.190951585769653e-9 | 0 |
| B4 / forced F21 | 0 | 0.0360966749212821 | 4.9072034016717225e-5 |
| B3 / L2 | 0 | 0 | 0 |
| B3 / forced F21 | 0 | 1.5910348594161405e-9 | 9.689360425113591e-12 |

Every identity/reference-order control is bit-identical across all observed
Wick, preactivation, cumulant and centered-moment checkpoints. The QR map uses
one complete weighted-Gram QR and direct fullrank reprojection per layer. It
isolates QR roundtrip sensitivity, **not** the exact cached-factor-rotation order
of the autonomous compact implementation. The frozen youngest source retains
its symbolic identity P; its reconstructed P is saved but is not substituted
for that symbolic identity in decoding.

On B4/L2, same-input differences first appear in layer0 Wick terms `(0,3)` and
`(4,1)` at about `5.96e-8`, followed by contraction and birth rounding. The
QR/reference-order control first differs at layer1 D21 by `3.73e-9`. Small
perturbations amplify strongly in this fixed recurrence. In the forced F21
comparison, the first same-input discrepancy is already layer0 covariance
(`9.30e-6` maximum normalized difference): the independent operator uses plain
matmul while the forced reference uses Strassen. Thus QR perturbation and
arithmetic reassociation are both live causes; the controls do not uniquely
allocate the autonomous failure to either one.

`b4_real_backend/b4_localization_summary.json` links every raw checkpoint and
operand. F21 forcing changes only `_STRASSEN_MIN_BLOCK=1` in a separately loaded
module; frozen bytes are unchanged. The fixtures execute at most three (n8) or
two (n4) Strassen levels, so none of these records certifies production L4 at
1024x16.

## Reference-order identity bridge and cost implications

Using the existing F21 nonlinear/operator calls after an exact fullrank identity
encoding is faithful, and these controls verify it. It is a **dense wrapper**,
not a nontrivial compressed fullrank algorithm or a cure for the failed QR path.
The operator receives the reconstructed wide A_D, A_L and M tower; all ages and
all baseline transports/contractions remain present. Reducing rank afterward
would introduce a new approximation and still cannot bypass the failed fullrank
QR prerequisite.

The literal positive-control implementation performs seven extra n-by-n dense
matmuls per retained age per roundtrip: two identity-basis projections, two
carried-Gram products, two leg reconstructions and one residual reconstruction.
It additionally copies/concatenates all dense source payloads and carries Q,
Gram and source factors. Observed tiny extra roundtrip charges are 116,600
counted FLOPs across B4 and 6,348 across B3; QR increases these to 156,030 and
8,580 respectively. These are tiny diagnostic charges, not production pricing.

A parameterized leading-order extrapolation for the literal identity wrapper is
`14*n^3*sum(k)` extra f32 FLOPs, before quadratic work and casts. At n1024/d16,
`sum(k)=120`, giving about `1.804e12` additional FLOPs before the unchanged F21
operator. This is an algebraic operation inventory, not a measured full bill.
Symbolic identity aliases can eliminate the redundant identity products and
unused Gram work without changing results, but then the representation reduces
to the same dense baseline storage and gains no compression. Any useful bridge
must establish a numerically acceptable nontrivial representation and a complete
executed cost ledger separately; no rank run is justified by this wrapper.

## Provenance failures preserved and repaired

The original historical B4 artifact remains at SHA256
`a5c6295c45695cc104f6db4d4f779cce059b18773184dfb418092953e420e52c`, with its original
85 failures. `b4_f64.json` and the initial f32 artifacts in this directory ran
through the local NumPy **shim** due to direct-script import precedence. Their
installed-distribution version metadata did not prove which module executed.
They remain preserved in commits `7491d72` and `2c6c503`; do not cite those f32
records as real-backend evidence. The current runners remove the local shim
search path before import, refuse a previously loaded shim and save actual
module paths. The real-backend records are separate files under
`b4_real_backend/`; the real f64 repair is `b4_f64_v2.json`.

The independent Skeptic's NaN mutation exposed loss of failure output under
strict JSON encoding. `json_safe` now preserves nonfinite operands as explicit
`nonfinite` tags. `b4_f64_nan_mutation.json` contains the first layer0
`wick.(4,1)` failure, 33 checks and all captured operands; the original passing
artifact is unchanged. Seven tests cover full NaN failure publication, original
threshold preservation, output/field mutations, hidden-f64 rejection and exact
AST equivalence of the backend's declared allocator/reassignment adaptation.
Command: `uv run python -W ignore local_sim/test_p2_compact_precision_e14.py -v`.
Discovery over every package in local_sim can preload the shim; the fail-closed
check correctly rejects that environment. Use the direct test entrypoint or
preload installed flopscope before targeted discovery.

Two initial implementation-plumbing failures occurred before complete terminal
evidence: a NumPy scalar JSON serialization error, then a copied B3 depth
assertion. Their fixes changed serialization/fixture plumbing only. A subsequent
localization attempt failed before the first layer because the shim has no
meter; that discovery triggered the real-package correction above. No threshold,
fixture value or mathematical recurrence was changed in response to a result.

Compression remains blocked on the failed actual-f32 prerequisite. No scientific
kill of all shared-basis methods, deployment authorization, production accuracy,
score improvement or private-rerun conclusion follows.
