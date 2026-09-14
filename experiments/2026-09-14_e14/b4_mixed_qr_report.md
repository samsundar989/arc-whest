# Mixed-f64 QR remedy: bounded output control passes, exact roundtrip does not

Prediction/protocol committed `832c0d6`, after the unchanged f32 QR failures.
Four existing fixed cases only: B4 n8/d6 and B3 n4/d4, each against L2 and forced
F21. No rank, data, candidate, threshold or frozen-source changes.

Only the complete source representation, Gram, QR basis and factors use f64.
Encoding uses explicit counted f32-to-f64 casts directly from wrapped frozen
tower arrays; decoding reconstructs in f64 and explicitly casts the dense tower
to f32 before the unchanged frozen `_layer`. No zero mask or hidden cleanup is
applied. Means, covariance and nonlinear/reference operator precision remain f32.

All four final-output squared drifts are **exactly zero**, below the unchanged
`2e-10` allowance. Every saved original-`1e-5` operator checkpoint passes. The
candidate and reference operator bills are identical within each case.

Exact decoded-state recovery is **false** in all four cases, retained separately
from output acceptance. B4 has 121 bit-pattern mismatches including 48 entries
that were zero and become nonzero; B3 has 12 mismatches, all zero-to-nonzero.
The first consumed mismatch is layer1 A_L. First maximum absolute discrepancies
range from `5.64e-19` to `1.81e-18`. Fullrank QR reconstruction leaves tiny
roundoff at exact zeros; casting to f32 does not make those values vanish.
The youngest P is symbolically identity in the frozen operator and is recorded
as an unconsumed extra diagnostic; it is excluded from consumed-state counts.

| Fixture/reference | f64 encoder FLOPs | f64 reconstruction + f32 cast FLOPs | Unchanged operator FLOPs | Maximum representation payload |
|---|---:|---:|---:|---:|
| B4/L2 | 208,020 | 106,720 | 147,648 | 9,664 bytes |
| B4/forced F21 | 208,020 | 106,720 | 342,484 | 9,664 bytes |
| B3/L2 | 11,316 | 6,048 | 14,758 | 1,696 bytes |
| B3/forced F21 | 11,316 | 6,048 | 25,165 | 1,696 bytes |

These are actual tiny diagnostic operation charges. Payload counts unique array
wrapper objects in the representation, excluding dense reference/candidate state,
transient work, allocator metadata and process overhead; they are not peak RAM.
No full-shape bill or savings claim is made. f64 factors double element storage
relative to corresponding f32 factors, and all encoding/decoding operations are
additional to the existing operator. This remains a dense-reference wrapper.

Evidence: `b4_mixed_qr/b4_mixed_qr_summary.json` links four complete raw records.
They retain every decoded operand and zero/bit mismatch, all operator checkpoints,
exact source hashes, real-package import paths, one-thread backend provenance and
per-layer encoding, decoding and operator counts. Two lightweight tests verify
f64 representation/f32 operator boundaries and distinguish structural-zero drift
from signed-zero bit changes:
`uv run python -W ignore local_sim/test_p2_compact_mixed_qr_e14.py -v`.

The mechanism-grounded stabilization succeeds for the fixed reference-order output
control. It does not establish exact source recovery, a numerically conformant
**autonomous** compact program, useful cost, production L4 behavior, compression,
accuracy or promotion. The independent interpreter's plain-versus-Strassen mismatch
remains explicit. The previous f32 fullrank QR refusals remain unchanged; this
wrapper cannot be used to sidestep the prerequisite before measuring rank.
