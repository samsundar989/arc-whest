# Abstract K4-to-K3 feed embedding: focused fixture result

This truth-free fixture implements the IMPORTANT contract in `experiments/LOG.md`
commit `60ab638`. It tests the induced cubic K3 feed of one chosen weighted
discrete fourth-cumulant source, not a physical ReLU source or a complete
estimator. Source SHA256 is
`f50c199adb1204e47bd931283f7f1973d85b5a256d0a0cbeb0dc1f32560751c4`;
`result.json` SHA256 is
`e11bb7c9fbedd032274792feaa98d838a006b1397f2044b5559950ce26d43d5c`.
`result_receipt.json` binds both exact byte hashes.

The fixture independently enumerates the positive fourth raw moment, three
covariance-pairing negatives, and all placements of the declared quadratic
K3 feed. It builds the reference's positive-minus-`a` Source and negative
pairing Source, reconstructs `M=P diag(s)+3A diag(e)+Z L.T`, and enumerates
their complete cubic tensor. The direct `dense_slices` and full-rank
weighted-Gram/QR `compact_slices` readouts agree at birth and after two fixed
signed dense carries. It also separates positive, `a` negative and pairing
negative terms, tests zero-`w2` and zero-source cases, and rejects a dropped
pairing mutant. The focused result is **48/48 PASS** with maximum numerical
error `2.22e-15` against the `1e-10` absolute gate. The mutant gap is
`1.161675`, above its fixed `1e-3` threshold.

For the registered three-atom equilateral law, covariance has rank 2 while
the full feed unfolding has rank 3 and determinant `-2.5`. This blocks a
rank-2 common A/P/M span with zero residual and no optional Y birth leg for
that example. It does not exclude other encodings or establish the required
rank for a width-1024 physical source. The literal negative subsource carries
two `n×n` factors per birth; 14 births use **112 MiB** of A/Q payload and 105
transports cost **450,971,566,080 conventional FLOPs**. Those frozen arithmetic
figures exclude joins, QR, materialization, the K4 source bank and the
ordinary K3 backbone. They are not a flopscope measurement or a score.

The original focused stdout is `focused.log`. A current-byte default
preflight and independent fresh-context Skeptic review remain open.
