# Frozen nonlinear old-history fixture: FAIL

The IMPORTANT contract was committed in `experiments/LOG.md` at `dca4e7f`
before the first focused run. The unchanged order-48/order-64 positive bank,
bounded source contrasts, B/D maps, gates, and controls are in
`docs/research/2026-09-12_nonlinear_old_history_proposal.md` at `3950b0c`.
The focused command in `result_receipt.json` ran as a tiny CPU-only fixture
and exited **1: FAIL 54/56**. The full raw result is `result.json`; it is
retained without changing the bank, contrasts, or thresholds.

The two failures are both positive-law numerical gates at the *third*
activation layer. The maximum material-relative order-48/order-64 raw-moment
gap is `3.573549143066814e-05`, and order-48/direct-continuous is
`3.1351810768989914e-05`, against the frozen `1e-8` gate. Order-64 versus
continuous still differs by up to `4.383680661678227e-06`. The worst
moment is degree four `(3,1)`; degree-one moments also fail. The independent
continuous angular-sector integration reports maximum per-moment
reported/doubling bounds of `5.651273612334797e-14` at layer two and
`7.299233333272794e-14` at layer three. At layer two, order-48/order-64
and order-48/continuous raw gaps are at most `9.230294561410627e-16`
and `1.2939110862268893e-15`. This pattern is consistent with the final
ReLU introducing two new angular kinks inside the accepted bank's B-sector
interval, at `tan(theta)=1.125/1.735` and `tan(theta)=0.94/0.83`; the
fixed bank was not split there. This is an interpretation of the failure,
not a repaired result.

The remaining frozen checks pass on the fixed order-48 bank. Both weighted
moment matrices have rank ten and condition below 49; positive references
match all degree-<=3 moments to at most `6.661338147750939e-16`.
Independent atom-by-atom partition cumulants, all dense K4/K3 entries,
four counterfactual contrasts, D4/D22/both oriented D31, the declared
quadratic K4-to-K3 feed, source-order identities, and zero controls pass.
The layer-three nonlinear old-history gap is `0.002919803602011131`
(ratio `1.2267796837641833` to the layer-two conditional old K4 maximum),
while replacing the final ReLU with the signed linear map gives a gap
`2.5647886592317093e-15`. The swapped-D31 mutant gap is
`1.4082726134355412e-06`. The layer-three K4 age interaction maximum is
`2.211268987695192e-06`, independently matched within
`2.4424906541753444e-15`. The true K3 contrasts are reported separately
from the declared truncated feed; their maximum difference is
`0.006863146979689153`.

This is a **failed numerical fixture**, not an accepted three-layer source
or nonlinear closure. The same-bank identities establish only the specified
conditional signed-atomic-law bookkeeping on its finite bank. The old
source's lower-moment leakage and higher-moment content are retained; K4
alone does not encode them. No width-1024 rank, counted FLOPs, server
residual, labels, score, estimator, seed block, public split, or vault was
used. A new bank design would require a new preregistered contract and a
fresh independent Skeptic after this FAIL is recorded.

Source SHA-256:
`991a4e38eefe90ee2f761556dd60ac1b3b6c86280437d096d8a8d8f29a6d5c61`.
Raw result SHA-256:
`fead616c0b5149ad48ca109a12eafe56a5ec698a0e45ddcba86e9af738735df9`.
