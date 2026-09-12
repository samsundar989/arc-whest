# Full-polar split-bank nonlinear fixture: focused PASS

The separate IMPORTANT contract was committed in `experiments/LOG.md` at
`ad856c6` before this new fixture ran. Its design memo is
`docs/research/2026-09-12_full_polar_nonlinear_bank_proposal.md` at
`b33481b`. The original `nonlinear_old_history/` result remains FAIL
54/56 and is not changed by this new run. The focused command in
`result_receipt.json` exited zero: **PASS 77/77** fixed checks. All raw
per-moment and per-gate values are retained in `result.json`.

The full-polar fixed-sign bank partitions the original correlated Gaussian
at both initial ReLU coordinates, B's zero lines, and D's two additional
zero lines. The independent dynamic root finder matches all nine fixed
boundary endpoints within `1.1102230246251565e-16`. The three-node
Rayleigh rule has positive nodes/weights and maximum defect
`3.907985046680551e-14` for its six degree-0..5 moments. The 32/48/64
per-sector angular rules produce 768/1152/1536 indexed particles. All
weights are positive, and the unrescaled full mass misses one by at most
`6.661338147750939e-16`. The largest single bank array is 24,576 bytes.

Every unreweighted raw monomial through degree four at layers two and three
meets the `1e-8` material-relative gate. At layer three, worst 32/48,
48/64, and 48/direct-continuous gaps are respectively
`1.7648343725246743e-15`, `1.6387747744871975e-15`, and
`3.4036091470118717e-15`. At layer two, 48/direct-continuous is at most
`2.911299944010501e-15`. The independent continuous scalar integration
reports maximum per-moment error estimates of `5.651273612334797e-14`
and `7.299233333272794e-14` for layers two and three, under the fixed
`1e-10` gate. The unchanged unsplit order-48 bank remains a negative
control: its layer-three/direct gap is `3.1351810768989914e-05`, above
the frozen `1e-6` discriminator. This supports the numerical-kink repair
for the newly defined toy bank; it does not reverse the old failed result.

Only after the positive-bank gates passed, the fixture constructed the
unchanged bounded two-source reweighting on the new order-48 indexed bank.
Both weighted moment matrices have rank ten, conditions 46.33 and 39.06,
positive references, and degree-<=3 raw-moment defects at most
`2.220446049250313e-16`. The maximum same-bank raw/cumulant/consumer/feed
identity defect is `6.439293542825908e-15` against `1e-10`. All dense K4
and K3 entries, four counterfactuals, D4/D22/both oriented D31,
covariance, declared quadratic K4-to-K3 feed, source order and zero
controls pass. The nonlinear conditional old-history K4 gap is
`0.0059254615716694695`, ratio `0.8870270194819272` to the layer-two
old K4 maximum, versus strict `1e-6` absolute and `1e-3` relative gates.
The signed-linear D control gap is `3.1866870253693946e-15`, and the
swapped-D31 mutant gap is `2.037516764634706e-06`. The K4 age-interaction
maximum is `3.4332154437155538e-06`, independently matched within
`1.5543122344752192e-15`. True K3 contrasts and the declared truncated
feed differ by as much as `0.012437823027753463`; they are not equated.

This is a fixed two-coordinate positive-law and conditional signed-atomic
identity result, awaiting current-byte default preflight and a fresh
independent Skeptic. The bank looks ahead at B/D to place angular splits,
so it is a toy diagnostic rather than a viable width-1024 construction.
The atomic old source contains moments above fourth order, and ReLU can
create lower-moment leakage. No K4-only nonlinear closure, compressed
rank, counted full cost, server residual, score, protected label, estimator,
decision seed, public split, or vault claim follows.

The separate executable imports the unchanged old-history atomwise/source
helpers (SHA-256 `991a4e38eefe90ee2f761556dd60ac1b3b6c86280437d096d8a8d8f29a6d5c61`)
and the unchanged axis-bank helper for the unsplit negative control
(SHA-256 `1f2f421bcc696dec263a77847b076914982bc41eba08f251a4034925a6f4d1d9`).
The new split bank and positive-bank gates are separate code; the direct
continuous oracle does not use that new bank.

Source SHA-256:
`e3708d9d878f3191cbf61dcd632e188507d0b575e77e692219108c280d3ff59f`.
Raw result SHA-256:
`9a8692559f2744bccd4a0bcd6fef1941f9bab551af984f3fdeb316fe15e277ab`.

Workerq 996 ran live default preflight at tested HEAD `238ce53` with
5 GiB RAM, zero GPU, two CPUs and no source snapshot. It **SUCCEEDED
GREEN 12/16**, four Full-only skips, at measured process peak
`2946.8125 MiB`. Raw `preflight_996.log` SHA256 is
`c791e0bb250b8d8ab464f48d44905d7665a3834d9e7cbb4fe5627ca29fcfd788`.
The first post-run PowerShell hash enumerator failed and wrote an empty
`python` object; that intermediate receipt was not accepted as evidence.
`preflight_996_after.json` was regenerated from the exact 249 paths
listed in `preflight_996_before.json` using Python. HEAD, four named
source/helper/result hashes and **249/249 Python hashes** now match;
the independent Skeptic also recomputed them from live files. The
default preflight did not run the focused 77-check fixture or enforce
the grader's 8 GiB cap on Windows. Skeptic verdict remains pending.
