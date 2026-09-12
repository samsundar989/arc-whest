# Integrated first-source synthetic cost instrument — design proposal only

**Status:** unmeasured design. No protected data, estimator change, decision
block, vault touch, source implementation, or cost verdict is attached. This
proposal is for **one externally supplied weighted-discrete first K4 source**
and its induced K3 feeds across a full-width 1024 × 16 schedule. It does not
construct physical later nonlinear K4 increments or establish an eligible
estimator, score, rank adequacy, or server residual headroom.

## Existing meters and what they count

The supplied-core compact reference
[`live_meter.json`](../../experiments/2026-09-10_stage1b_compact/live_meter.json)
bills 818,494,496,296 official-package FLOPs. It has 15 ordinary K3 source
births, 10 weighted-Gram/thin-QR joins, rank 384, and ordinary residual rank
16. It also contains the **old analytic K4 core and its embedded K3 birth**.
Its supplied-core bill does not establish a physical source identity; the
separate frozen arm's source construction has a known cumulant defect.

The separately accepted
[`first_source_kernel/meter.json`](../../experiments/2026-09-11_stage2_compact_p07/first_source_kernel/meter.json)
bills 704,164,664,384 FLOPs for one supplied S64 bank. It carries that bank
linearly through 15 K4 uses and forms 14 induced K3 feeds. It reads those feeds
through repeated dense products; it has **no ordinary K3 backbone or actual
shared weighted-Gram/QR state**. Its bill partitions without overlap as:

| First-source family | Counted FLOPs |
|---|---:|
| Bank ingest, centering, first map | 135,461,952 |
| Later map casts and K4 bank transport | 2,028,011,520 |
| K4 D4/D22/D31 construction | 5,705,462,784 |
| Feed birth setup | 1,892,352 |
| Direct feed-age A/P/a transports | 239,681,756,160 |
| Direct feed reconstruction and readout | 456,612,079,616 |
| **Total** | **704,164,664,384** |

The 456.612B direct readout includes 390.651B in two negative D21 dense
products. The 239.682B transport includes 225.376B in dense A carries. Both
are work on the feeds that the proposed compact Sources must represent. They
cannot be added to compact transport, join and readout costs for those same
feeds. Conversely, the 818.494B compact bill covers **ordinary** K3 sources;
adding 28 feed sub-sources changes its calls and cannot be priced by reusing
that fixed total. Its measured groups include 98.012B join/rebase, 184.397B
transport, 111.852B materialization and 319.573B contraction on its existing
schedule. These are accounting anchors, not marginal prices per new Source.

## One-executable design and overlap rule

Use a **single** `flopscope.BudgetContext` and one operation-family ledger.
Freeze deterministic, truth-free input bytes: 16 width-1024 dense maps,
positive S64 weights and nodes for one first-layer bank, and nonconstant
response vectors. The K4 bank uses the same map bytes as the ordinary
mean/covariance path. One cast of each shared layer map is charged once: the
first-source meter's `cast_source_map` and 15 `cast_layer_map` calls overlap
all 16 of the compact reference's `cast_weight_f32` calls **if** the integrated
input fixes source map = W0 and later maps = W1–W15. The separate historical
meters used different maps and did not prove this sharing. The bank transport
`W @ V` remains a separate multiplication from `W @ C` or `W @ mu` and is
charged.

At each noninitial layer, transport the first bank and construct discrete
fourth-cumulant D4 and, except at the final layer, D22 and **both orientations
of D31**. Supply those slices to the existing K4 mean/covariance consumers
in place of the analytic core's slices. Split the old core's K4-induced K3
birth out of ordinary K3 source construction. Feed the discrete source through
the accepted two-Source representation for each birth: one positive-minus-a
leg with all S64 residual columns, and one negative covariance-pairing leg
with literal dense n-column A/P. Keep ordinary K3 sources and their own
residual rank 16. At age boundaries, join **both** feed Sources through the
same young/shared weighted-Gram/thin-QR state as the ordinary sources, using
the declared rank 384 and the actual projection, carried-Gram update, factor
rotation and transport. Read D3 and oriented D21 once from the unified state
and apply each mean/covariance consumer once.

The K4 covariance `C = V diag(p) V.T` needed by D22/D31 is the same C at a
feed birth. Reuse it for `q = diag(w1) C` and
`a = w2 * diag(C)`; do not rebuild Q from a second dense product. Form the
positive `Aplus = diag(w2) (V * V)` and its residual separately. This reuse
requires that the K4 consumer and feed birth see identical bank, weights and
layer frame. Assert those identities in the tiny schedule. It avoids a named
duplicate; it does not make the whole physical-source construction free.

The old source-kernel direct feed-age transports, Q/H reconstruction, positive
and negative D21 products and dense readout are **replaced** by the compact
feed Sources' transport/readout. The compact reference's analytic K4 consumer
and embedded K4 birth are also **replaced**. Ordinary K3 history, the full
mean/covariance backbone, nonlinear coefficients, and their distinct products
remain. Merely setting its `births` flag to false and adding the first-source
total would change the ordinary source path; the integrated executable must
make the split explicitly and bill the actual resulting calls.

## Schedule predictions to audit before the live meter

At layers L0–L15, predict 15 K4 bank uses at L1–L15, of which 14 need full
D4/D22/D31 and L15 needs D4 for the final mean. Predict 14 feed births at
L1–L14, giving 28 compact feed Sources, alongside the reference's 15
ordinary K3 births. The feeds have 105 later-layer age uses: 91 nonfinal full
D21 uses plus 14 final D3-only uses. Because every feed has two sub-sources,
predict **210 feed sub-source D3 uses** and **182 feed sub-source D21 uses**.
With the reference's 120 ordinary D3 and 105 ordinary D21 uses, predicted
totals are **330 D3 Source uses** and **287 D21 Source uses**. For recent=4,
predict **28 joins**: 10 ordinary plus 18 feed sub-source joins. An ordinary
L0 source inserts at L1 and first joins at L5; a feed born L1 inserts at L2
and first joins at L6. The final L15 skips joins, so only feed births L1--L9
join (nine pairs), not ten. The original design predicted 30; this static
correction precedes any integrated-fixture measurement. L0 has no
history readout, and L15 has no new birth, join or D21. These are frozen
design predictions, **not observations**; the tiny executable schedule must
audit ages, calls and each count before a live-width cost is interpreted.

The full-rank n4 two-birth fixture proves exact abstract feed addition and
carries for its fixed law. Its rank-2 result shows material loss. Neither
certifies the rank-384/residual-16 live state. The S64 positive residual is
kept rather than silently cut to 16; any such cut needs a separate error and
cost intervention.

## Required cost and memory ledger

Name each operation family and reconcile the family sum to the context's
atomic FLOP total. Account for bank cast/center/map, K4 transport and every
fourth-cumulant pairing, source covariance and Aplus/q construction, both
Source legs, D4/D22/D31 consumers, ordinary and feed K3 mean/covariance
consumers, all young A/P/Z/Y transports, weighted A/P Gram sketches, QR,
join projection and Gram updates, carried Gram rotation, factor rebasing,
shared-Q transport, full D3/D21 reconstruction, output casts/stack and
first/final shortcuts. In particular, bill the negative leg's full n-column
A/P rather than treating a covariance-rank observation as a compression
proof. If a physical quadrature or later nonlinear signed increment is
actually generated inside the executable, bill it as a separate named family;
otherwise report it explicitly as external and unmeasured.

For each layer, record resident bytes for weights, discrete bank, all young
and shared Source fields, Q/Gram, ordinary backbone, and output; record peak
live scratch during Gram/QR, factor rotation and materialization. The two old
payload estimates cannot be added: the compact reference reports 345,587,712
bytes including 134,217,728 external weight bytes; the first-source meter
reports 134,988,288 external input bytes and 114,089,984 bytes of internal
payload, with most layer maps duplicating the weights. The literal 14
negative A/P pairs alone occupy 112 MiB at float32. Also record largest
single array and workerq's observed process peak. Windows peak observation
is not enforcement of the grader's 8 GiB limit; no local wall time is server
residual authority.

## First falsifiable metered milestone

Before the n1024 run, preregister exact inputs, family categories, the schedule
above, cost threshold and falsifiers in `experiments/LOG.md`, commit source
and a tiny independent n4 dense cumulant/feed/readout oracle, then run the
tiny CPU-only schedule. The first large milestone is the **one-source
integrated** executable, not all later nonlinear K4 increments. Required
checks: exact context/family reconciliation; audited 15/14/105/91/28 and
330/287 counts and ages; no duplicate charge for shared casts/C or removed
dense feed path; deterministic output and hash; final shortcut execution;
each instrumented array below 4 GiB; explicit counted FLOPs relative to
`2**41`; and observed process peak below 8 GiB. Any failed check remains a
FAIL. A passed cost instrument still says nothing about accuracy or server
residual headroom.

After frozen source/input/installed-package hashes and HEAD are recorded,
run the live-width meter only through workerq with a conservative reservation,
for example `workerq submit --project arc-whest --ram 12 --vram 0 --cpus 2 -- uv run python <meter> --width 1024 --depth 16 --json <result>`.
The queue reservation protects the host; **12 GiB reserved does not satisfy
the grader's 8 GiB limit**. Keep the tiny run separate from that queue job,
preserve the workerq raw manifest/log/result and byte-preserving archive,
verify source/input/package/result and HEAD before and after, then run a
separate current-byte default preflight through workerq and obtain an
independent fresh-context Skeptic. Do not infer a complete physical-source
or eligible estimator cost from this single-source milestone.
