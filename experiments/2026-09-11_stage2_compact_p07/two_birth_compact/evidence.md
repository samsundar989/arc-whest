# Two-birth abstract compact K3-feed integration

The preregistered IMPORTANT contract is `experiments/LOG.md` at `424477e`.
The fixed, truth-free, n=4 fixture is
`local_sim/test_p2_k4_two_birth_compact.py`. Its direct oracle enumerates
weighted raw fourth moments, all three covariance pairings, all three
quadratic K4-to-K3 placements, and both signed-map cubic carries by explicit
index loops. The system under test uses the accepted abstract feed Source
constructor and the actual compact `weighted_gram_join`, `transport_shared`,
`reconstruct`, and `compact_slices` functions. Both birth banks and their
quadratic weights differ. The two maps contain positive and negative entries.

The focused command was:

```powershell
uv run python local_sim/test_p2_k4_two_birth_compact.py --json experiments/2026-09-11_stage2_compact_p07/two_birth_compact/result.json
```

`focused.log` records **74/74 PASS**. At birth 1, after carry 1, after birth 2
and the two joins, and after carry 2, the full-rank reconstructed cubic tensor,
D3, and oriented D21 agree with the independent oracle within the registered
`1e-10` absolute gate. Worst absolute errors are respectively
`3.77e-15`, `3.77e-15`, and `4.05e-15`. Independent weighted A/P Gram
reconstruction, Gram symmetry, factor rotation, source ages/counts, positive
unit bank masses, zero-source control, and final mean-only D3 also pass.

The dropped negative-pairing mutant differs at birth 2 by `1.26869171366`,
above its registered `1e-3` minimum. The rank-2 run uses the same banks/maps
and the first two identity columns as omega. Its maximum full-tensor loss is
`1.22818928303` at birth 2; final tensor/D3/D21 losses are
`0.42448916231`, `0.22869262106`, and `0.42448916231`. These are
diagnostics, not an exactness gate or a wide-rank estimate.

Frozen source SHA256:
`552dd841a3baacfe682c86271e601cdac7a5c52383e420ba4ad646bc65e7cdf6`.
Result SHA256:
`1b750c57b65ef64f611ab8434887145817410b40f2ff76704a9ae52e30a26713`.
The receipt binds both. `result.json` also binds the imported feed constructor
and compact reference source bytes. No physical nonlinear source, live-width
rank adequacy, complete challenge cost, residual, or score was measured.

Shared live workerq 994 ran default preflight at tested HEAD `e84b16c` under
5 GiB RAM, zero GPU and two CPUs: **GREEN 12/16**, four Full-only skips,
peak process RAM `2842.9140625 MiB`. Raw transcript
`../shared_preflight_994.log` SHA256
`02d2dd28ee190044cd1cf754fcd3ab2c6756e218cc9c72d34e6c7a23b9003a26`.
Before/after receipts agree on HEAD, both fixture source/result pairs and
248 scoped Python hashes. An untracked design memo and receipt appeared
during the live job, but no tested source/result changed. The default gate
does not run this focused fixture or enforce the Windows 8 GiB grader cap.
Independent Skeptic review is pending.
