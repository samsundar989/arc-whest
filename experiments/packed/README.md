# experiments/packed — Sprint 4 artifacts

Everything the packed-activation-propagation sprint measured, and the receipts
for how it was measured. Narrative goes in `experiments/LOG.md`; this file only
says what each artifact is and which rule governs it.

## Files

| file | tier | what it is |
|---|---|---|
| `D11_truth_256x32.npz` | DECIDE | 200 canonical MLPs from seed block **D11**, final-layer post-ReLU means at 2²³ samples, drawn from the evaluator's own `spawn[1]` substream. Weights are **not** stored — `mint_provenance` proves they are a bit-exact function of the input seed. GT-noise MSE 6.33e-9 (0.45 % of the 1.4e-6 decision signal). |
| `D12_truth_256x32.npz` | DECIDE | the same for seed block **D12**. **D12 is single-use.** |
| `D11_residual.json` | instrument | uninstrumented wall time per arm, minimum of repeats, measured on an otherwise idle machine. The `x dense` ratio is the only column that transfers; the absolute level is this box's Python overhead, not the grader's. |
| `D11_ablations.json` | DECIDE | per-MLP MSE, counted FLOPs, residual seconds, op counts and analytic-oracle cost for every component, paired with common random numbers. |
| `D12_decision.json` | DECIDE | the S1 gate. One run, one frozen design. |

## Rules that apply here

* **D11 is the design block.** Prefix rule, bucket edges, Strassen depth, pilot
  size and the on-neuron threshold were all chosen against it. Tuning is
  permitted on D11 and nowhere else.
* **D12 is locked and single-use** (`seed_blocks.SINGLE_USE_BLOCKS`). It was
  touched once, against a design frozen before the touch. If a future entry
  quotes D12 for a different configuration, that number is not evidence.
* **The public mini split is CONFIRM tier and was not used to choose anything**
  in this sprint (`docs/VALIDATION.md` §0).
* **The vault was not touched.** Sprint 4 spends vault touches only at S3, and
  only after S0–S2 have all passed.

## Reproducing

```powershell
.venv-gpu\Scripts\python.exe local_sim/packed_truth.py --block D11 --mlps 200
uv run python local_sim/packed_sparse_probe.py --mode parity   --mlps 8
uv run python local_sim/test_packed_sparse.py
uv run python local_sim/packed_sparse_probe.py --mode residual --mlps 3 --repeats 3   # idle machine
uv run python local_sim/packed_sparse_probe.py --mode ablate   --block D11 --mlps 30 --shifts 8
```

The probe imports the **real** `flopscope`, not the `local_sim` shim — it evicts
the script's own directory from `sys.path` before importing, because Python puts
it first and `local_sim/flopscope` would otherwise win. Every FLOP quoted in this
directory is the accountant's, not a model's.
