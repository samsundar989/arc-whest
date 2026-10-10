# flopmodel: an exact shape-only interpreter of flopscope 0.12.1 billing

Built on Oct 10 by the strategist's cost-census agent and checked by an independent verifier. It runs an estimator file against fake `flopscope` and `whestbench` modules that track **shapes only** and bill every op by flopscope 0.12.1's rules. It needs no GPU and no data, uses under 3 GB of RAM, and takes about 1 s per run.

## Validation

It matches the real meter to the FLOP and the op, including the harness's output cast (+32,768 FLOPs, +1 op):

| estimator | FLOPs | ops | source of the real-meter figure |
|---|---:|---:|---|
| v77 | 395,412,204,457 | — | real-meter census anchor |
| v81d_starall2 | 387,664,679,697 | 25,741 | `experiments/2026-10-08_p3/v81d_fresh_a.json` |
| v81d_starall2ws | 396,493,421,481 (F/B 0.18030) | 23,181 | interpreter only |

Every billing rule was first checked against the real 0.12.1 package at small shapes (`probe_bill.py`, `probe2.py`, `probe3.py`).

## Run

```
cd docs/research/2026-10-10_flopmodel
python census_v81d.py ../../../estimator_k3_v81d_starall2.py
```

- **Use a fresh process.** `fakefs.install()` replaces `flopscope` and `whestbench` in `sys.modules`.
- **Output files.** The run writes `<estimator>.mm.json` (every `_mm` call with shapes, Strassen depth, FLOPs and ops) and `<estimator>.ledger.json` (every metered op) into the working directory. Run it from a copy if you want a clean tree.
- **Path order.** The scripts first insert `~/scratch` into `sys.path`; that is where they were written. Delete that line, or make sure no stale copies exist there.

## Files

- `fakefs.py`: the interpreter and its billing rules.
- `run_census.py`: loads an estimator, wraps `_mm`, runs `predict` on a 1024×16 shape-only MLP.
- `census_v81d.py`: per-site report.
- `sbmodel.py`: the exact `_sb` (batched Strassen–Winograd) cost recurrence.
- `variants.py` … `variants10.py`: patch functions measured on Oct 10. They form an import chain, and `variants7.py` crashes on its third run. Contents:
  - `fnp.block` reassembly;
  - fused `_advance` products;
  - the join's plain `@` via `_mm`;
  - skipping joins;
  - two-level sandwich split;
  - last-layer rows;
  - old rebuild on kept rows;
  - optimal depth;
  - two-sided D (`v8_plus_v6`, variants8);
  - columns-only D (`cols_only_D` = "V8b", variants9).
- `kink_tables.json`: FIT q50 kept counts and kink-active counts at τ 2.0, 2.5 and 3.0.
- `pernet.py` / `pernet_F.json`: F with per-network keep counts.
- `estimator_k3_v81d_starall2.py.mm.json`: all 285 `_mm` calls of v81d.

## Billing rules (flopscope 0.12.1, as verified)

- **dtype rates:** float16 = float32 = complex64 = 1; float64 = int64 = 2.
- **matmul:** batch·m·n·(2k−1).
- **einsum:** (operands − 1)·α + α − M.
- **Copies:**
  - `concatenate` and `stack` bill numel(out);
  - `block` bills numel(out) once at any nesting;
  - `reshape` bills numel even as a view;
  - slicing, `.T`, transpose and diagonal are free.
- **Elementwise:** numel; `clip` 2×; tanh, exp and log 16×; `norm.cdf` 96 per element, `norm.pdf` 54.
- **Reductions:** numel − numel(out).
- **Gathers and sorting:** `where`, `take` and fancy getitem 4 per element; `argsort` 4·n·⌈log₂n⌉.
- **QR:** 2(2mnk − 2k³/3).
- **Constructors:** `eye` and `diag(v)` bill n; `zeros` is free.
