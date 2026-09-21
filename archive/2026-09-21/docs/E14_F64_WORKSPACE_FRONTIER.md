# E14 full-f64 workspace frontier

September 14, 2026. This is a static liveness analysis of the failed full-f64 isolation control, not a proposed implementation or release claim. Exact candidate SHA256: `9e221c83e6e36838e624d599006399e9f0a7ebc7c60fd30d92d85d7485505512`. The enforced Linux failure and controls are in `experiments/e14_tree_f64/linux_memory_gate.json`.

At the first failed allocation, `_advance` has a source age of 14 and `_mm` is multiplying an approximately `1024 × 14336` tower block by its conjugate. `_sb` batches seven Winograd products at each of four levels. For float64, the simultaneous recursive input arrays have the following sizes; views, product outputs, gluing arrays, the prior tower, covariance/core and runtime are additional.

| recursion level | batch | operand dimensions | `SA` GiB | `SB` GiB |
|---:|---:|---|---:|---:|
| 1 | 7 | 512 × 7168; 7168 × 512 | 0.191 | 0.191 |
| 2 | 49 | 256 × 3584; 3584 × 256 | 0.335 | 0.335 |
| 3 | 343 | 128 × 1792; 1792 × 128 | 0.586 | 0.586 |
| 4 | 2401 | 64 × 896; 896 × 64 | 1.026 | 1.026 |

The eight `SA`/`SB` arrays alone total **4.277 GiB** while the recursive call stack holds parent arrays. The failed request was the 1.026 GiB level-4 `SB` array, reported as `(2401, 896, 64)`, with address space already near the enforced 8 GiB cap. This matches a liveness bottleneck rather than an individual-array 4 GiB violation. The executed Linux path reached 8,168.4 MiB VmPeak, then raised `MemoryError`; it produced no prediction. The full uncapped counted bill is **2,029,122,417,342 FLOPs**, only **169.901 billion** below 2²⁴¹. Its chooser projection underestimates that bill by **72.187 billion**, so generic reduced budgets also need correction.

Simply shortening the recursion from four levels to three would avoid the two largest input buffers but changes the multiplication schedule and cost. The dated f32 tariff for the tower was 1.181830T at depth 3 versus 1.068212T at depth 4, a 113.618B increase before dtype effects. Doubling that difference as a rough f64 estimate gives ~227.236B, already larger than the candidate's 169.901B budget headroom; this is **not** a metered depth-3 bill or a proved impossibility. A workspace schedule that frees or streams parent operands while preserving the exact arithmetic and complete charges is a different intervention, as is selectively retaining f64 only for numerically sensitive state/consumers. Neither has been specified, costed, tested or predicted by Samuel. The f32 tree still fails its unchanged tiny precision gate. No fresh truth should be spent on these ideas before a prospective contract and independent numerical/memory proof.
