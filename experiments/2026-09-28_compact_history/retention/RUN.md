# E2 retention screen (diagnostic, not a candidate)

Hypothesis: a q-dim projection of history older than 3 births retains most of v72's
final-mean signal. Expected (prior D5: rank64@age4 = +546% raw; fold A3U3@r256 = +31.6%):
large loss. Stop rule: if the optimistic rank-64 bound is > +96% raw vs v72 (break-even
at a projected ~0.11B core cost), the age3/q64 carrier cannot beat v72 without training
recovering most of it; go to E5.

Arms built by `make_arms.py` from v72 sha256 3b6cc397 (see its docstring).
Panel TRAIN full 0..7. Twin torch forward, float32, no grad.
Resource review: one net resident (64 MiB weights), no autograd, no caching beyond
final means (8x1024 per arm). Comparable job2302 peak 8.53 GiB host / 6.05 GiB VRAM.
Declared 16 GiB host / 10 GiB VRAM / 2 CPU. Rerunnable; no checkpoint needed.
