# Hand-over to Lane P: J3 on the v87 stack (Lane O, O4), 2026-10-13

**Lever.** `_JOIN_SKIP_TAIL = 3` (v84: 2). No old-tier join in the last three layers before the final one, so the
lineage that would join at layer 12 stays exact. It is atom `o4` in `tools/atoms/o4.py` (lane-m): one constant,
order-independent, and it touches no S/K anchor.

    uv run python tools/build_v85.py --params <params_best.npz> --atoms "sb2+sbtrim+depth8+k0:...+k1:2.5_3.0+k2:3.0:...+k3+o4"

With R8 params and K's `:unrefit` arms, this rebuilds the screened file byte for byte (`cda06ee1`).

| | v87 (K stack B on sb2/sbtrim/depth8) | v87 + o4 |
|---|---:|---:|
| F/B (K's out=-aware meter) | 0.14250 | 0.14256 (+0.04 %) |
| ops | 25,211 | **24,780 (−431)**, so the 26,100 gate margin grows from 889 to 1,320 |
| VAL-100 raw, unrefit, paired (stack B over v84) | 1.769215e-8 | 1.764510e-8 (**−0.27 %**, −0.27 ± 0.05 % per net, 75/100) |
| net score | | **−0.22 %** |

**Evidence history:** on v84 at R2 (M6.2) J3 was −0.26 % raw on top of age 6, but it cost +0.36 % F there, so the
net was neutral and it was parked. On K's pruned stack the F cost vanished.

**Parity:** a single Python constant. The twin runs the same source, so twin vs flopscope parity is the base
file's (Lane P's existing v87 parity check covers it). I have not run a separate parity check.

**Refit:** not done by Lane O. J3 adds no parameters, and the corrector and gains were fitted to the J2 chain. The
cheapest valid confirmation is to include J3 as a matched arm in Lane K's stack refit (pod 2), the same recipe with
and without o4. M6's J3 refit on fold (R3, early-stopped) read −0.78 % against its control at the matched step,
so I expect the refit to keep or widen the unrefit gain. Lane P decides.

**Wall:** −431 ops ≈ −0.16 s median server wall at 0.36 ms/op, and slightly lower projected residual.
