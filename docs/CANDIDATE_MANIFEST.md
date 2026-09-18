# Candidate manifest — 2026-09-18

This index distinguishes the installed ship file, the latest graded file, and
the strongest completed DESIGN file. Hashes are SHA-256 of exact source bytes.
It is an index to evidence, not a release certificate.

| role | exact file | SHA-256 | configuration and evidence | certificate |
|---|---|---|---|---|
| legacy shipped | `estimator.py` | `00240eb1e60810f5e9a6f743b8ad89a1533cfd8671c08e07525f1ed1de54d9ec` | v2.6.1; byte-identical archive `estimator_v2_6_1.py`; standard preflight targets this file | historical ship-file checks only |
| latest graded | `estimator_k3_v31.py` | `6dde5fd43ad41e8ec7199b742dd554d22d506c9cf85a532027e86a590d0a787c` | submission #331317; actual Phase-2 score 8.7520303e-9; paired P12 control in `experiments/2026-09-18_s17_factorial/score.json` | graded incumbent; future releases require their own checks |
| strongest completed DESIGN | `estimator_k3_v43_complete_c10.py` | `fd8d497db6dfbc3affcf579119c7115c3aee4d190a3e92fec4d85c366152d1bf` | C10: gram join, age4/rank256, P4d5, P3 off, exact old query; P12 official-precision forecast 6.486609e-9, same-panel v31 8.860665e-9 | Linux 8 GiB and MATH/LOCAL checks pass; full certificate REFUSED at 23,421 vs 22,000 predicted-call screen; full mini #1938 queued; no approval |
| fixed P14 DESIGN trial | `estimator_k3_v45_p2_diag.py` | `fc8e547435e026f70b2e8e95954daf9a61337ae1747f32fc0aa0b61d10e6b855` | C10 plus one active ordinary P2 birth-diagonal compensation; tiny source proof passed, complete cost 480,851,916,161 FLOPs; P14 two-replica truth finalized | paired score #1920 queued; no candidate certificate or upload approval |

P13 SINGLE_USE remains untouched at this checkpoint. The 2× major-release gate
is a fresh paired comparison against v31 in both observed and official precision.
Samuel permits individually approved diagnostic uploads without that milestone,
but requires a concrete candidate certificate and approval of exact bytes for
each upload. Agents do not designate submissions.

C10 diagnostic contract is committed and prospectively narrowed to server
residual measurement after the predicted margin refusal. Linux memory gate
#1891 passed with 5,754.75 MiB peak virtual address space. Its full official
subprocess mini remains pending. The same-closure reference
meter #1921 is queued and has no truth or score result.
