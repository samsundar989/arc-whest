# S2t3 equal-step control — written before the run (2026-09-27 ~16:50 UTC, worker-q 2236)

**Question.** S2t3 (S3 init, TRAIN 900 + d8b 2,048, lr 1.5e-4, 736-step cosine) reached
VAL-100 (full rows 900–999) 2.246946e-08 at step 368 (−0.79% vs S3's 2.264823e-08). The only
existing control, S3b, ran 224 steps on TRAIN 900 with a 224-step cosine (best −0.28% at step 140).
Is S2t3's extra gain the d8b data, or simply more optimisation steps?

**Control.** Identical flags except `--train-panel train` (TRAIN 900 only) and `--epochs 7`
(784-step cosine; the twin supports whole epochs only, so the schedule at step 368 runs ~10%
hotter than S2t3's 736-step cosine). Read the step-368 VAL-100 evaluation; cancel after it.

**Hypothesis / expected effect (mine).** Most of the gain is steps, not d8b: the control reaches
−0.45% to −0.70% at step 368; d8b is worth ≤ 0.3 pp. P(control within 0.15 pp of S2t3) = 40%.

**Resources.** 16 GiB RAM / 12 GiB VRAM / 4 CPU declared (S2t3's measured envelope), ~26 s/step
→ ~2.7 h of GPU. Low priority, preemptible (resumes from ckpt.pt).

**Stop condition.** Cancel after the step-368 evaluation line appears. Attribution only: this run
does not gate the v70 canary, whose evidence is DEV-100 paired against v68_wino.
