# Compact-history carrier (PLAN rev17 B5, E0–E5) — closed at the E2 retention screen

**Decision: the age3/q32 and age3/q64 symmetric-core carrier is closed before implementation,
and the first E5 fallback (cheaper D21 contraction) is closed at its price/retention screen.**
No training was run, no candidate was produced, nothing was submitted from this lane.
All numbers below are TRAIN diagnostics (full rows 0–7), not candidate scores.
Final-layer reference noise floor ≈ 7.7e-11 for every MSE quoted.

## E0 — teacher

Teacher = the baked incumbent `estimator_k3_v72_gt.py`, sha256 `3b6cc397…2bac`, run through the
torch twin with `--raw` (the shipped arithmetic, float32). Its twin/official DEV parity is
1.2e-4 (RESULTS v72 row). Two independent twin runs on TRAIN 0–7 agree bit-for-bit.
No trained student exists, so no training/export path needed to reproduce v72.

## Why the core was bounded instead of built

A symmetric core `G(q,q,q)` in a thin basis `U(n,q)` carries exactly the projection of the old
history onto span(U) in all three modes. v72's existing old tier already carries such a
projection for the A_L/M legs (legs = `Q F`, shared basis rank r from a consumption-weighted
Gram range finder), while keeping its P2 residual and hub factors **exact**. Therefore v72 with
`_SHARED_AGE=3, _SHARED_RANK=q` retains at least as much information as the specified
age3/q core with the same basis family. It is an **optimistic bound** on the analytic student.
It needs no new tensor code, so a failed bound saves the E1 build.

Arms: `retention/make_arms.py`, `retention/make_arms2.py` (one source change each from v72).
Probe: `retention/probe.py`, workerq 2319 and 2325. Declared 16 GiB host / 10 GiB VRAM / 2 CPU.
Runtime was ~0.5 s per net per arm (local twin throughput, not a server figure).

## Results (TRAIN full 0–7, twin float32, 8/8 nets each)

| arm | mean final MSE | × v72 | per-net × range | wins vs v72 | F/B (real meter) | paired score change |
|---|---:|---:|---|---:|---:|---:|
| v72 teacher | 2.0439e-08 | 1.000 | — | — | 0.21473 | — |
| age3 bound r256 | 3.8079e-08 | 1.863 | 1.58–2.13 | 0/8 | **0.18725** | **−62.5%** (t95 −74.5, −50.4) |
| age3 bound r128 | 1.3956e-07 | 6.83 | 5.56–8.35 | 0/8 | — | — |
| **age3 bound r64** | 3.2047e-07 | **15.68** | 12.7–18.7 | 0/8 | — | — |
| age3 bound r64, 3 power passes | 3.0827e-07 | 15.08 | 12.0–18.0 | 0/8 | — | — |
| **age3 bound r32** | 5.4266e-07 | **26.55** | 21.9–32.8 | 0/8 | — | — |
| age3 zero history (E2 control) | 9.8659e-07 | 48.27 | 40.9–57.9 | 0/8 | — | — |
| D21 read, youngest 4 lineages | 3.1964e-07 | 15.64 | 13.3–19.0 | 0/8 | **0.18573** | −1253% |
| D21 read, youngest 3 | 5.7043e-07 | 27.91 | | 0/8 | — | — |
| D21 read, youngest 2 | 1.0158e-06 | 49.70 | | 0/8 | — | — |
| D21 read, youngest 1 | 1.8298e-06 | 89.53 | | 0/8 | — | — |

Paired score change = mean over nets of (s_v72 − s_arm) / mean s_v72, with s = MSE × max(0.1, F/B),
t95 with 7 d.o.f. Costs are measured on `tools/_cost_worker.py` (f64 read-only grader twin, DEV net 0):
age3_r256 411,777,318,977 FLOPs / 24,024 ops; d21age4 408,425,256,681 FLOPs / 22,037 ops.
Summary JSON: `retention/summary.json`. Raw per-net rows and final means: `retention/retention*.json`,
`retention/finals*.npz`. Logs: `retention/job23*.log`.

## Reading

1. **E2 triage technically passes, and is irrelevant.** r64 beats the zero-history control on
   teacher distortion 8/8 (3.05e-7 vs 9.68e-7), which is the E2 admission wording. But the
   *optimistic* bound is ×15.7 raw (r64) and ×26.6 (r32). Analytic core cost is **estimated,
   not metered**, at ≈0.115B (q32) to ≈0.134B (q64). Recent age3 history is 42/65 of v72's
   recent lineage-layers, and the per-layer core readout/fold is ~n q³. At 0.12B, a 20% gain
   needs raw ≤ 1.43× v72. The pilot's shared gates would need to remove ~91% (q64) to ~95%
   (q32) of the bound's error. They cannot enlarge the retained subspace, and they are fixed,
   bounded, per-neuron gates. E3's long-run bar (≥15% on full VAL) is not reachable from here,
   so building E1 and running a 60-step pilot would spend days to confirm this.
2. **Basis quality is not the limit.** Three power passes move r64 only 15.7 → 15.1.
   The capacity curve is steep: r256 ×1.86, r128 ×6.8, r64 ×15.7. The value in old κ3 history
   is spread over hundreds of directions. This extends D5 (rank 64 at age 4: +546%) and the
   September 26 fold result (A3U3 +31.6%). A dense q ≥ 128 core costs ~n q³ per layer to read,
   which is 4.3 GFLOPs/layer at q = 128 and far more at 256. That removes the price advantage.
3. **E5 fallback, recent-factor compression.** The most favourable recent compression that
   keeps the existing old tier is folding lineages at age 3 instead of 5 into the rank-256
   tier. It is measured at −12.8% cost for ×1.86 raw, a **62.5% worse** score. Break-even at
   −13% cost allows at most +15% raw.
4. **E5 fallback, cheaper D21 contraction (new mechanism, one test).** Reading the dense
   (2,1) slice only from the youngest m lineages keeps D3 exact, and the birth still subtracts
   the slice actually read. So unread content propagates through the linear Wick transport,
   and only its covariance/closure/corrector feedback is lost. The identity check
   (`retention/check_young.py`, job 2326) matches production `_dslice21` exactly for k ≤ 5
   and to 2.6e-5 (Strassen reassociation) with the old tier present. It is measured at
   −13.5% cost for ×15.6 raw even at m = 4. Older lineages' D21 feedback into covariance is
   essential, not a small correction.

## Price-screen conclusion for the fallback

Every measured way to cheapen recent or old history in this representation loses accuracy
far faster than it saves cost. The r256/age5 point is a local score optimum in both rank
directions (RESULTS Sep 26) and in age. The largest measured structural saving with a usable
answer is −12.8%, and it cannot plausibly support a 15% score gain. Stopped at the price
screen, per PLAN E5.

**Missing evidence for a ≥15% route:** a representation of the third cumulant whose dense
(2,1)-slice readout is materially cheaper than one n × (lineage columns) × n contraction per
lineage per layer, while retaining ≳256-dimensional content per mode. None of the tested
carriers provides that. The observed leader (0.108B, raw 1.616e-8) indicates one exists, but
its mechanism is unknown.
