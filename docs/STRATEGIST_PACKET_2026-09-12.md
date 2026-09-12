# Strategist Packet — causal compact nonlinear source for ARC Whest Phase 2

**Private research packet, September 12, 2026.** Address this without repository
access or chat history. We need a falsifiable architecture decision, not a
parameter fit or an open-ended literature survey. The evidence below is
scoped; no source or score improvement is established.

## Task and operating limits

For each random He-init, bias-free width-1024 depth-16 ReLU MLP, predict each
neuron's expected post-ReLU activation for Gaussian input, an array of shape
`(16,1024)`. The private score is the mean over MLPs of final-layer MSE times
`max(0.1, FLOPs/2^41)`, lower is better. Phase 2 has lambda zero: the
instrumented FLOP count, not residual time, is the cost factor. A residual
time over 0.4 s per MLP fails that MLP; memory limit is 8 GiB, not enforceable
on our Windows host. Official local stack: WhestBench 0.16.1/flopscope 0.12.1.
The grader file may import only flopscope, its numpy facade, whestbench and
stdlib, and arrays are immutable. There is no license to route math through
uninstrumented raw arrays.

Our standing robust L2 public candidate #330566 has adjusted score about
`2.284e-8`, 0/50 public failures and dated worst displayed wall 33.0 s;
the best public L4 #330290 is about `1.836e-8` with dated worst displayed
wall 114.68 s. The public leader is much better, but its method is unknown.
Private rerun selects up to two designated submissions; no submission or
designation is currently authorized. An independent contribution write-up
is a second objective. Preserve the existing candidate, shipped estimator,
sealed vault and human-owned model file.

## What is actually known

The production certification/measurement instrument has independent acceptance.
An old compact *supplied-core* all-history K3/K4 reference at width1024/depth16
measured **818,494,496,296** official-package FLOPs, about 1,014 MiB process
peak. This is a cost/algebra reference for its supplied core, not a physical
K4 source or deployable prediction. The four frozen K4 arms all failed a
pre-registered accuracy gate on already exposed P01 **DESIGN** data. Their
source-cumulant implementation has an independent defect; this does not kill
the coherent K4 hypothesis. P07 is also exposed reusable DESIGN; P08 is
exhausted. No untouched fresh DECIDE block has been spent on a corrected arm.

For a fixed physical two-coordinate cascade, independent continuous
integration and a fresh Skeptic accepted exact **signed-law accounting**:

```
Z~N(0,[[1,.5],[.5,1]]), X=ReLU(Z),
B=[[.8,-.3],[-.55,1.4]], D=[[-.65,1.1],[.9,-.4]],
Y=ReLU(BX), T=ReLU(DY).
E[X_1 X_2]=1/6+sqrt(3)/(4*pi).
```

```
P_l = F_l#P_(l-1),  F_l(x)=ReLU(W_l x)
R_l = ReLU#N(0, E[(W_l X_(l-1))(W_l X_(l-1))^T])
Delta_l = F_l#R_(l-1) - R_l
P_l = R_l + sum_{a=2}^l F_l#...#F_(a+1)#Delta_a.
```

The Gaussian reset uses the **second raw preactivation moment**, not centered
covariance. For layers 2/3, `Delta2=P2-R2`, `Q3=F_D#R2`,
`Delta3=Q3-R3`, `O3=F_D#Delta2=P3-Q3`. The conditional fourth-cumulant
increments `kappa4(Q3)-kappa4(R3)` and `kappa4(P3)-kappa4(Q3)` telescope;
one cannot add a standalone cumulant of a mass-zero signed measure. The
conditional difference includes first-, second- and third-raw-moment cross
terms via the set-partition formula. On the fixed cascade, nonlinear old
versus linear K4 transport differs by **0.32645** against old K4 norm
**0.57230**; the signed linear-map control is `3.3e-16`. Old signed lower raw
moments reach **0.91379** and conditional K4 differs from bare fourth-raw
contrast by **1.81510**. Independent root enumeration, all continuous
moments and oriented K3/K4 checks passed; current-byte default preflight
passed 12/16 with four deployment-only skips. The finite angular bank
looked ahead to future map boundaries; exact symbolic signed kernels can
defer splitting, but retain a whole distribution. None of this proves a
width1024 finite-rank source.

An integrated synthetic first-source program puts one S64 discrete K4 bank
and 14 induced K3 births into ordinary compact all-history weighted-Gram/
thin-QR state, in one flopscope context with all K4/K3 mean and covariance
consumers. Its frozen n4 dense check **failed 93/172** from an unstable
trajectory: by L10 preactivation variance was `2.66e-8` while |D4| was
`0.373`, leading to a K4 mean correction `1.60e10` and later overflow.
A separate full-rank n8 test matched dense tensor/Taylor algebra to
`2.41e-6` material-relative and met schedule/count assertions, but **failed
186/187** because L3 preactivation variance was **-0.117716**. Removing the
K4 covariance term would make it **+0.033250**; the K4 contribution alone
is **-0.150966**. The external bank carries linearly through raw W, and
is not a physical later nonlinear source. We did not run or price width1024
on this invalid path.

One exact efficiency repair is independently accepted: the positive K3
feed Source uses S columns rather than padding to n, skips zero-A work,
and column-scales the diagonal negative-A birth. The negative covariance
Source still needs literal n-column A/P. The repaired tiny result passed
789/789 and reduced its counted FLOPs **3,936,609 to 3,213,553 (18.37%)**;
the n8 negative variance is identical before/after. This ratio is not a
width1024 cost estimate. Per-layer modeled resident bytes omit co-live
temporaries, and neither those numbers nor preflight's ~3 GiB process peak
prove an 8 GiB deployed limit.

A subsequent separate shape-only audit now enumerates the integrated
prototype's retained source fields and nine sampled Python-frame
boundaries. After two independently caught missing-lifetime defects,
fresh Skeptics accepted the n8 and thinner n16/rank8/residual2 traces
only in that scope (63/63 and 93/93). At n1024/S64/rank384 the
*inclusive named-array payload model* is 715,844,864 bytes with the
positive S-column leg versus 910,988,032 padded; the negative full
n-column leg remains. This symbolic count excludes unbounded helper,
library, allocator and process overhead. No stable full-width integrated
run has measured an actual peak or complete counted FLOPs, and the 8 GiB
limit has not been tested. Raw evidence and verdicts are at the LOG tail.

## The decision we need from you

1. Derive the smallest **causal, compressed** state that could carry an older
   signed ReLU deviation through another nonlinear layer while retaining
   the lower raw moments needed for conditional K3/K4. State exactly what
   distributional information is lost, how that loss biases final expected
   activations, and whether a finite cumulant closure can be justified at all.
   A fixed-rank assumption alone is not an answer.
2. Give one concrete architecture with update/readout equations supported by
   flopscope-like dense matmul/elementwise/QR primitives. Include source
   construction, all old-source transport, negative full-rank costs,
   covariance stability or PSD handling, joint contractions, counted-FLOP
   scaling at n=1024,d=16, residual-time risk and co-live memory. Explicitly
   distinguish an exact representation from a controlled approximation.
3. Specify the **cheapest truth-free falsifier first**: an independently
   integrated n2/n4 nonlinear two-birth dense reference, a stable
   full-depth tiny input with negative/omitted-source controls, and a
   preregistered official-context full-width cost gate. What exact result
   should stop implementation before any DESIGN or fresh DECIDE label?
4. If no credible architecture fits `2^41` FLOPs and 8 GiB while preserving
   enough information, say so. Recommend a bounded fallback using the
   standing L2 candidate and an honest contribution write-up, rather than
   proposing an unpriced sweep or post-hoc fit.

Do not treat P01/P07 as untouched decision data, request the sealed vault,
reopen eligibility/submission work, or tune theta/lambda before conformance.
Return equations, a cost table with overlapping work charged once, one
falsifier and a clear go/no-go threshold. We will preregister any chosen
experiment and obtain an independent Skeptic before believing it.
