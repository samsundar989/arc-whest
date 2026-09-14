# Competition review

Read in Samuel's Chrome on September 14, 2026, approximately 07:20–07:40 UTC. These are transcribed visible-page observations, not full-precision downloaded competitor reports. All leaderboard MSEs below have an **unavailable official reference-noise floor**. Public score observations are CONFIRM/context only and must not select or fit a candidate.

## Current position

| Phase / entry | Visible adjusted score | Visible final MSE | Other observation |
|---|---:|---:|---|
| Phase 2 leader suliman_tadros, #330876 | board 2.50e-9; detail 2.540e-9 | detail 1.891e-8 | mean budget 13.43%; public rows vary approximately 2.77e11–3.11e11 FLOPs |
| Phase 2 marius_binner | 2.60e-9 | 1.70e-8 | rank 2 |
| Phase 2 J2W | 2.70e-9 | 1.93e-8 | rank 3 |
| Phase 2 Puffi | 2.80e-9 | 1.94e-8 | rank 4 |
| Phase 2 504aldo | 5.40e-9 | 2.13e-8 | tied rank 26 |
| Phase 2 neopok919, #330950 F21 | detail 1.596e-8 | detail 3.779e-8 | tied rank 58 of 156; 14 Phase-2 entries; latest Sep14 05:41 UTC |
| Phase 1 ednacob | 1.84e-8 | 3.63e-8 | public rank 1; completed phase |
| Phase 1 J2W | 5.39e-8 | 2.78e-7 | public rank 2 |
| Phase 1 dstepanov | 6.22e-8 | 2.20e-7 | public rank 3 |

Phase 1 has 312 board entries and our account rank 103. Its scores must not be directly compared with Phase 2: network dimensions and accounting differ. The Phase-2 detail page improves the precision of the leader target; the board's 2.50e-9 and detail's 2.540e-9 are a display precision difference, not a new method result. The Insights page independently displays best adjusted 2.54e-9 and best raw MSE 1.50e-8; these need not belong to the same submission. At that reading it also showed 939 registered participants and 21,297 total submissions across rounds. These activity counts do not measure distinct competitive methods.

Sources: [Phase 1 board](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards?round=phase-1), [Phase 2 board](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards?round=phase-2), [leader detail](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/330876), [our latest detail](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/330950), [Insights](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/insights).

## Latest submission

F21 is visibly graded successfully, 50/50 public MLPs completed and zero public failures. The ledger heading says 100 scored. Private accuracy remains sealed. Its Configuration tab reports WhestBench 0.16.1, flopscope client/server 0.12.1, participant shim 0.1.0 and Python 3.10.21; smoke test passed, 11 worker passes and no worker failures. Public mean cost is 42.22% of budget. The archived official report, rather than rounded UI cells, remains the authority for exact F21 timing and counts: `experiments/run_reports/f21_diagnostic_330950_graded_report.json`.

The inspected submission pages expose results and configuration, not a downloadable competitor source archive. Therefore the leader's algorithm is unknown. Different per-network FLOPs show different executed work; they do not prove adaptive rank, a particular cumulant closure, or any specific implementation. No error payload was used to reconstruct competitor code. The submission detail's generic prose implies its sealed 50 decide final rank; the official Rules and Overview explicitly specify a separate fresh private rerun. Use the latter as authority.

## Rules and schedule

Rules v14 remains current on the inspected page. Phase 2 is width 1024, depth 16, budget 2^41 FLOPs per network, lambda zero, residual cap 0.4 seconds, total wall cap 120 seconds and import/setup cap 5 seconds. The participant process has an 8-GB published allowance; installed enforcement uses 8,192 MiB. The process is CPU-only, one physical core / two vCPUs, with a separately metered backend. Final prize ranking uses up to two selected entries rerun on fresh networks; absent selection the two best valid public entries are used. Registration/team freeze is October 2, submissions October 17 at 23:59 UTC, and contribution write-up October 24. The generic October 31 competition-period prose does not extend the specific Phase-2 submission deadline.

Public resources and AI assistance are permitted, while source authenticity also excludes derivation from another participant's submission. The interaction remains unresolved for a particular derivative. A public MIT release and a successful grade do not themselves settle prize eligibility. The existing unsent clarification in `docs/ELIGIBILITY_REVIEW_2026-09-12.md` remains relevant. No message was sent, rules accepted, submission made, or entry designated during this review. [Rules](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/challenge_rules), [Overview](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026), [Resources](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/dynamic_contents).

Rules v14 §6 also requires each contribution-prize write-up to correspond to exactly one of the participant's submissions successfully evaluated for that phase; the entry need not rank among score-prize winners. This was rechecked directly in Chrome during the independent review. It is a requirement on the write-up's associated submission, not a restriction to one algorithm explored during research. [Rules §6](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/challenge_rules).

## Discussion findings that affect strategy

All 29 titles available in the competition's Discussion index were expanded and scanned. The following bodies were read directly; this is not a claim to have read every historical attachment or watched every linked talk.

1. **504aldo factorized K3 release, topic 18218, now three posts.** Its source-linked history reports that regenerated fourth-order structure and K3 birth feedback improved accuracy, while later shared-basis and arithmetic changes reduced cost. These are author measurements on reused development/public panels, not our independent DECIDE evidence. Its broad claims excluding all cheap representations exceed what the specific failed designs establish. On September 14 the author reports others have reduced cost another 7–22% and says the sponsor confirmed on September 11 that metered Strassen-Winograd is permitted without a recursion-depth limit. The latter is a participant report of sponsor correspondence; the original sponsor message was not inspected. No new winning architecture is disclosed. [Original post](https://discourse.aicrowd.com/t/18218/1), [new reply](https://discourse.aicrowd.com/t/18218/3).
2. **Local residual versus grader, topic 18202.** Read all five posts, including the author's corrections. The final technical correction attributes failures to client scalar conversions that flush pending computation; removing 166 conversions changed a reported 5/50 failure to 50/50 completion at the same approximate call count. This supersedes that thread's earlier call-count/payload-only explanation. Large batched request failures remain an unanswered later concern. [Corrected mechanism](https://discourse.aicrowd.com/t/18202/4), [remaining request-size question](https://discourse.aicrowd.com/t/18202/5).
3. **Calibration interpretation, topic 18216.** The author separates local one-step error from accumulated propagation drift, retracting an earlier claim that fitted constants simply represented local skewness. This is a valuable test-design warning, not a frontier score: reported score is around 1.07e-7. [Write-up discussion](https://discourse.aicrowd.com/t/18216).
4. **Exact bivariate covariance, topic 18214.** The author reports only 7.4% raw improvement from a more accurate Gaussian pairwise calculation, and a much larger gain from output calibration on that baseline. This supports prioritizing missing joint higher-order structure over further polishing Gaussian-only moments; it does not quantify that tradeoff on F21. [Technical post](https://discourse.aicrowd.com/t/18214).
5. **Error-message information leak, topic 18217.** AIcrowd acknowledged global status messages could expose details and said it would fix them. This provides no authorized method disclosure or scientific evidence about competitors. [Organizer response](https://discourse.aicrowd.com/t/18217/2).
6. **Phase-2 dataset difficulty, topic 18199.** A participant reports much less score skew than Phase 1 on covariance-propagation panels, including the full public split. This cautions against carrying Phase-1 tail statistics forward unchanged. It does not prove absence of tail failures for a new adaptive estimator or exact equality between local and private scores. [Discussion](https://discourse.aicrowd.com/t/18199).
7. **Organizer talks, topic 18206.** The thread links two talks and contains no further technical result in its three posts. Their written companion paper is the primary technical authority used in the literature review. The videos were not watched in this audit. [References](https://discourse.aicrowd.com/t/18206).

Public release availability has improved; the main uncertainty is still the complete cost-and-accuracy tradeoff of a locally correct architecture. A repeated rounded score across accounts cannot establish source identity, authorship, or eligibility.
