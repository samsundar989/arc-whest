## Skeptic verdict: PROVISIONAL

**Strongest challenge**  The first optional numerical consumer accepted malformed receipts because it duplicated only a subset of the canonical validator. That code defect is now resolved; the affected final preflight remains pending.

**Evidence for it**  The initial seven independent semantic mutations were accepted by the first supplement. The reviewed repair, `local_sim/candidate_release_record.py` SHA256 `303be0a7220a9e1b43b263578edba39d04995a35a2d91790144c24b334c906d4`, now delegates to `certify_candidate.validate_edge_case_receipt`, checks the runner source, binds referenced receipt hashes, and compares embedded/file JSON including types.

**Evidence for the original claim**  Independent read-only checks accepted the actual pinned numerical summary and exactly regenerated `f21_release_record_v3.json` from `f21_release_recipe_v3.json`. Ten independent semantic mutations now refuse: runner hash, edge-source hash, fake backend, integer result, receipt hash, receipt case, missing metered predictions, wrong denominator, wrong shape, and duplicate case. For semantic receipt mutations, mocked file contents and their hashes were changed together, so the canonical validator was tested beyond embedded/file equality. The caller's stricter environment setting was restored after every check. No candidate execution, protected truth, source edit, or numerical rerun was used. The repaired index reports only THREE_TARGET_PERMUTATION_CASES_PASS, retains LOCAL REFUSED and SERVER_MEASURED PASS with PROVISIONAL exact-byte attribution, and keeps normal_release_ready=false.

**Unresolved ambiguity**  The supplement was added after workerq 1222's accepted source freeze. That preflight does not cover this addition. The affected final default preflight must finish before combined final-tree acceptance. The accepted numerical result remains limited to three declared target-shape permutation cases, not full numerical, deployment, or accuracy certification.

**Cheapest discriminating follow-up**  Inspect the already planned final queued default preflight and its exact before/after source maps. No seed block, labels, or additional candidate execution is needed for this consumer repair.

**Checklist items triggered**  14, 17, 18.

**Blocking code fixes remaining:** none in this optional consumer. Final affected preflight remains the procedural acceptance condition. Prior Stage 0 acceptance and the independently reviewed underlying numerical evidence remain intact.

## Final independent addendum: ACCEPT

The pending procedural condition is satisfied. Independently inspected workerq 1230's successful exit-0 receipt and terminal log, `preflight_final/receipt.json`, both source maps, and `preflight_final/verdicts.txt`. The default gate reports GREEN 12/16 with four explicitly skipped Full checks. Both maps contain 309 identical entries, and every recorded source still matches disk bytes at review. The map binds the repaired consumer to the exact SHA256 reviewed above, `303be0a7220a9e1b43b263578edba39d04995a35a2d91790144c24b334c906d4`.

Final receipt SHA256: `6402404e93cb9220c2bc20b63714695ae2fa1624e2f4d7c916b6d69362a2d622`. Each source-map file SHA256: `1bc0318af60c0c34877e36a19648bce7547cff267206910bcc3dd388356fdff5`. Verdict transcript SHA256: `f8a2e9c23c23f3fdbcc8c804007b6165abff663be13b35d9727552fbd321af3a`.

No supplement blockers remain. ACCEPT is limited to the optional evidence consumer and its declared three-case index. It does not establish candidate Full certification or override F21 LOCAL REFUSED, PROVISIONAL server-byte attribution, or `normal_release_ready=false`. No additional tests, labels, or candidate execution were used for this final review.
