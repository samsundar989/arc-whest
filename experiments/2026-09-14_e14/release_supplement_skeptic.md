## Skeptic verdict: PROVISIONAL

**Strongest challenge**  The first optional numerical consumer accepted malformed receipts because it duplicated only a subset of the canonical validator. That code defect is now resolved; the affected final preflight remains pending.

**Evidence for it**  The initial seven independent semantic mutations were accepted by the first supplement. The reviewed repair, `local_sim/candidate_release_record.py` SHA256 `303be0a7220a9e1b43b263578edba39d04995a35a2d91790144c24b334c906d4`, now delegates to `certify_candidate.validate_edge_case_receipt`, checks the runner source, binds referenced receipt hashes, and compares embedded/file JSON including types.

**Evidence for the original claim**  Independent read-only checks accepted the actual pinned numerical summary and exactly regenerated `f21_release_record_v3.json` from `f21_release_recipe_v3.json`. Ten independent semantic mutations now refuse: runner hash, edge-source hash, fake backend, integer result, receipt hash, receipt case, missing metered predictions, wrong denominator, wrong shape, and duplicate case. For semantic receipt mutations, mocked file contents and their hashes were changed together, so the canonical validator was tested beyond embedded/file equality. The caller's stricter environment setting was restored after every check. No candidate execution, protected truth, source edit, or numerical rerun was used. The repaired index reports only THREE_TARGET_PERMUTATION_CASES_PASS, retains LOCAL REFUSED and SERVER_MEASURED PASS with PROVISIONAL exact-byte attribution, and keeps normal_release_ready=false.

**Unresolved ambiguity**  The supplement was added after workerq 1222's accepted source freeze. That preflight does not cover this addition. The affected final default preflight must finish before combined final-tree acceptance. The accepted numerical result remains limited to three declared target-shape permutation cases, not full numerical, deployment, or accuracy certification.

**Cheapest discriminating follow-up**  Inspect the already planned final queued default preflight and its exact before/after source maps. No seed block, labels, or additional candidate execution is needed for this consumer repair.

**Checklist items triggered**  14, 17, 18.

**Blocking code fixes remaining:** none in this optional consumer. Final affected preflight remains the procedural acceptance condition. Prior Stage 0 acceptance and the independently reviewed underlying numerical evidence remain intact.
