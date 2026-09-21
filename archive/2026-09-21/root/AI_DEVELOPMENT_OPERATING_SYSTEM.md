# AI Development Operating System
## Human-Comprehension-Preserving Agentic Engineering for Claude Code, Codex, and Research Agents

> **Purpose:** Maximize validated AI-assisted development velocity **without allowing implementation velocity to outrun human understanding, evaluation quality, or strategic control**.
>
> Use this file together with `AI_PROJECT_STACK_SELECTOR.md`.  
> The stack selector decides **which tools and methodology this repository needs**.  
> This document decides **how humans and agents should work once the stack is selected**.

---

# 0. QUICK INVOCATION

Use this prompt when bootstrapping or re-auditing a repository:

```text
Read `AI_DEVELOPMENT_OPERATING_SYSTEM.md` and `AI_PROJECT_STACK_SELECTOR.md`, then inspect this repository, its current task/competition/product requirements, existing agent configuration, git history, tests, evaluators, experiment records, documentation, and tooling.

Treat both files as decision frameworks, not install lists.

Your job is to establish the smallest project-specific development harness that maximizes validated progress while preserving human strategic understanding.

First audit the repository and classify:
- the project mode(s);
- the current bottleneck;
- evaluator trustworthiness;
- current architecture/comprehension debt;
- the appropriate work classification policy;
- the minimum effective Claude/Codex/tool stack.

Then propose and make only non-destructive repository/configuration changes needed to establish the project-specific harness.

Requirements:
1. Choose one dominant development methodology.
2. Prefer existing scripts -> CLI -> Skill -> narrow MCP -> orchestration framework.
3. Do not enable autonomous iteration until evaluation is trustworthy.
4. Keep `AGENTS.md` / `CLAUDE.md` concise; move deeper knowledge into durable docs and Skills.
5. Establish explicit ROUTINE / IMPORTANT / STRATEGIC work classes.
6. For strategic work, enforce:
   orient -> hypothesize -> predict -> implement -> evaluate -> falsify -> synthesize -> human teach-back -> next decision.
7. Build or repair evaluation before scaling agent parallelism.
8. Separate implementation from independent review.
9. Require evidence packets rather than self-reported success.
10. Make the repository easier for a human to understand after every strategic change.
11. Create a human-owned mental model file that agents may read and critique but may not silently rewrite.
12. Add hooks/tests only where they enforce genuinely important invariants.
13. Explicitly reject tools that do not remove a current bottleneck.
14. Do not modify core solution logic during the bootstrap unless required to repair evaluation or reproducibility and clearly justified.

Before changing files, output a concise audit and proposed harness. Then implement the obvious non-destructive setup changes and leave a short "How to work in this repo" summary.

Do not optimize for process ceremony. Optimize for:
validated progress per human hour,
reproducibility,
failure visibility,
strategic learning,
and the human operator's ability to direct the next high-value experiment.
```

---

# 1. THE OPERATING PRINCIPLE

AI can now generate, search, refactor, experiment, and parallelize faster than a human can read the resulting work.

Therefore:

> **The bottleneck is no longer code production. It is evaluation, synthesis, and strategic understanding.**

The goal is **not** to slow the AI down until the human can follow every line.

The goal is to ensure the human remains **epistemically in control** of the critical path:

- understands what the system is optimizing;
- understands the major architecture and data/control flow;
- knows the important assumptions and invariants;
- can explain why the current best approach should work;
- can recognize major failure modes;
- can judge whether evidence supports a claimed improvement;
- can identify what to investigate next;
- can generate or evaluate novel hypotheses.

AI should own most mechanical execution.

The human should increasingly own:

```text
problem framing
-> causal model
-> hypothesis selection
-> evidence standards
-> tradeoffs
-> strategic direction
```

---

# 2. THE CORE EQUATION

Treat project progress as:

```text
Useful Progress
=
Implementation Velocity
x Evaluation Reliability
x Knowledge Integration
x Strategic Quality
```

If any factor approaches zero, raw coding speed creates little durable advantage.

This implies four separate systems must exist:

1. **Execution system** — agents can modify the project quickly.
2. **Evaluation system** — the project can objectively tell whether changes helped.
3. **Knowledge system** — durable project understanding improves with each important change.
4. **Strategy system** — hypotheses and experiments are selected deliberately rather than through random patch accumulation.

---

# 3. NON-NEGOTIABLE PRINCIPLES

## 3.1 The evaluator, not the agent, decides whether work succeeded

Never accept:

```text
"I implemented it and it looks good."
```

as evidence.

Prefer:

```text
exact command
+ exact configuration
+ exact commit
+ raw outputs
+ baseline comparison
+ regressions
+ runtime/cost
+ uncertainty
```

If the evaluator is weak, improving it takes priority over adding more autonomous agents.

---

## 3.2 Understanding is a workflow output

"Understand the code later" is not a valid plan.

For strategically important work, human understanding must be produced as part of task completion.

Do not require the human to understand every generated line.

Require understanding of the **competitive/product/scientific critical path**.

---

## 3.3 Environment engineering beats repeated prompting

If an agent makes the same mistake twice, do not keep reminding it conversationally.

Promote the lesson to the lowest durable layer that solves it:

```text
Repeated fact / invariant
    -> AGENTS.md or durable doc

Repeated workflow
    -> Skill

Repeated mechanical check
    -> script / test / linter

Must-never-happen behavior
    -> hook / permission / CI rule

Repeated debugging blind spot
    -> trace / visualization / fixture / evaluator improvement

Repeated human misunderstanding
    -> SYSTEM_MAP / concepts note / targeted teach-back
```

This is the **two-strike escalation rule**.

---

## 3.4 AI concurrency is not human cognitive concurrency

Agents may work in parallel.

Human strategic attention should remain deliberately constrained.

Across all active projects, distinguish:

- **PRIMARY COGNITIVE PROJECT** — architecture-changing work receives active human comprehension.
- **SECONDARY PROJECT** — limited strategic changes; mostly well-scoped work.
- **MAINTENANCE / AUTONOMOUS PROJECT** — only tasks with strong automated verification unless promoted.

Do not interpret "I can run 10 agents" as "I should personally track 10 evolving architectures."

---

## 3.5 One strategic hypothesis per slice

Do not combine multiple independent strategic ideas into a single experiment unless unavoidable.

A result that changes many variables teaches little.

Prefer:

```text
one hypothesis
-> smallest discriminating intervention
-> measurable prediction
-> result
-> belief update
```

---

## 3.6 Fresh review beats self-review

The implementation agent must not certify its own strategically important work.

Use:

```text
Strategist
-> Builder
-> Independent Skeptic
-> Evaluator
-> Human synthesis
```

The same model may fill multiple roles only when contexts are independent and prior conclusions are not blindly inherited.

---

## 3.7 Research must change a decision

Research is not progress merely because many sources were collected.

Persist research when it:

- changes a hypothesis;
- identifies a new mechanism;
- invalidates an assumption;
- changes implementation priority;
- provides a useful baseline;
- reveals a competitor/open-source approach;
- identifies an evaluation flaw;
- creates a new experiment.

Avoid repeatedly researching the same question because chat context was lost.

---

# 4. WORK CLASSIFICATION

Every meaningful task is classified before execution.

## 4.1 ROUTINE

Examples:

- formatting;
- straightforward tests;
- logging;
- deterministic refactors;
- renaming;
- packaging;
- type fixes;
- boilerplate;
- mechanical data conversions;
- simple CI work;
- obvious bug fixes with strong regression tests.

Policy:

```text
AI autonomy: HIGH
Plan: optional
Human checkpoint: none
Verification: required
Knowledge artifact: usually none
Independent review: only if risk warrants it
```

Do not create process overhead for ROUTINE work.

---

## 4.2 IMPORTANT

Examples:

- new subsystem;
- non-trivial performance change;
- data pipeline change;
- evaluator change;
- significant integration;
- architecture-affecting refactor;
- meaningful product behavior.

Policy:

```text
AI autonomy: HIGH
Plan: concise
Prediction: required
Verification: required
Experiment/decision record: usually required
Knowledge delta: required
Independent review: when consequential
Human checkpoint: short when critical-path understanding changes
```

---

## 4.3 STRATEGIC

Examples:

- competition strategy;
- new search/planning algorithm;
- model choice;
- representation change;
- evaluation methodology;
- data split/validation strategy;
- opponent modeling;
- reward/scoring logic;
- agent/tool policy;
- security attack/defense strategy;
- product wedge;
- major architecture choice;
- scientific hypothesis;
- anything intended to create differentiated advantage.

Policy:

```text
AI execution autonomy: HIGH after slice contract
Human strategic involvement: REQUIRED
Pre-implementation prediction: REQUIRED
Independent review: REQUIRED unless evaluator makes review redundant
Objective evidence: REQUIRED
Knowledge synthesis: REQUIRED
Human teach-back: REQUIRED
```

A STRATEGIC slice is not complete merely because the code works.

---

# 5. REQUIRED HUMAN UNDERSTANDING DEPTH

Do not demand equal understanding everywhere.

Use four levels.

## LEVEL 1 — Interface

The human knows:

- purpose;
- inputs;
- outputs;
- external contract.

Use for commodity infrastructure.

## LEVEL 2 — Mechanism

The human can trace one representative execution path.

Required for critical runtime paths.

## LEVEL 3 — Modification

The human can:

- predict effects of a meaningful change;
- identify likely failures;
- find where to debug;
- explain important assumptions.

Required for competition/product differentiators.

## LEVEL 4 — Innovation

The human can:

- generate materially different approaches;
- compare mechanisms;
- reason about tradeoffs;
- design discriminating experiments.

Required only for the few areas where novel ideas create strategic advantage.

The project harness should explicitly identify which components need Levels 2-4.

---

# 6. STRATEGIC SLICE LIFECYCLE

Use this lifecycle for STRATEGIC work and a lightweight form for IMPORTANT work.

```text
ORIENT
  ↓
HYPOTHESIZE
  ↓
PREDICT
  ↓
PLAN MINIMAL EXPERIMENT
  ↓
IMPLEMENT
  ↓
EVALUATE
  ↓
FALSIFY / ADVERSARIAL REVIEW
  ↓
SYNTHESIZE KNOWLEDGE
  ↓
HUMAN TEACH-BACK
  ↓
UPDATE BELIEFS
  ↓
SELECT NEXT SLICE
```

No direct:

```text
implementation result
-> another architecture-changing implementation
```

without synthesis.

---

# 7. THE SLICE CONTRACT

Before strategic implementation, create a compact contract.

Recommended format:

```markdown
# Slice Contract

## Strategic question
What exactly are we trying to learn or improve?

## Current bottleneck
What currently limits performance/progress?

## Evidence
What do we currently know?

## Hypothesis
What do we believe is true?

## Mechanism
Why should the proposed change produce the expected result?

## Relevant system path
What components/interfaces are involved?

## Human concepts required
What must the operator understand to evaluate this slice?

## Proposed intervention
What is the smallest meaningful change?

## Prediction
What measurable result should occur if the hypothesis is correct?

## Falsifier
What result would weaken or reject the hypothesis?

## Baseline
What is the current comparison point?

## Evaluation
What exact command/protocol decides the result?

## Confounders
What else could explain a positive or negative result?

## Scope
What files/components are expected to change?

## Do not change
What variables must remain fixed?

## Evidence required from builder
What logs, metrics, artifacts, diffs, traces, seeds, or outputs must be returned?
```

Keep it short enough that it is actually used.

---

# 8. PRE-IMPLEMENTATION HUMAN CHECKPOINT

For STRATEGIC work, the agent must ask the human to predict before revealing or implementing the final conclusion when interactive workflow permits.

Use at most 2-3 short questions:

```text
1. What part of the system do you think this changes?
2. What behavior do you expect to improve?
3. What result would convince you the hypothesis is correct?
```

Short answers are sufficient.

"I don't know" is a valid and useful answer.

When the human does not know:

1. teach only the missing concept;
2. use real project files/paths/examples;
3. show one representative execution trace if possible;
4. return immediately to the task.

Do not turn every checkpoint into a long tutorial.

Target human time for a normal checkpoint:

```text
~2-5 minutes
```

---

# 9. IMPLEMENTATION POLICY

The Builder receives the Slice Contract.

The Builder must:

1. inspect the current implementation;
2. verify assumptions against code;
3. minimize changed variables;
4. add or repair tests/evaluation where needed;
5. implement the smallest useful intervention;
6. run objective verification;
7. record exact evidence;
8. avoid silently broadening scope;
9. flag discoveries that invalidate the original contract.

If the evidence contradicts the hypothesis, do not disguise the failure.

A failed hypothesis with clear evidence is useful progress.

---

# 10. EVIDENCE PACKET

Builders return evidence, not merely prose.

Required where applicable:

```text
git commit / diff
exact commands executed
configuration
seed(s)
dataset / environment version
test results
evaluation results
baseline
new metric
delta
regression count
new success count
runtime
memory / cost
logs / trace locations
screenshots / artifacts
failures encountered
unexpected behavior
remaining uncertainty
```

The Strategist or reviewer should inspect the evidence directly whenever possible.

Do not make one model judge another model only from the implementer's narrative summary.

---

# 11. THE SKEPTIC / FALSIFICATION ROLE

For strategically important changes, use a fresh context.

Default prompt:

```text
Assume the claimed improvement or conclusion is wrong.

Inspect the Slice Contract, implementation, diff, evaluation protocol, and raw evidence.

Try to explain the result through competing causes before accepting the stated mechanism.

Check especially for:
- invalid baseline;
- leakage;
- overfitting;
- random-seed luck;
- benchmark contamination;
- hidden configuration changes;
- implementation bugs;
- stale state;
- unit/frame errors;
- timing/race issues;
- hidden coupling;
- data distribution shift;
- prompt/provider changes;
- evaluator bugs;
- cherry-picked cases;
- regression masking;
- runtime/cost tradeoffs;
- security scope violations where applicable.

Return:
1. strongest challenge to the claim;
2. evidence supporting the challenge;
3. evidence supporting the original hypothesis;
4. unresolved ambiguity;
5. cheapest discriminating follow-up experiment;
6. verdict: ACCEPT / PROVISIONAL / REJECT.
```

Independent review should attempt to **falsify**, not merely praise or restate.

---

# 12. POST-IMPLEMENTATION HUMAN TEACH-BACK

After a strategic slice, before the next strategic implementation, test understanding.

Do not ask the human to explain every file.

Ask 2-4 questions such as:

```text
1. Why did this change improve or fail?
2. What component is most causally important?
3. What assumption remains uncertain?
4. If the metric regressed tomorrow, where would you inspect first?
```

If the human cannot answer:

- identify the specific missing concept;
- explain only that concept;
- ask one transfer question;
- then continue.

This is a **comprehension gate**, not an exam for its own sake.

---

# 13. KNOWLEDGE DELTA

Every IMPORTANT or STRATEGIC slice should produce:

```markdown
## Knowledge Delta

### Learned
What new fact or mechanism is now supported?

### Changed belief
Which prior assumption was strengthened, weakened, or rejected?

### New invariant
Did we discover something that must remain true?

### New failure signature
Did we learn how a class of failures presents?

### Remaining uncertainty
What is still unresolved?

### Documentation impact
Which durable file should change, if any?

### Next strategic implication
What does this change about the next experiment?
```

If there is no meaningful knowledge delta, do not manufacture one.

---

# 14. DURABLE REPOSITORY KNOWLEDGE

Recommended structure, customized rather than blindly created:

```text
AGENTS.md
CLAUDE.md

docs/
  NORTH_STAR.md
  CURRENT_STATE.md
  SYSTEM_MAP.md
  ARCHITECTURE.md
  EVALUATION.md
  INVARIANTS.md
  FAILURE_MODES.md
  HUMAN_MODEL.md

  hypotheses/
  experiments/
  research/
  decisions/
  concepts/

scripts/
  setup
  test
  eval
  benchmark
  reproduce

.claude/
  skills/
  agents/
  settings.json
```

Do not create empty files for appearance.

---

# 15. TOP-LEVEL AGENT INSTRUCTIONS

`AGENTS.md` / `CLAUDE.md` should remain concise.

They should contain only:

- one-sentence objective;
- critical invariants;
- setup command;
- test command;
- evaluation command;
- important directories;
- dominant workflow;
- work classification rules;
- pointers to deeper docs;
- enabled tools;
- explicitly disabled tools;
- critical safety/scope restrictions.

They are maps and contracts, not encyclopedias.

When both Claude and Codex are used, prefer one canonical shared instruction source and make the other file point/import it where supported.

---

# 16. HUMAN_MODEL.md

This file is special.

It represents **what the human operator currently understands and believes**, not what the AI says the human should believe.

Suggested template:

```markdown
# My Mental Model

## Objective
What is this project trying to optimize?

## System from memory
Draw or describe the major flow.

## Current bottleneck
What do I believe limits progress right now?

## Why
What evidence makes me believe that?

## Current strongest hypothesis
What do I currently think will unlock progress?

## Dangerous assumptions
1.
2.
3.

## Things I do not understand yet
1.
2.
3.

## Current failure model
What are the most likely ways the system can fail?

## Next experiment
I expect ______ because ______.

## Confidence
What am I confident about vs uncertain about?
```

Agents may:

- read it;
- quiz against it;
- point out contradictions;
- propose updates.

Agents should **not silently rewrite it**.

If the environment supports deterministic hooks/permissions, protect this file from agent writes.

---

# 17. SYSTEM_MAP.md

The human should be able to reproduce the project's critical-path map from memory.

Prefer simple diagrams such as:

```text
input
  ↓
parsing / representation
  ↓
core decision / model / algorithm
  ↓
action / output
  ↓
environment / user / evaluator

                 EVALUATION PATH
candidate ──┐
baseline ───┼──> evaluator ──> metrics ──> experiment record
holdout ────┘
```

Customize this for the project.

The goal is not perfect documentation.

The goal is a compact causal model.

---

# 18. EVALUATION TRUST LEVELS

Before agent autonomy, classify the evaluator.

## E0 — Untrusted

Symptoms:

- manual judgment;
- unreproducible;
- metric mismatch;
- unknown leakage;
- flaky outputs;
- no baseline;
- no held-out set;
- hidden environment differences.

Policy:

```text
No autonomous optimization.
Fix evaluation first.
```

## E1 — Directional

Useful for debugging but not promotion.

Examples:

- small local sample;
- noisy simulator;
- partial unit tests;
- approximate proxy metric.

Policy:

```text
Agents may explore.
Strategic promotion requires stronger evidence.
```

## E2 — Promotion-grade

Properties:

- reproducible;
- representative;
- baseline exists;
- regressions visible;
- seeds/config recorded;
- leakage understood;
- metric matches real objective closely enough.

Policy:

```text
Safe for Best-of-N and bounded autonomous optimization.
```

## E3 — Strongly automated

Properties:

- fast;
- deterministic or statistically robust;
- machine-readable;
- regression protection;
- rollback;
- artifact capture;
- promotion rule encoded.

Policy:

```text
Safe for broader parallelism and bounded autonomous loops.
```

---

# 19. PROMOTION RULE

Every project should define how an experimental change becomes the new baseline.

Example:

```text
Promote only if:
- primary metric improves by >= threshold;
- no critical regression occurs;
- result reproduces across required seeds/splits/opponents;
- runtime/cost remains within budget;
- evaluator protocol is unchanged unless evaluator change is the experiment;
- independent review finds no blocking flaw.
```

Customize the threshold and evidence burden.

Do not promote because the implementation "feels cleaner."

---

# 20. EXPERIMENT LEDGER

Do not use chat history as the project memory.

Recommended experiment record:

```yaml
id:
date:
git_commit:
strategic_question:
hypothesis:
mechanism:
change:
baseline:
evaluation_protocol:
seed_or_split:
result:
delta:
regressions:
runtime:
cost:
falsifier_result:
review_verdict:
interpretation:
knowledge_delta:
next_step:
```

For high-volume ML work, use MLflow or equivalent if justified.

For smaller competitions, markdown/JSON/CSV may be better.

Use the minimum effective system.

---

# 21. FAILURE-MODE DRIVEN DEBUGGING

When a system fails, do not immediately patch the symptom.

Use:

```text
symptom
-> 3-5 competing hypotheses
-> distinguishing evidence for each
-> cheapest discriminating test
-> result
-> narrow fix
```

Maintain `FAILURE_MODES.md` when recurring failures have recognizable signatures.

Example:

```markdown
## Symptom
Performance collapses only after environment reset.

### Plausible causes
- stale cached state
- initialization order
- seed handling
- hidden global mutable state

### Distinguishing evidence
...

### Known fix / diagnostic
...
```

This prevents broad "try things until it works" loops.

---

# 22. RESEARCH / STRATEGIST ROLE

A research/planning agent should not merely generate long research reports.

Its primary output is a **decision-quality belief update**.

Recommended loop:

```text
current bottleneck
-> inspect current project evidence
-> official docs / rules / primary sources
-> papers / repos
-> targeted discovery
-> compare mechanisms
-> identify gaps
-> propose discriminating experiments
-> Slice Contract
```

The Strategist should separate:

```text
CONFIRMED
INFERRED
SPECULATIVE
```

For external research, record:

- source;
- date/freshness where relevant;
- claim;
- confidence;
- how it changes the project decision;
- reproducible reference or artifact.

---

# 23. BUILDER ROLE

The Builder optimizes for:

```text
minimal intervention
+ correct implementation
+ objective evidence
+ reproducibility
```

The Builder should not redefine the strategy mid-slice without flagging it.

If implementation reveals the premise is wrong, return that discovery rather than forcing the original plan through.

---

# 24. TUTOR ROLE

A Tutor is optional.

Use one when the human repeatedly fails comprehension checkpoints around the same concept.

The Tutor should:

1. inspect real project code;
2. explain the concept in the context of the current architecture;
3. trace one actual example;
4. ask the human to predict a nearby case;
5. update `concepts/` only if the explanation is durably useful.

Avoid generic textbook lectures unless needed.

---

# 25. TOOL-SELECTION INTEGRATION

Use `AI_PROJECT_STACK_SELECTOR.md` as the detailed catalog and decision tree.

For every potential tool, record:

```text
Failure mode / bottleneck:
Required capability:
Lowest viable layer:
Chosen tool:
Project scope:
Expected benefit:
How benefit will be measured:
Why lighter option is insufficient:
Why overlapping tools are rejected:
```

Selection order:

```text
existing project capability
-> standard CLI
-> reusable Skill
-> narrow project-scoped MCP
-> dedicated orchestration
```

Do not install tools because they are popular.

Install them because a known project bottleneck justifies them.

---

# 26. METHODOLOGY SELECTION

Choose one dominant methodology per repository.

Typical defaults:

```text
Benchmark / algorithm / research competition
    -> Superpowers-style disciplined engineering

ML / data competition
    -> Superpowers-style loop + experiment/evaluation discipline

Agent / LLM / security competition
    -> Superpowers-style loop + traces + adversarial evaluator

Product hackathon
    -> gstack-style product/engineering/QA loop

Startup discovery / prototype
    -> gstack-style discovery -> wedge -> prototype -> user signal

Requirements-heavy long-lived product
    -> Spec Kit or BMAD only when justified

Large existing application feature
    -> Feature Dev or Superpowers depending task
```

Do not stack several overlapping methodologies.

---

# 27. PROJECT-MODE ADAPTATIONS

## 27.1 Benchmark / simulation / strategy competition

Examples:

- Pokémon TCG strategy/simulation;
- ARC-style challenges;
- search/optimization competitions.

Priorities:

```text
trustworthy evaluator
-> baseline
-> error taxonomy
-> hypothesis ledger
-> minimal strategic slices
-> independent review
-> held-out / robust comparison
-> bounded parallel search
```

Human Level 3-4 understanding should focus on:

- representation;
- decision logic;
- search;
- scoring/value functions;
- environment assumptions;
- evaluation methodology;
- failure modes.

Do not require deep understanding of commodity infrastructure.

---

## 27.2 ML / data competition

Priorities:

```text
metric reproduction
-> split strategy
-> leakage prevention
-> data validation
-> baseline
-> experiment lineage
-> model improvements
```

Every strategic model change should record:

- data version;
- features;
- split;
- seed;
- hyperparameters;
- runtime;
- out-of-fold / held-out result;
- leakage checks.

Human understanding should focus on:

- dataset;
- validation logic;
- error distribution;
- modeling assumptions;
- sources of generalization.

---

## 27.3 Agent / LLM competition

Priorities:

```text
traceability
-> task-level evaluator
-> prompt/model/tool versioning
-> failure taxonomy
-> adversarial cases
-> regression suite
```

Record:

- prompt/version;
- model/provider;
- tool calls;
- trajectory;
- environment state;
- success/failure;
- latency/cost;
- evaluator verdict.

Use Promptfoo / Phoenix / equivalent only when they materially simplify this.

---

## 27.4 Security / adversarial research

Maintain strict scope documentation.

Separate roles:

```text
attacker / proposer
defender / mitigator
evaluator
reviewer
```

Require:

- isolated execution;
- no production credentials;
- reproducible attack cases;
- untrusted-input handling;
- trace preservation;
- objective attack-success metric;
- scope compliance.

Human understanding should focus on:

- threat model;
- attack surface;
- trust boundaries;
- evaluator;
- mitigations;
- residual risk.

---

## 27.5 Product hackathon

The evaluator includes product reality.

Priorities:

```text
working end-to-end user journey
-> judging criteria
-> user value
-> reliability
-> demo quality
-> polish
```

Use browser/QA tooling when the judge/user will experience a UI.

Strategic hypotheses may be product hypotheses rather than algorithmic hypotheses.

Examples:

```text
"Users understand the value within 30 seconds."
"The demo's core agent loop is reliable enough for judging."
"This interaction creates a differentiated wow moment."
```

Evidence can include:

- user test;
- successful E2E runs;
- latency;
- failure rate;
- judge criteria coverage.

---

## 27.6 Startup discovery

Do not confuse research volume with market validation.

Loop:

```text
pain hypothesis
-> evidence
-> competitor map
-> technical feasibility
-> smallest wedge
-> prototype
-> user signal
-> belief update
```

Human Level 4 understanding should focus on:

- user pain;
- why now;
- wedge;
- alternatives;
- differentiation;
- willingness to adopt/pay;
- distribution;
- technical moat if relevant.

The "evaluator" becomes user evidence and product metrics.

---

## 27.7 Scientific / Bio / research project

Priorities:

```text
primary literature
-> data provenance
-> reproducibility
-> explicit hypothesis
-> controlled experiment
-> statistical validity
-> artifact lineage
```

Distinguish scientific evidence from model-generated interpretation.

Record citations, dataset versions, code versions, assumptions, and analysis outputs.

---

# 28. PARALLELISM POLICY

Parallelism is useful when tasks are independent and evaluation can arbitrate.

Good uses:

- independent research;
- independent root-cause investigations;
- alternative algorithms;
- Best-of-N implementations;
- benchmark optimization;
- adversarial review;
- separate test/evaluator construction.

Requirements:

- worktree/branch isolation;
- clear task boundary;
- independent evaluation;
- no shared mutable working tree;
- merge only after evidence review.

Do not parallelize several architecture-changing hypotheses into one branch.

---

# 29. BEST-OF-N POLICY

Use when:

- evaluation is cheap;
- candidates can be independent;
- outcome is objective;
- time pressure is high.

Example:

```text
Candidate A — conservative mechanism
Candidate B — alternate algorithm
Candidate C — simplified baseline
Candidate D — aggressive optimization

        ↓
same evaluator
        ↓
compare
        ↓
review winner
        ↓
promote only with evidence
```

Do not merge all candidates together automatically.

---

# 30. AUTONOMOUS ITERATION POLICY

Autonomous loops are earned, not assumed.

Require:

```text
Evaluator >= E2
clear objective
bounded iterations
bounded token/compute/cost budget
regression protection
inspectable changes
rollback
promotion rule
```

Good:

```text
Improve held-out score for at most 12 iterations while all critical regression tests pass.
```

Bad:

```text
Keep improving this until it is excellent.
```

STRATEGIC architecture changes should normally remain human-gated even when routine optimization becomes autonomous.

---

# 31. COGNITIVE WIP LIMIT

Track separately:

```text
Agent WIP
Human Strategic WIP
```

Suggested default:

```text
Primary cognitive projects: 1
Secondary strategic projects: <= 1
Concurrent strategic slices requiring human synthesis: <= 2
Routine autonomous agent tasks: as many as evaluator/review capacity supports
```

If unsynthesized strategic work exceeds the limit:

```text
STOP opening new strategic slices.
SYNTHESIZE first.
```

---

# 32. COMPREHENSION DEBT METRICS

Do not measure understanding by pages of documentation.

Track simple signals:

```text
unsynthesized strategic slices
unresolved HUMAN_MODEL gaps
cold-start explanation failures
unreproduced claimed improvements
open evaluator doubts
number of critical assumptions without tests
number of repeated agent mistakes not promoted into the harness
```

A repository is accumulating comprehension debt when these trend upward while code output accelerates.

---

# 33. COLD-START TEST

After approximately 3-5 strategic slices, after a major architecture change, or at the end of a focused workday, run a brief cold-start test.

Without reading generated summaries, the human should be able to answer:

1. What is the project optimizing?
2. What is the critical execution path?
3. What is the current baseline?
4. What is the current bottleneck?
5. What are the three most dangerous assumptions?
6. What did the last major experiment teach us?
7. What is the next experiment and why?

If the human cannot answer, target those gaps before adding another large strategic change.

Do not require perfect recall.

Require enough understanding to direct the project.

---

# 34. REPOSITORY RE-AUDIT TRIGGERS

Re-run the harness audit when:

- competition phase changes;
- evaluator becomes trustworthy;
- architecture changes materially;
- new model/data modality is introduced;
- project moves from exploration to exploitation;
- frontend/deployment becomes important;
- agent count becomes hard to manage;
- context/tool overhead becomes large;
- repeated failures expose a missing capability;
- startup prototype becomes a real product;
- strategic velocity repeatedly exceeds comprehension;
- a major Claude/Codex capability makes current tooling redundant.

---

# 35. BOOTSTRAP OUTPUT REQUIRED FROM CLAUDE / CODEX

Before substantial setup changes, produce:

```markdown
# Project Harness Audit

## Project
...

## Primary objective
...

## Cognitive priority
PRIMARY / SECONDARY / MAINTENANCE

## Classification
Primary mode:
Secondary modes:

## Current bottleneck
...

## Evaluator trust level
E0 / E1 / E2 / E3

Why:
...

## Current comprehension risks
- ...
- ...

## Critical path requiring human understanding
- ...
- ...

## Chosen methodology
...

## Chosen stack

| Capability | Tool/layer | Enable? | Scope | Failure mode removed | Success signal |
|---|---|---:|---|---|---|

## Explicitly rejected tools
- tool: reason
- tool: reason

## Work classification policy
ROUTINE:
IMPORTANT:
STRATEGIC:

## Evaluation loop
...

## Research loop
...

## Strategic slice loop
...

## Normal implementation loop
...

## Review policy
...

## Parallel-agent policy
...

## Human comprehension policy
...

## Durable knowledge plan
...

## Proposed repository changes
...

## Highest-value next three environment improvements
1.
2.
3.
```

Then implement only the obvious, non-destructive harness improvements.

Do not modify core strategy merely to demonstrate activity.

---

# 36. RECOMMENDED PROJECT FILES

Create only when useful.

## `docs/NORTH_STAR.md`

Keep concise:

```text
objective
official metric / product goal
deadline
constraints
current best result
current bottleneck
current hypothesis
next decisive experiment
```

## `docs/CURRENT_STATE.md`

Machine-maintainable current snapshot:

```text
baseline
active branch/commit
current experiment
known blockers
evaluation status
recent decisions
next queued work
```

## `docs/EVALUATION.md`

Document:

```text
primary metric
local evaluator
validation protocol
baseline
promotion rule
seeds/splits/opponents
leakage risks
runtime/cost constraints
leaderboard/external-eval policy
known evaluator weaknesses
```

## `docs/INVARIANTS.md`

Record facts that must not silently change.

Examples:

```text
units
coordinate frames
schema assumptions
data split rules
competition legality
security scope
API version
model/prompt version expectations
environment reset behavior
timing constraints
```

## `docs/FAILURE_MODES.md`

Symptom -> hypotheses -> distinguishing evidence -> known diagnostics.

## `docs/hypotheses/`

Maintain active strategic beliefs.

## `docs/experiments/`

One durable record per meaningful experiment.

## `docs/research/`

Persist only decision-relevant research.

## `docs/decisions/`

Use lightweight ADR-style records for durable architecture/strategy decisions.

---

# 37. CROSS-MODEL HANDOFF PROTOCOL

When using separate Strategist and Builder systems, do not hand off giant chat transcripts.

Use structured artifacts.

## Strategist -> Builder

Send:

```text
Slice Contract
+ relevant file paths
+ relevant research references
+ evaluator command
+ constraints
```

## Builder -> Strategist

Send:

```text
Evidence Packet
+ diff/commit
+ experiment record
+ unresolved uncertainty
```

## Builder -> Skeptic

Send:

```text
Slice Contract
+ diff
+ evaluator protocol
+ raw evidence
```

## Skeptic -> Strategist/Human

Send:

```text
falsification report
+ verdict
+ follow-up experiment if needed
```

This reduces narrative distortion between agents.

---

# 38. DEFAULT RESEARCH / BUILD / REVIEW ROLE MAPPING

The roles are conceptual, not tied to a vendor.

A practical mapping may be:

```text
Research / Strategist:
deep-research-capable reasoning model

Builder:
Claude Code / Codex coding agent

Skeptic:
fresh Claude / Codex / other strong model context

Tutor:
any strong model with repository access

Evaluator:
deterministic code whenever possible
```

Avoid making model brand the architecture.

The workflow should survive model changes.

---

# 39. HOW TO HANDLE AN EXTERNAL RESEARCH PLAN

When the user pastes a research model's plan into Claude/Codex:

Do **not** immediately implement it.

Run:

```text
1. Map recommendations to current repository state.
2. Verify assumptions against source code and evaluator.
3. Separate:
   - confirmed applicable;
   - plausible but unverified;
   - incompatible/stale.
4. Convert the highest-value recommendation into a Slice Contract.
5. Run the required human pre-check if STRATEGIC.
6. Implement minimally.
7. Return evidence.
```

External research is an input to the project belief system, not an instruction oracle.

---

# 40. HOW TO HANDLE AN IMPLEMENTATION REPORT

When the Strategist receives an implementation report:

Do not validate based only on prose.

Require access to or request:

```text
diff
commit
evaluation output
experiment record
logs/traces
configuration
```

Then ask:

```text
Did the evidence support the mechanism?
Could the result be explained another way?
Did the change alter the evaluator?
What new uncertainty exists?
What did we learn?
What is the highest-value next discriminating experiment?
```

Then produce the next Slice Contract.

---

# 41. DO NOT LET DOCUMENTATION BECOME ANOTHER BOTTLENECK

Durable docs should be:

- short;
- high-signal;
- current;
- linked to evidence;
- updated only when knowledge changes.

Avoid:

- giant generated architecture essays;
- duplicate docs;
- daily narrative logs;
- repeated summaries of unchanged facts;
- documentation that no agent or human actually reads.

Prefer diagrams, tables, invariants, experiment records, and decision logs.

---

# 42. STOP CONDITIONS

Agents should stop and surface a strategic decision when:

- evidence is ambiguous;
- evaluator is not trustworthy enough;
- a major assumption is unverified;
- several approaches have similar evidence;
- required scope expands materially;
- strategy change would invalidate prior comparisons;
- human critical-path understanding has fallen below the required level;
- security/safety scope becomes unclear;
- expected compute/cost exceeds budget;
- the next step depends on product/business judgment rather than implementation.

Stopping is not failure when the remaining problem is genuinely strategic.

---

# 43. SUCCESS CRITERIA FOR THE DEVELOPMENT HARNESS

The harness is working when:

1. the evaluator catches false improvements;
2. strategic experiments become smaller and more interpretable;
3. agents can operate with less repeated instruction;
4. the human can explain the critical path from memory;
5. repeated mistakes become tests/hooks/skills instead of repeated conversations;
6. research produces hypotheses rather than giant reports;
7. builder handoffs contain evidence rather than claims;
8. independent review catches real issues;
9. parallelism increases without merge/context chaos;
10. documentation becomes more useful as the codebase grows;
11. the human can increasingly identify what the AI may be overlooking;
12. validated progress per human hour increases.

---

# 44. FINAL RULES

When deciding whether to add a process step:

> **Does this measurably improve evaluation, understanding, reproducibility, or strategic decision quality?**

If no, remove it.

When deciding whether to add a tool:

> **What concrete failure mode does this remove, and how will we know it helped?**

If no concrete answer exists, do not add it.

When deciding whether the AI may proceed autonomously:

> **Can the environment objectively detect success, regression, and unsafe divergence?**

If no, keep the work bounded.

When deciding what the human must learn:

> **Will understanding this help direct strategy, detect failure, or generate a better next hypothesis?**

If no, delegate it.

The desired steady state is:

```text
research
-> hypothesis
-> prediction
-> minimal implementation
-> objective observation
-> falsification
-> knowledge update
-> human strategic understanding
-> better hypothesis
-> repeat
```

The AI should become faster over time.

The human should also become **more capable of directing it over time**, not less.

That is the operating system.
