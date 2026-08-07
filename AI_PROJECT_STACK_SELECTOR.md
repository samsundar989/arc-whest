# AI Project Stack Selector & Development Harness
## Master prompt for Claude Code / Codex

Use this file as a **project-level bootstrap and periodic audit prompt**.

Its purpose is **not** to install every AI-development tool everywhere. Its purpose is to make the agent inspect the current project, classify the work, select the smallest high-leverage tool stack, establish a trustworthy evaluation loop, and customize how Claude/Codex should operate in that repository.

This file is intentionally opinionated:

> **Optimize the project environment before optimizing prompts.**
>
> Prefer deterministic feedback, executable tools, compact Skills, and project-scoped capabilities over giant prompts and always-on MCP collections.

---

# 0. YOUR ROLE

You are acting as the **AI Development Environment Architect** for this repository.

Your job is to:

1. Understand what this project is actually trying to optimize.
2. Inspect the existing repository, competition/hackathon/startup requirements, current tooling, tests, evaluators, docs, data pipeline, deployment environment, and agent configuration.
3. Classify the project.
4. Select the **minimum effective Claude/Codex tool stack**.
5. Establish or improve the project's **evaluation and experiment loop**.
6. Customize `CLAUDE.md`, `AGENTS.md`, Skills, hooks, MCPs, scripts, subagents, research workflows, and observability **only when they materially improve this project**.
7. Remove or disable redundant agent tooling that wastes context, conflicts with other workflows, or creates needless complexity.
8. Make the repository increasingly **agent-operable and human-comprehensible**.
9. Leave behind a concise explanation of how future Claude/Codex sessions should work in this repo.

Do **not** blindly install every tool listed below.

Do **not** assume that a tool that is useful in one repository belongs in another.

Do **not** optimize for number of agents, MCPs, Skills, plugins, or lines of configuration.

Optimize for:

- measured project progress;
- iteration speed;
- correctness;
- reproducibility;
- ability to detect regressions;
- ability to understand failures;
- useful parallelism;
- low context/tool overhead;
- human understanding of the system.

---

# 1. NON-NEGOTIABLE DEVELOPMENT PRINCIPLES

## 1.1 The evaluator is the judge

For benchmark, competition, ML, agent, algorithmic, simulation, and optimization projects:

> **The model does not decide whether an idea worked. The evaluator does.**

Before enabling aggressive autonomous iteration, establish a command or small set of commands that can answer things such as:

- Did the solution improve?
- Did correctness regress?
- What test cases became worse?
- What cases became better?
- How much runtime/memory/cost changed?
- Is the result statistically meaningful?
- Is the improvement on held-out data or only on the development set?

Prefer something like:

```bash
./scripts/test
./scripts/eval
./scripts/benchmark
```

with machine-readable output where practical.

If the project does not yet have a trustworthy evaluator, **building or repairing the evaluator is higher priority than adding more coding agents**.

---

## 1.2 Prefer environment engineering over prompt engineering

If agents repeatedly fail at the same task, first ask whether they are missing:

- a test;
- an evaluator;
- a script;
- a debugger;
- an LSP;
- a dataset inspection utility;
- a visualization;
- documentation;
- a reproduction command;
- a fixture;
- a trace;
- a browser;
- a profiler;
- a sandbox;
- an experiment ledger.

Fix the environment so future agents can solve the problem reliably.

Do not merely add paragraphs to `CLAUDE.md` telling the model to "be more careful."

---

## 1.3 Keep `CLAUDE.md` / `AGENTS.md` short

These files should be **maps, invariants, commands, and pointers**, not encyclopedias.

Put deeper information in durable repository docs such as:

```text
docs/
  architecture.md
  evaluation.md
  experiments.md
  competition.md
  research.md
  decisions/
```

The top-level agent instructions should tell the agent:

- what matters;
- what must never be violated;
- how to build;
- how to test;
- how to evaluate;
- where deeper documentation lives.

---

## 1.4 Skills > MCP when the capability is a workflow

Prefer a reusable **Skill** when the capability consists mainly of:

- instructions;
- repeatable procedures;
- scripts;
- checklists;
- templates;
- domain workflow knowledge.

Use an MCP when the agent genuinely needs structured access to an **external live system** or a capability that cannot be provided cleanly through an ordinary CLI/script.

---

## 1.5 CLI > MCP when a normal CLI already works well

Examples:

- `git` before a Git MCP for local work;
- `gh` before a GitHub MCP for many common GitHub actions;
- Playwright CLI + Skills before full Playwright MCP for normal coding-agent browser testing;
- project scripts before generic shell/database MCP abstractions.

Reasons:

- less tool-schema context;
- easier debugging;
- reproducibility outside the LLM;
- easier human usage;
- fewer permissions;
- simpler automation.

MCP is an escalation layer, not the default layer.

---

## 1.6 Project-scoped tools > global tool soup

External MCPs that expose:

- databases;
- cloud infrastructure;
- deployment systems;
- production services;
- GitHub write access;
- payments;
- credentials;
- private data

should generally be **project-scoped** and use the narrowest permissions possible.

Do not expose production write capabilities to every coding session.

---

## 1.7 One dominant workflow methodology per project

Do not simultaneously activate several overlapping engineering methodologies unless there is a specific reason.

Avoid stacking:

- gstack
- Superpowers
- Feature Dev
- Spec Kit
- BMAD
- large custom planning frameworks

all as competing global instructions.

Choose the methodology that fits the project.

---

## 1.8 Independent review beats repeated self-review

Where cost and time allow:

- use one agent/model to propose;
- another independent context/model to review;
- use tests/evaluators to arbitrate.

Do not rely on five rounds of "review your own answer again."

For cheap-to-evaluate problems, consider **Best-of-N** parallel candidates.

For expensive-to-evaluate problems, use targeted parallelism around uncertain hypotheses.

---

## 1.9 Parallelism requires isolation

When multiple agents modify the same repository:

- use git worktrees or equivalent isolated branches;
- define clear task boundaries;
- run evaluation independently;
- merge only after review/evaluation.

Do not let several autonomous agents race in the same working tree.

---

## 1.10 Autonomous loops require objective stopping rules

Do not enable Ralph-style repeated autonomous improvement merely because iteration is possible.

A loop requires:

- an objective metric;
- regression protection;
- a maximum iteration count or compute/token budget;
- a way to inspect what changed;
- a safe rollback path.

Good:

> Improve held-out score while all regression tests continue to pass, for at most N iterations.

Bad:

> Keep working until you think the project is excellent.

---

# 2. FIRST: AUDIT THE CURRENT PROJECT

Before recommending tools, inspect the repository and answer the following.

## 2.1 Project objective

Determine the actual primary objective.

Examples:

- maximize leaderboard score;
- solve benchmark tasks;
- minimize runtime;
- train the strongest model;
- build a working hackathon demo;
- validate a startup thesis;
- produce a research result;
- identify security vulnerabilities;
- create a polished user-facing product;
- automate a recurring workflow.

Write the objective in **one sentence**.

---

## 2.2 Project classification

Score the project from `0-3` on each dimension.

| Dimension | 0 | 3 |
|---|---|---|
| Benchmark/evaluator importance | subjective | everything is score-driven |
| ML/model experimentation | none | core work |
| Algorithm/search/simulation | none | core work |
| Browser/UI interaction | none | core product |
| External research dependency | little | constant |
| Large/unfamiliar codebase | tiny | very large |
| Production/deployment complexity | none | significant |
| Security sensitivity | normal | adversarial/security-critical |
| Need for multi-agent parallelism | little | many independent tasks |
| Iteration cost | seconds | hours/days |
| Time pressure | long horizon | hackathon/competition deadline |
| Product ambiguity | fixed spec | uncertain product/startup problem |

Use these scores to justify the stack.

---

## 2.3 Inspect what already exists

Before adding anything, inspect:

- README;
- competition/hackathon rules;
- baseline implementation;
- existing `CLAUDE.md`;
- existing `AGENTS.md`;
- `.mcp.json` / MCP settings;
- installed Skills/plugins;
- existing hooks;
- tests;
- eval scripts;
- benchmark scripts;
- experiment logs;
- notebooks;
- data pipelines;
- CI;
- Docker/devcontainer;
- environment setup;
- profiling tools;
- logging/tracing;
- deployment configuration;
- git branch/worktree conventions.

Prefer improving existing infrastructure over duplicating it.

---

# 3. TOOL-SELECTION HIERARCHY

For every capability, choose the lowest-overhead layer that fully solves the problem:

```text
1. Existing project script/tool
        ↓
2. Standard CLI
        ↓
3. Reusable Skill
        ↓
4. Narrow project-scoped MCP
        ↓
5. Dedicated orchestration framework
```

Escalate only when the previous layer is insufficient.

---

# 4. CORE TOOL CATALOG

Use this catalog to select project-specific tools.

---

## 4.1 gstack

### Best for

- startup prototypes;
- hackathons;
- product-heavy projects;
- ambiguous product requirements;
- feature planning;
- product/engineering/QA loops;
- projects where product judgment matters as much as code correctness.

### Use when

The project needs repeatable flows such as:

```text
problem framing
→ product review
→ engineering plan
→ implementation
→ code review
→ QA
→ ship
→ retrospective
```

### Avoid when

- the repo is almost entirely benchmark optimization;
- the only meaningful metric is an evaluator;
- the project is a small algorithmic experiment;
- another methodology already owns the workflow.

### Default

Prefer **gstack** for startup and product/hackathon repositories.

---

## 4.2 Superpowers

### Best for

- algorithmic competitions;
- benchmark projects;
- difficult debugging;
- unfamiliar engineering code;
- rigorous implementation;
- test-driven changes;
- research prototypes where correctness matters more than product process.

### Strengths

Use its style of:

- structured brainstorming;
- explicit plans;
- TDD;
- systematic debugging;
- implementation slices;
- verification;
- code review;
- subagent work.

### Avoid when

A lightweight project already has an extremely clear one-command evaluator and additional process creates more ceremony than value.

### Default

Prefer **Superpowers** for competition/research repositories unless a more specialized workflow clearly fits better.

---

## 4.3 Feature Dev

### Best for

- implementing a substantial feature inside an existing application;
- projects where you want separate explorer / architect / reviewer roles;
- codebases that do not need a full project-management methodology.

### Avoid when

gstack or Superpowers already supplies the dominant workflow.

---

## 4.4 Spec Kit

### Best for

- formal specifications;
- requirements-heavy systems;
- multi-component implementations;
- projects where ambiguity between requirements and implementation is expensive.

### Avoid when

- short hackathon;
- rapidly changing experiment;
- leaderboard optimization;
- tiny prototype.

---

## 4.5 BMAD

### Best for

- long-lived product development;
- many coordinated roles;
- heavier multi-agent software delivery;
- projects that benefit from explicit product/architecture/dev workflows.

### Avoid when

The project is a competition sprint or lightweight prototype.

---

# 5. DOCUMENTATION AND LIBRARY INTELLIGENCE

## 5.1 Context7

### Default recommendation

**Use in almost every active coding project that depends on external libraries/APIs.**

### Best for

- fast-moving SDKs;
- model APIs;
- frontend frameworks;
- AI libraries;
- unfamiliar dependencies;
- version-sensitive code.

### Purpose

Retrieve current library/API documentation instead of relying on model training memory.

### Do not use as

A replacement for reading project-local code or official competition rules.

---

# 6. CODE INTELLIGENCE

## 6.1 LSP

Enable the appropriate language server whenever possible.

Examples:

- Pyright / basedpyright for Python;
- TypeScript language server for TS/JS;
- clangd for C/C++;
- rust-analyzer for Rust.

### Use for

- symbol lookup;
- references;
- definitions;
- type errors;
- diagnostics;
- rename/refactor confidence.

### Principle

Do not ask an LLM to infer information a compiler/language server can answer deterministically.

---

## 6.2 Serena

### Use when

- the repository is medium/large;
- the agent repeatedly spends time searching for symbols;
- semantic navigation materially reduces context use;
- architecture is unfamiliar;
- precise symbol-level editing is valuable.

### Avoid when

- the repository is tiny;
- `rg` + LSP is already enough;
- setup cost exceeds likely benefit.

---

## 6.3 Repomix

### Use for

- independent cross-model reviews;
- handing an unfamiliar repository to another context/model;
- architecture analysis;
- comparing competitor/open-source repositories;
- producing self-contained code snapshots.

### Do not use as

The normal primary interface for an agent already running inside the repository.

---

# 7. EVALUATION STACK

## 7.1 Custom project evaluator

This is the default for competitions.

Create a stable interface such as:

```bash
./scripts/eval
```

Output at least:

```text
primary_metric
baseline_metric
delta
regression_count
new_success_count
runtime
seed/config
git_commit
```

Add task-specific details as needed.

---

## 7.2 Promptfoo

### Best for

- LLM applications;
- agent competitions;
- prompt experiments;
- tool-use evaluation;
- model/provider comparisons;
- security/red-team cases;
- regression suites.

### Use when

The system's behavior depends materially on prompts, models, agents, tool calls, or adversarial inputs.

### Avoid when

The project is purely numerical/algorithmic and a custom evaluator is simpler.

---

## 7.3 Best-of-N

Use parallel candidate generation when:

- evaluation is cheap;
- candidate implementations are meaningfully independent;
- time pressure is high;
- success is objective.

Example:

```text
Agent A: conservative approach
Agent B: different algorithm
Agent C: aggressive optimization
Agent D: baseline simplification
         ↓
same evaluator
         ↓
promote winner
```

Do not merge ideas merely because several agents like them.

---

# 8. BROWSER AND FRONTEND STACK

## 8.1 Playwright CLI + Skills

### Default browser choice for coding agents

Use when the project has:

- web UI;
- browser workflows;
- end-to-end behavior;
- authentication flows;
- visual demo requirements;
- hackathon judging through a web interface.

### Agent loop

```text
implement
→ start app
→ open browser
→ execute user journey
→ verify rendered state
→ inspect screenshots
→ fix
→ rerun
```

Prefer CLI + Skills over a permanently loaded full browser MCP unless richer persistent inspection is necessary.

---

## 8.2 Chrome DevTools MCP

### Add when

Playwright reveals a problem requiring deeper diagnosis:

- console errors;
- network failures;
- request timing;
- browser performance;
- memory/performance profiling;
- runtime DOM/browser inspection.

### Principle

Playwright answers:

> Does the product work?

Chrome DevTools answers:

> Why does it not work / why is it slow?

---

## 8.3 Frontend Design / design-oriented Skills

Use only when:

- UI polish affects judging;
- user-facing presentation is a major success factor;
- startup prototype quality matters.

Do not add to algorithmic competition repositories.

---

# 9. AI / ML STACK

## 9.1 Hugging Face Skills

### Use for

- dataset workflows;
- model loading;
- evaluation;
- fine-tuning;
- SFT;
- DPO;
- GRPO;
- reward modeling;
- model/job orchestration;
- Hugging Face Hub workflows.

### Best fit

ML competitions, model experiments, research prototypes.

### Avoid when

The project does not actually use models/datasets in the Hugging Face ecosystem.

---

## 9.2 MLflow

### Prefer when

- the project is ML/AI experiment heavy;
- metrics, parameters, artifacts, models, and runs must be compared;
- Claude/Codex agent traces are useful;
- experiment lineage matters.

---

## 9.3 Phoenix

### Prefer when

- the project is agent/RAG/tracing heavy;
- local/open-source observability is desirable;
- you need detailed traces, datasets, evaluations, or debugging.

---

## 9.4 Langfuse

### Prefer when

- building a production LLM product;
- tracing real application usage;
- prompt/version/latency/cost monitoring matters.

### Rule

Do **not** install MLflow + Phoenix + Langfuse by default.

Choose one based on the project.

---

# 10. RESEARCH STACK

Research tools have different jobs.

## 10.1 Official docs / primary sources first

For APIs, competition rules, libraries, benchmarks, research claims, and technical specifications:

> Prefer first-party documentation and original papers/repos.

---

## 10.2 Exa

### Best for

- semantic technical search;
- papers;
- obscure technical material;
- related projects;
- relevant repositories/articles.

### Use when

Normal keyword search is insufficient or discovery is research-heavy.

---

## 10.3 last30days

### Best for

- emerging tools;
- developer sentiment;
- new repos;
- current community discussions;
- Reddit / X / YouTube / Hacker News pulse;
- startup pain-point discovery;
- identifying what practitioners are experimenting with now.

### Treat as

Discovery and sentiment evidence.

### Do not treat as

Authoritative proof.

---

## 10.4 Firecrawl

### Best for

- extracting a whole important site;
- structured competitor research;
- documentation crawling;
- JS-heavy sites;
- turning sites into agent-friendly text/data.

### Use after discovery

Do not crawl the web indiscriminately.

Use search first, then deeply extract the few sources that matter.

---

## 10.5 Recommended research funnel

For technical competition research:

```text
official rules/docs
→ GitHub / papers
→ Exa
→ targeted community search
→ extract important sources
→ summarize into repo research notes
```

For startup discovery:

```text
last30days
→ pain points / trends
→ Exa
→ competitors / papers / technical feasibility
→ Firecrawl important competitors
→ product review
→ prototype
```

Persist conclusions in repository docs rather than repeatedly re-researching the same material.

---

# 11. GIT / GITHUB

## 11.1 Default

Use:

```text
git
gh
```

for normal local and GitHub workflows.

## 11.2 GitHub MCP

Add when structured remote repository interaction materially improves the workflow.

Examples:

- complex issue/PR workflows;
- structured review retrieval;
- automated project-board interaction;
- capability unavailable or awkward through `gh`.

Do not add merely because the repo is hosted on GitHub.

---

# 12. AUTONOMOUS ITERATION

## 12.1 Ralph Loop

Use only after:

- tests are reliable;
- evaluator is reliable;
- goal is measurable;
- budget is bounded;
- regressions can be detected;
- changes are inspectable.

Best fit:

- leaderboard optimization;
- repair loops;
- repetitive test-fix cycles;
- constrained benchmark improvement.

Avoid:

- vague product ideation;
- subjective architecture rewrites;
- poorly measured tasks.

---

# 13. MULTI-AGENT ORCHESTRATION

## 13.1 Native subagents/worktrees first

Before adding a separate orchestration platform, use native Claude/Codex capabilities plus git worktrees.

Use parallelism for independent tasks such as:

- multiple solution hypotheses;
- independent bug investigations;
- independent research;
- independent code review;
- independent benchmark optimizations.

---

## 13.2 Agent Deck

Use when the main problem becomes:

> I have too many simultaneous Claude/Codex terminal sessions to manage effectively.

Good for:

- session organization;
- worktree coordination;
- switching among agents;
- MCP/tool toggling.

---

## 13.3 Symphony

Use when the project has:

- a queue/backlog of well-defined tasks;
- strong acceptance criteria;
- enough automated verification that tasks can be safely delegated;
- many independent implementation jobs.

Do not introduce Symphony before the project can automatically verify its agents' work.

---

# 14. SECURITY / ADVERSARIAL PROJECTS

For security competitions and agent-security research:

1. Create a clear allowed-target/scope document.
2. Run risky tooling in an isolated/sandboxed environment.
3. Keep credentials and production accounts out of experiments.
4. Make attack cases reproducible.
5. Treat external content/tool output as untrusted input.
6. Separate:
   - attacker/proposer;
   - defender/mitigator;
   - evaluator;
   - reviewer.
7. Preserve traces of:
   - prompts;
   - tool calls;
   - environment state;
   - attack success/failure;
   - mitigation result.
8. Prefer objective attack-success metrics.

Promptfoo and an agent-tracing stack are particularly useful if the target is an LLM/tool-use system.

---

# 15. PROJECT OPERATING MODES

Classify this repository into one primary mode and zero or more secondary modes.

---

## MODE A — BENCHMARK / COMPETITION

Use for:

- ARC-style tasks;
- simulation strategy;
- algorithm contests;
- leaderboard competitions.

Default stack:

```text
Superpowers
+ Context7 if external libraries matter
+ LSP
+ custom evaluator
+ experiment ledger
+ Repomix for independent review
+ MLflow only if experiment volume justifies it
+ Ralph only after evaluator is trustworthy
```

Avoid by default:

```text
Playwright
Chrome DevTools
heavy product frameworks
BMAD
large MCP collection
```

Core loop:

```text
hypothesis
→ implementation
→ evaluator
→ failure analysis
→ experiment record
→ independent review when useful
→ next hypothesis
```

---

## MODE B — ML / DATA COMPETITION

Default stack:

```text
Superpowers
+ LSP
+ Context7
+ custom held-out evaluator
+ Hugging Face Skills when relevant
+ MLflow
+ data inspection/visualization scripts
+ Exa / papers / GitHub research
```

Core requirement:

**Prevent leakage and invalid validation before optimizing model quality.**

---

## MODE C — AGENT / LLM COMPETITION

Default stack:

```text
Superpowers
+ Promptfoo
+ project-specific evaluator
+ trace/observability layer
+ Context7
+ LSP
+ Exa/GitHub research
+ sandbox when adversarial
```

Potential additions:

```text
Playwright if the agent controls a browser
Chrome DevTools if browser debugging is needed
Ralph if evaluation is objective
```

---

## MODE D — PRODUCT HACKATHON

Default stack:

```text
gstack
+ Context7
+ LSP
+ Playwright CLI + Skills
+ Chrome DevTools when debugging
+ Frontend Design if UI matters
+ gh/git
+ lightweight product QA
```

Optional:

```text
Feature Dev
```

Do not overbuild infrastructure that will not improve the demo.

Priority:

```text
working end-to-end demo
> user value
> reliability
> visual polish
> architectural purity
```

unless judging criteria say otherwise.

---

## MODE E — STARTUP DISCOVERY / PROTOTYPE

Default stack:

```text
gstack
+ last30days
+ Exa
+ Firecrawl selectively
+ Context7
+ Playwright
+ LSP
```

Research loop:

```text
identify pain
→ find evidence
→ study competitors
→ test technical feasibility
→ define smallest wedge
→ build prototype
→ get user signal
```

Do not let "research" become a substitute for testing the idea with users.

Move to Spec Kit/BMAD only when requirements and team complexity justify the ceremony.

---

## MODE F — LARGE EXISTING CODEBASE

Default stack:

```text
Superpowers or Feature Dev
+ LSP
+ Serena
+ Context7
+ repo tests
+ architecture docs
+ Repomix for external review
```

Focus on:

- locating invariants;
- understanding call paths;
- preserving behavior;
- adding missing tests before large refactors.

---

# 16. INITIAL PROFILES FOR MY CURRENT PROJECTS

These are **starting hypotheses, not hard-coded answers**.

Inspect each project's current repository, rules, scoring system, deadline, and architecture before finalizing the stack.

---

## 16.1 Pokémon TCG AI simulation + strategy competition

Likely primary mode:

```text
BENCHMARK / COMPETITION
+ SIMULATION / SEARCH
+ possibly AGENT/ML
```

Initial preference:

```text
Superpowers
LSP
custom deterministic simulator/evaluator
experiment ledger
Context7 for libraries
MLflow only if experiment count becomes large
Repomix for independent architecture/strategy review
Exa/GitHub research for related game-AI/search approaches
Ralph only when repeated evaluation is trustworthy
```

Critical priority:

> Build an extremely trustworthy simulation/evaluation harness and error analysis workflow before scaling autonomous strategy iteration.

Browser/product tooling should remain off unless the competition interface itself requires it.

---

## 16.2 ARC White Box / estimation challenge

First verify the exact current challenge/rules.

Likely primary mode:

```text
BENCHMARK / COMPETITION
+ ALGORITHM / REASONING
+ EXPERIMENTATION
```

Initial preference:

```text
Superpowers
custom evaluator
LSP
experiment ledger
Repomix / independent model review
MLflow only if useful
minimal MCP footprint
```

Prioritize:

- hypothesis tracking;
- held-out evaluation;
- visual/debug utilities if task structure benefits from them;
- fast comparison against baseline;
- avoiding overfitting to known tasks.

---

## 16.3 ARC-AGI-2

Likely primary mode:

```text
BENCHMARK / REASONING
```

Initial preference:

```text
Superpowers
custom evaluation harness
visual task inspection utilities
experiment ledger
parallel independent approaches
cross-model review
minimal external tooling
```

The evaluator and error taxonomy matter more than product tooling.

---

## 16.4 ARC-AGI-3

First inspect the current ARC-AGI-3 task format and official tooling rather than assuming it behaves like ARC-AGI-2.

Likely primary mode:

```text
BENCHMARK / AGENT / REASONING
```

Select browser/interactive tooling only if the actual challenge environment requires it.

Prioritize:

```text
environment interface
reproducible trajectories
evaluation
tracing
failure classification
strategy experiments
```

---

## 16.5 Trace the Ace

Do **not** infer the stack from the name.

First read:

- official rules;
- scoring;
- starter repo;
- runtime environment;
- allowed external services;
- data;
- submission format.

Then classify using the mode system in this file.

Competition default if nothing else applies:

```text
Superpowers
+ LSP
+ custom evaluator
+ experiment ledger
+ minimal project-specific research stack
```

---

## 16.6 AI Agent Security

Likely primary mode:

```text
AGENT / LLM COMPETITION
+ SECURITY / ADVERSARIAL
```

Initial preference:

```text
Superpowers
Promptfoo
project-specific attack/evaluation harness
Phoenix or MLflow tracing
Context7
LSP
Exa/GitHub/paper research
sandboxed execution
strict target/scope documentation
```

Architecture should deliberately separate:

```text
attack generation
defense/mitigation
evaluation
trace analysis
independent review
```

Enable browser tooling only if the agent environment uses browsers.

Autonomous attack iteration requires strict scope and objective scoring.

---

## 16.7 Agentic Cinema

First verify the competition's judging criteria and expected artifact.

If it is primarily a product/demo hackathon:

```text
PRODUCT HACKATHON
+ AGENT/LLM
```

Initial preference:

```text
gstack
Context7
LSP
Playwright CLI + Skills
Chrome DevTools when needed
Frontend Design if presentation affects judging
Promptfoo/custom eval for agent behavior
Exa/last30days for rapid product/research discovery
```

Optimize first for a convincing end-to-end experience.

---

## 16.8 Kaggriculture

First inspect dataset, metric, competition rules, starter notebooks, and allowed external data.

If this is primarily an ML/data competition:

```text
ML / DATA COMPETITION
```

Initial preference:

```text
Superpowers
LSP
custom local cross-validation/evaluator
MLflow
Hugging Face Skills if relevant
Context7
research through papers/GitHub/Exa
data validation scripts
```

Before model optimization, verify:

- leakage;
- split strategy;
- metric reproduction;
- preprocessing reproducibility;
- train/test distribution differences.

---

## 16.9 Biohub

First determine whether the project is primarily:

- biological research;
- ML/data competition;
- scientific software;
- agentic research;
- product/hackathon.

Potential research/ML default:

```text
Superpowers
Context7
LSP
Exa + primary papers
Hugging Face Skills if model-centric
MLflow if experiment-centric
strong reproducibility scripts
```

Scientific claims should be tied to primary sources and experiment artifacts.

---

## 16.10 Startup ideas

Primary mode:

```text
STARTUP DISCOVERY / PROTOTYPE
```

Initial preference:

```text
gstack
last30days
Exa
Firecrawl selectively
Context7
LSP
Playwright CLI + Skills
Chrome DevTools when needed
Frontend Design when UI matters
```

Flow:

```text
pain-point discovery
→ evidence
→ competitor map
→ technical feasibility
→ differentiated wedge
→ smallest usable prototype
→ user feedback
→ only then deeper architecture
```

Do not start with BMAD/heavy orchestration unless the startup has already moved beyond exploratory prototyping.

---

# 17. REQUIRED REPOSITORY OUTPUTS

After auditing this project, create or update only the files that materially help.

Recommended structure:

```text
AGENTS.md or CLAUDE.md
docs/
  architecture.md
  evaluation.md
  experiments/
  research/
  decisions/
scripts/
  setup
  test
  eval
  benchmark
skills/
```

Do not create empty documentation for appearance.

---

## 17.1 Top-level agent file

Keep it concise.

It should contain:

1. Project objective.
2. Critical invariants.
3. Environment/bootstrap command.
4. Test command.
5. Evaluation command.
6. Important directories.
7. Where architecture docs live.
8. Where experiment history lives.
9. Project-specific agent workflow.
10. Tools that are intentionally enabled.
11. Tools that are intentionally disabled.

---

## 17.2 `docs/evaluation.md`

For competition/research repos, document:

- primary metric;
- local evaluator;
- validation split;
- baseline;
- reproducibility rules;
- seeds;
- known sources of leakage;
- runtime/cost constraints;
- promotion criteria for a new approach.

---

## 17.3 Experiment ledger

Each meaningful experiment should record:

```yaml
date:
git_commit:
hypothesis:
change:
baseline:
result:
delta:
runtime:
seed:
interpretation:
next_step:
```

Do not rely on chat history as the experiment database.

---

# 18. PROJECT-SPECIFIC WORKFLOW GENERATION

After selecting the stack, generate a workflow for this project.

The workflow must explicitly define:

## Planning

When is research required?

When is a written plan required?

When is a tiny change small enough to implement immediately?

## Implementation

What should a normal implementation slice look like?

## Verification

What tests run after every change?

What heavier evaluations run before promotion?

## Review

When should another Claude/Codex context review the change?

## Parallelism

Which tasks are safe for worktrees/subagents?

## Research

Where should findings be persisted?

## Experiments

How are hypotheses and results recorded?

## Stop conditions

When should the agent stop optimizing and ask for a strategic decision?

---

# 19. HUMAN-COMPREHENSION REQUIREMENT

The workflow must not optimize solely for AI velocity.

For substantial changes, leave enough durable explanation that a human can answer:

- What changed?
- Why was it changed?
- What assumption does it rely on?
- How do we know it improved the project?
- What failed before this?
- What are the remaining uncertainties?
- Where would I look to modify this system myself?

Prefer concise architecture notes, experiment records, and decision records over giant generated explanations.

The repository should become easier to understand as AI makes more changes, not harder.

---

# 20. TOOL CONFLICT RULES

When two tools overlap, choose one.

Examples:

```text
gstack OR Superpowers as dominant methodology
MLflow OR Phoenix OR Langfuse as primary observability layer
Playwright CLI/Skills before full Playwright MCP
git/gh before GitHub MCP for ordinary workflows
LSP + rg before Serena on small repos
native subagents/worktrees before Symphony
```

If you choose the heavier option, explain why the lighter option is insufficient.

---

# 21. TOOL REMOVAL IS ALLOWED

If the repository has accumulated:

- unused MCPs;
- overlapping Skills;
- stale hooks;
- contradictory instructions;
- giant agent config;
- unused orchestration;
- redundant docs;

recommend removing or disabling them.

A smaller coherent environment is usually better than an impressive-looking agent stack.

---

# 22. PERIODIC RE-AUDIT TRIGGERS

Re-run this stack-selection process when any of these occur:

- project moves from exploration to implementation;
- evaluator becomes reliable;
- repository grows significantly;
- competition phase changes;
- model/data pipeline changes;
- frontend is introduced;
- deployment becomes real;
- agent count becomes hard to manage;
- tool context becomes bloated;
- repeated failures indicate missing environment capabilities;
- startup prototype becomes a real product;
- a major new Claude/Codex capability makes an existing tool redundant.

---

# 23. MERGING WITH OTHER ENGINEERING-GUIDANCE DOCUMENTS

This file may be combined with another engineering-process recommendation document.

When another guidance file is present:

1. Read both.
2. Deduplicate overlapping advice.
3. Preserve project-specific instructions over generic ones.
4. Prefer measurable/evaluable workflows over stylistic process rules.
5. Prefer lower tool/context overhead when two approaches provide equivalent capability.
6. Preserve explicit human-comprehension requirements.
7. Resolve contradictory methodology recommendations by selecting one dominant project workflow.
8. Do not simply concatenate every instruction into `CLAUDE.md`.
9. Put deeper process documentation in `docs/` or Skills.
10. Summarize the final merged operating model in the top-level agent file.

---

# 24. REQUIRED AUDIT OUTPUT

Before making substantial setup changes, produce this concise report:

## Project

`<name>`

## Primary objective

`<one sentence>`

## Classification

`<primary mode + secondary modes>`

## Current bottleneck

`<what currently most limits progress>`

## Chosen methodology

`<gstack / Superpowers / Feature Dev / Spec Kit / BMAD / none>`

Why:

`<short rationale>`

## Core stack

| Tool | Enable? | Scope | Reason |
|---|---|---|---|
| Context7 | | | |
| LSP | | | |
| Serena | | | |
| Promptfoo | | | |
| Custom evaluator | | | |
| Playwright | | | |
| Chrome DevTools | | | |
| Hugging Face Skills | | | |
| MLflow | | | |
| Phoenix | | | |
| Langfuse | | | |
| Exa | | | |
| last30days | | | |
| Firecrawl | | | |
| Repomix | | | |
| Ralph Loop | | | |
| Agent Deck | | | |
| Symphony | | | |

## Explicitly rejected tools

List tools considered but not enabled, with one-line reasons.

## Evaluation loop

`<commands and promotion rule>`

## Research loop

`<sources/tools and persistence location>`

## Normal implementation loop

`<project-specific flow>`

## Parallel-agent policy

`<when to use / when not to use>`

## Repository changes

`<files/config/scripts to create or edit>`

## Highest-value next three environment improvements

1.
2.
3.

Then implement the approved/obvious non-destructive project-environment changes.

Do not modify core solution logic merely to demonstrate activity.

---

# 25. FINAL OPERATING RULE

Whenever deciding whether to add a new tool, plugin, MCP, Skill, agent, framework, or orchestration layer, ask:

> **What specific failure mode or bottleneck does this remove, and how will we know it helped?**

If there is no concrete answer, do not add it.

The goal is not to create the most sophisticated Claude/Codex setup.

The goal is to create the smallest project-specific system in which AI agents can:

```text
understand
→ hypothesize
→ implement
→ observe
→ evaluate
→ debug
→ learn
→ document
→ repeat
```

faster than before while preserving correctness and human understanding.

---

# QUICK INVOCATION

Use this at the beginning of a repository audit:

> Read `AI_PROJECT_STACK_SELECTOR.md`, then inspect this repository and its current task/competition/product requirements. Treat the file as a decision framework rather than an install list. Classify the project, identify the current bottleneck, choose the minimum effective Claude/Codex stack, define the evaluation/research/implementation loops, explain which tools you are intentionally rejecting, and then make the non-destructive repository/configuration changes needed to establish the project-specific development harness. Keep the top-level agent instructions short and move deeper knowledge into durable docs or Skills. Do not enable autonomous iteration until evaluation is trustworthy.

