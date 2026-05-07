# SkillMAS Core

## Paper

**SkillMAS: Skill Co-Evolution with LLM-based Multi-Agent System**

Large language model (LLM) agent systems are increasingly expected to improve after deployment, but existing work often decouples two adaptation targets: skill evolution and multi-agent system (MAS) restructuring. This separation can create organization bottlenecks, context pressure, and mis-specialization. We present **SkillMAS**, a non-parametric framework for adaptive specialization in multi-agent systems that couples skill evolution with MAS restructuring. SkillMAS uses Utility Learning to assign credit from verified execution traces, bounded skill evolution to refine reusable procedures without unfiltered library growth, and evidence-gated MAS restructuring when retained failures and Executor Utility indicate a structural mismatch. Across embodied manipulation, command-line execution, and retail workflows, SkillMAS is competitive under the reported harnesses while clarifying how post-deployment specialization is attributed, updated, and applied.

## Core Artifact

This repository contains the core non-parametric SkillMAS algorithm components:

- task-conditioned skill utility estimation
- task-conditioned agent utility estimation
- similarity-plus-utility skill selection
- retained-evidence construction
- trajectory-driven skill design
- post-hoc skill refinement and pruning
- validation-pool promotion for shadow skills
- evidence-gated MAS restructuring
- round-level orchestration for the SkillMAS adaptation loop
- markdown-based agent and skill specifications

This release intentionally excludes evaluation-specific integrations, prompt trees, and environment wrappers. It is organized as an algorithm artifact: users provide an `executeBatch` runtime that returns verified traces, and the core controller applies Utility Learning, retained-evidence construction, bounded skill evolution, validation-pool promotion, and evidence-gated MAS restructuring.

## Repository Scope

The `system/` directory contains the core TypeScript source modules:

- `src/spec/`: schemas for agent and skill specs
- `src/parser/`: markdown frontmatter parsing for agent cards
- `src/skill/`: utility learning, selection, skill design, and skill maintenance
- `src/round/`: Algorithm 1 core loop, retained evidence, policy index, validation pool, and MAS restructuring
- `src/tool/`: tool registry and execution interfaces
- `src/llm/`: chat and embedding client abstractions
- `src/messaging/`: in-process message bus and lightweight delegation primitive

## Algorithm Coverage

The `src/round/` modules correspond to the paper's round-level contract:

- `retention.ts`: constructs the retained evidence set from repeated failures, near misses, reusable successes, and retrieval/execution mismatches.
- `policy-index.ts`: builds the policy-card index from seed skills, validated skills, and optional expert cards.
- `validation-pool.ts`: keeps newly created or heavily revised skills in shadow status until verified use supports promotion.
- `restructuring.ts`: builds structural artifacts and applies one bounded executor edit: keep, add, merge/remove, or modify.
- `round-controller.ts`: runs the Algorithm 1 loop around an externally supplied batch executor.

## Explicit Omissions

This repository does **not** include:

- evaluation-specific integrations
- environment bridges
- prompt trees
- domain-specific routing templates
- evaluation-specific repair contracts

## Notes

Some modules still expose generic interfaces such as `taskType`, because SkillMAS is task-conditioned by design. However, the code in this repository is intended to be domain-agnostic and free of evaluation-specific routing logic.
