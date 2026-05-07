# SkillMAS Core

## Paper

**SkillMAS: Skill Co-Evolution with LLM-based Multi-Agent System**

Large language model (LLM) agent systems are increasingly expected to improve after deployment, but existing work often decouples two adaptation targets: skill evolution and multi-agent system (MAS) restructuring. This separation can create organization bottlenecks, context pressure, and mis-specialization. We present **SkillMAS**, a non-parametric framework for adaptive specialization in multi-agent systems that couples skill evolution with MAS restructuring. SkillMAS uses Utility Learning to assign credit from verified execution traces, bounded skill evolution to refine reusable procedures without unfiltered library growth, and evidence-gated MAS restructuring when retained failures and Executor Utility indicate a structural mismatch. Across embodied manipulation, command-line execution, and retail workflows, SkillMAS is competitive under the reported harnesses while clarifying how post-deployment specialization is attributed, updated, and applied.

## Core Artifact

This repository contains the core non-parametric SkillMAS algorithm components:

- task-conditioned skill utility estimation
- task-conditioned agent utility estimation
- similarity-plus-utility skill selection
- trajectory-driven skill design
- post-hoc skill refinement and pruning
- markdown-based agent and skill specifications

This release intentionally excludes evaluation-specific integrations, prompt trees, and environment wrappers. It is organized as an algorithm artifact rather than a full end-to-end reproduction stack.

## Repository Scope

The `system/` directory contains the core TypeScript source modules:

- `src/spec/`: schemas for agent and skill specs
- `src/parser/`: markdown frontmatter parsing for agent cards
- `src/skill/`: utility learning, selection, skill design, and skill maintenance
- `src/tool/`: tool registry and execution interfaces
- `src/llm/`: chat and embedding client abstractions
- `src/messaging/`: in-process message bus and lightweight delegation primitive

## Explicit Omissions

This repository does **not** include:

- evaluation-specific integrations
- environment bridges
- prompt trees
- domain-specific routing templates
- evaluation-specific repair contracts

## Notes

Some modules still expose generic interfaces such as `taskType`, because SkillMAS is task-conditioned by design. However, the code in this repository is intended to be domain-agnostic and free of evaluation-specific routing logic.
