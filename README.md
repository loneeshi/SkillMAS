# SkillMAS Core

This repository contains the core non-parametric SkillMAS algorithm components:

- task-conditioned skill utility estimation
- task-conditioned agent utility estimation
- similarity-plus-utility skill selection
- trajectory-driven skill design
- post-hoc skill refinement and pruning
- markdown-based agent and skill specifications

This release intentionally excludes benchmark adapters, benchmark-specific trees, and environment wrappers. It is organized as an algorithm artifact rather than a full end-to-end benchmark reproduction stack.

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

- benchmark adapters
- benchmark-specific environment bridges
- benchmark-specific prompt trees
- hard-coded domain routing templates
- benchmark-specific repair contracts

## Notes

Some modules still expose generic interfaces such as `taskType`, because SkillMAS is task-conditioned by design. However, the code in this repository is intended to be domain-agnostic and free of benchmark-specific hard rules.

