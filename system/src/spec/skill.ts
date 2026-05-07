/**
 * SkillSpec schemas for the core SkillMAS artifact.
 *
 * The schema keeps the algorithm-facing fields used by utility learning,
 * selection, and skill evolution while avoiding evaluation-specific contracts.
 */

import { z } from "zod"

export const SkillInvocationSchema = z.object({
  modelCanInvoke: z.boolean().default(true),
  userCanInvoke: z.boolean().default(true),
})

export type SkillInvocation = z.infer<typeof SkillInvocationSchema>

export const SkillArgumentSchema = z.object({
  name: z.string(),
  description: z.string(),
  required: z.boolean().default(false),
})

export type SkillArgument = z.infer<typeof SkillArgumentSchema>

export const SkillHookSchema = z.object({
  event: z.enum(["preInvoke", "postInvoke", "onError"]),
  command: z.string(),
})

export type SkillHook = z.infer<typeof SkillHookSchema>

export const SkillSeedRefSchema = z.object({
  skillId: z.string().min(1),
  family: z.string().optional(),
  reason: z.string().optional(),
})

export type SkillSeedRef = z.infer<typeof SkillSeedRefSchema>

export const SkillProvenanceSchema = z.object({
  strategy: z.enum(["seed", "distilled", "composed"]).default("seed"),
  sourceSeeds: z.array(z.string()).default([]),
  sourceLessons: z.array(z.string()).default([]),
  sourceRound: z.number().int().nonnegative().optional(),
  notes: z.array(z.string()).default([]),
})

export type SkillProvenance = z.infer<typeof SkillProvenanceSchema>

export const SkillDifficultySchema = z.object({
  effect: z.string().default(""),
  level: z.enum(["low", "medium", "high", "unknown"]).default("unknown"),
})

export type SkillDifficulty = z.infer<typeof SkillDifficultySchema>

export const RuntimeSkillPatchSchema = z.object({
  type: z.enum(["tool_call_repair"]),
  taskType: z.string().optional(),
  tool: z.string().min(1),
  match: z.record(z.string(), z.unknown()).default({}),
  repair: z.record(z.string(), z.unknown()).default({}),
  source: z.string().optional(),
  evidence: z.string().optional(),
})

export type RuntimeSkillPatch = z.infer<typeof RuntimeSkillPatchSchema>

export const RuntimeSkillContractSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("tool_contract"),
    taskType: z.string().optional(),
    tool: z.string().min(1),
    requiredArgs: z.record(z.string(), z.unknown()).default({}),
    source: z.string().optional(),
    evidence: z.string().optional(),
  }),
  z.object({
    type: z.literal("communication_contract"),
    taskType: z.string().optional(),
    requiredInfo: z.array(z.string().min(1)).default([]),
    source: z.string().optional(),
    evidence: z.string().optional(),
  }),
  z.object({
    type: z.literal("provenance_contract"),
    taskType: z.string().optional(),
    tools: z.array(z.string().min(1)).default([]),
    required: z.array(z.string().min(1)).default([]),
    source: z.string().optional(),
    evidence: z.string().optional(),
  }),
])

export type RuntimeSkillContract = z.infer<typeof RuntimeSkillContractSchema>

export const SkillSpecSchema = z.object({
  id: z.string().min(1),
  description: z.string().min(1),

  type: z.enum(["reference", "task", "workflow", "knowledge"]).default("reference"),

  /** Agentic skill tuple fields: k = <intent, method, difficulty, tool hint>. */
  intent: z.string().optional(),
  method: z.string().optional(),
  difficulty: SkillDifficultySchema.optional(),
  toolHints: z.array(z.string()).default([]),

  whenToUse: z.string().default(""),
  when_to_use: z.string().optional(),
  steps: z.array(z.string()).default([]),
  tags: z.array(z.string()).default([]),
  version: z.string().optional(),
  family: z.string().optional(),

  invocation: SkillInvocationSchema.default({ modelCanInvoke: true, userCanInvoke: true }),

  allowedTools: z.array(z.string()).optional(),

  arguments: z.array(SkillArgumentSchema).optional(),

  context: z.enum(["inline", "fork"]).default("inline"),
  agent: z.string().optional(),
  model: z.string().optional(),

  hooks: z.array(SkillHookSchema).optional(),

  generatedBy: z.enum(["manual", "evolution", "extension", "trajectory-learning"]).optional(),
  seed_refs: z.array(SkillSeedRefSchema).default([]),
  provenance: SkillProvenanceSchema.optional(),
  origin: z.enum(["manual", "skill-design", "skill-forge", "extension", "trajectory-learning", "failure-pattern"]).optional(),
  source_task_types: z.array(z.string()).default([]),
  not_to_do: z.array(z.string()).default([]),
  runtimePatches: z.array(RuntimeSkillPatchSchema).default([]),
  runtimeContracts: z.array(RuntimeSkillContractSchema).default([]),
  status: z.enum(["active", "shadow", "disabled"]).default("active"),
})

export type SkillSpec = z.infer<typeof SkillSpecSchema>

export type SkillSpecInput = z.input<typeof SkillSpecSchema>
