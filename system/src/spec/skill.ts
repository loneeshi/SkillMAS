/**
 * SkillSpec v2 — aligned with Claude Code's skill framework + A2S research extensions.
 *
 * Claude Code concepts mapped:
 *   name → id (path-based: "alfworld/task_decomposition")
 *   description → description
 *   disable-model-invocation + user-invocable → invocation
 *   allowed-tools → allowedTools
 *   context → context (inline | fork)
 *   agent → agent
 *   model → model
 *   hooks → hooks
 *
 * A2S-specific extensions:
 *   whenToUse, steps, tags — original fields
 *   type — content classification (reference | task | workflow)
 *   arguments — structured argument schema
 *   generatedBy — auto-generation tracking
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
  type: z.enum(["taubench_action_repair"]),
  taskType: z.string().optional(),
  tool: z.string().min(1),
  match: z.record(z.string(), z.unknown()).default({}),
  repair: z.record(z.string(), z.unknown()).default({}),
  source: z.enum(["action_mismatch"]).optional(),
  evidence: z.string().optional(),
})

export type RuntimeSkillPatch = z.infer<typeof RuntimeSkillPatchSchema>

export const RuntimeSkillContractSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("taubench_action_contract"),
    taskType: z.string().optional(),
    tool: z.string().min(1),
    requiredArgs: z.record(z.string(), z.unknown()).default({}),
    source: z.enum(["action_mismatch"]).optional(),
    evidence: z.string().optional(),
  }),
  z.object({
    type: z.literal("taubench_communication_contract"),
    taskType: z.string().optional(),
    requiredInfo: z.array(z.string().min(1)).default([]),
    source: z.enum(["communicate_check"]).optional(),
    evidence: z.string().optional(),
  }),
  z.object({
    type: z.literal("taubench_payment_delta_communication_contract"),
    taskType: z.string().optional(),
    requiredInfo: z.array(z.string().min(1)).default([]),
    requireOrderIds: z.boolean().default(true),
    source: z.enum(["communicate_check", "payment_delta_failure"]).optional(),
    evidence: z.string().optional(),
  }),
  z.object({
    type: z.literal("taubench_item_order_provenance_contract"),
    taskType: z.string().optional(),
    tools: z.array(z.string().min(1)).default([]),
    required: z.array(z.enum([
      "candidate_orders_inspected",
      "current_item_in_order",
      "replacement_item_from_catalog",
      "payment_method_grounded",
    ])).default(["candidate_orders_inspected", "current_item_in_order", "replacement_item_from_catalog", "payment_method_grounded"]),
    source: z.enum(["provenance_failure"]).optional(),
    evidence: z.string().optional(),
  }),
  z.object({
    type: z.literal("taubench_partial_cancel_contract"),
    taskType: z.string().optional(),
    source: z.enum(["partial_cancel_failure"]).optional(),
    evidence: z.string().optional(),
  }),
  z.object({
    type: z.literal("taubench_multi_order_subgoal_contract"),
    taskType: z.string().optional(),
    minSuccessfulWrites: z.number().int().positive().default(2),
    source: z.enum(["subgoal_failure"]).optional(),
    evidence: z.string().optional(),
  }),
  z.object({
    type: z.literal("taubench_multi_order_binding_contract"),
    taskType: z.string().optional(),
    tools: z.array(z.string().min(1)).default([]),
    minDistinctOrders: z.number().int().positive().default(2),
    requireOrderIdsInClosure: z.boolean().default(true),
    source: z.enum(["multi_order_binding_failure"]).optional(),
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
