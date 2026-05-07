import { z } from "zod"

export const ToolPolicySchema = z.object({
  allow: z.array(z.string()).default([]),
  deny: z.array(z.string()).default([]),
})

export type ToolPolicy = z.infer<typeof ToolPolicySchema>

export const MemoryConfigSchema = z.object({
  mode: z.enum(["light", "full", "off"]).default("light"),
  store: z.enum(["md", "jsonl", "sqlite"]).default("md"),
  ttlHours: z.number().positive().optional(),
  capacity: z.number().int().positive().default(200),
})

export type MemoryConfig = z.infer<typeof MemoryConfigSchema>

export const AgentCardSchema = z.object({
  agentId: z.string().min(1),
  name: z.string().min(1),
  roleDescription: z.string().default(""),
  tools: z.array(z.string()).default([]),
  skillPool: z.array(z.string()).default([]),
})

export type AgentCard = z.infer<typeof AgentCardSchema>

export const AgentSpecSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  role: z.enum(["worker", "manager", "orchestrator"]),
  mode: z.enum(["all", "primary", "subagent"]).default("all"),
  description: z.string().optional(),
  role_description: z.string().optional(),
  tools: ToolPolicySchema.default({ allow: [], deny: [] }),
  memory: MemoryConfigSchema.default({ mode: "light", store: "md", capacity: 200 }),
  skills: z.array(z.string()).default([]),
  skill_pool: z.array(z.string()).optional(),
  metadata: z.record(z.string(), z.any()).default({}),
})

export type AgentSpec = z.infer<typeof AgentSpecSchema>
