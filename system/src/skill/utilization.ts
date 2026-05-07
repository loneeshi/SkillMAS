import type { ToolCallRecord } from "../runtime/types.js"

export interface SkillCatalogEntry {
  description: string
  whenToUse: string
  tags: string[]
  allowedTools?: string[]
  toolHints?: string[]
  family?: string
  agent?: string
}

export interface SkillUsageResult {
  usedSkills: Record<string, string[]>
  candidateOnly: string[]
}

export interface SkillToolMappingConfig {
  skillToolMapping?: Record<string, string[]>
}

/**
 * Recover which agent actually called which tools from a possibly nested
 * delegation trace.
 */
export function buildAgentToolUsageIndex(
  rootAgentId: string,
  toolCalls: ToolCallRecord[],
): Map<string, Set<string>> {
  const toolsByAgent = new Map<string, Set<string>>()

  const ensure = (agentId: string): Set<string> => {
    const existing = toolsByAgent.get(agentId)
    if (existing) return existing
    const created = new Set<string>()
    toolsByAgent.set(agentId, created)
    return created
  }

  const walk = (agentId: string, calls: ToolCallRecord[]) => {
    const bucket = ensure(agentId)
    for (const tc of calls) {
      bucket.add(tc.tool)
      if (tc.subTrace) {
        walk(tc.subTrace.agentId, tc.subTrace.toolCalls)
      }
    }
  }

  walk(rootAgentId, toolCalls)
  return toolsByAgent
}

/**
 * Infer which tools provide causal evidence that a skill was operationally
 * relevant in the episode. Domain-level mappings take precedence; otherwise
 * we fall back to skill metadata and explicit tool mentions in the content.
 */
export function inferSkillToolHints(
  skillId: string,
  config: SkillToolMappingConfig,
  catalogEntry?: SkillCatalogEntry,
): Set<string> {
  const normalizedId = normalizeSkillId(skillId)
  const hints = new Set<string>()

  const mappedTools = config.skillToolMapping?.[normalizedId]
  if (mappedTools) {
    for (const toolName of mappedTools) hints.add(toolName)
    return hints
  }

  for (const toolName of catalogEntry?.allowedTools ?? []) hints.add(toolName)
  for (const toolName of catalogEntry?.toolHints ?? []) hints.add(toolName)

  const tags = new Set((catalogEntry?.tags ?? []).map((tag) => tag.toLowerCase()))
  const loweredId = normalizedId.toLowerCase()
  if (
    loweredId.includes("task_decomposition") ||
    tags.has("manager") ||
    tags.has("decomposition")
  ) {
    hints.add("delegate")
  }

  return hints
}

/**
 * Determine which skills were supported by execution evidence.
 *
 * Unlike the legacy "selected means used" heuristic, attribution is now
 * per-agent and trace-based: a selected skill only receives credit when the
 * same agent's observed tool trace overlaps with the skill's mapped or
 * inferred tool affordances.
 */
export function computeActualSkillsUsed(
  rootAgentId: string,
  injectedSkills: Record<string, string[]>,
  toolCalls: ToolCallRecord[],
  config: SkillToolMappingConfig,
  catalog: Map<string, SkillCatalogEntry>,
): SkillUsageResult {
  const toolsByAgent = buildAgentToolUsageIndex(rootAgentId, toolCalls)
  const usedSkills: Record<string, string[]> = {}
  const candidateOnly = new Set<string>()

  for (const [agentId, skillIds] of Object.entries(injectedSkills)) {
    const agentTools = toolsByAgent.get(agentId) ?? new Set<string>()
    const used: string[] = []

    for (const skillId of skillIds) {
      const normalizedId = normalizeSkillId(skillId)
      const hints = inferSkillToolHints(normalizedId, config, catalog.get(normalizedId) ?? catalog.get(skillId))
      const wasUsed = hints.size > 0 && [...hints].some((toolName) => agentTools.has(toolName))
      if (wasUsed) {
        used.push(skillId)
      } else {
        candidateOnly.add(skillId)
      }
    }

    if (used.length > 0) {
      usedSkills[agentId] = used
    }
  }

  return {
    usedSkills,
    candidateOnly: [...candidateOnly],
  }
}

export function extractToolHintsFromText(content: string): string[] {
  const hints = new Set<string>()
  const matcher = /\b(?:env\.)?[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+\b/g
  for (const match of content.matchAll(matcher)) {
    hints.add(match[0])
  }
  return [...hints]
}

function normalizeSkillId(skillId: string): string {
  return skillId.replace(/\\/g, "/")
}
