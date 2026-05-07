import matter from "gray-matter"
import { AgentCardSchema, AgentSpecSchema, type AgentCard, type AgentSpec } from "../spec/agent.js"

export interface ParsedAgent {
  spec: AgentSpec
  card: AgentCard
  prompt: string
  raw: string
  filePath?: string
}

function skillName(skillId: string): string {
  return skillId.split("/").pop() ?? skillId
}

export function parseAgentFile(raw: string, filePath?: string): ParsedAgent {
  const { data, content } = matter(raw)
  const parsed = AgentSpecSchema.parse(data)
  const prompt = content.trim()
  const spec: AgentSpec = {
    ...parsed,
    description: parsed.description ?? parsed.role_description,
    role_description: parsed.role_description ?? parsed.description,
    skills: (parsed.skill_pool ?? parsed.skills).map((skillId) => skillId.replace(/\\/g, "/")),
    skill_pool: (parsed.skill_pool ?? parsed.skills).map((skillId) => skillId.replace(/\\/g, "/")),
  }
  const card = AgentCardSchema.parse({
    agentId: spec.id,
    name: spec.name,
    roleDescription: spec.role_description ?? spec.description ?? "",
    tools: spec.tools.allow,
    skillPool: (spec.skill_pool ?? spec.skills).map(skillName),
  })
  return { spec, card, prompt, raw, filePath }
}

export function stringifyAgentFile(spec: AgentSpec, prompt: string): string {
  const { id, name, role, mode, description, role_description, tools, memory, skills, skill_pool, metadata } = spec

  const frontmatter: Record<string, unknown> = {
    id,
    name,
    role,
    mode,
  }

  if (description) frontmatter.description = description
  if (role_description && role_description !== description) frontmatter.role_description = role_description

  if (tools.allow.length > 0 || tools.deny.length > 0) {
    frontmatter.tools = tools
  }

  if (memory.mode !== "light" || memory.store !== "md" || memory.capacity !== 200 || memory.ttlHours !== undefined) {
    frontmatter.memory = memory
  }

  if (skills.length > 0) frontmatter.skills = skills
  if (skill_pool && skill_pool.length > 0 && JSON.stringify(skill_pool) !== JSON.stringify(skills)) {
    frontmatter.skill_pool = skill_pool
  }

  if (Object.keys(metadata).length > 0) frontmatter.metadata = metadata

  return matter.stringify(prompt, frontmatter)
}
