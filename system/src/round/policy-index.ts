import type { SkillManager, ResolvedSkill } from "../skill/manager.js"
import type { SkillSpec } from "../spec/skill.js"

export interface PolicyCard {
  id: string
  source: "seed" | "validated" | "expert"
  title: string
  content: string
  tags: string[]
  skillId?: string
}

export interface PolicyIndexOptions {
  expertCards?: PolicyCard[]
  maxCardChars?: number
}

export class PolicyIndex {
  private skillManager: SkillManager
  private expertCards: PolicyCard[]
  private maxCardChars: number

  constructor(skillManager: SkillManager, options?: PolicyIndexOptions) {
    this.skillManager = skillManager
    this.expertCards = options?.expertCards ?? []
    this.maxCardChars = options?.maxCardChars ?? 1600
  }

  async search(query: string, limit = 4): Promise<PolicyCard[]> {
    const cards = await this.cards()
    return cards
      .map((card) => ({ card, score: lexicalScore(query, searchableText(card)) }))
      .filter(({ score }) => score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map(({ card }) => card)
  }

  async cards(): Promise<PolicyCard[]> {
    const specs = await this.skillManager.list()
    const cards: PolicyCard[] = []

    for (const spec of specs) {
      if (spec.status === "disabled") continue
      const resolved = await this.skillManager.get(spec.id)
      if (!resolved) continue
      const source = classifySkill(spec)
      if (!source) continue
      cards.push(this.toCard(resolved, source))
    }

    return [...cards, ...this.expertCards]
  }

  private toCard(skill: ResolvedSkill, source: PolicyCard["source"]): PolicyCard {
    return {
      id: `${source}:${skill.spec.id}`,
      source,
      title: skill.spec.description,
      content: truncate(skill.content, this.maxCardChars),
      tags: skill.spec.tags,
      skillId: skill.spec.id,
    }
  }
}

function classifySkill(spec: SkillSpec): PolicyCard["source"] | null {
  if (!spec.generatedBy || spec.generatedBy === "manual") return "seed"
  if (spec.status === "active") return "validated"
  return null
}

function searchableText(card: PolicyCard): string {
  return [card.id, card.title, card.content, ...card.tags].join(" ")
}

function lexicalScore(query: string, candidate: string): number {
  const q = tokens(query)
  const c = tokens(candidate)
  if (q.size === 0 || c.size === 0) return 0

  let overlap = 0
  for (const token of q) {
    if (c.has(token)) overlap++
  }
  return overlap / Math.sqrt(q.size * c.size)
}

function tokens(value: string): Set<string> {
  return new Set(
    value
      .toLowerCase()
      .split(/[^a-z0-9_]+/)
      .filter((token) => token.length >= 2),
  )
}

function truncate(value: string, maxChars: number): string {
  if (value.length <= maxChars) return value
  return `${value.slice(0, maxChars - 3)}...`
}
