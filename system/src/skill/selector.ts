/**
 * SkillSelector — replaces static tag-matching with Q-value-weighted selection.
 *
 * Two-phase selection (MemRL-inspired):
 *   1. Semantic filter: compute similarity between skill descriptions and task
 *   2. Q-weighted ranking: combine similarity + Q-value + UCB exploration
 *
 * Text similarity defaults to Azure/OpenAI-compatible embeddings
 * (`text-embedding-3-large`) with lexical fallback when credentials are absent.
 *
 * Selection is deterministic (greedy top-K by composite score).
 */

import type { SkillQTable } from "./q-table"
import {
  buildSkillSemanticText,
  getDefaultSimilarityScorer,
  type TextSimilarityScorer,
} from "../llm/embedding"

/** Input context for skill selection. */
export interface SelectionContext {
  agentId: string
  taskType: string
  taskDescription: string
  availableSkills: string[]
  maxSkills: number
  mandatorySkills?: string[]
}

/** A skill selected with its score and selection reason. */
export interface SelectedSkill {
  skillId: string
  score: number
  reason: "q_value" | "similarity" | "exploration" | "mandatory"
}

export class SkillSelector {
  private qTable: SkillQTable
  private similarityScorer: TextSimilarityScorer

  constructor(qTable: SkillQTable, similarityScorer: TextSimilarityScorer = getDefaultSimilarityScorer()) {
    this.qTable = qTable
    this.similarityScorer = similarityScorer
  }

  /**
   * Select skills for an agent given a task.
   *
   * Flow:
   *   1. Compute similarity between each skill's description/whenToUse and the task
   *   2. Build composite scores via QTable.selectTopK
   *   3. Return greedy top-K skills by score
   */
  async selectSkills(
    context: SelectionContext,
    allSkills: Map<string, { description: string; whenToUse: string; tags: string[] }>,
  ): Promise<SelectedSkill[]> {
    const { availableSkills, taskDescription, taskType, agentId, maxSkills, mandatorySkills } = context

    if (availableSkills.length === 0) return []

    const k = Math.min(maxSkills, availableSkills.length)
    const taskText = `${taskType} ${taskDescription}`.toLowerCase()

    const similarityInputs: Record<string, string> = {}
    for (const id of availableSkills) {
      const skill = allSkills.get(id)
      if (!skill) continue
      similarityInputs[id] = buildSkillSemanticText({
        description: skill.description,
        whenToUse: skill.whenToUse,
        tags: skill.tags,
      })
    }
    const similarities = await this.similarityScorer.scoreMany(taskText, similarityInputs)

    for (const id of availableSkills) {
      this.qTable.initialize(id)
    }

    const ranked = this.qTable.selectTopK(
      availableSkills,
      { taskType, agentId, similarities },
      availableSkills.length,
    )

    const rankedById = new Map(ranked.map((r) => [r.skillId, r]))
    const mandatory = (mandatorySkills ?? []).filter((id) => availableSkills.includes(id))
    if (mandatory.length >= k) {
      return mandatory.slice(0, k).map((id) => ({
        skillId: id,
        score: rankedById.get(id)?.score ?? 1,
        reason: "mandatory",
      }))
    }

    const remaining = ranked.filter((r) => !mandatory.includes(r.skillId))
    const remainingSlots = k - mandatory.length

    const baseSelected: SelectedSkill[] = mandatory.map((id) => ({
      skillId: id,
      score: rankedById.get(id)?.score ?? 1,
      reason: "mandatory",
    }))

    return baseSelected.concat(
      remaining.slice(0, remainingSlots).map((r) => ({
        skillId: r.skillId,
        score: r.score,
        reason: r.reason as SelectedSkill["reason"],
      })),
    )
  }
}
