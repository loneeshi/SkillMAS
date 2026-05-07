import type { SkillManager } from "./manager.js"
import { isAutoSkill, isSeedSkill } from "./seed-skill.js"

export interface ProbationEpisodeResult {
  episode: number | string
  taskType?: string
  success: boolean
}

export interface SkillPromotionInput {
  skillId: string
  baselineEpisodes: ProbationEpisodeResult[]
  candidateEpisodes: ProbationEpisodeResult[]
  seedSkillsModified: number
  candidateUsedSkill?: boolean
}

export interface SkillPromotionDecision {
  skillId: string
  promoted: boolean
  baselineSuccessRate: number
  candidateSuccessRate: number
  regressions: Array<number | string>
  fixes: Array<number | string>
  reason: string
}

export function evaluateSkillPromotion(input: SkillPromotionInput): SkillPromotionDecision {
  const baseline = new Map(input.baselineEpisodes.map((episode) => [String(episode.episode), episode]))
  const candidate = new Map(input.candidateEpisodes.map((episode) => [String(episode.episode), episode]))
  const commonIds = [...baseline.keys()].filter((episodeId) => candidate.has(episodeId))
  const regressions: Array<number | string> = []
  const fixes: Array<number | string> = []

  for (const episodeId of commonIds) {
    const before = baseline.get(episodeId)
    const after = candidate.get(episodeId)
    if (!before || !after) continue
    if (before.success && !after.success) regressions.push(before.episode)
    if (!before.success && after.success) fixes.push(before.episode)
  }

  const baselineSuccessRate = successRate(input.baselineEpisodes)
  const candidateSuccessRate = successRate(input.candidateEpisodes)
  const hasPositiveLift = candidateSuccessRate > baselineSuccessRate && fixes.length > regressions.length
  const reason = input.seedSkillsModified > 0
    ? "seed skills changed during probation"
    : regressions.length > 0
      ? `introduced ${regressions.length} regression(s)`
      : candidateSuccessRate < baselineSuccessRate
        ? "candidate slice success rate is below seed baseline"
        : input.candidateUsedSkill === false
          ? "candidate did not use probation skill"
          : !hasPositiveLift
            ? "candidate used probation skill but did not improve slice success rate"
          : "probation replay passed"

  return {
    skillId: input.skillId,
    promoted: input.seedSkillsModified === 0 &&
      regressions.length === 0 &&
      candidateSuccessRate >= baselineSuccessRate &&
      input.candidateUsedSkill !== false &&
      hasPositiveLift,
    baselineSuccessRate,
    candidateSuccessRate,
    regressions,
    fixes,
    reason,
  }
}

export async function promoteShadowSkillIfSafe(
  skillManager: SkillManager,
  input: SkillPromotionInput,
): Promise<SkillPromotionDecision> {
  const skill = await skillManager.get(input.skillId)
  const decision = evaluateSkillPromotion(input)
  if (!skill) return { ...decision, promoted: false, reason: "skill not found" }
  if (isSeedSkill(skill.spec)) return { ...decision, promoted: false, reason: "seed skills are never promoted by probation" }
  if (!isAutoSkill(skill.spec)) return { ...decision, promoted: false, reason: "only auto skills can be promoted by probation" }
  if (skill.spec.status !== "shadow") return { ...decision, promoted: false, reason: "only shadow skills can enter probation promotion" }
  if (skill.spec.origin !== "failure-pattern") return { ...decision, promoted: false, reason: "only failure-pattern patch skills can be promoted" }
  if (!decision.promoted) return decision

  await skillManager.update(input.skillId, { status: "active" })
  return decision
}

function successRate(episodes: ProbationEpisodeResult[]): number {
  if (episodes.length === 0) return 0
  return episodes.filter((episode) => episode.success).length / episodes.length
}
