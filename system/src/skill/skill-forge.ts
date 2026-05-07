/**
 * SkillForge — manages the quality of existing skills via Refine and Prune.
 *
 * Separated from SkillDesigner (which creates NEW skills) to match the
 * SkillMAS skill-evolution stack:
 *   - SkillDesign (designer.ts):  creates new skills from contrastive trajectory analysis
 *   - SkillForge  (skill-forge.ts): refines, prunes, and penalizes existing skills
 *
 * SkillForge operates AFTER SkillDesign in the round loop:
 *   1. Observe whether the round dropped
 *   2. Refine (improve underperforming skills via LLM)
 *   3. Prune (disable skills that consistently harm performance)
 *
 * Important boundary:
 * SkillForge does NOT update Q-values directly. Q remains an execution-side
 * signal; Forge should only polish or prune skills based on observed patterns.
 */

import type { LLMClient } from "../llm/client"
import type { SkillManager } from "./manager"
import type { SkillQTable } from "./q-table"
import type { RoundStats } from "./designer"
import type { SkillSpec } from "../spec/skill"
import { isSeedSkill } from "./seed-skill.js"

/** Actions taken by SkillForge in a single round. */
export interface ForgeActions {
  refined: Array<{ skillId: string; reason: string; previousQ: number }>
  pruned: Array<{ skillId: string; reason: string; finalQ: number }>
  dropSignalObserved: boolean
  dropMagnitudeObserved: number
}

interface ForgeOptions {
  refineThreshold?: number
  pruneThreshold?: number
  minTrialsForPrune?: number
  /** Success rate drop threshold to trigger penalty. Default: 0.05 (5%) */
  dropThreshold?: number
}

/** Maximum skills refined per round. */
const MAX_REFINES_PER_ROUND = 3

export class SkillForge {
  private llm: LLMClient
  private qTable: SkillQTable
  private skillManager: SkillManager
  private refineThreshold: number
  private pruneThreshold: number
  private minTrialsForPrune: number
  private dropThreshold: number

  constructor(
    llm: LLMClient,
    qTable: SkillQTable,
    skillManager: SkillManager,
    options?: ForgeOptions,
  ) {
    this.llm = llm
    this.qTable = qTable
    this.skillManager = skillManager
    this.refineThreshold = options?.refineThreshold ?? 0.2
    this.pruneThreshold = options?.pruneThreshold ?? 0.05  // prune skills below 0.05 Q (not just negative)
    this.minTrialsForPrune = options?.minTrialsForPrune ?? 5  // prune faster, only need 5 trials
    this.dropThreshold = options?.dropThreshold ?? 0.02
  }

  /**
   * Main entry: refine/prune existing skills after a round.
   *
 * Executes in order: DROP SIGNAL → REFINE → PRUNE.
  */
  async forgeSkills(stats: RoundStats): Promise<ForgeActions> {
    let dropSignalObserved = false
    let dropMagnitudeObserved = 0

    const drop = this.detectDrop(stats)
    if (drop > 0) {
      dropSignalObserved = true
      dropMagnitudeObserved = drop
    }

    const refined = await this.refineUnderperformers(stats)
    const pruned = await this.pruneDeadWeight()

    return { refined, pruned, dropSignalObserved, dropMagnitudeObserved }
  }

  /**
   * Detect success rate drop magnitude. Returns drop amount (0 if no drop).
   */
  private detectDrop(stats: RoundStats): number {
    if (stats.prevSuccessRate === undefined) return 0
    const drop = stats.prevSuccessRate - stats.successRate
    return drop > this.dropThreshold ? drop : 0
  }

  /**
   * Refine underperforming skills using LLM.
   *
   * Selects skills with Q below refineThreshold that have enough trials,
   * gathers failure/success traces, and asks LLM to improve them.
   */
  private async refineUnderperformers(
    stats: RoundStats,
  ): Promise<Array<{ skillId: string; reason: string; previousQ: number }>> {
    const underperformers = this.qTable.getUnderperformers(
      this.refineThreshold,
      Math.max(3, Math.floor(this.minTrialsForPrune / 2)),
    )

    const refined: Array<{ skillId: string; reason: string; previousQ: number }> = []
    const toRefine = underperformers.slice(0, MAX_REFINES_PER_ROUND)

    for (const skillId of toRefine) {
      const qStats = this.qTable.getStats(skillId)
      if (!qStats) continue

      const existing = await this.skillManager.get(skillId)
      if (!existing) continue
      if (isSeedSkill(existing.spec)) continue

      const failureSummaries = this.getFailureSummariesForSkill(skillId, stats)
      const successSummaries = this.getSuccessSummariesForSkill(skillId, stats)

      try {
        const improved = await this.refineWithLLM(
          existing.content,
          qStats.globalQ,
          qStats.globalN,
          failureSummaries,
          successSummaries,
        )

        await this.skillManager.update(skillId, {}, improved)

        refined.push({
          skillId,
          reason: `Q=${qStats.globalQ.toFixed(2)} after ${qStats.globalN} trials (threshold: ${this.refineThreshold})`,
          previousQ: qStats.globalQ,
        })
      } catch {
        // LLM refinement failed — skip
      }
    }

    return refined
  }

  /**
   * Prune low-utility skills.
   *
   * Skills with Q below pruneThreshold after enough trials are disabled.
   * Manual (seed) skills are protected from pruning.
   */
  private async pruneDeadWeight(): Promise<Array<{ skillId: string; reason: string; finalQ: number }>> {
    const candidates = this.qTable.getPruneCandidates(this.pruneThreshold, this.minTrialsForPrune)
    const pruned: Array<{ skillId: string; reason: string; finalQ: number }> = []

    for (const skillId of candidates) {
      const qStats = this.qTable.getStats(skillId)
      if (!qStats) continue

      const existing = await this.skillManager.get(skillId)
      if (!existing) continue

      if (isSeedSkill(existing.spec)) continue

      await this.skillManager.update(skillId, {
        status: "disabled",
      })

      pruned.push({
        skillId,
        reason: `Q=${qStats.globalQ.toFixed(2)} after ${qStats.globalN} trials — below prune threshold ${this.pruneThreshold}`,
        finalQ: qStats.globalQ,
      })
    }

    return pruned
  }

  /* ----- LLM interaction ----- */

  private async refineWithLLM(
    currentContent: string,
    qValue: number,
    nTrials: number,
    failureSummaries: string[],
    successSummaries: string[],
  ): Promise<string> {
    const failureBlock = failureSummaries.length > 0
      ? failureSummaries.join("\n")
      : "(no failures recorded)"
    const successBlock = successSummaries.length > 0
      ? successSummaries.join("\n")
      : "(no successes recorded)"

    const prompt = `This skill has Q-value ${qValue.toFixed(2)} after ${nTrials} uses (threshold: ${this.refineThreshold}).

Recent failures when this skill was active:
${failureBlock}

Recent successes when this skill was active:
${successBlock}

Current skill content:
${currentContent}

Improve the skill to increase success rate. Keep what works, fix what doesn't.
Output ONLY the improved markdown content.`

    const response = await this.llm.chat(
      [
        { role: "system", content: "You are a skill improvement expert. Output ONLY the improved markdown content." },
        { role: "user", content: prompt },
      ],
      { temperature: 0.3 },
    )

    return response.content.trim()
  }

  /* ----- Trace analysis helpers ----- */

  private getFailureSummariesForSkill(skillId: string, stats: RoundStats): string[] {
    return stats.failedEpisodes
      .filter((ep) => flattenSkills(ep.skillsUsed).includes(skillId))
      .slice(0, 5)
      .map((ep) => {
        const err = ep.errorMessage ? ` | Error: ${ep.errorMessage}` : ""
        const summary = ep.summary ? ` | Summary: ${ep.summary}` : ""
        return `- [${ep.taskType}] ${ep.task} (${ep.steps} steps${err}${summary})`
      })
  }

  private getSuccessSummariesForSkill(skillId: string, stats: RoundStats): string[] {
    return stats.successfulEpisodes
      .filter((ep) => flattenSkills(ep.skillsUsed).includes(skillId))
      .slice(0, 5)
      .map((ep) => {
        const summary = ep.summary ? ` | Summary: ${ep.summary}` : ""
        return `- [${ep.taskType}] ${ep.task} (${ep.steps} steps${summary})`
      })
  }
}

/* ----- Helpers ----- */

function flattenSkills(skillsUsed: Record<string, string[]>): string[] {
  const all = new Set<string>()
  for (const ids of Object.values(skillsUsed)) {
    for (const id of ids) all.add(id)
  }
  return [...all]
}
