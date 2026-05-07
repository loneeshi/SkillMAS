/**
 * RewardCalculator — computes multi-dimensional rewards for SkillRL.
 *
 * Reward formula:
 *   R = w_s * R_success + w_e * R_efficiency + w_u * R_utilization
 *
 * Design choices:
 *   - Asymmetric success: failures penalized at -0.3 (not -1.0) to avoid
 *     catastrophic Q collapse for skills that are sometimes useful.
 *   - Efficiency only rewards successes: no point measuring step efficiency on failures.
 *   - Utilization bonus: skills actively used in successes get a small uplift.
 *   - Utility updates are trace-grounded: only actually used skills receive updates.
 */

/** Episode outcome data needed for reward computation. */
export interface EpisodeOutcome {
  success: boolean
  steps: number
  maxSteps: number
  delegateCalls: number
  taskType: string
  /** agentId → skill IDs actually used during the episode. */
  skillsUsed: Record<string, string[]>
  /** Skills selected as candidates. Retained for analysis only, not direct reward updates. */
  candidateSkills: string[]
  /** Optional auxiliary protocol/quality signal in [-1, 1]. Main-safe: defaults to 0 when absent. */
  auxiliaryReward?: number
}

/** Computed rewards for an episode. */
export interface SkillRewards {
  /** skillId → reward for each skill actually used. */
  perSkill: Record<string, number>
  /** Overall episode reward (before per-skill assignment). */
  episodeReward: number
}

interface RewardWeights {
  success?: number
  efficiency?: number
  utilization?: number
  auxiliary?: number
}

const DEFAULT_WEIGHTS = {
  success: 0.7,
  efficiency: 0.2,
  utilization: 0.1,
  auxiliary: 0.0,
}

/** Reward for used skills on a successful episode. */
const UTILIZATION_BONUS = 0.1
export class RewardCalculator {
  private weights: { success: number; efficiency: number; utilization: number; auxiliary: number }

  constructor(weights?: RewardWeights) {
    this.weights = {
      success: weights?.success ?? DEFAULT_WEIGHTS.success,
      efficiency: weights?.efficiency ?? DEFAULT_WEIGHTS.efficiency,
      utilization: weights?.utilization ?? DEFAULT_WEIGHTS.utilization,
      auxiliary: weights?.auxiliary ?? DEFAULT_WEIGHTS.auxiliary,
    }
  }

  /**
   * Calculate per-skill rewards from an episode outcome.
   *
   * Used skills receive the full composite reward + utilization bonus.
   * Candidate-only skills do not receive direct utility updates.
   */
  calculate(outcome: EpisodeOutcome): SkillRewards {
    const rSuccess = outcome.success ? 1.0 : -0.3
    const rEfficiency = outcome.success
      ? Math.max(0, 1 - outcome.steps / outcome.maxSteps)
      : 0
    const rUtilization = outcome.success ? UTILIZATION_BONUS : 0
    const rAuxiliary = Math.max(-1, Math.min(1, outcome.auxiliaryReward ?? 0))
 
    const episodeReward =
      this.weights.success * rSuccess +
      this.weights.efficiency * rEfficiency +
      this.weights.utilization * rUtilization +
      this.weights.auxiliary * rAuxiliary


    const perSkill: Record<string, number> = {}

    const usedSkillIds = new Set<string>()
    for (const skillIds of Object.values(outcome.skillsUsed)) {
      for (const id of skillIds) {
        usedSkillIds.add(id)
      }
    }

    for (const id of usedSkillIds) {
      perSkill[id] = episodeReward
    }

    return { perSkill, episodeReward }
  }
}
