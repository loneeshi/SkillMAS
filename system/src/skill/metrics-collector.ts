/**
 * MetricsCollector — captures and aggregates SkillMAS experiment metrics.
 *
 * Tracks per-episode and per-round data:
 *   - Success rate by task family
 *   - Skill retrieval/usage patterns
 *   - Skill utility delta (Q-table changes)
 *   - Skill library growth (new/revised/pruned)
 *   - Average steps and context size
 */

export interface EpisodeMetrics {
  episodeId: string
  taskFamily: string
  taskType: string
  skillsRetrieved: string[]
  skillsUsed: string[]
  steps: number
  success: boolean
  contextSize: number
  durationMs: number
  workerId?: string
  delegationCount: number
}

export interface RoundMetrics {
  roundNum: number
  totalEpisodes: number
  successRate: number
  successByFamily: Record<string, { total: number; success: number; rate: number }>
  skills: {
    total: number
    new: number
    revised: number
    pruned: number
    retrievedRatio: number
    usedRatio: number
  }
  skillUtilityDelta: Record<string, { before: number; after: number; delta: number }>
  steps: {
    avg: number
    min: number
    max: number
  }
  contextSize: {
    avg: number
    min: number
    max: number
  }
  delegation: {
    total: number
    avgPerEpisode: number
  }
}

export interface ExperimentMetrics {
  experimentId: string
  mode: ExperimentMode
  tree: string
  rounds: RoundMetrics[]
  learningCurve: Array<{ round: number; successRate: number }>
  skillGrowthCurve: Array<{ round: number; skillCount: number }>
}

export type ExperimentMode =
  | "FULL_SKILLMAS"
  | "NO_SKILL_EVOLUTION"
  | "NO_ORGANIZATION"
  | "RETRIEVAL_ONLY"
  | "FIXED_TOPOLOGY"

export class MetricsCollector {
  private episodes: EpisodeMetrics[] = []
  private roundData: Map<number, { episodes: EpisodeMetrics[] }> = new Map()
  private skillCounts: Map<number, { total: number; new: number; revised: number; pruned: number }> = new Map()
  private skillQValues: Map<string, Array<{ round: number; q: number }>> = new Map()

  recordEpisode(episode: EpisodeMetrics): void {
    this.episodes.push(episode)
  }

  recordRoundEpisodes(roundNum: number, episodes: EpisodeMetrics[]): void {
    this.roundData.set(roundNum, {
      episodes: [...episodes],
    })
  }

  recordSkillCounts(
    roundNum: number,
    counts: { total: number; new: number; revised: number; pruned: number },
  ): void {
    this.skillCounts.set(roundNum, { ...counts })
  }

  recordSkillQ(skillId: string, roundNum: number, qValue: number): void {
    if (!this.skillQValues.has(skillId)) {
      this.skillQValues.set(skillId, [])
    }
    this.skillQValues.get(skillId)!.push({ round: roundNum, q: qValue })
  }

  aggregateRound(roundNum: number): RoundMetrics | null {
    const round = this.roundData.get(roundNum)
    if (!round || round.episodes.length === 0) return null

    const episodes = round.episodes
    const totalEpisodes = episodes.length
    const successCount = episodes.filter((e) => e.success).length
    const successRate = totalEpisodes > 0 ? successCount / totalEpisodes : 0

    // Success by family
    const familyMap = new Map<string, { total: number; success: number }>()
    for (const ep of episodes) {
      const curr = familyMap.get(ep.taskFamily) ?? { total: 0, success: 0 }
      curr.total++
      if (ep.success) curr.success++
      familyMap.set(ep.taskFamily, curr)
    }
    const successByFamily: Record<string, { total: number; success: number; rate: number }> = {}
    for (const [family, counts] of familyMap) {
      successByFamily[family] = { ...counts, rate: counts.total > 0 ? counts.success / counts.total : 0 }
    }

    // Skill stats
    const skillCounts = this.skillCounts.get(roundNum) ?? { total: 0, new: 0, revised: 0, pruned: 0 }
    const allRetrieved = episodes.flatMap((e) => e.skillsRetrieved)
    const allUsed = episodes.flatMap((e) => e.skillsUsed)
    const uniqueRetrieved = new Set(allRetrieved)
    const retrievedRatio = skillCounts.total > 0 ? uniqueRetrieved.size / skillCounts.total : 0
    const usedRatio = uniqueRetrieved.size > 0
      ? new Set(allUsed.filter((s) => uniqueRetrieved.has(s))).size / uniqueRetrieved.size
      : 0

    // Q-value deltas
    const skillUtilityDelta: Record<string, { before: number; after: number; delta: number }> = {}
    for (const [skillId, history] of this.skillQValues) {
      const currentRound = history.filter((h) => h.round === roundNum)
      const prevRound = history.filter((h) => h.round === roundNum - 1)
      if (prevRound.length > 0 && currentRound.length > 0) {
        const before = prevRound[prevRound.length - 1].q
        const after = currentRound[currentRound.length - 1].q
        skillUtilityDelta[skillId] = { before, after, delta: after - before }
      }
    }

    // Step stats
    const steps = episodes.map((e) => e.steps)
    const stepStats = {
      avg: steps.length > 0 ? steps.reduce((a, b) => a + b, 0) / steps.length : 0,
      min: Math.min(...steps, 0),
      max: Math.max(...steps, 0),
    }

    // Context size stats
    const contexts = episodes.map((e) => e.contextSize)
    const contextStats = {
      avg: contexts.length > 0 ? contexts.reduce((a, b) => a + b, 0) / contexts.length : 0,
      min: Math.min(...contexts, 0),
      max: Math.max(...contexts, 0),
    }

    // Delegation stats
    const totalDelegations = episodes.reduce((sum, e) => sum + e.delegationCount, 0)

    return {
      roundNum,
      totalEpisodes,
      successRate,
      successByFamily,
      skills: {
        total: skillCounts.total,
        new: skillCounts.new,
        revised: skillCounts.revised,
        pruned: skillCounts.pruned,
        retrievedRatio,
        usedRatio,
      },
      skillUtilityDelta,
      steps: stepStats,
      contextSize: contextStats,
      delegation: {
        total: totalDelegations,
        avgPerEpisode: totalEpisodes > 0 ? totalDelegations / totalEpisodes : 0,
      },
    }
  }

  compileExperiment(
    experimentId: string,
    mode: ExperimentMode,
    tree: string,
    roundNums: number[],
  ): ExperimentMetrics {
    const rounds: RoundMetrics[] = []
    for (const r of roundNums) {
      const metrics = this.aggregateRound(r)
      if (metrics) rounds.push(metrics)
    }

    return {
      experimentId,
      mode,
      tree,
      rounds,
      learningCurve: rounds.map((r) => ({ round: r.roundNum, successRate: r.successRate })),
      skillGrowthCurve: rounds.map((r) => ({ round: r.roundNum, skillCount: r.skills.total })),
    }
  }

  clear(): void {
    this.episodes = []
    this.roundData.clear()
    this.skillCounts.clear()
    this.skillQValues.clear()
  }
}
