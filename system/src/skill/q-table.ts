/**
 * SkillQTable — non-parametric Q-value storage for skill utility estimation.
 *
 * Inspired by MemRL's two-phase retrieval:
 *   score = (1-λ) * similarity + λ * Q_normalized + c * UCB_bonus
 *
 * Lambda is fixed at 0.3 (similarity-weighted with moderate Q-value influence).
 *
 * Three levels of Q-values:
 *   - Global: Q(skill) — overall utility
 *   - Task-conditioned: Q(skill, taskType) — utility for specific task types
 *   - Agent-conditioned: Q(skill, agentId) — utility for specific agents
 */

import { readFile, writeFile, mkdir } from "node:fs/promises"
import { dirname, join } from "node:path"

function normalizeSkillId(skillId: string): string {
  return skillId.replace(/\\/g, "/")
}

/** Single Q-value entry with visit count. */
export interface QEntry {
  q: number
  n: number
}

/** Full Q-value record for a single skill. */
export interface SkillQRecord {
  skillId: string
  globalQ: QEntry
  taskQ: Record<string, QEntry>
  agentQ: Record<string, QEntry>
  contentVersions: Array<{
    version: number
    timestamp: string
    contentHash: string
    avgReward: number
    n: number
  }>
  createdAt: string
  lastUsed: string
  totalReward: number
}

interface QTableOptions {
  alpha?: number
  alphaDecay?: number
  explorationCoeff?: number
  /** File path to auto-save to. If set, enableAutosave() will be called automatically. */
  autosavePath?: string
  /** Debounce delay in ms for auto-save (default: 5000) */
  autosaveDebounceMs?: number
}

/** Default initial Q-value — only used when no existing records exist yet. */
const Q_INIT = 0.5

/** Maximum UCB exploration bonus — lower cap reduces over-exploration of new auto-skills. */
const MAX_UCB = 0.15

/** Fixed lambda: balance between similarity (1-λ) and Q-value (λ). */
const LAMBDA = 0.3

export class SkillQTable {
  private records: Map<string, SkillQRecord> = new Map()
  private alpha: number
  private alphaDecay: number
  private explorationCoeff: number
  private totalSelections: number = 0
  private autosavePath: string | undefined
  private autosaveDebounceMs: number
  private autosaveTimer: ReturnType<typeof setTimeout> | undefined

  constructor(options?: QTableOptions) {
    this.alpha = options?.alpha ?? 1
    this.alphaDecay = options?.alphaDecay ?? 1
    this.explorationCoeff = options?.explorationCoeff ?? 1.0
    this.autosavePath = options?.autosavePath
    this.autosaveDebounceMs = options?.autosaveDebounceMs ?? 5000
    if (options?.autosavePath) {
      this.enableAutosave(options.autosavePath, options.autosaveDebounceMs)
    }
  }

  /**
   * Initialize a skill's Q-value.
   *
   * New skills inherit the current average globalQ of existing trained records
   * rather than a fixed optimistic prior. This prevents newly generated skills
   * from automatically outscoring battle-tested ones.
   */
  initialize(skillId: string): void {
    const normalizedId = normalizeSkillId(skillId)
    if (this.records.has(normalizedId)) return
    const initQ = this.getAverageGlobalQ()
    this.records.set(normalizedId, {
      skillId: normalizedId,
      globalQ: { q: initQ, n: 0 },
      taskQ: {},
      agentQ: {},
      contentVersions: [],
      createdAt: new Date().toISOString(),
      lastUsed: new Date().toISOString(),
      totalReward: 0,
    })
  }

  /** Average globalQ across all records that have been used at least once, or Q_INIT if none. */
  private getAverageGlobalQ(): number {
    let sum = 0, count = 0
    for (const rec of this.records.values()) {
      if (rec.globalQ.n > 0) { sum += rec.globalQ.q; count++ }
    }
    return count > 0 ? sum / count : Q_INIT
  }

  /**
   * Update Q-values after an episode.
   *
   * Uses Monte Carlo-style update (MemRL):
   *   Q(s) ← Q(s) + α_eff * (R - Q(s))
   *
   * where the default α_eff = 1 / (1 + n), matching count-based
   * empirical success-probability estimation.
   */
  update(skillId: string, context: { taskType: string; agentId: string; reward: number }): void {
    const normalizedId = normalizeSkillId(skillId)
    this.ensureRecord(normalizedId)
    const rec = this.records.get(normalizedId)!
    const { taskType, agentId, reward } = context

    rec.lastUsed = new Date().toISOString()
    rec.totalReward += reward

    this.updateEntry(rec.globalQ, reward)

    if (!rec.taskQ[taskType]) rec.taskQ[taskType] = { q: rec.globalQ.q, n: 0 }
    this.updateEntry(rec.taskQ[taskType], reward)

    if (!rec.agentQ[agentId]) rec.agentQ[agentId] = { q: rec.globalQ.q, n: 0 }
    this.updateEntry(rec.agentQ[agentId], reward)

    this.totalSelections++
    this.scheduleAutosave()
  }

  /**
   * Get composite score for skill selection.
   *
   * score = (1-λ) * sim_norm + λ * Q_norm + c * sqrt(ln(N)/n)
   *
   * Where Q_combined = 0.5 * globalQ + 0.3 * taskQ + 0.2 * agentQ
   */
  getScore(
    skillId: string,
    context: { taskType: string; agentId: string; similarity: number },
  ): number {
    const rec = this.records.get(normalizeSkillId(skillId))
    if (!rec) return context.similarity

    const combinedQ = this.getCombinedQ(rec, context.taskType, context.agentId)
    const ucb = this.getUCBBonus(rec.globalQ.n)

    return (1 - LAMBDA) * context.similarity + LAMBDA * combinedQ + ucb
  }

  /**
   * Select top-K skills from candidates using composite scoring.
   *
   * Applies z-score normalization to both similarity and Q-values
   * before combining, then sorts by score descending.
   */
  selectTopK(
    candidates: string[],
    context: { taskType: string; agentId: string; similarities: Record<string, number> },
    k: number,
  ): Array<{ skillId: string; score: number; reason: string }> {
    if (candidates.length === 0) return []

    const rawSims: number[] = []
    const rawQs: number[] = []
    const candidateData: Array<{ skillId: string; sim: number; q: number; n: number }> = []

    for (const id of candidates) {
      const normalizedId = normalizeSkillId(id)
      const sim = context.similarities[id] ?? context.similarities[normalizedId] ?? 0
      const rec = this.records.get(normalizedId)
      const q = rec ? this.getCombinedQ(rec, context.taskType, context.agentId) : Q_INIT
      const n = rec?.globalQ.n ?? 0
      rawSims.push(sim)
      rawQs.push(q)
      candidateData.push({ skillId: normalizedId, sim, q, n })
    }

    const normSims = zNormalize(rawSims)
    const normQs = zNormalize(rawQs)

    const scored: Array<{ skillId: string; score: number; reason: string }> = []

    for (let i = 0; i < candidateData.length; i++) {
      const { skillId, n } = candidateData[i]
      const simNorm = normSims[i]
      const qNorm = normQs[i]
      const ucb = this.getUCBBonus(n)

      const simComponent = (1 - LAMBDA) * simNorm
      const qComponent = LAMBDA * qNorm
      const score = simComponent + qComponent + ucb

      let reason: "q_value" | "similarity" | "exploration"
      if (ucb > simComponent && ucb > qComponent) reason = "exploration"
      else if (qComponent >= simComponent) reason = "q_value"
      else reason = "similarity"

      scored.push({ skillId, score, reason })
    }

    scored.sort((a, b) => b.score - a.score)
    return scored.slice(0, k)
  }

  /** Get Q-value stats for a skill (for designer decisions). */
  getStats(skillId: string): { globalQ: number; globalN: number; taskQ: Record<string, QEntry>; avgReward: number } | undefined {
    const rec = this.records.get(normalizeSkillId(skillId))
    if (!rec) return undefined
    return {
      globalQ: rec.globalQ.q,
      globalN: rec.globalQ.n,
      taskQ: { ...rec.taskQ },
      avgReward: rec.globalQ.n > 0 ? rec.totalReward / rec.globalQ.n : 0,
    }
  }

  /** Record a content version for rollback tracking. */
  recordContentVersion(skillId: string, contentHash: string): void {
    const normalizedId = normalizeSkillId(skillId)
    this.ensureRecord(normalizedId)
    const rec = this.records.get(normalizedId)!
    const nextVersion = rec.contentVersions.length + 1
    rec.contentVersions.push({
      version: nextVersion,
      timestamp: new Date().toISOString(),
      contentHash,
      avgReward: rec.globalQ.n > 0 ? rec.totalReward / rec.globalQ.n : 0,
      n: rec.globalQ.n,
    })
  }

  /** Get underperforming skills whose Q is below threshold with enough trials. */
  getUnderperformers(threshold: number, minTrials: number): string[] {
    const result: string[] = []
    for (const [id, rec] of this.records) {
      if (rec.globalQ.n >= minTrials && rec.globalQ.q < threshold) {
        result.push(id)
      }
    }
    return result
  }

  /** Get low-value skills that are candidates for pruning. */
  getPruneCandidates(threshold: number, minTrials: number): string[] {
    const result: string[] = []
    for (const [id, rec] of this.records) {
      if (rec.globalQ.n >= minTrials && rec.globalQ.q < threshold) {
        result.push(id)
      }
    }
    return result
  }

  /** Get all tracked skill IDs. */
  getSkillIds(): string[] {
    return [...this.records.keys()]
  }

  /** Get total selection count (N for UCB). */
  getTotalSelections(): number {
    return this.totalSelections
  }

  /* ----- Persistence ----- */

  enableAutosave(filePath: string, debounceMs?: number): void {
    this.autosavePath = filePath
    this.autosaveDebounceMs = debounceMs ?? 5000
  }

  async flushAutosave(): Promise<void> {
    if (this.autosaveTimer) {
      clearTimeout(this.autosaveTimer)
      this.autosaveTimer = undefined
    }
    if (this.autosavePath) {
      await this.save(this.autosavePath)
    }
  }

  async save(filePath: string): Promise<void> {
    await mkdir(dirname(filePath), { recursive: true })
    const data = JSON.stringify(this.toJSON(), null, 2)
    await writeFile(filePath, data, "utf-8")
  }

  /** Backward-compatible artifact writer used by the example runners. */
  async saveUtilityArtifacts(dir: string): Promise<void> {
    await this.save(join(dir, "q-table.json"))
  }

  static async load(filePath: string): Promise<SkillQTable> {
    const raw = await readFile(filePath, "utf-8")
    const data = JSON.parse(raw)
    return SkillQTable.fromJSON(data)
  }

  /** Backward-compatible artifact loader used by the example runners. */
  static async loadFromArtifacts(dir: string): Promise<SkillQTable> {
    return SkillQTable.load(join(dir, "q-table.json"))
  }

  toJSON(): object {
    return {
      alpha: this.alpha,
      alphaDecay: this.alphaDecay,
      explorationCoeff: this.explorationCoeff,
      totalSelections: this.totalSelections,
      records: Object.fromEntries(this.records),
    }
  }

  static fromJSON(data: Record<string, unknown>): SkillQTable {
    const table = new SkillQTable({
      alpha: data.alpha as number | undefined,
      alphaDecay: data.alphaDecay as number | undefined,
      explorationCoeff: data.explorationCoeff as number | undefined,
    })
    table.totalSelections = (data.totalSelections as number) ?? 0

    const records = data.records as Record<string, SkillQRecord> | undefined
    if (records) {
      for (const [id, rec] of Object.entries(records)) {
        const normalizedId = normalizeSkillId(id)
        const normalizedRec: SkillQRecord = {
          ...rec,
          skillId: normalizeSkillId(rec.skillId ?? id),
        }
        const existing = table.records.get(normalizedId)
        if (!existing) {
          table.records.set(normalizedId, normalizedRec)
          continue
        }

        existing.globalQ = pickMoreInformativeEntry(existing.globalQ, normalizedRec.globalQ)
        existing.totalReward += normalizedRec.totalReward
        existing.createdAt = existing.createdAt <= normalizedRec.createdAt ? existing.createdAt : normalizedRec.createdAt
        existing.lastUsed = existing.lastUsed >= normalizedRec.lastUsed ? existing.lastUsed : normalizedRec.lastUsed

        for (const [taskType, entry] of Object.entries(normalizedRec.taskQ)) {
          existing.taskQ[taskType] = pickMoreInformativeEntry(existing.taskQ[taskType], entry)
        }
        for (const [agentId, entry] of Object.entries(normalizedRec.agentQ)) {
          existing.agentQ[agentId] = pickMoreInformativeEntry(existing.agentQ[agentId], entry)
        }

        const versions = [...existing.contentVersions, ...normalizedRec.contentVersions]
        const deduped = new Map<string, SkillQRecord["contentVersions"][number]>()
        for (const version of versions) {
          deduped.set(`${version.version}:${version.contentHash}:${version.timestamp}`, version)
        }
        existing.contentVersions = [...deduped.values()].sort((a, b) => a.version - b.version)
      }
    }
    return table
  }

  /* ----- Private helpers ----- */

  private scheduleAutosave(): void {
    if (!this.autosavePath) return
    if (this.autosaveTimer) clearTimeout(this.autosaveTimer)
    this.autosaveTimer = setTimeout(() => {
      this.autosaveTimer = undefined
      if (this.autosavePath) {
        this.save(this.autosavePath).catch((err) => {
          console.warn(`[SkillQTable] Auto-save failed: ${err instanceof Error ? err.message : String(err)}`)
        })
      }
    }, this.autosaveDebounceMs)
  }

  private ensureRecord(skillId: string): void {
    const normalizedId = normalizeSkillId(skillId)
    if (!this.records.has(normalizedId)) this.initialize(normalizedId)
  }

  /** Apply adaptive-alpha MC update to a single QEntry. */
  private updateEntry(entry: QEntry, reward: number): void {
    const effectiveAlpha = this.alpha / (1 + entry.n * this.alphaDecay)
    entry.q += effectiveAlpha * (reward - entry.q)
    entry.n += 1
  }

  /** Weighted combination: 0.5 * global + 0.3 * task + 0.2 * agent. */
  private getCombinedQ(rec: SkillQRecord, taskType: string, agentId: string): number {
    const gQ = rec.globalQ.q
    // For missing task/agent dimensions, fall back to the record's own globalQ
    // rather than Q_INIT to avoid inflating scores for untested contexts.
    const tQ = rec.taskQ[taskType]?.q ?? gQ
    const aQ = rec.agentQ[agentId]?.q ?? gQ
    return 0.5 * gQ + 0.3 * tQ + 0.2 * aQ
  }

  /** UCB exploration bonus: c * sqrt(ln(N+1) / max(1, n)), capped to prevent new skills from dominating. */
  private getUCBBonus(n: number): number {
    const raw = this.explorationCoeff * Math.sqrt(
      Math.log(this.totalSelections + 1) / Math.max(1, n),
    )
    return Math.min(raw, MAX_UCB)
  }
}

/* ----- Utility: z-score normalization ----- */

function pickMoreInformativeEntry(primary: QEntry | undefined, secondary: QEntry | undefined): QEntry {
  if (!primary) return secondary ? { ...secondary } : { q: Q_INIT, n: 0 }
  if (!secondary) return { ...primary }
  return secondary.n > primary.n ? { ...secondary } : { ...primary }
}

function zNormalize(values: number[]): number[] {
  if (values.length === 0) return []
  if (values.length === 1) return [0.5]

  const mean = values.reduce((s, v) => s + v, 0) / values.length
  const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length
  const std = Math.sqrt(variance)

  if (std < 1e-8) return values.map(() => 0.5)

  return values.map((v) => {
    const z = (v - mean) / std
    return 1 / (1 + Math.exp(-z))
  })
}
