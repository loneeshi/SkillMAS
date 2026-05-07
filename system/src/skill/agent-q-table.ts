/**
 * AgentQTable — Q-value storage for agent routing decisions.
 *
 * Two-level Q architecture:
 *   - Q_agent(agent, taskType) — used by manager for routing
 *   - Q_skill(skill, taskType) — used within agent (in SkillQTable)
 *
 * This class handles the agent-level Q values. Manager uses these
 * to decide which worker to delegate to for a given task type.
 *
 * New agents created by extension get Q = baseline + exploration_bonus
 * to encourage exploration of new capabilities.
 */

import { readFile, writeFile, mkdir } from "node:fs/promises"
import { dirname, join } from "node:path"

/** Single Q-value entry with visit count. */
export interface AgentQEntry {
  q: number
  n: number
}

/** Full Q-value record for a single agent. */
export interface AgentQRecord {
  agentId: string
  globalQ: AgentQEntry
  taskQ: Record<string, AgentQEntry>
  createdAt: string
  lastUsed: string
  totalReward: number
  /** True if this agent was created by extension (vs manual). */
  isExtension: boolean
  /** Task types this agent was designed to handle (from extension trigger). */
  targetTaskTypes?: string[]
}

interface AgentQTableOptions {
  alpha?: number
  alphaDecay?: number
  explorationBonus?: number
}

/** Default initial Q-value — neutral prior. */
const Q_INIT = 0.5

/** Bonus added to baseline for new extension agents. */
const DEFAULT_EXPLORATION_BONUS = 0.1

/**
 * Multiplicative penalty applied to extension agents when evaluated
 * on task types outside their declared target. Keeps specialists
 * focused while still allowing occasional exploration via softmax.
 */
const OFF_TARGET_PENALTY = 0.5

export class AgentQTable {
  private records: Map<string, AgentQRecord> = new Map()
  private alpha: number
  private alphaDecay: number
  private explorationBonus: number
  private totalSelections: number = 0

  constructor(options?: AgentQTableOptions) {
    this.alpha = options?.alpha ?? 1
    this.alphaDecay = options?.alphaDecay ?? 1
    this.explorationBonus = options?.explorationBonus ?? DEFAULT_EXPLORATION_BONUS
  }

  /** Initialize an agent with neutral Q-value. */
  initialize(agentId: string, isExtension = false, targetTaskTypes?: string[]): void {
    if (this.records.has(agentId)) return
    this.records.set(agentId, {
      agentId,
      globalQ: { q: Q_INIT, n: 0 },
      taskQ: {},
      createdAt: new Date().toISOString(),
      lastUsed: new Date().toISOString(),
      totalReward: 0,
      isExtension,
      targetTaskTypes,
    })
  }

  /**
   * Initialize a new extension agent with Q slightly above baseline.
   *
   * This encourages the system to try new agents while still
   * requiring them to prove their worth.
   */
  initializeExtensionAgent(
    agentId: string,
    baseline: number,
    targetTaskTypes?: string[],
    explorationBonusOverride?: number,
  ): void {
    if (this.records.has(agentId)) return

    const bonus = explorationBonusOverride ?? this.explorationBonus
    const initialQ = Math.min(1.0, baseline + bonus)

    const taskQ: Record<string, AgentQEntry> = {}
    if (targetTaskTypes) {
      for (const tt of targetTaskTypes) {
        // Give even higher initial Q for the specific task types
        // the agent was designed to handle
        taskQ[tt] = { q: Math.min(1.0, baseline + bonus * 2), n: 0 }
      }
    }

    this.records.set(agentId, {
      agentId,
      globalQ: { q: initialQ, n: 0 },
      taskQ,
      createdAt: new Date().toISOString(),
      lastUsed: new Date().toISOString(),
      totalReward: 0,
      isExtension: true,
      targetTaskTypes,
    })
  }

  /**
   * Update Q-values after an episode.
   *
   * Uses Monte Carlo-style update with default α_eff = 1 / (1 + n):
   *   Q(agent, taskType) ← Q + α_eff * (R - Q)
   */
  update(agentId: string, taskType: string, reward: number): void {
    this.ensureRecord(agentId)
    const rec = this.records.get(agentId)!

    rec.lastUsed = new Date().toISOString()
    rec.totalReward += reward

    // Update global Q
    this.updateEntry(rec.globalQ, reward)

    // Update task-specific Q
    if (!rec.taskQ[taskType]) rec.taskQ[taskType] = { q: Q_INIT, n: 0 }
    this.updateEntry(rec.taskQ[taskType], reward)

    this.totalSelections++
  }

  /**
   * Get Q value for an agent on a specific task type.
   *
   * Returns weighted combination: 0.6 * taskQ + 0.4 * globalQ
   * Falls back to global Q if no task-specific data.
   *
   * Extension agents receive a penalty when evaluated on task types
   * outside their declared target, preventing them from being routed
   * to tasks they weren't designed for.
   */
  getQ(agentId: string, taskType: string): number {
    const rec = this.records.get(agentId)
    if (!rec) return Q_INIT

    const globalQ = rec.globalQ.q
    const taskQ = rec.taskQ[taskType]?.q ?? globalQ

    // Weight task-specific Q more heavily if we have data
    const taskN = rec.taskQ[taskType]?.n ?? 0
    let q: number
    if (taskN >= 3) {
      q = 0.6 * taskQ + 0.4 * globalQ
    } else {
      q = globalQ
    }

    // Penalize extension agents on non-target task types.
    // This keeps specialized agents focused on what they were created for,
    // while still allowing them to be tried occasionally (softmax exploration).
    if (rec.isExtension && rec.targetTaskTypes && rec.targetTaskTypes.length > 0) {
      if (!rec.targetTaskTypes.includes(taskType)) {
        q *= OFF_TARGET_PENALTY
      }
    }

    return q
  }

  /**
   * Select best agent for a task type using softmax with temperature.
   *
   * Returns agents sorted by probability (highest first).
   */
  selectAgent(
    candidates: string[],
    taskType: string,
    temperature: number = 0.5,
  ): Array<{ agentId: string; q: number; prob: number }> {
    if (candidates.length === 0) return []
    if (candidates.length === 1) {
      const q = this.getQ(candidates[0], taskType)
      return [{ agentId: candidates[0], q, prob: 1.0 }]
    }

    const scores = candidates.map((id) => ({
      agentId: id,
      q: this.getQ(id, taskType),
    }))

    // Softmax
    const maxQ = Math.max(...scores.map((s) => s.q))
    const exps = scores.map((s) => Math.exp((s.q - maxQ) / Math.max(0.01, temperature)))
    const sumExp = exps.reduce((a, b) => a + b, 0)
    const probs = exps.map((e) => e / sumExp)

    const result = scores.map((s, i) => ({
      ...s,
      prob: probs[i],
    }))

    return result.sort((a, b) => b.prob - a.prob)
  }

  /** Get stats for an agent. */
  getStats(agentId: string): { globalQ: number; globalN: number; taskQ: Record<string, AgentQEntry> } | undefined {
    const rec = this.records.get(agentId)
    if (!rec) return undefined
    return {
      globalQ: rec.globalQ.q,
      globalN: rec.globalQ.n,
      taskQ: { ...rec.taskQ },
    }
  }

  /** Check if an agent is from extension. */
  isExtensionAgent(agentId: string): boolean {
    return this.records.get(agentId)?.isExtension ?? false
  }

  /** Get the full Q-record for an agent (read-only inspection). */
  getRecord(agentId: string): Readonly<AgentQRecord> | undefined {
    return this.records.get(agentId)
  }

  /** Get all tracked agent IDs. */
  getAgentIds(): string[] {
    return [...this.records.keys()]
  }

  /** Get current baseline (average Q across all agents). */
  getBaseline(): number {
    const ids = this.getAgentIds()
    if (ids.length === 0) return Q_INIT
    const sum = ids.reduce((s, id) => s + (this.records.get(id)?.globalQ.q ?? Q_INIT), 0)
    return sum / ids.length
  }

  /** Get baseline for a specific task type. */
  getTaskBaseline(taskType: string): number {
    const ids = this.getAgentIds()
    const withData = ids.filter((id) => (this.records.get(id)?.taskQ[taskType]?.n ?? 0) > 0)
    if (withData.length === 0) return this.getBaseline()
    const sum = withData.reduce((s, id) => s + (this.records.get(id)?.taskQ[taskType]?.q ?? Q_INIT), 0)
    return sum / withData.length
  }

  /* ----- Persistence ----- */

  async save(filePath: string): Promise<void> {
    await mkdir(dirname(filePath), { recursive: true })
    const data = JSON.stringify(this.toJSON(), null, 2)
    await writeFile(filePath, data, "utf-8")
  }

  /** Backward-compatible artifact writer used by the example runners. */
  async saveUtilityArtifacts(dir: string): Promise<void> {
    await this.save(join(dir, "agent-q-table.json"))
  }

  static async load(filePath: string): Promise<AgentQTable> {
    const raw = await readFile(filePath, "utf-8")
    const data = JSON.parse(raw)
    return AgentQTable.fromJSON(data)
  }

  /** Backward-compatible artifact loader used by the example runners. */
  static async loadFromArtifacts(dir: string): Promise<AgentQTable> {
    return AgentQTable.load(join(dir, "agent-q-table.json"))
  }

  toJSON(): object {
    return {
      alpha: this.alpha,
      alphaDecay: this.alphaDecay,
      explorationBonus: this.explorationBonus,
      totalSelections: this.totalSelections,
      records: Object.fromEntries(this.records),
    }
  }

  static fromJSON(data: Record<string, unknown>): AgentQTable {
    const table = new AgentQTable({
      alpha: data.alpha as number | undefined,
      alphaDecay: data.alphaDecay as number | undefined,
      explorationBonus: data.explorationBonus as number | undefined,
    })
    table.totalSelections = (data.totalSelections as number) ?? 0

    const records = data.records as Record<string, AgentQRecord> | undefined
    if (records) {
      for (const [id, rec] of Object.entries(records)) {
        table.records.set(id, rec)
      }
    }
    return table
  }

  /* ----- Private helpers ----- */

  private ensureRecord(agentId: string): void {
    if (!this.records.has(agentId)) this.initialize(agentId)
  }

  /** Apply adaptive-alpha MC update to a single entry. */
  private updateEntry(entry: AgentQEntry, reward: number): void {
    const effectiveAlpha = this.alpha / (1 + entry.n * this.alphaDecay)
    entry.q += effectiveAlpha * (reward - entry.q)
    entry.n += 1
  }
}
