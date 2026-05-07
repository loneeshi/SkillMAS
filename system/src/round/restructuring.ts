import type { AgentStore } from "../spec/store.js"
import type { AgentSpec } from "../spec/agent.js"
import type { AgentQTable } from "../skill/agent-q-table.js"
import type { RetainedEvidence } from "./retention.js"

export type RestructuringAction = "keep" | "add" | "merge_remove" | "modify"

export interface StructuralArtifact {
  taskType: string
  failures: number
  successes: number
  implicatedExecutors: string[]
  implicatedSkills: string[]
  executorUtilities: Record<string, number>
  skillOverlap: number
  pendingSkillIds: string[]
  reasons: string[]
}

export interface RestructuringDecision {
  action: RestructuringAction
  reason: string
  artifact?: StructuralArtifact
  add?: {
    agentId: string
    spec: AgentSpec
    prompt: string
  }
  modify?: Array<{
    agentId: string
    specUpdates: Partial<AgentSpec>
    promptAppendix?: string
  }>
  mergeRemove?: {
    removeAgentId: string
    targetAgentId: string
    transferredSkills: string[]
  }
}

export interface RestructuringResult {
  decision: RestructuringDecision
  changed: boolean
  patchedAgentIds: string[]
  createdAgentIds: string[]
  removedAgentIds: string[]
}

export interface RestructuringOptions {
  minFailureMass?: number
  lowUtilityThreshold?: number
  overlapThreshold?: number
}

export class MASRestructurer {
  private agentStore: AgentStore
  private agentQTable: AgentQTable
  private minFailureMass: number
  private lowUtilityThreshold: number
  private overlapThreshold: number

  constructor(agentStore: AgentStore, agentQTable: AgentQTable, options?: RestructuringOptions) {
    this.agentStore = agentStore
    this.agentQTable = agentQTable
    this.minFailureMass = options?.minFailureMass ?? 2
    this.lowUtilityThreshold = options?.lowUtilityThreshold ?? 0.45
    this.overlapThreshold = options?.overlapThreshold ?? 0.5
  }

  async buildArtifacts(
    evidence: RetainedEvidence[],
    pendingSkillIds: string[],
  ): Promise<StructuralArtifact[]> {
    const groups = new Map<string, RetainedEvidence[]>()
    for (const item of evidence) {
      const arr = groups.get(item.trace.taskType) ?? []
      arr.push(item)
      groups.set(item.trace.taskType, arr)
    }

    const artifacts: StructuralArtifact[] = []
    for (const [taskType, items] of groups) {
      const failures = items.filter((item) => !item.trace.success).length
      const successes = items.filter((item) => item.trace.success).length
      const implicatedExecutors = unique(items.flatMap((item) => item.trace.executors))
      const implicatedSkills = unique(items.flatMap((item) => flattenSkillMap(item.trace.usedSkills)))
      const executorUtilities = Object.fromEntries(
        implicatedExecutors.map((agentId) => [agentId, this.agentQTable.getQ(agentId, taskType)]),
      )
      const skillOverlap = await this.computeSkillOverlap(implicatedExecutors)
      const reasons = unique(items.flatMap((item) => item.reasons))

      artifacts.push({
        taskType,
        failures,
        successes,
        implicatedExecutors,
        implicatedSkills,
        executorUtilities,
        skillOverlap,
        pendingSkillIds: pendingSkillIds.filter((skillId) => implicatedSkills.includes(skillId)),
        reasons,
      })
    }

    return artifacts.sort((a, b) => b.failures - a.failures)
  }

  async decide(artifacts: StructuralArtifact[]): Promise<RestructuringDecision> {
    const candidates = artifacts.filter((artifact) => artifact.failures >= this.minFailureMass)
    if (candidates.length === 0) {
      return { action: "keep", reason: "no retained failure cluster reached restructuring mass" }
    }

    const artifact = candidates[0]
    const agents = await this.agentStore.list()
    const existingIds = new Set(agents.map((agent) => agent.spec.id))
    const lowUtilityExecutors = artifact.implicatedExecutors.filter(
      (agentId) => artifact.executorUtilities[agentId] < this.lowUtilityThreshold,
    )

    if (artifact.skillOverlap >= this.overlapThreshold && artifact.implicatedExecutors.length >= 2) {
      const [targetAgentId, removeAgentId] = this.pickMergePair(artifact)
      const removed = await this.agentStore.get(removeAgentId)
      return {
        action: "merge_remove",
        artifact,
        reason: `skill ownership overlap ${artifact.skillOverlap.toFixed(2)} supports merging executor boundaries`,
        mergeRemove: {
          targetAgentId,
          removeAgentId,
          transferredSkills: removed?.spec.skills ?? [],
        },
      }
    }

    if (lowUtilityExecutors.length > 0 && artifact.pendingSkillIds.length > 0) {
      const agentId = lowUtilityExecutors[0]
      const existingSkills = (await this.agentStore.get(agentId))?.spec.skills ?? []
      const nextSkills = unique([...existingSkills, ...artifact.pendingSkillIds])
      return {
        action: "modify",
        artifact,
        reason: `executor utility ${artifact.executorUtilities[agentId].toFixed(2)} remains low after skill updates`,
        modify: [{
          agentId,
          specUpdates: {
            skills: nextSkills,
            skill_pool: nextSkills,
          },
          promptAppendix: boundaryAppendix(artifact),
        }],
      }
    }

    const newAgentId = uniqueAgentId(`specialist_${normalizeId(artifact.taskType)}`, existingIds)
    return {
      action: "add",
      artifact,
      reason: `retained failures for ${artifact.taskType} support adding one specialist`,
      add: {
        agentId: newAgentId,
        spec: {
          id: newAgentId,
          name: titleCase(`${artifact.taskType} specialist`),
          role: "worker",
          mode: "subagent",
          description: `Specialist for ${artifact.taskType} tasks`,
          role_description: `Handle ${artifact.taskType} tasks when retained evidence indicates a structural mismatch.`,
          tools: { allow: [], deny: [] },
          memory: { mode: "light", store: "md", capacity: 200 },
          skills: unique([...artifact.implicatedSkills, ...artifact.pendingSkillIds]),
          metadata: {
            createdBy: "SkillMAS",
            taskType: artifact.taskType,
            restructuringReason: artifact.reasons.join(", "),
          },
        },
        prompt: [
          `You are the ${artifact.taskType} specialist.`,
          "Use the assigned skills only when their applicability conditions match the delegated subtask.",
          "Report blockers explicitly so the manager can keep responsibility boundaries clear.",
        ].join("\n"),
      },
    }
  }

  async apply(decision: RestructuringDecision): Promise<RestructuringResult> {
    if (decision.action === "keep") {
      return { decision, changed: false, patchedAgentIds: [], createdAgentIds: [], removedAgentIds: [] }
    }

    if (decision.action === "add" && decision.add) {
      await this.agentStore.create(decision.add.spec, decision.add.prompt)
      this.agentQTable.initializeExtensionAgent(
        decision.add.agentId,
        this.agentQTable.getTaskBaseline(decision.artifact?.taskType ?? ""),
        decision.artifact ? [decision.artifact.taskType] : undefined,
      )
      return { decision, changed: true, patchedAgentIds: [], createdAgentIds: [decision.add.agentId], removedAgentIds: [] }
    }

    if (decision.action === "modify" && decision.modify) {
      const patched: string[] = []
      for (const patch of decision.modify) {
        const existing = await this.agentStore.get(patch.agentId)
        if (!existing) continue
        const prompt = patch.promptAppendix
          ? `${existing.prompt}\n\n${patch.promptAppendix}`
          : existing.prompt
        const ok = await this.agentStore.update(patch.agentId, patch.specUpdates, prompt)
        if (ok) patched.push(patch.agentId)
      }
      return { decision, changed: patched.length > 0, patchedAgentIds: patched, createdAgentIds: [], removedAgentIds: [] }
    }

    if (decision.action === "merge_remove" && decision.mergeRemove) {
      const target = await this.agentStore.get(decision.mergeRemove.targetAgentId)
      const removed = await this.agentStore.get(decision.mergeRemove.removeAgentId)
      if (!target || !removed) {
        return { decision, changed: false, patchedAgentIds: [], createdAgentIds: [], removedAgentIds: [] }
      }
      const nextSkills = unique([
        ...target.spec.skills,
        ...removed.spec.skills,
        ...(decision.artifact?.pendingSkillIds ?? []),
      ])
      await this.agentStore.update(target.spec.id, {
        skills: nextSkills,
        skill_pool: nextSkills,
      }, `${target.prompt}\n\n${boundaryAppendix(decision.artifact)}`)
      await this.agentStore.remove(removed.spec.id)
      return {
        decision,
        changed: true,
        patchedAgentIds: [target.spec.id],
        createdAgentIds: [],
        removedAgentIds: [removed.spec.id],
      }
    }

    return { decision, changed: false, patchedAgentIds: [], createdAgentIds: [], removedAgentIds: [] }
  }

  private async computeSkillOverlap(agentIds: string[]): Promise<number> {
    if (agentIds.length < 2) return 0
    const agents = await Promise.all(agentIds.map((agentId) => this.agentStore.get(agentId)))
    const pools = agents
      .filter(Boolean)
      .map((agent) => new Set(agent!.spec.skills))
      .filter((pool) => pool.size > 0)
    if (pools.length < 2) return 0

    let maxOverlap = 0
    for (let i = 0; i < pools.length; i++) {
      for (let j = i + 1; j < pools.length; j++) {
        const overlap = jaccard(pools[i], pools[j])
        if (overlap > maxOverlap) maxOverlap = overlap
      }
    }
    return maxOverlap
  }

  private pickMergePair(artifact: StructuralArtifact): [string, string] {
    const sorted = [...artifact.implicatedExecutors].sort(
      (a, b) => (artifact.executorUtilities[b] ?? 0) - (artifact.executorUtilities[a] ?? 0),
    )
    return [sorted[0], sorted[sorted.length - 1]]
  }
}

function boundaryAppendix(artifact?: StructuralArtifact): string {
  if (!artifact) return "SkillMAS boundary update: keep responsibility narrow and report handoff needs explicitly."
  return [
    "SkillMAS boundary update:",
    `- Primary task region: ${artifact.taskType}`,
    `- Retained evidence: ${artifact.reasons.join(", ") || "structural mismatch"}`,
    "- Keep responsibility narrow; hand off subtasks outside this region instead of absorbing them silently.",
  ].join("\n")
}

function flattenSkillMap(value?: Record<string, string[]>): string[] {
  return Object.values(value ?? {}).flat()
}

function unique<T>(values: T[]): T[] {
  return [...new Set(values.filter(Boolean))]
}

function jaccard<T>(a: Set<T>, b: Set<T>): number {
  const union = new Set([...a, ...b])
  if (union.size === 0) return 0
  let intersection = 0
  for (const value of a) {
    if (b.has(value)) intersection++
  }
  return intersection / union.size
}

function normalizeId(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "task"
}

function uniqueAgentId(base: string, existing: Set<string>): string {
  if (!existing.has(base)) return base
  for (let i = 2; ; i++) {
    const candidate = `${base}_${i}`
    if (!existing.has(candidate)) return candidate
  }
}

function titleCase(value: string): string {
  return value.replace(/\b[a-z]/g, (match) => match.toUpperCase())
}
