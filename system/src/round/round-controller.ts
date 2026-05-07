import type { SkillManager } from "../skill/manager.js"
import { SkillQTable } from "../skill/q-table.js"
import { AgentQTable } from "../skill/agent-q-table.js"
import type { SkillDesigner, DesignerActions, RoundStats, EpisodeTrajectory } from "../skill/designer.js"
import type { SkillForge, ForgeActions } from "../skill/skill-forge.js"
import {
  computeActualSkillsUsed,
  type SkillCatalogEntry,
  type SkillToolMappingConfig,
} from "../skill/utilization.js"
import { EvidenceRetainer, type RetainedEvidence, type VerifiedTrace } from "./retention.js"
import { ValidationPool, type ValidationPoolDecision } from "./validation-pool.js"
import { PolicyIndex, type PolicyCard } from "./policy-index.js"
import { MASRestructurer, type RestructuringResult } from "./restructuring.js"

export interface RoundState {
  round: number
  skillManager: SkillManager
  skillQTable: SkillQTable
  agentQTable: AgentQTable
  policyIndex: PolicyIndex
  validationPool: ValidationPool
}

export interface RoundExecutionInput {
  state: RoundState
}

export interface RoundExecutionResult {
  traces: VerifiedTrace[]
}

export interface SkillMASRoundControllerOptions {
  executeBatch: (input: RoundExecutionInput) => Promise<RoundExecutionResult>
  skillManager: SkillManager
  skillQTable?: SkillQTable
  agentQTable?: AgentQTable
  policyIndex?: PolicyIndex
  validationPool?: ValidationPool
  retainer?: EvidenceRetainer
  skillDesigner?: SkillDesigner
  skillForge?: SkillForge
  restructurer?: MASRestructurer
  skillToolMapping?: SkillToolMappingConfig
  expertCards?: PolicyCard[]
}

export interface SkillMASRoundResult {
  round: number
  traces: VerifiedTrace[]
  retainedEvidence: RetainedEvidence[]
  designerActions?: DesignerActions
  forgeActions?: ForgeActions
  validationDecisions: ValidationPoolDecision[]
  restructuring?: RestructuringResult
  policyCards: PolicyCard[]
}

export class SkillMASRoundController {
  private executeBatch: (input: RoundExecutionInput) => Promise<RoundExecutionResult>
  private skillManager: SkillManager
  private skillQTable: SkillQTable
  private agentQTable: AgentQTable
  private policyIndex: PolicyIndex
  private validationPool: ValidationPool
  private retainer: EvidenceRetainer
  private skillDesigner?: SkillDesigner
  private skillForge?: SkillForge
  private restructurer?: MASRestructurer
  private skillToolMapping: SkillToolMappingConfig

  constructor(options: SkillMASRoundControllerOptions) {
    this.executeBatch = options.executeBatch
    this.skillManager = options.skillManager
    this.skillQTable = options.skillQTable ?? new SkillQTable()
    this.agentQTable = options.agentQTable ?? new AgentQTable()
    this.policyIndex = options.policyIndex ?? new PolicyIndex(options.skillManager, { expertCards: options.expertCards })
    this.validationPool = options.validationPool ?? new ValidationPool(options.skillManager)
    this.retainer = options.retainer ?? new EvidenceRetainer()
    this.skillDesigner = options.skillDesigner
    this.skillForge = options.skillForge
    this.restructurer = options.restructurer
    this.skillToolMapping = options.skillToolMapping ?? {}
  }

  async runRound(round: number): Promise<SkillMASRoundResult> {
    const state: RoundState = {
      round,
      skillManager: this.skillManager,
      skillQTable: this.skillQTable,
      agentQTable: this.agentQTable,
      policyIndex: this.policyIndex,
      validationPool: this.validationPool,
    }

    const { traces } = await this.executeBatch({ state })
    const catalog = await this.buildSkillCatalog()
    const normalizedTraces = traces.map((trace) => this.ensureUsedSkills(trace, catalog))

    this.learnUtilities(normalizedTraces)
    const retainedEvidence = this.retainer.retain(normalizedTraces)
    const policyCards = await this.policyIndex.search(
      retainedEvidence.map((item) => `${item.trace.taskType} ${item.trace.summary ?? ""} ${item.trace.errorMessage ?? ""}`).join("\n"),
    )

    const roundStats = toRoundStats(round, normalizedTraces)
    const designerActions = this.skillDesigner
      ? await this.skillDesigner.designSkills(roundStats)
      : undefined
    const newlyCreated = [
      ...(designerActions?.learned.map((item) => item.skillId) ?? []),
      ...(designerActions?.created.map((item) => item.skillId) ?? []),
    ]
    await this.validationPool.addMany(newlyCreated)

    const forgeActions = this.skillForge
      ? await this.skillForge.forgeSkills(roundStats)
      : undefined
    const validationDecisions = await this.validationPool.evaluate(normalizedTraces)

    const validatedSkillIds = validationDecisions
      .filter((decision) => decision.action === "promote")
      .map((decision) => decision.skillId)
    const pendingSkillIds = unique([...newlyCreated, ...validatedSkillIds])

    let restructuring: RestructuringResult | undefined
    if (this.restructurer) {
      const artifacts = await this.restructurer.buildArtifacts(retainedEvidence, pendingSkillIds)
      const decision = await this.restructurer.decide(artifacts)
      restructuring = await this.restructurer.apply(decision)
    }

    return {
      round,
      traces: normalizedTraces,
      retainedEvidence,
      designerActions,
      forgeActions,
      validationDecisions,
      restructuring,
      policyCards,
    }
  }

  private learnUtilities(traces: VerifiedTrace[]): void {
    for (const trace of traces) {
      const reward = trace.success ? 1 : 0
      for (const [agentId, skillIds] of Object.entries(trace.usedSkills ?? {})) {
        for (const skillId of skillIds) {
          this.skillQTable.update(skillId, { taskType: trace.taskType, agentId, reward })
        }
      }
      for (const agentId of trace.executors) {
        this.agentQTable.update(agentId, trace.taskType, reward)
      }
    }
  }

  private ensureUsedSkills(
    trace: VerifiedTrace,
    catalog: Map<string, SkillCatalogEntry>,
  ): VerifiedTrace {
    if (trace.usedSkills) return trace
    if (!trace.selectedSkills || !trace.toolCalls) return { ...trace, usedSkills: {} }
    const usage = computeActualSkillsUsed(
      trace.primaryExecutor ?? trace.executors[0] ?? "root",
      trace.selectedSkills,
      trace.toolCalls,
      this.skillToolMapping,
      catalog,
    )
    return { ...trace, usedSkills: usage.usedSkills }
  }

  private async buildSkillCatalog(): Promise<Map<string, SkillCatalogEntry>> {
    const result = new Map<string, SkillCatalogEntry>()
    for (const spec of await this.skillManager.list()) {
      result.set(spec.id, {
        description: spec.description,
        whenToUse: spec.whenToUse,
        tags: spec.tags,
        allowedTools: spec.allowedTools,
        toolHints: spec.toolHints,
        family: spec.family,
        agent: spec.agent,
      })
    }
    return result
  }
}

function toRoundStats(round: number, traces: VerifiedTrace[]): RoundStats {
  const trajectories = traces.map(toEpisodeTrajectory)
  const successfulEpisodes = trajectories.filter((trajectory) => trajectory.success)
  const failedEpisodes = trajectories
    .filter((trajectory) => !trajectory.success)
    .map((trajectory) => ({
      taskType: trajectory.taskType,
      task: trajectory.task,
      errorMessage: trajectory.errorMessage ?? "",
      steps: trajectory.steps,
      skillsUsed: trajectory.skillsUsed,
    }))
  const successRate = traces.length > 0
    ? traces.filter((trace) => trace.success).length / traces.length
    : 0

  return {
    roundNum: round,
    totalEpisodes: traces.length,
    successRate,
    successfulEpisodes,
    failedEpisodes,
    trajectories,
  }
}

function toEpisodeTrajectory(trace: VerifiedTrace): EpisodeTrajectory {
  return {
    episodeId: trace.episodeId,
    taskType: trace.taskType,
    task: trace.task,
    success: trace.success,
    steps: trace.steps,
    score: trace.score,
    skillsUsed: trace.usedSkills ?? {},
    agentUsed: trace.primaryExecutor ?? trace.executors[0],
    delegatedAgents: trace.delegatedExecutors,
    errorMessage: trace.errorMessage,
    summary: trace.summary,
    actionTrace: trace.actionTrace,
    stepAttribution: trace.stepAttribution?.map((step) => ({
      actorId: step.agentId ?? step.delegatedAgentId ?? trace.primaryExecutor ?? trace.executors[0] ?? "unknown",
      tool: step.tool ?? "unknown",
      activeSkillIds: step.skillIds ?? [],
      delegatedAgentId: step.delegatedAgentId,
      stepOutcome: step.stepOutcome ?? "success",
    })),
    toolCalls: trace.toolCalls,
  }
}

function unique<T>(values: T[]): T[] {
  return [...new Set(values)]
}
