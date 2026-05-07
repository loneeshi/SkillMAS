/**
 * SkillDesigner — creates NEW skills via contrastive trajectory analysis.
 *
 * Part of the ARISE four-module architecture:
 *   - SkillDesign (this file): creates new skills from trajectory data
 *   - SkillForge  (skill-forge.ts): refines, prunes, and penalizes existing skills
 *
 * Two creation modes:
 *   1. LEARN: Extract skills from successful trajectories via contrastive analysis
 *   2. CREATE: Propose new skills for uncovered failure patterns
 *
 * LEARN is inspired by MemSkill's trajectory-based skill extraction but uses
 * contrastive LLM analysis (success vs failure) rather than a parametric controller.
 *
 * The design pipeline (per essay specification):
 *   Step 1: Partition trajectories into success/failure groups per task type
 *   Step 2: Action pattern abstraction — extract tool-call sequences and compute
 *           success-only vs failure-only patterns (set difference)
 *   Step 3: LLM synthesis — feed abstracted patterns to LLM for skill generation
 *   Step 4: Deduplication — check new skill against existing library for similarity
 */

import type { LLMClient } from "../llm/client.js"
import type { SkillManager } from "./manager.js"
import type { SkillQTable } from "./q-table.js"
import type { SkillSpecInput } from "../spec/skill.js"
import type { FailurePatternArtifact } from "../extension/types.js"
import type { ChatMessage, ToolSchema } from "../llm/types.js"
import type { ToolCallRecord } from "../runtime/types.js"
import {
  getDefaultSimilarityScorer,
  type TextSimilarityScorer,
} from "../llm/embedding.js"
import { createHash } from "node:crypto"
import { getSkillExemplar } from "./content-validator.js"

/** Actions taken by the designer in a single round (creation only). */
export interface DesignerActions {
  learned: Array<{ skillId: string; reason: string; sourceEpisodes: number }>
  created: Array<{ skillId: string; reason: string }>
  failurePatternArtifacts: FailurePatternArtifact[]
  /** Skills rejected by deduplication. */
  deduplicated: Array<{ candidateId: string; existingId: string; similarity: number }>
}

/** Condensed trajectory for a single episode (for designer analysis). */
export interface EpisodeTrajectory {
  episodeId?: string
  taskType: string
  task: string
  success: boolean
  steps: number
  score?: number
  skillsUsed: Record<string, string[]>
  agentUsed?: string
  delegatedAgents?: string[]
  errorMessage?: string
  /** Compact episode summary for contrastive analysis. */
  summary?: string
  /** Flattened action trace: "agent > tool(args) → result_snippet" */
  actionTrace?: string[]
  stepAttribution?: Array<{
    actorId: string
    tool: string
    activeSkillIds: string[]
    delegatedAgentId?: string
    stepOutcome: "success" | "failure"
  }>
  toolCalls?: ToolCallRecord[]
  messages?: ChatMessage[]
}

/** Statistics from a single round of episodes. */
export interface RoundStats {
  roundNum: number
  successRate: number
  totalEpisodes: number
  failedEpisodes: Array<{
    taskType: string
    task: string
    skillsUsed: Record<string, string[]>
    errorMessage?: string
    steps: number
    summary?: string
  }>
  successfulEpisodes: Array<{
    taskType: string
    task: string
    skillsUsed: Record<string, string[]>
    steps: number
    summary?: string
  }>
  /** Full episode trajectories (both success and failure) for LEARN phase. */
  trajectories?: EpisodeTrajectory[]
  /** Router-facing snapshots of the current agents/boundaries for failure analysis. */
  agentBoundaries?: Record<string, AgentBoundarySnapshot>
  /** Per-taskType success rates for this round. */
  taskTypeStats?: Map<string, { total: number; success: number; rate: number }>
  /** Previous round's success rate for drop detection. */
  prevSuccessRate?: number
  /** Total rounds in this experiment (for convergence decisions). */
  totalRounds?: number
}

interface DesignerOptions {
  /** Q threshold for "good enough" skill — above this, diversity rules apply. Default: 0.85 */
  goodSkillThreshold?: number
  /** Target success rate — stop creating when above this. Default: 0.90 */
  targetSuccessRate?: number
  /** A taskType is a "weak link" if its rate is below overall - this delta. Default: 0.05 */
  weakLinkDelta?: number
  /** Min failure rate (of total episodes) to trigger CREATE for a taskType. Default: 0.02 (2%) */
  minFailureRateToCreate?: number
  /** Min taskType sample ratio (of total) to consider for LEARN. Default: 0.02 (2%) */
  minSampleRatioToLearn?: number
  /** TaskType success rate threshold for "proven pattern". Default: 0.60 (60%) */
  provenPatternRate?: number

  // ── Domain-specific configuration ──────────────────────────────────

  /** Human-readable domain description for LLM prompts (e.g. "MyBenchmark (a simulated environment)"). */
  domainDescription?: string
  /** Tool syntax examples for LLM prompts (e.g. ["tool.action arg1", "tool.move target"]). */
  toolExamples?: string[]
  /** Prefix for auto-generated skill IDs. Defaults to "auto". */
  skillIdPrefix?: string
  /** Max episodes per contrastive analysis block. Default: 3 */
  maxEpisodesPerBlock?: number
  /** Max action trace lines per episode. Default: 20 */
  maxTraceLines?: number
  /** Optional expert skill library used as reference material when synthesizing new skills. */
  referenceSkillManager?: SkillManager

  // ── Deduplication ──────────────────────────────────────────────────

  /** Jaccard similarity threshold for dedup (0-1). Default: 0.45 */
  dedupThreshold?: number
  similarityScorer?: TextSimilarityScorer
}

/** Buffered failure case for analysis. */
interface FailureCase {
  episodeId?: string
  taskType: string
  task: string
  error?: string
  steps: number
  score?: number
  summary?: string
  actionTrace?: string[]
  skillsUsed: Record<string, string[]>
  skillsActive: string[]
  assignedAgent?: string
  delegatedAgents?: string[]
  stepAttribution?: Array<{
    actorId: string
    tool: string
    activeSkillIds: string[]
    delegatedAgentId?: string
    stepOutcome: "success" | "failure"
  }>
  toolCalls?: ToolCallRecord[]
  messages?: ChatMessage[]
}

/** Maximum new skills created per round. */
const MAX_CREATES_PER_ROUND = 3
/** Maximum skills learned from success trajectories per round. */
const MAX_LEARNS_PER_ROUND = 3
/** Maximum auto-generated skills allowed in the library at any time. */
const MAX_AUTO_SKILLS_IN_LIBRARY = 15

/** How many episodes to include in contrastive analysis prompts. */
const MAX_EPISODES_PER_BLOCK = 3
/** How many action trace lines to include per episode. */
const MAX_TRACE_LINES = 20
/** Word limit for contrastive analysis (LEARN) skill output. */
const LEARN_WORD_LIMIT = 400
/** Word limit for failure-generated (CREATE) skill output. */
const CREATE_WORD_LIMIT = 300
/** Maximum failure clusters that receive deep analysis in a round. */
const MAX_FAILURE_ANALYSIS_CLUSTERS = 3
/** Maximum failure analyst turns per cluster. */
const MAX_FAILURE_ANALYST_TURNS = 6

/** Default skill ID prefix for auto-generated skills. */
const DEFAULT_SKILL_PREFIX = "auto"

type FailureSelectionReason =
  | "coverage_gap"
  | "skill_overlap_conflict"
  | "near_miss"
  | "routing_boundary"

type FailurePriorityTier = "P0" | "P1" | "P2"
type FailureRecommendedAction = "create_skill" | "scale_agent" | "both" | "observe"

interface FailureCluster {
  key: string
  taskType: string
  assignedAgent?: string
  failures: FailureCase[]
  matchedSuccesses: EpisodeTrajectory[]
  implicatedAgents: string[]
  implicatedSkills: string[]
  selectionReason: FailureSelectionReason
  priority: number
  tier: FailurePriorityTier
}

interface FailureAnalysisDraft {
  continueAnalysis?: boolean
  verifiedCause?: boolean
  pattern?: string
  selectionReason?: FailureSelectionReason
  confidence?: number
  implicatedAgents?: string[]
  implicatedSkills?: string[]
  suggestedCapability?: string
  suggestedSkills?: string[]
  recommendedAction?: FailureRecommendedAction
  causalExplanation?: string
  nextEvidenceFocus?: string
}

interface FailureAnalysisResult {
  artifact: FailurePatternArtifact | null
  verifiedCause: boolean
  recommendedAction: FailureRecommendedAction
}

export interface AgentBoundarySnapshot {
  roleDescription?: string
  promptBoundary?: string
  skills?: string[]
  tools?: string[]
}

interface FailureAnalysisSubmission {
  pattern?: string
  selectionReason?: FailureSelectionReason
  confidence?: number
  implicatedAgents?: string[]
  implicatedSkills?: string[]
  suggestedCapability?: string
  suggestedSkills?: string[]
  recommendedAction?: FailureRecommendedAction
  causalExplanation?: string
  proposedFix?: string
  evidenceRefs?: Array<{
    kind?: "failure" | "success"
    index?: number
    observation?: string
  }>
}

interface FailureAnalystContext {
  cluster: FailureCluster
  selectedFailures: FailureCase[]
  selectedSuccesses: EpisodeTrajectory[]
  creationReason?: string
  agentBoundaries?: Record<string, AgentBoundarySnapshot>
}

export class SkillDesigner {
  private llm: LLMClient
  private qTable: SkillQTable
  private skillManager: SkillManager
  private failureBuffer: FailureCase[] = []
  private goodSkillThreshold: number
  private targetSuccessRate: number
  private weakLinkDelta: number
  private minFailureRateToCreate: number
  private minSampleRatioToLearn: number
  private provenPatternRate: number
  private domainDescription: string
  private toolExamples: string[]
  private skillIdPrefix: string
  private maxEpisodesPerBlock: number
  private maxTraceLines: number
  private referenceSkillManager?: SkillManager
  private dedupThreshold: number
  private similarityScorer: TextSimilarityScorer

  constructor(
    llm: LLMClient,
    qTable: SkillQTable,
    skillManager: SkillManager,
    options?: DesignerOptions,
  ) {
    this.llm = llm
    this.qTable = qTable
    this.skillManager = skillManager
    this.goodSkillThreshold = options?.goodSkillThreshold ?? 0.85
    this.targetSuccessRate = options?.targetSuccessRate ?? 0.90
    this.weakLinkDelta = options?.weakLinkDelta ?? 0.05
    this.minFailureRateToCreate = options?.minFailureRateToCreate ?? 0.02
    this.minSampleRatioToLearn = options?.minSampleRatioToLearn ?? 0.02
    this.provenPatternRate = options?.provenPatternRate ?? 0.60
    this.domainDescription = options?.domainDescription ?? "the target domain"
    this.toolExamples = options?.toolExamples ?? []
    this.skillIdPrefix = options?.skillIdPrefix ?? DEFAULT_SKILL_PREFIX
    this.maxEpisodesPerBlock = options?.maxEpisodesPerBlock ?? MAX_EPISODES_PER_BLOCK
    this.maxTraceLines = options?.maxTraceLines ?? MAX_TRACE_LINES
    this.referenceSkillManager = options?.referenceSkillManager
    this.dedupThreshold = options?.dedupThreshold ?? 0.45
    this.similarityScorer = options?.similarityScorer ?? getDefaultSimilarityScorer()
  }

  /**
   * Main entry: design new skills after a round.
   *
   * Executes in order: LEARN → CREATE.
   * LEARN extracts skills from successful trajectories (contrastive analysis).
   * CREATE proposes skills for uncovered failure patterns.
   *
   * Refine/Prune/Drop are handled by SkillForge (skill-forge.ts).
   */
  async designSkills(stats: RoundStats): Promise<DesignerActions> {
    this.dedupLog = []
    this.clearBuffer()

    const failureTrajectories = stats.trajectories?.filter((trajectory) => !trajectory.success) ?? []
    if (failureTrajectories.length > 0) {
      for (const trajectory of failureTrajectories) {
        this.addFailure({
          episodeId: trajectory.episodeId,
          taskType: trajectory.taskType,
          task: trajectory.task,
          error: trajectory.errorMessage,
          steps: trajectory.steps,
          summary: trajectory.summary,
          actionTrace: trajectory.actionTrace,
          skillsUsed: trajectory.skillsUsed,
          skillsActive: flattenSkills(trajectory.skillsUsed),
          assignedAgent: trajectory.agentUsed,
          delegatedAgents: trajectory.delegatedAgents,
          stepAttribution: trajectory.stepAttribution,
        })
      }
    } else {
      for (const ep of stats.failedEpisodes) {
        this.addFailure({
          taskType: ep.taskType,
          task: ep.task,
          error: ep.errorMessage,
          steps: ep.steps,
          skillsUsed: ep.skillsUsed,
          skillsActive: flattenSkills(ep.skillsUsed),
        })
      }
    }

    const learned = await this.learnFromSuccess(stats)
    const { created, failurePatternArtifacts } = await this.analyzeAndCreate(stats)

    return { learned, created, failurePatternArtifacts, deduplicated: this.dedupLog }
  }

  /** Dedup log for the current round — reset at start of each designSkills call. */
  private dedupLog: Array<{ candidateId: string; existingId: string; similarity: number }> = []

  /**
   * Check candidate skill content against all existing skills in the library.
   * Returns the best-matching existing skill if similarity exceeds the threshold,
   * or null if the candidate is sufficiently novel.
   */
  private async findNearDuplicate(
    candidateContent: string,
    candidateSpec: SkillSpecInput,
  ): Promise<{ existingId: string; similarity: number } | null> {
    const existingSkills = await this.skillManager.list()
    if (existingSkills.length === 0) return null

    // Tokenize the candidate: combine description + whenToUse + content
    const candidateText = [
      candidateSpec.description ?? "",
      candidateSpec.whenToUse ?? "",
      candidateContent,
    ].join(" ")
    if (!candidateText.trim()) return null

    let bestMatch: { existingId: string; similarity: number } | null = null
    const existingTexts: Record<string, string> = {}

    for (const existing of existingSkills) {
      // Build comparison text from spec fields + content if available
      const existingText = [
        existing.description ?? "",
        existing.whenToUse ?? "",
        ...(existing.tags ?? []),
      ].join(" ")

      // Also fetch content for richer comparison
      let existingContent = ""
      try {
        const full = await this.skillManager.get(existing.id)
        if (full?.content) existingContent = full.content
      } catch {
        // If content retrieval fails, compare spec fields only
      }

      const combinedText = `${existingText} ${existingContent}`.trim()
      if (!combinedText) continue
      existingTexts[existing.id] = combinedText
    }

    const similarities = await this.similarityScorer.scoreMany(candidateText, existingTexts)

    for (const [existingId, sim] of Object.entries(similarities)) {
      if (sim >= this.dedupThreshold && (!bestMatch || sim > bestMatch.similarity)) {
        bestMatch = { existingId, similarity: sim }
      }
    }

    return bestMatch
  }

  /** Add a failure case to the buffer for later analysis. */
  addFailure(failure: FailureCase): void {
    this.failureBuffer.push(failure)
  }

  /** Clear the failure buffer. */
  clearBuffer(): void {
    this.failureBuffer = []
  }

  /**
   * LEARN phase: extract skills from successful trajectories via contrastive analysis.
   *
   * For each task type that has BOTH successes and failures this round,
   * use LLM to compare "what worked vs what didn't" and distill the
   * winning strategy into a reusable skill.
   *
   * This is the key insight from MemRL/MemSkill: learn from episodic memory
   * of successful executions, not just from failure patterns.
   *
   * Learning conditions (percentage-based for cross-benchmark compatibility):
   *   - TaskType sample ratio >= minSampleRatioToLearn (has enough data)
   *   - Has both successes and failures (contrast available)
   *   - Either:
   *     (a) No learned skill with Q > 0.8 for this taskType
   *     (b) TaskType success rate >= provenPatternRate AND < targetSuccessRate
   */
  private async learnFromSuccess(
    stats: RoundStats,
  ): Promise<Array<{ skillId: string; reason: string; sourceEpisodes: number }>> {
    const trajectories = stats.trajectories
    if (!trajectories || trajectories.length === 0) return []

    const totalEpisodes = stats.totalEpisodes || trajectories.length

    const byType = new Map<string, { successes: EpisodeTrajectory[]; failures: EpisodeTrajectory[] }>()
    for (const t of trajectories) {
      const entry = byType.get(t.taskType) ?? { successes: [], failures: [] }
      if (t.success) entry.successes.push(t)
      else entry.failures.push(t)
      byType.set(t.taskType, entry)
    }

    // Build taskType stats
    const taskTypeStats = stats.taskTypeStats ?? this.computeTaskTypeStats(stats)

    const existingSkills = await this.skillManager.list()
    const existingIds = new Set(existingSkills.map((s) => s.id))

    const learned: Array<{ skillId: string; reason: string; sourceEpisodes: number }> = []

    // Check global auto-skill cap
    const currentAutoSkills = existingSkills.filter(s => s.generatedBy !== "manual").length
    if (currentAutoSkills >= MAX_AUTO_SKILLS_IN_LIBRARY) {
      return learned  // Library is full — don't create more, let pruning free slots
    }

    const candidates = [...byType.entries()]
      .filter(([_, v]) => {
        const total = v.successes.length + v.failures.length
        const sampleRatio = total / totalEpisodes
        // Need: enough samples, has contrast (both success and failure)
        return sampleRatio >= this.minSampleRatioToLearn &&
               v.successes.length > 0 &&
               v.failures.length > 0
      })
      .map(([taskType, data]) => {
        const score = this.computeLearnPriority(
          taskType,
          data.successes.length,
          data.failures.length,
          existingSkills,
          taskTypeStats,
          totalEpisodes,
        )
        return { taskType, data, score }
      })
      .filter((x) => x.score.shouldLearn)
      .sort((a, b) => b.score.priority - a.score.priority)
      .slice(0, MAX_LEARNS_PER_ROUND)

    for (const { taskType, data: { successes, failures }, score } of candidates) {
      const skillId = `${this.skillIdPrefix}/${taskType}_learned_r${stats.roundNum}`
      if (existingIds.has(skillId)) continue

      try {
        const content = await this.contrastiveAnalysis(taskType, successes, failures)
        const spec: SkillSpecInput = {
          id: skillId,
          description: `Strategy for ${taskType} tasks, learned from ${successes.length} successful episodes (round ${stats.roundNum})`,
          whenToUse: `When handling ${taskType} tasks — apply these proven strategies`,
          steps: [],
          tags: [taskType, "trajectory-learned", `round-${stats.roundNum}`],
          generatedBy: "trajectory-learning",
        }

        // Step 4: Dedup — check against existing library
        const dup = await this.findNearDuplicate(content, spec)
        if (dup) {
          this.dedupLog.push({ candidateId: skillId, existingId: dup.existingId, similarity: dup.similarity })
          continue
        }

        await this.skillManager.create(spec, content)

        this.qTable.initialize(skillId)
        this.qTable.recordContentVersion(skillId, hashContent(content))

        learned.push({
          skillId,
          reason: score.reason,
          sourceEpisodes: successes.length,
        })
      } catch {
        // LLM call failed — skip
      }
    }

    return learned
  }

  /**
   * Compute whether to learn a new skill from success trajectories.
   * Uses percentage-based thresholds for cross-benchmark compatibility.
   */
  private computeLearnPriority(
    taskType: string,
    successCount: number,
    failureCount: number,
    existingSkills: Array<{ id: string; tags: string[]; whenToUse: string; generatedBy?: string }>,
    taskTypeStats: Map<string, { total: number; success: number; rate: number }>,
    totalEpisodes: number,
  ): { shouldLearn: boolean; priority: number; reason: string } {
    // Check existing learned skills for this taskType
    const learnedSkills = existingSkills.filter(
      (s) =>
        s.generatedBy === "trajectory-learning" &&
        (s.tags.includes(taskType) || s.whenToUse.includes(taskType)),
    )
    const bestLearnedQ = learnedSkills
      .map((s) => this.qTable.getStats(s.id)?.globalQ ?? 0)
      .reduce((a, b) => Math.max(a, b), 0)

    // TaskType success rate
    const typeStats = taskTypeStats.get(taskType)
    const typeSuccessRate = typeStats?.rate ?? 0
    const typeTotal = successCount + failureCount
    const typeSampleRatio = typeTotal / totalEpisodes

    // Decision conditions (percentage-based)
    const hasExcellentLearnedSkill = bestLearnedQ >= 0.8
    const belowTargetRate = typeSuccessRate < this.targetSuccessRate
    const isProvenPattern = typeSuccessRate >= this.provenPatternRate  // e.g., >= 60% success

    let shouldLearn = false
    let reason = ""
    let priority = 0

    if (!hasExcellentLearnedSkill) {
      // No good learned skill yet — learn from successes
      shouldLearn = true
      reason = `${(typeSuccessRate * 100).toFixed(0)}% success rate, no learned skill with Q>0.8`
      priority = 100 + typeSampleRatio * 50 + typeSuccessRate * 30
    } else if (belowTargetRate && isProvenPattern) {
      // Has learned skill but rate still below target and pattern is proven (>= 60% success)
      shouldLearn = true
      reason = `proven pattern (${(typeSuccessRate * 100).toFixed(0)}% success), below target ${(this.targetSuccessRate * 100).toFixed(0)}%`
      priority = 70 + typeSuccessRate * 30
    }

    return { shouldLearn, priority, reason }
  }

  private async formatReferenceSkillBlock(taskType: string): Promise<string> {
    if (!this.referenceSkillManager) return ""

    try {
      const normalizedTaskType = taskType.toLowerCase()
      const ranked = (await this.referenceSkillManager.list())
        .filter((spec) => spec.status !== "disabled")
        .map((spec) => {
          const tags = new Set((spec.tags ?? []).map((tag) => tag.toLowerCase()))
          const searchable = [
            spec.id,
            spec.description ?? "",
            spec.whenToUse ?? "",
            ...(spec.tags ?? []),
          ].join(" ").toLowerCase()
          let score = 0
          if (tags.has(normalizedTaskType)) score += 4
          if (searchable.includes(normalizedTaskType)) score += 3
          if (tags.has("alfworld")) score += 2
          if (tags.has("manager") && normalizedTaskType.includes("examine")) score += 1
          return { spec, score }
        })
        .filter(({ score }) => score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, 4)

      const blocks: string[] = []
      for (const { spec } of ranked) {
        const resolved = await this.referenceSkillManager.get(spec.id)
        if (!resolved) continue
        blocks.push(`### ${spec.id}\nDescription: ${spec.description ?? ""}\nWhen to use: ${spec.whenToUse ?? ""}\n\n${truncate(resolved.content, 1400)}`)
      }

      if (blocks.length === 0) return ""
      return `## Expert Reference Skills\nUse these as reference material. Do not blindly copy; adapt them to the current traces and failure evidence.\n\n${blocks.join("\n\n")}`
    } catch {
      return ""
    }
  }

  /**
   * Use LLM to do contrastive analysis: what do successful trajectories do
   * differently from failed ones for the same task type?
   *
   * Pipeline (matching essay spec):
   *   Step 2: Action pattern abstraction — extract tool sequences, compute set differences
   *   Step 3: LLM synthesis — feed abstracted patterns to LLM for skill generation
   */
  private async contrastiveAnalysis(
    taskType: string,
    successes: EpisodeTrajectory[],
    failures: EpisodeTrajectory[],
  ): Promise<string> {
    // ── Step 2: Action pattern abstraction ──
    const successPatterns = this.extractActionPatterns(successes)
    const failurePatterns = this.extractActionPatterns(failures)

    const successOnly = setDifference(successPatterns.toolSequences, failurePatterns.toolSequences)
    const failureOnly = setDifference(failurePatterns.toolSequences, successPatterns.toolSequences)

    const patternSummary = this.formatPatternSummary(
      successPatterns, failurePatterns, successOnly, failureOnly,
    )

    // ── Step 3: LLM synthesis with abstracted patterns ──
    const successExamples = this.selectSuccessExamples(successes)
    const failureExamples = this.selectFailureExamples(
      failures.map((failure) => this.toFailureCase(failure)),
      average(successes.map((success) => success.steps)),
    ).map((failure) => this.toEpisodeTrajectory(failure))

    const successBlock = successExamples
      .slice(0, this.maxEpisodesPerBlock)
      .map((s, i) => {
        const trace = s.actionTrace?.slice(0, this.maxTraceLines).join("\n  ") ?? "(no trace)"
        return `### Success ${i + 1}: "${s.task}" (${s.steps} steps)\n  ${trace}`
      })
      .join("\n\n")

    const failureBlock = failureExamples
      .slice(0, this.maxEpisodesPerBlock)
      .map((f, i) => {
        const trace = f.actionTrace?.slice(0, this.maxTraceLines).join("\n  ") ?? "(no trace)"
        const err = f.errorMessage ? `\n  Error: ${f.errorMessage}` : ""
        return `### Failure ${i + 1}: "${f.task}" (${f.steps} steps)${err}\n  ${trace}`
      })
      .join("\n\n")

    const toolLine = this.toolExamples.length > 0
      ? `- Reference specific tool calls (${this.toolExamples.join(", ")}, etc.)\n`
      : ""

    const exemplar = getSkillExemplar("task")
    const referenceBlock = await this.formatReferenceSkillBlock(taskType)

    const prompt = `You are analyzing agent execution traces for "${taskType}" tasks in ${this.domainDescription}.

## Pre-computed Action Pattern Analysis
${patternSummary}

## Successful Executions
${successBlock}

## Failed Executions
${failureBlock}

${referenceBlock}

## Your Task
Using BOTH the pre-computed pattern analysis AND the raw traces, synthesize a reusable skill.
The pattern analysis tells you WHAT differs; the traces tell you HOW and WHY.
If expert reference skills are provided, use them as domain knowledge and adapt them to the observed traces.

Focus on:
1. The success-only patterns — these are the winning moves
2. The failure-only patterns — these are the mistakes to avoid
3. The optimal tool sequence for this task type

## Example of Good Skill Structure

The following is an example of a well-structured skill (adapt content to your specific domain, not copied verbatim):

${exemplar}

Write a concise skill guide that captures the winning strategy. Requirements:
- Use ## headers for sections
- Include a clear step-by-step procedure
- Include "Critical Do's" and "Critical Don'ts" sections
${toolLine}- Keep it under ${LEARN_WORD_LIMIT} words
- Be very specific and actionable — no generic advice

Output ONLY the markdown content, no frontmatter.`

    const response = await this.llm.chat(
      [
        { role: "system", content: "You are an expert at analyzing agent behavior traces and extracting reusable strategies. Output ONLY markdown content." },
        { role: "user", content: prompt },
      ],
      { temperature: 0.3 },
    )

    return response.content.trim()
  }

  // ─── Action Pattern Abstraction (Step 2) ────────────────────────────

  /**
   * Extract structured action patterns from a set of episode trajectories.
   *
   * Parses action traces into tool-call sequences and computes:
   *   - Tool frequency (which tools are used and how often)
   *   - Bigram sequences (common tool-call pairs)
   *   - Unique tool sequences (for set difference computation)
   */
  private extractActionPatterns(episodes: EpisodeTrajectory[]): ActionPatterns {
    const toolFreq = new Map<string, number>()
    const bigramFreq = new Map<string, number>()
    const toolSequences = new Set<string>()
    let totalSteps = 0

    for (const ep of episodes) {
      if (!ep.actionTrace || ep.actionTrace.length === 0) continue

      const tools: string[] = []
      for (const line of ep.actionTrace) {
        const toolName = extractToolName(line)
        if (toolName) {
          tools.push(toolName)
          toolFreq.set(toolName, (toolFreq.get(toolName) ?? 0) + 1)
        }
      }

      // Bigrams: consecutive tool pairs (captures common sequences)
      for (let i = 0; i < tools.length - 1; i++) {
        const bigram = `${tools[i]} → ${tools[i + 1]}`
        bigramFreq.set(bigram, (bigramFreq.get(bigram) ?? 0) + 1)
        toolSequences.add(bigram)
      }

      // Trigrams for richer patterns
      for (let i = 0; i < tools.length - 2; i++) {
        const trigram = `${tools[i]} → ${tools[i + 1]} → ${tools[i + 2]}`
        toolSequences.add(trigram)
      }

      totalSteps += ep.steps
    }

    return {
      toolFreq,
      bigramFreq,
      toolSequences,
      avgSteps: episodes.length > 0 ? totalSteps / episodes.length : 0,
      episodeCount: episodes.length,
    }
  }

  /**
   * Format the pattern analysis for injection into the LLM prompt.
   */
  private formatPatternSummary(
    successPatterns: ActionPatterns,
    failurePatterns: ActionPatterns,
    successOnly: Set<string>,
    failureOnly: Set<string>,
  ): string {
    const lines: string[] = []

    lines.push(`**Success episodes**: ${successPatterns.episodeCount} (avg ${successPatterns.avgSteps.toFixed(1)} steps)`)
    lines.push(`**Failure episodes**: ${failurePatterns.episodeCount} (avg ${failurePatterns.avgSteps.toFixed(1)} steps)`)
    lines.push("")

    // Top tools in successes vs failures
    const successTools = sortedEntries(successPatterns.toolFreq, 8)
    const failureTools = sortedEntries(failurePatterns.toolFreq, 8)
    if (successTools.length > 0) {
      lines.push(`### Tool Usage (Success)`)
      for (const [tool, count] of successTools) lines.push(`- \`${tool}\`: ${count}x`)
    }
    if (failureTools.length > 0) {
      lines.push(`### Tool Usage (Failure)`)
      for (const [tool, count] of failureTools) lines.push(`- \`${tool}\`: ${count}x`)
    }

    // Success-only patterns (the winning moves)
    if (successOnly.size > 0) {
      lines.push("")
      lines.push(`### Patterns ONLY in Successes (${successOnly.size})`)
      for (const p of [...successOnly].slice(0, 10)) lines.push(`- ${p}`)
    }

    // Failure-only patterns (the mistakes)
    if (failureOnly.size > 0) {
      lines.push("")
      lines.push(`### Patterns ONLY in Failures (${failureOnly.size})`)
      for (const p of [...failureOnly].slice(0, 10)) lines.push(`- ${p}`)
    }

    // Top bigrams in successes
    const topBigrams = sortedEntries(successPatterns.bigramFreq, 5)
    if (topBigrams.length > 0) {
      lines.push("")
      lines.push(`### Most Common Success Sequences`)
      for (const [seq, count] of topBigrams) lines.push(`- ${seq} (${count}x)`)
    }

    return lines.join("\n")
  }

  /**
   * Analyze failures and propose new skills.
   *
   * Decision logic based on three principles (percentage-based):
   *   1. Q-value convergence: only saturated if best Q > goodSkillThreshold AND low variance
   *   2. Failure coverage: create if taskType failure rate >= minFailureRateToCreate
   *   3. Balance (anti-shift): prioritize "weak link" taskTypes below overall average
   *
   * A taskType qualifies for new skill creation if:
   *   - Failure rate >= minFailureRateToCreate (e.g., 2% of total episodes), AND
   *   - One of:
   *     (a) No skill with Q > goodSkillThreshold (0.85) for this taskType
   *     (b) TaskType success rate < targetSuccessRate (0.90) AND is a weak link
   *     (c) Q-values for this taskType have high variance (not converged)
   */
  private async analyzeAndCreate(
    stats: RoundStats,
  ): Promise<{
    created: Array<{ skillId: string; reason: string }>
    failurePatternArtifacts: FailurePatternArtifact[]
  }> {
    const failuresByType = new Map<string, FailureCase[]>()
    for (const failure of this.failureBuffer) {
      const arr = failuresByType.get(failure.taskType) ?? []
      arr.push(failure)
      failuresByType.set(failure.taskType, arr)
    }

    const totalEpisodes = stats.totalEpisodes || (stats.successfulEpisodes.length + stats.failedEpisodes.length)

    // Build taskType stats if not provided
    const taskTypeStats = stats.taskTypeStats ?? this.computeTaskTypeStats(stats)

    const existingSkills = await this.skillManager.list()
    const existingIds = new Set(existingSkills.map((s) => s.id))

    const created: Array<{ skillId: string; reason: string }> = []
    const failurePatternArtifacts: FailurePatternArtifact[] = []
    const createdTaskTypes = new Set<string>()

    // Check global auto-skill cap
    const currentAutoSkills = existingSkills.filter(s => s.generatedBy !== "manual").length
    const allowSkillCreation = currentAutoSkills < MAX_AUTO_SKILLS_IN_LIBRARY

    // Score each taskType by creation priority (percentage-based filter)
    const scoredTypes = [...failuresByType.entries()]
      .filter(([_, failures]) => {
        const failureRate = failures.length / totalEpisodes
        return failureRate >= this.minFailureRateToCreate
      })
      .map(([taskType, failures]) => {
        const score = this.computeCreationPriority(
          taskType,
          failures.length,
          totalEpisodes,
          existingSkills,
          taskTypeStats,
          stats.successRate,
        )
        return { taskType, failures, score }
      })
      .filter((x) => x.score.shouldCreate)
      .sort((a, b) => b.score.priority - a.score.priority)
      .slice(0, MAX_CREATES_PER_ROUND)

    const creationScores = new Map(scoredTypes.map(({ taskType, score }) => [taskType, score]))
    const clusters = this.buildFailureClusters(stats, taskTypeStats)
      .filter((cluster) => cluster.tier !== "P2")
      .slice(0, MAX_FAILURE_ANALYSIS_CLUSTERS)

    for (const cluster of clusters) {
      try {
        const analysis = await this.analyzeFailureCluster(
          cluster,
          creationScores.get(cluster.taskType)?.reason,
          stats.agentBoundaries,
        )
        if (analysis.artifact) failurePatternArtifacts.push(analysis.artifact)
      } catch {
        // Skip artifact creation when analysis fails
      }
    }

    for (const { taskType, failures, score } of scoredTypes) {
      if (!allowSkillCreation) break
      const skillId = `${this.skillIdPrefix}/${taskType}_rl_r${stats.roundNum}`
      if (existingIds.has(skillId) || createdTaskTypes.has(taskType)) continue

      const cluster = clusters.find((candidate) => candidate.taskType === taskType)
      if (!cluster) continue

      const matchingArtifact = failurePatternArtifacts.find((artifact) =>
        artifact.taskType === taskType && artifact.verifiedCause === true,
      )
      const recommendedAction = String(matchingArtifact?.metadata?.recommendedAction ?? "create_skill") as FailureRecommendedAction
      if (!["create_skill", "both"].includes(recommendedAction)) continue

      try {
        const content = await this.generateSkillFromFailures(taskType, cluster, matchingArtifact)
        const spec: SkillSpecInput = {
          id: skillId,
          description: `RL-generated skill for ${taskType} tasks (round ${stats.roundNum})`,
          whenToUse: `When handling ${taskType} tasks`,
          steps: [],
          tags: [taskType, "rl-generated", `round-${stats.roundNum}`],
          generatedBy: "evolution",
        }

        // Step 4: Dedup — check against existing library
        const dup = await this.findNearDuplicate(content, spec)
        if (dup) {
          this.dedupLog.push({ candidateId: skillId, existingId: dup.existingId, similarity: dup.similarity })
          continue
        }

        await this.skillManager.create(spec, content)

        this.qTable.initialize(skillId)
        this.qTable.recordContentVersion(skillId, hashContent(content))

        created.push({
          skillId,
          reason: matchingArtifact?.pattern
            ? `${score.reason}; verified failure pattern: ${matchingArtifact.pattern}`
            : score.reason,
        })
        createdTaskTypes.add(taskType)
      } catch {
        // LLM call failed — skip creation silently
      }
    }

    return { created, failurePatternArtifacts }
  }

  /**
   * Compute whether to create a new skill for a taskType and with what priority.
   * Uses percentage-based thresholds for cross-benchmark compatibility.
   */
  private computeCreationPriority(
    taskType: string,
    failureCount: number,
    totalEpisodes: number,
    existingSkills: Array<{ id: string; tags: string[]; whenToUse: string }>,
    taskTypeStats: Map<string, { total: number; success: number; rate: number }>,
    overallSuccessRate: number,
  ): { shouldCreate: boolean; priority: number; reason: string } {
    // Get Q-values for skills covering this taskType
    const relevantSkills = existingSkills.filter(
      (s) => s.tags.includes(taskType) || s.whenToUse.includes(taskType),
    )
    const qValues = relevantSkills
      .map((s) => this.qTable.getStats(s.id))
      .filter((q) => q && q.globalN >= 3)
      .map((q) => q!.globalQ)

    const bestQ = qValues.length > 0 ? Math.max(...qValues) : 0
    const qVariance = qValues.length > 1 ? this.variance(qValues) : 1.0

    // TaskType success rate and failure rate (percentage-based)
    const typeStats = taskTypeStats.get(taskType)
    const typeSuccessRate = typeStats?.rate ?? 0
    const failureRate = failureCount / totalEpisodes

    // Check conditions
    const hasExcellentSkill = bestQ >= this.goodSkillThreshold
    const belowTargetRate = typeSuccessRate < this.targetSuccessRate
    const isWeakLink = typeSuccessRate < overallSuccessRate - this.weakLinkDelta
    const hasHighVariance = qVariance > 0.02 // Q-values not converged
    const hasEnoughData = qValues.length >= 2

    // Decision logic
    let shouldCreate = false
    let reason = ""
    let priority = 0

    if (!hasExcellentSkill) {
      // Case (a): No skill reached 0.85 yet — definitely create
      shouldCreate = true
      reason = `${(failureRate * 100).toFixed(1)}% failures, best Q=${bestQ.toFixed(2)} < ${this.goodSkillThreshold}`
      priority = 100 + failureRate * 100 + (1 - bestQ) * 50
    } else if (belowTargetRate && isWeakLink) {
      // Case (b): Has good skill but taskType is still a weak link
      shouldCreate = true
      reason = `weak link: ${taskType} rate=${(typeSuccessRate * 100).toFixed(0)}% < overall=${(overallSuccessRate * 100).toFixed(0)}%-${(this.weakLinkDelta * 100).toFixed(0)}%`
      priority = 80 + (overallSuccessRate - typeSuccessRate) * 100
    } else if (hasHighVariance && belowTargetRate && hasEnoughData) {
      // Case (c): Q-values not converged and still below target
      shouldCreate = true
      reason = `Q not converged (var=${qVariance.toFixed(3)}), ${taskType} rate=${(typeSuccessRate * 100).toFixed(0)}%`
      priority = 60 + qVariance * 100
    }

    return { shouldCreate, priority, reason }
  }

  /** Compute variance of an array of numbers. */
  private variance(values: number[]): number {
    if (values.length < 2) return 0
    const mean = values.reduce((a, b) => a + b, 0) / values.length
    const squaredDiffs = values.map((v) => (v - mean) ** 2)
    return squaredDiffs.reduce((a, b) => a + b, 0) / values.length
  }

  /** Compute per-taskType success rates from round stats. */
  private computeTaskTypeStats(
    stats: RoundStats,
  ): Map<string, { total: number; success: number; rate: number }> {
    const result = new Map<string, { total: number; success: number; rate: number }>()

    for (const ep of stats.successfulEpisodes) {
      const entry = result.get(ep.taskType) ?? { total: 0, success: 0, rate: 0 }
      entry.total++
      entry.success++
      result.set(ep.taskType, entry)
    }

    for (const ep of stats.failedEpisodes) {
      const entry = result.get(ep.taskType) ?? { total: 0, success: 0, rate: 0 }
      entry.total++
      result.set(ep.taskType, entry)
    }

    // Compute rates
    for (const [taskType, entry] of result) {
      entry.rate = entry.total > 0 ? entry.success / entry.total : 0
      result.set(taskType, entry)
    }

    return result
  }

  /* ----- LLM interaction ----- */

  private async generateSkillFromFailures(
    taskType: string,
    cluster: FailureCluster,
    artifact?: FailurePatternArtifact,
  ): Promise<string> {
    const failureSummary = this.selectFailureExamples(
      cluster.failures,
      average(cluster.matchedSuccesses.map((success) => success.steps)),
    )
      .map((failure, i) => this.formatFailureCase(failure, i + 1))
      .join("\n\n")
    const successSummary = this.selectSuccessExamples(cluster.matchedSuccesses)
      .map((success, i) => this.formatSuccessTrajectory(success, i + 1))
      .join("\n\n")
    const analysisBlock = artifact
      ? [
          `Verified failure pattern: ${artifact.pattern}`,
          `Selection reason: ${artifact.selectionReason ?? "coverage_gap"}`,
          `Confidence: ${(artifact.confidence ?? 0).toFixed(2)}`,
          `Causal explanation: ${String(artifact.metadata?.causalExplanation ?? "n/a")}`,
        ].join("\n")
      : "No verified failure artifact was produced; use the failures below conservatively."

    const exemplar = getSkillExemplar("task")
    const referenceBlock = await this.formatReferenceSkillBlock(taskType)

    const prompt = `You are designing a reusable skill guide for an agent.

  The agent repeatedly fails at "${taskType}" tasks. Here are the selected crucial failures:

${failureSummary}

## Verified Failure Analysis

${analysisBlock}

## Matched Success Exemplars

${successSummary || "(none available)"}

${referenceBlock}

## Example of Good Skill Structure

The following is an example of a well-structured skill (adapt content to your specific domain, not copied verbatim):

${exemplar}

Write a concise markdown skill guide that teaches the agent how to succeed at these tasks.
If expert reference skills are provided, use them as domain knowledge and specialize them to the observed failure mode.

Requirements:
- Use ## headers for sections
- Include a step-by-step workflow
- Include common mistakes to avoid
- Keep it under ${CREATE_WORD_LIMIT} words
- Be specific and actionable

Output ONLY the markdown content, no frontmatter.`

    const response = await this.llm.chat(
      [
        { role: "system", content: "You are a skill documentation writer. Output ONLY markdown content." },
        { role: "user", content: prompt },
      ],
      { temperature: 0.4 },
    )

    return response.content.trim()
  }

  private buildFailureClusters(
    stats: RoundStats,
    taskTypeStats: Map<string, { total: number; success: number; rate: number }>,
  ): FailureCluster[] {
    const successByType = new Map<string, EpisodeTrajectory[]>()
    for (const trajectory of stats.trajectories ?? []) {
      if (!trajectory.success) continue
      const entry = successByType.get(trajectory.taskType) ?? []
      entry.push(trajectory)
      successByType.set(trajectory.taskType, entry)
    }

    const clusters = new Map<string, FailureCluster>()

    for (const failure of this.failureBuffer) {
      const implicatedAgents = inferFailureAgents(failure)
      const skillSignature = normalizeSkillList(failure.skillsActive)
      const agentSignature = implicatedAgents.join("|") || failure.assignedAgent || "unassigned"
      const key = `${failure.taskType}::${agentSignature}::${skillSignature.join("|") || "no_skills"}`
      const existing = clusters.get(key)
      if (existing) {
        existing.failures.push(failure)
        existing.implicatedAgents = uniqueStrings([...existing.implicatedAgents, ...implicatedAgents])
        existing.implicatedSkills = uniqueStrings([...existing.implicatedSkills, ...failure.skillsActive])
        continue
      }

      clusters.set(key, {
        key,
        taskType: failure.taskType,
        assignedAgent: failure.assignedAgent,
        failures: [failure],
        matchedSuccesses: successByType.get(failure.taskType) ?? [],
        implicatedAgents,
        implicatedSkills: normalizeSkillList(failure.skillsActive),
        selectionReason: "coverage_gap",
        priority: 0,
        tier: "P2",
      })
    }

    return [...clusters.values()]
      .map((cluster) => {
        const enriched = this.scoreFailureCluster(cluster, taskTypeStats)
        return {
          ...cluster,
          selectionReason: enriched.selectionReason,
          priority: enriched.priority,
          tier: enriched.tier,
        }
      })
      .sort((a, b) => b.priority - a.priority)
  }

  private scoreFailureCluster(
    cluster: FailureCluster,
    taskTypeStats: Map<string, { total: number; success: number; rate: number }>,
  ): { priority: number; selectionReason: FailureSelectionReason; tier: FailurePriorityTier } {
    const successRate = taskTypeStats.get(cluster.taskType)?.rate ?? 0
    const coverageGap = clamp(1 - successRate, 0, 1)
    const contrastValue = cluster.matchedSuccesses.length > 0 ? 1 : 0.35
    const overlapSignal = clamp(
      Math.max(0, cluster.implicatedSkills.length - 1) * 0.35 +
      (cluster.implicatedAgents.length > 1 ? 0.5 : 0),
      0,
      1,
    )
    const traceRichness = clamp(average(
      cluster.failures.map((failure) => {
        const traceScore = clamp((failure.actionTrace?.length ?? 0) / Math.max(1, this.maxTraceLines), 0, 1)
        const errorScore = failure.error ? 1 : 0.25
        const summaryScore = failure.summary ? 0.5 : 0
        return average([traceScore, errorScore, summaryScore])
      }),
    ), 0, 1)
    const scalingLeverage = cluster.implicatedAgents.length > 1
      ? 1
      : cluster.assignedAgent && cluster.implicatedAgents.includes(cluster.assignedAgent)
        ? 0.65
        : 0.35

    const priority = (
      0.30 * coverageGap +
      0.25 * contrastValue +
      0.20 * overlapSignal +
      0.15 * traceRichness +
      0.10 * scalingLeverage
    )

    let selectionReason: FailureSelectionReason = "coverage_gap"
    if (cluster.implicatedAgents.length > 1) {
      selectionReason = "routing_boundary"
    } else if (cluster.implicatedSkills.length > 1) {
      selectionReason = "skill_overlap_conflict"
    } else if (cluster.matchedSuccesses.length > 0) {
      const avgFailureSteps = average(cluster.failures.map((failure) => failure.steps))
      const avgSuccessSteps = average(cluster.matchedSuccesses.map((success) => success.steps))
      if (avgFailureSteps <= avgSuccessSteps + 2) selectionReason = "near_miss"
    }

    const tier: FailurePriorityTier = priority >= 0.70 ? "P0" : priority >= 0.45 ? "P1" : "P2"
    return { priority, selectionReason, tier }
  }

  private async analyzeFailureCluster(
    cluster: FailureCluster,
    creationReason?: string,
    agentBoundaries?: Record<string, AgentBoundarySnapshot>,
  ): Promise<FailureAnalysisResult> {
    const selectedFailures = this.selectFailureExamples(
      cluster.failures,
      average(cluster.matchedSuccesses.map((success) => success.steps)),
    )
    const selectedSuccesses = this.selectSuccessExamples(cluster.matchedSuccesses)
    const context: FailureAnalystContext = {
      cluster,
      selectedFailures,
      selectedSuccesses,
      creationReason,
      agentBoundaries,
    }

    const loopResult = await this.runFailureAnalystLoop(context)
    if (!loopResult) {
      return { artifact: null, verifiedCause: false, recommendedAction: "observe" }
    }

    const { submission, turns } = loopResult
    const artifact: FailurePatternArtifact = {
      id: `failure_${hashContent(cluster.key).slice(0, 12)}`,
      taskType: cluster.taskType,
      pattern: submission.pattern ?? `${cluster.taskType} failure cluster requires follow-up`,
      analysisMode: "iterative_failure",
      selectionReason: submission.selectionReason ?? cluster.selectionReason,
      verifiedCause: true,
      confidence: clamp(submission.confidence ?? 0.75, 0, 1),
      implicatedAgents: uniqueStrings(submission.implicatedAgents ?? cluster.implicatedAgents),
      implicatedSkills: uniqueStrings(submission.implicatedSkills ?? cluster.implicatedSkills),
      suggestedCapability: submission.suggestedCapability,
      suggestedSkills: normalizeSkillList(submission.suggestedSkills ?? []),
      severity: cluster.tier === "P0" ? 0.9 : 0.65,
      evidence: buildArtifactEvidence(submission, selectedFailures, selectedSuccesses),
      metadata: {
        tier: cluster.tier,
        clusterKey: cluster.key,
        assignedAgent: cluster.assignedAgent,
        priority: Number(cluster.priority.toFixed(4)),
        matchedSuccessCount: selectedSuccesses.length,
        failureCount: cluster.failures.length,
        causalExplanation: submission.causalExplanation,
        proposedFix: submission.proposedFix,
        recommendedAction: submission.recommendedAction ?? "observe",
        analysisTurns: turns,
      },
    }

    return {
      artifact,
      verifiedCause: true,
      recommendedAction: submission.recommendedAction ?? "observe",
    }
  }

  private async runFailureAnalystLoop(
    context: FailureAnalystContext,
  ): Promise<{ submission: FailureAnalysisSubmission; turns: number } | null> {
    const messages: ChatMessage[] = [
      {
        role: "system",
        content: [
          `You are Error Analyst (A-) for ${this.domainDescription}.`,
          "You must investigate failures through tool use, not by guessing.",
          "On each turn, call one or more analysis tools to inspect evidence.",
          "Finish ONLY by calling submit_failure_analysis.",
          "A valid submission must identify a concrete root cause, propose a fix, and cite evidence refs.",
          "If you cannot reach a grounded causal explanation within the turn budget, do not fabricate one.",
        ].join("\n"),
      },
      {
        role: "user",
        content: [
          `Analyze a failure cluster for task type "${context.cluster.taskType}".`,
          `Priority tier: ${context.cluster.tier}`,
          `Selection reason: ${context.cluster.selectionReason}`,
          context.creationReason ? `Creation signal: ${context.creationReason}` : "",
          `Assigned agent: ${context.cluster.assignedAgent ?? "unknown"}`,
          `Implicated agents: ${context.cluster.implicatedAgents.join(", ") || "(none)"}`,
          `Implicated skills: ${context.cluster.implicatedSkills.join(", ") || "(none)"}`,
          "Use the tools to inspect full traces, IO artifacts, benchmark outcome comparison, and skill/agent boundaries.",
          "If the evidence never supports a fixable causal explanation, let the budget expire and the cluster will be excluded from the patch pool.",
        ].filter(Boolean).join("\n"),
      },
    ]

    const tools = this.buildFailureAnalysisToolSchemas()

    for (let turn = 0; turn < MAX_FAILURE_ANALYST_TURNS; turn++) {
      const response = await this.llm.chat(messages, {
        temperature: 0.1,
        tools,
        toolChoice: "required",
      })
      if (!response.toolCalls.length) return null

      messages.push({
        role: "assistant",
        content: response.content ?? "",
        tool_calls: response.toolCalls,
      })

      for (const toolCall of response.toolCalls) {
        const { name, arguments: rawArgs } = toolCall.function
        const args = parseStructuredJSON<Record<string, unknown>>(rawArgs) ?? {}
        const outcome = await this.executeFailureAnalysisTool(name, args, context)
        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: outcome.output,
        })

        if (outcome.submission) {
          const validated = validateFailureAnalysisSubmission(
            outcome.submission,
            context.cluster.selectionReason,
            context.selectedFailures.length,
            context.selectedSuccesses.length,
          )
          if (validated.valid && validated.submission) {
            return { submission: validated.submission, turns: turn + 1 }
          }
        }
      }
    }

    return null
  }

  private buildFailureAnalysisToolSchemas(): ToolSchema[] {
    return [
      {
        type: "function",
        function: {
          name: "list_evidence",
          description: "List the available failure/success exemplars, implicated skills, implicated agents, and what evidence can be inspected.",
          parameters: {
            type: "object",
            properties: {},
            additionalProperties: false,
          },
        },
      },
      {
        type: "function",
        function: {
          name: "read_trace",
          description: "Inspect the full action trace and step attribution for a selected failure or success exemplar.",
          parameters: {
            type: "object",
            properties: {
              kind: { type: "string", enum: ["failure", "success"] },
              index: { type: "integer", minimum: 0 },
            },
            required: ["kind", "index"],
            additionalProperties: false,
          },
        },
      },
      {
        type: "function",
        function: {
          name: "read_episode_io",
          description: "Inspect input/output artifacts for an exemplar, including task text, final assistant reply, user.reply outputs, and raw message snippets.",
          parameters: {
            type: "object",
            properties: {
              kind: { type: "string", enum: ["failure", "success"] },
              index: { type: "integer", minimum: 0 },
            },
            required: ["kind", "index"],
            additionalProperties: false,
          },
        },
      },
      {
        type: "function",
        function: {
          name: "compare_answer_to_ground_truth",
          description: "Compare the exemplar's answer/output against benchmark ground truth or benchmark scoring feedback when exact ground truth text is unavailable.",
          parameters: {
            type: "object",
            properties: {
              kind: { type: "string", enum: ["failure", "success"] },
              index: { type: "integer", minimum: 0 },
            },
            required: ["kind", "index"],
            additionalProperties: false,
          },
        },
      },
      {
        type: "function",
        function: {
          name: "read_skill_boundary",
          description: "Read the current content and usage boundary of an implicated skill.",
          parameters: {
            type: "object",
            properties: {
              skillId: { type: "string" },
            },
            required: ["skillId"],
            additionalProperties: false,
          },
        },
      },
      {
        type: "function",
        function: {
          name: "read_agent_boundary",
          description: "Read the current role boundary and scaling boundary of an implicated agent.",
          parameters: {
            type: "object",
            properties: {
              agentId: { type: "string" },
            },
            required: ["agentId"],
            additionalProperties: false,
          },
        },
      },
      {
        type: "function",
        function: {
          name: "submit_failure_analysis",
          description: "Submit the final grounded failure analysis after enough evidence has been inspected.",
          parameters: {
            type: "object",
            properties: {
              pattern: { type: "string" },
              selectionReason: {
                type: "string",
                enum: ["coverage_gap", "skill_overlap_conflict", "near_miss", "routing_boundary"],
              },
              confidence: { type: "number" },
              implicatedAgents: { type: "array", items: { type: "string" } },
              implicatedSkills: { type: "array", items: { type: "string" } },
              suggestedCapability: { type: "string" },
              suggestedSkills: { type: "array", items: { type: "string" } },
              recommendedAction: {
                type: "string",
                enum: ["create_skill", "scale_agent", "both", "observe"],
              },
              causalExplanation: { type: "string" },
              proposedFix: { type: "string" },
              evidenceRefs: {
                type: "array",
                items: {
                  type: "object",
                  properties: {
                    kind: { type: "string", enum: ["failure", "success"] },
                    index: { type: "integer", minimum: 0 },
                    observation: { type: "string" },
                  },
                  required: ["kind", "index"],
                  additionalProperties: false,
                },
              },
            },
            required: ["pattern", "causalExplanation", "proposedFix", "evidenceRefs"],
            additionalProperties: false,
          },
        },
      },
    ]
  }

  private async executeFailureAnalysisTool(
    name: string,
    args: Record<string, unknown>,
    context: FailureAnalystContext,
  ): Promise<{ output: string; submission?: FailureAnalysisSubmission }> {
    switch (name) {
      case "list_evidence":
        return { output: JSON.stringify(this.buildFailureEvidenceIndex(context), null, 2) }
      case "read_trace":
        return { output: JSON.stringify(this.readFailureTrace(args, context), null, 2) }
      case "read_episode_io":
        return { output: JSON.stringify(this.readEpisodeIO(args, context), null, 2) }
      case "compare_answer_to_ground_truth":
        return { output: JSON.stringify(this.compareAnswerToGroundTruth(args, context), null, 2) }
      case "read_skill_boundary":
        return { output: JSON.stringify(await this.readSkillBoundary(args), null, 2) }
      case "read_agent_boundary":
        return { output: JSON.stringify(this.readAgentBoundary(args, context), null, 2) }
      case "submit_failure_analysis": {
        const submission = parseFailureAnalysisSubmission(args)
        const validation = validateFailureAnalysisSubmission(
          submission,
          context.cluster.selectionReason,
          context.selectedFailures.length,
          context.selectedSuccesses.length,
        )
        return {
          output: JSON.stringify({
            accepted: validation.valid,
            issues: validation.issues,
          }, null, 2),
          submission,
        }
      }
      default:
        return { output: JSON.stringify({ error: `Unknown analysis tool: ${name}` }) }
    }
  }

  private buildFailureEvidenceIndex(context: FailureAnalystContext): Record<string, unknown> {
    return {
      taskType: context.cluster.taskType,
      tier: context.cluster.tier,
      selectionReason: context.cluster.selectionReason,
      assignedAgent: context.cluster.assignedAgent ?? null,
      implicatedAgents: context.cluster.implicatedAgents,
      implicatedSkills: context.cluster.implicatedSkills,
      failures: context.selectedFailures.map((failure, index) => ({
        index,
        episodeId: failure.episodeId,
        steps: failure.steps,
        score: failure.score ?? null,
        summary: failure.summary ?? null,
      })),
      successes: context.selectedSuccesses.map((success, index) => ({
        index,
        episodeId: success.episodeId,
        steps: success.steps,
        score: success.score ?? null,
        summary: success.summary ?? null,
      })),
      availableTools: [
        "list_evidence",
        "read_trace",
        "read_episode_io",
        "compare_answer_to_ground_truth",
        "read_skill_boundary",
        "read_agent_boundary",
        "submit_failure_analysis",
      ],
    }
  }

  private readFailureTrace(
    args: Record<string, unknown>,
    context: FailureAnalystContext,
  ): Record<string, unknown> {
    const selected = selectTrajectoryForAnalysis(args, context)
    if (!selected) return { error: "invalid_trace_reference" }
    const { kind, trajectory } = selected
    const toolCalls = (trajectory as FailureCase).toolCalls ?? (trajectory as EpisodeTrajectory).toolCalls
    return {
      kind,
      episodeId: trajectory.episodeId ?? null,
      task: trajectory.task,
      summary: trajectory.summary ?? null,
      steps: trajectory.steps,
      error: getTrajectoryError(trajectory),
      actionTrace: trajectory.actionTrace ?? [],
      stepAttribution: trajectory.stepAttribution ?? [],
      toolCalls: summarizeToolCalls(toolCalls ?? []),
    }
  }

  private readEpisodeIO(
    args: Record<string, unknown>,
    context: FailureAnalystContext,
  ): Record<string, unknown> {
    const selected = selectTrajectoryForAnalysis(args, context)
    if (!selected) return { error: "invalid_trace_reference" }
    const { kind, trajectory } = selected
    const messages = (trajectory as FailureCase).messages ?? (trajectory as EpisodeTrajectory).messages ?? []
    const taskText = trajectory.task
    const finalAssistantMessage = [...messages].reverse().find((message) => message.role === "assistant" && message.content.trim().length > 0)?.content ?? null
    const userReplies = extractUserReplySnippets((trajectory as FailureCase).toolCalls ?? (trajectory as EpisodeTrajectory).toolCalls ?? [])
    return {
      kind,
      episodeId: trajectory.episodeId ?? null,
      task: taskText,
      finalAssistantMessage,
      userReplyOutputs: userReplies,
      recentMessages: messages.slice(-6).map((message) => ({
        role: message.role,
        content: truncate(message.content, 240),
      })),
    }
  }

  private compareAnswerToGroundTruth(
    args: Record<string, unknown>,
    context: FailureAnalystContext,
  ): Record<string, unknown> {
    const selected = selectTrajectoryForAnalysis(args, context)
    if (!selected) return { error: "invalid_trace_reference" }
    const { kind, trajectory } = selected
    const messages = (trajectory as FailureCase).messages ?? (trajectory as EpisodeTrajectory).messages ?? []
    const finalAssistantMessage = [...messages].reverse().find((message) => message.role === "assistant" && message.content.trim().length > 0)?.content ?? null
    const userReplies = extractUserReplySnippets((trajectory as FailureCase).toolCalls ?? (trajectory as EpisodeTrajectory).toolCalls ?? [])
    return {
      kind,
      exactGroundTruthAvailable: false,
      benchmarkOutcome: {
        success: "success" in trajectory ? trajectory.success : false,
        score: trajectory.score ?? null,
        error: getTrajectoryError(trajectory),
      },
      answerArtifacts: {
        finalAssistantMessage,
        userReplyOutputs: userReplies,
      },
      comparisonGuidance: "Use benchmark success/score as the ground-truth proxy when exact reference answers are unavailable.",
    }
  }

  private async readSkillBoundary(args: Record<string, unknown>): Promise<Record<string, unknown>> {
    const skillId = typeof args.skillId === "string" ? args.skillId : ""
    if (!skillId) return { error: "missing_skill_id" }
    const skill = await this.skillManager.get(skillId)
    if (!skill) return { error: `skill_not_found:${skillId}` }
    return {
      skillId,
      description: skill.spec.description,
      whenToUse: skill.spec.whenToUse,
      tags: skill.spec.tags,
      generatedBy: skill.spec.generatedBy ?? "manual",
      content: skill.content,
    }
  }

  private readAgentBoundary(
    args: Record<string, unknown>,
    context: FailureAnalystContext,
  ): Record<string, unknown> {
    const agentId = typeof args.agentId === "string" ? args.agentId : ""
    if (!agentId) return { error: "missing_agent_id" }
    const snapshot = context.agentBoundaries?.[agentId]
    if (!snapshot) return { error: `agent_boundary_not_found:${agentId}` }
    return {
      agentId,
      roleDescription: snapshot.roleDescription ?? null,
      promptBoundary: snapshot.promptBoundary ?? null,
      skills: snapshot.skills ?? [],
      tools: snapshot.tools ?? [],
    }
  }

  private selectSuccessExamples(successes: EpisodeTrajectory[]): EpisodeTrajectory[] {
    return [...successes]
      .sort((a, b) => this.scoreSuccessExample(b) - this.scoreSuccessExample(a))
      .slice(0, this.maxEpisodesPerBlock)
  }

  private selectFailureExamples(failures: FailureCase[], avgSuccessSteps: number): FailureCase[] {
    return [...failures]
      .sort((a, b) => this.scoreFailureExample(b, avgSuccessSteps) - this.scoreFailureExample(a, avgSuccessSteps))
      .slice(0, this.maxEpisodesPerBlock)
  }

  private scoreSuccessExample(trajectory: EpisodeTrajectory): number {
    const traceDensity = clamp((trajectory.actionTrace?.length ?? 0) / Math.max(1, this.maxTraceLines), 0, 1)
    const stepEfficiency = 1 / Math.max(1, trajectory.steps)
    const skillSimplicity = 1 / Math.max(1, flattenSkills(trajectory.skillsUsed).length)
    return 0.45 * stepEfficiency + 0.35 * traceDensity + 0.20 * skillSimplicity
  }

  private scoreFailureExample(failure: FailureCase, avgSuccessSteps: number): number {
    const traceDensity = clamp((failure.actionTrace?.length ?? 0) / Math.max(1, this.maxTraceLines), 0, 1)
    const nearMiss = avgSuccessSteps > 0
      ? 1 - clamp(Math.abs(failure.steps - avgSuccessSteps) / Math.max(avgSuccessSteps, 1), 0, 1)
      : 0.4
    const overlapSignal = clamp(
      Math.max(0, failure.skillsActive.length - 1) * 0.25 +
      (inferFailureAgents(failure).length > 1 ? 0.35 : 0),
      0,
      1,
    )
    const errorSignal = failure.error ? 1 : 0.2
    return 0.35 * nearMiss + 0.25 * traceDensity + 0.20 * overlapSignal + 0.20 * errorSignal
  }

  private formatFailureCase(failure: FailureCase, index: number): string {
    const trace = failure.actionTrace?.slice(0, this.maxTraceLines).join("\n  ") ?? "(no trace)"
    const errorLine = failure.error ? `\nError: ${failure.error}` : ""
    const skillLine = failure.skillsActive.length > 0 ? `\nActive skills: ${failure.skillsActive.join(", ")}` : ""
    const summaryLine = failure.summary ? `\nSummary: ${failure.summary}` : ""
    return `### Failure ${index}: "${failure.task}" (${failure.steps} steps)${errorLine}${skillLine}${summaryLine}\n  ${trace}`
  }

  private formatSuccessTrajectory(success: EpisodeTrajectory, index: number): string {
    const trace = success.actionTrace?.slice(0, this.maxTraceLines).join("\n  ") ?? "(no trace)"
    const summaryLine = success.summary ? `\nSummary: ${success.summary}` : ""
    return `### Success ${index}: "${success.task}" (${success.steps} steps)${summaryLine}\n  ${trace}`
  }

  private toFailureCase(trajectory: EpisodeTrajectory): FailureCase {
    return {
      episodeId: trajectory.episodeId,
      taskType: trajectory.taskType,
      task: trajectory.task,
      error: trajectory.errorMessage,
      steps: trajectory.steps,
      score: trajectory.score,
      summary: trajectory.summary,
      actionTrace: trajectory.actionTrace,
      skillsUsed: trajectory.skillsUsed,
      skillsActive: flattenSkills(trajectory.skillsUsed),
      assignedAgent: trajectory.agentUsed,
      delegatedAgents: trajectory.delegatedAgents,
      stepAttribution: trajectory.stepAttribution,
      toolCalls: trajectory.toolCalls,
      messages: trajectory.messages,
    }
  }

  private toEpisodeTrajectory(failure: FailureCase): EpisodeTrajectory {
    return {
      episodeId: failure.episodeId,
      taskType: failure.taskType,
      task: failure.task,
      success: false,
      steps: failure.steps,
      score: failure.score,
      skillsUsed: failure.skillsUsed,
      agentUsed: failure.assignedAgent,
      delegatedAgents: failure.delegatedAgents,
      errorMessage: failure.error,
      summary: failure.summary,
      actionTrace: failure.actionTrace,
      stepAttribution: failure.stepAttribution,
      toolCalls: failure.toolCalls,
      messages: failure.messages,
    }
  }
}

/* ----- Types ----- */

/** Structured action patterns extracted from episode trajectories. */
interface ActionPatterns {
  toolFreq: Map<string, number>
  bigramFreq: Map<string, number>
  toolSequences: Set<string>
  avgSteps: number
  episodeCount: number
}

/* ----- Helpers ----- */

function flattenSkills(skillsUsed: Record<string, string[]>): string[] {
  const all = new Set<string>()
  for (const ids of Object.values(skillsUsed)) {
    for (const id of ids) all.add(id)
  }
  return [...all]
}

function inferFailureAgents(failure: FailureCase): string[] {
  const agents = new Set<string>()
  if (failure.assignedAgent) agents.add(failure.assignedAgent)
  for (const agentId of Object.keys(failure.skillsUsed ?? {})) {
    if (agentId) agents.add(agentId)
  }
  for (const delegated of failure.delegatedAgents ?? []) {
    if (delegated) agents.add(delegated)
  }
  for (const step of failure.stepAttribution ?? []) {
    if (step.actorId && step.actorId !== "unknown") agents.add(step.actorId)
    if (step.delegatedAgentId) agents.add(step.delegatedAgentId)
  }
  return [...agents].sort()
}

function hashContent(content: string): string {
  return createHash("sha256").update(content).digest("hex").slice(0, 16)
}

/**
 * Extract tool name from an action trace line.
 *
 * Supports formats:
 *   - "agentId > tool(args) → result"   (delegate sub-trace)
 *   - "tool(args) → result"             (direct tool call)
 */
function extractToolName(line: string): string | null {
  // "agent > tool(args) → ..."
  const delegateMatch = line.match(/>\s*([a-zA-Z0-9_.]+)\(/)
  if (delegateMatch) return delegateMatch[1]

  // "tool(args) → ..."
  const directMatch = line.match(/^([a-zA-Z0-9_.]+)\(/)
  if (directMatch) return directMatch[1]

  return null
}

/** Compute set difference: elements in A but not in B. */
function setDifference(a: Set<string>, b: Set<string>): Set<string> {
  const result = new Set<string>()
  for (const item of a) {
    if (!b.has(item)) result.add(item)
  }
  return result
}

/** Sort map entries by value descending, take top N. */
function sortedEntries(map: Map<string, number>, n: number): Array<[string, number]> {
  return [...map.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
}

function average(values: number[]): number {
  if (values.length === 0) return 0
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values.filter((value) => value.length > 0))].sort()
}

function normalizeSkillList(values: string[]): string[] {
  return uniqueStrings(values.map((value) => value.trim()).filter(Boolean))
}

function parseFailureAnalysisSubmission(
  args: Record<string, unknown>,
): FailureAnalysisSubmission {
  return {
    pattern: typeof args.pattern === "string" ? args.pattern.trim() : "",
    selectionReason: isFailureSelectionReason(args.selectionReason) ? args.selectionReason : undefined,
    confidence: typeof args.confidence === "number" ? args.confidence : undefined,
    implicatedAgents: Array.isArray(args.implicatedAgents)
      ? args.implicatedAgents.filter((value): value is string => typeof value === "string")
      : undefined,
    implicatedSkills: Array.isArray(args.implicatedSkills)
      ? args.implicatedSkills.filter((value): value is string => typeof value === "string")
      : undefined,
    suggestedCapability: typeof args.suggestedCapability === "string"
      ? args.suggestedCapability.trim()
      : undefined,
    suggestedSkills: Array.isArray(args.suggestedSkills)
      ? args.suggestedSkills.filter((value): value is string => typeof value === "string")
      : undefined,
    recommendedAction: isFailureRecommendedAction(args.recommendedAction)
      ? args.recommendedAction
      : undefined,
    causalExplanation: typeof args.causalExplanation === "string"
      ? args.causalExplanation.trim()
      : undefined,
    proposedFix: typeof args.proposedFix === "string"
      ? args.proposedFix.trim()
      : undefined,
    evidenceRefs: Array.isArray(args.evidenceRefs)
      ? args.evidenceRefs
        .filter((value): value is Record<string, unknown> => Boolean(value) && typeof value === "object")
        .map((value) => ({
          kind: value.kind === "failure" || value.kind === "success" ? value.kind : undefined,
          index: typeof value.index === "number" ? value.index : undefined,
          observation: typeof value.observation === "string" ? value.observation.trim() : undefined,
        }))
      : undefined,
  }
}

function validateFailureAnalysisSubmission(
  analysis: FailureAnalysisSubmission,
  fallbackReason: FailureSelectionReason,
  failureCount: number,
  successCount: number,
): { valid: boolean; issues: string[]; submission?: FailureAnalysisSubmission } {
  const normalizedReason = isFailureSelectionReason(analysis.selectionReason)
    ? analysis.selectionReason
    : fallbackReason
  const normalizedAction = isFailureRecommendedAction(analysis.recommendedAction)
    ? analysis.recommendedAction
    : normalizedReason === "routing_boundary"
      ? "scale_agent"
      : "create_skill"
  const normalized: FailureAnalysisSubmission = {
    pattern: typeof analysis.pattern === "string" ? analysis.pattern.trim() : "",
    selectionReason: normalizedReason,
    confidence: typeof analysis.confidence === "number" ? analysis.confidence : undefined,
    implicatedAgents: Array.isArray(analysis.implicatedAgents)
      ? analysis.implicatedAgents.filter((value): value is string => typeof value === "string")
      : undefined,
    implicatedSkills: Array.isArray(analysis.implicatedSkills)
      ? analysis.implicatedSkills.filter((value): value is string => typeof value === "string")
      : undefined,
    suggestedCapability: typeof analysis.suggestedCapability === "string"
      ? analysis.suggestedCapability.trim()
      : undefined,
    suggestedSkills: Array.isArray(analysis.suggestedSkills)
      ? analysis.suggestedSkills.filter((value): value is string => typeof value === "string")
      : undefined,
    recommendedAction: normalizedAction,
    causalExplanation: typeof analysis.causalExplanation === "string"
      ? analysis.causalExplanation.trim()
      : undefined,
    proposedFix: typeof analysis.proposedFix === "string"
      ? analysis.proposedFix.trim()
      : undefined,
    evidenceRefs: Array.isArray(analysis.evidenceRefs)
      ? analysis.evidenceRefs
        .filter((value): value is { kind?: "failure" | "success"; index?: number; observation?: string } => Boolean(value))
      : undefined,
  }
  const issues: string[] = []
  if (!normalized.pattern) issues.push("missing_pattern")
  if (!normalized.causalExplanation || normalized.causalExplanation.length < 24) issues.push("missing_causal_explanation")
  if (!normalized.proposedFix || normalized.proposedFix.length < 16) issues.push("missing_proposed_fix")
  const validRefs = (normalized.evidenceRefs ?? []).filter((ref) =>
    (ref.kind === "failure" || ref.kind === "success") &&
    typeof ref.index === "number" &&
    ref.index >= 0 &&
    ((ref.kind === "failure" && ref.index < failureCount) || (ref.kind === "success" && ref.index < successCount)),
  )
  if (validRefs.length === 0) issues.push("missing_valid_evidence_refs")
  if ((normalized.recommendedAction ?? "observe") === "observe") issues.push("recommended_action_observe")
  if (!normalized.implicatedAgents?.length && !normalized.implicatedSkills?.length) {
    issues.push("missing_implicated_targets")
  }
  normalized.evidenceRefs = validRefs
  return issues.length === 0
    ? { valid: true, issues, submission: normalized }
    : { valid: false, issues }
}

function isFailureSelectionReason(value: unknown): value is FailureSelectionReason {
  return value === "coverage_gap" ||
    value === "skill_overlap_conflict" ||
    value === "near_miss" ||
    value === "routing_boundary"
}

function buildArtifactEvidence(
  submission: FailureAnalysisSubmission,
  failures: FailureCase[],
  successes: EpisodeTrajectory[],
): NonNullable<FailurePatternArtifact["evidence"]> {
  const refs = submission.evidenceRefs ?? []
  return refs.map((ref) => {
    const source = ref.kind === "success" ? successes[ref.index ?? -1] : failures[ref.index ?? -1]
    if (!source) {
      return {
        summary: ref.observation ?? "missing evidence source",
      }
    }
    return {
      episodeId: source.episodeId,
      task: source.task,
      summary: ref.observation ?? source.summary ?? getTrajectoryError(source) ?? `evidence from ${ref.kind} exemplar ${ref.index}`,
      trajectoryRef: source.episodeId,
    }
  })
}

function selectTrajectoryForAnalysis(
  args: Record<string, unknown>,
  context: FailureAnalystContext,
): { kind: "failure" | "success"; trajectory: FailureCase | EpisodeTrajectory } | null {
  const kind = args.kind === "failure" || args.kind === "success" ? args.kind : null
  const index = typeof args.index === "number" ? args.index : NaN
  if (!kind || Number.isNaN(index) || index < 0) return null
  const source = kind === "failure" ? context.selectedFailures : context.selectedSuccesses
  const trajectory = source[index]
  return trajectory ? { kind, trajectory } : null
}

function summarizeToolCalls(toolCalls: ToolCallRecord[]): Array<Record<string, unknown>> {
  return toolCalls.map((call) => ({
    tool: call.tool,
    actorId: call.actorId ?? null,
    delegatedAgentId: call.delegatedAgentId ?? null,
    activeSkillIds: call.activeSkillIds ?? [],
    stepOutcome: call.stepOutcome ?? (call.ok ? "success" : "failure"),
    args: sanitizeJson(call.args),
    result: truncate(call.result, 240),
  }))
}

function extractUserReplySnippets(toolCalls: ToolCallRecord[]): string[] {
  const outputs: string[] = []
  const visit = (calls: ToolCallRecord[]) => {
    for (const call of calls) {
      if (call.tool === "user.reply" && call.result) {
        outputs.push(truncate(call.result, 240))
      }
      if (call.subTrace?.toolCalls) visit(call.subTrace.toolCalls)
    }
  }
  visit(toolCalls)
  return outputs.slice(-5)
}

function truncate(text: string, maxChars: number): string {
  return text.length <= maxChars ? text : `${text.slice(0, Math.max(0, maxChars - 3))}...`
}

function sanitizeJson(value: unknown): unknown {
  try {
    return JSON.parse(JSON.stringify(value))
  } catch {
    return String(value)
  }
}

function getTrajectoryError(source: FailureCase | EpisodeTrajectory): string | null {
  if (typeof (source as FailureCase).error === "string") return (source as FailureCase).error ?? null
  if (typeof (source as EpisodeTrajectory).errorMessage === "string") return (source as EpisodeTrajectory).errorMessage ?? null
  return null
}

function isFailureRecommendedAction(value: unknown): value is FailureRecommendedAction {
  return value === "create_skill" ||
    value === "scale_agent" ||
    value === "both" ||
    value === "observe"
}

function parseStructuredJSON<T>(text: string): T | null {
  let raw = text.trim()
  if (raw.startsWith("```")) {
    const lines = raw.split("\n")
    raw = lines.slice(1, -1).join("\n")
  }
  try {
    return JSON.parse(raw) as T
  } catch {
    const match = raw.match(/\{[\s\S]*\}/)
    if (match) {
      try {
        return JSON.parse(match[0]) as T
      } catch {
        return null
      }
    }
    return null
  }
}

/* ----- Deduplication ----- */

const DEDUP_STOPWORDS = new Set([
  "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
  "have", "has", "had", "do", "does", "did", "will", "would", "could",
  "should", "may", "might", "shall", "can", "need", "dare", "ought",
  "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
  "into", "through", "during", "before", "after", "and", "but", "or",
  "nor", "not", "so", "yet", "both", "either", "neither", "each",
  "this", "that", "these", "those", "it", "its", "when", "where",
  "how", "what", "which", "who", "whom", "if", "then", "than",
  "use", "using", "task", "tasks", "step", "steps", "agent",
])

/** Tokenize text for dedup similarity comparison. */
function dedupTokenize(text: string): Set<string> {
  const words = text.toLowerCase().replace(/[^a-z0-9\s_.-]/g, " ").split(/\s+/)
  const tokens = new Set<string>()
  for (const w of words) {
    if (w.length > 1 && !DEDUP_STOPWORDS.has(w)) tokens.add(w)
  }
  return tokens
}

/** Compute Jaccard similarity between two token sets. */
function jaccardSimilarity(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) return 1
  if (a.size === 0 || b.size === 0) return 0
  let intersection = 0
  for (const t of a) {
    if (b.has(t)) intersection++
  }
  const union = new Set([...a, ...b]).size
  return union === 0 ? 0 : intersection / union
}
