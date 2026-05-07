import type { ToolCallRecord } from "../runtime/types.js"

export type RetentionReason =
  | "repeated_failure"
  | "near_miss"
  | "reusable_success"
  | "retrieval_execution_mismatch"

export interface VerifiedTrace {
  episodeId?: string
  taskType: string
  task: string
  success: boolean
  score?: number
  steps: number
  maxSteps?: number
  selectedSkills?: Record<string, string[]>
  usedSkills?: Record<string, string[]>
  executors: string[]
  primaryExecutor?: string
  delegatedExecutors?: string[]
  toolCalls?: ToolCallRecord[]
  actionTrace?: string[]
  stepAttribution?: Array<{
    agentId?: string
    delegatedAgentId?: string
    skillIds?: string[]
    tool?: string
    stepOutcome?: "success" | "failure"
  }>
  errorMessage?: string
  summary?: string
  retrievalExecutionMismatch?: boolean
  metadata?: Record<string, unknown>
}

export interface RetainedEvidence {
  trace: VerifiedTrace
  reasons: RetentionReason[]
  priority: number
}

export interface RetentionOptions {
  repeatedFailureThreshold?: number
  nearMissStepRatio?: number
  reusableSuccessMaxStepRatio?: number
  maxRetained?: number
  previousFailureCounts?: Record<string, number>
}

export class EvidenceRetainer {
  private repeatedFailureThreshold: number
  private nearMissStepRatio: number
  private reusableSuccessMaxStepRatio: number
  private maxRetained: number
  private failureCounts: Map<string, number>

  constructor(options?: RetentionOptions) {
    this.repeatedFailureThreshold = options?.repeatedFailureThreshold ?? 2
    this.nearMissStepRatio = options?.nearMissStepRatio ?? 0.8
    this.reusableSuccessMaxStepRatio = options?.reusableSuccessMaxStepRatio ?? 0.6
    this.maxRetained = options?.maxRetained ?? 64
    this.failureCounts = new Map(Object.entries(options?.previousFailureCounts ?? {}))
  }

  retain(traces: VerifiedTrace[]): RetainedEvidence[] {
    const retained: RetainedEvidence[] = []

    for (const trace of traces) {
      const reasons = this.getReasons(trace)
      if (reasons.length === 0) continue
      retained.push({
        trace,
        reasons,
        priority: this.priority(trace, reasons),
      })
    }

    return retained
      .sort((a, b) => b.priority - a.priority)
      .slice(0, this.maxRetained)
  }

  snapshotFailureCounts(): Record<string, number> {
    return Object.fromEntries(this.failureCounts)
  }

  private getReasons(trace: VerifiedTrace): RetentionReason[] {
    const reasons: RetentionReason[] = []

    if (!trace.success) {
      const key = failureKey(trace)
      const count = (this.failureCounts.get(key) ?? 0) + 1
      this.failureCounts.set(key, count)
      if (count >= this.repeatedFailureThreshold) reasons.push("repeated_failure")
    }

    if (isNearMiss(trace, this.nearMissStepRatio)) {
      reasons.push("near_miss")
    }

    if (isReusableSuccess(trace, this.reusableSuccessMaxStepRatio)) {
      reasons.push("reusable_success")
    }

    if (trace.retrievalExecutionMismatch || hasSkillMismatch(trace)) {
      reasons.push("retrieval_execution_mismatch")
    }

    return unique(reasons)
  }

  private priority(trace: VerifiedTrace, reasons: RetentionReason[]): number {
    let score = 0
    if (reasons.includes("repeated_failure")) score += 4
    if (reasons.includes("retrieval_execution_mismatch")) score += 3
    if (reasons.includes("near_miss")) score += 2
    if (reasons.includes("reusable_success")) score += 1
    if (!trace.success) score += 1
    score += Math.min(1, (trace.toolCalls?.length ?? trace.actionTrace?.length ?? 0) / 20)
    return score
  }
}

export function retainEvidence(
  traces: VerifiedTrace[],
  options?: RetentionOptions,
): RetainedEvidence[] {
  return new EvidenceRetainer(options).retain(traces)
}

function failureKey(trace: VerifiedTrace): string {
  const agents = trace.executors.length > 0 ? trace.executors.join("|") : "unknown"
  const error = trace.errorMessage ? normalize(trace.errorMessage).slice(0, 80) : "no_error"
  return `${trace.taskType}::${agents}::${error}`
}

function isNearMiss(trace: VerifiedTrace, stepRatio: number): boolean {
  if (trace.success) return false
  if (typeof trace.score === "number" && trace.score > 0 && trace.score < 1) return true
  if (!trace.maxSteps || trace.maxSteps <= 0) return false
  return trace.steps / trace.maxSteps >= stepRatio
}

function isReusableSuccess(trace: VerifiedTrace, maxStepRatio: number): boolean {
  if (!trace.success) return false
  if (!trace.maxSteps || trace.maxSteps <= 0) return true
  return trace.steps / trace.maxSteps <= maxStepRatio
}

function hasSkillMismatch(trace: VerifiedTrace): boolean {
  const selected = flattenSkillMap(trace.selectedSkills)
  if (selected.size === 0) return false
  const used = flattenSkillMap(trace.usedSkills)
  if (used.size === 0) return true
  for (const skillId of selected) {
    if (!used.has(skillId)) return true
  }
  return false
}

function flattenSkillMap(value?: Record<string, string[]>): Set<string> {
  const result = new Set<string>()
  for (const skillIds of Object.values(value ?? {})) {
    for (const skillId of skillIds) result.add(skillId)
  }
  return result
}

function unique<T>(values: T[]): T[] {
  return [...new Set(values)]
}

function normalize(value: string): string {
  return value.toLowerCase().replace(/\s+/g, " ").trim()
}
