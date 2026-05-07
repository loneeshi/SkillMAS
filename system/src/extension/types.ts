/**
 * Extension types — proposals for dynamically extending the agent tree.
 *
 * v2: Added subtask tracking and LLM-designed expansion.
 * v3: Added optimization triggers, step tracking, efficiency metrics,
 *     staged expansion (rescue → optimize), convergence tracking.
 */

export type ExtensionType =
  | "add_worker"
  | "add_skill"
  | "refine_skill"
  | "add_composite_tool"
  | "scale_agents"

export interface ExtensionProposal {
  extensionType: ExtensionType
  reason: string
  priority: number
  trigger: ExpansionTrigger
  details: Record<string, unknown>
  /** Human-readable explanation of WHY this extension was created.
   *  E.g. "Created returns_specialist because 15 episodes involved exchanges/returns
   *  that required specialized knowledge of item variants and one-shot exchange calls." */
  explanation?: string
}

export interface TaskResult {
  taskId: string
  taskType: string
  agentUsed: string
  success: boolean
  errorMessage?: string
  durationMs?: number
  /** Total tool calls (env actions) in this episode */
  steps?: number
  /** Number of delegate calls from manager */
  delegateCalls?: number
  metadata?: Record<string, unknown>
}

export interface SubtaskResult {
  taskId: string
  subtaskIndex: number
  subtaskType: string
  workerId: string
  success: boolean
  toolCalls: Array<{ tool: string; args: Record<string, unknown>; result: string }>
  errorMessage?: string
  timestamp?: number
  /** How many tool calls this subtask took */
  steps?: number
  graduationStage?: string
  proposal?: {
    kind: "ready" | "blocked" | "executed" | "unknown"
    tool?: string
    args?: Record<string, unknown>
    rawOutput: string
  }
}

// ─── Performance Stats ──────────────────────────────────────────────

export interface TypeStats {
  total: number
  success: number
  rate: number
}

export interface TypeStatsWithEfficiency extends TypeStats {
  /** Step counts for successful tasks only */
  stepStats: StepStats
}

export interface StepStats {
  /** Number of successful tasks with step data */
  count: number
  /** Minimum steps among successes */
  min: number
  /** Maximum steps among successes */
  max: number
  /** Mean steps */
  mean: number
  /** Median steps */
  median: number
  /** 25th percentile */
  p25: number
  /** 75th percentile */
  p75: number
  /** Standard deviation */
  stddev: number
}

export interface PerformanceStats {
  totalTasks: number
  successCount: number
  failureCount: number
  successRate: number
  byTaskType: Record<string, TypeStatsWithEfficiency>
  byAgent: Record<string, TypeStats>
  bySubtask: Record<string, TypeStats>
  failurePatterns: Record<string, number>
}

// ─── Expansion Triggers ─────────────────────────────────────────────

export type TriggerType =
  | "agent_scaling"              // LLM decided the current team shape needs splitting/rebalancing
  | "worker_capability_gap"      // success rate too low → need new worker
  | "skill_gap"                  // worker has tools but wrong strategy → need new skill
  | "step_efficiency"            // tasks succeed but take too many steps → refine skill
  | "near_miss"                  // tasks barely succeed (use >80% of max steps) → refine strategy
  | "consistency_gap"            // high variance in steps for same task type → stabilize strategy
  | "subtask_bottleneck"         // one worker is the weak link → refine that worker's skill
  | "composite_tool_need"        // repeated tool sequences → create composite tool

export interface ExpansionTrigger {
  type: TriggerType
  reason: string
  evidence: {
    failureCount: number
    successRate: number
    recentFailures: TaskResult[]
    /** For optimization triggers */
    stepStats?: StepStats
    /** Which task type this trigger is about */
    taskType?: string
    /** Which worker is the bottleneck */
    bottleneckWorker?: string
  }
  gap: {
    description: string
    currentCoverage: string[]
    neededCapability: string
  }
}

// ─── Agent Scaling ─────────────────────────────────────────────────

export interface FailurePatternArtifact {
  id?: string
  taskType: string
  pattern: string
  analysisMode?: "iterative_failure"
  selectionReason?: "coverage_gap" | "skill_overlap_conflict" | "near_miss" | "routing_boundary"
  verifiedCause?: boolean
  confidence?: number
  evidence?: Array<{
    episodeId?: string
    task?: string
    summary?: string
    trajectoryRef?: string
  }>
  implicatedAgents?: string[]
  implicatedSkills?: string[]
  suggestedCapability?: string
  suggestedSkills?: string[]
  severity?: number
  metadata?: Record<string, unknown>
}

export interface AgentPatchPlan {
  agentId: string
  roleDescriptionPatch?: string
  promptBoundaryPatch?: string
  addSkills?: string[]
  removeSkills?: string[]
  rationale?: string
}

export interface SkillAssignmentPlan {
  skillId: string
  assignTo: string[]
  removeFrom?: string[]
  rationale?: string
}

export interface AgentScalingDecision {
  shouldScale: boolean
  reason: string
  newWorker?: {
    id: string
    name?: string
    roleDescription: string
    tools: string[]
    skills: string[]
    promptBoundary: string
    routingHint?: string
  }
  existingAgentPatches?: AgentPatchPlan[]
  skillAssignments?: SkillAssignmentPlan[]
  expectedBenefit?: string
  risk?: string
}

export interface AgentScalingResult {
  skipped: boolean
  reason?: string
  decision?: AgentScalingDecision
  proposals: ExtensionProposal[]
  newAgentIds: string[]
  patchedAgentIds: string[]
}

// ─── Extension Phase ────────────────────────────────────────────────

/**
 * The extension engine operates in two phases:
 *
 * Phase 1 (rescue): success rate < threshold
 *   - Creates new workers for capability gaps
 *   - Creates new skills for strategy gaps
 *   - Goal: get success rate above threshold
 *
 * Phase 2 (optimize): success rate >= threshold
 *   - Refines existing skills for efficiency
 *   - Stabilizes inconsistent strategies
 *   - Goal: reduce step count and variance
 */
export type ExtensionPhase = "rescue" | "optimize"

// ─── Convergence Tracking ───────────────────────────────────────────

/**
 * Tracks the system's evolution trajectory over time.
 * Used to verify the system is converging toward a better state.
 */
export interface ConvergenceCheckpoint {
  timestamp: string
  episode: number
  phase: ExtensionPhase
  metrics: {
    successRate: number
    avgSteps: number
    medianSteps: number
    /** How many task types are above rescue threshold */
    typesAboveThreshold: number
    totalTypes: number
  }
  /** What changed at this checkpoint */
  changes: Array<{
    type: "agent_added" | "skill_added" | "skill_refined" | "agent_disabled"
    entityId: string
    reason: string
  }>
}

// ─── LLM Design Outputs ────────────────────────────────────────────

export interface LLMWorkerDesign {
  id: string
  role: "worker"
  description: string
  capabilities: string[]
  tools: string[]
  skills: string[]
  prompt_strategy: string
  mount_point: string
  differentiation: string
  routing_hint?: string
}

export interface LLMSkillDesign {
  id: string
  type: "manual" | "knowledge" | "workflow"
  for_worker: string
  description: string
  content: string
  examples?: string
  allowedTools?: string[]
}

export interface LLMSkillRefinement {
  skill_id: string
  changes: string
  new_sections: string
  removed_sections: string[]
  rationale: string
}

export interface LLMCompositeToolDesign {
  id: string
  description: string
  parameters: Record<string, { type: string; description: string; required: boolean }>
  steps: Array<{ tool: string; args_template: Record<string, string> }>
  for_workers: string[]
}

export interface ExpansionRecord {
  id: string
  type: "worker" | "skill" | "skill_refinement" | "composite_tool" | "agent_patch"
  createdAt: string
  trigger: ExpansionTrigger
  entityId: string
  performance: {
    tasksHandled: number
    successRate: number
    baselineRate: number
    /** For optimization records: baseline avg steps before refinement */
    baselineAvgSteps?: number
    /** Current avg steps after refinement */
    currentAvgSteps?: number
  }
  status: "active" | "disabled"
  /** Reason for final decision */
  verdict?: string
  metadata?: Record<string, unknown>
}
