/**
 * Runtime type definitions.
 *
 * Separating types from implementation keeps imports clean and avoids
 * circular dependencies between runtime modules.
 */

import type { LLMClient } from "../llm/client.js"
import type { ToolExecutor } from "../tool/executor.js"
import type { ToolRegistry } from "../tool/registry.js"
import type { SkillManager } from "../skill/manager.js"
import type { RuntimeSkillContract, RuntimeSkillPatch } from "../spec/skill.js"
import type { ToolMode } from "../llm/types.js"

export type SessionMode = "single" | "episodic" | "batch"

export interface ContextBudget {
  maxChars?: number
  maxMessages?: number
  reserveChars?: number
}

export interface CheckpointingOptions {
  enabled?: boolean
  intervalSteps?: number
  outputDir?: string
}

export interface SessionRunResult {
  runId?: string
  response: string
  success?: boolean
  metadata?: Record<string, unknown>
}

export interface MemoryManagerPort {
  search?(query: string, limit?: number): Promise<string[]>
  append?(entry: string, metadata?: Record<string, unknown>): Promise<void>
}

export interface AgentSkillPoolPort {
  resolve?(agentId: string, taskType?: string): Promise<RuntimeSkillTuple[]>
}

export type FailureClass =
  | "tool_error"
  | "delegation_error"
  | "policy_error"
  | "task_failure"
  | "unknown"

export interface ContextFragment {
  id: string
  source: string
  purpose: string
  channel: "system" | "user" | "planner" | "delegate"
  priority: number
  dedupeKey: string
  charEstimate: number
  tokenEstimate: number
  included: boolean
  renderedText: string
  provenance?: Record<string, unknown>
}

export interface ContextTrace {
  fragments: ContextFragment[]
  totalChars: number
  estimatedTokens: number
}

export interface CaseState {
  [key: string]: unknown
}

export interface CaseDelta {
  [key: string]: unknown
}

export type CognitiveStage =
  | "discover"
  | "propose"
  | "check"
  | "execute"
  | "finalize"
  | "reflect"
  | "unknown"

export type ActionDomain = "U" | "T" | "F"

export interface RuntimeSkillTuple {
  skillId: string
  intent?: string
  method?: string
  difficulty?: {
    effect?: string
    level?: "low" | "medium" | "high" | "unknown"
  }
  toolHints?: string[]
  family?: string
  agent?: string
  runtimePatches?: RuntimeSkillPatch[]
  runtimeContracts?: RuntimeSkillContract[]
}

export interface SkillCompositionTrace {
  operator: "online_phi"
  activeSkillIds: string[]
  constraints: string
  stage: CognitiveStage
}

export interface AgenticContext {
  cognitiveStage?: CognitiveStage
  activeSkills?: RuntimeSkillTuple[]
  skillComposition?: SkillCompositionTrace
}

export interface AgenticObservation {
  activeSkillIds: string[]
  historyLength: number
  cognitiveStage: CognitiveStage
  actionDomain: ActionDomain
  skillComposition?: SkillCompositionTrace
}

export interface SubagentToolBudget {
  maxSteps?: number
  maxRetries?: number
}

export interface SubagentCall {
  workerId?: string
  taskType?: string
  graduationStage?: string
  workItemId?: string
  objective: string
  caseSlice?: CaseState
  exitCriteria?: string[]
  toolScope?: string[]
  toolBudget?: SubagentToolBudget
  rawTask?: string
}

export interface ReplyAct {
  kind: "reply"
  message: string
}

export interface ToolAct {
  kind: "tool"
  name: string
  args: Record<string, unknown>
}

export interface DelegateAct {
  kind: "delegate"
  call: SubagentCall
}

export interface FinishAct {
  kind: "finish"
}

export interface CognitiveAct {
  kind: "cognitive"
  name: "think" | "revise_plan" | "compose_skill_patch" | "propose_skill_use" | "request_shadow_check"
  content: string
}

export type AgentAct = ReplyAct | ToolAct | DelegateAct | FinishAct | CognitiveAct

export interface WorkerResult {
  status: "completed" | "needs_user" | "blocked" | "failed"
  caseDelta?: CaseDelta
  workItemUpdate?: {
    workItemId?: string
    status?: "open" | "ready" | "running" | "blocked" | "completed"
    completionOwner?: "manager" | "worker"
    summary?: string | null
  }
  userReply?: string | null
  failureClass?: string | null
  telemetry?: {
    toolsUsed?: string[]
    steps?: number
    iterations?: number
  }
  rawOutput?: string
}

export interface AgentDeps {
  llm: LLMClient
  toolExecutor: ToolExecutor
  toolRegistry: ToolRegistry
  memoryManager?: MemoryManagerPort
  skillManager: SkillManager
  /** Isolated skill pool for this specific agent */
  skillPool?: AgentSkillPoolPort
  /** For orchestrators: returns current worker descriptions to inject into prompt */
  getWorkerDescriptions?: () => string | null
  /** Global policy rules (from policy.md) injected into every agent's system prompt */
  globalPolicy?: string | null
  /** Absolute path to the active global policy file, when present. */
  globalPolicyPath?: string | null
}

export interface RunOptions {
  maxIterations?: number
  maxCyclesPerEpisode?: number
  maxDelegationSteps?: number
  maxDelegationRetries?: number
  sessionMode?: SessionMode
  contextBudget?: ContextBudget
  checkpointing?: CheckpointingOptions
  taskType?: string
  signal?: AbortSignal
  temperature?: number
  model?: string
  /** Override tool calling strategy: "native" (API tools param) or "prompt" (text-based). */
  toolMode?: ToolMode
  /** Force the model to call a tool on the first iteration (prevents zero-step episodes). */
  forceFirstToolCall?: boolean
  /** Return control after the first completed user.reply round-trip. */
  stopAfterUserReplyRoundTrip?: boolean
  /** Return control after the first completed delegate round-trip. */
  stopAfterDelegateRoundTrip?: boolean
  /** Extra content appended to the system prompt (e.g. domain policy documents). */
  systemPromptSuffix?: string
  /** Per-run dynamic natural-language steering, e.g. task-conditioned advisor guidance. */
  instanceAdvice?: string
  /** Absolute path to the code that generated instanceAdvice, when known. */
  instanceAdvicePath?: string
  /** Optional shared conversation transcript injected before the current input. */
  sharedMessages?: import("../llm/types.js").ChatMessage[]
  /** Optional shared canonical case state injected into the prompt/runtime. */
  sharedState?: CaseState
  /** Agentic process observation context: o_t = <K_t, h_t, sigma_t>. */
  agenticContext?: AgenticContext
  onToolCall?: (toolName: string, args: Record<string, unknown>) => void
  onToolResult?: (toolName: string, result: string, ok: boolean) => void
  onIteration?: (iteration: number) => void
}

export interface ToolCallRecord {
  tool: string
  act?: AgentAct
  actorId?: string
  activeSkillIds?: string[]
  agentic?: AgenticObservation
  delegatedAgentId?: string
  /** Structured metadata for delegate calls after runtime routing/retries. */
  delegation?: {
    agentId: string
    requestedWorkerId?: string
    taskType?: string
    graduationStage?: string
    success: boolean
    attempts?: string[]
    summary?: string
    failureReason?: string
    subtaskId?: string
    planNodeId?: string
    failureClass?: FailureClass
    checkpointId?: string
  }
  stepOutcome?: "success" | "failure"
  args: Record<string, unknown>
  result: string
  ok: boolean
  durationMs?: number
  attribution?: {
    contextFragmentIds?: string[]
    promptFiles?: string[]
    sharedTranscriptPresent?: boolean
    sharedTranscriptMessages?: number
  }
  workerResult?: WorkerResult
  /** For delegate calls: structured sub-agent execution trace. */
  subTrace?: {
    agentId: string
    toolCalls: ToolCallRecord[]
    iterations: number
    usage: { promptTokens: number; completionTokens: number }
    contextTrace?: ContextTrace
    messages?: import("../llm/types.js").ChatMessage[]
  }
}

export interface RunResult {
  response: string
  toolCalls: ToolCallRecord[]
  iterations: number
  usage: {
    promptTokens: number
    completionTokens: number
  }
  /** Full LLM conversation history (system + user + assistant + tool messages). */
  messages?: import("../llm/types.js").ChatMessage[]
  /** Structured breakdown of prompt/context fragments included in the run. */
  contextTrace?: ContextTrace
  /** Number of conversation messages trimmed by the sliding window. */
  trimmedMessages?: number
}
