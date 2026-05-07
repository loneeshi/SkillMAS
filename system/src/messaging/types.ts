/**
 * Messaging types for inter-agent communication.
 *
 * Supports three patterns:
 *   1. Direct message — agent-to-agent (p2p)
 *   2. Broadcast — one agent → all agents in the tree
 *   3. Delegation — manager → worker task dispatch with response
 */

export interface AgentMessage {
  id: string
  from: string
  to: string | "*"
  type: MessageType
  content: string
  metadata?: Record<string, unknown>
  timestamp: string
  replyTo?: string
}

export type MessageType =
  | "task"
  | "result"
  | "info"
  | "error"
  | "delegate"
  | "delegate_result"

export interface DelegateRequest {
  taskId: string
  from: string
  to: string
  input: string
  metadata?: Record<string, unknown>
  subagentCall?: import("../runtime/types").SubagentCall
}

export interface DelegateResult {
  taskId: string
  from: string
  to: string
  success: boolean
  output: string
  toolCalls?: Array<{ tool: string; args: Record<string, unknown>; result: string }>
  error?: string
  /** Full sub-agent RunResult for structured trace capture. */
  runResult?: import("../runtime/types").RunResult
  /** Structured worker outcome for reducer-style runtimes. */
  workerResult?: import("../runtime/types").WorkerResult
}

export type MessageHandler = (message: AgentMessage) => void | Promise<void>
