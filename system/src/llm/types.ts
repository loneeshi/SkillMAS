/**
 * LLM types — OpenAI-compatible chat completion format.
 *
 * Kept as plain interfaces (no runtime deps) so any module can import
 * without pulling in the HTTP client.
 */

/** Whether tools are passed via the API `tools` param or injected into the prompt as text. */
export type ToolMode = "native" | "prompt"

export type ToolChoice =
  | "auto"
  | "required"
  | "none"
  | { type: "function"; function: { name: string } }

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool"
  content: string
  tool_calls?: ToolCallRequest[]
  tool_call_id?: string
  name?: string
}

export interface ToolCallRequest {
  id: string
  type: "function"
  function: {
    name: string
    arguments: string
  }
}

export interface ChatResponse {
  content: string
  toolCalls: ToolCallRequest[]
  finishReason: string
  /** Non-null when the model refuses to comply (GPT-4+). */
  refusal?: string | null
  usage?: {
    promptTokens: number
    completionTokens: number
    totalTokens: number
    /** Reasoning/thinking tokens used by o1/o3 models. */
    reasoningTokens?: number
  }
}

export interface ToolSchema {
  type: "function"
  function: {
    name: string
    description: string
    /** Enables strict schema validation (GPT-4o+ / GPT-5). */
    strict?: boolean
    parameters: Record<string, unknown>
  }
}

export interface ChatOptions {
  model?: string
  temperature?: number
  maxTokens?: number
  tools?: ToolSchema[]
  toolChoice?: ToolChoice
  toolMode?: ToolMode
  signal?: AbortSignal
}
