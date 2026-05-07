/**
 * Tool executor — bridges ToolDefinition schemas to actual handler functions.
 *
 * The registry stores *definitions* (JSON metadata).
 * The executor stores *handlers* (functions that actually run).
 * Together they give an agent both the schema (for LLM) and the logic (for execution).
 */

import type { ToolDefinition } from "./registry.js"
import type { ChatMessage, ToolSchema } from "../llm/types.js"
import type { CaseState, RuntimeSkillTuple } from "../runtime/types.js"

export interface ToolResult {
  ok: boolean
  output: string
  error?: string
  durationMs?: number
  /** Opaque metadata returned by the handler (e.g., structured sub-agent trace). */
  metadata?: Record<string, unknown>
}

export interface ToolExecutionContext {
  agentId: string
  messages?: ChatMessage[]
  sharedState?: CaseState
  iteration?: number
  activeSkillIds?: string[]
  activeSkills?: RuntimeSkillTuple[]
}

export type ToolHandler = (
  args: Record<string, unknown>,
  context?: ToolExecutionContext,
) => Promise<string | { output: string; metadata: Record<string, unknown> }>

export class ToolExecutor {
  private handlers = new Map<string, ToolHandler>()

  register(toolId: string, handler: ToolHandler): void {
    this.handlers.set(toolId, handler)
  }

  registerMany(tools: Record<string, ToolHandler>): void {
    for (const [id, handler] of Object.entries(tools)) {
      this.handlers.set(id, handler)
    }
  }

  has(toolId: string): boolean {
    return this.handlers.has(toolId)
  }

  list(): string[] {
    return Array.from(this.handlers.keys())
  }

  async execute(toolId: string, args: Record<string, unknown>, context?: ToolExecutionContext): Promise<ToolResult> {
    const handler = this.handlers.get(toolId)
    if (!handler) {
      return { ok: false, output: "", error: `Tool "${toolId}" not registered` }
    }

    const start = Date.now()
    try {
      const raw = await handler(args, context)
      const durationMs = Date.now() - start
      if (typeof raw === "string") {
        return { ok: true, output: raw, durationMs }
      }
      return { ok: true, output: raw.output, metadata: raw.metadata, durationMs }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      return { ok: false, output: "", error: msg, durationMs: Date.now() - start }
    }
  }

  /**
   * Convert ToolDefinition[] to OpenAI function-calling schemas.
   * Only includes tools that have registered handlers.
   */
  toOpenAISchemas(definitions: ToolDefinition[]): ToolSchema[] {
    return definitions
      .filter((d) => this.handlers.has(d.id))
      .map((d) => {
        const properties: Record<string, Record<string, unknown>> = {}
        for (const [name, param] of Object.entries(d.parameters)) {
          const prop: Record<string, unknown> = { type: param.type, description: param.description }
          // Preserve array item schema if present
          if (param.type === "array" && (param as any).items) {
            prop.items = (param as any).items
          }
          properties[name] = prop
        }
        return {
          type: "function" as const,
          function: {
            name: d.id,
            description: d.description,
            parameters: {
              type: "object",
              properties,
              required: Object.entries(d.parameters)
                .filter(([, param]) => param.required)
                .map(([name]) => name),
            },
          },
        }
      })
  }
}
