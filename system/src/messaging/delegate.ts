/**
 * Delegation — a manager dispatches a sub-task to a worker and
 * awaits the result.
 *
 * Supports two paths:
 *   1. TaskManager path (durable): every delegation becomes a TaskRecord
 *      with journal, artifacts, and optional routing.
 *   2. Legacy path (in-memory): direct agent.run() for backward compatibility.
 *
 * To enable legacy mode (no TaskManager), omit `taskManager` in options.
 */

import type { AgentRuntime } from "../runtime/agent"
import type { RunOptions, SubagentCall, WorkerResult } from "../runtime/types"
import type { DelegateResult } from "./types"
import type { MessageBus } from "./bus"
import type { TaskManager } from "../task/manager"
import type { TaskExecutionResult, CreateTaskInput } from "../task/types"

export interface DelegatorOptions {
  bus: MessageBus
  agents: Map<string, AgentRuntime>
  /** If provided, delegation goes through durable task lifecycle. */
  taskManager?: TaskManager
}

export class Delegator {
  private bus: MessageBus
  private agents: Map<string, AgentRuntime>
  private taskManager?: TaskManager

  /** Legacy counter for backward compatibility when taskManager is not set. */
  private _legacyTaskId = 0

  constructor(options: DelegatorOptions) {
    this.bus = options.bus
    this.agents = options.agents
    this.taskManager = options.taskManager
  }

  async delegate(
    params: {
      from: string
      to: string
      input: string
      metadata?: Record<string, unknown>
      runOptions?: RunOptions
      successCheck?: () => boolean
    },
  ): Promise<DelegateResult> {
    // ── New path: through TaskManager ──
    if (this.taskManager) {
      return this.delegateViaTaskManager(params)
    }

    // ── Legacy path: direct agent.run() ──
    return this.delegateLegacy(params)
  }

  private buildSubagentCall(params: {
    to: string
    input: string
    metadata?: Record<string, unknown>
    runOptions?: RunOptions
  }): SubagentCall {
    return {
      workerId: params.to,
      taskType: params.metadata?.taskType as string | undefined,
      objective: params.input,
      rawTask: params.input,
      toolBudget: {
        maxSteps: params.runOptions?.maxIterations,
        maxRetries: params.runOptions?.maxDelegationRetries,
      },
    }
  }

  private buildWorkerResult(params: {
    success: boolean
    output: string
    error?: string
    runResult?: import("../runtime/types").RunResult
  }): WorkerResult {
    return {
      status: params.success ? "completed" : "failed",
      failureClass: params.error ?? null,
      telemetry: {
        toolsUsed: params.runResult?.toolCalls.map((tc) => tc.tool) ?? [],
        steps: params.runResult?.toolCalls.length ?? 0,
        iterations: params.runResult?.iterations,
      },
      rawOutput: params.output,
    }
  }

  async delegateMany(
    params: {
      from: string
      tasks: Array<{ to: string; input: string; metadata?: Record<string, unknown> }>
      runOptions?: RunOptions
    },
  ): Promise<DelegateResult[]> {
    return Promise.all(
      params.tasks.map((t) =>
        this.delegate({
          from: params.from,
          to: t.to,
          input: t.input,
          metadata: t.metadata,
          runOptions: params.runOptions,
        }),
      ),
    )
  }

  /* ================================================================ */
  /*  New path: TaskManager-based delegation                           */
  /* ================================================================ */

  private async delegateViaTaskManager(params: {
    from: string
    to: string
    input: string
    metadata?: Record<string, unknown>
    runOptions?: RunOptions
    successCheck?: () => boolean
  }): Promise<DelegateResult> {
    const tm = this.taskManager!
    const agents = this.agents

    const createInput: CreateTaskInput = {
      kind: "delegate",
      taskType: (params.metadata?.taskType as string) ?? "unknown",
      fromAgent: params.from,
      toAgent: params.to,
      input: params.input,
      metadata: params.metadata,
    }
    const subagentCall = this.buildSubagentCall(params)

    const { task, delegateResult } = await tm.executeTask(
      createInput,
      async (taskRecord) => {
        const agentId = taskRecord.toAgent!
        const agent = agents.get(agentId)
        if (!agent) {
          return {
            success: false,
            output: "",
            error: `Agent "${agentId}" not found`,
          }
        }

        const runResult = await agent.run(params.input, params.runOptions)
        const success = params.successCheck ? params.successCheck() : true

        const execResult: TaskExecutionResult = {
          success,
          output: runResult.response,
          runResult,
          toolCalls: runResult.toolCalls,
          resultSummary: runResult.response.slice(0, 200),
          workerResult: this.buildWorkerResult({
            success,
            output: runResult.response,
            runResult,
          }),
        }

        return execResult
      },
    )

    // Notify bus (for backward compatibility / observability)
    await this.bus.send({
      from: delegateResult.from,
      to: delegateResult.to,
      type: "delegate_result",
      content: delegateResult.output,
      metadata: {
        taskId: task.taskId,
        success: delegateResult.success,
        attempts: task.attempts,
        subagentCall,
      },
    })

    return {
      ...delegateResult,
      workerResult: delegateResult.workerResult ?? this.buildWorkerResult({
        success: delegateResult.success,
        output: delegateResult.output,
        error: delegateResult.error,
        runResult: delegateResult.runResult,
      }),
    }
  }

  /* ================================================================ */
  /*  Legacy path: direct agent.run() (unchanged from original)        */
  /* ================================================================ */

  private async delegateLegacy(params: {
    from: string
    to: string
    input: string
    metadata?: Record<string, unknown>
    runOptions?: RunOptions
    successCheck?: () => boolean
  }): Promise<DelegateResult> {
    const taskId = `task_${Date.now()}_${++this._legacyTaskId}`
    const subagentCall = this.buildSubagentCall(params)

    await this.bus.send({
      from: params.from,
      to: params.to,
      type: "delegate",
      content: params.input,
      metadata: { taskId, subagentCall, ...params.metadata },
    })

    const agent = this.agents.get(params.to)
    if (!agent) {
      const result: DelegateResult = {
        taskId,
        from: params.to,
        to: params.from,
        success: false,
        output: "",
        error: `Agent "${params.to}" not found`,
        workerResult: this.buildWorkerResult({
          success: false,
          output: "",
          error: `Agent "${params.to}" not found`,
        }),
      }
      await this.bus.send({
        from: params.to,
        to: params.from,
        type: "delegate_result",
        content: JSON.stringify(result),
        metadata: { taskId },
      })
      return result
    }

    try {
      const runResult = await agent.run(params.input, params.runOptions)
      const success = params.successCheck ? params.successCheck() : true

      const result: DelegateResult = {
        taskId,
        from: params.to,
        to: params.from,
        success,
        output: runResult.response,
        toolCalls: runResult.toolCalls.map((tc) => ({
          tool: tc.tool,
          args: tc.args,
          result: tc.result,
        })),
        runResult,
        workerResult: this.buildWorkerResult({
          success,
          output: runResult.response,
          runResult,
        }),
      }

      await this.bus.send({
        from: params.to,
        to: params.from,
        type: "delegate_result",
        content: runResult.response,
        metadata: { taskId, iterations: runResult.iterations, usage: runResult.usage },
      })

      return result
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      const result: DelegateResult = {
        taskId,
        from: params.to,
        to: params.from,
        success: false,
        output: "",
        error: msg,
        workerResult: this.buildWorkerResult({
          success: false,
          output: "",
          error: msg,
        }),
      }

      await this.bus.send({
        from: params.to,
        to: params.from,
        type: "delegate_result",
        content: `Error: ${msg}`,
        metadata: { taskId },
      })

      return result
    }
  }
}
