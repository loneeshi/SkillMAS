/**
 * Lightweight delegation primitive used by the core SkillMAS release.
 *
 * Production runtimes can wrap this interface with durable task
 * journals or environment-specific routing without changing the core contract.
 */

import type { RunOptions, SubagentCall, WorkerResult, RunResult } from "../runtime/types.js"
import type { DelegateResult } from "./types.js"
import type { MessageBus } from "./bus.js"

export interface AgentRuntimePort {
  run(input: string, options?: RunOptions): Promise<RunResult>
}

export interface DelegatorOptions {
  bus: MessageBus
  agents: Map<string, AgentRuntimePort>
}

export class Delegator {
  private bus: MessageBus
  private agents: Map<string, AgentRuntimePort>
  private taskCounter = 0

  constructor(options: DelegatorOptions) {
    this.bus = options.bus
    this.agents = options.agents
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
    const taskId = `task_${Date.now()}_${++this.taskCounter}`
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
      return this.failDelegation(taskId, params.to, params.from, `Agent "${params.to}" not found`)
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
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      return this.failDelegation(taskId, params.to, params.from, message)
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
      params.tasks.map((task) =>
        this.delegate({
          from: params.from,
          to: task.to,
          input: task.input,
          metadata: task.metadata,
          runOptions: params.runOptions,
        }),
      ),
    )
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
    runResult?: RunResult
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

  private async failDelegation(
    taskId: string,
    from: string,
    to: string,
    message: string,
  ): Promise<DelegateResult> {
    const result: DelegateResult = {
      taskId,
      from,
      to,
      success: false,
      output: "",
      error: message,
      workerResult: this.buildWorkerResult({
        success: false,
        output: "",
        error: message,
      }),
    }

    await this.bus.send({
      from,
      to,
      type: "delegate_result",
      content: `Error: ${message}`,
      metadata: { taskId },
    })

    return result
  }
}
